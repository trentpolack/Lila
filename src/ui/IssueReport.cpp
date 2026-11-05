#include "ui/IssueReport.h"

#include "core/CrashHandler.h"
#include "core/Paths.h"
#include "core/PrefStore.h"
#include "core/Version.h"
#include "render/Renderer.h"
#include "ui/FileDialog.h"
#include "ui/PerfTrace.h"
#include "ui/SupportBundle.h"
#include "ui/ViewportToast.h"

#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QCloseEvent>
#include <QKeyEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFile>
#include <QFileInfo>
#include <QFormLayout>
#include <QHeaderView>
#include <QHttpMultiPart>
#include <QLabel>
#include <QLineEdit>
#include <QMainWindow>
#include <QMessageBox>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QPlainTextEdit>
#include <QPointer>
#include <QPushButton>
#include <QRegularExpression>
#include <QStatusBar>
#include <QSysInfo>
#include <QTextBrowser>
#include <QTimer>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>

namespace lila::ui::issueReport
{
namespace
{
namespace fs = std::filesystem;

/** The telemetry Worker (services/telemetry-worker/src/issues.js). LILA_ISSUES_URL overrides it. */
constexpr const char* kWorker = "https://lila-telemetry.lila3d.workers.dev";
constexpr int kStatusBatch = 10;
constexpr int64_t kCheckIntervalSecs = 6 * 3600;
constexpr int64_t kForgetClosedSecs = 30 * 86400;
constexpr int kStartupDelayMs = 20 * 1000;
/** The Worker's per-attachment limits; together they stay under Cloudflare's 100 MB request cap. */
constexpr qint64 kMaxFileBytes = 25ll * 1024 * 1024;
constexpr qint64 kMaxBundleBytes = 25ll * 1024 * 1024;
/** For files too big to attach. */
constexpr const char* kSupportEmail = "lila3d.app@gmail.com";

QString worker()
{
    if (const char* v = std::getenv("LILA_ISSUES_URL"); v && *v)
        return QString::fromUtf8(v);
    return kWorker;
}

QNetworkAccessManager* network()
{
    static QPointer<QNetworkAccessManager> nam;
    if (!nam)
        nam = new QNetworkAccessManager(qApp);
    return nam;
}

/** Reports sent from this machine, with the token that reads their status. */
fs::path storeFile()
{
    return PrefStore::get().userDir() / "reports" / "reports.json";
}

json loadStore()
{
    std::ifstream in(storeFile());
    json j = in ? json::parse(in, nullptr, false) : json();
    if (!j.is_object())
        j = json::object();
    if (!j.contains("reports") || !j["reports"].is_array())
        j["reports"] = json::array();
    return j;
}

void saveStore(const json& j)
{
    std::error_code ec;
    fs::create_directories(storeFile().parent_path(), ec);
    std::ofstream(storeFile()) << j.dump(2);
}

std::vector<std::string> labelsOf(const json& r)
{
    std::vector<std::string> out;
    if (r.contains("labels") && r["labels"].is_array())
        for (const json& l : r["labels"])
            if (l.is_string())
                out.push_back(l.get<std::string>());
    return out;
}

int replyCount(const json& r)
{
    return r.contains("replies") && r["replies"].is_array() ? int(r["replies"].size()) : 0;
}

std::string errorText(QNetworkReply* reply)
{
    json j = json::parse(reply->readAll().toStdString(), nullptr, false);
    if (j.is_object() && j.contains("error") && j["error"].is_string())
        return j["error"].get<std::string>();
    if (reply->error() != QNetworkReply::NoError)
        return "Can't reach the report service: " + reply->errorString().toStdString();
    return "The report service answered " + std::to_string(reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt()) + ".";
}

/** Folds the Worker's status answers into the store; returns the numbers that gained replies. */
std::vector<int> mergeStatus(json& store, const json& answers)
{
    std::vector<int> fresh;
    for (const json& a : answers)
    {
        if (!a.is_object() || a.contains("error") || !a.contains("number"))
            continue;
        for (json& r : store["reports"])
        {
            if (r.value("number", 0) != a["number"].get<int>())
                continue;
            int seen = r.value("seenReplies", 0);
            for (const char* key : {"state", "labels", "replies", "updated", "title"})
                if (a.contains(key))
                    r[key] = a[key];
            if (r.value("state", "open") == "closed" && !r.contains("closedAt"))
                r["closedAt"] = QDateTime::currentSecsSinceEpoch();
            else if (r.value("state", "open") != "closed")
                r.erase("closedAt");
            if (replyCount(r) > seen)
                fresh.push_back(r["number"].get<int>());
        }
    }
    return fresh;
}

/** Reports worth asking about: open ones, and closed ones for a month after they closed. */
json pendingQueries(const json& store)
{
    json out = json::array();
    const int64_t now = QDateTime::currentSecsSinceEpoch();
    for (const json& r : store["reports"])
    {
        if (r.value("state", "open") == "closed" && now - r.value("closedAt", now) > kForgetClosedSecs)
            continue;
        out.push_back({{"number", r.value("number", 0)}, {"token", r.value("token", "")}});
    }
    return out;
}

/** Asks the Worker about the queries in batches, then calls done with every answer. */
void fetchStatus(json queries, std::function<void(json answers, std::string error)> done)
{
    struct Run
    {
        json queries;
        json answers = json::array();
        size_t next = 0;
        std::function<void(json, std::string)> done;
    };
    auto run = std::make_shared<Run>(Run{std::move(queries), json::array(), 0, std::move(done)});
    auto step = std::make_shared<std::function<void()>>();
    *step = [run, step] {
        if (run->next >= run->queries.size())
        {
            run->done(run->answers, {});
            *step = nullptr;
            return;
        }
        json batch = json::array();
        for (; run->next < run->queries.size() && batch.size() < kStatusBatch; ++run->next)
            batch.push_back(run->queries[run->next]);
        QNetworkRequest req{QUrl(worker() + "/issues/status")};
        req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
        req.setTransferTimeout(15000);
        QNetworkReply* reply = network()->post(req, QByteArray::fromStdString(json{{"reports", batch}}.dump()));
        QObject::connect(reply, &QNetworkReply::finished, qApp, [run, step, reply] {
            reply->deleteLater();
            const int code = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            if (reply->error() != QNetworkReply::NoError || code != 200)
            {
                run->done(run->answers, errorText(reply));
                *step = nullptr;
                return;
            }
            json j = json::parse(reply->readAll().toStdString(), nullptr, false);
            if (j.is_object() && j.contains("reports") && j["reports"].is_array())
                for (json& a : j["reports"])
                    run->answers.push_back(std::move(a));
            (*step)();
        });
    };
    (*step)();
}

void refreshAll(std::function<void(std::vector<int> fresh, std::string error)> done)
{
    json queries = pendingQueries(loadStore());
    if (queries.empty())
    {
        done({}, {});
        return;
    }
    fetchStatus(std::move(queries), [done](json answers, std::string error) {
        json store = loadStore();
        std::vector<int> fresh = mergeStatus(store, answers);
        store["lastCheck"] = QDateTime::currentSecsSinceEpoch();
        saveStore(store);
        done(std::move(fresh), std::move(error));
    });
}

QByteArray screenshotPng(QMainWindow* window)
{
    QByteArray png;
    QBuffer buf(&png);
    buf.open(QIODevice::WriteOnly);
    window->grab().save(&buf, "PNG");
    return png;
}

void addPart(QHttpMultiPart* multi, const QString& name, const QString& file, const QByteArray& type, const QByteArray& data)
{
    QHttpPart part;
    QString disposition = "form-data; name=\"" + name + "\"";
    if (!file.isEmpty())
        disposition += "; filename=\"" + file + "\"";
    part.setHeader(QNetworkRequest::ContentDispositionHeader, disposition);
    part.setHeader(QNetworkRequest::ContentTypeHeader, type);
    part.setBody(data);
    multi->append(part);
}

QByteArray readBytes(const fs::path& file)
{
    QFile f(QString::fromStdWString(file.wstring()));
    return f.open(QIODevice::ReadOnly) ? f.readAll() : QByteArray();
}

void remember(const json& sent, const std::string& title, const std::string& kind)
{
    json store = loadStore();
    store["reports"].push_back({{"number", sent.value("number", 0)},
                                {"url", sent.value("url", "")},
                                {"token", sent.value("token", "")},
                                {"title", title},
                                {"kind", kind},
                                {"state", "open"},
                                {"labels", json::array({kind, "Pending"})},
                                {"sent", QDateTime::currentSecsSinceEpoch()},
                                {"seenReplies", 0}});
    saveStore(store);
}
} // namespace

std::string statusText(const std::string& state, const std::vector<std::string>& labels)
{
    for (const char* triage : {"Solved", "Dismissed", "On Hold", "Pending"})
        if (std::find(labels.begin(), labels.end(), triage) != labels.end())
            return triage;
    return state == "closed" ? "Closed" : "Open";
}

namespace
{
/** The report being written, kept across restarts (Restart as Administrator for a trace) until it is sent or cleared. */
fs::path draftFile()
{
    return storeFile().parent_path() / "draft.json";
}

/** A draft saved while its dialog was open reopens the dialog at startup, if it is this fresh. */
constexpr int64_t kReopenSecs = 60 * 60;

/** One question on the form. The same key in two kinds shares its answer, so switching kind keeps what was typed. */
struct Ask
{
    const char* key;
    const char* label;
    const char* hint;
    bool required;
};

struct Kind
{
    const char* id;    // the issue label
    const char* name;  // shown in the Type list
    std::vector<Ask> asks;
    bool bundle;
    bool screenshot;
};

/** VIDEO_HOSTS in the Worker (services/telemetry-worker/src/issues.js), which refuses any other. */
constexpr const char* kVideoHosts[] = {"youtube.com", "youtu.be", "vimeo.com", "x.com", "twitter.com", "streamable.com", "loom.com", "medal.tv"};

bool videoLink(const QString& text)
{
    const QUrl url(text, QUrl::StrictMode);
    if (text.contains(' ') || !url.isValid() || (url.scheme() != "https" && url.scheme() != "http"))
        return false;
    QString host = url.host().toLower();
    if (host.startsWith("www."))
        host = host.mid(4);
    for (const char* h : kVideoHosts)
        if (host == h || host.endsWith(QString(".") + h))
            return true;
    return false;
}

constexpr const char* kStepsHint ="If you know them:\n1. Open a new scene\n2. Create a cube\n3. Bevel it with 3 segments";

const std::vector<Kind>& kinds()
{
    static const std::vector<Kind> all = {
        {"Bug", "Bug",
         {{"doing", "What were you doing?", "", true},
          {"happened", "What happened?", "", true},
          {"expected", "What did you expect?", "", true},
          {"steps", "Steps to reproduce", kStepsHint, false}},
         true, true},
        {"Crash", "Crash",
         {{"doing", "What were you doing when it crashed?", "", true}, {"crash", "Crash", "", true}, {"steps", "Steps to reproduce", kStepsHint, false}},
         true, false},
        {"Performance", "Performance",
         {{"slow", "What is slow, and how slow?", "Orbiting drops to 10 fps", true},
          {"scene", "Scene size", "200 objects, 3M triangles", false},
          {"steps", "Steps to reproduce", kStepsHint, false},
          {"trace", "Performance trace", "", true}},
         true, false},
        {"Request", "Feature Request",
         {{"want", "What do you want to do?", "", true},
          {"today", "How do you do it today?", "Or what gets in the way.", false},
          {"video", "Video link", "A video of it in another app, if you have one", false}},
         false, false},
        {"Question", "Question", {{"question", "Your question", "", true}}, false, false},
    };
    return all;
}

const Kind& kindById(const QString& id)
{
    for (const Kind& k : kinds())
        if (id == k.id)
            return k;
    return kinds().front();
}

/** Every answer field, in form order; each kind shows its own subset. */
constexpr const char* kTextKeys[] = {"doing", "happened", "expected", "slow", "scene", "want", "today", "video", "question", "steps"};

bool singleLine(std::string_view key)
{
    return key == "scene" || key == "video";
}

/** The crash's one-line summary without its "Lila crashed: " prefix. */
QString crashSummary(const fs::path& dir)
{
    std::string summary = crash::reportSummary(dir);
    const std::string prefix = "Lila crashed: ";
    if (summary.starts_with(prefix))
        summary = summary.substr(prefix.size());
    // Source locations keep file:line only; the folders are the build machine's and the issue is public.
    static const QRegularExpression folders(R"(\(([^()]*[\\/])([^\\/()]+)\))");
    return QString::fromStdString(summary).replace(folders, "(\\2)");
}

/** The summary up to the source location, short enough for a title. */
QString crashHeadline(const fs::path& dir)
{
    return crashSummary(dir).section(", at ", 0, 0).left(120);
}

QString crashWhen(const fs::path& dir)
{
    std::error_code ec;
    const auto t = fs::last_write_time(dir / "report.txt", ec);
    if (ec)
        return QString::fromStdString(paths::toUtf8(dir.filename()));
    const auto secs = paths::unixSeconds(t);
    return QDateTime::fromSecsSinceEpoch(secs).toString("d MMM yyyy, h:mm ap");
}

void sentMessage(QWidget* parent, const json& sent)
{
    QMessageBox box(parent);
    box.setWindowTitle("Report Issue");
    QString text = QString("Sent as #%1. Replies show in Help > My Reports.").arg(sent.value("number", 0));
    if (sent.value("dropped", false))
        text += "\n\nThe attachments weren't kept: today's upload limit was reached.";
    box.setText(text);
    QPushButton* open = box.addButton("Open Issue", QMessageBox::ActionRole);
    box.addButton(QMessageBox::Close);
    box.exec();
    if (box.clickedButton() == open)
        QDesktopServices::openUrl(QUrl(QString::fromStdString(sent.value("url", ""))));
}

/** A few lines tall by default; it scrolls past that. */
class AnswerEdit : public QPlainTextEdit
{
public:
    using QPlainTextEdit::QPlainTextEdit;
    QSize sizeHint() const override { return {320, 72}; }
};

/** Modeless, so the viewport stays usable while a performance trace records. */
class ReportDialog : public QDialog
{
public:
    explicit ReportDialog(QMainWindow* window) : QDialog(window), mWindow(window)
    {
        setWindowTitle("Report Issue");
        setAttribute(Qt::WA_DeleteOnClose);
        auto* layout = new QVBoxLayout(this);
        mForm = new QFormLayout;
        mKind = new QComboBox(this);
        for (const Kind& k : kinds())
            mKind->addItem(k.name, QString(k.id));
        mTitle = new QLineEdit(this);
        mTitle->setPlaceholderText("Short summary");
        mTitle->setMaxLength(120);
        mForm->addRow("Type", mKind);
        mForm->addRow("Title", mTitle);
        for (const char* key : kTextKeys)
        {
            QWidget* w = nullptr;
            if (singleLine(key))
            {
                auto* e = new QLineEdit(this);
                connect(e, &QLineEdit::textChanged, this, [this] { changed(); });
                w = e;
            }
            else
            {
                auto* e = new AnswerEdit(this);
                e->setMinimumHeight(56);
                e->setTabChangesFocus(true);
                connect(e, &QPlainTextEdit::textChanged, this, [this] { changed(); });
                w = e;
            }
            mAnswers[key] = w;
            mForm->addRow(key, w);
            if (std::string_view(key) == "doing")
            {
                mCrash = new QComboBox(this);
                mCrash->setToolTip("Its report and minidump go in the support bundle.");
                mCrash->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
                mCrash->setMinimumContentsLength(40);
                for (const fs::path& dir : crash::reports())
                {
                    mCrash->addItem(crashWhen(dir) + "  " + crashHeadline(dir), QString::fromStdWString(dir.wstring()));
                    mCrash->setItemData(mCrash->count() - 1, crashSummary(dir), Qt::ToolTipRole);
                }
                if (mCrash->count() == 0)
                    mCrash->addItem("No crash reports", QString());
                mForm->addRow("Crash", mCrash);
            }
        }
        mPerf = buildTraceRow();
        mForm->addRow("Performance trace", mPerf);
        mGithub = new QLineEdit(this);
        mGithub->setPlaceholderText("Optional");
        mGithub->setToolTip("Mentioned on the issue so GitHub notifies you. Replies also show in Help > My Reports.");
        mForm->addRow("GitHub user", mGithub);
        layout->addLayout(mForm);

        mBundle = new QCheckBox("Attach support bundle", this);
        mBundle->setToolTip("Lila's version, your system and GPU, this session's log, recent crash reports and your prefs (tokens and passwords removed).");
        mScene = new QCheckBox("Include the open scene", this);
        mSceneAllowed = PrefStore::get().value<bool>("general", support::kAllowScenePref, true);
        if (!mSceneAllowed)
            mScene->setToolTip("Turned off by your studio");
        mShot = new QCheckBox("Attach screenshot of the Lila window", this);
        layout->addWidget(mBundle);
        auto* sceneRow = new QHBoxLayout;
        sceneRow->addSpacing(20);
        sceneRow->addWidget(mScene);
        layout->addLayout(sceneRow);
        layout->addWidget(mShot);
        auto* fileRow = new QHBoxLayout;
        auto* pick = new QPushButton("Attach File...", this);
        pick->setToolTip("One more file (up to 25 MB)");
        mFileName = new QLabel(this);
        mFileName->setStyleSheet("color: #9a9a9a;");
        mUnpick = new QPushButton("Remove", this);
        fileRow->addWidget(pick);
        fileRow->addWidget(mFileName, 1);
        fileRow->addWidget(mUnpick);
        layout->addLayout(fileRow);

        auto* note = new QLabel("The title and answers are public on "
                                "<a href=\"https://github.com/lila-3d/issues\">github.com/lila-3d/issues</a>. Attachments stay private.",
                                this);
        note->setOpenExternalLinks(true);
        note->setWordWrap(true);
        note->setStyleSheet("color: #9a9a9a;");
        layout->addWidget(note);
        mError = new QLabel(this);
        mError->setWordWrap(true);
        mError->setStyleSheet("color: #e0a050;");
        mError->hide();
        layout->addWidget(mError);

        auto* buttons = new QDialogButtonBox(this);
        mSend = buttons->addButton("Send", QDialogButtonBox::AcceptRole);
        mSend->setDefault(true);
        QPushButton* clearButton = buttons->addButton("Clear", QDialogButtonBox::ResetRole);
        clearButton->setToolTip("Empties the form and forgets the draft.");
        buttons->addButton(QDialogButtonBox::Close);
        connect(buttons, &QDialogButtonBox::rejected, this, [this] {
            closedByUser();
            close();
        });
        layout->addWidget(buttons);

        connect(pick, &QPushButton::clicked, this, [this] {
            const QString f = fileDialog::getOpenFileName(this, "Attach File");
            if (!f.isEmpty() && checkSize(f))
                setFile(f);
        });
        connect(mUnpick, &QPushButton::clicked, this, [this] { setFile({}); });
        connect(mSend, &QPushButton::clicked, this, [this] { send(); });
        connect(clearButton, &QPushButton::clicked, this, [this] { clear(); });
        connect(mKind, &QComboBox::currentIndexChanged, this, [this] { syncKind(!mLoading); });
        connect(mBundle, &QCheckBox::toggled, this, [this] { mScene->setEnabled(mSceneAllowed && mBundle->isChecked()); });
        for (QLineEdit* e : {mTitle, mGithub})
            connect(e, &QLineEdit::textChanged, this, [this] { changed(); });
        for (QCheckBox* c : {mBundle, mScene, mShot})
            connect(c, &QCheckBox::toggled, this, [this] { changed(); });
        connect(mCrash, &QComboBox::currentIndexChanged, this, [this] { changed(); });

        mSaveTimer.setSingleShot(true);
        mSaveTimer.setInterval(400);
        connect(&mSaveTimer, &QTimer::timeout, this, [this] { saveDraft(true); });
        mTracePoll.setInterval(1000);
        connect(&mTracePoll, &QTimer::timeout, this, [this] { pollTrace(); });
        // A trace saved before the dialog opened is for Choose Existing, not attached by itself.
        mSeenTrace = QString::fromStdString(perfTrace::status().value("path", ""));

        mLoading = true;
        setFile({});
        setTrace({});
        mLoading = false;
        if (!loadDraft())
            applyKindDefaults();
        syncKind(false);
        mScene->setEnabled(mSceneAllowed && mBundle->isChecked());
        saveDraft(true);
        resize(600, sizeHint().height());
    }

    /** Lila closing with the dialog open (Restart as Administrator) keeps it open in the draft, so it comes back. */
    ~ReportDialog() override
    {
        if (!mClosedByUser)
            saveDraft(true);
    }

    /** Switches to a Crash report about `dir`. */
    void reportCrash(const fs::path& dir)
    {
        mKind->setCurrentIndex(mKind->findData(QString("Crash")));
        const int i = mCrash->findData(QString::fromStdWString(dir.wstring()));
        if (i >= 0)
            mCrash->setCurrentIndex(i);
        if (mTitle->text().trimmed().isEmpty())
        {
            mTitle->setText(crashHeadline(dir));
            mTitle->setCursorPosition(0);
        }
    }

protected:
    // Lila closing closes the dialog too, but not spontaneously: only the title bar, Esc and Close count as the user's.
    void closeEvent(QCloseEvent* e) override
    {
        if (e->spontaneous())
            closedByUser();
        QDialog::closeEvent(e);
    }
    void keyPressEvent(QKeyEvent* e) override
    {
        if (e->key() == Qt::Key_Escape)
            closedByUser();
        QDialog::keyPressEvent(e);
    }

private:
    void closedByUser()
    {
        saveDraft(false);
        mClosedByUser = true;
    }

    const Kind& kind() const { return kindById(mKind->currentData().toString()); }

    QString answer(const char* key) const
    {
        QWidget* w = mAnswers.at(key);
        if (auto* e = qobject_cast<QLineEdit*>(w))
            return e->text();
        return static_cast<QPlainTextEdit*>(w)->toPlainText();
    }

    void setAnswer(const char* key, const QString& text)
    {
        QWidget* w = mAnswers.at(key);
        if (auto* e = qobject_cast<QLineEdit*>(w))
            e->setText(text);
        else
            static_cast<QPlainTextEdit*>(w)->setPlainText(text);
    }

    QWidget* buildTraceRow()
    {
        auto* box = new QWidget(this);
        auto* row = new QHBoxLayout(box);
        row->setContentsMargins(0, 0, 0, 0);
        auto* record = new QPushButton("Record Trace...", box);
        record->setToolTip("Opens the recorder: press Record, then do the slow thing until it stops. The trace attaches here by itself.");
        auto* existing = new QPushButton("Choose Existing...", box);
        mTraceName = new QLabel(box);
        mTraceName->setStyleSheet("color: #9a9a9a;");
        mUntrace = new QPushButton("Remove", box);
        row->addWidget(record);
        row->addWidget(existing);
        row->addWidget(mTraceName, 1);
        row->addWidget(mUntrace);
        connect(record, &QPushButton::clicked, this, [this] { perfTrace::showDialog(mWindow); });
        connect(existing, &QPushButton::clicked, this, [this] {
            const QString f = fileDialog::getOpenFileName(this, "Choose Performance Trace", QString::fromStdWString(perfTrace::folder().wstring()),
                                                          "Performance traces (*.zip)");
            if (!f.isEmpty() && checkSize(f))
                setTrace(f);
        });
        connect(mUntrace, &QPushButton::clicked, this, [this] { setTrace({}); });
        return box;
    }

    bool checkSize(const QString& f)
    {
        if (QFileInfo(f).size() <= kMaxFileBytes)
            return true;
        showError("That file is over 25 MB. Attach a smaller one, or email it to " + QString(kSupportEmail) + ".");
        return false;
    }

    void showError(const QString& text)
    {
        mError->setText(text);
        mError->setVisible(!text.isEmpty());
    }

    void setFile(const QString& f)
    {
        mFile = f;
        mFileName->setText(f.isEmpty() ? QString() : QFileInfo(f).fileName());
        mUnpick->setVisible(!f.isEmpty());
        changed();
    }

    void setTrace(const QString& f)
    {
        mTrace = f;
        mTraceName->setText(f.isEmpty() ? QString("Needed for a performance report") : QFileInfo(f).fileName());
        mUntrace->setVisible(!f.isEmpty());
        changed();
    }

    void pollTrace()
    {
        const json st = perfTrace::status();
        const QString path = QString::fromStdString(st.value("path", ""));
        if (st.value("state", "") != "done" || path.isEmpty() || path == mSeenTrace)
            return;
        mSeenTrace = path;
        if (checkSize(path))
            setTrace(path);
        raise();
    }

    void applyKindDefaults()
    {
        mBundle->setChecked(kind().bundle);
        mShot->setChecked(kind().screenshot);
    }

    /** Shows the kind's questions; a kind the user picked also gets its attachment defaults. */
    void syncKind(bool userPicked)
    {
        const Kind& k = kind();
        auto shown = [&](std::string_view key) -> const Ask* {
            for (const Ask& a : k.asks)
                if (key == a.key)
                    return &a;
            return nullptr;
        };
        auto setRow = [&](QWidget* field, const char* key) {
            const Ask* a = shown(key);
            mForm->setRowVisible(field, a != nullptr);
            if (!a)
                return;
            if (auto* label = qobject_cast<QLabel*>(mForm->labelForField(field)))
                label->setText(a->required ? QString(a->label) : QString(a->label) + " (optional)");
            if (auto* line = qobject_cast<QLineEdit*>(field))
                line->setPlaceholderText(a->hint);
            else if (auto* text = qobject_cast<QPlainTextEdit*>(field))
                text->setPlaceholderText(a->hint);
        };
        for (const char* key : kTextKeys)
            setRow(mAnswers[key], key);
        setRow(mCrash, "crash");
        setRow(mPerf, "trace");
        if (shown("trace"))
            mTracePoll.start();
        else
            mTracePoll.stop();
        if (userPicked)
            applyKindDefaults();
        changed();
        if (isVisible())
            resize(width(), sizeHint().height());
    }

    QStringList missing() const
    {
        QStringList out;
        if (mTitle->text().trimmed().isEmpty())
            out << "a title";
        for (const Ask& a : kind().asks)
        {
            const std::string_view key = a.key;
            if (key == "trace")
            {
                if (mTrace.isEmpty())
                    out << "a performance trace";
                continue;
            }
            if (key == "crash")
                continue;
            const QString text = answer(a.key).trimmed();
            if (a.required && text.isEmpty())
                out << QString(a.label).toLower().remove('?');
            if (key == "video" && !text.isEmpty() && !videoLink(text))
                out << "a YouTube, Vimeo, X, Streamable, Loom or Medal link for the video";
        }
        return out;
    }

    void changed()
    {
        if (mSending)
            return;
        const QStringList need = missing();
        mSend->setEnabled(need.isEmpty());
        mSend->setToolTip(need.isEmpty() ? QString() : "Needs " + need.join(", "));
        if (!mLoading)
            mSaveTimer.start();
    }

    void saveDraft(bool open)
    {
        if (mSent || mLoading)
            return;
        json answers = json::object();
        for (const char* key : kTextKeys)
            answers[key] = answer(key).toStdString();
        const json d = {{"title", mTitle->text().toStdString()},
                        {"kind", mKind->currentData().toString().toStdString()},
                        {"answers", answers},
                        {"crash", mCrash->currentData().toString().toStdString()},
                        {"github", mGithub->text().toStdString()},
                        {"bundle", mBundle->isChecked()},
                        {"scene", mScene->isChecked()},
                        {"screenshot", mShot->isChecked()},
                        {"file", mFile.toStdString()},
                        {"trace", mTrace.toStdString()},
                        {"open", open},
                        {"savedAt", QDateTime::currentSecsSinceEpoch()}};
        std::error_code ec;
        fs::create_directories(draftFile().parent_path(), ec);
        std::ofstream(draftFile()) << d.dump(2);
    }

    /** False when there was no draft. */
    bool loadDraft()
    {
        std::ifstream in(draftFile());
        const json d = in ? json::parse(in, nullptr, false) : json();
        if (!d.is_object())
            return false;
        mLoading = true;
        auto str = [&](const char* k) { return QString::fromStdString(d.value(k, "")); };
        auto existing = [](const QString& f) { return !f.isEmpty() && QFileInfo::exists(f) ? f : QString(); };
        mTitle->setText(str("title"));
        if (const int i = mKind->findData(str("kind")); i >= 0)
            mKind->setCurrentIndex(i);
        const json answers = d.value("answers", json::object());
        for (const char* key : kTextKeys)
            setAnswer(key, QString::fromStdString(answers.is_object() ? answers.value(key, "") : ""));
        if (const int i = mCrash->findData(str("crash")); i >= 0)
            mCrash->setCurrentIndex(i);
        mGithub->setText(str("github"));
        mBundle->setChecked(d.value("bundle", kind().bundle));
        mScene->setChecked(d.value("scene", false));
        mShot->setChecked(d.value("screenshot", kind().screenshot));
        setFile(existing(str("file")));
        setTrace(existing(str("trace")));
        mLoading = false;
        return true;
    }

    void clear()
    {
        mLoading = true;
        mTitle->clear();
        for (const char* key : kTextKeys)
            setAnswer(key, {});
        setFile({});
        setTrace({});
        applyKindDefaults();
        mScene->setChecked(false);
        mLoading = false;
        showError({});
        changed();
    }

    void send()
    {
        showError({});
        mSending = true;
        mSend->setEnabled(false);
        mSend->setText("Sending...");

        const Kind& k = kind();
        json sections = json::array();
        fs::path crashDir;
        for (const Ask& a : k.asks)
        {
            const std::string_view key = a.key;
            if (key == "trace")
                continue;
            if (key == "crash")
            {
                crashDir = fs::path(mCrash->currentData().toString().toStdWString());
                if (!crashDir.empty())
                    sections.push_back({{"key", "crash"}, {"heading", "Crash"}, {"text", (crashWhen(crashDir) + ": " + crashSummary(crashDir)).toStdString()}});
                continue;
            }
            const QString text = answer(a.key).trimmed();
            if (!text.isEmpty())
                sections.push_back({{"key", a.key}, {"heading", a.label}, {"text", text.toStdString()}});
        }
        render::Renderer::GpuInfo gpu = render::Renderer::get().gpuInfo();
        const json meta = {{"title", mTitle->text().trimmed().toStdString()},
                           {"kind", k.id},
                           {"sections", sections},
                           {"github", mGithub->text().trimmed().toStdString()},
                           {"version", version::full()},
                           {"os", QSysInfo::prettyProductName().toStdString()},
                           {"gpu", gpu.name}};
        auto* multi = new QHttpMultiPart(QHttpMultiPart::FormDataType);
        addPart(multi, "report", {}, "application/json", QByteArray::fromStdString(meta.dump()));
        if (mBundle->isChecked())
        {
            const fs::path zip = fs::temp_directory_path() / ("lila-report-" + std::to_string(QDateTime::currentMSecsSinceEpoch()) + ".zip");
            const bool withScene = mScene->isEnabled() && mScene->isChecked();
            QByteArray bytes = support::writeBundle(zip, withScene, nullptr, crashDir) ? readBytes(zip) : QByteArray();
            std::error_code ec;
            fs::remove(zip, ec);
            if (bytes.size() > kMaxBundleBytes)
            {
                delete multi;
                mSending = false;
                mSend->setText("Send");
                showError(withScene ? "The support bundle is over 25 MB with the scene in it. Untick Include the open scene, or email the scene to " +
                                          QString(kSupportEmail) + "."
                                    : "The support bundle is over 25 MB. Email it to " + QString(kSupportEmail) + " instead.");
                changed();
                return;
            }
            if (!bytes.isEmpty())
                addPart(multi, "bundle", "bundle.zip", "application/zip", bytes);
        }
        if (mShot->isChecked())
            addPart(multi, "screenshot", "screenshot.png", "image/png", screenshotPng(mWindow));
        if (!mTrace.isEmpty() && std::any_of(k.asks.begin(), k.asks.end(), [](const Ask& a) { return std::string_view(a.key) == "trace"; }))
            addPart(multi, "trace", "trace.zip", "application/zip", readBytes(fs::path(mTrace.toStdWString())));
        if (!mFile.isEmpty())
            addPart(multi, "file", QFileInfo(mFile).fileName(), "application/octet-stream", readBytes(fs::path(mFile.toStdWString())));

        QNetworkRequest req{QUrl(worker() + "/issues/new")};
        req.setTransferTimeout(180000);
        QNetworkReply* reply = network()->post(req, multi);
        multi->setParent(reply);
        connect(this, &QObject::destroyed, reply, &QNetworkReply::abort);
        connect(reply, &QNetworkReply::finished, this, [this, reply, meta] {
            reply->deleteLater();
            mSending = false;
            mSend->setText("Send");
            const int code = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            const json j = reply->error() == QNetworkReply::NoError && code == 200 ? json::parse(reply->readAll().toStdString(), nullptr, false) : json();
            if (!j.is_object() || !j.contains("number"))
            {
                showError(QString::fromStdString(errorText(reply)));
                changed();
                return;
            }
            remember(j, meta["title"].get<std::string>(), meta["kind"].get<std::string>());
            mSent = true;
            std::error_code ec;
            fs::remove(draftFile(), ec);
            QPointer<QMainWindow> window(mWindow);
            close();
            if (window)
                sentMessage(window, j);
        });
    }

    QMainWindow* mWindow;
    QFormLayout* mForm = nullptr;
    QComboBox* mKind = nullptr;
    QLineEdit* mTitle = nullptr;
    std::map<std::string, QWidget*> mAnswers;
    QComboBox* mCrash = nullptr;
    QWidget* mPerf = nullptr;
    QLabel* mTraceName = nullptr;
    QPushButton* mUntrace = nullptr;
    QLineEdit* mGithub = nullptr;
    QCheckBox* mBundle = nullptr;
    QCheckBox* mScene = nullptr;
    QCheckBox* mShot = nullptr;
    QLabel* mFileName = nullptr;
    QPushButton* mUnpick = nullptr;
    QLabel* mError = nullptr;
    QPushButton* mSend = nullptr;
    QTimer mSaveTimer;
    QTimer mTracePoll;
    QString mFile;
    QString mTrace;
    QString mSeenTrace;
    bool mSceneAllowed = true;
    bool mLoading = false;
    bool mSending = false;
    bool mSent = false;
    bool mClosedByUser = false;
};

QPointer<ReportDialog> gReport;

ReportDialog* openReport(QMainWindow* window)
{
    // A closed dialog is only waiting to be deleted.
    if (!gReport || !gReport->isVisible())
        gReport = new ReportDialog(window);
    gReport->show();
    gReport->raise();
    gReport->activateWindow();
    return gReport;
}
} // namespace

void showDialog(QMainWindow* window)
{
    openReport(window);
}

void showCrash(QMainWindow* window, const fs::path& crashDir)
{
    QPointer<QMainWindow> win(window);
    auto open = [win, crashDir] {
        if (win)
            openReport(win)->reportCrash(crashDir);
    };
    // Asked from the modal Lila Crashed dialog: open once it closes, or the report would sit behind it unusable.
    if (auto* modal = qobject_cast<QDialog*>(QApplication::activeModalWidget()))
        QObject::connect(modal, &QDialog::finished, win, [open] { QTimer::singleShot(0, qApp, open); }, Qt::SingleShotConnection);
    else
        open();
}

/** Reopens a report whose dialog was open when Lila last closed, such as for Restart as Administrator. */
void reopenDraft(QMainWindow* window)
{
    std::ifstream in(draftFile());
    const json d = in ? json::parse(in, nullptr, false) : json();
    if (d.is_object() && d.value("open", false) && QDateTime::currentSecsSinceEpoch() - d.value("savedAt", int64_t(0)) < kReopenSecs)
        showDialog(window);
}

void showMyReports(QWidget* parent)
{
    QDialog dlg(parent);
    dlg.setWindowTitle("My Reports");
    auto* layout = new QVBoxLayout(&dlg);
    auto* list = new QTreeWidget(&dlg);
    list->setHeaderLabels({"#", "Title", "Status", "Replies"});
    list->setRootIsDecorated(false);
    list->setUniformRowHeights(true);
    list->header()->setStretchLastSection(false);
    list->header()->setSectionResizeMode(1, QHeaderView::Stretch);
    auto* replies = new QTextBrowser(&dlg);
    replies->setOpenExternalLinks(true);
    replies->setPlaceholderText("No replies");
    auto* empty = new QLabel("No reports sent from this machine. Help > Report Issue sends one.", &dlg);
    empty->setStyleSheet("color: #9a9a9a;");
    layout->addWidget(empty);
    layout->addWidget(list, 2);
    layout->addWidget(replies, 3);
    auto* error = new QLabel(&dlg);
    error->setWordWrap(true);
    error->setStyleSheet("color: #e0a050;");
    error->hide();
    layout->addWidget(error);

    auto* buttons = new QDialogButtonBox(&dlg);
    QPushButton* open = buttons->addButton("Open Issue", QDialogButtonBox::ActionRole);
    QPushButton* refresh = buttons->addButton("Refresh", QDialogButtonBox::ActionRole);
    buttons->addButton(QDialogButtonBox::Close);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dlg, &QDialog::reject);
    layout->addWidget(buttons);

    auto fill = [=] {
        const int keep = list->currentItem() ? list->currentItem()->data(0, Qt::UserRole).toInt() : 0;
        list->clear();
        json store = loadStore();
        for (auto it = store["reports"].rbegin(); it != store["reports"].rend(); ++it)
        {
            const json& r = *it;
            const int n = r.value("number", 0);
            const int count = replyCount(r);
            const int unseen = count - r.value("seenReplies", 0);
            auto* item = new QTreeWidgetItem(list, {QString::number(n), QString::fromStdString(r.value("title", "")),
                                                    QString::fromStdString(statusText(r.value("state", "open"), labelsOf(r))),
                                                    unseen > 0 ? QString("%1 new").arg(unseen) : QString::number(count)});
            item->setData(0, Qt::UserRole, n);
            item->setTextAlignment(0, Qt::AlignRight | Qt::AlignVCenter);
            if (n == keep)
                list->setCurrentItem(item);
        }
        for (int c : {0, 2, 3})
            list->resizeColumnToContents(c);
        const bool any = list->topLevelItemCount() > 0;
        empty->setVisible(!any);
        list->setVisible(any);
        replies->setVisible(any);
        open->setEnabled(list->currentItem() != nullptr);
    };

    QObject::connect(list, &QTreeWidget::currentItemChanged, &dlg, [=](QTreeWidgetItem* item) {
        open->setEnabled(item != nullptr);
        replies->clear();
        if (!item)
            return;
        const int n = item->data(0, Qt::UserRole).toInt();
        json store = loadStore();
        for (json& r : store["reports"])
        {
            if (r.value("number", 0) != n)
                continue;
            QStringList parts;
            if (r.contains("replies") && r["replies"].is_array())
                for (const json& reply : r["replies"])
                {
                    const QDateTime at = QDateTime::fromString(QString::fromStdString(reply.value("at", "")), Qt::ISODate);
                    parts << "**" + at.toLocalTime().toString("d MMM yyyy, h:mm ap") + "**\n\n" + QString::fromStdString(reply.value("body", ""));
                }
            replies->setMarkdown(parts.join("\n\n---\n\n"));
            if (r.value("seenReplies", 0) != replyCount(r))
            {
                r["seenReplies"] = replyCount(r);
                saveStore(store);
                item->setText(3, QString::number(replyCount(r)));
            }
        }
    });
    QObject::connect(open, &QPushButton::clicked, &dlg, [=] {
        if (!list->currentItem())
            return;
        const int n = list->currentItem()->data(0, Qt::UserRole).toInt();
        for (const json& r : loadStore()["reports"])
            if (r.value("number", 0) == n)
                QDesktopServices::openUrl(QUrl(QString::fromStdString(r.value("url", ""))));
    });
    QPointer<QDialog> guard(&dlg);
    auto doRefresh = [=] {
        refresh->setEnabled(false);
        refresh->setText("Refreshing...");
        refreshAll([=](std::vector<int>, std::string why) {
            if (!guard)
                return;
            refresh->setEnabled(true);
            refresh->setText("Refresh");
            error->setText(QString::fromStdString(why));
            error->setVisible(!why.empty());
            fill();
        });
    };
    QObject::connect(refresh, &QPushButton::clicked, &dlg, doRefresh);

    fill();
    if (list->topLevelItemCount() > 0)
    {
        list->setCurrentItem(list->topLevelItem(0));
        doRefresh();
    }
    else
    {
        refresh->setEnabled(false);
    }
    dlg.resize(620, 460);
    dlg.exec();
}

void startup(QMainWindow* window)
{
    QPointer<QMainWindow> win(window);
    QTimer::singleShot(0, window, [win] {
        if (win)
            reopenDraft(win);
    });
    QTimer::singleShot(kStartupDelayMs, qApp, [win] {
        json store = loadStore();
        const int64_t now = QDateTime::currentSecsSinceEpoch();
        if (store["reports"].empty() || now - store.value("lastCheck", int64_t(0)) < kCheckIntervalSecs)
            return;
        refreshAll([win](std::vector<int> fresh, std::string) {
            if (!win || fresh.empty())
                return;
            const QString which = fresh.size() == 1 ? QString("report #%1").arg(fresh.front()) : QString("%1 reports").arg(fresh.size());
            const QString text = "New reply on " + which + ": Help > My Reports";
            win->statusBar()->showMessage(text, 20000);
            viewportToast::showActive(text, 6000);
        });
    });
}
} // namespace lila::ui::issueReport
