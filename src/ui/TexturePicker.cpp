#include "ui/TexturePicker.h"

#include "ops/Command.h"
#include "core/Log.h"
#include "core/Paths.h"
#include "core/ProjectTextures.h"
#include "core/TextMatch.h"
#include "core/Workspace.h"
#include "scene/TextureFolders.h"
#include "ui/Icons.h"
#include "ui/FileDialog.h"
#include "ui/ImagePicker.h"

#include <QApplication>
#include <QCheckBox>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QPainter>
#include <QPlainTextEdit>
#include <QPointer>
#include <QScreen>
#include <QScrollBar>
#include <QThreadPool>
#include <QToolButton>
#include <QVBoxLayout>

#include <format>
#include <unordered_map>
#include <unordered_set>

namespace lila::ui::texturePicker
{
namespace
{
namespace fs = std::filesystem;
namespace pt = projectTextures;
namespace tf = scene::textureFolders;
using json = nlohmann::json;

constexpr int kIcon = 32;

enum Kind
{
    kNone,
    kBrowse,
    kHeader,
    /** A library row. */
    kFile,
    kProject,
};
constexpr int kKindRole = Qt::UserRole;
constexpr int kValueRole = Qt::UserRole + 1;
constexpr int kThumbRole = Qt::UserRole + 2;
constexpr int kFileRole = Qt::UserRole + 3;
constexpr const char* kImages = "Images (*.png *.tga *.jpg *.jpeg *.exr *.tif *.tiff *.psd *.hdr *.bmp *.dds *.webp)";

QString qs(const std::string& s)
{
    return QString::fromStdString(s);
}

json runCommand(const char* name, const json& call)
{
    try
    {
        return ops::execute(name, call, CallSource::UI);
    }
    catch (const CommandError& e)
    {
        log::error("{}", e.what());
    }
    return nullptr;
}

/** Icon-sized previews read on the thread pool, keyed by path and modification time; unreadable files stay blank. */
struct Thumbs
{
    std::unordered_map<std::string, QPixmap> done;
    std::unordered_set<std::string> pending;
    std::vector<std::function<void(const std::string&)>> waiting;

    static Thumbs& get()
    {
        static Thumbs t;
        return t;
    }

    static std::string key(const fs::path& file)
    {
        std::error_code ec;
        auto t = fs::last_write_time(file, ec);
        return std::format("{}|{}", paths::toUtf8(file), ec ? 0 : t.time_since_epoch().count());
    }

    /** The cached pixmap, or null after queueing a read; then arrived() reports the key. */
    const QPixmap* find(const std::string& k, const fs::path& file)
    {
        if (auto it = done.find(k); it != done.end())
            return &it->second;
        if (pending.insert(k).second)
            QThreadPool::globalInstance()->start([k, file] {
                QImage img = imagePicker::thumbnailImage(file, kIcon * 2);
                QMetaObject::invokeMethod(qApp, [k, img] { get().arrived(k, img); }, Qt::QueuedConnection);
            });
        return nullptr;
    }

    void arrived(const std::string& k, const QImage& img)
    {
        pending.erase(k);
        if (done.size() > 4000)
            done.clear();
        QPixmap pm(kIcon, kIcon);
        pm.fill(Qt::transparent);
        if (!img.isNull())
        {
            QImage s = img.scaled(kIcon, kIcon, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            QPainter p(&pm);
            p.drawImage((kIcon - s.width()) / 2, (kIcon - s.height()) / 2, s);
        }
        done[k] = pm;
        for (auto& fn : waiting)
            if (fn)
                fn(k);
    }
};

/** A path compared without touching the disk: canonical where it exists, lowercased. */
std::string comparable(const std::string& stored)
{
    if (stored.empty())
        return {};
    std::error_code ec;
    return textMatch::lower(paths::toUtf8(fs::weakly_canonical(paths::resolvePath(stored), ec)));
}

class Popup : public QFrame
{
public:
    Popup(QWidget* anchor, Options options, std::function<void(const std::string&)> pick)
        : QFrame(anchor->window(), Qt::Popup), mAnchor(anchor), mOptions(std::move(options)), mPick(std::move(pick))
    {
        setAttribute(Qt::WA_DeleteOnClose);
        setFrameShape(QFrame::StyledPanel);
        mCurrentFile = comparable(mOptions.current);
        if (mOptions.items)
            mItems = mOptions.items();

        auto* lay = new QVBoxLayout(this);
        lay->setContentsMargins(6, 6, 6, 6);
        lay->setSpacing(4);
        mEdit = new QLineEdit(this);
        mEdit->setPlaceholderText("Search " + mOptions.title.toLower() + (mOptions.title.endsWith('s') ? "" : "s"));
        mEdit->setClearButtonEnabled(true);
        mList = new QListWidget(this);
        mList->setIconSize({kIcon, kIcon});
        mList->setFocusPolicy(Qt::NoFocus);
        mList->setTextElideMode(Qt::ElideMiddle);
        auto* foot = new QHBoxLayout;
        mStatus = new QLabel(this);
        mStatus->setEnabled(false);
        foot->addWidget(mStatus, 1);
        if (mOptions.projectTextures)
        {
            auto* refresh = new QToolButton(this);
            refresh->setIcon(icon("refresh"));
            refresh->setToolTip("Rescan the project folder");
            refresh->setAutoRaise(true);
            auto* folders = new QToolButton(this);
            folders->setText("Texture Folders...");
            folders->setIcon(icon("filter"));
            folders->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
            folders->setToolTip("Which folders of the project are listed, for the workspace and this scene");
            folders->setAutoRaise(true);
            foot->addWidget(refresh);
            foot->addWidget(folders);
            connect(refresh, &QToolButton::clicked, this, [] { pt::Index::get().refresh(); });
            connect(folders, &QToolButton::clicked, this, [this] {
                QPointer<QWidget> parent = mAnchor ? mAnchor->window() : nullptr;
                close();
                showFoldersDialog(parent);
            });
        }
        if (!mOptions.actions.empty())
        {
            auto* more = new QToolButton(this);
            more->setIcon(icon("library-plus"));
            more->setToolTip("Add to the library, or open its folders");
            more->setAutoRaise(true);
            more->setPopupMode(QToolButton::InstantPopup);
            auto* menu = new QMenu(more);
            for (const auto& [label, run] : mOptions.actions)
                connect(menu->addAction(label), &QAction::triggered, this, [this, run] {
                    close();
                    if (run)
                        run();
                });
            more->setMenu(menu);
            foot->addWidget(more);
        }
        lay->addWidget(mEdit);
        lay->addWidget(mList, 1);
        lay->addLayout(foot);

        mEdit->installEventFilter(this);
        connect(mEdit, &QLineEdit::textChanged, this, [this] { refill(); });
        connect(mList, &QListWidget::itemClicked, this, [this](QListWidgetItem* it) { activate(it); });
        connect(mList->verticalScrollBar(), &QScrollBar::valueChanged, this, [this] { requestThumbs(); });

        if (mOptions.projectTextures)
        {
            mListener = pt::Index::get().addListener([this] { refill(); });
            pt::Index::get().follow();
        }
        mThumbSlot = Thumbs::get().waiting.size();
        Thumbs::get().waiting.push_back([this](const std::string& k) { thumbArrived(k); });
        refill();
        place();
    }

    ~Popup() override
    {
        if (mOptions.projectTextures)
            pt::Index::get().removeListener(mListener);
        Thumbs::get().waiting[mThumbSlot] = nullptr;
    }

    void start()
    {
        show();
        mEdit->setFocus();
        requestThumbs();
    }

protected:
    bool eventFilter(QObject* obj, QEvent* e) override
    {
        if (obj != mEdit || e->type() != QEvent::KeyPress)
            return false;
        auto* k = static_cast<QKeyEvent*>(e);
        switch (k->key())
        {
        case Qt::Key_Down:
        case Qt::Key_Up:
        case Qt::Key_PageDown:
        case Qt::Key_PageUp:
        {
            const int step = k->key() == Qt::Key_Down ? 1 : k->key() == Qt::Key_Up ? -1 : k->key() == Qt::Key_PageDown ? 10 : -10;
            moveCurrent(step);
            return true;
        }
        case Qt::Key_Return:
        case Qt::Key_Enter: activate(mList->currentItem()); return true;
        case Qt::Key_Escape: close(); return true;
        default: return false;
        }
    }

private:
    /** Moves the highlight by step rows, over the headings. */
    void moveCurrent(int step)
    {
        const int n = mList->count();
        if (!n)
            return;
        int row = std::clamp(mList->currentRow() + step, 0, n - 1);
        const int dir = step > 0 ? 1 : -1;
        while (row >= 0 && row < n && mList->item(row)->data(kKindRole).toInt() == kHeader)
            row += dir;
        if (row < 0 || row >= n)
            return;
        mList->setCurrentRow(row);
    }

    void place()
    {
        int w = std::max(mAnchor->width(), 380);
        resize(w, 460);
        QPoint at = mAnchor->mapToGlobal(QPoint(0, mAnchor->height()));
        if (QScreen* screen = mAnchor->screen())
        {
            QRect avail = screen->availableGeometry();
            if (at.y() + height() > avail.bottom())
                at.setY(mAnchor->mapToGlobal(QPoint(0, 0)).y() - height());
            at.setX(std::clamp(at.x(), avail.left(), std::max(avail.left(), avail.right() - width())));
        }
        move(at);
    }

    bool isCurrent(const Item& item) const
    {
        return !mOptions.current.empty() && (item.value == mOptions.current || (!mCurrentFile.empty() && comparable(item.value) == mCurrentFile));
    }

    QListWidgetItem* header(const QString& text)
    {
        auto* it = new QListWidgetItem(text, mList);
        it->setData(kKindRole, kHeader);
        it->setFlags(Qt::NoItemFlags);
        QFont f = it->font();
        f.setBold(true);
        it->setFont(f);
        return it;
    }

    QListWidgetItem* fileRow(const QString& text, const fs::path& file, Kind kind, const QString& value)
    {
        auto* it = new QListWidgetItem(text, mList);
        it->setData(kKindRole, kind);
        it->setData(kValueRole, value);
        it->setData(kFileRole, QString::fromStdU16String(file.u16string()));
        it->setToolTip(QString::fromStdU16String(file.u16string()));
        it->setSizeHint({0, kIcon + 6});
        return it;
    }

    void refill()
    {
        QString keep;
        if (QListWidgetItem* it = mList->currentItem())
            keep = it->data(kValueRole).toString();
        const std::string queryText = mEdit->text().trimmed().toStdString();
        textMatch::Query query(queryText);

        mList->clear();
        auto special = [&](Kind kind, const char* text, const char* iconName, const char* words) {
            if (!query.empty() && !query.matches({std::string_view(words)}))
                return;
            auto* it = new QListWidgetItem(icon(iconName), text, mList);
            it->setData(kKindRole, kind);
            it->setSizeHint({0, kIcon + 4});
        };
        special(kNone, "None", "ban", "none clear");
        special(kBrowse, "Browse...", "folder-open", "browse file");

        QListWidgetItem* select = nullptr;
        auto mark = [&](QListWidgetItem* it, bool current) {
            if (current)
            {
                QFont bold = it->font();
                bold.setBold(true);
                it->setFont(bold);
                if (!select && keep.isEmpty())
                    select = it;
            }
            if (!keep.isEmpty() && it->data(kValueRole).toString() == keep)
                select = it;
        };

        // The library, by group; a group's heading shows only while it has matches.
        QString group;
        bool listedCurrent = false;
        size_t libraryShown = 0;
        for (const Item& item : mItems)
        {
            const std::string name = item.name.toStdString(), groupText = item.group.toStdString();
            if (!query.matches({std::string_view(name), std::string_view(groupText)}))
                continue;
            if (item.group != group || !libraryShown)
            {
                group = item.group;
                if (!group.isEmpty())
                    header(group);
            }
            ++libraryShown;
            const bool current = isCurrent(item);
            listedCurrent |= current;
            mark(fileRow(item.name, item.file, kFile, QString::fromStdString(item.value)), current);
        }

        // The current file when the library doesn't hold it.
        if (!mOptions.current.empty() && !listedCurrent && !mOptions.projectTextures && query.empty())
        {
            const fs::path file = paths::resolvePath(mOptions.current);
            std::error_code ec;
            header("Current");
            const QString name = QString::fromStdU16String(file.filename().u16string()) + (fs::exists(file, ec) ? "" : " (missing)");
            mark(fileRow(name, file, kFile, QString::fromStdString(mOptions.current)), true);
        }

        size_t projectShown = 0;
        pt::Index::Snapshot snap;
        if (mOptions.projectTextures)
        {
            snap = pt::Index::get().snapshot();
            mRoot = snap.root;
            tf::Rules rules = tf::current();
            std::vector<std::string> files = rules.filter.apply(snap.files);
            // Matches in the file name rank above ones that only match a folder.
            std::vector<const std::string*> byName, byFolder;
            for (const std::string& f : files)
            {
                if (!mOptions.extensions.empty())
                {
                    const std::string ext = textMatch::lower(paths::toUtf8(fs::path(paths::fromUtf8(f)).extension()));
                    if (std::ranges::find(mOptions.extensions, ext) == mOptions.extensions.end())
                        continue;
                }
                if (!query.matches({f}))
                    continue;
                size_t slash = f.rfind('/');
                std::string_view name = slash == std::string::npos ? std::string_view(f) : std::string_view(f).substr(slash + 1);
                (query.empty() || query.matches({name}) ? byName : byFolder).push_back(&f);
            }
            projectShown = byName.size() + byFolder.size();
            if (projectShown && !mItems.empty())
                header("Project");
            for (const auto* list : {&byName, &byFolder})
                for (const std::string* f : *list)
                {
                    fs::path abs = snap.root / paths::fromUtf8(*f);
                    size_t slash = f->rfind('/');
                    QString name = qs(slash == std::string::npos ? *f : f->substr(slash + 1));
                    QString folder = slash == std::string::npos ? QString() : qs(f->substr(0, slash));
                    const bool current = !mCurrentFile.empty() && textMatch::lower(paths::toUtf8(abs.lexically_normal())) == mCurrentFile;
                    mark(fileRow(folder.isEmpty() ? name : name + "\n" + folder, abs, kProject, qs(paths::toUtf8(abs))), current);
                }
        }

        if (!select && !query.empty())
            for (int i = 0; i < mList->count() && !select; ++i)
                if (int kind = mList->item(i)->data(kKindRole).toInt(); kind == kFile || kind == kProject)
                    select = mList->item(i);
        if (!select && mList->count())
            select = mList->item(0);
        mList->setCurrentItem(select);
        if (select)
            mList->scrollToItem(select);

        QString status;
        if (!mItems.empty())
            status = QString("%1 of %2 %3").arg(libraryShown).arg(mItems.size()).arg(mOptions.title.toLower() + (mOptions.title.endsWith('s') ? "" : "s"));
        if (mOptions.projectTextures)
        {
            if (!status.isEmpty())
                status += ", ";
            if (snap.root.empty() && !snap.scanning)
                status += mItems.empty() ? "No project folder: open a workspace or save the scene" : "no project folder";
            else if (snap.scanning && snap.files.empty())
                status += "scanning the project...";
            else
            {
                status += QString("%1 project textures").arg(projectShown);
                if (snap.truncated)
                    status += " (folder too large, partial)";
                else if (snap.scanning)
                    status += " (rescanning)";
            }
            mStatus->setToolTip(QString::fromStdU16String(snap.root.u16string()));
        }
        if (!status.isEmpty())
            status[0] = status[0].toUpper();
        mStatus->setText(status);
        requestThumbs();
    }

    void requestThumbs()
    {
        if (!isVisible() || !mList->count())
            return;
        QRect view = mList->viewport()->rect();
        int first = mList->row(mList->itemAt(view.topLeft()));
        int last = mList->row(mList->itemAt(view.bottomLeft()));
        if (first < 0)
            first = 0;
        if (last < 0)
            last = mList->count() - 1;
        for (int i = first; i <= last && i < mList->count(); ++i)
        {
            QListWidgetItem* it = mList->item(i);
            const int kind = it->data(kKindRole).toInt();
            if ((kind != kFile && kind != kProject) || !it->data(kThumbRole).isNull())
                continue;
            fs::path file(it->data(kFileRole).toString().toStdU16String());
            std::string k = Thumbs::key(file);
            it->setData(kThumbRole, qs(k));
            if (const QPixmap* pm = Thumbs::get().find(k, file))
                it->setIcon(*pm);
        }
    }

    void thumbArrived(const std::string& k)
    {
        QString key = qs(k);
        for (int i = 0; i < mList->count(); ++i)
            if (QListWidgetItem* it = mList->item(i); it->data(kThumbRole).toString() == key)
                if (auto found = Thumbs::get().done.find(k); found != Thumbs::get().done.end())
                    it->setIcon(found->second);
    }

    void activate(QListWidgetItem* it)
    {
        if (!it)
            return;
        const int kind = it->data(kKindRole).toInt();
        if (kind == kHeader)
            return;
        auto pick = mPick;
        QPointer<QWidget> anchor = mAnchor;
        const fs::path root = mRoot;
        const QString title = mOptions.title;
        const QString filter = mOptions.browseFilter.isEmpty() ? QString(kImages) : mOptions.browseFilter;
        std::string chosen = it->data(kValueRole).toString().toStdString();
        close();
        if (kind == kBrowse)
        {
            QString start = root.empty() ? QString() : QString::fromStdU16String(root.u16string());
            QString f = ui::fileDialog::getOpenFileName(anchor ? anchor->window() : nullptr, title, start, filter + ";;All files (*)");
            if (f.isEmpty())
                return;
            chosen = paths::toUtf8(fs::path(f.toStdU16String()));
        }
        else if (kind == kNone)
            chosen.clear();
        if (pick)
            pick(chosen);
    }

    QPointer<QWidget> mAnchor;
    Options mOptions;
    std::vector<Item> mItems;
    std::function<void(const std::string&)> mPick;
    /** The current value as a file, canonical and lowercased, to compare against listed paths without touching the disk. */
    std::string mCurrentFile;
    fs::path mRoot;
    QLineEdit* mEdit = nullptr;
    QListWidget* mList = nullptr;
    QLabel* mStatus = nullptr;
    int mListener = 0;
    size_t mThumbSlot = 0;
};

QString joinLines(const json& list)
{
    QStringList out;
    if (list.is_array())
        for (const json& v : list)
            if (v.is_string())
                out << qs(v.get<std::string>());
    return out.join('\n');
}

json splitLines(const QPlainTextEdit* edit)
{
    json out = json::array();
    for (const QString& line : edit->toPlainText().split('\n', Qt::SkipEmptyParts))
        if (QString t = line.trimmed(); !t.isEmpty())
            out.push_back(t.toStdString());
    return out;
}

class FoldersDialog : public QDialog
{
public:
    explicit FoldersDialog(QWidget* parent) : QDialog(parent)
    {
        setWindowTitle("Texture Folders");
        setAttribute(Qt::WA_DeleteOnClose);
        auto* lay = new QVBoxLayout(this);

        pt::Index::get().follow();
        fs::path root = pt::projectRoot();
        auto* where = new QLabel(root.empty() ? QString("No project folder: open a workspace or save the scene.")
                                              : QString("Project folder: %1").arg(QString::fromStdU16String(root.u16string())),
                                 this);
        where->setWordWrap(true);
        lay->addWidget(where);
        auto* help = new QLabel("One glob per line, relative to the project folder. * stays within a folder name, ** spans "
                                "folders, a pattern without / matches at any depth and a folder takes everything in it. "
                                "Examples: Textures/**   **/*_old.*   Saved. An empty include list takes everything; "
                                "excludes win.",
                                this);
        help->setWordWrap(true);
        help->setEnabled(false);
        lay->addWidget(help);

        json ws = tf::workspaceRules();
        json sc = tf::sceneRules();
        auto* wsBox = new QGroupBox(Workspace::get().isOpen() ? "Workspace (.lila/settings.json)" : "User settings (no workspace open)", this);
        auto* wsForm = new QFormLayout(wsBox);
        mWsInclude = editor(joinLines(ws.value("include", json::array())));
        mWsExclude = editor(joinLines(ws.value("exclude", json::array())));
        wsForm->addRow("Include", mWsInclude);
        wsForm->addRow("Exclude", mWsExclude);
        lay->addWidget(wsBox);

        auto* scBox = new QGroupBox("This scene (overrides the workspace, list by list)", this);
        auto* scForm = new QFormLayout(scBox);
        mScOverrideInclude = new QCheckBox("Override include", scBox);
        mScOverrideExclude = new QCheckBox("Override exclude", scBox);
        mScOverrideInclude->setChecked(sc.contains("include"));
        mScOverrideExclude->setChecked(sc.contains("exclude"));
        mScInclude = editor(joinLines(sc.value("include", json::array())));
        mScExclude = editor(joinLines(sc.value("exclude", json::array())));
        scForm->addRow(mScOverrideInclude);
        scForm->addRow("Include", mScInclude);
        scForm->addRow(mScOverrideExclude);
        scForm->addRow("Exclude", mScExclude);
        lay->addWidget(scBox);

        mPreview = new QLabel(this);
        lay->addWidget(mPreview);
        auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
        lay->addWidget(buttons);
        connect(buttons, &QDialogButtonBox::accepted, this, [this] { apply(); accept(); });
        connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);
        for (QCheckBox* c : {mScOverrideInclude, mScOverrideExclude})
            connect(c, &QCheckBox::toggled, this, [this] { update(); });
        for (QPlainTextEdit* e : {mWsInclude, mWsExclude, mScInclude, mScExclude})
            connect(e, &QPlainTextEdit::textChanged, this, [this] { update(); });
        mListener = pt::Index::get().addListener([this] { update(); });
        update();
        resize(520, 560);
    }

    ~FoldersDialog() override { pt::Index::get().removeListener(mListener); }

private:
    QPlainTextEdit* editor(const QString& text)
    {
        auto* e = new QPlainTextEdit(text, this);
        e->setTabChangesFocus(true);
        e->setFixedHeight(fontMetrics().lineSpacing() * 5 + 12);
        return e;
    }

    json workspaceEdit() const { return {{"include", splitLines(mWsInclude)}, {"exclude", splitLines(mWsExclude)}}; }

    json sceneEdit() const
    {
        json s = json::object();
        if (mScOverrideInclude->isChecked())
            s["include"] = splitLines(mScInclude);
        if (mScOverrideExclude->isChecked())
            s["exclude"] = splitLines(mScExclude);
        return s;
    }

    void update()
    {
        mScInclude->setEnabled(mScOverrideInclude->isChecked());
        mScExclude->setEnabled(mScOverrideExclude->isChecked());
        pt::Index::Snapshot snap = pt::Index::get().snapshot();
        tf::Rules r = tf::resolve(workspaceEdit(), sceneEdit());
        QString text = QString("Lists %1 of %2 images").arg(r.filter.apply(snap.files).size()).arg(snap.files.size());
        if (snap.scanning)
            text += " (scanning)";
        mPreview->setText(text);
    }

    void apply()
    {
        json ws = workspaceEdit();
        json wsNow = tf::workspaceRules();
        if (ws["include"] != wsNow.value("include", json::array()) || ws["exclude"] != wsNow.value("exclude", json::array()))
            runCommand("textureFolders", ws);
        json sc = sceneEdit();
        if (sc != tf::sceneRules())
        {
            json call = sc;
            call["scene"] = true;
            call["clear"] = true;
            runCommand("textureFolders", call);
        }
    }

    QPlainTextEdit* mWsInclude = nullptr;
    QPlainTextEdit* mWsExclude = nullptr;
    QCheckBox* mScOverrideInclude = nullptr;
    QCheckBox* mScOverrideExclude = nullptr;
    QPlainTextEdit* mScInclude = nullptr;
    QPlainTextEdit* mScExclude = nullptr;
    QLabel* mPreview = nullptr;
    int mListener = 0;
};
} // namespace

void open(QWidget* anchor, Options options, std::function<void(const std::string&)> pick)
{
    if (!anchor)
        return;
    (new Popup(anchor, std::move(options), std::move(pick)))->start();
}

void open(QWidget* anchor, const std::string& current, std::function<void(const std::string&)> pick)
{
    Options o;
    o.current = current;
    open(anchor, std::move(o), std::move(pick));
}

void showFoldersDialog(QWidget* parent)
{
    auto* d = new FoldersDialog(parent);
    d->show();
}
} // namespace lila::ui::texturePicker
