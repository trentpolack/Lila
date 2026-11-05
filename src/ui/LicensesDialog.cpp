#include "ui/LicensesDialog.h"

#include "core/Version.h"

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDialog>
#include <QDir>
#include <QFile>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QPointer>
#include <QRegularExpression>
#include <QSplitter>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>

#include <utility>

namespace lila::ui
{
namespace licenses
{
namespace
{
QString readFile(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    return QString::fromUtf8(f.readAll());
}

QStringList cells(const QString& row)
{
    QString s = row.trimmed();
    if (s.startsWith('|'))
        s.remove(0, 1);
    if (s.endsWith('|'))
        s.chop(1);
    QStringList out;
    for (const QString& c : s.split('|'))
        out << c.trimmed();
    return out;
}

/** The licenses/<id>.txt a component's text ships as: its vcpkg port name, or the vendored file's name. */
QString fileId(const QString& name)
{
    static const std::pair<const char*, const char*> kIds[] = {
        {"Qt 6", "qtbase"}, {"Eigen", "eigen3"}, {"oneTBB", "tbb"}, {"hwloc", "hwloc"}, {"Alembic", "alembic"}, {"OpenUSD", "usd"},
        {"Draco", "draco"}, {"Imath", "imath"}, {"Manifold", "manifold"}, {"Clipper2", "clipper2"}, {"Slang", "shader-slang"},
        {"Vulkan-Headers", "vulkan-headers"}, {"Vulkan Memory Allocator", "vulkan-memory-allocator"}, {"volk", "volk"},
        {"meshoptimizer", "meshoptimizer"}, {"Jolt", "joltphysics"}, {"GLM", "glm"}, {"nlohmann", "nlohmann-json"},
        {"fast_float", "fast-float"},
        {"cpp-httplib", "cpp-httplib"}, {"Brotli", "brotli"}, {"cgltf", "cgltf"}, {"libwebp", "libwebp"}, {"Zstandard", "zstd"},
        {"Tracy", "tracy"}, {"Capstone", "capstone"}, {"doctest", "doctest"}, {"Dem Bones", "dem-bones"}, {"FFmpeg", "ffmpeg"},
        {"miniaudio", "miniaudio"}, {"Archivo", "archivo"}, {"Inter", "inter"}, {"JetBrains Mono", "jetbrains-mono"},
        {"Tabler Icons", "tabler-icons"}, {"OpenVDB", "openvdb"}, {"MaterialX", "materialx"},
    };
    for (const auto& [prefix, id] : kIds)
        if (name.startsWith(QLatin1String(prefix)))
            return id;
    return name.toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
}
} // namespace

std::vector<Component> parse(const QString& markdown)
{
    std::vector<Component> out;
    QString section;
    QString notes;
    size_t sectionStart = 0;
    auto closeSection = [&] {
        for (size_t i = sectionStart; i < out.size(); ++i)
            out[i].notes = notes.trimmed();
    };
    bool header = false;
    for (const QString& raw : markdown.split('\n'))
    {
        const QString line = raw.trimmed();
        if (line.startsWith("## "))
        {
            closeSection();
            section = line.mid(3).trimmed();
            notes.clear();
            sectionStart = out.size();
            header = true;
            continue;
        }
        if (line.startsWith('|'))
        {
            const QStringList c = cells(line);
            // The first row of each table names its columns and the second is the |---| rule.
            if (header || c.value(0).startsWith("---"))
            {
                header = c.value(0).startsWith("---") ? false : header;
                continue;
            }
            if (c.size() >= 2 && !section.isEmpty())
                out.push_back({c.value(0), c.value(1), c.value(2), c.value(3), section, {}});
            continue;
        }
        if (!line.isEmpty() && !section.isEmpty() && !line.startsWith('#'))
            notes += line + "\n\n";
    }
    closeSection();
    return out;
}

QString noticesPath()
{
    const QString beside = QDir(QCoreApplication::applicationDirPath()).filePath("THIRD-PARTY-NOTICES.md");
    if (QFile::exists(beside))
        return beside;
    return QDir(QString::fromUtf8(LILA_SOURCE_DIR)).filePath("THIRD-PARTY-NOTICES.md");
}

QString licenseText(const Component& c)
{
    return readFile(QDir(QCoreApplication::applicationDirPath()).filePath("licenses/" + fileId(c.name) + ".txt"));
}
} // namespace licenses

namespace
{
QString detailHtml(const licenses::Component& c)
{
    auto esc = [](const QString& s) { return s.toHtmlEscaped(); };
    QString html = "<h3>" + esc(c.name) + "</h3><table cellspacing=4>";
    auto row = [&](const char* label, const QString& value) {
        if (!value.isEmpty())
            html += QString("<tr><td><b>%1</b></td><td>%2</td></tr>").arg(label, value);
    };
    row("License", esc(c.license));
    row("Copyright", esc(c.copyright));
    const QString source = c.source.startsWith("http") ? QString("<a href=\"%1\">%1</a>").arg(esc(c.source)) : esc(c.source);
    row("Source", source);
    html += "</table>";
    if (!c.notes.isEmpty())
        html += "<p>" + esc(c.notes).replace("\n\n", "</p><p>") + "</p>";
    const QString text = licenses::licenseText(c);
    if (!text.isEmpty())
        html += "<hr><pre style=\"white-space: pre-wrap\">" + esc(text) + "</pre>";
    return html;
}
} // namespace

void showLicenses(QWidget* parent)
{
    static QPointer<QDialog> open;
    if (open)
    {
        open->raise();
        open->activateWindow();
        return;
    }
    auto* dlg = new QDialog(parent);
    open = dlg;
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setObjectName("licensesDialog");
    dlg->setWindowTitle("Licenses");
    dlg->resize(900, 620);

    const std::vector<licenses::Component> all = licenses::parse(licenses::readFile(licenses::noticesPath()));

    auto* layout = new QVBoxLayout(dlg);
    auto* search = new QLineEdit(dlg);
    search->setPlaceholderText("Search");
    search->setClearButtonEnabled(true);
    layout->addWidget(search);
    auto* split = new QSplitter(dlg);
    auto* list = new QListWidget(split);
    list->setObjectName("licensesList");
    auto* text = new QTextBrowser(split);
    text->setObjectName("licensesText");
    text->setOpenExternalLinks(true);
    split->addWidget(list);
    split->addWidget(text);
    split->setStretchFactor(1, 1);
    split->setSizes({260, 640});
    layout->addWidget(split, 1);

    for (size_t i = 0; i < all.size(); ++i)
    {
        auto* item = new QListWidgetItem(all[i].name, list);
        item->setData(Qt::UserRole, int(i));
        item->setToolTip(all[i].license);
    }
    QObject::connect(list, &QListWidget::currentRowChanged, dlg, [list, text, all](int) {
        QListWidgetItem* item = list->currentItem();
        text->setHtml(item ? detailHtml(all[size_t(item->data(Qt::UserRole).toInt())]) : QString());
    });
    QObject::connect(search, &QLineEdit::textChanged, dlg, [list, all](const QString& q) {
        for (int r = 0; r < list->count(); ++r)
        {
            QListWidgetItem* item = list->item(r);
            const licenses::Component& c = all[size_t(item->data(Qt::UserRole).toInt())];
            const bool hit = q.isEmpty() || c.name.contains(q, Qt::CaseInsensitive) || c.license.contains(q, Qt::CaseInsensitive) ||
                             c.copyright.contains(q, Qt::CaseInsensitive);
            item->setHidden(!hit);
        }
        if (QListWidgetItem* cur = list->currentItem(); !cur || cur->isHidden())
            for (int r = 0; r < list->count(); ++r)
                if (!list->item(r)->isHidden())
                {
                    list->setCurrentRow(r);
                    break;
                }
    });
    if (list->count() > 0)
        list->setCurrentRow(0);
    else
        text->setPlainText("THIRD-PARTY-NOTICES.md is missing beside the executable.");
    dlg->show();
}

namespace
{
/** Prose in the UI font: blank-line paragraphs, the title and short numbered lines ("1. Grant") as headings. */
QString licenseHtml(const QString& text)
{
    static const QRegularExpression kHeading("^\\d+\\.\\s+\\S.{0,60}$");
    QString html;
    bool first = true;
    for (const QString& para : text.split(QRegularExpression("\\r?\\n\\s*\\r?\\n"), Qt::SkipEmptyParts))
    {
        const QString p = para.trimmed();
        if (first || (!p.contains('\n') && kHeading.match(p).hasMatch()))
            html += QString("<h%1>%2</h%1>").arg(first ? 3 : 4).arg(p.toHtmlEscaped());
        else
            html += "<p>" + p.toHtmlEscaped().replace(QRegularExpression("\\r?\\n"), "<br>") + "</p>";
        first = false;
    }
    return html;
}
} // namespace

void showLilaLicense(QWidget* parent)
{
    QString license = licenses::readFile(QDir(QCoreApplication::applicationDirPath()).filePath("LICENSE"));
    if (license.isEmpty())
        license = licenses::readFile(QDir(QString::fromUtf8(LILA_SOURCE_DIR)).filePath("LICENSE"));
    if (license.isEmpty())
    {
        const QString sha = QString::fromUtf8(version::gitSha());
        const QString ref = sha == "unknown" ? QString("main") : sha;
        QDesktopServices::openUrl(QUrl("https://github.com/lila-3d/Lila/blob/" + ref + "/LICENSE"));
        return;
    }

    static QPointer<QDialog> open;
    if (open)
    {
        open->raise();
        open->activateWindow();
        return;
    }
    auto* dlg = new QDialog(parent);
    open = dlg;
    dlg->setAttribute(Qt::WA_DeleteOnClose);
    dlg->setObjectName("lilaLicenseDialog");
    dlg->setWindowTitle("License");
    dlg->resize(720, 620);
    auto* layout = new QVBoxLayout(dlg);
    auto* text = new QTextBrowser(dlg);
    text->setObjectName("lilaLicenseText");
    text->setOpenExternalLinks(true);
    text->document()->setDocumentMargin(12);
    text->setHtml(licenseHtml(license));
    layout->addWidget(text, 1);
    dlg->show();
}
} // namespace lila::ui
