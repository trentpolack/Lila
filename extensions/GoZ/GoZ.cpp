// GoZ: meshes to ZBrush and back through its GoZ folders, and .GoZ files through File > Import and Export.

#include "GoZFile.h"

#include <lila/Actions.h>
#include <lila/Command.h>
#include <lila/Deformer.h>
#include <lila/Events.h>
#include <lila/Extension.h>
#include <lila/FileTranslator.h>
#include <lila/Graph.h>
#include <lila/Log.h>
#include <lila/Math.h>
#include <lila/Menus.h>
#include <lila/Mesh.h>
#include <lila/Node.h>
#include <lila/Paths.h>
#include <lila/Transform.h>
#include <lila/Viewport.h>

#include <QApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QLabel>
#include <QMessageBox>
#include <QProcess>
#include <QString>
#include <QTimer>
#include <QVBoxLayout>

#include <glm/gtc/matrix_transform.hpp>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <set>

using namespace lila;
using namespace lila::scene;
using namespace lila::ui;
using json = nlohmann::json;
namespace fs = std::filesystem;

namespace goz
{
namespace
{
constexpr const char* kAppName = "Lila";

QTimer* gWatch = nullptr;
ext::Context* gCtx = nullptr;
/** Get from ZBrush's window while it waits for ZBrush. */
QDialog* gWaitDialog = nullptr;

/**
 * A send ZBrush hasn't taken yet. ZBrush rewrites the .GoZ files as it imports them; one still starting drops the request,
 * so it is repeated until the files change.
 */
struct PendingSend
{
    std::vector<std::pair<fs::path, fs::file_time_type>> files;
    std::chrono::steady_clock::time_point started, launched;
    bool waited = false;
};
std::optional<PendingSend> gPendingSend;

fs::path publicRoot()
{
    if (const char* o = std::getenv("LILA_GOZ_ROOT"); o && *o)
        return paths::fromUtf8(o);
#ifdef _WIN32
    const char* pub = std::getenv("PUBLIC");
    return (pub && *pub ? fs::path(pub) : fs::path("C:/Users/Public")) / "Pixologic";
#else
    return "/Users/Shared/Pixologic";
#endif
}

fs::path brushDir() { return publicRoot() / "GoZBrush"; }
fs::path appDir() { return publicRoot() / "GoZApps" / kAppName; }
/** Written by the GoZApps edit command when ZBrush sends to Lila. */
fs::path receiveFlag() { return appDir() / "GoZ_Receive.txt"; }

/** Whether Lila is one of ZBrush's GoZ applications. */
bool installed() { return fs::exists(appDir() / "GoZ_Info.txt"); }

std::string readText(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}

void writeText(const fs::path& p, std::string_view s)
{
    std::error_code ec;
    fs::create_directories(p.parent_path(), ec);
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(s.data(), std::streamsize(s.size()));
    if (!f)
        throw CommandError("can't write " + paths::toUtf8(p));
}

std::string trim(std::string_view s)
{
    const auto b = s.find_first_not_of(" \t\r\n\"");
    const auto e = s.find_last_not_of(" \t\r\n\"");
    return b == std::string_view::npos ? std::string() : std::string(s.substr(b, e - b + 1));
}

fs::path projectDir()
{
    const std::string p = trim(readText(brushDir() / "GoZ_ProjectPath.txt"));
    fs::path dir = p.empty() ? publicRoot() / "GoZProjects" / "Default" : paths::fromUtf8(p);
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

constexpr double kDefaultScale = 0.01;

/**
 * Scene space to ZBrush's: Y down and Z away, and scaled (cm to ZBrush units 100 to 1 by default, so a person is about
 * two units tall, the size ZBrush's DynaMesh and ZRemesher expect).
 */
glm::dmat4 sceneToFile(double scale)
{
    glm::dmat4 m(1.0);
    if (math::upAxis() == math::UpAxis::Z) // (x, y, z) -> (x, z, -y)
        m = glm::dmat4(glm::dvec4(1, 0, 0, 0), glm::dvec4(0, 0, -1, 0), glm::dvec4(0, 1, 0, 0), glm::dvec4(0, 0, 0, 1));
    return glm::scale(glm::dmat4(1.0), glm::dvec3(scale, -scale, -scale)) * m;
}

double scaleArg(const Args& a, const char* cmd)
{
    const double s = a.get<double>("scale", kDefaultScale);
    if (!(s > 0.0))
        throw CommandError(std::string(cmd) + ": scale must be above 0");
    return s;
}



void transformPoints(geo::Mesh& mesh, const glm::dmat4& m)
{
    for (geo::Index p = 0; p < mesh.pointCapacity(); ++p)
        if (mesh.pointValid(p))
            mesh.setPosition(p, glm::dvec3(m * glm::dvec4(mesh.position(p), 1.0)));
    if (glm::determinant(glm::dmat3(m)) < 0.0)
        for (geo::Index f = 0; f < mesh.faceCapacity(); ++f)
            if (mesh.faceValid(f))
                mesh.flipFace(f);
}

std::string fileName(std::string s)
{
    while (!s.empty() && s.front() == '|')
        s.erase(s.begin());
    for (char& c : s)
        if (c == '|' || c == ':' || c == '/' || c == '\\')
            c = '_';
    return s.empty() ? "Mesh" : s;
}

/** The node's short name when it finds the node, otherwise its path made into a file name. */
std::string baseName(Graph& g, const Node& n)
{
    try
    {
        if (g.find(n.name()) == &n)
            return fileName(n.name());
    }
    catch (const CommandError&)
    {
    }
    return fileName(n.path());
}

std::string lower(std::string s)
{
    for (char& c : s)
        c = char(std::tolower(uint8_t(c)));
    return s;
}

constexpr const char* kNameAttr = "gozName";

/** The subtool name a mesh went to ZBrush under, or empty. */
std::string storedName(const Node& n)
{
    const int at = n.type().attrIndex(kNameAttr);
    return at < 0 ? std::string() : n.as<std::string>(at);
}

void storeName(Graph& g, Node& n, const std::string& name)
{
    if (storedName(n) == name)
        return;
    if (n.type().attrIndex(kNameAttr) < 0)
        ops::execute("addAttr", {{"args", {n.path()}},
                                 {"longName", kNameAttr},
                                 {"dataType", "string"},
                                 {"hidden", true},
                                 {"doc", "The ZBrush subtool this mesh goes to and comes back from."}});
    g.setValue({n.id(), n.type().attrIndex(kNameAttr), -1}, name);
}

std::vector<Node*> allMeshes(Graph& g)
{
    std::vector<Node*> out;
    for (NodeId id : g.nodes())
        if (Node* n = g.node(id); n && n->type().isA("mesh"))
            out.push_back(n);
    return out;
}

/**
 * The subtool name a mesh goes to ZBrush under, kept on the mesh after its first send so renames don't orphan its subtool.
 * New names end in "-", which node names never hold: ZBrush's GoZ puts a mesh into the first subtool whose name starts its
 * own ("TressDressPocket" lands in "TressDress"), and no such name starts another. fresh gives the mesh a name no subtool
 * has yet, so ZBrush adds a new one.
 */
std::string gozName(Graph& g, Node& n, bool fresh, const fs::path& project)
{
    const std::string base = baseName(g, n);
    std::set<std::string> taken;
    std::vector<std::string> others;
    for (Node* o : allMeshes(g))
        if (o != &n)
        {
            if (std::string name = storedName(*o); !name.empty())
                taken.insert(lower(name));
            others.push_back(lower(o->name()));
        }
    std::string name = storedName(n);
    const bool startsAnother = std::ranges::any_of(others, [&](const std::string& o) { return o.size() > name.size() && o.starts_with(lower(name)); });
    const bool keep = !name.empty() && !fresh && (!taken.contains(lower(name)) || name == base + "-") && (name.ends_with('-') || !startsAnother);
    if (!keep)
    {
        auto used = [&](const std::string& candidate) {
            return taken.contains(lower(candidate)) || (fresh && (candidate == name || fs::exists(project / paths::fromUtf8(candidate + ".GoZ"))));
        };
        std::string candidate = base + "-";
        for (int k = 2; used(candidate); ++k)
            candidate = base + "_" + std::to_string(k) + "-";
        name = candidate;
        storeName(g, n, name);
    }
    return name;
}

Node* findMesh(Graph& g, const std::string& name)
{
    try
    {
        Node* n = g.find(name);
        return n && n->type().isA("mesh") ? n : nullptr;
    }
    catch (const CommandError&)
    {
        return nullptr;
    }
}

/** The mesh a subtool belongs to: the one that went under its name, else the one named like it. */
Node* meshNamed(Graph& g, const std::string& name)
{
    const std::string want = lower(name);
    for (Node* n : allMeshes(g))
        if (lower(storedName(*n)) == want)
            return n;
    if (Node* n = findMesh(g, name))
        return n;
    const size_t end = name.find_last_not_of("-_");
    return end != std::string::npos && end + 1 < name.size() ? findMesh(g, name.substr(0, end + 1)) : nullptr;
}

/** The named meshes and the meshes under them, the selection by default. */
std::vector<Node*> meshesOf(Graph& g, const Args& a, size_t firstObject, const char* cmd)
{
    std::vector<NodeId> roots;
    for (size_t k = firstObject; k < a.positional.size(); ++k)
    {
        Node* n = g.find(a.positional[k].get<std::string>());
        if (!n)
            throw CommandError(std::string(cmd) + ": no node named '" + a.positional[k].get<std::string>() + "'");
        roots.push_back(n->id());
    }
    if (roots.empty())
        roots = g.selection();
    std::vector<Node*> out;
    auto addMesh = [&](NodeId id) {
        Node* n = g.node(id);
        if (n && n->type().isA("mesh") && std::ranges::find(out, n) == out.end())
            out.push_back(n);
    };
    for (NodeId r : roots)
    {
        addMesh(r);
        for (NodeId d : g.descendants(r))
            addMesh(d);
    }
    if (out.empty())
        throw CommandError(std::string(cmd) + ": select a mesh");
    return out;
}

/** The mesh's cage (its base, under any modifiers) in GoZ file space. */
geo::Mesh fileMesh(Graph& g, Node& n, double scale)
{
    MeshPtr cage = modifiers::cage(g, n.id());
    if (!cage)
        throw CommandError(n.name() + " has no mesh");
    geo::Mesh mesh = *cage;
    transformPoints(mesh, sceneToFile(scale) * n.as<glm::dmat4>(Transform::kWorldMatrix));
    return mesh;
}

File readFile(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw CommandError("can't read " + paths::toUtf8(path));
    const std::string bytes{std::istreambuf_iterator<char>(f), {}};
    try
    {
        return read(bytes);
    }
    catch (const std::runtime_error& e)
    {
        throw CommandError(paths::toUtf8(path) + ": " + e.what());
    }
}

bool sameTopology(const geo::Mesh& a, const geo::Mesh& b)
{
    if (a.numPoints() != b.numPoints() || a.numFaces() != b.numFaces() || a.faceCapacity() != b.faceCapacity())
        return false;
    for (geo::Index f = 0; f < a.faceCapacity(); ++f)
        if (a.faceValid(f) != b.faceValid(f) || (a.faceValid(f) && a.facePoints(f) != b.facePoints(f)))
            return false;
    return true;
}

/**
 * Puts what ZBrush sent into an existing mesh. The same topology moves the points and keeps everything else (UVs,
 * materials, hard edges); new topology replaces the base mesh.
 */
void update(Graph& g, Node& n, File& file, double scale)
{
    transformPoints(file.mesh, glm::inverse(sceneToFile(scale) * n.as<glm::dmat4>(Transform::kWorldMatrix)));
    const PlugRef base = modifiers::editPlug(g, n.id());
    g.evaluate();
    const MeshPtr& cur = std::get<MeshPtr>(g.storedValue(base));
    if (!cur)
    {
        g.setValue(base, MeshPtr(std::make_shared<geo::Mesh>(std::move(file.mesh))));
        return;
    }
    auto edited = std::make_shared<geo::Mesh>(*cur);
    edited->compact();
    if (!sameTopology(*edited, file.mesh))
    {
        g.setValue(base, MeshPtr(std::make_shared<geo::Mesh>(std::move(file.mesh))));
        return;
    }
    std::ranges::copy(file.mesh.positions(), edited->positions().begin());
    if (std::span<const float> m = file.mesh.pointAttrs().get<float>(kMaskLayer); !m.empty())
        std::ranges::copy(m, edited->pointAttrs().getOrAdd<float>(kMaskLayer).begin());
    if (std::span<const int32_t> gr = file.mesh.faceAttrs().get<int32_t>(kPolygroupLayer); !gr.empty())
        std::ranges::copy(gr, edited->faceAttrs().getOrAdd<int32_t>(kPolygroupLayer).begin());
    if (geo::colors::has(file.mesh, kPolypaintSet))
    {
        std::string set = geo::colors::current(*edited);
        if (set.empty())
        {
            set = kPolypaintSet;
            geo::colors::add(*edited, set);
        }
        geo::colors::setFromPoints(*edited, set, geo::colors::pointColors(file.mesh, kPolypaintSet));
    }
    g.setValue(base, MeshPtr(std::move(edited)));
}

Node* create(Graph& g, const std::string& name, File& file, double scale)
{
    transformPoints(file.mesh, glm::inverse(sceneToFile(scale)));
    Node* n = g.createNode(*types::find("mesh"), Graph::validName(name) ? name : "Mesh", kNoNode, true);
    g.setValue({n->id(), MeshNode::kInMesh, -1}, MeshPtr(std::make_shared<geo::Mesh>(std::move(file.mesh))));
    return n;
}

/** {updated, created}: meshes named like a file are updated, the rest are new. */
json receive(const std::vector<fs::path>& files, bool updateExisting, double scale)
{
    Graph& g = Graph::current();
    Graph::Batch batch(g);
    json updated = json::array(), created = json::array();
    std::vector<NodeId> ids;
    for (const fs::path& path : files)
    {
        File file = readFile(path);
        std::string name = paths::toUtf8(path.stem());
        Node* n = updateExisting ? meshNamed(g, name) : nullptr;
        if (n)
        {
            update(g, *n, file, scale);
            updated.push_back(n->path());
        }
        else
        {
            n = create(g, name, file, scale);
            created.push_back(n->path());
        }
        storeName(g, *n, name);
        ids.push_back(n->id());
    }
    g.select(ids);
    return {{"updated", std::move(updated)}, {"created", std::move(created)}};
}

/** GoZ_ObjectList.txt: one path per line, usually without the .GoZ extension. */
std::vector<fs::path> objectList()
{
    std::vector<fs::path> files;
    const std::string text = readText(brushDir() / "GoZ_ObjectList.txt");
    size_t at = 0;
    while (at < text.size())
    {
        size_t end = text.find('\n', at);
        if (end == std::string::npos)
            end = text.size();
        if (std::string line = trim(std::string_view(text).substr(at, end - at)); !line.empty())
        {
            fs::path p = paths::fromUtf8(line);
            if (p.extension() != ".GoZ" && p.extension() != ".goz")
                p += ".GoZ";
            files.push_back(std::move(p));
        }
        at = end + 1;
    }
    return files;
}

/** Written by gozSend: the object list's write time as Lila left it, so a receive can tell whether ZBrush wrote since. */
fs::path sentStamp() { return appDir() / "GoZ_Sent.txt"; }

std::string listStamp()
{
    std::error_code ec;
    const auto t = fs::last_write_time(brushDir() / "GoZ_ObjectList.txt", ec);
    return ec ? std::string() : std::format("{}", t.time_since_epoch().count());
}

std::string exeName(const fs::path& exe) { return paths::toUtf8(exe.filename()); }

/** GoZApps/Lila: what makes Lila one of ZBrush's GoZ applications. */
fs::path install(const fs::path& exe, bool keepConfig)
{
    const fs::path dir = appDir();
    // No flip flags: Lila turns to and from ZBrush's axes itself (sceneToFile), so every file is in ZBrush space.
    writeText(dir / "GoZ_Info.txt", "NAME\t\t\t=\t\"Lila\"\r\n"
                                    "GOZ_VERSION\t\t=\t1\r\n"
                                    "EXTENSION\t\t=\t\".GoZ\"\r\n"
                                    "TAMPLATE\t\t=\t\"GoZ Complete Binary.GoZ\"\r\n"
                                    "LOCATE_COMMAND\t=\t\"GoZLocateLila.bat\"\r\n"
                                    "EDIT_COMMAND\t=\t\"GoZBrushToLila.bat\"\r\n"
                                    "\r\n"
                                    "NORMAL_MAP_FLIP_VERT\t=\tTRUE\r\n");
    std::string path = paths::toUtf8(exe.lexically_normal());
    std::ranges::replace(path, '/', '\\');
    if (!keepConfig || !fs::exists(dir / "GoZ_Config.txt"))
        writeText(dir / "GoZ_Config.txt","PATH = \"" + path + "\"\r\n");
    // ZBrush's "path to Lila" runs this and waits for the config to be written.
    writeText(dir / "GoZLocateLila.bat", "@echo PATH = \"" + path + "\"> \"%~dp0GoZ_Config.txt\"\r\n");
    // Leaves the flag for a running Lila to pick up, or starts Lila, which picks it up as it loads.
    writeText(dir / "GoZBrushToLila.bat", "@echo off\r\n"
                                          "type nul > \"%~dp0GoZ_Receive.txt\"\r\n"
                                          "\"%SystemRoot%\\System32\\tasklist.exe\" /FI \"IMAGENAME eq " +
                                              exeName(exe) + "\" 2>nul | \"%SystemRoot%\\System32\\find.exe\" /I \"" + exeName(exe) +
                                              "\" >nul && exit /b 0\r\n"
                                              "start \"\" \"" +
                                              path + "\"\r\n");
    return dir;
}

#ifdef _WIN32
bool zbrushRunning()
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE)
        return false;
    PROCESSENTRY32W e{};
    e.dwSize = sizeof(e);
    bool found = false;
    for (BOOL ok = Process32FirstW(snap, &e); ok && !found; ok = Process32NextW(snap, &e))
        found = _wcsicmp(e.szExeFile, L"ZBrush.exe") == 0;
    CloseHandle(snap);
    return found;
}
#else
bool zbrushRunning() { return true; }
#endif

fs::path lilaExe()
{
    fs::path exe = paths::executableDir() / "lila.exe";
#ifndef _WIN32
    exe = paths::executableDir() / "lila";
#endif
    return exe;
}

/** Has ZBrush pick up GoZ_ObjectList.txt. */
void launchGoZ()
{
    const fs::path launcher = brushDir() / ("GoZBrushFromApp" + std::string(
#ifdef _WIN32
                                                                   ".exe"
#else
                                                                   ".app"
#endif
                                                                   ));
    if (!fs::exists(launcher))
        throw CommandError("gozSend: ZBrush's GoZ isn't installed (no " + paths::toUtf8(launcher) + "); run ZBrush once, or reinstall GoZ from ZBrush");
    if (!QProcess::startDetached(QString::fromStdWString(launcher.wstring()), {}, QString::fromStdWString(brushDir().wstring())))
        throw CommandError("gozSend: couldn't start " + paths::toUtf8(launcher));
}

json send(const Args& a)
{
    Graph& g = Graph::current();
    std::vector<Node*> meshes = meshesOf(g, a, 0, "gozSend");
    g.evaluate();
    const double scale = scaleArg(a, "gozSend");
    if (a.get<bool>("launch", true) && !zbrushRunning())
        throw CommandError("gozSend: ZBrush isn't open; start ZBrush, then send again");
    // ZBrush reads GoZApps as it starts, and one that doesn't know Lila quits on the send.
    if (!installed())
        throw CommandError("gozSend: Lila isn't one of ZBrush's GoZ applications yet; run gozInstall(), restart ZBrush, then send again");
    install(lilaExe(), true);
    const fs::path project = projectDir();
    const bool fresh = a.get<bool>("newSubtools");
    std::string list;
    json files = json::array();
    for (Node* n : meshes)
    {
        const std::string name = gozName(g, *n, fresh, project);
        const fs::path base = project / paths::fromUtf8(name);
        fs::path file = base;
        file += ".GoZ";
        writeText(file, write(fileMesh(g, *n, scale), name));
        // ZBrush keeps its own tool next to the mesh; an empty one marks the name as taken.
        fs::path ztn = base;
        ztn += ".ztn";
        if (!fs::exists(ztn))
            writeText(ztn, "");
        std::string line = paths::toUtf8(base);
        std::ranges::replace(line, '\\', '/');
        list += line + "\n";
        files.push_back(paths::toUtf8(file));
    }
    writeText(brushDir() / "GoZ_ObjectList.txt", list);
    writeText(sentStamp(), listStamp());
    writeText(brushDir() / "GoZ_Application.txt", kAppName);
    if (a.get<bool>("launch", true))
    {
        launchGoZ();
        PendingSend pending;
        pending.started = pending.launched = std::chrono::steady_clock::now();
        for (const json& f : files)
        {
            const fs::path path = paths::fromUtf8(f.get<std::string>());
            std::error_code ec;
            pending.files.emplace_back(path, fs::last_write_time(path, ec));
        }
        gPendingSend = std::move(pending);
    }
    return files;
}

void registerCommands(ext::Context& ctx)
{
    using T = ArgType;
    ctx.registerCommand({
        .name = "gozSend",
        .doc = "Send meshes to ZBrush through GoZ: each opens as a subtool, or replaces the subtool it came from (the name it went under is kept on the mesh as gozName). Sends the base mesh "
               "under any modifiers, in world space. Needs gozInstall first. Returns the .GoZ files written.",
        .positionalDoc = "meshes (default the selection)",
        .flags = {{"launch", "", T::Bool, true, "Start or switch to ZBrush. False only writes the files."},
                  {"scale", "", T::Float, kDefaultScale, "ZBrush units per cm. Receive with the same scale."},
                  {"newSubtools", "", T::Bool, false, "Arrive as new subtools beside any from earlier sends, instead of replacing them."}},
        .undoable = false,
        .run = [](const Args& a) -> json { return send(a); },
        .examples = {"gozSend()", "gozSend(['Head', 'Body'])"},
    });
    ctx.registerCommand({
        .name = "gozReceive",
        .doc = "Bring in what ZBrush last sent with GoZ. A mesh named like a subtool takes its shape (and its polypaint, mask and "
               "polygroups): the same topology moves the points and keeps UVs, materials and hard edges; new topology replaces "
               "the base mesh. Other subtools come in as new meshes. Runs on its own when ZBrush sends to Lila. Returns "
               "{updated, created}.",
        .flags = {{"scale", "", T::Float, kDefaultScale, "ZBrush units per cm, as sent."},
                  {"force", "", T::Bool, false, "Read the files even when ZBrush hasn't sent anything since Lila last sent."}},
        .run = [](const Args& a) -> json {
            if (!a.get<bool>("force") && readText(sentStamp()) == listStamp())
                throw CommandError("gozReceive: ZBrush hasn't sent anything since Send to ZBrush; press GoZ in ZBrush, with Lila as the "
                                   "target in Preferences > GoZ");
            std::vector<fs::path> files = objectList();
            std::erase_if(files, [](const fs::path& p) { return !fs::exists(p); });
            if (files.empty())
                throw CommandError("gozReceive: ZBrush hasn't sent anything");
            return receive(files, true, scaleArg(a, "gozReceive"));
        },
    });
    ctx.registerCommand({
        .name = "gozInstall",
        .doc = "Make Lila one of ZBrush's GoZ applications, so gozSend can send to ZBrush and ZBrush's GoZ button can send back. "
               "Restart ZBrush afterwards. Returns the folder written.",
        .flags = {{"exe", "", T::String, nullptr, "The Lila to start when ZBrush sends and Lila isn't running. Default: this one."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            const fs::path exe = a.has("exe") ? paths::fromUtf8(a.get<std::string>("exe")) : lilaExe();
            return paths::toUtf8(install(exe, false));
        },
    });
    ctx.registerCommand({
        .name = "importGoZ",
        .doc = "Import .GoZ files as new meshes, with their UVs, polypaint, mask and polygroups. Returns the new meshes.",
        .positionalDoc = "path",
        .flags = {{"scale", "", T::Float, kDefaultScale, "ZBrush units per cm."}},
        .run = [](const Args& a) -> json {
            if (a.positional.empty() || !a.positional[0].is_string())
                throw CommandError("importGoZ: path required");
            return receive({paths::fromUtf8(a.positional[0].get<std::string>())}, false, scaleArg(a, "importGoZ"))["created"];
        },
    });
    ctx.registerCommand({
        .name = "exportGoZ",
        .doc = "Export meshes to a .GoZ file, in world space; several meshes go out as one. Returns the path.",
        .positionalDoc = "path, meshes (default the selection)",
        .flags = {{"scale", "", T::Float, kDefaultScale, "ZBrush units per cm."}},
        .undoable = false,
        .run = [](const Args& a) -> json {
            if (a.positional.empty() || !a.positional[0].is_string())
                throw CommandError("exportGoZ: path required");
            const fs::path path = paths::fromUtf8(a.positional[0].get<std::string>());
            Graph& g = Graph::current();
            std::vector<Node*> meshes = meshesOf(g, a, 1, "exportGoZ");
            g.evaluate();
            const double scale = scaleArg(a, "exportGoZ");
            geo::Mesh mesh = fileMesh(g, *meshes.front(), scale);
            for (size_t k = 1; k < meshes.size(); ++k)
                mesh.append(fileMesh(g, *meshes[k], scale));
            writeText(path, write(mesh, paths::toUtf8(path.stem())));
            return paths::toUtf8(path);
        },
    });
    io::translators::add(ctx, {.id = "goz",
                               .label = "GoZ",
                               .doc = "ZBrush GoZ meshes: points, triangles and quads, UVs, polypaint, mask and polygroups.",
                               .extensions = {".goz"},
                               .import = {.command = "importGoZ"},
                               .exportSelection = {.command = "exportGoZ"}});
}

/** Repeats a send until ZBrush takes it (a ZBrush still starting ignores it), and says so if it never does. */
void pollSend()
{
    if (!gPendingSend)
        return;
    using namespace std::chrono;
    const PendingSend& p = *gPendingSend;
    const bool taken = std::ranges::any_of(p.files, [](const auto& f) {
        std::error_code ec;
        const auto t = fs::last_write_time(f.first, ec);
        return !ec && t != f.second;
    });
    using viewportToast::Kind;
    if (taken)
    {
        if (p.waited)
            viewportToast::showActive("Sent to ZBrush", 3000);
        gPendingSend.reset();
        return;
    }
    const auto now = steady_clock::now();
    const char* problem = nullptr;
    if (!zbrushRunning())
        problem = "ZBrush closed before it took the meshes";
    else if (now - p.started > minutes(2))
        problem = "ZBrush didn't take the meshes: send again once it has finished loading";
    if (problem)
    {
        gPendingSend.reset();
        viewportToast::showActive(problem, 8000, Kind::Error);
        return;
    }
    if (!p.waited && now - p.started > seconds(3))
    {
        gPendingSend->waited = true;
        viewportToast::showActive("Waiting for ZBrush to finish loading...", 7000);
    }
    if (now - p.launched > seconds(10))
    {
        viewportToast::showActive("Waiting for ZBrush to finish loading...", 7000);
        gPendingSend->launched = now;
        try
        {
            launchGoZ();
        }
        catch (const std::exception& e)
        {
            gPendingSend.reset();
            gCtx->log(LogLevel::Error, std::string("GoZ: ") + e.what());
        }
    }
}

void pollReceive()
{
    std::error_code ec;
    if (!fs::exists(receiveFlag(), ec))
        return;
    fs::remove(receiveFlag(), ec);
    try
    {
        ops::execute("gozReceive", {{"force", true}}, CallSource::UI);
    }
    catch (const std::exception& e)
    {
        gCtx->log(LogLevel::Error, std::string("GoZ: ") + e.what());
    }
    if (gWaitDialog)
        gWaitDialog->accept();
}

/** Send to ZBrush: the first time, offers to add Lila to ZBrush's GoZ applications instead of sending. */
void sendFromMenu()
{
    QWidget* parent = QApplication::activeWindow();
    if (installed())
    {
        if (!zbrushRunning())
        {
            QMessageBox::information(parent, "Send to ZBrush", "ZBrush isn't open. Start ZBrush, then send again.");
            return;
        }
        actions::runWithOptions("gozSend", json::object());
        return;
    }
    if (QMessageBox::question(parent, "Send to ZBrush",
                              "Lila isn't one of ZBrush's GoZ applications yet. Add it now?\n\nZBrush will need a restart before Lila can send to it.",
                              QMessageBox::Ok | QMessageBox::Cancel, QMessageBox::Ok) != QMessageBox::Ok)
        return;
    actions::runWithOptions("gozInstall", json::object());
    if (!installed())
        return;
    QMessageBox::information(parent, "Send to ZBrush", "Lila is now one of ZBrush's GoZ applications. Restart ZBrush, then send again.");
}

/**
 * Get from ZBrush: Lila can't ask ZBrush for its subtools, so this says which ZBrush button sends them and waits until they
 * arrive, the user cancels or it times out.
 */
void getFromMenu()
{
    QWidget* parent = QApplication::activeWindow();
    if (!installed())
    {
        QMessageBox::information(parent, "Get from ZBrush", "ZBrush can't send to Lila yet. Use Send to ZBrush once to set it up.");
        return;
    }
    if (!zbrushRunning())
    {
        QMessageBox::information(parent, "Get from ZBrush", "ZBrush isn't open. Start ZBrush with your model, then try again.");
        return;
    }
    std::error_code ec;
    fs::remove(receiveFlag(), ec);

    QDialog dialog(parent);
    dialog.setWindowTitle("Get from ZBrush");
    auto* layout = new QVBoxLayout(&dialog);
    auto* text = new QLabel("In ZBrush's <b>Tool</b> palette, press <b>GoZ</b> to send the current subtool, or <b>All</b> to send every "
                            "subtool. Lila brings them in when they arrive.<br><br>If ZBrush asks which application to send to, pick "
                            "<b>Lila</b>.");
    text->setWordWrap(true);
    text->setMinimumWidth(360);
    layout->addWidget(text);
    if (QPixmap shot(QString::fromStdWString((gCtx->resources() / "zbrush-goz-buttons.png").wstring())); !shot.isNull())
    {
        auto* image = new QLabel;
        image->setPixmap(shot);
        image->setAlignment(Qt::AlignCenter);
        layout->addWidget(image);
    }
    layout->addWidget(new QLabel("Waiting for ZBrush..."));
    auto* buttons = new QDialogButtonBox(QDialogButtonBox::Cancel);
    QObject::connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(buttons);
    constexpr int kTimedOut = 2;
    QTimer::singleShot(std::chrono::minutes(3), &dialog, [&dialog] { dialog.done(kTimedOut); });

    gWaitDialog = &dialog;
    const int result = dialog.exec();
    gWaitDialog = nullptr;
    if (result == kTimedOut)
        QMessageBox::information(parent, "Get from ZBrush",
                                 "Nothing arrived from ZBrush. Press GoZ in ZBrush whenever you're ready: Lila brings in what ZBrush sends "
                                 "without this window open.");
}

void registerUi(ext::Context& ctx)
{
    actions::add(ctx, {.id = "goz.send",
                       .label = "Send to ZBrush",
                       .category = "File",
                       .run = sendFromMenu,
                       .doc = "Sends the selected meshes to ZBrush. Press GoZ in ZBrush to send them back.",
                       .python = "gozSend()",
                       .command = "gozSend",
                       .icon = "send"});
    actions::add(ctx, {.id = "goz.receive",
                       .label = "Get from ZBrush",
                       .category = "File",
                       .run = getFromMenu,
                       .doc = "Shows how to send subtools back from ZBrush and waits for them.",
                       .python = "gozReceive()",
                       .command = "gozReceive"});
    menus::addItem(ctx, {.id = "goz.menu.section", .menu = "File", .kind = menus::Kind::Section, .label = "ZBrush", .before = "File/#Unreal"});
    menus::addItem(ctx, {.id = "goz.menu.send", .menu = "File", .action = "goz.send", .before = "File/#Unreal"});
    menus::addItem(ctx, {.id = "goz.menu.receive", .menu = "File", .action = "goz.receive", .before = "File/#Unreal"});

    // Only the UI event loop sends AppIdle, so lilabatch and tests leave the flag alone.
    ctx.subscribe(events::Event::AppIdle, [](const events::EventData&) {
        if (gWatch)
            return;
        gWatch = new QTimer();
        QObject::connect(gWatch, &QTimer::timeout, [] {
            pollReceive();
            pollSend();
        });
        gWatch->start(500);
        pollReceive();
    });
}
} // namespace
} // namespace goz

LILA_EXTENSION_EXPORT int lilaExtensionInit(lila::ext::Context& ctx)
{
    goz::gCtx = &ctx;
    goz::registerCommands(ctx);
    goz::registerUi(ctx);
    return 0;
}

LILA_EXTENSION_EXPORT void lilaExtensionUninit(lila::ext::Context&)
{
    delete goz::gWatch;
    goz::gWatch = nullptr;
    goz::gPendingSend.reset();
}
