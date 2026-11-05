#include "io/SceneHistory.h"

#include "core/Paths.h"
#include "core/Workspace.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>
#include <tuple>

namespace lila::io::history
{
namespace
{
bool allDigits(std::string_view s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

bool isStamp(std::string_view s)
{
    return s.size() == 15 && s[8] == '-' && allDigits(s.substr(0, 8)) && allDigits(s.substr(9));
}

std::string hash8(const std::string& s)
{
    uint32_t h = 2166136261u;
    for (unsigned char c : s)
        h = (h ^ uint32_t(std::tolower(c))) * 16777619u;
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x", h);
    return buf;
}

bool isSceneExt(const fs::path& p)
{
    return p.extension() == ".lila" || p.extension() == ".lilaa";
}

std::string pad(int n, int width)
{
    std::string s = std::to_string(n);
    if (int(s.size()) < width)
        s.insert(0, size_t(width) - s.size(), '0');
    return s;
}
} // namespace

const char* kindName(Kind k)
{
    switch (k)
    {
    case Kind::Autosave:
        return "autosave";
    case Kind::Emergency:
        return "emergency";
    case Kind::Backup:
        return "backup";
    case Kind::Increment:
        return "increment";
    case Kind::Revision:
        return "revision";
    }
    return "?";
}

VersionedStem parseVersionedStem(const std::string& stem)
{
    size_t at = stem.rfind("_v");
    if (at == std::string::npos || at == 0 || !allDigits(std::string_view(stem).substr(at + 2)))
        return {stem, 0, 0};
    std::string digits = stem.substr(at + 2);
    return {stem.substr(0, at), std::stoi(digits), int(digits.size())};
}

fs::path nextIncrement(const fs::path& scene)
{
    VersionedStem v = parseVersionedStem(scene.stem().string());
    int width = v.width ? v.width : 3;
    int n = v.width ? v.number + 1 : 2;
    fs::path ext = scene.extension().empty() ? fs::path(".lila") : scene.extension();
    std::error_code ec;
    for (;; ++n)
    {
        std::string stem = v.base + "_v" + pad(n, width);
        fs::path candidate = scene.parent_path() / (stem + ext.string());
        if (!fs::exists(candidate, ec) && !fs::exists(scene.parent_path() / (stem + ".lila"), ec) &&
            !fs::exists(scene.parent_path() / (stem + ".lilaa"), ec))
            return candidate;
    }
}

std::string timestamp(std::chrono::system_clock::time_point t)
{
    std::time_t tt = std::chrono::system_clock::to_time_t(t);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm);
    return buf;
}

std::string stemFor(const fs::path& scene)
{
    return scene.empty() ? std::string("untitled") : scene.stem().string();
}

fs::path folderFor(const fs::path& base, const fs::path& scene, const fs::path& workspaceRoot)
{
    if (scene.empty())
        return base / "untitled";
    std::string stem = scene.stem().string();
    std::error_code ec;
    fs::path abs = fs::weakly_canonical(fs::absolute(scene, ec), ec);
    if (!workspaceRoot.empty())
    {
        fs::path root = fs::weakly_canonical(workspaceRoot, ec);
        fs::path rel = abs.parent_path().lexically_relative(root);
        if (!rel.empty() && *rel.begin() != "..")
            return (rel == "." ? base : base / rel) / stem;
    }
    return base / (stem + "." + hash8(abs.generic_string()));
}

std::optional<ParsedName> parseName(const std::string& filename, const std::string& stem)
{
    if (filename.size() <= stem.size() + 1 || filename.compare(0, stem.size(), stem) != 0 || filename[stem.size()] != '.')
        return std::nullopt;
    std::string_view rest = std::string_view(filename).substr(stem.size() + 1);
    if (rest.size() < 15 || !isStamp(rest.substr(0, 15)))
        return std::nullopt;
    ParsedName out;
    out.stamp = std::string(rest.substr(0, 15));
    rest.remove_prefix(15);
    if (!rest.empty() && rest[0] == '-')
    {
        size_t dot = rest.find('.');
        if (dot == std::string_view::npos || !allDigits(rest.substr(1, dot - 1)))
            return std::nullopt;
        out.counter = std::stoi(std::string(rest.substr(1, dot - 1)));
        rest.remove_prefix(dot);
    }
    if (rest == ".lila" || rest == ".lilaa")
        return out;
    for (const char* tag : {"autosave", "emergency"})
        if (rest == std::string(".") + tag + ".lila")
        {
            out.tag = tag;
            return out;
        }
    return std::nullopt;
}

fs::path uniqueName(const fs::path& folder, const std::string& stem, const std::string& stamp, const std::string& tag,
                    const std::string& ext)
{
    std::string suffix = tag.empty() ? ext : "." + tag + ".lila";
    // Past the highest counter in use, not the first gap, so a name freed by pruning never sorts before its elders.
    int last = -1;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(folder, ec))
        if (auto parsed = parseName(e.path().filename().string(), stem); parsed && parsed->stamp == stamp && parsed->tag == tag)
            last = std::max(last, parsed->counter);
    if (last < 0)
        return folder / (stem + "." + stamp + suffix);
    return folder / (stem + "." + stamp + "-" + std::to_string(std::max(2, last + 1)) + suffix);
}

std::vector<fs::path> entries(const fs::path& folder, const std::string& stem, const std::string& tag)
{
    std::vector<std::tuple<std::string, int, fs::path>> found;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(folder, ec))
    {
        if (!e.is_regular_file(ec))
            continue;
        auto parsed = parseName(e.path().filename().string(), stem);
        if (parsed && parsed->tag == tag)
            found.emplace_back(parsed->stamp, parsed->counter, e.path());
    }
    std::sort(found.begin(), found.end());
    std::vector<fs::path> out;
    for (auto& f : found)
        out.push_back(std::get<2>(f));
    return out;
}

int prune(const fs::path& folder, const std::string& stem, const std::string& tag, int keep)
{
    if (keep <= 0)
        return 0;
    std::vector<fs::path> all = entries(folder, stem, tag);
    int removed = 0;
    std::error_code ec;
    for (size_t i = 0; i + size_t(keep) < all.size(); ++i)
        if (fs::remove(all[i], ec))
            ++removed;
    return removed;
}

fs::path baseDir(const char* name)
{
    return Workspace::get().isOpen() ? Workspace::get().stateDir(false) / name : paths::userConfigDir() / name;
}

fs::path autosaveFolder(const fs::path& scene)
{
    return folderFor(baseDir("autosave"), scene, Workspace::get().root());
}

fs::path backupFolder(const fs::path& scene)
{
    return folderFor(baseDir("backups"), scene, Workspace::get().root());
}

fs::path backup(const fs::path& scene, const fs::path& folder, int keep)
{
    std::error_code ec;
    if (!fs::is_regular_file(scene, ec))
        return {};
    fs::create_directories(folder, ec);
    std::string ext = scene.extension() == ".lilaa" ? ".lilaa" : ".lila";
    fs::path target = uniqueName(folder, scene.stem().string(), timestamp(), "", ext);
    if (!fs::copy_file(scene, target, ec))
        return {};
    // A read-only Perforce original copies read-only, which would stop pruning from deleting it later.
    fs::permissions(target, fs::perms::owner_write, fs::perm_options::add, ec);
    prune(folder, scene.stem().string(), "", keep);
    return target;
}

std::vector<fs::path> increments(const fs::path& scene)
{
    std::vector<fs::path> out;
    if (scene.empty())
        return out;
    std::string base = parseVersionedStem(scene.stem().string()).base;
    std::error_code ec;
    fs::path self = fs::weakly_canonical(scene, ec);
    for (const auto& e : fs::directory_iterator(scene.parent_path(), ec))
    {
        if (!e.is_regular_file(ec) || !isSceneExt(e.path()))
            continue;
        std::string stem = e.path().stem().string();
        VersionedStem v = parseVersionedStem(stem);
        if ((stem == base || (v.base == base && v.width)) &&
            fs::weakly_canonical(e.path(), ec) != self)
            out.push_back(e.path());
    }
    std::sort(out.begin(), out.end());
    return out;
}

int64_t unixTime(const fs::path& file)
{
    std::error_code ec;
    auto t = fs::last_write_time(file, ec);
    if (ec)
        return 0;
    return paths::unixSeconds(t);
}

std::vector<Version> list(const fs::path& scene)
{
    std::vector<Version> out;
    auto add = [&](const fs::path& p, Kind k) {
        std::error_code ec;
        out.push_back({p, k, unixTime(p), fs::file_size(p, ec), {}, {}});
    };
    std::string stem = stemFor(scene);
    fs::path autos = autosaveFolder(scene);
    for (const fs::path& p : entries(autos, stem, "autosave"))
        add(p, Kind::Autosave);
    for (const fs::path& p : entries(autos, stem, "emergency"))
        add(p, Kind::Emergency);
    if (!scene.empty())
    {
        for (const fs::path& p : entries(backupFolder(scene), stem, ""))
            add(p, Kind::Backup);
        for (const fs::path& p : increments(scene))
            add(p, Kind::Increment);
    }
    std::stable_sort(out.begin(), out.end(), [](const Version& a, const Version& b) { return a.time > b.time; });
    return out;
}
} // namespace lila::io::history
