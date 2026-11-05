#include "maya/MayaFile.h"

#include "core/Decimal.h"

#include "core/Profile.h"
#include "core/Progress.h"

#include <charconv>
#include <format>
#include <fstream>
#include <optional>
#include <set>
#include <sstream>

namespace lila::maya
{
namespace
{
struct Token
{
    std::string_view text; // without quotes; escapes still in place
    bool quoted = false;
};

std::string unescape(std::string_view s)
{
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i)
    {
        if (s[i] != '\\' || i + 1 == s.size())
        {
            out.push_back(s[i]);
            continue;
        }
        char c = s[++i];
        out.push_back(c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c);
    }
    return out;
}

bool toNumber(std::string_view s, double* out)
{
    if (s == "yes" || s == "on" || s == "true")
        return *out = 1.0, true;
    if (s == "no" || s == "off" || s == "false")
        return *out = 0.0, true;
    if (!s.empty() && s[0] == '+')
        s.remove_prefix(1);
    auto [p, ec] = lila::fromChars(s.data(), s.data() + s.size(), *out);
    if (ec == std::errc::result_out_of_range)
        return *out = (s[0] == '-' ? -1e308 : 1e308), true;
    return ec == std::errc() && p == s.data() + s.size();
}

int toInt(std::string_view s, int fallback = 0)
{
    double d = 0.0;
    return toNumber(s, &d) ? int(d) : fallback;
}

bool isFlag(const Token& t)
{
    return !t.quoted && t.text.size() > 1 && t.text[0] == '-' && ((t.text[1] >= 'a' && t.text[1] <= 'z') || (t.text[1] >= 'A' && t.text[1] <= 'Z'));
}

std::string stripRoot(std::string_view name)
{
    while (!name.empty() && name[0] == ':')
        name.remove_prefix(1);
    return std::string(name);
}

/** Splits MEL into statements of tokens. Comments run to the end of the line; strings may hold anything. */
class Lexer
{
public:
    explicit Lexer(std::string_view text) : mText(text) {}

    size_t offset() const { return mPos; }

    /** False at the end. `line` is the line the statement starts on. */
    bool next(std::vector<Token>& out, int& line)
    {
        out.clear();
        line = 0;
        while (mPos < mText.size())
        {
            char c = mText[mPos];
            if (c == '\n')
            {
                ++mLine;
                ++mPos;
                continue;
            }
            if (c == ' ' || c == '\t' || c == '\r')
            {
                ++mPos;
                continue;
            }
            if (line == 0)
                line = mLine;
            if (c == ';')
            {
                ++mPos;
                if (!out.empty())
                    return true;
                continue;
            }
            if (c == '/' && mPos + 1 < mText.size() && mText[mPos + 1] == '/')
            {
                while (mPos < mText.size() && mText[mPos] != '\n')
                    ++mPos;
                continue;
            }
            if (c == '"')
            {
                size_t start = ++mPos;
                while (mPos < mText.size() && mText[mPos] != '"')
                {
                    if (mText[mPos] == '\\')
                        ++mPos;
                    else if (mText[mPos] == '\n')
                        ++mLine;
                    ++mPos;
                }
                out.push_back({mText.substr(start, std::min(mPos, mText.size()) - start), true});
                ++mPos;
                continue;
            }
            size_t start = mPos;
            while (mPos < mText.size())
            {
                char d = mText[mPos];
                if (d == ' ' || d == '\t' || d == '\r' || d == '\n' || d == ';' || d == '"')
                    break;
                ++mPos;
            }
            out.push_back({mText.substr(start, mPos - start), false});
        }
        return !out.empty();
    }

private:
    std::string_view mText;
    size_t mPos = 0;
    int mLine = 1;
};

/** Flags of setAttr that take a value. */
bool setAttrFlagTakesValue(std::string_view f)
{
    return f == "-s" || f == "-size" || f == "-type" || f == "-typ" || f == "-k" || f == "-keyable" || f == "-l" || f == "-lock" ||
           f == "-cb" || f == "-channelBox" || f == "-ca" || f == "-caching" || f == "-ch" || f == "-capacityHint";
}

void fillValues(AttrValue& v, std::span<const Token> vals, std::vector<std::string>& warnings, int line)
{
    const std::string& type = v.type;
    auto strings = [&](size_t from) {
        for (size_t i = from; i < vals.size(); ++i)
            v.strs.push_back(vals[i].quoted ? unescape(vals[i].text) : std::string(vals[i].text));
    };
    if (type == "string")
    {
        if (!vals.empty())
            v.strs.push_back(unescape(vals[0].text));
        return;
    }
    if (type == "stringArray" || type == "componentList")
    {
        strings(vals.empty() ? 0 : 1);
        return;
    }
    if (type == "polyFaces")
    {
        std::vector<std::string> raw;
        raw.reserve(vals.size());
        for (const Token& t : vals)
            raw.emplace_back(t.text);
        std::string warn;
        v.mesh = parsePolyFaces(raw, &warn);
        if (!warn.empty())
            warnings.push_back(std::format("line {}: {}", line, warn));
        return;
    }
    if (type == "matrix" && !vals.empty() && vals[0].text == "xform")
    {
        v.type = "matrixXform";
        strings(1);
        return;
    }
    size_t from = 0;
    if (type == "Int32Array" || type == "doubleArray" || type == "floatArray" || type == "pointArray" || type == "vectorArray")
        from = 1;
    v.nums.reserve(vals.size());
    for (size_t i = from; i < vals.size(); ++i)
    {
        double d = 0.0;
        if (vals[i].quoted || !toNumber(vals[i].text, &d))
        {
            // Text payloads (dataPolyComponent, attributeAlias, curves) are kept as raw tokens.
            v.nums.clear();
            strings(0);
            return;
        }
        v.nums.push_back(d);
    }
}
} // namespace

double fpsOf(std::string_view u)
{
    static const std::pair<std::string_view, double> kUnits[] = {
        {"game", 15}, {"film", 24}, {"pal", 25}, {"ntsc", 30}, {"show", 48}, {"palf", 50}, {"ntscf", 60},
        {"sec", 1}, {"millisec", 1000}, {"min", 1.0 / 60.0}, {"hour", 1.0 / 3600.0},
    };
    for (const auto& [n, f] : kUnits)
        if (u == n)
            return f;
    if (u.ends_with("fps"))
    {
        double d = 0.0;
        if (toNumber(u.substr(0, u.size() - 3), &d) && d > 0.0)
            return d;
    }
    return 0.0;
}

double cmPerUnit(std::string_view u)
{
    static const std::pair<std::string_view, double> kUnits[] = {
        {"mm", 0.1}, {"millimeter", 0.1}, {"cm", 1.0}, {"centimeter", 1.0}, {"m", 100.0}, {"meter", 100.0},
        {"km", 1e5}, {"kilometer", 1e5}, {"in", 2.54}, {"inch", 2.54}, {"ft", 30.48}, {"foot", 30.48},
        {"yd", 91.44}, {"yard", 91.44}, {"mi", 160934.4}, {"mile", 160934.4},
    };
    for (const auto& [n, f] : kUnits)
        if (u == n)
            return f;
    return 0.0;
}

std::vector<AttrSeg> splitAttr(std::string_view attr)
{
    std::vector<AttrSeg> out;
    while (!attr.empty() && attr[0] == '.')
        attr.remove_prefix(1);
    size_t i = 0;
    while (i <= attr.size())
    {
        size_t j = attr.find('.', i);
        if (j == std::string_view::npos)
            j = attr.size();
        std::string_view seg = attr.substr(i, j - i);
        AttrSeg s;
        size_t b = seg.find('[');
        s.name = std::string(seg.substr(0, b));
        if (b != std::string_view::npos)
        {
            std::string_view inner = seg.substr(b + 1, seg.find(']', b) - b - 1);
            if (inner == "*")
                s.lo = s.hi = -2;
            else if (size_t c = inner.find(':'); c != std::string_view::npos)
            {
                s.lo = toInt(inner.substr(0, c), 0);
                s.hi = toInt(inner.substr(c + 1), s.lo);
            }
            else
                s.lo = s.hi = toInt(inner, 0);
        }
        out.push_back(std::move(s));
        if (j == attr.size())
            break;
        i = j + 1;
    }
    return out;
}

bool parseComponent(std::string_view s, std::string* kind, int* lo, int* hi)
{
    if (auto dot = s.rfind('.'); dot != std::string_view::npos)
        s = s.substr(dot + 1);
    size_t b = s.find('[');
    if (b == std::string_view::npos || s.back() != ']')
        return false;
    *kind = std::string(s.substr(0, b));
    std::string_view inner = s.substr(b + 1, s.size() - b - 2);
    if (inner == "*")
    {
        *lo = 0;
        *hi = -1;
        return true;
    }
    if (size_t c = inner.find(':'); c != std::string_view::npos)
    {
        *lo = toInt(inner.substr(0, c));
        *hi = toInt(inner.substr(c + 1));
    }
    else
        *lo = *hi = toInt(inner);
    return true;
}

std::vector<RefEdit> parseRefEdits(std::span<const std::string> tok, std::string* warning)
{
    // Owner reference node, then groups of (reference node, edit count, edits).
    auto arity = [](int kind) {
        switch (kind)
        {
        case 0: return 3;
        case 1: return 4;
        case 2: return 3;
        case 3: return 2;
        case 5: return 5;
        case 7: return 2;
        case 8: return 2;
        default: return -1;
        }
    };
    std::vector<RefEdit> out;
    size_t i = 1;
    while (i + 1 < tok.size())
    {
        ++i; // the group's reference node
        int count = toInt(tok[i++], 0);
        for (int e = 0; e < count && i < tok.size(); ++e)
        {
            RefEdit ed;
            ed.kind = toInt(tok[i++], -1);
            int n = arity(ed.kind);
            if (n < 0)
            {
                if (warning)
                    *warning = std::format("reference edit of unknown kind {}; the rest were ignored", ed.kind);
                return out;
            }
            for (int k = 0; k < n && i < tok.size(); ++k)
                ed.args.push_back(tok[i++]);
            out.push_back(std::move(ed));
        }
    }
    return out;
}

std::shared_ptr<MeshData> parsePolyFaces(std::span<const std::string> tok, std::string* warning)
{
    auto data = std::make_shared<MeshData>();
    MeshData::Loop* loop = nullptr;
    size_t i = 0;
    auto count = [&]() -> int { return i < tok.size() ? toInt(tok[i++], 0) : 0; };
    auto ints = [&](int n) {
        std::vector<int32_t> v;
        v.reserve(size_t(std::max(n, 0)));
        for (int k = 0; k < n && i < tok.size(); ++k)
            v.push_back(toInt(tok[i++]));
        return v;
    };
    while (i < tok.size())
    {
        const std::string& tag = tok[i++];
        if (tag == "f" || tag == "h")
        {
            if (tag == "f" || data->faces.empty())
                data->faces.emplace_back();
            data->faces.back().loops.push_back({.edges = ints(count())});
            loop = &data->faces.back().loops.back();
        }
        else if (tag == "mu" || tag == "mc")
        {
            int set = count();
            std::vector<int32_t> idx = ints(count());
            if (!loop || set < 0 || set > 1000)
                continue;
            auto& slots = tag == "mu" ? loop->uv : loop->color;
            if (slots.size() <= size_t(set))
                slots.resize(size_t(set) + 1);
            slots[size_t(set)] = std::move(idx);
        }
        else if (tag == "fc" || tag == "mf")
        {
            ints(count());
        }
        else
        {
            if (warning)
                *warning = "polyFaces: unknown entry '" + tag + "', rest of the face list ignored";
            break;
        }
    }
    return data;
}

File readAscii(std::string_view text)
{
    LILA_ZONE();

    File f;
    Lexer lex(text);
    std::vector<Token> t;
    int line = 0;
    int current = -1; // node that setAttr / addAttr apply to
    std::set<std::string> unloaded; // reference nodes of the references saved unloaded
    // Reference edits parse one-line snippets; only whole files report.
    std::optional<progress::Scope> scope;
    if (text.size() > (1u << 16))
        scope.emplace("Parsing", double(text.size()));
    size_t statements = 0;
    while (lex.next(t, line))
    {
        if (scope && (++statements & 1023) == 0)
        {
            scope->set(double(lex.offset()));
            scope->checkCancel();
        }
        const std::string_view cmd = t[0].text;
        if (cmd == "createNode")
        {
            Node n;
            for (size_t i = 1; i < t.size(); ++i)
            {
                std::string_view a = t[i].text;
                if (isFlag(t[i]))
                {
                    if ((a == "-n" || a == "-name") && i + 1 < t.size())
                        n.name = std::string(t[++i].text);
                    else if ((a == "-p" || a == "-parent") && i + 1 < t.size())
                        n.parent = std::string(t[++i].text);
                    else if (a == "-s" || a == "-shared")
                        n.shared = true;
                }
                else if (n.type.empty())
                    n.type = std::string(a);
            }
            if (n.type.empty())
            {
                f.warnings.push_back(std::format("line {}: createNode without a type", line));
                current = -1;
                continue;
            }
            if (n.name.empty())
                n.name = n.type + "1";
            f.nodes.push_back(std::move(n));
            current = int(f.nodes.size()) - 1;
        }
        else if (cmd == "setAttr")
        {
            if (current < 0)
                continue;
            AttrValue v;
            size_t i = 1;
            for (; i < t.size(); ++i)
            {
                if (!isFlag(t[i]))
                    break;
                std::string_view fl = t[i].text;
                if (setAttrFlagTakesValue(fl) && i + 1 < t.size())
                {
                    std::string_view val = t[++i].text;
                    if (fl == "-s" || fl == "-size")
                        v.size = toInt(val, -1);
                    else if (fl == "-l" || fl == "-lock")
                        v.lock = val == "on" || val == "yes" || val == "true" || val == "1" ? 1 : 0;
                    else if (fl == "-type" || fl == "-typ")
                        v.type = std::string(val);
                }
            }
            if (i >= t.size())
                continue;
            v.attr = std::string(t[i].text);
            while (!v.attr.empty() && v.attr[0] == '.')
                v.attr.erase(v.attr.begin());
            ++i;
            // -type may also follow the attribute name.
            if (i + 1 < t.size() && !t[i].quoted && (t[i].text == "-type" || t[i].text == "-typ"))
            {
                v.type = std::string(t[i + 1].text);
                i += 2;
            }
            // Maya splits long face lists into several setAttrs and types only the first.
            if (v.type.empty() && i < t.size() && (t[i].text == "f" || t[i].text == "h") && (v.attr.starts_with("fc") || v.attr.starts_with("face")))
                v.type = "polyFaces";
            fillValues(v, std::span<const Token>(t).subspan(i), f.warnings, line);
            if (v.type == "dataReferenceEdits")
            {
                std::string warn;
                f.nodes[size_t(current)].refEdits = parseRefEdits(v.strs, &warn);
                if (!warn.empty())
                    f.warnings.push_back(std::format("line {}: {}", line, warn));
            }
            f.nodes[size_t(current)].attrs.push_back(std::move(v));
        }
        else if (cmd == "connectAttr")
        {
            Connection c;
            std::vector<std::string_view> args;
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (isFlag(t[i]))
                {
                    if (t[i].text == "-na" || t[i].text == "-nextAvailable")
                        c.nextAvailable = true;
                    continue;
                }
                args.push_back(t[i].text);
            }
            if (args.size() >= 2)
            {
                c.src = stripRoot(args[0]);
                c.dst = stripRoot(args[1]);
                f.connections.push_back(std::move(c));
            }
        }
        else if (cmd == "select")
        {
            std::string name;
            bool ne = false;
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (isFlag(t[i]))
                    ne = ne || t[i].text == "-ne" || t[i].text == "-noExpand";
                else
                    name = stripRoot(t[i].text);
            }
            current = -1;
            if (!ne || name.empty())
                continue;
            for (size_t k = f.nodes.size(); k-- > 0;)
                if (f.nodes[k].name == name)
                {
                    current = int(k);
                    break;
                }
            if (current < 0)
            {
                f.nodes.push_back({.name = name, .existing = true});
                current = int(f.nodes.size()) - 1;
            }
        }
        else if (cmd == "addAttr")
        {
            if (current < 0)
                continue;
            AddedAttr a;
            for (size_t i = 1; i + 1 < t.size(); ++i)
            {
                std::string_view fl = t[i].text;
                if (fl == "-ln" || fl == "-longName")
                    a.longName = std::string(t[++i].text);
                else if (fl == "-sn" || fl == "-shortName")
                    a.shortName = std::string(t[++i].text);
                else if (fl == "-at" || fl == "-attributeType" || fl == "-dt" || fl == "-dataType")
                    a.type = std::string(t[++i].text);
            }
            if (a.shortName.empty())
                a.shortName = a.longName;
            f.nodes[size_t(current)].added.push_back(std::move(a));
        }
        else if (cmd == "currentUnit")
        {
            for (size_t i = 1; i + 1 < t.size(); ++i)
            {
                std::string_view fl = t[i].text;
                if (fl == "-l" || fl == "-linear")
                    f.linearUnit = std::string(t[++i].text);
                else if (fl == "-a" || fl == "-angle")
                    f.angleUnit = std::string(t[++i].text);
                else if (fl == "-t" || fl == "-time")
                    f.timeUnit = std::string(t[++i].text);
            }
        }
        else if (cmd == "fileInfo")
        {
            if (t.size() >= 3)
                f.fileInfo[unescape(t[1].text)] = unescape(t[2].text);
        }
        else if (cmd == "requires")
        {
            for (size_t i = 1; i < t.size(); ++i)
                if (t[i].quoted)
                {
                    f.requires_.push_back(unescape(t[i].text));
                    break;
                }
            if (t.size() >= 3 && t[1].quoted && t[1].text == "maya")
                f.version = unescape(t[2].text);
        }
        else if (cmd == "file")
        {
            bool ref = false, depthOne = false, deferred = false;
            Reference r;
            for (size_t i = 1; i < t.size(); ++i)
            {
                std::string_view a = t[i].text;
                if (!t[i].quoted && (a == "-r" || a == "-reference"))
                    ref = true;
                else if (!t[i].quoted && a == "-dr" && i + 1 < t.size())
                    deferred = t[++i].text != "0";
                else if (!t[i].quoted && a == "-rdi" && i + 1 < t.size())
                    depthOne = t[++i].text == "1";
                else if (!t[i].quoted && (a == "-ns" || a == "-namespace") && i + 1 < t.size())
                    r.ns = unescape(t[++i].text);
                else if (!t[i].quoted && (a == "-typ" || a == "-type") && i + 1 < t.size())
                    r.type = unescape(t[++i].text);
                else if (!t[i].quoted && a == "-rfn" && i + 1 < t.size())
                    r.refNode = unescape(t[++i].text);
                else if (!t[i].quoted && (a == "-op" || a == "-shd" || a == "-gl"))
                {
                    if (a != "-gl" && i + 1 < t.size())
                        ++i;
                }
                else if (t[i].quoted)
                    r.path = unescape(a);
            }
            // The -r line always defers; the scene's load state is on its -rdi 1 line.
            if (depthOne && deferred && !r.refNode.empty())
                unloaded.insert(r.refNode);
            else if (ref && !r.path.empty())
                f.references.push_back(std::move(r));
        }
        else if (cmd == "parent")
        {
            ParentEdit p;
            std::vector<std::string_view> args;
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (isFlag(t[i]))
                {
                    p.shape = p.shape || t[i].text == "-s" || t[i].text == "-shape";
                    p.add = p.add || t[i].text == "-add" || t[i].text == "-addObject";
                }
                else
                    args.push_back(t[i].text);
            }
            if (args.size() >= 2)
            {
                p.child = stripRoot(args[0]);
                p.parent = stripRoot(args[1]);
                f.parents.push_back(std::move(p));
            }
        }
        else if (cmd == "relationship")
        {
            Relationship r;
            for (size_t i = 1; i < t.size(); ++i)
            {
                if (isFlag(t[i]))
                    continue;
                if (r.type.empty())
                    r.type = std::string(t[i].text);
                else if (r.linker.empty())
                    r.linker = stripRoot(t[i].text);
                else
                    r.plugs.push_back(stripRoot(t[i].text));
            }
            if (r.plugs.size() >= 2)
                f.relationships.push_back(std::move(r));
        }
        // rename -uid, lockNode, dataStructure and the rest carry nothing a conversion needs.
    }
    for (Reference& r : f.references)
        r.loaded = !unloaded.contains(r.refNode);
    if (f.version.empty())
        if (auto it = f.fileInfo.find("version"); it != f.fileInfo.end())
            f.version = it->second;
    return f;
}

bool looksBinary(std::span<const uint8_t> b)
{
    if (b.size() < 12)
        return false;
    std::string_view tag(reinterpret_cast<const char*>(b.data()), 4);
    return tag == "FOR4" || tag == "FOR8";
}

File readFile(const std::filesystem::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("can't open " + path.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    std::string bytes = std::move(ss).str();
    std::span<const uint8_t> view(reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    if (looksBinary(view))
        return readBinary(view);
    if (bytes.find("createNode") == std::string::npos && bytes.find("requires") == std::string::npos)
        throw std::runtime_error(path.filename().string() + " is not a Maya ASCII or binary file");
    return readAscii(bytes);
}
} // namespace lila::maya
