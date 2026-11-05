#include "curve/Sketch.h"

#include "core/Decimal.h"

#include "core/Profile.h"
#include "curve/Tessellate.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>
#include <stdexcept>

namespace lila::curve::sketch
{
namespace
{
constexpr double kPi = glm::pi<double>();

json vec(const D3& v) { return json::array({v.x, v.y, v.z}); }

D3 vecFrom(const json& j)
{
    if (!j.is_array() || j.size() != 3)
        throw std::invalid_argument("expected [x, y, z]");
    return D3(j[0].get<double>(), j[1].get<double>(), j[2].get<double>());
}

double cross2(const D2& a, const D2& b) { return a.x * b.y - a.y * b.x; }

/** Unit vector, or the vector itself when it is too short to have a direction (no NaN). */
D2 unit(const D2& d)
{
    const double l = glm::length(d);
    return l > 1e-12 ? d / l : d;
}

/** The signed angle from a to b, radians; 0 for a zero vector. */
double angleBetween(const D2& a, const D2& b) { return std::atan2(a.x * b.y - a.y * b.x, glm::dot(a, b)); }

/**
 * How far an angle is from the nearest half turn, so parallel either way counts: an angle residual has a slope
 * everywhere, where the cross product of unit vectors goes flat at a right angle and stalls the solve there.
 */
double fold(double a) { return a - kPi * std::round(a / kPi); }

double wrapAngle(double a)
{
    a = std::fmod(a + kPi, 2.0 * kPi);
    if (a < 0.0)
        a += 2.0 * kPi;
    return a - kPi;
}

const char* const kTypeNames[] = {"coincident", "horizontal",   "vertical", "parallel",           "perpendicular",    "tangent", "equal",
                                  "concentric", "midpoint",     "symmetric", "fixed",             "pointOnCurve",     "collinear",
                                  "distance",   "horizontalDistance", "verticalDistance", "length", "radius", "diameter", "angle"};
static_assert(std::size(kTypeNames) == size_t(Type::Count));
} // namespace

// Refs

std::string refName(const Ref& r)
{
    const std::string s = std::to_string(r.spline);
    const std::string i = r.index < 0 ? std::string("end") : std::to_string(r.index);
    switch (r.kind)
    {
    case RefKind::Point: return "p" + s + "." + i;
    case RefKind::Line: return "l" + s + "." + i;
    case RefKind::Curve: return "c" + s;
    case RefKind::Center: return "o" + s;
    case RefKind::HandleL: return "hl" + s + "." + i;
    case RefKind::HandleR: return "hr" + s + "." + i;
    }
    return {};
}

bool parseRef(std::string_view t, Ref* out)
{
    Ref r;
    size_t k = 0;
    if (t.starts_with("hl"))
        r.kind = RefKind::HandleL, k = 2;
    else if (t.starts_with("hr"))
        r.kind = RefKind::HandleR, k = 2;
    else if (!t.empty() && t[0] == 'p')
        r.kind = RefKind::Point, k = 1;
    else if (!t.empty() && t[0] == 'l')
        r.kind = RefKind::Line, k = 1;
    else if (!t.empty() && t[0] == 'c')
        r.kind = RefKind::Curve, k = 1;
    else if (!t.empty() && t[0] == 'o')
        r.kind = RefKind::Center, k = 1;
    else
        return false;
    t.remove_prefix(k);
    const size_t dot = t.find('.');
    std::string_view a = t.substr(0, dot);
    auto [pa, ea] = std::from_chars(a.data(), a.data() + a.size(), r.spline);
    if (ea != std::errc() || pa != a.data() + a.size() || r.spline < 0)
        return false;
    const bool wantsIndex = r.kind != RefKind::Curve && r.kind != RefKind::Center;
    if (!wantsIndex)
    {
        if (dot != std::string_view::npos)
            return false;
        r.index = 0;
    }
    else
    {
        if (dot == std::string_view::npos)
            return false;
        std::string_view b = t.substr(dot + 1);
        if (b == "end")
            r.index = -1;
        else
        {
            auto [pb, eb] = std::from_chars(b.data(), b.data() + b.size(), r.index);
            if (eb != std::errc() || pb != b.data() + b.size() || r.index < 0)
                return false;
        }
    }
    *out = r;
    return true;
}

const char* typeName(Type t) { return size_t(t) < std::size(kTypeNames) ? kTypeNames[size_t(t)] : "?"; }

bool parseType(std::string_view name, Type* out)
{
    for (size_t k = 0; k < std::size(kTypeNames); ++k)
        if (name == kTypeNames[k])
        {
            *out = Type(k);
            return true;
        }
    return false;
}

// Storage

json Constraint::toJson() const
{
    json r = json::array();
    for (const Ref& ref : refs)
        r.push_back(refName(ref));
    json j = {{"id", id}, {"type", typeName(type)}, {"refs", r}};
    if (isDimension(type))
    {
        j["value"] = value;
        j["name"] = name;
        if (!expr.empty())
            j["expr"] = expr;
    }
    if (sign != 1.0)
        j["sign"] = sign;
    if (!anchors.empty())
    {
        json a = json::array();
        for (const D3& p : anchors)
            a.push_back(vec(p));
        j["anchors"] = a;
    }
    if (endA >= 0 || endB >= 0)
        j["ends"] = {endA, endB};
    return j;
}

Constraint Constraint::fromJson(const json& j)
{
    Constraint c;
    c.id = j.value("id", 0u);
    if (!parseType(j.value("type", std::string()), &c.type))
        throw std::invalid_argument("unknown constraint type '" + j.value("type", std::string()) + "'");
    for (const json& r : j.value("refs", json::array()))
    {
        Ref ref;
        if (!r.is_string() || !parseRef(r.get<std::string>(), &ref))
            throw std::invalid_argument("bad constraint ref " + r.dump());
        c.refs.push_back(ref);
    }
    c.value = j.value("value", 0.0);
    c.expr = j.value("expr", std::string());
    c.name = j.value("name", std::string());
    c.sign = j.value("sign", 1.0);
    for (const json& a : j.value("anchors", json::array()))
        c.anchors.push_back(vecFrom(a));
    if (j.contains("ends") && j["ends"].is_array() && j["ends"].size() == 2)
    {
        c.endA = j["ends"][0].get<int>();
        c.endB = j["ends"][1].get<int>();
    }
    return c;
}

Plane planeFrom(const D3& origin, const D3& normal)
{
    D3 n = glm::length(normal) > 1e-12 ? glm::normalize(normal) : D3(0, 1, 0);
    Plane p;
    p.origin = origin;
    // Horizontal is world X unless the plane faces along X; vertical completes a right-handed frame with the normal.
    D3 ref = std::abs(n.x) > 0.9 ? D3(0, 0, n.x > 0 ? -1 : 1) : D3(1, 0, 0);
    D3 u = ref - n * glm::dot(ref, n);
    p.u = glm::normalize(u);
    p.v = glm::normalize(glm::cross(n, p.u));
    return p;
}

Plane fitPlane(const Splines& s, const D3& fallbackNormal)
{
    std::vector<D3> pts;
    for (const Spline& sp : s)
        for (const D3& p : sp.points)
            pts.push_back(p);
    D3 c(0.0);
    for (const D3& p : pts)
        c += p;
    if (!pts.empty())
        c /= double(pts.size());
    D3 n = pts.size() >= 3 ? planeNormal(pts, 1e-6) : D3(0.0);
    if (glm::length(n) < 0.5)
    {
        // Collinear points: any plane through the line works, so take the one closest to the fallback.
        n = fallbackNormal;
        if (pts.size() >= 2)
        {
            D3 d = pts.back() - pts.front();
            for (const D3& p : pts)
                if (glm::length(p - pts.front()) > glm::length(d))
                    d = p - pts.front();
            if (glm::length(d) > 1e-12)
            {
                d = glm::normalize(d);
                D3 m = n - d * glm::dot(n, d);
                if (glm::length(m) > 1e-6)
                    n = m;
            }
        }
    }
    else if (glm::dot(n, fallbackNormal) < 0.0)
        n = -n;
    return planeFrom(c, n);
}

const Constraint* Sketch::find(uint32_t id) const
{
    for (const Constraint& c : constraints)
        if (c.id == id)
            return &c;
    return nullptr;
}

uint32_t Sketch::add(Constraint c)
{
    c.id = nextId++;
    if (isDimension(c.type) && c.name.empty())
    {
        std::set<std::string> taken;
        for (const Constraint& o : constraints)
            taken.insert(o.name);
        do
            c.name = "d" + std::to_string(nextName++);
        while (taken.contains(c.name));
    }
    constraints.push_back(std::move(c));
    return constraints.back().id;
}

bool Sketch::remove(uint32_t id)
{
    return std::erase_if(constraints, [id](const Constraint& c) { return c.id == id; }) > 0;
}

json Sketch::toJson() const
{
    json cs = json::array();
    for (const Constraint& c : constraints)
        cs.push_back(c.toJson());
    return {{"plane", {{"origin", vec(plane.origin)}, {"u", vec(plane.u)}, {"v", vec(plane.v)}}},
            {"constraints", cs},
            {"nextId", nextId},
            {"nextName", nextName}};
}

std::string Sketch::toText() const { return toJson().dump(); }

Sketch Sketch::fromJson(const json& j)
{
    Sketch k;
    if (j.contains("plane"))
    {
        const json& p = j["plane"];
        k.plane.origin = vecFrom(p.at("origin"));
        D3 u = vecFrom(p.at("u")), v = vecFrom(p.at("v"));
        if (glm::length(u) < 1e-9 || glm::length(v) < 1e-9 || glm::length(glm::cross(u, v)) < 1e-9)
            throw std::invalid_argument("degenerate sketch plane");
        k.plane.u = glm::normalize(u);
        k.plane.v = glm::normalize(v - k.plane.u * glm::dot(v, k.plane.u));
    }
    for (const json& c : j.value("constraints", json::array()))
        k.constraints.push_back(Constraint::fromJson(c));
    uint32_t maxId = 0;
    for (const Constraint& c : k.constraints)
        maxId = std::max(maxId, c.id);
    k.nextId = std::max(j.value("nextId", 1u), maxId + 1);
    k.nextName = std::max(1, j.value("nextName", 1));
    return k;
}

Sketch Sketch::fromText(std::string_view text)
{
    if (text.find_first_not_of(" \t\r\n") == std::string_view::npos)
        return {};
    json j = json::parse(text, nullptr, false);
    if (j.is_discarded() || !j.is_object())
        throw std::invalid_argument("sketch data is not a JSON object");
    return fromJson(j);
}

// Geometry

CircleFit circleOf(const Spline& s, const Plane& pl)
{
    CircleFit out;
    if (s.kind != Kind::Nurbs || s.degree != 2 || s.weights.empty() || !s.rational() || !isValid(s))
        return out;
    const int spans = spanCount(s);
    if (spans < 1)
        return out;
    // Sample densely: three points give the circle, the rest must lie on it and in a plane parallel to the sketch.
    const int n = std::max(12, spans * 4);
    std::vector<D2> q;
    std::vector<double> h;
    for (int k = 0; k <= n; ++k)
    {
        D3 p = pointAt(s, double(spans) * k / n);
        q.push_back(pl.to2d(p));
        h.push_back(pl.height(p));
    }
    const D2 a = q[0], b = q[size_t(n / 3)], c = q[size_t(2 * n / 3)];
    const double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));
    if (std::abs(d) < 1e-12)
        return out;
    const double a2 = glm::dot(a, a), b2 = glm::dot(b, b), c2 = glm::dot(c, c);
    const D2 center((a2 * (b.y - c.y) + b2 * (c.y - a.y) + c2 * (a.y - b.y)) / d, (a2 * (c.x - b.x) + b2 * (a.x - c.x) + c2 * (b.x - a.x)) / d);
    const double r = glm::distance(a, center);
    if (r < 1e-9)
        return out;
    const double tol = std::max(1e-7, r * 1e-6);
    for (size_t k = 0; k < q.size(); ++k)
        if (std::abs(glm::distance(q[k], center) - r) > tol || std::abs(h[k] - h[0]) > tol)
            return out;
    out.center = center;
    out.radius = r;
    out.height = h[0];
    // Turning direction from the signed area swept around the centre.
    double turn = 0.0;
    for (size_t k = 1; k < q.size(); ++k)
        turn += wrapAngle(std::atan2(q[k].y - center.y, q[k].x - center.x) - std::atan2(q[k - 1].y - center.y, q[k - 1].x - center.x));
    out.start = std::atan2(a.y - center.y, a.x - center.x);
    if (s.closed)
    {
        out.shape = Shape::Circle;
        out.sweep = turn > 0.0 ? 2.0 * kPi : -2.0 * kPi;
    }
    else
    {
        out.shape = Shape::Arc;
        out.sweep = turn;
        if (std::abs(turn) < 1e-9 || std::abs(turn) > 2.0 * kPi - 1e-9)
            out.shape = Shape::Points;
    }
    return out;
}

namespace
{
size_t lastIndex(const Spline& s) { return s.size() ? s.size() - 1 : 0; }

size_t pointIndex(const Spline& s, int index) { return index < 0 ? lastIndex(s) : size_t(index); }

size_t lineCount(const Spline& s) { return s.size() < 2 ? 0 : (s.closed ? s.size() : s.size() - 1); }
} // namespace

Ref normalized(const Splines& s, const Ref& r)
{
    Ref o = r;
    if (r.spline >= 0 && size_t(r.spline) < s.size() && r.index < 0)
        o.index = int(lastIndex(s[size_t(r.spline)]));
    return o;
}

bool refValid(const Splines& s, const Plane& pl, const Ref& r, std::string* why)
{
    auto fail = [&](std::string w) {
        if (why)
            *why = refName(r) + ": " + w;
        return false;
    };
    if (r.spline < 0 || size_t(r.spline) >= s.size())
        return fail("no such spline");
    const Spline& sp = s[size_t(r.spline)];
    const CircleFit cf = circleOf(sp, pl);
    switch (r.kind)
    {
    case RefKind::Point:
        if (cf.shape == Shape::Circle)
            return fail("a circle has no points to constrain; use its centre (o) or the curve (c)");
        if (cf.shape == Shape::Arc && r.index > 0 && size_t(r.index) != lastIndex(sp))
            return fail("an arc's points are its ends: index 0 or end");
        if (pointIndex(sp, r.index) >= sp.size())
            return fail("no such point");
        return true;
    case RefKind::Line:
        if (cf.shape != Shape::Points)
            return fail("a circle or arc has no lines; use the curve (c)");
        if (r.index < 0 || size_t(r.index) >= lineCount(sp))
            return fail("no such line");
        return true;
    case RefKind::Curve: return true;
    case RefKind::Center:
        if (cf.shape == Shape::Points)
            return fail("not a circle or arc, so it has no centre");
        return true;
    case RefKind::HandleL:
    case RefKind::HandleR:
        if (sp.kind != Kind::Bezier || sp.handleL.size() != sp.size())
            return fail("only Bezier anchors have handles");
        if (pointIndex(sp, r.index) >= sp.size())
            return fail("no such point");
        return true;
    }
    return fail("unknown ref");
}

// The solver's model of the splines: variables, their weights, and the equations constraints become.

namespace
{
using X = std::vector<double>;

struct SplineModel
{
    Shape shape = Shape::Points;
    /** Points: first variable (point i at base + 2i). Circle: cx, cy, r. Arc: cx, cy, r, start angle, end angle. */
    int base = -1;
    /** Bezier handles that are variables: offset from the anchor, two variables at the index, else -1. */
    std::vector<int> hl, hr;
    std::vector<D2> chl, chr; // offsets of handles that aren't variables
    std::vector<double> hhl, hhr; // handle offsets off the plane
    std::vector<double> h;        // point heights off the plane
    CircleFit fit;
};

struct Eq
{
    uint32_t cid = 0;
    std::vector<int> vars;
    std::function<double(const X&)> f;
};

struct Model
{
    const Splines* s = nullptr;
    Plane pl;
    double scale = 1.0;
    std::vector<SplineModel> sp;
    X x;
    std::vector<double> w; // variable weights for the minimum-norm step
    std::vector<double> step; // finite difference step per variable
    std::vector<Eq> eqs;
    std::vector<uint32_t> invalid;
    std::vector<std::string> warnings;

    int newVar(double value, double weight, double h)
    {
        x.push_back(value);
        w.push_back(weight);
        step.push_back(h);
        return int(x.size()) - 1;
    }

    const SplineModel& of(const Ref& r) const { return sp[size_t(r.spline)]; }
    size_t index(const Ref& r) const { return pointIndex((*s)[size_t(r.spline)], r.index); }

    D2 point(const X& v, const Ref& r) const
    {
        const SplineModel& m = of(r);
        if (r.kind == RefKind::Center)
            return {v[size_t(m.base)], v[size_t(m.base) + 1]};
        if (m.shape == Shape::Arc)
        {
            const D2 c(v[size_t(m.base)], v[size_t(m.base) + 1]);
            const double rad = v[size_t(m.base) + 2];
            const double a = v[size_t(m.base) + (index(r) == 0 ? 3 : 4)];
            return c + rad * D2(std::cos(a), std::sin(a));
        }
        const size_t i = index(r);
        const D2 p(v[size_t(m.base) + 2 * i], v[size_t(m.base) + 2 * i + 1]);
        if (r.kind == RefKind::HandleL)
            return p + (m.hl[i] >= 0 ? D2(v[size_t(m.hl[i])], v[size_t(m.hl[i]) + 1]) : m.chl[i]);
        if (r.kind == RefKind::HandleR)
            return p + (m.hr[i] >= 0 ? D2(v[size_t(m.hr[i])], v[size_t(m.hr[i]) + 1]) : m.chr[i]);
        return p;
    }

    Ref pointRef(int spline, size_t i) const { return {RefKind::Point, spline, int(i)}; }

    /** A line's ends. */
    std::pair<D2, D2> line(const X& v, const Ref& r) const
    {
        const Spline& s0 = (*s)[size_t(r.spline)];
        return {point(v, pointRef(r.spline, size_t(r.index))), point(v, pointRef(r.spline, (size_t(r.index) + 1) % s0.size()))};
    }

    D2 center(const X& v, const Ref& r) const { return {v[size_t(of(r).base)], v[size_t(of(r).base) + 1]}; }
    double radius(const X& v, const Ref& r) const { return v[size_t(of(r).base) + 2]; }

    void varsOf(const Ref& r, std::vector<int>& out) const
    {
        const SplineModel& m = of(r);
        auto two = [&](int b) {
            if (b >= 0)
            {
                out.push_back(b);
                out.push_back(b + 1);
            }
        };
        switch (r.kind)
        {
        case RefKind::Center: two(m.base); break;
        case RefKind::Curve:
            if (m.shape == Shape::Points)
            {
                for (size_t i = 0; i < m.h.size(); ++i)
                {
                    two(m.base + int(2 * i));
                    two(m.hl[i]);
                    two(m.hr[i]);
                }
            }
            else
                for (int k = 0; k < (m.shape == Shape::Arc ? 5 : 3); ++k)
                    out.push_back(m.base + k);
            break;
        case RefKind::Line:
        {
            const Spline& s0 = (*s)[size_t(r.spline)];
            varsOf(pointRef(r.spline, size_t(r.index)), out);
            varsOf(pointRef(r.spline, (size_t(r.index) + 1) % s0.size()), out);
            break;
        }
        default:
            if (m.shape == Shape::Arc)
                for (int k = 0; k < 5; ++k)
                    out.push_back(m.base + k);
            else
            {
                const size_t i = index(r);
                two(m.base + int(2 * i));
                if (r.kind == RefKind::HandleL)
                    two(m.hl[i]);
                if (r.kind == RefKind::HandleR)
                    two(m.hr[i]);
            }
        }
    }

    /** The spline as the variables put it, object space. */
    Spline build(const X& v, size_t si) const
    {
        const Spline& src = (*s)[si];
        const SplineModel& m = sp[si];
        const D3 n = pl.normal();
        if (m.shape == Shape::Circle || m.shape == Shape::Arc)
        {
            const D2 c(v[size_t(m.base)], v[size_t(m.base) + 1]);
            const double r = std::max(std::abs(v[size_t(m.base) + 2]), 1e-9);
            const D3 c3 = pl.to3d(c, m.fit.height);
            if (m.shape == Shape::Circle)
                return circle(c3, m.fit.sweep >= 0.0 ? n : -n, r);
            const double a0 = v[size_t(m.base) + 3];
            double sweep = v[size_t(m.base) + 4] - a0;
            // Keep the arc turning the way it did, less than a full turn.
            if (m.fit.sweep > 0.0)
                while (sweep <= 1e-9)
                    sweep += 2.0 * kPi;
            else
                while (sweep >= -1e-9)
                    sweep -= 2.0 * kPi;
            sweep = std::clamp(sweep, -2.0 * kPi + 1e-6, 2.0 * kPi - 1e-6);
            const D3 start = pl.to3d(c + r * D2(std::cos(a0), std::sin(a0)), m.fit.height);
            return arc(c3, sweep > 0.0 ? n : -n, start, std::abs(sweep));
        }
        Spline out = src;
        for (size_t i = 0; i < src.size(); ++i)
            out.points[i] = pl.to3d(D2(v[size_t(m.base) + 2 * i], v[size_t(m.base) + 2 * i + 1]), m.h[i]);
        if (src.kind == Kind::Bezier && src.handleL.size() == src.size())
        {
            std::vector<std::pair<size_t, int>> edited;
            for (size_t i = 0; i < src.size(); ++i)
            {
                const D2 ol = m.hl[i] >= 0 ? D2(v[size_t(m.hl[i])], v[size_t(m.hl[i]) + 1]) : m.chl[i];
                const D2 orr = m.hr[i] >= 0 ? D2(v[size_t(m.hr[i])], v[size_t(m.hr[i]) + 1]) : m.chr[i];
                out.handleL[i] = out.points[i] + pl.u * ol.x + pl.v * ol.y + n * m.hhl[i];
                out.handleR[i] = out.points[i] + pl.u * orr.x + pl.v * orr.y + n * m.hhr[i];
                if (m.hl[i] >= 0)
                    edited.push_back({i, 0});
                if (m.hr[i] >= 0)
                    edited.push_back({i, 1});
            }
            // updateHandles turns an edited Auto or Vector handle into Aligned or Free; handles in a constraint already are.
            updateHandles(out, edited);
        }
        return out;
    }
};

/** Characteristic length of the splines, for scaling unitless residuals and tolerances. */
double sketchScale(const Splines& s)
{
    glm::dvec3 lo(1e300), hi(-1e300);
    bool any = false;
    for (const Spline& sp : s)
        for (const D3& p : sp.points)
        {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
            any = true;
        }
    return any ? std::max(1.0, glm::length(hi - lo)) : 1.0;
}

/** Which Bezier handles constraints touch: they become variables. */
std::set<std::pair<int, size_t>> handleUses(const Splines& s, const Sketch& k, bool left)
{
    std::set<std::pair<int, size_t>> out;
    auto add = [&](int si, int idx) {
        if (si >= 0 && size_t(si) < s.size() && s[size_t(si)].kind == Kind::Bezier && s[size_t(si)].size())
            out.insert({si, pointIndex(s[size_t(si)], idx)});
    };
    for (const Constraint& c : k.constraints)
    {
        for (const Ref& r : c.refs)
            if ((left && r.kind == RefKind::HandleL) || (!left && r.kind == RefKind::HandleR))
                add(r.spline, r.index);
        if (c.type != Type::Tangent)
            continue;
        if (c.refs.size() == 1 && c.refs[0].kind == RefKind::Point)
            add(c.refs[0].spline, c.refs[0].index);
        for (size_t k2 = 0; k2 < c.refs.size() && k2 < 2; ++k2)
        {
            const Ref& r = c.refs[k2];
            const int end = k2 == 0 ? c.endA : c.endB;
            if (end < 0 || r.spline < 0 || size_t(r.spline) >= s.size())
                continue;
            const Spline& sp = s[size_t(r.spline)];
            if (sp.kind != Kind::Bezier || sp.size() < 2)
                continue;
            // A segment's direction at its start is the start anchor's outgoing handle, at its end the end anchor's incoming one.
            if (r.kind == RefKind::Line)
            {
                const size_t i = size_t(r.index), j = (i + 1) % sp.size();
                if (end == 0 && !left)
                    out.insert({r.spline, i});
                if (end == 1 && left)
                    out.insert({r.spline, j});
            }
            else if (r.kind == RefKind::Curve)
            {
                if (end == 0 && !left)
                    out.insert({r.spline, 0});
                if (end == 1 && left)
                    out.insert({r.spline, sp.size() - 1});
            }
        }
    }
    return out;
}

Model buildModel(const Splines& s, const Sketch& k)
{
    Model m;
    m.s = &s;
    m.pl = k.plane;
    m.scale = sketchScale(s);
    const double h = 1e-7 * m.scale;
    const auto useL = handleUses(s, k, true), useR = handleUses(s, k, false);
    const D3 n = m.pl.normal();
    m.sp.resize(s.size());
    for (size_t si = 0; si < s.size(); ++si)
    {
        const Spline& sp = s[si];
        SplineModel& sm = m.sp[si];
        sm.fit = circleOf(sp, m.pl);
        sm.shape = sm.fit.shape;
        if (sm.shape != Shape::Points)
        {
            const double r = std::max(sm.fit.radius, 1e-9);
            sm.base = m.newVar(sm.fit.center.x, 1.0, h);
            m.newVar(sm.fit.center.y, 1.0, h);
            m.newVar(sm.fit.radius, 1.0, h);
            if (sm.shape == Shape::Arc)
            {
                // Turning an end by one radian moves it a radius, so angles weigh as that much movement.
                m.newVar(sm.fit.start, 1.0 / (r * r), 1e-7);
                m.newVar(sm.fit.start + sm.fit.sweep, 1.0 / (r * r), 1e-7);
            }
            continue;
        }
        sm.h.resize(sp.size());
        sm.hl.assign(sp.size(), -1);
        sm.hr.assign(sp.size(), -1);
        sm.chl.assign(sp.size(), D2(0.0));
        sm.chr.assign(sp.size(), D2(0.0));
        sm.hhl.assign(sp.size(), 0.0);
        sm.hhr.assign(sp.size(), 0.0);
        for (size_t i = 0; i < sp.size(); ++i)
        {
            const D2 p = m.pl.to2d(sp.points[i]);
            sm.h[i] = m.pl.height(sp.points[i]);
            const int b = m.newVar(p.x, 1.0, h);
            m.newVar(p.y, 1.0, h);
            if (i == 0)
                sm.base = b;
        }
        if (sp.kind == Kind::Bezier && sp.handleL.size() == sp.size())
            for (size_t i = 0; i < sp.size(); ++i)
            {
                const D3 dl = sp.handleL[i] - sp.points[i], dr = sp.handleR[i] - sp.points[i];
                sm.chl[i] = D2(glm::dot(dl, m.pl.u), glm::dot(dl, m.pl.v));
                sm.chr[i] = D2(glm::dot(dr, m.pl.u), glm::dot(dr, m.pl.v));
                sm.hhl[i] = glm::dot(dl, n);
                sm.hhr[i] = glm::dot(dr, n);
                if (useL.contains({int(si), i}))
                {
                    sm.hl[i] = m.newVar(sm.chl[i].x, 1.0, h);
                    m.newVar(sm.chl[i].y, 1.0, h);
                }
                if (useR.contains({int(si), i}))
                {
                    sm.hr[i] = m.newVar(sm.chr[i].x, 1.0, h);
                    m.newVar(sm.chr[i].y, 1.0, h);
                }
            }
    }
    return m;
}

/** Direction a curve leaves (end 0) or arrives at (end 1) one of its ends, for tangency. */
D2 endDirection(const Model& m, const X& v, const Ref& r, int end)
{
    const Spline& sp = (*m.s)[size_t(r.spline)];
    const SplineModel& sm = m.of(r);
    size_t a, b; // the direction is from anchor a toward b
    if (r.kind == RefKind::Line)
    {
        a = size_t(r.index);
        b = (a + 1) % sp.size();
    }
    else
    {
        a = 0;
        b = sp.size() > 1 ? 1 : 0;
        if (end == 1)
        {
            a = sp.size() - 2;
            b = sp.size() - 1;
        }
    }
    if (sp.kind == Kind::Bezier && sm.hl.size() == sp.size())
    {
        // The handle at the meeting end gives the direction; a handle sitting on its anchor falls back to the chord.
        const Ref anchor = m.pointRef(r.spline, end == 0 ? a : b);
        const Ref handle{end == 0 ? RefKind::HandleR : RefKind::HandleL, r.spline, end == 0 ? int(a) : int(b)};
        const D2 d = m.point(v, handle) - m.point(v, anchor);
        if (glm::length(d) > 1e-9 * m.scale)
            return end == 0 ? d : -d;
    }
    return m.point(v, m.pointRef(r.spline, b)) - m.point(v, m.pointRef(r.spline, a));
}

/** The ends of a curve or line as point refs (start, end). */
std::pair<Ref, Ref> endsOf(const Splines& s, const Ref& r)
{
    const Spline& sp = s[size_t(r.spline)];
    if (r.kind == RefKind::Line)
        return {{RefKind::Point, r.spline, r.index}, {RefKind::Point, r.spline, int((size_t(r.index) + 1) % sp.size())}};
    return {{RefKind::Point, r.spline, 0}, {RefKind::Point, r.spline, -1}};
}

void addEquations(Model& m, const Constraint& c, const std::map<uint32_t, double>& values)
{
    const Splines& s = *m.s;
    const double L = m.scale;
    const uint32_t cid = c.id;
    const Model* mp = &m;
    auto deps = [&](std::initializer_list<Ref> refs) {
        std::vector<int> v;
        for (const Ref& r : refs)
            m.varsOf(r, v);
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        return v;
    };
    auto eq = [&](std::vector<int> vars, std::function<double(const X&)> f) { m.eqs.push_back({cid, std::move(vars), std::move(f)}); };
    auto shapeOf = [&](const Ref& r) { return m.of(r).shape; };
    const std::vector<Ref>& R = c.refs;
    auto value = [&]() {
        auto it = values.find(c.id);
        return it != values.end() ? it->second : c.value;
    };
    auto pointsEqual = [&](const Ref& a, const Ref& b) {
        std::vector<int> v = deps({a, b});
        eq(v, [mp, a, b](const X& x) { return mp->point(x, a).x - mp->point(x, b).x; });
        eq(v, [mp, a, b](const X& x) { return mp->point(x, a).y - mp->point(x, b).y; });
    };
    auto onLine = [&](const Ref& p, const Ref& l) {
        eq(deps({p, l}), [mp, p, l](const X& x) {
            auto [a, b] = mp->line(x, l);
            return cross2(unit(b - a), mp->point(x, p) - a);
        });
    };

    switch (c.type)
    {
    case Type::Coincident: pointsEqual(R[0], R[1]); break;
    case Type::Horizontal:
    case Type::Vertical:
    {
        const int axis = c.type == Type::Horizontal ? 1 : 0;
        if (R.size() == 1)
            eq(deps({R[0]}), [mp, r = R[0], axis](const X& x) {
                auto [a, b] = mp->line(x, r);
                return b[axis] - a[axis];
            });
        else
            eq(deps({R[0], R[1]}), [mp, p = R[0], q = R[1], axis](const X& x) { return mp->point(x, q)[axis] - mp->point(x, p)[axis]; });
        break;
    }
    case Type::Parallel:
    case Type::Perpendicular:
    {
        const bool par = c.type == Type::Parallel;
        eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1], par, L](const X& x) {
            auto [a0, a1] = mp->line(x, a);
            auto [b0, b1] = mp->line(x, b);
            const D2 da = a1 - a0, db = b1 - b0;
            return fold(angleBetween(da, db) - (par ? 0.0 : kPi * 0.5)) * L;
        });
        break;
    }
    case Type::Collinear:
    {
        auto [b0, b1] = endsOf(s, R[1]);
        onLine(b0, R[0]);
        onLine(b1, R[0]);
        break;
    }
    case Type::Tangent:
    {
        if (R.size() == 1)
        {
            // G1 at a Bezier anchor: its two handles stay on one line through it.
            const Ref hl{RefKind::HandleL, R[0].spline, R[0].index}, hr{RefKind::HandleR, R[0].spline, R[0].index};
            eq(deps({hl, hr}), [mp, hl, hr, p = R[0], L](const X& x) {
                const D2 o = mp->point(x, p);
                return fold(angleBetween(mp->point(x, hl) - o, mp->point(x, hr) - o)) * L;
            });
            break;
        }
        const Shape sa = R[0].kind == RefKind::Curve ? shapeOf(R[0]) : Shape::Points;
        const Shape sb = R[1].kind == RefKind::Curve ? shapeOf(R[1]) : Shape::Points;
        const bool roundA = sa != Shape::Points, roundB = sb != Shape::Points;
        if (roundA && roundB && c.endA >= 0 && c.endB >= 0)
        {
            // Arcs sharing an end: their radii there lie on one line.
            const Ref ma{RefKind::Point, R[0].spline, c.endA == 0 ? 0 : -1}, mb{RefKind::Point, R[1].spline, c.endB == 0 ? 0 : -1};
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1], ma, mb, L](const X& x) {
                return fold(angleBetween(mp->point(x, ma) - mp->center(x, a), mp->point(x, mb) - mp->center(x, b))) * L;
            });
        }
        else if (roundA && roundB)
        {
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1], sign = c.sign](const X& x) {
                const double d = glm::distance(mp->center(x, a), mp->center(x, b));
                const double ra = mp->radius(x, a), rb = mp->radius(x, b);
                return sign > 0.0 ? d - (ra + rb) : d - std::abs(ra - rb);
            });
        }
        else if (roundA || roundB)
        {
            const Ref round = roundA ? R[0] : R[1], other = roundA ? R[1] : R[0];
            const int roundEnd = roundA ? c.endA : c.endB, otherEnd = roundA ? c.endB : c.endA;
            if (roundEnd < 0 && otherEnd < 0)
                eq(deps({round, other}), [mp, round, other, sign = c.sign](const X& x) {
                    auto [a, b] = mp->line(x, other);
                    return cross2(unit(b - a), mp->center(x, round) - a) - sign * mp->radius(x, round);
                });
            else
            {
                // At the shared end the other side runs square to the radius.
                const auto [e0, e1] = endsOf(s, other);
                const Ref meet = roundEnd >= 0 ? Ref{RefKind::Point, round.spline, roundEnd == 0 ? 0 : -1} : (otherEnd == 1 ? e1 : e0);
                eq(deps({round, other, meet}), [mp, round, other, meet, otherEnd, L](const X& x) {
                    const D2 d = endDirection(*mp, x, other, std::max(otherEnd, 0));
                    return fold(angleBetween(mp->point(x, meet) - mp->center(x, round), d) - kPi * 0.5) * L;
                });
            }
        }
        else
        {
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1], ea = c.endA, eb = c.endB, L](const X& x) {
                return fold(angleBetween(endDirection(*mp, x, a, ea), endDirection(*mp, x, b, eb))) * L;
            });
        }
        break;
    }
    case Type::Equal:
        if (R[0].kind == RefKind::Line)
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1]](const X& x) {
                auto [a0, a1] = mp->line(x, a);
                auto [b0, b1] = mp->line(x, b);
                return glm::distance(a0, a1) - glm::distance(b0, b1);
            });
        else
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1]](const X& x) { return mp->radius(x, a) - mp->radius(x, b); });
        break;
    case Type::Concentric: pointsEqual(R[0], R[1]); break;
    case Type::Midpoint:
    {
        std::vector<int> v = deps({R[0], R[1]});
        for (int axis = 0; axis < 2; ++axis)
            eq(v, [mp, p = R[0], l = R[1], axis](const X& x) {
                auto [a, b] = mp->line(x, l);
                return mp->point(x, p)[axis] - (a[axis] + b[axis]) * 0.5;
            });
        break;
    }
    case Type::Symmetric:
    {
        const Ref mirror = R[2];
        auto pair = [&](const Ref& p, const Ref& q) {
            std::vector<int> v = deps({p, q, mirror});
            eq(v, [mp, p, q, mirror](const X& x) {
                auto [a, b] = mp->line(x, mirror);
                return glm::dot(mp->point(x, q) - mp->point(x, p), unit(b - a));
            });
            eq(v, [mp, p, q, mirror](const X& x) {
                auto [a, b] = mp->line(x, mirror);
                return cross2(unit(b - a), (mp->point(x, p) + mp->point(x, q)) * 0.5 - a);
            });
        };
        if (R[0].isPointLike())
            pair(R[0], R[1]);
        else if (R[0].kind == RefKind::Line)
        {
            auto [a0, a1] = endsOf(s, R[0]);
            auto [b0, b1] = endsOf(s, R[1]);
            pair(a0, c.endA == 1 ? b1 : b0);
            pair(a1, c.endA == 1 ? b0 : b1);
        }
        else
        {
            pair({RefKind::Center, R[0].spline, 0}, {RefKind::Center, R[1].spline, 0});
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1]](const X& x) { return mp->radius(x, a) - mp->radius(x, b); });
        }
        break;
    }
    case Type::Fixed:
    {
        const Ref r = R[0];
        auto hold = [&](const Ref& p, const D3& at) {
            const D2 q = m.pl.to2d(at);
            std::vector<int> v = deps({p});
            eq(v, [mp, p, q](const X& x) { return mp->point(x, p).x - q.x; });
            eq(v, [mp, p, q](const X& x) { return mp->point(x, p).y - q.y; });
        };
        if (r.isPointLike())
        {
            if (!c.anchors.empty())
                hold(r, c.anchors[0]);
        }
        else if (r.kind == RefKind::Line)
        {
            auto [a, b] = endsOf(s, r);
            if (c.anchors.size() >= 2)
            {
                hold(a, c.anchors[0]);
                hold(b, c.anchors[1]);
            }
        }
        else if (shapeOf(r) != Shape::Points)
        {
            if (c.anchors.size() < 2)
                break;
            const Ref ctr{RefKind::Center, r.spline, 0};
            hold(ctr, c.anchors[0]);
            const double rad = glm::distance(m.pl.to2d(c.anchors[1]), m.pl.to2d(c.anchors[0]));
            const int base = m.of(r).base;
            eq({base + 2}, [base, rad](const X& x) { return x[size_t(base) + 2] - rad; });
            if (shapeOf(r) == Shape::Arc && c.anchors.size() >= 3)
            {
                const D2 c0 = m.pl.to2d(c.anchors[0]);
                const double a0 = std::atan2(m.pl.to2d(c.anchors[1]).y - c0.y, m.pl.to2d(c.anchors[1]).x - c0.x);
                const double a1 = std::atan2(m.pl.to2d(c.anchors[2]).y - c0.y, m.pl.to2d(c.anchors[2]).x - c0.x);
                eq({base + 3}, [base, a0, rad](const X& x) { return wrapAngle(x[size_t(base) + 3] - a0) * rad; });
                eq({base + 4}, [base, a1, rad](const X& x) { return wrapAngle(x[size_t(base) + 4] - a1) * rad; });
            }
        }
        else
        {
            const Spline& sp = s[size_t(r.spline)];
            for (size_t i = 0; i < sp.size() && i < c.anchors.size(); ++i)
                hold(m.pointRef(r.spline, i), c.anchors[i]);
        }
        break;
    }
    case Type::PointOnCurve:
    {
        const Ref p = R[0], cv = R[1];
        if (cv.kind == RefKind::Line)
            onLine(p, cv);
        else if (shapeOf(cv) != Shape::Points)
            eq(deps({p, cv}), [mp, p, cv](const X& x) { return glm::distance(mp->point(x, p), mp->center(x, cv)) - mp->radius(x, cv); });
        else
        {
            // Any other curve: the point's parameter along it joins the unknowns, and the point sits where the curve is there.
            const Spline& sp = s[size_t(cv.spline)];
            const int spans = spanCount(sp);
            if (spans < 1)
                break;
            const double u0 = closestParam(sp, m.pl.to3d(m.point(m.x, p), m.pl.height(sp.points[0])));
            const double perSpan = std::max(length(sp) / spans, 1e-6);
            const int tv = m.newVar(u0, 1.0 / (perSpan * perSpan), 1e-6);
            std::vector<int> v = deps({p, cv});
            v.push_back(tv);
            const size_t si = size_t(cv.spline);
            auto at2d = [mp, si, tv, spans, closed = sp.closed](const X& x) {
                double u = x[size_t(tv)];
                u = closed ? u - std::floor(u / spans) * spans : std::clamp(u, 0.0, double(spans));
                return mp->pl.to2d(pointAt(mp->build(x, si), u));
            };
            eq(v, [mp, p, at2d](const X& x) { return mp->point(x, p).x - at2d(x).x; });
            eq(v, [mp, p, at2d](const X& x) { return mp->point(x, p).y - at2d(x).y; });
        }
        break;
    }
    case Type::Distance:
    {
        const double d = value();
        if (R[0].isPointLike() && R[1].isPointLike())
            eq(deps({R[0], R[1]}), [mp, a = R[0], b = R[1], d](const X& x) { return glm::distance(mp->point(x, a), mp->point(x, b)) - d; });
        else if (R[0].isPointLike())
            eq(deps({R[0], R[1]}), [mp, p = R[0], l = R[1], d, sign = c.sign](const X& x) {
                auto [a, b] = mp->line(x, l);
                return cross2(unit(b - a), mp->point(x, p) - a) - sign * d;
            });
        else
            eq(deps({R[0], R[1]}), [mp, l0 = R[0], l1 = R[1], d, sign = c.sign](const X& x) {
                auto [a, b] = mp->line(x, l0);
                auto [p, q] = mp->line(x, l1);
                return cross2(unit(b - a), (p + q) * 0.5 - a) - sign * d;
            });
        break;
    }
    case Type::HorizontalDistance:
    case Type::VerticalDistance:
    {
        const int axis = c.type == Type::HorizontalDistance ? 0 : 1;
        const double d = value();
        Ref a = R[0], b = R.size() > 1 ? R[1] : R[0];
        if (R.size() == 1)
            std::tie(a, b) = endsOf(s, R[0]);
        eq(deps({a, b}), [mp, a, b, axis, d, sign = c.sign](const X& x) { return mp->point(x, b)[axis] - mp->point(x, a)[axis] - sign * d; });
        break;
    }
    case Type::Length:
    {
        const double d = value();
        eq(deps({R[0]}), [mp, l = R[0], d](const X& x) {
            auto [a, b] = mp->line(x, l);
            return glm::distance(a, b) - d;
        });
        break;
    }
    case Type::Radius:
    case Type::Diameter:
    {
        const double d = c.type == Type::Diameter ? value() * 0.5 : value();
        const int base = m.of(R[0]).base;
        eq({base + 2}, [base, d](const X& x) { return x[size_t(base) + 2] - d; });
        break;
    }
    case Type::Angle:
    {
        const double a = glm::radians(value());
        eq(deps({R[0], R[1]}), [mp, l0 = R[0], l1 = R[1], a, sign = c.sign, L](const X& x) {
            auto [p0, p1] = mp->line(x, l0);
            auto [q0, q1] = mp->line(x, l1);
            const D2 da = p1 - p0, db = q1 - q0;
            return wrapAngle(std::atan2(cross2(da, db), glm::dot(da, db)) - sign * a) * L;
        });
        break;
    }
    default: break;
    }
}

/** Refs, counts and kinds a constraint type takes; throws std::invalid_argument naming what is wrong. */
void checkShape(const Constraint& c, const Splines& s, const Plane& pl)
{
    const std::string t = typeName(c.type);
    const std::vector<Ref>& R = c.refs;
    auto need = [&](size_t n) {
        if (R.size() != n)
            throw std::invalid_argument(t + " takes " + std::to_string(n) + (n == 1 ? " ref" : " refs") + ", got " + std::to_string(R.size()));
    };
    auto round = [&](const Ref& r) { return r.kind == RefKind::Curve && circleOf(s[size_t(r.spline)], pl).shape != Shape::Points; };
    auto line = [&](const Ref& r) { return r.kind == RefKind::Line; };
    auto pt = [&](const Ref& r) { return r.isPointLike(); };
    auto bad = [&](const char* what) { throw std::invalid_argument(t + " needs " + what); };
    switch (c.type)
    {
    case Type::Coincident:
        need(2);
        if (!pt(R[0]) || !pt(R[1]))
            bad("two points");
        break;
    case Type::Horizontal:
    case Type::Vertical:
        if (R.size() == 1 ? !line(R[0]) : (R.size() != 2 || !pt(R[0]) || !pt(R[1])))
            bad("a line or two points");
        break;
    case Type::Parallel:
    case Type::Perpendicular:
    case Type::Collinear:
    case Type::Angle:
        need(2);
        if (!line(R[0]) || !line(R[1]))
            bad("two lines");
        break;
    case Type::Tangent:
        if (R.size() == 1)
        {
            if (R[0].kind != RefKind::Point || s[size_t(R[0].spline)].kind != Kind::Bezier)
                bad("two curves, or one Bezier point");
            break;
        }
        need(2);
        for (const Ref& r : R)
            if (r.kind != RefKind::Line && r.kind != RefKind::Curve)
                bad("two curves or lines");
        break;
    case Type::Equal:
        need(2);
        if (!((line(R[0]) && line(R[1])) || (round(R[0]) && round(R[1]))))
            bad("two lines or two circles/arcs");
        break;
    case Type::Concentric:
        need(2);
        for (const Ref& r : R)
            if (r.kind != RefKind::Center)
                bad("two circles or arcs");
        break;
    case Type::Midpoint:
        need(2);
        if (!pt(R[0]) || !line(R[1]))
            bad("a point and a line");
        break;
    case Type::Symmetric:
        need(3);
        if (!line(R[2]))
            bad("two points, lines or circles and a mirror line last");
        if (!((pt(R[0]) && pt(R[1])) || (line(R[0]) && line(R[1])) || (round(R[0]) && round(R[1]))))
            bad("two points, two lines or two circles/arcs, then the mirror line");
        break;
    case Type::Fixed: need(1); break;
    case Type::PointOnCurve:
        need(2);
        if (!pt(R[0]) || (R[1].kind != RefKind::Line && R[1].kind != RefKind::Curve))
            bad("a point and a line or curve");
        if (R[1].kind == RefKind::Curve && R[0].kind == RefKind::Point && R[0].spline == R[1].spline)
            bad("a point of another curve");
        break;
    case Type::Distance:
        need(2);
        if (!((pt(R[0]) && pt(R[1])) || (pt(R[0]) && line(R[1])) || (line(R[0]) && line(R[1]))))
            bad("two points, a point and a line, or two lines");
        break;
    case Type::HorizontalDistance:
    case Type::VerticalDistance:
        if (R.size() == 1 ? !line(R[0]) : (R.size() != 2 || !pt(R[0]) || !pt(R[1])))
            bad("a line or two points");
        break;
    case Type::Length:
        need(1);
        if (!line(R[0]))
            bad("a line");
        break;
    case Type::Radius:
    case Type::Diameter:
        need(1);
        if (!round(R[0]))
            bad("a circle or arc");
        break;
    default: throw std::invalid_argument("unknown constraint type");
    }
}
} // namespace

D3 refPosition(const Splines& s, const Plane& pl, const Ref& r0)
{
    std::string why;
    if (!refValid(s, pl, r0, &why))
        return D3(0.0);
    const Ref r = normalized(s, r0);
    const Spline& sp = s[size_t(r.spline)];
    const CircleFit cf = circleOf(sp, pl);
    switch (r.kind)
    {
    case RefKind::Point:
        if (cf.shape == Shape::Arc)
            return pointAt(sp, r.index == 0 ? 0.0 : double(spanCount(sp)));
        return sp.points[size_t(r.index)];
    case RefKind::Line: return (sp.points[size_t(r.index)] + sp.points[(size_t(r.index) + 1) % sp.size()]) * 0.5;
    case RefKind::Curve: return spanCount(sp) ? pointAt(sp, spanCount(sp) * 0.5) : sp.points.front();
    case RefKind::Center: return pl.to3d(cf.center, cf.height);
    case RefKind::HandleL: return sp.handleL[size_t(r.index)];
    case RefKind::HandleR: return sp.handleR[size_t(r.index)];
    }
    return D3(0.0);
}

void prepare(Constraint& c, const Splines& s, const Plane& pl)
{
    // A circle or arc stands for its centre where concentric wants one.
    if (c.type == Type::Concentric)
        for (Ref& r : c.refs)
            if (r.kind == RefKind::Curve)
                r.kind = RefKind::Center;
    for (const Ref& r : c.refs)
    {
        std::string why;
        if (!refValid(s, pl, r, &why))
            throw std::invalid_argument(why);
    }
    // Points go first; coincident with a line or curve is point on curve.
    if ((c.type == Type::Coincident || c.type == Type::Midpoint || c.type == Type::PointOnCurve || c.type == Type::Distance) && c.refs.size() == 2 &&
        !c.refs[0].isPointLike() && c.refs[1].isPointLike())
        std::swap(c.refs[0], c.refs[1]);
    if (c.type == Type::Coincident && c.refs.size() == 2 && c.refs[0].isPointLike() && !c.refs[1].isPointLike())
        c.type = Type::PointOnCurve;
    // An arc's far end is stored as "end" so it survives the arc gaining or losing pieces.
    for (Ref& r : c.refs)
        if (r.kind == RefKind::Point && r.index > 0 && circleOf(s[size_t(r.spline)], pl).shape == Shape::Arc)
            r.index = -1;
    checkShape(c, s, pl);

    Model m = buildModel(s, Sketch{.plane = pl});
    const X& x = m.x;
    const std::vector<Ref>& R = c.refs;
    auto p2 = [&](const Ref& r) { return m.point(x, normalized(s, r)); };
    auto line2 = [&](const Ref& r) { return m.line(x, r); };
    const double tol = 1e-3 * m.scale;
    switch (c.type)
    {
    case Type::Fixed:
    {
        c.anchors.clear();
        const Ref r = R[0];
        if (r.isPointLike())
            c.anchors.push_back(refPosition(s, pl, r));
        else if (r.kind == RefKind::Line)
        {
            auto [a, b] = endsOf(s, r);
            c.anchors = {refPosition(s, pl, a), refPosition(s, pl, b)};
        }
        else
        {
            const CircleFit cf = circleOf(s[size_t(r.spline)], pl);
            if (cf.shape == Shape::Points)
                for (const D3& p : s[size_t(r.spline)].points)
                    c.anchors.push_back(p);
            else
            {
                c.anchors.push_back(pl.to3d(cf.center, cf.height));
                c.anchors.push_back(pl.to3d(cf.center + cf.radius * D2(std::cos(cf.start), std::sin(cf.start)), cf.height));
                if (cf.shape == Shape::Arc)
                {
                    const double e = cf.start + cf.sweep;
                    c.anchors.push_back(pl.to3d(cf.center + cf.radius * D2(std::cos(e), std::sin(e)), cf.height));
                }
            }
        }
        break;
    }
    case Type::Tangent:
    {
        if (R.size() != 2)
            break;
        auto roundOf = [&](const Ref& r) { return r.kind == RefKind::Curve && m.of(r).shape != Shape::Points; };
        const bool ra = roundOf(R[0]), rb = roundOf(R[1]);
        // The ends that meet: the closest pair. Round sides only need one when the other side is curved.
        auto endsOfSide = [&](const Ref& r) -> std::vector<std::pair<int, D2>> {
            if (roundOf(r))
            {
                if (m.of(r).shape == Shape::Circle)
                    return {};
                return {{0, p2({RefKind::Point, r.spline, 0})}, {1, p2({RefKind::Point, r.spline, -1})}};
            }
            auto [a, b] = endsOf(s, r);
            return {{0, p2(a)}, {1, p2(b)}};
        };
        const auto ea = endsOfSide(R[0]), eb = endsOfSide(R[1]);
        double best = 1e300;
        c.endA = c.endB = -1;
        for (auto& [ia, pa] : ea)
            for (auto& [ib, pb] : eb)
                if (glm::distance(pa, pb) < best)
                {
                    best = glm::distance(pa, pb);
                    c.endA = ia;
                    c.endB = ib;
                }
        if (ra && rb && best > tol)
        {
            // Apart, two circles touch outside or inside, whichever they are nearer.
            const double d = glm::distance(m.center(x, R[0]), m.center(x, R[1]));
            const double r0 = m.radius(x, R[0]), r1 = m.radius(x, R[1]);
            c.sign = std::abs(d - (r0 + r1)) <= std::abs(d - std::abs(r0 - r1)) ? 1.0 : -1.0;
            c.endA = c.endB = -1;
            break;
        }
        // A straight line and a circle that don't share an end (or a full circle) keep the line at a radius from the centre;
        // sharing an end, the line runs square to the radius there, which stays well conditioned beside a coincident.
        const Ref other = ra ? R[1] : R[0];
        const bool straightToRound = (ra || rb) && other.kind == RefKind::Line && s[size_t(other.spline)].kind != Kind::Bezier;
        if (straightToRound && best > tol)
        {
            const Ref round = ra ? R[0] : R[1];
            auto [a, b] = line2(other);
            const double side = cross2(unit(b - a), m.center(x, round) - a);
            c.sign = side >= 0.0 ? 1.0 : -1.0;
            c.endA = c.endB = -1;
            break;
        }
        if (best > tol)
            throw std::invalid_argument("tangent: the curves must meet at an end (make the ends coincident first)");
        break;
    }
    case Type::Symmetric:
        if (R[0].kind == RefKind::Line)
        {
            // Pair each end with the end of the other line nearest its mirror image.
            auto [m0, m1] = line2(R[2]);
            const D2 d = unit(m1 - m0);
            auto mirror = [&](const D2& p) {
                const D2 rel = p - m0;
                return m0 + d * glm::dot(rel, d) * 2.0 - rel;
            };
            auto [a0, a1] = line2(R[0]);
            auto [b0, b1] = line2(R[1]);
            const double same = glm::distance(mirror(a0), b0) + glm::distance(mirror(a1), b1);
            const double swapped = glm::distance(mirror(a0), b1) + glm::distance(mirror(a1), b0);
            c.endA = swapped < same ? 1 : 0;
        }
        break;
    case Type::Distance:
        if (!(R[0].isPointLike() && R[1].isPointLike()))
        {
            // Point to line: the side the point is on. Line to line: the side the second line's middle is on.
            const auto [a, b] = line2(R[0].isPointLike() ? R[1] : R[0]);
            const D2 p = R[0].isPointLike() ? p2(R[0]) : (line2(R[1]).first + line2(R[1]).second) * 0.5;
            c.sign = cross2(unit(b - a), p - a) >= 0.0 ? 1.0 : -1.0;
        }
        break;
    case Type::HorizontalDistance:
    case Type::VerticalDistance:
    {
        const int axis = c.type == Type::HorizontalDistance ? 0 : 1;
        D2 a, b;
        if (R.size() == 1)
            std::tie(a, b) = line2(R[0]);
        else
        {
            a = p2(R[0]);
            b = p2(R[1]);
        }
        c.sign = b[axis] - a[axis] >= 0.0 ? 1.0 : -1.0;
        break;
    }
    case Type::Angle:
    {
        auto [p0, p1] = line2(R[0]);
        auto [q0, q1] = line2(R[1]);
        c.sign = cross2(p1 - p0, q1 - q0) >= 0.0 ? 1.0 : -1.0;
        break;
    }
    default: break;
    }
}

double measure(const Constraint& c, const Splines& s, const Plane& pl)
{
    Model m = buildModel(s, Sketch{.plane = pl});
    const X& x = m.x;
    const std::vector<Ref>& R = c.refs;
    auto p2 = [&](const Ref& r) { return m.point(x, normalized(s, r)); };
    switch (c.type)
    {
    case Type::Distance:
        if (R[0].isPointLike() && R[1].isPointLike())
            return glm::distance(p2(R[0]), p2(R[1]));
        if (R[0].isPointLike())
        {
            auto [a, b] = m.line(x, R[1]);
            return std::abs(cross2(unit(b - a), p2(R[0]) - a));
        }
        else
        {
            auto [a, b] = m.line(x, R[0]);
            auto [p, q] = m.line(x, R[1]);
            return std::abs(cross2(unit(b - a), (p + q) * 0.5 - a));
        }
    case Type::HorizontalDistance:
    case Type::VerticalDistance:
    {
        const int axis = c.type == Type::HorizontalDistance ? 0 : 1;
        if (R.size() == 1)
        {
            auto [a, b] = m.line(x, R[0]);
            return std::abs(b[axis] - a[axis]);
        }
        return std::abs(p2(R[1])[axis] - p2(R[0])[axis]);
    }
    case Type::Length:
    {
        auto [a, b] = m.line(x, R[0]);
        return glm::distance(a, b);
    }
    case Type::Radius: return m.radius(x, R[0]);
    case Type::Diameter: return 2.0 * m.radius(x, R[0]);
    case Type::Angle:
    {
        auto [p0, p1] = m.line(x, R[0]);
        auto [q0, q1] = m.line(x, R[1]);
        return glm::degrees(std::abs(std::atan2(cross2(p1 - p0, q1 - q0), glm::dot(p1 - p0, q1 - q0))));
    }
    default: return 0.0;
    }
}

// Expressions

namespace
{
class ExprParser
{
public:
    ExprParser(std::string_view t, const std::map<std::string, double>& names) : mText(t), mNames(names) {}

    double parse()
    {
        double v = sum();
        skip();
        if (mPos != mText.size())
            throw std::invalid_argument("unexpected '" + std::string(mText.substr(mPos, 1)) + "'");
        return v;
    }

private:
    void skip()
    {
        while (mPos < mText.size() && std::isspace(static_cast<unsigned char>(mText[mPos])))
            ++mPos;
    }
    bool take(char c)
    {
        skip();
        if (mPos < mText.size() && mText[mPos] == c)
        {
            ++mPos;
            return true;
        }
        return false;
    }
    std::string ident()
    {
        skip();
        size_t b = mPos;
        while (mPos < mText.size() && (std::isalnum(static_cast<unsigned char>(mText[mPos])) || mText[mPos] == '_'))
            ++mPos;
        return std::string(mText.substr(b, mPos - b));
    }
    double sum()
    {
        double v = product();
        for (;;)
        {
            if (take('+'))
                v += product();
            else if (take('-'))
                v -= product();
            else
                return v;
        }
    }
    double product()
    {
        double v = power();
        for (;;)
        {
            if (take('*'))
                v *= power();
            else if (take('/'))
            {
                double d = power();
                if (std::abs(d) < 1e-300)
                    throw std::invalid_argument("division by zero");
                v /= d;
            }
            else
                return v;
        }
    }
    double power()
    {
        double v = unary();
        if (take('^'))
            v = std::pow(v, power());
        return v;
    }
    double unary()
    {
        if (take('-'))
            return -unary();
        if (take('+'))
            return unary();
        return withUnit(atom());
    }
    /** A unit right after a value scales it to cm or degrees. */
    double withUnit(double v)
    {
        skip();
        size_t save = mPos;
        std::string u = ident();
        static const std::map<std::string, double> units = {{"mm", 0.1}, {"cm", 1.0}, {"m", 100.0}, {"km", 100000.0}, {"in", 2.54},
                                                            {"ft", 30.48}, {"deg", 1.0}, {"rad", 180.0 / kPi}};
        if (auto it = units.find(u); it != units.end())
            return v * it->second;
        mPos = save;
        return v;
    }
    double atom()
    {
        skip();
        if (take('('))
        {
            double v = sum();
            if (!take(')'))
                throw std::invalid_argument("missing ')'");
            return v;
        }
        if (mPos < mText.size() && (std::isdigit(static_cast<unsigned char>(mText[mPos])) || mText[mPos] == '.'))
        {
            size_t b = mPos;
            while (mPos < mText.size() && (std::isdigit(static_cast<unsigned char>(mText[mPos])) || mText[mPos] == '.'))
                ++mPos;
            if (mPos < mText.size() && (mText[mPos] == 'e' || mText[mPos] == 'E') && mPos + 1 < mText.size() &&
                (std::isdigit(static_cast<unsigned char>(mText[mPos + 1])) || mText[mPos + 1] == '-' || mText[mPos + 1] == '+'))
            {
                mPos += 2;
                while (mPos < mText.size() && std::isdigit(static_cast<unsigned char>(mText[mPos])))
                    ++mPos;
            }
            double v = 0.0;
            std::string num(mText.substr(b, mPos - b));
            auto [p, ec] = lila::fromChars(num.data(), num.data() + num.size(), v);
            if (ec != std::errc() || p != num.data() + num.size())
                throw std::invalid_argument("bad number '" + num + "'");
            return v;
        }
        std::string id = ident();
        if (id.empty())
            throw std::invalid_argument(mPos < mText.size() ? "unexpected '" + std::string(mText.substr(mPos, 1)) + "'" : "missing value");
        static const std::map<std::string, double (*)(double)> fns = {
            {"sin", [](double a) { return std::sin(glm::radians(a)); }},  {"cos", [](double a) { return std::cos(glm::radians(a)); }},
            {"tan", [](double a) { return std::tan(glm::radians(a)); }},  {"asin", [](double a) { return glm::degrees(std::asin(a)); }},
            {"acos", [](double a) { return glm::degrees(std::acos(a)); }}, {"atan", [](double a) { return glm::degrees(std::atan(a)); }},
            {"sqrt", [](double a) { return std::sqrt(a); }},             {"abs", [](double a) { return std::abs(a); }},
            {"round", [](double a) { return std::round(a); }},           {"floor", [](double a) { return std::floor(a); }},
            {"ceil", [](double a) { return std::ceil(a); }}};
        if (auto f = fns.find(id); f != fns.end())
        {
            if (!take('('))
                throw std::invalid_argument(id + " needs (...)");
            double a = sum();
            if (!take(')'))
                throw std::invalid_argument("missing ')'");
            double r = f->second(a);
            if (!std::isfinite(r))
                throw std::invalid_argument(id + " is undefined there");
            return r;
        }
        if (id == "pi")
            return kPi;
        if (auto it = mNames.find(id); it != mNames.end())
            return it->second;
        throw std::invalid_argument("unknown name '" + id + "'");
    }

    std::string_view mText;
    const std::map<std::string, double>& mNames;
    size_t mPos = 0;
};

/** Names an expression mentions (identifiers that aren't functions, units or pi). */
std::set<std::string> namesIn(std::string_view t)
{
    std::set<std::string> out;
    size_t i = 0;
    while (i < t.size())
    {
        if (std::isalpha(static_cast<unsigned char>(t[i])) || t[i] == '_')
        {
            size_t b = i;
            while (i < t.size() && (std::isalnum(static_cast<unsigned char>(t[i])) || t[i] == '_'))
                ++i;
            // Right after a digit it is a unit, not a name.
            if (b > 0 && (std::isdigit(static_cast<unsigned char>(t[b - 1])) || t[b - 1] == '.'))
                continue;
            out.insert(std::string(t.substr(b, i - b)));
        }
        else
            ++i;
    }
    return out;
}
} // namespace

std::optional<double> evalExpression(std::string_view text, const std::map<std::string, double>& names, std::string* why)
{
    try
    {
        double v = ExprParser(text, names).parse();
        if (!std::isfinite(v))
            throw std::invalid_argument("not a finite number");
        return v;
    }
    catch (const std::exception& e)
    {
        if (why)
            *why = e.what();
        return std::nullopt;
    }
}

std::map<uint32_t, double> dimensionValues(const Sketch& k)
{
    std::map<uint32_t, double> out;
    std::map<std::string, double> known;
    std::vector<const Constraint*> pending;
    for (const Constraint& c : k.constraints)
    {
        if (!isDimension(c.type))
            continue;
        if (c.expr.empty())
        {
            out[c.id] = c.value;
            known[c.name] = c.value;
        }
        else
            pending.push_back(&c);
    }
    // Resolve expressions whose names are all known until none are left; a pass that resolves nothing means a cycle.
    while (!pending.empty())
    {
        bool progress = false;
        for (auto it = pending.begin(); it != pending.end();)
        {
            const Constraint& c = **it;
            bool ready = true;
            for (const std::string& n : namesIn(c.expr))
            {
                bool isName = false;
                for (const Constraint& o : k.constraints)
                    if (isDimension(o.type) && o.name == n)
                        isName = true;
                if (isName && !known.contains(n))
                    ready = false;
            }
            if (!ready)
            {
                ++it;
                continue;
            }
            std::string why;
            std::optional<double> v = evalExpression(c.expr, known, &why);
            if (!v)
                throw std::invalid_argument(c.name + " = " + c.expr + ": " + why);
            out[c.id] = *v;
            known[c.name] = *v;
            it = pending.erase(it);
            progress = true;
        }
        if (!progress)
            throw std::invalid_argument("dimension expressions refer to each other in a loop: " + pending.front()->name + " = " + pending.front()->expr);
    }
    return out;
}

// Solving

namespace
{
/** Cholesky of a symmetric positive definite n x n matrix (row-major) in place; false when it is not. */
bool cholesky(std::vector<double>& A, size_t n)
{
    for (size_t j = 0; j < n; ++j)
    {
        double d = A[j * n + j];
        for (size_t k = 0; k < j; ++k)
            d -= A[j * n + k] * A[j * n + k];
        if (!(d > 1e-300))
            return false;
        d = std::sqrt(d);
        A[j * n + j] = d;
        for (size_t i = j + 1; i < n; ++i)
        {
            double s = A[i * n + j];
            for (size_t k = 0; k < j; ++k)
                s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / d;
        }
    }
    return true;
}

void choleskySolve(const std::vector<double>& L, size_t n, std::vector<double>& b)
{
    for (size_t i = 0; i < n; ++i)
    {
        double s = b[i];
        for (size_t k = 0; k < i; ++k)
            s -= L[i * n + k] * b[k];
        b[i] = s / L[i * n + i];
    }
    for (size_t i = n; i-- > 0;)
    {
        double s = b[i];
        for (size_t k = i + 1; k < n; ++k)
            s -= L[k * n + i] * b[k];
        b[i] = s / L[i * n + i];
    }
}

/** Sparse Jacobian rows: per equation, (variable, derivative). Central differences. */
using Row = std::vector<std::pair<int, double>>;

std::vector<Row> jacobian(const std::vector<const Eq*>& eqs, X& x, const std::vector<double>& step)
{
    std::vector<Row> J(eqs.size());
    for (size_t i = 0; i < eqs.size(); ++i)
    {
        const Eq& e = *eqs[i];
        J[i].reserve(e.vars.size());
        for (int v : e.vars)
        {
            const double h = step[size_t(v)], keep = x[size_t(v)];
            x[size_t(v)] = keep + h;
            const double fp = e.f(x);
            x[size_t(v)] = keep - h;
            const double fm = e.f(x);
            x[size_t(v)] = keep;
            const double d = (fp - fm) / (2.0 * h);
            if (std::isfinite(d) && d != 0.0)
                J[i].push_back({v, d});
        }
    }
    return J;
}

double maxAbs(const std::vector<double>& r)
{
    double m = 0.0;
    for (double v : r)
        m = std::max(m, std::isfinite(v) ? std::abs(v) : 1e300);
    return m;
}

double sumSq(const std::vector<double>& r)
{
    double s = 0.0;
    for (double v : r)
        s += std::isfinite(v) ? v * v : 1e300;
    return s;
}

std::vector<double> residuals(const std::vector<const Eq*>& eqs, const X& x)
{
    std::vector<double> r(eqs.size());
    for (size_t i = 0; i < eqs.size(); ++i)
        r[i] = eqs[i]->f(x);
    return r;
}

struct Local
{
    bool ok = false;
    double error = 0.0;
    int iterations = 0;
};

/**
 * Damped Gauss-Newton with weighted minimum-norm steps: each step is the smallest change (by the variable weights) that
 * zeroes the linearised residuals, so under-constrained geometry moves as little as it can and dragged points, weighted
 * heavily, stay at their targets. The damping (Levenberg-Marquardt, in the equation space) keeps singular and
 * over-constrained systems stable.
 */
Local solveEqs(const std::vector<const Eq*>& eqs, X& x, const std::vector<double>& w, const std::vector<double>& step, int maxIt, double tol)
{
    Local out;
    const size_t m = eqs.size();
    std::vector<double> r = residuals(eqs, x);
    out.error = maxAbs(r);
    if (m == 0 || out.error <= tol)
    {
        out.ok = true;
        return out;
    }
    double lambda = 1e-10;
    std::vector<double> A(m * m), L, y;
    for (int it = 0; it < maxIt; ++it)
    {
        out.iterations = it + 1;
        std::vector<Row> J = jacobian(eqs, x, step);
        // A = J W J^T through the variables' rows, so the cost follows the sparsity.
        std::fill(A.begin(), A.end(), 0.0);
        std::map<int, std::vector<std::pair<size_t, double>>> byVar;
        for (size_t i = 0; i < m; ++i)
            for (auto [v, d] : J[i])
                byVar[v].push_back({i, d});
        for (auto& [v, list] : byVar)
        {
            const double wv = w[size_t(v)];
            for (auto [i, di] : list)
                for (auto [j, dj] : list)
                    A[i * m + j] += di * dj * wv;
        }
        double diagMean = 0.0;
        for (size_t i = 0; i < m; ++i)
            diagMean += A[i * m + i];
        diagMean = std::max(diagMean / double(m), 1e-30);
        const double before = sumSq(r);
        bool accepted = false;
        for (int tries = 0; tries < 12 && !accepted; ++tries)
        {
            L = A;
            for (size_t i = 0; i < m; ++i)
                L[i * m + i] += lambda * std::max(A[i * m + i], diagMean * 1e-6) + 1e-14 * diagMean;
            if (!cholesky(L, m))
            {
                lambda = std::max(lambda * 10.0, 1e-8);
                continue;
            }
            y.assign(m, 0.0);
            for (size_t i = 0; i < m; ++i)
                y[i] = -r[i];
            choleskySolve(L, m, y);
            X trial = x;
            for (size_t i = 0; i < m; ++i)
                for (auto [v, d] : J[i])
                    trial[size_t(v)] += w[size_t(v)] * d * y[i];
            std::vector<double> rt = residuals(eqs, trial);
            if (sumSq(rt) < before)
            {
                x = std::move(trial);
                r = std::move(rt);
                lambda = std::max(lambda * 0.1, 1e-12);
                accepted = true;
            }
            else
                lambda = std::max(lambda * 10.0, 1e-8);
        }
        out.error = maxAbs(r);
        if (out.error <= tol)
        {
            out.ok = true;
            return out;
        }
        if (!accepted)
            break;
    }
    return out;
}

/** Equations grouped by the variables they share: groups solve independently. */
std::vector<std::vector<const Eq*>> components(const Model& m, const std::vector<const Eq*>& eqs)
{
    std::vector<int> parent(m.x.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) { return parent[size_t(a)] == a ? a : parent[size_t(a)] = find(parent[size_t(a)]); };
    for (const Eq* e : eqs)
        for (size_t k = 1; k < e->vars.size(); ++k)
            parent[size_t(find(e->vars[k]))] = find(e->vars[0]);
    std::map<int, std::vector<const Eq*>> groups;
    for (const Eq* e : eqs)
        if (!e->vars.empty())
            groups[find(e->vars[0])].push_back(e);
    std::vector<std::vector<const Eq*>> out;
    for (auto& [root, list] : groups)
        out.push_back(std::move(list));
    return out;
}

/** The model with every valid constraint's equations. */
Model modelFor(const Splines& s, const Sketch& k, std::map<uint32_t, double>* valuesOut = nullptr)
{
    Model m = buildModel(s, k);
    std::map<uint32_t, double> values;
    try
    {
        values = dimensionValues(k);
    }
    catch (const std::exception& e)
    {
        m.warnings.push_back(e.what());
    }
    for (const Constraint& c : k.constraints)
    {
        bool ok = true;
        std::string why;
        for (const Ref& r : c.refs)
            if (!refValid(s, k.plane, r, &why))
                ok = false;
        if (ok)
        {
            try
            {
                checkShape(c, s, k.plane);
            }
            catch (const std::exception& e)
            {
                ok = false;
                why = e.what();
            }
        }
        if (!ok)
        {
            m.invalid.push_back(c.id);
            m.warnings.push_back("constraint " + std::to_string(c.id) + " (" + typeName(c.type) + ") skipped: " + why);
            continue;
        }
        Constraint cc = c;
        for (Ref& r : cc.refs)
            r = normalized(s, r);
        addEquations(m, cc, values);
    }
    if (valuesOut)
        *valuesOut = std::move(values);
    return m;
}
} // namespace

SolveResult solve(Splines& s, const Sketch& k, const SolveOptions& o)
{
    LILA_ZONE();

    SolveResult res;
    Model m = modelFor(s, k);
    res.invalid = m.invalid;
    res.warnings = m.warnings;
    const X original = m.x;
    // Dragged points start at their targets and weigh a million times more, so the solve moves everything else first.
    std::vector<double> w = m.w;
    for (const Drag& d : o.drags)
    {
        std::string why;
        if (!refValid(s, k.plane, d.point, &why))
        {
            res.warnings.push_back("drag: " + why);
            continue;
        }
        const Ref r = normalized(s, d.point);
        const SplineModel& sm = m.of(r);
        const D2 target = k.plane.to2d(d.to);
        if (r.kind == RefKind::Curve && sm.shape != Shape::Points)
        {
            // A circle or arc dragged by its rim takes the radius out to the target.
            const int rv = sm.base + 2;
            m.x[size_t(rv)] = std::max(glm::distance(target, m.center(m.x, r)), 1e-9);
            w[size_t(rv)] *= 1e-6;
            continue;
        }
        if (!r.isPointLike())
        {
            res.warnings.push_back("drag: " + refName(d.point) + " is not a point");
            continue;
        }
        if (r.kind == RefKind::Center || sm.shape == Shape::Points)
        {
            std::vector<int> v;
            m.varsOf(r, v);
            const D2 now = m.point(m.x, r);
            // A handle's variables are its anchor and its offset; only the offset moves when the handle is dragged.
            const bool handle = r.kind == RefKind::HandleL || r.kind == RefKind::HandleR;
            const int hv = handle ? (r.kind == RefKind::HandleL ? sm.hl[m.index(r)] : sm.hr[m.index(r)]) : -1;
            const int bv = r.kind == RefKind::Center ? sm.base : (handle ? hv : sm.base + int(2 * m.index(r)));
            if (bv >= 0)
            {
                m.x[size_t(bv)] += target.x - now.x;
                m.x[size_t(bv) + 1] += target.y - now.y;
                w[size_t(bv)] *= 1e-6;
                w[size_t(bv) + 1] *= 1e-6;
            }
            else if (handle)
                res.warnings.push_back("drag: " + refName(d.point) + " is not free to move (its handle type follows the anchor)");
        }
        else
        {
            // An arc end: move the end angle (and radius) toward the target, about the centre.
            const D2 c = m.center(m.x, r);
            const int av = sm.base + (m.index(r) == 0 ? 3 : 4);
            m.x[size_t(av)] = m.x[size_t(av)] + wrapAngle(std::atan2(target.y - c.y, target.x - c.x) - m.x[size_t(av)]);
            m.x[size_t(sm.base) + 2] = glm::distance(target, c);
            w[size_t(av)] *= 1e-6;
        }
    }
    std::vector<const Eq*> all;
    for (const Eq& e : m.eqs)
        all.push_back(&e);
    const double tol = 1e-12 * m.scale, failTol = 1e-6 * m.scale;
    for (const std::vector<const Eq*>& group : components(m, all))
    {
        const X start = m.x;
        Local l = solveEqs(group, m.x, w, m.step, o.maxIterations, tol);
        res.iterations = std::max(res.iterations, l.iterations);
        res.error = std::max(res.error, l.error);
        if (l.error <= failTol)
            continue;
        res.ok = false;
        std::set<uint32_t> ids;
        for (const Eq* e : group)
            ids.insert(e->cid);
        if (o.findConflicts && ids.size() <= 64)
        {
            for (uint32_t id : ids)
            {
                std::vector<const Eq*> without;
                for (const Eq* e : group)
                    if (e->cid != id)
                        without.push_back(e);
                X trial = start;
                Local t = solveEqs(without, trial, w, m.step, o.maxIterations, tol);
                if (t.error <= failTol)
                    res.conflicting.push_back(id);
            }
        }
        if (res.conflicting.empty())
        {
            // No single constraint to blame: report the ones left unsatisfied.
            std::set<uint32_t> bad;
            for (const Eq* e : group)
                if (std::abs(e->f(m.x)) > failTol)
                    bad.insert(e->cid);
            res.conflicting.assign(bad.begin(), bad.end());
        }
        // A failed group keeps its geometry as it was rather than a half-solved compromise.
        for (const Eq* e : group)
            for (int v : e->vars)
                m.x[size_t(v)] = original[size_t(v)];
    }
    for (size_t si = 0; si < s.size(); ++si)
    {
        Spline b = m.build(m.x, si);
        if (isValid(b))
            s[si] = std::move(b);
    }
    return res;
}

// Degrees of freedom

namespace
{
/** Row reduces M (rows x cols, row-major) in place. Returns the pivot column of each pivot row. */
std::vector<int> rowReduce(std::vector<double>& M, size_t rows, size_t cols, double tol)
{
    std::vector<int> pivots;
    size_t r = 0;
    for (size_t c = 0; c < cols && r < rows; ++c)
    {
        size_t best = r;
        for (size_t i = r + 1; i < rows; ++i)
            if (std::abs(M[i * cols + c]) > std::abs(M[best * cols + c]))
                best = i;
        if (std::abs(M[best * cols + c]) <= tol)
            continue;
        if (best != r)
            for (size_t k = 0; k < cols; ++k)
                std::swap(M[r * cols + k], M[best * cols + k]);
        const double p = M[r * cols + c];
        for (size_t k = 0; k < cols; ++k)
            M[r * cols + k] /= p;
        for (size_t i = 0; i < rows; ++i)
        {
            if (i == r)
                continue;
            const double f = M[i * cols + c];
            if (f == 0.0)
                continue;
            for (size_t k = 0; k < cols; ++k)
                M[i * cols + k] -= f * M[r * cols + k];
        }
        pivots.push_back(int(c));
        ++r;
    }
    return pivots;
}

/** The scaled Jacobian of a group: rows normalised, columns by the square root of the weights. */
std::vector<double> denseJacobian(const std::vector<const Eq*>& eqs, Model& m, const std::vector<int>& cols, const std::map<int, size_t>& colOf)
{
    std::vector<Row> J = jacobian(eqs, m.x, m.step);
    std::vector<double> D(eqs.size() * cols.size(), 0.0);
    for (size_t i = 0; i < J.size(); ++i)
    {
        double norm = 0.0;
        for (auto [v, d] : J[i])
        {
            const double s = d * std::sqrt(m.w[size_t(v)]);
            D[i * cols.size() + colOf.at(v)] = s;
            norm += s * s;
        }
        norm = std::sqrt(norm);
        if (norm > 0.0)
            for (size_t k = 0; k < cols.size(); ++k)
                D[i * cols.size() + k] /= norm;
    }
    return D;
}
} // namespace

bool Status::lineFixed(size_t spline, size_t index) const
{
    if (spline >= pointFixed.size() || pointFixed[spline].empty())
        return false;
    const auto& p = pointFixed[spline];
    return index < p.size() && p[index] && p[(index + 1) % p.size()];
}

bool Status::pointIsFixed(size_t spline, size_t index) const
{
    return spline < pointFixed.size() && index < pointFixed[spline].size() && pointFixed[spline][index];
}

Status analyze(const Splines& s, const Sketch& k)
{
    LILA_ZONE();

    Status st;
    Model m = modelFor(s, k);
    std::vector<bool> fixedVar(m.x.size(), false);
    std::vector<const Eq*> all;
    for (const Eq& e : m.eqs)
        all.push_back(&e);
    int rank = 0;
    for (const std::vector<const Eq*>& group : components(m, all))
    {
        std::vector<int> cols;
        for (const Eq* e : group)
            cols.insert(cols.end(), e->vars.begin(), e->vars.end());
        std::sort(cols.begin(), cols.end());
        cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
        std::map<int, size_t> colOf;
        for (size_t c = 0; c < cols.size(); ++c)
            colOf[cols[c]] = c;
        const std::vector<double> D = denseJacobian(group, m, cols, colOf);
        std::vector<double> R = D;
        const std::vector<int> pivots = rowReduce(R, group.size(), cols.size(), 1e-8);
        rank += int(pivots.size());
        // A variable is pinned when no null-space direction moves it: its pivot row has nothing in the free columns.
        std::vector<bool> isPivot(cols.size(), false);
        for (int p : pivots)
            isPivot[size_t(p)] = true;
        for (size_t row = 0; row < pivots.size(); ++row)
        {
            bool pinned = true;
            for (size_t c = 0; c < cols.size() && pinned; ++c)
                if (!isPivot[c] && std::abs(R[row * cols.size() + c]) > 1e-7)
                    pinned = false;
            if (pinned)
                fixedVar[size_t(cols[size_t(pivots[row])])] = true;
        }
        // Redundant constraints: dropping their rows leaves the rank as it was.
        std::set<uint32_t> ids;
        for (const Eq* e : group)
            ids.insert(e->cid);
        if (ids.size() <= 150)
            for (uint32_t id : ids)
            {
                std::vector<double> sub;
                size_t rows = 0;
                for (size_t i = 0; i < group.size(); ++i)
                    if (group[i]->cid != id)
                    {
                        sub.insert(sub.end(), D.begin() + ptrdiff_t(i * cols.size()), D.begin() + ptrdiff_t((i + 1) * cols.size()));
                        ++rows;
                    }
                if (rowReduce(sub, rows, cols.size(), 1e-8).size() == pivots.size())
                    st.redundant.push_back(id);
            }
    }
    st.dof = int(m.x.size()) - rank;
    st.pointFixed.resize(s.size());
    st.splineFixed.assign(s.size(), false);
    for (size_t si = 0; si < s.size(); ++si)
    {
        const SplineModel& sm = m.sp[si];
        const int b = sm.base;
        if (sm.shape == Shape::Points)
        {
            bool all2 = s[si].size() > 0;
            st.pointFixed[si].resize(s[si].size());
            for (size_t i = 0; i < s[si].size(); ++i)
            {
                const bool f = fixedVar[size_t(b) + 2 * i] && fixedVar[size_t(b) + 2 * i + 1];
                st.pointFixed[si][i] = f;
                all2 = all2 && f;
            }
            st.splineFixed[si] = all2;
        }
        else
        {
            const bool c = fixedVar[size_t(b)] && fixedVar[size_t(b) + 1] && fixedVar[size_t(b) + 2];
            st.pointFixed[si].assign(s[si].size(), false);
            if (sm.shape == Shape::Arc)
            {
                const bool a0 = c && fixedVar[size_t(b) + 3], a1 = c && fixedVar[size_t(b) + 4];
                if (!st.pointFixed[si].empty())
                {
                    st.pointFixed[si].front() = a0;
                    st.pointFixed[si].back() = a1;
                }
                st.splineFixed[si] = a0 && a1;
            }
            else
                st.splineFixed[si] = c;
        }
    }
    return st;
}

AddCheck checkAdd(const Splines& s, const Sketch& k, const Constraint& c)
{
    AddCheck out;
    Sketch with = k;
    with.constraints.push_back(c);
    if (isDimension(c.type) && with.constraints.back().name.empty())
        with.constraints.back().name = "_new";
    Splines trial = s;
    out.result = solve(trial, with, {});
    if (!out.result.ok)
        return out;
    // Already implied: the constraint's rows add no rank at the solved geometry.
    Status st = analyze(trial, with);
    out.redundant = std::ranges::find(st.redundant, c.id) != st.redundant.end();
    return out;
}

// Inference

Inference infer(const Splines& s, const Plane& pl, D2 at, const std::optional<D2>& previous, double tol, const std::vector<Ref>& ignore)
{
    Inference out;
    out.at = at;
    const Ref self{RefKind::Point, kNew, 0};
    auto ignored = [&](const Ref& r) { return std::ranges::find(ignore, r) != ignore.end(); };
    // Existing points and centres first: coincident wins over everything.
    double best = tol;
    std::optional<Ref> hit;
    D2 hitAt(0.0);
    for (size_t si = 0; si < s.size(); ++si)
    {
        const Spline& sp = s[si];
        const CircleFit cf = circleOf(sp, pl);
        std::vector<std::pair<Ref, D2>> cands;
        if (cf.shape == Shape::Points)
            for (size_t i = 0; i < sp.size(); ++i)
                cands.push_back({{RefKind::Point, int(si), int(i)}, pl.to2d(sp.points[i])});
        else
        {
            cands.push_back({{RefKind::Center, int(si), 0}, cf.center});
            if (cf.shape == Shape::Arc)
            {
                cands.push_back({{RefKind::Point, int(si), 0}, cf.center + cf.radius * D2(std::cos(cf.start), std::sin(cf.start))});
                const double e = cf.start + cf.sweep;
                cands.push_back({{RefKind::Point, int(si), -1}, cf.center + cf.radius * D2(std::cos(e), std::sin(e))});
            }
        }
        for (auto& [r, p] : cands)
            if (!ignored(r) && glm::distance(p, at) < best)
            {
                best = glm::distance(p, at);
                hit = r;
                hitAt = p;
            }
    }
    if (hit)
    {
        out.at = hitAt;
        Constraint c;
        c.type = Type::Coincident;
        c.refs = {self, *hit};
        out.constraints.push_back(c);
        return out;
    }
    if (previous)
    {
        const D2 d = at - *previous;
        const Ref seg{RefKind::Line, kNew, 0};
        if (glm::length(d) > tol)
        {
            Constraint c;
            c.refs = {seg};
            if (std::abs(d.y) < tol)
            {
                out.at.y = previous->y;
                c.type = Type::Horizontal;
                out.constraints.push_back(c);
            }
            else if (std::abs(d.x) < tol)
            {
                out.at.x = previous->x;
                c.type = Type::Vertical;
                out.constraints.push_back(c);
            }
        }
    }
    if (!out.constraints.empty())
        return out;
    // On a line or curve.
    best = tol;
    std::optional<std::pair<Ref, D2>> on;
    for (size_t si = 0; si < s.size(); ++si)
    {
        const Spline& sp = s[si];
        const CircleFit cf = circleOf(sp, pl);
        if (cf.shape != Shape::Points)
        {
            const D2 rel = at - cf.center;
            const double d = std::abs(glm::length(rel) - cf.radius);
            const Ref r{RefKind::Curve, int(si), 0};
            if (d < best && glm::length(rel) > 1e-12 && !ignored(r))
            {
                best = d;
                on = {r, cf.center + rel / glm::length(rel) * cf.radius};
            }
            continue;
        }
        for (size_t i = 0; i < lineCount(sp); ++i)
        {
            const D2 a = pl.to2d(sp.points[i]), b = pl.to2d(sp.points[(i + 1) % sp.size()]);
            const D2 ab = b - a;
            const double len2 = glm::dot(ab, ab);
            if (len2 < 1e-24)
                continue;
            const double t = std::clamp(glm::dot(at - a, ab) / len2, 0.0, 1.0);
            const D2 q = a + ab * t;
            const Ref r{RefKind::Line, int(si), int(i)};
            if (glm::distance(q, at) < best && !ignored(r) && (sp.kind == Kind::Poly || (sp.kind == Kind::Nurbs && sp.degree == 1)))
            {
                best = glm::distance(q, at);
                on = {r, q};
            }
        }
    }
    if (on)
    {
        out.at = on->second;
        Constraint c;
        c.type = Type::PointOnCurve;
        c.refs = {self, on->first};
        out.constraints.push_back(c);
    }
    return out;
}
} // namespace lila::curve::sketch
