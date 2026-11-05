#include "curve/Pipe.h"

#include "core/Decimal.h"

#include "core/Profile.h"
#include "curve/CurveMath.h"
#include "curve/Internal.h"
#include "curve/Ops.h"
#include "geo/Attribute.h"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <limits>

namespace lila::curve
{
namespace
{
using detail::addFaceUv;
using detail::setFaceInt;
using geo::Index;

constexpr double kLengthEps = 1e-9;
constexpr double kCornerAngle = 0.0175;

void setAllFaces(geo::Mesh& m, std::string_view name, int value)
{
    auto v = m.faceAttrs().getOrAdd<int32_t>(name);
    for (Index f = 0; f < m.faceCapacity(); ++f)
        if (m.faceValid(f))
            v[f] = value;
}

/** Faces turned outward on a closed mesh: flipped when the enclosed volume comes out negative. */
void orientOutward(geo::Mesh& m)
{
    double vol = 0.0;
    for (Index f = 0; f < m.faceCapacity(); ++f)
        if (m.faceValid(f))
            vol += glm::dot(m.faceCentroid(f), m.faceAreaVector(f));
    if (vol < 0.0)
        for (Index f = 0; f < m.faceCapacity(); ++f)
            if (m.faceValid(f))
                m.flipFace(f);
}

void addBox(geo::Mesh& m, const D3& lo, const D3& hi)
{
    Index p[8];
    for (int i = 0; i < 8; ++i)
        p[i] = m.addPoint(D3(i & 1 ? hi.x : lo.x, i & 2 ? hi.y : lo.y, i & 4 ? hi.z : lo.z));
    const int quads[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
    const std::vector<glm::vec2> uv = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    for (auto& q : quads)
    {
        const Index face[4] = {p[q[0]], p[q[1]], p[q[2]], p[q[3]]};
        addFaceUv(m, face, uv, false);
    }
}

/** A station along a path: position, frame and the path's own scale, interpolated between its points. */
struct Station
{
    D3 origin{0.0};
    Frame frame;
    double scale = 1.0;
};

Station sampleAt(const Path& P, double s)
{
    size_t n = P.size();
    double L = P.closed ? P.length : P.s.back();
    s = P.closed ? s - std::floor(s / std::max(L, kLengthEps)) * L : std::clamp(s, 0.0, L);
    size_t i = size_t(std::upper_bound(P.s.begin(), P.s.end(), s) - P.s.begin());
    i = i == 0 ? 0 : i - 1;
    size_t j = i + 1;
    double sj = 0.0;
    if (j >= n)
    {
        if (!P.closed)
            return {P.points.back(), P.frames.back(), P.scale.back()};
        j = 0;
        sj = L;
    }
    else
        sj = P.s[j];
    double f = sj - P.s[i] > kLengthEps ? (s - P.s[i]) / (sj - P.s[i]) : 0.0;
    Station st;
    st.origin = glm::mix(P.points[i], P.points[j], f);
    st.frame.t = glm::normalize(glm::mix(P.frames[i].t, P.frames[j].t, f) + D3(1e-30));
    st.frame.n = perpUnit(glm::mix(P.frames[i].n, P.frames[j].n, f), st.frame.t);
    if (glm::length(st.frame.n) < 0.5)
        st.frame.n = P.frames[i].n;
    st.frame.b = glm::cross(st.frame.t, st.frame.n);
    st.scale = glm::mix(P.scale[i], P.scale[j], f);
    return st;
}

/** The part of an open path between two arc lengths, frames kept. */
Path subPath(const Path& P, double s0, double s1)
{
    Path out;
    double L = P.closed ? P.length : P.s.back();
    if (!P.closed)
    {
        s0 = std::clamp(s0, 0.0, L);
        s1 = std::clamp(s1, 0.0, L);
    }
    auto push = [&](const D3& p, const Frame& f, double sc, double s, bool kink, const D3& tin, const D3& tout) {
        out.points.push_back(p);
        out.frames.push_back(f);
        out.scale.push_back(sc);
        out.s.push_back(s - s0);
        out.kink.push_back(kink ? 1 : 0);
        out.tangentIn.push_back(tin);
        out.tangentOut.push_back(tout);
    };
    Station a = sampleAt(P, s0);
    push(a.origin, a.frame, a.scale, s0, false, a.frame.t, a.frame.t);
    for (size_t i = 0; i < P.size(); ++i)
        if (P.s[i] > s0 + 1e-6 && P.s[i] < s1 - 1e-6)
            push(P.points[i], P.frames[i], P.scale[i], P.s[i], P.kink[i] != 0, P.tangentIn[i], P.tangentOut[i]);
    Station b = sampleAt(P, s1);
    push(b.origin, b.frame, b.scale, s1, false, b.frame.t, b.frame.t);
    out.length = s1 - s0;
    return out;
}

/** Arc length of the path point nearest p, and its distance. */
double nearestS(const Path& P, const D3& p, double* dist)
{
    size_t n = P.size();
    size_t segs = P.closed ? n : n - 1;
    double best = std::numeric_limits<double>::max(), bestS = 0.0;
    for (size_t i = 0; i < segs; ++i)
    {
        const D3& a = P.points[i];
        const D3& b = P.points[(i + 1) % n];
        D3 d = b - a;
        double l2 = glm::dot(d, d);
        double t = l2 > 0.0 ? std::clamp(glm::dot(p - a, d) / l2, 0.0, 1.0) : 0.0;
        double dd = glm::distance(a + d * t, p);
        if (dd < best)
        {
            best = dd;
            bestS = P.s[i] + t * std::sqrt(l2);
        }
    }
    if (dist)
        *dist = best;
    return bestS;
}

/** Blur width of a swell as a fraction of the shortest span between points whose radii differ. */
constexpr double kSwellSoftness = 0.3;

/**
 * Gaussian blur of per-point values by arc length (sigma in cm). Closed paths wrap; open ends reflect through the end
 * value, which keeps the ends and any straight taper exact. Samples are weighted by the length they cover, so uneven
 * spacing doesn't bias the result.
 */
void blurAlong(const Path& P, std::vector<double>& v, double sigma)
{
    const size_t n = P.size();
    if (sigma <= kLengthEps || n < 3 || v.size() != n)
        return;
    const double L = P.closed ? P.length : P.s.back();
    const double reach = 3.0 * sigma;
    struct Sample
    {
        double s, v, w;
    };
    std::vector<Sample> ext;
    ext.reserve(n * 3);
    for (size_t i = 0; i < n; ++i)
    {
        const double prev = i > 0 ? P.s[i - 1] : (P.closed ? P.s[n - 1] - L : P.s[0]);
        const double next = i + 1 < n ? P.s[i + 1] : (P.closed ? P.s[0] + L : P.s[n - 1]);
        const double w = std::max(0.5 * (next - prev), 1e-9);
        ext.push_back({P.s[i], v[i], w});
        if (P.closed)
        {
            ext.push_back({P.s[i] - L, v[i], w});
            ext.push_back({P.s[i] + L, v[i], w});
        }
        else
        {
            ext.push_back({-P.s[i], 2.0 * v.front() - v[i], w});
            ext.push_back({2.0 * L - P.s[i], 2.0 * v.back() - v[i], w});
        }
    }
    std::ranges::sort(ext, {}, &Sample::s);
    const double k = -0.5 / (sigma * sigma);
    for (size_t j = 0; j < n; ++j)
    {
        const double sj = P.s[j];
        auto it = std::ranges::lower_bound(ext, sj - reach, {}, &Sample::s);
        // A weighted line fit rather than a weighted mean, so uneven spacing can't bend a straight taper.
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, t0 = 0.0, t1 = 0.0;
        for (; it != ext.end() && it->s <= sj + reach; ++it)
        {
            // Tapered to zero at the reach, so a sample landing on either edge of the window can't tip the balance.
            const double d = it->s - sj, edge = 1.0 - d * d / (reach * reach);
            const double w = it->w * std::exp(k * d * d) * edge * edge;
            s0 += w;
            s1 += w * d;
            s2 += w * d * d;
            t0 += w * it->v;
            t1 += w * d * it->v;
        }
        const double det = s0 * s2 - s1 * s1;
        if (std::abs(det) > 1e-12 * s0 * s2)
            v[j] = (s2 * t0 - s1 * t1) / det;
        else if (s0 > 0.0)
            v[j] = t0 / s0;
    }
}

/**
 * The input's per-point radius at each path point, by arc length: a monotone cubic (Fritsch-Carlson) through the
 * input points' stations, so the pipe swells and narrows smoothly through them with no plateaus or overshoot.
 */
std::vector<double> radiusAlong(const Path& P, const Spline& input)
{
    const size_t n = P.size();
    const double L = P.closed ? P.length : P.s.back();
    std::vector<double> xs, ys;
    // Stations in order: each point is searched for only past the previous one, so a path crossing itself can't jump.
    size_t from = 0;
    for (size_t k = 0; k < input.size(); ++k)
    {
        double best = std::numeric_limits<double>::max(), bestS = xs.empty() ? 0.0 : xs.back();
        for (size_t i = from; i + 1 < n; ++i)
        {
            const D3 &a = P.points[i], &b = P.points[i + 1], d = b - a;
            const double l2 = glm::dot(d, d);
            const double t = l2 > 0.0 ? std::clamp(glm::dot(input.points[k] - a, d) / l2, 0.0, 1.0) : 0.0;
            const double dd = glm::distance(a + d * t, input.points[k]);
            if (dd < best - 1e-9)
            {
                best = dd;
                bestS = P.s[i] + t * std::sqrt(l2);
                from = i;
            }
        }
        if (k == 0)
            bestS = 0.0;
        else if (k + 1 == input.size() && !input.closed)
            bestS = L;
        if (!xs.empty() && bestS <= xs.back() + 1e-6)
            continue;
        xs.push_back(bestS);
        ys.push_back(input.radius[k]);
    }
    if (input.closed && !xs.empty())
    {
        xs.push_back(L);
        ys.push_back(ys.front());
    }
    std::vector<double> out(n, ys.empty() ? 1.0 : ys.front());
    const size_t m = xs.size();
    if (m < 2)
        return out;
    std::vector<double> delta(m - 1), tan(m, 0.0);
    for (size_t i = 0; i + 1 < m; ++i)
        delta[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]);
    tan[0] = delta[0];
    tan[m - 1] = delta[m - 2];
    for (size_t i = 1; i + 1 < m; ++i)
        tan[i] = delta[i - 1] * delta[i] <= 0.0 ? 0.0 : (delta[i - 1] + delta[i]) * 0.5;
    for (size_t i = 0; i + 1 < m; ++i)
    {
        if (std::abs(delta[i]) < 1e-12)
        {
            tan[i] = tan[i + 1] = 0.0;
            continue;
        }
        const double a = tan[i] / delta[i], b = tan[i + 1] / delta[i], h = a * a + b * b;
        if (h > 9.0)
        {
            const double t = 3.0 / std::sqrt(h);
            tan[i] = t * a * delta[i];
            tan[i + 1] = t * b * delta[i];
        }
    }
    for (size_t j = 0; j < n; ++j)
    {
        const double x = std::clamp(P.s[j], xs.front(), xs.back());
        const size_t i = std::min(size_t(std::upper_bound(xs.begin(), xs.end(), x) - xs.begin()), m - 1) - 1;
        const double h = xs[i + 1] - xs[i], t = (x - xs[i]) / h, t2 = t * t, t3 = t2 * t;
        out[j] = (2 * t3 - 3 * t2 + 1) * ys[i] + (t3 - 2 * t2 + t) * h * tan[i] + (-2 * t3 + 3 * t2) * ys[i + 1] +
                 (t3 - t2) * h * tan[i + 1];
    }
    // The cubic is only C1, so its curvature jumps at every station and shows as a ring-shaped ridge in the shading.
    double gap = std::numeric_limits<double>::max();
    for (size_t i = 0; i + 1 < m; ++i)
        if (std::abs(ys[i + 1] - ys[i]) > 1e-9)
            gap = std::min(gap, xs[i + 1] - xs[i]);
    if (gap < std::numeric_limits<double>::max())
        blurAlong(P, out, gap * kSwellSoftness);
    return out;
}

/** Extra points so no segment is longer than step, at whole multiples of it along the curve. */
Polyline resample(const Polyline& pl, double step)
{
    if (step <= kLengthEps)
        return pl;
    Polyline out;
    out.closed = pl.closed;
    out.length = pl.length;
    size_t n = pl.size();
    size_t segs = pl.closed ? n : n - 1;
    for (size_t i = 0; i < n; ++i)
    {
        out.points.push_back(pl.points[i]);
        out.tangentIn.push_back(pl.tangentIn[i]);
        out.tangentOut.push_back(pl.tangentOut[i]);
        out.params.push_back(pl.params[i]);
        out.arcLength.push_back(pl.arcLength[i]);
        out.kink.push_back(pl.kink[i]);
        if (i >= segs)
            continue;
        size_t j = (i + 1) % n;
        double s0 = pl.arcLength[i], s1 = j == 0 ? pl.length : pl.arcLength[j];
        double u1 = j == 0 ? pl.params[i] + (pl.params[i] - (i ? pl.params[i - 1] : pl.params[i])) : pl.params[j];
        for (double s = (std::floor(s0 / step) + 1.0) * step; s < s1 - step * 0.05; s += step)
        {
            double f = (s - s0) / std::max(s1 - s0, kLengthEps);
            D3 t = glm::normalize(glm::mix(pl.tangentOut[i], pl.tangentIn[j], f) + D3(1e-30));
            out.points.push_back(glm::mix(pl.points[i], pl.points[j], f));
            out.tangentIn.push_back(t);
            out.tangentOut.push_back(t);
            out.params.push_back(glm::mix(pl.params[i], u1, f));
            out.arcLength.push_back(s);
            out.kink.push_back(0);
        }
    }
    return out;
}

/** The part laid in its station: local x along the normal, y the binormal, z the tangent. */
void place(geo::Mesh& dst, geo::Mesh part, const D3& origin, const Frame& f, PipePart kind, int segment, int material)
{
    for (Index p = 0; p < part.pointCapacity(); ++p)
        if (part.pointValid(p))
        {
            D3 q = part.position(p);
            part.setPosition(p, origin + f.n * q.x + f.b * q.y + f.t * q.z);
        }
    setAllFaces(part, "pipePart", int(kind));
    setAllFaces(part, "part", segment);
    if (material >= 0)
        setAllFaces(part, geo::attr::kMaterial, material);
    dst.append(part);
}

/** A user part with its long axis turned onto +z and scaled. */
geo::Mesh userPart(const geo::Mesh& src, int axis, double scale)
{
    geo::Mesh m = src;
    for (Index p = 0; p < m.pointCapacity(); ++p)
        if (m.pointValid(p))
        {
            D3 q = m.position(p);
            D3 r = axis == 0 ? D3(q.y, q.z, q.x) : axis == 1 ? D3(q.x, -q.z, q.y) : q;
            m.setPosition(p, r * scale);
        }
    return m;
}

/**
 * Moves a built-in part's uvs into the pipe's layout: lathe sides (0 to 1 around, 0 to 1 along) into the body rect, lathe
 * ends (u from 1.1) into the caps rect.
 */
void partUvs(geo::Mesh& part, const StripUv& uv)
{
    auto c = part.cornerAttrs().get<glm::vec2>(geo::attr::uvName(0));
    for (glm::vec2& t : c)
    {
        if (t.x >= 1.05f)
            t = uv.caps.at(t.x - 1.1, t.y);
        else
            t = uv.layout == UvLayout::Along ? uv.rect.at(t.y, t.x) : uv.rect.at(t.x, t.y);
    }
}

double hexHeadRadius(double size) { return size * 0.866; }
double nutHeight(double size) { return size * 0.8; }
} // namespace

const char* couplingKindName(CouplingKind k)
{
    switch (k)
    {
    case CouplingKind::None: return "none";
    case CouplingKind::Flange: return "flange";
    case CouplingKind::Collar: return "collar";
    case CouplingKind::Clamp: return "clamp";
    case CouplingKind::ElbowCollar: return "elbowCollar";
    case CouplingKind::Mesh: return "mesh";
    }
    return "none";
}

const char* fastenerKindName(FastenerKind k)
{
    switch (k)
    {
    case FastenerKind::None: return "none";
    case FastenerKind::BoltAndNut: return "boltAndNut";
    case FastenerKind::Bolt: return "bolt";
    case FastenerKind::Nut: return "nut";
    case FastenerKind::Mesh: return "mesh";
    }
    return "none";
}

std::vector<double> parseNumbers(std::string_view list)
{
    std::vector<double> out;
    size_t i = 0;
    while (i < list.size())
    {
        while (i < list.size() && (list[i] == ' ' || list[i] == ',' || list[i] == '\t'))
            ++i;
        size_t j = i;
        while (j < list.size() && list[j] != ' ' && list[j] != ',' && list[j] != '\t')
            ++j;
        double v = 0.0;
        if (j > i && lila::fromChars(list.data() + i, list.data() + j, v).ec == std::errc())
            out.push_back(v);
        i = j;
    }
    return out;
}

std::function<double(double)> parseRamp(std::string_view keys)
{
    std::vector<std::pair<double, double>> k;
    std::string s(keys);
    std::replace(s.begin(), s.end(), ':', ' ');
    std::vector<double> nums = parseNumbers(s);
    for (size_t i = 0; i + 1 < nums.size(); i += 2)
        k.push_back({std::clamp(nums[i], 0.0, 1.0), nums[i + 1]});
    if (k.empty())
        return {};
    std::ranges::sort(k);
    return [k](double t) {
        if (t <= k.front().first)
            return k.front().second;
        if (t >= k.back().first)
            return k.back().second;
        size_t i = 1;
        while (i < k.size() && k[i].first < t)
            ++i;
        double span = k[i].first - k[i - 1].first;
        double f = span > kLengthEps ? (t - k[i - 1].first) / span : 1.0;
        f = f * f * (3.0 - 2.0 * f);
        return glm::mix(k[i - 1].second, k[i].second, f);
    };
}

geo::Mesh lathe(const std::vector<glm::dvec2>& input, bool closed, int sides, double phase)
{
    geo::Mesh m;
    sides = std::max(3, sides);
    std::vector<D2> P = input;
    bool capStart = false, capEnd = false;
    if (!closed)
    {
        if (!P.empty() && P.front().x < 1e-9)
        {
            P.erase(P.begin());
            capStart = true;
        }
        if (!P.empty() && P.back().x < 1e-9)
        {
            P.pop_back();
            capEnd = true;
        }
    }
    size_t n = P.size();
    if (n < (closed ? 3u : 1u))
        return m;
    std::vector<double> v(n + 1, 0.0);
    for (size_t j = 1; j <= n; ++j)
        v[j] = v[j - 1] + (j < n || closed ? glm::distance(P[j - 1], P[j % n]) : 0.0);
    double vl = std::max(closed ? v[n] : v[n - 1], kLengthEps);
    for (size_t j = 0; j < n; ++j)
        for (int i = 0; i < sides; ++i)
        {
            double a = phase + glm::two_pi<double>() * i / sides;
            m.addPoint(D3(P[j].x * std::cos(a), P[j].x * std::sin(a), P[j].y));
        }
    auto idx = [&](size_t j, int i) { return Index(j * size_t(sides) + size_t(i % sides)); };
    size_t segs = closed ? n : n - 1;
    for (size_t j = 0; j < segs; ++j)
    {
        size_t j1 = (j + 1) % n;
        float v0 = float(v[j] / vl), v1 = float(v[j + 1] / vl);
        for (int i = 0; i < sides; ++i)
        {
            float u0 = float(i) / float(sides), u1 = float(i + 1) / float(sides);
            const Index face[4] = {idx(j, i), idx(j, i + 1), idx(j1, i + 1), idx(j1, i)};
            const glm::vec2 uv[4] = {{u0, v0}, {u1, v0}, {u1, v1}, {u0, v1}};
            addFaceUv(m, face, uv, false);
        }
    }
    // The start ring's cap runs against the quads beside it, the end ring's with them.
    auto cap = [&](size_t j, bool start) {
        std::vector<Index> pts;
        std::vector<glm::vec2> uvs;
        for (int i = 0; i < sides; ++i)
        {
            double a = phase + glm::two_pi<double>() * i / sides;
            pts.push_back(idx(j, i));
            uvs.push_back(glm::vec2(float(1.1 + 0.5 + 0.5 * std::cos(a)), float(0.5 + 0.5 * std::sin(a))));
        }
        addFaceUv(m, pts, uvs, start);
    };
    if (capStart)
        cap(0, true);
    if (capEnd)
        cap(n - 1, false);
    orientOutward(m);
    return m;
}

geo::Mesh flangePart(double ri, double ro, double L, int sides)
{
    ro = std::max(ro, ri * 1.05);
    double rh = ri + (ro - ri) * 0.35, hub = L * 0.5, g = L * 0.04, gd = (ro - rh) * 0.15;
    return lathe({{ri, -L / 2 - hub},
                  {rh, -L / 2 - hub},
                  {rh, -L / 2},
                  {ro, -L / 2},
                  {ro, -g},
                  {ro - gd, -g},
                  {ro - gd, g},
                  {ro, g},
                  {ro, L / 2},
                  {rh, L / 2},
                  {rh, L / 2 + hub},
                  {ri, L / 2 + hub}},
                 true, sides);
}

geo::Mesh collarPart(double ri, double ro, double L, int sides)
{
    ro = std::max(ro, ri * 1.02);
    double c = std::min(L * 0.2, (ro - ri) * 0.5);
    return lathe({{ri, -L / 2}, {ro - c, -L / 2}, {ro, -L / 2 + c}, {ro, L / 2 - c}, {ro - c, L / 2}, {ri, L / 2}}, true, sides);
}

geo::Mesh clampPart(double ri, double ro, double L, int sides, bool bolt)
{
    ro = std::max(ro, ri * 1.1);
    double band = ri + (ro - ri) * 0.25, w = L * 0.5;
    geo::Mesh m = lathe({{ri, -w / 2}, {band, -w / 2}, {band, w / 2}, {ri, w / 2}}, true, sides);
    double gap = (ro - ri) * 0.15, ear = (band - ri) * 1.5, x0 = band * 0.97, x1 = ro + (ro - ri) * 0.3;
    addBox(m, D3(x0, gap, -w / 2), D3(x1, gap + ear, w / 2));
    addBox(m, D3(x0, -gap - ear, -w / 2), D3(x1, -gap, w / 2));
    if (bolt)
    {
        double size = std::max(ear * 0.6, 1e-3), y0 = gap + ear;
        geo::Mesh b = hexBolt(size, 2.0 * y0 + nutHeight(size) * 1.3);
        geo::Mesh nut = hexNut(size);
        double x = (band + x1) * 0.5;
        // Bolt +z onto +y, head below the lower ear.
        for (Index p = 0; p < b.pointCapacity(); ++p)
            if (b.pointValid(p))
            {
                D3 q = b.position(p);
                b.setPosition(p, D3(q.x + x, q.z - y0, -q.y));
            }
        for (Index p = 0; p < nut.pointCapacity(); ++p)
            if (nut.pointValid(p))
            {
                D3 q = nut.position(p);
                nut.setPosition(p, D3(q.x + x, q.z + y0 + nutHeight(size) * 0.5, -q.y));
            }
        m.append(b);
        m.append(nut);
    }
    return m;
}

geo::Mesh hexBolt(double size, double length)
{
    double rh = hexHeadRadius(size) * 1.15, hh = size * 0.65, rs = size * 0.5;
    geo::Mesh m = lathe({{0.0, -hh}, {rh, -hh}, {rh, 0.0}, {0.0, 0.0}}, false, 6, glm::pi<double>() / 6.0);
    m.append(lathe({{0.0, 0.0}, {rs, 0.0}, {rs, length}, {0.0, length}}, false, 8));
    return m;
}

geo::Mesh hexNut(double size)
{
    double h = nutHeight(size), rh = hexHeadRadius(size) * 1.15;
    return lathe({{size * 0.5, -h / 2}, {rh, -h / 2}, {rh, h / 2}, {size * 0.5, h / 2}}, true, 6, glm::pi<double>() / 6.0);
}

PipeResult makePipe(const Spline& input, const PipeOptions& o)
{
    LILA_ZONE();

    PipeResult r;
    if (!isValid(input))
        return r;

    // A start loop sets the profile and the radius.
    const bool fromLoop = o.startLoop.size() >= 3;
    double R = std::max(o.radius, 1e-4);
    D3 loopCentre(0.0);
    if (fromLoop)
    {
        for (const D3& p : o.startLoop)
            loopCentre += p;
        loopCentre /= double(o.startLoop.size());
        double sum = 0.0;
        for (const D3& p : o.startLoop)
            sum += glm::distance(p, loopCentre);
        R = std::max(sum / double(o.startLoop.size()), 1e-4);
    }

    struct Corner
    {
        D3 p;
        double theta;
    };
    std::vector<Corner> corners;
    {
        std::vector<Piece> ps = pieces(input);
        size_t K = ps.size();
        for (size_t j = 0; j < K; ++j)
        {
            if (!input.closed && j == 0)
                continue;
            double theta = angleBetween(endTangent(ps[(j + K - 1) % K], true), endTangent(ps[j], false));
            if (theta > kCornerAngle && theta < glm::pi<double>() - 1e-3)
                corners.push_back({ps[j].start(), theta});
        }
    }
    Spline path = input;
    const double bendR = o.bendRadius > 0.0 ? o.bendRadius : 2.0 * R;
    if (o.autoFillet && !corners.empty())
    {
        FilletOptions fo;
        fo.radius = bendR;
        fo.minAngle = kCornerAngle;
        path = fillet(input, fo);
    }

    Polyline pl = tessellate(path, o.tess);
    if (pl.size() < 2)
        return r;
    const bool corrugated = o.corrugationDepth > 0.0 && o.corrugationPeriod > 1e-4;
    if (corrugated)
        pl = resample(pl, o.corrugationPeriod / std::max(2, o.corrugationRings));
    const bool swells = input.radius.size() == input.size() && std::ranges::adjacent_find(input.radius, std::not_equal_to<>()) != input.radius.end();
    // Rings close enough together to follow the swell, which straight spans would otherwise jump across in one cone.
    if (swells && pl.length > 1e-6)
    {
        double step = R * std::ranges::min(input.radius) * 0.5;
        for (size_t k = 0; k + 1 < input.size(); ++k)
            if (std::abs(input.radius[k + 1] - input.radius[k]) > 1e-9)
                step = std::min(step, glm::distance(input.points[k], input.points[k + 1]) * kSwellSoftness * 0.5);
        pl = resample(pl, std::max(step, pl.length / 1500.0));
    }
    std::vector<Frame> fr = frames(pl, o.frames, tiltAlong(path, pl));
    Path P = makePath(pl, fr, {});
    if (swells)
        P.scale = radiusAlong(P, input);
    const double L = P.closed ? P.length : P.s.back();
    r.length = L;

    // Bends: where each corner of the input went, and the half length of the arc that rounds it.
    struct Bend
    {
        double s, half;
    };
    std::vector<Bend> bends;
    for (const Corner& c : corners)
    {
        double d = 0.0;
        double s = nearestS(P, c.p, &d);
        double k = 1.0 / std::cos(c.theta * 0.5) - 1.0;
        double half = d > 1e-6 && k > 1e-9 ? d / k * c.theta * 0.5 : 0.0;
        bends.push_back({s, half});
    }

    // Tightest bend of the centreline, from the turn at each point.
    r.tightestBend = std::numeric_limits<double>::infinity();
    {
        size_t n = P.size();
        for (size_t i = 0; i < n; ++i)
        {
            if ((!P.closed && (i == 0 || i + 1 == n)) || P.kink[i])
                continue;
            const D3& a = P.points[(i + n - 1) % n];
            const D3& b = P.points[i];
            const D3& c = P.points[(i + 1) % n];
            double la = glm::distance(a, b), lc = glm::distance(b, c);
            if (la < kLengthEps || lc < kLengthEps)
                continue;
            double theta = angleBetween(b - a, c - b);
            if (theta > 1e-4)
                r.tightestBend = std::min(r.tightestBend, 0.5 * (la + lc) / (2.0 * std::sin(theta * 0.5)));
        }
    }
    const double minBend = o.minBendRadius > 0.0 ? o.minBendRadius : R;
    if (r.tightestBend < minBend * 0.999)
        r.warnings.push_back(std::format("tightest bend is {:.2f} cm, under the {:.2f} cm minimum: the inside of the bend folds over", r.tightestBend,
                                         minBend));
    if (o.autoFillet && bendR < R)
        r.warnings.push_back(std::format("bend radius {:.2f} cm is under the pipe radius {:.2f} cm", bendR, R));

    // Segments.
    std::vector<double> splits;
    if (o.splitEvery > 1e-4)
        for (double s = o.splitEvery; s < L - 1e-4; s += o.splitEvery)
            splits.push_back(s);
    if (o.splitAtBends)
        for (const Bend& b : bends)
            splits.push_back(b.s);
    for (double f : o.splitAt)
        if (f > 0.0 && f < 1.0)
            splits.push_back(f * L);
    std::ranges::sort(splits);
    splits.erase(std::unique(splits.begin(), splits.end(), [](double a, double b) { return std::abs(a - b) < 1e-4; }), splits.end());
    std::erase_if(splits, [&](double s) { return s < 1e-4 || s > L - 1e-4; });
    r.segmentStarts = {0.0};
    r.segmentStarts.insert(r.segmentStarts.end(), splits.begin(), splits.end());
    auto segmentAt = [&](double s) { return int(std::upper_bound(r.segmentStarts.begin(), r.segmentStarts.end(), s + 1e-6) - r.segmentStarts.begin()) - 1; };

    // Coupling stations: each split (a bend's split may take two, one each side of its arc) and optionally the ends.
    struct Slot
    {
        double s;
        bool bend;
        double half;
    };
    std::vector<Slot> slots;
    const double Lc = std::max(o.couplingLength, 1e-3);
    if (o.coupling != CouplingKind::None)
    {
        for (double s : splits)
        {
            auto b = std::ranges::find_if(bends, [&](const Bend& bd) { return std::abs(bd.s - s) < 1e-4; });
            if (b == bends.end())
                slots.push_back({s, false, 0.0});
            else if (o.coupling == CouplingKind::ElbowCollar || b->half < Lc * 0.5)
                slots.push_back({s, true, b->half});
            else
            {
                slots.push_back({s - b->half - Lc * 0.5, false, 0.0});
                slots.push_back({s + b->half + Lc * 0.5, false, 0.0});
            }
        }
        if (o.partAtStart && !P.closed)
            slots.push_back({std::min(Lc * 0.5, L * 0.5), false, 0.0});
        if (o.partAtEnd && !P.closed)
            slots.push_back({std::max(L - Lc * 0.5, L * 0.5), false, 0.0});
    }

    auto ramp = o.radiusRamp;
    auto rampAt = [ramp, L](double s) { return ramp ? ramp(L > 0.0 ? s / L : 0.0) : 1.0; };
    std::vector<std::pair<double, double>> calm;
    for (const Slot& sl : slots)
        calm.push_back({sl.s - sl.half - Lc * 0.6, sl.s + sl.half + Lc * 0.6});
    const double depth = std::clamp(o.corrugationDepth, 0.0, 0.9), period = o.corrugationPeriod;
    auto corrugation = [=](double s) {
        if (!corrugated)
            return 1.0;
        for (auto [a, b] : calm)
            if (s >= a && s <= b)
                return 1.0;
        return 1.0 - depth * 0.5 * (1.0 - std::cos(glm::two_pi<double>() * s / period));
    };

    // Profile.
    Profile prof;
    std::vector<size_t> loopOrder;
    if (fromLoop)
    {
        const Frame& f0 = P.frames[0];
        const D3 o0 = P.points[0];
        for (size_t j = 0; j < o.startLoop.size(); ++j)
        {
            D3 d = o.startLoop[j] - o0;
            prof.points.push_back(D3(glm::dot(d, f0.n), glm::dot(d, f0.b), 0.0));
            loopOrder.push_back(j);
        }
        // Counter-clockwise about the tangent already, so the sweep keeps the loop's order.
        if (prof.signedArea() < 0.0)
        {
            std::reverse(prof.points.begin() + 1, prof.points.end());
            std::reverse(loopOrder.begin() + 1, loopOrder.end());
        }
    }
    else
        prof = circleProfile(o.sides, R);

    const bool hollow = o.wall > 1e-6 && o.wall < R * 0.98;
    SweepOptions so;
    so.scale = [=](double s) { return rampAt(s) * corrugation(s); };
    so.startCap = hollow ? CapMode::None : o.startCap;
    so.endCap = hollow ? CapMode::None : o.endCap;
    so.roundSegments = std::max(1, o.roundSegments);
    so.uv = o.uv;
    so.splits = splits;
    so.flip = o.flip;
    if (o.segmentMaterials)
        for (size_t k = 0; k <= splits.size(); ++k)
            so.partMaterials.push_back(int(k % 2));
    SweepResult outer = sweep(prof, P, so);
    geo::Mesh& m = r.mesh;
    m = std::move(outer.mesh);
    r.rings = outer.rings.size();
    if (outer.rings.empty())
        return r;
    if (fromLoop)
    {
        r.startRing.assign(o.startLoop.size(), geo::kInvalid);
        Index first = outer.rings.front().firstPoint;
        for (size_t j = 0; j < loopOrder.size(); ++j)
        {
            m.setPosition(first + Index(j), o.startLoop[loopOrder[j]]);
            r.startRing[loopOrder[j]] = first + Index(j);
        }
    }

    if (hollow)
    {
        Profile inner = prof;
        D2 c(0.0);
        for (const D3& p : prof.points)
            c += D2(p);
        c /= double(prof.size());
        const double k = (R - o.wall) / R;
        for (D3& p : inner.points)
            p = D3(c + (D2(p) - c) * k, p.z);
        SweepOptions si = so;
        si.flip = !o.flip;
        si.uv.rect = o.innerUv;
        SweepResult in = sweep(inner, P, si);
        Index off = m.append(in.mesh);
        if (!P.closed)
        {
            const size_t np = prof.size();
            std::vector<std::pair<Index, Index>> hard;
            for (int end = 0; end < 2; ++end)
            {
                const SweepRing& ro = end ? outer.rings.back() : outer.rings.front();
                const SweepRing& ri = end ? in.rings.back() : in.rings.front();
                Index O = ro.firstPoint, I = ri.firstPoint + off;
                for (size_t j = 0; j < np; ++j)
                {
                    Index j1 = Index((j + 1) % np);
                    double a0 = glm::two_pi<double>() * double(j) / double(np), a1 = glm::two_pi<double>() * double(j + 1) / double(np);
                    const UvRect& cr = o.uv.caps;
                    std::vector<glm::vec2> uv = {cr.at(0.5 + 0.5 * std::cos(a0), 0.5 + 0.5 * std::sin(a0)),
                                                 cr.at(0.5 + 0.5 * std::cos(a1), 0.5 + 0.5 * std::sin(a1)),
                                                 cr.at(0.5 + 0.5 * k * std::cos(a1), 0.5 + 0.5 * k * std::sin(a1)),
                                                 cr.at(0.5 + 0.5 * k * std::cos(a0), 0.5 + 0.5 * k * std::sin(a0))};
                    // Counter-clockwise about +t at the end ring; the start rim faces back.
                    const Index face[4] = {O + Index(j), O + j1, I + j1, I + Index(j)};
                    Index f = addFaceUv(m, face, uv, (end == 0) != o.flip);
                    setFaceInt(m, "part", f, ro.part);
                    if (size_t(ro.part) < so.partMaterials.size())
                        setFaceInt(m, geo::attr::kMaterial, f, so.partMaterials[size_t(ro.part)]);
                    hard.push_back({O + Index(j), O + j1});
                    hard.push_back({I + Index(j), I + j1});
                }
            }
            auto h = m.edgeAttrs().getOrAdd<bool>(geo::attr::kHard);
            for (auto [a, b] : hard)
                if (Index e = m.findEdge(a, b); e != geo::kInvalid && e < h.size())
                    h[e] = true;
        }
    }
    setAllFaces(m, "pipePart", int(PipePart::Body));

    // Parts.
    const int partMaterial = o.segmentMaterials ? 2 : -1;
    const int cSides = o.couplingSides >= 3 ? o.couplingSides : std::max(3, fromLoop ? int(prof.size()) : o.sides);
    for (const Slot& sl : slots)
    {
        Station st = sampleAt(P, sl.s);
        const double radiusHere = o.scaleToRadius ? R * st.scale * rampAt(sl.s) : R;
        const double ratio = radiusHere / R;
        const double ro = radiusHere * std::max(1.0, o.couplingScale);
        const int seg = segmentAt(sl.s);
        if (sl.bend && sl.half > 1e-6 && o.coupling == CouplingKind::ElbowCollar)
        {
            Path sub = subPath(P, sl.s - sl.half - Lc * 0.5, sl.s + sl.half + Lc * 0.5);
            if (sub.size() >= 2)
            {
                SweepOptions eo;
                eo.startCap = eo.endCap = CapMode::Grid;
                eo.uv = o.uv;
                geo::Mesh sleeve = sweep(circleProfile(cSides, ro), sub, eo).mesh;
                place(m, std::move(sleeve), D3(0.0), Frame{D3(0, 0, 1), D3(1, 0, 0), D3(0, 1, 0)}, PipePart::Coupling, seg, partMaterial);
                r.placements.push_back({sl.s, st.origin, st.frame, radiusHere, PipePart::Coupling, true});
            }
            continue;
        }
        geo::Mesh part;
        switch (o.coupling)
        {
        case CouplingKind::Flange: part = flangePart(radiusHere, ro, Lc, cSides); break;
        case CouplingKind::Clamp: part = clampPart(radiusHere, ro, Lc, cSides, o.fasteners != FastenerKind::None); break;
        case CouplingKind::Mesh:
            if (o.couplingMesh)
                part = userPart(*o.couplingMesh, o.meshAxis, o.meshScale * ratio);
            break;
        default: part = collarPart(radiusHere, ro, Lc, cSides); break;
        }
        if (part.numFaces() == 0)
            continue;
        if (o.coupling != CouplingKind::Mesh)
            partUvs(part, o.uv);
        place(m, std::move(part), st.origin, st.frame, PipePart::Coupling, seg, partMaterial);
        r.placements.push_back({sl.s, st.origin, st.frame, radiusHere, PipePart::Coupling, sl.bend});

        // A bolt ring through flanges (and user couplings).
        const bool ringed = o.coupling == CouplingKind::Flange || o.coupling == CouplingKind::Mesh;
        if (!ringed || o.fasteners == FastenerKind::None || o.boltCount < 1)
            continue;
        const double rh = radiusHere + (ro - radiusHere) * 0.35;
        const double circle = o.boltCircle > 0.0 ? o.boltCircle * ratio : (rh + ro) * 0.5;
        const double size = o.boltSize > 0.0 ? o.boltSize * ratio : std::max(0.02, (ro - rh) * 0.3);
        const double nh = nutHeight(size);
        geo::Mesh fastener;
        switch (o.fasteners)
        {
        case FastenerKind::BoltAndNut:
        {
            fastener = hexBolt(size, Lc + nh * 1.3);
            geo::Mesh nut = hexNut(size);
            for (D3& p : nut.positions())
                p.z += Lc + nh * 0.5;
            fastener.append(nut);
            for (D3& p : fastener.positions())
                p.z -= Lc * 0.5;
            break;
        }
        case FastenerKind::Bolt:
            fastener = hexBolt(size, Lc + nh);
            for (D3& p : fastener.positions())
                p.z -= Lc * 0.5;
            break;
        case FastenerKind::Nut:
        {
            fastener = hexNut(size);
            geo::Mesh other = fastener;
            for (D3& p : fastener.positions())
                p.z -= Lc * 0.5 + nh * 0.5;
            for (D3& p : other.positions())
                p.z += Lc * 0.5 + nh * 0.5;
            fastener.append(other);
            break;
        }
        case FastenerKind::Mesh:
            if (o.boltMesh)
                fastener = userPart(*o.boltMesh, o.meshAxis, o.meshScale * ratio);
            break;
        default: break;
        }
        if (fastener.numFaces() == 0)
            continue;
        if (o.fasteners != FastenerKind::Mesh)
            partUvs(fastener, o.uv);
        for (int b = 0; b < o.boltCount; ++b)
        {
            double a = o.boltPhase + glm::two_pi<double>() * b / o.boltCount;
            D3 at = st.origin + (st.frame.n * std::cos(a) + st.frame.b * std::sin(a)) * circle;
            place(m, fastener, at, st.frame, PipePart::Fastener, seg, partMaterial);
            r.placements.push_back({sl.s, at, st.frame, radiusHere, PipePart::Fastener, sl.bend});
        }
    }
    if (o.segmentMaterials)
        m.materialNames() = {"pipeBody", "pipeBodyAlt", "pipeParts"};
    return r;
}
} // namespace lila::curve
