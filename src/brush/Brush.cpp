#include "brush/Brush.h"

#include "brush/Alphas.h"
#include "core/ImageIO.h"
#include "core/PrefStore.h"

#include <algorithm>
#include <cmath>

namespace lila::brush
{
namespace
{
constexpr std::array<std::string_view, size_t(Shape::Count)> kShapeNames = {"smooth", "sphere", "root",  "sharp",
                                                                            "linear", "constant", "bell", "curve"};
constexpr const char* kPrefs = "general";

/** Every key toJson writes, for loading them back from prefs one by one. */
constexpr std::string_view kKeys[] = {"radiusUnit", "radius", "worldRadius", "strength", "flow", "spacing", "falloff",
                                      "pressureSize", "pressureOpacity", "pressureFlow", "pressureCurve", "jitter",
                                      "angle", "followStroke", "roundness", "alpha", "strokeMode", "lazyRadius",
                                      "smoothing", "mirror", "radial", "seed"};

json curveJson(const PressureCurve& c) { return {{"floor", c.floor}, {"exponent", c.exponent}}; }

void readCurve(const json& j, PressureCurve& c)
{
    if (!j.is_object())
        return;
    c.floor = std::clamp(j.value("floor", c.floor), 0.0, 1.0);
    c.exponent = std::clamp(j.value("exponent", c.exponent), 0.05, 20.0);
}

template <class T> void read(const json& j, const char* key, T& out)
{
    if (auto it = j.find(key); it != j.end() && !it->is_null())
        try
        {
            out = it->get<T>();
        }
        catch (const json::exception&)
        {
        }
}
} // namespace

std::string_view shapeName(Shape s) { return s < Shape::Count ? kShapeNames[size_t(s)] : std::string_view(); }

std::optional<Shape> shapeFromName(std::string_view name)
{
    for (size_t i = 0; i < kShapeNames.size(); ++i)
        if (kShapeNames[i] == name)
            return Shape(i);
    return std::nullopt;
}

double shapeWeight(Shape s, double t)
{
    if (t >= 1.0)
        return 0.0;
    t = std::max(t, 0.0);
    const double u = 1.0 - t;
    switch (s)
    {
    case Shape::Smooth: return 3.0 * u * u - 2.0 * u * u * u;
    case Shape::Sphere: return std::sqrt(1.0 - t * t);
    case Shape::Root: return std::sqrt(u);
    case Shape::Sharp: return u * u;
    case Shape::Linear: return u;
    case Shape::Constant: return 1.0;
    case Shape::Bell: return (1.0 - t * t) * (1.0 - t * t);
    case Shape::Curve: return std::clamp(scene::softSelect::Curve().evaluate(t), 0.0, 1.0);
    case Shape::Count: break;
    }
    return 0.0;
}

double Falloff::evaluate(double t) const
{
    if (shape != Shape::Curve)
        return shapeWeight(shape, t);
    return t >= 1.0 ? 0.0 : std::clamp(curve.evaluate(std::max(t, 0.0)), 0.0, 1.0);
}

std::optional<Falloff> Falloff::find(std::string_view name)
{
    if (auto s = shapeFromName(name); s && *s != Shape::Curve)
        return Falloff(*s);
    if (const auto* p = scene::softSelect::findBuiltin(name))
        return Falloff(p->curve);
    for (const auto& p : scene::softSelect::userPresets())
        if (p.name == name)
            return Falloff(p.curve);
    return std::nullopt;
}

Falloff Falloff::fromName(std::string_view name) { return find(name).value_or(Falloff(scene::softSelect::Curve())); }

FalloffTable::FalloffTable(const Falloff& f)
{
    for (size_t i = 0; i + 1 < mValues.size(); ++i)
        mValues[i] = float(std::clamp(f.evaluate(double(i) / double(mValues.size() - 1)), 0.0, 1.0));
    // The rim itself: curves may stay above zero there, and dabs reach exactly radius.
    mValues.back() = float(std::clamp(f.shape == Shape::Curve ? f.curve.evaluate(1.0) : shapeWeight(f.shape, 1.0), 0.0, 1.0));
}

float FalloffTable::operator()(double t) const
{
    if (!(t >= 0.0))
        t = 0.0;
    if (t > 1.0)
        return 0.0f;
    const float x = float(t) * float(mValues.size() - 1);
    const size_t k = std::min(size_t(x), mValues.size() - 2);
    return mValues[k] + (mValues[k + 1] - mValues[k]) * (x - float(k));
}

double PressureCurve::map(double pressure) const
{
    if (!enabled)
        return 1.0;
    const double p = std::clamp(pressure, 0.0, 1.0);
    return floor + (1.0 - floor) * std::pow(p, std::max(exponent, 1e-3));
}

float AlphaImage::sample(glm::dvec2 uv) const
{
    if (width <= 0 || height <= 0 || uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0)
        return 0.0f;
    const double x = uv.x * (width - 1), y = uv.y * (height - 1);
    const int x0 = std::min(int(x), width - 1), y0 = std::min(int(y), height - 1);
    const int x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
    const float fx = float(x - x0), fy = float(y - y0);
    auto at = [&](int px, int py) { return values[size_t(py) * size_t(width) + size_t(px)]; };
    const float top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
    const float bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
    return top + (bottom - top) * fy;
}

std::shared_ptr<const AlphaImage> AlphaImage::load(const std::string& path, std::string* error)
{
    image::Image img = image::read(path, error);
    if (img.width <= 0 || img.height <= 0)
        return nullptr;
    auto out = std::make_shared<AlphaImage>();
    out->width = img.width;
    out->height = img.height;
    out->values.resize(size_t(img.width) * size_t(img.height));
    for (size_t i = 0; i < out->values.size(); ++i)
    {
        const float* p = &img.rgba[i * 4];
        out->values[i] = std::clamp((0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]) * p[3], 0.0f, 1.0f);
    }
    return out;
}

void BrushSettings::ensureAlpha()
{
    if (alphaPath.empty())
        alpha.reset();
    else if (!alpha)
        alpha = AlphaImage::load(alphas::resolve(alphaPath));
}

json BrushSettings::toJson() const
{
    return {{"radiusUnit", radiusUnit == RadiusUnit::World ? "world" : "screen"},
            {"radius", radius},
            {"worldRadius", worldRadius},
            {"strength", strength},
            {"flow", flow},
            {"spacing", spacing},
            {"falloff", falloff},
            {"pressureSize", pressureSize.enabled},
            {"pressureOpacity", pressureOpacity.enabled},
            {"pressureFlow", pressureFlow.enabled},
            {"pressureCurve", {{"size", curveJson(pressureSize)}, {"opacity", curveJson(pressureOpacity)}, {"flow", curveJson(pressureFlow)}}},
            {"jitter", {{"position", jitter.position}, {"size", jitter.size}, {"opacity", jitter.opacity}, {"angle", jitter.angle}}},
            {"angle", angle},
            {"followStroke", followStroke},
            {"roundness", roundness},
            {"alpha", alphaPath},
            {"strokeMode", mode == StrokeMode::Line ? "line" : "freehand"},
            {"lazyRadius", lazyRadius},
            {"smoothing", smoothing},
            {"mirror", mirror},
            {"radial", {{"count", radial.count}, {"axis", radial.axis}}},
            {"seed", seed}};
}

BrushSettings BrushSettings::fromJson(const json& j)
{
    return fromJson(j, BrushSettings{});
}

BrushSettings BrushSettings::fromJson(const json& j, const BrushSettings& defaults)
{
    BrushSettings s = defaults;
    if (!j.is_object())
        return s;
    std::string unit = s.radiusUnit == RadiusUnit::World ? "world" : "screen";
    read(j, "radiusUnit", unit);
    s.radiusUnit = unit == "world" ? RadiusUnit::World : RadiusUnit::Screen;
    read(j, "radius", s.radius);
    read(j, "worldRadius", s.worldRadius);
    read(j, "strength", s.strength);
    read(j, "flow", s.flow);
    read(j, "spacing", s.spacing);
    read(j, "falloff", s.falloff);
    read(j, "pressureSize", s.pressureSize.enabled);
    read(j, "pressureOpacity", s.pressureOpacity.enabled);
    read(j, "pressureFlow", s.pressureFlow.enabled);
    if (auto it = j.find("pressureCurve"); it != j.end() && it->is_object())
    {
        readCurve(it->value("size", json()), s.pressureSize);
        readCurve(it->value("opacity", json()), s.pressureOpacity);
        readCurve(it->value("flow", json()), s.pressureFlow);
    }
    if (auto it = j.find("jitter"); it != j.end() && it->is_object())
    {
        read(*it, "position", s.jitter.position);
        read(*it, "size", s.jitter.size);
        read(*it, "opacity", s.jitter.opacity);
        read(*it, "angle", s.jitter.angle);
    }
    read(j, "angle", s.angle);
    read(j, "followStroke", s.followStroke);
    read(j, "roundness", s.roundness);
    std::string alpha = s.alphaPath;
    read(j, "alpha", alpha);
    if (alpha != s.alphaPath)
    {
        s.alphaPath = alpha;
        s.alpha.reset();
    }
    std::string mode = s.mode == StrokeMode::Line ? "line" : "freehand";
    read(j, "strokeMode", mode);
    s.mode = mode == "line" ? StrokeMode::Line : StrokeMode::Freehand;
    read(j, "lazyRadius", s.lazyRadius);
    read(j, "smoothing", s.smoothing);
    read(j, "mirror", s.mirror);
    if (auto it = j.find("radial"); it != j.end() && it->is_object())
    {
        read(*it, "count", s.radial.count);
        read(*it, "axis", s.radial.axis);
    }
    read(j, "seed", s.seed);

    s.radius = std::clamp(s.radius, 1.0, 5000.0);
    s.worldRadius = std::clamp(s.worldRadius, 1e-4, 1e6);
    s.strength = std::clamp(s.strength, 0.0, 1.0);
    s.flow = std::clamp(s.flow, 0.0, 1.0);
    s.spacing = std::clamp(s.spacing, 0.01, 4.0);
    s.jitter.position = std::clamp(s.jitter.position, 0.0, 4.0);
    s.jitter.size = std::clamp(s.jitter.size, 0.0, 1.0);
    s.jitter.opacity = std::clamp(s.jitter.opacity, 0.0, 1.0);
    s.jitter.angle = std::clamp(s.jitter.angle, 0.0, 180.0);
    s.roundness = std::clamp(s.roundness, 0.01, 1.0);
    s.lazyRadius = std::clamp(s.lazyRadius, 0.0, 1000.0);
    s.smoothing = std::clamp(s.smoothing, 0.0, 0.99);
    s.radial.count = std::clamp(s.radial.count, 1, 64);
    s.radial.axis = std::clamp(s.radial.axis, 0, 2);
    return s;
}

void BrushSettings::loadPrefs(std::string_view prefix)
{
    const PrefStore& p = PrefStore::get();
    json j = json::object();
    for (std::string_view k : kKeys)
    {
        json v = p.value(kPrefs, std::string(prefix) + "/" + std::string(k), json(nullptr));
        if (!v.is_null())
            j[std::string(k)] = std::move(v);
    }
    *this = fromJson(j, *this);
}

void BrushSettings::savePrefs(std::string_view prefix) const
{
    PrefStore& p = PrefStore::get();
    const json all = toJson();
    for (auto& [k, v] : all.items())
        if (p.value(kPrefs, std::string(prefix) + "/" + k, json(nullptr)) != v)
            p.set(kPrefs, std::string(prefix) + "/" + k, v);
}

float tipWeight(const Dab& d, const FalloffTable& falloff, const glm::dvec3& rel)
{
    const double r = std::max(d.radius, 1e-12);
    if (roundTip(d) || glm::dot(d.tangent, d.tangent) < 0.5)
        return falloff(glm::length(rel) / r);
    const glm::dvec3 n = d.normal;
    const glm::dvec3 u = d.tangent;
    const glm::dvec3 v = glm::cross(n, u);
    const double a = glm::dot(rel, u), b = glm::dot(rel, v), c = glm::dot(rel, n);
    const double across = b / std::max(d.roundness, 1e-3);
    float w = falloff(std::sqrt(a * a + across * across + c * c) / r);
    if (w > 0.0f && d.alpha)
        w *= d.alpha->sample(glm::dvec2(a / r, -across / r) * 0.5 + 0.5);
    return w;
}
} // namespace lila::brush
