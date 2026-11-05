#pragma once

#include "core/Export.h"

#include "scene/SoftSelect.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

/**
 * The brush core every painting and sculpting tool shares: settings (radius, spacing, flow, opacity, falloff, pressure
 * curves, jitter, tip angle, roundness and alpha), stroke sampling (brush/Stroke.h) and symmetry (brush/Symmetry.h).
 * No Qt; the tablet input and cursor live in brush/BrushUi.h.
 */
namespace lila::brush
{
using json = nlohmann::json;

/** Falloff shapes. The first six are sculpt's, in its order; Bell is (1 - t^2)^2; Curve uses a soft select curve. */
enum class Shape : uint8_t
{
    Smooth,
    Sphere,
    Root,
    Sharp,
    Linear,
    Constant,
    Bell,
    Curve,
    Count
};

LILA_SCENE_API std::string_view shapeName(Shape s);
LILA_SCENE_API std::optional<Shape> shapeFromName(std::string_view name);
/** Weight at t = distance / radius: 1 at the centre, 0 from the rim on. Curve gives the default soft select curve. */
LILA_SCENE_API double shapeWeight(Shape s, double t);

struct LILA_SCENE_API Falloff
{
    Shape shape = Shape::Smooth;
    /** Used when shape is Curve. */
    scene::softSelect::Curve curve;

    Falloff() = default;
    Falloff(Shape s) : shape(s) {}
    Falloff(scene::softSelect::Curve c) : shape(Shape::Curve), curve(std::move(c)) {}

    double evaluate(double t) const;
    /**
     * A shape name ("smooth", "bell", ...), else a soft select preset (built-in, then user) by name as a curve. Unknown
     * names give the default soft select curve.
     */
    static Falloff fromName(std::string_view name);
    /** The same, nullopt for unknown names. */
    static std::optional<Falloff> find(std::string_view name);
};

/** A falloff sampled for fast lookups, linearly interpolated. */
class LILA_SCENE_API FalloffTable
{
public:
    FalloffTable() : FalloffTable(Falloff()) {}
    explicit FalloffTable(const Falloff& f);
    /** t = distance / radius; 0 from 1 on. */
    float operator()(double t) const;

private:
    std::array<float, 257> mValues{};
};

/** Maps pen pressure to a multiplier: floor + (1 - floor) * pressure^exponent, or 1 when off. */
struct LILA_SCENE_API PressureCurve
{
    bool enabled = false;
    double floor = 0.0;
    /** Below 1 a light touch already reaches far; above 1 it takes a firm press. */
    double exponent = 1.0;

    double map(double pressure) const;
    bool operator==(const PressureCurve&) const = default;
};

/** Grayscale brush tip, sampled over the tip square: uv (0, 0) one corner, (1, 1) the opposite. */
struct LILA_SCENE_API AlphaImage
{
    int width = 0;
    int height = 0;
    std::vector<float> values;

    /** Bilinear; 0 outside the square. */
    float sample(glm::dvec2 uv) const;
    /** Luminance times alpha of an image file (anything image::read takes); null when it can't be read. */
    static std::shared_ptr<const AlphaImage> load(const std::string& path, std::string* error = nullptr);
};

/** Random variation per dab, deterministic for a stroke's seed. */
struct Jitter
{
    /** Position scatter, as a fraction of the radius. */
    double position = 0.0;
    /** Size and opacity drop by up to this fraction. */
    double size = 0.0;
    double opacity = 0.0;
    /** Degrees either way. */
    double angle = 0.0;

    bool any() const { return position > 0.0 || size > 0.0 || opacity > 0.0 || angle > 0.0; }
    bool operator==(const Jitter&) const = default;
};

/** Copies of each dab turned about an object axis through the object's origin. */
struct Radial
{
    /** 1 is off. */
    int count = 1;
    /** 0 x, 1 y, 2 z. */
    int axis = 1;

    bool any() const { return count > 1; }
    bool operator==(const Radial&) const = default;
};

enum class RadiusUnit : uint8_t
{
    Screen,
    World,
};

enum class StrokeMode : uint8_t
{
    Freehand,
    /** Dabs from the press to the release in a straight line. */
    Line,
};

struct LILA_SCENE_API BrushSettings
{
    RadiusUnit radiusUnit = RadiusUnit::Screen;
    /** Screen pixels. */
    double radius = 50.0;
    /** World units (cm), when radiusUnit is World. */
    double worldRadius = 5.0;
    /** Most one stroke does, 0 to 1. */
    double strength = 1.0;
    /** Per dab amount, 0 to 1. */
    double flow = 1.0;
    /** Dab spacing as a fraction of the radius. */
    double spacing = 0.15;
    /** A shape name or soft select preset name, see Falloff::fromName. */
    std::string falloff = "smooth";

    PressureCurve pressureSize;
    PressureCurve pressureOpacity;
    PressureCurve pressureFlow;
    Jitter jitter;

    /** Tip angle in degrees, counterclockwise on screen; with followStroke it is relative to the direction of travel. */
    double angle = 0.0;
    bool followStroke = false;
    /** 1 round, toward 0 a flat ellipse across the angle. */
    double roundness = 1.0;
    /** Image file shaping the tip, or a library alpha (alpha://source/file, see brush/Alphas.h); empty for none. */
    std::string alphaPath;
    std::shared_ptr<const AlphaImage> alpha;

    StrokeMode mode = StrokeMode::Freehand;
    /** Lazy mouse: the brush trails the cursor by this many pixels, so strokes come out steady. 0 is off. */
    double lazyRadius = 0.0;
    /** 0 to 1: how much each move eases toward the cursor instead of jumping to it. */
    double smoothing = 0.0;

    bool mirror = false;
    Radial radial;
    uint32_t seed = 0;

    Falloff resolveFalloff() const { return Falloff::fromName(falloff); }
    /** Loads alpha from alphaPath when it is set and not loaded yet. */
    void ensureAlpha();

    json toJson() const;
    /** Keys missing from j keep the values in `defaults`. */
    static BrushSettings fromJson(const json& j);
    static BrushSettings fromJson(const json& j, const BrushSettings& defaults);

    /**
     * Reads and writes the settings as keys under prefix (e.g. "/vertexPaint") in the general prefs: radius, strength,
     * spacing, falloff, pressureSize and pressureOpacity as plain values (what the tools stored before sharing this),
     * the rest as more keys beside them.
     */
    void loadPrefs(std::string_view prefix);
    void savePrefs(std::string_view prefix) const;
};

/** One brush stamp, in the painted mesh's object space. Sinks read what they need and ignore the rest. */
struct Dab
{
    glm::dvec3 center{0.0};
    /** Surface normal under the dab, unit length. */
    glm::dvec3 normal{0.0, 1.0, 0.0};
    double radius = 1.0;
    /** Raw pen pressure, 0 to 1 (1 for a mouse). */
    double pressure = 1.0;
    /** Weight multiplier: the pressure curve, jitter. */
    double opacity = 1.0;
    /** Per dab amount for brushes that build up. */
    double flow = 1.0;
    /** Into the screen; zero disables facing tests. */
    glm::dvec3 viewDir{0.0};
    /** Camera position, for occlusion tests. */
    glm::dvec3 eye{0.0};
    /** Grab-style brushes: how far the cursor moved since the last dab. */
    glm::dvec3 delta{0.0};
    /** Direction of travel since the last dab; zero when unknown. */
    glm::dvec3 direction{0.0};
    /** The tip's long axis on the surface, unit length; zero for a round tip. */
    glm::dvec3 tangent{0.0};
    /** Tip angle on screen, radians. */
    double angle = 0.0;
    double roundness = 1.0;
    /** Shared, not copied: one per stroke. */
    std::shared_ptr<const AlphaImage> alpha;
    /** Where the dab was on screen, widget pixels. */
    glm::dvec2 screen{0.0};
    /** Order in the stroke. */
    uint32_t index = 0;
    /** 0 for the dab itself, k for its k-th symmetry copy. */
    uint32_t copy = 0;
};

/**
 * The weight a dab gives a point at rel = point - dab.center: the falloff over the distance, squeezed across the tip
 * angle by roundness and multiplied by the alpha. A round tip without alpha is exactly falloff(|rel| / radius).
 */
LILA_SCENE_API float tipWeight(const Dab& d, const FalloffTable& falloff, const glm::dvec3& rel);
/** True when tipWeight is plain radial falloff. */
inline bool roundTip(const Dab& d) { return d.roundness >= 1.0 && !d.alpha; }
} // namespace lila::brush
