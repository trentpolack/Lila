#include "ui/ViewportWidgetDetail.h"

#include "scene/LightingChannels.h"
#include "scene/SceneIndex.h"
#include "scene/Turntable.h"
#include "scene/ViewState.h"
#include "ui/CameraBookmarks.h"
#include <QPlatformSurfaceEvent>

namespace lila::ui
{
namespace
{
scene::viewState::ActiveCamera activeCameraOf(const Camera& camera, double aspect)
{
    scene::viewState::ActiveCamera c;
    c.world = glm::inverse(camera.view());
    c.orthographic = camera.ortho;
    const glm::dvec2 extents = camera.viewExtents(aspect);
    if (camera.ortho)
        c.orthoSize = extents * 2.0;
    else
    {
        c.horizontalFov = 2.0 * std::atan(extents.x);
        c.verticalFov = 2.0 * std::atan(extents.y);
    }
    return c;
}
} // namespace

using namespace viewportDetail;

ViewportWidget::ViewportWidget(QWidget* parent) : QWidget(parent)
{
    setAttribute(Qt::WA_NativeWindow);
    setAttribute(Qt::WA_PaintOnScreen);
    setAttribute(Qt::WA_NoSystemBackground);
    setAttribute(Qt::WA_OpaquePaintEvent);
    setFocusPolicy(Qt::StrongFocus);
    setMouseTracking(true); // manipulator hover
    setCursor(Qt::ArrowCursor);
    setMinimumSize(64, 64);
    setProperty("hotkeyContext", "viewport");
    deletion::claimScene(this);
    setAcceptDrops(true);
    static int panels = 0;
    mHudPanel = panels++;
    viewportRenderers::setRedraw(mHudPanel, [this] { requestFrame(); });
    viewportRenderers::setRefreshRate(mHudPanel, [this] { return screen() ? screen()->refreshRate() : 0.0; });

    mCamera.zUp = PrefStore::get().value<std::string>("general", "/scene/upAxis", "y") == "z";

    mGraphListener = scene::Graph::current().addListener([this](scene::Change c, scene::NodeId id) {
        // Synchronous, so a scene load that clears and then restores the view's camera ends on the restored one.
        if (mLookThrough != scene::kNoNode && (c == scene::Change::Cleared || (c == scene::Change::Deleted && id == mLookThrough)))
        {
            mCameraEdit.reset();
            setView(mReturnView);
        }
        if (c == scene::Change::Cleared && mIsolation.active())
            mIsolation.clear();
        // Nodes made while the active view isolates join it, so a curve drawn or primitive placed there stays visible.
        // Undo/redo restoring a node is not making one.
        if (c == scene::Change::Created && mIsolation.active() && scene::isolation::active() == &mIsolation && !UndoStack::get().replaying())
        {
            mIsolation.nodes.insert(id);
            mIsolation.touch();
        }
        requestFrame();
    });
    mWheelCommit = new QTimer(this);
    mWheelCommit->setSingleShot(true);
    mWheelCommit->setInterval(500);
    connect(mWheelCommit, &QTimer::timeout, this, [this] { commitCameraEdit(); });
    // Kept mesh draws are rebuilt after anything but a camera move or a graph edit, which they track per node.
    mToolListener = tools::addListener([this] {
        ++mDisplayEpoch;
        requestFrame();
    });
    mSnapSurfaceListener = surfaceSnap::addListener([this] { requestFrame(); });
    mComponentListener = scene::components::addListener([this] { requestFrame(); });
    // Display layers hide on playback: starting or stopping changes what is drawn without changing the time.
    mPlaybackListener = scene::anim::addPlaybackListener([this] {
        ++mDisplayEpoch;
        requestFrame();
    });
    mViewportToolListener = viewportTools::addListener([this] {
        ++mDisplayEpoch;
        requestFrame();
    });
    mGizmoListener = gizmos::addListener([this] {
        syncGizmos();
        updateCursor();
        requestFrame();
    });
    mArrangeListener = bake::arrange::addListener([this] {
        ++mDisplayEpoch;
        requestFrame();
    });
    mInheritListener = scene::isolation::addInheritListener([this](const std::vector<scene::NodeId>& made, const std::vector<scene::NodeId>& from) {
        if (scene::isolation::inheritInto(mIsolation, made, from))
            setIsolation(std::move(mIsolation));
    });
    mHandlerListener = selectionHandlers::addListener([this] {
        mHandlerHover.reset();
        requestFrame();
    });
    // Changing the time only dirties the graph; nothing else asks for a frame.
    onTimeChanged(this, [this] { requestFrame(); });
    mRenderSettingsListener = scene::addRenderSettingsListener([this] {
        ++mDisplayEpoch;
        syncCameraSettings();
        requestFrame();
    });
    syncCameraSettings();
    mPrefListener = PrefStore::get().onChanged([this](const std::string& file) {
        ++mDisplayEpoch;
        if (file != "general")
            return;
        mCamera.zUp = PrefStore::get().value<std::string>("general", "/scene/upAxis", "y") == "z";
        // Display modes, toggles and environment settings all live in this file.
        requestFrame();
    });
}

ViewportWidget::~ViewportWidget()
{
    cursorBadges::publish(this, {});
    viewportRenderers::release(mHudPanel);
    scene::Graph::current().removeListener(mGraphListener);
    PrefStore::get().removeOnChanged(mPrefListener);
    tools::removeListener(mToolListener);
    surfaceSnap::removeListener(mSnapSurfaceListener);
    scene::components::removeListener(mComponentListener);
    scene::anim::removePlaybackListener(mPlaybackListener);
    viewportTools::removeListener(mViewportToolListener);
    gizmos::removeListener(mGizmoListener);
    selectionHandlers::removeListener(mHandlerListener);
    bake::arrange::removeListener(mArrangeListener);
    scene::removeRenderSettingsListener(mRenderSettingsListener);
    scene::isolation::removeInheritListener(mInheritListener);
    if (mId)
        render::Renderer::get().removeViewport(mId);
}

std::string ViewportWidget::historyKey() const
{
    return mLookThrough != scene::kNoNode ? "#" + std::to_string(mLookThrough) : mViewName;
}

CameraView ViewportWidget::cameraView() const
{
    CameraView v{historyKey(), mCamera.target, mCamera.distance, mCamera.yaw, mCamera.pitch, mCamera.orthoWidth};
    if (const scene::Node* n = scene::Graph::current().node(mLookThrough))
    {
        v.node = true;
        v.translate = n->as<glm::dvec3>(scene::Transform::kTranslate);
        v.rotate = n->as<glm::dvec3>(scene::Transform::kRotate);
        v.scale = n->as<glm::dvec3>(scene::Transform::kScale);
        v.orthoWidth = n->as<double>(scene::Camera::kOrthographicWidth);
    }
    return v;
}

void ViewportWidget::applyCameraView(const CameraView& v)
{
    if (v.view != historyKey())
        return;
    if (v.node)
    {
        scene::Graph& g = scene::Graph::current();
        if (!g.node(mLookThrough))
            return;
        UndoStack& u = UndoStack::get();
        bool was = u.enabled();
        u.setEnabled(false);
        g.setValue({mLookThrough, scene::Transform::kTranslate, -1}, v.translate);
        g.setValue({mLookThrough, scene::Transform::kRotate, -1}, v.rotate);
        g.setValue({mLookThrough, scene::Transform::kScale, -1}, v.scale);
        g.setValue({mLookThrough, scene::Camera::kOrthographicWidth, -1}, v.orthoWidth);
        u.setEnabled(was);
        io::markModified();
        mCamera.distance = v.distance;
    }
    else
    {
        mCamera.target = v.target;
        mCamera.distance = v.distance;
        mCamera.yaw = v.yaw;
        mCamera.pitch = v.pitch;
        mCamera.orthoWidth = v.orthoWidth;
    }
    requestFrame();
    Q_EMIT viewStateChanged();
}

void ViewportWidget::recordCamera(const CameraView& before, uint64_t gesture)
{
    // Only moves are camera undo steps, on the camera that moved; switching cameras is not one.
    if (mCameraRestoring || before.view != historyKey())
        return;
    mCameraHistories[before.view].record(before, cameraView(), gesture, nowMs());
    io::uiChanged("viewports");
}

void ViewportWidget::cameraUndo()
{
    commitCameraEdit();
    if (auto v = mCameraHistories[historyKey()].undo(cameraView()))
    {
        applyCameraView(*v);
        io::uiChanged("viewports");
    }
}

void ViewportWidget::cameraRedo()
{
    commitCameraEdit();
    if (auto v = mCameraHistories[historyKey()].redo(cameraView()))
    {
        applyCameraView(*v);
        io::uiChanged("viewports");
    }
}

QSize ViewportWidget::pixelSize() const
{
    qreal dpr = devicePixelRatioF();
    return {std::max(1, int(width() * dpr)), std::max(1, int(height() * dpr))};
}

double ViewportWidget::aspect() const
{
    QSize s = pixelSize();
    return double(s.width()) / double(s.height());
}

void ViewportWidget::syncNativeSurface()
{
    if (!render::Renderer::get().running())
        return;
    const QSize s = pixelSize();
    if (!mId)
        mId = render::Renderer::get().addViewport(uintptr_t(winId()), s.width(), s.height());
    else
        render::Renderer::get().resize(mId, s.width(), s.height());
}

void ViewportWidget::showEvent(QShowEvent* e)
{
    QWidget::showEvent(e);
    syncNativeSurface();
    requestFrame();
}

void ViewportWidget::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    if (mId)
    {
        QSize s = pixelSize();
        render::Renderer::get().resize(mId, s.width(), s.height());
    }
    submit();
    // The native window can settle after the event (dock layout restore); a second frame picks up its final size.
    requestFrame();
}

void ViewportWidget::paintEvent(QPaintEvent*)
{
    requestFrame();
}

bool ViewportWidget::event(QEvent* e)
{
#ifdef __APPLE__
    // Dock reparenting can replace the NSView. Retire its Vulkan objects before preparing the new view.
    if (e->type() == QEvent::WinIdChange ||
        (e->type() == QEvent::PlatformSurface &&
         static_cast<QPlatformSurfaceEvent*>(e)->surfaceEventType() == QPlatformSurfaceEvent::SurfaceAboutToBeDestroyed))
    {
        if (mId)
        {
            render::Renderer::get().removeViewport(mId);
            mId = 0;
        }
        if (e->type() == QEvent::WinIdChange && internalWinId())
            QTimer::singleShot(0, this, [this] {
                if (isVisible())
                {
                    syncNativeSurface();
                    requestFrame();
                }
            });
    }
    if (e->type() == QEvent::DevicePixelRatioChange)
        QTimer::singleShot(0, this, [this] {
            if (mId)
                syncNativeSurface();
            requestFrame();
        });
#endif
    // Native child windows don't get a paint after un-minimize on every platform.
    if (e->type() == QEvent::WindowStateChange || e->type() == QEvent::UpdateRequest)
        requestFrame();
    if (e->type() == QEvent::Leave)
        clearPreselect();
    if (e->type() == QEvent::FocusOut)
    {
        mTabHeld = false;
        mTabUnder.reset();
    }
    // Tab is a held modifier for drag selection here, not focus navigation. Viewport tools (Retopo Draw) keep their own Tab.
    if (e->type() == QEvent::ShortcutOverride || e->type() == QEvent::KeyPress || e->type() == QEvent::KeyRelease)
        if (auto* k = static_cast<QKeyEvent*>(e); k->key() == Qt::Key_Tab || k->key() == Qt::Key_Backtab)
        {
            if (e->type() == QEvent::KeyRelease && !k->isAutoRepeat())
            {
                mTabHeld = false;
                mTabUnder.reset();
                requestFrame();
            }
            if (!viewportTools::active())
            {
                if (e->type() == QEvent::KeyPress && !k->isAutoRepeat())
                {
                    mTabHeld = true;
                    updateTabHover(mapFromGlobal(QCursor::pos().toPointF()));
                }
                e->accept();
                return true;
            }
        }
    return QWidget::event(e);
}

void ViewportWidget::leaveEvent(QEvent* e)
{
    // Drops the cursor badges from the view the cursor left.
    requestFrame();
    QWidget::leaveEvent(e);
}

void ViewportWidget::requestFrame()
{
    if (mFrameQueued)
        return;
    mFrameQueued = true;
    QTimer::singleShot(0, this, [this] {
        mFrameQueued = false;
        submit();
    });
}

void ViewportWidget::updateCursor()
{
    if (mFramePickArmed)
    {
        if (cursor().shape() != Qt::CrossCursor)
            setCursor(Qt::CrossCursor);
        return;
    }
    if (ViewportTool* tool = viewportTools::active())
    {
        QCursor want = tool->cursor().value_or(QCursor(Qt::ArrowCursor));
        QCursor have = cursor();
        if (want.shape() != have.shape() || (want.shape() == Qt::BitmapCursor && want.pixmap().cacheKey() != have.pixmap().cacheKey()))
            setCursor(want);
        return;
    }
    // Set explicitly: the native viewport window otherwise keeps whatever cursor it was entered with, like a text caret.
    Gizmo* gz = currentGizmo();
    Qt::CursorShape shape = mNodeHandles.dragging()   ? Qt::ClosedHandCursor
                            : mNodeHandles.hovering() ? Qt::PointingHandCursor
                            : !gz                    ? Qt::ArrowCursor
                            : gz->dragging()       ? Qt::ClosedHandCursor
                            : gz->hoveringLocked() ? Qt::ForbiddenCursor
                            : gz->hovering()       ? Qt::OpenHandCursor
                                                   : Qt::ArrowCursor;
    if (cursor().shape() != shape)
        setCursor(shape);
}

void ViewportWidget::submit()
{
    LILA_ZONE();
    updateCursor();

    syncLookThrough();
    if (!mId)
        return;
    render::Renderer::get().submit(mId, buildFrame(nullptr));
}

render::FrameState ViewportWidget::renderFrame(const RenderView& view)
{
    return buildFrame(&view);
}

render::FrameState ViewportWidget::buildFrame(const RenderView* rv)
{
    LILA_ZONE();
    const Camera& camera = rv ? rv->camera : mCamera;
    const double viewAspect = rv ? double(rv->width) / double(std::max(rv->height, 1)) : aspect();
    scene::Graph& g = scene::Graph::current();
    if (!rv && cameraBookmarks::activeView() == this)
        scene::viewState::setActiveCamera(activeCameraOf(camera, viewAspect));
    g.evaluate();
    const bool inherit = scene::inheritVisibility();

    render::FrameState s;
    s.view = camera.view();
    s.proj = camera.proj(viewAspect);
    s.eye = camera.eye();
    s.zUp = camera.zUp;
    const glm::vec4 themeBg = themeSrgb("viewportBackground", glm::vec4(0.30f, 0.30f, 0.30f, 1.0f));
    auto bg = PrefStore::get().value("general", "/viewport/background", json::array({themeBg.r, themeBg.g, themeBg.b}));
    if (bg.is_array() && bg.size() == 3)
        s.background = {bg[0].get<float>(), bg[1].get<float>(), bg[2].get<float>()};
    {
        LILA_ZONE_N("buildFrame settings");
        const PrefStore& p = PrefStore::get();
        s.envPath = environments::resolve(p.value<std::string>("general", "/viewport/env/path", ""));
        s.envRotation = p.value<float>("general", "/viewport/env/rotation", 0.0f) + float(scene::turntable::environmentTurn(scene::Graph::current()));
        s.envIntensity = environments::intensity();
        s.envBackground = p.value<bool>("general", "/viewport/env/background", true);
        s.envBlur = p.value<float>("general", "/viewport/env/blur", 0.6f);
        s.envSaturation = p.value<float>("general", "/viewport/env/saturation", 0.0f);
        s.envLightSaturation = p.value<float>("general", environments::kLightSaturation, 1.0f);
        s.envBackgroundExposure = p.value<float>("general", "/viewport/env/backgroundExposure", 0.25f);
        // Orthographic views use the flat backdrop, like Maya's; an environment has no single direction there.
        if (camera.ortho)
            s.envBackground = false;
        s.groundShadow = p.value<float>("general", "/viewport/groundShadow", 0.5f);
        s.ambientOcclusion = p.value<bool>("general", "/viewport/ao/enabled", true);
        s.aoRadius = p.value<float>("general", "/viewport/ao/radius", 30.0f);
        s.aoIntensity = p.value<float>("general", "/viewport/ao/intensity", 1.0f);
        s.aoFast = p.value<std::string>("general", "/viewport/ao/mode", "accurate") == "fast";
        s.aoRayTraced = p.value<std::string>("general", "/viewport/ao/mode", "accurate") == "raytraced";
        {
            LILA_ZONE_N("buildFrame display settings");
            viewportDisplay::apply(s);
        }
        if (rv)
        {
            if (rv->lens)
                viewportDisplay::applyLens(s, *rv->lens);
        }
        else if (const scene::Node* cam = mLookThrough != scene::kNoNode ? g.node(mLookThrough) : nullptr; cam && cam->type().isA("camera"))
            viewportDisplay::applyLens(s, scene::camera::lens(*cam));
        viewportDisplay::applyCurveWidths(s, devicePixelRatioF());
        if (!rv)
            displayAids::apply(s, mHudPanel);
    }
    if (!rv)
    {
        viewportRenderers::Request req;
        const QSize px = pixelSize();
        req.width = px.width();
        req.height = px.height();
        req.view = viewInfo();
        req.nearClip = mCamera.nearPlane();
        req.farClip = mCamera.farClip.value_or(mCamera.ortho ? mCamera.distance * 2.0 + 1e5 : 0.0);
        req.time = g.time();
        req.exposure = s.exposure;
        req.gamma = s.gamma;
        viewportRenderers::apply(mHudPanel, std::move(req), s);
    }

    // Like Maya, everything under a selected transform highlights too, a step dimmer.
    std::unordered_set<scene::NodeId> underSelection;
    const bool highlight = !rv && viewportDisplay::highlightSelection();
    if (highlight)
    {
        LILA_ZONE_N("buildFrame selection sets");
        for (scene::NodeId id : g.selection())
            for (scene::NodeId d : g.descendants(id))
                underSelection.insert(d);
    }

    const bool aids = !rv && displayAids::anyDrawn(mHudPanel);
    std::vector<displayAids::MeshEntry> aidMeshes;
    {
        // The HUD counts skip the same hidden levels, so they share this frame's set.
        mLodHidden = scene::lod::hidden(g, lodDisplay::view(mHudPanel, viewInfo()));
        size_t key = mLodHidden.size();
        for (scene::NodeId id : mLodHidden)
            key += std::hash<scene::NodeId>()(id) * 0x9E3779B97F4A7C15ull;
        mLodHiddenKey = key;
    }
    drawMeshNodes(s, rv, mLodHidden, underSelection, highlight, aids ? &aidMeshes : nullptr);
    // A render is the scene alone: no components, tools, locators, grid or HUD.
    if (rv)
    {
        s.grid = false;
        return s;
    }

    drawComponents(s);
    drawPreselect(s);
    drawHandlerHover(s);
    OverlayPainter painter(s.overlayTris, devicePixelRatioF());
    {
        constexpr int kCols = 8, kRows = 6;
        auto probe = render::Renderer::get().environmentProbe(s.envPath);
        OverlayPainter::classifyBackdrop(render::backdropLuminance(s, probe.get(), kCols, kRows), mBackdropBright);
        painter.setBackdrop(mBackdropBright, kCols, kRows, QSizeF(width(), height()));
    }
    drawGate(painter);
    const ViewInfo view = viewInfo();
    const scene::sceneIndex::Hierarchy& hierarchy = scene::sceneIndex::hierarchy(g, inherit);
    if (aids)
    {
        for (scene::Node* n : scene::sceneIndex::matching(g, "pointProvider", [](const scene::Node& n) { return scene::components::pointProvider(n); }))
            if (const scene::NodeId id = n->id(); id != mLookThrough && hierarchy.visible(id) && mIsolation.shows(id) && scene::selectionMask::shown(*n))
                aidMeshes.push_back({nullptr, shownWorld(*n), scene::objectBounds(g, id)});
        displayAids::draw(s, painter, view, mHudPanel, aidMeshes, devicePixelRatioF());
    }
    // Locators and extension node types. Highlighted draws last: with an equal-depth test the later line wins, so a
    // selected locator sitting exactly on an unselected one stays visible.
    const std::vector<scene::Node*>& drawn = scene::sceneIndex::matching(g, "drawOverride" + std::to_string(drawOverrides::version()), [](const scene::Node& n) {
        const drawOverrides::DrawOverride* o = drawOverrides::forType(n.type());
        return o && o->draw;
    });
    const viewportDisplay::SelectionOrderColors order = viewportDisplay::selectionOrderColors(g);
    for (int pass = 0; pass < 3; ++pass)
    {
        LILA_ZONE_N("buildFrame draw overrides");
        for (scene::Node* n : drawn)
        {
            const scene::NodeId id = n->id();
            const drawOverrides::DrawOverride* o = drawOverrides::forType(n->type());
            if (id == mLookThrough || !hierarchy.visible(id) || !scene::selectionMask::shown(*n) || !mIsolation.shows(id))
                continue;
            bool hot = mPreselect.mode == scene::ComponentMode::Object && mPreselect.node == id;
            int rank = highlight && g.isSelected(id) ? 2 : hot || underSelection.contains(id) ? 1 : 0;
            if (rank != pass)
                continue;
            using HL = drawOverrides::Highlight;
            drawOverrides::DrawContext dc{.frame = s, .painter = painter, .view = view, .node = *n};
            dc.world = n->as<glm::dmat4>(scene::Transform::kWorldMatrix);
            dc.highlight = rank == 2 ? HL::Selected : hot ? HL::Preselected : rank == 1 ? HL::UnderSelected : HL::None;
            dc.color = rank == 2 ? themeLinear(order.other(id) ? "wireSelectedOther" : "wireSelected", kSelectedColor)
                       : hot ? themeLinear("wirePreselect", kPreselectColor)
                       : rank == 1 ? themeLinear("wireUnderSelected", kChildOfSelectedColor)
                                   : layerLineColor(g, id, themeLinear("wireframe", kLocatorColor));
            o->draw(dc);
        }
    }
    renderPasses::Context passes{.frame = s, .painter = painter, .view = view, .panel = mHudPanel};
    renderPasses::run(renderPasses::Stage::Scene, passes);
    {
        const std::vector<viewportHooks::Drawer> drawers = viewportHooks::drawers();
        const std::vector<std::string> drawerIds = LILA_PROFILER_CONNECTED() ? viewportHooks::drawerIds() : std::vector<std::string>();
        for (size_t i = 0; i < drawers.size(); ++i)
        {
            LILA_ZONE_N("buildFrame viewport drawer");
            if (i < drawerIds.size())
                LILA_ZONE_TEXT(drawerIds[i].data(), drawerIds[i].size());
            drawers[i](s, painter, view);
        }
    }
    if (ViewportTool* tool = viewportTools::active())
        tool->draw(s, painter, view);
    if (Gizmo* gz = drawnGizmo())
    {
        gz->setView(view);
        gz->draw(painter);
    }
    mNodeHandles.draw(painter, view);
    renderPasses::run(renderPasses::Stage::Overlay, passes);
    drawSoftSelectRadius(painter);
    if (hud::shown(mHudPanel, "originAxis"))
        hud::drawOriginAxis(s, painter, view);
    drawViewIndicator(painter);
    drawViewCube(painter);
    drawHud(painter);
    if (QPointF cursor = mapFromGlobal(QCursor::pos().toPointF()); rect().contains(cursor.toPoint()))
    {
        std::vector<const cursorBadges::Badge*> badges = cursorBadges::active(state(cursor));
        cursorBadges::draw(painter, cursor, badges);
        cursorBadges::publish(this, badges);
    }
    else
        cursorBadges::publish(this, {});
    renderPasses::run(renderPasses::Stage::Final, passes);
    // The ground grid is edge-on in side views, so they get a grid in the view plane through the origin instead.
    if (s.grid && mCamera.ortho && std::abs(glm::dot(mCamera.forward(), mCamera.up())) < 0.5)
    {
        glm::dvec3 r = mCamera.right(), u = glm::cross(r, mCamera.forward());
        glm::dvec2 ext = mCamera.viewExtents(aspect());
        double halfW = ext.x, halfH = ext.y;
        double step = std::pow(10.0, std::floor(std::log10(std::max(1e-3, mCamera.orthoWidth / 20.0))));
        glm::dvec3 centre = mCamera.target - mCamera.forward() * glm::dot(mCamera.target, mCamera.forward());
        double cx = glm::dot(centre, r), cy = glm::dot(centre, u);
        auto lineColor = [](double v, double stepSize) {
            bool axis = std::abs(v) < stepSize * 0.5;
            bool major = std::abs(std::fmod(std::abs(v) + stepSize * 0.5, stepSize * 10.0)) < stepSize;
            float g = axis ? 0.0f : major ? 0.045f : 0.025f; // linear grays
            return glm::vec4(g, g, g, 1.0f);
        };
        for (double x = std::floor((cx - halfW) / step) * step; x <= cx + halfW; x += step)
        {
            glm::vec4 c = lineColor(x, step);
            s.lines.push_back({glm::vec3(r * x + u * (cy - halfH)), c});
            s.lines.push_back({glm::vec3(r * x + u * (cy + halfH)), c});
        }
        for (double y = std::floor((cy - halfH) / step) * step; y <= cy + halfH; y += step)
        {
            glm::vec4 c = lineColor(y, step);
            s.lines.push_back({glm::vec3(r * (cx - halfW) + u * y), c});
            s.lines.push_back({glm::vec3(r * (cx + halfW) + u * y), c});
        }
    }
    if (!s.overlayTris.empty())
        s.atlas = painter.atlas();

    if (mMarquee && mLassoDrag && mLasso.size() > 1)
    {
        auto ndc = [&](glm::dvec2 p) { return glm::vec3(float(p.x / width() * 2.0 - 1.0), float(p.y / height() * 2.0 - 1.0), 0.5f); };
        // Dashes alternate light and dark so the outline reads on any background.
        constexpr double kDash = 6.0;
        const glm::vec4 shades[2] = {{0.9f, 0.9f, 0.9f, 0.95f}, {0.12f, 0.12f, 0.12f, 0.95f}};
        double run = 0.0;
        for (size_t i = 0; i < mLasso.size(); ++i)
        {
            glm::dvec2 a = mLasso[i], b = mLasso[(i + 1) % mLasso.size()];
            double len = glm::distance(a, b);
            for (double t = 0.0; t < len;)
            {
                double step = std::min(len - t, kDash - std::fmod(run, kDash));
                const glm::vec4& color = shades[int(run / kDash) % 2];
                s.overlay.push_back({ndc(a + (b - a) * (t / len)), color});
                s.overlay.push_back({ndc(a + (b - a) * ((t + step) / len)), color});
                t += step;
                run += step;
            }
        }
    }
    else if (mMarquee)
    {
        auto ndc = [&](QPointF p) {
            return glm::vec3(float(p.x() / width() * 2.0 - 1.0), float(p.y() / height() * 2.0 - 1.0), 0.5f);
        };
        glm::vec3 a = ndc(mPressPos), b = ndc(mLastPos);
        glm::vec3 c[4] = {a, {b.x, a.y, 0.5f}, b, {a.x, b.y, 0.5f}};
        const glm::vec4 color(0.9f, 0.9f, 0.9f, 0.9f);
        for (int i = 0; i < 4; ++i)
        {
            s.overlay.push_back({c[i], color});
            s.overlay.push_back({c[(i + 1) % 4], color});
        }
    }
    return s;
}

ViewInfo ViewportWidget::viewInfo() const
{
    ViewInfo v;
    v.view = mCamera.view();
    v.proj = mCamera.proj(aspect());
    v.eye = mCamera.eye();
    v.width = width();
    v.height = height();
    v.camera = mViewName;
    v.ortho = mCamera.ortho;
    v.film = mCamera.filmExtents();
    v.isolation = &mIsolation;
    return v;
}

void ViewportWidget::wrapCursor(QPointF p)
{
    const glm::dvec2 target = gizmo::CursorWrap::target({p.x(), p.y()}, width(), height());
    if (target == glm::dvec2(p.x(), p.y()))
        return;
    const QPoint to = QPointF(target.x, target.y).toPoint();
    QCursor::setPos(mapToGlobal(to));
    mCursorWrap.warped({p.x(), p.y()}, {double(to.x()), double(to.y())});
    if (mGizmoDrag)
        mGizmoDrag->warped(QPointF(to) - p);
}

void ViewportWidget::syncGizmos()
{
    auto fit = [this](GizmoSlot& slot, const std::string& id) {
        if (slot.gizmo && slot.id == id && gizmos::find(id))
            return;
        if (slot.gizmo && mGizmoDrag == slot.gizmo.get())
        {
            mGizmoDrag->cancel();
            mGizmoDrag = nullptr;
        }
        slot.gizmo = id.empty() ? nullptr : gizmos::create(id);
        slot.id = slot.gizmo ? id : std::string();
    };
    const std::string active = gizmos::activeId();
    fit(mGizmo, active);
    fit(mDefaultGizmo, active == gizmos::kDefault ? std::string() : std::string(gizmos::kDefault));
}

Gizmo* ViewportWidget::currentGizmo()
{
    syncGizmos();
    if (mGizmoDrag)
        return mGizmoDrag;
    auto pick = [this](std::string_view mode) -> Gizmo* {
        for (GizmoSlot* slot : {&mGizmo, &mDefaultGizmo})
            if (slot->gizmo && slot->gizmo->supports(mode))
                return slot->gizmo.get();
        return nullptr;
    };
    if (ViewportTool* tool = viewportTools::active())
        if (Gizmo* gz = pick(tool->name()))
            return gz;
    return pick(tools::name(tools::current()));
}

Gizmo* ViewportWidget::drawnGizmo()
{
    Gizmo* gz = currentGizmo();
    if (!gz)
        return nullptr;
    ViewportTool* tool = viewportTools::active();
    return !tool || gz->supports(tool->name()) ? gz : nullptr;
}

ViewportState ViewportWidget::state(QPointF cursor)
{
    ViewportState st;
    st.view = viewInfo();
    st.panel = mHudPanel;
    st.cursor = cursor;
    st.gizmo = currentGizmo();
    if (ViewportTool* tool = viewportTools::active())
        st.tool = tool->name();
    st.navigating = mNav != Nav::None;
    if (mTabStroke.active())
        st.tabSelects = mTabStroke.op() != paintSelect::Stroke::Op::Remove;
    else if (mTabHeld && !viewportTools::active())
    {
        Qt::KeyboardModifiers mods = QGuiApplication::queryKeyboardModifiers();
        st.tabSelects = tabSelect::modeAtPress(mTabUnder, mods & Qt::ShiftModifier, mods & Qt::ControlModifier) == tabSelect::Mode::Select;
    }
    return st;
}

std::optional<QPointF> ViewportWidget::project(const glm::dvec3& world) const
{
    glm::dvec4 clip = mCamera.proj(aspect()) * mCamera.view() * glm::dvec4(world, 1.0);
    if (clip.w <= 1e-9)
        return std::nullopt;
    return QPointF((clip.x / clip.w * 0.5 + 0.5) * width(), (clip.y / clip.w * 0.5 + 0.5) * height());
}

std::vector<std::string> ViewportWidget::hitTest(QPointF a, QPointF b)
{
    LILA_ZONE();

    scene::Graph& g = scene::Graph::current();
    g.evaluate();
    const bool inherit = scene::inheritVisibility();
    QRectF box = QRectF(a, b).normalized();
    bool click = box.width() < 4 && box.height() < 4;

    json hits = json::array();
    double best = 6.0; // px
    std::string bestName;
    ViewInfo view = viewInfo();
    for (scene::NodeId id : g.nodes())
    {
        scene::Node* n = g.node(id);
        const drawOverrides::DrawOverride* o = drawOverrides::forType(n->type());
        if (!o || !o->pick || id == mLookThrough || !viewportPickable(g, id, inherit) || !scene::selectionMask::pickable(*n) ||
            !mIsolation.shows(id))
            continue;
        std::optional<double> px = o->pick({.view = view, .node = *n, .world = n->as<glm::dmat4>(scene::Transform::kWorldMatrix), .a = a, .b = b, .click = click});
        if (!px)
            continue;
        // The override only knows the lasso's bounds, so its pivot has to fall inside the lasso itself.
        if (!click && mLassoPick)
            if (auto s = project(glm::dvec3(n->as<glm::dmat4>(scene::Transform::kWorldMatrix)[3])); !s || !inSelectRegion(box, *s))
                continue;
        if (!click)
            hits.push_back(n->path());
        else if (*px < best)
        {
            best = *px;
            bestName = n->path();
        }
    }
    // Meshes: a click takes the nearest surface under the cursor unless a locator line is right there; a box takes
    // any mesh with a vertex inside it.
    glm::dvec3 rayO, rayD;
    view.ray(a, rayO, rayD);
    double nearest = 1e300;
    std::string nearestMesh;
    const std::unordered_set<scene::NodeId> lodHidden = scene::lod::hidden(g, lodDisplay::view(mHudPanel, view));
    for (scene::NodeId id : g.nodes())
    {
        scene::Node* n = g.node(id);
        if (!n->type().isA("mesh") || overridden(*n) || !viewportPickable(g, id, inherit) || !scene::selectionMask::pickable(*n) ||
            !mIsolation.shows(id) || viewportHooks::hiddenInView(id) || lodHidden.contains(id))
            continue;
        scene::MeshPtr outMesh = n->as<scene::MeshPtr>(scene::MeshNode::kOutMesh);
        auto rm = renderMeshFor(outMesh);
        if (!rm || rm->vertices.empty())
            continue;
        glm::dmat4 world = shownWorld(*n);
        // Array copies pick as their array, so a click on one selects the generator rather than the source.
        const std::vector<scene::arrays::Copy> copies = scene::arrays::copiesOf(g, id);
        if (click)
        {
            if (auto t = isolate::raycast(&mIsolation, id, outMesh, world, rayO, rayD); t && t->distance < nearest)
            {
                nearest = t->distance;
                nearestMesh = n->path();
            }
            for (const scene::arrays::Copy& c : copies)
                if (auto t = isolate::raycast(&mIsolation, id, outMesh, c.transform * world, rayO, rayD); t && t->distance < nearest)
                {
                    nearest = t->distance;
                    nearestMesh = g.node(c.array)->path();
                }
            continue;
        }
        size_t step = std::max<size_t>(1, rm->vertices.size() / 20000);
        for (const scene::arrays::Copy& c : copies)
        {
            const std::string arrayPath = g.node(c.array)->path();
            if (std::ranges::any_of(hits, [&](const json& h) { return h.get<std::string>() == arrayPath; }))
                continue;
            const glm::dmat4 at = c.transform * world;
            for (size_t k = 0; k < rm->vertices.size(); k += step)
                if (auto p = project(glm::dvec3(at * glm::dvec4(glm::dvec3(rm->vertices[k].pos), 1.0))); p && inSelectRegion(box, *p))
                {
                    hits.push_back(arrayPath);
                    break;
                }
        }
        if (mIsolation.isPartial(id))
        {
            // Only the isolated faces' points count toward a marquee.
            const geo::Mesh& m = *outMesh;
            auto vis = isolate::visibility(&mIsolation, id, outMesh);
            for (geo::Index p = 0; p < m.pointCapacity(); ++p)
                if (m.pointValid(p) && vis->point(p))
                    if (auto s = project(glm::dvec3(world * glm::dvec4(m.position(p), 1.0))); s && inSelectRegion(box, *s))
                    {
                        hits.push_back(n->path());
                        break;
                    }
            continue;
        }
        for (size_t k = 0; k < rm->vertices.size(); k += step)
            if (auto p = project(glm::dvec3(world * glm::dvec4(glm::dvec3(rm->vertices[k].pos), 1.0))); p && inSelectRegion(box, *p))
            {
                hits.push_back(n->path());
                break;
            }
    }
    for (const auto& picker : viewportHooks::pickers())
        for (const viewportHooks::PickHit& h : picker(a, b, click, view))
        {
            const scene::Node* hn = findNode(h.name);
            if (hn ? !scene::selectionMask::pickable(*hn) : !scene::selectionMask::enabled(scene::selectionMask::Kind::Other))
                continue;
            if (hn && (!scene::displayLayers::pickable(g, hn->id()) || !mIsolation.shows(hn->id())))
                continue;
            if (!click)
                hits.push_back(h.name);
            else if (h.pixels < best)
            {
                best = h.pixels;
                bestName = h.name;
            }
        }
    if (click && bestName.empty() && !nearestMesh.empty())
        hits.push_back(nearestMesh);
    if (click && !bestName.empty())
        hits.push_back(bestName);
    return hits.get<std::vector<std::string>>();
}

} // namespace lila::ui
