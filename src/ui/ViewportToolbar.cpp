#include "ui/ViewportToolbar.h"

#include "ops/Command.h"
#include "ops/Undo.h"
#include "core/Log.h"
#include "core/PrefStore.h"
#include "scene/ComponentPick.h"
#include "scene/Graph.h"
#include "scene/Components.h"
#include "scene/MeshNodes.h"
#include "scene/RenderSettings.h"
#include "ui/Actions.h"
#include "ui/DisplayAids.h"
#include "ui/Environments.h"
#include "ui/GizmoMath.h"
#include "ui/Icons.h"
#include "ui/ImagePlaneUi.h"
#include "ui/LodDisplay.h"
#include "ui/MaterialDraw.h"
#include "ui/OptionBox.h"
#include "ui/PaintSelect.h"
#include "ui/RetopoDrawTool.h"
#include "ui/SpinScrub.h"
#include "ui/TearOff.h"
#include "ui/Tools.h"
#include "ui/UiCommon.h"
#include "ui/ViewportContext.h"
#include "ui/ViewportDisplay.h"
#include "ui/ViewportHud.h"
#include "ui/ViewportRenderers.h"
#include "ui/ViewportWidget.h"
#include "ui/FileDialog.h"

#include <QActionGroup>
#include <QCheckBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QMainWindow>
#include <QMouseEvent>
#include <QFileDialog>
#include <QFormLayout>
#include <QFrame>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QPointer>
#include <QSignalBlocker>
#include <QSlider>
#include <QSpinBox>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <QWidgetAction>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <limits>
#include <unordered_set>
#include <utility>
#include <string>

namespace lila::ui
{
namespace
{
namespace vd = viewportDisplay;

constexpr int kHeight = 24;
constexpr int kIcon = 16;
constexpr int kSpacing = 1;
constexpr const char* kViews[] = {"persp", "top", "front", "right", "left", "back", "bottom"};

/** Shift+click on a toolbar button tears off a main menu at the cursor instead of clicking the button. */
class ShiftClickTearOff final : public QObject
{
public:
    ShiftClickTearOff(QObject* parent, std::vector<std::string> menu) : QObject(parent), mMenu(std::move(menu)) {}

    bool eventFilter(QObject* obj, QEvent* e) override
    {
        if (e->type() != QEvent::MouseButtonPress && e->type() != QEvent::MouseButtonRelease && e->type() != QEvent::MouseButtonDblClick)
            return false;
        auto* m = static_cast<QMouseEvent*>(e);
        if (m->button() != Qt::LeftButton || !(m->modifiers() & Qt::ShiftModifier))
            return false;
        // The press is swallowed too, so the button neither toggles nor starts its long-press menu.
        if (e->type() == QEvent::MouseButtonRelease)
            if (auto* window = qobject_cast<QMainWindow*>(static_cast<QWidget*>(obj)->window()))
                if (QMenu* menu = tearoff::findMenu(window, mMenu))
                    tearoff::tearOff(menu, m->globalPosition().toPoint());
        return true;
    }

private:
    std::vector<std::string> mMenu;
};

QString labelOf(const std::string& id)
{
    const Action* a = actions::find(id);
    return QString::fromStdString(a ? a->label : id);
}

QString tipFor(const std::string& id)
{
    return QString::fromStdString(actions::tooltip(id));
}

std::string snapPref(Tool t, const char* leaf)
{
    return std::string("/manipulator/snap/") + tools::snapKey(t) + "/" + leaf;
}

QString stepText(double v)
{
    return QString::number(v, 'g', 6);
}

/**
 * The rotate increments as two columns in one menu entry: common angles, and divisions of a turn (360/n). Checkable
 * flat buttons stand in for menu items, and a click closes the menu.
 */
QWidgetAction* rotateStepColumns(QMenu* menu, double current, const std::string& stepKey, bool& matched)
{
    auto* holder = new QWidget(menu);
    auto* grid = new QGridLayout(holder);
    grid->setContentsMargins(4, 2, 4, 2);
    grid->setHorizontalSpacing(8);
    grid->setVerticalSpacing(0);
    auto header = [&](const QString& text, int col) {
        auto* l = new QLabel(text, holder);
        l->setStyleSheet("color: #8c8c8c; padding: 2px 6px;");
        grid->addWidget(l, 0, col);
    };
    header("Angle", 0);
    header("Divisions of 360", 1);
    auto add = [&](double v, const QString& text, int row, int col) {
        auto* b = new QToolButton(holder);
        b->setText(text);
        b->setCheckable(true);
        bool on = std::abs(v - current) < 1e-9;
        matched = matched || on;
        b->setChecked(on);
        b->setAutoRaise(true);
        b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        b->setStyleSheet("QToolButton { text-align: left; padding: 3px 6px; border: none; }"
                         "QToolButton:hover { background: #4a4a4a; } QToolButton:checked { background: #5c5c5c; }");
        QObject::connect(b, &QToolButton::clicked, menu, [menu, stepKey, v] {
            PrefStore::get().set("general", stepKey, v);
            menu->close();
        });
        grid->addWidget(b, row, col);
    };
    const QString deg(QChar(0x00B0));
    int row = 1;
    for (double v : gizmo::kRotateSteps)
        add(v, stepText(v) + deg, row++, 0);
    row = 1;
    for (int n : gizmo::kTurnDivisions)
    {
        double v = gizmo::turnDivision(n);
        add(v, stepText(v) + deg + "  (360/" + QString::number(n) + ")", row++, 1);
    }
    grid->setRowStretch(row, 1);
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(holder);
    return action;
}
/** The select, transform and selection tools, and the component modes: which one is in use. */
std::optional<bool> toolActionState(const std::string& id)
{
    const bool viewportTool = viewportTools::active() != nullptr;
    for (Tool t : {Tool::Select, Tool::Move, Tool::Rotate, Tool::Scale})
        if (id == std::string("tool.") + tools::name(t))
            return !viewportTool && tools::current() == t;
    if (id == "select.marqueeTool" || id == "select.lassoTool")
        return !viewportTool && tools::current() == Tool::Select && generalPref(paintSelect::kLassoPref, false) == (id == "select.lassoTool");
    if (id == "select.paintTool")
        return viewportTools::activeId() == paintSelect::kToolId;
    using scene::ComponentMode;
    for (auto [mode, action] : {std::pair{ComponentMode::Object, "select.objectMode"}, {ComponentMode::Vertex, "select.vertexMode"},
                                {ComponentMode::Edge, "select.edgeMode"}, {ComponentMode::Face, "select.faceMode"}})
        if (id == action)
            return scene::components::mode() == mode;
    return std::nullopt;
}
} // namespace

std::optional<bool> viewportActionState(const std::string& id, const ViewportWidget* view)
{
    // Wireframe, Shaded and Textured are one choice, as in Maya.
    const bool wire = PrefStore::get().value<std::string>("general", "/viewport/displayMode", "shaded") == "wireframe";
    if (id == "viewport.wireframe")
        return wire;
    if (id == "viewport.shaded")
        return !wire && !materialsShown();
    if (id == "viewport.textured")
        return !wire && materialsShown();
    if (id == "rig.xrayJoints")
        return generalPref("/rig/xrayJoints", false);
    if (id == "retopo.toggleSnapSpace")
        return RetopoDrawTool::snapObjectSpace();
    if (std::optional<bool> tool = toolActionState(id))
        return tool;
    struct Toggle
    {
        const char* id;
        const char* key;
        bool fallback;
    };
    static const Toggle toggles[] = {
        {"viewport.wireOnShaded", "/viewport/wireOnShaded", false},
        {"viewport.toggleAO", "/viewport/ao/enabled", true},
        {"viewport.toggleAA", vd::kAaEnabled, true},
        {"viewport.toggleGrid", vd::kGrid, true},
        {"viewport.toggleXray", vd::kXray, false},
        {"viewport.toggleBackfaceCulling", vd::kBackfaceCulling, false},
        {"viewport.toggleKuwahara", vd::kKuwahara, false},
        {"viewport.toggleOutline", vd::kOutline, false},
        {"viewport.toggleShadows", vd::kShadows, true},
        {"viewport.defaultMaterial", vd::kDefaultMaterial, false},
        {"viewport.greyBaseColor", vd::kGreyBaseColor, false},
        {"viewport.smoothSubdivisions", vd::kSmoothSubdivisions, false},
        {"viewport.toggleEnvBackground", "/viewport/env/background", true},
        {"viewport.toggleToolbar", vd::kToolbar, true},
        {"viewport.toggleBottomBar", vd::kBottomBar, true},
        {"viewport.grayscaleWeights", vd::kGrayscaleWeights, false},
        {"viewport.monotone", vd::kMonotone, false},
        {"viewport.toggleHighlightSelection", vd::kHighlightSelection, true},
        {"viewport.togglePreselectObjects", "/viewport/preselect/objects", false},
        {"viewport.togglePreselectComponents", "/viewport/preselect/components", true},
        {"window.toggleToolBox", "/ui/showToolBox", true},
        {"softSelect.toggle", "/softSelect/enabled", false},
        {"measure.toggleShow", "/measure/show", true},
        {"select.toggleCameraBased", "/select/cameraBased", false},
        {"select.toggleBackfaceCulling", "/select/backfaceCulling", false},
        {"select.toggleLasso", "/select/lasso", false},
        {"select.edgesByCenter", scene::componentPick::kEdgesByCenter, false},
        {"select.facesByCenter", scene::componentPick::kFacesByCenter, false},
    };
    for (const Toggle& t : toggles)
        if (id == t.id)
            return generalPref(t.key, t.fallback);
    if (id.starts_with("select.mask."))
        for (scene::selectionMask::Kind k : scene::selectionMask::kAll)
            if (id.substr(12) == scene::selectionMask::name(k))
                return scene::selectionMask::enabled(k);
    if (id.starts_with("viewport.show."))
        for (scene::selectionMask::Kind k : scene::selectionMask::kAll)
            if (id.substr(14) == scene::selectionMask::name(k))
                return scene::selectionMask::shown(k);
    if (id.starts_with("viewport.aaSamples"))
        return generalPref(vd::kAaEnabled, true) && !vd::temporalAa() &&
               PrefStore::get().value<int>("general", vd::kAaSamples, vd::kDefaultSamples) == std::stoi(id.substr(18));
    if (id == "viewport.aaTemporal")
        return generalPref(vd::kAaEnabled, true) && vd::temporalAa();
    if (id == "viewport.toggleVertexColors")
        return PrefStore::get().value<std::string>("general", vd::kVertexColors, "off") != "off";
    if (id.starts_with("viewport.vertexColors."))
        return PrefStore::get().value<std::string>("general", vd::kVertexColors, "off") == id.substr(22);
    if (id.starts_with("viewport.colorChannel."))
        return PrefStore::get().value<std::string>("general", vd::kColorChannel, "rgb") == id.substr(22);
    if (id.starts_with("viewport.lighting."))
        return PrefStore::get().value<std::string>("general", vd::kLighting, "environment") == id.substr(18);
    if (view && id.starts_with("viewport.aid."))
        return displayAids::actionState(id, view);
    if (const Action* a = actions::find(id); a && a->checked)
        return a->checked();
    if (!view)
        return std::nullopt;
    if (std::optional<bool> hudState = hud::actionState(id, view))
        return hudState;
    if (id == "viewport.isolate")
        return view->isolating();
    if (id == "viewport.toggleImagePlanes")
        return imagePlanes::shownIn(view->viewName());
    if (id.starts_with("viewport.view."))
        return view->viewName() == id.substr(14);
    if (id == "viewport.toggleResolutionGate" || id == "viewport.toggleFilmGate")
    {
        scene::CameraSettings c = scene::renderSettings().camera(view->viewName());
        return id == "viewport.toggleFilmGate" ? c.filmGate : c.resolutionGate;
    }
    return std::nullopt;
}

namespace
{
/** Intensity of the current HDRI as a slider (0..4) and a field (0..10), remembered per HDRI; Reset follows the global pref. */
QWidgetAction* environmentIntensityRow(QMenu* menu)
{
    static constexpr double kSliderMax = 4.0;
    constexpr int kSliderSteps = 400;
    auto* holder = new QWidget(menu);
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(8, 4, 8, 4);
    row->setSpacing(6);
    row->addWidget(new QLabel("Intensity", holder));
    auto* slider = new QSlider(Qt::Horizontal, holder);
    slider->setRange(0, kSliderSteps);
    slider->setMinimumWidth(120);
    auto* spin = new QDoubleSpinBox(holder);
    spin->setRange(0.0, 10.0);
    spin->setSingleStep(0.05);
    spin->setDecimals(2);
    spin->setKeyboardTracking(false);
    spin->setFixedWidth(52);
    setSpinDefault(spin, 1.0);
    auto* reset = new QToolButton(holder);
    reset->setText("Reset");
    reset->setAutoRaise(true);
    reset->setToolTip("Use the default intensity from Preferences for this HDRI");
    row->addWidget(slider, 1);
    row->addWidget(spin);
    row->addWidget(reset);

    auto show = [slider, spin, reset] {
        double v = environments::intensity();
        QSignalBlocker a(slider), b(spin);
        slider->setValue(int(std::lround(std::min(v, kSliderMax) / kSliderMax * kSliderSteps)));
        spin->setValue(v);
        reset->setEnabled(environments::hasOwnIntensity());
    };
    show();
    QObject::connect(slider, &QSlider::valueChanged, holder, [show](int i) {
        environments::setIntensity(float(i * kSliderMax / kSliderSteps));
        show();
    });
    QObject::connect(spin, &QDoubleSpinBox::valueChanged, holder, [show](double v) {
        environments::setIntensity(float(v));
        show();
    });
    QObject::connect(reset, &QToolButton::clicked, holder, [show] {
        environments::setIntensity(std::nullopt);
        show();
    });
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(holder);
    return action;
}

/** How much of the HDRI's tint its light keeps on surfaces, 0..1; the backdrop keeps its own saturation. */
QWidgetAction* environmentLightColorRow(QMenu* menu)
{
    constexpr int kSliderSteps = 100;
    auto* holder = new QWidget(menu);
    auto* row = new QHBoxLayout(holder);
    row->setContentsMargins(8, 4, 8, 4);
    row->setSpacing(6);
    auto* label = new QLabel("Light Color", holder);
    label->setToolTip("0 lights surfaces with the HDRI's brightness only (neutral gray), 1 with its full colour");
    row->addWidget(label);
    auto* slider = new QSlider(Qt::Horizontal, holder);
    slider->setRange(0, kSliderSteps);
    slider->setMinimumWidth(120);
    auto* spin = new QDoubleSpinBox(holder);
    spin->setRange(0.0, 1.0);
    spin->setSingleStep(0.05);
    spin->setDecimals(2);
    spin->setKeyboardTracking(false);
    spin->setFixedWidth(52);
    setSpinDefault(spin, 1.0);
    row->addWidget(slider, 1);
    row->addWidget(spin);

    auto show = [slider, spin] {
        double v = std::clamp(double(PrefStore::get().value<float>("general", environments::kLightSaturation, 1.0f)), 0.0, 1.0);
        QSignalBlocker a(slider), b(spin);
        slider->setValue(int(std::lround(v * kSliderSteps)));
        spin->setValue(v);
    };
    show();
    QObject::connect(slider, &QSlider::valueChanged, holder, [show](int i) {
        PrefStore::get().set("general", environments::kLightSaturation, double(i) / kSliderSteps);
        show();
    });
    QObject::connect(spin, &QDoubleSpinBox::valueChanged, holder, [show](double v) {
        PrefStore::get().set("general", environments::kLightSaturation, v);
        show();
    });
    auto* action = new QWidgetAction(menu);
    action->setDefaultWidget(holder);
    return action;
}
} // namespace

void fillCameraMenu(QMenu* menu, ViewportWidget* view, std::function<void()> activate)
{
    menu->clear();
    if (!view)
        return;
    QPointer<ViewportWidget> target(view);
    auto pick = [menu, activate](QAction* a, std::function<void()> fn) {
        QObject::connect(a, &QAction::triggered, menu, [activate, fn] {
            if (activate)
                activate();
            fn();
        });
    };
    for (const char* v : kViews)
    {
        std::string id = std::string("viewport.view.") + v;
        QAction* a = menu->addAction(labelOf(id));
        a->setCheckable(true);
        a->setChecked(view->viewName() == v && view->lookThroughCamera() == scene::kNoNode);
        pick(a, [id] { actions::run(id); });
    }
    scene::Graph& g = scene::Graph::current();
    bool first = true;
    for (scene::NodeId id : g.nodes())
    {
        const scene::Node* n = g.node(id);
        if (!n->type().isA("camera"))
            continue;
        if (std::exchange(first, false))
            menu->addSeparator();
        QAction* a = menu->addAction(icon("camera"), QString::fromStdString(n->name()));
        a->setToolTip(QString::fromStdString(n->fullPath()));
        a->setCheckable(true);
        a->setChecked(view->lookThroughCamera() == id);
        pick(a, [target, id] {
            if (target)
                target->lookThrough(id);
        });
    }
    menu->addSeparator();
    for (const char* id : {"viewport.lookThroughSelected", "create.cameraFromView"})
    {
        QAction* a = menu->addAction(labelOf(id));
        a->setEnabled(actions::isEnabled(nlohmann::json(id)));
        pick(a, [id = std::string(id)] { actions::run(id); });
    }
}

void fillEnvironmentMenu(QMenu* menu)
{
    menu->clear();
    std::string resolved = environments::resolve(PrefStore::get().value<std::string>("general", "/viewport/env/path", ""));
    auto item = [&](const QString& label, const std::string& value, bool checked) {
        QAction* a = menu->addAction(label);
        a->setCheckable(true);
        a->setChecked(checked);
        QObject::connect(a, &QAction::triggered, menu, [value] { PrefStore::get().set("general", "/viewport/env/path", value); });
    };
    for (const std::filesystem::path& hdri : environments::builtIn())
    {
        std::u8string name = hdri.filename().u8string();
        std::string value(name.begin(), name.end());
        item(QString::fromStdString(environments::displayName(hdri)), value, !resolved.empty() && environments::resolve(value) == resolved);
    }
    item("Procedural Studio", environments::kProcedural, resolved.empty());
    menu->addSeparator();
    QObject::connect(menu->addAction("Browse..."), &QAction::triggered, menu, [menu] {
        QString f = ui::fileDialog::getOpenFileName(menu->parentWidget(), "Choose Environment", QString(), "Radiance HDR (*.hdr)");
        if (!f.isEmpty())
            PrefStore::get().set("general", "/viewport/env/path", f.toStdString());
    });
    menu->addSeparator();
    menu->addAction(environmentIntensityRow(menu));
    menu->addAction(environmentLightColorRow(menu));
}

namespace
{
QString barStyle(const char* name, const char* edge)
{
    return QString(R"(
        #%1 { background: #383838; %2: 1px solid #262626; }
        #%1 QToolButton { border: none; border-radius: 2px; padding: 2px; }
        #%1 QToolButton:hover { background: #4a4a4a; }
        #%1 QToolButton:checked { background: #5c5c5c; }
        #%1 QToolButton[partial="true"] { background: #444444; border: 1px solid #5c5c5c; }
        #%1 QToolButton::menu-indicator { image: none; width: 0; }
        #%1 QDoubleSpinBox { background: #2a2a2a; border: 1px solid #262626; padding: 0 2px; }
        #%1 QFrame#toolbarSeparator { color: #5a5a5a; }
    )")
        .arg(name, edge);
}

QFrame* separatorLine(QWidget* parent, int width = 13)
{
    auto* line = new QFrame(parent);
    line->setFrameShape(QFrame::VLine);
    line->setObjectName("toolbarSeparator");
    line->setFixedSize(width, kHeight - 8);
    return line;
}

bool isSeparator(const QWidget* w)
{
    return w->objectName() == "toolbarSeparator";
}

QToolButton* makeChevron(QWidget* parent, const QString& tip)
{
    auto* chevron = new QToolButton(parent);
    chevron->setIcon(icon("chevrons-right"));
    chevron->setIconSize(QSize(kIcon, kIcon));
    chevron->setToolTip(tip);
    chevron->setPopupMode(QToolButton::InstantPopup);
    chevron->setMenu(new QMenu(chevron));
    return chevron;
}

void setRendererSetting(int panel, const std::string& key, json value)
{
    try
    {
        ops::execute("viewportRendererSetting", {{"args", {key, std::move(value)}}, {"panel", panel}}, CallSource::UI);
    }
    catch (const CommandError& e)
    {
        log::error("{}", e.what());
    }
}

/** The settings of the renderer a viewport draws with, applied as they are edited. */
void showRendererSettings(QWidget* parent, int panel)
{
    auto* dialog = new QDialog(parent);
    dialog->setAttribute(Qt::WA_DeleteOnClose);
    dialog->setWindowTitle(QString::fromStdString(viewportRenderers::selected(panel)) + " Settings");
    auto* form = new QFormLayout(dialog);
    for (const viewportRenderers::Setting& s : viewportRenderers::settings(panel))
    {
        const QString label = QString::fromStdString(s.name.empty() ? s.key : s.name);
        const std::string key = s.key;
        QWidget* field = nullptr;
        if (s.value.is_boolean())
        {
            auto* box = new QCheckBox(dialog);
            box->setChecked(s.value.get<bool>());
            QObject::connect(box, &QCheckBox::toggled, dialog, [panel, key](bool on) { setRendererSetting(panel, key, on); });
            field = box;
        }
        else if (s.value.is_number_integer())
        {
            auto* spin = new QSpinBox(dialog);
            spin->setRange(std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
            spin->setValue(s.value.get<int>());
            QObject::connect(spin, &QSpinBox::editingFinished, dialog, [panel, key, spin] { setRendererSetting(panel, key, spin->value()); });
            field = spin;
        }
        else if (s.value.is_number())
        {
            auto* spin = new QDoubleSpinBox(dialog);
            spin->setRange(-1e9, 1e9);
            spin->setDecimals(4);
            spin->setValue(s.value.get<double>());
            QObject::connect(spin, &QDoubleSpinBox::editingFinished, dialog, [panel, key, spin] { setRendererSetting(panel, key, spin->value()); });
            field = spin;
        }
        else if (s.value.is_string())
        {
            auto* edit = new QLineEdit(QString::fromStdString(s.value.get<std::string>()), dialog);
            QObject::connect(edit, &QLineEdit::editingFinished, dialog, [panel, key, edit] { setRendererSetting(panel, key, edit->text().toStdString()); });
            field = edit;
        }
        if (field)
        {
            field->setToolTip(QString::fromStdString(key));
            form->addRow(label, field);
        }
    }
    dialog->show();
}
} // namespace

ViewportToolbar::ViewportToolbar(ViewportWidget* view, QWidget* parent) : QWidget(parent), mView(view)
{
    namespace vc = viewportContext;
    setObjectName("viewportToolbar");
    setAttribute(Qt::WA_StyledBackground);
    setFixedHeight(kHeight);
    setStyleSheet(barStyle("viewportToolbar", "border-bottom"));

    mCameraButton = menuButton("camera", "Camera", [this](QMenu* m) { fillCameraMenu(m, mView, [this] { activateView(); }); });
    // Right-click selects the camera node this view looks through, as Maya's panel camera button does.
    mCameraButton->setContextMenuPolicy(Qt::CustomContextMenu);
    connect(mCameraButton, &QWidget::customContextMenuRequested, this, [this] {
        const scene::Node* n = scene::Graph::current().node(mView->lookThroughCamera());
        if (!n)
        {
            log::info("{} is a built-in view; there is no camera node to select.", mView->viewName());
            return;
        }
        try
        {
            UndoChunk chunk("Select Camera");
            ops::execute("selectMode", {{"args", {"object"}}}, CallSource::UI);
            ops::execute("select", {{"args", {n->path()}}}, CallSource::UI);
        }
        catch (const CommandError& e)
        {
            log::error("{}", e.what());
        }
    });
    actionButton("viewport.cameraSettings", "camera-cog");
    separator();
    actionButton("viewport.toggleGrid", "grid-4x4");
    actionButton("viewport.toggleFilmGate", "frame");
    actionButton("viewport.toggleResolutionGate", "aspect-ratio");
    // Click toggles this view's image planes; a long press opens the camera's image plane menu; Shift+click tears off
    // Create > Reference Images.
    QToolButton* planes = actionButton("viewport.toggleImagePlanes", "photo");
    planes->installEventFilter(new ShiftClickTearOff(planes, {"Create", "Reference Images"}));
    auto* planeMenu = new QMenu(planes);
    connect(planeMenu, &QMenu::aboutToShow, this, [this, planeMenu] {
        activateView();
        planeMenu->clear();
        imagePlanes::fillCameraMenu(planeMenu, mView->viewName());
    });
    planes->setMenu(planeMenu);
    planes->setPopupMode(QToolButton::DelayedPopup);
    QToolButton* hudButton = menuButton("device-desktop-analytics", "Heads Up Display", [this](QMenu* m) { hud::fillMenu(m, mView); });
    // Decorated after the fill above is connected, so the option box squares go onto the rows it just made.
    optionBox::decorateMenu(hudButton->menu());
    separator();
    actionButton("viewport.wireframe", "cube");
    actionButton("viewport.shaded", "sphere");
    actionButton("viewport.textured", "texture");
    actionButton("viewport.wireOnShaded", "box-model");
    actionButton("viewport.defaultMaterial", "ball-bowling", vc::Materials);
    actionButton("viewport.greyBaseColor", "contrast", vc::Materials);
    menuButton("palette", "Vertex Colors", [this](QMenu* m) {
        for (const char* group : {"vertexColors", "colorChannel"})
        {
            auto* exclusive = new QActionGroup(m);
            for (const char* v : std::string(group) == "vertexColors" ? std::initializer_list<const char*>{"off", "shaded", "unlit"}
                                                                       : std::initializer_list<const char*>{"rgb", "r", "g", "b", "a"})
            {
                std::string id = std::string("viewport.") + group + "." + v;
                QAction* a = m->addAction(labelOf(id));
                a->setCheckable(true);
                a->setChecked(viewportActionState(id, mView).value_or(false));
                exclusive->addAction(a);
                connect(a, &QAction::triggered, this, [id] { actions::run(id); });
            }
            m->addSeparator();
        }
        QAction* gray = m->addAction(labelOf("viewport.grayscaleWeights"));
        gray->setCheckable(true);
        gray->setChecked(viewportActionState("viewport.grayscaleWeights", mView).value_or(false));
        connect(gray, &QAction::triggered, this, [] { actions::run("viewport.grayscaleWeights"); });
    });
    actionButton("viewport.monotone", "palette");
    separator();
    menuButton("aperture", "Renderer", [this](QMenu* m) {
        auto* group = new QActionGroup(m);
        const std::string current = viewportRenderers::selected(mView->hudPanel());
        for (const auto& [id, name] : viewportRenderers::choices())
        {
            QAction* a = m->addAction(QString::fromStdString(name));
            a->setCheckable(true);
            a->setChecked(id == current);
            group->addAction(a);
            connect(a, &QAction::triggered, this, [this, id] {
                activateView();
                try
                {
                    ops::execute("viewportRenderer", {{"args", {id}}, {"panel", mView->hudPanel()}}, CallSource::UI);
                }
                catch (const CommandError& e)
                {
                    log::error("{}", e.what());
                }
            });
        }
        if (!viewportRenderers::settings(mView->hudPanel()).empty())
        {
            m->addSeparator();
            connect(m->addAction("Renderer Settings..."), &QAction::triggered, this, [this] { showRendererSettings(this, mView->hudPanel()); });
        }
    });
    mLilaRendererOnly = {actionButton("viewport.toggleShadows", "shadow"), actionButton("viewport.toggleAO", "blur"),
                         actionButton("viewport.toggleAA", "wave-saw-tool")};
    mSamplesButton = menuButton({}, "Antialiasing Samples", [this](QMenu* m) {
        auto* group = new QActionGroup(m);
        for (const char* n : {"2", "4", "8"})
        {
            std::string id = std::string("viewport.aaSamples") + n;
            QAction* a = m->addAction(QString(n) + "x");
            a->setCheckable(true);
            a->setChecked(!vd::temporalAa() && PrefStore::get().value<int>("general", vd::kAaSamples, vd::kDefaultSamples) == std::stoi(n));
            group->addAction(a);
            connect(a, &QAction::triggered, this, [id] { actions::run(id); });
        }
        QAction* temporal = m->addAction("Temporal");
        temporal->setCheckable(true);
        temporal->setChecked(vd::temporalAa());
        temporal->setToolTip("One sample a frame, refined once the view stops");
        group->addAction(temporal);
        connect(temporal, &QAction::triggered, this, [] { actions::run("viewport.aaTemporal"); });
    });
    mSamplesButton->setToolButtonStyle(Qt::ToolButtonTextOnly);
    mLilaRendererOnly.push_back(mSamplesButton);
    separator();
    actionButton("viewport.toggleEnvBackground", "background");
    mLilaRendererOnly.push_back(menuButton("sun", "Lighting", [this](QMenu* m) {
        auto* group = new QActionGroup(m);
        for (const char* mode : {"environment", "studio", "flat"})
        {
            std::string id = std::string("viewport.lighting.") + mode;
            QAction* a = m->addAction(labelOf(id));
            a->setCheckable(true);
            a->setChecked(viewportActionState(id, mView).value_or(false));
            group->addAction(a);
            connect(a, &QAction::triggered, this, [id] { actions::run(id); });
        }
    }));
    menuButton("world", "Environment", [](QMenu* m) { fillEnvironmentMenu(m); });
    separator();
    actionButton("viewport.isolate", "focus-2");
    actionButton("viewport.show.curve", "vector-spline");
    actionButton("viewport.show.mesh", "box");
    actionButton("viewport.show.joint", "bone");
    actionButton("viewport.show.light", "bulb");
    separator();
    actionButton("viewport.toggleBackfaceCulling", "square-toggle");
    actionButton("viewport.toggleOutline", "border-outer");
    menuButton("stack-2", "Level of Detail", [this](QMenu* m) { lodDisplay::fillMenu(m, mView); });
    separator();
    actionButton("viewport.frameSelected", "focus-centered");
    actionButton("viewport.frameAll", "arrows-maximize");
    actionButton("viewport.frameUnderCursor", "focus");

    mChevron = makeChevron(this, "More viewport options");
    QMenu* overflow = mChevron->menu();
    optionBox::decorateMenu(overflow);
    connect(overflow, &QMenu::aboutToShow, this, [this, overflow] {
        overflow->clear();
        for (const Item& item : mItems)
            if (item.overflow && item.relevant && item.widget->isHidden())
                item.overflow(overflow);
    });

    mPrefListener = PrefStore::get().onChanged([this](const std::string& file) {
        if (file == "general")
            refresh();
    });
    mRenderSettingsListener = scene::addRenderSettingsListener([this] { refresh(); });
    mToolsListener = tools::addListener([this] { refresh(); });
    mRendererListener = viewportRenderers::addListener([this](int panel) {
        if (panel == mView->hudPanel())
            refresh();
    });
    mContextListener = viewportContext::addListener([this] {
        if (refreshRelevance())
            layoutItems();
    });
    connect(mView, &ViewportWidget::viewStateChanged, this, &ViewportToolbar::refresh);
    refresh();
    refreshTooltips();
}

ViewportToolbar::~ViewportToolbar()
{
    PrefStore::get().removeOnChanged(mPrefListener);
    scene::removeRenderSettingsListener(mRenderSettingsListener);
    tools::removeListener(mToolsListener);
    viewportRenderers::removeListener(mRendererListener);
    viewportContext::removeListener(mContextListener);
}

QToolButton* ViewportToolbar::actionButton(const char* id, const char* iconName, uint32_t needs)
{
    auto* b = new QToolButton(this);
    std::string action = id;
    QAction* a = new QAction(icon(iconName), labelOf(action), b);
    a->setCheckable(viewportActionState(action, mView).has_value());
    b->setDefaultAction(a);
    b->setIconSize(QSize(kIcon, kIcon));
    // The action's own checked state comes from refresh(); the click only runs the registered action.
    connect(a, &QAction::triggered, this, [this, action] {
        activateView();
        actions::run(action);
        refresh();
    });
    mActionButtons.emplace_back(b, action);
    mItems.push_back({b, [a](QMenu* m) { m->addAction(a); }, needs, needs ? action : std::string()});
    return b;
}

QToolButton* ViewportToolbar::menuButton(const char* iconName, const QString& tip, std::function<void(QMenu*)> fill)
{
    auto* b = new QToolButton(this);
    if (iconName)
        b->setIcon(icon(iconName));
    b->setIconSize(QSize(kIcon, kIcon));
    b->setToolTip(tip);
    b->setPopupMode(QToolButton::InstantPopup);
    auto* menu = new QMenu(b);
    b->setMenu(menu);
    connect(menu, &QMenu::aboutToShow, this, [this, menu, fill] {
        activateView();
        menu->clear();
        fill(menu);
    });
    mItems.push_back({b, [this, tip, iconName, fill](QMenu* m) {
                          QMenu* sub = m->addMenu(iconName ? icon(iconName) : QIcon(), tip);
                          connect(sub, &QMenu::aboutToShow, this, [sub, fill] {
                              sub->clear();
                              fill(sub);
                          });
                      }});
    return b;
}

void ViewportToolbar::separator()
{
    mItems.push_back({separatorLine(this), nullptr});
}

void ViewportToolbar::activateView()
{
    mView->setFocus(Qt::OtherFocusReason);
}

void ViewportToolbar::refresh()
{
    for (auto& [b, id] : mActionButtons)
        if (std::optional<bool> on = viewportActionState(id, mView))
        {
            QSignalBlocker block(b->defaultAction());
            b->defaultAction()->setChecked(*on);
            if (id == "viewport.monotone")
                b->defaultAction()->setIcon(icon(*on ? "palette-off" : "palette"));
        }
    mCameraButton->setToolTip("Camera: " + QString::fromStdString(mView->viewName()) + "\nRight-click to select it");
    int samples = PrefStore::get().value<int>("general", vd::kAaSamples, vd::kDefaultSamples);
    mSamplesButton->setText(vd::temporalAa() ? QString("TAA") : QString::number(samples) + "x");
    const bool lilaDraws = viewportRenderers::selected(mView->hudPanel()) == viewportRenderers::kLila;
    for (QToolButton* b : mLilaRendererOnly)
    {
        if (QAction* a = b->defaultAction())
            a->setEnabled(lilaDraws);
        b->setEnabled(lilaDraws);
    }
    mSamplesButton->setEnabled(lilaDraws && generalPref(vd::kAaEnabled, true));
    if (refreshRelevance())
        layoutItems();
}

bool ViewportToolbar::refreshRelevance()
{
    const uint32_t present = generalPref(vd::kHideUnusedItems, true) ? viewportContext::present() : ~0u;
    bool changed = false;
    for (Item& item : mItems)
    {
        const bool relevant = !item.needs || (present & item.needs) || (!item.stateId.empty() && viewportActionState(item.stateId, mView).value_or(false));
        changed |= std::exchange(item.relevant, relevant) != relevant;
    }
    return changed;
}

void ViewportToolbar::refreshTooltips()
{
    for (auto& [b, id] : mActionButtons)
    {
        b->defaultAction()->setText(labelOf(id));
        b->defaultAction()->setToolTip(tipFor(id));
    }
}

QSize ViewportToolbar::sizeHint() const
{
    int w = 4;
    for (const Item& item : mItems)
        if (item.relevant)
            w += item.widget->sizeHint().width() + kSpacing;
    return {w, kHeight};
}

QSize ViewportToolbar::minimumSizeHint() const
{
    return {0, kHeight};
}

void ViewportToolbar::layoutItems()
{
    const int chevronWidth = mChevron->sizeHint().width();
    const int available = width() - 2;
    size_t last = 0;
    for (size_t i = 0; i < mItems.size(); ++i)
        if (mItems[i].relevant)
            last = i;
    int x = 2;
    bool overflowed = false;
    // Starts true so a row never opens with a separator; also drops the second of two that a hidden run left adjacent.
    bool afterSeparator = true;
    for (size_t i = 0; i < mItems.size(); ++i)
    {
        QWidget* w = mItems[i].widget;
        const bool line = isSeparator(w);
        if (!mItems[i].relevant || (line && afterSeparator))
        {
            w->setVisible(false);
            continue;
        }
        QSize s = w->sizeHint();
        if (QSize fixed = w->maximumSize(); fixed.width() < QWIDGETSIZE_MAX)
            s = fixed;
        // The last item may use the chevron's space; anything before it has to leave room for the chevron.
        int limit = i == last ? available : available - chevronWidth;
        if (!overflowed && x + s.width() > limit)
            overflowed = true;
        w->setVisible(!overflowed);
        if (!overflowed)
        {
            w->setGeometry(x, (kHeight - s.height()) / 2, s.width(), s.height());
            x += s.width() + kSpacing;
            afterSeparator = line;
        }
    }
    mChevron->setVisible(overflowed);
    if (overflowed)
    {
        QSize s = mChevron->sizeHint();
        mChevron->setGeometry(available - s.width(), (kHeight - s.height()) / 2, s.width(), s.height());
    }
}

void ViewportToolbar::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    layoutItems();
}

bool ViewportToolbar::event(QEvent* e)
{
    // Hotkeys are live-editable, so tooltips pick up the current binding whenever the pointer comes over the row.
    if (e->type() == QEvent::Enter)
        refreshTooltips();
    else if (e->type() == QEvent::Show || e->type() == QEvent::LayoutRequest)
        layoutItems();
    return QWidget::event(e);
}

namespace
{
constexpr int kGroupGap = 17;
} // namespace

ViewportBottomBar::ViewportBottomBar(ViewportWidget* view, QWidget* parent) : QWidget(parent), mView(view)
{
    setObjectName("viewportBottomBar");
    setAttribute(Qt::WA_StyledBackground);
    setFixedHeight(kHeight);
    setStyleSheet(barStyle("viewportBottomBar", "border-top"));

    buildHighlightRow();
    buildMaskRow();
    buildDisplayRow();
    buildRetopoRow();
    buildSymmetryRow();
    mSnapRow = group(true, 4, nullptr);
    snapGroup(mSnapRow, Tool::Move, "grid-dots", {0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000}, " cm");
    snapGroup(mSnapRow, Tool::Rotate, "angle", {}, QString(QChar(0x00B0)));
    snapGroup(mSnapRow, Tool::Scale, "resize", {0.0625, 0.1, 0.125, 0.25, 0.5, 1}, "");
    buildExposureRow();
    buildPresenterRow();

    mChevron = makeChevron(this, "More viewport options");
    QMenu* overflow = mChevron->menu();
    connect(overflow, &QMenu::aboutToShow, this, [this, overflow] {
        overflow->clear();
        for (const Group& g : mGroups)
            if (g.row->isHidden() && (!g.active || g.active()))
            {
                if (!overflow->isEmpty())
                    overflow->addSeparator();
                g.overflow(overflow);
            }
    });

    mPrefListener = PrefStore::get().onChanged([this](const std::string& file) {
        if (file == "general")
            refresh();
    });
    mToolsListener = tools::addListener([this] { refresh(); });
    mViewportToolListener = viewportTools::addListener([this] { refreshRetopo(); });
    mContextListener = viewportContext::addListener([this] {
        if (refreshRelevance())
            layoutItems();
    });
    if (mView)
        connect(mView, &ViewportWidget::viewStateChanged, this, &ViewportBottomBar::refresh);
    refresh();
    refreshTooltips();
}

ViewportBottomBar::~ViewportBottomBar()
{
    PrefStore::get().removeOnChanged(mPrefListener);
    tools::removeListener(mToolsListener);
    viewportTools::removeListener(mViewportToolListener);
    viewportContext::removeListener(mContextListener);
    scene::Graph::current().removeListener(mGraphListener);
}

QWidget* ViewportBottomBar::group(bool right, int keep, std::function<void(QMenu*)> overflow, std::function<bool()> active)
{
    auto* row = new QWidget(this);
    auto* layout = new QHBoxLayout(row);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(kSpacing);
    if (!overflow)
        overflow = [row](QMenu* m) {
            for (QToolButton* b : row->findChildren<QToolButton*>(Qt::FindDirectChildrenOnly))
                if (QAction* a = b->defaultAction(); a && !b->isHidden())
                    m->addAction(a);
        };
    mGroups.push_back({row, separatorLine(this, kGroupGap), right, keep, std::move(active), std::move(overflow)});
    return row;
}

QToolButton* ViewportBottomBar::toggleButton(QWidget* row, const std::string& action, const char* iconName, uint32_t needs)
{
    auto* b = new QToolButton(row);
    auto* a = new QAction(icon(iconName), labelOf(action), b);
    a->setCheckable(true);
    b->setDefaultAction(a);
    b->setIconSize(QSize(kIcon, kIcon));
    connect(a, &QAction::triggered, this, [this, action] {
        activateView();
        actions::run(action);
        refresh();
    });
    mActionButtons.emplace_back(b, action);
    if (needs)
        mRelevant.push_back({b, needs, action});
    row->layout()->addWidget(b);
    return b;
}

QDoubleSpinBox* ViewportBottomBar::field(QWidget* row, const char* key, double fallback, double lo, double hi, double step, const QString& tip)
{
    auto* s = new QDoubleSpinBox(row);
    s->setRange(lo, hi);
    s->setSingleStep(step);
    s->setDecimals(2);
    s->setKeyboardTracking(false);
    s->setButtonSymbols(QAbstractSpinBox::NoButtons);
    s->setAlignment(Qt::AlignRight);
    s->setFixedSize(42, kHeight - 6);
    s->setToolTip(tip);
    setSpinDefault(s, fallback);
    std::string k = key;
    connect(s, &QDoubleSpinBox::valueChanged, this, [k](double v) { PrefStore::get().set("general", k, v); });
    connect(s, &QDoubleSpinBox::editingFinished, this, [this] { activateView(); });
    mFields.push_back({s, {k, fallback}});
    row->layout()->addWidget(s);
    return s;
}

void ViewportBottomBar::snapGroup(QWidget* row, Tool t, const char* iconName, std::vector<double> steps, const QString& unit)
{
    auto* layout = static_cast<QHBoxLayout*>(row->layout());
    if (!mSnapGroups.empty())
        layout->addWidget(separatorLine(row));
    std::string action = std::string("snap.") + tools::snapKey(t) + "Increment";
    QToolButton* toggle = toggleButton(row, action, iconName);

    auto* value = new QToolButton(row);
    value->setToolButtonStyle(Qt::ToolButtonTextOnly);
    value->setPopupMode(QToolButton::InstantPopup);
    value->setToolTip(labelOf(action) + " increment" + (unit.trimmed().isEmpty() ? QString() : " (" + unit.trimmed() + ")"));
    auto* menu = new QMenu(value);
    value->setMenu(menu);
    std::string stepKey = snapPref(t, "step");
    connect(menu, &QMenu::aboutToShow, this, [this, t, menu, steps, unit, stepKey] {
        menu->clear();
        double current = tools::incrementSnap(t).step;
        bool custom = true;
        if (t == Tool::Rotate)
        {
            bool matched = false;
            menu->addAction(rotateStepColumns(menu, current, stepKey, matched));
            custom = !matched;
        }
        auto* group = new QActionGroup(menu);
        for (double v : steps)
        {
            bool on = std::abs(v - current) < 1e-9;
            custom = custom && !on;
            QAction* item = menu->addAction(t == Tool::Scale && v == 0.1 ? QString("0.1 (10%)") : stepText(v) + unit);
            item->setCheckable(true);
            item->setChecked(on);
            group->addAction(item);
            connect(item, &QAction::triggered, this, [stepKey, v] { PrefStore::get().set("general", stepKey, v); });
        }
        menu->addSeparator();
        QAction* other = menu->addAction(custom ? "Custom (" + stepText(current) + unit + ")..." : QString("Custom..."));
        other->setCheckable(true);
        other->setChecked(custom);
        connect(other, &QAction::triggered, this, [this, t, stepKey, current] {
            bool ok = false;
            double v = QInputDialog::getDouble(this, "Snap Increment", labelOf(std::string("snap.") + tools::snapKey(t) + "Increment"), current,
                                               1e-4, 1e6, 4, &ok);
            if (ok && v > 0.0)
                PrefStore::get().set("general", stepKey, v);
        });
        if (t == Tool::Move)
        {
            std::string absKey = snapPref(t, "absolute");
            QAction* abs = menu->addAction("Snap to World Grid");
            abs->setCheckable(true);
            abs->setChecked(PrefStore::get().value<bool>("general", absKey, false));
            abs->setToolTip("Land on multiples of the increment instead of stepping from where the drag started");
            connect(abs, &QAction::triggered, this, [absKey](bool on) { PrefStore::get().set("general", absKey, on); });
        }
    });

    layout->addWidget(value);
    mSnapGroups.push_back({t, toggle, value, unit});
}

void ViewportBottomBar::buildHighlightRow()
{
    QWidget* row = group(false, 0, nullptr);
    toggleButton(row, "viewport.toggleHighlightSelection", "highlight");
    toggleButton(row, "viewport.togglePreselectObjects", "pointer");
    toggleButton(row, "viewport.togglePreselectComponents", "vector-triangle");
}

void ViewportBottomBar::buildMaskRow()
{
    namespace sm = scene::selectionMask;
    QWidget* row = group(false, 1, nullptr);
    for (auto [kind, iconName] : {std::pair{sm::Kind::Curve, "vector-spline"}, {sm::Kind::Mesh, "box"}, {sm::Kind::Joint, "bone"}, {sm::Kind::Light, "bulb"},
                                  {sm::Kind::Locator, "crosshair"}, {sm::Kind::Other, "dots"}})
        toggleButton(row, std::string("select.mask.") + sm::name(kind), iconName);
}

void ViewportBottomBar::buildDisplayRow()
{
    QWidget* row = group(false, 1, nullptr);
    toggleButton(row, "viewport.hud.originAxis", "axis-tripod");
    toggleButton(row, "viewport.toggleXray", "ghost");
    toggleButton(row, "rig.xrayJoints", "skull", viewportContext::Joints);
    toggleButton(row, "viewport.toggleKuwahara", "brush");
    auto* aids = new QToolButton(row);
    aids->setIcon(icon("eye"));
    aids->setIconSize(QSize(kIcon, kIcon));
    aids->setToolTip("Mesh Display");
    aids->setPopupMode(QToolButton::InstantPopup);
    auto* menu = new QMenu(aids);
    aids->setMenu(menu);
    connect(menu, &QMenu::aboutToShow, this, [this, menu] {
        activateView();
        menu->clear();
        displayAids::fillMenu(menu, mView);
    });
    row->layout()->addWidget(aids);
    mGroups.back().overflow = [this, row](QMenu* m) {
        for (QToolButton* b : row->findChildren<QToolButton*>(Qt::FindDirectChildrenOnly))
            if (QAction* a = b->defaultAction(); a && !b->isHidden())
                m->addAction(a);
        QMenu* sub = m->addMenu(icon("eye"), "Mesh Display");
        connect(sub, &QMenu::aboutToShow, this, [this, sub] {
            sub->clear();
            displayAids::fillMenu(sub, mView);
        });
    };
}

void ViewportBottomBar::buildExposureRow()
{
    QWidget* row = group(true, 0, nullptr);
    toggleButton(row, "viewport.resetExposure", "exposure")->defaultAction()->setCheckable(false);
    field(row, vd::kExposure, 0.0, -10.0, 10.0, 0.1, "Exposure (stops)");
    field(row, vd::kGamma, 1.0, 0.1, 4.0, 0.05, "Gamma");
}

void ViewportBottomBar::buildPresenterRow()
{
    QWidget* row = group(true, 2, nullptr);
    toggleButton(row, "view.presenter", "presentation");
    mPresenterLayout = toggleButton(row, "view.presenterLayout", "layout-board-split");
}

void ViewportBottomBar::buildSymmetryRow()
{
    mSymmetryRow = group(true, 3, [this](QMenu* m) { m->addAction(mSymmetryToggle->defaultAction()); });
    auto* layout = static_cast<QHBoxLayout*>(mSymmetryRow->layout());
    mSymmetryToggle = new QToolButton(mSymmetryRow);
    auto* a = new QAction(icon("flip-vertical"), labelOf("mesh.toggleSymmetry"), mSymmetryToggle);
    a->setCheckable(true);
    mSymmetryToggle->setDefaultAction(a);
    mSymmetryToggle->setIconSize(QSize(kIcon, kIcon));
    connect(a, &QAction::triggered, this, [this] {
        activateView();
        actions::run("mesh.toggleSymmetry");
        refreshSymmetry();
    });

    mSymmetryAxis = new QToolButton(mSymmetryRow);
    mSymmetryAxis->setToolButtonStyle(Qt::ToolButtonTextOnly);
    mSymmetryAxis->setPopupMode(QToolButton::InstantPopup);
    mSymmetryAxis->setToolTip("Symmetry axis (object space); picking one turns symmetry on with it");
    auto* menu = new QMenu(mSymmetryAxis);
    mSymmetryAxis->setMenu(menu);
    connect(menu, &QMenu::aboutToShow, this, [this, menu] {
        menu->clear();
        scene::Graph& g = scene::Graph::current();
        scene::SymmetryState st = scene::symmetryState(g, g.selection());
        auto* group = new QActionGroup(menu);
        for (int i = 0; i < 3; ++i)
        {
            std::string id = std::string("mesh.symmetry") + char('X' + i);
            QAction* item = menu->addAction(labelOf(id));
            item->setCheckable(true);
            item->setChecked(st.on > 0 && st.axis == i);
            group->addAction(item);
            connect(item, &QAction::triggered, this, [this, i, id] {
                PrefStore::get().set("general", "/symmetry/axis", std::string(1, char('x' + i)));
                activateView();
                actions::run(id);
                refreshSymmetry();
            });
        }
        // Radial: the setting, then its count and mirror; picking either turns radial symmetry on with it.
        QAction* radial = menu->addAction(labelOf("mesh.symmetryRadial"));
        radial->setCheckable(true);
        radial->setChecked(st.on > 0 && st.axis == scene::SymmetryState::kRadialState);
        group->addAction(radial);
        connect(radial, &QAction::triggered, this, [this] {
            PrefStore::get().set("general", "/symmetry/axis", std::string("radial"));
            activateView();
            actions::run("mesh.symmetryRadial");
            refreshSymmetry();
        });
        auto setRadial = [this](json flags) {
            scene::Graph& graph = scene::Graph::current();
            json names = json::array();
            for (scene::NodeId id : graph.selection())
                if (const scene::Node* n = graph.node(id); n && n->type().isA("mesh"))
                    names.push_back(n->path());
            if (names.empty())
                return;
            flags["args"] = names;
            flags["axis"] = "radial";
            PrefStore::get().set("general", "/symmetry/axis", std::string("radial"));
            activateView();
            try
            {
                ops::execute("setSymmetry", flags, CallSource::UI);
            }
            catch (const std::exception& e)
            {
                log::warn("{}", e.what());
            }
            refreshSymmetry();
        };
        QMenu* counts = menu->addMenu("Radial Count");
        for (int count : {2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32})
        {
            QAction* item = counts->addAction(QString::number(count));
            item->setCheckable(true);
            item->setChecked(st.axis == scene::SymmetryState::kRadialState && st.radialCount == count);
            connect(item, &QAction::triggered, this, [setRadial, count] { setRadial({{"count", count}}); });
        }
        QAction* mirrored = menu->addAction("Radial Mirror");
        mirrored->setCheckable(true);
        mirrored->setChecked(st.axis == scene::SymmetryState::kRadialState && st.radialCount > 0 && st.radialMirror);
        mirrored->setToolTip("Also mirror each wedge across a plane through the axis (dihedral symmetry: rims, flowers)");
        connect(mirrored, &QAction::triggered, this, [setRadial](bool on) { setRadial({{"mirror", on}}); });
        menu->addSeparator();
        bool custom = false;
        for (scene::NodeId id : g.selection())
            if (const scene::Node* n = g.node(id); n && n->type().isA("mesh") && n->as<int64_t>(scene::MeshNode::kSymmetryPlane) == 1)
                custom = true;
        QAction* fromSel = menu->addAction(labelOf("mesh.symmetryPlaneFromSelection"));
        fromSel->setEnabled(scene::components::mode() == scene::ComponentMode::Vertex || scene::components::mode() == scene::ComponentMode::Edge);
        connect(fromSel, &QAction::triggered, this, [this] {
            activateView();
            actions::run("mesh.symmetryPlaneFromSelection");
            refreshSymmetry();
        });
        QAction* atPivot = menu->addAction(labelOf("mesh.symmetryPlaneFromPivot"));
        atPivot->setCheckable(true);
        atPivot->setChecked(!custom);
        atPivot->setEnabled(custom);
        connect(atPivot, &QAction::triggered, this, [this] {
            activateView();
            actions::run("mesh.symmetryPlaneFromPivot");
            refreshSymmetry();
        });
        menu->addSeparator();
        QAction* symSelect = menu->addAction(labelOf("select.toggleSymmetric"));
        symSelect->setCheckable(true);
        symSelect->setChecked(PrefStore::get().value<bool>("general", "/select/symmetric", true));
        connect(symSelect, &QAction::triggered, this, [] { actions::run("select.toggleSymmetric"); });
        QAction* symLine = menu->addAction(labelOf("viewport.toggleSymmetryLine"));
        symLine->setCheckable(true);
        symLine->setChecked(PrefStore::get().value<bool>("general", viewportDisplay::kSymmetryEdges, true));
        connect(symLine, &QAction::triggered, this, [] { actions::run("viewport.toggleSymmetryLine"); });
        QAction* symmetrize = menu->addAction(labelOf("select.symmetrize"));
        symmetrize->setEnabled(scene::components::mode() != scene::ComponentMode::Object);
        connect(symmetrize, &QAction::triggered, this, [this] {
            activateView();
            actions::run("select.symmetrize");
        });
        menu->addSeparator();
        QAction* off = menu->addAction(labelOf("mesh.symmetryOff"));
        off->setEnabled(st.on > 0);
        connect(off, &QAction::triggered, this, [this] {
            activateView();
            actions::run("mesh.symmetryOff");
            refreshSymmetry();
        });
    });
    layout->addWidget(mSymmetryToggle);
    layout->addWidget(mSymmetryAxis);

    // Value changes arrive per drag step, so coalesce them; symmetry only needs the latest state.
    mSymmetryRefresh = new QTimer(this);
    mSymmetryRefresh->setSingleShot(true);
    mSymmetryRefresh->setInterval(30);
    connect(mSymmetryRefresh, &QTimer::timeout, this, &ViewportBottomBar::refreshSymmetry);
    mGraphListener = scene::Graph::current().addListener([this](scene::Change c, scene::NodeId) {
        if (c != scene::Change::Renamed && c != scene::Change::Reparented)
            mSymmetryRefresh->start();
    });
}

void ViewportBottomBar::buildRetopoRow()
{
    mRetopoRow = group(true, 2, nullptr, [] { return activeRetopoDraw() != nullptr; });
    auto* layout = static_cast<QHBoxLayout*>(mRetopoRow->layout());
    auto* title = new QLabel("Retopo", mRetopoRow);
    title->setStyleSheet("color: #8c8c8c; padding: 0 4px;");
    layout->addWidget(title);

    mRetopoSpace = new QToolButton(mRetopoRow);
    auto* space = new QAction(labelOf("retopo.toggleSnapSpace"), mRetopoSpace);
    space->setCheckable(true);
    mRetopoSpace->setDefaultAction(space);
    mRetopoSpace->setToolButtonStyle(Qt::ToolButtonTextOnly);
    connect(space, &QAction::triggered, this, [this] {
        actions::run("retopo.toggleSnapSpace");
        refreshRetopo();
    });
    layout->addWidget(mRetopoSpace);

    mRetopoAngle = new QToolButton(mRetopoRow);
    mRetopoAngle->setToolButtonStyle(Qt::ToolButtonTextOnly);
    mRetopoAngle->setPopupMode(QToolButton::InstantPopup);
    mRetopoAngle->setToolTip("Retopo Draw snap angle: Shift places the next dot on a line from the last one at a multiple of this");
    auto* menu = new QMenu(mRetopoAngle);
    mRetopoAngle->setMenu(menu);
    connect(menu, &QMenu::aboutToShow, this, [this, menu] {
        menu->clear();
        double current = RetopoDrawTool::snapAngle();
        bool matched = false;
        menu->addAction(rotateStepColumns(menu, current, "/retopoDraw/snapAngle", matched));
        menu->addSeparator();
        QAction* other = menu->addAction(!matched ? "Custom (" + stepText(current) + QChar(0x00B0) + ")..." : QString("Custom..."));
        other->setCheckable(true);
        other->setChecked(!matched);
        connect(other, &QAction::triggered, this, [this, current] {
            bool ok = false;
            double v = QInputDialog::getDouble(this, "Retopo Snap Angle", "Snap angle (degrees)", current, 0.1, 180.0, 2, &ok);
            if (ok)
                RetopoDrawTool::setSnapAngle(v);
        });
    });
    layout->addWidget(mRetopoAngle);
    layout->addWidget(separatorLine(mRetopoRow));

    mRetopoSurface = new QToolButton(mRetopoRow);
    auto* surface = new QAction(icon("target"), labelOf("retopo.setReference"), mRetopoSurface);
    mRetopoSurface->setDefaultAction(surface);
    mRetopoSurface->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    mRetopoSurface->setIconSize(QSize(kIcon, kIcon));
    connect(surface, &QAction::triggered, this, [this] {
        activateView();
        actions::run("retopo.setReference");
        refreshRetopo();
    });
    layout->addWidget(mRetopoSurface);
}

void ViewportBottomBar::refreshRetopo()
{
    RetopoDrawTool* tool = activeRetopoDraw();
    const bool active = tool != nullptr;
    if (mRetopoRow->property("active").toBool() != active)
    {
        mRetopoRow->setProperty("active", active);
        layoutItems();
    }
    if (!tool)
        return;
    const bool object = RetopoDrawTool::snapObjectSpace();
    QAction* space = mRetopoSpace->defaultAction();
    if (QString text = object ? "Object" : "World"; space->text() != text)
    {
        QSignalBlocker block(space);
        space->setChecked(object);
        space->setText(text);
        space->setToolTip(tipFor("retopo.toggleSnapSpace") + "\nNow: " + (object ? "object axes" : "world axes"));
    }
    QString angle = stepText(RetopoDrawTool::snapAngle()) + QChar(0x00B0);
    const scene::Node* ref = scene::Graph::current().node(tool->reference());
    QString name = ref ? QString::fromStdString(ref->name()) : QString("(none)");
    if (mRetopoAngle->text() != angle || mRetopoSurface->defaultAction()->text() != name)
    {
        mRetopoAngle->setText(angle);
        QAction* surface = mRetopoSurface->defaultAction();
        surface->setText(name);
        surface->setToolTip("Retopo surface: " + (ref ? QString::fromStdString(ref->path()) : QString("none")) + "\n" +
                            tipFor("retopo.setReference"));
        mRetopoRow->adjustSize();
        layoutItems();
    }
}

void ViewportBottomBar::refreshSymmetry()
{
    scene::Graph& g = scene::Graph::current();
    scene::SymmetryState st = scene::symmetryState(g, g.selection());
    const bool all = st.meshes > 0 && st.on == st.meshes;
    const bool partial = st.on > 0 && !all;
    mSymmetryRow->setEnabled(st.meshes > 0);
    {
        QSignalBlocker block(mSymmetryToggle->defaultAction());
        mSymmetryToggle->defaultAction()->setChecked(all);
    }
    if (mSymmetryToggle->property("partial").toBool() != partial)
    {
        mSymmetryToggle->setProperty("partial", partial);
        mSymmetryToggle->style()->unpolish(mSymmetryToggle);
        mSymmetryToggle->style()->polish(mSymmetryToggle);
    }
    std::string remembered = PrefStore::get().value<std::string>("general", "/symmetry/axis", "x");
    auto radialText = [&st] {
        return st.radialCount > 0 ? QString("%1%2").arg(st.radialMirror ? "D" : "R").arg(st.radialCount) : QString("R");
    };
    QString axis = st.on > 0 && st.axis < 0                      ? QString("Mix")
                   : st.axis == scene::SymmetryState::kRadialState ? radialText()
                   : st.axis >= 0                                   ? QString(QChar('X' + st.axis))
                   : remembered == "radial"                         ? QString("R")
                                                                    : QString::fromStdString(remembered).toUpper();
    const bool resized = mSymmetryAxis->text() != axis;
    mSymmetryAxis->setText(axis);

    QString state = st.meshes == 0 ? QString("No mesh selected")
                    : all          ? "On (" + axis + ")"
                    : partial      ? QString("On for %1 of %2 selected meshes; click to turn on for all").arg(st.on).arg(st.meshes)
                                   : QString("Off");
    QAction* a = mSymmetryToggle->defaultAction();
    a->setText(labelOf("mesh.toggleSymmetry"));
    a->setToolTip(tipFor("mesh.toggleSymmetry") + "\nSymmetry: " + state);
    if (resized)
    {
        mSymmetryRow->adjustSize();
        layoutItems();
    }
}

void ViewportBottomBar::activateView()
{
    if (mView)
        mView->setFocus(Qt::OtherFocusReason);
}

void ViewportBottomBar::refresh()
{
    for (auto& [b, id] : mActionButtons)
        if (std::optional<bool> on = viewportActionState(id, mView))
        {
            QSignalBlocker block(b->defaultAction());
            b->defaultAction()->setChecked(*on);
        }
    for (auto& [s, kf] : mFields)
    {
        double v = PrefStore::get().value<double>("general", kf.first, kf.second);
        if (std::abs(s->value() - v) > 1e-6 && !s->hasFocus())
        {
            QSignalBlocker block(s);
            s->setValue(v);
        }
    }
    refreshRelevance();
    for (const SnapGroup& g : mSnapGroups)
    {
        QSignalBlocker block(g.toggle->defaultAction());
        bool on = PrefStore::get().value<bool>("general", snapPref(g.tool, "enabled"), false);
        // Holding the rotate snap key flips it for the drag without touching the pref.
        if (g.tool == Tool::Rotate && (tools::snaps() & SnapRotateHold))
            on = !on;
        g.toggle->defaultAction()->setChecked(on);
        g.value->setText(stepText(tools::incrementSnap(g.tool).step) + g.unit.trimmed());
    }
    mSnapRow->adjustSize();
    if (mPresenterLayout)
        mPresenterLayout->setEnabled(actions::isEnabled(nlohmann::json("view.presenterLayout")));
    refreshRetopo();
    refreshSymmetry();
    layoutItems();
}

bool ViewportBottomBar::refreshRelevance()
{
    const uint32_t present = generalPref(vd::kHideUnusedItems, true) ? viewportContext::present() : ~0u;
    bool changed = false;
    for (const Relevant& r : mRelevant)
    {
        const bool relevant = (present & r.needs) || viewportActionState(r.action, mView).value_or(false);
        if (r.button->isHidden() == relevant)
        {
            r.button->setVisible(relevant);
            r.button->parentWidget()->adjustSize();
            changed = true;
        }
    }
    return changed;
}

void ViewportBottomBar::refreshTooltips()
{
    for (auto& [b, id] : mActionButtons)
    {
        b->defaultAction()->setText(labelOf(id));
        b->defaultAction()->setToolTip(tipFor(id));
    }
    refreshSymmetry();
}

QSize ViewportBottomBar::sizeHint() const
{
    int w = 4;
    for (const Group& g : mGroups)
        if (!g.active || g.active())
            w += g.row->sizeHint().width() + kGroupGap;
    return {w, kHeight};
}

QSize ViewportBottomBar::minimumSizeHint() const
{
    return {0, kHeight};
}

void ViewportBottomBar::layoutItems()
{
    // Visibility is applied once at the end, and only where it changes: showing or hiding a child asks for another
    // layout, which would otherwise run this again on every pass of the event loop.
    std::unordered_set<QWidget*> visible;
    std::vector<const Group*> shown;
    for (const Group& g : mGroups)
        if (!g.active || g.active())
            shown.push_back(&g);
    std::vector<const Group*> byKeep = shown;
    std::stable_sort(byKeep.begin(), byKeep.end(), [](const Group* a, const Group* b) { return a->keep > b->keep; });
    // Groups keep their places in order of `keep` while they fit; once one does not, every group after it goes too.
    auto fit = [&](int room) {
        std::vector<const Group*> kept;
        int used = 0;
        for (const Group* g : byKeep)
        {
            int w = g->row->sizeHint().width() + (kept.empty() ? 0 : kGroupGap);
            if (used + w > room)
                break;
            used += w;
            kept.push_back(g);
        }
        return kept;
    };
    const int chevronWidth = mChevron->sizeHint().width();
    std::vector<const Group*> kept = fit(width() - 4);
    const bool overflowed = kept.size() < shown.size();
    if (overflowed)
        kept = fit(width() - 6 - chevronWidth);
    auto isKept = [&](const Group* g) { return std::find(kept.begin(), kept.end(), g) != kept.end(); };

    int left = 2;
    int right = width() - 2 - (overflowed ? chevronWidth + 2 : 0);
    bool anyLeft = false, anyRight = false;
    auto place = [&](QWidget* w, int x) {
        QSize s = w->sizeHint();
        const QRect r(x, (kHeight - s.height()) / 2, s.width(), s.height());
        if (w->geometry() != r)
            w->setGeometry(r);
        visible.insert(w);
    };
    for (auto it = shown.rbegin(); it != shown.rend(); ++it)
    {
        const Group* g = *it;
        if (!g->right || !isKept(g))
            continue;
        if (std::exchange(anyRight, true))
        {
            right -= kGroupGap;
            place(g->line, right);
        }
        right -= g->row->sizeHint().width();
        place(g->row, right);
    }
    // The main groups sit centred in the bar, pushed left only as far as the right-hand groups need.
    int total = 0;
    for (const Group* g : shown)
        if (!g->right && isKept(g))
            total += g->row->sizeHint().width() + (total ? kGroupGap : 0);
    left = std::max(left, std::min((width() - total) / 2, right - (anyRight ? kGroupGap : 0) - total));
    for (const Group* g : shown)
    {
        if (g->right || !isKept(g))
            continue;
        if (std::exchange(anyLeft, true))
        {
            place(g->line, left);
            left += kGroupGap;
        }
        place(g->row, left);
        left += g->row->sizeHint().width();
    }
    auto apply = [&](QWidget* w) {
        const bool on = visible.contains(w);
        if (w->isVisibleTo(this) != on)
            w->setVisible(on);
    };
    for (const Group& g : mGroups)
    {
        apply(g.line);
        apply(g.row);
    }
    if (mChevron->isVisibleTo(this) != overflowed)
        mChevron->setVisible(overflowed);
    if (overflowed)
    {
        QSize s = mChevron->sizeHint();
        mChevron->setGeometry(width() - 2 - s.width(), (kHeight - s.height()) / 2, s.width(), s.height());
    }
}

void ViewportBottomBar::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    layoutItems();
}

bool ViewportBottomBar::event(QEvent* e)
{
    if (e->type() == QEvent::Enter)
        refreshTooltips();
    else if (e->type() == QEvent::Show || e->type() == QEvent::LayoutRequest)
        layoutItems();
    return QWidget::event(e);
}

ViewportPanel::ViewportPanel(QWidget* parent) : QWidget(parent)
{
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    mView = new ViewportWidget(this);
    mToolbar = new ViewportToolbar(mView, this);
    mBottomBar = new ViewportBottomBar(mView, this);
    v->addWidget(mToolbar);
    v->addWidget(mView, 1);
    v->addWidget(mBottomBar);
    imagePlanes::acceptDrops(mView);
    auto showBars = [this] {
        mToolbar->setVisible(generalPref(vd::kToolbar, true));
        mBottomBar->setVisible(generalPref(vd::kBottomBar, true));
    };
    showBars();
    mPrefListener = PrefStore::get().onChanged([showBars](const std::string& file) {
        if (file == "general")
            showBars();
    });
}

ViewportPanel::~ViewportPanel()
{
    PrefStore::get().removeOnChanged(mPrefListener);
}
} // namespace lila::ui
