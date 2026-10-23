"""Lazy, cancellable component-field sampling of calculation snapshots."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QLayout, QLineEdit,
    QMenu, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from temsim.gui.field_vector_canvas import FieldVectorCanvas
from temsim.gui.compact_status_label import CompactStatusLabel
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox
from temsim.gui.projection_angle_controls import ProjectionAngleControls
from temsim.gui.transverse_projection import format_projection_angle
from temsim.field_density_texture import section_grid_axes


@dataclass(frozen=True)
class _Samples:
    points_m: np.ndarray
    b_t: np.ndarray
    e_v_per_m: np.ndarray
    force_n: np.ndarray
    magnetic_valid: np.ndarray
    electric_valid: np.ndarray
    force_valid: np.ndarray
    notes: tuple[str, ...]
    grid_axes_m: tuple[np.ndarray, np.ndarray] | None = None
    section_angle_deg: float = 0.


class _Signals(QObject):
    finished = Signal(int, object, object)


class _PrepareWorker(QRunnable):
    def __init__(self, generation, state, magnetic_scene, z_limits_mm):
        super().__init__()
        self.generation = generation
        self.state, self.magnetic_scene, self.z_limits_mm = state, magnetic_scene, z_limits_mm
        self.cancelled = Event()
        self.signals = _Signals()

    def run(self):
        result, error = None, None
        try:
            if not self.cancelled.is_set():
                from temsim.field_vector_scene import prepare_field_vector_scene
                from temsim.cpu_resources import numerical_job
                with numerical_job(1, cancelled=self.cancelled.is_set):
                    result = prepare_field_vector_scene(self.state, magnetic_scene=self.magnetic_scene,
                                                        z_limits_mm=self.z_limits_mm)
                    from temsim.field_display_regions import build_field_display_regions
                    result = (result, build_field_display_regions(self.state, result.sources))
        except Exception as exc:
            error = str(exc)
        if self.cancelled.is_set():
            result, error = None, None
        self.signals.finished.emit(self.generation, result, error)


class _SampleWorker(QRunnable):
    def __init__(self, generation, scene, points, source_keys, energy, direction, bounds_m, section,
                 grid_axes_m=None, section_angle_deg=0.):
        super().__init__()
        self.generation, self.scene, self.points = generation, scene, points
        self.source_keys, self.energy, self.direction = source_keys, energy, direction
        self.bounds_m, self.section = bounds_m, section
        self.grid_axes_m = grid_axes_m
        self.section_angle_deg = float(section_angle_deg)
        self.cancelled = Event()
        self.signals = _Signals()

    def run(self):
        result, error = None, None
        try:
            from temsim.cpu_resources import numerical_job
            chunks = []
            with numerical_job(1, cancelled=self.cancelled.is_set):
                for start in range(0, len(self.points), 512):
                    if self.cancelled.is_set():
                        break
                    chunks.append(self.scene.sample(self.points[start:start+512], self.source_keys,
                                                    kinetic_energy_ev=self.energy, direction=self.direction))
            if chunks and not self.cancelled.is_set():
                names = ("points_m", "b_t", "e_v_per_m", "force_n", "magnetic_valid", "electric_valid", "force_valid")
                values = {name: np.concatenate([getattr(chunk, name) for chunk in chunks]) for name in names}
                notes = tuple(dict.fromkeys(note for chunk in chunks for note in chunk.notes))
                result = (_Samples(**values, notes=notes, grid_axes_m=self.grid_axes_m,
                                   section_angle_deg=self.section_angle_deg),
                          self.bounds_m, self.section)
        except Exception as exc:
            error = str(exc)
        if self.cancelled.is_set():
            result, error = None, None
        self.signals.finished.emit(self.generation, result, error)


class _GeometryWorker(QRunnable):
    """Build captured material meshes independently of field sampling."""

    def __init__(self, generation, state, sources):
        super().__init__()
        self.generation, self.state, self.sources = generation, state, sources
        self.cancelled = Event()
        self.signals = _Signals()

    def run(self):
        result, error = None, None
        try:
            if not self.cancelled.is_set():
                from temsim.field_vector_geometry import build_field_vector_geometry
                from temsim.cpu_resources import numerical_job
                with numerical_job(1, cancelled=self.cancelled.is_set):
                    result = build_field_vector_geometry(self.state, self.sources)
        except Exception as exc:
            error = str(exc)
        if self.cancelled.is_set():
            result, error = None, None
        self.signals.finished.emit(self.generation, result, error)


def _spin(minimum, maximum, value, suffix, decimals=5):
    control = QDoubleSpinBox()
    control.setDecimals(decimals)
    control.setRange(minimum, maximum)
    control.setValue(value)
    control.setSuffix(suffix)
    control.setKeyboardTracking(False)
    control.setMinimumWidth(90)
    return control


class FieldVectorPage(QWidget):
    """Select physical field sources and inspect B, E or q(E + v cross B).

    Only update_snapshot supplies a new optical state. Source capture and bounded
    sampling run in a worker, with cooperative cancellation between sample chunks
    and generation guards for providers which cannot be interrupted. Input edits
    clear the old display immediately and block requests until a fresh snapshot.
    """

    settings_changed = Signal()
    component_selected = Signal(str)
    navigation_requested = Signal(str, str, float)
    projection_angle_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._prepared_scene = None
        self._result_reference = None
        self._z_limits_mm = None
        self._scene = None
        self._display_regions = None
        self._selected_component = None
        self._selection_needs_fit = False
        self._sample = None
        self._projection_angle_deg = 0.
        self._active = False
        self._fields_stale = False
        self._generation = 0
        self._worker = None
        self._jobs = {}
        self._geometry = None
        self._geometry_error = None
        self._geometry_generation = 0
        self._geometry_worker = None
        self._geometry_jobs = {}
        # The callback owns only the job dictionary, never the QWidget. Parent
        # destruction may bypass closeEvent; every outstanding job is cancelled.
        jobs = self._jobs
        self.destroyed.connect(lambda *_args: [worker.cancelled.set() for worker in tuple(jobs.values())])
        geometry_jobs = self._geometry_jobs
        self.destroyed.connect(lambda *_args: [worker.cancelled.set() for worker in tuple(geometry_jobs.values())])
        self._remembered_keys = None
        self._all_sources_intent = True
        self._fit_initial = True
        self._energy_initialized = False
        self._request_timer = QTimer(self)
        self._request_timer.setSingleShot(True)
        self._request_timer.setInterval(100)
        self._request_timer.timeout.connect(self._request_sample)
        self._base_sample_data = None
        self._detail_cache = OrderedDict()
        self._detail_key = None
        self._view_revision = 0
        self._detail_serial = 0
        self._detail_worker = None
        self._detail_jobs = {}
        detail_jobs = self._detail_jobs
        self.destroyed.connect(lambda *_args: [job.cancelled.set() for job in tuple(detail_jobs.values())])
        self._detail_dirty = False
        self._detail_timer = QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(250)
        self._detail_timer.timeout.connect(self._request_view_detail)

        self.sources = QListWidget()
        self.sources.setObjectName("fieldVectorSources")
        self.sources.setMinimumHeight(80)
        self.sources.setMaximumHeight(145)
        self.sources.setMinimumWidth(150)
        self.sources.itemChanged.connect(self._sources_changed)
        self.sources.currentItemChanged.connect(self._update_details)
        self.sources.currentItemChanged.connect(self._source_focused)
        self.sources.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sources.customContextMenuRequested.connect(self._source_context_menu)
        self.field_scope = QComboBox()
        self.field_scope.setObjectName("fieldVectorScope")
        self.field_scope.addItem("All fields superposed", "all")
        self.field_scope.addItem("Selected component only", "component")
        self.field_scope.addItem("Checked sources", "checked")
        self.field_scope.setToolTip(
            "Choose contributions at the same physical observation region. All fields includes fringe "
            "contributions from other components; a shared circuit remains one inseparable source.")
        self.selected_label = QLabel("Observation: all physical components", self)
        self.selected_label.hide()
        self.component_picker = QComboBox()
        self.component_picker.setObjectName("fieldVectorComponent")
        self.component_picker.addItem("All physical components", None)
        self.component_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.component_picker.setMinimumContentsLength(18)
        self.component_picker.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.component_picker.setToolTip("Choose the physical observation region. Right-click to navigate to this component.")
        self.component_picker.currentIndexChanged.connect(self._component_picked)
        self.component_picker.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.component_picker.customContextMenuRequested.connect(
            lambda position: self._component_context_menu(
                self._selected_component, self.component_picker.mapToGlobal(position)))
        self.component_picker.setEnabled(False)
        self.clear_selection = QPushButton("Whole column")
        self.clear_selection.setToolTip("Observe the union of physical component envelopes.")
        self.clear_selection.clicked.connect(lambda: self.focus_component(None))
        self.select_all = QPushButton("All")
        self.select_none = QPushButton("None")
        self.fit_selected = QPushButton("Fit selected")
        self.select_all.clicked.connect(lambda: self._check_all(True))
        self.select_none.clicked.connect(lambda: self._check_all(False))
        self.fit_selected.clicked.connect(self._fit_sources)
        self.section = QComboBox()
        for label, value in (("Longitudinal section (Z–U)", "UZ"), ("3D volume", "3D")):
            self.section.addItem(label, value)
        self.section.setToolTip(
            "Z always runs horizontally. The longitudinal section rotates through the column axis "
            "with the shared Ray Diagram / Physical Layout angle; 0° is X–Z and 90° is Y–Z.")
        self.projection_controls = ProjectionAngleControls()
        self.projection_controls.angle_changed.connect(self._angle_requested)
        self.quantity = QComboBox()
        for label, value in (("B + E overlay", "overlay"), ("Magnetic B", "B"), ("Electric E", "E"), ("Electron force", "force")):
            self.quantity.addItem(label, value)
        self.quantity.setObjectName("fieldVectorQuantity")
        self.quantity.setToolTip(
            "Strength hatching uses one quantity at a time: B and E have different units "
            "and cannot share one strength scale.")
        self.render_style = QComboBox()
        self.render_style.setObjectName("fieldVectorRenderStyle")
        self.render_style.addItem("Arrows", "arrows")
        self.render_style.addItem("Strength hatching (2D)", "density")
        self.render_style.setToolTip(
            "Arrows show direction. Hatching density shows the full vector magnitude in a 2D section, "
            "including the component normal to the plane. Rendering reuses cached field samples.")
        self.density = QComboBox()
        for label, count in (("9 samples / axis", 9), ("15 samples / axis", 15), ("21 samples / axis", 21)):
            self.density.addItem(label, count)
        self.density.setCurrentIndex(1)
        self.density.setToolTip("3D volumes are capped at 9 samples per axis (729 points). This controls display sampling only.")
        self.centre_x = _spin(-1e5, 1e5, 0., " mm")
        self.centre_y = _spin(-1e5, 1e5, 0., " mm")
        for control in (self.centre_x, self.centre_y):
            control.setToolTip(
                "In the longitudinal section, Centre X cos(angle) + Centre Y sin(angle) "
                "sets the U centre of the observation window. The section still passes through "
                "the column axis at V = 0. In 3D, these are the X and Y sampling centres.")
        self.centre_z = _spin(-1e5, 1e5, 0., " mm")
        self.radius = _spin(.000001, 1e5, 1., " mm", 6)
        self.axial_half_span = _spin(.000001, 1e5, 5., " mm", 6)
        self.energy = _spin(.001, 1e9, 300000., " eV", 3)
        self.energy.setToolTip("Uniform local test-electron kinetic energy; this is not a propagated trajectory or local gun energy.")
        self.direction_x = _spin(-1., 1., 0., "", 5)
        self.direction_y = _spin(-1., 1., 0., "", 5)
        self.direction_z = _spin(-1., 1., 1., "", 5)
        self.direction_x.setToolTip("Velocity direction in global right-handed XYZ; normalized before force evaluation.")
        self.direction_y.setToolTip(self.direction_x.toolTip())
        self.direction_z.setToolTip(self.direction_x.toolTip())
        self.fit_view = QPushButton("Reset view")
        self.fit_view.clicked.connect(lambda: self.canvas.reset_view())
        self.canvas = FieldVectorCanvas()
        self.strength_scale = QComboBox()
        self.strength_scale.setObjectName("fieldVectorStrengthScale")
        self.strength_scale.addItem("Auto scale", "auto")
        self.strength_scale.addItem("Fixed scale", "fixed")
        self.strength_scale.setToolTip(
            "Auto: fit each quantity's current sampled peak for readability. "
            "Fixed: retain B/E/F references across zoom and calculations to compare strength changes. "
            "Actual field values are unchanged; hover reads SI values in either mode.")
        self.canvas.set_automatic_reference(True)
        self.strength_scale.currentIndexChanged.connect(self._strength_scale_changed)
        # Axes/footer occupy 111 px. A 90 px canvas leaves no field area when
        # the scrollable controls are opened on a laptop-sized window.
        self.canvas.setMinimumHeight(220)
        self.canvas.setObjectName("fieldVectorCanvas")
        self.canvas.projection_angle_requested.connect(self._angle_requested)
        self.canvas.component_selected.connect(self._canvas_component_selected)
        self.canvas.component_context_requested.connect(self._component_context_menu)
        self.canvas.view_changed.connect(self._view_changed)
        self.auto_detail = QCheckBox("Auto section detail")
        self.auto_detail.setObjectName("fieldVectorAutoDetail")
        self.auto_detail.setChecked(True)
        self.auto_detail.setToolTip(
            "After navigation stops, sample only the visible part of the current 2D section. "
            "At most 1600 points; the captured field model, physical limits and strength references stay fixed. "
            "3D volumes reuse their existing samples.")
        self.auto_detail.toggled.connect(self._auto_detail_changed)
        # Match Ray Diagram readouts: status text must not resize the canvas
        # and trigger another refinement just because a message wraps.
        self.detail_status = CompactStatusLabel("Zoom a section to refine visible samples.")
        self.show_models = QCheckBox("Physical models")
        self.show_models.setObjectName("fieldVectorShowModels")
        self.show_models.setToolTip("Overlay the selected sources' captured coil, pole and support geometry. Geometry is prepared on first use.")
        self.model_opacity = QSlider(Qt.Orientation.Horizontal)
        self.model_opacity.setObjectName("fieldVectorModelOpacity")
        self.model_opacity.setRange(0, 100)
        self.model_opacity.setValue(35)
        self.model_opacity.setMaximumWidth(180)
        self.model_opacity.setToolTip("Material opacity: 0% transparent, 100% opaque. Field vectors remain in front for inspection.")
        self.model_opacity_label = QLabel("35%")
        self.fit_models = QPushButton("Fit models")
        self.fit_models.setEnabled(False)
        self.fit_models.setToolTip("Fit selected physical models and current field samples. Sampling positions and field values stay fixed.")
        self.model_status = QLabel("Physical models hidden.")
        self.model_status.setWordWrap(True)
        self.model_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.model_status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.model_status.hide()
        self.canvas.set_model_opacity(.35)
        self.show_models.toggled.connect(self._model_visibility_changed)
        self.model_opacity.valueChanged.connect(self._model_opacity_changed)
        self.fit_models.clicked.connect(self._fit_models)
        self.status = QLabel("Calculate the column, then select Field vectors.")
        self.legend = QLabel("B [T] | E [V/m] | electron force [N]")
        self.convention = QLabel("Z → right | U → up | View angle follows Ray Diagram | Physical envelopes only")
        self.convention.setToolTip(
            "Global right-handed XYZ; +Z downstream. Independent B/E/F strength references. "
            "Samples stay in the chosen physical plane and inside component envelopes, including the beam bore. "
            "Hollow squares mean unavailable, not zero.")
        for label in (self.status, self.legend, self.convention):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.details_toggle = QPushButton("Model details and valid ranges")
        self.details_toggle.setCheckable(True)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(150)
        self.details.hide()
        self.details_toggle.toggled.connect(self.details.setVisible)

        source_box = QVBoxLayout()
        source_box.addWidget(QLabel("Physical sources (check to sum)"))
        source_box.addWidget(self.sources)
        source_buttons = QHBoxLayout()
        for button in (self.select_all, self.select_none, self.fit_selected):
            source_buttons.addWidget(button)
        source_box.addLayout(source_buttons)
        controls = QGridLayout()
        entries = (("View", self.section), ("Sampling", self.density),
                   ("Centre X", self.centre_x), ("Centre Y", self.centre_y), ("Centre Z", self.centre_z),
                   ("U / XY half width", self.radius), ("Z half span", self.axial_half_span))
        for index, (label, widget) in enumerate(entries):
            row, col = divmod(index, 2)
            controls.addWidget(QLabel(label), row, col*2)
            controls.addWidget(widget, row, col*2+1)
        top = QHBoxLayout()
        top.addLayout(source_box, 2)
        top.addLayout(controls, 3)
        force_controls = QHBoxLayout()
        force_controls.addWidget(QLabel("Test electron"))
        force_controls.addWidget(self.energy)
        for label, control in (("vX", self.direction_x), ("vY", self.direction_y), ("vZ", self.direction_z)):
            force_controls.addWidget(QLabel(label))
            force_controls.addWidget(control)
        force_controls.addWidget(self.fit_view)
        reference_controls = QHBoxLayout()
        self.reference_inputs = {}
        for key, unit in (("B", "T"), ("E", "V/m"), ("F", "N")):
            control = QLineEdit()
            control.setPlaceholderText("First nonzero peak")
            control.setToolTip(
                "Fixed strength reference in SI for arrows and hatching, retained across snapshots "
                "and zoom levels. Enter a positive number, e.g. 1e-6.")
            control.setMinimumWidth(65)
            control.setMaximumWidth(140)
            control.editingFinished.connect(lambda name=key: self._reference_changed(name))
            self.reference_inputs[key] = control
            reference_controls.addWidget(QLabel(f"{key} ref [{unit}]"))
            reference_controls.addWidget(control)
        self.match_references = QPushButton("Use current peaks")
        self.match_references.clicked.connect(self._match_references)
        reference_controls.addWidget(self.match_references)
        layout = QVBoxLayout(self)
        layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(4)
        angle_row = QHBoxLayout()
        angle_row.setContentsMargins(0, 0, 0, 0)
        angle_row.addWidget(self.projection_controls)
        angle_row.addStretch()
        layout.addLayout(angle_row)
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Field contributions"))
        scope_row.addWidget(self.field_scope)
        scope_row.addWidget(QLabel("Observe"))
        scope_row.addWidget(self.component_picker, 1)
        scope_row.addWidget(self.clear_selection)
        layout.addLayout(scope_row)
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.addLayout(top)
        control_layout.addLayout(force_controls)
        control_layout.addLayout(reference_controls)
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setWidget(control_widget)
        self.control_scroll.setMinimumSize(0, 50)
        self.control_scroll.setMaximumHeight(220)
        self.control_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.controls_toggle = QPushButton("Sources and section controls")
        self.controls_toggle.setCheckable(True)
        self.controls_toggle.setChecked(False)
        self.controls_toggle.toggled.connect(self.control_scroll.setVisible)
        self.control_scroll.hide()
        layout.addWidget(self.controls_toggle)
        layout.addWidget(self.control_scroll)
        model_controls = QHBoxLayout()
        model_controls.addWidget(self.show_models)
        model_controls.addWidget(QLabel("Model opacity"))
        model_controls.addWidget(self.model_opacity)
        model_controls.addWidget(self.model_opacity_label)
        model_controls.addWidget(self.fit_models)
        model_controls.addStretch()
        layout.addLayout(model_controls)
        detail_row = QHBoxLayout()
        detail_row.addWidget(QLabel("Display"))
        detail_row.addWidget(self.render_style)
        detail_row.addWidget(self.quantity)
        detail_row.addWidget(self.strength_scale)
        detail_row.addWidget(self.auto_detail)
        detail_row.addWidget(self.detail_status, 1)
        layout.addLayout(detail_row)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.model_status)
        layout.addWidget(self.legend)
        layout.addWidget(self.convention)
        layout.addWidget(self.status)
        layout.addWidget(self.details_toggle)
        layout.addWidget(self.details)

        for control in (self.centre_x, self.centre_y, self.centre_z, self.radius, self.axial_half_span,
                        self.energy, self.direction_x, self.direction_y, self.direction_z):
            control.valueChanged.connect(self._controls_changed)
        self.section.currentIndexChanged.connect(self._section_changed)
        self.density.currentIndexChanged.connect(self._controls_changed)
        self.quantity.currentIndexChanged.connect(self._quantity_changed)
        self.render_style.currentIndexChanged.connect(self._render_style_changed)
        self.field_scope.currentIndexChanged.connect(self._scope_changed)

    def minimumSizeHint(self):
        # An optional hidden QStackedWidget page must not resize other plots.
        return QSize(0, 0)

    @property
    def selected_source_keys(self):
        if self._scene is not None and self.field_scope.currentData() == "all":
            return tuple(source.key for source in self._scene.sources)
        if self.field_scope.currentData() == "component":
            if self._display_regions is None or self._selected_component is None:
                return ()
            return self._display_regions.source_keys_for_component(self._selected_component)
        return tuple(self.sources.item(index).data(Qt.ItemDataRole.UserRole)
                     for index in range(self.sources.count())
                     if self.sources.item(index).checkState() == Qt.CheckState.Checked)

    def set_projection_angle(self, angle):
        """Follow the shared angle and refresh only the rotated section samples."""
        angle = float(angle)
        if not np.isfinite(angle):
            return
        # Preserve the slider's 360-degree endpoint consistently on all pages;
        # the physical plane still uses a periodic angle.
        self.projection_controls.set_angle(angle if 0. <= angle <= 360. else angle % 360.)
        angle %= 360.
        if abs((angle - self._projection_angle_deg + 180.) % 360. - 180.) < 1e-10:
            return
        self._projection_angle_deg = angle
        rotates_plane = self.section.currentData() == "UZ"
        preparing = (self._scene is None and isinstance(self._worker, _PrepareWorker)
                     and self._worker.generation == self._generation
                     and not self._worker.cancelled.is_set())
        if rotates_plane:
            # Invalidate work for the previous plane, but retain its accepted
            # display at its actual angle while the latest request is sampled.
            # Provider capture is independent of the section angle. Preserve
            # an in-flight capture; its first sample will use the latest angle.
            if not preparing:
                self._cancel_jobs()
            self._sample = None
            if self._fields_stale:
                message = "Inputs changed. Recalculate to obtain current fields."
            elif self._state is None:
                message = "Calculate the column, then select Field vectors."
            elif not self._active:
                message = "Rotated section awaits samples."
            elif preparing:
                message = "Preparing field providers; the latest longitudinal angle will be sampled."
            else:
                message = "Rotating section; sampling the new longitudinal plane…"
            if self.canvas.sample is not None and not self._fields_stale:
                self.canvas.set_pending_section_angle(angle)
                self.status.setText(
                    f"Updating section to {format_projection_angle(angle)}°; showing the accepted "
                    f"{format_projection_angle(self.canvas.projection_angle_deg)}° section until sampling completes.")
                self.fit_models.setEnabled(False)
            else:
                self.canvas.clear(message)
                self.canvas.set_projection_angle(angle)
                self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        else:
            self.canvas.set_projection_angle(angle)
        if rotates_plane and not preparing:
            self._schedule(250)

    def _angle_requested(self, angle):
        self.set_projection_angle(angle)
        self.projection_angle_requested.emit(self.projection_controls.angle_deg)

    def _populate_component_picker(self):
        from temsim.component_representation import non_material_role
        assembly = getattr(self._state, "_resolved_assembly", None)
        parts = sorted(getattr(assembly, "parts", ()), key=lambda part: (part.center_z_mm, part.key))
        with QSignalBlocker(self.component_picker):
            self.component_picker.clear()
            self.component_picker.addItem("All physical components", None)
            for part in parts:
                if not non_material_role(part.data):
                    self.component_picker.addItem(f"{part.name} · Z {part.center_z_mm:.3f} mm", part.key)
        self._sync_component_picker()
        self.component_picker.setEnabled(self._state is not None and not self._fields_stale)

    def _sync_component_picker(self):
        with QSignalBlocker(self.component_picker):
            index = self.component_picker.findData(self._selected_component)
            if index < 0 and self._selected_component is not None:
                self.component_picker.addItem(f"{self._selected_component} [unavailable]", self._selected_component)
                index = self.component_picker.count() - 1
            self.component_picker.setCurrentIndex(max(0, index))
        self.component_picker.setToolTip(
            self.selected_label.text() + "\nRight-click to navigate to this component.")

    def _component_picked(self, _index):
        key = self.component_picker.currentData()
        if key is None:
            self.focus_component(None)
        else:
            self._canvas_component_selected(key)

    def _scope_changed(self, *_args):
        self._update_details()
        self._controls_changed()
        self._refresh_geometry()

    def _source_component_key(self, item):
        if item is None or self._scene is None:
            return None
        key = item.data(Qt.ItemDataRole.UserRole)
        source = next((value for value in self._scene.sources if value.key == key), None)
        if source is None:
            return None
        # The source label is not necessarily a physical part (joint circuits).
        assembly = getattr(self._state, "_resolved_assembly", None)
        physical = {part.key: part for part in getattr(assembly, "parts", ())}
        from temsim.component_representation import non_material_role, SHARED_DEFLECTOR_HOSTS
        for component in source.component_keys:
            part = physical.get(component)
            if part is not None and not non_material_role(part.data):
                return component
            host = SHARED_DEFLECTOR_HOSTS.get(component)
            if part is not None and not host and part.data.get("physical_host_status") == "modelled":
                host = part.data.get("physical_host_key")
            if host in physical and not non_material_role(physical[host].data):
                return host
        return None

    def _source_focused(self, current, _previous=None):
        key = self._source_component_key(current)
        if key is not None:
            self._canvas_component_selected(key)

    def _canvas_component_selected(self, key):
        self.focus_component(key)
        self.component_selected.emit(str(key))

    def focus_component(self, key, z_mm=None):
        key = str(key) if key else None
        changed = key != self._selected_component
        self._selected_component = key
        assembly = getattr(self._state, "_resolved_assembly", None)
        part = next((p for p in getattr(assembly, "parts", ()) if p.key == key), None)
        self.selected_label.setText(
            f"Observation: {part.name if part is not None else key}" if key else
            "Observation: all physical components")
        self._sync_component_picker()
        self.canvas.set_selected_component(key)
        if key is not None:
            self.show_models.setChecked(True)
        self._refresh_geometry()
        if not changed:
            return
        self._selection_needs_fit = True
        if self._scene is not None and self._display_regions is not None:
            self._fit_sources()

    def _source_context_menu(self, position):
        item = self.sources.itemAt(position)
        key = self._source_component_key(item)
        if key:
            self._component_context_menu(key, self.sources.viewport().mapToGlobal(position))

    def component_navigation_menu(self, key):
        """Build a testable menu using captured physical identity and position."""
        assembly = getattr(self._state, "_resolved_assembly", None)
        part = next((p for p in getattr(assembly, "parts", ()) if p.key == key), None)
        if part is None or self._fields_stale:
            return None
        menu = QMenu(self)
        for label, destination in (("Go to Ray Diagram", "ray"),
                                   ("Go to Physical Layout", "physical"),
                                   ("Open component / 3D part", "parts"),
                                   ("Go to Vacuum map", "vacuum")):
            action = menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, target=destination, k=part.key, z=part.center_z_mm:
                self.navigation_requested.emit(k, target, float(z)))
        return menu

    def _component_context_menu(self, key, global_position):
        menu = self.component_navigation_menu(key)
        if menu is not None:
            # Right-click navigation selects the real body without launching a ray calculation.
            if key != self._selected_component:
                self._canvas_component_selected(key)
            menu.aboutToHide.connect(menu.deleteLater)
            menu.popup(global_position)

    def update_snapshot(self, state, records=(), z_limits_mm=None, *, prepared_scene=None, result_reference=None):
        self._remember_selection()
        self._cancel_jobs()
        self._invalidate_geometry()
        self._state, self._prepared_scene = state, prepared_scene
        self._result_reference, self._z_limits_mm = result_reference, z_limits_mm
        self._scene, self._sample, self._fields_stale = None, None, False
        self._display_regions = None
        self._selection_needs_fit = self._selected_component is not None
        self._populate_component_picker()
        with QSignalBlocker(self.sources):
            self.sources.clear()
        self.details.clear()
        self.canvas.clear("Preparing the latest field snapshot…")
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        self.status.setText("Current calculation captured; field sampling starts when this view is selected.")
        if not self._energy_initialized:
            voltage = getattr(state, "beam_voltage_kv", None)
            if voltage is not None and np.isfinite(voltage) and voltage > 0:
                with QSignalBlocker(self.energy):
                    self.energy.setValue(float(voltage) * 1000.)
                self._energy_initialized = True
        self._schedule()

    def mark_inputs_stale(self):
        self._remember_selection()
        self._cancel_jobs()
        self._invalidate_geometry()
        self._fields_stale = True
        self.component_picker.setEnabled(False)
        self._sample = None
        self.canvas.clear("Inputs changed. Recalculate to obtain current fields.")
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        self.status.setText("Stale: inputs changed; waiting for the next completed calculation. Old vectors have been cleared.")

    def invalidate(self, message="Field snapshot pending calculation."):
        self._remember_selection()
        self._cancel_jobs()
        self._invalidate_geometry()
        self._state = self._prepared_scene = self._scene = self._sample = self._result_reference = None
        self._display_regions = None
        self._fields_stale = True
        self._populate_component_picker()
        self.canvas.clear(message)
        self.status.setText(message)
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        with QSignalBlocker(self.sources):
            self.sources.clear()
        self.details.clear()

    def set_active(self, active):
        self._active = bool(active)
        if not self._active:
            self._cancel_jobs(keep_detail_cache=True)
            self._cancel_geometry_jobs()
        else:
            if self._sample is None:
                self._schedule()
            self._ensure_geometry()
            if self._detail_dirty:
                self._detail_timer.start()

    def _cancel_jobs(self, *, keep_detail_cache=False):
        if (keep_detail_cache and self.auto_detail.isChecked() and self._base_sample_data is not None
                and (self._detail_worker is not None or self._detail_timer.isActive())):
            self._detail_dirty = True
        self._generation += 1
        self._request_timer.stop()
        for worker in self._jobs.values():
            worker.cancelled.set()
        self._worker = None
        self._detail_timer.stop()
        self._view_revision += 1
        for job in self._detail_jobs.values():
            job.cancelled.set()
        # Retain the in-flight pointer until its completion: viewport sampling
        # has one worker, not one newly queued job per wheel event.
        if not keep_detail_cache:
            self._base_sample_data = None
            self._detail_cache.clear()
            self._detail_key = None
            self._detail_dirty = False
            self.detail_status.setText("Visible detail awaits current section samples.")

    def _cancel_geometry_jobs(self):
        self._geometry_generation += 1
        for worker in self._geometry_jobs.values():
            worker.cancelled.set()
        self._geometry_worker = None

    def _invalidate_geometry(self):
        self._cancel_geometry_jobs()
        self._geometry = self._geometry_error = None
        self.canvas.set_physical_meshes(())
        self.fit_models.setEnabled(False)
        self.model_status.setText("Physical models await a current captured configuration.")

    def _model_visibility_changed(self, checked):
        self.canvas.set_model_visible(checked)
        self.model_status.setVisible(checked)
        if checked:
            self._geometry_error = None
            self._ensure_geometry()
        self._refresh_geometry()
        self.settings_changed.emit()

    def _model_opacity_changed(self, value):
        self.model_opacity_label.setText(f"{value}%")
        self.canvas.set_model_opacity(value / 100.)
        self.settings_changed.emit()

    def _fit_models(self):
        self.canvas.fit_physical_models()
        self.settings_changed.emit()

    def _ensure_geometry(self):
        if (not self.show_models.isChecked() or not self._active or self._fields_stale
                or self._state is None or self._scene is None):
            return
        if self._geometry is not None:
            self._refresh_geometry()
            return
        if self._geometry_worker is not None or self._geometry_error is not None:
            return
        self._geometry_generation += 1
        token = self._geometry_generation
        worker = _GeometryWorker(token, self._state, self._scene.sources)
        worker.signals.finished.connect(self._geometry_ready)
        self._geometry_jobs[token] = self._geometry_worker = worker
        self.model_status.setText("Preparing physical models from the captured configuration…")
        QThreadPool.globalInstance().start(worker)

    @Slot(int, object, object)
    def _geometry_ready(self, generation, geometry, error):
        self._geometry_jobs.pop(generation, None)
        if generation != self._geometry_generation or self._fields_stale or not self._active:
            return
        self._geometry_worker = None
        self._geometry, self._geometry_error = geometry, error
        if geometry is None and error is None:
            self._geometry_error = "Geometry capture cancelled."
        self._refresh_geometry()
        self._update_details()

    def _refresh_geometry(self):
        meshes = ()
        if self._geometry is not None:
            if self._selected_component is not None and hasattr(self._geometry, "select_component"):
                meshes = self._geometry.select_component(self._selected_component)
            else:
                meshes = self._geometry.select(self.selected_source_keys)
        self.canvas.set_physical_meshes(meshes)
        self.canvas.set_selected_components(tuple(mesh.key for mesh in meshes)
                                            if self._selected_component is not None else ())
        self.fit_models.setEnabled(self.show_models.isChecked() and bool(meshes) and self._sample is not None)
        if self._geometry_error:
            self.model_status.setText(f"Physical models unavailable: {self._geometry_error}")
        elif self._geometry is not None:
            missing = sum(not self._geometry.source_mesh_keys.get(key) for key in self.selected_source_keys)
            parts = len({mesh.key for mesh in meshes})
            text = f"Captured physical models: {parts} parts. Copper coils, magnetic poles and supports retain their material colours."
            if missing:
                text += f" {missing} selected sources have no material model; see model details."
            self.model_status.setText(text)

    def _remember_selection(self):
        if self.sources.count():
            self._remembered_keys = {
                self.sources.item(index).data(Qt.ItemDataRole.UserRole)
                for index in range(self.sources.count())
                if self.sources.item(index).checkState() == Qt.CheckState.Checked
            }

    def _schedule(self, delay_ms=100):
        if self._active and self._state is not None and not self._fields_stale:
            self._request_timer.start(delay_ms)

    def _controls_changed(self, *_args):
        self._cancel_jobs()
        self._sample = None
        self.canvas.clear("Sampling controls changed…" if not self._fields_stale else "Inputs changed. Recalculate to obtain current fields.")
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        self.settings_changed.emit()
        self._schedule()

    def _sources_changed(self, *_args):
        with QSignalBlocker(self.field_scope):
            self.field_scope.setCurrentIndex(self.field_scope.findData("checked"))
        self._remember_selection()
        if self.sources.count():
            self._all_sources_intent = len(self.selected_source_keys) == self.sources.count()
        self._update_details()
        self._controls_changed()
        self._refresh_geometry()

    def _sync_render_controls(self):
        """Keep density on one SI quantity and a physical 2D sampling plane."""
        is_2d = self.section.currentData() != "3D"
        density_index = self.render_style.findData("density")
        density_item = self.render_style.model().item(density_index)
        density_item.setEnabled(is_2d)
        density_item.setToolTip(
            "Crosshatch density represents full vector magnitude in the longitudinal section."
            if is_2d else "Strength hatching is available only for the longitudinal section; 3D uses arrows.")
        if not is_2d and self.render_style.currentData() == "density":
            with QSignalBlocker(self.render_style):
                self.render_style.setCurrentIndex(self.render_style.findData("arrows"))
        self.render_style.setToolTip(
            "Arrows show direction. Hatching density shows full vector magnitude using cached samples."
            if is_2d else "3D volume uses arrows. Select the longitudinal section to enable strength hatching.")
        is_density = self.render_style.currentData() == "density"
        overlay_item = self.quantity.model().item(self.quantity.findData("overlay"))
        overlay_item.setEnabled(not is_density)
        overlay_item.setToolTip(
            "B and E have different units. Choose Magnetic B, Electric E or Electron force for strength hatching.")
        if is_density and self.quantity.currentData() == "overlay":
            with QSignalBlocker(self.quantity):
                self.quantity.setCurrentIndex(self.quantity.findData("B"))

    def _update_render_style(self):
        self._sync_render_controls()
        self.canvas.set_mode(self.quantity.currentData())
        self.canvas.set_render_style(self.render_style.currentData())
        self.legend.setText(self.canvas.legend_text)

    def _section_changed(self, *_args):
        self._update_render_style()
        self._controls_changed()

    def _render_style_changed(self, *_args):
        self._update_render_style()
        self.settings_changed.emit()

    def _quantity_changed(self, *_args):
        self._update_render_style()
        self.settings_changed.emit()

    def _refresh_references(self):
        for key, reference in self.canvas.references.items():
            if reference is not None:
                self.reference_inputs[key].setText(f"{reference:.9g}")
        self.legend.setText(self.canvas.legend_text)

    def _strength_scale_changed(self, *_args):
        self.canvas.set_automatic_reference(self.strength_scale.currentData() == "auto")
        self._refresh_references()
        self.settings_changed.emit()

    def _reference_changed(self, key):
        try:
            self.canvas.set_reference(key, float(self.reference_inputs[key].text()))
        except ValueError:
            self.status.setText(f"{key} strength reference must be a finite positive SI value; previous reference retained.")
        else:
            self.strength_scale.setCurrentIndex(self.strength_scale.findData("fixed"))
        self._refresh_references()

    def _match_references(self):
        self.canvas.use_current_peaks()
        self._refresh_references()

    def _check_all(self, checked):
        self._all_sources_intent = bool(checked)
        if not checked:
            self._remembered_keys = set()
        with QSignalBlocker(self.sources):
            for index in range(self.sources.count()):
                self.sources.item(index).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._sources_changed()
        if checked:
            self.field_scope.setCurrentIndex(self.field_scope.findData("all"))

    def _fit_sources(self, *_args):
        if self._scene is None:
            return
        bounds = (None if self._display_regions is None else
                  self._display_regions.bounds_for_component(self._selected_component))
        self._selection_needs_fit = False
        if bounds is None:
            self._controls_changed()
            return
        lower, upper = np.asarray(bounds)
        if self._selected_component is not None:
            # A winding may be centimetres wide while its equivalent field is
            # valid only near the axis. Guide the default grid by that source's
            # support, without replacing the physical mask or changing the
            # observation window when the contribution mode changes.
            owner_keys = set(self._display_regions.source_keys_for_component(self._selected_component))
            intersections = []
            for source in self._scene.sources:
                if source.key in owner_keys:
                    lo = np.maximum(lower, source.bounds_m[0])
                    hi = np.minimum(upper, source.bounds_m[1])
                    if np.isfinite((lo, hi)).all() and np.all(hi > lo):
                        intersections.append((lo, hi))
            if intersections:
                lower = np.min([item[0] for item in intersections], axis=0)
                upper = np.max([item[1] for item in intersections], axis=0)
        centre = (lower + upper) * 500.
        half = np.maximum((upper - lower) * 500., 1e-6)
        for control, value in ((self.centre_x, centre[0]), (self.centre_y, centre[1]), (self.centre_z, centre[2]),
                               (self.radius, max(half[0], half[1])), (self.axial_half_span, half[2])):
            with QSignalBlocker(control):
                control.setValue(float(value))
        self.canvas.reset_view()
        self._controls_changed()

    def _sample_grid(self):
        centre = np.array([self.centre_x.value(), self.centre_y.value(), self.centre_z.value()]) * .001
        half = np.array([self.radius.value(), self.radius.value(), self.axial_half_span.value()]) * .001
        section = self.section.currentData()
        count = int(self.density.currentData())
        if section == "3D":
            count = min(count, 9)
            vectors = [np.linspace(centre[axis]-half[axis], centre[axis]+half[axis], count)
                       for axis in range(3)]
            points = np.column_stack([grid.ravel() for grid in np.meshgrid(*vectors, indexing="ij")])
        else:
            theta = np.deg2rad(self._projection_angle_deg)
            cosine, sine = np.cos(theta), np.sin(theta)
            u_centre = centre[0] * cosine + centre[1] * sine
            u = np.linspace(u_centre-half[0], u_centre+half[0], count)
            z = np.linspace(centre[2]-half[2], centre[2]+half[2], count)
            uu, zz = np.meshgrid(u, z, indexing="ij")
            points = np.column_stack((uu.ravel()*cosine, uu.ravel()*sine, zz.ravel()))
        return points, np.array([centre-half, centre+half]), section

    def _section_sampling_plan(self, points, section):
        from temsim.field_section_sampling import longitudinal_field_grid, with_source_support
        sources = () if self._scene is None else self._scene.sources
        selected = tuple(source for source in sources if source.key in self.selected_source_keys)
        if section != "UZ" or self._scene is None:
            axes = section_grid_axes(points, section, angle_deg=self._projection_angle_deg)
            return points, with_source_support(axes, selected)
        assembly = getattr(self._state, "_resolved_assembly", None)
        centres = [(part.key, float(part.center_z_mm)*.001)
                   for part in getattr(assembly, "parts", ())]
        centres.extend((lens.key, float(lens.z_mm)*.001)
                       for lens in getattr(self._state, "lenses", ()))
        return longitudinal_field_grid(
            points, self._scene.sources, angle_deg=self._projection_angle_deg,
            component_centres=centres, max_points=1600, interpolation_sources=selected)

    @Slot()
    def _request_sample(self):
        if not self._active or self._state is None or self._fields_stale:
            return
        # Cancellation is cooperative: an older provider may still be inside
        # its current chunk. Keep only one base capture/sample in flight.
        if self._worker is not None or self._jobs:
            return
        self._generation += 1
        token = self._generation
        if self._scene is None:
            worker = _PrepareWorker(token, self._state, self._prepared_scene, self._z_limits_mm)
            worker.signals.finished.connect(self._prepared)
            self.status.setText("Capturing supported field providers from the completed calculation…")
        else:
            if self.field_scope.currentData() == "component" and self._selected_component is None:
                self.status.setText("Select a physical component to show its own field.")
                self.canvas.clear(self.status.text())
                return
            direction = (self.direction_x.value(), self.direction_y.value(), self.direction_z.value())
            if np.linalg.norm(direction) == 0:
                self.status.setText("Velocity direction must be nonzero; no vectors are displayed.")
                self.canvas.clear(self.status.text())
                return
            points, bounds, section = self._sample_grid()
            angle = self._projection_angle_deg
            points, grid_axes_m = self._section_sampling_plan(points, section)
            if self._display_regions is None:
                points = points[:0]
            else:
                points = points[self._display_regions.mask(points, self._selected_component)]
            if not len(points):
                self._sample = None
                self.canvas.clear("No sample locations inside the selected physical envelope.")
                self._refresh_geometry()
                self.status.setText("No physical observation region at this slice. Select a component or use Fit selected; missing geometry is not inferred from field support.")
                return
            worker = _SampleWorker(token, self._scene, points, self.selected_source_keys,
                                   self.energy.value(), direction, bounds, section, grid_axes_m, angle)
            worker.signals.finished.connect(self._sampled)
            self.status.setText(f"Sampling {len(points)} physical locations from {len(self.selected_source_keys)} selected sources…")
        self._jobs[token] = self._worker = worker
        QThreadPool.globalInstance().start(worker)

    def _resume_latest_request(self):
        if (not self._jobs and self._worker is None and self._sample is None
                and not self._request_timer.isActive()):
            self._schedule()

    @Slot(int, object, object)
    def _prepared(self, generation, scene, error):
        self._jobs.pop(generation, None)
        if generation != self._generation or self._fields_stale or not self._active:
            self._resume_latest_request()
            return
        self._worker = None
        if error or scene is None:
            self.status.setText(f"Field providers unavailable: {error or 'capture cancelled'}")
            self.canvas.clear(self.status.text())
            return
        scene, self._display_regions = scene
        self._scene = scene
        with QSignalBlocker(self.sources):
            self.sources.clear()
            for source in scene.sources:
                marker = "" if source.supported else " [unmodelled]"
                item = QListWidgetItem(f"{source.label} · {source.category}{marker}")
                item.setData(Qt.ItemDataRole.UserRole, source.key)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if self._all_sources_intent or source.key in (self._remembered_keys or ()) else Qt.CheckState.Unchecked)
                item.setToolTip(f"{source.model}\n" + "\n".join(source.notes))
                self.sources.addItem(item)
        self._update_details()
        self.focus_component(self._selected_component)
        self._ensure_geometry()
        if self._fit_initial or self._selection_needs_fit:
            self._fit_initial = False
            self._fit_sources()
        else:
            self._schedule()

    @Slot(int, object, object)
    def _sampled(self, generation, result, error):
        self._jobs.pop(generation, None)
        if generation != self._generation or self._fields_stale or not self._active:
            self._resume_latest_request()
            return
        self._worker = None
        if error or result is None:
            self._sample = None
            self.status.setText(f"Field sampling unavailable: {error or 'sampling cancelled'}")
            self.canvas.clear(self.status.text())
            return
        self._base_sample_data = result
        self._detail_key = None
        self._publish_sample(result)
        if (self.auto_detail.isChecked()
                and (np.any(self.canvas._zoom != 1.) or not self.canvas._pan.isNull())):
            self._view_changed()
        if self._detail_dirty:
            self._detail_timer.start()

    def _publish_sample(self, result):
        self._sample, bounds, section = result
        # Switch the field and its physical cut together, never reprojecting
        # samples from the old U-Z plane as if they belonged to the new one.
        with QSignalBlocker(self.canvas):
            self.canvas.set_projection_angle(
                self._sample.section_angle_deg if section == "UZ" else self._projection_angle_deg)
            self.canvas.set_pending_section_angle(None)
        regions = (() if self._display_regions is None else
                   self._display_regions.regions_for_component(self._selected_component))
        self.canvas.set_source_regions(tuple((region.label, region.bounds_m) for region in regions))
        self.canvas.set_observation_regions(self._display_regions, self._selected_component)
        # Input/source invalidation has already cleared old geometry; accepted
        # angle/viewport results belong to this same captured configuration.
        self.canvas.set_sample(self._sample, bounds, section, preserve_geometry=True)
        self.canvas.set_mode(self.quantity.currentData())
        self.canvas.set_render_style(self.render_style.currentData())
        self._refresh_geometry()
        self._refresh_references()
        sample = self._sample
        counts = ", ".join(f"{label} {np.count_nonzero(valid)}/{len(valid)}" for label, valid in
                           (("B", sample.magnetic_valid), ("E", sample.electric_valid), ("force", sample.force_valid)))
        self.status.setText(f"Current snapshot · {len(self.selected_source_keys)} selected sources · valid samples: {counts}. "
                            "Physical-envelope samples only. Unknown values remain unavailable; no extrapolation. Force uses the uniform test-electron settings above.")
        if self.field_scope.currentData() == "component" and not self.selected_source_keys:
            self.status.setText(self.status.text() + " Selected component has no independent field source.")
        elif self.field_scope.currentData() == "component":
            shared = [source.label for source in self._scene.sources
                      if source.key in self.selected_source_keys and len(source.component_keys) > 1]
            if shared:
                self.status.setText(self.status.text() + " Shared source retained intact: " + ", ".join(shared) + ".")
        if self._detail_key is None:
            self.detail_status.setText(
                f"Cached section: {len(sample.points_m)} samples; zoom or pan for visible detail."
                if self.auto_detail.isChecked() else
                "Fixed section samples; navigation only redraws cached data.")
        self._update_details()

    def _view_changed(self):
        if not self.auto_detail.isChecked() or self._base_sample_data is None:
            return
        self._detail_dirty = True
        self._view_revision += 1
        if self._detail_worker is not None:
            self._detail_worker.cancelled.set()
        if self._active and not self._fields_stale:
            self._detail_timer.start()
            self.detail_status.setText("Navigating: cached fields; visible detail follows after a short pause.")

    def _auto_detail_changed(self, enabled):
        self._view_revision += 1
        self._detail_timer.stop()
        if self._detail_worker is not None:
            self._detail_worker.cancelled.set()
        if enabled:
            self._view_changed()
        else:
            self._detail_dirty = False
            self._detail_key = None
            if self._base_sample_data is not None and not self._fields_stale:
                self._publish_sample(self._base_sample_data)
            self.detail_status.setText("Fixed section samples; navigation only redraws cached data.")

    def _request_view_detail(self):
        if (not self._active or self._fields_stale or not self.auto_detail.isChecked()
                or self._scene is None or self._base_sample_data is None):
            return
        if self._detail_worker is not None or self._worker is not None:
            return
        plan = self.canvas.viewport_sample_grid(int(self.density.currentData()), max_points=1600)
        if plan is None:
            self._detail_dirty = False
            if self._detail_key is not None:
                self._detail_key = None
                self._publish_sample(self._base_sample_data)
            self.detail_status.setText(
                "3D uses cached samples and screen-spaced arrows." if self.section.currentData() == "3D" else
                "Section is edge-on: cached samples retained; rotate the view to inspect its plane.")
            return
        points, key = plan
        self._detail_dirty = False
        if key == self._detail_key:
            self.detail_status.setText(f"Visible detail retained: {len(self._sample.points_m)} samples.")
            return
        cached = self._detail_cache.get(key)
        if cached is not None:
            self._detail_cache.move_to_end(key)
            self._detail_key = key
            self._publish_sample(cached)
            self.detail_status.setText(f"Visible detail from cache: {len(cached[0].points_m)} samples.")
            return
        _sample, bounds, section = self._base_sample_data
        angle = self._projection_angle_deg
        points, grid_axes_m = self._section_sampling_plan(points, section)
        if self._display_regions is None:
            points = points[:0]
        else:
            points = points[self._display_regions.mask(points, self._selected_component)]
        if not len(points):
            names = ("points_m", "b_t", "e_v_per_m", "force_n", "magnetic_valid", "electric_valid", "force_valid")
            empty = _Samples(**{name: getattr(_sample, name)[:0] for name in names}, notes=_sample.notes,
                             section_angle_deg=angle)
            self._detail_key = key
            self._publish_sample((empty, bounds, section))
            self.detail_status.setText("No locations inside physical envelopes in this viewport.")
            return
        self._detail_serial += 1
        token = self._detail_serial
        worker = _SampleWorker(token, self._scene, points, self.selected_source_keys, self.energy.value(),
                               (self.direction_x.value(), self.direction_y.value(), self.direction_z.value()),
                               bounds, section, grid_axes_m, angle)
        worker.context = (self._generation, self._view_revision)
        worker.request_key = key
        worker.signals.finished.connect(self._view_detail_ready)
        self._detail_jobs[token] = self._detail_worker = worker
        self.detail_status.setText(f"Refining {len(points)} visible locations using the captured field…")
        QThreadPool.globalInstance().start(worker)

    @Slot(int, object, object)
    def _view_detail_ready(self, token, result, error):
        worker = self._detail_jobs.pop(token, None)
        if worker is None:
            return
        if self._detail_worker is worker:
            self._detail_worker = None
        accepted = (worker.context == (self._generation, self._view_revision)
                    and self._active and not self._fields_stale and self.auto_detail.isChecked()
                    and not worker.cancelled.is_set())
        if accepted and result is not None and not error:
            self._detail_key = worker.request_key
            self._detail_cache[self._detail_key] = result
            self._detail_cache.move_to_end(self._detail_key)
            while len(self._detail_cache) > 4:
                self._detail_cache.popitem(last=False)
            self._publish_sample(result)
            self.detail_status.setText(
                f"Visible detail: {len(result[0].points_m)} samples · axis/source detail · model unchanged.")
        elif accepted and error:
            self.detail_status.setText(f"Visible refinement unavailable: {error}. Previous samples retained.")
        if self._detail_dirty and self._active and not self._fields_stale and not self._detail_timer.isActive():
            self._detail_timer.start()

    def _update_details(self, *_args):
        if self._scene is None:
            return
        selected = set(self.selected_source_keys)
        lines = ["Fields are read from propagation providers. Shared physical channels / joint magnetic circuits appear once.",
                 "Bounds below are enclosing boxes only; validity also depends on each provider's bore, map and near-axis limits.",
                 "E and B are not added numerically. Electron force is q(E + v × B), using a uniform test velocity.", ""]
        lines.extend(str(note) for note in self._scene.notes)
        if self._display_regions is not None:
            lines.extend(("", "Physical observation envelopes (including beam bores):",
                          *self._display_regions.notes))
        lines.append(f"Observation component: {self._selected_component or 'whole column'}; field contribution mode: {self.field_scope.currentText()}.")
        if self._geometry is not None:
            lines.extend(("", "Captured physical geometry:"))
            lines.extend(str(note) for note in self._geometry.notes)
            lines.extend(str(error) for error in self._geometry.errors)
        for source in self._scene.sources:
            if source.key not in selected:
                continue
            bound = np.asarray(source.bounds_m) * 1000
            lines.extend(("", f"{source.label} ({source.key})", f"Model: {source.model}",
                          f"Components: {', '.join(source.component_keys)}",
                          f"Enclosing XYZ bounds [mm]: {bound[0].tolist()} to {bound[1].tolist()}",
                          f"Supported: {'yes' if source.supported else 'no; unavailable, not zero'}"))
            lines.extend(str(note) for note in source.notes)
            if self._geometry is not None:
                lines.extend(self._geometry.source_notes.get(source.key, ()))
        if self._sample is not None:
            lines.extend(("", "Sampling diagnostics:"))
            lines.extend(str(note) for note in self._sample.notes)
        text = "\n".join(lines)
        if text != self.details.toPlainText():
            self.details.setPlainText(text)

    def closeEvent(self, event):
        self.set_active(False)
        super().closeEvent(event)
