"""Lazy, cancellable component-field sampling of calculation snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, QSize, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QGridLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QLayout, QLineEdit,
    QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from temsim.gui.field_vector_canvas import FieldVectorCanvas
from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.gui.input_policy import WheelSafeDoubleSpinBox as QDoubleSpinBox


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
        except Exception as exc:
            error = str(exc)
        if self.cancelled.is_set():
            result, error = None, None
        self.signals.finished.emit(self.generation, result, error)


class _SampleWorker(QRunnable):
    def __init__(self, generation, scene, points, source_keys, energy, direction, bounds_m, section):
        super().__init__()
        self.generation, self.scene, self.points = generation, scene, points
        self.source_keys, self.energy, self.direction = source_keys, energy, direction
        self.bounds_m, self.section = bounds_m, section
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
                result = (_Samples(**values, notes=notes), self.bounds_m, self.section)
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._prepared_scene = None
        self._result_reference = None
        self._z_limits_mm = None
        self._scene = None
        self._sample = None
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

        self.sources = QListWidget()
        self.sources.setObjectName("fieldVectorSources")
        self.sources.setMinimumHeight(80)
        self.sources.setMaximumHeight(145)
        self.sources.setMinimumWidth(150)
        self.sources.itemChanged.connect(self._sources_changed)
        self.sources.currentItemChanged.connect(self._update_details)
        self.select_all = QPushButton("All")
        self.select_none = QPushButton("None")
        self.fit_selected = QPushButton("Fit selected")
        self.select_all.clicked.connect(lambda: self._check_all(True))
        self.select_none.clicked.connect(lambda: self._check_all(False))
        self.fit_selected.clicked.connect(self._fit_sources)
        self.section = QComboBox()
        for label, value in (("XY section", "XY"), ("XZ section", "XZ"), ("YZ section", "YZ"), ("3D volume", "3D")):
            self.section.addItem(label, value)
        self.quantity = QComboBox()
        for label, value in (("B + E overlay", "overlay"), ("Magnetic B", "B"), ("Electric E", "E"), ("Electron force", "force")):
            self.quantity.addItem(label, value)
        self.density = QComboBox()
        for label, count in (("9 samples / axis", 9), ("15 samples / axis", 15), ("21 samples / axis", 21)):
            self.density.addItem(label, count)
        self.density.setCurrentIndex(1)
        self.density.setToolTip("3D volumes are capped at 9 samples per axis (729 points). This controls display sampling only.")
        self.centre_x = _spin(-1e5, 1e5, 0., " mm")
        self.centre_y = _spin(-1e5, 1e5, 0., " mm")
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
        self.canvas.setMinimumHeight(90)
        self.canvas.setObjectName("fieldVectorCanvas")
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
        self.convention = QLabel("Global right-handed XYZ; +Z downstream. Fixed, independent B/E/F arrow references; hollow squares mean unavailable, not zero.")
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
        entries = (("View", self.section), ("Quantity", self.quantity), ("Sampling", self.density),
                   ("Centre X", self.centre_x), ("Centre Y", self.centre_y), ("Centre Z / slice", self.centre_z),
                   ("XY half width", self.radius), ("Z half span", self.axial_half_span))
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
            control.setToolTip("Fixed arrow reference in SI, retained across snapshots. Enter a positive number, e.g. 1e-6.")
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
        self.controls_toggle.setChecked(True)
        self.controls_toggle.toggled.connect(self.control_scroll.setVisible)
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
        self.section.currentIndexChanged.connect(self._controls_changed)
        self.density.currentIndexChanged.connect(self._controls_changed)
        self.quantity.currentIndexChanged.connect(self._quantity_changed)

    def minimumSizeHint(self):
        # An optional hidden QStackedWidget page must not resize other plots.
        return QSize(0, 0)

    @property
    def selected_source_keys(self):
        return tuple(self.sources.item(index).data(Qt.ItemDataRole.UserRole)
                     for index in range(self.sources.count())
                     if self.sources.item(index).checkState() == Qt.CheckState.Checked)

    def update_snapshot(self, state, records=(), z_limits_mm=None, *, prepared_scene=None, result_reference=None):
        self._remember_selection()
        self._cancel_jobs()
        self._invalidate_geometry()
        self._state, self._prepared_scene = state, prepared_scene
        self._result_reference, self._z_limits_mm = result_reference, z_limits_mm
        self._scene, self._sample, self._fields_stale = None, None, False
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
        self._sample = None
        self.canvas.clear("Inputs changed. Recalculate to obtain current fields.")
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        self.status.setText("Stale: inputs changed; waiting for the next completed calculation. Old vectors have been cleared.")

    def invalidate(self, message="Field snapshot pending calculation."):
        self._remember_selection()
        self._cancel_jobs()
        self._invalidate_geometry()
        self._state = self._prepared_scene = self._scene = self._sample = self._result_reference = None
        self._fields_stale = True
        self.canvas.clear(message)
        self.status.setText(message)
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        with QSignalBlocker(self.sources):
            self.sources.clear()
        self.details.clear()

    def set_active(self, active):
        self._active = bool(active)
        if not self._active:
            self._cancel_jobs()
            self._cancel_geometry_jobs()
        else:
            if self._sample is None:
                self._schedule()
            self._ensure_geometry()

    def _cancel_jobs(self):
        self._generation += 1
        self._request_timer.stop()
        for worker in self._jobs.values():
            worker.cancelled.set()
        self._worker = None

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
        meshes = () if self._geometry is None else self._geometry.select(self.selected_source_keys)
        self.canvas.set_physical_meshes(meshes)
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
            self._remembered_keys = set(self.selected_source_keys)

    def _schedule(self):
        if self._active and self._state is not None and not self._fields_stale:
            self._request_timer.start()

    def _controls_changed(self, *_args):
        self._cancel_jobs()
        self._sample = None
        self.canvas.clear("Sampling controls changed…" if not self._fields_stale else "Inputs changed. Recalculate to obtain current fields.")
        self.legend.setText("B [T] | E [V/m] | electron force [N] — no current samples")
        self.settings_changed.emit()
        self._schedule()

    def _sources_changed(self, *_args):
        self._remember_selection()
        if self.sources.count():
            self._all_sources_intent = len(self.selected_source_keys) == self.sources.count()
        self._update_details()
        self._controls_changed()
        self._refresh_geometry()

    def _quantity_changed(self, *_args):
        self.canvas.set_mode(self.quantity.currentData())
        self.legend.setText(self.canvas.legend_text)
        self.settings_changed.emit()

    def _refresh_references(self):
        for key, reference in self.canvas.references.items():
            if reference is not None:
                self.reference_inputs[key].setText(f"{reference:.9g}")
        self.legend.setText(self.canvas.legend_text)

    def _reference_changed(self, key):
        try:
            self.canvas.set_reference(key, float(self.reference_inputs[key].text()))
        except ValueError:
            self.status.setText(f"{key} arrow reference must be a finite positive SI value; previous reference retained.")
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

    def _fit_sources(self, *_args):
        if self._scene is None:
            return
        selected = set(self.selected_source_keys)
        sources = [source for source in self._scene.sources if source.key in selected]
        bounds = [np.asarray(source.bounds_m) for source in sources]
        if not bounds:
            bounds = [np.asarray(self._scene.bounds_m)]
        lower, upper = np.min([bound[0] for bound in bounds], axis=0), np.max([bound[1] for bound in bounds], axis=0)
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
            axes = (0, 1, 2)
        else:
            axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[section]
        vectors = [np.linspace(centre[axis]-half[axis], centre[axis]+half[axis], count)
                   if axis in axes else np.array([centre[axis]]) for axis in range(3)]
        points = np.column_stack([grid.ravel() for grid in np.meshgrid(*vectors, indexing="ij")])
        return points, np.array([centre-half, centre+half]), section

    @Slot()
    def _request_sample(self):
        if not self._active or self._state is None or self._fields_stale:
            return
        if self._worker is not None:
            return
        self._generation += 1
        token = self._generation
        if self._scene is None:
            worker = _PrepareWorker(token, self._state, self._prepared_scene, self._z_limits_mm)
            worker.signals.finished.connect(self._prepared)
            self.status.setText("Capturing supported field providers from the completed calculation…")
        else:
            direction = (self.direction_x.value(), self.direction_y.value(), self.direction_z.value())
            if np.linalg.norm(direction) == 0:
                self.status.setText("Velocity direction must be nonzero; no vectors are displayed.")
                self.canvas.clear(self.status.text())
                return
            points, bounds, section = self._sample_grid()
            worker = _SampleWorker(token, self._scene, points, self.selected_source_keys,
                                   self.energy.value(), direction, bounds, section)
            worker.signals.finished.connect(self._sampled)
            self.status.setText(f"Sampling {len(points)} physical locations from {len(self.selected_source_keys)} selected sources…")
        self._jobs[token] = self._worker = worker
        QThreadPool.globalInstance().start(worker)

    @Slot(int, object, object)
    def _prepared(self, generation, scene, error):
        self._jobs.pop(generation, None)
        if generation != self._generation or self._fields_stale or not self._active:
            return
        self._worker = None
        if error or scene is None:
            self.status.setText(f"Field providers unavailable: {error or 'capture cancelled'}")
            self.canvas.clear(self.status.text())
            return
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
        self._ensure_geometry()
        if self._fit_initial:
            self._fit_initial = False
            self._fit_sources()
        else:
            self._schedule()

    @Slot(int, object, object)
    def _sampled(self, generation, result, error):
        self._jobs.pop(generation, None)
        if generation != self._generation or self._fields_stale or not self._active:
            return
        self._worker = None
        if error or result is None:
            self._sample = None
            self.status.setText(f"Field sampling unavailable: {error or 'sampling cancelled'}")
            self.canvas.clear(self.status.text())
            return
        self._sample, bounds, section = result
        selected = set(self.selected_source_keys)
        self.canvas.set_source_regions(tuple((source.label, source.bounds_m)
                                              for source in self._scene.sources if source.key in selected))
        self.canvas.set_sample(self._sample, bounds, section)
        self.canvas.set_mode(self.quantity.currentData())
        self._refresh_geometry()
        self._refresh_references()
        sample = self._sample
        counts = ", ".join(f"{label} {np.count_nonzero(valid)}/{len(valid)}" for label, valid in
                           (("B", sample.magnetic_valid), ("E", sample.electric_valid), ("force", sample.force_valid)))
        self.status.setText(f"Current snapshot · {len(self.selected_source_keys)} selected sources · valid samples: {counts}. "
                            "Unknown values remain unavailable; no extrapolation. Force uses the uniform test-electron settings above.")
        self._update_details()

    def _update_details(self, *_args):
        if self._scene is None:
            return
        selected = set(self.selected_source_keys)
        lines = ["Fields are read from propagation providers. Shared physical channels / joint magnetic circuits appear once.",
                 "Bounds below are enclosing boxes only; validity also depends on each provider's bore, map and near-axis limits.",
                 "E and B are not added numerically. Electron force is q(E + v × B), using a uniform test velocity.", ""]
        lines.extend(str(note) for note in self._scene.notes)
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
        self.details.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        self.set_active(False)
        super().closeEvent(event)
