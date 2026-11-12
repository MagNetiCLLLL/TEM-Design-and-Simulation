"""Near-specimen particle-conditioned mixed-wave sessions and exact planes.

Transverse Beam can use the applied instrument captured by an accepted
particle result. Each observation starts from the executed particle boundary near the specimen.
Mixed modes contribute intensities, with individual complex fields preserved.
"""

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
import math
from numbers import Real
from threading import Event
from time import monotonic
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRectF, QRunnable, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
    QVBoxLayout, QWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView,
)

from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.gui.coherent_result_labels import coherent_preview_text as _coherent_preview_text
from temsim.calculation_progress import CalculationProgress, callback_progress
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.simulation_modes import coherent_mode_issue, require_coherent_mode
from temsim.physics.mixed_wave_pipeline import MixedWaveObservationSession, MixedWaveRequest, simulate_mixed_wave
from temsim.physics.mixed_wave_particles import particle_source_from_result
from temsim.physics.compute_backend import require_coherent_cuda

GIB = 1024**3


def _coherent_backend_text(checkpoint):
    """Describe executed column/material records, never the requested policy."""
    column_rows = None
    material_rows = None
    record = getattr(checkpoint, "record", None)
    seen = set()
    while isinstance(record, Mapping) and id(record) not in seen:
        seen.add(id(record))
        schema = record.get("schema")
        if column_rows is None and schema in ("executed-column-wave-v1", "executed-column-segment-v1"):
            # Only the current column stage supplies its device evidence. An
            # older upstream GPU stage must not relabel a CPU fallback.
            column_rows = record.get("modes", ())
        if material_rows is None and schema in ("executed-specimen-wave-v1", "executed-inelastic-trajectories-v1"):
            material_rows = []
            material_column = []
            for mode in record.get("modes", ()):
                steps = mode.get("steps", mode.get("slices", ()))
                for step in steps:
                    material_rows.extend(step[key] for key in ("phase_before", "phase_after") if key in step)
                if steps:
                    material_column.extend(steps[-1].get("column", ()))
            if column_rows is None:
                column_rows = material_column
        if column_rows is not None and material_rows is not None:
            break
        record = record.get("upstream")

    def describe(name, rows):
        rows = [row for row in (rows or ()) if isinstance(row, Mapping)]
        backends = sorted({str(row["compute_backend"]) for row in rows if row.get("compute_backend")})
        if not backends:
            return f"{name}: execution device not recorded"
        labels = {"cupy": "GPU (CuPy)", "numpy": "CPU (NumPy)"}
        text = f"{name}: " + ", ".join(labels.get(value, value) for value in backends)
        precision = sorted({str(row["numeric_precision"]) for row in rows if row.get("numeric_precision")})
        text += "; " + (", ".join(precision) if precision else "precision not recorded")
        reasons = sorted({str(row["fallback_reason"]) for row in rows if row.get("fallback_reason")})
        if reasons:
            text += "; fallback: " + "; ".join(reasons)
        return text

    return " | ".join((describe("Column", column_rows), describe("Material", material_rows), "Upstream: accepted particles"))


def _wave_resource_claim(state, request, *, source_pending=True):
    """Reserve mixed-state construction and one downstream propagation stage."""
    stage_peak = max(request.wave_grid.maximum_working_bytes, request.maximum_readout_bytes,
                     request.source.maximum_working_bytes if source_pending else 0)
    cache = request.execution.maximum_ram_cache_bytes
    return ResourceClaim(stage_peak+cache+256*1024**2+_wave_read_session_bytes(request))


def _wave_read_session_bytes(request):
    """Bound short-lived checked modes separately from persistent wave caches."""
    return min(128*1024**2, request.maximum_readout_bytes//4,
               request.execution.maximum_ram_cache_bytes//4)


@dataclass(frozen=True)
class _Preview:
    density: np.ndarray
    bounds_um: np.ndarray
    probability: float
    mode_count: int
    retained_bytes: int
    grid_shapes: tuple[tuple[int, int], ...] = ()
    pair_token: str | None = None
    comparison: object = None
    comparison_error: str | None = None
    axial_bz_t: float | None = None
    magnetic_gauge: object = None
    wave_moments: object = None


def _intensity_preview(checkpoint, *, cancelled=lambda: False, bins=128, include_moments=False):
    """Conservative affine-cell probability density, not detector counts.

    Values are probabilities per emitted electron per square micrometre. Native
    cells from different mode grids are redistributed by area overlap, so a
    change of cell area does not change the displayed integrated probability.
    This is a display only; amplitudes and per-mode phase are untouched.
    """
    from temsim.physics.affine_cell_histogram import affine_cell_histogram
    from temsim.physics.beam_comparison import _WavePlaneMoments

    moments = _WavePlaneMoments(checkpoint) if include_moments else None
    bounds = np.array(((np.inf, -np.inf), (np.inf, -np.inf)))
    count = 0
    resident = 0
    grid_shapes = set()
    modes = checkpoint.beam.modes
    geometries = getattr(modes, "geometries", None)
    # Stored modes expose checked geometry without loading/checksumming each
    # large complex amplitude. The intensity pass below still reads every mode.
    geometry_rows = (geometries() if callable(geometries) else
                     ((mode.plane.amplitude.shape, mode.plane.basis_m,
                       mode.plane.origin_m) for mode in modes))
    for shape, basis, origin in geometry_rows:
        if cancelled():
            raise InterruptedError("Coherent beam display cancelled")
        ny, nx = shape
        grid_shapes.add((int(ny), int(nx)))
        indices = np.array(((-(nx//2)-.5, nx-1-nx//2+.5,
                             nx-1-nx//2+.5, -(nx//2)-.5),
                            (-(ny//2)-.5, -(ny//2)-.5,
                             ny-1-ny//2+.5, ny-1-ny//2+.5)))
        corners = (origin[:, None]+basis@indices)*1e6
        bounds[:, 0] = np.minimum(bounds[:, 0], corners.min(axis=1))
        bounds[:, 1] = np.maximum(bounds[:, 1], corners.max(axis=1))
        count += 1
    if not count:
        # A completely absorbed beam still has an honest zero intensity.
        bounds = np.array(((-.5, .5), (-.5, .5)))
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Executed wave has no finite observation-plane geometry")
    probability = np.zeros((bins, bins))
    for mode in modes:
        if cancelled():
            raise InterruptedError("Coherent beam display cancelled")
        weight = abs(mode.plane.amplitude)**2
        if moments is not None:
            moments.add(mode, weight)
        weight *= mode.weight_per_reference_electron
        contribution, _ = affine_cell_histogram(weight, mode.plane.basis_m*1e6,
            mode.plane.origin_m*1e6, bounds, bins)
        probability += contribution
        resident += sum(getattr(mode.plane, name).nbytes for name in (
            "amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")
            if isinstance(getattr(mode.plane, name), np.ndarray))
    # A streamed checkpoint retains disk references, not all mode arrays.
    if hasattr(checkpoint.beam, "content_identity"):
        resident = 0
    cell_area = np.prod((bounds[:, 1]-bounds[:, 0])/bins)
    density = probability/cell_area
    density.setflags(write=False)
    bounds.setflags(write=False)
    return _Preview(density, bounds, float(probability.sum()), count,
                    resident+density.nbytes+bounds.nbytes, tuple(sorted(grid_shapes)),
                    wave_moments=moments)


class _Signals(QObject):
    solved = Signal(int, int, object, object)
    failed = Signal(int, str)
    progress = Signal(int, str)
    progress_detail = Signal(int, object)
    finished = Signal(int)


def _plane_magnetic_gauge(state, z_mm):
    """Capture diagnostic metadata from this execution's installed optics."""
    from temsim.physics.wave_plane_observables import capture_column_plane_magnetic_gauge
    return capture_column_plane_magnetic_gauge(state, z_mm)


class _WaveWorker(QRunnable):
    def __init__(self, generation, session, state, request, event, retained=(), *, observer=None,
                 pair_context=None, plane_only=False):
        super().__init__()
        self.generation = generation
        self.session = session
        self.state = state
        self.request = request
        self.event = event
        self.observer = observer
        self.pair_context = pair_context
        self.plane_only = plane_only
        # Inventory roots only: these executed displays are never supplied as
        # a replacement source to a mixed-wave calculation.
        self.existing_result = (retained+((pair_context,) if pair_context is not None else ())
                                +((observer,) if observer is not None else ()))
        self._last_progress_time = 0.
        self.job_input_identity = f"coherent-session:{session}:Z:{request.observation_z_mm.hex()}"
        self.resource_claim = _wave_resource_claim(state, request,
            source_pending=getattr(observer, "session", None) is None)
        self.signals = _Signals()
        self.progress_state = CalculationProgress(
            f"coherent:{id(self)}:{generation}", "Coherent beam", "Preparing wave transport",
            stages_completed=0, stages_total=3, stage_name="Wave transport",
            parent_id=getattr(pair_context, "token", None))

    def _report_progress(self, done, total, text):
        now = monotonic()
        update = callback_progress(self.progress_state, done, total, text)
        completed_step = update.total is not None and update.completed == update.total
        if not self.event.is_set() and (update.completed == 0 or completed_step or now-self._last_progress_time >= .05):
            self._last_progress_time = now
            self.progress_state = replace(update,
                stages_completed=self.progress_state.stages_completed, stages_total=3,
                stage_name=self.progress_state.stage_name)
            self.signals.progress_detail.emit(self.generation, self.progress_state)
            suffix = f" ({update.completed}/{update.total})" if update.total is not None else ""
            self.signals.progress.emit(self.generation, f"{update.step}{suffix}")

    def _report_stage(self, completed_stages, label):
        if not self.event.is_set():
            self.progress_state = replace(self.progress_state, step=label, stage_name=label,
                completed=None, total=None, stages_completed=completed_stages)
            self.signals.progress_detail.emit(self.generation, self.progress_state)
            self.signals.progress.emit(self.generation, label)

    def run(self):
        from temsim.physics.wave_checkpoint_store import wave_read_session
        # Prefix verification, display, and matched moments read the same
        # immutable checkpoint. Retain a bounded snapshot for this request
        # only; the next request still verifies all files it reads.
        with wave_read_session(_wave_read_session_bytes(self.request), cancelled=self.event.is_set) as read_session:
            self._run_with_read_session(read_session)

    def _run_with_read_session(self, read_session):
        try:
            if self.pair_context is not None:
                from temsim.instrument_snapshot import capture_instrument_snapshot
                if capture_instrument_snapshot(self.state).digest != self.pair_context.instrument_identity:
                    raise ValueError("Captured coherent source or optics do not match this pair")
                self.pair_context.verify_particle()
            # CoordinatedPool holds the application's shared numerical lease
            # and its CPU budget. Do not create a nested independent pool.
            if self.observer is None:
                raise ValueError("A mixed-wave calculation requires an accepted particle observer")
            result = self.observer.observe(self.request.observation_z_mm,
                cancelled=self.event.is_set, progress_callback=self._report_progress)
            self._report_stage(1, "Reading observation-plane fields" if self.plane_only
                               else "Building intensity and phase readout")
            # Probability current needs the executed magnetic state, not the
            # live instrument which may have changed while this job ran.
            unified_column = getattr(result.checkpoint, "record", {}).get("unified_column_executed", False)
            if unified_column:
                from temsim.physics.unified_wave_runtime import unified_checkpoint_magnetic_gauge
                gauge = unified_checkpoint_magnetic_gauge(result.checkpoint)
            else:
                gauge = _plane_magnetic_gauge(self.state, result.checkpoint.plane_z_mm)
                from temsim.physics.wave_plane_observables import checkpoint_magnetic_gauge
                gauge = checkpoint_magnetic_gauge(gauge, result.checkpoint)
            axial_bz_t = None if unified_column or gauge is None or gauge.fields else gauge.aligned_bz_t
            if self.plane_only:
                # Transverse Beam has its own readout worker. Avoid reading
                # every mode here just to construct a second intensity map.
                if not self.event.is_set():
                    self._report_stage(2, "Publishing coherent plane")
                    self.signals.solved.emit(self.generation, self.session, result, (axial_bz_t, gauge))
                return
            preview = _intensity_preview(result.checkpoint, cancelled=self.event.is_set,
                                         include_moments=self.pair_context is not None)
            preview = replace(preview, magnetic_gauge=gauge, axial_bz_t=axial_bz_t)
            if self.pair_context is not None and not self.event.is_set():
                context = self.pair_context
                try:
                    context.verify_wave(result)
                    from temsim.gui.beam_plane_data import sample_beam_plane
                    from temsim.physics.beam_comparison import compare_beam_planes
                    plane = sample_beam_plane(context.particle_result, result.checkpoint.plane_z_mm)
                    if (context.specimen_active and plane.z_mm > context.specimen_z_mm
                            and plane.provenance != "Specimen exit"):
                        raise ValueError("Executed specimen-exit particles are unavailable at this Z; the optical reference is not a specimen comparison")
                    comparison = compare_beam_planes(plane, result.checkpoint, preview.wave_moments)
                    preview = replace(preview, pair_token=context.token, comparison=comparison)
                except (ValueError, TypeError) as error:
                    preview = replace(preview, pair_token=context.token,
                                      comparison_error=str(error))
            preview = replace(preview, wave_moments=None)
            if not self.event.is_set():
                self._report_stage(2, "Publishing coherent result")
                self.signals.solved.emit(self.generation, self.session, result, preview)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
        finally:
            read_session.clear()
            # The exception variable has now released its numerical frames.
            # Drain retired device pools even after cancellation or failure.
            from temsim.physics.wave_device import release_device_memory
            try:
                release_device_memory()
            finally:
                self.signals.finished.emit(self.generation)


@dataclass
class _TransverseWaveSession:
    particle_result: object
    state: object
    request: object
    observer: object
    z_mm: float
    completed: tuple | None = None


class _ObservableLabel(QLabel):
    text_changed = Signal(str)

    def setText(self, text):
        if text == self.text():
            return
        super().setText(text)
        self.text_changed.emit(text)


def _label(text="", *, observable=False):
    result = (_ObservableLabel if observable else QLabel)(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                   | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    result.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
    result.setWordWrap(True)
    result.setMinimumWidth(0)
    return result


def _double(value, minimum, maximum, suffix="", decimals=5):
    control = QDoubleSpinBox()
    control.setDecimals(decimals)
    control.setRange(minimum, maximum)
    control.setValue(value)
    control.setSuffix(suffix)
    control.setKeyboardTracking(False)
    return control


def _integer(value, minimum, maximum):
    control = QSpinBox()
    control.setRange(minimum, maximum)
    control.setValue(value)
    return control


class CoherentIntensityView(QWidget):
    """A lazy display of an accepted preview; this widget never propagates waves."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot = pg.PlotWidget(background="#080e19")
        self.plot.setLabel("bottom", "X", units="µm")
        self.plot.setLabel("left", "Y", units="µm")
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=.2)
        self.image = pg.ImageItem(axisOrder="col-major")
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.plot.addItem(self.image)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self._pending = None
        self._rendered = None
        self._bounds = None
        self.fit_button = QPushButton("Fit full")
        self.fit_button.setToolTip("Fit the displayed wave area once. Moving Z preserves your zoom and position.")
        self.fit_button.setEnabled(False)
        self.fit_button.clicked.connect(self.fit_full)

    @Slot(object, object, int)
    def set_observation(self, result, preview, scale):
        if preview is None:
            return
        self._pending = (result, preview, scale)
        if self.isVisible():
            self._render()

    def _render(self):
        if self._pending is None:
            return
        result, preview, scale = self._pending
        identity = (id(result), id(preview), scale)
        if identity == self._rendered:
            return
        # A changed wave domain is data geometry, not a request to move the
        # observer's viewport. Also stop persistent auto-range after a manual A.
        self.plot.disableAutoRange()
        density = preview.density
        maximum = max(float(density.max()), np.finfo(float).tiny)
        if scale == 1:
            display = np.log1p(1000.*(density/maximum))/np.log1p(1000.)
            self.image.setImage(display, autoLevels=False, levels=(0., 1.))
        else:
            self.image.setImage(density, autoLevels=False, levels=(0., maximum))
        bounds = preview.bounds_um
        first_image = self._bounds is None
        if self._bounds is None or not np.array_equal(bounds, self._bounds):
            self.image.setRect(QRectF(bounds[0, 0], bounds[1, 0],
                bounds[0, 1]-bounds[0, 0], bounds[1, 1]-bounds[1, 0]))
            self._bounds = bounds
        self.fit_button.setEnabled(True)
        if first_image:
            self.fit_full()
        label = ("log display of intensity density" if scale == 1
                 else "intensity density")
        # A long single-line title forces PlotItem's minimum width beyond the
        # narrow Ray sidebar and clips the right half of the physical screen.
        self.plot.setTitle(f"Exact Z {result.checkpoint.plane_z_mm:.9g} mm<br>{label}")
        self.plot.setToolTip("Intensity density / µm² per emitted electron. "
                            "The logarithmic scale changes display only.")
        self._rendered = identity

    @Slot()
    def fit_full(self):
        """One-shot display fit; never propagate or modify a captured result."""
        if self._bounds is not None:
            self.plot.setRange(xRange=self._bounds[0], yRange=self._bounds[1], padding=.02)

    def showEvent(self, event):
        super().showEvent(event)
        self._render()


class CoherentBeamMirror(QWidget):
    """Ray Diagram readout of the page's one captured calculation session."""

    controls_requested = Signal()

    def __init__(self, owner, parent=None):
        super().__init__(parent)
        self.setObjectName("rayCoherentBeamReadout")
        self.follow_ray = QCheckBox("Follow Ray Z")
        self.follow_ray.setChecked(owner.follow_ray.isChecked())
        self.follow_ray.setToolTip("Shares the Electron beam page setting. When unchecked, Ray Diagram Z does not change the observation plane.")
        self.follow_ray.toggled.connect(owner.follow_ray.setChecked)
        owner.follow_ray.toggled.connect(self.follow_ray.setChecked)
        self.intensity_scale = QComboBox()
        for index in range(owner.intensity_scale.count()):
            self.intensity_scale.addItem(owner.intensity_scale.itemText(index))
        self.intensity_scale.setCurrentIndex(owner.intensity_scale.currentIndex())
        self.intensity_scale.currentIndexChanged.connect(owner.intensity_scale.setCurrentIndex)
        owner.intensity_scale.currentIndexChanged.connect(self.intensity_scale.setCurrentIndex)
        self.controls_button = QPushButton("Controls…")
        self.controls_button.clicked.connect(self.controls_requested)
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.follow_ray)
        toolbar.addWidget(self.intensity_scale)
        toolbar.addStretch()
        toolbar.addWidget(self.controls_button)
        self.status = _label(owner.status.text())
        self.timing_status = _label(owner.timing_status.text())
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.plane_positions = _label(owner.plane_positions.text())
        self.plane_positions.setToolTip(owner.plane_positions.toolTip())
        self.screen = CoherentIntensityView()
        toolbar.insertWidget(toolbar.count()-1, self.screen.fit_button)
        self.readout = _label(owner.readout.text())
        self.comparison_status = _label(owner.comparison_status.text())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(toolbar)
        layout.addWidget(self.plane_positions)
        layout.addWidget(self.status)
        layout.addWidget(self.timing_status)
        layout.addWidget(self.screen, 1)
        layout.addWidget(self.readout)
        layout.addWidget(self.comparison_status)
        owner.status.text_changed.connect(self.status.setText)
        owner.plane_positions.text_changed.connect(self.plane_positions.setText)
        owner.readout.text_changed.connect(self.readout.setText)
        owner.comparison_status.text_changed.connect(self.comparison_status.setText)
        owner.timing_status.text_changed.connect(self.timing_status.setText)
        owner.observation_changed.connect(self.screen.set_observation)
        self.screen.set_observation(owner.result, owner.preview,
                                    owner.intensity_scale.currentIndex())


class _AcceptedParticleObserver:
    """Prepare an exact physical boundary lazily inside the numerical worker."""

    def __init__(self, particle_result, state, request):
        self.particle_result, self.state, self.request = particle_result, state, request
        self.source = self.session = None

    def observe(self, z_mm, *, cancelled=lambda: False, progress_callback=None):
        require_coherent_cuda("Require GPU")
        if cancelled():
            raise InterruptedError("Mixed-wave observation cancelled")
        if self.session is None:
            self.source = particle_source_from_result(self.particle_result,
                cancelled=cancelled, progress_callback=progress_callback)
            if cancelled():
                raise InterruptedError("Mixed-wave boundary preparation cancelled")
            self.session = MixedWaveObservationSession(self.state, self.request, self.source)
        return self.session.observe(z_mm, cancelled=cancelled, progress_callback=progress_callback)


class CoherentBeamPage(QWidget):
    """Mixed waves start from one accepted particle result near the specimen."""

    QUERY_INTERVAL_MS = 40
    CACHE_LIMIT = 64
    observation_changed = Signal(object, object, int)
    source_applied = Signal(object)
    tip_editor_requested = Signal()
    paired_calculation_requested = Signal()
    pair_invalidated = Signal()
    progress_detail = Signal(object)
    transverse_wave_ready = Signal(object, object, object)
    transverse_wave_failed = Signal(float, str)
    _SCOPE = (
        "Particles retain the applied source and installed optics up to the specimen entrance. "
        "A declared reduced mixed state starts there; it does not recover relative source phases. "
        "CUDA propagates each retained mode through the specimen and downstream optics. "
        "Independent modes add intensity. Observation planes before the handoff are unavailable."
    )

    def __init__(self, parent=None, *, state_provider=None, particle_result_provider=None):
        super().__init__(parent)
        self.setObjectName("coherentBeamPage")
        self.state_provider = state_provider
        self.particle_result_provider = particle_result_provider
        self.source_applier = None
        self.paired_busy_provider = None
        self.live_refresh_enabled = False
        self._state = self._captured = self._request = self._observation_session = None
        self._particle_result = None
        self._session = self._generation = self._transverse_generation = 0
        self._worker = self._transverse_worker = self._transverse_session = None
        self._transverse_progress = None
        self._pending_z = self._closed = self._session_ready = False
        self._stale = True
        self._cache = OrderedDict()
        self.result = self.preview = self._pair_context = None
        self._target_z_mm = 1.
        self._target_initialized = False
        self.calculate_button = QPushButton("Calculate mixed wave")
        self.calculate_button.setObjectName("calculateElectronBeam")
        self.calculate_button.setProperty("calculationAction", True)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.compare_classical = QCheckBox("Compare accepted particles")
        self.compare_classical.hide()
        self.target_z = _double(1., -1e6, 1e6, " mm", 12)
        self.target_z.setObjectName("coherentObservationZ")
        self.follow_ray = QCheckBox("Follow Ray Z")
        self.follow_ray.setChecked(True)
        self.intensity_scale = QComboBox()
        self.intensity_scale.addItems(("Linear intensity", "Log intensity"))
        toolbar = QHBoxLayout()
        for widget in (self.calculate_button, self.cancel_button, QLabel("Z"), self.target_z,
                       self.follow_ray, self.intensity_scale):
            toolbar.addWidget(widget)
        toolbar.addStretch()
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setObjectName("coherentZSlider")
        self.z_slider.setRange(0, 10000)
        self.z_range_min = _double(0., -1e6, 1e6, " mm", 9)
        self.z_range_max = _double(1., -1e6, 1e6, " mm", 9)
        browse = QHBoxLayout()
        for widget in (QLabel("Z range"), self.z_range_min, self.z_slider, self.z_range_max):
            browse.addWidget(widget, 1 if widget is self.z_slider else 0)

        self.numerical_settings = QGroupBox("Mixed-wave numerical settings")
        self.numerical_settings.setObjectName("coherentNumericalSettings")
        form = QFormLayout(self.numerical_settings)
        self.source_info = _label("Calculate particles first with Preview or High accuracy, then calculate the mixed wave.")
        self.mode_status = _label()
        form.addRow(self.source_info)
        form.addRow(self.mode_status)
        default = MixedWaveRequest()
        self._source_controls = {}
        for field in fields(default.source):
            value = getattr(default.source, field.name)
            if field.name.endswith("bytes"):
                control = _double(value/GIB, .01, 2048., " GiB", 3)
                control.setProperty("byteScale", GIB)
            elif type(value) is int:
                control = _integer(value, 1, min(2**31-1, max(65536, value)))
            elif type(value) is float:
                control = _double(value, 0., max(1e6, value), decimals=9)
            elif type(value) is bool:
                control = QCheckBox()
                control.setChecked(value)
            else:
                continue
            labels = {
                "pixels": "Initial wave pixels", "width_nm": "Wave width (nm)",
                "maximum_modes": "Maximum retained modes", "maximum_mode_loss_fraction": "Maximum omitted mode fraction",
                "maximum_working_bytes": "Source construction memory", "source_image_rms_pm": "Source-image RMS (pm)",
                "spatial_quadrature_order": "Source-image quadrature order",
                "frequency_support_inv_angstrom": "Frequency support (1/Å)",
            }
            if field.name == "pixels":
                control.setRange(32, 8192)
            elif field.name == "maximum_modes":
                control.setRange(1, 4096)
            elif field.name == "spatial_quadrature_order":
                control.setRange(1, 32)
            elif field.name == "maximum_mode_loss_fraction":
                control.setDecimals(12)
                control.setRange(1e-12, 1e-5)
            control.setObjectName("mixed_source_"+field.name)
            self._source_controls[field.name] = control
            form.addRow(labels.get(field.name, field.name.replace("_", " ").capitalize()), control)
        self.maximum_grid_pixels = _integer(default.wave_grid.maximum_pixels, 32, 65536)
        self.maximum_grid_pixels.setObjectName("coherent_maximum_grid_pixels")
        self.column_step = _double(default.column_step_mm, 1e-6, 100., " mm", 6)
        self.column_step.setObjectName("coherent_column_step")
        self.working_gib = _double(8., .01, 2048., " GiB", 3)
        self.gpu_working_gib = _double(24., .01, 2048., " GiB", 3)
        self.ram_cache_gib = _double(default.execution.maximum_ram_cache_bytes/GIB, 0., 2048., " GiB", 3)
        self.disk_cache_gib = _double(default.execution.maximum_disk_cache_bytes/GIB, 0., 4096., " GiB", 3)
        self.material_trajectories = _integer(default.inelastic.trajectories_per_mode, 1, 1000000)
        for label, control in (("Maximum refined grid pixels", self.maximum_grid_pixels),
                ("Downstream propagation step", self.column_step),
                ("Working memory limit", self.working_gib), ("GPU working limit", self.gpu_working_gib),
                ("RAM cache limit", self.ram_cache_gib), ("Disk cache limit", self.disk_cache_gib),
                ("Material trajectories per mode", self.material_trajectories)):
            form.addRow(label, control)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(self.numerical_settings)
        self.controls_scroll.setMinimumWidth(260)
        self.screen = CoherentIntensityView()
        from temsim.gui.electron_beam_observation import ElectronBeamObservation
        self.observation = ElectronBeamObservation(self.screen)
        # The former ideal reference restarts the Tip chain. It is not a
        # comparison of this declared near-specimen mixed state.
        index = self.observation.mode.findData("wavefront_comparison")
        if index >= 0:
            self.observation.mode.removeItem(index)
        self.plot, self.image = self.screen.plot, self.screen.image
        self.observation_changed.connect(self.observation.set_observation)
        splitter = QSplitter()
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(self.observation)
        splitter.setStretchFactor(1, 1)
        self.scope_label = _label(self._SCOPE)
        self.status = _label("Calculate particles, then calculate the mixed wave at the selected plane.", observable=True)
        self.timing_status = _label("Arrival times come from the accepted particles.", observable=True)
        self.timing_status.setObjectName("coherentEmissionTimeStatus")
        self.plane_positions = _label(observable=True)
        self.plane_positions.setObjectName("coherentPlanePositions")
        self.plane_positions.setToolTip("Requested, calculating and displayed planes are separate. Moving Z does not relabel an earlier result.")
        self.readout = _label("Intensity is relative to the original emitted current; lost current is not restored.", observable=True)
        self.comparison_status = _label("The accepted particle snapshot supplies the mixed state.", observable=True)
        self.comparison_table = QTableWidget(0, 3)
        self.comparison_table.setHorizontalHeaderLabels(("Same-Z observable", "Particles", "Coherent modes"))
        self.comparison_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.comparison_table.verticalHeader().hide()
        self.comparison_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.comparison_table.setMaximumHeight(205)
        self.comparison_table.hide()
        layout = QVBoxLayout(self)
        for row in (toolbar, browse):
            layout.addLayout(row)
        for widget in (self.scope_label, self.plane_positions, self.status, self.timing_status):
            layout.addWidget(widget)
        layout.addWidget(splitter, 1)
        for widget in (self.readout, self.comparison_status, self.comparison_table):
            layout.addWidget(widget)
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self.pool.coordinator.register_retained(self, "_retained_wave_roots")
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._start_query)
        self._transverse_pool = CoordinatedPool(self, coordinator=self.pool.coordinator)
        self._transverse_timer = QTimer(self)
        self._transverse_timer.setSingleShot(True)
        self._transverse_timer.timeout.connect(self._start_transverse_query)
        self.calculate_button.clicked.connect(self.calculate)
        self.cancel_button.clicked.connect(self.cancel)
        self.target_z.valueChanged.connect(self._z_changed)
        self.z_slider.valueChanged.connect(self._slider_changed)
        self.z_range_min.valueChanged.connect(self._sync_z_slider)
        self.z_range_max.valueChanged.connect(self._sync_z_slider)
        self.intensity_scale.currentIndexChanged.connect(self._refresh_intensity)
        for control in (*self._source_controls.values(), self.maximum_grid_pixels, self.column_step,
                        self.working_gib, self.gpu_working_gib, self.ram_cache_gib, self.disk_cache_gib,
                        self.material_trajectories):
            signal = control.toggled if isinstance(control, QCheckBox) else control.valueChanged
            signal.connect(self.mark_inputs_stale)
        self._update_plane_positions()

    def update_simulation_mode(self, state):
        issue = coherent_mode_issue(state)
        self.mode_status.setText(issue or "Mixed waves use the accepted particle snapshot and CUDA.")
        self.calculate_button.setEnabled(issue is None)

    def particle_result_published(self, result):
        """A newly accepted publication retires calculations using older arrays."""
        if not self._stale:
            self.mark_inputs_stale()
        state = getattr(result, "state_snapshot", None)
        if state is not None:
            self.source_info.setText(
                "Accepted particle snapshot available. The exact specimen entrance, source weights, "
                "energies and arrival times are checked when calculation starts.")

    def show_numerical_settings(self):
        self.controls_scroll.ensureWidgetVisible(self.numerical_settings)

    def set_state(self, state, *, reset_source=False):
        if self._closed:
            return
        self.mark_inputs_stale()
        self._state = state
        self.update_simulation_mode(state)
        if not self._target_initialized and state is not None:
            z_mm = float(state.sample.z_mm)-float(getattr(state.sample, "thickness_nm", 0.))*.5e-6
            self._target_z_mm = z_mm
            self.target_z.blockSignals(True)
            self.target_z.setValue(z_mm)
            self.target_z.blockSignals(False)
            self.z_range_min.setValue(z_mm)
            self.z_range_max.setValue(z_mm+1.)
            self._target_initialized = True
            self._update_plane_positions()

    def _make_request(self, state=None, *, observation_z_mm=None, **_unused):
        default = MixedWaveRequest()
        source_values = {}
        for name, control in self._source_controls.items():
            value = control.isChecked() if isinstance(control, QCheckBox) else control.value()
            if control.property("byteScale"):
                value = round(value*GIB)
            source_values[name] = value
        working = round(self.working_gib.value()*GIB)
        return replace(default, observation_z_mm=self._target_z_mm if observation_z_mm is None else observation_z_mm,
            source=replace(default.source, **source_values), column_step_mm=self.column_step.value(),
            wave_grid=replace(default.wave_grid, maximum_pixels=self.maximum_grid_pixels.value(),
                maximum_working_bytes=working, compute_backend="Require GPU", acceleration_enabled=True,
                maximum_device_working_bytes=round(self.gpu_working_gib.value()*GIB)),
            inelastic=replace(default.inelastic, trajectories_per_mode=self.material_trajectories.value()),
            maximum_readout_bytes=working,
            execution=replace(default.execution, maximum_ram_cache_bytes=round(self.ram_cache_gib.value()*GIB),
                maximum_disk_cache_bytes=round(self.disk_cache_gib.value()*GIB))).validate()

    def _capture_particle_inputs(self, particle_result, z_mm, *, for_execution=True):
        particle_state = getattr(particle_result, "state_snapshot", None)
        if particle_state is None:
            raise ValueError("Calculate particles first; an accepted instrument snapshot is required.")
        require_coherent_mode(particle_state)
        start_z = float(particle_state.sample.z_mm)-float(getattr(particle_state.sample, "thickness_nm", 0.))*.5e-6
        if z_mm < start_z:
            raise ValueError(f"Mixed waves start at specimen entrance Z {start_z:.12g} mm; select that plane or a downstream plane.")
        request = self._make_request(particle_state, observation_z_mm=z_mm)
        if for_execution:
            require_coherent_cuda("Require GPU")
        state = capture_instrument_snapshot(particle_state).restore()
        return state, request

    def capture_calculation_inputs(self, *, for_execution=True):
        particle_result = self.particle_result_provider() if self.particle_result_provider is not None else None
        state, request = self._capture_particle_inputs(particle_result, self._target_z_mm,
                                                     for_execution=for_execution)
        self._particle_result = particle_result
        summary = {"status": "PARTICLE_SNAPSHOT_READY", "model": "Near-specimen particle-conditioned mixed state"}
        return state, request, summary

    def calculate(self):
        if self._closed:
            return
        self._cancel_request()
        try:
            state, request, summary = self.capture_calculation_inputs()
            self.start_captured_calculation(state, request, summary)
        except Exception as error:
            self.status.setText(f"Mixed wave not started: {error}")
            self.progress_detail.emit(CalculationProgress(f"mixed-check:{uuid4().hex}",
                "Mixed wave", "Checking accepted particle inputs", stages_completed=0,
                stages_total=1).terminal("failed", str(error)))

    def start_captured_calculation(self, captured, request, summary=None, *, pair_context=None):
        if self._closed:
            return
        particle_result = pair_context.particle_result if pair_context is not None else self._particle_result
        if particle_result is None:
            raise ValueError("An accepted particle result is required for a mixed wave.")
        require_coherent_cuda("Require GPU")
        require_coherent_mode(captured)
        self._cancel_request()
        self._session += 1
        self._cache.clear()
        self._session_ready = False
        self._captured, self._request = captured, request
        self._particle_result, self._pair_context = particle_result, pair_context
        self._observation_session = _AcceptedParticleObserver(particle_result, captured, request)
        self._stale = False
        self.status.setText("Preparing the accepted particle boundary near the specimen; CUDA propagation follows.")
        self.timing_status.setText("The handoff retains executed particle energies and flight times.")
        self.observation.mark_previous_calculation("A new mixed-wave calculation is running. Previous complete data remains displayed.")
        self._start_query()

    def request_transverse_wave(self, particle_result, z_mm):
        if self._closed:
            return
        if isinstance(z_mm, bool) or not isinstance(z_mm, Real) or not math.isfinite(z_mm):
            self.release_transverse_wave()
            self.transverse_wave_failed.emit(float("nan"), "A coherent observation needs a finite Z.")
            return
        z_mm = float(z_mm)
        session = self._transverse_session
        if session is None or session.particle_result is not particle_result:
            self.release_transverse_wave()
            try:
                state, request = self._capture_particle_inputs(particle_result, z_mm)
                session = _TransverseWaveSession(particle_result, state, request,
                    _AcceptedParticleObserver(particle_result, state, request), z_mm)
            except Exception as error:
                self.transverse_wave_failed.emit(z_mm, str(error))
                return
            self._transverse_session = session
        session.z_mm = z_mm
        if session.completed is not None and session.completed[0].plane_z_mm == z_mm:
            self.transverse_wave_ready.emit(*session.completed)
        elif self._transverse_worker is None and not self._transverse_timer.isActive():
            self._transverse_timer.start(self.QUERY_INTERVAL_MS)

    @property
    def transverse_wave_readout_budget(self):
        session = self._transverse_session
        return 512*1024**2 if session is None else session.request.maximum_readout_bytes

    def mark_inputs_stale(self, *_args):
        if self._closed:
            return
        session = self._transverse_session
        if session is not None:
            self.release_transverse_wave()
            self.transverse_wave_failed.emit(session.z_mm,
                "Mixed-wave inputs changed. Turn Coherent wave off and on to calculate with the current settings.")
        self._cancel_request()
        self._stale, self._session_ready = True, False
        self._cache.clear()
        self._observation_session = None
        self.invalidate_pair("Settings changed; calculate a new mixed wave from updated particles.")
        self.observation.mark_previous_calculation("Previous calculation · inputs changed. Calculate the mixed wave again.")
        self.status.setText("Inputs changed. Calculate particles if needed, then calculate the mixed wave.")

    def _update_cancel_enabled(self):
        paired_busy = bool(self.paired_busy_provider and self.paired_busy_provider())
        self.cancel_button.setEnabled(self._worker is not None or self._transverse_worker is not None
            or self._transverse_timer.isActive() or paired_busy)

    def invalidate_pair(self, message="Previous complete data is retained."):
        self._pair_context = None
        self.comparison_table.hide()
        self.comparison_status.setText(message)
        self.pair_invalidated.emit()

    def _retained_wave_roots(self):
        return (self.result, self.preview, self._cache, self._captured, self._particle_result,
                self._observation_session, self._transverse_session)

    def _display(self, result, preview):
        self.result, self.preview = result, preview
        self._display_comparison(preview)
        self._update_plane_positions()
        self._refresh_intensity()
        self.observation.clear_previous_calculation_notice()
        continuation = (f"Updating requested Z {self._target_z_mm:.12g} mm."
            if result.checkpoint.plane_z_mm != self._target_z_mm else
            "Move Z to propagate from the same accepted particle boundary.")
        self.status.setText(f"Completed mixed wave at exact Z {result.checkpoint.plane_z_mm:.12g} mm. "
                            + continuation)
        grids = "; ".join(f"{nx} × {ny}" for ny, nx in preview.grid_shapes)
        self.readout.setText(f"{_coherent_backend_text(result.checkpoint)} | {preview.mode_count} modes | "
            f"Wave grids: {grids or 'unavailable'} | probability {preview.probability:.8g} per emitted electron | "
            f"reference current {result.checkpoint.reference_current_a*1e9:.6g} nA. "
            "Mixed modes add intensity; phase belongs to individual modes.")

    def cancel(self):
        if self._closed:
            return
        session = self._transverse_session
        self.release_transverse_wave()
        if session is not None:
            self.transverse_wave_failed.emit(session.z_mm, "Mixed-wave calculation cancelled.")
        self._cancel_request()
        self.invalidate_pair("Mixed-wave calculation cancelled. Previous complete data is retained.")
        self.status.setText("Mixed-wave request cancelled.")
        self.observation.mark_previous_calculation("The requested calculation was cancelled. No new plane was accepted.")

    def shutdown(self, msecs=3000):
        if not self._closed:
            self.cancel()
            self._closed = True
            self._observation_session = None
        observation_done = self.observation.shutdown(msecs)
        transverse_done = self._transverse_pool.waitForDone(msecs)
        return self.pool.waitForDone(msecs) and observation_done and transverse_done

    def closeEvent(self, event):
        if self.shutdown():
            event.accept()
        else:
            event.ignore()

    @Slot()
    def _sync_z_slider(self, *_args):
        lower, upper = self.z_range_min.value(), self.z_range_max.value()
        self.z_slider.setEnabled(upper > lower)
        if upper > lower:
            fraction = min(1., max(0., (self._target_z_mm-lower)/(upper-lower)))
            value = round(fraction*self.z_slider.maximum())
            self.z_slider.blockSignals(True)
            self.z_slider.setValue(value)
            self.z_slider.blockSignals(False)


    @Slot(int)
    def _slider_changed(self, value):
        lower, upper = self.z_range_min.value(), self.z_range_max.value()
        if upper <= lower:
            return
        z_mm = lower+(upper-lower)*value/self.z_slider.maximum()
        self.target_z.blockSignals(True)
        self.target_z.setValue(z_mm)
        self.target_z.blockSignals(False)
        self._z_changed(z_mm)


    def _cancel_request(self):
        self._terminal_progress("cancelled", "Coherent request cancelled")
        if hasattr(self, "observation"):
            self.observation.cancel_reference_work()
        self._generation += 1
        self.timer.stop()
        self._pending_z = False
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()
        self._worker = None
        self._update_cancel_enabled()
        self._update_plane_positions()


    @Slot()
    def _start_transverse_query(self):
        session = self._transverse_session
        if self._closed or session is None or self._transverse_worker is not None:
            return
        if session.completed is not None and session.completed[0].plane_z_mm == session.z_mm:
            return
        self._transverse_generation += 1
        request = replace(session.request, observation_z_mm=session.z_mm)
        worker = _WaveWorker(self._transverse_generation, -self._transverse_generation,
                             session.state, request, Event(), observer=session.observer, plane_only=True)
        worker.signals.solved.connect(self._transverse_solved)
        worker.signals.failed.connect(self._transverse_failed)
        worker.signals.progress_detail.connect(self._transverse_progressed)
        worker.signals.finished.connect(self._transverse_finished)
        self._transverse_worker = worker
        self._update_cancel_enabled()
        self._transverse_progressed(worker.generation, worker.progress_state)
        if self._transverse_worker is worker:
            self._transverse_pool.start(worker)


    @Slot(int, int, object, object)
    def _transverse_solved(self, generation, _session, result, fields):
        worker = self._transverse_worker
        if worker is None or generation != worker.generation:
            return
        checkpoint = result.checkpoint
        if checkpoint.plane_z_mm != worker.request.observation_z_mm:
            self._transverse_failed(generation, "Executed Z differs from the requested observation plane.")
            return
        self._end_transverse_progress("succeeded", "Completed coherent plane")
        session = self._transverse_session
        if self._transverse_worker is worker and session.z_mm == checkpoint.plane_z_mm:
            session.completed = (checkpoint, *fields)
            self.transverse_wave_ready.emit(*session.completed)


    @Slot(int, str)
    def _transverse_failed(self, generation, message):
        worker = self._transverse_worker
        if worker is None or generation != worker.generation:
            return
        self._end_transverse_progress("failed", message)
        session = self._transverse_session
        if self._transverse_worker is worker and session.z_mm == worker.request.observation_z_mm:
            self.transverse_wave_failed.emit(session.z_mm, message)


    @Slot(int, object)
    def _transverse_progressed(self, generation, update):
        worker = self._transverse_worker
        if worker is not None and generation == worker.generation:
            self._transverse_progress = update
            self.progress_detail.emit(update)


    def _end_transverse_progress(self, status, message):
        progress, self._transverse_progress = self._transverse_progress, None
        if progress is not None:
            self.progress_detail.emit(progress.terminal(status, message))


    @Slot(int)
    def _transverse_finished(self, generation):
        worker = self._transverse_worker
        if worker is None or generation != worker.generation:
            return
        self._transverse_worker = None
        self._end_transverse_progress("cancelled", "Wave request ended without a new accepted result")
        self._update_cancel_enabled()
        session = self._transverse_session
        if session is not None and session.z_mm != worker.request.observation_z_mm:
            # The completed query may have published a reusable upstream
            # prefix. Only the newest still-requested Z gets another job.
            self._transverse_timer.start(0)


    @Slot()
    def release_transverse_wave(self):
        """Cancel queued/running inline queries without affecting other wave views."""
        self._transverse_timer.stop()
        self._transverse_session = None
        worker, self._transverse_worker = self._transverse_worker, None
        if worker is not None:
            worker.event.set()
            self._transverse_pool.clear()
        self._end_transverse_progress("cancelled", "Transverse Beam request cancelled")
        self._update_cancel_enabled()


    def _update_plane_positions(self):
        worker = self._worker
        calculating = (f"{worker.request.observation_z_mm:.9g} mm"
                       if worker is not None else "—")
        displayed = (f"{self.result.checkpoint.plane_z_mm:.9g} mm"
                     if self.result is not None else "—")
        self.plane_positions.setText(
            f"Requested Z {self._target_z_mm:.9g} mm | "
            f"Calculating Z {calculating} | Displayed Z {displayed}")


    @Slot(float)
    def set_selected_z(self, z_mm):
        if self._closed or not self.follow_ray.isChecked():
            return
        if isinstance(z_mm, bool) or not isinstance(z_mm, Real):
            return
        z_mm = float(z_mm)
        if not math.isfinite(z_mm) or not self.target_z.minimum() <= z_mm <= self.target_z.maximum():
            return
        # Retain the exact linked float even if the spinbox display rounds it.
        if z_mm == self._target_z_mm:
            return
        self.target_z.blockSignals(True)
        self.target_z.setValue(z_mm)
        self.target_z.blockSignals(False)
        self._z_changed(z_mm)


    @Slot(float)
    def _z_changed(self, z_mm):
        if self._closed:
            return
        self.observation.cancel_reference_work()
        self._target_z_mm = float(z_mm)
        self._update_plane_positions()
        self._sync_z_slider()
        if self._stale or (not self._session_ready and self._worker is None):
            self.status.setText("Click Calculate beam to start a session at this Z."
                                + (" Previous optics remain displayed." if self.result is not None else ""))
            return
        cached = self._cache.get(float(z_mm))
        if cached is not None:
            self._pending_z = False
            self.timer.stop()
            self._cache.move_to_end(float(z_mm))
            self._display(*cached)
            return
        self._pending_z = True
        shown = (f" Showing completed Z {self.result.checkpoint.plane_z_mm:.9g} mm."
                 if self.result is not None else " First upstream calculation is still running.")
        self.status.setText(f"Exact Z {z_mm:.9g} mm requested.{shown}")
        # Throttle, never debounce or cancel on motion: a stream of input must
        # make progress. At most one running solve and one latest Z are owned.
        if self._worker is None and not self.timer.isActive():
            self.timer.start(self.QUERY_INTERVAL_MS)


    @Slot()
    def _start_query(self):
        if self._closed or self._stale or self._captured is None or self._worker is not None:
            return
        self.timer.stop()
        self._pending_z = False
        self._generation += 1
        request = replace(self._request, observation_z_mm=self._target_z_mm)
        retained = tuple(self._cache.values())+(self.result, self.preview)
        worker = _WaveWorker(self._generation, self._session, self._captured, request, Event(), retained,
                             observer=self._observation_session, pair_context=self._pair_context)
        worker.signals.solved.connect(self._solved)
        worker.signals.failed.connect(self._failed)
        worker.signals.progress.connect(self._progress)
        worker.signals.progress_detail.connect(self._progress_detail)
        worker.signals.finished.connect(self._finished)
        self._worker = worker
        self._progress_detail(self._generation, worker.progress_state)
        self._update_plane_positions()
        self.cancel_button.setEnabled(True)
        self.pool.start(worker)


    @Slot(int, int, object, object)
    def _solved(self, generation, session, result, preview):
        if self._closed or self._stale or generation != self._generation or session != self._session:
            return
        z_mm = float(result.checkpoint.plane_z_mm)
        submitted = (self._worker.request.observation_z_mm if self._worker is not None
                     else self._target_z_mm)
        if z_mm != submitted:
            self.status.setText("Rejected wave result: executed Z differs from the requested observation plane.")
            self._terminal_progress("failed", "Executed Z differs from the requested observation plane")
            return
        self._session_ready = True
        self._cache[z_mm] = (result, preview)
        self._cache.move_to_end(z_mm)
        if self._target_z_mm in self._cache:
            self._cache.move_to_end(self._target_z_mm)
            self._pending_z = False
        budget = self._request.execution.maximum_ram_cache_bytes
        while len(self._cache) > self.CACHE_LIMIT or (
                len(self._cache) > 1 and sum(value[1].retained_bytes for value in self._cache.values()) > budget):
            self._cache.popitem(last=False)
        # A requested cache hit has priority over the completion of an older
        # in-flight query. Otherwise show useful progress with its true Z.
        current = self._cache.get(self._target_z_mm)
        self._display(*(current if current is not None else (result, preview)))
        progress = getattr(self, "_progress_state", None)
        if progress is not None and progress.parent_id is not None:
            self._progress_state = replace(progress, pair_error=preview.comparison_error)
        self._terminal_progress("succeeded", "Accepted coherent result")


    def _display_comparison(self, preview):
        context = self._pair_context
        if context is None or preview.pair_token != context.token:
            self.comparison_table.hide()
            return
        if preview.comparison is None:
            self.comparison_table.hide()
            self.comparison_status.setText(f"Pair {context.token[:8]} | comparison unavailable: {preview.comparison_error or 'no matched data'}")
            return
        values = preview.comparison
        particle, wave = values["particle"], values["wave"]
        self.comparison_status.setText(
            f"Pair {context.token[:8]} | same captured inputs {context.physical_identity[:12]} | "
            f"compared Z {wave['z_mm']:.9g} mm | laboratory X/Y. "
            "Interference can change wave intensity and width; agreement is not assumed. "
            "Wave energy is the propagated mode reference. Phase belongs to individual wave modes only.")
        self.comparison_status.setToolTip(
            f"Particle energy: {particle['energy_definition']}\nWave energy: {wave['energy_definition']}\n"
            "Statistics use the complete retained populations, not the plotted points or image viewport.")
        rows = (("Mean energy (eV)", "mean_energy_ev", 1.),
                ("Energy RMS (eV)", "rms_energy_ev", 1.),
                ("Current (pA)", "current_a", 1e12),
                ("Emitted reference fraction (%)", "source_fraction", 100.),
                ("Centre X, Y (µm)", "centroid_xy_m", 1e6),
                ("RMS X, Y (µm)", "rms_xy_m", 1e6),
                ("Radial RMS (µm)", "radial_rms_m", 1e6))
        self.comparison_table.setRowCount(len(rows))
        for row, (label, key, scale) in enumerate(rows):
            self.comparison_table.setItem(row, 0, QTableWidgetItem(label))
            for col, data in ((1, particle), (2, wave)):
                value = data[key]
                text = ("Unavailable" if value is None else
                        ", ".join(f"{float(v)*scale:.6g}" for v in value)
                        if isinstance(value, (tuple, list)) else f"{float(value)*scale:.6g}")
                self.comparison_table.setItem(row, col, QTableWidgetItem(text))
        self.comparison_table.show()


    def _refresh_intensity(self, *_args):
        if self.preview is None:
            return
        self.observation_changed.emit(self.result, self.preview,
                                      self.intensity_scale.currentIndex())


    @Slot(int, str)
    def _failed(self, generation, message):
        if self._closed or generation != self._generation:
            return
        self._terminal_progress("failed", str(message))
        # The cursor may already request a newer plane. Attribute this error
        # to the submitted worker, never to the latest requested position.
        failed_plane = (f" at Z {self._worker.request.observation_z_mm:.9g} mm"
                        if self._worker is not None else "")
        retained = (f" Previous complete plane Z {self.result.checkpoint.plane_z_mm:.9g} mm remains displayed."
                    if self.result is not None else "")
        self.status.setText(f"Coherent beam calculation failed{failed_plane}: {message}. "
                            f"No new image was produced.{retained}")
        self.observation.mark_previous_calculation(
            "Previous calculation · the requested calculation failed. No new phase plate was produced.")


    @Slot(int, str)
    def _progress(self, generation, message):
        if not self._closed and generation == self._generation:
            shown = (f" | Displayed Z {self.result.checkpoint.plane_z_mm:.9g} mm"
                     if self.result is not None else "")
            self.status.setText(f"Requested Z {self._target_z_mm:.9g} mm{shown} | {message}")


    @Slot(int, object)
    def _progress_detail(self, generation, update):
        if not self._closed and generation == self._generation:
            previous = getattr(self, "_progress_state", None)
            if previous is not None and previous.operation_id == update.operation_id and not previous.active:
                return
            self._progress_state = update
            self.progress_detail.emit(update)


    def _terminal_progress(self, status, message):
        previous = getattr(self, "_progress_state", None)
        if previous is not None and previous.active:
            self._progress_state = previous.terminal(status, message)
            self.progress_detail.emit(self._progress_state)


    @Slot(int)
    def _finished(self, generation):
        if generation == self._generation:
            self._terminal_progress("cancelled", "Wave request ended without a new accepted result")
            self._worker = None
            self._update_plane_positions()
            self._update_cancel_enabled()
            if self._pending_z and not self._closed and not self._stale:
                # The prior task is allowed to commit reusable upstream work.
                # No queue of obsolete planes is submitted to the coordinator.
                self.timer.start(0)


    def set_projection_angle(self, angle):
        self.observation.set_projection_angle(angle)
