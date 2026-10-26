"""Explicit ideal-reference computation and read-only, matched-mode phase views."""
from collections import OrderedDict
from dataclasses import asdict
from threading import Event

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QTransform
from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from temsim.cache_memory import retained_memory_inventory
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.immutable_json import json_digest


def _mode_rows(checkpoint):
    modes = checkpoint.beam.modes
    rows = getattr(modes, "rows", None)
    if rows is not None:
        return tuple((row["mode_id"], row["energy_kev"]) for row in rows)
    return tuple((mode.mode_id, mode.energy_kev) for mode in modes)


def _observation_key(actual):
    """Session-local cache identity; backend checks full immutable provenance."""
    return (str(actual.instrument_digest), json_digest(asdict(actual.request)),
            float(actual.checkpoint.plane_z_mm), getattr(actual, "resolved_scan_identity", None),
            id(actual.checkpoint), "ideal-lens-reference-v1")


class _Signals(QObject):
    ready = Signal(int, object, object)
    failed = Signal(int, str)
    progress = Signal(int, str)
    finished = Signal(int)


class _ReferenceWorker(QRunnable):
    def __init__(self, generation, key, state, actual, retained):
        super().__init__()
        from temsim.gui.coherent_beam import _wave_resource_claim
        self.generation, self.key, self.state, self.actual = generation, key, state, actual
        self.event = Event()
        self.signals = _Signals()
        self.resource_claim = _wave_resource_claim(state, actual.request)
        self.job_input_identity = f"ideal-wave-reference:{key}"
        self.existing_result = (state, actual, retained)

    def run(self):
        from temsim.gui.coherent_beam import _wave_read_session_bytes
        from temsim.physics.wave_checkpoint_store import wave_read_session
        from temsim.physics.tip_wave_pipeline import simulate_ideal_reference
        try:
            if self.event.is_set():
                return
            with wave_read_session(_wave_read_session_bytes(self.actual.request), cancelled=self.event.is_set):
                ideal = simulate_ideal_reference(self.state, self.actual, cancelled=self.event.is_set,
                    progress_callback=lambda done, total, text: self.signals.progress.emit(
                        self.generation, f"{text} ({done}/{total})") if not self.event.is_set() else None)
            if float(ideal.checkpoint.plane_z_mm) != self.key[2]:
                raise ValueError("Ideal reference returned a different observation plane")
            if not self.event.is_set():
                self.signals.ready.emit(self.generation, self.key, ideal)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
        finally:
            self.signals.finished.emit(self.generation)


class _ReadoutWorker(QRunnable):
    def __init__(self, generation, key, actual, ideal, mode_id, remove_piston, retained):
        super().__init__()
        self.generation, self.key, self.actual, self.ideal = generation, key, actual, ideal
        self.mode_id, self.remove_piston = mode_id, remove_piston
        self.event = Event()
        self.signals = _Signals()
        self.maximum_working_bytes = actual.request.maximum_readout_bytes
        self.resource_claim = ResourceClaim(self.maximum_working_bytes)
        self.job_input_identity = f"wavefront-readout:{key}:{mode_id}:{remove_piston}"
        self.existing_result = (actual, ideal, retained)

    def run(self):
        from temsim.physics.wavefront_comparison import compare_wavefronts
        try:
            if not self.event.is_set():
                data = compare_wavefronts(self.actual.checkpoint, self.ideal.checkpoint, self.mode_id,
                    maximum_working_bytes=self.maximum_working_bytes,
                    remove_piston=self.remove_piston, cancelled=self.event.is_set)
                if float(data.metadata["plane_z_mm"]) != self.key[2] or data.metadata["mode_id"] != self.mode_id:
                    raise ValueError("Comparison readout returned a different plane or mode")
                if not self.event.is_set():
                    self.signals.ready.emit(self.generation, self.key, data)
        except Exception as error:
            if not self.event.is_set():
                self.signals.failed.emit(self.generation, str(error))
        finally:
            self.signals.finished.emit(self.generation)


class WavefrontComparison(QWidget):
    """An explicit Compute action owns reference propagation; controls read it."""
    CACHE_LIMIT = 4

    def __init__(self, state_provider, parent=None):
        super().__init__(parent)
        self.state_provider = state_provider
        self.actual = self.ideal = self.data = None
        self._key = None
        self._cache = OrderedDict()
        self._generation = 0
        self._worker = None
        self._closed = False
        self._mode_id = None
        self._bounds = None
        self._angle_deg = 0.
        self.pool = CoordinatedPool(self)
        self.pool.coordinator.register_retained(self, "retained_roots")

        self.compute = QPushButton("Compute reference")
        self.compute.setObjectName("computeIdealWaveReference")
        self.compute.setToolTip("Run an additional wave calculation from the same captured source and optics, with lens aberrations removed.")
        self.cancel_button = QPushButton("Cancel reference")
        self.cancel_button.setObjectName("cancelIdealWaveReference")
        self.cancel_button.setEnabled(False)
        self.mode = QComboBox()
        self.mode.setObjectName("wavefrontComparisonMode")
        self.mode.setToolTip("Match the same physical mode ID in the actual and ideal reference. Independent modes have no aggregate phase.")
        self.remove_piston = QCheckBox("Remove constant phase offset")
        self.remove_piston.setObjectName("wavefrontRemovePiston")
        self.remove_piston.setToolTip("Subtract the fitted constant phase difference on the common valid support from the difference image only. Actual and ideal phases stay unchanged; no new propagation is run.")
        toolbar = QHBoxLayout()
        toolbar.addWidget(self.compute)
        toolbar.addWidget(self.cancel_button)
        toolbar.addWidget(QLabel("Mode"))
        toolbar.addWidget(self.mode, 1)
        toolbar.addWidget(self.remove_piston)
        self.scope = QLabel(
            "The reference retains focusing, rotation, quadrupoles, deflection and defocus (C1). "
            "It removes spherical and explicit intrinsic aberrations and nonlinear hexapole action. "
            "Each energy mode is compared separately; energy-spread chromatic effects are not removed.")
        self.scope.setWordWrap(True)
        self.scope.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.status = QLabel("Calculate the actual beam, then choose Compute reference.")
        self.status.setObjectName("wavefrontComparisonStatus")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid = QHBoxLayout()
        self.plots, self.images, self.colourbars = [], [], []
        for title in ("Actual", "Ideal lens reference", "Phase difference"):
            plot = pg.PlotWidget(background="#080e19")
            plot.setTitle(title)
            plot.setLabel("bottom", "U", units="µm")
            plot.setLabel("left", "V", units="µm")
            plot.getAxis("bottom").enableAutoSIPrefix(False)
            plot.getAxis("left").enableAutoSIPrefix(False)
            plot.setAspectLocked(True)
            image = pg.ImageItem(axisOrder="row-major")
            image.setAutoDownsample(False)
            plot.addItem(image)
            colourbar = pg.ColorBarItem(values=(-np.pi, np.pi),
                colorMap=pg.colormap.get("CET-C2"), interactive=False, label="Phase (rad)")
            colourbar.setImageItem(image, insert_in=plot.getPlotItem())
            if self.plots:
                plot.setXLink(self.plots[0])
                plot.setYLink(self.plots[0])
            grid.addWidget(plot, 1)
            self.plots.append(plot)
            self.images.append(image)
            self.colourbars.append(colourbar)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.scope)
        layout.addLayout(grid, 1)
        layout.addWidget(self.status)
        self.compute.clicked.connect(self.compute_reference)
        self.cancel_button.clicked.connect(self.cancel)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.remove_piston.toggled.connect(self._readout_changed)
        self.compute.setEnabled(False)

    def retained_roots(self):
        return self.actual, self.ideal, self.data, self._cache

    def set_observation(self, actual):
        if self._closed:
            return
        key = _observation_key(actual) if actual is not None else None
        if key == self._key:
            return
        self._cancel_work()
        self.actual, self._key = actual, key
        self.ideal = self._cache.get(key)
        self._clear_views()
        self.mode.blockSignals(True)
        self.mode.clear()
        if actual is not None:
            for mode_id, energy in _mode_rows(actual.checkpoint):
                self.mode.addItem(f"{mode_id} | {energy:.7g} keV", mode_id)
        index = self.mode.findData(self._mode_id)
        self.mode.setCurrentIndex(index if index >= 0 else 0)
        self.mode.blockSignals(False)
        self._mode_id = self.mode.currentData()
        self.mode.setEnabled(self.mode.count() > 1)
        self.compute.setEnabled(actual is not None)
        if self.ideal is not None:
            self._cache.move_to_end(key)
            self._start_readout()
        elif actual is not None:
            self.status.setText(f"Actual plane Z {key[2]:.9g} mm. Compute reference to compare this captured plane.")
        else:
            self.status.setText("Calculate the actual beam before computing an ideal reference.")

    def invalidate(self, message):
        self._cancel_work()
        self.actual = self.ideal = self._key = None
        self._cache.clear()
        self._clear_views()
        self.compute.setEnabled(False)
        self.status.setText(message)

    def _clear_views(self):
        self.data = None
        self._bounds = None
        for image in self.images:
            image.clear()

    def _cancel_work(self):
        self._generation += 1
        if self._worker is not None:
            self._worker.event.set()
        self.pool.clear()
        self._worker = None
        self.cancel_button.setEnabled(False)
        self.compute.setEnabled(self.actual is not None)

    def cancel(self):
        self._cancel_work()
        self.status.setText("Reference operation cancelled. Actual beam and any completed comparison are retained.")

    def _start(self, worker):
        self._worker = worker
        worker.signals.failed.connect(self._failed)
        worker.signals.progress.connect(self._progress)
        worker.signals.finished.connect(self._finished)
        self.cancel_button.setEnabled(True)
        self.compute.setEnabled(False)
        self.pool.start(worker)

    def compute_reference(self):
        if self._closed or self.actual is None:
            return
        self._cancel_work()
        if self.ideal is not None:
            self._start_readout()
            return
        try:
            from temsim.physics.compute_backend import require_coherent_cuda
            require_coherent_cuda()
            state = self.state_provider(self.actual)
            if state is None:
                raise ValueError("The captured inputs for this executed plane are unavailable; calculate the actual beam again")
            worker = _ReferenceWorker(self._generation, self._key, state, self.actual, self.retained_roots())
            worker.signals.ready.connect(self._reference_ready)
            self.status.setText(f"Computing ideal lens reference at exact Z {self._key[2]:.9g} mm…")
            self._start(worker)
        except Exception as error:
            self.status.setText(f"Reference not started: {error}")

    def _reference_ready(self, generation, key, ideal):
        if self._closed or generation != self._generation or key != self._key:
            return
        self.ideal = ideal
        self._cache[key] = ideal
        self._cache.move_to_end(key)
        budget = self.actual.request.execution.maximum_ram_cache_bytes
        while len(self._cache) > 1 and (len(self._cache) > self.CACHE_LIMIT
                or sum(retained_memory_inventory(self._cache).values()) > budget):
            self._cache.popitem(last=False)
        # Its finished signal starts readout after the reference has returned;
        # do not mark a successfully completed propagation as cancelled.

    def _mode_changed(self):
        self._mode_id = self.mode.currentData()
        self._readout_changed()

    def _readout_changed(self):
        if isinstance(self._worker, _ReferenceWorker):
            return  # Finish the same reference, then read the newest selection.
        if self.ideal is not None:
            self._start_readout()

    def _start_readout(self):
        if self._closed or self.actual is None or self.ideal is None or self._mode_id is None:
            return
        self._cancel_work()
        # A different mode or phase convention must never leave the previous
        # image under the newly selected controls, including after a failure.
        self._clear_views()
        worker = _ReadoutWorker(self._generation, self._key, self.actual, self.ideal,
            self._mode_id, self.remove_piston.isChecked(), self.retained_roots())
        worker.signals.ready.connect(self._readout_ready)
        self.status.setText(f"Reading matched mode {self._mode_id} at Z {self._key[2]:.9g} mm…")
        self._start(worker)

    def _readout_ready(self, generation, key, data):
        if self._closed or generation != self._generation or key != self._key:
            return
        self.data = data
        self._draw()

    def _draw(self):
        if self.data is None:
            return
        data = self.data
        angle = np.deg2rad(self._angle_deg)
        rotation = np.array(((np.cos(angle), np.sin(angle)), (-np.sin(angle), np.cos(angle))))
        basis = rotation @ data.basis_m * 1e6
        ny, nx = data.actual_phase_rad.shape
        origin = rotation @ data.origin_m * 1e6 - basis @ np.array((nx//2+.5, ny//2+.5))
        transform = QTransform(basis[0, 0], basis[1, 0], basis[0, 1], basis[1, 1], *origin)
        for image, values in zip(self.images, (data.actual_phase_rad, data.ideal_phase_rad, data.difference_rad)):
            image.setImage(values, autoLevels=False, levels=(-np.pi, np.pi))
            image.setTransform(transform)
        corners = origin[:, None] + basis @ np.array(((0, nx, nx, 0), (0, 0, ny, ny)))
        self._bounds = np.stack((corners.min(axis=1), corners.max(axis=1)), axis=1)
        self.fit_view()
        removed = (f"Constant phase offset removed from difference only ({data.metadata['piston_phase_rad']:.6g} rad)"
                   if data.metadata["piston_removed"] else "Common source phase retained")
        difference = "Actual − Ideal − constant offset" if data.metadata["piston_removed"] else "Actual − Ideal"
        self.status.setText(f"Exact Z {data.metadata['plane_z_mm']:.9g} mm | mode {data.metadata['mode_id']} | "
            f"common valid area {100*data.metadata['valid_fraction']:.4g}% | {removed}. "
            f"Difference is {difference}, wrapped to [−π, π]; blank pixels have no valid comparison.")

    def set_projection_angle(self, angle):
        self._angle_deg = float(angle)
        self._draw()

    def fit_view(self):
        if self._bounds is not None:
            self.plots[0].setRange(xRange=self._bounds[0], yRange=self._bounds[1], padding=.02)

    def _failed(self, generation, message):
        if not self._closed and generation == self._generation:
            self.status.setText(f"Wavefront comparison unavailable: {message}. Actual beam is retained.")

    def _progress(self, generation, message):
        if not self._closed and generation == self._generation:
            self.status.setText(f"Ideal reference Z {self._key[2]:.9g} mm | {message}")

    def _finished(self, generation):
        if generation == self._generation:
            reference_finished = isinstance(self._worker, _ReferenceWorker)
            self._worker = None
            self.cancel_button.setEnabled(False)
            self.compute.setEnabled(self.actual is not None)
            if reference_finished and self.ideal is not None:
                self._start_readout()

    def shutdown(self, msecs=3000):
        self._closed = True
        self._cancel_work()
        return self.pool.waitForDone(msecs)
