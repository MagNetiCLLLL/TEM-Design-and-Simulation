"""Explicit reference-plane steering; physical drives remain in the instrument."""
from __future__ import annotations

from threading import Event
from time import perf_counter

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot, QSignalBlocker
from PySide6.QtWidgets import (
    QFormLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeDoubleSpinBox
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.instrument_snapshot import capture_instrument_snapshot


def _text(value=""):
    result = QLabel(value)
    result.setWordWrap(True)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                   | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return result


class _Signals(QObject):
    ready = Signal(int, object, object, float)
    failed = Signal(int, str)
    finished = Signal(int)


class _SteeringWorker(QRunnable):
    def __init__(self, generation, snapshot, z_mm, *, calibrator=None,
                 existing_result=None, kind="tilt", target=(0., 0.)):
        super().__init__()
        self.generation = generation
        self.snapshot = snapshot
        self.z_mm = z_mm
        self.calibrator = calibrator
        self.existing_result = existing_result
        self.kind = kind
        self.target = target
        self.cancellation = Event()
        self.job_input_identity = snapshot.physical_digest
        self.resource_claim = ResourceClaim(2 * 1024**3, 512 * 1024**2)
        self.signals = _Signals()

    def run(self):
        started = perf_counter()
        try:
            from temsim.physics.beam_steering_calibration import (
                BeamSteeringCalibrator, capture_beam_steering_reference,
            )
            if self.cancellation.is_set():
                return
            calibrator = self.calibrator
            if calibrator is None:
                state = self.snapshot.restore()
                reference = capture_beam_steering_reference(
                    state, result=self.existing_result, target_z_mm=self.z_mm,
                    cancelled=self.cancellation.is_set,
                )
                calibrator = BeamSteeringCalibrator(state, reference, target_z_mm=self.z_mm)
                result = calibrator.calibrate(cancelled=self.cancellation.is_set)
            else:
                result = calibrator.solve(kind=self.kind, x=self.target[0], y=self.target[1],
                                          cancelled=self.cancellation.is_set)
            if not self.cancellation.is_set():
                self.signals.ready.emit(self.generation, calibrator, result, perf_counter()-started)
        except Exception as exc:
            if not self.cancellation.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class BeamSteeringControls(QWidget):
    """Solve off the GUI thread and commit only a still-current, verified result."""

    runtime_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("beamSteeringControls")
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._state = None
        self._revision = 0
        self._kind = "tilt"
        self._generation = 0
        self._worker = None
        self._calibrator = None
        self._calibration_identity = None
        self._retained_result = None
        self._plane_edited = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.addWidget(_text("Calibrated reference-plane steering"))
        self.scope = _text(
            "Calibrate the entrance pair around one recorded source electron. "
            "Tilt holds its position; shift holds its mechanical direction at the chosen Z. "
            "The finite beam centroid can respond differently. Internal corrector settings stay fixed."
        )
        layout.addWidget(self.scope)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.reference_z = WheelSafeDoubleSpinBox()
        self.reference_z.setObjectName("beamSteeringReferenceZ")
        self.reference_z.setRange(-1e6, 1e6)
        self.reference_z.setDecimals(6)
        self.reference_z.setSuffix(" mm")
        self.reference_z.setKeyboardTracking(False)
        form.addRow(_text("Reference Z"), self.reference_z)
        self.x = WheelSafeDoubleSpinBox()
        self.y = WheelSafeDoubleSpinBox()
        for name, editor in (("X", self.x), ("Y", self.y)):
            editor.setObjectName(f"beamSteeringChange{name}")
            editor.setRange(-1e6, 1e6)
            editor.setDecimals(6)
            editor.setKeyboardTracking(False)
            form.addRow(_text(f"Requested change {name}"), editor)
        layout.addLayout(form)
        self.target_note = _text("Changes are relative to the captured calibration state.")
        layout.addWidget(self.target_note)
        buttons = QHBoxLayout()
        self.calibrate_button = QPushButton("Calculate calibration")
        self.calibrate_button.setObjectName("calculateBeamSteeringCalibration")
        self.apply_button = QPushButton("Calculate and apply tilt")
        self.apply_button.setObjectName("calculateBeamSteeringChange")
        for button in (self.calibrate_button, self.apply_button):
            button.setProperty("calculationAction", True)
            buttons.addWidget(button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelBeamSteering")
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.status = _text("Click Calculate calibration for the current instrument.")
        self.status.setObjectName("beamSteeringStatus")
        layout.addWidget(self.status)
        self.reference_z.valueChanged.connect(self._plane_changed)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.apply_button.clicked.connect(self.calculate_and_apply)
        self.cancel_button.clicked.connect(self.cancel)
        self.set_kind("tilt")
        self._update_buttons()

    def set_kind(self, kind):
        if kind not in {"tilt", "shift"}:
            raise ValueError("Unknown beam steering task")
        if kind != self._kind:
            self.cancel()
            for editor in (self.x, self.y):
                editor.setValue(0.)
        self._kind = kind
        unit = " mrad" if kind == "tilt" else " nm"
        for editor in (self.x, self.y):
            editor.setSuffix(unit)
            editor.setSingleStep(.01 if kind == "tilt" else 1.)
        self.apply_button.setText(f"Calculate and apply {kind}")

    def set_state(self, state):
        if state is not self._state:
            self.invalidate("Instrument changed. Calculate calibration again.")
        self._state = state
        if state is not None and not self._plane_edited:
            with QSignalBlocker(self.reference_z):
                self.reference_z.setValue(float(state.sample.z_mm))
        self._update_buttons()

    def set_revision(self, revision):
        if int(revision) != self._revision:
            self._revision = int(revision)
            self.invalidate("Inputs changed. Calculate calibration again.")

    def publish_result(self, result):
        self._retained_result = result

    def _plane_changed(self):
        self._plane_edited = True
        self.invalidate("Reference plane changed. Calculate calibration again.")

    def invalidate(self, message):
        self.cancel()
        self._calibrator = None
        self._calibration_identity = None
        self.status.setText(message)
        self._update_buttons()

    def _update_buttons(self):
        busy = self._worker is not None
        self.calibrate_button.setEnabled(self._state is not None and not busy)
        self.apply_button.setEnabled(self._calibrator is not None and not busy)
        self.cancel_button.setEnabled(busy)
        self.reference_z.setEnabled(not busy)
        self.x.setEnabled(not busy)
        self.y.setEnabled(not busy)

    def _start(self, *, apply):
        if self._state is None or self._worker is not None:
            return
        try:
            snapshot = capture_instrument_snapshot(self._state)
            if apply and snapshot.physical_digest != self._calibration_identity:
                self.invalidate("Calibration is stale. Calculate calibration again.")
                return
            self._generation += 1
            scale = 1e-3 if self._kind == "tilt" else 1e-9
            worker = _SteeringWorker(
                self._generation, snapshot, self.reference_z.value(),
                calibrator=self._calibrator if apply else None,
                existing_result=None if apply else self._retained_result,
                kind=self._kind, target=(self.x.value()*scale, self.y.value()*scale),
            )
            worker.signals.ready.connect(self._ready)
            worker.signals.failed.connect(self._failed)
            worker.signals.finished.connect(self._finished)
            self._worker = worker
            self.status.setText("Calculating and checking the requested change…" if apply
                                else "Calculating the reference path and four-drive response…")
            self._update_buttons()
            self.pool.start(worker)
        except Exception as exc:
            self._worker = None
            self.status.setText(f"Steering calculation not started: {exc}")
            self._update_buttons()

    def calibrate(self):
        self._start(apply=False)

    def calculate_and_apply(self):
        self._start(apply=True)

    @Slot(int, object, object, float)
    def _ready(self, generation, calibrator, result, seconds):
        worker = self._worker
        if worker is None or generation != self._generation or worker.cancellation.is_set():
            return
        try:
            if capture_instrument_snapshot(self._state).physical_digest != worker.snapshot.physical_digest:
                self.invalidate("Inputs changed during calculation; no settings were applied.")
                return
            if worker.calibrator is None:
                self._calibrator = calibrator
                self._calibration_identity = worker.snapshot.physical_digest
                condition = float(result.condition_number)
                self.status.setText(
                    f"Calibration ready | condition {condition:.4g} | {result.backend} | {seconds:.2f} s\n"
                    f"{calibrator.reference.provenance}\n"
                    "Enter a relative change, then calculate and apply."
                )
            else:
                from temsim.physics.beam_steering_calibration import apply_beam_steering_solution
                apply_beam_steering_solution(self._state, result)
                residual = np.asarray(result.residual, dtype=float)
                message = (f"Applied {self._kind} | reference residual X/Y "
                           f"{residual[0]*1e9:.3g} / {residual[1]*1e9:.3g} nm; "
                           f"slope {residual[2]*1e6:.3g} / {residual[3]*1e6:.3g} µrad | "
                           f"{seconds:.2f} s. Recalibrate before another change.")
                self._calibrator = None
                self._calibration_identity = None
                self._worker = None
                # Existing application invalidation refreshes every other view.
                self.runtime_changed.emit("beam_deflector.calibrated_drives")
                self.status.setText(message)
        except Exception as exc:
            self._calibrator = None
            self._calibration_identity = None
            self.status.setText(f"Steering result not applied: {exc}")
        self._update_buttons()

    @Slot(int, str)
    def _failed(self, generation, message):
        if generation == self._generation and self._worker is not None:
            self.status.setText(f"Steering calculation failed: {message}")

    @Slot(int)
    def _finished(self, generation):
        if generation == self._generation:
            self._worker = None
            self._update_buttons()

    def cancel(self):
        if self._worker is not None:
            self._worker.cancellation.set()
            self._generation += 1
            self._worker = None
            self.pool.clear()
            self.status.setText("Cancelled; instrument settings were not changed.")
        self._update_buttons()

    def shutdown(self):
        self.cancel()
        return self.pool.waitForDone(3_000)
