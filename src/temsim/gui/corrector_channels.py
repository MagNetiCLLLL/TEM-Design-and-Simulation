"""Explicit model-calibrated aberration channels over existing physical drives."""
from __future__ import annotations

from threading import Event
from time import perf_counter

import numpy as np

from PySide6.QtCore import QObject, QRunnable, Qt, Signal, Slot
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.instrument_snapshot import capture_instrument_snapshot


def _text(value=""):
    label = QLabel(value)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                  | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return label


class _Signals(QObject):
    ready = Signal(int, object, float)
    failed = Signal(int, str)
    finished = Signal(int)


class _ChannelWorker(QRunnable):
    def __init__(self, generation, snapshot, system, channel, alpha, step,
                 *, calibration=None, delta=(0., 0.)):
        super().__init__()
        self.generation, self.snapshot = generation, snapshot
        self.system, self.channel = system, channel
        self.alpha, self.step = alpha, step
        self.calibration, self.delta = calibration, delta
        self.cancellation = Event()
        self.signals = _Signals()
        self.resource_claim = ResourceClaim(2 * 1024**3, 512 * 1024**2)
        self.job_input_identity = snapshot.physical_digest

    def run(self):
        started = perf_counter()
        try:
            from temsim.optics.corrector_channels import calibrate_channel, prepare_channel_change
            if self.cancellation.is_set():
                return
            state = self.snapshot.restore()
            if self.calibration is None:
                result = calibrate_channel(state, self.system, self.channel,
                    semiangle_mrad=self.alpha, maximum_step_mm=self.step,
                    cancelled=self.cancellation.is_set)
            else:
                result = prepare_channel_change(state, self.calibration, self.delta,
                                                cancelled=self.cancellation.is_set)
            if not self.cancellation.is_set():
                self.signals.ready.emit(self.generation, result, perf_counter()-started)
        except Exception as exc:
            if not self.cancellation.is_set():
                self.signals.failed.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class CorrectorChannelControls(QWidget):
    """Calibrate and verify off-thread; commit to an unchanged live instrument."""

    runtime_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("correctorChannelControls")
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._state = None
        self._revision = self._generation = 0
        self._worker = self._calibration = self._identity = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.addWidget(_text("A1 / A2 coarse and fine channels"))
        layout.addWidget(_text(
            "Calibrate the current optical model, then apply a relative aberration change. "
            "Coarse and fine use different physical drives. Field length and propagation "
            "remain active; coupled changes in spherical aberration are measured. "
            "This is a simulator calibration, not a manufacturer's channel map."
        ))
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.system = WheelSafeComboBox()
        self.system.setObjectName("correctorChannelSystem")
        self.system.addItem("Probe corrector", "probe")
        self.system.addItem("Image corrector / B-COR", "image")
        self.channel = WheelSafeComboBox()
        self.channel.setObjectName("correctorChannelKind")
        for key, label in (("a1_coarse", "A1 coarse"), ("a1_fine", "A1 fine"),
                           ("a2_coarse", "A2 coarse"), ("a2_fine", "A2 fine")):
            self.channel.addItem(label, key)
        self.alpha = WheelSafeDoubleSpinBox()
        self.alpha.setObjectName("correctorChannelPupil")
        self.alpha.setRange(.01, 100.)
        self.alpha.setDecimals(3)
        self.alpha.setValue(.5)
        self.alpha.setSuffix(" mrad")
        self.step = WheelSafeDoubleSpinBox()
        self.step.setObjectName("correctorChannelAxialStep")
        self.step.setRange(.0001, 1.)
        self.step.setDecimals(4)
        self.step.setValue(.05)
        self.step.setSuffix(" mm")
        self.x, self.y = WheelSafeDoubleSpinBox(), WheelSafeDoubleSpinBox()
        for name, editor in (("Normal", self.x), ("Skew", self.y)):
            editor.setObjectName("correctorChannelDelta" + name)
            editor.setRange(-1e6, 1e6)
            editor.setDecimals(6)
            editor.setSingleStep(1.)
            editor.setSuffix(" nm")
            editor.setToolTip("Signed Cartesian aberration-coefficient change in the laboratory X/Y basis.")
        for editor in (self.alpha, self.step, self.x, self.y):
            editor.setKeyboardTracking(False)
        for label, editor in (("System", self.system), ("Channel", self.channel),
                              ("Calibration pupil", self.alpha), ("Maximum axial step", self.step),
                              ("Requested normal change", self.x), ("Requested skew change", self.y)):
            form.addRow(_text(label), editor)
        layout.addLayout(form)
        buttons = QVBoxLayout()
        self.calibrate_button = QPushButton("Calculate response")
        self.calibrate_button.setObjectName("calculateCorrectorChannelResponse")
        self.apply_button = QPushButton("Calculate and apply change")
        self.apply_button.setObjectName("applyCorrectorChannelChange")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("cancelCorrectorChannel")
        for button in (self.calibrate_button, self.apply_button):
            button.setProperty("calculationAction", True)
        for button in (self.calibrate_button, self.apply_button, self.cancel_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = _text("Choose a channel and calculate its response at the current settings.")
        self.status.setObjectName("correctorChannelStatus")
        self.measured = _text()
        self.measured.setObjectName("correctorChannelMeasuredChange")
        layout.addWidget(self.status)
        layout.addWidget(self.measured)
        for editor in (self.system, self.channel):
            editor.currentIndexChanged.connect(self._options_changed)
        for editor in (self.alpha, self.step):
            editor.valueChanged.connect(self._options_changed)
        self.calibrate_button.clicked.connect(self.calibrate)
        self.apply_button.clicked.connect(self.calculate_and_apply)
        self.cancel_button.clicked.connect(self.cancel)
        self._update_buttons()

    def _options_changed(self, *_):
        self.invalidate("Channel or calibration settings changed. Calculate response again.")

    def set_state(self, state):
        if state is not self._state:
            self.invalidate("Instrument changed. Calculate response for the current settings.")
        self._state = state
        self._update_buttons()

    def set_revision(self, revision):
        if int(revision) != self._revision:
            self._revision = int(revision)
            self.invalidate("Instrument settings changed. Calculate response again.")

    def invalidate(self, message):
        self.cancel()
        self._calibration = self._identity = None
        self.measured.clear()
        self.status.setText(message)
        self._update_buttons()

    def _update_buttons(self):
        busy = self._worker is not None
        installed = self._state is not None and bool(getattr(
            self._state, f"{self.system.currentData()}_corrector_installed", False))
        self.calibrate_button.setEnabled(installed and not busy)
        self.apply_button.setEnabled(installed and self._calibration is not None and not busy)
        self.cancel_button.setEnabled(busy)
        for editor in (self.system, self.channel, self.alpha, self.step, self.x, self.y):
            editor.setEnabled(not busy)
        self.calibrate_button.setToolTip("" if installed else "Install the selected corrector first.")

    def calibrate(self):
        self._start(apply=False)

    def calculate_and_apply(self):
        self._start(apply=True)

    def _start(self, *, apply):
        if self._state is None or self._worker is not None:
            return
        try:
            snapshot = capture_instrument_snapshot(self._state)
            if apply and (self._calibration is None or snapshot.physical_digest != self._identity):
                self.invalidate("Response is stale. Calculate response again.")
                return
            self._generation += 1
            worker = _ChannelWorker(self._generation, snapshot,
                self.system.currentData(), self.channel.currentData(), self.alpha.value(), self.step.value(),
                calibration=self._calibration if apply else None,
                delta=(self.x.value()*1e-6, self.y.value()*1e-6))  # nm -> mm
            worker.signals.ready.connect(self._ready)
            worker.signals.failed.connect(self._failed)
            worker.signals.finished.connect(self._finished)
            self._worker = worker
            self.measured.clear()
            self.status.setText("Propagating and checking the requested change…" if apply else
                                "Measuring positive and negative physical-drive responses…")
            self._update_buttons()
            self.pool.start(worker)
        except Exception as exc:
            self._worker = None
            self.status.setText("Channel calculation could not start: " + str(exc))
            self._update_buttons()

    @Slot(int, object, float)
    def _ready(self, generation, result, seconds):
        worker = self._worker
        if worker is None or generation != self._generation or worker.cancellation.is_set():
            return
        try:
            if capture_instrument_snapshot(self._state).physical_digest != worker.snapshot.physical_digest:
                self.invalidate("Inputs changed during calculation; no settings were applied.")
                return
            if worker.calibration is None:
                self._calibration = result
                self._identity = worker.snapshot.physical_digest
                self.status.setText(f"Response ready · {seconds:.2f} s. Enter a relative normal/skew change.")
                self.measured.setText(_coefficient_text(result.baseline, "Measured baseline")
                                      + "\n" + _calibration_text(result))
            else:
                from temsim.optics.corrector_channels import apply_channel_change
                apply_channel_change(self._state, result, cancelled=worker.cancellation.is_set)
                achieved = result.achieved_delta_xy_mm
                message = (f"Applied {worker.channel.replace('_', ' ')} · measured normal/skew change "
                           f"{achieved[0]*1e6:.6g} / {achieved[1]*1e6:.6g} nm · {seconds:.2f} s. "
                           "Calculate a new response before the next change.")
                measured = (_coefficient_text(result.before, "Before") + "\n"
                            + _coefficient_text(result.after, "After") + "\n"
                            + "Before/after use the same captured reference. "
                              "Recalculate the coherent beam to inspect the updated phase at any supported section. "
                              "Tilt, defocus and coupled aberrations remain in the propagated wave.")
                self._worker = None  # Synchronous invalidation must not cancel an already committed result.
                self._generation += 1
                self._calibration = self._identity = None
                self.runtime_changed.emit("corrector_channels.calibrated_drives")
                self.status.setText(message)
                self.measured.setText(measured)
        except Exception as exc:
            self.status.setText("Channel change was not applied: " + str(exc))
        self._update_buttons()

    @Slot(int, str)
    def _failed(self, generation, message):
        if generation == self._generation:
            self.status.setText("Channel calculation unavailable: " + message)

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


def _coefficient_text(report, prefix):
    c = report.coefficients
    text = (f"{prefix}: A1 {c.a1_mm*1e6:.6g} nm at {c.a1_azimuth_deg:.5g}°; "
            f"A2 {c.a2_mm*1e6:.6g} nm at {c.a2_azimuth_deg:.5g}°; "
            f"C3 {c.c3_mm:.8g} mm; C1 {c.c1_mm*1e6:.6g} nm")
    holdout = getattr(report, "diagnostics", {}).get("holdout_rms_m")
    if holdout is not None:
        text += f"; pupil-fit check RMS {float(holdout)*1e9:.5g} nm"
    return text


def _calibration_text(calibration):
    inverse = np.linalg.pinv(calibration.jacobian * calibration.steps)
    limits_nm = .5 / np.max(np.abs(inverse), axis=0) * 1e6
    drives = ", ".join(dict.fromkeys(spec.key for spec in calibration.control_specs))
    return (f"Physical drives: {drives}\n"
            f"Single-axis local limits: normal ±{limits_nm[0]:.6g} nm; skew ±{limits_nm[1]:.6g} nm. "
            "Combined changes must also stay inside the calibrated range. "
            "Each candidate is checked by propagation before application. "
            "A new calibration establishes a new nominal reference; compare coefficient changes within one calibration.")
