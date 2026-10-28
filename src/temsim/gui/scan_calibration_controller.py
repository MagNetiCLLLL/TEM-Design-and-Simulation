"""Cancellable scan calibration on captured inputs, with guarded drive commits."""
from copy import deepcopy
from threading import Event

from PySide6.QtCore import QObject, QRunnable, Signal

from temsim import input_io
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument
from temsim.physics.scan_geometry import calibrate_scan_system, recalibrate_descan_only
from temsim.physics.scan_preparation import copy_resolved_scan_drives


class _Signals(QObject):
    result = Signal(int, object)
    error = Signal(int, str)
    progress = Signal(int, str)
    finished = Signal(int)


class _Worker(QRunnable):
    def __init__(self, generation, snapshot, operation):
        super().__init__()
        self.generation, self.snapshot, self.operation = generation, snapshot, operation
        self._archive_inputs = snapshot.graph.get("archived_inputs")
        self.cancel_event = Event()
        self.signals = _Signals()
        self.job_input_identity = json_digest((snapshot.digest, operation))

    @input_io.using_state_inputs
    def run(self):
        try:
            if self.cancel_event.is_set():
                return
            state = self.snapshot.restore()
            state._tuning_cancelled = self.cancel_event.is_set
            from temsim.physics.transport_progress import transport_progress
            with transport_progress(lambda label: self.signals.progress.emit(self.generation, str(label))):
                if self.operation == "hold":
                    calibrate_scan_system(state, force=True, hold=True)
                elif self.operation == "descan":
                    recalibrate_descan_only(state)
                elif self.operation == "restore_ac":
                    from temsim.physics.scan_calibration import restore_held
                    restore_held(state, restore_descan=False)
                    state.descan_deflector.validate()
                else:
                    calibrate_scan_system(state)
            if not self.cancel_event.is_set():
                del state._tuning_cancelled
                self.snapshot.verify_current_inputs(state)
                self.signals.result.emit(self.generation, state)
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.error.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class ScanCalibrationController(QObject):
    applied = Signal(str)
    failed = Signal(str)
    discarded = Signal(str)
    progress = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        from temsim.gui.job_coordinator import CoordinatedPool
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0
        self._request = None
        self._closed = False
        # QObject deletion can bypass closeEvent (for example a removed page).
        # The coordinator owns running workers until their safe cancellation boundary.
        pool, coordinator = self.pool, self.pool.coordinator
        self.destroyed.connect(lambda *_: coordinator.clear(pool))

    @property
    def busy(self):
        return self._request is not None

    def _clear_request(self):
        was_busy = self.busy
        self._request = None
        if was_busy:
            self.busy_changed.emit(False)

    def cancel(self):
        self._generation += 1
        self.pool.clear()
        self._clear_request()

    def invalidate_if_changed(self, state):
        if self._request is None:
            return
        original, snapshot, _operation = self._request
        try:
            unchanged = state is original and json_digest(encode_instrument(state)) == json_digest(snapshot.graph)
            if unchanged:
                snapshot.verify_current_inputs(state)
        except (ValueError, TypeError, OSError):
            unchanged = False
        if not unchanged:
            self.cancel()
            self.discarded.emit("Inputs changed; the previous scan calibration was discarded.")

    def submit(self, state, operation="automatic"):
        if self._closed:
            raise ValueError("Scan calibration is shutting down")
        if operation not in {"automatic", "hold", "descan", "restore_ac"}:
            raise ValueError("Unknown scan calibration operation")
        self.cancel()
        snapshot = capture_instrument_snapshot(state)
        self._request = state, snapshot, operation
        worker = _Worker(self._generation, snapshot, operation)
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.progress.connect(self._accept_progress)
        worker.signals.finished.connect(self._accept_finished)
        self.busy_changed.emit(True)
        try:
            self.pool.start(worker)
        except Exception:
            self.cancel()
            raise

    def _accept_result(self, generation, candidate):
        if self._closed or generation != self._generation or self._request is None:
            return
        state, snapshot, operation = self._request
        try:
            unchanged = json_digest(encode_instrument(state)) == json_digest(snapshot.graph)
            if unchanged:
                snapshot.verify_current_inputs(state)
        except (ValueError, TypeError, OSError):
            unchanged = False
        if not unchanged:
            self._clear_request()
            self.discarded.emit("Inputs changed; the previous scan calibration was discarded.")
            return
        components = state.ac_deflector, state.descan_deflector
        before = [deepcopy({key: value for key, value in vars(part).items()
                            if key != "_physical_host"}) for part in components]
        try:
            # Never replace the state or a shared physical coil host. Install only
            # calibration-owned values, after comparing every captured input.
            copy_resolved_scan_drives(candidate, state)
            state.ac_deflector.calibration_mode = candidate.ac_deflector.calibration_mode
            state.ac_deflector.calibration_record_json = candidate.ac_deflector.calibration_record_json
        except Exception as exc:
            for part, saved in zip(components, before):
                host = vars(part).get("_physical_host")
                vars(part).clear()
                vars(part).update(saved)
                if host is not None:
                    object.__setattr__(part, "_physical_host", host)
            self._accept_error(generation, str(exc))
            return
        self._clear_request()
        self.applied.emit(operation)

    def _accept_error(self, generation, message):
        if not self._closed and generation == self._generation and self._request is not None:
            self._clear_request()
            self.failed.emit(message)

    def _accept_progress(self, generation, message):
        if not self._closed and generation == self._generation and self.busy:
            self.progress.emit(message)

    def _accept_finished(self, generation):
        if generation == self._generation and self._request is not None:
            self._clear_request()
            if not self._closed:
                self.discarded.emit("Scan calibration cancelled; previous drives retained.")

    def shutdown(self, timeout_ms=3000):
        self._closed = True
        try:
            self.cancel()
            return self.pool.waitForDone(timeout_ms)
        finally:
            # A parent window may ignore close while another owner is stopping.
            # Generation cancellation already prevents late result installation.
            self._closed = False
