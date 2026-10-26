"""Owned, cancellable CAD-kernel jobs; a result cannot impersonate its input."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal, Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QVBoxLayout

from temsim.cad_runtime import cad_interpreter, cad_process_root, setup_instructions


def request_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


class CadGeometryJob(QObject):
    """Run one immutable request in the separate CadQuery interpreter.

    The caller additionally owns the draft identity check: this class establishes
    which request produced the receipt, not whether that draft is still current.
    Cancellation and failures never emit a usable geometry result.
    """
    completed = Signal(bool, object, str)
    output = Signal(str)

    def __init__(self, request, parent=None, *, project_root=None, interpreter=None, worker=None):
        super().__init__(parent)
        self.project_root = Path(project_root or cad_process_root()).resolve()
        self.interpreter = cad_interpreter(self.project_root, interpreter)
        self.worker = Path(worker or Path(__file__).resolve().parents[1] / "cad_geometry_worker.py")
        self.request = deepcopy(request)
        self.request.setdefault("request_id", str(uuid4()))
        self.request_sha256 = request_digest(self.request)
        self.busy = False
        self.cancelled = False
        self.finished = False
        self.receipt = None
        self.log_text = ""
        self._temporary = None
        self._result_path = None
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_owned_process)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_for_exit)
        if parent is not None:
            parent.destroyed.connect(self._stop_for_exit)

    def start(self):
        if self.busy or self.finished:
            raise RuntimeError("This CAD operation has already been started")
        if not self.interpreter.is_file():
            raise FileNotFoundError("The optional STEP geometry engine is not installed.\n\n"
                                    + setup_instructions(self.project_root, self.interpreter))
        if not self.worker.is_file():
            raise FileNotFoundError("The STEP geometry worker is missing from this installation")
        self._temporary = tempfile.TemporaryDirectory(prefix="temsim-cad-geometry-")
        directory = Path(self._temporary.name)
        input_path = directory / "request.json"
        input_path.write_text(json.dumps(self.request, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        result_dir = directory / "result"
        self._result_path = result_dir / "result.json"
        environment = QProcessEnvironment.systemEnvironment()
        for key in ("PYTHONPATH", "PYTHONHOME"):
            environment.remove(key)
        environment.insert("PYTHONNOUSERSITE", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.project_root))
        self.busy = True
        self.process.start(str(self.interpreter), [str(self.worker), str(input_path), "--output", str(result_dir)])

    def _read_output(self):
        value = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if value:
            self.log_text = (self.log_text + value)[-64000:]
            self.output.emit(value)

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._finish(False, None, "STEP geometry engine could not start: " + self.process.errorString())

    def _process_finished(self, exit_code, exit_status):
        self._read_output()
        if self.cancelled:
            self._finish(False, None, "CAD operation cancelled. The previous draft and geometry are unchanged.")
            return
        try:
            receipt = json.loads(self._result_path.read_text(encoding="utf-8"))
            self.receipt = receipt
            if not isinstance(receipt, dict) or receipt.get("schema") != "temsim-cad-worker-result-v1":
                raise ValueError("The CAD engine returned an unsupported result")
            if (receipt.get("request_sha256") != self.request_sha256
                    or receipt.get("request_id") != self.request["request_id"]
                    or receipt.get("draft_digest") != self.request.get("draft_digest")):
                raise ValueError("The CAD result does not belong to the captured request; it was not applied")
            if exit_code != 0 or exit_status != QProcess.ExitStatus.NormalExit or receipt.get("status") != "complete":
                error = receipt.get("error")
                if isinstance(error, dict):
                    error = error.get("message") or error.get("type")
                raise ValueError(str(error or "CAD geometry evaluation did not complete"))
            if not isinstance(receipt.get("result"), dict):
                raise ValueError("The CAD engine did not return geometry")
            self._finish(True, receipt, "CAD geometry evaluation complete")
        except Exception as exc:
            self._finish(False, None, "CAD geometry was not changed: " + str(exc))

    def _finish(self, success, result, message):
        if self.finished:
            return
        self.finished = True
        self.busy = False
        self._kill_timer.stop()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.completed.emit(bool(success), result, message)

    def cancel(self):
        if not self.busy:
            return
        self.cancelled = True
        self.process.terminate()
        self._kill_timer.start(1500)

    def _kill_owned_process(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _stop_for_exit(self):
        if self.busy:
            self.cancelled = True
            self.process.terminate()
            if not self.process.waitForFinished(500):
                self.process.kill()
                self.process.waitForFinished(500)


class CadGeometryProgress(QDialog):
    """A modeless cancel/status surface for an owned kernel job."""
    def __init__(self, job, title, parent=None):
        super().__init__(parent)
        self.job = job
        self._close_pending = False
        self.setWindowTitle(title)
        self.resize(620, 320)
        self.setModal(False)
        layout = QVBoxLayout(self)
        self.status = QLabel("Processing in the background. The current geometry remains available.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.action = QPushButton("Cancel operation")
        self.action.clicked.connect(self.reject)
        for widget in (self.status, self.progress, self.log, self.action):
            layout.addWidget(widget)
        job.output.connect(self._output)
        job.completed.connect(self._finished)

    def _output(self, value):
        self.log.moveCursor(self.log.textCursor().MoveOperation.End)
        self.log.insertPlainText(value)

    def _finished(self, success, _result, message):
        self.progress.setRange(0, 1)
        self.progress.setValue(int(success))
        self.status.setText(message)
        self.action.setText("Close")
        self.action.setEnabled(True)
        if success or self._close_pending:
            super().reject()

    def reject(self):
        if self.job.busy:
            self._close_pending = True
            self.action.setEnabled(False)
            self.status.setText("Stopping this CAD operation…")
            self.job.cancel()
        else:
            super().reject()

    def closeEvent(self, event):
        if self.job.busy:
            self.reject()
            event.ignore()
        else:
            super().closeEvent(event)
