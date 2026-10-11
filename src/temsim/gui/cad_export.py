"""Nonblocking STEP export through the optional isolated CAD interpreter."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout


def setup_instructions():
    return ("Install the optional STEP exporter from the project folder:\n"
            "python -m venv environment/cad\n"
            "environment/cad/Scripts/python.exe -m pip install -r requirements-cad.txt\n"
            "This uses a separate environment from the simulator.")


def draft_catalog(document, *, source, runtime_values=None, aperture_indices=None):
    """Freeze the current editor draft and visible operating openings."""
    from temsim.cad_catalog import module_catalog
    module = module_catalog(deepcopy(document), source=str(source),
                            runtime_values=deepcopy(runtime_values or {}),
                            aperture_indices=deepcopy(aperture_indices or {}))
    if not module["solids"]:
        raise ValueError("This module has no closed material solids to export.")
    return {"schema": "temsim-analytic-material-cad-v1", "units": "mm", "modules": [module],
            "scope": "Current editor draft and displayed operating openings. Reconstructed material geometry; no OEM certification or field recalibration.",
            "construction": "Analytic material solids; one STEP per optical unit and a complete module assembly."}


class CadExportDialog(QDialog):
    completed = Signal(bool, str)

    def __init__(self, catalog, output_directory, parent=None, *, project_root=None,
                 interpreter=None, exporter=None):
        super().__init__(parent)
        self.setWindowTitle("Export STEP")
        self.resize(660, 390)
        self.setModal(False)
        self.project_root = Path(project_root or Path(__file__).resolve().parents[3]).resolve()
        self.interpreter = Path(interpreter or self.project_root / "environment/cad/Scripts/python.exe")
        self.exporter = Path(exporter or self.project_root / "scripts/export_cad_models.py")
        self.output_directory = Path(output_directory).resolve()
        self.catalog = deepcopy(catalog)
        self.busy = False
        self.cancelled = False
        self.success = False
        self._close_pending = False
        self._finished = False
        self._temporary = None
        self._log_text = ""
        self.status = QLabel("Preparing the current draft for export.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        self.destination = QLabel(str(self.output_directory))
        self.destination.setWordWrap(True)
        self.destination.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.action = QPushButton("Cancel export")
        self.action.clicked.connect(self.cancel)
        layout = QVBoxLayout(self)
        for widget in (self.status, self.destination, self.log, self.action):
            layout.addWidget(widget)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_owned_process)
        QApplication.instance().aboutToQuit.connect(self._stop_for_exit)
        if parent is not None:
            parent.destroyed.connect(self._stop_for_exit)

    def start(self):
        if self.busy or self._finished:
            raise RuntimeError("This export has already been started.")
        if not self.interpreter.is_file():
            raise FileNotFoundError("The optional STEP exporter is not installed.\n\n" + setup_instructions())
        if not self.exporter.is_file():
            raise FileNotFoundError("The STEP export script is missing from this installation.")
        encoded = json.dumps(self.catalog, indent=2, ensure_ascii=False, allow_nan=False)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        # The retained input makes an unsaved-draft export reproducible.
        (self.output_directory / "catalog-input.json").write_text(encoded, encoding="utf-8")
        self._temporary = tempfile.TemporaryDirectory(prefix="temsim-step-")
        input_path = Path(self._temporary.name) / "catalog.json"
        input_path.write_text(encoded, encoding="utf-8")
        environment = QProcessEnvironment.systemEnvironment()
        for name in ("PYTHONPATH", "PYTHONHOME"):
            environment.remove(name)
        environment.insert("PYTHONNOUSERSITE", "1")
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.project_root))
        self.busy = True
        self.status.setText("Exporting the captured draft in the background. You can continue using the simulator.")
        self.process.start(str(self.interpreter), [str(self.exporter), str(input_path),
                                                 "--output", str(self.output_directory)])

    def _read_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self._log_text = (self._log_text + text)[-64000:]
            self.log.moveCursor(self.log.textCursor().MoveOperation.End)
            self.log.insertPlainText(text)

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self._finish(False, "STEP exporter could not start: " + self.process.errorString())

    def _process_finished(self, exit_code, exit_status):
        self._read_output()
        if self.cancelled:
            self._finish(False, "Export cancelled. Any completed files are retained; this export is incomplete.")
            return
        try:
            if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
                raise RuntimeError("STEP export did not complete. See the details below.")
            report = json.loads((self.output_directory / "step-manifest.json").read_text(encoding="utf-8"))
            expected = hashlib.sha256(json.dumps(self.catalog, sort_keys=True,
                separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            if report.get("status") != "passed" or report.get("catalog_sha256") != expected:
                raise RuntimeError("The exporter did not confirm completion for this captured draft.")
            self._finish(True, "STEP export complete. Separate optical units and the module assembly are saved below.")
        except Exception as exc:
            message = str(exc)
            if "No module named" in self._log_text or "CadQuery" in self._log_text:
                message += "\n\n" + setup_instructions()
            self._finish(False, message)

    def _finish(self, success, message):
        if self._finished:
            return
        self._finished = True
        self.busy = False
        self.success = bool(success)
        self._kill_timer.stop()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
        self.status.setText(message)
        self.action.setText("Close")
        self.action.setEnabled(True)
        self.completed.emit(bool(success), message)
        if self._close_pending:
            self.reject()

    def cancel(self):
        if not self.busy:
            self.reject()
            return
        self.cancelled = True
        self.action.setEnabled(False)
        self.status.setText("Stopping this STEP export…")
        self.process.terminate()
        self._kill_timer.start(1500)

    def _kill_owned_process(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    def _stop_for_exit(self):
        # Shutdown affects only this QProcess, never another CAD application.
        if self.busy:
            self.cancelled = True
            self.process.terminate()
            if not self.process.waitForFinished(500):
                self.process.kill()
                self.process.waitForFinished(500)

    def closeEvent(self, event):
        if self.busy:
            self._close_pending = True
            self.cancel()
            event.ignore()
        else:
            super().closeEvent(event)

    def reject(self):
        # QDialog's Escape shortcut calls reject() directly, not closeEvent().
        if self.busy:
            self._close_pending = True
            self.cancel()
        else:
            super().reject()
