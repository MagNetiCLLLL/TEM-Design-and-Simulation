"""Run a fixed-current, three-dimensional probe-corrector solve outside the GUI process."""

from pathlib import Path
import sys

from PySide6.QtCore import QProcess, QStandardPaths, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)


class CorrectorFieldDialog(QDialog):
    def __init__(self, column, parent=None):
        super().__init__(parent)
        self.column = Path(column)
        self.setWindowTitle("Probe corrector field")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.currents = {}
        for name, current in (("HP1", 1.0), ("HP2", -1.0), ("TL21", 5.0), ("TL22", -5.0)):
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-1e6, 1e6)
            spin.setValue(current)
            spin.setSuffix(" A")
            form.addRow(f"{name} signed winding current", spin)
            self.currents[name] = spin
        self.backend = QComboBox()
        self.backend.addItems(["CPU", "CUDA"])
        form.addRow("Sparse solver", self.backend)
        layout.addLayout(form)
        source = QLabel(
            '<a href="https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf#page=3">'
            "CESCOR dimensions and hexapole photograph</a>"
        )
        source.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        source.setOpenExternalLinks(True)
        layout.addWidget(source)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlainText(
            "Uses the saved column materials. The output contains the fixed-current "
            "magnetic solution for both hexapoles and both transfer lenses."
        )
        layout.addWidget(self.log)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.calculate = QPushButton("Calculate and save…")
        buttons.addButton(self.calculate, QDialogButtonBox.ButtonRole.ActionRole)
        self.calculate.clicked.connect(self.start)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)

    def start(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save corrector magnetic solution", "probe-corrector.npz", "NumPy archive (*.npz)"
        )
        if not path:
            return
        cache = (
            Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.CacheLocation))
            / "corrector-mesh"
        )
        args = [
            "-u",
            "-m",
            "temsim.corrector_field_cli",
            "solve",
            path,
            "--column",
            str(self.column),
            "--mesh-cache",
            str(cache),
            "--backend",
            self.backend.currentText().lower(),
        ]
        for name, spin in self.currents.items():
            args.extend(["--" + name.lower(), str(spin.value())])
        self.output = path
        self.log.setPlainText("Meshing the installed magnetic regions…")
        self.calculate.setEnabled(False)
        self.process.start(sys.executable, args)

    def _read(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace").strip()
        if text:
            self.log.appendPlainText(text)

    def _finished(self, code, status):
        self._read()
        self.calculate.setEnabled(True)
        if code == 0 and status == QProcess.ExitStatus.NormalExit:
            self.log.appendPlainText(f"Saved {self.output}")

    def _error(self, _):
        self.log.appendPlainText(self.process.errorString())
        self.calculate.setEnabled(True)

    def reject(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
        super().reject()
