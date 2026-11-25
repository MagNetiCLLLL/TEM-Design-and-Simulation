"""Desktop presentation of the solved, fixed-excitation joint FEM instrument."""

import json
import sys
from uuid import uuid4

from PySide6.QtCore import QProcess, Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from temsim.branding import application_cache_root
from temsim.joint_fem import load_profile
from temsim.paths import OPERATING_MODE_CONFIG_ROOT


class FigureView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))

    def load(self, path):
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError(f"Could not read {path.name}")
        self.scene().clear()
        self.scene().addPixmap(pixmap).setTransformationMode(Qt.SmoothTransformation)
        self.scene().setSceneRect(self.scene().itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.sceneRect().isEmpty():
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class JointFEMPage(QWidget):
    open_editor = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._error)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        layout.addLayout(controls)
        self.working_point = QComboBox()
        self.working_point.setAccessibleName("Joint FEM working point")
        self.working_point.addItem("Nanoprobe Normal  |  HPol A2 adjustment", None)
        original = OPERATING_MODE_CONFIG_ROOT / "fei_50v" / "profile.json"
        if original.is_file():
            self.working_point.addItem("Nanoprobe Normal  |  +50 V joint FEM", str(original))
        relay_profile = OPERATING_MODE_CONFIG_ROOT / "fei_50v_corrector_relay" / "profile.json"
        if relay_profile.is_file():
            self.working_point.addItem("Nanoprobe Normal  |  Corrector relay", str(relay_profile))
        controls.addWidget(self.working_point)
        controls.addStretch()
        self.retrace = QPushButton("Trace particles")
        self.retrace.clicked.connect(self.start)
        controls.addWidget(self.retrace)
        self.open_folder = QPushButton("Open result folder")
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output)))
        )
        controls.addWidget(self.open_folder)
        editor = QPushButton("Switch to editable instrument")
        editor.clicked.connect(self.open_editor)
        controls.addWidget(editor)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status)
        tabs = QTabWidget()
        layout.addWidget(tabs, 1)
        self.column = FigureView()
        self.condenser = FigureView()
        self.sample = FigureView()
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        tabs.addTab(self.column, "Column trajectories")
        tabs.addTab(self.condenser, "C1 and C2 source images")
        tabs.addTab(self.sample, "Specimen landings")
        tabs.addTab(self.details, "Field and excitation")
        self.working_point.currentIndexChanged.connect(self.start)

    def start(self):
        if self.process.state() != QProcess.NotRunning:
            return
        self.output = application_cache_root("joint_fem_runs") / uuid4().hex
        self.open_folder.setEnabled(False)
        self.column.scene().clear()
        self.condenser.scene().clear()
        self.sample.scene().clear()
        self.details.clear()
        try:
            path = self.working_point.currentData()
            self.profile = load_profile(path)
            count = self.profile["source_count"]
            if not isinstance(count, int) or count < 1000:
                raise ValueError("Invalid original particle count in the working point.")
        except (ValueError, KeyError, OSError) as exc:
            self.status.setText(f"Unable to load working point: {exc}")
            return
        self.details.setPlainText(
            json.dumps(self.profile, indent=2, ensure_ascii=False)
        )
        self.retrace.setText(f"Trace {count:,} particles")
        self.status.setText(
            f"Tracing {count:,} weighted electrons from the emitting tip to the specimen upper surface…"
        )
        self.retrace.setEnabled(False)
        self.working_point.setEnabled(False)
        args = ["-m", "temsim.joint_fem", str(self.output)]
        if path:
            args.extend(["--profile", path])
        self.process.start(sys.executable, args)

    def _finished(self, code, exit_status):
        self.retrace.setEnabled(True)
        self.working_point.setEnabled(True)
        if code or exit_status != QProcess.NormalExit:
            message = bytes(self.process.readAllStandardOutput()).decode(
                errors="replace"
            )
            self.status.setText(f"Trace failed. {message[-1500:]}")
            return
        try:
            result = json.loads(
                (self.output / "result.json").read_text(encoding="utf-8")
            )
            if (
                result["field_sha256"] != self.profile["solution_sha256"]
                or result["emitted"] != self.profile["source_count"]
            ):
                raise ValueError(
                    "The result does not match the selected working point."
                )
            self.column.load(self.output / "column-rays.png")
            self.condenser.load(self.output / "condenser-relay.png")
            self.sample.load(self.output / "specimen-landings.png")
            self.details.setPlainText(
                json.dumps(
                    {"working_point": self.profile, "trace": result},
                    indent=2,
                    ensure_ascii=False,
                )
            )
        except (OSError, KeyError, ValueError) as exc:
            self.status.setText(f"Unable to present this trace: {exc}")
            return
        self.status.setText(
            f"{result['emitted']:,} emitted  ·  {result['gun_exit']} at gun exit  ·  {result['sample']} at specimen  ·  "
            f"{result['sample_nA']:.3f} nA  ·  D95 {result['D95_um']:.3f} µm"
        )
        self.open_folder.setEnabled(True)

    def _error(self, error):
        if error == QProcess.FailedToStart:
            self.retrace.setEnabled(True)
            self.working_point.setEnabled(True)
            self.status.setText(
                f"Could not start particle transport: {self.process.errorString()}"
            )

    def shutdown(self):
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()
            self.process.waitForFinished(3000)
