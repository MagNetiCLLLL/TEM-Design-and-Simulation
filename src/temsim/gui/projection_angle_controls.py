"""One consistent observation-angle control for longitudinal column views."""

import math

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QWidget


VIEW_OPTION_BUTTON_STYLE = """
        QPushButton {
            min-height: 24px;
            padding: 2px 7px;
            border: 1px solid #64748b;
            border-radius: 6px;
            background: #f8fafc;
            color: #0f172a;
            font-size: 12px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #e2e8f0;
            border-color: #334155;
        }
        QPushButton:checked {
            background: #2563eb;
            border-color: #1d4ed8;
            color: white;
        }
        QPushButton:checked:hover {
            background: #1d4ed8;
        }
    """


class ProjectionAngleControls(QWidget):
    """Share angle presentation; receiving a value never echoes a user request."""

    angle_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("projectionAngleControls")
        self._angle_deg = 0.0
        self.label = QLabel("Angle")
        self.label.setToolTip(
            "Rotate the displayed transverse axis about Z. This changes the observation "
            "angle, not the hardware or the particle calculation."
        )
        self.xz = QPushButton("X-Z")
        self.xz.setObjectName("projectionXZButton")
        self.yz = QPushButton("Y-Z")
        self.yz.setObjectName("projectionYZButton")
        for button, angle in ((self.xz, 0.0), (self.yz, 90.0)):
            button.setCheckable(True)
            button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
            button.setStyleSheet(VIEW_OPTION_BUTTON_STYLE)
            button.setToolTip(f"Observe the {button.text()} plane ({angle:g}°).")
            button.clicked.connect(lambda _checked=False, value=angle: self._request_angle(value))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setObjectName("projectionAngleSlider")
        self.slider.setRange(0, 3600)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(50)
        self.slider.setMinimumWidth(120)
        self.slider.setMaximumWidth(180)
        self.slider.setToolTip("Transverse projection angle, in tenths of a degree")
        self.slider.valueChanged.connect(lambda value: self._request_angle(value / 10.0))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        for widget in (self.label, self.xz, self.yz, self.slider):
            layout.addWidget(widget)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.set_angle(0.0)

    @property
    def angle_deg(self):
        return self._angle_deg

    def set_angle(self, angle_deg):
        """Receive an angle in degrees, preserving 360° as the slider endpoint."""
        angle = float(angle_deg)
        if not math.isfinite(angle):
            return
        self._angle_deg = angle = min(360.0, max(0.0, angle))
        with QSignalBlocker(self.slider):
            self.slider.setValue(int(math.floor(angle * 10.0 + 0.5)))
        normalized = angle % 360.0
        self.xz.setChecked(math.isclose(normalized, 0.0, abs_tol=0.05))
        self.yz.setChecked(math.isclose(normalized, 90.0, abs_tol=0.05))

    def _request_angle(self, angle):
        changed = angle != self._angle_deg
        self.set_angle(angle)
        if changed:
            self.angle_changed.emit(self._angle_deg)
