"""Maximum/percentage controls with separate intrinsic and learning matrices."""
from copy import copy
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox, WheelSafeDoubleSpinBox
from temsim.runtime_parameters import runtime_targets, validate_runtime_assignment
from temsim.optics.probe_service import (
    OUTPUTS, ROUTES, apply_probe_service, output_reading, preview_probe_service,
    preview_probe_output, service_settings, set_probe_output_percent,
)


def _label(text=""):
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse
                                  | Qt.TextInteractionFlag.TextSelectableByKeyboard)
    return label


def _percent():
    editor = WheelSafeDoubleSpinBox()
    editor.setDecimals(8)
    editor.setRange(-100., 100.)
    editor.setSingleStep(.001)
    editor.setSuffix(" %")
    editor.setKeyboardTracking(False)
    return editor


def _preview_text(changes):
    return "\n".join(f"{c.component}.{c.field}   {c.before:.10g} → {c.after:.10g} {c.unit}"
                     for c in changes)


class ProbeServiceControls(QWidget):
    runtime_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._preview = ()
        self._loaded_settings = None
        self.tool = WheelSafeComboBox()
        for name in dict.fromkeys(route.tool for route in ROUTES.values()):
            self.tool.addItem(name)
        self.input = WheelSafeComboBox()
        self.delta_percent = _percent()
        self.delta_percent.setValue(.001)
        self.learning_enabled = QCheckBox("Apply calibrated learning corrections")
        self.evidence = _label()
        self.gains = QTableWidget(0, 5)
        self.gains.setHorizontalHeaderLabels(("Output", "Role", "Model maximum", "Current (%)", "Response gain"))
        self.gains.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.gains.verticalHeader().hide()
        self.gains.verticalHeader().setDefaultSectionSize(23)
        self.gains.setMinimumHeight(130)
        self.gains.setMaximumHeight(210)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMinimumHeight(90)
        self.preview.setMaximumHeight(120)
        self.apply = QPushButton("Apply relative percentage")
        self.reset = QPushButton("Reset intrinsic defaults and clear learning")
        self.status = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(_label(
            "Each output uses its installed model maximum. Response gain is the change in output "
            "percentage points per input percentage point. Small secondary responses belong to learning. "
            "Learning starts disabled and its matrix starts at zero."))
        form = QFormLayout()
        form.addRow("Service tool", self.tool)
        form.addRow("Input", self.input)
        form.addRow("Relative full-scale change", self.delta_percent)
        layout.addLayout(form)
        layout.addWidget(self.learning_enabled)
        layout.addWidget(self.evidence)
        layout.addWidget(self.gains)
        layout.addWidget(_label("Proposed physical drives"))
        layout.addWidget(self.preview)
        actions = QHBoxLayout()
        actions.addWidget(self.reset)
        actions.addWidget(self.apply)
        layout.addLayout(actions)
        layout.addWidget(self.status)
        self.tool.currentIndexChanged.connect(self._tool_changed)
        self.input.currentIndexChanged.connect(self._input_changed)
        self.delta_percent.valueChanged.connect(self._edit_changed)
        self.learning_enabled.toggled.connect(self._edit_changed)
        self.gains.itemChanged.connect(self._edit_changed)
        self.reset.clicked.connect(lambda: self._input_changed(defaults=True))
        self.apply.clicked.connect(self._apply)
        self.tool.setCurrentText("AT_A2")

    def _tool_changed(self, *_):
        with QSignalBlocker(self.input):
            self.input.clear()
            for route in ROUTES.values():
                if route.tool == self.tool.currentText():
                    label = {"AT_A2.upper": "Upper · HP1/HP2 differential",
                             "AT_A2.lower": "Lower · TL21/TL22 differential"}.get(route.key, route.input_name.title())
                    self.input.addItem(label, route.key)
        self._input_changed()

    def _input_changed(self, *_, defaults=False):
        key = self.input.currentData()
        route = ROUTES[key]
        self.evidence.setText(route.evidence)
        self._loaded_settings = service_settings(self._state, key)
        primary, learned, enabled = service_settings(None, key) if defaults else self._loaded_settings
        with QSignalBlocker(self.learning_enabled):
            self.learning_enabled.setChecked(enabled)
        self._rows = [(name, "Intrinsic", value) for name, value in primary.items()]
        self._rows += [(name, "Learning", value) for name, value in learned.items()]
        with QSignalBlocker(self.gains):
            self.gains.setRowCount(len(self._rows))
            for row, (name, role, gain) in enumerate(self._rows):
                for column, value in enumerate((name, role, "", "", format(gain, ".9g"))):
                    item = QTableWidgetItem(value)
                    if column != 4:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    else:
                        item.setData(Qt.ItemDataRole.UserRole, gain)
                        item.setToolTip(f"{gain!r} output percentage points per input percentage point")
                    self.gains.setItem(row, column, item)
        self._edit_changed()

    def _edit_changed(self, *_):
        self.status.clear()
        self.refresh()

    def set_state(self, state):
        if state is not self._state:
            self._state = state
            self._input_changed()
        else:
            self.refresh()

    def _arguments(self):
        primary, learned = {}, {}
        for row, (name, role, _) in enumerate(self._rows):
            item = self.gains.item(row, 4)
            text, exact = item.text().strip(), item.data(Qt.ItemDataRole.UserRole)
            try:
                gain = exact if text == format(exact, ".9g") else float(text or "0")
            except ValueError:
                raise ValueError(f"{name} needs a finite response gain") from None
            (primary if role == "Intrinsic" else learned)[name] = gain
        return dict(delta_percent=self.delta_percent.value(), intrinsic=primary, learning=learned,
                    learning_enabled=self.learning_enabled.isChecked())

    def refresh(self):
        key = self.input.currentData()
        if service_settings(self._state, key) != self._loaded_settings:
            self._input_changed()
            return
        with QSignalBlocker(self.gains):
            for row, (name, _, _) in enumerate(self._rows):
                try:
                    r = output_reading(self._state, name)
                    maximum, current = f"{r.maximum:.8g} {r.unit}", f"{r.percent:.8g}"
                except ValueError:
                    maximum, current = "Unavailable", ""
                self.gains.item(row, 2).setText(maximum)
                self.gains.item(row, 3).setText(current)
        try:
            self._preview = preview_probe_service(self._state, key, **self._arguments())
        except (ValueError, TypeError) as exc:
            self._preview, text = (), str(exc)
        else:
            text = _preview_text(self._preview)
        self.preview.setPlainText(text)
        self.apply.setEnabled(bool(self._preview))

    def _apply(self):
        if not self._preview:
            return
        try:
            applied = apply_probe_service(self._state, self.input.currentData(), **self._arguments(), expected=self._preview)
        except (ValueError, TypeError) as exc:
            self.refresh()
            self.status.setText(str(exc))
            return
        self.runtime_changed.emit("corrector_channels.service_drives")
        self.refresh()
        self.status.setText(f"Applied {len(applied)} drive changes. Intrinsic and learning settings were saved separately.")


class ProbeOutputControls(QWidget):
    runtime_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = None
        self._preview = ()
        self._dirty = False
        self.output = WheelSafeComboBox()
        self.output.addItems(list(OUTPUTS))
        self.maximum = _label()
        self.current = _label()
        self.frame = _label()
        self.enabled = QCheckBox("Component enabled")
        self.percent = _percent()
        self.preview = _label()
        self.status = _label()
        self.apply = QPushButton("Set output percentage")
        layout = QVBoxLayout(self)
        layout.addWidget(_label(
            "Set an absolute output percentage using the installed model maximum. "
            "Signed multipole and deflector outputs allow negative values. "
            "The normal/skew vector and shared physical coils retain their combined limits."))
        form = QFormLayout()
        for label, widget in (("Physical output", self.output), ("Model maximum", self.maximum),
                              ("Current output", self.current), ("Set percentage", self.percent)):
            form.addRow(label, widget)
        layout.addLayout(form)
        layout.addWidget(self.enabled)
        layout.addWidget(self.frame)
        layout.addWidget(self.preview)
        layout.addWidget(self.apply)
        layout.addWidget(self.status)
        layout.addStretch()
        self.output.currentIndexChanged.connect(self._select)
        self.percent.valueChanged.connect(self._edit)
        self.enabled.toggled.connect(self._toggle_enabled)
        self.apply.clicked.connect(self._apply)
        self._select()

    def set_state(self, state):
        if state is not self._state:
            self._state = state
            self._select()
        else:
            self.refresh()

    def _select(self, *_):
        self._dirty = False
        self.status.clear()
        self.refresh()

    def _edit(self, *_):
        self._dirty = True
        self.refresh()

    def _toggle_enabled(self, checked):
        if self._state is None:
            return
        name = self.output.currentText()
        try:
            output_reading(self._state, name)
            target = runtime_targets(self._state)[OUTPUTS[name].component]
            value = validate_runtime_assignment(target, "enabled", checked)
            candidate = copy(target.obj)
            candidate.enabled = value
            candidate.validate()
            target.obj.enabled = value
        except (ValueError, TypeError) as exc:
            self.refresh()
            self.status.setText(str(exc))
            return
        self.runtime_changed.emit("corrector_channels.output_percent")
        self.refresh()

    def refresh(self, *_):
        name = self.output.currentText()
        try:
            current = output_reading(self._state, name)
            target = runtime_targets(self._state)[OUTPUTS[name].component]
            if not self._dirty:
                with QSignalBlocker(self.percent):
                    self.percent.setMinimum(0. if OUTPUTS[name].limit_field == "max_percent" else -100.)
                    self.percent.setValue(current.percent)
            with QSignalBlocker(self.enabled):
                self.enabled.setChecked(target.obj.enabled)
            self.enabled.setEnabled(True)
            self.maximum.setText(f"{current.maximum:.10g} {current.unit}")
            self.current.setText(f"{current.percent:.8g}%   ({current.native:.10g} {current.unit})")
            self.frame.setText(current.frame + " Specimen and image directions follow the column transfer.")
            self._preview = (preview_probe_output(self._state, name, percent=self.percent.value())
                             if self._dirty else ())
        except (ValueError, TypeError) as exc:
            self._preview = ()
            if self._state is None or not getattr(self._state, "probe_corrector_installed", False):
                self.enabled.setEnabled(False)
            self.preview.setText(str(exc))
        else:
            self.preview.setText(_preview_text(self._preview))
        self.apply.setEnabled(bool(self._preview))

    def _apply(self):
        try:
            set_probe_output_percent(self._state, self.output.currentText(), percent=self.percent.value(), expected=self._preview)
        except (ValueError, TypeError) as exc:
            self.refresh()
            self.status.setText(str(exc))
            return
        self.runtime_changed.emit("corrector_channels.output_percent")
        self._dirty = False
        self.refresh()
        self.status.setText("Applied the absolute output percentage.")
