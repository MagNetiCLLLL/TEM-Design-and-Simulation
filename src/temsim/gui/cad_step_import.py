"""Review STEP placement without deriving an optical datum from its bounds."""
from pathlib import Path
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout

from temsim.gui.input_policy import WheelSafeComboBox


class StepImportDialog(QDialog):
    def __init__(self, path, *, selected_part=None, existing_keys=(), parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import STEP")
        self.resize(600, 350)
        self.path = Path(path).resolve()
        self.selected_part = dict(selected_part or {})
        self.existing_keys = set(existing_keys)
        self.result_value = None
        layout = QVBoxLayout(self)
        source = QLabel(str(self.path))
        source.setWordWrap(True)
        source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(source)
        form = QFormLayout()
        self.mode = WheelSafeComboBox()
        self.mode.setObjectName("stepImportMode")
        self.mode.addItem("New mechanical component — preserve STEP coordinates", "add")
        if self.selected_part:
            self.mode.addItem("Replace the selected component's material geometry", "replace")
        form.addRow("Import as", self.mode)
        key = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.path.stem).strip("_.-") or "part"
        key = "step_" + key
        stem, index = key, 2
        while key in self.existing_keys:
            key, index = f"{stem}_{index}", index+1
        self.key = QLineEdit(key)
        self.key.setObjectName("stepImportKey")
        self.name = QLineEdit(self.path.stem)
        self.name.setObjectName("stepImportName")
        self._new_key, self._new_name = key, self.path.stem
        form.addRow("Component key", self.key)
        form.addRow("Name", self.name)
        self.registration = WheelSafeComboBox()
        self.registration.setObjectName("stepImportRegistration")
        self.registration.addItem("Preserve STEP coordinates in this module", "preserve")
        self.registration.addItem("Centre the material at the saved component datum", "center")
        form.addRow("Coordinate registration", self.registration)
        self.registration.currentIndexChanged.connect(self._registration_changed)
        layout.addLayout(form)
        self.registration_note = QLabel()
        self.registration_note.setWordWrap(True)
        self.registration_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.registration_note)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setStyleSheet("color:#ffb4a9;")
        layout.addWidget(self.error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.import_button.setText("Import")
        buttons.accepted.connect(self._submit)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    def _mode_changed(self):
        replace = self.mode.currentData() == "replace"
        if replace:
            self._new_key, self._new_name = self.key.text(), self.name.text()
        self.key.setReadOnly(replace)
        self.name.setReadOnly(replace)
        self.key.setText(self.selected_part.get("key", "") if replace else self._new_key)
        self.name.setText(self.selected_part.get("name", self.selected_part.get("key", "")) if replace else self._new_name)
        self.registration.setEnabled(replace)
        if not replace:
            self.registration.setCurrentIndex(0)
        self._registration_changed()

    def _registration_changed(self):
        replace = self.mode.currentData() == "replace"
        if replace:
            center = self.selected_part.get("optical_reference_local_z_mm", self.selected_part.get("local_center_z_mm"))
            placement = ("Keep the STEP XYZ coordinates in the existing module frame. No automatic centering is performed. "
                         if self.registration.currentData() == "preserve" else
                         "Explicitly translate the STEP bounding-box centre to the existing component centre. "
                         "This registers the material; it does not infer or move an optical plane. ")
            self.registration_note.setText(
                placement + "Coordinates are millimetres. The component's existing physical placement is retained. "
                f"The saved optical datum ({center} mm) and all optical settings are retained; "
                "the STEP bounding box does not define a new optical plane. Only this component's material "
                "is replaced; its children remain separately defined.")
        else:
            self.registration_note.setText(
                "Keep the STEP XYZ coordinates in the current module frame, in millimetres. "
                "The imported object is mechanical material only. Its bounds do not create a lens, "
                "detector acceptance, or optical reference plane. Use the existing Dimensions and Place "
                "controls for an explicit placement after import.")

    def _submit(self):
        key, name = self.key.text().strip(), self.name.text().strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*", key):
            self.error.setText("Use a component key containing letters, numbers, _, . or -; start with a letter or _.")
            return
        if not name or (self.mode.currentData() == "add" and key in self.existing_keys):
            self.error.setText("Enter a name and a new, unused component key.")
            return
        self.result_value = dict(mode=self.mode.currentData(), key=key, name=name,
                                 path=str(self.path), registration=self.registration.currentData(),
                                 coordinate_frame="module_local_mm")
        self.accept()
