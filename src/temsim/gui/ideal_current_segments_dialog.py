"""Edit closed current groups without editing the winding geometry."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)
from temsim.gui.input_policy import WheelSafeSpinBox as QSpinBox
from temsim.ideal_coil_segments import (
    FIELD, SCHEMA, MAXIMUM_SEGMENTS_PER_WINDING, axial_recipe,
    bind_segments, normalize_recipe, require_ideal_mode,
)
from temsim.coil_winding import OWNER_CURRENT_FIELD, INTERVAL_TURNS_FIELD


class IdealCurrentSegmentsDialog(QDialog):
    """The draft remains detached until the caller explicitly accepts it."""

    def __init__(self, state, owner_key, parent=None):
        super().__init__(parent)
        require_ideal_mode(state)
        self.setWindowTitle("Internal forward / reverse coil currents")
        self.resize(1120, 560)
        self.setObjectName("idealCurrentSegmentsDialog")
        self.owner_key = owner_key
        self.parts = tuple(part for part in state._resolved_assembly.parts
            if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
            and part.data.get("field_source_key") == owner_key)
        if not self.parts:
            raise ValueError("Select an installed round lens with physical winding turns")
        owner = state._resolved_assembly.part(owner_key)
        self.master = owner.data.get(OWNER_CURRENT_FIELD)
        if self.master is None:
            raise ValueError("Define the prescribed master current before editing current groups")
        lens = next(lens for lens in state.lenses if lens.key == owner_key)
        self.polarity = getattr(lens, "polarity", 1)
        self.scale = lens.percent*.01*getattr(lens, "polarity", 1) if lens.enabled else 0.
        self.counts = {}
        for part in self.parts:
            counts = part.data.get(INTERVAL_TURNS_FIELD)
            if counts is None:
                if len(part.data.get("material_intervals_mm", (None,))) != 1:
                    raise ValueError("Define fixed interval turns before adding internal current groups")
                counts = (part.data.get("winding_turn_count"),)
            self.counts[part.key] = tuple(counts)
        original = state.lens_field_map_descriptors.get(owner_key, {}).get(FIELD)
        self.recipe = normalize_recipe(deepcopy(original)) if original is not None else self._uniform_recipe()
        self._rows = []
        layout = QVBoxLayout(self)
        note = QLabel(
            "Current groups occupy the existing winding envelope. Inner/outer diameters, axial intervals, "
            "physical turns and conductor area stay unchanged. A negative ratio reverses a complete "
            "closed subwinding. No resistance or constant-power constraint is applied.")
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(note)
        controls = QHBoxLayout()
        self.groups = QSpinBox()
        self.groups.setObjectName("idealCurrentGroupCount")
        maximum = min(min(counts) for counts in self.counts.values())
        maximum = min(maximum, min(MAXIMUM_SEGMENTS_PER_WINDING//len(counts) for counts in self.counts.values()))
        self.groups.setRange(1, maximum)
        generate = QPushButton("Split axially")
        generate.setObjectName("splitIdealCurrentGroups")
        generate.setToolTip("Prepare equal-density groups by integer turns; reset their current ratios to one")
        generate.clicked.connect(self._generate)
        uniform = QPushButton("Set all ratios to 1")
        uniform.clicked.connect(self._uniform_currents)
        load = QPushButton("Import groups…")
        load.clicked.connect(self._import)
        save = QPushButton("Export groups…")
        save.clicked.connect(self._export)
        controls.addWidget(QLabel("Groups per original interval"))
        controls.addWidget(self.groups)
        for widget in (generate, uniform, load, save):
            controls.addWidget(widget)
        controls.addStretch(1)
        layout.addLayout(controls)
        self.table = QTableWidget(0, 8)
        self.table.setObjectName("idealCurrentSegmentsTable")
        self.table.setHorizontalHeaderLabels(("Physical winding", "Interval", "Radial fraction",
            "Axial fraction", "Turns", "Current ratio", "At 100% (A)", "Active current (A)"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._update_currents)
        layout.addWidget(self.table, 1)
        self.error = QLabel()
        self.error.setWordWrap(True)
        self.error.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.error)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        remove = self.buttons.addButton("Remove grouping", QDialogButtonBox.ButtonRole.ResetRole)
        remove.setObjectName("removeIdealCurrentGrouping")
        remove.clicked.connect(self._remove)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._show_recipe()

    def _uniform_recipe(self):
        windings = {}
        for key, counts in self.counts.items():
            windings.update(axial_recipe(key, counts, 1)["windings"])
        return dict(schema=SCHEMA, windings=windings)

    def _validate(self, recipe):
        result = normalize_recipe(recipe)
        if set(result["windings"])-set(self.counts):
            raise ValueError("The imported groups refer to another physical winding")
        for key in result["windings"]:
            bind_segments(result, key, self.counts[key])
        return result

    def _show_recipe(self):
        self.recipe = self._validate(self.recipe)
        self._rows = [(key, deepcopy(cell)) for key, cells in self.recipe["windings"].items() for cell in cells]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._rows))
        for row, (key, cell) in enumerate(self._rows):
            values = (key, str(cell["interval_index"]+1),
                "–".join(format(value, ".9g") for value in cell["radial_fraction"]),
                "–".join(format(value, ".9g") for value in cell["axial_fraction"]),
                str(cell["turn_count"]), format(cell["current_ratio"], ".17g"), "", "")
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column != 5:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)
        self._update_currents()

    def _draft(self):
        windings = {}
        for row, (key, original) in enumerate(self._rows):
            cell = deepcopy(original)
            cell["current_ratio"] = float(self.table.item(row, 5).text())
            windings.setdefault(key, []).append(cell)
        return self._validate(dict(schema=SCHEMA, windings=windings))

    def _update_currents(self, *_args):
        try:
            self._draft()
            self.table.blockSignals(True)
            for row, (key, _) in enumerate(self._rows):
                ratio = float(self.table.item(row, 5).text())
                value = self.master*ratio*self.polarity
                active = self.master*ratio*self.scale
                if not math.isfinite(value) or not math.isfinite(active):
                    raise ValueError("Segment current exceeds the finite numerical range")
                self.table.item(row, 6).setText(format(value, ".10g"))
                self.table.item(row, 7).setText(format(active, ".10g"))
            self.table.blockSignals(False)
            self.error.clear()
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)
        except (TypeError, ValueError, OverflowError) as error:
            self.table.blockSignals(False)
            self.error.setText(str(error))
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def _generate(self):
        try:
            windings = {}
            for key, counts in self.counts.items():
                windings.update(axial_recipe(key, counts, self.groups.value())["windings"])
            self.recipe = dict(schema=SCHEMA, windings=windings)
            self._show_recipe()
        except ValueError as error:
            self.error.setText(str(error))

    def _uniform_currents(self):
        self.table.blockSignals(True)
        for row in range(self.table.rowCount()):
            self.table.item(row, 5).setText("1")
        self.table.blockSignals(False)
        self._update_currents()

    def _import(self):
        name, _ = QFileDialog.getOpenFileName(self, "Import ideal-coil current groups", "", "Current groups (*.json)")
        if name:
            try:
                recipe = json.loads(Path(name).read_text(encoding="utf-8"))
                self.recipe = self._validate(recipe)
                self._show_recipe()
            except (OSError, TypeError, ValueError) as error:
                self.error.setText(str(error))

    def _export(self):
        try:
            draft = self._draft()
            name, _ = QFileDialog.getSaveFileName(self, "Export ideal-coil current groups", "", "Current groups (*.json)")
            if name:
                Path(name).write_text(json.dumps(draft, indent=2)+"\n", encoding="utf-8")
        except (OSError, TypeError, ValueError) as error:
            self.error.setText(str(error))

    def _remove(self):
        self.recipe = dict(schema=SCHEMA, windings={})
        super().accept()

    def accept(self):
        try:
            self.recipe = self._draft()
        except (TypeError, ValueError, OverflowError) as error:
            self.error.setText(str(error))
            return
        super().accept()
