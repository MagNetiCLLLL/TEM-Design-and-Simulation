"""Transactional editor for the displayed vacuum plumbing, not a pump solver."""
from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout,
    QWidget,
)

from temsim.gui.input_policy import WheelSafeComboBox as QComboBox
from temsim.vacuum import boundary_anchors
from temsim.vacuum_pumping import (
    default_pumping_layout, effective_pumping_layout, validate_pumping_layout,
)


PUMP_TYPES = (
    ("Ion getter pump (IGP)", "igp"), ("Turbomolecular pump", "turbo"),
    ("Dry scroll pump", "scroll"), ("Rotary pump", "rotary"),
    ("Cryopump", "cryo"), ("Diffusion pump", "diffusion"), ("Other pump", "other"),
)
LINE_ROLES = (
    ("High vacuum", "high_vacuum"), ("Roughing", "roughing"),
    ("Backing / foreline", "backing"), ("Chamber transfer", "transfer"),
)


def _choice(options, value):
    combo = QComboBox()
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(10)
    for label, data in options:
        combo.addItem(label, data)
    index = combo.findData(value)
    if index < 0 and value is not None:
        combo.addItem(f"{value} [missing]", value)
        index = combo.count() - 1
    combo.setCurrentIndex(max(0, index))
    return combo


class VacuumPumpingDialog(QDialog):
    """Keep changes in a draft until the entire graph validates."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vacuum map — pumps and pipes")
        self.resize(940, 560)
        self._state = state
        self._filling = False
        self._base = {}
        self.result_layout = None
        self.regions = tuple(state.vacuum_map.regions)
        self.anchors = tuple(boundary_anchors(state))
        outer = QVBoxLayout(self)
        note = QLabel(
            "Edit installed pumps and their connections. One pump can serve several chambers. "
            "Valves describe this schematic; they do not operate hardware or calculate pressure.")
        note.setWordWrap(True)
        outer.addWidget(note)
        self.tabs = QTabWidget()
        self.pumps = self._table(("Pump name", "Type"))
        self.auxiliaries = self._table(("Auxiliary chamber / manifold", "Location reference"))
        self.lines = self._table(("From", "To", "Service", "Valve name", "Valve state"))
        for title, table, add, remove in (
            ("Pumps", self.pumps, self.add_pump, lambda: self._remove_node(self.pumps, "pump")),
            ("Auxiliary chambers", self.auxiliaries, self.add_auxiliary,
             lambda: self._remove_node(self.auxiliaries, "aux")),
            ("Pipes and valves", self.lines, self.add_line, self.remove_line),
        ):
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.addWidget(table, 1)
            buttons = QHBoxLayout()
            add_button, remove_button = QPushButton("Add"), QPushButton("Remove selected")
            add_button.clicked.connect(lambda _checked=False, callback=add: callback())
            remove_button.clicked.connect(lambda _checked=False, callback=remove: callback())
            buttons.addWidget(add_button)
            buttons.addWidget(remove_button)
            buttons.addStretch()
            layout.addLayout(buttons)
            self.tabs.addTab(panel, title)
        outer.addWidget(self.tabs, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        footer = QHBoxLayout()
        defaults = QPushButton("Use illustrative connections")
        defaults.setToolTip("Replace this draft with the illustrative layout. Cancel discards all draft edits.")
        defaults.clicked.connect(lambda: self.set_layout(default_pumping_layout(self.regions)))
        footer.addWidget(defaults)
        footer.addStretch()
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        footer.addWidget(self.buttons)
        outer.addLayout(footer)
        self.pumps.itemChanged.connect(self._refresh_endpoints)
        self.auxiliaries.itemChanged.connect(self._refresh_endpoints)
        self.set_layout(effective_pumping_layout(state.vacuum_map))

    @staticmethod
    def _table(labels):
        table = QTableWidget(0, len(labels))
        table.setHorizontalHeaderLabels(labels)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setMinimumSectionSize(100)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        return table

    @staticmethod
    def _key(table, row):
        return table.verticalHeaderItem(row).data(Qt.ItemDataRole.UserRole)

    @classmethod
    def _new_key(cls, table, prefix):
        keys = {cls._key(table, row) for row in range(table.rowCount())}
        number = 1
        while f"{prefix}_{number}" in keys:
            number += 1
        return f"{prefix}_{number}"

    @staticmethod
    def _row(table, key):
        row = table.rowCount()
        table.insertRow(row)
        header = QTableWidgetItem(str(row + 1))
        header.setData(Qt.ItemDataRole.UserRole, key)
        table.setVerticalHeaderItem(row, header)
        return row

    def set_layout(self, layout):
        self._filling = True
        self._base = deepcopy(layout)
        try:
            for table in (self.pumps, self.auxiliaries, self.lines):
                table.setRowCount(0)
            for pump in layout["pumps"]:
                self.add_pump(pump)
            for chamber in layout["auxiliaries"]:
                self.add_auxiliary(chamber)
            for line in layout["lines"]:
                self.add_line(line)
        finally:
            self._filling = False
        self.status.clear()

    def add_pump(self, value=None):
        value = value or {"key": self._new_key(self.pumps, "pump"), "name": "New pump", "kind": "scroll"}
        row = self._row(self.pumps, value["key"])
        self.pumps.setItem(row, 0, QTableWidgetItem(value["name"]))
        self.pumps.setCellWidget(row, 1, _choice(PUMP_TYPES, value["kind"]))
        self._refresh_endpoints()
        self.pumps.selectRow(row)

    def add_auxiliary(self, value=None):
        value = value or {"key": self._new_key(self.auxiliaries, "chamber"),
                          "name": "New chamber", "anchor": "sample"}
        row = self._row(self.auxiliaries, value["key"])
        self.auxiliaries.setItem(row, 0, QTableWidgetItem(value["name"]))
        location = _choice(((key, key) for key in self.anchors), value["anchor"])
        location.setToolTip("Associates this off-axis chamber with a column reference; no beam-path gas region is added.")
        self.auxiliaries.setCellWidget(row, 1, location)
        self._refresh_endpoints()
        self.auxiliaries.selectRow(row)

    def _endpoints(self):
        options = [(region.name, f"region:{region.key}") for region in self.regions]
        for table, prefix, label in ((self.auxiliaries, "aux", "Chamber"), (self.pumps, "pump", "Pump")):
            for row in range(table.rowCount()):
                item = table.item(row, 0)
                if item is not None:
                    options.append((f"{label}: {item.text()}", f"{prefix}:{self._key(table, row)}"))
        return options

    def _refresh_endpoints(self, *_args):
        if self._filling:
            return
        options = self._endpoints()
        for row in range(self.lines.rowCount()):
            for column in (0, 1):
                previous = self.lines.cellWidget(row, column)
                value = previous.currentData() if previous is not None else None
                self.lines.setCellWidget(row, column, _choice(options, value))

    def add_line(self, value=None):
        options = self._endpoints()
        first_pump = next((data for _name, data in options if data.startswith("pump:")), "")
        value = value or {
            "key": self._new_key(self.lines, "pipe"),
            "source": options[0][1] if options else "",
            "target": first_pump, "role": "high_vacuum", "valve": "V", "open": False,
        }
        row = self._row(self.lines, value["key"])
        self.lines.setCellWidget(row, 0, _choice(options, value["source"]))
        self.lines.setCellWidget(row, 1, _choice(options, value["target"]))
        self.lines.setCellWidget(row, 2, _choice(LINE_ROLES, value["role"]))
        valve = QTableWidgetItem(value["valve"])
        valve.setToolTip("Leave blank for a permanently connected pipe; otherwise give the valve a display name.")
        self.lines.setItem(row, 3, valve)
        self.lines.setCellWidget(row, 4, _choice((("Open / connected", True), ("Closed / isolated", False)), value["open"]))
        self.lines.selectRow(row)

    def _remove_node(self, table, prefix):
        row = table.currentRow()
        if row < 0:
            return
        endpoint = f"{prefix}:{self._key(table, row)}"
        table.removeRow(row)
        removed = 0
        for index in range(self.lines.rowCount()-1, -1, -1):
            if any(self.lines.cellWidget(index, column).currentData() == endpoint for column in (0, 1)):
                self.lines.removeRow(index)
                removed += 1
        self._refresh_endpoints()
        self.status.setText(f"Removed the node and {removed} attached pipes from this draft.")

    def remove_line(self):
        row = self.lines.currentRow()
        if row >= 0:
            self.lines.removeRow(row)

    def candidate_layout(self):
        layout = deepcopy(self._base)
        layout["pumps"] = [
            {"key": self._key(self.pumps, row), "name": self.pumps.item(row, 0).text().strip(),
             "kind": self.pumps.cellWidget(row, 1).currentData()}
            for row in range(self.pumps.rowCount())
        ]
        layout["auxiliaries"] = [
            {"key": self._key(self.auxiliaries, row), "name": self.auxiliaries.item(row, 0).text().strip(),
             "anchor": self.auxiliaries.cellWidget(row, 1).currentData()}
            for row in range(self.auxiliaries.rowCount())
        ]
        layout["lines"] = [
            {"key": self._key(self.lines, row), "source": self.lines.cellWidget(row, 0).currentData(),
             "target": self.lines.cellWidget(row, 1).currentData(),
             "role": self.lines.cellWidget(row, 2).currentData(),
             "valve": self.lines.item(row, 3).text().strip(),
             "open": self.lines.cellWidget(row, 4).currentData()}
            for row in range(self.lines.rowCount())
        ]
        validate_pumping_layout(layout, [region.key for region in self.regions])
        return layout

    def select_item(self, endpoint="", line_key=""):
        if line_key:
            table, tab, key = self.lines, 2, line_key
        elif endpoint.startswith("pump:"):
            table, tab, key = self.pumps, 0, endpoint.split(":", 1)[1]
        elif endpoint.startswith("aux:"):
            table, tab, key = self.auxiliaries, 1, endpoint.split(":", 1)[1]
        else:
            return
        self.tabs.setCurrentIndex(tab)
        for row in range(table.rowCount()):
            if self._key(table, row) == key:
                table.selectRow(row)
                table.scrollTo(table.model().index(row, 0))
                break

    def accept(self):
        try:
            self.result_layout = self.candidate_layout()
        except (ValueError, TypeError) as exc:
            self.status.setText(f"Not saved: {exc}")
            return
        super().accept()
