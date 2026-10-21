"""Readable material-pair results from the existing exact geometry checker."""
from copy import deepcopy

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHeaderView, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)


class MechanicalOverlapPanel(QWidget):
    component_requested = Signal(str, str)
    check_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.auto_check = QCheckBox("Check overlaps after edits")
        self.auto_check.setChecked(True)
        self.auto_check.setToolTip("Run the exact material check in the background after editing pauses. No optical settings are changed.")
        self.check_now = QPushButton("Check now")
        self.check_now.clicked.connect(self.check_requested.emit)
        self.status = QLabel("Edit a dimension or choose Check now to list overlapping components.")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.filter = QComboBox()
        self.filter.setObjectName("mechanicalOverlapFilter")
        self.filter.addItem("Material overlaps", "overlaps")
        self.filter.addItem("Surface contacts", "contacts")
        self.filter.addItem("Uncertain / unresolved", "uncertain")
        self.filter.setToolTip("Choose which geometry results to inspect. Contacts and unresolved geometry are not confirmed material overlaps.")
        self.filter.currentIndexChanged.connect(self._refresh_table)
        self.table = QTableWidget(0, 6)
        self.table.setObjectName("mechanicalOverlapTable")
        self.table.setHorizontalHeaderLabels(("Component A", "Region A", "Component B", "Region B", "Intersection (mm³)", "Status"))
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.cellDoubleClicked.connect(self._activate)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.addWidget(self.auto_check)
        layout.addWidget(self.check_now)
        layout.addWidget(self.status)
        layout.addWidget(self.filter)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.note)
        self._current = False
        self._records = {}
        self._names = {}
        self._refresh_table()

    def pending(self, text="Dimensions changed. Waiting to check the current draft…"):
        self._current = False
        self._records = {}
        self.table.setRowCount(0)
        self.status.setText(text)

    def set_report(self, report, *, names=None, current=True):
        self._names = dict(names or {})
        self._current = bool(current)
        self._records = {}
        self.table.setRowCount(0)
        if not current:
            self.status.setText("Previous draft only — results are stale. Check the current dimensions.")
            return
        assemblies = report.get("geometry_report", {}).get("assemblies", ())
        pairs = [pair for assembly in assemblies for pair in assembly.get("collisions", ())]
        contacts = sum(len(a.get("contacts", ())) for a in assemblies)
        unresolved = sum(len(a.get("uncertain_occupancy_intersections", ())) +
                         len(a.get("sub_tolerance_intersections", ())) for a in assemblies)
        unknown = sum(len(a.get("unresolved_solids", ())) for a in assemblies)
        self._records = {
            "overlaps": [(deepcopy(pair), "Material overlap") for pair in pairs],
            "contacts": [(deepcopy(pair), "Surface contact") for assembly in assemblies
                         for pair in assembly.get("contacts", ())],
            "uncertain": [(deepcopy(pair), label) for assembly in assemblies
                          for key, label in (("uncertain_occupancy_intersections", "Uncertain occupancy"),
                                             ("sub_tolerance_intersections", "Below volume tolerance"))
                          for pair in assembly.get(key, ())]
                         + [(dict(first=deepcopy(solid)), "Unresolved solid") for assembly in assemblies
                            for solid in assembly.get("unresolved_solids", ())],
        }
        self._refresh_table()
        self.status.setText(
            f"Current draft: {len(pairs)} material overlap pairs; {contacts} surface contacts; "
            f"{unresolved} uncertain/tiny intersections; {unknown} unresolved solids.\n" +
            report.get("scope", "Checked material geometry only."))

    def _refresh_table(self, *_):
        kind = self.filter.currentData()
        notes = {
            "overlaps": "Declared material intersections above the reported tolerance. Double-click either component to select it.",
            "contacts": "Surface contacts have no resolved intersection volume; they are not material overlaps. Double-click either component to select it.",
            "uncertain": "Envelope intersections and tiny volumes do not establish material overlap. Unresolved solids have no measured intersection volume (—). Double-click a component to select it.",
        }
        self.note.setText(notes[kind])
        records = self._records.get(kind, ()) if self._current else ()
        self.table.setRowCount(len(records))
        for row, (pair, status) in enumerate(records):
            first, second = pair["first"], pair.get("second")
            volume = pair.get("intersection_volume_mm3")
            values = (self._names.get(first["key"], first["key"]), first.get("region", "—"),
                      self._names.get(second["key"], second["key"]) if second else "—",
                      second.get("region", "—") if second else "—",
                      f"{volume:.7g}" if volume is not None else "—", status)
            for column, value in enumerate(values):
                side = first if column < 2 or column == 5 else second
                item = QTableWidgetItem(str(value))
                if side is not None:
                    item.setData(Qt.ItemDataRole.UserRole, (side["key"], side.get("region", "body")))
                    item.setToolTip(f'{side["key"]} / {side.get("region", "—")}\nModule: {side.get("module_key", "")}\n{status}')
                if column == 4:
                    item.setToolTip("Intersection of the reported shapes; consult Status before interpreting it as material overlap.")
                self.table.setItem(row, column, item)

    def _activate(self, row, column):
        if self._current:
            item = self.table.item(row, column)
            if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None:
                self.component_requested.emit(*item.data(Qt.ItemDataRole.UserRole))
