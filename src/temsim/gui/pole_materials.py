"""Constitutive material choices on the installed objective and correctors."""
from collections import defaultdict
from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFormLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from temsim import input_io
from temsim.component_operations import PartChangeSet
from temsim.gui.input_policy import WheelSafeComboBox
from temsim.gui.material_reference import MaterialReferenceLabel
from temsim.manifest_editor import ManifestTarget
from temsim.part_materials import (
    NONMAGNETIC_STEEL, POLE_GRADES, hardware_regions, material_catalog, material_for_region,
    update_material_regions,
)


@dataclass(frozen=True)
class MaterialTarget:
    group: str
    role: str
    key: str
    region: str
    label: str


def installed_targets(assembly):
    if assembly is None:
        return ()
    targets = []
    keys = {part.key for part in assembly.parts}
    for part in assembly.parts:
        row, owner = part.data, part.parent_key or ""
        if part.key.startswith("objective_"):
            group = "Objective"
        elif (part.key.startswith("probe_") or owner == "adapter_lens"
              or (owner == "mini_condenser" and "probe_hp1_hexapole" in keys)):
            group = "Probe corrector"
        elif part.key.startswith("image_"):
            group = "Image corrector"
        else:
            continue
        regions = hardware_regions(row)
        role = row.get("mechanical_part_role")
        pairs = []
        if "pole_pins" in regions:
            pairs = [("Pole pieces", "pole_pins", "six pole pins"),
                     ("Return yokes", "return_yoke", "outer yoke")]
            if "pole_root_sleeves" in regions:
                pairs.append(("Sleeves", "pole_root_sleeves", "six pole-root sleeves"))
        elif row.get("mechanical_profile") == "magnetic_pole_piece":
            pairs = [("Pole pieces", "body", "")]
            if "terminal_tube" in regions:
                pairs.append(("Terminal vacuum tube", "terminal_tube", f'{row["terminal_tube_length_mm"]:g} mm wall'))
        elif role in {"pole_sleeves", "objective_pole_sleeves", "radial_return_plates", "objective_return_plates"}:
            family = "Sleeves" if "sleeves" in role else "Return plates"
            pairs = [(family, side, side) for side in ("upper", "lower")]
        elif role == "auxiliary_return_sleeve":
            pairs = [("Sleeves", "body", "")]
        elif role == "yoke" and (owner == "objective_lens" or owner == "adapter_lens"
                                  or owner.startswith(("probe_", "image_"))):
            sides = ("upper", "lower") if owner == "objective_lens" else ("body",)
            pairs = [("Return yokes", side, side if side != "body" else "") for side in sides]
        for family, region, detail in pairs:
            label = f"{part.name} ({detail})" if detail else part.name
            targets.append(MaterialTarget(group, family, part.key, region, label))
    return tuple(targets)


class PoleMaterialsPage(QWidget):
    save_requested = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parts = {}
        self.targets = ()
        self._matching = []
        self._assembly = None
        self._source_bytes = {}
        self._archived_assignments = {}
        self._archived_selection = None
        self.read_only = True
        self.group, self.role, self.target, self.material = (WheelSafeComboBox() for _ in range(4))
        for widget, name in zip((self.group, self.role, self.target, self.material),
                                ("magneticAssembly", "magneticRole", "magneticPart", "poleMaterial")):
            widget.setObjectName(name)
        self.catalog = {r["material_key"]: r for r in material_catalog()}
        for key in (*POLE_GRADES, NONMAGNETIC_STEEL, "femm_pure_iron"):
            if key in self.catalog:
                self.material.addItem(self.catalog[key]["label"], key)
        self.current, self.scope = QLabel(), QLabel()
        self.source = MaterialReferenceLabel()
        for label in (self.current, self.scope):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.save = QPushButton("Save material to installed column")
        self.save.clicked.connect(self._save)
        self.solve_corrector = QPushButton("Calculate probe corrector field…")
        self.solve_corrector.clicked.connect(self._solve_corrector)
        form = QFormLayout()
        for label, control in (("Assembly", self.group), ("Component", self.role), ("Applies to", self.target),
                               ("Current material", self.current), ("Material", self.material)):
            form.addRow(label, control)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.source)
        layout.addWidget(self.scope)
        layout.addWidget(self.save, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.solve_corrector, 0, Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()
        self.group.currentIndexChanged.connect(self._roles)
        self.role.currentIndexChanged.connect(self._targets)
        self.target.currentIndexChanged.connect(self._current)
        self.material.currentIndexChanged.connect(self._source)
        self._current()

    @staticmethod
    def _populate(box, options):
        previous = box.currentData()
        box.blockSignals(True)
        box.clear()
        for label, value in options:
            box.addItem(label, value)
        box.setCurrentIndex(max(0, box.findData(previous)))
        box.blockSignals(False)

    def set_assembly(self, assembly, *, read_only=False):
        if self._assembly is assembly and self.read_only == read_only:
            return
        self._assembly = assembly
        self.parts = {p.key: p for p in assembly.parts} if assembly else {}
        self.targets = installed_targets(assembly)
        self.read_only = read_only
        # Later selector changes run outside the archive's input scope.
        self._archived_assignments = {
            (t.key, t.region): material_for_region(self.parts[t.key], t.region)
            for t in self.targets
        } if read_only else {}
        self.material.setEnabled(bool(self.targets) and not read_only)
        self._source_bytes = {}
        if self.targets and not read_only:
            for path in {self.parts[t.key].source_file for t in self.targets}:
                self._source_bytes[path] = input_io.read_bytes(assembly.root / path)
        groups = {t.group for t in self.targets}
        self._populate(self.group, [(g, g) for g in ("Objective", "Probe corrector", "Image corrector") if g in groups])
        self._roles()

    def _roles(self):
        families = {t.role for t in self.targets if t.group == self.group.currentData()}
        self._populate(self.role, [(r, r) for r in ("Pole pieces", "Sleeves", "Return plates", "Return yokes",
                                                     "Terminal vacuum tube") if r in families])
        self._targets()

    def _targets(self):
        self._matching = [t for t in self.targets if t.group == self.group.currentData() and t.role == self.role.currentData()]
        choices = [("All listed components", "*")] if len(self._matching) > 1 else []
        self._populate(self.target, choices + [(t.label, f"{t.key}/{t.region}") for t in self._matching])
        self._current()

    def _selected(self):
        scope = self.target.currentData()
        return [t for t in self._matching if scope in ("*", f"{t.key}/{t.region}")]

    def _current(self):
        selected = self._selected()
        rows = []
        for target in selected:
            row = (self._archived_assignments[target.key, target.region] if self.read_only
                   else material_for_region(self.parts[target.key], target.region))
            if row is None and target.region == "terminal_tube":
                row = self.catalog[NONMAGNETIC_STEEL]
            if (row is None and not self.read_only and target.group == "Probe corrector"
                    and target.key.startswith(("probe_hp1_hexapole", "probe_hp2_hexapole",
                                               "probe_tl21_lens", "probe_tl22_lens"))):
                row = self.catalog["armco_pure_iron"]
            rows.append(row)
        materials = {row["material_key"] if row else None for row in rows}
        self._archived_selection = None
        if len(materials) == 1 and (not self.read_only or all(row == rows[0] for row in rows)):
            key, = materials
            fallback = ("Grade unspecified" if any(t.key.endswith("_hexapole") for t in selected)
                        else "Lens material setting")
            self.current.setText(rows[0]["label"] if rows[0] else fallback)
            if self.read_only:
                self._archived_selection = rows[0]
            choice = key if self.read_only else key or "armco_pure_iron"
            self.material.setCurrentIndex(self.material.findData(choice))
        else:
            self.current.setText("Mixed materials" if materials else "No installed component")
            self.material.setCurrentIndex(-1)
        self._source()

    def _source(self):
        row = (self._archived_selection if self.read_only else self.catalog.get(self.material.currentData())) or {}
        curve = row.get("bh_material")
        if curve:
            citation = curve.get("source_block", curve["label"])
            self.source.set_reference(f'{citation}\nB–H range 0–{curve["b_t"][-1]:g} T.', curve["source_url"])
        else:
            self.source.set_reference(row.get("scope", ""))
        selected = self._selected()
        has_pins = any(t.region in {"pole_pins", "pole_root_sleeves", "return_yoke"} for t in selected)
        if self.read_only:
            message = "Archived column materials are read-only." if selected else ""
        elif self.role.currentData() == "Terminal vacuum tube":
            message = "Changes only the terminal tube wall in the objective field model. The bore and upstream tube are preserved."
        else:
            message = "Round-lens geometry fields use component assignments. Nonlinear FEM uses the B–H curve."
        probe = self.group.currentData() == "Probe corrector"
        if has_pins and probe:
            message += " The 3D corrector calculation uses the six pins, root sleeves, yokes and both transfer-lens circuits."
        elif has_pins:
            message += " Hexapole assignments describe the 3D hardware. The current multipole field model does not solve its magnetic core."
        self.solve_corrector.setVisible(probe and not self.read_only and
            {"probe_hp1_hexapole","probe_hp2_hexapole","probe_tl21_lens","probe_tl22_lens"} <= self.parts.keys())
        self.scope.setText(message)
        self.save.setEnabled(bool(selected and row and not self.read_only))

    def _solve_corrector(self):
        from .corrector_field_dialog import CorrectorFieldDialog
        path = self._assembly.root / self.parts["probe_hp1_hexapole"].source_file
        CorrectorFieldDialog(path,self).exec()

    def _save(self):
        if not self.save.isEnabled():
            return
        chosen = self.catalog[self.material.currentData()]
        changes = defaultdict(dict)
        for target in self._selected():
            changes[target.key][target.region] = chosen
        updates = {}
        for key, regions in changes.items():
            updates.update(update_material_regions(self.parts[key], regions))
        # Installed objective and corrector bodies are scoped to the same column variant.
        paths = {self.parts[key].source_file for key in changes}
        path, = paths
        self.save_requested.emit(ManifestTarget(path), PartChangeSet(updates, expected_source_bytes=self._source_bytes[path]))
