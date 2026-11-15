"""Explicit CAD-only radial winding inputs for the two probe hexapoles."""
from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QPlainTextEdit, QVBoxLayout

from temsim.cad_probe_winding import (
    AREA_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD, probe_winding_dimensions,
)
from temsim.gui.coil_winding_dialog import _finite_number, _label, _number_text
from temsim.probe_hexapole_geometry import probe_hexapole_dimensions
from temsim.part_model_document import PROBE_WINDING_SOURCE_NOTE


SOURCE_NOTE = PROBE_WINDING_SOURCE_NOTE


class ProbeWindingConfigurationDialog(QDialog):
    """Configure all six equivalent coils without creating an excitation."""

    def __init__(self, part, *, runtime_values=None, commit=None, parent=None):
        super().__init__(parent)
        self.part = deepcopy(part)
        self.owner = self.part  # The editor's shared runtime-update contract.
        self.runtime_values = deepcopy(runtime_values or {})
        self._commit = commit
        self._values = None
        self.setObjectName('probeWindingConfigurationDialog')
        self.setWindowTitle('Configure probe hexapole winding')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(620, 550)
        d = probe_hexapole_dimensions(part)
        layout = QVBoxLayout(self)
        layout.addWidget(_label(f"{part.get('name', part['key'])}\nSix radial-axis excitation windings; CAD inputs only."))
        self.envelope_label = _label(
            f"Fixed winding envelope per coil: radial length {d['coil_radial_length_mm']:g} mm; "
            f"tangential width {d['coil_width_mm']:g} mm; axial height {d['coil_height_mm']:g} mm.\n"
            f"Existing core window: {d['coil_window_width_mm']:g} × {d['coil_window_height_mm']:g} mm.")
        layout.addWidget(self.envelope_label)
        self.turn_count = QLineEdit()
        self.turn_count.setObjectName('probeWindingTurnCountPerCoil')
        self.turn_count.setPlaceholderText('Not specified; positive integer for each of six coils')
        self.wire_cross_section = QLineEdit()
        self.wire_cross_section.setObjectName('probeWindingWireCrossSection')
        self.wire_cross_section.setPlaceholderText('Not specified; conductor metal area, excluding insulation')
        self.wire_diameter = _label('Unknown until conductor metal area is entered')
        self.wire_diameter.setObjectName('probeWindingDerivedDiameter')
        self.turn_count.setText('' if TURN_FIELD not in part else str(part[TURN_FIELD]))
        source = part.get('parameter_metadata', {}).get(TURN_FIELD, {})
        self.turn_count.setToolTip(str(source.get('source_note', part.get(NOTE_FIELD, ''))))
        if source.get('source_kind') == 'estimated':
            layout.addWidget(_label('Saved turn count: engineering estimate; not measured or OEM winding data.'))
        self.wire_cross_section.setText(_number_text(part.get(AREA_FIELD)))
        form = QFormLayout()
        form.addRow('Conductor turns per coil', self.turn_count)
        form.addRow('Conductor metal cross-section (mm²)', self.wire_cross_section)
        form.addRow('Derived round-wire diameter', self.wire_diameter)
        layout.addLayout(form)
        layout.addWidget(_label(
            'Known turns may be saved with metal area left blank; bulk winding CAD is retained. '
            'Enter metal area to generate individual turns, or clear both inputs to remove the winding definition. '
            'Insulation fills the remaining original winding allocation automatically. The pole pins, bore, '
            'yoke, supports and outer winding envelope remain fixed. Native multipole strength and all '
            'optical controls remain unchanged; no current calibration or wire rating is inferred.'))
        self.packing = QPlainTextEdit()
        self.packing.setReadOnly(True)
        self.packing.setObjectName('probeWindingPackingReadback')
        self.packing.setMinimumHeight(160)
        layout.addWidget(self.packing, 1)
        self.validation = _label()
        self.validation.setObjectName('probeWindingValidation')
        layout.addWidget(self.validation)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText('Apply to draft')
        self.apply_button.clicked.connect(self._apply)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.turn_count.textChanged.connect(self._refresh)
        self.wire_cross_section.textChanged.connect(self._refresh)
        self._refresh()

    def set_runtime_values(self, values):
        self.runtime_values = deepcopy(values or {})

    def values(self):
        if self._values is None:
            raise ValueError('Enter valid per-coil turns with optional metal area, or clear both inputs')
        return deepcopy(self._values)

    def _refresh(self):
        self._values = None
        self.wire_diameter.setText('Unknown until conductor metal area is entered')
        turns_text = self.turn_count.text().strip()
        area_text = self.wire_cross_section.text().strip()
        try:
            if not turns_text and not area_text:
                self._values = dict(turn_count_per_coil=None, wire_cross_section_mm2=None)
                self.packing.setPlainText('Individual turns are undefined. The existing bulk copper and passive supports are retained.')
            else:
                if not turns_text:
                    raise ValueError('Conductor metal area requires a known per-coil turn count')
                try:
                    turns = int(turns_text)
                except (ValueError, OverflowError) as error:
                    raise ValueError('Per-coil turn count must be a positive integer') from error
                if turns <= 0:
                    raise ValueError('Per-coil turn count must be a positive integer')
                area = _finite_number(area_text, 'Conductor metal area', positive=True) if area_text else None
                candidate = {**self.part, TURN_FIELD: turns,
                             STATUS_FIELD: 'engineering_reconstruction_not_oem', NOTE_FIELD: SOURCE_NOTE}
                candidate.pop(AREA_FIELD, None)
                if area is not None:
                    candidate[AREA_FIELD] = area
                d = probe_winding_dimensions(candidate)
                if d is None:
                    self.packing.setPlainText(
                        f'Physical coil count: 6\nTurns per coil: {turns}\nTotal conductor turns: {6*turns}\n'
                        'Conductor metal area and current are unspecified. Bulk copper and passive supports are retained.\n'
                        'Enter conductor metal area to validate packing and generate individual turns. '
                        'The turn count is fixed until explicitly edited; geometry edits do not re-estimate it.')
                else:
                    self.wire_diameter.setText(f"{d['wire_diameter_mm']:.9g} mm (calculated)")
                    self.packing.setPlainText(
                    f"Physical coil count: 6\nTurns per coil: {d['turn_count_per_coil']}\n"
                    f"Total conductor turns: {d['total_turn_count']}\n"
                    f"Conductor metal area per turn: {d['wire_cross_section_mm2']:.9g} mm²\n"
                    f"Radial layers: {d['radial_layers']}; nested offset rings: {d['offset_rings']}\n"
                    f"Centre pitch: {d['centre_pitch_mm']:.9g} mm\n"
                    f"Interwire gap: {d['interwire_gap_mm']:.9g} mm\n"
                    f"Metal volume per coil: {d['conductor_volume_per_coil_mm3']:.9g} mm³\n"
                    f"Remaining insulation per coil: {d['remaining_insulation_volume_per_coil_mm3']:.9g} mm³\n"
                    'The original radial, tangential and axial winding bounds are unchanged.')
                self._values = dict(turn_count_per_coil=turns, wire_cross_section_mm2=area)
            self.validation.setText('Valid CAD-only configuration. Apply stages one Undo operation; Save remains separate.')
            self.validation.setStyleSheet('')
        except (ValueError, TypeError, OverflowError) as error:
            self.packing.setPlainText('The preceding draft and native optical strength remain unchanged.')
            self.validation.setText(str(error))
            self.validation.setStyleSheet('color: #c65b5b')
        self.apply_button.setEnabled(self._values is not None)

    def _apply(self):
        if self._values is None:
            return
        try:
            if self._commit is not None:
                self._commit(self.values())
        except (ValueError, TypeError, RuntimeError) as error:
            self.validation.setText(str(error))
            self.validation.setStyleSheet('color: #c65b5b')
            return
        self.accept()
