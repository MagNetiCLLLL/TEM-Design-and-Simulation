"""Explicit fixed-envelope multipole windings and signed physical currents."""
from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QPlainTextEdit, QVBoxLayout

from temsim.gui.coil_winding_dialog import _finite_number, _label
from temsim.gui.transverse_winding_dialog import _roundtrip_text, _signed_number
from temsim.part_model_document import MULTIPOLE_WINDING_SOURCE_NOTE
from temsim.multipole_current import (
    MATRIX_FIELD, BIAS_FIELD, electrical_channel_count, validate_multipole_current_inputs,
)
from temsim.multipole_winding import (
    INSULATION_FIELD, STATUS_FIELD, NOTE_FIELD, area_field, turn_field,
    multipole_winding_definition, multipole_winding_envelopes,
)


class MultipoleWindingConfigurationDialog(QDialog):
    """Stage one validated draft transaction, without fitting or live mutation."""

    def __init__(self,part,owner,by_key,*,runtime_values=None,commit=None,parent=None):
        super().__init__(parent)
        self.part,self.owner,self.by_key=deepcopy(part),deepcopy(owner),deepcopy(by_key)
        self.runtime_values=deepcopy(runtime_values or {})
        self._commit,self._values=commit,None
        self.nfield,self.afield=turn_field(part),area_field(part)
        self.channel_count=electrical_channel_count(part)
        self.channel_labels=['Main winding'] if self.channel_count==1 else ['Normal winding group','Skew winding group']
        self.order=3 if str(part.get('key','')).endswith('_hexapole') else 2
        self.unit='m⁻³' if self.order==3 else 'm⁻²'
        self.setObjectName('multipoleWindingConfigurationDialog')
        self.setWindowTitle('Configure multipole winding and currents')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(730,770)
        layout=QVBoxLayout(self)
        coils=multipole_winding_envelopes(part,by_key)
        layout.addWidget(_label(f"{part.get('name',part['key'])}\n"
            f"Order {self.order}; physical coils: {len(coils)}; electrical channels: {self.channel_count}. "
            'Fixed winding allocation; ideal vacuum currents exclude magnetic cores and pole pieces.'))
        layout.addWidget(_label('Engineering winding reconstruction, not measured or OEM geometry. '
            'Auxiliary winding-only CAD retains existing cartridge ownership, clear bores and transport boundaries.'))
        form=QFormLayout()
        self.turn_count=QLineEdit('' if self.nfield not in part else str(part[self.nfield]))
        self.turn_count.setObjectName('multipoleWindingTurnCountPerCoil')
        self.wire_cross_section=QLineEdit(_roundtrip_text(part.get(self.afield)))
        self.wire_cross_section.setObjectName('multipoleWindingMetalArea')
        self.wire_cross_section.setPlaceholderText('Optional conductor metal area; excludes insulation')
        self.insulation_thickness=QLineEdit(_roundtrip_text(part.get(INSULATION_FIELD)))
        self.insulation_thickness.setObjectName('multipoleWindingInsulationThickness')
        self.insulation_thickness.setPlaceholderText('Optional radial insulation around each conductor')
        self.wire_diameter=_label('Unknown until conductor metal area is supplied')
        self.wire_diameter.setObjectName('multipoleWindingDerivedDiameter')
        for label,widget in (('Turns per physical coil',self.turn_count),
            ('Conductor metal cross-section (mm²)',self.wire_cross_section),
            ('Wire insulation radial thickness (mm)',self.insulation_thickness),
            ('Derived bare-conductor diameter',self.wire_diameter)):
            form.addRow(label,widget)
        layout.addLayout(form)
        layout.addWidget(_label('All requested closed turns and remaining insulation fit the original winding envelope. '
            'Impossible packing fails. Known turns without metal area retain bulk CAD. Clearing turns and sizes '
            'removes this winding recipe and electrical calibration. Current inputs apply only to Ideal prescribed-current '
            'coils in vacuum; no edit changes native optical commands, resizes the envelope or automatically refits current.'))
        bias_form=QFormLayout();self.current_bias=[]
        bias=part.get(BIAS_FIELD,[0.]*self.channel_count)
        for index,label in enumerate(self.channel_labels):
            edit=QLineEdit(_roundtrip_text(bias[index] if len(bias)==self.channel_count else None))
            edit.setObjectName(f'multipoleCurrentBias{index}')
            edit.setEnabled(MATRIX_FIELD in part)
            self.current_bias.append(edit);bias_form.addRow(f'{label} additional current (A)',edit)
        layout.addLayout(bias_form)
        self.reference=QPlainTextEdit();self.reference.setReadOnly(True)
        self.reference.setObjectName('multipoleCurrentCalibrationReference');self.reference.setMaximumHeight(165)
        layout.addWidget(self.reference)
        self.packing=QPlainTextEdit();self.packing.setReadOnly(True)
        self.packing.setObjectName('multipoleWindingPackingReadback');self.packing.setMinimumHeight(100)
        layout.addWidget(self.packing,1)
        self.validation=_label();self.validation.setObjectName('multipoleWindingValidation')
        layout.addWidget(self.validation)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Apply|QDialogButtonBox.StandardButton.Cancel)
        self.apply_button=buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText('Apply to draft');self.apply_button.clicked.connect(self._apply)
        buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        for edit in (self.turn_count,self.wire_cross_section,self.insulation_thickness,*self.current_bias):
            edit.textChanged.connect(self._refresh)
        self._refresh()

    def set_runtime_values(self,values):
        self.runtime_values=deepcopy(values or {});self._refresh()

    def values(self):
        if self._values is None:raise ValueError('Enter a valid multipole winding configuration')
        return deepcopy(self._values)

    def _reference_text(self,bias):
        if MATRIX_FIELD not in self.part:
            return ('Electrical current calibration is undefined. Native multipole coefficients remain in '
                f'{self.unit}; no conductor current is guessed or fitted by this dialog.')
        data=validate_multipole_current_inputs(self.part,self.by_key)
        lines=[f"Saved reference: {data['reference_energy_ev']/1000:.9g} keV; {data['status'].replace('_',' ')}",
            f'Fixed matrix A/({self.unit}); rows = physical currents, columns = native coefficients:',
            'Rows: '+', '.join(self.channel_labels)]
        lines.extend('  '.join(f'{value:.10g}' for value in row) for row in data['matrix_a_per_unit'])
        orientation=self.part.get('ideal_multipole_calibration_reference_orientation_rad')
        if orientation is not None:
            lines.append(f'Saved native orientation: {float(orientation):.10g} rad (read-only reference)')
        source={**self.owner,**self.runtime_values}
        fields=[f'strength_m{self.order}']
        if self.channel_count==2:fields.append(f'skew_strength_m{self.order}')
        if self.channel_count==1 and float(source.get(f'skew_strength_m{self.order}',0.))!=0.:
            lines.append('Current readback unavailable: nonzero main-HP skew drive has no second physical winding basis.')
        elif fields[0] in source:
            commands=[float(source.get(field,0.)) for field in fields]
            if self.channel_count==2 and fields[1] not in source:
                lines.append('The scalar native owner supplies zero skew command; the orthogonal winding remains independently biasable.')
            if self.order==3 and self.channel_count==2:
                reference=float(self.part.get('ideal_multipole_calibration_reference_orientation_rad',0.))
                angle=float(source.get('orientation_rad',reference))
                phase=3*(angle-reference)
                normal,skew=commands
                commands=[normal*math.cos(phase)-skew*math.sin(phase),
                          normal*math.sin(phase)+skew*math.cos(phase)]
            if all(math.isfinite(value) for value in commands):
                currents=[sum(value*u for value,u in zip(row,commands))+b
                    for row,b in zip(data['matrix_a_per_unit'],bias)]
                lines.append('Current readback (A): '+', '.join(f'{value:.10g}' for value in currents))
        lines.append('Biases add amperes to saved native-command currents. Reference energy never rescales actual currents.')
        return '\n'.join(lines)

    def _refresh(self):
        self._values=None
        self.wire_diameter.setText('Unknown until conductor metal area is supplied')
        try:
            nt,at,it=(edit.text().strip() for edit in (self.turn_count,self.wire_cross_section,self.insulation_thickness))
            if not nt:
                if at or it:raise ValueError('Conductor and insulation size require a known per-coil turn count')
                self._values=dict(turn_count_per_coil=None,wire_cross_section_mm2=None,insulation_thickness_mm=None)
                self.packing.setPlainText('Clearing removes the native multipole winding and its electrical calibration.')
                self.reference.setPlainText(self._reference_text([0.]*self.channel_count))
            else:
                try:turns=int(nt)
                except (ValueError,OverflowError) as error:
                    raise ValueError('Turns per physical coil must be a positive integer') from error
                if turns<=0:raise ValueError('Turns per physical coil must be a positive integer')
                area=_finite_number(at,'Conductor metal area',positive=True) if at else None
                insulation=_finite_number(it,'Radial insulation thickness') if it else None
                candidate={**self.part,self.nfield:turns,STATUS_FIELD:'engineering_reconstruction_not_oem',
                    NOTE_FIELD:MULTIPOLE_WINDING_SOURCE_NOTE}
                for field,value in ((self.afield,area),(INSULATION_FIELD,insulation)):
                    candidate.pop(field,None)
                    if value is not None:candidate[field]=value
                d=multipole_winding_definition(candidate,self.by_key)
                self._values=dict(turn_count_per_coil=turns,wire_cross_section_mm2=area,insulation_thickness_mm=insulation)
                bias=[_signed_number(edit.text(),label+' current bias') for edit,label in zip(self.current_bias,self.channel_labels)]
                if MATRIX_FIELD in self.part:self._values['current_bias_a']=bias
                self.reference.setPlainText(self._reference_text(bias))
                lines=[f"Physical coils: {d['coil_count']}; turns per coil: {d['turn_count_per_coil']}; total turns: {d['total_turn_count']}"]
                if d['detailed_cad']:
                    self.wire_diameter.setText(f"{d['wire_diameter_mm']:.10g} mm (calculated)")
                    pitches=[p['pitch_mm'] for p in d['packing']]
                    lines.extend([f'Centre pitch: {min(pitches):.9g} to {max(pitches):.9g} mm',
                        'Every turn and remaining insulation stays within its original allocation.'])
                else:lines.append('Conductor metal area is unknown; bulk winding CAD is retained.')
                if turns!=self.part.get(self.nfield) and MATRIX_FIELD in self.part:
                    lines.append('Turn edit makes the original field-strength fit stale; numeric currents are retained.')
                self.packing.setPlainText('\n'.join(lines))
            self.validation.setText('Valid draft. Apply is one Undo operation; Save and live-instrument loading are separate.')
            self.validation.setStyleSheet('')
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            self._values=None;self.validation.setText(str(error));self.validation.setStyleSheet('color: #c65b5b')
            self.packing.setPlainText('The preceding draft and live instrument remain unchanged.')
        self.apply_button.setEnabled(self._values is not None)

    def _apply(self):
        if self._values is None:return
        try:
            if self._commit is not None:self._commit(self.values())
        except (ValueError,TypeError,RuntimeError) as error:
            self.validation.setText(str(error));self.validation.setStyleSheet('color: #c65b5b');return
        self.accept()
