"""Physical transverse winding CAD and signed ideal-current bias inputs."""
from __future__ import annotations

from copy import deepcopy
import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QPlainTextEdit, QVBoxLayout,
)

from temsim.gui.coil_winding_dialog import _finite_number, _label, _number_text
from temsim.part_model_document import TRANSVERSE_WINDING_SOURCE_NOTE
from temsim.transverse_current import (
    BIAS_FIELD, MATRIX_FIELD, NOTE_FIELD as CURRENT_NOTE, REFERENCE_ENERGY_FIELD,
    STATUS_FIELD as CURRENT_STATUS, electrical_channel_count, validate_transverse_current_inputs,
)
from temsim.transverse_winding import (
    AREA_FIELD, INSULATION_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
    transverse_winding_definition, transverse_winding_envelopes,
)


def _signed_number(text,label):
    try:
        value=float(text)
    except (ValueError,TypeError,OverflowError) as error:
        raise ValueError(f'{label} must be a finite signed number') from error
    if not math.isfinite(value):
        raise ValueError(f'{label} must be a finite signed number')
    return value


def _roundtrip_text(value):
    """Use Python's shortest exact binary64 round-trip representation."""
    return '' if value is None else repr(float(value))


class TransverseWindingConfigurationDialog(QDialog):
    """Stage one validated draft operation without changing a live instrument."""

    def __init__(self,part,owner,by_key,*,runtime_values=None,commit=None,parent=None):
        super().__init__(parent)
        self.part=deepcopy(part)
        self.owner=deepcopy(owner)
        self.by_key=deepcopy(by_key)
        self.runtime_values=deepcopy(runtime_values or {})
        self._commit=commit
        self._values=None
        self.setObjectName('transverseWindingConfigurationDialog')
        self.setWindowTitle('Configure deflector / stigmator winding')
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.resize(700,760)
        layout=QVBoxLayout(self)
        coils=transverse_winding_envelopes(part)
        self.channel_count=electrical_channel_count(part)
        self.stigmator=bool(coils) and coils[0]['family']=='radial_rectangle'
        self.gun_stigmator=part.get('gun_stigmator_geometry_profile')=='eight_coil_slotted_carrier'
        self.gun_deflector=part.get('blanking_implementation')=='gun_tilt_coil_drive'
        self.command_unit=('T/m' if self.gun_stigmator else 'mT' if self.gun_deflector
                           else 'percent' if self.stigmator else 'mrad')
        self.channel_labels=(['Group X','Group Y'] if self.stigmator else
            [f'Station {i//2+1} {"XY"[i%2]}' for i in range(self.channel_count)])
        self.command_labels=(['Normal gradient','Skew gradient'] if self.gun_stigmator else
            [f'Station {i//2+1} B{"xy"[i%2]}' for i in range(self.channel_count)] if self.gun_deflector
            else ['X strength','Y strength'] if self.stigmator else self.channel_labels)
        layout.addWidget(_label(f"{part.get('name',part['key'])}\n"
            f"Physical coils: {len(coils)}; electrical channels: {self.channel_count}. "
            'Fixed winding envelope; ideal vacuum currents exclude magnetic cores and pole pieces.'))
        layout.addWidget(_label('Engineering winding reconstruction; measured OEM turns, wire size and current ratings are not asserted.'))
        form=QFormLayout()
        self.turn_count=QLineEdit('' if TURN_FIELD not in part else str(part[TURN_FIELD]))
        self.turn_count.setObjectName('transverseWindingTurnCountPerCoil')
        self.turn_count.setToolTip(str(part.get('parameter_metadata',{}).get(TURN_FIELD,{}).get(
            'source_note',part.get(NOTE_FIELD,''))))
        self.wire_cross_section=QLineEdit(_roundtrip_text(part.get(AREA_FIELD)))
        self.wire_cross_section.setObjectName('transverseWindingMetalArea')
        self.wire_cross_section.setPlaceholderText('Optional conductor metal area; excludes insulation')
        self.insulation_thickness=QLineEdit(_roundtrip_text(part.get(INSULATION_FIELD)))
        self.insulation_thickness.setObjectName('transverseWindingInsulationThickness')
        self.insulation_thickness.setPlaceholderText('Optional; radial thickness around each conductor')
        self.wire_diameter=_label('Unknown until conductor metal area is supplied')
        self.wire_diameter.setObjectName('transverseWindingDerivedDiameter')
        form.addRow('Turns per physical coil',self.turn_count)
        form.addRow('Conductor metal cross-section (mm²)',self.wire_cross_section)
        form.addRow('Wire insulation radial thickness (mm)',self.insulation_thickness)
        form.addRow('Derived bare-conductor diameter',self.wire_diameter)
        layout.addLayout(form)
        layout.addWidget(_label('Known turns may retain bulk CAD while metal area is unknown. '
            'Supplying metal area generates every closed turn and fills the remaining allocation with insulation. '
            'Clear turns and sizes to remove this winding and its electrical definition. '
            'No envelope resizing, hidden turn removal or current refitting is performed. '
            'Electrical current inputs drive the Ideal prescribed-current coils in vacuum model only; '
            'other field models retain their native optical or magnetic-field commands.'))
        self.current_bias=[]
        bias_form=QFormLayout()
        has_calibration=MATRIX_FIELD in part
        saved_bias=part.get(BIAS_FIELD,[0.]*self.channel_count)
        for i,label in enumerate(self.channel_labels):
            edit=QLineEdit(_roundtrip_text(saved_bias[i] if len(saved_bias)==self.channel_count else None))
            edit.setObjectName(f'transverseCurrentBias{i}')
            edit.setEnabled(has_calibration)
            self.current_bias.append(edit)
            bias_form.addRow(f'{label} additional current (A)',edit)
        layout.addLayout(bias_form)
        self.reference=QPlainTextEdit()
        self.reference.setObjectName('transverseCurrentCalibrationReference')
        self.reference.setReadOnly(True)
        self.reference.setMaximumHeight(145)
        layout.addWidget(self.reference)
        self.packing=QPlainTextEdit()
        self.packing.setObjectName('transverseWindingPackingReadback')
        self.packing.setReadOnly(True)
        self.packing.setMinimumHeight(100)
        layout.addWidget(self.packing,1)
        self.validation=_label()
        self.validation.setObjectName('transverseWindingValidation')
        layout.addWidget(self.validation)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Apply|QDialogButtonBox.StandardButton.Cancel)
        self.apply_button=buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.setText('Apply to draft')
        self.apply_button.clicked.connect(self._apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for edit in (self.turn_count,self.wire_cross_section,self.insulation_thickness,*self.current_bias):
            edit.textChanged.connect(self._refresh)
        self._refresh()

    def set_runtime_values(self,values):
        self.runtime_values=deepcopy(values or {})
        self._refresh()

    def values(self):
        if self._values is None:
            raise ValueError('Enter a valid transverse winding configuration')
        return deepcopy(self._values)

    def _reference_text(self,bias):
        if MATRIX_FIELD not in self.part:
            return ('Electrical current calibration is undefined. Native mrad / percentage commands '
                'retain their existing meaning. No current is guessed or fitted by this dialog.')
        data=validate_transverse_current_inputs(self.part,self.by_key)
        unit=self.command_unit
        matrix_unit=f'A/({unit})' if '/' in unit else f'A/{unit}'
        status={'calibrated_native_peak':'calibrated to original peak field',
            'manual_current_edit':'manual current bias; previous match not refitted',
            'stale_geometry_or_turn_edit':'previous match stale after geometry or turn edit'}.get(
                data['status'],data['status'].replace('_',' '))
        lines=[f"Saved reference: {data['reference_energy_ev']/1000:.9g} keV; {status}",
            f'Fixed matrix ({matrix_unit}); rows = physical currents, columns = native commands:',
            'Current rows: '+', '.join(self.channel_labels),
            'Command columns: '+', '.join(self.command_labels)]
        lines.extend('  '.join(f'{v:.10g}' for v in row) for row in data['matrix_a_per_unit'])
        fields=(('gradient_t_per_m','rotation_deg') if self.gun_stigmator else
            ('upper_field_x_mt','upper_field_y_mt','lower_field_x_mt','lower_field_y_mt') if self.gun_deflector else
            ('strength_x_percent','strength_y_percent') if self.stigmator else
            ('kick_x_mrad','kick_y_mrad') if self.channel_count==2 else
            ('upper_x_mrad','upper_y_mrad','lower_x_mrad','lower_y_mrad'))
        source={**self.owner,**self.runtime_values}
        if all(field in source for field in fields):
            commands=[float(source[field]) for field in fields]
            if self.gun_stigmator:
                gradient,angle=commands
                angle=math.radians(2*angle)
                commands=[gradient*math.cos(angle),gradient*math.sin(angle)]
            elif self.gun_deflector and source.get('beam_blanked'):
                commands=[0.,float(source.get('blanking_field_y_mt',0.)),0.,0.]
            if all(math.isfinite(value) for value in commands):
                currents=[sum(value*u for value,u in zip(row,commands))+b
                    for row,b in zip(data['matrix_a_per_unit'],bias)]
                lines.append('Command readback (A): '+', '.join(f'{v:.10g}' for v in currents))
        lines.append('Biases add amperes without changing native commands. Shared/dynamic scan currents require runtime field diagnostics.')
        return '\n'.join(lines)

    def _refresh(self):
        self._values=None
        self.wire_diameter.setText('Unknown until conductor metal area is supplied')
        try:
            turns_text=self.turn_count.text().strip()
            area_text=self.wire_cross_section.text().strip()
            insulation_text=self.insulation_thickness.text().strip()
            if not turns_text:
                if area_text or insulation_text:
                    raise ValueError('Conductor and insulation size require a known per-coil turn count')
                self._values=dict(turn_count_per_coil=None,wire_cross_section_mm2=None,insulation_thickness_mm=None)
                self.packing.setPlainText('Clearing removes the transverse winding and electrical calibration; bulk CAD remains.')
                self.reference.setPlainText(self._reference_text([0.]*self.channel_count))
            else:
                try:
                    turns=int(turns_text)
                except (ValueError,OverflowError) as error:
                    raise ValueError('Turns per physical coil must be a positive integer') from error
                if turns<=0:
                    raise ValueError('Turns per physical coil must be a positive integer')
                area=_finite_number(area_text,'Conductor metal area',positive=True) if area_text else None
                insulation=_finite_number(insulation_text,'Radial insulation thickness') if insulation_text else None
                candidate={**self.part,TURN_FIELD:turns,STATUS_FIELD:'engineering_reconstruction_not_oem',
                    NOTE_FIELD:TRANSVERSE_WINDING_SOURCE_NOTE}
                for field,value in ((AREA_FIELD,area),(INSULATION_FIELD,insulation)):
                    candidate.pop(field,None)
                    if value is not None:
                        candidate[field]=value
                d=transverse_winding_definition(candidate,self.by_key)
                self._values=dict(turn_count_per_coil=turns,wire_cross_section_mm2=area,insulation_thickness_mm=insulation)
                if MATRIX_FIELD in self.part:
                    bias=[_signed_number(edit.text(),f'{label} current bias')
                        for edit,label in zip(self.current_bias,self.channel_labels)]
                    self._values['current_bias_a']=bias
                    self.reference.setPlainText(self._reference_text(bias))
                else:
                    self.reference.setPlainText(self._reference_text([0.]*self.channel_count))
                lines=[f"Physical coils: {d['coil_count']}; turns per coil: {d['turn_count_per_coil']}; "
                    f"total turns: {d['total_turn_count']}"]
                if d['detailed_cad']:
                    self.wire_diameter.setText(f"{d['wire_diameter_mm']:.10g} mm (calculated)")
                    pitches=[p['pitch_mm'] for p in d['packing']]
                    gaps=[p['minimum_gap_mm'] for p in d['packing']]
                    lines.extend([f'Centre pitch range: {min(pitches):.9g} to {max(pitches):.9g} mm',
                        f'Minimum gap outside wire insulation: {min(gaps):.9g} mm',
                        'All turns and remaining insulation stay inside the original winding allocation.'])
                else:
                    lines.append('Conductor metal area is unknown; original bulk CAD is retained.')
                if turns!=self.part.get(TURN_FIELD) and MATRIX_FIELD in self.part:
                    lines.append('Current calibration becomes stale after this turn edit. Numerical currents are retained; no fit is performed.')
                self.packing.setPlainText('\n'.join(lines))
            self.validation.setText('Valid draft. Apply is one Undo operation; Save and live-instrument loading remain separate.')
            self.validation.setStyleSheet('')
        except (ValueError,TypeError,KeyError,OverflowError) as error:
            self._values=None
            self.validation.setText(str(error))
            self.validation.setStyleSheet('color: #c65b5b')
            self.packing.setPlainText('The preceding draft and live instrument remain unchanged.')
        self.apply_button.setEnabled(self._values is not None)

    def _apply(self):
        if self._values is None:
            return
        try:
            if self._commit is not None:
                self._commit(self.values())
        except (ValueError,TypeError,RuntimeError) as error:
            self.validation.setText(str(error))
            self.validation.setStyleSheet('color: #c65b5b')
            return
        self.accept()
