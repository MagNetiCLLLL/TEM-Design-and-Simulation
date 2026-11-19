"""Verify original optical controls, tolerances and non-winding inputs exactly."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from unittest.mock import patch

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.simulation_modes import switch_mode
from temsim.transverse_current import FIELDS as ELECTRICAL_FIELDS, validate_transverse_current_inputs
from temsim.transverse_winding import FIELDS as WINDING_FIELDS, validate_transverse_winding


ROOT=Path(__file__).resolve().parents[3]
VALIDATION=Path(__file__).resolve().parent/'validation'


def forbidden(*args,**kwargs):
    raise AssertionError('Input equivalence must not emit or transport source particles')


def normalized(value):
    return json.loads(json.dumps(value))


def main():
    originals=json.loads((VALIDATION/'before/inputs.json').read_text(encoding='utf-8'))
    added=WINDING_FIELDS|ELECTRICAL_FIELDS
    installed=[];hosts=0
    for record in originals:
        before=tomllib.loads((VALIDATION/'before'/record['path']).read_text(encoding='utf-8'))
        path=ROOT/record['path']
        current=tomllib.loads(path.read_text(encoding='utf-8'))
        for part in current.get('parts',[]):
            if WINDING_FIELDS.intersection(part):
                hosts+=1
                validate_transverse_winding(part)
                assert validate_transverse_current_inputs(part) is not None
            for name in added: part.pop(name,None)
            if 'parameter_metadata' in part:
                for name in added:part['parameter_metadata'].pop(name,None)
                if not part['parameter_metadata']:part.pop('parameter_metadata')
        assert current==before, f'Other source values changed: {record["path"]}'
        installed.append(dict(path=record['path'],sha256=sha256(path.read_bytes()).hexdigest()))
    assert hosts==96
    states=json.loads((VALIDATION/'before/optical-states.json').read_text(encoding='utf-8'))
    catalog=AssemblyCatalog();default=catalog.default_selection();field_count=0
    for original in states:
        selection=AssemblySelection(**original['selection'])
        state=default_state();catalog.apply(state,selection);switch_mode(state,'unified')
        if selection.column==default.column:
            apply_operating_mode_pair(state,'nano_probe','diffraction',
                column_name=selection.column,recording_name=selection.recording)
        apply_physical_layout_to_state(state)
        assert normalized(state.to_dict())==original['optical_state'], 'Native working-point controls changed'
        fields=capture_posed_column_fields(state,include_aligned=True)
        assert normalized([asdict(f) for f in fields])==original['native_fields'], 'Legacy Gaussian fields changed'
        field_count+=len(fields)
        print('Verified unchanged optical state:',selection.column,'/',selection.recording,flush=True)
    receipt=dict(status='passed',source_files=len(originals),physical_winding_hosts=hosts,
        assembly_states_exact=len(states),legacy_native_field_records_exact=field_count,
        other_source_values_exact=True,source_distribution_boundaries_and_tolerances_unchanged=True,
        particle_or_coherent_calculation_executed=False,installed_inputs=installed)
    (VALIDATION/'input-equivalence.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
