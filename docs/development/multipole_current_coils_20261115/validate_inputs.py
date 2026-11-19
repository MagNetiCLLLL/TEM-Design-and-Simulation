"""Verify unchanged optical controls, native fields, source laws and clipping."""
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from unittest.mock import patch

from prepare_baseline import geometry_content, forbidden
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.multipole_current import FIELDS as CURRENT_FIELDS, validate_multipole_current_inputs
from temsim.multipole_winding import (
    FIELDS as WINDING_FIELDS, area_field, multipole_winding_definition,
    supported_multipole_winding, uses_legacy_probe_winding,
)
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.simulation_modes import switch_mode


ROOT=Path(__file__).resolve().parents[3]
V=Path(__file__).resolve().parent/'validation'


def normalized(value):
    return json.loads(json.dumps(value))


def main():
    original_inputs=json.loads((V/'before/inputs.json').read_text(encoding='utf-8'))
    all_configs=json.loads((V/'before/all-configs.json').read_text(encoding='utf-8'))
    affected={row['path'] for row in original_inputs}
    untouched=[]
    for row in all_configs:
        if row['path'] not in affected:
            assert sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'], row['path']
            untouched.append(row['path'])
    count=0;installed=[];hp_counts=[]
    for row in original_inputs:
        old=tomllib.loads((V/'before'/row['path']).read_text(encoding='utf-8'))
        path=ROOT/row['path']
        new=tomllib.loads(path.read_text(encoding='utf-8'))
        old_parts={part['key']:part for part in old['parts']}
        for part in new['parts']:
            if not supported_multipole_winding(part):
                continue
            count+=1
            previous=old_parts[part['key']]
            definition=multipole_winding_definition(part)
            assert definition is not None and validate_multipole_current_inputs(part) is not None
            if uses_legacy_probe_winding(part):
                assert definition['turn_count_per_coil']==previous['hexapole_winding_turn_count_per_coil']==2
                hp_counts.append(dict(path=row['path'],key=part['key'],old_N=2,new_N=2))
            allowed=set(WINDING_FIELDS)|set(CURRENT_FIELDS)
            if uses_legacy_probe_winding(part):
                legacy_area=area_field(part)
                assert legacy_area not in previous, 'A previously known physical wire size must not be replaced'
                allowed.add(legacy_area)
            for name in allowed:
                part.pop(name,None)
            if 'parameter_metadata' in part:
                for name in allowed:
                    if name in previous.get('parameter_metadata',{}):
                        part['parameter_metadata'][name]=previous['parameter_metadata'][name]
                    else:
                        part['parameter_metadata'].pop(name,None)
                if not part['parameter_metadata'] and 'parameter_metadata' not in previous:
                    part.pop('parameter_metadata')
        assert new==old, f'Original non-winding source values changed: {row["path"]}'
        installed.append(dict(path=row['path'],sha256=sha256(path.read_bytes()).hexdigest()))
    assert count==40 and len(hp_counts)==6
    catalog=AssemblyCatalog();default=catalog.default_selection()
    saved=json.loads((V/'before/optical-states.json').read_text(encoding='utf-8'))
    native_count=0
    for row in saved:
        selection=AssemblySelection(**row['selection'])
        state=default_state();catalog.apply(state,selection);switch_mode(state,'unified')
        if selection.column==default.column:
            apply_operating_mode_pair(state,'nano_probe','diffraction',
                column_name=selection.column,recording_name=selection.recording)
        apply_physical_layout_to_state(state)
        assert normalized(state.to_dict())==row['optical_state'], 'An optical/source/tolerance parameter changed'
        fields=capture_posed_column_fields(state,include_aligned=True)
        assert normalized([asdict(field) for field in fields])==row['native_fields'], 'Legacy native field changed'
        assert normalized(geometry_content(state))==row['clipping'], 'Declared clipping/ownership changed'
        native_count+=len(fields)
        print('Verified unchanged controls, native fields and clipping:',selection.column,'/',selection.recording,flush=True)
    receipt=dict(status='passed',affected_config_files=len(installed),multipole_hosts=count,
        optical_states_exact=len(saved),legacy_native_field_records_exact=native_count,
        existing_probe_HP_counts_exact=hp_counts,source_distribution_energy_boundaries_tolerances_unchanged=True,
        default_optical_working_point_unchanged=True,declared_clipping_and_ownership_exact=True,
        other_config_files_byte_exact=len(untouched),monochromator_and_energy_filter_unchanged=True,
        finite_source_particle_or_coherent_execution=False,installed_inputs=installed,
        geometry_scope='Declared runtime geometry, bores/liners/anchors and ownership; no circular-wave admission of filter slits')
    (V/'input-equivalence.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in {'installed_inputs','existing_probe_HP_counts_exact'}},indent=2))


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
