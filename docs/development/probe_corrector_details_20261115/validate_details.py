"""Probe-corrector CAD detail receipts with unchanged native optics/fields.

Default changes add only explicitly engineered passive outer winding jackets.
Detailed turns remain opt-in and require user-supplied N and conductor area.
This script does not emit/trace a source, focus a beam or propagate a wave.
"""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.cad_multipoles import multipole_solids
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.posed_column_fields import capture_posed_column_fields
from temsim.simulation_modes import switch_mode

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/probe-corrector-details-20261115'
CURATED=Path(__file__).resolve().parent/'validation'
HOSTS=('probe_hp1_hexapole','probe_hp2_hexapole')
FIELD='hexapole_support_outer_insulation_thickness_mm'


def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def preserved(name):
    for root in (OUT,CURATED):
        p=root/name
        if p.is_file():return p
    raise FileNotFoundError(name)


def normalized(value):return json.loads(json.dumps(value))


def capture():
    catalog=AssemblyCatalog();rows=[];solids={}
    for column in catalog.columns:
        if not column.properties.get('probe_corrector'):continue
        for recording in catalog.recording_systems:
            state=default_state();catalog.apply(state,AssemblySelection('FEG',column.name,recording.name))
            switch_mode(state,'unified')
            if column.name==catalog.default_selection().column:
                apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=column.name,recording_name=recording.name)
            apply_physical_layout_to_state(state);state.step_mm=.05
            controls=normalized(state.to_dict())
            keys=tuple(c.key for c in (*state.corrector_elements,*state.stigmators,*state.deflectors)
                       if c.key.startswith('probe_'))
            fields=capture_posed_column_fields(state,include_aligned=True,component_keys=keys)
            records=[]
            for field in fields:
                z=field.center_z_m
                points=np.array([[0.,0.,z],[1e-6,2e-6,z],[-2e-6,3e-6,z-.0005],
                                 [4e-6,-1e-6,z+.0005]])
                records.append(dict(component=field.component_key,potential=asdict(field),
                                    points_m=points.tolist(),b_t=field.field_at_global_positions_t(points).tolist()))
            enabled_hosts={c.key for c in state.corrector_elements if c.key in HOSTS and c.enabled}
            assert {r['component'] for r in records}.issuperset(enabled_hosts)
            assert normalized(state.to_dict())==controls
            rows.append(dict(column=column.name,recording=recording.name,state=controls,fields=normalized(records)))
            if recording.name=='No Energy Filter':
                by_key={p.key:dict(p.data) for p in state._resolved_assembly.parts}
                for key in HOSTS:
                    solids[column.name+' / '+key]=normalized([asdict(s) for s in multipole_solids(by_key[key],by_key)])
            print('Captured unchanged optics/fields:',column.name,'/',recording.name,flush=True)
    return dict(cases=rows,solids=solids)


def apply():
    hashes=json.loads(preserved('before/inputs.json').read_text())
    assert not (OUT/'before/state-fields.json').exists(),'Do not overwrite the original field receipt'
    save('before/state-fields.json',capture());staged={}
    for name,digest in hashes.items():
        path=ROOT/name;assert sha256(path.read_bytes()).hexdigest()==digest
        doc=module_manifest.read_document(path);updates={}
        for key in HOSTS:
            part=next(p for p in doc['parts'] if p['key']==key)
            assert FIELD not in part
            metadata=deepcopy(part.get('parameter_metadata',{}))
            metadata[FIELD]=dict(source_kind='estimated',source_note=(
                'User-requested internal CAD enrichment on 2026-11-15: 0.15 mm exterior winding insulation '
                'is an explicit engineering assumption, not OEM/measured hardware. The jacket expands '
                'only the coil outer tangential/axial sides inside the original HP host envelope; '
                'radial start/end, former flanges, pole/core/yoke, beam clearance and all field inputs remain fixed.'))
            updates['parts',key,FIELD]=.15
            updates['parts',key,'parameter_metadata']=metadata
        staged[name]=module_manifest.stage_manifest_text(path.read_bytes().decode('utf-8'),updates,source_path=path)
    for name,text in staged.items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==hashes[name]
        (ROOT/name).write_bytes(text.encode('utf-8'))


def validate():
    import tomllib
    baseline=json.loads(preserved('before/state-fields.json').read_text());current=capture()
    assert baseline['cases']==current['cases'],'Native optics, enabled state, field coefficients or B samples changed'
    assert len(current['cases'])==6
    for key,old in baseline['solids'].items():
        new=current['solids'][key]
        assert len(old)==25 and len(new)==31
        assert new[:25]==old,'Original pole/copper/support CAD changed'
        assert all(s['material']=='insulator' and s['region'].endswith('_outer_insulation') for s in new[25:])
    hashes=json.loads(preserved('before/inputs.json').read_text())
    for name in hashes:
        old=tomllib.loads(preserved('before/'+name).read_text(encoding='utf-8'))
        new=tomllib.loads((ROOT/name).read_text(encoding='utf-8'))
        for part in new['parts']:
            if part['key'] in HOSTS:
                assert part.pop(FIELD)==.15
                part.get('parameter_metadata',{}).pop(FIELD)
                if not part.get('parameter_metadata'):part.pop('parameter_metadata',None)
                assert not any(k.startswith('hexapole_winding_') for k in part),'Default HP turns/metal area were invented'
        assert old==new,'Non-detail source inputs changed'
    field_count=sum(len(c['fields']) for c in current['cases'])
    nonzero=sum(any(abs(v)>0 for row in f['b_t'] for v in row) for c in current['cases'] for f in c['fields'])
    save('state-fields.json',current)
    save('summary.json',dict(status='passed',default_source_files=3,probe_hosts=6,assembly_cases=6,
        original_regions_per_hp=25,new_regions_per_hp=31,outer_insulation_thickness_mm=.15,
        native_field_records_exact=field_count,nonzero_native_field_records=nonzero,
        full_optical_source_state_exact=True,original_25_solids_exact=True,
        only_authorized_passive_fields_added=True,default_hp_turns_area_or_current_inferred=False,
        source_transport_focus_material_solve_or_coherent_executed=False))
    print('Verified',field_count,'exact native fields and all original HP solids',flush=True)


def forbidden(*args,**kwargs):raise AssertionError('CAD/field-only validation must not emit or transport particles')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        if args.apply:apply()
        validate()
