"""Explicit one-time fine-wire authoring and fixed-current peak calibration.

Original source files are preserved by prepare_baseline.py. Geometry authoring
never fits a field. Current calibration never changes N, metal area or envelope.
"""
from argparse import ArgumentParser
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tomllib
from unittest.mock import patch

from temsim import module_manifest
from temsim.transverse_winding import (
    AREA_FIELD, INSULATION_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
    SOURCE_NOTE, estimate_transverse_winding, supported_transverse_winding,
    transverse_winding_envelopes,
)


ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
VALIDATION=REPORT/'validation'


def forbidden(*args,**kwargs):
    raise AssertionError('Input and field calibration must not emit or transport particles')


def write_source(name,text):
    """Retain the preserved input's newline convention after targeted edits."""
    original=(VALIDATION/'before'/name).read_bytes()
    if b'\r\n' in original:
        text=text.replace('\r\n','\n').replace('\n','\r\n')
    (ROOT/name).write_bytes(text.encode('utf-8'))


def wire_assumption(part):
    coils=transverse_winding_envelopes(part)
    if coils[0]['family']=='radial_rectangle':
        return .02,.005,'Fine round conductor for radial stigmator windings'
    if part['key'].startswith('image_dph'):
        return .0003,.002,'Ultrafine round conductor for the existing 0.08 mm radial image-DPH allocation'
    return .01,.005,'Fine round conductor for saddle deflector windings'


def seed_geometry():
    destination=VALIDATION/'turn-initialization.json'
    assert not destination.exists(), 'Preserve the first initialization receipt'
    original=json.loads((VALIDATION/'before/inputs.json').read_text(encoding='utf-8'))
    staged={};receipts=[]
    for record in original:
        path=ROOT/record['path']
        assert sha256(path.read_bytes()).hexdigest()==record['sha256'], 'Source changed before explicit migration'
        document=tomllib.loads(path.read_text(encoding='utf-8'));updates={}
        for part in document.get('parts',[]):
            if not supported_transverse_winding(part): continue
            area,enamel,reason=wire_assumption(part)
            estimate=estimate_transverse_winding(part,area,.60,enamel)
            note=(f'{reason}: metal area {area:g} mm2, radial enamel assumption {enamel:g} mm, '
                f'nominal metal fill 0.60. Fixed initial count {estimate["turn_count_per_coil"]} per coil '
                'is estimated from the limiting existing pack section and validated insulated-wire packing. '
                'No measured wire gauge, OEM winding count, thermal rating or current limit is asserted. '+SOURCE_NOTE)
            fields=deepcopy(estimate['fields']);fields[NOTE_FIELD]=note
            metadata=deepcopy(part.get('parameter_metadata',{}))
            for field in (TURN_FIELD,AREA_FIELD,INSULATION_FIELD):
                metadata[field]=dict(source_kind='estimated',source_note=note)
            for name,value in fields.items(): updates['parts',part['key'],name]=value
            updates['parts',part['key'],'parameter_metadata']=metadata
            receipts.append(dict(path=record['path'],key=part['key'],wire_assumption=reason,
                **{k:v for k,v in estimate.items() if k not in {'packing','fields'}}))
        staged[record['path']]=module_manifest.stage_manifest_text(
            path.read_text(encoding='utf-8'),updates,source_path=path,
            validate=not record['path'].endswith('projector_stack.toml'))
    for record in original:
        path=ROOT/record['path']
        assert sha256(path.read_bytes()).hexdigest()==record['sha256']
    for name,text in staged.items(): write_source(name,text)
    destination.write_text(json.dumps(dict(schema='transverse-fine-wire-initialization-v1',
        source_files=len(staged),hosts=len(receipts),records=receipts),indent=2)+'\n',encoding='utf-8')
    print(f'Initialized {len(receipts)} physical winding hosts across {len(staged)} source files')


def calibrate_currents(*, refine=False, final=False):
    from dataclasses import asdict
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.physics.transverse_coil_field import calibrate_transverse_current_matrix
    from temsim.simulation_modes import switch_mode
    from temsim.transverse_current import MATRIX_FIELD, NOTE_FIELD as CURRENT_NOTE
    destination=VALIDATION/('current-calibration-final.json' if final else
        'current-calibration-refined.json' if refine else 'current-calibration.json')
    assert not destination.exists(), 'Preserve the first current calibration receipt'
    originals=json.loads((VALIDATION/'before/inputs.json').read_text(encoding='utf-8'))
    catalog=AssemblyCatalog();default=catalog.default_selection()
    columns={Path(option.file).name:option.name for option in catalog.columns}
    guns={Path(option.file).name:option.name for option in catalog.guns}
    states={};staged={};records=[]
    source_hashes={r['path']:sha256((ROOT/r['path']).read_bytes()).hexdigest() for r in originals}
    for record in originals:
        path=ROOT/record['path']
        if '/column/' in record['path']:
            selection=AssemblySelection(default.gun,columns[path.name],default.recording)
        elif '/gun/' in record['path']:
            selection=AssemblySelection(guns[path.name],default.column,default.recording)
        else:
            selection=AssemblySelection(default.gun,default.column,
                'Energy Filter' if path.name=='projector_stack.toml' else 'No Energy Filter')
        if selection not in states:
            state=default_state();catalog.apply(state,selection);switch_mode(state,'unified')
            if selection.column==default.column:
                apply_operating_mode_pair(state,'nano_probe','diffraction',
                    column_name=selection.column,recording_name=selection.recording)
            apply_physical_layout_to_state(state);states[selection]=state
        state=states[selection]
        before=json.loads(json.dumps(state.to_dict()))
        document=tomllib.loads(path.read_text(encoding='utf-8'));updates={}
        for part in document.get('parts',[]):
            if not supported_transverse_winding(part): continue
            if refine or final:
                assert MATRIX_FIELD in part, 'Refinement requires an existing saved electrical calibration'
            else:
                assert MATRIX_FIELD not in part, 'Do not overwrite a prior electrical calibration'
            result=calibrate_transverse_current_matrix(state,part['key'])
            metadata=deepcopy(part.get('parameter_metadata',{}))
            for name,value in result['fields'].items():
                updates['parts',part['key'],name]=value
                metadata[name]=dict(source_kind='estimated',source_note=result['fields'][CURRENT_NOTE])
            updates['parts',part['key'],'parameter_metadata']=metadata
            records.append(dict(path=record['path'],selection=asdict(selection),**result))
            print('Calibrated fixed-N source:',path.name,part['key'],flush=True)
        assert before==json.loads(json.dumps(state.to_dict())), 'Current fitting changed native controls'
        staged[record['path']]=module_manifest.stage_manifest_text(
            path.read_bytes().decode('utf-8'),updates,source_path=path,
            validate=not record['path'].endswith('projector_stack.toml'))
    for name,digest in source_hashes.items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest, 'Input changed while calibrating'
    for name,text in staged.items(): write_source(name,text)
    destination.write_text(json.dumps(dict(schema='transverse-original-peak-current-calibration-v1',
        source_files=len(staged),hosts=len(records),records=records),indent=2)+'\n',encoding='utf-8')
    print(f'Calibrated {len(records)} material hosts; no native control or tolerance changed')


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--seed-geometry',action='store_true')
    parser.add_argument('--calibrate-currents',action='store_true')
    parser.add_argument('--refine-calibration',action='store_true')
    parser.add_argument('--finalize-calibration',action='store_true')
    args=parser.parse_args()
    if args.seed_geometry: seed_geometry()
    if args.calibrate_currents: calibrate_currents()
    if args.refine_calibration: calibrate_currents(refine=True)
    if args.finalize_calibration: calibrate_currents(final=True)


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
