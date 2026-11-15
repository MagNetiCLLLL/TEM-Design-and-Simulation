"""Preserve the first authoring stage and separate co-located image windings.

Image QPol and HPol share a declared envelope. Their engineering radial
sub-packs must be disjoint. This refinement keeps the original envelope,
optical controls and every pre-task physical count unchanged. Only newly
authored Image auxiliary counts are re-estimated before final current fitting.
"""
from copy import deepcopy
import json
from pathlib import Path
import tomllib

from temsim import module_manifest
from temsim.multipole_winding import (
    NOTE_FIELD, TURN_FIELD, estimate_multipole_winding, multipole_winding_definition,
)


ROOT=Path(__file__).resolve().parents[3]
V=Path(__file__).resolve().parent/'validation'


def main():
    destination=V/'turn-initialization-final.json'
    assert not destination.exists(), 'Preserve the final authoring receipt'
    history=V/'history/initial-authoring'
    assert not history.exists(), 'Do not overwrite the initial authoring snapshot'
    history.mkdir(parents=True)
    initial=json.loads((V/'turn-initialization.json').read_text(encoding='utf-8'))
    originals=json.loads((V/'before/inputs.json').read_text(encoding='utf-8'))
    for name in ('turn-initialization.json','current-calibration.json','default-native-cad.json',
                 'independent-peak-field-checks.json','input-equivalence.json'):
        source=V/name
        if source.exists():
            (history/name).write_bytes(source.read_bytes())
    for row in originals:
        source=ROOT/row['path'];saved=history/row['path']
        saved.parent.mkdir(parents=True,exist_ok=True)
        saved.write_bytes(source.read_bytes())
    refinements={};staged={}
    for original in originals:
        path=ROOT/original['path']
        raw=path.read_bytes()
        document=tomllib.loads(raw.decode('utf-8-sig'))
        updates={}
        for part in document['parts']:
            if part['key'] not in {'image_qpol_quadrupole','image_hpol_hexapole'}:
                continue
            estimate=estimate_multipole_winding(part,.02,.60,.005)
            count=estimate['turn_count_per_coil']
            assert part[TURN_FIELD]==156 and count==63
            base_note=part[NOTE_FIELD].replace('Per-coil N=156;', 'Per-coil N=63;').replace(
                'count method fixed_physical_probe_HP_or_fine_wire_limiting_pack_v1.',
                'count method fine_wire_limiting_disjoint_image_radial_subpack_v1.')
            note=(base_note+' Final engineering pack-clearance refinement: '
                'co-located Image QPol uses radial fractions 0.34..0.47 and Image HPol '
                'uses 0.53..0.66 of their unchanged annular envelope. The complete '
                'QPol pack corner radius lies below the HPol minimum radius. '
                'New initial count is 63 per coil from the same 0.02 mm2 metal, '
                '0.005 mm enamel and nominal 60% fill estimator. The prior 156 '
                'was an unreleased authoring stage, not a pre-task physical count. '
                'No old measured/adopted count, bore, outer envelope, axial bound '
                'or optical working-point value is changed. Refit current explicitly.')
            candidate={**part,TURN_FIELD:count,NOTE_FIELD:note}
            definition=multipole_winding_definition(candidate)
            assert definition['detailed_cad']
            metadata=deepcopy(part.get('parameter_metadata',{}))
            for field,value in ((TURN_FIELD,count),(NOTE_FIELD,note)):
                updates['parts',part['key'],field]=value
                metadata[field]=dict(source_kind='estimated',source_note=note)
            updates['parts',part['key'],'ideal_multipole_calibration_status']='stale_geometry_changed'
            updates['parts',part['key'],'parameter_metadata']=metadata
            refinements[original['path'],part['key']]=dict(count=count,note=note,estimate=estimate,definition=definition)
        if updates:
            staged[original['path']]=module_manifest.stage_manifest_text(raw.decode('utf-8-sig'),
                updates,source_path=path,validate=False)
    assert len(refinements)==8
    for name,text in staged.items():
        if b'\r\n' in (V/'before'/name).read_bytes():
            text=text.replace('\r\n','\n').replace('\n','\r\n')
        (ROOT/name).write_bytes(text.encode('utf-8'))
    final=deepcopy(initial)
    final['schema']='multipole-final-engineering-winding-initialization-v1'
    final['initial_authoring_preserved']='history/initial-authoring'
    for row in final['records']:
        refined=refinements.get((row['path'],row['key']))
        if refined is None:
            continue
        row['turn_count_per_coil']=refined['count']
        row['total_turn_count']=refined['definition']['total_turn_count']
        row['count_method']='fine_wire_limiting_disjoint_image_radial_subpack_v1'
        row['limiting_cross_section_mm2']=refined['estimate']['limiting_cross_section_mm2']
        row['actual_metal_fill_fraction']=refined['count']*.02/row['limiting_cross_section_mm2']
        row['fields'][TURN_FIELD]=refined['count'];row['fields'][NOTE_FIELD]=refined['note']
    destination.write_text(json.dumps(final,indent=2)+'\n',encoding='utf-8')
    print('Separated 8 co-located Image auxiliary recipes; 63 turns per coil; prior authoring bytes preserved')


if __name__=='__main__':
    main()
