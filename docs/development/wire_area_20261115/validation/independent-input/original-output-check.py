"""Independent decimal-geometry and NI audit; no source or propagation."""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import tomllib

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
BASE = ROOT/'outputs/wire-area-20261115'
OUT = Path(__file__).resolve().parent
CURRENT = 'ideal_coil_current_a_at_100_percent'
REGIONAL = 'winding_interval_turn_counts'
ELECTRICAL = {'winding_turn_count',REGIONAL,CURRENT}


def fraction(value):
    return Fraction(str(value))


def stripped(document):
    value=deepcopy(document)
    for part in value.get('parts',()):
        for field in ELECTRICAL:
            part.pop(field,None)
        metadata=part.get('parameter_metadata',{})
        for field in ELECTRICAL:
            metadata.pop(field,None)
        if not metadata:
            part.pop('parameter_metadata',None)
    return value


def occupied(coil, owner):
    if all(f'{side}_yoke_start_local_z_mm' in owner for side in ('upper','lower')):
        inset=fraction(owner.get('mechanical_coil_axial_inset_mm',0))
        return [(fraction(owner[f'{side}_yoke_start_local_z_mm'])+inset,
                 fraction(owner[f'{side}_yoke_end_local_z_mm'])-inset) for side in ('upper','lower')]
    raw=coil.get('material_intervals_mm',[(coil['local_start_z_mm'],coil['local_end_z_mm'])])
    return [(fraction(a),fraction(b)) for a,b in raw]


def main():
    baseline=json.loads((BASE/'before/baseline.json').read_text())
    rows=[];hashes={}
    for name in baseline['files']:
        raw=(ROOT/name).read_bytes();hashes[name]=sha256(raw).hexdigest()
        current=tomllib.loads(raw.decode('utf-8'))
        previous=tomllib.loads((BASE/'before'/name).read_text(encoding='utf-8'))
        assert stripped(current)==stripped(previous),name
        old={p['key']:p for p in previous['parts']};by={p['key']:p for p in current['parts']}
        for coil in current['parts']:
            owner=by.get(coil.get('field_source_key'),{})
            if (coil.get('mechanical_profile')!='magnetic_excitation_coil' or
                    owner.get('mechanical_profile') not in {'magnetic_lens_assembly','integrated_magnetic_lens_channel'}):
                continue
            intervals=occupied(coil,owner)
            width=(fraction(coil['mechanical_outer_diameter_mm'])-fraction(coil['mechanical_inner_diameter_mm']))/2
            area=width*sum((b-a for a,b in intervals),Fraction(0))
            quota=area/9
            expected=(2*quota.numerator+quota.denominator)//(2*quota.denominator)
            assert coil['winding_turn_count']==expected,(name,owner['key'])
            assert sum(coil[REGIONAL])==expected
            before_coil=old[coil['key']];before_owner=old[owner['key']]
            old_n=before_coil['winding_turn_count'];new_n=coil['winding_turn_count']
            old_i=before_owner[CURRENT];new_i=owner[CURRENT]
            expected_i=old_i if old_n==new_n else (old_n*old_i)/new_n
            assert new_i==expected_i,(name,owner['key'],'inverse current normalization')
            delta=abs(new_n-old_n)/old_n
            assert delta<=.20,(name,owner['key'],'twenty-percent count variation')
            old_ni=old_n*old_i;new_ni=new_n*new_i
            assert abs(new_ni-old_ni)<=2e-14*abs(old_ni)
            assert 'winding_wire_diameter_mm' not in coil
            assert 'winding_wire_cross_section_mm2' not in coil
            if owner['key']=='objective_lens':
                assert new_n==1160 and list(coil[REGIONAL])==[580,580]
                source=coil['parameter_metadata']['winding_turn_count']
                assert source['source_kind']=='estimated' and '586' in source['source_note']
                assert area/new_n==9
            rows.append(dict(file=name,owner=owner['key'],area_mm2=float(area),old_turns=old_n,
                new_turns=new_n,relative_turn_change=delta,old_I100_a=old_i,new_I100_a=new_i,
                reference_NI_relative_change=(new_ni-old_ni)/old_ni))
    assert len(rows)==78
    for name,digest in hashes.items():
        assert sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Input changed during audit'
    result=dict(scope='Independent decimal/Fraction raw-input audit; no source, particles or coherent propagation',
        nominal_allocation_mm2_per_turn=9,raw_owner_checks=len(rows),objective_counts=[580,580],
        reference_photo_counts_preserved_in_provenance=[586,586],all_turn_changes_within_twenty_percent=True,
        changed_estimated_counts=sum(r['old_turns']!=r['new_turns'] for r in rows),
        maximum_relative_turn_change=max(r['relative_turn_change'] for r in rows),
        maximum_reference_NI_relative_change=max(abs(r['reference_NI_relative_change']) for r in rows),
        conductor_metal_area_inferred=False,nonelectrical_inputs_unchanged=True,input_sha256=hashes,rows=rows)
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'rows','input_sha256'}},indent=2))


if __name__=='__main__':
    main()
