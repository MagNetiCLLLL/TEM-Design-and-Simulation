"""Render complete installed fine-wire CAD without emitting particles or waves."""
from copy import deepcopy
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from temsim.cad_catalog import part_solids
from temsim.cad_mesh import meshes_from_solids
from temsim.module_manifest import read_document
from temsim.transverse_winding import FIELDS,transverse_winding_definition

ROOT=Path(__file__).resolve().parents[3]
REPORT=Path(__file__).resolve().parent
CASES=(('column/C3_ProbeCorrector.toml','probe_dp22_coils'),
       ('column/C3_ProbeCorrector.toml','beam_deflector'),
       ('column/C3_ProbeCorrector.toml','condenser_stigmator'),
       ('project_and_recording_system/NoEnergyFilter.toml','diffraction_stigmator'),
       ('gun/FEG.toml','feg_deflector'),('gun/FEG.toml','feg_stigmator'),
       ('column/C3_ImageCorrector.toml','image_dph1_coils'))


def forbidden(*args,**kwargs):
    raise AssertionError('CAD validation must not emit or transport a source')


def packing_only():
    from temsim.transverse_winding import TURN_FIELD,_section
    records=[]
    for path in (ROOT/'configs/instruments').rglob('*.toml'):
        for part in read_document(path).get('parts',[]):
            if TURN_FIELD not in part:continue
            d=transverse_winding_definition(part)
            radius=d['wire_diameter_mm']/2+d['insulation_thickness_mm']
            for coil,pack in zip(d['coils'],d['packing']):
                a,b=_section(coil)
                assert all(radius-1e-12<=x<=a-radius+1e-12 and
                           radius-1e-12<delta<=b-radius+1e-12 for x,delta in pack['positions'])
                assert pack['pitch_mm']>=2*radius-1e-12
            records.append(dict(path=path.relative_to(ROOT).as_posix(),component=part['key'],
                per_coil_N=d['turn_count_per_coil'],total_N=d['total_turn_count'],
                all_insulated_sections_contained=True))
    output=REPORT/'validation/all-installed-packing-bounds.json'
    output.write_text(json.dumps(dict(status='passed',
        scope='Pure fixed-pack coordinate and insulated-wire section bounds; not full mesh or field execution',
        records=records),indent=2)+'\n',encoding='utf-8')
    print(len(records),'raw installed winding rows checked')


def main(component=None, *, final=False):
    records=[]
    for module,key in CASES:
        if component is not None and component!=key:continue
        path=ROOT/'configs/instruments'/module
        document=read_document(path);rows={p['key']:p for p in document['parts']}
        part=rows[key];before=deepcopy(part)
        d=transverse_winding_definition(part)
        if d is None:
            # Some image variants expose only a non-material control channel.
            continue
        old={k:v for k,v in part.items() if k not in FIELDS}
        previous=part_solids(old,{**rows,key:old})
        started=perf_counter();solids=part_solids(part,rows)
        passive={s.region:s for s in previous if s.material!='copper_winding'}
        current={s.region:s for s in solids}
        assert all(current[name]==value for name,value in passive.items())
        meshes=meshes_from_solids(solids,16)
        n=sum(s.region.startswith('winding_turn_transverse_') for s in solids)
        assert n==d['total_turn_count']
        assert len(meshes)==len(solids) and sum(len(m.faces) for m in meshes)<=4_000_000
        assert part==before
        row=dict(module=module,component=key,source_sha256=sha256(path.read_bytes()).hexdigest(),
            turns_per_coil=d['turn_count_per_coil'],physical_coil_count=d['coil_count'],
            total_wire_turns=n,wire_metal_area_mm2=d['wire_cross_section_mm2'],
            material_regions=len(solids),mesh_regions=len(meshes),
            actual_triangles=sum(len(m.faces) for m in meshes),
            original_passive_regions_exact=len(passive),seconds=perf_counter()-started,
            display_resolution_reduced=any('resolution reduced' in mesh.description for mesh in meshes))
        records.append(row);print(json.dumps(row),flush=True)
    name=('final-native-cad.json' if final else
          'default-native-cad.json' if component is None else f'default-native-cad-{component}.json')
    output=REPORT/'validation'/name
    output.write_text(json.dumps(dict(status='passed',scope='Complete default fine-wire CAD only; no physical source or wave calculation',
        display_triangle_budget=4_000_000,every_physical_turn_retained=True,records=records),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--component');parser.add_argument('--packing-only',action='store_true');parser.add_argument('--final',action='store_true',help='Preserve earlier mesh receipts and write a separate final-code receipt');args=parser.parse_args()
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        if args.packing_only:packing_only()
        else:main(args.component,final=args.final)
