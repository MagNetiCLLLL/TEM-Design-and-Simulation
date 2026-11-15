"""Execute complete installed multipole winding CAD with every saved turn."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import tomllib
from unittest.mock import patch

from temsim.cad_catalog import part_solids
from temsim.cad_mesh import meshes_from_solids
from temsim.component_representation import non_material_role
from temsim.multipole_winding import multipole_winding_definition

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
CASES=(('C3_ProbeCorrector','probe_hp1_hexapole'),
       ('C3_ImageCorrector','image_hp1_hexapole'),
       ('C3_ProbeCorrector','probe_qpol_quadrupole'),
       ('C3_ProbeCorrector','probe_hpol_hexapole'),
       ('C3_ImageCorrector','image_qpol_quadrupole'),
       ('C3_ImageCorrector','image_hpol_hexapole'))


def forbidden(*args,**kwargs):
    raise AssertionError('Complete winding CAD verification must not emit or transport particles')


def main(component=None,*,final=False):
    records=[]
    for module,key in CASES:
        if component is not None and key!=component:continue
        path=ROOT/f'configs/instruments/column/{module}.toml'
        snapshot=path.read_bytes()
        rows={p['key']:p for p in tomllib.loads(snapshot.decode('utf-8-sig'))['parts']}
        part=rows[key];definition=multipole_winding_definition(part,rows)
        assert definition is not None and definition['detailed_cad']
        started=perf_counter()
        solids=part_solids(part,rows)
        actual_turns=sum(s.region.startswith('winding_turn_multipole_') for s in solids)
        assert actual_turns==definition['total_turn_count']
        meshes=meshes_from_solids(solids,32)
        triangles=sum(len(mesh.faces) for mesh in meshes)
        assert len(meshes)==len(solids) and triangles<=4_000_000
        row=dict(component=key,path=path.relative_to(ROOT).as_posix(),source_snapshot_sha256=sha256(snapshot).hexdigest(),
            turns_per_coil=definition['turn_count_per_coil'],coil_count=definition['coil_count'],
            total_wire_turns=actual_turns,wire_metal_area_mm2=definition['wire_cross_section_mm2'],
            angular_datum_deg=definition['basis_angle_deg'],material_regions=len(solids),
            mesh_regions=len(meshes),actual_triangles=triangles,requested_angular_segments=32,
            display_resolution_reduced=any('resolution reduced' in mesh.description for mesh in meshes),
            transport_role=non_material_role(part) or 'existing_material_host',
            seconds=perf_counter()-started)
        records.append(row);print(json.dumps(row),flush=True)
    output=HERE/'validation'/('final-native-cad.json' if final else
        'default-native-cad.json' if component is None else f'default-native-cad-{component}.json')
    output.write_text(json.dumps(dict(status='passed',scope='Complete native CAD only; no particle or coherent execution',
        every_physical_turn_retained=True,display_triangle_budget=4_000_000,records=records),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--component');parser.add_argument('--final',action='store_true');args=parser.parse_args()
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main(args.component,final=args.final)
