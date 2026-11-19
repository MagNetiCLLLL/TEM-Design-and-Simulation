"""Bounded exact BRep, volume partition and STEP roundtrip for radial coils."""
from argparse import ArgumentParser
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
INPUT=HERE/'validation/bounded-step-input.json'


def prepare():
    from temsim.cad_multipole_winding import multipole_winding_solids
    from temsim.cad_transverse_winding import transverse_bulk_shape
    from temsim.module_manifest import read_document
    from temsim.multipole_winding import turn_field,multipole_winding_envelopes
    rows={p['key']:p for p in read_document(ROOT/'configs/instruments/column/C3_ProbeCorrector.toml')['parts']}
    cases=[]
    for key in ('probe_hp1_hexapole','probe_hpol_hexapole'):
        part=dict(rows[key]);part[turn_field(part)]=2
        solids=[s for s in multipole_winding_solids(part,rows) if s.parameters['coil_index']==1]
        assert len(solids)==3
        cases.append(dict(component=key,solids=[asdict(s) for s in solids],
            bulk_shape=transverse_bulk_shape(multipole_winding_envelopes(part)[0])))
    INPUT.write_text(json.dumps(dict(scope='Explicit bounded two-turn one-coil CAD examples; no field calculation',cases=cases),indent=2)+'\n',encoding='utf-8')
    print(INPUT)


def replay():
    spec=importlib.util.spec_from_file_location('bounded_cad_step',ROOT/'src/temsim/cad_step.py')
    cad=importlib.util.module_from_spec(spec);spec.loader.exec_module(cad)
    records=[]
    for case in json.loads(INPUT.read_text(encoding='utf-8'))['cases']:
        shapes=[cad.build_shape(row['shape']) for row in case['solids']]
        metrics=[cad._shape_metrics(shape) for shape in shapes]
        assert all(m['valid'] and m['solid_count']==1 for m in metrics)
        bulk=cad.build_shape(case['bulk_shape']);volume=cad.shape_volume(bulk)
        total=sum(cad.shape_volume(shape) for shape in shapes)
        assert abs(total-volume)<1e-8
        assert cad.shape_volume(shapes[0].intersect(shapes[1]))<1e-10
        assert all(cad.shape_volume(wire.cut(bulk))<1e-8 for wire in shapes[:-1])
        catalog=dict(units='mm',modules=[dict(key='bounded_'+case['component'],name='Bounded multipole coil',
            solids=case['solids'],units=[dict(key=case['component'],name='Two-turn single radial coil',
                solid_ids=[row['key']+'/'+row['region'] for row in case['solids']])])])
        with tempfile.TemporaryDirectory() as directory:
            report=cad.export_catalog(catalog,directory)
        assert report['status']=='passed'
        records.append(dict(component=case['component'],shape_metrics=metrics,
            total_volume_mm3=total,bulk_volume_mm3=volume,step_report=report))
    output=HERE/'validation/bounded-step-receipt.json'
    output.write_text(json.dumps(dict(status='passed',scope='Bounded analytic CAD/STEP only; not full-default STEP or field execution',records=records),indent=2)+'\n',encoding='utf-8')
    print(output)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:replay()
