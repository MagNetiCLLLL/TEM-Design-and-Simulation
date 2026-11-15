"""Prepare or replay bounded exact transverse winding/STEP verification."""
from argparse import ArgumentParser
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
HERE=Path(__file__).resolve().parent
INPUT=HERE/'validation/transverse-step-input.json'


def prepare():
    from temsim.module_manifest import read_document
    from temsim.cad_transverse_winding import transverse_winding_solids,transverse_bulk_shape
    from temsim.transverse_winding import AREA_FIELD,TURN_FIELD,INSULATION_FIELD,STATUS_FIELD,NOTE_FIELD,SOURCE_NOTE,transverse_winding_envelopes
    parts={p['key']:p for p in read_document(ROOT/'configs/instruments/column/C3_ProbeCorrector.toml')['parts']}
    cases=[]
    for key in ('probe_dp22_coils','condenser_stigmator'):
        part={**parts[key],TURN_FIELD:2,AREA_FIELD:.01,INSULATION_FIELD:.005,
              STATUS_FIELD:'engineering_reconstruction_not_oem',NOTE_FIELD:SOURCE_NOTE}
        specs=[s for s in transverse_winding_solids(part) if s.parameters['coil_index']==1]
        cases.append(dict(component=key,solids=[asdict(s) for s in specs],
                          bulk_shape=transverse_bulk_shape(transverse_winding_envelopes(part)[0])))
    INPUT.parent.mkdir(parents=True,exist_ok=True)
    INPUT.write_text(json.dumps(dict(scope='Explicit bounded two-turn one-coil examples; not default counts',cases=cases),indent=2)+'\n',encoding='utf-8')
    print(INPUT)


def replay():
    spec=importlib.util.spec_from_file_location('transverse_cad_step',ROOT/'src/temsim/cad_step.py')
    cad=importlib.util.module_from_spec(spec);spec.loader.exec_module(cad)
    records=[]
    for case in json.loads(INPUT.read_text(encoding='utf-8'))['cases']:
        shapes=[cad.build_shape(row['shape']) for row in case['solids']]
        metrics=[cad._shape_metrics(shape) for shape in shapes]
        assert all(m['valid'] and m['solid_count']==1 for m in metrics)
        bulk=cad.build_shape(case['bulk_shape']);bulk_volume=cad.shape_volume(bulk)
        total=sum(cad.shape_volume(shape) for shape in shapes)
        assert abs(total-bulk_volume)<1e-8,(total,bulk_volume)
        for wire in shapes[:-1]:
            assert cad.shape_volume(wire.cut(bulk))<1e-8
        assert cad.shape_volume(shapes[0].intersect(shapes[1]))<1e-10
        catalog=dict(units='mm',modules=[dict(key='bounded_'+case['component'],name='Bounded transverse coil',
            solids=case['solids'],units=[dict(key=case['component'],name='Two-turn one-coil example',
                solid_ids=[row['key']+'/'+row['region'] for row in case['solids']])])])
        with tempfile.TemporaryDirectory() as directory:
            report=cad.export_catalog(catalog,directory)
        assert report['status']=='passed'
        records.append(dict(component=case['component'],shape_metrics=metrics,
                            total_volume_mm3=total,bulk_volume_mm3=bulk_volume,step_report=report))
    destination=HERE/'validation/transverse-step-receipt.json'
    destination.write_text(json.dumps(dict(status='passed',scope='Bounded analytic CAD and STEP roundtrip only; no source or field calculation',records=records),indent=2)+'\n',encoding='utf-8')
    print(destination)


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.prepare:prepare()
    else:replay()
