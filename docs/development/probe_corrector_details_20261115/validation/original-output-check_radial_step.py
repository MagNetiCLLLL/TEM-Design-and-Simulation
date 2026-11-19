"""Bounded exact radial winding CAD/STEP check in the optional CAD runtime."""
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUT=Path(__file__).resolve().parent
definition=importlib.util.spec_from_file_location('bounded_radial_step',ROOT/'src/temsim/cad_step.py')
cad=importlib.util.module_from_spec(definition);definition.loader.exec_module(cad)
payload=json.loads((OUT/'radial-step-input.json').read_text())
records=[]
for row in payload['solids']:
    shape=cad.build_shape(row['shape'])
    validation=cad._shape_metrics(shape)
    assert validation['valid'] and validation['solid_count']==1
    records.append(dict(region=row['region'],validation=validation,volume_mm3=cad.shape_volume(shape)))
total=sum(row['volume_mm3'] for row in records)
expected=payload['dimensions']['bulk_coil_volume_mm3']
assert abs(total-expected)<1e-8,(total,expected)
metal=sum(row['volume_mm3'] for row in records if row['region'].startswith('winding_turn_'))
expected_metal=payload['dimensions']['conductor_volume_per_coil_mm3']
assert abs(metal-expected_metal)<1e-8,(metal,expected_metal)
catalog=dict(units='mm',modules=[dict(key='bounded_radial_probe',name='Bounded radial probe winding',
    solids=payload['solids'],units=[dict(key='probe_hp1_hexapole',name='One radial coil',
        solid_ids=[row['key']+'/'+row['region'] for row in payload['solids']])])])
with tempfile.TemporaryDirectory() as directory:
    report=cad.export_catalog(catalog,directory)
assert report['status']=='passed',report
result=dict(scope='Two-turn one-coil exact CAD and STEP only; no magnetic/source/particle/coherent calculation',
    records=records,total_volume_mm3=total,expected_bulk_volume_mm3=expected,
    metal_volume_mm3=metal,expected_analytic_metal_volume_mm3=expected_metal,step_report=report)
(OUT/'radial-step-receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='passed',metal_volume_mm3=metal,total_volume_mm3=total,step_status=report['status'])))
