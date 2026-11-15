"""Bind executed CAD receipts after an electrical-only calibration refinement.

Reconstruct the earlier tested source bytes with each preserved calibration
table, require the recorded SHA256, then compare complete current CAD recipes.
No source files are written and no CAD mesh or physical calculation is rerun.
"""
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import tomllib

from temsim.cad_catalog import part_solids
from temsim.module_manifest import stage_manifest_text

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
V=HERE/'validation'


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    candidates=[(p.name,json.loads(p.read_text(encoding='utf-8'))['records'])
                for p in sorted(V.glob('current-calibration*.json'))]
    records=[]
    receipts=('default-native-cad.json','default-native-cad-image_dph1_coils.json')
    for receipt_name in receipts:
        for tested in json.loads((V/receipt_name).read_text(encoding='utf-8'))['records']:
            name='configs/instruments/'+tested['module'];path=ROOT/name
            current_bytes=path.read_bytes();current=tomllib.loads(current_bytes.decode('utf-8-sig'))
            current_rows={p['key']:p for p in current['parts']}
            reconstructed=None;calibration=None
            if sha256(current_bytes).hexdigest()==tested['source_sha256']:
                reconstructed=current;calibration='unchanged_source_bytes'
            else:
                for table,entries in candidates:
                    updates={}
                    for entry in entries:
                        if entry['path']!=name:continue
                        part=current_rows[entry['host_key']]
                        metadata=deepcopy(part.get('parameter_metadata',{}))
                        fields=entry['fields']
                        for field,value in fields.items():
                            updates['parts',part['key'],field]=value
                            metadata[field]=dict(source_kind='estimated',source_note=fields['ideal_transverse_calibration_source_note'])
                        updates['parts',part['key'],'parameter_metadata']=metadata
                    text=stage_manifest_text(current_bytes.decode('utf-8-sig'),updates,source_path=path,validate=False)
                    if b'\r\n' in current_bytes:text=text.replace('\r\n','\n').replace('\n','\r\n')
                    previous_bytes=text.encode('utf-8')
                    if sha256(previous_bytes).hexdigest()==tested['source_sha256']:
                        reconstructed=tomllib.loads(text);calibration=table;break
            assert reconstructed is not None, f'Cannot reproduce tested source bytes for {name}'
            previous_rows={p['key']:p for p in reconstructed['parts']}
            key=tested['component']
            old=[asdict(s) for s in part_solids(previous_rows[key],previous_rows)]
            new=[asdict(s) for s in part_solids(current_rows[key],current_rows)]
            assert old==new,'Current CAD geometry, metadata or physical turn count changed'
            records.append(dict(component=key,path=name,original_receipt=receipt_name,
                tested_source_sha256=tested['source_sha256'],current_source_sha256=sha256(current_bytes).hexdigest(),
                reconstructed_calibration_table=calibration,complete_solid_recipes_exact=True,
                solid_recipe_sha256=digest(new),solid_regions=len(new),
                physical_turns=tested['total_wire_turns'],tested_actual_triangles=tested['actual_triangles']))
    output=V/'cad-electrical-refinement-equivalence.json'
    output.write_text(json.dumps(dict(status='passed',scope='Exact source-byte reconstruction and unchanged complete CAD recipes; no mesh rerender',
        reused_mesh_receipts_are_original=True,records=records),indent=2)+'\n',encoding='utf-8')
    print(f'Bound {len(records)} executed complete-CAD cases to current electrical-only refinement')


if __name__=='__main__':main()
