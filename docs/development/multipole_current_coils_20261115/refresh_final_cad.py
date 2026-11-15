"""Refresh Image mesh receipts after annotation-only source-byte finalization.

Unchanged Probe receipts are reused only when the complete source file SHA256
still matches. All original executed receipts remain unmodified.
"""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

import check_default_cad as worker


ROOT=Path(__file__).resolve().parents[3]
V=Path(__file__).resolve().parent/'validation'


def main():
    target=V/'final-native-cad-current.json'
    assert not target.exists(), 'Preserve the refreshed mesh evidence'
    original=json.loads((V/'final-native-cad.json').read_text(encoding='utf-8'))
    records=[]
    for row in original['records']:
        current=sha256((ROOT/row['path']).read_bytes()).hexdigest()
        if current==row['source_snapshot_sha256']:
            records.append({**row,'source_receipt':'final-native-cad.json'})
            continue
        assert row['component'].startswith('image_'), 'Unexpected source-byte drift'
        receipt=V/f'default-native-cad-{row["component"]}.json'
        assert not receipt.exists(), 'Preserve the component mesh receipt'
        worker.main(row['component'])
        new=json.loads(receipt.read_text(encoding='utf-8'))
        assert len(new['records'])==1
        record=new['records'][0]
        assert record['source_snapshot_sha256']==current
        records.append({**record,'source_receipt':receipt.name})
    assert len(records)==6
    assert all(sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['source_snapshot_sha256'] for row in records)
    receipt={**original,'scope':'Final source-byte-bound complete native CAD; Probe receipts reused at exact SHA, Image meshes refreshed after note formatting',
        'original_receipts_preserved':True,'records':records}
    target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('All six complete CAD cases bound to final installed source bytes')


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',worker.forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',worker.forbidden):
        main()
