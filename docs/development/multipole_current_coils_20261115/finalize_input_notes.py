"""Lead active engineering annotations with final counts, preserving history."""
from copy import deepcopy
import json
from pathlib import Path
import tomllib

from temsim import module_manifest
from temsim.multipole_winding import NOTE_FIELD


ROOT=Path(__file__).resolve().parents[3]
V=Path(__file__).resolve().parent/'validation'


def current_note(text):
    return text.replace('Per-coil N=156;', 'Per-coil N=63;').replace(
        'count method fixed_physical_probe_HP_or_fine_wire_limiting_pack_v1.',
        'count method fine_wire_limiting_disjoint_image_radial_subpack_v1.')


def main():
    history=V/'history/image-note-format-v1'
    assert not history.exists(), 'Do not overwrite the annotation history'
    history.mkdir(parents=True)
    target=V/'turn-initialization-final.json'
    old=target.read_bytes();(history/target.name).write_bytes(old)
    authored=json.loads(old)
    rows=json.loads((V/'before/inputs.json').read_text(encoding='utf-8'))
    changed=0
    for row in rows:
        path=ROOT/row['path'];raw=path.read_bytes()
        document=tomllib.loads(raw.decode('utf-8-sig'));updates={}
        for part in document['parts']:
            if part['key'] not in {'image_qpol_quadrupole','image_hpol_hexapole'}:
                continue
            note=current_note(part[NOTE_FIELD])
            assert 'Per-coil N=63;' in note
            metadata=deepcopy(part.get('parameter_metadata',{}))
            for key,value in metadata.items():
                if key.startswith('multipole_winding_') and isinstance(value,dict) and isinstance(value.get('source_note'),str):
                    value['source_note']=current_note(value['source_note'])
            updates['parts',part['key'],NOTE_FIELD]=note
            updates['parts',part['key'],'parameter_metadata']=metadata
            for record in authored['records']:
                if record['path']==row['path'] and record['key']==part['key']:
                    record['fields'][NOTE_FIELD]=note
            changed+=1
        if updates:
            saved=history/row['path'];saved.parent.mkdir(parents=True,exist_ok=True);saved.write_bytes(raw)
            text=module_manifest.stage_manifest_text(raw.decode('utf-8-sig'),updates,source_path=path)
            if b'\r\n' in raw:
                text=text.replace('\r\n','\n').replace('\n','\r\n')
            path.write_bytes(text.encode('utf-8'))
    assert changed==8
    target.write_text(json.dumps(authored,indent=2)+'\n',encoding='utf-8')
    (V/'annotation-finalization.json').write_text(json.dumps(dict(status='passed',
        scope='Text/provenance only; no numeric input or calibration field changed',
        image_auxiliary_records=changed,preserved='history/image-note-format-v1'),indent=2)+'\n',encoding='utf-8')
    print('Final active annotations lead with N63; numeric inputs and current matrices unchanged')


if __name__=='__main__':
    main()
