"""Verify the compact historical bundle without extracting files or running physics."""
import gzip
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema']!='temsim-probe-tail-publication-bundle-v1':
        raise ValueError('Unsupported archive manifest')
    seen=set()
    for row in manifest['payload_files']:
        path=(root/row['archived_relative_path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Archive entry escapes its bundle')
        value=path.read_bytes()
        if len(value)!=row['archived_bytes'] or digest(value)!=row['archived_sha256']:
            raise ValueError('Archived bytes do not match: '+str(path))
        if row['encoding']=='gzip':value=gzip.decompress(value)
        elif row['encoding']!='identity':raise ValueError('Unknown archive encoding')
        if len(value)!=row['original_bytes'] or digest(value)!=row['original_sha256']:
            raise ValueError('Original-byte identity does not match: '+str(path))
        seen.add(row['archived_relative_path'])
    print(json.dumps(dict(status='ARCHIVE_HASHES_VERIFIED',unique_payload_files=len(seen),
        source_mappings=len(manifest['payload_files']),physics_executed=False,
        omitted_raw_arrays_verified=False,
        scope='Archived bytes only; omitted local particle arrays and scientific conclusions were not revalidated')))


if __name__=='__main__':main()
