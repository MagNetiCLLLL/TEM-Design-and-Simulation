"""Capture the two current image-corrected column variants for analytic STEP export.

This writes JSON only. Run export_cad_models.py with the isolated CAD Python to
produce and independently read back the STEP files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/image_corrector_20261011/cad'))
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'src'))
    from temsim.cad_catalog import module_catalog
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import dependencies
    args.output.mkdir(parents=True, exist_ok=True)
    paths = [root / 'configs/instruments/column' / name for name in
             ('C3_ImageCorrector.toml', 'C3_ProbeCorrector_ImageCorrector.toml')]
    source_paths = set(paths)
    for path in paths:
        source_paths.update(dependencies(path))
    files = {path: path.read_bytes() for path in source_paths}
    records = []
    for path, data in sorted(files.items()):
        name = path.relative_to(root).as_posix()
        snapshot = args.output / 'source-inputs' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(data)
        records.append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
    implementation = list((root / 'src/temsim').glob('cad_*.py')) + [root / p for p in (
        'src/temsim/part_model_3d.py', 'src/temsim/module_manifest.py', 'src/temsim/lens_pose.py',
        'src/temsim/part_materials.py', 'src/temsim/part_model_document.py',
        'src/temsim/component_representation.py', 'src/temsim/optics/field_hardware_binding.py',
        'scripts/export_image_corrector_catalog.py', 'scripts/export_cad_models.py')]
    implementation_hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(implementation)}
    catalog = dict(schema='temsim-analytic-material-cad-v1', units='mm',
        scope='Two alternative full column modules with corrected image assemblies; not two modules to stack together. Mechanical validity is not optical or OEM qualification.',
        construction='Analytic SolidSpecs. Image HP internals are explicit CETCOR family inference; unverified optical channels have no invented material.',
        source_files=records, implementation_sha256=implementation_hashes,
        modules=[module_catalog(PartModelDocument(path).document, source=path.relative_to(root).as_posix()) for path in paths])
    if any(path.read_bytes() != data for path, data in files.items()):
        raise RuntimeError('Inputs changed during catalog capture')
    output = args.output / 'catalog.json'
    output.write_text(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(path=str(output), modules=len(catalog['modules']),
        units=sum(len(m['units']) for m in catalog['modules']),
        solids=sum(len(m['solids']) for m in catalog['modules']))), flush=True)


if __name__ == '__main__':
    main()
