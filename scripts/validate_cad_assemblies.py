"""Capture native assembly combinations, then audit JSON in the isolated CAD env.

Prepare with the project Python. Audit with environment/cad/Scripts/python.exe.
No optical solver or GUI is started, and no configuration is changed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
from itertools import product
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace


def load_auditor(root):
    spec = importlib.util.spec_from_file_location('temsim_collision_audit', root / 'src/temsim/cad_assembly_validation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, *, material_liners=True, holder_retractions=(0., 50., 160., 320., 400.)):
    implementation_paths = set((root / 'src/temsim').rglob('*.py')) | {Path(__file__).resolve()}
    implementation = {path: path.read_bytes() for path in implementation_paths}
    sys.path.insert(0, str(root / 'src'))
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_catalog import build_catalog, module_catalog, posed_solids
    from temsim.column.module_assembly import resolve_module_assembly
    from temsim.component_representation import non_material_role
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import dependencies
    catalog = AssemblyCatalog(root / 'configs/instruments')
    source_paths = set((root / 'configs/instruments').rglob('*.toml'))
    for path in list(source_paths):
        source_paths.update(dependencies(path))
    originals = {path: path.read_bytes() for path in source_paths}
    modules = build_catalog(root)['modules']
    documents = {m['key']: PartModelDocument(root / m['source']).document for m in modules}
    for module in modules:
        by_key = {p['key']: p for p in documents[module['key']]['parts']}
        material_keys = {spec['key'] for spec in module['solids']}
        for omitted in module['omitted']:
            part = by_key[omitted['key']]
            role = non_material_role(part)
            has_children = any(p.get('parent_key') == part['key'] and p['key'] in material_keys for p in by_key.values())
            omitted['classification'] = role or ('assembly_parent_with_material_children' if has_children
                                                 else 'no_independent_material_specification')
    module_map = {m['key']: m for m in modules}
    auditor = load_auditor(root)
    assemblies = []
    liner_adapter = None
    liner_decisions = None
    if material_liners:
        try:
            from temsim.cad_stage_holder import physical_liner_rows, liner_material_decisions
            liner_adapter = physical_liner_rows
            liner_decisions = liner_material_decisions
        except ModuleNotFoundError as exc:
            if exc.name != 'temsim.cad_stage_holder':
                raise
            raise RuntimeError('Physical liner adapter is unavailable. For a scoped preliminary diagnostic only, use --no-material-liners.') from exc
    holder_baseline = None
    for gun, blanker, column, recording in product(catalog.guns, catalog.beam_blankers, catalog.columns, catalog.recording_systems):
        has_probe, has_image = column.properties['probe_corrector'], column.properties['image_corrector']
        corrector = ('double_corrector' if has_probe and has_image else 'probe_corrector' if has_probe
                     else 'image_corrector' if has_image else 'no_corrector')
        configuration = SimpleNamespace(electron_gun_type='thermionic' if gun.properties['electron_gun'] == 'Thermionic' else 'cold_feg',
            monochromator_installed=bool(gun.properties['monochromator']),
            corrector=SimpleNamespace(value=corrector),
            image_corrector_design=column.properties.get('image_corrector_design', 'cetcor'),
            c3_hardware=SimpleNamespace(value='three_condenser' if column.properties['c3_lens'] else 'two_condenser'),
            energy_filter_selected=bool(recording.properties['energy_filter']), nanopulser_installed=blanker.name != 'None')
        assembly = resolve_module_assembly(configuration, root=root / 'configs/instruments')
        selection = dict(gun=gun.name, beam_blanker=blanker.name, column=column.name, recording=recording.name)
        placements = []
        for module in assembly.modules:
            rows = [p for p in assembly.parts if p.module_key == module.key]
            origins = [p.center_z_mm - float(p.data['local_center_z_mm']) for p in rows]
            if not origins or any(not math.isclose(value, origins[0], abs_tol=1e-8, rel_tol=0.) for value in origins):
                raise RuntimeError(f'Module {module.key} lacks one rigid resolved translation')
            placements.append(dict(module_key=module.key, offset_mm=[0., 0., origins[0]]))
        if liner_adapter is not None:
            rows = list(liner_adapter(assembly))
            by_key = {row['key']: row for row in rows}
            solids = []
            for segment_index, row in enumerate(rows):
                for spec in posed_solids(row, by_key):
                    # The adapter describes actual material tubes with stage
                    # passage cuts, distinct from the electric-domain boundary.
                    spec = replace(spec,
                                   note=spec.note + ' Resolved physical liner adapter; not the uncut axisymmetric electric boundary.')
                    data = asdict(spec)
                    data['instance_key'] = f'{spec.key}/{spec.region}/segment-{segment_index:04d}'
                    data['color'] = list(spec.color)
                    solids.append(data)
            key = '@physical_liners:' + auditor.digest(solids)
            if key not in module_map:
                row = dict(key=key, name='Resolved physical vacuum liners', solids=solids, omitted=[])
                modules.append(row)
                module_map[key] = row
            placements.append(dict(module_key=key, offset_mm=[0., 0., 0.]))
        assemblies.append(dict(key=' | '.join(selection.values()), selection=selection, placements=placements,
            clearance_focus_keys=['sample_holder'],
            material_liner_scope='physical_adapter' if liner_adapter else 'excluded_pending_physical_adapter',
            liner_material_decisions=liner_decisions(assembly) if liner_decisions else []))
        if (gun.name == 'FEG' and blanker.name == 'None' and has_probe and has_image
                and not recording.properties['energy_filter']):
            holder_baseline = assemblies[-1]
    if holder_retractions:
        if holder_baseline is None:
            raise RuntimeError('Full corrector assembly for holder withdrawal check is unavailable')
        base_placement = next(p for p in holder_baseline['placements'] if p['module_key'].startswith('column:'))
        base_module = module_map[base_placement['module_key']]
        document = documents[base_placement['module_key']]
        holder = next((p for p in document['parts'] if p.get('mechanical_profile') == 'side_entry_sample_holder'), None)
        if holder is None:
            raise RuntimeError('Configured full corrector assembly has no physical holder')
        for distance in holder_retractions:
            distance = float(distance)
            if not math.isfinite(distance) or distance < 0:
                raise ValueError('Holder withdrawal positions must be finite nonnegative millimetres')
            varied = deepcopy(document)
            next(p for p in varied['parts'] if p['key'] == holder['key'])['holder_retraction_mm'] = distance
            module = module_catalog(varied, source=base_module['source'])
            module['omitted'] = deepcopy(base_module['omitted'])
            module['key'] += f'@holder_retraction_{distance:g}_mm'
            module['diagnostic_geometry_override'] = dict(part_key=holder['key'], holder_retraction_mm=distance,
                scope='Neutral aligned holder translation only; other geometry unchanged')
            modules.append(module)
            module_map[module['key']] = module
            item = deepcopy(holder_baseline)
            item['key'] += f' | holder retraction {distance:g} mm'
            item['selection']['holder_retraction_mm'] = distance
            for placement in item['placements']:
                if placement['module_key'] == base_placement['module_key']:
                    placement['module_key'] = module['key']
            item['scope'] = 'Discrete neutral holder withdrawal position; not continuous motion clearance'
            assemblies.append(item)
    if any(path.read_bytes() != data for path, data in originals.items()):
        raise RuntimeError('Configuration changed while capturing assembly combinations; repeat after edits settle')
    if any(path.read_bytes() != data for path, data in implementation.items()):
        raise RuntimeError('Implementation changed while capturing assembly combinations; repeat after edits settle')
    return dict(schema='temsim-material-collision-input-v1', units='mm',
                source_files=[dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(v).hexdigest()) for p, v in sorted(originals.items())],
                implementation_sha256={p.relative_to(root).as_posix(): hashlib.sha256(data).hexdigest()
                    for p, data in sorted(implementation.items())},
                capture_scope='Static catalog assembly combinations; component poses and the filter mount are included by posed_solids.',
                holder_retractions_mm=list(holder_retractions),
                material_liner_scope='physical_adapter' if liner_adapter else 'excluded_pending_physical_adapter',
                modules=modules, assemblies=assemblies)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, default=Path('outputs/cad_collision_20261011'))
    parser.add_argument('--audit', type=Path, help='Frozen JSON input to audit in the optional CAD interpreter')
    parser.add_argument('--limit', type=int, help='Explicit diagnostic assembly count, not full acceptance')
    parser.add_argument('--no-material-liners', action='store_true', help='Exclude physical liners for a clearly scoped early diagnostic')
    parser.add_argument('--holder-retractions', type=float, nargs='*', default=[0., 50., 160., 320., 400.],
                        help='Additional neutral full-corrector holder positions in mm; empty list disables this explicit diagnostic')
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    auditor = load_auditor(root)
    if not args.audit:
        value = prepare(root, material_liners=not args.no_material_liners, holder_retractions=args.holder_retractions)
        (args.output / 'inputs.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(dict(assemblies=len(value['assemblies']), modules=len(value['modules']), material_liner_scope=value['material_liner_scope'])), flush=True)
        return
    document = json.loads(args.audit.read_text(encoding='utf-8'))
    captured_count = len(document['assemblies'])
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        document['assemblies'] = document['assemblies'][:args.limit]
    completed = []
    def progress(row):
        print(json.dumps(dict(assembly=row['key'], collisions=len(row['collisions']),
                              contacts=len(row['contacts']), uncertain=len(row['uncertain_occupancy_intersections']),
                              minimum_positive_clearance_mm=row['minimum_positive_material_clearance_mm'], seconds=row['elapsed_s'])), flush=True)
        (args.output / 'latest-assembly.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        completed.append(row)
        partial = dict(status='running', partial_diagnostic=True, source_input=str(args.audit),
                       source_file_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                       captured_assembly_count=captured_count, completed_assembly_count=len(completed),
                       assemblies=completed)
        (args.output / 'partial-report.json').write_text(json.dumps(partial, indent=2, allow_nan=False), encoding='utf-8')
    try:
        report = auditor.audit_document(document, progress=progress)
    except BaseException as exc:
        partial = dict(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       error=f'{type(exc).__name__}: {exc}', partial_diagnostic=True,
                       source_input=str(args.audit),
                       source_file_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest(),
                       captured_assembly_count=captured_count, completed_assembly_count=len(completed),
                       assemblies=completed)
        (args.output / 'partial-report.json').write_text(json.dumps(partial, indent=2, allow_nan=False), encoding='utf-8')
        raise
    report.update(captured_assembly_count=captured_count, audited_assembly_count=len(document['assemblies']),
                  material_liner_scope=document['material_liner_scope'],
                  partial_diagnostic=len(document['assemblies']) < captured_count,
                  source_input=str(args.audit), source_file_sha256=hashlib.sha256(args.audit.read_bytes()).hexdigest())
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'partial-report.json').write_text(json.dumps(dict(status='completed',
        final_report='report.json', partial_diagnostic=report['partial_diagnostic']), indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'audited_assembly_count', 'collision_occurrences', 'elapsed_s', 'statistics')}), flush=True)


if __name__ == '__main__':
    main()
