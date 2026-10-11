"""Capture one installed CAD assembly with an unsaved component draft.

Captured module translations are retained, never solved again from a guessed
configuration. Only declared source files are read. This is preparation for
the isolated geometry worker, not an assembly or optical calculation.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import math
from pathlib import Path

from temsim.assembly_model_3d import _document_value
from temsim.cad_optical_binding import canonical_digest


def _assembly_identity(assembly):
    if assembly is None:
        return None
    return canonical_digest(dict(root=str(Path(assembly.root).resolve()),
        paths=list(assembly.selected_module_paths),
        modules=[dict(key=m.key, source_file=m.source_file, geometry=_document_value(m.geometry),
            entrance=m.entrance_z_mm, exit=m.exit_z_mm) for m in assembly.modules],
        parts=[dict(module=p.module_key, key=p.key, start=p.start_z_mm, center=p.center_z_mm,
                    end=p.end_z_mm, data=_document_value(p.data)) for p in assembly.parts]))


def _origin_identity(origins):
    return canonical_digest({str(Path(p).resolve()): float(value) for p, value in (origins or {}).items()})


def _source_records(captured):
    rows = []
    for path, data in sorted((captured or {}).items(), key=lambda item: str(item[0])):
        path = Path(path).resolve()
        if data is None or not path.is_file() or path.read_bytes() != data:
            raise ValueError('Installed assembly source changed or is unavailable; reload the assembly before checking it: '+str(path))
        stat = path.stat()
        rows.append(dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                         bytes=len(data), mtime_ns=stat.st_mtime_ns))
    return rows


def assembly_edit_context_is_current(catalog, *, assembly=None, module_origins=None,
        coordinate_source_bytes=None, runtime_values=None, aperture_indices=None, full_source_check=False):
    """Cheap UI refresh uses stats; accepting a worker receipt rehashes files."""
    context = catalog.get('edit_context', {})
    if not context:
        return True
    if canonical_digest(runtime_values or {}) != context['runtime_values_sha256'] or canonical_digest(aperture_indices or {}) != context['aperture_indices_sha256']:
        return False
    if context['scope'] == 'module_only':
        return _assembly_identity(assembly) == context.get('assembly_sha256')
    if (_assembly_identity(assembly) != context['assembly_sha256']
            or _origin_identity(module_origins) != context['module_origins_sha256']):
        return False
    captured = {str(Path(p).resolve()): v for p, v in (coordinate_source_bytes or {}).items()}
    try:
        for row in context['source_files']:
            path = Path(row['path'])
            value = captured.get(str(path))
            if value is None or hashlib.sha256(value).hexdigest() != row['sha256']:
                return False
            stat = path.stat()
            if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
                return False
            if full_source_check and hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                return False
    except OSError:
        return False
    return True


def _replace_subassembly(before, parent_path, child_path, draft):
    from temsim.subassemblies import definitions, resolved_origins, ownership, reflow
    from temsim.component_operations import translated_component
    matches = [(entry, child) for entry, path, child in definitions(before, parent_path) if path == child_path]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError('A draft must have one explicit subassembly owner')
    entry, old = matches[0]
    offset = resolved_origins(before)[entry['key']]
    previous = {p['key'] for p in old['parts']}
    candidate = deepcopy(before)
    kept = [p for p in candidate['parts'] if p['key'] not in previous]
    new = [translated_component(p, offset) for p in draft['parts']]
    if {p['key'] for p in kept} & {p['key'] for p in new}:
        raise ValueError('Draft subassembly duplicates another installed component key')
    candidate['parts'] = kept + new
    owners = ownership(before, parent_path)
    owners.update({p['key']: entry['key'] for p in new})
    return reflow(before, candidate, parent_path, owners=owners)


def _placed_preview(assembly, documents, origins):
    """Reuse the production vacuum span law with fixed captured placements."""
    from temsim.column.module_assembly import (
        ModulePart, AssemblyPart, VacuumLinerSegment, _freeze,
        _module_vacuum_segments, _module_continuous_vacuum_tube,
    )
    modules, parts = [], []
    for old in assembly.modules:
        data = documents[old.key]
        module_parts = tuple(ModulePart(p['key'], p.get('name', p['key']), p.get('branch', 'main'),
            float(p['local_start_z_mm']), float(p['local_center_z_mm']), float(p['local_end_z_mm']),
            float(p['length_mm']), p.get('parent_key'), _freeze(deepcopy(p))) for p in data['parts'])
        geometry = deepcopy(data.get('geometry', _document_value(old.geometry)))
        module = replace(old, parts=module_parts, geometry=_freeze(geometry))
        modules.append(module)
        z = origins[old.key]
        parts.extend(AssemblyPart(module.key, module.source_file, p.key, p.name, p.branch,
            p.start_z_mm+z, p.center_z_mm+z, p.end_z_mm+z, p.length_mm, p.parent_key, p.data) for p in module_parts)
    if len({p.key for p in parts}) != len(parts):
        raise ValueError('Duplicate part keys across installed modules')
    bores, liners = [], []
    for module in modules:
        z = origins[module.key]
        segments = _module_vacuum_segments(module, z, parts)
        bores.extend(segments)
        wall = float(module.geometry['vacuum_liner_wall_thickness_mm'])
        tube = _module_continuous_vacuum_tube(module, z)
        if tube is not None:
            start, end, inner, outer = tube
            liners.append(VacuumLinerSegment('@vacuum_liner:c1_c2_to_upper_objective',
                'C1/C2 to upper objective continuous vacuum tube', start, end, inner, outer, (outer-inner)/2))
        for s in segments:
            if tube is None or s.end_z_mm <= tube[0] or s.start_z_mm >= tube[1]:
                liners.append(VacuumLinerSegment('@vacuum_liner:'+s.key, s.name+' vacuum liner',
                    s.start_z_mm, s.end_z_mm, s.inner_diameter_mm, s.inner_diameter_mm+2*wall, wall))
    return replace(assembly, modules=tuple(modules), parts=tuple(parts),
                   vacuum_bore_segments=tuple(bores), vacuum_liner_segments=tuple(liners))


def build_current_assembly_edit_catalog(session, *, project_root=None, project_paths=(), module_origins=None,
        coordinate_source_bytes=None, runtime_values=None, aperture_indices=None, assembly=None):
    """Return a worker catalog for one active assembly or an explicit fallback.

    Partial subassembly drafts are embedded in their declared owner. Synthetic
    physical liners are generated once in global coordinates from that preview.
    No files or live assembled objects are changed.
    """
    from temsim.cad_catalog import module_catalog, posed_solids
    from temsim.part_model_document import PartModelDocument
    from temsim.optics.field_hardware_binding import resolve_field_hardware_parts
    from temsim.optics.shared_deflectors import resolve_shared_deflector_parts
    runtime_values, aperture_indices = runtime_values or {}, aperture_indices or {}
    context = dict(scope='module_only', draft_sha256=canonical_digest(session.document),
        draft_path=str(session.path), assembly_sha256=_assembly_identity(assembly),
        runtime_values_sha256=canonical_digest(runtime_values), aperture_indices_sha256=canonical_digest(aperture_indices),
        binding_document=deepcopy(session.document), source_files=[], needs_reassemble=False,
        note='Only this module is checked; no active assembly placement is inferred.')

    def standalone():
        # An external/inactive file cannot borrow current operating apertures.
        context['binding_runtime_values'] = {}
        module = module_catalog(session.document, source=session.path, aperture_indices=aperture_indices)
        return dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=[module], edit_context=context)

    if assembly is None or project_root is None or not project_paths:
        return standalone()
    root = Path(project_root).resolve()
    active_paths = [(root/str(p)).resolve() for p in project_paths]
    selected = [(root/str(p)).resolve() for _, p in assembly.selected_module_paths]
    if root != Path(assembly.root).resolve() or active_paths != selected:
        raise ValueError('Current assembly selection does not match the captured editor context')
    # Do not examine unrelated file layouts to guess membership.
    captured = {Path(p).resolve(): value for p, value in (coordinate_source_bytes or {}).items()}
    if session.path not in active_paths and session.path not in captured:
        return standalone()
    sources = _source_records(captured)
    if not captured or any(path not in captured for path in active_paths):
        raise ValueError('The installed source snapshot is incomplete; reload the assembly')
    session.assert_source_current()
    documents, origins, affected, originals = {}, {}, [], {}
    origin_map = {Path(p).resolve(): float(value) for p, value in (module_origins or {}).items()}
    for old in assembly.modules:
        path = (root/old.source_file).resolve()
        if path not in active_paths or path not in origin_map:
            raise ValueError('An installed module has no captured global placement')
        z = origin_map[path]
        actual = [p.center_z_mm-float(p.data['local_center_z_mm']) for p in assembly.parts if p.module_key == old.key]
        if not math.isfinite(z) or not actual or any(not math.isclose(v, z, rel_tol=0., abs_tol=1e-8) for v in actual):
            raise ValueError('Captured module translation does not match the installed assembly')
        baseline = PartModelDocument(path).document
        originals[old.key] = deepcopy(baseline)
        candidate = deepcopy(session.document) if path == session.path else _replace_subassembly(baseline, path, session.path, session.document)
        if candidate is not None:
            if candidate['module']['key'] != old.key:
                raise ValueError('Draft changes the installed module identity')
            affected.append(old.key)
        else:
            candidate = baseline
        documents[old.key] = resolve_field_hardware_parts(resolve_shared_deflector_parts(candidate))
        origins[old.key] = z
    if not affected:
        # A shared Tip/other definition without an explicit subassembly
        # placement needs its own adapter; never infer placement from a name.
        return standalone()
    if len(affected) != 1:
        raise ValueError('Draft has multiple installed owners; inspect each explicit instance separately')
    modules = [module_catalog(doc, source=root/next(m.source_file for m in assembly.modules if m.key==key),
        runtime_values=runtime_values, aperture_indices=aperture_indices) for key, doc in documents.items()]
    preview = _placed_preview(assembly, documents, origins)
    from temsim.cad_stage_holder import physical_liner_rows, liner_material_decisions
    rows = physical_liner_rows(preview, runtime_values=runtime_values, aperture_indices=aperture_indices)
    by_key = {p['key']: p for p in rows}
    solids = []
    for index, row in enumerate(rows):
        for spec in posed_solids(row, by_key):
            data = asdict(spec)
            data.update(instance_key=f'{spec.key}/{spec.region}/segment-{index:04d}', color=list(spec.color))
            solids.append(data)
    liner_key = '@physical_liners:'+canonical_digest(solids)
    modules.append(dict(key=liner_key, name='Resolved physical vacuum liners', solids=solids, omitted=[]))
    placements = [dict(module_key=key, offset_mm=[0., 0., z]) for key, z in origins.items()]
    placements.append(dict(module_key=liner_key, offset_mm=[0., 0., 0.]))
    changed = bool(session.dirty)
    context.update(scope='current_assembly_captured_placements', source_files=sources,
        module_origins_sha256=_origin_identity(module_origins), binding_document=documents[affected[0]],
        binding_runtime_values=deepcopy(runtime_values), affected_module_key=affected[0],
        needs_reassemble=changed, note='Current captured module placements plus draft geometry; no module positions were automatically solved again. Save and assemble before using changed geometry for physical calculations.')
    # Recheck all bytes after reading dependent definitions and building shapes.
    _source_records(captured)
    return dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=modules,
        assemblies=[dict(key='current_assembly_with_draft', placements=placements,
            material_liner_scope='physical_adapter', liner_material_decisions=liner_material_decisions(preview),
            scope=context['note'])], edit_context=context)
