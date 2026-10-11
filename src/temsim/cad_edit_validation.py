"""Immutable requests and receipts for interactive CAD geometry checking.

No CAD kernel is loaded here. The GUI submits the request to its owned worker
and only presents a receipt while the complete draft identity still matches.
"""
from __future__ import annotations

from copy import deepcopy
import math

from temsim.cad_optical_binding import build_optical_binding_report, canonical_digest


def build_edit_validation_request(document, catalog, *, runtime_values=None, aperture_indices=None,
                                  request_id=None):
    context = catalog.get('edit_context', {})
    if context and context.get('draft_sha256') != canonical_digest(document):
        raise ValueError('Assembly geometry catalog belongs to a different draft')
    binding_document = context.get('binding_document', document)
    binding = build_optical_binding_report(binding_document, catalog,
                                          runtime_values=context.get('binding_runtime_values', runtime_values),
                                          aperture_indices=aperture_indices)
    # A partial subassembly is inspected in its owning module coordinates,
    # while staleness still belongs to the actual editable source document.
    binding['runtime_values_sha256'] = canonical_digest(runtime_values or {})
    catalog = deepcopy(catalog)
    if 'edit_context' in catalog:
        # The kernel and UI receipt need identities, not a second embedded
        # copy of every imported B-rep/source mesh in the owner document.
        catalog['edit_context'].pop('binding_document', None)
        catalog['edit_context'].pop('binding_runtime_values', None)
    if 'modules' not in catalog:
        catalog = dict(schema='temsim-analytic-material-cad-v1', units='mm', modules=[catalog])
    if catalog.get('units', 'mm') != 'mm':
        raise ValueError('CAD geometry checks require millimetre coordinates')
    assemblies = catalog.get('assemblies', ())
    if assemblies:
        if len(assemblies) != 1:
            raise ValueError('Interactive optical checks require one explicitly selected assembly')
        assembly = assemblies[0]
        owners = [p for p in assembly['placements'] if p['module_key'] == binding['module_key']]
        if len(owners) != 1:
            raise ValueError('Optical probes require one unambiguous installed owner module')
        placement = owners[0]
        if any(key in placement for key in ('rotation', 'rotation_matrix', 'transform')):
            raise ValueError('Module placement must be the captured translation; branch and component rotations are already applied')
        offset = list(placement.get('offset_mm', (0., 0., 0.)))
        if len(offset) != 3 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in offset):
            raise ValueError('Invalid captured optical module translation')
        binding['assembly_optical_scope'] = dict(assembly_key=assembly['key'],
            coordinate_frame='global_column_mm', module_offset_mm=offset,
            checked_materials='All placed modules and physical liners in this captured assembly.',
            transform_rule='Probe shape remains owner-module local in the request; the kernel adds the captured module translation exactly once.')
        for probe in binding['optical_probes']:
            probe.update(assembly_key=assembly['key'], module_offset_mm=offset,
                         shape_coordinate_frame='owner_module_column_mm')
            if placement.get('instance_key') is not None:
                probe['owner_instance_key'] = placement['instance_key']
        for row in binding['optical_references']:
            row.update(assembly_key=assembly['key'], module_offset_mm=offset,
                       global_origin_mm=[float(v)+float(d) for v, d in zip(row['origin_mm'], offset)],
                       global_normal=list(row['normal']))
    binding['catalog_sha256'] = canonical_digest(catalog)
    request = dict(op='validate_catalog', draft_digest=canonical_digest(document),
        catalog=catalog, optical_probes=deepcopy(binding['optical_probes']),
        binding_report=binding, binding_report_sha256=canonical_digest(binding),
        scope=(context.get('note') or 'Owning-module material intersections and explicitly listed configured optical probes. '
              'Resolved assembly placement is checked only if catalog.assemblies is supplied.'))
    if request_id is not None:
        request['request_id'] = str(request_id)
    return request


def validation_is_current(request, document, *, runtime_values=None, aperture_indices=None):
    """Operating openings are part of the identity even without a TOML edit."""
    binding = request['binding_report']
    return (canonical_digest(document) == request['draft_digest']
        and canonical_digest(runtime_values or {}) == binding['runtime_values_sha256']
        and canonical_digest(aperture_indices or {}) == binding['aperture_indices_sha256'])


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid geometry receipt {name}')
    return float(value)


def finalize_edit_validation(request, receipt, *, document=None, runtime_values=None, aperture_indices=None):
    """Reject stale, incomplete or unrelated evidence; never mutate a draft.

    Geometry success cannot override an unresolved change to a configured
    acceptance boundary. The full receipt remains available for review.
    """
    if document is not None and not validation_is_current(request, document,
            runtime_values=runtime_values, aperture_indices=aperture_indices):
        raise ValueError('Geometry result is stale; the current draft or operating aperture changed')
    if receipt.get('status') != 'complete' or receipt.get('request_sha256') != canonical_digest(request):
        raise ValueError('Geometry result does not belong to this completed request')
    if receipt.get('draft_digest') != request['draft_digest']:
        raise ValueError('Geometry result has a different draft identity')
    if request.get('request_id') is not None and receipt.get('request_id') != request['request_id']:
        raise ValueError('Geometry result has a different request ID')
    result = receipt.get('result', {})
    geometry = result.get('geometry_report')
    if not isinstance(geometry, dict):
        raise ValueError('CAD engine did not return a geometry report')
    if result.get('schema') != 'temsim-cad-geometry-validation-v1' or geometry.get('status') != 'completed':
        raise ValueError('CAD engine did not complete the requested geometry checks')
    if result.get('binding_report_sha256') != request['binding_report_sha256']:
        raise ValueError('Geometry report does not cover the captured optical bindings')
    if result.get('catalog_sha256') != canonical_digest(request['catalog']):
        raise ValueError('Geometry report checked a different material catalog')
    unsigned = {k: v for k, v in result.items() if k != 'report_sha256'}
    if result.get('report_sha256') != canonical_digest(unsigned):
        raise ValueError('Geometry report integrity check failed')
    expected_ids = {p['id'] for p in request['optical_probes']}
    actual = result.get('probe_results')
    if not isinstance(actual, list) or len({p.get('id') for p in actual}) != len(actual) or {p.get('id') for p in actual} != expected_ids:
        raise ValueError('Geometry report is missing, duplicating or inventing an optical probe')
    binding = request['binding_report']
    blockers = [deepcopy(item) for item in binding['warnings'] if item.get('blocking')]
    warnings = [deepcopy(item) for item in binding['warnings'] if not item.get('blocking')]
    # Worker supplies exact per-module/assembly pair records; the finalizer
    # intentionally does not deduce material distribution from an envelope.
    reports = geometry.get('assemblies', [])
    expected = request['catalog'].get('assemblies') or request['catalog']['modules']
    if len(reports) != len(expected) or {m.get('key') for m in reports} != {m['key'] for m in expected}:
        raise ValueError('Geometry report did not check every requested module or assembly')
    for row in reports:
        if row.get('collisions'):
            blockers.append(dict(code='material_overlap', module_key=row['key'],
                                 message='Declared material bodies intersect.', records=row['collisions']))
    for row in actual:
        requested = next(p for p in request['optical_probes'] if p['id'] == row['id'])
        if row.get('kind') != requested['kind'] or row.get('module_key') != requested['module_key']:
            raise ValueError('Optical probe coordinate scope or kind changed')
        if requested.get('assembly_key') is not None:
            if (row.get('assembly_key') != requested['assembly_key']
                    or row.get('coordinate_frame') != 'global_column_mm'
                    or row.get('module_offset_mm') != requested['module_offset_mm']):
                raise ValueError('Optical probe did not check the requested assembly coordinates')
        elif row.get('assembly_key') is not None:
            raise ValueError('Module optical probe unexpectedly changed to an assembly scope')
        # These signed-off booleans are accompanied by explicit measured
        # volumes; require finite evidence, not a truthy string.
        if row.get('status') not in {'passed', 'failed', 'unresolved'}:
            raise ValueError('Optical probe has no explicit geometry outcome')
        for name in ('probe_volume_mm3', 'uncovered_volume_mm3'):
            if row.get(name) is not None:
                _finite_nonnegative(row[name], name)
        for item in row.get('material_blockers', []) + row.get('unresolved_intersections', []):
            _finite_nonnegative(item.get('volume_mm3'), 'intersection volume')
        if row['status'] != 'passed' or row.get('material_blockers') or row.get('unresolved_intersections') or (row.get('uncovered_volume_mm3') or 0.) > 1e-9:
            blockers.append(dict(code='optical_geometry_mismatch', probe_id=row['id'],
                                 message='Material geometry conflicts with the configured optical region.', evidence=row))
    unresolved = [row['key'] for row in reports if row.get('unresolved_solids') or row.get('sub_tolerance_intersections')]
    return dict(schema='temsim-cad-edit-validation-v1', draft_digest=request['draft_digest'],
        request_sha256=canonical_digest(request), binding_report_sha256=request['binding_report_sha256'],
        status='conflict' if blockers else 'checked_with_declared_scope',
        can_apply=not blockers, blockers=blockers, warnings=warnings,
        limitations=deepcopy(binding['limitations']), incomplete_material_modules=unresolved,
        optical_binding=binding, geometry_report=geometry, probe_results=actual,
        scope=request['scope'], electromagnetic_fields_recomputed=False,
        optical_roles_inferred_from_step=False,
        assembly_edit_context=deepcopy(request['catalog'].get('edit_context', {})))
