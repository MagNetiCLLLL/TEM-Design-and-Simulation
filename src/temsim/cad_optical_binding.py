"""Read-only links between declared optical datums and material CAD.

The result is in the same owning-module millimetres as ``module_catalog``.
It never infers an optical element, field law or active area from a STEP body.
Finite probe slabs are geometry checks, not electron propagation or FEM.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

import numpy as np

from temsim.cad_geometry import annulus, box, transformed
from temsim.component_representation import SHARED_DEFLECTOR_HOSTS, non_material_role
from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
from temsim.optics.field_hardware_binding import resolve_field_hardware_parts


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _frame(part, by_key, z=None, *, mechanical=False):
    from temsim.energy_filter_model_3d import (
        supports_energy_filter_part, energy_filter_frame_mm,
        FILTER_TO_COLUMN_ROTATION, energy_filter_origin_mm, _sector,
    )
    if supports_energy_filter_part(part) and part['key'] != 'energy_filter_entrance_aperture':
        if part.get('mechanical_profile') == 'tapered_sector_prism':
            _, radius, angle, entrance, _, _ = _sector(by_key)
            # A declared curved reference orbit has no single flat lens plane.
            # This is its mid-orbit tangent datum, not an inferred thin lens.
            origin = entrance + (radius*np.sin(angle/2), 0., -radius*(1-np.cos(angle/2)))
            tangent = np.array((np.cos(angle/2), 0., -np.sin(angle/2)))
            y = np.array((0., 1., 0.))
            basis = np.column_stack((np.cross(y, tangent), y, tangent))
        else:
            origin, basis = energy_filter_frame_mm(part, by_key)
        mount = np.asarray(FILTER_TO_COLUMN_ROTATION, dtype=float)
        origin = mount @ origin + (0., 0., energy_filter_origin_mm(by_key))
        basis = mount @ basis
    else:
        origin = np.array((0., 0., float(z if z is not None else part.get(
            'optical_reference_local_z_mm', part.get('local_center_z_mm', 0.)))))
        basis = np.eye(3)
    transform = mechanical_part_transform_mm if mechanical else effective_part_transform_mm
    rotation, translation = transform(part, by_key)
    origin, basis = rotation @ origin + translation, rotation @ basis
    if not np.isfinite(origin).all() or not np.isfinite(basis).all():
        raise ValueError(f"{part['key']}: non-finite optical frame")
    return origin, basis


def _pose(shape, frame):
    origin, basis = frame
    return transformed(shape, rotation=basis.tolist(), offset=origin.tolist())


def _positive(value, name):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return float(value)


def _material_ids(module, key, region=None):
    from temsim.cad_assembly_validation import occupancy_kind
    return [str(row.get('instance_key', f"{row['key']}/{row['region']}"))
            for row in module.get('solids', ()) if row['key'] == key
            and (region is None or row['region'] == region) and occupancy_kind(row) == 'material']


def build_optical_binding_report(document, catalog, *, runtime_values=None, aperture_indices=None):
    """Snapshot configured optics and produce explicit, limited OCC probes.

    ``document`` is a resolved module document, never an assembly translated a
    second time. A whole catalog is accepted only if it contains this module.
    Probes use the same module frame; assembly placements remain separate.
    """
    original = deepcopy(document)
    from temsim.optics.shared_deflectors import resolve_shared_deflector_parts
    document = resolve_field_hardware_parts(resolve_shared_deflector_parts(original))
    module_key = str(document['module']['key'])
    modules = catalog.get('modules', [catalog])
    matches = [m for m in modules if str(m['key']) == module_key]
    if len(matches) != 1:
        raise ValueError('Optical binding requires exactly one matching catalog module')
    module = matches[0]
    by_key = {p['key']: p for p in document['parts']}
    if len(by_key) != len(document['parts']):
        raise ValueError('Duplicate optical/component key')
    if any(row['key'] not in by_key for row in module.get('solids', ())):
        raise ValueError('Catalog material refers to a component absent from the draft')
    source_digest = canonical_digest(original)
    linked = module.get('source_document_sha256')
    if linked is not None and linked != source_digest:
        raise ValueError('Catalog was generated from a different document')
    records, probes, mechanical, warnings, limitations = [], [], [], [], []
    runtime_values, aperture_indices = runtime_values or {}, aperture_indices or {}
    for part in document['parts']:
        key, profile = part['key'], part.get('mechanical_profile', '')
        role = non_material_role(part)
        custom = str(part.get('mechanical_part_role', '')).startswith('custom_')
        detector = profile in {'transverse_eds_detector_array', 'zebra_eels_detector',
                               'retractable_detector_plane', 'camera_sensor_plane'}
        from temsim.energy_filter_model_3d import supports_energy_filter_part
        branch_component = supports_energy_filter_part(part)
        optical = (not custom and (detector or branch_component or (not part.get('mechanical_only') and (
            'optical_reference_local_z_mm' in part or profile == 'magnetic_lens_assembly'))))
        if not optical:
            mechanical.append(dict(key=key, optical_role=None,
                solid_ids=_material_ids(module, key),
                note='Mechanical geometry does not create an optical field or acceptance region.'))
            continue
        frame = _frame(part, by_key)
        material_frame = _frame(part, by_key, mechanical=True)
        origin, basis = frame
        physical_host = part.get('physical_host_key') or SHARED_DEFLECTOR_HOSTS.get(key)
        reference_host = part.get('optical_reference_host_key')
        for host in (physical_host, reference_host):
            if host and host not in by_key:
                raise ValueError(f'{key}: missing optical/physical host {host}')
        record = dict(key=key, module_key=module_key, role=role or ('detector' if detector else 'configured_optical_reference'),
            origin_mm=origin.tolist(), normal=basis[:, 2].tolist(), basis=basis.tolist(),
            physical_host_key=physical_host, optical_reference_host_key=reference_host,
            physical_host_solid_ids=_material_ids(module, physical_host or key),
            mechanical_reference_origin_mm=material_frame[0].tolist(),
            mechanical_reference_normal=material_frame[1][:, 2].tolist(),
            optical_reference_only=bool(reference_host and not physical_host),
            parameters_are_inferred_from_geometry=False, acceptance=None)
        if 'interaction_centers_local_z_mm' in part:
            record['interaction_centers_mm'] = [_frame(part, by_key, z)[0].tolist()
                                               for z in part['interaction_centers_local_z_mm']]

        def probe(name, kind, shape, *, expected=(), scope, probe_frame=None):
            probes.append(dict(id=f'{module_key}/{key}/{name}', module_key=module_key,
                part_key=key, kind=kind, shape=_pose(shape, frame if probe_frame is None else probe_frame),
                expected_solid_ids=list(expected), scope=scope))

        from temsim.part_model_apertures import is_strip_aperture, _opening, plate_thickness_field
        if is_strip_aperture(part) or profile == 'fixed_differential_pumping_aperture':
            if is_strip_aperture(part):
                diameter, paths, offsets = _opening(part, runtime_values.get(key), aperture_indices.get(key, 0))
                thickness = part.get(plate_thickness_field(part), 0.)
            else:
                diameter, paths, offsets = part.get('mechanical_bore_diameter_mm'), (), (0., 0.)
                thickness = part.get('plate_thickness_mm', 0.)
            record['acceptance'] = dict(kind='circular_opening', diameter_mm=diameter,
                offsets_mm=list(offsets), thickness_mm=thickness,
                parameter_paths=[list(path) for path in paths])
            if diameter is None or thickness <= 0:
                limitations.append(f'{key}: opening or plate thickness is unspecified; no passage certification.')
            elif diameter > 0:
                # Check the whole physical plate depth. A newly enlarged hole
                # is separately restricted below: passage alone cannot detect it.
                shape = transformed(annulus(-thickness/2, thickness/2, 0., diameter/2),
                                    offset=(*offsets, 0.))
                probe('opening', 'clearance', shape, probe_frame=material_frame,
                      scope='Configured mechanical circular passage through the physical plate thickness; optical-axis tilt is excluded.')
                if not np.allclose(frame[1], material_frame[1], atol=1e-12, rtol=0):
                    limitations.append(f'{key}: optical-stop tilt is independent of the fixed mechanical plate. '
                                       'This geometry check does not certify equivalence of the effective optical acceptance and the physical tilted-hole transport.')
            else:
                limitations.append(f'{key}: closed aperture; no nonempty passage probe.')
        elif profile in {'retractable_detector_plane', 'camera_sensor_plane'}:
            outer = _positive(part['outer_width_mm'], f'{key}.outer_width_mm')
            inner = float(part.get('inner_width_mm', 0.))
            depth = float(part.get('local_end_z_mm', 0.))-float(part.get('optical_reference_local_z_mm', 0.))
            record['acceptance'] = dict(kind='square' if profile == 'camera_sensor_plane' else 'annulus',
                outer_width_mm=outer, inner_width_mm=inner, reception_datum='upstream_surface')
            targets = _material_ids(module, key)
            if targets and depth > 0:
                slab = min(depth/2, .001)
                shape = box((0., 0., slab/2), (outer, outer, slab)) if profile == 'camera_sensor_plane' else annulus(0., slab, inner/2, outer/2)
                probe('reception_surface', 'coverage', shape, expected=targets,
                    scope='Configured detector reception footprint in a thin downstream material slab, not detector efficiency.')
            else:
                limitations.append(f'{key}: material sensor distribution is unresolved; reception coverage not certified.')
        elif profile == 'zebra_eels_detector':
            from temsim.energy_filter_model_3d import _zebra_meshes
            record['acceptance'] = dict(kind='configured_spectral_strips', alignment='logical guide; not a second material sensor')
            for mesh in _zebra_meshes(part):
                corners = np.asarray(mesh.vertices, dtype=float)
                record.setdefault('readout_regions', []).append(dict(region=mesh.region,
                    corners_mm=(corners @ basis.T + origin).tolist(), material_sensor=mesh.region.startswith('active_strip_')))
                if not mesh.region.startswith('active_strip_'):
                    continue
                targets = _material_ids(module, key, mesh.region+'_scintillator')
                if not targets:
                    limitations.append(f'{key}/{mesh.region}: no declared conversion material; coverage unverified.')
                    continue
                slab = min(_positive(part['mechanical_scintillator_thickness_mm'], key)/2, .001)
                low, high = corners.min(axis=0), corners.max(axis=0)
                probe(mesh.region, 'coverage', box(((low[0]+high[0])/2, (low[1]+high[1])/2, slab/2),
                    (high[0]-low[0], high[1]-low[1], slab)), expected=targets,
                    scope='Original spectral strip at the saved branch path datum; alignment remains a logical guide.')
        elif profile == 'transverse_eds_detector_array':
            record['acceptance'] = dict(kind='angular_photon_acceptance',
                definition_file=part.get('eds_detector_definition_file'),
                planar_active_area_inferred=False,
                declared_parameters={k: deepcopy(v) for k, v in part.items() if k.startswith(('takeoff_', 'azimuth_', 'solid_angle_'))})
            limitations.append(f'{key}: angular acceptance does not intersect the illustrative mechanical Si disks; no planar acceptance probe.')
        elif profile == 'xo_energy_slit_assembly':
            from temsim.energy_filter_model_3d import slit_opening_mm
            gap, center = slit_opening_mm(part, runtime_values.get(key))
            height, thickness = float(part['clear_height_mm']), float(part['blade_thickness_mm'])
            record['acceptance'] = dict(kind='rectangular_energy_slit', gap_mm=gap,
                center_mm=center, height_mm=height, thickness_mm=thickness)
            if gap > 0:
                probe('slit_opening', 'clearance', box((center, 0., 0.), (gap, height, thickness)),
                    scope='Saved slit opening through blade thickness in the mounted branch frame.')
        if profile == 'tapered_sector_prism':
            record['datum_scope'] = 'Midpoint and tangent of the declared curved reference orbit, not a thin-lens plane.'

        model = part.get('model_3d', {})
        edited = (model.get('base', {}).get('kind', 'existing') != 'existing'
                  or any(f.get('enabled', True) for f in model.get('features', ()))
                  or list(model.get('transform', {}).get('scale_xy', (1., 1.))) != [1., 1.])
        if edited and record['acceptance'] is not None:
            # Enlarging a bore or adding an unmodelled aperture changes the
            # physical stop even if the old opening is still perfectly clear.
            record['requires_optical_review'] = True
            warnings.append(dict(key=key, code='acceptance_geometry_edit_requires_optical_review',
                blocking=True, message='CAD base/scale/Boolean edits do not update this configured acceptance. Preserve the draft and edit the authoritative opening/readout parameters or explicitly reconcile the optical model.'))
        elif edited:
            warnings.append(dict(key=key, code='analytic_field_not_recomputed', blocking=False,
                message='Mechanical shape changed; configured analytical fields and aberration coefficients were not inferred or recalculated.'))
        records.append(record)
    return dict(schema='temsim-cad-optical-binding-v1', module_key=module_key, units='mm',
        coordinate_frame='owning_module_column_coordinates; energy-filter mounting already applied once',
        document_sha256=source_digest, catalog_sha256=canonical_digest(catalog),
        catalog_source_link='verified' if linked is not None else 'legacy_catalog_source_not_attested',
        runtime_values_sha256=canonical_digest(runtime_values), aperture_indices_sha256=canonical_digest(aperture_indices),
        optical_references=records, mechanical_only=mechanical, optical_probes=probes,
        warnings=warnings, limitations=limitations,
        scope='Configured reference and acceptance geometry only. No new optical role, electromagnetic solve, propagation or OEM qualification is inferred from CAD.')
