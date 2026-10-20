"""Detector material must preserve existing reception and branch datums."""
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_detectors import (
    EDS_FIELDS, EELS_FIELDS, EDS_PROFILE, EELS_PROFILE, STATUS, REFERENCE,
    detector_solids, eds_head_geometry, validate_detector_material_geometry,
)
from temsim.cad_mesh import _manifold, meshes_from_solids
from temsim.detector.eds_geometry import resolve_eds_detector_part_data, EDSDetectorArrayGeometry
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def column():
    document = PartModelDocument(ROOT/'configs/instruments/column/C3_ProbeCorrector.toml').document
    return {row['key']: row for row in document['parts']}


def _declare(part, fields, values):
    part.update(dict(zip(fields, values, strict=True)),
                detector_material_geometry_status=STATUS,
                detector_material_geometry_source=REFERENCE)
    return part


def _eds(column):
    part = resolve_eds_detector_part_data(next(row for row in column.values()
        if row.get('mechanical_profile') == EDS_PROFILE))
    return _declare(part, EDS_FIELDS, (.45, .3, .25, .5))


@pytest.fixture
def eels():
    document = tomllib.loads((ROOT/'configs/subassemblies/energy_filter.toml').read_text(encoding='utf-8'))
    by_key = {row['key']: row for row in document['parts']}
    part = next(row for row in by_key.values() if row.get('mechanical_profile') == EELS_PROFILE)
    _declare(part, EELS_FIELDS, (.01, 1., 1., 6., .5))
    return part, by_key


def test_legacy_detector_has_no_implicit_solid_or_new_thickness(column, eels):
    for part, by_key, fields in ((_eds(column), column, EDS_FIELDS), (*eels, EELS_FIELDS)):
        for key in (*fields, 'detector_material_geometry_status', 'detector_material_geometry_source'):
            part.pop(key)
        assert detector_solids(part, by_key) is None
        part[fields[0]] = .1
        with pytest.raises(ValueError, match='Incomplete'):
            validate_detector_material_geometry(part, by_key)


@pytest.mark.parametrize('field,value', [
    ('mechanical_sensor_thickness_mm', True), ('mechanical_sensor_thickness_mm', float('nan')),
    ('mechanical_support_thickness_mm', 0.), ('mechanical_housing_wall_mm', 2.),
    ('mechanical_back_wall_mm', 4.), ('detector_material_geometry_status', 'OEM_verified'),
])
def test_eds_rejects_invalid_or_unattributed_material_geometry(column, field, value):
    part = _eds(column)
    part[field] = value
    with pytest.raises(ValueError):
        detector_solids(part, column)


def test_eds_heads_retain_side_widths_positions_and_real_angular_definition(column):
    part = _eds(column)
    before = deepcopy(part)
    angular = EDSDetectorArrayGeometry.from_part_data(part)
    geometry = eds_head_geometry(part, column)
    assert geometry['face_half_width_mm'] == 2.25
    assert geometry['housing_length_mm'] == 4.5
    assert geometry['front_half_width_mm'] == 2.8125
    assert geometry['rear_half_width_mm'] == 1.845
    assert len(geometry['heads']) == angular.segment_count == 6
    sample = np.array((0., 0., part['local_center_z_mm']))
    for head, azimuth in zip(geometry['heads'], angular.azimuth_centers_deg, strict=True):
        frame = np.asarray(head['rotation'])
        np.testing.assert_allclose(frame.T@frame, np.eye(3), atol=3e-16)
        assert np.linalg.det(frame) == pytest.approx(1.)
        ray = np.asarray(head['center_mm'])-sample
        np.testing.assert_allclose(ray/np.linalg.norm(ray), frame[:, 2], atol=2e-15)
        assert np.degrees(np.arctan2(-ray[2], np.linalg.norm(ray[:2]))) == pytest.approx(angular.takeoff_angle_deg)
        assert head['azimuth_deg'] == azimuth
    detector_solids(part, column)
    assert EDSDetectorArrayGeometry.from_part_data(part) == angular
    assert part == before


def test_eds_square_die_support_and_taper_have_analytic_material_volume(column):
    part = _eds(column)
    geometry = eds_head_geometry(part, column)
    specs = detector_solids(part, column)
    half = geometry['face_half_width_mm']
    front, rear = geometry['front_half_width_mm'], geometry['rear_half_width_mm']
    depth = geometry['housing_length_mm']
    wall, back = part['mechanical_housing_wall_mm'], part['mechanical_back_wall_mm']
    cavity_front = front-wall
    cavity_rear = front+(rear-front)*(depth-back)/depth-wall
    frustum_volume = lambda a, b, length: 4*length*(a*a+a*b+b*b)/3
    shell_volume = frustum_volume(front, rear, depth)-frustum_volume(cavity_front, cavity_rear, depth-back)
    for head in geometry['heads']:
        leaves = [s for s in specs if s.parameters['index'] == head['index']]
        for spec, thickness in zip(leaves[:2], (part['mechanical_sensor_thickness_mm'],
                                                part['mechanical_support_thickness_mm']), strict=True):
            local = spec.shape['shape']
            assert local['kind'] == 'box'
            np.testing.assert_allclose(local['size_mm'], (2*half, 2*half, thickness))
            assert _manifold(spec.shape, 24).volume() == pytest.approx((2*half)**2*thickness)
            assert spec.parameters['mechanical_footprint_shape'] == 'square'
            assert 'not verified OEM shapes' in spec.note
        assert _manifold(leaves[2].shape, 24).volume() == pytest.approx(shell_volume)
        # Transform the actual native sensor vertices back into the head frame:
        # all four square corners survive, including points outside the old disk.
        mesh, = meshes_from_solids(leaves[:1], 24)
        local_points = (mesh.vertices-np.asarray(head['center_mm']))@np.asarray(head['rotation'])
        assert set(map(tuple, np.round(local_points[:, :2], 9))) == {
            (-half, -half), (-half, half), (half, -half), (half, half)}


def test_all_installed_column_aliases_declare_square_eds_without_changing_acceptance():
    paths = sorted((ROOT/'configs/instruments/column').glob('*.toml'))
    count = 0
    for path in paths:
        by_key = {row['key']: row for row in PartModelDocument(path).document['parts']}
        for part in by_key.values():
            if part.get('mechanical_profile') != EDS_PROFILE:
                continue
            count += 1
            before = deepcopy(by_key)
            assert part['mechanical_footprint_shape'] == 'square'
            assert 'not a measured OEM' in part['mechanical_footprint_source']
            angular = EDSDetectorArrayGeometry.from_part_data(resolve_eds_detector_part_data(part))
            specs = detector_solids(part, by_key)
            assert len(specs) == 3*angular.segment_count == 18
            assert all(spec.parameters['mechanical_footprint_shape'] == 'square' for spec in specs)
            assert by_key == before
    assert count == 7


@pytest.mark.parametrize('profile,shape', [(EDS_PROFILE, 'disk'), (EELS_PROFILE, 'square')])
def test_detector_shape_convention_rejects_incompatible_metadata(column, eels, profile, shape):
    part, by_key = (_eds(column), column) if profile == EDS_PROFILE else eels
    part['mechanical_footprint_shape'] = shape
    with pytest.raises(ValueError, match='mechanical_footprint_shape'):
        detector_solids(part, by_key)


def test_eds_material_closed_nonoverlapping_and_clear_of_full_objective(column):
    from temsim.cad_catalog import posed_solids
    part = _eds(column)
    before = deepcopy(column)
    specs = detector_solids(part, column)
    assert len(specs) == 18
    meshes = meshes_from_solids(specs, 48)
    assert all(np.isfinite(mesh.vertices).all() and not mesh.is_exact for mesh in meshes)
    solids = [_manifold(spec.shape, 48) for spec in specs]
    assert all(s.volume() > 0 for s in solids)
    for i, a in enumerate(solids):
        for b in solids[i+1:]:
            assert (a ^ b).volume() < 1e-8
    # Test actual segmented coil/yoke geometry, not merely the two pole outlines.
    tested = set()
    for other in column.values():
        if other.get('parent_key') != 'objective_lens' or other['key'] == part['key']:
            continue
        for body in posed_solids(other, column):
            tested.add(body.key)
            material = _manifold(body.shape, 48)
            assert all((head ^ material).volume() < 1e-8 for head in solids), body.key
    assert {'objective_lens_excitation_coil', 'objective_lens_yoke',
            'objective_lens_housing', 'objective_upper_pole', 'objective_lower_pole'} <= tested
    assert column == before


def test_eels_faces_match_existing_strips_branch_frame_once_and_no_extra_alignment_sensor(eels):
    from temsim.energy_filter_model_3d import _zebra_meshes, energy_filter_frame_mm
    part, by_key = eels
    before = deepcopy(by_key)
    origin, rotation = energy_filter_frame_mm(part, by_key)
    guides = [g for g in _zebra_meshes(part) if g.region.startswith('active_strip_')]
    specs = detector_solids(part, by_key)
    strips = [s for s in specs if s.region.endswith('_scintillator')]
    assert len(specs) == 7 and len(strips) == len(guides) == 5
    assert not any('alignment' in spec.region for spec in specs)
    for guide, spec in zip(guides, strips, strict=True):
        shape = spec.shape
        assert shape['kind'] == 'transform' and shape['shape']['kind'] == 'box'
        np.testing.assert_array_equal(shape['rotation'], rotation)
        np.testing.assert_array_equal(shape['offset_mm'], origin)
        local = shape['shape']
        lo = np.asarray(local['center_mm'])-np.asarray(local['size_mm'])/2
        hi = np.asarray(local['center_mm'])+np.asarray(local['size_mm'])/2
        np.testing.assert_allclose(lo[:2], guide.vertices.min(axis=0)[:2], atol=1e-14)
        np.testing.assert_allclose(hi[:2], guide.vertices.max(axis=0)[:2], atol=1e-14)
        assert lo[2] == 0. and hi[2] == part['mechanical_scintillator_thickness_mm']
    assert by_key == before


def test_eels_material_is_closed_and_layer_stack_does_not_overlap(eels):
    part, by_key = eels
    specs = detector_solids(part, by_key)
    meshes = meshes_from_solids(specs, 32)
    assert len(meshes) == 7
    solids = [_manifold(spec.shape, 32) for spec in specs]
    for i, solid in enumerate(solids):
        assert solid.volume() > 0
        for other in solids[i+1:]:
            assert (solid ^ other).volume() < 1e-10
    part['mechanical_optical_coupling_depth_mm'] = 20.
    with pytest.raises(ValueError, match='fit ahead'):
        detector_solids(part, by_key)


def test_eels_receiving_regions_are_real_rectangles_in_cad_and_runtime(eels):
    from temsim.energy_filter_model_3d import _zebra_meshes
    from temsim.optics.energy_filter_detector import ZebraEELSDetector

    part, by_key = eels
    assert part['mechanical_footprint_shape'] == 'rectangle'
    guides = _zebra_meshes(part)
    strips = [g for g in guides if g.region.startswith('active_strip_')]
    assert len(strips) == 5
    for guide in strips:
        np.testing.assert_allclose(np.ptp(guide.vertices, axis=0), (28.672, .8, 0.))
    alignment = next(g for g in guides if 'alignment' in g.region)
    np.testing.assert_allclose(np.ptp(alignment.vertices, axis=0), (28.672, 3.584, 0.))
    for spec in detector_solids(part, by_key):
        assert spec.parameters['mechanical_footprint_shape'] == 'rectangle'
    detector = ZebraEELSDetector().validate()
    for alignment_mode, half_height in ((False, .4), (True, 1.792)):
        detector.alignment_mode = alignment_mode
        # Inside a long-strip corner, outside its narrow height, and outside width.
        x_mm = np.array([14., 0., 14.4])
        y_mm = np.array([.9*half_height, 1.1*half_height, 0.])
        np.testing.assert_array_equal(detector.recording_mask(x_mm*1e-3, y_mm*1e-3),
                                      [True, False, False])
        detector.inserted = False
        assert not detector.recording_mask(x_mm*1e-3, y_mm*1e-3).any()
        detector.inserted = True


@pytest.mark.parametrize('width,height', [('strip_active_width_mm', 'strip_active_height_mm'),
                                         ('alignment_active_width_mm', 'alignment_active_height_mm')])
def test_eels_cannot_silently_become_square(eels, width, height):
    from temsim.energy_filter_model_3d import _zebra_meshes

    part, by_key = eels
    part[height] = part[width]
    with pytest.raises(ValueError, match='distinct width and height'):
        _zebra_meshes(part)
    with pytest.raises(ValueError, match='distinct width and height'):
        detector_solids(part, by_key)
