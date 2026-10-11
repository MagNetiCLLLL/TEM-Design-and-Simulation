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


def test_eds_heads_retain_old_envelope_and_real_angular_definition(column):
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
