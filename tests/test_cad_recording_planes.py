"""Recording meshes agree with configured reception footprints and planes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from temsim.cad_mesh import _manifold
from temsim.cad_recording_planes import recording_plane_meshes, recording_plane_solids
from temsim.part_model_3d import part_model_from_document


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def document():
    return tomllib.loads((ROOT/"configs/instruments/project_and_recording_system/NoEnergyFilter.toml")
                        .read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def parts(document):
    return {part["key"]: part for part in document["parts"]}


def _area(mesh, group):
    triangles = mesh.vertices[mesh.faces[mesh.face_groups == group]]
    return .5*np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                     triangles[:, 2]-triangles[:, 0]), axis=1).sum()


def test_camera_square_matches_particle_corner_acceptance_and_retains_front_z(document, parts):
    from temsim.detector.camera import CAMERA_DETECTOR_DEFINITION
    from temsim.cad_catalog import part_solids

    part = parts["camera"]
    before = deepcopy(part)
    native = part_model_from_document(document, "camera", include_children=False)
    mesh, = native.meshes
    spec, = part_solids(part, parts)
    assert spec.shape["kind"] == "box"
    width = part["outer_width_mm"]
    z0, z1 = part["local_start_z_mm"], part["local_end_z_mm"]
    np.testing.assert_allclose(mesh.vertices.min(axis=0), (-width/2, -width/2, z0))
    np.testing.assert_allclose(mesh.vertices.max(axis=0), (width/2, width/2, z1))
    assert _area(mesh, "collection_surface") == pytest.approx(width**2)
    assert spec.parameters["upstream_surface_z_mm"] == part["optical_reference_local_z_mm"]
    detector = CAMERA_DETECTOR_DEFINITION.create_component()
    hit_before = detector.hit_mask(np.array([.45, .51])*width, np.array([.45, 0.])*width)
    np.testing.assert_array_equal(hit_before, [True, False])
    assert np.hypot(.45*width, .45*width) > width/2  # Old circular model lost this accepted corner.
    assert part == before
    assert set(mesh.surfaces) == {"collection_surface", "rear_surface", "rim_negative_x",
                                  "rim_positive_x", "rim_negative_y", "rim_positive_y"}


def test_recording_faces_match_runtime_stops_without_altering_readout_or_retraction(document, parts):
    from temsim.detector.recording_system import default_recording_planes
    from temsim.physics.record_plane import runtime_recording_stops

    state = SimpleNamespace(sample=SimpleNamespace(z_mm=0.),
                            selected_area_aperture=SimpleNamespace(
                                z_mm=parts["selected_area_aperture"]["optical_reference_local_z_mm"]))
    state.recording_planes = default_recording_planes(state)
    for detector in state.recording_planes:
        detector.inserted = True
        if hasattr(detector, "readout_enabled"):
            detector.readout_enabled = False
    before = runtime_recording_stops(state)
    for stop in before:
        model = part_model_from_document(document, stop.key, include_children=False)
        mesh, = model.meshes
        front = mesh.vertices[mesh.faces[mesh.face_groups == "collection_surface"]]
        np.testing.assert_allclose(front[:, :, 2], stop.z_mm)
        spec, = recording_plane_solids(parts[stop.key], parts)
        assert spec.parameters["footprint"] == stop.geometry
        inner, outer = stop.inner_diameter_mm/2, stop.outer_width_mm/2
        position_mm = np.array([0., (inner+outer)/2, outer*1.1])
        np.testing.assert_array_equal(stop.hit_mask(position_mm*1e-3, np.zeros(3)),
                                      [not bool(inner), True, False])
        assert not stop.non_blocking
    assert runtime_recording_stops(state) == before
    assert all(not stop.readout_enabled for stop in before if stop.key in {"haadf", "df", "bf"})
    for detector in state.recording_planes:
        detector.inserted = False
    assert runtime_recording_stops(state) == ()
    # The physical part inventory remains available for inspection even when
    # a detector is retracted from the electron beam.
    for stop in before:
        assert part_model_from_document(document, stop.key, include_children=False).meshes
    assert runtime_recording_stops(state) == ()


@pytest.mark.parametrize("key", ["haadf", "df", "bf", "flu_screen"])
def test_circular_receiver_keeps_configured_annular_or_disk_area_and_open_bore(parts, key):
    part = parts[key]
    before = deepcopy(part)
    (mesh,), notes = recording_plane_meshes(part, 64)
    spec, = recording_plane_solids(part, parts)
    inner, outer = part["inner_diameter_mm"]/2, part["outer_width_mm"]/2
    area = .5*64*np.sin(2*np.pi/64)*(outer**2-inner**2)
    assert _area(mesh, "collection_surface") == pytest.approx(area)
    assert _area(mesh, "rear_surface") == pytest.approx(area)
    front = mesh.vertices[mesh.faces[mesh.face_groups == "collection_surface"]]
    np.testing.assert_allclose(front[:, :, 2], part["optical_reference_local_z_mm"])
    assert ("bore_rim" in mesh.surfaces) == bool(inner)
    np.testing.assert_allclose(mesh.surfaces["collection_surface"]["normal"], (0., 0., -1.))
    np.testing.assert_allclose(mesh.surfaces["rear_surface"]["normal"], (0., 0., 1.))
    assert _manifold(spec.shape, 64).volume() == pytest.approx(area*part["length_mm"])
    assert part == before
    assert "No sensor stack" in notes[0] and not mesh.is_exact


@pytest.mark.parametrize("profile, inner", [("camera_sensor_plane", 0.),
                                           ("retractable_detector_plane", 0.),
                                           ("retractable_detector_plane", 2.)])
def test_reference_without_thickness_is_a_guide_never_a_manufactured_solid(profile, inner):
    part = dict(key="reference", mechanical_profile=profile, local_start_z_mm=10.,
                local_center_z_mm=10., local_end_z_mm=10., length_mm=0.,
                outer_width_mm=4., inner_diameter_mm=inner)
    assert recording_plane_solids(part) == ()
    (mesh,), notes = recording_plane_meshes(part, 24)
    np.testing.assert_array_equal(mesh.vertices[:, 2], 10.)
    assert mesh.wireframe and not mesh.is_exact
    assert set(mesh.surfaces) == {"collection_surface"}
    assert "no finite thickness" in notes[0]


@pytest.mark.parametrize("field,value", [("outer_width_mm", 0.), ("outer_width_mm", True),
                                       ("outer_width_mm", float("nan")), ("inner_diameter_mm", -1.),
                                       ("inner_diameter_mm", 57.344)])
def test_recording_dimensions_are_explicit_and_validated(parts, field, value):
    part = {**parts["camera"], field: value}
    with pytest.raises(ValueError):
        recording_plane_solids(part)


def test_camera_does_not_silently_accept_annular_geometry(parts):
    with pytest.raises(ValueError, match="square camera"):
        recording_plane_solids({**parts["camera"], "inner_diameter_mm": 1.})


@pytest.mark.parametrize('profile', ['camera_sensor_plane', 'retractable_detector_plane'])
@pytest.mark.parametrize('field,value', [('outer_width_mm', 0.), ('outer_width_mm', -2.),
                                       ('outer_width_mm', True), ('outer_width_mm', float('nan')),
                                       ('outer_width_mm', float('inf')), ('inner_diameter_mm', -1.),
                                       ('inner_diameter_mm', True), ('inner_diameter_mm', float('nan')),
                                       ('inner_diameter_mm', 4.), ('inner_diameter_mm', 5.)])
def test_recording_footprint_validator_rejects_invalid_dimensions_before_rendering(profile, field, value):
    from temsim.cad_recording_planes import validate_detector_footprint

    part = dict(key='receiver', mechanical_profile=profile, outer_width_mm=4., inner_diameter_mm=0.)
    part[field] = value
    with pytest.raises(ValueError):
        validate_detector_footprint(part)


@pytest.mark.parametrize('profile,expected', [('camera_sensor_plane', 'square'),
                                            ('retractable_detector_plane', 'disk')])
def test_missing_legacy_inner_diameter_retains_zero_default(profile, expected):
    from temsim.cad_recording_planes import validate_detector_footprint

    part = dict(key='legacy', mechanical_profile=profile, outer_width_mm=4.,
                local_start_z_mm=0., local_end_z_mm=.5, length_mm=.5)
    assert validate_detector_footprint(part) == expected
    spec, = recording_plane_solids(part)
    assert spec.parameters['inner_diameter_mm'] == 0.


@pytest.mark.parametrize('key,field,value', [('camera', 'inner_diameter_mm', 1.),
                                           ('camera', 'outer_width_mm', 0.),
                                           ('haadf', 'inner_diameter_mm', 22.),
                                           ('haadf', 'outer_width_mm', 4.)])
def test_invalid_recording_edits_preserve_draft_history_and_saved_file(tmp_path, key, field, value):
    from temsim.module_manifest import validate_document
    from temsim.part_model_document import PartModelDocument

    source = ROOT/'configs/instruments/project_and_recording_system/NoEnergyFilter.toml'
    source_bytes = source.read_bytes()
    path = tmp_path/'recording.toml'
    draft = PartModelDocument(source)
    draft.save_copy(path)
    before, saved = deepcopy(draft.document), path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(('parts', key, field), value)
    assert draft.document == before and not draft.dirty and not draft.can_undo
    assert path.read_bytes() == saved
    # A malformed externally supplied draft also fails document validation
    # and persistence, rather than waiting for a render-time error.
    draft.part(key)[field] = value
    with pytest.raises(ValueError):
        validate_document(draft.document)
    with pytest.raises(ValueError):
        draft.save()
    assert path.read_bytes() == saved and source.read_bytes() == source_bytes


def test_unrelated_part_is_not_promoted_to_recording_hardware(parts):
    part = parts["post_projector_detector_chamber"]
    assert recording_plane_solids(part) is None
    assert recording_plane_meshes(part, 24) is None


@pytest.mark.parametrize('key,wrong_shape', [('camera', 'disk'), ('flu_screen', 'square'),
                                           ('haadf', 'disk'), ('df', 'square'), ('bf', 'annulus')])
def test_recording_shape_metadata_cannot_disagree_with_receiving_role(parts, key, wrong_shape):
    part = {**parts[key], 'mechanical_footprint_shape': wrong_shape}
    with pytest.raises(ValueError, match='mechanical_footprint_shape'):
        recording_plane_solids(part)
    with pytest.raises(ValueError, match='mechanical_footprint_shape'):
        recording_plane_meshes(part, 24)


def test_all_recording_definitions_persist_same_shape_convention():
    from temsim.cad_recording_planes import validate_detector_footprint
    from temsim.part_model_document import PartModelDocument

    expected = {'camera': 'square', 'haadf': 'annulus', 'df': 'annulus',
                'bf': 'disk', 'flu_screen': 'disk'}
    for name in ('NoEnergyFilter', 'EnergyFilter'):
        path = ROOT/f'configs/instruments/project_and_recording_system/{name}.toml'
        document = PartModelDocument(path).document
        rows = {row['key']: row for row in document['parts']}
        for key, shape in expected.items():
            assert rows[key]['mechanical_footprint_shape'] == shape
            assert validate_detector_footprint(rows[key]) == shape
            spec, = recording_plane_solids(rows[key], rows)
            assert spec.parameters['footprint'] == shape
        if name == 'EnergyFilter':
            eftem = rows['energy_filter_eftem_output_plane']
            assert eftem['mechanical_footprint_shape'] == 'square'
            assert validate_detector_footprint(eftem) == 'square'
            assert eftem['local_start_z_mm'] == eftem['local_end_z_mm']
            assert recording_plane_solids(eftem, rows) is None  # reference, no invented package


def test_camera_face_semantics_and_plane_follow_one_mechanical_transform(document, parts):
    from temsim.lens_pose import mechanical_part_transform_mm

    updated = deepcopy(document)
    part = next(row for row in updated["parts"] if row["key"] == "camera")
    part.update(offset_x_mm=2., offset_y_mm=-1., offset_z_mm=.3,
                rotation_x_mrad=17., rotation_y_mrad=-11., rotation_z_mrad=23.)
    source, = part_model_from_document(document, "camera", include_children=False).meshes
    actual, = part_model_from_document(updated, "camera", include_children=False).meshes
    rotation, translation = mechanical_part_transform_mm(part, parts)
    np.testing.assert_allclose(actual.vertices, source.vertices @ rotation.T+translation, atol=1e-12)
    np.testing.assert_allclose(actual.surfaces["collection_surface"]["normal"],
                               rotation @ np.array([0., 0., -1.]), atol=1e-12)
