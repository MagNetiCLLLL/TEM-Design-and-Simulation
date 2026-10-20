"""Bounded CAD checks against configured physical surfaces and branch frames."""
from copy import deepcopy
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_auxiliary import accelerator_solids, aperture_solids, filter_solids
from temsim.cad_catalog import part_solids, posed_solids


ROOT = Path(__file__).resolve().parents[1]


def _document(path):
    return tomllib.loads((ROOT/path).read_text(encoding="utf-8"))


def _rigid(shape):
    """Independently compose nested column-vector rigid placements."""
    rotation, translation = np.eye(3), np.zeros(3)
    while shape["kind"] == "transform":
        translation = translation+rotation@np.asarray(shape["offset_mm"])
        rotation = rotation@np.asarray(shape["rotation"])
        shape = shape["shape"]
    return shape, rotation, translation


@pytest.mark.parametrize("filename", ["FEG.toml", "FEG_Mono.toml", "Thermionic.toml"])
def test_accelerator_keeps_each_configured_electrode_face_and_separate_insulation(filename):
    document = _document("configs/instruments/gun/"+filename)
    part = next(p for p in document["parts"] if "stage_centers_z_mm" in p)
    before = deepcopy(part)
    solids = accelerator_solids(part)
    if "electrode_thickness_mm" not in part:
        assert len(solids) == 1
        assert solids[0].status == "configured_envelope"
        assert "not supplied" in solids[0].note
        assert part == before
        return
    electrodes = [s for s in solids if s.material == "electrode"]
    assert len(electrodes) == len(part["stage_centers_z_mm"])
    declared = "electrode_thickness_mm" in part
    thickness = part["electrode_thickness_mm"] if declared else electrodes[0].parameters["electrode_thickness_mm"]
    assert thickness > 0
    intervals = []
    for centre, spec in zip(part["stage_centers_z_mm"], electrodes, strict=True):
        section = np.asarray(spec.shape["section_zr_mm"])
        assert spec.status == ("configured_geometry" if declared else "engineering_reconstruction")
        if not declared:
            assert "reconstruct" in spec.note.lower()
        assert section[:, 0].min() == centre-thickness/2
        assert section[:, 0].max() == centre+thickness/2
        assert section[:, 1].min() == part["bore_diameter_mm"]/2
        assert section[:, 1].max() == part["outer_diameter_mm"]/2
        intervals.append((centre-thickness/2, centre+thickness/2))
    for spec in solids:
        if spec.material != "insulator":
            continue
        section = np.asarray(spec.shape["section_zr_mm"])
        lo, hi = section[:, 0].min(), section[:, 0].max()
        assert part["local_start_z_mm"] <= lo < hi <= part["local_end_z_mm"]
        assert not any(max(lo, a) < min(hi, b) for a, b in intervals)
        assert spec.status == "engineering_reconstruction"
        assert part["stage_centers_z_mm"][0] < lo < hi < part["stage_centers_z_mm"][-1]
    assert part == before


def test_wien_alternative_patent_does_not_invent_this_units_pole_count():
    document = _document("configs/instruments/gun/FEG_Mono.toml")
    part = next(p for p in document["parts"] if p["key"] == "feg_monochromator_wien")
    spec, = part_solids(part, {p["key"]: p for p in document["parts"]})
    assert spec.region == "body" and spec.status == "configured_envelope"
    assert "alternative" in spec.note


@pytest.mark.parametrize("holes,index,diameter,offset", [
    (None, 0, .024, (.2, -.1)), ([.015, .04], 1, .04, (0., 0.)),
])
def test_aperture_working_hole_thickness_and_offsets_match_native_plate(holes, index, diameter, offset):
    from temsim.part_model_apertures import strip_meshes
    document = _document("configs/instruments/column/C3_ProbeCorrector.toml")
    part = next(p for p in document["parts"] if p["key"] == "condenser_aperture_2")
    # Retain the legacy unmeasured-outline contract; the opt-in C2 attachment
    # has separate native/CAD and working-hole coverage in its dedicated suite.
    part = {key: value for key, value in part.items() if not key.startswith("aperture_support_")}
    if holes is not None:
        part["aperture_hole_diameters_mm"] = holes
    runtime = dict(radius_mm=.012, offset_x_mm=.2, offset_y_mm=-.1)
    before = deepcopy(part)
    solid, = aperture_solids(part, runtime, index)
    base = solid.shape["base"]
    tool, rotation, translation = _rigid(solid.shape["tools"][0])
    assert solid.shape["kind"] == "difference"
    assert base["size_mm"][2] == part["plate_thickness_mm"]
    assert base["center_mm"][2] == part["optical_reference_local_z_mm"]
    assert max(p[1] for p in tool["section_zr_mm"]) == diameter/2
    np.testing.assert_array_equal(rotation, np.eye(3))
    np.testing.assert_array_equal(translation[:2], offset)
    meshes, _ = strip_meshes(part, 32, runtime, index)
    vertices = meshes[0].vertices
    np.testing.assert_allclose(vertices.min(axis=0), np.asarray(base["center_mm"])-np.asarray(base["size_mm"])/2, atol=1e-12)
    np.testing.assert_allclose(vertices.max(axis=0), np.asarray(base["center_mm"])+np.asarray(base["size_mm"])/2, atol=1e-12)
    assert part == before


def test_unspecified_aperture_and_zero_thickness_reference_are_not_invented():
    document = _document("configs/instruments/column/C3_ProbeCorrector.toml")
    part = next(p for p in document["parts"] if p["key"] == "condenser_aperture_2")
    part = {key: value for key, value in part.items() if not key.startswith("aperture_support_")}
    spec, = aperture_solids(part)
    assert spec.parameters["working_diameter_mm"] is None
    assert "not specified" in spec.note
    assert spec.shape["kind"] == "box"  # Explicitly schematic plate, no guessed carrier-sized hole.
    reference = dict(key="active_plane", local_start_z_mm=4., local_center_z_mm=4.,
                     local_end_z_mm=4., mechanical_outer_diameter_mm=20.)
    assert part_solids(reference, {reference["key"]: reference}) == ()
    unknown = dict(key="unknown", local_start_z_mm=0., local_center_z_mm=1., local_end_z_mm=2.)
    assert part_solids(unknown, {unknown["key"]: unknown}) == ()


@pytest.fixture
def filter_parts():
    document = _document("configs/subassemblies/energy_filter.toml")
    parts = {p["key"]: p for p in document["parts"]}
    # A nonzero mechanical datum catches omitted and duplicated mounts.
    for part in parts.values():
        for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
            if field in part:
                part[field] += 1800.
    return parts


def test_filter_branch_mount_preserves_entrance_and_horizontal_exit(filter_parts):
    parts = filter_parts
    before = deepcopy(parts)
    aperture = parts["energy_filter_entrance_aperture"]
    spec, = posed_solids(aperture, parts, runtime=dict(radius_mm=.2))
    _, rotation, offset = _rigid(spec.shape)
    np.testing.assert_allclose(rotation@np.array((0., 0., 1800.))+offset, (0., 0., 1800.), atol=1e-12)
    np.testing.assert_allclose(rotation@np.array((0., 0., 1.)), (0., 0., 1.), atol=1e-12)
    # The bias tube is after the 90-degree prism. Independent arc construction
    # gives (R+distance, 0, entrance_Z+entrance_drift+R) in column coordinates.
    part = parts["energy_filter_bias_tube"]
    prism = parts["energy_filter_tapered_prism"]
    assert part["path_reference"] == "prism_exit" and prism["bend_angle_deg"] == 90.
    spec, = posed_solids(part, parts)
    _, rotation, offset = _rigid(spec.shape)
    expected = (prism["prism_radius_mm"]+part["path_center_mm"], 0.,
                1800.+prism["path_entrance_mm"]+prism["prism_radius_mm"])
    np.testing.assert_allclose(offset, expected, atol=1e-12)
    np.testing.assert_allclose(rotation@np.array((0., 0., 1.)), (1., 0., 0.), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    assert parts == before


def test_filter_prism_gap_and_runtime_slit_boundaries_are_unchanged(filter_parts):
    prism = filter_parts["energy_filter_tapered_prism"]
    poles = filter_solids(prism, filter_parts)
    gap = prism["pole_gap_mm"]
    for spec in poles:
        sector, rotation, offset = _rigid(spec.shape)
        assert sector["inner_mm"] == prism["prism_radius_mm"]-prism["radial_clear_half_width_mm"]
        assert sector["outer_mm"] == prism["prism_radius_mm"]+prism["radial_clear_half_width_mm"]
        # Cylinder axial coordinate becomes branch +/-Y. Added material must
        # touch, but never fill, the existing field/vacuum gap.
        ys = [(rotation@np.array((0., 0., z))+offset)[1] for z in (sector["start_mm"], sector["end_mm"])]
        assert min(abs(v) for v in ys) == gap/2
        assert ys[0]*ys[1] > 0
    slit = filter_parts["energy_filter_slit"]
    blades = filter_solids(slit, filter_parts, runtime=dict(gap_m=36e-6, centre_m=5e-6))
    local = [_rigid(spec.shape)[0] for spec in blades]
    edges = sorted((b["center_mm"][0]-b["size_mm"][0]/2,
                    b["center_mm"][0]+b["size_mm"][0]/2) for b in local)
    assert edges[0][1] == pytest.approx(.005-.036/2)
    assert edges[1][0] == pytest.approx(.005+.036/2)
    assert all(b["size_mm"][2] == slit["blade_thickness_mm"] for b in local)
    assert all(b["size_mm"][1] == slit["clear_height_mm"] for b in local)


def test_electrostatic_filter_elements_are_electrodes_not_magnetic_coils(filter_parts):
    for key in ("energy_filter_dynamic_focus_electrostatic_quadrupole", "energy_filter_shutter", "energy_filter_camera_deflector"):
        solids = filter_solids(filter_parts[key], filter_parts)
        assert solids and all(s.material in {"electrode", "insulator", "nonmagnetic_housing"} for s in solids)
        assert any(s.material == "electrode" for s in solids)
        assert not any(s.region == "housing" for s in solids)
    assert filter_solids(filter_parts["energy_filter_zebra"], filter_parts) == ()


def test_filter_optical_channel_does_not_assign_physical_poles(filter_parts):
    for part in filter_parts.values():
        if part.get("mechanical_profile") == "energy_filter_multipole_carrier":
            spec, = filter_solids(part, filter_parts)
            assert spec.status == "configured_envelope" and spec.region == "body"
            assert spec.material == "unspecified"


@pytest.mark.parametrize("field,values", [("offset_mm", [1., 0., 0.]), ("rotation_deg", [0., 3., 0.])])
def test_gun_editor_pose_is_preserved_by_native_and_exported_geometry(field, values):
    from temsim.cad_mesh import meshes_from_solids
    from temsim.part_model_3d import part_model_from_document

    document = _document("configs/instruments/gun/FEG.toml")
    part = next(p for p in document["parts"] if p["key"] == "feg_extractor")
    original = meshes_from_solids(posed_solids(part, {part["key"]: part}), 32)
    native_original = part_model_from_document(document, part["key"], angular_segments=32).meshes
    part["model_3d"] = dict(schema_version=1, transform={field: values})
    before = deepcopy(document)
    exported = meshes_from_solids(posed_solids(part, {part["key"]: part}), 32)
    native = part_model_from_document(document, part["key"], angular_segments=32).meshes
    center = np.array((0., 0., part["local_center_z_mm"]))
    angle = np.deg2rad(3.) if field == "rotation_deg" else 0.
    rotation = np.array(((np.cos(angle), 0., np.sin(angle)), (0., 1., 0.),
                         (-np.sin(angle), 0., np.cos(angle))))
    offset = np.asarray(values) if field == "offset_mm" else np.zeros(3)
    for source, transformed_meshes in ((original, exported), (native_original, native)):
        assert len(source) == len(transformed_meshes)
        for baseline, changed in zip(source, transformed_meshes, strict=True):
            expected = (baseline.vertices - center) @ rotation.T + center + offset
            # Tessellation may reorder vertices; compare the complete point set.
            def ordered(points):
                return np.asarray(sorted(map(tuple, np.round(points, 10))))
            np.testing.assert_allclose(ordered(changed.vertices), ordered(expected), rtol=0, atol=1e-9)
    assert document == before


def test_native_cad_affine_preserves_nested_placement_scale_shear_and_volume():
    from itertools import product
    from temsim.cad_geometry import SolidSpec, box, transformed
    from temsim.cad_features import affine
    from temsim.cad_mesh import meshes_from_solids

    center, size = np.array((1., 2., 3.)), np.array((2., 3., 4.))
    inner_rotation = np.array(((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.)))
    inner_offset = np.array((.2, .4, -.6))
    matrix = np.array(((1.5, .2, 0.), (0., .5, .3), (0., 0., 1.2)))
    offset = np.array((.3, -.2, 7.))
    outer_rotation = np.array(((0., -1., 0.), (1., 0., 0.), (0., 0., 1.)))
    outer_offset = np.array((-1., 2., 3.))
    shape = transformed(affine(transformed(box(center, size),
        rotation=inner_rotation.tolist(), offset=inner_offset), matrix, offset),
        rotation=outer_rotation.tolist(), offset=outer_offset)
    before = deepcopy(shape)
    mesh, = meshes_from_solids((SolidSpec("fixture", "body", "electrode", shape),))

    corners = center + np.array(list(product((-1., 1.), repeat=3))) * size/2
    expected = ((corners @ inner_rotation.T + inner_offset) @ matrix.T + offset)
    expected = expected @ outer_rotation.T + outer_offset
    def ordered(points):
        return np.asarray(sorted(map(tuple, np.round(points, 10))))
    np.testing.assert_allclose(ordered(mesh.vertices), ordered(expected), rtol=0., atol=1e-9)
    triangles = mesh.vertices[mesh.faces]
    volume = np.einsum("ij,ij->i", triangles[:, 0],
                       np.cross(triangles[:, 1], triangles[:, 2])).sum()/6
    assert volume == pytest.approx(float(np.prod(size))*1.5*.5*1.2, rel=1e-12)
    assert shape == before


@pytest.mark.parametrize("matrix,offset", [
    (np.ones((2, 3)), (0., 0., 0.)),
    (np.eye(3), (0., 0.)),
    (np.diag((1., np.nan, 1.)), (0., 0., 0.)),
    (np.eye(3), (0., np.inf, 0.)),
    (np.diag((1., 0., 1.)), (0., 0., 0.)),
    (np.diag((-1., 1., 1.)), (0., 0., 0.)),
    (np.diag((1., 1., 1e-16)), (0., 0., 0.)),
])
def test_native_cad_affine_rejects_invalid_or_nonpositive_placements(matrix, offset):
    from temsim.cad_geometry import SolidSpec, box
    from temsim.cad_features import affine
    from temsim.cad_mesh import meshes_from_solids

    shape = affine(box((0., 0., 0.), (1., 1., 1.)), matrix, offset)
    with pytest.raises(ValueError, match="CAD affine"):
        meshes_from_solids((SolidSpec("fixture", "body", "electrode", shape),))
