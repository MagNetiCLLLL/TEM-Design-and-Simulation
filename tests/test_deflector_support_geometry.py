"""Equivalent winding supports keep real clearances and existing field owners."""

from collections import Counter
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest

from temsim.cad_catalog import part_solids
from temsim.cad_geometry import annulus
from temsim.cad_mesh import _manifold
from temsim.cad_multipoles import deflector_insert_solids
from temsim.corrector_model_3d import deflector_coil_meshes
from temsim.deflector_support_geometry import (
    END_COVER_FIELD,
    END_REGIONS,
    END_SIDE_FIELD,
    GAP_FILL_FIELD,
    GAP_REGIONS,
    SUPPORT_PROFILE,
    deflector_support_dimensions,
    deflector_support_solids,
    full_material_radial_extent,
)
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


RELATIVE = "column/C3_ProbeCorrector.toml"
COILS = ("probe_dph2_coils", "probe_dp22_coils", "probe_dp21_coils",
         "probe_dph1_coils", "probe_dp11_coils")
COPPER = {"x_winding_1", "x_winding_2", "y_winding_1", "y_winding_2"}
SUPPORT_RADII = {
    "x_carrier": (11.06, 11.10),
    "x_radial_cover": (11.42, 11.46),
    "interlayer_bond": (11.46, 11.54),
    "y_carrier": (11.54, 11.58),
    "y_radial_cover": (11.90, 11.94),
}
SUPPORT_FIELDS = ("winding_support_carrier_thickness_mm",
                  "winding_support_cover_thickness_mm")


@pytest.fixture
def document():
    return read_document(INSTRUMENT_CONFIG_ROOT / RELATIVE)


@pytest.fixture
def draft(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs

    root = tmp_path / "instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (RELATIVE,))
    return PartModelDocument(root / RELATIVE)


def _rows(document):
    return {row["key"]: row for row in document["parts"]}


def _without_support(row):
    return {key: value for key, value in row.items()
            if not key.startswith("winding_support_")}


def _closed_volume(mesh):
    assert np.isfinite(mesh.vertices).all()
    triangles = mesh.vertices[mesh.faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0],
                     triangles[:, 2] - triangles[:, 0])
    assert np.all(np.linalg.norm(cross, axis=1) > 0)
    edges = Counter(tuple(sorted((int(a), int(b))))
                    for face in mesh.faces for a, b in zip(face, np.roll(face, -1)))
    assert set(edges.values()) == {2}
    shifted = triangles - mesh.vertices.mean(axis=0)
    volume = np.einsum("ij,ij->i", shifted[:, 0],
                       np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6
    assert volume > 0
    return volume


@pytest.mark.parametrize("key", COILS)
def test_actual_inserts_have_matching_native_and_cad_supports(document, key):
    rows = _rows(document)
    row = rows[key]
    assert row["winding_support_profile"] == SUPPORT_PROFILE
    assert deflector_support_dimensions(row) is not None
    specs = {spec.region: spec for spec in part_solids(row, rows)}
    meshes = {mesh.region: mesh for mesh in part_model_from_document(document, key).meshes}
    assert row[GAP_FILL_FIELD] is True
    side = row.get(END_SIDE_FIELD, "both")
    end_regions = (set(END_REGIONS if side == "both" else
                       (END_REGIONS[0 if side == "upstream" else 1],))
                   if END_COVER_FIELD in row else set())
    assert set(specs) == set(meshes) == COPPER | SUPPORT_RADII.keys() | set(GAP_REGIONS) | end_regions
    for region in (*SUPPORT_RADII, *GAP_REGIONS):
        spec, mesh = specs[region], meshes[region]
        assert mesh.material_class == spec.material
        assert spec.material != "copper_winding"
        assert not mesh.is_exact
        assert spec.evidence
        assert "OEM" in spec.note or "manufacturer" in spec.note
        centre = (row["local_start_z_mm"] + row["local_end_z_mm"]) / 2
        half = row.get("winding_axial_length_mm", row["length_mm"]) / 2
        assert np.min(mesh.vertices[:, 2]) == pytest.approx(centre - half)
        assert np.max(mesh.vertices[:, 2]) == pytest.approx(centre + half)


def test_supports_are_opt_in_and_do_not_rebuild_or_relabel_copper(document):
    row = _rows(document)[COILS[0]]
    original = _without_support(row)
    assert deflector_support_dimensions(original) is None
    assert deflector_support_solids(original) == ()
    assert full_material_radial_extent(original) == pytest.approx((11.1, 11.9))
    assert full_material_radial_extent(row) == pytest.approx((11.06, 11.94))
    before_meshes, _ = deflector_coil_meshes(original)
    after_meshes, _ = deflector_coil_meshes(row)
    after_meshes = {mesh.region: mesh for mesh in after_meshes}
    assert {mesh.region for mesh in before_meshes} == COPPER
    for before in before_meshes:
        after = after_meshes[before.region]
        np.testing.assert_array_equal(after.vertices, before.vertices)
        np.testing.assert_array_equal(after.faces, before.faces)
        assert after.material_class == before.material_class
        assert after.color == before.color
    before_specs = {spec.region: spec for spec in deflector_insert_solids(original)}
    after_specs = {spec.region: spec for spec in deflector_insert_solids(row)}
    for region in COPPER:
        assert after_specs[region].shape == before_specs[region].shape
        assert after_specs[region].material == before_specs[region].material


def test_supports_are_closed_single_volumes_and_have_no_material_penetration(document):
    row = _rows(document)[COILS[0]]
    specs = deflector_insert_solids(row)
    meshes, _ = deflector_coil_meshes(row, angular_segments=72)
    for mesh in meshes:
        _closed_volume(mesh)
    # A circular support touching a saddle can intersect its polygonal
    # approximation at the contact surface. Test interior volumes inset by
    # 3 um, greater than the 180-segment sagitta but far below the 40 um film.
    interiors = []
    for spec in specs:
        # Filled conductor layers have coincident curved copper interfaces.
        # Independent polygonal Boolean rebuilds make contact slivers there;
        # their analytic occupancy and actual display meshes are checked below.
        if spec.region in GAP_REGIONS:
            continue
        shape = spec.shape
        if spec.region in SUPPORT_RADII:
            section = np.asarray(shape["section_zr_mm"])
            low, high = section[:, 1].min(), section[:, 1].max()
            assert (low, high) == pytest.approx(SUPPORT_RADII[spec.region])
            solid = _manifold(shape, 180)
            assert len(solid.decompose()) == 1
            assert solid.volume() > 0
            shape = annulus(section[:, 0].min(), section[:, 0].max(),
                            low + .003, high - .003)
        interiors.append((spec.region, _manifold(shape, 180)))
    for (left_name, left), (right_name, right) in combinations(interiors, 2):
        assert (left ^ right).volume() < 1e-9, (left_name, right_name)


def test_support_faces_keep_selectable_dimension_provenance(document):
    key = COILS[0]
    model = part_model_from_document(document, key)
    for mesh in model.meshes:
        assert len(mesh.face_groups) == len(mesh.faces)
        assert set(mesh.face_groups) == set(mesh.surfaces)
        for surface in mesh.surfaces.values():
            # Curved cylindrical groups deliberately have no single normal.
            if "normal" in surface:
                assert np.linalg.norm(surface["normal"]) == pytest.approx(1.)
        if mesh.region in SUPPORT_RADII:
            paths = {tuple(path) for surface in mesh.surfaces.values()
                     for path in surface["parameter_paths"]}
            expected = (SUPPORT_FIELDS if mesh.region == "interlayer_bond" else
                        (SUPPORT_FIELDS[0],) if mesh.region.endswith("carrier") else
                        (SUPPORT_FIELDS[1],))
            assert {("parts", key, name) for name in expected} <= paths
            assert all(path[1] == key for path in paths)


def test_current_supports_clear_actual_tube_and_hp_material(document):
    rows = _rows(document)
    tube_radius = document["geometry"]["c1_c2_objective_vacuum_tube_outer_diameter_mm"] / 2
    assert tube_radius == pytest.approx(9.6)
    assert document["geometry"]["c1_c2_objective_vacuum_tube_inner_diameter_mm"] == pytest.approx(5.76)
    for key in COILS:
        row = rows[key]
        inner, outer = full_material_radial_extent(row)
        assert inner - tube_radius == pytest.approx(1.46)
        assert inner >= row["mechanical_clear_bore_diameter_mm"] / 2
        hp_key = row.get("optical_reference_host_key")
        if hp_key:
            hp = rows[hp_key]
            assert hp["mechanical_inner_diameter_mm"] / 2 - outer == pytest.approx(.06)
            assert row["local_start_z_mm"] >= hp["local_start_z_mm"]
            assert row["local_end_z_mm"] <= hp["local_end_z_mm"]


def test_support_edits_roundtrip_undo_and_keep_finite_field_owner(draft, tmp_path):
    key = COILS[0]
    channel_key = "probe_dph2_deflector"
    original_channel = deepcopy(draft.part(channel_key))
    original_hp = deepcopy(draft.part("probe_hp2_hexapole"))
    fields = {field.path[-1]: field for field in part_dimension_specs(draft.document, key)}
    manifest = ManifestEditor(draft.path.parents[1])
    editable = {field.path[-1]: field for field in manifest.fields(ManifestTarget(RELATIVE, key))}
    for name in SUPPORT_FIELDS:
        assert fields[name].editable and fields[name].unit == "mm"
        assert fields[name].meaning.category == "physical"
        assert fields[name].meaning.source_kind == "estimated"
        assert editable[name].editable
    name = SUPPORT_FIELDS[1]
    path = ("parts", key, name)
    draft.set_dimension(path, .05)
    assert full_material_radial_extent(draft.part(key)) == pytest.approx((11.06, 11.95))
    assert draft.part(channel_key) == original_channel
    draft.undo()
    assert draft.part(key)[name] == .04
    draft.redo()
    target = tmp_path / "supported-coils.toml"
    draft.save_copy(target)
    loaded = PartModelDocument(target)
    assert loaded.part(key)[name] == .05
    assert loaded.part(channel_key) == original_channel
    assert loaded.part("probe_hp2_hexapole") == original_hp
    assert loaded.part(channel_key)["physical_host_key"] == key
    assert loaded.part(channel_key)["effective_thickness_mm"] == 4.


INVALID_CHANGES = [
    {"winding_support_profile": "unverified_product"},
    {"winding_support_profile": None},
    {SUPPORT_FIELDS[0]: None},
    {SUPPORT_FIELDS[0]: False},
    {SUPPORT_FIELDS[0]: 0.},
    {SUPPORT_FIELDS[0]: -0.1},
    {SUPPORT_FIELDS[0]: float("nan")},
    {SUPPORT_FIELDS[0]: 1.2},  # Enters the declared clear bore.
    {SUPPORT_FIELDS[1]: True},
    {SUPPORT_FIELDS[1]: float("inf")},
    {SUPPORT_FIELDS[1]: .2},  # Consumes the positive interlayer bond.
    {GAP_FILL_FIELD: 1},
    {GAP_FILL_FIELD: "true"},
]


@pytest.mark.parametrize("change", INVALID_CHANGES)
def test_invalid_supports_rejected_at_all_direct_builders(document, change):
    row = deepcopy(_rows(document)[COILS[0]])
    row.update(change)
    row = {key: value for key, value in row.items() if value is not None}
    for build in (deflector_support_dimensions, deflector_support_solids,
                  full_material_radial_extent, deflector_coil_meshes,
                  deflector_insert_solids):
        with pytest.raises(ValueError):
            build(row)


def test_invalid_support_is_atomic_in_draft_manifest_and_save(draft):
    key = COILS[0]
    name = SUPPORT_FIELDS[1]
    before = deepcopy(draft.document)
    before_bytes = draft.path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", key, name), .2)
    assert draft.document == before and not draft.dirty
    candidate = deepcopy(before)
    _rows(candidate)[key][name] = .2
    with pytest.raises(ValueError):
        validate_document(candidate)
    editor = ManifestEditor(draft.path.parents[1])
    with pytest.raises(ValueError):
        editor.save(ManifestTarget(RELATIVE, key), {("parts", key, name): .2}, None)
    assert draft.path.read_bytes() == before_bytes


def test_invalid_support_cannot_be_loaded_into_editor(draft, tmp_path):
    import tomli_w
    document = deepcopy(draft.document)
    _rows(document)[COILS[0]][SUPPORT_FIELDS[0]] = -0.1
    path = tmp_path / "invalid-support.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        PartModelDocument(path)
    assert path.read_bytes() == before


def test_support_growth_cannot_use_parent_overlap_to_enter_hp_material(draft):
    key = COILS[0]
    before = deepcopy(draft.document)
    # Positive layer and bond thickness alone permit this value, but it
    # enlarges the full insert OD to 24.02 mm inside a 24 mm HP bore.
    candidate = deepcopy(before)
    _rows(candidate)[key][SUPPORT_FIELDS[1]] = .11
    assert deflector_support_dimensions(_rows(candidate)[key])["bond_thickness_mm"] > 0
    with pytest.raises(ValueError, match="material bore"):
        validate_document(candidate)
    with pytest.raises(ValueError, match="material bore"):
        draft.set_dimension(("parts", key, SUPPORT_FIELDS[1]), .11)
    assert draft.document == before and not draft.dirty


def test_full_support_envelope_is_used_for_adjacent_material_overlap(document):
    from temsim.module_manifest import _validate_column_mechanical_overlaps
    row = _rows(document)[COILS[0]]
    obstacle = dict(key="annular_obstacle", mechanical_profile="magnetic_lens_housing",
                    mechanical_only=True, mechanical_inner_diameter_mm=23.82,
                    mechanical_outer_diameter_mm=24., length_mm=row["length_mm"],
                    local_start_z_mm=row["local_start_z_mm"], local_end_z_mm=row["local_end_z_mm"])
    _validate_column_mechanical_overlaps([_without_support(row), obstacle])
    with pytest.raises(ValueError, match="overlap"):
        _validate_column_mechanical_overlaps([row, obstacle])


@pytest.mark.parametrize("key", COILS)
def test_filled_laminate_uses_only_saved_copper_layer_voids(document, key):
    from test_cad_multipoles import _contains

    row = _rows(document)[key]
    saved = deepcopy(row)
    full = {spec.region: spec for spec in deflector_insert_solids(row)}
    previous = {spec.region: spec for spec in deflector_insert_solids({**row, GAP_FILL_FIELD: False})}
    assert row == saved
    assert set(full) - set(previous) == set(GAP_REGIONS)
    assert all(full[name] == spec for name, spec in previous.items())
    for axis, region, radii in (("x", GAP_REGIONS[0], (11.10, 11.42)),
                                ("y", GAP_REGIONS[1], (11.58, 11.90))):
        spec = full[region]
        assert spec.status == "engineering_reconstruction_family_inference"
        assert spec.parameters["new_dimensions_introduced"] is False
        assert spec.evidence and "not verified" in spec.note
        copper = [full[f"{axis}_winding_{index}"] for index in (1, 2)]
        assert spec.shape["tools"] == [item.shape for item in copper]
        # Independent cylindrical samples check complete partition of each
        # copper layer, including winding windows, bands and angular gaps.
        rng = np.random.default_rng(512)
        theta = rng.uniform(-np.pi, np.pi, 20000)
        radius = rng.uniform(radii[0]+1e-5, radii[1]-1e-5, len(theta))
        z = rng.uniform(row["local_start_z_mm"]+1e-5, row["local_end_z_mm"]-1e-5, len(theta))
        points = np.column_stack((radius*np.cos(theta), radius*np.sin(theta), z))
        occupancy = sum(_contains(item.shape, points).astype(int) for item in (*copper, spec))
        centre = (row["local_start_z_mm"] + row["local_end_z_mm"]) / 2
        half = row.get("winding_axial_length_mm", row["length_mm"]) / 2
        # Sample the entire original allocation: its newly cleared ends must
        # stay empty, while copper and gap fill partition the actual laminate.
        expected = ((z > centre - half) & (z < centre + half)).astype(int)
        np.testing.assert_array_equal(occupancy, expected)
        assert _contains(spec.shape, points).any()


@pytest.mark.parametrize("segments", (16, 32, 72))
def test_gap_display_contacts_copper_without_rebuilding_or_intersecting_it(document, segments):
    import manifold3d as m

    row = _rows(document)[COILS[0]]
    baseline, _ = deflector_coil_meshes({**row, GAP_FILL_FIELD: False}, angular_segments=segments)
    added, _ = deflector_coil_meshes(row, angular_segments=segments)
    by_region = {mesh.region: mesh for mesh in added}
    for before in baseline:
        after = by_region[before.region]
        np.testing.assert_array_equal(after.vertices, before.vertices)
        np.testing.assert_array_equal(after.faces, before.faces)
    for region in GAP_REGIONS:
        mesh = by_region[region]
        _closed_volume(mesh)
        gap = m.Manifold(m.Mesh64(mesh.vertices.copy(), mesh.faces.astype(np.uint64)))
        for name in COPPER:
            copper = by_region[name]
            solid = m.Manifold(m.Mesh64(copper.vertices.copy(), copper.faces.astype(np.uint64)))
            assert (gap ^ solid).volume() < 1e-9
        radius = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        assert radius.min() > row["mechanical_clear_bore_diameter_mm"]/2
        assert radius.max() <= row["mechanical_outer_diameter_mm"]/2 + 1e-10
        assert mesh.vertices[:, 2].min() >= row["local_start_z_mm"] - 1e-10
        assert mesh.vertices[:, 2].max() <= row["local_end_z_mm"] + 1e-10
