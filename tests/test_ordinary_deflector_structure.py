"""Ordinary dual-stage supports preserve winding geometry and field ownership."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import part_solids
from temsim.cad_mesh import _manifold
from temsim.component_representation import shared_deflector_field_owner
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.module_manifest import read_document, validate_document
from temsim.ordinary_deflector_geometry import (
    ordinary_deflector_support_solids,
    support_dimensions,
    validate_ordinary_deflector_part,
)
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


RELATIVE = "column/C3_ProbeCorrector.toml"
HOSTS = ("condenser_deflector", "beam_deflector", "ac_deflector",
         "image_diffraction_deflector")
PREFIX = "deflector_support_"
PROFILE = "rigid_insulating_carriers_pinned"
FIELDS = tuple(PREFIX + field for field in (
    "radial_skin_thickness_mm", "casing_thickness_mm", "pin_diameter_mm"))
COPPER = {f"coil_{station}_{axis}_winding_{side}"
          for station in (1, 2) for axis in "xy" for side in (1, 2)}
SUPPORTS = {f"coil_{station}_{region}" for station in (1, 2) for region in (
    "x_carrier", "y_carrier", "interaxis_spacer", "outer_casing",
    "registration_pin_1", "registration_pin_2")}


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
    return {key: value for key, value in row.items() if not key.startswith(PREFIX)}


def _specs(document, key):
    rows = _rows(document)
    return {spec.region: spec for spec in part_solids(rows[key], rows)}


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


@pytest.mark.parametrize("key", HOSTS)
def test_actual_dual_stage_hosts_have_matching_native_and_cad_regions(document, key):
    row = _rows(document)[key]
    before = deepcopy(row)
    assert row[PREFIX + "profile"] == PROFILE
    assert support_dimensions(row) is not None
    specs = _specs(document, key)
    meshes = {mesh.region: mesh for mesh in part_model_from_document(document, key).meshes}
    assert set(specs) == set(meshes) == COPPER | SUPPORTS
    assert row == before
    assert {region for region, spec in specs.items()
            if spec.material == "copper_winding"} == COPPER
    for region in SUPPORTS:
        spec, mesh = specs[region], meshes[region]
        assert mesh.material_class == spec.material
        assert spec.material == "insulator"
        assert spec.evidence
        assert "OEM" in spec.note or "manufacturer" in spec.note
        assert not mesh.is_exact
        _closed_volume(mesh)
        # Closed, positive topology is checked separately from material evidence.
        solid = _manifold(spec.shape, 72)
        assert len(solid.decompose()) == 1
        assert solid.volume() > 0
        assert mesh.vertices[:, 2].min() >= row["local_start_z_mm"] - 1e-8
        assert mesh.vertices[:, 2].max() <= row["local_end_z_mm"] + 1e-8
        # Polygonal chord sagitta at 32 samples can undercut a circular bore;
        # the support's engineered skin clearance is much larger than this.
        radii = np.linalg.norm(mesh.vertices[:, :2], axis=1)
        assert radii.min() >= row["mechanical_clear_bore_diameter_mm"] / 2 - .05
        assert radii.max() <= row["mechanical_outer_diameter_mm"] / 2 + 1e-8


@pytest.mark.parametrize("key", HOSTS)
def test_support_opt_in_keeps_exact_copper_geometry_and_legacy_shape(document, key):
    legacy = deepcopy(document)
    legacy_row = _rows(legacy)[key]
    for name in tuple(legacy_row):
        if name.startswith(PREFIX):
            del legacy_row[name]
    assert support_dimensions(legacy_row) is None
    validate_ordinary_deflector_part(legacy_row)
    assert ordinary_deflector_support_solids(legacy_row, ()) == ()
    before_specs, after_specs = _specs(legacy, key), _specs(document, key)
    assert set(before_specs) == COPPER
    for region, before in before_specs.items():
        after = after_specs[region]
        assert after.shape == before.shape
        assert after.material == before.material
        assert after.parameters == before.parameters
    before_meshes = {mesh.region: mesh for mesh in part_model_from_document(legacy, key).meshes}
    after_meshes = {mesh.region: mesh for mesh in part_model_from_document(document, key).meshes}
    for region, before in before_meshes.items():
        after = after_meshes[region]
        np.testing.assert_array_equal(after.vertices, before.vertices)
        np.testing.assert_array_equal(after.faces, before.faces)
        assert after.color == before.color


def test_support_faces_keep_existing_editable_source_paths(document):
    key = HOSTS[0]
    model = part_model_from_document(document, key)
    for mesh in model.meshes:
        if mesh.region not in SUPPORTS:
            continue
        assert len(mesh.face_groups) == len(mesh.faces)
        assert set(mesh.face_groups) == set(mesh.surfaces)
        paths = {tuple(path) for surface in mesh.surfaces.values()
                 for path in surface["parameter_paths"]}
        assert {("parts", key, name) for name in FIELDS} <= paths
        assert all(path[1] == key for path in paths)


def test_support_dimension_edit_roundtrip_preserves_all_optical_values_and_corrector(draft, tmp_path):
    key = "image_diffraction_deflector"
    before = deepcopy(draft.document)
    before_specs = _specs(before, key)
    fields = {field.path[-1]: field for field in part_dimension_specs(draft.document, key)}
    editor = ManifestEditor(draft.path.parents[1])
    editable = {field.path[-1]: field for field in editor.fields(ManifestTarget(RELATIVE, key))}
    for name in FIELDS:
        assert fields[name].editable and fields[name].unit == "mm"
        assert fields[name].meaning.category == "physical"
        assert fields[name].meaning.source_kind == "estimated"
        assert editable[name].editable
    name = FIELDS[0]
    original_value = draft.part(key)[name]
    changed_value = original_value * 1.1
    draft.set_dimension(("parts", key, name), changed_value)
    assert _without_support(draft.part(key)) == _without_support(_rows(before)[key])
    for other_key, other_row in _rows(before).items():
        if other_key != key:
            assert draft.part(other_key) == other_row
    after_specs = _specs(draft.document, key)
    assert all(after_specs[region].shape == before_specs[region].shape for region in COPPER)
    assert any(after_specs[region].shape != before_specs[region].shape for region in SUPPORTS)
    draft.undo()
    assert draft.document == before
    draft.redo()
    target = tmp_path / "ordinary-deflector-supported.toml"
    draft.save_copy(target)
    loaded = PartModelDocument(target)
    assert loaded.part(key)[name] == changed_value
    assert _without_support(loaded.part(key)) == _without_support(_rows(before)[key])
    for other_key, other_row in _rows(before).items():
        if other_key != key:
            assert loaded.part(other_key) == other_row


INVALID_CHANGES = [
    {PREFIX + "profile": "unverified_structure"},
    {PREFIX + "profile": None},
    {PREFIX + "geometry_status": "manufacturer_verified"},
    {PREFIX + "source": None},
    {PREFIX + "source_note": " "},
    {PREFIX + "unrecognized_dimension_mm": 1.},
    {FIELDS[0]: None},
    {FIELDS[0]: False},
    {FIELDS[0]: 0.},
    {FIELDS[0]: -0.1},
    {FIELDS[0]: float("nan")},
    {FIELDS[0]: 100.},
    {FIELDS[1]: None},
    {FIELDS[1]: True},
    {FIELDS[1]: float("inf")},
    {FIELDS[1]: 100.},
    {FIELDS[2]: None},
    {FIELDS[2]: False},
    {FIELDS[2]: 0.},
    {FIELDS[2]: -1.},
    {FIELDS[2]: float("nan")},
    {FIELDS[2]: 100.},
    {"interaction_centers_local_z_mm": [350., 350.]},
    {"interaction_centers_local_z_mm": [False, 370.]},
    {"interaction_centers_local_z_mm": [350., float("inf")]},
    {"mechanical_coil_length_mm": 100.},
]


@pytest.mark.parametrize("change", INVALID_CHANGES)
def test_invalid_supports_rejected_by_dimensions_builders_and_manifest(document, change):
    candidate = deepcopy(document)
    row = _rows(candidate)[HOSTS[0]]
    row.update(change)
    for name in tuple(row):
        if row[name] is None:
            del row[name]
    for validate in (support_dimensions, validate_ordinary_deflector_part,
                     lambda part: part_solids(part, _rows(candidate))):
        with pytest.raises(ValueError):
            validate(row)
    with pytest.raises(ValueError):
        validate_document(candidate)


@pytest.mark.parametrize("profile", [False, None, 0])
def test_explicit_invalid_host_profile_is_not_treated_as_absent(document, profile):
    candidate = deepcopy(document)
    row = _rows(candidate)[HOSTS[0]]
    row["mechanical_profile"] = profile
    for validate in (support_dimensions, validate_ordinary_deflector_part,
                     lambda part: part_solids(part, _rows(candidate))):
        with pytest.raises(ValueError, match="material magnetic deflector host"):
            validate(row)
    with pytest.raises(ValueError):
        validate_document(candidate)


@pytest.mark.parametrize("name", FIELDS)
def test_invalid_support_edit_is_atomic_for_draft_and_manifest(draft, name):
    key = HOSTS[0]
    before = deepcopy(draft.document)
    before_bytes = draft.path.read_bytes()
    with pytest.raises(ValueError):
        draft.set_dimension(("parts", key, name), 100.)
    assert draft.document == before and not draft.dirty
    assert not draft.can_undo
    editor = ManifestEditor(draft.path.parents[1])
    with pytest.raises(ValueError):
        editor.save(ManifestTarget(RELATIVE, key), {("parts", key, name): 100.}, None)
    assert draft.path.read_bytes() == before_bytes


@pytest.mark.parametrize("name", FIELDS)
def test_invalid_support_cannot_be_loaded_into_editor(draft, tmp_path, name):
    import tomli_w

    candidate = deepcopy(draft.document)
    _rows(candidate)[HOSTS[0]][name] = -0.1
    path = tmp_path / "invalid-ordinary-support.toml"
    path.write_text(tomli_w.dumps(candidate), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        PartModelDocument(path)
    assert path.read_bytes() == before


def test_descan_remains_a_nonmaterial_channel_on_image_diffraction_hardware(document, draft):
    rows = _rows(document)
    channel = rows["descan_deflector"]
    assert channel["physical_host_key"] == "image_diffraction_deflector"
    assert channel["interaction_centers_local_z_mm"] == rows["image_diffraction_deflector"]["interaction_centers_local_z_mm"]
    assert part_solids(channel, rows) == ()
    assert not part_model_from_document(document, "descan_deflector").meshes
    assert not any(name.startswith(PREFIX) for name in channel)
    before = deepcopy(draft.document)
    for name in (*FIELDS, PREFIX + "profile", PREFIX + "future_dimension_mm"):
        assert shared_deflector_field_owner("descan_deflector", name) == "image_diffraction_deflector"
    for name in FIELDS:
        with pytest.raises(ValueError, match="shared|image_diffraction_deflector"):
            draft.set_dimension(("parts", "descan_deflector", name), .1)
    assert draft.document == before and not draft.dirty


def test_support_builder_requires_the_declared_material_winding_set(document):
    from dataclasses import replace

    key = HOSTS[0]
    row = _rows(document)[key]
    copper = [spec for region, spec in _specs(document, key).items() if region in COPPER]
    assert len(ordinary_deflector_support_solids(row, copper)) == 12
    for invalid in (copper[:-1], [replace(copper[0], material="insulator"), *copper[1:]]):
        with pytest.raises(ValueError, match="eight copper"):
            ordinary_deflector_support_solids(row, invalid)


def test_corrector_flexible_supports_cannot_be_reinterpreted_as_ordinary_carriers(document):
    rows = _rows(document)
    corrector = deepcopy(rows["probe_dph2_coils"])
    assert support_dimensions(corrector) is None
    corrector.update({name: value for name, value in rows[HOSTS[0]].items()
                      if name.startswith(PREFIX)})
    with pytest.raises(ValueError, match="material magnetic deflector host"):
        validate_ordinary_deflector_part(corrector)
