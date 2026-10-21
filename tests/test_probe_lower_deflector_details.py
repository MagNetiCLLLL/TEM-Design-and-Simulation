"""Downstream probe laminates: sealed copper and real assembly clearances."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.cad_catalog import part_solids, posed_solids
from temsim.cad_mesh import _manifold
from temsim.cad_vacuum_tube import CONTINUOUS_TUBE_KEY, tube_solids
from temsim.corrector_model_3d import deflector_coil_meshes, winding_axial_extent
from temsim.deflector_support_geometry import (
    END_COVER_FIELD, END_METADATA_FIELDS, END_REGIONS, END_SIDE_FIELD,
    deflector_support_dimensions, full_material_axial_extent,
)
from temsim.manifest_editor import ManifestEditor, ManifestTarget
from temsim.module_manifest import read_document, validate_document
from temsim.part_model_3d import part_dimension_specs, part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT


MODULE = "column/C3_ProbeCorrector.toml"
SIDES = dict.fromkeys(("probe_dp22_coils", "probe_dp21_coils",
                       "probe_dph1_coils", "probe_dp11_coils"), "both")
VARIANTS = ("C3_ProbeCorrector", "C3_ProbeCorrector_ImageCorrector", "C3_ProbeCorrector_BCOR")
COPPER_LENGTHS = {"probe_dp22_coils": 14.22, "probe_dp21_coils": 14.32, "probe_dp11_coils": 14.32}
CONTACTS = (("probe_dp22_coils", "probe_tl22_lens_lower_pole", 729.1, -1),
            ("probe_dp21_coils", "probe_tl21_lens_upper_pole", 787.2, 1),
            ("probe_dp11_coils", "probe_tl12_lens_upper_pole", 903.6, 1))
END_FIELDS = (END_COVER_FIELD, END_SIDE_FIELD, *END_METADATA_FIELDS)


def _rows(document):
    return {part["key"]: part for part in document["parts"]}


@pytest.fixture
def document():
    return read_document(INSTRUMENT_CONFIG_ROOT / MODULE)


@pytest.fixture
def draft(tmp_path):
    from temsim.shared_tip import copy_catalog_inputs
    root = tmp_path / "instruments"
    copy_catalog_inputs(INSTRUMENT_CONFIG_ROOT, root, (MODULE,))
    return PartModelDocument(root / MODULE)


def _ends(side):
    return set(END_REGIONS if side == "both" else
               (END_REGIONS[0 if side == "upstream" else 1],))


@pytest.mark.parametrize("key", SIDES)
@pytest.mark.parametrize("variant", VARIANTS)
def test_probe_variants_keep_the_same_explicit_laminate_and_double_end_covers(document, key, variant):
    other = read_document(INSTRUMENT_CONFIG_ROOT / "column" / (variant + ".toml"))
    validate_document(other)
    rows = _rows(other)
    row = rows[key]
    assert row[END_SIDE_FIELD] == SIDES[key]
    assert row[END_COVER_FIELD] == .04
    assert row["winding_support_fill_conductor_gaps"] is True
    if key in COPPER_LENGTHS:
        assert row["winding_axial_length_mm"] == COPPER_LENGTHS[key]
        assert row["length_mm"] - row["winding_axial_length_mm"] == pytest.approx(.28)
    expected = part_solids(_rows(document)[key], _rows(document))
    actual = part_solids(row, rows)
    assert actual == expected
    assert len(actual) == 11 + len(_ends(SIDES[key]))
    assert {s.region for s in actual} & set(END_REGIONS) == _ends(SIDES[key])


@pytest.mark.parametrize("key", SIDES)
def test_adding_end_covers_preserves_explicitly_allocated_copper_cad_and_native_meshes(document, key):
    rows = _rows(document)
    row = rows[key]
    legacy = {k: v for k, v in row.items() if not k.startswith("winding_support_")}
    before = {s.region: s for s in part_solids(legacy, rows)}
    after = {s.region: s for s in part_solids(row, rows)}
    for region, copper in before.items():
        assert after[region] == copper
    old, _ = deflector_coil_meshes(legacy)
    new, _ = deflector_coil_meshes(row)
    meshes = {m.region: m for m in new}
    for copper in old:
        np.testing.assert_array_equal(meshes[copper.region].vertices, copper.vertices)
        np.testing.assert_array_equal(meshes[copper.region].faces, copper.faces)
    start, end = winding_axial_extent(row)
    side = SIDES[key]
    expected = (start - (.04 if side in ("upstream", "both") else 0.),
                end + (.04 if side in ("downstream", "both") else 0.))
    assert full_material_axial_extent(row) == pytest.approx(expected, abs=1e-10)
    for region in _ends(side):
        mesh = meshes[region]
        assert np.isfinite(mesh.vertices).all()
        edges = np.sort(mesh.faces[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert np.all(counts == 2)
        paths = {tuple(p) for data in mesh.surfaces.values() for p in data["parameter_paths"]}
        assert ("parts", key, END_COVER_FIELD) in paths


@pytest.mark.parametrize("key,pole,z,side", CONTACTS)
def test_three_former_contacts_have_complete_end_insulation_and_positive_pole_gap(document, key, pole, z, side):
    rows = _rows(document)
    pole_spec, = posed_solids(rows[pole], rows)
    pole_solid = _manifold(pole_spec.shape, 128)
    specs = {spec.region: spec for spec in posed_solids(rows[key], rows)}
    end_region = END_REGIONS[0] if side == -1 else END_REGIONS[1]
    cap = _manifold(specs[end_region].shape, 128)
    assert specs[end_region].material == "insulator"
    assert pole_solid.min_gap(cap, .3) == pytest.approx(.1, abs=1e-9)
    areas = []
    for region, spec in specs.items():
        body = _manifold(spec.shape, 128)
        assert (pole_solid ^ body).volume() < 1e-8
        if spec.material == "copper_winding":
            assert pole_solid.min_gap(body, .3) == pytest.approx(.14, abs=1e-9)
            assert (cap ^ body).volume() < 1e-8
            assert cap.min_gap(body, .1) == pytest.approx(0., abs=1e-9)
            # The cap covers the whole actual copper end section, without
            # crossing either copper or the retained iron pole boundary.
            copper_section = body.slice(z - side * (.14 + 1e-7))
            cap_section = cap.slice(z - side * .12)
            assert (copper_section - cap_section).area() < 1e-9
            areas.append(copper_section.area())
    assert len(areas) == 4 and all(area > 7. for area in areas)


@pytest.mark.parametrize("key", SIDES)
def test_supported_lower_inserts_clear_actual_tube_and_dph1_host(document, key):
    rows = _rows(document)
    row = rows[key]
    d = deflector_support_dimensions(row)
    geometry = document["geometry"]
    tube_spec, = tube_solids({**geometry, "key": CONTINUOUS_TUBE_KEY,
        "mechanical_inner_diameter_mm": geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"]})
    tube = _manifold(tube_spec.shape, 128)
    solids = [_manifold(s.shape, 128) for s in posed_solids(row, rows)]
    assert min(s.min_gap(tube, 2.) for s in solids) == pytest.approx(1.46, abs=.002)
    assert max((s ^ tube).volume() for s in solids) < 1e-8
    if key == "probe_dph1_coils":
        host = [_manifold(s.shape, 128) for s in posed_solids(rows["probe_hp1_hexapole"], rows)]
        assert min(a.min_gap(b, .1) for a in solids for b in host) == pytest.approx(.06, abs=.002)
        assert max((a ^ b).volume() for a in solids for b in host) < 1e-8
        assert d["material_start_mm"] - rows["probe_hp1_hexapole"]["local_start_z_mm"] == pytest.approx(1.76)
        assert rows["probe_hp1_hexapole"]["local_end_z_mm"] - d["material_end_mm"] == pytest.approx(1.76)


@pytest.mark.parametrize("side", (True, 0, None, "", "Upstream", "both ", ["upstream"]))
def test_invalid_side_is_rejected_before_geometry(document, side):
    row = _rows(document)["probe_dp22_coils"]
    row[END_SIDE_FIELD] = side
    with pytest.raises(ValueError, match="end-cover side"):
        deflector_support_dimensions(row)
    with pytest.raises(ValueError, match="end-cover side"):
        validate_document(document)


def test_side_alone_does_not_silently_enable_an_unspecified_cover(document):
    row = _rows(document)["probe_dp22_coils"]
    for field in END_FIELDS:
        row.pop(field, None)
    row[END_SIDE_FIELD] = "downstream"
    with pytest.raises(ValueError, match="finite number"):
        deflector_support_dimensions(row)


@pytest.mark.parametrize("side", ("both", "upstream", "downstream"))
def test_unrepresentably_thin_cover_does_not_silently_disappear(document, side):
    row = _rows(document)["probe_dp22_coils"]
    row[END_SIDE_FIELD] = side
    row[END_COVER_FIELD] = 1e-16
    with pytest.raises(ValueError, match="too small at its allocated position"):
        deflector_support_dimensions(row)


@pytest.mark.parametrize("key,pole,z,side", CONTACTS)
def test_restoring_full_copper_length_cannot_ignore_the_sealed_end_overlap(document, key, pole, z, side):
    from temsim.module_manifest import _validate_column_mechanical_overlaps
    rows = _rows(document)
    row = deepcopy(rows[key])
    row["winding_axial_length_mm"] = row["length_mm"]
    with pytest.raises(ValueError, match="Undeclared mechanical overlap"):
        _validate_column_mechanical_overlaps([row, rows[pole]])


@pytest.mark.parametrize("key", SIDES)
def test_material_edits_change_selected_covers_and_persist_without_changing_optical_channel(draft, key, tmp_path):
    channel = key.replace("_coils", "_deflector")
    before_channel = deepcopy(draft.part(channel))
    before_extent = full_material_axial_extent(draft.part(key))
    dimensions = {s.path[-1]: s for s in part_dimension_specs(draft.document, key)}
    assert dimensions[END_COVER_FIELD].editable
    draft.set_dimension(("parts", key, END_COVER_FIELD), .05)
    assert draft.part(channel) == before_channel
    actual = full_material_axial_extent(draft.part(key))
    assert actual[0] == pytest.approx(before_extent[0] - (.01 if SIDES[key] != "downstream" else 0.))
    assert actual[1] == pytest.approx(before_extent[1] + (.01 if SIDES[key] != "upstream" else 0.))
    before_save = deepcopy(draft.document)
    draft.undo()
    assert full_material_axial_extent(draft.part(key)) == pytest.approx(before_extent)
    draft.redo()
    assert draft.document == before_save
    destination = tmp_path / (key + ".toml")
    draft.save_copy(destination)
    saved = PartModelDocument(destination)
    assert full_material_axial_extent(saved.part(key)) == pytest.approx(actual)
    assert saved.part(channel) == before_channel


@pytest.mark.parametrize("key", SIDES)
def test_detached_copies_keep_cover_side_and_follow_their_new_position(draft, key):
    source = deepcopy(draft.part(key))
    copied = draft.copy_component_from(draft.document, key, "detached_lower_insert", 2400., include_children=False)
    draft.place_component(copied, 2500.)
    row = draft.part(copied)
    assert row[END_SIDE_FIELD] == source[END_SIDE_FIELD]
    delta = 2500. - source["local_center_z_mm"]
    np.testing.assert_allclose(full_material_axial_extent(row),
                               np.asarray(full_material_axial_extent(source)) + delta, atol=1e-10)
    model = part_model_from_document(draft.document, copied, include_children=False)
    assert {m.region for m in model.meshes} & set(END_REGIONS) == _ends(SIDES[key])


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("key", COPPER_LENGTHS)
def test_copper_length_edits_match_both_editors_and_change_every_material_region_only(tmp_path, variant, key):
    from temsim.shared_tip import copy_catalog_tree

    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    module = "column/" + variant + ".toml"
    draft = PartModelDocument(root / module)
    original = deepcopy(draft.document)
    storage_original = read_document(root / module)
    path = ("parts", key, "winding_axial_length_mm")
    dimensions = {spec.path[-1]: spec for spec in part_dimension_specs(original, key)}
    assert dimensions[path[-1]].editable
    assert dimensions[path[-1]].meaning.category == "physical"
    assert not dimensions["length_mm"].editable
    assert dimensions["length_mm"].meaning.category == "envelope"
    before = {spec.region: spec for spec in posed_solids(draft.part(key), _rows(original))}
    old_meshes, _ = deflector_coil_meshes(draft.part(key))
    value = draft.part(key)[path[-1]] - .02
    draft.set_dimension(path, value)
    expected = deepcopy(original)
    _rows(expected)[key][path[-1]] = value
    # Whole-document equality checks fixed optical centres, effective lengths,
    # controls, all other parts, and the independently saved installation span.
    assert draft.document == expected
    after = {spec.region: spec for spec in posed_solids(draft.part(key), _rows(draft.document))}
    assert before.keys() == after.keys() and len(after) == 13
    assert all(before[region].shape != after[region].shape for region in before)
    new_meshes, _ = deflector_coil_meshes(draft.part(key))
    assert all(not np.array_equal(a.vertices, b.vertices) for a, b in zip(old_meshes, new_meshes))
    draft.undo()
    assert draft.document == original
    draft.redo()
    assert draft.document == expected
    copy_path = tmp_path / (variant + "-edited.toml")
    draft.save_copy(copy_path)
    assert PartModelDocument(copy_path).document == {k: v for k, v in expected.items() if k != "subassemblies"}
    editor = ManifestEditor(root)
    editor.save(ManifestTarget(module, key), {path: value}, None)
    storage_expected = deepcopy(storage_original)
    _rows(storage_expected)[key][path[-1]] = value
    assert read_document(root / module) == storage_expected
    # Both editor sessions resolve shared Descan ownership metadata. Compare
    # like-for-like resolved views as well as the unchanged raw storage fields.
    assert PartModelDocument(root / module).document == expected


@pytest.mark.parametrize("key", COPPER_LENGTHS)
@pytest.mark.parametrize("field", ("winding_axial_length_mm", END_COVER_FIELD))
def test_buildable_overlaps_remain_editable_drafts_but_both_save_paths_reject_atomically(tmp_path, key, field):
    from temsim.shared_tip import copy_catalog_tree

    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    draft = PartModelDocument(root / MODULE)
    before = deepcopy(draft.document)
    value = draft.part(key)["length_mm"] if field == "winding_axial_length_mm" else .15
    path = ("parts", key, field)
    texts = {p: p.read_bytes() for p in root.parent.rglob("*.toml")}
    # The established editor intentionally permits a buildable conflicting
    # draft so the user can inspect overlap pairs and adjust neighbouring parts.
    draft.set_dimension(path, value)
    invalid, history = deepcopy(draft.document), deepcopy(draft._history)
    assert draft.dirty and draft.can_undo
    rows = _rows(invalid)
    _, pole, _, side = next(contact for contact in CONTACTS if contact[0] == key)
    cap_region = END_REGIONS[0] if side == -1 else END_REGIONS[1]
    pole_body = _manifold(posed_solids(rows[pole], rows)[0].shape, 128)
    cap = _manifold(next(s.shape for s in posed_solids(rows[key], rows) if s.region == cap_region), 128)
    assert (pole_body ^ cap).volume() > .1
    for operation in (draft.validate, draft.save, lambda: draft.save_copy(tmp_path / "invalid-copy.toml")):
        with pytest.raises(ValueError, match="Undeclared mechanical overlap") as error:
            operation()
        # The preliminary manifest check names the containing lens allocation;
        # the exact material report identifies its particular pole region.
        assert key in str(error.value) and rows[pole]["parent_key"] in str(error.value)
        assert draft.document == invalid and draft._history == history
    assert not (tmp_path / "invalid-copy.toml").exists()
    assert all(p.read_bytes() == data for p, data in texts.items())
    draft.undo()
    assert draft.document == before
    draft.redo()
    assert draft.document == invalid
    with pytest.raises(ValueError, match="Undeclared mechanical overlap"):
        ManifestEditor(root).save(ManifestTarget(MODULE, key), {path: value}, None)
    assert all(p.read_bytes() == data for p, data in texts.items())



@pytest.mark.parametrize("side", ("upstream", "downstream", "both"))
def test_selective_cover_api_keeps_explicit_copper_length_and_requested_extent(document, side):
    row = _rows(document)["probe_dp22_coils"]
    row[END_SIDE_FIELD] = side
    start, end = winding_axial_extent(row)
    assert full_material_axial_extent(row) == pytest.approx((
        start - (.04 if side != "downstream" else 0.),
        end + (.04 if side != "upstream" else 0.)))
    assert {s.region for s in part_solids(row, _rows(document))} & set(END_REGIONS) == _ends(side)
    validate_document(document)
