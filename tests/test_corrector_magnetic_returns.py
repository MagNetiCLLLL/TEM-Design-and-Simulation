"""Magnetic fits in the installed corrector CAD, bore layout and round-lens FEM."""
from dataclasses import replace
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.cad_mesh import _manifold
from temsim.cad_multipoles import multipole_solids, validate_auxiliary_return_sleeve
from temsim.cad_round_lenses import round_lens_solids
from temsim.component_representation import belongs_to_material_selection
from temsim.module_manifest import read_document, validate_document
from temsim.multipole_winding import multipole_winding_envelopes
from temsim.optics.column import default_state
from temsim.part_materials import configured_region_colour, material_for_region
from temsim.part_model_3d import part_model_from_document
from temsim.part_model_document import PartModelDocument
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import _part_mask
from temsim.physics.lens_field_provider import lens_geometry_binding, resolve_runtime_lens_field_provider
from temsim.pole_geometry import section_radial_bands


COLUMN = INSTRUMENT_CONFIG_ROOT / "column"
PROBE = {"adapter_lens", "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens"}
IMAGE = {"image_adapter_lens", "image_tl11_lens", "image_tl12_lens", "image_tl21_lens", "image_tl22_lens"}
BCOR = IMAGE | {"image_tl31_lens", "image_tl32_lens"}


@pytest.mark.parametrize("module,lenses,aux_count", [
    ("C2", set(), 0), ("C3", set(), 0), ("C3_ProbeCorrector", PROBE, 6),
    ("C3_ImageCorrector", IMAGE, 1), ("C3_BCOR", BCOR, 1),
    ("C3_ProbeCorrector_ImageCorrector", PROBE | IMAGE, 7),
    ("C3_ProbeCorrector_BCOR", PROBE | BCOR, 7),
])
def test_shipped_return_plates_seat_on_both_poles_and_preserve_open_gap(module, lenses, aux_count):
    doc = validate_document(read_document(COLUMN / f"{module}.toml"))
    rows = {p["key"]: p for p in doc["parts"]}
    for role in ("pole_sleeves", "radial_return_plates"):
        assert {p["parent_key"] for p in rows.values() if p.get("mechanical_part_role") == role} == lenses
    for lens in lenses:
        sleeve, plate, yoke, coil = (rows[lens+"_"+suffix] for suffix in
                                    ("pole_sleeves", "return_plates", "yoke", "excitation_coil"))
        assert sleeve["mechanical_outer_diameter_mm"] == plate["mechanical_inner_diameter_mm"]
        assert plate["mechanical_outer_diameter_mm"] == yoke["mechanical_inner_diameter_mm"]
        assert sleeve["mechanical_outer_diameter_mm"] < coil["mechanical_inner_diameter_mm"]
        for row in (sleeve, plate):
            assert row["axial_vacuum_context_only"] and row["mechanical_only"]
            shapes = round_lens_solids(row, rows)
            assert [(s.region, s.material) for s in shapes] == [("upper", "soft_magnetic"), ("lower", "soft_magnetic")]
        for side, (s0, s1), (p0, p1) in zip(("upper", "lower"), sleeve["material_intervals_mm"],
                                             plate["material_intervals_mm"], strict=True):
            pole = rows[f"{lens}_{side}_pole"]
            solid, = round_lens_solids(pole, rows)
            assert solid.material == "soft_magnetic"
            radius = sleeve["mechanical_inner_diameter_mm"]/2
            assert any(b.z0 <= s0 < s1 <= b.z1 and b.outer0 == radius == b.outer1
                       for b in section_radial_bands(solid.shape["section_zr_mm"]))
            assert s0 <= p0 < p1 <= s1
            assert p1 < coil["local_start_z_mm"] or p0 > coil["local_end_z_mm"]
    sleeves = [p for p in rows.values() if p.get("mechanical_part_role") == "auxiliary_return_sleeve"]
    assert len(sleeves) == aux_count
    for sleeve in sleeves:
        assert sleeve["axial_vacuum_context_only"] and len(round_lens_solids(sleeve, rows)) == 1
        for key in sleeve["mechanical_shared_winding_keys"]:
            assert belongs_to_material_selection(sleeve, {key})
            assert sleeve["local_center_z_mm"] == pytest.approx(rows[key]["local_center_z_mm"])
            packs = multipole_winding_envelopes(rows[key])
            assert packs and all(math.hypot(p["radial_end_mm"], p["outer_width_mm"]/2)
                                 < sleeve["mechanical_inner_diameter_mm"]/2 for p in packs)
    if IMAGE <= lenses:
        assert rows["image_polarizer_return_sleeve"]["mechanical_shared_winding_keys"] == [
            "image_hpol_hexapole", "image_qpol_quadrupole"]
        assert not any(p["parent_key"] == "image_dstg_quadrupole" for p in sleeves)


@pytest.mark.parametrize("key", ["probe_hp1_hexapole", "probe_hp2_hexapole", "image_hp1_hexapole",
                                  "image_hp2_hexapole", "image_hp3_hexapole"])
def test_six_installed_pin_root_sleeves_contact_the_pin_and_yoke_and_clear_winding(key):
    doc = read_document(COLUMN / "C3_ProbeCorrector_BCOR.toml")
    rows = {p["key"]: p for p in doc["parts"]}
    hp = rows[key]
    solids = {s.region: s for s in multipole_solids(hp, rows)}
    original = {k: v for k, v in hp.items() if not k.startswith("pole_root_sleeve_")}
    for s in multipole_solids(original, rows):
        assert solids[s.region].shape == s.shape
    yoke = _manifold(solids["return_yoke"].shape, 96)
    for i in range(1, 7):
        spec = solids[f"pole_root_sleeve_{i}"]
        pin = _manifold(solids[f"pole_pin_{i}"].shape, 96)
        sleeve = _manifold(spec.shape, 96)
        assert len(sleeve.decompose()) == 1 and spec.material == "soft_magnetic"
        for seat in (pin, yoke):
            assert sleeve.min_gap(seat, 1.) < 1e-7
            assert abs((sleeve ^ seat).volume()) < 1e-6
        packs = multipole_winding_envelopes(hp, rows)
        if packs:
            assert spec.parameters["seat_start_radius_mm"] > packs[i-1]["radial_end_mm"]
        else:
            assert sleeve.min_gap(_manifold(solids[f"coil_{i}"].shape, 96), 10.) > 0


def test_editor_keeps_split_regions_and_rejects_intersecting_sleeves_atomically(tmp_path):
    draft = PartModelDocument(COLUMN / "C3_ProbeCorrector_ImageCorrector.toml")
    root = "probe_hp2_hexapole"
    with pytest.raises(ValueError, match="clear the winding"):
        draft.set_dimension(("parts", root, "pole_root_sleeve_depth_mm"), 6.)
    assert not draft.can_undo
    sleeve = draft.part("image_polarizer_return_sleeve")
    rows = {p["key"]: p for p in draft.document["parts"]}
    with pytest.raises(ValueError, match="clear the winding"):
        validate_auxiliary_return_sleeve({**sleeve, "mechanical_inner_diameter_mm": 51.}, rows)
    key = "image_tl11_lens_pole_sleeves"
    first, last = draft.part(key)["material_intervals_mm"]
    with pytest.raises(ValueError, match="separate, ordered"):
        draft.set_dimension(("parts", key, "material_intervals_mm", 0, 1), last[0])
    assert not draft.can_undo
    with pytest.raises(ValueError, match="Derived axial envelope"):
        draft.set_dimension(("parts", key, "length_mm"), 8.)
    draft.set_dimension(("parts", key, "material_intervals_mm", 0, 0), first[0]+.05)
    assert draft.part(key)["local_start_z_mm"] == pytest.approx(first[0]+.05)
    assert draft.part(key)["length_mm"] == pytest.approx(last[1]-first[0]-.05)
    draft.assign_material(key, "copper", region="upper")
    saved = tmp_path / "split-material.toml"
    draft.save_copy(saved)
    restored = PartModelDocument(saved)
    restored.validate()
    meshes = part_model_from_document(restored.document, key).meshes
    assert [m.region for m in meshes] == ["upper", "lower"]
    saved_part = restored.part(key)
    assert material_for_region(saved_part, "upper")["material_key"] == "copper"
    assert material_for_region(saved_part, "lower") is None
    colours = [configured_region_colour(saved_part, m.region, m.color) for m in meshes]
    assert colours[0] != colours[1]


@pytest.mark.parametrize("part,field,value,joint,defect", [
    ("pole_sleeves", ("mechanical_inner_diameter_mm",), 66.83, "upper_pole to pole_sleeves", "air gap"),
    ("pole_sleeves", ("mechanical_inner_diameter_mm",), 65.33, "upper_pole to pole_sleeves", "overlap"),
    ("lower_pole", ("mechanical_outer_diameter_mm",), 67.33, "lower_pole to pole_sleeves", "overlap"),
    ("return_plates", ("mechanical_inner_diameter_mm",), 71.33, "pole_sleeves to return_plates", "air gap"),
    ("return_plates", ("mechanical_inner_diameter_mm",), 69.33, "pole_sleeves to return_plates", "overlap"),
    ("return_plates", ("mechanical_outer_diameter_mm",), 100.5109425, "return_plates to yoke", "air gap"),
    ("yoke", ("mechanical_inner_diameter_mm",), 100.5109425, "return_plates to yoke", "overlap"),
    ("pole_sleeves", ("material_intervals_mm", 0, 0), 704.35, "pole_sleeves to return_plates", "air gap"),
    ("pole_sleeves", ("material_intervals_mm", 0, 0), 704.15, "pole_sleeves to return_plates", "edge"),
    ("pole_sleeves", ("material_intervals_mm", 1, 1), 728.35, "lower end, pole_sleeves to return_plates", "air gap"),
])
def test_edited_return_joints_reject_gaps_edge_contacts_and_interpenetration(part, field, value, joint, defect):
    source = COLUMN / "C3_ProbeCorrector.toml"
    draft = PartModelDocument(source)
    key = "probe_tl22_lens_" + part
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match=f"{joint}.*{defect}"):
        draft.set_dimension(("parts", key, *field), value)
    assert draft.document == before and not draft.can_undo

    # Also exercise a direct manifest edit that bypasses the 3D editor.
    doc = read_document(source)
    row = next(row for row in doc["parts"] if row["key"] == key)
    if len(field) == 1:
        row[field[0]] = value
    else:
        from temsim.magnetic_geometry import corrector_return_extent
        row[field[0]][field[1]][field[2]] = value
        row.update(corrector_return_extent(row))
    with pytest.raises(ValueError, match=f"{joint}.*{defect}"):
        validate_document(doc)


def test_pole_shoulder_cannot_intersect_a_plate_beyond_the_sleeve():
    doc = read_document(COLUMN / "C3_ImageCorrector.toml")
    rows = {part["key"]: part for part in doc["parts"]}
    pole = rows["image_tl11_lens_upper_pole"]
    plate = rows["image_tl11_lens_return_plates"]
    # The pole still seats on the sleeve for its full axial span. Beyond it,
    # a wider shoulder clears the original plate, but not this extended plate.
    pole["mechanical_outer_diameter_mm"] = 112.
    pole["magnetic_radial_profile_mm"] = [[0., 10., 51.59], [2.2, 10., 51.59],
                                         [3.2, 10., 56.], [4., 10., 56.], [12.5, 10., 20.]]
    validate_document(doc)
    plate["material_intervals_mm"][0][1] = 1044.3
    with pytest.raises(ValueError, match="upper_pole/body to return_plates/upper.*material overlap"):
        validate_document(doc)


def test_local_plate_cut_preserves_contact_but_an_offset_causes_interference(tmp_path):
    draft = PartModelDocument(COLUMN / "C3_ProbeCorrector.toml")
    key = "probe_tl22_lens_return_plates"
    hole = dict(id="through_plate", kind="hole", axis="z", center_mm=[42., 0., 0.],
                depth_mm=40., diameter_mm=2.)
    # A rotation moves the hole around the axis but preserves both round seats.
    draft.set_model_3d(key, dict(schema_version=1, transform={"rotation_deg": [0., 0., 13.]}))
    draft.set_model_3d(key, dict(schema_version=1, features=[hole]))
    draft.update_model_3d(key, {"transform": {"rotation_deg": [0., 0., 13.]}})
    before = deepcopy(draft.document)
    with pytest.raises(ValueError, match=r"air gap 0\.\d+ mm in the 3D mesh"):
        draft.update_model_3d(key, {"transform": {"scale_xy": [1.01, 1.01]}})
    with pytest.raises(ValueError, match="material overlap"):
        draft.update_model_3d(key, {"transform": {"offset_mm": [.2, 0., 0.]}})
    assert draft.document == before
    path = tmp_path / "plate-with-hole.toml"
    draft.save_copy(path)
    PartModelDocument(path).validate()


@pytest.mark.parametrize("host", ["image_hpol_hexapole", "image_qpol_quadrupole"])
def test_both_image_polarizer_previews_select_the_same_single_sleeve(host):
    doc = read_document(COLUMN / "C3_ProbeCorrector_ImageCorrector.toml")
    meshes = part_model_from_document(doc, host, angular_segments=16).meshes
    assert any(mesh.key == host and mesh.material_class == "copper_winding" for mesh in meshes)
    assert sum(mesh.key == "image_polarizer_return_sleeve" for mesh in meshes) == 1


@pytest.mark.parametrize("key", ["probe_tl22_lens", "image_tl11_lens"])
def test_installed_return_material_modifies_round_lens_field_at_fixed_ampere_turns(key):
    state, catalog = default_state(), AssemblyCatalog()
    selection = replace(catalog.default_selection(), column="C3 + Probe Corrector + Image Corrector")
    assembly = catalog.apply(state, selection)
    state.simulation_mode = "custom"
    guides = {key+"_return_plates", key+"_pole_sleeves"}
    assert not guides.intersection(segment.key for segment in assembly.vacuum_bore_segments)
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, "material_class": "guide_control"}) if p.key in guides else p
        for p in assembly.parts))
    geometry = json.loads(lens_geometry_binding(state, key).canonical_geometry_json)
    payload = {p["key"]: p for p in geometry["lens_assembly"]["parts"]}
    for guide in guides:
        row = payload[guide]
        intervals = np.array(row["data"]["material_intervals_mm"])
        r = .25e-3*(row["data"]["mechanical_outer_diameter_mm"] + row["data"]["mechanical_inner_diameter_mm"])
        z = np.array([*intervals.mean(axis=1), assembly.part(key).center_z_mm])*1e-3
        np.testing.assert_equal(_part_mask(row, r, z), [True, True, False])
    lens = next(item for item in state.lenses if item.key == key)
    points = np.zeros((65, 3))
    points[:, 2] = np.linspace(assembly.part(key).start_z_mm, assembly.part(key).end_z_mm, len(points))*1e-3
    fields = []
    for mu in (1., 100.):
        state.lens_field_map_descriptors = {key: dict(
            solver="axisymmetric_linear_fem", relative_permeability=100.,
            material_permeabilities={"guide_control": mu}, ampere_turns=100.,
            radial_nodes=40, axial_nodes=80, padding_factor=2.)}
        provider = resolve_runtime_lens_field_provider(state, key, lens)
        fields.append(provider.field_map.field_at_global_positions_t(points)[:, 2])
    assert np.isfinite(fields).all()
    assert np.linalg.norm(fields[1]-fields[0]) > .01*np.linalg.norm(fields[0])
