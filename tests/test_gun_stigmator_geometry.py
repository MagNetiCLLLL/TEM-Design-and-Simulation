"""Bounded gun stigmator reconstruction; unchanged optical authority."""
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import tomllib

import numpy as np
import pytest

from temsim.cad_geometry import annulus
from temsim.cad_gun_stigmator import (
    PREFIX, FIELDS, gun_stigmator_dimensions, gun_stigmator_solids,
)
from temsim.cad_mesh import _manifold, meshes_from_solids


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [("FEG.toml", "FEG", "feg_stigmator", 433.),
            ("FEG_Mono.toml", "FEG + Mono", "feg_stigmator", 483.),
            ("Thermionic.toml", "Thermionic", "thermionic_stigmator", 433.)]


def _document(filename):
    return tomllib.loads((ROOT/"configs/instruments/gun"/filename).read_text(encoding="utf-8"))


@pytest.fixture
def part():
    return next(p for p in _document("FEG.toml")["parts"] if p["key"] == "feg_stigmator")


@pytest.mark.parametrize("filename,selection,key,centre", VARIANTS)
def test_saved_gun_variants_opt_in_without_changing_envelope_or_active_field(filename, selection, key, centre):
    part = next(p for p in _document(filename)["parts"] if p["key"] == key)
    original = deepcopy(part)
    assert part["local_center_z_mm"] == centre
    assert (part["local_start_z_mm"], part["local_end_z_mm"]) == (centre-6, centre+6)
    assert (part["length_mm"], part["active_length_mm"], part["bore_diameter_mm"],
            part["outer_diameter_mm"]) == (12., 8., 10., 90.)
    assert {name for name in part if name.startswith(PREFIX)} == FIELDS
    specs = gun_stigmator_solids(part)
    meshes = meshes_from_solids(specs, 32)
    assert len(specs) == len(meshes) == 33
    assert all(s.status == "engineering_reconstruction" for s in specs)
    assert all(not m.is_exact and not m.wireframe for m in meshes)
    assert all(not s.parameters["field_source_added"] for s in specs)
    assert all("not OEM" in s.note and "objective-stigmator" in s.note for s in specs)
    vertices = np.concatenate([m.vertices for m in meshes])
    assert vertices[:, 2].min() == pytest.approx(centre-6)
    assert vertices[:, 2].max() == pytest.approx(centre+6)
    radii = np.linalg.norm(vertices[:, :2], axis=1)
    assert radii.min() == pytest.approx(5.)
    assert radii.max() == pytest.approx(45.)
    coils = [s for s in specs if s.material == "copper_winding"]
    assert len(coils) == 8
    assert [s.parameters["coil_azimuth_deg"] for s in coils] == list(np.arange(0., 360., 45.))
    assert [s.parameters["quadrupole_basis"] for s in coils].count("normal") == 4
    assert [s.parameters["quadrupole_basis"] for s in coils].count("skew") == 4
    assert part == original


def test_all_materials_are_positive_disjoint_and_keep_the_entire_beam_bore_open(part):
    specs = gun_stigmator_solids(part)
    solids = [(s.region, _manifold(s.shape, 32)) for s in specs]
    bore = _manifold(annulus(426., 440., 0., 5.), 32)
    for region, solid in solids:
        assert solid.volume() > 0, region
        assert (solid ^ bore).volume() == pytest.approx(0., abs=1e-7), region
    for (name_a, a), (name_b, b) in combinations(solids, 2):
        assert (a ^ b).volume() == pytest.approx(0., abs=1e-7), (name_a, name_b)
    # Eight open slots enter from the downstream end; the continuous carrier
    # base and central annulus remain present, rather than a beam-blocking cap.
    carrier = solids[0][1]
    assert carrier.volume() < _manifold(annulus(427., 439., 5., 45.), 32).volume()
    assert len(carrier.decompose()) == 1


def test_coil_former_and_adhesive_share_mounting_faces_without_intersection(part):
    d = gun_stigmator_dimensions(part)
    meshes = {m.region: m for m in meshes_from_solids(gun_stigmator_solids(part), 32)}
    copper = meshes["coil_1"].vertices
    former = meshes["coil_former_1"].vertices
    coating = meshes["coil_insulation_1"].vertices
    adhesive = meshes["adhesive_seat_1"].vertices
    assert copper[:, 2].min() == pytest.approx(d["coil_start_mm"])
    assert copper[:, 2].max() == pytest.approx(d["coil_end_mm"])
    assert adhesive[:, 2].max() == pytest.approx(former[:, 2].min())
    assert adhesive[:, 2].max() == pytest.approx(coating[:, 2].min())
    assert adhesive[:, 2].min() == pytest.approx(d["pocket_start_mm"])
    assert d["start_mm"] < d["pocket_start_mm"] < former[:, 2].min()


@pytest.mark.parametrize("filename,selection,key,centre", VARIANTS)
def test_native_cad_and_reconstruction_permissions_use_one_shared_builder(filename, selection, key, centre):
    from temsim.cad_catalog import part_solids, native_reconstruction, is_reconstruction_region
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument
    document = PartModelDocument(ROOT/"configs/instruments/gun"/filename).document
    parts = {p["key"]: p for p in document["parts"]}
    part = parts[key]
    expected = gun_stigmator_solids(part)
    assert part_solids(part, parts) == native_reconstruction(part, parts) == expected
    native = part_model_from_document(document, key, include_children=False, angular_segments=32)
    exported = meshes_from_solids(expected, 32)
    assert len(native.meshes) == len(exported) == 33
    for mesh, cad in zip(native.meshes, exported, strict=True):
        np.testing.assert_allclose(mesh.vertices, cad.vertices, atol=1e-12)
        np.testing.assert_array_equal(mesh.faces, cad.faces)
        assert mesh.surfaces and mesh.face_groups is not None
        assert mesh.material_class == cad.material_class and not mesh.is_exact
        assert is_reconstruction_region(part, mesh.region)


@pytest.mark.parametrize("field,value", [
    ("coil_length_mm", 6.), ("coil_width_mm", 7.), ("coil_inner_radius_mm", 17.),
    ("coil_outer_radius_mm", 33.), ("window_width_mm", 3.2), ("window_length_mm", 4.2),
    ("insulation_thickness_mm", .3), ("former_wall_thickness_mm", .6),
    ("adhesive_thickness_mm", .3), ("basis_angle_deg", 7.),
])
def test_saved_geometry_parameters_change_materials_but_not_active_field_length(part, field, value):
    before = gun_stigmator_solids(part)
    updated = {**part, PREFIX+field: value}
    after = gun_stigmator_solids(updated)
    assert [s.shape for s in before] != [s.shape for s in after]
    assert updated["active_length_mm"] == part["active_length_mm"] == 8.
    assert updated["length_mm"] == part["length_mm"] == 12.
    assert all(_manifold(s.shape, 24).volume() > 0 for s in after)


@pytest.mark.parametrize("field,value", [
    ("coil_length_mm", True), ("coil_length_mm", float("nan")), ("coil_length_mm", -1.),
    ("coil_length_mm", 12.), ("coil_width_mm", 50.), ("coil_inner_radius_mm", 5.),
    ("coil_outer_radius_mm", 45.), ("window_width_mm", 1.), ("window_length_mm", 8.),
    ("insulation_thickness_mm", 2.), ("former_wall_thickness_mm", 2.),
    ("adhesive_thickness_mm", 5.), ("basis_angle_deg", float("inf")),
    ("coil_count", True), ("coil_count", 4), ("geometry_status", "OEM"),
    ("geometry_profile", "unknown"), ("support_source", ""), ("source_note", ""),
])
def test_malformed_or_intersecting_geometry_is_rejected(part, field, value):
    with pytest.raises(ValueError):
        gun_stigmator_solids({**part, PREFIX+field: value})


def test_unknown_partial_and_nonmaterial_declarations_are_rejected(part):
    for update in ({PREFIX+"unexpected_length_mm": 1.}, {"layout_role": "control_channel"},
                   {"key": "condenser_stigmator"}):
        with pytest.raises(ValueError):
            gun_stigmator_dimensions({**part, **update})
    missing = dict(part)
    del missing[PREFIX+"former_wall_thickness_mm"]
    with pytest.raises(ValueError):
        gun_stigmator_dimensions(missing)
    legacy = {name: value for name, value in part.items() if not name.startswith(PREFIX)}
    assert gun_stigmator_dimensions(legacy) is None
    assert gun_stigmator_solids(legacy) is None


def test_integer_overflow_is_a_validation_error_not_an_unhandled_conversion(part):
    with pytest.raises(ValueError, match="must be finite"):
        gun_stigmator_dimensions({**part, PREFIX+"coil_length_mm": 10**1000})


@pytest.mark.parametrize("filename,selection,key,centre", VARIANTS)
def test_current_gun_field_and_neighbour_liner_clearance_are_preserved(filename, selection, key, centre):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.cad_stage_holder import physical_liner_rows
    from temsim.cad_catalog import part_solids
    from temsim.optics.column import default_state
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun=selection),
                  preserve_operating_parameters=True)
    stigmator = state.electron_gun.stigmator
    stigmator.gradient_t_per_m, stigmator.rotation_deg = .8, 27.
    before = deepcopy(asdict(stigmator))
    positions = np.array([[.001, .002, centre*.001], [-.002, .001, (centre-3)*.001],
                          [.001, -.001, (centre+6)*.001]])
    field = stigmator.field_at_global_positions_t(positions)
    assert np.any(field != 0)
    document = _document(filename)
    parts = {p["key"]: p for p in document["parts"]}
    gun_stigmator_solids(parts[key])
    gun_stigmator_solids({**parts[key], PREFIX+"coil_length_mm": 6.})
    assert asdict(stigmator) == before
    np.testing.assert_array_equal(stigmator.field_at_global_positions_t(positions), field)
    assert (stigmator.mechanical_length_mm, stigmator.effective_length_mm,
            stigmator.mechanical_clear_bore_diameter_mm, stigmator.mechanical_outer_diameter_mm) == (12., 8., 10., 90.)
    lo, hi = centre-6, centre+6
    for neighbour in parts.values():
        if neighbour["key"] != key:
            assert neighbour["local_end_z_mm"] <= lo or neighbour["local_start_z_mm"] >= hi
    # Production liners are generated from the same resolved assembly. None
    # traverses the new material allocation; no separate tube is invented here.
    for liner in physical_liner_rows(state._resolved_assembly):
        if liner["local_start_z_mm"] < hi and liner["local_end_z_mm"] > lo:
            solids = gun_stigmator_solids(parts[key])
            other = part_solids(liner, {liner["key"]: liner})
            for a in solids:
                for b in other:
                    assert (_manifold(a.shape, 24) ^ _manifold(b.shape, 24)).volume() == pytest.approx(0., abs=1e-7)


@pytest.mark.parametrize("filename,selection,key,centre", VARIANTS)
def test_independent_copy_replacement_owns_geometry_and_restore_is_atomic(filename, selection, key, centre):
    from temsim.cad_catalog import posed_solids
    from temsim.module_manifest import validate_document
    from temsim.part_model_3d import part_model_from_document
    from temsim.part_model_document import PartModelDocument

    draft = PartModelDocument(ROOT/"configs/instruments/gun"/filename)
    original = deepcopy(draft.part(key))
    copied = draft.copy_component_from(draft.document, key, "replaced_gun_stigmator",
                                       2500., include_children=False)
    copied_native = deepcopy(draft.document)
    draft.update_model_3d(copied, {"base": {"kind": "box", "width_mm": 10.,
                                          "height_mm": 10., "length_mm": 1.}})
    validate_document(draft.document)
    assert gun_stigmator_dimensions(draft.part(copied)) is None
    assert gun_stigmator_solids(draft.part(copied)) is None
    model = part_model_from_document(draft.document, copied, include_children=False)
    assert len(model.meshes) == 1
    np.testing.assert_allclose(np.ptp(model.meshes[0].vertices, axis=0), [10., 10., 1.])
    solids = posed_solids(draft.part(copied), {p["key"]: p for p in draft.document["parts"]})
    assert len(solids) == 1 and _manifold(solids[0].shape, 32).volume() == pytest.approx(100.)
    assert draft.part(key) == original
    before, history = deepcopy(draft.document), deepcopy(draft._history)
    history_index = draft._history_index
    native = deepcopy(draft.part(copied)["model_3d"])
    native["base"] = {"kind": "existing"}
    with pytest.raises(ValueError, match="body envelope"):
        draft.set_model_3d(copied, native)
    assert draft.document == before and draft._history == history
    assert draft._history_index == history_index
    draft.undo()
    assert draft.document == copied_native
    assert len(part_model_from_document(draft.document, copied, include_children=False).meshes) == 33
    draft.redo()
    assert draft.document == before
