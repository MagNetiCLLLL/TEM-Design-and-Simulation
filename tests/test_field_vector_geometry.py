"""Captured physical material versus selected field sources, without a GUI."""
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from temsim.assembly_model_3d import assembly_model_from_assembly
from temsim.column.module_assembly import AssemblyPart, ModulePart
from temsim.field_vector_geometry import build_field_vector_geometry


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _part(key="coil", **changes):
    row = dict(key=key, name=key, branch="illumination", parent_key=None,
               local_start_z_mm=10., local_center_z_mm=12., local_end_z_mm=14., length_mm=4.,
               mechanical_profile="magnetic_excitation_coil", mechanical_inner_diameter_mm=4.,
               mechanical_outer_diameter_mm=8., material_class="insulated_copper_winding")
    row.update(changes)
    return row


def _state(rows, *, shift=100.):
    module, active = [], []
    for row in rows:
        values = (row["key"], row["name"], row["branch"], row["local_start_z_mm"],
                  row["local_center_z_mm"], row["local_end_z_mm"], row["length_mm"],
                  row.get("parent_key"), _freeze(deepcopy(row)))
        module.append(ModulePart(*values))
        active.append(AssemblyPart("module", "captured.toml", *values[:3],
                                   values[3] + shift, values[4] + shift, values[5] + shift, *values[6:]))
    assembly = SimpleNamespace(modules=(SimpleNamespace(key="module", parts=tuple(module)),),
                               parts=tuple(active), vacuum_liner_segments=())
    return SimpleNamespace(_resolved_assembly=assembly)


def _source(key, *components):
    return SimpleNamespace(key=key, component_keys=components or (key,))


def test_global_xyz_metres_matches_physical_layout_and_converts_edges_once():
    state = _state([_part()], shift=987.)
    original = assembly_model_from_assembly(state._resolved_assembly, angular_segments=8).meshes[0]
    result = build_field_vector_geometry(state, [_source("coil")], angular_segments=8)
    mesh, = result.meshes
    np.testing.assert_array_equal(mesh.vertices, original.vertices * 1e-3)
    np.testing.assert_array_equal(mesh.faces, original.faces)
    assert mesh.vertices[:, 2].min() == pytest.approx(.997, abs=3e-16)
    assert mesh.vertices[:, 2].max() == pytest.approx(1.001, abs=3e-16)
    assert mesh.color == original.color and mesh.material_class == original.material_class
    assert mesh.is_exact == original.is_exact and mesh.wireframe == original.wireframe
    assert len(mesh.edges) > 0
    for new, old in zip(mesh.edges, original.edges, strict=True):
        np.testing.assert_array_equal(new["vertices"], old["vertices"] * 1e-3)
    for array in (mesh.vertices, mesh.faces, mesh.edges[0]["vertices"]):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        result.source_mesh_keys["coil"] = ()


def test_shared_circuit_descendants_and_repeated_selections_draw_each_region_once():
    state = _state([
        _part("lens1", mechanical_profile="magnetic_lens_assembly"),
        _part("lens2", mechanical_profile="magnetic_lens_assembly"),
        _part("shared_coil", parent_key="lens1", magnetic_lens_keys=["lens1", "lens2"]),
        _part("coil_support", parent_key="shared_coil"),
        _part("unrelated"),
    ])
    result = build_field_vector_geometry(state, [_source("joint", "lens1", "lens2"),
                                               _source("second", "lens2")], angular_segments=8)
    assert result.source_mesh_keys["joint"] == ("coil_support", "shared_coil")
    assert result.source_mesh_keys["second"] == result.source_mesh_keys["joint"]
    selected = result.select(("joint", "second", "joint"))
    assert len(selected) == 2
    assert set(mesh.key for mesh in selected) == {"shared_coil", "coil_support"}
    assert result.select(()) == ()
    with pytest.raises(ValueError, match="Unknown field source"):
        result.select(("unknown",))


def test_descan_and_image_diffraction_share_one_physical_host():
    state = _state([_part("image_diffraction_deflector"),
                    _part("descan_deflector", physical_host_key="image_diffraction_deflector")])
    result = build_field_vector_geometry(state, [_source("descan_deflector"),
                                               _source("image_diffraction_deflector")], angular_segments=8)
    assert result.source_mesh_keys["descan_deflector"] == ("image_diffraction_deflector",)
    assert len(result.select()) == 1
    assert "shared material host" in " ".join(result.source_notes["descan_deflector"])


def test_explicit_insert_binding_works_but_optical_reference_does_not_create_material():
    state = _state([
        _part("insert", mechanical_part_role="deflector_coil_insert", mechanical_only=True),
        _part("channel", layout_role="control_channel", physical_host_key="insert", physical_host_status="modelled"),
        _part("unverified", layout_role="control_channel", physical_host_key="insert", physical_host_status="unverified"),
        _part("reference", layout_role="control_channel", optical_reference_host_key="insert"),
    ])
    result = build_field_vector_geometry(state, [_source(key) for key in ("channel", "unverified", "reference")],
                                         angular_segments=8)
    assert result.source_mesh_keys["channel"] == ("insert",)
    assert result.source_mesh_keys["unverified"] == result.source_mesh_keys["reference"] == ()
    assert all("no independently modelled material" in " ".join(result.source_notes[key])
               for key in ("unverified", "reference"))


def test_old_snapshot_never_rereads_files_or_borrows_changed_geometry(monkeypatch):
    row = _part()
    old_state = _state([row])
    row["mechanical_outer_diameter_mm"] = 80.
    new_state = _state([row], shift=200.)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Geometry must come from captured resolved rows")

    monkeypatch.setattr(Path, "read_text", unexpected_read)
    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    old = build_field_vector_geometry(old_state, [_source("coil")], angular_segments=8)
    new = build_field_vector_geometry(new_state, [_source("coil")], angular_segments=8)
    assert old.meshes[0].vertices[:, 0].max() == .004
    assert old.meshes[0].vertices[:, 2].min() == .11
    assert new.meshes[0].vertices[:, 0].max() == .04
    assert new.meshes[0].vertices[:, 2].min() == .21
    np.testing.assert_array_equal(build_field_vector_geometry(old_state, [_source("coil")], angular_segments=8).meshes[0].vertices,
                                  old.meshes[0].vertices)


def test_material_translation_follows_parent_but_optical_tilt_does_not_rotate_it():
    parent = _part("lens", mechanical_profile="magnetic_lens_assembly")
    child = _part("winding", parent_key="lens")
    base = build_field_vector_geometry(_state([parent, child]), [_source("lens")], angular_segments=8)
    parent.update(offset_x_mm=3., offset_y_mm=-2., offset_z_mm=4.,
                  rotation_x_mrad=120., rotation_y_mrad=-50., rotation_z_mrad=80.)
    moved = build_field_vector_geometry(_state([parent, child]), [_source("lens")], angular_segments=8)
    np.testing.assert_allclose(moved.meshes[0].vertices, base.meshes[0].vertices + (.003, -.002, .004),
                               rtol=0, atol=3e-17)
    np.testing.assert_array_equal(moved.meshes[0].faces, base.meshes[0].faces)


def test_missing_snapshot_retains_sources_and_explicit_unavailable_message():
    result = build_field_vector_geometry(SimpleNamespace(), [_source("old")], angular_segments=8)
    assert not result.meshes and result.source_mesh_keys["old"] == ()
    assert "current files are not substituted" in " ".join(result.notes)


def test_unavailable_part_does_not_erase_other_geometry_and_approximation_stays_labelled():
    state = _state([_part("bad", mechanical_inner_diameter_mm=20.), _part("good"),
                    _part("approx", mechanical_profile="unknown_profile")])
    result = build_field_vector_geometry(state, [_source(key) for key in ("bad", "good", "approx")], angular_segments=8)
    assert len(result.errors) == 1 and result.errors[0].startswith("bad:")
    assert result.source_mesh_keys["bad"] == ()
    assert result.source_mesh_keys["good"] == ("good",)
    assert not next(mesh for mesh in result.meshes if mesh.key == "approx").is_exact
    assert "Envelope only" in " ".join(result.source_notes["approx"])


@pytest.mark.parametrize("segments", [True, 2, 4.5])
def test_invalid_tessellation_rejected_even_when_assembly_is_missing(segments):
    with pytest.raises(ValueError, match="angular_segments"):
        build_field_vector_geometry(SimpleNamespace(), (), angular_segments=segments)


def test_duplicate_field_source_keys_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_field_vector_geometry(SimpleNamespace(), [_source("same"), _source("same")])


def test_real_hp2_tl22_and_dph2_reuse_production_physical_regions_and_positions():
    from temsim.optics.column import default_state
    state = default_state()
    sources = [_source(key) for key in ("probe_hp2_hexapole", "probe_tl22_lens", "probe_dph2_deflector")]
    result = build_field_vector_geometry(state, sources, angular_segments=8)
    assert not result.errors
    physical = assembly_model_from_assembly(state._resolved_assembly, angular_segments=8)
    originals = {(mesh.key, mesh.region): mesh for mesh in physical.meshes}
    for mesh in result.meshes:
        original = originals[mesh.key, mesh.region]
        np.testing.assert_array_equal(mesh.vertices, original.vertices * 1e-3)
        np.testing.assert_array_equal(mesh.faces, original.faces)
        assert mesh.color == original.color and mesh.is_exact == original.is_exact
    hp = [mesh for mesh in result.meshes if mesh.key == "probe_hp2_hexapole"]
    assert sum(mesh.region.startswith("pole_pin_") for mesh in hp) == 6
    assert {mesh.region for mesh in hp if mesh.material_class == "copper_winding"} == {
        f"coil_{index}" for index in range(1, 7)}
    assert sum(mesh.region.endswith("_former") for mesh in hp) == 6
    assert sum(mesh.region.endswith("_core_bedding") for mesh in hp) == 6
    assert len(hp) == 25
    assert {mesh.key for mesh in result.select(("probe_tl22_lens",))} == {
        "probe_tl22_lens_upper_pole", "probe_tl22_lens_lower_pole", "probe_tl22_lens_housing",
        "probe_tl22_lens_yoke", "probe_tl22_lens_excitation_coil",
    }
    hp2 = next(component for component in state.corrector_elements if component.key == "probe_hp2_hexapole")
    yoke = next(mesh for mesh in hp if mesh.region == "return_yoke")
    assert np.mean([yoke.vertices[:, 2].min(), yoke.vertices[:, 2].max()]) == pytest.approx(hp2.z_mm * 1e-3)
    # The DPH insert is mechanical context within HP2; selecting its field as
    # well must not overlay a second opaque copy of the same hardware.
    both = result.select(("probe_hp2_hexapole", "probe_dph2_deflector"))
    assert len(both) == len(result.select(("probe_hp2_hexapole",)))
    assert "assembly context" in " ".join(result.notes)
