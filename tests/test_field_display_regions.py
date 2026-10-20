"""Read-only physical display envelopes and source ownership, without a solve."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from temsim.column.module_assembly import AssemblyPart, ModulePart
from temsim.field_display_regions import build_field_display_regions


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
    assembly = SimpleNamespace(modules=(SimpleNamespace(key="module", parts=tuple(module), geometry={}),),
                               parts=tuple(active))
    return SimpleNamespace(_resolved_assembly=assembly)


def _source(key, *components, **changes):
    return SimpleNamespace(key=key, component_keys=components or (key,), **changes)


def test_mask_uses_captured_global_physical_outer_envelope_including_bore():
    row = _part()
    state = _state([row], shift=987.)
    source = _source("coil", bounds_m=np.array([[-1., -1., -1.], [1., 1., 10.]]))
    region_index = build_field_display_regions(state, [source])
    row["mechanical_outer_diameter_mm"] = 800.
    np.testing.assert_allclose(region_index.bounds_for_component("coil"),
                               [[-.004, -.004, .997], [.004, .004, 1.001]])
    points = [[0., 0., .999], [.003, 0., .999], [.0041, 0., .999], [0., 0., .996]]
    assert region_index.mask(points, "coil").tolist() == [True, True, False, False]
    assert region_index.source_keys_for_component("coil") == ("coil",)
    assert source.bounds_m[1, 2] == 10.  # Provider support cannot define this mask.
    region, = region_index.regions_for_component("coil")
    for array in (region.bounds_m, region.origin_m, region.inverse_linear, region.half_extents_m,
                  region_index.bounds_for_component()):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        region_index.component_source_keys["coil"] = ()
    with pytest.raises(FrozenInstanceError):
        region.component_key = "other"


def test_all_regions_are_a_union_and_passive_body_can_observe_without_own_field():
    rows = [_part("objective_lens", mechanical_profile="magnetic_lens_assembly"),
            _part("sleeve", mechanical_profile="objective_electronics_sleeve",
                  parent_key="objective_lens", mechanical_only=True, local_start_z_mm=20.,
                  local_center_z_mm=22., local_end_z_mm=24.),
            _part("aperture", mechanical_profile="aperture_plate", parent_key="objective_lens",
                  local_start_z_mm=30., local_center_z_mm=32., local_end_z_mm=34.),
            _part("copy", mechanical_part_role="custom_mechanical_copy",
                  parent_key="objective_lens", field_source_key="objective_lens")]
    result = build_field_display_regions(_state(rows), [_source("objective_lens")])
    points = [[0., 0., z] for z in (.112, .117, .122, .127, .132)]
    assert result.mask(points).tolist() == [True, False, True, False, True]
    assert result.mask(points, "sleeve").tolist() == [False, False, True, False, False]
    for key in ("sleeve", "aperture", "copy"):
        assert result.source_keys_for_component(key) == ()
        assert result.regions_for_component(key)
    assert result.regions_for_component("missing") == ()
    assert result.bounds_for_component("missing") is None
    assert not result.mask(points, "missing").any()


def test_objective_children_and_shared_nonlinear_circuit_select_complete_groups_once():
    rows = [_part("objective_lens", mechanical_profile="magnetic_lens_assembly"),
            _part("mini_condenser", mechanical_profile="magnetic_lens_assembly",
                  parent_key="objective_lens"),
            _part("objective_upper_pole", mechanical_profile="magnetic_pole_piece",
                  parent_key="objective_lens"),
            _part("objective_lower_pole", mechanical_profile="magnetic_pole_piece",
                  parent_key="objective_lens"),
            _part("objective_coil", parent_key="objective_lens", field_source_key="objective_lens"),
            _part("mini_coil", parent_key="mini_condenser", field_source_key="mini_condenser"),
            _part("shared_body", mechanical_profile="c1_c2_pole_piece_cartridge",
                  parent_key="objective_lens", magnetic_lens_keys=["objective_lens", "mini_condenser"])]
    state = _state(rows)
    independent = build_field_display_regions(state, [_source("objective_lens"), _source("mini_condenser")])
    for key in ("objective_upper_pole", "objective_lower_pole", "objective_coil"):
        assert independent.source_keys_for_component(key) == ("objective_lens",)
    assert independent.source_keys_for_component("mini_coil") == ("mini_condenser",)
    assert independent.source_keys_for_component("shared_body") == ("objective_lens", "mini_condenser")
    joint = build_field_display_regions(state, [_source("joint", "objective_lens", "mini_condenser")])
    for row in rows:
        assert joint.source_keys_for_component(row["key"]) == ("joint",)


def test_shared_descan_and_bound_corrector_insert_map_both_directions_without_virtual_body():
    rows = [_part("image_diffraction_deflector"),
            _part("descan_deflector", physical_host_key="image_diffraction_deflector"),
            _part("insert", mechanical_part_role="deflector_coil_insert", mechanical_only=True),
            _part("channel", layout_role="control_channel", physical_host_key="insert",
                  physical_host_status="modelled"),
            _part("optical_alias", layout_role="control_channel", optical_reference_host_key="insert"),
            _part("unverified", layout_role="control_channel", physical_host_key="insert",
                  physical_host_status="unverified"),
            _part("probe_dp12_scan_deflector")]
    sources = [_source("pair", "image_diffraction_deflector", "descan_deflector"),
               _source("channel"), _source("optical_alias"), _source("unverified"),
               _source("probe_dp12_scan_deflector", model="reference_zero")]
    result = build_field_display_regions(_state(rows), sources)
    assert result.source_keys_for_component("image_diffraction_deflector") == ("pair",)
    assert result.source_keys_for_component("descan_deflector") == ("pair",)
    assert result.regions_for_component("descan_deflector") == result.regions_for_component("image_diffraction_deflector")
    assert result.source_keys_for_component("insert") == ("channel",)
    assert result.source_keys_for_component("channel") == ("channel",)
    assert result.regions_for_component("channel") == result.regions_for_component("insert")
    assert not result.regions_for_component("optical_alias")
    assert not result.regions_for_component("unverified")
    assert result.source_keys_for_component("probe_dp12_scan_deflector") == ()
    assert result.regions_for_component("probe_dp12_scan_deflector") == ()


@pytest.mark.parametrize("shape", ["box", "elliptic_cylinder"])
def test_cad_rotation_offset_and_scale_mask_in_local_frame_not_global_aabb(shape):
    row = _part("passive", mechanical_profile="custom_mechanical",
                mechanical_part_role="custom_mechanical",
                model_3d=dict(schema_version=1,
                              base=dict(kind=shape, width_mm=8., height_mm=2., length_mm=4.),
                              transform=dict(scale_xy=[2., 1.], rotation_deg=[25., 35., 40.],
                                             offset_mm=[3., -2., 5.])))
    result = build_field_display_regions(_state([row], shift=987.), ())
    region, = result.regions
    from temsim.part_model_features import _transform
    matrix, offset = _transform(row)
    centre = np.array([0., 0., 12.])
    local_mm = np.array([[0., 0., 0.], [3., .1, .2], [4.1, 0., 0.], [0., 0., 2.1]])
    global_m = (local_mm @ matrix.T + centre + offset + [0., 0., 987.]) * 1e-3
    assert region.contains(global_m).tolist() == [True, True, False, False]
    # A global box corner is inside the AABB but outside the rotated primitive.
    corner = region.bounds_m[0] + .01 * (region.bounds_m[1] - region.bounds_m[0])
    assert not region.contains(corner[None, :])[0]
    np.testing.assert_allclose(region.origin_m, [0.003, -.002, 1.004], atol=1e-15)


def test_native_cylinder_cad_transform_and_parent_material_pose_match_layout_semantics():
    parent = _part("lens", mechanical_profile="magnetic_lens_assembly",
                   offset_x_mm=3., offset_y_mm=-2., offset_z_mm=4.,
                   rotation_x_mrad=120., rotation_y_mrad=-50.)
    child = _part("winding", parent_key="lens",
                  model_3d=dict(schema_version=1, transform=dict(rotation_deg=[0., 45., 0.])))
    result = build_field_display_regions(_state([parent, child]), [_source("lens")])
    parent_region, = result.regions_for_component("lens")
    child_region, = result.regions_for_component("winding")
    np.testing.assert_array_equal(parent_region.inverse_linear, np.eye(3))
    np.testing.assert_allclose(parent_region.origin_m, [.003, -.002, .116])
    assert not np.array_equal(child_region.inverse_linear, np.eye(3))
    np.testing.assert_allclose(child_region.origin_m, parent_region.origin_m)
    # Existing legacy lens CAD transforms are physical placement, not a second
    # shape rotation/translation, so no double application is allowed.
    legacy = _part("lens", mechanical_profile="magnetic_lens_assembly",
                   model_3d=dict(schema_version=1, transform=dict(offset_mm=[3., -2., 4.],
                                                                 rotation_deg=[10., 20., 0.])))
    captured = build_field_display_regions(_state([legacy]), [_source("lens")]).regions[0]
    np.testing.assert_allclose(captured.origin_m, parent_region.origin_m)
    np.testing.assert_array_equal(captured.inverse_linear, np.eye(3))


def test_tube_profile_preserves_reduced_radius_and_rotated_envelope():
    from temsim.cad_vacuum_tube import PROFILE
    from temsim.part_model_features import _transform
    row = _part("tube", mechanical_profile="vacuum_liner",
                model_3d=dict(schema_version=1, transform=dict(rotation_deg=[0., 40., 0.])))
    row[PROFILE] = [[10., 8.], [12., 2.], [14., 4.]]
    result = build_field_display_regions(_state([row]), ())
    region, = result.regions
    matrix, offset = _transform(row)
    local_mm = np.array([[0., 0., 0.], [1.2, 0., 0.], [3., 0., -1.9], [2.1, 0., 1.9]])
    points = (local_mm @ matrix.T + [0., 0., 112.]) * 1e-3
    assert region.contains(points).tolist() == [True, False, True, False]
    assert region.axial_radius_profile_m == ((-.002, .004), (0., .001), (.002, .002))
    # Tight rotated bounds use the declared radius at each node, not max OD
    # extruded through the entire tube. Surface samples stay inside the bound.
    angle = np.linspace(0., 2 * np.pi, 180, endpoint=False)
    local_surface = np.concatenate([np.column_stack((r * np.cos(angle), r * np.sin(angle),
                                                    np.full(len(angle), z)))
                                    for z, r in region.axial_radius_profile_m])
    surface = local_surface @ matrix.T + region.origin_m
    assert np.all(surface >= region.bounds_m[0] - 1e-12)
    assert np.all(surface <= region.bounds_m[1] + 1e-12)
    invalid = deepcopy(row)
    invalid[PROFILE][1][0] = 9.
    assert not build_field_display_regions(_state([invalid]), ()).regions


def test_only_independent_physical_liners_are_observation_volumes():
    state = _state([])
    def segment(key, start):
        return SimpleNamespace(key=key, name=key, start_z_mm=start, end_z_mm=start+4.,
                               inner_diameter_mm=2., outer_diameter_mm=4.)
    state._resolved_assembly.vacuum_liner_segments = (
        segment("@vacuum_liner:electrode", 100.),
        segment("@vacuum_liner:@vacuum_drift:gap", 110.),
        segment("declared_tube", 120.))
    result = build_field_display_regions(state, ())
    assert {r.component_key for r in result.regions} == {
        "@vacuum_liner:@vacuum_drift:gap", "declared_tube"}
    assert result.mask([[0., 0., z] for z in (.102, .112, .122)]).tolist() == [False, True, True]
    assert result.source_keys_for_component("declared_tube") == ()


def test_missing_unsupported_and_invalid_geometry_never_use_field_bounds_or_erase_good_parts(monkeypatch):
    def no_io_or_field(*args, **kwargs):
        raise AssertionError("Observation indexing must not read files, construct meshes or evaluate fields")
    monkeypatch.setattr(Path, "read_text", no_io_or_field)
    monkeypatch.setattr(Path, "read_bytes", no_io_or_field)
    from temsim import part_model_3d
    monkeypatch.setattr(part_model_3d, "part_model_from_document", no_io_or_field)
    sources = [_source("missing", bounds_m=np.array([[-1., -1., 0.], [1., 1., 20.]])),
               _source("good", field_at_global_positions_t=no_io_or_field)]
    rows = [_part("good"), _part("missing", mechanical_outer_diameter_mm=None),
            _part("step", model_3d=dict(schema_version=1, base=dict(kind="step", asset_json="missing.json"))),
            _part("branch", branch_path_only=True),
            _part("flat", local_start_z_mm=12., local_end_z_mm=12., length_mm=0.),
            _part("invalid", model_3d=dict(schema_version=1, transform=dict(scale_xy=[0., 1.])))]
    result = build_field_display_regions(_state(rows), sources)
    assert [r.component_key for r in result.regions] == ["good"]
    assert len(result.notes) > 3
    assert "field-support substitute" in " ".join(result.component_notes["step"])
    assert result.source_keys_for_component("missing") == ("missing",)
    assert result.bounds_for_component("missing") is None
    missing = build_field_display_regions(SimpleNamespace(), sources)
    assert not missing.regions and missing.bounds_for_component() is None
    assert "no captured resolved assembly" in " ".join(missing.notes)
    assert not missing.mask([[0., 0., 0.]]).any()


@pytest.mark.parametrize("points", [[[0., 0.]], [0., 0., 0.], [[np.nan, 0., 0.]], [[0., np.inf, 0.]]])
def test_invalid_query_points_are_rejected(points):
    with pytest.raises(ValueError, match="finite N by 3"):
        build_field_display_regions(SimpleNamespace(), ()).mask(points)


def test_duplicate_source_keys_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_field_display_regions(SimpleNamespace(), [_source("same"), _source("same")])


def test_real_captured_assembly_mapping_uses_current_host_and_preserves_settings():
    from temsim.optics.column import default_state
    state = default_state()
    before = deepcopy(state.to_dict())
    sources = [_source("objective_lens"), _source("mini_condenser"),
               _source("condenser_stigmator"), _source("probe_dph2_deflector"),
               _source("image_diffraction_deflector", "image_diffraction_deflector", "descan_deflector")]
    result = build_field_display_regions(state, sources)
    assert result.source_keys_for_component("objective_upper_pole") == ("objective_lens",)
    assert result.source_keys_for_component("objective_lower_pole") == ("objective_lens",)
    assert result.source_keys_for_component("mini_condenser_upper_pole") == ("mini_condenser",)
    assert result.source_keys_for_component("probe_dph2_coils") == ("probe_dph2_deflector",)
    assert result.source_keys_for_component("descan_deflector") == ("image_diffraction_deflector",)
    assert result.source_keys_for_component("objective_upper_electronics_sleeve") == ()
    assert result.regions_for_component("objective_upper_electronics_sleeve")
    assert not result.regions_for_component("probe_qph1_quadrupole")
    assert state.to_dict() == before
