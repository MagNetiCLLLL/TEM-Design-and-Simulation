"""Assembly roll, transverse field error and diameter tilt have distinct actions."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.lens_pose import (
    effective_part_transform_mm, mechanical_part_transform_mm, physical_pose_values,
    lens_pose_registration, mechanical_pose_registration, rotation_matrix_mrad,
)
from temsim.module_manifest import read_document
from temsim.optics.column import default_state


def document():
    return read_document(Path(__file__).resolve().parents[1] /
                         "configs/instruments/column/C3_ProbeCorrector.toml")


def rz(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)))


def test_field_error_preserves_tilted_axis_and_material_at_nonzero_mount_angle():
    part = dict(key="probe_qpol_quadrupole", local_center_z_mm=10.,
                assembly_azimuth_deg=37., rotation_x_mrad=30., rotation_y_mrad=-20.,
                rotation_z_mrad=11., offset_x_mm=.2)
    rows = {part["key"]: part}
    original_r, original_t = effective_part_transform_mm(part, rows)
    material = mechanical_part_transform_mm(part, rows)
    part["field_azimuth_error_deg"] = -13.
    actual_r, actual_t = effective_part_transform_mm(part, rows)
    np.testing.assert_allclose(actual_r, original_r @ rz(-13.), atol=2e-15)
    np.testing.assert_array_equal(actual_r[:, 2], original_r[:, 2])
    np.testing.assert_array_equal(actual_t, original_t)
    for actual, expected in zip(mechanical_part_transform_mm(part, rows), material):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(material[0], rz(37.), atol=2e-15)
    old = {k: v for k, v in part.items() if k not in {"assembly_azimuth_deg", "field_azimuth_error_deg"}}
    old_r, _ = effective_part_transform_mm(old, {old["key"]: old})
    np.testing.assert_array_equal(old_r, rotation_matrix_mrad((30., -20., 11.)))


def test_parent_field_error_keeps_child_field_position_and_direction_independent():
    parent = dict(key="probe_hp1_hexapole", local_center_z_mm=10., assembly_azimuth_deg=23.)
    child = dict(key="probe_dph1_coils", parent_key=parent["key"], local_center_z_mm=11.,
                 mechanical_part_role="deflector_coil_insert", offset_x_mm=.2,
                 field_azimuth_error_deg=7.)
    rows = {p["key"]: p for p in (parent, child)}
    original = effective_part_transform_mm(child, rows)
    parent["field_azimuth_error_deg"] = 31.
    for actual, expected in zip(effective_part_transform_mm(child, rows), original):
        np.testing.assert_array_equal(actual, expected)


def test_assembly_azimuth_turns_tilted_field_about_column_z_at_fixed_center():
    part = dict(key="probe_qpol_quadrupole", local_center_z_mm=10.,
                rotation_x_mrad=30., rotation_y_mrad=-20., rotation_z_mrad=11.,
                offset_x_mm=.2, offset_y_mm=-.3, field_azimuth_error_deg=7.)
    rows = {part["key"]: part}
    initial_r, initial_t = effective_part_transform_mm(part, rows)
    part["assembly_azimuth_deg"] = 37.
    actual_r, actual_t = effective_part_transform_mm(part, rows)
    np.testing.assert_allclose(actual_r, rz(37.) @ initial_r, atol=2e-15)
    center = np.array((.2, -.3, 10.))
    points = np.array(((.1, .2, 10.), (0., 0., 11.)))
    before = points @ initial_r.T + initial_t
    np.testing.assert_allclose(points @ actual_r.T + actual_t,
                               (before-center) @ rz(37.).T + center, atol=2e-15)
    np.testing.assert_allclose(actual_r @ (0., 0., 10.) + actual_t, center, atol=2e-15)


@pytest.mark.parametrize("key", ["probe_hp1_hexapole", "probe_dph1_coils", "ac_deflector", "objective_stigmator"])
def test_assembly_azimuth_rotates_actual_cad_windings_and_edges(key):
    from temsim.part_model_3d import part_model_from_document
    doc = document()
    rows = {p["key"]: p for p in doc["parts"]}
    before = part_model_from_document(doc, key, angular_segments=8)
    rows[key]["assembly_azimuth_deg"] = 31.
    after = part_model_from_document(doc, key, angular_segments=8)
    assert before.meshes and len(after.meshes) == len(before.meshes)
    for base, moved in zip(before.meshes, after.meshes):
        np.testing.assert_allclose(moved.vertices, base.vertices @ rz(31.).T, atol=2e-11)
        for base_edge, moved_edge in zip(base.edges, moved.edges):
            np.testing.assert_allclose(moved_edge["vertices"], base_edge["vertices"] @ rz(31.).T, atol=2e-11)
    rows[key]["field_azimuth_error_deg"] = 9.
    error = part_model_from_document(doc, key, angular_segments=8)
    for mounted, field_error in zip(after.meshes, error.meshes):
        np.testing.assert_array_equal(mounted.vertices, field_error.vertices)


@pytest.fixture(scope="module")
def native():
    return default_state()


@pytest.mark.parametrize("key", ["probe_hp1_hexapole", "probe_qpol_quadrupole", "probe_dph1_deflector",
                                "ac_deflector", "objective_stigmator"])
def test_field_vector_rotates_with_mount_and_error_once(native, key):
    from test_posed_column_fields import component_state
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    state, component = component_state(native, key)
    base = capture_posed_column_fields(state, include_aligned=True)
    host = native._resolved_assembly.part(key).data.get("physical_host_key", key)
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**p.data, "assembly_azimuth_deg": 23., "field_azimuth_error_deg": 7.})
        if p.key == host else p for p in state._resolved_assembly.parts))
    rotated = capture_posed_column_fields(state, include_aligned=True)
    assert base and len(base) == len(rotated)
    for original, actual in zip(base, rotated):
        points = np.array(((1e-6, 2e-6, original.center_z_m), (-2e-6, 1e-6, original.center_z_m)))
        np.testing.assert_allclose(actual.field_at_global_positions_t(points @ rz(30.).T),
                                   original.field_at_global_positions_t(points) @ rz(30.).T,
                                   rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(mechanical_pose_registration(state, key).rotation_array, rz(23.), atol=2e-15)


def test_rotated_rectangular_aperture_uses_same_ray_and_wave_opening():
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wave import _wave_apertures
    aperture = SimpleNamespace(key="objective_aperture", z_mm=10., enabled=True,
        transmission_mask=lambda x, y: (abs(x) <= .3) & (abs(y) <= .1))
    row = dict(key=aperture.key, local_center_z_mm=10., assembly_azimuth_deg=90.)
    state = SimpleNamespace(apertures=[aperture], _resolved_assembly=SimpleNamespace(parts=(
        SimpleNamespace(key=aperture.key, module_key="column", data=row, center_z_mm=10.),)))
    x, y = np.array([[.2, 0.], [.2, 0.]])*1e-3, np.array([[0., .2], [0., .2]])*1e-3
    alive, _, _ = clip_segment(state, [9., 11.], x, y)
    assert alive.tolist() == [False, True]
    wave, = _wave_apertures(state, 9., 11.)
    np.testing.assert_array_equal(wave.transmission_mask(x[-1]*1e3, y[-1]*1e3), alive)


def test_bcor_third_hexapole_azimuth_reaches_actual_field_capture():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    from test_posed_column_fields import component_state
    installed = default_state()
    AssemblyCatalog().apply(installed, AssemblySelection("FEG", "C3 + B-COR", "No Energy Filter"))
    key = "image_hp3_hexapole"
    state, component = component_state(installed, key)
    original, = capture_posed_column_fields(state, include_aligned=True)
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**p.data, "assembly_azimuth_deg": 23., "field_azimuth_error_deg": 7.})
        if p.key == key else p for p in state._resolved_assembly.parts))
    actual, = capture_posed_column_fields(state)
    points = np.array(((1e-6, 2e-6, original.center_z_m), (-2e-6, 1e-6, original.center_z_m)))
    np.testing.assert_allclose(actual.field_at_global_positions_t(points @ rz(30.).T),
                               original.field_at_global_positions_t(points) @ rz(30.).T,
                               rtol=2e-12, atol=1e-18)
    plan = build_propagation_plan(state, component.z_mm-.05, component.z_mm+.05, ())
    assert any(field.lens_key == key for field in plan.mapped_fields)
    np.testing.assert_array_equal(plan.hex_normal_m3, 0.)
    np.testing.assert_array_equal(plan.hex_skew_m3, 0.)
    component.hexapole_strength_components_m3 = lambda z: (0., 0.)
    with pytest.raises(ValueError, match="declared native Gaussian"):
        capture_posed_column_fields(state)


def test_azimuth_editor_units_ownership_invalid_values_and_undo():
    from temsim.part_model_document import PartModelDocument
    from temsim.part_model_3d import part_dimension_specs
    path = Path(__file__).resolve().parents[1] / "configs/instruments/column/C3_ProbeCorrector.toml"
    draft = PartModelDocument(path)
    for key in ("probe_hp1_hexapole", "probe_qpol_quadrupole", "ac_deflector", "objective_stigmator"):
        fields = {s.path[-1]: s for s in part_dimension_specs(draft.document, key)}
        assert fields["assembly_azimuth_deg"].unit == fields["field_azimuth_error_deg"].unit == "deg"
        assert fields["rotation_x_mrad"].unit == "mrad"
        assert "rotation_z_mrad" not in fields
    fields = part_dimension_specs(draft.document, "probe_dph1_deflector")
    assert any(s.path == ("parts", "probe_dph1_coils", "assembly_azimuth_deg") and s.editable for s in fields)
    draft.set_physical_pose("probe_dph1_coils", {"assembly_azimuth_deg": 12., "field_azimuth_error_deg": -.3})
    assert physical_pose_values(draft.part("probe_dph1_coils"))["assembly_azimuth_deg"] == 12.
    saved = deepcopy(draft.document)
    with pytest.raises(ValueError):
        draft.set_physical_pose("probe_dph1_coils", {"assembly_azimuth_deg": float("nan")})
    assert draft.document == saved
    with pytest.raises(ValueError, match="physical host"):
        draft.set_physical_pose("probe_dph1_deflector", {"assembly_azimuth_deg": 1.})
    draft.undo()
    assert physical_pose_values(draft.part("probe_dph1_coils"))["assembly_azimuth_deg"] == 0.


def test_azimuth_snapshot_and_cache_preserve_independent_angles(native):
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    from temsim.calculation_cache import calculation_signatures
    state = decode_instrument(encode_instrument(native))
    before = calculation_signatures(state)
    key = "probe_hp1_hexapole"
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**p.data, "assembly_azimuth_deg": -17., "field_azimuth_error_deg": .6})
        if p.key == key else p for p in state._resolved_assembly.parts))
    assert calculation_signatures(state)["incident"] != before["incident"]
    restored = decode_instrument(encode_instrument(state))
    assert lens_pose_registration(restored, key) == lens_pose_registration(state, key)
    assert mechanical_pose_registration(restored, key) == mechanical_pose_registration(state, key)
