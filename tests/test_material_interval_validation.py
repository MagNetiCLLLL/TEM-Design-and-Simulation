"""Overlap checks consume physical material sections, not display envelopes."""

import tomllib
from dataclasses import replace
import json

import numpy as np
import pytest

from temsim.module_manifest import _validate_simple_magnetic_layer_geometry, validate_document, read_document
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import _part_mask


def test_disjoint_projector_material_sections_match_renderer_and_fem():
    source = INSTRUMENT_CONFIG_ROOT / "project_and_recording_system/EnergyFilter.toml"
    document = read_document(source)
    by_key = {part["key"]: part for part in document["parts"]}
    yoke = by_key["intermediate_lens_yoke"]
    coil = by_key["intermediate_lens_excitation_coil"]
    yoke["mechanical_inner_diameter_mm"] = 68.0
    yoke["material_intervals_mm"] = [[137.5, 162.0], [343.0, 367.5]]
    model = part_model_from_document(document, yoke["key"])
    mesh = model.meshes[0]
    triangle_z = mesh.vertices[mesh.faces, 2]
    assert np.all((triangle_z.max(axis=1) <= 162.0) | (triangle_z.min(axis=1) >= 343.0))
    z = np.linspace(130.0, 380.0, 2501) * 1e-3
    r = np.full_like(z, 0.04)
    def mask(part):
        return _part_mask(dict(key=part["key"], start_z_mm=part["local_start_z_mm"],
                               end_z_mm=part["local_end_z_mm"], data=part), r, z)
    assert not np.any(mask(yoke) & mask(coil))
    validate_document(document)
    yoke["material_intervals_mm"][0][1] = 163.0
    with pytest.raises(ValueError, match="Mechanical radial layers overlap"):
        validate_document(document)


def _parts():
    parent = dict(key="lens", mechanical_profile="magnetic_lens_assembly", local_start_z_mm=0.0)
    def layer(key, profile, start, end, intervals=None):
        result = dict(key=key, parent_key="lens", mechanical_profile=profile,
                      local_start_z_mm=start, local_center_z_mm=(start + end) / 2,
                      local_end_z_mm=end, length_mm=end - start,
                      mechanical_inner_diameter_mm=20, mechanical_outer_diameter_mm=30,
                      vacuum_inner_diameter_mm=10, material_class="test_material")
        if intervals is not None:
            result["material_intervals_mm"] = intervals
        return result
    return [parent, layer("coil", "magnetic_excitation_coil", 0, 10, [[0, 2], [8, 10]]),
            layer("housing", "magnetic_lens_housing", 3, 7)]


def test_parent_owned_split_intervals_take_precedence_over_child_intervals():
    parts = _parts()
    parts[0].update(upper_yoke_start_local_z_mm=0, upper_yoke_end_local_z_mm=2,
                    lower_yoke_start_local_z_mm=8, lower_yoke_end_local_z_mm=10)
    parts[1]["material_intervals_mm"] = [[0, 10]]
    _validate_simple_magnetic_layer_geometry(parts)
    # Removing the parent's geometry makes the explicit child sections active.
    del parts[0]["upper_yoke_start_local_z_mm"]
    with pytest.raises(ValueError, match="Mechanical radial layers overlap"):
        _validate_simple_magnetic_layer_geometry(parts)


@pytest.mark.parametrize("intervals", [[], [[2, 1]], [[1, 1]], [[0, float("nan")]],
                                        [[0, float("inf")]], [[0]], [[0, 1, 2]], [[False, 1]], "bad"])
def test_invalid_explicit_material_sections_are_rejected(intervals):
    parts = _parts()
    parts[1]["material_intervals_mm"] = intervals
    with pytest.raises(ValueError, match="material_intervals_mm"):
        _validate_simple_magnetic_layer_geometry(parts)


def _split_intermediate_coil():
    from temsim.optics.column import default_state
    state = default_state()
    document = read_document(INSTRUMENT_CONFIG_ROOT / "project_and_recording_system/NoEnergyFilter.toml")
    coil = next(p for p in document["parts"] if p["key"] == "intermediate_lens_excitation_coil")
    start, end = coil["local_start_z_mm"], coil["local_end_z_mm"]
    coil["material_intervals_mm"] = [[start + 1., start + 61.], [end - 61., end - 1.]]
    validate_document(document)
    assembly = state._resolved_assembly
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, "material_intervals_mm": coil["material_intervals_mm"]})
        if p.key == coil["key"] else p for p in assembly.parts))
    state.lens_field_map_descriptors["intermediate_lens"] = dict(
        solver="axisymmetric_linear_fem", relative_permeability=50., ampere_turns=100.,
        radial_nodes=12, axial_nodes=16, padding_factor=2.,
    )
    return state, document, coil


@pytest.mark.parametrize("translation_mm", [0., 137.])
def test_assembled_split_coil_intervals_follow_module_translation(translation_mm):
    from temsim.physics.lens_field_provider import lens_geometry_binding
    state, document, local = _split_intermediate_coil()
    assembly = state._resolved_assembly
    original = assembly.part(local["key"])
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, start_z_mm=p.start_z_mm + translation_mm,
                center_z_mm=p.center_z_mm + translation_mm, end_z_mm=p.end_z_mm + translation_mm)
        if p.module_key == original.module_key else p for p in assembly.parts))
    lens = next(lens for lens in state.lenses if lens.key == "intermediate_lens")
    lens.z_mm += translation_mm
    binding = lens_geometry_binding(state, lens.key)
    part = next(p for p in json.loads(binding.canonical_geometry_json)["lens_assembly"]["parts"]
                if p["key"] == local["key"])
    offset = original.start_z_mm - local["local_start_z_mm"] + translation_mm
    expected = np.asarray(local["material_intervals_mm"]) + offset
    np.testing.assert_allclose(part["data"]["material_intervals_mm"], expected, rtol=0, atol=1e-12)
    # Both material sections exist at the assembled position; the real gap is empty.
    centres = np.r_[expected.mean(axis=1), expected[:, 1].min() + 1.]
    radius = (local["mechanical_inner_diameter_mm"] + local["mechanical_outer_diameter_mm"]) * .25e-3
    np.testing.assert_array_equal(_part_mask(part, np.full(3, radius), centres * 1e-3), [True, True, False])
    # Building the FEM payload must not rewrite the saved/CAD local coordinates.
    assert state._resolved_assembly.part(local["key"]).data["material_intervals_mm"] == local["material_intervals_mm"]
    mesh = part_model_from_document(document, local["key"], include_children=False).meshes[0]
    assert mesh.vertices[:, 2].min() == pytest.approx(local["material_intervals_mm"][0][0])
    assert mesh.vertices[:, 2].max() == pytest.approx(local["material_intervals_mm"][-1][1])


def test_split_coil_produces_fem_field_used_by_transport():
    from copy import deepcopy
    from types import SimpleNamespace
    from temsim.physics.core import fields
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_map
    from temsim.physics.lens_field_provider import lens_geometry_binding, resolve_runtime_lens_field_provider
    state, _, local = _split_intermediate_coil()
    state.simulation_mode = "custom"
    for lens in state.lenses:
        lens.enabled = lens.key == "intermediate_lens"
    lens = next(lens for lens in state.lenses if lens.enabled)
    binding = lens_geometry_binding(state, lens.key)
    reference = deepcopy(json.loads(binding.canonical_geometry_json))
    part = next(p for p in reference["lens_assembly"]["parts"] if p["key"] == local["key"])
    offset = state._resolved_assembly.part(local["key"]).start_z_mm - local["local_start_z_mm"]
    part["data"]["material_intervals_mm"] = (np.asarray(local["material_intervals_mm"]) + offset).tolist()
    reference_map = solve_geometry_field_map(
        SimpleNamespace(canonical_geometry_json=json.dumps(reference, sort_keys=True)),
        state.lens_field_map_descriptors[lens.key],
    )
    provider = resolve_runtime_lens_field_provider(state, lens.key, lens)
    assert provider.model_status == "measured_or_fem_geometry_bound"
    for actual, expected in zip(provider.field_map.components_t, reference_map.components_t):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    assert max(np.abs(component).max() for component in provider.field_map.components_t) > 1e-3
    actual_bz = fields(np.array([lens.z_mm]), state)[0][0]
    expected_bz = provider.field_at_global_positions_t([0., 0., lens.z_mm * 1e-3])[2]
    assert abs(actual_bz) > 1e-5
    assert actual_bz == pytest.approx(expected_bz, rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("profile", ["magnetic_lens_yoke", "magnetic_excitation_coil"])
def test_objective_owned_intervals_are_global_once_and_override_child_sections(profile):
    from temsim.optics.column import default_state
    from temsim.magnetic_geometry import objective_layer_intervals_mm
    from temsim.physics.lens_field_provider import lens_geometry_binding
    state = default_state()
    assembly = state._resolved_assembly
    parent = assembly.part("objective_lens")
    child = next(p for p in assembly.parts if p.parent_key == parent.key and p.data.get("mechanical_profile") == profile)
    expected = objective_layer_intervals_mm(parent.data, parent.start_z_mm, profile)
    state._resolved_assembly = replace(assembly, parts=tuple(
        replace(p, data={**p.data, "material_intervals_mm": [[p.data["local_start_z_mm"], p.data["local_end_z_mm"]]]})
        if p.key == child.key else p for p in assembly.parts))
    geometry = json.loads(lens_geometry_binding(state, parent.key).canonical_geometry_json)
    encoded = next(p for p in geometry["lens_assembly"]["parts"] if p["key"] == child.key)
    np.testing.assert_allclose(encoded["data"]["material_intervals_mm"], expected, rtol=0, atol=1e-12)
