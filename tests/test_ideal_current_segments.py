"""Closed electrical partitions: actual native fields, without emitting rays."""
from copy import deepcopy
import json

import numpy as np
import pytest

from temsim.ideal_coil_segments import (
    FIELD, SCHEMA, axial_recipe, bind_segments, normalize_recipe, set_current_segments,
)
from temsim.optics.column import default_state
from temsim.immutable_json import thaw_json
from temsim.physics.potential_capture import capture_round_lens_potential
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential


@pytest.fixture
def state(monkeypatch):
    from temsim.optics.electron_gun.field_emission import ColdFieldEmitter
    def forbidden(*args, **kwargs):
        raise AssertionError("Source emission is forbidden in electrical field tests")
    monkeypatch.setattr(ColdFieldEmitter, "emit", forbidden)
    result = default_state()
    result.simulation_mode = "unified"
    result.round_lens_field_model = "copper_coil"
    return result


def definition(state, key="condenser_lens_1"):
    part = next(part for part in state._resolved_assembly.parts
        if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
        and part.data.get("field_source_key") == key)
    counts = part.data.get("winding_interval_turn_counts", (part.data["winding_turn_count"],))
    return part.key, tuple(counts)


def test_uniform_partition_keeps_original_fields_and_mechanical_inputs(state):
    key, counts = definition(state)
    original = capture_round_lens_potential(state, "condenser_lens_1").potential
    parts = tuple((part.key, thaw_json(part.data)) for part in state._resolved_assembly.parts)
    controls = [(lens.key, lens.percent, lens.polarity) for lens in state.lenses]
    set_current_segments(state, "condenser_lens_1", axial_recipe(key, counts, 7))
    current = capture_round_lens_potential(state, "condenser_lens_1").potential
    assert len(current.coils) == len(original.coils)
    points = np.array([[0., 0., .5], [1e-5, -2e-5, .48], [-2e-4, 1e-4, .53]])
    for name, before in original.sample(points).items():
        np.testing.assert_array_equal(current.sample(points)[name], before)
    assert parts == tuple((part.key, thaw_json(part.data)) for part in state._resolved_assembly.parts)
    assert controls == [(lens.key, lens.percent, lens.polarity) for lens in state.lenses]
    assert current.digest != original.digest


def test_zero_net_ni_does_not_erase_separated_opposing_fields(state):
    key, counts = definition(state)
    assert counts[0] % 2 == 0
    recipe = axial_recipe(key, counts, 2, current_ratios=[1., -1.])
    set_current_segments(state, "condenser_lens_1", recipe)
    potential = capture_round_lens_potential(state, "condenser_lens_1").potential
    assert len(potential.coils) == 2
    assert sum(coil.ampere_turns for coil in potential.coils) == 0.
    centers = np.array([coil.center_z_m for coil in potential.coils])
    points = np.column_stack((np.zeros((2, 2)), centers))
    assert np.linalg.norm(potential.sample(points)["B_t"]) > 0.
    metadata = potential.provenance["internal_current_distribution"]
    assert metadata["segmented_physical_turn_count"] == counts[0]
    assert not metadata["resistance_model"] and not metadata["constant_power_constraint"]
    restored = CopperCoilAssemblyPotential.from_dict(json.loads(json.dumps(potential.to_dict())))
    np.testing.assert_array_equal(restored.sample(points)["B_t"], potential.sample(points)["B_t"])


def test_current_edit_invalidates_cached_potential_and_reset_recovers_original(state):
    key, counts = definition(state)
    original = capture_round_lens_potential(state, "condenser_lens_1").potential
    recipe = axial_recipe(key, counts, 2, current_ratios=[.3, -1.2])
    set_current_segments(state, "condenser_lens_1", recipe)
    first = capture_round_lens_potential(state, "condenser_lens_1").potential
    recipe["windings"][key][0]["current_ratio"] = .7
    assert state.lens_field_map_descriptors["condenser_lens_1"][FIELD]["windings"][key][0]["current_ratio"] == .3
    set_current_segments(state, "condenser_lens_1", recipe)
    second = capture_round_lens_potential(state, "condenser_lens_1").potential
    assert first.digest != second.digest
    point = np.array([[0., 0., first.coils[0].center_z_m]])
    assert not np.array_equal(first.sample(point)["B_t"], second.sample(point)["B_t"])
    set_current_segments(state, "condenser_lens_1", dict(schema=SCHEMA, windings={}))
    assert capture_round_lens_potential(state, "condenser_lens_1").potential.digest == original.digest


def test_objective_preserves_each_original_interval_turn_count(state):
    key, counts = definition(state, "objective_lens")
    assert len(counts) == 2
    recipe = axial_recipe(key, counts, 3, current_ratios=[1., -.5, .8, .9, -.2, .7])
    set_current_segments(state, "objective_lens", recipe)
    original_parts = state._resolved_assembly.parts
    potential = capture_round_lens_potential(state, "objective_lens").potential
    rows = potential.provenance["internal_current_distribution"]["cells"]
    assert [sum(row["turn_count"] for row in rows if row["interval_index"] == i)
            for i in range(2)] == list(counts)
    assert state._resolved_assembly.parts is original_parts
    bad = deepcopy(recipe)
    bad["windings"][key][0]["turn_count"] -= 1
    bad["windings"][key][3]["turn_count"] += 1
    with pytest.raises(ValueError, match="each original interval"):
        set_current_segments(state, "objective_lens", bad)


@pytest.mark.parametrize("kind", ["gap", "overlap", "outside", "turns", "density", "nonfinite", "boolean"])
def test_invalid_partitions_are_rejected_before_mutation(state, kind):
    key, counts = definition(state)
    recipe = axial_recipe(key, counts, 2)
    cells = recipe["windings"][key]
    if kind == "gap":cells[0]["axial_fraction"][1] -= .01
    elif kind == "overlap":cells[0]["axial_fraction"][1] += .01
    elif kind == "outside":cells[0]["radial_fraction"][0] = -.1
    elif kind == "turns":cells[0]["turn_count"] += 1
    elif kind == "density":
        cells[0]["axial_fraction"][1] = cells[1]["axial_fraction"][0] = .4
    elif kind == "nonfinite":cells[0]["current_ratio"] = float("inf")
    else:cells[0]["current_ratio"] = True
    before = deepcopy(state.lens_field_map_descriptors)
    with pytest.raises(ValueError):set_current_segments(state, "condenser_lens_1", recipe)
    assert state.lens_field_map_descriptors == before


def test_axial_partition_handles_thin_rational_end_cells():
    recipe = axial_recipe("original", (65535,), 1024)
    cells = bind_segments(recipe, "original", (65535,))
    assert sum(cell.turn_count for cell in cells) == 65535
    assert cells[-1].axial_fraction[1] == 1.


@pytest.mark.parametrize("mode", ["ideal_thin", "ideal", "custom", "linear_geometry"])
def test_electrical_editor_is_ideal_coil_mode_only(state, mode):
    key, counts = definition(state)
    state.simulation_mode = mode
    with pytest.raises(ValueError, match="only in Ideal"):
        set_current_segments(state, "condenser_lens_1", axial_recipe(key, counts, 2))


def test_gaussian_mode_does_not_enable_segment_editor(state):
    key, counts = definition(state)
    state.round_lens_field_model = "gaussian"
    with pytest.raises(ValueError, match="only in Ideal"):
        set_current_segments(state, "condenser_lens_1", axial_recipe(key, counts, 2))


def test_missing_winding_and_current_overflow_are_explicit(state):
    with pytest.raises(ValueError, match="absent"):
        set_current_segments(state, "condenser_lens_1", axial_recipe("invented", (300,), 2))
    key, counts = definition(state)
    with pytest.raises(ValueError, match="finite numerical range"):
        set_current_segments(state, "condenser_lens_1", axial_recipe(key, counts, 2, current_ratios=[1e308, 1.]))
    with pytest.raises(ValueError, match="finite real"):
        normalize_recipe(axial_recipe(key, counts, 1, current_ratios=[10**1000]))
