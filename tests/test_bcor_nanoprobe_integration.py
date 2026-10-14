"""Merged B-COR/range contracts; synthetic topology is not optical qualification."""
from copy import deepcopy
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest
import tomli_w

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.component_keys import BCOR_KEYS
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import compatible_modes, condenser_mode_key, mode_by_key
from temsim.optics.column import default_state
from temsim.optics.condenser_topology import (
    LARGE_ANGLE_REQUIREMENT, REQUIREMENT, evaluate, is_applicable,
    requested_capture_planes, requirement_for_state,
)
from temsim.optics.model import State
from temsim.profile_io import apply_profile_values, read_profile, save_profile


RANGES = (("normal", "nano_probe", REQUIREMENT),
          ("large_angle", "nano_probe_large_angle", LARGE_ANGLE_REQUIREMENT))


def bcor_state(probe, nanoprobe_range):
    catalog, state = AssemblyCatalog(), default_state()
    selection = AssemblySelection(
        "FEG", "C3 + Probe Corrector + B-COR" if probe else "C3 + B-COR",
        "No Energy Filter")
    catalog.apply(state, selection)
    state.illumination_mode = "STEM"
    state.nanoprobe_range = nanoprobe_range
    state.mini_condenser.signed_excitation_percent = -51.25
    next(lens for lens in state.lenses if lens.key == "condenser_lens_3").percent = 37.125
    state.image_corrector_system.tl31_lens.percent = 12.345
    state.image_corrector_system.tl32_lens.percent = 45.625
    state.image_corrector_system.tl32_lens.enabled = False
    state.image_corrector_system.hp3_hexapole.strength_m3 = 1234.5
    state.image_corrector_system.hp3_hexapole.orientation_rad = .234
    return state, selection


def assert_joint_state(state, selection, probe, nanoprobe_range):
    _, key, requirement = next(row for row in RANGES if row[0] == nanoprobe_range)
    assert state.image_corrector_design == "bcor"
    assert state.image_corrector_installed
    assert state.probe_corrector_installed is probe
    assert AssemblyCatalog().selection_for_resolved(state._resolved_assembly) == selection
    assert tuple(component.key for component in state.image_corrector_system.components) == BCOR_KEYS
    assert state.nanoprobe_range == nanoprobe_range
    assert condenser_mode_key(state) == key
    assert requirement_for_state(state) == requirement
    assert is_applicable(state) is probe
    assert state.mini_condenser.signed_excitation_percent == -51.25
    assert next(lens for lens in state.lenses if lens.key == "condenser_lens_3").percent == 37.125
    assert state.image_corrector_system.tl31_lens.percent == 12.345
    assert state.image_corrector_system.tl32_lens.percent == 45.625
    assert not state.image_corrector_system.tl32_lens.enabled
    assert state.image_corrector_system.hp3_hexapole.strength_m3 == 1234.5
    assert state.image_corrector_system.hp3_hexapole.orientation_rad == .234


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("nanoprobe_range,key,requirement", RANGES)
def test_bcor_and_range_round_trip_together_without_reapplying_presets(
    tmp_path, monkeypatch, probe, nanoprobe_range, key, requirement,
):
    import temsim.operating_modes as modes

    state, selection = bcor_state(probe, nanoprobe_range)
    before = state.to_dict()
    assert mode_by_key(key).targets["condenser_topology"] == requirement
    monkeypatch.setattr(modes, "apply_operating_mode_pair",
                        lambda *_a, **_k: pytest.fail("Restoration must not apply a preset"))
    state_copy = State.from_dict(deepcopy(before))
    snapshot_copy = capture_instrument_snapshot(state).restore()
    path = tmp_path / "bcor-range.toml"
    save_profile(path, state, selection)
    saved, values = read_profile(path)
    profile_copy = default_state()
    AssemblyCatalog().apply(profile_copy, saved)
    apply_profile_values(profile_copy, values)
    for restored in (state_copy, snapshot_copy, profile_copy):
        assert_joint_state(restored, selection, probe, nanoprobe_range)
        payload = restored.to_dict()
        for field in ("lenses", "corrector_elements", "electron_gun", "stigmators"):
            assert payload[field] == before[field], field
    assert state.to_dict() == before


@pytest.mark.parametrize("probe", [False, True])
def test_previous_state_and_complete_profile_default_range_without_losing_bcor(tmp_path, probe):
    state, selection = bcor_state(probe, "large_angle")
    previous = state.to_dict()
    previous["schema_version"] = 79
    del previous["nanoprobe_range"]
    original = deepcopy(previous)
    restored = State.from_dict(previous)
    assert_joint_state(restored, selection, probe, "normal")
    assert previous == original

    path = tmp_path / "previous-bcor.toml"
    save_profile(path, state, selection)
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    del document["devices"]["simulation"]["nanoprobe_range"]
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    saved, values = read_profile(path)
    assert saved == selection
    apply_profile_values(state, values)
    assert_joint_state(state, selection, probe, "normal")
    state.nanoprobe_range = "large_angle"
    apply_profile_values(state, {"simulation": {"step_mm": .125}})
    assert_joint_state(state, selection, probe, "large_angle")


@pytest.mark.parametrize("probe", [False, True])
def test_invalid_range_does_not_partially_change_bcor_controls(probe):
    state, selection = bcor_state(probe, "large_angle")
    before = state.to_dict()
    damaged = deepcopy(before)
    damaged["nanoprobe_range"] = "unrecognized"
    with pytest.raises(ValueError, match="Nanoprobe range"):
        State.from_dict(damaged)
    with pytest.raises(ValueError):
        apply_profile_values(state, {
            "image_hp3_hexapole": {"strength_m3": 9876.5},
            "simulation": {"nanoprobe_range": "unrecognized"},
        })
    assert state.to_dict() == before
    assert_joint_state(state, selection, probe, "large_angle")


@pytest.mark.parametrize("probe", [False, True])
def test_bcor_condenser_compatibility_requires_probe_hardware_and_keeps_projector_boundary(probe):
    state, selection = bcor_state(probe, "large_angle")
    condenser = {mode.key for mode in compatible_modes(
        "condenser", selection.column, selection.recording)}
    nanoprobe = {"nano_probe", "nano_probe_large_angle"}
    assert nanoprobe.intersection(condenser) == (nanoprobe if probe else set())
    assert not compatible_modes("projector", selection.column, selection.recording)
    assert is_applicable(state) is probe
    installed = {part.key for part in state._resolved_assembly.parts}
    assert ("adapter_lens" in installed) is probe
    assert "image_adapter_lens" in installed


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("nanoprobe_range,key,requirement", RANGES)
def test_condenser_only_mode_uses_c3_range_without_changing_bcor_or_projectors(
    probe, nanoprobe_range, key, requirement,
):
    from temsim.operating_modes import apply_condenser_mode

    state, selection = bcor_state(probe, "normal")
    before = state.to_dict()
    arguments = dict(column_name=selection.column, recording_name=selection.recording)
    if not probe:
        with pytest.raises(ValueError, match="not calibrated|not compatible"):
            apply_condenser_mode(state, key, **arguments)
        assert state.to_dict() == before
        return
    result = apply_condenser_mode(state, key, **arguments)
    assert result.condenser.key == condenser_mode_key(state) == key
    assert result.projector is None
    assert state.nanoprobe_range == nanoprobe_range
    assert requirement_for_state(state) == requirement
    assert state.image_corrector_design == "bcor"
    assert state.mini_condenser.signed_excitation_percent == -51.25
    controls = {lens.key: lens for lens in state.lenses}
    for lens_key in ("condenser_lens_3", "adapter_lens"):
        assert controls[lens_key].percent == mode_by_key(key).devices[lens_key]["percent"]
    after = state.to_dict()
    assert after["projector_mode"] == before["projector_mode"]
    preserved = {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}
    for field in ("lenses", "corrector_elements"):
        def unchanged_controls(payload):
            return [item for item in payload[field]
                    if item["key"].startswith("image_") or item["key"] in preserved]
        assert unchanged_controls(after) == unchanged_controls(before)


def synthetic_checkpoints(state, second_waist):
    """Analytical two-dimensional bundles at the assembled component positions."""
    positions = {lens.key: lens.z_mm for lens in state.lenses}
    first = (positions["condenser_lens_2"] + positions["condenser_lens_3"]) / 2
    second = (positions["condenser_lens_3"] + positions["adapter_lens"]) / 2
    z = np.unique(np.r_[requested_capture_planes(state), first, second])
    phi = np.arange(16) * 2 * np.pi / 16
    radius = (z-first) * 1e-6
    slope = np.full(z.size, .001)
    if second_waist:
        radius *= z-second
        slope = (2*z-first-second) * .001
    checkpoints = SimpleNamespace(
        z_mm=z,
        x_m=radius[:, None]*np.cos(phi) + 1e-8*np.cos(2*phi),
        y_m=radius[:, None]*np.sin(phi) + 1e-8*np.sin(2*phi),
        tx_rad=slope[:, None]*np.cos(phi), ty_rad=slope[:, None]*np.sin(phi))
    return checkpoints, np.ones(checkpoints.x_m.shape, bool), np.ones(16)


@pytest.mark.parametrize("nanoprobe_range,second,passed,failure", [
    ("normal", False, True, None),
    ("normal", True, False, "forbidden_c3_adl_waist"),
    ("large_angle", False, False, "missing_c3_adl_waist"),
    ("large_angle", True, True, None),
])
def test_bcor_assembly_uses_remote_c3_crossover_policy(
    nanoprobe_range, second, passed, failure,
):
    state, _ = bcor_state(True, nanoprobe_range)
    before = capture_instrument_snapshot(state).digest
    checkpoints, masks, weights = synthetic_checkpoints(state, second)
    report = evaluate(state, checkpoints, masks, weights)
    assert report["passed"] is passed, report
    assert report["failures"] == ([] if passed else [failure])
    assert report["requirement"] == requirement_for_state(state)
    assert report["nanoprobe_range"] == nanoprobe_range
    assert report["intervals"]["c2_c3"]["requirement"] == "at_least_one_executed_2d_waist"
    assert report["intervals"]["c3_adl"]["requirement"] == (
        "at_least_one_executed_2d_waist" if nanoprobe_range == "large_angle"
        else "no_converging_to_diverging_waist")
    assert capture_instrument_snapshot(state).digest == before
