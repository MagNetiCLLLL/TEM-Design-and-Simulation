"""Preset selection and restoration checks without particle or wave propagation."""

from copy import deepcopy
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.operating_modes import (
    apply_condenser_mode,
    apply_operating_mode_pair,
    load_operating_mode_catalog,
    mode_by_key,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
from temsim.simulation_modes import switch_mode


FOCUSED_CONTROLS = {
    "condenser_lens_2": {"percent": 13.899040164772698},
    "condenser_lens_3": {"percent": 15.68823681182136},
    "objective_lens": {"percent": 68.97810600833917},
    "probe_hp2_hexapole": {"strength_m3": 4371028.449290636},
    "probe_hp1_hexapole": {
        "strength_m3": 2616320.416243901,
        "orientation_rad": -0.042012393991127016,
    },
}


@pytest.fixture
def state():
    state = default_state()
    assembly = AssemblyCatalog()
    assembly.apply(state, assembly.default_selection())
    return state


def assert_controls(state, expected):
    targets = runtime_targets(state)
    for key, values in expected.items():
        for field, value in values.items():
            runtime_field = "polarity" if field == "field_polarity" else field
            assert getattr(targets[key].obj, runtime_field) == value, (key, field)


def test_unified_normal_uses_focused_controls_with_an_independent_qualification_record(state):
    switch_mode(state, "unified")
    source = deepcopy(state.electron_gun.to_dict())
    step = state.step_mm
    result = apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert_controls(state, FOCUSED_CONTROLS)
    assert state.mini_condenser.percent == 43.88646617222637
    assert state.mini_condenser.polarity == -1
    assert state.electron_gun.to_dict() == source
    assert state.step_mm == step
    targets = result.condenser.targets
    assert targets["calibration_model"] == "unified"
    assert targets["target_convergence_sem_angle_mrad"] == 10.0
    assert targets["calibration_ray_count"] == 49
    assert targets["calibration_step_mm"] == 0.05
    assert targets["validation_step_mm"] == 0.025
    assert targets["sampling_convergence"] == "NOT_TESTED"
    assert targets["topology_qualification"] == "NOT_TESTED"
    assert "condenser_topology" not in targets
    assert "corrected_nonlinear_error_nm" not in targets
    assert "0.05 and 0.025 mm" in result.condenser.calibration_reference


@pytest.mark.parametrize("model", ["custom", "ideal", "analytical", "ideal_thin"])
def test_legacy_models_keep_their_stored_normal_preset(state, model):
    switch_mode(state, model)
    result = apply_condenser_mode(state, "nano_probe")
    legacy = mode_by_key("nano_probe", load_operating_mode_catalog())
    assert result.condenser == legacy
    assert_controls(state, legacy.devices)
    assert legacy.devices["condenser_lens_2"]["percent"] == 23.225625142668733
    assert legacy.targets["target_convergence_sem_angle_mrad"] == 30.0


def test_normal_large_angle_normal_and_repeated_application_restore_the_focused_preset(state):
    switch_mode(state, "unified")
    apply_condenser_mode(state, "nano_probe")
    normal = deepcopy(state.to_dict())
    apply_condenser_mode(state, "nano_probe_large_angle")
    assert_controls(state, {"condenser_lens_3": {"percent": 46.059815218359184}})
    apply_condenser_mode(state, "nano_probe")
    assert state.to_dict() == normal
    apply_condenser_mode(state, "nano_probe")
    assert state.to_dict() == normal
    assert_controls(state, FOCUSED_CONTROLS)


def test_finite_field_preset_keeps_a_manually_selected_cm_magnitude(state):
    switch_mode(state, "unified")
    state.mini_condenser.percent = 37.25
    apply_condenser_mode(state, "nano_probe")
    assert state.mini_condenser.percent == 37.25
    assert state.mini_condenser.polarity == -1
    assert_controls(state, FOCUSED_CONTROLS)


def test_explicit_catalog_wins_over_the_field_model(state):
    switch_mode(state, "unified")
    legacy = load_operating_mode_catalog()
    original = mode_by_key("nano_probe", legacy)
    devices = deepcopy(original.devices)
    devices["condenser_lens_2"]["percent"] = 17.5
    custom = replace(original, devices=devices)
    explicit = replace(legacy, modes=tuple(
        custom if item.key == "nano_probe" else item for item in legacy.modes
    ))
    result = apply_condenser_mode(state, "nano_probe", catalog=explicit)
    assert result.condenser is custom
    assert_controls(state, devices)
    assert mode_by_key("nano_probe", legacy).devices["condenser_lens_2"]["percent"] == 23.225625142668733


def test_model_switch_does_not_apply_a_preset_or_restore_shelved_currents(state):
    apply_condenser_mode(state, "nano_probe")
    runtime_targets(state)["condenser_lens_2"].obj.percent = 17.5
    before = [(lens.key, lens.percent, lens.polarity) for lens in state.lenses]
    correctors = deepcopy(state.to_dict()["corrector_elements"])
    switch_mode(state, "unified")
    assert [(lens.key, lens.percent, lens.polarity) for lens in state.lenses] == before
    assert state.to_dict()["corrector_elements"] == correctors
    apply_condenser_mode(state, "nano_probe")
    assert_controls(state, FOCUSED_CONTROLS)


def test_failed_projector_application_rolls_back_the_resolved_finite_field_preset(state, monkeypatch):
    from temsim import operating_modes

    switch_mode(state, "unified")
    before = deepcopy(state.to_dict())
    apply_values = operating_modes._apply_values

    def reject_projector(candidate, definition):
        if definition.family == "projector":
            assert_controls(candidate, FOCUSED_CONTROLS)
            raise ValueError("Synthetic projector failure")
        return apply_values(candidate, definition)

    monkeypatch.setattr(operating_modes, "_apply_values", reject_projector)
    with pytest.raises(ValueError, match="Synthetic projector failure"):
        apply_operating_mode_pair(state, "nano_probe", "diffraction")
    assert state.to_dict() == before
