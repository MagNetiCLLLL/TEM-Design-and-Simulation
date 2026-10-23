"""State-level regression for the optional empirical objective chromatic kick."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.chromatic import objective_chromatic_kick_from_state


_MISSING = object()


def _state(cc_mm):
    objective = SimpleNamespace()
    if cc_mm is not _MISSING:
        objective.cc_mm = cc_mm
    return SimpleNamespace(
        objective_lens=objective,
        beam_voltage_kv=300.0,
        chromatic_aberration_enabled=True,
        simulation_mode="ideal",
        lens_field_map_descriptors={},
    )


@pytest.mark.parametrize(
    ("configured_cc_mm", "effective_cc_mm"),
    [(0.0, 0.0), (None, 2.0), (_MISSING, 2.0), (1.7, 1.7)],
    ids=["explicit-zero", "legacy-none", "legacy-missing", "nonzero"],
)
def test_state_kick_respects_zero_and_legacy_defaults_without_mutation(
    configured_cc_mm, effective_cc_mm
):
    state = _state(configured_cc_mm)
    x = np.array([1e-6, -2e-6, 3e-6])
    y = np.array([-4e-6, 5e-6, 6e-6])
    energy_offset = np.array([10.0, -10.0, 0.0])
    original = [value.copy() for value in (x, y, energy_offset)]
    for value in (x, y, energy_offset):
        value.setflags(write=False)
    objective_before = vars(state.objective_lens).copy()

    actual_x, actual_y = objective_chromatic_kick_from_state(
        state, x, y, energy_offset, resolved_focal_mm=2.5
    )

    # Independently express the declared thin-lens perturbation in SI units.
    delta_f_m = effective_cc_mm * 1e-3 * energy_offset / 300000.0
    np.testing.assert_allclose(actual_x, x * delta_f_m / (2.5e-3) ** 2,
                               rtol=1e-14, atol=0.0)
    np.testing.assert_allclose(actual_y, y * delta_f_m / (2.5e-3) ** 2,
                               rtol=1e-14, atol=0.0)
    for value, before in zip((x, y, energy_offset), original):
        np.testing.assert_array_equal(value, before)
    assert vars(state.objective_lens) == objective_before
    assert state.lens_field_map_descriptors == {}
    assert state.chromatic_aberration_enabled


def test_real_instrument_state_zero_cc_is_zero_through_focal_resolution(monkeypatch):
    from temsim.optics.column import default_state

    state = default_state()
    state.simulation_mode = "ideal"
    state.chromatic_aberration_enabled = True
    state.objective_lens.cc_mm = 0.0
    calls = []

    def focal_length(lens, voltage):
        calls.append((lens, voltage))
        return 2.5

    monkeypatch.setattr("temsim.optics.lens_focal_length.focal_length_mm", focal_length)
    result = objective_chromatic_kick_from_state(
        state, np.array([1e-6]), np.array([2e-6]), np.array([10.0])
    )

    np.testing.assert_array_equal(result, np.zeros((2, 1)))
    assert calls == [(state.objective_lens, state.beam_voltage_kv)]
    assert state.objective_lens.cc_mm == 0.0


@pytest.mark.parametrize(
    ("enabled", "mode", "recipe"),
    [
        (False, "ideal", {}),
        (True, "linear_geometry", {"solver": "axisymmetric_linear_fem"}),
        (True, "nonlinear_material", {"solver": "axisymmetric_nonlinear_fem"}),
        (True, "custom", {"solver": "axisymmetric_linear_fem"}),
        (True, "custom", {"solver": "axisymmetric_nonlinear_fem"}),
    ],
    ids=["disabled", "linear-mode", "nonlinear-mode", "custom-linear", "custom-nonlinear"],
)
def test_disabled_or_generated_field_routes_do_not_add_empirical_kick(
    monkeypatch, enabled, mode, recipe
):
    state = _state(1.7)
    state.chromatic_aberration_enabled = enabled
    state.simulation_mode = mode
    state.lens_field_map_descriptors = {"objective_lens": recipe}

    def unexpected_focal(*args):
        pytest.fail("A disabled or generated-FEM chromatic kick must not resolve a focal length")

    monkeypatch.setattr("temsim.optics.lens_focal_length.focal_length_mm", unexpected_focal)
    result = objective_chromatic_kick_from_state(
        state, np.array([1e-6]), np.array([2e-6]), np.array([10.0])
    )

    np.testing.assert_array_equal(result, np.zeros((2, 1)))
    assert state.objective_lens.cc_mm == 1.7
    assert state.lens_field_map_descriptors == {"objective_lens": recipe}


def test_explicitly_disabled_resolved_focal_retains_zero_kick():
    result = objective_chromatic_kick_from_state(
        _state(1.7), np.array([1e-6]), np.array([2e-6]), np.array([10.0]),
        resolved_focal_mm=None,
    )
    np.testing.assert_array_equal(result, np.zeros((2, 1)))
