"""Analytical phase-space control checks, separate from any host relocation."""
from __future__ import annotations

import numpy as np
import pytest

from temsim.physics.descan_phase_space import solve_position_angle_cancellation


def drift_responses(upper_distance=.12, lower_distance=.08):
    identity = np.eye(2)
    return np.vstack((upper_distance*identity, identity)), np.vstack((lower_distance*identity, identity))


def solve(upper, lower, incident, **kwargs):
    return solve_position_angle_cancellation(upper, lower, incident,
        separation_mm=kwargs.pop("separation_mm", 40.), upper_gain=kwargs.pop("upper_gain", .5), **kwargs)


def test_two_drift_coils_match_analytic_position_and_angle_cancellation():
    upper, lower = drift_responses()
    position = np.array([[2e-7, .4e-7], [-.2e-7, 1.5e-7]])
    angle = np.array([[1e-6, -.2e-6], [.3e-6, -.6e-6]])
    incident = np.vstack((position, angle))
    saved = [value.copy() for value in (upper, lower, incident)]
    result = solve(upper, lower, incident)
    expected_upper = (-position+.08*angle)/(.12-.08)
    expected_lower = -angle-expected_upper
    np.testing.assert_allclose(result.upper_kick_matrix_mrad, expected_upper*1e3, rtol=2e-15, atol=1e-18)
    np.testing.assert_allclose(result.lower_kick_matrix_mrad, expected_lower*1e3, rtol=2e-15, atol=1e-18)
    np.testing.assert_allclose(.5*result.command_matrix_mrad, result.upper_kick_matrix_mrad, rtol=0, atol=0)
    np.testing.assert_allclose(result.lower_from_upper @ result.upper_kick_matrix_mrad,
                               result.lower_kick_matrix_mrad, rtol=2e-15, atol=1e-18)
    assert result.position_residual_m < 1e-21
    assert result.angle_residual_rad < 1e-20
    assert result.position_relative_residual < 1e-14
    assert result.angle_relative_residual < 1e-14
    for original, before in zip((upper, lower, incident), saved):
        np.testing.assert_array_equal(original, before)
    assert not result.command_matrix_mrad.flags.writeable


def test_zero_incident_angle_has_finite_relative_diagnostic():
    upper, lower = drift_responses()
    result = solve(upper, lower, np.vstack((np.diag([1e-8, 3e-8]), np.zeros((2, 2)))))
    np.testing.assert_allclose(result.lower_kick_matrix_mrad, -result.upper_kick_matrix_mrad, atol=1e-18)
    assert np.isfinite(result.angle_relative_residual)
    assert result.angle_relative_residual < 1e-14
    assert result.angle_residual_rad < 1e-20


def test_coupled_rotated_lens_responses_recover_independent_physical_kicks(monkeypatch):
    upper, lower = drift_responses()
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    quadrupole = np.block([[np.eye(2), np.zeros((2, 2))],
                          [np.array([[2., .4], [.4, -1.]]), np.eye(2)]])
    downstream = np.block([[rotation, .23*rotation], [np.zeros((2, 2)), rotation]]) @ quadrupole
    upper = downstream @ upper @ rotation
    lower = downstream @ lower @ rotation.T
    expected_upper = np.array([[2e-6, -.6e-6], [.3e-6, 1.7e-6]])
    expected_lower = np.array([[-3e-6, .5e-6], [1.2e-6, -.8e-6]])
    incident = -(upper @ expected_upper + lower @ expected_lower)
    monkeypatch.setattr(np.linalg, "inv", lambda *_args: pytest.fail("The solve must not form an inverse"))
    result = solve(upper, lower, incident, upper_gain=-.7)
    np.testing.assert_allclose(result.upper_kick_matrix_mrad, expected_upper*1e3, rtol=2e-14, atol=1e-17)
    np.testing.assert_allclose(result.lower_kick_matrix_mrad, expected_lower*1e3, rtol=2e-14, atol=1e-17)
    assert result.position_relative_residual < 1e-14
    assert result.angle_relative_residual < 1e-14


def test_position_row_scaling_gives_dimensionally_consistent_conditioning():
    upper, lower = drift_responses()
    incident = np.vstack((np.diag([1e-7, 2e-7]), np.diag([1e-6, 3e-6])))
    first = solve(upper, lower, incident)
    for value in (upper, lower, incident):
        value[:2] *= 1000.
    second = solve(upper, lower, incident, separation_mm=40000.)
    np.testing.assert_allclose(first.upper_kick_matrix_mrad, second.upper_kick_matrix_mrad, rtol=2e-14)
    np.testing.assert_allclose(first.lower_kick_matrix_mrad, second.lower_kick_matrix_mrad, rtol=2e-14)
    assert second.condition_number == pytest.approx(first.condition_number, rel=2e-14)


@pytest.mark.parametrize("bad", [
    np.zeros((2, 4)), np.zeros((4,)), np.zeros((4, 2, 1)), np.full((4, 2), True),
    [[True, 0.], [0., 1.], [1., 0.], [0., 1.]], np.full((4, 2), np.nan),
    np.full((4, 2), np.inf), np.ones((4, 2), dtype=complex), np.full((4, 2), "1"),
])
@pytest.mark.parametrize("argument", ["upper", "lower", "incident"])
def test_invalid_phase_matrices_are_rejected(bad, argument):
    upper, lower = drift_responses()
    values = dict(upper=upper, lower=lower, incident=np.vstack((np.eye(2)*1e-7, np.eye(2)*1e-6)))
    values[argument] = bad
    with pytest.raises(ValueError, match="finite real 4x2"):
        solve(**values)


@pytest.mark.parametrize("field,value", [
    ("separation_mm", 0.), ("separation_mm", -1.), ("separation_mm", True),
    ("separation_mm", np.nan), ("separation_mm", np.inf), ("separation_mm", None),
    ("upper_gain", 0.), ("upper_gain", True), ("upper_gain", np.nan), ("upper_gain", np.inf),
    ("condition_limit", .9), ("condition_limit", 0.), ("condition_limit", True),
    ("condition_limit", np.nan), ("condition_limit", np.inf),
])
def test_invalid_scalars_are_rejected(field, value):
    upper, lower = drift_responses()
    with pytest.raises(ValueError):
        solve(upper, lower, np.ones((4, 2)), **{field: value})


def test_coincident_coils_and_nearly_singular_response_are_rejected():
    upper, _ = drift_responses()
    incident = np.vstack((np.eye(2)*1e-7, np.eye(2)*1e-6))
    with pytest.raises(ValueError, match="phase-space response.*singular or ill-conditioned"):
        solve(upper, upper, incident)
    upper, lower = drift_responses(.1, .1+1e-14)
    with pytest.raises(ValueError, match="phase-space response.*singular or ill-conditioned"):
        solve(upper, lower, incident)


def test_singular_upper_command_cannot_be_hidden_by_a_pseudoinverse():
    upper, lower = drift_responses()
    ku, kl = np.diag([1e-6, 0.]), np.eye(2)*2e-6
    with pytest.raises(ValueError, match="Upper coil command factorization.*singular or ill-conditioned"):
        solve(upper, lower, -(upper @ ku + lower @ kl))


def test_finite_inputs_that_overflow_scaling_or_commands_fail_explicitly():
    upper, lower = drift_responses()
    incident = np.vstack((np.eye(2)*1e-7, np.eye(2)*1e-6))
    with pytest.raises(ValueError, match="finite numerical range"):
        solve(upper, lower, incident, upper_gain=1e-320)
    upper[:2] = 1e308
    with pytest.raises(ValueError, match="finite numerical range"):
        solve(upper, lower, incident, separation_mm=1e-300)


@pytest.mark.parametrize("name", ["separation_mm", "upper_gain", "condition_limit"])
def test_out_of_float_range_integer_controls_raise_validation_error(name):
    upper, lower = drift_responses()
    incident = np.vstack((np.eye(2)*1e-7, np.eye(2)*1e-6))
    with pytest.raises(ValueError, match="finite real number"):
        solve(upper, lower, incident, **{name: 10**1000})
