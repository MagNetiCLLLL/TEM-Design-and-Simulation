"""Math-only dense polynomial wall checks; no field or particle integration."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.rk import Dop853DenseOutput
from scipy.optimize import brentq

from scripts.ideal_lens_wall import classify_cylindrical_wall


def _quadratics(rows, *, start=2., step=4., width=4):
    """Known x(s)=a+b*s+c*s**2; y=0, with exact analytic slope/clock."""
    rows = np.asarray(rows, float)
    initial = np.zeros((len(rows), width))
    initial[:, 0], initial[:, 2] = rows[:, 0], rows[:, 1]/step
    stages = np.zeros((7, len(rows), width))
    stages[0, :, 0], stages[1, :, 0] = rows[:, 1]+rows[:, 2], -rows[:, 2]
    stages[0, :, 2] = 2*rows[:, 2]/step
    if width == 5:
        initial[:, 4] = 1e-9
        stages[0, :, 4] = step*1e-8
    local = Dop853DenseOutput(start, start+step, initial.ravel(), stages.reshape(7, -1))
    return OdeSolution([start, start+step], [local])


def test_straight_crossing_outside_entrance_and_no_hit_preserve_source_ids():
    solution = _quadratics([[.1, 1.1, 0.], [1.1, 0., 0.], [.2, .1, 0.]])
    result = classify_cylindrical_wall(solution, 1., [91, 7, 1003])
    np.testing.assert_array_equal(result["source_ids"], [91, 7, 1003])
    assert result["status"].tolist() == ["wall_contact", "outside_at_entrance", "survived"]
    assert result["classification_complete"]
    assert result["hit_z_m"][0] == pytest.approx(2+4*.9/1.1, abs=2e-13)
    assert result["hit_z_m"][1] == 2.
    assert np.isnan(result["hit_z_m"][2])
    np.testing.assert_allclose(result["hit_phase"][0], solution(result["hit_z_m"][0]).reshape(3, 4)[0], atol=2e-14)
    np.testing.assert_array_equal(result["alive_at_end"], [False, False, True])


def test_exit_and_reentry_are_detected_when_both_endpoints_are_inside():
    solution = _quadratics([[.5, 3., -3.]])
    result = classify_cylindrical_wall(solution, 1.)
    fraction = (1-np.sqrt(1/3))/2
    assert solution(2.)[0] < 1. and solution(6.)[0] < 1.
    assert result["status"].tolist() == ["wall_contact"]
    assert result["hit_z_m"][0] == pytest.approx(2+4*fraction, abs=2e-13)
    assert result["hit_radial_residual_m"][0] < 1e-13


def test_stationary_tangent_contact_is_retained_without_sign_change():
    solution = _quadratics([[.5, 2., -2.]])
    result = classify_cylindrical_wall(solution, 1.)
    assert result["status"].tolist() == ["grazing_contact"]
    assert result["hit_z_m"][0] == pytest.approx(4., abs=2e-13)
    assert result["classification_complete"]


def test_near_tangent_below_wall_is_unresolved_not_a_certified_loss():
    solution = _quadratics([[.5-5e-13, 2., -2.]])
    result = classify_cylindrical_wall(solution, 1.)
    assert result["status"].tolist() == ["unresolved"]
    assert not result["classification_complete"]
    assert not result["alive_at_end"][0]
    assert np.isnan(result["hit_z_m"][0])


def test_roundoff_sized_positive_tangent_is_unresolved():
    solution = _quadratics([[.5+2e-15, 2., -2.]])
    result = classify_cylindrical_wall(solution, 1.)
    assert result["status"].tolist() == ["unresolved"]
    assert not result["classification_complete"]
    assert not result["physical_loss_mask"][0]


def test_five_component_state_returns_actual_interpolated_clock():
    solution = _quadratics([[.1, 1.1, 0.]], width=5)
    result = classify_cylindrical_wall(solution, 1., state_width=5)
    assert result["hit_phase"].shape == (1, 5)
    np.testing.assert_allclose(result["hit_phase"][0], solution(result["hit_z_m"][0]), atol=2e-14)
    assert result["hit_phase"][0, 4] == pytest.approx(1e-9+4e-8*.9/1.1, rel=1e-13)


def test_concatenated_intervals_keep_first_contact_and_include_later_bounds():
    first = _quadratics([[.1, .2, 0.]], start=0., step=1.)
    second = _quadratics([[.3, .2, 0.]], start=1., step=1.)
    # The constant slope is continuous at the shared node.
    solution = OdeSolution([0., 1., 2.], [first.interpolants[0], second.interpolants[0]])
    result = classify_cylindrical_wall(solution, .4)
    assert result["hit_z_m"][0] == pytest.approx(1.5, abs=2e-13)
    assert result["counterfactual_max_radius_upper_bound_m"][0] >= .5
    assert result["counterfactual_endpoint_max_radius_m"][0] == pytest.approx(.5)


def test_rotation_of_xy_does_not_change_contact_and_bounds_enclose_dense_curve():
    solution = _quadratics([[.5, 3., -3.]])
    local = solution.interpolants[0]
    angle = .71
    # Rotate the complete x/y and slope pair in the stored SciPy representation.
    initial = local.y_old.reshape(-1, 4).copy()
    stages = local.F.reshape(7, -1, 4).copy()
    for values in (initial, stages):
        x, u = values[..., 0].copy(), values[..., 2].copy()
        values[..., 0], values[..., 1] = np.cos(angle)*x, np.sin(angle)*x
        values[..., 2], values[..., 3] = np.cos(angle)*u, np.sin(angle)*u
    rotated = OdeSolution(solution.ts, [Dop853DenseOutput(2., 6., initial.ravel(), stages.reshape(7, -1))])
    expected = classify_cylindrical_wall(solution, 1.)
    result = classify_cylindrical_wall(rotated, 1.)
    np.testing.assert_allclose(result["hit_z_m"], expected["hit_z_m"], atol=1e-13)
    samples = rotated(np.linspace(2., 6., 1001))
    assert np.hypot(samples[0], samples[1]).max() <= result["counterfactual_max_radius_upper_bound_m"][0]


def test_all_seven_dense_rows_match_scipy_evaluation_at_the_contact():
    random = np.random.default_rng(83421)
    stages = random.normal(0., .002, (7, 4))
    stages[0] = [1.1, 0., 0., 0.]
    local = Dop853DenseOutput(2., 6., [.1, 0., 0., 0.], stages)
    solution = OdeSolution([2., 6.], [local])
    expected = brentq(lambda z: np.hypot(*solution(z)[:2])-1., 2., 6.)
    result = classify_cylindrical_wall(solution, 1.)
    assert result["hit_z_m"][0] == pytest.approx(expected, abs=2e-12)
    np.testing.assert_allclose(result["hit_phase"][0], solution(expected), atol=2e-12)
    assert result["physical_loss_mask"].tolist() == [True]
    assert result["unresolved_mask"].tolist() == [False]


@pytest.mark.parametrize("ids", [[1, 1], [1], [1., 2.], [1, -2]])
def test_invalid_original_source_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="Source IDs"):
        classify_cylindrical_wall(_quadratics([[.1, 0., 0.], [.2, 0., 0.]]), 1., ids)


@pytest.mark.parametrize("radius", [0., -1., np.nan, True])
def test_invalid_wall_radius_is_rejected(radius):
    with pytest.raises(ValueError, match="radius"):
        classify_cylindrical_wall(_quadratics([[.1, 0., 0.]]), radius)


def test_invalid_interval_nonfinite_state_and_discontinuity_are_rejected():
    solution = _quadratics([[.1, .2, 0.]], start=0., step=1.)
    with pytest.raises(ValueError, match="nodes"):
        classify_cylindrical_wall(SimpleNamespace(ts=[0., 0.], interpolants=solution.interpolants), 1.)
    local = solution.interpolants[0]
    bad = Dop853DenseOutput(0., 1., np.array([np.nan, 0., 0., 0.]), local.F)
    with pytest.raises(ValueError, match="finite flattened"):
        classify_cylindrical_wall(OdeSolution([0., 1.], [bad]), 1.)
    jumped = _quadratics([[.5, .2, 0.]], start=1., step=1.)
    with pytest.raises(ValueError, match="discontinuity"):
        classify_cylindrical_wall(OdeSolution([0., 1., 2.], [local, jumped.interpolants[0]]), 1.)


def test_clock_discontinuity_is_not_hidden_by_a_position_sized_tolerance():
    first = _quadratics([[.1, .2, 0.]], start=0., step=1., width=5)
    second = _quadratics([[.3, .2, 0.]], start=1., step=1., width=5)
    second.interpolants[0].y_old[4] = first(1.)[4]+1e-16
    with pytest.raises(ValueError, match="discontinuity"):
        classify_cylindrical_wall(OdeSolution([0., 1., 2.],
            [first.interpolants[0], second.interpolants[0]]), 1., state_width=5)
