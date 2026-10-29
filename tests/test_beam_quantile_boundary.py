"""Analytic current-containment boundaries, independent of optical transport."""
from fractions import Fraction

import numpy as np
import pytest

from temsim.physics.beam_statistics import _weighted_quantile, transverse_beam_statistics


@pytest.mark.parametrize("count", [1000, 2000, 4096, 12289])
@pytest.mark.parametrize("fraction", [.5, .9, .95, .99, 1.])
def test_equal_current_selects_the_exact_integer_rank(count, fraction):
    values = np.arange(1., count + 1)
    weights = np.full(count, 1. / count)
    weights /= weights.sum()
    rational_rank = Fraction(str(fraction)) * count
    expected = -(-rational_rank.numerator // rational_rank.denominator)
    assert _weighted_quantile(values, weights, fraction) == expected
    assert _weighted_quantile(values[::-1], weights[::-1], fraction) == expected


def test_decimal_boundary_and_zero_current_entries():
    values = np.array([0., 1., 2., 3., 100.])
    weights = np.array([0., .3, .6, .1, 0.])
    assert _weighted_quantile(values, weights, .9) == 2.
    assert _weighted_quantile(values, weights, 1.) == 3.


def test_small_resolved_current_deficit_is_not_rounded_into_containment():
    deficit = 1e-12
    weights = np.array([.95 - deficit, .05 + deficit])
    assert _weighted_quantile(np.array([1., 10.]), weights, .95) == 10.


def test_weight_scale_and_splitting_one_current_packet_preserve_quantile():
    values = np.array([1., 2., 3., 4.])
    weights = np.array([.2, .3, .45, .05])
    for scale in (1., 1e-20, 1e20):
        assert _weighted_quantile(values, weights * scale, .95) == 3.
        assert _weighted_quantile(np.array([1., 2., 3., 3., 4.]),
            np.array([.2, .3, .225, .225, .05]) * scale, .95) == 3.


def test_public_radial_and_angular_statistics_share_the_boundary():
    # Opposite pairs give an exact zero chief. Each radius contains two equal
    # current particles, so the 95% radius is pair 950 of 1000.
    r = np.repeat(np.arange(1., 1001.), 2) * 1e-9
    sign = np.tile([-1., 1.], 1000)
    slopes = np.repeat(np.arange(1., 1001.), 2) * 1e-6 * sign
    result = transverse_beam_statistics(r * sign, np.zeros(2000), slopes,
        np.zeros(2000), weights=np.full(2000, .0005))
    assert result.radius_95_m == pytest.approx(950e-9, rel=0, abs=1e-21)
    assert result.convergence_95_rad == pytest.approx(np.arctan(950e-6), rel=0, abs=1e-17)


@pytest.mark.parametrize("weights", [
    [1., 1e-16],
    [1., 1.5e-16],
    [1., 1.5e-16, 0.],
])
def test_full_containment_includes_tiny_positive_current_but_not_zero(weights):
    values = np.arange(1., len(weights)+1)
    weights = np.asarray(weights)
    assert _weighted_quantile(values, weights, 1.) == 2.
    assert _weighted_quantile(values[::-1], weights[::-1], 1.) == 2.
