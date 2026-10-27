"""Independent high-precision checks of the complete kinetic Taylor bound."""
from decimal import Decimal, localcontext
from fractions import Fraction
from math import comb

import numpy as np
import pytest

from temsim.physics.potential_kinetic_expansion import (
    validate_kinetic_order, kinetic_coefficient, kinetic_remainder_phase_bound,
    differential_power_norm_bound,
)


def test_coefficients_are_exact_positive_branch_values():
    expected = (Fraction(1, 2), Fraction(1, 8), Fraction(1, 16), Fraction(5, 128), Fraction(7, 256))
    for power, coefficient in enumerate(expected, 1):
        assert Fraction.from_float(kinetic_coefficient(power)) == coefficient


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("radius", [1e-5, .03, .099])
def test_full_signed_spectrum_remainder_is_bounded_without_reducing_domain(order, radius):
    with localcontext() as context:
        context.prec = 90
        x = Decimal.from_float(radius)
        worst = Decimal(0)
        for numerator in range(-32, 33):
            value = x*Decimal(numerator)/32
            exact = 1-(1-value).sqrt()
            polynomial = sum(Decimal(comb(2*n, n))*value**n/Decimal(4**n*(2*n-1))
                             for n in range(1, order//2+1))
            worst = max(worst, abs(exact-polynomial))
        bound = kinetic_remainder_phase_bound(order, 1., 1., 1., radius)
        assert float(worst) <= bound
        assert bound > 0
        # The signed interval and radius are identical for every tested order.
        if order > 2:
            assert bound < kinetic_remainder_phase_bound(order-2, 1., 1., 1., radius)


@pytest.mark.parametrize("power", [2, 3, 4])
def test_positive_word_norm_bound_remains_nonzero_when_subtraction_loses_it(power):
    linear, multiplication = 1e-30, 1.
    assert (linear+multiplication)**power-multiplication**power == 0.
    actual = differential_power_norm_bound(power, linear, multiplication)
    assert actual > 0.
    assert actual == pytest.approx(power*linear, rel=2e-16)


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 4., "6", 0, 3, 10, np.int64(8)])
def test_kinetic_order_is_an_explicit_supported_integer(invalid):
    with pytest.raises(ValueError, match="kinetic_order"):
        validate_kinetic_order(invalid)


@pytest.mark.parametrize("order", [2, 4])
def test_historical_remainder_arithmetic_is_unchanged(order):
    k, distance, beta, x = 5.932e11, 1.79e-4, .913, .00132
    expected = (k*abs(distance)/beta*x**2/(8*(1-x)**1.5) if order == 2 else
                k*abs(distance)/beta*x**3/(16*(1-x)**2.5))
    assert kinetic_remainder_phase_bound(order, k, distance, beta, x) == expected
