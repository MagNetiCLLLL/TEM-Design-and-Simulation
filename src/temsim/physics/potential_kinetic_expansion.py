"""Positive-branch kinetic Taylor coefficients and complete-domain bounds.

For a bounded Hermitian K, H = (1-sqrt(1-beta^2 K))/beta. Order 2N
retains powers through K^N. The remainder uses the FULL represented spectrum,
even when real multiplication powers are exponentiated separately. These are
finite-grid kinetic approximation bounds, not spatial or Strang error bounds.
"""
import math
from numbers import Real

import numpy as np


KINETIC_ORDERS = (2, 4, 6, 8)


def validate_kinetic_order(order):
    if type(order) is not int or order not in KINETIC_ORDERS:
        raise ValueError("kinetic_order must be an integer: 2, 4, 6 or 8")
    return order


def kinetic_coefficient(power):
    """Exact binary64 coefficients for all retained and first omitted powers."""
    if type(power) is not int or not 1 <= power <= 5:
        raise ValueError("Kinetic expansion power must be an integer from 1 through 5")
    return math.comb(2*power, power)/(4**power*(2*power-1))


def _nonnegative(value, name, *, positive=False):
    try:
        valid = (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
                 and math.isfinite(float(value)) and (value > 0 if positive else value >= 0))
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} real scalar")
    return float(value)


def kinetic_remainder_phase_bound(order, wave_number, distance, beta, squared_root_bound):
    """Taylor-Lagrange bound for |beta^2 K| <= x < 1 on the entire grid.

    The maximum (N+1)th derivative of 1-sqrt(1-s) on [-x,x] occurs at x.
    Hence ||remainder|| <= c_(N+1) x^(N+1)/(1-x)^(N+1/2).
    The historical order-2/4 arithmetic is kept unchanged.
    """
    validate_kinetic_order(order)
    k = _nonnegative(wave_number, "Wave number", positive=True)
    if not isinstance(distance, Real) or isinstance(distance, (bool, np.bool_)):
        raise ValueError("Propagation length must be a finite real scalar")
    length = _nonnegative(abs(distance), "Propagation length")
    scale = _nonnegative(beta, "Momentum scale", positive=True)
    x = _nonnegative(squared_root_bound, "Complete kinetic spectral bound")
    if x >= 1:
        raise ValueError("Complete kinetic spectral bound must be less than one")
    if order == 2:
        return k*length/scale*x**2/(8*(1-x)**1.5)
    if order == 4:
        return k*length/scale*x**3/(16*(1-x)**2.5)
    power = order//2+1
    return k*length/scale*kinetic_coefficient(power)*x**power/(1-x)**(power-.5)


def differential_power_norm_bound(power, linear_norm, multiplication_norm):
    """Bound ||K^n-U^n|| by positive terms, with K=L+U.

    This is sum_{j=1}^n binomial(n,j) ||L||^j ||U||^(n-j), counting
    noncommuting words without subtracting two nearly equal large powers.
    """
    if type(power) is not int or not 1 <= power <= 4:
        raise ValueError("Differential power must be an integer from 1 through 4")
    linear = _nonnegative(linear_norm, "Linear operator norm")
    multiplication = _nonnegative(multiplication_norm, "Multiplication norm")
    return math.fsum(math.comb(power, j)*linear**j*multiplication**(power-j)
                     for j in range(1, power+1))


__all__ = ("KINETIC_ORDERS", "validate_kinetic_order", "kinetic_coefficient",
           "kinetic_remainder_phase_bound", "differential_power_norm_bound")
