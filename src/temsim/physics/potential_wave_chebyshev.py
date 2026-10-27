"""Certified coefficients for an optional bounded Hermitian exponential.

The certificate covers exact-arithmetic polynomial truncation and binary64
coefficient conversion. FFT, normalization by the supplied operator bound,
and recurrence roundoff remain separate. This module never relaxes a flux,
boundary, spectrum, kinetic, or domain gate.
"""
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np


_MAX_DEGREE = 256
_MAX_GROUP = 32
_BASE_BYTES_PER_CELL = 640
_MINIMUM_BASE_BYTES_PER_CELL = 384
# In addition to the existing residual workspace, reserve two complex grids
# for the extra recurrence state and an in-place accumulation scratch array.
_RECURRENCE_RESERVE_PER_CELL = 32


class ChebyshevPlanUnavailable(ValueError):
    """The unchanged Taylor method is required before any state is executed."""


def _positive_integer(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be an explicit {'nonnegative' if zero else 'positive'} integer")
    return value


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be {'positive and ' if positive else ''}finite")
    return result


def _up(value):
    result = float(value)
    if Fraction(result) < value:
        result = float(np.nextafter(result, np.inf))
    return result


def _ceil(value):
    return -(-value.numerator // value.denominator)


def _tail_bound(x, degree):
    """Rational bound on 2*sum(abs(J_n(x)), n>degree), for real x.

    The absolute Bessel series gives a_n=(|x|/2)^n/n! *
    exp(x^2/(4(n+1))). Its successive ratio is at most |x|/(2(n+1)).
    exp(y)<=2^ceil(2*y), since log(2)>1/2, avoids floating log/exp bounds.
    See https://dlmf.nist.gov/10.2.E2 and https://dlmf.nist.gov/10.12.
    """
    x = abs(x)
    if not x:
        return Fraction(0)
    n = degree + 1
    ratio = x / (2 * (n + 1))
    if ratio >= 1:
        return None
    exponent = _ceil(x*x / (2 * (n + 1)))
    first = (x/2)**n / math.factorial(n) * 2**exponent
    return 2*first / (1-ratio)


def _coefficient(n, x, threshold, cancelled):
    """Exact alternating Bessel series and float-conversion certificate."""
    term = (x/2)**n / math.factorial(n)
    total, k = term, 0
    while True:
        if k % 16 == 0 and cancelled():
            raise InterruptedError("Carrier potential propagation cancelled")
        ratio = x*x / (4*(k+1)*(n+k+1))
        following = -term*ratio
        if ratio < 1 and abs(following) < threshold:
            value = float(total)
            return value, abs(Fraction(value)-total)+abs(following)
        total += following
        term, k = following, k+1
        if k > 1024:
            raise ChebyshevPlanUnavailable("Bessel coefficient evaluation exceeds its bounded series budget")


@dataclass(frozen=True, slots=True)
class ChebyshevGroup:
    count: int
    degree: int
    coefficients: tuple
    truncation_bound: float
    coefficient_roundoff_bound: float
    error_bound: float
    recurrence_roundoff_amplification: float


@dataclass(frozen=True, slots=True)
class ChebyshevPlan:
    groups: tuple
    count: int
    old_applications: int
    applications: int
    norm_bound: float
    error_budget: float
    error_bound: float
    truncation_bound: float
    coefficient_roundoff_bound: float
    working_bytes: int
    base_bytes_per_cell: int = _BASE_BYTES_PER_CELL

    def to_metadata(self):
        return {
            "method": "grouped_chebyshev",
            "group_sizes": [group.count for group in self.groups],
            "group_degrees": [group.degree for group in self.groups],
            "original_substep_checks": self.count,
            "old_taylor_planned_applications": self.old_applications,
            "actual_operator_applications": self.applications,
            "generator_norm_bound": self.norm_bound,
            "series_error_budget": self.error_budget,
            "series_error_bound": self.error_bound,
            "polynomial_truncation_bound": self.truncation_bound,
            "coefficient_roundoff_bound": self.coefficient_roundoff_bound,
            "coefficient_time_roundoff_bound": 0.0,
            "time_definition": "exact Fraction(k_distance)*Fraction(norm_bound)*j/original_count; no rounded kappa substituted",
            "group_bounds": [group.error_bound for group in self.groups],
            "estimated_working_bytes": self.working_bytes,
            "planning_base_bytes_per_cell": self.base_bytes_per_cell,
            "workspace": f"{self.base_bytes_per_cell} bytes/cell residual base + 32 bytes/cell extra recurrence/axpy reserve + 16 bytes/cell/group-output + bounded coefficient table",
            "recurrence_roundoff_amplification": math.fsum(group.recurrence_roundoff_amplification for group in self.groups),
            "roundoff_scope": "Diagnostic sum of local conditional recurrence/FFT injection weights, not a complete propagated floating-point guarantee. Later approximate-group amplification, FFT errors and reduction errors remain separate. Not part of the rational series certificate; does not increase the original Taylor flux tolerance.",
            "assumption": "The same supplied bound M admits a self-adjoint H with spectrum(H/M) in [-1,1]",
        }


def chebyshev_working_bytes(cells, group_size, degree=_MAX_DEGREE, *,
                           base_bytes_per_cell=_BASE_BYTES_PER_CELL):
    """Conservative preallocation bound; no temporary group-by-grid products."""
    _positive_integer(cells, "cells")
    _positive_integer(group_size, "group_size")
    _positive_integer(degree, "degree", zero=True)
    _positive_integer(base_bytes_per_cell, "base_bytes_per_cell")
    if base_bytes_per_cell < _MINIMUM_BASE_BYTES_PER_CELL:
        raise ValueError("Chebyshev residual base must be at least 384 bytes per cell")
    # Three bounded coefficient tables cover complex coefficients plus the
    # real absolute-weight and planning temporaries. Group recurrence arrays
    # are already covered by the explicit per-cell reserve above.
    return (base_bytes_per_cell + _RECURRENCE_RESERVE_PER_CELL + 16*group_size)*cells + 48*group_size*(degree+1)


def plan_chebyshev_exponential(k_distance, norm_bound, count, degree, error_budget, *,
        cells, maximum_working_bytes, maximum_group=8, cancelled=lambda: False,
        base_bytes_per_cell=_BASE_BYTES_PER_CELL):
    """Plan without executing H; fail explicitly when Taylor must be retained.

    ``count`` and ``degree`` are the ORIGINAL Taylor plan, including its
    ceil(fl(|k_distance|*M)) check schedule. Every one of those times remains.
    """
    k_distance = _finite(k_distance, "k_distance")
    norm_bound = _finite(norm_bound, "generator_norm_bound", positive=True)
    error_budget = _finite(error_budget, "series_error_budget", positive=True)
    _positive_integer(count, "count")
    _positive_integer(degree, "degree")
    if count > 1024 or degree > 64 or count*degree > 32768:
        raise ValueError("Chebyshev execution requires a bounded original Taylor plan")
    _positive_integer(cells, "cells")
    _positive_integer(maximum_working_bytes, "maximum_working_bytes")
    _positive_integer(maximum_group, "maximum_chebyshev_group")
    # The default retains the CPU workspace contract. Only a separately
    # validated backend may select a smaller base; the caller still enforces
    # its actual allocation limit and independently reserves host inputs.
    chebyshev_working_bytes(cells, 1, base_bytes_per_cell=base_bytes_per_cell)
    if maximum_group > _MAX_GROUP:
        raise ValueError(f"maximum_chebyshev_group must not exceed {_MAX_GROUP}")
    if cancelled():
        raise InterruptedError("Carrier potential propagation cancelled")
    exact_step = Fraction(k_distance)*Fraction(norm_bound)/count
    # Keep the old bounded-step schedule, allowing only its product-rounding
    # difference. A mismatched arbitrary schedule is not a valid opt-in plan.
    rounded_kappa = abs(k_distance)*norm_bound
    if not math.isfinite(rounded_kappa) or count != max(1, math.ceil(rounded_kappa)):
        raise ValueError("Chebyshev execution requires the original bounded Taylor substep schedule")
    target = Fraction(min(error_budget, 1e-12))
    last_reason = "No grouped Chebyshev workspace fits the declared budget"
    for size in range(min(maximum_group, count), 0, -1):
        if chebyshev_working_bytes(cells, size, base_bytes_per_cell=base_bytes_per_cell) > maximum_working_bytes:
            continue
        if cancelled():
            raise InterruptedError("Carrier potential propagation cancelled")
        sizes = tuple(min(size, count-start) for start in range(0, count, size))
        allowance = target/(2*len(sizes))
        threshold = min(Fraction(1, 2**190), allowance/(2**20*_MAX_DEGREE))
        group_cache, groups = {}, []
        composed = truncation_composed = coefficient_composed = Fraction(1)
        try:
            for amount in sizes:
                if amount not in group_cache:
                    xmax = abs(exact_step*amount)
                    polynomial_degree = 1
                    while polynomial_degree <= _MAX_DEGREE:
                        tail = _tail_bound(xmax, polynomial_degree)
                        if tail is not None and tail <= allowance/2:
                            break
                        polynomial_degree += 1
                    if polynomial_degree > _MAX_DEGREE:
                        raise ChebyshevPlanUnavailable("Chebyshev degree exceeds its bounded planning budget")
                    rows, errors, coefficient_errors, tails, fp_weights = [], [], [], [], []
                    for j in range(1, amount+1):
                        x = exact_step*j
                        row, error = [], Fraction(0)
                        for n in range(polynomial_degree+1):
                            value, conversion = _coefficient(n, x, threshold, cancelled)
                            factor = 1 if n == 0 else 2
                            row.append(factor*((-1j)**n)*value)
                            error += factor*conversion
                        tail = _tail_bound(x, polynomial_degree)
                        rows.append(tuple(row)); coefficient_errors.append(error)
                        tails.append(tail); errors.append(tail+error)
                        fp_weights.append(math.fsum(abs(c)*(.5*n*(n+1)+polynomial_degree+1)
                                                    for n, c in enumerate(row)))
                    maximum_error = max(errors)
                    if maximum_error > allowance:
                        raise ChebyshevPlanUnavailable("Bessel coefficient rounding cannot meet this grouped series allowance")
                    group_cache[amount] = (ChebyshevGroup(amount, polynomial_degree, tuple(rows),
                        _up(max(tails)), _up(max(coefficient_errors)), _up(maximum_error), max(fp_weights)),
                        maximum_error, max(tails), max(coefficient_errors))
                group, error, tail, conversion = group_cache[amount]
                groups.append(group)
                composed *= 1+error
                truncation_composed *= 1+tail
                coefficient_composed *= 1+conversion
            if composed-1 > target:
                raise ChebyshevPlanUnavailable("Composed Chebyshev certificate exceeds the original series allowance")
            applications = sum(group.degree for group in groups)
            if applications >= count*degree:
                raise ChebyshevPlanUnavailable("Grouped Chebyshev does not reduce the original operator count")
            return ChebyshevPlan(tuple(groups), count, count*degree, applications, norm_bound,
                error_budget, _up(composed-1), _up(truncation_composed-1), _up(coefficient_composed-1),
                chebyshev_working_bytes(cells, size, base_bytes_per_cell=base_bytes_per_cell),
                base_bytes_per_cell)
        except ChebyshevPlanUnavailable as error:
            last_reason = str(error)
    raise ChebyshevPlanUnavailable(last_reason)


__all__ = ("ChebyshevPlanUnavailable", "ChebyshevPlan", "chebyshev_working_bytes", "plan_chebyshev_exponential")
