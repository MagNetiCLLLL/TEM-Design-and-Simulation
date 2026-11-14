"""Continuous cylindrical-wall classification of retained DOP853 polynomials.

This module does not integrate rays or change a field. It finds first contact
with an absorbing, coaxial cylinder along each numerical dense trajectory.
The interpolation and root calculations are floating-point approximations,
not a proof about the exact Lorentz solution. Independent transport refinement
remains necessary. Counterfactual trajectory extensions after contact never
contribute to the returned surviving population.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.optimize import brentq


_EPS = np.finfo(float).eps
_DEGREE = 7
_POWER_TO_BERNSTEIN = np.array([
    [math.comb(k, j) / math.comb(_DEGREE, j) if j <= k else 0.
     for j in range(_DEGREE + 1)]
    for k in range(_DEGREE + 1)
])


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0.):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _coefficients(interpolant, width):
    """Expand SciPy's alternating x/(1-x) DOP853 dense representation."""
    try:
        raw_initial, raw_stages = np.asarray(interpolant.y_old), np.asarray(interpolant.F)
    except AttributeError as error:
        raise ValueError("Only DOP853 local interpolants are supported") from error
    if raw_initial.dtype.kind not in "iuf" or raw_stages.dtype.kind not in "iuf":
        raise ValueError("DOP853 states and coefficients must be real numeric arrays")
    initial, stages = np.asarray(raw_initial, dtype=float), np.asarray(raw_stages, dtype=float)
    if (initial.ndim != 1 or initial.size == 0 or initial.size % width
            or stages.shape != (7, initial.size)
            or not np.isfinite(initial).all() or not np.isfinite(stages).all()):
        raise ValueError("Each DOP853 interpolant needs finite flattened particle states and seven F rows")
    coefficient = np.zeros((1, initial.size))
    for index, row in enumerate(stages[::-1]):
        coefficient[0] += row
        expanded = np.zeros((len(coefficient) + 1, initial.size))
        if index % 2 == 0:
            expanded[1:] = coefficient
        else:
            expanded[:-1] = coefficient
            expanded[1:] -= coefficient
        coefficient = expanded
    coefficient[0] += initial
    if not np.isfinite(coefficient).all():
        raise ValueError("DOP853 polynomial coefficients exceed the finite range")
    return coefficient.reshape(8, -1, width)


def _first_root(x, y, radius, wall_tolerance):
    """Return contact fraction, classification, residual and an audit reason.

    Stationary points divide the degree-14 radial polynomial into monotone
    intervals. This catches exit/reentry within a step and tangent contacts.
    Ambiguous near-real critical roots and near misses fail closed as unresolved.
    """
    x, y = x / radius, y / radius
    polynomial = np.polynomial.polynomial.polyadd(
        np.polynomial.polynomial.polymul(x, x),
        np.polynomial.polynomial.polymul(y, y))
    polynomial[0] -= 1.
    if not np.isfinite(polynomial).all():
        return None, "unresolved", math.nan, "nonfinite_radial_polynomial"
    scale = max(1., float(np.sum(np.abs(polynomial))))
    rounding = 128 * _EPS * scale
    tolerance = max(2 * wall_tolerance / radius + (wall_tolerance / radius)**2,
                    rounding)
    derivative = np.polynomial.polynomial.polyder(polynomial)
    try:
        roots = np.polynomial.polynomial.polyroots(derivative)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, "unresolved", math.nan, "critical_root_solver_failed"
    if not np.isfinite(roots).all():
        return None, "unresolved", math.nan, "nonfinite_critical_roots"
    critical = []
    derivative_scale = max(float(np.sum(np.abs(derivative))), np.finfo(float).tiny)
    for root in roots:
        real, imaginary = float(np.real(root)), abs(float(np.imag(root)))
        if not -1e-10 <= real <= 1. + 1e-10:
            continue
        if 1e-10 < imaginary <= 1e-6:
            return None, "unresolved", math.nan, "ambiguous_near_real_critical_root"
        if imaginary > 1e-6 or not 0. < real < 1.:
            continue
        residual = abs(float(np.polynomial.polynomial.polyval(real, derivative)))
        if residual > 2e-8 * derivative_scale:
            return None, "unresolved", math.nan, "critical_root_residual_exceeds_tolerance"
        critical.append(real)
    knots = np.unique(np.r_[0., critical, 1.])
    values = np.polynomial.polynomial.polyval(knots, polynomial)
    if not np.isfinite(values).all():
        return None, "unresolved", math.nan, "nonfinite_radial_values"
    if values[0] > 0.:
        return None, "unresolved", math.nan, "unclassified_outside_step_start"
    if values[0] == 0.:
        return 0., "wall_contact", abs(float(values[0])), "step_start_contact"
    if values[0] >= -tolerance:
        return None, "unresolved", math.nan, "step_start_within_wall_uncertainty"
    for index in range(1, len(knots)):
        left, right = float(knots[index-1]), float(knots[index])
        lower, upper = float(values[index-1]), float(values[index])
        if lower < 0. and upper > rounding:
            try:
                root = brentq(lambda t: float(np.polynomial.polynomial.polyval(t, polynomial)),
                              left, right, xtol=4*_EPS, rtol=4*_EPS)
            except (ValueError, RuntimeError):
                return None, "unresolved", math.nan, "bracketed_wall_root_failed"
            residual = abs(float(np.polynomial.polynomial.polyval(root, polynomial)))
            if residual > tolerance:
                return None, "unresolved", residual, "wall_root_residual_exceeds_tolerance"
            return root, "wall_contact", residual, "bracketed_first_contact"
        if upper == 0.:
            tangent = index < len(knots)-1
            return right, "grazing_contact" if tangent else "wall_contact", 0., (
                "stationary_numerical_tangent" if tangent else "step_end_contact")
        if upper >= -tolerance:
            return None, "unresolved", math.nan, "grazing_or_near_miss_within_wall_uncertainty"
    return None, "survived", math.nan, "stationary_partition_strictly_inside"


def classify_cylindrical_wall(solution, radius_m, source_ids=None, *,
                             state_width=4, wall_tolerance_m=1e-12):
    """Classify every original particle in a SciPy ``OdeSolution``.

    Dense states are flattened particle rows (x, y, dx/dz, dy/dz[, clock]).
    The independent coordinate is axial position in metres. Concatenated
    DOP853 interpolants are supported if their intervals and states agree.
    No current normalization or loss restoration occurs here. The caller must
    reject ``classification_complete=False``; unresolved particles are neither
    survivors nor certified wall losses.
    """
    radius = _positive(radius_m, "Cylinder radius")
    tolerance = _positive(wall_tolerance_m, "Wall tolerance")
    if tolerance >= radius * 1e-3:
        raise ValueError("Wall tolerance must be smaller than 0.001 of the radius")
    if type(state_width) is not int or state_width not in (4, 5):
        raise ValueError("State width must be four or five")
    interpolants = tuple(getattr(solution, "interpolants", ()))
    if not interpolants:
        raise ValueError("A nonempty DOP853 OdeSolution is required")
    nodes = np.asarray(getattr(solution, "ts", ()), dtype=float)
    if (nodes.shape != (len(interpolants)+1,) or not np.isfinite(nodes).all()
            or np.any(np.diff(nodes) <= 0.)):
        raise ValueError("Dense solution nodes must be finite, increasing and complete")
    first = _coefficients(interpolants[0], state_width)
    count = first.shape[1]
    identifiers = np.arange(count, dtype=np.int64) if source_ids is None else np.asarray(source_ids)
    if (identifiers.shape != (count,) or identifiers.dtype.kind not in "iu"
            or np.any(identifiers < 0) or len(np.unique(identifiers)) != count):
        raise ValueError("Source IDs must be unique nonnegative integers, one per original particle")
    identifiers = identifiers.copy()
    status = np.full(count, "survived", dtype="U32")
    quality = np.full(count, "bernstein_inside_bound", dtype="U64")
    hit_z = np.full(count, np.nan)
    hit_phase = np.full((count, state_width), np.nan)
    root_residual = np.full(count, np.nan)
    unresolved_interval = np.full((count, 2), np.nan)
    maximum_bound = np.zeros(count)
    maximum_observed = np.zeros(count)
    initial_radius = np.hypot(first[0, :, 0], first[0, :, 1])
    outside, at_wall = initial_radius > radius, initial_radius == radius
    initial_contact = outside | at_wall
    status[outside], status[at_wall] = "outside_at_entrance", "entrance_wall_contact"
    quality[initial_contact] = "executed_initial_position"
    hit_z[initial_contact] = nodes[0]
    hit_phase[initial_contact] = first[0, initial_contact]
    root_residual[initial_contact] = np.abs(initial_radius[initial_contact]-radius)
    uncertain = ~initial_contact & (radius-initial_radius <= tolerance)
    status[uncertain], quality[uncertain] = "unresolved", "entrance_within_wall_uncertainty"
    unresolved_interval[uncertain] = nodes[0]
    previous_end = previous_scale = None
    slow_candidates = 0
    for index, interpolant in enumerate(interpolants):
        try:
            start, step = float(interpolant.t_old), float(interpolant.h)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("Only DOP853 local interpolants are supported") from error
        if (not np.isfinite((start, step)).all() or step <= 0.
                or start != nodes[index]
                or abs((start+step)-nodes[index+1]) > 8*_EPS*max(abs(nodes[index+1]), step)):
            raise ValueError("Local DOP853 intervals must exactly cover the declared solution")
        coefficients = first if index == 0 else _coefficients(interpolant, state_width)
        if coefficients.shape[1] != count:
            raise ValueError("Every interval must retain all original particle columns")
        if previous_end is not None:
            # Each component retains its own units, including a small flight
            # clock. A unit-sized absolute floor would hide clock discontinuities.
            scale = np.maximum(np.finfo(float).tiny, previous_scale+np.abs(coefficients[0]))
            if np.any(np.abs(previous_end-coefficients[0]) > 512*_EPS*scale):
                raise ValueError("Dense solution has a state discontinuity between intervals")
        previous_end = coefficients.sum(axis=0)
        previous_scale = np.sum(np.abs(coefficients), axis=0)
        xy = coefficients[:, :, :2]
        bernstein = np.einsum("ij,jnk->ink", _POWER_TO_BERNSTEIN, xy)
        control_radius = np.hypot(bernstein[:, :, 0], bernstein[:, :, 1])
        # The curve lies in the convex hull of its vector Bernstein controls.
        # Include floating-point expansion allowance; this is not interval arithmetic.
        allowance = 512*_EPS*np.sum(np.linalg.norm(xy, axis=2), axis=0)
        bounds = control_radius.max(axis=0)+allowance
        maximum_bound = np.maximum(maximum_bound, bounds)
        endpoint_radius = np.hypot(previous_end[:, 0], previous_end[:, 1])
        maximum_observed = np.maximum(maximum_observed, np.maximum(
            np.hypot(coefficients[0, :, 0], coefficients[0, :, 1]), endpoint_radius))
        candidates = np.flatnonzero((status == "survived") & (bounds >= radius-tolerance))
        slow_candidates += len(candidates)
        for particle in candidates:
            fraction, outcome, residual, reason = _first_root(
                xy[:, particle, 0], xy[:, particle, 1], radius, tolerance)
            quality[particle] = reason
            if outcome == "survived":
                continue
            status[particle] = outcome
            if outcome == "unresolved":
                unresolved_interval[particle] = nodes[index:index+2]
                continue
            phase = np.polynomial.polynomial.polyval(fraction, coefficients[:, particle])
            radial_residual = abs(float(np.hypot(phase[0], phase[1]))-radius)
            if radial_residual > max(tolerance, 512*_EPS*radius):
                status[particle], quality[particle] = "unresolved", "hit_phase_wall_residual_exceeds_tolerance"
                unresolved_interval[particle] = nodes[index:index+2]
                continue
            hit_z[particle], hit_phase[particle] = start+step*fraction, phase
            root_residual[particle] = radial_residual
    complete = not np.any(status == "unresolved")
    return {
        "schema": "ideal-lens-continuous-wall-v1", "source_ids": identifiers,
        "state_width": state_width, "radius_m": radius, "wall_tolerance_m": tolerance,
        "status": status, "quality": quality, "alive_at_end": status == "survived",
        "physical_loss_mask": (status != "survived") & (status != "unresolved"),
        "unresolved_mask": status == "unresolved",
        "classification_complete": bool(complete), "hit_z_m": hit_z,
        "hit_xy_m": hit_phase[:, :2].copy(), "hit_phase": hit_phase,
        "hit_radial_residual_m": root_residual,
        "unresolved_z_interval_m": unresolved_interval,
        "counterfactual_max_radius_upper_bound_m": maximum_bound,
        "counterfactual_endpoint_max_radius_m": maximum_observed,
        "counterfactual_clearance_lower_bound_m": radius-maximum_bound,
        "dense_intervals": len(interpolants), "polynomial_root_candidates": slow_candidates,
        "scope": "First contact of numerical DOP853 dense polynomials with an absorbing coaxial cylinder; no field feedback or current renormalization",
        "bound_scope": "Floating-point Bernstein convex-hull bound over all retained counterfactual dense intervals, including extensions after contact; not an RK-stage bound",
        "qualification": "Numerical contact classification only; independently refine transport and wall tolerance. Unresolved cases must be rejected, not counted as physical losses.",
    }


__all__ = ("classify_cylindrical_wall",)
