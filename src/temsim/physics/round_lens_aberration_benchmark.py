"""Opt-in frozen-field round-lens ray benchmarks, never a preset calibration.

The measured pupil points OUTWARD, opposite the nominal converging image-ray
slopes. At the fixed nominal paraxial image plane the convention is
``delta_r = -Cs * theta * |theta|^2 + Cc * theta * (delta_E/E0)``.
Thus Cs is the existing wave-gradient C3 coefficient, while Cc is MINUS the
energy derivative of that convention's C1 coefficient. Both are reported;
neither is injected as an additional kick or phase plate.

Only centred, axis-aligned, static magnetic round lenses are admitted here.
Rays start with a declared linear, zero-canonical-momentum paraxial input;
these same mechanical directions and positions remain fixed during energy
scans. There is no energy-by-energy refocus, current change, recentering,
sample, detector, electric field, spin, space charge, or wave qualification.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp

from temsim.immutable_json import json_digest
from temsim.optics.aberration_basis import CARTESIAN_NAMES, predict_displacement_m
from temsim.optics.field_aberrations import fit_wave_gradient, pupil_rays
from .high_order_round_lens import HighOrderRoundLensPotential
from .relativistic_lorentz import ELECTRON, momentum_from_kinetic_energy_ev
from .vector_field_transport import magnetic_slope_derivative


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive finite scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite scalar")
    return value


def _field(field):
    if type(field) is not HighOrderRoundLensPotential:
        raise ValueError("Round-lens benchmark requires an immutable high-order potential")
    if (field.registration.origin_global_m[:2] != (0., 0.)
            or not np.array_equal(field.registration.rotation_array, np.eye(3))):
        raise ValueError("Round-lens aberration reference requires a centred axis-aligned field")
    return field


def _charge_over_momentum(energy_ev):
    energy_ev = _positive(energy_ev, "Kinetic energy in eV")
    p = momentum_from_kinetic_energy_ev(energy_ev, (0., 0., 1.))
    return ELECTRON.charge_c/float(np.linalg.norm(p))


def _interval(field, z_start_m, z_end_m, maximum_step_m):
    field = _field(field)
    if (not np.isfinite((z_start_m, z_end_m)).all() or z_end_m <= z_start_m):
        raise ValueError("Benchmark reference planes must be finite and ordered")
    field.sample([[0., 0., z_start_m], [0., 0., z_end_m]])
    return float(z_start_m), float(z_end_m), _positive(maximum_step_m, "Maximum ray step in metres")


def trace_full_lorentz(field, initial_xy_slopes, *, energy_ev, z_start_m,
                       z_end_m, maximum_step_m, method="DOP853", relative_tolerance=2e-11):
    """Full magnetic Lorentz slope equations at constant relativistic |p|.

    Input and output rows are (x_m, y_m, dx/dz, dy/dz). Z is the independent
    coordinate: this is a forward-ray solver, not a turning-particle solver.
    RK4 permits an independent fixed-step refinement against adaptive DOP853.
    """
    start, stop, step = _interval(field, z_start_m, z_end_m, maximum_step_m)
    k = _charge_over_momentum(energy_ev)
    rtol = _positive(relative_tolerance, "Relative tolerance")
    initial = np.asarray(initial_xy_slopes, float)
    if initial.ndim != 2 or initial.shape[1] != 4 or len(initial) == 0 or not np.isfinite(initial).all():
        raise ValueError("Initial rays must be a nonempty finite N by 4 array")
    field.sample(np.column_stack((initial[:, :2], np.full(len(initial), start))))
    calls = 0

    def rhs(z, flat):
        nonlocal calls
        calls += 1
        state = flat.reshape(-1, 4)
        positions = np.column_stack((state[:, :2], np.full(len(state), z)))
        acceleration = magnetic_slope_derivative((field,), positions, state[:, 2:], k)
        return np.column_stack((state[:, 2:], acceleration)).ravel()

    if method == "DOP853":
        tolerance = np.tile((1e-14, 1e-14, 1e-13, 1e-13), len(initial))
        solution = solve_ivp(rhs, (start, stop), initial.ravel(), method="DOP853",
                             rtol=rtol, atol=tolerance, max_step=step)
        if not solution.success:
            raise ValueError("Full Lorentz reference integration failed: "+solution.message)
        final, intervals = solution.y[:, -1].reshape(-1, 4), len(solution.t)-1
    elif method == "RK4":
        grid = np.linspace(start, stop, max(1, math.ceil((stop-start)/step))+1)
        final = initial.ravel().copy()
        for left, right in zip(grid, grid[1:]):
            h = right-left
            k1 = rhs(left, final)
            k2 = rhs(left+h/2, final+h*k1/2)
            k3 = rhs(left+h/2, final+h*k2/2)
            k4 = rhs(right, final+h*k3)
            final += h*(k1+2*k2+2*k3+k4)/6
        final, intervals = final.reshape(-1, 4), len(grid)-1
    else:
        raise ValueError("Round-lens benchmark method must be DOP853 or RK4")
    if not np.isfinite(final).all():
        raise ValueError("Nonfinite full Lorentz ray result")
    return {"final_xy_slopes": final, "method": method, "intervals": intervals,
            "function_evaluations": calls, "field_digest": field.digest,
            "energy_ev": float(energy_ev), "reference_interval_m": [start, stop]}


def paraxial_focus_reference(field, *, energy_ev, z_start_m, focus_search_end_m,
                             maximum_step_m):
    """Independent Larmor equation h''+(q B0/(2p))^2 h=0 and rotation."""
    start, stop, step = _interval(field, z_start_m, focus_search_end_m, maximum_step_m)
    k = _charge_over_momentum(energy_ev)
    origin = field.registration.origin_global_m[2]

    def rhs(z, state):
        b0 = float(field.axial_derivatives(z-origin)[0])
        return (state[1], -(.5*k*b0)**2*state[0], -.5*k*b0)

    def focus(_z, state):
        return state[0]
    focus.terminal, focus.direction = True, -1
    solution = solve_ivp(rhs, (start, stop), (1., 0., 0.), events=focus,
        method="DOP853", rtol=2e-12, atol=(1e-13, 1e-11, 1e-13), max_step=step)
    if not solution.success or len(solution.t_events[0]) != 1:
        raise ValueError("No first paraxial focus found inside the declared reference interval")
    z, (_, slope, angle) = float(solution.t_events[0][0]), solution.y_events[0][0]
    if abs(slope) < 1e-12:
        raise ValueError("Paraxial focus has a singular pupil mapping")
    rotation = np.array(((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))))
    b_start = float(field.axial_derivatives(start-origin)[0])
    initial_slope = np.array(((0., .5*k*b_start), (-.5*k*b_start, 0.)))
    return {"image_z_m": z, "larmor_angle_rad": float(angle), "radial_slope_per_m": float(slope),
        "image_slopes_per_initial_position_m1": slope*rotation,
        "initial_slopes_per_position_m1": initial_slope,
        "initial_condition": "h=1,h_prime=0; mechanical slope matrix fixed at reference energy",
        "field_digest": field.digest}


def _linear_cubic_reference(field, energy_ev, start, stop, step, initial_slopes):
    """Independent first/third-order expansion of the full Lorentz equations.

    Expand x=epsilon*x1+epsilon^3*x3, and similarly y,u,v. The forcing below
    is derived explicitly from Bz=B0-r^2 B0''/4, Br=-r B0'/2+r^3 B0'''/16,
    and sqrt(1+u^2+v^2). It does not evaluate the full ray RHS or fit rays.
    """
    k = _charge_over_momentum(energy_ev)
    origin = field.registration.origin_global_m[2]

    def rhs(z, state):
        x, y, u, v, x3, y3, u3, v3 = state
        b, d, dd, ddd = field.axial_derivatives(z-origin, 3)
        if field.radial_order == 1:
            dd = ddd = 0.
        r2, s2 = x*x+y*y, u*u+v*v
        du = k*(v*b+.5*y*d)
        dv = k*(-u*b-.5*x*d)
        fu = k*(-v*r2*dd/4-y*r2*ddd/16+.5*d*(u*u*y-u*v*x)+.5*s2*(v*b+.5*y*d))
        fv = k*(u*r2*dd/4+x*r2*ddd/16+.5*d*(-v*v*x+u*v*y)+.5*s2*(-u*b-.5*x*d))
        return (u, v, du, dv, u3, v3,
                k*(v3*b+.5*y3*d)+fu, k*(-u3*b-.5*x3*d)+fv)

    initial = (1., 0., initial_slopes[0, 0], initial_slopes[1, 0], 0., 0., 0., 0.)
    solution = solve_ivp(rhs, (start, stop), initial, method="DOP853", rtol=2e-12,
                         atol=1e-13, max_step=step)
    if not solution.success:
        raise ValueError("Independent order-by-order ray reference failed: "+solution.message)
    return solution.y[:, -1]


def third_order_reference(field, *, energy_ev, z_start_m, focus_search_end_m,
                           maximum_step_m, energy_relative_step=.0005):
    """Independent perturbation-ODE Cs and linear frozen-hardware Cc, in mm."""
    focus = paraxial_focus_reference(field, energy_ev=energy_ev, z_start_m=z_start_m,
        focus_search_end_m=focus_search_end_m, maximum_step_m=maximum_step_m)
    delta = _positive(energy_relative_step, "Relative energy step")
    if delta >= .1:
        raise ValueError("Relative energy reference step must be below 0.1")
    args = (z_start_m, focus["image_z_m"], maximum_step_m, focus["initial_slopes_per_position_m1"])
    central = _linear_cubic_reference(field, energy_ev, *args)
    pupil = -focus["image_slopes_per_initial_position_m1"][:, 0]
    scale = float(np.linalg.norm(pupil))
    direction = pupil/scale
    cs = -float(np.dot(central[4:6], direction))/scale**3
    minus = _linear_cubic_reference(field, energy_ev*(1-delta), *args)
    plus = _linear_cubic_reference(field, energy_ev*(1+delta), *args)
    cc = float(np.dot((plus[:2]-minus[:2])/(2*delta), direction))/scale
    return {"cs_mm": cs*1e3, "cc_mm": cc*1e3, "image_z_m": focus["image_z_m"],
        "linear_focus_residual_per_initial_height": central[:2].tolist(),
        "field_digest": field.digest, "energy_relative_step": delta,
        "method": "independent first/third-order Lorentz perturbation ODE; no finite-pupil fit"}


def measure_round_lens_aberrations(field, *, energy_ev, z_start_m, focus_search_end_m,
        maximum_step_m, semiangle_rad=.01, energy_relative_steps=(.002, .001, .0005),
        method="DOP853", relative_tolerance=2e-11):
    """Finite-pupil Cs/Cc with holdouts and a fixed field, rays and image plane."""
    _field(field)
    alpha = _positive(semiangle_rad, "Pupil semiangle")
    if alpha > .1:
        raise ValueError("Round-lens fit pupil exceeds the declared 100 mrad benchmark scope")
    deltas = np.asarray(energy_relative_steps, float)
    if (deltas.ndim != 1 or not 3 <= len(deltas) <= 6 or not np.isfinite(deltas).all()
            or np.any(deltas < 1e-7) or np.any(deltas > .01) or np.any(np.diff(deltas) >= 0)):
        raise ValueError("Cc scan needs 3-6 decreasing relative energy steps in [1e-7, 0.01]")
    immutable_receipt = json_digest(field.to_dict())
    focus = paraxial_focus_reference(field, energy_ev=energy_ev, z_start_m=z_start_m,
        focus_search_end_m=focus_search_end_m, maximum_step_m=maximum_step_m)
    training, holdout = pupil_rays(alpha), pupil_rays(alpha, holdout=True)
    pupil = np.vstack((training, holdout))
    positions = np.linalg.solve(-focus["image_slopes_per_initial_position_m1"], pupil.T).T
    slopes = positions@focus["initial_slopes_per_position_m1"].T
    inputs = np.vstack((np.zeros((1, 4)), np.column_stack((positions, slopes))))
    input_digest = json_digest(inputs.tolist())
    traces = []

    def intercept(energy):
        trace = trace_full_lorentz(field, inputs, energy_ev=energy, z_start_m=z_start_m,
            z_end_m=focus["image_z_m"], maximum_step_m=maximum_step_m, method=method,
            relative_tolerance=relative_tolerance)
        traces.append({key: value for key, value in trace.items() if key != "final_xy_slopes"})
        # The raw chief intercept is retained; no alignment/recentering is done.
        return trace["final_xy_slopes"][:, :2]

    central = intercept(energy_ev)
    coefficients, fit_rms, condition = fit_wave_gradient(training, central[1:1+len(training)], alpha)
    prediction = predict_displacement_m(holdout, coefficients)
    error = prediction-central[1+len(training):]
    cc, chromatic_c1 = [], []
    for delta in deltas:
        derivative = (intercept(energy_ev*(1+delta))-intercept(energy_ev*(1-delta)))/(2*delta)
        fitted, _, _ = fit_wave_gradient(training, derivative[1:1+len(training)], alpha)
        chromatic_c1.append(float(fitted[0]))
        cc.append(-float(fitted[0]))
    if json_digest(field.to_dict()) != immutable_receipt or json_digest(inputs.tolist()) != input_digest:
        raise ValueError("Frozen field or fixed input rays changed during aberration measurement")
    comparisons = [{"relative_energy_step": float(delta), "absolute_change_mm": abs(value-previous)}
        for delta, previous, value in zip(deltas[1:], cc, cc[1:])]
    return {"schema": "frozen-round-lens-aberration-benchmark-v1", "field_digest": field.digest,
        "field": field.to_metadata(), "input_rays_digest": input_digest,
        "reference_energy_ev": float(energy_ev), "reference_interval_m": [float(z_start_m), focus["image_z_m"]],
        "focus": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in focus.items()},
        "semiangle_rad": alpha, "training_rays": len(training), "holdout_rays": len(holdout),
        "chief_intercept_m": central[0].tolist(), "cs_mm": float(coefficients[CARTESIAN_NAMES.index("C3")]),
        "cc_mm": cc[-1], "cartesian_names": list(CARTESIAN_NAMES), "coefficients_mm": coefficients.tolist(),
        "fit_rms_m": fit_rms, "fit_condition_number": condition,
        "holdout_rms_m": float(np.sqrt(np.mean(error*error))),
        "holdout_max_error_m": float(np.linalg.norm(error, axis=1).max()),
        "energy_scan": {"relative_steps": deltas.tolist(), "cc_mm": cc,
            "wave_c1_energy_derivative_mm": chromatic_c1, "comparisons": comparisons,
            "fixed_field": True, "fixed_initial_rays": True, "fixed_image_plane": True},
        "integration": traces, "applied_manual_aberrations": False,
        "sign_convention": "outward pupil; displacement=-Cs*theta*|theta|^2+Cc*theta*(delta_E/E0)",
        "convergence_status": "NOT_ASSESSED; vary pupil, field order and integration independently",
        "scope": "Single frozen analytical magnetic lens; synthetic numerical evidence, not OEM or wave validation"}


__all__ = ("trace_full_lorentz", "paraxial_focus_reference", "third_order_reference",
           "measure_round_lens_aberrations")
