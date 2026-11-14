"""Classical ideal-lens research utilities with explicit source and wall ledgers.

The retained dense trajectories are counterfactual vacuum characteristics.
Only their prefix before each passive absorbing wall is physical. No space
charge, material feedback, empirical aberration, specimen or wave is used.
"""
from __future__ import annotations

from hashlib import sha256
import math

import numpy as np
from scipy.integrate import OdeSolution, solve_ivp

from temsim.optics.aberration_basis import CARTESIAN_NAMES, predict_displacement_m
from temsim.optics.field_aberrations import fit_wave_gradient
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.relativistic_lorentz import ELECTRON, momentum_from_kinetic_energy_ev, velocity_from_momentum_m_per_s
from temsim.physics.vector_field_transport import magnetic_slope_derivative


def array_hash(value):
    a = np.ascontiguousarray(value)
    return sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def radical_inverse(indices, base):
    values = np.asarray(indices, dtype=np.int64).copy()
    result = np.zeros(values.shape)
    factor = 1.0/base
    while np.any(values):
        result += factor*(values % base)
        values //= base
        factor /= base
    return result


def source_population(count, radius_m, initial_slope_matrix, energy_ev=300000.):
    """Fixed inverse CDF; growing count appends without refitting any sample."""
    if type(count) is not int or count < 1000:
        raise ValueError("Use at least 1000 original positive-current particles")
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Source radius must be a finite positive length")
    matrix = np.asarray(initial_slope_matrix, dtype=float)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("Initial slope matrix must be finite and 2 by 2")
    if not np.isfinite(energy_ev) or energy_ev <= 0:
        raise ValueError("Source kinetic energy must be finite and positive")
    ids = np.arange(count, dtype=np.int64)
    u, phi = radical_inverse(ids+1, 2), 2*np.pi*radical_inverse(ids+1, 3)
    xy = radius_m*np.sqrt(u)[:, None]*np.column_stack((np.cos(phi), np.sin(phi)))
    slope = xy@matrix.T
    phase = np.column_stack((xy, slope, np.zeros(count)))
    direction = np.column_stack((slope, np.ones(count)))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    return dict(source_id=ids, initial_phase=phase, emission_direction=direction,
        kinetic_energy_ev=np.full(count, energy_ev), emission_time_s=np.zeros(count),
        training=radical_inverse(ids+1, 5) < .5)


def gaussian_field(peak_t, sigma_m, order=9, valid_radius_m=.004):
    native = FrozenAnalyticField("ideal-gaussian-round", ((float(peak_t), 0., float(sigma_m)),),
        CoordinateRegistration(), (-8*sigma_m*1e3, 8*sigma_m*1e3), valid_radius_m)
    return HighOrderRoundLensPotential(native, order, valid_radius_m, "gaussian_all_z")


def propagate(field, initial_phase, *, energy_ev, start_m, end_m, sigma_m,
              step_over_sigma=1/16, relative_tolerance=2e-11):
    """Complete forward relativistic magnetic Lorentz equations plus time.

    Far-tail intervals permit longer steps, while still evaluating the SAME
    all-Z Gaussian at every stage. No hard field cutoff or drift substitute.
    """
    initial = np.asarray(initial_phase, float)
    if initial.ndim != 2 or initial.shape[1] != 5 or len(initial) < 1000:
        raise ValueError("A partition needs at least 1000 original N by 5 phases")
    if not np.isfinite(initial).all() or not (end_m > start_m and sigma_m > 0 and step_over_sigma > 0):
        raise ValueError("Invalid research interval, step or source phase")
    p = momentum_from_kinetic_energy_ev(energy_ev, (0., 0., 1.))
    speed = float(np.linalg.norm(velocity_from_momentum_m_per_s(p)))
    charge_over_p = ELECTRON.charge_c/float(np.linalg.norm(p))
    count = len(initial)

    def rhs(z, flat):
        phase = flat.reshape(count, 5)
        xyz = np.column_stack((phase[:, :2], np.full(count, z)))
        acceleration = magnetic_slope_derivative((field,), xyz, phase[:, 2:4], charge_over_p)
        time_rate = np.sqrt(1+np.sum(phase[:, 2:4]**2, axis=1))/speed
        return np.column_stack((phase[:, 2:4], acceleration, time_rate)).ravel()

    splits = sorted(set([start_m, end_m]+[x for x in (-8*sigma_m, 8*sigma_m) if start_m < x < end_m]))
    times, interpolants, current, receipts = [], [], initial.ravel(), []
    for start, stop in zip(splits[:-1], splits[1:]):
        tail = stop <= -8*sigma_m or start >= 8*sigma_m
        max_step = 4*sigma_m if tail else step_over_sigma*sigma_m
        result = solve_ivp(rhs, (start, stop), current, method="DOP853", rtol=relative_tolerance,
            atol=np.tile((1e-14, 1e-14, 1e-13, 1e-13, 1e-21), count),
            max_step=max_step, dense_output=True)
        if not result.success or not np.isfinite(result.y).all():
            raise ValueError("Full Lorentz propagation failed: "+result.message)
        if not times:
            times.extend(result.sol.ts)
        else:
            times.extend(result.sol.ts[1:])
        interpolants.extend(result.sol.interpolants)
        current = result.y[:, -1]
        receipts.append(dict(start_m=start, end_m=stop, max_step_m=max_step,
            intervals=len(result.t)-1, evaluations=result.nfev, far_tail=tail))
    return OdeSolution(times, interpolants), dict(method="DOP853", rtol=relative_tolerance,
        atol_phase=[1e-14, 1e-14, 1e-13, 1e-13, 1e-21], segments=receipts,
        energy_ev=energy_ev, constant_speed_m_per_s=speed, field_digest=field.digest,
        counterfactual_after_contact=True, empirical_aberrations_applied=False)


def phase_statistics(phase, alive, weights, source_current_a):
    alive = np.asarray(alive, bool)
    if not alive.any():
        return dict(surviving_positive=0, effective=0., current_a=0., status="NO_TRANSMISSION")
    p = np.asarray(phase)
    s = transverse_beam_statistics(p[:, 0], p[:, 1], p[:, 2], p[:, 3], weights=weights, alive=alive)
    w = np.asarray(weights)[alive]
    return dict(status="MEASURED", surviving_positive=int(alive.sum()),
        effective=float(w.sum()**2/np.sum(w*w)), current_a=float(source_current_a*w.sum()),
        transmission=float(w.sum()), d95_nm=float(2*s.radius_95_m*1e9),
        rms_radius_nm=float(s.radius_rms_m*1e9), alpha95_mrad=float(s.convergence_95_rad*1e3),
        angular_rms_mrad=float(s.convergence_rms_rad*1e3), centroid_m=[s.mean_x_m, s.mean_y_m])


def spherical_fit(initial_phase, at_image, training, outward_pupil_matrix, edge_slope, cs_reference_mm):
    """Fit all PREDECLARED vacuum characteristics, never wall-selected rays."""
    pupil = np.asarray(initial_phase)[:, :2]@np.asarray(outward_pupil_matrix).T
    training = np.asarray(training, bool)
    coefficients, rms, condition = fit_wave_gradient(pupil[training], at_image[training, :2], edge_slope)
    prediction = predict_displacement_m(pupil[~training], coefficients)
    residual = prediction-at_image[~training, :2]
    cs = float(coefficients[CARTESIAN_NAMES.index("C3")])
    c5 = float(coefficients[CARTESIAN_NAMES.index("C5")])
    radii = np.linalg.norm(at_image[:, :2], axis=1)
    cubic = -cs_reference_mm*1e-3*pupil*np.sum(pupil*pupil, axis=1)[:, None]
    norm = np.linalg.norm(pupil, axis=1)
    direction = pupil/norm[:, None]
    q = norm/edge_slope
    radial = -np.sum(at_image[:, :2]*direction, axis=1)
    geometric = []
    for powers in ((1, 3, 5), (1, 3, 5, 7)):
        design = np.column_stack([q**power for power in powers])
        c, _, rank, singular = np.linalg.lstsq(design[training], radial[training], rcond=1e-12)
        if rank != len(powers):
            raise ValueError("Axisymmetric radial fit is rank deficient")
        error = design[~training]@c-radial[~training]
        geometric.append(dict(powers=list(powers), coefficients_mm=(c/edge_slope**np.asarray(powers)*1e3).tolist(),
            cs_mm=float(c[1]/edge_slope**3*1e3), relative_cs_difference=abs(float(c[1]/edge_slope**3*1e3)/cs_reference_mm-1),
            holdout_rms_m=float(np.sqrt(np.mean(error**2))), condition_number=float(singular[0]/singular[-1])))
    return dict(cs_fit_mm=cs, c5_fit_mm=c5, cs_reference_mm=cs_reference_mm,
        relative_cs_difference=abs(cs/cs_reference_mm-1),
        cs_reference_agreement=abs(cs/cs_reference_mm-1) <= .001,
        fit_rms_m=rms, condition_number=condition,
        holdout_rms_m=float(np.sqrt(np.mean(residual**2))),
        holdout_max_m=float(np.linalg.norm(residual, axis=1).max()),
        training_count=int(training.sum()), holdout_count=int((~training).sum()),
        coefficients_mm=coefficients.tolist(), coefficient_names=list(CARTESIAN_NAMES),
        paraxial_edge_slope_rad=edge_slope, paraxial_edge_angle_mrad=float(np.arctan(edge_slope)*1e3),
        cubic_edge_transverse_error_nm=float(abs(cs_reference_mm*1e-3*edge_slope**3)*1e9),
        maximum_actual_axis_radius_nm=float(radii.max()*1e9),
        higher_order_departure_max_nm=float(np.linalg.norm(at_image[:, :2]-cubic, axis=1).max()*1e9),
        radial_geometric_fits=geometric,
        population="Complete declared counterfactual vacuum source; no intercepted-ray deletion",
        higher_order_coefficient_scope="Finite-pupil diagnostics; C5 is not independently qualified by a converged third-order reference")


def field_order_evidence(peak_t, sigma_m, beam_radius_m):
    """Compare series only in the declared beam cylinder, not an unused tube."""
    z = np.linspace(-8*sigma_m, 8*sigma_m, 257)
    radii = np.linspace(0., beam_radius_m, 33)
    zz, rr = np.meshgrid(z, radii)
    points = np.column_stack((rr.ravel(), np.zeros(rr.size), zz.ravel()))
    values, rows = {}, []
    for order in (3, 5, 7, 9):
        f = gaussian_field(peak_t, sigma_m, order)
        sampled = f.sample(points)
        values[order] = sampled
        residual = f.maxwell_residual(points)["curl_B_t_per_m"]
        rows.append(dict(order=order,
            normalized_curl_residual=float(sigma_m*np.linalg.norm(residual, axis=1).max()/peak_t)))
    for row in rows:
        row["relative_B_difference_to_order9"] = float(np.linalg.norm(values[row["order"]]["B_t"]-values[9]["B_t"], axis=1).max()/peak_t)
    return dict(scope="Fixed cylindrical sample grid within the represented beam radius; sampled estimates, not a rigorous global error bound",
        tested_radius_m=beam_radius_m, grid_shape=[len(radii), len(z)], orders=rows,
        order7_vs9_relative_B=rows[2]["relative_B_difference_to_order9"],
        order9_normalized_curl=rows[3]["normalized_curl_residual"],
        sampled_field_gates_passed=rows[2]["relative_B_difference_to_order9"] <= 1e-4 and rows[3]["normalized_curl_residual"] <= 1e-3)
