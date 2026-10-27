"""Opt-in canonical-pupil diagnostics of immutable static magnetic potentials.

P_perp/p_ref=(p*u_perp+q*A_perp)/p_ref is used at BOTH reference planes.
The input Lagrangian family, nominal transfer, reference momentum, gauge and
image plane are captured once. No image recentering, tilt removal, refocus,
aberration kick, hardware calibration or empirical coefficient is applied.

The canonical energy scan keeps physical canonical input momenta fixed. Its
Cc_response is explicitly NOT the field-free fixed-geometrical-angle Cc
convention when the entrance is immersed in a magnetic field. A separate
fixed-mechanical scan can measure geometrical chromatic response, but is not
automatically a family of scalar wavefronts in that immersed entrance.

Coefficients are diagnostic only. Finite-pupil residuals and canonical
symplectic checks can reject a scalar-wavefront interpretation. This module
does not propagate a coherent wave or certify a real instrument.
"""
from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy.constants import c, e, m_e

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.optics.aberration_basis import CARTESIAN_NAMES, POWERS, polynomial_basis
from temsim.optics.field_aberrations import pupil_rays
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


REFERENCE_SCHEMA = "canonical-potential-pupil-v1"


def _positive(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a positive finite real value")
    return float(value)


def momentum(energy_ev):
    raw = np.asarray(energy_ev)
    if raw.dtype.kind not in "iuf":
        raise ValueError("Kinetic energy must contain real numerical values")
    energy = np.asarray(raw, dtype=float)*e
    if not np.isfinite(energy).all() or np.any(energy <= 0):
        raise ValueError("Canonical momentum requires positive finite kinetic energy")
    return np.sqrt(energy*(energy+2*m_e*c*c))/c


def _snapshot(snapshot):
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("Canonical pupil requires an immutable electromagnetic potential snapshot")
    if any(row.kind != "magnetic" for row in snapshot.contributions):
        raise ValueError("This canonical diagnostic currently admits static magnetic potentials only")
    return snapshot


def canonical_coordinates(snapshot, mechanical, z_m, energy_ev, reference_momentum):
    """Convert raw (x,y,tx,ty) to (x,y,Px/p_ref,Py/p_ref), without centering."""
    _snapshot(snapshot)
    rays = np.asarray(mechanical)
    if rays.dtype.kind not in "iuf" or rays.ndim != 2 or rays.shape[1] != 4 or not np.isfinite(rays).all():
        raise ValueError("Mechanical ray coordinates must be finite N by 4")
    rays = np.asarray(rays, dtype=float)
    reference_momentum = _positive(reference_momentum, "Reference momentum")
    points = np.c_[rays[:, :2], np.full(len(rays), float(z_m))]
    potential = snapshot.sample(points)["A_t_m"][:, :2]
    directions = rays[:, 2:]/np.sqrt(1+np.sum(rays[:, 2:]**2, axis=1))[:, None]
    p = np.broadcast_to(momentum(energy_ev), (len(rays),))
    return np.c_[rays[:, :2], (p[:, None]*directions-e*potential)/reference_momentum]


def mechanical_coordinates(snapshot, canonical, z_m, energy_ev, reference_momentum):
    """Invert the same finite-energy canonical chart on its forward branch."""
    _snapshot(snapshot)
    rays = np.asarray(canonical)
    if rays.dtype.kind not in "iuf" or rays.ndim != 2 or rays.shape[1] != 4 or not np.isfinite(rays).all():
        raise ValueError("Canonical ray coordinates must be finite N by 4")
    rays = np.asarray(rays, dtype=float)
    reference_momentum = _positive(reference_momentum, "Reference momentum")
    points = np.c_[rays[:, :2], np.full(len(rays), float(z_m))]
    potential = snapshot.sample(points)["A_t_m"][:, :2]
    p = np.broadcast_to(momentum(energy_ev), (len(rays),))
    directions = (reference_momentum*rays[:, 2:]+e*potential)/p[:, None]
    norm2 = np.sum(directions*directions, axis=1)
    if np.any(norm2 >= 1.):
        raise ValueError("Canonical pupil leaves the forward momentum hemisphere")
    return np.c_[rays[:, :2], directions/np.sqrt(1-norm2)[:, None]]


def capture_canonical_pupil(snapshot, z_m, *, energy_ev, semiangle_rad=.01,
                            maximum_step_m=.00025, stop_planes=(),
                            transfer_position_step_m=1e-8, transfer_momentum_step=1e-6,
                            transfer_refinement_factors=(1., .1, .01, .001),
                            method="DOP853", sampling_backend="snapshot"):
    """Freeze an inverse first-order canonical image-pupil construction.

    Central differences at independent perturbation sizes are traced in
    one batch, using the full Lorentz equations. The unmodified smaller-step
    matrix is retained; it is never projected onto a symplectic matrix.
    """
    _snapshot(snapshot)
    nodes = np.asarray(z_m)
    if (nodes.dtype.kind not in "iuf" or nodes.ndim != 1 or len(nodes) < 2
            or not np.isfinite(nodes).all() or not np.all(np.diff(nodes) > 0)):
        raise ValueError("Canonical reference planes must increase")
    nodes = np.asarray(nodes, dtype=float)
    energy_ev = _positive(energy_ev, "Reference energy")
    semiangle_rad = _positive(semiangle_rad, "Canonical pupil radius")
    if semiangle_rad > .1:
        raise ValueError("Canonical pupil radius must be in (0, 0.1]")
    sizes = np.array([_positive(transfer_position_step_m, "Position perturbation")]*2
                     +[_positive(transfer_momentum_step, "Momentum perturbation")]*2)
    factors = np.asarray(transfer_refinement_factors)
    if (factors.dtype.kind not in "iuf" or factors.ndim != 1 or len(factors) < 2
            or not np.isfinite(factors).all() or np.any(factors <= 0) or np.any(factors > 1.)
            or not np.all(np.diff(factors) < 0) or np.any(sizes*factors[-1] == 0.)):
        raise ValueError("Transfer refinement factors must strictly decrease in (0, 1]")
    reference_p = float(momentum(energy_ev))
    perturbations = np.vstack([np.zeros((1, 4)),
        *(item for factor in factors for item in (np.diag(sizes*factor), -np.diag(sizes*factor)))])
    initial = mechanical_coordinates(snapshot, perturbations, nodes[0], energy_ev, reference_p)
    position_tolerance = float(min(sizes[:2])*factors[-1]*1e-9)
    momentum_tolerance = float(min(sizes[2:])*factors[-1]*1e-9)
    traced = trace_potential_rays(snapshot, initial, nodes, energy_ev=energy_ev,
        maximum_step_m=maximum_step_m, method=method, stop_planes=stop_planes,
        relative_tolerance=2e-12, position_tolerance_m=position_tolerance,
        slope_tolerance_rad=momentum_tolerance, sampling_backend=sampling_backend)
    if not traced.alive[-1].all():
        raise ValueError("Apertures intercept the infinitesimal canonical reference family")
    final = canonical_coordinates(snapshot, traced.xy_slopes[-1], nodes[-1], energy_ev, reference_p)
    matrices = [(final[1+8*i:5+8*i]-final[5+8*i:9+8*i]).T/(2*sizes*factor)
                for i, factor in enumerate(factors)]
    matrix = matrices[-1]
    units = np.array([1e-6, 1e-6, 1e-3, 1e-3])
    scaled = [value*units[None, :]/units[:, None] for value in matrices]
    symplectic = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    defects = [float(np.linalg.norm(item.T@symplectic@item-symplectic)/np.linalg.norm(symplectic)) for item in scaled]
    defect = defects[-1]
    changes = [float(np.linalg.norm(current-previous)/max(np.linalg.norm(current), 1.))
               for previous, current in zip(scaled, scaled[1:])]
    transfer_change = changes[-1]
    condition = float(np.linalg.cond(scaled[-1]))
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError("Canonical transfer is too ill-conditioned for inverse pupil construction")
    train, holdout = pupil_rays(semiangle_rad), pupil_rays(semiangle_rad, holdout=True)
    pupil = np.vstack((train, holdout))
    sheet = np.linalg.solve(matrix, np.vstack((np.zeros((2, 2)), np.eye(2))))
    sheet_form = sheet[:2].T@sheet[2:]-sheet[2:].T@sheet[:2]
    sheet_defect = float(np.linalg.norm(sheet_form)/max(np.linalg.norm(sheet[:2])*np.linalg.norm(sheet[2:]), 1e-30))
    inputs = np.vstack((np.zeros((1, 4)), pupil@sheet.T))
    entrance = mechanical_coordinates(snapshot, inputs, nodes[0], energy_ev, reference_p)
    entrance_field = snapshot.sample(np.array([0., 0., nodes[0]]))
    pupil_entrance_field = snapshot.sample(np.c_[inputs[:, :2], np.full(len(inputs), nodes[0])])
    exit_field = snapshot.sample(np.array([*final[0, :2], nodes[-1]]))
    source_position_condition = float(np.linalg.cond(sheet[:2]))
    source_curvature = (np.linalg.solve(sheet[:2].T, sheet[2:].T).T
                        if np.isfinite(source_position_condition) and source_position_condition < 1e10 else None)
    record = {"schema": REFERENCE_SCHEMA, "snapshot_digest": snapshot.numerical_digest,
        "gauge_digest": json_digest(snapshot.gauge.to_dict()), "energy_ev": float(energy_ev),
        "reference_momentum_kg_m_s": reference_p, "semiangle_rad": float(semiangle_rad),
        "z_m": nodes.tolist(), "stop_planes": [list(row) for row in stop_planes],
        "input_canonical": inputs.tolist(), "input_mechanical_reference_energy": entrance.tolist(),
        "training_count": len(train), "holdout_count": len(holdout),
        "training_reference_pupil": train.tolist(), "holdout_reference_pupil": holdout.tolist(),
        "transfer_matrix": matrix.tolist(), "transfer_condition": condition,
        "transfer_perturbation_steps": sizes.tolist(), "relative_transfer_change": transfer_change,
        "transfer_refinement_factors": factors.tolist(), "transfer_relative_changes": changes,
        "transfer_scaled_symplectic_defects": defects,
        "scaled_symplectic_defect": defect, "nominal_chief_canonical": final[0].tolist(),
        "input_lagrangian_relative_defect": sheet_defect,
        "entrance_axis_A_t_m": entrance_field["A_t_m"].tolist(),
        "entrance_axis_B_t": entrance_field["B_t"].tolist(),
        "entrance_pupil_A_t_m": pupil_entrance_field["A_t_m"].tolist(),
        "entrance_pupil_B_t": pupil_entrance_field["B_t"].tolist(),
        "source_canonical_curvature_m1": None if source_curvature is None else source_curvature.tolist(),
        "source_position_chart_condition": source_position_condition if np.isfinite(source_position_condition) else None,
        "source_phase_scope": "candidate quadratic S_in/p_ref=0.5*x^T*F*x; no symmetrization; validity requires the recorded Lagrangian check",
        "exit_chief_A_t_m": exit_field["A_t_m"].tolist(), "exit_chief_B_t": exit_field["B_t"].tolist(),
        "maximum_step_m": float(maximum_step_m),
        "method": method, "transfer_position_tolerance_m": position_tolerance,
        "transfer_slope_tolerance_rad": momentum_tolerance,
        "sampling_backend": sampling_backend,
        "actual_sampling_backend": traced.metadata["potential_sampling_backend"],
        "input_family": "candidate fixed canonical focusing plane from inverse nominal transfer; Lagrangian defect checked separately",
        "chart": "x,y in m; P_perp/p_ref=(p*u_perp-e*A_perp)/p_ref; reference momentum fixed",
        "reference_was_symplectified": False, "hardware_was_modified": False}
    return freeze_json({**record, "digest": json_digest(record)})


def _reference(reference, snapshot):
    if not isinstance(reference, Mapping) or reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError("A captured canonical pupil reference is required")
    record = thaw_json(reference)
    identity = record.pop("digest", None)
    if identity != json_digest(record) or reference["gauge_digest"] != json_digest(snapshot.gauge.to_dict()):
        raise ValueError("Canonical reference identity or common gauge mismatch")
    return reference


def _gradient_fit(pupil, positions, scale):
    gradient = polynomial_basis(np.asarray(pupil)/scale, gradient=True)
    constants = np.broadcast_to(np.eye(2), (len(pupil), 2, 2))
    gradient = np.concatenate((constants, gradient), axis=2)
    design = np.concatenate((gradient[:, 0], gradient[:, 1]))
    target = -np.asarray(positions).T.ravel()*scale
    coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=1e-11)
    if rank != len(CARTESIAN_NAMES)+2:
        raise ValueError("Canonical scalar-gradient fit is rank deficient")
    prediction = -(design@coefficients).reshape(2, len(pupil)).T/scale
    return coefficients, prediction, float(singular[0]/singular[-1])


def fit_canonical_intercepts(snapshot, reference, mechanical, *, energy_ev=None):
    """Fit already traced, ordered chief/training/holdout rays at the fixed plane.

    This can examine an explicitly verified unchanged-prefix continuation. It
    does not itself establish the provenance or Lagrangian nature of supplied
    rays, so no scalar-wavefront qualification is granted by this fit alone.
    """
    _snapshot(snapshot)
    _reference(reference, snapshot)
    raw = np.asarray(mechanical)
    count = 1+reference["training_count"]+reference["holdout_count"]
    if raw.shape != (count, 4):
        raise ValueError("Canonical fit needs the ordered captured chief, training and holdout rays")
    energy = reference["energy_ev"] if energy_ev is None else energy_ev
    canonical = canonical_coordinates(snapshot, raw, reference["z_m"][-1], energy,
                                       reference["reference_momentum_kg_m_s"])
    train, scale = reference["training_count"], reference["semiangle_rad"]
    coefficients, prediction, condition = _gradient_fit(canonical[1:1+train, 2:], canonical[1:1+train, :2], scale)
    gradient = polynomial_basis(canonical[1+train:, 2:]/scale, gradient=True)
    gradient = np.concatenate((np.broadcast_to(np.eye(2), (len(gradient), 2, 2)), gradient), axis=2)
    holdout = -np.einsum("nij,j->ni", gradient, coefficients)/scale
    return {"coefficients_mm": coefficients[2:]/scale**POWERS*1000,
        "constant_gradient_intercept_m": -coefficients[:2]/scale,
        "gradient_training_rms_m": float(np.sqrt(np.mean((prediction-canonical[1:1+train, :2])**2))),
        "gradient_holdout_rms_m": float(np.sqrt(np.mean((holdout-canonical[1+train:, :2])**2))),
        "gradient_condition": condition, "raw_output_canonical": canonical,
        "scalar_wavefront_qualification": "not granted by an intercept-only fit"}


def measure_canonical_aberrations(snapshot, reference, *, energy_relative_steps=(.002, .001, .0005),
                                  maximum_step_m=None, input_convention="fixed_canonical",
                                  gradient_error_budget_m=1e-9, symplectic_defect_budget=1e-6,
                                  transfer_change_budget=1e-7, retain_nominal_trajectory=False,
                                  sampling_backend=None):
    """Retain raw rays and report a held-out canonical gradient diagnostic.

    The two constant-gradient columns retain chief/tilt; C1 is fitted and
    retained. No fitted term is removed from a ray or a physical potential.
    A different snapshot is allowed deliberately for fixed-reference hardware
    comparisons; it is identified explicitly and never treated as the original
    nominal field. Each energy scan still freezes its CURRENT snapshot. Empty
    energy steps request a nominal-only diagnostic and grant no Cc qualification.
    """
    _snapshot(snapshot)
    _reference(reference, snapshot)
    for value, name in ((gradient_error_budget_m, "Gradient error budget"),
                        (symplectic_defect_budget, "Symplectic defect budget"),
                        (transfer_change_budget, "Transfer change budget")):
        _positive(value, name)
    if input_convention not in {"fixed_canonical", "fixed_mechanical"}:
        raise ValueError("Choose an explicit canonical or mechanical energy-scan input family")
    if type(retain_nominal_trajectory) is not bool:
        raise ValueError("Nominal trajectory retention must be a boolean")
    raw_steps = np.asarray(energy_relative_steps)
    if raw_steps.dtype.kind not in "iuf":
        raise ValueError("Energy scan steps must contain real numerical values")
    steps = np.asarray(raw_steps, dtype=float)
    if (steps.ndim != 1 or not np.isfinite(steps).all()
            or np.any(steps <= 0) or np.any(steps > .01) or np.any(np.diff(steps) >= 0)):
        raise ValueError("Energy scan steps must decrease in (0, 0.01]")
    deltas = np.r_[0., np.column_stack((-steps, steps)).ravel()]
    canonical = np.asarray(reference["input_canonical"])
    count = len(canonical)
    energy = np.repeat(reference["energy_ev"]*(1+deltas), count)
    nodes = np.asarray(reference["z_m"])
    p_ref = reference["reference_momentum_kg_m_s"]
    if input_convention == "fixed_canonical":
        inputs = mechanical_coordinates(snapshot, np.tile(canonical, (len(deltas), 1)), nodes[0], energy, p_ref)
    else:
        inputs = np.tile(np.asarray(reference["input_mechanical_reference_energy"]), (len(deltas), 1))
    result = trace_potential_rays(snapshot, inputs, nodes, energy_ev=energy,
        maximum_step_m=reference["maximum_step_m"] if maximum_step_m is None else maximum_step_m,
        method=reference["method"], stop_planes=tuple(tuple(row) for row in reference["stop_planes"]),
        relative_tolerance=2e-11, position_tolerance_m=1e-15, slope_tolerance_rad=1e-13,
        sampling_backend=reference["sampling_backend"] if sampling_backend is None else sampling_backend)
    if not result.alive[-1].all():
        raise ValueError("Physical apertures intercept the controlled canonical pupil; reduce its declared radius")
    final = result.xy_slopes[-1]
    canon = canonical_coordinates(snapshot, final, nodes[-1], energy, p_ref).reshape(len(deltas), count, 4)
    raw = final.reshape(len(deltas), count, 4)
    train = reference["training_count"]
    scale = reference["semiangle_rad"]
    fits, holdout_errors, fit_rms, conditions = [], [], [], []
    for row in canon:
        coefficients, prediction, condition = _gradient_fit(row[1:1+train, 2:], row[1:1+train, :2], scale)
        fits.append(coefficients)
        fit_rms.append(float(np.sqrt(np.mean((prediction-row[1:1+train, :2])**2))))
        conditions.append(condition)
        gradient = polynomial_basis(row[1+train:, 2:]/scale, gradient=True)
        gradient = np.concatenate((np.broadcast_to(np.eye(2), (len(gradient), 2, 2)), gradient), axis=2)
        predicted = -np.einsum("nij,j->ni", gradient, coefficients)/scale
        holdout_errors.append(float(np.sqrt(np.mean((predicted-row[1+train:, :2])**2))))
    fits = np.asarray(fits)
    coefficients_mm = fits[:, 2:]/scale**POWERS*1000
    cc = [float(-(coefficients_mm[2*i+2, 0]-coefficients_mm[2*i+1, 0])/(2*step))
          for i, step in enumerate(steps)]
    checks = {"canonical_input_family": input_convention == "fixed_canonical",
        "symplectic_reference": reference["scaled_symplectic_defect"] <= symplectic_defect_budget,
        "lagrangian_input_plane": reference["input_lagrangian_relative_defect"] <= symplectic_defect_budget,
        "transfer_perturbation_converged": reference["relative_transfer_change"] <= transfer_change_budget,
        "gradient_training_residual": max(fit_rms) <= gradient_error_budget_m,
        "independent_gradient_holdout": max(holdout_errors) <= gradient_error_budget_m}
    cc_changes = np.abs(np.diff(cc))
    cc_bound = 1e-6+.01*abs(cc[-1]) if cc else None
    # Convert successive Cc derivative estimates to the observable they
    # predict, at the declared pupil edge and largest scanned energy offset.
    # A coefficient percentage alone cannot guarantee an image-space budget.
    energy_extent = float(np.max(steps)) if len(steps) else 0.
    cc_image_changes = cc_changes*1e-3*scale*energy_extent
    observable_cc_pass = bool(len(steps) >= 3 and np.all(cc_image_changes[-2:] <= gradient_error_budget_m))
    chromatic_status = ("NOT_RUN" if not len(steps) else "INSUFFICIENT_LEVELS" if len(steps) < 3
        else "PASS" if all(checks.values()) and observable_cc_pass
            and np.all(cc_changes[-2:] <= cc_bound) else "REJECTED")
    report = {"schema": "canonical-potential-aberration-diagnostic-v1",
        "snapshot_digest": snapshot.numerical_digest, "reference_digest": reference["digest"],
        "same_field_as_reference": snapshot.numerical_digest == reference["snapshot_digest"],
        "reference_reuse_scope": "fixed canonical input/reference plane; current field explicitly captured; no per-control refocus",
        "actual_sampling_backend": result.metadata["potential_sampling_backend"],
        "reference": thaw_json(reference), "input_convention": input_convention,
        "scalar_family_eligible": input_convention == "fixed_canonical",
        "coefficient_names": list(CARTESIAN_NAMES), "coefficients_mm": coefficients_mm.tolist(),
        "constant_gradient_intercept_m": (-fits[:, :2]/scale).tolist(),
        "cs_response_mm": float(coefficients_mm[0, CARTESIAN_NAMES.index("C3")]),
        "cc_response_mm": cc, "energy_relative_steps": steps.tolist(),
        "chromatic_qualification": {"status": chromatic_status,
            "relative_energy_step_count": len(steps), "required_step_count": 3,
            "successive_cc_changes_mm": cc_changes.tolist(), "absolute_change_budget_mm": cc_bound,
            "successive_cc_image_error_m": cc_image_changes.tolist(),
            "image_error_budget_m": float(gradient_error_budget_m),
            "declared_pupil_radius": float(scale), "maximum_relative_energy_offset": energy_extent,
            "observable_budget_pass": observable_cc_pass,
            "observable_convention": "abs(delta Cc)*1e-3*pupil_radius*maximum_relative_energy_offset; no phase qualification",
            "same_current_field_for_all_energies": True,
            "scope": "fixed-canonical-source response at the captured current field; not unqualified standard Cc"},
        "relative_energy_deltas": deltas.tolist(), "gradient_training_rms_m": fit_rms,
        "gradient_holdout_rms_m": holdout_errors, "gradient_condition": conditions,
        "finite_basis_gradient_qualification": {"status": "PASS" if all(checks.values()) else "REJECTED",
            "checks": checks, "gradient_error_budget_m": gradient_error_budget_m,
            "symplectic_defect_budget": symplectic_defect_budget,
            "transfer_change_budget": transfer_change_budget,
            "scope": "finite-pupil scalar-gradient representation only; not field/ray-step or coherent-wave qualification"},
        "raw_input_mechanical": inputs.reshape(len(deltas), count, 4),
        "raw_output_mechanical": raw, "raw_output_canonical": canon,
        "raw_chief_mechanical": raw[:, 0], "raw_chief_canonical": canon[:, 0],
        "raw_reduced_ray_action_m": result.reduced_action_m[-1].reshape(len(deltas), count),
        "empirical_aberrations_applied": False, "chief_tilt_defocus_removed": False,
        "production_admission": False, "coherent_wave_admission": False,
        "scope": "canonical scalar-gradient diagnostic; eligibility is not a convergence/gradient qualification",
        "cc_convention": "fixed physical canonical input source and fixed p_ref; not unqualified field-free fixed-angle Cc"
            if input_convention == "fixed_canonical" else "fixed mechanical input directions; immersed-source scalar-wavefront eligibility not asserted"}
    if retain_nominal_trajectory:
        def detached(values):
            data = np.asarray(values)
            return np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(data.shape)
        report["nominal_trajectory_z_m"] = detached(result.z_m)
        report["nominal_trajectory_xy_slopes"] = detached(result.xy_slopes[:, :count])
        report["nominal_trajectory_ray_action_m"] = detached(result.reduced_action_m[:, :count])
    return report


__all__ = ("canonical_coordinates", "mechanical_coordinates", "capture_canonical_pupil",
           "fit_canonical_intercepts", "measure_canonical_aberrations")
