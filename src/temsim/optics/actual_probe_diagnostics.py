"""Actual captured particles through the probe corrector; no reference rays.

The observer resumes a retained float64 checkpoint.  Source weights, identities,
energies, and clocks survive unchanged.  The affine residual below is a property
of this finite beam and this optical path, not an independent aberration fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from time import perf_counter

import numpy as np

from temsim import input_io
from temsim.cpu_resources import NumericalJobCancelled, numerical_job
from temsim.physics.beam_statistics import transverse_beam_statistics


SCOPE = (
    "Captured geometric particles, continued from an executed upstream checkpoint; "
    "CL3/BTlt/BSh history is inherited, not replaced by a reference ring. "
    "Position residuals remove the best current-weighted affine map of entrance "
    "(x,y,tx,ty), using the same surviving particles at both planes. They include "
    "all retained nonlinear/chromatic effects and finite-source sampling; they "
    "are not independent C3, A1, A2 or B2 coefficients, nor wave intensities. "
    "Dynamic drives retain the particle solver's captured-epoch approximation."
)


@dataclass(frozen=True)
class ActualProbeDiagnostic:
    planes: tuple[dict, ...]
    checkpoints: object
    alive: np.ndarray
    weights: np.ndarray
    source_ray_id: np.ndarray
    affine_residuals_m: tuple[np.ndarray, ...]
    provenance: dict
    scope: str = SCOPE


def affine_position_residual(entrance, output_xy, weights, alive):
    """Weighted, centred least squares on actual 4D entrance coordinates.

    Scaling each input dimension avoids conflating SI position/angle units with
    numerical rank.  Rank-deficient illumination is reported, not filled with
    synthetic rays.  Zero-current support probes never influence this fit.
    """
    source = np.asarray(entrance, float)
    target = np.asarray(output_xy, float)
    weights = np.asarray(weights, float)
    mask = np.asarray(alive, bool).copy()
    if (source.ndim != 2 or source.shape[1] != 4 or target.shape != (len(source), 2)
            or weights.shape != (len(source),) or mask.shape != weights.shape):
        raise ValueError("Actual-beam affine fit requires aligned rays and four entrance coordinates")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Actual-beam weights must be finite and nonnegative")
    mask &= weights > 0
    if not np.isfinite(source[mask]).all() or not np.isfinite(target[mask]).all():
        raise ValueError("Nonfinite surviving coordinates cannot enter an actual-beam fit")
    residual = np.full(target.shape, np.nan)
    if np.count_nonzero(mask) < 6:
        return dict(rms_m=None, rank=0, rays=int(mask.sum()),
                    status="At least six positive-current surviving rays are required"), residual
    w = weights[mask] / weights[mask].sum()
    x, y = source[mask], target[mask]
    x = x - w @ x
    y = y - w @ y
    scale = np.sqrt(w @ (x*x))
    nonzero = scale > 0
    design = x[:, nonzero] / scale[nonzero]
    root = np.sqrt(w)
    matrix, _, rank, singular = np.linalg.lstsq(design*root[:, None], y*root[:, None], rcond=1e-11)
    residual[mask] = y-design@matrix
    rms = float(np.sqrt(w @ np.sum(residual[mask]**2, axis=1)))
    condition = float(singular[0]/singular[rank-1]) if rank else None
    return dict(rms_m=rms, rank=int(rank), rays=int(mask.sum()),
                condition_number=condition, status="Weighted in-sample affine residual; not an aberration coefficient"), residual


def _check(cancelled):
    if cancelled is not None and cancelled():
        raise NumericalJobCancelled("Actual incident-beam diagnostic cancelled")


def _points(checkpoints, index):
    return np.column_stack([getattr(checkpoints, name)[index]
                            for name in ("x_m", "y_m", "tx_rad", "ty_rad")])


def _hp_planes(state):
    from temsim.lens_pose import lens_pose_registration
    from temsim.physics.core import FIELD_SIGMA_CUTOFF
    from temsim.physics.lens_field_provider import _posed_axial_support
    by_key = {item.key: item for item in state.corrector_elements}
    rows = []
    for key in ("probe_hp2_hexapole", "probe_hp1_hexapole"):
        hp = by_key.get(key)
        if hp is None or not hp.enabled:
            raise ValueError("Actual probe-corrector diagnostics need installed, enabled HP2 and HP1")
        width = float(hp.effective_length_mm)/2.355
        registration = lens_pose_registration(state, key)
        native = (float(hp.z_mm)-FIELD_SIGMA_CUTOFF*width,
                  float(hp.z_mm)+FIELD_SIGMA_CUTOFF*width)
        low, high = _posed_axial_support(native, registration,
            float(hp.mechanical_clear_bore_diameter_mm)*.0005)
        centre = float(registration.rotation_array[2, 2]*hp.z_mm
                       + registration.origin_array_m[2]*1e3)
        for label, z in (("field entrance", low), ("centre", centre), ("field exit", high)):
            rows.append(dict(label=f"{hp.name} {label}", z_mm=float(z), component_key=key,
                             registration=registration))
    return rows


@input_io.using_state_inputs
def _execute(state, checkpoints, index, plane_z, weights, initial_alive, initial_stops,
             initial_keys, cancelled):
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.instrument_magnetic import active_column_events, events_overlapping_interval
    _check(cancelled)
    start, stop = float(checkpoints.z_mm[index]), float(plane_z[-1])
    events = events_overlapping_interval(state, active_column_events(state), start, stop)
    state._tuning_cancelled = lambda: (_check(cancelled) or False)
    clip_planes = tuple(float(item.z_mm) for item in (*state.apertures, *state.recording_planes)
                        if start <= float(item.z_mm) <= stop)
    plan = build_propagation_plan(state, start, stop, events,
        checkpoint_z_mm=plane_z, save_z_mm=(*plane_z, *clip_planes))
    # Retain only the normal history / physical interception nodes in float64,
    # rather than allocating a full (every integration node x ray) checkpoint.
    plan = replace(plan, checkpoint_index=plan.save_index)
    # Stopped particles may legitimately have unknown phase space, energy and
    # clocks. Execute only the positive-current entrance population, retaining
    # its original order; no synthetic coordinates are assigned to dead rays.
    selected = np.flatnonzero(initial_alive & (weights > 0))
    output = execute_propagation_plan(state, plan, checkpoints.x_m[index, selected], checkpoints.tx_rad[index, selected],
        checkpoints.y_m[index, selected], checkpoints.ty_rad[index, selected],
        initial_kinetic_energy_ev=checkpoints.kinetic_energy_ev[index, selected],
        initial_time_s=checkpoints.flight_time_s[index, selected], return_flight_times=True,
        include_initial_plane_kicks=False, defer_nonfinite_until_clipping=True)
    _check(cancelled)
    saved = output[-1]
    z, x, y = saved.z_mm, saved.x_m, saved.y_m
    alive, stops, keys = clip_segment(state, z, x, y, np.ones(len(selected), bool),
                                     initial_stops[selected].copy(), [initial_keys[i] for i in selected])
    alive, stops, keys = clip_recording_planes(state, z, x, y, alive, stops, keys)
    alive, stops, keys = clip_column_wall(state, z, x, y, alive, stops, keys)
    masks = ~np.isfinite(stops)[None, :] | (stops[None, :] > saved.z_mm[:, None])
    for row in range(len(saved.z_mm)):
        if not np.isfinite(_points(saved, row)[masks[row]]).all():
            raise ValueError("Nonfinite trajectory before a physical stop in actual-beam continuation")
    # Return the original ID axis for plotting and paired statistics. NaN is
    # deliberate for every unexecuted column, not a zero-weight replacement ray.
    expanded = {}
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        values = np.full((len(saved.z_mm), len(weights)), np.nan)
        values[:, selected] = getattr(saved, name)
        expanded[name] = values
    full_masks = np.zeros((len(saved.z_mm), len(weights)), bool)
    full_masks[:, selected] = masks
    return replace(saved, **expanded), full_masks


def calculate_actual_probe_diagnostic(result, *, cancelled=None):
    """Continue the captured result only; never sample or regenerate its source.

    Residual-medium replay requires its stochastic transport checkpoint and is
    intentionally refused here.  A material specimen is permitted: the observed
    terminal plane is the existing incident reference, before its interaction.
    """
    from temsim.calculation_cache import calculation_signatures
    from temsim.calculation_manifest import assert_external_input_inventory_unchanged, solver_source_identity
    from temsim.physics.backend_execution import backend_receipts
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    _check(cancelled)
    started = perf_counter()
    original = getattr(result, "state_snapshot", None)
    if original is None:
        raise ValueError("Run a particle calculation first; its captured state is required")
    expected = getattr(result, "signatures", {}).get("incident")
    if not expected or calculation_signatures(original)["incident"] != expected:
        raise ValueError("The retained incident result is not bound to this captured source/optics state")
    if bool(getattr(getattr(original, "vacuum_map", None), "enabled", False)):
        raise ValueError("Actual-beam continuation cannot replay residual-medium scattering from phase space alone")
    if not bool(getattr(original, "probe_corrector_installed", False)):
        raise ValueError("The captured instrument has no installed probe corrector")
    state = decode_instrument(encode_instrument(original))
    external = getattr(result, "external_inputs", None)
    if external is not None:
        assert_external_input_inventory_unchanged(state, external)
    simulation = result.simulation
    checkpoints = getattr(simulation, "incident_checkpoints", None)
    if checkpoints is None or checkpoints.kinetic_energy_ev is None or checkpoints.flight_time_s is None:
        raise ValueError("Retained float64 incident checkpoints with energies and clocks are required")
    rows = _hp_planes(state)
    sample_z = float(state.sample.z_mm)
    if max(row["z_mm"] for row in rows) >= sample_z:
        raise ValueError("Both HP fields must precede the incident specimen reference")
    candidates = np.flatnonzero(checkpoints.z_mm < min(row["z_mm"] for row in rows))
    if not len(candidates) or checkpoints.z_mm[-1] < sample_z-1e-9:
        raise ValueError("The retained calculation does not cover the corrector entrance through the specimen")
    index = int(candidates[-1])
    start = float(checkpoints.z_mm[index])
    rows = [dict(label="Retained actual entrance", z_mm=start, component_key="", registration=None),
            *rows, dict(label="Incident specimen reference", z_mm=sample_z, component_key="", registration=None)]
    branch = simulation.incident
    count = checkpoints.x_m.shape[1]
    weights = np.asarray(branch.ray_weight, dtype=float).copy()
    ids = np.asarray(branch.source_ray_id).copy()
    stops = np.asarray(branch.blocked_z, dtype=float).copy()
    final_alive = np.asarray(branch.alive, bool)
    keys = list(branch.blocked_key)
    if (weights.shape != (count,) or ids.shape != (count,) or stops.shape != (count,)
            or final_alive.shape != (count,) or len(keys) != count or len(np.unique(ids)) != count):
        raise ValueError("Retained ray identities, weights and stop ancestry must match the entire checkpoint population")
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Retained source weights must be finite, nonnegative and nonzero")
    initial_alive = (final_alive & ~np.isfinite(stops)) | (stops > start)
    if not np.any(initial_alive & (weights > 0)):
        raise ValueError("No positive-current particles reach the retained corrector entrance")
    executed = initial_alive & (weights > 0)
    for values in (_points(checkpoints, index), checkpoints.kinetic_energy_ev[index, :, None],
                   checkpoints.flight_time_s[index, :, None]):
        if not np.isfinite(values[executed]).all():
            raise ValueError("Actual continuation requires finite retained phase space, energies and clocks")
    if np.any(checkpoints.kinetic_energy_ev[index, executed] <= 0) or np.any(checkpoints.flight_time_s[index, executed] < 0):
        raise ValueError("Retained entrance energy/time is invalid")
    # Later baseline losses must be recomputed; earlier stops keep their identity.
    stops[initial_alive] = np.nan
    for ray in np.flatnonzero(initial_alive):
        keys[int(ray)] = ""
    boundary_digest = sha256()
    for array in (_points(checkpoints, index), checkpoints.kinetic_energy_ev[index],
                  checkpoints.flight_time_s[index], weights, ids, initial_alive):
        value = np.ascontiguousarray(array)
        boundary_digest.update(str((value.dtype.str, value.shape)).encode())
        boundary_digest.update(value.tobytes())
    receipts = []
    with numerical_job(1, cancelled=cancelled), backend_receipts(receipts):
        saved, masks = _execute(state, checkpoints, index,
            tuple(sorted({row["z_mm"] for row in rows})), weights, initial_alive, stops, keys, cancelled)
    _check(cancelled)
    entrance = _points(checkpoints, index)
    report, residuals = [], []
    for row in rows:
        point_index = int(np.argmin(abs(saved.z_mm-row["z_mm"])))
        if abs(saved.z_mm[point_index]-row["z_mm"]) > 1e-9:
            raise ValueError("Actual-beam diagnostic is missing an exact requested plane")
        values, mask = _points(saved, point_index), masks[point_index]
        positive = mask & (weights > 0)
        metrics = None if not np.any(positive) else asdict(transverse_beam_statistics(
            *values.T, alive=mask, weights=weights))
        affine, residual = affine_position_residual(entrance, values[:, :2], weights, mask)
        local = None
        if metrics is not None and row["registration"] is not None:
            lab = np.array([metrics["mean_x_m"], metrics["mean_y_m"], row["z_mm"]*.001])
            local = row["registration"].positions_global_to_local_m(lab)[:2].tolist()
        report.append(dict(label=row["label"], z_mm=row["z_mm"], component_key=row["component_key"],
            checkpoint_index=point_index, statistics=metrics, affine=affine, local_centroid_xy_m=local))
        residuals.append(residual)
    if external is not None:
        assert_external_input_inventory_unchanged(state, external)
    _check(cancelled)
    if calculation_signatures(original)["incident"] != expected:
        raise ValueError("The captured source/optics changed during this diagnostic")
    manifest = getattr(result, "calculation_manifest", None)
    old_sample_rows = np.flatnonzero(abs(checkpoints.z_mm-sample_z) <= 1e-9)
    sample_difference = None
    sample_comparison_rays = 0
    if len(old_sample_rows) == 1:
        common = final_alive & masks[-1] & (weights > 0)
        if np.any(common):
            delta = _points(saved, -1)[common, :2]-_points(checkpoints, int(old_sample_rows[0]))[common, :2]
            sample_difference = float(np.sqrt(weights[common] @ np.sum(delta*delta, axis=1)/weights[common].sum()))
            sample_comparison_rays = int(common.sum())
    provenance = dict(schema="actual-incident-probe-v1", incident_signature=expected,
        producer_manifest=getattr(manifest, "digest", None),
        producer_solver=getattr(getattr(manifest, "solver", None), "source_digest", None),
        consumer_solver=solver_source_identity(), boundary_z_mm=start,
        boundary_sha256=boundary_digest.hexdigest(), source_ray_count=count,
        entrance_surviving_rays=int(initial_alive.sum()), source_weight_sum=float(weights.sum()),
        executed_positive_current_rays=int(executed.sum()),
        unexecuted_columns="Original IDs and weights retained; upstream-stopped and zero-current columns have NaN output and false observation masks",
        step_mm=float(state.step_mm), history_step_mm=float(state.history_step_mm),
        captured_epoch_s=float(getattr(state, "simulation_time_s", 0)), backend_receipts=receipts,
        elapsed_s=perf_counter()-started, external_inputs_verified=external is not None,
        retained_sample_difference_rms_m=sample_difference, retained_sample_comparison_rays=sample_comparison_rays,
        fit_population="Each plane's positive-current survivors, paired with the identical entrance ray IDs",
        precision="Float64 executed checkpoints; no interpolation or source resampling",
        clipping_sampling="Float64 saved history and explicit aperture/recording planes; tilted plates and walls use the existing piecewise-linear segment interception",
        numerical_scope="New exact observation nodes can change finite-step quadrature; retained-sample difference is reported, not assumed zero or an independent convergence test")
    return ActualProbeDiagnostic(tuple(report), saved, masks, weights, ids, tuple(residuals), provenance)
