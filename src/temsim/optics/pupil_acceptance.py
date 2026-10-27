"""Zero-current, source-to-specimen mechanical pupil observations.

The reference pupil is a two-dimensional direction slice at one declared
source position and energy. It is not the four-dimensional finite-source
acceptance, a current quantile, or a coherent diffraction calculation.
"""
from __future__ import annotations

import math
import numpy as np


def direction_angles(slopes, reference_slopes):
    """Exact spherical separation from a retained chief direction, in radians."""
    slopes = np.asarray(slopes, float)
    reference = np.asarray(reference_slopes, float)
    if slopes.shape[-1:] != (2,) or reference.shape != (2,) or not (
            np.isfinite(slopes).all() and np.isfinite(reference).all()):
        raise ValueError("Angles require finite two-component slopes")
    vectors = np.concatenate((slopes, np.ones((*slopes.shape[:-1], 1))), axis=-1)
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    axis = np.r_[reference, 1.]
    axis /= np.linalg.norm(axis)
    return np.arctan2(np.linalg.norm(np.cross(vectors, axis), axis=-1), vectors @ axis)


def trace_tip_probes(state, launch_xy_slopes, planes_mm, *, step_mm=.05,
                     emission_energy_ev=None):
    """Trace declared auxiliary launches through the unchanged gun and column.

    Launch columns are x, y in metres and local dx/dz, dy/dz. Directions beyond
    the configured source cutoff can measure geometric acceptance but cannot
    create emitted intensity. The current implementation deliberately rejects
    surface/coherent tip models rather than invent their launch conditions.
    """
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.simulation_modes import is_unified
    from temsim.optics.electron_gun.base import EmissionBundle
    from temsim.optics.electron_gun.tip_curvature import curve_bundle, support_radius_nm
    from temsim.optics.electron_gun.tracing import trace_feg_diagnostic_to_exit
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.physics.unified_field_particle import trace_unified_response
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.vacuum import bind_gun_environment
    launch = np.asarray(launch_xy_slopes, float)
    planes = np.asarray(planes_mm, float)
    if (launch.ndim != 2 or launch.shape[1] != 4 or not len(launch)
            or not np.isfinite(launch).all() or planes.ndim != 1 or not len(planes)
            or not np.isfinite(planes).all() or np.any(np.diff(planes) <= 0)
            or isinstance(step_mm, bool) or not math.isfinite(step_mm) or step_mm <= 0):
        raise ValueError("Pupil probes require finite launch states and ordered observation planes")
    before = capture_instrument_snapshot(state)
    candidate = before.restore()
    if callable(getattr(state, "_tuning_cancelled", None)):
        candidate._tuning_cancelled = state._tuning_cancelled
    gun = candidate.electron_gun
    emitter = gun.emitter
    if (not is_unified(candidate) or gun.type_key != "cold_feg"
            or emitter.surface_model is not None or emitter.coherence is not None
            or candidate.vacuum_map.enabled or any(part.dynamic for part in column_dipole_fields(candidate))):
        raise ValueError("Pupil observation currently requires static unified cold-FEG classical vacuum transport")
    if planes[0] <= gun.exit_plane_z_mm or planes[-1] > candidate.sample.upper_surface_z_mm:
        raise ValueError("Pupil column planes must follow gun exit and precede the specimen")
    energy = float(emitter.emission_energy_ev if emission_energy_ev is None else emission_energy_ev)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("Reference emission energy must be positive, separately from its FWHM")
    if np.any(np.hypot(launch[:, 0], launch[:, 1]) > support_radius_nm(emitter)*1e-9*(1+1e-12)):
        raise ValueError("Diagnostic source positions must remain on the configured emitting support")
    n = len(launch)
    bundle = EmissionBundle(launch[:, 0].copy(), launch[:, 1].copy(), launch[:, 2].copy(),
        launch[:, 3].copy(), np.full(n, energy-emitter.emission_energy_ev),
        np.zeros(n), np.arange(n, dtype=np.int64))
    bundle = curve_bundle(bundle, emitter)
    bind_gun_environment(candidate)
    with instrument_gun_field_context(candidate):
        traced = trace_feg_diagnostic_to_exit(gun, bundle,
            cancelled=getattr(candidate, "_tuning_cancelled", None))
    emitted = traced.exit_bundle
    selected = np.flatnonzero(emitted.alive)
    phase = np.full((len(planes), n, 4), np.nan)
    kinetic = np.full((len(planes), n), np.nan)
    times = np.full_like(kinetic, np.nan)
    blocked = np.asarray(traced.blocked_z_mm).copy()
    labels = list(traced.blocked_key)
    receipt = None
    if selected.size:
        initial = np.column_stack((emitted.x_m, emitted.y_m, emitted.tx_rad, emitted.ty_rad))[selected]
        response = trace_unified_response(candidate, initial, gun.exit_plane_z_mm*.001, planes*.001,
            energy_ev=gun.nominal_exit_energy_ev+emitted.energy_offset_ev[selected],
            flight_time_s=emitted.flight_time_s[selected], maximum_step_m=step_mm*.001)
        phase[:, selected] = response["xy_slopes"]
        kinetic[:, selected] = response["kinetic_energy_ev"]
        times[:, selected] = response["flight_time_s"]
        blocked[selected] = response["blocked_z_m"]*1000
        for local, original in enumerate(selected):
            labels[original] = response["blocked_key"][local]
        receipt = dict(response["receipt"])
    alive = np.asarray(emitted.alive)[None, :] & (
        ~np.isfinite(blocked)[None, :] | (blocked[None, :] > planes[:, None]))
    if capture_instrument_snapshot(state).digest != before.digest:
        raise ValueError("Physical inputs changed during pupil observation")
    return dict(z_mm=planes.copy(), xy_slopes=phase, kinetic_energy_ev=kinetic,
        flight_time_s=times, alive=alive, blocked_z_mm=blocked, blocked_key=labels,
        zero_current=True, launch_xy_slopes=launch.copy(), reference_emission_energy_ev=energy,
        source_direction_supported=np.hypot(launch[:, 2], launch[:, 3]) <= emitter.angular_cutoff_mrad*.001,
        snapshot_digest=before.digest, receipt=receipt,
        gun_arrivals=[dict(key=row.key, z_mm=row.z_mm, x_m=row.x_m, y_m=row.y_m,
            reached=row.reached, transmitted=row.transmitted) for row in traced.plane_arrivals])


def measure_radial_boundary(trace, *, azimuth_count=16, radial_slopes=None,
                            subdivisions=8, refinements=4):
    """Bracket the first transmitted-to-stopped transition along each spoke.

    ``trace(radii, azimuths)`` returns actual terminal survival and stop keys.
    A radial scan tests the connected chief-ray pupil; observed re-entry or an
    unbracketed sector stays unresolved. Subdivision is deterministic and
    batched. No alpha95 or finite-population maximum is used as fallback.
    """
    if (type(azimuth_count) is not int or not 8 <= azimuth_count <= 256
            or type(subdivisions) is not int or not 2 <= subdivisions <= 32
            or type(refinements) is not int or not 0 <= refinements <= 8):
        raise ValueError("Invalid bounded pupil sampling budget")
    radii = np.asarray([0., .003, .01, .03, .1, .3, 1., 3., 10., 30.]
                       if radial_slopes is None else radial_slopes, float)
    if (radii.ndim != 1 or len(radii) < 3 or radii[0] != 0
            or not np.isfinite(radii).all() or np.any(np.diff(radii) <= 0)):
        raise ValueError("Pupil radial search must increase from zero")
    phi = np.arange(azimuth_count)*2*np.pi/azimuth_count
    evaluations = 0

    def execute(grid):
        nonlocal evaluations
        result = trace(grid.ravel(), np.repeat(phi, grid.shape[1]))
        survival = np.asarray(result["transmitted"])
        keys = np.asarray(result["blocked_key"], dtype=str)
        if (survival.dtype.kind != "b" or survival.shape != (grid.size,)
                or keys.shape != survival.shape or np.any(~survival & (keys == ""))):
            raise ValueError("Pupil transport must return physical stops for every rejected probe")
        evaluations += grid.size
        return survival.reshape(grid.shape), keys.reshape(grid.shape)

    grid = np.tile(radii, (azimuth_count, 1))
    live, keys = execute(grid)
    lower, upper = np.full(azimuth_count, np.nan), np.full(azimuth_count, np.nan)
    stop_keys = [""]*azimuth_count
    reasons = [""]*azimuth_count
    for i in range(azimuth_count):
        if not live[i, 0]:
            reasons[i] = "chief_ray_blocked"
        elif np.all(live[i]):
            reasons[i] = "no_mechanical_boundary_in_searched_launch_directions"
        else:
            first = np.flatnonzero(~live[i])[0]
            if np.any(live[i, first:]):
                reasons[i] = "disconnected_acceptance_detected"
            else:
                lower[i], upper[i] = grid[i, first-1:first+1]
                stop_keys[i] = str(keys[i, first])
    # Retain a fixed number of azimuths for stable correspondence. Unresolved
    # sectors are sampled at zero and cannot be promoted by later iterations.
    for _ in range(refinements):
        valid = np.isfinite(lower)
        if not valid.any():
            break
        grid = np.zeros((azimuth_count, subdivisions+1))
        grid[valid] = lower[valid, None]+(upper-lower)[valid, None]*np.linspace(0., 1., subdivisions+1)
        live, keys = execute(grid)
        for i in np.flatnonzero(valid):
            if not live[i, 0] or live[i, -1] or np.any(np.diff(live[i].astype(int)) > 0):
                reasons[i] = "unstable_or_disconnected_boundary"
                lower[i] = upper[i] = np.nan
                continue
            first = np.flatnonzero(~live[i])[0]
            lower[i], upper[i] = grid[i, first-1:first+1]
            stop_keys[i] = str(keys[i, first])
    rows = [dict(azimuth_rad=float(phi[i]), accepted_launch_slope=float(lower[i]) if np.isfinite(lower[i]) else None,
        rejected_launch_slope=float(upper[i]) if np.isfinite(upper[i]) else None,
        limiting_component=stop_keys[i] or None, status="BRACKETED" if np.isfinite(lower[i]) else "UNRESOLVED",
        reason=reasons[i] or None) for i in range(azimuth_count)]
    return dict(schema="physical-pupil-radial-boundary-v1", sectors=rows,
        status="BRACKETED" if np.isfinite(lower).all() else "UNRESOLVED",
        evaluations=evaluations, azimuth_count=azimuth_count,
        radial_search_limit_slope=float(radii[-1]),
        scope="Connected chief-ray direction slice; finite radial and azimuth sampling, not full finite-source phase space")


def summarize_boundary(boundary, accepted_trace, chief_trace, *, target_mrad=30., tolerance_mrad=.3):
    """Map the same accepted-side rim rays at every executed reference plane."""
    if (not np.isfinite((target_mrad, tolerance_mrad)).all() or target_mrad <= 0
            or not 0 <= tolerance_mrad < target_mrad):
        raise ValueError("Pupil target and tolerance require finite positive physical units")
    rows = boundary["sectors"]
    if boundary["status"] != "BRACKETED":
        return dict(status="NOT_MEASURED", target_mrad=target_mrad, alpha_edge_mrad=None)
    phase = np.asarray(accepted_trace["xy_slopes"])
    chief = np.asarray(chief_trace["xy_slopes"])
    for key in ("snapshot_digest", "reference_emission_energy_ev"):
        if (key not in accepted_trace or key not in chief_trace
                or accepted_trace[key] != chief_trace[key]):
            raise ValueError("Boundary and chief must share the physical snapshot and reference emission energy")
    if (phase.shape != (len(accepted_trace["z_mm"]), len(rows), 4)
            or chief.shape != (len(accepted_trace["z_mm"]), 1, 4)
            or not np.array_equal(accepted_trace["z_mm"], chief_trace["z_mm"])
            or not np.asarray(accepted_trace["alive"]).all() or not np.asarray(chief_trace["alive"]).all()):
        raise ValueError("Boundary and chief observations require matching planes and surviving rays")
    planes = []
    for z, rim, axis in zip(accepted_trace["z_mm"], phase, chief[:, 0]):
        angles = direction_angles(rim[:, 2:], axis[2:])*1000
        planes.append(dict(z_mm=float(z), alpha_edge_by_source_azimuth_mrad=angles.tolist(),
            alpha_edge_min_mrad=float(angles.min()), alpha_edge_max_mrad=float(angles.max()),
            chief_x_m=float(axis[0]), chief_y_m=float(axis[1]),
            chief_tx=float(axis[2]), chief_ty=float(axis[3]),
            edge_mean_tx=float(rim[:, 2].mean()), edge_mean_ty=float(rim[:, 3].mean()),
            edge_xy_m=rim[:, :2].tolist(), edge_slopes=rim[:, 2:].tolist()))
    sample = planes[-1]
    passes = (sample["alpha_edge_min_mrad"] >= target_mrad-tolerance_mrad
              and sample["alpha_edge_max_mrad"] <= target_mrad+tolerance_mrad)
    return dict(status="WITHIN_TARGET_PENDING_CONVERGENCE" if passes else "OUTSIDE_TARGET",
        target_mrad=target_mrad, tolerance_mrad=tolerance_mrad, planes=planes,
        specimen=sample, nominal_reference_energy_only=True,
        physical_qualification=False, alpha95_used_for_target=False,
        limitation="Accepted-side sampled rim; azimuth/radial/step convergence and finite-source filling reported separately")
