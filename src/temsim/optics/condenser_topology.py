"""Executed, bounded-sampling evidence for nanoprobe condenser ray paths.

Both ranges require a two-dimensional waist between C2 and C3. Normal range
forbids an additional waist between C3 and ADL; large-angle range requires it.
This gate neither fits lens strengths nor qualifies the independent specimen
focus. These are declared simulator requirements, not OEM mode qualification.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np


REQUIREMENT = "c2_c3_waist_no_c3_adl_waist"
LARGE_ANGLE_REQUIREMENT = "c2_c3_waist_and_c3_adl_waist"
_KEYS = ("condenser_lens_2", "condenser_lens_3", "adapter_lens")
_COORDINATES = ("x_m", "y_m", "tx_rad", "ty_rad")
_BOUNDARY_TOLERANCE_MM = 1e-6
_MINIMUM_RAYS = 5


def requirement_for_state(state):
    """Select the branch, including legacy lightweight mode observers."""
    explicit_large = str(getattr(state, "probe_mode", "")).lower() == "nano_probe_large_angle"
    selected = getattr(state, "nanoprobe_range", "large_angle" if explicit_large else "normal")
    if explicit_large and selected != "large_angle":
        raise ValueError("Conflicting Nanoprobe range and probe mode")
    if selected == "normal":
        return REQUIREMENT
    if selected == "large_angle":
        return LARGE_ANGLE_REQUIREMENT
    raise ValueError(f"Unknown Nanoprobe range: {selected!r}")


def _boundaries(state):
    lenses = {}
    for lens in state.lenses:
        if lens.key in _KEYS:
            if lens.key in lenses:
                raise ValueError("Duplicate condenser topology component")
            lenses[lens.key] = lens
    if any(key not in lenses or not getattr(lenses[key], "enabled", True)
           or not getattr(lenses[key], "installed", True) for key in _KEYS):
        raise ValueError("Condenser topology requires installed, enabled C2, C3 and ADL")
    z = tuple(float(lenses[key].z_mm) for key in _KEYS)
    if not np.all(np.isfinite(z)) or not z[0] < z[1] < z[2]:
        raise ValueError("Condenser topology requires finite, ordered C2, C3 and ADL planes")
    return z


def is_applicable(state):
    """Whether all three devices are installed and enabled in this assembly.

    Invalid coordinates or ordering must still reach ``evaluate`` and fail;
    geometry mistakes must never silently disable an otherwise applicable gate.
    """
    try:
        return all(any(lens.key == key and getattr(lens, "enabled", True)
                       and getattr(lens, "installed", True) for lens in state.lenses)
                   for key in _KEYS)
    except (AttributeError, TypeError):
        return False


def _sampling_geometry(state, maximum_spacing_mm):
    spacing = float(maximum_spacing_mm)
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("Condenser topology checkpoint spacing must be finite and positive")
    c2, c3, adl = _boundaries(state)
    # Both sides of a boundary must be observed. Otherwise a waist on C3 could
    # disappear between the open C2-C3 and C3-ADL intervals.
    guard = min(spacing / 2, (c3-c2) / 8, (adl-c3) / 8)
    gun_exit = float(state.electron_gun.exit_plane_z_mm)
    surface = float(state.sample.upper_surface_z_mm)
    if (not np.isfinite([gun_exit, surface]).all()
            or c2-guard < gun_exit or adl+guard > surface):
        raise ValueError("Condenser topology boundary guards must fit the incident column")
    return (c2, c3, adl), guard, spacing


def requested_capture_planes(state, *, maximum_spacing_mm=1.0):
    """Exact checkpoint requests, including all three centres and their sides.

    The spacing bounds the evidence, not the integration step. The caller
    must request full-precision propagation checkpoints, never plot history.
    """
    (c2, c3, adl), guard, spacing = _sampling_geometry(state, maximum_spacing_mm)
    edges = (c2-guard, c2, c3, adl, adl+guard)
    counts = [max(1, math.ceil((right-left)/spacing))
              for left, right in zip(edges, edges[1:])]
    if sum(counts) > 20000:
        raise ValueError("Condenser topology requests too many checkpoints")
    pieces = [np.linspace(left, right, count+1)
              for left, right, count in zip(edges, edges[1:], counts)]
    # Explicit near-C3 planes make boundary waists visible on either side.
    return tuple(map(float, np.unique(np.r_[*pieces, c3-guard, c3+guard])))


def _report(state):
    try:
        requirement = requirement_for_state(state)
    except ValueError:
        requirement = None
    selected = ("large_angle" if requirement == LARGE_ANGLE_REQUIREMENT else "normal"
                if requirement == REQUIREMENT else getattr(state, "nanoprobe_range", None))
    return dict(schema="condenser-topology-v1", requirement=requirement,
                nanoprobe_range=selected,
                status="UNRESOLVED", passed=False, failures=[], message="",
                intervals={}, candidates=[], refinement_planes_mm=[],
                qualification="Executed checkpoints at bounded axial sampling; not OEM qualification")


def _reject(report, failure, message, *, unresolved=True):
    report.update(status="UNRESOLVED" if unresolved else "FAIL", passed=False,
                  message=message)
    report["failures"].append(failure)
    return report


def _strict_spatial_minima(checkpoints, weights, candidates):
    """Reject a flat direction hidden by eigenvalue ordering changes.

    Sorting the two principal variances independently at neighbouring planes
    can swap their physical directions. A fixed-width Y beam and an X line
    focus can then appear to shrink both sorted values. Positive-definite
    covariance growth on each side requires shrinking in every transverse
    direction, and is invariant under a common rotation of the coordinates.
    """
    w = weights / weights.sum()
    accepted = []
    for row in candidates:
        index = int(np.searchsorted(checkpoints.z_mm, row["z_mm"]))
        xy = np.stack((checkpoints.x_m[index-1:index+2],
                       checkpoints.y_m[index-1:index+2]), axis=1)
        xy -= (xy @ w)[:, :, None]
        covariance = np.einsum("zir,zjr,r->zij", xy, xy, w)
        noise = 64*np.finfo(float).eps * np.max(np.abs(covariance))
        growth = np.array([np.linalg.eigvalsh(covariance[side]-covariance[1])
                           for side in (0, 2)])
        if np.all(growth > noise):
            accepted.append(dict(row, bracket_covariance_growth_m2=growth.tolist()))
    return accepted


def evaluate(state, checkpoints, masks, weights, *, maximum_spacing_mm=1.0):
    """Evaluate actual checkpoints using one positive-current survivor cohort.

    Missing coverage, sparse samples, nonfinite data or fewer than five rays
    never count as evidence that a forbidden waist is absent. Interpolated
    roots only propose further planes: each required waist additionally needs
    an executed two-dimensional local minimum. A forbidden root or one on a
    component boundary fails closed before any refinement is attempted.
    """
    report = _report(state)
    try:
        large_angle = requirement_for_state(state) == LARGE_ANGLE_REQUIREMENT
        (c2, c3, adl), guard, spacing = _sampling_geometry(state, maximum_spacing_mm)
        z = np.asarray(checkpoints.z_mm, dtype=float)
        arrays = [np.asarray(getattr(checkpoints, key), dtype=float) for key in _COORDINATES]
        live_masks = np.asarray(masks)
        weight = np.asarray(weights, dtype=float)
        if (z.ndim != 1 or z.size < 3 or np.any(~np.isfinite(z))
                or np.any(np.diff(z) <= 0) or weight.ndim != 1
                or live_masks.dtype.kind != "b"
                or live_masks.shape != (z.size, weight.size)
                or any(a.shape != live_masks.shape for a in arrays)
                or np.any(~np.isfinite(weight)) or np.any(weight < 0)
                or not math.isfinite(float(weight.sum())) or weight.sum() <= 0):
            return _reject(report, "invalid_checkpoint_arrays", "Invalid condenser checkpoint arrays or weights")
    except (AttributeError, TypeError, ValueError) as error:
        return _reject(report, "invalid_geometry_or_checkpoints", str(error))

    report["intervals"] = {
        "c2_c3": dict(bounds_mm=[c2, c3], requirement="at_least_one_executed_2d_waist", waists=[]),
        "c3_adl": dict(bounds_mm=[c3, adl], requirement=(
            "at_least_one_executed_2d_waist" if large_angle
            else "no_converging_to_diverging_waist"), waists=[]),
    }
    lower, upper = c2-guard, adl+guard
    tolerance = 1e-9
    if any(not np.any(np.isclose(z, plane, rtol=0, atol=tolerance))
           for plane in (lower, c2, c3, adl, upper)):
        return _reject(report, "missing_boundary_checkpoints", "Missing exact component or boundary-guard checkpoints")
    selected = np.flatnonzero((z >= lower-tolerance) & (z <= upper+tolerance))
    z = z[selected]
    if z.size < 5 or np.max(np.diff(z)) > spacing + tolerance:
        return _reject(report, "insufficient_axial_sampling", "Condenser checkpoint spacing exceeds the declared evidence limit")
    masks = live_masks[selected]
    if np.any((~masks[:-1] & masks[1:]) & (weight > 0)[None, :]):
        return _reject(report, "inconsistent_survivor_masks", "A stopped ray reappears inside the condenser interval")
    cohort = np.all(masks, axis=0) & (weight > 0)
    if np.count_nonzero(cohort) < _MINIMUM_RAYS:
        return _reject(report, "insufficient_surviving_rays", "Fewer than five positive-current rays survive the entire condenser interval")
    arrays = [a[np.ix_(selected, cohort)] for a in arrays]
    if any(np.any(~np.isfinite(a)) for a in arrays):
        return _reject(report, "nonfinite_surviving_coordinates", "Nonfinite coordinates in the common current-carrying cohort")
    weight = weight[cohort]
    report["cohort"] = dict(ray_count=int(cohort.sum()),
        source_current_fraction=float(weight.sum()/np.asarray(weights, float).sum()),
        definition="same positive-current rays surviving all C2-to-ADL boundary-guard checkpoints")
    report["coverage"] = dict(start_mm=float(z[0]), end_mm=float(z[-1]),
        checkpoint_count=int(z.size), maximum_spacing_mm=float(np.max(np.diff(z))),
        requested_maximum_spacing_mm=spacing)
    cp = SimpleNamespace(z_mm=z, **dict(zip(_COORDINATES, arrays)))
    fixed_masks = np.ones(arrays[0].shape, bool)
    from temsim.optics.beam_path_audit import crossover_candidates
    from temsim.physics.corrector_crossovers import measured_corrector_crossovers
    try:
        candidates = crossover_candidates(cp, fixed_masks, weight)
        intervals = {"c2_c3": (c2, c3), "c3_adl": (c3, adl)}
        interval_candidates = {key: [row for row in candidates if left < row["z_mm"] < right]
                               for key, (left, right) in intervals.items()}
        required_intervals = ("c2_c3", "c3_adl") if large_angle else ("c2_c3",)
        boundaries = [row for row in candidates
                      if any(abs(row["z_mm"]-plane) <= _BOUNDARY_TOLERANCE_MM
                             for plane in (c2, c3, adl))]
        measured = {key: _strict_spatial_minima(cp, weight,
            measured_corrector_crossovers(cp, fixed_masks, weight, *intervals[key]))
                    for key in required_intervals}
    except (ValueError, FloatingPointError) as error:
        return _reject(report, "unresolved_waist_evidence", str(error))
    report["candidates"] = candidates
    for key, rows in interval_candidates.items():
        report["intervals"][key].update(candidate_count=len(rows), waists=measured.get(key, rows))
    if boundaries:
        report["boundary_candidates"] = boundaries
        return _reject(report, "crossover_on_component_boundary",
                       "A crossover touches C2, C3 or ADL and cannot be assigned to a permitted open interval")
    if not large_angle and interval_candidates["c3_adl"]:
        return _reject(report, "forbidden_c3_adl_waist",
                       "Executed ray covariances bracket an additional waist between C3 and ADL", unresolved=False)
    # Every required interval must have a bracket before expensive refinement.
    for key in required_intervals:
        if not interval_candidates[key]:
            label = key.replace("_", "-to-").upper()
            return _reject(report, f"missing_{key}_waist",
                           f"No {label} converging-to-diverging waist is bracketed at the declared sampling",
                           unresolved=False)
    unresolved = [key for key in required_intervals if not measured[key]]
    if unresolved:
        refinement, messages = [], []
        for key in unresolved:
            left, right = intervals[key]
            for row in interval_candidates[key]:
                root = float(row["z_mm"])
                delta = min(.01, spacing/20, (root-left)/4, (right-root)/4)
                if delta > 10*_BOUNDARY_TOLERANCE_MM:
                    refinement.extend((root-delta, root, root+delta))
            report["failures"].append(f"unexecuted_{key}_waist")
            messages.append(key.replace("_", "-to-").upper())
        report["refinement_planes_mm"] = sorted(set(refinement))
        report["message"] = (" and ".join(messages)
            + " waist brackets still need executed two-dimensional minimum checkpoints")
        return report
    report.update(status="PASS", passed=True,
                  message=("C2-to-C3 and C3-to-ADL physical waists confirmed at the declared axial sampling"
                           if large_angle else
                           "C2-to-C3 physical waist confirmed; no C3-to-ADL waist found at the declared axial sampling"))
    return report


def validate(state, *, step_mm=.05, maximum_spacing_mm=1.0):
    """Execute the bounded topology audit, refining only required waist planes.

    Reuses the production gun cache and obeys the state's compute backend.
    Existing raster/static drive settings are preserved; callers preparing a
    detached static alignment state retain responsibility for their settings.
    """
    from temsim.optics.beam_path_audit import incident_checkpoints
    try:
        planes = requested_capture_planes(state, maximum_spacing_mm=maximum_spacing_mm)
        if not math.isfinite(step_mm) or step_mm <= 0:
            raise ValueError("Condenser topology integration step must be finite and positive")
        for attempt in range(3):
            gun, cp, masks = incident_checkpoints(state, planes, step_mm=step_mm)
            report = evaluate(state, cp, masks, gun.exit_bundle.weight,
                              maximum_spacing_mm=maximum_spacing_mm)
            report.update(integration_step_mm=float(step_mm), executions=attempt+1)
            extra = report["refinement_planes_mm"]
            if not extra or report["passed"] or attempt == 2:
                return report
            updated = tuple(sorted(set(planes).union(extra)))
            if updated == planes:
                return report
            planes = updated
    except (AttributeError, TypeError, ValueError, FloatingPointError) as error:
        return _reject(_report(state), "execution_failed", str(error))
