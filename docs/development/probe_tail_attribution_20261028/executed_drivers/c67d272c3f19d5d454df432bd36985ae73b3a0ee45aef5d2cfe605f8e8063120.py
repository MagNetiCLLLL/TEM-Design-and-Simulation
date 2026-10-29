"""Observe the native current-D95 waist in actual classical column fields.

This is a vacuum-probe diagnostic, without specimen material interactions.
Every reported plane is an executed Lorentz checkpoint. No controls, source
weights, coordinates or focus are fitted, and no free-drift estimate is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

SCHEMA = "temsim-native-d95-waist-observation-v1"
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")
MAX_PLANES = 10001


def finite(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite and numeric")
    return float(value)


def capture_planes(state, *, window_nm=100., spacing_nm=1., extra_z_offset_nm=()):
    """Return exact requested coordinates; never round or merge nearby planes."""
    window, spacing = finite(window_nm, "Window"), finite(spacing_nm, "Spacing")
    if window <= 0 or spacing <= 0 or spacing > window:
        raise ValueError("Require 0 < spacing <= window")
    count = math.ceil(2 * window / spacing)
    if count + 4 > MAX_PLANES:
        raise ValueError("Through-focus checkpoint budget exceeded")
    try:
        extras = tuple(finite(value, "Extra offset") for value in extra_z_offset_nm)
    except TypeError as error:
        raise ValueError("Extra offsets must be a numeric sequence") from error
    if len(extras) + count + 4 > MAX_PLANES or any(abs(value) > window for value in extras):
        raise ValueError("Extra offsets exceed the declared window or checkpoint budget")
    surface = finite(state.sample.upper_surface_z_mm, "Specimen upper surface")
    center = finite(state.sample.z_mm, "Sample reference")
    lower = finite(state.sample.lower_surface_z_mm, "Specimen lower surface")
    left, right = surface-window*1e-6, surface+window*1e-6
    if not left <= center <= lower <= right:
        raise ValueError("The declared window must include the sample center and lower reference")
    from temsim.physics.particle_sections import section_limits
    start, stop = section_limits(state)
    if not start < left < surface < right <= stop:
        raise ValueError("Through-focus window exceeds the physical straight-column domain")
    return np.unique(np.r_[surface+np.linspace(-window, window, count+1)*1e-6,
        surface, center, lower, surface+np.asarray(extras)*1e-6])


def _json(path, value):
    from temsim.benchmark_suite import atomic_json
    atomic_json(path, value)


def _npz(path, arrays):
    handle, temporary = tempfile.mkstemp(prefix=".waist-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(value):
    a = np.ascontiguousarray(value)
    return sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def audit_extended_plan(state, plan, stop):
    """Audit the actual extended interval with the existing action/owner rules."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
    from audit_joint_probe_fields import _inputs, _plan_actions, _field_ownership, _c2_geometry
    before = capture_instrument_snapshot(state)
    aperture, start, surface = _inputs(state)
    runtime = plan.unified_runtime
    if type(runtime) is not UnifiedFieldRuntime or stop <= surface:
        raise ValueError("An extended production unified runtime is required")
    runtime.require_state(state)
    if runtime.source_identity != before.digest or not np.any(plan.z_mm[plan.checkpoint_index] == surface):
        raise ValueError("Extended plan must bind the original inputs and exact surface checkpoint")
    result = dict(schema="temsim-extended-probe-field-audit-v1", status="PASS", passed=True,
        plan_actions=_plan_actions(plan, runtime, start, stop),
        field_ownership=_field_ownership(state, runtime),
        c2_aperture=_c2_geometry(aperture, runtime, start, stop),
        identities=dict(input_digest=before.digest, runtime_digest=runtime.digest,
            potential_digest=runtime.snapshot.numerical_digest, geometry_digest=runtime.geometry.digest,
            plan_signature=plan.signature),
        scope="Executed extended column field/action ownership; specimen material interactions omitted")
    if capture_instrument_snapshot(state).digest != before.digest:
        raise ValueError("Field audit changed the original inputs")
    return result


def execute_scan(state, planes, cancelled, *, batch_size=128, _cache=None):
    """Execute complete original-source batches on one identical full-source mesh."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.simulation_modes import is_unified
    from temsim.specimen.source import specimen_is_vacuum
    from temsim.physics.unified_field_runtime import unified_runtime_preflight
    from temsim.physics.particle_sections import _trace_segment, _events, gun_dependency_signature
    from temsim.physics.ray_identity import emitted_source_identity
    from temsim.physics.simulation import _validate_gun_flight_times
    from temsim.physics.beam_current import effective_source_current_a
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    if not is_unified(state) or state.electron_gun.source_representation != "classical_particles":
        raise ValueError("Only original classical particles in unified fields are admitted")
    if any(getattr(state.sample, flag, False) for flag in ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled")):
        raise ValueError("Coherent calculations are not admitted by this classical diagnostic")
    if not specimen_is_vacuum(state.sample):
        raise ValueError("This vacuum-probe diagnostic requires an already-vacuum specimen state")
    require_physical_gun_source(state.electron_gun)
    issues = unified_runtime_preflight(state)["issues"]
    if issues:
        raise ValueError("Unsupported physical inputs: " + "; ".join(issues))
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("Batch size must be an integer in [1, 128]")
    original = capture_instrument_snapshot(state)
    state._tuning_cancelled = cancelled
    signature = gun_dependency_signature(state)
    if capture_instrument_snapshot(state).digest != original.digest:
        raise ValueError("Candidate is not canonical; source preparation changed its inputs")
    if cancelled():
        raise InterruptedError("Observation cancelled before source execution")
    gun_reused = bool(_cache is not None and signature and _cache.get("gun_signature") == signature
        and _cache.get("gun") is not None)
    gun = _cache["gun"] if gun_reused else trace_source_to_exit(state)
    if not _validate_gun_flight_times(gun):
        raise ValueError("The physical gun must retain actual finite flight times")
    e = gun.exit_bundle
    identity = emitted_source_identity(gun)
    weights, ids = np.asarray(e.weight), np.asarray(identity[0])
    if (weights.ndim != 1 or ids.shape != weights.shape or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
            or not np.isfinite(weights).all() or np.any(weights < 0) or not weights.sum() > 0):
        raise ValueError("Invalid original source weights or unique IDs")
    initial = tuple(np.asarray(getattr(e, key)) for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    offsets = np.asarray(e.energy_offset_ev)
    kinetic = float(state.electron_gun.nominal_exit_energy_ev)+offsets
    survival = (np.asarray(e.alive), np.asarray(gun.blocked_z_mm), np.asarray(gun.blocked_key, dtype=str))
    current = finite(effective_source_current_a(state), "Source current")
    if current <= 0:
        raise ValueError("Source current must be positive")
    pieces, ledgers, receipts, next_segments = [], [], [], []
    previous = _cache.get("segments", {}) if gun_reused else {}
    common_z = common_mesh = common_plan = None
    for lower in range(0, len(weights), batch_size):
        if cancelled():
            raise InterruptedError("Observation cancelled between source batches")
        selection = slice(lower, min(lower+batch_size, len(weights)))
        segment, resume, reused = _trace_segment(state, "incident", float(state.electron_gun.exit_plane_z_mm),
            float(planes[-1]), tuple(a[selection] for a in initial), offsets[selection], weights[selection],
            tuple(np.asarray(a)[selection] for a in identity),
            (survival[0][selection], survival[1][selection], survival[2][selection].tolist()),
            _events(state), previous.get((selection.start, selection.stop)), math.inf,
            initial_kicks=True, save_z_mm=tuple(planes),
            initial_kinetic_energy_ev=kinetic[selection], checkpoint_ray_count=len(weights))
        cp, branch = segment.checkpoints, segment.branch
        audit = audit_extended_plan(state, segment.plan, float(planes[-1]))
        if (not np.array_equal(branch.source_ray_id, ids[selection])
                or not np.array_equal(branch.ray_weight, weights[selection])):
            raise ValueError("Source identity or weights changed during actual transport")
        if common_z is None:
            common_z, common_mesh, common_plan = cp.z_mm.copy(), segment.plan.z_mm.copy(), segment.plan.signature
        elif (not np.array_equal(common_z, cp.z_mm) or not np.array_equal(common_mesh, segment.plan.z_mm)
              or common_plan != segment.plan.signature):
            raise ValueError("Source batches used different actual integration/checkpoint meshes")
        if any(np.count_nonzero(cp.z_mm == value) != 1 for value in planes):
            raise ValueError("An exact requested physical checkpoint is missing")
        pieces.append({**{key: np.asarray(getattr(cp, key)).copy() for key in PHASE},
            "alive": np.asarray(cp.unified_state.active).copy()})
        ledgers.append((np.asarray(branch.blocked_z).copy(), np.asarray(branch.blocked_key, dtype=str),
            np.asarray(branch.alive).copy()))
        receipts.append(dict(source_slice=[selection.start, selection.stop], resume_z_mm=resume,
            prefix_reused=bool(reused), field_audit=audit))
        if _cache is not None:
            next_segments.append(((selection.start, selection.stop), segment))
        del segment, branch, cp
    if capture_instrument_snapshot(state).digest != original.digest:
        raise ValueError("Actual propagation changed candidate inputs")
    arrays = {key: np.concatenate([part[key] for part in pieces], axis=1) for key in (*PHASE, "alive")}
    arrays.update(z_mm=common_z, integration_z_mm=common_mesh, source_id=ids.copy(), weights=weights.copy(),
        source_azimuth_rad=np.asarray(identity[1]).copy(),
        blocked_z_mm=np.concatenate([part[0] for part in ledgers]),
        blocked_key=np.concatenate([part[1] for part in ledgers]),
        terminal_alive=np.concatenate([part[2] for part in ledgers]),
        gun_exit_alive=survival[0].copy(), gun_blocked_z_mm=survival[1].copy(),
        gun_blocked_key=survival[2].copy(), source_energy_offset_ev=offsets.copy(),
        gun_exit_kinetic_energy_ev=kinetic.copy())
    arrays.update({"gun_exit_"+key: value.copy() for key, value in zip(
        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"), initial)})
    arrays.update({"gun_"+key: np.asarray(getattr(gun, key)).copy()
        for key in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s")})
    arrays.update({"emission_"+key: np.asarray(value).copy() for key, value in gun.emission_reference.items()})
    if _cache is not None:
        _cache.update(gun_signature=signature, gun=gun, segments=dict(next_segments))
    emission_hashes = {key: _array_hash(value) for key, value in arrays.items()
        if key.startswith("emission_") or key in ("weights", "source_id")}
    return arrays, dict(status="COMPLETE", source_current_a=current, source_points=len(weights),
        gun_dependency_signature=signature, input_digest=original.digest,
        gun_reused=gun_reused, column_prefix_reused=any(row["prefix_reused"] for row in receipts),
        actual_trace_batches=len(receipts),
        emitted_identity_digest=sha256(json.dumps(emission_hashes, sort_keys=True).encode()).hexdigest(),
        implementation=original.to_dict()["implementation"], batches=receipts,
        integration_mesh_sha256=_array_hash(common_mesh),
        source_sha256={key: _array_hash(value) for key, value in arrays.items()
            if key.startswith(("gun_exit_", "emission_")) or key in ("weights", "source_id", "source_energy_offset_ev")})


class ProfileEvaluator:
    """One actual profile execution per trial with production dependency proofs.

    Controls are existing physical fields. Cached sources and section prefixes
    are admitted only by the production gun signature and exact restart checks.
    """
    def __init__(self, state, controls, *, offsets_nm=(-20., -5., -1., 0., 1., 5., 20.),
                 cancelled=lambda: False, batch_size=128):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from solve_joint_probe import control_space
        controls = tuple(controls)
        checked = control_space(state, tuple(c.name for c in controls),
            {c.name: [c.lower, c.upper] for c in controls})
        if controls != checked:
            raise ValueError("Profile controls do not match the physical control contract")
        offsets = tuple(sorted(set(finite(value, "Profile offset") for value in offsets_nm)))
        if len(offsets) < 3 or len(offsets) > MAX_PLANES or offsets[0] >= 0 or offsets[-1] <= 0 or 0. not in offsets:
            raise ValueError("Profile must contain exact surface and physical points on both sides")
        self.snapshot = capture_instrument_snapshot(state)
        self.controls, self.offsets_nm = controls, offsets
        self.cancelled, self.batch_size, self._cache = cancelled, batch_size, {}

    def evaluate(self, values):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from validate_joint_probe import _restore_exact
        from solve_joint_probe import set_values, prepare_objective_observations, Observation
        from temsim.physics.particle_sections import section_limits
        state = _restore_exact(self.snapshot)
        set_values(state, self.controls, values)
        preparation = prepare_objective_observations(state)
        before = capture_instrument_snapshot(state)
        surface = float(state.sample.upper_surface_z_mm)
        planes = np.unique(np.r_[surface+np.asarray(self.offsets_nm)*1e-6, surface])
        lower, upper = section_limits(state)
        if not lower < planes[0] < surface < planes[-1] <= upper:
            raise ValueError("Profile exceeds the physical straight-column domain")
        arrays, receipt = execute_scan(state, planes, self.cancelled, batch_size=self.batch_size, _cache=self._cache)
        try:
            analysis = analyze_scan(arrays, planes, surface_z_mm=surface, source_current_a=receipt["source_current_a"])
        except ValueError as error:
            # A completed, physically blocked trial is usable rejection
            # evidence. All identity/shape/reappearance checks ran before this
            # specific failure. Validate every nonempty plane too, so an empty
            # plane cannot conceal nonfinite active phase or invalid energy.
            if str(error) != "Nonfinite or missing actual positive-current phase at an observed plane":
                raise
            z, alive, weights = (np.asarray(arrays[key]) for key in ("z_mm", "alive", "weights"))
            supported = [plane for plane in planes if np.any(alive[int(np.flatnonzero(z == plane)[0])] & (weights > 0))]
            if len(supported) == len(planes):
                raise
            if supported:
                analyze_scan(arrays, np.asarray(supported), surface_z_mm=float(supported[0]),
                    source_current_a=receipt["source_current_a"])
            analysis = dict(status="NO_NUMERICAL_SUPPORT", planes=[],
                surface=dict(z_mm=surface, d95_nm=None),
                reason="Actual completed transport has no positive-current survivors at one or more requested profile planes")
        if capture_instrument_snapshot(state).digest != before.digest:
            raise ValueError("Profile observation changed candidate inputs")
        report = dict(receipt, profile=analysis["planes"], surface=analysis["surface"], analysis=analysis,
            input_preparation=preparation, qualification="PROPOSAL_ONLY",
            scope="vacuum-probe-only; native D95 profile; sparse objective samples are not a waist qualification")
        if analysis.get("status") == "NO_NUMERICAL_SUPPORT":
            report.update(status="NO_NUMERICAL_SUPPORT", reason=analysis["reason"])
        return Observation(report, arrays, state)


def observed_minimum(z, values, surface):
    """A discrete minimum and its actual neighbours, never an interpolated root."""
    values = np.asarray(values)
    indices = np.flatnonzero(values == np.min(values))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    intervals = [[float(z[g[0]]), float(z[g[-1]])] for g in groups]
    boundary = indices[0] == 0 or indices[-1] == len(z)-1
    bracket = [float(z[max(0, indices[0]-1)]), float(z[min(len(z)-1, indices[-1]+1)])]
    status = "UNRESOLVED_BOUNDARY_MINIMUM" if boundary else (
        "UNRESOLVED_MULTIPLE_MINIMA" if len(groups) > 1 else "BOUNDED_OBSERVED_MINIMUM")
    return dict(status=status, minimum=float(values[indices[0]]),
        observed_minimum_planes_mm=[float(z[i]) for i in indices], plateau_intervals_mm=intervals,
        neighbour_bracket_mm=bracket, neighbour_bracket_offset_nm=[(v-surface)*1e6 for v in bracket],
        representative_offset_nm=float((z[indices[0]]-surface)*1e6),
        interpretation="Smallest observed value; no claim of an exact continuous argmin")


def analyze_scan(arrays, planes, *, surface_z_mm, source_current_a, placement_tolerance_nm=1.):
    """Compute native quantiles on each actual positive-current population."""
    from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile
    z, weights, ids, alive = (np.asarray(arrays[key]) for key in ("z_mm", "weights", "source_id", "alive"))
    if (z.ndim != 1 or len(z) < 3 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
            or weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0)
            or not math.isfinite(float(weights.sum())) or weights.sum() <= 0
            or ids.shape != weights.shape or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or len(np.unique(ids)) != len(ids) or alive.dtype.kind != "b" or alive.shape != (len(z), len(weights))
            or any(np.shape(arrays[key]) != alive.shape for key in PHASE)):
        raise ValueError("Invalid original checkpoint population")
    if np.any(~alive[:-1] & alive[1:] & (weights > 0)[None, :]):
        raise ValueError("A stopped positive-current source particle reappeared")
    if any(np.count_nonzero(z == value) != 1 for value in planes):
        raise ValueError("Missing exact through-focus checkpoint")
    current = finite(source_current_a, "Source current")
    tolerance = finite(placement_tolerance_nm, "Placement tolerance")
    if current <= 0 or tolerance <= 0:
        raise ValueError("Current and placement tolerance must be positive")
    rows, populations = [], []
    for plane in planes:
        i = int(np.flatnonzero(z == plane)[0])
        selected = alive[i] & (weights > 0)
        if not selected.any() or any(not np.isfinite(np.asarray(arrays[key])[i, selected]).all() for key in PHASE):
            raise ValueError("Nonfinite or missing actual positive-current phase at an observed plane")
        if np.any(arrays["kinetic_energy_ev"][i, selected] <= 0) or np.any(arrays["flight_time_s"][i, selected] < 0):
            raise ValueError("Observed live energy must be positive and clocks nonnegative")
        stats = transverse_beam_statistics(*(arrays[key][i] for key in PHASE[:4]), alive=selected, weights=weights)
        w = weights[selected]/weights[selected].sum()
        radius = np.hypot(arrays["x_m"][i, selected]-stats.mean_x_m,
                          arrays["y_m"][i, selected]-stats.mean_y_m)
        rows.append(dict(z_mm=float(plane), offset_nm=float((plane-surface_z_mm)*1e6),
            d50_nm=2e9*_weighted_quantile(radius, w, .5), d90_nm=2e9*_weighted_quantile(radius, w, .9),
            d95_nm=2e9*stats.radius_95_m, rms_radius_nm=1e9*stats.radius_rms_m,
            alpha95_mrad=stats.convergence_95_mrad, centroid_x_m=stats.mean_x_m, centroid_y_m=stats.mean_y_m,
            current_a=current*float(weights[selected].sum()/weights.sum()),
            absolute_source_weight=float(weights[selected].sum()), surviving_rays=int(selected.sum()),
            effective_rays=float(1/(w@w)), source_ids=ids[selected].tolist()))
        populations.append(selected)
    if any(not np.isfinite([row[key] for key in ("d50_nm", "d90_nm", "d95_nm", "rms_radius_nm", "alpha95_mrad")]).all() for row in rows):
        raise ValueError("Nonfinite native through-focus statistic")
    positions = np.asarray(planes)
    at_surface = [row for row in rows if row["z_mm"] == surface_z_mm]
    if len(at_surface) != 1:
        raise ValueError("The exact specimen upper surface was not observed")
    surface = at_surface[0]
    d95 = observed_minimum(positions, [row["d95_nm"] for row in rows], surface_z_mm)
    rms = observed_minimum(positions, [row["rms_radius_nm"] for row in rows], surface_z_mm)
    stable = all(np.array_equal(populations[0], row) for row in populations[1:])
    left, right = d95["neighbour_bracket_mm"]
    low, high = surface_z_mm-tolerance*1e-6, surface_z_mm+tolerance*1e-6
    bounded = d95["status"] == "BOUNDED_OBSERVED_MINIMUM" and stable
    placement = bounded and low <= left <= right <= high
    size = 0 <= surface["d95_nm"] < .7
    support = all(row["surviving_rays"] >= 16 and row["effective_rays"] >= 16*(1-32*np.finfo(float).eps) for row in rows)
    unresolved = not bounded or (not placement and not (right < low or left > high)) or not support
    passed = placement and size and support
    return dict(planes=rows, surface=surface, d95_waist=d95, rms_waist_diagnostic=rms,
        positive_current_membership_stable=stable,
        focus_definition="Native D95(z) discrete minimum about each actual current-weighted centroid",
        surface_gate=dict(status="BOUNDED_D95_WAIST_AT_SURFACE" if passed else "UNRESOLVED" if unresolved else "FAIL",
            passed=passed, placement_passed=placement, strict_surface_d95_passed=size,
            support_passed=support, placement_tolerance_nm=tolerance, maximum_d95_nm=.7,
            rms_and_d90_role="Diagnostic only; no acceptance gate"))


def measure(state, output_directory, *, window_nm=100., spacing_nm=1., extra_z_offset_nm=(),
            threads=8, time_budget_s=1800., cancel_file=None, batch_size=128, candidate=None, executor=None):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from validate_joint_probe import _restore_exact
    if type(threads) is not int or threads < 1:
        raise ValueError("Threads must be a positive integer")
    budget = finite(time_budget_s, "Time budget")
    if budget <= 0:
        raise ValueError("Time budget must be positive")
    extra_z_offset_nm = tuple(extra_z_offset_nm)
    planes = capture_planes(state, window_nm=window_nm, spacing_nm=spacing_nm, extra_z_offset_nm=extra_z_offset_nm)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    original = capture_instrument_snapshot(state)
    started = perf_counter()
    def cancelled():
        return perf_counter()-started >= budget or bool(cancel_file and Path(cancel_file).exists())
    report = dict(schema=SCHEMA, utc=datetime.now(timezone.utc).isoformat(), execution_status="RUNNING", passed=False,
        candidate=candidate, input_digest=original.digest, implementation=original.to_dict()["implementation"],
        observer_sha256=_hash(Path(__file__)), scope="vacuum-probe-only; classical unified Lorentz fields; no specimen scattering",
        numerical_convergence="NOT_EVALUATED", source_count_convergence="NOT_EVALUATED", topology="NOT_EVALUATED",
        qualification="NOT_QUALIFIED", original_state_unchanged=False,
        parameters=dict(window_nm=window_nm, spacing_nm=spacing_nm, extra_z_offset_nm=list(extra_z_offset_nm),
            requested_planes_mm=planes.tolist(), threads=threads, batch_size=batch_size,
            time_budget_s=budget, cancellation="Cooperative checks; not a hard process deadline"), errors=[])
    destination = output / "report.json"
    _json(output / "inputs.json", original.to_dict())
    _json(destination, report)
    try:
        if cancelled():
            raise InterruptedError("Observation cancelled before execution")
        detached = _restore_exact(original)
        from numba import get_num_threads, set_num_threads
        from threadpoolctl import threadpool_limits
        old_threads = get_num_threads()
        try:
            set_num_threads(threads)
            with threadpool_limits(limits=threads):
                arrays, receipt = (executor or execute_scan)(detached, planes, cancelled, batch_size=batch_size)
        finally:
            set_num_threads(old_threads)
        if capture_instrument_snapshot(detached).digest != original.digest:
            raise ValueError("Observation changed candidate settings")
        if receipt.get("input_digest") != original.digest or receipt.get("status") != "COMPLETE" or receipt.get("implementation") != original.to_dict()["implementation"]:
            raise ValueError("Execution receipt does not match exact captured inputs")
        if not receipt.get("batches") or any(row.get("field_audit", {}).get("passed") is not True for row in receipt["batches"]):
            raise ValueError("Executed field audit is incomplete")
        _npz(output / "particles.npz", arrays)
        report["receipt"] = receipt
        report["analysis"] = analyze_scan(arrays, planes, surface_z_mm=float(detached.sample.upper_surface_z_mm),
            source_current_a=receipt["source_current_a"])
        report["bounded_surface_gate_passed"] = report["analysis"]["surface_gate"]["passed"]
        report["execution_status"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as error:
        report["execution_status"] = "CANCELLED" if cancelled() or isinstance(error, (KeyboardInterrupt, InterruptedError)) else "FAILED"
        report["errors"].append(dict(type=type(error).__name__, message=str(error)))
    finally:
        report["original_state_unchanged"] = capture_instrument_snapshot(state).digest == original.digest
        if not report["original_state_unchanged"]:
            report["execution_status"] = "FAILED"
            report["errors"].append(dict(type="StateMutation", message="Original candidate inputs changed"))
        report["elapsed_s"] = perf_counter()-started
        report["files_sha256"] = {name: _hash(output/name) for name in ("inputs.json", "particles.npz") if (output/name).exists()}
        _json(destination, report)
        _json(output / "checksums.json", {**report["files_sha256"], "report.json": _hash(destination)})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-nm", type=float, default=100.)
    parser.add_argument("--spacing-nm", type=float, default=1.)
    parser.add_argument("--extra-z-offset-nm", type=float, action="append", default=[])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--time-budget-s", type=float, default=1800.)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    from validate_joint_probe import load_candidate
    try:
        state, identity = load_candidate(args.candidate)
        if identity["restoration_kind"] != "complete_snapshot":
            raise ValueError("A complete instrument snapshot is required")
        report = measure(state, args.output, window_nm=args.window_nm, spacing_nm=args.spacing_nm,
            extra_z_offset_nm=args.extra_z_offset_nm, threads=args.threads, time_budget_s=args.time_budget_s,
            cancel_file=args.cancel_file, batch_size=args.batch_size, candidate=identity)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(2, f"Waist observation failed before execution: {error}\n")
    print(json.dumps(dict(execution_status=report["execution_status"], qualification=report["qualification"],
        report=str(args.output.resolve()/"report.json"))))
    return 0 if report["execution_status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
