"""Executed classical focus profiles for the current FEG extractor."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / name) for name in ("src", "scripts")]

from measure_probe_waist import PHASE, _array_hash, _hash, _json, _npz, analyze_scan, audit_extended_plan
from solve_joint_probe import prepare_objective_observations
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.particle_sections import _events, _selection_boundary, _trace_segment, gun_dependency_signature
from validate_joint_probe import _restore_exact
from validate_physical_hybrid import topology_intervals

SOURCE_MINIMUM = 1000
FOCUS_KEYS = ("mini_condenser", "objective_lens")
OPTICAL_DRIVES = {"percent", "strength_m3", "skew_strength_m3", "strength_m2",
                  "strength_x_percent", "strength_y_percent"}
PRIMARY_HEXAPOLE_AXES = {"probe_hp1_hexapole", "probe_hp2_hexapole"}
LOCAL_OBJECTIVE_SENSITIVITY_NM_PER_PERCENT = -572942.1967990099
TARGET_ALPHA_MRAD = 30.0


def load_executed_gun(directory):
    """Read only a current, complete production gun receipt with intact arrays."""
    from temsim.calculation_manifest import solver_source_identity
    from temsim.optics.electron_gun.base import GunExitBundle, GunTraceResult
    from temsim.physics.simulation import _validate_gun_flight_times

    directory = Path(directory).resolve()
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    snapshot = InstrumentSnapshot.from_dict(json.loads((directory / "inputs.json").read_text(encoding="utf-8")))
    if (receipt["status"] != "COMPLETE" or receipt["arm"] != "new"
            or receipt["input_digest"] != snapshot.digest
            or receipt["implementation"] != snapshot.implementation
            or snapshot.implementation != solver_source_identity()):
        raise ValueError("Current completed new-extractor receipt and exact snapshot required")
    with np.load(directory / "source.npz", allow_pickle=False) as source:
        arrays = dict(source)
    if {key: _array_hash(value) for key, value in arrays.items()} != receipt["source_sha256"]:
        raise ValueError("Executed gun source checksum mismatch")
    with np.load(directory / "launch.npz", allow_pickle=False) as source:
        launch = dict(source)
    if {key: _array_hash(value) for key, value in launch.items()} != receipt["launch_sha256"]:
        raise ValueError("Original emitted population checksum mismatch")
    bundle = GunExitBundle(**{key: arrays["gun_exit_" + key] for key in
        ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "energy_offset_ev", "weight", "ray_id", "alive")})
    emission = {key.removeprefix("emission_"): value for key, value in arrays.items()
                if key.startswith("emission_")}
    weights = np.asarray(bundle.weight)
    if (weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0.)
            or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)
            or np.count_nonzero(weights > 0) < SOURCE_MINIMUM
            or np.count_nonzero(weights > 0) != receipt["emitted_positive_current_points"]
            or len(np.unique(bundle.ray_id)) != len(weights)
            or not np.array_equal(bundle.ray_id, emission["ray_id"])
            or not np.array_equal(bundle.ray_id, launch["ray_id"])
            or not np.array_equal(weights, launch["weight"])
            or not np.array_equal(bundle.ray_id, arrays["source_id"])
            or not np.array_equal(weights, arrays["weights"])):
        raise ValueError("Gun cache must preserve at least 1000 distinct original positive-current emissions")
    gun = GunTraceResult(**{key: arrays["gun_" + key] for key in
        ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "blocked_z_mm")},
        exit_bundle=bundle, blocked_key=tuple(arrays["gun_blocked_key"].tolist()),
        emitted_current_a=float(receipt["reference_current_a"]),
        dpa_transmitted_current_a=None, c1_transmitted_current_a=None,
        emission_reference=emission)
    if not _validate_gun_flight_times(gun):
        raise ValueError("Executed gun cache lacks physical first-arrival histories")
    return snapshot, gun, receipt


def set_optics(state, controls):
    components = {item.key: item for item in (*state.lenses, *state.corrector_elements, *state.stigmators)}
    changes = []
    for key, fields in controls.items():
        item = components[key]
        for field, value in fields.items():
            field_allowed = field in OPTICAL_DRIVES or (field == "orientation_rad" and key in PRIMARY_HEXAPOLE_AXES)
            if (not field_allowed or type(value) not in (float, int)
                    or not math.isfinite(value)):
                raise ValueError(f"Invalid physical optical drive: {key}.{field}")
            if field == "percent" and not 0. <= value <= item.max_percent:
                raise ValueError(f"Excitation exceeds installed supply limits: {key}")
            changes.append(dict(key=key, field=field, before=getattr(item, field), after=value))
            setattr(item, field, float(value))
        if hasattr(item, "validate"):
            item.validate()
    prepare_objective_observations(state)
    return changes


def observation_planes(state, *, window_nm):
    surface = float(state.sample.upper_surface_z_mm)
    offsets = np.unique(np.r_[np.linspace(-window_nm, window_nm, 151),
        np.arange(-2500., 2501., 20.), np.arange(-250., 251., 2.),
        np.arange(-8., 8.125, .25)])
    offsets = offsets[np.abs(offsets) <= window_nm]
    focus = np.unique(np.r_[surface + offsets * 1e-6, state.sample.z_mm,
                            state.sample.lower_surface_z_mm])
    intervals = list(topology_intervals(state))
    label, lo, _, step = intervals[-1]
    intervals[-1] = (label, lo, float(focus[-1]), step)
    topology = np.concatenate([np.linspace(lo, hi, math.ceil((hi - lo) / step) + 1)
                               for _, lo, hi, step in intervals])
    apertures = [float(item.z_mm) for item in state.apertures
                 if state.electron_gun.exit_plane_z_mm < item.z_mm < focus[-1]
                 and item.enabled and getattr(item, "installed", True) and getattr(item, "inserted", True)]
    return focus, intervals, np.unique(np.r_[focus, topology, apertures])


def narrow_restart_boundary(state, previous, start, stop, planes, boundary, ray_count):
    """Find a verified cutoff quickly before the native continuation rechecks it."""
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.instrument_magnetic import events_overlapping_interval
    from temsim.physics.simulation import _column_checkpoint_planes
    from temsim.physics.unified_particle_restart import unified_prefix_matches
    from temsim.calculation_manifest import solver_source_identity

    if previous is None or state.vacuum_map.enabled:
        return boundary
    old, checkpoints = previous.plan, previous.checkpoints
    implementation = solver_source_identity()
    retained = checkpoints.z_mm[(checkpoints.z_mm >= start) & (checkpoints.z_mm <= stop)]
    # All candidate rows must belong to the same endpoint-exact mesh that
    # _trace_segment will construct. The cutoff below uses an existing row.
    observations = tuple(sorted(set((start, stop, *planes, *retained,
        *_column_checkpoint_planes(start, stop, ray_count)))))
    events = events_overlapping_interval(state, _events(state), start, stop)
    plan = build_propagation_plan(state, start, stop, events,
        save_z_mm=observations, checkpoint_z_mm=observations)
    candidates = np.flatnonzero((checkpoints.z_mm > start) & (checkpoints.z_mm < min(boundary, stop)))
    low, high = -1, len(candidates)
    while high - low > 1:
        test = (low + high) // 2
        row = int(candidates[test])
        z = float(checkpoints.z_mm[row])
        oi, ni = (int(np.searchsorted(item.z_mm, z)) for item in (old, plan))
        if (oi < len(old.z_mm) and ni < len(plan.z_mm) and old.z_mm[oi] == plan.z_mm[ni] == z
                and unified_prefix_matches(old, plan, checkpoints, row, oi, ni,
                    implementation_digest=implementation)):
            low = test
        else:
            high = test
    return float(np.nextafter(checkpoints.z_mm[candidates[low]], math.inf)) if low >= 0 else start


def execute_profile(state, planes, gun, receipt, *, previous=None, boundary=math.inf, cancelled=lambda: False):
    """Trace the complete original source in one production section with safe reuse."""
    from temsim.physics.beam_current import effective_source_current_a
    from temsim.physics.ray_identity import emitted_source_identity
    from temsim.physics.unified_field_runtime import unified_runtime_preflight
    from temsim.specimen.source import specimen_is_vacuum
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source

    original = capture_instrument_snapshot(state)
    require_physical_gun_source(state.electron_gun)
    state._tuning_cancelled = cancelled
    signature = gun_dependency_signature(state)
    if (not signature or signature != receipt["gun_dependency_signature"]
            or capture_instrument_snapshot(state).digest != original.digest
            or effective_source_current_a(state) != gun.emitted_current_a
            or len(gun.exit_bundle.weight) != state.electron_gun.emitter.ray_count):
        raise ValueError("Executed gun is incompatible with the exact candidate source or its dependencies")
    issues = unified_runtime_preflight(state)["issues"]
    if (issues or not specimen_is_vacuum(state.sample) or state.simulation_mode != "unified"
            or state.electron_gun.source_representation != "classical_particles"):
        raise ValueError("Unsupported physical vacuum diagnostic: " + "; ".join(issues))
    if cancelled():
        raise InterruptedError("Focus evaluation cancelled before column execution")
    bundle = gun.exit_bundle
    ids, azimuth = emitted_source_identity(gun)
    weights = np.asarray(bundle.weight)
    offsets = np.asarray(bundle.energy_offset_ev)
    initial = tuple(np.asarray(getattr(bundle, key)) for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    survival = (np.asarray(bundle.alive), np.asarray(gun.blocked_z_mm), list(gun.blocked_key))
    start, stop = float(state.electron_gun.exit_plane_z_mm), float(planes[-1])
    selected_boundary = narrow_restart_boundary(state, previous, start, stop, planes, boundary, len(weights))
    segment, resume, reused = _trace_segment(state, "incident", start, stop,
        initial, offsets, weights, (ids, azimuth), survival, _events(state), previous, selected_boundary,
        initial_kicks=True, save_z_mm=tuple(planes),
        initial_kinetic_energy_ev=float(state.electron_gun.nominal_exit_energy_ev) + offsets)
    cp, branch = segment.checkpoints, segment.branch
    if (capture_instrument_snapshot(state).digest != original.digest
            or not np.array_equal(branch.source_ray_id, ids)
            or not np.array_equal(branch.ray_weight, weights)):
        raise ValueError("Execution changed original inputs or emitted source identity")
    requested = np.searchsorted(cp.z_mm, planes)
    if np.any(requested >= len(cp.z_mm)) or not np.array_equal(cp.z_mm[requested], planes):
        raise ValueError("Requested native physical plane is missing")
    arrays = {key: np.asarray(getattr(cp, key)) for key in PHASE}
    arrays.update(z_mm=np.asarray(cp.z_mm), integration_z_mm=np.asarray(segment.plan.z_mm),
        alive=np.asarray(cp.unified_state.active), weights=weights, source_id=ids,
        source_azimuth_rad=azimuth, gun_exit_alive=survival[0],
        blocked_z_mm=np.asarray(branch.blocked_z), blocked_key=np.asarray(branch.blocked_key, dtype=str),
        terminal_alive=np.asarray(branch.alive))
    audit = audit_extended_plan(state, segment.plan, stop)
    info = dict(input_digest=original.digest, implementation=original.implementation,
        gun_reused=True, gun_dependency_signature=signature, gun_input_digest=receipt["input_digest"],
        source_current_a=gun.emitted_current_a, original_source_count=len(weights),
        emitted_positive_current_points=int(np.count_nonzero(weights > 0)),
        column_prefix_reused=bool(reused), column_resume_z_mm=resume,
        verified_search_boundary_z_mm=selected_boundary if math.isfinite(selected_boundary) else None,
        source_identity_sha256=sha256(json.dumps(receipt["source_sha256"], sort_keys=True).encode()).hexdigest(),
        integration_nodes=len(segment.plan.z_mm), checkpoint_rows=len(cp.z_mm),
        integration_mesh_sha256=_array_hash(segment.plan.z_mm), field_audit=audit)
    return arrays, info, segment


def phase_summary(arrays, index, mask, current):
    weights = arrays["weights"]
    keep = mask & (weights > 0.)
    result = dict(count=int(keep.sum()), current_a=float(current * weights[keep].sum()),
                  source_current_fraction=float(weights[keep].sum()))
    if np.any(keep):
        stats = transverse_beam_statistics(*(arrays[key][index] for key in PHASE[:4]),
                                           alive=keep, weights=weights)
        result.update(alpha95_mrad=stats.convergence_95_mrad,
                      rms_angle_mrad=1e3 * stats.convergence_rms_rad,
                      d95_nm=2e9 * stats.radius_95_m, rms_radius_nm=1e9 * stats.radius_rms_m,
                      radial_covariance_m_rad=stats.radial_position_angle_covariance_m_rad,
                      rms_tangent_drift_offset_nm=1e9 * stats.waist_offset_m)
    return result


def surface_core_and_tails(arrays, row, current):
    valid = arrays["alive"][row] & (arrays["weights"] > 0.)
    columns = np.flatnonzero(valid)
    w = arrays["weights"][columns]
    x, y = (arrays[key][row, columns] for key in ("x_m", "y_m"))
    axis_radius = np.hypot(x, y)
    result = dict(fixed_optical_axis_50pm_current_a=float(current * w[axis_radius <= 50e-12].sum()),
                  fixed_optical_axis_50pm_positive_count=int(np.count_nonzero(axis_radius <= 50e-12)),
                  surface_tail_original_rays=[])
    if not len(columns):
        return result
    center = [np.average(x, weights=w), np.average(y, weights=w)]
    radius = np.hypot(x-center[0], y-center[1])
    for index in np.argsort(radius)[-5:][::-1]:
        col = columns[index]
        points = []
        for plane in (row-1, row, row+1):
            if 0 <= plane < len(arrays["z_mm"]) and arrays["alive"][plane, col]:
                px, py, tx, ty = (float(arrays[key][plane, col]) for key in PHASE[:4])
                points.append(dict(z_mm=float(arrays["z_mm"][plane]),
                    optical_axis_radius_nm=1e9 * math.hypot(px, py),
                    slope_angle_from_axis_mrad=1e3 * math.atan(math.hypot(tx, ty))))
        result["surface_tail_original_rays"].append(dict(source_id=int(arrays["source_id"][col]),
            centroid_radius_nm=1e9 * float(radius[index]), source_current_a=float(current * w[index]),
            native_surface_neighbourhood=points))
    return result


def analyze_topology(state, arrays, intervals, current):
    from temsim.optics.waist_evidence import finite_interval, observed_aperture_losses

    z = arrays["z_mm"]
    surface_row = int(np.searchsorted(z, state.sample.upper_surface_z_mm))
    actual = arrays["alive"]
    common = actual & actual[surface_row]
    weights, ids = arrays["weights"], arrays["source_id"]
    cp = SimpleNamespace(**{key: arrays[key] for key in ("z_mm", *PHASE[:4])})
    losses = observed_aperture_losses(state, cp, actual, weights, ids)
    report = {}
    for label, lo, hi, step in intervals:
        result = {}
        for kind, masks in (("actual_population", actual), ("sample_surviving_cohort", common)):
            evidence = finite_interval(cp, masks, weights, lo, hi, maximum_spacing_mm=step + 1e-9,
                source_ids=ids, known_interceptions=losses)
            for region in evidence["regions"]:
                index = int(np.searchsorted(z, region["representative_z_mm"]))
                region["current_phase_space"] = phase_summary(arrays, index, masks[index], current)
            result[kind] = evidence
        report[label] = result
    return report


def topology_summary(topology):
    expected = {
        "condenser_lens_2 to condenser_lens_3": 1,
        "condenser_lens_3 to adapter_lens": 0,
        "probe_tl22_lens to probe_tl21_lens": 1,
        "ac_upper to ac_lower": 1,
        "ac_lower to mini_condenser": 0,
        "mini_condenser to specimen_entrance": 1,
    }
    return {label: {kind: dict(status=evidence["status"], expected_regions=expected.get(label),
                    observed_regions=len(evidence["regions"]),
                    centers_mm=[row["representative_z_mm"] for row in evidence["regions"]],
                    radial_candidates_mm=[row["representative_z_mm"] for row in evidence["radial_candidates"]
                                          if "representative_z_mm" in row])
                    for kind, evidence in populations.items()}
            for label, populations in topology.items()}


def local_tangent_proposal(arrays, profile, surface):
    """Propose sub-grid motion from an actual phase plane, without certifying it."""
    from temsim.physics.beam_statistics import _weighted_quantile

    waist = profile["d95_waist"]
    if (waist["status"] != "BOUNDED_OBSERVED_MINIMUM"
            or len(waist["observed_minimum_planes_mm"]) != 1
            or not profile["positive_current_membership_stable"]):
        return None
    z0 = waist["observed_minimum_planes_mm"][0]
    row = int(np.searchsorted(arrays["z_mm"], z0))
    weights = arrays["weights"]
    keep = arrays["alive"][row] & (weights > 0.)
    weights = weights[keep]
    centered = [v - np.average(v, weights=weights)
                for v in (arrays[key][row, keep] for key in PHASE[:4])]
    x, y, tx, ty = centered

    def diameter(z_offset_nm):
        distance_m = (z_offset_nm - (z0-surface)*1e6)*1e-9
        return 2e9 * _weighted_quantile(np.hypot(x + tx*distance_m, y + ty*distance_m), weights, .95)

    lo, hi = waist["neighbour_bracket_offset_nm"]
    offset = waist["representative_offset_nm"]
    for size in (201, 101):
        grid = np.linspace(lo, hi, size)
        i = int(np.argmin([diameter(item) for item in grid]))
        offset = float(grid[i])
        lo, hi = float(grid[max(i-1, 0)]), float(grid[min(i+1, size-1)])
    return dict(offset_nm=offset, predicted_d95_nm=diameter(offset),
                status="Local tangent proposal only; no replacement for executed D95 bracket")


def analyze_trial(state, arrays, focus, intervals, info):
    surface = float(state.sample.upper_surface_z_mm)
    current = info["source_current_a"]
    z = arrays["z_mm"]
    row = int(np.searchsorted(z, surface))
    weights, keys = arrays["weights"], arrays["blocked_key"]
    if np.any(~arrays["terminal_alive"] & (keys == "")):
        raise ValueError("Column current audit found an unlabelled stop")
    losses = {key: dict(count=int(np.count_nonzero((keys == key) & (weights > 0.))),
                        current_a=float(current * weights[keys == key].sum()))
              for key in sorted(set(keys)) if key}
    total = current * weights.sum()
    terminal_current = current * weights[arrays["terminal_alive"]].sum()
    residual = float(total - terminal_current - sum(loss["current_a"] for loss in losses.values()))
    if abs(residual) > 1e-12 * current:
        raise ValueError("Original current accounting failed")
    result = dict(surface=phase_summary(arrays, row, arrays["alive"][row], current),
                  incident_current_a=float(total),
                  gun_exit_current_a=float(current * weights[arrays["gun_exit_alive"]].sum()),
                  terminal_current_a=float(terminal_current), per_stage_losses=losses,
                  absolute_current_balance_residual_a=residual)
    result["surface"].update(surface_core_and_tails(arrays, row, current))
    result["aperture_plane_phase"] = {}
    for item in state.apertures:
        index = int(np.searchsorted(z, item.z_mm))
        if index < len(z) and z[index] == item.z_mm:
            result["aperture_plane_phase"][item.key] = phase_summary(arrays, index, arrays["alive"][index], current)
    supported = np.any(arrays["alive"][np.searchsorted(z, focus)] & (weights > 0.), axis=1)
    if np.all(supported):
        profile = analyze_scan(arrays, focus, surface_z_mm=surface, source_current_a=current)
        for plane in profile["planes"]:
            plane.pop("source_ids")
        focus_sizes = np.asarray([plane["d95_nm"] for plane in profile["planes"]])
        local_minima = np.flatnonzero((focus_sizes[1:-1] < focus_sizes[:-2])
                                     & (focus_sizes[1:-1] < focus_sizes[2:])) + 1
        profile["smallest_observed_local_minima"] = [
            {key: profile["planes"][i][key] for key in ("z_mm", "offset_nm", "d95_nm")}
            for i in local_minima[np.argsort(focus_sizes[local_minima])[:8]]]
        profile["local_tangent_proposal"] = local_tangent_proposal(arrays, profile, surface)
        result["focus"] = profile
        result["surface"]["alpha95_error_from_target_mrad"] = result["surface"]["alpha95_mrad"] - TARGET_ALPHA_MRAD
    else:
        result["focus"] = dict(status="NO_POSITIVE_CURRENT_SUPPORT_ON_FULL_PROFILE",
            unsupported_offsets_nm=((focus[~supported] - surface) * 1e6).tolist())
    result["topology"] = analyze_topology(state, arrays, intervals, current)
    result["topology_summary"] = topology_summary(result["topology"])
    return result


def objective_proposal(history, original_objective, installed_limit, diagnostic_slope):
    last = history[-1]
    profile = last["focus"]
    if "d95_waist" not in profile or profile["surface_gate"]["passed"]:
        return None
    objective = last["controls"]["objective_lens"]
    offset = profile["d95_waist"]["representative_offset_nm"]
    local_proposal = profile.get("local_tangent_proposal")
    if local_proposal:
        offset = local_proposal["offset_nm"]
    derivative = diagnostic_slope
    origin = "Local zero-current objective response used as proposal seed"
    if len(history) > 1 and profile["d95_waist"]["status"] == "BOUNDED_OBSERVED_MINIMUM":
        previous = history[-2]
        previous_waist = previous["focus"].get("d95_waist")
        delta = objective - previous["controls"]["objective_lens"]
        if previous_waist is not None and previous_waist["status"] == "BOUNDED_OBSERVED_MINIMUM" and delta:
            previous_proposal = previous["focus"].get("local_tangent_proposal")
            previous_offset = (previous_proposal["offset_nm"] if previous_proposal
                               else previous_waist["representative_offset_nm"])
            measured = (offset - previous_offset) / delta
            if -5e6 < measured < -2e4:
                derivative = measured
                origin = "Same-original-source measured objective secant"
    if profile["d95_waist"]["status"] == "UNRESOLVED_BOUNDARY_MINIMUM":
        drift = last["surface"].get("rms_tangent_drift_offset_nm")
        if drift is not None and np.isfinite(drift) and abs(drift) > abs(offset) and drift * offset > 0.:
            offset = drift
            origin += "; RMS tangent used only to propose a wider move"
    proposed = objective + float(np.clip(-offset / derivative, -.08, .08))
    proposed = float(np.clip(proposed, max(0., original_objective - .25),
                            min(installed_limit, original_objective + .25)))
    if any(abs(proposed - trial["controls"]["objective_lens"]) < 1e-10 for trial in history):
        return None
    return proposed, dict(slope_nm_per_percent=derivative, estimated_offset_nm=offset, origin=origin,
                          used_subgrid_tangent_proposal=bool(local_proposal))


def await_next_controls(directory, trial, seconds, cancelled, publish):
    command = directory / f"trial-{trial:02d}.json"
    publish(f"Awaiting next physical controls at {command}")
    until = time.perf_counter() + seconds
    while time.perf_counter() < until and not cancelled():
        if command.is_file():
            return json.loads(command.read_text(encoding="utf-8-sig")), command
        time.sleep(.5)
    return {}, command


def run(args):
    from temsim.cpu_resources import numerical_job
    from temsim.physics.transport_progress import transport_progress

    if (not np.isfinite([args.step_mm, args.window_nm, args.budget_s, args.inbox_timeout_s, args.objective_slope]).all()
            or min(args.step_mm, args.window_nm, args.budget_s, args.inbox_timeout_s) <= 0.
            or not -5e6 < args.objective_slope < -2e4
            or not 1 <= args.trials <= 8 or not 1 <= args.threads <= 2):
        raise ValueError("Use positive numerical inputs, 1–8 trials and at most two numerical threads")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history = []
    report = dict(status="RUNNING", utc=datetime.now(timezone.utc).isoformat(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        driver_sha256=_hash(Path(__file__)), scope="Classical unified vacuum-side focus at the exact sample upper surface",
        convergence_target_alpha95_mrad=TARGET_ALPHA_MRAD, placement_tolerance_nm=1.,
        requested_step_mm=args.step_mm, window_nm=args.window_nm, requested_trials=args.trials,
        objective_diagnostic_slope_nm_per_percentage_point=args.objective_slope,
        gun_cache=str(args.gun_cache.resolve()), qualification="EXPLORATORY",
        step_and_source_convergence="NOT_EVALUATED", trials=history)

    def write():
        _json(out / "report.json", report)

    def cancelled():
        return (time.perf_counter() - started >= args.budget_s or (out / "CANCEL").exists()
                or bool(args.stop_file and args.stop_file.exists()))

    def log(message):
        print(f"{time.perf_counter() - started:8.1f}s  {message}", flush=True)

    write()
    try:
        snapshot, gun, receipt = load_executed_gun(args.gun_cache)
        state = _restore_exact(snapshot)
        surface = float(state.sample.upper_surface_z_mm)
        report["source_input_changes"] = dict(specimen_mode_before=state.sample.specimen_mode,
            specimen_mode_after="vacuum", cif_path_before=state.sample.cif_path, cif_path_after="",
            archived_step_mm=state.step_mm, executed_step_mm=args.step_mm)
        state.sample.specimen_mode = "vacuum"
        state.sample.cif_path = ""
        for key in ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled"):
            if getattr(state.sample, key, False):
                raise ValueError(f"Candidate carries a coherent request: {key}")
        if float(state.sample.upper_surface_z_mm) != surface:
            raise ValueError("Vacuum diagnostic changed the specimen upper-surface reference")
        state.step_mm = args.step_mm
        report["optical_control_seeds"] = []
        for path in args.controls:
            proposal = json.loads(path.read_text(encoding="utf-8-sig"))
            report["optical_control_seeds"].append(dict(path=str(path.resolve()),
                sha256=_hash(path), changes=set_optics(state, proposal["controls"])))
        prepare_objective_observations(state)
        initial_objective = float(state.objective_lens.percent)
        original_focus = {key: float(next(lens.percent for lens in state.lenses if lens.key == key))
                          for key in FOCUS_KEYS}
        focus, intervals, planes = observation_planes(state, window_nm=args.window_nm)
        boundary = _selection_boundary(state, FOCUS_KEYS)
        report["sample_upper_surface_z_mm"] = surface
        report["focus_only_controls"] = list(FOCUS_KEYS)
        report["objective_focus_baseline"] = original_focus
        report["prefix_search_upper_z_mm"] = boundary
        report["focus_offsets_nm"] = ((focus - surface) * 1e6).tolist()
        previous = None
        proposal = None
        objective_history = []
        previous_progress = [-20.]

        def progress(label):
            now = time.perf_counter() - started
            if now - previous_progress[0] >= 15.:
                log(str(label))
                previous_progress[0] = now

        with numerical_job(requested=args.threads, cancelled=cancelled) as budget, transport_progress(progress):
            report["cpu_receipt"] = budget.to_dict()
            for i in range(args.trials):
                case = out / f"trial-{i:02d}"
                case.mkdir()
                snapshot = capture_instrument_snapshot(state)
                _json(case / "inputs.json", snapshot.to_dict())
                drives = {lens.key: float(lens.percent) for lens in state.lenses if lens.key in FOCUS_KEYS}
                report["running_trial"] = dict(index=i, controls=drives, input_digest=snapshot.digest)
                write()
                log(f"Trial {i}: objective {drives['objective_lens']:.12f}%, {len(planes)} requested axial planes")
                case_started = time.perf_counter()
                arrays, info, previous = execute_profile(state, planes, gun, receipt,
                    previous=previous, boundary=boundary, cancelled=cancelled)
                result = analyze_trial(state, arrays, focus, intervals, info)
                trial = dict(result, execution=info, index=i, controls=drives, proposal=proposal,
                             seconds=time.perf_counter() - case_started)
                selected = np.searchsorted(arrays["z_mm"], planes)
                checkpoint_arrays = {key: (value[selected] if key in (*PHASE, "alive", "z_mm") else value)
                                     for key, value in arrays.items()}
                _npz(case / "observations.npz", checkpoint_arrays)
                trial["observations_sha256"] = _hash(case / "observations.npz")
                _json(case / "report.json", trial)
                compact = {key: value for key, value in trial.items() if key not in ("topology", "focus")}
                compact["focus"] = {key: value for key, value in trial["focus"].items() if key != "planes"}
                history.append(compact)
                objective_history.append(compact)
                del arrays, checkpoint_arrays
                waist = trial["focus"].get("d95_waist", {}).get("representative_offset_nm")
                log(f"Trial {i}: D95 waist offset {waist} nm; alpha95 {trial['surface'].get('alpha95_mrad')} mrad; resume {info['column_resume_z_mm']} mm")
                del trial
                report.pop("running_trial")
                write()
                if i + 1 == args.trials:
                    break
                if args.inbox:
                    report["waiting_for_trial_controls"] = i + 1
                    write()
                    next_spec, path = await_next_controls(args.inbox.resolve(), i + 1, args.inbox_timeout_s,
                                                         cancelled, log)
                    report.pop("waiting_for_trial_controls")
                    if "controls" in next_spec:
                        controls = next_spec["controls"]
                        proposal = dict(origin=str(path), sha256=_hash(path), changes=set_optics(state, controls))
                        if any(key != "objective_lens" for key in controls):
                            objective_history = []
                            initial_objective = float(state.objective_lens.percent)
                        continue
                    if not next_spec.get("auto_objective"):
                        break
                update = objective_proposal(objective_history, initial_objective, state.objective_lens.max_percent,
                                            args.objective_slope)
                if update is None:
                    break
                candidate, proposal = update
                set_optics(state, {"objective_lens": {"percent": candidate}})
        report["status"] = "COMPLETE"
    except (KeyboardInterrupt, InterruptedError):
        report["status"] = "CANCELLED"
        raise
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gun-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controls", type=Path, action="append", default=[],
        help="Optical control document. Supply multiple files to apply them in sequence.")
    parser.add_argument("--step-mm", type=float, default=.2)
    parser.add_argument("--window-nm", type=float, default=15000.)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--budget-s", type=float, default=5400.)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--inbox", type=Path)
    parser.add_argument("--inbox-timeout-s", "--inbox-wait-s", type=float, default=240.)
    parser.add_argument("--objective-slope", type=float, default=LOCAL_OBJECTIVE_SENSITIVITY_NM_PER_PERCENT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
