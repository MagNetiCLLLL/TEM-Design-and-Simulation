"""Measure cumulative geometric Cs at finite-field lens crossovers and a fixed target.

Run on a complete vacuum instrument snapshot. The source and the independent
nominal-energy pencil share the actual electromagnetic field, mechanical stops
and exact integration mesh. Only original tip particles carry current.
"""
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from measure_probe_waist import PHASE, _array_hash, _json, _npz, analyze_scan, audit_extended_plan, observed_minimum
from solve_joint_probe import prepare_objective_observations
from temsim.cpu_resources import numerical_job
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot, decode_instrument
from temsim.lens_pose import lens_pose_registration
from temsim.optics.actual_probe_diagnostics import _hp_planes
from temsim.optics.crossover_aberrations import measure_geometric_cs, reference_pencil
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.optics.waist_evidence import finite_interval
from temsim.physics.beam_current import effective_source_current_a
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
from temsim.physics.ray_identity import emitted_source_identity
from temsim.physics.recording_clipping import clip_recording_planes
from temsim.physics.unified_field_particle import (
    MAX_UNIFIED_HISTORY_BYTES, build_unified_plan, execute_unified_plan,
    merge_unified_stops, unified_history_memory_bytes,
)
from temsim.physics.unified_field_runtime import get_unified_field_runtime, unified_runtime_preflight
from temsim.simulation_modes import is_unified
from temsim.specimen.source import specimen_is_vacuum


def component_plane(state, lens, offset_mm=0.):
    pose = lens_pose_registration(state, lens.key)
    return float(pose.origin_array_m[2]*1000 + pose.rotation_array[2, 2]*(lens.z_mm+offset_mm))


def geometric_threefold_terms(measurement):
    """Threefold intercept terms in the fitted outgoing mechanical-slope chart.

    In the complex intercept x+iy, A2 multiplies conjugate(u)**2. Two
    degree-four vector harmonics can also have threefold rotational symmetry.
    Return length coefficients before multiplication by the angular power.
    """
    if measurement.get("c3_mm") is None or measurement.get("status") != "MEASURED_AT_FIXED_STEP":
        return None
    fit = measurement["fit"]
    scale = fit["local_slope_radius"]
    vector = np.asarray(fit["scaled_vector_coefficients_m"])
    coefficients = dict(zip(map(tuple, fit["powers_xy"]), vector[:, 0] + 1j*vector[:, 1]))
    a2 = (coefficients[2, 0] - coefficients[0, 2] + 1j*coefficients[1, 1]) / (4*scale**2)
    phi = 2*np.pi*np.arange(32)/32
    fourth = sum(value*np.cos(phi)**i*np.sin(phi)**j for (i, j), value in coefficients.items()
                 if i+j == 4) / scale**4
    neg2 = np.mean(fourth*np.exp(2j*phi))
    pos4 = np.mean(fourth*np.exp(-4j*phi))
    return dict(a2_magnitude_um=float(abs(a2)*1e6),
                a2_xy_um=[a2.real*1e6, a2.imag*1e6],
                fourth_neg2_xy_mm=[neg2.real*1e3, neg2.imag*1e3],
                fourth_pos4_xy_mm=[pos4.real*1e3, pos4.imag*1e3])


def target_current_kernel(arrays, target, source_current, *, pitch_nm=.02, pixels=256):
    """Absolute current per image bin, at the commanded origin. Retain out-of-view flux."""
    index = np.flatnonzero(arrays["z_mm"] == target).item()
    weights = arrays["weights"]
    live = arrays["alive"][index] & (weights > 0)
    w = source_current*weights[live]/np.sum(weights)
    edges = (np.arange(pixels+1)-pixels/2)*pitch_nm
    y, x = (arrays[name][index, live]*1e9 for name in ("y_m", "x_m"))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite current-carrying positions in the target plane")
    grid = np.histogram2d(y, x, bins=(edges, edges), weights=w)[0]
    outside = (x < edges[0]) | (x > edges[-1]) | (y < edges[0]) | (y > edges[-1])
    return dict(z_mm=target, pitch_nm=pitch_nm, x_edges_nm=edges, y_edges_nm=edges,
                current_a_yx=grid, outside_current_a=float(np.sum(w[outside])),
                transmitted_current_a=float(np.sum(w)), emitted_current_a=source_current,
                surviving_source_ids=arrays["source_id"][live], xy_nm=np.column_stack((x, y)),
                particle_current_a=w)


def trace_checkpoints(state, plan, phase, energy, times, initial_alive, initial_stops, initial_keys):
    """Batch only particle columns. All batches share one immutable field plan."""
    z = plan.z_mm[plan.checkpoint_index]
    fixed = unified_history_memory_bytes(state, plan, 0, history_policy="checkpoints")
    unit = unified_history_memory_bytes(state, plan, 1, history_policy="checkpoints") - fixed
    maximum = (MAX_UNIFIED_HISTORY_BYTES-fixed)//unit
    if maximum < 1:
        raise ValueError("Requested checkpoint grid exceeds the physical history budget")
    history = {k: np.empty((len(z), len(phase))) for k in PHASE}
    live = np.empty((len(z), len(phase)), bool)
    blocked, keys = initial_stops.copy(), list(initial_keys)
    for first in range(0, len(phase), maximum):
        sl = slice(first, min(first+maximum, len(phase)))
        x, y, tx, ty = phase[sl].T
        raw = execute_unified_plan(state, plan, x, tx, y, ty,
            initial_time_s=times[sl], initial_kinetic_energy_ev=energy[sl], return_flight_times=True,
            defer_nonfinite_until_clipping=True, history_policy="checkpoints")
        cp = raw[-1]
        if not np.array_equal(cp.z_mm, z):
            raise ValueError("Exact physical checkpoint missing from a particle batch")
        alive, stop, names = merge_unified_stops(raw[0], initial_alive[sl], blocked[sl], keys[sl])
        _, stop, names = clip_recording_planes(state, raw[0], raw[1], raw[3], alive, stop, names)
        live[:, sl] = cp.unified_state.active & (~np.isfinite(stop) | (z[:,None] < stop))
        for k in PHASE:
            history[k][:, sl] = getattr(cp, k)
        blocked[sl], keys[sl] = stop, names
        del cp, raw
    return SimpleNamespace(z_mm=z, **history), live, blocked, keys


def survey(state, folder, *, radius_um, curvature_m1, step_mm, spacing_mm,
           target_mm=None, stop_mm=None, holdout_nm=.01, focus_spacing_nm=1.):
    if not is_unified(state) or not specimen_is_vacuum(state.sample):
        raise ValueError("An explicitly prepared unified vacuum incident-column snapshot is required")
    if state.electron_gun.emitter.ray_count < 1000:
        raise ValueError("At least 1000 original positive-current source particles are required")
    if not np.isfinite((step_mm, spacing_mm, holdout_nm)).all() or min(step_mm, spacing_mm, holdout_nm) <= 0:
        raise ValueError("Step, observation spacing and holdout budget must be positive")
    preflight = unified_runtime_preflight(state)
    if preflight["issues"]:
        raise ValueError(str(preflight["issues"]))
    runtime = get_unified_field_runtime(state)
    if isinstance(runtime.snapshot, DynamicPotentialSnapshot):
        raise ValueError("A static operating point is required for this geometric coefficient measurement")
    start = float(state.electron_gun.exit_plane_z_mm)
    surface = float(state.sample.upper_surface_z_mm)
    center, bottom = float(state.sample.z_mm), float(state.sample.lower_surface_z_mm)
    target = surface if target_mm is None else float(target_mm)
    if not np.isfinite(focus_spacing_nm) or not .05 <= focus_spacing_nm <= 10.:
        raise ValueError("Focus sampling must be between 0.05 and 10 nm")
    offsets = np.linspace(-100., 100., int(np.ceil(200./focus_spacing_nm))+1)*1e-6
    entrance_profile = np.unique(np.r_[surface, surface+offsets])
    target_profile = np.unique(np.r_[target, target+offsets])
    profile = np.unique(np.r_[entrance_profile, target_profile, center, bottom])
    stop = profile[-1] if stop_mm is None else float(stop_mm)
    if not start < profile[0] <= profile[-1] <= stop <= runtime.stop_z_m*1000:
        raise ValueError("Target and observation window must be inside the executed column interval")
    if (stop-start)/spacing_mm > 16000:
        raise ValueError("Requested observation grid exceeds 16000 intervals")
    pencils = [reference_pencil(radius_um*factor*1e-6, curvature_m1) for factor in (1., .25)]
    lens_planes = sorted((component_plane(state, lens), lens.key) for lens in state.lenses if lens.enabled)
    registry = dict(S_in=surface, S_top=surface, S_center=center, S_bottom=bottom, S_target=target)
    # The physical faces of the coupled ADL--TL12 assembly are bookkeeping
    # boundaries. No field support is truncated there.
    lenses = {lens.key: lens for lens in state.lenses}
    for name, lens, sign in (("E", lenses["adapter_lens"], -1), ("X", lenses["probe_tl12_lens"], 1)):
        center_offset = lens.mechanical_center_from_tip_mm-lens.optical_reference_from_tip_mm
        registry[name] = component_plane(state, lens, center_offset+sign*lens.mechanical_length_mm/2)
    hp_planes = _hp_planes(state)
    for row in hp_planes:
        registry[row["label"]] = row["z_mm"]
    registry["E_core"] = min(row["z_mm"] for row in hp_planes)
    bounded = [(z,k) for z,k in lens_planes if start < z < stop]
    nodes = np.unique(np.r_[np.linspace(start, stop, int(np.ceil((stop-start)/spacing_mm))+1),
                            [z for z,_ in bounded], list(registry.values()), profile])
    if nodes[0] != start or nodes[-1] != stop:
        raise ValueError("The selected interval must cover the complete probe corrector")
    plan = build_unified_plan(state, start, stop, checkpoint_z_mm=nodes, maximum_step_mm=step_mm)
    audit = audit_extended_plan(state, plan, stop)
    identity = capture_instrument_snapshot(state)
    gun = trace_source_to_exit(state)
    e = gun.exit_bundle
    ids, _ = emitted_source_identity(gun)
    ids, weights = np.asarray(ids), np.asarray(e.weight)
    count = len(ids)
    if (weights.shape != (count,) or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or len(np.unique(ids)) != count or not np.isfinite(weights).all()
            or (weights < 0).any() or np.count_nonzero(weights > 0) < 1000):
        raise ValueError("Invalid or insufficient original positive-current emission")
    pencil = np.vstack([p for p,_ in pencils])
    src = np.column_stack([e.x_m, e.y_m, e.tx_rad, e.ty_rad])
    phase = np.vstack((np.where(np.asarray(e.alive)[:, None], src, np.nan), pencil))
    energy = np.r_[state.electron_gun.nominal_exit_energy_ev + np.asarray(e.energy_offset_ev),
                   np.full(len(pencil), state.beam_voltage_kv*1000.)]
    times = np.r_[e.flight_time_s, np.zeros(len(pencil))]
    cp, live, blocked, keys = trace_checkpoints(state, plan, phase, energy, times,
        np.r_[e.alive, np.ones(len(pencil), bool)],
        np.r_[gun.blocked_z_mm, np.full(len(pencil), np.nan)], [*gun.blocked_key, *([""]*len(pencil))])
    if not np.array_equal(nodes, cp.z_mm) or capture_instrument_snapshot(state).digest != identity.digest:
        raise ValueError("Executed observation grid or input identity changed")
    current = float(effective_source_current_a(state))
    arrays = {k: getattr(cp, k)[:, :count] for k in PHASE}
    arrays.update(z_mm=nodes, alive=live[:, :count], source_id=ids, weights=weights,
                  blocked_z_mm=blocked[:count], blocked_key=np.asarray(keys[:count],str))
    # Keep the legacy entrance waist evidence local even for a target far away.
    measured = analyze_scan(arrays, entrance_profile, surface_z_mm=surface, source_current_a=current)
    def at(z):
        i = int(np.flatnonzero(nodes == z).item())
        fits = []
        offset = count
        for launch, n_train in pencils:
            where = slice(offset, offset+len(launch))
            rays = np.column_stack([getattr(cp,k)[i, where] for k in ("x_m","y_m","tx_rad","ty_rad")])
            rays[~live[i, where]] = np.nan
            fit = measure_geometric_cs(launch[:,:2], rays, n_train, holdout_budget_m=holdout_nm*1e-9)
            fit["geometric_threefold"] = geometric_threefold_terms(fit)
            fits.append(fit)
            offset += len(launch)
        return dict(z_mm=z, pencils=fits, c3_mm=fits[0]["c3_mm"], smaller_pencil_c3_mm=fits[1]["c3_mm"])
    regions = []
    bounds = [(start, "gun exit"), *bounded, (stop, "end")]
    for (lo, lens), (hi, _) in zip(bounds, bounds[1:]):
        obs = finite_interval(SimpleNamespace(**arrays), arrays["alive"], weights, lo, hi,
                              maximum_spacing_mm=spacing_mm, source_ids=ids)
        for row in obs["regions"]:
            row["geometric_aberrations"] = at(row["z_mm"])
        regions.append(dict(upstream_reference=lens, **obs))
    for z, key in lens_planes:
        if z >= stop:
            regions.append(dict(upstream_reference=key, status="OUTSIDE_OBSERVED_INTERVAL", regions=[]))
    waist_planes = [waist["z_mm"] for interval in regions for waist in interval["regions"]]
    observation_planes = np.unique(np.r_[profile, waist_planes])
    measured_planes = analyze_scan(arrays, observation_planes, surface_z_mm=surface, source_current_a=current)["planes"]
    by_z = {row["z_mm"]: row for row in measured_planes}
    for row in measured_planes:
        i = int(np.searchsorted(nodes,row["z_mm"]))
        disk = np.hypot(arrays["x_m"][i], arrays["y_m"][i]) <= .05e-9
        row["axis_core_50pm_current_a"] = float(current*weights[arrays["alive"][i]&disk].sum()/weights.sum())
    for interval in regions:
        for waist in interval["regions"]:
            waist["probe_statistics"] = {key: value for key, value in by_z[waist["z_mm"]].items()
                                        if key != "source_ids"}
    all_rows = [by_z[z] for z in profile]
    target_metrics = next(row for row in all_rows if row["z_mm"] == target)
    target_coordinates = set(target_profile)
    local = [row for row in all_rows if row["z_mm"] in target_coordinates]
    local_z = np.array([row["z_mm"] for row in local])
    focus = {key: observed_minimum(local_z, [row[key] for row in local], target)
             for key in ("d50_nm", "d95_nm", "rms_radius_nm")}
    peak = max(row["axis_core_50pm_current_a"] for row in local)
    focus["axis_core_50pm"] = dict(best_current_a=peak, target_current_a=target_metrics["axis_core_50pm_current_a"],
        observed_maximum_planes_mm=[row["z_mm"] for row in local if row["axis_core_50pm_current_a"] == peak],
        center="fixed commanded laboratory axis", disk_area_nm2=float(np.pi*.05**2))
    report = dict(schema="unified-crossover-geometric-cs-v1", qualification="NOT_QUALIFIED",
        input_digest=identity.digest, planes_mm=registry, physical_target=target_metrics,
        target_equals_entrance=target==surface, field_audit=audit, actual_source_count=count,
        original_positive_current_count=int(np.count_nonzero(weights>0)), original_current_a=current,
        physical_source_ids_sha256=_array_hash(ids), source_weights_sha256=_array_hash(weights),
        entrance_profile=measured, plane_metrics=all_rows, through_focus=focus,
        lens_intervals=regions, target_aberrations=at(target),
        gun_interior="not observed upstream of the executed gun exit",
        pupil=dict(launch_z_mm=start, radii_um=[radius_um, radius_um*.25], curvature_m1=curvature_m1,
                   nominal_energy_ev=float(state.beam_voltage_kv*1000.), holdout_budget_nm=holdout_nm),
        declared_maximum_step_mm=step_mm, actual_maximum_step_mm=float(np.max(np.diff(plan.z_mm))),
        maximum_observation_spacing_mm=spacing_mm, scope="Vacuum geometric probe and fixed axial virtual source pencil")
    _json(folder/"inputs.json", identity.to_dict())
    _npz(folder/"target-current.npz", target_current_kernel(arrays, target, current))
    _npz(folder/"rays.npz", dict(arrays, integration_z_mm=plan.z_mm, launch_pencil=pencil,
        pencil_xy_slopes=np.stack([getattr(cp,k)[:,count:] for k in ("x_m","y_m","tx_rad","ty_rad")],axis=-1),
        pencil_alive=live[:,count:], pencil_blocked_z_mm=blocked[count:], pencil_blocked_key=np.array(keys[count:],str),
        gun_exit_xy_slopes=src, gun_exit_kinetic_energy_ev=energy[:count], gun_exit_flight_time_s=times[:count]))
    _json(folder/"report.json", report)
    return report


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("snapshot", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--pupil-radius-um", required=True, type=float)
    p.add_argument("--curvature-m1", required=True, type=float)
    p.add_argument("--step-mm", default=.05, type=float)
    p.add_argument("--spacing-mm", default=.5, type=float)
    p.add_argument("--focus-spacing-nm", default=1., type=float)
    p.add_argument("--target-mm", type=float)
    p.add_argument("--stop-mm", type=float, help="Optional downstream vacuum end including projector lenses")
    p.add_argument("--holdout-nm", default=.01, type=float)
    p.add_argument("--hp-pair-scale", default=1., type=float, help="Scale only HP1 and HP2; zero selects H0")
    p.add_argument("--hp1-relative-scale", default=1., type=float)
    p.add_argument("--hp1-angle-delta-rad", default=0., type=float)
    p.add_argument("--objective-delta-percent", default=0., type=float, help="Add percentage points")
    p.add_argument("--source-count", type=int, help="Independent original-tip count refinement")
    p.add_argument("--threads", default=4, type=int)
    args = p.parse_args()
    old = InstrumentSnapshot.from_dict(json.loads(args.snapshot.read_text(encoding="utf-8")))
    state = decode_instrument(old.graph)
    old._verify_external_inputs(state)
    new = capture_instrument_snapshot(state)
    if old.graph != new.graph or old.external_inputs != new.external_inputs:
        raise ValueError("Input import changed the original instrument or external sources")
    def field_inventory():
        return {row.source_id: row.potential.digest for row in get_unified_field_runtime(state).snapshot.contributions}
    original_fields = field_inventory()
    allowed = set()
    for hp, factor in ((state.hp1_hexapole, args.hp_pair_scale*args.hp1_relative_scale),
                       (state.hp2_hexapole, args.hp_pair_scale)):
        value = hp.strength_m3*factor
        if not np.isfinite(value) or abs(value)>hp.maximum_strength_m3:
            p.error("Main hexapole drive exceeds its physical control range")
        if value != hp.strength_m3:
            allowed.add(hp.key)
            hp.strength_m3 = value
    if args.hp1_angle_delta_rad:
        if not np.isfinite(args.hp1_angle_delta_rad):
            p.error("HP1 rotation must be finite")
        state.hp1_hexapole.orientation_rad += args.hp1_angle_delta_rad
        allowed.add(state.hp1_hexapole.key)
    if args.objective_delta_percent:
        lens = state.objective_lens
        value = lens.percent + args.objective_delta_percent
        if not 0 <= value <= lens.max_percent:
            p.error("Objective drive exceeds its physical control range")
        lens.percent = value
        prepare_objective_observations(state)
        allowed.add(lens.key)
    if args.source_count is not None:
        if args.source_count < 1000:
            p.error("A physical trial requires at least 1000 original positive-current particles")
        state.electron_gun.emitter.ray_count = args.source_count
        state.electron_gun.validate()
    fields = field_inventory()
    changed = {k for k in fields.keys()|original_fields.keys() if fields.get(k) != original_fields.get(k)}
    if changed != allowed:
        raise ValueError(f"Physical field fingerprint mismatch. Expected {sorted(allowed)}, observed {sorted(changed)}")
    args.output.mkdir(parents=True, exist_ok=False)
    _json(args.output/"provenance.json", dict(original_digest=old.digest, imported_digest=new.digest,
        original_file=str(args.snapshot.resolve()), changed_field_ids=sorted(changed),
        original_fields=original_fields, applied_fields=fields,
        control_changes=dict(hp_pair_scale=args.hp_pair_scale, hp1_relative_scale=args.hp1_relative_scale,
            hp1_angle_delta_rad=args.hp1_angle_delta_rad, objective_delta_percent=args.objective_delta_percent,
            explicit_source_count=args.source_count)))
    with numerical_job(args.threads):
        survey(state,args.output,radius_um=args.pupil_radius_um,curvature_m1=args.curvature_m1,
               step_mm=args.step_mm,spacing_mm=args.spacing_mm,target_mm=args.target_mm,
               stop_mm=args.stop_mm,holdout_nm=args.holdout_nm,focus_spacing_nm=args.focus_spacing_nm)


if __name__ == "__main__":
    main()
