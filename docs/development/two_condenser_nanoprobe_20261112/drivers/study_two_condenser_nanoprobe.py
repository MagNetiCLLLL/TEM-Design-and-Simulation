"""Measure true C1+C2 Nanoprobe focus with original finite-source particles.

This classical vacuum diagnostic retains the actual specimen upper-surface
reference, finite fields, physical stops, source identities and absolute weights.
It never substitutes a C3-off column or applies a missing probe corrector.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from measure_probe_waist import PHASE, _array_hash, _json, _npz, analyze_scan, audit_extended_plan
from temsim.assembly_catalog import AssemblyCatalog
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.particle_sections import _events, _trace_segment, gun_dependency_signature
from temsim.simulation_modes import switch_mode


def installed_lenses(state):
    return {item.key: item for item in state.lenses}


def prepare_state(count, step, *, raw_column_defaults=False):
    if count < 1000 or not math.isfinite(step) or step <= 0:
        raise ValueError("Use at least 1000 original positive-current particles and a positive step")
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), column="C2", recording="No Energy Filter")
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    from temsim.operating_modes import apply_operating_mode_pair, apply_projector_mode
    if raw_column_defaults:
        apply_projector_mode(state, "diffraction", column_name=selection.column,
                             recording_name=selection.recording)
    else:
        apply_operating_mode_pair(state, "nano_probe", "diffraction", column_name=selection.column,
                                  recording_name=selection.recording)
    state.illumination_mode = "STEM"
    lenses = installed_lenses(state)
    lenses["mini_condenser"].polarity = -1
    state.electron_gun.emitter.ray_count = count
    state.step_mm = step
    state.history_step_mm = 2.
    state.sample.inserted = False
    for flag in ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled", "stem_image_enabled"):
        setattr(state.sample, flag, False)
    for item in (*state.deflectors, *state.stigmators, *state.corrector_elements):
        for flag in ("scan_enabled", "wobble_enabled"):
            if hasattr(item, flag):
                setattr(item, flag, False)
    if lenses["condenser_lens_3"].enabled or lenses["adapter_lens"].enabled:
        raise ValueError("The selected C2 assembly must not install C3 or a probe corrector")
    physical = {item.key for item in state._resolved_assembly.parts}
    if (not {"condenser_lens_1", "condenser_lens_2"} <= physical
            or "condenser_lens_3" in physical or "adapter_lens" in physical
            or any(key.startswith("probe_hp") or key.startswith("probe_tl") for key in physical)):
        raise ValueError("Resolved physical hardware is not the independent C1+C2 assembly")
    return state, selection


def set_controls(state, controls):
    allowed = {"condenser_lens_1", "condenser_lens_2", "mini_condenser", "objective_lens"}
    lenses = installed_lenses(state)
    for key, value in controls.items():
        if key not in allowed or not math.isfinite(float(value)) or not 0 < value < lenses[key].max_percent:
            raise ValueError(f"Invalid installed two-condenser drive: {key}")
        lenses[key].percent = float(value)
    state.sync_objective()


def observation_planes(state, window_nm, spacing_nm, topology):
    surface = float(state.sample.upper_surface_z_mm)
    if window_nm <= 0 or spacing_nm <= 0:
        raise ValueError("Focus window and spacing must be positive")
    count = math.ceil(2*window_nm/spacing_nm)
    if count > 10001:
        raise ValueError("Focus observation budget exceeded")
    focus = np.unique(np.r_[surface+np.linspace(-window_nm, window_nm, count+1)*1e-6,
                            surface, state.sample.z_mm, state.sample.lower_surface_z_mm])
    exact = [float(a.z_mm) for a in state.apertures if a.enabled]
    exact.extend(float(a.z_mm) for a in state.lenses if a.enabled)
    intervals = []
    if topology:
        lenses = installed_lenses(state)
        chain = [("gun exit", float(state.electron_gun.exit_plane_z_mm)),
                 ("C1", float(lenses["condenser_lens_1"].z_mm)),
                 ("C2", float(lenses["condenser_lens_2"].z_mm)),
                 ("Mini", float(lenses["mini_condenser"].z_mm)), ("surface", surface)]
        for (name, lo), (other, hi) in zip(chain[:-1], chain[1:]):
            spacing = .2 if name == "Mini" else .5 if name == "C1" else 1.
            intervals.append((name+" to "+other, lo, hi, spacing))
            exact.extend(np.linspace(lo, hi, math.ceil((hi-lo)/spacing)+1))
        from temsim.physics.potential_capture import capture_round_lens_potential
        starts = [float(capture_round_lens_potential(state, key).potential.field.native_support_mm[0])
                  for key in ("mini_condenser", "objective_lens")]
        lower, upper = min(starts), float(focus[-1])
        intervals.append(("Final combined field through focus", lower, upper, .2))
        exact.extend(np.linspace(lower, upper, math.ceil((upper-lower)/.2)+1))
    start = float(state.electron_gun.exit_plane_z_mm)
    exact = [z for z in exact if start <= z <= focus[-1]]
    return focus, intervals, np.unique(np.r_[focus, exact])


def execute(state, planes, cache):
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.beam_current import effective_source_current_a
    from temsim.physics.ray_identity import emitted_source_identity
    from temsim.physics.simulation import _validate_gun_flight_times
    from temsim.physics.unified_field_runtime import unified_runtime_preflight
    from temsim.specimen.source import specimen_is_vacuum

    if state.simulation_mode != "unified" or not specimen_is_vacuum(state.sample):
        raise ValueError("Only classical unified-field vacuum observations are supported")
    signature = gun_dependency_signature(state)
    original = capture_instrument_snapshot(state)
    if unified_runtime_preflight(state)["issues"]:
        raise ValueError("The complete finite-field runtime did not pass its physical preflight")
    reused = signature in cache
    gun = cache[signature] if reused else trace_source_to_exit(state)
    if not signature or not _validate_gun_flight_times(gun):
        raise ValueError("A current physical source identity and actual flight clocks are required")
    bundle = gun.exit_bundle
    ids, azimuth = emitted_source_identity(gun)
    weights = np.asarray(bundle.weight)
    if (not np.isfinite(weights).all() or np.any(weights < 0) or not weights.sum() > 0
            or np.count_nonzero(weights > 0) < 1000
            or np.unique(ids[weights > 0]).size != np.count_nonzero(weights > 0)):
        raise ValueError("At least 1000 distinct original positive-current particles are required")
    initial = tuple(np.asarray(getattr(bundle, key)) for key in
                    ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    offsets = np.asarray(bundle.energy_offset_ev)
    survival = (np.asarray(bundle.alive), np.asarray(gun.blocked_z_mm), list(gun.blocked_key))
    start, stop = float(state.electron_gun.exit_plane_z_mm), float(planes[-1])
    segment, resume, prefix_reused = _trace_segment(state, "incident", start, stop, initial,
        offsets, weights, (ids, azimuth), survival, _events(state), None, math.inf,
        initial_kicks=True, save_z_mm=tuple(planes),
        initial_kinetic_energy_ev=float(state.electron_gun.nominal_exit_energy_ev)+offsets)
    cp, branch = segment.checkpoints, segment.branch
    if (capture_instrument_snapshot(state).digest != original.digest
            or not np.array_equal(branch.source_ray_id, ids)
            or not np.array_equal(branch.ray_weight, weights)):
        raise ValueError("Transport changed original physical inputs, particle identities or weights")
    arrays = {key: np.asarray(getattr(cp, key)) for key in PHASE}
    arrays.update(z_mm=np.asarray(cp.z_mm), integration_z_mm=np.asarray(segment.plan.z_mm),
        alive=np.asarray(cp.unified_state.active), weights=weights, source_id=ids,
        source_azimuth_rad=azimuth, gun_exit_alive=survival[0],
        blocked_z_mm=np.asarray(branch.blocked_z), blocked_key=np.asarray(branch.blocked_key, dtype=str),
        terminal_alive=np.asarray(branch.alive),
        **{"gun_exit_"+key: value for key, value in zip(
            ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"), initial)},
        **{"emission_"+key: np.asarray(value) for key, value in gun.emission_reference.items()})
    cache[signature] = gun
    info = dict(input_digest=original.digest, implementation=original.implementation,
        gun_dependency_signature=signature, gun_reused=reused,
        reference_current_a=float(effective_source_current_a(state)),
        emitted_count=len(weights), emitted_positive_count=int(np.count_nonzero(weights > 0)),
        gun_exit_positive_count=int(np.count_nonzero(survival[0] & (weights > 0))),
        initial_dependency_signature=segment.initial_dependency_signature,
        original_source_sha256={key: _array_hash(value) for key, value in arrays.items()
            if key in ("weights", "source_id", "source_azimuth_rad") or key.startswith(("emission_", "gun_exit_"))},
        active_lens_owners=[item.key for item in state.lenses if item.enabled],
        column_hardware=state.layout_c3_hardware, column_mode=state.column_mode,
        recording_receivers={item.key: dict(inserted=bool(item.inserted))
            for item in (state.fluorescent_screen, state.camera)},
        integration_nodes=len(segment.plan.z_mm), checkpoint_rows=len(cp.z_mm),
        field_audit=audit_extended_plan(state, segment.plan, stop))
    return arrays, info, original


def summarize(state, arrays, info, focus, intervals):
    surface = float(state.sample.upper_surface_z_mm)
    row = int(np.searchsorted(arrays["z_mm"], surface))
    weights = arrays["weights"]
    keep = arrays["alive"][row] & (weights > 0)
    current = info["reference_current_a"]
    result = dict(surface_z_mm=surface, surviving_positive_count=int(keep.sum()),
        effective_count=float(weights[keep].sum()**2/np.sum(weights[keep]**2)) if keep.any() else 0.,
        sample_current_a=float(current*weights[keep].sum()),
        emitted_current_transmission=float(weights[keep].sum()/weights.sum()))
    if keep.any():
        stats = transverse_beam_statistics(*(arrays[key][row] for key in PHASE[:4]), alive=keep, weights=weights)
        result.update(d95_nm=2e9*stats.radius_95_m, rms_radius_nm=1e9*stats.radius_rms_m,
            alpha95_mrad=1e3*stats.convergence_95_rad, angular_rms_mrad=1e3*stats.convergence_rms_rad,
            rms_tangent_waist_offset_nm=1e9*stats.waist_offset_m)
        focus_rows = np.searchsorted(arrays["z_mm"], focus)
        if np.all(np.any(arrays["alive"][focus_rows] & (weights > 0), axis=1)):
            result["native_focus"] = analyze_scan(arrays, focus, surface_z_mm=surface, source_current_a=current)
            for plane in result["native_focus"]["planes"]:
                plane.pop("source_ids", None)
        else:
            result["native_focus"] = dict(status="NO_POSITIVE_CURRENT_SUPPORT_ON_FULL_FOCUS_WINDOW")
    keys = arrays["blocked_key"]
    result["physical_losses"] = {str(key): dict(count=int(np.count_nonzero((keys == key) & (weights > 0))),
        current_a=float(current*weights[keys == key].sum())) for key in sorted(set(keys)) if key}
    terminal = float(current*weights[arrays["terminal_alive"]].sum())
    residual = float(current*weights.sum()-terminal-sum(v["current_a"] for v in result["physical_losses"].values()))
    if abs(residual) > 1e-12*current:
        raise ValueError("Absolute original current balance failed")
    result["absolute_current_balance_residual_a"] = residual
    if intervals:
        from types import SimpleNamespace
        from temsim.optics.waist_evidence import finite_interval, observed_aperture_losses
        cp = SimpleNamespace(**{key: arrays[key] for key in ("z_mm", *PHASE[:4])})
        losses = observed_aperture_losses(state, cp, arrays["alive"], weights, arrays["source_id"])
        result["topology"] = {}
        for label, lo, hi, spacing in intervals:
            result["topology"][label] = {}
            for cohort, masks in (("actual", arrays["alive"]), ("sample_survivors", arrays["alive"] & keep)):
                result["topology"][label][cohort] = finite_interval(cp, masks, weights, lo, hi,
                    maximum_spacing_mm=spacing+1e-8, source_ids=arrays["source_id"], known_interceptions=losses)
        last = result["topology"]["Final combined field through focus"]["sample_survivors"]
        bounded = last["status"] == "RESOLVED_AT_BOUNDED_SAMPLING"
        regions = last["regions"]
        final_waist_near_surface = len(regions) == 1 and (
            float(focus[0]) <= regions[0]["representative_z_mm"] <= float(focus[-1]))
        placed = bool(result.get("native_focus", {}).get("surface_gate", {}).get("passed"))
        passed = bounded and final_waist_near_surface and placed
        result["direct_surface_focus"] = dict(passed=passed,
            status="BOUNDED_DIRECT_SURFACE_FOCUS" if passed else "UNRESOLVED_OR_NOT_AT_SURFACE",
            native_d95_surface_gate=placed, final_field_topology_status=last["status"],
            observed_final_field_2d_waists=len(regions),
            final_2d_waist_inside_surface_observation_window=bool(final_waist_near_surface),
            scope="Bounded native D95 placement and one final 2D waist across complete Mini/Objective support")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--reference-count", type=int,
                        help="Repeat the same installed controls with another complete original population")
    parser.add_argument("--step", type=float, default=.1)
    parser.add_argument("--controls", type=Path, help="JSON list of independent installed-drive dictionaries")
    parser.add_argument("--window-nm", type=float, default=10000.)
    parser.add_argument("--spacing-nm", type=float, default=500.)
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--raw-column-defaults", action="store_true",
                        help="Observe uncalibrated module defaults instead of applying the C2 operating pair")
    args = parser.parse_args()
    counts = [args.count] + ([] if args.reference_count is None else [args.reference_count])
    if any(count < 1000 for count in counts):
        parser.error("Every trial must use at least 1000 original positive-current source particles")
    args.output.mkdir(parents=True, exist_ok=False)
    observer = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__).resolve(), ROOT/"scripts/measure_probe_waist.py")}
    candidates = json.loads(args.controls.read_text()) if args.controls else [{}]
    trials = [(count, controls) for count in counts for controls in candidates]
    from temsim.cpu_resources import numerical_job
    from temsim.profile_io import save_profile
    cache, summary = {}, []
    with numerical_job(args.threads):
        for index, (count, controls) in enumerate(trials):
            out = args.output/f"trial-{index:02d}"
            out.mkdir()
            started = perf_counter()
            state, selection = prepare_state(count, args.step, raw_column_defaults=args.raw_column_defaults)
            set_controls(state, controls)
            focus, intervals, planes = observation_planes(state, args.window_nm, args.spacing_nm, args.topology)
            _json(out/"request.json", dict(controls=controls, count=count, step_mm=args.step,
                requested_planes_mm=planes.tolist(), focus_planes_mm=focus.tolist(), topology_intervals=intervals))
            print(f"Trial {index}: original source {count}; controls {controls}", flush=True)
            arrays, info, snapshot = execute(state, planes, cache)
            _json(out/"inputs.json", snapshot.to_dict())
            save_profile(out/"input.toml", state, selection)
            _npz(out/"observations.npz", arrays)
            try:
                result = summarize(state, arrays, info, focus, intervals)
            except Exception as error:
                _json(out/"report.json", dict(status="ANALYSIS_FAILED_RAW_OBSERVATIONS_RETAINED",
                    execution=info, error=str(error)))
                raise
            result.update(status="COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION", execution=info,
                observer_sha256=observer,
                controls={key: float(installed_lenses(state)[key].percent) for key in
                    ("condenser_lens_1", "condenser_lens_2", "mini_condenser", "objective_lens")},
                observations_sha256=sha256((out/"observations.npz").read_bytes()).hexdigest(),
                elapsed_s=perf_counter()-started,
                scope="True C1+C2; vacuum incident probe; original absolute weights; no coherent or projector calibration")
            _json(out/"report.json", result)
            summary.append({key: result[key] for key in ("controls", "d95_nm", "rms_radius_nm", "alpha95_mrad",
                "angular_rms_mrad", "surviving_positive_count", "sample_current_a", "elapsed_s") if key in result})
            if "d95_waist" in result.get("native_focus", {}):
                summary[-1]["d95_waist"] = result["native_focus"]["d95_waist"]
                summary[-1]["rms_waist"] = result["native_focus"]["rms_waist_diagnostic"]
            _json(args.output/"summary.json", summary)
            print(json.dumps(summary[-1]), flush=True)


if __name__ == "__main__":
    main()
