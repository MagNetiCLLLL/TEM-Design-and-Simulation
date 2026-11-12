"""Record an actual FEG/full-column current baseline and optional hybrid trial.
Tip-origin coherent execution is retired; use python -m temsim.mixed_wave_cli --help.

Explicit optical proposals are recorded and require fresh validation. There
is no source replacement, current renormalisation, or aperture withdrawal.
An explicit coherence length selects a labelled tip boundary for both paths.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import traceback

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.profile_io import read_profile, apply_profile_values, save_profile


def write_report(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def population(weight, alive, current_a):
    selected = np.asarray(weight)[np.asarray(alive, bool)]
    total = float(selected.sum())
    return {"surviving_positive_count": int(np.count_nonzero(selected > 0)),
        "effective_count": total**2/float(selected@selected) if total else 0.,
        "weight_per_emitted_electron": total, "current_a": current_a*total}


def phase_space_summary(x, y, tx, ty, weight, alive):
    valid = np.asarray(alive, bool) & (np.asarray(weight) > 0)
    if not np.any(valid):
        return {"status": "NO_TRANSMITTED_CURRENT"}
    x, y, tx, ty, w = (np.asarray(v)[valid] for v in (x, y, tx, ty, weight))
    centre = np.array((np.dot(w, x), np.dot(w, y)))/w.sum()
    radius = np.hypot(x-centre[0], y-centre[1])
    angle = np.arctan(np.hypot(tx, ty))
    def quantile(values):
        order = np.argsort(values)
        return float(values[order[np.searchsorted(np.cumsum(w[order]), .95*w.sum())]])
    return {"status": "OBSERVED_WITHOUT_REFOCUS", "centroid_m": centre.tolist(),
        "d95_m": 2*quantile(radius), "radial_rms_m": float(np.sqrt(np.dot(w, radius**2)/w.sum())),
        "alpha95_rad": quantile(angle),
        "centroid_core50pm_original_weight": float(w[radius <= 50e-12].sum()),
        "centroid_core500pm_original_weight": float(w[radius <= 500e-12].sum())}


def topology_intervals(state):
    parts = {item.key: item for item in state.lenses}
    chain = [(key, float(parts[key].z_mm)) for key in ("condenser_lens_2", "condenser_lens_3",
        "adapter_lens", "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens")]
    chain += [("ac_upper", state.ac_deflector.upper_z_mm), ("ac_lower", state.ac_deflector.lower_z_mm),
        ("mini_condenser", parts["mini_condenser"].z_mm),
        ("specimen_entrance", state.sample.z_mm-state.sample.thickness_nm*.5e-6)]
    return [(a+" to "+b, za, zb, .05 if a == "ac_upper" else .5)
        for (a, za), (b, zb) in zip(chain[:-1], chain[1:])]


def prepare(args):
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    if args.profile:
        selection, values = read_profile(args.profile)
        catalog.apply(state, catalog.normalise_selection(selection))
        apply_profile_values(state, values)
    state.simulation_mode = "unified"
    state.acceleration_backend = "Auto"
    state.step_mm = args.step_mm
    state.electron_gun.emitter.ray_count = args.source_particles
    changes = []
    if args.optical_control_seed:
        # Historical results are not accepted. Copy scalar drive inputs only
        # into today's installed hardware, then perform a fresh calculation.
        record = json.loads(args.optical_control_seed.read_text(encoding="utf-8"))
        from temsim.instrument_snapshot import InstrumentSnapshot
        InstrumentSnapshot.from_dict(record)
        parts = {item.key: item for item in (*state.lenses, *state.corrector_elements, *state.stigmators)}
        for node in record["graph"]["nodes"]:
            values = node.get("attributes", {})
            item = parts.get(values.get("key"))
            if item is None or not item.enabled or not values.get("enabled", False):
                continue
            for name in ("percent", "strength_m3", "strength_x_percent", "strength_y_percent"):
                if name in values and hasattr(item, name):
                    value = values[name]
                    if type(value) not in (int, float) or not np.isfinite(value):
                        raise ValueError("Optical seed contains an invalid scalar drive")
                    before = getattr(item, name)
                    if before != value:
                        changes.append({"key": item.key, "control": name,
                            "before": before, "after": value, "origin": "historical scalar proposal; requires fresh verification"})
                        setattr(item, name, value)
    if args.static_acquisition:
        # A separately declared stationary acquisition, not a frozen sample
        # of an active waveform. Physical coils and their DC settings remain.
        for item in (*state.deflectors, *state.stigmators, *state.corrector_elements):
            for name in ("scan_enabled", "wobble_enabled"):
                if bool(getattr(item, name, False)):
                    changes.append({"key": item.key, "control": name,
                                    "before": True, "after": False})
                    setattr(item, name, False)
    if args.optical_controls:
        proposal = json.loads(args.optical_controls.read_text(encoding="utf-8"))
        parts = {item.key: item for item in (*state.lenses, *state.corrector_elements, *state.stigmators)}
        for key, controls in proposal["controls"].items():
            item = parts[key]
            for name, value in controls.items():
                if name not in {"percent", "strength_m3", "skew_strength_m3", "strength_m2", "strength_x_percent", "strength_y_percent"}:
                    raise ValueError("Only explicit physical scalar drives may be proposed")
                if type(value) not in (float, int) or not np.isfinite(value):
                    raise ValueError("A finite scalar optical drive is required")
                if name == "percent" and not 0 <= value <= item.max_percent:
                    raise ValueError("Optical excitation exceeds the installed supply limit")
                before = getattr(item, name)
                setattr(item, name, value)
                changes.append({"key": key, "control": name, "before": before,
                    "after": value, "origin": str(args.optical_controls.resolve())})
            if hasattr(item, "validate"):
                item.validate()
    if args.coherence_length_nm is not None:
        from temsim.physics.coherent_inputs import TipEmissionSettings, candidate_tip_emission
        state.electron_gun = candidate_tip_emission(state, TipEmissionSettings(enabled=True,
            boundary_model="driven_gaussian_schell",
            transverse_coherence_length_nm=args.coherence_length_nm))
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.detector.recording_system import ensure_recording_system
    apply_physical_layout_to_state(state)
    ensure_recording_system(state)
    if args.camera_acquisition:
        for plane in state.recording_planes:
            inserted = plane.key == args.detector
            if plane.inserted != inserted:
                changes.append({"key": plane.key, "control": "inserted",
                    "before": plane.inserted, "after": inserted})
                plane.inserted = inserted
    if args.silicon_110:
        from temsim.specimen.cif_io import read_cif_atoms
        from temsim.specimen.geometry import quaternion_from_zone_axes, set_sample_orientation
        state.sample.cif_path = str((Path(__file__).resolve().parents[1]/"tests/fixtures/cif/Si.cif").resolve())
        atoms = read_cif_atoms(state.sample.cif_path)
        q = quaternion_from_zone_axes(atoms.cell, [1, 1, 0], [-1, 1, 0])
        set_sample_orientation(state.sample, q)
        state.sample.specimen_mode = "atomic"
        state.sample.inserted = True
        state.sample.thickness_nm = args.thickness_nm
        state.sample.wave_slice_thickness_angstrom = 2.
        state.sample.wave_grid_pixels = args.sample_pixels
        state.sample.wave_field_of_view_angstrom = args.sample_width_nm*10.
        state.sample.wave_frozen_phonon_enabled = False
    # Source tracing binds these derived environment inputs. Capture them
    # before the immutable field snapshot so execution does not add inputs.
    from temsim.vacuum import bind_gun_environment
    bind_gun_environment(state)
    # Resolve sample-dependent objective reference planes through the same
    # field context used by gun tracing before publishing immutable inputs.
    # This neither retunes a lens nor changes the physical sample/field.
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    with instrument_gun_field_context(state):
        pass
    return state, selection, changes


def classical(state, args, report):
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.physics.hybrid_pipeline import _source_count
    from temsim.physics.unified_field_particle import build_unified_plan, execute_unified_plan, merge_unified_stops
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.transport_progress import transport_progress

    current = state.electron_gun.emitted_current_a
    if not np.isfinite(current) or current <= 0:
        raise ValueError("Physical trials require a finite positive source current")
    detector = next(p for p in state.recording_planes if p.key == args.detector and p.inserted)
    runtime = capture_unified_field_runtime(state, stop_z_m=detector.z_mm*.001)
    write_report(args.output/"instrument.json", capture_instrument_snapshot(state).to_dict())
    report["runtime_digest"] = runtime.digest
    report["field_digest"] = runtime.snapshot.numerical_digest
    report["geometry_digest"] = runtime.geometry.digest
    report["field_providers"] = [{"source_id": row.source_id, "kind": row.kind,
        "owner_ids": list(row.owner_ids)} for row in runtime.snapshot.contributions]
    report["geometry_piece_count"] = len(runtime.geometry.pieces)
    report["stage"] = "CLASSICAL_GUN"
    write_report(args.output/"report.json", report)
    print("Tracing actual tip emission through the physical FEG", flush=True)
    gun = trace_source_to_exit(state, count=args.source_particles)
    report["source_counts"] = _source_count(gun)
    bundle = gun.exit_bundle
    weight, initial_alive = np.asarray(bundle.weight), np.asarray(bundle.alive)
    source_arrays = {key: np.asarray(value) for key, value in gun.emission_reference.items()}
    source_arrays.update({"exit_"+key: np.asarray(getattr(bundle, key)) for key in
        ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "flight_time_s", "weight", "ray_id", "alive")})
    source_arrays.update(blocked_z_mm=gun.blocked_z_mm, blocked_key=np.asarray(gun.blocked_key, str))
    np.savez_compressed(args.output/"original_source.npz", **source_arrays)
    report["reference_current_a"] = gun.emitted_current_a
    report["emitted"] = population(weight, np.ones_like(initial_alive), gun.emitted_current_a)
    report["gun_exit"] = population(weight, initial_alive, gun.emitted_current_a)
    report["gun_losses"] = {key: population(weight, np.asarray(gun.blocked_key) == key,
        gun.emitted_current_a) for key in sorted(set(gun.blocked_key)) if key}
    report["stage"] = "CLASSICAL_COLUMN"
    write_report(args.output/"report.json", report)
    try:
        runtime.require_state(state)
    except ValueError:
        write_report(args.output/"changed_inputs.json", capture_instrument_snapshot(state).to_dict())
        raise
    entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    offsets = np.arange(-args.focus_scan_half_width_nm, args.focus_scan_half_width_nm+args.focus_scan_step_nm*.5,
        args.focus_scan_step_nm) if args.focus_scan_half_width_nm else np.array([0.])
    focus_planes = entrance+offsets*1e-6
    intervals = topology_intervals(state) if args.topology else ()
    topology_planes = np.unique(np.concatenate([np.linspace(lo, hi, int(np.ceil((hi-lo)/step))+1)
        for _, lo, hi, step in intervals])) if intervals else ()
    observations = (*topology_planes, *focus_planes, entrance, *(p.z_mm for p in state.recording_planes
        if state.electron_gun.exit_plane_z_mm < p.z_mm <= detector.z_mm))
    plan = build_unified_plan(state, state.electron_gun.exit_plane_z_mm, detector.z_mm,
        save_z_mm=observations, _captured_runtime=runtime)
    # Stopped originals retain their IDs/weights and are never propagated as
    # live particles. Their finite exit placeholders cannot resurrect them.
    inputs = [np.where(initial_alive, getattr(bundle, name), np.nan)
              for name in ("x_m", "tx_rad", "y_m", "ty_rad")]
    energies = []
    last_progress = [0.]
    def progress(label):
        if time.monotonic()-last_progress[0] >= 10:
            print(str(label), flush=True)
            last_progress[0] = time.monotonic()
    with transport_progress(progress):
        z, x, tx, y, ty, flight, checkpoints = execute_unified_plan(state, plan, *inputs,
            bundle.energy_offset_ev, initial_time_s=bundle.flight_time_s,
            initial_kinetic_energy_ev=state.electron_gun.nominal_exit_energy_ev+bundle.energy_offset_ev,
            return_flight_times=True, energy_output=energies, history_policy="sampled",
            defer_nonfinite_until_clipping=True)
    alive, blocked, keys = merge_unified_stops(z, initial_alive, gun.blocked_z_mm, list(gun.blocked_key))
    alive, blocked, keys = clip_recording_planes(state, z, x, y, alive, blocked, keys)
    keys = np.asarray(keys, str)
    np.savez_compressed(args.output/"classical_column.npz", z_mm=z, x_m=x, y_m=y,
        tx_rad=tx, ty_rad=ty, flight_time_s=flight, energy_ev=energies[0],
        alive=alive, blocked_z_mm=blocked, blocked_key=keys, weight=weight,
        original_id=bundle.ray_id)
    report["planes"] = []
    for label, target in (("specimen_entrance", entrance), *((p.key, p.z_mm)
            for p in state.recording_planes if p.z_mm <= detector.z_mm)):
        index = int(np.argmin(abs(z-target)))
        if abs(z[index]-target) > 1e-8:
            raise ValueError("A requested physical observation plane was not retained")
        before = initial_alive & (~np.isfinite(blocked) | (blocked >= target))
        report["planes"].append({"key": label, "z_mm": float(target),
            **population(weight, before, gun.emitted_current_a),
            "phase_space": phase_space_summary(x[index], y[index], tx[index], ty[index], weight, before)})
    report["interceptions"] = {key: population(weight, keys == key, gun.emitted_current_a)
        for key in sorted(set(keys)) if key}
    report["remaining"] = population(weight, alive, gun.emitted_current_a)
    report["focus_profile"] = []
    for target in focus_planes:
        index = int(np.argmin(abs(z-target)))
        if abs(z[index]-target) > 1e-9:
            raise ValueError("A requested focus observation plane was not retained")
        before = initial_alive & (~np.isfinite(blocked) | (blocked >= target))
        row = phase_space_summary(x[index], y[index], tx[index], ty[index], weight, before)
        row.update(z_mm=float(z[index]), offset_nm=float((z[index]-entrance)*1e6))
        report["focus_profile"].append(row)
    residual = float(weight.sum()-weight[alive].sum()-sum(weight[keys == key].sum()
        for key in set(keys) if key))
    report["absolute_weight_balance_residual"] = residual
    if abs(residual) > 1e-12:
        raise ValueError("Absolute original-source weight accounting failed")
    report["classical_status"] = "EXECUTED_PHYSICAL_CONFIGURATION_NOT_YET_QUALIFIED"
    report["classical_material_interaction"] = "Not evaluated; this is the geometric field/geometry reference for the coherent material calculation"
    if intervals:
        from types import SimpleNamespace
        from temsim.optics.waist_evidence import finite_interval, observed_aperture_losses
        cp = SimpleNamespace(z_mm=z, x_m=x, y_m=y, tx_rad=tx, ty_rad=ty)
        masks = initial_alive[None, :] & (~np.isfinite(blocked)[None, :] | (z[:, None] <= blocked[None, :]))
        interceptions = observed_aperture_losses(state, cp, masks, weight, bundle.ray_id)
        topology = {}
        common = masks & masks[np.argmin(abs(z-entrance))][None, :]
        for label, lo, hi, step in intervals:
            topology[label] = {
                "actual_population": finite_interval(cp, masks, weight, lo, hi,
                    maximum_spacing_mm=step+1e-9, source_ids=bundle.ray_id, known_interceptions=interceptions),
                "specimen_surviving_cohort": finite_interval(cp, common, weight, lo, hi,
                    maximum_spacing_mm=step+1e-9, source_ids=bundle.ray_id)}
        write_report(args.output/"topology.json", {"intervals": topology,
            "source_counts": report["source_counts"], "original_weights_preserved": True})
    report["integration_nodes"] = len(plan.z_mm)
    report["retained_history_nodes"] = len(z)
    write_report(args.output/"report.json", report)


def hybrid(state, args, report):
    from temsim.physics.hybrid_pipeline import HybridGrid, HybridRequest, simulate_hybrid
    from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics
    from temsim.physics.tip_wave_pipeline import TipWaveRequest
    from temsim.hybrid_cli import save_result
    wave = TipWaveRequest(stop="gun_exit")
    wave = replace(wave, source=replace(wave.source, grid_pixels=args.source_grid_pixels,
        energy_samples=args.energy_samples), gun=replace(wave.gun,
        near_tip_end_z_nm=args.near_tip_end_z_nm,
        near_tip_transverse_radius_um=args.near_tip_radius_um), execution=replace(wave.execution,
        cache_directory=str((args.wave_cache or args.output/"wave_cache").resolve())))
    start = None
    if args.coherent_start_before_sample_nm is not None:
        start = state.sample.z_mm-state.sample.thickness_nm*.5e-6-args.coherent_start_before_sample_nm*1e-6
        wave = replace(wave, column_support_projection=True, column_weighted_resampling=True)
    wave = replace(wave, column_exponential_method=args.column_exponential_method,
        column_kinetic_order=args.column_kinetic_order)
    if args.column_spatial_compaction:
        # Existing step-boundary spatial projection. It records every discarded
        # tail in the original cumulative support ledger, without normalization.
        wave = replace(wave, column_support_projection=True,
            column_weighted_resampling=False)
    if args.column_segment_resampling:
        # Existing opt-in numerical representation policy. Keep the original
        # strict probability/L2 limits, all energies and the physical fields.
        wave = replace(wave, column_support_projection=True, column_weighted_resampling=True,
            column_exploratory_preview=True, column_preview_segment_resampling=True,
            column_preview_pixels=8192)
    numerics = SemiclassicalNumerics(frame_stride=args.frame_stride,
        window_sigma_pixels=float(args.frame_stride), packet_batch=512,
        maximum_packets=args.maximum_packets)
    request = HybridRequest(args.detector, HybridGrid(state.sample.z_mm-state.sample.thickness_nm*.5e-6,
        args.sample_pixels, args.sample_width_nm), source_particles=args.source_particles,
        exit_particles=args.exit_particles, wave=wave, semiclassical=numerics,
        coherent_start_z_mm=start,
        geometry_step_mm=args.geometry_step_mm,
        maximum_geometry_steps=args.maximum_geometry_steps)
    report["hybrid_request"] = asdict(request)
    report["stage"] = "HYBRID"
    write_report(args.output/"report.json", report)
    last_progress = [0.]
    def progress(done, total, message):
        if time.monotonic()-last_progress[0] >= 5 or (total and done == total):
            print(f"[{done}/{total}] {message}", flush=True)
            write_report(args.output/"hybrid_progress.json", dict(done=done, total=total, message=str(message),
                utc=datetime.now(timezone.utc).isoformat()))
            last_progress[0] = time.monotonic()
    result = simulate_hybrid(state, request, progress_callback=progress,
        cancelled=lambda: bool(args.cancel_file and args.cancel_file.exists()))
    save_result(result, args.output/"hybrid")
    report["hybrid_status"] = result.report["status"]


def reuse_classical_reference(state, args, report):
    """Reuse only a completed exact-input reference, not a new source boundary."""
    from temsim.instrument_snapshot import InstrumentSnapshot
    folder = args.classical_reference.resolve()
    expected = InstrumentSnapshot.from_dict(json.loads((folder/"instrument.json").read_text()))
    current = capture_instrument_snapshot(state)
    if expected.digest != current.digest:
        raise ValueError("Classical reference has different complete instrument inputs or solver bytes")
    old = json.loads((folder/"report.json").read_text())
    if old["status"] != "COMPLETED_DECLARED_SCOPE" or old["stage"] != "CLASSICAL_COLUMN":
        raise ValueError("Only a completed classical reference may be reused")
    with np.load(folder/"original_source.npz", allow_pickle=False) as arrays:
        weights, ids = arrays["exit_weight"], arrays["exit_ray_id"]
        if np.count_nonzero(weights > 0) != args.source_particles or len(np.unique(ids)) != len(ids):
            raise ValueError("Classical source count/identity does not match the requested original source")
    report["classical_reference"] = {"directory": str(folder), "input_digest": expected.digest,
        "sha256": {name: sha256((folder/name).read_bytes()).hexdigest() for name in
            ("report.json", "original_source.npz", "classical_column.npz")},
        "scope": "Read-only exact-input classical evidence; hybrid still traces the complete actual gun source"}
    for key in ("source_counts", "reference_current_a", "emitted", "gun_exit", "gun_losses", "planes",
        "interceptions", "remaining", "absolute_weight_balance_residual", "focus_profile", "classical_status"):
        report[key] = old[key]
    write_report(args.output/"instrument.json", current.to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--optical-control-seed", type=Path,
        help="Use only scalar lens/corrector drives from a historical snapshot as an unqualified new proposal")
    parser.add_argument("--optical-controls", type=Path)
    parser.add_argument("--camera-acquisition", action="store_true",
        help="Explicitly withdraw other recording receivers, retaining apertures and the actual detector position")
    parser.add_argument("--silicon-110", action="store_true")
    parser.add_argument("--thickness-nm", type=float, default=.4)
    parser.add_argument("--focus-scan-half-width-nm", type=float, default=0.)
    parser.add_argument("--focus-scan-step-nm", type=float, default=1.)
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--source-particles", type=int, default=1024)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--coherence-length-nm", type=float)
    parser.add_argument("--detector", default="camera")
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--wave-cache", type=Path,
        help="Optional existing cache; normal exact-input and code checks remain mandatory")
    parser.add_argument("--classical-reference", type=Path,
        help="Reuse completed classical evidence only when its full instrument identity exactly matches")
    parser.add_argument("--static-acquisition", action="store_true",
        help="Explicitly disable scan/wobble drives for a separate stationary acquisition; retain coils and DC controls")
    parser.add_argument("--source-grid-pixels", type=int, default=128)
    parser.add_argument("--energy-samples", type=int, default=3)
    parser.add_argument("--near-tip-end-z-nm", type=float, default=4000.)
    parser.add_argument("--near-tip-radius-um", type=float, default=10.)
    parser.add_argument("--sample-pixels", type=int, default=256)
    parser.add_argument("--sample-width-nm", type=float, default=128.)
    parser.add_argument("--exit-particles", type=int, default=4096)
    parser.add_argument("--geometry-step-mm", type=float, default=.1)
    parser.add_argument("--maximum-geometry-steps", type=int, default=10000)
    parser.add_argument("--coherent-start-before-sample-nm", type=float,
        help="Use the same physical gun and global coherent field propagation up to this distance before the sample, then semiclassical propagation")
    parser.add_argument("--frame-stride", type=int, default=32)
    parser.add_argument("--maximum-packets", type=int, default=262144)
    parser.add_argument("--column-exponential-method", choices=("taylor", "chebyshev"), default="taylor")
    parser.add_argument("--column-kinetic-order", type=int, choices=(2, 4, 6, 8), default=4,
        help="Existing positive-branch kinetic expansion; retain the declared cumulative phase budget")
    parser.add_argument("--column-segment-resampling", action="store_true",
        help="Use the existing exploratory segment-start resampling with unchanged strict support budgets")
    parser.add_argument("--column-spatial-compaction", action="store_true",
        help="Use existing step-boundary spatial support projection with unchanged cumulative error budgets")
    parser.add_argument("--cancel-file", type=Path,
        help="Creating this file requests cancellation at the next checked operator boundary")
    args = parser.parse_args(argv)
    if args.hybrid:
        from temsim.physics.retired_tip_wave import reject_retired_tip_coherent_workflow
        reject_retired_tip_coherent_workflow()
    if args.column_segment_resampling and args.column_spatial_compaction:
        parser.error("Select either segment spectral resampling or step-boundary spatial compaction")
    if args.source_particles < 1000:
        parser.error("Physical trials require at least 1000 original positive-current particles")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema": "physical-feg-hybrid-validation-v1", "status": "RUNNING",
        "utc": datetime.now(timezone.utc).isoformat(), "stage": "PREPARATION",
        "requested_source_particles": args.source_particles,
        "source_assumption": "Configured classical source" if args.coherence_length_nm is None else
            "Explicit uncalibrated Gaussian-Schell tip coherence length; physical geometry, width, current and energy retained",
        "lens_retuning": bool(args.optical_control_seed or args.optical_controls),
        "aperture_withdrawal": False, "receiver_withdrawal": args.camera_acquisition,
        "qualification": "Development validation; not a measured or OEM FEG reconstruction"}
    try:
        if args.hybrid:
            from temsim.physics.compute_backend import require_coherent_cuda
            require_coherent_cuda("Require GPU")
            import cupy as cp
            report["cuda"] = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                "executed_reduction": float(cp.arange(1024, dtype=cp.float64).sum().get())}
        state, selection, changes = prepare(args)
        report["acquisition"] = {"mode": "stationary" if args.static_acquisition else "configured",
            "control_changes": changes, "dynamic_fields_frozen": False}
        from temsim.optics.electron_gun.tip_coherence import describe_physical_tip_source, generate_tip_boundary_emission, TipWaveNumerics
        report["source_definition"] = thaw_json(describe_physical_tip_source(state.electron_gun.emitter))
        report["sample"] = {"inserted": state.sample.inserted, "mode": state.sample.specimen_mode,
            "z_mm": state.sample.z_mm, "thickness_nm": state.sample.thickness_nm}
        report["receivers"] = [{"key": p.key, "z_mm": p.z_mm, "inserted": p.inserted}
            for p in state.recording_planes]
        if args.coherence_length_nm is not None:
            emission = generate_tip_boundary_emission(state.electron_gun,
                TipWaveNumerics(grid_pixels=args.source_grid_pixels, energy_samples=args.energy_samples))
            report["source_wave_preflight"] = thaw_json(emission.record)
        save_profile(args.output/"input.toml", state, selection)
        if args.classical_reference:
            reuse_classical_reference(state, args, report)
        else:
            classical(state, args, report)
        if args.hybrid:
            hybrid(state, args, report)
        report["status"] = "COMPLETED_DECLARED_SCOPE"
    except Exception as error:
        report.update(status="CANCELLED" if isinstance(error, InterruptedError) else "FAILED",
            error_type=type(error).__name__, error=str(error))
        (args.output/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        report["elapsed_s"] = time.monotonic()-started
        write_report(args.output/"report.json", report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
