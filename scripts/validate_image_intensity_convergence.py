"""Collect real Image-screen particles while independently refining N and h.

The previous comparison supplies input parameters only. Its historical result
cache is never relabelled or reused. Every particle count executes a fresh
physical gun under the current implementation, then the public section pipeline
reuses its identity-checked gun for fixed C2 perturbations and integration steps.
This vacuum diagnostic does not certify material imaging or coherent resolution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
SCHEMA = "image-intensity-convergence-v1"
COLLECTOR_SCHEMA = "image-intensity-collector-v2-source-sampling"
SOURCE_MODES = ("legacy", "sobol-antithetic")
CASES = {"baseline": {}, "Cs_plus20mm": {"cs_mm": 20.},
         "A1_plus0.03mm": {"a1_mm": .03}, "B2_plus1mm": {"b2_mm": 1.}}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def matrix_pairs(counts, steps, fixed_population, count_step):
    """Independent axes, with their shared cell executed once."""
    cells = {(int(n), float(count_step)) for n in counts}
    cells.update((int(fixed_population), float(h)) for h in steps)
    return sorted(cells, key=lambda item: (item[0], -item[1]))


def run_identifier(count, step, case):
    return f"N{count}_h{repr(float(step)).replace('.', 'p')}_{case}"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        default=Path("outputs/c2-screen-aberrations-20261017"))
    parser.add_argument("--output", type=Path,
                        help="fresh output directory; defaults depend on the explicit source mode")
    parser.add_argument("--source-mode", choices=SOURCE_MODES, default="legacy")
    parser.add_argument("--source-seed", type=int, default=0,
                        help="captured Sobol scramble seed; legacy requires the unused default zero")
    parser.add_argument("--counts", type=int, nargs="+")
    parser.add_argument("--steps", type=float, nargs="+")
    parser.add_argument("--fixed-population", type=int)
    parser.add_argument("--count-step", type=float, default=.00625)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=list(CASES))
    parser.add_argument("--backend", choices=("CPU", "Numba CPU", "Prefer GPU", "Require GPU"),
                        default="Require GPU")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="continue only an exactly matching protocol and verified source/result files")
    parser.add_argument("--prepare-only", action="store_true",
                        help="validate input-only restoration and prepare the manifest; no propagation")
    parser.add_argument("--emit-only", action="store_true",
                        help="save actual emitted launch samples and diagnostics; no gun or column propagation")
    args = parser.parse_args(argv)
    new_source = args.source_mode == "sobol-antithetic"
    if args.output is None:
        args.output = Path("outputs/image-intensity-sobol-convergence-20261017" if new_source
                           else "outputs/image-intensity-convergence-20261017")
    if args.counts is None:
        args.counts = [4096, 16384, 65536] if new_source else [3000, 12000, 48000]
    if args.steps is None:
        args.steps = [.00625] if new_source else [.05, .025, .0125, .00625]
    if args.fixed_population is None:
        args.fixed_population = min(args.counts)
    if args.prepare_only and args.emit_only:
        parser.error("Choose only one of --prepare-only and --emit-only")
    if args.source_seed < 0 or args.source_seed >= 2**32:
        parser.error("Source seed must be an unsigned 32-bit integer")
    if not new_source and args.source_seed != 0:
        parser.error("Legacy emission has no scramble seed; choose --source-mode explicitly")
    if min(*args.counts, args.fixed_population) < 16 or args.threads < 1:
        parser.error("Use at least 16 rays and a positive CPU thread limit")
    if new_source and any(n > 1048576 or n & (n-1) for n in (*args.counts, args.fixed_population)):
        parser.error("Sobol antithetic emission requires complete power-of-two populations up to 1048576")
    if not all(0 < step <= .1 for step in (*args.steps, args.count_step)):
        parser.error("Every integration step must lie in (0, 0.1] mm")
    args.counts = sorted(set(args.counts))
    args.steps = sorted(set(args.steps), reverse=True)
    args.cases = [case for case in CASES if case in args.cases]
    return args


def configure_source(emitter, mode, seed):
    """Select a captured production sampling record, never a replacement source."""
    if mode not in SOURCE_MODES:
        raise ValueError("Unsupported source mode")
    if emitter.quadrature is not None:
        raise ValueError("This joint-source protocol cannot truncate or replace product quadrature")
    if mode == "legacy":
        if seed != 0 or getattr(emitter, "sampling", None) is not None:
            raise ValueError("Legacy protocol requires the recorded default emission rule")
        return dict(mode=mode, record=None)
    from temsim.optics.electron_gun.emitter import EmissionSampling
    emitter.sampling = EmissionSampling(seed=seed)
    return dict(mode=mode, record=asdict(emitter.sampling))


def emission_diagnostics(emitter, source_rule):
    """Describe the production sampler at the actual requested population."""
    if source_rule["mode"] == "sobol-antithetic":
        from temsim.optics.electron_gun.tip_joint_sampling import sampling_diagnostics
        result = dict(sampling_diagnostics(emitter), requested_rays=int(emitter.ray_count))
    else:
        result = dict(schema="legacy-halton-finite-moment-v1", rays=int(emitter.ray_count),
            seed=None, energy_mapping="finite-population moment normalisation",
            nested_full_phase_space=False)
    # Fail here if a future diagnostic silently introduces NaN or opaque data.
    canonical_digest(result)
    return result


def emitted_arrays(emitter):
    """Retain actual production launch output without pretending it was propagated."""
    import numpy as np
    from temsim.physics.ray_identity import emission_reference
    bundle = emitter.emit()
    arrays = {key: np.asarray(getattr(bundle, key)) for key in
              ("ray_id", "x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight")}
    arrays.update({"emission_"+key: np.asarray(value) for key, value in
                   emission_reference(bundle, emitter.surface_model).items()})
    weights = arrays["weight"]
    if (any(value.dtype.hasobject for value in arrays.values())
            or np.any(~np.isfinite(weights)) or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError("Emitted samples require finite nonnegative probabilities summing to one")
    return arrays


def array_digest(arrays):
    import numpy as np
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        data = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(data.dtype.str.encode())
        digest.update(repr(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()


def write_npz(path, arrays):
    import numpy as np
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def source_arrays(result):
    import numpy as np
    gun = result.simulation.gun_trace
    bundle = gun.exit_bundle
    arrays = {"gun_exit_" + key: np.asarray(getattr(bundle, key)) for key in
        ("ray_id", "x_m", "y_m", "tx_rad", "ty_rad", "weight", "energy_offset_ev", "alive", "flight_time_s")}
    arrays.update(gun_blocked_key=np.asarray(gun.blocked_key, dtype=str),
                  gun_blocked_z_mm=np.asarray(gun.blocked_z_mm, dtype=np.float64))
    if gun.emission_reference is not None:
        arrays.update({"emission_" + key: np.asarray(value) for key, value in gun.emission_reference.items()})
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError("Fresh gun evidence must contain complete numerical arrays or Unicode stop labels")
    count = len(bundle.ray_id)
    if any(arrays[key].shape != (count,) for key in ("gun_blocked_key", "gun_blocked_z_mm")):
        raise ValueError("Executed gun stops must retain one value per source particle")
    return arrays


def validate_launch_against_gun(launch, source):
    """Verify regenerated deterministic launch coordinates against executed evidence."""
    import numpy as np
    for key in ("ray_id", "position_m", "direction", "normal"):
        if not np.array_equal(launch["emission_"+key], source["emission_"+key]):
            raise AssertionError("Executed gun emission differs from retained launch samples")
    if not np.array_equal(launch["weight"], source["gun_exit_weight"]):
        raise AssertionError("Executed gun changed emitted source probabilities")


def sampling_prefix_check(left_launch, right_launch, left_source=None, right_source=None):
    """Separate exact emission nesting from population-dependent gun integration."""
    import numpy as np
    left_n, right_n = len(left_launch["ray_id"]), len(right_launch["ray_id"])
    if not 0 < left_n <= right_n:
        raise ValueError("Prefix comparison requires increasing positive populations")
    fields = ("ray_id", "x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev")
    same = {key: bool(np.array_equal(left_launch[key], right_launch[key][:left_n]))
            for key in fields}
    weights_same = bool(np.array_equal(left_launch["weight"]*left_n,
                                      right_launch["weight"][:left_n]*right_n))
    report = dict(smaller_population=left_n, larger_population=right_n,
        emission_prefix_equal=same, weights_times_population_equal=weights_same,
        full_emission_prefix_equal=all(same.values()) and weights_same)
    if left_source is None or right_source is None:
        return report
    ids, li, ri = np.intersect1d(left_source["gun_exit_ray_id"], right_source["gun_exit_ray_id"],
                                assume_unique=True, return_indices=True)
    live_left, live_right = left_source["gun_exit_alive"][li], right_source["gun_exit_alive"][ri]
    alive = live_left & live_right
    differences = {}
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "flight_time_s"):
        a, b = left_source["gun_exit_"+key][li][alive], right_source["gun_exit_"+key][ri][alive]
        if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)):
            raise ValueError("Surviving gun prefix must contain finite phase space and clocks")
        delta = b-a
        differences[key] = dict(equal=bool(np.array_equal(a, b)),
            rms=float(np.sqrt(np.mean(delta*delta))) if delta.size else None,
            maximum_absolute=float(np.max(np.abs(delta))) if delta.size else None)
    report["gun_exit"] = dict(common_ids=len(ids), common_survivors=int(alive.sum()),
        changed_survival=int(np.count_nonzero(live_left != live_right)), fields=differences,
        interpretation="Actual common-ID gun differences; the shared adaptive time step may depend on the full population. Exact emitted prefixes do not guarantee identical gun trajectories.")
    return report


def screen_arrays(result, target, source_reference, emitted_current_a):
    import numpy as np
    from temsim.component_keys import FLUORESCENT_SCREEN
    segment = result.simulation.section_checkpoint.segments[-1]
    cp, branch = segment.checkpoints, segment.branch
    if float(cp.z_mm[-1]) != target:
        raise AssertionError("Receiver position is not an executed exact checkpoint")
    hits = (np.asarray(branch.blocked_key) == FLUORESCENT_SCREEN) & (branch.ray_weight > 0)
    arrays = dict(xy_m=np.column_stack((cp.x_m[-1], cp.y_m[-1])),
        slopes_rad=np.column_stack((cp.tx_rad[-1], cp.ty_rad[-1])),
        weight=np.asarray(branch.ray_weight), survivor=hits,
        source_id=np.asarray(branch.source_ray_id), kinetic_energy_ev=np.asarray(cp.kinetic_energy_ev[-1]),
        flight_time_s=np.asarray(cp.flight_time_s[-1]),
        blocked_key=np.asarray(branch.blocked_key, dtype=str),
        blocked_z_mm=np.asarray(branch.blocked_z, dtype=np.float64),
        final_alive=np.asarray(branch.alive, dtype=bool))
    count = len(arrays["source_id"])
    if any(arrays[key].shape != (count,) for key in ("blocked_key", "blocked_z_mm", "final_alive")):
        raise AssertionError("Physical-stop records must retain one value per source particle")
    if arrays["xy_m"].dtype != np.float64 or not np.any(hits):
        raise AssertionError("Exact float64 physical screen hits are required")
    if not np.array_equal(arrays["source_id"], source_reference["gun_exit_ray_id"]):
        raise AssertionError("Screen columns lost their original source ancestry")
    if not np.array_equal(arrays["weight"], source_reference["gun_exit_weight"]):
        raise AssertionError("Screen weights differ from original source probabilities")
    if float(result.simulation.gun_trace.emitted_current_a) != emitted_current_a:
        raise AssertionError("Emitted current changed within one particle count")
    return arrays


def main():
    args = arguments()
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import numpy as np
    from temsim.calculation_manifest import solver_source_identity
    from temsim.component_keys import FLUORESCENT_SCREEN
    from temsim.cpu_resources import numerical_job
    from temsim.detector.point_spread import DetectorPointSpread
    from temsim.input_design import restore_input_design
    from temsim.input_io import input_scope
    from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
    from temsim.particle_section_io import load_section_result, save_section_result
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.particle_sections import gun_dependency_signature
    from temsim.simulation_pipeline import calculate_particle_section

    started = perf_counter()
    reference, output = args.reference.resolve(), args.output.resolve()
    snapshot_path, old_report_path = reference / "baseline.snapshot.json", reference / "comparison.json"
    original = InstrumentSnapshot.from_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))
    old_report = json.loads(old_report_path.read_text(encoding="utf-8"))
    # This explicitly restores verified parameters for NEW execution. It never
    # changes an archived result's implementation identity or authorizes reuse.
    base = restore_input_design(original)
    if base.sample.inserted or base.sample.specimen_mode != "vacuum":
        raise ValueError("This protocol requires the recorded vacuum specimen inputs")
    if not base.fluorescent_screen.inserted:
        raise ValueError("The recorded main screen must be inserted")
    if any(receiver.inserted and str(receiver.key) != FLUORESCENT_SCREEN for receiver in base.recording_planes):
        raise ValueError("Another inserted receiver would change the physical first hit")
    base.electron_gun.emitter.ray_count = args.fixed_population
    source_rule = configure_source(base.electron_gun.emitter, args.source_mode, args.source_seed)
    base.acceleration_enabled = True
    base.acceleration_backend = args.backend
    base.projector_mode = "image"
    controls = old_report["projector_controls"]["imaging"]["lens_percent"]
    lenses = {lens.key: lens for lens in base.lenses}
    if not set(controls) <= set(lenses):
        raise ValueError("Recorded projector components are absent from the input design")
    for key, percent in controls.items():
        lenses[key].percent = percent
    c2_key = "condenser_lens_2"
    c2_cs = float(old_report["baseline_c2_cs_mm"])
    lenses[c2_key].cs_mm = c2_cs
    target = float(base.fluorescent_screen.z_mm)
    # A real positive-length entrance segment avoids the column-wall helper's
    # unsupported zero-interval observation. The retained gun exit stays exact.
    seed_target = float(base.electron_gun.exit_plane_z_mm) + .01
    implementation = solver_source_identity()
    pairs = matrix_pairs(args.counts, args.steps, args.fixed_population, args.count_step)
    protocol = dict(schema=SCHEMA, collector_schema=COLLECTOR_SCHEMA,
        implementation=implementation, source_sampling=source_rule,
        reference_snapshot_sha256=digest_file(snapshot_path), reference_snapshot_digest=original.digest,
        reference_report_sha256=digest_file(old_report_path),
        reference_implementation=original.implementation, script_sha256=digest_file(Path(__file__)),
        counts=args.counts, steps=args.steps, fixed_population=args.fixed_population,
        particle_count_step_mm=args.count_step, cases=args.cases, backend=args.backend,
        threads=args.threads, projector_controls=controls, receiver_z_mm=target,
        seed_boundary_z_mm=seed_target,
        coefficient_changes={case: CASES[case] for case in args.cases})
    protocol_identity = canonical_digest(protocol)
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("runs", "sources", "snapshots", "emission"):
        (output / folder).mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise ValueError("Output already exists; use --resume only for an identical protocol")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_identity") != protocol_identity:
            raise ValueError("Stored protocol/source/script/input identity differs; use a fresh output directory")
    else:
        manifest = dict(schema=SCHEMA, status="PREPARED", protocol=protocol,
            protocol_identity=protocol_identity, source_implementation=implementation,
            source_sampling=source_rule, source_sampling_prefix_checks={}, emission_probes={},
            fixed_population=args.fixed_population, particle_count_step_mm=args.count_step,
            matrix=[dict(rays=n, step_mm=h) for n, h in pairs], runs={}, sources={},
            point_spread=asdict(DetectorPointSpread.from_component(base.fluorescent_screen)),
            projector_controls=dict(controls),
            receiver_z_mm=target,
            scope="Fresh physical gun per N; public tip-to-screen sections; vacuum specimen; same physical source law and fixed Image projector settings",
            source_sampling_note=(
                "Complete scrambled Sobol populations with adjacent antithetic emission pairs and fixed energy calibration. Emitted phase-space prefixes are checked exactly; shared gun integration steps may still create small common-ID exit differences. Weights are original 1/N probabilities. No IID confidence interval is inferred from this one captured seed."
                if source_rule["mode"] == "sobol-antithetic" else
                "Halton positions/directions have nested prefixes, but the production energy distribution is moment-normalised independently at each N. Across-N ray IDs do not identify identical full physical initial states. No bootstrap or replicated particles are used."))

    def save():
        manifest["last_invocation_elapsed_s"] = perf_counter() - started
        write_json(manifest_path, manifest)

    def guard():
        if solver_source_identity() != implementation:
            raise RuntimeError("Production implementation changed during the experiment; no mixed result is accepted")

    def execute(state, z, previous):
        guard()
        result = calculate_particle_section(state, z, existing_result=previous, acquire_stem=False)
        guard()
        return result

    current_snapshot = capture_instrument_snapshot(base)
    manifest["current_input_snapshot_digest"] = current_snapshot.digest
    write_json(output / "current-input.snapshot.json", current_snapshot.to_dict())
    save()
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "cells": len(pairs), "runs": len(pairs)*len(args.cases),
                          "protocol_identity": protocol_identity, "output": str(output)}), flush=True)
        return
    if args.emit_only:
        if manifest["runs"] or manifest["sources"]:
            raise ValueError("Emission-only probing requires an output without executed transport")
        previous_launch = None
        for count in sorted({n for n, _ in pairs}):
            guard()
            probe_state = current_snapshot.restore()
            probe_state.electron_gun.emitter.ray_count = count
            launch = emitted_arrays(probe_state.electron_gun.emitter)
            diagnostic = emission_diagnostics(probe_state.electron_gun.emitter, source_rule)
            guard()
            launch_path = output / "emission" / f"launch_N{count}.npz"
            write_npz(launch_path, launch)
            manifest["emission_probes"][str(count)] = dict(rays=count,
                particles_path=str(launch_path.relative_to(output)), particles_sha256=digest_file(launch_path),
                launch_digest=array_digest(launch), sampler_diagnostics=diagnostic,
                scope="Production emission only; no gun or column propagation")
            if previous_launch is not None:
                check = sampling_prefix_check(previous_launch, launch)
                if source_rule["mode"] == "sobol-antithetic" and not check["full_emission_prefix_equal"]:
                    raise AssertionError("Configured Sobol sampler failed its exact emission-prefix contract")
                manifest["source_sampling_prefix_checks"][str(count)] = check
            previous_launch = launch
        manifest["status"] = "EMISSION_ONLY"
        save()
        print(json.dumps({"status": "EMISSION_ONLY", "counts": sorted({n for n, _ in pairs}),
                          "protocol_identity": protocol_identity, "output": str(output)}), flush=True)
        return
    manifest["status"] = "RUNNING"
    save()
    try:
        with numerical_job(args.threads):
            previous_launch = previous_source = None
            for count in sorted({n for n, _ in pairs}):
                state = current_snapshot.restore()
                state.electron_gun.emitter.ray_count = count
                state.step_mm = max(h for n, h in pairs if n == count)
                with input_scope(state, inherit=False):
                    signature = gun_dependency_signature(state)
                source_key = str(count)
                seed_path = output / "sources" / f"source_N{count}.temsection"
                source_path = output / "sources" / f"source_N{count}.npz"
                launch_path = output / "emission" / f"launch_N{count}.npz"
                launch = emitted_arrays(state.electron_gun.emitter)
                sampler_diagnostics = emission_diagnostics(state.electron_gun.emitter, source_rule)
                guard()
                if previous_launch is not None:
                    check = sampling_prefix_check(previous_launch, launch)
                    if source_rule["mode"] == "sobol-antithetic" and not check["full_emission_prefix_equal"]:
                        raise AssertionError("Configured Sobol sampler failed its exact emission-prefix contract")
                stored_source = manifest["sources"].get(source_key)
                if stored_source:
                    if (stored_source["gun_dependency_signature"] != signature
                            or stored_source["archive_sha256"] != digest_file(seed_path)
                            or stored_source["particles_sha256"] != digest_file(source_path)
                            or stored_source["launch_sha256"] != digest_file(launch_path)
                            or stored_source["launch_digest"] != array_digest(launch)
                            or stored_source["sampler_diagnostics"] != sampler_diagnostics):
                        raise ValueError(f"N={count} saved gun identity or files changed")
                    seed = load_section_result(seed_path)
                    source_reference = source_arrays(seed)
                    if array_digest(source_reference) != stored_source["source_digest"]:
                        raise ValueError("Saved gun source differs from its retained evidence")
                    print(f"N={count}: verified current-implementation gun restored", flush=True)
                else:
                    t0, receipts = perf_counter(), []
                    with backend_receipts(receipts):
                        seed = execute(state, seed_target, None)
                    source_reference = source_arrays(seed)
                    validate_launch_against_gun(launch, source_reference)
                    if len(source_reference["gun_exit_ray_id"]) != count:
                        raise AssertionError("The physical gun did not execute the requested source population")
                    # An interrupted invocation may have published this file
                    # before its manifest row. Replace that orphan only with
                    # this freshly executed source, never infer its validity.
                    save_section_result(seed, seed_path, overwrite=seed_path.exists())
                    write_npz(source_path, source_reference)
                    write_npz(launch_path, launch)
                    manifest["sources"][source_key] = dict(rays=count, runtime_s=perf_counter()-t0,
                        seed_boundary_z_mm=seed_target,
                        gun_dependency_signature=signature, source_digest=array_digest(source_reference),
                        archive_path=str(seed_path.relative_to(output)), archive_sha256=digest_file(seed_path),
                        particles_path=str(source_path.relative_to(output)), particles_sha256=digest_file(source_path),
                        launch_path=str(launch_path.relative_to(output)), launch_sha256=digest_file(launch_path),
                        launch_digest=array_digest(launch), sampler_diagnostics=sampler_diagnostics,
                        emitted_current_a=float(seed.simulation.gun_trace.emitted_current_a), backend_receipts=receipts)
                    save()
                    print(f"N={count}: fresh physical gun saved in {perf_counter()-t0:.2f}s", flush=True)
                validate_launch_against_gun(launch, source_reference)
                if previous_launch is not None:
                    manifest["source_sampling_prefix_checks"][str(count)] = sampling_prefix_check(
                        previous_launch, launch, previous_source, source_reference)
                    save()
                previous_launch, previous_source = launch, source_reference
                source_digest = array_digest(source_reference)
                source_current = float(seed.simulation.gun_trace.emitted_current_a)
                previous = seed
                result = None
                for n, step in (cell for cell in pairs if cell[0] == count):
                    for case in args.cases:
                        run_id = run_identifier(count, step, case)
                        run_state = current_snapshot.restore()
                        run_state.electron_gun.emitter.ray_count = count
                        run_state.step_mm = step
                        lens = next(lens for lens in run_state.lenses if lens.key == c2_key)
                        lens.cs_mm = c2_cs + CASES[case].get("cs_mm", 0.)
                        intrinsic = dict(run_state.lens_intrinsic_aberrations.get(c2_key, {}))
                        for key, value in CASES[case].items():
                            if key != "cs_mm":
                                intrinsic[key] = intrinsic.get(key, 0.) + value
                        run_state.lens_intrinsic_aberrations[c2_key] = intrinsic
                        run_snapshot = capture_instrument_snapshot(run_state)
                        request_identity = canonical_digest(dict(protocol=protocol_identity,
                            state=run_snapshot.digest, target_z_mm=target, run_id=run_id))
                        particles_path = output / "runs" / f"{run_id}.npz"
                        snapshot_file = output / "snapshots" / f"{run_id}.snapshot.json"
                        stored = manifest["runs"].get(run_id)
                        if stored and stored.get("status") == "COMPLETE":
                            if (stored["request_identity"] != request_identity
                                    or stored["source_digest"] != source_digest
                                    or digest_file(snapshot_file) != stored["snapshot_sha256"]
                                    or digest_file(particles_path) != stored["particles_sha256"]):
                                raise ValueError(f"Completed run integrity mismatch: {run_id}")
                            print(run_id + ": verified completed result retained", flush=True)
                            continue
                        write_json(snapshot_file, run_snapshot.to_dict())
                        actual_controls = {lens.key: float(lens.percent) for lens in run_state.lenses if lens.key in controls}
                        if actual_controls != controls:
                            raise AssertionError("Fixed Image projector controls changed")
                        row = dict(status="RUNNING", rays=count, step_mm=step, case=case,
                            request_identity=request_identity, source_digest=source_digest,
                            source_sampling=source_rule,
                            particles_path=str(particles_path.relative_to(output)),
                            snapshot_path=str(snapshot_file.relative_to(output)), snapshot_sha256=digest_file(snapshot_file),
                            receiver_z_mm=target, projector_controls=actual_controls,
                            implementation_before=implementation)
                        manifest["runs"][run_id] = row
                        save()
                        t0, receipts = perf_counter(), []
                        with backend_receipts(receipts):
                            result = execute(run_state, target, previous)
                        if array_digest(source_arrays(result)) != source_digest:
                            raise AssertionError("Source particles/weights/energies changed within one N")
                        if not result.simulation.metrics["section_gun_reused"]:
                            raise AssertionError("Compatible same-N execution did not reuse its validated physical gun")
                        arrays = screen_arrays(result, target, source_reference, source_current)
                        hits = arrays["survivor"]
                        write_npz(particles_path, arrays)
                        row.update(status="COMPLETE", runtime_s=perf_counter()-t0,
                            particles_sha256=digest_file(particles_path), backend_receipts=receipts,
                            implementation_after=solver_source_identity(),
                            intercepted_rays=int(hits.sum()), intercepted_probability=float(arrays["weight"][hits].sum()),
                            emitted_current_a=float(result.simulation.gun_trace.emitted_current_a),
                            current_pa=float(result.simulation.gun_trace.emitted_current_a)*1e12*float(arrays["weight"][hits].sum()),
                            gun_reused=True, section_reused_prefix=bool(result.simulation.metrics["section_reused_prefix"]),
                            column_cache=result.simulation.metrics.get("column_segment_cache"))
                        save()
                        print(f"{run_id}: {row['runtime_s']:.2f}s; {row['intercepted_rays']}/{count} physical screen hits", flush=True)
                        previous = result
                del previous, seed, result
                import gc
                gc.collect()
        guard()
        manifest["status"] = "COMPLETE"
        save()
        print(f"COMPLETE: {len(manifest['runs'])} real particle runs; {perf_counter()-started:.2f}s this invocation", flush=True)
    except Exception as exc:
        manifest.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


if __name__ == "__main__":
    main()
