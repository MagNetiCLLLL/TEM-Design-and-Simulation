"""Replay one frozen classical column from its real, archived gun-exit source.

No source generation, optical fitting, count reduction, retry, or tolerance
adaptation is performed. The separately documented legacy-model waist-size
allowance is not an optimization error budget. All paths are explicit evidence;
an output prefix is reserved once and never overwritten, even after failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from temsim.calculation_manifest import solver_source_identity
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
from temsim.optics.model import State
from temsim.physics.particle_sections import _digest_arrays

# Support direct execution and importlib-based regression tests equally.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))
from validate_particle_count_convergence import (  # noqa: E402
    clean, execution_identity, physical_input_signature, require_finite_population,
)

DEFAULT_DIRECTORY = SCRIPT_DIRECTORY.parent / "outputs/extended-optical-validation-20261027"
PHASE_KEYS = ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")
SOURCE_KEYS = ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s",
               "energy_offset_ev", "weight", "alive", "ray_id")
EXACT_KEYS = ("z_mm", "alive", "weight", "source_id", "source_energy_offset_ev")
# Fixed before execution; allclose uses the native reference as its scale.
TOLERANCES = {
    "x_m": (5e-15, 5e-10), "y_m": (5e-15, 5e-10),
    "tx_rad": (5e-13, 5e-10), "ty_rad": (5e-13, 5e-10),
    "flight_time_s": (5e-21, 5e-12), "kinetic_energy_ev": (2e-8, 0.0),
}
PLAN_KEYS = ("field_digest", "geometry_digest", "integration_rows", "checkpoint_rows",
             "z_sha256", "checkpoint_index_sha256", "observation_z_sha256")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, data) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(data), stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_paths(output: Path) -> dict[str, Path]:
    if output.suffix.lower() != ".json":
        raise ValueError("--output must name a new .json report")
    return {name: output.with_suffix(suffix) for name, suffix in (
        ("report", ".json"), ("arrays", ".npz"), ("inputs", ".inputs.json"),
        ("snapshot", ".snapshot.json"), ("reservation", ".started.json"))}


def reserve_output(output: Path) -> dict[str, Path]:
    paths = output_paths(output)
    occupied = [str(path) for path in paths.values() if path.exists()]
    if occupied:
        raise FileExistsError("Preserve existing benchmark evidence: " + ", ".join(occupied))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_new(paths["reservation"], dict(
        utc=datetime.now(timezone.utc).isoformat(), output=str(output.resolve()),
        policy="One attempt; never replace existing evidence or retry automatically",
        tolerances={key: dict(atol=value[0], rtol=value[1]) for key, value in TOLERANCES.items()},
        exact_arrays=list(EXACT_KEYS)))
    return paths


def bytes_equal(first, second) -> bool:
    """Compare actual bytes, including signed zero and NaN payloads."""
    first, second = np.asarray(first), np.asarray(second)
    return bool(first.shape == second.shape and first.dtype == second.dtype
                and np.array_equal(np.ascontiguousarray(first).view(np.uint8),
                                   np.ascontiguousarray(second).view(np.uint8)))


def compare_arrays(actual: dict, reference: dict) -> dict:
    """Bounded-memory comparisons; inactive paired NaNs are not live results."""
    rows = {}
    for key in (*PHASE_KEYS, *EXACT_KEYS):
        value, old = np.asarray(actual[key]), np.asarray(reference[key])
        same_layout = value.shape == old.shape and value.dtype == old.dtype
        row = dict(shape=list(value.shape), reference_shape=list(old.shape),
                   dtype=str(value.dtype), reference_dtype=str(old.dtype),
                   bitwise_equal=bytes_equal(value, old), passed=False)
        rows[key] = row
        if not same_layout:
            row["reason"] = "Shape or dtype changed"
            continue
        if key in EXACT_KEYS:
            row.update(passed=row["bitwise_equal"], comparison="exact_bytes")
            if np.issubdtype(value.dtype, np.number):
                row["max_absolute_difference"] = float(np.max(
                    np.abs(value.astype(float) - old.astype(float)), initial=0.0))
            else:
                row["different_elements"] = int(np.count_nonzero(value != old))
            continue
        atol, rtol = TOLERANCES[key]
        if value.shape != np.asarray(actual["alive"]).shape or value.shape != np.asarray(reference["alive"]).shape:
            row["reason"] = "State and alive-mask shapes differ"
            continue
        a, b = value.reshape(-1), old.reshape(-1)
        live_a = np.asarray(actual["alive"], bool).reshape(-1)
        live_b = np.asarray(reference["alive"], bool).reshape(-1)
        maximum = 0.0
        unequal = outside = bad_nonfinite = 0
        for start in range(0, a.size, 262144):
            stop = start + 262144
            aa, bb = a[start:stop], b[start:stop]
            finite = np.isfinite(aa) & np.isfinite(bb)
            paired_inactive_nan = (np.isnan(aa) & np.isnan(bb)
                                   & ~live_a[start:stop] & ~live_b[start:stop])
            bad_nonfinite += int(np.count_nonzero(~finite & ~paired_inactive_nan))
            difference = np.abs(aa[finite] - bb[finite])
            maximum = max(maximum, float(np.max(difference, initial=0.0)))
            unequal += int(np.count_nonzero(difference))
            outside += int(np.count_nonzero(difference > atol + rtol * np.abs(bb[finite])))
        row.update(atol=atol, rtol=rtol, comparison="abs(actual-reference) <= atol + rtol*abs(reference)",
                   max_absolute_difference=maximum, unequal_finite_values=unequal,
                   outside_tolerance=outside, invalid_nonfinite_values=bad_nonfinite,
                   passed=outside == 0 and bad_nonfinite == 0)
    return dict(passed=all(row["passed"] for row in rows.values()), arrays=rows,
                all_arrays_bitwise_equal=all(row["bitwise_equal"] for row in rows.values()))


def check_snapshot_identity(inputs, baseline, historical, current) -> None:
    """Implementation may change; complete physical graph and external bytes may not."""
    if physical_input_signature(inputs) != baseline["physical_input_signature"]:
        raise ValueError("Archived physical inputs do not match the baseline report")
    if historical.digest != baseline["snapshot_digest"]:
        raise ValueError("Archived snapshot digest does not match the baseline report")
    if execution_identity(historical) != baseline["execution_identity"]:
        raise ValueError("Baseline solver or external input identity disagrees with its snapshot")
    if historical.graph != current.graph:
        raise ValueError("The complete physical/numerical input graph changed")
    if historical.external_inputs != current.external_inputs:
        raise ValueError("Content-pinned external input identities changed")
    if execution_identity(current)["external_inputs"] != baseline["execution_identity"]["external_inputs"]:
        raise ValueError("External input identity changed")


def check_source(source, reference, baseline, frozen, *, nominal_exit_energy_ev: float):
    """Validate real cached source ordering and return the original initial K."""
    identity = _digest_arrays(*(source[key] for key in SOURCE_KEYS))
    if identity != baseline["source_population_identity"] or identity != frozen["source_population_identity"]:
        raise ValueError("Real cached gun source hash changed")
    count = int(baseline["rays"])
    if any(np.asarray(source[key]).shape != (count,) for key in SOURCE_KEYS):
        raise ValueError("Gun cache population shape changed")
    for key in SOURCE_KEYS:
        if not np.isfinite(source[key]).all():
            raise ValueError(f"Nonfinite original gun source: {key}")
    # This archived case has no gun stops. Refuse a different source contract
    # instead of inferring unavailable baseline stop identities.
    if (not np.asarray(source["alive"]).all() or not (source["weight"] > 0).all()
            or np.isfinite(source["blocked_z_mm"]).any() or np.any(source["blocked_key"] != "")):
        raise ValueError("Replay requires the archived wholly transmitted gun source")
    mapping = {key: reference[key][0] for key in PHASE_KEYS[:-1]}
    mapping.update(energy_offset_ev=reference["source_energy_offset_ev"],
                   weight=reference["weight"], alive=reference["alive"][0], ray_id=reference["source_id"])
    if any(not bytes_equal(source[key], mapping[key]) for key in SOURCE_KEYS):
        raise ValueError("Gun cache differs from the archived entrance phase, clock, weights or IDs")
    initial_k = float(nominal_exit_energy_ev) + source["energy_offset_ev"]
    if not np.isfinite(initial_k).all() or not (initial_k > 0).all():
        raise ValueError("Invalid original nominal-exit-plus-offset energy")
    return identity, initial_k


def baseline_column_seconds(timing, baseline, actual_plan) -> float:
    """Only the complete column stage is a like-for-like timing reference."""
    stages = [row for row in timing["stages"] if row["label"] == "complete_column_batch"]
    if len(stages) != 1:
        raise ValueError("Need exactly one complete_column_batch timing")
    seconds = float(stages[0]["seconds"])
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid complete column timing")
    if timing["optimized_implementation"] != baseline["execution_identity"]["solver"]:
        raise ValueError("Column timing belongs to a different baseline solver")
    if timing["column_batch_populations"] != [int(baseline["rays"])]:
        raise ValueError("Column timing belongs to a different source population")
    if len(timing["plans"]) != 1 or any(
            timing["plans"][0][key] != actual_plan[key] for key in PLAN_KEYS[:-1]):
        raise ValueError("Baseline timing used a different field, geometry or full mesh")
    for key in ("same_physical_input_signature", "same_source_population", "same_observation_mesh"):
        if timing.get(key) is not True:
            raise ValueError("Incomplete baseline provenance: " + key)
    return seconds


def code_identity() -> dict:
    return dict(solver=solver_source_identity(), benchmark=file_sha256(Path(__file__)),
                source_validation=file_sha256(SCRIPT_DIRECTORY / "validate_particle_count_convergence.py"))


@contextmanager
def monitor_execution(frozen_code, checks):
    """Emit real step progress and code-freeze evidence at ten-second intervals."""
    stopped, lock = Event(), Lock()
    latest = dict(label="Entering the column executor", completed=None, total=None)
    failures = []
    started = perf_counter()

    def callback(label):
        if failures:
            raise RuntimeError(failures[0])
        local = getattr(label, "local_progress", None)
        with lock:
            latest.update(label=str(label), completed=getattr(local, "completed", None),
                          total=getattr(local, "total", None))

    def heartbeat():
        while not stopped.wait(10.0):
            try:
                identity = code_identity()
                matched = identity == frozen_code
                checks.append(dict(stage="during_execute", elapsed_s=perf_counter()-started,
                                   matched=matched, identity=identity))
                if not matched:
                    failures.append("Source code changed during the frozen replay")
                with lock:
                    progress = latest.copy()
                print(json.dumps(dict(event="progress", elapsed_s=perf_counter()-started,
                                      code_frozen=matched, **progress)), flush=True)
            except Exception as exc:
                failures.append(f"Code-freeze monitoring failed: {exc}")
                return

    thread = Thread(target=heartbeat, name="column-benchmark-progress", daemon=True)
    thread.start()
    try:
        yield callback
    finally:
        stopped.set()
        thread.join()
    if failures:
        raise RuntimeError(failures[0])


def run(args, paths, report) -> None:
    from temsim import input_io
    from temsim.cpu_resources import numerical_job
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.core import build_propagation_plan
    from temsim.physics.particle_sections import _events
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.simulation import _column_checkpoint_planes
    from temsim.physics.transport_progress import transport_progress
    from temsim.physics.unified_field_particle import (
        execute_unified_plan, merge_unified_stops, unified_history_memory_bytes,
    )

    frozen_code = code_identity()
    report.update(code_freeze=[], solver_identity=frozen_code["solver"],
                  code_identity=frozen_code, source_recomputed=False, execution_attempts=0)

    def check_code(stage):
        actual = code_identity()
        matched = actual == frozen_code
        report["code_freeze"].append(dict(stage=stage, matched=matched, identity=actual))
        if not matched:
            raise RuntimeError("Source code changed during benchmark preparation or execution")

    check_code("before_evidence_load")
    evidence_paths = dict(baseline=args.baseline, inputs=args.inputs or args.baseline.with_suffix(".inputs.json"),
                          snapshot=args.snapshot or args.baseline.with_suffix(".snapshot.json"),
                          arrays=args.arrays or args.baseline.with_suffix(".npz"), source=args.source,
                          plan=args.frozen_plan, timing=args.baseline_timing)
    report["evidence"] = {key: dict(path=str(path.resolve()), sha256=file_sha256(path))
                          for key, path in evidence_paths.items()}
    inputs, baseline, frozen, timing = (read_json(evidence_paths[key]) for key in
                                       ("inputs", "baseline", "plan", "timing"))
    historical = InstrumentSnapshot.from_dict(read_json(evidence_paths["snapshot"]))
    with np.load(evidence_paths["arrays"], allow_pickle=False) as saved:
        reference = {key: saved[key] for key in (*PHASE_KEYS, *EXACT_KEYS)}
    with np.load(evidence_paths["source"], allow_pickle=False) as saved:
        source = {key: saved[key] for key in (*SOURCE_KEYS, "blocked_z_mm", "blocked_key")}
    state = State.from_dict(inputs)
    if state.to_dict() != inputs:
        raise ValueError("State restoration changed archived physical or numerical inputs")
    if state.step_mm != baseline["step_mm"] or state.electron_gun.emitter.ray_count != baseline["rays"]:
        raise ValueError("Archived count or integration step differs from the baseline report")
    current = capture_instrument_snapshot(state)
    check_snapshot_identity(inputs, baseline, historical, current)
    current.verify_current_inputs(state)
    source_identity, initial_k = check_source(source, reference, baseline, frozen,
        nominal_exit_energy_ev=state.electron_gun.nominal_exit_energy_ev)
    z = reference["z_mm"]
    if (z.ndim != 1 or len(z) < 2 or not np.isfinite(z).all() or not (np.diff(z) > 0).all()
            or z[0] != state.electron_gun.exit_plane_z_mm or z[-1] != state.sample.upper_surface_z_mm):
        raise ValueError("Archived exact observation planes do not span gun exit to specimen entrance")
    require_finite_population(SimpleNamespace(**reference), reference["alive"], source["weight"])
    count = len(source["ray_id"])
    report.update(rays=count, step_mm=state.step_mm, source_population_identity=source_identity,
                  physical_input_signature=physical_input_signature(inputs), snapshot_digest=current.digest,
                  execution_identity=execution_identity(current), baseline_solver_identity=historical.implementation,
                  initial_energy_definition="Original nominal_exit_energy_ev + cached source energy_offset_ev; never checkpoint K",
                  initial_energy_sha256=_digest_arrays(initial_k),
                  source=dict(emission_mean_ev=state.electron_gun.emitter.emission_energy_ev,
                              energy_spread_fwhm_ev=state.electron_gun.emitter.energy_spread_fwhm_ev,
                              total_original_weight=float(source["weight"].sum())),
                  timing_scope="execute_unified_plan only; source replay, plan, evidence checks, clipping and output I/O excluded")
    with input_io.input_scope(state, inherit=False):
        requested = tuple(sorted(set((z[0], z[-1], *z,
            *_column_checkpoint_planes(float(z[0]), float(z[-1]), count)))))
        plan = build_propagation_plan(state, float(z[0]), float(z[-1]), _events(state),
                                      save_z_mm=requested, checkpoint_z_mm=requested)
        actual_plan = dict(field_digest=plan.unified_runtime.snapshot.numerical_digest,
            geometry_digest=plan.unified_runtime.geometry.digest, integration_rows=len(plan.z_mm),
            checkpoint_rows=len(plan.checkpoint_index), z_sha256=_digest_arrays(plan.z_mm),
            checkpoint_index_sha256=_digest_arrays(plan.checkpoint_index), observation_z_sha256=_digest_arrays(z))
        if any(actual_plan[key] != frozen[key] for key in PLAN_KEYS):
            raise ValueError("Frozen field, geometry, integration/checkpoint mesh or observation planes changed")
        previous_seconds = baseline_column_seconds(timing, baseline, actual_plan)
        report.update(plan=actual_plan, baseline_column_s=previous_seconds,
                      baseline_cpu_threads=timing["optimized_threads"],
                      history_policy="checkpoints", history_budget_bytes=unified_history_memory_bytes(
                          state, plan, count, history_policy="checkpoints"), backend_receipts=[])
        check_code("after_preflight")
        write_json_new(paths["inputs"], inputs)
        write_json_new(paths["snapshot"], current.to_dict())
        if args.preflight_only:
            report.update(status="PREFLIGHT_ONLY", passed=False, preflight_passed=True,
                          limitation="No propagation or performance/accuracy claim")
            return
        print(json.dumps(dict(event="start", rays=count, step_mm=state.step_mm,
                              plan=actual_plan, baseline_column_s=previous_seconds,
                              tolerances={k: dict(atol=v[0], rtol=v[1]) for k, v in TOLERANCES.items()})), flush=True)
        with numerical_job(args.cpu_threads) as cpu, backend_receipts(report["backend_receipts"]):
            report["cpu_resources"] = asdict(cpu)
            with monitor_execution(frozen_code, report["code_freeze"]) as callback, transport_progress(callback):
                check_code("immediately_before_execute")
                report["execution_attempts"] = 1
                started = perf_counter()
                output = execute_unified_plan(state, plan,
                    source["x_m"], source["tx_rad"], source["y_m"], source["ty_rad"], source["energy_offset_ev"],
                    initial_time_s=source["flight_time_s"], initial_kinetic_energy_ev=initial_k,
                    return_flight_times=True, history_policy="checkpoints")
                report["column_s"] = perf_counter() - started
        check_code("immediately_after_execute")
        print(json.dumps(dict(event="execution_finished", column_s=report["column_s"])), flush=True)
        alive, stops, labels = merge_unified_stops(output[0], source["alive"],
                                                   source["blocked_z_mm"], list(source["blocked_key"]))
        alive, stops, labels = clip_recording_planes(state, output[0], output[1], output[3], alive, stops, labels)
        cp = output[-1]
        indexes = np.searchsorted(cp.z_mm, z)
        if np.any(indexes >= len(cp.z_mm)) or not bytes_equal(np.asarray(cp.z_mm)[indexes], z):
            raise ValueError("Executor did not retain exact requested observation planes")
        actual = {key: np.asarray(getattr(cp, key))[indexes].copy() for key in PHASE_KEYS}
        actual.update(z_mm=z.copy(), weight=source["weight"].copy(), source_id=source["ray_id"].copy(),
                      source_energy_offset_ev=source["energy_offset_ev"].copy(),
                      alive=source["alive"][None, :] & (~np.isfinite(stops)[None, :] | (stops[None, :] > z[:, None])))
        # Persist raw results before gates, so a failed scientific comparison is
        # diagnosable and never replaced by a retry or a relaxed budget.
        write_started = perf_counter()
        with paths["arrays"].open("xb") as stream:
            np.savez_compressed(stream, **actual, blocked_z_mm=stops, blocked_key=np.asarray(labels, dtype=str))
        report["archive_write_s"] = perf_counter()-write_started
        report["array_comparison"] = compare_arrays(actual, reference)
        require_finite_population(SimpleNamespace(**actual), actual["alive"], source["weight"])
        current.verify_current_inputs(state)
        if state.to_dict() != inputs or capture_instrument_snapshot(state).digest != current.digest:
            raise ValueError("Execution changed archived physical or numerical inputs")
        for key, path in evidence_paths.items():
            if file_sha256(path) != report["evidence"][key]["sha256"]:
                raise ValueError("An archived evidence file changed during execution: " + key)
        check_code("after_comparison_and_archive")
        report.update(passed=report["array_comparison"]["passed"],
                      status="PASS" if report["array_comparison"]["passed"] else "FAILED",
                      speedup=previous_seconds/report["column_s"], physical_inputs_changed=False,
                      original_ids_weights_offsets_preserved=True, surviving_particles=int(actual["alive"][-1].sum()),
                      specimen_statistics=asdict(transverse_beam_statistics(
                          *(actual[key][-1] for key in ("x_m", "y_m", "tx_rad", "ty_rad")),
                          weights=source["weight"], alive=actual["alive"][-1])),
                      limitation="Classical gun-exit-to-specimen replay only; legacy topology compatibility is assessed separately")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New .json report; companion files are also protected")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_DIRECTORY / "count-n12289-h005-optimized.json")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--arrays", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_DIRECTORY / "optimized-gun-exit.npz")
    parser.add_argument("--frozen-plan", type=Path, default=DEFAULT_DIRECTORY / "frozen-12289-plan.json")
    parser.add_argument("--baseline-timing", type=Path, default=DEFAULT_DIRECTORY / "complete-performance-comparison.json")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true", help="Validate identities and mesh without executing propagation")
    args = parser.parse_args(argv)
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    paths = reserve_output(args.output)
    report = dict(schema="frozen-unified-column-benchmark-v1", utc=datetime.now(timezone.utc).isoformat(),
                  status="FAILED", passed=False,
                  tolerances={key: dict(atol=value[0], rtol=value[1]) for key, value in TOLERANCES.items()},
                  exact_arrays=list(EXACT_KEYS))
    try:
        run(args, paths, report)
    except BaseException as exc:
        report.update(status="FAILED", passed=False, error=dict(type=type(exc).__name__, message=str(exc)))
        write_json_new(paths["report"], report)
        raise
    write_json_new(paths["report"], report)
    print(json.dumps(dict(event="complete", status=report["status"], passed=report["passed"],
                          report=str(paths["report"].resolve()), column_s=report.get("column_s"),
                          speedup=report.get("speedup"))), flush=True)
    return 0 if report["passed"] or report.get("preflight_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
