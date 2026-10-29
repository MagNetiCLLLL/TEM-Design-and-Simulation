"""Reproducible, bounded benchmarks of existing production calculations.

The supervisor starts a fresh process for every case. Numerical kernels and
physical source models are never substituted. Preparation, observations and
archive I/O are recorded separately from the measured operation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from threading import Event, Thread, Timer
from time import perf_counter


SCHEMA = "temsim-unified-benchmark-suite-v1"
CASES = ("nanoprobe", "continuation", "scan_calibration", "coherent")


@dataclass(frozen=True)
class BenchmarkConfig:
    rays: int = 49
    step_mm: float = .05
    warm_repeats: int = 3
    reuse_repeats: int = 3
    threads: int | None = None
    budget_seconds: float = 1200.
    cleanup_grace_seconds: float = 30.
    profile: bool = False
    cases: tuple[str, ...] = CASES

    def validate(self):
        for name, minimum in (("rays", 1), ("warm_repeats", 3), ("reuse_repeats", 3)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.threads is not None and (type(self.threads) is not int or self.threads < 1):
            raise ValueError("threads must be a positive integer")
        for name in ("step_mm", "budget_seconds", "cleanup_grace_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.profile) is not bool:
            raise ValueError("profile must be Boolean")
        if not self.cases or len(set(self.cases)) != len(self.cases) or any(case not in CASES for case in self.cases):
            raise ValueError(f"cases must be unique members of {CASES}")
        return self


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"Unsupported benchmark JSON value: {type(value).__name__}")


def atomic_json(path, value):
    """Publish strict JSON atomically inside an already reserved directory."""
    path = Path(path)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_json_value) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _utc():
    return datetime.now(timezone.utc).isoformat()


class MemorySampler:
    """Optional sampled process RSS; never represent unmeasured VRAM as zero."""
    def __init__(self, interval=.02):
        self.interval = interval
        self.done = Event()
        self.thread = None
        self.process = None
        self.values = []
        self.error = None
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            self.error = "psutil is not installed; RSS was not measured"

    def _sample(self):
        try:
            self.values.append(int(self.process.memory_info().rss))
        except Exception as error:
            self.error = f"RSS sampling unavailable: {type(error).__name__}: {error}"

    def _monitor(self):
        while not self.done.wait(self.interval):
            self._sample()

    def start(self):
        if self.process is not None:
            self._sample()
            self.thread = Thread(target=self._monitor, daemon=True)
            self.thread.start()

    def finish(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2.)
            self._sample()
        return {"peak_rss_bytes": max(self.values) if self.values else None,
                "peak_vram_bytes": None, "rss_status": "MEASURED_SAMPLED" if self.values else "NOT_MEASURED",
                "rss_sampling_interval_seconds": self.interval, "rss_samples": len(self.values),
                "rss_scope": "benchmark_worker_process_only", "rss_includes_descendants": False,
                "rss_note": self.error or "Worker RSS sampled at intervals; descendants are excluded and brief peaks may be missed",
                "vram_status": "NOT_MEASURED", "vram_note": "These cases use classical CPU execution"}


def configured_state(config, *, scan=False):
    """Capture the existing default finite-field preset without fitting it."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.vacuum import bind_gun_environment
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    apply_operating_mode_pair(state, "nano_probe", "diffraction",
        column_name=selection.column, recording_name=selection.recording)
    state.electron_gun.emitter.ray_count = config.rays
    state.step_mm = config.step_mm
    state.history_step_mm = max(config.step_mm, .5)
    state._tuning_quality = "High accuracy"
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    for part in (state.ac_deflector, state.descan_deflector):
        part.scan_enabled = bool(scan)
        if hasattr(part, "wobble_enabled"):
            part.wobble_enabled = False
    apply_physical_layout_to_state(state)
    bind_gun_environment(state)
    return state


class CaseRecorder:
    def __init__(self, directory, case_id, config, cancelled):
        self.directory, self.case_id = Path(directory), case_id
        self.config, self.cancelled = config, cancelled
        self.started = perf_counter()
        self.document = {"schema": SCHEMA, "case_id": case_id, "status": "RUNNING",
            "started_utc": _utc(), "pid": os.getpid(), "configuration": asdict(config),
            "runs": [], "auxiliary_operations": [], "limitations": [
                "First call in a fresh worker; OS, JIT disk and gun-field disk caches are not purged.",
                "Warm recomputation may reuse legitimate upstream gun/field caches; counters describe actual execution.",
                "Performance and repeatability are not scientific convergence or instrument qualification.",
                "Measured operation excludes state preparation, result observation and archive I/O; those durations are reported separately.",
                "Classical source population is complete for the requested N; the reduced default N=49 is a bounded benchmark, not a qualified production population."]}
        self.save()

    def save(self):
        self.document["worker_elapsed_seconds"] = perf_counter() - self.started
        atomic_json(self.directory / "case.json", self.document)

    def check(self):
        if self.cancelled.is_set():
            raise TimeoutError("Benchmark cooperative budget reached")

    def auxiliary(self, label, operation):
        self.check()
        self.document["active_operation"] = {"label": label, "timed_category": "auxiliary"}
        self.save()
        started = perf_counter()
        row = {"label": label, "status": "FAILED"}
        try:
            value = operation()
            row["status"] = "COMPLETE"
            return value
        except BaseException as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["elapsed_seconds"] = perf_counter() - started
            self.document["auxiliary_operations"].append(row)
            self.document.pop("active_operation", None)
            self.save()

    def measure(self, phase, repeat, operation, observe, counts=None):
        self.check()
        label = f"{phase}_{repeat}"
        self.document["active_operation"] = {"label": label, "timed_category": "calculation"}
        self.save()
        before = counts.snapshot() if counts is not None else None
        column_start = len(counts.column_calls) if counts is not None else 0
        sampler = MemorySampler()
        sampler.start()
        row = {"case_id": self.case_id, "phase": phase, "repeat": repeat,
               "execution_status": "FAILED", "observation": None}
        started = perf_counter()
        error = None
        value = None
        try:
            value = operation()
        except BaseException as caught:
            error = caught
        row["elapsed_seconds"] = perf_counter() - started
        row["memory"] = sampler.finish()
        if counts is not None:
            row["operation_counts"] = counts.since(before)
            row["column_calls"] = counts.column_calls[column_start:]
        observed = perf_counter()
        if error is None:
            try:
                observe(value, row)
                self.check()
                row["execution_status"] = "COMPLETE"
            except BaseException as caught:
                error = caught
        row["observation_seconds"] = perf_counter() - observed
        if error is not None:
            row["error"] = f"{type(error).__name__}: {error}"
        self.document["runs"].append(row)
        self.document.pop("active_operation", None)
        self.save()
        print(json.dumps({"case_id": self.case_id, "phase": phase, "repeat": repeat,
            "elapsed_seconds": row["elapsed_seconds"], "status": row["execution_status"]}), flush=True)
        if error is not None:
            raise error
        return value

    def profile(self, operation):
        if not self.config.profile:
            return
        import cProfile
        import pstats
        def run():
            profiler = cProfile.Profile()
            try:
                return profiler.runcall(operation)
            finally:
                profiler.dump_stats(str(self.directory / "profile.prof"))
                with (self.directory / "profile.txt").open("w", encoding="utf-8") as stream:
                    stream.write("Scope: benchmark worker process only; child-process execution is not profiled.\n"
                                 "Blocking wait/readline/lock entries can represent child work, not physical kernels.\n\n")
                    stream.write("Top 30 by cumulative time\n")
                    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
                    stream.write("\nTop 30 by own time\n")
                    pstats.Stats(profiler, stream=stream).sort_stats("tottime").print_stats(30)
        self.auxiliary("separate_profile_pass_not_in_benchmark_timings", run)


def _require_continuation_execution(value, row, prefix_z):
    """Reject an entrance restart presented as nonzero prefix reuse."""
    metrics = value.simulation.metrics
    if not metrics.get("section_gun_reused") or not metrics.get("section_reused_prefix"):
        raise AssertionError("Requested continuation did not reuse the executed gun and column prefix")
    if metrics.get("section_resume_z_mm") != prefix_z:
        raise AssertionError("Requested continuation did not resume at the saved prefix plane")
    segments = value.simulation.section_checkpoint.segments
    if len(segments) != 1 or segments[0].name != "incident":
        raise AssertionError("Bounded continuation requires one incident-column segment")
    plan = segments[0].plan
    total = len(plan.z_mm)-1
    calls = row.get("column_calls", ())
    if len(calls) != 1:
        raise AssertionError("Bounded continuation requires one measured column execution")
    call = calls[0]
    index, intervals = call["start_index"], call["requested_intervals"]
    if (not 0 < index < total or not 0 < intervals < total or intervals != total-index
            or plan.z_mm[index] != prefix_z or call["start_z_mm"] != prefix_z
            or call["stop_z_mm"] != plan.z_mm[-1]
            or row["operation_counts"].get("requested_column_intervals") != intervals):
        raise AssertionError("Continuation did not reduce the original plan by an exact nonzero prefix")
    return {"resume_z_mm": prefix_z, "start_index": index, "total_plan_intervals": total,
            "executed_intervals": intervals, "retained_prefix_intervals": index}


def _particle_case(recorder, state, snapshot, counts, *, continuation):
    from temsim.simulation_pipeline import calculate_particle_section
    from temsim.particle_section_io import save_section_result, load_section_result
    from temsim.particle_benchmark import (section_terminal_arrays, require_exact_arrays,
        require_exact_prefix, section_result_receipt, continuation_planes)
    from temsim.performance_accuracy import observe_calculation
    prefix_z, target = continuation_planes(state) if continuation else (None, float(state.sample.upper_surface_z_mm))
    recorder.document["scope"] = (
        "Production tip emission and physical gun with bounded incident-column continuation; "
        "saved prefix to a local pre-specimen terminal plane; static raster drives; no specimen observation"
        if continuation else
        "Complete production tip emission and incident column to the specimen upper surface; static raster drives; no specimen scattering")
    recorder.document["target_z_mm"] = target
    reference = None

    def fresh():
        value = snapshot.restore()
        value._tuning_cancelled = recorder.cancelled.is_set
        return value

    def observe(value, row, previous=None, reuse=False):
        row["production_receipt"] = section_result_receipt(value)
        if previous is not None:
            if not value.simulation.metrics.get("section_gun_reused"):
                raise AssertionError("Requested continuation did not reuse the executed gun")
            if not value.simulation.metrics.get("section_reused_prefix"):
                raise AssertionError("Requested continuation did not reuse the executed prefix")
            if continuation and not reuse:
                row["continuation_execution"] = _require_continuation_execution(value, row, prefix_z)
            row["exact_prefix"] = require_exact_prefix(previous, value)
        row["observation"] = observe_calculation(value.state_snapshot, value)
        if reference is not None:
            row["terminal_comparison"] = require_exact_arrays(section_terminal_arrays(reference),
                section_terminal_arrays(value), "Repeated terminal result")
        if reuse and row["operation_counts"].get("requested_column_intervals", 0):
            raise AssertionError("Exact-result reuse unexpectedly executed new column intervals")
        if not reuse and not row["operation_counts"].get("requested_column_intervals", 0):
            raise AssertionError("Declared particle transport executed no new column intervals")

    initial = None
    archive = recorder.directory / "prefix.temresult"
    if continuation:
        recorder.document["prefix_z_mm"] = prefix_z
        recorder.document["cold_semantics"] = "First continuation in a fresh worker, after separately timed production prefix calculation and archive save/load"
        current = recorder.auxiliary("prefix_state_preparation", fresh)
        initial = recorder.auxiliary("prefix_production_calculation",
            lambda: calculate_particle_section(current, prefix_z, (), acquire_stem=False))
        recorder.auxiliary("prefix_archive_save", lambda: save_section_result(initial, archive))
    else:
        recorder.document["cold_semantics"] = "First tip-to-specimen production calculation in a fresh worker"

    def prepare(label):
        if continuation:
            previous = recorder.auxiliary(label + "_prefix_archive_load", lambda: load_section_result(archive))
            previous.state_snapshot._tuning_cancelled = recorder.cancelled.is_set
            return previous.state_snapshot, previous
        return recorder.auxiliary(label + "_state_preparation", fresh), None

    for phase, repeats in (("cold", 1), ("warm", recorder.config.warm_repeats)):
        for repeat in range(repeats):
            current, previous = prepare(f"{phase}_{repeat}")
            value = recorder.measure(phase, repeat,
                lambda: calculate_particle_section(current, target, (), existing_result=previous, acquire_stem=False),
                lambda value, row: observe(value, row, previous), counts)
            if reference is None:
                reference = value
    recorder.auxiliary("terminal_archive_save",
        lambda: save_section_result(reference, recorder.directory / "terminal.temresult"))
    for repeat in range(recorder.config.reuse_repeats):
        current, _ = prepare(f"reuse_{repeat}")
        recorder.measure("reuse", repeat,
            lambda: calculate_particle_section(current, target, (), existing_result=reference, acquire_stem=False),
            lambda value, row: observe(value, row, reference, True), counts)
    current, previous = prepare("profile") if recorder.config.profile else (None, None)
    recorder.profile(lambda: calculate_particle_section(current, target, (), existing_result=previous, acquire_stem=False))


def _scan_case(recorder, snapshot):
    from copy import deepcopy
    import numpy as np
    from temsim.immutable_json import thaw_json
    from temsim.physics.unified_scan_cache import clear_scan_response_cache
    from temsim.physics.unified_scan_calibration import calibrate_unified_scan
    from temsim.particle_benchmark import array_inventory, require_exact_arrays
    reference = None
    recorder.document["scope"] = "Actual shared-field finite-coil AC/Descan response and drive solve; differential response probes, not emitted source particles"
    recorder.document["cold_semantics"] = "First real response measurement in a fresh worker with an isolated empty response cache; JIT and OS caches are retained"

    def fresh():
        state = snapshot.restore()
        state._tuning_cancelled = recorder.cancelled.is_set
        return state

    def arrays(state):
        return {"ac_command_mrad": np.asarray(state.ac_deflector.scan_command_matrix_mrad),
                "descan_command_mrad": np.asarray(state.descan_deflector.scan_command_matrix_mrad),
                "ac_coupling": np.stack(state.ac_deflector.coil_kick_matrices()),
                "descan_coupling": np.stack(state.descan_deflector.coil_kick_matrices())}

    def inspect(state, value, row, expected_source):
        nonlocal reference
        if value is None:
            raise AssertionError("Calibration performed no requested response measurement")
        receipt = thaw_json(getattr(state.ac_deflector, "_scan_execution_receipt", {}))
        source = receipt.get("response_cache", {}).get("source")
        if source != expected_source:
            raise AssertionError(f"Expected actual {expected_source} response, got {source}")
        measured = arrays(state)
        if any(not np.isfinite(value).all() for value in measured.values()):
            raise AssertionError("Calibration produced nonfinite matrices")
        comparison = require_exact_arrays(reference, measured, "Calibration matrices") if reference is not None else None
        if reference is None:
            reference = deepcopy(measured)
        row["calibration_observation"] = {"schema": "temsim-calibration-benchmark-observation-v1",
            "input_identity": snapshot.digest, "implementation_identity": snapshot.implementation,
            "matrices": array_inventory(measured), "comparison": comparison,
            "receipt": receipt, "sample_position_relative_residual": float(value[1]),
            "qualification": "Executed local response; not probe or microscope qualification"}
        row.pop("observation", None)

    for phase, repeats in (("cold", 1), ("warm", recorder.config.warm_repeats), ("reuse", recorder.config.reuse_repeats)):
        for repeat in range(repeats):
            expected_source = "computed"
            if phase != "reuse":
                recorder.auxiliary(f"{phase}_{repeat}_isolated_response_cache_clear",
                    lambda: clear_scan_response_cache(include_disk=True))
            elif repeat % 2:
                # Real disk decoding in the same worker, not a fabricated cold process.
                recorder.auxiliary(f"reuse_{repeat}_memory_cache_clear",
                    lambda: clear_scan_response_cache(include_disk=False))
                expected_source = "disk"
            else:
                expected_source = "memory"
            state = recorder.auxiliary(f"{phase}_{repeat}_state_preparation", fresh)
            recorder.measure(phase, repeat, lambda: calibrate_unified_scan(state, force=True),
                lambda value, row: inspect(state, value, row, expected_source))
    if recorder.config.profile:
        recorder.auxiliary("profile_isolated_response_cache_clear", lambda: clear_scan_response_cache(include_disk=True))
        state = recorder.auxiliary("profile_state_preparation", fresh)
        recorder.profile(lambda: calibrate_unified_scan(state, force=True))


def run_worker(config, directory, case_id):
    """Run one case; this entry point is launched only by the supervisor."""
    config.validate()
    directory = Path(directory)
    os.environ["TEMSIM_SCAN_RESPONSE_CACHE_DIR"] = str((directory / "scan-response-cache").resolve())
    cancelled = Event()
    recorder = CaseRecorder(directory, case_id, config, cancelled)
    timer = Timer(config.budget_seconds, cancelled.set)
    timer.daemon = True
    timer.start()
    try:
        from temsim.cpu_resources import initialize_cpu_resources, numerical_job
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.calculation_manifest import solver_source_identity
        from temsim.particle_benchmark import OperationCounts, environment_receipt, counter_overhead_probe
        initialize_cpu_resources()
        recorder.document["environment"] = environment_receipt()
        recorder.document["environment"]["executable"] = sys.executable
        recorder.document["environment"]["psutil"] = _version("psutil")
        recorder.document["implementation_identity"] = solver_source_identity()
        state = recorder.auxiliary("default_state_and_preset_preparation", lambda: configured_state(config, scan=case_id == "scan_calibration"))
        snapshot = recorder.auxiliary("input_snapshot_capture", lambda: capture_instrument_snapshot(state))
        recorder.document["input_identity"] = snapshot.digest
        recorder.document["workload"] = {"model": "unified", "illumination": "nano_probe", "projector": "diffraction",
            "source": "Existing full tip emission distribution and physical gun propagation",
            "rays": config.rays, "step_mm": config.step_mm, "requested_backend": state.acceleration_backend,
            "scan_enabled": case_id == "scan_calibration", "lens_controls": "Stored default preset; no fit or retuning"}
        recorder.document["input_snapshot"] = str(directory / "inputs.json")
        recorder.auxiliary("input_snapshot_save", lambda: atomic_json(directory / "inputs.json", snapshot.to_dict()))
        state._tuning_cancelled = cancelled.is_set
        counts = OperationCounts()
        recorder.document["counter_overhead_probe"] = counter_overhead_probe()
        with numerical_job(config.threads) as budget, counts.observe():
            recorder.document["cpu_resources"] = budget.to_dict()
            if case_id in ("nanoprobe", "continuation"):
                _particle_case(recorder, state, snapshot, counts, continuation=case_id == "continuation")
            elif case_id == "scan_calibration":
                _scan_case(recorder, snapshot)
            else:
                raise ValueError("Coherent case is declaration-only and must not execute")
        recorder.document["operation_counts_total"] = counts.snapshot()
        recorder.check()
        after = solver_source_identity()
        recorder.document["implementation_identity_after"] = after
        if after != snapshot.implementation:
            raise RuntimeError("Implementation changed during benchmark execution")
        recorder.document["status"] = "COMPLETE"
    except BaseException as error:
        recorder.document.update(status="CANCELLED" if cancelled.is_set() else "FAILED",
            error=f"{type(error).__name__}: {error}")
    finally:
        timer.cancel()
        recorder.document["finished_utc"] = _utc()
        recorder.save()
    return 0 if recorder.document["status"] == "COMPLETE" else 1


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _summaries(root, cases):
    from temsim.performance_accuracy import summarize_runs, format_report
    particle_runs = [run for case in cases if case["case_id"] in ("nanoprobe", "continuation") for run in case.get("runs", ())]
    report = summarize_runs(particle_runs)
    report["suite_execution"] = [{"case_id": case["case_id"], "status": case["status"],
        "error": case.get("error")} for case in cases]
    report["calibration_summary"] = []
    for case in cases:
        if case["case_id"] != "scan_calibration":
            continue
        runs = case.get("runs", ())
        report["calibration_summary"].append({"status": case["status"],
            "medians_seconds": {phase: statistics.median(values) for phase in ("cold", "warm", "reuse")
                if (values := [row["elapsed_seconds"] for row in runs
                               if row["phase"] == phase and row["execution_status"] == "COMPLETE"])},
            "runs": len(runs), "scope": case.get("scope"),
            "cache_sources": [row.get("calibration_observation", {}).get("receipt", {}).get("response_cache", {}).get("source") for row in runs]})
    atomic_json(root / "accuracy-summary.json", report)
    lines = [format_report(report).rstrip(), "", "Suite execution (separate from numerical acceptance)"]
    for case in cases:
        lines.append(f"{case['case_id']}: {case['status']}")
        detail = case.get("error") or case.get("reason")
        if detail:
            lines.append("  " + str(detail))
    for row in report["calibration_summary"]:
        lines.extend(("", f"Scan calibration: {row['status']}; {row['runs']} retained runs",
                      "  " + str(row.get("scope", ""))))
        for phase, seconds in row["medians_seconds"].items():
            lines.append(f"  {phase} median: {seconds:.9g} s")
        lines.append("  Response sources: " + ", ".join(str(value) for value in row["cache_sources"]))
        lines.append("  Local response and exact matrix repeatability only; no full-beam qualification.")
    (root / "accuracy-summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _require_case_run_inventory(document, config, case_id):
    """Validate every expected worker run before suite-level aggregation."""
    if document.get("schema") != SCHEMA or document.get("case_id") != case_id:
        raise ValueError("Worker receipt schema or case identity changed")
    rows = document.get("runs")
    if not isinstance(rows, list):
        raise ValueError("Worker receipt has no run inventory")
    expected = {(phase, repeat) for phase, count in (("cold", 1),
        ("warm", config.warm_repeats), ("reuse", config.reuse_repeats)) for repeat in range(count)}
    seen = set()
    for row in rows:
        if (not isinstance(row, dict) or row.get("case_id") != case_id
                or row.get("execution_status") != "COMPLETE"
                or type(row.get("phase")) is not str or type(row.get("repeat")) is not int):
            raise ValueError("Worker run has incomplete execution or phase/repeat identity")
        key = (row["phase"], row["repeat"])
        if key not in expected or key in seen:
            raise ValueError("Worker run has an unexpected or duplicate phase/repeat")
        elapsed = row.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Worker run has no finite nonnegative elapsed time")
        seen.add(key)
    if seen != expected:
        raise ValueError("Worker run inventory is incomplete for the requested cold/warm/reuse counts")


def run_suite(config, output):
    config.validate()
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    configuration = root / "configuration.json"
    atomic_json(configuration, asdict(config))
    suite = {"schema": SCHEMA, "status": "RUNNING", "started_utc": _utc(),
             "configuration": asdict(config), "cases": [], "acceptance_status": "PENDING",
             "particle_repeatability": "PENDING" if any(case in ("nanoprobe", "continuation") for case in config.cases)
                                       else "NOT_APPLICABLE"}
    atomic_json(root / "suite.json", suite)
    from temsim.validation_process import run_bounded, ValidationCleanupError
    active_directory = None
    active_case_id = None
    try:
        for case_id in config.cases:
            directory = root / case_id
            directory.mkdir()
            active_directory, active_case_id = directory, case_id
            if case_id == "coherent":
                document = {"schema": SCHEMA, "case_id": case_id, "status": "NOT_RUN", "runs": [],
                    "reason": "Declared separately: coherent execution requires an explicitly scoped fixture and verified usable CUDA; no coherent calculation is requested by this suite"}
                atomic_json(directory / "case.json", document)
            else:
                command = [sys.executable, "-m", "temsim.benchmark_suite", "--worker-config", str(configuration),
                           "--worker-case", case_id, "--worker-directory", str(directory)]
                environment = dict(os.environ)
                source_root = str(Path(__file__).resolve().parents[1])
                environment["PYTHONPATH"] = os.pathsep.join(filter(None, (source_root, environment.get("PYTHONPATH"))))
                environment["QT_QPA_PLATFORM"] = "offscreen"
                if config.threads is not None:
                    environment["TEMSIM_CPU_THREADS"] = str(config.threads)
                began = perf_counter()
                atomic_json(directory / "case.json", {"schema": SCHEMA, "case_id": case_id,
                    "status": "STARTING", "runs": [], "started_utc": _utc()})
                with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
                    result = run_bounded(command, timeout=config.budget_seconds + config.cleanup_grace_seconds,
                        env=environment, stdout=stdout, stderr=stderr)
                document = json.loads((directory / "case.json").read_text(encoding="utf-8"))
                document["supervisor"] = {"elapsed_seconds": perf_counter()-began, "returncode": result.returncode,
                    "hard_timeout_seconds": config.budget_seconds + config.cleanup_grace_seconds}
                if result.returncode != 0 or document.get("status") != "COMPLETE":
                    document["status"] = "TIMED_OUT" if result.returncode == 124 else "FAILED"
                    document.setdefault("error", f"Worker exited with code {result.returncode}; see stdout.log and stderr.log")
                else:
                    try:
                        _require_case_run_inventory(document, config, case_id)
                        document["run_inventory_status"] = "VALID"
                    except (ValueError, TypeError, OverflowError) as error:
                        document.update(status="FAILED", run_inventory_status="INVALID",
                                        error=f"Invalid worker run inventory: {error}")
                atomic_json(directory / "case.json", document)
            suite["cases"].append(document)
            active_directory = None
            active_case_id = None
            atomic_json(root / "suite.json", suite)
        suite["status"] = "COMPLETE" if all(case["status"] in ("COMPLETE", "NOT_RUN") for case in suite["cases"]) else "FAILED"
        report = _summaries(root, suite["cases"])
        particles_requested = any(case in ("nanoprobe", "continuation") for case in config.cases)
        suite["particle_repeatability"] = ((report or {}).get("repeatability", {}).get("status", "NOT_EVALUATED")
                                           if particles_requested else "NOT_APPLICABLE")
        suite["acceptance_status"] = (
            "FAIL" if suite["status"] != "COMPLETE" or
                      (particles_requested and suite["particle_repeatability"] != "PASS")
            else "PASS" if particles_requested else "NOT_APPLICABLE")
    except BaseException as error:
        suite.update(status="CLEANUP_UNCONFIRMED" if isinstance(error, ValidationCleanupError) else "FAILED",
            error=f"{type(error).__name__}: {error}", acceptance_status="FAIL")
        if active_directory is not None:
            path = active_directory / "case.json"
            try:
                interrupted = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                interrupted = {"schema": SCHEMA, "case_id": active_case_id, "runs": []}
            interrupted.update(status=suite["status"], error=suite["error"], finished_utc=_utc())
            atomic_json(path, interrupted)
            suite["cases"].append(interrupted)
        if isinstance(error, (KeyboardInterrupt, SystemExit, ValidationCleanupError)):
            raise
    finally:
        suite["finished_utc"] = _utc()
        atomic_json(root / "suite.json", suite)
    return 0 if suite["status"] == "COMPLETE" and suite["acceptance_status"] in ("PASS", "NOT_APPLICABLE") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; existing evidence is never overwritten")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--rays", type=int, default=49)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--reuse-repeats", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--budget-seconds", type=float, default=1200., help="Cooperative budget per worker, including setup and observations")
    parser.add_argument("--cleanup-grace-seconds", type=float, default=30.)
    parser.add_argument("--profile", action="store_true", help="Additional separately recorded cProfile pass")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.worker_config:
            values = json.loads(args.worker_config.read_text(encoding="utf-8"))
            values["cases"] = tuple(values["cases"])
            return run_worker(BenchmarkConfig(**values), args.worker_directory, args.worker_case)
        if args.output is None:
            parser.error("--output is required")
        config = BenchmarkConfig(args.rays, args.step_mm, args.warm_repeats, args.reuse_repeats,
            args.threads, args.budget_seconds, args.cleanup_grace_seconds, args.profile, tuple(args.cases)).validate()
        return run_suite(config, args.output)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
