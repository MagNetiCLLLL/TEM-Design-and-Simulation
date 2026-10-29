"""Headless, bounded classical experiments using the existing recipe records.

No QApplication is constructed. Plans and checkpoints use the same .temexp
format as the Design Explorer; completed points are written atomically.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
from hashlib import file_digest
import json
import math
from pathlib import Path
import signal
import sys
from threading import Event
from time import monotonic


EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_TIMEOUT = 124
EXIT_CANCELLED = 130
MAXIMUM_POINTS = 256
DEFAULT_MAXIMUM_POINTS = 16
DEFAULT_TIMEOUT_SECONDS = 600.0
_WAVE_FLAGS = ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer")
    return parsed


def _point_limit(value: str) -> int:
    parsed = _positive_int(value)
    if parsed > MAXIMUM_POINTS:
        raise argparse.ArgumentTypeError(f"At most {MAXIMUM_POINTS} points are supported")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Use a finite positive number")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Capture a bounded plan without propagating particles")
    source = plan.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", type=Path, help="Existing saved DesignRecipe JSON")
    source.add_argument("--default-nanoprobe", action="store_true", help="Capture current standard Finite-Field Nanoprobe controls")
    plan.add_argument("--rays", type=_positive_int, help="Required with --default-nanoprobe; explicit source sample count")
    plan.add_argument("--step-mm", type=_positive_float, help="Required with --default-nanoprobe; explicit integration step in mm")
    plan.add_argument("--name", default="Headless classical experiment")
    plan.add_argument("--axes", type=Path, required=True, help="JSON list of {path, values, unit} axes")
    plan.add_argument("--tolerances", type=Path, help="Optional JSON list of {metric, minimum, maximum}")
    plan.add_argument("--maximum-points", type=_point_limit, default=DEFAULT_MAXIMUM_POINTS)
    plan.add_argument("--output", type=Path, required=True, help="New .temexp plan; never overwritten")
    for name, help_text in (("run", "Run a plan into a new checkpoint"),
                            ("resume", "Continue only missing points of an existing checkpoint")):
        run = commands.add_parser(name, help=help_text)
        run.add_argument("input", type=Path, help="Existing .temexp record")
        if name == "run":
            run.add_argument("--output", type=Path, required=True, help="New .temexp run; never overwritten")
        run.add_argument("--timeout-seconds", type=_positive_float, default=DEFAULT_TIMEOUT_SECONDS,
                         help="Cooperative wall-time budget for this invocation (default: 600 seconds)")
    export = commands.add_parser("export", help="Export retained observations without calculating")
    export.add_argument("input", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--format", choices=("csv", "json", "report"), default="csv",
                        help="CSV plus manifest, joint JSON report, or readable joint report")
    return result


def _json_file(path: Path):
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Axis/tolerance input exceeds the 8 MiB budget")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _axes(path: Path):
    from temsim.design_experiments import SweepAxis
    document = _json_file(path)
    if not isinstance(document, list) or not document:
        raise ValueError("Axes must be a nonempty JSON list")
    axes = []
    for row in document:
        if (not isinstance(row, dict) or set(row) - {"path", "values", "unit"}
                or not isinstance(row.get("path"), str)
                or not isinstance(row.get("values"), list)
                or not isinstance(row.get("unit", ""), str)):
            raise ValueError("Each axis requires path, a numeric values list, and an optional unit")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row["values"]):
            raise ValueError("Axis values must be finite numbers, not booleans or strings")
        axes.append(SweepAxis(row["path"], tuple(row["values"]), row.get("unit", "")))
    return tuple(axes)


def _tolerances(path: Path | None):
    from temsim.design_experiments import ToleranceRule
    if path is None:
        return ()
    document = _json_file(path)
    if not isinstance(document, list):
        raise ValueError("Tolerances must be a JSON list")
    rules = []
    for row in document:
        if (not isinstance(row, dict) or set(row) - {"metric", "minimum", "maximum"}
                or not isinstance(row.get("metric"), str) or not row["metric"].strip()):
            raise ValueError("Each tolerance requires a metric and numeric minimum/maximum")
        for key in ("minimum", "maximum"):
            value = row.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("Tolerance bounds must be finite numbers")
        rules.append(ToleranceRule(**row))
    if len({row.metric for row in rules}) != len(rules):
        raise ValueError("A tolerance metric may appear only once")
    return tuple(rules)


def _reject_wave_payload(payload) -> None:
    representation = payload.get("electron_gun", {}).get("source_representation", "classical_particles")
    if representation != "classical_particles":
        raise ValueError("Classical experiment plans require a physical classical_particles source; the captured source was not changed")
    sample = payload.get("sample", {})
    enabled = [name for name in _WAVE_FLAGS if bool(sample.get(name, False))]
    if enabled:
        raise ValueError("This experiment CLI supports classical particles only; coherent inputs were retained and rejected: "
                         + ", ".join(enabled))


def _classical_plan(recipe, sweep) -> None:
    _reject_wave_payload(recipe.state_payload)
    for point in sweep.points:
        _reject_wave_payload(point.state_payload)


def _calculate_classical(state, **kwargs):
    if getattr(state.electron_gun, "source_representation", "classical_particles") != "classical_particles":
        raise ValueError("Classical experiment execution requires a physical classical_particles source")
    enabled = [name for name in _WAVE_FLAGS if bool(getattr(state.sample, name, False))]
    if enabled:
        raise ValueError("Classical experiment execution rejects coherent inputs: " + ", ".join(enabled))
    from temsim.simulation_pipeline import calculate
    return calculate(state, **kwargs)


def _default_recipe(args):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.benchmark_suite import BenchmarkConfig, configured_state
    from temsim.design_experiments import recipe_from_snapshot
    from temsim.design_explorer import capture_design_snapshot, HighAccuracyRequest
    if args.rays is None or args.step_mm is None:
        raise ValueError("--default-nanoprobe requires explicit --rays and --step-mm")
    state = configured_state(BenchmarkConfig(rays=args.rays, step_mm=args.step_mm))
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    return recipe_from_snapshot(capture_design_snapshot(state, selection, slot="A",
        request=HighAccuracyRequest(args.rays, args.step_mm)), name=args.name)


def _observer_options():
    from temsim.performance_accuracy import OBSERVER_IDENTITY, observe_calculation
    return dict(point_observer=observe_calculation, point_observer_identity=OBSERVER_IDENTITY)


@contextmanager
def _new_destination(path: Path):
    """Reserve a new destination exclusively before an atomic publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    try:
        yield
    except BaseException:
        # A failed publication may leave our empty reservation, never user data.
        if path.exists() and path.stat().st_size == 0:
            path.unlink()
        raise


@contextmanager
def _record_lock(path: Path):
    """Prevent concurrent CLI writers from replacing each other's checkpoints."""
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("x", encoding="utf-8") as stream:
        stream.write("An experiment CLI writer owns this record.\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _record_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _plan(args) -> int:
    from temsim.design_experiments import load_recipe, plan_parameter_sweep
    from temsim.design_sweep_execution import execute_parameter_sweep
    from temsim.experiment_records import save_experiment
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    if args.recipe is not None:
        if args.rays is not None or args.step_mm is not None:
            raise ValueError("Saved recipes retain their numerical inputs; --rays/--step-mm apply only to --default-nanoprobe")
        recipe = load_recipe(args.recipe)
    else:
        recipe = _default_recipe(args)
    sweep = plan_parameter_sweep(recipe, _axes(args.axes), maximum_points=args.maximum_points)
    rules = _tolerances(args.tolerances)
    _classical_plan(recipe, sweep)
    # Reuse the execution preflight and its complete identity, with no point
    # calculation. A plan is an ordinary zero-point experiment record.
    planned = execute_parameter_sweep(recipe, sweep, tolerance_rules=rules,
        cancel_requested=lambda: True, calculator=_calculate_classical, **_observer_options())
    planned = replace(planned, cancelled=False,
        analysis_notes=("Plan only: no particle propagation or scientific qualification was performed.",))
    with _new_destination(args.output):
        save_experiment(args.output, recipe, sweep, planned, rules)
    print(f"Saved plan for {len(sweep.points)} points: {args.output}")
    return 0


@contextmanager
def _cancellation(deadline_seconds: float):
    cancelled = Event()
    deadline = monotonic() + deadline_seconds
    old_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda *_: cancelled.set())
    try:
        yield cancelled, lambda: monotonic() >= deadline
    finally:
        signal.signal(signal.SIGINT, old_handler)


def _execute(args) -> int:
    from temsim.design_sweep_execution import execute_parameter_sweep
    from temsim.experiment_records import load_experiment, save_experiment
    output = args.input if args.command == "resume" else args.output
    if args.command == "run" and output.exists():
        raise FileExistsError(f"Output already exists; use resume to continue it: {output}")
    with _record_lock(output):
        input_hash = _record_hash(args.input)
        recipe, sweep, prior, rules = load_experiment(args.input)
        if _record_hash(args.input) != input_hash:
            raise ValueError("Experiment record changed while being read; refusing to use stale inputs")
        _classical_plan(recipe, sweep)
        if args.command == "run":
            prior = replace(prior, point_results=(), sensitivities=(), tolerances=(), cancelled=False)
        observer = _observer_options()
        # Validate even an empty plan against current inputs, runtime, observer
        # and tolerances before creating or replacing a checkpoint.
        initial = execute_parameter_sweep(recipe, sweep, tolerance_rules=rules,
            resume=prior, cancel_requested=lambda: True,
            calculator=_calculate_classical, **observer)
        if _record_hash(args.input) != input_hash:
            raise ValueError("Experiment record changed during preflight; refusing to overwrite it")
        expected_hash = input_hash
        if args.command == "run":
            with _new_destination(output):
                save_experiment(output, recipe, sweep, initial, rules)
            expected_hash = _record_hash(output)
        last_complete = initial

        def checkpoint(partial):
            nonlocal expected_hash, last_complete
            if _record_hash(output) != expected_hash:
                raise ValueError("Experiment checkpoint changed outside this run; refusing to overwrite it")
            save_experiment(output, recipe, sweep, partial, rules)
            expected_hash = _record_hash(output)
            last_complete = partial

        def progress(value):
            if value.stage_completed == value.stage_total:
                print(f"{value.completed_points}/{value.total_points}: {value.stage}", flush=True)

        with _cancellation(args.timeout_seconds) as (cancelled, timed_out):
            try:
                final = execute_parameter_sweep(recipe, sweep, tolerance_rules=rules,
                    resume=prior, calculator=_calculate_classical, progress_callback=progress,
                    cancel_requested=lambda: cancelled.is_set() or timed_out(),
                    point_completed_callback=checkpoint, **observer)
            except KeyboardInterrupt:
                cancelled.set()
                final = replace(last_complete, cancelled=True,
                    analysis_notes=(*last_complete.analysis_notes, "Interrupted; only completed point checkpoints were retained."))
            checkpoint(final)
            print(f"Retained {final.completed_points}/{len(sweep.points)} points: {output}")
            if final.cancelled:
                return EXIT_TIMEOUT if timed_out() and not cancelled.is_set() else EXIT_CANCELLED
        failed = any(row.status == "FAILED" for row in final.point_results)
        failed = failed or any(not rule.passed for assessment in final.tolerances for rule in assessment.results)
        return EXIT_FAILED if failed else 0


def _export(args) -> int:
    from temsim.experiment_records import load_experiment, export_experiment_table
    from temsim.immutable_json import thaw_json
    recipe, sweep, result, _rules = load_experiment(args.input)
    if args.format == "csv":
        manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
        with ExitStack() as stack:
            stack.enter_context(_new_destination(args.output))
            stack.enter_context(_new_destination(manifest))
            export_experiment_table(args.output, recipe, sweep, result)
    else:
        from temsim.performance_accuracy import summarize_runs, format_report, write_report
        runs = []
        for row in result.point_results:
            observation = thaw_json(row.evidence.get("point_observer"))
            runs.append(dict(case_id=f"{recipe.digest}:point-{row.point_index}:{row.state_digest}",
                phase="single", elapsed_seconds=row.duration_s,
                execution_status=row.status, failure_reason=row.failure_reason,
                observation=observation))
        report = summarize_runs(runs)
        with _new_destination(args.output):
            if args.format == "json":
                write_report(report, args.output)
            else:
                args.output.write_text(format_report(report) + "\n", encoding="utf-8")
    print(f"Exported retained observations: {args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"plan": _plan, "run": _execute, "resume": _execute, "export": _export}[args.command](args)
    except KeyboardInterrupt:
        print("Experiment interrupted; previously committed checkpoints remain available.", file=sys.stderr)
        return EXIT_CANCELLED
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        print(f"Experiment error: {exc}", file=sys.stderr)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
