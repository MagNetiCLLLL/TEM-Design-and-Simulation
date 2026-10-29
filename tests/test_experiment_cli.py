"""Headless plans and durable classical experiment boundaries, without physics runs."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from temsim import experiment_cli as cli
from temsim.calculation_cache import calculation_signatures
from temsim.design_experiments import (
    SweepAxis, ToleranceRule, plan_parameter_sweep, save_recipe,
)
from temsim.design_sweep_execution import execute_parameter_sweep
from temsim.experiment_records import load_experiment, save_experiment
from temsim.immutable_json import thaw_json
from temsim.simulation_pipeline import CalculationResult


ROOT = Path(__file__).resolve().parents[1]
AXIS = "lenses[condenser_lens_2].percent"


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "a" * 64)
    return cli._default_recipe(SimpleNamespace(rays=9, step_mm=2.5, name="Controlled CLI test"))


def _axis_file(tmp_path, values=(20., 40., 60.)):
    path = tmp_path / "axes.json"
    path.write_text(json.dumps([dict(path=AXIS, values=values, unit="%")]), encoding="utf-8")
    return path


def _observer(state, result):
    return {"scope": "Controlled scalar observer; no physical qualification",
            "value": next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_2")}


def _value(state):
    return next(lens.percent for lens in state.lenses if lens.key == "condenser_lens_2")


def _result(state):
    return CalculationResult(simulation=SimpleNamespace(metrics={
        "sample_illumination_diameter_95_um": _value(state),
        "sample_surviving_current_pa": 10.,
    }), energy_filter=None, state_snapshot=state, signatures=calculation_signatures(state))


@pytest.fixture
def planned(source, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_observer_options", lambda: dict(
        point_observer=_observer, point_observer_identity="controlled-observer-v1"))
    recipe = tmp_path / "source.json"
    save_recipe(recipe, source)
    axes = _axis_file(tmp_path)
    path = tmp_path / "plan.temexp"
    assert cli.main(["plan", "--recipe", str(recipe), "--axes", str(axes), "--output", str(path)]) == 0
    return path


def _subprocess(*args):
    code = """
import runpy, sys
sys.argv = ['temsim.experiment_cli', *sys.argv[1:]]
try:
    runpy.run_module('temsim.experiment_cli', run_name='__main__')
except SystemExit as exc:
    from PySide6.QtWidgets import QApplication
    assert QApplication.instance() is None, 'CLI created a QApplication'
    print('NO_QAPPLICATION')
    raise
"""
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), QT_QPA_PLATFORM="offscreen")
    return subprocess.run([sys.executable, "-c", code, *args], cwd=ROOT, env=environment,
                          capture_output=True, text=True, timeout=60)


def test_help_subprocess_does_not_create_a_qapplication():
    result = _subprocess("--help")
    assert result.returncode == 0, result.stderr
    assert "NO_QAPPLICATION" in result.stdout
    assert "plan" in result.stdout and "resume" in result.stdout


def test_default_plan_subprocess_is_explicit_and_does_not_run_physics(tmp_path):
    axes = _axis_file(tmp_path, (13.899040164772698,))
    output = tmp_path / "default.temexp"
    result = _subprocess("plan", "--default-nanoprobe", "--rays", "49", "--step-mm", ".05",
                         "--axes", str(axes), "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "NO_QAPPLICATION" in result.stdout
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["recipe"]["request"] == {"ray_count": 49, "step_mm": .05}
    assert record["recipe"]["base_state_payload"]["simulation_mode"] == "unified"
    assert record["result"]["point_results"] == []
    assert "no particle propagation" in record["result"]["analysis_notes"][0]


def test_plan_from_recipe_preserves_captured_controls_and_rejects_overwrite(planned, source):
    recipe, sweep, result, _ = load_experiment(planned)
    assert recipe.digest == source.digest
    assert recipe.instrument_snapshot == source.instrument_snapshot
    assert len(sweep.points) == 3 and not result.point_results
    before = planned.read_bytes()
    assert cli.main(["run", str(planned), "--output", str(planned)]) == cli.EXIT_INVALID
    assert planned.read_bytes() == before


@pytest.mark.parametrize("arguments", [[], ["--rays", "9"], ["--step-mm", ".05"]])
def test_default_plan_requires_both_explicit_numerical_inputs(tmp_path, arguments):
    output = tmp_path / "absent.temexp"
    assert cli.main(["plan", "--default-nanoprobe", *arguments, "--axes", str(_axis_file(tmp_path)),
                     "--output", str(output)]) == cli.EXIT_INVALID
    assert not output.exists()


@pytest.mark.parametrize("values", [[True], ["20"], [float("nan")], []])
def test_invalid_axis_values_are_rejected_before_execution(source, tmp_path, values):
    path = tmp_path / "source.json"
    save_recipe(path, source)
    output = tmp_path / "absent.temexp"
    assert cli.main(["plan", "--recipe", str(path), "--axes", str(_axis_file(tmp_path, values)),
                     "--output", str(output)]) == cli.EXIT_INVALID
    assert not output.exists()


@pytest.mark.parametrize("source_kind", ["wave_readout", "exit_source"])
def test_classical_plan_rejects_coherent_inputs_without_changing_them(source, tmp_path, source_kind):
    payload = thaw_json(source.base_state_payload)
    if source_kind == "wave_readout":
        payload["sample"]["stem_wave_enabled"] = True
    else:
        payload["electron_gun"]["source_representation"] = "coherent_exit_wave"
    coherent = replace(source, base_state_payload=payload)
    path = tmp_path / "coherent.json"
    save_recipe(path, coherent)
    before = path.read_bytes()
    output = tmp_path / "absent.temexp"
    assert cli.main(["plan", "--recipe", str(path), "--axes", str(_axis_file(tmp_path)),
                     "--output", str(output)]) == cli.EXIT_INVALID
    assert path.read_bytes() == before and not output.exists()


def test_runtime_entry_rejects_nonclassical_source_before_calculation(monkeypatch):
    state = SimpleNamespace(electron_gun=SimpleNamespace(source_representation="coherent_exit_wave"))
    monkeypatch.setattr("temsim.simulation_pipeline.calculate", lambda *_args, **_kwargs: pytest.fail("Rejected source must not calculate"))
    with pytest.raises(ValueError, match="classical_particles"):
        cli._calculate_classical(state)


def test_run_checkpoints_success_and_failure_before_the_next_point(planned, tmp_path, monkeypatch):
    output = tmp_path / "run.temexp"
    seen = []
    def calculate(state, **kwargs):
        record = load_experiment(output)[2]
        seen.append([row.status for row in record.point_results])
        if _value(state) == 40.:
            raise ValueError("Controlled failed point")
        return _result(state)
    monkeypatch.setattr(cli, "_calculate_classical", calculate)
    assert cli.main(["run", str(planned), "--output", str(output)]) == cli.EXIT_FAILED
    assert seen == [[], ["COMPLETE"], ["COMPLETE", "FAILED"]]
    result = load_experiment(output)[2]
    assert [row.status for row in result.point_results] == ["COMPLETE", "FAILED", "COMPLETE"]
    assert result.point_results[1].metrics == {}
    assert result.point_results[0].evidence["point_observer"]["value"] == 20.
    assert result.point_results[0].evidence["observer_duration_s"] >= 0
    assert not output.with_name(output.name + ".lock").exists()


def test_interrupt_retains_complete_points_and_resume_only_executes_missing(planned, tmp_path, monkeypatch):
    output = tmp_path / "run.temexp"
    calls = []
    def interrupted(state, **kwargs):
        calls.append(_value(state))
        if _value(state) == 40.:
            raise KeyboardInterrupt
        return _result(state)
    monkeypatch.setattr(cli, "_calculate_classical", interrupted)
    assert cli.main(["run", str(planned), "--output", str(output)]) == cli.EXIT_CANCELLED
    saved = load_experiment(output)[2]
    assert saved.cancelled and [row.point_index for row in saved.point_results] == [0]
    def calculate(state, **kwargs):
        calls.append(_value(state))
        return _result(state)
    monkeypatch.setattr(cli, "_calculate_classical", calculate)
    assert cli.main(["resume", str(output)]) == 0
    assert calls == [20., 40., 40., 60.]
    complete = load_experiment(output)[2]
    assert not complete.cancelled and complete.completed_points == 3
    assert complete.point_results[0].request_signature == saved.point_results[0].request_signature


def test_complete_resume_preserves_measured_and_derived_results_without_calculating(planned, tmp_path, monkeypatch):
    recipe, _sweep, _result_record, _rules = load_experiment(planned)
    recipe_path = tmp_path / "complete-source.json"
    save_recipe(recipe_path, recipe)
    tolerances = tmp_path / "tolerances.json"
    derived_key = f"sensitivity_abs/sample_illumination_diameter_95_um/{AXIS}"
    tolerances.write_text(json.dumps([
        dict(metric="sample_surviving_current_pa", minimum=9., maximum=11.),
        dict(metric=derived_key, minimum=.9, maximum=1.1),
    ]), encoding="utf-8")
    plan = tmp_path / "complete-plan.temexp"
    assert cli.main(["plan", "--recipe", str(recipe_path), "--axes", str(_axis_file(tmp_path)),
                     "--tolerances", str(tolerances), "--output", str(plan)]) == 0
    run = tmp_path / "complete-run.temexp"
    monkeypatch.setattr(cli, "_calculate_classical", lambda state, **kwargs: _result(state))
    assert cli.main(["run", str(plan), "--output", str(run)]) == 0
    before = load_experiment(run)[2]
    assert before.completed_points == 3
    assert derived_key in before.point_results[0].metrics
    assert all(value.passed for row in before.tolerances for value in row.results)
    calls = []
    def unexpected_calculation(state, **kwargs):
        calls.append(state)
        raise AssertionError("A completed resume must not calculate any point")
    monkeypatch.setattr(cli, "_calculate_classical", unexpected_calculation)
    assert cli.main(["resume", str(run)]) == 0
    after = load_experiment(run)[2]
    assert calls == []
    assert after.point_results == before.point_results
    assert after.tolerances == before.tolerances
    assert after.metric_definitions == before.metric_definitions
    assert after.execution_identity == before.execution_identity
    assert after.sensitivities == before.sensitivities
    assert not any(row.metric.startswith("sensitivity_abs/") for row in after.sensitivities)


def test_timeout_discards_inflight_point_and_preserves_earlier_checkpoint(planned, tmp_path, monkeypatch):
    output = tmp_path / "run.temexp"
    clock = [0.]
    monkeypatch.setattr(cli, "monotonic", lambda: clock[0])
    def calculate(state, *, progress_callback, **kwargs):
        clock[0] = .25 if _value(state) == 20. else 2.
        progress_callback(1, 1, "Controlled boundary")
        return _result(state)
    monkeypatch.setattr(cli, "_calculate_classical", calculate)
    assert cli.main(["run", str(planned), "--output", str(output), "--timeout-seconds", "1"]) == cli.EXIT_TIMEOUT
    saved = load_experiment(output)[2]
    assert saved.cancelled and [row.point_index for row in saved.point_results] == [0]


def test_resume_rejects_an_external_rewrite_during_preflight(planned, tmp_path, monkeypatch):
    from temsim import design_sweep_execution
    output = tmp_path / "run.temexp"
    monkeypatch.setattr(cli, "_calculate_classical", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(["run", str(planned), "--output", str(output)]) == cli.EXIT_CANCELLED
    execute = design_sweep_execution.execute_parameter_sweep
    rewritten = []
    def preflight_then_external_write(*args, **kwargs):
        result = execute(*args, **kwargs)
        if not rewritten:
            recipe, sweep, external, rules = load_experiment(output)
            external = replace(external, analysis_notes=("External writer changed this record during preflight",))
            save_experiment(output, recipe, sweep, external, rules)
            rewritten.append(output.read_bytes())
        return result
    monkeypatch.setattr(design_sweep_execution, "execute_parameter_sweep", preflight_then_external_write)
    monkeypatch.setattr(cli, "_calculate_classical", lambda state, **kwargs: _result(state))
    assert cli.main(["resume", str(output)]) == cli.EXIT_INVALID
    assert output.read_bytes() == rewritten[0]


@pytest.mark.parametrize("changed", ["runtime", "implementation", "tolerances", "checksum", "observer"])
def test_resume_rejects_changed_identity_without_overwriting_record(planned, tmp_path, monkeypatch, changed):
    output = tmp_path / "run.temexp"
    monkeypatch.setattr(cli, "_calculate_classical", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main(["run", str(planned), "--output", str(output)]) == cli.EXIT_CANCELLED
    if changed == "runtime":
        monkeypatch.setattr("temsim.input_io.runtime_identity", lambda: {"changed": True})
    elif changed == "implementation":
        monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "b" * 64)
    elif changed == "tolerances":
        recipe, sweep, result, _ = load_experiment(output)
        save_experiment(output, recipe, sweep, result,
                        (ToleranceRule("sample_surviving_current_pa", minimum=1.),))
    elif changed == "observer":
        monkeypatch.setattr(cli, "_observer_options", lambda: dict(
            point_observer=_observer, point_observer_identity="changed-observer-v2"))
    else:
        output.write_text(output.read_text(encoding="utf-8").replace("Controlled CLI test", "tampered"), encoding="utf-8")
    before = output.read_bytes()
    assert cli.main(["resume", str(output)]) == cli.EXIT_INVALID
    assert output.read_bytes() == before


@pytest.mark.parametrize("format_name", ["csv", "json", "report"])
def test_export_uses_retained_points_and_never_overwrites_existing_files(planned, tmp_path, monkeypatch, format_name):
    run = tmp_path / "run.temexp"
    monkeypatch.setattr(cli, "_calculate_classical", lambda state, **kwargs: _result(state))
    assert cli.main(["run", str(planned), "--output", str(run)]) == 0
    monkeypatch.setattr(cli, "_calculate_classical", lambda *_args, **_kwargs: pytest.fail("Export must not calculate"))
    output = tmp_path / f"export.{format_name}"
    assert cli.main(["export", str(run), "--format", format_name, "--output", str(output)]) == 0
    before = output.read_bytes()
    assert before
    if format_name == "json":
        report = json.loads(before)
        assert len({row["case_id"] for row in report["runs"]}) == 3
        assert all(row["phase"] == "single" for row in report["runs"])
    assert cli.main(["export", str(run), "--format", format_name, "--output", str(output)]) == cli.EXIT_INVALID
    assert output.read_bytes() == before


@pytest.mark.parametrize("failure", ["metrics", "observer_type", "observer_mutation"])
def test_point_failures_are_retained_and_checkpointed(source, failure):
    sweep = plan_parameter_sweep(source, (SweepAxis(AXIS, (20., 40.)),))
    checkpoints = []
    def calculate(state, **kwargs):
        result = _result(state)
        if failure == "metrics" and _value(state) == 20.:
            result.simulation.metrics = object()
        return result
    def observe(state, result):
        if _value(state) == 20.:
            if failure == "observer_type":
                return []
            if failure == "observer_mutation":
                state.objective_lens.percent += 1.
        return _observer(state, result)
    result = execute_parameter_sweep(source, sweep, calculator=calculate,
        point_observer=observe, point_observer_identity="controlled-v1",
        point_completed_callback=checkpoints.append)
    assert [row.status for row in result.point_results] == ["FAILED", "COMPLETE"]
    assert [len(row.point_results) for row in checkpoints] == [1, 2]
    assert not checkpoints[0].point_results[0].metrics


def test_checkpoint_write_failure_is_fatal_and_not_reclassified_as_a_point_failure(source):
    sweep = plan_parameter_sweep(source, (SweepAxis(AXIS, (20., 40.)),))
    calls = []
    def calculate(state, **kwargs):
        calls.append(_value(state))
        return _result(state)
    def failed_write(partial):
        assert partial.point_results[0].status == "COMPLETE"
        raise OSError("Controlled checkpoint write failure")
    with pytest.raises(OSError, match="checkpoint write"):
        execute_parameter_sweep(source, sweep, calculator=calculate,
                                point_completed_callback=failed_write)
    assert calls == [20.]


def test_observer_requires_explicit_identity(source):
    sweep = plan_parameter_sweep(source, (SweepAxis(AXIS, (20.,)),))
    with pytest.raises(ValueError, match="identity"):
        execute_parameter_sweep(source, sweep, point_observer=_observer)
