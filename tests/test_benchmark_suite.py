"""Bounded benchmark orchestration checks, without substituting physical runs."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from threading import Event

import pytest

from temsim import benchmark_suite as suite


@pytest.mark.parametrize("changes", [
    {"rays": True}, {"rays": 0}, {"warm_repeats": 2}, {"reuse_repeats": 0},
    {"threads": 0}, {"step_mm": float("nan")}, {"step_mm": True},
    {"budget_seconds": 0}, {"cleanup_grace_seconds": float("inf")},
    {"cases": ()}, {"cases": ("nanoprobe", "nanoprobe")}, {"cases": ("unknown",)},
    {"profile": 1},
])
def test_configuration_rejects_ambiguous_or_unbounded_requests(changes):
    with pytest.raises(ValueError):
        replace(suite.BenchmarkConfig(), **changes).validate()


def test_configuration_allows_complete_production_population_and_required_repeats():
    value = suite.BenchmarkConfig(rays=3000, step_mm=.05, threads=4).validate()
    assert value.warm_repeats == value.reuse_repeats == 3


def test_atomic_json_rejects_nonfinite_without_damaging_previous_receipt(tmp_path):
    path = tmp_path / "case.json"
    suite.atomic_json(path, {"status": "RUNNING"})
    original = path.read_bytes()
    with pytest.raises(ValueError):
        suite.atomic_json(path, {"seconds": float("nan")})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    suite.atomic_json(path, {"status": "FAILED", "path": Path("evidence")})
    assert json.loads(path.read_text())["status"] == "FAILED"


def test_failure_is_published_with_elapsed_time_without_acceptance(tmp_path):
    recorder = suite.CaseRecorder(tmp_path, "nanoprobe", suite.BenchmarkConfig(), Event())
    def failed():
        raise RuntimeError("Controlled orchestration failure")
    with pytest.raises(RuntimeError, match="orchestration"):
        recorder.measure("cold", 0, failed, lambda value, row: pytest.fail("Do not inspect a failed calculation"))
    saved = json.loads((tmp_path / "case.json").read_text())
    row, = saved["runs"]
    assert row["execution_status"] == "FAILED"
    assert row["observation"] is None
    assert row["elapsed_seconds"] >= 0
    assert row["memory"]["peak_vram_bytes"] is None
    assert row["memory"]["rss_scope"] == "benchmark_worker_process_only"
    assert row["memory"]["rss_includes_descendants"] is False
    assert "active_operation" not in saved


def test_observation_time_is_not_in_calculation_time(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(suite, "perf_counter", lambda: clock[0])
    recorder = suite.CaseRecorder(tmp_path, "nanoprobe", suite.BenchmarkConfig(), Event())
    def operation():
        clock[0] += 2.
        return "result"
    def observe(value, row):
        assert value == "result"
        clock[0] += 7.
        row["observation"] = {"scope": "Orchestration-only fixture"}
    recorder.measure("cold", 0, operation, observe)
    row, = recorder.document["runs"]
    assert row["elapsed_seconds"] == 2.
    assert row["observation_seconds"] == 7.
    assert row["execution_status"] == "COMPLETE"


def test_budget_prevents_starting_another_operation(tmp_path):
    cancelled = Event()
    recorder = suite.CaseRecorder(tmp_path, "nanoprobe", suite.BenchmarkConfig(), cancelled)
    cancelled.set()
    with pytest.raises(TimeoutError):
        recorder.auxiliary("forbidden", lambda: pytest.fail("Cancelled work ran"))
    with pytest.raises(TimeoutError):
        recorder.measure("warm", 0, lambda: pytest.fail("Cancelled work ran"), lambda *_: None)


def test_supervisor_timeout_records_failure_and_owns_new_directory(tmp_path, monkeypatch):
    from temsim import validation_process
    calls = []
    def timeout(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, 124)
    monkeypatch.setattr(validation_process, "run_bounded", timeout)
    monkeypatch.setattr(suite, "_summaries", lambda *_: None)
    output = tmp_path / "run"
    config = suite.BenchmarkConfig(cases=("nanoprobe",), budget_seconds=2., cleanup_grace_seconds=1.)
    assert suite.run_suite(config, output) == 1
    saved = json.loads((output / "suite.json").read_text())
    assert saved["status"] == "FAILED"
    assert saved["cases"][0]["status"] == "TIMED_OUT"
    command, options = calls[0]
    assert options["timeout"] == 3.
    assert command[1:3] == ["-m", "temsim.benchmark_suite"]
    assert options["env"]["QT_QPA_PLATFORM"] == "offscreen"
    before = (output / "suite.json").read_bytes()
    with pytest.raises(FileExistsError):
        suite.run_suite(config, output)
    assert (output / "suite.json").read_bytes() == before
    assert len(calls) == 1


def test_cleanup_failure_blocks_later_cases_and_is_durable(tmp_path, monkeypatch):
    from temsim import validation_process
    calls = []
    def failed(command, **options):
        calls.append(command)
        raise validation_process.ValidationCleanupError(123, "Controlled test cleanup failure")
    monkeypatch.setattr(validation_process, "run_bounded", failed)
    output = tmp_path / "blocked"
    with pytest.raises(validation_process.ValidationCleanupError):
        suite.run_suite(suite.BenchmarkConfig(cases=("nanoprobe", "scan_calibration")), output)
    saved = json.loads((output / "suite.json").read_text())
    assert saved["status"] == "CLEANUP_UNCONFIRMED"
    case = json.loads((output / "nanoprobe" / "case.json").read_text())
    assert case["status"] == "CLEANUP_UNCONFIRMED"
    assert len(calls) == 1
    assert not (output / "scan_calibration").exists()


@pytest.mark.parametrize("error", [OSError("Controlled launch failure"), KeyboardInterrupt()])
def test_interrupted_or_failed_launch_closes_current_case_receipt(tmp_path, monkeypatch, error):
    from temsim import validation_process
    def fail(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(validation_process, "run_bounded", fail)
    output = tmp_path / "failed-launch"
    config = suite.BenchmarkConfig(cases=("nanoprobe",))
    if isinstance(error, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            suite.run_suite(config, output)
    else:
        assert suite.run_suite(config, output) == 1
    case = json.loads((output / "nanoprobe" / "case.json").read_text())
    assert case["status"] == "FAILED"
    assert "finished_utc" in case
    saved = json.loads((output / "suite.json").read_text())
    assert saved["cases"] == [case]


def test_coherent_is_explicitly_not_run_without_loading_a_solver(tmp_path, monkeypatch):
    from temsim import validation_process
    monkeypatch.setattr(validation_process, "run_bounded", lambda *_args, **_kwargs: pytest.fail("Declared coherent case executed"))
    monkeypatch.setattr(suite, "_summaries", lambda *_: None)
    assert suite.run_suite(suite.BenchmarkConfig(cases=("coherent",)), tmp_path / "declared") == 0
    record = json.loads((tmp_path / "declared" / "coherent" / "case.json").read_text())
    assert record["status"] == "NOT_RUN"
    assert record["runs"] == []
    assert "CUDA" in record["reason"]


def test_profile_pass_is_separate_and_has_both_rankings(tmp_path):
    recorder = suite.CaseRecorder(tmp_path, "nanoprobe", suite.BenchmarkConfig(profile=True), Event())
    recorder.profile(lambda: sum(range(20)))
    assert recorder.document["runs"] == []
    row, = recorder.document["auxiliary_operations"]
    assert row["label"] == "separate_profile_pass_not_in_benchmark_timings"
    text = (tmp_path / "profile.txt").read_text()
    assert "Top 30 by cumulative time" in text and "Top 30 by own time" in text
    assert "child-process execution is not profiled" in text


def test_default_fixture_preserves_preset_optics_and_declares_static_source():
    # Constructs production inputs only. No field solve, gun or column execution.
    state = suite.configured_state(suite.BenchmarkConfig())
    from temsim.simulation_modes import is_unified
    assert is_unified(state)
    assert state.electron_gun.emitter.ray_count == 49
    assert state.step_mm == .05
    assert state.history_step_mm == .5
    assert state._tuning_quality == "High accuracy"
    assert state.acceleration_backend == "Numba CPU"
    assert not state.ac_deflector.scan_enabled and not state.descan_deflector.scan_enabled
    assert state.mini_condenser.percent == 43.88646617222637
    assert state.electron_gun.emitter.surface_model is None
    assert state.electron_gun.emitter.coherence is None
    from temsim.particle_benchmark import continuation_planes
    assert continuation_planes(state) == (460., 470.)


def test_fixture_history_spacing_tracks_larger_integration_steps():
    state = suite.configured_state(suite.BenchmarkConfig(step_mm=.75))
    assert state.step_mm == .75
    assert state.history_step_mm == .75


def _continuation_receipt_fixture():
    # Only tests receipt admission; this fixture never substitutes a solver.
    from types import SimpleNamespace
    metrics = {"section_gun_reused": True, "section_reused_prefix": True,
               "section_resume_z_mm": 460.}
    plan = SimpleNamespace(z_mm=(450., 455., 460., 465., 470.))
    segment = SimpleNamespace(name="incident", plan=plan)
    result = SimpleNamespace(simulation=SimpleNamespace(metrics=metrics,
        section_checkpoint=SimpleNamespace(segments=(segment,))))
    row = {"column_calls": [{"start_index": 2, "requested_intervals": 2,
                            "start_z_mm": 460., "stop_z_mm": 470.}],
           "operation_counts": {"requested_column_intervals": 2}}
    return result, row


def test_continuation_receipt_requires_positive_saved_prefix_and_reduced_execution():
    result, row = _continuation_receipt_fixture()
    evidence = suite._require_continuation_execution(result, row, 460.)
    assert evidence == {"resume_z_mm": 460., "start_index": 2, "total_plan_intervals": 4,
                        "executed_intervals": 2, "retained_prefix_intervals": 2}


@pytest.mark.parametrize("failure", ["gun", "prefix", "entrance_resume", "entrance_execution",
                                     "zero_execution", "missing_call", "wrong_count", "wrong_plane"])
def test_continuation_receipt_rejects_unproven_or_entrance_only_reuse(failure):
    result, row = _continuation_receipt_fixture()
    metrics, call = result.simulation.metrics, row["column_calls"][0]
    if failure == "gun":
        metrics["section_gun_reused"] = False
    elif failure == "prefix":
        metrics["section_reused_prefix"] = False
    elif failure == "entrance_resume":
        metrics["section_resume_z_mm"] = 450.
    elif failure == "entrance_execution":
        call.update(start_index=0, requested_intervals=4, start_z_mm=450.)
        row["operation_counts"]["requested_column_intervals"] = 4
    elif failure == "zero_execution":
        call.update(start_index=4, requested_intervals=0, start_z_mm=470.)
        row["operation_counts"]["requested_column_intervals"] = 0
    elif failure == "missing_call":
        row["column_calls"] = []
    elif failure == "wrong_count":
        row["operation_counts"]["requested_column_intervals"] = 3
    else:
        call.update(start_index=1, requested_intervals=3)
        row["operation_counts"]["requested_column_intervals"] = 3
    with pytest.raises(AssertionError):
        suite._require_continuation_execution(result, row, 460.)


def test_readable_summary_includes_calibration_failure_and_unrun_cases(tmp_path):
    cases = [
        {"case_id": "nanoprobe", "status": "FAILED", "error": "No completed calculation", "runs": []},
        {"case_id": "scan_calibration", "status": "COMPLETE", "scope": "Local response only", "runs": [
            {"phase": "cold", "execution_status": "COMPLETE", "elapsed_seconds": 2.,
             "calibration_observation": {"receipt": {"response_cache": {"source": "computed"}}}}]},
        {"case_id": "coherent", "status": "NOT_RUN", "reason": "Explicit CUDA fixture is required", "runs": []},
    ]
    report = suite._summaries(tmp_path, cases)
    assert report["repeatability"]["status"] != "PASS"
    text = (tmp_path / "accuracy-summary.txt").read_text()
    assert "nanoprobe: FAILED" in text and "No completed calculation" in text
    assert "Scan calibration: COMPLETE" in text and "cold median: 2 s" in text
    assert "Response sources: computed" in text
    assert "coherent: NOT_RUN" in text and "Explicit CUDA fixture is required" in text
    data = json.loads((tmp_path / "accuracy-summary.json").read_text())
    assert data["suite_execution"][0]["status"] == "FAILED"
    assert data["calibration_summary"][0]["runs"] == 1


@pytest.mark.parametrize("repeatability, expected_exit", [("PASS", 0), ("FAIL", 1), ("NOT_EVALUATED", 1), (None, 1)])
def test_successful_worker_cannot_hide_failed_or_missing_joint_acceptance(tmp_path, monkeypatch, repeatability, expected_exit):
    # Receipt-only supervisor fixture; no physical solver is executed here.
    from temsim import validation_process
    def complete(command, **options):
        directory = Path(command[command.index("--worker-directory") + 1])
        suite.atomic_json(directory / "case.json", _inventory_document("continuation"))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(validation_process, "run_bounded", complete)
    report = {"repeatability": {"status": repeatability}, "physical_qualification": "UNRESOLVED"} if repeatability else None
    monkeypatch.setattr(suite, "_summaries", lambda *_: report)
    output = tmp_path / "acceptance"
    assert suite.run_suite(suite.BenchmarkConfig(cases=("continuation",)), output) == expected_exit
    saved = json.loads((output / "suite.json").read_text())
    assert saved["status"] == "COMPLETE"
    assert saved["particle_repeatability"] == (repeatability or "NOT_EVALUATED")
    assert saved["acceptance_status"] == ("PASS" if expected_exit == 0 else "FAIL")


def test_declaration_only_suite_does_not_claim_particle_acceptance(tmp_path):
    output = tmp_path / "declaration"
    assert suite.run_suite(suite.BenchmarkConfig(cases=("coherent",)), output) == 0
    saved = json.loads((output / "suite.json").read_text())
    assert saved["status"] == "COMPLETE"
    assert saved["acceptance_status"] == saved["particle_repeatability"] == "NOT_APPLICABLE"


def _inventory_document(case_id):
    return {"schema": suite.SCHEMA, "case_id": case_id, "status": "COMPLETE", "runs": [
        {"case_id": case_id, "phase": phase, "repeat": repeat, "execution_status": "COMPLETE",
         "elapsed_seconds": .1}
        for phase, count in (("cold", 1), ("warm", 3), ("reuse", 3)) for repeat in range(count)]}


@pytest.mark.parametrize("failure", ["empty", "missing_warm", "duplicate", "wrong_case", "failed_run",
                                     "nan_time", "negative_time", "boolean_time", "boolean_repeat"])
def test_case_inventory_rejects_missing_ambiguous_or_failed_runs(failure):
    document = _inventory_document("nanoprobe")
    if failure == "empty":
        document["runs"] = []
    elif failure == "missing_warm":
        document["runs"].pop(1)
    elif failure == "duplicate":
        document["runs"].append(document["runs"][0].copy())
    elif failure == "wrong_case":
        document["runs"][0]["case_id"] = "continuation"
    elif failure == "failed_run":
        document["runs"][0]["execution_status"] = "FAILED"
    elif failure == "boolean_repeat":
        document["runs"][0]["repeat"] = False
    else:
        document["runs"][0]["elapsed_seconds"] = {
            "nan_time": float("nan"), "negative_time": -.1, "boolean_time": True}[failure]
    with pytest.raises(ValueError):
        suite._require_case_run_inventory(document, suite.BenchmarkConfig(), "nanoprobe")


@pytest.mark.parametrize("missing", ["empty_case", "one_warm"])
def test_one_complete_case_cannot_hide_another_incomplete_run_inventory(tmp_path, monkeypatch, missing):
    from temsim import validation_process
    def complete(command, **options):
        directory = Path(command[command.index("--worker-directory") + 1])
        case_id = command[command.index("--worker-case") + 1]
        document = _inventory_document(case_id)
        if case_id == "continuation":
            if missing == "empty_case":
                document["runs"] = []
            else:
                document["runs"].pop(1)
        suite.atomic_json(directory / "case.json", document)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(validation_process, "run_bounded", complete)
    monkeypatch.setattr(suite, "_summaries", lambda *_: {"repeatability": {"status": "PASS"}})
    output = tmp_path / "partial-inventory"
    assert suite.run_suite(suite.BenchmarkConfig(cases=("nanoprobe", "continuation")), output) == 1
    saved = json.loads((output / "suite.json").read_text())
    assert saved["status"] == "FAILED"
    assert saved["acceptance_status"] == "FAIL"
    assert saved["cases"][0]["run_inventory_status"] == "VALID"
    assert saved["cases"][1]["run_inventory_status"] == "INVALID"
