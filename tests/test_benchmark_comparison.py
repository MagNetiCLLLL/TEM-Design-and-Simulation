"""Synthetic receipt checks for the comparison CLI; no physical runs are faked."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest

from temsim.benchmark_suite import BenchmarkConfig
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import InstrumentSnapshot, physical_identity_graph
from temsim.performance_accuracy import METRICS, OBSERVATION_SCHEMA, OBSERVER_IDENTITY


_spec = importlib.util.spec_from_file_location("benchmark_comparison_script",
    Path(__file__).resolve().parents[1] / "scripts" / "compare_benchmark_suites.py")
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _suite(root, *, implementation="a"*64, elapsed=2., cases=("nanoprobe", "scan_calibration", "coherent")):
    configuration = json.loads(json.dumps(asdict(BenchmarkConfig(cases=cases, threads=8))))
    graph = {"root": {"ref": 0}, "nodes": [{"type": "temsim.optics.model:State",
        "attributes": {"step_mm": .05, "history_step_mm": .5, "physical_control": 3.}}]}
    snapshot = InstrumentSnapshot(graph, (), implementation)
    inputs = json_digest({"graph": physical_identity_graph(snapshot.graph), "external_inputs": snapshot.external_inputs})
    rows = []
    for case_id in cases:
        if case_id == "coherent":
            rows.append({"schema": comparison.SUITE_SCHEMA, "case_id": case_id, "status": "NOT_RUN",
                         "runs": [], "reason": "No coherent calculation requested"})
            continue
        case = {"schema": comparison.SUITE_SCHEMA, "case_id": case_id, "status": "COMPLETE",
                "configuration": configuration, "supervisor": {"returncode": 0},
                "implementation_identity": implementation, "implementation_identity_after": implementation,
                "input_identity": snapshot.digest, "cpu_resources": {"available_cpus": 32, "blas_threads": 1,
                    "numba_threads": 8, "numerical_jobs": 1, "numerical_thread_budget": 8}, "runs": []}
        for phase, count in (("cold", 1), ("warm", 3), ("reuse", 3)):
            for repeat in range(count):
                row = {"case_id": case_id, "phase": phase, "repeat": repeat,
                       "execution_status": "COMPLETE", "elapsed_seconds": elapsed}
                if case_id in comparison.PARTICLE_CASES:
                    row["observation"] = {"schema": OBSERVATION_SCHEMA,
                        "execution": {"status": "COMPLETE"}, "identity": {"status": "CAPTURED",
                            "observer": OBSERVER_IDENTITY, "implementation": implementation,
                            "inputs_digest": inputs, "physical_inputs_digest": "physical",
                            "numerical_inputs_digest": "numerical"},
                        "population": {"status": "VALID", "fingerprints": {"phase": "retained-array"},
                            "source_fingerprints": {"source": "original-source"},
                            "checkpoint_fingerprints": {"checkpoint": "saved-prefix"}},
                        "observables": {"status": "MEASURED", "plane_kind": "specimen_upper_surface",
                            "plane_z_mm": 100., **{name: 1. for name in METRICS}},
                        "qualification": {"physical": {"status": "UNRESOLVED"}}}
                else:
                    source = "computed" if phase != "reuse" else "disk" if repeat % 2 else "memory"
                    matrices = {name: {"dtype": "<f8", "shape": shape, "sha256": "c"*64,
                        "nbytes": 8*comparison.math.prod(shape)} for name, shape in comparison.MATRICES.items()}
                    row["calibration_observation"] = {"schema": "temsim-calibration-benchmark-observation-v1",
                        "input_identity": snapshot.digest, "implementation_identity": implementation,
                        "matrices": matrices, "sample_position_relative_residual": .00001,
                        "receipt": {"response_cache": {"source": source}, "execution": {
                            "mode": "independent_processes", "workers": 4, "trace_count": 17, "wall_seconds": elapsed}}}
                case["runs"].append(row)
        rows.append(case)
        _write(root / case_id / "inputs.json", snapshot.to_dict())
    acceptance = "PASS" if any(case in comparison.PARTICLE_CASES for case in cases) else "NOT_APPLICABLE"
    suite = {"schema": comparison.SUITE_SCHEMA, "status": "COMPLETE", "configuration": configuration, "cases": rows,
             "acceptance_status": acceptance, "particle_repeatability": acceptance}
    _save_suite(root, suite)
    return suite


def _save_suite(root, suite):
    _write(root / "suite.json", suite)
    _write(root / "configuration.json", suite["configuration"])
    for case in suite["cases"]:
        _write(root / case["case_id"] / "case.json", case)


def _pair(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    first = _suite(before)
    second = _suite(after, implementation="b"*64, elapsed=1.)
    return before, after, first, second


def test_same_inputs_allow_implementation_change_and_report_both_timing_families(tmp_path):
    before, after, *_ = _pair(tmp_path)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "PASS", report["failures"]
    assert report["particle_comparison"]["status"] == "PASS"
    assert report["scan_comparison"]["status"] == "PASS"
    assert len(report["performance"]) == 6
    assert all(row["speedup"] == 2. for row in report["performance"])
    assert report["declarations"][0]["comparison"] == "NOT_EVALUATED"
    assert report["physical_qualification"] == "UNRESOLVED"


@pytest.mark.parametrize("failure", ["suite_failed", "case_failed", "run_failed", "missing_repeat",
    "duplicate_repeat", "cpu_budget", "configuration", "matrix", "residual", "cache_source",
    "particle_array", "observer", "input_binding", "implementation_changed", "missing_elapsed"])
def test_missing_changed_or_failed_evidence_never_passes(tmp_path, failure):
    before, after, _, suite = _pair(tmp_path)
    particle, scan = suite["cases"][:2]
    if failure == "suite_failed":
        suite["status"] = "FAILED"
    elif failure == "case_failed":
        scan["status"] = "FAILED"
    elif failure == "run_failed":
        particle["runs"][1]["execution_status"] = "FAILED"
    elif failure == "missing_repeat":
        scan["runs"].pop()
    elif failure == "duplicate_repeat":
        scan["runs"][-1]["repeat"] = 1
    elif failure == "cpu_budget":
        particle["cpu_resources"]["numba_threads"] = 4
    elif failure == "configuration":
        suite["configuration"]["step_mm"] = .1
    elif failure == "matrix":
        scan["runs"][-1]["calibration_observation"]["matrices"]["ac_command_mrad"]["sha256"] = "d"*64
    elif failure == "residual":
        scan["runs"][-1]["calibration_observation"]["sample_position_relative_residual"] = .00002
    elif failure == "cache_source":
        scan["runs"][-1]["calibration_observation"]["receipt"]["response_cache"]["source"] = "disk"
    elif failure == "particle_array":
        particle["runs"][-1]["observation"]["population"]["fingerprints"]["phase"] = "changed-array"
    elif failure == "observer":
        particle["runs"][-1]["observation"]["identity"]["observer"] = "other-observer"
    elif failure == "input_binding":
        particle["runs"][-1]["observation"]["identity"]["inputs_digest"] = "different-inputs"
    elif failure == "implementation_changed":
        particle["implementation_identity_after"] = "d"*64
    else:
        particle["runs"][0].pop("elapsed_seconds")
    _save_suite(after, suite)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "FAIL"
    assert report["failures"]


def test_scan_snapshot_graph_change_cannot_be_hidden_by_equal_matrices(tmp_path):
    before, after, _, suite = _pair(tmp_path)
    path = after / "scan_calibration" / "inputs.json"
    data = json.loads(path.read_text())
    graph = deepcopy(data["graph"])
    graph["nodes"][0]["attributes"]["physical_control"] = 4.
    changed = InstrumentSnapshot(graph, (), data["implementation"])
    _write(path, changed.to_dict())
    case = suite["cases"][1]
    case["input_identity"] = changed.digest
    for row in case["runs"]:
        row["calibration_observation"]["input_identity"] = changed.digest
    _save_suite(after, suite)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "FAIL"
    assert "graph or external inputs changed" in report["failures"][0]


def test_scan_only_comparison_and_coherent_only_are_distinct(tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    _suite(before, cases=("scan_calibration",))
    _suite(after, cases=("scan_calibration",), implementation="b"*64)
    assert comparison.compare_suites(before, after)["status"] == "PASS"
    _suite(before, cases=("coherent",))
    _suite(after, cases=("coherent",))
    report = comparison.compare_suites(before, after)
    assert report["status"] == "NOT_EVALUATED"
    assert report["performance"] == []


@pytest.mark.parametrize("text", ['{"status": NaN}', '{"status": Infinity}', '{"time": 1e999}',
                                  '{"status": "COMPLETE", "status": "FAILED"}'])
def test_strict_json_rejects_nonfinite_numbers_and_duplicate_keys(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        comparison._read(path)


def test_cli_writes_strict_artifacts_and_refuses_existing_directory(tmp_path):
    before, after, *_ = _pair(tmp_path)
    output = tmp_path / "comparison"
    args = [str(before), str(after), "--output", str(output)]
    assert comparison.main(args) == 0
    saved = (output / "comparison.json").read_bytes()
    assert comparison._read(output / "comparison.json")["status"] == "PASS"
    text = (output / "comparison.txt").read_text()
    assert "scan_calibration / warm" in text and "nanoprobe / warm" in text
    assert "coherent: NOT_RUN; comparison NOT_EVALUATED" in text
    with pytest.raises(SystemExit) as caught:
        comparison.main(args)
    assert caught.value.code == 2
    assert (output / "comparison.json").read_bytes() == saved


def test_cli_missing_inputs_publishes_failed_receipt(tmp_path):
    output = tmp_path / "comparison"
    assert comparison.main([str(tmp_path / "missing-before"), str(tmp_path / "missing-after"),
                            "--output", str(output)]) == 1
    report = comparison._read(output / "comparison.json")
    assert report["status"] == "FAIL" and report["failures"]


@pytest.mark.parametrize("change", ["missing", "serial", "workers", "trace_count"])
def test_actual_scan_scheduling_must_match_even_with_same_parent_cpu_budget(tmp_path, change):
    before, after, _, candidate = _pair(tmp_path)
    for row in candidate["cases"][1]["runs"]:
        receipt = row["calibration_observation"]["receipt"]
        if change == "missing":
            del receipt["execution"]
        elif change == "serial":
            receipt["execution"].update(mode="serial", workers=1)
        elif change == "workers":
            receipt["execution"]["workers"] = 2
        else:
            receipt["execution"]["trace_count"] = 19
    _save_suite(after, candidate)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "FAIL"
    assert report["failures"]


def _change_observer(suite, version, plane):
    for row in suite["cases"][0]["runs"]:
        row["observation"]["identity"]["observer"] = version
        row["observation"]["observables"]["plane_kind"] = plane


@pytest.mark.parametrize("plane", ["specimen_upper_surface", "section_terminal"])
def test_additive_v2_to_v3_observer_migration_on_unchanged_planes_is_explicit(tmp_path, plane):
    before, after, baseline, candidate = _pair(tmp_path)
    _change_observer(baseline, comparison.OBSERVER_V2, plane)
    _change_observer(candidate, comparison.OBSERVER_V3, plane)
    _save_suite(before, baseline)
    _save_suite(after, candidate)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "PASS", report["failures"]
    assert report["observer_compatibility"]["nanoprobe"]["contract"] == "v2_to_v3_unchanged_plane"


def test_new_reference_plane_is_supported_only_with_matching_v3_observers(tmp_path):
    before, after, baseline, candidate = _pair(tmp_path)
    _change_observer(baseline, comparison.OBSERVER_V3, "sample_reference_plane")
    _change_observer(candidate, comparison.OBSERVER_V3, "sample_reference_plane")
    _save_suite(before, baseline)
    _save_suite(after, candidate)
    report = comparison.compare_suites(before, after)
    assert report["status"] == "PASS", report["failures"]
    assert report["observer_compatibility"]["nanoprobe"]["contract"] == "same_version"


@pytest.mark.parametrize("versions, plane", [
    ((comparison.OBSERVER_V2, comparison.OBSERVER_V3), "sample_reference_plane"),
    ((comparison.OBSERVER_V2, comparison.OBSERVER_V2), "sample_reference_plane"),
    ((comparison.OBSERVER_V3, comparison.OBSERVER_V2), "section_terminal"),
    (("unknown-observer", "unknown-observer"), "section_terminal"),
    ((comparison.OBSERVER_V2, "unknown-observer"), "section_terminal"),
])
def test_unknown_or_unsupported_observer_contracts_remain_rejected(tmp_path, versions, plane):
    before, after, baseline, candidate = _pair(tmp_path)
    _change_observer(baseline, versions[0], plane)
    _change_observer(candidate, versions[1], plane)
    _save_suite(before, baseline)
    _save_suite(after, candidate)
    assert comparison.compare_suites(before, after)["status"] == "FAIL"


@pytest.mark.parametrize("change", ["array", "metric", "plane_position", "inputs"])
def test_observer_compatibility_does_not_relax_strict_scientific_equality(tmp_path, change):
    before, after, baseline, candidate = _pair(tmp_path)
    _change_observer(baseline, comparison.OBSERVER_V2, "section_terminal")
    _change_observer(candidate, comparison.OBSERVER_V3, "section_terminal")
    for row in candidate["cases"][0]["runs"]:
        observation = row["observation"]
        if change == "array":
            observation["population"]["fingerprints"]["phase"] = "different-array"
        elif change == "metric":
            observation["observables"]["current_a"] += 1e-10
        elif change == "plane_position":
            observation["observables"]["plane_z_mm"] += .001
        else:
            observation["identity"]["inputs_digest"] = "changed-inputs"
    _save_suite(before, baseline)
    _save_suite(after, candidate)
    assert comparison.compare_suites(before, after)["status"] == "FAIL"
