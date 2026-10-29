"""Compare completed production suites without restoring or executing physics."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

SCHEMA = "temsim-benchmark-suite-comparison-v1"
SUITE_SCHEMA = "temsim-unified-benchmark-suite-v1"
PARTICLE_CASES = {"nanoprobe", "continuation"}
MATRICES = {"ac_command_mrad": [2, 2], "descan_command_mrad": [2, 2],
            "ac_coupling": [2, 2, 2], "descan_coupling": [2, 2, 2]}
CPU_FIELDS = {"available_cpus", "blas_threads", "numba_threads", "numerical_jobs", "numerical_thread_budget"}
OBSERVER_V2 = "temsim-performance-accuracy-observer-v2"
OBSERVER_V3 = "temsim-performance-accuracy-observer-v3"
UNCHANGED_PLANES = {"specimen_upper_surface", "section_terminal"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    _require(math.isfinite(number), f"Nonfinite JSON number: {value}")
    return number


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_object, parse_constant=_invalid_constant, parse_float=_finite_float)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _expected_runs(configuration):
    return {(phase, repeat) for phase, count in (("cold", 1),
        ("warm", configuration["warm_repeats"]), ("reuse", configuration["reuse_repeats"]))
        for repeat in range(count)}


def _load_suite(directory):
    from temsim.benchmark_suite import BenchmarkConfig
    from temsim.immutable_json import json_digest
    from temsim.instrument_snapshot import InstrumentSnapshot, physical_identity_graph
    root = Path(directory).resolve()
    suite = _read(root / "suite.json")
    _require(suite.get("schema") == SUITE_SCHEMA, "Unsupported suite schema")
    _require(suite.get("status") == "COMPLETE", "Suite is not COMPLETE")
    configuration = suite["configuration"]
    BenchmarkConfig(**configuration).validate()
    expected_acceptance = "PASS" if any(case in PARTICLE_CASES for case in configuration["cases"]) else "NOT_APPLICABLE"
    _require(suite.get("acceptance_status") == expected_acceptance
             and suite.get("particle_repeatability") == expected_acceptance,
             "Suite lacks successful applicable repeatability acceptance")
    _require(_canonical(_read(root / "configuration.json")) == _canonical(configuration),
             "Configuration file disagrees with suite receipt")
    cases = {}
    inputs = {}
    for case in suite["cases"]:
        case_id = case["case_id"]
        _require(case_id in configuration["cases"] and case_id not in cases, "Unknown or duplicate case")
        _require(case.get("schema") == SUITE_SCHEMA, f"{case_id}: unsupported case schema")
        _require(_canonical(_read(root / case_id / "case.json")) == _canonical(case),
                 f"{case_id}: suite and case receipts disagree")
        cases[case_id] = case
        if case_id == "coherent":
            _require(case.get("status") == "NOT_RUN" and case.get("runs") == [] and case.get("reason"),
                     "Coherent case must remain an explicit NOT_RUN declaration")
            continue
        _require(case.get("status") == "COMPLETE", f"{case_id}: case is not COMPLETE")
        _require(_canonical(case.get("configuration")) == _canonical(configuration),
                 f"{case_id}: case configuration differs from suite")
        _require(case.get("supervisor", {}).get("returncode") == 0, f"{case_id}: missing successful worker exit")
        implementation = case.get("implementation_identity")
        _require(_digest(implementation) and case.get("implementation_identity_after") == implementation,
                 f"{case_id}: implementation identity is missing or changed during execution")
        cpu = case.get("cpu_resources", {})
        _require(CPU_FIELDS <= set(cpu) and all(type(cpu[key]) is int and cpu[key] > 0 for key in CPU_FIELDS),
                 f"{case_id}: incomplete CPU resource evidence")
        snapshot = InstrumentSnapshot.from_dict(_read(root / case_id / "inputs.json"))
        _require(snapshot.implementation == implementation and snapshot.digest == case.get("input_identity"),
                 f"{case_id}: snapshot is not bound to the executed case")
        normalized = json_digest({"graph": physical_identity_graph(snapshot.graph),
                                  "external_inputs": snapshot.external_inputs})
        inputs[case_id] = {"physical_and_numerical_inputs_digest": normalized,
                          "snapshot_digest": snapshot.digest, "implementation": implementation}
        seen = set()
        for row in case["runs"]:
            key = (row.get("phase"), row.get("repeat"))
            _require(type(key[1]) is int and key not in seen, f"{case_id}: duplicate or invalid phase/repeat")
            seen.add(key)
            _require(row.get("case_id") == case_id and row.get("execution_status") == "COMPLETE",
                     f"{case_id}: incomplete run")
            _require(_number(row.get("elapsed_seconds")), f"{case_id}: missing finite elapsed time")
            if case_id in PARTICLE_CASES:
                identity = row.get("observation", {}).get("identity", {})
                _require(identity.get("inputs_digest") == normalized and identity.get("implementation") == implementation,
                         f"{case_id}: particle observation is not bound to the executed inputs and implementation")
                _require(isinstance(identity.get("observer"), str) and identity["observer"],
                         f"{case_id}: missing observer identity")
        _require(seen == _expected_runs(configuration), f"{case_id}: phase/repeat inventory is incomplete")
    _require(set(cases) == set(configuration["cases"]), "Declared case inventory is incomplete")
    return {"directory": str(root), "configuration": configuration, "cases": cases, "inputs": inputs}


def _scan_evidence(case, identity):
    evidence = {}
    reference = None
    for row in case["runs"]:
        key = (row["phase"], row["repeat"])
        item = row.get("calibration_observation", {})
        _require(item.get("schema") == "temsim-calibration-benchmark-observation-v1", "Missing scan observation")
        _require(item.get("input_identity") == identity["snapshot_digest"]
                 and item.get("implementation_identity") == identity["implementation"],
                 "Scan observation is not bound to its captured inputs")
        matrices = item.get("matrices", {})
        _require(set(matrices) == set(MATRICES), "Incomplete scan matrix inventory")
        for name, shape in MATRICES.items():
            matrix = matrices[name]
            _require(matrix.get("shape") == shape and matrix.get("dtype") == "<f8"
                     and matrix.get("nbytes") == math.prod(shape)*8 and _digest(matrix.get("sha256")),
                     f"Invalid scan matrix fingerprint: {name}")
        residual = item.get("sample_position_relative_residual")
        _require(_number(residual), "Missing finite scan residual")
        source = item.get("receipt", {}).get("response_cache", {}).get("source")
        expected = "computed" if key[0] != "reuse" else "disk" if key[1] % 2 else "memory"
        _require(source == expected, "Scan cache source does not match the measured phase/repeat")
        execution = item.get("receipt", {}).get("execution", {})
        mode, workers, traces = (execution.get(name) for name in ("mode", "workers", "trace_count"))
        _require(mode in {"serial", "independent_processes"} and type(workers) is int
                 and type(traces) is int and 0 < workers <= traces
                 and (workers == 1 if mode == "serial" else workers > 1),
                 "Missing or invalid scan execution scheduling evidence")
        resources = {"mode": mode, "workers": workers, "trace_count": traces}
        values = {"matrices": matrices, "sample_position_relative_residual": residual, "execution": resources}
        if reference is not None:
            _require(_canonical(values) == reference, "Scan matrices, residual or execution resources changed within the suite")
        reference = _canonical(values)
        evidence[key] = {**values, "cache_source": source}
    return evidence


def _observer_compatibility(before, after):
    """Admit only the documented additive v3 change for unchanged v2 planes."""
    versions = []
    planes = []
    for case in (before, after):
        observed = {row["observation"]["identity"]["observer"] for row in case["runs"]}
        _require(len(observed) == 1 and observed <= {OBSERVER_V2, OBSERVER_V3},
                 "Unknown or inconsistent observer version")
        version = next(iter(observed))
        kinds = {row["observation"].get("observables", {}).get("plane_kind") for row in case["runs"]}
        allowed = UNCHANGED_PLANES if version == OBSERVER_V2 else UNCHANGED_PLANES | {"sample_reference_plane"}
        _require(len(kinds) == 1 and kinds <= allowed, "Observer version does not support the recorded plane kind")
        versions.append(version)
        planes.append(kinds)
    _require(planes[0] == planes[1], "Observer comparison requires the same plane kind")
    migration = versions == [OBSERVER_V2, OBSERVER_V3] and planes[0] <= UNCHANGED_PLANES
    _require(versions[0] == versions[1] or migration,
             "Observer versions are incompatible; only v2-to-v3 on unchanged planes is admitted")
    return {"before": versions[0], "after": versions[1], "plane_kind": next(iter(planes[0])),
            "contract": "v2_to_v3_unchanged_plane" if migration else "same_version",
            "scope": "No relaxation of input, array, plane-position or zero-tolerance observable comparison"}


def _timings(before, after, case_id):
    rows = []
    for phase in ("cold", "warm", "reuse"):
        old = statistics.median(row["elapsed_seconds"] for row in before["runs"] if row["phase"] == phase)
        new = statistics.median(row["elapsed_seconds"] for row in after["runs"] if row["phase"] == phase)
        rows.append({"case_id": case_id, "phase": phase, "before_median_seconds": old,
                     "after_median_seconds": new, "speedup": old/new if new > 0 else None})
    return rows


def compare_suites(before_directory, after_directory):
    """Fail closed on absent or mismatched execution evidence; never run physics."""
    from temsim.performance_accuracy import compare_reports, summarize_runs
    result = {"schema": SCHEMA, "status": "FAIL", "failures": [], "particle_comparison": None,
              "scan_comparison": None, "declarations": [], "performance": [],
              "scope": "Exact same-input equivalence and elapsed-time comparison; not convergence or physical qualification",
              "physical_qualification": "UNRESOLVED", "numerical_convergence": "NOT_RUN",
              "memory_scope": "No cross-process peak-memory claim; input suite RSS excludes descendants",
              "before_directory": str(Path(before_directory).resolve()),
              "after_directory": str(Path(after_directory).resolve())}
    try:
        before, after = _load_suite(before_directory), _load_suite(after_directory)
        _require(_canonical(before["configuration"]) == _canonical(after["configuration"]),
                 "Suite configurations or case/phase/repeat budgets differ")
        _require(set(before["cases"]) == set(after["cases"]), "Case inventories differ")
        result["configuration"] = before["configuration"]
        result["inputs_and_implementations"] = {"before": before["inputs"], "after": after["inputs"]}
        result["cpu_resources"] = {}
        result["environments"] = {}
        result["observer_compatibility"] = {}
        before_particles, after_particles = [], []
        for case_id in before["configuration"]["cases"]:
            first, second = before["cases"][case_id], after["cases"][case_id]
            if case_id == "coherent":
                result["declarations"].append({"case_id": case_id, "status": "NOT_RUN",
                    "before_reason": first["reason"], "after_reason": second["reason"],
                    "comparison": "NOT_EVALUATED"})
                continue
            _require(_canonical(first["cpu_resources"]) == _canonical(second["cpu_resources"]),
                     f"{case_id}: CPU resource allocations differ")
            result["cpu_resources"][case_id] = first["cpu_resources"]
            result["environments"][case_id] = {"before": first.get("environment"), "after": second.get("environment")}
            old_id, new_id = before["inputs"][case_id], after["inputs"][case_id]
            _require(old_id["physical_and_numerical_inputs_digest"] == new_id["physical_and_numerical_inputs_digest"],
                     f"{case_id}: captured physical/numerical graph or external inputs changed")
            if case_id in PARTICLE_CASES:
                result["observer_compatibility"][case_id] = _observer_compatibility(first, second)
                before_particles.extend(first["runs"])
                after_particles.extend(second["runs"])
            else:
                old, new = _scan_evidence(first, old_id), _scan_evidence(second, new_id)
                _require(_canonical(sorted((list(key), value) for key, value in old.items())) ==
                         _canonical(sorted((list(key), value) for key, value in new.items())),
                         "Scan matrices, residuals, cache sources or execution resources differ between suites")
                result["scan_comparison"] = {"status": "PASS", "runs": len(old),
                    "input_identity": old_id["physical_and_numerical_inputs_digest"],
                    "run_evidence": [{"phase": key[0], "repeat": key[1], **value} for key, value in sorted(old.items())],
                    "scope": "Exact local-response matrix/residual equivalence; not full emitted-beam qualification"}
                result["performance"].extend(_timings(first, second, case_id))
        if before_particles:
            old_report, new_report = summarize_runs(before_particles), summarize_runs(after_particles)
            result["retained_physical_qualification"] = {
                "before": old_report["physical_qualification"], "after": new_report["physical_qualification"]}
            if "FAIL" in result["retained_physical_qualification"].values():
                result["physical_qualification"] = "FAIL"
            comparison = compare_reports(old_report, new_report)
            result["particle_comparison"] = comparison
            _require(comparison["status"] == "PASS", "Particle report comparison did not pass")
            result["performance"].extend(comparison["performance"])
        result["status"] = "PASS" if result["particle_comparison"] or result["scan_comparison"] else "NOT_EVALUATED"
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError, OverflowError) as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
    return result


def format_comparison(report):
    lines = ["Benchmark suite comparison", "", f"Same-input equivalence: {report['status']}",
             report["scope"], "Numerical convergence: NOT_RUN",
             f"Physical qualification: {report['physical_qualification']}", ""]
    lines.extend("Failure: " + failure for failure in report["failures"])
    for name in ("particle_comparison", "scan_comparison"):
        if report.get(name):
            lines.append(f"{name}: {report[name]['status']}")
    for row in report["performance"]:
        speedup = "unavailable" if row["speedup"] is None else f"{row['speedup']:.6g}x"
        lines.append(f"{row['case_id']} / {row['phase']}: before {row['before_median_seconds']:.9g} s; "
                     f"after {row['after_median_seconds']:.9g} s; speedup {speedup}")
    for row in report["declarations"]:
        lines.append(f"{row['case_id']}: NOT_RUN; comparison NOT_EVALUATED")
    lines.append("Timing ratios are descriptive; a faster result does not establish physical validity.")
    return "\n".join(lines) + "\n"


def _atomic_write(path, text):
    descriptor, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="Completed baseline suite directory")
    parser.add_argument("after", type=Path, help="Completed candidate suite directory")
    parser.add_argument("--output", required=True, type=Path, help="New comparison directory; existing paths are rejected")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.exit(2, f"Cannot reserve a new comparison directory: {error}\n")
    report = compare_suites(args.before, args.after)
    _atomic_write(output / "comparison.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(output / "comparison.txt", format_comparison(report))
    print(format_comparison(report), end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
