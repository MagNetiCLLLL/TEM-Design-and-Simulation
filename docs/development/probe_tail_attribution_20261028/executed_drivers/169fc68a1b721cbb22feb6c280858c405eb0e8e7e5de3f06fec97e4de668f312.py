"""Run only order 7 after rechecking immutable completed order-3/5 evidence.

The original exact full-source replay remains the baseline authority. This
wrapper records its own identity and the frozen dependency driver; it does not
edit previous reports, regenerate a source or propagate orders 3/5 again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent/"agent.md").is_file())
for directory in (ROOT/"src", ROOT/"scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_round_field_sensitivity as radial
from run_column_energy_diagnostic import load_donor
from measure_probe_waist import _array_hash, _hash, _json, _npz, analyze_scan

SCHEMA = "temsim-round-radial-order7-reuse-v1"


def preserved_keys(arrays):
    return [key for key in arrays if key.startswith(("gun_", "gun_exit_", "emission_"))
        or key in ("source_id", "weights", "source_azimuth_rad", "source_energy_offset_ev")]


def verify_prior(donor, previous):
    """Check all prior proof files and independently repeat the raw replay gate."""
    previous = Path(previous).resolve()
    report = json.loads((previous/"report.json").read_text(encoding="utf-8"))
    radial.require(report.get("status") == "COMPLETE" and report.get("source_input_unchanged") is True,
        "Prior radial comparison is incomplete or changed inputs")
    radial.require(report.get("source_input_digest") == donor.snapshot.digest,
        "Prior radial comparison uses another complete input snapshot")
    radial.require(report.get("donor_files_sha256") == donor.files_sha256,
        "Prior radial comparison uses another donor archive")
    dependency = Path(radial.__file__)
    radial.require(_hash(previous/"driver-source.py") == report.get("driver_sha256") == _hash(dependency),
        "The reviewed radial driver differs from the archived prior driver")
    radial.require(report.get("input_arrays_sha256") == {key: _array_hash(value) for key, value in donor.arrays.items()},
        "Prior comparison did not use these exact original arrays")
    stages = {row["order"]: row for row in report.get("stages", [])}
    radial.require(len(stages) == 2 and set(stages) == {3, 5}, "Complete prior orders 3 and 5 are required")
    radial.require(stages[3].get("baseline_reproduction", {}).get("passed") is True,
        "Prior order-3 replay did not pass")
    hashes = {"report.json": _hash(previous/"report.json"), "driver-source.py": _hash(previous/"driver-source.py")}
    arrays, analyses = {}, {}
    for order in (3, 5):
        stage = stages[order]
        radial.require(stage.get("status") == "COMPLETE", "A prior stage did not complete")
        pinned = stage.get("files_sha256", {})
        radial.require(set(pinned) == {"runtime.json", "field-audit.json", "particles.npz"},
            "Prior stage lacks its complete immutable raw/runtime manifest")
        folder = previous/f"order-{order}"
        for name, digest in pinned.items():
            radial.require(_hash(folder/name) == digest, f"Prior {order} file checksum mismatch: {name}")
            hashes[f"order-{order}/{name}"] = digest
        radial.require(json.loads((folder/"report.json").read_text(encoding="utf-8")) == stage,
            "Prior standalone stage report disagrees with the final run report")
        hashes[f"order-{order}/report.json"] = _hash(folder/"report.json")
        with np.load(folder/"particles.npz", allow_pickle=False) as saved:
            raw = {key: saved[key].copy() for key in saved.files}
        for key in ("z_mm", "integration_z_mm", *preserved_keys(donor.arrays)):
            radial.require(_array_hash(raw[key]) == _array_hash(donor.arrays[key]),
                f"Prior order {order} changed the original source or mesh: {key}")
        if order == 3:
            radial.require(radial.baseline_comparison(raw, donor.arrays)["passed"],
                "Independent recheck of the full order-3 saved phase failed")
        arrays[order] = raw
        analyses[order] = stage["analysis"]
    return dict(directory=str(previous), files_sha256=hashes, driver_sha256=report["driver_sha256"],
        donor_input_digest=donor.snapshot.digest, exact_replay_rechecked=True,
        prior_propagations_reused=[3, 5], new_propagations_requested=[7]), arrays, analyses


def compare_raw(old, new, surface):
    radial.require(np.array_equal(old["z_mm"], new["z_mm"]), "Paired comparison requires exact common checkpoints")
    common = old["alive"] & new["alive"]
    delta = np.hypot(new["x_m"]-old["x_m"], new["y_m"]-old["y_m"])
    i = int(np.flatnonzero(old["z_mm"] == surface)[0])
    radial.require(common.any() and common[i].any(), "No paired live support")
    j, k = np.unravel_index(np.argmax(np.where(common, delta, -1)), delta.shape)
    return dict(same_actual_population=bool(np.array_equal(old["alive"], new["alive"])),
        same_ids=bool(np.array_equal(old["source_id"], new["source_id"])),
        same_absolute_weights=bool(np.array_equal(old["weights"], new["weights"])),
        maximum_raw_xy_shift_nm=float(delta[j, k]*1e9), maximum_shift_z_mm=float(old["z_mm"][j]),
        maximum_shift_source_id=int(old["source_id"][k]),
        maximum_surface_raw_xy_shift_nm=float(delta[i, common[i]].max()*1e9),
        scope="Raw laboratory coordinates on actual common survivors; no centering/rescaling/reweighting")


def run(donor_path, prior_path, output, *, threads=4, budget_s=1800., cancel_file=None,
        executor=radial.execute_fixed_source):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_runtime import get_unified_field_runtime, UnifiedFieldRuntime
    from temsim.cpu_resources import numerical_job
    radial.require(type(threads) is int and 1 <= threads <= 4, "Use at most four threads")
    radial.require(isinstance(budget_s, (int, float)) and not isinstance(budget_s, bool)
        and np.isfinite(budget_s) and budget_s > 0, "Budget must be finite and positive")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    def cancelled():
        return perf_counter()-started >= budget_s or bool(cancel_file and Path(cancel_file).exists())
    report = dict(schema=SCHEMA, status="RUNNING", qualification="NATIVE_ENGINEERING_RADIAL_SENSITIVITY_ONLY",
        scope=radial.SCOPE, radial_order=7, budget_s=budget_s, threads_requested=threads,
        new_column_orders=[], gun_traces=0, driver_sha256=_hash(Path(__file__)))
    (output/"driver-source.py").write_bytes(Path(__file__).read_bytes())
    _json(output/"report.json", report)
    donor = None
    try:
        with numerical_job(threads, cancelled=cancelled) as cpu:
            report["cpu_resources"] = cpu.to_dict()
            donor = load_donor(donor_path)
            proof, prior_arrays, prior_analyses = verify_prior(donor, prior_path)
            report.update(prior_proof=proof, donor_files_sha256=donor.files_sha256,
                input_digest=donor.snapshot.digest, dependency_driver_sha256=_hash(Path(radial.__file__)),
                donor_loader_sha256=_hash(Path(__file__).with_name("run_column_energy_diagnostic.py")))
            (output/"dependency-driver-source.py").write_bytes(Path(radial.__file__).read_bytes())
            _json(output/"inputs.json", donor.snapshot.to_dict())
            base = get_unified_field_runtime(donor.state)
            for order in (3, 5):
                # Static construction only. This independently binds the prior
                # executable fields to the same unchanged source and geometry.
                plan, audit = radial.prepare_plan(donor, base, order)
                saved_runtime = UnifiedFieldRuntime.from_dict(json.loads(
                    (Path(proof["directory"])/f"order-{order}/runtime.json").read_text(encoding="utf-8")))
                radial.require(plan.unified_runtime.digest == saved_runtime.digest,
                    "Prior runtime does not equal the current explicit construction")
            if cancelled():
                raise InterruptedError("Order-7 diagnostic cancelled before propagation")
            plan, audit = radial.prepare_plan(donor, base, 7)
            _json(output/"runtime.json", plan.unified_runtime.to_dict())
            _json(output/"field-audit.json", audit)
            report["new_column_orders"] = [7]
            _json(output/"report.json", report)
            raw, execution = executor(donor, plan, cancelled)
            radial.require(all(_array_hash(raw[key]) == _array_hash(donor.arrays[key])
                for key in preserved_keys(donor.arrays)), "Order-7 execution changed original source arrays")
            _npz(output/"particles.npz", raw)
            analysis = analyze_scan(raw, donor.planes, surface_z_mm=float(donor.state.sample.upper_surface_z_mm),
                source_current_a=donor.report["receipt"]["source_current_a"])
            report.update(execution=execution, analysis=analysis,
                change_from_order3=radial.profile_comparison(prior_analyses[3], analysis),
                change_from_order5=radial.profile_comparison(prior_analyses[5], analysis),
                raw_change_from_order5=compare_raw(prior_arrays[5], raw, float(donor.state.sample.upper_surface_z_mm)))
            radial.require(all(_hash(Path(proof["directory"])/name) == digest for name, digest in proof["files_sha256"].items()),
                "Immutable prior evidence changed during order-7 execution")
            radial.require(all(_hash(donor.directory/name) == digest for name, digest in donor.files_sha256.items()),
                "Original donor archive changed during order-7 execution")
            radial.require(capture_instrument_snapshot(donor.state).digest == donor.snapshot.digest,
                "Order-7 execution changed instrument inputs")
            report.update(status="COMPLETE", prior_and_donor_unchanged=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="CANCELLED" if cancelled() or isinstance(error, (KeyboardInterrupt, InterruptedError)) else "FAILED",
            error=f"{type(error).__name__}: {error}")
    finally:
        report["elapsed_s"] = perf_counter()-started
        _json(output/"report.json", report)
        _json(output/"checksums.json", {path.name: _hash(path) for path in output.iterdir()
            if path.is_file() and path.name != "checksums.json"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--budget-s", type=float, default=1800.)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    result = run(args.donor, args.prior, args.output, threads=args.threads,
        budget_s=args.budget_s, cancel_file=args.cancel_file)
    print(json.dumps(dict(status=result["status"], report=str(args.output.resolve()/"report.json"))))
    return 0 if result["status"] == "COMPLETE" else 130 if result["status"] == "CANCELLED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
