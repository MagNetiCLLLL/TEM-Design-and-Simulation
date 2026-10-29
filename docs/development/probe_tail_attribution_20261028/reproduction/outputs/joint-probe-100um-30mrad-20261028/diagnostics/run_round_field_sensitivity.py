"""Explicit native round-field radial truncation sensitivity, with fixed source.

This attachment does not change production capture, state, source generation,
controls, geometry or empirical aberrations. It constructs immutable runtimes
and passes them through the production explicit-runtime plan hook. Order 3
must reproduce the saved donor mesh and phase before order 5 is executed.
Results concern the native engineering expansion, not OEM fields or pure Cs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent/"agent.md").is_file())
for directory in (ROOT/"src", ROOT/"scripts", Path(__file__).resolve().parent):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from measure_probe_waist import PHASE, _array_hash, _hash, _json, _npz, analyze_scan

SCHEMA = "temsim-explicit-round-radial-sensitivity-v1"
SCOPE = "Fixed saved gun-exit source; native engineering round-field radial truncation only; no OEM or pure-Cs qualification"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def variant_runtime(base, order):
    """Replace only native round A_phi radial order in an immutable capture."""
    from temsim.immutable_json import thaw_json
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
    require(type(order) is int and order in (3, 5, 7), "Diagnostic radial order must be 3, 5 or 7")
    require(type(base) is UnifiedFieldRuntime and type(base.snapshot) is ElectromagneticPotentialSnapshot,
        "This diagnostic requires an exact static unified capture")
    metadata = thaw_json(base.metadata)
    require(metadata.get("round_radial_order") == 3, "The donor baseline must declare production order 3")
    for key in ("empirical_aberrations_applied", "equivalent_image_maps_applied"):
        require(metadata.get(key) is False, "Donor has additional or unknown empirical action")
    rows, manifest = [], []
    for row in base.snapshot.contributions:
        old = row.potential
        field = old
        axis = None
        if type(old) is HighOrderRoundLensPotential:
            require(old.radial_order == 3, "Every native round donor provider must use order 3")
            field = old if order == 3 else replace(old, radial_order=order)
            # At the native local axis, A, B and its first two A derivatives
            # do not change when r^5/r^7 terms are added. This tests the frozen
            # providers themselves, not a hand-written surrogate field.
            centers = np.asarray([term[1] for term in old.native.terms_t_m], dtype=float)
            local = np.column_stack((np.zeros((len(centers), 2)), centers))
            points = local@old.registration.rotation_array.T+old.registration.origin_array_m
            expected = {**old.sample(points), **old.sample_jet(points)}
            actual = {**field.sample(points), **field.sample_jet(points)}
            axis = {key: dict(exact=bool(np.array_equal(expected[key], actual[key])),
                maximum_absolute_difference=float(np.max(np.abs(expected[key]-actual[key]), initial=0.)))
                for key in expected}
            require(all(value["exact"] for value in axis.values()), "The overridden provider changed its native on-axis field/jet")
            require(field.native.fingerprint == old.native.fingerprint and field.valid_radius_m == old.valid_radius_m
                and field.axial_domain == old.axial_domain and field.remote_domain == old.remote_domain,
                "Radial-order override changed native coefficients, registration or declared domain")
        rows.append(row if field is old else PotentialContribution(row.source_id, row.owner_ids, field, row.kind))
        manifest.append(dict(source_id=row.source_id, owner_ids=list(row.owner_ids), kind=row.kind,
            provider_type=type(old).__name__, original_digest=old.digest, executed_digest=field.digest,
            changed=field is not old, order_before=getattr(old, "radial_order", None),
            order_after=getattr(field, "radial_order", None), native_axis_checks=axis))
    require(any(row["order_before"] == 3 for row in manifest), "No native round providers were captured")
    if order == 3:
        result = base
    else:
        snapshot = ElectromagneticPotentialSnapshot(base.snapshot.scene, tuple(rows), base.snapshot.gauge)
        metadata.update(round_radial_order=order, explicit_diagnostic_override=dict(
            kind="native_round_radial_order", baseline_runtime_digest=base.digest,
            baseline_order=3, executed_order=order, scope=SCOPE))
        result = UnifiedFieldRuntime(snapshot, base.geometry, base.source_identity, metadata)
    require(result.source_identity == base.source_identity and result.geometry is base.geometry
        and result.snapshot.scene is base.snapshot.scene and result.snapshot.gauge is base.snapshot.gauge,
        "Explicit runtime changed instrument, scene, gauge or geometry")
    return result, manifest


def prepare_plan(donor, base, order):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_particle import build_unified_plan
    from audit_joint_probe_fields import _inputs, _plan_actions, _field_ownership, _c2_geometry
    state = donor.state
    require(capture_instrument_snapshot(state).digest == donor.snapshot.digest, "Donor state changed before plan construction")
    base.require_state(state)
    baseline_owners = _field_ownership(state, base)
    runtime, manifest = variant_runtime(base, order)
    z, cp = donor.arrays["integration_z_mm"], donor.arrays["z_mm"]
    plan = build_unified_plan(state, float(z[0]), float(z[-1]),
        save_z_mm=tuple(z), checkpoint_z_mm=tuple(cp), _captured_runtime=runtime)
    require(np.array_equal(plan.z_mm, z), "Explicit plan changed the donor integration mesh")
    require(np.array_equal(plan.z_mm[plan.checkpoint_index], cp), "Explicit plan changed the donor checkpoint mesh")
    if order == 3:
        require(plan.signature == donor.field_identity["plan_signature"], "Order-3 plan signature differs from the executed donor")
    aperture, start, surface = _inputs(state)
    require(z[0] == start and z[-1] > surface, "Donor is not a complete gun-exit through-focus interval")
    audit = dict(schema=SCHEMA, scope=SCOPE, status="PASS", passed=True,
        plan_actions=_plan_actions(plan, runtime, start, float(z[-1])),
        c2_aperture=_c2_geometry(aperture, runtime, start, float(z[-1])),
        baseline_field_ownership=baseline_owners, provider_manifest=manifest,
        original_input_digest=donor.snapshot.digest, executed_runtime_digest=runtime.digest,
        executed_field_digest=runtime.snapshot.numerical_digest, geometry_digest=runtime.geometry.digest,
        radial_order=order, explicit_override=order != 3)
    require(capture_instrument_snapshot(state).digest == donor.snapshot.digest, "Plan preparation changed donor inputs")
    return plan, audit


def execute_fixed_source(donor, plan, cancelled, *, batch_size=128):
    """Use real Lorentz integration/contact checks and saved absolute source."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_particle import execute_unified_plan, merge_unified_stops
    from temsim.physics.recording_clipping import clip_recording_planes
    require(type(batch_size) is int and 1 <= batch_size <= 128, "Batch size must lie in [1,128]")
    state, saved = donor.state, donor.arrays
    original = capture_instrument_snapshot(state).digest
    state._tuning_cancelled = cancelled
    pieces, ledgers = [], []
    count = len(saved["source_id"])
    for first in range(0, count, batch_size):
        if cancelled():
            raise InterruptedError("Round-field diagnostic stage budget/cancellation reached")
        selected = slice(first, min(first+batch_size, count))
        alive = saved["gun_exit_alive"][selected].copy()
        # Match the production section adapter: absorbed gun placeholders are
        # retained in the saved source, but not launched as live column rays.
        initial = [np.where(alive, saved["gun_exit_"+key][selected], np.nan)
            for key in ("x_m", "tx_rad", "y_m", "ty_rad")]
        output = execute_unified_plan(state, plan, *initial, saved["source_energy_offset_ev"][selected],
            include_initial_plane_kicks=True, defer_nonfinite_until_clipping=True,
            initial_time_s=saved["gun_exit_flight_time_s"][selected], return_flight_times=True,
            initial_kinetic_energy_ev=saved["gun_exit_kinetic_energy_ev"][selected], history_policy="checkpoints")
        z, x, tx, y, ty, clock, cp = output
        require(np.array_equal(cp.z_mm, saved["z_mm"]), "Executor changed exact checkpoint rows")
        alive, stopped, labels = merge_unified_stops(z, alive, saved["gun_blocked_z_mm"][selected],
            saved["gun_blocked_key"][selected].tolist())
        alive, stopped, labels = clip_recording_planes(state, z, x, y, alive, stopped, labels)
        pieces.append({**{key: np.asarray(getattr(cp, key)).copy() for key in PHASE},
            "alive": np.asarray(cp.unified_state.active).copy()})
        ledgers.append((alive.copy(), stopped.copy(), np.asarray(labels, dtype=str)))
    require(capture_instrument_snapshot(state).digest == original, "Execution changed donor instrument inputs")
    arrays = {key: np.concatenate([part[key] for part in pieces], axis=1) for key in (*PHASE, "alive")}
    arrays.update({key: value.copy() for key, value in saved.items()
        if key not in (*PHASE, "alive", "terminal_alive", "blocked_z_mm", "blocked_key")})
    arrays.update(terminal_alive=np.concatenate([part[0] for part in ledgers]),
        blocked_z_mm=np.concatenate([part[1] for part in ledgers]),
        blocked_key=np.concatenate([part[2] for part in ledgers]))
    return arrays, dict(column_batches=len(pieces), gun_traces=0, prefix_reuse=False,
        integration_nodes=len(plan.z_mm), checkpoints=len(saved["z_mm"]), history_policy="checkpoints",
        scope="Every integration/contact step is executed; only retained output rows are sparse")


def baseline_comparison(actual, saved):
    """Fail closed on any phase difference; exact replay is stricter than roundoff."""
    keys = (*PHASE, "alive", "z_mm", "integration_z_mm", "source_id", "weights",
        "terminal_alive", "blocked_z_mm", "blocked_key")
    result = {}
    for key in keys:
        a, b = np.asarray(actual[key]), np.asarray(saved[key])
        equal = a.shape == b.shape and a.dtype == b.dtype and (np.array_equal(a, b, equal_nan=True)
            if a.dtype.kind in "fc" else np.array_equal(a, b))
        finite = np.isfinite(a) & np.isfinite(b) if a.shape == b.shape and a.dtype.kind in "fc" else None
        result[key] = dict(exact=bool(equal), maximum_absolute_difference=(
            float(np.max(np.abs(a[finite]-b[finite]), initial=0.)) if finite is not None else None))
    return dict(passed=all(row["exact"] for row in result.values()), arrays=result,
        criterion="Exact float64 values, common NaN placeholders, masks, IDs, absolute weights and stop ledger; no loose optical tolerance")


def profile_comparison(reference, changed):
    require([row["z_mm"] for row in reference["planes"]] == [row["z_mm"] for row in changed["planes"]],
        "Comparison profiles must use the same exact observed planes")
    old = np.array([row["d95_nm"] for row in reference["planes"]])
    new = np.array([row["d95_nm"] for row in changed["planes"]])
    require(np.all(old > 0), "Relative D95 sensitivity requires positive donor diameters")
    return dict(maximum_relative_d95_change=float(np.max(np.abs(new/old-1))),
        surface_d95_delta_nm=changed["surface"]["d95_nm"]-reference["surface"]["d95_nm"],
        observed_minimum_offset_delta_nm=changed["d95_waist"]["representative_offset_nm"]-
            reference["d95_waist"]["representative_offset_nm"],
        scope="Same original source and integration mesh; field truncation sensitivity, not OEM validation")


def run(donor, output, *, through_order=5, threads=4, stage_budget_s=1800., cancel_file=None,
        executor=execute_fixed_source):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    require(through_order in (3, 5, 7), "Through-order must be 3, 5 or 7")
    require(type(threads) is int and 1 <= threads <= 4, "This diagnostic allows at most four threads")
    require(isinstance(stage_budget_s, (int, float)) and not isinstance(stage_budget_s, bool)
        and math.isfinite(stage_budget_s) and stage_budget_s > 0, "Stage budget must be positive and finite")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, status="RUNNING", scope=SCOPE, qualification="NOT_QUALIFIED",
        source_input_digest=donor.snapshot.digest, threads=threads, stage_budget_s=stage_budget_s,
        stages=[], requested_orders=[order for order in (3, 5, 7) if order <= through_order],
        input_arrays_sha256={key: _array_hash(value) for key, value in donor.arrays.items()},
        driver_sha256=_hash(Path(__file__)), donor_files_sha256=dict(donor.files_sha256),
        donor_report_sha256=_hash(donor.directory/"report.json"),
        donor_loader_sha256=_hash(Path(__file__).with_name("run_column_energy_diagnostic.py")))
    (output/"driver-source.py").write_bytes(Path(__file__).read_bytes())
    _json(output/"inputs.json", donor.snapshot.to_dict())
    _json(output/"report.json", report)
    started, baseline, previous_analysis = perf_counter(), None, None
    cancelled = lambda: bool(cancel_file and Path(cancel_file).exists())
    try:
        base = get_unified_field_runtime(donor.state)
        base.require_state(donor.state)
        expected = donor.report["receipt"]["batches"][0]["field_audit"]["identities"]
        require(base.digest == expected["runtime_digest"] and base.snapshot.numerical_digest == expected["potential_digest"]
            and base.geometry.digest == expected["geometry_digest"], "Reconstructed base field/geometry differs from the executed donor")
        from temsim.cpu_resources import numerical_job
        with numerical_job(threads) as cpu:
            report["cpu_resources"] = cpu.to_dict()
            for order in report["requested_orders"]:
                stage_start = perf_counter()
                def cancelled():
                    return perf_counter()-stage_start >= stage_budget_s or bool(cancel_file and Path(cancel_file).exists())
                if cancelled():
                    raise InterruptedError("Round-field stage cancelled before preparation")
                folder = output/f"order-{order}"
                folder.mkdir()
                row = dict(order=order, status="RUNNING")
                report["stages"].append(row)
                _json(output/"report.json", report)
                plan, audit = prepare_plan(donor, base, order)
                _json(folder/"field-audit.json", audit)
                _json(folder/"runtime.json", plan.unified_runtime.to_dict())
                require(capture_instrument_snapshot(donor.state).digest == donor.snapshot.digest, "State drift before execution")
                arrays, execution = executor(donor, plan, cancelled)
                # This gate binds even a mocked/test executor to every saved
                # source array, while allowing actual propagated phase to vary.
                preserved = [key for key in donor.arrays if key.startswith(("gun_", "gun_exit_", "emission_"))
                    or key in ("source_id", "weights", "source_azimuth_rad", "source_energy_offset_ev")]
                require(all(_array_hash(arrays[key]) == report["input_arrays_sha256"][key] for key in preserved),
                    "Diagnostic execution changed the retained original source")
                _npz(folder/"particles.npz", arrays)
                analysis = analyze_scan(arrays, donor.planes,
                    surface_z_mm=float(donor.state.sample.upper_surface_z_mm),
                    source_current_a=donor.report["receipt"]["source_current_a"])
                row.update(status="EVALUATED", execution=execution, analysis=analysis,
                    elapsed_s=perf_counter()-stage_start)
                if order == 3:
                    row["baseline_reproduction"] = baseline_comparison(arrays, donor.arrays)
                    require(row["baseline_reproduction"]["passed"], "Order-3 exact replay failed; higher orders are forbidden")
                    baseline = analysis
                else:
                    require(baseline is not None, "No validated order-3 baseline")
                    row["change_from_order3"] = profile_comparison(baseline, analysis)
                    row["change_from_previous_order"] = profile_comparison(previous_analysis, analysis)
                previous_analysis = analysis
                row["status"] = "COMPLETE"
                row["files_sha256"] = {name: _hash(folder/name) for name in
                    ("field-audit.json", "runtime.json", "particles.npz")}
                _json(folder/"report.json", row)
                _json(output/"report.json", report)
        require(all(_hash(donor.directory/name) == digest for name, digest in donor.files_sha256.items()),
            "Donor archive changed during the diagnostic")
        report["status"] = "COMPLETE"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="CANCELLED" if cancelled() or isinstance(error, (InterruptedError, KeyboardInterrupt)) else "FAILED",
            error=f"{type(error).__name__}: {error}")
        if report["stages"] and report["stages"][-1]["status"] != "COMPLETE":
            report["stages"][-1].update(status=report["status"], error=report["error"])
            _json(output/f"order-{report['stages'][-1]['order']}"/"report.json", report["stages"][-1])
    finally:
        report["elapsed_s"] = perf_counter()-started
        report["source_input_unchanged"] = capture_instrument_snapshot(donor.state).digest == donor.snapshot.digest
        if not report["source_input_unchanged"]:
            report["status"] = "FAILED"
        _json(output/"report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through-order", type=int, choices=(3, 5, 7), default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--stage-budget-s", type=float, default=1800.)
    parser.add_argument("--cancel-file", type=Path)
    args = parser.parse_args(argv)
    from run_column_energy_diagnostic import load_donor
    try:
        donor = load_donor(args.donor)
        report = run(donor, args.output, through_order=args.through_order, threads=args.threads,
            stage_budget_s=args.stage_budget_s, cancel_file=args.cancel_file)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Round-field diagnostic failed before execution: {error}\n")
    print(json.dumps(dict(status=report["status"], report=str(args.output.resolve()/"report.json"))))
    return 0 if report["status"] == "COMPLETE" else 130 if report["status"] == "CANCELLED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
