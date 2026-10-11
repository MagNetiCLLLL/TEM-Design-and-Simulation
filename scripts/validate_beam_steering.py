"""Executed default-source GUI-request / finite steering validation.

No physical defaults are edited. Numerical ray density and integration step
are explicit. Reports include real backend receipts; headless widget/worker
API execution is not represented as a visible-desktop user-interface test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from threading import Event
from time import perf_counter
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--rays", type=int, default=3000)
    parser.add_argument("--step-mm", type=float, default=.05)
    args = parser.parse_args()
    destination = Path(args.directory)
    destination.mkdir(parents=True, exist_ok=True)
    from PySide6.QtCore import QCoreApplication
    app = QCoreApplication.instance() or QCoreApplication([])
    from temsim.optics.column import default_state
    from temsim.gui.calculation_request import CapturedCalculationRequest
    from temsim.gui.calculation_controller import CalculationWorker
    from temsim.instrument_snapshot import encode_instrument, decode_instrument, capture_instrument_snapshot
    from temsim.immutable_json import json_digest
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.beam_steering_calibration import (
        capture_beam_steering_reference, BeamSteeringCalibrator,
        apply_beam_steering_solution, DRIVE_FIELDS, _transport,
    )
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.compute_backend import cuda_capability
    from temsim.cpu_resources import numerical_job
    from temsim.optics.actual_probe_diagnostics import calculate_actual_probe_diagnostic
    report = dict(scope="Actual default physical source and optics; GUI calculation request/worker APIs; "
        "one identified electron for steering, full population for actual-probe diagnostics; not native desktop interaction",
        status="running", implementation=solver_source_identity(), stages=[], backend_receipts=[])
    start = perf_counter()

    def save():
        (destination/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False,
            default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)), encoding="utf-8")

    def progress(message, **values):
        row = dict(message=message, elapsed_s=perf_counter()-start, **values)
        report["stages"].append(row)
        print(json.dumps(row), flush=True)
        save()

    try:
        state = default_state()
        # Same default physical inputs. These two changes only select requested
        # numerical density; shared global Auto must itself choose CUDA.
        state.step_mm = args.step_mm
        state.electron_gun.emitter.ray_count = args.rays
        report["gpu_capability"] = asdict(cuda_capability())
        report["global_backend_policy"] = state.acceleration_backend
        report["ray_count"] = args.rays
        report["column_step_mm"] = args.step_mm
        report["initial_drives"] = {key: getattr(state.beam_deflector, key) for key in DRIVE_FIELDS}
        report["scan_epoch_s"] = float(getattr(state, "simulation_time_s", 0.))
        progress("Capturing actual GUI ray request")
        capture = CapturedCalculationRequest.capture(state, "High accuracy", args.rays, args.step_mm, workflow="rays")
        request = capture.prepare(Event())
        request.calculation_manifest.to_dict()
        (destination/"request.json").write_text(json.dumps(request.calculation_manifest.to_dict(), indent=2), encoding="utf-8")
        result_box, errors = [], []
        worker = CalculationWorker(1, "High accuracy", request.snapshot,
            model_signature=request.model_signature, request_signatures=request.request_signatures,
            calculation_manifest=request.calculation_manifest, external_inputs=request.external_inputs,
            workflow="rays")
        worker.signals.result.connect(lambda generation, quality, result, seconds: result_box.append((result, seconds)))
        worker.signals.error.connect(lambda generation, quality, message: errors.append(message))
        worker.signals.progress.connect(lambda generation, quality, done, total, message: progress(message, done=done, total=total))
        worker.run()
        if errors or not result_box:
            raise RuntimeError("GUI ray worker failed: "+"; ".join(errors))
        result, duration = result_box[0]
        report["ray_seconds"] = duration
        report["ray_backend_receipts"] = result.performance["backend_stages"]
        report["ray_model_signature"] = result.model_signature
        progress("Actual CalculationResult returned; selecting reference electron", backend=result.state_snapshot.active_backend)
        # Critically exercise current live-state + separately calculated result,
        # rather than relabelling a historical boundary or the worker snapshot.
        with numerical_job(), backend_receipts(report["backend_receipts"]):
            reference = capture_beam_steering_reference(state, result=result, target_z_mm=state.sample.z_mm)
            report["reference"] = asdict(reference)
            original = encode_instrument(state)
            solver = BeamSteeringCalibrator(state, reference, target_z_mm=state.sample.z_mm, maximum_step_mm=args.step_mm)
            calibration = solver.calibrate()
            report["calibration"] = asdict(calibration)
            assert json_digest(encode_instrument(state)) == json_digest(original)
            progress("Entrance Jacobian calibrated", condition=calibration.condition_number, backend=calibration.backend)
            solutions = []
            for kind, x, y in (("tilt", 1e-4, -5e-5), ("shift", 1e-9, -5e-10)):
                stage = perf_counter()
                solution = solver.solve(kind=kind, x=x, y=y)
                local = decode_instrument(original)
                apply_beam_steering_solution(local, solution)
                expected = decode_instrument(original)
                for key, value in zip(DRIVE_FIELDS, solution.drives_mrad):
                    setattr(expected.beam_deflector, key, value)
                assert json_digest(encode_instrument(local)) == json_digest(encode_instrument(expected))
                endpoint, backend = _transport(local, reference, state.sample.z_mm, args.step_mm/2.)
                baseline_state = decode_instrument(original)
                refined_baseline, _ = _transport(baseline_state, reference, state.sample.z_mm, args.step_mm/2.)
                achieved = np.asarray(endpoint.phase_space)
                baseline = np.asarray(refined_baseline.phase_space)
                target = baseline.copy()
                if kind == "tilt":
                    target[2:] = np.tan(np.arctan(target[2:])+np.array([x, y]))
                else:
                    target[:2] += (x, y)
                residual = achieved-target
                row = dict(kind=kind, command_xy_SI=[x, y], solution=asdict(solution),
                    refined_step_mm=args.step_mm/2., refined_endpoint=achieved.tolist(),
                    refined_baseline=baseline.tolist(), refined_residual=residual.tolist(),
                    backend=backend, only_four_drives_changed=True, seconds=perf_counter()-stage)
                solutions.append(row)
                report["solutions"] = solutions
                progress(kind+" verified with half integration step", residual=residual.tolist())
                if np.max(np.abs(residual[:2])) > 1e-11 or np.max(np.abs(residual[2:])) > 1e-9:
                    raise AssertionError("Half-step response does not meet the stated steering tolerances")
            progress("Calculating actual input population at corrector planes")
            diagnostic = calculate_actual_probe_diagnostic(result)
            report["actual_probe"] = dict(planes=diagnostic.planes, provenance=diagnostic.provenance, scope=diagnostic.scope)
            np.savez_compressed(destination/"actual_probe.npz", z_mm=diagnostic.checkpoints.z_mm,
                x_m=diagnostic.checkpoints.x_m, y_m=diagnostic.checkpoints.y_m,
                tx_rad=diagnostic.checkpoints.tx_rad, ty_rad=diagnostic.checkpoints.ty_rad,
                kinetic_energy_ev=diagnostic.checkpoints.kinetic_energy_ev,
                flight_time_s=diagnostic.checkpoints.flight_time_s, alive=diagnostic.alive,
                weights=diagnostic.weights, source_ray_id=diagnostic.source_ray_id)
        assert solver_source_identity() == report["implementation"], "Implementation changed during validation"
        executed = [row for row in report["backend_receipts"] if row["outcome"] == "returned"]
        assert any(row["actual"] == "CUDA GPU" for row in executed), "No actual CUDA receipt"
        report["status"] = "passed"
        progress("All scoped validation completed")
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        save()
        raise
    finally:
        report["total_seconds"] = perf_counter()-start
        save()


if __name__ == "__main__":
    main()
