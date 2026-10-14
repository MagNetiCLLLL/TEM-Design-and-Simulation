"""CPU-only selected-nanoprobe topology and specimen-focus acceptance.

Runs a selected incident-column assembly at two integration steps with a fixed
particle budget. This is not particle-sampling convergence, wave-optical probe
qualification or an OEM calibration. Candidate input never changes the catalog.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter


_COLUMNS = ("C3 + Probe Corrector", "C3 + Probe Corrector + B-COR")


def _positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range", choices=("normal", "large_angle"), default="normal",
                        help="Nanoprobe optical branch to validate")
    parser.add_argument("--column", choices=_COLUMNS, default=_COLUMNS[0],
                        help="incident column; B-COR retains its downstream projector controls")
    parser.add_argument("--rays", type=int, default=3000,
                        help="fixed emitted-particle budget (default: calibrated 3000); smaller budgets can change alpha95 and do not relax acceptance")
    parser.add_argument("--step", type=_positive, default=.05,
                        help="coarse integration step in mm; also runs step/2")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--target", type=_positive, default=30.,
                        help="target current-contained alpha95 in mrad (default: 30)")
    parser.add_argument("--candidate", type=Path,
                        help="JSON with lenses:{key:percent}, hexapoles:{key:{strength_m3,orientation_rad}}")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/nanoprobe_condenser_validation.json"))
    args = parser.parse_args()
    if args.rays < 16:
        parser.error("--rays must be at least 16; surviving/effective rays are checked separately")
    if args.candidate and args.candidate.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --candidate")
    return args


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _load_candidate(state, path):
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or set(values) != {"lenses", "hexapoles"}:
        raise ValueError("Candidate requires exactly the lenses and hexapoles tables")
    lenses = {lens.key: lens for lens in state.lenses}
    hexapoles = {item.key: item for item in state.corrector_elements
                 if hasattr(item, "strength_m3") and hasattr(item, "orientation_rad")}
    for table in ("lenses", "hexapoles"):
        if not isinstance(values[table], dict):
            raise ValueError(f"Candidate {table} must be an object")
    for key, value in values["lenses"].items():
        if key not in lenses:
            raise ValueError(f"Unknown candidate lens: {key}")
        percent = _finite_number(value, key)
        if not 0 <= percent <= lenses[key].max_percent:
            raise ValueError(f"Candidate lens {key} exceeds its excitation limits")
        lenses[key].percent = percent
    for key, fields in values["hexapoles"].items():
        if key not in hexapoles or not isinstance(fields, dict):
            raise ValueError(f"Unknown or invalid candidate hexapole: {key}")
        if set(fields) != {"strength_m3", "orientation_rad"}:
            raise ValueError(f"Candidate {key} requires strength_m3 and orientation_rad")
        for name, value in fields.items():
            setattr(hexapoles[key], name, _finite_number(value, f"{key}.{name}"))


def _actual_backends(receipts):
    return sorted({row["actual"] for row in receipts if row["outcome"] == "returned"})


def _configured_state(column_name, nanoprobe_range):
    """Construct detached incident optics without qualifying a B-COR projector."""
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_condenser_mode, apply_operating_mode_pair
    from temsim.optics.column import default_state

    if column_name not in _COLUMNS:
        raise ValueError(f"Unsupported incident-column reference: {column_name}")
    if nanoprobe_range not in {"normal", "large_angle"}:
        raise ValueError(f"Unknown Nanoprobe range: {nanoprobe_range}")
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), column=column_name)
    catalog.apply(state, selection)
    mode_key = "nano_probe_large_angle" if nanoprobe_range == "large_angle" else "nano_probe"
    compatibility = dict(column_name=selection.column, recording_name=selection.recording)
    projector_applied = column_name == _COLUMNS[0]
    if projector_applied:
        apply_operating_mode_pair(state, mode_key, "diffraction", **compatibility)
    else:
        apply_condenser_mode(state, mode_key, **compatibility)
    return state, selection, mode_key, projector_applied


def _execute(args, report):
    # Set limits before importing numerical libraries, even on CUDA-equipped hosts.
    for name in ("TEMSIM_CPU_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "MKL_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    # A persistent verification process may already own a Numba thread pool.
    # Resizing its launch configuration raises on the next compiled call; mask
    # that existing pool below instead, while a fresh CLI uses this small pool.
    if "numba" not in sys.modules:
        os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    from numba import config, set_num_threads, get_num_threads
    from temsim.optics.condenser_topology import validate
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.optics.surface_probe_focus import measure_surface_focus
    from temsim.physics.backend_execution import backend_receipts

    set_num_threads(min(args.threads, config.NUMBA_NUM_THREADS))
    report["numba_threads"] = get_num_threads()
    # This new state is detached from any running GUI or saved instrument.
    state, selection, mode_key, projector_applied = _configured_state(args.column, args.range)
    state.simulation_mode = "custom"
    for deflector in (state.ac_deflector, state.descan_deflector):
        deflector.scan_enabled = False
        if hasattr(deflector, "wobble_enabled"):
            deflector.wobble_enabled = False
    state.electron_gun.emitter.ray_count = args.rays
    state.step_mm = args.step
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    if args.candidate:
        _load_candidate(state, args.candidate)
    report.update(assembly=asdict(selection), voltage_kv=state.beam_voltage_kv,
        simulation_mode=state.simulation_mode,
        modes={"condenser": mode_key, "projector": state.projector_mode},
        projector_preset_applied=projector_applied,
        validation_plane="specimen entrance; downstream image-corrector/projector not qualified",
        nanoprobe_range=state.nanoprobe_range,
        static_drives={"ac_raster": False, "dc_raster": False, "wobble": False},
        lenses={lens.key: float(lens.percent) for lens in state.lenses},
        lens_polarities={lens.key: int(lens.polarity) for lens in state.lenses
                         if hasattr(lens, "polarity")},
        hexapoles={item.key: {"strength_m3": float(item.strength_m3),
                             "orientation_rad": float(item.orientation_rad)}
                   for item in state.corrector_elements if hasattr(item, "strength_m3")})
    with backend_receipts(report["backend_receipts"]):
        gun = trace_source_to_exit(state)
    bundle = gun.exit_bundle
    weights = np.asarray(bundle.weight, float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("The executed source must have finite nonnegative weights and positive current")
    current_rays = (weights > 0) & np.asarray(bundle.alive, bool)
    fraction = float(weights[current_rays].sum() / weights.sum())
    report["source"] = dict(representation=state.electron_gun.source_representation,
        requested_rays=args.rays, executed_rays=int(weights.size),
        positive_weight_rays=int(np.count_nonzero(weights > 0)),
        gun_exit_current_carrying_rays=int(current_rays.sum()),
        emitted_current_a=float(gun.emitted_current_a),
        gun_exit_source_current_fraction=fraction,
        gun_exit_current_a=float(gun.emitted_current_a * fraction))
    for step in (args.step, args.step / 2):
        print(f"Checking {args.rays} rays at step {step:g} mm...", flush=True)
        row = dict(step_mm=step, passed=False, backend_receipts=[], errors=[])
        report["step_checks"].append(row)
        with backend_receipts(row["backend_receipts"]):
            for name, operation in (
                ("topology", lambda: validate(state, step_mm=step)),
                ("surface_focus", lambda: measure_surface_focus(state, step_mm=step)),
            ):
                try:
                    measured = operation()
                    if name == "topology":
                        row[name] = measured
                    else:
                        row[name] = asdict(measured)
                        stats = measured.statistics
                        row[name].update(alpha95_mrad=stats.convergence_95_mrad,
                            sample_current_a=gun.emitted_current_a * stats.surviving_fraction,
                            passed=measured.accepts(args.target,
                                angle_relative_tolerance=.01, focus_tolerance_nm=1.))
                except Exception as error:
                    row["errors"].append(dict(stage=name, type=type(error).__name__, message=str(error)))
        row["actual_backends"] = _actual_backends(row["backend_receipts"])
        row["cpu_only"] = bool(row["actual_backends"]) and set(row["actual_backends"]) <= {"CPU", "Numba CPU"}
        row["passed"] = bool(not row["errors"] and row["cpu_only"]
            and row.get("topology", {}).get("passed", False)
            and row.get("surface_focus", {}).get("passed", False))
        print(f"Step {step:g} mm: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
    coarse, fine = report["step_checks"]
    if "surface_focus" in coarse and "surface_focus" in fine:
        left, right = (row["surface_focus"] for row in (coarse, fine))
        report["step_differences"] = dict(
            alpha95_mrad=abs(left["alpha95_mrad"] - right["alpha95_mrad"]),
            rms_radius_nm=abs(left["statistics"]["radius_rms_m"] - right["statistics"]["radius_rms_m"]) * 1e9,
            local_waist_offset_nm=abs(left["local_waist_offset_nm"] - right["local_waist_offset_nm"]))
    report["passed"] = all(row["passed"] for row in report["step_checks"])


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def main():
    args = _arguments()
    report = dict(schema="nanoprobe-condenser-validation-v1", passed=False,
        requested_backend="Numba CPU", particle_budget=args.rays,
        nanoprobe_range=args.range,
        requested_column=args.column,
        integration_steps_mm=[args.step, args.step / 2],
        target_alpha95_mrad=args.target, angle_relative_tolerance=.01,
        focus_tolerance_nm=1., minimum_surviving_and_effective_rays=16,
        candidate=str(args.candidate.resolve()) if args.candidate else None,
        calibration_reference_particle_budget=3000,
        sampling_convergence="NOT_TESTED_FIXED_PARTICLE_BUDGET",
        sampling_note="Changing the particle budget can change focus and alpha95; acceptance tolerances are unchanged",
        qualification="Classical particles at two integration steps; no wave-optical or OEM qualification",
        backend_receipts=[], step_checks=[], errors=[])
    started = perf_counter()
    if args.rays != 3000:
        print("Particle budget differs from the 3000-ray calibration; focus and alpha95 may differ. Acceptance tolerances are unchanged.", flush=True)
    try:
        _execute(args, report)
    except Exception as error:
        report["errors"].append(dict(type=type(error).__name__, message=str(error)))
        report["passed"] = False
    report["elapsed_s"] = perf_counter() - started
    receipts = report["backend_receipts"] + [receipt
        for row in report["step_checks"] for receipt in row["backend_receipts"]]
    report["actual_backends"] = _actual_backends(receipts)
    report["status"] = "PASS_AT_FIXED_PARTICLE_BUDGET" if report["passed"] else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: {args.output.resolve()}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
