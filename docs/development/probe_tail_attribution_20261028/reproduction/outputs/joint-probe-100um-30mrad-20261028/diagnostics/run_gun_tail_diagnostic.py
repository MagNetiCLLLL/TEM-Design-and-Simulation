"""Prepared, opt-in gun-only step diagnostic; never a probe qualification.

Run only after the supervising workflow has released its physical-job slot.
This script uses the original 769-particle snapshot and changes only the gun
maximum trace step from 0.2 to 0.1 mm. It does not tune controls or change the
field grid, launch phase space, weights, energies, or physical stop rules.
Exact first crossings are produced by the production gun observer, not by
interpolating the retained display/equal-time histories.

Example (from the repository root, after explicit scheduling authorization):
  .venv/Scripts/python.exe outputs/joint-probe-100um-30mrad-20261028/diagnostics/run_gun_tail_diagnostic.py --threads 4 --output outputs/joint-probe-100um-30mrad-20261028/gun-step-diagnostic

The maximum-radius ray is not itself evidence of a long tail. Shape ratios
and its fraction of radial variance are reported at every actual plane.
Use --mode grid only after the step comparison: it observes native grids
16, 20 and 24 at a fixed gun step of 0.1 mm. The original grid is 16;
32 exceeds the existing vertex budget and is deliberately not requested.
No column propagation or monochromatic control experiment is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "agent.md").is_file() and (p / "src/temsim").is_dir())
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import numpy as np

SCHEMA = "temsim-no-retune-gun-step-diagnostic-v1"
DEFAULT_CANDIDATE = ROOT / "outputs/joint-probe-100um-30mrad-20261028/seed769-h02/best.json"
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")
STEPS_MM = (.2, .1)


def array_hash(values):
    array = np.ascontiguousarray(values)
    if array.dtype.hasobject:
        raise ValueError("Object arrays cannot define reproducible raw evidence")
    return sha256(str((array.dtype.str, array.shape)).encode() + array.tobytes()).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    from temsim.benchmark_suite import atomic_json
    atomic_json(path, value)


def write_npz(path, arrays):
    fd, name = tempfile.mkstemp(prefix=".gun-diagnostic-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def exact_planes(field, exit_mm):
    labels = {0.: ["emission"], float(exit_mm): ["gun_exit"]}
    selected = []
    for ring in field.request["rings"]:
        key = ring["key"]
        if key not in ("extractor", "electrostatic_lens") and not key.startswith("accelerator:"):
            continue
        selected.append(key)
        for face, name in (("start_m", "entrance"), ("stop_m", "exit")):
            z = float(ring[face]) * 1000.
            if not math.isfinite(z) or not 0 <= z <= exit_mm:
                raise ValueError("Physical gun electrode face is outside the gun interval")
            labels.setdefault(z, []).append(f"{key}:{name}")
    if not {"extractor", "electrostatic_lens"} <= set(selected) or not any(
            key.startswith("accelerator:") for key in selected):
        raise ValueError("Complete physical electrode geometry was not resolved")
    planes = np.asarray(sorted(labels), dtype=float)
    return planes, [dict(z_mm=float(z), labels=labels[z]) for z in planes]


def field_identity(electric, magnetic):
    from temsim.physics.closed_gun_field import ClosedGunField
    if not isinstance(electric, ClosedGunField):
        raise ValueError("This diagnostic is restricted to the original planar closed-gun field")
    identities = {name: getattr(magnetic, name, None)
                  for name in ("physical_identity", "numerical_identity")}
    if not all(isinstance(value, str) and value for value in identities.values()):
        raise ValueError("The complete instrument magnetic field has no reproducible identity")
    return dict(electric_type=type(electric).__name__, request_sha256=digest(electric.request),
                physical_request_sha256=digest({key: value for key, value in electric.request.items() if key != "numerics"}),
                radial_grid_sha256=array_hash(electric.r), axial_grid_sha256=array_hash(electric.z),
                voltage_sha256=array_hash(electric.voltage), grid_shape=list(electric.voltage.shape),
                magnetic=identities)


def memory_snapshot():
    import psutil
    virtual = psutil.virtual_memory()
    process = psutil.Process().memory_info()
    return dict(available_physical_bytes=int(virtual.available), total_physical_bytes=int(virtual.total),
                process_memory_bytes={key: int(value) for key, value in process._asdict().items()
                                      if key != "num_page_faults"})


def metrics(observed, weights, ids, emitted_current_a, labels):
    from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile
    if (weights.ndim != 1 or len(weights) != 769 or ids.shape != weights.shape
            or len(np.unique(ids)) != len(ids) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Invalid complete original source population")
    expected_shape = (len(observed.z_mm), len(ids))
    if any(np.asarray(getattr(observed, name)).shape != expected_shape
           for name in (*PHASE, "alive", "reached", "plane_residual_m")):
        raise ValueError("Exact observed arrays have inconsistent population or plane shapes")
    records = []
    for row, z in enumerate(observed.z_mm):
        active = np.asarray(observed.alive[row], bool) & (weights > 0)
        record = dict(z_mm=float(z), labels=labels[row]["labels"], surviving_rays=int(active.sum()),
                      source_current_fraction=float(weights[active].sum() / weights.sum()),
                      current_a=float(emitted_current_a * weights[active].sum() / weights.sum()),
                      active_ids_sha256=array_hash(ids[active]))
        if not active.any():
            records.append(dict(record, status="NOT_EVALUATED", reason="No positive-current survivors"))
            continue
        if (not observed.reached[row, active].all()
                or any(not np.isfinite(getattr(observed, name)[row, active]).all() for name in PHASE)
                or np.any(observed.kinetic_energy_ev[row, active] <= 0)
                or np.any(observed.flight_time_s[row, active] < 0)
                or not np.isfinite(observed.plane_residual_m[row, active]).all()
                or np.max(np.abs(observed.plane_residual_m[row, active])) > 1e-12):
            raise ValueError("Invalid actual positive-current exact-plane phase or residual")
        stats = transverse_beam_statistics(*(getattr(observed, name)[row] for name in PHASE[:4]),
                                            alive=active, weights=weights)
        normalized = weights[active] / weights[active].sum()
        radius = np.hypot(observed.x_m[row, active] - stats.mean_x_m,
                          observed.y_m[row, active] - stats.mean_y_m)
        diameters = {f"d{int(fraction * 100)}_nm": 2e9 * _weighted_quantile(radius, normalized, fraction)
                     for fraction in (.5, .8, .9, .95, .99)}
        variance = normalized * radius**2
        selected = np.flatnonzero(ids[active] == 306)
        index = int(selected[0]) if len(selected) else None
        id306 = dict(present=index is not None, radius_nm=None, radial_variance_fraction=None,
                     descending_radius_rank=None)
        if index is not None:
            id306.update(radius_nm=float(radius[index] * 1e9),
                         radial_variance_fraction=float(variance[index] / variance.sum()) if variance.sum() > 0 else None,
                         descending_radius_rank=int(1 + np.count_nonzero(radius > radius[index])))
        d50 = diameters["d50_nm"]
        records.append(dict(record, status="MEASURED", **diameters,
            rms_radius_nm=float(stats.radius_rms_m * 1e9), alpha95_mrad=float(stats.convergence_95_rad * 1e3),
            centroid_x_m=float(stats.mean_x_m), centroid_y_m=float(stats.mean_y_m),
            d95_over_d50=diameters["d95_nm"] / d50 if d50 > 0 else None,
            d99_over_d50=diameters["d99_nm"] / d50 if d50 > 0 else None,
            rms_over_d50=float(stats.radius_rms_m * 1e9) / d50 if d50 > 0 else None,
            effective_rays=float(1. / np.sum(normalized**2)), id306=id306,
            maximum_plane_residual_m=float(np.max(np.abs(observed.plane_residual_m[row, active])))))
    return records


def run(candidate, output, *, budget_s=1800., cancel_file=None, threads=4, mode="step"):
    from temsim.cpu_resources import numerical_job
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.electron_gun.tracing import trace_feg_with_observations, geometry_step_criteria
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.physics.closed_gun_field import closed_field_request, mesh_axes
    from temsim.physics.electrode_mesh import MAX_VERTICES
    from temsim.physics.ray_identity import emission_reference
    from temsim.vacuum import bind_gun_environment
    from validate_joint_probe import load_candidate, _restore_exact
    from solve_joint_probe import snapshot_changes

    if not math.isfinite(budget_s) or budget_s <= 0:
        raise ValueError("Budget must be positive and finite")
    if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 8:
        raise ValueError("Thread budget must be an integer from 1 through 8")
    if mode not in ("step", "grid"):
        raise ValueError("Mode must be step or grid")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    def cancelled():
        return perf_counter() - started >= budget_s or bool(cancel_file and Path(cancel_file).exists())
    def check_budget():
        if cancelled():
            raise InterruptedError("Cooperative gun diagnostic budget or cancellation requested")
    report = dict(schema=SCHEMA, status="RUNNING", qualification="NOT_QUALIFIED",
        model="classical_particles", scope="gun-only fixed-source diagnostic", mode=mode,
        source_count=769, gun_steps_mm=list(STEPS_MM) if mode == "step" else [.1], numerical_threads_requested=threads,
        unchanged_geometry_step_criteria=geometry_step_criteria(),
        cooperative_budget_s=budget_s, started_utc=datetime.now(timezone.utc).isoformat(), runs=[],
        limitations=["No column propagation, lens fitting, source editing, or focus qualification.",
            "Maximum radius alone is not evidence of a long tail or its cause.",
            "A two-step result is a bounded numerical comparison, not global convergence.",
            "The unchanged field grid and physical electrode approximation are not validated by step refinement.",
            "Budget checks are cooperative; compiled field/linear-algebra calls may return after the deadline.",
            "Retained display and equal-time histories are raw context only; all measured planes use exact gun observations."])
    if mode == "grid":
        report["limitations"][3] = "Native grid refinement does not validate the physical electrode/cathode approximation."
        report.update(grid_cells_per_bore=[16, 20, 24], existing_vertex_budget=MAX_VERTICES,
                      minimum_available_physical_bytes_before_field=16 * 1024**3,
                      memory_gate_scope="Operational headroom only; not a mathematical solver peak-memory bound")
    write_json(output / "report.json", report)
    try:
        state, provenance = load_candidate(candidate)
        original = capture_instrument_snapshot(state)
        gun = state.electron_gun
        original_cells = int(gun._gun_field_cells_per_bore)
        if (provenance["restoration_kind"] != "complete_snapshot" or gun.type_key != "cold_feg"
                or gun.ray_count != 769 or gun.trace_step_mm != .2
                or gun.source_representation != "classical_particles"
                or gun.emitter.surface_model is not None or gun.emitter.curvature_nm_inv != 0.
                or gun.monochromator_installed):
            raise ValueError("Expected the complete original 769-ray, 0.2-mm, flat classical cold-FEG seed")
        if mode == "grid" and original_cells != 16:
            raise ValueError("Native grid comparison requires the actual original 16-cell candidate")
        report.update(candidate=provenance, original_input_digest=original.digest,
                      execution_implementation=original.to_dict()["implementation"],
                      archived_candidate_implementation=json.loads(Path(provenance["path"]).read_text(encoding="utf-8"))["implementation"],
                      script_sha256=file_hash(__file__))
        write_json(output / "original-inputs.json", original.to_dict())
        reference_field = reference_source = reference_planes = reference_actual_launch = None
        with numerical_job(threads, cancelled=cancelled) as cpu:
            report["cpu_resources"] = cpu.to_dict()
            variants = [(step, original_cells) for step in STEPS_MM] if mode == "step" else [(.1, cells) for cells in (16, 20, 24)]
            for ordinal, (step, cells) in enumerate(variants, 1):
                check_budget()
                work = _restore_exact(original)
                work.electron_gun.trace_step_mm = step
                work.electron_gun._gun_field_cells_per_bore = cells
                snapshot = capture_instrument_snapshot(work)
                proof = _restore_exact(snapshot)
                proof.electron_gun.trace_step_mm = .2
                proof.electron_gun._gun_field_cells_per_bore = original_cells
                if capture_instrument_snapshot(proof).digest != original.digest:
                    raise ValueError("A setting other than the explicit gun step/grid inputs changed")
                directory = output / (f"step-{ordinal}" if mode == "step" else f"grid-{cells}")
                directory.mkdir()
                write_json(directory / "inputs.json", snapshot.to_dict())
                entry = dict(status="RUNNING", trace_step_mm=step, cells_per_bore=cells, input_digest=snapshot.digest,
                             explicit_input_changes=snapshot_changes(original.to_dict()["graph"], snapshot.to_dict()["graph"]))
                report["runs"].append(entry)
                write_json(output / "report.json", report)
                bind_gun_environment(work)
                gun = work.electron_gun
                launch = gun.emit()
                launch_reference = emission_reference(launch)
                arrays = {f"launch_{field.name}": np.array(getattr(launch, field.name), copy=True) for field in fields(launch)}
                arrays.update({f"emission_{key}": np.array(value, copy=True) for key, value in launch_reference.items()})
                arrays["launch_kinetic_energy_ev"] = gun.emitter.emission_energy_ev + launch.energy_offset_ev
                source_identity = {key: array_hash(value) for key, value in arrays.items()}
                if (not np.isfinite(arrays["launch_kinetic_energy_ev"]).all()
                        or np.any(arrays["launch_kinetic_energy_ev"] <= 0)):
                    raise ValueError("The original source contains invalid launch energies")
                if reference_source is not None and source_identity != reference_source:
                    raise ValueError("The original spatial/angular/energy source, IDs, or weights changed")
                reference_source = source_identity
                write_npz(directory / "launch.npz", arrays)
                check_budget()
                with instrument_gun_field_context(work) as magnetic:
                    expected_request = closed_field_request(gun)
                    if expected_request["numerics"]["cells_per_bore"] != cells:
                        raise ValueError("The executed field request ignored the explicit grid setting")
                    radial, axial = mesh_axes(expected_request)
                    if len(radial) * len(axial) > MAX_VERTICES:
                        raise ValueError("The native field exceeds its unchanged vertex budget")
                    before_field = memory_snapshot()
                    entry.update(before_field_memory=before_field,
                                 grid_preflight=dict(radial_nodes=len(radial), axial_nodes=len(axial),
                                                     vertices=len(radial)*len(axial)))
                    write_json(output / "report.json", report)
                    if mode == "grid" and before_field["available_physical_bytes"] < report["minimum_available_physical_bytes_before_field"]:
                        raise MemoryError("Insufficient available physical memory for the declared grid headroom policy")
                    field_started = perf_counter()
                    electric = gun.base_electric_field
                    entry.update(field_prepare_elapsed_s=perf_counter()-field_started, after_field_memory=memory_snapshot())
                    identity = field_identity(electric, magnetic)
                    if identity["request_sha256"] != digest(expected_request):
                        raise ValueError("The executed potential differs from its exact preflight request")
                    planes, labels = exact_planes(electric, gun.exit_plane_z_mm)
                    if reference_field is not None:
                        if mode == "step" and identity != reference_field:
                            raise ValueError("The actual electric grid/voltage/request or complete magnetic field changed")
                        if mode == "grid" and any(identity[key] != reference_field[key]
                                                  for key in ("physical_request_sha256", "magnetic")):
                            raise ValueError("Grid refinement changed a physical field boundary or magnetic contributor")
                    if reference_planes is not None and not np.array_equal(planes, reference_planes):
                        raise ValueError("The exact physical electrode observation planes changed")
                    reference_field, reference_planes = identity, planes.copy()
                    write_json(directory / "field.json", dict(identity=identity, request=electric.request, report=electric.report))
                    check_budget()
                    trace_started = perf_counter()
                    traced, observed = trace_feg_with_observations(gun, planes, cancelled=cancelled)
                    trace_elapsed = perf_counter() - trace_started
                    if field_identity(electric, magnetic) != identity:
                        raise ValueError("The executed field changed during gun tracing")
                for key, expected in launch_reference.items():
                    if traced.emission_reference is None or array_hash(traced.emission_reference[key]) != array_hash(expected):
                        raise ValueError("The actual trace did not use the saved original emission")
                for key in ("weight", "ray_id"):
                    if array_hash(getattr(traced.exit_bundle, key)) != array_hash(getattr(launch, key)):
                        raise ValueError("The gun changed the original raw weights or particle identity")
                if (observed.z_mm[0] != 0 or not observed.reached[0].all()
                        or not np.allclose(observed.kinetic_energy_ev[0], arrays["launch_kinetic_energy_ev"],
                                           rtol=0., atol=1e-9)):
                    raise ValueError("Actual launch kinetic energies disagree with the preserved source")
                # K=(gamma-1)mc^2 has rest-energy-scaled cancellation near 0.3 eV.
                # The small round-trip allowance above is not an inter-run
                # tolerance: actual retained launch arrays must match exactly.
                actual_launch = {name: array_hash(getattr(observed, name)[0])
                                 for name in (*PHASE, "alive", "reached")}
                if reference_actual_launch is not None and actual_launch != reference_actual_launch:
                    raise ValueError("The actual retained launch phase changed between steps")
                reference_actual_launch = actual_launch
                if capture_instrument_snapshot(work).digest != snapshot.digest:
                    raise ValueError("The gun trace changed the frozen instrument input graph")
                for field in fields(observed):
                    value = getattr(observed, field.name)
                    if isinstance(value, np.ndarray):
                        arrays[f"observed_{field.name}"] = value
                for field in fields(traced.exit_bundle):
                    value = getattr(traced.exit_bundle, field.name)
                    if value is not None:
                        arrays[f"exit_{field.name}"] = np.asarray(value)
                for key in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"):
                    value = getattr(traced, key)
                    if value is not None:
                        arrays[f"display_history_{key}"] = np.asarray(value)
                if traced.equal_time_history is not None:
                    for field in fields(traced.equal_time_history):
                        arrays[f"equal_time_history_{field.name}"] = np.asarray(getattr(traced.equal_time_history, field.name))
                arrays.update(blocked_z_mm=np.asarray(traced.blocked_z_mm), blocked_key=np.asarray(traced.blocked_key, dtype=str))
                write_npz(directory / "particles.npz", arrays)
                entry.update(status="COMPLETE", trace_elapsed_s=trace_elapsed, after_trace_memory=memory_snapshot(), exact_plane_schema=observed.schema,
                    field_identity=identity, source_array_fingerprints=source_identity,
                    actual_launch_fingerprints=actual_launch, launch_energy_round_trip_allowance_ev=1e-9,
                    source_current_a=float(traced.emitted_current_a), planes=labels,
                    raw_file_sha256=file_hash(directory / "particles.npz"), raw_array_fingerprints={key: array_hash(value) for key, value in arrays.items()},
                    electrostatic_model_report=traced.electrostatic_model_report,
                    vacuum_report=traced.vacuum_report, source_count=len(launch.ray_id),
                    measurements=metrics(observed, np.asarray(launch.weight), np.asarray(launch.ray_id), traced.emitted_current_a, labels))
                write_json(directory / "receipt.json", entry)
                write_json(output / "report.json", report)
            comparisons = []
            names = ("d50_nm", "d90_nm", "d95_nm", "d99_nm", "rms_radius_nm", "alpha95_mrad", "current_a", "d95_over_d50")
            for before, after in zip(report["runs"][:-1], report["runs"][1:]):
                deltas = []
                for left, right in zip(before["measurements"], after["measurements"]):
                    row = dict(z_mm=left["z_mm"], same_active_ids=left["active_ids_sha256"] == right["active_ids_sha256"],
                               baseline_status=left["status"], refined_status=right["status"])
                    row["changes"] = {key: dict(absolute=right[key] - left[key], relative=(right[key] - left[key]) / abs(left[key]) if left[key] else None)
                                      for key in names if left.get(key) is not None and right.get(key) is not None}
                    deltas.append(row)
                comparisons.append(dict(before=dict(step_mm=before["trace_step_mm"], cells_per_bore=before["cells_per_bore"]),
                                        after=dict(step_mm=after["trace_step_mm"], cells_per_bore=after["cells_per_bore"]), per_plane=deltas))
            report.update(status="COMPLETE", comparison=dict(status="MEASURED_NOT_QUALIFIED",
                same_source=True, same_field_grid_and_voltages=mode == "step", same_complete_magnetic_field=True,
                same_physical_field_boundaries=True, same_exact_observation_planes=True, transitions=comparisons))
    except BaseException as error:
        report.update(status="CANCELLED" if isinstance(error, (KeyboardInterrupt, InterruptedError)) or cancelled() else "FAILED",
                      error=dict(type=type(error).__name__, message=str(error)))
        for entry in report["runs"]:
            if entry["status"] == "RUNNING":
                entry["status"] = report["status"]
        raise
    finally:
        report["elapsed_s"] = perf_counter() - started
        write_json(output / "report.json", report)
        write_json(output / "checksums.json", {str(path.relative_to(output)).replace("\\", "/"): file_hash(path)
                   for path in sorted(output.rglob("*")) if path.is_file() and path.name != "checksums.json"})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-s", type=float, default=1800.)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", choices=("step", "grid"), default="step")
    args = parser.parse_args(argv)
    run(args.candidate, args.output, budget_s=args.budget_s, cancel_file=args.cancel_file, threads=args.threads, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
