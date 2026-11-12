"""Run particle transport followed by near-specimen mixed coherence on CUDA."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def _positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Enter a positive finite value")
    return result


def _particle_count(value):
    count = int(value)
    if count < 1000:
        raise argparse.ArgumentTypeError("Use at least 1000 original positive-current particles")
    return count


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--result", type=Path, help="Existing checked .temresult particle archive")
    inputs.add_argument("--profile", type=Path, help="Input working point (.temwp); calculate particles first")
    result.add_argument("--particles", type=_particle_count, help="Required with --profile; original source count, at least 1000")
    result.add_argument("--quality", choices=("Preview", "High accuracy"), help="Particle calculation quality; profile default: High accuracy")
    result.add_argument("--step-mm", type=_positive, help="Explicit particle step override; otherwise retain the saved profile step")
    result.add_argument("--output", type=Path, required=True, help="New output directory; existing results are never overwritten")
    plane = result.add_mutually_exclusive_group()
    plane.add_argument("--z-mm", type=_positive, help="Exact observation plane at or downstream of the specimen upper face")
    plane.add_argument("--plane", choices=("upper", "exit"), help="Specimen face to observe; default: exit")
    result.add_argument("--pixels", type=int, default=512, help="Near-specimen numerical grid pixels per axis")
    result.add_argument("--width-nm", type=_positive, default=4., help="Numerical field of view, not a fitted probe diameter")
    result.add_argument("--source-image-rms-pm", type=float, default=10., help="Declared independent source-image translation RMS")
    result.add_argument("--maximum-modes", type=int, default=96)
    result.add_argument("--working-gib", type=_positive, default=24., help="Mixed-state, propagation and device working-memory budget")
    return result


def _particle_result(args, progress):
    if args.result is not None:
        from temsim.particle_section_io import load_section_result
        return load_section_result(args.result)
    from temsim.working_point import WorkingPointCheckpoint
    from temsim.simulation_pipeline import calculate
    from temsim.simulation_modes import require_coherent_mode

    state = WorkingPointCheckpoint.read_package(args.profile).compatible_state()
    require_coherent_mode(state)
    emitter = state.electron_gun.emitter
    emitter.ray_count = args.particles
    if args.step_mm is not None:
        state.step_mm = args.step_mm
    # The ordinary ray workflow owns the physical source and exact integration.
    # Do not apply the old small Preview population or a hidden step cap.
    state._tuning_quality = args.quality or "High accuracy"
    return calculate(state, workflow="rays", progress_callback=progress)


def _save_result(destination, result, source, inputs, request_record, *, particle_result=None):
    """Stream non-object arrays; no pickle or unbounded all-mode accumulator."""
    from temsim.immutable_json import thaw_json
    from temsim.physics.wave_plane_observables import canonical_angular_spectrum

    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    checkpoint = result.checkpoint
    canonical_shift = getattr(source, "canonical_frequency_shift_inv_angstrom", None)
    report = {"schema": "particle-near-specimen-mixed-wave-cli-v1", "status": "EXPORTING",
        "request": request_record, "instrument_digest": result.instrument_digest,
        "source_digest": source.digest, "handoff_z_mm": source.z_mm,
        "observation_z_mm": checkpoint.plane_z_mm,
        "reference_current_a": source.reference_current_a,
        "original_positive_particles": source.original_positive_count,
        "handoff_particle_count": int(len(source.ray_id)),
        "handoff_absolute_weight": float(source.source_fraction),
        "output_absolute_weight": float(checkpoint.beam.total_weight),
        "source_provenance": thaw_json(source.provenance),
        "canonical_frequency_initialization": {
            "shift_supplied": canonical_shift is not None,
            "array": "source_canonical_frequency_shift_inv_angstrom" if canonical_shift is not None else None,
            "frequency_units": "cycles/angstrom",
            "shift_formula": "electron q*A_xy/h, q=-e; SI inverse metres multiplied by 1e-10",
            "mechanical_trajectory_slopes_retained": True,
            "scope": "Wave canonical frequency receives the declared gauge shift; source tx_rad and ty_rad remain mechanical slopes"},
        "checkpoint_record": thaw_json(checkpoint.record),
        "coherent_backend_requirement": "CUDA; no CPU propagation fallback",
        "intensity_units": "probability per original reference electron per square metre",
        "diffraction_units": "probability per original reference electron per canonical angular bin",
        "diffraction_scope": "Complete-phase canonical angular spectrum; not kinetic angles inside a magnetic field",
        "phase_scope": "Phase is defined within each mode; independent modes add intensity",
        "modes": []}
    (path / "inputs.json").write_text(json.dumps(inputs.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        with ZipFile(path / "mixed_wave_arrays.npz", "w", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            def put(name, values):
                array = np.asarray(values)
                if array.dtype.hasobject:
                    raise ValueError("Mixed-wave export cannot contain Python object arrays")
                with archive.open(name + ".npy", "w", force_zip64=True) as handle:
                    np.lib.format.write_array(handle, array, allow_pickle=False)

            for name in ("x_m", "y_m", "tx_rad", "ty_rad", "ray_id", "weight", "kinetic_energy_ev", "flight_time_s"):
                put("source_" + name, getattr(source, name))
            if canonical_shift is not None:
                put("source_canonical_frequency_shift_inv_angstrom", canonical_shift)
            if particle_result is not None:
                gun = particle_result.simulation.gun_trace
                for name in ("ray_id", "weight", "alive"):
                    put("original_gun_" + name, getattr(gun.exit_bundle, name))
                put("original_gun_blocked_z_mm", gun.blocked_z_mm)
                put("original_gun_blocked_key", np.asarray(gun.blocked_key, dtype=str))
            for index, mode in enumerate(checkpoint.beam.modes):
                prefix = f"mode_{index:05d}_"
                wave = mode.plane
                for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                    if getattr(wave, name) is not None:
                        put(prefix + name, getattr(wave, name))
                density = mode.weight_per_reference_electron * abs(wave.amplitude) ** 2 / abs(np.linalg.det(wave.basis_m))
                put(prefix + "intensity_per_m2", density)
                spectrum = canonical_angular_spectrum(mode, maximum_working_bytes=result.request.maximum_readout_bytes)
                put(prefix + "canonical_angles_rad", spectrum["canonical_angles_rad"])
                put(prefix + "diffraction_probability", spectrum["probability_per_tip_electron"])
                report["modes"].append({"array_prefix": prefix, "mode_id": mode.mode_id,
                    "energy_kev": mode.energy_kev, "absolute_weight": mode.weight_per_reference_electron,
                    "scattering_history": thaw_json(mode.scattering_history)})
        report["status"] = "COMPLETED"
    except Exception as error:
        report.update(status="FAILED_DURING_EXPORT", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (path / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    from temsim.physics.compute_backend import require_coherent_cuda
    require_coherent_cuda("Require GPU")
    if args.output.exists():
        raise ValueError("Output directory already exists; choose a new destination")
    if args.profile is not None and args.particles is None:
        command.error("--profile requires --particles >= 1000")
    if args.result is not None and any(value is not None for value in (args.particles, args.quality, args.step_mm)):
        command.error("--result retains its executed particle settings; use --profile for a new particle calculation")
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.mixed_wave_particles import particle_source_from_result
    from temsim.physics.mixed_wave_pipeline import MixedWaveObservationSession, MixedWaveRequest
    from temsim.simulation_modes import require_coherent_mode

    default = MixedWaveRequest()
    budget = round(args.working_gib * 1024 ** 3)
    request = replace(default, observation_z_mm=args.z_mm,
        source=replace(default.source, pixels=args.pixels, width_nm=args.width_nm,
            source_image_rms_pm=args.source_image_rms_pm, maximum_modes=args.maximum_modes,
            maximum_working_bytes=budget),
        wave_grid=replace(default.wave_grid, maximum_working_bytes=budget,
            maximum_device_working_bytes=budget)).validate()
    started = perf_counter()
    def progress(done, total, message):
        print(json.dumps({"completed": done, "total": total, "message": str(message)}), flush=True)

    particles = _particle_result(args, progress)
    state = particles.state_snapshot
    require_coherent_mode(state)
    inputs = capture_instrument_snapshot(state)
    thickness = float(state.sample.thickness_nm) * .5e-6
    z_mm = args.z_mm if args.z_mm is not None else float(state.sample.z_mm) + (-thickness if args.plane == "upper" else thickness)
    request = replace(request, observation_z_mm=z_mm).validate()
    prepared_source = []
    def source_supplier():
        source = particle_source_from_result(particles, progress_callback=progress)
        prepared_source.append(source)
        return source

    session = MixedWaveObservationSession(state, request, source_supplier)
    try:
        result = session.observe(z_mm, progress_callback=progress)
        if len(prepared_source) != 1:
            raise ValueError("Mixed-wave calculation did not consume exactly one executed particle boundary")
        record = {"particle_input": str((args.result or args.profile).resolve()),
            "particle_input_kind": "result" if args.result else "profile",
            "requested_particles": args.particles, "particle_quality": args.quality or ("saved" if args.result else "High accuracy"),
            "requested_step_mm": args.step_mm, "executed_step_mm": float(state.step_mm),
            "mixed_wave": asdict(request), "elapsed_before_export_s": perf_counter() - started}
        report = _save_result(args.output, result, prepared_source[0], inputs, record, particle_result=particles)
    finally:
        session.close()
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"],
        "handoff_particles": report["handoff_particle_count"], "elapsed_s": perf_counter() - started}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
