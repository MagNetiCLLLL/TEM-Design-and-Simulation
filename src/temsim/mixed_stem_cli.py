"""Acquire a CUDA mixed coherent STEM raster from a captured particle source."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--profile", type=Path, help="Working point; its selected particles are transported first")
    source.add_argument("--result", type=Path, help="Checked executed particle .temresult; retain its captured inputs")
    source.add_argument("--boundary", type=Path, help="Validated immutable particle phase-space boundary; no classical restart")
    result.add_argument("--output", type=Path, required=True, help="New output directory")
    result.add_argument("--particles", type=int, help="Explicit original source count, at least 1000; profile only")
    result.add_argument("--material-levels", help="Explicit grid:slices pairs, for example 1920:800,2160:1600")
    result.add_argument("--host-gib", type=float, help="Explicit host stage budget; does not change source or GPU settings")
    return result


def _write(path, value):
    from temsim.immutable_json import thaw_json
    Path(path).write_text(json.dumps(thaw_json(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def save_acquisition(acquisition, output):
    """Write detector and diffraction arrays with their unmodified source ledger."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    output = Path(output)
    files = ("scan-and-diffraction.npz", "entrance-source-arrays.npz",
             "original-gun.npz", "inputs.json", "report.json", "source-checkpoint.json",
             "checksums.json")
    if any((output/name).exists() for name in files):
        raise FileExistsError("Mixed scan output files already exist; choose a new destination")
    output.mkdir(parents=True, exist_ok=True)
    scan = acquisition.scan
    arrays = {"scan_x_nm": scan.scan_x_nm, "scan_y_nm": scan.scan_y_nm,
              "qx_inv_angstrom": scan.qx_inv_angstrom,
              "qy_inv_angstrom": scan.qy_inv_angstrom}
    if scan.diffraction_probability is not None:
        arrays["diffraction_probability"] = scan.diffraction_probability
    for key, values in scan.fractions.items():
        arrays["fraction_"+key] = values
    for key, values in scan.current_a.items():
        arrays["current_a_"+key] = values
    if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
        raise ValueError("Mixed scan exports require non-object arrays")
    np.savez_compressed(output/"scan-and-diffraction.npz", **arrays)
    checkpoint = scan.source_checkpoint
    source = checkpoint.auxiliary_arrays
    np.savez_compressed(output/"entrance-source-arrays.npz", **source)
    particles = acquisition.particle_result
    boundary = getattr(acquisition, "particle_boundary", None)
    if particles is not None:
        gun = particles.simulation.gun_trace
        original = {name: np.asarray(getattr(gun.exit_bundle, name)) for name in (
            "ray_id", "weight", "alive", "x_m", "y_m", "tx_rad", "ty_rad",
            "energy_offset_ev", "flight_time_s")}
        original.update(blocked_z_mm=gun.blocked_z_mm,
                        blocked_key=np.asarray(gun.blocked_key, dtype=str))
    elif boundary is not None:
        original = dict(boundary.original_gun)
    else:
        raise ValueError("Export requires the executed original gun ledger")
    np.savez_compressed(output/"original-gun.npz", **original)
    _write(output/"inputs.json", capture_instrument_snapshot(acquisition.state_snapshot).to_dict())
    _write(output/"report.json", scan.record)
    _write(output/"source-checkpoint.json", checkpoint.record)
    _write(output/"checksums.json", {name: sha256((output/name).read_bytes()).hexdigest()
                                    for name in files if name != "checksums.json"})


def main(argv=None):
    args = parser().parse_args(argv)
    from temsim.physics.compute_backend import require_coherent_cuda
    require_coherent_cuda("Require GPU")
    if args.output.exists():
        raise FileExistsError("Choose a new output directory; completed results are never overwritten")
    if args.particles is not None and (args.profile is None or args.particles < 1000):
        raise ValueError("--particles requires --profile and at least 1000 original source particles")
    if args.profile is not None:
        from temsim.working_point import WorkingPointCheckpoint
        state = WorkingPointCheckpoint.read_package(args.profile).compatible_state()
        if args.particles is not None:
            state.electron_gun.emitter.ray_count = args.particles
        existing = None
    elif args.result is not None:
        from temsim.particle_section_io import load_section_result
        existing = load_section_result(args.result)
        state = existing.state_snapshot
    else:
        from temsim.physics.particle_wave_source_io import load_particle_wave_source
        boundary = load_particle_wave_source(args.boundary)
        state, existing = boundary.restore_inputs_for_wave(), None
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    def progress(done, total, message):
        print(json.dumps({"completed": done, "total": total,
                          "elapsed_s": perf_counter()-started, "message": str(message)}), flush=True)
    from dataclasses import replace
    from temsim.physics.mixed_stem_scan import MixedStemScanRequest
    from temsim.mixed_stem_workflow import calculate_mixed_stem, calculate_mixed_stem_boundary
    request = MixedStemScanRequest()
    if args.material_levels is not None:
        levels = tuple(tuple(int(value) for value in pair.split(":")) for pair in args.material_levels.split(","))
        request = replace(request, material_levels=levels)
    if args.host_gib is not None:
        if not np.isfinite(args.host_gib) or args.host_gib <= 0:
            raise ValueError("Host budget must be finite and positive")
        request = replace(request, maximum_host_working_bytes=round(args.host_gib*1024**3))
    request.validate()
    try:
        acquisition = (calculate_mixed_stem_boundary(boundary, request=request, progress_callback=progress)
            if args.boundary is not None else calculate_mixed_stem(state, request=request,
                existing_result=existing, progress_callback=progress))
        save_acquisition(acquisition, args.output)
    except BaseException as error:
        _write(args.output/"status.json", {"status": "FAILED", "error": str(error),
                                           "error_type": type(error).__name__,
                                           "elapsed_s": perf_counter()-started})
        raise
    _write(args.output/"status.json", {"status": "COMPLETED", "elapsed_s": perf_counter()-started})
    print(json.dumps({"output": str(args.output.resolve()), "status": "COMPLETED"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
