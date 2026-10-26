"""Repeatable CUDA propagation of an unchanged, executed Si checkpoint.

This is a bounded stage benchmark, not a new source simulation. Run separate
processes with --source-root pointing at each revision's src directory. Saved
input, configuration and numerical settings are identical across revisions.
"""
from __future__ import annotations

import argparse
import cProfile
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True,
                        help="One current-schema captured instrument, identical for both revisions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("At least one timed repeat is required")
    sys.path.insert(0, str(args.source_root.resolve()))
    from temsim.physics.compute_backend import require_coherent_cuda
    require_coherent_cuda()
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT"):
        os.environ[key] = "16"
    for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    import cupy as cp
    import numpy as np
    import temsim
    from temsim.cpu_resources import numerical_job
    from temsim.immutable_json import thaw_json
    from temsim.instrument_snapshot import decode_instrument
    from temsim.physics.column_wave import _prepare_column, _propagate_column
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_device import release_device_memory
    from temsim.physics.wave_grid import WaveGridNumerics

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(temsim.__file__).resolve()
    if not source.is_relative_to(args.source_root.resolve()):
        raise RuntimeError(f"Wrong revision imported: {source}")
    report = json.loads((args.fixture / "report.json").read_text(encoding="utf-8"))
    row = next(r for r in report["results"] if r["label"] == "objective_diffraction")
    storage = row["record"]["storage"]
    began = time.perf_counter()
    store = ExecutedWaveStore(Path(storage["directory"]).parent, storage["dependency"], 192 * 1024**3)
    checkpoint = store.get(storage["key"])
    if checkpoint is None or checkpoint.digest != row["checkpoint_digest"]:
        raise ValueError("Saved executed input checkpoint differs from its receipt")
    index_seconds = time.perf_counter() - began
    instrument_data = args.instrument.read_bytes()
    state = decode_instrument(json.loads(instrument_data))
    numerics = WaveGridNumerics(**report["request"]["wave_grid"])
    if numerics.compute_backend != "Require GPU" or not numerics.acceleration_enabled:
        raise ValueError("This fixture must explicitly require CUDA")
    target = checkpoint.plane_z_mm + .001
    began = time.perf_counter()
    prepared = _prepare_column(state, checkpoint.plane_z_mm, target, report["request"]["column_step_mm"])
    prepare_seconds = time.perf_counter() - began
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    gpu = props["name"]
    gpu = gpu.decode() if isinstance(gpu, bytes) else str(gpu)
    times = []
    result = None

    def execute():
        nonlocal result
        result = None
        release_device_memory()
        cp.cuda.Stream.null.synchronize()
        with numerical_job(16):
            value = _propagate_column(state, checkpoint, target,
                maximum_step_mm=report["request"]["column_step_mm"],
                grid_numerics=numerics, _prepared=prepared)
            cp.cuda.Stream.null.synchronize()
        modes = value.record["modes"]
        if not modes or any(m["compute_backend"] != "cupy" or m.get("fallback_reason") for m in modes):
            raise RuntimeError("CUDA execution was not established for every mode")
        result = value

    try:
        for index in range(args.repeats + 1):
            began = time.perf_counter()
            execute()
            elapsed = time.perf_counter() - began
            times.append(elapsed)
            print(f"{'First' if index == 0 else 'Warmed'} propagation {index}: {elapsed:.6f} s", flush=True)
        if args.profile:
            profiler = cProfile.Profile()
            profiler.runcall(execute)
            profiler.dump_stats(str(output / "column.prof"))
            stats = pstats.Stats(profiler)
            stages = []
            for (file, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
                if "temsim" in file and cumulative >= .01:
                    stages.append(dict(file=file, line=line, name=name, calls=calls,
                                       own_seconds=own, inclusive_seconds=cumulative))
            stages.sort(key=lambda x: x["inclusive_seconds"], reverse=True)
            (output / "stages.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
        modes = []
        for index, mode in enumerate(result.beam.modes):
            plane = mode.plane
            np.save(output / f"mode-{index}-amplitude.npy", plane.amplitude, allow_pickle=False)
            modes.append(dict(mode_id=mode.mode_id, energy_kev=mode.energy_kev,
                weight=mode.weight_per_reference_electron, shape=list(plane.amplitude.shape),
                probability=plane.probability, basis_m=plane.basis_m.tolist(),
                origin_m=plane.origin_m.tolist(), curvature_m1=plane.curvature_m1.tolist(),
                tilt_rad=plane.tilt_rad.tolist()))
        measurement = dict(source_root=str(args.source_root.resolve()), loaded_package=str(source),
            gpu=gpu, cupy=cp.__version__, numpy=np.__version__, fixture=str(args.fixture.resolve()),
            fixture_source_identity=report["source_identity"], checkpoint_digest=checkpoint.digest,
            instrument_file=str(args.instrument.resolve()),
            instrument_sha256=hashlib.sha256(instrument_data).hexdigest(), numerics=asdict(numerics),
            input_z_mm=checkpoint.plane_z_mm, output_z_mm=target,
            physical_steps=len(prepared[0].step_m), index_seconds=index_seconds,
            prepare_seconds=prepare_seconds, propagation_seconds=times,
            warmed_median_seconds=statistics.median(times[1:]), modes=modes,
            execution_record=thaw_json(result.record),
            timing_scope="Same saved 3-mode Si boundary and same captured current instrument; "
                         "0.001 mm actual column continuation, disk checksum/load, "
                         "CPU/device transfers and output snapshots included. First run separate; warmed OS cache, "
                         "pool cleanup included and GPU synchronized. Profiling is a separate untimed pass. "
                         "No cold-disk or full-tip-to-detector speed claim.")
        (output / "measurement.json").write_text(json.dumps(measurement, indent=2), encoding="utf-8")
        print(json.dumps({k: measurement[k] for k in ("gpu", "prepare_seconds", "propagation_seconds",
                                                     "warmed_median_seconds", "instrument_sha256")}), flush=True)
    finally:
        release_device_memory()


if __name__ == "__main__":
    main()
