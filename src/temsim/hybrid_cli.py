"""Retired Tip-origin hybrid CLI; use python -m temsim.mixed_wave_cli --help."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from temsim.physics.hybrid_pipeline import HybridGrid, HybridRequest, simulate_hybrid
from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics
from temsim.physics.tip_wave_pipeline import TipWaveRequest


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--profile", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True, help="New output directory; existing results are never overwritten")
    result.add_argument("--detector", required=True)
    result.add_argument("--sample-width-nm", type=float, required=True, help="Numerical wave field of view, not a fitted probe diameter")
    result.add_argument("--sample-pixels", type=int, default=256)
    result.add_argument("--hybrid-start-z-mm", type=float)
    result.add_argument("--intermediate-grids", type=Path, help="JSON list of HybridGrid records")
    result.add_argument("--source-particles", type=int, default=1024)
    result.add_argument("--exit-particles", type=int, default=4096)
    result.add_argument("--exit-window-nm", type=float, default=.5, help="Physical Gaussian position/momentum analysis width")
    result.add_argument("--source-grid-pixels", type=int, default=128)
    result.add_argument("--frame-stride", type=int, default=8)
    result.add_argument("--window-sigma-pixels", type=float, default=8.)
    result.add_argument("--maximum-packets", type=int, default=65536)
    result.add_argument("--step-mm", type=float, default=.1)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--inelastic", choices=("trajectories", "zero_loss"), default="trajectories")
    return result


def save_result(result, destination):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=False)
    arrays = {"exit_"+name: getattr(result.exit_particles, name) for name in
        ("position_m", "direction", "energy_ev", "weight", "mode_index")}
    arrays.update({"original_"+key: value for key, value in result.original_source.items()})
    arrays.update({"downstream_"+key: value for key, value in result.downstream.items() if isinstance(value, np.ndarray)})
    spectra = []
    for index, record in enumerate(result.diffraction):
        spectra.append({key: value for key, value in record.items() if not isinstance(value, np.ndarray)})
        arrays.update({f"diffraction_{index}_{key}": value for key, value in record.items() if isinstance(value, np.ndarray)})
    from temsim.immutable_json import thaw_json
    checkpoints = {}
    for label, checkpoint in (("incident", result.specimen_incident), ("exit", result.specimen_exit)):
        mode_records = []
        for index, mode in enumerate(checkpoint.beam.modes):
            names = {}
            for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
                value = getattr(mode.plane, name)
                if value is not None:
                    key = f"wave_{label}_{index}_{name}"
                    arrays[key], names[name] = np.asarray(value), key
            mode_records.append({"mode_id": mode.mode_id, "energy_kev": mode.energy_kev,
                "weight_per_reference_electron": mode.weight_per_reference_electron,
                "scattering_history": thaw_json(mode.scattering_history), "arrays": names,
                "axial_reference": None if mode.axial_reference is None else asdict(mode.axial_reference)})
        checkpoints[label] = {"digest": checkpoint.digest, "plane_z_mm": checkpoint.plane_z_mm,
            "reference_plane": checkpoint.beam.reference_plane, "reference_current_a": checkpoint.reference_current_a,
            "modes": mode_records}
    np.savez_compressed(path/"hybrid_arrays.npz", **arrays)
    report = {**result.report, "exit_sampling": thaw_json(result.exit_particles.record), "diffraction": spectra,
        "wave_checkpoints": checkpoints,
        "downstream_accounting": {key: value for key, value in result.downstream.items() if not isinstance(value, np.ndarray)},
        "specimen_incident": thaw_json(result.specimen_incident.record),
        "specimen_exit": thaw_json(result.specimen_exit.record)}
    (path/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    args = parser().parse_args(argv)
    from temsim.physics.retired_tip_wave import reject_retired_tip_coherent_workflow
    reject_retired_tip_coherent_workflow()
    if args.output.exists():
        raise ValueError("Output directory already exists; choose a new destination")
    from temsim.physics.compute_backend import require_coherent_cuda, validate_coherent_cuda_policy
    require_coherent_cuda()
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.profile_io import read_profile, apply_profile_values
    from temsim.column.state_layout import apply_physical_layout_to_state
    selection, values = read_profile(args.profile)
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.normalise_selection(selection))
    apply_profile_values(state, values)
    validate_coherent_cuda_policy(state.acceleration_backend,
        acceleration_enabled=state.acceleration_enabled)
    apply_physical_layout_to_state(state)
    wave = TipWaveRequest(stop="gun_exit")
    wave = replace(wave, source=replace(wave.source, grid_pixels=args.source_grid_pixels),
        inelastic=replace(wave.inelastic, method=args.inelastic))
    grids = () if args.intermediate_grids is None else tuple(HybridGrid(**row)
        for row in json.loads(args.intermediate_grids.read_text(encoding="utf-8")))
    request = HybridRequest(args.detector, HybridGrid(state.sample.z_mm-state.sample.thickness_nm*.5e-6,
        args.sample_pixels, args.sample_width_nm), args.source_particles, args.exit_particles, args.seed,
        SemiclassicalNumerics(frame_stride=args.frame_stride, window_sigma_pixels=args.window_sigma_pixels,
            maximum_packets=args.maximum_packets, maximum_step_m=args.step_mm*.001), wave,
        coherent_start_z_mm=args.hybrid_start_z_mm, intermediate_grids=grids, exit_window_sigma_nm=args.exit_window_nm)
    def progress(done, total, message):
        print(f"[{done}/{total}] {message}", flush=True)
    result = simulate_hybrid(state, request, progress_callback=progress)
    save_result(result, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "detector_current_a": result.report["detector_current_a"],
        "status": result.report["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
