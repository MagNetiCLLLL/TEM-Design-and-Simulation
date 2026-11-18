"""Measure the unretuned desktop ideal-coil probe at the specimen entrance."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import monotonic

import numpy as np

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.particle_section_io import save_section_result
from temsim.physics.beam_statistics import original_source_population, transverse_beam_statistics
from temsim.round_lens_models import set_round_lens_field_model
from temsim.simulation_modes import switch_mode
from temsim.simulation_pipeline import calculate_particle_section


def desktop_state(particles):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    apply_operating_mode_pair(
        state,
        "nano_probe",
        "diffraction",
        column_name=selection.column,
        recording_name=selection.recording,
    )
    apply_physical_layout_to_state(state)
    state.step_mm = 0.05
    set_round_lens_field_model(state, "copper_coil")
    state.electron_gun.emitter.ray_count = particles
    return selection, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--particles", type=int, default=3000)
    parser.add_argument("--backend", choices=("Auto", "Prefer GPU"), default="Auto")
    args = parser.parse_args()
    if args.particles < 1000:
        parser.error("Use at least 1000 original positive-current particles.")
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    selection, state = desktop_state(args.particles)
    state.acceleration_backend = args.backend
    target = state.sample.upper_surface_z_mm
    inputs = capture_instrument_snapshot(state)
    (output / "inputs.json").write_text(json.dumps(inputs.to_dict()), encoding="utf-8")
    print(f"Transporting {args.particles} original particles to {target:.12f} mm", flush=True)
    last = 0.0

    def progress(done, total, message):
        nonlocal last
        if monotonic() - last > 60:
            step = getattr(message, "progress", None)
            detail = (
                f" ({step.local_completed / step.local_total:.1%})" if step and step.local_total else ""
            )
            print(f"{done}/{total} {message}{detail}", flush=True)
            last = monotonic()

    result = calculate_particle_section(
        state, target_z_mm=target, acquire_stem=False, progress_callback=progress
    )
    save_section_result(result, output / "entrance.temresult")
    sim = result.simulation
    gun = sim.gun_trace
    cp = sim.incident_checkpoints
    i = -1
    if abs(cp.z_mm[i] - target) > 4 * abs(np.spacing(target)):
        raise ValueError("Final solver checkpoint differs from the specimen upper surface")
    branch = sim.incident
    weights = np.asarray(branch.ray_weight)
    population = original_source_population(gun)
    if population["positive_current_count"] != args.particles:
        raise ValueError("Executed original positive-current count differs from the request")
    live = (weights > 0) & (branch.alive | (branch.blocked_z > target))
    physics = [getattr(cp, key)[i, live] for key in ("x_m", "y_m", "tx_rad", "ty_rad")]
    if not all(np.all(np.isfinite(a)) for a in physics):
        raise ValueError("Nonfinite coordinates in transmitted particles")
    w = weights[live]
    stats = transverse_beam_statistics(*physics, weights=w) if len(w) else None
    report = dict(
        assembly=asdict(selection),
        model=state.round_lens_field_model,
        input_digest=inputs.digest,
        specimen_upper_surface_mm=target,
        scan_enabled=state.ac_deflector.scan_enabled,
        descan_enabled=state.descan_deflector.scan_enabled,
        step_mm=state.step_mm,
        emitted_positive_count=population["positive_current_count"],
        surviving_positive_count=int(live.sum()),
        effective_count=float(w.sum() ** 2 / (w @ w)) if len(w) else 0.0,
        reference_current_a=gun.emitted_current_a,
        transmitted_current_a=gun.emitted_current_a * w.sum(),
        D95_nm=2e9 * stats.radius_95_m if stats else None,
        rms_radius_nm=1e9 * stats.radius_rms_m if stats else None,
        alpha95_mrad=stats.convergence_95_mrad if stats else None,
        angular_rms_mrad=1e3 * stats.convergence_rms_rad if stats else None,
        radial_covariance_m_rad=stats.radial_position_angle_covariance_m_rad if stats else None,
        centroid_nm=[1e9 * stats.mean_x_m, 1e9 * stats.mean_y_m] if stats else None,
    )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
