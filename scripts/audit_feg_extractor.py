"""Classical FEI-inspired extractor audit; retain the full emitted population.

This explicitly selects the curved classical tip on a standalone gun. It does
not qualify the flat-tip default, a coherent beam, or downstream optical tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim import module_manifest
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_assembly import model_from_part


def effective_count(weight):
    return float(weight.sum()**2/np.dot(weight, weight)) if np.any(weight) else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--cap-half-angle-deg", type=float, default=10.)
    parser.add_argument("--cells-per-bore", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1000:
        parser.error("Physical trials require at least 1000 original positive-current particles")
    gun = FieldEmissionGun()
    model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    model = replace(model, emission=replace(model.emission, cap_half_angle_deg=args.cap_half_angle_deg))
    if args.cells_per_bore is not None:
        model = replace(model, field_numerics=replace(model.field_numerics,
                        electrode_cells_per_bore=args.cells_per_bore))
    gun.emitter.surface_model = model.validate()
    gun.emitter.ray_count = args.count
    emitted = gun.emit(args.count)
    positive = emitted.weight > 0
    if np.count_nonzero(positive) < 1000 or len(np.unique(emitted.ray_id[positive])) < 1000:
        raise ValueError("Actual original positive-current source population is below 1000")
    args.output.mkdir(parents=True, exist_ok=True)
    # Snapshot before executing; changing source support or mesh is a new trial.
    (args.output/"input.json").write_text(
        json.dumps(json.loads(gun._cache_key(args.count)), indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(args.output/"source.npz", ray_id=emitted.ray_id, weight=emitted.weight,
        position_m=emitted.surface_position_m, direction=emitted.surface_direction,
        energy_ev=emitted.surface_energy_ev)
    print(f"Tracing {positive.sum()} original positive-current particles; cap {args.cap_half_angle_deg:g} deg", flush=True)
    started = perf_counter()
    from temsim.physics.transport_progress import transport_progress
    with transport_progress(lambda message: print(str(message), flush=True)):
        result = gun.trace_to_exit(args.count)
    elapsed = perf_counter()-started
    output = result.exit_bundle
    alive = np.asarray(output.alive, bool)
    np.testing.assert_array_equal(result.emission_reference["position_m"], emitted.surface_position_m)
    np.testing.assert_array_equal(result.emission_reference["direction"], emitted.surface_direction)
    np.testing.assert_array_equal(output.ray_id, emitted.ray_id)
    np.testing.assert_array_equal(output.weight[alive], emitted.weight[alive])
    blocked = np.asarray(result.blocked_key)
    stops = {}
    for key, count in Counter(blocked[~alive]).items():
        selected = (~alive) & (blocked == key)
        z = result.blocked_z_mm[selected]
        stops[key or "unlabelled"] = {"particles": count,
            "current_a": float(gun.emitted_current_a*emitted.weight[selected].sum()),
            "z_min_mm": float(np.nanmin(z)), "z_max_mm": float(np.nanmax(z))}
    from temsim.extractor_geometry import ExtractorContour
    contour = ExtractorContour.from_component(gun.extractor)
    direction = emitted.surface_direction
    initial_angle = np.degrees(np.arctan2(np.hypot(direction[:, 0], direction[:, 1]), direction[:, 2]))
    # Only demonstrated passages count, not arbitrary non-extractor stops.
    passed_extractor = alive | (np.isfinite(result.blocked_z_mm) & (result.blocked_z_mm > contour.end_mm))
    sampled = initial_angle[passed_extractor & positive]
    report = dict(scope="standalone classical curved-tip physical trial; not OEM or full-column qualification",
        cap_half_angle_deg=args.cap_half_angle_deg, elapsed_seconds=elapsed,
        emitted_positive_particles=int(positive.sum()), emitted_effective_particles=effective_count(emitted.weight),
        exit_positive_particles=int(np.count_nonzero(alive & positive)),
        exit_effective_particles=effective_count(emitted.weight[alive]),
        emitted_current_a=gun.emitted_current_a,
        exit_current_a=float(gun.emitted_current_a*emitted.weight[alive].sum()),
        transmitted_fraction=float(emitted.weight[alive].sum()), blocked=stops,
        extractor_straight_ray_acceptance=contour.straight_ray_acceptance(),
        sampled_extractor_passage={"particles": int(len(sampled)),
            "initial_global_z_angle_min_deg": float(sampled.min()) if len(sampled) else None,
            "initial_global_z_angle_max_deg": float(sampled.max()) if len(sampled) else None,
            "interpretation": "Finite sample extrema, not the maximum accepted launch angle; emitting-cap polar angle differs from direction."},
        electrostatic_model=result.electrostatic_model_report,
        limitations=["One sampling/mesh point is not convergence.",
                     "Rear EEE/cathode support are axisymmetric engineering adaptations; EEE wiring is assumed. No dielectric polarization.",
                     "No cleaning filament, secondary/backscattered electrons, space charge or tunnelling current.",
                     "Undisclosed patent dimensions and downstream geometry remain engineering assumptions."])
    np.savez_compressed(args.output/"transport.npz", ray_id=output.ray_id, original_weight=emitted.weight,
        exit_weight=output.weight, alive=alive, blocked_key=blocked, blocked_z_mm=result.blocked_z_mm,
        exit_x_m=output.x_m, exit_y_m=output.y_m, exit_tx_rad=output.tx_rad, exit_ty_rad=output.ty_rad,
        exit_energy_offset_ev=output.energy_offset_ev)
    (args.output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"electrostatic_model", "limitations"}}, indent=2))


if __name__ == "__main__":
    main()
