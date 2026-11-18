"""Run stationary ideal-coil trials from the physical emitter and retain native focus planes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic
from types import SimpleNamespace

import numpy as np

from measure import desktop_state
from temsim.ideal_coil_segments import set_current_segments
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.physics.backend_execution import classical_backend_preflight
from temsim.physics.beam_statistics import _weighted_quantile, original_source_population, transverse_beam_statistics
from temsim.physics.particle_sections import _events, _trace_segment
from temsim.physics.ray_identity import emitted_source_identity
from temsim.physics.unified_field_runtime import unified_runtime_preflight

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from measure_probe_waist import observed_minimum, waist_separation


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plane_metrics(cp, index, weights, current):
    live = cp.unified_state.active[index] & (weights > 0)
    if not live.any():
        return dict(z_mm=float(cp.z_mm[index]), surviving=0, current_a=0.0)
    phase = [getattr(cp, key)[index, live] for key in ("x_m", "y_m", "tx_rad", "ty_rad")]
    if not all(np.isfinite(value).all() for value in phase):
        raise ValueError("Nonfinite live mechanical phase")
    w = weights[live]
    stats = transverse_beam_statistics(*phase, weights=w)
    radius = np.hypot(phase[0] - stats.mean_x_m, phase[1] - stats.mean_y_m)
    return dict(
        z_mm=float(cp.z_mm[index]),
        surviving=int(live.sum()),
        effective=float(w.sum() ** 2 / (w @ w)),
        current_a=float(current * w.sum()),
        rms_radius_nm=stats.radius_rms_m * 1e9,
        D95_nm=stats.radius_95_m * 2e9,
        D90_nm=_weighted_quantile(radius, w, .9) * 2e9,
        D50_nm=_weighted_quantile(radius, w, .5) * 2e9,
        alpha95_mrad=stats.convergence_95_mrad,
        angular_rms_mrad=stats.convergence_rms_rad * 1e3,
        core_50pm_current_a=float(current * w[np.hypot(phase[0], phase[1]) <= 50e-12].sum()),
        radial_covariance_m_rad=stats.radial_position_angle_covariance_m_rad,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--particles", type=int, default=1024)
    parser.add_argument("--step-mm", type=float, default=0.05)
    parser.add_argument("--resume-gun", type=Path)
    parser.add_argument("--gun-only", action="store_true")
    parser.add_argument("--tabulated-gun", action="store_true", help="Use the separately verified magnetic gun table for this study")
    args = parser.parse_args()
    if args.particles < 1000:
        parser.error("At least 1000 original positive-current particles are required.")
    out = args.output
    out.mkdir(parents=True, exist_ok=bool(args.resume_gun))
    _, state = desktop_state(args.particles)
    state.acceleration_backend = "Prefer GPU"
    state.step_mm = args.step_mm
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    surface = float(state.sample.upper_surface_z_mm)
    state.sample.inserted = False
    if args.recipe:
        recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
        for owner, groups in recipe["recipes"].items():
            set_current_segments(state, owner, groups)
        for owner, value in recipe.get("hexapole_strength_m3", {}).items():
            component = next(c for c in state.corrector_elements if c.key == owner)
            if not np.isfinite(value) or abs(value) > component.maximum_strength_m3:
                raise ValueError("Invalid physical hexapole command")
            component.strength_m3 = value
        for key, commands in recipe.get("multipole_commands", {}).items():
            owner = next(c for c in state.corrector_elements if c.key == key)
            for attr, value in commands.items():
                if attr not in ("strength_m3", "skew_strength_m3") or not hasattr(owner, attr):
                    raise ValueError("Unknown physical hexapole command")
                setattr(owner, attr, float(value))
            owner.validate()
    before = capture_instrument_snapshot(state)
    if (out / "inputs.json").exists() and json.loads((out / "inputs.json").read_text())[
        "digest"
    ] != before.digest:
        raise ValueError("Output folder contains a different input snapshot")
    write(out / "inputs.json", before.to_dict())
    classical_backend_preflight(state)
    issues = unified_runtime_preflight(state)["issues"]
    if issues:
        raise ValueError("; ".join(issues))
    started = monotonic()
    if args.resume_gun:
        cached = json.loads((args.resume_gun / "gun.json").read_text())
        if bool(cached.get("magnetic_table")) != args.tabulated_gun:
            raise ValueError("Select the same magnetic gun evaluation method as the checkpoint")
        if cached["input_digest"] != before.digest:
            raise ValueError("Saved gun belongs to different inputs")
        with np.load(args.resume_gun / "gun.npz") as data:
            arrays = {key: data[key] for key in data.files}
        # The first trial's already inspected checkpoint predates embedded hashes.
        expected = cached.get(
            "sha256", "8fa665031e0169882ccda4c0495f2061ed1252818e6760b7683960fd83aa7885"
        )
        if sha256((args.resume_gun / "gun.npz").read_bytes()).hexdigest() != expected:
            raise ValueError("Saved gun checksum mismatch")
        e = SimpleNamespace(**arrays)
        population = cached["population"]
        if not np.array_equal(e.ray_id, np.unique(e.ray_id)) or np.sum(e.weight > 0) != args.particles:
            raise ValueError("Incomplete saved original gun population")
        if "source_azimuth_rad" in arrays:
            azimuth = e.source_azimuth_rad
            labels = e.blocked_key
        else:
            # The first local checkpoint predates the display-azimuth field.
            # Its current and phase were saved in full; no ray was intercepted.
            if not np.all(e.alive):
                raise ValueError("The early gun checkpoint lacks physical stop labels")
            azimuth = np.full(len(e.ray_id), np.nan)
            labels = tuple("" for _ in e.ray_id)
        ids = (e.ray_id, azimuth)
        gun = SimpleNamespace(
            blocked_z_mm=e.blocked_z_mm,
            blocked_key=labels,
            emitted_current_a=float(e.emitted_current_a),
        )
        print("Resuming matching original gun", population, flush=True)
    else:
        print("Executing the original source in the candidate's all-Z field", flush=True)
        gun_evidence = {}
        if args.tabulated_gun:
            from gun_table import trace_gun
            gun, gun_evidence["magnetic_table"] = trace_gun(state)
        else:
            gun = trace_source_to_exit(state)
        population = original_source_population(gun)
        if population["positive_current_count"] != args.particles:
            raise ValueError("Original positive-current population mismatch")
        e = gun.exit_bundle
        ids = emitted_source_identity(gun)
        np.savez_compressed(
            out / "gun.npz",
            **{
                key: getattr(e, key)
                for key in (
                    "x_m",
                    "y_m",
                    "tx_rad",
                    "ty_rad",
                    "ray_id",
                    "weight",
                    "alive",
                    "energy_offset_ev",
                    "flight_time_s",
                )
            },
            blocked_z_mm=gun.blocked_z_mm,
            blocked_key=np.asarray(gun.blocked_key, dtype=str),
            source_azimuth_rad=ids[1],
            emitted_current_a=gun.emitted_current_a,
            **{"emission_" + key: value for key, value in gun.emission_reference.items()}
        )
        write(
            out / "gun.json",
            dict(
                population=population,
                seconds=monotonic() - started,
                input_digest=before.digest,
                sha256=sha256((out / "gun.npz").read_bytes()).hexdigest(),
                **gun_evidence,
            ),
        )
        print("Gun complete", population, flush=True)
    if args.gun_only:
        return
    from blocks import capture_blocks

    prefix_counts = [n for n in (1024, 2048, 4096, 8192, 16384) if n < args.particles]
    if prefix_counts and not np.array_equal(ids[0], np.arange(len(ids[0]))):
        raise ValueError("Nested diagnostic expects the original append-only ID order")

    def measure_row(cp, index, weights, current):
        row = plane_metrics(cp, index, weights, current)
        if prefix_counts and abs(cp.z_mm[index] - surface) <= 2.000001:
            row["nested"] = {
                str(n): plane_metrics(cp, index, weights * (ids[0] < n), current) for n in prefix_counts
            }
        return row

    report, rows = capture_blocks(
        state, gun, e, ids, out, before, surface, measure_row, observed_minimum, waist_separation
    )
    summaries = []
    for n in prefix_counts:
        selected = [r["nested"][str(n)] for r in rows if "nested" in r]
        if selected and all(r["surviving"] for r in selected):
            z = np.array([r["z_mm"] for r in selected])
            summaries.append(
                dict(
                    original_prefix=n,
                    surface=next(r for r in selected if r["z_mm"] == surface),
                    local_D95=observed_minimum(z, [r["D95_nm"] for r in selected], surface),
                    local_RMS=observed_minimum(z, [r["rms_radius_nm"] for r in selected], surface),
                )
            )
    report["nested_prefixes"] = summaries
    report.update(
        input_digest=before.digest,
        specimen_upper_surface_mm=surface,
        parked_scan=True,
        specimen_retracted_for_focus=True,
        population=population,
        full_profile=rows,
        elapsed_s=monotonic() - started,
    )
    write(out / "metrics.json", report)
    print("Surface", report["surface"], flush=True)
    if capture_instrument_snapshot(state).digest != before.digest:
        raise ValueError("Execution changed the candidate inputs")


if __name__ == "__main__":
    main()
