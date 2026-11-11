"""Measure classical Microprobe illumination from one verified physical FEG source."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / name) for name in ("src", "scripts")]
FOCUS_DRIVER = ROOT / "outputs/fege-focus-20261110/focus/run_focus.py"
specification = importlib.util.spec_from_file_location("executed_focus_driver", FOCUS_DRIVER)
focus_driver = importlib.util.module_from_spec(specification)
specification.loader.exec_module(focus_driver)

from measure_probe_waist import PHASE, _hash, _json, _npz
from solve_joint_probe import prepare_objective_observations
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.beam_statistics import _weighted_quantile, transverse_beam_statistics
from temsim.physics.particle_sections import _selection_boundary
from validate_joint_probe import _restore_exact
from validate_physical_hybrid import topology_intervals


def observation_planes(state, topology_spacing_mm):
    surface = float(state.sample.upper_surface_z_mm)
    nearby = np.unique(np.r_[surface + np.array([-100., -10., -1., 0., 1., 10., 100.]) * 1e-3,
                             state.sample.z_mm, state.sample.lower_surface_z_mm])
    intervals = []
    rows = []
    for label, lower, upper, _ in topology_intervals(state):
        spacing = .05 if label == "ac_upper to ac_lower" else topology_spacing_mm
        if label.startswith("mini_condenser to "):
            upper = float(nearby[-1])
        intervals.append((label, lower, upper, spacing))
        rows.append(np.linspace(lower, upper, math.ceil((upper - lower) / spacing) + 1))
    apertures = [float(aperture.z_mm) for aperture in state.apertures
                 if state.electron_gun.exit_plane_z_mm < aperture.z_mm < nearby[-1]
                 and aperture.enabled and getattr(aperture, "installed", True)
                 and getattr(aperture, "inserted", True)]
    return nearby, intervals, np.unique(np.r_[nearby, np.concatenate(rows), apertures])


def plane_metrics(arrays, index, current):
    weights = arrays["weights"]
    selected = arrays["alive"][index] & (weights > 0.)
    w = weights[selected]
    result = dict(z_mm=float(arrays["z_mm"][index]), count=int(selected.sum()),
        effective_count=float(w.sum()**2 / np.dot(w, w)) if w.size else 0.,
        current_a=float(current * w.sum()), source_current_fraction=float(w.sum()))
    if not w.size:
        return result
    phase = np.column_stack([arrays[key][index, selected] for key in PHASE[:4]])
    if not np.isfinite(phase).all():
        raise ValueError("Nonfinite positive-current phase at an actual observation")
    stats = transverse_beam_statistics(*(arrays[key][index] for key in PHASE[:4]),
                                       alive=selected, weights=weights)
    normalized = w / w.sum()
    centered = phase - np.average(phase, axis=0, weights=w)
    position, slopes = centered[:, :2], centered[:, 2:]
    radius = np.linalg.norm(position, axis=1)
    spatial = np.einsum("n,ni,nj->ij", normalized, position, position)
    angular = np.einsum("n,ni,nj->ij", normalized, slopes, slopes)
    eigenvalues = np.maximum(np.linalg.eigvalsh(spatial), 0.)
    correlation_denominator = math.sqrt(float(np.trace(spatial) * np.trace(angular)))
    radius95_um = stats.radius_95_m * 1e6
    result.update(
        d50_um=float(2e6 * _weighted_quantile(radius, normalized, .50)),
        d90_um=float(2e6 * _weighted_quantile(radius, normalized, .90)),
        d95_um=float(2 * radius95_um), radial_rms_um=float(stats.radius_rms_m * 1e6),
        alpha_rms_mrad=float(stats.convergence_rms_rad * 1e3),
        alpha95_mrad=float(stats.convergence_95_mrad), alpha99_mrad=float(stats.convergence_99_mrad),
        maximum_sampled_angle_mrad=float(stats.convergence_edge_rad * 1e3),
        centroid_xy_um=[stats.mean_x_m * 1e6, stats.mean_y_m * 1e6],
        chief_slope_xy_mrad=[stats.mean_tx_rad * 1e3, stats.mean_ty_rad * 1e3],
        chief_angle_from_axis_mrad=float(np.arctan(np.hypot(stats.mean_tx_rad, stats.mean_ty_rad)) * 1e3),
        spatial_covariance_m2=spatial.tolist(), spatial_covariance_eigenvalues_m2=eigenvalues.tolist(),
        spatial_rms_principal_axes_um=(np.sqrt(eigenvalues) * 1e6).tolist(),
        spatial_rms_axis_ratio=(float(math.sqrt(eigenvalues[-1] / eigenvalues[0])) if eigenvalues[0] else None),
        mechanical_slope_covariance_rad2=angular.tolist(),
        mechanical_rms_slope_axes_mrad=(np.sqrt(np.maximum(np.diag(angular), 0.)) * 1e3).tolist(),
        radial_position_angle_covariance_m_rad=float(stats.radial_position_angle_covariance_m_rad),
        radial_position_angle_correlation=(float(stats.radial_position_angle_covariance_m_rad / correlation_denominator)
                                           if correlation_denominator else None),
        radial_wavefront_curvature_per_m=float(stats.radial_wavefront_curvature_per_m),
        rms_tangent_waist_offset_mm=float(stats.waist_offset_m * 1e3),
        d95_containment_disk_area_um2=float(np.pi * radius95_um**2),
        current_definition="Absolute original source weight reaching this actual plane",
        angular_definition="Mechanical direction about the current-weighted chief direction",
    )
    return result


def analyze(state, arrays, nearby, intervals, execution, *, topology):
    current = execution["source_current_a"]
    surface = float(state.sample.upper_surface_z_mm)
    exact = {float(value): int(np.searchsorted(arrays["z_mm"], value)) for value in nearby}
    profile = [plane_metrics(arrays, exact[float(value)], current) for value in nearby]
    for row in profile:
        row["offset_from_surface_um"] = (row["z_mm"] - surface) * 1e3
    losses = {}
    weights = arrays["weights"]
    keys = arrays["blocked_key"]
    if np.any(~arrays["terminal_alive"] & (keys == "")):
        raise ValueError("A stopped original source particle has no physical loss label")
    for key in sorted(set(keys) - {""}):
        mask = (keys == key) & (weights > 0.)
        losses[key] = dict(count=int(mask.sum()), current_a=float(current * weights[mask].sum()))
    total = float(current * weights.sum())
    terminal = float(current * weights[arrays["terminal_alive"]].sum())
    residual = total - terminal - sum(row["current_a"] for row in losses.values())
    if abs(residual) > 1e-12 * current:
        raise ValueError("Physical original-current balance does not close")
    result = dict(surface=next(row for row in profile if row["z_mm"] == surface), axial_profile=profile,
        incident_current_a=total, gun_exit_current_a=float(current * weights[arrays["gun_exit_alive"]].sum()),
        terminal_current_a=terminal, per_stage_losses=losses,
        absolute_current_balance_residual_a=float(residual),
        interpretation="Stationary incident illumination in the full current finite fields; no specimen scattering")
    if topology:
        detailed = focus_driver.analyze_topology(state, arrays, intervals, current)
        result["topology"] = detailed
        result["topology_summary"] = {
            label: {cohort: dict(status=evidence["status"], count=len(evidence["regions"]),
                centers_mm=[row["representative_z_mm"] for row in evidence["regions"]],
                unresolved=evidence["unresolved"]) for cohort, evidence in populations.items()}
            for label, populations in detailed.items()}
    return result


def await_controls(directory, trial, timeout_s, cancelled):
    path = directory / f"trial-{trial:02d}.json"
    until = time.perf_counter() + timeout_s
    while not cancelled() and time.perf_counter() < until:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8-sig")), path
        time.sleep(.5)
    return {}, path


def run(args):
    from temsim.cpu_resources import numerical_job
    from temsim.physics.transport_progress import transport_progress
    from temsim.profile_io import read_profile, save_profile

    if (not np.isfinite([args.step_mm, args.budget_s, args.topology_spacing_mm, args.inbox_timeout_s]).all()
            or min(args.step_mm, args.budget_s, args.topology_spacing_mm, args.inbox_timeout_s) <= 0.
            or not 1 <= args.threads <= 2 or not 1 <= args.trials <= 16):
        raise ValueError("Positive numerical limits, 1 to 16 trials and at most two threads are required")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    history = []
    report = dict(status="RUNNING", utc=datetime.now(timezone.utc).isoformat(),
        scope="Classical Microprobe illumination in current combined finite fields",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        driver_sha256=_hash(Path(__file__)), reused_focus_driver_sha256=_hash(FOCUS_DRIVER),
        gun_cache=str(args.gun_cache.resolve()), requested_step_mm=args.step_mm,
        qualification="EXPLORATORY", step_and_source_convergence="NOT_EVALUATED", trials=history)

    def write():
        _json(out / "report.json", report)

    def cancelled():
        return (time.perf_counter() - started >= args.budget_s or (out / "CANCEL").exists()
                or bool(args.stop_file and args.stop_file.exists()))

    def log(message):
        print(f"{time.perf_counter() - started:8.1f}s  {message}", flush=True)

    write()
    try:
        snapshot, gun, receipt = focus_driver.load_executed_gun(args.gun_cache)
        state = _restore_exact(snapshot)
        if state.mini_condenser.polarity != 1 or state.illumination_mode != "TEM":
            raise ValueError("The fresh source snapshot must already select positive-polarity Microprobe")
        surface = float(state.sample.upper_surface_z_mm)
        report["vacuum_optical_observation"] = dict(
            specimen_mode_before=state.sample.specimen_mode, cif_path_before=state.sample.cif_path,
            specimen_mode_after="vacuum", sample_upper_surface_z_mm=surface)
        state.sample.specimen_mode, state.sample.cif_path = "vacuum", ""
        if state.sample.upper_surface_z_mm != surface:
            raise ValueError("Vacuum observation changed the physical sample upper-surface reference")
        if any(getattr(state.sample, name, False) for name in ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled")):
            raise ValueError("Coherent products must be inactive for this classical illumination study")
        state.step_mm = args.step_mm
        report["control_seeds"] = []
        for path in args.controls:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            report["control_seeds"].append(dict(path=str(path.resolve()), sha256=_hash(path),
                changes=focus_driver.set_optics(state, payload["controls"])))
        prepare_objective_observations(state)
        nearby, intervals, planes = observation_planes(state, args.topology_spacing_mm)
        selection, _ = read_profile(args.gun_cache / "input.toml")
        report["observation_planes_mm"] = planes.tolist()
        report["microprobe_control"] = dict(mini_magnitude_percent=state.mini_condenser.percent,
            mini_field_polarity=state.mini_condenser.polarity,
            mini_signed_excitation_percent=state.mini_condenser.signed_excitation_percent)
        previous, changes = None, []
        previous_progress = [-30.]

        def progress(message):
            now = time.perf_counter() - started
            if now - previous_progress[0] >= 20.:
                log(str(message))
                previous_progress[0] = now

        with numerical_job(requested=args.threads, cancelled=cancelled) as cpu, transport_progress(progress):
            report["cpu_receipt"] = cpu.to_dict()
            for index in range(args.trials):
                case = out / f"trial-{index:02d}"
                case.mkdir()
                frozen = capture_instrument_snapshot(state)
                _json(case / "inputs.json", frozen.to_dict())
                save_profile(case / "input.toml", state, selection)
                drives = {item.key: {key: float(getattr(item, key)) for key in
                    ("percent", "strength_m3", "skew_strength_m3", "orientation_rad") if hasattr(item, key)}
                    for item in (*state.lenses, *state.corrector_elements) if item.enabled}
                report["running_trial"] = dict(index=index, input_digest=frozen.digest, controls=drives)
                write()
                log(f"Trial {index}: C2 {drives['condenser_lens_2']['percent']:.10f}%, "
                    f"C3 {drives['condenser_lens_3']['percent']:.10f}%")
                tick = time.perf_counter()
                changed_keys = tuple(sorted({row["key"] for row in changes if row["before"] != row["after"]}))
                boundary = _selection_boundary(state, changed_keys)
                arrays, execution, previous = focus_driver.execute_profile(state, planes, gun, receipt,
                    previous=previous, boundary=boundary, cancelled=cancelled)
                trial = analyze(state, arrays, nearby, intervals, execution, topology=args.topology)
                trial.update(index=index, controls=drives, execution=execution,
                             seconds=time.perf_counter()-tick, optical_changes=changes)
                selected = np.searchsorted(arrays["z_mm"], planes)
                retained = {key: (value[selected] if key in (*PHASE, "alive", "z_mm") else value)
                            for key, value in arrays.items()}
                _npz(case / "observations.npz", retained)
                trial["observations_sha256"] = _hash(case / "observations.npz")
                _json(case / "report.json", trial)
                history.append({key: value for key, value in trial.items() if key != "topology"})
                report.pop("running_trial")
                report["status"] = "AWAITING_CONTROLS" if args.inbox and index + 1 < args.trials else "RUNNING"
                write()
                summary = trial["surface"]
                log(f"Trial {index}: D95 {summary.get('d95_um')} um, alpha95 {summary.get('alpha95_mrad')} mrad, "
                    f"current {summary['current_a'] * 1e6:.8g} uA; resume {execution['column_resume_z_mm']} mm")
                del arrays, retained
                if not args.inbox or index + 1 == args.trials:
                    break
                command, path = await_controls(args.inbox.resolve(), index + 1, args.inbox_timeout_s, cancelled)
                if not command:
                    break
                changes = focus_driver.set_optics(state, command["controls"])
                report["next_control_file"] = dict(path=str(path), sha256=_hash(path))
        report["status"] = "COMPLETE"
    except (KeyboardInterrupt, InterruptedError):
        report["status"] = "CANCELLED"
        raise
    except Exception as error:
        report.update(status="ERROR", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gun-cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--controls", action="append", type=Path, default=[])
    parser.add_argument("--step-mm", type=float, default=.2)
    parser.add_argument("--topology-spacing-mm", type=float, default=2.)
    parser.add_argument("--topology", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--inbox", type=Path)
    parser.add_argument("--inbox-timeout-s", type=float, default=480.)
    parser.add_argument("--budget-s", type=float, default=5400.)
    parser.add_argument("--stop-file", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
