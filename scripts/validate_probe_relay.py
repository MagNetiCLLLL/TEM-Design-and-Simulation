"""Validate current nanoprobe defaults with physical Tip rays on CUDA.

No settings are fitted or saved. Current catalog optics are tested in custom
mode, including Cs, hexapoles and actual source energies. The optional
linear/Cs-only/Cs-plus-hexapole comparison is separate evidence. This is a bounded
particle validation, not wave-image or experimental microscope qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


LIMITS = dict(hp_b_norm_m_per_rad=2e-5, hp_magnification_error=.02,
              sample_rms_nm=.2, sample_diameter95_nm=.7,
              sample_waist_offset_nm=2., step_radius_difference_nm=.005,
              step_waist_difference_nm=.5, sample_alpha95_mrad=30.,
              sample_alpha95_relative_error=.01)


def save_report(path, report):
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def subset(checkpoints, selection):
    return SimpleNamespace(z_mm=checkpoints.z_mm[selection], **{
        name: getattr(checkpoints, name)[selection]
        for name in ("x_m", "y_m", "tx_rad", "ty_rad")})


def statistics(checkpoints, index, weights, alive):
    from temsim.physics.beam_statistics import transverse_beam_statistics
    return transverse_beam_statistics(
        checkpoints.x_m[index], checkpoints.y_m[index],
        checkpoints.tx_rad[index], checkpoints.ty_rad[index],
        weights=weights, alive=alive)


def source_identity(bundle):
    digest = hashlib.sha256()
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "alive", "energy_offset_ev"):
        array = np.ascontiguousarray(getattr(bundle, name))
        digest.update(str((name, array.dtype.str, array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def surface_measurement(checkpoints, masks, weights, planes):
    from temsim.optics.surface_probe_focus import local_waist_from_radii
    indices = [int(np.argmin(abs(checkpoints.z_mm - z))) for z in planes]
    if not np.allclose(checkpoints.z_mm[indices], planes, rtol=0, atol=1e-10):
        raise ValueError("Missing exact specimen entrance checkpoints")
    live = np.all(masks[indices], axis=0) & (weights > 0)
    rows = [statistics(checkpoints, i, weights, live) for i in indices]
    radii = [row.radius_rms_m for row in rows]
    offset, second = local_waist_from_radii(radii, 100e-9)
    result = dict(z_mm=float(planes[-1]), statistics=asdict(rows[-1]),
                  rms_radius_nm=rows[-1].radius_rms_m*1e9,
                  diameter95_nm=2*rows[-1].radius_95_m*1e9,
                  covariance_waist_offset_nm=rows[-1].waist_offset_m*1e9,
                  derivative_waist_offset_nm=offset,
                  variance_second_derivative=second,
                  upstream_rms_radii_nm=(np.asarray(radii)*1e9).tolist())
    return result


def assert_cuda(rows):
    columns = [row for row in rows if row["stage"] == "column"]
    if not columns or any(row["actual"] != "CUDA GPU" or row["outcome"] != "returned"
                          for row in columns):
        raise RuntimeError("The physical column did not complete on CUDA GPU")


def trace(state, planes, step, receipts):
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.physics.backend_execution import backend_receipts
    batch = []
    with backend_receipts(batch):
        result = incident_checkpoints(state, planes, step_mm=step)
    receipts.extend(dict(row, maximum_step_mm=step) for row in batch)
    assert_cuda(batch)
    return result


def compare_aberrations(state, report):
    """Keep all round-lens settings and the same physical emitted population."""
    from temsim.optics.probe_calibration import IncidentProbeModel
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.beam_statistics import transverse_beam_statistics
    # The entire validation state is already detached and custom. Per-trace
    # operator switches preserve its optics and cached physical emission.
    if state.simulation_mode != "custom":
        raise ValueError("Aberration comparison requires the custom validation state")
    nonlinear = state
    executions = []
    with backend_receipts(executions):
        model = IncidentProbeModel(nonlinear)
        if source_identity(model.emitted) != report["physical_source"]["exit_bundle_sha256"]:
            raise RuntimeError("Changing model fidelity changed the physical source population")
        cases = dict(
            linear=model.trace(spherical=False, hexapoles=False, step_mm=.025),
            cs_without_hexapoles=model.trace(spherical=True, hexapoles=False, step_mm=.025),
            cs_with_hexapoles=model.trace(spherical=True, hexapoles=True, step_mm=.025),
        )
    assert_cuda(executions)
    report["backend_receipts"].extend(executions)
    first = cases["linear"]
    if any(not np.array_equal(row.weights, first.weights) for row in cases.values()):
        raise RuntimeError("Aberration comparison changed physical source weights")
    live = np.logical_and.reduce([row.alive for row in cases.values()]) & (first.weights > 0)
    if np.count_nonzero(live) < 5:
        raise RuntimeError("Insufficient common current for aberration comparison")
    weights = first.weights[live]/first.weights[live].sum()
    measured = {}
    for name, row in cases.items():
        stats = transverse_beam_statistics(row.x_m, row.y_m, row.tx_rad, row.ty_rad,
                                          weights=row.weights, alive=live)
        displacement = (row.x_m[live]-first.x_m[live])**2 + (row.y_m[live]-first.y_m[live])**2
        measured[name] = dict(statistics=asdict(stats), rms_radius_nm=stats.radius_rms_m*1e9,
                             nonlinear_displacement_rms_nm=float(np.sqrt(weights@displacement)*1e9))
    report["nonlinear_comparison"] = dict(
        mode="custom; identical source and round-lens settings; same surviving weighted cohort",
        z_mm=float(state.sample.upper_surface_z_mm), coordinate_precision="existing particle probe diagnostic",
        common_rays=int(np.count_nonzero(live)),
        source_current_fraction=float(first.weights[live].sum()/first.weights.sum()),
        cases=measured,
        hexapoles_reduce_nonlinear_displacement=(
            measured["cs_with_hexapoles"]["nonlinear_displacement_rms_nm"]
            < measured["cs_without_hexapoles"]["nonlinear_displacement_rms_nm"]),
    )


def validate(args, report):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.optics.direct_alignment import canonical_transfers
    from temsim.optics.beam_path_audit import crossover_candidates
    from temsim.physics.compute_backend import choose_ray_backend
    from temsim.physics.corrector_crossovers import measured_corrector_crossovers

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    apply_operating_mode_pair(state, "nano_probe", "diffraction")
    # Require active nonlinear optics; ideal mode suppresses Cs and hexapoles
    # and uses nominal reference energy, so it cannot qualify this calibration.
    state.simulation_mode = "custom"
    original_scan_flags = dict(ac_scan=bool(state.ac_deflector.scan_enabled),
                               descan_scan=bool(state.descan_deflector.scan_enabled),
                               ac_wobble=bool(state.ac_deflector.wobble_enabled))
    # A static calibration bench must not consume an unprepared raster drive.
    # Leave the physical coils enabled and preserve all static field biases.
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    state.ac_deflector.wobble_enabled = False
    state.electron_gun.emitter.ray_count = args.rays
    state.acceleration_enabled = True
    state.acceleration_backend = "Require GPU"
    state.step_mm = .05
    if state.electron_gun.source_representation != "classical_particles":
        raise ValueError("This validator requires the physical particle Tip source")
    selected, reason = choose_ray_backend("Require GPU", acceleration_enabled=True, ray_count=args.rays)
    if selected != "CUDA GPU" or reason:
        raise RuntimeError("CUDA GPU must be available before starting this validation")

    assembly = state._resolved_assembly
    lower = float(assembly.part("probe_tl12_lens").end_z_mm)
    upper = float(assembly.part("mini_condenser").start_z_mm)
    if not lower < upper:
        raise ValueError("TL1 exit must precede Mini Condenser entrance")
    sample = float(state.sample.upper_surface_z_mm)
    sample_planes = np.asarray([sample-.0002, sample-.0001, sample])
    snapshot = capture_instrument_snapshot(state)
    (args.output / "inputs.json").write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    report.update(input_identity=snapshot.digest, model_mode=state.simulation_mode,
                  nominal_voltage_kv=state.beam_voltage_kv, emitted_rays=args.rays,
                  interval_mm=[lower, upper], specimen_entrance_mm=sample,
                  lenses={lens.key: dict(percent=float(lens.percent), z_mm=float(lens.z_mm))
                          for lens in state.lenses},
                  backend_receipts=[], nonlinear_comparison={"status": "NOT_RUN"})
    report["protocol"] = dict(
        kind="detached static beam validation",
        original_scan_flags=original_scan_flags,
        dynamic_drive="AC raster, Descan raster and AC wobble disabled on validation state only",
        physical_coils="enabled flags and static biases preserved",
        scan_validation="NOT_RUN; requires separate calibrated scan drive")

    hp2, hp1 = state.hp2_hexapole, state.hp1_hexapole
    transfer = canonical_transfers(state, hp2.z_mm, [hp1.z_mm])[float(hp1.z_mm)]
    magnifications = np.linalg.svd(transfer.j_img, compute_uv=False)
    hp_b = float(np.linalg.norm(transfer.j_diff_m_per_rad, ord=2))
    report["hp2_to_hp1"] = dict(
        source_z_mm=float(hp2.z_mm), target_z_mm=float(hp1.z_mm),
        input_basis=transfer.input_basis, output_basis="laboratory mechanical slopes",
        matrix=transfer.matrix.tolist(), b_norm_m_per_rad=hp_b,
        magnifications=magnifications.tolist(),
        passed=bool(hp_b <= LIMITS["hp_b_norm_m_per_rad"]
                    and np.max(abs(magnifications-1)) <= LIMITS["hp_magnification_error"]),
    )
    print("Tracing physical Tip and coarse TL1-CM checkpoints", flush=True)
    planes = np.unique(np.r_[np.arange(lower, upper, .5), upper, sample_planes])
    gun, cp, masks = trace(state, planes, .05, report["backend_receipts"])
    report["physical_source"] = dict(
        parameters=state.electron_gun.to_dict(),
        exit_bundle_sha256=source_identity(gun.exit_bundle),
        source_backend="CPU emitter-to-gun-exit implementation; exact reuse on subsequent traces")
    in_interval = (cp.z_mm >= lower) & (cp.z_mm <= upper)
    proposals = crossover_candidates(subset(cp, in_interval), masks[in_interval], gun.exit_bundle.weight)
    proposals = [row for row in proposals if lower+.001 < row["z_mm"] < upper-.001]
    report["coarse_crossover_proposals"] = proposals
    report["coarse_sample"] = surface_measurement(cp, masks, gun.exit_bundle.weight, sample_planes)
    if not proposals:
        raise RuntimeError("No converging-to-diverging Tip-ray waist bracket between TL1 and CM")
    roots = np.array([row["z_mm"] for row in proposals])
    refined_planes = np.unique(np.r_[roots-.001, roots, roots+.001, sample_planes])
    report["refined"] = []
    for step in (.05, .025):
        print(f"Executing refined crossover and sample checkpoints at step {step:g} mm", flush=True)
        fine_gun, fine_cp, fine_masks = trace(state, refined_planes, step, report["backend_receipts"])
        if source_identity(fine_gun.exit_bundle) != report["physical_source"]["exit_bundle_sha256"]:
            raise RuntimeError("Refinement changed the physical source population")
        waists = measured_corrector_crossovers(
            fine_cp, fine_masks, fine_gun.exit_bundle.weight, lower, upper)
        focus = surface_measurement(fine_cp, fine_masks, fine_gun.exit_bundle.weight, sample_planes)
        report["refined"].append(dict(maximum_step_mm=step, crossovers=waists, sample=focus))
    coarse, fine = (row["sample"] for row in report["refined"])
    checks = dict(
        gpu_column_executed=True,
        hp2_hp1_conjugacy=report["hp2_to_hp1"]["passed"],
        real_2d_crossover_at_both_steps=all(bool(row["crossovers"]) for row in report["refined"]),
        sample_rms_radius=fine["rms_radius_nm"] < LIMITS["sample_rms_nm"],
        sample_current95_diameter=fine["diameter95_nm"] < LIMITS["sample_diameter95_nm"],
        sample_alpha95=abs(fine["statistics"]["convergence_95_rad"]*1e3/LIMITS["sample_alpha95_mrad"]-1)
                       <= LIMITS["sample_alpha95_relative_error"],
        sample_covariance_focus=abs(fine["covariance_waist_offset_nm"]) < LIMITS["sample_waist_offset_nm"],
        sample_derivative_focus=(abs(fine["derivative_waist_offset_nm"]) < LIMITS["sample_waist_offset_nm"]
                                 and fine["variance_second_derivative"] > 0),
        sample_step_radius=abs(coarse["rms_radius_nm"]-fine["rms_radius_nm"]) < LIMITS["step_radius_difference_nm"],
        sample_step_focus=abs(coarse["covariance_waist_offset_nm"]-fine["covariance_waist_offset_nm"])
                          < LIMITS["step_waist_difference_nm"],
    )
    report["checks"] = checks
    if args.compare_aberrations:
        print("Comparing custom-mode linear, Cs-only and Cs-plus-hexapole transport", flush=True)
        compare_aberrations(state, report)
        checks["hexapoles_reduce_nonlinear_displacement"] = report["nonlinear_comparison"][
            "hexapoles_reduce_nonlinear_displacement"]
    report["status"] = "PASS" if all(checks.values()) else "FAIL"
    return cp, masks, fine_cp, fine_masks, gun.exit_bundle.weight


def figure(path, data, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cp, masks, fine, fine_masks, weights = data
    lower, upper = report["interval_mm"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, points, alive, label in zip(axes[:2], (cp, fine), (masks, fine_masks),
                                      ("Executed TL1-CM envelope", "Executed waist checkpoints")):
        selected = (points.z_mm > lower) & (points.z_mm < upper)
        cohort = np.all(alive[selected], axis=0) & (weights > 0)
        radii = [statistics(points, i, weights, cohort).radius_rms_m*1e9
                 for i in np.flatnonzero(selected)]
        ax.semilogy(points.z_mm[selected], radii, ".-")
        ax.set(xlabel="Z (mm)", ylabel="Weighted RMS radius (nm)", title=label)
        ax.grid(alpha=.2)
    stats = statistics(fine, -1, weights, fine_masks[-1])
    live = fine_masks[-1] & (weights > 0)
    axes[2].scatter((fine.x_m[-1, live]-stats.mean_x_m)*1e9,
                    (fine.y_m[-1, live]-stats.mean_y_m)*1e9, s=2, alpha=.5)
    axes[2].set(xlabel="X about centroid (nm)", ylabel="Y about centroid (nm)",
                title="Sample entrance: executed particles", aspect="equal")
    fig.suptitle(f"Catalog defaults: {report['status']} | {report['model_mode']} optics | physical Tip | CUDA column")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/probe_relay_validation"))
    parser.add_argument("--rays", type=int, default=3000)
    parser.add_argument("--compare-aberrations", action="store_true",
                        help="Also trace identical-source custom-mode linear/Cs/hexapole diagnostics")
    parser.add_argument("--figure", action="store_true", help="Save a static figure from executed data")
    args = parser.parse_args()
    if args.rays < 5:
        parser.error("At least five physical source rays are required")
    args.output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    report = dict(schema="probe-relay-validation-v1", status="RUNNING",
                  started_utc=datetime.now(timezone.utc).isoformat(), limits=LIMITS,
                  scope="Current custom-mode nanoprobe defaults including nonlinear optics and actual source energies; fixed particle budget and two transport steps",
                  limitations=["Not wave propagation, source-sampling convergence or experimental calibration",
                               "Conjugate HP planes and an actual Tip-ray waist are independent checks",
                               "No catalog settings are fitted, adopted or overwritten"])
    path = args.output / "report.json"
    save_report(path, report)
    try:
        data = validate(args, report)
        if args.figure:
            figure(args.output / "validation.png", data, report)
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
    finally:
        report["elapsed_seconds"] = perf_counter()-started
        save_report(path, report)
    print(f"{report['status']}: {path.resolve()}", flush=True)
    if "error" in report:
        print(report["error"], flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
