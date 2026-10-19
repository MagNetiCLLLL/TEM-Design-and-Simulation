"""Matched tip-origin C2 aberration comparison in vacuum, without desktop use.

This is a bounded diagnostic of the current Normal C3 + Probe Corrector model.
The public particle-section pipeline executes the gun, finite column fields,
apertures and physical screen interception. The specimen is explicitly vacuum;
there is no material scattering, coherent diffraction or Ronchigram claim.
Coefficient increments illustrate sensitivity and are not an OEM calibration.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=3000)
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--single-step", action="store_true",
                        help="omit the integration step/2 comparison")
    parser.add_argument("--analyse-only", action="store_true",
                        help="quantify paired response convergence and redraw saved physical hits; no propagation")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/c2-screen-aberrations"))
    args = parser.parse_args()
    if args.rays < 16 or not 0 < args.step <= .1:
        parser.error("at least 16 rays and a step in (0, 0.1] mm are required")
    return args


def main():
    args = arguments()
    for key in ("TEMSIM_CPU_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    os.environ["NUMBA_DISABLE_CUDA"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    if args.analyse_only:
        analyse_stored_responses(args.output.resolve())
        return
    import numpy as np
    from scripts.validate_nanoprobe_condenser import _configured_state, _json_safe
    from temsim.calculation_manifest import solver_source_identity
    from temsim.component_keys import FLUORESCENT_SCREEN
    from temsim.detector.point_spread import DetectorPointSpread
    from temsim.detector.receiver_image import receiver_image
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.operating_modes import apply_projector_mode
    from temsim.particle_section_io import load_section_result, save_section_result
    from temsim.optics.aberrations import intrinsic_lens_aberration_profile
    from temsim.optics.direct_alignment import diffraction_transfers
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.scan_geometry import classify_sample_plane_transfer
    from temsim.simulation_pipeline import calculate_particle_section

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    base, selection, _, _ = _configured_state("C3 + Probe Corrector", "normal")
    base.simulation_mode = "custom"
    base.acceleration_enabled = True
    base.acceleration_backend = "Numba CPU"
    base.electron_gun.emitter.ray_count = args.rays
    base.sample.specimen_mode = "vacuum"
    base.sample.inserted = False
    for drive in (base.ac_deflector, base.descan_deflector):
        drive.scan_enabled = drive.wobble_enabled = False
    for receiver in base.recording_planes:
        receiver.inserted = str(receiver.key) == FLUORESCENT_SCREEN
    c2_key = "condenser_lens_2"
    c2 = next(lens for lens in base.lenses if lens.key == c2_key)
    cs = intrinsic_lens_aberration_profile(c2, base.beam_voltage_kv).cs_mm
    base_cs = float(cs)
    c2.cs_mm = base_cs
    cases = {"baseline": {}, "Cs_plus20mm": {"cs_mm": 20.},
             "A1_plus0.03mm": {"a1_mm": .03}, "B2_plus1mm": {"b2_mm": 1.}}
    report = dict(schema="matched-c2-screen-particles-v1", status="RUNNING",
        scope=__doc__, rays=args.rays, voltage_kv=base.beam_voltage_kv,
        specimen="vacuum; no material execution", desktop_interaction=False,
        coefficient_changes=cases, baseline_c2_cs_mm=base_cs,
        source="fresh physical tip/gun execution followed by identity-checked checkpoint reuse",
        integration_steps_mm=[args.step] if args.single_step else [args.step, args.step/2],
        fixed_population="original source IDs; per-plane survivors and same-ID common cohort",
        point_spread=asdict(DetectorPointSpread.from_component(base.fluorescent_screen)),
        solver_source_identity=solver_source_identity(),
        backend_receipts=[], steps=[], checks=[])
    arrays, results, points = {}, {}, {}
    source_reference = None

    def save():
        report["elapsed_s"] = perf_counter()-started
        (out / "comparison.json").write_text(json.dumps(_json_safe(report), indent=2)+"\n", encoding="utf-8")

    def log(message):
        print(f"{perf_counter()-started:.2f}s {message}", flush=True)
        save()

    def execute(state, target, previous):
        nonlocal source_reference
        if solver_source_identity() != report["solver_source_identity"]:
            report.update(status="ERROR", error="Simulator source changed during this validation; restart with frozen source files")
            save()
            raise RuntimeError(report["error"])
        result = calculate_particle_section(state, target, existing_result=previous, acquire_stem=False)
        if solver_source_identity() != report["solver_source_identity"]:
            report.update(status="ERROR", error="Simulator source changed during transport; no mixed implementation result is accepted")
            save()
            raise RuntimeError(report["error"])
        bundle = result.simulation.gun_trace.exit_bundle
        source_fields = ("ray_id", "x_m", "y_m", "tx_rad", "ty_rad", "weight", "energy_offset_ev", "alive")
        values = tuple(np.asarray(getattr(bundle, key)) for key in source_fields)
        if source_reference is None:
            source_reference = tuple(value.copy() for value in values)
            arrays.update({"gun_exit_"+key: value for key, value in zip(source_fields, source_reference)})
        elif any(not np.array_equal(a, b, equal_nan=True) for a, b in zip(source_reference, values)):
            raise AssertionError("C2 cases must retain identical executed source particles")
        return result

    def clone(state):
        # The instrument graph contains immutable mapping proxies; its public
        # snapshot codec preserves shared component ownership without pickle.
        return capture_instrument_snapshot(state).restore()

    def coordinates(result):
        sim = result.simulation
        branch = sim.incident if not sim.branches else next(iter(sim.branches.values()))
        cp = sim.section_checkpoint.segments[-1].checkpoints
        assert float(cp.z_mm[-1]) == float(sim.metrics["section_target_z_mm"])
        xy = np.column_stack((cp.x_m[-1], cp.y_m[-1]))
        slopes = np.column_stack((cp.tx_rad[-1], cp.ty_rad[-1]))
        if sim.branches:
            mask = np.asarray(branch.blocked_key) == FLUORESCENT_SCREEN
        else:
            mask = np.asarray(branch.alive, bool)
        return xy, slopes, mask & (branch.ray_weight > 0), np.asarray(branch.ray_weight)

    def metrics(value, baseline=None, common=None):
        xy, slopes, alive, weights = value
        selected = alive if common is None else common
        if not np.any(selected):
            raise ValueError("No current-carrying particles in the requested cohort")
        q, w = xy[selected], weights[selected]/weights[selected].sum()
        mean = w @ q
        centred = q-mean
        covariance = centred.T @ (w[:, None]*centred)
        row = dict(rays=int(selected.sum()), source_fraction=float(weights[selected].sum()),
            centroid_nm=(mean*1e9).tolist(),
            radial_rms_nm=float(np.sqrt(w @ np.sum(centred**2, axis=1))*1e9),
            principal_rms_nm=(np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0))*1e9).tolist())
        if baseline is not None:
            delta = xy[selected]-baseline[0][selected]
            row["rms_displacement_from_baseline_nm"] = float(np.sqrt(w @ np.sum(delta**2, axis=1))*1e9)
            row["centroid_displacement_from_baseline_nm"] = (w @ delta*1e9).tolist()
        return row

    seed_path = out / "gun-and-c2-seed.temsection"
    seed = load_section_result(seed_path) if seed_path.exists() else None
    report["seed_archive"] = str(seed_path)
    projector_settings = {}
    report["projector_controls"] = {}
    focus_z = None
    with backend_receipts(report["backend_receipts"]):
        for step_index, step in enumerate(report["integration_steps_mm"]):
            base.step_mm = step
            step_row = dict(step_mm=step, cases={})
            report["steps"].append(step_row)
            if focus_z is None:
                log("Executing physical gun and preliminary C2 plane")
                seed = execute(clone(base), 672., seed)
                # Preserve the complete validated numerical source/section so
                # a plotting failure never requires synthesizing a replacement.
                save_section_result(seed, seed_path, overwrite=True)
                report["preliminary_gun_reused"] = bool(seed.simulation.metrics["section_gun_reused"])
                xy, slopes, mask, weights = coordinates(seed)
                stats = transverse_beam_statistics(xy[:, 0], xy[:, 1], slopes[:, 0], slopes[:, 1],
                    alive=mask, weights=weights)
                focus_z = 672.+stats.waist_offset_m*1e3
                if not 650. < focus_z < 915.:
                    raise ValueError("Baseline covariance waist is not between C2 and C3")
                report["observation_planes_mm"] = [focus_z-.03, focus_z, focus_z+.03]
                report["focus_selection"] = "Weighted baseline covariance estimate from executed Z=672 mm; all three resulting planes are executed exactly and fixed for every case and integration step"
                gun = seed.simulation.gun_trace
                arrays["source_weight"] = np.asarray(gun.exit_bundle.weight)
                arrays["source_id"] = np.asarray(gun.exit_bundle.ray_id)
                report["gun"] = dict(executed_rays=len(gun.exit_bundle.weight),
                    exit_alive=int(np.count_nonzero(gun.exit_bundle.alive)),
                    emitted_current_a=float(gun.emitted_current_a),
                    exit_z_mm=float(base.electron_gun.exit_plane_z_mm))
                log(f"Fixed executed C2 observation centre: {focus_z:.9f} mm")
            for name, change in cases.items():
                state = clone(base)
                lens = next(lens for lens in state.lenses if lens.key == c2_key)
                lens.cs_mm = base_cs+change.get("cs_mm", 0.)
                intrinsic = dict(state.lens_intrinsic_aberrations.get(c2_key, {}))
                for field, value in change.items():
                    if field != "cs_mm":
                        intrinsic[field] = intrinsic.get(field, 0.)+value
                state.lens_intrinsic_aberrations[c2_key] = intrinsic
                row = dict(c2_cs_mm=lens.cs_mm, intrinsic=intrinsic, observations={})
                step_row["cases"][name] = row
                previous = seed
                for index, z in enumerate(report["observation_planes_mm"]):
                    previous = execute(clone(state), z, previous)
                    key = (step_index, name, f"waist{index}")
                    points[key] = coordinates(previous)
                    row["observations"][key[-1]] = metrics(points[key])
                waist_seed = previous
                screen_seed = waist_seed
                for mode in ("imaging", "diffraction"):
                    mode_state = clone(state)
                    if mode not in projector_settings:
                        applied = apply_projector_mode(mode_state, mode, column_name=selection.column,
                                                       recording_name=selection.recording)
                        selected_keys = {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}
                        projector_settings[mode] = {lens.key: float(lens.percent) for lens in mode_state.lenses
                                                    if lens.key in selected_keys}
                        report["projector_controls"][mode] = dict(lens_percent=projector_settings[mode],
                            image_projection=(asdict(applied.image_projection) if applied.image_projection else None),
                            scope="Solved/applied once on baseline; exactly the same D/I/P1/P2 used for every C2 case and integration step")
                    else:
                        mode_state.projector_mode = "image" if mode == "imaging" else "diffraction"
                        for lens in mode_state.lenses:
                            if lens.key in projector_settings[mode]:
                                lens.percent = projector_settings[mode][lens.key]
                    assert all(lens.percent == projector_settings[mode][lens.key] for lens in mode_state.lenses
                               if lens.key in projector_settings[mode])
                    z = float(mode_state.fluorescent_screen.z_mm)
                    result = execute(mode_state, z, screen_seed)
                    screen_seed = result
                    key = (step_index, name, mode)
                    points[key] = coordinates(result)
                    row["observations"][mode] = metrics(points[key])
                    row["observations"][mode]["z_mm"] = z
                    row["observations"][mode]["gun_checkpoint_reused"] = bool(result.simulation.metrics["section_gun_reused"])
                    transfer = diffraction_transfers(mode_state, (z,))[z]
                    kind, image_residual, diff_residual = classify_sample_plane_transfer(transfer)
                    row["observations"][mode]["conjugacy"] = dict(kind=kind,
                        image_residual_m_per_rad=image_residual, diffraction_residual=diff_residual,
                        j_img=transfer.j_img.tolist(), j_diff_m_per_rad=transfer.j_diff_m_per_rad.tolist())
                    expected = "image" if mode == "imaging" else "diffraction"
                    if kind != expected:
                        raise AssertionError(f"Main-screen {mode} transfer is {kind}")
                    if step_index == 0:
                        results[name, mode] = result
                    if name == "baseline" and mode == "diffraction":
                        seed = result
                log(f"Step {step:g} mm: {name} waist + both screen modes complete")

    for step_index, step_row in enumerate(report["steps"]):
        for observation in ("waist0", "waist1", "waist2", "imaging", "diffraction"):
            common = np.logical_and.reduce([points[step_index, name, observation][2] for name in cases])
            baseline = points[step_index, "baseline", observation]
            for name in cases:
                key = (step_index, name, observation)
                value = points[key]
                step_row["cases"][name]["observations"][observation]["common_cohort"] = metrics(value, baseline, common)
                for label, array in zip(("xy_m", "slopes_rad", "survivor", "weight"), value):
                    arrays[f"step{step_index}_{name}_{observation}_{label}"] = array
    if len(report["steps"]) == 2:
        report["step_convergence"] = {}
        for name in cases:
            rows = report["step_convergence"][name] = {}
            for observation in ("waist1", "imaging", "diffraction"):
                coarse, fine = (points[index, name, observation] for index in (0, 1))
                common = coarse[2] & fine[2]
                value = metrics(coarse, fine, common)
                rows[observation] = dict(common_rays=value["rays"],
                    rms_step_difference_nm=value["rms_displacement_from_baseline_nm"])
                baseline_coarse = points[0, "baseline", observation]
                baseline_fine = points[1, "baseline", observation]
                matched = common & baseline_coarse[2] & baseline_fine[2]
                w = coarse[3][matched]/coarse[3][matched].sum()
                coarse_delta = coarse[0][matched]-baseline_coarse[0][matched]
                fine_delta = fine[0][matched]-baseline_fine[0][matched]
                rows[observation]["rms_change_of_aberration_response_nm"] = float(
                    np.sqrt(w @ np.sum((coarse_delta-fine_delta)**2, axis=1))*1e9)
    np.savez_compressed(out / "particles.npz", **arrays)
    snapshot = capture_instrument_snapshot(base)
    (out / "baseline.snapshot.json").write_text(json.dumps(snapshot.to_dict()), encoding="utf-8")

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    report["sampling"] = {}
    fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    for column, observation in enumerate(("waist1", "imaging", "diffraction")):
        span = max(float(np.max(np.abs(points[0, name, observation][0][points[0, name, observation][2]]))) for name in cases)*1.08
        scale = 1e9 if observation == "waist1" else 1e3
        unit = "nm" if observation == "waist1" else "mm"
        for row_index, name in enumerate(cases):
            xy, _, mask, _ = points[0, name, observation]
            ax = axes[row_index, column]
            ax.scatter(*(xy[mask].T*scale), s=2, alpha=.45, rasterized=True)
            ax.set(xlim=(-span*scale, span*scale), ylim=(-span*scale, span*scale),
                   aspect="equal", xlabel=f"X ({unit})", ylabel=f"Y ({unit})", title=f"{name} | {observation}")
            ax.grid(alpha=.15)
    fig.suptitle("Actual tip-origin particles: fixed planes and per-column scales\nVacuum specimen; C3 + Probe Corrector / Normal; positions are not coherent diffraction")
    fig.savefig(out / "waist-and-screen-hits.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 11), constrained_layout=True)
    for column, observation in enumerate(("waist1", "imaging", "diffraction")):
        reference = points[0, "baseline", observation]
        for row_index, name in enumerate(tuple(cases)[1:]):
            value = points[0, name, observation]
            common = reference[2] & value[2]
            delta = (value[0][common]-reference[0][common])*1e9
            limit = max(float(np.max(abs(delta)))*1.08, 1e-12)
            ax = axes[row_index, column]
            ax.scatter(*delta.T, s=2, alpha=.45, rasterized=True)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal",
                   xlabel="Delta X (nm)", ylabel="Delta Y (nm)",
                   title=f"{name} | {observation}")
            ax.ticklabel_format(style="sci", axis="both", scilimits=(-2, 3))
            ax.grid(alpha=.2)
    fig.suptitle("Same-source-ID displacement relative to baseline (not particle density)\nEach panel uses its labelled displacement scale to expose small finite responses")
    fig.savefig(out / "aberration-displacements.png", dpi=150)
    plt.close(fig)

    for mode in ("imaging", "diffraction"):
        span_mm = max(float(np.max(abs(points[0, name, mode][0][points[0, name, mode][2]])))*1e3 for name in cases)*1.08
        extent = (-span_mm, span_mm, -span_mm, span_mm)
        response_span = max(span_mm, 4.*max(report["point_spread"]["sigma_x_mm"], report["point_spread"]["sigma_y_mm"]))
        response_extent = (-response_span, response_span, -response_span, response_span)
        fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
        images = {}
        for name in cases:
            result = results[name, mode]
            old = receiver_image(result, FLUORESCENT_SCREEN, pixels=192)
            detail = receiver_image(result, FLUORESCENT_SCREEN, pixels=512, extent_mm=extent)
            response = receiver_image(result, FLUORESCENT_SCREEN, pixels=512, extent_mm=response_extent)
            assert old.status == detail.status == response.status == "AVAILABLE"
            assert abs(old.current_pa-detail.current_pa) < 1e-12
            images[name] = (old, detail, response)
            report["sampling"][f"{mode}:{name}"] = dict(old_grid_pixel_mm=float(old.x_mm[1]-old.x_mm[0]),
                old_occupied_pixels=int(np.count_nonzero(old.ideal_probability)),
                shared_roi_mm=extent, roi_pixel_mm=float(detail.x_mm[1]-detail.x_mm[0]),
                roi_occupied_pixels=int(np.count_nonzero(detail.ideal_probability)),
                shared_response_roi_mm=response_extent,
                whole_screen_current_pa=detail.current_pa,
                whole_screen_probability=detail.metrics["intercepted_probability"],
                displayed_probability=float(detail.ideal_probability.sum()),
                response_roi_probability=float(response.response_probability.sum()))
        old_peak = max(float(image.ideal_probability.max()) for image, _, _ in images.values())
        peak = max(float(image.ideal_probability.max()) for _, image, _ in images.values())
        response_peak = max(float(image.response_probability.max()) for _, _, image in images.values())
        for row_index, (name, (old, detail, response)) in enumerate(images.items()):
            for column, (image, data, title) in enumerate(((old, old.ideal_probability, "Full area: 192 pixels"),
                (detail, detail.ideal_probability, "Shared hit ROI: 512 pixels"),
                (response, response.response_probability, "Configured physical screen PSF"))):
                dx, dy = image.x_mm[1]-image.x_mm[0], image.y_mm[1]-image.y_mm[0]
                bounds = (image.x_mm[0]-dx/2, image.x_mm[-1]+dx/2,
                          image.y_mm[0]-dy/2, image.y_mm[-1]+dy/2)
                ax = axes[row_index, column]
                vmax = response_peak if column == 2 else old_peak if column == 0 else peak
                ax.imshow(data, origin="lower", extent=bounds, norm=LogNorm(max(vmax*1e-5, 1e-30), vmax), cmap="magma")
                if column == 0:
                    ax.set_xlim(-span_mm, span_mm)
                    ax.set_ylim(-span_mm, span_mm)
                ax.set(title=f"{name}\n{title}", xlabel="X (mm)", ylabel="Y (mm)")
        fig.suptitle(f"{mode}: rebinning retained physical first hits; same ROI and contrast across cases\nLeft shows the original coarse cells zoomed; right retains the provisional detector PSF (sigma 0.6 mm)")
        fig.savefig(out / f"{mode}-screen-sampling.png", dpi=150)
        plt.close(fig)
    for step_row in report["steps"]:
        for name in tuple(cases)[1:]:
            for observation in ("waist1", "imaging", "diffraction"):
                delta = step_row["cases"][name]["observations"][observation]["common_cohort"]["rms_displacement_from_baseline_nm"]
                if not delta > 0:
                    raise AssertionError(f"No finite C2 response: {name} / {observation}")
    if solver_source_identity() != report["solver_source_identity"]:
        raise RuntimeError("Simulator source changed during rendering; restart with frozen source files")
    report["status"] = "PASS_BOUNDED_PARTICLE_COMPARISON"
    report["limitations"] = [f"Fixed {args.rays}-ray source sampling is not particle-sampling convergence.",
        "Coefficients are controlled perturbations, not measured hardware values.",
        "Detector PSF remains active; rebinning cannot recover detail removed by physical blur.",
        "Vacuum optical diagnostic, not material imaging, phase contrast, Bragg diffraction or a Ronchigram."]
    log("Completed matched particle, conjugacy and screen sampling comparison")
    analyse_stored_responses(out)


def analyse_stored_responses(out):
    """Read retained first hits, not a synthetic or substitute receiver result."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from temsim.calculation_manifest import solver_source_identity
    from temsim.detector.point_spread import DetectorPointSpread
    from temsim.detector.receiver_image import _response_from_hits

    path = out / "comparison.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["solver_source_identity"] != solver_source_identity():
        raise RuntimeError("Stored transport and current response implementations differ; rerun validation")
    with np.load(out / "particles.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    cases = tuple(report["coefficient_changes"])
    observations = ("waist1", "imaging", "diffraction")

    def value(step, name, observation, field):
        return arrays[f"step{step}_{name}_{observation}_{field}"]

    def weighted_rms(delta, weights):
        return float(np.sqrt((weights/weights.sum()) @ np.sum(delta**2, axis=1)))

    extents = {observation: max(float(np.max(abs(value(step, name, observation, "xy_m")[
        value(step, name, observation, "survivor")]))) for step in range(len(report["steps"])) for name in cases)*1.08
        for observation in observations}

    def histogram(step, name, observation, *, pixels=512, extent=None):
        xy = value(step, name, observation, "xy_m")*1e3
        mask = value(step, name, observation, "survivor")
        weights = value(step, name, observation, "weight")
        half = extents[observation]*1e3 if extent is None else extent
        edges = np.linspace(-half, half, pixels+1)
        return np.histogram2d(xy[mask, 1], xy[mask, 0], bins=(edges, edges), weights=weights[mask])[0]

    convergence = {"relative_tolerance": .05,
        "criterion": "Paired same-ID displacement response changes by at most 5 percent on halving the integration step; fixed particle budget, not sampling convergence",
        "interpretation": "Declared deterministic two-step stability check, not statistical significance, an integration-order proof, or a rigorous error bound",
        "histogram_note": "Fixed common 512x512 bins, source-normalised probabilities, all actual survivors per case. Histogram response includes bin crossings and changes in physical survival. No detector-PSF convergence is claimed.",
        "cases": {}}
    if len(report["steps"]) == 2:
        all_resolved = True
        for name in cases[1:]:
            rows = convergence["cases"][name] = {}
            for observation in observations:
                mask = np.logical_and.reduce([value(step, case, observation, "survivor")
                    for step in (0, 1) for case in ("baseline", name)])
                if not np.any(mask):
                    raise ValueError(f"No shared current-carrying source IDs for {name} / {observation}")
                weights = value(0, name, observation, "weight")[mask]
                delta = [value(step, name, observation, "xy_m")[mask]
                         - value(step, "baseline", observation, "xy_m")[mask] for step in (0, 1)]
                amplitude = [weighted_rms(item, weights)*1e9 for item in delta]
                error = weighted_rms(delta[0]-delta[1], weights)*1e9
                ratio = error/amplitude[1] if amplitude[1] > 0 else None
                resolved = ratio is not None and ratio <= convergence["relative_tolerance"]
                all_resolved &= resolved
                hist_delta = [histogram(step, name, observation)-histogram(step, "baseline", observation) for step in (0, 1)]
                intensity_signal = float(np.abs(hist_delta[1]).sum())
                intensity_error = float(np.abs(hist_delta[0]-hist_delta[1]).sum())
                coarse_mask, fine_mask = (value(step, name, observation, "survivor") for step in (0, 1))
                source_weights = value(0, name, observation, "weight")
                rows[observation] = dict(common_rays=int(mask.sum()),
                    coarse_response_rms_nm=amplitude[0], fine_response_rms_nm=amplitude[1],
                    paired_response_step_error_nm=error, paired_response_relative_error=ratio,
                    displacement_response_resolved=bool(resolved),
                    histogram_fine_response_l1=intensity_signal, histogram_response_step_error_l1=intensity_error,
                    histogram_response_relative_error=(intensity_error/intensity_signal if intensity_signal else None),
                    histogram_response_step_stable=(intensity_error/intensity_signal <= .05 if intensity_signal else None),
                    histogram_status="nonzero_difference_at_current_bins" if intensity_signal else "no_change_resolved_at_current_bins",
                    coarse_survivors=int(coarse_mask.sum()), fine_survivors=int(fine_mask.sum()),
                    step_changed_survivor_ids=int(np.count_nonzero(coarse_mask ^ fine_mask)),
                    coarse_source_fraction=float(source_weights[coarse_mask].sum()),
                    fine_source_fraction=float(source_weights[fine_mask].sum()))
        convergence["all_displacement_responses_resolved"] = bool(all_resolved)
    else:
        convergence["all_displacement_responses_resolved"] = None
    report["paired_response_convergence"] = convergence
    report["execution_status"] = report.get("execution_status", report["status"])
    report["status"] = ("PASS_MATCHED_DISPLACEMENT_RESPONSE_CONVERGENCE" if convergence["all_displacement_responses_resolved"]
        else "COMPLETED_WITH_UNRESOLVED_RESPONSE_CONVERGENCE")
    report["source"] = ("validated persisted physical gun/section seed with exact source equality across all cases"
        if report.get("preliminary_gun_reused") else "fresh physical gun execution with exact source equality across all cases")

    through_focus = ("waist0", "waist1", "waist2")
    limit_nm = max(float(np.max(abs(value(0, name, plane, "xy_m")[value(0, name, plane, "survivor")])))
                   for name in cases for plane in through_focus)*1e9*1.08
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 13), sharex=True, sharey=True, constrained_layout=True)
    for row, name in enumerate(cases):
        for column, plane in enumerate(through_focus):
            xy = value(0, name, plane, "xy_m")[value(0, name, plane, "survivor")]*1e9
            ax = axes[row, column]
            ax.scatter(*xy.T, s=2, alpha=.45, rasterized=True)
            ax.set(xlim=(-limit_nm, limit_nm), ylim=(-limit_nm, limit_nm), aspect="equal",
                   xlabel="X (nm)", ylabel="Y (nm)",
                   title=f"{name} | Z = {report['observation_planes_mm'][column]:.6f} mm")
            ax.grid(alpha=.2)
    fig.suptitle("Actual tip-origin particles through the C2 waist: common axes and fixed planes\n"
                 "A1 can be nearly circular at the middle plane; compare the two neighbouring planes")
    fig.savefig(out / "c2-through-focus.png", dpi=150)
    plt.close(fig)
    name = "A1_plus0.03mm"
    common = np.logical_and.reduce([value(0, name, plane, "survivor") for plane in through_focus])
    weights = value(0, name, "waist1", "weight")[common]
    weights = weights/weights.sum()
    axis_rows = []
    for plane in through_focus:
        xy = value(0, name, plane, "xy_m")[common]
        centred = xy-weights @ xy
        eig, vectors = np.linalg.eigh(centred.T @ (weights[:, None]*centred))
        angle = float(np.degrees(np.arctan2(vectors[1, -1], vectors[0, -1])) % 180.)
        axis_rows.append(dict(plane=plane, principal_rms_nm=(np.sqrt(np.maximum(eig, 0))*1e9).tolist(),
                              major_axis_deg=angle))
    difference = abs((axis_rows[2]["major_axis_deg"]-axis_rows[0]["major_axis_deg"]+90.) % 180.-90.)
    report["a1_through_focus"] = dict(common_rays=int(common.sum()), rows=axis_rows,
        before_after_major_axis_angle_deg=difference,
        note="Weighted actual particle covariance in laboratory axes, without removing Larmor rotation, tilt or defocus. The middle covariance is nearly isotropic, so its principal-axis angle is not a stable orientation diagnostic.")

    psf = DetectorPointSpread(**report["point_spread"]).validated()
    probability_cmap = plt.colormaps["magma"].copy()
    probability_cmap.set_bad("black")
    for mode in ("imaging", "diffraction"):
        half = extents[mode]*1e3
        physical_half = report["sampling"][f"{mode}:baseline"]["old_grid_pixel_mm"]*192/2
        response_half = max(half, 4*max(psf.sigma_x_mm, psf.sigma_y_mm))
        if np.sqrt(2.)*response_half > physical_half:
            raise ValueError("Response redraw ROI must remain inside this centred circular screen; use the receiver API to evaluate its boundary mask")
        response_edges = np.linspace(-response_half, response_half, 513)
        images = {}
        for name in cases:
            mask = value(0, name, mode, "survivor")
            xy = value(0, name, mode, "xy_m")[mask]*1e3
            weights = value(0, name, mode, "weight")[mask]
            response, method, pitch = _response_from_hits(xy, weights, response_edges, response_edges, psf)
            images[name] = (histogram(0, name, mode, pixels=192, extent=physical_half),
                            histogram(0, name, mode), response)
            report["sampling"][f"{mode}:{name}"].update(
                comparison_roi_including_both_steps_mm=(-half, half, -half, half),
                redraw_response_evaluation=method, redraw_response_grid_pixel_size_mm=pitch,
                redraw_provenance="Retained actual first-hit coordinates and source weights from particles.npz; no propagation, no new particles, no replacement CalculationResult")
        peaks = [max(float(row[column].max()) for row in images.values()) for column in range(3)]
        fig, axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
        for row, (name, data) in enumerate(images.items()):
            for column, (pixels, radius, title) in enumerate(zip(data, (physical_half, half, response_half),
                    ("Full area: 192 pixels (zoomed)", "Shared hit ROI: 512 pixels", "Configured physical screen PSF"))):
                ax = axes[row, column]
                scale = 1e6 if mode == "imaging" and column < 2 else 1.
                unit = "nm" if scale != 1. else "mm"
                display_radius = radius*scale
                ax.imshow(pixels, origin="lower", extent=(-display_radius, display_radius, -display_radius, display_radius),
                    norm=LogNorm(max(peaks[column]*1e-5, 1e-30), peaks[column]), cmap=probability_cmap)
                if column == 0:
                    ax.set_xlim(-half*scale, half*scale)
                    ax.set_ylim(-half*scale, half*scale)
                ax.set(title=f"{name}\n{title}", xlabel=f"X ({unit})", ylabel=f"Y ({unit})")
        for column in range(3):
            colorbar = fig.colorbar(axes[0, column].images[0], ax=axes[:, column],
                orientation="horizontal", shrink=.85, fraction=.03, pad=.025)
            colorbar.set_label("Probability per emitted electron / pixel", fontsize=9)
        fig.suptitle(f"{mode}: retained physical hits at step {report['integration_steps_mm'][0]:g} mm\n"
            "Fixed ROI and contrast per column across cases; PSF is a provisional detector model, not a resolution claim")
        fig.savefig(out / f"{mode}-screen-sampling.png", dpi=150)
        plt.close(fig)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "paired_response_convergence": convergence}, indent=2), flush=True)


if __name__ == "__main__":
    main()
