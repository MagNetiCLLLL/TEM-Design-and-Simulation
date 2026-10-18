"""Frozen physical-Tip comparison of installed or candidate Descan transport.

The source is executed from the selected current Nanoprobe configuration, never
loaded from a historical strong-C3 beam. This is a vacuum optical experiment:
the specimen interaction and detector recording response are not simulated.
Candidate coil providers are diagnostic only; they do not move installed CAD,
the SAA, or the main fluorescent-screen plane.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from numbers import Real
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from validate_probe_relay import save_report as _save_report, source_identity


FACTORS = ((0., 0.), (-1., -1.), (1., -1.), (-1., 1.), (1., 1.))
BACKENDS = {"cpu": "Numba CPU", "gpu": "Require GPU"}


def save_report(path, report):
    def arrays(value):
        if isinstance(value, np.ndarray):
            return arrays(value.tolist())
        if isinstance(value, dict):
            return {key: arrays(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [arrays(item) for item in value]
        return value
    _save_report(path, arrays(report))


def finite_matrix(value, label, *, well_conditioned=False):
    try:
        raw = np.asarray(value, dtype=object)
        if raw.shape != (2, 2) or any(isinstance(item, (bool, np.bool_))
                or not isinstance(item, Real) for item in raw.flat):
            raise ValueError
        matrix = np.asarray(raw, float)
        if not np.isfinite(matrix).all():
            raise ValueError
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite real 2x2 matrix without booleans") from error
    if well_conditioned:
        try:
            condition = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{label} condition could not be determined") from error
        if not np.isfinite(condition) or condition > 1.0e10:
            raise ValueError(f"{label} is singular or ill-conditioned (limit 1e10)")
    return matrix


def load_candidate(path):
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"upper_z_mm", "lower_z_mm", "upper_kick_matrix_mrad",
                "lower_kick_matrix_mrad"}
    if not required.issubset(value):
        raise ValueError(f"Candidate requires {sorted(required)}")
    for key in ("upper_z_mm", "lower_z_mm"):
        if isinstance(value[key], bool) or not np.isfinite(value[key]):
            raise ValueError(f"Candidate {key} must be finite")
    if value["lower_z_mm"] <= value["upper_z_mm"]:
        raise ValueError("Candidate coil centres must increase downstream")
    for key in ("upper_kick_matrix_mrad", "lower_kick_matrix_mrad"):
        finite_matrix(value[key], f"Candidate {key}")
    return value


def prepare_state(args):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.physics.scan_geometry import calibrate_scan_system
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), column="C3 + Probe Corrector")
    catalog.apply(state, selection)
    selected_range = getattr(args, "nanoprobe_range", "normal")
    apply_operating_mode_pair(state, "nano_probe_large_angle" if selected_range == "large_angle" else "nano_probe", "diffraction",
        column_name=selection.column, recording_name=selection.recording)
    if state.nanoprobe_range != selected_range:
        raise ValueError("The requested Nanoprobe range was not applied")
    if float(state.beam_voltage_kv) != 300.:
        raise ValueError("This protocol requires the configured 300 kV column")
    if dict(state._resolved_assembly.selected_module_paths).get("column") != "column/C3_ProbeCorrector.toml":
        raise ValueError("This protocol requires the C3 + Probe Corrector column module")
    state.simulation_mode = "custom"
    state.acceleration_enabled = True
    state.acceleration_backend = BACKENDS[args.backend]
    state.step_mm = .05  # Frozen AC study calibration, independent of tail refinement.
    state.electron_gun.emitter.ray_count = args.rays
    if state.electron_gun.source_representation != "classical_particles":
        raise ValueError("Physical classical Tip emission is required")
    if bool(getattr(getattr(state, "vacuum_map", None), "enabled", False)):
        raise ValueError("Residual-medium transport is outside this optical protocol")
    state.ac_deflector.wobble_enabled = False
    for component in (state.ac_deflector, state.descan_deflector):
        component.scan_enabled = True
    if getattr(args, "descan_constraint", None) is not None:
        state.descan_deflector.descan_constraint = args.descan_constraint
    state.ac_deflector.calibration_mode = "automatic"
    state.ac_deflector.scan_lines = state.ac_deflector.scan_pixels_x
    state.ac_deflector.scan_pixel_size_nm = args.fov_nm / state.ac_deflector.scan_pixels_x
    result = calibrate_scan_system(state, observation_stop_z_mm=state.fluorescent_screen.z_mm)
    if result is None:
        raise ValueError("No finite scan calibration was consumed")
    command, scale_residual, descan = result
    calibration = dict(ac_command_matrix_mrad=np.asarray(command).tolist(),
                       scale_residual=float(scale_residual),
                       descan=None if descan is None else asdict(descan))
    return state, calibration


def measure(cp, index, weights, alive):
    """Raw weighted centroid/slopes; normalization affects statistics only."""
    arrays = np.stack([getattr(cp, key)[index] for key in
                       ("x_m", "y_m", "tx_rad", "ty_rad")])
    live = np.asarray(alive, bool) & (weights > 0)
    if np.any(~np.isfinite(arrays[:, live])):
        raise ValueError("Nonfinite current-carrying observation coordinates")
    current = float(weights[live].sum())
    result = dict(z_mm=float(cp.z_mm[index]), rays=int(live.sum()),
                  surviving_weight=current, source_weight=float(weights.sum()),
                  source_current_fraction=current / float(weights.sum()))
    if not live.any():
        return dict(result, mean_position_m=None, mean_mechanical_slope_rad=None,
                    rms_x_m=None, rms_y_m=None, rms_radius_m=None,
                    rms_mechanical_slope_rad=None)
    w = weights[live] / current
    means = arrays[:, live] @ w
    variance = ((arrays[:, live] - means[:, None]) ** 2) @ w
    return dict(result, mean_position_m=means[:2].tolist(),
                mean_mechanical_slope_rad=means[2:].tolist(),
                rms_x_m=float(np.sqrt(variance[0])), rms_y_m=float(np.sqrt(variance[1])),
                rms_radius_m=float(np.sqrt(variance[:2].sum())),
                rms_mechanical_slope_rad=float(np.sqrt(variance[2:].sum())))


def exact_index(cp, z):
    matches = np.flatnonzero(abs(np.asarray(cp.z_mm) - z) <= 1e-10)
    if len(matches) != 1:
        raise ValueError(f"Missing exact observation at {z} mm")
    return int(matches[0])


def frozen_providers(state, factors, descan_on, candidate, calibration, static):
    from temsim.physics.core import electron
    from temsim.physics.instrument_magnetic import column_dipole_fields
    factors = np.asarray(factors, float)
    state.ac_deflector.wobble_enabled = False
    for component in (state.ac_deflector, state.descan_deflector):
        component.scan_enabled = False
    ac_delta = np.asarray(calibration["ac_command_matrix_mrad"]) @ factors
    state.ac_deflector.kick_x_mrad, state.ac_deflector.kick_y_mrad = static["ac"] + ac_delta
    descan_delta = (np.asarray(state.descan_deflector.scan_command_matrix_mrad) @ factors
                   if descan_on and candidate is None else np.zeros(2))
    state.descan_deflector.kick_x_mrad, state.descan_deflector.kick_y_mrad = static["descan"] + descan_delta
    state.ac_deflector.validate()
    state.descan_deflector.validate()
    fields = list(column_dipole_fields(state))
    if candidate is not None:
        indices = [i for i, field in enumerate(fields) if "descan_deflector" in field.drive_keys]
        indices.sort(key=lambda i: int(fields[i].key.rsplit(":", 1)[1]))
        if len(indices) != 2:
            raise ValueError("Candidate must replace exactly the shared Image/Descan pair")
        charge = float(electron(state)[0])
        for index, name in zip(indices, ("upper", "lower")):
            field = fields[index]
            if field.registration is not None:
                raise ValueError("Candidate native-Z relocation requires an unposed physical host")
            centre = float(candidate[f"{name}_z_mm"])
            shift = (centre - field.event_z_mm) * 1e-3
            delta = (np.asarray(candidate[f"{name}_kick_matrix_mrad"]) @ factors * 1e-3
                     if descan_on else np.zeros(2))
            scale = field.reference_momentum / charge / (field.upper_m - field.lower_m)
            fields[index] = replace(field, lower_m=field.lower_m + shift,
                upper_m=field.upper_m + shift, event_z_mm=centre,
                event_dx_rad=field.event_dx_rad + float(delta[0]),
                event_dy_rad=field.event_dy_rad + float(delta[1]),
                bx_t=field.bx_t + scale * float(delta[1]),
                by_t=field.by_t - scale * float(delta[0]))
            host = getattr(state.descan_deflector, "_physical_host", state.descan_deflector)
            limit = float(host.maximum_kick_mrad)
            if max(abs(fields[index].event_dx_rad), abs(fields[index].event_dy_rad)) * 1e3 > limit:
                raise ValueError("Candidate summed drive exceeds the shared physical coil limit")
    return tuple(fields)


def check_backend(rows, backend):
    expected = "CUDA GPU" if backend == "gpu" else "Numba CPU"
    column = [row for row in rows if row.get("stage") == "column"]
    if not column or any(row.get("actual") != expected or row.get("outcome") != "returned"
                         for row in column):
        raise RuntimeError(f"Requested {expected} column execution was not verified")


def propagate_case(state, prefix, initial, prefix_alive, weights, energy, fields,
                   targets, step, backend, report, *, diagnostic_fields=True):
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall
    from temsim.physics.backend_execution import backend_receipts
    state.acceleration_backend = BACKENDS[backend]
    state.step_mm = float(step)
    stop = float(targets[-1])
    fields = tuple(field for field in fields
                   if field.field_support_mm[0] < stop and field.field_support_mm[1] > prefix)
    events = tuple((field.event_z_mm, field.event_dx_rad, field.event_dy_rad) for field in fields)
    apertures = [float(a.z_mm) for a in state.apertures if a.enabled and prefix <= a.z_mm <= stop]
    receipts = []
    with backend_receipts(receipts):
        plan = build_propagation_plan(state, prefix, stop, events,
            include_spherical_aberration=True, include_hexapole=True,
            include_intrinsic_aberrations=True, save_z_mm=apertures,
            checkpoint_z_mm=targets, maximum_step_mm=step,
            _diagnostic_dipole_fields=fields if diagnostic_fields else None)
        z, x, tx, y, ty, cp = execute_propagation_plan(state, plan, *initial, energy)
    check_backend(receipts, backend)
    report["backend_receipts"].extend(dict(row, scope=("frozen diagnostic post-AC prefix replay"
        if diagnostic_fields else "installed production provider plan and actual clipping")) for row in receipts)
    alive = prefix_alive.copy()
    stops = np.where(alive, np.nan, prefix)
    keys = [""] * len(alive)
    alive, stops, keys = clip_segment(state, z, x, y, alive, stops, keys)
    alive, stops, keys = clip_column_wall(state, z, x, y, alive, stops, keys)
    masks = prefix_alive[None, :] & (np.isnan(stops)[None, :] | (stops[None, :] > cp.z_mm[:, None]))
    return cp, masks, dict(stop_counts={key: keys.count(key) for key in sorted(set(keys)) if key})


def crossover_rows(cp, masks, weights, lower, upper):
    from temsim.optics.beam_path_audit import crossover_candidates
    select = (cp.z_mm >= lower) & (cp.z_mm <= upper)
    subset = SimpleNamespace(z_mm=cp.z_mm[select], **{
        key: getattr(cp, key)[select] for key in ("x_m", "y_m", "tx_rad", "ty_rad")})
    if len(subset.z_mm) < 2:
        return []
    return crossover_candidates(subset, masks[select], weights)


def chief_diagnostics(state, calibration, candidate, sample, saa, screen, args):
    """Separate local drive Jacobians from the finite emitted distribution."""
    from temsim.physics.finite_scan_response import finite_scan_responses
    grid = np.unique(np.r_[np.arange(sample, saa, args.grid_mm), saa, screen])
    ac = finite_scan_responses(state, state.ac_deflector, grid, maximum_step_mm=args.step_mm)
    descan = finite_scan_responses(state, state.descan_deflector, grid,
        upper_z_mm=None if candidate is None else candidate["upper_z_mm"],
        lower_z_mm=None if candidate is None else candidate["lower_z_mm"],
        maximum_step_mm=args.step_mm)
    ac_command = np.asarray(calibration["ac_command_matrix_mrad"]) * 1e-3
    ac_maps = [np.asarray(value) @ ac_command for value in state.ac_deflector.coil_kick_matrices()]
    if candidate is None:
        command = np.asarray(state.descan_deflector.scan_command_matrix_mrad) * 1e-3
        descan_maps = [np.asarray(value) @ command for value in state.descan_deflector.coil_kick_matrices()]
    else:
        descan_maps = [np.asarray(candidate[f"{name}_kick_matrix_mrad"]) * 1e-3 for name in ("upper", "lower")]
    scan = ac.upper @ ac_maps[0] + ac.lower @ ac_maps[1]
    compensation = descan.upper @ descan_maps[0] + descan.lower @ descan_maps[1]
    result = dict(scope="Local chief-ray derivatives about the neutral raster; not finite Tip centroids or beam waists",
        coordinate_order=["x_m", "y_m", "tx_mechanical_rad", "ty_mechanical_rad"],
        raster_basis="2 columns per dimensionless full-FOV factor",
        sampling_scope="Sampled 0.2 mm grid by default; closest approach alone does not certify an exact common pivot",
        modes={})
    for on, response in ((False, scan), (True, scan + compensation)):
        axis_norms = np.linalg.norm(response[:, :2, :], axis=1)
        select = grid <= saa
        relative = np.max(axis_norms[select], axis=1) / (args.fov_nm * .5e-9)
        i = int(np.flatnonzero(select)[np.argmin(relative)])
        observations = {}
        for key, z in (("selected_area_aperture", saa), ("fluorescent_screen", screen)):
            j = int(np.flatnonzero(grid == z)[0])
            observations[key] = dict(z_mm=z, phase_response=response[j].tolist())
        result["modes"]["descan_on" if on else "descan_off"] = dict(
            closest_sampled_approach_z_mm=float(grid[i]),
            both_axis_relative_residual=float(relative.min()),
            response_at_closest=response[i].tolist(), observations=observations,
            accepted_common_pivot=False,
            acceptance_note="No root refinement or acceptance from this sampled minimum")
    return result


def observation_differences(left, right):
    result = {}
    for key, coarse in left.items():
        fine = right[key]
        row = dict(source_current_fraction_change=fine["source_current_fraction"] - coarse["source_current_fraction"],
                   surviving_ray_count_change=fine["rays"] - coarse["rays"])
        if fine["rays"] and coarse["rays"]:
            row.update(centroid_difference_nm=float(np.linalg.norm(
                np.asarray(fine["mean_position_m"]) - coarse["mean_position_m"]) * 1e9),
                mechanical_slope_difference_urad=float(np.linalg.norm(
                    np.asarray(fine["mean_mechanical_slope_rad"]) - coarse["mean_mechanical_slope_rad"]) * 1e6),
                rms_radius_difference_nm=(fine["rms_radius_m"] - coarse["rms_radius_m"]) * 1e9)
        result[key] = row
    return result


def add_comparisons(report):
    """Report actual shifts and loss; never centre or rescale stored rays."""
    groups = sorted({(row.get("variant", "installed"), row["step_mm"], row["descan_on"])
                     for row in report["runs"]})
    for variant, step, on in groups:
        rows = [row for row in report["runs"] if row["descan_on"] == on
                and row["step_mm"] == step and row.get("variant", "installed") == variant]
        if not rows:
            continue
        centre = next(row for row in rows if row["factors"] == [0., 0.])
        for row in rows:
            row["relative_to_same_mode_centre"] = {}
            for key, value in row["observations"].items():
                ref = centre["observations"][key]
                delta = dict(source_current_fraction_change=value["source_current_fraction"] - ref["source_current_fraction"])
                if value["rays"] and ref["rays"]:
                    position = np.asarray(value["mean_position_m"]) - ref["mean_position_m"]
                    angle = np.asarray(value["mean_mechanical_slope_rad"]) - ref["mean_mechanical_slope_rad"]
                    delta.update(position_m=position.tolist(), mechanical_slope_rad=angle.tolist(),
                                 position_norm_nm=float(np.linalg.norm(position)*1e9),
                                 mechanical_slope_norm_urad=float(np.linalg.norm(angle)*1e6),
                                 rms_radius_change_m=value["rms_radius_m"] - ref["rms_radius_m"])
                row["relative_to_same_mode_centre"][key] = delta


def final_comparisons(report, output, weights):
    """Keep unconditional observations and add explicitly matched-ID deltas."""
    groups = {(row["variant"], row["step_mm"], row["descan_on"]) for row in report["runs"]}
    for variant, step, on in sorted(groups):
        rows = [row for row in report["runs"] if (row["variant"], row["step_mm"], row["descan_on"]) == (variant, step, on)]
        reference = next(row for row in rows if row["factors"] == [0., 0.])
        with np.load(output / f"{reference['label']}.npz") as centre:
            for row in rows:
                result = {}
                with np.load(output / f"{row['label']}.npz") as current:
                    if not np.array_equal(current["ray_id"], centre["ray_id"]) or not np.array_equal(current["z_mm"], centre["z_mm"]):
                        raise ValueError("Matched-ID observation files have different ray IDs or physical planes")
                    for i, key in enumerate(row["observations"]):
                        live = current["alive"][i] & centre["alive"][i] & (weights > 0)
                        item = dict(rays=int(live.sum()), source_current_fraction=float(weights[live].sum()/weights.sum()),
                            current_only_rays=int(np.count_nonzero(current["alive"][i] & ~centre["alive"][i])),
                            centre_only_rays=int(np.count_nonzero(centre["alive"][i] & ~current["alive"][i])))
                        if live.any():
                            w = weights[live] / weights[live].sum()
                            a = np.stack([current[name][i, live] for name in ("x_m", "y_m", "tx_rad", "ty_rad")])
                            b = np.stack([centre[name][i, live] for name in ("x_m", "y_m", "tx_rad", "ty_rad")])
                            means, base = a @ w, b @ w
                            difference = means - base
                            item.update(current_mean_position_m=means[:2].tolist(),
                                centre_mean_position_m=base[:2].tolist(),
                                current_mean_mechanical_slope_rad=means[2:].tolist(),
                                centre_mean_mechanical_slope_rad=base[2:].tolist(),
                                delta_position_m=difference[:2].tolist(),
                                delta_mechanical_slope_rad=difference[2:].tolist(),
                                delta_position_norm_nm=float(np.linalg.norm(difference[:2])*1e9),
                                delta_mechanical_slope_norm_urad=float(np.linalg.norm(difference[2:])*1e6))
                        result[key] = item
                row["matched_id_relative_to_centre"] = result
    steps = sorted({row["step_mm"] for row in report["runs"]}, reverse=True)
    comparisons = []
    if len(steps) == 2:
        coarse, fine = steps
        for old in report["runs"]:
            if old["step_mm"] != coarse:
                continue
            new = next(row for row in report["runs"] if row["step_mm"] == fine
                and row["variant"] == old["variant"] and row["factors"] == old["factors"]
                and row["descan_on"] == old["descan_on"])
            comparisons.append(dict(variant=old["variant"], factors=old["factors"],
                descan_on=old["descan_on"], coarse_step_mm=coarse, fine_step_mm=fine,
                differences=observation_differences(old["observations"], new["observations"])))
    report["all_case_step_comparisons"] = comparisons
    report["comparison_scope"] = "Unconditional centroids/current include clipping. Matched-ID deltas are additional diagnostics and never replace raw coordinates or throughput."
    first = report["runs"][0]
    with np.load(output / f"{first['label']}.npz") as baseline:
        reference_ids, reference_alive = baseline["ray_id"].copy(), baseline["alive"].copy()
        reference_z = baseline["z_mm"].copy()
    cohort = dict(reference_label=first["label"], identical_ids_and_masks_all_cases=True,
        per_plane_reference_rays=reference_alive.sum(axis=1).tolist(), z_mm=reference_z.tolist(),
        changed_cases=[])
    for row in report["runs"]:
        with np.load(output / f"{row['label']}.npz") as current:
            if not np.array_equal(current["ray_id"], reference_ids) or not np.array_equal(current["z_mm"], reference_z):
                raise ValueError("Cross-constraint comparison changed ray IDs or observation planes")
            added = np.count_nonzero(current["alive"] & ~reference_alive, axis=1)
            removed = np.count_nonzero(reference_alive & ~current["alive"], axis=1)
            if added.any() or removed.any():
                cohort["identical_ids_and_masks_all_cases"] = False
                cohort["changed_cases"].append(dict(label=row["label"], added=added.tolist(), removed=removed.tolist()))
    report["cohort_comparison_across_constraints_and_scan_states"] = cohort


def summary(report):
    finest = min(row["step_mm"] for row in report["runs"])
    result = dict(scope=report["scope"], finest_tail_step_mm=finest,
                  prefix=report["physical_source"], variants={})
    for variant in sorted({row["variant"] for row in report["runs"]}):
        modes = {}
        for on in (False, True):
            rows = [row for row in report["runs"] if row["variant"] == variant
                    and row["step_mm"] == finest and row["descan_on"] == on]
            if not rows:
                continue
            centre = next(row for row in rows if row["factors"] == [0., 0.])
            modes["on" if on else "off"] = dict(centre_absolute=centre["observations"],
                finite_beam_crossover_proposals=centre.get("refined_crossover_proposals", centre["finite_beam_crossovers"]),
                planes={key: dict(
                    maximum_scan_centroid_motion_nm=max(row["relative_to_same_mode_centre"][key].get("position_norm_nm", float("nan")) for row in rows),
                    maximum_scan_mechanical_slope_motion_urad=max(row["relative_to_same_mode_centre"][key].get("mechanical_slope_norm_urad", float("nan")) for row in rows),
                    source_current_fraction_range=[min(row["observations"][key]["source_current_fraction"] for row in rows),
                                                   max(row["observations"][key]["source_current_fraction"] for row in rows)])
                    for key in centre["observations"]})
        result["variants"][variant] = modes
    result["all_case_step_comparisons"] = report.get("all_case_step_comparisons", [])
    result["backend_comparison"] = report.get("backend_comparison", {})
    return result


def verify_local_crossover(state, prefix, initial, prefix_alive, weights, energy,
                           calibration, static, centre_mm, backend="gpu"):
    """Execute a finite-beam waist bracket, with a common positive-current cohort."""
    offsets = np.array([-.01, -.005, 0., .005, .01])
    targets = float(centre_mm) + offsets
    fields = frozen_providers(state, (0., 0.), False, None, calibration, static)
    result = dict(scope=f"Vacuum optical transport of the same frozen current-{state.nanoprobe_range} Tip prefix; no specimen interaction",
                  kind="finite_beam_crossover_not_scan_chief_pivot", proposed_centre_z_mm=float(centre_mm),
                  claimed_axial_bracket_mm=[float(targets[0]), float(targets[-1])],
                  prefix_z_mm=float(prefix), prefix_step_mm=.05, backend_receipts=[], steps=[])
    for step in (.05, .025):
        cp, alive, loss = propagate_case(state, prefix, initial, prefix_alive, weights, energy,
            fields, targets, step, backend, result)
        common = np.all(alive, axis=0) & (weights > 0)
        if np.count_nonzero(common) < 5:
            raise ValueError("Too little common current to verify the finite-beam crossover")
        w = weights[common] / weights[common].sum()
        rows = []
        for i, z in enumerate(targets):
            j = exact_index(cp, z)
            value = measure(cp, j, weights, common)
            x, y, tx, ty = (getattr(cp, name)[j, common] for name in ("x_m", "y_m", "tx_rad", "ty_rad"))
            covariance = float(w @ ((x - w @ x) * (tx - w @ tx) + (y - w @ y) * (ty - w @ ty)))
            rows.append(dict(value, weighted_centred_position_slope_covariance_m_rad=covariance))
        centre = rows[2]
        passed = (all(row["weighted_centred_position_slope_covariance_m_rad"] < 0 for row in rows[:2])
                  and all(row["weighted_centred_position_slope_covariance_m_rad"] > 0 for row in rows[3:])
                  and all(centre["rms_radius_m"] < row["rms_radius_m"] for row in rows[:2] + rows[3:]))
        result["steps"].append(dict(step_mm=step, observations=rows, same_ids_at_all_planes=bool(np.all(alive == common)),
            common_rays=int(common.sum()), common_source_current_fraction=float(weights[common].sum()/weights.sum()),
            centre_is_sampled_rms_minimum_and_covariance_crosses_zero=bool(passed), **loss))
    result["passed"] = all(row["centre_is_sampled_rms_minimum_and_covariance_crosses_zero"]
                           and row["same_ids_at_all_planes"] for row in result["steps"])
    result["centre_rms_step_difference_nm"] = abs(result["steps"][0]["observations"][2]["rms_radius_m"]
                                                 - result["steps"][1]["observations"][2]["rms_radius_m"]) * 1e9
    return result


def production_replay_from_saved_prefix(source_directory, output, backend="gpu", nanoprobe_range="normal", held_study=None):
    """Verify current installed wiring and a finite waist using captured source IDs."""
    from temsim.instrument_snapshot import capture_instrument_snapshot
    previous = json.loads((source_directory / "report.json").read_text(encoding="utf-8"))
    if previous.get("execution_complete") is not True:
        raise ValueError("Installed replay requires a completed physical-source comparison")
    if previous["nanoprobe_range"] != nanoprobe_range:
        raise ValueError("Saved prefix belongs to a different Nanoprobe range")
    provenance_path = source_directory / "prefix-provenance.json"
    if not provenance_path.exists():
        raise ValueError("Saved prefix has no verified physical-input and numeric-array provenance receipt")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(backend=backend, step_mm=.05, rays=previous["rays"],
        fov_nm=previous["fov_nm"], descan_constraint="position_and_angle", nanoprobe_range=nanoprobe_range)
    state, calibration = prepare_state(args)
    for lens in state.lenses:
        expected = previous["lenses"][lens.key]
        if float(lens.percent) != expected["percent"] or float(lens.z_mm) != expected["z_mm"]:
            raise ValueError(f"{lens.key}: cannot reuse source prefix after a lens change")
    source_parameters = json.loads(json.dumps(state.electron_gun.to_dict(),
        default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item()))
    if source_parameters != previous["physical_source"]["source_parameters"]:
        raise ValueError("Cannot reuse a prefix after changing physical source settings")
    if not np.allclose(calibration["ac_command_matrix_mrad"], previous["calibration"]["ac_command_matrix_mrad"], rtol=1e-12, atol=1e-14):
        raise ValueError("Installed replay changed the frozen AC command")
    planes = dict(specimen_entrance=float(state.sample.upper_surface_z_mm),
        selected_area_aperture=next(float(a.z_mm) for a in state.apertures if a.key == "selected_area_aperture"),
        fluorescent_screen=float(state.fluorescent_screen.z_mm))
    if planes != previous["fixed_observation_planes_mm"]:
        raise ValueError("Installed replay moved a physical observation plane")
    from descan_prefix_provenance import verify_prefix_provenance
    prefix_state = capture_instrument_snapshot(state).restore()
    prefix_state.ac_deflector.wobble_enabled = False
    for name, component in (("ac", prefix_state.ac_deflector), ("descan", prefix_state.descan_deflector)):
        component.scan_enabled = False
        component.kick_x_mrad, component.kick_y_mrad = previous["static_bias_mrad"][name]
    verify_prefix_provenance(provenance, prefix_state, source_directory / "frozen_prefix.npz")
    held = None
    if held_study is not None:
        from temsim.physics.scan_calibration import restore_held
        from temsim.physics.scan_geometry import synchronize_scan_raster
        from study_descan_phase_space import _accepted_drives
        if nanoprobe_range != "large_angle":
            raise ValueError("Normal-held replay requires a fresh Large Angle physical prefix")
        held = json.loads(held_study.read_text(encoding="utf-8"))["normal_held_comparison"]
        if held["field_of_view_nm"] != [previous["fov_nm"]] * 2 or not np.allclose(
                held["physical_centres_mm"], [state.descan_deflector.upper_z_mm,
                state.descan_deflector.lower_z_mm], rtol=0., atol=1e-9):
            raise ValueError("Held record has different physical centres or FOV")
        current_lenses = {lens.key: float(lens.percent) for lens in state.lenses if lens.enabled}
        if current_lenses != held["large_angle_lens_excitation_percent"]:
            raise ValueError("Held study and prefix use different Large Angle optics")
        state.ac_deflector.calibration_record_json = json.dumps(held["normal_accepted_record"], allow_nan=False)
        state.ac_deflector.calibration_mode = "held"
        restore_held(state)
        synchronize_scan_raster(state.ac_deflector, state.descan_deflector)
        if _accepted_drives(state) != held["normal_accepted_drives"]:
            raise ValueError("Restoring the held record changed its accepted physical drives")
        calibration = dict(ac_command_matrix_mrad=np.asarray(state.ac_deflector.scan_command_matrix_mrad).tolist(),
            source="Literal accepted Normal held record restored under Large Angle optics; no solve after restore")
    output.mkdir(parents=True, exist_ok=True)
    snapshot = capture_instrument_snapshot(state)
    save_report(output / "installed-inputs.json", snapshot.to_dict())
    with np.load(source_directory / "frozen_prefix.npz") as frozen:
        weights, energy, alive = (frozen[key].copy() for key in ("weights", "energy_offset_ev", "alive"))
        initial = [frozen[key].copy() for key in ("x_m", "tx_rad", "y_m", "ty_rad")]
        prefix = float(frozen["prefix_z_mm"])
        identity = str(frozen["exit_bundle_sha256"])
        if not np.array_equal(frozen["ray_id"], np.arange(len(weights))) or identity != previous["physical_source"]["exit_bundle_sha256"]:
            raise ValueError("Prefix identity does not match the executed physical source")
    static = {name: np.array([component.kick_x_mrad, component.kick_y_mrad], float)
              for name, component in (("ac", state.ac_deflector), ("descan", state.descan_deflector))}
    targets = np.unique(np.r_[prefix, np.arange(planes["specimen_entrance"], planes["selected_area_aperture"], .2), *planes.values()])
    reference_variants = [name for name, candidate in previous["cases"].items() if candidate is not None
        and np.allclose([candidate["upper_z_mm"], candidate["lower_z_mm"]],
            [state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm], rtol=0., atol=1e-9)
        and (candidate.get("production_constraint") == "position_and_angle"
             or candidate.get("solution", "full_phase_space") == "full_phase_space"
             and "production_constraint" not in candidate)]
    if len(reference_variants) != 1:
        raise ValueError("Saved comparison needs one full-phase case matching the installed centres")
    reference_variant = reference_variants[0]
    result = dict(scope=f"Freshly loaded current installed {nanoprobe_range}; production finite-provider capture and actual host/aperture clipping; exact existing physical Tip prefix reused",
        nanoprobe_range=nanoprobe_range, reference_variant=reference_variant,
        prefix_provenance=provenance,
        source_input_identity=previous["input_identity"], source_exit_bundle_sha256=identity,
        installed_input_identity=snapshot.digest, fixed_observation_planes_mm=planes,
        descan_constraint=state.descan_deflector.descan_constraint,
        descan_centres_mm=[state.descan_deflector.upper_z_mm, state.descan_deflector.lower_z_mm],
        calibration=calibration, backend_receipts=[], runs=[])
    if held is not None:
        result.update(scope="Literal accepted Normal held AC and Descan record restored under Large Angle optics; no recalibration after restore; same fresh Large Angle physical prefix",
            held_record=held["normal_accepted_record"], held_physical_drives=held["normal_accepted_drives"],
            held_source_study=str(held_study))
    for index, factors in enumerate(FACTORS if held is not None else ((0., 0.), (1., 1.))):
        fields = frozen_providers(state, factors, True, None, calibration, static)
        cp, masks, loss = propagate_case(state, prefix, initial, alive, weights, energy,
            fields, targets, .05, backend, result, diagnostic_fields=False)
        indices = {key: exact_index(cp, z) for key, z in planes.items()}
        observed = {key: measure(cp, j, weights, masks[j]) for key, j in indices.items()}
        diagnostic = next(row for row in previous["runs"] if row["variant"] == reference_variant
            and row["step_mm"] == .05 and row["descan_on"] and row["factors"] == list(factors))
        result["runs"].append(dict(label=f"installed-{index}", variant="normal_held" if held is not None else "installed",
            step_mm=.05, descan_on=True, factors=list(factors), observations=observed,
            versus_diagnostic=observation_differences(diagnostic["observations"], observed), **loss))
        selected = np.array(list(indices.values()))
        np.savez_compressed(output / f"installed-{index}.npz", ray_id=np.arange(len(weights)),
            z_mm=cp.z_mm[selected], alive=masks[selected],
            **{key: getattr(cp, key)[selected] for key in ("x_m", "y_m", "tx_rad", "ty_rad")})
    centre = next(row for row in previous["runs"] if row["variant"] == reference_variant
        and row["step_mm"] == .05 and row["descan_on"] and row["factors"] == [0., 0.])
    proposals = centre.get("refined_crossover_proposals", centre["finite_beam_crossovers"])
    downstream = [row for row in proposals if planes["specimen_entrance"] + .02 < row["z_mm"]
                  < float(state.descan_deflector.upper_z_mm)]
    if downstream and held is None:
        proposal = min(downstream, key=lambda row: row["z_mm"])
        result["finite_waist_verification"] = verify_local_crossover(state, prefix, initial, alive,
            weights, energy, calibration, static, proposal["z_mm"], backend=backend)
    elif held is None:
        result["finite_waist_verification"] = dict(passed=False,
            reason="No finite-beam waist was bracketed between specimen and upper Descan; no hard-coded waist assumed")
    if held is not None:
        if _accepted_drives(state) != held["normal_accepted_drives"]:
            raise ValueError("Physical replay changed the held scan matrices")
        result["held_matrices_unchanged_after_replay"] = True
    add_comparisons(result)
    final_comparisons(result, output, weights)
    result["maximum_scan_motion"] = {key: dict(
        position_nm=max(row["relative_to_same_mode_centre"][key]["position_norm_nm"] for row in result["runs"]),
        mechanical_slope_urad=max(row["relative_to_same_mode_centre"][key]["mechanical_slope_norm_urad"] for row in result["runs"]))
        for key in planes}
    result["complete"] = True
    save_report(output / "installed-replay.json", result)
    return result


def selected_cases(args, state, calibration):
    if getattr(args, "compare_constraints", False):
        return [(f"installed_{constraint}", dict(production_constraint=constraint,
            upper_z_mm=float(state.descan_deflector.upper_z_mm),
            lower_z_mm=float(state.descan_deflector.lower_z_mm), nanoprobe_range=state.nanoprobe_range))
            for constraint in ("position", "position_and_angle")]
    if args.study is None:
        candidate = load_candidate(args.candidate)
        if candidate is not None and candidate.get("nanoprobe_range", "normal") != state.nanoprobe_range:
            raise ValueError("Candidate belongs to a different Nanoprobe range")
        return [("installed" if args.candidate is None else args.candidate.stem, candidate)]
    study = json.loads(args.study.read_text(encoding="utf-8"))
    if study["nanoprobe_range"] != state.nanoprobe_range:
        raise ValueError("Candidate study belongs to a different Nanoprobe range")
    fov = np.asarray(study["field_of_view_nm"], float)
    if fov.shape != (2,) or not np.isfinite(fov).all() or not np.allclose(
            fov, args.fov_nm, rtol=0., atol=1e-9):
        raise ValueError("Candidate study FOV differs from this validation FOV")
    planes = {"selected_area_aperture": next(float(a.z_mm) for a in state.apertures
              if a.key == "selected_area_aperture"), "flu_screen": float(state.fluorescent_screen.z_mm)}
    if set(study["target_z_mm"]) != set(planes) or any(not np.isclose(
            float(study["target_z_mm"][key]), value, rtol=0., atol=1e-9)
            for key, value in planes.items()) or not np.isclose(
            float(study["sample_z_mm"]), float(state.sample.z_mm), rtol=0., atol=1e-9):
        raise ValueError("Candidate study and current instrument have different fixed physical planes")
    frozen = study["frozen_ac"]
    expected = finite_matrix(frozen["logical_command_basis_mrad"], "Frozen AC logical command", well_conditioned=True)
    upper = finite_matrix(frozen["upper_physical_drive_basis_mrad"], "Frozen AC upper drive", well_conditioned=True)
    lower = finite_matrix(frozen["lower_physical_drive_basis_mrad"], "Frozen AC lower drive", well_conditioned=True)
    if not np.allclose(np.asarray(calibration["ac_command_matrix_mrad"]), expected, rtol=1e-10, atol=1e-12):
        raise ValueError("Current AC calibration differs from the candidate study frozen AC")
    calibration["ac_command_matrix_mrad"] = expected.tolist()
    state.ac_deflector.set_scan_command_matrix_mrad(expected, frozen["scale_residual"])
    state.ac_deflector.set_pure_shift_coupling(np.linalg.solve(upper.T, lower.T).T, 0.)
    if not all(np.allclose(np.asarray(m) @ expected, desired, rtol=1e-12, atol=1e-14)
               for m, desired in zip(state.ac_deflector.coil_kick_matrices(), (upper, lower))):
        raise ValueError("Frozen physical AC matrices cannot be represented exactly")
    def candidate_for(shift, solution):
        options = [row["case"] for row in study["validations"]
                   if row["case"]["shift_mm"] == shift and row["case"]["step_mm"] == .05
                   and row["case"]["perturbation_rad"] == 1e-6]
        item = options[0] if options else next(row for row in study["baseline_candidates"] if row["shift_mm"] == shift)
        fit = item["targets"]["selected_area_aperture"]["solutions"][solution]
        centres = np.asarray(item["centres_mm"], float)
        if centres.shape != (2,) or not np.isfinite(centres).all() or centres[1] <= centres[0]:
            raise ValueError("Study candidate coil centres must be finite and increase downstream")
        for name in ("upper", "lower"):
            finite_matrix(fit[f"physical_{name}_drive_basis_mrad"], f"Study candidate {name} drive")
        candidate = dict(upper_z_mm=item["centres_mm"][0], lower_z_mm=item["centres_mm"][1],
            upper_kick_matrix_mrad=fit["physical_upper_drive_basis_mrad"],
            lower_kick_matrix_mrad=fit["physical_lower_drive_basis_mrad"],
            fov_nm=study["field_of_view_nm"][0], shift_mm=shift,
            source_study=str(args.study), calibration_step_mm=item["step_mm"], solution=solution,
            nanoprobe_range=state.nanoprobe_range)
        return candidate

    legacy = candidate_for(0., "legacy_position_only")
    save_report(args.output / "legacy0-candidate.json", legacy)
    result = [("legacy0", legacy)]
    for shift in args.shifts:
        candidate = candidate_for(shift, "full_phase_space")
        label = f"phase{shift:g}"
        save_report(args.output / f"{label}-candidate.json", candidate)
        result.append((label, candidate))
    return result


def validate(args, report):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.instrument_magnetic import column_dipole_fields
    state, calibration = prepare_state(args)
    cases = selected_cases(args, state, calibration)
    for _, candidate in cases:
        if candidate is not None and "fov_nm" in candidate and not np.isclose(
                float(candidate["fov_nm"]), args.fov_nm, rtol=0., atol=1e-9):
            raise ValueError("Candidate matrix FOV differs from this validation FOV")
    static = {name: np.array([component.kick_x_mrad, component.kick_y_mrad], float)
              for name, component in (("ac", state.ac_deflector), ("descan", state.descan_deflector))}
    sample = float(state.sample.upper_surface_z_mm)
    saa = next(float(a.z_mm) for a in state.apertures if a.key == "selected_area_aperture")
    screen = float(state.fluorescent_screen.z_mm)
    if not sample < saa < screen:
        raise ValueError("Expected specimen entrance < fixed SAA < fixed fluorescent screen")
    planes = {"specimen_entrance": sample, "selected_area_aperture": saa,
              "fluorescent_screen": screen}
    snapshot = capture_instrument_snapshot(state)
    save_report(args.output / "inputs.json", snapshot.to_dict())
    report.update(scope=f"Current Nanoprobe {state.nanoprobe_range} physical Tip; custom finite fields; vacuum optical transport without specimen interaction or detector intensity response",
        nanoprobe_range=state.nanoprobe_range, projector_mode="diffraction", rays=args.rays,
        column_module=dict(state._resolved_assembly.selected_module_paths).get("column"),
        installed_descan_centres_mm=[float(state.descan_deflector.upper_z_mm), float(state.descan_deflector.lower_z_mm)],
        initial_descan_constraint=state.descan_deflector.descan_constraint,
        voltage_kv=float(state.beam_voltage_kv), study_scan_motion_targets=dict(centroid_nm=1., mechanical_slope_urad=.01,
            scope="Study targets at declared FOV, not universal microscope acceptance criteria"),
        input_identity=snapshot.digest, fov_nm=args.fov_nm, scan_factors=list(FACTORS),
        factor_definition="Centre and geometric FOV corners, not raster pixel centres",
        step_mm=args.step_mm, grid_mm=args.grid_mm, calibration=calibration,
        cases=dict(cases), fixed_observation_planes_mm=planes,
        candidate_scope=("Actual installed production providers and host clipping for both constraints"
            if getattr(args, "compare_constraints", False) else
            "Diagnostic finite providers only; host clearance and CAD relocation not qualified"),
        lenses={lens.key: {"percent": float(lens.percent), "z_mm": float(lens.z_mm)} for lens in state.lenses},
        static_bias_mrad={key: value.tolist() for key, value in static.items()},
        backend_receipts=[], runs=[], execution_complete=False,
        crossover_definition="Finite current-weighted envelope waist; same surviving cohort on each bracket. Distinct from intersection of scan chief trajectories.")
    if args.prepare_only:
        report["status"] = "PREPARED_NOT_PROPAGATED"
        return
    fields = frozen_providers(state, (0., 0.), False, None, calibration, static)
    prefix = min(field.field_support_mm[0] for field in fields if "ac_deflector" in field.drive_keys) - .001
    for _, candidate in cases:
        if candidate is not None and "production_constraint" not in candidate:
            replaced_fields = frozen_providers(state, (0., 0.), False, candidate, calibration, static)
            if any(field.field_support_mm[0] < prefix for field in replaced_fields if "descan_deflector" in field.drive_keys):
                raise ValueError("Candidate Descan support begins before the frozen upstream prefix")
    print(f"Executing current {state.nanoprobe_range} Tip and upstream prefix to Z={prefix:g} mm", flush=True)
    receipts = []
    prefix_started = perf_counter()
    state.step_mm = .05
    prefix_snapshot = capture_instrument_snapshot(state)
    save_report(args.output / "prefix-inputs.json", prefix_snapshot.to_dict())
    upstream_planes = np.array([prefix])
    if getattr(args, "upstream_checkpoints", False):
        lens_z = {lens.key: float(lens.z_mm) for lens in state.lenses}
        c2, c3, adl = (lens_z[key] for key in ("condenser_lens_2", "condenser_lens_3", "adapter_lens"))
        guard = min(args.grid_mm / 2, (c3-c2) / 8, (adl-c3) / 8)
        upstream_planes = np.unique(np.r_[np.arange(c2, adl, args.grid_mm),
            c2-guard, c2, c2+guard, c3-guard, c3, c3+guard, adl-guard, adl, adl+guard, prefix])
    with backend_receipts(receipts):
        gun, prefix_cp, masks = incident_checkpoints(state, upstream_planes, step_mm=.05)
    report["prefix_elapsed_s"] = perf_counter() - prefix_started
    print(f"Fresh physical prefix completed in {report['prefix_elapsed_s']:.1f} seconds", flush=True)
    check_backend(receipts, args.backend)
    report["backend_receipts"].extend(dict(row, scope="current physical Tip and upstream prefix") for row in receipts)
    bundle = gun.exit_bundle
    identity = source_identity(bundle)
    weights = np.asarray(bundle.weight, float).copy()
    energy = np.asarray(bundle.energy_offset_ev, float).copy()
    prefix_index = exact_index(prefix_cp, prefix)
    initial = [getattr(prefix_cp, key)[prefix_index].copy() for key in ("x_m", "tx_rad", "y_m", "ty_rad")]
    prefix_alive = masks[prefix_index].copy()
    frozen_hash = hashlib.sha256(b"".join(np.ascontiguousarray(value).tobytes()
        for value in (*initial, weights, energy, prefix_alive))).hexdigest()
    report["physical_source"] = dict(exit_bundle_sha256=identity, ray_ids="unchanged original index 0..N-1",
        prefix_z_mm=prefix, prefix_surviving_rays=int(prefix_alive.sum()),
        prefix_source_current_fraction=float(weights[prefix_alive].sum()/weights.sum()),
        source_parameters=state.electron_gun.to_dict())
    np.savez_compressed(args.output / "frozen_prefix.npz", ray_id=np.arange(len(weights)),
        weights=weights, energy_offset_ev=energy, alive=prefix_alive,
        x_m=initial[0], tx_rad=initial[1], y_m=initial[2], ty_rad=initial[3],
        prefix_z_mm=prefix, exit_bundle_sha256=identity)
    from descan_prefix_provenance import create_prefix_provenance
    provenance = create_prefix_provenance(prefix_snapshot.restore(), args.output / "frozen_prefix.npz",
        capture_kind="contemporaneous_prefix_inputs", source_snapshot_digest=prefix_snapshot.digest,
        checkpoint_z_mm=upstream_planes)
    save_report(args.output / "prefix-provenance.json", provenance)
    if getattr(args, "upstream_checkpoints", False):
        np.savez_compressed(args.output / "upstream-checkpoints.npz", ray_id=np.arange(len(weights)),
            z_mm=prefix_cp.z_mm, weights=weights, energy_offset_ev=energy, alive=masks,
            **{key: getattr(prefix_cp, key) for key in ("x_m", "y_m", "tx_rad", "ty_rad")})
        np.savez_compressed(args.output / "gun-exit.npz", ray_id=np.arange(len(weights)),
            weights=weights, energy_offset_ev=energy, alive=np.asarray(bundle.alive),
            z_mm=float(gun.z_mm[-1]), blocked_z_mm=np.asarray(gun.blocked_z_mm), blocked_key=np.asarray(gun.blocked_key),
            exit_bundle_sha256=identity, **{key: np.asarray(getattr(bundle, key)) for key in ("x_m", "y_m", "tx_rad", "ty_rad")})
        report["upstream_crossover_proposals"] = dict(
            c2_to_c3=crossover_rows(prefix_cp, masks, weights, c2, c3),
            c3_to_adl=crossover_rows(prefix_cp, masks, weights, c3, adl))
    targets = np.unique(np.r_[prefix, np.arange(sample, saa, args.grid_mm), sample, saa, screen])
    save_report(args.output / "report.json", report)
    for variant, candidate in cases:
        production = candidate is not None and "production_constraint" in candidate
        active_candidate = None if production else candidate
        if production:
            from temsim.physics.scan_geometry import calibrate_scan_system
            state.step_mm = .05
            for name, component in (("ac", state.ac_deflector), ("descan", state.descan_deflector)):
                component.kick_x_mrad, component.kick_y_mrad = static[name]
                component.scan_enabled = True
            state.descan_deflector.descan_constraint = candidate["production_constraint"]
            command, residual, descan = calibrate_scan_system(state, observation_stop_z_mm=screen)
            if not np.allclose(command, calibration["ac_command_matrix_mrad"], rtol=1e-12, atol=1e-14):
                raise ValueError("Changing Descan constraint changed the frozen AC calibration")
            report.setdefault("production_calibrations", {})[variant] = asdict(descan)
        if args.chief_diagnostics:
            frozen_providers(state, (0., 0.), False, active_candidate, calibration, static)
            print(f"{variant}: measuring separate local chief-ray response maps", flush=True)
            report.setdefault("scan_chief_diagnostics", {})[variant] = chief_diagnostics(
                state, calibration, active_candidate, sample, saa, screen, args)
        for step in ((args.step_mm, args.step_mm / 2) if args.step_refinement else (args.step_mm,)):
            for on in (False, True):
                for index, factors in enumerate(FACTORS):
                    label = f"{variant}_h{step:g}_descan_{'on' if on else 'off'}_{index}"
                    print(f"{label}: factors={factors}", flush=True)
                    case_started = perf_counter()
                    fields = frozen_providers(state, factors, on, active_candidate, calibration, static)
                    cp, live, loss = propagate_case(state, prefix, initial, prefix_alive, weights,
                        energy, fields, targets, step, args.backend, report, diagnostic_fields=not production)
                    indices = {key: exact_index(cp, z) for key, z in planes.items()}
                    row = dict(label=label, descan_on=on, factors=list(factors), backend=args.backend, variant=variant, step_mm=step,
                        observations={key: measure(cp, i, weights, live[i]) for key, i in indices.items()},
                        finite_beam_crossovers=crossover_rows(cp, live, weights, sample, saa), **loss)
                    curve = [measure(cp, i, weights, live[i]) for i in range(len(cp.z_mm)) if sample <= cp.z_mm[i] <= saa]
                    row["finite_beam_envelope"] = curve
                    if args.refine_crossovers and row["finite_beam_crossovers"]:
                        fine_targets = np.unique(np.r_[sample, saa, [z for proposal in row["finite_beam_crossovers"]
                            for z in np.r_[np.arange(*proposal["bracket_mm"], .05), proposal["bracket_mm"][1], proposal["z_mm"]]]])
                        fine, fine_live, _ = propagate_case(state, prefix, initial, prefix_alive, weights,
                            energy, fields, fine_targets, step, args.backend, report, diagnostic_fields=not production)
                        row["refined_crossover_proposals"] = [root
                            for proposal in row["finite_beam_crossovers"]
                            for root in crossover_rows(fine, fine_live, weights, *proposal["bracket_mm"])]
                        row["exact_crossover_bracket_samples"] = [measure(fine, i, weights, fine_live[i])
                            for i, z in enumerate(fine.z_mm) if any(a <= z <= b
                                for proposal in row["finite_beam_crossovers"] for a, b in [proposal["bracket_mm"]])]
                        row["refinement_scope"] = "0.05 mm exact axial samples inside original brackets; interpolation remains a proposal, not full focus acceptance"
                    selected = np.array(list(indices.values()))
                    np.savez_compressed(args.output / f"{label}.npz", ray_id=np.arange(len(weights)),
                        z_mm=cp.z_mm[selected], alive=live[selected],
                        **{key: getattr(cp, key)[selected] for key in ("x_m", "y_m", "tx_rad", "ty_rad")})
                    row["elapsed_s"] = perf_counter() - case_started
                    print(f"{label} completed in {row['elapsed_s']:.1f} seconds", flush=True)
                    report["runs"].append(row)
                    add_comparisons(report)
                    save_report(args.output / "report.json", report)
        if args.compare_backend is not None:
            backend = args.compare_backend
            print(f"{variant} backend comparison: final corner, {backend}, step={step:g} mm", flush=True)
            replay, replay_alive, loss = propagate_case(state, prefix, initial, prefix_alive, weights,
                energy, fields, targets, step, backend, report, diagnostic_fields=not production)
            observed = {name: measure(replay, exact_index(replay, z), weights,
                                     replay_alive[exact_index(replay, z)]) for name, z in planes.items()}
            report.setdefault("backend_comparison", {})[variant] = dict(
                scope="Final (+1,+1) corner, Descan on; identical executed upstream prefix",
                backend=backend, step_mm=step, observations=observed,
                differences=observation_differences(report["runs"][-1]["observations"], observed), **loss)
            save_report(args.output / "report.json", report)
    if source_identity(bundle) != identity:
        raise RuntimeError("The frozen physical source changed during the comparison")
    if hashlib.sha256(b"".join(np.ascontiguousarray(value).tobytes()
            for value in (*initial, weights, energy, prefix_alive))).hexdigest() != frozen_hash:
        raise RuntimeError("Replay modified the frozen prefix coordinates, IDs or weights")
    if planes != {"specimen_entrance": float(state.sample.upper_surface_z_mm),
                  "selected_area_aperture": next(float(a.z_mm) for a in state.apertures if a.key == "selected_area_aperture"),
                  "fluorescent_screen": float(state.fluorescent_screen.z_mm)}:
        raise RuntimeError("Observation planes moved during the comparison")
    final_comparisons(report, args.output, weights)
    save_report(args.output / "summary.json", summary(report))
    report.update(execution_complete=True, status="COMPLETE_MEASURED_NOT_INSTRUMENT_QUALIFIED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--nanoprobe-range", choices=("normal", "large_angle"), default="normal")
    parser.add_argument("--compare-constraints", action="store_true",
        help="Compare both Descan constraints using actual installed production providers and one frozen source")
    parser.add_argument("--upstream-checkpoints", action="store_true",
        help="Retain executed C2-to-ADL checkpoints and the physical gun exit for independent topology checks")
    parser.add_argument("--replay-prefix-source", type=Path,
        help="Directory containing a completed comparison and frozen_prefix.npz; replay current installed wiring and exact finite waist")
    parser.add_argument("--normal-held-study", type=Path,
        help="With a Large Angle prefix replay, restore the exact Normal held record from this linear study without recalibrating")
    parser.add_argument("--shifts", nargs="+", type=float, default=[-100., 0.])
    parser.add_argument("--rays", type=int, default=3000)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--grid-mm", type=float, default=.2)
    parser.add_argument("--fov-nm", type=float, default=1000.)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="gpu")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--chief-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refine-crossovers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--step-refinement", action="store_true")
    parser.add_argument("--compare-backend", choices=tuple(BACKENDS))
    parser.add_argument("--descan-constraint", choices=("position", "position_and_angle"))
    args = parser.parse_args()
    if sum(value is not None for value in (args.candidate, args.study, args.replay_prefix_source)) + args.compare_constraints > 1:
        parser.error("Choose at most one candidate, study, or saved-prefix replay")
    if args.replay_prefix_source is not None:
        if args.prepare_only:
            parser.error("Saved-prefix replay executes transport; prepare-only is unavailable")
        if args.output.resolve() == args.replay_prefix_source.resolve():
            parser.error("Replay output must differ from its source directory")
    if args.normal_held_study is not None and (args.replay_prefix_source is None or args.nanoprobe_range != "large_angle"):
        parser.error("Normal-held study requires --replay-prefix-source and --nanoprobe-range large_angle")
    if args.rays < 5 or any(not np.isfinite(value) or value <= 0 for value in
                           (args.step_mm, args.grid_mm, args.fov_nm)):
        parser.error("At least five rays and finite positive step/grid/FOV are required")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    started = perf_counter()
    try:
        if args.replay_prefix_source is None:
            validate(args, report)
        else:
            report.update(production_replay_from_saved_prefix(args.replay_prefix_source, args.output,
                backend=args.backend, nanoprobe_range=args.nanoprobe_range, held_study=args.normal_held_study))
            report["status"] = "COMPLETE_INSTALLED_REPLAY_NOT_INSTRUMENT_QUALIFIED"
    except Exception as error:
        report.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"] = perf_counter() - started
        save_report(args.output / "report.json", report)


if __name__ == "__main__":
    main()
