"""Read-only finite-coil Descan study in an explicit STEM Nanoprobe range.

Candidate centres and independent physical foil drives are diagnostic values.
This script does not move hardware in a saved instrument, alter production
calibration, certify mechanical clearance, or establish finite-beam performance.

Reproduce the historical placement after changing the installed column with
--baseline-pair-center-mm 1786.9. This sets only the runtime shared deflector
centre; installed SAA and screen planes remain fixed for every comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=_json, indent=2, allow_nan=False), encoding="utf-8")


def _peak(matrix_rad):
    """Per-axis absolute maximum over the complete square raster, in mrad."""
    return np.sum(np.abs(matrix_rad), axis=1) * 1e3


def _relative(value, reference):
    return float(np.linalg.norm(value) / max(float(np.linalg.norm(reference)), 1e-30))


def _configured_state(nanoprobe_range="normal"):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_operating_mode_pair, validate_nanoprobe_range
    from temsim.optics.column import default_state

    selected_range = validate_nanoprobe_range(nanoprobe_range)
    mode = "nano_probe_large_angle" if selected_range == "large_angle" else "nano_probe"
    state = default_state()
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), column="C3 + Probe Corrector")
    catalog.apply(state, selection)
    apply_operating_mode_pair(state, mode, "diffraction", column_name=selection.column,
                              recording_name=selection.recording)
    if state.beam_voltage_kv != 300. or state.nanoprobe_range != selected_range:
        raise ValueError("Study requires the selected 300 kV Nanoprobe range")
    state.acceleration_enabled, state.acceleration_backend = True, "Numba CPU"
    state.ac_deflector.wobble_enabled = False
    state.ac_deflector.scan_enabled = state.descan_deflector.scan_enabled = True
    state.ac_deflector.calibration_mode = "automatic"
    state.ac_deflector.scan_pixels_x = state.ac_deflector.scan_lines = 4
    state.ac_deflector.scan_pixel_size_nm = .02
    state.step_mm = .05
    return state, mode


def _accepted_drives(state):
    return {
        str(pair.key): {
            "logical_command_matrix_mrad": np.asarray(pair.scan_command_matrix_mrad).tolist(),
            "coil_matrices": np.asarray(pair.coil_kick_matrices()).tolist(),
            "physical_drive_basis_mrad": [
                (np.asarray(mapping) @ np.asarray(pair.scan_command_matrix_mrad)).tolist()
                for mapping in pair.coil_kick_matrices()],
        } for pair in (state.ac_deflector, state.descan_deflector)
    }


def _measure_accepted_drives(state, targets, fov_nm):
    from temsim.optics.direct_alignment import canonical_source_basis
    from temsim.physics.finite_scan_response import finite_scan_responses

    planes = {"sample": float(state.sample.z_mm), **targets}
    ac, ds = state.ac_deflector, state.descan_deflector
    ac_finite = finite_scan_responses(state, ac, planes.values(), maximum_step_mm=.05)
    ds_finite = finite_scan_responses(state, ds, planes.values(), maximum_step_mm=.05)
    ac_drives, ds_drives = [
        [np.asarray(mapping) @ np.asarray(pair.scan_command_matrix_mrad) * 1e-3
         for mapping in pair.coil_kick_matrices()] for pair in (ac, ds)]
    observations = {}
    for key, z in planes.items():
        au, al, _ = ac_finite.at(z)
        du, dl, _ = ds_finite.at(z)
        ac_phase = au @ ac_drives[0] + al @ ac_drives[1]
        observations[key] = _phase_metrics(ac_phase, du, dl, np.vstack(ds_drives))
        if key == "sample":
            total = ac_phase + du @ ds_drives[0] + dl @ ds_drives[1]
            desired = np.eye(2) * fov_nm * .5e-9
            g = canonical_source_basis(state, z)[2:, :2]
            observations[key]["position_target_relative_error"] = _relative(total[:2]-desired, desired)
            observations[key]["net_canonical_slope_basis_mrad"] = (total[2:] - g @ total[:2]) * 1e3
    return observations


def _normal_held_comparison(fov_nm):
    """Apply the Normal record to changed optics without fitting any new drive."""
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.physics.scan_calibration import restore_held
    from temsim.physics.scan_geometry import calibrate_scan_system

    normal, _ = _configured_state("normal")
    normal.ac_deflector.scan_pixel_size_nm = fov_nm / normal.ac_deflector.scan_pixels_x
    normal.descan_deflector.descan_constraint = "position_and_angle"
    calibrate_scan_system(normal, force=True, hold=True)
    accepted_record = normal.ac_deflector.calibration_record_json
    accepted_drives = _accepted_drives(normal)
    targets = {"selected_area_aperture": float(normal.selected_area_aperture.z_mm),
               "flu_screen": float(normal.fluorescent_screen.z_mm)}
    normal_measurement = _measure_accepted_drives(normal, targets, fov_nm)
    normal_lenses = {lens.key: lens.percent for lens in normal.lenses if lens.enabled}
    apply_operating_mode_pair(normal, "nano_probe_large_angle", "diffraction",
                              column_name="C3 + Probe Corrector")
    restore_held(normal)
    assert normal.ac_deflector.calibration_record_json == accepted_record
    assert _accepted_drives(normal) == accepted_drives
    held_measurement = _measure_accepted_drives(normal, targets, fov_nm)
    assert normal.ac_deflector.calibration_record_json == accepted_record
    assert _accepted_drives(normal) == accepted_drives

    fresh, _ = _configured_state("large_angle")
    fresh.ac_deflector.scan_pixel_size_nm = fov_nm / fresh.ac_deflector.scan_pixels_x
    fresh.descan_deflector.descan_constraint = "position_and_angle"
    calibrate_scan_system(fresh, force=True, hold=True)
    fresh_drives = _accepted_drives(fresh)
    fresh_measurement = _measure_accepted_drives(fresh, targets, fov_nm)
    record = json.loads(accepted_record)
    if record["ac_angle_target"] != "canonical":
        raise ValueError("Normal held comparison requires accepted canonical AC provenance")
    return {
        "scope": "Production Normal held record retained after applying Large Angle optics, versus independently calibrated Large Angle. No solve is called on the changed held state; local finite response only.",
        "physical_centres_mm": [normal.descan_deflector.upper_z_mm, normal.descan_deflector.lower_z_mm],
        "target_z_mm": targets, "field_of_view_nm": [fov_nm, fov_nm],
        "normal_lens_excitation_percent": normal_lenses,
        "large_angle_lens_excitation_percent": {lens.key: lens.percent for lens in normal.lenses if lens.enabled},
        "normal_accepted_record": record,
        "normal_accepted_drives": accepted_drives,
        "normal_accepted_in_normal_optics": normal_measurement,
        "normal_held_in_large_angle_optics": held_measurement,
        "fresh_large_angle_accepted_drives": fresh_drives,
        "fresh_large_angle_accepted_record": json.loads(fresh.ac_deflector.calibration_record_json),
        "fresh_large_angle_in_large_angle_optics": fresh_measurement,
        "normal_record_and_drives_unchanged_after_mode_switch_and_measurement": True,
        "fresh_versus_normal_logical_command_relative_change": {
            key: _relative(np.asarray(fresh_drives[key]["logical_command_matrix_mrad"])
                           - np.asarray(accepted_drives[key]["logical_command_matrix_mrad"]),
                           np.asarray(accepted_drives[key]["logical_command_matrix_mrad"]))
            for key in accepted_drives},
        "fresh_versus_normal_ac_ratio_relative_change": _relative(
            np.asarray(fresh.ac_deflector.pure_shift_lower_ratio_matrix) - np.asarray(record["ac_ratio"]),
            np.asarray(record["ac_ratio"])),
        "held_versus_fresh_physical_drive_relative_difference": {
            key: _relative(np.asarray(accepted_drives[key]["physical_drive_basis_mrad"])
                           - np.asarray(fresh_drives[key]["physical_drive_basis_mrad"]),
                           np.asarray(fresh_drives[key]["physical_drive_basis_mrad"]))
            for key in accepted_drives},
    }


def _phase_metrics(ac_phase, upper_response, lower_response, drive):
    residual = ac_phase + upper_response @ drive[:2] + lower_response @ drive[2:]
    return {
        "net_position_basis_nm": residual[:2] * 1e9,
        "net_mechanical_slope_basis_mrad": residual[2:] * 1e3,
        "position_relative_residual": _relative(residual[:2], ac_phase[:2]),
        "mechanical_slope_relative_residual": _relative(residual[2:], ac_phase[2:]),
        "peak_net_position_axis_nm": np.sum(np.abs(residual[:2]), axis=1) * 1e9,
        "peak_net_mechanical_slope_axis_mrad": _peak(residual[2:]),
    }


def _solve(ac_phase, upper, lower, condition_length_m, frozen_command_rad, gain):
    """Solve four physical foil channels, with dimensionless row scaling."""
    scale = np.diag((1. / condition_length_m, 1. / condition_length_m, 1., 1.))
    response = np.hstack((upper, lower))
    scaled_response = scale @ response
    condition = float(np.linalg.cond(scaled_response))
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError(f"Ill-conditioned Descan candidate: scaled condition {condition:g}")
    full = np.linalg.solve(scaled_response, -scale @ ac_phase)
    # The legacy position-only option fixes the upper physical drive to -gain
    # times the AC command, leaving two channels for the position equation.
    legacy_upper = -gain * frozen_command_rad
    legacy_lower = np.linalg.solve(lower[:2], -(ac_phase[:2] + upper[:2] @ legacy_upper))
    legacy = np.vstack((legacy_upper, legacy_lower))
    # This separate overdetermined best fit asks if that fixed upper drive can
    # satisfy both constraints at all, not whether the production solver tries.
    lower_scaled = scale @ lower
    rhs = -scale @ (ac_phase + upper @ legacy_upper)
    best_lower, *_ = np.linalg.lstsq(lower_scaled, rhs, rcond=None)
    best = np.vstack((legacy_upper, best_lower))
    return {
        "scaled_condition_number": condition,
        "condition_length_m": condition_length_m,
        "position_row_scaling": "Divide metres by physical coil centre separation in metres; slope rows unchanged",
        "full_phase_space": full,
        "legacy_position_only": legacy,
        "fixed_upper_best_phase_space_least_squares": best,
        "full_upper_change_relative_to_legacy": _relative(full[:2] - legacy_upper, legacy_upper),
        "fixed_upper_best_scaled_relative_residual": _relative(scale @ (ac_phase + response @ best), scale @ ac_phase),
    }


def _case(state, shift_mm, step_mm, perturbation_rad, frozen_command_rad, ac_drives_rad, targets, ac_response=None):
    from temsim.physics.finite_scan_response import finite_scan_responses

    pair = state.descan_deflector
    centres = float(pair.upper_z_mm + shift_mm), float(pair.lower_z_mm + shift_mm)
    planes = tuple(targets.values())
    if ac_response is None:
        ac_response = finite_scan_responses(state, state.ac_deflector, planes,
            maximum_step_mm=step_mm, perturbation_rad=perturbation_rad)
    ds_response = finite_scan_responses(state, pair, planes,
        upper_z_mm=centres[0], lower_z_mm=centres[1],
        maximum_step_mm=step_mm, perturbation_rad=perturbation_rad)
    phases = {}
    for key, z_mm in targets.items():
        au, al, _ = ac_response.at(z_mm)
        du, dl, _ = ds_response.at(z_mm)
        phases[key] = (au @ ac_drives_rad[0] + al @ ac_drives_rad[1], du, dl)
    results = {}
    for key, (ac_phase, upper, lower) in phases.items():
        solved = _solve(ac_phase, upper, lower, (centres[1] - centres[0]) * 1e-3,
                        frozen_command_rad, float(pair.upper_coil_gain))
        entries = {}
        for mode in ("full_phase_space", "legacy_position_only", "fixed_upper_best_phase_space_least_squares"):
            drive = solved.pop(mode)
            upper_peak, lower_peak = _peak(drive[:2]), _peak(drive[2:])
            entries[mode] = {
                "physical_upper_drive_basis_mrad": drive[:2] * 1e3,
                "physical_lower_drive_basis_mrad": drive[2:] * 1e3,
                "upper_peak_axis_mrad": upper_peak,
                "lower_peak_axis_mrad": lower_peak,
                "maximum_axis_drive_mrad": float(max(*upper_peak, *lower_peak)),
                "model_limit_mrad": float(pair.maximum_kick_mrad),
                "within_model_limit": bool(max(*upper_peak, *lower_peak) <= pair.maximum_kick_mrad),
                "observations": {target: _phase_metrics(*phase, drive) for target, phase in phases.items()},
            }
        results[key] = {**solved, "solutions": entries}
    return {
        "shift_mm": float(shift_mm), "centres_mm": centres,
        "coil_support_mm": ds_response.coil_support_mm,
        "physical_host_key": ds_response.physical_host_key,
        "step_mm": float(step_mm), "perturbation_rad": float(perturbation_rad),
        "targets": results,
    }, phases


def _validation(coarse, refined, refined_phases):
    result = {}
    for target in coarse["targets"]:
        old = coarse["targets"][target]["solutions"]["full_phase_space"]
        new = refined["targets"][target]["solutions"]["full_phase_space"]
        old_drive = np.vstack((old["physical_upper_drive_basis_mrad"], old["physical_lower_drive_basis_mrad"])) * 1e-3
        new_drive = np.vstack((new["physical_upper_drive_basis_mrad"], new["physical_lower_drive_basis_mrad"])) * 1e-3
        result[target] = {
            "physical_drive_relative_change": _relative(new_drive - old_drive, old_drive),
            "coarse_drive_in_refined_response": {
                key: _phase_metrics(*phase, old_drive) for key, phase in refined_phases.items()},
        }
    return result


def _ac_pivot_audit(state, ac_drives, fov_nm):
    """Find isolated AC-only common points; never label a stationary interval."""
    from temsim.optics.scan_coil_design import _pivot
    from temsim.physics.finite_scan_response import finite_scan_responses

    start = float(state.sample.z_mm)
    end = float(state.descan_deflector.lower_z_mm + state.descan_deflector.effective_thickness_mm / 2.)
    grid = np.linspace(start, end, int(np.ceil((end-start)/.5)) + 1)

    def responses_at(zs):
        finite = finite_scan_responses(state, state.ac_deflector, zs,
                                       maximum_step_mm=.05, perturbation_rad=1e-6)
        phases = []
        for z in zs:
            upper, lower, _ = finite.at(z)
            phase = upper @ ac_drives[0] + lower @ ac_drives[1]
            phases.append((phase[:2], phase[2:] * 1e-3))
        return phases

    initial = responses_at(grid)
    paths = np.asarray([p[0] for p in initial])
    slopes = np.asarray([p[1] for p in initial])
    desired = np.eye(2) * fov_nm * .5e-9
    result = _pivot(state, None, None, end, grid, paths, None, desired,
                    .05, 1e-6, slopes, responses_at)
    zs = [result.z_mm, *result.other_verified_roots_z_mm]
    verified = responses_at(zs)
    return {"scope": "AC-only scan displacement about the neutral chief, full 2x2 matrix; sampled specimen-to-baseline-lower-Descan-exit interval. A common scan point is not a finite-beam waist. Full-phase Descan cancels the downstream trajectory and does not define one unique isolated pivot.",
            "descan_equivalent_backward_continuation_verified": False,
            "upstream_scope": "No Descan-only backward-equivalent ray was traced. These AC intersections do not imply a causal upstream Descan response or a physical shared tilt centre.",
            "search": asdict(result),
            "selected_and_other_candidates": [{"z_mm": z,
                "position_basis_nm": p * 1e9,
                "mechanical_slope_basis_mrad": derivative * 1e6,
                "nonzero_slope": bool(np.linalg.norm(derivative) > 1e-18)}
                for z, (p, derivative) in zip(zs, verified)]}


def _descan_backward_equivalent_pivots(state, ac_drives, targets, candidate, pivot_audit):
    """Continue the Descan increment virtually through homogeneous optics.

    B(z) contains the four independent AC-foil fundamental responses. The
    continuation B(z) solve(B(SAA), D(SAA)) omits the physical Descan forcing
    upstream. Agreement with -AC follows from the measured SAA cancellation;
    it is a consistency check, not independent evidence or upstream causality.
    """
    from temsim.physics.finite_scan_response import finite_scan_responses

    if not pivot_audit["search"]["is_common_pivot"]:
        return {"verified": False, "reason": "No verified AC common intersection in the searched interval"}
    roots = [float(row["z_mm"]) for row in pivot_audit["selected_and_other_candidates"]]
    target = float(targets["selected_area_aperture"])
    centres = tuple(map(float, candidate["centres_mm"]))
    zs = [*roots, target]
    ac = finite_scan_responses(state, state.ac_deflector, zs, maximum_step_mm=.05)
    ds = finite_scan_responses(state, state.descan_deflector, zs,
        upper_z_mm=centres[0], lower_z_mm=centres[1], maximum_step_mm=.05)
    au, al, _ = ac.at(target)
    du, dl, _ = ds.at(target)
    fundamental = np.hstack((au, al))
    length_m = float(state.ac_deflector.lower_z_mm-state.ac_deflector.upper_z_mm) * 1e-3
    scale = np.diag((1./length_m, 1./length_m, 1., 1.))
    condition = float(np.linalg.cond(scale @ fundamental))
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError("AC fundamental response is singular or ill-conditioned for virtual backward continuation")
    fit = candidate["targets"]["selected_area_aperture"]["solutions"]["full_phase_space"]
    ku = np.asarray(fit["physical_upper_drive_basis_mrad"]) * 1e-3
    kl = np.asarray(fit["physical_lower_drive_basis_mrad"]) * 1e-3
    downstream_descan = du @ ku + dl @ kl
    equivalent = np.linalg.solve(scale @ fundamental, scale @ downstream_descan)
    physical_ac = np.vstack(ac_drives)
    tolerance = float(pivot_audit["search"].get("relative_tolerance", 1e-6))
    # The FOV basis is the same per-axis normalization used to verify AC roots.
    sample_upper, sample_lower, _ = finite_scan_responses(state, state.ac_deflector,
        [float(state.sample.z_mm)], maximum_step_mm=.05).at(float(state.sample.z_mm))
    sample_map = np.hstack((sample_upper, sample_lower)) @ physical_ac
    half_extents = np.linalg.norm(sample_map[:2], axis=0)
    if np.any(half_extents <= 0) or not np.isfinite(half_extents).all():
        raise ValueError("Virtual pivot verification requires nonzero finite specimen scan axes")
    observations = []
    for z in roots:
        au, al, _ = ac.at(z)
        du, dl, _ = ds.at(z)
        basis = np.hstack((au, al))
        ac_phase = basis @ physical_ac
        virtual = basis @ equivalent
        physical_descan = du @ ku + dl @ kl
        error = virtual + ac_phase
        relative_position = float(np.max(np.linalg.norm(virtual[:2] / half_extents[None, :], axis=0)))
        nonzero = bool(np.linalg.norm(virtual[2:]) > 1e-12)
        observations.append({"z_mm": z,
            "virtual_position_basis_nm": virtual[:2] * 1e9,
            "virtual_mechanical_slope_basis_mrad": virtual[2:] * 1e3,
            "virtual_plus_ac_position_residual_nm": float(np.linalg.norm(error[:2])) * 1e9,
            "virtual_plus_ac_slope_residual_urad": float(np.linalg.norm(error[2:])) * 1e6,
            "virtual_position_relative_to_sample_half_fov": relative_position,
            "nonzero_virtual_slope": nonzero,
            "physical_descan_upstream_response_is_zero": bool(np.array_equal(physical_descan, np.zeros((4, 2)))),
            "verified_common_virtual_intersection": bool(relative_position <= tolerance and nonzero)})
    return {
        "scope": "Virtual homogeneous backward optical continuation of Descan-only SAA increments, not physical upstream Descan action and not a finite-beam waist.",
        "dependence": "B(z) solve(B(SAA), D(SAA)) uses the same finite linear responses and measured SAA cancellation. Matching -AC is a dependent consistency check, not independent validation.",
        "target_z_mm": target, "physical_descan_centres_mm": centres,
        "condition_length_m": length_m, "scaled_fundamental_condition_number": condition,
        "relative_position_tolerance": tolerance,
        "saa_total_position_residual_nm": float(np.linalg.norm((fundamental @ physical_ac + downstream_descan)[:2])) * 1e9,
        "saa_total_slope_residual_urad": float(np.linalg.norm((fundamental @ physical_ac + downstream_descan)[2:])) * 1e6,
        "observations": observations,
        "verified": all(row["verified_common_virtual_intersection"] for row in observations),
    }


def run(args):
    from temsim.instrument_snapshot import encode_instrument
    from temsim.physics.finite_scan_response import finite_scan_responses
    from temsim.physics.scan_geometry import calibrate_ac_scan_scale

    started = time.perf_counter()
    selected_range = getattr(args, "nanoprobe_range", "normal")
    if getattr(args, "compare_normal_held", False) and selected_range != "large_angle":
        raise ValueError("Normal held comparison requires the Large Angle study")
    output = ROOT / args.output
    previous_path = output / "study.json"
    if previous_path.is_file():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("nanoprobe_range") != selected_range:
            raise ValueError("Study output already belongs to a different Nanoprobe range")
    state, mode = _configured_state(selected_range)
    host = state.image_diffraction_deflector
    installed_centres = (float(host.upper_z_mm), float(host.lower_z_mm))
    baseline = getattr(args, "baseline_pair_center_mm", None)
    if baseline is not None:
        # Use the shared hardware owner. Avoid a full layout rebuild here:
        # its SAA anchor would otherwise move the observation plane too.
        host.mechanical_center_below_sample_mm += float(baseline) - .5 * sum(installed_centres)
    command_small_mrad, ac_scale_residual = calibrate_ac_scan_scale(state)
    fov_small = np.asarray((state.ac_deflector.scan_field_of_view_x_nm,
                           state.ac_deflector.scan_field_of_view_y_nm))
    command_rad = np.asarray(command_small_mrad) * 1e-3 @ np.diag(np.full(2, args.fov_nm) / fov_small)
    ac_maps = tuple(map(np.asarray, state.ac_deflector.coil_kick_matrices()))
    ac_drives = tuple(mapping @ command_rad for mapping in ac_maps)
    targets = {"selected_area_aperture": float(state.selected_area_aperture.z_mm),
               "flu_screen": float(state.fluorescent_screen.z_mm)}
    # The local derivative helper uses min(state.step_mm, maximum_step_mm).
    # A .1 mm state ceiling permits actual .1 versus .05 comparisons while
    # retaining the AC matrices already accepted above at .05 mm.
    state.step_mm = .1
    frozen = encode_instrument(state)
    report = {
        "scope": "First-order local finite-coil derivatives of the production field plan. Diagnostic candidate centres and independent foil commands only; no saved placement or calibration edits, no clearance qualification or finite-source/finite-amplitude validation.",
        "voltage_kv": state.beam_voltage_kv, "illumination_mode": state.illumination_mode,
        "nanoprobe_range": state.nanoprobe_range, "condenser_preset": mode,
        "projector_mode": state.projector_mode, "field_of_view_nm": [args.fov_nm, args.fov_nm],
        "target_z_mm": targets, "sample_z_mm": state.sample.z_mm,
        "installed_descan_centres_mm": installed_centres,
        "baseline_descan_centres_mm": (float(host.upper_z_mm), float(host.lower_z_mm)),
        "baseline_pair_center_override_mm": baseline,
        "placement_reference": "Candidate shifts are relative to baseline_descan_centres_mm; installed SAA and screen targets stay fixed.",
        "column_module": dict(state._resolved_assembly.selected_module_paths).get("column"),
        "frozen_ac": {"calibration_step_mm": .05, "angle_target": "canonical",
            "scale_residual": ac_scale_residual, "logical_command_basis_mrad": command_rad * 1e3,
            "upper_physical_drive_basis_mrad": ac_drives[0] * 1e3,
            "lower_physical_drive_basis_mrad": ac_drives[1] * 1e3,
            "upper_peak_axis_mrad": _peak(ac_drives[0]), "lower_peak_axis_mrad": _peak(ac_drives[1])},
        "lens_excitation_percent": {lens.key: lens.percent for lens in state.lenses if lens.enabled},
        "baseline_candidates": [], "validations": [],
    }
    coarse_ac = finite_scan_responses(state, state.ac_deflector, tuple(targets.values()),
                                      maximum_step_mm=.1, perturbation_rad=1e-6)
    for shift in args.shifts:
        row, _ = _case(state, shift, .1, 1e-6, command_rad, ac_drives, targets, coarse_ac)
        report["baseline_candidates"].append(row)
        saa = row["targets"]["selected_area_aperture"]
        full = saa["solutions"]["full_phase_space"]
        legacy = saa["solutions"]["legacy_position_only"]
        print(json.dumps({"shift_mm": shift, "condition": saa["scaled_condition_number"],
            "full_peak_mrad": full["maximum_axis_drive_mrad"],
            "legacy_peak_mrad": legacy["maximum_axis_drive_mrad"],
            "legacy_target_angle_relative_residual": legacy["observations"]["selected_area_aperture"]["mechanical_slope_relative_residual"],
            "full_upper_change_relative_to_legacy": saa["full_upper_change_relative_to_legacy"]}), flush=True)
        _write(output / "study.json", report)
    if not args.skip_validation:
        ranked = sorted(report["baseline_candidates"], key=lambda item:
            item["targets"]["selected_area_aperture"]["solutions"]["full_phase_space"]["maximum_axis_drive_mrad"])
        selected = (list(dict.fromkeys(args.shifts)) if getattr(args, "validate_all_candidates", False)
                    else list(dict.fromkeys(([0.] if 0. in args.shifts else []) + [ranked[0]["shift_mm"]])))
        for h in (1e-6, .5e-6, 2e-6):
            fine_ac = finite_scan_responses(state, state.ac_deflector, tuple(targets.values()),
                                           maximum_step_mm=.05, perturbation_rad=h)
            for shift in selected:
                original = next(row for row in report["baseline_candidates"] if row["shift_mm"] == shift)
                refined, phases = _case(state, shift, .05, h, command_rad, ac_drives, targets, fine_ac)
                report["validations"].append({"case": refined, "versus_coarse": _validation(original, refined, phases)})
                print(json.dumps({"validation_shift_mm": shift, "step_mm": .05,
                    "perturbation_rad": h,
                    "drive_relative_change": report["validations"][-1]["versus_coarse"]["selected_area_aperture"]["physical_drive_relative_change"]}), flush=True)
                _write(output / "study.json", report)
    if args.pivot_audit:
        report["ac_only_pivot_audit"] = _ac_pivot_audit(state, ac_drives, args.fov_nm)
        print(json.dumps({"ac_only_pivot": report["ac_only_pivot_audit"]["search"]}, default=_json), flush=True)
    if getattr(args, "backward_equivalent_pivot_audit", False):
        if not args.pivot_audit:
            raise ValueError("Backward-equivalent pivot check requires --pivot-audit")
        options = [row["case"] for row in report["validations"]
            if row["case"]["step_mm"] == .05 and row["case"]["perturbation_rad"] == 1e-6]
        options = options or report["baseline_candidates"]
        candidate = min(options, key=lambda row:
            float(np.linalg.norm(np.asarray(row["centres_mm"])-np.asarray(installed_centres))))
        report["descan_backward_equivalent_pivot_audit"] = _descan_backward_equivalent_pivots(
            state, ac_drives, targets, candidate, report["ac_only_pivot_audit"])
        report["ac_only_pivot_audit"]["descan_equivalent_backward_continuation_verified"] = report[
            "descan_backward_equivalent_pivot_audit"]["verified"]
        report["ac_only_pivot_audit"]["upstream_scope"] = (
            "Separately checked virtual Descan backward continuation is recorded below; "
            "it does not imply causal upstream Descan action or a finite-beam waist.")
    if getattr(args, "compare_normal_held", False):
        report["normal_held_comparison"] = _normal_held_comparison(args.fov_nm)
        print(json.dumps({"normal_held_comparison_drive_relative_difference":
            report["normal_held_comparison"]["held_versus_fresh_physical_drive_relative_difference"]}), flush=True)
    report["input_state_unchanged_after_candidate_study"] = encode_instrument(state) == frozen
    assert report["input_state_unchanged_after_candidate_study"]
    report["elapsed_seconds"] = time.perf_counter() - started
    _write(output / "study.json", report)
    print(f"Saved {output / 'study.json'} ({report['elapsed_seconds']:.1f} s)", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/descan_optimization_20261018")
    parser.add_argument("--fov-nm", type=float, default=1000.)
    parser.add_argument("--nanoprobe-range", choices=("normal", "large_angle"), default="normal")
    parser.add_argument("--baseline-pair-center-mm", type=float, default=None,
        help="Override only the runtime shared pair centre while preserving installed separation and fixed target planes; historical centre: 1786.9 mm")
    parser.add_argument("--shifts", type=float, nargs="+", default=[0., -60., -40., -20., 10.])
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-all-candidates", action="store_true")
    parser.add_argument("--pivot-audit", action="store_true")
    parser.add_argument("--backward-equivalent-pivot-audit", action="store_true")
    parser.add_argument("--compare-normal-held", action="store_true",
        help="Compare a literal Normal held calibration under Large Angle optics against a fresh Large Angle calibration at installed geometry")
    args = parser.parse_args()
    if not np.isfinite(args.fov_nm) or args.fov_nm <= 0:
        parser.error("FOV must be finite and positive")
    if args.baseline_pair_center_mm is not None and not np.isfinite(args.baseline_pair_center_mm):
        parser.error("Baseline pair centre must be finite")
    if not all(np.isfinite(args.shifts)) or not args.shifts:
        parser.error("Candidate shifts must be nonempty and finite")
    if args.compare_normal_held and args.nanoprobe_range != "large_angle":
        parser.error("Normal held comparison requires --nanoprobe-range large_angle")
    if args.backward_equivalent_pivot_audit and not args.pivot_audit:
        parser.error("Backward-equivalent pivot check requires --pivot-audit")
    run(args)


if __name__ == "__main__":
    main()
