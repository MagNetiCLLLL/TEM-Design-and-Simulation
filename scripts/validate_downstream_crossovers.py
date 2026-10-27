"""Unchanged-source vacuum optical reference from specimen entrance to screen.

This extends saved, executed source particles through the production shared
field and physical receiver stops. It deliberately does not scatter them in
the specimen and therefore cannot represent a Si diffraction pattern or image.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from temsim.calculation_manifest import solver_source_identity
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.crossover_acceptance import evaluate_crossover_chain
from temsim.optics.model import State
from temsim.physics.beam_current import effective_source_current_a
from temsim.physics.particle_sections import _digest_arrays


def clean(value):
    if isinstance(value, Mapping):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(arrays):
    """Verify the retained exact gun-exit population, including original IDs."""
    return _digest_arrays(arrays["x_m"][0], arrays["tx_rad"][0], arrays["y_m"][0],
        arrays["ty_rad"][0], arrays["flight_time_s"][0], arrays["source_energy_offset_ev"],
        arrays["weight"], arrays["alive"][0], arrays["source_id"])


def downstream_regions(state):
    """Bounded source-waist intervals; lens centres are labels, not field ends."""
    start, stop = float(state.sample.upper_surface_z_mm), float(state.fluorescent_screen.z_mm)
    if not np.isfinite([start, stop]).all() or start >= stop:
        raise ValueError("The physical main-screen reference must follow the specimen")
    groups = {}
    for lens in state.lenses:
        if (lens.enabled and getattr(lens, "installed", True)
                and start+.001 < float(lens.z_mm) < stop):
            groups.setdefault(float(lens.z_mm), []).append(lens.key)
    boundaries = [(start, "specimen_entrance_inside_objective_field")]
    boundaries += [(z, "+".join(sorted(keys))) for z, keys in sorted(groups.items())]
    boundaries.append((stop, "main_screen_reference"))
    return {left[1]+"__"+right[1]: dict(bounds_mm=[left[0], right[0]], requirement="record_all_waists")
            for left, right in zip(boundaries, boundaries[1:])}


def requested_planes(state, *, spacing_mm=.5):
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Observation spacing must be finite and positive")
    start, stop = float(state.sample.upper_surface_z_mm), float(state.fluorescent_screen.z_mm)
    pieces = [np.linspace(start, stop, int(np.ceil((stop-start)/spacing_mm))+1)]
    for collection in (state.lenses, state.apertures, state.recording_planes):
        pieces.append(np.array([float(part.z_mm) for part in collection
                               if getattr(part, "installed", True) and start < part.z_mm < stop]))
    return np.unique(np.concatenate(pieces))


def propagate_archive(state, arrays, planes, *, step_mm):
    """Continue actual entrance states with unchanged energy, time and weights.

    Mechanical aperture/bore action belongs to the shared-field executor.
    Existing production receiver clipping is applied once at exact retained
    receiver coordinates. An absorbing receiver is observed at arrival, before
    absorption; particles intercepted there remain absent at later planes.
    """
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics.unified_field_particle import trace_unified_response
    from temsim.simulation_modes import is_unified

    before = capture_instrument_snapshot(state)
    current = before.restore()
    if (not is_unified(current) or current.vacuum_map.enabled or current.energy_filter.enabled
            or any(coil.dynamic for coil in column_dipole_fields(current))):
        raise ValueError("Downstream reference requires static shared fields, no medium, and no bent energy filter")
    start = float(current.sample.upper_surface_z_mm)
    planes = np.asarray(planes, float)
    if (planes.ndim != 1 or len(planes) < 3 or not np.isfinite(planes).all()
            or np.any(np.diff(planes) <= 0) or abs(planes[0]-start) > 1e-9
            or abs(planes[-1]-current.fluorescent_screen.z_mm) > 1e-9
            or isinstance(step_mm, bool) or not np.isfinite(step_mm) or step_mm <= 0):
        raise ValueError("Ordered exact downstream planes and a positive step are required")
    if abs(float(arrays["z_mm"][-1])-start) > 1e-9:
        raise ValueError("Saved entrance must match the actual specimen upper surface")
    weights = np.asarray(arrays["weight"], float)
    active = np.asarray(arrays["alive"][-1], bool) & (weights > 0)
    chosen = np.flatnonzero(active)
    count = len(weights)
    if not len(chosen):
        raise ValueError("No original source particles reached the specimen entrance")
    phase0 = np.column_stack([arrays[key][-1] for key in ("x_m", "y_m", "tx_rad", "ty_rad")])
    kinetic0, time0 = arrays["kinetic_energy_ev"][-1], arrays["flight_time_s"][-1]
    if not all(np.isfinite(value[active]).all() for value in (phase0, kinetic0, time0)):
        raise ValueError("Actual surviving entrance coordinates, kinetic energy and clocks are required")
    result = trace_unified_response(current, phase0[chosen], start*.001, planes[1:]*.001,
        energy_ev=kinetic0[chosen], flight_time_s=time0[chosen], maximum_step_m=step_mm*.001)
    full_phase = np.full((len(planes), count, 4), np.nan)
    full_kinetic = np.full((len(planes), count), np.nan)
    full_clock = np.full_like(full_kinetic, np.nan)
    full_phase[0], full_kinetic[0], full_clock[0] = phase0, kinetic0, time0
    full_phase[1:, chosen] = result["xy_slopes"]
    full_kinetic[1:, chosen] = result["kinetic_energy_ev"]
    full_clock[1:, chosen] = result["flight_time_s"]
    blocked = np.full(count, np.nan)
    labels = [""]*count
    blocked[chosen] = np.asarray(result["blocked_z_m"])*1000.
    for local, original in enumerate(chosen):
        labels[original] = result["blocked_key"][local]
    # Only entrance survivors are passed to receiver clipping, avoiding the
    # legacy receiver helper's final mask from resurrecting a prior gun loss.
    _, chosen_stops, chosen_labels = clip_recording_planes(current, planes,
        full_phase[:, chosen, 0], full_phase[:, chosen, 1],
        ~np.isfinite(blocked[chosen]), blocked[chosen], [labels[i] for i in chosen])
    blocked[chosen] = chosen_stops
    for local, original in enumerate(chosen):
        labels[original] = chosen_labels[local]
    receiver_keys = {part.key for part in current.recording_planes if part.inserted}
    receiver_arrival = np.array([key in receiver_keys for key in labels])[None, :] & (
        planes[:, None] == blocked[None, :])
    masks = active[None, :] & (~np.isfinite(blocked)[None, :]
                              | (planes[:, None] < blocked[None, :]) | receiver_arrival)
    if capture_instrument_snapshot(state).digest != before.digest:
        raise ValueError("The downstream reference changed physical inputs")
    return dict(z_mm=planes.copy(), x_m=full_phase[:, :, 0], y_m=full_phase[:, :, 1],
        tx_rad=full_phase[:, :, 2], ty_rad=full_phase[:, :, 3], kinetic_energy_ev=full_kinetic,
        flight_time_s=full_clock, alive=masks, weight=weights.copy(), source_id=arrays["source_id"].copy(),
        source_energy_offset_ev=arrays["source_energy_offset_ev"].copy(),
        blocked_z_mm=blocked, blocked_key=np.array(labels, dtype=str),
        receipt=result["receipt"], original_snapshot_digest=before.digest)


def evaluate_capture(state, arrays, identity):
    cp = SimpleNamespace(**{key: arrays[key] for key in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad")})
    report = evaluate_crossover_chain(cp, arrays["alive"], arrays["weight"], arrays["source_id"],
        source_current_a=effective_source_current_a(state), regions=downstream_regions(state),
        maximum_spacing_mm=.5, source_identity=identity)
    report["schema"] = "downstream-optical-reference-crossover-chain-v1"
    report["screen_reference"] = report.pop("specimen")
    report["screen_reference"]["region"] = "main_screen_reference"
    report["screen_reference"]["status"] = "EXACT_REFERENCE_PLANE_NOT_A_SCREEN_EXPOSURE"
    for row in report["angular_relationships"]:
        if row["to_region"] == "specimen_entrance":
            row["to_region"] = "main_screen_reference"
    report["scope"] = dict(specimen_interaction="OMITTED_VACUUM_OPTICAL_REFERENCE",
        original_source_ancestry="Same saved source IDs, weights, kinetic energy and flight time at entrance",
        incident_column="Reused historical executed entrance; not retraced here",
        scan="STATIC_REFERENCE_NOT_DYNAMIC_SCAN_VALIDATION",
        conjugacy="NOT_QUALIFIED_EMPIRICAL_PARTICLE_REGRESSION_ONLY",
        detector_observations="Exact receiver arrivals before absorption; upstream inserted receivers still intercept",
        main_screen_inserted=bool(state.fluorescent_screen.inserted),
        pupil_edge="NOT_MEASURED_BY_WEIGHTED_PARTICLE_ANGLES",
        particle_count_convergence="NOT_ESTABLISHED_BY_THIS_193_PARTICLE_REFERENCE")
    return report


def refined_planes(planes, report):
    additions = []
    for candidate in report["interpolated_candidates_diagnostic_only"]:
        left, right = candidate["bracket_mm"]
        centre = candidate["z_mm"]
        half = min(.1, (right-left)/4, (centre-planes[0])/2, (planes[-1]-centre)/2)
        if half > 1e-10:
            additions.extend(centre+np.linspace(-half, half, 9))
    return np.unique(np.r_[planes, additions])


def compare_steps(coarse, fine):
    """Compare identical observation planes and original particle identities."""
    from temsim.physics.beam_statistics import transverse_beam_statistics
    if not np.array_equal(coarse["source_id"], fine["source_id"]) or not np.array_equal(coarse["weight"], fine["weight"]):
        raise ValueError("Step comparison must retain the identical source population")
    indices = np.searchsorted(fine["z_mm"], coarse["z_mm"])
    if np.any(indices >= len(fine["z_mm"])) or not np.array_equal(fine["z_mm"][indices], coarse["z_mm"]):
        raise ValueError("Step comparison requires the same exact physical planes")
    same = coarse["alive"] & fine["alive"][indices]
    position_errors, angle_errors, energy_errors = [], [], []
    for key, collection, scale in (("x_m", position_errors, 1e9), ("y_m", position_errors, 1e9),
            ("tx_rad", angle_errors, 1e3), ("ty_rad", angle_errors, 1e3),
            ("kinetic_energy_ev", energy_errors, 1.)):
        errors = np.abs(coarse[key]-fine[key][indices])[same]
        if len(errors):
            collection.append(float(np.max(errors))*scale)
    quantile_error, radius_error, fraction_error = [], [], []
    for i, j in enumerate(indices):
        if not coarse["alive"][i].any() or not fine["alive"][j].any():
            continue
        a, b = [transverse_beam_statistics(*(values[key][index] for key in
            ("x_m", "y_m", "tx_rad", "ty_rad")), alive=values["alive"][index], weights=values["weight"])
            for values, index in ((coarse, i), (fine, j))]
        quantile_error.append(abs(a.convergence_95_rad-b.convergence_95_rad)*1e3)
        radius_error.append(abs(a.radius_rms_m-b.radius_rms_m)*1e9)
        fraction_error.append(abs(a.surviving_fraction-b.surviving_fraction))
    return dict(status="BOUNDED_SAME_ENTRANCE_STEP_COMPARISON",
        changed_particle_plane_survival_count=int(np.count_nonzero(coarse["alive"] != fine["alive"][indices])),
        maximum_same_particle_position_coordinate_change_nm=max(position_errors, default=None),
        maximum_same_particle_mechanical_slope_coordinate_change_mrad=max(angle_errors, default=None),
        maximum_same_particle_kinetic_energy_change_ev=max(energy_errors, default=None),
        maximum_actual_alpha95_change_mrad=max(quantile_error, default=None),
        maximum_actual_rms_radius_change_nm=max(radius_error, default=None),
        maximum_actual_source_current_fraction_change=max(fraction_error, default=None),
        exact_compared_planes=len(indices),
        scope="Downstream integration and observation mesh only; historical upstream error is retained in both runs")


def sample_region_diagnostic(observations, report):
    """Explain an unresolved near-specimen radial minimum using executed data."""
    from temsim.physics.beam_statistics import transverse_beam_statistics
    z = observations["z_mm"]
    candidates = [row for row in report["unresolved_candidates"] if row["z_mm"] < z[0]+.001]
    if not candidates:
        return dict(status="NO_UNRESOLVED_NEAR_SPECIMEN_CANDIDATE")
    candidate = candidates[0]
    index = int(np.argmin(abs(z-candidate["z_mm"])))
    if abs(z[index]-candidate["z_mm"]) > 1e-9:
        return dict(status="CANDIDATE_NOT_AN_EXECUTED_PLANE", candidate=candidate)
    live = observations["alive"][index] & (observations["weight"] > 0)
    weights = observations["weight"][live]
    weights = weights/weights.sum()
    position = np.stack((observations["x_m"][index, live], observations["y_m"][index, live]))
    slope = np.stack((observations["tx_rad"][index, live], observations["ty_rad"][index, live]))
    position -= (position@weights)[:, None]
    slope -= (slope@weights)[:, None]
    cross = np.einsum("ir,jr,r->ij", position, slope, weights)
    directional = np.linalg.eigvalsh((cross+cross.T)/2)
    stats = transverse_beam_statistics(*(observations[key][index] for key in
        ("x_m", "y_m", "tx_rad", "ty_rad")), weights=observations["weight"], alive=live)
    return dict(status="UNRESOLVED_TWO_DIMENSIONAL_WAIST", exact_observation_z_mm=float(z[index]),
        entrance_offset_nm=float((z[index]-z[0])*1e6), radial_rms_nm=stats.radius_rms_m*1e9,
        alpha95_mrad=stats.convergence_95_mrad,
        symmetric_position_slope_covariance_eigenvalues_m_rad=directional.tolist(),
        principal_rms_nm=(np.sqrt(np.linalg.eigvalsh(np.einsum(
            "ir,jr,r->ij", position, position, weights)))*1e9).tolist(),
        opposite_directional_covariance_signs=bool(directional[0] < 0 < directional[-1]),
        interpretation="A radial minimum alone does not confirm simultaneous focus in every transverse direction")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--particle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-mm", type=float, default=.05)
    args = parser.parse_args()
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/"report.json"
    if target.exists():
        raise FileExistsError("Preserve completed validation; select a new output directory")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    prior = json.loads(args.particle_report.read_text(encoding="utf-8"))
    if json.loads(args.particle_report.with_suffix(".inputs.json").read_text(encoding="utf-8")) != inputs:
        raise ValueError("Archived source report must retain the exact current input settings")
    state = State.from_dict(inputs)
    if state.to_dict() != inputs:
        raise ValueError("Restoring inputs changed physical parameters")
    before = capture_instrument_snapshot(state)
    implementation = solver_source_identity()
    with np.load(args.particles, allow_pickle=False) as raw:
        source = {key: raw[key].copy() for key in raw.files}
    if (source_identity(source) != prior["source_population_identity"]
            or abs(float(source["z_mm"][0])-state.electron_gun.exit_plane_z_mm) > 1e-9
            or len(source["weight"]) != prior["rays"]):
        raise ValueError("Archived particle identity, count or exact gun exit is inconsistent")
    identity = prior["source_population_identity"]
    cases, case_observations = {}, {}
    planes = requested_planes(state)
    for label, step in (("baseline", args.step_mm), ("half_step_refined", args.step_mm/2)):
        if label != "baseline":
            planes = refined_planes(planes, cases["baseline"]["report"])
        print(f"Executing {label}: {len(source['weight'])} original particles; {step:g} mm; {len(planes)} checkpoints", flush=True)
        begin = perf_counter()
        observations = propagate_archive(state, source, planes, step_mm=step)
        report = evaluate_capture(state, observations, identity)
        metadata = {key: value for key, value in observations.items() if not isinstance(value, np.ndarray)}
        np.savez_compressed(args.output/(label+".npz"), **{
            key: value for key, value in observations.items() if isinstance(value, np.ndarray)})
        write_json(args.output/(label+".json"), dict(report=report, transport=metadata,
            elapsed_s=perf_counter()-begin, maximum_step_mm=step))
        cases[label] = dict(report=report, maximum_step_mm=step, elapsed_s=perf_counter()-begin)
        case_observations[label] = observations
        print(json.dumps(clean(dict(case=label, elapsed_s=cases[label]["elapsed_s"], status=report["status"],
            waists=[(row["z_mm"], row["actual"]["alpha95_mrad"]) for row in report["executed_waists"]],
            unresolved=report["unresolved_candidates"], screen=report["screen_reference"]["actual"]))), flush=True)
    if capture_instrument_snapshot(state).digest != before.digest or state.to_dict() != inputs:
        raise ValueError("Validation changed the original optical settings")
    report = dict(schema="extended-downstream-optical-validation-v1", utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=perf_counter()-started, source_population_identity=identity,
        unchanged_physical_inputs=True, snapshot_digest=before.digest, solver_identity=implementation,
        input_sha256=sha256(args.inputs), historical_particle_sha256=sha256(args.particles),
        historical_particle_report_sha256=sha256(args.particle_report),
        specimen_interaction="No material scattering: unchanged optical vacuum reference through the specimen region",
        current_a=effective_source_current_a(state), main_screen_reference_z_mm=state.fluorescent_screen.z_mm,
        main_screen_inserted=bool(state.fluorescent_screen.inserted),
        recorded_image=False, cases=cases,
        step_comparison=compare_steps(case_observations["baseline"], case_observations["half_step_refined"]),
        sample_region_diagnostic=sample_region_diagnostic(case_observations["half_step_refined"],
                                                         cases["half_step_refined"]["report"]),
        qualification="Bounded unchanged-source optical observations; not Si scattering, image validation or source-count convergence")
    write_json(target, report)
    print(str(target.resolve()), flush=True)


if __name__ == "__main__":
    main()
