"""Measure a frozen working point's physical pupil and joint incident optics.

Example (run with PYTHONPATH=src):
  python scripts/measure_optical_acceptance.py --inputs INPUTS.json
    --particles OBSERVATIONS.npz --particle-report OBSERVATIONS.json --output DIRECTORY

Inputs are never fitted or saved over. Local diagnostic caches are content
addressed by the complete snapshot, implementation, launch states and planes.
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

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.model import State
from temsim.optics.pupil_acceptance import trace_tip_probes, measure_radial_boundary, summarize_boundary, direction_angles
from temsim.optics.crossover_acceptance import evaluate_crossover_chain, regions_for_state
from temsim.physics.beam_current import effective_source_current_a
from temsim.calculation_manifest import solver_source_identity


def clean(value):
    if isinstance(value, Mapping):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--particles", type=Path, required=True)
    parser.add_argument("--particle-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Reuse content-addressed diagnostics from an earlier output")
    parser.add_argument("--azimuths", type=int, default=16)
    parser.add_argument("--refinements", type=int, default=4)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--target-mrad", type=float, default=30.)
    args = parser.parse_args()
    started = perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output/"report.json"
    if destination.exists():
        raise FileExistsError("Preserve completed reports; select a new output directory")
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    state = State.from_dict(inputs)
    if state.to_dict() != inputs:
        raise ValueError("Restoring the input changed its physical settings")
    snapshot = capture_instrument_snapshot(state)
    implementation = solver_source_identity()
    archive_report = json.loads(args.particle_report.read_text(encoding="utf-8"))
    archived_inputs = args.particle_report.with_suffix(".inputs.json")
    if not archived_inputs.exists() or json.loads(archived_inputs.read_text(encoding="utf-8")) != inputs:
        raise ValueError("Archived particle inputs must match every restored physical setting")
    # Snapshot digests also contain implementation identities. The new audit
    # modules necessarily change that identity; preserve the original receipt
    # and explicitly label the existing physical population as historical.
    with np.load(args.particles, allow_pickle=False) as saved:
        arrays = {key:saved[key].copy() for key in saved.files}
    from temsim.physics.particle_sections import _digest_arrays
    source_identity = _digest_arrays(*(arrays[key][0] for key in
        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s")),
        arrays["source_energy_offset_ev"], arrays["weight"], arrays["alive"][0], arrays["source_id"])
    if (source_identity != archive_report["source_population_identity"]
            or len(arrays["weight"]) != archive_report["rays"]
            or arrays["z_mm"][0] != state.electron_gun.exit_plane_z_mm
            or arrays["z_mm"][-1] != state.sample.upper_surface_z_mm):
        raise ValueError("Particle archive does not match its executed source and observation receipt")
    cp = SimpleNamespace(**{key:arrays[key] for key in
        ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev")})
    chain = evaluate_crossover_chain(cp, arrays["alive"], arrays["weight"], arrays["source_id"],
        source_current_a=effective_source_current_a(state), regions=regions_for_state(state),
        source_identity=archive_report["source_population_identity"])
    waists = [row["z_mm"] for row in chain["executed_waists"]]
    apertures = [a for a in state.apertures if a.enabled and getattr(a, "installed", True) and getattr(a, "inserted", True)
        and state.electron_gun.exit_plane_z_mm < a.z_mm < state.sample.upper_surface_z_mm]
    planes = np.unique([*waists, *(a.z_mm for a in apertures), state.sample.upper_surface_z_mm])
    cache = args.cache_dir if args.cache_dir is not None else args.output/"cache"
    cache.mkdir(parents=True, exist_ok=True)
    original_timing_path = cache.parent/"progress.json"
    original_timings = (json.loads(original_timing_path.read_text(encoding="utf-8"))
                        if args.cache_dir is not None and original_timing_path.exists() else [])
    log, executed = [], []

    def observe(launch, label, step=args.step_mm):
        payload = dict(snapshot=snapshot.digest, implementation=implementation,
            launch=np.asarray(launch).tolist(), planes=planes.tolist(), step_mm=step,
            emission_energy_ev=state.electron_gun.emitter.emission_energy_ev)
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = cache/(key+".json")
        start = perf_counter()
        if path.exists() and path.with_suffix(".npz").exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as raw:
                result.update({k:raw[k].copy() for k in raw.files})
            reused = True
        else:
            result = trace_tip_probes(state, launch, planes, step_mm=step)
            numeric = {k:v for k,v in result.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(path.with_suffix(".npz"), **numeric)
            save_json(path, {k:v for k,v in result.items() if k not in numeric})
            reused = False
        record = dict(label=label, rays=len(launch), elapsed_s=perf_counter()-start,
            cache_reused=reused, cache_key=key, transmitted=int(result["alive"][-1].sum()))
        log.append(record)
        save_json(args.output/"progress.json", log)
        print(json.dumps(record), flush=True)
        return result

    def radial_trace(radius, phi):
        launch = np.column_stack((np.zeros((len(radius),2)), radius*np.cos(phi), radius*np.sin(phi)))
        trace = observe(launch, f"radial_search_{len(executed)}")
        executed.append(trace)
        return dict(transmitted=trace["alive"][-1], blocked_key=trace["blocked_key"])

    boundary = measure_radial_boundary(radial_trace, azimuth_count=args.azimuths,
                                      refinements=args.refinements)
    save_json(args.output/"boundary.json", boundary)
    chief = observe(np.zeros((1,4)), "chief_ray")
    pupil = dict(status="NOT_MEASURED", target_mrad=args.target_mrad, alpha_edge_mrad=None)
    convergence = dict(status="NOT_MEASURED")
    rim = None
    if boundary["status"] == "BRACKETED":
        phi = np.array([row["azimuth_rad"] for row in boundary["sectors"]])
        radius = np.array([row["accepted_launch_slope"] for row in boundary["sectors"]])
        launch = np.column_stack((np.zeros((len(phi),2)), radius*np.cos(phi), radius*np.sin(phi)))
        rim = observe(launch, "accepted_rim")
        pupil = summarize_boundary(boundary, rim, chief, target_mrad=args.target_mrad)
        fine = observe(launch, "accepted_rim_half_step", args.step_mm/2)
        fine_chief = observe(np.zeros((1,4)), "chief_half_step", args.step_mm/2)
        if fine["alive"].all():
            refined = summarize_boundary(boundary, fine, fine_chief, target_mrad=args.target_mrad)
            coarse_angle = np.array(pupil["specimen"]["alpha_edge_by_source_azimuth_mrad"])
            fine_angle = np.array(refined["specimen"]["alpha_edge_by_source_azimuth_mrad"])
            convergence = dict(status="BOUNDED_STEP_CHECK", maximum_change_mrad=float(np.max(abs(fine_angle-coarse_angle))),
                fraction_of_30_mrad=float(np.max(abs(fine_angle-coarse_angle))/args.target_mrad),
                coarse_step_mm=args.step_mm, fine_step_mm=args.step_mm/2,
                radial_bracket_relative_width_max=max((row["rejected_launch_slope"]-row["accepted_launch_slope"])/row["rejected_launch_slope"] for row in boundary["sectors"]),
                azimuth_refinement="NOT_MEASURED_SINGLE_AZIMUTH_COUNT",
                caveat="Same launch rays; this checks transport, not a newly solved fine-step mechanical boundary")
            if args.azimuths % 2 == 0 and args.azimuths >= 16:
                nested = coarse_angle[::2]
                convergence["azimuth_nested_envelope_check"] = dict(coarse_count=len(nested), fine_count=len(coarse_angle),
                    minimum_change_mrad=float(coarse_angle.min()-nested.min()),
                    maximum_change_mrad=float(coarse_angle.max()-nested.max()),
                    scope="Nested finite azimuth envelope comparison; does not prove continuous angular support")
        else:
            convergence = dict(status="BOUNDARY_SURVIVAL_CHANGED_WITH_STEP", blocked_key=fine["blocked_key"])

    # Nominal-energy source-support witnesses: centre and eight spatial rim
    # points crossed with all configured angular-rim azimuths. These carry no
    # current and do not replace the original mixed-energy particle population.
    from temsim.optics.electron_gun.tip_curvature import support_radius_nm
    phi = np.arange(args.azimuths)*2*np.pi/args.azimuths
    spatial_phi = np.arange(8)*2*np.pi/8
    support_r = support_radius_nm(state.electron_gun.emitter)*1e-9
    spatial = np.vstack((np.zeros((1,2)), support_r*np.c_[np.cos(spatial_phi), np.sin(spatial_phi)]))
    source_cutoff = state.electron_gun.emitter.angular_cutoff_mrad*.001
    directions = source_cutoff*np.c_[np.cos(phi), np.sin(phi)]
    launch = np.column_stack((np.repeat(spatial, len(phi), axis=0), np.tile(directions,(len(spatial),1))))
    supported = observe(launch, "finite_source_support_witnesses")
    support_live = supported["alive"][-1]
    support_angles = direction_angles(supported["xy_slopes"][-1, support_live, 2:], chief["xy_slopes"][-1,0,2:])*1e3 if support_live.any() else np.array([])
    filling = []
    for part in apertures:
        from temsim.physics.aperture_clipping import posed_aperture_registration
        registration = posed_aperture_registration(state, part)
        if registration is not None and (np.any(registration.origin_array_m)
                or not np.array_equal(registration.rotation_array, np.eye(3))):
            filling.append(dict(key=part.key, status="NOT_MEASURED_POSED_APERTURE_REQUIRES_LOCAL_PLANE_SAMPLING"))
            continue
        index = np.flatnonzero(np.isclose(cp.z_mm,part.z_mm,rtol=0,atol=1e-9))
        if not index.size:
            filling.append(dict(key=part.key, status="NOT_MEASURED_NO_EXACT_PARTICLE_PLANE")); continue
        i = int(index[0]); live = arrays["alive"][i] & (arrays["weight"]>0)
        r = np.hypot(cp.x_m[i,live]-part.offset_x_mm*.001, cp.y_m[i,live]-part.offset_y_mm*.001)
        weight = arrays["weight"][live]
        order = np.argsort(r)
        r95 = r[order[min(np.searchsorted(np.cumsum(weight[order]), .95*weight.sum()), len(r)-1)]] if len(r) else None
        diagnostic_i = int(np.searchsorted(planes, part.z_mm))
        valid = supported["alive"][diagnostic_i]
        xy = supported["xy_slopes"][diagnostic_i,valid,:2] - np.array([part.offset_x_mm,part.offset_y_mm])*.001
        support_fill = np.linalg.norm(xy, axis=1)/(part.radius_mm*.001)
        filling.append(dict(key=part.key, name=part.name, z_mm=part.z_mm, aperture_radius_um=part.radius_mm*1000,
            plate_bounds_mm=[part.z_mm-part.plate_thickness_mm/2, part.z_mm+part.plate_thickness_mm/2],
            metric_plane_role="Surviving population at plate midplane; upstream plate-face losses are not reconstructed",
            actual_sampled_max_radius_fraction=float(r.max()/(part.radius_mm*.001)) if len(r) else None,
            actual_current95_radius_fraction=float(r95/(part.radius_mm*.001)) if r95 is not None else None,
            nominal_energy_source_support_sampled_max_fraction=float(support_fill.max()) if len(support_fill) else None,
            transmitted_current_a=effective_source_current_a(state)*float(weight.sum()/arrays["weight"].sum()),
            definition="Distance from actual aperture centre; finite current sample and source-support witnesses, not a filled-area estimate"))
    radial_envelope = []
    for trace in executed:
        good = trace["alive"][-1]
        if good.any():
            radial_envelope.extend(direction_angles(trace["xy_slopes"][-1,good,2:],chief["xy_slopes"][-1,0,2:]).tolist())
    if capture_instrument_snapshot(state).digest != snapshot.digest or state.to_dict() != inputs:
        raise ValueError("Acceptance measurement changed physical inputs")
    report = dict(schema="joint-incident-optical-acceptance-v1", utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=perf_counter()-started, input_file=str(args.inputs.resolve()), input_sha256=file_hash(args.inputs),
        particle_archive=str(args.particles.resolve()), particle_archive_sha256=file_hash(args.particles),
        historical_particle_solver_identity="See archived report; reused observations were not rerun by the new diagnostic solver",
        historical_particle_snapshot_digest=archive_report.get("snapshot_digest"),
        archived_physical_inputs_match=True,
        archived_source_arrays_match_receipt=True,
        snapshot_digest=snapshot.digest, diagnostic_solver_identity=implementation, physical_inputs_unchanged=True,
        source=dict(emission_mean_energy_ev=state.electron_gun.emitter.emission_energy_ev,
            energy_spread_fwhm_ev=state.electron_gun.emitter.energy_spread_fwhm_ev,
            source_cutoff_slope_mrad=source_cutoff*1e3, actual_source_current_a=effective_source_current_a(state)),
        pupil_reference=dict(source_xy_m=[0.,0.], energy_ev=state.electron_gun.emitter.emission_energy_ev,
            role="Reference-source mechanical pupil; source-unreachable rim rays carry zero current"),
        boundary=boundary, physical_pupil=pupil, crossover_chain=chain, aperture_filling=filling,
        archived_gun_observation=archive_report.get("gun"),
        specimen_focus=dict(radial_covariance_waist_offset_nm=chain["specimen"]["actual"]["statistics"]["waist_offset_m"]*1e9,
            provisional_absolute_tolerance_nm=1.,
            interpretation="Local phase-space drift estimate inside objective field; not a newly executed two-sided waist bracket"),
        source_support=dict(status="SAMPLED_FINITE_SOURCE_WITNESSES", rays=len(launch), transmitted=int(support_live.sum()),
            spatial_positions_m=spatial.tolist(), angular_rim_slope=source_cutoff,
            sample_angle_min_mrad=float(support_angles.min()) if len(support_angles) else None,
            sample_angle_max_mrad=float(support_angles.max()) if len(support_angles) else None,
            all_supported_witnesses_transmitted=bool(supported["alive"][-1].all()),
            scope="Nominal energy and sampled support rim; not a rigorous mixed-energy four-dimensional envelope"),
        interior_direction_scan_max_mrad=1e3*max(radial_envelope) if radial_envelope else None,
        edge_interpretation="Mapped physical rim; nonlinear direction folding can make it differ from the full angular projection envelope",
        numerical_checks=convergence, timings=log,
        reused_cache_original_timings=original_timings,
        qualification=dict(overall="NOT_QUALIFIED", target="30 mrad physical pupil edge; alpha95 reported separately",
            current_limit=None, relay_conjugacy="NOT_QUALIFIED", scan="NOT_MEASURED_STATIC_RUN",
            particle_count_convergence="PRIOR_EVIDENCE_NOT_CONVERGED", gun_internal_waists="NOT_MEASURED_EXACT_GUN_PLANES_REQUIRED",
            post_specimen="OUTSIDE_INCIDENT_ACCEPTANCE_SCOPE"))
    if rim is not None:
        report["source_support"]["mechanical_rim_within_source_cutoff"] = rim["source_direction_supported"].tolist()
    save_json(destination, report)
    save_json(args.output/"inputs.json", inputs)
    print(str(destination.resolve()), flush=True)


if __name__ == "__main__":
    main()
