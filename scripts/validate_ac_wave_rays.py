"""Execute matched physical-Tip rays for the bounded AC wave-scan protocol.

The production particle plan freezes drives at the requested epoch. Its stored
flight times do not turn it into a per-electron dynamic-field integrator. This
script explicitly bounds the difference from coil-arrival sampling using the
actual retained flight times and finite-coil local response. A failed timing
gate prevents admission as a same-state approximation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from validate_probe_relay import assert_cuda, save_report, source_identity


def _index(points, z):
    matches = np.flatnonzero(np.isclose(points.z_mm, float(z), rtol=0., atol=1e-10))
    if len(matches) != 1:
        raise ValueError(f"Missing exact ray checkpoint at Z={z} mm")
    return int(matches[0])


def _metrics(points, index, weights, alive, reference_current_a):
    from temsim.physics.beam_statistics import transverse_beam_statistics
    stats = transverse_beam_statistics(points.x_m[index], points.y_m[index],
        points.tx_rad[index], points.ty_rad[index], weights=weights, alive=alive)
    selected = alive & (weights > 0)
    w = weights[selected]
    total = float(w.sum())
    xy = np.array((points.x_m[index, selected], points.y_m[index, selected]))
    centre = np.array((stats.mean_x_m, stats.mean_y_m))
    covariance = ((xy-centre[:, None])*w) @ (xy-centre[:, None]).T / total
    energy = np.asarray(points.kinetic_energy_ev[index])[selected]
    mean_energy = float(w @ energy / total)
    return dict(centroid_xy_m=centre.tolist(), radial_rms_m=stats.radius_rms_m,
        rms_xy_m=np.sqrt(np.maximum(np.diag(covariance), 0.)).tolist(),
        total_probability=total, source_fraction=total,
        current_a=reference_current_a*total, reference_current_a=reference_current_a,
        mean_energy_ev=mean_energy,
        rms_energy_ev=float(np.sqrt(w @ ((energy-mean_energy)**2) / total)),
        diameter95_m=2*stats.radius_95_m,
        mean_mechanical_direction_rad=[stats.mean_tx_rad, stats.mean_ty_rad],
        covariance_waist_offset_m=stats.waist_offset_m,
        population_count=stats.surviving_rays,
        coordinate_frame="laboratory XY; +Z downstream", statistics=asdict(stats))


def _survival(state, output, alive, stops, keys):
    from temsim.physics.aperture_clipping import clip_segment
    from temsim.physics.column_wall import clip_column_wall
    from temsim.physics.recording_clipping import clip_recording_planes
    z, x, _, y, _, _, points = output
    alive, stops, keys = clip_segment(state, z, x, y, alive.copy(), stops.copy(), list(keys))
    alive, stops, keys = clip_recording_planes(state, z, x, y, alive, stops, keys)
    alive, stops, keys = clip_column_wall(state, z, x, y, alive, stops, keys)
    return (alive, stops, keys,
        np.isnan(stops)[None, :] | (stops[None, :] > points.z_mm[:, None]))


def _archive_points(archive, prefix, points, masks):
    for name in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        archive[prefix+"_"+name] = np.asarray(getattr(points, name))
    archive[prefix+"_alive"] = masks


def _timing_report(state, points, masks, epoch, response, plane_rows, limit_nm):
    """Retain same-ray timing correlations in an explicit first-order estimate."""
    ac = state.ac_deflector
    nominal = ac.coil_kicks_mrad(*ac.instantaneous_kick_mrad(epoch))
    frame = float(ac.scan_frame_period_s)
    nlines = int(ac.scan_lines)
    line0 = int((epoch/frame % 1.)*nlines)
    coil_rows, maxima, per_ray_differences = [], [], []
    for ci, z in enumerate((ac.upper_z_mm, ac.lower_z_mm)):
        index = _index(points, z)
        times = np.asarray(points.flight_time_s[index])[masks[index]]
        if not times.size or not np.isfinite(times).all():
            raise ValueError("Arrival-time comparison needs all surviving ray clocks")
        current = np.array([ac.coil_kicks_mrad(*ac.instantaneous_kick_mrad(float(t)))[ci]
                            for t in times], dtype=float)
        differences_rad = (current-np.asarray(nominal[ci]))*1e-3
        maxima.append(np.max(abs(differences_rad), axis=0))
        aligned = np.full((masks.shape[1], 2), np.nan)
        aligned[masks[index]] = differences_rad
        per_ray_differences.append(aligned)
        lines = ((times/frame % 1.)*nlines).astype(int)
        coil_rows.append(dict(z_mm=float(z), minimum_arrival_time_s=float(times.min()),
            maximum_arrival_time_s=float(times.max()),
            minimum_elapsed_from_tip_s=float(times.min()-epoch),
            maximum_elapsed_from_tip_s=float(times.max()-epoch),
            maximum_command_difference_rad=maxima[-1].tolist(),
            raster_line_unchanged=bool(np.all(lines == line0))))
    bounds = []
    for plane in plane_rows:
        upper, lower, _ = response.at(float(plane["z_mm"]))
        # Retain the deliberately conservative independent-coil estimate too.
        extent = abs(upper[:2]) @ maxima[0] + abs(lower[:2]) @ maxima[1]
        selected = masks[_index(points, plane["z_mm"])]
        delta = (per_ray_differences[0][selected] @ upper[:2].T
                 + per_ray_differences[1][selected] @ lower[:2].T)
        if not len(delta) or not np.isfinite(delta).all():
            raise ValueError("Every observed ray must retain both coil-arrival clocks")
        correlated_maximum = float(np.max(np.linalg.norm(delta, axis=1))*1e9)
        bounds.append(dict(plane=plane["label"], z_mm=plane["z_mm"],
            independent_xy_bound_nm=(extent*1e9).tolist(),
            independent_radial_bound_nm=float(np.linalg.norm(extent)*1e9),
            correlated_local_maximum_nm=correlated_maximum))
    passed = all(c["raster_line_unchanged"] for c in coil_rows) and all(
        b["correlated_local_maximum_nm"] <= limit_nm for b in bounds)
    return dict(model="Production particle epoch-frozen fields; wave uses per-mode coil-arrival fields",
        bound_method="Actual same-ray upper/lower flight times combined through finite-coil local response; maximum over the executed population. Independent triangle estimate retained. First-order estimate, not a nonlinear dynamical proof",
        limit_nm=limit_nm, coils=coil_rows, plane_bounds=bounds, passed=passed)


def validate(directory, report, *, timing_limit_nm=.01):
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.optics.electron_gun.source import trace_source_to_exit
    from temsim.optics.electron_gun.tip_coherence import DRIVEN_GAUSSIAN_SCHELL
    from temsim.physics.backend_execution import backend_receipts
    from temsim.physics.core import propagate
    from temsim.physics.finite_scan_response import finite_scan_responses
    from temsim.physics.instrument_magnetic import active_column_events, column_dipole_fields
    from temsim.physics.scan_preparation import prepare_scan_drives
    from temsim.specimen.scene import SpecimenScene

    snapshot = InstrumentSnapshot.from_dict(json.loads((directory/"inputs.json").read_text(encoding="utf-8")))
    protocol = json.loads((directory/"protocol.json").read_text(encoding="utf-8"))
    state = snapshot.restore()
    if protocol["input_identity"] != snapshot.digest:
        raise ValueError("Protocol and captured instrument identity differ")
    emitter = state.electron_gun.emitter
    if emitter.ray_count != 3000 or getattr(emitter.coherence, "boundary_model", None) != DRIVEN_GAUSSIAN_SCHELL:
        raise ValueError("This protocol requires the captured 3000-ray Driven Tip")
    if state.simulation_mode != "custom" or not SpecimenScene.from_state(state).is_vacuum:
        raise ValueError("This comparison requires custom optics with a vacuum specimen")
    if state.acceleration_backend != "Require GPU" or not state.acceleration_enabled:
        raise ValueError("Capture Require GPU for this physical ray validation")
    if state.vacuum_map.enabled:
        raise ValueError("Residual-medium scattering is outside this matched optics check")
    planes = protocol["planes"]
    step = float(protocol["column_step_mm"])
    z_values = np.array([float(p["z_mm"]) for p in planes])
    if not len(z_values) or np.any(np.diff(z_values) <= 0):
        raise ValueError("Protocol planes must be finite and strictly ordered")
    prepare_scan_drives(state, snapshot.digest, observation_stop_z_mm=float(z_values[-1]))
    dynamic = [c for c in column_dipole_fields(state) if c.dynamic]
    if len(dynamic) != 2 or any(state.ac_deflector.key not in c.drive_keys for c in dynamic):
        raise ValueError("This prefix-reuse protocol requires AC as the only dynamic pair")
    prefix_z = min(c.field_support_mm[0] for c in dynamic)-.001
    gun_exit = float(state.electron_gun.exit_plane_z_mm)
    if not gun_exit < prefix_z < z_values[0]:
        raise ValueError("The physical pre-AC prefix is outside this observation interval")
    report.update(input_identity=snapshot.digest, protocol=protocol,
        source_model="Driven Tip: same captured physical inputs as wave; particles use its geometric-optics representation",
        timing_model="Production fields frozen at the requested tip epoch; saved clocks only integrate transit time",
        timing_qualification="Pending actual flight-time bound; not exact per-ray dynamic drive execution",
        source_count=3000, prefix_z_mm=prefix_z, backend_receipts=[], results=[], positions=[],
        coordinate_archive="rays_coordinates.npz", status="running")
    archive = {}
    print("Tracing the captured Driven Tip to gun exit", flush=True)
    rows = []
    with backend_receipts(rows):
        gun = trace_source_to_exit(state)
    report["backend_receipts"].extend(rows)
    bundle = gun.exit_bundle
    alive0 = np.asarray(bundle.alive, bool)
    weights = np.asarray(bundle.weight, float)
    total_weight = float(weights.sum())
    if not np.isfinite(total_weight) or abs(total_weight-1.) > 1e-8:
        raise ValueError("Tip ray weights must remain relative to one emitted electron")
    gun_times = np.asarray(bundle.flight_time_s, float)
    if gun_times.shape != weights.shape or not np.isfinite(gun_times[alive0]).all():
        raise ValueError("The executed gun did not provide finite surviving-ray clocks")
    report["source_exit_identity"] = source_identity(bundle)
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "alive", "energy_offset_ev", "ray_id", "flight_time_s"):
        archive["gun_exit_"+name] = np.asarray(getattr(bundle, name))
    reference_current = float(emitter.emission_current_na)*1e-9
    report["reference_current_a"] = reference_current
    print("Executing one common pre-AC population on CUDA", flush=True)
    rows = []
    with backend_receipts(rows):
        initial = propagate(state, gun_exit, prefix_z,
            bundle.x_m, bundle.tx_rad, bundle.y_m, bundle.ty_rad,
            events=active_column_events(state), energy_offset_ev=bundle.energy_offset_ev,
            include_spherical_aberration=True, include_hexapole=True,
            checkpoint_z_mm=(prefix_z,), return_checkpoints=True, maximum_step_mm=step,
            initial_time_s=gun_times, return_flight_times=True,
            defer_nonfinite_until_clipping=True)
    assert_cuda(rows)
    report["backend_receipts"].extend(rows)
    alive, stops, keys, masks = _survival(state, initial, alive0, gun.blocked_z_mm, gun.blocked_key)
    prefix = initial[-1]
    masks &= alive0[None, :] & (weights > 0)[None, :]
    _archive_points(archive, "prefix", prefix, masks)
    pi = _index(prefix, prefix_z)
    prefix_alive = masks[pi]
    report["prefix_surviving_rays"] = int(prefix_alive.sum())
    prefix_coordinates = [np.asarray(getattr(prefix, n)[pi]) for n in ("x_m", "tx_rad", "y_m", "ty_rad")]
    prefix_times = np.asarray(prefix.flight_time_s[pi])
    prefix_energy = np.asarray(prefix.kinetic_energy_ev[pi])
    print("Measuring finite-coil response for the explicit timing-error estimate", flush=True)
    response = finite_scan_responses(state, state.ac_deflector, z_values, maximum_step_mm=step)
    report["timing_response"] = dict(z_mm=response.z_mm.tolist(), upper=response.upper.tolist(), lower=response.lower.tolist())
    observation_z = np.unique(np.r_[z_values, state.ac_deflector.upper_z_mm, state.ac_deflector.lower_z_mm])
    for n, position in enumerate(protocol["positions"]):
        epoch = float(position["tip_time_s"])
        state.simulation_time_s = epoch
        actual_factors = state.ac_deflector.scan_factors(epoch)
        if not np.allclose(actual_factors, position["factors"], rtol=0., atol=1e-12):
            raise ValueError("Protocol raster factors disagree with the captured AC controls")
        print(f"CUDA particle raster {position['label']} at tip epoch {epoch:.12g} s", flush=True)
        rows = []
        with backend_receipts(rows):
            output = propagate(state, prefix_z, float(z_values[-1]), *prefix_coordinates,
                events=active_column_events(state), energy_offset_ev=bundle.energy_offset_ev,
                include_spherical_aberration=True, include_hexapole=True,
                save_z_mm=observation_z, checkpoint_z_mm=observation_z,
                return_checkpoints=True, maximum_step_mm=step,
                initial_time_s=prefix_times+epoch, return_flight_times=True,
                initial_kinetic_energy_ev=prefix_energy, defer_nonfinite_until_clipping=True)
        assert_cuda(rows)
        report["backend_receipts"].extend(dict(r, position=position["label"]) for r in rows)
        _, _, _, alive_at = _survival(state, output, prefix_alive, stops, keys)
        points = output[-1]
        alive_at &= prefix_alive[None, :] & (weights > 0)[None, :]
        _archive_points(archive, f"position_{n}", points, alive_at)
        timing = _timing_report(state, points, alive_at, epoch, response, planes, timing_limit_nm)
        report["positions"].append(dict(position=position["label"], tip_time_s=epoch,
            factors=list(actual_factors), timing=timing))
        for plane in planes:
            index = _index(points, plane["z_mm"])
            report["results"].append(dict(position=position["label"], plane=plane["label"],
                z_mm=float(plane["z_mm"]), tip_time_s=epoch,
                metrics=_metrics(points, index, weights, alive_at[index], reference_current),
                timing_admitted=timing["passed"]))
        save_report(directory/"rays.json", report)
    archive["position_labels"] = np.asarray([p["label"] for p in protocol["positions"]])
    archive["plane_labels"] = np.asarray([p["label"] for p in planes])
    archive["plane_z_mm"] = z_values
    np.savez_compressed(directory/"rays_coordinates.npz", **archive)
    report["timing_admitted"] = all(p["timing"]["passed"] for p in report["positions"])
    report["timing_qualification"] = ("Admitted same-state approximation under the declared local-response timing bound"
        if report["timing_admitted"] else "NOT admitted: arrival-time field mismatch exceeds the declared bound")
    report["status"] = "complete" if report["timing_admitted"] else "timing_gate_failed"
    snapshot.verify_current_inputs(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("outputs/ac_wave_scan_20261010"))
    parser.add_argument("--timing-limit-nm", type=float, default=.01)
    args = parser.parse_args()
    if not np.isfinite(args.timing_limit_nm) or args.timing_limit_nm <= 0:
        parser.error("The timing-error limit must be finite and positive")
    started = perf_counter()
    report = {}
    try:
        validate(args.directory, report, timing_limit_nm=args.timing_limit_nm)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["elapsed_s"] = perf_counter()-started
        save_report(args.directory/"rays.json", report)
    print(json.dumps(dict(status=report["status"], elapsed_s=report["elapsed_s"],
                          timing_admitted=report["timing_admitted"])), flush=True)
    return 0 if report["timing_admitted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
