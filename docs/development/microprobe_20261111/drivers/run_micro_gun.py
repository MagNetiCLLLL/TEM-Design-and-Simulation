"""Fresh current-source production gun capture for the Microprobe study."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
import tomllib

import numpy as np
import tomli_w


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PROFILE = ROOT / "docs/development/fege_focus_20261110/normal-nanoprobe-si110.toml"
sys.path.insert(0, str(ROOT / "scripts"))


def historical_catalog():
    from temsim.paths import INSTRUMENT_CONFIG_ROOT
    from temsim.shared_tip import copy_catalog_tree

    target = HERE / "historical-catalog" / "configs" / "instruments"
    path = target / "gun/FEG.toml"
    if not path.exists():
        copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, target)
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
        old_text = subprocess.check_output(["git", "show", "9554a3a:configs/instruments/gun/FEG.toml"],
                                           cwd=ROOT, encoding="utf-8")
        old = next(part for part in tomllib.loads(old_text)["parts"] if part["key"] == "feg_extractor")
        extractor = next(part for part in doc["parts"] if part["key"] == "feg_extractor")
        for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "length_mm",
                      "vacuum_inner_diameter_mm", "optical_reference_local_z_mm", "outer_diameter_mm",
                      "bore_diameter_mm"):
            extractor[field] = old[field]
        for field in ("counterbore_depth_mm", "counterbore_diameter_mm", "extractor_geometry_source", "parameter_metadata"):
            extractor.pop(field, None)
        path.write_text(tomli_w.dumps(doc), encoding="utf-8")
    return target


def prepare_state(arm, count):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.detector.recording_system import ensure_recording_system
    from temsim.optics.column import default_state
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.profile_io import apply_profile_values, read_profile
    from temsim.vacuum import bind_gun_environment

    if arm != "new":
        raise ValueError("The Microprobe study uses the installed new extractor")
    selection, values = read_profile(PROFILE)
    fixture = ROOT / values["sample"]["cif_path"]
    expected_cif = "944c5c81df5d813d051f96102eb9be37fac4b7647c2a9d1492d6333122be42c7"
    if sha256(fixture.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected_cif:
        raise ValueError("The local silicon fixture differs from the validated source profile")
    values["sample"]["cif_path"] = str(fixture)
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.normalise_selection(selection))
    apply_profile_values(state, values)
    state.simulation_mode = "unified"
    state.acceleration_backend = "CPU"
    state.acceleration_enabled = False
    state.electron_gun.emitter.ray_count = count
    apply_physical_layout_to_state(state, preserve_operating_parameters=True)
    magnitude = float(state.mini_condenser.percent)
    state.mini_condenser.set_probe_mode("micro_probe")
    state.illumination_mode = "TEM"
    if state.mini_condenser.percent != magnitude or state.mini_condenser.polarity != 1:
        raise ValueError("Microprobe mode must preserve the Mini Condenser magnitude and set positive polarity")
    if any(getattr(component, key, False) for component in
           (state.ac_deflector, state.descan_deflector) for key in ("scan_enabled", "wobble_enabled")):
        raise ValueError("The captured Microprobe source must retain stationary optical controls")
    if any(getattr(state.sample, key, False) for key in
           ("wave_enabled", "stem_wave_enabled", "stem_fourdstem_enabled")):
        raise ValueError("Only the classical incident source is admitted")
    ensure_recording_system(state)
    bind_gun_environment(state)
    with instrument_gun_field_context(state):
        pass
    return state, selection


def field_profile(field):
    z_mm = np.unique(np.r_[0., np.geomspace(.000001, .1, 220), np.linspace(.1, 20., 801),
                           .3, .75, 1.3, 6., 8., 10., 18., 100., 200., 450.])
    points = np.c_[np.zeros((len(z_mm), 2)), z_mm * 1e-3]
    phi, electric = field.interpolate(points)
    points[:, 0] = 2e-6
    _, off_axis = field.interpolate(points)
    return dict(z_mm=z_mm, phi_rise_v=phi, ez_v_per_m=electric[:, 2],
                ex_at_x2um_v_per_m=off_axis[:, 0])


def main(argv=None):
    from measure_probe_waist import _array_hash
    from temsim.cpu_resources import numerical_job
    from temsim.extractor_geometry import ExtractorContour
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.electron_gun.tracing import trace_feg_with_observations
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.physics.particle_sections import gun_dependency_signature
    from temsim.physics.simulation import _validate_gun_flight_times
    from temsim.physics.transport_progress import transport_progress
    from temsim.profile_io import save_profile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("new",), default="new")
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--budget-s", type=float, default=750.)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.count < 1000 or args.budget_s <= 0:
        parser.error("Use at least 1000 original source particles and a positive budget")
    out = HERE / f"{args.arm}-n{args.count}"
    out.mkdir(exist_ok=False)
    started = time.perf_counter()
    deadline = started + args.budget_s
    cancelled = lambda: time.perf_counter() > deadline or (HERE / "CANCEL").exists()

    def write(name, record):
        path = out / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def log(message):
        print(f"{args.arm} {time.perf_counter() - started:.1f}s {message}", flush=True)

    def pop_summary(x, y, tx, ty, weights, keep, emitted_current):
        current = weights[keep]
        result = dict(count=int(keep.sum()), weight_sum=float(current.sum()),
                      current_a=float(emitted_current * current.sum()),
                      effective_count=float(current.sum() ** 2 / np.dot(current, current)) if len(current) else 0.)
        if np.any(keep):
            s = transverse_beam_statistics(x, y, tx, ty, alive=keep, weights=weights)
            result.update(radial_rms_m=s.radius_rms_m, d95_m=2 * s.radius_95_m,
                          alpha_rms_rad=s.convergence_rms_rad, alpha95_rad=s.convergence_95_rad,
                          centroid_m=[s.mean_x_m, s.mean_y_m])
        return result

    with numerical_job(requested=args.threads, cancelled=cancelled):
        log("Preparing immutable inputs")
        state, selection = prepare_state(args.arm, args.count)
        gun = state.electron_gun
        contour = ExtractorContour.from_component(gun.extractor)
        expected = (0.3, 1.3, .5, .45, 4.) if args.arm == "new" else (6., 10., 4., 0., 0.)
        actual = (contour.start_mm, contour.end_mm, contour.bore_diameter_mm,
                  contour.counterbore_depth_mm, contour.counterbore_diameter_mm)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
        signature = gun_dependency_signature(state)
        if not signature:
            raise ValueError("Missing production gun dependency signature")
        snapshot = capture_instrument_snapshot(state)
        write("inputs.json", snapshot.to_dict())
        save_profile(out / "input.toml", state, selection)
        emitted = gun.emit(args.count)
        emitted_weights = np.asarray(emitted.weight)
        if np.count_nonzero(emitted_weights > 0) != args.count or len(np.unique(emitted.ray_id)) != args.count:
            raise ValueError("Actual emission must contain the declared distinct positive-current original particles")
        launch = {key: np.asarray(getattr(emitted, key)) for key in
                  ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "ray_id", "energy_offset_ev")}
        for key in ("surface_position_m", "surface_direction", "surface_normal", "surface_energy_ev"):
            value = getattr(emitted, key, None)
            if value is not None:
                launch[key] = np.asarray(value)
        launch_sha = {key: _array_hash(value) for key, value in launch.items()}
        np.savez_compressed(out / "launch.npz", **launch)
        write("prepared.json", dict(status="PREPARED", gun_dependency_signature=signature,
             input_digest=snapshot.digest, input_profile=str(PROFILE), extractor=asdict(contour),
             launch_sha256=launch_sha, count=args.count))
        with instrument_gun_field_context(state):
            log("Solving or reading the production gun field")
            field = gun.base_electric_field
            write("electrostatic_request.json", field.request)
            np.savez_compressed(out / "axis_field.npz", **field_profile(field))
            log("Tracing the production gun and exact first-crossing planes")
            checkpoints = np.unique(np.r_[.75, 1.3, 10., 18., 100., float(gun.exit_plane_z_mm),
                                          float(gun.dpa_aperture.z_mm), float(gun.c1_aperture.z_mm)])
            last = [0.]

            def progress(message):
                if time.perf_counter() - last[0] >= 15.:
                    last[0] = time.perf_counter()
                    log(str(message))

            with transport_progress(progress):
                result, observations = trace_feg_with_observations(gun, checkpoints,
                                                                   count=args.count, cancelled=cancelled)
        log("Validating the native gun result")
        if capture_instrument_snapshot(state).digest != snapshot.digest:
            raise ValueError("Gun execution mutated its captured physical or numerical inputs")
        if not _validate_gun_flight_times(result):
            raise ValueError("Invalid first-arrival clock")
        exit_bundle = result.exit_bundle
        np.testing.assert_array_equal(exit_bundle.ray_id, emitted.ray_id)
        np.testing.assert_array_equal(exit_bundle.weight, emitted.weight)
        np.testing.assert_array_equal(result.emission_reference["ray_id"], emitted.ray_id)
        weights = np.asarray(exit_bundle.weight)
        alive = np.asarray(exit_bundle.alive)
        blocked_key = np.asarray(result.blocked_key, dtype=str)
        if np.any(~alive & (blocked_key == "")):
            raise ValueError("Unlabelled stopped source particle")
        arrays = {"gun_" + name: np.asarray(getattr(result, name)) for name in
                  ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "blocked_z_mm")}
        arrays["gun_blocked_key"] = blocked_key
        arrays.update({"gun_exit_" + name: np.asarray(getattr(exit_bundle, name)) for name in
                       ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "energy_offset_ev", "weight", "ray_id", "alive")})
        arrays.update({"emission_" + key: np.asarray(value) for key, value in result.emission_reference.items()})
        arrays.update(source_id=np.asarray(exit_bundle.ray_id), weights=weights,
                      source_energy_offset_ev=np.asarray(exit_bundle.energy_offset_ev))
        np.savez_compressed(out / "source.npz", **arrays)
        exact = {key: value for key, value in vars(observations).items() if isinstance(value, np.ndarray)}
        np.savez_compressed(out / "exact_observations.npz", **exact)
        current = float(result.emitted_current_a)
        rows = []
        for i, z in enumerate(exact["z_mm"]):
            keep = exact["alive"][i] & (weights > 0)
            row = pop_summary(*(exact[key][i] for key in ("x_m", "y_m", "tx_rad", "ty_rad")), weights, keep, current)
            row["z_mm"] = float(z)
            row["crossing_count"] = int(exact["reached"][i].sum())
            if np.any(keep):
                row["mean_kinetic_energy_ev"] = float(np.average(exact["kinetic_energy_ev"][i, keep], weights=weights[keep]))
            rows.append(row)
        report = dict(status="COMPLETE", scope="Classical physical-gun source for a Microprobe configuration at the existing Mini Condenser magnitude",
            arm=args.arm, elapsed_seconds=time.perf_counter() - started,
            input_digest=snapshot.digest, gun_dependency_signature=signature,
            implementation=snapshot.to_dict()["implementation"],
            extractor=asdict(contour), geometrical_acceptance=contour.straight_ray_acceptance(),
            reference_current_a=current, emitted_positive_current_points=args.count,
            gun_exit=pop_summary(exit_bundle.x_m, exit_bundle.y_m, exit_bundle.tx_rad,
                                 exit_bundle.ty_rad, weights, alive & (weights > 0), current),
            losses={key: dict(count=int(np.count_nonzero((blocked_key == key) & ~alive)),
                              current_a=float(current * weights[(blocked_key == key) & ~alive].sum()))
                    for key in sorted(set(blocked_key[~alive]))},
            physical_interception_balance_a=float(current * (weights.sum() - weights[alive].sum() - weights[~alive].sum())),
            planes=rows, launch_sha256=launch_sha,
            source_sha256={key: _array_hash(value) for key, value in arrays.items()},
            electrostatic_model=result.electrostatic_model_report)
        write("receipt.json", report)
        log(f"Complete. Exit {report['gun_exit']['count']}/{args.count}; current {report['gun_exit']['current_a']:.8g} A")


if __name__ == "__main__":
    main()
