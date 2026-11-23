"""Propagate matched original source particles and retain the gun-exit handoff."""

from dataclasses import fields
import argparse
import json

import numpy as np

from gun_focus import (CASE, HERE, INPUTS, ROOT, EmissionBundle, ExactGunPlaneObserver,
                       _trace_feg_to_exit, gun_step_workers, prepare, set_voltage, write_json)
from temsim.physics.beam_statistics import transverse_beam_statistics


def transport(kv, mesh=None, check=False):
    gun, record = prepare()
    if mesh:
        gun._gun_field_cells_per_bore = mesh
    if check:
        gun.trace_step_mm *= .5
    identity = set_voltage(gun, kv)
    magnet_id = record["bore_field"]["sha256"]
    source_id = record["particle_source"]["sha256"]
    with np.load(INPUTS / record["particle_source"]["file"]) as parent:
        original_ids = parent["ids"]
        selected_ids = parent["selected_ids"]
        emission = gun.emit(len(original_ids))
        np.testing.assert_array_equal(original_ids, emission.ray_id)
        np.testing.assert_array_equal(parent["weight"], emission.weight)
        np.testing.assert_array_equal(parent["launch"], np.column_stack(
            [getattr(emission, k) for k in ("x_m", "y_m", "tx_rad", "ty_rad")]))
        assert float(parent["emitted_A"]) == gun.emitted_current_a
    assert len(original_ids) == len(np.unique(original_ids)) == 16384
    assert np.all(emission.weight > 0) and np.sum(emission.weight) == 1.
    assert not hasattr(emission, "surface_position_m") and gun.emitter.curvature_nm_inv == 0
    if check:
        paired_check(gun, emission, kv, identity)
        return
    scratch = ROOT / "outputs/gun-lens-24mm" / (identity + "-" + magnet_id[:12] + "-" + source_id[:12])
    scratch.mkdir(parents=True, exist_ok=True)
    parts = []
    for start in range(0, len(original_ids), 4096):
        path = scratch / f"original-ids-{start}.npz"
        s = slice(start, start+4096)
        expected = original_ids[s]
        if not path.exists():
            print(f"Propagating {len(expected)} originals starting with ID {expected[0]}", flush=True)
            bundle = EmissionBundle(**{f.name: getattr(emission, f.name)[s] for f in fields(EmissionBundle)})
            with gun_step_workers():
                result = _trace_feg_to_exit(gun, emitted_bundle=bundle)
            out = result.exit_bundle
            np.testing.assert_array_equal(out.ray_id, expected)
            np.testing.assert_array_equal(out.weight, bundle.weight)
            assert result.electrostatic_model_report["maximum_exit_energy_error_ev"] < 1e-3
            near = {arrival.key: arrival for arrival in result.plane_arrivals}
            dpa = near[gun.dpa_aperture.key]
            c1 = near[gun.c1_aperture.key]
            assert np.array_equal(out.alive, near[f"{gun.c1_aperture.key}:exit"].transmitted)
            selected = np.isin(out.ray_id, selected_ids)
            phase = np.column_stack((out.x_m, out.y_m, out.tx_rad, out.ty_rad, out.flight_time_s))
            maximum_r = float(np.nanmax(np.hypot(result.equal_time_history.x_m,
                                                  result.equal_time_history.y_m)))
            temp = path.with_suffix(".partial.npz")
            np.savez_compressed(temp, ids=out.ray_id, weight=out.weight, alive=out.alive, phase=phase,
                energy_offset_ev=out.energy_offset_ev, dpa_xy_m=np.column_stack([dpa.x_m, dpa.y_m]),
                dpa_passed=dpa.transmitted, c1_passed=c1.transmitted,
                loss_z_mm=result.blocked_z_mm, loss_key=np.array(result.blocked_key, dtype=str),
                gun_z_mm=result.z_mm, selected_ids=out.ray_id[selected],
                gun_paths_m=np.stack([result.x_m[:, selected], result.y_m[:, selected]], axis=2),
                maximum_radius_m=maximum_r, max_energy_error_eV=result.electrostatic_model_report[
                    "maximum_exit_energy_error_ev"], source_id=source_id, magnet_id=magnet_id, electric_id=identity)
            temp.replace(path)
            del result, out, near, dpa, c1
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["ids"], expected)
            np.testing.assert_array_equal(saved["weight"], emission.weight[s])
            assert (str(saved["source_id"]), str(saved["magnet_id"]), str(saved["electric_id"])) == (
                source_id, magnet_id, identity)
            parts.append({k: saved[k] for k in saved.files})
        print(f"Completed through ID {expected[-1]}", flush=True)
    first = parts[0]
    for part in parts[1:]:
        np.testing.assert_array_equal(part["gun_z_mm"], first["gun_z_mm"])
    current_keys = ("ids", "weight", "alive", "phase", "energy_offset_ev", "dpa_xy_m",
                    "dpa_passed", "c1_passed", "loss_key", "loss_z_mm", "selected_ids")
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in current_keys}
    np.testing.assert_array_equal(arrays["selected_ids"], selected_ids)
    with np.load(INPUTS / record["bore_field"]["file"]) as f:
        radius_limit = json.loads(str(f["receipt"]))["local_radius_m"]
    max_r = max(float(p["maximum_radius_m"]) for p in parts)
    if max_r >= radius_limit:
        raise ValueError(f"Actual beam radius {max_r:g} m exceeds the {radius_limit:g} m magnetic fit")
    path = HERE / "gun-exit-particles.npz"
    temp = path.with_suffix(".partial.npz")
    np.savez_compressed(temp, **arrays, gun_z_mm=first["gun_z_mm"],
        gun_paths_m=np.concatenate([p["gun_paths_m"] for p in parts], axis=1),
        launch=np.column_stack([getattr(emission, k) for k in ("x_m", "y_m", "tx_rad", "ty_rad")]),
        launch_energy_offset_ev=emission.energy_offset_ev, emitted_A=gun.emitted_current_a, lens_kV=kv,
        cells_per_bore=gun.base_electric_field.request["numerics"]["cells_per_bore"],
        max_energy_error_eV=max(float(p["max_energy_error_eV"]) for p in parts),
        maximum_radius_m=max_r, magnetic_validity_radius_m=radius_limit, parent_case=CASE,
        source_id=source_id, magnet_id=magnet_id, electric_id=identity)
    temp.replace(path)
    print(f"Saved {path.name} with {arrays['alive'].sum()} survivors; max radius {max_r*1e6:.4f} um", flush=True)


def paired_check(gun, emission, kv, identity):
    # These are 1,024 original IDs with their full-run absolute weights.
    # This independent run checks the source image and the exit-angle statistic.
    bundle = EmissionBundle(**{f.name: getattr(emission, f.name)[:1024] for f in fields(EmissionBundle)})
    planes = [23.9, 23.975, 24., 24.01, 24.025, 24.075, 24.1, 26., 450.]
    observer = ExactGunPlaneObserver(planes, gun.exit_plane_z_mm)
    with gun_step_workers():
        result = _trace_feg_to_exit(gun, emitted_bundle=bundle, observer=observer)
    obs = observer.result()
    np.testing.assert_array_equal(bundle.ray_id, result.exit_bundle.ray_id)
    assert obs.reached.all() and obs.alive.all() and result.exit_bundle.alive.all()
    rows = []
    for i, z in enumerate(planes):
        s = transverse_beam_statistics(obs.x_m[i], obs.y_m[i], obs.tx_rad[i], obs.ty_rad[i],
                                       weights=bundle.weight)
        rows.append(dict(z_mm=z, rms_radius_nm=s.radius_rms_m*1e9, D95_nm=s.radius_95_m*2e9,
            alpha95_urad=s.convergence_95_rad*1e6,
            rms_envelope_slope_urad=s.radial_position_angle_covariance_m_rad/s.radius_rms_m*1e6))
    write_json(HERE / "matched-refinement.json", dict(lens_kV=kv, field_sha256=identity,
        grid_shape=gun.base_electric_field.report["grid_shape"], trace_step_mm=gun.trace_step_mm,
        number_original=len(bundle.ray_id), first_id=int(bundle.ray_id[0]), last_id=int(bundle.ray_id[-1]),
        represented_current_uA=float(bundle.weight.sum()*gun.emitted_current_a*1e6), planes=rows))
    print("Refined finite-source exit and foil:", rows[-1], rows[2], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kv", type=float)
    parser.add_argument("--mesh", type=int)
    parser.add_argument("--paired-check", action="store_true")
    options = parser.parse_args()
    transport(options.kv, options.mesh, options.paired_check)
