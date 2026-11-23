"""Replay the fixed Nanoprobe Normal column and compare its scan-region topology."""

import json

import numpy as np

from column_current import ACTIVE, OWNER, START, SURFACE, current_ledger, source_digest
from normal_working_point import AUDIT, EVIDENCE, file_hash, operating_point
from plot_internal_trajectories import EDGES, select_tracks
from relay_transfer import FOIL, PARTS, RADIUS, advance
from scan_coil_waist_origin import nearest
from scan_waist_audit import archived_scan, field_cases
from temsim.corrector_field_cli import _save


TRACKS = EVIDENCE / "normal-joint-rays.npz"
REPORT = EVIDENCE / "normal-joint-rays.json"
WINDOW_MM = (1340., 1422.)
AC = PARTS["ac_deflector"]
SCAN_MM = 1000*START + np.asarray(AC["interaction_centers_local_z_mm"])


def passage(model, field):
    stations = [*FOIL, WINDOW_MM[0]*.001, WINDOW_MM[1]*.001, SURFACE]
    return advance(model.t, field, model.t.input, model.en, model.phi0, model.loss, 0,
                   nearest(model.m.z, np.array(stations)))


def display_rays(front, rear, surface_loss):
    chosen, bands, _ = select_tracks(front[0], {"loss": surface_loss})
    kept = bands >= 0
    intercepted = np.flatnonzero((front[1] == -1) & (rear[1] >= 0))
    azimuth = np.arctan2(front[0][intercepted, 1], front[0][intercepted, 0])
    ordered = intercepted[np.argsort(azimuth)]
    n = min(12, len(ordered))
    blocked = ordered[np.linspace(0, len(ordered)-1, n, dtype=int)]
    return np.r_[chosen[kept], blocked], np.r_[bands[kept], np.full(n, -1)]


def beam_moments(phase, mask, weights):
    points = phase[mask, :4]
    w = weights[mask] / weights[mask].sum()
    centered = points - np.average(points, axis=0, weights=w)
    cov = (centered.T * w) @ centered
    # Any unit direction u has half-variance derivative u.T @ growth @ u.
    growth = (cov[:2, 2:] + cov[:2, 2:].T) * .5e9
    return (np.sqrt(cov[0, 0] + cov[1, 1])*1e6, growth[0, 0], growth[1, 1],
            *np.linalg.eigvalsh(growth))


def brackets(z, covariance):
    starts = np.flatnonzero((covariance[:-1] < 0) & (covariance[1:] >= 0))
    return np.column_stack((z[starts], z[starts+1])).tolist()


def save_rays(arrays):
    if TRACKS.is_file():
        with np.load(TRACKS, allow_pickle=False) as saved:
            if set(saved.files) == set(arrays) and all(
                np.array_equal(saved[name], values, equal_nan=True) for name, values in arrays.items()
            ):
                return
    _save(TRACKS, **arrays)


def describe(z, rows):
    radial = rows[:, 1] + rows[:, 2]
    near = (z >= 1403) & (z <= 1413)
    ac = (z >= START*1000 + AC["local_start_z_mm"]) & (z <= START*1000 + AC["local_end_z_mm"])
    return dict(radial_waist_brackets_mm=brackets(z, radial),
                x_waist_brackets_mm=brackets(z, rows[:, 1]), y_waist_brackets_mm=brackets(z, rows[:, 2]),
                radial_covariance_1403_1413_um_mrad=[float(radial[near].min()), float(radial[near].max())],
                min_principal_growth_in_AC_um_mrad=float(rows[ac, 3].min()),
                scan_center_rms_radius_um=np.interp(SCAN_MM, z, rows[:, 0]).tolist(),
                scan_center_radial_covariance_um_mrad=np.interp(SCAN_MM, z, radial).tolist())


def capture(model, field, decision, common, selected, *, full):
    z = model.m.z
    local_targets = np.r_[np.arange(WINDOW_MM[0], WINDOW_MM[1]+.01, .125), SCAN_MM, 1408.]
    local = np.unique(nearest(z, local_targets*.001))
    stations = local
    if full:
        overview = np.r_[np.arange(START*1000, SURFACE*1000, .75), FOIL*1000, SURFACE*1000]
        stations = np.unique(np.r_[nearest(z, overview*.001), local])
    rows, matched, tracks = [], [], []
    local_set = set(local)
    t = model.t
    j, phase, loss = 0, t.input, model.loss
    expected_live = decision[2][1] == -1
    np.testing.assert_array_equal(expected_live, decision[3][1] == -1)
    for indices in np.array_split(stations, (len(stations)+19)//20):
        states = advance(t, field, phase, model.en, model.phi0, loss, j, indices)
        for idx, (phase, loss) in zip(indices, states, strict=True):
            if full:
                tracks.append(np.where((loss[selected] == -1)[:, None], phase[selected, :4], np.nan))
            if idx in local_set:
                np.testing.assert_array_equal(loss == -1, expected_live)
                current = beam_moments(phase, expected_live, t.weights)
                rows.append(current)
                matched.append(current if np.array_equal(common, expected_live)
                               else beam_moments(phase, common, t.weights))
        j = indices[-1]
    if full:
        np.testing.assert_array_equal(phase, decision[-1][0])
        np.testing.assert_array_equal(loss, decision[-1][1])
    z_mm, rows, matched = z[local]*1000, np.asarray(rows), np.asarray(matched)
    arrays = dict(z_mm=z_mm, live_moments=rows, matched_moments=matched)
    if full:
        arrays.update(track_z_mm=z[stations]*1000, tracks=np.array(tracks),
                      surface_phase=phase, surface_loss=loss, local_live=expected_live)
    return arrays, dict(all_live=describe(z_mm, rows), common=describe(z_mm, matched))


def run():
    archived, old_drives, base = operating_point(archived=True)
    original = EVIDENCE / "normal-objective-landings.npz"
    if file_hash(original) != base["landing_archive_sha256"]:
        raise ValueError("Archived control particle data has changed")
    old_static = archived.field(old_drives)
    print("Replaying the archived control from the gun", flush=True)
    old = passage(archived, old_static)
    with np.load(original, allow_pickle=False) as saved:
        for attr, name in (("ids", "source_id"), ("input", "gun_exit_phase"), ("weights", "weight"),
                           ("de", "energy_offset_eV")):
            np.testing.assert_array_equal(getattr(archived.t, attr), saved[name])
        np.testing.assert_array_equal(old[-1][0], saved["after_phase"])
        np.testing.assert_array_equal(old[-1][1], saved["after_loss"])
    selected, bands = display_rays(old[0], old[1], old[-1][1])
    model, new_drives, _ = operating_point()
    # Other windings, mesh and source must agree before comparing the two orbits.
    mask = model.m.fem_round_windings
    np.testing.assert_array_equal(new_drives[~mask], old_drives[~mask])
    np.testing.assert_array_equal(archived.m.z, model.m.z)
    try:
        model.field(old_drives)
    except ValueError as error:
        if "separate vacuum-group current" not in str(error):
            raise
    else:
        raise ValueError("A retired vacuum winding can still be energized")
    print("Tracing the joint FEM and the original source", flush=True)
    new_static = model.field(new_drives)
    new = passage(model, new_static)
    common = (old[2][1] == -1) & (new[2][1] == -1)
    t = model.t
    archive = dict(source_id=t.ids, weight=t.weights, gun_exit_phase=t.input, energy_offset_eV=t.de,
                   track_source_id=t.ids[selected], track_band=bands, common_live=common)
    joint_config = json.loads((EVIDENCE / "normal-current-working-point.json").read_text())
    report = dict(mode=base["mode"], gun_exit_mm=START*1000, specimen_top_mm=SURFACE*1000,
                  aperture_diameter_um=base["pupil_diameter_um"], source_packets=len(t.ids),
                  source_current_A=t.current, source_sha256=source_digest(t), objective_A=base["after"]["objective_A"],
                  objective_At=model.ni, scan_command=archived_scan(), scan_centers_mm=SCAN_MM.tolist(),
                  local_interval_mm=list(WINDOW_MM), matched_local_IDs=int(common.sum()),
                  matched_local_A=float(t.current*t.weights[common].sum()),
                  joint_solution_sha256=joint_config["solution_sha256"],
                  joint_currents_A=base["after"]["corrector_A"] | joint_config["replacement_current_A"],
                  replacement_turns=joint_config["replacement_turns"], inherited_settings=joint_config["base"],
                  background_owners=OWNER[ACTIVE].tolist(), archive_verified_bitwise=True,
                  moment_columns=["rms_radius_um", "cov_x_tx_um_mrad", "cov_y_ty_um_mrad",
                                  "min_principal_growth_um_mrad", "max_principal_growth_um_mrad"])
    for name, m, drives, states in (("old", archived, old_drives, old), ("joint", model, new_drives, new)):
        print("Capturing full column and scan interval:", name, flush=True)
        _, _, field, total_bz = next(field_cases(m, drives))
        arrays, summary = capture(m, field, states, common, selected, full=True)
        fill = np.linalg.norm(states[0][0][selected, :2], axis=1) / RADIUS
        plot_bands = np.where(states[-1][1][selected] == -1, np.digitize(fill, EDGES[1:-1]), -1)
        if name == "old":
            np.testing.assert_array_equal(plot_bands, bands)
        arrays.update(front_loss=states[0][1], rear_loss=states[1][1],
                      track_band=plot_bands, track_foil_radius_um=fill*RADIUS*1e6)
        for key, value in arrays.items():
            archive[name+"_"+key] = value
        local_bz = np.interp(arrays["z_mm"], m.m.nodes*1000, total_bz)
        archive[name+"_column_Bz_T"] = local_bz
        surface_moments = beam_moments(states[-1][0], states[-1][1] == -1, t.weights)
        summary.update(local_survivors=int((states[2][1] == -1).sum()),
                       local_A=float(t.current*t.weights[states[2][1] == -1].sum()),
                       ledger=current_ledger(m.m, t, states[-1][1]), step_mm=m.step,
                       corrector_table_sha256=file_hash(m.path), vacuum_group_A=drives.tolist(),
                       surface_radial_covariance_um_mrad=surface_moments[1]+surface_moments[2],
                       column_Bz_at_scans_mT=(1000*np.interp(SCAN_MM, m.m.nodes*1000, total_bz)).tolist())
        report[name] = summary
    with np.load(AUDIT / "axial-profiles.npz", allow_pickle=False) as profiles:
        report["joint"]["corrector_Bz_at_scans_mT"] = (1000*np.interp(SCAN_MM, profiles["z_mm"],
                                                                       profiles["total"])).tolist()
    actual_old = np.array(report["old"]["all_live"]["radial_waist_brackets_mm"])
    expected = json.loads((EVIDENCE / "normal-scan-waist-origin.json").read_text())["scan_off_installed"]
    old_brackets = np.array(expected["waist_brackets_mm"])
    if (actual_old.shape != old_brackets.shape or
            np.any(np.maximum(actual_old[:, 0], old_brackets[:, 0]) > np.minimum(actual_old[:, 1], old_brackets[:, 1]))):
        raise ValueError("The archived radial waist was not reproduced")
    print("Repeating the joint FEM at half the axial integration step", flush=True)
    fine, drives, _ = operating_point(step_mm=model.step/2)
    fine_field = fine.field(drives)
    half = passage(fine, fine_field)
    for i in range(len(half)):
        np.testing.assert_array_equal(new[i][1] == -1, half[i][1] == -1)
    arrays, summary = capture(fine, fine_field, half, common, selected, full=False)
    for key, value in arrays.items():
        archive["half_"+key] = value
    summary.update(step_mm=fine.step, unchanged_survivor_IDs=True)
    for population in ("all_live", "common"):
        for axis in ("radial", "x", "y"):
            key = axis+"_waist_brackets_mm"
            nominal = report["joint"][population][key]
            refined = summary[population][key]
            if len(nominal) != len(refined):
                raise ValueError("Half the axial integration step changes the scan-region topology")
            np.testing.assert_allclose(nominal, refined, rtol=0, atol=.25)
    refined_radius = np.interp(archive["joint_z_mm"], arrays["z_mm"], arrays["live_moments"][:, 0])
    summary["max_local_rms_change_um"] = float(abs(refined_radius-archive["joint_live_moments"][:, 0]).max())
    report["half_step_joint"] = summary
    save_rays(archive)
    report["ray_archive_sha256"] = file_hash(TRACKS)
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for name in ("old", "joint", "half_step_joint"):
        print(name, report[name]["all_live"], flush=True)


if __name__ == "__main__":
    run()
