"""Attribute the two final Nanoprobe waists using the original-current ensemble."""

import json
from hashlib import sha256

import numpy as np

from column_current import ACTIVE, OWNER, source_digest
from downstream_pivot import DrivenTrace, write
from illuminated_focus import column_planes
from refocus_a2 import objective_receipt
from relay_transfer import HERE, PARTS, START, WORK, Native, advance
from scan_waist_audit import archived_scan, field_cases
from scan_winding_basis import grid as scan_grid


EVIDENCE = HERE / "evidence"
CAPTURE = WORK / "normal-core-fixed-0.0375-capture.npz"
REPORT = EVIDENCE / "normal-scan-waist-origin.json"


def nearest(grid, targets):
    """Return existing integration nodes so that observing a plane cannot change the orbit."""
    right = np.clip(np.searchsorted(grid, targets), 1, len(grid)-1)
    return np.where(abs(grid[right]-targets) < abs(grid[right-1]-targets), right, right-1)


def observe(model, field, indices, cohort, *, restart=None, checkpoint_index=None):
    t = model.t
    weights = t.weights[cohort]
    live_at_entry = None
    checkpoint = None
    rows = []
    start, phase, loss = (0, t.input, model.loss) if restart is None else restart
    for chunk in np.array_split(indices, (len(indices)+15)//16):
        states = advance(t, field, phase, model.en, model.phi0, loss, start, chunk)
        for idx, (phase, loss) in zip(chunk, states, strict=True):
            live = loss == -1
            if live_at_entry is None:
                live_at_entry = live.copy()
            if not np.array_equal(live, live_at_entry) or not np.all(live[cohort]):
                raise ValueError("A local contact changed the comparison cohort")
            if idx == checkpoint_index:
                checkpoint = (idx, phase.copy(), loss.copy())
            points = phase[cohort, :4]
            mean = np.average(points, weights=weights, axis=0)
            centered = points - mean
            squared = np.sum(centered[:, :2]**2, axis=1)
            cross = np.sum(centered[:, :2]*centered[:, 2:4], axis=1)
            rows.append([*mean[:2]*1e6, np.sqrt(np.average(squared, weights=weights))*1e6,
                         np.average(cross, weights=weights)*1e9])
        start = chunk[-1]
    return np.array(rows), live_at_entry, checkpoint


def summarize(z_mm, rows, live, tracker):
    # Half the derivative of centered mean square radius equals cov(x,tx) + cov(y,ty).
    cross = rows[:, 3]
    lo = np.flatnonzero((cross[:-1] < 0) & (cross[1:] >= 0))
    return dict(z_mm=z_mm.tolist(), rms_radius_um=rows[:, 2].tolist(),
                radial_covariance_um_mrad=cross.tolist(),
                waist_brackets_mm=np.stack((z_mm[lo], z_mm[lo+1]), axis=1).tolist(),
                local_survivors=int(live.sum()), local_current_A=float(tracker.current*tracker.weights[live].sum()))


def run():
    probe = json.loads((EVIDENCE / "normal-objective-probe.json").read_text())
    final = probe["after"]
    q = np.array([a for _, a in probe["fixed_windings"]])
    np.testing.assert_array_equal(OWNER[ACTIVE], [owner for owner, _ in probe["fixed_windings"]])
    path = WORK / "normal-core-field.npz"
    if sha256(path.read_bytes()).hexdigest() != final["corrector_table_sha256"]:
        raise ValueError("Incorrect final corrector table")
    receipt = objective_receipt(final["objective_At"])
    if receipt["curve_sha256"] != final["objective_material_sha256"]:
        raise ValueError("Incorrect objective material")
    m = Native(path, probe["max_step_mm"], probe["source_packets"], final["objective_At"], column_planes())
    t = m.t
    if (source_digest(t) != probe["source_digest"] or np.unique(t.ids[t.weights > 0]).size != 32768
            or t.current != probe["source_current_A"]):
        raise ValueError("Incorrect original-current source")
    with np.load(CAPTURE) as data:
        np.testing.assert_array_equal(data["source_id"], t.ids)
        np.testing.assert_array_equal(data["integration_z_m"], m.m.z)
        np.testing.assert_array_equal(data["weight"], t.weights)
        cohort = data["specimen_loss"] == -1
        observed = data["track_z_m"]
        archive_rms = 1e6*np.hypot(data["moments"][:, 1], data["moments"][:, 2])
    with np.load(EVIDENCE / "normal-objective-landings.npz") as data:
        np.testing.assert_array_equal(cohort, data["after_loss"] == -1)
        np.testing.assert_array_equal(t.ids, data["source_id"])

    ac = PARTS["ac_deflector"]
    centers = 1000*START + np.asarray(ac["interaction_centers_local_z_mm"])
    entrance_mm = 1000*START + ac["local_start_z_mm"]
    exit_mm = 1000*START + ac["local_end_z_mm"]
    tl12 = PARTS["probe_tl12_lens"]
    tl_span = [1000*START + tl12[key] for key in ("local_start_z_mm", "local_end_z_mm")]
    indices = np.unique(nearest(m.m.z, np.r_[np.arange(1340., 1422.01, .25), centers, entrance_mm]*.001))
    z_mm = m.m.z[indices]*1000
    entry = nearest(m.m.z, entrance_mm*.001)
    local = indices >= entry
    _, _, nominal_field, bz = next(field_cases(m, q))
    mask = OWNER[ACTIVE] == "probe_tl12_lens"
    tl_bz = q[mask] @ m.coils[mask, :, 0]
    q_off = q.copy()
    q_off[mask] = 0.
    field_off = m.field(q_off)
    static_scan = archived_scan()
    print("Replaying the archived scan-off column", flush=True)
    base, original_live, checkpoint = observe(m, nominal_field, indices, cohort, checkpoint_index=entry)
    np.testing.assert_array_equal(cohort, original_live)
    common_z, i, j = np.intersect1d(observed, m.m.z[indices], return_indices=True)
    if len(common_z) < 80:
        raise ValueError("Too few exact common observations with the published ray diagram")
    np.testing.assert_allclose(base[j, 2], archive_rms[i], rtol=2e-10, atol=1e-8)
    print("Disabling only the three TL12 windings", flush=True)
    off, off_live, _ = observe(m, field_off, indices, cohort)
    local_off, local_live, _ = observe(m, field_off, indices[local], cohort, restart=checkpoint)
    np.testing.assert_array_equal(local_off[0], base[local][0])
    w = t.weights[cohort]
    z_field = m.m.nodes*1000
    report = dict(reference="normal-objective-probe.json / after", source_sha256=source_digest(t),
                  emitted=len(t.ids), emitted_A=t.current, effective_cohort=float(w.sum()**2 / (w @ w)),
                  cohort_count=int(cohort.sum()), cohort_A=float(t.current*w.sum()),
                  cohort_ID_sha256=sha256(t.ids[cohort].tobytes()).hexdigest(), max_step_mm=m.step,
                  corrector_table_sha256=final["corrector_table_sha256"], objective_At=final["objective_At"],
                  TL12_model_winding_A=q[mask].tolist(), scan_off=static_scan,
                  scan_centers_mm=centers.tolist(), AC_span_mm=[entrance_mm, exit_mm], TL12_span_mm=tl_span,
                  field=dict(z_mm=z_mm.tolist(), total_Bz_T=np.interp(z_mm, z_field, bz).tolist(),
                             TL12_Bz_T=np.interp(z_mm, z_field, tl_bz).tolist(),
                             center_total_Bz_T=np.interp(centers, z_field, bz).tolist(),
                             center_TL12_Bz_T=np.interp(centers, z_field, tl_bz).tolist()),
                  archive_exact_common_planes=len(common_z),
                  scan_off_installed=summarize(z_mm, base, original_live, t),
                  TL12_off_from_gun=summarize(z_mm, off, off_live, t),
                  TL12_off_from_AC_entrance=summarize(z_mm[local], local_off, local_live, t))

    # Apply the four installed saddle windings on the same static-corrector grid.
    with np.load(path) as table:
        for axis, values in zip(("z_local_m", "xy_m"), scan_grid(), strict=True):
            np.testing.assert_array_equal(table[axis], values)
    trace = DrivenTrace(m)
    mix = np.array(final["downstream_calibration"]["station_command_matrix"])
    probes = np.unique(nearest(z_mm, np.r_[centers, entrance_mm, exit_mm, 1375., 1375.25, 1375.5,
                                          1408., 1408.25, 1408.5, 1408.75]))
    runs = []
    for direction in (np.zeros(2), np.array([1., 0.]), np.array([-1., 0.]), np.array([0., 1.]), np.array([0., -1.])):
        amps = trace.set_drive(mix @ direction)
        print("Testing scan command (urad)", direction, flush=True)
        rows, live, _ = observe(m, nominal_field, indices[probes], cohort)
        np.testing.assert_array_equal(live, original_live)
        if not direction.any():
            zero = rows
            np.testing.assert_allclose(zero, base[probes], rtol=5e-8, atol=1e-8)
        runs.append(dict(command_urad=direction.tolist(), coil_A=amps.tolist(),
                         centroid_shift_nm=((rows[:, :2]-zero[:, :2])*1000).tolist(),
                         rms_change_nm=((rows[:, 2]-zero[:, 2])*1000).tolist(),
                         radial_covariance_um_mrad=rows[:, 3].tolist()))
    report["driven_scan"] = dict(planes_mm=z_mm[probes].tolist(), runs=runs,
                                 station_matrix=mix.tolist(), winding_receipt=trace.receipt)
    write(REPORT, report)


if __name__ == "__main__":
    run()
