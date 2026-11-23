"""Compare the static TL12 crossover and the specimen-referenced front focal plane."""

import gzip
import json
from hashlib import sha256

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from column_current import ACTIVE, FACTOR, OWNER, POINT, source_digest
from condenser_normal import FIELD
from particles import Tracker
from relay_transfer import HERE, PARTS, START, SURFACE, Native, advance


EVIDENCE = HERE / "evidence"
J = np.array([[0., -1.], [1., 0.]])
ENERGY = 300000.


def archived_scan():
    nodes = json.loads(gzip.decompress((POINT / "inputs.json.gz").read_bytes()))["graph"]["nodes"]
    (ac,) = [n["attributes"] for n in nodes if n.get("type", "").endswith(":AcDeflectorComponent")]
    (dp12,) = [n["attributes"] for n in nodes if n.get("type", "").endswith(":Dp12ScanDeflectorComponent")]
    settings = {k: ac[k] for k in ("scan_enabled", "wobble_enabled", "kick_x_mrad", "kick_y_mrad")}
    offsets = {k: dp12[k] for k in ("upper_x_mrad", "upper_y_mrad", "lower_x_mrad", "lower_y_mrad")}
    if any(settings.values()) or any(offsets.values()):
        raise ValueError("The reference is not a zero-command static scan")
    settings["DP12_offsets"] = offsets
    return settings


def field_cases(model, drives):
    mesh = model.m
    full = model.field(drives)
    grid, index, fraction = model.t.tables
    mid = grid.shape[1] // 2
    axis = grid[:, mid, mid, 2]
    correction = (1 - fraction) * axis[index] + fraction * axis[index + 1]
    yield "installed", model.t, full, (full[:, 0] + correction).get()
    # The material solve includes both poles of one objective circuit. Use
    # its incident-side field up to the specimen, including the coupled fringe.
    mini = OWNER[ACTIVE] == "mini_condenser"
    isolated = model.bg - mesh.group[0] + np.einsum("i,ijk->jk", drives[mini], model.coils[mini])
    plain = cp.asarray(np.column_stack((isolated, np.zeros_like(mesh.hex))))
    yield "mini_and_objective", Tracker(mesh), plain, isolated[:, 0]


def angular_map(matrices, k_surface, rotation):
    position = matrices[..., :2, 2:]
    mechanical = matrices[..., 2:, 2:]
    canonical = mechanical - k_surface * (J @ position)
    rotated = np.swapaxes(rotation, -1, -2) @ canonical
    return (np.trace(rotated, axis1=-2, axis2=-1) / 2, position, mechanical, canonical)


def rotation_to_sample(model, tracker, axis, planes):
    kin = ENERGY + tracker.ep[:, 0, 0].get()
    k = FACTOR * tracker.pc * axis / np.sqrt(kin * (kin + 2 * tracker.rest))
    angle = np.r_[0., np.cumsum(np.diff(model.m.z) * (k[:-2:2] + 4*k[1::2] + k[2::2]) / 6)]
    at = model.indices(planes)
    delta = angle[model.indices([SURFACE])[0]] - angle[at]
    return (np.cos(delta)[:, None, None]*np.eye(2) + np.sin(delta)[:, None, None]*J,
            k[2*model.indices([SURFACE])[0]])


def differential(tracker, field, first, stops, scale=1., angular_only=False):
    amplitudes = scale * np.array([1e-7, 1e-7, 1e-6, 1e-6])
    basis = np.diag(amplitudes)
    if angular_only:
        basis, amplitudes = basis[2:], amplitudes[2:]
    n = len(amplitudes)
    launch = np.column_stack((np.vstack((basis, -basis)), np.zeros(2*n)))
    states = advance(tracker, field, launch, np.full(2*n, ENERGY), np.zeros(2*n),
                     np.full(2*n, -1, np.int32), first, stops)
    if any((loss != -1).any() for _, loss in states):
        raise ValueError("A differential ray left the physical bore")
    return np.stack([((p[:n, :4] - p[n:, :4]) / (2 * amplitudes[:, None])).T for p, _ in states])


def locate_planes(model, drives, grid):
    roots = {}
    idx = model.indices(grid)
    for name, tracker, field, axis in field_cases(model, drives):
        forward = differential(tracker, field, idx[0], idx)
        to_sample = forward[-1] @ np.linalg.inv(forward)
        rotation, ks = rotation_to_sample(model, tracker, axis, grid)
        values = angular_map(to_sample, ks, rotation)[0]
        fit = CubicSpline(grid, values)
        brackets = np.flatnonzero(values[:-1] * values[1:] < 0)
        roots[name] = [brentq(fit, grid[i], grid[i+1], xtol=2e-14) for i in brackets]
    return roots


def check_planes(model, drives, root_sets, stations):
    result = {}
    for name, tracker, field, axis in field_cases(model, drives):
        planes = np.unique(np.r_[root_sets[name], stations])
        rotation, ks = rotation_to_sample(model, tracker, axis, planes)
        rows = []
        for z, rot, idx in zip(planes, rotation, model.indices(planes)):
            checks = []
            for amplitude in (1., .5):
                # Independent starts at each candidate test both input tilts directly.
                differential_columns = differential(tracker, field, idx, model.indices([SURFACE]),
                                                    amplitude, angular_only=True)[0]
                matrix = np.zeros((4, 4))
                matrix[:, 2:] = differential_columns
                d, b, mech, canon = angular_map(matrix, ks, rot)
                checks.append(dict(symmetric_tilt_urad=amplitude, signed_larmor_D=float(d),
                                   canonical_D_singular=np.linalg.svd(canon, compute_uv=False).tolist(),
                                   mechanical_D_singular=np.linalg.svd(mech, compute_uv=False).tolist(),
                                   B_singular_mm=(1e3*np.linalg.svd(b, compute_uv=False)).tolist()))
            rows.append(dict(z_mm=float(1e3*z), root_candidate=bool(z in root_sets[name]), checks=checks))
        result[name] = rows
    return result


def static_pass(model, drives, stations, scale, cohort_ids=None):
    t = model.t
    controls = drives.copy()
    controls[OWNER[ACTIVE] == "probe_tl12_lens"] *= scale
    field = model.field(controls)
    p = cp.asarray(t.input.copy())
    n = len(p)
    if n < 1000 or np.unique(t.ids[t.weights > 0]).size != n:
        raise ValueError("Insufficient distinct positive-current particles")
    w, en, ref = map(cp.asarray, (t.weights, model.en, model.phi0))
    cohort = cp.asarray(np.ones(n, bool) if cohort_ids is None else np.isin(t.ids, cohort_ids))
    loss, peak, margin = cp.full(n, -1, cp.int32), cp.zeros(n), cp.full(n, cp.inf)
    rows = []
    first_ids = None
    first = 0
    for end in model.indices(stations):
        for i in range(first, end, 160):
            t.kernel(((n+127)//128,), (128,), (p, en, ref, field, t.ep, t.er, np.int32(len(t.r2)),
                     t.h, t.rad, loss, peak, margin, np.int32(i), np.int32(min(i+160, end)),
                     np.int32(n), t.rest, FACTOR, t.pc))
        live = loss == -1
        if first_ids is None:
            first_ids = t.ids[live.get()]
        weights = cp.where(live & cohort, w, 0.)
        weights /= cp.sum(weights)
        centered = p[:, :4] - weights @ p[:, :4]
        covariance = (centered.T * weights) @ centered
        rows.append(cp.stack((live.sum(), covariance[0,2], covariance[1,3],
                              covariance[0,0], covariance[1,1])))
        first = end
    rows = cp.stack(rows).get()
    final_ids = t.ids[(loss == -1).get()]
    if np.any(loss.get() < -1) or not np.array_equal(first_ids, final_ids):
        raise ValueError("A loss or unresolved step interrupts the local waist bracket")
    measured_ids = final_ids if cohort_ids is None else np.intersect1d(final_ids, cohort_ids)
    if cohort_ids is not None and not np.array_equal(measured_ids, cohort_ids):
        raise ValueError("The fixed comparison cohort lost a particle inside the trial")
    windows = []
    for col in (1, 2):
        indices = np.flatnonzero((rows[:-1, col] < 0) & (rows[1:, col] >= 0))
        windows.append([[float(1000*stations[i]), float(1000*stations[i+1])] for i in indices])
    measured_w = t.weights[np.isin(t.ids, measured_ids)]
    record = dict(TL12_gain=scale, emitted=n, local_survivors=len(final_ids), measured=len(measured_ids),
                effective_measured=float(measured_w.sum()**2 / (measured_w @ measured_w)),
                local_current_A=float(t.current*t.weights[np.isin(t.ids, final_ids)].sum()),
                measured_current_A=float(t.current*measured_w.sum()),
                surviving_ID_sha256=sha256(final_ids.tobytes()).hexdigest(),
                measured_ID_sha256=sha256(measured_ids.tobytes()).hexdigest(),
                source_sha256=source_digest(t), region_mm=(1000*stations[[0,-1]]).tolist(),
                observation_spacing_mm=float(1000*np.max(np.diff(stations))),
                x_waist_brackets_mm=windows[0], y_waist_brackets_mm=windows[1])
    return record, final_ids


def main():
    rec = json.loads((EVIDENCE / "illuminated-normal.json").read_text())["final"]
    drives = np.array(rec["currents_A"])
    if sha256(FIELD.read_bytes()).hexdigest() != rec["corrector_sha256"]:
        raise ValueError("The required corrector field has changed")
    pair = START + np.array(PARTS["ac_deflector"]["interaction_centers_local_z_mm"]) * .001
    grid = np.r_[np.arange(1340., 1599.2, .5)*.001, SURFACE]
    coarse = Native(FIELD, .075, 32768, rec["objective_At"], grid)
    roots = locate_planes(coarse, drives, grid)
    del coarse
    cp.get_default_memory_pool().free_all_blocks()
    waist_grid = np.arange(1340., 1422.25, .25)*.001
    stations = np.r_[pair, pair.mean(), 1.3755, 1.4085]
    fine = Native(FIELD, .0375, 32768, rec["objective_At"],
                  np.unique(np.r_[waist_grid, stations, *roots.values()]))
    checks = check_planes(fine, drives, roots, stations)
    static, ids = zip(*(static_pass(fine, drives, waist_grid, scale) for scale in (1., 0.)))
    common = np.intersect1d(*ids)
    paired = [static_pass(fine, drives, waist_grid, scale, common)[0] for scale in (1., 0.)]
    if any(case["source_sha256"] != rec["source_digest"] for case in (*static, *paired)):
        raise ValueError("The gun-exit boundary differs from the reported working point")
    report = dict(source="Archived 300 keV gun exit at 450 mm; identical input for the matched control.",
                  static_scan=archived_scan(), scan_centers_mm=(1000*pair).tolist(),
                  objective_At=rec["objective_At"], specimen_upper_mm=1000*SURFACE,
                  working_point="illuminated-normal.json / final", corrector_sha256=rec["corrector_sha256"],
                  focal_definition="Point tilt to the specimen upper surface. A canonical front-focal plane requires both singular values of the canonical D block to vanish and a full-rank position-from-tilt B block. Mechanical slopes are reported in the immersion field.",
                  focal_domain_mm=(grid[[0,-1]]*1000).tolist(), locate_step_mm=.075, verify_step_mm=.0375,
                  signed_root_candidates_mm={k:(np.array(v)*1000).tolist() for k,v in roots.items()},
                  direct_plane_checks=checks, static_original_particles=static, common_particle_control=paired,
                  driven_pivot="Not measured; raster and wobble were off.")
    out = EVIDENCE / "scan-waist-audit.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for name, values in roots.items():
        print(name, 1000*np.array(values), flush=True)
    print(static, flush=True)


if __name__ == "__main__":
    main()
