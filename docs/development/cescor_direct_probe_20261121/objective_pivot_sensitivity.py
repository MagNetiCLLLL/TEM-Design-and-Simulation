"""Measure fixed-drive scan and specimen-scale sensitivity to objective current."""

import json
from hashlib import sha256

import cupy as cp
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import polar
from scipy.optimize import brentq, minimize_scalar

from downstream_pivot import DrivenTrace, inputs, moment, scaled, write
from relay_transfer import Native, START, SURFACE
from scan_waist_audit import EVIDENCE, FIELD, check_planes, locate_planes
from column_current import source_digest
from compare import shared_inputs


def closest_pivot(stations, response, columns=(slice(None), 0, 1)):
    # d(centroid displacement)/dz is the mechanical-slope response for a fixed cohort.
    spline = CubicHermiteSpline(stations, response[:, :2], response[:, 2:4], axis=0)
    def minimum(cols):
        def error(z):
            return np.linalg.norm(spline(z)[:, cols])**2 * 1e18
        # Use an offset coordinate so the stopping tolerance is relative to the local span.
        z0 = stations[len(stations)//2]
        fit = minimize_scalar(lambda dz: error(z0+dz), bounds=(stations[0]-z0, stations[-1]-z0),
                              method="bounded", options={"xatol": 3e-14})
        if not fit.success or not stations[1] < z0+fit.x < stations[-2]:
            raise ValueError("The scan-pivot minimum is not bracketed")
        return z0 + fit.x
    return [minimum(cols) for cols in columns]


def summary(response):
    pos, tilt = response[:2]*1e9, response[2:4]*1e6
    return dict(position_nm_per_urad=pos.tolist(), tilt_urad_per_urad=tilt.tolist(),
                position_singular_nm_per_urad=np.linalg.svd(pos, compute_uv=False).tolist(),
                tilt_singular_urad_per_urad=np.linalg.svd(tilt, compute_uv=False).tolist())


def trial(ni, original, focus0, q, mix, step):
    field_path = shared_inputs(ni) / f"armco-{abs(ni):.12g}-gap-fine-p8.npz"
    coarse_grid = np.r_[np.arange(1465., 1599., .5)*.001, SURFACE]
    model = Native(FIELD, step, 32768, ni, coarse_grid)
    (focus,) = locate_planes(model, q, coarse_grid)["mini_and_objective"]
    del model
    near = focus0 + np.arange(-4, 4.1, .25) * .001
    planes = np.unique(np.r_[near, focus, SURFACE])
    model = Native(FIELD, step, 32768, ni, planes)
    if source_digest(model.t) != original["source_digest"]:
        raise ValueError("Gun-exit checkpoint changed")
    focal_check = check_planes(model, q, {"installed": [], "mini_and_objective": [focus]}, [focus])
    focal_check = focal_check["mini_and_objective"][0]["checks"]
    _, observations = model.replay(q, near)
    weights = model.t.weights
    live = observations[0][1] == -1
    if not all(np.array_equal(loss == -1, live) for _, loss in observations):
        raise ValueError("The waist scan changed the surviving cohort")
    moments = np.array([moment(p[live], weights[live]) for p, _ in observations])
    del observations
    i, = np.flatnonzero((moments[:-1, 1] < 0) & (moments[1:, 1] >= 0))
    waist = brentq(lambda z: np.interp(z, near[i:i+2], moments[i:i+2, 1]), near[i], near[i+1])
    trace = DrivenTrace(model)
    field = model.field(q)
    response, _, masks = trace.response(field, mix, planes, 1.)
    pivots = closest_pivot(near, response[np.searchsorted(planes, near)])
    sample = summary(response[-1])
    at_focus = summary(response[np.searchsorted(planes, focus)])
    original_live = model.t.ids[masks[-1]]
    del trace, model, field, masks
    cp.get_default_memory_pool().free_all_blocks()

    # Fresh launches at the fitted plane and half the command amplitude independently check the fit.
    verify = np.unique(np.r_[focus, pivots[0], SURFACE])
    model = Native(FIELD, step, 32768, ni, verify)
    trace = DrivenTrace(model)
    field = model.field(q)
    half, even, masks = trace.response(field, mix, verify, .5)
    if not np.array_equal(original_live, model.t.ids[masks[-1]]):
        raise ValueError("Changing observation stations changed specimen IDs")
    return dict(objective_At=ni, objective_current_A=ni/1160., objective_field_sha256=sha256(field_path.read_bytes()).hexdigest(),
                source_sha256=source_digest(model.t), emitted=len(model.t.ids), surviving=int(masks[-1].sum()),
                effective=float(weights[masks[-1]].sum()**2/(weights[masks[-1]]@weights[masks[-1]])),
                surviving_ID_sha256=sha256(original_live.tobytes()).hexdigest(),
                specimen_current_A=float(model.t.current*weights[masks[-1]].sum()),
                focus_mm=focus*1e3, two_lens_focal_checks=focal_check, radial_waist_mm=waist*1e3,
                radial_waist_bracket_mm=(near[i:i+2]*1e3).tolist(),
                closest_common_X_Y_pivots_mm=(np.array(pivots)*1e3).tolist(),
                common_plane_half_drive_check=summary(half[np.searchsorted(verify, pivots[0])]),
                common_plane_even_nm=float(1e9*np.linalg.norm(even[np.searchsorted(verify,pivots[0]), :2])),
                fixed_drives_at_new_focus=at_focus, specimen=sample, specimen_half_drive_check=summary(half[-1]))


def main():
    reference, focus0 = inputs()
    fit = json.loads((EVIDENCE/"downstream-pivot.json").read_text())
    mixing = np.array(fit["runs"][-1]["observations"]["matched"]["station_command_matrix"])
    q = scaled(reference["currents_A"], fit["TL12_gain"])
    ni0 = reference["objective_At"]
    runs = []
    for delta in (0., 116., -116.):
        row = trial(ni0+delta, reference, focus0, q, mixing, .075)
        if not runs:
            reference_scale = np.array(row["specimen"]["position_nm_per_urad"])
        scale = np.array(row["specimen"]["position_nm_per_urad"]) @ np.linalg.inv(reference_scale)
        rotation, stretch = polar(scale)
        row["relative_scale_principal_percent"] = ((np.linalg.eigvalsh(stretch)-1)*100).tolist()
        row["relative_scan_rotation_deg"] = float(np.rad2deg(np.arctan2(rotation[1,0], rotation[0,0])))
        runs.append(row)
        print(ni0+delta, row['focus_mm'], row['closest_common_X_Y_pivots_mm'],
              row['relative_scale_principal_percent'], flush=True)
        cp.get_default_memory_pool().free_all_blocks()
    write(EVIDENCE/"objective-pivot-sensitivity.json", dict(source="Frozen 300 keV gun-exit checkpoint",
          source_plane_mm=START*1e3, specimen_upper_mm=SURFACE*1e3, corrector_sha256=reference["corrector_sha256"],
          objective_turns=1160, max_step_mm=.075, TL12_gain=fit['TL12_gain'], station_command_matrix=mixing.tolist(),
          focus_definition="Canonical tilt-transfer zero from the isolated Mini and incident-side objective field to the specimen upper surface.",
          pivot_definition="Minimum Frobenius norm of the two-axis fixed-drive centroid position response in the complete field, with independent X and Y minima and residuals.",
          scale_definition="Singular stretches and polar rotation of each specimen scan-position matrix relative to the nominal matrix; quasistatic near the scan center.", runs=runs))


if __name__ == "__main__":
    main()
