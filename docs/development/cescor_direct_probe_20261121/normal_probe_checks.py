"""Verify the chosen Nanoprobe setting at fixed currents, including its calibrated raster."""

import argparse
import json
from hashlib import sha256

import cupy as cp
import numpy as np

from column_current import source_digest
from downstream_pivot import DrivenTrace, inputs, write
from fit_threefold import EVIDENCE
from normal_corrector_search import field_path, fit_surface
from objective_pivot_sensitivity import closest_pivot, summary
from refocus_a2 import objective_receipt
from relay_transfer import Native, PARTS, START, SURFACE, WORK
from scan_waist_audit import locate_planes


def coarser_pitch(tag):
    path = field_path(tag+"-pitch030")
    with np.load(field_path(tag)) as f:
        r = json.loads(str(f["report"]))
        grid = f["xy_m"][::2]
        r["transverse_pitch_mm"] = float(1000*(grid[1]-grid[0]))
        np.savez(path, xy_m=grid, z_local_m=f["z_local_m"], valid_radius_m=f["valid_radius_m"],
                 B_zyx_T=f["B_zyx_T"][:,::2,::2], report=json.dumps(r,sort_keys=True))
    return path


def measure(tag, ni, step, *, coarse=False, scan=False):
    reference, _ = inputs()
    q = np.asarray(reference["currents_A"])
    path = coarser_pitch(tag) if coarse else field_path(tag)
    receipt = objective_receipt(ni)
    m = Native(path, step, 32768, ni)
    row, p, loss = m.measure(q)
    if row["source_digest"] != reference["source_digest"]:
        raise ValueError("Full original reference source required")
    with np.load(path) as data:
        desc = json.loads(str(data["report"]))
        pitch = np.diff(data["xy_m"])[0]*1000
    out = dict(tag=tag, max_step_mm=step, transverse_pitch_mm=pitch, material_mesh_mm=desc["material_size_mm"],
               table_sha256=sha256(path.read_bytes()).hexdigest(), objective_At=ni,
               objective_field_digest=receipt["field_digest"], source_digest=row["source_digest"],
               transmitted_id_digest=sha256(m.t.ids[loss==-1].tobytes()).hexdigest(),
               current_ledger=row["current_ledger"], edge_mrad=row["rim"]["angle_mrad_min_median_max"],
               edge_reached=row["rim"]["reached_sample"], corrector_A=desc["currents_A"],
               source_count=len(loss), **fit_surface(p, loss, m.t))
    if scan:
        mix = np.array(json.loads((EVIDENCE/"fine-focus-scan.json").read_text())["station_command_matrix"])
        trace = DrivenTrace(m)
        j, even, masks = trace.response(m.field(q), mix, [SURFACE], 1.)
        np.testing.assert_array_equal(masks[0],loss==-1)
        out["scan"] = dict(amplitude_urad=1., station_command_matrix=mix.tolist(),
                           centroid_even_nm=(even[0,:2]*1e9).tolist(), **summary(j[0]))
    print(tag, step, pitch, out["D95_um"], out["edge_mrad"], out["axis_50nm_nA"],flush=True)
    return out


def downstream_planes(tag, ni):
    reference, _ = inputs()
    currents = np.asarray(reference["currents_A"])
    step = .075
    grid = np.r_[np.arange(1465., 1599., .5)*.001, SURFACE]
    model = Native(field_path(tag), step, 32768, ni, grid)
    focus, = locate_planes(model, currents, grid)["mini_and_objective"]
    del model
    lower_exit = START + PARTS["ac_deflector"]["local_end_z_mm"]*.001
    stations = np.arange(lower_exit, 1.513, .001)
    planes = np.unique(np.r_[stations, focus, SURFACE])
    mix = np.array(json.loads((EVIDENCE/"fine-focus-scan.json").read_text())["station_command_matrix"])
    model = Native(field_path(tag), step, 32768, ni, planes)
    if source_digest(model.t) != reference["source_digest"]:
        raise ValueError("The downstream pivot check requires the matched source")
    trace = DrivenTrace(model)
    response, _, masks = trace.response(model.field(currents), mix, planes, 1.)
    swept = response[np.searchsorted(planes, stations)]
    norm = np.linalg.norm(swept[:, :2], axis=(1,2))
    i = int(norm.argmin())
    result = dict(tag=tag, objective_At=ni, lower_scan_exit_mm=lower_exit*1000, max_step_mm=step,
                  isolated_front_focal_plane_mm=focus*1000, source_count=len(model.t.ids),
                  source_digest=reference["source_digest"], station_command_matrix=mix.tolist(),
                  surviving=int(masks[-1].sum()), measured_plane_range_mm=(stations[[0,-1]]*1000).tolist(),
                  best_sampled_plane_mm=stations[i]*1000,
                  minimum_sampled_frobenius_position_nm_per_urad=float(norm[i]*1e9),
                  fixed_drive_at_front_focus=summary(response[np.searchsorted(planes,focus)]),
                  specimen=summary(response[-1]))
    # A minimum at a search boundary cannot establish an interior downstream pivot.
    if 2 <= i < len(stations)-2:
        local = slice(i-2,i+3)
        pivot, = closest_pivot(stations[local], swept[local], columns=(slice(None),))
        result["closest_common_pivot_mm"] = pivot*1000
    write(WORK/(tag+"-planes.json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("ni", type=float)
    parser.add_argument("--mesh", action="store_true")
    parser.add_argument("--planes", action="store_true", help="Measure the two-lens front focus and the fixed-drive downstream pivot")
    parser.add_argument("--reference", help="Reuse identical before measurements from a completed comparison")
    args = parser.parse_args()
    if args.planes:
        downstream_planes(args.tag,args.ni)
        raise SystemExit(0)
    if args.mesh:
        cases = [("m330-32", -14551.5, .0375, False, False),
                 (args.tag+"-32", args.ni, .0375, False, False)]
    else:
        cases = [(tag, ni, step, coarse, scan) for tag, ni in
                 (("m330",-14551.5),(args.tag,args.ni)) for step,coarse,scan in
                 ((.075,False,False),(.0375,False,True),(.01875,False,False),(.0375,True,False))]
    suffix = "-mesh-checks.json" if args.mesh else "-checks.json"
    result = []
    if args.reference:
        baseline = "m330-32" if args.mesh else "m330"
        prior = json.loads((WORK/(args.reference+suffix)).read_text())
        result = [r for r in prior if r["tag"] == baseline]
        expected = [c for c in cases if c[0] == baseline]
        if len(expected) != len(result):
            raise ValueError("Incomplete fixed reference")
        for row, (tag, ni, step, coarse, scan) in zip(result, expected):
            path = field_path(tag+"-pitch030" if coarse else tag)
            if (row["objective_At"] != ni or row["max_step_mm"] != step or ("scan" in row) != scan
                    or row["table_sha256"] != sha256(path.read_bytes()).hexdigest()):
                raise ValueError("The archived reference field or numerical settings differ")
        cases = [c for c in cases if c[0] != baseline]
    for tag, ni, step, coarse, scan in cases:
        result.append(measure(tag,ni,step,coarse=coarse,scan=scan))
        write(WORK/(args.tag+suffix), result)
        cp.get_default_memory_pool().free_all_blocks()
