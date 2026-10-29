"""Read-only native profile comparison of explicit column-energy diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_column_energy_diagnostic import file_hash, array_hash
from measure_probe_waist import analyze_scan, observed_minimum
from solve_joint_probe import atomic_json

SCHEMA = "temsim-controlled-energy-profile-comparison-v1"


def load(path):
    path = Path(path).resolve()
    report = json.loads((path/"report.json").read_text(encoding="utf-8"))
    if report.get("execution_status", report.get("status")) != "COMPLETE":
        raise ValueError("Only completed observations may be compared")
    digest = file_hash(path/"particles.npz")
    if "files_sha256" in report:
        expected = report["files_sha256"]["particles.npz"]
    else:
        expected = json.loads((path/"checksums.json").read_text())["particles.npz"]
    if digest != expected:
        raise ValueError("Particle checksum differs from the completed receipt")
    with np.load(path/"particles.npz",allow_pickle=False) as raw:
        arrays = {key:raw[key].copy() for key in raw.files}
    return path, report, arrays, digest


def summarize(data, planes, surface, current):
    path, report, arrays, digest = data
    scan = analyze_scan(arrays,planes,surface_z_mm=surface,source_current_a=current)
    rows = []
    watched = np.flatnonzero(arrays["source_id"] == 306)
    for row in scan["planes"]:
        row = {key:value for key,value in row.items() if key != "source_ids"}
        i = int(np.flatnonzero(arrays["z_mm"] == row["z_mm"])[0])
        row["id306_centered_radius_nm"] = None
        if len(watched) == 1 and arrays["alive"][i,watched[0]]:
            j = int(watched[0])
            row["id306_centered_radius_nm"] = float(np.hypot(
                arrays["x_m"][i,j]-row["centroid_x_m"], arrays["y_m"][i,j]-row["centroid_y_m"])*1e9)
        rows.append(row)
    return dict(path=str(path), particles_sha256=digest, report_sha256=file_hash(path/"report.json"),
        source_id_sha256=array_hash(arrays["source_id"]), weights_sha256=array_hash(arrays["weights"]),
        original_source_points=len(arrays["source_id"]), original_source_current_a=current,
        surface=next(row for row in rows if row["z_mm"] == surface),
        d95_minimum=scan["d95_waist"],
        d90_minimum=observed_minimum(planes,[row["d90_nm"] for row in rows],surface),
        positive_current_membership_stable=scan["positive_current_membership_stable"], planes=rows)


def paired(left, right, surface):
    a,b = left[2],right[2]
    for key in ("source_id","weights"):
        if not np.array_equal(a[key],b[key]):
            raise ValueError(f"Paired comparison requires identical complete {key}")
    planes=np.intersect1d(a["z_mm"],b["z_mm"])
    if surface not in planes:
        raise ValueError("Paired comparison lacks the exact sample surface")
    rows=[]
    for z in planes:
        ai=int(np.flatnonzero(a["z_mm"] == z)[0]);bi=int(np.flatnonzero(b["z_mm"] == z)[0])
        both=a["alive"][ai] & b["alive"][bi] & (a["weights"] > 0)
        delta=np.hypot(a["x_m"][ai,both]-b["x_m"][bi,both],a["y_m"][ai,both]-b["y_m"][bi,both])*1e9
        if not np.isfinite(delta).all():
            raise ValueError("Paired live coordinates must be finite")
        rows.append(dict(z_mm=float(z),offset_nm=float((z-surface)*1e6),
            paired_live_points=int(both.sum()),changed_survival_points=int(np.count_nonzero(a["alive"][ai] != b["alive"][bi])),
            max_position_difference_nm=float(delta.max()) if len(delta) else None,
            max_difference_source_id=int(a["source_id"][both][delta.argmax()]) if len(delta) else None))
    finite=[r for r in rows if r["max_position_difference_nm"] is not None]
    return dict(coordinates="Original global x,y, without centroid subtraction or rescaling", exact_shared_planes=len(rows),
        same_complete_ids_and_absolute_weights=True,surface=next(r for r in rows if r["z_mm"]==surface),
        largest_saved_plane_difference=max(finite,key=lambda r:r["max_position_difference_nm"]) if finite else None,
        planes=rows)


def compare(donor,intervention,output,mono=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    base=load(donor);changed=load(intervention)
    report=base[1]
    planes=np.asarray(report["parameters"]["requested_planes_mm"])
    surface=float(report["analysis"]["surface"]["z_mm"])
    current=float(report["receipt"]["source_current_a"])
    if changed[1].get("donor_files_sha256",{}).get("particles.npz") != base[3]:
        raise ValueError("Intervention is not bound to this exact donor")
    if changed[1].get("full_original_population") is not True or changed[1]["source_current_a"] != current:
        raise ValueError("Intervention lacks the full original current population")
    variants=dict(poly_baseline=summarize(base,planes,surface,current),
        poly_phase_changed_energy=summarize(changed,planes,surface,current))
    pairs=dict(poly_to_controlled_injection=paired(base,changed,surface))
    gun_comparison=None
    if mono is not None:
        fullmono=load(mono)
        if fullmono[1]["receipt"]["source_current_a"] != current:
            raise ValueError("Full-mono original current differs")
        variants["full_mono_gun_and_column"]=summarize(fullmono,planes,surface,current)
        pairs["poly_to_full_mono"]=paired(base,fullmono,surface)
        pairs["controlled_injection_to_full_mono"]=paired(changed,fullmono,surface)
        a,b=base[2],fullmono[2]
        source_checks={key:bool(np.array_equal(a[key],b[key])) for key in
            ("emission_position_m","emission_direction","source_id","weights","gun_exit_alive")}
        live=a["gun_exit_alive"] & b["gun_exit_alive"]
        endpoint={key:dict(max_abs_difference=float(np.max(np.abs(a[key][live]-b[key][live]))),
            poly_sha256=array_hash(a[key]),mono_sha256=array_hash(b[key])) for key in
            ("gun_exit_x_m","gun_exit_y_m","gun_exit_tx_rad","gun_exit_ty_rad","gun_exit_flight_time_s","gun_exit_kinetic_energy_ev")}
        watched=[]
        for label,arrays in (("poly",a),("full_mono",b)):
            j=int(np.flatnonzero(arrays["source_id"]==306)[0])
            watched.append(dict(case=label,source_id=306,kinetic_energy_ev=float(arrays["gun_exit_kinetic_energy_ev"][j]),
                radius_um=float(np.hypot(arrays["gun_exit_x_m"][j],arrays["gun_exit_y_m"][j])*1e6),
                slope_norm_mrad=float(np.hypot(arrays["gun_exit_tx_rad"][j],arrays["gun_exit_ty_rad"][j])*1e3)))
        gun_comparison=dict(original_source_exact_checks=source_checks,exit_fields=endpoint,id306=watched,
            interpretation="Changing emitted energy already changes the executed low-energy gun phase before the column; this comparison does not isolate its magnitude from downstream interactions.")
    result=dict(schema=SCHEMA,qualification="CONTROLLED_DIAGNOSTIC_NOT_QUALIFIED",
        scope="Native current-weighted quantiles on exact saved planes; no interpolation, refocusing or source filtering",
        interpretation="Controlled injection changes entry energy at fixed original exit phase; full mono also changes gun-generated phase. No quadrature subtraction or additive attribution of D95 is valid.",
        no_size_acceptance_threshold=True,variants=variants,paired_position_comparisons=pairs,
        poly_to_full_mono_gun_exit=gun_comparison,
        analysis_script_sha256=file_hash(__file__))
    atomic_json(output/"comparison.json",result)
    lines=["# Controlled column-energy comparison","",
        result["scope"]+".","",result["interpretation"],"",
        "| Case | Surface D90 (nm) | Surface D95 (nm) | Smallest observed D95 (nm) | D95 minimum offset (nm) | Alpha95 (mrad) | Current (uA) | ID306 radius (nm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key,v in variants.items():
        s=v["surface"];m=v["d95_minimum"]
        lines.append(f"| {key} | {s['d90_nm']:.9g} | {s['d95_nm']:.9g} | {m['minimum']:.9g} | {m['representative_offset_nm']:.9g} | {s['alpha95_mrad']:.9g} | {s['current_a']*1e6:.9g} | {s['id306_centered_radius_nm']:.9g} |")
    lines.extend(["","All minima are discrete observed minima within the saved window, not exact continuous argmins. RMS is auxiliary only. There is no probe-size acceptance threshold in this diagnostic.",""])
    for key,v in pairs.items():
        s=v["surface"]
        lines.append(f"- {key}: exact shared planes = {v['exact_shared_planes']}; maximum raw paired displacement at the surface = {s['max_position_difference_nm']:.9g} nm (source ID {s['max_difference_source_id']}); changed survival at surface = {s['changed_survival_points']}.")
    lines.extend(["","Raw particle and report SHA256 digests, every native through-focus row, and every exact shared checkpoint displacement are retained in comparison.json.",""])
    (output/"comparison.md").write_text("\n".join(lines),encoding="utf-8")
    (output/"analysis-source.py").write_bytes(Path(__file__).read_bytes())
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor",type=Path,required=True)
    parser.add_argument("--intervention",type=Path,required=True)
    parser.add_argument("--mono",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compare(args.donor,args.intervention,args.output,args.mono)
