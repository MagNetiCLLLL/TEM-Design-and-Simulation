"""Publish the matched specimen phase space and operating-point evidence for objective-only focus."""

import argparse
from hashlib import sha256
import json
from shutil import copyfile
from types import SimpleNamespace

import numpy as np
from scipy.linalg import polar

from downstream_pivot import inputs, write
from fit_threefold import EVIDENCE
from normal_corrector_search import KEYS, field_path, fit_surface
from refocus_a2 import WORK as OBJECTIVE_WORK, objective_receipt
from relay_transfer import SURFACE, WORK
from render_normal_probe import render
from temsim.physics.beam_statistics import transverse_beam_statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def archive_fields(tag, ni, field_digest):
    solution = WORK / (tag + "-solution.npz")
    with np.load(solution, allow_pickle=False) as material, np.load(field_path(tag), allow_pickle=False) as table:
        raw, sampled = [json.loads(str(f["report"])) for f in (material, table)]
        for key in ("mesh_id", "currents_A", "residual", "material_peaks"):
            if raw[key] != sampled[key]:
                raise ValueError(f"Corrector solution and tracking field disagree on {key}")
    objective = OBJECTIVE_WORK / "column" / f"armco-{abs(ni):.12g}-gap-fine-p8.npz"
    if objective_receipt(ni)["field_digest"] != field_digest:
        raise ValueError("The objective field has changed since the specimen capture")
    files = {"normal-final-corrector-solution.npz": solution,
             "normal-final-objective-field.npz": objective,
             "normal-final-objective-field.json": objective.with_suffix(".json")}
    for name, source in files.items():
        copyfile(source, EVIDENCE / name)
    return {name: sha256((EVIDENCE / name).read_bytes()).hexdigest() for name in files}


def pack(stage, raw, row, field, arrays, current):
    tracker = SimpleNamespace(ids=raw["source_id"], weights=raw["weight"], current=current)
    phase, loss = raw["specimen_phase"], raw["specimen_loss"]
    measurement = fit_surface(phase,loss,tracker)
    np.testing.assert_allclose(measurement["D95_um"]*1000,row["specimen"]["D95_nm"],rtol=1e-11)
    np.testing.assert_allclose(measurement["axis_50nm_nA"]*.001,row["specimen"]["axis_50nm_uA"])
    arrays[stage+"_phase"],arrays[stage+"_loss"] = phase,loss
    arrays[stage+"_loss_component"] = raw["loss_component"]
    offsets, profiles = raw["offsets_nm"], raw["profiles"]
    if profiles.shape[1] != 5 or not np.all(profiles[:,3] == (loss==-1).sum()):
        raise ValueError("A common native through-focus population and both absolute core disks are required")
    for name, val in (("offset_nm",offsets),("profiles",profiles)):
        arrays[stage+"_"+name] = val
    idx = np.where(offsets==0)[0].item()
    maximum = profiles[:,4].max()
    covariance = raw["moments"][:,3] + raw["moments"][:,4]
    z = raw["track_z_m"]*1000
    zeros = np.flatnonzero((z[:-1]>=1350) & (z[1:]<=1505) & (covariance[:-1]<0) & (covariance[1:]>=0))
    with np.load(field) as f:
        desc = json.loads(str(f["report"]))
    result = dict(**measurement, specimen_top_mm=SURFACE*1000, corrector_A=desc["currents_A"],
                  magnetic_material_peak=desc["material_peaks"], material_mesh_mm=desc["material_size_mm"],
                  corrector_table_sha256=sha256(field.read_bytes()).hexdigest(),
                  corrector_residual=desc["residual"], objective_material_sha256=row["objective_material_sha256"],
                  objective_At=row["objective_At"], objective_A=row["objective_At"]/1160.,
                  aperture_edge_mrad=row["rim_azimuth_check"]["angle_mrad_min_median_max"],
                  aperture_edge_azimuths=row["rim_azimuth_check"]["azimuths"],
                  aperture_edge_reached=row["rim_azimuth_check"]["reached_sample"],
                  D95_minimum_bracket_nm=row["D95_minimum_bracket_nm"], RMS_minimum_bracket_nm=row["RMS_minimum_bracket_nm"],
                  D95_minimum_um=row["D95_minimum_value_nm"]*.001,
                  axis_50pm_nA=row["specimen"]["core_50pm_nA"], axis_50nm_packets=int(profiles[idx,4]),
                  through_focus_max_axis_50nm_nA=maximum*tracker.weights[0]*current*1e9,
                  through_focus_core_max_offsets_nm=offsets[profiles[:,4]==maximum].tolist(),
                  through_focus_sample_core_fraction=float(profiles[idx,4]/maximum),
                  source_waists_mm=row["waists_mm"], downstream_radial_waists_mm=[
                      [z[i],z[i+1]] for i in zeros], current_ledger=row["current_ledger"])
    result["original_source_prefixes"] = []
    for count in (8192, 16384, len(loss)):
        live = loss[:count] == -1
        p, w = phase[:count][live], tracker.weights[:count][live]
        stats = transverse_beam_statistics(*p[:, :4].T, weights=w)
        core = np.linalg.norm(p[:, :2], axis=1) <= 50e-9
        result["original_source_prefixes"].append(dict(emitted=count, surviving=int(live.sum()),
            D95_um=2e6*float(stats.radius_95_m), core_packets=int(core.sum()),
            fraction_of_surviving_current_inside_50nm=float(w[core].sum()/w.sum())))
    return result


def publish(tag, alternatives):
    ref, _ = inputs()
    focus = read(WORK/(tag+"-focus.json"))
    labels = ["m330-fixed-0.0375-capture",tag+"-fixed-0.0375-capture"]
    captures = [read(WORK/(name+".json")) for name in labels]
    if (captures[1]["objective_At"] != focus["objective_At"]
            or captures[1]["corrector_sha256"] != focus["corrector_sha256"]):
        raise ValueError("The selected focus and native specimen capture use different fields")
    output = dict(mode="Three-condenser Nanoprobe Normal, 300 kV", specimen_top_mm=SURFACE*1000,
                  source_current_A=15e-6,source_packets=32768,source_digest=ref["source_digest"],
                  pupil_diameter_um=100.,objective_turns=1160,objective_evaluation_increment_At=focus.get("increment_At",.1),
                  optical_model="Classical finite-source Lorentz transport from the frozen gun exit through the installed fields.",
                  fixed_windings=list(zip(ref["winding_owner"],ref["currents_A"])),
                  focus_control="Objective supply only after illumination and relay calibration",
                  corrector_actuators=list(KEYS), max_step_mm=.0375,
                  coefficient_convention="Gradient coefficients in micrometres, scaled to 30 mrad; displacement is minus the wave-aberration gradient. D4 azimuth-RMS equals sqrt(17)/5 times the vector amplitude.",
                  disk_convention="Fixed laboratory axis. D95 is the current-weighted 95% diameter about the centroid.",
                  profile_columns=captures[0]["profile_columns"])
    arrays = {}
    with np.load(WORK/(labels[0]+".npz")) as before, np.load(WORK/(labels[1]+".npz")) as after:
        for k in ("source_id","weight","gun_exit_phase","energy_offset_eV"):
            np.testing.assert_array_equal(before[k],after[k])
            arrays[k] = before[k]
        if np.unique(arrays["source_id"]).size != 32768 or np.any(arrays["weight"]<=0):
            raise ValueError("Expected the full original source")
        for stage, data, row, field in zip(("before","after"),(before,after),captures,
                                           (field_path("m330"),field_path(tag))):
            np.testing.assert_array_equal(ref["currents_A"],row["currents_A"])
            if row["source_digest"] != output["source_digest"]:
                raise ValueError("Source digest changed during the matched experiment")
            output[stage] = pack(stage,data,row,field,arrays,output["source_current_A"])
    both = ("before","after")
    xy = [arrays[k+"_phase"][arrays[k+"_loss"]==-1,:2]*1e6 for k in both]
    pitch = .125
    half = np.ceil(max(float(abs(a).max()) for a in xy)/pitch)
    edges = (np.arange(-half,half+2)-.5)*pitch
    arrays["pixel_edges_um"] = edges
    for k, points in zip(both,xy):
        current = arrays["weight"][arrays[k+"_loss"]==-1]*output["source_current_A"]*1e9
        bins = np.histogram2d(*points.T,bins=(edges,edges),weights=current)[0]
        np.testing.assert_allclose(bins.sum(),current.sum(),rtol=1e-12)
        arrays[k+"_nA_per_pixel"] = bins
    output["changed_survivor_membership_count"] = int(np.sum((arrays["before_loss"]==-1)!=(arrays["after_loss"]==-1)))
    checks = read(WORK/(tag+"-checks.json"))
    mesh = read(WORK/(tag+"-mesh-checks.json"))
    if len(checks) != 8 or len(mesh) != 2:
        raise ValueError("A complete before/after mesh, field pitch and integration-step study is required")
    for r in checks + mesh:
        stage = "before" if r["tag"].removesuffix("-32") == "m330" else "after"
        if (r["objective_At"] != output[stage]["objective_At"] or r["source_digest"] != ref["source_digest"]
                or r["corrector_A"] != output[stage]["corrector_A"]):
            raise ValueError("Field controls or original source changed across a numerical check")
    output["numerical_checks"] = checks+mesh
    native = [r for r in checks if "scan" in r]
    if [r["tag"] for r in native] != ["m330",tag]:
        raise ValueError("Matched before and after scan-response measurements required")
    for stage, r in zip(both, native):
        np.testing.assert_allclose(output[stage]["D95_um"],r["D95_um"],rtol=1e-7)
        np.testing.assert_allclose(output[stage]["axis_50nm_nA"],r["axis_50nm_nA"],rtol=1e-12)
    J0,J1 = [np.array(r["scan"]["position_nm_per_urad"]) for r in native]
    rotation, stretch = polar(J1@np.linalg.inv(J0))
    output["relative_scan_scale_change_percent"] = (100*(np.linalg.eigvalsh(stretch)-1)).tolist()
    output["relative_scan_rotation_deg"] = float(np.rad2deg(np.arctan2(rotation[1,0],rotation[0,0])))
    output["objective_focus_search"] = focus
    output["field_archive_sha256"] = archive_fields(tag,focus["objective_At"],native[1]["objective_field_digest"])
    for stage, name in zip(both, ("m330", tag)):
        fine_search = read(WORK / (name+"-32-focus.json"))
        ni = fine_search["objective_At"]
        fine_row, = [r for r in fine_search["trials"] if r["objective_At"] == ni]
        finer, = [r for r in mesh if r["tag"] == name+"-32"]
        if fine_search["corrector_sha256"] != finer["table_sha256"]:
            raise ValueError("Independent objective refocus used a different corrector field")
        output[stage]["finer_mesh_refocus"] = dict(fine_row, objective_A=ni/1160.,
            material_mesh_mm=finer["material_mesh_mm"], table_sha256=finer["table_sha256"])
        planes = read(WORK / (name+"-planes.json"))
        if planes["objective_At"] != output[stage]["objective_At"] or planes["source_digest"] != ref["source_digest"]:
            raise ValueError("Downstream calibration check used another operating point")
        output[stage]["downstream_calibration"] = planes
    output["exact_alternatives"] = {}
    for other in alternatives:
        f = read(WORK/(other+"-focus.json"))
        row = next(v for v in f["trials"] if v["objective_At"]==f["objective_At"])
        with np.load(field_path(other)) as data:
            drives = json.loads(str(data["report"]))["currents_A"]
        optical = read(WORK/(other+"-optical.json"))
        if optical["objective_At"] != f["objective_At"] or optical["table_sha256"] != f["corrector_sha256"]:
            raise ValueError("The alternative pupil measurement uses different controls")
        output["exact_alternatives"][other] = dict(row,corrector_A=drives,table_sha256=f["corrector_sha256"],
                aperture_edge_mrad=optical["edge_mrad"], aperture_edge_reached=optical["edge_reached"])
        finer = WORK / (other+"-mesh-checks.json")
        if finer.is_file():
            measured = [r for r in read(finer) if r["tag"] == other+"-32"]
            if len(measured) != 1 or measured[0]["objective_At"] != row["objective_At"]:
                raise ValueError("The alternative material-grid check uses different controls")
            output["exact_alternatives"][other]["finer_material_check"] = measured[0]
    for name in ("normal-corrector-screen","normal-compact","normal-trim"):
        d = read(WORK/(name+".json")); settings = np.array([r["settings"] for r in d["trials"]])
        output.setdefault("search_domains",{})[name] = dict(
            source_count=d["emitted"], proposals=len(settings), coordinate_min=settings.min(axis=0).tolist(),
            coordinate_max=settings.max(axis=0).tolist(), controls=["objective_At",*KEYS],
            method=d["scope"])
    path = EVIDENCE/"normal-objective-landings.npz"
    np.savez_compressed(path,**arrays)
    output["landing_archive_sha256"] = sha256(path.read_bytes()).hexdigest()
    write(EVIDENCE/"normal-objective-probe.json",output)
    render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("--alternatives",nargs="*",default=[])
    args = parser.parse_args()
    publish(args.tag,args.alternatives)
