"""Archive the fields, calibration and original-current measurements of a verified trial."""
import argparse
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
from scipy.interpolate import CubicSpline

from condenser_source_fields import write
from material_relay_fields import SAVE,WORK,parent,sha
from material_relay_response import groups


def archive(label,scan_label=None,pre_label=None,pre_scan_label=None):
    folder=SAVE/label
    r=json.loads((folder/"tracking.json").read_text())
    audit=json.loads((folder/"field-audit.json").read_text())
    focus=json.loads((folder/"surface-focus.json").read_text())
    scan_record=json.loads((SAVE/(scan_label or label)/"tracking.json").read_text())
    if scan_record["field"]["solution_sha256"]!=r["field"]["solution_sha256"]:
        raise ValueError("Scan verification uses a different excitation")
    scan=scan_record["pivot"]
    front=r["isolated_pair_fronts"][0]["z_mm"]
    pivot=min(scan["candidates"],key=lambda x:abs(x["z_mm"]-front))
    arrays={}
    fields={}
    for title,receipt in (("static",r["field"]),("isolated_pair",r["pair"])):
        full=WORK/receipt["source"]
        if sha(full)!=receipt["solution_sha256"]:raise ValueError("Material solution identity changed")
        for channel in (["dc"] if title=="isolated_pair" else ["dc","ac_0_x","ac_0_y","ac_1_x","ac_1_y"]):
            local_step=receipt["step_mm"] if channel=="dc" else scan_record["field"]["step_mm"]
            file=WORK/f"{full.stem}-{channel}-bore-{local_step:g}mm.npz"
            key=title if channel=="dc" else channel
            with np.load(file) as data:
                check=json.loads(str(data["receipt"]))
                if check["solution_sha256"]!=sha(full):raise ValueError("Field does not match operating point")
                arrays[key+"_z_m"]=data["z"];arrays[key+"_coef_T"]=data["coef"]
                fields[key]=dict(sha256=sha(file),file=file.name,receipt=check)
    np.savez_compressed(folder/"fields-for-replay.npz",**arrays)
    copyfile(Path(r["gun_source"]),folder/"effective-source-and-exit.npz")
    with np.load(folder/"column-and-landings.npz") as data:
        z=data["z_mm"];stats=data["stats"]
        source_ids,weights,loss=data["source_ids"],data["weights"],data["loss"]
        if len(np.unique(source_ids))!=r["count"] or np.min(weights)<=0:raise ValueError("Invalid source ledger")
        transmitted=float(data["source_A"]*weights[loss==-1].sum())
        np.testing.assert_allclose(transmitted*1e6,r["surface"]["specimen_uA"],rtol=2e-12)
    waist=[z for z in r["waists_mm"] if 1420<z<1512.5]
    radius=CubicSpline(z,stats[:,0])
    extent={f"{offset:+g}":float(radius(front+offset)*1000) for offset in (-5,-1,0,1,5)}
    coil=r["field"]["bias"];keys=groups(coil)
    pre=None
    if pre_label:
        initial=json.loads((SAVE/pre_label/"tracking.json").read_text())
        changed=[k for k,v in coil.items() if initial["field"]["bias"][k]!=v]
        if set(changed)-set(keys["objective"]):raise ValueError("Final adjustment changed another winding")
        initial_scan = initial
        if pre_scan_label:
            initial_scan = json.loads((SAVE/pre_scan_label/"tracking.json").read_text())
            if initial_scan["field"]["solution_sha256"] != initial["field"]["solution_sha256"]:
                raise ValueError("The pre-focus scan calibration uses a different excitation")
        if not np.array_equal(scan["mix"],initial_scan["pivot"]["mix"]):
            raise ValueError("Final validation changed the scan calibration")
        pre=dict(label=pre_label,currents_A=initial["field"]["bias"],changed_in_final_step=changed,
                 focus_before_mm=initial["isolated_pair_fronts"],waists_before_mm=initial["waists_mm"],
                 scan_label=pre_scan_label or pre_label)
    record=dict(mode="Three-condenser Nanoprobe Normal",source="Effective source at z=0",source_signature=parent()["source_signature"],
        C2_aperture_diameter_um=100.,source_particles=r["count"],scan_particles=scan_record["count"],
        source_current_A=float(transmitted/(weights[loss==-1].sum())),specimen_current_A=transmitted,
        specimen_upper_surface_mm=r["surface_z_mm"],specimen=r["surface"],surface_focus=focus,
        signed_currents_A=coil,zero_scan_A=r["AC_command_static"],optics=pre,
        measured_pivot=pivot,isolated_pair_fronts=r["isolated_pair_fronts"],operational_fronts=r["operational_fronts"],
        radial_waists_mm=r["waists_mm"],principal_waists_mm=r["principal_waists_mm"],
        downstream_waist_minus_front_mm=(min(waist,key=lambda a:abs(a-front))-front if waist else None),
        front_offset_mm_to_rms_radius_nm=extent,field_audit_sha256=sha(folder/"field-audit.json"),
        geometry_sha256=audit["assembly_sha256"],mesh_id=r["field"]["mesh_id"],fields=fields,
        saved_fields_sha256=sha(folder/"fields-for-replay.npz"),source_exit_sha256=sha(Path(r["gun_source"])),
        original_landings_sha256=sha(folder/"column-and-landings.npz"))
    if scan_record["count"] < r["count"]:
        with np.load(r["gun_source"]) as full, np.load(scan_record["gun_source"]) as smaller:
            selected=np.searchsorted(full["ids"],smaller["ids"])
            np.testing.assert_array_equal(full["ids"][selected],smaller["ids"])
            np.testing.assert_array_equal(full["launch"][selected],smaller["launch"])
            change=full["phase"][selected]-smaller["phase"]
            pos_pm=1e12*np.linalg.norm(change[:,:2],axis=1)
            slope_nrad=1e9*np.linalg.norm(change[:,2:4],axis=1)
            record["same_field_gun_sampling"]=dict(
                source_counts=[int(r["count"]),int(scan_record["count"])],
                matched_original_ids=len(selected),
                comparison="Zero scan in the identical DC field, with the same launch coordinates",
                median_exit_position_difference_pm=float(np.median(pos_pm)),
                maximum_exit_position_difference_pm=float(pos_pm.max()),
                median_exit_slope_difference_nrad=float(np.median(slope_nrad)),
                maximum_exit_slope_difference_nrad=float(slope_nrad.max()))
    if pre_label:
        proof=json.loads((WORK/"winding-load-projections.json").read_text())
        if (proof["geometry_sha256"]!=audit["assembly_sha256"] or proof["mesh_id"]!=r["field"]["mesh_id"]
                or set(proof["channels"])!=set(coil)):
            raise ValueError("The independent winding-load check refers to a different assembly")
        for name,checked in proof["channels"].items():
            array=WORK/"loads"/proof["mesh_id"]/(name.replace("/","__")+".npy")
            if checked["stored_load_sha256"]!=sha(array) or not checked["maximum_relative_to_peak"]<.005:
                raise ValueError(f"Stored loading does not reproduce direct Biot-Savart for {name}")
        for filename in ("winding-load-projections.json","particle-input-independence.json"):
            copyfile(WORK/filename,folder/filename)
        check_path=folder/"axisymmetric-refinement.json"
        if check_path.is_file():
            mesh_check=json.loads(check_path.read_text())
            if (abs(mesh_check["all_material_3d_reported_mm"]-front)>1e-8
                    or mesh_check.get("pair_sha256",r["pair"]["solution_sha256"])!=r["pair"]["solution_sha256"]):
                raise ValueError("The magnetic refinement used a different focal plane")
            record["round_geometry_refinement_sha256"]=sha(check_path)
        record["winding_projection_checks_sha256"]=sha(folder/"winding-load-projections.json")
        record["particle_input_check_sha256"]=sha(folder/"particle-input-independence.json")
    adjustment=folder/"adjustment.json"
    if adjustment.is_file():
        source_trial=json.loads(adjustment.read_text())["from_label"]
        other=json.loads((SAVE/source_trial/"tracking.json").read_text())
        with np.load(Path(other["gun_source"])) as prev,np.load(Path(r["gun_source"])) as current:
            for key in ("launch","ids","weight","emitted_A"):
                np.testing.assert_array_equal(prev[key],current[key])
            if prev["signature"].item()==current["signature"].item():
                raise ValueError("Different field excitations selected the same gun handoff")
            record["gun_relaunch_comparison"]=dict(reference=source_trial,matching_launch_ids_and_weights=True,
                 maximum_exit_position_difference_m=float(np.linalg.norm(current["phase"][:,:2]-prev["phase"][:,:2],axis=1).max()),
                 maximum_exit_slope_difference_rad=float(np.linalg.norm(current["phase"][:,2:4]-prev["phase"][:,2:4],axis=1).max()))
    write(folder/"working-point.json",record)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("label");p.add_argument("--scan-label");p.add_argument("--pre-label")
    p.add_argument("--pre-scan-label")
    a=p.parse_args();archive(a.label,a.scan_label,a.pre_label,a.pre_scan_label)
