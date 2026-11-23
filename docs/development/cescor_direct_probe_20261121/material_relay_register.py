"""Verify the finite-source waist correction with newly solved fields and a frozen scan matrix."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import CubicSpline

from condenser_source_fields import write
from material_relay_case import command
from material_relay_feedback import matched_fields
from material_relay_fields import SAVE, WORK
from material_relay_response import groups


def verify(reference="relay-final", other=None, label="relay-registered"):
    base=SAVE/reference
    record=json.loads((base/"tracking.json").read_text())
    target=SAVE/label
    target.mkdir(exist_ok=True)
    initial=[WORK/record[k]["source"] for k in ("field","pair")]
    anchors=[WORK/n for n in ("global-bc4e4dc45528d9a5.npz","global-de970c680c3e33b4.npz")]
    bias=record["field"]["bias"];keys=groups(bias)
    recipe={}
    if other:
        end=SAVE/other
        other_record=json.loads((end/"tracking.json").read_text())
        other_bias=other_record["field"]["bias"]
        free=keys["tl"]+keys["objective"]
        if any(bias[k]!=other_bias[k] for k in bias if k not in free):
            raise ValueError("Both bracket endpoints must share every fixed current")
        slopes=[]
        for path in (base,end):
            plane=json.loads((path/"plane-check.json").read_text())["full_population"]
            if plane["original_particles"]!=32768:
                raise ValueError("The sign bracket needs both full source populations")
            slopes.append(plane["front_half_variance_slopes_nm2_per_mm"][0])
        if slopes[0]*slopes[1]>=0:
            raise ValueError("The measured full-source covariance does not bracket zero")
        fraction=slopes[0]/(slopes[0]-slopes[1])
        delta=fraction*np.array([other_bias[keys[k][0]]-bias[keys[k][0]] for k in ("tl","objective")])
        recipe=dict(second_label=other,measured_slopes_nm2_per_mm=slopes,interpolation_fraction=fraction)
    else:
        correction=json.loads((base/"directional-response.json").read_text())
        focus=json.loads((base/"surface-focus.json").read_text())
        with np.load(base/"column-and-landings.npz") as rays:
            z=rays["z_mm"][-20:]
            cov=1e-9*rays["stats"][-20:,1:3].sum(axis=1)
            axial_adjustment=CubicSpline(z,cov)(z[-1],1)*focus["D95_min_offset_um"]*.001
        state=np.array(correction["current_state"])
        jac=np.array(correction["derivative_per_A"])[[0,3],:2]
        delta=np.linalg.solve(jac,[-state[0],axial_adjustment])
    tl=bias[keys["tl"][0]]+delta[0]
    obj=bias[keys["objective"][0]]+delta[1]
    pre=matched_fields(*initial,{"tl":tl},anchors)
    final=matched_fields(*initial,{"tl":tl,"objective":obj},anchors)
    write(target/"adjustment.json",dict(from_label=base.name,TL12_A=tl,objective_A=obj,
          delta_A=delta.tolist(),pre=[p.name for p in pre],final=[p.name for p in final],**recipe))
    mix=SAVE/"relay-before-final-focus"/"scan-calibration.json"
    channels=("ac_0_x","ac_0_y","ac_1_x","ac_1_y")
    for f in dict.fromkeys([*final,*pre]):
        command("bore",f,"dc","--step",.5)

    def beam():
        command("column",final[0],"--pair",final[1],"--label",target.name,"--count",32768,
                "--step",.075,"--bore-step",.5,"--full")
        command("focus_check",final[0],"--label",target.name)
        command("plane_check",target.name)
        command("round_check",target.name)
        command("audit",final[0],"--pair",final[1],"--label",target.name)
        command("field_plot",target.name)

    pre_label="relay-registration-reference" if label=="relay-registered" else label+"-pre"
    scan_label=label+"-scan"
    def scan():
        for (f,p),scan_name,extra in ((pre,pre_label,[]),(final,scan_label,["--source-scan"])):
            command("response",f,"--keys",*channels)
            command("bore",f,*channels,"--step",.5)
            command("column",f,"--pair",p,"--label",scan_name,"--count",1024,"--step",.075,
                    "--bore-step",.5,"--mix",mix,"--scan",*extra)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(fn) for fn in (beam,scan)]
        for task in tasks:task.result()
    tracked=json.loads((target/"tracking.json").read_text())
    signed=json.loads((SAVE/scan_label/"tracking.json").read_text())
    if tracked["field"]["solution_sha256"]!=signed["field"]["solution_sha256"]:
        raise ValueError("The final signed and static field identities differ")
    tracked.update(pivot=signed["pivot"],pivot_source_particles=signed["count"],
                   pivot_record=f"../{scan_label}/tracking.json")
    write(target/"tracking.json",tracked)
    command("figures",target.name)
    command("archive",target.name,"--scan-label",scan_label,"--pre-label",pre_label)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference",default="relay-final")
    parser.add_argument("--other")
    parser.add_argument("--label",default="relay-registered")
    args=parser.parse_args()
    verify(args.reference,args.other,args.label)
