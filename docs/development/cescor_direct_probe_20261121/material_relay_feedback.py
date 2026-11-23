"""Iterate real common-circuit fields and finite-source tracking on the selected relay branch."""
import argparse
import json
from pathlib import Path

import numpy as np

from material_relay_case import command
from material_relay_fields import SAVE, WORK, digest
from material_relay_response import groups
from material_relay_steer import fit


def solve_bias(seed, changes, pair=False):
    with np.load(seed) as old:
        currents = json.loads(str(old["report"]))["excitation"]
    for name, keys in groups(currents).items():
        if name in changes:
            for key in keys:
                currents[key] = changes[name]
    if pair:
        currents = {k: v if k.startswith(("objective_lens/", "mini_condenser/")) else 0. for k,v in currents.items()}
    path = WORK / ("global-" + digest(currents)[:16] + ".npz")
    if not path.exists():
        args = [item for k,v in changes.items() for item in ("--" + k, v)]
        command("newton", seed, *args, *(["--pair"] if pair else []))
    return path


def track(full,pair,label,n,step,bore):
    for field in (full,pair):
        command("bore",field,"dc","--step",bore)
    command("column",full,"--pair",pair,"--label",label,"--count",n,"--step",step,"--bore-step",bore,"--full")

def matched_fields(full,pair,changes,anchors):
    with np.load(full) as data:
        q=json.loads(str(data["report"]))["excitation"]
    for name,keys in groups(q).items():
        if name in changes:
            for key in keys:q[key]=changes[name]
    pair_q={k:v if k.startswith(("objective_lens/","mini_condenser/")) else 0. for k,v in q.items()}
    result=[WORK/("global-"+digest(v)[:16]+".npz") for v in (q,pair_q)]
    if not all(path.exists() for path in result):
        args=[item for k,v in changes.items() for item in ("--"+k,v)]
        command("newton",full,*args,"--pair-seed",pair,"--predictor",anchors[0],"--pair-predictor",anchors[1])
    return result


def iterate(label,prefix,rounds):
    for i in range(rounds):
        folder=SAVE/label
        r=json.loads((folder/"tracking.json").read_text())
        full,pair=(WORK/r[key]["source"] for key in ("field","pair"))
        if i==0:
            anchor_full,anchor_pair=full,pair
            for field,modes in ((full,("tl","objective","mini","c3")),(pair,("objective","mini"))):
                command("response",field,"--keys",*modes)
                command("bore",field,*modes,"--step",r["field"]["step_mm"])
        s=fit(full,pair,folder,anchor_full,anchor_pair)
        front=s["current_true_front_mm"]
        measured=[z for z in r["waists_mm"] if 1420<z<1512.5]
        split=[[z for z in axis if 1420<z<1512.5] for axis in r["principal_waists_mm"]]
        print("Measured waists, isolated front and objective-only finish",measured,split,front,s["objective_only_A"],flush=True)
        if (measured and abs(measured[-1]-front)<.12 and s["objective_only_A"] is not None
                and abs(s["objective_only_A"]-s["base"]["objective"])<.002):
            return label
        changes=s["candidates"][0]["currents"]
        full1,pair1=matched_fields(full,pair,changes,[anchor_full,anchor_pair])
        label=f"{prefix}-{i+1}"
        print("Finite-source verification",label,flush=True)
        track(full1,pair1,label,r["count"],r["step_mm"],r["field"]["step_mm"])
    return label


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("start");p.add_argument("--prefix",default="relay-pass");p.add_argument("--rounds",type=int,default=3)
    a=p.parse_args()
    print("Last verified working point",iterate(a.start,a.prefix,a.rounds))
