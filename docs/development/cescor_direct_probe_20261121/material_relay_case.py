"""Replay a commissioning trial from the physical source and a particular coupled FEM state."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from material_relay_fields import HERE,SAVE,WORK,digest


def command(name,*args):
    subprocess.run([sys.executable,str(HERE/f"material_relay_{name}.py"),*map(str,args)],check=True)


def setup(path,label,count,step,bore,pair_seed=None,steering=False,scan=False,mix=None,from_source=False):
    with np.load(path) as data:dc=json.loads(str(data["report"]))
    q=dc["excitation"]
    pair_drive={k:(v if k.startswith(("objective_lens/","mini_condenser/")) else 0.) for k,v in q.items()}
    pair=WORK/("global-"+digest(pair_drive)[:16]+".npz")
    command("bore",path,"dc","--step",bore)
    dur=["--label",label,"--count",count,"--step",step,"--bore-step",bore,"--full"]
    if not pair.exists():
        # Save the real landings before the isolated two-lens solve. Static tracks are independent of this diagnostic.
        command("column",path,*dur)
        if pair_seed:
            command("newton",pair_seed,"--objective",next(v for k,v in pair_drive.items() if k.startswith("objective_lens/")),
                    "--mini",next(v for k,v in pair_drive.items() if k.startswith("mini_condenser/")),"--pair")
        else:command("newton",path,"--pair","--cold")
    command("bore",pair,"dc","--step",bore)
    names=["tl","objective","mini","c3"] if steering else []
    if scan:names.extend(f"ac_{i}_{a}" for i in (0,1) for a in "xy")
    if names:
        command("response",path,"--keys",*names)
        command("bore",path,*names,"--step",bore)
    if steering:
        command("response",pair,"--keys","objective","mini")
        command("bore",pair,"objective","mini","--step",bore)
    scan_args=(["--scan"]+(["--mix",mix] if mix else ["--calibrate"])+(["--source-scan"] if from_source else [])) if scan else []
    command("column",path,"--pair",pair,*dur,*scan_args)
    command("audit",path,"--label",label,"--pair",pair)
    command("figures",label)
    if steering:command("steer",path,pair,"--label",label)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("solution",type=Path);p.add_argument("--label",required=True)
    p.add_argument("--pair-seed",type=Path);p.add_argument("--mix",type=Path)
    p.add_argument("--count",type=int,default=4096);p.add_argument("--step",type=float,default=.075)
    p.add_argument("--bore-step",type=float,default=.5)
    for key in ("steer","scan","source-scan"):p.add_argument("--"+key,action="store_true")
    a=p.parse_args();setup(a.solution,a.label,a.count,a.step,a.bore_step,a.pair_seed,a.steer,a.scan,a.mix,a.source_scan)
