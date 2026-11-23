"""Profile fixed-surface D95 during an objective-only focus sweep and measure the local caustic."""
import json
import re
import sys

import cupy as cp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from condenser_pair_column import specimen
from condenser_source_fields import write
from material_relay_fields import SAVE,WORK
from material_relay_tracking import Trial,column_coefficients
from material_relay_run import paths
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import advance
from trace_normal_joint_field import beam_moments


def profile(tl=-83.5,label="relay-candidate"):
    folder=SAVE/label
    state=json.loads((folder/"tracking.json").read_text())
    gun=state["gun_source"]
    _,table=paths(tl,24)
    m=Trial(RefinedObjectiveFields(.5),2048,.12,gun,table)
    mm=np.r_[np.arange(1510,1599.,.5),1599.,1599.1,1599.1999975]
    ids=m.indices(mm*.001)
    maps={}
    prefix="circuit-ampere-turns-condenser_lens_36.710114-mini_condenser239.91265-all-"
    for p in sorted(WORK.glob(prefix+"*.npz"),key=lambda p:"-0.5mm" in p.name):
        amps=float(re.search(r"all-([-\d.]+)A",p.name)[1])
        maps[amps]=p
    amps=sorted(maps)
    coeff=PchipInterpolator(amps,[column_coefficients(maps[a],m.m.nodes).get()[:,:8] for a in amps],axis=0)
    data={}
    def trial(obj):
        obj=float(obj)
        if obj in data:return data[obj]
        field=np.zeros((len(m.m.nodes),20));field[:,:8]=coeff(obj)
        values=advance(m.t,cp.asarray(field),m.t.input,m.en,m.phi0,m.loss,0,ids)
        rms=[];d95=[];cov=[]
        for p,loss in values:
            live=loss==-1
            s=specimen(p[live],m.t.weights[live],m.t.current)
            mom=beam_moments(p,live,m.t.weights)
            rms.append(s["RMS_radius_um"]);d95.append(s["D95_um"]);cov.append(mom[1]+mom[2])
        entry=dict(objective=obj,D95=d95[-1],rms=rms[-1],cov=cov[-1],curvature=(cov[-1]-cov[-2])/.0999975,
                   smallest_axial_D95_mm=float(mm[np.argmin(d95)]),smallest_axial_D95=float(min(d95)))
        data[obj]=entry
        return entry
    grid=np.arange(amps[0]+.2,amps[-1],1.)
    for a in grid:trial(a)
    vals=[data[a]["D95"] for a in grid]
    roots=[]
    for i in np.flatnonzero((np.array(vals[1:-1])<vals[:-2])&(np.array(vals[1:-1])<vals[2:]))+1:
        r=minimize_scalar(lambda q:trial(q)["D95"],method="bounded",bounds=grid[[i-1,i+1]],options={"xatol":1e-4})
        roots.append(trial(r.x))
    print("Objective-only specimen D95 minima",roots,flush=True)
    write(folder/"objective-profile.json",dict(source=gun,basis="Search using interpolated solved fields; verify by full nonlinear re-solve",
          TL12_A=tl,originals=2048,minima=roots,scan=sorted(data.values(),key=lambda q:q["objective"])))


if __name__=="__main__":
    profile(*([float(sys.argv[1]),sys.argv[2]] if len(sys.argv)>2 else []))
