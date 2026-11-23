"""Resolve the local controllability of a physical two-dimensional waist with the round supplies."""
import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

from condenser_source_fields import write
from material_relay_branches import fast
from material_relay_column import Column, Field, source_and_electrodes
from material_relay_fields import WORK, SAVE
from relay_transfer import advance


def evaluate(label, anchor, pair_anchor):
    folder=SAVE/label
    record=json.loads((folder/"tracking.json").read_text())
    base=Field(WORK/record["field"]["source"])
    pair=Field(WORK/record["pair"]["source"])
    modes=("tl","objective","mini","c3")
    step=np.array([.025,.0005,.025,.00015])
    model=Column(source_and_electrodes(),base,Path(record["gun_source"]),record["count"],record["step_mm"])
    delta=[Field(anchor,key) for key in modes]
    pair_delta=[None,Field(pair_anchor,"objective"),Field(pair_anchor,"mini"),None]
    tables=[cp.asarray(d.table()) for d in delta]
    grid=model.t.tables[0]
    z=np.linspace(.45,record["surface_z_mm"]*.001,35001)
    target=record["isolated_pair_fronts"][0]["z_mm"]*.001

    def focal_plane(dq):
        b=pair.c[:,0,2].copy()
        for v,d in zip(dq,pair_delta,strict=True):
            if d is not None:b+=v*d.c[:,0,2]
        a,b,c,d=fast(z,CubicSpline(pair.z,b)(z)).T
        D=-c[-1]*b+d[-1]*a
        ids=np.flatnonzero((z[:-1]>1.42)&(z[:-1]<1.51)&(D[:-1]*D[1:]<=0))
        i=ids[np.argmin(abs(z[ids]-target))]
        return z[i]-(z[i+1]-z[i])*D[i]/(D[i+1]-D[i])

    zero=np.zeros(len(modes))
    offset=target-focal_plane(zero)
    def state(dq):
        ff=focal_plane(dq)+offset
        model.t.tables[0]=grid.copy()
        bz=base.c[:,0,2].copy()
        for v,d,t in zip(dq,delta,tables,strict=True):
            if v:
                model.t.tables[0]+=v*t
                bz+=v*d.c[:,0,2]
        idx=model.indices([ff])[0]
        ids=np.r_[idx-3:idx+4,len(model.m.z)-1]
        values=[]
        for phase,loss in advance(model.t,model.full,model.t.input,model.en,model.phi0,model.loss,0,ids):
            if np.any(loss!=-1):raise ValueError("A field perturbation changed the transmitted cohort")
            p=phase[:,:4]-np.average(phase[:,:4],axis=0,weights=model.t.weights)
            g=(p[:,:2].T*model.t.weights)@p[:,2:]/model.t.weights.sum()
            values.append([np.trace(g),g[0,0]-g[1,1],g[1,0]+g[0,1]])
        va=np.asarray(values)
        return np.r_[CubicSpline(model.m.z[ids[:-1]],va[:-1])(ff),va[-1,0],CubicSpline(base.z,bz)(ff)]

    y=state(zero)
    slopes=[]
    for i,h in enumerate(step):
        dq=zero.copy();dq[i]=h
        slopes.append((state(dq)-state(-dq))/(2*h))
    J=np.array(slopes).T
    # Keep the observed specimen-plane D95 focus. Require near-zero axial field to
    # preserve positive envelope curvature in this exceptionally shallow relay.
    desired=np.array([0.,0.,0.,y[3],0.])
    selected=[0,3]
    control=zero.copy()
    control[:2]=np.linalg.solve(J[selected,:2],(desired-y)[selected])
    # A tensor match needs three independent transverse growth conditions.
    # Test a conservative local neighborhood and report any remaining anisotropy.
    search_radius=np.array([.5,.015,.5,.005])
    scales=np.array([1e-18,1e-18,1e-18,1e-17,1e-7])
    design=J*search_radius/scales[:,None]
    sol=lsq_linear(design,(desired-y)/scales,bounds=(-1.,1.),tol=1e-13)
    full=sol.x*search_radius
    outcome=dict(drives=list(modes),nominal_step_A=step.tolist(),state_units=["m rad","m rad","m rad","m rad","T"],
                 current_state=y.tolist(),derivative_per_A=J.tolist(),two_controls_delta_A=control.tolist(),
                 two_controls_native=state(control).tolist(),all_controls_delta_A=full.tolist(),
                 all_controls_native=state(full).tolist(),bounded_radius_A=search_radius.tolist(),
                 scaled_singular_values=np.linalg.svd(design,compute_uv=False).tolist())
    write(folder/"directional-response.json",outcome)
    print(json.dumps(outcome,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("label");p.add_argument("anchor",type=Path);p.add_argument("pair_anchor",type=Path)
    a=p.parse_args();evaluate(a.label,a.anchor,a.pair_anchor)
