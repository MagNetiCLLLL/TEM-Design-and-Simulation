"""Commission the relay using the local derivatives of the two nonlinear 3-D material solutions."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq,least_squares

from material_relay_branches import fast,FACTOR,gun_sigma
from material_relay_fields import WORK,SAVE


def fit(full,pair,folder,tangent_full=None,tangent_pair=None,*,tracking=None,source=None,
        front_and_surface_cov=None,surface_target=0.,filename="steering.json"):
    record=tracking if tracking is not None else json.loads((folder/"tracking.json").read_text())
    step=record["field"]["step_mm"]
    z=np.linspace(.45,record["surface_z_mm"]*.001,11493)
    def load(path,key):
        name=WORK/f"{path.stem}-{key}-bore-{step:g}mm.npz"
        with np.load(name) as data:
            return CubicSpline(data["z"],data["coef"][:,0,2])(z)
    modes=("tl","objective","mini","c3")
    f0=load(full,"dc");g0=load(pair,"dc")
    derivatives=np.array([load(tangent_full or full,k) for k in modes])
    pair_derivatives=np.array([load(tangent_pair or pair,k) for k in ("objective","mini")])
    from material_relay_response import groups
    amps=record["field"]["bias"];keys=groups(amps)
    currents=np.array([amps[keys[k][0]] for k in modes])
    x,v,w=gun_sigma(source or record["gun_source"])
    t=v-1j*FACTOR*f0[0]*x
    xx=np.sum(w*abs(x)**2);xv=np.sum(w*(x.conj()*t).real)
    eps=np.sqrt(xx*np.sum(w*abs(t)**2)-xv**2)
    slope=xv/xx+1j*eps/xx
    target=record["isolated_pair_fronts"][0]["z_mm"]*.001
    with np.load(folder/"column-and-landings.npz") as data:
        positions=data["z_mm"]*.001;cov=(data["stats"][:,1:3].sum(axis=1))*1e-9
        true_cov=CubicSpline(positions,cov)
    def pair_root(g):
        a,b,c,d=fast(z,g).T
        da=-c[-1]*b+d[-1]*a
        ids=np.flatnonzero((z[:-1]>1.420)&(z[:-1]<1.53)&(da[:-1]*da[1:]<=0))
        if not len(ids):raise ValueError("No illumination-side front on the selected branch")
        i=ids[np.argmin(abs(z[ids]-target))]
        return z[i]-(z[i+1]-z[i])*da[i]/(da[i+1]-da[i])
    front_offset=target-pair_root(g0)
    def state(dq):
        f=f0+np.einsum("i,ij->j",dq,derivatives)
        g=g0+dq[1]*pair_derivatives[0]+dq[2]*pair_derivatives[1]
        front=pair_root(g)+front_offset
        a,b,c,d=fast(z,f).T
        pos=a+b*slope;angle=c+d*slope
        covariance=xx*(pos.conj()*angle).real
        return np.array([CubicSpline(z,covariance)(front),covariance[-1],np.interp(front,z,f)]),front
    zero=np.zeros(4)
    raw,ff0=state(zero)
    measured=front_and_surface_cov if front_and_surface_cov is not None else (true_cov(target),true_cov(positions[-1]))
    correction=np.array([measured[0]-raw[0],measured[1]-raw[1],0.])
    def physical(dq):
        val,ff=state(dq)
        return val+correction,ff
    def resid(dq):
        val,_=physical(dq)
        return (val-np.array([0.,surface_target,0.]))/np.array([1e-12,1e-12,.02])
    cases=[]
    for held in ("c3","tl"):
        free=[i for i,k in enumerate(modes) if k!=held]
        def emit(v):
            out=zero.copy();out[free]=v
            return out
        limits=np.array([105.,6.,85.,.45])[free];scales=np.array([45.,2.,40.,.20])[free]
        sol=least_squares(lambda v:resid(emit(v)),np.zeros(3),bounds=(-limits,limits),x_scale=scales,
                          xtol=2e-13,ftol=2e-13,gtol=2e-13,max_nfev=120)
        dq=emit(sol.x);val,ff=physical(dq)
        cases.append(dict(held=held,currents=dict(zip(modes,(currents+dq).tolist())),delta=dq.tolist(),front_mm=1000*ff,
                          residual_scaled=sol.fun.tolist(),unscaled=val.tolist()))
    # For the user's last objective-only step, find the nearby specimen focus.
    def obj(v):
        # An objective-only specimen focus is defined even if the isolated
        # diagnostic front leaves the selected scan interval during bracketing.
        axial=f0+v*derivatives[1]
        a,b,c,d=fast(z,axial)[-1]
        return xx*((a+b*slope).conjugate()*(c+d*slope)).real+correction[1]-surface_target
    grid=np.linspace(-4,4,81);y=np.array([obj(v) for v in grid])
    roots=[brentq(obj,grid[i],grid[i+1]) for i in np.flatnonzero(y[:-1]*y[1:]<=0)]
    candidates=[]
    for delta in roots:
        f=f0+delta*derivatives[1]
        a,b,c,d=fast(z,f)[-1]
        cur=xx*(abs(c+d*slope)**2-(FACTOR*f[-1])**2*abs(a+b*slope)**2)
        if cur>0:candidates.append((abs(delta),currents[1]+delta))
    receipt=dict(base=dict(zip(modes,currents.tolist())),current_true_front_mm=1000*target,
                 search_tangents=[(tangent_full or full).name,(tangent_pair or pair).name],
                 proposal_source=str(source or record["gun_source"]),surface_target_cov_m_rad=surface_target,
                 current_covariance_m_rad=physical(zero)[0].tolist(),emittance_m_rad=float(eps),
                 measured_minus_meridional=correction.tolist(),candidates=cases,
                 objective_only_A=min(candidates)[1] if candidates else None)
    (folder/filename).write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("full",type=Path);p.add_argument("pair",type=Path)
    p.add_argument("--label",required=True)
    p.add_argument("--tangent-full",type=Path);p.add_argument("--tangent-pair",type=Path);q=p.parse_args()
    fit(q.full,q.pair,SAVE/q.label,q.tangent_full,q.tangent_pair)
