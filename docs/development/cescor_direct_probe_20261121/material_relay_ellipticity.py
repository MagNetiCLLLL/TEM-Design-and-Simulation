"""Test whether the available round controls can register both directional beam minima."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from material_relay_branches import FACTOR,fast,gun_sigma
from material_relay_column import Column,Field
from material_relay_fields import SAVE,WORK
from material_relay_response import groups
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import advance


def search(label,anchor,pair_anchor):
    folder=SAVE/label;r=json.loads((folder/"tracking.json").read_text())
    f,g=[WORK/r[k]["source"] for k in ("field","pair")]
    target=r["isolated_pair_fronts"][0]["z_mm"]*.001
    top=r["surface_z_mm"]*.001
    model=Column(RefinedObjectiveFields(.5),Field(f),Path(r["gun_source"]),r["count"],r["step_mm"])
    i=np.searchsorted(model.m.z,target);stops=np.r_[i-2:i+2,len(model.m.z)-1]
    measured=[]
    for p,loss in advance(model.t,model.full,model.t.input,model.en,model.phi0,model.loss,0,stops):
        w=model.t.weights[loss==-1];w=w/w.sum()
        points=p[loss==-1,:4];points=points-np.average(points,weights=w,axis=0)
        x=points[:,0]+1j*points[:,1];v=points[:,2]+1j*points[:,3]
        measured.append([np.sum(w*(x.conj()*v).real),np.sum(w*(x*v).real),np.sum(w*(x*v).imag)])
    measured=np.array(measured)
    obs=np.r_[CubicSpline(model.m.z[stops[:-1]],measured[:-1])(target),measured[-1,0]]
    z=np.linspace(.45,top,5747)
    def load(path,key):
        with np.load(WORK/f"{path.stem}-{key}-bore-0.5mm.npz") as a:return CubicSpline(a["z"],a["coef"][:,0,2])(z)
    f0,g0=load(f,"dc"),load(g,"dc");modes=("tl","objective","mini","c3")
    deriv=np.array([load(anchor,k) for k in modes])
    gp=np.array([load(pair_anchor,k) for k in ("objective","mini")])
    currents=r["field"]["bias"];base=np.array([currents[groups(currents)[k][0]] for k in modes])
    x,v,w=gun_sigma(r["gun_source"]);v-=1j*FACTOR*f0[0]*x
    xx=np.sum(w*abs(x)**2);xv=np.sum(w*(x.conj()*v).real);vv=np.sum(w*abs(v)**2)
    xx2=np.sum(w*x*x);xv2=np.sum(w*x*v);vv2=np.sum(w*v*v)
    em=np.sqrt(xx*vv-xv*xv)
    def front(g):
        a,b,c,d=fast(z,g).T;y=-c[-1]*b+d[-1]*a
        ids=np.flatnonzero((z[:-1]>1.42)&(z[:-1]<1.51)&(y[:-1]*y[1:]<=0))
        if not len(ids):raise ValueError("Isolated front moved outside the scan interval")
        i=ids[np.argmin(abs(z[ids]-target))]
        return z[i]-(z[i+1]-z[i])*y[i]/(y[i+1]-y[i])
    offset=target-front(g0)
    def calc(dq):
        field=f0+np.einsum("i,ij->j",dq,deriv)
        g=g0+np.einsum("i,ij->j",dq[1:3],gp)
        ff=front(g)+offset
        t=fast(z,field)
        a,b,c,d=CubicSpline(z,t)(ff)
        k=FACTOR*np.interp(ff,z,field)
        rad=a*c*xx+(a*d+b*c)*xv+b*d*vv
        var=a*a*xx+2*a*b*xv+b*b*vv
        q=a*c*xx2+(a*d+b*c)*xv2+b*d*vv2+1j*k*(a*a*xx2+2*a*b*xv2+b*b*vv2)
        th=np.interp(ff,z,np.r_[0.,np.cumsum(.5*(field[:-1]+field[1:])*np.diff(z))])*FACTOR
        raw=q*np.exp(2j*th)
        a,b,c,d=t[-1]
        return np.array([rad,raw.real,raw.imag,a*c*xx+(a*d+b*c)*xv+b*d*vv]),k*var/em,ff
    zero=np.zeros(4);bias=obs-calc(zero)[0];scale=np.array([1e-15,1e-17,1e-17,1e-15])
    def fun(q):
        val,rot,ff=calc(q)
        return np.r_[(val+bias)/scale, max(0,abs(rot)-.8)]
    solutions=[]
    for start in [zero,[-7.,.001,-.005,-.01],[7.,.001,-.005,.01]]:
        opt=least_squares(fun,start,bounds=([-24,-.35,-4,-.14],[24,.35,4,.14]),x_scale=[5.,.1,1.,.03],
                          xtol=2e-11,ftol=2e-11,gtol=2e-11,max_nfev=65)
        vec,rot,ff=calc(opt.x)
        solutions.append(dict(currents=dict(zip(modes,(base+opt.x).tolist())),normalized_residual=opt.fun.tolist(),
              field_to_emittance_ratio=rot,front_mm=1000*ff,weighted_residual_norm=float(np.linalg.norm(opt.fun))))
    payload=dict(measured_m_rad=obs.tolist(),meridional_offset_m_rad=bias.tolist(),candidates=solutions,
                 purpose="Local tangent search; every selected drive requires another nonlinear FEM and original-particle replay")
    (folder/"directional-search.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("label");p.add_argument("anchor",type=Path);p.add_argument("pair_anchor",type=Path)
    a=p.parse_args();search(a.label,a.anchor,a.pair_anchor)
