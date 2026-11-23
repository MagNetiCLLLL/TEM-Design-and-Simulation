"""Shoot the physical C2 foil angle and specimen focus after an independent nonlinear trace."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from screen import WORK, KEYS, SEED, FACTOR


@njit(cache=True)
def transfer(z,k):
    out=np.empty((len(z),4))
    a,b,ap,bp=1.,0.,0.,1.
    out[0]=[a,b,ap,bp]
    for j in range(len(k)):
        h=z[j+1]-z[j]
        w=k[j]
        co=np.cos(w*h)
        sn=np.sin(w*h)/w if abs(w)>1e-13 else h
        a,b,ap,bp=co*a+sn*ap,co*b+sn*bp,co*ap-w*w*sn*a,co*bp-w*w*sn*b
        out[j+1]=[a,b,ap,bp]
    return out


def propose(folder, signed_edge, tangent, margin, scan_z, keep_relay, trim_output, strong, focus_relay, all_stops):
    r=json.loads((folder/"tracking.json").read_text())
    rows=r["field"]["bias"]
    names=["condenser_lens_1/physical-pack-0","condenser_lens_2/physical-pack-1",
           "condenser_lens_3/physical-pack-2","adapter_lens","probe_tl12_lens",
           "mini_condenser/physical-pack-3","objective_lens/physical-pack-4"]
    ref=np.array([rows[n] for n in names])
    if focus_relay:
        vary=np.array([3,4,6])
    elif keep_relay:
        vary=np.array([1,2,6])
    elif trim_output:
        vary=np.array([1,2,4,6])
    elif strong:
        vary=np.array([1,3,4,6])
    else:
        vary=np.array([1,2,3,4,5,6])
    targets=np.array([1208.15,scan_z,1599.1999975])*.001
    foil=np.array([.7649,.7651])
    stop_z=np.array([1.0099,1.0101])
    z=np.unique(np.r_[np.linspace(.45,targets[-1],4301),targets,foil,stop_z])
    mid=(z[:-1]+z[1:])/2
    path=Path(r["solution"])
    with np.load(WORK/f'{path.stem}-dc-bore-{r["bore_step_mm"]:g}mm.npz') as raw:
        dc=CubicSpline(raw["z"],raw["coef"][:,0,2])
    cols=[]
    for i,key in enumerate(KEYS):
        if i not in vary:
            cols.append(np.zeros_like(mid))
            continue
        with np.load(WORK/f"{tangent}-{key}-bore-0.75mm.npz") as raw:
            cols.append(CubicSpline(raw["z"],raw["coef"][:,0,2])(mid))
    fields=np.array(cols)
    with np.load(r["gun_source"]) as g:
        a=g["phase"];w=g["weight"]
        xy=a[:,0]+1j*a[:,1];v=a[:,2]+1j*a[:,3]
        xy-=w@xy;v-=w@v
    u=v-1j*FACTOR*dc(z[0])*xy
    X=w@abs(xy)**2; S=w@(xy.conj()*u).real; V=w@abs(u)**2
    real=S/X; imag=np.sqrt(max(0.,X*V-S*S))/X
    base=dc(mid)
    geometric_masks=[]
    if all_stops:
        from material_trial import source_and_electrodes
        from condenser_pair_column import column_mesh
        hardware,_=column_mesh(source_and_electrodes(),r["step_mm"])
        intervals=np.clip(np.searchsorted(hardware.z,z,side="right")-1,0,len(hardware.radius)-1)
        radius=hardware.radius[intervals]*1000
        for l,h in ((.45,.76489),(.76511,1.00989),(1.00989,1.01011),(1.01011,1.32),(1.32,1.5125),(1.5125,1.60)):
            geometric_masks.append(np.flatnonzero((z>=l)&(z<h)&np.isfinite(radius)))
    ix=np.searchsorted(z,targets)
    jf=np.searchsorted(z,foil)
    def evaluate(p):
        q=ref.copy();q[vary]=p
        m=transfer(z,FACTOR*(base+(q-ref)@fields))
        a,b,c,d=m.T
        y=a+real*b;v=c+real*d
        cv=y*v+imag**2*b*d
        roots=np.where((cv[:-1]<0)&(cv[1:]>=0))[0]
        rd=(z[roots]-np.diff(z)[roots]*cv[roots]/(cv[roots+1]-cv[roots]))*1000
        both=(b[-1]*a[jf]-a[-1]*b[jf])*1000
        edge_element=both[np.argmax(abs(both))]
        required=np.abs(b[-1]*a-a[-1]*b)*1000*np.tan(.030)
        fill=np.array([np.max(required[ids]/radius[ids]) for ids in geometric_masks])
        return y[ix],v[ix],edge_element,rd,float(np.hypot(v[-1],imag*d[-1])*np.sqrt(X)*1e3),fill
    y0,v0,b0,roots0,_,_=evaluate(ref[vary])
    from_native=np.zeros(3)
    for i,t in enumerate(targets[:2]):
        observed=[x for x in r["waists_mm"] if abs(x-t*1000)<110.]
        estimated=[x for x in roots0 if abs(x-t*1000)<110.]
        if observed and estimated:
            from_native[i]=(min(observed,key=lambda x:abs(x-t*1000)) -
                            min(estimated,key=lambda x:abs(x-t*1000)))*.001
    with np.load(folder/"column-and-landings.npz") as col:
        live=col["loss"]==-1
        ph=col["surface_phase"][live,:4].copy()
        w=col["weights"][live]
        ph-=np.average(ph,axis=0,weights=w)
        ratio=np.sum(w*np.sum(ph[:,:2]*ph[:,2:],axis=1))/np.sum(w*np.sum(ph[:,2:]**2,axis=1))
    if abs(ratio)<.02 and abs(y0[-1]/v0[-1])<.02:
        from_native[-1]=y0[-1]/v0[-1]-ratio
    scales=np.maximum(np.abs(v0),4.)
    trust=np.array([0.,5.,5.,40.,80.,95.,18.])[vary]*margin
    lo=np.maximum(ref[vary]-trust,np.array([-100.,4.,0.,-350.,-700.,-700.,-90.])[vary])
    hi=np.minimum(ref[vary]+trust,np.array([0.,50.,12.,350.,700.,700.,90.])[vary])
    desired=signed_edge*.05/np.tan(.03)
    def residual(p):
        y,v,coef,_,_,fill=evaluate(p)
        planes=(y-from_native*v)/scales*1000/[.4,.4,.02]
        if focus_relay:
            return planes
        return np.r_[planes[[0,2]] if keep_relay else planes,(coef-desired)/.035,200*np.maximum(fill-.98,0.),
                     .024*(p-ref[vary])/trust]
    starts=[ref[vary]]
    if strong or focus_relay or all_stops:
        rng=np.random.default_rng(2211)
        starts.extend(lo+rng.random((20,len(vary)))*(hi-lo))
    solutions=[least_squares(residual,x,bounds=(lo,hi),max_nfev=150,xtol=1e-11) for x in starts]
    def quality(fit):
        _,_,_,roots,_,_=evaluate(fit.x)
        g=roots[roots<1599.12]
        proper=(len(g)==4 and 450<g[0]<550 and 750<g[1]<830 and
                1179.1<g[2]<1237.2 and 1420<g[3]<1512.5)
        return (not proper,fit.cost)
    sol=min(solutions,key=quality)
    q=ref.copy();q[vary]=sol.x
    y,v,coef,roots,ang,fill=evaluate(sol.x)
    out=dict(proposed=dict(zip(KEYS,q.tolist())),signed_c2_B_mm=coef,
             formal_edge_mrad=1000*np.arctan(.05/abs(coef)),screening_waists_mm=roots.tolist(),
             finite_source_rms_angle_mrad=ang,source_particles=len(xy),reference=r["solution"],
             reference_waists_mm=r["waists_mm"],reference_paraxial_roots_mm=roots0.tolist(),
             local_native_offset_mm=(1000*from_native).tolist(),local_cost=sol.cost,
             required_hardware_radius_fractions_at_30mrad=fill.tolist())
    prefix="all-stops" if all_stops else "geometric-focus" if focus_relay else "condenser" if keep_relay else "output" if trim_output else "strong" if strong else "edge"
    (folder/f"{prefix}-proposal-{signed_edge:+d}.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    p.add_argument("--tangent",default=SEED)
    p.add_argument("--margin",type=float,default=1.)
    p.add_argument("--scan-z",type=float,default=1437.)
    p.add_argument("--keep-relay",action="store_true")
    p.add_argument("--trim-output",action="store_true")
    p.add_argument("--strong",action="store_true")
    p.add_argument("--focus-relay",action="store_true")
    p.add_argument("--all-stops",action="store_true")
    a=p.parse_args()
    for sign in ((1,) if a.focus_relay else (-1,1)):
        propose(a.folder,sign,a.tangent,a.margin,a.scan_z,a.keep_relay,a.trim_output,a.strong,a.focus_relay,a.all_stops)
