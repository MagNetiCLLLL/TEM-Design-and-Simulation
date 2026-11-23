"""Bracket downstream field solutions and distinguish axial minima from maxima."""
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root
from numba import njit

from column_current import FACTOR
from condenser_source_fields import FEM
from joint_relay_search import transfer
from material_relay_fields import WORK,SAVE

Z=np.linspace(.45,1.5991999975,2501)


def axis(path):
    with np.load(path) as raw:
        if "coeff" in raw:return np.interp(Z,raw["z"],2*raw["coeff"][:,0])
        half=len(raw["xy_m"])//2
        return np.interp(Z-.45,raw["z_local_m"],raw["B_zyx_T"][:,half,half,2])


def gun_sigma(path):
    with np.load(path) as g:
        p=g["phase"];w=g["weight"]/g["weight"].sum()
    p=p[:,:4]-np.average(p[:,:4],axis=0,weights=w)
    x=p[:,0]+1j*p[:,1];t=p[:,2]+1j*p[:,3]
    return x,t,w


@njit(cache=True)
def fast(z,field):
    result=np.empty((len(z),4))
    A,B,C,D=1.,0.,0.,1.
    result[0]=[A,B,C,D]
    for i in range(1,len(z)):
        step=z[i]-z[i-1];k=FACTOR*.5*(field[i]+field[i-1]);cos=np.cos(k*step)
        sin=np.sin(k*step)/k if abs(k)>1e-9 else step
        A,B,C,D=cos*A+sin*C,cos*B+sin*D,cos*C-k*k*sin*A,cos*D-k*k*sin*B
        result[i]=[A,B,C,D]
    return result


class RoundMaps:
    def __init__(self):
        self.maps=[]
        for prefix in ("circuit-ampere-turns-condenser_lens_36.710114-mini_condenser239.91265","circuit-ampere-turns"):
            row=[]
            for flag in ("all","pair"):
                found={}
                for p in sorted(WORK.glob(f"{prefix}-{flag}-*.npz"),key=lambda p:("-0.5mm" in p.name)):
                    cur=float(re.search(rf"-{flag}-([-\d.]+)A",p.name)[1])
                    found[cur]=axis(p)
                vals=sorted(found)
                row.append(PchipInterpolator(vals,[found[k] for k in vals],axis=0))
            self.maps.append(row)
        corrector={10.:axis(FEM.parent.parent/"normal-joint-field.npz")}
        for p in sorted(FEM.parent.glob("tl12-*-field-*.npz"),key=lambda p:"field-24" in p.name):
            corrector[float(re.search(r"tl12-([-\d.]+)A",p.name)[1])]=axis(p)
        vals=sorted(corrector)
        self.corrector=PchipInterpolator(vals,[corrector[k] for k in vals],axis=0)

    def __call__(self,tl,obj,mini):
        lo,hi=self.maps
        s=(mini-239.912653194986)/(350.67461-239.912653194986)
        base,pair=(v(obj) for v in lo)
        high=hi[1](obj)
        offset=s*(high-pair)
        return base+offset+self.corrector(tl),pair+offset


def assess(fields,source,tl,obj,mini,front_hint=1.46):
    full,pair=fields(tl,obj,mini)
    A,B,C,D=fast(Z,full).T
    a,b,c,d=fast(Z,pair).T
    signed=-c[-1]*b+d[-1]*a
    roots=np.flatnonzero((Z[:-1]>1.419)&(Z[:-1]<1.514)&(signed[:-1]*signed[1:]<=0))
    if not len(roots):raise ValueError("No downstream front")
    i=roots[np.argmin(abs(Z[roots]-front_hint))]
    front=Z[i]-(Z[i+1]-Z[i])*signed[i]/(signed[i+1]-signed[i])
    x,t,w=source;v=t-1j*FACTOR*full[0]*x
    xx=np.sum(w*abs(x)**2);xv=np.sum(w*np.real(x.conj()*v));vv=np.sum(w*abs(v)**2)
    cov=A*C*xx+(A*D+B*C)*xv+B*D*vv
    var=A*A*xx+2*A*B*xv+B*B*vv
    ang=C*C*xx+2*C*D*xv+D*D*vv
    deriv=ang-(FACTOR*full)**2*var
    center=np.interp(front,Z,cov)
    slope=np.interp(front,Z,deriv)
    radius=np.sqrt(np.maximum(var,0))
    roots=np.flatnonzero((cov[:-1]<0)&(cov[1:]>=0))
    return dict(front=front,waist_cov=center,surface_cov=cov[-1],surface_curvature=deriv[-1],
        front_curvature=slope,front_radius=np.interp(front,Z,radius),surface_radius=radius[-1],
        additional=[float(Z[i]) for i in roots if front+.002<Z[i]<Z[-1]-.001],
        condenser_roots=[float(Z[i]) for i in roots if .45<Z[i]<1.070],
        up=[float(Z[i]) for i in roots if 1.350<Z[i]<front-.002])


def search(fields,src,minis):
    solutions=[]
    for mini in minis:
        found=set()
        for tl0 in (-110.,-50.,10.,100.,210.,265.):
            for obj0 in (-22.,-30.,-37.,-42.):
                def fun(p):
                    tl,obj=p*np.array([100.,30.])
                    if not -300<tl<450 or not -55<obj<-10:return [100.,100.]
                    try:a=assess(fields,src,tl,obj,mini)
                    except ValueError:return [1.,1.]
                    return np.array([a["waist_cov"],a["surface_cov"]])*1e11
                sol=root(fun,[tl0/100.,obj0/30.])
                if not sol.success or np.linalg.norm(fun(sol.x))>2e-6:continue
                t,o=sol.x*np.array([100.,30.]);key=(round(t,1),round(o,2))
                if key in found or not -125<t<278 or not -44<o<-12:continue
                found.add(key);a=assess(fields,src,t,o,mini)
                rec=dict(tl=t,obj=o,mini=mini,**a)
                print(f"{mini:8.2f} {t:9.3f} {o:8.4f} {a['front']*1000:8.3f} "
                      f"{a['front_curvature']*1e9:9.5f} {a['surface_curvature']*1e9:9.5f} "
                      f"{a['surface_radius']*1e6:8.4f} {a['additional']}",flush=True)
                solutions.append(rec)
    return solutions


def search_weak():
    maps=RoundMaps()
    gun=json.loads((SAVE/"optical-start/tracking.json").read_text())["gun_source"]
    src=gun_sigma(gun);q0=6.710114018150848
    p=next(WORK.glob("circuit-ampere-turns-condenser_lens_30-mini_condenser*-all--28.22027595A-1mm.npz"))
    ref=next(WORK.glob("circuit-ampere-turns-condenser_lens_36.710114-mini_condenser*-all--28.22027595A-1mm.npz"))
    delta=(axis(ref)-axis(p))/q0
    out=[]
    for c3 in [6.6,6.5,6.4,6.2,6.,5.8,5.6,5.2,4.5,3.,1.,.1,6.8,6.95,7.1]:
        def fields(t,o,m):
            full,pair=maps(t,o,m)
            return full+(c3-q0)*delta,pair
        branch=search(fields,src,[-150.,0.,80.,239.912653194986])
        for a in branch:
            if a["surface_curvature"]>0 and a["front_curvature"]>0 and not a["additional"]:
                out.append(dict(c3=c3,**a))
                print("Positive minima",out[-1],flush=True)
    (WORK/"weak-minima.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":
    import sys
    if "--weak" in sys.argv:
        search_weak();sys.exit()
    r=RoundMaps();g=json.loads((SAVE/"optical-start/tracking.json").read_text())["gun_source"]
    ans=search(r,gun_sigma(g),[80.,140.,190.,220.,239.912653194986,265.,300.,350.,400.,-120.,-220.,-350.,0.,480.])
    (WORK/"optical-branches.json").write_text(json.dumps(ans,indent=2))
