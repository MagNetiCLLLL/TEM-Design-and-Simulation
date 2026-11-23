"""Use solved DC field families to select a two-waist relay and objective focus trial."""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq,least_squares

from material_relay_branches import FACTOR,RoundMaps,WORK,SAVE,Z,axis,fast,gun_sigma,assess
from material_relay_match import fit


class Atlas:
    def __init__(self,dc):
        self.dc=dc
        self.base={k:self.read(k) for k in ("dc","tl","c3","objective","mini")}
        self.c0=6.71011401815085;self.m0=239.912653194986
        self.t0=-65.;self.o0=-30.1
        self.curves={}
        for p in WORK.glob("circuit-ampere-turns-condenser_lens_36.710114-mini_condenser*-pair-*A-1mm.npz"):
            m,o=map(float,re.search(r"mini_condenser([-\d.]+)-pair-([-\d.]+)A",p.name).groups())
            self.curves.setdefault(m,{})[o]=axis(p)
        mini=sorted(m for m,values in self.curves.items() if min(values)<=-40 and max(values)>=40)
        obj=np.array([-40.,-29.45,-20.,-10.,0.,10.,20.,30.,40.])
        pairs=[]
        for m in mini:
            row=self.curves[m];xs=sorted(row)
            pairs.append(PchipInterpolator(xs,[row[o] for o in xs],axis=0)(obj))
        self.minigrid=np.array(mini);self.ogrid=obj
        self.grid=np.array(pairs)
        self.by_m=PchipInterpolator(mini,self.grid,axis=0)
        self.round=RoundMaps()
        self.baseline=self.pair(self.o0,self.m0)
        self.tl0=self.round.corrector(self.t0)
        self.tlder=self.round.corrector.derivative()(self.t0)
        self.coef={}
    def read(self,key):
        with np.load(WORK/f"{self.dc.stem}-{key}-bore-0.5mm.npz") as f:
            return np.interp(Z,f["z"],f["coef"][:,0,2])
    def pair(self,o,m):
        if not hasattr(self,"last_m") or self.last_m!=m:
            self.last_m=m;self.obj_curve=PchipInterpolator(self.ogrid,self.by_m(m),axis=0)
        return self.obj_curve(o)
    def __call__(self,t,o,c,m):
        pair=self.pair(o,m)
        corr=self.round.corrector(t)-self.tl0
        # The response discrepancy is a local first-order correction to independently solved fields.
        corr+=np.clip(t-self.t0,-80.,80.)*(self.base["tl"]-self.tlder)
        return self.base["dc"]+(pair-self.baseline)+corr+(c-self.c0)*self.base["c3"],pair


def plane(field,hint=1.465):
    a,b,c,d=fast(Z,field).T
    dd=-c[-1]*b+d[-1]*a
    idx=np.flatnonzero((Z[:-1]>1.42)&(Z[:-1]<1.512)&(dd[:-1]*dd[1:]<=0))
    if not len(idx):return None
    i=min(idx,key=lambda j:abs(Z[j]-hint))
    f=Z[i]-(Z[i+1]-Z[i])*dd[i]/(dd[i+1]-dd[i])
    af,bf,cf,df=[np.interp(f,Z,k) for k in (a,b,c,d)]
    return f,a[-1]*df-b[-1]*cf,-a[-1]*bf+b[-1]*af


def roots(atlas,src,mini,beta):
    rows=[];os=np.linspace(-40,40,201)
    for o in os:
        p=plane(atlas.pair(o,mini))
        rows.append(None if p is None else list(p))
    candidates=[]
    for i,(left,right) in enumerate(zip(rows[:-1],rows[1:])):
        if left is None or right is None or left[1]*right[1]>0 or abs(left[0]-right[0])>.01:continue
        o=brentq(lambda o:plane(atlas.pair(o,mini))[1],os[i],os[i+1])
        f,a,b=plane(atlas.pair(o,mini))
        print("Fourier pair",mini,o,1000*f,b,flush=True)
        for t in [-110.,-62.,20.,110.,220.,267.]:
            for c in [6.4,7.15,7.45]:
                r=fit(atlas,src,[t,o,c],mini,beta,([-124.,-41.,.5],[276.5,41.,8.3]))
                if r and r["front_curvature"]>0 and r["surface_curvature"]>0 and not r["additional"]:
                    if not any(abs(x["tl"]-r["tl"])<.5 for x in candidates):
                        print("Candidate",r,flush=True);candidates.append(r)
    return candidates


if __name__=="__main__":
    dc=Path(sys.argv[1])
    atlas=Atlas(dc)
    from material_relay_fields import digest,sha,parent
    field_path=WORK/f"{dc.stem}-dc-bore-0.5mm.npz"
    gun=WORK/("common-gun-"+digest(dict(field=sha(field_path),physical_source=parent()["source_signature"],
                                        n=4096,gun_sampler="cartesian-cubic-v2"))+".npz")
    src=gun_sigma(gun);found=[]
    for m in [240.,180.,100.,50.,0.,-50.,-100.,-175.,280.,320.,360.,400.,450.]:
        found.extend(roots(atlas,src,m,.021))
        (WORK/"common-candidates.json").write_text(json.dumps(found,indent=2))
