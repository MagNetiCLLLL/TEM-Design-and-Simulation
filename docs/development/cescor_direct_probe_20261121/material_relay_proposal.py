"""Use a paraxial transfer to bracket trials; confirm selected settings by finite-source tracking."""
import json
import os
import re
import sys

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq

from column_current import FACTOR
from condenser_source_fields import FINAL, FEM
from joint_relay_search import transfer, mono_roots
from material_relay_fields import PARENT, WORK


END = 1.5991999975
Z = np.linspace(.45,END,9201)


def bz(path):
    with np.load(path) as raw:
        if "coeff" in raw:
            return np.interp(Z,raw["z"],2*raw["coeff"][:,0])
        mid=len(raw["xy_m"])//2
        return np.interp(Z-.45,raw["z_local_m"],raw["B_zyx_T"][:,mid,mid,2])


def interpolate(curves):
    keys=sorted(curves)
    return PchipInterpolator(keys,[curves[a] for a in keys],axis=0,extrapolate=False)


def survey(gun=None):
    folder=FEM.parent
    corr={float(re.search(r"tl12-([-0-9.]+)A",p.name).group(1)):bz(p) for p in folder.glob("*field-8.npz")}
    corr[10.]=bz(folder.parent/"normal-joint-field.npz")
    corr=interpolate(corr)
    roundmaps=[]
    prefix="circuit-ampere-turns"
    for key,var in (("condenser_lens_3","TEMSIM_RELAY_C3_A"),("mini_condenser","TEMSIM_RELAY_MINI_A")):
        if var in os.environ:
            prefix+=f"-{key}{float(os.environ[var]):.8g}"
    for tag in ("all","pair"):
        vals={}
        for p in sorted(WORK.glob(f"{prefix}-{tag}-*.npz"),key=lambda p:"-0.5mm." in p.name):
            with np.load(p) as raw:
                obj=float(json.loads(str(raw["recipe"]))["obj_A"])
            vals[obj]=bz(p)
        roundmaps.append(interpolate(vals))
    with np.load(gun or PARENT/"fresh-emission-and-exit.npz") as g:
        mask=g["weight"]>0
        phase=g["phase"][mask][:4096,:4]
    phase-=phase.mean(axis=0)
    x=phase[:,0]+1j*phase[:,1]
    t=phase[:,2]+1j*phase[:,3]
    def branches(obj, verbose=False):
        b=roundmaps[0](obj)
        a0,b0,c0,d0=transfer(Z,roundmaps[1](obj)).T
        D=-c0[-1]*b0+d0[-1]*a0
        fronts=sorted(np.r_[mono_roots(Z,D),mono_roots(Z,-D)])
        fronts=[f for f in fronts if 1.42<f<1.5125]
        res=[]
        for front in fronts:
            def inspect(tl):
                field=b+corr(tl)
                w=t-1j*FACTOR*field[0]*x
                X=np.mean(abs(x)**2);S=np.real(np.vdot(x,w))/len(x);V=np.mean(abs(w)**2)
                A,B,C,D=transfer(Z,field).T
                cov=1e9*(A*C*X+(A*D+B*C)*S+B*D*V)
                size=1e6*np.sqrt(np.maximum(A*A*X+2*A*B*S+B*B*V,0))
                return float(np.interp(front,Z,cov)),cov,size
            grid=np.linspace(corr.x[0],corr.x[-1],60)
            vals=[inspect(v)[0] for v in grid]
            for i in np.flatnonzero(np.array(vals[:-1])*vals[1:]<0):
                tl=brentq(lambda v:inspect(v)[0],grid[i],grid[i+1])
                _,cov,size=inspect(tl)
                roots=mono_roots(Z,cov)
                up=roots[(roots>1.35)&(roots<front-.001)]
                post=roots[(roots>front+.001)&(roots<END-.0005)]
                if len(roots) and min(abs(roots-front))<.0005:
                    res.append([round(obj,5),round(tl,5),round(1000*front,4),round(cov[-1],6),
                        round(size[-1],5),list(np.round(up*1000,2)),list(np.round(post*1000,2))])
        return res
    for obj in np.linspace(max(v.x[0] for v in roundmaps),min(v.x[-1] for v in roundmaps),45):
        for row in branches(obj):
            print(row,flush=True)

if __name__=="__main__":
    survey(*sys.argv[1:])
