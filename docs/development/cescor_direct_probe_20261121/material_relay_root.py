"""Eliminate the large meridional ray before fitting the finite-emittance waist."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from material_relay_branches import FACTOR, WORK, SAVE, Z, fast,gun_sigma,assess
from material_relay_search import Atlas,plane


def find(atlas,src,m,beta,seed_obj):
    x,t,w=src
    xx=np.sum(w*abs(x)**2)
    def tail(tl,obj,return_z=False):
        field,pair=atlas(tl,obj,atlas.c0,m)
        f=plane(pair)
        if f is None:return np.nan
        zf=f[0]
        idx=np.searchsorted(Z,zf)
        a,b,c,d=fast(np.r_[zf,Z[idx:]],np.r_[np.interp(zf,Z,field),field[idx:]])[-1]
        return zf if return_z else -d*c*beta-b*a/beta
    solutions=[]
    def evaluate(tl,expo=False):
        og=np.linspace(seed_obj-4.,seed_obj+4.,25)
        vals=np.array([tail(tl,o) for o in og])
        ids=np.flatnonzero(vals[:-1]*vals[1:]<=0)
        if not len(ids):return None
        i=min(ids,key=lambda j:abs(og[j]-seed_obj))
        o=brentq(lambda v:tail(tl,v),og[i],og[i+1])
        btotal,_=atlas(tl,o,atlas.c0,m)
        def match(c):
            field=btotal+(c-atlas.c0)*atlas.base["c3"]
            a,b,h,d=fast(Z,field)[-1]
            v=t-1j*FACTOR*field[0]*x
            xv=np.sum(w*(x.conj()*v).real);vv=np.sum(w*abs(v)**2)
            eps=np.sqrt(xx*vv-xv*xv)
            slope=(xv+1j*eps)/xx
            return (h+d*slope-1j*(a+b*slope)/beta)*np.sqrt(xx*beta/eps)
        cs=np.linspace(4.,8.4,36)
        g=np.array([match(c).imag for c in cs])
        returned=[]
        for i in np.flatnonzero(g[:-1]*g[1:]<=0):
            c=brentq(lambda c:match(c).imag,cs[i],cs[i+1])
            val=match(c).real
            returned.append((c,o,val))
        return returned
    grid=np.linspace(-124.,276.,45)
    last=None
    for tt in grid:
        this=evaluate(tt)
        if not this:last=None;continue
        if last is not None:
            t0,lasts=last
            for c,o,v in this:
                prev=min(lasts,key=lambda a:abs(a[0]-c))
                if abs(prev[0]-c)>.35 or prev[2]*v>0:continue
                def residual(t):
                    vals=evaluate(t)
                    return min(vals,key=lambda a:abs(a[0]-c))[2]
                tl=brentq(residual,t0,tt,xtol=1e-6)
                c,o,v=min(evaluate(tl),key=lambda a:abs(a[0]-c))
                r=assess(lambda t,o,m:atlas(t,o,c,m),src,tl,o,m)
                r.update(tl=tl,c3=c,obj=o,mini=m,beta=beta)
                print("Finite source",r,flush=True);solutions.append(r)
        last=tt,this
    return solutions


if __name__=="__main__":
    import sys
    a=Atlas(Path("global-7a804e13d71e743b.npz"))
    src=gun_sigma(WORK/"common-gun-d052a5dced69e29cfbe8c02a142a92b493e9987ce0941abf1c92cbf3c0ce9cbe.npz")
    out=[]
    for m,o in [(180.,-2.22),(100.,4.05),(-175.,1.71),(-100.,-4.05)]:
        for b in (.25,.55,1.,.13,.05,1.8,3.):
            s=find(a,src,m,b,o);out+=s
            if s:(WORK/"common-finite-candidates.json").write_text(json.dumps(out,indent=2))
