"""Find candidate two-waist solutions with a field null near the entrance focal plane."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq,least_squares
from material_relay_branches import FACTOR,WORK,Z,fast,gun_sigma,assess
from material_relay_search import Atlas,plane


def candidate(atlas,src,tl,mini0,obj0):
    x,t,w=src
    xx=np.sum(w*abs(x)**2)
    def step(q,details=False):
        obj,mini=q
        f,p=atlas(tl,obj,atlas.c0,mini)
        pp=plane(p)
        if pp is None:return None if details else [5.,5.]
        zf=pp[0]; k=np.searchsorted(Z,zf)
        seg=np.r_[zf,Z[k:]]
        a,b,c,d=fast(seg,np.r_[np.interp(zf,Z,f),f[k:]])[-1]
        return (zf,a,b,c,d,np.interp(zf,Z,f)) if details else [a,80*np.interp(zf,Z,f)]
    sol=least_squares(step,[obj0,mini0],x_scale=[5.,130.],bounds=([-42.,-201.],[42.,448.]),max_nfev=90,
                      ftol=1e-12,xtol=1e-12,gtol=1e-12)
    if np.linalg.norm(sol.fun)>2e-4:return []
    obj,mini=sol.x; zf,a,b,c,d,bz=step(sol.x,True)
    def sample(c3):
        field,pair=atlas(tl,obj,c3,mini)
        aa,bb,cc,dd=fast(Z,field)[-1]
        v=t-1j*FACTOR*field[0]*x
        s=np.sum(w*(x.conj()*v).real)/xx
        return aa+bb*s
    cs=np.linspace(1.,8.3,70);vals=np.array([sample(c) for c in cs])
    results=[]
    for i in np.flatnonzero(vals[1:]*vals[:-1]<=0):
        c3=brentq(sample,cs[i],cs[i+1])
        # Back-transfer the specimen waist, taking the finite-source emittance into account.
        def eq(q,ret=False):
            obj2,mini2,c32=q
            f,p=atlas(tl,obj2,c32,mini2)
            pars=plane(p)
            if pars is None:return [5,5,5]
            z2=pars[0];idx=np.searchsorted(Z,z2)
            psi=fast(Z,f)
            a1,b1,c1,d1=psi[-1]
            aa,bb,cc,dd=fast(np.r_[z2,Z[idx:]],np.r_[np.interp(z2,Z,f),f[idx:]])[-1]
            v=t-1j*FACTOR*f[0]*x
            xv=np.sum(w*(x.conj()*v).real);vv=np.sum(w*abs(v)**2)
            eps=np.sqrt(xx*vv-xv*xv)
            cov=np.array([[xx,xv],[xv,vv]])/eps
            # Match focus by the allowed objective adjustment; suppress axial focusing at the intermediate waist.
            thrust=np.array([[a1,b1],[c1,d1]])
            vout=thrust@cov@thrust.T
            return [vout[0,1]/max(vout[1,1],1e-16)*300,
                    -dd*cc*vout[0,0]**2-bb*aa,80*np.interp(z2,Z,f)]
        optimum=least_squares(eq,[obj,mini,c3],bounds=([-42.,-201.,.01],[42.,448.,8.35]),x_scale=[2.,60.,.4],
                              xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=80)
        if np.linalg.norm(optimum.fun)>5e-5:continue
        o,m,c=optimum.x
        a=assess(lambda t,o,m:atlas(t,o,c,m),src,tl,o,m)
        a.update(tl=tl,mini=m,c3=c,obj=o,residual=optimum.fun.tolist())
        results.append(a)
    return results


if __name__=="__main__":
    a=Atlas(Path("global-7a804e13d71e743b.npz"))
    from material_relay_fields import sha,digest,parent
    field=WORK/(a.dc.stem+"-dc-bore-0.5mm.npz")
    src=gun_sigma(WORK/("common-gun-"+digest(dict(field=sha(field),physical_source=parent()["source_signature"],
                 n=4096,gun_sampler="cartesian-cubic-v2"))+".npz"))
    out=[]
    for tl in (-65.,0.,100.,200.,-120.,260.,65.,-25.,150.):
        for m,o in [(180.,-2.2),(100.,4.05),(-175.,1.7),(-100.,-4.05),(320.,-18.),(400.,-28.)]:
            for result in candidate(a,src,tl,m,o):
                if any(abs(result["tl"]-r["tl"])<.01 and abs(result["obj"]-r["obj"])<.001 for r in out):continue
                print("Waist candidate",result,flush=True)
                out.append(result)
                (WORK/"common-frontier.json").write_text(json.dumps(out,indent=2))
