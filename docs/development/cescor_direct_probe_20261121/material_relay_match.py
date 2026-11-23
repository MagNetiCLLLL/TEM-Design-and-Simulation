"""Match a finite-emittance source to two true waists using the complex linear transfer."""
import json
import sys

import numpy as np
from scipy.optimize import least_squares

from material_relay_branches import FACTOR, RoundMaps, SAVE, WORK, Z, axis, fast, gun_sigma, assess


def fit(fields,src,p0,mini,beta_s,bounds):
    x,t,w=src
    def evaluate(p,output=False):
        tl,obj,c3=p
        full,pair=fields(tl,obj,c3,mini)
        a,b,c,d=fast(Z,pair).T
        dd=-c[-1]*b+d[-1]*a
        ind=np.flatnonzero((Z[:-1]>1.420)&(Z[:-1]<1.512)&(dd[:-1]*dd[1:]<=0))
        if not len(ind):return np.full(3,3e2)
        i=ind[np.argmin(abs(Z[ind]-1.46))]
        zf=Z[i]-(Z[i+1]-Z[i])*dd[i]/(dd[i+1]-dd[i])
        A,B,C,D=fast(Z,full).T
        af,bf,cf,df=[np.interp(zf,Z,v) for v in (A,B,C,D)]
        at,bt,ct,dt=A[-1]*df-B[-1]*cf,-A[-1]*bf+B[-1]*af,C[-1]*df-D[-1]*cf,-C[-1]*bf+D[-1]*af
        # Back-transfer a specimen waist of beta_s to the measured front focal plane.
        f_cov=-dt*ct*beta_s-bt*at/beta_s
        v=t-1j*FACTOR*full[0]*x
        xx=np.sum(w*abs(x)**2);xv=np.sum(w*(x.conj()*v).real);vv=np.sum(w*abs(v)**2)
        eps=np.sqrt(xx*vv-xv*xv)
        vrel=xv/xx+1j*eps/xx
        psi=(A[-1]+B[-1]*vrel)*np.sqrt(xx/eps)
        angle=(C[-1]+D[-1]*vrel)*np.sqrt(xx/eps)
        match=(angle-1j*psi/beta_s)*np.sqrt(beta_s)
        if not output:return np.array([match.real,match.imag,f_cov])
        rec=assess(lambda t,o,m:fields(t,o,c3,m),src,tl,obj,mini)
        return dict(tl=tl,obj=obj,c3=c3,mini=mini,beta_s=beta_s,emittance=eps,**rec)
    solution=least_squares(evaluate,np.clip(p0,np.array(bounds[0])+.00001,np.array(bounds[1])-.00001),
                bounds=bounds,x_scale=[100,20,2],max_nfev=50,xtol=1e-11,ftol=1e-11,gtol=1e-11)
    if "--debug" in sys.argv:
        print("Fit",p0,solution.x,solution.fun,flush=True)
    if np.linalg.norm(solution.fun)<2e-4:return evaluate(solution.x,True)


def main():
    maps=RoundMaps();ref=next(WORK.glob("circuit-ampere-turns-condenser_lens_36.710114-mini_condenser*-all--28.22027595A-1mm.npz"))
    off=next(WORK.glob("circuit-ampere-turns-condenser_lens_30-mini_condenser*-all--28.22027595A-1mm.npz"))
    q0=6.710114018150848;delta=(axis(ref)-axis(off))/q0
    def fields(t,o,c,m):
        total,pair=maps(t,o,m)
        return total+(c-q0)*delta,pair
    src=gun_sigma(json.loads((SAVE/"optical-start/tracking.json").read_text())["gun_source"])
    out=[];seen=set()
    for mini in (239.912653194986,140.,80.,0.,-120.,350.):
        for beta in (.025,.06,.012,.004,.1,.002):
            for t,o,c in ((-70,-30,7.2),(100,-32,7.15),(250,-32,7.10),(-90,-17,7.2),(-40,-28,5.)):
                a=fit(fields,src,[t,o,c],mini,beta,([-124.,-44.,.01],[276.5,-12.,8.]))
                if not a or (mini,beta,round(a["tl"],1)) in seen:continue
                seen.add((mini,beta,round(a["tl"],1)))
                print("Matched",a,flush=True)
                out.append(a)
        (WORK/"twiss-solutions.json").write_text(json.dumps(out,indent=2))


if __name__=="__main__":main()
