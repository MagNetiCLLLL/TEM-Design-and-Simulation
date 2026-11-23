"""Screen winding proposals in the Larmor frame. Final trials require a new material solve."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.constants import c, e, m_e
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2] / "outputs/material-relay-20261122"
SEED = "global-23b8b2425e729ecb"
KEYS = ["c1", "c2", "c3", "adl", "tl", "mini", "objective"]
REFERENCE = np.array([-30., 16., 7.114073222824603, 10., -118.24704018784674,
                      226.70154583016074, -10.854710898968472])
LOW = np.array([-58., 4., 0., -110., -280., -40., -58.])
HIGH = np.array([-13., 32., 12., 110., 130., 440., 60.])
FACTOR = e*c/(2*np.sqrt(300000*e*(300000*e+2*m_e*c*c)))
Z = np.linspace(.45, 1.5991999975, 1841)
DZ = Z[1]-Z[0]


def load(folder, tangent):
    trial = json.loads((folder/"tracking.json").read_text())
    exact = Path(trial["solution"]).stem
    with np.load(trial["gun_source"]) as source:
        w = source["weight"]
        g = source["phase"]
        x, v = g[:, 0]+1j*g[:, 1], g[:, 2]+1j*g[:, 3]
        x -= w@x
        v -= w@v
    fields = []
    for group in ["dc", *KEYS]:
        name = f"{exact}-dc-bore-0.25mm.npz" if group=="dc" else f"{tangent}-{group}-bore-0.75mm.npz"
        with np.load(WORK/name) as f:
            spl = CubicSpline(f["z"], f["coef"][:, 0, 2])
            fields.append(spl(Z))
    u = v-1j*FACTOR*fields[0][0]*x
    var, cov, sv = w@abs(x)**2, w@(x.conj()*u).real, w@abs(u)**2
    initial = np.array([cov/var, np.sqrt(max(0., var*sv-cov**2))/var, np.sqrt(var)])
    ref = REFERENCE.copy()
    for i,k in enumerate(KEYS):
        if k in trial["currents"]:
            ref[i]=trial["currents"][k]
    return np.array(fields).T.copy(), initial, ref


@njit(cache=True)
def trace(p, field, initial, ref):
    # Real and imaginary parts propagate the measured radial second moments.
    a, b, ap, bp = 1., 0., initial[0], initial[1]
    last = a*ap+b*bp
    roots = np.zeros(12)
    count = 0
    rim, peak, hp, scan = 0., 0., 0., 0.
    for j in range(len(Z)-1):
        bv = .5*(field[j,0]+field[j+1,0])
        for k in range(len(p)):
            bv += .5*(field[j,k+1]+field[j+1,k+1])*(p[k]-ref[k])
        omega = FACTOR*bv
        # Exact constant-field propagation on each short slice.
        angle = omega*DZ
        co = np.cos(angle)
        sn = np.sin(angle)/omega if abs(omega)>1e-10 else DZ
        a, ap = co*a+sn*ap, co*ap-omega*omega*sn*a
        b, bp = co*b+sn*bp, co*bp-omega*omega*sn*b
        cv = a*ap+b*bp
        if last < 0 <= cv:
            if count < len(roots):
                roots[count] = (Z[j]-DZ*last/(cv-last))*1000
            count += 1
        last = cv
        if abs(Z[j+1]-.765)<DZ*.51:
            rim = np.hypot(a,b)*initial[2]*1e6
        if Z[j+1]>1.51:
            peak = max(peak,np.hypot(a,b)*initial[2]*1e6)
        if Z[j] <= 1.20815 < Z[j+1]:
            hp = a-(Z[j+1]-1.20815)*ap
        if Z[j] <= 1.444 < Z[j+1]:
            scan = a-(Z[j+1]-1.444)*ap
    return (roots,count,a,b,ap,bp,rim,peak,hp,scan)


@njit(cache=True)
def score(p, field, initial, ref):
    roots,n,a,b,ap,bp,rim,peak,hp,scan = trace(p,field,initial,ref)
    # A near-surface fifth waist counts as the terminal image.
    if n and n<=len(roots) and roots[n-1]>1588:
        n-=1
    limits = ((450.,610.),(750.,830.),(1179.1,1237.2),(1420.,1512.5))
    penalty = 4.*abs(n-4)
    for i in range(min(n,4)):
        lo,hi = limits[i]
        penalty += .06*max(0.,lo-roots[i],roots[i]-hi)
    pos = np.hypot(a,b)
    slope = np.hypot(ap,bp)
    # Estimate displacement of the near-surface image, in mm.
    defocus = pos/max(slope,1e-12)*1000
    return penalty+np.log1p(defocus/.4)-np.log(max(slope,1e-12))


@njit(parallel=True,cache=True)
def batch(ps,field,initial,ref):
    out=np.empty(len(ps))
    for i in prange(len(ps)):
        out[i]=score(ps[i],field,initial,ref)
    return out


def search(wide,folder,tangent):
    set_num_threads(min(14,get_num_threads()))
    f,ic,ref=load(folder,tangent)
    rng=np.random.default_rng(11022)
    samples=qmc.Sobol(7,scramble=True,seed=11022).random_base2(15)
    lo,hi=(np.array([-100.,4.,0.,-450.,-900.,-800.,-130.]),
           np.array([-13.,65.,12.,450.,300.,1000.,130.])) if wide else (LOW,HIGH)
    candidates=lo+(hi-lo)*samples
    candidates[0]=ref
    candidates[1:4097]=np.clip(ref+rng.normal(size=(4096,7))*(HIGH-LOW)*.055,lo,hi)
    if wide:
        earlier=json.loads((HERE/"paraxial-proposals.json").read_text())["candidates"]
        for i,row in enumerate(earlier):
            v=np.array([row["currents"][k] for k in KEYS])
            candidates[4100+256*i:4356+256*i]=np.clip(v+rng.normal(size=(256,7))*(hi-lo)*.015,lo,hi)
    for scale in [.10,.07,.05,.035,.02,.012,.007]:
        scores=batch(candidates,f,ic,ref)
        order=np.argsort(scores)
        elite=candidates[order[:128]]
        candidates=np.vstack((elite,np.clip(elite[rng.integers(len(elite),size=10000)]
                 +rng.normal(size=(10000,7))*(hi-lo)*scale,lo,hi)))
    order=np.argsort(batch(candidates,f,ic,ref))
    best=[]
    for idx in order[:100]:
        v=candidates[idx].copy()
        # Shoot the objective to the fixed specimen plane without moving other supplies.
        def distance(i):
            v[-1]=i[0]
            _,_,a,b,ap,bp,*_=trace(v,f,ic,ref)
            return [(a*ap+b*bp)/max(1e-12,ap*ap+bp*bp)*1000]
        sol=least_squares(distance,[v[-1]],bounds=(lo[-1],hi[-1]),xtol=1e-11)
        v[-1]=sol.x[0]
        roots,n,a,b,ap,bp,rim,peak,hp,scan=trace(v,f,ic,ref)
        valid=roots[:min(n,12)]
        interm=valid[valid<1599.17]
        if (len(interm)!=4 or not (450<interm[0]<610 and 750<interm[1]<830 and
                1179.1<interm[2]<1237.2 and 1420<interm[3]<1512.5)):
            continue
        best.append(dict(currents=dict(zip(KEYS,v.tolist())),rms_nm=ic[2]*np.hypot(a,b)*1e9,
               rms_angle_mrad=ic[2]*np.hypot(ap,bp)*1e3,waists_mm=valid.tolist(),
               aperture_rms_um=rim,final_section_peak_rms_um=peak))
    best.sort(key=lambda q:q["rms_nm"])
    print("Best estimates",json.dumps(best[:3]),flush=True)
    output="wide-proposals.json" if wide else "paraxial-proposals.json"
    output_path=HERE/output if folder.name=="objective-only" else folder/output
    output_path.write_text(json.dumps(dict(
            screening_only=True,reference=ref.tolist(),current_bounds_A=[lo.tolist(),hi.tolist()],
            preserved_supplies=["HP1","HP2","TL21","TL22"],candidates=best[:12]),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide",action="store_true")
    parser.add_argument("--from-folder",type=Path,default=HERE/"objective-only")
    parser.add_argument("--tangent",default=SEED)
    args=parser.parse_args()
    search(args.wide,args.from_folder,args.tangent)
