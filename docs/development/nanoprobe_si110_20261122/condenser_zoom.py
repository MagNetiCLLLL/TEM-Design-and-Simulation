"""Find single-image three-condenser settings before solving the full biased magnetic circuit."""
import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from scipy.interpolate import CubicSpline
from scipy.optimize import root as solve_root

from match import transfer
from screen import FACTOR, HERE, WORK


CHANNELS = ("c1", "c2", "c3", "objective")
PACKS = ("condenser_lens_1/physical-pack-0", "condenser_lens_2/physical-pack-1",
         "condenser_lens_3/physical-pack-2", "objective_lens/physical-pack-4")
SURFACE = 1.5991999975
ANCHOR = .375


def read_axis(path):
    with np.load(path) as data:
        return CubicSpline(data["z"], data["coef"][:, 0, 2])


def prepare(folder, tangent, pitch_mm=.4, planes=()):
    record = json.loads((folder/"tracking.json").read_text())
    bias = np.array([record["field"]["bias"][key] for key in PACKS])
    solution = Path(record["solution"])
    dc = read_axis(WORK/f"{solution.stem}-dc-bore-{record['bore_step_mm']:g}mm.npz")
    derivatives = [read_axis(WORK/f"{tangent}-{key}-bore-0.75mm.npz") for key in CHANNELS]
    with np.load(record["gun_source"]) as gun:
        w = gun["weight"]
        if np.count_nonzero(w > 0) < 1000:
            raise ValueError("A finite gun-exit measurement requires at least 1000 source particles")
        phase = gun["phase"]
        xy = phase[:, 0]+1j*phase[:, 1]
        vel = phase[:, 2]+1j*phase[:, 3]
        xy -= np.average(xy, weights=w)
        vel -= np.average(vel, weights=w)
        velocity = vel-1j*FACTOR*dc(.450)*xy
        cov = np.array([[np.average(abs(xy)**2,weights=w), np.average((xy.conj()*velocity).real,weights=w)],
                        [np.average((xy.conj()*velocity).real,weights=w), np.average(abs(velocity)**2,weights=w)]])
    z = np.unique(np.r_[np.linspace(ANCHOR,SURFACE,1+round((SURFACE-ANCHOR)*1000/pitch_mm)),
                        .45,.7649,1.0099,1.5225,1.5345,SURFACE,planes])
    mid = (z[1:]+z[:-1])*.5
    baseline = transfer(z,FACTOR*dc(mid))
    to_anchor = np.linalg.inv(baseline[np.searchsorted(z,.45)].reshape(2,2))
    root = to_anchor @ np.linalg.cholesky(cov)
    field = np.ascontiguousarray(FACTOR*np.stack([dc(mid),*[d(mid) for d in derivatives]],axis=1))
    return z, field, np.array([root[0,0],root[0,1],root[1,0],root[1,1]]), bias


@njit(cache=True)
def evaluate(drive, z, field, root, reference):
    a,b,va,vb = root
    cov = a*va+b*vb
    extrema = np.zeros(8)
    n = 0
    foil = mini = objective = 0.
    for j in range(len(z)-1):
        h = z[j+1]-z[j]
        k = field[j,0]
        for d in range(4):
            k += (drive[d]-reference[d])*field[j,d+1]
        co = np.cos(k*h)
        sn = np.sin(k*h)/k if abs(k)>1e-12 else h
        a,va = co*a+sn*va,co*va-k*k*sn*a
        b,vb = co*b+sn*vb,co*vb-k*k*sn*b
        new_cov = a*va+b*vb
        if cov<0.<=new_cov and z[j]>=.45:
            if n<len(extrema):
                extrema[n] = 1000*(z[j]-h*cov/(new_cov-cov))
            n += 1
        cov = new_cov
        if z[j+1]==.7649:
            foil = np.hypot(a,b)*1e6
        if z[j+1]==1.5225:
            mini = np.hypot(a,b)*1e6
        if z[j+1]==1.5345:
            objective = np.hypot(a,b)*1e6
    slope2 = va*va+vb*vb
    return extrema, n, np.sqrt(slope2)*1e3, np.hypot(a,b)*1e9, 1000*cov/slope2, foil, mini, objective


@njit(cache=True,parallel=True)
def search_table(drives,z,field,root,ref):
    out = np.empty((len(drives),13))
    for i in prange(len(drives)):
        cross,n,ang,r,focus,foil,mini,obj = evaluate(drives[i],z,field,root,ref)
        out[i,:6] = cross[:6]
        out[i,6:] = np.array([float(n),ang,r,focus,foil,mini,obj])
    return out


def search(folder, tangent, c1_min, c1_max, tag):
    z,field,root,ref = prepare(folder,tangent)
    print("Reference matrix prediction",evaluate(ref,z,field,root,ref),flush=True)
    set_num_threads(min(14,get_num_threads()))
    grid = np.meshgrid(np.arange(c1_min,c1_max+.01,2),np.arange(4.,46.01,.4),
                       np.arange(0.,8.01,.25),indexing="ij")
    drives = np.column_stack([g.ravel() for g in grid]+[np.full(grid[0].size,ref[3])])
    data = search_table(drives,z,field,root,ref)
    # Count four intermediate images and permit an objective focus close to either side of the surface.
    ok = ((data[:,6]==4)|((data[:,6]==5)&(data[:,4]>1589.))) & (data[:,0]<524.72) & (
        (data[:,1]>750.)&(data[:,1]<830.)&(data[:,2]>1179.1)&(data[:,2]<1237.2)&
        (data[:,3]>1420.)&(data[:,3]<1512.5))
    rows=[]
    for c1 in np.unique(drives[:,0]):
        eligible = np.flatnonzero(ok & (drives[:,0]==c1))
        if not len(eligible):
            continue
        # An angular branch close to the physical entrance can be focused with the series objective.
        order = np.argsort(-data[eligible,7])
        best=eligible[order[0]]
        close=eligible[np.abs(data[eligible,9])<4.]
        focused=close[np.argmax(data[close,7])] if len(close) else None
        for kind,i in (("angle",best),("near-focus",focused)):
            if i is not None:
                rows.append(dict(kind=kind,currents_A=drives[i].tolist(),roots_mm=data[i,:int(min(6,data[i,6]))].tolist(),
                                 sample_rms_angle_mrad=data[i,7],sample_rms_nm=data[i,8],
                                 surface_offset_mm=data[i,9],foil_rms_um=data[i,10],mini_rms_um=data[i,11],
                                 upper_pole_rms_um=data[i,12]))
    base=HERE/"c1-zoom"
    base.mkdir(exist_ok=True)
    (base/(tag+".json")).write_text(json.dumps(dict(source=folder.name,tangent=tangent,grid_size=len(drives),
         angular_candidates=int(ok.sum()),channels=CHANNELS,rows=rows),indent=2)+"\n")
    print("Screened",len(drives),"eligible",int(ok.sum()),flush=True)
    for row in rows:
        if row["kind"]=="near-focus":
            print(row["currents_A"],row["sample_rms_angle_mrad"],np.round(row["roots_mm"],1),flush=True)


@njit(cache=True)
def conjugate_residual(drive,z,field,initial,reference,plane_index):
    # The dominant correlated source mode vanishes at a first-order source image.
    a,va = initial[0],initial[2]
    rms0 = np.hypot(initial[0],initial[1])
    waist = 0.
    for j in range(len(z)-1):
        h=z[j+1]-z[j]
        k=field[j,0]
        for i in range(4):
            k+=(drive[i]-reference[i])*field[j,i+1]
        co=np.cos(k*h)
        sn=np.sin(k*h)/k if abs(k)>1e-12 else h
        a,va=co*a+sn*va,co*va-k*k*sn*a
        if j+1==plane_index:
            waist=a/rms0
    return np.array([waist,a/rms0])


@njit(cache=True)
def meridional_shots(drive,z,field,initial,reference,indices):
    a,va=initial[0],initial[2]
    values=np.zeros(len(indices))
    slopes=np.zeros(len(indices))
    p=0
    for j in range(len(z)-1):
        h=z[j+1]-z[j]
        k=field[j,0]+np.dot(drive-reference,field[j,1:])
        co=np.cos(k*h)
        sn=np.sin(k*h)/k if abs(k)>1e-12 else h
        a,va=co*a+sn*va,co*va-k*k*sn*a
        if j+1==indices[p]:
            values[p],slopes[p]=a,va
            p+=1
            if p==len(indices):
                break
    return values,slopes


def propose(folder,tangent,c1,c2_mm,post_mm,objective_guess=None,offsets=(0.,0.,0.)):
    target=np.array([c2_mm,post_mm,SURFACE*1000])*.001
    z,field,initial,ref=prepare(folder,tangent,.2,planes=target)
    ix=np.searchsorted(z,target)
    def distance(p):
        drive=np.r_[c1,p]
        a,v=meridional_shots(drive,z,field,initial,ref,ix)
        return (a-v*np.asarray(offsets)*.001)/np.hypot(initial[0],initial[1])
    sol=solve_root(distance,[ref[1],ref[2],ref[3] if objective_guess is None else objective_guess])
    if not sol.success or max(abs(sol.fun))>1e-5:
        return None
    drive=np.r_[c1,sol.x]
    crossing,n,angular,rad,offset,foil,mini,pole=evaluate(drive,z,field,initial,ref)
    return dict(currents_A=drive.tolist(),roots_mm=crossing[:min(n,len(crossing))].tolist(),
                rms_angle_mrad=angular,mini_rms_um=mini,surface_rms_nm=rad,surface_offset_mm=offset)


def objective_branches(folder,tangent,post_mm):
    plane=post_mm*.001
    z,field,_,ref=prepare(folder,tangent,.3,planes=[plane])
    at=np.searchsorted(z,plane)
    z=z[at:]; field=field[at:]
    ix=np.array([len(z)-1])
    initial=np.array([0.,0.,1.,0.])
    def endpoint(amps):
        drive=ref.copy();drive[3]=amps
        return meridional_shots(drive,z,field,initial,ref,ix)[0][0]
    from scipy.optimize import brentq
    trial=np.linspace(-180.,180.,1441)
    val=np.array([endpoint(k) for k in trial])
    output=[]
    for i in np.flatnonzero(val[1:]*val[:-1]<0.):
        amp=brentq(endpoint,trial[i],trial[i+1])
        drive=ref.copy();drive[3]=amp
        b,d=meridional_shots(drive,z,field,initial,ref,np.arange(1,len(z)))
        earlier=int(np.count_nonzero(b[:-2]*b[1:-1]<0.))
        output.append(dict(objective_A=amp,D=d[-1],intermediate_images=earlier))
    return output


def shoot(folder,tangent,c1_min,c1_max,tag,post_mm,reverse):
    plane=post_mm*.001
    z,field,initial,ref=prepare(folder,tangent,planes=[plane])
    target=np.searchsorted(z,plane)
    results=[]
    c2_values=np.arange(4.,33.,.5)
    if reverse:
        c2_values=np.r_[-c2_values[::-1],c2_values]
    for c1 in np.arange(c1_min,c1_max+.01,2.):
        for c2 in c2_values:
            def residual(pair):
                drive=np.array([c1,c2,pair[0],pair[1]])
                return conjugate_residual(drive,z,field,initial,ref,target)
            initial_guesses=[[6.7,ref[-1]],[3.,ref[-1]],[.5,ref[-1]]]
            if reverse:
                initial_guesses.extend([[-3.,ref[-1]],[-7.,ref[-1]]])
            accepted=[]
            for guess in initial_guesses:
                sol=solve_root(residual,guess)
                c3,obj=sol.x
                if not sol.success or np.linalg.norm(sol.fun)>1e-6 or not(-8.0 if reverse else 0.)<=c3<=8.0 or abs(obj)>55.:
                    continue
                if any(np.linalg.norm(sol.x-prior)<1e-3 for prior in accepted):
                    continue
                accepted.append(sol.x)
                drive=np.array([c1,c2,c3,obj])
                crossing,n,angle,radius,offset,foil,mini,pole=evaluate(drive,z,field,initial,ref)
                intermediate=crossing[:min(len(crossing),n)]
                intermediate=intermediate[intermediate<1599.05]
                if len(intermediate)!=4:
                    continue
                x1,x2,x3,x4=intermediate
                if not (450.<x1<524.72 and 750.<x2<830. and 1179.1<x3<1237.2 and 1420.<x4<1512.5):
                    continue
                results.append(dict(currents_A=drive.tolist(),roots_mm=intermediate.tolist(),sample_rms_angle_mrad=angle,
                                    sample_rms_nm=radius,foil_rms_um=foil,mini_rms_um=mini,upper_pole_rms_um=pole))
    results.sort(key=lambda a:-a["sample_rms_angle_mrad"])
    base=HERE/"c1-zoom"
    base.mkdir(exist_ok=True)
    meta=dict(source=folder.name,tangent=tangent,post_target_mm=post_mm,channels=CHANNELS,rows=results)
    (base/(tag+".json")).write_text(json.dumps(meta,indent=2)+"\n")
    for key in np.unique([row["currents_A"][0] for row in results]):
        row=next(q for q in results if q["currents_A"][0]==key)
        print(np.round(row["currents_A"],3),np.round(row["roots_mm"],2),round(row["sample_rms_angle_mrad"],6),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--from-folder",type=Path,default=HERE/"relay-edge")
    cli.add_argument("--tangent",default="global-23b8b2425e729ecb")
    cli.add_argument("--c1-min",type=float,default=-84.)
    cli.add_argument("--c1-max",type=float,default=-32.)
    cli.add_argument("--tag",default="condenser-survey")
    cli.add_argument("--shoot",action="store_true")
    cli.add_argument("--post-mm",type=float,default=1424.7702)
    cli.add_argument("--include-reverse",action="store_true")
    args=cli.parse_args()
    if args.shoot:
        shoot(args.from_folder.resolve(),args.tangent,args.c1_min,args.c1_max,args.tag,args.post_mm,args.include_reverse)
    else:
        search(args.from_folder.resolve(),args.tangent,args.c1_min,args.c1_max,args.tag)
