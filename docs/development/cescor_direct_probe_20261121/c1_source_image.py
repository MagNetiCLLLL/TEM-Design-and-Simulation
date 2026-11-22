"""Test for C1 source conjugacy with tagged emission data and native transfer probes."""

import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar

from condenser_normal import FIELD, POINT, WORK, coils_at, reference, source_digest
from relay_transfer import START, Native, advance
from temsim.physics.beam_statistics import transverse_beam_statistics


def emission_basis(gun):
    r = gun["emission_position_m"][:, :2] @ np.array([1.,1j])*1e9
    directions = gun["emission_direction"]
    u = (directions[:,:2] @ np.array([1.,1j])) / directions[:,2] * 1e3
    e = gun["energy_offset_ev"] - np.average(gun["energy_offset_ev"],weights=gun["weight"])
    # Use nm and mrad in the regression and restore SI in the response.
    return np.column_stack((r,u,r*abs(r)**2,u*abs(u)**2,r*abs(u)**2,u*abs(r)**2,
                            r*r*u.conjugate(),u*u*r.conjugate(),r*e,u*e,r*e*e,u*e*e)), u


def image_search(model, q, gun, design, u):
    f=model.field(q)
    axis=model.m.z
    zreq=np.unique(np.r_[np.arange(450.,560.01,.35),np.arange(560.,840.01,.5)])*.001
    stops=np.unique(np.minimum(np.searchsorted(axis,zreq),len(axis)-1))
    probes=np.zeros((17,5))
    for j,scale in enumerate((1.,.5)):
        d=scale*np.array([1e-7,1e-7,1e-6,1e-6])
        probes[1+8*j:5+8*j,:4]=np.diag(d)
        probes[5+8*j:9+8*j,:4]=-np.diag(d)
    snapshots=advance(model.t,f,probes,np.full(17,np.median(model.en)),
                      np.full(17,np.median(model.phi0)),np.full(17,-1,np.int32),0,stops)
    if any(np.any(lost!=-1) for _,lost in snapshots):
        raise ValueError("The reference transfer probes encountered a stop")
    jout=[]
    for number,scale in enumerate((1.,.5)):
        j=number*8
        pairs=np.array([(p[j+1:j+5,:4]-p[j+5:j+9,:4]).T for p,_ in snapshots])
        jac=pairs/(2*scale*np.array([1e-7,1e-7,1e-6,1e-6]))
        # Axisymmetric complex response for x+iy, including magnetic rotation.
        positions=jac[:,0,:]+1j*jac[:,1,:]
        slopes=jac[:,2,:]+1j*jac[:,3,:]
        jout.append(np.column_stack([.5*(v[:,i]-1j*v[:,i+1]) for v in (positions,slopes) for i in (0,2)]))
    z=axis[stops]
    positions=np.column_stack((gun["x_m"]+1j*gun["y_m"],gun["tx_rad"]+1j*gun["ty_rad"]))
    t=model.t
    settings=[]
    for cut in (100,60,30):
        mask=abs(u)<=np.percentile(abs(u),cut)
        gun_fit=np.linalg.lstsq(design[mask],positions[mask],rcond=1e-12)[0][:2].T
        gun_fit=gun_fit*np.array([1e9,1e3])
        trace=jout[-1]
        ray=trace[:,:2]@gun_fit
        velocity=trace[:,2:]@gun_fit
        # Inside one coarse interval a cubic Hermite interpolation uses the native slopes.
        def interp(i,frac):
            h=z[i+1]-z[i]
            return ((1+2*frac)*(1-frac)**2*ray[i] + frac*(1-frac)**2*h*velocity[i]
                    + frac**2*(3-2*frac)*ray[i+1]+frac**2*(frac-1)*h*velocity[i+1])
        b=abs(ray[:,1])
        local=np.flatnonzero((b[1:-1]<b[:-2])&(b[1:-1]<b[2:]))+1
        planes=[]
        for i in local:
            pieces=[]
            for k in (i-1,i):
                sol=minimize_scalar(lambda a:abs(interp(k,a)[1])**2,bounds=(0.,1.),method="bounded")
                val=interp(k,sol.x)
                pieces.append((abs(val[1]),z[k]+sol.x*(z[k+1]-z[k]),abs(val[0])))
            bmin,where,mag=min(pieces)
            planes.append(dict(z_mm=where*1000,position_response=mag,slope_response_mm=bmin*1e3))
        settings.append(dict(emission_angle_percentile=cut,direction_response_minima=planes))
    # Test with the whole tagged source at the predicted planes. Keep all arrived packets.
    zfit=[v["z_mm"]*.001 for v in settings[0]["direction_response_minima"]]
    nodes=np.unique(np.minimum(np.searchsorted(axis,zfit),len(axis)-1))
    native=[]
    kprev=0
    p=t.input.copy()
    loss=model.loss.copy()
    for k in nodes:
        ((p,loss),)=advance(t,f,p,model.en,model.phi0,loss,kprev,[k])
        kprev=k
        mask=loss==-1
        stat=transverse_beam_statistics(*p[mask,:4].T,weights=t.weights[mask])
        tag_fit=np.linalg.lstsq(design[mask],p[mask,0]+1j*p[mask,1],rcond=1e-12)[0]
        native.append(dict(z_mm=axis[k]*1000,source_count=int(mask.sum()),tagged_position_gain=abs(tag_fit[0])*1e9,
                           tagged_direction_response_mm=abs(tag_fit[1])*1e6,D95_nm=2e9*stat.radius_95_m))
    return dict(fit_subsets=settings,nearest_native_nodes=native,
                max_probe_step_change_A=float(np.max(abs(jout[0][:,0]-jout[1][:,0]))),
                max_probe_step_change_B_mm=float(1e3*np.max(abs(jout[0][:,1]-jout[1][:,1]))))


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("seed")
    arg.add_argument("name")
    args=arg.parse_args()
    ref=reference()
    if args.seed=="baseline":
        q,ni=coils_at(ref,1.542,.26),ref["objective_At"]
    else:
        from illuminated_normal import controls
        rec=json.loads((WORK/args.seed).read_text())
        q,ni=controls(rec["gains"],True),rec["objective_At"]
    m=Native(FIELD,.01875,32768,ni)
    with np.load(POINT/"gun.npz") as g:
        gun=dict(g)
    np.testing.assert_array_equal(gun["emission_ray_id"],m.t.ids)
    basis,u=emission_basis(gun)
    result=image_search(m,q,gun,basis,u)
    result.update(source_digest=source_digest(m.t),source_plane_mm=0.,gun_exit_mm=1000*START,
                  source_checkpoint="Original emission tags and archived gun transfer to 450 mm",
                  currents_A=q.tolist(),objective_At=ni)
    (WORK/(args.name+".json")).write_text(json.dumps(result,indent=2))
    minima=result["fit_subsets"][0]["direction_response_minima"]
    print(args.name,"C1 smallest angular response",min((v for v in minima if v["z_mm"]<=550),
          key=lambda v:v["slope_response_mm"]),"inter-condenser waist",minima[-1],flush=True)
