"""Observe all original packets on one registered mesh and save the actual probe."""

import argparse
from hashlib import sha256
import json

import cupy as cp
import numpy as np

from condenser_normal import FACTOR, FIELD, FOIL, HERE, OWNER, ACTIVE, RADIUS, SURFACE, WORK, spans_mm
from illuminated_focus import column_planes, focus_offsets_nm
from illuminated_normal import controls
from observe import axis_crossings
from plot_internal_trajectories import select_tracks
from refocus_a2 import objective_receipt
from relay_transfer import Native


def capture(seed, name, count, step, table=FIELD, *, axis_core=False):
    rec = json.loads((WORK/seed).read_text())
    gains, ni = rec["gains"], rec["objective_At"]
    q = controls(gains,True)
    observation = column_planes()
    model = Native(table,step,count,ni,observation)
    t = model.t
    row, final, contacts = model.measure(q,details=True)
    field, (front,back) = model.replay(q, FOIL)
    cohort = contacts == -1
    weights = t.weights[cohort]
    if np.unique(t.ids[t.weights>0]).size != count or np.any(t.weights != t.weights[0]):
        raise ValueError("Original distinct equal-current source packets required")
    chosen, band, fill = select_tracks(front[0], dict(loss=contacts))
    keep = band >= 0
    indices, band = chosen[keep], band[keep]
    # Include a small, azimuthally distributed set that actually hit the aperture.
    blocked = np.flatnonzero((front[1]==-1)&(back[1]!=-1))
    if len(blocked):
        arg = np.angle(front[0][blocked,0]+1j*front[0][blocked,1])
        ordered = blocked[np.argsort(arg)]
        clipped = ordered[np.linspace(0,len(ordered)-1,min(12,len(ordered)),dtype=int)]
        indices = np.r_[indices,clipped]
        band = np.r_[band,np.full(len(clipped),-1)]
    w = cp.asarray(weights / weights.sum())
    at = cp.asarray(np.flatnonzero(cohort))
    ids = cp.asarray(indices)
    n = len(t.ids)
    p, en, phi = cp.asarray(t.input).copy(), cp.asarray(model.en), cp.asarray(model.phi0)
    loss, peak, margin = cp.full(n,-1,np.int32), cp.zeros(n), cp.full(n,cp.inf)
    tracks = cp.empty((len(observation),len(indices),4))
    moments = cp.empty((len(observation),8))
    offsets = focus_offsets_nm()
    focal_z = SURFACE + offsets*1e-9
    focal_indices = dict(zip(np.searchsorted(observation,focal_z),range(len(focal_z))))
    profiles = np.empty((len(offsets),5 if axis_core else 4))
    start = 0
    for j,end in enumerate(model.indices(observation)):
        for i in range(start,end,160):
            t.kernel(((n+127)//128,),(128,), (p,en,phi,field,t.ep,t.er,np.int32(len(t.r2)),
                     t.h,t.rad,loss,peak,margin,np.int32(i),np.int32(min(i+160,end)),np.int32(n),t.rest,FACTOR,t.pc))
        tracks[j] = cp.where((loss[ids] == -1)[:,None],p[ids,:4],cp.nan)
        a = p[at,:4]
        c = a - cp.sum(a*w[:,None],axis=0)
        m = (c.T*w) @ c
        moments[j] = cp.stack((cp.sum(w),cp.sqrt(m[0,0]),cp.sqrt(m[1,1]),m[0,2],m[1,3],
                               .5*(m[0,3]+m[1,2]),m[0,1],m[2,3]))
        if j in focal_indices:
            live = loss == -1
            xy = p[live,:2]
            r2 = cp.sum((xy-xy.mean(axis=0))**2,axis=1)
            k = int(np.ceil(.95*len(xy)))-1
            axis_r2 = cp.sum(xy*xy,axis=1)
            values = [2e9*cp.sqrt(cp.partition(r2,k)[k]), 1e9*cp.sqrt(r2.mean()),
                      cp.sum(axis_r2 <= (50e-12)**2),cp.sum(live)]
            if axis_core:
                values.append(cp.sum(axis_r2 <= (50e-9)**2))
            profiles[focal_indices[j]] = cp.asnumpy(cp.stack(values))
        if observation[j] == SURFACE:
            np.testing.assert_array_equal(p.get(),final)
            np.testing.assert_array_equal(loss.get(),contacts)
        start = end
    moments = moments.get()
    spans = spans_mm()*.001
    bounds = dict(C2_C3_gap=[spans[0,1],spans[1,0]],C3_ADL_gap=[spans[1,1],spans[2,0]],
                  C2_C3_centers=[spans[0].mean(),spans[1].mean()],C3_ADL_centers=[spans[1].mean(),spans[2].mean()],
                  Mini_to_presample=[1.5225,SURFACE-.00025])
    waists = {key:axis_crossings(observation,moments,*value) for key,value in bounds.items()}
    all_angles = []
    for az in np.linspace(0,np.pi,36,endpoint=False):
        cs, sn = np.cos(az), np.sin(az)
        covariance = cs*cs*moments[:,3] + sn*sn*moments[:,4] + 2*sn*cs*moments[:,5]
        probe_moments = np.column_stack([moments[:,:3],covariance,covariance])
        all_angles.append({k:axis_crossings(observation,probe_moments,*v)[0] for k,v in bounds.items()})
    footprint = np.maximum(fill,np.linalg.norm(back[0][:,:2],axis=1)/RADIUS)
    taz = np.angle(front[0][:,0]+1j*front[0][:,1])
    hist = np.histogram(taz[cohort & (footprint>=.95)],bins=np.linspace(-np.pi,np.pi,17))[0]
    row.update(gains=gains,objective_At=ni,objective_material_sha256=objective_receipt(ni)["curve_sha256"],
               geometry_mm=spans_mm().tolist(),gun_z_mm=float(observation[0]*1e3),surface_z_mm=SURFACE*1000,
               source_current_A=t.current,waists_mm=waists,azimuth_waists_mm=all_angles,
               outer_5pct_rim_16_sector_counts=hist.tolist(),case=name,
               maximum_foil_incident_radius_um=float(fill[front[1]==-1].max()*RADIUS*1e6),
               maximum_passing_foil_radius_um=float(footprint[cohort].max()*RADIUS*1e6),
               corrector_sha256=sha256(table.read_bytes()).hexdigest())
    for key,col in (("D95",0),("RMS",1)):
        i = np.argmin(profiles[:,col])
        row[key+"_minimum_bracket_nm"]=offsets[max(0,i-1):min(len(offsets),i+2)].tolist()
        row[key+"_minimum_value_nm"]=float(profiles[i,col])
    center = np.flatnonzero(offsets==0).item()
    np.testing.assert_allclose(profiles[center,0],row["specimen"]["D95_nm"],rtol=5e-12)
    if axis_core:
        np.testing.assert_allclose(profiles[center,4]*t.current*t.weights[0]*1e6,
                                   row["specimen"]["axis_50nm_uA"], rtol=1e-12)
        row["profile_columns"] = ["centroid_D95_nm", "centroid_RMS_radius_nm", "axis_50pm_count",
                                  "survivor_count", "axis_50nm_count"]
    row["focus_scan_survivors_min_max"]=[int(profiles[:,3].min()),int(profiles[:,3].max())]
    c3 = OWNER[ACTIVE] == "condenser_lens_3"
    self_field = model.coils[c3,:,0].T @ q[c3]
    row["C3_lens_peak_self_T"]=float(abs(self_field[(model.m.nodes>=spans[1,0])&(model.m.nodes<=spans[1,1])]).max())
    row["winding_owner"] = OWNER[ACTIVE].tolist()
    (WORK/(name+".json")).write_text(json.dumps(row,indent=2,allow_nan=False),encoding="utf-8")
    np.savez_compressed(WORK/(name+".npz"),source_id=t.ids,weight=t.weights,energy_offset_eV=t.de,
                        gun_exit_phase=t.input,front_phase=front[0],front_loss=front[1],back_phase=back[0],
                        back_loss=back[1],specimen_phase=final,specimen_loss=contacts,
                        loss_component=np.where(contacts>=0,model.m.keys[model.m.owner[np.maximum(contacts,0)]],""),
                        integration_z_m=model.m.z,track_z_m=observation,tracks=tracks.get(),track_source_id=t.ids[indices],
                        track_band=band,track_fill=fill[indices],moments=moments,offsets_nm=offsets,profiles=profiles)
    print("Waists",waists,"D95 bracket",row["D95_minimum_bracket_nm"],"sample",row["specimen"],
          "edge",row["rim_azimuth_check"]["angle_mrad_min_median_max"],"sectors",hist.tolist(),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("seed")
    parser.add_argument("name")
    parser.add_argument("--count",type=int,default=32768)
    parser.add_argument("--step",type=float,default=.075)
    args=parser.parse_args()
    capture(args.seed,args.name,args.count,args.step)
