"""Find the two-sided geometrical caustic across the fixed, unscattered specimen entrance."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from column_current import ApertureTracker
from condenser_pair_column import specimen
from condenser_source_fields import write
from material_relay_column import Column,Field,source_and_electrodes
from material_relay_fields import SAVE
from relay_transfer import advance
from stationary_pupil import half_nodes
from trace_normal_joint_field import beam_moments


def check(path,folder):
    record=json.loads((folder/"tracking.json").read_text())
    field=Field(path,step=record["field"]["step_mm"])
    f=source_and_electrodes()
    full=Column(f,field,Path(record["gun_source"]),record["count"],record["step_mm"])
    with np.load(folder/"column-and-landings.npz") as saved:
        phase,mask=saved["surface_phase"],saved["loss"]==-1
        zz=saved["z_mm"];before=zz>=zz[-1]-.40001
        stats=saved["stats"][before]
        st=np.column_stack((stats[:,1:3].sum(axis=1),stats[:,3:5]))
    start=record["surface_z_mm"]*.001
    z=start+np.arange(81)*.000005
    nodes=half_nodes(z);e=f.electric.base_field
    idx=np.clip(np.searchsorted(e.z,nodes,side="right")-1,0,len(e.z)-2)
    dz=np.diff(e.z)[idx]
    mesh=SimpleNamespace(z=z,nodes=nodes,radius=np.full(len(z)-1,np.inf),owner=np.full(len(z)-1,-1),
            keys=[],electric=(e.r,e.voltage,idx,(nodes-e.z[idx])/dz,dz))
    table=dict(z_local_m=field.localz,xy_m=field.xy,B_zyx_T=field.table(),valid_radius_m=field.receipt["local_radius_m"])
    t=ApertureTracker(mesh,table,"cubic",source=full.source)
    stops=np.arange(1,len(z));lost=np.zeros(len(mask),np.int32);lost[mask]=-1
    result=advance(t,cp.zeros((len(nodes),20)),phase,full.en,full.phi0,lost,0,stops)
    rows=record["focus_profile"].copy();new=[]
    for k,(p,loss) in enumerate(result,1):
        live=loss==-1
        if not np.array_equal(live,mask):raise ValueError("Changed identities in unscattered continuation")
        s=specimen(p[live],t.weights[live],t.current)
        sm=beam_moments(p,live,t.weights)
        new.append([sm[1]+sm[2],sm[3],sm[4]])
        rows.append(dict(z_mm=z[k]*1000,D95_um=s["D95_um"],rms_um=s["RMS_radius_um"],
                    core_50nm_nA=s["axis_50nm_nA"],core_50pm_pA=s["axis_50pm_pA"]))
    absc=np.r_[zz[before],z[1:]*1000]
    roots=[]
    for covariance in np.vstack((st,new)).T:
        spline=CubicSpline(absc,covariance)
        cross=np.flatnonzero((covariance[:-1]<0)&(covariance[1:]>=0))
        roots.append([float(brentq(spline,absc[i],absc[i+1])) for i in cross])
    # Replay to a saved observation upstream, then cross the entrance on a uniform
    # one-micrometre grid. The physical entrance phase is an independent closure check.
    j=int(full.indices([start-.00006])[0])
    (initial,initial_loss),=advance(full.t,full.full,full.t.input,full.en,full.phi0,full.loss,0,[j])
    np.testing.assert_allclose(full.m.z[j],start-.00006,rtol=0,atol=2e-12)
    fine_z=np.linspace(full.m.z[j],start+.00006,121)
    fine_nodes=half_nodes(fine_z)
    ix=np.clip(np.searchsorted(e.z,fine_nodes,side="right")-1,0,len(e.z)-2)
    fdz=np.diff(e.z)[ix]
    fine_mesh=SimpleNamespace(z=fine_z,nodes=fine_nodes,radius=np.full(120,np.inf),owner=np.full(120,-1),keys=[],
                             electric=(e.r,e.voltage,ix,(fine_nodes-e.z[ix])/fdz,fdz))
    tracer=ApertureTracker(fine_mesh,table,"cubic",source=full.source)
    fine_field=cp.zeros((len(fine_nodes),20))
    fine_rows=[];closure=None;previous=0
    for group in np.array_split(np.arange(1,len(fine_z)),12):
        native=advance(tracer,fine_field,initial,full.en,full.phi0,initial_loss,previous,group)
        for stop,(initial,initial_loss) in zip(group,native,strict=True):
            live=initial_loss==-1
            np.testing.assert_array_equal(live,mask)
            s=specimen(initial[live],tracer.weights[live],tracer.current)
            fine_rows.append(dict(z_mm=fine_z[stop]*1000,D95_um=s["D95_um"],rms_um=s["RMS_radius_um"],
                        core_50nm_nA=s["axis_50nm_nA"],core_50pm_pA=s["axis_50pm_pA"]))
            if stop==60:
                closure=float(np.linalg.norm(initial[live,:2]-phase[live,:2],axis=1).max())
                if closure>1e-13:raise ValueError("Refined focus replay disagrees with original specimen landings")
        previous=group[-1]
    rows=sorted([row for row in rows if abs(row["z_mm"]-start*1000)>.06001]+fine_rows,key=lambda row:row["z_mm"])
    best=min(rows,key=lambda r:r["D95_um"])
    write(folder/"surface-focus.json",dict(surface_mm=1000*start,trace_after_surface="Vacuum continuation without specimen scattering",
          window_mm=[float(absc[0]),float(absc[-1])],radial_waists_mm=roots[0],principal_waists_mm=roots[1:],D95_min_sampled=best,
          D95_min_bracketed_by_samples=rows[0] is not best and rows[-1] is not best,entrance_replay_max_position_error_m=closure,
          D95_min_offset_um=1000*(best["z_mm"]-start*1000),profile=rows))
    print("Near-entrance waists (mm)",roots,"minimum sampled D95",best,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("solution",type=Path);p.add_argument("--label",required=True)
    a=p.parse_args();check(a.solution,SAVE/a.label)
