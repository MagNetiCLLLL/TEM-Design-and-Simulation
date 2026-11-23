"""Relaunch the declared source in a newly solved round field and trace its original particles."""

import argparse
import json

import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline

from condenser_pair_column import specimen
from condenser_pair_render import draw_probe, draw_rays
from condenser_source_fields import FEM, FINAL, GunMagnet, MAT, sha, write
from joint_relay_search import mono_roots
from material_relay_fields import PARENT, SAVE, WORK, RoundAssembly, digest, parent
from material_relay_tracking import Trial
from refined_objective_normal import RefinedObjectiveFields
from relay_transfer import advance
from temsim.optics.electron_gun.tracing import _trace_feg_to_exit
from temsim.physics.corrector_vacuum_field import vacuum_field
from temsim.physics.instrument_electric import configure_instrument_electric_domain, instrument_electric_end_mm
from temsim.vacuum import bind_gun_environment
from trace_normal_joint_field import beam_moments


def paths(tl, half):
    if tl==10:
        return FEM.parent.parent/"5-170-both-edge.npz", FEM.parent.parent/"normal-joint-field.npz"
    base=FEM.parent/f"tl12-{tl:.10g}A"
    return base.with_name(base.name+"-solution.npz"),base.with_name(base.name+f"-field-{half}.npz")


def bound_corrector(tl,half):
    solution,table=paths(tl,half)
    with np.load(MAT) as parent_raw, np.load(solution) as raw, np.load(table) as tab:
        if json.loads(str(raw["recipe"]))!=json.loads(str(parent_raw["recipe"])):
            raise ValueError("Corrector winding, material or geometry changed")
        expected={**parent()["corrector_A"],"probe_tl12_lens":tl}
        physical=json.loads(str(raw["report"]))
        sampled=json.loads(str(tab["report"]))
        if (physical["currents_A"]!=expected or sampled["currents_A"]!=expected or
                physical["mesh_id"]!=sampled["mesh_id"] or physical["residual"]>2e-9):
            raise ValueError("TL12 excitation or solved field mismatches this trial")
    return solution,table


def emission(f, round_path, raw_path, n):
    assert 1000<=n<=len(f.emission.weight)
    key=digest(dict(source=parent()["source_signature"],round_sha=sha(round_path),corrector_sha=sha(raw_path),n=n,
                    algorithm="source-v1"))
    path=WORK/("source-"+key+".npz")
    if path.is_file():
        return path
    k=sha(raw_path)
    far=WORK/f"gun-exterior-{k}.npz"
    if not far.is_file():
        z=np.linspace(-.001,.453,228)
        delta=3e-5
        xy=np.array([[0,0],[delta,0],[-delta,0],[0,delta],[0,-delta]])
        pts=np.zeros((len(z),5,3))
        pts[:,:,:2]=xy
        pts[:,:,2]=z[:,None]-.45
        with np.load(raw_path) as src:
            sol={k:src[k] for k in ("corners_m","moments_Am2","recipe","report")}
        values=vacuum_field(pts.reshape(-1,3),sol,quadrature=1).reshape(pts.shape)
        np.savez_compressed(far,z=z,b=np.stack([values[:,0],(values[:,1]-values[:,2])/(2*delta),
                                                     (values[:,3]-values[:,4])/(2*delta)],axis=1))
    z=np.linspace(-.002,.455,1829)
    with np.load(far) as data:
        exterior=CubicSpline(data["z"],data["b"])(z).reshape(len(z),9)
    with np.load(round_path) as data:
        spline=make_interp_spline(data["z"],2*data["coeff"][:,0],k=3)
        axial=make_interp_spline(z,spline(z),k=5)
    matrix=np.column_stack([axial(z,nu=i) for i in range(5)]+[exterior])
    magnet=GunMagnet.__new__(GunMagnet)
    magnet.dz=z[1]-z[0]
    magnet.coeff=np.ascontiguousarray(CubicSpline(z,matrix).c.transpose(1,2,0))
    return launch_in_field(f,n,key,path,magnet)


def launch_in_field(f,n,key,path,magnet):
    if n<1000 or n>len(f.emission.weight):
        raise ValueError("Invalid finite source population")
    gun=f.state.electron_gun
    bind_gun_environment(f.state)
    configure_instrument_electric_domain(gun,instrument_electric_end_mm(f.state))
    gun._instrument_magnetic_field=magnet
    gun._instrument_magnetic_identity=None
    gun.emitter.ray_count=n
    bundle=gun.emit()
    names=("x_m","y_m","tx_rad","ty_rad")
    for name in (*names,"energy_offset_ev","ray_id"):
        np.testing.assert_array_equal(getattr(f.emission,name)[:n],getattr(bundle,name))
    if not np.all(bundle.weight>0) or not np.isclose(bundle.weight.sum(),1.):
        raise ValueError("Invalid physical launch weights")
    print("Fresh effective-source propagation",n,flush=True)
    result=_trace_feg_to_exit(gun,emitted_bundle=bundle)
    out=result.exit_bundle
    np.testing.assert_array_equal(bundle.ray_id,out.ray_id)
    np.testing.assert_array_equal(bundle.weight,out.weight)
    if not out.alive.all():
        raise ValueError("Gun interception requires an explicit source ledger")
    sel=np.linspace(0,n-1,64,dtype=int)
    np.savez_compressed(path,signature=key,phase=np.column_stack([getattr(out,k) for k in (*names,"flight_time_s")]),
                        launch=np.column_stack([getattr(bundle,k) for k in names]),ids=bundle.ray_id,
                        weight=bundle.weight,emitted_A=result.emitted_current_a,energy_offset_ev=out.energy_offset_ev,
                        gun_z_mm=result.z_mm,gun_paths_m=np.stack([result.x_m[:,sel],result.y_m[:,sel]],axis=2),
                        selected_ids=bundle.ray_id[sel])
    return path


def static(trial, gun, record_dir, full, field_label="Material FEM"):
    n=len(trial.t.weights)
    stops=trial.stops if full else np.unique(np.r_[trial.indices(np.arange(450,1599.1,.5)*.001),len(trial.m.z)-1])
    selected=np.linspace(0,n-1,64,dtype=int)
    p,lost,prev=trial.t.input,trial.loss,0
    stats,tracks,focus=[],[],[]
    top_mm=1000*trial.m.z[-1]
    for group in np.array_split(stops,(len(stops)+19)//20):
        states=advance(trial.t,trial.full,p,trial.en,trial.phi0,lost,prev,group)
        for station,(p,lost) in zip(group,states,strict=True):
            live=lost==-1
            if not live.any(): raise ValueError("No particles reached this plane")
            stats.append(beam_moments(p,live,trial.t.weights))
            plane=1000*trial.m.z[station]
            if plane>=top_mm-.401:
                s=specimen(p[live],trial.t.weights[live],trial.t.current)
                focus.append(dict(z_mm=float(plane),D95_um=s["D95_um"],rms_um=s["RMS_radius_um"],
                                  core_50nm_nA=s["axis_50nm_nA"],core_50pm_pA=s["axis_50pm_pA"]))
            if full:
                tracks.append(np.where(live[selected,None],p[selected,:2],np.nan))
        prev=group[-1]
    stats=np.array(stats)
    z=trial.m.z[stops]*1000
    surface=specimen(p[live],trial.t.weights[live],trial.t.current)
    roots=mono_roots(z,stats[:,1]+stats[:,2])
    record=dict(count=n,surface_z_mm=float(z[-1]),surface=surface,waists_mm=roots.tolist(),
                principal_waists_mm=[mono_roots(z,stats[:,j]).tolist() for j in (3,4)],
                surface_radial_cov_um_mrad=float(stats[-1,1]+stats[-1,2]),focus_profile=focus,
                gun_source=str(gun),finite_source_effective=float(trial.t.weights[live].sum()**2/(trial.t.weights[live]@trial.t.weights[live])))
    print("Waists (mm)",record["waists_mm"],"surface",surface,flush=True)
    if full:
        data=dict(z_mm=z,stats=stats,paths_m=tracks,selected_ids=trial.t.ids[selected],source_ids=trial.t.ids,
                  weights=trial.t.weights,loss=lost,source_A=trial.t.current,surface_phase=p)
        np.savez_compressed(record_dir/"column-and-landings.npz",**data)
        with np.load(gun) as g:
            draw_rays(g,data,record,output_dir=record_dir,full_only=True,field_label=field_label)
        draw_probe(data,record,output_dir=record_dir,field_label=field_label)
    return record


def run(a):
    f=RefinedObjectiveFields(.5)
    model=RoundAssembly(f)
    round_path=model.run(a.objective,a.pitch)
    raw,table=bound_corrector(a.tl,a.half)
    gun=emission(f,round_path,raw,a.count)
    trial=Trial(f,a.count,a.step,gun,table)
    trial.material(model,a.objective,a.pitch)
    folder=SAVE/a.label
    folder.mkdir(exist_ok=True,parents=True)
    record=static(trial,gun,folder,a.full)
    record.update(fronts=trial.fronts(),objective_A=a.objective,TL12_A=a.tl,pitch_mm=a.pitch,
                  step_mm=a.step,round_sha=sha(round_path),corrector_sha=sha(table),table_half=a.half)
    write(folder/"tracking.json",record)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--count",type=int,default=1024)
    ap.add_argument("--step",type=float,default=.1)
    ap.add_argument("--pitch",type=float,default=.5)
    ap.add_argument("--tl",type=float,default=FINAL["TL12_A"])
    ap.add_argument("--objective",type=float,default=FINAL["objective_A"])
    ap.add_argument("--half",type=int,default=24)
    ap.add_argument("--label",default="fixed-currents")
    ap.add_argument("--full",action="store_true")
    run(ap.parse_args())
