"""Native-field chief-ray diagnostics; not a finite-source physical trial."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
import numpy as np
from temsim.instrument_snapshot import InstrumentSnapshot,capture_instrument_snapshot
from temsim.physics.core import propagate
from temsim.physics.instrument_magnetic import active_column_events,events_overlapping_interval,column_deflector_drives
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum
from temsim.cpu_resources import numerical_job
from validate_joint_probe import _restore_exact

spec=importlib.util.spec_from_file_location("micro_rotation",ROOT/"outputs/microprobe-20261111/analyze_micro_rotation.py")
rotation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)

CASE=HERE/"column/n1024-first/trial-00"
snapshot=InstrumentSnapshot.from_dict(json.loads((CASE/"inputs.json").read_text()))
state=_restore_exact(snapshot)
state.step_mm=.1
surface=float(state.sample.upper_surface_z_mm)
energy=float(state.electron_gun.nominal_exit_energy_ev)
parts={part.key:part for part in (*state.lenses,*state.corrector_elements,*state.deflectors)}
started=time.perf_counter()


def log(message):
    print(f"{time.perf_counter()-started:.1f}s {message}",flush=True)


def trace(z0,planes,initial):
    points=np.asarray(initial,float)
    events=events_overlapping_interval(state,active_column_events(state),z0,max(planes))
    result=propagate(state,z0,max(planes),points[:,0],points[:,2],points[:,1],points[:,3],
        events=events,include_hexapole=True,include_spherical_aberration=True,
        save_z_mm=planes,checkpoint_z_mm=planes,return_checkpoints=True,maximum_step_mm=.1,
        initial_kinetic_energy_ev=energy,initial_time_s=np.zeros(len(points)))
    cp=result[-1]
    selected=np.searchsorted(cp.z_mm,planes)
    if not np.array_equal(cp.z_mm[selected],planes):
        raise ValueError("Missing native diagnostic observation")
    values=np.stack([getattr(cp,key)[selected] for key in ("x_m","y_m","tx_rad","ty_rad")],axis=1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic ray; reduce stencil or inspect physical stops")
    return values


def basis(z0,planes,scale=1.):
    h=np.array((2e-9,2e-9,2e-7,2e-7))*scale
    x=np.r_[np.zeros((1,4)),np.diag(h),-np.diag(h)]
    values=trace(z0,planes,x)
    return (values[:,:,1:5]-values[:,:,5:9])/(2*h[None,None,:])


def canonical(g):
    c=np.eye(4)
    c[2,1],c[3,0]=-g,g
    return c


with numerical_job(requested=1):
    names={key:float(parts[key].z_mm) for key in
        ("condenser_lens_1","condenser_lens_2","condenser_lens_3","adapter_lens",
         "probe_hp2_hexapole","probe_tl22_lens","probe_tl21_lens","probe_hp1_hexapole","probe_tl12_lens",
         "probe_dph2_deflector","probe_dp22_deflector","probe_dp21_deflector","probe_dph1_deflector","probe_dp11_deflector")}
    names.update(ac_upper=float(state.ac_deflector.upper_z_mm),ac_lower=float(state.ac_deflector.lower_z_mm),
                 mini=float(state.mini_condenser.z_mm),sample=surface)
    planes=np.unique(np.r_[np.arange(1300.,surface,.2),list(names.values())])
    cached=HERE/"diagnostic-global-native.npz"
    if cached.exists():
        with np.load(cached,allow_pickle=False) as arrays:
            if str(arrays["digest"])!=capture_instrument_snapshot(state).digest or not np.array_equal(arrays["z_mm"],planes):
                raise ValueError("Diagnostic cache input mismatch")
            global_maps=arrays["maps"]
    else:
        log("Tracing native nine-ray reference basis")
        global_maps=basis(state.electron_gun.exit_plane_z_mm,planes)
        np.savez_compressed(cached,z_mm=planes,maps=global_maps,digest=capture_instrument_snapshot(state).digest)
    field=rotation.magnetic_snapshot(state)
    sample=field.sample(np.c_[np.zeros((len(planes),2)),planes*.001])
    bz=np.asarray(sample["B_t"])[:,2]
    g=.5*_charge_over_momentum(energy)*bz
    c=np.array([canonical(value) for value in g])
    final=int(np.searchsorted(planes,surface))
    to_sample=np.array([global_maps[final]@np.linalg.inv(value) for value in global_maps])
    can_to_sample=np.array([c[final]@value@np.linalg.inv(source) for value,source in zip(to_sample,c)])
    def one(z0,z1):
        a,b=(int(np.searchsorted(planes,z)) for z in (z0,z1))
        m=global_maps[b]@np.linalg.inv(global_maps[a])
        mc=c[b]@m@np.linalg.inv(c[a])
        return dict(mechanical=m.tolist(),canonical=mc.tolist(),canonical_b_norm_m_per_rad=float(np.linalg.norm(mc[:2,2:],2)),
            canonical_a_singular_values=np.linalg.svd(mc[:2,:2],compute_uv=False).tolist(),
            mechanical_b_norm_m_per_rad=float(np.linalg.norm(m[:2,2:],2)),
            source_bz_t=float(bz[a]),target_bz_t=float(bz[b]))
    dnorm=np.linalg.norm(can_to_sample[:,2:,2:],axis=(1,2))
    best=np.flatnonzero((dnorm[1:-1]<dnorm[:-2])&(dnorm[1:-1]<=dnorm[2:]))+1
    best=[int(i) for i in best if planes[i]>1300.]
    report=dict(scope="Nine-ray native full-field nominal-energy derivative; no finite-source qualification",
        source_input_digest=snapshot.digest,diagnostic_input_digest=capture_instrument_snapshot(state).digest,
        energy_ev=energy,source_particle_setting_unchanged=state.electron_gun.emitter.ray_count,
        step_mm=.1,coordinates="x,y in metres; mechanical tx,ty slopes in radians; symmetric-gauge canonical increments separate",
        observed_planes_mm=names,core=one(names["probe_hp2_hexapole"],names["probe_hp1_hexapole"]),
        hp1_to_sample=one(names["probe_hp1_hexapole"],surface),ac_upper_to_sample=one(names["ac_upper"],surface),
        ac_lower_to_sample=one(names["ac_lower"],surface),
        downstream_canonical_d_minima=[dict(z_mm=float(planes[i]),bracket_mm=planes[i-1:i+2].tolist(),
            d_frobenius=float(dnorm[i]),mechanical_d=to_sample[i,2:,2:].tolist(),
            canonical_d=can_to_sample[i,2:,2:].tolist(),b_singular_values=np.linalg.svd(can_to_sample[i,:2,2:],compute_uv=False).tolist()) for i in best])
    log("Native reference matrices complete")
    # AC opposite-coil drive and DP12 common-coil bias span four physical outputs.
    channels=[(state.ac_deflector,"kick_x_mrad"),(state.ac_deflector,"kick_y_mrad"),
              (parts["probe_dp12_scan_deflector"],"alignment_x_mrad"),
              (parts["probe_dp12_scan_deflector"],"alignment_y_mrad")]
    response=[]
    h=.001
    for item,name in channels:
        saved=float(getattr(item,name))
        points=[]
        for sign in (1.,-1.):
            setattr(item,name,saved+sign*h)
            points.append(trace(1370.,np.array([surface]),np.zeros((1,4)))[0,:,0])
        setattr(item,name,saved)
        response.append((points[0]-points[1])/(2*h))
        log("AC physical channel "+name)
    jac=np.asarray(response).T
    scaled=jac/np.array((1e-6,1e-6,1e-3,1e-3))[:,None]
    report["ac_response"]=dict(channels=[item.key+"."+name for item,name in channels],
        response_m_and_rad_per_mrad=jac.tolist(),scaled_condition=float(np.linalg.cond(scaled)),
        normalization="1 um position, 1 mrad mechanical angle",targets={})
    for label,target in (("shift_x_50nm",(50e-9,0.,0.,0.)),("shift_y_50nm",(0.,50e-9,0.,0.)),
                         ("tilt_x_005mrad",(0.,0.,.05e-3,0.)),("tilt_y_005mrad",(0.,0.,0.,.05e-3))):
        solution=np.linalg.solve(jac,np.asarray(target))
        for (item,name),drive in zip(channels,solution):
            setattr(item,name,float(drive))
        achieved=trace(1370.,np.array([surface]),np.zeros((1,4)))[0,:,0]
        physical=[dict(key=d.key,events=d.events) if hasattr(d,"key") else str(d) for d in ()]
        control={}
        for (item,name),drive in zip(channels,solution):
            control.setdefault(item.key,{})[name]=float(drive)
        report["ac_response"]["targets"][label]=dict(target=list(target),drives_mrad=solution.tolist(),
            achieved=achieved.tolist(),residual=(achieved-target).tolist(),controls=control)
        for item,name in channels:
            setattr(item,name,0.)
        log("Native chief validation "+label)
    (HERE/"native-maps.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
    np.savez_compressed(HERE/"native-maps.npz",z_mm=planes,global_mechanical=global_maps,
        to_sample_mechanical=to_sample,to_sample_canonical=can_to_sample,bz_t=bz)
log("Saved native maps and AC command proposals")
