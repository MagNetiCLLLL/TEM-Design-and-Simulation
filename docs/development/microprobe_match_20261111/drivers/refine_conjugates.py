"""Native chief-ray refinement of the Microprobe pupil and focal planes."""
from pathlib import Path
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
from temsim.physics.instrument_magnetic import active_column_events,events_overlapping_interval
from temsim.physics.round_lens_aberration_benchmark import _charge_over_momentum
from temsim.cpu_resources import numerical_job
from validate_joint_probe import _restore_exact

spec=importlib.util.spec_from_file_location("micro_rotation",ROOT/"outputs/microprobe-20261111/analyze_micro_rotation.py")
rotation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(rotation)
snap=InstrumentSnapshot.from_dict(json.loads((HERE/"column/n1024-first/trial-00/inputs.json").read_text()))
state=_restore_exact(snap)
state.step_mm=.1
surface=float(state.sample.upper_surface_z_mm)
energy=float(state.electron_gun.nominal_exit_energy_ev)
parts={part.key:part for part in (*state.lenses,*state.corrector_elements,*state.deflectors)}
field=rotation.magnetic_snapshot(state)
g=.5*_charge_over_momentum(energy)*float(field.sample(np.array([[0.,0.,surface*.001]]))["B_t"][0,2])
j=np.array(((0.,-1.),(1.,0.)))
started=time.perf_counter()


def basis(z0,targets,step=.1):
    targets=np.unique(targets)
    h=np.array((2e-9,2e-9,2e-7,2e-7))
    initial=np.r_[np.zeros((1,4)),np.diag(h),-np.diag(h)]
    events=events_overlapping_interval(state,active_column_events(state),z0,float(targets[-1]))
    cp=propagate(state,z0,float(targets[-1]),initial[:,0],initial[:,2],initial[:,1],initial[:,3],
        events=events,save_z_mm=targets,checkpoint_z_mm=targets,return_checkpoints=True,
        maximum_step_mm=step,initial_kinetic_energy_ev=energy,initial_time_s=np.zeros(9))[-1]
    rows=np.searchsorted(cp.z_mm,targets)
    if not np.array_equal(cp.z_mm[rows],targets):
        raise ValueError("Missing native conjugacy plane")
    phase=np.stack([getattr(cp,key)[rows] for key in ("x_m","y_m","tx_rad","ty_rad")],axis=1)
    return (phase[:,:,1:5]-phase[:,:,5:9])/(2*h[None,None,:])


def scalar_d(matrix):
    b=matrix[:2,2:]
    u,_,v=np.linalg.svd(b)
    r=u@v
    d=matrix[2:,2:]+g*j@b
    return float(np.trace(r.T@d)/2),d


with numerical_job(requested=1):
    with np.load(HERE/"native-maps.npz",allow_pickle=False) as saved:
        z=saved["z_mm"]
        m=saved["to_sample_mechanical"]
    interval=np.flatnonzero((z>1584.)&(z<1584.8))
    values=np.array([scalar_d(m[index])[0] for index in interval])
    crossing=np.flatnonzero(values[:-1]*values[1:]<0)
    if len(crossing)!=1:
        raise ValueError("An isolated canonical focal root is required")
    q=int(crossing[0]);a,b=z[interval[q]],z[interval[q+1]]
    va,vb=values[q:q+2]
    slope=(vb-va)/(b-a)
    root=float(a-va/slope)
    focal=[]
    for iteration in range(2):
        native=basis(root,[surface])[0]
        value,d=scalar_d(native)
        focal.append(dict(z_mm=root,mechanical_matrix=native.tolist(),canonical_d=d.tolist(),
            canonical_d_frobenius=float(np.linalg.norm(d)),mechanical_d_frobenius=float(np.linalg.norm(native[2:,2:])),
            b_singular_values_m_per_rad=np.linalg.svd(native[:2,2:],compute_uv=False).tolist()))
        print(f"{time.perf_counter()-started:.1f}s focal root {root:.9f} mm; canonical D {np.linalg.norm(d):.6g}",flush=True)
        root-=value/slope
    # Resolve the distinct HP aperture image directly, with no source-waist gate.
    hp1=float(parts["probe_hp1_hexapole"].z_mm)
    targets=np.arange(1598.2,1598.60001,.001)
    observed=basis(hp1,targets)
    bnorm=np.linalg.norm(observed[:,:2,2:],axis=(1,2))
    idx=int(np.argmin(bnorm))
    aa,bb=observed[idx,:2,:2],observed[idx,:2,2:]
    u,_,v=np.linalg.svd(aa);r=u@v
    signed=np.einsum("ij,nij->n",r,observed[:,:2,2:])/2
    flip=np.flatnonzero(signed[:-1]*signed[1:]<0)
    if len(flip)!=1:
        raise ValueError("An isolated HP pupil-image root is required")
    q=int(flip[0]);z0,z1=targets[q:q+2];f0,f1=signed[q:q+2]
    pupil=float(z0-f0*(z1-z0)/(f1-f0))
    native=basis(hp1,[pupil])[0]
    pupil_report=dict(z_mm=pupil,initial_native_bracket_mm=[float(z0),float(z1)],
        mechanical_matrix=native.tolist(),b_norm_m_per_rad=float(np.linalg.norm(native[:2,2:],2)),
        a_singular_values=np.linalg.svd(native[:2,:2],compute_uv=False).tolist(),
        interpretation="HP1-conjugate aperture plane; not the source image and not asserted coma-free in Microprobe")
    print(f"{time.perf_counter()-started:.1f}s HP pupil image {pupil:.9f} mm; B {pupil_report['b_norm_m_per_rad']:.6g}",flush=True)
    hp2=float(parts["probe_hp2_hexapole"].z_mm)
    core=basis(hp2,[hp1],step=.05)[0]
    report=dict(scope="Native nominal-energy reference derivatives, never finite-source qualification",input_digest=snap.digest,
        diagnostic_digest=capture_instrument_snapshot(state).digest,energy_ev=energy,
        canonical_focal_refinement=focal,hp1_pupil_image=pupil_report,
        core_step005=dict(mechanical_matrix=core.tolist(),b_norm_m_per_rad=float(np.linalg.norm(core[:2,2:],2)),
                         a_singular_values=np.linalg.svd(core[:2,:2],compute_uv=False).tolist()),
        source_waist_note="Current-weighted finite-source RMS/D95 minima are separate measured observables")
    # A compensated mechanical shift has no obligation to share the canonical point pivot.
    with np.load(HERE/"native-maps.npz",allow_pickle=False) as saved:
        z=saved["z_mm"];global_maps=saved["global_mechanical"]
    later=np.flatnonzero((z>state.ac_deflector.lower_z_mm+state.ac_deflector.mechanical_coil_length_mm*.5)&(z<surface))
    inverse=np.linalg.inv(global_maps[-1])
    shifts={}
    for label,target in (("mechanical_shift_x_50nm",np.array((50e-9,0.,0.,0.))),
                         ("mechanical_shift_y_50nm",np.array((0.,50e-9,0.,0.)))):
        trajectory=np.array([value@inverse@target for value in global_maps[later]])
        radius=np.linalg.norm(trajectory[:,:2],axis=1)
        idx=int(np.argmin(radius))
        shifts[label]=dict(nearest_axis_z_mm=float(z[later[idx]]),minimum_reference_radius_nm=float(radius[idx]*1e9),
            observation_note="Linear native reference response downstream of physical coils; no assumed common zero-radius pivot")
    report["mechanical_shift_backpropagation"]=shifts
    (HERE/"refined-conjugates.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
print("Refined conjugate diagnostics complete",flush=True)
