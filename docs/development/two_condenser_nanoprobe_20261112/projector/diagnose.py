"""Zero-current differential projector observer; never emits source particles."""
from pathlib import Path
import json, math, sys, time
import numpy as np
from scipy.constants import e,m_e,c
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from study_two_condenser_nanoprobe import prepare_state,set_controls
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.operating_modes import apply_operating_mode_pair,mode_by_key
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
from temsim.physics.unified_field_particle import trace_unified_response
from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot

OUT=Path(__file__).resolve().parent

def forbidden(*args,**kwargs):
    raise AssertionError('This observer must never invoke a source emitter')
ColdFieldEmitter.emit=forbidden
state,selection=prepare_state(1024,.1)
apply_operating_mode_pair(state,'nano_probe','diffraction',column_name='C2',recording_name=selection.recording)
controls={'condenser_lens_1':90.,'condenser_lens_2':35.,'mini_condenser':10.,'objective_lens':29.223516026455023}
set_controls(state,controls)
original=capture_instrument_snapshot(state)
source=float(state.sample.upper_surface_z_mm)*1e-3
reference=float(state.fluorescent_screen.z_mm)*1e-3
energy=300000.
r=energy*e/(m_e*c*c)
p=m_e*c*math.sqrt(r)*math.sqrt(r+2.)
hp,ha=1e-10,1e-7
initial=np.zeros((9,4))
for dim,step in enumerate((hp,hp,ha,ha)):
    initial[1+2*dim,dim]=step
    initial[2+2*dim,dim]=-step
runtime=get_unified_field_runtime(state)
sampler=compile_unified_potential_sampler(runtime.snapshot,prefer_plane=True)
positions=np.c_[initial[:5,:2],np.full(5,source)]
values=sampler.sample(positions)
A=np.asarray(values['A_t_m'])
grad=np.column_stack(((A[1,:2]-A[2,:2])/(2*hp),(A[3,:2]-A[4,:2])/(2*hp)))
basis=np.eye(4)
basis[2:,:2]=e/p*grad
record={'scope':'Zero-current differential pencils only; no emitter invocation, finite-source transport, specimen scattering, detector signal or wave qualification.',
        'configuration':'FEG / physical C1+C2 / No Energy Filter / Nanoprobe / Diffraction',
        'source_plane':'specimen upper surface', 'source_z_mm':source*1e3,
        'catalog_source_plane_z_mm':float(state.sample.z_mm),
        'reference_plane':'TOML main-screen camera-length reference, retracted in Nanoprobe',
        'reference_z_mm':reference*1e3,'controls_percent':controls,
        'projector_percent':{x.key:x.percent for x in state.lenses if x.key in ('diffraction_lens','intermediate_lens','projector_lens_1','projector_lens_2')},
        'receiver_insertion':{x.key:x.inserted for x in (*state.stem_detectors,state.fluorescent_screen,state.camera)},
        'input_energy_ev':energy,'input_diagnostic_clock_s':0.,'position_increment_m':hp,'mechanical_slope_increment_rad':ha,
        'source_A_t_m':A[0].tolist(),'source_A_gradient_t':grad.tolist(),'canonical_to_mechanical_basis':basis.tolist(),
        'input_physical_digest':original.physical_digest,'runs':[]}
for step_mm in (.2,.1):
    t=time.perf_counter()
    response=trace_unified_response(state,initial,source,np.array([reference]),energy_ev=energy,flight_time_s=0.,maximum_step_m=step_mm*1e-3)
    phase=np.asarray(response['xy_slopes'])[0]
    jac=np.column_stack([(phase[1+2*i]-phase[2+2*i])/(2*inc) for i,inc in enumerate((hp,hp,ha,ha))])
    canonical=jac@basis
    B=canonical[:2,2:]
    Ablock=canonical[:2,:2]
    singular=np.linalg.svd(B,compute_uv=False) if np.isfinite(B).all() else np.full(2,np.nan)
    camera=math.sqrt(abs(float(np.linalg.det(B)))) if np.isfinite(B).all() else None
    row={'maximum_step_mm':step_mm,'elapsed_s':time.perf_counter()-t,
         'all_pencils_reach_reference':bool(np.asarray(response['alive']).all()),
         'blocked_key':list(response['blocked_key']),
         'blocked_z_mm':(np.asarray(response['blocked_z_m'])*1e3).tolist(),
         'mechanical_jacobian':jac.tolist(),'canonical_jacobian':canonical.tolist(),
         'canonical_position_residual_norm':float(np.linalg.norm(Ablock,ord=2)) if np.isfinite(Ablock).all() else None,
         'effective_camera_length_m':camera,'camera_length_singular_values_m':singular.tolist(),
         'camera_length_relative_error_to_50mm':abs(camera/.05-1) if camera is not None else None,
         'chief_reference_phase':phase[0].tolist(),'receipt':thaw_json(response['receipt'])}
    record['runs'].append(row)
    assert capture_instrument_snapshot(state).digest==original.digest,'Observer changed the prepared working point'
    (OUT/'response.json').write_text(json.dumps(record,indent=2,allow_nan=True)+'\n',encoding='utf-8')
    print(json.dumps({key:row[key] for key in ('maximum_step_mm','elapsed_s','all_pencils_reach_reference','canonical_position_residual_norm','effective_camera_length_m','camera_length_relative_error_to_50mm')}),flush=True)
record['source_controls_unchanged']=True
record['status']='DIAGNOSTIC_ONLY_NOT_A_PROJECTOR_QUALIFICATION'
(OUT/'response.json').write_text(json.dumps(record,indent=2,allow_nan=True)+'\n',encoding='utf-8')
