"""Exact final Objective verification with nine zero-current production pencils."""
from pathlib import Path
import copy,gzip,hashlib,json,math,sys,time
import numpy as np
from scipy.constants import e,m_e,c
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from study_two_condenser_nanoprobe import prepare_state,set_controls
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
from temsim.physics.unified_field_particle import trace_unified_response
from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.profile_io import save_profile
OUT=Path(__file__).resolve().parent

def forbidden(*args,**kwargs):
    raise AssertionError('No source emitter is permitted in this zero-current observer')
ColdFieldEmitter.emit=forbidden
state,selection=prepare_state(2048,.1)
apply_operating_mode_pair(state,'nano_probe','diffraction',column_name='C2',recording_name=selection.recording)
controls={'condenser_lens_1':90.,'condenser_lens_2':35.,'mini_condenser':10.,'objective_lens':29.223518226455024}
set_controls(state,controls)
previous=json.loads((OUT/'proposal.json').read_text())
projector=previous['projector_percent']
lenses={item.key:item for item in state.lenses}
for key,value in projector.items():lenses[key].percent=value
state.sync_objective()
prepared=capture_instrument_snapshot(state)
snapshot_path=OUT/'final-current-implementation-inputs.json.gz'
snapshot_path.write_bytes(gzip.compress(json.dumps(prepared.to_dict(),indent=2).encode(),mtime=0))
profile_path=OUT/'final-current-implementation-input.toml'
save_profile(profile_path,state,selection)
assert capture_instrument_snapshot(state).digest==prepared.digest,'Saving inputs changed the working point'
source=float(state.sample.upper_surface_z_mm)*1e-3
reference=float(state.fluorescent_screen.z_mm)*1e-3
energy=300000.; ratio=energy*e/(m_e*c*c)
p=m_e*c*math.sqrt(ratio)*math.sqrt(ratio+2.)
hp,ha=1e-10,1e-7
initial=np.zeros((9,4))
for i,h in enumerate((hp,hp,ha,ha)):
    initial[1+2*i,i]=h;initial[2+2*i,i]=-h
runtime=get_unified_field_runtime(state)
sampler=compile_unified_potential_sampler(runtime.snapshot,prefer_plane=True)
points=np.c_[initial[:5,:2],np.full(5,source)]
A=np.asarray(sampler.sample(points)['A_t_m'])
grad=np.column_stack(((A[1,:2]-A[2,:2])/(2*hp),(A[3,:2]-A[4,:2])/(2*hp)))
basis=np.eye(4);basis[2:,:2]=e/p*grad
started=time.perf_counter()
response=trace_unified_response(state,initial,source,np.array([reference]),energy_ev=energy,flight_time_s=0.,maximum_step_m=.05e-3)
phase=np.asarray(response['xy_slopes'])[0]
jac=np.column_stack([(phase[1+2*i]-phase[2+2*i])/(2*inc) for i,inc in enumerate((hp,hp,ha,ha))])
canonical=jac@basis
B=canonical[:2,2:]
residual=float(np.linalg.norm(canonical[:2,:2],ord=2))
camera=math.sqrt(abs(float(np.linalg.det(B))))
relative=abs(camera/.05-1)
assert capture_instrument_snapshot(state).digest==prepared.digest,'Observer changed the prepared working point'
assert np.asarray(response['alive']).all() and np.isfinite(canonical).all()
assert residual<1e-3 and relative<.03
suggested=copy.deepcopy(previous['suggested_column_mode_override'])
suggested['calibration_reference']='Independent C1+C2 Nanoprobe diffraction differential reference at exact Objective 29.223518226455024% and the specimen upper surface. D/I were jointly matched with P1/P2 held fixed. Production unified nine-pencil verification used 0.05 mm at this exact final point. Independent 0.1/0.05 mm and half-increment checks belong to nearby Objective 29.2235172%, as recorded separately. Source emission was forbidden; no finite-source, finite-angle, detector-signal, wave or experimental qualification.'
targets=suggested['targets']
targets.update(objective_excitation_percent=controls['objective_lens'],
    achieved_effective_camera_length_m=camera,achieved_diffraction_conjugacy_residual=residual,
    camera_length_relative_error=relative,validation_steps_mm=[.05],diagnostic_position_steps_m=[hp],diagnostic_angle_steps_rad=[ha],
    exact_working_point_validation_step_mm=.05,nearby_step_refinement_objective_excitation_percent=29.2235172,
    nearby_step_refinement_steps_mm=[.1,.05],nearby_half_increment_check=True)
record={'status':'FINAL_CURRENT_IMPLEMENTATION_EXACT_OBJECTIVE_UNIFIED_DIFFERENTIAL_REFERENCE_ONLY',
    'scope':'Nine zero-current differential pencils, emitter forbidden. No finite-source population or current-weighted beam metrics were generated.',
    'configuration':'FEG / physical C1+C2 / No Energy Filter / Nanoprobe / Diffraction',
    'upper_controls_percent':controls,'projector_percent':projector,'source_plane':'specimen upper surface',
    'source_z_mm':source*1e3,'reference_z_mm':reference*1e3,'energy_ev':energy,'diagnostic_clock_s':0.,
    'maximum_step_mm':.05,'position_increment_m':hp,'mechanical_slope_increment_rad':ha,
    'receiver_insertion':{item.key:item.inserted for item in (*state.stem_detectors,state.fluorescent_screen,state.camera)},
    'canonical_to_mechanical_basis':basis.tolist(),'mechanical_jacobian':jac.tolist(),'canonical_jacobian':canonical.tolist(),
    'canonical_position_residual_norm':residual,'effective_camera_length_m':camera,
    'camera_length_relative_error_to_50mm':relative,'camera_length_singular_values_m':np.linalg.svd(B,compute_uv=False).tolist(),
    'all_pencils_reach_reference':bool(np.asarray(response['alive']).all()),'blocked_key':list(response['blocked_key']),
    'elapsed_s':time.perf_counter()-started,'receipt':thaw_json(response['receipt']),
    'input_snapshot_digest':prepared.digest,'input_physical_digest':prepared.physical_digest,
    'source_controls_unchanged':True,'emitter_calls':0,'no_finite_source_population_was_created':True,
    'input_snapshot':snapshot_path.name,'input_profile':profile_path.name,
    'artifacts':{path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in (Path(__file__),snapshot_path,profile_path)},
    'nearby_validation_record':'proposal.json','suggested_column_mode_override':suggested}
(OUT/'final-current-implementation-objective.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:record[key] for key in ('status','canonical_position_residual_norm','effective_camera_length_m','camera_length_relative_error_to_50mm','all_pencils_reach_reference','elapsed_s')}),flush=True)

