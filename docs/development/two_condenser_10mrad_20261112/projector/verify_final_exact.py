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
BASE=Path(__file__).resolve().parent
settings=json.loads((BASE/'final-upstream-controls.json').read_text())
if not settings.get('ready',False):
    raise ValueError('Wait for the exact final upstream controls')
case_name=str(settings['case_name'])
reference_case=str(settings.get('projector_reference_case','weak-c2-provisional'))
for value in (case_name,reference_case):
    if not value or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in value):
        raise ValueError('Use simple unique case identifiers')
OUT=BASE/case_name
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'final-exact-objective.json').exists():
    raise ValueError('Retain old exact-point records and select a new case name')
settings_path=OUT/'upstream-controls.json'
settings_path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
controls={key:float(value) for key,value in settings['controls_percent'].items()}
if set(controls)!={'condenser_lens_1','condenser_lens_2','mini_condenser','objective_lens'} or not all(math.isfinite(value) and value>0 for value in controls.values()):
    raise ValueError('Supply all four exact final installed upstream strengths')
aperture_mm=float(settings.get('c2_aperture_diameter_mm',.07))
if abs(aperture_mm-.07)>1e-14:
    raise ValueError('This study requires the 70 um C2 aperture')

def forbidden(*args,**kwargs):
    raise AssertionError('No source emitter is permitted in this zero-current observer')
ColdFieldEmitter.emit=forbidden
state,selection=prepare_state(int(settings.get('configured_source_particles_not_emitted',2048)),.05)
apply_operating_mode_pair(state,'nano_probe','diffraction',column_name='C2',recording_name=selection.recording)
state.condenser_aperture_2.diameter_mm=aperture_mm
state.condenser_aperture_2.validate()
set_controls(state,controls)
previous=json.loads((BASE/reference_case/'proposal.json').read_text())
projector=previous['projector_percent']
lenses={item.key:item for item in state.lenses}
for key,value in projector.items():lenses[key].percent=value
state.sync_objective()
prepared=capture_instrument_snapshot(state)
snapshot_path=OUT/'final-exact-inputs.json.gz'
snapshot_path.write_bytes(gzip.compress(json.dumps(prepared.to_dict(),indent=2).encode(),mtime=0))
profile_path=OUT/'final-exact-input.toml'
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
suggested['calibration_reference']=(f"Independent C1+C2 Nanoprobe diffraction differential reference at exact Objective {controls['objective_lens']!r}% and the specimen upper surface, with 70 um physical C2 aperture. D/I were jointly matched with P1/P2 held fixed. Production unified nine-pencil verification used 0.05 mm at this exact point. "
    f"Independent 0.1/0.05 mm and half-increment checks belong to nearby Objective {previous['upper_controls_percent']['objective_lens']!r}%, as recorded separately. Source emission was forbidden; no finite-source, finite-angle, detector-signal, wave or experimental qualification.")
targets=suggested['targets']
targets.update(objective_excitation_percent=controls['objective_lens'],
    achieved_effective_camera_length_m=camera,achieved_diffraction_conjugacy_residual=residual,
    camera_length_relative_error=relative,validation_steps_mm=[.05],diagnostic_position_steps_m=[hp],diagnostic_angle_steps_rad=[ha],
    exact_working_point_validation_step_mm=.05,nearby_step_refinement_objective_excitation_percent=previous['upper_controls_percent']['objective_lens'],
    nearby_step_refinement_steps_mm=[.1,.05],nearby_half_increment_check=True)
record={'status':'FINAL_EXACT_OBJECTIVE_UNIFIED_DIFFERENTIAL_REFERENCE_ONLY',
    'scope':'Nine zero-current differential pencils, emitter forbidden. No finite-source population or current-weighted beam metrics were generated.',
    'configuration':'FEG / physical C1+C2 / No Energy Filter / Nanoprobe / Diffraction',
    'upper_controls_percent':controls,'c2_aperture_diameter_mm':aperture_mm,'projector_percent':projector,'source_plane':'specimen upper surface',
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
    'artifacts':{path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in (Path(__file__),snapshot_path,profile_path,settings_path)},
    'nearby_validation_record':'../'+reference_case+'/proposal.json','suggested_column_mode_override':suggested}
(OUT/'final-exact-objective.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({key:record[key] for key in ('status','canonical_position_residual_norm','effective_camera_length_m','camera_length_relative_error_to_50mm','all_pencils_reach_reference','elapsed_s')}),flush=True)
