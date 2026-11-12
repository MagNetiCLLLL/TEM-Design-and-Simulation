"""C2 projector proposal from captured on-axis fields, verified by zero-current pencils."""
from pathlib import Path
import gzip, hashlib, json, math, sys, time
import numpy as np
from scipy.constants import e,m_e,c
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from study_two_condenser_nanoprobe import prepare_state,set_controls
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.operating_modes import apply_operating_mode_pair
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.unified_field_particle import trace_unified_response
from temsim.optics.direct_alignment import _rk4_axisymmetric_larmor_matrix
from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.profile_io import save_profile
BASE=Path(__file__).resolve().parent
settings=json.loads((BASE/'upstream-controls.json').read_text())
if not settings.get('ready',False):
    raise ValueError('Wait for the final upstream working point; no diagnostic is started')
case_name=str(settings['case_name'])
if not case_name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in case_name):
    raise ValueError('Use a simple unique case name')
OUT=BASE/case_name
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'proposal.json').exists():
    raise ValueError('Choose a new case name; existing projector evidence is retained')
settings_path=OUT/'upstream-controls.json'
settings_path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
controls={key:float(value) for key,value in settings['controls_percent'].items()}
if set(controls)!={'condenser_lens_1','condenser_lens_2','mini_condenser','objective_lens'} or not all(math.isfinite(value) and value>0 for value in controls.values()):
    raise ValueError('Supply all four final installed upstream strengths')
aperture_mm=float(settings.get('c2_aperture_diameter_mm',.07))
if abs(aperture_mm-.07)>1e-14:
    raise ValueError('This study requires the 70 um physical C2 aperture')

def forbidden(*args,**kwargs):
    raise AssertionError('Zero-current projector diagnostics must not invoke an emitter')
ColdFieldEmitter.emit=forbidden
state,selection=prepare_state(1024,.1)
apply_operating_mode_pair(state,'nano_probe','diffraction',column_name='C2',recording_name=selection.recording)
state.condenser_aperture_2.diameter_mm=aperture_mm
state.condenser_aperture_2.validate()
set_controls(state,controls)
lenses={x.key:x for x in state.lenses}
keys=('diffraction_lens','intermediate_lens','projector_lens_1','projector_lens_2')
original={key:float(lenses[key].percent) for key in keys}
source=float(state.sample.upper_surface_z_mm)*1e-3
reference=float(state.fluorescent_screen.z_mm)*1e-3
energy=300000.; ratio=energy*e/(m_e*c*c)
p=m_e*c*math.sqrt(ratio)*math.sqrt(ratio+2.)
runtime=get_unified_field_runtime(state)
snapshot=runtime.snapshot
if type(snapshot) is not ElectromagneticPotentialSnapshot:
    raise ValueError('This proposal requires the captured static round-lens field')
count=math.ceil((reference-source)/(.05e-3))
z=np.linspace(source,reference,count+1)
stages=np.empty(2*len(z)-1)
stages[::2]=z; stages[1::2]=.5*(z[:-1]+z[1:])
points=np.c_[np.zeros((len(stages),2)),stages]
full=compile_unified_potential_sampler(snapshot,prefer_plane=False).sample(points)
fullB=np.asarray(full['B_t'])
assert np.max(np.abs(fullB[:,:2]))<1e-12,'Axisymmetric proposal unavailable for nonzero on-axis transverse B'
print('AXIS_PHI',float(np.min(full['phi_v'])),float(np.max(full['phi_v'])),flush=True)
assert np.ptp(np.asarray(full['phi_v']))<1.,'Variable-energy projector needs a different proposal model'
profiles=[]
for key in keys[:2]:
    rows=tuple(row for row in snapshot.contributions if row.source_id==key or key in row.owner_ids)
    if len(rows)!=1:
        raise ValueError((key,[(r.source_id,r.owner_ids) for r in rows]))
    private=ElectromagneticPotentialSnapshot(snapshot.scene,rows,snapshot.gauge)
    b=np.asarray(compile_unified_potential_sampler(private,prefer_plane=False).sample(points)['B_t'])[:,2]
    profiles.append(b/original[key])
fixed=fullB[:,2]-profiles[0]*original[keys[0]]-profiles[1]*original[keys[1]]

def model(v):
    g=-e/(2*p)*(fixed+profiles[0]*v[0]+profiles[1]*v[1])
    return _rk4_axisymmetric_larmor_matrix(np.ascontiguousarray(g),z)
base=model([original[keys[0]],original[keys[1]]])
print('BASE_AXIS',base.tolist(),flush=True)
solutions=[]
for sign in (1.,-1.):
    for seed in ([original[keys[0]],original[keys[1]]],[15.,10.],[15.,30.],[25.,20.]):
        fit=least_squares(lambda v:np.array([model(v)[0,0],model(v)[0,1]/(.05*sign)-1]),seed,
            bounds=([.00001,.00001],[99.,99.]),max_nfev=180,x_scale=[10.,10.],
            xtol=1e-12,ftol=1e-12,gtol=1e-12)
        matrix=model(fit.x)
        error=float(np.linalg.norm([matrix[0,0],matrix[0,1]/(.05*sign)-1]))
        row={'drives':fit.x.tolist(),'target_signed_B_m':.05*sign,'residual':error,'matrix':matrix.tolist(),'nfev':fit.nfev}
        solutions.append(row)
        print('AXIS_FIT',json.dumps(row),flush=True)
        if error<1e-8:
            break
valid=[row for row in solutions if row['residual']<1e-6]
if not valid:
    (OUT/'axis-proposals.json').write_text(json.dumps(solutions,indent=2)+'\n',encoding='utf-8')
    raise ValueError('No fixed-P1/P2 D/I proposal reached both axial constraints')
best=min(valid,key=lambda row:np.linalg.norm((np.asarray(row['drives'])-np.array([original[keys[0]],original[keys[1]]]))/10.))
for key,value in zip(keys[:2],best['drives']): lenses[key].percent=float(value)
state.sync_objective()
prepared=capture_instrument_snapshot(state)
snapshot_path=OUT/'projector-inputs.json.gz'
snapshot_path.write_bytes(gzip.compress(json.dumps(prepared.to_dict(),indent=2).encode(),mtime=0))
profile_path=OUT/'projector-input.toml'
save_profile(profile_path,state,selection)
runtime=get_unified_field_runtime(state)
sampler=compile_unified_potential_sampler(runtime.snapshot,prefer_plane=True)
record={'scope':'Independent two-condenser zero-current projector reference. Source emitter is forbidden. Not a finite-source, finite-angle, specimen/detector-signal or wave qualification.',
        'source_plane':'specimen upper surface','source_z_mm':source*1e3,'reference_z_mm':reference*1e3,
        'upper_controls_percent':controls,'original_projector_percent':original,
        'projector_percent':{key:float(lenses[key].percent) for key in keys},
        'receiver_insertion':{x.key:x.inserted for x in (*state.stem_detectors,state.fluorescent_screen,state.camera)},
        'proposal_model':'Axisymmetric Larmor RK4 on sampled Bz from the same captured static potential; constant 300 keV; no legacy fitted imaging map.',
        'proposal_axial_maximum_step_mm':float(np.max(np.diff(z))*1e3),
        'proposal_source_B_t_by_unit_percent':{key:float(profile[0]) for key,profile in zip(keys[:2],profiles)},
        'baseline_axial_matrix':base.tolist(),'axis_solutions':solutions,'selected_axis_solution':best,
        'state_physical_digest':prepared.physical_digest,'input_snapshot_digest':prepared.digest,
        'c2_aperture_diameter_mm':aperture_mm,
        'input_snapshot':snapshot_path.name,'input_profile':profile_path.name,
        'proposal_axis_potential_min_v':float(np.min(full['phi_v'])),
        'proposal_axis_potential_max_v':float(np.max(full['phi_v'])),
        'artifacts':{path.name:{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size} for path in (Path(__file__),snapshot_path,profile_path,settings_path)},
        'no_finite_source_population_was_created':True,'emitter_calls':0,'runs':[]}
for step_mm,scale in ((.1,1.),(.05,1.),(.05,.5)):
    hp,ha=1e-10*scale,1e-7*scale
    initial=np.zeros((9,4))
    for i,h in enumerate((hp,hp,ha,ha)):
        initial[1+2*i,i]=h;initial[2+2*i,i]=-h
    inputpoints=np.c_[initial[:5,:2],np.full(5,source)]
    A=np.asarray(sampler.sample(inputpoints)['A_t_m'])
    grad=np.column_stack(((A[1,:2]-A[2,:2])/(2*hp),(A[3,:2]-A[4,:2])/(2*hp)))
    basis=np.eye(4);basis[2:,:2]=e/p*grad
    t=time.perf_counter()
    response=trace_unified_response(state,initial,source,np.array([reference]),energy_ev=energy,flight_time_s=0.,maximum_step_m=step_mm*1e-3)
    phase=np.asarray(response['xy_slopes'])[0]
    jac=np.column_stack([(phase[1+2*i]-phase[2+2*i])/(2*inc) for i,inc in enumerate((hp,hp,ha,ha))])
    canonical=jac@basis;Ablock=canonical[:2,:2];B=canonical[:2,2:]
    camera=math.sqrt(abs(float(np.linalg.det(B))))
    row={'maximum_step_mm':step_mm,'actual_uniform_step_bound_mm':min(float(state.step_mm),step_mm),'difference_scale':scale,
         'position_increment_m':hp,'mechanical_slope_increment_rad':ha,'elapsed_s':time.perf_counter()-t,
         'all_pencils_reach_reference':bool(np.asarray(response['alive']).all()),
         'blocked_key':list(response['blocked_key']), 'canonical_to_mechanical_basis':basis.tolist(),
         'mechanical_jacobian':jac.tolist(),'canonical_jacobian':canonical.tolist(),
         'canonical_position_residual_norm':float(np.linalg.norm(Ablock,ord=2)),
         'effective_camera_length_m':camera,'camera_length_singular_values_m':np.linalg.svd(B,compute_uv=False).tolist(),
         'camera_length_relative_error_to_50mm':abs(camera/.05-1), 'receipt':thaw_json(response['receipt'])}
    record['runs'].append(row)
    assert capture_instrument_snapshot(state).digest==prepared.digest,'Observer mutated prepared controls'
    (OUT/'proposal.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('PRODUCTION',json.dumps({key:row[key] for key in ('maximum_step_mm','difference_scale','elapsed_s','all_pencils_reach_reference','canonical_position_residual_norm','effective_camera_length_m','camera_length_relative_error_to_50mm')}),flush=True)
record['status']='UNIFIED_DIFFERENTIAL_REFERENCE_ONLY'
record['all_differential_checks_pass']=all(row['all_pencils_reach_reference'] and row['canonical_position_residual_norm']<=1e-3 and row['camera_length_relative_error_to_50mm']<=.03 for row in record['runs'])
record['source_controls_unchanged']=True
record['suggested_column_mode_override']={'calibration_status':'computed_unified_c2_zero_current_diffraction_reference',
    'calibration_reference':'Independent C1+C2 Nanoprobe diffraction differential reference at the recorded Objective and specimen upper surface. D/I jointly matched with P1/P2 held fixed. Verified by production unified zero-current pencil responses at 0.1 and 0.05 mm and half increments; source emission forbidden. No finite-source, finite-angle, detector-signal, wave or experimental qualification.',
    'compatible_recording_systems':['No Energy Filter'],
    'devices':{key:{'percent':float(lenses[key].percent)} for key in keys},'apertures':{},
    'targets':{'conjugate_plane':'mode_dependent_diffraction_reference_plane','reference_surface':'stem_main_screen',
       'calibration_model':'unified','calibration_scope':'zero_current_differential_pencils_only',
       'source_plane':'specimen_upper_surface','source_z_mm':source*1e3,'reference_z_mm':reference*1e3,
       'objective_excitation_percent':controls['objective_lens'],'target_effective_camera_length_m':.05,
       'achieved_effective_camera_length_m':record['runs'][-1]['effective_camera_length_m'],
       'achieved_diffraction_conjugacy_residual':record['runs'][-1]['canonical_position_residual_norm'],
       'validation_step_mm':.05,'validation_steps_mm':[.1,.05],'diagnostic_pencil_count':9,'c2_aperture_diameter_um':70.,'finite_source_qualification':'NOT_TESTED','finite_angle_qualification':'NOT_TESTED','wave_qualification':'NOT_TESTED'}}
(OUT/'proposal.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print('FINAL',record['all_differential_checks_pass'],flush=True)

