"""Bounded native local-linear diagnostic; 17 mathematical probes, no beam replacement."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from run_column_energy_diagnostic import ROOT, DEFAULT_DONOR, PHASE, load_donor, file_hash, array_hash, verify_initial_energy
from solve_joint_probe import atomic_json, atomic_npz

SCHEMA='temsim-native-column-local-linear-v1'
KINETIC_EV=300000.3
DEFAULT_CONSTANT=ROOT/'outputs/probe-tail-attribution-20261028/poly-phase-constant-k769-h005'
STEPS=np.array([1e-9,1e-9,1e-8,1e-8])
POSITION_STABILITY_NM=.01
SLOPE_STABILITY_RAD=1e-6


def require(condition,message):
    if not condition:raise ValueError(message)


def calibration_inputs(source,steps=STEPS):
    require(np.shape(steps)==(4,) and np.isfinite(steps).all() and np.all(steps>0),'Four finite positive coordinate steps required')
    w=np.asarray(source['weights']);live=np.asarray(source['gun_exit_alive']) & (w>0)
    require(live.all(),'This diagnostic requires the complete unfiltered transmitting donor')
    phase=np.column_stack([source['gun_exit_'+k] for k in ('x_m','y_m','tx_rad','ty_rad')])
    require(np.isfinite(phase).all() and np.isfinite(w).all() and w.sum()>0,'Finite complete reference source required')
    reference=np.average(phase,axis=0,weights=w)
    clock=float(np.average(source['gun_exit_flight_time_s'],weights=w))
    probes=np.vstack((reference,reference+np.diag(steps),reference-np.diag(steps),
                      reference+np.diag(steps/2),reference-np.diag(steps/2)))
    require(np.unique(probes,axis=0).shape==(17,4),'Perturbations must be distinct representable phase points')
    return phase,reference,clock,probes


def prepare_plan(donor):
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.unified_field_particle import build_unified_plan
    from measure_probe_waist import audit_extended_plan
    runtime=get_unified_field_runtime(donor.state)
    require(type(runtime.snapshot) is ElectromagneticPotentialSnapshot,'Only proven static captured fields are admitted')
    require(runtime.metadata.get('dynamic_snapshot_digest') is None,'Dynamic metadata cannot supply a static diagnostic')
    z=donor.arrays['integration_z_mm'];cp=donor.arrays['z_mm']
    plan=build_unified_plan(donor.state,float(z[0]),float(z[-1]),save_z_mm=tuple(z),checkpoint_z_mm=tuple(cp),_captured_runtime=runtime)
    require(np.array_equal(plan.z_mm,z) and np.array_equal(plan.z_mm[plan.checkpoint_index],cp),'Exact donor integration and checkpoint mesh required')
    audit=audit_extended_plan(donor.state,plan,float(z[-1]))
    require(audit['identities']==donor.field_identity,'Field, geometry or plan identity changed')
    return plan,audit


def execute_probes(donor,plan,probes,clock,cancelled):
    from temsim.physics.unified_field_particle import execute_unified_plan, merge_unified_stops
    from temsim.physics.recording_clipping import clip_recording_planes
    donor.state._tuning_cancelled=cancelled
    require(not cancelled(),'Cancelled before calibration execution')
    k=np.full(17,KINETIC_EV);times=np.full(17,clock)
    result=execute_unified_plan(donor.state,plan,probes[:,0],probes[:,2],probes[:,1],probes[:,3],
        initial_kinetic_energy_ev=k,initial_time_s=times,return_flight_times=True,history_policy='checkpoints')
    z,x,tx,y,ty,time,cp=result
    alive,blocked,labels=merge_unified_stops(z,np.ones(17,bool),np.full(17,np.nan),['']*17)
    alive,blocked,labels=clip_recording_planes(donor.state,z,x,y,alive,blocked,labels)
    internal=cp.unified_state
    raw={key:np.asarray(getattr(cp,key)).copy() for key in PHASE}
    raw.update(z_mm=cp.z_mm.copy(),integration_z_mm=plan.z_mm.copy(),active=internal.active.copy(),
        terminal_alive=alive,blocked_z_mm=blocked,blocked_key=np.asarray(labels,dtype=str),
        reference_energy_ev=internal.reference_energy_ev.copy(),reference_phi_v=internal.reference_phi_v.copy(),
        induction_work_ev=internal.induction_work_ev.copy(),swept_bounds_m=internal.swept_bounds_m.copy(),
        calibration_initial_xy_slopes=probes.copy(),calibration_initial_clock_s=times)
    return raw,dict(runtime_digest=internal.runtime_digest,implementation_digest=internal.implementation_digest,
        emission_time_s=internal.emission_time_s,spatial_dependency_identities=list(internal.spatial_dependency_identities))


def validate_probes(raw,donor,probes,clock):
    require(np.array_equal(raw['z_mm'],donor.arrays['z_mm']),'Calibration lost exact checkpoints')
    require(raw['active'].all() and raw['terminal_alive'].all() and np.isnan(raw['blocked_z_mm']).all()
        and not np.any(raw['blocked_key']!=''),'Every mathematical calibration probe must survive without any contact')
    require(all(np.shape(raw[key])==(len(raw['z_mm']),17) and np.isfinite(raw[key]).all() for key in PHASE),'Invalid complete calibration phase')
    require(np.all(raw['kinetic_energy_ev']>0) and np.all(raw['flight_time_s']>=0),'Invalid calibration energy or clock')
    entrance=np.stack([raw[k][0] for k in ('x_m','y_m','tx_rad','ty_rad')],axis=-1)
    require(np.array_equal(entrance,probes) and np.array_equal(raw['flight_time_s'][0],np.full(17,clock)),'Calibration entrance phase or clock changed')
    require(np.array_equal(raw['reference_energy_ev'],np.full(17,KINETIC_EV)),'Calibration must use the exact constant kinetic energy')
    require(np.count_nonzero(raw['induction_work_ev'])==0,'Static-field diagnostic cannot contain induction work')
    verify_initial_energy(raw['kinetic_energy_ev'][0],raw['reference_energy_ev'],raw['reference_phi_v'],np.ones(17,bool))


def derive_prediction(raw,probes,source_phase,reference):
    values=np.stack([raw[k] for k in ('x_m','y_m','tx_rad','ty_rad')],axis=-1)
    require(values.shape[1:]==(17,4),'17 four-coordinate mathematical probes required')
    derivatives=[];predictions=[]
    for first,second in ((1,5),(9,13)):
        width=np.array([probes[first+j,j]-probes[second+j,j] for j in range(4)])
        require(np.all(width>0),'Central difference denominators must be positive')
        jac=((values[:,first:first+4]-values[:,second:second+4])/width[None,:,None]).transpose(0,2,1)
        pred=values[:,0,None,:]+np.einsum('zoi,ni->zno',jac,source_phase-reference)
        derivatives.append(jac);predictions.append(pred)
    return derivatives,predictions


def shape_rows(phase,z,planes,weights,ids,surface):
    from temsim.physics.beam_statistics import transverse_beam_statistics,_weighted_quantile
    rows=[]
    for plane in planes:
        where=np.flatnonzero(z==plane);require(len(where)==1,'Exact observation required')
        i=int(where[0]);x,y,tx,ty=phase[i].T
        stats=transverse_beam_statistics(x,y,tx,ty,alive=np.ones(len(weights),bool),weights=weights)
        radius=np.hypot(x-stats.mean_x_m,y-stats.mean_y_m);watched=np.flatnonzero(ids==306)
        rows.append(dict(z_mm=float(plane),offset_nm=float((plane-surface)*1e6),d95_nm=float(2e9*stats.radius_95_m),
            d90_nm=float(2e9*_weighted_quantile(radius,weights/weights.sum(),.9)),rms_radius_nm=float(1e9*stats.radius_rms_m),
            alpha95_mrad=float(stats.convergence_95_mrad),centroid_xy_m=[stats.mean_x_m,stats.mean_y_m],
            id306_radius_nm=float(radius[watched[0]]*1e9) if len(watched)==1 else None))
    return rows


def comparison(donor,actual,probes,raw,phase,reference):
    from measure_probe_waist import observed_minimum
    jac,pred=derive_prediction(raw,probes,phase,reference)
    actual_phase=np.stack([actual[k] for k in ('x_m','y_m','tx_rad','ty_rad')],axis=-1)
    z=raw['z_mm'];surface=float(donor.state.sample.upper_surface_z_mm);planes=donor.planes
    require(actual['alive'].all(),'Nonlinear comparator must preserve the complete transmitted source for this diagnostic')
    selected=np.array([int(np.flatnonzero(z==plane)[0]) for plane in planes])
    delta=pred[0]-pred[1]
    xy=np.linalg.norm(delta[selected,:,:2],axis=-1)*1e9;angles=np.linalg.norm(delta[selected,:,2:],axis=-1)
    stable=dict(maximum_prediction_xy_difference_nm=float(xy.max()),maximum_prediction_slope_difference_rad=float(angles.max()),
        position_limit_nm=POSITION_STABILITY_NM,slope_limit_rad=SLOPE_STABILITY_RAD,
        scope='Every original source point at each exact requested through-focus plane',
        passed=bool(xy.max()<=POSITION_STABILITY_NM and angles.max()<=SLOPE_STABILITY_RAD))
    summaries={}
    for label,values in (('actual_full_field',actual_phase),('linear_prediction_h',pred[0]),('linear_prediction_half_h',pred[1])):
        rows=shape_rows(values,z,planes,donor.arrays['weights'],donor.arrays['source_id'],surface)
        summaries[label]=dict(scope='Executed full original population' if label.startswith('actual') else 'Mathematical first-order prediction using original weights; not propagated and not a physical beam result',
            surface=next(r for r in rows if r['z_mm']==surface),d95_minimum=observed_minimum(planes,[r['d95_nm'] for r in rows],surface),planes=rows)
    remainder=actual_phase-pred[1];i=int(np.flatnonzero(z==surface)[0]);j=int(np.flatnonzero(donor.arrays['source_id']==306)[0])
    rawnorm=np.linalg.norm(remainder[i,:,:2],axis=-1)*1e9
    summary=dict(stability=stable,profiles=summaries,
        nonlinear_remainder=dict(definition='Actual full-field raw phase minus local first-order mathematical prediction; no coordinate rescaling or centroid subtraction',
            surface_maximum_xy_nm=float(rawnorm.max()),surface_maximum_source_id=int(donor.arrays['source_id'][rawnorm.argmax()]),
            id306_surface_xy_remainder_nm=(remainder[i,j,:2]*1e9).tolist(),id306_surface_xy_remainder_norm_nm=float(rawnorm[j]),
            id306_actual_surface_xy_nm=(actual_phase[i,j,:2]*1e9).tolist(),id306_predicted_surface_xy_nm=(pred[1][i,j,:2]*1e9).tolist(),
            scope='A local linearization remainder; not a unique Cs coefficient or additive decomposition of D95'))
    arrays=dict(z_mm=z,source_id=donor.arrays['source_id'],original_absolute_weights=donor.arrays['weights'],
        jacobian_h=jac[0],jacobian_half_h=jac[1],prediction_h_xy_slopes=pred[0],prediction_half_h_xy_slopes=pred[1],
        actual_minus_half_h_prediction_xy_slopes=remainder)
    return summary,arrays


def run(donor_path,constant_path,output,threads=4,budget_s=900.,cancel_file=None):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.cpu_resources import numerical_job
    require(type(threads) is int and 1<=threads<=4 and np.isfinite(budget_s) and budget_s>0,'Bounded resources required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=perf_counter();cancelled=lambda:perf_counter()-started>=budget_s or bool(cancel_file and Path(cancel_file).exists())
    report=dict(schema=SCHEMA,status='RUNNING',qualification='LOCAL_LINEAR_DIAGNOSTIC_NOT_QUALIFIED',
        mathematical_probe_count=17,physical_current_assigned_to_calibration_probes=False,full_source_propagations=0,
        threads_requested=threads,budget_s=budget_s,steps_xy_m_slopes_rad=STEPS.tolist(),
        difference_scale_rationale='1 nm is small relative to the sub-micron gun-exit source; 10 nrad is below its angular spread. Exact h/2 predictions provide the required empirical stability check.',
        acceptance_limits=dict(maximum_xy_prediction_change_nm=POSITION_STABILITY_NM,maximum_slope_prediction_change_rad=SLOPE_STABILITY_RAD),
        no_size_acceptance_threshold=True,driver_sha256=file_hash(__file__))
    atomic_json(output/'report.json',report)
    try:
        donor=load_donor(donor_path);constant_path=Path(constant_path).resolve()
        measured=json.loads((constant_path/'report.json').read_text());checks=json.loads((constant_path/'checksums.json').read_text())
        require(measured['status']=='COMPLETE' and measured['full_original_population'] and measured['mode']=='constant'
            and measured['intervention']['constant_kinetic_ev']==KINETIC_EV,'Completed exact constant-K comparator required')
        require(measured['donor_files_sha256']==donor.files_sha256,'Constant-K result belongs to another donor')
        for name in ('report.json','particles.npz','injection.npz'):
            require(file_hash(constant_path/name)==checks[name],'Constant-K evidence checksum mismatch')
        with np.load(constant_path/'particles.npz',allow_pickle=False) as f:actual={k:f[k].copy() for k in f.files}
        with np.load(constant_path/'injection.npz',allow_pickle=False) as f:
            require(np.array_equal(f['injected_kinetic_energy_ev'],np.full(len(donor.arrays['source_id']),KINETIC_EV)),'Changed comparator energies')
        for key in ('source_id','weights','z_mm','integration_z_mm'):
            require(np.array_equal(actual[key],donor.arrays[key]),'Comparator changed exact source/grid: '+key)
        phase,reference,clock,probes=calibration_inputs(donor.arrays)
        plan,audit=prepare_plan(donor)
        report.update(donor_files_sha256=donor.files_sha256,constant_case_path=str(constant_path),
            constant_case_hashes={k:checks[k] for k in ('report.json','particles.npz','injection.npz')},
            input_digest=donor.snapshot.digest,reference_xy_m_slopes_rad=reference.tolist(),reference_clock_s=clock,
            reference_definition='Actual current-weighted original poly gun-exit mean, used only as the mathematical expansion point',
            kinetic_energy_ev=KINETIC_EV,field_audit=audit,source_points=len(phase),actual_source_current_a=measured['source_current_a'])
        atomic_json(output/'inputs.json',donor.snapshot.to_dict())
        atomic_npz(output/'reference-and-source.npz',dict(original_source_id=donor.arrays['source_id'],original_absolute_weights=donor.arrays['weights'],
            original_gun_exit_xy_slopes=phase,reference_xy_slopes=reference,calibration_initial_xy_slopes=probes,original_gun_exit_clock_s=donor.arrays['gun_exit_flight_time_s']))
        (output/'driver-source.py').write_bytes(Path(__file__).read_bytes())
        atomic_json(output/'report.json',report)
        with numerical_job(threads,cancelled=cancelled) as cpu:
            report['cpu_resources']=cpu.to_dict();atomic_json(output/'report.json',report)
            raw,receipt=execute_probes(donor,plan,probes,clock,cancelled)
        atomic_npz(output/'calibration-probes.npz',raw);report['execution_receipt']=receipt
        validate_probes(raw,donor,probes,clock)
        require(receipt['runtime_digest']==plan.unified_runtime.digest and receipt['implementation_digest']==donor.snapshot.to_dict()['implementation'],'Probe runtime/implementation mismatch')
        require(capture_instrument_snapshot(donor.state).digest==donor.snapshot.digest,'Frozen donor inputs changed')
        summary,predictions=comparison(donor,actual,probes,raw,phase,reference)
        atomic_npz(output/'mathematical-predictions.npz',predictions);report['comparison']=summary
        require(all(file_hash(donor.directory/k)==v for k,v in donor.files_sha256.items()),'Original donor files changed')
        require(all(file_hash(constant_path/k)==checks[k] for k in ('report.json','particles.npz','injection.npz')),'Original constant-K files changed')
        report['status']='COMPLETE';report['stability_status']='PASS' if summary['stability']['passed'] else 'UNRESOLVED_STEP_SENSITIVITY'
        report['interpretation']='A native local-linear diagnostic only. Mathematical predictions have no independent physical-current or qualification claim. No controls were tuned and no empirical aberration was added.'
    except BaseException as error:
        report.update(status='CANCELLED' if cancelled() or isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'FAILED',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['elapsed_s']=perf_counter()-started;atomic_json(output/'report.json',report)
        atomic_json(output/'checksums.json',{str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--donor',type=Path,default=DEFAULT_DONOR);parser.add_argument('--constant-case',type=Path,default=DEFAULT_CONSTANT)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--budget-s',type=float,default=900.);parser.add_argument('--cancel-file',type=Path)
    a=parser.parse_args();r=run(a.donor,a.constant_case,a.output,a.threads,a.budget_s,a.cancel_file)
    print(json.dumps({'status':r['status'],'stability_status':r['stability_status']}))
    raise SystemExit(0 if r['stability_status']=='PASS' else 1)
