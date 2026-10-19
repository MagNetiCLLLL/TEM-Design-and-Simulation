"""Full archive loading against actual numerical table construction, no transport."""
from hashlib import sha256
import json
import numpy as np
import pytest
from scipy.stats import qmc
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from scripts.source_radial_quadrature import solve_radial_partitions, gauss_table
from scripts.analyse_image_intensity_convergence import CASES, PROJECTOR_KEYS, canonical_digest
from scripts.validate_image_intensity_convergence import array_digest
from scripts.validate_conditional_source_convergence import compose_energy_orbit, reweight_radial_table, weighted_root_budget
from scripts.validate_conditional_source_convergence import quadrature_arrays
from scripts.analyse_conditional_source_convergence import (
    ENERGY_FIELDS, load_conditional_runs, analyse_conditional, compact_summary,
    validate_weighted_roots)
from test_conditional_source_analysis import sample_run


def write_energy_archive(path, m=2, orders=(2,4), band=(.143,.149)):
    path.mkdir(exist_ok=True)
    def save_json(name, values):
        p=path/name;p.write_text(json.dumps(values));return sha256(p.read_bytes()).hexdigest()
    def save_arrays(name, values):
        p=path/name;np.savez(p,**values);return sha256(p.read_bytes()).hexdigest()
    emitter=ColdFieldEmitter(0.,1.,1.)
    bases=qmc.Sobol(5,scramble=True,seed=3,bits=30).random_base2(m.bit_length()-1)
    plan=compose_energy_orbit(emitter,bases,8,band);expanded=plan['points'];r=len(expanded)
    channels=tuple(f'{case}:{sign:+d}' for case in CASES for sign in (1,-1))
    shifts=np.repeat([0.,-1e-4,-.002,.001],2)
    def margin(base,u):return (.45+.02*expanded[base,3]-u)[:,None]+shifts[None,:]
    solution=solve_radial_partitions(margin,r,channels,u_tol=1e-12)
    part=dict(roots=solution.roots,partitions=[dict(base_index=p.base_index,edges=p.edges.tolist())
        for p in solution.partitions],weighted_root_budget=weighted_root_budget(
            solution.roots,plan['mapping']['base_probability'],channels))
    pdigest=canonical_digest(part)
    controls=dict.fromkeys(PROJECTOR_KEYS,20.)
    energy_physical=dict(schema=plan['energy_metadata']['schema'],band_ev=list(band),cuts_metadata=plan['cuts_metadata'])
    physical=dict(step_mm=.00625,source_base_seed=3,projector_controls=controls,nominal_current_a=1e-5,
        azimuth_orbit_schema='offline-common-source-azimuth-orbit-v1',energy_stratification=energy_physical)
    protocol=dict(physical_protocol=physical,implementation='synthetic archive only')
    snapshot=dict(digest='captured synthetic energy law',graph=dict(nodes=[
        dict(type='temsim.optics.electron_gun.emitter:ColdFieldEmitter',
             attributes={name:getattr(emitter,name) for name in ENERGY_FIELDS})]))
    snapshot_sha=save_json('snapshot.json',snapshot)
    d=dict(schema='conditional-source-convergence-v1',status='COMPLETE',implementation='synthetic archive only',
        base_count=m,base_seed=3,azimuth_orbit_size=8,expanded_base_count=r,
        step_mm=.00625,receiver_z_mm=2700.,projector_controls=controls,
        protocol=protocol,protocol_digest=canonical_digest(protocol),physical_protocol=physical,
        physical_protocol_digest=canonical_digest(physical),
        energy_stratification=plan['energy_metadata'],energy_cuts_metadata=plan['cuts_metadata'],
        base_points_path='base.npz',base_points_sha256=save_arrays('base.npz',{'points':bases}),
        base_digest=array_digest({'points':bases}),expanded_base_points_path='expanded.npz',
        expanded_base_points_sha256=save_arrays('expanded.npz',{'points':expanded}),
        expanded_base_digest=array_digest({'points':expanded}),partition_path='partition.json',
        partition_sha256=save_json('partition.json',part),partition_digest=pdigest,
        root_diagnostics=solution.diagnostics,point_spread=dict(model='gaussian',sigma_x_mm=.6,sigma_y_mm=.6,rotation_deg=0.),
        sources={},runs={})
    for prefix,values in [('energy_cuts',plan['cuts']),('source_base_measure',plan['mapping'])]:
        d[prefix+'_path']=prefix+'.npz';d[prefix+'_sha256']=save_arrays(prefix+'.npz',values)
        d[prefix+'_digest']=array_digest(values)
    for order in orders:
        group=f'M{m}_E3_K8_G{order}'
        radial=reweight_radial_table(gauss_table(solution,order=order),plan['mapping']['base_probability'])
        table=quadrature_arrays(expanded,radial)
        table.update({key:value[table['base_index']] for key,value in plan['mapping'].items()})
        record={key:table[key].tolist() for key in ('coordinates','weights','signs','ray_ids')}
        record.update(schema='tip-explicit-cdf-v1',role='quadrature',construction='analytical test fixture')
        tdigest=canonical_digest(record);n=len(table['weights'])
        source=dict(gun_exit_weight=table['weights'],gun_exit_ray_id=table['ray_ids'],
            gun_exit_alive=np.ones(n,bool),gun_blocked_key=np.full(n,''),gun_blocked_z_mm=np.full(n,np.nan))
        sdigest=array_digest(source)
        receipt=dict(base_count=m,gauss_order=order,azimuth_orbit_size=8,expanded_base_count=r,
            partition_digest=pdigest,source_table_digest=tdigest,source_digest=sdigest,
            table_record_path=group+'-table.json',table_record_sha256=save_json(group+'-table.json',record),
            table_path=group+'-table.npz',table_sha256=save_arrays(group+'-table.npz',table),
            source_path=group+'-source.npz',source_sha256=save_arrays(group+'-source.npz',source),
            snapshot_path='snapshot.json',snapshot_sha256=snapshot_sha,emitted_current_a=1e-5)
        d['sources'][group]=receipt
        margins=margin(table['base_index'],table['angular_cdf'])
        for i,case in enumerate(CASES):
            label=group+'_'+case;physical_pass=margins[np.arange(n),2*i+(table['signs']<0)]>=0
            xy=np.column_stack((1e-7*(table['angular_cdf']-.5),2e-8*(expanded[table['base_index'],1]-.5)))
            xy[:,0]+=i*1e-9
            arrays=dict(xy_m=xy,weight=table['weights'],source_id=table['ray_ids'],
                survivor=physical_pass&(table['weights']>0),blocked_key=np.where(physical_pass,'flu_screen','aperture'),
                blocked_z_mm=np.where(physical_pass,2700.,765.),final_alive=np.zeros(n,bool))
            row=dict(base_count=m,gauss_order=order,azimuth_orbit_size=8,expanded_base_count=r,
                case=case,status='COMPLETE',particles_path=label+'.npz',particles_sha256=save_arrays(label+'.npz',arrays),
                snapshot_path='snapshot.json',snapshot_sha256=snapshot_sha,source_table_digest=tdigest,
                source_digest=sdigest,partition_digest=pdigest,receiver_z_mm=2700.,projector_controls=controls.copy(),
                implementation_before=d['implementation'],implementation_after=d['implementation'])
            row['request_identity']=canonical_digest(dict(protocol=d['protocol_digest'],state=snapshot['digest'],
                target_z_mm=2700.,run_id=label,source_table=tdigest,partition=pdigest))
            d['runs'][label]=row
    save_json('manifest.json',d)
    return d


def test_complete_energy_archives_reconstruct_weights_and_keep_m_and_gauss_distinct(tmp_path):
    for m in (2,4):write_energy_archive(tmp_path/f'M{m}',m)
    runs,info=load_conditional_runs([tmp_path/'M2',tmp_path/'M4'])
    assert len(runs)==4 and not info['pending']
    report=analyse_conditional(runs,pixels=[32]);summary=compact_summary(report)
    assert report['axis_availability']['base_count']=='EVALUATED'
    assert report['axis_availability']['radial_gauss_order']=='EVALUATED'
    for run in runs:
        assert run.label==f'M{run.base_count}_E3_K8_G{run.gauss_order}'
        assert run.expanded_base_count>run.base_count*4
        assert run.uncertainty['source_table_measure_receipt']['status']=='EXACT_ARCHIVED_SOURCE_MEASURE_RECONSTRUCTION'
    for row in summary['cases']:
        expected={'Cs_plus20mm':-1e-4,'A1_plus0.03mm':-.002,'B2_plus1mm':.001}[row['case']]
        assert row['signed_full_plane_probability_change']==pytest.approx(expected,abs=2e-12)
        assert row['source_measure_method']['band_ev']==[.143,.149]
        assert row['zero_weight_probe_nodes']>0
        assert row['actual_centroid_change_mm'][0]!=0.
    assert all(row['qualification_status']=='UNRESOLVED_BOUNDARY_SCOPE' for row in report['axes'])


def test_loader_rejects_rehashed_wrong_per_node_energy_measure(tmp_path):
    d=write_energy_archive(tmp_path,orders=(2,));receipt=next(iter(d['sources'].values()))
    path=tmp_path/receipt['table_path']
    with np.load(path) as f:table={key:f[key] for key in f.files}
    table['base_probability'][0]*=2
    np.savez(path,**table);receipt['table_sha256']=sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='lost its original energy/base mapping'):
        load_conditional_runs([tmp_path])


def test_energy_and_uniform_or_different_band_are_not_one_refinement_axis(tmp_path):
    write_energy_archive(tmp_path/'a',orders=(2,))
    write_energy_archive(tmp_path/'b',orders=(4,),band=(.14,.15))
    a,_=load_conditional_runs([tmp_path/'a']);b,_=load_conditional_runs([tmp_path/'b'])
    with pytest.raises(ValueError,match='cannot mix'):
        analyse_conditional(a+b,pixels=[32])
    with pytest.raises(ValueError,match='cannot mix'):
        analyse_conditional(a+[sample_run(4,2)],pixels=[32])


def test_rehashed_wrong_weighted_root_budget_is_rejected(tmp_path):
    d=write_energy_archive(tmp_path,orders=(2,));path=tmp_path/d['partition_path']
    part=json.loads(path.read_bytes());part['weighted_root_budget']['per_channel']['baseline:+1']+=1e-12
    path.write_text(json.dumps(part));d['partition_sha256']=sha256(path.read_bytes()).hexdigest()
    d['partition_digest']=canonical_digest(part)
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='original nonuniform source probabilities'):
        load_conditional_runs([tmp_path])


def recorded_eight_channel_budget():
    # Actual M32 channel totals exposed a one-ULP difference between adding
    # four channels in archive order and grouping the two signs first.
    totals=np.array([2.179551914756805e-13,2.179551914756805e-13,
        2.212287367708648e-13,2.212287367708648e-13,
        2.1796528474583842e-13,2.1796528474583842e-13,
        2.2125299809121026e-13,2.206321682330844e-13])
    channels=tuple(f'{case}:{sign:+d}' for case in CASES for sign in (1,-1))
    probability=np.array([.25,.75])
    roots=[dict(base_index=0,channel_key=key,u=float(4*value),
        bracket=[0.,float(8*value)],bracket_width=float(8*value),
        actual_margin_residual=0.) for key,value in zip(channels,totals)]
    archive=json.loads(json.dumps(weighted_root_budget(roots,probability,channels)))
    assert list(archive['per_channel'].values())==totals.tolist()
    return dict(roots=roots,weighted_root_budget=archive),dict(
        all_located_roots_converged=True,root_count=len(roots),angular_cdf_tolerance=2e-12),dict(
        base_probability=probability)


def test_recorded_channel_totals_validate_without_relaxing_exact_budget_identity():
    partition,diagnostics,measure=recorded_eight_channel_budget()
    values=np.array(list(partition['weighted_root_budget']['per_channel'].values()))
    for index in (1,2):
        case=CASES[index]
        paired=(values[0]+values[1])+(values[2*index]+values[2*index+1])
        archived=partition['weighted_root_budget']['difference_bound_by_case'][case]
        assert abs(archived-paired)==np.spacing(archived)
    actual=validate_weighted_roots(partition,diagnostics,measure)
    assert actual['per_channel']==partition['weighted_root_budget']['per_channel']
    assert actual['case_minus_baseline']==partition['weighted_root_budget']['difference_bound_by_case']


@pytest.mark.parametrize('changed', ['channel_one_ulp','total_one_ulp','source_probability'])
def test_recorded_channel_budget_still_rejects_modified_measure_or_budget(changed):
    partition,diagnostics,measure=recorded_eight_channel_budget()
    budget=partition['weighted_root_budget']
    if changed=='channel_one_ulp':
        budget['per_channel']['baseline:+1']=float(np.nextafter(budget['per_channel']['baseline:+1'],np.inf))
    elif changed=='total_one_ulp':
        budget['difference_bound_by_case'][CASES[1]]=float(np.nextafter(
            budget['difference_bound_by_case'][CASES[1]],np.inf))
    else:
        measure['base_probability']=np.array([.26,.74])
    with pytest.raises(ValueError,match='original nonuniform source probabilities'):
        validate_weighted_roots(partition,diagnostics,measure)
