"""Offline conditional-axis checks, without source tracing or fitted optics."""
import json
from hashlib import sha256
import numpy as np
import pytest
from scripts.analyse_conditional_source_convergence import (ConditionalRun,analyse_conditional,
    load_conditional_runs,located_root_bounds,load_final_anchors,compare_final_anchors,compact_summary,
    analysis_source_identity,verify_analysis_source_identity)
from scripts.analyse_image_intensity_convergence import CASES, Cloud, PROJECTOR_KEYS, canonical_digest
from scripts.validate_image_intensity_convergence import array_digest


def test_analysis_identity_rejects_imported_reference_change(tmp_path):
    files=(tmp_path/'analyser.py',tmp_path/'gaussian_reference.py')
    for path in files:path.write_text('original numerical implementation')
    identity=analysis_source_identity(files)
    verify_analysis_source_identity(identity,files)
    files[1].write_text('changed reference formula')
    with pytest.raises(RuntimeError,match='imported numerical reference changed'):
        verify_analysis_source_identity(identity,files)


def sample_run(m,order,*,bound=0.,factor=None):
    n=m*order*2
    xy=np.tile([[-.0001,.00002],[.0001,-.00002]],(n//2,1))
    weight=np.full(n,1/n); ids=np.arange(n); alive=np.ones(n,dtype=bool)
    factor=1+1/m+1/order**2 if factor is None else factor
    clouds={'baseline':Cloud(xy.copy(),weight,alive,ids)}
    for i,case in enumerate(CASES[1:],1):
        clouds[case]=Cloud(xy+[factor*i*1e-7,0.],weight,alive,ids)
    return ConditionalRun(f'M{m}_G{order}',m,order,0,np.arange(m*5).reshape(m,5)/100,
        f'partition{m}',f'table{m}_{order}','same physics',.00625,clouds,{},dict.fromkeys(CASES[1:],bound),
        dict(topology_completeness='SAMPLED_BRACKETS_ONLY_NOT_CERTIFIED',batch_drift_cdf_bound=None))


def test_m_and_gauss_refinement_are_independent_and_do_not_claim_overall_convergence():
    runs=[sample_run(m,order) for m in (2,4) for order in (2,4)]
    report=analyse_conditional(runs,pixels=[32])
    assert report['axis_availability']==dict(base_count='EVALUATED',radial_gauss_order='EVALUATED',
        azimuth_orbit_size='NOT_RUN_NO_INDEPENDENT_AXIS_PAIR')
    assert len(report['axes'])==12
    for row in report['axes']:
        if row['axis']=='base_count':
            assert row['coarse_gauss_order']==row['fine_gauss_order']
            assert row['coarse_base_count']==2 and row['fine_base_count']==4
        else:
            assert row['coarse_base_count']==row['fine_base_count']
            assert row['coarse_gauss_order']==2 and row['fine_gauss_order']==4
        assert row['signal_l1']>0 and row['error_l1']>0
        assert row['status']=='FAIL_DECLARED_5_PERCENT'
        assert row['qualification_status']=='UNRESOLVED_BOUNDARY_SCOPE'
        assert len(row['unresolved_scope'])==4
    assert all(g['physical_response']['axis_availability']['particle_count']=='NOT_RUN_NO_INDEPENDENT_AXIS_PAIR' for g in report['groups'])


def test_flat_gauss_difference_does_not_pass_a_signal_smaller_than_root_probability_uncertainty():
    report=analyse_conditional([sample_run(2,2,bound=1e-4,factor=1.),sample_run(2,4,bound=1e-4,factor=1.)],pixels=[32])
    assert all(row['status']=='UNRESOLVED_NUMERICAL_BOUND' for row in report['axes'])
    assert all(row['located_boundary_probability_bound_l1']==2e-4 for row in report['axes'])
    assert all(row['relative_error']<1e-10 for row in report['axes'])


def test_conditional_group_retains_lost_flux_and_real_centroid_without_renormalising():
    run=sample_run(2,2)
    case=run.clouds['A1_plus0.03mm']
    case.survivor=case.survivor.copy();case.survivor[0]=False
    original=case.weight.copy()
    report=analyse_conditional([run],pixels=[32])
    detail=report['groups'][0]['physical_response']['runs'][0]['cases']['A1_plus0.03mm']['survival_decomposition']
    assert detail['signed_full_plane_probability_change']==-.125
    assert detail['components']['lost_negative']['count']==1
    assert detail['components']['lost_negative']['source_ids']==[0]
    expected=case.weight[1:]@case.xy_mm[1:]/case.weight[1:].sum()
    np.testing.assert_allclose(detail['case_centroid_mm'],expected,atol=1e-19)
    np.testing.assert_array_equal(case.weight,original)
    assert all(value=='NOT_RUN_NO_INDEPENDENT_AXIS_PAIR' for value in report['axis_availability'].values())
    summary=compact_summary(report)
    row=next(r for r in summary['cases'] if r['case']=='A1_plus0.03mm')
    assert row['signed_full_plane_probability_change']==-.125
    assert row['lost_signed_probability']==-.125 and row['gained_signed_probability']==0.
    np.testing.assert_allclose(row['case_centroid_mm'],expected,atol=1e-19)
    assert row['raw_histogram_pixel_comparison']['status']=='NOT_RUN_NO_PIXEL_AXIS_PAIR'
    assert row['observed_anchor_cdf_estimate'] is None


def test_zero_weight_diagnostic_anchors_are_distinct_from_positive_source_nodes():
    run=sample_run(2,2)
    for name,cloud in list(run.clouds.items()):
        run.clouds[name]=Cloud(np.vstack((cloud.xy_mm,np.full((2,2),np.nan))),np.r_[cloud.weight,0.,0.],
            np.r_[cloud.survivor,False,False],np.arange(len(cloud.weight)+2),
            np.array(['flu_screen']*len(cloud.weight)+['flu_screen','aperture']),
            np.r_[np.full(len(cloud.weight),1000.),1000.,765.])
    run.clouds['A1_plus0.03mm'].survivor[0]=False
    report=analyse_conditional([run],pixels=[32])
    group=report['groups'][0]
    assert group['base_count']==2 and group['positive_source_nodes']==8
    assert group['zero_weight_diagnostic_nodes']==2 and group['final_ray_count']==10
    assert group['positive_weight_nodes']==8 and group['zero_weight_probe_nodes']==2
    counts=group['node_counts_by_case']['A1_plus0.03mm']
    assert counts['positive_weight_receiver_nodes']==7
    assert counts['positive_weight_lost_nodes']==1 and counts['positive_weight_gained_nodes']==0
    assert counts['zero_weight_receiver_probe_nodes']==1 and counts['zero_weight_other_probe_nodes']==1
    detail=group['physical_response']['runs'][0]['cases']['A1_plus0.03mm']['survival_decomposition']
    assert detail['components']['lost_negative']['count']==1
    assert detail['signed_full_plane_probability_change']==-.125


@pytest.mark.parametrize('field',['protocol_digest','base_seed','step_mm','partition_digest','base_points','weights'])
def test_conditional_comparison_rejects_changed_inputs_on_an_allegedly_single_axis(field):
    coarse,fine=sample_run(2,2),sample_run(2,4)
    if field=='base_points':fine.base_points=fine.base_points.copy();fine.base_points[0,0]+=.001
    elif field=='weights':
        c=fine.clouds['Cs_plus20mm'];c.weight=c.weight.copy();c.weight[0]+=.001
    else:setattr(fine,field,{'base_seed':1,'step_mm':.005}.get(field,'different'))
    with pytest.raises(ValueError):analyse_conditional([coarse,fine],pixels=[32])


def test_m_refinement_rejects_same_shape_but_nonprefix_source_bases():
    a,b=sample_run(2,2),sample_run(4,2)
    b.base_points=b.base_points.copy();b.base_points[0,0]+=.001
    with pytest.raises(ValueError,match='exact prefix'):
        analyse_conditional([a,b],pixels=[32])


def orbit_run(m,k,order):
    run=sample_run(m,order,factor=1+1/m+1/k+1/order**2)
    run.azimuth_orbit_size=k
    run.label=f'M{m}_K{k}_G{order}'
    run.partition_digest=f'partition{m}_{k}'
    expanded=np.repeat(run.base_points,k//2,axis=0).copy()
    if k!=2:
        expanded[:,[1,2]]=(expanded[:,[1,2]]+np.tile(np.arange(k//2),m)[:,None]/k)%1.
    run.expanded_base_points=expanded
    return run


def test_m_k_and_gauss_axes_hold_both_other_axes_fixed():
    report=analyse_conditional([orbit_run(m,k,g) for m in (2,4) for k in (2,8) for g in (2,4)],pixels=[32])
    assert set(report['axis_availability'].values())=={'EVALUATED'}
    assert len(report['axes'])==36
    fields={'base_count':'base_count','radial_gauss_order':'gauss_order','azimuth_orbit_size':'azimuth_orbit_size'}
    for row in report['axes']:
        for axis,field in fields.items():
            if row['axis']==axis:assert row['coarse_'+field]<row['fine_'+field]
            else:assert row['coarse_'+field]==row['fine_'+field]
        assert row['fine_expanded_base_count']==row['fine_base_count']*row['fine_azimuth_orbit_size']//2


def test_orbit_metadata_cannot_hide_one_azimuth_not_rotating():
    run=orbit_run(2,8,2)
    run.expanded_base_points[:,2]=np.repeat(run.base_points[:,2],4)
    with pytest.raises(ValueError,match='rotate both'):
        analyse_conditional([run],pixels=[32])


def test_cross_schema_comparison_keeps_original_protocols_and_requires_mapping_receipt():
    old,new=orbit_run(2,2,2),orbit_run(2,8,2)
    original=[]
    for run,schema in ((old,None),(new,'offline-common-source-azimuth-orbit-v1')):
        run.protocol_digest=f'archived protocol {schema}'
        original.append(run.protocol_digest)
        run.uncertainty.update(physical_protocol_comparison_digest='same exact other physical fields',
            azimuth_protocol_equivalence=dict(archived_physical_protocol_digest=run.protocol_digest,
                archived_azimuth_orbit_schema=schema,base_and_radial_table_mapping_verified=True))
    report=analyse_conditional([old,new],pixels=[32])
    assert len(report['axes'])==3 and all(r['axis']=='azimuth_orbit_size' for r in report['axes'])
    assert [old.protocol_digest,new.protocol_digest]==original
    old.uncertainty['azimuth_protocol_equivalence']['base_and_radial_table_mapping_verified']=False
    with pytest.raises(ValueError,match='verified legacy K2'):
        analyse_conditional([old,new],pixels=[32])


def test_cross_schema_comparison_rejects_changed_physics_even_with_mapping_receipt():
    old,new=orbit_run(2,2,2),orbit_run(2,8,2)
    old.uncertainty['physical_protocol_comparison_digest']='original lens parameters'
    new.uncertainty['physical_protocol_comparison_digest']='different lens parameters'
    with pytest.raises(ValueError,match='one physical protocol'):
        analyse_conditional([old,new],pixels=[32])


def test_comparison_rejects_different_propagation_code_with_identical_physical_controls():
    old,new=sample_run(2,2),sample_run(2,4)
    old.uncertainty['transport_implementation_sha256']='original physical solver'
    new.uncertainty['transport_implementation_sha256']='changed physical solver'
    with pytest.raises(ValueError,match='same archived propagation implementation'):
        analyse_conditional([old,new],pixels=[32])


def anchor_arrays(run):
    bases=run.expanded_base_points;n=len(bases)
    cdf=np.column_stack((np.repeat(bases[:,:2],2,axis=0),np.tile([0.,1.],n),np.repeat(bases[:,2:],2,axis=0)))
    return np.repeat(cdf,2,axis=0),np.tile([1,-1],2*n)


def test_final_anchor_comparison_joins_physical_cdf_and_separates_common_mode_drift():
    a,b=orbit_run(2,2,2),orbit_run(4,8,2)
    for run in (a,b):
        run.anchor_coordinates,run.anchor_signs=anchor_arrays(run)
        x=run.anchor_coordinates
        baseline=x[:,0]+2*x[:,1]+x[:,2]+run.anchor_signs*.01
        run.anchor_margins_mm=np.column_stack((baseline,baseline+x[:,3]*.001,baseline+x[:,4]*.002,baseline+x[:,5]*.003))
    b.anchor_margins_mm+=.001
    b.anchor_margins_mm[:,2]+=2e-6
    reverse=np.arange(len(b.anchor_signs)-1,-1,-1)
    b.anchor_coordinates=b.anchor_coordinates[reverse];b.anchor_signs=b.anchor_signs[reverse];b.anchor_margins_mm=b.anchor_margins_mm[reverse]
    row=compare_final_anchors(a,b,'diagnostic_cross_M_and_K')
    assert row['matched_physical_probe_count']==8
    assert row['status']=='OBSERVED_NOT_CERTIFIED'
    assert row['cases']['baseline']['absolute_margin_change']['maximum_absolute_mm']==pytest.approx(.001,abs=1e-15)
    assert row['cases']['Cs_plus20mm']['case_minus_baseline_margin_change']['maximum_absolute_mm']<1e-15
    assert row['cases']['A1_plus0.03mm']['case_minus_baseline_margin_change']['maximum_absolute_mm']==pytest.approx(2e-6,abs=1e-15)
    assert 'relative_error' not in row


def test_final_anchor_loading_binds_hash_rows_cases_and_signs(tmp_path):
    run=orbit_run(2,2,2);coordinates,signs=anchor_arrays(run)
    table=dict(coordinates=coordinates,signs=signs,weights=np.zeros(len(signs)),cell_index=np.full(len(signs),-1))
    margins=np.arange(32,dtype=float).reshape(4,8)
    path=tmp_path/'anchors.npz';np.savez(path,anchor_margin_mm=margins)
    receipt=dict(anchor_path='anchors.npz',anchor_sha256=sha256(path.read_bytes()).hexdigest())
    x,s,m,r=load_final_anchors(tmp_path,receipt,table,run.expanded_base_points)
    np.testing.assert_array_equal(m[0],[0,2,4,6]);np.testing.assert_array_equal(m[1],[1,3,5,7])
    assert r['probe_count']==8 and r['status']=='SHA_VERIFIED_FINAL_C2_ENDPOINTS'
    table['signs']=table['signs'][::-1]
    with pytest.raises(ValueError,match='archived zero-weight endpoint'):
        load_final_anchors(tmp_path,receipt,table,run.expanded_base_points)
    path.write_bytes(b'tampered margin archive')
    with pytest.raises(ValueError,match='checksum mismatch'):
        load_final_anchors(tmp_path,receipt,table,run.expanded_base_points)


def test_missing_retained_anchors_are_unavailable_not_zero_drift():
    row=compare_final_anchors(sample_run(2,2),sample_run(4,2),'base_count')
    assert row['status']=='NOT_AVAILABLE_RETAINED_FINAL_ANCHORS_MISSING'
    assert 'cases' not in row


def test_relative_error_and_root_uncertainty_must_fit_one_combined_five_percent_budget():
    coarse=sample_run(2,2,bound=1e-9,factor=1.049)
    fine=sample_run(2,4,bound=1e-9,factor=1.)
    for run in (coarse,fine):run.uncertainty=dict(topology_completeness='CERTIFIED',batch_drift_cdf_bound=0.)
    report=analyse_conditional([coarse,fine],pixels=[32])
    for row in report['axes']:
        assert row['relative_error']<.05
        assert row['status']=='PASS_DECLARED_5_PERCENT'  # Preserved legacy metric, not final gate.
        assert row['conservative_relative_error_upper']>.05
        assert row['robust_status']=='UNRESOLVED_COMBINED_ERROR_BUDGET'
        assert row['qualification_status']=='UNRESOLVED_COMBINED_ERROR_BUDGET'


def root_record(channel,width,base=0):
    return dict(base_index=base,channel_key=channel,u=.5,bracket=[.5-width/2,.5+width/2],
                bracket_width=width,actual_margin_residual=0.)


def test_root_probability_bound_uses_actual_case_and_sign_weights_not_all_cases():
    roots=[root_record('baseline:+1',1e-8),root_record('baseline:-1',1e-8),
           root_record('Cs_plus20mm:+1',1e-7),root_record('A1_plus0.03mm:-1',8e-7)]
    diagnostics=dict(all_located_roots_converged=True,root_count=4,angular_cdf_tolerance=1e-6)
    bounds,summary=located_root_bounds(dict(roots=roots),diagnostics,4)
    assert bounds['Cs_plus20mm']==pytest.approx(1.2e-7/8,abs=1e-16)
    assert bounds['A1_plus0.03mm']==pytest.approx(8.2e-7/8,abs=1e-16)
    assert bounds['B2_plus1mm']==pytest.approx(2e-8/8,abs=1e-16)
    assert summary['all_channel_probability']>bounds['Cs_plus20mm']
    roots[0]['bracket_width']=1e-3
    with pytest.raises(ValueError,match='root bracket'):
        located_root_bounds(dict(roots=roots),diagnostics,4)


def write_archive(path):
    path.mkdir(exist_ok=True)
    def save_json(name,values):
        p=path/name;p.write_text(json.dumps(values));return sha256(p.read_bytes()).hexdigest()
    def save_arrays(name,values):
        p=path/name;np.savez(p,**values);return sha256(p.read_bytes()).hexdigest()
    controls=dict.fromkeys(PROJECTOR_KEYS,20.)
    physical=dict(step_mm=.00625,source_base_seed=0,projector_controls=controls,nominal_current_a=1e-5)
    protocol=dict(physical_protocol=physical,implementation='actual solver')
    bases={'points':np.arange(10).reshape(2,5)/100}
    part=dict(roots=[root_record('baseline:+1',1e-8)],partitions=[])
    part_digest=canonical_digest(part)
    table=dict(coordinates=np.full((16,6),.5),weights=np.full(16,1/16),signs=np.tile([1,-1],8),ray_ids=np.arange(16))
    record={k:v.tolist() for k,v in table.items()};record.update(schema='tip-explicit-cdf-v1',role='quadrature',construction='analytic archive fixture')
    table_digest=canonical_digest(record)
    source=dict(gun_exit_weight=table['weights'],gun_exit_ray_id=table['ray_ids'],gun_exit_alive=np.ones(16,dtype=bool),
                gun_blocked_key=np.full(16,''),gun_blocked_z_mm=np.full(16,np.nan))
    source_digest=array_digest(source)
    receipt=dict(base_count=2,gauss_order=2,partition_digest=part_digest,source_table_digest=table_digest,source_digest=source_digest,
        table_record_path='table.json',table_record_sha256=save_json('table.json',record),
        table_path='table.npz',table_sha256=save_arrays('table.npz',table),
        source_path='source.npz',source_sha256=save_arrays('source.npz',source),emitted_current_a=1e-5)
    d=dict(schema='conditional-source-convergence-v1',status='COMPLETE',implementation='actual solver',
        base_count=2,base_seed=0,step_mm=.00625,receiver_z_mm=2700.,projector_controls=controls,
        protocol=protocol,protocol_digest=canonical_digest(protocol),physical_protocol=physical,
        physical_protocol_digest=canonical_digest(physical),base_points_path='base.npz',
        base_points_sha256=save_arrays('base.npz',bases),base_digest=array_digest(bases),
        partition_path='partition.json',partition_sha256=save_json('partition.json',part),partition_digest=part_digest,
        root_diagnostics=dict(all_located_roots_converged=True,root_count=1,angular_cdf_tolerance=1e-6,
                              completeness='SAMPLED_BRACKETS_ONLY_NOT_CERTIFIED'),
        point_spread=dict(model='gaussian',sigma_x_mm=.6,sigma_y_mm=.6,rotation_deg=0.),sources={'M2_G2':receipt},runs={})
    for case in CASES:
        label=f'M2_G2_{case}'
        arrays=dict(xy_m=np.tile([[-1e-7,2e-8],[1e-7,-2e-8]],(8,1)),weight=table['weights'],source_id=table['ray_ids'],
                    survivor=np.ones(16,dtype=bool),blocked_key=np.full(16,'flu_screen'),blocked_z_mm=np.full(16,2700.),final_alive=np.zeros(16,dtype=bool))
        snapshot={'digest':f'physical {case}'}
        row=dict(base_count=2,gauss_order=2,case=case,status='COMPLETE',
            particles_path=f'{case}.npz',particles_sha256=save_arrays(f'{case}.npz',arrays),
            snapshot_path=f'{case}.json',snapshot_sha256=save_json(f'{case}.json',snapshot),
            source_table_digest=table_digest,source_digest=source_digest,partition_digest=part_digest,
            receiver_z_mm=2700.,projector_controls=controls.copy(),implementation_before='actual solver',implementation_after='actual solver')
        row['request_identity']=canonical_digest(dict(protocol=d['protocol_digest'],state=snapshot['digest'],
            target_z_mm=2700.,run_id=label,source_table=table_digest,partition=part_digest))
        d['runs'][label]=row
    save_json('manifest.json',d)
    return d


def test_loader_verifies_complete_shared_table_and_marks_partial_groups(tmp_path):
    d=write_archive(tmp_path)
    runs,info=load_conditional_runs([tmp_path])
    assert len(runs)==1 and info['pending']==[]
    assert runs[0].base_count==2 and runs[0].gauss_order==2
    assert runs[0].clouds['baseline'].weight.sum()==1
    assert runs[0].uncertainty['source_current_a']==1e-5
    assert runs[0].uncertainty['transport_implementation_sha256']=='actual solver'
    assert runs[0].root_probability_bounds['Cs_plus20mm']==pytest.approx(1e-8/4,abs=1e-16)
    d['runs']['M2_G2_A1_plus0.03mm']['status']='RUNNING'
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    runs,info=load_conditional_runs([tmp_path])
    assert runs==[] and any(p['status']=='INCOMPLETE_SCREEN_CASE_GROUP' for p in info['pending'])


def test_loader_checks_actual_conditional_base_mapping_before_legacy_equivalence(tmp_path):
    d=write_archive(tmp_path);receipt=d['sources']['M2_G2']
    with np.load(tmp_path/'table.npz') as f:table={key:f[key] for key in f.files}
    table['base_index']=np.zeros(16,dtype=np.int64);table['angular_cdf']=table['coordinates'][:,2]
    np.savez(tmp_path/'table.npz',**table)
    receipt['table_sha256']=sha256((tmp_path/'table.npz').read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    # The retained .5 coordinates do not match original 5D source base zero.
    with pytest.raises(ValueError,match='actual base/radial mapping'):
        load_conditional_runs([tmp_path])


def test_loader_orbit_archive_uses_expanded_denominator_and_independent_m(tmp_path):
    d=write_archive(tmp_path);k=8;m=2
    with np.load(tmp_path/'base.npz') as f:bases=f['points']
    expanded=np.repeat(bases,k//2,axis=0)
    expanded[:,[1,2]]=(expanded[:,[1,2]]+np.tile(np.arange(k//2),m)[:,None]/k)%1.
    np.savez(tmp_path/'expanded.npz',points=expanded)
    d.update(azimuth_orbit_size=k,expanded_base_count=m*k//2,expanded_base_points_path='expanded.npz',
        expanded_base_points_sha256=sha256((tmp_path/'expanded.npz').read_bytes()).hexdigest(),
        expanded_base_digest=array_digest({'points':expanded}))
    source=d['sources'].pop('M2_G2');source.update(azimuth_orbit_size=k,expanded_base_count=8)
    d['sources']['M2_K8_G2']=source
    rows={}
    for label,row in d['runs'].items():
        label=label.replace('M2_G2','M2_K8_G2');row.update(azimuth_orbit_size=k,expanded_base_count=8)
        snapshot=json.loads((tmp_path/row['snapshot_path']).read_text())
        row['request_identity']=canonical_digest(dict(protocol=d['protocol_digest'],state=snapshot['digest'],
            target_z_mm=2700.,run_id=label,source_table=source['source_table_digest'],partition=d['partition_digest']))
        rows[label]=row
    d['runs']=rows
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    runs,_=load_conditional_runs([tmp_path]);run=runs[0]
    assert run.base_count==2 and run.expanded_base_count==8 and run.azimuth_orbit_size==8
    assert run.root_probability_bounds['Cs_plus20mm']==pytest.approx(1e-8/16,abs=1e-16)
    d['expanded_base_count']=2
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Expanded radial base count'):
        load_conditional_runs([tmp_path])


@pytest.mark.parametrize('change',['screen_hash','source_digest','snapshot_request','table_record','gun_weights','projector'])
def test_loader_rejects_tampered_or_cross_bound_archives(tmp_path,change):
    d=write_archive(tmp_path);row=d['runs']['M2_G2_Cs_plus20mm'];source=d['sources']['M2_G2']
    if change=='screen_hash':(tmp_path/row['particles_path']).write_bytes(b'changed particles')
    elif change=='source_digest':row['source_digest']='another gun'
    elif change=='snapshot_request':row['request_identity']='another request'
    elif change=='table_record':
        record=json.loads((tmp_path/'table.json').read_text());record['construction']='different source table'
        (tmp_path/'table.json').write_text(json.dumps(record));source['table_record_sha256']=sha256((tmp_path/'table.json').read_bytes()).hexdigest()
    elif change=='gun_weights':
        with np.load(tmp_path/'source.npz') as f:arrays={k:f[k] for k in f.files}
        arrays['gun_exit_weight']=arrays['gun_exit_weight'].copy();arrays['gun_exit_weight'][0]+=.01
        np.savez(tmp_path/'source.npz',**arrays);source['source_sha256']=sha256((tmp_path/'source.npz').read_bytes()).hexdigest();source['source_digest']=array_digest(arrays)
    else:row['projector_controls']['diffraction_lens']+=.1
    (tmp_path/'manifest.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):load_conditional_runs([tmp_path])
