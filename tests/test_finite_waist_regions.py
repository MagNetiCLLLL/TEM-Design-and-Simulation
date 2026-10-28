"""Analytical finite-source acceptance regressions; no field propagation."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from temsim.physics.finite_waist_regions import measure_finite_waist_regions


def data(z,*,split=.4,rotation=0.,constant_y=False,two=False,oscillating=False,offset_maxima=False,plateau=False):
    z=np.unique(np.r_[0.,z,10.]);phi=np.arange(32)*2*np.pi/32
    if plateau:
        q=np.where(z<4,z-4,np.where(z>6,z-6,0.))
        a=b=q*1e-6;da=db=np.where((z<4)|(z>6),1e-6,0.)
    elif offset_maxima:
        va=(300+z**3/3-5*z*z+16*z)*1e-12
        vb=(300+z**3/3-5.5*z*z+24*z)*1e-12
        a,b=np.sqrt(2*va),np.sqrt(2*vb)
        da=(z-2)*(z-8)*1e-12/a;db=(z-3)*(z-8)*1e-12/b
    elif oscillating:
        q=z-5;va=(100+q*q+2*np.sin(4*q))*1e-12;vb=(100+q*q-2*np.sin(4*q))*1e-12
        a,b=np.sqrt(2*va),np.sqrt(2*vb)
        da=(2*q+8*np.cos(4*q))*1e-12/a;db=(2*q-8*np.cos(4*q))*1e-12/b
    elif two:
        a=b=(z-3)*(z-7)*1e-6;da=db=(2*z-10)*1e-6
    else:
        a=(z-(5-split/2))*1e-6;b=(z-(5+split/2))*1e-6
        da=db=np.full(len(z),1e-6)
    x=a[:,None]*np.cos(phi)+1e-8*np.sin(2*phi)
    y=b[:,None]*np.sin(phi)+1e-8*np.cos(2*phi)
    tx=da[:,None]*np.cos(phi)*1e3;ty=db[:,None]*np.sin(phi)*1e3
    if constant_y:y[:]=1e-8*np.sin(phi);ty[:]=0
    Q=np.array([[np.cos(rotation),-np.sin(rotation)],[np.sin(rotation),np.cos(rotation)]])
    xy=np.stack([x,y],axis=-1)@Q.T;tt=np.stack([tx,ty],axis=-1)@Q.T
    cp=SimpleNamespace(z_mm=z,x_m=xy[:,:,0]+3e-6,y_m=xy[:,:,1]-4e-6,tx_rad=tt[:,:,0],ty_rad=tt[:,:,1])
    return cp,np.ones(x.shape,bool),1+.2*np.cos(3*phi),np.arange(32)+100


def observe(value,*,spacing=.51):
    cp,masks,weights,ids=value
    return measure_finite_waist_regions(cp,masks,weights,0.,10.,maximum_spacing_mm=spacing,source_ids=ids)


@pytest.mark.parametrize('step,phase',[(.5,0.),(.2,.07),(.05,.017),(.01,.003)])
def test_one_finite_astigmatic_region_is_stable_under_refinement_and_grid_phase(step,phase):
    report=observe(data(np.arange(phase,10,step)))
    assert report['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
    region,=report['regions']
    assert region['radial_root_interpolated_mm']==pytest.approx(5.,abs=1e-10)
    assert region['bracket_mm'][0]<4.8<5.2<region['bracket_mm'][1]
    assert region['finite_region_span_mm']<2.
    assert region['exact_common_stationary_plane_claimed'] is False
    assert all(r['converging_to_diverging']==1 and r['diverging_to_converging']==0 for r in region['directional_sign_transitions'])


@pytest.mark.parametrize('angle',[.173,.7,1.1,2.4])
def test_region_definition_is_rotation_invariant(angle):
    z=np.arange(0,10.01,.05)
    a=observe(data(z));b=observe(data(z,rotation=angle))
    assert a['status']==b['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
    assert a['regions'][0]['bracket_mm']==b['regions'][0]['bracket_mm']
    assert a['regions'][0]['rms_radius_nm']==pytest.approx(b['regions'][0]['rms_radius_nm'],rel=1e-12)


def test_explicit_region_sides_are_not_replaced_by_adjacent_checkpoints():
    value=data(np.arange(0,10.01,.05))
    region,=observe(value)['regions']
    left,center,right=region['executed_indices']
    assert left<center-1<center+1<right
    assert value[0].z_mm[[left,right]].tolist()==region['bracket_mm']
    assert value[0].z_mm[center]==region['representative_z_mm']
    assert np.all(np.array(region['bracket_covariance_growth_m2'])>0)
    covariance=np.array(region['bracket_covariance_eigenvalues_m_rad'])
    assert np.all(covariance[0]<0) and np.all(covariance[2]>0)


def test_historical_point_counter_is_unchanged_and_finite_region_is_explicit_opt_in():
    from temsim.physics.corrector_crossovers import measured_corrector_crossovers
    cp,masks,weights,ids=data(np.arange(0,10.01,.05))
    assert measured_corrector_crossovers(cp,masks,weights,0.,10.)==[]
    assert len(observe((cp,masks,weights,ids))['regions'])==1


@pytest.mark.parametrize('angle',[0.,.4,1.2])
def test_constant_width_direction_and_single_line_focus_never_qualify(angle):
    report=observe(data(np.arange(0,10.01,.05),constant_y=True,rotation=angle))
    assert report['status']=='UNRESOLVED' and not report['regions']


def test_two_independent_waists_are_not_merged_into_one_region():
    report=observe(data(np.arange(0,10.01,.05),two=True))
    assert len(report['radial_candidates'])==2
    assert report['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
    assert len(report['regions'])==2
    assert report['regions'][0]['bracket_mm'][1]<5.<report['regions'][1]['bracket_mm'][0]


def test_repeated_directional_refocusing_is_not_hidden_by_one_trace_minimum():
    report=observe(data(np.arange(0,10.001,.01),oscillating=True))
    assert len(report['radial_candidates'])==1
    assert not report['regions'] and report['status']=='UNRESOLVED'
    assert report['unresolved'][0]['reason']=='missing_or_repeated_directional_transition'
    assert any(r['diverging_to_converging']>0 for r in report['unresolved'][0]['directional_sign_transitions'])


def test_separated_directional_maxima_before_one_focus_do_not_create_extra_waists():
    report=observe(data(np.arange(0,10.001,.01),offset_maxima=True))
    assert report['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
    region,=report['regions']
    assert region['representative_z_mm']==pytest.approx(8.,abs=1e-10)
    # Linear interpolation across a curved moment is a diagnostic only.
    assert region['radial_root_interpolated_mm']==pytest.approx(8.,abs=1e-4)
    assert any(row['diverging_to_converging'] for row in region['basin_directional_sign_transitions'])
    assert all(row['converging_to_diverging']==1 and row['diverging_to_converging']==0
               for row in region['directional_sign_transitions'])


def test_extended_stationary_plateau_is_not_reported_as_one_local_waist():
    report=observe(data(np.arange(0,10.001,.05),plateau=True))
    assert report['status']=='UNRESOLVED' and not report['regions']
    assert report['unresolved'][0]['reason']=='extended_radial_stationary_plateau'


@pytest.mark.parametrize('damage',['gap','missing_boundary','interception','resurrection'])
def test_sparse_missing_or_population_changing_evidence_cannot_pass(damage):
    cp,masks,weights,ids=data(np.arange(0,10.01,.1))
    if damage=='interception':masks[cp.z_mm>4,0]=False
    elif damage=='resurrection':masks[cp.z_mm<4,0]=False
    else:
        keep=cp.z_mm!=0 if damage=='missing_boundary' else ((cp.z_mm<=4)|(cp.z_mm>=6))
        cp=SimpleNamespace(**{k:v[keep] for k,v in vars(cp).items()});masks=masks[keep]
    report=observe((cp,masks,weights,ids))
    assert report['status']=='UNRESOLVED' and not report['regions']


def test_no_physical_recentering_reweighting_or_array_mutation():
    value=data(np.arange(0,10.01,.05));before=deepcopy(value)
    report=observe(value)
    assert report['source_ids']==value[3].tolist()
    assert report['original_weights']==value[2].tolist()
    for key,array in vars(value[0]).items():np.testing.assert_array_equal(array,getattr(before[0],key))
    for i in (1,2,3):np.testing.assert_array_equal(value[i],before[i])


def test_region_cannot_borrow_a_wider_side_outside_component_bounds():
    cp,masks,weights,ids=data(np.arange(0,10.01,.05),split=2.)
    report=measure_finite_waist_regions(cp,masks,weights,4.,6.,maximum_spacing_mm=.1,source_ids=ids)
    assert report['status']=='UNRESOLVED' and not report['regions']


def test_weight_scale_and_particle_permutation_leave_regions_unchanged():
    cp,masks,weights,ids=data(np.arange(0,10.01,.05))
    original=observe((cp,masks,weights,ids))
    order=np.arange(len(ids))[::-1]
    reordered=SimpleNamespace(z_mm=cp.z_mm,**{key:getattr(cp,key)[:,order]
        for key in ('x_m','y_m','tx_rad','ty_rad')})
    result=observe((reordered,masks[:,order],weights[order]*7.5,ids[order]))
    assert result['regions'][0]['bracket_mm']==original['regions'][0]['bracket_mm']
    assert result['regions'][0]['rms_radius_nm']==pytest.approx(original['regions'][0]['rms_radius_nm'],rel=1e-12)
    assert result['source_ids']==ids[order].tolist()
    assert result['original_weights']==(weights[order]*7.5).tolist()


def test_preinterval_interception_preserves_original_source_denominator():
    cp,masks,weights,ids=data(np.arange(0,10.01,.05))
    masks[:,0]=False
    report=observe((cp,masks,weights,ids))
    assert report['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
    assert report['source_ids']==ids[1:].tolist()
    assert report['original_weights']==weights[1:].tolist()
    assert report['original_source_current_fraction']==pytest.approx(weights[1:].sum()/weights.sum())


@pytest.mark.parametrize('damage',['nan','negative_weight','duplicate_id','shape','spacing'])
def test_invalid_evidence_is_unresolved(damage):
    cp,masks,weights,ids=data(np.arange(0,10.01,.05))
    if damage=='nan':cp.x_m[5,0]=np.nan
    elif damage=='negative_weight':weights[0]=-1
    elif damage=='duplicate_id':ids[0]=ids[1]
    elif damage=='shape':cp.y_m=cp.y_m[:,:-1]
    report=observe((cp,masks,weights,ids),spacing=0. if damage=='spacing' else .51)
    assert report['status']=='UNRESOLVED' and not report['regions']
