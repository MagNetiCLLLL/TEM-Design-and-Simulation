"""Analytical consumer-path checks; no source or field propagation."""
from copy import deepcopy
from importlib.util import module_from_spec,spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from temsim.optics import condenser_topology
from temsim.optics.crossover_acceptance import evaluate_crossover_chain,regions_for_state
from temsim.optics.waist_evidence import observed_aperture_losses

spec=spec_from_file_location('finite_consumer_qualification',Path(__file__).resolve().parents[1]/'scripts/qualify_finite_field_nanoprobe.py')
qualification=module_from_spec(spec);spec.loader.exec_module(qualification)


def column(monkeypatch,step=.1,*,defocus_nm=0.,clipping=None,source_harmonic=5):
    lens=lambda key,z:SimpleNamespace(key=key,z_mm=z,enabled=True,installed=True,
        mechanical_center_from_tip_mm=z,mechanical_length_mm=2.)
    state=SimpleNamespace(simulation_mode='unified',probe_mode='nano_probe',nanoprobe_range='normal',
        ac_deflector=SimpleNamespace(lower_z_mm=42.,mechanical_coil_length_mm=1.),
        lenses=[lens(k,z) for k,z in [('condenser_lens_1',5.),('condenser_lens_2',10.),
            ('condenser_lens_3',20.),('adapter_lens',25.),('probe_tl22_lens',28.),
            ('probe_tl21_lens',34.),('probe_tl12_lens',36.5),('mini_condenser',45.)]],
        mini_condenser=SimpleNamespace(enabled=True,installed=True,mechanical_center_from_tip_mm=45.,mechanical_length_mm=2.),
        electron_gun=SimpleNamespace(exit_plane_z_mm=0.),sample=SimpleNamespace(upper_surface_z_mm=50.),apertures=[])
    monkeypatch.setattr('temsim.physics.instrument_magnetic.column_dipole_fields',lambda state:[
        SimpleNamespace(key=f'ac_deflector:{i}',arrival_z_mm=z,dynamic=False) for i,z in enumerate((38.,42.))])
    monkeypatch.setattr('temsim.physics.aperture_clipping.posed_aperture_registration',lambda state,aperture:None)
    z=np.unique(np.r_[np.arange(0.,50.,step),condenser_topology.requested_capture_planes(state,maximum_spacing_mm=.5),
        5.,10.,20.,25.,28.,34.,35.5,38.,42.,42.25,42.5,45.,50.])
    phi=np.arange(32)*2*np.pi/32
    root=np.select([z<25,z<36,z<44],[15.,31.,43.],default=50.+defocus_nm*1e-6)
    split=np.select([z<25,z<36,z<44],[.4,.4,.04],default=0.)
    tx=np.broadcast_to(.002*np.cos(phi),(len(z),32)).copy()
    ty=np.broadcast_to(.002*np.sin(phi),(len(z),32)).copy()
    # The default finite source spread is orthogonal to first-harmonic directions
    # with the third-harmonic current weights below. A second harmonic would
    # split the terminal X/Y waists by 1 nm despite zero radial covariance.
    cp=SimpleNamespace(z_mm=z,x_m=(z-root+split/2)[:,None]*1e-3*tx+1e-11*np.cos(source_harmonic*phi),
        y_m=(z-root-split/2)[:,None]*1e-3*ty+1e-11*np.sin(source_harmonic*phi),tx_rad=tx,ty_rad=ty)
    masks=np.ones(cp.x_m.shape,bool);weights=1+.2*np.cos(3*phi);ids=np.arange(32)+100
    if clipping is not None:
        plane=float(clipping)
        masks[z>=plane,0]=False
        cp.x_m[z<=plane,0]=1e-4
        state.apertures=[SimpleNamespace(key='test_aperture',z_mm=plane,radius_mm=.03,
            offset_x_mm=0.,offset_y_mm=0.,enabled=True,installed=True,inserted=True)]
    return state,cp,masks,weights,ids


def topology(data):
    state,cp,masks,w,ids=data
    return condenser_topology.evaluate(state,cp,masks,w,maximum_spacing_mm=.5,source_ids=ids)


def corrector(data):
    state,cp,masks,w,ids=data
    return qualification.evaluate_corrector_crossovers(cp,masks,w,qualification.corrector_crossover_regions(state),
        source_current_a=1e-5,source_ids=ids,waist_detector='finite_region')


def chain(data):
    state,cp,masks,w,ids=data
    return evaluate_crossover_chain(cp,masks,w,ids,source_current_a=1e-5,regions=regions_for_state(state),
        maximum_spacing_mm=.5,waist_detector='finite_region',
        known_interceptions=observed_aperture_losses(state,cp,masks,w,ids))


@pytest.mark.parametrize('step',[.5,.05,.013])
def test_unified_nanoprobe_new_qualification_uses_finite_regions_at_all_sampling_densities(monkeypatch,step):
    data=column(monkeypatch,step)
    report=topology(data)
    assert report['passed'],report
    assert report['waist_detector']=='finite_region'
    assert report['corrector']['passed'] and report['direct_focus']['passed']
    assert 'legacy_point_diagnostic' in report
    corr=corrector(data)
    assert corr['waist_evidence_passed'],corr
    for interval in corr['intervals'].values():
        waist,=interval['executed_waists']
        assert waist['actual_waist_support']['same_explicit_covariance_growth_bracket']
        assert waist['actual_waist_support']['explicit_bracket_checkpoint_count']>=3
    inventory=chain(data)
    assert inventory['required_topology_satisfied_at_sampling'],inventory['intervals']
    assert len([r for r in inventory['executed_waists'] if r['region']=='corrector_internal'])==1


def test_radial_endpoint_cancellation_remains_unresolved_in_both_consumers(monkeypatch):
    # Retain the original fixture as insufficient evidence, rather than calling
    # its separated directional waists a proven extra two-dimensional crossover.
    data=column(monkeypatch,source_harmonic=2)
    topo,inventory=topology(data),chain(data)
    assert not topo['passed'] and not inventory['required_topology_satisfied_at_sampling']
    for report in (topo['direct_focus'],inventory['direct_focus']):
        assert report['status']=='UNRESOLVED' and not report['passed']
        assert any(row.get('reason')=='radial_stationarity_on_component_boundary'
            for row in report['unresolved_candidates'])


@pytest.mark.parametrize('plane',[2.,12.])
def test_known_aperture_loss_does_not_impose_a_full_transmission_requirement(monkeypatch,plane):
    data=column(monkeypatch,.05,clipping=plane)
    report=topology(data)
    assert report['passed'],report
    loss,=report['population_change_evidence']
    assert loss['status']=='CAPTURED_LOSS_SUPPORTED_BY_OPENING_GEOMETRY'
    assert loss['lost_source_ids']==[100]
    inventory=chain(data)
    assert inventory['required_topology_satisfied_at_sampling'],inventory['intervals']
    assert inventory['status']=='UNRESOLVED'  # The full inventory is not relabeled.
    waist=next(row for row in inventory['executed_waists'] if row['region']=='c2_to_c3')
    assert waist['source_current_fraction']<1.
    assert waist['actual_source_ids']==data[4][1:].tolist()


def test_unattributed_loss_inside_a_required_interval_cannot_qualify(monkeypatch):
    data=column(monkeypatch,.05,clipping=12.)
    data[0].apertures=[]
    assert not topology(data)['passed']
    assert not chain(data)['required_topology_satisfied_at_sampling']


def test_unattributed_upstream_loss_is_reported_without_relabeling_core_evidence(monkeypatch):
    data=column(monkeypatch,.05,clipping=2.)
    data[0].apertures=[]
    inventory=chain(data)
    assert inventory['required_topology_satisfied_at_sampling']
    assert inventory['unattributed_population_change_count']==1
    measurement=SimpleNamespace(accepts=lambda target:True,statistics=SimpleNamespace(radius_95_m=.2e-9))
    failures=qualification.qualification_failures(topology(data),measurement,corrector(data),30.,crossover_chain=inventory)
    assert failures==['Unattributed physical interception remains unresolved in the incident capture']


def test_finite_region_cannot_hide_sample_defocus_or_missing_diameter_angle_gates(monkeypatch):
    data=column(monkeypatch,.05,defocus_nm=-3.572)
    before=deepcopy(data[1:])
    topo=topology(data);corr=corrector(data);inventory=chain(data)
    assert not topo['passed'] and not inventory['required_topology_satisfied_at_sampling']
    assert 'missing_or_ambiguous_executed_ac_2d_waist' in corr['failures']
    first, extra = corr['intervals']['ac']['executed_waists']
    assert first['z_mm'] == pytest.approx(43.)
    assert 49.9999 < extra['z_mm'] < 50.
    measurement=SimpleNamespace(accepts=lambda target:False,statistics=SimpleNamespace(radius_95_m=2.5e-9))
    failures=qualification.qualification_failures(topo,measurement,corr,30.,crossover_chain=inventory)
    assert any('Surface focus' in row for row in failures)
    assert any('D95' in row for row in failures)
    assert any('required' in row.lower() for row in failures)
    for key,array in vars(data[1]).items():np.testing.assert_array_equal(array,getattr(before[0],key))
    for i in range(2,5):np.testing.assert_array_equal(data[i],before[i-1])


def test_new_qualification_requires_explicit_chain_evidence(monkeypatch):
    data=column(monkeypatch)
    measurement=SimpleNamespace(accepts=lambda target:True,statistics=SimpleNamespace(radius_95_m=.2e-9))
    failures=qualification.qualification_failures(topology(data),measurement,corrector(data),30.)
    assert failures==['Required source-image regions failed or have incomplete whole-capture evidence',
        'Direct Mini Condenser/objective focus topology failed or is unresolved']


def test_missing_required_interval_planes_fail_closed(monkeypatch):
    data=column(monkeypatch)
    state,cp,masks,w,ids=data
    keep=(cp.z_mm<42.5)|(cp.z_mm>44.)
    cp=SimpleNamespace(**{key:value[keep] for key,value in vars(cp).items()})
    report=topology((state,cp,masks[keep],w,ids))
    assert not report['passed'] and report['status']=='UNRESOLVED'


def test_static_waist_inside_scan_pair_cannot_supply_downstream_pivot_evidence(monkeypatch):
    data=column(monkeypatch,.05)
    state,cp,masks,w,ids=data
    section=(cp.z_mm>=36.) & (cp.z_mm<44.)
    for pos,slope in (('x_m','tx_rad'),('y_m','ty_rad')):
        getattr(cp,pos)[section] += .003 * getattr(cp,slope)[section]
    report=chain(data)
    assert report['intervals']['ac_pair']['status']=='FAIL_FORBIDDEN_WAIST'
    assert not report['downstream_output_waist']['regions']
    assert not report['required_topology_satisfied_at_sampling']
    assert corrector(data)['driven_pivot_match']=='UNMEASURED_IN_STATIC_CAPTURE'


def test_custom_mode_retains_the_historical_counter_by_default(monkeypatch):
    state,cp,masks,w,ids=column(monkeypatch,.013)
    state.simulation_mode='custom'
    report=condenser_topology.evaluate(state,cp,masks,w,maximum_spacing_mm=.5)
    assert report['schema']=='condenser-topology-v1'
    assert 'legacy_point_diagnostic' not in report


@pytest.mark.parametrize('damage',['missing_weights','bad_ids','resurrected_ray'])
def test_unified_consumer_invalid_inputs_fail_closed(monkeypatch,damage):
    state,cp,masks,w,ids=column(monkeypatch)
    if damage=='missing_weights':w=None
    elif damage=='bad_ids':ids=ids[:-1]
    else:masks[:5,0]=False
    report=condenser_topology.evaluate(state,cp,masks,w,maximum_spacing_mm=.5,source_ids=ids)
    assert report['status']=='UNRESOLVED' and not report['passed']


def test_extra_pre_specimen_finite_region_rejects_even_when_followed_by_clipping(monkeypatch):
    from temsim.optics.direct_focus_topology import evaluate
    state,cp,masks,w,ids=column(monkeypatch)
    selected=cp.z_mm>=44.
    phi=np.arange(32)*2*np.pi/32
    cp.x_m[selected]=(cp.z_mm[selected]-47.)[:,None]*1e-3*.002*np.cos(phi)
    cp.y_m[selected]=(cp.z_mm[selected]-47.)[:,None]*1e-3*.002*np.sin(phi)
    masks[cp.z_mm>48.,0]=False
    report=evaluate(cp,masks,w,lower_z_mm=44.,surface_z_mm=50.,surface_focus_tolerance_nm=1.,
        maximum_spacing_mm=.5,waist_detector='finite_region')
    assert report['status']=='FAIL' and report['forbidden_waists']
    assert report['forbidden_waists'][0]['source_current_fraction']==pytest.approx(1.)


def test_known_interception_cannot_hide_a_forbidden_c3_adl_waist(monkeypatch):
    data=column(monkeypatch,.05,clipping=22.)
    state,cp,masks,w,ids=data
    selected=(cp.z_mm>=20.)&(cp.z_mm<=25.)
    phi=np.arange(32)*2*np.pi/32;root=21.975
    cp.x_m[selected]=(cp.z_mm[selected]-root)[:,None]*1e-3*.002*np.cos(phi)+1e-11*np.cos(2*phi)
    cp.y_m[selected]=(cp.z_mm[selected]-root)[:,None]*1e-3*.002*np.sin(phi)+1e-11*np.sin(2*phi)
    cp.tx_rad[selected]=.002*np.cos(phi);cp.ty_rad[selected]=.002*np.sin(phi)
    for index,position in ((0,1e-4),(16,-1e-4*w[0]/w[16])):
        masks[cp.z_mm>=22.,index]=False
        cp.x_m[selected&(cp.z_mm<=22.),index]=position
        cp.y_m[selected,index]=0.;cp.tx_rad[selected,index]=0.;cp.ty_rad[selected,index]=0.
    report=topology(data)
    assert report['population_change_evidence'][0]['status']=='CAPTURED_LOSS_SUPPORTED_BY_OPENING_GEOMETRY'
    assert report['intervals']['c3_adl']['finite_region_evidence']['radial_candidates']==[]
    assert report['status']=='FAIL' and not report['passed']
    assert report['failures']==['forbidden_c3_adl_waist']
    assert report['forbidden_adjacent_cohort_candidates']
    assert not chain(data)['required_topology_satisfied_at_sampling']


def test_waist_inside_c2_body_does_not_qualify_the_open_gap(monkeypatch):
    data=column(monkeypatch,.1)
    state=data[0]
    # The fixture waist at 15 mm is after the C2 optical reference at 10 mm,
    # but before the exit face of this installed 12 mm lens at 16 mm.
    state.lenses[1].mechanical_length_mm=12.
    gate=topology(data)
    assert not gate['passed']
    assert gate['intervals']['c2_c3']['bounds_mm']==[16.,19.]
    assert not gate['intervals']['c2_c3']['waists']
    inventory=chain(data)
    assert not inventory['required_topology_satisfied_at_sampling']
    assert inventory['intervals']['c2_exit_section']['waist_z_mm']
    assert not inventory['intervals']['c2_to_c3']['waist_z_mm']


def test_installed_condenser_components_supply_the_finite_gap_faces():
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.optics.column import default_state

    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection('FEG', 'C3 + Probe Corrector', 'No Energy Filter'))
    state.simulation_mode = 'unified'
    state.probe_mode = 'nano_probe'
    for deflector in (state.ac_deflector, state.descan_deflector):
        deflector.scan_enabled = False
        deflector.wobble_enabled = False
    gaps = condenser_topology.inter_lens_gaps_mm(state)
    np.testing.assert_allclose(gaps, [[750., 830.], [1000., 1070.]], rtol=0, atol=1e-9)
    capture = condenser_topology.requested_capture_planes(state)
    assert all(face in capture for gap in gaps for face in gap)
    assert regions_for_state(state)['c2_to_c3']['requirement'] == 'exactly_one_2d_waist'
    state.nanoprobe_range = 'large_angle'
    assert regions_for_state(state)['c2_to_c3']['requirement'] == 'required_2d_waist'


def test_real_source_ids_reach_direct_focus_subreports_from_both_consumers(monkeypatch):
    data=column(monkeypatch)
    for report in (topology(data)['direct_focus'],chain(data)['direct_focus']):
        assert report['source_identity_basis']=='original_source_ids'
        evidence=report['finite_region_evidence']
        assert evidence['source_ids']==data[4].tolist()
        assert evidence['source_identity_basis']=='original_source_ids'


def test_direct_focus_without_ids_labels_column_index_fallback(monkeypatch):
    from temsim.optics.direct_focus_topology import evaluate_for_state
    state,cp,masks,w,ids=column(monkeypatch)
    report=evaluate_for_state(state,cp,masks,w)
    assert report['passed']
    assert report['source_identity_basis']=='checkpoint_column_indices'
    assert report['finite_region_evidence']['source_identity_basis']=='checkpoint_column_indices'


@pytest.mark.parametrize('damage',['short','duplicate'])
def test_direct_focus_entry_rejects_invalid_source_ids_before_terminal_handling(monkeypatch,damage):
    from temsim.optics.direct_focus_topology import evaluate_for_state
    state,cp,masks,w,ids=column(monkeypatch)
    if damage=='short':ids=ids[:-1]
    else:ids=np.zeros_like(ids)
    report=evaluate_for_state(state,cp,masks,w,source_ids=ids)
    assert report['status']=='UNRESOLVED' and not report['passed']
    assert report['failures']==['invalid_direct_focus_arrays']
