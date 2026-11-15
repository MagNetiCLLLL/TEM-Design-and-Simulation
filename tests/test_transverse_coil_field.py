"""Pure field/current/potential checks; no source population or wave execution."""
from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.constants import mu_0

from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.transverse_coil_field import (
    TransverseCoilPotential,FrozenTransverseCoilField,quadrupole_gradient_t_per_m,
    calibrate_transverse_current_matrix,capture_transverse_coil_fields,
)
from temsim.physics.dynamic_potential import NativeTransverseDrive,DynamicPotentialSnapshot
from temsim.physics.dynamic_transverse_coil import DynamicTransverseCoilPotential
from temsim.transverse_current import BIAS_FIELD


def rectangular_coil(**changes):
    result=dict(family='radial_rectangle',coil_index=1,station_index=0,group='x',current_sign=1,
        azimuth_deg=0.,center_z_mm=0.,radial_start_mm=19.9,radial_end_mm=20.1,
        outer_width_mm=24.2,outer_height_mm=24.2,window_width_mm=24.,window_height_mm=24.)
    return {**result,**changes}


def potential(currents=(1.,0.),**changes):
    return TransverseCoilPotential('test_loop',(rectangular_coil(),),7,currents,
        pack_order=8,path_order=24,provenance={'clear_bore_radius_m':.005},**changes)


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Pure winding-field tests must not emit or trace a source')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


def test_rectangular_loop_pack_matches_independent_on_axis_formula():
    p=potential()
    point=np.array([[.020,0.,0.]])
    actual=p.field_at_global_positions_t(point)[0]
    # Independent exact rectangular-loop expression, averaged with a higher
    # two-dimensional rule rather than summing the production line elements.
    x,w=np.polynomial.legendre.leggauss(48)
    radial=.020+.0001*x
    fraction=(x+1)/2
    a=(24.+.2*fraction)*.0005
    offset=.020-radial[:,None]
    squared=offset**2
    expected=7*mu_0/np.pi*np.sum((w[:,None]*w[None,:]/4)*a[None,:]**2/
        np.sqrt(2*a[None,:]**2+squared)*2/(a[None,:]**2+squared))
    np.testing.assert_allclose(actual,[expected,0.,0.],rtol=5e-13,atol=1e-18)


def test_retained_A_curl_matches_B_and_jet_derivatives():
    p=potential()
    point=np.array([.001,-.002,.003])
    jet=p.sample_jet(point)
    j=jet['jac_A_t']
    curl=np.array([j[2,1]-j[1,2],j[0,2]-j[2,0],j[1,0]-j[0,1]])
    np.testing.assert_allclose(curl,p.sample(point)['B_t'],rtol=2e-13,atol=1e-18)
    h=1e-7
    for axis in range(3):
        delta=np.eye(3)[axis]*h
        plus,minus=p.sample_jet(point+delta),p.sample_jet(point-delta)
        np.testing.assert_allclose((plus['A_t_m']-minus['A_t_m'])/(2*h),j[:,axis],rtol=2e-9,atol=1e-13)
        np.testing.assert_allclose((plus['jac_A_t']-minus['jac_A_t'])/(2*h),jet['hess_A_t_per_m'][:,:,axis],rtol=2e-8,atol=1e-10)


def test_current_turn_linearity_zero_and_full_axial_tails():
    p=potential()
    points=np.array([[0.,0.,0.],[0.,0.,.5]])
    b=p.field_at_global_positions_t(points)
    assert np.linalg.norm(b[1])>0 and np.linalg.norm(b[1])<np.linalg.norm(b[0])
    np.testing.assert_array_equal(potential((0.,0.)).field_at_global_positions_t(points),np.zeros_like(points))
    np.testing.assert_allclose(potential((-.25,0.)).field_at_global_positions_t(points),-.25*b,rtol=2e-14,atol=1e-20)
    np.testing.assert_allclose(replace(p,turn_count_per_coil=14).field_at_global_positions_t(points),2*b,rtol=2e-14,atol=1e-20)


def test_rigid_pose_rotates_source_and_vector_field_consistently():
    p=potential()
    r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    shift=np.array([.001,.002,.003])
    posed=replace(p,registration=CoordinateRegistration(tuple(shift),tuple(map(tuple,r))))
    points=np.array([[0.,0.,0.],[.001,-.002,.003]])
    np.testing.assert_allclose(posed.field_at_global_positions_t(points@r.T+shift),p.field_at_global_positions_t(points)@r.T,rtol=2e-13,atol=1e-18)


def test_winding_material_is_excluded_without_clipping_axial_tails():
    p=potential()
    with pytest.raises(ValueError,match='occupied winding pack'):
        p.sample(np.array([.020,.01205,0.]))
    assert np.isfinite(p.sample(np.array([.020,0.,0.]))['B_t']).all()


def test_serialization_is_detached_and_geometry_current_change_identity():
    p=potential()
    record=json.loads(json.dumps(p.to_dict()))
    restored=TransverseCoilPotential.from_dict(record)
    assert restored.digest==p.digest
    assert replace(p,currents_a=(2.,0.)).digest!=p.digest
    assert replace(p,turn_count_per_coil=8).digest!=p.digest
    record['payload']['coils'][0]['radial_start_mm']=19.8
    with pytest.raises(ValueError,match='identity mismatch'):
        TransverseCoilPotential.from_dict(record)


@pytest.mark.parametrize('values',[(True,0.),(float('nan'),0.),(float('inf'),0.),('1',0.),(1.,)])
def test_invalid_currents_fail(values):
    with pytest.raises(ValueError):potential(values)


@pytest.mark.parametrize('updates',[
    {'radial_start_mm':-1.},{'radial_end_mm':19.8},{'window_width_mm':25.},
    {'outer_height_mm':0.},{'window_height_mm':float('nan')},{'station_index':True},
    {'coil_index':0},{'current_sign':True},{'azimuth_deg':float('inf')},
    {'group':'z'},{'family':'open_wire'},
])
def test_invalid_physical_loop_descriptors_reject_before_sampling(updates):
    with pytest.raises(ValueError):
        TransverseCoilPotential('invalid',(rectangular_coil(**updates),),1,(1.,0.))


def test_duplicate_coils_and_excessive_quadrature_fail_without_silent_omission():
    coil=rectangular_coil()
    with pytest.raises(ValueError,match='counted twice'):
        TransverseCoilPotential('duplicate',(coil,coil),1,(1.,0.))
    coils=tuple(rectangular_coil(coil_index=i+1,azimuth_deg=45*i) for i in range(8))
    p=TransverseCoilPotential('resource',coils,1,(1.,0.),pack_order=64,path_order=64)
    with pytest.raises(ValueError,match='No source, turn or integration order was omitted'):
        p.sample(np.zeros(3))


def dynamic(raster=False):
    base=potential((.2,-.1))
    drive=(NativeTransverseDrive('scan','raster',((.3,.2),(-.2,.4)),.7,7) if raster else
           NativeTransverseDrive('wobble','sine',((.3,0.),(-.2,0.)),.7,phase_rad=.2))
    return DynamicTransverseCoilPotential(base,('test_loop',),(drive,),(((.4,.1),(-.2,.5)),),.021)


def test_dynamic_current_uses_native_waveform_and_retains_induced_E():
    p=dynamic()
    point=np.array([.001,-.002,.003]);time=.157
    sample=p.sample_jet(point,time)
    current,first,_=p.coefficients(time)
    expected=replace(p.base,currents_a=tuple(current)).sample(point)
    np.testing.assert_allclose(sample['B_t'],expected['B_t'],rtol=5e-13,atol=1e-18)
    dt=1e-6
    derivative=(p.sample(point,time+dt)['A_t_m']-p.sample(point,time-dt)['A_t_m'])/(2*dt)
    np.testing.assert_allclose(sample['E_v_per_m'],-derivative,rtol=2e-9,atol=1e-13)
    np.testing.assert_array_equal(sample['E_v_per_m'],-sample['dA_dt_t_m_per_s'])
    assert np.linalg.norm(sample['E_v_per_m'])>0
    j=sample['jac_A_t']
    np.testing.assert_allclose(sample['B_t'],[j[2,1]-j[1,2],j[0,2]-j[2,0],j[1,0]-j[0,1]],rtol=5e-13,atol=1e-18)


def test_dynamic_current_raster_keeps_temporal_jump_guard_and_roundtrips():
    p=dynamic(raster=True)
    with pytest.raises(ValueError,match='flyback'):p.require_time_interval(.09,.11)
    with pytest.raises(ValueError,match='flyback'):p.sample(np.zeros(3),0.)
    restored=DynamicTransverseCoilPotential.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.digest==p.digest
    assert p.temporal_bounds()['derivatives'][1]['A_norm_bound']>0


def test_dynamic_current_rejects_complex_positions_or_electrical_gains():
    p=dynamic()
    with pytest.raises(ValueError,match='finite real XYZ'):
        p.sample(np.array([0.+1e-12j,0.,0.]),.157)
    with pytest.raises(ValueError,match='finite real numbers'):
        replace(p,current_gain_matrices_a_per_mrad=(((1.+1e-12j,0.),(0.,1.)),))


def test_dynamic_current_snapshot_and_unified_sampler_roundtrip():
    from temsim.physics.physical_scene import PhysicalSceneSnapshot,_SCOPE
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    scene=PhysicalSceneSnapshot(dict(scope=_SCOPE,geometry={},components={},component_order=[],
        settings={},gun_inputs={},ownership={},circuits=[],unresolved_inputs=[],identity_complete=True))
    snapshot=DynamicPotentialSnapshot(ElectromagneticPotentialSnapshot(scene,()),(dynamic(),))
    restored=DynamicPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.numerical_digest==snapshot.numerical_digest
    points=np.array([[0.,0.,0.],[.001,.002,.003]])
    expected=snapshot.sample(points,.157)
    actual=compile_unified_potential_sampler(restored).sample(points,.157)
    for key in expected:np.testing.assert_allclose(actual[key],expected[key],rtol=2e-14,atol=1e-20)


@pytest.fixture(scope='module')
def installed_state():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.simulation_modes import switch_mode
    state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
    switch_mode(state,'unified');apply_physical_layout_to_state(state)
    return state


def test_peak_calibration_reference_is_detached_and_full_station_matrix(installed_state):
    s=installed_state
    before=s.to_dict()
    result=calibrate_transverse_current_matrix(s,'beam_deflector',pack_order=6,path_order=24)
    response,target,matrix=(np.array(result[k]) for k in ('unit_current_response','native_target_per_unit','matrix_a_per_unit'))
    assert response.shape==target.shape==matrix.shape==(4,4)
    assert np.max(abs(response[:2,2:]))>0 # Actual remote-station fringe coupling.
    np.testing.assert_allclose(response@matrix,target,rtol=2e-13,atol=1e-15)
    from temsim.physics.core import electron
    charge,momentum,_=electron(s)
    owner=next(c for name in ('deflectors','corrector_elements') for c in getattr(s,name) if c.key=='beam_deflector')
    width=getattr(owner,'effective_thickness_mm',getattr(owner,'thickness_mm',0.))
    conversion=momentum/charge*.001/(width*.001)
    expected=np.zeros((4,4))
    for stage in (0,1):expected[stage*2:stage*2+2,stage*2:stage*2+2]=[[0.,conversion],[-conversion,0.]]
    np.testing.assert_array_equal(target,expected) # Existing project constants, not a newer SciPy mass.
    assert s.to_dict()==before


def test_fixed_current_capture_is_energy_independent_and_native_mode_unchanged(installed_state):
    s=installed_state
    original_assembly=s._resolved_assembly
    before=s.to_dict()
    original_model=s.round_lens_field_model
    fields=calibrate_transverse_current_matrix(s,'probe_dph2_coils',pack_order=6,path_order=24)['fields']
    fields[BIAS_FIELD]=[.02,-.03]
    s._resolved_assembly=replace(original_assembly,parts=tuple(replace(p,data={**p.data,**fields})
        if p.key=='probe_dph2_coils' else p for p in original_assembly.parts))
    had_energy_override=hasattr(s,'_propagation_energy_kev')
    original_energy_override=getattr(s,'_propagation_energy_kev',None)
    try:
        s.round_lens_field_model='copper_coil'
        a,=capture_transverse_coil_fields(s,component_keys={'probe_dph2_deflector'})
        assert type(a) is FrozenTransverseCoilField
        s._propagation_energy_kev=200.
        b,=capture_transverse_coil_fields(s,component_keys={'probe_dph2_deflector'})
        assert a.potential.digest==b.potential.digest
        points=np.array([[0.,0.,a.event_z_mm*.001],[0.,0.,a.event_z_mm*.001+.2]])
        np.testing.assert_array_equal(a.field_at_global_positions_t(points),b.field_at_global_positions_t(points))
        s.round_lens_field_model='gaussian'
        assert capture_transverse_coil_fields(s,component_keys={'probe_dph2_deflector'})==()
    finally:
        s._resolved_assembly=original_assembly
        if had_energy_override:s._propagation_energy_kev=original_energy_override
        else:del s._propagation_energy_kev
        s.round_lens_field_model=original_model


def test_installed_ideal_sources_reach_shared_field_and_potential_graph_once(installed_state):
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    s=installed_state;original=s.round_lens_field_model
    try:
        s.round_lens_field_model='copper_coil'
        snapshot=capture_selected_column_potentials(s,auxiliary_keys=('probe_dph2_deflector','condenser_stigmator'),
            include_gun_alignment=True)
        assert any(type(row.potential) is TransverseCoilPotential for row in snapshot.contributions)
        assert all(type(row.potential) is TransverseCoilPotential for row in snapshot.contributions)
        assert len({owner for row in snapshot.contributions for owner in row.owner_ids})==len(snapshot.contributions)
        source=capture_instrument_magnetic_field(s)
        rows=[r for r in source._sources if type(r.provider) is FrozenTransverseCoilField]
        assert len(rows)>=8
        assert len({row.key for row in rows})==len(rows)
        assert all(row.identity is not None for row in rows)
        assert source.numerical_identity is not None
    finally:s.round_lens_field_model=original


def test_actual_ac_waveform_capture_converts_all_station_commands_to_current(installed_state):
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from temsim.physics.dynamic_transverse_coil import capture_dynamic_transverse_coil
    from temsim.physics.instrument_magnetic import column_deflector_drives
    from temsim.immutable_json import thaw_json
    s=installed_state;original_model=s.round_lens_field_model
    ac=s.ac_deflector
    original=(ac.scan_enabled,ac.wobble_enabled,np.array(ac.scan_command_matrix_mrad).copy())
    try:
        s.round_lens_field_model='copper_coil';ac.scan_enabled=True;ac.wobble_enabled=False
        ac.set_scan_command_matrix_mrad(((.001,0.),(0.,.002)))
        rows=capture_selected_column_potentials(s,auxiliary_keys=(ac.key,),time_s=.000123).contributions
        components={c.key:c for name in ('stigmators','corrector_elements','deflectors') for c in getattr(s,name)}
        dynamic_rows=tuple(capture_dynamic_transverse_coil(row,components) for row in rows)
        assert len(dynamic_rows)==2 and all(type(v) is DynamicTransverseCoilPotential for v in dynamic_rows)
        for time in (.000124,.000125):
            commands=next(row[1] for row in column_deflector_drives(s,time_s=time) if row[0].key==ac.key)
            command=np.array([(x*1000,y*1000) for _,x,y in commands]).reshape(-1)
            for stage,row in enumerate(dynamic_rows):
                electrical=thaw_json(row.base.provenance['electrical'])
                expected=(np.array(electrical['matrix_a_per_unit'])@command+electrical['current_bias_a'])[stage*2:stage*2+2]
                np.testing.assert_allclose(row.coefficients(time)[0],expected,rtol=2e-13,atol=1e-14)
    finally:
        ac.scan_enabled,ac.wobble_enabled=original[:2]
        ac.set_scan_command_matrix_mrad(original[2])
        s.round_lens_field_model=original_model


def test_current_diagnostic_increment_keeps_cross_station_compensation():
    from temsim.physics.unified_field_particle import _ideal_current_diagnostic_increment
    matrix=np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.],[13.,14.,15.,16.]])
    increments={'ac_deflector:0':(1e-6,-2e-6)}
    for stage in (0,1):
        coil=dict(family='saddle',coil_index=1,station_index=stage,group='x',current_sign=1,
            azimuth_deg=0.,center_z_mm=0.,start_z_mm=-5.,end_z_mm=5.,inner_radius_mm=10.,
            outer_radius_mm=10.1,span_deg=120.,band_mm=1.,window_half_angle_rad=np.pi/3-1/10.05)
        base=TransverseCoilPotential(f'ac_deflector:{stage}',(coil,),7,(1.,0.),
            provenance={'component_key':'ac_deflector','station_index':stage,'command_unit':'mrad',
                        'electrical':{'matrix_a_per_unit':matrix.tolist()}})
        altered,found=_ideal_current_diagnostic_increment(base,increments)
        expected=np.array(base.currents_a)+matrix[stage*2:stage*2+2]@np.array([.001,-.002,0.,0.])
        np.testing.assert_array_equal(altered.currents_a,expected)
        assert found==set(increments) and altered.digest!=base.digest


def test_ideal_current_edits_cannot_reuse_an_upstream_column_prefix(installed_state):
    from temsim.physics.particle_sections import _selection_boundary,section_limits
    s=installed_state;original=s.round_lens_field_model
    try:
        s.round_lens_field_model='copper_coil'
        for key in ('probe_dph2_deflector','condenser_stigmator','ac_deflector'):
            assert _selection_boundary(s,(key,))==section_limits(s)[0]
    finally:s.round_lens_field_model=original


def test_remote_diagnostic_window_retains_closed_coil_tails(installed_state):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    s=installed_state;original_model=s.round_lens_field_model;assembly=s._resolved_assembly
    s._resolved_assembly=replace(assembly,parts=tuple(replace(p,data={**p.data,BIAS_FIELD:[.02,-.03]})
        if p.key=='probe_dph2_coils' else p for p in assembly.parts))
    try:
        s.round_lens_field_model='copper_coil'
        full=prepare_magnetic_scene(s)
        bounded=prepare_magnetic_scene(s,z_limits_mm=(1750.,1760.))
        a=next(row for row in full._sources if row.key=='probe_dph2_deflector:0')
        b=next(row for row in bounded._sources if row.key==a.key)
        assert a.provider.field_support_mm[1]<1750.
        point=np.array([[0.,0.,1.755]])
        expected=a.provider.field_at_global_positions_t(point)
        assert np.linalg.norm(expected)>0
        np.testing.assert_array_equal(b.provider.field_at_global_positions_t(point),expected)
        np.testing.assert_array_equal(b.bounds_m[:,2],[1.75,1.76])
    finally:
        s.round_lens_field_model=original_model;s._resolved_assembly=assembly


@pytest.mark.parametrize('name',('deflector','stigmator'))
def test_gun_projection_never_substitutes_native_fields_for_powered_current_coils(installed_state,name):
    from temsim.physics.instrument_magnetic import _captured_gun_sources,gun_paraxial_fields
    s=installed_state;original_model=s.round_lens_field_model;assembly=s._resolved_assembly
    component=getattr(s.electron_gun,name)
    bias=[.02,-.03,.01,0.] if name=='deflector' else [.02,-.03]
    s._resolved_assembly=replace(assembly,parts=tuple(replace(p,data={**p.data,BIAS_FIELD:bias})
        if p.key==component.key else p for p in assembly.parts))
    try:
        s.round_lens_field_model='copper_coil'
        keys={getattr(s.electron_gun,k).key for k in ('deflector','stigmator')}
        rows=_captured_gun_sources(s)
        current_rows=[row for row in rows if row.key.rsplit(':',1)[0] in keys]
        assert len(current_rows)==3 and all(type(row.provider) is FrozenTransverseCoilField for row in current_rows)
        sources=capture_transverse_coil_fields(s,include_gun=True,component_keys=keys)
        z=(component.upper_center_from_tip_mm if name=='deflector' else component.optical_reference_from_tip_mm)*.001
        point=np.array([[.0001,.0002,z]])
        actual=sum(row.provider.field_at_global_positions_t(point) for row in current_rows)
        expected=sum(row.field_at_global_positions_t(point) for row in sources)
        assert np.linalg.norm(actual)>0
        np.testing.assert_array_equal(actual,expected)
        with pytest.raises(ValueError,match='full vector Lorentz'):
            gun_paraxial_fields(s,np.array([z*1000]))
    finally:
        s.round_lens_field_model=original_model;s._resolved_assembly=assembly


def test_actual_ideal_unified_runtime_retains_dynamic_current_sources(installed_state):
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime,UnifiedFieldRuntime
    s=installed_state;original_model=s.round_lens_field_model
    logical=(s.ac_deflector,s.descan_deflector)
    saved=tuple((c.scan_enabled,getattr(c,'wobble_enabled',None),np.array(c.scan_command_matrix_mrad).copy()) for c in logical)
    try:
        s.round_lens_field_model='copper_coil'
        for c in logical:
            c.scan_enabled=True
            if hasattr(c,'wobble_enabled'):c.wobble_enabled=False
            c.set_scan_command_matrix_mrad(((.001,0.),(0.,.002)))
        runtime=capture_unified_field_runtime(s)
        snapshot=runtime.snapshot
        assert type(snapshot) is DynamicPotentialSnapshot
        assert len(snapshot.dynamic_coils)==4
        assert all(type(row) is DynamicTransverseCoilPotential for row in snapshot.dynamic_coils)
        assert not ({row.source_id for row in snapshot.dynamic_coils}&
                    {row.source_id for row in snapshot.static_snapshot.contributions})
        restored=UnifiedFieldRuntime.from_dict(json.loads(json.dumps(runtime.to_dict())))
        assert restored.digest==runtime.digest
        point=np.array([[1e-6,-2e-6,.8]])
        values=restored.snapshot.sample(point,.001234)
        assert all(np.isfinite(value).all() for value in values.values())
        assert np.linalg.norm(values['dA_dt_t_m_per_s'])>0
    finally:
        for c,values in zip(logical,saved):
            c.scan_enabled=values[0]
            if values[1] is not None:c.wobble_enabled=values[1]
            c.set_scan_command_matrix_mrad(values[2])
        s.round_lens_field_model=original_model
