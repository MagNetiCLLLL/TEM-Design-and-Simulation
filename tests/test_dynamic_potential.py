"""Retained time-dependent potentials, induction, work and native drive parity."""
from dataclasses import replace
import json
import math

import numpy as np
import pytest
from scipy.constants import c, e, m_e
from scipy.integrate import solve_ivp

from temsim.physics.dynamic_potential import NativeTransverseDrive, DynamicDipolePotential, DynamicPotentialSnapshot
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.potential_snapshot import PotentialContribution
from temsim.physics.unified_field_particle import _sampler, trace_unified_response
from test_unified_field_particle import controlled, execute
from test_unified_field_runtime import parked_state


def coil(*, registration=CoordinateRegistration(), raster=False):
    p = np.sqrt(300000*e*(300000*e+2*m_e*c*c))/c
    base = FrozenPosedMultipole("coil:0","coil",registration,((1,0,2e-3),(0,1,-1e-3)),
        "uniform",.451,.002,(450.,452.),.005,p,captured_time_s=0.)
    law = (NativeTransverseDrive("scan","raster",((10.,2.),(-3.,5.)),1e-7,4)
           if raster else NativeTransverseDrive("wobble","sine",((10.,0.),(-3.,0.)),1e-9,phase_rad=.3))
    return DynamicDipolePotential(AnalyticPotentialDomain(AnalyticMagneticPotential(base)),("coil:0",),(law,),0.)


def test_raster_and_sine_commands_derivatives_and_bounds():
    raster = NativeTransverseDrive("scan","raster",((.3,-.2),(.1,.4)),.7,7)
    time = np.array([-.342,.021,.154,.571])
    phase = (time/.7)%1
    line = np.floor(phase*7)
    expected = np.stack((2*(phase*7-line)-1,2*(line+.5)/7-1),axis=-1)@np.array(raster.matrix_mrad).T
    value, first, second = raster.evaluate(time)
    np.testing.assert_allclose(value,expected,rtol=0,atol=2e-15)
    np.testing.assert_allclose(first,np.broadcast_to((6.,2.),first.shape),rtol=2e-15)
    assert np.count_nonzero(second)==0
    sine = NativeTransverseDrive("wobble","sine",((.3,0.),(-.7,0.)),.7,phase_rad=.2)
    for law in (raster,sine):
        values = law.evaluate(time)
        epsilon = 1e-6
        np.testing.assert_allclose((law.evaluate(time+epsilon)[0]-law.evaluate(time-epsilon)[0])/(2*epsilon),
                                  values[1],rtol=1e-9,atol=1e-10)
        for order,array in enumerate(values):
            assert np.all(np.abs(array)<=law.derivative_bound(order))
    with pytest.raises(ValueError,match="flyback"):
        raster.require_time_interval(.09,.11)
    with pytest.raises(ValueError,match="flyback"):
        raster.require_differentiable_time(0.)
    raster.require_time_interval(.021,.022)


def test_vector_temporal_guard_matches_authoritative_scalar_intervals_bitwise():
    law=NativeTransverseDrive("scan","raster",((.1,.2),(.3,.4)),.7,7)
    period=law.period_s/law.lines
    times=np.r_[np.linspace(-10.,10.,3001),np.nextafter(np.arange(-8,9)*period,-np.inf),
                np.arange(-8,9)*period,np.nextafter(np.arange(-8,9)*period,np.inf)]
    expected=[]
    for time in times:
        index=math.floor(float(time)/period)
        while time<index*period:index-=1
        while time>=(index+1)*period:index+=1
        expected.append((index+1)*period)
    expected=np.asarray(expected)
    np.testing.assert_array_equal(law._next_time_events(times),expected)
    law.require_time_interval(times,np.nextafter(expected,-np.inf))
    with pytest.raises(ValueError,match="flyback"):
        law.require_time_interval(times,expected)


def test_raster_waveform_and_flyback_guard_share_adjacent_clock_identity():
    law=NativeTransverseDrive("scan","raster",((1.,0.),(0.,1.)),.01,256)
    old_start=.0011328125
    old_stop=np.nextafter(old_start,np.inf)
    # The former waveform jumped here while the independent event formula did
    # not. With one clock authority both values belong to the same new line.
    law.require_time_interval(old_start,old_stop)
    assert abs(law.evaluate(old_stop)[0][0]-law.evaluate(old_start)[0][0])<1e-12
    rng=np.random.default_rng(174)
    indices=np.r_[np.arange(-300,301),rng.integers(-100000,100000,1000)]
    boundaries=indices*(law.period_s/law.lines)
    before=np.nextafter(boundaries,-np.inf)
    after=np.nextafter(boundaries,np.inf)
    np.testing.assert_array_equal(law._next_time_events(before),boundaries)
    np.testing.assert_array_equal(law.evaluate(boundaries)[0][...,0],-np.ones(len(boundaries)))
    assert np.all(law.evaluate(before)[0][...,0]>.999999)
    with pytest.raises(ValueError,match="flyback"):
        law.require_time_interval(before,after)
    with pytest.raises(ValueError,match="flyback"):
        law.require_differentiable_time(boundaries)
    law.require_time_interval(boundaries,after)
    assert law.to_dict()["time_policy"]=="rounded-line-interval-v1"


def test_dynamic_potential_curl_and_time_derivative_are_one_model():
    potential=coil(registration=CoordinateRegistration(rotation_local_to_global=((0.,-1.,0.),(1.,0.,0.),(0.,0.,1.))))
    point=np.array((2e-5,-3e-5,.451));time=2.15e-9
    value=potential.sample_jet(point,time)
    dt=1e-15
    difference=(potential.sample(point,time+dt)["A_t_m"]-potential.sample(point,time-dt)["A_t_m"])/(2*dt)
    np.testing.assert_allclose(value["E_v_per_m"],-difference,rtol=5e-10,atol=1e-10)
    j=value["jac_A_t"]
    np.testing.assert_allclose(value["B_t"],(j[2,1]-j[1,2],j[0,2]-j[2,0],j[1,0]-j[0,1]),rtol=2e-15)
    jdt=value["jac_dA_dt_t_per_s"]
    curl_e=-np.array((jdt[2,1]-jdt[1,2],jdt[0,2]-jdt[2,0],jdt[1,0]-jdt[0,1]))
    bdot=(potential.sample(point,time+dt)["B_t"]-potential.sample(point,time-dt)["B_t"])/(2*dt)
    np.testing.assert_allclose(curl_e,-bdot,rtol=5e-10,atol=1e-3)
    assert not np.count_nonzero(value["hess_A_t_per_m"])


def test_shared_sampler_retains_uncompiled_analytic_round_provider():
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    field=FrozenAnalyticField("round",((.1,.451,.002),),CoordinateRegistration(),(400.,500.),.005)
    potential=AnalyticPotentialDomain(AnalyticMagneticPotential(field))
    state=parked_state()
    snapshot=controlled(state,(PotentialContribution("round",("round",),potential,"magnetic"),)).snapshot
    points=np.array([[1e-5,-2e-5,.451]])
    for preference in (False,True):
        actual=compile_unified_potential_sampler(snapshot,prefer_plane=preference).sample(points)
        for key,value in snapshot.sample(points).items():
            np.testing.assert_array_equal(actual[key],value)


def test_actual_normal_dynamic_source_reaches_specimen_with_unchanged_optical_inputs(record_property):
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_condenser_mode
    from temsim.simulation_pipeline import calculate_particle_section
    state=default_state();state.simulation_mode="unified";state.acceleration_backend="CPU"
    apply_condenser_mode(state,"nano_probe")
    # Only the numerical count changes; this is the actual default emission
    # distribution and electrode transport, not a constructed Gaussian source.
    state.electron_gun.emitter.ray_count=9
    source_names=("virtual_source_fwhm_nm","emission_energy_ev","energy_spread_fwhm_ev",
                  "angular_rms_mrad","angular_cutoff_mrad")
    original_source={key:getattr(state.electron_gun.emitter,key) for key in source_names}
    original_lenses=tuple((row.key,row.z_mm,row.percent) for row in state.lenses)
    result=calculate_particle_section(state,target_z_mm=state.sample.z_mm,acquire_stem=False)
    simulation=result.simulation;branch=simulation.incident
    assert tuple((row.key,row.z_mm,row.percent) for row in state.lenses)==original_lenses
    assert {key:getattr(state.electron_gun.emitter,key) for key in source_names}==original_source
    assert state.ac_deflector.scan_enabled and state.descan_deflector.scan_enabled
    assert branch.z[-1]==state.sample.z_mm and branch.alive.any()
    np.testing.assert_array_equal(branch.ray_weight,simulation.gun_trace.exit_bundle.weight)
    assert np.isfinite(branch.flight_time_s[-1,branch.alive]).all()
    assert simulation.metrics["emission_time_s"]==0.
    assert simulation.metrics["dynamic_snapshot_digest"]
    assert not simulation.metrics["unified_empirical_aberrations_applied"]
    runtime=simulation.incident_plan.unified_runtime
    assert len(runtime.snapshot.dynamic_coils)==4
    assert runtime.metadata["round_remote_domain_policy"]
    record_property("actual_normal_dynamic_source",json.dumps({"source":original_source,
        "numerical_particle_count":9,"surviving_count":int(branch.alive.sum()),
        "absolute_incident_weight":float(branch.ray_weight.sum()),
        "absolute_surviving_weight":float(branch.ray_weight[branch.alive].sum()),
        "field_digest":runtime.snapshot.numerical_digest,"geometry_digest":runtime.geometry.digest,
        "scope":"Actual default FEG distribution and current Normal preset; dynamic native coils and retained static gun electric tail; no material transport"}))


def test_dynamic_snapshot_roundtrip_domain_and_compiled_sampler():
    state=parked_state(); static=controlled(state).snapshot
    source=coil(); snapshot=DynamicPotentialSnapshot(static,(source,))
    restored=DynamicPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.numerical_digest==snapshot.numerical_digest
    points=np.array([[1e-5,2e-5,.451],[1e6,-1e6,.1]])
    time=np.array([2.2e-9,0.])
    for name,value in snapshot.sample(points,time).items():
        np.testing.assert_array_equal(restored.sample(points,time)[name],value)
        np.testing.assert_allclose(_sampler(snapshot)(points,time)[name],value,rtol=2e-15,atol=1e-30)
    assert not snapshot.sample(points,time)["B_t"][1].any()
    with pytest.raises(ValueError,match="radial"):
        snapshot.sample(np.array([.006,0,.451]),2.2e-9)
    with pytest.raises(ValueError,match="Duplicate"):
        DynamicPotentialSnapshot(static,(source,source))
    duplicate=replace(static,contributions=(PotentialContribution("other",("coil:0",),source.base,"magnetic"),))
    with pytest.raises(ValueError,match="coverage"):
        DynamicPotentialSnapshot(duplicate,(source,))


def test_axial_one_sided_jets_do_not_move_faces_or_relax_tilted_domains():
    source=coil();point=np.array((1e-5,2e-5,.45));time=2.2e-9
    with pytest.raises(ValueError,match="hard axial face"):
        source.sample_jet(point,time)
    inside=source.sample_jet(point,time,axial_side="inside")
    np.testing.assert_array_equal(inside["A_t_m"],source.sample(point,time)["A_t_m"])
    outside=source.sample_jet(point,time,axial_side="outside")
    assert all(np.count_nonzero(value)==0 for value in outside.values())
    angle=.1
    rotation=((np.cos(angle),0.,np.sin(angle)),(0.,1.,0.),(-np.sin(angle),0.,np.cos(angle)))
    tilted=coil(registration=CoordinateRegistration(rotation_local_to_global=rotation))
    with pytest.raises(ValueError,match="aligned"):
        tilted.sample_jet(point,time,axial_side="inside")


def test_dynamic_particle_requires_real_clock_and_rejects_temporal_flyback(monkeypatch):
    import temsim.physics.unified_field_particle as module
    state=parked_state();runtime=controlled(state)
    runtime=replace(runtime,snapshot=DynamicPotentialSnapshot(runtime.snapshot,(coil(raster=True),)))
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:runtime)
    plan=module.build_unified_plan(state,450.,452.,maximum_step_mm=.02)
    with pytest.raises(ValueError,match="actual finite"):
        module.execute_unified_plan(state,plan,[0.],[0.],[0.],[0.])
    with pytest.raises(ValueError,match="flyback"):
        module.execute_unified_plan(state,plan,[0.],[0.],[0.],[0.],initial_time_s=np.array([2.5e-8-1e-13]))


def test_dynamic_epoch_and_energy_clock_resume_keep_absolute_time_once(monkeypatch):
    import temsim.physics.unified_field_particle as module
    from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
    state=parked_state();state.simulation_time_s=.21e-9
    runtime=controlled(state)
    runtime=replace(runtime,snapshot=DynamicPotentialSnapshot(runtime.snapshot,(coil(),)),
                    metadata={"emission_time_s":state.simulation_time_s})
    runtime=UnifiedFieldRuntime.from_dict(json.loads(json.dumps(runtime.to_dict())))
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:runtime)
    initial=np.array([[1e-5,-2e-5,.002,.001]])
    def advance(start,stop,phase,energy,time):
        plan=module.build_unified_plan(state,start,stop,maximum_step_mm=.02)
        energies=[]
        result=module.execute_unified_plan(state,plan,phase[:,0],phase[:,2],phase[:,1],phase[:,3],
            initial_kinetic_energy_ev=energy,initial_time_s=time,return_flight_times=True,energy_output=energies)
        phase=np.c_[result[1][-1],result[3][-1],result[2][-1],result[4][-1]]
        return phase,energies[0][-1],result[5][-1]
    whole=advance(450.,452.,initial,300000.,np.array([2e-9]))
    prefix=advance(450.,451.,initial,300000.,np.array([2e-9]))
    resumed=advance(451.,452.,*prefix)
    for expected,actual in zip(whole,resumed):
        np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=1e-18)
    zero_epoch=replace(runtime,metadata={"emission_time_s":0.})
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:zero_epoch)
    with pytest.raises(ValueError,match="emission epoch"):
        advance(450.,452.,initial,300000.,np.array([2e-9]))
    with pytest.raises(ValueError,match="finite real"):
        replace(runtime,metadata={"emission_time_s":False})


@pytest.mark.parametrize("electric_gradient",(0.,2e6))
def test_dynamic_particle_induction_work_matches_independent_momentum_ode(monkeypatch,electric_gradient):
    state=parked_state(); source=coil(); runtime=controlled(state)
    if electric_gradient:
        from temsim.physics.planar_gun_field import PlanarGunField
        from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
        radius,z=np.array([0.,.02]),np.array([.449,.453])
        phi=np.broadcast_to(200000.+electric_gradient*(z-.45),(2,2)).copy()
        electric=ElectricPotentialSnapshot.from_provider(PlanarGunField({},radius,z,phi))
        runtime=controlled(state,(PotentialContribution("electric",("electric",),electric,"electric"),))
    runtime=replace(runtime,snapshot=DynamicPotentialSnapshot(runtime.snapshot,(source,)))
    initial=np.array([[2e-5,-3e-5,.002,-.001]])
    _,result,energies=execute(state,runtime,initial,monkeypatch,step=.02)
    energy=300000*e;p=np.sqrt(energy*(energy+2*m_e*c*c))/c
    direction=np.array((.002,-.001,1.));momentum=p*direction/np.linalg.norm(direction)
    def full_momentum(z, values):
        position=np.array((values[0],values[1],z))
        momentum=values[2:5]; gamma_mass=np.sqrt(m_e*m_e+np.dot(momentum,momentum)/(c*c))
        velocity=momentum/gamma_mass
        fields=source.sample(position,values[5])
        electric=fields["E_v_per_m"]+np.array((0.,0.,-electric_gradient))
        force=-e*(electric+np.cross(velocity,fields["B_t"]))
        return np.r_[velocity[:2]/velocity[2],force/velocity[2],1/velocity[2]]
    reference=solve_ivp(full_momentum,(.45,.452),np.r_[initial[0,:2],momentum,2e-9],
        method="DOP853",rtol=2e-13,atol=(1e-17,1e-17,1e-34,1e-34,1e-34,1e-21),max_step=1e-5).y[:,-1]
    expected=np.r_[reference[:2],reference[2:4]/reference[4]]
    actual=np.array((result[1][-1,0],result[3][-1,0],result[2][-1,0],result[4][-1,0]))
    np.testing.assert_allclose(actual,expected,rtol=2e-9,atol=2e-13)
    expected_energy=(np.sqrt(np.dot(reference[2:5],reference[2:5])*c*c+(m_e*c*c)**2)-m_e*c*c)/e
    np.testing.assert_allclose(energies[-1,0],expected_energy,rtol=0,atol=2e-8)
    assert abs(energies[-1,0]-300000-electric_gradient*.002)>1e-4
    np.testing.assert_allclose(result[5][-1,0],reference[5],rtol=1e-12)


def test_native_normal_four_dynamic_physical_coils_match_original_time_laws():
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_condenser_mode
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime,UnifiedFieldRuntime
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    state=default_state();state.simulation_mode="unified"
    apply_condenser_mode(state,"nano_probe")
    runtime=capture_unified_field_runtime(state)
    assert len(runtime.snapshot.dynamic_coils)==4
    assert {coil.source_id.split(":")[0] for coil in runtime.snapshot.dynamic_coils}=={"ac_deflector","image_diffraction_deflector"}
    assert state.ac_deflector.scan_enabled and state.descan_deflector.scan_enabled
    for time in (.00123,.0141,.23456,-.0123):
        originals={field.key:field for field in capture_posed_column_fields(state,time_s=time,include_aligned=True,
                    component_keys=("ac_deflector","image_diffraction_deflector"))}
        for coil in runtime.snapshot.dynamic_coils:
            field=originals[coil.source_id]
            point=np.array((1e-6,-2e-6,field.center_z_m))
            # The raw polynomial may almost cancel at this off-axis point;
            # the absolute tolerance is below 1e-14 of either source term.
            np.testing.assert_allclose(coil.sample(point,time)["A_t_m"],field.vector_potential_at_global_positions_t_m(point),rtol=2e-13,atol=1e-24)
            np.testing.assert_allclose(coil.sample(point,time)["B_t"],field.field_at_global_positions_t(point),rtol=2e-13,atol=1e-18)
    assert UnifiedFieldRuntime.from_dict(runtime.to_dict()).digest==runtime.digest
    electric=[row for row in runtime.snapshot.static_snapshot.contributions if row.kind=="electric"]
    assert electric and np.any(electric[0].potential.sample(np.array([0.,0.,.45]))["E_v_per_m"]!=0)


def test_static_response_helper_uses_finite_az_increment_and_preserves_input(monkeypatch):
    import temsim.physics.unified_field_particle as module
    state=parked_state();source=coil().base
    runtime=controlled(state,(PotentialContribution(source.source_id,("coil:0",),source,"magnetic"),))
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:runtime)
    kwargs=dict(energy_ev=300000.,flight_time_s=2e-9,maximum_step_m=2e-5)
    baseline=trace_unified_response(state,np.zeros((1,4)),.45,(.451,.452),**kwargs)
    changed=trace_unified_response(state,np.zeros((1,4)),.45,(.451,.452),coil_perturbations_rad={"coil:0":(1e-5,0.)},**kwargs)
    assert changed["receipt"]["parent_runtime_digest"]==runtime.digest
    assert changed["receipt"]["runtime_digest"]!=runtime.digest
    assert changed["alive"].all()
    assert abs(changed["xy_slopes"][-1,0,2]-baseline["xy_slopes"][-1,0,2])>9e-6
    assert changed["vector_potential_t_m"].shape==(2,1,3)
    assert np.any(changed["vector_potential_t_m"]!=baseline["vector_potential_t_m"])


def test_dynamic_plan_archive_preserves_time_laws_and_execution(monkeypatch):
    from temsim.particle_section_io import _pack_record_graph,_unpack_record_graph
    from temsim.physics.unified_field_particle import execute_unified_plan
    state=parked_state();runtime=controlled(state)
    runtime=replace(runtime,snapshot=DynamicPotentialSnapshot(runtime.snapshot,(coil(),)))
    plan,original,_=execute(state,runtime,np.array([[1e-5,2e-5,.001,-.001]]),monkeypatch)
    arrays={};record=_pack_record_graph(plan,arrays)
    restored=_unpack_record_graph(record,arrays,maximum_unpacked_bytes=32*1024**2)
    assert type(restored.unified_runtime.snapshot) is DynamicPotentialSnapshot
    assert restored.unified_runtime.snapshot.numerical_digest==runtime.snapshot.numerical_digest
    replay=execute_unified_plan(state,restored,[1e-5],[.001],[2e-5],[-.001],
        initial_kinetic_energy_ev=300000.,initial_time_s=np.array([2e-9]),return_flight_times=True)
    for left,right in zip(original[:6],replay[:6]):
        np.testing.assert_array_equal(left,right)


@pytest.mark.parametrize("first_survives",(True,False))
def test_absorbed_source_placeholders_do_not_enter_dynamic_column(monkeypatch,first_survives):
    import temsim.physics.unified_field_particle as module
    from temsim.physics.particle_sections import _trace_segment
    state=parked_state();runtime=controlled(state)
    runtime=replace(runtime,snapshot=DynamicPotentialSnapshot(runtime.snapshot,(coil(),)))
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:runtime)
    phase=np.zeros(2)
    initial=(phase,phase,phase,phase,np.array([2e-9 if first_survives else np.nan,np.nan]))
    survival=(np.array([first_survives,False]),np.array([np.nan if first_survives else 425.,420.]),
              ("" if first_survives else "gun_aperture","gun_aperture"))
    identity=(np.array([0,1]),np.zeros(2))
    segment,_,_=_trace_segment(state,"incident",450.,452.,initial,np.zeros(2),np.array([.3,.7]),
        identity,survival,(),None,float("inf"),initial_kicks=True)
    np.testing.assert_array_equal(segment.branch.alive,[first_survives,False])
    np.testing.assert_array_equal(segment.branch.ray_weight,[.3,.7])
    assert segment.branch.blocked_key[1]=="gun_aperture"
    assert np.isnan(segment.branch.x[:,1]).all()


@pytest.mark.parametrize("radius_mm",(0.,2e-4))
def test_actual_gun_absorption_keeps_zero_or_partial_absolute_source_weight(radius_mm,record_property):
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_condenser_mode
    from temsim.physics.particle_sections import run_particle_section
    state=default_state();state.simulation_mode="unified";state.acceleration_backend="CPU"
    apply_condenser_mode(state,"nano_probe")
    state.electron_gun.emitter.ray_count=9
    state.electron_gun.c1_aperture.radius_mm=radius_mm
    result=run_particle_section(state,observation_stop_z_mm=451.)
    source=result.gun_trace.exit_bundle
    assert not source.alive.all()
    if radius_mm==0.:
        assert not source.alive.any() and not result.incident.alive.any()
    else:
        assert source.alive.any()
    np.testing.assert_array_equal(result.incident.alive,source.alive)
    np.testing.assert_array_equal(result.incident.ray_weight,source.weight)
    assert np.isnan(result.incident.x[-1,~source.alive]).all()
    record_property("physical_gun_aperture",json.dumps({"radius_mm":radius_mm,
        "emitted_weight":float(np.sum(source.weight)),"transmitted_weight":float(np.sum(source.weight[source.alive])),
        "alive":int(source.alive.sum()),"source_model":"Actual native FEG; only the physical aperture opening changed"}))


@pytest.mark.parametrize("prefer_plane",(False,True))
def test_shared_sampler_preserves_static_electric_gauge_and_dynamic_fields(prefer_plane):
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import FrozenAnalyticField
    from temsim.physics.potential_snapshot import StaticPolynomialGauge
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    state=parked_state()
    radius,z=np.array([0.,.02]),np.array([.449,.453])
    phi=np.broadcast_to(200000.+2e6*(z-.45),(2,2)).copy()
    electric=ElectricPotentialSnapshot.from_provider(PlanarGunField({},radius,z,phi))
    round_lens=HighOrderRoundLensPotential(FrozenAnalyticField("round",((.02,.451,.005),),
        CoordinateRegistration(),(400.,500.),.005),axial_domain="gaussian_all_z")
    runtime=controlled(state,(PotentialContribution("electric",("electric",),electric,"electric"),
        PotentialContribution("round",("round",),round_lens,"magnetic")))
    gauge=StaticPolynomialGauge(0.,(1e-8,2e-8,3e-8),((1e-6,2e-6,0.),(2e-6,-1e-6,0.),(0.,0.,0.)))
    snapshot=DynamicPotentialSnapshot(replace(runtime.snapshot,gauge=gauge),(coil(),))
    sampler=compile_unified_potential_sampler(snapshot,prefer_plane=prefer_plane)
    assert compile_unified_potential_sampler(DynamicPotentialSnapshot.from_dict(snapshot.to_dict()),
        prefer_plane=prefer_plane) is sampler
    x,y=np.meshgrid(np.linspace(-2e-5,2e-5,17),np.linspace(-3e-5,3e-5,19))
    points=np.stack((x,y,np.full_like(x,.451)),axis=-1)
    for name,value in snapshot.sample(points,2.2e-9).items():
        np.testing.assert_allclose(sampler(points,2.2e-9)[name],value,rtol=2e-13,atol=1e-20)
    assert sampler.snapshot_digest==snapshot.numerical_digest
    with pytest.raises(ValueError,match="absolute laboratory"):
        sampler(points)
    with pytest.raises(ValueError,match="finite absolute"):
        sampler(points,float("nan"))
    with pytest.raises(ValueError,match="domain"):
        sampler(np.array([.006,0.,.451]),2.2e-9)
