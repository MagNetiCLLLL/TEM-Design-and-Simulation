"""Production adapters against independent Lorentz and physical geometry."""
from dataclasses import replace
import json
import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.core import build_propagation_plan,execute_propagation_plan
from temsim.physics.lens_field_provider import CoordinateRegistration,FrozenAnalyticField
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.posed_wave_hardware import PosedWaveBore
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_geometry import PotentialGeometrySnapshot,PotentialBoundaryHost
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot,PotentialContribution
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
from temsim.physics.unified_field_particle import merge_unified_stops,_sampler
from test_unified_field_runtime import parked_state


def controlled(state,rows=(),pieces=()):
    scene=capture_physical_scene(state)
    snapshot=ElectromagneticPotentialSnapshot(scene,tuple(rows))
    geometry=PotentialGeometrySnapshot(.45,.452,tuple(PotentialBoundaryHost(p.key,(p.key,),(p,)) for p in pieces),
        source_scene_geometry_digest=scene.geometry_digest)
    return UnifiedFieldRuntime(snapshot,geometry,capture_instrument_snapshot(state).digest)


def execute(state,runtime,initial,monkeypatch,energy=300000.,step=.02):
    import temsim.physics.unified_field_particle as module
    monkeypatch.setattr(module,"get_unified_field_runtime",lambda _:runtime)
    plan=build_propagation_plan(state,450,452,maximum_step_mm=step,checkpoint_z_mm=(450,452))
    output=[]
    result=execute_propagation_plan(state,plan,initial[:,0],initial[:,2],initial[:,1],initial[:,3],
        initial_kinetic_energy_ev=energy,initial_time_s=np.full(len(initial),2e-9),
        return_flight_times=True,energy_output=output)
    return plan,result,output[0]


def test_production_lorentz_matches_independent_potential_reference_with_overlap(monkeypatch):
    state=parked_state();state.step_mm=.02
    field=FrozenAnalyticField("lens",((.04,.451,.002),),CoordinateRegistration((2e-5,-1e-5,0.)),(400.,500.),.005)
    round_p=HighOrderRoundLensPotential(field,radial_order=3,axial_domain="gaussian_all_z")
    dipole=FrozenPosedMultipole("dipole","dipole",CoordinateRegistration(),((1,0,1e-5),(0,1,2e-5)),
        "uniform",.451,.002,(450.,452.),.005,1e-22)
    runtime=controlled(state,(PotentialContribution("lens",("lens",),round_p,"magnetic"),
        PotentialContribution("dipole",("dipole",),AnalyticPotentialDomain(AnalyticMagneticPotential(dipole)),"magnetic")))
    initial=np.array([[1e-5,-2e-5,.002,-.001],[-2e-5,1e-5,-.001,.003]])
    energy=np.array([200000.,300000.])
    plan,result,energies=execute(state,runtime,initial,monkeypatch,energy)
    reference=trace_potential_rays(runtime.snapshot,initial,(.45,.452),energy_ev=energy,maximum_step_m=1e-5,
        relative_tolerance=1e-12,position_tolerance_m=1e-15,slope_tolerance_rad=1e-13)
    actual=np.c_[result[1][-1],result[3][-1],result[2][-1],result[4][-1]]
    np.testing.assert_allclose(actual,reference.xy_slopes[-1],rtol=2e-8,atol=2e-12)
    np.testing.assert_allclose(energies,np.broadcast_to(energy,energies.shape),rtol=0,atol=0)
    assert np.all(np.diff(result[5],axis=0)>0)
    assert np.all(plan.cs_kick_m3==0) and np.all(plan.thin_power_m1==0)
    points=np.c_[initial[:,:2],(.451,.451)]
    np.testing.assert_allclose(_sampler(runtime.snapshot)(points)["B_t"],runtime.snapshot.sample(points)["B_t"],rtol=2e-14,atol=1e-18)


def test_finite_aperture_and_bore_stop_once_before_invalid_field_query(monkeypatch):
    state=parked_state();state.step_mm=.5
    aperture=PosedWaveAperture("iris",450.5,450.6,.01,CoordinateRegistration())
    bore=PosedWaveBore("tube",450.,452.,.03,CoordinateRegistration())
    runtime=controlled(state,pieces=(aperture,bore))
    initial=np.array([[0.,0.,0.,0.],[2e-5,0.,0.,0.],[4e-5,0.,0.,0.]])
    plan,result,energies=execute(state,runtime,initial,monkeypatch,step=.5)
    alive,stop,names=merge_unified_stops(result[0],np.ones(3,bool),np.full(3,np.nan),["","",""])
    np.testing.assert_array_equal(alive,[True,False,False])
    np.testing.assert_allclose(stop[1:],[450.5,450.],rtol=0,atol=1e-12)
    assert names==["","iris","tube"]
    assert np.isnan(energies[-1,1:]).all()


def test_actual_emission_reaches_requested_unified_section_without_source_substitution(tmp_path):
    from temsim.physics.particle_sections import run_particle_section
    state=parked_state();state.step_mm=.1
    state.electron_gun.emitter.ray_count=9
    result=run_particle_section(state,observation_stop_z_mm=451.)
    assert result.gun_trace.exit_bundle.x_m.size==9
    assert result.incident.z[-1]==451.
    assert result.metrics["unified_empirical_aberrations_applied"] is False
    assert result.metrics["unified_runtime_digest"]==result.incident_plan.unified_runtime.digest
    np.testing.assert_array_equal(result.incident.ray_weight,result.gun_trace.exit_bundle.weight)
    assert np.isfinite(result.incident.flight_time_s[-1,result.incident.alive]).all()
    from temsim.artifact_store import ArtifactStore
    from temsim.calculation_manifest import capture_calculation_manifest
    store=ArtifactStore(tmp_path,quota_bytes=100_000_000)
    manifest=capture_calculation_manifest(state)
    store.put_incident_simulation_seed(manifest,result)
    loaded=store.get_incident_simulation_seed(manifest)
    assert loaded.incident_plan.unified_runtime.digest==result.incident_plan.unified_runtime.digest
    np.testing.assert_array_equal(loaded.incident.x,result.incident.x)


def test_completed_unified_vacuum_result_does_not_run_legacy_conjugate_diagnostics(monkeypatch):
    import temsim.physics.simulation as simulation_module
    import temsim.physics.particle_sections as section_module
    import temsim.physics.unified_field_particle as particle_module
    import temsim.optics.direct_alignment as alignment_module
    import temsim.diagnostics as diagnostics_module
    state=parked_state();state.step_mm=.1
    state.electron_gun.emitter.ray_count=9
    state.sample.z_mm=451.
    state.sample.inserted=False
    state.sample.specimen_mode="vacuum"
    runtimes=[]
    def capture_runtime(current):
        runtime=replace(controlled(current),metadata={
            "source_contract":"Actual FEG emission into a declared vacuum test column"})
        runtimes.append(runtime)
        return runtime
    monkeypatch.setattr(particle_module,"get_unified_field_runtime",capture_runtime)
    monkeypatch.setattr(section_module,"section_limits",lambda _:(450.,452.))
    def forbidden(*args,**kwargs):
        raise AssertionError("Legacy conjugate diagnostics must not run for unified fields")
    monkeypatch.setattr(alignment_module,"diffraction_transfer",forbidden)
    monkeypatch.setattr(diagnostics_module,"optical_transfer_records",forbidden)
    result=simulation_module.run(state,resolved_layout=object())
    assert result.metrics["canonical_transfer_status"]=="unavailable"
    assert "j_img" not in result.metrics
    assert result.sample_to_analysis_transfer is None
    assert result.optical_transfers==()
    assert result.incident.z[-1]==451.
    assert result.branches["000"].z[-1]==452.
    assert result.metrics["unified_runtime_digest"]==runtimes[0].digest
    np.testing.assert_array_equal(result.incident.ray_weight,result.gun_trace.exit_bundle.weight)


def test_gui_equivalent_particle_section_keeps_actual_source_and_common_runtime():
    from temsim.simulation_pipeline import calculate_particle_section
    state=parked_state();state.step_mm=.1
    state.electron_gun.emitter.ray_count=9
    result=calculate_particle_section(state,target_z_mm=451.,acquire_stem=False)
    simulation=result.simulation
    assert simulation.incident.z[-1]==451.
    assert simulation.gun_trace.exit_bundle.x_m.size==9
    runtime=simulation.incident_plan.unified_runtime
    assert simulation.metrics["unified_runtime_digest"]==runtime.digest
    assert simulation.metrics["unified_field_digest"]==runtime.snapshot.numerical_digest
    np.testing.assert_array_equal(simulation.incident.ray_weight,simulation.gun_trace.exit_bundle.weight)


def test_public_full_simulation_rejects_unmigrated_material_before_gun_execution(monkeypatch):
    from temsim.physics.simulation import run
    import temsim.optics.electron_gun.source as source_module
    state=parked_state()
    state.sample.inserted=True
    state.sample.specimen_mode="atomic"
    def forbidden(*args,**kwargs):
        raise AssertionError("Material capability must be checked before gun execution")
    monkeypatch.setattr(source_module,"trace_source_to_exit",forbidden)
    with pytest.raises(ValueError,match="particle specimen transport"):
        run(state)


@pytest.mark.parametrize("source_scope",["actual_default_source","bounded_low_angle_emitter"])
def test_high_accuracy_ray_workflow_preserves_actual_source_and_native_domain(source_scope,record_property,monkeypatch):
    from temsim.simulation_pipeline import calculate
    import temsim.physics.intrinsic_aberrations as intrinsic_module
    state=parked_state();state.step_mm=1.
    state.electron_gun.emitter.ray_count=9
    def forbidden(*args,**kwargs):
        raise AssertionError("Unified optical transport must not capture legacy material Cc")
    monkeypatch.setattr(intrinsic_module,"specimen_chromatic_energy_change",forbidden)
    # A separately declared classical source fixture, propagated by the actual
    # FEG solver. The other case retains the complete default source unchanged.
    # No optical element or native radial radius is changed in either case.
    if source_scope=="bounded_low_angle_emitter":
        state.electron_gun.emitter.virtual_source_fwhm_nm=.5
        state.electron_gun.emitter.angular_rms_mrad=.01
        state.electron_gun.emitter.angular_cutoff_mrad=.03
    result=calculate(state,workflow="rays")
    simulation=result.simulation
    assert simulation.gun_trace.exit_bundle.x_m.size==9
    assert simulation.incident.z[-1]==state.sample.z_mm
    assert simulation.metrics["unified_empirical_aberrations_applied"] is False
    assert simulation.metrics["unified_equivalent_image_maps_applied"] is False
    assert simulation.metrics["unified_runtime_digest"]==simulation.incident_plan.unified_runtime.digest
    np.testing.assert_array_equal(simulation.incident.ray_weight,simulation.gun_trace.exit_bundle.weight)
    assert np.any(simulation.incident.alive)
    receiver_keys=tuple(plane.key for plane in state.recording_planes if plane.inserted)
    receiver_count=sum(np.count_nonzero(np.isin(branch.blocked_key,receiver_keys))
                       for branch in simulation.branches.values())
    assert receiver_count>0
    record_property("actual_source_fixture",json.dumps({
        "source_scope":source_scope,
        **{name:getattr(state.electron_gun.emitter,name) for name in
           ("virtual_source_fwhm_nm","angular_rms_mrad","angular_cutoff_mrad")},
        "sample_survivors":int(simulation.incident.alive.sum()),"recorded_electrons":int(receiver_count),
        "source_weight":float(simulation.gun_trace.exit_bundle.weight.sum()),
        "scope":"Actual FEG transport; static optical reference without material; explicit Gaussian remote-roundoff certificate"}))


def test_unqualified_near_field_rejects_without_absorption_or_legacy_fallback(monkeypatch):
    from temsim.physics.gaussian_remote_domain import POLICY
    field=FrozenAnalyticField("near-field",((.01,.451,.002),),CoordinateRegistration(),(400.,500.),1e-6)
    potential=HighOrderRoundLensPotential(field,radial_order=3,axial_domain="gaussian_all_z",remote_domain=POLICY)
    state=parked_state()
    runtime=controlled(state,(PotentialContribution("near-field",("near-field",),potential,"magnetic"),))
    with pytest.raises(ValueError,match="qualification"):
        execute(state,runtime,np.array([[2e-6,0.,0.,0.]]),monkeypatch)


def test_changed_state_rejects_stale_executor_runtime(monkeypatch):
    state=parked_state();runtime=controlled(state)
    next(item for item in state.lenses if item.key=="condenser_lens_2").percent+=.1
    with pytest.raises(ValueError,match="changed"):
        execute(state,runtime,np.zeros((1,4)),monkeypatch)


def test_actual_source_crosses_c2_c3_with_step_refinement_and_absolute_stop_accounting(record_property):
    from temsim.physics.particle_sections import run_particle_section
    results=[]
    for step in (.25,.125):
        state=parked_state();state.step_mm=step
        state.electron_gun.emitter.ray_count=9
        result=run_particle_section(state,observation_stop_z_mm=950.)
        results.append(result)
        assert result.incident.z[-1]==950.
        assert result.gun_trace.exit_bundle.x_m.size==9
        np.testing.assert_array_equal(result.incident.ray_weight,result.gun_trace.exit_bundle.weight)
        assert result.metrics["unified_field_digest"]
        branch=result.incident
        total=float(np.sum(branch.ray_weight))
        surviving=float(np.sum(branch.ray_weight[branch.alive]))
        stopped=float(np.sum(branch.ray_weight[~branch.alive]))
        assert total==pytest.approx(surviving+stopped,rel=2e-15)
    coarse,fine=results
    np.testing.assert_array_equal(coarse.incident.alive,fine.incident.alive)
    live=fine.incident.alive
    assert live.any(),"The actual source has no surviving electron at the requested plane"
    differences={}
    for name,tolerance in (("x",2e-8),("y",2e-8),("tx",2e-6),("ty",2e-6)):
        a,b=getattr(coarse.incident,name)[-1,live],getattr(fine.incident,name)[-1,live]
        differences[name]=float(np.max(np.abs(a-b)))
        np.testing.assert_allclose(a,b,rtol=0.,atol=tolerance)
    record_property("actual_source_cross_c2_c3",json.dumps({"start_mm":450.,"stop_mm":950.,
        "particle_count":9,"steps_mm":[.25,.125],"absolute_endpoint_differences":differences,
        "incident_weight":float(np.sum(fine.incident.ray_weight)),
        "surviving_weight":float(np.sum(fine.incident.ray_weight[live])),
        "runtime_digest":fine.metrics["unified_runtime_digest"],
        "scope":"actual physical FEG source; finite-field optical section; no specimen interactions"}))


def test_finite_material_cannot_fall_back_to_legacy_fields_in_unified_mode():
    from temsim.specimen.vector_field_transport import SpecimenFieldTransport
    with pytest.raises(ValueError,match="finite-volume specimen"):
        SpecimenFieldTransport(parked_state())


def test_unified_plan_archive_retains_complete_executable_bundle(monkeypatch):
    from temsim.particle_section_io import _pack_record_graph,_unpack_record_graph
    state=parked_state();runtime=controlled(state)
    plan,result,_=execute(state,runtime,np.zeros((1,4)),monkeypatch)
    arrays={}
    graph=_pack_record_graph(plan,arrays)
    restored=_unpack_record_graph(graph,arrays,maximum_unpacked_bytes=32*1024**2)
    assert restored.unified_runtime.digest==runtime.digest
    assert restored.physics_contract["unified_capture"]["field"]==runtime.snapshot.numerical_digest
    result2=execute_propagation_plan(state,restored,[0.],[0.],[0.],[0.],initial_kinetic_energy_ev=300000.)
    for a,b in zip(result[1:5],result2[1:5]):np.testing.assert_array_equal(a,b)


def test_oversized_numerical_history_rejected_before_allocation(monkeypatch):
    import temsim.physics.unified_field_particle as module
    state=parked_state();runtime=controlled(state)
    monkeypatch.setattr(module,"MAX_UNIFIED_HISTORY_BYTES",1)
    with pytest.raises(ValueError,match="numerical budget"):
        execute(state,runtime,np.zeros((1,4)),monkeypatch)


def test_uniform_electric_energy_and_clock_match_independent_relativistic_solution(monkeypatch):
    from scipy.constants import e,m_e,c
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    state=parked_state();state.step_mm=.02
    radius,z=np.array([0.,.02]),np.array([.449,.453])
    gradient=2e6
    phi=np.broadcast_to(200000.+gradient*(z-.45),(2,2)).copy()
    potential=ElectricPotentialSnapshot.from_provider(PlanarGunField({},radius,z,phi))
    runtime=controlled(state,(PotentialContribution("electric",("electric",),potential,"electric"),))
    initial=np.array([[1e-5,2e-5,.02,-.03],[-2e-5,1e-5,-.01,.02]])
    energy=np.array([100000.,300000.])
    _,result,energies=execute(state,runtime,initial,monkeypatch,energy=energy)
    reference=trace_potential_rays(runtime.snapshot,initial,(.45,.452),energy_ev=energy,maximum_step_m=1e-5)
    actual=np.c_[result[1][-1],result[3][-1],result[2][-1],result[4][-1]]
    np.testing.assert_allclose(actual,reference.xy_slopes[-1],rtol=2e-9,atol=2e-13)
    final_energy=energy+gradient*.002
    np.testing.assert_allclose(energies[-1],final_energy,rtol=0,atol=2e-10)
    def momentum(kev):
        work=kev*e
        return np.sqrt(work*(work+2*m_e*c*c))/c
    p0,p1=momentum(energy),momentum(final_energy)
    pz0=p0/np.sqrt(1+np.sum(initial[:,2:]**2,axis=1))
    pt2=np.sum((initial[:,2:]*pz0[:,None])**2,axis=1)
    duration=(np.sqrt(p1*p1-pt2)-pz0)/(e*gradient)
    np.testing.assert_allclose(result[5][-1]-2e-9,duration,rtol=2e-9,atol=1e-20)


def test_recording_stage_does_not_apply_captured_apertures_again(monkeypatch):
    from temsim.physics.recording_clipping import clip_recording_planes
    from temsim.physics import aperture_clipping
    state=parked_state()
    for plane in state.recording_planes:plane.inserted=False
    aperture=next(a for a in state.apertures if a.z_mm>=state.sample.z_mm)
    aperture.enabled=True
    aperture.radius_mm=0.
    def reject(*args,**kwargs):
        assert not args[1],"Unified receiver stage must not repeat posed aperture action"
        return args[-3:]
    monkeypatch.setattr(aperture_clipping,"clip_posed_apertures",reject)
    z=np.array([aperture.z_mm-1,aperture.z_mm+1])
    alive,_,_=clip_recording_planes(state,z,np.zeros((2,1)),np.zeros((2,1)),[True],[np.nan],[""])
    assert alive[0]
