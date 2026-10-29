"""Low-cost guards for the output-only injection driver; no physical tracing."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

PATH = Path(__file__).with_name("run_column_energy_diagnostic.py")
spec=importlib.util.spec_from_file_location("controlled_column_energy_diagnostic",PATH)
driver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver;spec.loader.exec_module(driver)


def synthetic_donor():
    n=4;z=np.array([450.,451.,452.]);ids=np.arange(n,dtype=np.int64);w=np.full(n,.25)
    k=300000.+np.array([.1,.2,.3,.4]);x=np.arange(n)*1e-10
    a=dict(z_mm=z,integration_z_mm=z,source_id=ids,weights=w,emission_ray_id=ids.copy(),
        emission_position_m=np.column_stack([x,x,np.zeros(n)]),emission_direction=np.tile([0.,0.,1.],(n,1)),
        emission_normal=np.tile([0.,0.,1.],(n,1)),source_azimuth_rad=np.zeros(n),
        gun_exit_alive=np.ones(n,bool),gun_blocked_z_mm=np.full(n,np.nan),gun_blocked_key=np.full(n,""),
        source_energy_offset_ev=k-300000.,gun_exit_kinetic_energy_ev=k.copy(),
        blocked_z_mm=np.full(n,np.nan),blocked_key=np.full(n,""),terminal_alive=np.ones(n,bool),
        alive=np.ones((3,n),bool))
    for key in driver.INITIAL:
        base=np.full(n,1e-9) if key=="flight_time_s" else x.copy()
        a["gun_exit_"+key]=base
        a[key]=np.tile(base,(3,1))
    a["kinetic_energy_ev"]=np.tile(k,(3,1))
    identity=dict(input_digest="input",runtime_digest="runtime",potential_digest="potential",geometry_digest="geometry",plan_signature="plan")
    state=SimpleNamespace(beam_voltage_kv=300.,sample=SimpleNamespace(upper_surface_z_mm=451.),
        electron_gun=SimpleNamespace(nominal_exit_energy_ev=300000.,exit_plane_z_mm=450.))
    snapshot=SimpleNamespace(digest="input",to_dict=lambda:dict(implementation="synthetic",digest="input"))
    report=dict(parameters=dict(batch_size=2),receipt=dict(source_current_a=1e-5))
    return driver.Donor(Path("synthetic"),state,snapshot,a,report,z.copy(),identity,{})


@pytest.fixture
def donor(monkeypatch):
    import temsim.instrument_snapshot
    import temsim.physics.particle_sections
    d=synthetic_donor()
    monkeypatch.setattr(temsim.instrument_snapshot,"capture_instrument_snapshot",lambda state:d.snapshot)
    monkeypatch.setattr(temsim.physics.particle_sections,"_events",lambda state:())
    return d


def fake_executor(donor,calls):
    def trace(state,name,start,stop,initial,energy,weights,identity,survival,events,previous,boundary,**kwargs):
        ids=identity[0];calls.append(dict(ids=ids.copy(),initial=deepcopy(initial),energy=energy.copy(),
            weights=weights.copy(),survival=deepcopy(survival),previous=previous,**kwargs))
        cp={key:donor.arrays[key][:,ids].copy() for key in driver.PHASE}
        cp["kinetic_energy_ev"]=np.tile(kwargs['initial_kinetic_energy_ev'],(3,1))
        checkpoint=SimpleNamespace(**cp,z_mm=donor.arrays['z_mm'].copy(),
            unified_state=SimpleNamespace(active=donor.arrays['alive'][:,ids].copy(),
                reference_energy_ev=kwargs['initial_kinetic_energy_ev'].copy(),reference_phi_v=np.zeros(len(ids))))
        branch=SimpleNamespace(source_ray_id=ids.copy(),ray_weight=weights.copy(),blocked_z=np.full(len(ids),np.nan),
            blocked_key=[""]*len(ids),alive=np.ones(len(ids),bool))
        plan=SimpleNamespace(z_mm=donor.arrays['integration_z_mm'].copy())
        return SimpleNamespace(checkpoints=checkpoint,branch=branch,plan=plan),450.,False
    return trace


def audit(donor):
    return lambda *args:dict(identities=donor.field_identity,c2_aperture=dict(local_start_z_mm=450.9,local_end_z_mm=451.1,
        origin_global_m=[0.,0.,0.],diameter_mm=.1))


def test_constant_intervention_preserves_all_donor_arrays(donor):
    before=deepcopy(donor.arrays)
    k,e,manifest=driver.prepare_energy(donor,'constant')
    np.testing.assert_array_equal(k,np.full(4,300000.3))
    np.testing.assert_array_equal(e,k-300000.)
    assert manifest['mode']=='constant'
    assert manifest['original_kinetic_sha256']!=manifest['injected_kinetic_sha256']
    for key in before:np.testing.assert_array_equal(before[key],donor.arrays[key])


@pytest.mark.parametrize('energy',[0.,-1.,np.nan,np.inf,True])
def test_invalid_constant_rejected(donor,energy):
    with pytest.raises(ValueError):driver.prepare_energy(donor,'constant',constant_ev=energy)


def test_original_mode_is_bitwise_energy_and_offset_replay(donor):
    k,e,_=driver.prepare_energy(donor,'original')
    np.testing.assert_array_equal(k,donor.arrays['gun_exit_kinetic_energy_ev'])
    np.testing.assert_array_equal(e,donor.arrays['source_energy_offset_ev'])
    assert k is not donor.arrays['gun_exit_kinetic_energy_ev']


def test_crossed_energy_allows_changed_exit_phase_but_pins_original_source(donor):
    energy=deepcopy(donor);energy.arrays['gun_exit_x_m']*=7
    energy.arrays['gun_exit_kinetic_energy_ev']+=.01;energy.arrays['source_energy_offset_ev']+=.01
    k,_,_=driver.prepare_energy(donor,'donor-energy',energy_donor=energy)
    np.testing.assert_array_equal(k,energy.arrays['gun_exit_kinetic_energy_ev'])
    assert not np.array_equal(energy.arrays['gun_exit_x_m'],donor.arrays['gun_exit_x_m'])


@pytest.mark.parametrize('key',['source_id','weights','emission_position_m','emission_direction','gun_exit_alive'])
def test_crossed_source_mismatch_rejects_without_filtering(donor,key):
    other=deepcopy(donor)
    if key=='gun_exit_alive':other.arrays[key][0]=False
    else:other.arrays[key].flat[0]+=1
    with pytest.raises(ValueError):driver.prepare_energy(donor,'donor-energy',energy_donor=other)


def test_crossed_field_change_rejected(donor):
    other=deepcopy(donor);other.field_identity['potential_digest']='changed'
    with pytest.raises(ValueError):driver.prepare_energy(donor,'donor-energy',energy_donor=other)


@pytest.mark.parametrize('fault',['duplicate_id','bad_weight','revival','invalid_energy','invalid_clock','missing_plane'])
def test_invalid_saved_evidence_rejected(donor,fault):
    a=donor.arrays
    if fault=='duplicate_id':a['source_id'][0]=1
    elif fault=='bad_weight':a['weights'][0]=-1
    elif fault=='revival':a['alive'][1,0]=False
    elif fault=='invalid_energy':a['gun_exit_kinetic_energy_ev'][0]=0
    elif fault=='invalid_clock':a['gun_exit_flight_time_s'][0]=-1
    else:donor.planes=np.array([450.,451.5,452.])
    with pytest.raises(ValueError):driver.validate_arrays(a,donor.planes)


def test_mock_original_replay_proves_full_population_mesh_and_no_reuse(donor):
    driver.validate_arrays(donor.arrays,donor.planes)
    k,e,_=driver.prepare_energy(donor,'original');calls=[]
    arrays,batches=driver.execute_injection(donor,k,e,lambda:False,trace_segment=fake_executor(donor,calls),audit_plan=audit(donor))
    assert len(calls)==2 and len(arrays['source_id'])==4
    assert all(c['checkpoint_ray_count']==4 and c['initial_kicks'] and c['previous'] is None for c in calls)
    assert driver.compare_replay(donor,arrays,0,4)['status']=='EXACT_REPLAY'
    assert [b['source_slice'] for b in batches]==[[0,2],[2,4]]


def test_mock_constant_changes_only_injected_energy_arguments(donor):
    k,e,_=driver.prepare_energy(donor,'constant');calls=[]
    result,_=driver.execute_injection(donor,k,e,lambda:False,trace_segment=fake_executor(donor,calls),audit_plan=audit(donor))
    np.testing.assert_array_equal(result['kinetic_energy_ev'][0],k)
    for call in calls:
        for key,value in zip(driver.INITIAL,call['initial']):
            np.testing.assert_array_equal(value,donor.arrays['gun_exit_'+key][call['ids']])
    assert driver.compare_replay(donor,result,0,4)['status']=='REPLAY_MISMATCH'


def test_wrong_field_identity_rejected(donor):
    k,e,_=driver.prepare_energy(donor,'original')
    with pytest.raises(ValueError,match='field'):
        driver.execute_injection(donor,k,e,lambda:False,trace_segment=fake_executor(donor,[]),
            audit_plan=lambda *args:dict(identities={**donor.field_identity,'potential_digest':'changed'}))


def test_cancellation_before_trace(donor):
    k,e,_=driver.prepare_energy(donor,'original');calls=[]
    with pytest.raises(InterruptedError):
        driver.execute_injection(donor,k,e,lambda:True,trace_segment=fake_executor(donor,calls),audit_plan=audit(donor))
    assert calls==[]


def test_partial_replay_has_no_full_population_statistics(tmp_path,donor,monkeypatch):
    import temsim.cpu_resources
    @contextmanager
    def cpu(*args,**kwargs):yield SimpleNamespace(to_dict=lambda:dict(threads=4))
    monkeypatch.setattr(temsim.cpu_resources,'numerical_job',cpu)
    monkeypatch.setattr(driver,'load_donor',lambda p:donor)
    original=driver.execute_injection
    monkeypatch.setattr(driver,'execute_injection',lambda d,k,e,cancelled,**kw:
        original(d,k,e,cancelled,trace_segment=fake_executor(donor,[]),audit_plan=audit(donor),**kw))
    result=driver.run('synthetic',tmp_path/'new',mode='original',replay_batches=1)
    assert result['status']=='COMPLETE' and not result['full_original_population']
    assert result['processed_source_points']==2 and 'observations' not in result
    assert result['batches'][0]['replay']['status']=='EXACT_REPLAY'
    assert result['qualification']=='CONTROLLED_DIAGNOSTIC_NOT_QUALIFIED'
    with pytest.raises(FileExistsError):driver.run('synthetic',tmp_path/'new',mode='original',replay_batches=1)


def test_donor_load_failure_leaves_terminal_receipt(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'load_donor',lambda p:(_ for _ in ()).throw(ValueError('checksum mismatch')))
    with pytest.raises(ValueError):driver.run('bad',tmp_path/'new')
    report=json.loads((tmp_path/'new/report.json').read_text())
    assert report['status']=='FAILED' and 'checksum mismatch' in report['error']


def test_partial_intervention_forbidden_before_output(tmp_path):
    with pytest.raises(ValueError):driver.run('unused',tmp_path/'new',mode='constant',replay_batches=1)
    assert not (tmp_path/'new').exists()


def test_initial_energy_checks_exact_potential_work_roundoff():
    k=np.array([np.nextafter(300000.,np.inf)]);phi=np.array([300000.]);live=np.array([True])
    saved=k+phi-phi+np.zeros(1)
    assert not np.array_equal(saved,k)
    driver.verify_initial_energy(saved,k,phi,live)
    with pytest.raises(ValueError,match='potential-work'):
        driver.verify_initial_energy(np.nextafter(saved,np.inf),k,phi,live)


def test_changed_initial_phase_remains_fatal(donor):
    donor.arrays['tx_rad'][0,0]=np.nextafter(donor.arrays['tx_rad'][0,0],np.inf)
    with pytest.raises(ValueError,match='original gun exit'):
        driver.validate_arrays(donor.arrays,donor.planes)
