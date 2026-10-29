"""Pure-array and mocked failure tests for the bounded native response driver."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parent))
import run_native_linear_response as driver


def source():
    a=dict(source_id=np.array([0,1,306,4]),weights=np.array([.1,.2,.3,.4]),gun_exit_alive=np.ones(4,bool),
           gun_exit_flight_time_s=np.array([1.,2.,3.,4.])*1e-9)
    for i,key in enumerate(('x_m','y_m','tx_rad','ty_rad')):
        a['gun_exit_'+key]=(np.arange(4.)-1.5)*(1e-9 if i<2 else 1e-8)
    return a


def raw_fixture():
    a=source();phase,reference,clock,probes=driver.calibration_inputs(a)
    z=np.array([450.,451.,452.]);matrix=np.array([[2.,.1,.3,.02],[.05,1.7,.03,.2],[0.,0.,.9,.1],[0.,0.,-.1,1.1]])
    value=np.tile((probes@matrix.T)[None,:,:],(3,1,1));value[0]=probes
    raw={k:value[:,:,i].copy() for i,k in enumerate(('x_m','y_m','tx_rad','ty_rad'))}
    raw.update(z_mm=z,active=np.ones((3,17),bool),terminal_alive=np.ones(17,bool),blocked_z_mm=np.full(17,np.nan),
        blocked_key=np.full(17,''),flight_time_s=np.full((3,17),clock),kinetic_energy_ev=np.full((3,17),driver.KINETIC_EV),
        reference_energy_ev=np.full(17,driver.KINETIC_EV),reference_phi_v=np.zeros(17),induction_work_ev=np.zeros((3,17)))
    donor=SimpleNamespace(arrays={**a,'z_mm':z},planes=z,state=SimpleNamespace(sample=SimpleNamespace(upper_surface_z_mm=451.)))
    return donor,phase,reference,clock,probes,raw,matrix


def test_reference_preserves_actual_weighted_mean_and_17_separate_probes():
    a=source();phase,reference,clock,probes=driver.calibration_inputs(a)
    np.testing.assert_array_equal(reference,np.average(phase,axis=0,weights=a['weights']))
    assert probes.shape==(17,4) and len(np.unique(probes,axis=0))==17
    assert clock==np.average(a['gun_exit_flight_time_s'],weights=a['weights'])
    assert 'weights' not in driver.calibration_inputs.__code__.co_varnames


@pytest.mark.parametrize('steps',[np.array([0.,1.,1.,1.]),np.array([np.nan,1.,1.,1.]),np.ones(3)])
def test_bad_steps_fail(steps):
    with pytest.raises(ValueError):driver.calibration_inputs(source(),steps)


def test_exact_affine_map_recovered_and_source_prediction_separate():
    donor,phase,reference,clock,probes,raw,matrix=raw_fixture()
    driver.validate_probes(raw,donor,probes,clock)
    jac,pred=driver.derive_prediction(raw,probes,phase,reference)
    for j,p in zip(jac,pred):
        np.testing.assert_allclose(j[1],matrix,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(p[1],phase@matrix.T,rtol=1e-13,atol=1e-20)
    assert raw['x_m'].shape[1]==17 and pred[0].shape[1]==4


@pytest.mark.parametrize('fault',['blocked','induction','wrong_k','wrong_entrance','nonfinite'])
def test_invalid_calibration_fails_closed(fault):
    donor,phase,reference,clock,probes,raw,matrix=raw_fixture()
    if fault=='blocked':raw['active'][1,3]=False
    elif fault=='induction':raw['induction_work_ev'][1,0]=1e-20
    elif fault=='wrong_k':raw['reference_energy_ev'][0]+=1e-6
    elif fault=='wrong_entrance':raw['x_m'][0,0]+=1e-20
    else:raw['x_m'][1,0]=np.nan
    with pytest.raises(ValueError):driver.validate_probes(raw,donor,probes,clock)


def test_affine_prediction_comparison_passes_and_has_no_probe_current():
    donor,phase,reference,clock,probes,raw,matrix=raw_fixture()
    _,pred=driver.derive_prediction(raw,probes,phase,reference)
    actual={k:pred[0][:,:,i] for i,k in enumerate(('x_m','y_m','tx_rad','ty_rad'))}
    actual['alive']=np.ones((3,4),bool)
    result,arrays=driver.comparison(donor,actual,probes,raw,phase,reference)
    assert result['stability']['passed']
    assert result['nonlinear_remainder']['surface_maximum_xy_nm']<1e-10
    assert all('current_a' not in row for p in result['profiles'].values() for row in p['planes'])


def test_step_sensitive_prediction_is_explicitly_unresolved():
    donor,phase,reference,clock,probes,raw,matrix=raw_fixture()
    # Deliberate inconsistency in the full-step response, far above declared tolerance.
    raw['x_m'][1:,1]+=1e-6
    actual={k:np.zeros((3,4)) for k in ('x_m','y_m','tx_rad','ty_rad')};actual['alive']=np.ones((3,4),bool)
    result,_=driver.comparison(donor,actual,probes,raw,phase,reference)
    assert not result['stability']['passed']


def test_failure_leaves_terminal_receipt_and_refuses_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'load_donor',lambda path:(_ for _ in ()).throw(ValueError('Bad saved checksum')))
    with pytest.raises(ValueError):driver.run('none','none',tmp_path/'new')
    r=json.loads((tmp_path/'new/report.json').read_text())
    assert r['status']=='FAILED' and not r['physical_current_assigned_to_calibration_probes']
    with pytest.raises(FileExistsError):driver.run('none','none',tmp_path/'new')
