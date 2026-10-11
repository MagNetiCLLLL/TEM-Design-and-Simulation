"""Focused receiving-aperture window, phase and publication boundary checks."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.optics.model import Aperture
from temsim.physics.column_acceptance_window import (
    AcceptanceWindowNumerics, AcceptanceWindowUnavailable, crop_acceptance_window,
    find_acceptance_span, sample_aperture_plane, stationary_input_radius,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics


def field(count=128, pitch=.025, *, origin=(.018, -.013), q=3.):
    rotation=np.array(((np.cos(.2),-np.sin(.2)),(np.sin(.2),np.cos(.2))))
    plane=PlaneWave(np.ones((count,count),complex),pitch*rotation,np.asarray(origin),
                    np.eye(2)*q,np.array((.12,-.03)))
    xy=plane.coordinates_m()
    amplitude=np.exp(-np.sum(xy*xy,axis=0)/2+.17j*xy[0])*(1+.1*xy[1])*pitch
    return replace(plane,amplitude=amplitude)


def test_crop_preserves_full_complex_carrier_on_shifted_rotated_lattice():
    wave=field(); wavelength=.2; radius=1.
    policy=AcceptanceWindowNumerics(minimum_taper_cells=8)
    result, record=crop_acceptance_window(wave,radius_m=radius,wavelength_m=wavelength,
        numerics=policy,grid_numerics=WaveGridNumerics())
    assert record['preinterpolation_factor']==1
    xy=result.coordinates_m();rho=np.sqrt(np.sum(xy*xy,axis=0))
    t=np.clip((rho/radius-.7)/.3,0,1);window=1-t**3*(10+t*(-15+6*t))
    original_envelope=np.exp(-np.sum(xy*xy,axis=0)/2+.17j*xy[0])*(1+.1*xy[1])*.025
    delta=xy-wave.origin_m[:,None,None]
    action=.5*np.einsum('iyx,ij,jyx->yx',delta,wave.curvature_m1,delta)+np.einsum('i,iyx->yx',wave.tilt_rad,delta)
    local=xy-result.origin_m[:,None,None]
    result_action=.5*np.einsum('iyx,ij,jyx->yx',local,result.curvature_m1,local)+np.einsum('i,iyx->yx',result.tilt_rad,local)
    np.testing.assert_allclose(result.amplitude*np.exp(2j*np.pi*result_action/wavelength),
        original_envelope*window*np.exp(2j*np.pi*action/wavelength),atol=2e-15,rtol=2e-13)
    assert result.probability < wave.probability
    assert 'NOT physical absorption' in record['scope']


def test_aperture_reconstruction_preserves_absolute_density_and_phase_cpu():
    wave=field(pitch=.01,q=1.2)
    policy=AcceptanceWindowNumerics(aperture_radius_cells=8,output_halfwidth_radii=2.,
        maximum_relative_complex_change=.01)
    result,record=sample_aperture_plane(wave,radius_m=.2,wavelength_m=.1,curvature=np.eye(2)*.8,
        numerics=policy,grid_numerics=WaveGridNumerics())
    xy=result.coordinates_m();core=np.sum(xy*xy,axis=0)<=.2**2
    delta=xy-wave.origin_m[:,None,None]
    phase=.5*np.einsum('iyx,ij,jyx->yx',delta,wave.curvature_m1,delta)+np.einsum('i,iyx->yx',wave.tilt_rad,delta)-.4*np.sum(xy*xy,axis=0)
    expected=np.exp(-np.sum(xy*xy,axis=0)/2+.17j*xy[0])*(1+.1*xy[1])*.025*np.exp(2j*np.pi*phase/.1)
    np.testing.assert_allclose(result.amplitude[core],expected[core],atol=1e-10,rtol=1e-8)
    assert record['phase_fit'] is False
    assert record['complex_interpolation_relative_change']<1e-8


def test_aperture_refinement_uses_global_gpu_policy():
    from temsim.physics.compute_backend import cupy_module
    cp=cupy_module()
    if cp is None:
        pytest.skip('CuPy unavailable')
    try:
        if cp.cuda.runtime.getDeviceCount()<1:pytest.skip('CUDA device unavailable')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA runtime unavailable')
    wave=field(pitch=.01,q=1.2)
    policy=AcceptanceWindowNumerics(aperture_radius_cells=32,output_halfwidth_radii=2.)
    cpu, _=sample_aperture_plane(wave,radius_m=.2,wavelength_m=.1,curvature=np.eye(2)*.8,
        numerics=policy,grid_numerics=WaveGridNumerics())
    gpu, record=sample_aperture_plane(wave,radius_m=.2,wavelength_m=.1,curvature=np.eye(2)*.8,
        numerics=policy,grid_numerics=WaveGridNumerics(compute_backend='Require GPU'))
    np.testing.assert_allclose(gpu.amplitude,cpu.amplitude,atol=2e-14,rtol=1e-11)
    assert record['compute_backend']['compute_backend']=='cupy'


def test_stationary_gate_includes_original_envelope_bandwidth_and_all_roots():
    wave=replace(field(pitch=1e-4),curvature_m1=np.zeros((2,2)),tilt_rad=np.zeros(2),origin_m=np.zeros(2))
    matrix=np.eye(4);matrix[:2,2:]=np.eye(2)*.1
    path=SimpleNamespace(matrix=matrix,offset=np.zeros(4))
    root,record=stationary_input_radius(wave,[path],-1.,50e-6,2e-12)
    assert root>0 and record['maximum_envelope_displacement_m']>0
    assert root>50e-6-1e-12
    with pytest.raises(AcceptanceWindowUnavailable,match='bandwidth'):
        stationary_input_radius(wave,[path],-1.,50e-6,2e-6)
    with pytest.raises(AcceptanceWindowUnavailable,match='branches'):
        stationary_input_radius(wave,[path],1.,50e-6,2e-12)


def prepared_plan(aperture=None):
    n=4; zero=np.zeros(n-1)
    plan=SimpleNamespace(step_m=np.full(n-1,.001),z_mm=np.arange(n,dtype=float),
        midpoint_magnetic_t=zero,midpoint_sx_m2=zero,midpoint_sy_m2=zero,
        midpoint_sxy_m2=zero,midpoint_hex_normal_m3=zero,midpoint_hex_skew_m3=zero,
        kick_x_rad=np.zeros(n),kick_y_rad=np.zeros(n),cs_kick_m3=np.array((0.,-1.,0.,0.)),
        signature='test',posed_spherical_kicks=())
    ap=aperture or Aperture('Test','physical',3.,.05,enabled=True)
    return plan,np.full(n,2.5),{3:[ap]},[]


def test_span_consumes_next_real_circular_aperture_and_identity_changes():
    prepared=prepared_plan();span=find_acceptance_span(prepared)
    assert span['kick_z_mm']==1. and span['stop_z_mm']==3.
    assert span['prepared'][2][3][0] is prepared[2][3][0]
    other=find_acceptance_span(prepared_plan(Aperture('Test','physical',3.,.06,enabled=True)))
    assert span['identity']!=other['identity']
    detector=SimpleNamespace(radius_mm=.05,key='annular',offset_x_mm=0.,offset_y_mm=0.)
    assert find_acceptance_span(prepared_plan(detector)) is None
    plan,radii,stops,_=prepared
    assert find_acceptance_span((plan,radii,stops,[dict(z_mm=2.,dynamic=True)])) is None


def test_cancel_occurs_before_crop_or_aperture_publication():
    with pytest.raises(InterruptedError):
        crop_acceptance_window(field(),radius_m=1.,wavelength_m=.2,numerics=AcceptanceWindowNumerics(),
            grid_numerics=WaveGridNumerics(),cancelled=lambda:True)
    with pytest.raises(InterruptedError):
        sample_aperture_plane(field(),radius_m=.2,wavelength_m=.2,curvature=np.eye(2),
            numerics=AcceptanceWindowNumerics(),grid_numerics=WaveGridNumerics(),cancelled=lambda:True)


def test_request_identity_covers_enabled_policy_and_json_guards():
    from temsim.physics.tip_wave_pipeline import TipWaveRequest
    request=TipWaveRequest()
    assert request.acceptance_window.enabled
    enabled=request
    disabled=replace(request,acceptance_window=replace(request.acceptance_window,enabled=False))
    assert json_digest(asdict(request))!=json_digest(asdict(disabled))
    changed=replace(enabled,acceptance_window=replace(enabled.acceptance_window,aperture_radius_cells=256))
    assert json_digest(asdict(changed))!=json_digest(asdict(enabled))
    restored=AcceptanceWindowNumerics(**{**asdict(enabled.acceptance_window),"guard_scales":[.5,1.,2.]})
    restored.validate()
    assert restored == enabled.acceptance_window


def test_common_output_grid_rejects_unresolved_residual_phase():
    with pytest.raises(ValueError,match='residual carrier'):
        sample_aperture_plane(field(pitch=.01,q=100.),radius_m=.2,wavelength_m=.01,
            curvature=np.zeros((2,2)),numerics=AcceptanceWindowNumerics(aperture_radius_cells=8),
            grid_numerics=WaveGridNumerics())


def test_changing_energy_span_is_ineligible_before_any_output(monkeypatch):
    from temsim.physics.column_acceptance_window import execute_acceptance_span
    span=find_acceptance_span(prepared_plan())
    monkeypatch.setattr('temsim.physics.column_wave._electric_coefficients',
        lambda _: (np.array((0.,0.,.1,.1)),np.zeros(3),None,None))
    with pytest.raises(AcceptanceWindowUnavailable,match='constant axial kinetic energy'):
        execute_acceptance_span(None,SimpleNamespace(plane_z_mm=0.),span,
            numerics=AcceptanceWindowNumerics(enabled=True),grid_numerics=WaveGridNumerics(),
            maximum_step_mm=.1)


def test_private_hook_observes_each_cs_once_without_changing_default_transport():
    from test_tip_wave_pipeline import _quiet_prepared_column
    from test_wave_detector_readout import checkpoint
    from temsim.optics.column import default_state
    from temsim.physics.column_wave import _propagate_column
    state=default_state(); source=checkpoint()
    prepared=_quiet_prepared_column(state,source.plane_z_mm,source.plane_z_mm+.02,.01)
    prepared[0].cs_kick_m3[1:]=(.1,.2)
    calls=[]
    def hook(wave,**identity):
        calls.append(identity)
        return wave
    expected=_propagate_column(state,source,source.plane_z_mm+.02,_prepared=prepared)
    actual=_propagate_column(state,source,source.plane_z_mm+.02,_prepared=prepared,_before_spherical=hook)
    assert [x['spherical_m3'] for x in calls]==[.1,.2]
    assert [x['mode_id'] for x in calls]==[source.beam.modes[0].mode_id]*2
    np.testing.assert_allclose(actual.beam.modes[0].plane.amplitude,expected.beam.modes[0].plane.amplitude,
                               atol=1e-13,rtol=1e-11)
    assert actual.beam.total_weight==pytest.approx(expected.beam.total_weight,rel=1e-13)
