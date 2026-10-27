"""Bounded physical tests of first-crossing gun observations."""
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.optics.electron_gun.observations import ExactGunPlaneObserver
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
    velocity_from_momentum_m_per_s,
)


class Fields:
    def __init__(self, gradient=0.):
        self.gradient = gradient

    def field_at_global_positions_t(self, position):
        return np.zeros_like(position)

    def field_at_global_positions_v_per_m(self, position):
        result = np.zeros_like(position)
        result[..., 2] = -self.gradient
        return result

    def potential_v_at_global_positions(self, position):
        return self.gradient*np.asarray(position)[..., 2]


def test_exact_drift_waist_keeps_direction_energy_clock_and_input_state():
    phi = np.arange(8)*2*np.pi/8
    slopes = .001*np.c_[np.cos(phi),np.sin(phi)]
    direction = np.c_[slopes, np.ones(8)]
    momentum = momentum_from_kinetic_energy_ev(np.full(8, 100000.), direction)
    position = np.c_[-.004*slopes, np.zeros(8)]
    phase = RelativisticPhaseSpace(position, momentum)
    velocity = velocity_from_momentum_m_per_s(momentum)
    duration = .006/velocity[0, 2]
    observer = ExactGunPlaneObserver([0., 3., 4., 5.], 6.)
    observer.initialize(phase)
    observer.observe(position, momentum, 0., position+duration*velocity, duration,
        np.ones(8,bool), np.full(8,np.nan), Fields(), Fields())
    result = observer.result()
    np.testing.assert_allclose(result.x_m, (result.z_mm[:,None]*.001-.004)*slopes[:,0], atol=1e-18)
    np.testing.assert_allclose(result.y_m, (result.z_mm[:,None]*.001-.004)*slopes[:,1], atol=1e-18)
    np.testing.assert_allclose(result.tx_rad, np.broadcast_to(slopes[:,0],(4,8)), atol=1e-16)
    np.testing.assert_allclose(result.kinetic_energy_ev, 100000., rtol=1e-14)
    np.testing.assert_allclose(result.flight_time_s, np.broadcast_to(result.z_mm[:,None]*.001/velocity[:,2],(4,8)), atol=1e-20)
    assert result.alive.all() and result.reached.all()
    assert abs(result.plane_residual_m).max() < 1e-15
    np.testing.assert_array_equal(phase.position_m, position)


def test_accelerating_exact_crossings_preserve_electric_work_without_history_interpolation():
    from temsim.physics.static_energy_lorentz import static_energy_step
    fields = Fields(1e6)
    momentum = momentum_from_kinetic_energy_ev(np.array([.3]), np.array([[.001,0.,1.]]))
    start = RelativisticPhaseSpace(np.zeros((1,3)), momentum)
    duration = 2e-12
    half = static_energy_step(start, duration/2, fields, fields)
    end = static_energy_step(half, duration/2, fields, fields)
    planes = np.array([1e-7, 1e-5, 1e-4])  # mm, all before the accepted endpoint
    assert end.position_m[0,2] > planes[-1]*.001
    observer = ExactGunPlaneObserver(planes, 1.)
    observer.initialize(start)
    observer.observe(start.position_m, start.momentum_kg_m_per_s, 0., end.position_m,
        duration, np.array([True]), np.array([np.nan]), fields, fields)
    result = observer.result()
    np.testing.assert_allclose(result.kinetic_energy_ev[:,0], .3+planes*.001*1e6, atol=2e-10)
    assert np.all(np.diff(result.flight_time_s[:,0])>0)
    assert result.alive.all() and abs(result.plane_residual_m).max()<1e-13


def test_first_crossing_stop_order_and_missing_ray_do_not_resurrect_particles():
    momentum = momentum_from_kinetic_energy_ev(np.full(2,100000.), np.array([[0.,0.,1.],[0.,0.,1.]]))
    position = np.zeros((2,3)); velocity = velocity_from_momentum_m_per_s(momentum)
    duration = .006/velocity[0,2]
    observer = ExactGunPlaneObserver([1.,3.,5.],6.)
    observer.initialize(RelativisticPhaseSpace(position,momentum))
    observer.observe(position,momentum,0.,position+duration*velocity,duration,
        np.array([True,False]),np.array([.004/velocity[0,2],0.]),Fields(),Fields())
    result = observer.result()
    np.testing.assert_array_equal(result.alive, [[True,False],[True,False],[False,False]])
    saved = result.x_m.copy()
    observer.observe(position,momentum,0.,position+duration*velocity,duration,
        np.array([True,False]),np.array([.004/velocity[0,2],0.]),Fields(),Fields())
    np.testing.assert_array_equal(observer.result().x_m, saved)
    assert np.isnan(result.flight_time_s[-1]).all()


def test_invalid_observation_planes_fail_before_tracing_and_wrapper_is_opt_in(monkeypatch):
    from temsim.optics.electron_gun import tracing
    for planes in ([], [1.,1.], [-1.], [np.nan], [3.], [[0.]]):
        with pytest.raises(ValueError):
            ExactGunPlaneObserver(planes, 2.)
    calls=[]
    gun=SimpleNamespace(exit_plane_z_mm=2., validate=lambda:None)
    marker=object()
    def kernel(gun,count=None,**kwargs):
        calls.append((count,kwargs))
        if "observer" in kwargs:
            p=momentum_from_kinetic_energy_ev(np.array([1000.]),np.array([[0.,0.,1.]]))
            kwargs["observer"].initialize(RelativisticPhaseSpace(np.zeros((1,3)),p))
        return marker
    monkeypatch.setattr(tracing,"_trace_feg_to_exit",kernel)
    assert tracing.trace_feg_to_exit(gun,193) is marker
    result,observations=tracing.trace_feg_with_observations(gun,[0.],193)
    assert result is marker and observations.alive.all()
    assert calls[0] == (193,{"cancelled":None})
    assert calls[1][0] == 193 and "observer" in calls[1][1]
