"""Geometry-only inner accuracy must also apply during plane localization."""
from inspect import signature
from types import SimpleNamespace
import numpy as np
import pytest
from temsim.optics.electron_gun import tracing
from temsim.physics import static_energy_lorentz as solver
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
    kinetic_energy_ev_from_momentum, ELECTRON_MASS_KG,
    SPEED_OF_LIGHT_M_PER_S, ELEMENTARY_CHARGE_C,
)


def fields(electric=0., magnetic=0.):
    return (SimpleNamespace(field_at_global_positions_t=lambda x:
                np.tile([0.,0.,magnetic],(len(x),1))),
            SimpleNamespace(potential_v_at_global_positions=lambda x: electric*x[:,2],
                field_at_global_positions_v_per_m=lambda x:
                np.tile([0.,0.,-electric],(len(x),1))))


@pytest.mark.parametrize('tolerance',[1e-13,2e-13])
def test_geometry_retry_calls_use_explicit_inner_tolerance(monkeypatch,tolerance):
    original=solver.static_energy_step; calls=[]
    def observed(*args,**kwargs):
        assert kwargs['tolerance']==tolerance
        calls.append(float(args[1]))
        return original(*args,**kwargs)
    monkeypatch.setattr(tracing,'GEOMETRY_IMPLICIT_TOLERANCE',tolerance)
    monkeypatch.setattr(solver,'static_energy_step',observed)
    initial=RelativisticPhaseSpace(np.zeros((1,3)),
        momentum_from_kinetic_energy_ev([300000.],[[1e-6,0.,1.]]))
    p=initial.momentum_kg_m_per_s[0]
    gamma=np.sqrt(1+np.sum(p*p)/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)**2)
    omega=ELEMENTARY_CHARGE_C/(gamma*ELECTRON_MASS_KG)
    dt=.3/omega
    actual,used=tracing._surface_step(initial,dt,np.array([True]),*fields(magnetic=1.))
    assert used<dt and len(calls)>3
    expected=p[0]*np.array([np.cos(omega*used),np.sin(omega*used)])
    assert np.linalg.norm(actual.momentum_kg_m_per_s[0,:2]-expected)/abs(p[0])<1e-5


@pytest.mark.parametrize('tolerance',[1e-13,2e-13])
def test_geometry_plane_event_uses_the_same_inner_tolerance(monkeypatch,tolerance):
    original=solver.static_energy_step;calls=[]
    def observed(*args,**kwargs):
        assert kwargs['tolerance']==tolerance
        calls.append((float(args[0].time_s),float(args[1])))
        return original(*args,**kwargs)
    monkeypatch.setattr(tracing,'GEOMETRY_IMPLICIT_TOLERANCE',tolerance)
    monkeypatch.setattr(solver,'static_energy_step',observed)
    magnetic,electric=fields(electric=1e8)
    momentum=momentum_from_kinetic_energy_ev([1000.],[[0.,0.,1.]])[0]
    target=1e-5
    result=tracing._surface_plane_crossing(target,np.zeros(3),momentum,0.,1e-12,electric,magnetic)
    assert len(calls)>=4
    assert 0.<result.time_s<1e-12
    assert result.position_m[0,2]==pytest.approx(target,abs=1e-15)
    assert kinetic_energy_ev_from_momentum(result.momentum_kg_m_per_s)[0]==pytest.approx(2000.,abs=1e-6)


def test_general_static_solver_default_is_unchanged():
    from temsim.physics.discrete_gradient import ITERATION_TOLERANCE
    assert ITERATION_TOLERANCE==1e-11
    assert signature(solver.static_energy_step).parameters['tolerance'].default==1e-11
    assert tracing.geometry_step_criteria()['implicit_tolerance']==1e-13
