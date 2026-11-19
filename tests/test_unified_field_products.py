"""Synthetic field-product dispatch checks; no emission, trajectory or wave."""
from types import SimpleNamespace

import numpy as np

from temsim.physics.unified_potential_sampler import UnifiedPotentialSampler
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot


def _sampler(snapshot, compiled=None):
    sampler = object.__new__(UnifiedPotentialSampler)
    for key,value in (("snapshot",snapshot),("static_snapshot",snapshot),
                      ("compiled",compiled),("remaining",()),("backend","synthetic unit fixture")):
        object.__setattr__(sampler,key,value)
    return sampler


def _fields(points):
    return {"B_t":np.ones_like(points), "E_v_per_m":np.ones_like(points)*2.,
            "phi_v":np.ones(np.shape(points)[:-1])*3.}


def test_static_lorentz_product_uses_only_explicit_fields_api():
    calls=[]
    def fields(points):
        calls.append("fields")
        return _fields(points)
    def full(points):
        raise AssertionError("A static Lorentz request computed an unused vector potential")
    compiled=SimpleNamespace(sample_fields=fields,sample=full)
    sampler=_sampler(object(),compiled)
    points=np.zeros((3,3))
    result=sampler.sample_fields(points)
    assert calls==["fields"]
    assert set(result)=={"B_t","E_v_per_m","phi_v"}
    np.testing.assert_array_equal(result["B_t"],np.ones_like(points))


def test_reference_product_keeps_full_A_available_for_canonical_handoffs():
    def full(points):
        return dict(_fields(points),A_t_m=np.full_like(points,7.))
    sampler=_sampler(SimpleNamespace(sample=full))
    points=np.zeros((2,3))
    assert "A_t_m" not in sampler.sample_fields(points)
    np.testing.assert_array_equal(sampler.sample(points)["A_t_m"],np.full_like(points,7.))


def test_dynamic_field_product_keeps_original_time_derivatives(monkeypatch):
    snapshot=object.__new__(DynamicPotentialSnapshot)
    calls=[]
    def forbidden(points):
        raise AssertionError("Dynamic field request used a stationary fields-only shortcut")
    sampler=_sampler(snapshot,SimpleNamespace(sample_fields=forbidden))
    def full(self,points,time_s=None):
        calls.append(time_s)
        return dict(_fields(points),A_t_m=np.full_like(points,7.),
                    dA_dt_t_m_per_s=np.full_like(points,11.))
    monkeypatch.setattr(UnifiedPotentialSampler,"sample",full)
    points=np.zeros((2,3))
    result=sampler.sample_fields(points,.123)
    assert calls==[.123]
    assert "A_t_m" not in result
    np.testing.assert_array_equal(result["dA_dt_t_m_per_s"],np.full_like(points,11.))
