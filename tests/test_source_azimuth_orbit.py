"""Analytic source-measure tests; no optics or real propagation is claimed."""
import numpy as np
import pytest
from scripts.source_azimuth_orbit import expand_common_azimuth,complete_orientation_nodes


BASE=np.random.default_rng(169).random((7,5))


def test_k2_is_exact_identity_including_endpoint_one():
    points=BASE.copy();points[0,[1,2]]=1.
    np.testing.assert_array_equal(expand_common_azimuth(points,2).base_points,points)


@pytest.mark.parametrize('k',[2,4,8,16])
def test_source_probability_coordinates_relative_azimuth_and_prefix(k):
    original=BASE.copy();orbit=expand_common_azimuth(BASE,k)
    nodes,w=complete_orientation_nodes(orbit)
    assert len(orbit.base_points)==len(BASE)*k//2
    assert orbit.metadata['independent_sobol_base_count']==len(BASE)
    np.testing.assert_allclose(w.sum(),1.,atol=3e-16)
    np.testing.assert_array_equal(orbit.base_points[:,[0,3,4]],BASE[orbit.independent_base_index][:,[0,3,4]])
    expected=np.repeat((BASE[:,2]-BASE[:,1])%1.,k)
    np.testing.assert_allclose((nodes[:,2]-nodes[:,1])%1.,expected,atol=3e-16)
    prefix=expand_common_azimuth(BASE[:3],k)
    np.testing.assert_array_equal(prefix.base_points,orbit.base_points[:3*k//2])
    np.testing.assert_array_equal(BASE,original)


@pytest.mark.parametrize('k',[2,4,8,16])
def test_common_azimuth_harmonics_cancel_below_orbit_size_but_alias_at_k(k):
    nodes,w=complete_orientation_nodes(expand_common_azimuth(BASE,k))
    for harmonic in range(1,k):
        value=np.exp(2j*np.pi*harmonic*nodes[:,1]).reshape(len(BASE),k).mean(axis=1)
        np.testing.assert_allclose(value,0.,atol=5e-15)
    alias=np.exp(2j*np.pi*k*nodes[:,1]).reshape(len(BASE),k).mean(axis=1)
    np.testing.assert_allclose(alias,np.exp(2j*np.pi*k*BASE[:,1]),atol=8e-15)


def test_a1_first_order_cancels_at_k4_but_second_order_m4_requires_k8():
    point=BASE[:1];epsilon=2e-4;expected=3*epsilon**2
    errors={}
    for k in (2,4,8):
        nodes,w=complete_orientation_nodes(expand_common_azimuth(point,k))
        psi=2*np.pi*nodes[:,1]
        delta=epsilon*np.cos(2*psi)+epsilon**2*(3+np.cos(4*psi))
        errors[k]=float(w@delta-expected)
    assert abs(errors[2])>epsilon*1e-2
    np.testing.assert_allclose(errors[4],epsilon**2*np.cos(8*np.pi*point[0,1]),atol=3e-20)
    # Float64 modulo and trigonometric arguments round before cancellation.
    # Bound roundoff relative to the O(epsilon) summands, not O(epsilon**2).
    assert abs(errors[8])<16*np.finfo(float).eps*(epsilon+4*epsilon**2)


def test_nonzero_flux_and_true_downstream_centroid_are_retained():
    nodes,w=complete_orientation_nodes(expand_common_azimuth(BASE,8))
    psi=2*np.pi*nodes[:,1];eps=1e-4
    # A non-rotation-equivariant optical response has an actual constant shift.
    # The source orbit must retain it, not rotate images or re-centre them.
    flux=.7+eps*np.cos(2*psi)+3*eps**2
    x=.01+eps*(1+np.cos(2*psi));y=-.02+eps*np.sin(psi)
    probability=w@flux
    centroid=np.array([w@(flux*x),w@(flux*y)])/probability
    np.testing.assert_allclose(probability,.7+3*eps**2,atol=2e-16)
    np.testing.assert_allclose(centroid,[.01+eps+eps**2/(2*probability),-.02],atol=1e-16)
    assert probability<1 and centroid[0]>.01


@pytest.mark.parametrize('k',[True,4.,0,3,32])
def test_invalid_orbit_size_rejected(k):
    with pytest.raises(ValueError):expand_common_azimuth(BASE,k)


@pytest.mark.parametrize('points',[np.empty((0,5)),np.ones((2,6)),np.full((2,5),np.nan),np.full((2,5),1.1)])
def test_invalid_base_rejected(points):
    with pytest.raises(ValueError):expand_common_azimuth(points,4)
