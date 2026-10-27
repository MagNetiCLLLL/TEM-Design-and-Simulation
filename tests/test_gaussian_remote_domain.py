"""Independent Gaussian bounds and common Python/native remote admission."""
from dataclasses import replace
import json
import math

import mpmath as mp
import numpy as np
import pytest

from temsim.physics.gaussian_remote_domain import POLICY,remote_certificate
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import FrozenAnalyticField,CoordinateRegistration
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot,PotentialContribution
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_native_sampler import compile_native_sampler
from temsim.physics.potential_plane_sampler import compile_plane_sampler
from temsim.optics.column import default_state


def source(*,registration=CoordinateRegistration(),terms=((.1,0.,1.),),order=3,remote=True):
    field=FrozenAnalyticField("remote-test",terms,registration,(-3000.,3000.),.009)
    return HighOrderRoundLensPotential(field,radial_order=order,axial_domain="gaussian_all_z",
        remote_domain=POLICY if remote else "strict")


def snapshot(provider):
    return ElectromagneticPotentialSnapshot(capture_physical_scene(default_state()),
        (PotentialContribution(provider.source_id,(provider.source_id,),provider,"magnetic"),))


def test_strict_default_and_frozen_identity_are_preserved():
    strict=source(remote=False);remote=source()
    assert strict.native.fingerprint==remote.native.fingerprint
    assert strict.valid_radius_m==remote.valid_radius_m
    assert "remote_domain" not in strict.to_dict()
    assert remote.digest!=strict.digest
    assert remote.to_metadata()["remote_domain"]["policy"]==POLICY
    assert HighOrderRoundLensPotential.from_dict(json.loads(json.dumps(remote.to_dict()))).digest==remote.digest
    with pytest.raises(ValueError,match="domain"):
        strict.sample(np.array([.01,0.,50.]))
    with pytest.raises(ValueError,match="qualification"):
        remote.sample(np.array([.01,0.,0.]))
    with pytest.raises(ValueError,match="all-z"):
        replace(remote,axial_domain="native_support")


@pytest.mark.parametrize("order",(1,3,5,7,9))
def test_actual_normal_c3_failure_is_remote_qualified_without_changing_evaluation(order):
    terms=((.003149130803072318,.906,.0081),(.02869208065021445,.915,.00495),(.003149130803072318,.924,.0081))
    provider=source(terms=terms,order=order)
    point=np.array([.00900904769179203,0.,2.173429313760807])
    certificate=remote_certificate(provider.native,point,order)
    assert certificate["admitted"] and certificate["minimum_binary_margin"]>10000
    for values in (provider.sample(point),provider.sample_jet(point)):
        assert all(np.all(value==0.) for value in values.values())
    # Independent arbitrary-precision positive bound. No floating exp/log
    # determines either the production gate or this reference inequality.
    with mp.workdps(80):
        r,z=mp.mpf(point[0]),mp.mpf(point[2])
        M=sum(abs(mp.mpf(b))*mp.exp((r*r-(z-mp.mpf(centre))**2)/(2*mp.mpf(sigma)**2)) for b,centre,sigma in terms)
        assert 0<M<mp.power(2,-1075)


@pytest.mark.parametrize("placed",(False,True))
def test_native_plane_and_python_agree_with_signed_superposition_and_placed_jets(placed):
    angle=.13
    rotation=((math.cos(angle),0.,math.sin(angle)),(0.,1.,0.),(-math.sin(angle),0.,math.cos(angle)))
    registration=CoordinateRegistration((.1,-.2,.3),rotation) if placed else CoordinateRegistration()
    provider=source(registration=registration,terms=((.1,0.,1.),(-.08,.1,.8)))
    local=np.array([[.02,0.,50.],[.001,.002,50.]])
    points=local@registration.rotation_array.T+registration.origin_array_m
    scene=snapshot(provider)
    for sampler in (scene,compile_native_sampler(scene),compile_plane_sampler(scene)):
        actual=sampler.sample(points)
        for key,value in provider.sample(points).items():
            np.testing.assert_array_equal(actual[key],value)
    for sampler in (compile_native_sampler(scene),compile_plane_sampler(scene)):
        for key,value in provider.sample_jet(points).items():
            np.testing.assert_array_equal(sampler.sample_jet(points)[key],value)
    near=np.array([[.02,0.,0.]])@registration.rotation_array.T+registration.origin_array_m
    for sampler in (provider,compile_native_sampler(scene),compile_plane_sampler(scene)):
        with pytest.raises(ValueError,match="qualification"):
            sampler.sample(near)


def test_adjacent_binary64_gate_and_independent_polynomial_derivatives():
    provider=source()
    low,high=40.,60.
    while np.nextafter(low,np.inf)<high:
        middle=(low+high)/2
        if middle in (low,high):break
        if remote_certificate(provider.native,np.array([.02,0.,middle]),3)["admitted"]:
            high=middle
        else:low=middle
    assert np.nextafter(low,np.inf)==high
    with pytest.raises(ValueError,match="qualification"):
        provider.sample(np.array([.02,0.,low]))
    assert np.count_nonzero(provider.sample_jet(np.array([.02,0.,high]))["hess_A_t_per_m"])==0
    with mp.workdps(80):
        point=(mp.mpf(.02),mp.mpf(0),mp.mpf(high))
        def ax(x,y,z):
            b=mp.mpf(.1)*mp.exp(-z*z/2)
            return -y*(b/2-(x*x+y*y)*(z*z-1)*b/16)
        def ay(x,y,z):return -ax(y,x,z)
        for function in (ax,ay):
            for i in range(3):
                for j in range(3-i):
                    for k in range(3-i-j):
                        assert abs(mp.diff(function,point,(i,j,k)))<mp.power(2,-1075)


def test_no_cancellation_or_extreme_arithmetic_can_fake_remote_admission():
    provider=source(terms=((1e200,0.,1.),(-1e200,0.,1.)))
    with pytest.raises(ValueError,match="qualification"):
        provider.sample(np.array([.02,0.,40.]))
    # A nonfinite positive prefactor must reject, even when exp(-d^2) is zero.
    extreme=source(terms=((1.,0.,1e-150),),order=9)
    with pytest.raises(ValueError,match="qualification"):
        extreme.sample(np.array([.02,0.,10.]))
    ordinary=source(remote=False)
    remote=replace(ordinary,remote_domain=POLICY)
    points=np.array([[1e-3,2e-3,.1],[0.,0.,.5]])
    for key,value in ordinary.sample_jet(points).items():
        np.testing.assert_array_equal(remote.sample_jet(points)[key],value)


def test_remote_endpoint_admission_does_not_admit_nonconvex_swept_interior():
    from itertools import product
    provider=source()
    corners=np.array(tuple(product((-.02,.02),(-.02,.02),(-60.,60.))))
    assert all(np.all(value==0.) for value in provider.sample(corners).values())
    with pytest.raises(ValueError,match="swept box"):
        provider.require_global_box(corners)
    remote=np.array(tuple(product((-.02,.02),(-.02,.02),(59.,60.))))
    provider.require_global_box(remote)
    near_inside=np.array(tuple(product((-.001,.001),(-.001,.001),(-60.,60.))))
    provider.require_global_box(near_inside)


def test_strict_swept_api_certifies_enclosing_box_not_only_supplied_points():
    angle=.4
    registration=CoordinateRegistration(rotation_local_to_global=(
        (math.cos(angle),0.,math.sin(angle)),(0.,1.,0.),(-math.sin(angle),0.,math.cos(angle))))
    provider=source(remote=False,registration=registration)
    points=np.array([[0.,0.,-.1],[0.,0.,.1]])@registration.rotation_array.T
    provider.sample(points)
    with pytest.raises(ValueError,match="domain"):
        provider.require_global_box(points)
