"""Strict compiled sampling preserves the captured potential and domain laws."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_native_sampler import compile_native_sampler, FrozenNativePotentialSampler
from temsim.physics.potential_snapshot import (
    ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge,
)


def _snapshot(*providers, gauge=None):
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    return ElectromagneticPotentialSnapshot(scene, tuple(PotentialContribution(
        f"source:{index}", (f"coil:{index}",), provider, "magnetic")
        for index, provider in enumerate(providers)), gauge or StaticPolynomialGauge())


def _providers(order=3, *, posed=False, envelope="gaussian", axial_domain="gaussian_all_z"):
    registration = (CoordinateRegistration((1e-5, -3e-5, .002),
        tuple(map(tuple, Rotation.from_rotvec((.03, -.02, .11)).as_matrix())))
        if posed else CoordinateRegistration())
    round_field = FrozenAnalyticField("round", ((.2, .001, .004), (-.08, -.002, .002)),
                                      registration, (-20., 20.), .003)
    multipole = FrozenPosedMultipole("multipole", "physical-host", registration,
        ((1, 0, -2e-4), (0, 1, 1e-4), (2, 0, .03), (0, 2, -.03),
         (1, 1, .02), (3, 0, 1e6), (1, 2, -3e6)), envelope,
        .001, .004, (-10., 10.), .003, 1.)
    return (HighOrderRoundLensPotential(round_field, radial_order=order, axial_domain=axial_domain),
            AnalyticPotentialDomain(AnalyticMagneticPotential(multipole)))


def _compare(snapshot, sampler, points, selected=None):
    before = np.array(points, copy=True)
    expected, actual = snapshot.sample(points, selected), sampler.sample(points, selected)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=5e-13, atol=2e-20 if key == "A_t_m" else 2e-15)
        assert actual[key].shape == expected[key].shape
    np.testing.assert_array_equal(points, before)


@pytest.mark.parametrize("order", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("posed", [False, True])
def test_round_orders_and_rigid_mixed_sources_keep_potential_field_and_jets(order, posed):
    gauge = StaticPolynomialGauge(2e-15, (2e-7, -3e-7, 1e-7),
        ((.002, -.001, .003), (-.001, .004, -.002), (.003, -.002, .001)))
    snapshot = _snapshot(*_providers(order, posed=posed), gauge=gauge)
    sampler = compile_native_sampler(snapshot)
    rng = np.random.default_rng(4901)
    points = rng.uniform((-.0003, -.0003, -.005), (.0003, .0003, .005), (13, 3))
    _compare(snapshot, sampler, points)
    actual = sampler.sample_jet(points)
    expected = {key: sum(row.potential.sample_jet(points)[key] for row in snapshot.contributions)
                for key in actual}
    expected["A_t_m"] += gauge.vector_potential_shift_t_m(points)
    expected["jac_A_t"] += gauge.hessian_t
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=2e-14)
    gradient = actual["jac_A_t"]
    curl = np.stack((gradient[:, 2, 1]-gradient[:, 1, 2],
                     gradient[:, 0, 2]-gradient[:, 2, 0],
                     gradient[:, 1, 0]-gradient[:, 0, 1]), axis=-1)
    np.testing.assert_allclose(curl, sampler.sample(points)["B_t"], rtol=2e-12, atol=2e-15)


@pytest.mark.parametrize("order", [3, 5, 7])
def test_actual_29_source_probe_column_keeps_order_selection_and_shared_coil_ownership(order):
    from test_potential_probe_pilot import _state, _capture
    snapshot, _, _ = _capture(_state("nano_probe"), order)
    assert len(snapshot.contributions) == 29
    sampler = compile_native_sampler(snapshot)
    assert sampler.source_ids == tuple(row.source_id for row in snapshot.contributions)
    assert len({owner for row in snapshot.contributions for owner in row.owner_ids}) == sum(
        len(row.owner_ids) for row in snapshot.contributions)
    points = np.c_[np.linspace(-8e-6, 8e-6, 33), np.linspace(7e-6, -7e-6, 33), np.linspace(.6, 1.59919, 33)]
    _compare(snapshot, sampler, points)
    selected = tuple(reversed(sampler.source_ids[::3]))
    _compare(snapshot, sampler, points, selected)
    # Source-order summation is unchanged even when selection order differs.
    for key, value in sampler.sample(points, selected).items():
        np.testing.assert_array_equal(value, sampler.sample(points, selected[::-1])[key])


@pytest.mark.parametrize("shape", [(3,), (0, 3), (2, 3), (2, 2, 3)])
def test_shapes_and_empty_selection_keep_gauge_once(shape):
    gauge = StaticPolynomialGauge(1e-12, (1e-6, 2e-6, -3e-6), np.eye(3)*.001)
    snapshot = _snapshot(*_providers(), gauge=gauge)
    sampler = compile_native_sampler(snapshot)
    points = np.full(shape, 1e-5)
    _compare(snapshot, sampler, points)
    _compare(snapshot, sampler, points, ())


@pytest.mark.parametrize("envelope", ["uniform", "gaussian"])
def test_compact_zero_exterior_faces_radial_domain_and_derivative_face_gate(envelope):
    _, multipole = _providers(envelope=envelope)
    snapshot = _snapshot(multipole)
    sampler = compile_native_sampler(snapshot)
    _compare(snapshot, sampler, np.array(((.001, .001, -.01), (.001, .001, .01))))
    _compare(snapshot, sampler, np.array(((1e200, 1e200, -.010001),)))
    for points in ((.004, 0., 0.), (0., .004, .01)):
        for function in (snapshot.sample, sampler.sample):
            with pytest.raises(ValueError, match="domain"):
                function(points)
    with pytest.raises(ValueError, match="hard axial face"):
        sampler.sample_jet((0., 0., .01))


def test_round_native_support_stays_strict_and_pose_local_domain_is_not_global_radius():
    provider, _ = _providers(axial_domain="native_support", posed=True)
    snapshot = _snapshot(provider); sampler = compile_native_sampler(snapshot)
    # A point inside the local tube can lie outside a naive global radius.
    registration = provider.registration
    for local in ((.0029, 0., .003), (.0031, 0., .003), (0., 0., .0201)):
        point = np.asarray(local)@registration.rotation_array.T+registration.origin_array_m
        if local[0] == .0029:
            _compare(snapshot, sampler, point)
        else:
            for function in (snapshot.sample, sampler.sample):
                with pytest.raises(ValueError, match="domain"):
                    function(point)
    aligned, _ = _providers(axial_domain="native_support")
    for z in (-.02, .02):
        with pytest.raises(ValueError, match="domain"):
            compile_native_sampler(_snapshot(aligned)).sample((0., 0., z))


def test_identity_freezing_and_unsupported_or_dynamic_sources_fail_closed():
    round_provider, multipole = _providers()
    snapshot = _snapshot(round_provider, multipole)
    sampler = compile_native_sampler(snapshot)
    sampler.require_snapshot(snapshot)
    with pytest.raises(TypeError, match="compile_native_sampler"):
        FrozenNativePotentialSampler(snapshot_digest=snapshot.numerical_digest, _packed=())
    with pytest.raises(FrozenInstanceError):
        sampler.snapshot_digest = "changed"
    for value in (*sampler._packed, sampler._gauge_gradient, sampler._gauge_hessian):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(ValueError, match="numerical digest"):
        sampler.require_snapshot(_snapshot(replace(round_provider, radial_order=7), multipole))
    mapped = AxisymmetricMagneticPotential((0., .003), (-.01, .01), ((0., 0.), (.001, .001)))
    with pytest.raises(ValueError, match="unsupported"):
        compile_native_sampler(_snapshot(mapped))
    dynamic = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(multipole.base.field, dynamic=True)))
    with pytest.raises(ValueError, match="dynamic"):
        compile_native_sampler(_snapshot(dynamic))
    provenance_dynamic = AnalyticPotentialDomain(AnalyticMagneticPotential(multipole.base.field,
                                                                           {"dynamic_drive_frozen": True}))
    with pytest.raises(ValueError, match="dynamic"):
        compile_native_sampler(_snapshot(provenance_dynamic))


@pytest.mark.parametrize("points", [(True, False, True), (np.nan, 0., 0.), (1+2j, 0., 0.), (1., 2.)])
def test_invalid_points_are_not_cast_to_an_admitted_query(points):
    sampler = compile_native_sampler(_snapshot(*_providers()))
    with pytest.raises(ValueError, match="finite real XYZ"):
        sampler.sample(points)


def test_curl_field_matches_independent_potential_finite_difference():
    snapshot = _snapshot(*_providers(7, posed=True))
    sampler = compile_native_sampler(snapshot)
    point = np.array((1.1e-4, -8e-5, .0013))
    curls = []
    for step in (2e-8, 1e-8):
        gradient = np.column_stack([(sampler.sample(point+np.eye(3)[i]*step)["A_t_m"]
                                     -sampler.sample(point-np.eye(3)[i]*step)["A_t_m"])/(2*step)
                                    for i in range(3)])
        curls.append(np.array((gradient[2, 1]-gradient[1, 2], gradient[0, 2]-gradient[2, 0],
                               gradient[1, 0]-gradient[0, 1])))
    expected = snapshot.sample(point)["B_t"]
    assert np.linalg.norm(curls[1]-expected) < .3*np.linalg.norm(curls[0]-expected)
    # Richardson eliminates the independently observed second-order derivative
    # truncation; it does not relax the exact coefficient-parity tolerance.
    np.testing.assert_allclose((4*curls[1]-curls[0])/3, expected, rtol=2e-10, atol=2e-13)


def test_one_joint_source_is_sampled_once_and_empty_snapshot_keeps_pure_gauge():
    provider, _ = _providers()
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    snapshot = ElectromagneticPotentialSnapshot(scene, (PotentialContribution(
        "shared-circuit", ("physical-host:upper", "physical-host:lower"), provider, "magnetic"),))
    sampler = compile_native_sampler(snapshot)
    point = np.array((.0001, -.0002, .001))
    _compare(snapshot, sampler, point)
    np.testing.assert_allclose(sampler.sample(point)["B_t"], provider.sample(point)["B_t"], rtol=1e-14)
    gauge = StaticPolynomialGauge(3e-15, (1e-8, -3e-8, 2e-8), np.eye(3)*.002)
    empty = _snapshot(gauge=gauge)
    compiled = compile_native_sampler(empty)
    _compare(empty, compiled, point)
    np.testing.assert_array_equal(compiled.sample_jet(point)["jac_A_t"], gauge.hessian_t)


@pytest.mark.parametrize("selection", [("unknown",), ("source:0", "source:0"), "source:0", (1,)])
def test_invalid_selections_are_rejected(selection):
    sampler = compile_native_sampler(_snapshot(*_providers()))
    with pytest.raises(ValueError, match="source|identifiers"):
        sampler.sample((0., 0., 0.), selection)
