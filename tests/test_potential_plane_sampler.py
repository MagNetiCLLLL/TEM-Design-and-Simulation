"""Aligned-plane aggregation preserves the same frozen potentials and domains."""
from dataclasses import replace, FrozenInstanceError

import numpy as np
import pytest

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.potential_plane_sampler import compile_plane_sampler, FrozenPlanePotentialSampler
from temsim.physics.potential_snapshot import StaticPolynomialGauge
from test_potential_wave_carrier import _snapshot, _round, _multipole


def _rotated_multipole():
    angle = .37
    rotation = ((np.cos(angle), -np.sin(angle), 0.), (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.))
    native = replace(_multipole().base.field, registration=CoordinateRegistration((0., 0., .001), rotation),
        native_support_mm=(-5., 5.), envelope_kind="gaussian", sigma_m=.003,
        polynomial_terms=((1, 0, -1e-5), (0, 1, 3e-5), (2, 0, 12.),
                          (0, 2, -12.), (3, 0, 1e5), (1, 2, -3e5)))
    return AnalyticPotentialDomain(AnalyticMagneticPotential(native))


@pytest.mark.parametrize("order", [1, 3, 5, 7, 9])
@pytest.mark.parametrize("z", [-.012, -.002, .003, .012])
def test_multiple_rounds_rotated_multipole_and_gauge_match_original_potentials(order, z):
    round_one = _round(.2, order)
    round_two = HighOrderRoundLensPotential(replace(round_one.native,
        terms_t_m=((-0.07, .002, .008), (.11, -.005, .015)),
        registration=CoordinateRegistration((0., 0., -.001))), radial_order=order, axial_domain="gaussian_all_z")
    gauge = StaticPolynomialGauge(1e-15, (1e-10, -2e-10, 3e-10),
        ((2e-7, 1e-8, 3e-8), (1e-8, -1e-7, -5e-8), (3e-8, -5e-8, 2e-7)))
    snapshot = _snapshot(round_one, round_two, _rotated_multipole(), gauge=gauge)
    sampler = compile_plane_sampler(snapshot)
    assert sampler.aligned
    rng = np.random.default_rng(713)
    points = np.column_stack((rng.uniform(-2e-4, 2e-4, (87, 2)), np.full(87, z)))
    expected, actual = snapshot.sample(points), sampler.sample(points)
    for key in ("A_t_m", "B_t"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-13, atol=3e-18)
    for key, value in sampler.sample_jet(points).items():
        np.testing.assert_array_equal(value, sampler.fallback.sample_jet(points)[key])


def test_strict_domains_closed_compact_faces_and_known_zero_exterior_are_retained():
    compact = _rotated_multipole()
    sampler = compile_plane_sampler(_snapshot(compact))
    # Local support [-5,5] mm translated by +1 mm has closed global faces.
    for z in (-.004, .006):
        np.testing.assert_allclose(sampler.sample((1e-4, 0., z))["A_t_m"],
                                   compact.sample((1e-4, 0., z))["A_t_m"], rtol=2e-14)
        with pytest.raises(ValueError, match="outside"):
            sampler.sample((.004, 0., z))
    np.testing.assert_array_equal(sampler.sample((1., 0., .0061))["A_t_m"], (0., 0., 0.))
    strict = HighOrderRoundLensPotential(_round().native, axial_domain="native_support")
    sampler = compile_plane_sampler(_snapshot(strict))
    for point in ((0., 0., -.05), (0., 0., .05), (.004, 0., 0.)):
        with pytest.raises(ValueError, match="outside"):
            sampler.sample(point)


def test_rotated_radius_boundary_uses_original_membership_without_a_new_tolerance():
    snapshot = _snapshot(_rotated_multipole())
    sampler = compile_plane_sampler(snapshot)
    for radius in (.003, np.nextafter(.003, np.inf), np.nextafter(.003, 0.)):
        for angle in np.linspace(0., 2*np.pi, 33):
            point = (radius*np.cos(angle), radius*np.sin(angle), .001)
            try:
                expected = snapshot.sample(point)
            except ValueError:
                with pytest.raises(ValueError):
                    sampler.sample(point)
            else:
                for key, value in sampler.sample(point).items():
                    np.testing.assert_array_equal(value, expected[key])


def test_registration_validation_tolerance_does_not_admit_axisymmetric_reaggregation():
    # The general registration contract accepts this small distortion. It is
    # not an exact SO(2) identity and must keep its original transformed law.
    native = replace(_round().native, registration=CoordinateRegistration(
        (0., 0., 0.), ((1.+5e-11, 0., 0.), (0., 1., 0.), (0., 0., 1.))))
    snapshot = _snapshot(HighOrderRoundLensPotential(native, axial_domain="gaussian_all_z"))
    sampler = compile_plane_sampler(snapshot)
    assert not sampler.aligned
    point = (.003/(1.+2e-11), 0., 0.)
    assert np.hypot(point[0], point[1]) < .003
    with pytest.raises(ValueError, match="outside"):
        snapshot.sample(point)
    with pytest.raises(ValueError, match="outside"):
        sampler.sample(point)
    for key, value in sampler.sample((1e-4, 0., 0.)).items():
        np.testing.assert_array_equal(value, sampler.fallback.sample((1e-4, 0., 0.))[key])


@pytest.mark.parametrize("offset,tilt", [(1e-5, 0.), (0., .13)])
def test_off_axis_and_tilted_sources_take_exact_strict_compiled_fallback(offset, tilt):
    rotation = ((np.cos(tilt), 0., np.sin(tilt)), (0., 1., 0.),
                (-np.sin(tilt), 0., np.cos(tilt)))
    native = replace(_round().native, registration=CoordinateRegistration((offset, 0., 0.), rotation))
    sampler = compile_plane_sampler(_snapshot(HighOrderRoundLensPotential(native, axial_domain="gaussian_all_z")))
    assert not sampler.aligned
    points = np.array(((0., 0., .001), (1e-5, -2e-5, .001)))
    for key, value in sampler.sample(points).items():
        np.testing.assert_array_equal(value, sampler.fallback.sample(points)[key])


def test_nonplanar_selected_sources_empty_shape_identity_and_immutability():
    snapshot = _snapshot(_round(), _rotated_multipole())
    sampler = compile_plane_sampler(snapshot)
    points = np.array(((1e-5, 0., 0.), (0., 1e-5, .001)))
    for selection in (None, ("s1",), ("s0", "s1")):
        for key, value in sampler.sample(points, selection).items():
            np.testing.assert_array_equal(value, sampler.fallback.sample(points, selection)[key])
    assert sampler.sample(np.empty((0, 3)))["A_t_m"].shape == (0, 3)
    sampler.require_snapshot(snapshot)
    with pytest.raises(ValueError, match="exact captured"):
        sampler.require_snapshot(_snapshot(_round(.1)))
    with pytest.raises(FrozenInstanceError):
        sampler.aligned = False
    with pytest.raises(TypeError, match="compile_plane_sampler"):
        FrozenPlanePotentialSampler()
    with pytest.raises(ValueError, match="finite real"):
        sampler.sample((np.nan, 0., 0.))


@pytest.mark.parametrize("mode", ["nano_probe", "nano_probe_large_angle"])
def test_actual_29_source_plane_matches_original_at_crossover_lens_and_hexapole_planes(mode):
    from test_potential_probe_pilot import _state, _capture
    snapshot, _, _ = _capture(_state(mode), 3)
    sampler = compile_plane_sampler(snapshot)
    assert sampler.aligned and len(snapshot.contributions) == 29
    x, y = np.meshgrid(np.linspace(-1e-4, 1e-4, 17), np.linspace(-1e-4, 1e-4, 15))
    for z in (.67, .915, 1.135, 1.1445, 1.2, 1.451):
        points = np.stack((x, y, np.full_like(x, z)), axis=-1)
        expected, actual = snapshot.sample(points), sampler.sample(points)
        for key in ("A_t_m", "B_t"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=4e-12, atol=2e-16)
