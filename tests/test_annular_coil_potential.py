"""Independent field-operator checks; no finite-source or coherent transport."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import mu_0
from scipy.integrate import quad

from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.lens_field_provider import CoordinateRegistration


def _coil(**changes):
    return AnnularCoilPotential(**dict(dict(source_id="test-coil", inner_radius_m=.006,
        outer_radius_m=.01, length_m=.01, ampere_turns=3000., quadrature_order=12), **changes))


def test_axis_matches_independent_finite_solenoid_radial_integral_and_tail():
    coil = _coil()
    for z in (0., .003, .005, .02, .5):
        upper, lower = z+coil.length_m/2, z-coil.length_m/2
        integral = quad(lambda radius: upper/np.hypot(radius, upper)-lower/np.hypot(radius, lower),
                        coil.inner_radius_m, coil.outer_radius_m, epsabs=1e-14, epsrel=1e-13)[0]
        expected = mu_0*coil.ampere_turns/(2*coil.length_m*(coil.outer_radius_m-coil.inner_radius_m))*integral
        assert coil.axis_bz_t(z) == pytest.approx(expected, rel=2e-11, abs=2e-15)
    assert coil.axis_bz_t(.5) > 0.


def test_off_axis_matches_independent_azimuthal_biot_savart_sum():
    coil = _coil()
    point = np.array([.002, .001, .003])
    phi = (np.arange(512)+.5)*2*np.pi/512
    vector = np.zeros(3)
    potential = np.zeros(3)
    # The independent integrand uses source line elements, no elliptic formula.
    for radius, z, current in coil._loops:
        source = np.column_stack((radius*np.cos(phi), radius*np.sin(phi), np.full(phi.shape, z)))
        dl = radius*2*np.pi/512*np.column_stack((-np.sin(phi), np.cos(phi), np.zeros_like(phi)))
        separation = point-source
        distance = np.linalg.norm(separation, axis=1)
        vector += mu_0*current/(4*np.pi)*np.sum(np.cross(dl, separation)/distance[:, None]**3, axis=0)
        potential += mu_0*current/(4*np.pi)*np.sum(dl/distance[:, None], axis=0)
    result = coil.sample(point)
    np.testing.assert_allclose(result["B_t"], vector, rtol=2e-12, atol=2e-15)
    np.testing.assert_allclose(result["A_t_m"], potential, rtol=2e-12, atol=2e-18)


def test_analytic_jet_curl_is_same_field_and_derivatives_match_finite_differences():
    coil = _coil()
    points = np.array([[0., 0., .003], [.00001, .00002, .003], [.001, .002, -.002], [.005, .001, .008]])
    jet = coil.sample_jet(points)
    jac = jet["jac_A_t"]
    curl = np.stack((jac[:, 2, 1]-jac[:, 1, 2], jac[:, 0, 2]-jac[:, 2, 0], jac[:, 1, 0]-jac[:, 0, 1]), axis=1)
    np.testing.assert_allclose(curl, coil.sample(points)["B_t"], rtol=3e-11, atol=3e-14)
    np.testing.assert_allclose(jet["A_t_m"], coil.sample(points)["A_t_m"], rtol=3e-12, atol=3e-18)
    h = 1e-7
    for axis in range(3):
        offset = np.eye(3)[axis]*h
        numerical_jac = (coil.sample(points+offset)["A_t_m"]-coil.sample(points-offset)["A_t_m"])/(2*h)
        numerical_hess = (coil.sample_jet(points+offset)["jac_A_t"]-coil.sample_jet(points-offset)["jac_A_t"])/(2*h)
        np.testing.assert_allclose(jac[:, :, axis], numerical_jac, rtol=3e-7, atol=2e-10)
        np.testing.assert_allclose(jet["hess_A_t_per_m"][:, :, :, axis], numerical_hess, rtol=3e-6, atol=3e-7)


def test_axis_derivatives_and_quadrature_refinement():
    coil = _coil()
    z, h = np.array([-.012, 0., .004, .02]), 1e-7
    d = coil.axial_derivatives(z, 4)
    for order in range(4):
        difference = (coil.axial_derivatives(z+h, order)[order]-coil.axial_derivatives(z-h, order)[order])/(2*h)
        np.testing.assert_allclose(d[order+1], difference, rtol=2e-8, atol=1e-5)
    points = np.array([[.004, 0., .002], [.002, .001, .008], [0., 0., -.02]])
    refined = replace(coil, quadrature_order=24)
    np.testing.assert_allclose(coil.sample(points)["B_t"], refined.sample(points)["B_t"], rtol=2e-5, atol=1e-10)


def test_geometry_and_current_change_the_field_without_gaussian_coefficients():
    coil = _coil()
    p = [[.001, .002, .003]]
    original = coil.sample(p)
    for changes in (dict(inner_radius_m=.005), dict(outer_radius_m=.012), dict(length_m=.015)):
        changed = replace(coil, **changes)
        assert changed.digest != coil.digest
        assert not np.allclose(changed.sample(p)["B_t"], original["B_t"])
    negative = replace(coil, ampere_turns=-6000.)
    for key in ("A_t_m", "B_t"):
        np.testing.assert_allclose(negative.sample(p)[key], -2*original[key], rtol=0., atol=0.)
    zero = replace(coil, ampere_turns=0.)
    for value in zero.sample(p).values():
        np.testing.assert_array_equal(value, np.zeros((1, 3)))
    assert not hasattr(coil, "slope_derivative")  # Full Lorentz; no paraxial hook or Cs kick.


def test_rigid_pose_rotates_potential_field_and_derivatives():
    angle = .2
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    origin = np.array([.001, -.002, .05])
    local = _coil()
    posed = replace(local, registration=CoordinateRegistration(tuple(origin), tuple(map(tuple, rotation))))
    p = np.array([[.001, .002, .003], [0., 0., .01]])
    global_p = p@rotation.T+origin
    for key, value in local.sample(p).items():
        np.testing.assert_allclose(posed.sample(global_p)[key], value@rotation.T, rtol=5e-13, atol=3e-16)
    expected_jac = np.einsum("ai,nij,bj->nab", rotation, local.sample_jet(p)["jac_A_t"], rotation)
    np.testing.assert_allclose(posed.sample_jet(global_p)["jac_A_t"], expected_jac, rtol=2e-11, atol=3e-14)


def test_source_volume_is_excluded_even_at_zero_current_and_swept_boxes_fail_closed():
    coil = _coil(ampere_turns=0.)
    for point in ([.006, 0., 0.], [.01, 0., .005], [.007, 0., 0.]):
        with pytest.raises(ValueError, match="copper"):
            coil.sample(point)
    coil.sample([[.0059, 0., 0.], [.0101, 0., 0.], [.007, 0., .0051]])
    with pytest.raises(ValueError, match="copper"):
        coil.require_global_box([[.005, 0., 0.], [.011, 0., 0.]])
    coil.require_global_box([[-.001, -.001, -.1], [.001, .001, .1]])


def test_serialization_tampering_and_invalid_inputs():
    coil = _coil()
    restored = AnnularCoilPotential.from_dict(coil.to_dict())
    assert restored.digest == coil.digest
    np.testing.assert_array_equal(restored.sample([[.001, 0., 0.]])["B_t"], coil.sample([[.001, 0., 0.]])["B_t"])
    record = coil.to_dict()
    record["metadata"]["ampere_turns"] += 1
    with pytest.raises(ValueError, match="identity"):
        AnnularCoilPotential.from_dict(record)
    for update in (dict(inner_radius_m=0.), dict(outer_radius_m=.005), dict(length_m=0.),
                   dict(ampere_turns=np.nan), dict(quadrature_order=3), dict(quadrature_order=True)):
        with pytest.raises(ValueError):
            _coil(**update)
    for point in ([np.nan, 0., 0.], [0., 0.], [1+1j, 0., 0.]):
        with pytest.raises(ValueError):
            coil.sample(point)
