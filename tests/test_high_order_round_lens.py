"""Independent Gaussian-vacuum expansion, curl, jet and domain checks."""
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest
from scipy.special import eval_hermitenorm, i0, i1
from scipy.spatial.transform import Rotation

from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField


def gaussian_lens(order=5, *, posed=False, **kwargs):
    rotation = Rotation.from_rotvec((.03, -.02, .1)).as_matrix() if posed else np.eye(3)
    registration = CoordinateRegistration((.0001, -.0002, .3) if posed else (0., 0., 0.),
                                           tuple(map(tuple, rotation)))
    native = FrozenAnalyticField("gaussian-round", ((.3, 0., .006),), registration,
                                 (-80., 200.), .006)
    return HighOrderRoundLensPotential(native, order, .004, **kwargs)


def gaussian_vacuum_quadrature(radius_m, z_m):
    """Independent Fourier-Bessel integral, not an axial-derivative series.

    Gaussian axial Fourier spectrum times I0/I1 solves the axisymmetric
    vacuum Laplace equation. The finite radial test interval is deliberate;
    an analytic Gaussian continuation is not a real exterior pole field.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(96)
    u, v = nodes*radius_m/.006, nodes*z_m/.006
    scale = .3/np.sqrt(2*np.pi)
    bz = scale*np.sum(weights*i0(u)*np.cos(v))
    br = scale*np.sum(weights*i1(u)*np.sin(v))
    a = scale*np.sum(weights*.006*i1(u)/nodes*np.cos(v))
    return a, br, bz


@pytest.mark.parametrize("order", [1, 3, 5, 7, 9])
def test_axis_preserves_native_B0_and_regular_zero_potential(order):
    field = gaussian_lens(order)
    points = np.column_stack((np.zeros((7, 2)), np.linspace(-.02, .02, 7)))
    np.testing.assert_array_equal(field.sample(points)["A_t_m"], np.zeros_like(points))
    np.testing.assert_array_equal(field.sample(points)["B_t"], field.native.field_at_global_positions_t(points))
    assert not hasattr(field, "slope_derivative")


def test_order_one_exactly_matches_existing_near_axis_field_without_changing_it():
    field = gaussian_lens(1, posed=True)
    local = np.array([[.0001, -.0002, .003], [.0004, .0001, -.01]])
    points = local@field.registration.rotation_array.T+field.registration.origin_array_m
    np.testing.assert_allclose(field.sample(points)["B_t"], field.native.field_at_global_positions_t(points),
                               rtol=2e-15, atol=1e-18)


def test_gaussian_derivatives_match_independent_hermite_implementation():
    field = gaussian_lens(9)
    z = np.array([-.01, -.002, 0., .004, .013])
    actual = field.axial_derivatives(z, 12)
    u = z/.006
    for n in range(13):
        expected = .3*np.exp(-u*u/2)*(-1.)**n*eval_hermitenorm(n, u)/.006**n
        np.testing.assert_allclose(actual[n], expected, rtol=2e-14, atol=1e-14)


def test_field_order_converges_to_independent_fourier_bessel_vacuum_solution():
    radius, z = .002, .004
    a, br, bz = gaussian_vacuum_quadrature(radius, z)
    expected = np.array([a, br, bz])
    errors = []
    for order in (1, 3, 5, 7, 9):
        result = gaussian_lens(order).sample([radius, 0., z])
        actual = np.array([result["A_t_m"][1], result["B_t"][0], result["B_t"][2]])
        # Compare scaled A and B in the same unit; no tiny A hides a B error.
        errors.append(np.linalg.norm((actual-expected)/np.array([.006, 1., 1.])))
    assert all(previous > 5*current for previous, current in zip(errors, errors[1:]))
    assert errors[-1] < 1e-9


@pytest.mark.parametrize("order", [1, 3, 5, 7, 9])
def test_potential_curl_jacobian_hessian_and_exact_truncation_residual(order):
    field = gaussian_lens(order, posed=True)
    local = np.array([.0004, -.0003, .004])
    point = local@field.registration.rotation_array.T+field.registration.origin_array_m
    jet, sampled = field.sample_jet(point), field.sample(point)
    jac = jet["jac_A_t"]
    curl = np.array([jac[2, 1]-jac[1, 2], jac[0, 2]-jac[2, 0], jac[1, 0]-jac[0, 1]])
    np.testing.assert_allclose(curl, sampled["B_t"], rtol=2e-14, atol=1e-17)
    step, b_jac = 1e-7, []
    for axis, delta in enumerate(np.eye(3)*step):
        plus, minus = field.sample_jet(point+delta), field.sample_jet(point-delta)
        np.testing.assert_allclose((plus["A_t_m"]-minus["A_t_m"])/(2*step), jac[:, axis],
                                   rtol=5e-7, atol=1e-11)
        np.testing.assert_allclose((plus["jac_A_t"]-minus["jac_A_t"])/(2*step),
            jet["hess_A_t_per_m"][:, :, axis], rtol=5e-7, atol=1e-8)
        b_jac.append((field.sample(point+delta)["B_t"]-field.sample(point-delta)["B_t"])/(2*step))
    b_jac = np.array(b_jac).T
    residual = field.maxwell_residual(point)
    assert abs(np.trace(b_jac)) < 1e-7
    numerical_curl = [b_jac[2, 1]-b_jac[1, 2], b_jac[0, 2]-b_jac[2, 0], b_jac[1, 0]-b_jac[0, 1]]
    np.testing.assert_allclose(numerical_curl, residual["curl_B_t_per_m"], rtol=5e-5, atol=1e-7)
    assert residual["div_B_t_per_m"] == 0


@pytest.mark.parametrize("order", [1, 3, 5, 7, 9])
def test_unremoved_maxwell_residual_has_declared_radial_order(order):
    field = gaussian_lens(order)
    large = np.linalg.norm(field.maxwell_residual([.002, 0., .004])["curl_B_t_per_m"])
    small = np.linalg.norm(field.maxwell_residual([.001, 0., .004])["curl_B_t_per_m"])
    assert large > 0 and small > 0
    assert large/small == pytest.approx(2.**order, rel=2e-14)


@pytest.mark.parametrize("point", [[0., 0., -.08], [0., 0., .2], [.0041, 0., 0.],
                                  [0., np.nan, 0.], [0., 0.]])
def test_strict_domain_and_invalid_positions_fail(point):
    with pytest.raises(ValueError):
        gaussian_lens().sample(point)


def test_explicit_all_z_gaussian_domain_preserves_tail_without_cutting_A():
    strict = gaussian_lens()
    extended = gaussian_lens(axial_domain="gaussian_all_z")
    point = [.001, 0., -.081]
    with pytest.raises(ValueError, match="outside"):
        strict.sample(point)
    assert np.linalg.norm(extended.sample(point)["A_t_m"]) > 0
    assert extended.to_metadata()["domain"]["local_z_m"] == [None, None]
    with pytest.raises(ValueError, match="outside"):
        extended.sample([.005, 0., 1.])
    assert strict.digest != extended.digest


def test_detached_immutable_json_and_tampered_identity():
    field = gaussian_lens()
    restored = HighOrderRoundLensPotential.from_dict(json.loads(json.dumps(field.to_dict())))
    assert restored.digest == field.digest
    np.testing.assert_array_equal(restored.sample([.001, .002, .003])["B_t"], field.sample([.001, .002, .003])["B_t"])
    with pytest.raises(FrozenInstanceError):
        field.radial_order = 1
    record = field.to_dict()
    record["radial_order"] = 3
    with pytest.raises(ValueError, match="identity mismatch"):
        HighOrderRoundLensPotential.from_dict(record)
    changed = replace(field, radial_order=3)
    assert changed.digest != field.digest
    assert replace(field, valid_radius_m=.003).digest != field.digest


@pytest.mark.parametrize("order", [True, 2, 0, 11, 3.])
def test_unsupported_expansion_orders_are_not_coerced(order):
    with pytest.raises(ValueError, match="radial order"):
        gaussian_lens(order)
