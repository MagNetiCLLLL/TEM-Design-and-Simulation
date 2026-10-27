"""Independent Gaussian pixel masses, continuity and conservative PSF sampling."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from temsim.detector import gaussian_pixel_response as gaussian
from temsim.detector.point_spread import DetectorPointSpread
from temsim.detector.receiver_image import _response_from_hits


def _psf(angle=45.):
    return DetectorPointSpread("gaussian", .025, .020, angle,
                               "provisional_model_parameter", "analytic Gaussian fixture")


def _rectangle(position, bounds, psf):
    """Independent conditional-normal adaptive quadrature, not production code."""
    angle = math.radians(psf.rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    vx = (c*psf.sigma_x_mm)**2 + (s*psf.sigma_y_mm)**2
    vy = (s*psf.sigma_x_mm)**2 + (c*psf.sigma_y_mm)**2
    xy = c*s*(psf.sigma_x_mm**2-psf.sigma_y_mm**2)
    conditional = math.sqrt(vy-xy*xy/vx)
    x0, x1, y0, y1 = bounds
    def integrand(x):
        mu = position[1] + xy/vx*(x-position[0])
        return (math.exp(-.5*(x-position[0])**2/vx)/math.sqrt(2*math.pi*vx)
                * (ndtr((y1-mu)/conditional)-ndtr((y0-mu)/conditional)))
    return quad(integrand, x0, x1, epsabs=2e-14, epsrel=2e-13,
                points=[position[0]] if x0 < position[0] < x1 else None)[0]


@pytest.mark.parametrize("angle", [0., 1e-6, 45., 90., -37.])
def test_coarse_rotated_pixels_are_physical_integrals(angle):
    edges = np.arange(-8, 9)*.112
    position, psf = np.array([[.03, .03]]), _psf(angle)
    values, _, _ = _response_from_hits(position, np.array([.7]), edges, edges, psf)
    expected = .7*_rectangle(position[0], (0., .112, 0., .112), psf)
    assert values[8, 8] == pytest.approx(expected, abs=2e-14)
    assert values.sum() == pytest.approx(.7, abs=2e-14)


def test_rotation_does_not_switch_to_an_artificial_display_blur():
    edges = np.arange(-8, 9)*.112
    position, weights = np.array([[.03, .03]]), np.array([1.])
    before, _, _ = _response_from_hits(position, weights, edges, edges, _psf(0.))
    after, _, _ = _response_from_hits(position, weights, edges, edges, _psf(1e-6))
    assert np.abs(after-before).sum() < 1e-9
    assert after[8, 8]-before[8, 8] == pytest.approx(1.95525e-10, abs=2e-14)


def _tiny_rectangle_density(position, bounds, psf):
    """Independent double integral of density, avoiding small CDF differences."""
    angle = math.radians(psf.rotation_deg)
    c, s = math.cos(angle), math.sin(angle)
    def density(y, x):
        dx, dy = x-position[0], y-position[1]
        a, b = c*dx+s*dy, -s*dx+c*dy
        return math.exp(-.5*((a/psf.sigma_x_mm)**2+(b/psf.sigma_y_mm)**2))/(2*math.pi*psf.sigma_x_mm*psf.sigma_y_mm)
    x0, x1, y0, y1 = bounds
    return quad(lambda x: quad(density, y0, y1, args=(x,), epsabs=1e-25,
                               epsrel=2e-13)[0], x0, x1, epsabs=1e-25, epsrel=2e-13)[0]


def test_1024_tiny_view_keeps_compact_integrals_and_rotation_continuity():
    position = np.array([[.030123, .030456]])
    edges = np.linspace(-1e-5, 1e-5, 1025)
    probabilities = []
    for angle in (0., 1e-6, 45.):
        psf = _psf(angle)
        values, method, pitch = _response_from_hits(position, np.array([.7]), edges, edges, psf)
        expected = .7*_tiny_rectangle_density(position[0], (-1e-5, 1e-5, -1e-5, 1e-5), psf)
        assert "Compact" in method and pitch is None
        assert values.sum() == pytest.approx(expected, rel=2e-11)
        probabilities.append(values.sum())
    expected_change = (_tiny_rectangle_density(position[0], (-1e-5, 1e-5, -1e-5, 1e-5), _psf(1e-6))
                       / _tiny_rectangle_density(position[0], (-1e-5, 1e-5, -1e-5, 1e-5), _psf(0.))-1.)
    assert probabilities[1]/probabilities[0]-1. == pytest.approx(expected_change, abs=2e-11)


def test_low_order_rule_requires_a_resolved_derivative_bound():
    safe = gaussian._short_interval_rule(np.array([1.]), np.array([1.+1e-6]),
        np.ones(1), -.2, .3, .25, None)
    unsafe = gaussian._short_interval_rule(np.array([1.]), np.array([1.5]),
        np.ones(1), -.2, .3, .25, None)
    assert safe[0] and not unsafe[0]


def test_compact_output_work_is_tiled_without_increasing_quadrature_budget(monkeypatch):
    position, weights = np.array([[.030123, .030456]]), np.array([.7])
    edges = np.linspace(-1e-5, 1e-5, 65)
    expected, _, _ = _response_from_hits(position, weights, edges, edges, _psf())
    # Two evaluations for each requested output pixel alone exceed this
    # budget. Each deterministic tile must still obey the same bound.
    monkeypatch.setattr(gaussian, "DIRECT_EVALUATION_BUDGET", 2048)
    actual, method, pitch = _response_from_hits(position, weights, edges, edges, _psf())
    assert "Compact" in method and pitch is None
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=1e-25)


def test_large_compact_cloud_retains_moments_and_rotation_continuity():
    from temsim.detector.compact_gaussian_response import compact_gaussian_response
    offsets = np.array([[-2e-6, -1e-6], [2e-6, -1e-6], [-2e-6, 1e-6], [2e-6, 1e-6]])
    corners = np.array([.03, .03])+offsets
    corner_weights = np.array([.07, .14, .21, .28])
    positions = np.repeat(corners, 25_000, axis=0)
    weights = np.repeat(corner_weights/25_000, 25_000)
    edges = np.arange(-8, 9)*.112
    before, method, pitch = _response_from_hits(positions, weights, edges, edges, _psf(0.))
    after, rotated_method, rotated_pitch = _response_from_hits(positions, weights, edges, edges, _psf(1e-6))
    assert "Compact" in method and "Compact" in rotated_method
    assert pitch is None and rotated_pitch is None
    assert np.abs(after-before).sum() < 1e-9
    rotated = compact_gaussian_response(positions, weights, edges, edges, _psf())
    expected = sum(w*_rectangle(p, (0., .112, 0., .112), _psf())
                   for p, w in zip(corners, corner_weights))
    assert rotated.probability[8, 8] == pytest.approx(expected, abs=3e-14)
    centroid = np.average(corners, axis=0, weights=corner_weights)
    covariance = np.einsum("i,ia,ib->ab", corner_weights/corner_weights.sum(),
                           corners-centroid, corners-centroid)
    np.testing.assert_allclose(rotated.metadata["centre_mm"], centroid, atol=1e-16)
    np.testing.assert_allclose(rotated.metadata["central_covariance_mm2"], covariance, atol=1e-24)
    assert rotated.metadata["global_l1_truncation_bound"] < 1e-28
    assert rotated.metadata["whitening"] == "Cholesky of rotated covariance"


@pytest.mark.parametrize("grid", [False, True])
def test_pixel_subdivision_conserves_rotated_mass_and_edge_loss(monkeypatch, grid):
    if grid:
        monkeypatch.setattr(gaussian, "DIRECT_EVALUATION_BUDGET", 0)
    psf = _psf()
    positions, weights = np.array([[.098, .007], [-.027, -.015]]), np.array([.3, .2])
    original = positions.copy(), weights.copy()
    edges = np.linspace(-.1, .1, 17)
    fine_edges = np.linspace(-.1, .1, 65)
    coarse, method, _ = _response_from_hits(positions, weights, edges, edges, psf)
    fine, _, _ = _response_from_hits(positions, weights, fine_edges, fine_edges, psf)
    assert ("Bounded" in method) == grid
    np.testing.assert_allclose(fine.reshape(16, 4, 16, 4).sum(axis=(1, 3)), coarse, atol=2e-15)
    expected = sum(w*_rectangle(p, (-.1, .1, -.1, .1), psf) for p, w in zip(positions, weights))
    assert coarse.sum() == pytest.approx(expected, rel=2e-4 if grid else 1e-12)
    assert .2 < coarse.sum() < weights.sum()
    np.testing.assert_array_equal(positions, original[0])
    np.testing.assert_array_equal(weights, original[1])


@pytest.mark.parametrize("grid", [False, True])
def test_nonuniform_mixture_retains_centroid_and_rotated_covariance(monkeypatch, grid):
    if grid:
        monkeypatch.setattr(gaussian, "DIRECT_EVALUATION_BUDGET", 0)
    psf = _psf()
    positions = np.array([[.0123, -.0204], [-.0317, .0186]])
    weights = np.array([.14, .56])
    edges = np.linspace(-.3, .3, 257)
    values, method, _ = _response_from_hits(positions, weights, edges, edges, psf)
    assert ("Bounded" in method) == grid
    centres = .5*(edges[:-1]+edges[1:])
    xx, yy = np.meshgrid(centres, centres)
    coordinates = np.stack((xx, yy), axis=-1)
    measured = np.sum(values[..., None]*coordinates, axis=(0, 1))/values.sum()
    expected = np.average(positions, axis=0, weights=weights)
    np.testing.assert_allclose(measured, expected, atol=2e-7 if grid else 2e-14)
    covariance = np.einsum("ij,ija,ijb->ab", values/values.sum(), coordinates-measured, coordinates-measured)
    displacement = positions-expected
    particle_covariance = np.einsum("i,ia,ib->ab", weights/weights.sum(), displacement, displacement)
    angle = math.radians(psf.rotation_deg)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    optical_covariance = rotation @ np.diag([psf.sigma_x_mm**2, psf.sigma_y_mm**2]) @ rotation.T
    # Pixel-centre second moments include the well-resolved quantization variance.
    reference = particle_covariance+optical_covariance+np.eye(2)*(edges[1]-edges[0])**2/12
    np.testing.assert_allclose(covariance, reference, atol=4e-7 if grid else 2e-13)
    assert values.sum() == pytest.approx(weights.sum(), abs=2e-13)


def test_physical_grid_does_not_change_with_coarse_display_pitch(monkeypatch):
    monkeypatch.setattr(gaussian, "DIRECT_EVALUATION_BUDGET", 0)
    position, weights, psf = np.array([[.03, .03]]), np.array([1.]), _psf()
    edges = np.arange(-8, 9)*.112
    values, _, pitch = _response_from_hits(position, weights, edges, edges, psf)
    expected = _rectangle(position[0], (0., .112, 0., .112), psf)
    assert pitch == pytest.approx((.020/24, .020/24))
    assert values[8, 8] == pytest.approx(expected, rel=4e-4)


def test_duplicate_hits_share_exact_integrals_without_changing_weight():
    edges = np.arange(-8, 9)*.112
    positions = np.repeat([[.03, .03]], 100_000, axis=0)
    weights = np.full(len(positions), .7/len(positions))
    response, method, pitch = _response_from_hits(positions, weights, edges, edges, _psf())
    expected, _, _ = _response_from_hits(positions[:1], np.array([weights.sum()]), edges, edges, _psf())
    assert "Bounded" not in method and pitch is None
    np.testing.assert_allclose(response, expected, atol=2e-12)


def test_excessive_physical_grid_fails_instead_of_coarsening_psf(monkeypatch):
    monkeypatch.setattr(gaussian, "DIRECT_EVALUATION_BUDGET", 0)
    positions = np.array([[-28., -28.], [28., 28.]])
    edges = np.linspace(-29., 29., 513)
    with pytest.raises(ValueError, match="bounded physical response grid"):
        _response_from_hits(positions, np.array([.5, .5]), edges, edges, _psf())


def test_strong_anisotropy_has_resolved_conditional_pixel_integrals():
    psf = replace(_psf(), sigma_y_mm=.0005, rotation_deg=31.)
    position = np.array([[.003, -.001]])
    edges = np.linspace(-.05, .05, 9)
    response, _, _ = _response_from_hits(position, np.array([1.]), edges, edges, psf)
    for iy, ix in [(3, 3), (4, 4), (3, 4)]:
        expected = _rectangle(position[0], (edges[ix], edges[ix+1], edges[iy], edges[iy+1]), psf)
        assert response[iy, ix] == pytest.approx(expected, abs=3e-13)
