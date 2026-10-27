"""Independent analytic-mixture checks for the physical Gaussian renderer."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from temsim.detector.compact_gaussian_response import compact_gaussian_response
from temsim.detector.point_spread import DetectorPointSpread


def psf(sx=.6, sy=.6, angle=0.):
    return DetectorPointSpread('gaussian', sx, sy, angle,
                               'provisional_model_parameter', 'analytic fixture')


def exact_mixture(points, weights, x_edges, y_edges, spread):
    """Independent sums of the few actual Gaussian point-source integrals."""
    answer = np.zeros((len(y_edges)-1, len(x_edges)-1))
    def mass(edges, centre, sigma):
        a, b = (edges[:-1]-centre)/sigma, (edges[1:]-centre)/sigma
        return np.where(a > 0, ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
    for point, weight in zip(points, weights):
        answer += weight*np.outer(mass(y_edges, point[1], spread.sigma_y_mm),
                                  mass(x_edges, point[0], spread.sigma_x_mm))
    return answer


def corners(width=.0001):
    return np.array([[-width, -width], [-width, width], [width, -width], [width, width]])


@pytest.mark.parametrize('pixels', [32, 512])
@pytest.mark.parametrize('kind', ['variance', 'anisotropic', 'centroid'])
def test_replicated_particles_preserve_real_width_and_position_response(pixels, kind):
    base = corners()
    changed = {'variance': base*1.1,
               'anisotropic': base*np.array([1.1, .9]),
               'centroid': base+np.array([.00001, -.00002])}[kind]
    edges = np.linspace(-2.4, 2.4, pixels+1)
    response, exact = [], []
    for points in (base, changed):
        repeated = np.repeat(points, 10000, axis=0)
        weights = np.full(len(repeated), 1/len(repeated))
        value = compact_gaussian_response(repeated, weights, edges, edges, psf())
        assert value is not None
        response.append(value.probability)
        exact.append(exact_mixture(points, np.full(4, .25), edges, edges, psf()))
        simple = compact_gaussian_response(points, np.full(4, .25), edges, edges, psf())
        np.testing.assert_allclose(value.probability, simple.probability, rtol=2e-15, atol=1e-20)
        assert value.metadata['total_weight'] == 1.
        np.testing.assert_allclose(value.metadata['centre_mm'], points.mean(axis=0),
                                   atol=2*np.finfo(float).eps*np.abs(points).max())
        assert value.metadata['global_l1_truncation_bound'] < 1e-25
    reference = exact[1]-exact[0]
    signal = np.abs(reference).sum()
    relative_error = np.abs((response[1]-response[0])-reference).sum()/signal
    assert signal > 1e-9
    assert relative_error < 2e-5


def test_unequal_weights_shifted_cropped_anisotropic_psf_matches_exact_integrals():
    points = corners()+np.array([.37, -.24])
    weights = np.array([.03, .07, .11, .19])
    original_points, original_weights = points.copy(), weights.copy()
    xe, ye = np.linspace(.1, .8, 65), np.linspace(-.8, -.1, 49)
    spread = psf(.4, .9)
    value = compact_gaussian_response(points, weights, xe, ye, spread)
    assert value is not None
    expected = exact_mixture(points, weights, xe, ye, spread)
    assert np.abs(value.probability-expected).sum() < 1e-14
    assert 0 < value.probability.sum() < weights.sum()
    np.testing.assert_allclose(value.metadata['centre_mm'], np.average(points, axis=0, weights=weights), atol=1e-16)
    centre = np.asarray(value.metadata['centre_mm'])
    covariance = (points-centre).T@(weights[:, None]*(points-centre))/weights.sum()
    np.testing.assert_allclose(value.metadata['central_covariance_mm2'], covariance, rtol=2e-12, atol=1e-22)
    np.testing.assert_array_equal(points, original_points)
    np.testing.assert_array_equal(weights, original_weights)


def test_crop_does_not_discard_sources_outside_the_view_or_renormalise_flux():
    points, weights = corners()+[1.3, .2], np.array([.1, .2, .05, .15])
    spread = psf()
    xe, ye = np.linspace(-.2, .2, 65), np.linspace(-.3, .3, 65)
    assert np.all(points[:, 0] > xe[-1])
    value = compact_gaussian_response(points, weights, xe, ye, spread)
    expected = exact_mixture(points, weights, xe, ye, spread)
    assert value is not None and 0 < value.probability.sum() < .02
    np.testing.assert_allclose(value.probability, expected, rtol=2e-11, atol=1e-19)


def test_positive_tail_integrals_do_not_cancel_to_zero():
    points, weights = np.array([[.2, -.3]]), np.array([.4])
    spread = psf(.6, .9)
    xe = points[0, 0]+spread.sigma_x_mm*np.linspace(9., 9.2, 9)
    ye = points[0, 1]+spread.sigma_y_mm*np.linspace(8., 8.1, 7)
    value = compact_gaussian_response(points, weights, xe, ye, spread)
    assert value is not None and np.all(value.probability > 0.)
    normal = lambda t: math.exp(-.5*t*t)/math.sqrt(2*math.pi)
    mx = np.array([quad(normal, a, b, epsabs=1e-100, epsrel=2e-13)[0]
                   for a, b in zip((xe[:-1]-.2)/.6, (xe[1:]-.2)/.6)])
    my = np.array([quad(normal, a, b, epsabs=1e-100, epsrel=2e-13)[0]
                   for a, b in zip((ye[:-1]+.3)/.9, (ye[1:]+.3)/.9)])
    np.testing.assert_allclose(value.probability, .4*np.outer(my, mx), rtol=3e-12, atol=0.)


def test_far_tail_polynomials_remain_finite_without_floating_warnings():
    with np.errstate(all='raise'):
        value = compact_gaussian_response(corners(), np.full(4, .25),
            np.array([1e100, 2e100, 3e100]), np.array([1e100, 2e100]), psf())
    assert value is not None
    np.testing.assert_array_equal(value.probability, np.zeros((1, 2)))


@pytest.mark.parametrize('kind', ['broad', 'none', 'tiny_sigma', 'huge_sigma'])
def test_unsupported_domain_returns_none_for_existing_fallback(kind):
    points, spread = corners(), psf()
    if kind == 'broad':
        points *= 10
    elif kind == 'none':
        spread = replace(spread, model='none')
    elif kind == 'tiny_sigma':
        spread = psf(1e-300, .6)
    else:
        spread = psf(1e300, .6)
    with np.errstate(all='raise'):
        assert compact_gaussian_response(points, np.full(4, .25), [-1., 0., 1.], [-1., 1.], spread) is None


def test_zero_weight_nan_record_has_no_physical_influence():
    points = np.vstack((corners(), [np.nan, np.nan]))
    weights = np.array([.1, .2, .3, .4, 0.])
    edges = np.linspace(-2.4, 2.4, 33)
    actual = compact_gaussian_response(points, weights, edges, edges, psf())
    expected = compact_gaussian_response(points[:4], weights[:4], edges, edges, psf())
    np.testing.assert_array_equal(actual.probability, expected.probability)
    assert actual.metadata == expected.metadata


def test_no_positive_weight_returns_zero_without_inventing_a_centroid():
    value = compact_gaussian_response([[np.nan, np.nan]], [0.], [-1., 1.], [-1., 1.], psf())
    np.testing.assert_array_equal(value.probability, [[0.]])
    assert value.metadata['centre_mm'] is None
    assert value.metadata['total_weight'] == 0.


def test_reported_global_truncation_bound_is_weighted_whitened_radius_bound():
    points, weights, spread = corners()+[.2, -.1], np.array([.1, .2, .3, .1]), psf(.7, .3)
    value = compact_gaussian_response(points, weights, [-1., 1.], [-1., 1.], spread)
    centre = np.asarray(value.metadata['expansion_centre_mm'])
    radius = np.linalg.norm((points-centre)/[spread.sigma_x_mm, spread.sigma_y_mm], axis=1)
    bound = np.sum(weights*radius**7)/math.sqrt(math.factorial(7))
    assert value.metadata['global_l1_truncation_bound'] == pytest.approx(bound, rel=1e-11)
    assert 'floating-point' in value.metadata['bound_scope']


@pytest.mark.parametrize('angle', [17., 37., 90., 270., -123.])
def test_isotropic_rotation_does_not_change_physical_response(angle):
    points, weights = corners(), np.full(4, .25)
    edges = np.linspace(-2.4, 2.4, 33)
    actual = compact_gaussian_response(points, weights, edges, edges, psf(angle=angle))
    reference = exact_mixture(points, weights, edges, edges, psf())
    assert actual is not None
    np.testing.assert_allclose(actual.probability, reference, rtol=3e-12, atol=1e-18)


@pytest.mark.parametrize('angle', [90., 270., -90.])
def test_anisotropic_quarter_turn_exchanges_physical_sigma_axes(angle):
    points, weights = corners()+[.2, -.1], np.full(4, .25)
    edges = np.linspace(-2.4, 2.4, 33)
    actual = compact_gaussian_response(points, weights, edges, edges, psf(.4, .9, angle))
    reference = exact_mixture(points, weights, edges, edges, psf(.9, .4))
    assert actual is not None
    assert actual.metadata['laboratory_sigma_mm'] == [.9, .4]
    np.testing.assert_allclose(actual.probability, reference, rtol=3e-12, atol=1e-18)


def test_scalar_weight_has_an_explicit_shape_error():
    with pytest.raises(ValueError, match='matching Nx2'):
        compact_gaussian_response([[0., 0.]], 1., [-1., 1.], [-1., 1.], psf())


def test_extreme_weight_total_falls_back_without_an_overflow_exception():
    with np.errstate(all='raise'):
        assert compact_gaussian_response([[0., 0.], [0., 0.]], [1e308, 1e308],
            [-1., 1.], [-1., 1.], psf()) is None


def test_extreme_finite_edges_fall_back_before_standardised_coordinate_overflow():
    with np.errstate(all='raise'):
        assert compact_gaussian_response(corners()+[1e308, 0.], np.full(4, .25),
            [-1e308, 1e308], [-1., 1.], psf()) is None


@pytest.mark.parametrize('pixels', [32, 512])
def test_weak_symmetric_width_signal_agrees_with_independent_density_quadrature(pixels):
    """Stable exact Gaussian identity, without using a moment-series reference."""
    a, b, sigma = .0001, .0001*(1.+1e-5), .6
    alpha, beta = a/sigma, b/sigma
    edges = np.linspace(-2.4, 2.4, pixels+1)
    t = edges/sigma
    delta_square = (beta-alpha)*(beta+alpha)
    def difference_density(value):
        phi = math.exp(-.5*value*value)/math.sqrt(2*math.pi)
        difference_cosh = 2.*math.sinh(.5*(beta+alpha)*value)*math.sinh(.5*(beta-alpha)*value)
        return phi*math.exp(-.5*alpha*alpha)*(math.cosh(alpha*value)*math.expm1(-.5*delta_square)
            + math.exp(-.5*delta_square)*difference_cosh)
    dm = np.array([quad(difference_density, lo, hi, epsabs=1e-30, epsrel=1e-12)[0]
                   for lo, hi in zip(t[:-1], t[1:])])
    m = .5*(np.diff(ndtr(t-alpha))+np.diff(ndtr(t+alpha)))
    exact_delta = np.outer(dm, m)+np.outer(m, dm)+np.outer(dm, dm)
    images = []
    for width in (a, b):
        points = np.repeat(corners(width), 10000, axis=0)
        value = compact_gaussian_response(points, np.full(len(points), 1/len(points)), edges, edges, psf())
        assert value.metadata['expansion_centre_selection'] == 'viewport midpoint'
        images.append(value.probability)
    signal = np.abs(exact_delta).sum()
    assert 1e-13 < signal < 1e-12
    assert np.abs((images[1]-images[0])-exact_delta).sum()/signal < .01


def test_expansion_origin_is_not_a_physical_centroid_subtraction():
    edges = np.linspace(-2.4, 2.4, 65)
    points, weights = corners()+[.0001, -.00005], np.full(4, .25)
    value = compact_gaussian_response(points, weights, edges, edges, psf())
    assert value.metadata['expansion_centre_mm'] == [0., 0.]
    np.testing.assert_allclose(value.metadata['centre_mm'], [.0001, -.00005], atol=1e-20)
    expected = exact_mixture(points, weights, edges, edges, psf())
    zero_centred = exact_mixture(corners(), weights, edges, edges, psf())
    assert np.abs(value.probability-expected).sum() < 1e-14
    assert np.abs(value.probability-zero_centred).sum() > 1e-5


def test_noncompact_viewport_origin_uses_actual_mean_without_losing_offset():
    points, weights = corners()+[.8, -.3], np.full(4, .25)
    edges = np.linspace(-2.4, 2.4, 65)
    value = compact_gaussian_response(points, weights, edges, edges, psf())
    assert value.metadata['expansion_centre_selection'] == 'weighted mean'
    np.testing.assert_allclose(value.metadata['expansion_centre_mm'], [.8, -.3], atol=1e-16)
    np.testing.assert_allclose(value.probability, exact_mixture(points, weights, edges, edges, psf()),
                               rtol=3e-11, atol=1e-18)


@pytest.mark.parametrize('pixels', [32, 64])
def test_opposite_skewness_survives_equal_mean_and_covariance(pixels):
    spread = psf()
    a = .0004*spread.sigma_x_mm
    points = np.column_stack(([-a, 0., 2*a], np.zeros(3)))
    weights = np.array([.4, .4, .2])
    edges = np.linspace(-2.4, 2.4, pixels+1)
    exact, actual = [], []
    for value in (points, -points):
        repeated = np.repeat(value, 10000, axis=0)
        repeated_weights = np.repeat(weights/10000, 10000)
        response = compact_gaussian_response(repeated, repeated_weights, edges, edges, spread)
        assert response is not None
        np.testing.assert_allclose(response.metadata['centre_mm'], [0., 0.], atol=1e-19)
        actual.append(response.probability)
        exact.append(exact_mixture(value, weights, edges, edges, spread))
    np.testing.assert_array_equal(points.T@(weights[:, None]*points),
                                  (-points).T@(weights[:, None]*(-points)))
    assert weights@points[:, 0]**3 != 0.
    delta = exact[1]-exact[0]
    assert np.abs(delta).sum() > 1e-11
    assert np.abs((actual[1]-actual[0])-delta).sum()/np.abs(delta).sum() < .001
