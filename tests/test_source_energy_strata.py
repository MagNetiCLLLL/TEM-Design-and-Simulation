"""Source-measure tests only; no physical gun or optical transport evidence."""
from dataclasses import replace
import math

import numpy as np
import pytest
from scipy.special import roots_legendre
from scipy.stats import qmc

from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.tip_joint_sampling import calibrate_energy, energy_parameters
from scripts.source_energy_strata import energy_band_cdf_intervals, expand_energy_strata


def emitter(**kwargs):
    return ColdFieldEmitter(0., 1., 1., **kwargs)


def energy(source, young, boersch):
    cal = calibrate_energy(*energy_parameters(source))
    return cal.mean_ev+cal.offsets(young, boersch)


def test_inverse_band_cuts_agree_with_direct_physical_energy_thresholds():
    source = emitter()
    young = np.array([0., .001, .03, .2, .5, .9, 1.])
    band = (.143, .149)
    cuts = energy_band_cdf_intervals(source, young, band)
    v = (np.arange(65536)+.5)/65536
    values = energy(source, young[:, None], v[None, :])
    np.testing.assert_allclose(cuts.lower_cdf, (values < band[0]).mean(axis=1), rtol=0, atol=1/65536)
    np.testing.assert_allclose(cuts.upper_cdf, (values <= band[1]).mean(axis=1), rtol=0, atol=1/65536)
    for cut, threshold in zip((cuts.lower_cdf, cuts.upper_cdf), band):
        active = (cut > 1e-10)&(cut < 1-1e-10)
        lo, hi = cut[active].copy(), cut[active].copy()
        for _ in range(8):
            lo, hi = np.nextafter(lo, 0.), np.nextafter(hi, 1.)
        # In the far tail dE/dCDF is large; an energy-only absolute tolerance
        # would reject a quantile inverse accurate to a few float64 ulps.
        assert np.all(energy(source, young[active], lo) <= threshold+2e-15)
        assert np.all(energy(source, young[active], hi) >= threshold-2e-15)


def test_full_measure_original_coordinates_and_exact_count_prefix():
    source = emitter()
    original = qmc.Sobol(5, scramble=True, seed=17, bits=30).random_base2(8)
    before = original.copy()
    small = expand_energy_strata(source, original[:64], (.143, .149))
    large = expand_energy_strata(source, original, (.143, .149))
    n = len(small.base_points)
    for name in ("base_points", "independent_base_index", "stratum_index", "interval_lower", "interval_upper"):
        np.testing.assert_array_equal(getattr(small, name), getattr(large, name)[:n])
    np.testing.assert_array_equal(small.base_probability*64, large.base_probability[:n]*256)
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(large.base_points[:, :4], original[large.independent_base_index, :4])
    np.testing.assert_array_equal(large.base_points[:, 4], large.interval_lower+
        (large.interval_upper-large.interval_lower)*original[large.independent_base_index, 4])
    np.testing.assert_allclose(np.bincount(large.independent_base_index, weights=large.base_probability),
                               np.full(256, 1/256), rtol=0, atol=3e-18)
    assert math.fsum(large.base_probability) == pytest.approx(1., abs=3e-16)
    assert np.all(large.base_probability > 0)
    values = energy(source, large.base_points[:, 3], large.base_points[:, 4])
    assert np.isfinite(values).all() and np.all(values > 0)
    assert np.all(values[large.stratum_index == 0] < .143+2e-15)
    assert np.all(values[large.stratum_index == 1] >= .143-2e-15)
    assert np.all(values[large.stratum_index == 1] <= .149+2e-15)
    assert np.all(values[large.stratum_index == 2] > .149-2e-15)
    with pytest.raises(ValueError):
        large.base_points[0, 0] = .5


@pytest.mark.parametrize("which,young", [("minimum", .01), ("maximum", 1.)])
def test_shape_clipping_atom_is_preserved_inside_point_energy_band(which, young):
    source = emitter(energy_half_range_ev=.15, energy_spread_fwhm_ev=.15)
    endpoint = 0. if which == "minimum" else 1.
    atom = float(energy(source, np.array([young]), np.array([endpoint]))[0])
    cuts = energy_band_cdf_intervals(source, [young], [atom, atom])
    assert cuts.interval_probability[0, 1] > 1e-5
    if which == "minimum":
        assert cuts.lower_cdf[0] == 0.
    else:
        assert cuts.upper_cdf[0] == 1.
    v = (np.arange(262144)+.5)/262144
    direct_energy = energy(source, young, v)
    # Independently count both strict and inclusive events in the actual map.
    assert cuts.lower_cdf[0] == pytest.approx(np.mean(direct_energy < atom), abs=1/262144)
    assert cuts.upper_cdf[0] == pytest.approx(np.mean(direct_energy <= atom), abs=1/262144)
    direct_mass = np.mean(direct_energy == atom)
    assert cuts.interval_probability[0, 1] == pytest.approx(direct_mass, abs=1/262144)
    np.testing.assert_allclose(cuts.interval_probability[0],
        [np.mean(direct_energy < atom), direct_mass, np.mean(direct_energy > atom)], rtol=0, atol=1/262144)


def test_quantile_clipping_atoms_and_young_endpoint_clipping_are_preserved():
    source = emitter(energy_half_range_ev=10.)
    young = .5
    lo, hi = energy(source, young, np.array([0., 1.]))
    low = energy_band_cdf_intervals(source, [young], [lo, lo])
    high = energy_band_cdf_intervals(source, [young], [hi, hi])
    assert low.interval_probability[0, 1] == pytest.approx(1e-12, abs=5e-26)
    assert high.interval_probability[0, 1] == pytest.approx(1e-12, abs=2e-16)
    a = energy_band_cdf_intervals(source, [0., 1e-15], [.143, .149])
    np.testing.assert_array_equal(a.edges[0], a.edges[1])


@pytest.mark.parametrize("kwargs", [dict(boersch_sigma_ev=0.), dict(energy_spread_fwhm_ev=0.),
    dict(boersch_sigma_ev=0., young_decay_width_ev=0.)])
def test_constant_conditional_energy_has_one_original_mass_without_artificial_split(kwargs):
    source = emitter(**kwargs)
    original = qmc.Sobol(5, scramble=True, seed=3, bits=30).random_base2(4)
    result = expand_energy_strata(source, original, [.143, .149])
    assert len(result.base_points) == len(original)
    np.testing.assert_array_equal(result.base_points, original)
    np.testing.assert_array_equal(result.base_probability, np.full(16, 1/16))


@pytest.mark.parametrize("kwargs", [dict(boersch_sigma_ev=0.), dict(energy_spread_fwhm_ev=0.),
    dict(boersch_sigma_ev=0., young_decay_width_ev=0.)])
def test_constant_conditional_energy_point_band_uses_both_strict_and_inclusive_cuts(kwargs):
    source = emitter(**kwargs)
    young = .3
    v = np.linspace(0., 1., 257)
    actual = energy(source, young, v)
    np.testing.assert_array_equal(actual, np.full(len(v), actual[0]))
    point = float(actual[0])
    for band in ([point, point], [.8*point, .9*point], [1.1*point, 1.2*point]):
        cuts = energy_band_cdf_intervals(source, [young], band)
        assert cuts.lower_cdf[0] == np.mean(actual < band[0])
        assert cuts.upper_cdf[0] == np.mean(actual <= band[1])
        np.testing.assert_array_equal(cuts.interval_probability[0],
            [np.mean(actual < band[0]), np.mean((actual >= band[0])&(actual <= band[1])),
             np.mean(actual > band[1])])


def test_zero_width_nonatomic_band_has_zero_mass_and_preserves_both_complements():
    original = np.array([[.4, .2, .7, .5, .3]])
    result = expand_energy_strata(emitter(), original, [.146, .146])
    assert result.cuts.lower_cdf[0] == result.cuts.upper_cdf[0]
    assert result.cuts.interval_probability[0, 1] == 0
    np.testing.assert_array_equal(result.stratum_index, [0, 2])
    assert math.fsum(result.base_probability) == 1.


@pytest.mark.parametrize("band,stratum", [([.001, .002], 2), ([.001, 2.], 1), ([2., 3.], 0)])
def test_outside_or_enclosing_band_drops_only_zero_mass_rows(band, stratum):
    original = qmc.Sobol(5, scramble=True, seed=3, bits=30).random_base2(4)
    result = expand_energy_strata(emitter(), original, band)
    np.testing.assert_array_equal(result.base_points, original)
    np.testing.assert_array_equal(result.stratum_index, np.full(16, stratum))
    np.testing.assert_array_equal(result.base_probability, np.full(16, 1/16))
    assert result.metadata["omitted_zero_probability_strata"] == 32


def test_common_azimuth_expansion_preserves_nonuniform_original_energy_mass():
    from scripts.source_azimuth_orbit import expand_common_azimuth
    original = qmc.Sobol(5, scramble=True, seed=11, bits=30).random_base2(5)
    energy_rows = expand_energy_strata(emitter(), original, [.143, .149])
    orbit = expand_common_azimuth(energy_rows.base_points, 8)
    # Orbit's uniform radial_base_weight is not appropriate after stratification.
    probability = energy_rows.base_probability[orbit.independent_base_index]/4
    parent = energy_rows.independent_base_index[orbit.independent_base_index]
    np.testing.assert_allclose(np.bincount(parent, weights=probability), np.full(32, 1/32), rtol=0, atol=1e-17)
    np.testing.assert_array_equal(orbit.base_points[:, 3:], energy_rows.base_points[orbit.independent_base_index, 3:])
    assert math.fsum(probability) == pytest.approx(1., abs=3e-16)
    assert math.fsum(np.repeat(probability/2, 2)) == pytest.approx(1., abs=3e-16)


def test_weighted_energy_moments_and_nonlinear_observable_against_independent_cdf_integral():
    source = emitter()
    nodes, weights = roots_legendre(1024)
    u, w = (nodes+1)/2, np.outer(weights, weights)/4
    e = energy(source, u[:, None], u[None, :])
    reference = np.array([np.sum(w*e), np.sum(w*e**2), np.sum(w*np.exp(-e/.2))])
    bases = qmc.Sobol(5, scramble=True, seed=17, bits=30).random_base2(16)
    estimates = []
    for count in (1024, 65536):
        result = expand_energy_strata(source, bases[:count], [.143, .149])
        actual = energy(source, result.base_points[:, 3], result.base_points[:, 4])
        estimates.append(np.array([np.sum(result.base_probability*actual),
            np.sum(result.base_probability*actual**2), np.sum(result.base_probability*np.exp(-actual/.2))]))
    error = np.abs(estimates[-1]-reference)
    assert np.all(error < np.abs(estimates[0]-reference))
    np.testing.assert_allclose(estimates[-1], reference, rtol=0, atol=3e-5)


@pytest.mark.parametrize("band", [[.149, .143], [0., .1], [.1, float("nan")], [.1], [.1, .2, .3]])
def test_invalid_physical_energy_band_rejected(band):
    with pytest.raises(ValueError, match="band"):
        energy_band_cdf_intervals(emitter(), [.2], band)


@pytest.mark.parametrize("points", [np.empty((0, 5)), np.zeros((3, 4)), np.full((3, 5), np.nan), np.full((3, 5), 1.1)])
def test_invalid_original_cdf_bases_rejected(points):
    with pytest.raises(ValueError, match="bases"):
        expand_energy_strata(emitter(), points, [.143, .149])
