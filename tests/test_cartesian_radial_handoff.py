from dataclasses import replace

import numpy as np
import pytest

from temsim.physics.cartesian_radial_handoff import (
    CartesianRadialNumerics, CartesianRadialUnavailable, cartesian_to_radial,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveMemoryBudgetError


def executed_fixture(*, phase=.73, rotation=.37, anisotropy=0., size=64):
    """Independent complex radial function; deliberately not a Gaussian fit."""
    pitch, sigma = 4e-7, 1.5e-6
    axis = (np.arange(size)-size//2)*pitch
    x, y = np.meshgrid(axis, axis)
    r2 = (x*x+y*y)/(sigma*sigma)
    density = ((1-.12*r2)+.025j*r2*r2)*np.exp(-r2/2)*np.exp(1j*phase)
    density *= np.exp(anisotropy*(x*x-y*y)/(sigma*sigma))
    basis = pitch*np.array(((np.cos(rotation), -np.sin(rotation)),
                            (np.sin(rotation), np.cos(rotation))))
    return PlaneWave(density*pitch/sigma, basis, np.zeros(2), np.eye(2)*1234.)


def convert(wave, **kwargs):
    return cartesian_to_radial(wave, wavelength_m=2e-12, samples=4096, **kwargs)


def test_complete_complex_field_global_phase_and_carrier_are_not_fitted():
    wave = executed_fixture()
    radial, evidence = convert(wave)
    assert radial.angular_order == 0
    assert radial.curvature_m1 == 1234.
    assert radial.probability == pytest.approx(wave.probability, rel=1e-8)
    # Test an independently defined non-Gaussian COMPLEX shape, retaining
    # its varying phase, global phase, absolute amplitude and zero crossing.
    r2 = (radial.radius_m/1.5e-6)**2
    expected = ((1-.12*r2)+.025j*r2*r2)*np.exp(-r2/2)*np.exp(.73j)/1.5e-6
    assert np.sqrt(2*np.pi*radial.dln*np.sum(abs(radial.radius_m*(radial.amplitude-expected))**2)
                   /wave.probability) < 1e-6
    assert evidence["relative_complex_roundtrip_error"] < 1e-6
    assert not evidence["normalisation_correction"]
    assert not evidence["phase_fit"]
    assert not evidence["angular_average"]
    assert len(evidence["angular_checks"]) == 2
    shifted, _ = convert(executed_fixture(phase=1.23))
    np.testing.assert_allclose(shifted.amplitude, radial.amplitude*np.exp(.5j), rtol=1e-6, atol=1e-6)


def test_radial_replay_reads_same_original_field_without_renormalizing():
    wave = executed_fixture()
    low, _ = convert(wave)
    high, evidence = cartesian_to_radial(wave, wavelength_m=2e-12, samples=8192)
    np.testing.assert_allclose(high.radius_m[::2], low.radius_m, rtol=1e-13)
    np.testing.assert_allclose(high.amplitude[::2], low.amplitude, rtol=1e-12, atol=1e-10)
    assert evidence["input_norm"] == wave.probability
    assert high.probability == pytest.approx(low.probability, rel=1e-8)


@pytest.mark.parametrize("change", ["offset", "tilt", "curvature", "lattice", "envelope"])
def test_nonaxisymmetric_fields_decline_without_mutation(change):
    wave = executed_fixture(anisotropy=.15 if change == "envelope" else 0.)
    if change == "offset":
        wave = replace(wave, origin_m=np.array((1e-9, 0.)))
    elif change == "tilt":
        wave = replace(wave, tilt_rad=np.array((1e-6, 0.)))
    elif change == "curvature":
        wave = replace(wave, curvature_m1=np.diag((1234., 1235.)))
    elif change == "lattice":
        wave = replace(wave, basis_m=wave.basis_m@np.diag((1., 1.1)))
    before = wave.amplitude.copy()
    with pytest.raises(CartesianRadialUnavailable):
        convert(wave)
    np.testing.assert_array_equal(wave.amplitude, before)


def test_square_truncation_cannot_be_silently_discarded():
    wave = executed_fixture(size=16)
    with pytest.raises(CartesianRadialUnavailable, match="square-domain boundary"):
        convert(wave)


def test_budget_and_cancellation_prevent_publication():
    wave = executed_fixture()
    with pytest.raises(WaveMemoryBudgetError, match="handoff needs"):
        convert(wave, numerics=CartesianRadialNumerics(maximum_working_bytes=1024))
    with pytest.raises(CartesianRadialUnavailable, match="sample budget"):
        convert(wave, numerics=CartesianRadialNumerics(maximum_samples=2048))
    with pytest.raises(InterruptedError):
        convert(wave, cancelled=lambda: True)


def test_validation_and_empty_field():
    wave = executed_fixture()
    with pytest.raises(ValueError):
        convert(wave, numerics=CartesianRadialNumerics(complex_tolerance=0.))
    with pytest.raises(CartesianRadialUnavailable, match="empty"):
        convert(replace(wave, amplitude=np.zeros_like(wave.amplitude)))


def test_anisotropic_envelope_phase_is_not_hidden_by_equal_current():
    wave = executed_fixture()
    x, y = wave.coordinates_m()
    wave = replace(wave, amplitude=wave.amplitude*np.exp(.2j*x*y/(1.5e-6)**2))
    with pytest.raises(CartesianRadialUnavailable) as rejected:
        convert(wave)
    assert rejected.value.evidence["relative_complex_roundtrip_error"] > .01
    assert max(row["relative_complex_error"] for row in rejected.value.evidence["angular_checks"]) > .01


def test_large_spherical_carrier_remains_analytical():
    wave = replace(executed_fixture(), curvature_m1=np.eye(2)*1e8)
    radial, _ = convert(wave)
    assert radial.curvature_m1 == 1e8
