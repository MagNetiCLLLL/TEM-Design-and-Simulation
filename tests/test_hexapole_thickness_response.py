"""Finite Gaussian hexapole response, not an OEM corrector qualification.

The measured cubic *exit slope* coefficient is positive in the laboratory
coordinates used here. It must not be confused with the negative image-plane
Cs after the corrector's relay. Both arise from action along a finite field.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.optics.hexapole import HexapoleComponent
from temsim.physics.core import propagate


def isolated_hexapole_state(width_mm, peak_m3, *, orientation_rad=0.):
    """One physical distributed component, with no other aberration source."""
    hexapole = HexapoleComponent(
        name="Finite-field analytical fixture", key="fixture_hexapole",
        z_mm=2., strength_m3=peak_m3, maximum_strength_m3=1e10,
        effective_length_mm=width_mm, enabled=True, colour="#000000",
        mechanical_center_from_tip_mm=2., mechanical_length_mm=3.,
        mechanical_outer_diameter_mm=5., mechanical_clear_bore_diameter_mm=3.,
        optical_reference_from_tip_mm=2., orientation_rad=orientation_rad,
    ).validate()
    return SimpleNamespace(
        lenses=[], stigmators=[], corrector_elements=[hexapole], deflectors=[],
        apertures=[], recording_planes=[], beam_voltage_kv=300., step_mm=.01,
        history_step_mm=.01, acceleration_enabled=False, acceleration_backend="CPU",
        simulation_mode="ideal", projector_mode="diffraction",
        equivalent_image_lenses_enabled=False, sample=SimpleNamespace(z_mm=0.),
        objective_image_plane_z_mm=4., simulation_time_s=0.,
        lens_intrinsic_aberrations={},
    )


def _exit_slope_coefficients(state, maximum_step_mm):
    radius = 1e-4
    x = radius*np.array((1., -1., .5, -.5))
    zero = np.zeros_like(x)
    checkpoints = propagate(
        state, 0., 4., x, zero, zero, zero,
        include_spherical_aberration=False, include_hexapole=True,
        maximum_step_mm=maximum_step_mm,
        checkpoint_z_mm=(4.,), return_checkpoints=True,
    )[-1]
    slopes = checkpoints.tx_rad[-1]
    radii = radius*np.array((1., .5))
    even = (slopes[::2]+slopes[1::2])/(2*radii**2)
    odd = (slopes[::2]-slopes[1::2])/(2*radii**3)
    # Parity isolates the quadratic/cubic actions. Extrapolation removes the
    # leading fourth/fifth-order contamination without fitting a display Cs.
    return np.array(((4*even[1]-even[0])/3, (4*odd[1]-odd[0])/3))


@pytest.mark.parametrize("policy", ("fixed_peak", "fixed_integral"))
def test_finite_thickness_response_matches_independent_gaussian_integrals(policy):
    widths = np.array((.1, .2, .4))
    measured = []
    for width in widths:
        peak = 1e8 if policy == "fixed_peak" else 1e8*.2/width
        state = isolated_hexapole_state(width, peak)
        sigma = width/2.355*1e-3
        # From x''=-h(z)x^2 on y=0, perturbation in entrance x gives
        # a2=-integral(h dz), a3=2*integral[h(z)*integral((z-s)h(s) ds) dz].
        # Evaluating these Gaussian integrals independently gives the values
        # below; the production integrator cuts the negligible tail at 7 sigma.
        expected = np.array((-peak*np.sqrt(2*np.pi)*sigma,
                             4*np.sqrt(np.pi)*peak**2*sigma**3))
        coarse = _exit_slope_coefficients(state, .01)
        fine = _exit_slope_coefficients(state, .005)
        np.testing.assert_allclose(fine, expected, rtol=5e-7, atol=1e-5)
        assert abs(fine[1]-expected[1]) < .1*abs(coarse[1]-expected[1])
        assert abs(coarse[1]/fine[1]-1.) < 5e-6
        measured.append(fine)
    measured = np.asarray(measured)
    if policy == "fixed_peak":
        np.testing.assert_allclose(measured[:, 0]/measured[0, 0], widths/widths[0], rtol=1e-8)
        np.testing.assert_allclose(measured[:, 1]/measured[0, 1], (widths/widths[0])**3, rtol=5e-7)
    else:
        np.testing.assert_allclose(measured[:, 0], measured[0, 0], rtol=1e-8)
        np.testing.assert_allclose(measured[:, 1]/measured[0, 1], widths/widths[0], rtol=5e-7)


def test_reversing_hexapole_changes_primary_sign_but_not_self_combination():
    positive = _exit_slope_coefficients(isolated_hexapole_state(.2, 1e8), .005)
    negative = _exit_slope_coefficients(isolated_hexapole_state(.2, -1e8), .005)
    assert positive[0] < 0. and positive[1] > 0.
    np.testing.assert_allclose(negative, positive*np.array((-1., 1.)), rtol=1e-8)


@pytest.mark.parametrize("policy", ("fixed_peak", "fixed_integral"))
def test_measured_system_cs_tracks_finite_field_not_legacy_display_coefficient(policy):
    from temsim.optics.system_residual_aberrations import measure_system_residual

    measured = []
    for width in (.1, .2, .4):
        peak = 1e8 if policy == "fixed_peak" else 1e8*.2/width
        state = isolated_hexapole_state(width, peak)
        state.image_aberrations = {"c3_mm": 123.}
        sigma = width/2.355*1e-3
        lower, upper = .002-7*sigma, .002+7*sigma

        def field(z):
            return peak*np.exp(-.5*((z-.002)/sigma)**2)

        def second_order_displacement(z):
            return quad(lambda s: (z-s)*field(s)*s*s, lower, z,
                        epsabs=1e-18, epsrel=1e-11)[0]

        # For rays leaving the sample with angle alpha, r0(z)=z*alpha.
        # The third-order transverse acceleration is radial and equals
        # 2*h(z)*z*I2(z)*alpha*|alpha|^2. Integrate its lever arm to the
        # screen and use dr=-1e-3*Cs_mm*alpha*|alpha|^2 (J_img=identity).
        # This independent quadrature includes varying ray height across a
        # real finite field; the simpler parallel-ray L^3 law is not assumed.
        expected_cs_mm = -2000*quad(
            lambda z: (.004-z)*field(z)*z*second_order_displacement(z),
            lower, upper, epsabs=1e-18, epsrel=1e-11,
        )[0]
        report = measure_system_residual(
            state, "image", semiangle_mrad=20., maximum_step_mm=.005,
        )
        assert report.coefficients.c3_mm == pytest.approx(expected_cs_mm, rel=1e-5)
        assert report.coefficients.c3_mm < 0.
        assert report.diagnostics["holdout_rms_m"] < 1e-13
        assert not report.diagnostics["coefficients_applied"]
        assert state.image_aberrations == {"c3_mm": 123.}
        measured.append(report.coefficients.c3_mm)
    assert np.all(np.diff(measured) < 0.)
