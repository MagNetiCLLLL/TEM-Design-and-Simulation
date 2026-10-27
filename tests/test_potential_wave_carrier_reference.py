"""CF4 advances quadratic action; general nonlinear splitting remains second order."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.constants import e
from scipy.ndimage import map_coordinates, spline_filter

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_snapshot import StaticPolynomialGauge
from temsim.physics.potential_wave_carrier import propagate_carrier_potential
from test_potential_wave_carrier import _snapshot, _multipole, MOMENTUM, WAVELENGTH


@pytest.mark.parametrize("gauged", [False, True])
def test_varying_quadratic_reference_matches_independent_complex_gaussian_ode_with_fourth_order(gauged):
    start, stop, sigma, pitch, width = -.0006, .0006, .0002, 3e-8, 8e-8
    k, alpha = 2*np.pi/WAVELENGTH, -e/MOMENTUM
    xy = (np.arange(64)-32)*pitch
    x, y = np.meshgrid(xy, xy)
    amplitude = np.exp(-(x*x+y*y)/(4*width*width))
    amplitude = amplitude/np.linalg.norm(amplitude)*np.exp(.37j)
    origin, tilt = np.array([1e-8, -2e-8]), np.array([2e-6, -1e-6])
    q = np.eye(2)*100.
    wave = PlaneWave(amplitude, np.eye(2)*pitch, origin, q, tilt)
    terms = ((1, 0, 2e-5), (0, 1, -1e-5),
             (2, 0, 300.), (0, 2, 150.), (1, 1, 100.))
    provider = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        polynomial_terms=terms, envelope_kind="gaussian", center_z_m=0., sigma_m=sigma)))
    gauge = StaticPolynomialGauge(2e-16, (1e-11, -2e-11, 3e-11),
        ((1e-5, 2e-6, 3e-6), (2e-6, -1e-5, -4e-6), (3e-6, -4e-6, 0.))) if gauged else StaticPolynomialGauge()
    snapshot = _snapshot(provider, gauge=gauge)
    complex_q = q+1j*np.eye(2)/(2*k*width*width)
    global_tilt = tilt-complex_q@origin
    initial_log = np.log(amplitude[32, 32]/pitch)+1j*k*(.5*origin@complex_q@origin-tilt@origin)
    initial = np.r_[complex_q.ravel(), global_tilt, initial_log]
    def rhs(z, values):
        matrix, linear = values[:4].reshape(2, 2), values[4:6]
        envelope = np.exp(-.5*(z/sigma)**2)
        stiffness = -alpha*envelope*np.array(((600., 100.), (100., 300.)))
        force = -alpha*envelope*np.array((2e-5, -1e-5))
        constant = 0.
        return np.r_[(-matrix@matrix-stiffness).ravel(), -matrix@linear-force,
                     -.5*np.trace(matrix)-1j*k*(.5*linear@linear+constant)]
    solved = solve_ivp(rhs, (start, stop), initial, method="DOP853", rtol=2e-13, atol=2e-14)
    assert solved.success
    final_q, final_t, final_log = solved.y[:4, -1].reshape(2, 2), solved.y[4:6, -1], solved.y[6, -1]
    if gauged:
        point = np.r_[origin, start]
        gradient = gauge.vector_potential_shift_t_m(point)
        wave = replace(wave, curvature_m1=q+alpha*np.asarray(gauge.hessian_t)[:2, :2],
            tilt_rad=tilt+alpha*gradient[:2], amplitude=amplitude*np.exp(1j*k*alpha*gauge.chi_t_m2(point)))
    errors = []
    for count in (8, 16, 32):
        output, record = propagate_carrier_potential(snapshot, wave, np.linspace(start, stop, count+1),
            energy_ev=200000., kinetic_order=2, reference_quadrature="cf4")
        coordinates = output.coordinates_m()
        expected = np.exp(final_log+1j*k*(.5*np.einsum("iyx,ij,jyx->yx", coordinates, final_q, coordinates)
                                       +np.einsum("i,iyx->yx", final_t, coordinates)))
        expected *= np.sqrt(np.linalg.det(output.basis_m))
        if gauged:
            xyz = np.moveaxis(np.concatenate((coordinates, np.full((1, *output.amplitude.shape), stop)), axis=0), 0, -1)
            expected *= np.exp(1j*k*alpha*gauge.chi_t_m2(xyz))
        # Compare in the output's known analytical chart. Multiplying BOTH
        # fields by the same exact unit phase leaves their complete complex
        # difference unchanged and does not sample an unresolved carrier.
        delta = coordinates-output.origin_m[:, None, None]
        action = .5*np.einsum("iyx,ij,jyx->yx", delta, output.curvature_m1, delta)
        action += np.einsum("i,iyx->yx", output.tilt_rad, delta)
        errors.append(float(np.linalg.norm(output.amplitude-expected*np.exp(-1j*k*action))))
        assert output.probability == pytest.approx(1., abs=2e-12)
        assert "general nonquadratic" in record["axial_order_scope"]
    assert errors[1] < errors[0]/10 and errors[2] < errors[1]/10
    assert errors[-1] < 2e-6


def test_varying_quadratic_non_gaussian_complex_field_matches_independent_full_spectral_ode():
    from temsim.physics.wave_grid import refine_plane_wave
    start, stop, sigma, pitch, width = -.0006, .0006, .0002, 3e-8, 8e-8
    k, alpha = 2*np.pi/WAVELENGTH, -e/MOMENTUM
    xy = (np.arange(64)-32)*pitch
    x, y = np.meshgrid(xy, xy)
    amplitude = np.exp(-(x*x+y*y)/(4*width*width))*(1+.2j*x/width)
    amplitude /= np.linalg.norm(amplitude)
    wave = PlaneWave(amplitude, np.eye(2)*pitch, np.zeros(2), None, None)
    terms = ((1, 0, 2e-5), (0, 1, -1e-5), (2, 0, 300.), (0, 2, 150.), (1, 1, 100.))
    provider = AnalyticPotentialDomain(AnalyticMagneticPotential(replace(_multipole().base.field,
        polynomial_terms=terms, envelope_kind="gaussian", center_z_m=0., sigma_m=sigma)))
    snapshot = _snapshot(provider)
    # This independent fixed grid must resolve the emerging physical phase,
    # unlike the tested analytical carrier representation. Refine it first.
    fixed = refine_plane_wave(wave, (256, 256))
    x, y = fixed.coordinates_m()
    frequency = np.fft.fftfreq(256, d=pitch/4)
    kinetic = .5*WAVELENGTH**2*(frequency[:, None]**2+frequency[None, :]**2)
    spatial = -alpha*(2e-5*x-1e-5*y+300*x*x+150*y*y+100*x*y)
    def rhs(z, values):
        field = values.reshape(256, 256)
        return (-1j*k*(np.fft.ifft2(kinetic*np.fft.fft2(field))
            +np.exp(-.5*(z/sigma)**2)*spatial*field)).ravel()
    reference = solve_ivp(rhs, (start, stop), fixed.amplitude.ravel(), method="DOP853", rtol=2e-13, atol=2e-15)
    assert reference.success
    refined = refine_plane_wave(replace(fixed, amplitude=reference.y[:, -1].reshape(256, 256)), (512, 512))
    coefficients = [spline_filter(part, order=5, mode="constant")
                    for part in (refined.amplitude.real, refined.amplitude.imag)]
    errors = []
    for count in (8, 16, 32):
        output, _ = propagate_carrier_potential(snapshot, wave, np.linspace(start, stop, count+1),
            energy_ev=200000., kinetic_order=2, reference_quadrature="cf4")
        coordinates = output.coordinates_m()
        indices = np.einsum("ij,jyx->iyx", np.linalg.inv(refined.basis_m), coordinates)+256
        expected = sum(multiplier*map_coordinates(values, indices[::-1], order=5,
            mode="constant", cval=0., prefilter=False)
            for multiplier, values in zip((1., 1j), coefficients))
        expected *= np.sqrt(np.linalg.det(output.basis_m)/np.linalg.det(refined.basis_m))
        delta = coordinates-output.origin_m[:, None, None]
        action = .5*np.einsum("iyx,ij,jyx->yx", delta, output.curvature_m1, delta)
        action += np.einsum("i,iyx->yx", output.tilt_rad, delta)
        errors.append(float(np.linalg.norm(output.amplitude-expected*np.exp(-1j*k*action))))
    assert errors[1] < errors[0]/10 and errors[2] < errors[1]/10
    assert errors[-1] < 2e-6


@pytest.mark.parametrize("ulps", [1, 2])
def test_cf4_tiny_segment_next_to_compact_face_keeps_exact_planes_without_endpoint_kick(ulps):
    from test_potential_wave_carrier import _wave
    face = 1.02
    after = face
    for _ in range(ulps):
        after = np.nextafter(after, np.inf)
    native = replace(_multipole().base.field, native_support_mm=(1019., 1020.),
                     polynomial_terms=((1, 0, -1e-4),))
    field = AnalyticPotentialDomain(AnalyticMagneticPotential(native))
    wave = _wave(n=32, width=2e-7)
    output, record = propagate_carrier_potential(_snapshot(field), wave, (face, after),
        energy_ev=200000., kinetic_order=2, reference_quadrature="cf4")
    assert record["axial_nodes_m"] == [face, after]
    np.testing.assert_array_equal(output.tilt_rad, wave.tilt_rad)
    before = np.nextafter(face, -np.inf)
    inside, _ = propagate_carrier_potential(_snapshot(field), wave, (before, face),
        energy_ev=200000., kinetic_order=2, reference_quadrature="cf4")
    np.testing.assert_allclose(inside.tilt_rad, (e/MOMENTUM*1e-4*(face-before), 0.), atol=1e-30, rtol=1e-13)
