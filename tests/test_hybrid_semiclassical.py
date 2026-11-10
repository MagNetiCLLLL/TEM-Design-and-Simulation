"""Small mathematical operators, not finite-source microscope trials."""
from dataclasses import replace
import numpy as np
import pytest

from temsim.physics.hybrid_semiclassical import (
    SemiclassicalNumerics, decompose_mode, evaluate_packets, lifted_hk_prefactor,
    propagate_mode, momentum,
)
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import WaveMode
from temsim.optics.electron_gun.tip_coherence import TIP_REFERENCE
from test_potential_ray import uniform_snapshot


def gaussian_mode(n=48):
    x = (np.arange(n)-n//2)*1e-9
    yy, xx = np.meshgrid(x, x, indexing="ij")
    a = np.exp(-(xx*xx+yy*yy)/(4*(3e-9)**2)).astype(complex)
    a /= np.linalg.norm(a)
    return WaveMode(PlaneWave(a, np.eye(2)*1e-9, np.zeros(2)), .4, TIP_REFERENCE, "mode:0", 200.)


def test_frame_preserves_complex_amplitude_without_current_normalisation():
    mode = gaussian_mode()
    numerics = SemiclassicalNumerics(frame_stride=8, window_sigma_pixels=8., maximum_packets=65536)
    packets = decompose_mode(mode, numerics, xp=np)
    rebuilt = evaluate_packets(packets, mode.plane, xp=np, numerics=numerics)
    np.testing.assert_allclose(rebuilt, mode.plane.amplitude, atol=2e-7)
    assert packets.record["packets_are_original_electrons"] is False
    assert mode.weight_per_reference_electron == .4


def test_caustic_lift_resolves_two_transverse_foci_and_full_winding():
    t = np.linspace(0, 2*np.pi, 129)
    matrices = np.zeros((len(t), 1, 4, 4))
    for i, value in enumerate(t):
        matrices[i, 0] = np.block([[np.eye(2)*np.cos(value), np.eye(2)*np.sin(value)],
                                   [-np.eye(2)*np.sin(value), np.eye(2)*np.cos(value)]])
    prefactor, record = lifted_hk_prefactor(matrices, 1.)
    np.testing.assert_allclose(prefactor[:, 0], np.exp(-1j*t), atol=2e-14)
    assert record["continuous_prefactor_phase_rad"][0] == pytest.approx(-2*np.pi)
    with pytest.raises(ValueError, match="undersampled"):
        lifted_hk_prefactor(matrices[::16], 1.)


def test_noncanonical_map_is_rejected_instead_of_repaired():
    maps = np.repeat(np.eye(4)[None, None], 2, axis=0)
    maps[-1, 0, 0, 0] = 1.1
    with pytest.raises(ValueError, match="symplectic"):
        lifted_hk_prefactor(maps, 1.)


def test_free_action_amplitude_matches_independent_fresnel_operator():
    mode = gaussian_mode()
    numerics = SemiclassicalNumerics(frame_stride=8, window_sigma_pixels=8.,
        maximum_packets=65536, maximum_step_m=1e-6, packet_batch=128, coefficient_cutoff=1e-7)
    distance = 1e-6
    output, record = propagate_mode(mode, uniform_snapshot(), .001, .001+distance,
        mode.plane, numerics, xp=np)
    from scipy.constants import h
    wavelength = h/momentum(200000.)
    f = np.fft.fftfreq(48, 1e-9)
    expected = np.fft.ifft2(np.fft.fft2(mode.plane.amplitude)*np.exp(-1j*np.pi*wavelength*distance*(f[:, None]**2+f[None, :]**2)))
    actual = output.plane.amplitude*np.sqrt(output.weight_per_reference_electron/mode.weight_per_reference_electron)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert record["restored_current"] is False
    assert record["maximum_symplectic_residual"] < 1e-5


def test_frame_cancellation_and_memory_are_explicit():
    mode = gaussian_mode()
    with pytest.raises(InterruptedError):
        decompose_mode(mode, SemiclassicalNumerics(), xp=np, cancelled=lambda: True)
    with pytest.raises(MemoryError):
        decompose_mode(mode, SemiclassicalNumerics(maximum_working_bytes=100), xp=np)


def test_cuda_array_operator_matches_fresnel_reference():
    """Bounded GPU mathematical operator; no finite-source instrument trial."""
    from temsim.physics.compute_backend import coherent_cuda_capability, require_coherent_cuda
    capability = coherent_cuda_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    require_coherent_cuda("Require GPU")
    import cupy as cp
    from scipy.constants import h
    mode = gaussian_mode()
    numerics = SemiclassicalNumerics(frame_stride=8, window_sigma_pixels=8.,
        maximum_packets=65536, maximum_step_m=1e-6, packet_batch=128, coefficient_cutoff=1e-7)
    output, receipt = propagate_mode(mode, uniform_snapshot(), .001, .001001,
        mode.plane, numerics, xp=cp)
    f = np.fft.fftfreq(48, 1e-9)
    expected = np.fft.ifft2(np.fft.fft2(mode.plane.amplitude)*np.exp(
        -1j*np.pi*h/momentum(200000.)*1e-6*(f[:, None]**2+f[None, :]**2)))
    actual = output.plane.amplitude*np.sqrt(output.weight_per_reference_electron/.4)
    np.testing.assert_allclose(actual, expected, atol=2e-5)
    assert receipt["maximum_symplectic_residual"] < 1e-5


def test_uniform_magnetic_field_matches_paraxial_gaussian_solution():
    """Bounded operator comparison with the analytic solenoid oscillator."""
    from scipy.constants import e, hbar
    from temsim.physics.compute_backend import coherent_cuda_capability, require_coherent_cuda
    capability = coherent_cuda_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    require_coherent_cuda("Require GPU")
    import cupy as cp
    initial = gaussian_mode(n=96)
    initial = replace(initial, plane=replace(initial.plane, basis_m=np.eye(2)*1e-8))
    numerics = SemiclassicalNumerics(frame_stride=4, window_sigma_pixels=4.,
        maximum_packets=262144, maximum_step_m=1e-4, packet_batch=128, coefficient_cutoff=1e-8)
    distance, bz, sigma = 1e-3, 1., 30e-9
    output, receipt = propagate_mode(initial, uniform_snapshot(bz=bz), 0., distance,
        initial.plane, numerics, xp=cp)
    k = float(momentum(200000.)/hbar)
    omega = e*bz/(2*float(momentum(200000.)))
    gamma = 1/(2*k*sigma*sigma)
    a, b, cc, d = np.cos(omega*distance), np.sin(omega*distance)/omega, -omega*np.sin(omega*distance), np.cos(omega*distance)
    factor = a+1j*gamma*b
    xy = initial.plane.coordinates_m()
    # The circular Gaussian is unchanged by the simultaneous Larmor rotation.
    expected = (initial.plane.amplitude[48, 48]/factor
                *np.exp(.5j*k*(cc+1j*gamma*d)/factor*np.sum(xy*xy, axis=0)))
    actual = output.plane.amplitude*np.sqrt(output.weight_per_reference_electron/.4)
    np.testing.assert_allclose(actual, expected, atol=3e-5)
    assert np.linalg.norm(actual-expected) < 5e-4
    assert receipt["maximum_symplectic_residual"] < numerics.symplectic_tolerance
