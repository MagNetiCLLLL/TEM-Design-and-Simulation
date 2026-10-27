"""Independent dense checks of the shared-FFT electromagnetic residual.

Random real multiplication fields intentionally occupy the full grid spectrum;
the oracle preserves the discrete product rule rather than assuming smooth
coefficient derivatives. This checks the represented operator, not source or
continuum convergence.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from temsim.physics.potential_wave_carrier import _apply_residual


def _case(shape=(7, 9), shear=.35):
    rng = np.random.default_rng(721)
    ny, nx = shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = np.einsum("ij,jyx->iyx", np.linalg.inv([[.7, shear], [.1, 1.2]]).T,
                          np.stack((fx, fy)))
    velocity = rng.normal(scale=.17, size=(2, ny, nx))
    scalar = rng.normal(scale=.13, size=shape)
    mechanical = rng.normal(scale=.25, size=(2, ny, nx))
    defect = rng.normal(scale=.19, size=shape)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    other = rng.normal(size=shape)+1j*rng.normal(size=shape)
    fy_matrix = np.exp(-2j*np.pi*np.outer(np.arange(ny), np.arange(ny))/ny)/np.sqrt(ny)
    fx_matrix = np.exp(-2j*np.pi*np.outer(np.arange(nx), np.arange(nx))/nx)/np.sqrt(nx)
    fourier = np.kron(fy_matrix, fx_matrix)
    derivatives = [fourier.conj().T@np.diag(symbol.ravel())@fourier for symbol in frequency]
    return amplitude, other, velocity, scalar, mechanical, frequency, defect, derivatives


def _dense(velocity, scalar, mechanical, defect, derivatives, order, beta):
    result = np.diag(scalar.ravel()).astype(complex)
    full_kinetic = np.diag(defect.ravel()).astype(complex)
    for axis, derivative in enumerate(derivatives):
        v = np.diag(velocity[axis].ravel())
        result += .5*(v@derivative+derivative@v)
        kinetic = derivative+np.diag(mechanical[axis].ravel())
        full_kinetic += kinetic@kinetic
    if order >= 4:
        result += beta**3*(full_kinetic@full_kinetic)/8
    if order >= 6:
        result += beta**5*(full_kinetic@full_kinetic@full_kinetic)/16
    if order >= 8:
        result += 5*beta**7*(full_kinetic@full_kinetic@full_kinetic@full_kinetic)/128
    return result


@pytest.mark.parametrize("order,quartic_split", [(2, False), (4, False), (4, True)])
@pytest.mark.parametrize("shape,shear", [((7, 9), 0.), ((8, 6), .35)])
@pytest.mark.parametrize("beta", [.71, 1.23])
def test_shared_fft_matches_dense_full_electromagnetic_operator_and_is_hermitian(order, shape, shear, beta, quartic_split):
    amplitude, other, velocity, scalar, mechanical, frequency, defect, derivatives = _case(shape, shear)
    original = amplitude.copy()
    expected = _dense(velocity, scalar, mechanical, defect, derivatives, order, beta)
    result = _apply_residual(amplitude, velocity, scalar, mechanical, frequency, order, defect, beta,
                             quartic_scalar_splitting=quartic_split)
    if quartic_split:
        multiplication = np.sum(mechanical**2, axis=0)+defect
        expected -= beta**3*np.diag(multiplication.ravel()**2)/8
    np.testing.assert_allclose(result.ravel(), expected@amplitude.ravel(), rtol=3e-13, atol=3e-14)
    other_result = _apply_residual(other, velocity, scalar, mechanical, frequency, order, defect, beta,
                                   quartic_scalar_splitting=quartic_split)
    left, right = np.vdot(other, result), np.vdot(other_result, amplitude)
    assert abs(left-right) <= 3e-14*np.linalg.norm(other)*np.linalg.norm(result)
    np.testing.assert_array_equal(amplitude, original)


@pytest.mark.parametrize("order", [2, 4])
def test_shared_fft_complex_evolution_matches_dense_exponential_without_phase_or_flux_fitting(order):
    amplitude, _, velocity, scalar, mechanical, frequency, defect, derivatives = _case((5, 7), .28)
    beta, distance = .83, .41
    dense = _dense(velocity, scalar, mechanical, defect, derivatives, order, beta)

    def rhs(_position, flat):
        return (-1j*_apply_residual(flat.reshape(amplitude.shape), velocity, scalar,
            mechanical, frequency, order, defect, beta)).ravel()

    solution = solve_ivp(rhs, (0., distance), amplitude.ravel(), method="DOP853",
                         rtol=2e-13, atol=2e-14)
    assert solution.success
    actual, expected = solution.y[:, -1], expm(-1j*distance*dense)@amplitude.ravel()
    assert np.linalg.norm(actual-expected) < 5e-12*np.linalg.norm(expected)
    assert abs(np.vdot(actual, actual).real-np.vdot(amplitude, amplitude).real) < 5e-12


def test_quartic_scalar_strang_converges_to_complete_dense_exponential_and_preserves_full_kinetic_budget(monkeypatch):
    import temsim.physics.potential_wave_carrier as module
    from temsim.physics.multiplane_wave import PlaneWave
    amplitude, _, velocity, scalar, mechanical, frequency, defect, derivatives = _case((8, 10), .32)
    amplitude /= np.linalg.norm(amplitude)
    velocity, scalar, mechanical = velocity*.02, scalar*.07, mechanical*.08+.08
    frequency, defect = frequency*.025, defect*.001
    derivatives = [item*.025 for item in derivatives]
    beta, distance = .91, 120.
    coefficients = velocity, scalar, mechanical, frequency, defect, beta
    monkeypatch.setattr(module, "_coefficients", lambda *_a, **_k: coefficients)
    wave = PlaneWave(amplitude, np.eye(2), np.zeros(2))
    dense = _dense(velocity, scalar, mechanical, defect, derivatives, 4, beta)
    expected = expm(-1j*distance*dense)@amplitude.ravel()
    errors = []
    # Synthetic finite periodic operator: broad boundary/spectrum bounds are
    # declared only here. This is not physical wavepacket-domain qualification.
    options = (None, 2*np.pi, 4, 1e-11, 1., .99, .99, lambda: False)
    for count in (16, 32, 64):
        current = wave
        for _ in range(count):
            current, evidence = module._residual(None, current, 0., distance/count,
                *options, scalar_splitting=True, quartic_scalar_splitting=True)
        errors.append(np.linalg.norm(current.amplitude.ravel()-expected))
        assert current.probability == pytest.approx(wave.probability, abs=2e-12)
        assert evidence["quartic_scalar_exact_strang"] is True
        assert evidence["scalar_splitting_error_in_series_bound"] is False
    assert errors[1] < .27*errors[0] and errors[2] < .27*errors[1]
    _, old = module._residual(None, wave, 0., distance/64, *options, scalar_splitting=True)
    for key in ("kinetic_remainder_phase_bound", "square_root_expansion_bound",
                "kinetic_slope_squared_bound", "electrostatic_momentum_defect_bound"):
        assert evidence[key] == old[key]
    assert evidence["generator_norm_bound"] < old["generator_norm_bound"]
    assert errors[2] < 1e-7


def test_quartic_scalar_phase_must_pass_existing_sampling_and_complete_kinetic_guards(monkeypatch):
    import temsim.physics.potential_wave_carrier as module
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_grid import WaveSamplingError
    shape = (8, 10)
    x = np.linspace(0., 1., shape[1])[None, :]+np.zeros(shape)
    mechanical = np.stack((.10+.1*x, np.zeros(shape)))
    zero = np.zeros(shape)
    coefficients = np.zeros((2, *shape)), zero, mechanical, np.zeros((2, *shape)), zero, 1.
    monkeypatch.setattr(module, "_coefficients", lambda *_a, **_k: coefficients)
    wave = PlaneWave(np.ones(shape, complex)/np.sqrt(np.prod(shape)), np.eye(2), np.zeros(2))
    # U^2 has a real phase gradient even when R2 and every D_i vanish.
    with pytest.raises(WaveSamplingError, match="scalar phase"):
        module._residual(None, wave, 0., 1e5, None, 2*np.pi, 4, 1e-10, 1e5,
            .99, .99, lambda: False, scalar_splitting=True, quartic_scalar_splitting=True)
    with pytest.raises(ValueError, match="kinetic truncation"):
        module._residual(None, wave, 0., 1., None, 2*np.pi, 4, 1e-10, 1e-30,
            .99, .99, lambda: False, scalar_splitting=True, quartic_scalar_splitting=True)


@pytest.mark.parametrize("options", [dict(quartic_scalar_splitting="true"),
    dict(quartic_scalar_splitting=True),
    dict(quartic_scalar_splitting=True, scalar_splitting=True, kinetic_order=2)])
def test_public_quartic_split_requires_explicit_compatible_options(options):
    from test_potential_wave_carrier import _snapshot, _wave, ENERGY_EV
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    with pytest.raises(ValueError, match="quartic_scalar_splitting"):
        propagate_carrier_potential(_snapshot(), _wave(), (0., 1e-7), energy_ev=ENERGY_EV, **options)


def test_quartic_scalar_vacuum_with_nonzero_tilt_has_exact_complete_fourier_phase():
    from test_potential_wave_carrier import _snapshot, _wave, ENERGY_EV, WAVELENGTH
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    wave = _wave(n=48, pitch=1e-7, width=2e-7, tilt=(.001, -.0003))
    distance = 1e-6
    result, record = propagate_carrier_potential(_snapshot(), wave, (0., distance),
        energy_ev=ENERGY_EV, kinetic_order=4, scalar_splitting=True,
        quartic_scalar_splitting=True, kinetic_remainder_phase_limit=1.)
    fy, fx = np.meshgrid(np.fft.fftfreq(48), np.fft.fftfreq(48), indexing="ij")
    frequency = WAVELENGTH*np.stack((fx, fy))/1e-7
    kinetic = np.sum((frequency+wave.tilt_rad[:, None, None])**2, axis=0)
    phase = -2*np.pi/WAVELENGTH*distance*(.5*np.sum(frequency**2, axis=0)+kinetic**2/8)
    expected = np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(1j*phase))
    expected *= np.exp(1j*2*np.pi/WAVELENGTH*distance*np.sum(wave.tilt_rad**2)/2)
    np.testing.assert_allclose(result.amplitude, expected, rtol=1e-8, atol=3e-11)
    np.testing.assert_allclose(result.origin_m, wave.tilt_rad*distance, atol=1e-18)
    assert record["exact_quartic_scalar_strang"] is True
    assert record["normalization_applied"] is False
    assert all(row["quartic_scalar_exact_strang"] for row in record["residual_steps"])


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("split", [False, True])
def test_high_order_keeps_all_covariant_powers_and_is_hermitian(order, split):
    amplitude, other, velocity, scalar, mechanical, frequency, defect, derivatives = _case((7, 9), .35)
    beta = .83
    dense = _dense(velocity, scalar, mechanical, defect, derivatives, order, beta)
    if split:
        multiplication = np.sum(mechanical**2, axis=0)+defect
        for power, coefficient in ((2, 1/8), (3, 1/16), (4, 5/128)):
            if 2*power <= order:
                dense -= coefficient*beta**(2*power-1)*np.diag(multiplication.ravel()**power)
    actual = _apply_residual(amplitude, velocity, scalar, mechanical, frequency, order, defect, beta,
                             higher_order_scalar_splitting=split)
    np.testing.assert_allclose(actual.ravel(), dense@amplitude.ravel(), rtol=3e-13, atol=3e-14)
    paired = _apply_residual(other, velocity, scalar, mechanical, frequency, order, defect, beta,
                             higher_order_scalar_splitting=split)
    left, right = np.vdot(other, actual), np.vdot(paired, amplitude)
    assert abs(left-right) <= 5e-14*np.linalg.norm(other)*np.linalg.norm(actual)


@pytest.mark.parametrize("order", [6, 8])
def test_high_order_scalar_recurrence_avoids_large_power_cancellation(order):
    from itertools import product
    amplitude, _, velocity, scalar, mechanical, frequency, defect, derivatives = _case((5, 7), .35)
    frequency = frequency*1e-15
    derivatives = [item*1e-15 for item in derivatives]
    velocity, scalar = np.zeros_like(velocity), np.zeros_like(scalar)
    multiplication = np.sum(mechanical**2, axis=0)+defect
    u = np.diag(multiplication.ravel())
    linear = np.zeros_like(derivatives[0])
    for axis, derivative in enumerate(derivatives):
        m = np.diag(mechanical[axis].ravel())
        linear += derivative@derivative+derivative@m+m@derivative
    expected = np.zeros_like(amplitude.ravel())
    identity = np.eye(amplitude.size)
    # Enumerate all noncommuting L/U words with at least one L. No K^n-U^n
    # subtraction is used in this independent cancellation-sensitive oracle.
    for power, coefficient in ((2, 1/8), (3, 1/16), (4, 5/128)):
        if 2*power > order:
            continue
        remainder = np.zeros_like(linear)
        for word in product((0, 1), repeat=power):
            if not any(word):
                continue
            term = identity
            for letter in word:
                term = term@(linear if letter else u)
            remainder += term
        expected += coefficient*(remainder@amplitude.ravel())
    actual = _apply_residual(amplitude, velocity, scalar, mechanical, frequency, order, defect, 1.,
                             higher_order_scalar_splitting=True).ravel()
    assert np.linalg.norm(actual-expected) < 3e-13*np.linalg.norm(expected)


@pytest.mark.parametrize("options", [dict(kinetic_order=6, higher_order_scalar_splitting="true"),
    dict(kinetic_order=6, higher_order_scalar_splitting=True),
    dict(kinetic_order=4, scalar_splitting=True, higher_order_scalar_splitting=True),
    dict(kinetic_order=8, scalar_splitting=True, quartic_scalar_splitting=True, higher_order_scalar_splitting=True)])
def test_public_high_order_split_requires_explicit_compatible_options(options):
    from test_potential_wave_carrier import _snapshot, _wave, ENERGY_EV
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    with pytest.raises(ValueError, match="scalar_splitting"):
        propagate_carrier_potential(_snapshot(), _wave(), (0., 1e-7), energy_ev=ENERGY_EV, **options)


@pytest.mark.parametrize("order", [6, 8])
def test_higher_order_resolves_the_same_full_grid_kinetic_budget_without_weakening_it(order):
    from test_potential_wave_carrier import _snapshot, _wave, ENERGY_EV, WAVELENGTH
    from temsim.physics.potential_wave_carrier import propagate_carrier_potential
    wave = _wave(n=48, pitch=1e-7, width=2e-7, tilt=(.1, 0.))
    distance = 4e-7
    with pytest.raises(ValueError, match="kinetic truncation"):
        propagate_carrier_potential(_snapshot(), wave, (0., distance), energy_ev=ENERGY_EV,
            kinetic_order=4, scalar_splitting=True, quartic_scalar_splitting=True,
            kinetic_remainder_phase_limit=.01)
    result, record = propagate_carrier_potential(_snapshot(), wave, (0., distance),
        energy_ev=ENERGY_EV, kinetic_order=order, scalar_splitting=True,
        higher_order_scalar_splitting=True, kinetic_remainder_phase_limit=.01)
    fy, fx = np.meshgrid(np.fft.fftfreq(48), np.fft.fftfreq(48), indexing="ij")
    frequency = WAVELENGTH*np.stack((fx, fy))/1e-7
    kinetic = np.sum((frequency+wave.tilt_rad[:, None, None])**2, axis=0)
    generator = .5*np.sum(frequency**2, axis=0)
    for power, coefficient in ((2, 1/8), (3, 1/16), (4, 5/128)):
        if power*2 <= order:
            generator += coefficient*kinetic**power
    phase = -2*np.pi/WAVELENGTH*distance*generator
    expected = np.fft.ifft2(np.fft.fft2(wave.amplitude)*np.exp(1j*phase))
    expected *= np.exp(1j*2*np.pi/WAVELENGTH*distance*np.sum(wave.tilt_rad**2)/2)
    assert np.linalg.norm(result.amplitude-expected) < 2e-11
    assert record["kinetic_remainder_phase_bound"] < .01
    assert record["exact_higher_order_scalar_strang"] is True
    assert record["normalization_applied"] is False
    for row in record["residual_steps"]:
        assert row["square_root_expansion_bound"] > .01
        assert row["higher_order_scalar_exact_strang"] is True


@pytest.mark.parametrize("axis,bin_index", [(0, 40), (0, 49), (0, 50), (0, 60),
                                           (1, 40), (1, 49), (1, 50), (1, 60)])
def test_envelope_sampling_suggests_retained_signed_frequency_not_a_fixed_doubling(axis, bin_index):
    from temsim.physics.potential_wave_carrier import _check_envelope_sampling
    from temsim.physics.wave_grid import WaveSamplingError
    spectrum = np.zeros((100, 100), complex)
    spectrum[(bin_index, 0) if axis == 0 else (0, bin_index)] = 1.+.3j
    amplitude = np.fft.ifft2(spectrum)
    original = amplitude.copy()
    with pytest.raises(WaveSamplingError) as caught:
        _check_envelope_sampling(amplitude, 1e-10)
    expected = abs(np.fft.fftfreq(100)[bin_index])/.4
    assert caught.value.required_scale == max(1., expected)
    assert caught.value.required_scale <= 1.25
    np.testing.assert_array_equal(amplitude, original)


def test_envelope_sampling_cumulative_weak_bins_are_not_discarded_individually():
    from temsim.physics.potential_wave_carrier import _check_envelope_sampling
    from temsim.physics.wave_grid import WaveSamplingError
    spectrum = np.zeros((100, 100), complex)
    spectrum[0, 0] = 1.
    spectrum[0, 40:50] = np.sqrt(3e-11)
    amplitude = np.fft.ifft2(spectrum)
    # Each bin is below 1e-10, but all ten contain about 3e-10 together.
    # The four highest bins already exceed the declared tail allowance.
    with pytest.raises(WaveSamplingError) as caught:
        _check_envelope_sampling(amplitude, 1e-10)
    assert caught.value.required_scale == pytest.approx(.46/.4, rel=0., abs=2e-16)
    accepted = _check_envelope_sampling(amplitude, 4e-10)
    assert accepted == pytest.approx(3e-10/(1.+3e-10), rel=1e-10)


def test_envelope_sampling_full_band_retains_nyquist_and_never_filters_any_bin():
    from temsim.physics.potential_wave_carrier import _check_envelope_sampling
    from temsim.physics.wave_grid import WaveSamplingError
    amplitude = np.zeros((64, 96), complex)
    amplitude[32, 48] = .7+.2j  # Equal nonzero power in every signed FFT bin.
    original = amplitude.tobytes()
    with pytest.raises(WaveSamplingError) as caught:
        _check_envelope_sampling(amplitude, 1e-10)
    assert caught.value.required_scale == 1.25
    assert amplitude.tobytes() == original


def test_sampling_suggestion_retries_original_input_and_rechecks_the_complete_gate():
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.potential_wave_carrier import _check_envelope_sampling
    from temsim.physics.wave_grid import WaveGridNumerics, apply_resolved_operator
    spectrum = np.zeros((64, 64), complex)
    spectrum[0, 29] = 1.
    wave = PlaneWave(np.fft.ifft2(spectrum, norm="ortho"), np.eye(2)*1e-8, np.zeros(2))
    original = wave.amplitude.tobytes()
    attempts = []
    def check(candidate):
        attempts.append(candidate.amplitude.shape)
        _check_envelope_sampling(candidate.amplitude, 1e-10)
        return candidate
    output, records = apply_resolved_operator(wave, check,
        numerics=WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=64*1024**2))
    assert attempts == [(64, 64), (80, 80)]
    assert records[0]["requested_scale"] == (29/64)/.4
    actual = np.fft.fft2(output.amplitude, norm="ortho")
    assert actual[0, 29] == pytest.approx(1., abs=4e-15)
    assert np.linalg.norm(actual-sparse_spectrum(actual.shape, (0, 29))) < 4e-15
    assert wave.amplitude.tobytes() == original
    assert output.probability == pytest.approx(wave.probability, rel=2e-15)


def sparse_spectrum(shape, index):
    spectrum = np.zeros(shape, complex)
    spectrum[index] = 1.
    return spectrum


def test_real_cpu_gpu_envelope_sampling_suggestions_and_gate_agree():
    from temsim.physics.compute_backend import cupy_capability
    from temsim.physics.potential_wave_carrier import _check_envelope_sampling
    from temsim.physics.wave_grid import WaveSamplingError
    cp = pytest.importorskip("cupy")
    status = cupy_capability()
    if not status.available:
        pytest.skip(status.detail)
    spectrum = np.zeros((128, 128), complex)
    spectrum[0, 0] = 1.
    spectrum[0, 53:64] = np.sqrt(2e-11)
    spectrum[64, 0] = np.sqrt(4e-11)  # The actual negative Nyquist bin survives.
    amplitude = np.fft.ifft2(spectrum)
    device = cp.asarray(amplitude)
    scales = []
    for values in (amplitude, device):
        with pytest.raises(WaveSamplingError) as caught:
            _check_envelope_sampling(values, 1e-10)
        scales.append(caught.value.required_scale)
    assert scales[0] == scales[1]
    assert _check_envelope_sampling(device, 1e-9) == pytest.approx(
        _check_envelope_sampling(amplitude, 1e-9), rel=5e-12, abs=1e-24)
    np.testing.assert_array_equal(cp.asnumpy(device), amplitude)
