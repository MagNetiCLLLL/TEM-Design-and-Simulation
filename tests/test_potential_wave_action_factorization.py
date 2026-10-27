"""Independent old-expression and dense checks of factored H6/H8 actions.

These tests preserve the original two-action recurrence as a reference. Dense
noncommuting L/U words provide a second oracle, including near-zero gradients
where subtracting full kinetic powers would lose the differential remainder.
"""
from itertools import product
import gc
import tracemalloc
import weakref

import numpy as np
import pytest

from temsim.physics import potential_wave_carrier as carrier
from temsim.physics import potential_wave_device as device
from test_potential_wave_carrier_fft import _case
from test_potential_wave_chebyshev import _input
from test_potential_wave_residual_memory import _input as _smooth_input


def _original_split_action(amplitude, velocity, scalar, mechanical, frequency,
        kinetic_order, momentum_defect=0., kinetic_scale=1., *,
        quartic_scalar_splitting=False, higher_order_scalar_splitting=False):
    """Original unfactored recurrence, independent of production helpers."""
    assert higher_order_scalar_splitting and kinetic_order in (6, 8)
    assert not quartic_scalar_splitting
    spectrum = np.fft.fft2(amplitude)
    derivatives = tuple(np.fft.ifft2(symbol*spectrum) for symbol in frequency)
    divergence = sum(frequency[axis]*np.fft.fft2(velocity[axis]*amplitude)
                     for axis in range(2))
    result = scalar*amplitude+.5*(np.fft.ifft2(divergence)
        +sum(velocity[axis]*derivatives[axis] for axis in range(2)))
    del divergence
    squared_frequency = np.sum(frequency**2, axis=0)
    multiplication = np.sum(mechanical**2, axis=0)+momentum_defect

    def kinetic(value, transformed=None, differentiated=None, *, full=True):
        if transformed is None:
            transformed = np.fft.fft2(value)
        if differentiated is None:
            differentiated = tuple(np.fft.ifft2(symbol*transformed) for symbol in frequency)
        total = squared_frequency*transformed
        for axis in range(2):
            total += frequency[axis]*np.fft.fft2(mechanical[axis]*value)
        if not full:
            return np.fft.ifft2(total)+sum(mechanical[axis]*differentiated[axis] for axis in range(2))
        return (np.fft.ifft2(total)+multiplication*value
                +sum(mechanical[axis]*differentiated[axis] for axis in range(2)))

    remainder = kinetic(amplitude, spectrum, derivatives, full=False)
    del spectrum, derivatives
    scalar_power = amplitude
    for power, coefficient in ((2, 1/8), (3, 1/16), (4, 5/128)):
        if 2*power > kinetic_order:
            break
        scalar_power = multiplication*scalar_power
        # Keep both original actions. Do not factor this reference together.
        remainder = kinetic(remainder)+kinetic(scalar_power, full=False)
        result += coefficient*kinetic_scale**(2*power-1)*remainder
    return result


def _word_matrix(velocity, scalar, mechanical, defect, derivatives, order, beta):
    """Enumerate every noncommuting word; never subtract K^n minus U^n."""
    result = np.diag(scalar.ravel()).astype(complex)
    linear = np.zeros_like(derivatives[0])
    for axis, derivative in enumerate(derivatives):
        v = np.diag(velocity[axis].ravel())
        m = np.diag(mechanical[axis].ravel())
        result += .5*(v@derivative+derivative@v)
        linear += derivative@derivative+derivative@m+m@derivative
    multiplication = np.diag((np.sum(mechanical**2, axis=0)+defect).ravel())
    for power, coefficient in ((2, 1/8), (3, 1/16), (4, 5/128)):
        if 2*power > order:
            break
        for word in product((False, True), repeat=power):
            if not any(word):
                continue
            term = np.eye(linear.shape[0], dtype=complex)
            for use_linear in word:
                term = term@(linear if use_linear else multiplication)
            result += coefficient*beta**(2*power-1)*term
    return result


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("shape", [(5, 7), (6, 8)])
@pytest.mark.parametrize("gradient_scale,magnetic_scale", [(1., 1.), (1e-15, 1.), (1e-6, 8.)])
def test_noncommuting_action_matches_original_and_dense_words_with_strong_multiplication(
        order, shape, gradient_scale, magnetic_scale):
    a, b, velocity, scalar, mechanical, frequency, defect, derivatives = _case(shape, .35)
    frequency *= gradient_scale
    derivatives = [value*gradient_scale for value in derivatives]
    mechanical *= magnetic_scale
    # Expose the tiny differential remainder without hiding it under an
    # unrelated scalar/velocity term in the strong-U cases.
    if gradient_scale != 1.:
        scalar = np.zeros_like(scalar)
        velocity = np.zeros_like(velocity)
    beta = .83
    args = (a, velocity, scalar, mechanical, frequency, order, defect, beta)
    originals = [np.asarray(value).copy() for value in args]
    expected = _original_split_action(*args, higher_order_scalar_splitting=True)
    actual = carrier._apply_residual(*args, higher_order_scalar_splitting=True)
    dense = _word_matrix(velocity, scalar, mechanical, defect, derivatives, order, beta)
    oracle = (dense@a.ravel()).reshape(shape)
    scale = np.linalg.norm(oracle)
    assert np.linalg.norm(actual-oracle) <= 3e-13*scale
    assert np.linalg.norm(actual-expected) <= 3e-13*scale
    paired = carrier._apply_residual(b, *args[1:], higher_order_scalar_splitting=True)
    assert abs(np.vdot(b, actual)-np.vdot(paired, a)) <= 5e-14*np.linalg.norm(b)*scale
    for before, value in zip(originals, args):
        np.testing.assert_array_equal(before, value)
    assert not np.shares_memory(actual, a)


@pytest.mark.parametrize("order,forward,inverse", [(6, 11, 10), (8, 14, 13)])
def test_factorization_removes_one_six_transform_action_per_retained_power(monkeypatch, order, forward, inverse):
    a, _, velocity, scalar, mechanical, frequency, defect, _ = _case((7, 9))
    counts = {"fft2": 0, "ifft2": 0}
    for name in counts:
        original = getattr(np.fft, name)
        def wrapped(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(np.fft, name, wrapped)
    carrier._apply_residual(a, velocity, scalar, mechanical, frequency, order, defect,
                            higher_order_scalar_splitting=True)
    assert counts == {"fft2": forward, "ifft2": inverse}


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_complete_exponential_matches_original_without_phase_fit_and_reverses(monkeypatch, order, method, sign):
    args, options, _ = _input(order, split=True, kappa=3.25)
    options.update(exponential_method=method, k_distance=sign*options["k_distance"],
                   phase=sign*options["phase"])
    initial = args[0].copy()
    with monkeypatch.context() as patch:
        patch.setattr(carrier, "_apply_residual", _original_split_action)
        expected, old_record = device.execute_residual_exponential(*args, **options)
    actual, record = device.execute_residual_exponential(*args, **options)
    assert np.linalg.norm(actual-expected) < 4e-13
    assert abs(np.vdot(actual, actual).real-1.) < 4e-13
    assert record["actual_operator_applications"] == old_record["actual_operator_applications"]
    assert record["original_flux_tolerance_unchanged"]
    if method == "chebyshev":
        assert record["chebyshev_certificate"] == old_record["chebyshev_certificate"]
    reverse = {**options, "k_distance": -options["k_distance"], "phase": -options["phase"]}
    restored, _ = device.execute_residual_exponential(actual, *args[1:], **reverse)
    assert np.linalg.norm(restored-initial) < 8e-13
    np.testing.assert_array_equal(args[0], initial)


@pytest.mark.parametrize("order", [6, 8])
@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
def test_complete_physical_carrier_keeps_raw_phase_norm_and_admission(monkeypatch, order, method):
    from test_potential_wave_carrier import _wave, _round, _multipole, _snapshot, _full_phase, ENERGY_EV
    wave = _wave(n=48, curvature=.2, tilt=(2e-7, -1e-7))
    snapshot = _snapshot(_round(), _multipole())
    options = dict(energy_ev=ENERGY_EV, kinetic_order=order, scalar_splitting=True,
                   higher_order_scalar_splitting=True, residual_exponential_method=method)
    with monkeypatch.context() as patch:
        patch.setattr(carrier, "_apply_residual", _original_split_action)
        expected, before = carrier.propagate_carrier_potential(snapshot, wave, (0., 1e-6, 2e-6), **options)
    actual, after = carrier.propagate_carrier_potential(snapshot, wave, (0., 1e-6, 2e-6), **options)
    assert np.linalg.norm(_full_phase(actual)-_full_phase(expected)) < 3e-13
    assert abs(actual.probability-expected.probability) < 3e-13
    for key in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(actual, key), getattr(expected, key))
    for key in ("series_amplitude_error_bound", "kinetic_remainder_phase_bound",
                "normalization_applied", "kinetic_order"):
        assert before[key] == after[key]
    assert all(row["boundary_check"] == "each bounded exponential substep" for row in after["residual_steps"])


@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
def test_cancelled_full_exponential_releases_action_outputs_and_preserves_input(monkeypatch, method):
    args, options, _ = _input(8, split=True, kappa=3.25)
    initial = args[0].copy()
    observed = []
    operation = carrier._apply_residual
    def track(*values, **kwargs):
        result = operation(*values, **kwargs)
        observed.append(weakref.ref(result))
        return result
    monkeypatch.setattr(carrier, "_apply_residual", track)
    options.update(exponential_method=method, cancelled=lambda: len(observed) >= 2)
    with pytest.raises(InterruptedError, match="cancelled"):
        device.execute_residual_exponential(*args, **options)
    gc.collect()
    assert len(observed) == 2 and all(value() is None for value in observed)
    np.testing.assert_array_equal(args[0], initial)


@pytest.mark.parametrize("shape", [(127, 131), (256, 256)])
def test_action_host_peak_does_not_add_a_retained_lattice(shape):
    amplitude, velocity, scalar, mechanical, frequency, defect, beta = _smooth_input(shape)
    args = amplitude, velocity, scalar, mechanical, frequency, 8, defect, beta
    def measure(operation):
        operation(*args, higher_order_scalar_splitting=True)
        tracemalloc.start()
        try:
            output = operation(*args, higher_order_scalar_splitting=True)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return output, peak
    expected, old_peak = measure(_original_split_action)
    actual, new_peak = measure(carrier._apply_residual)
    assert np.linalg.norm(actual-expected) < 3e-13*np.linalg.norm(expected)
    # Input coefficients are excluded, as in both measurements. Fixed Python
    # overhead is allowed, but even one extra retained complex grid is not.
    assert new_peak <= old_peak+8192
    assert new_peak <= 224*amplitude.size+1024**2


@pytest.mark.parametrize("method", ["taylor", "chebyshev"])
def test_complete_residual_host_peak_stays_within_unchanged_executor_budget(monkeypatch, method):
    from temsim.physics.wave_magnetic_residual import _series_plan
    args = _smooth_input((127, 131))
    cells = args[0].size
    count, degree, error = _series_plan(50.*.025, 1e-12)
    options = dict(phase=args[2]*25, scalar_splitting=True, kinetic_order=8,
        quartic_scalar_splitting=False, higher_order_scalar_splitting=True,
        count=count, degree=degree, k_distance=50., boundary_limit=1e-9,
        envelope_limit=1e-10, cancelled=lambda: False, requested="CPU",
        exponential_method=method, generator_norm_bound=.025,
        series_error_budget=error, maximum_chebyshev_group=2,
        maximum_host_working_bytes=1024*cells)
    def measure(operation):
        with monkeypatch.context() as patch:
            patch.setattr(carrier, "_apply_residual", operation)
            device.execute_residual_exponential(*args, **options)
            tracemalloc.start()
            try:
                output, record = device.execute_residual_exponential(*args, **options)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        return output, record, peak
    expected, before, old_peak = measure(_original_split_action)
    actual, after, new_peak = measure(carrier._apply_residual)
    assert before["exponential_method"] == after["exponential_method"] == method
    assert np.linalg.norm(actual-expected) < 3e-13
    assert abs(np.vdot(actual, actual).real-1.) < 3e-13
    assert new_peak <= old_peak+8192
    # Prepared host inputs remain separately owned. This bound covers new
    # executor allocations only, without changing either production estimate.
    assert new_peak <= device.RESIDUAL_HOST_BYTES_PER_CELL*cells
