"""Budgeted complex Fourier projection for segmented potential propagation.

The deterministic quadrature keeps selected *complex* Fourier coefficients,
their physical period and the analytical carrier. It does not draw particles
from an intensity map, replace the phase, or renormalize a retained mode. The
receipt measures only the initial projection; a later discretized propagator
still requires independent convergence tests on its changed lattice.
"""
from dataclasses import replace
import math

import numpy as np
from scipy.fft import next_fast_len

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import (
    WaveGridNumerics, _FOURIER_TILE_CELLS, _centred_inverse_phase,
    _cpu_chunked_probability, _inplace_axis_fft,
)


SPECTRAL_PROJECTION_SCOPE = (
    "initial same-period Fourier projection only; subsequent discretized "
    "propagation needs independent convergence"
)
GUARD_NYQUIST_FRACTION = .76


def _budget(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or not np.isreal(value)):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return value


def spectral_resampling_working_bytes(shape):
    """Bound private stage arrays, including input and immutable publication.

    Input plus private FFT plus reduced output use at most 48 bytes/input cell.
    Input plus output, immutable publication and finite check use at most 49.
    Marginals, signed frequencies, masks and FFT line workspace are bounded by
    1152 bytes per longest axis. The same bounded single-worker FFT tiles as
    lossless refinement use an additional 8 MiB reserve. This is not process
    RSS: callers must declare other retained buffers separately.
    """
    return 49*math.prod(shape)+8*1024**2+1152*max(shape)


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Potential-wave spectral resampling cancelled")


def _frequency(count):
    values = np.arange(count, dtype=np.int64)
    values[(count+1)//2:] -= count
    return values


def _axis_candidate(marginal, frequency, tail_budget, guard):
    shell = np.bincount(abs(frequency), weights=marginal)
    tails = np.cumsum(shell[::-1], dtype=float)[::-1]
    occupied = np.flatnonzero(tails > tail_budget)
    radius = int(occupied[-1]) if occupied.size else 0
    # The extra integer bin keeps the represented core strictly below the
    # guard, including either sign and the even negative-Nyquist convention.
    count = max(32, math.ceil(2*(radius+1)/guard))
    count += count % 2
    count = int(next_fast_len(count))
    while count % 2:
        count = int(next_fast_len(count+1))
    if count >= len(marginal):
        return len(marginal), (int(frequency.min()), int(frequency.max()))
    return count, (-radius, radius)


def _bands(count, interval):
    """Two unshifted slices for one contiguous signed-frequency interval."""
    lower, upper = interval
    negative = slice(lower, None) if lower else slice(0, 0)
    return ((slice(0, upper+1), slice(0, upper+1)),
            (slice(count+lower, count), negative))


def _even_fast_count(count):
    count = max(32, math.ceil(count))
    count += count % 2
    count = int(next_fast_len(count))
    while count % 2:
        count = int(next_fast_len(count+1))
    return count


def _pressure_candidate_shapes(shape, saved, cancelled):
    """Descending fixed-aspect family, not a global two-dimensional optimum.

    Scale the longest axis through even FFT-friendly lengths, rounding the
    other axis upward in the same way. Each axis retains at least 32 samples
    (or its unchanged shorter input length). This gives nested rectangular
    passbands, so the largest admitted family member also loses the least
    probability in this family. An unchanged input axis keeps its full band.
    """
    longest = max(shape)
    count, previous = 32, None
    candidates = []
    while count < longest:
        _cancel(cancelled)
        candidate = tuple(min(original, _even_fast_count(original*count/longest))
                          for original in shape)
        if candidate != previous and math.prod(candidate) <= (1.-saved)*math.prod(shape):
            candidates.append(candidate)
        previous = candidate
        count = _even_fast_count(count+1)
    return reversed(candidates)


def _pressure_selection(shape, frequencies, guard, saved, admissible_shape, cancelled):
    for candidate in _pressure_candidate_shapes(shape, saved, cancelled):
        _cancel(cancelled)
        admitted = admissible_shape(candidate)
        if not isinstance(admitted, (bool, np.bool_)):
            raise ValueError("Spectral target admission must return an explicit boolean")
        _cancel(cancelled)
        if not admitted:
            continue
        intervals = tuple((int(frequency.min()), int(frequency.max())) if count == original
            else (-(math.ceil(.5*guard*count)-1), math.ceil(.5*guard*count)-1)
            for count, original, frequency in zip(candidate, shape, frequencies))
        return candidate, intervals
    return None


def resample_controlled_spectrum(wave, *, absolute_l2_budget,
        absolute_probability_budget, numerics=WaveGridNumerics(),
        retained_bytes=0, cancelled=lambda: False,
        guard_nyquist_fraction=GUARD_NYQUIST_FRACTION, minimum_saved_fraction=.05,
        admissible_shape=None):
    """Project weak spectral tails within two absolute, unnormalized budgets.

    ``sqrt(mode_weight)`` and ``mode_weight`` conversions belong to the caller's
    immutable cumulative ledger; these arguments are in the current wave's
    probability convention. Cropped axes retain no coefficient outside 76%
    of their Nyquist frequency. Unchanged axes retain their entire input band.
    Every retained coefficient keeps its original complex value. Return the
    same object and ``None`` if disabled, empty, inadmissible, or fewer than
    the requested fraction of cells would be saved (5% by default). This
    operation is CPU-only and cancellable.

    With ``admissible_shape``, choose the largest resource-admitted grid from
    the fixed-aspect FFT-friendly family and retain its complete guarded
    complex passband. The callback performs only resource checks and returns
    an explicit boolean; cancellation and all callback exceptions propagate.
    The original wave is returned if that least-loss feasible grid exceeds
    either budget. There is one forward transform, regardless of the number
    of candidate resource checks. Without the callback the original
    budget-driven compact-grid selection is unchanged.
    """
    if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
        raise ValueError("Potential-wave spectral resampling requires a CPU PlaneWave")
    if admissible_shape is not None and not callable(admissible_shape):
        raise ValueError("Spectral target admission must be callable or None")
    l2 = _budget(absolute_l2_budget, "Absolute spectral L2 budget")
    probability_cap = _budget(absolute_probability_budget, "Absolute spectral probability budget")
    guard = _budget(guard_nyquist_fraction, "Spectral Nyquist guard")
    saved = _budget(minimum_saved_fraction, "Minimum saved fraction")
    if not 0. < guard < .8:
        raise ValueError("Spectral Nyquist guard must be strictly between zero and 0.8")
    if saved >= 1.:
        raise ValueError("Minimum saved fraction must be less than one")
    numerics.validate()
    _cancel(cancelled)
    if l2 == 0. or probability_cap == 0.:
        return wave, None
    shape = wave.amplitude.shape
    numerics.check_working_memory(shape,
        working_bytes=spectral_resampling_working_bytes(shape), retained_bytes=retained_bytes)
    before = _cpu_chunked_probability(wave.amplitude)
    if not math.isfinite(before):
        raise ValueError("Potential-wave spectral resampling requires finite total probability")
    if before == 0.:
        return wave, None
    target = min(probability_cap, l2*l2, .1*before)*(1.-32*np.finfo(float).eps)
    if target == 0.:
        return wave, None
    spectrum = np.empty(shape, dtype=np.complex128)
    # ifftshift without a full-image roll or copy temporary.
    for sy, dy in ((slice(shape[0]//2, None), slice(0, (shape[0]+1)//2)),
                   (slice(0, shape[0]//2), slice((shape[0]+1)//2, None))):
        for sx, dx in ((slice(shape[1]//2, None), slice(0, (shape[1]+1)//2)),
                       (slice(0, shape[1]//2), slice((shape[1]+1)//2, None))):
            spectrum[dy, dx] = wave.amplitude[sy, sx]
    _inplace_axis_fft(spectrum, 1, cancelled=cancelled)
    _inplace_axis_fft(spectrum, 0, cancelled=cancelled)
    marginal_y, marginal_x = np.zeros(shape[0]), np.zeros(shape[1])
    batch = max(1, _FOURIER_TILE_CELLS//shape[1])
    for first in range(0, shape[0], batch):
        _cancel(cancelled)
        values = spectrum[first:first+batch]
        with np.errstate(under="ignore"):
            probabilities = abs(values)**2
        if np.any((values != 0.) & (probabilities < np.finfo(float).tiny)):
            return wave, None  # Underflow cannot authorize unrecorded loss.
        marginal_y[first:first+batch] = probabilities.sum(axis=1)
        marginal_x += probabilities.sum(axis=0)
    frequencies = tuple(_frequency(count) for count in shape)
    if admissible_shape is None:
        selections = tuple(_axis_candidate(marginal, frequency, target/2., guard)
            for marginal, frequency in zip((marginal_y, marginal_x), frequencies))
        target_shape = tuple(item[0] for item in selections)
        if target_shape == shape or math.prod(target_shape) > (1.-saved)*math.prod(shape):
            return wave, None
        intervals = tuple(item[1] for item in selections)
    else:
        selection = _pressure_selection(shape, frequencies, guard, saved,
                                        admissible_shape, cancelled)
        if selection is None:
            return wave, None
        target_shape, intervals = selection
    keep = tuple((frequency >= interval[0]) & (frequency <= interval[1])
        for frequency, interval in zip(frequencies, intervals))
    losses = [float(marginal_y[~keep[0]].sum())]
    for first in range(0, shape[0], batch):
        _cancel(cancelled)
        selected_rows = keep[0][first:first+batch]
        values = spectrum[first:first+batch][selected_rows][:, ~keep[1]]
        losses.append(float(np.sum(abs(values)**2)))
    lost = math.fsum(losses)
    error = math.sqrt(lost)
    if lost > probability_cap or error > l2:
        return wave, None
    if admissible_shape is not None and lost > target:
        return wave, None
    _cancel(cancelled)
    output = np.zeros(target_shape, dtype=np.complex128)
    phases = tuple(_centred_inverse_phase(count) for count in target_shape)
    for sy, dy in _bands(shape[0], intervals[0]):
        for sx, dx in _bands(shape[1], intervals[1]):
            output[dy, dx] = spectrum[sy, sx]
            output[dy, dx] *= phases[0][dy, None]
            output[dy, dx] *= phases[1][None, dx]
    del spectrum, phases
    _inplace_axis_fft(output, 1, inverse=True, cancelled=cancelled)
    _inplace_axis_fft(output, 0, inverse=True, cancelled=cancelled)
    after = _cpu_chunked_probability(output)
    check_lossless_norm(before-lost, after, context="Budgeted Fourier projection remainder")
    _cancel(cancelled)
    basis = wave.basis_m@np.diag((shape[1]/target_shape[1], shape[0]/target_shape[0]))
    result = replace(wave, amplitude=output, basis_m=basis)
    object.__setattr__(result, "_cpu_probability", after)
    return result, {"kind": "numerical_spectral_projection", "physical_absorption": False,
        "probability_repair": False, "absolute_probability_loss": lost,
        "absolute_l2_error": error, "input_probability": before,
        "output_probability": after, "from_shape": list(shape), "to_shape": list(target_shape),
        "origin_shift_m": [0., 0.], "carrier_constant_action_m": 0.,
        "retained_frequency_intervals_yx": [list(interval) for interval in intervals],
        "guard_nyquist_fraction": guard,
        "error_scope": SPECTRAL_PROJECTION_SCOPE}
