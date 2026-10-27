"""Explicitly budgeted support compaction on an unchanged physical lattice.

This is a numerical approximation, not an aperture, source adjustment or
probability normalisation. The returned record measures both lost probability
and the L2 difference of the complex field. Later unitary propagation preserves
that L2 bound; absorbing operators can only reduce it. Bounds from successive
compactions must be accumulated by the caller. That statement assumes the same
subsequent physical operator; changing its numerical domain or boundary
discretization still requires independent convergence checks.
"""
from dataclasses import replace
import math

import numpy as np

from temsim.physics.wave_device import array_module, to_host


def _crop_interval(core_low, core_high, size):
    """Include a four-cell C2 taper and two zero guard cells on each side."""
    low, stop = max(0, core_low-6), min(size, core_high+7)
    wanted = max(32, stop-low)
    wanted += wanted % 2
    if wanted >= size:
        return 0, size
    low = max(0, min(size-wanted, low-(wanted-(stop-low)+1)//2))
    return low, low+wanted


def _marginal_core(marginal, maximum_loss_fraction):
    """Bound both tails directly, including budgets below unit roundoff."""
    forward = np.cumsum(marginal, dtype=float)
    backward = np.cumsum(marginal[::-1], dtype=float)
    budget = float(forward[-1])*maximum_loss_fraction/4
    size = len(marginal)
    low = min(size-1, int(np.searchsorted(forward, budget, side="left")))
    high = size-1-min(size-1, int(np.searchsorted(backward, budget, side="left")))
    return low, high


def compact_wave_support(wave, *, maximum_loss_fraction, wavelength_m):
    """Taper negligible tails, then crop without interpolation or renormalising.

    Each marginal tail receives one quarter of the probability-loss budget.
    Their union bounds the probability outside the retained core rectangle.
    A separable quintic C2 window is exactly one in that rectangle, and reaches
    zero four original cells beyond it. Cropping preserves those original cell
    centres, their areas, and their full complex values multiplied by the window.
    A carrier-gauge translation compensates for the changed lattice origin.

    Return the input and ``None`` when disabled, empty, or less than 25 percent
    of cells would be saved. Cropped axes are even and at least 32 cells; axes
    retained in their entirety keep their original size, including odd sizes.
    NumPy and CuPy amplitudes use the same algorithm; only 1D marginals are
    transferred to the host. No physical interpretation of the loss is implied.
    """
    if (isinstance(maximum_loss_fraction, (bool, np.bool_))
            or not math.isfinite(maximum_loss_fraction)
            or not 0 <= maximum_loss_fraction <= .1):
        raise ValueError("Support loss fraction must be finite and between zero and 0.1")
    if (isinstance(wavelength_m, (bool, np.bool_))
            or not math.isfinite(wavelength_m) or wavelength_m <= 0):
        raise ValueError("Support compaction requires a positive finite wavelength")
    if maximum_loss_fraction == 0:
        return wave, None
    xp = array_module(wave.amplitude)
    probability = xp.abs(wave.amplitude)**2
    total = float(probability.sum())
    if total == 0:
        return wave, None
    shape = wave.amplitude.shape
    cores, intervals = [], []
    for axis, size in enumerate(shape):
        marginal = to_host(probability.sum(axis=1-axis))
        low, high = _marginal_core(marginal, maximum_loss_fraction)
        cores.append((low, high))
        intervals.append(_crop_interval(low, high, size))
    new_shape = tuple(stop-low for low, stop in intervals)
    if math.prod(new_shape) > .75*math.prod(shape):
        return wave, None
    (y0, y1), (x0, x1) = intervals
    window_axes = []
    for (low, stop), (core_low, core_high) in zip(intervals, cores):
        coordinate = xp.arange(low, stop, dtype=float)
        distance = xp.maximum(xp.maximum(core_low-coordinate, coordinate-core_high), 0.)
        t = xp.minimum(distance/4., 1.)
        # This form is exactly zero at and beyond the end of the transition.
        window_axes.append(1.-t**3*(10.+t*(-15.+6.*t)))
    window = window_axes[0][:, None]*window_axes[1][None, :]
    retained_probability = probability[y0:y1, x0:x1]
    # Direct sums avoid cancellation when the loss is far below total norm.
    outside_probability = float(probability[:y0].sum()+probability[y1:].sum()
                                +probability[y0:y1, :x0].sum()+probability[y0:y1, x1:].sum())
    lost = outside_probability+float((retained_probability*(1.-window**2)).sum())
    squared_error = outside_probability+float((retained_probability*(1.-window)**2).sum())
    loss_fraction = lost/total
    if loss_fraction > maximum_loss_fraction:
        raise ValueError("Support window exceeded its integrated probability-loss budget")
    shift_indices_xy = np.array((x0+new_shape[1]//2-shape[1]//2,
                                 y0+new_shape[0]//2-shape[0]//2))
    origin_shift = wave.basis_m@shift_indices_xy
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    constant_action = float(.5*origin_shift@curvature@origin_shift+tilt@origin_shift)
    phase = 2*np.pi*constant_action/wavelength_m
    amplitude = wave.amplitude[y0:y1, x0:x1]*window*np.exp(1j*phase)
    # Use the symmetric part to preserve the actual scalar quadratic form even
    # when the stored matrix has permitted floating-point antisymmetric noise.
    new_tilt = tilt+.5*(curvature+curvature.T)@origin_shift
    result = replace(wave, amplitude=amplitude, origin_m=wave.origin_m+origin_shift,
                     tilt_rad=new_tilt if wave.tilt_rad is not None or np.any(new_tilt) else None)
    record = {
        "schema": "wave-support-compaction-v1",
        "method": "marginal probability core; four-cell quintic C2 taper; original-lattice crop",
        "maximum_loss_fraction": float(maximum_loss_fraction),
        "loss_fraction": float(loss_fraction),
        "absolute_probability_loss": float(lost),
        "relative_l2_error": float(math.sqrt(squared_error/total)),
        "absolute_l2_error": float(math.sqrt(squared_error)),
        "input_probability": total,
        "output_probability": result.probability,
        "from_shape": list(shape),
        "to_shape": list(new_shape),
        "crop_indices_yx": [[y0, y1], [x0, x1]],
        "core_indices_yx_inclusive": [list(core) for core in cores],
        "transition_cells": 4,
        "origin_shift_m": origin_shift.tolist(),
        "carrier_constant_action_m": constant_action,
        "renormalised": False,
        "physical_absorption": False,
    }
    return result, record


def compact_wave_spectrum(wave, *, maximum_loss_fraction):
    """Project negligible envelope frequencies before reducing its lattice.

    This is a declared orthogonal Fourier projection, not point decimation.
    The physical period, origin and analytical carriers are unchanged. The
    negative-Nyquist convention matches ``refine_plane_wave`` exactly. Lost
    squared L2 norm equals discarded spectral probability by Parseval. Neither
    retained Fourier coefficients nor transmitted probability are normalised.

    A 25 percent frequency-width margin is retained beyond the probability
    quantile core, unless the original lattice is already smaller. Dimensions
    use even FFT-friendly lengths, at least 32. Skip the operation when fewer
    than 25 percent of cells would be saved. NumPy and CuPy share this path.
    """
    from scipy.fft import next_fast_len

    if (isinstance(maximum_loss_fraction, (bool, np.bool_))
            or not math.isfinite(maximum_loss_fraction)
            or not 0 <= maximum_loss_fraction <= .1):
        raise ValueError("Spectral loss fraction must be finite and between zero and 0.1")
    if maximum_loss_fraction == 0:
        return wave, None
    xp = array_module(wave.amplitude)
    spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(wave.amplitude), norm="ortho"))
    spectral_probability = xp.abs(spectrum)**2
    total = float(spectral_probability.sum())
    if total == 0:
        return wave, None
    shape = wave.amplitude.shape
    target, cores, intervals = [], [], []
    for axis, size in enumerate(shape):
        marginal = to_host(spectral_probability.sum(axis=1-axis))
        low, high = _marginal_core(marginal, maximum_loss_fraction)
        frequency_low, frequency_high = low-size//2, high-size//2
        # An even output represents integer modes [-n/2, n/2-1]. In particular
        # the positive upper endpoint needs one more slot than the negative.
        width = max(-2*frequency_low, 2*(frequency_high+1), 2)
        wanted = max(32, math.ceil(1.25*width))
        wanted += wanted % 2
        wanted = int(next_fast_len(wanted))
        while wanted % 2:
            wanted = int(next_fast_len(wanted+1))
        count = min(size, wanted)
        first = size//2-count//2
        target.append(count)
        intervals.append((first, first+count))
        cores.append((low, high))
    if math.prod(target) > .75*math.prod(shape):
        return wave, None
    (y0, y1), (x0, x1) = intervals
    # Four disjoint tails: no subtraction of almost equal total norms.
    lost = float(spectral_probability[:y0].sum()+spectral_probability[y1:].sum()
                 +spectral_probability[y0:y1, :x0].sum()+spectral_probability[y0:y1, x1:].sum())
    loss_fraction = lost/total
    if loss_fraction > maximum_loss_fraction:
        raise ValueError("Spectral projection exceeded its integrated probability-loss budget")
    amplitude = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(spectrum[y0:y1, x0:x1]), norm="ortho"))
    basis = wave.basis_m@np.diag((shape[1]/target[1], shape[0]/target[0]))
    result = replace(wave, amplitude=amplitude, basis_m=basis)
    record = {
        "schema": "wave-spectrum-compaction-v1",
        "method": "orthogonal Fourier projection; unchanged physical period and analytical carriers",
        "maximum_loss_fraction": float(maximum_loss_fraction),
        "loss_fraction": float(loss_fraction),
        "absolute_probability_loss": lost,
        "relative_l2_error": float(math.sqrt(loss_fraction)),
        "absolute_l2_error": float(math.sqrt(lost)),
        "input_probability": total,
        "output_probability": result.probability,
        "from_shape": list(shape),
        "to_shape": list(target),
        "crop_indices_yx": [[y0, y1], [x0, x1]],
        "core_indices_yx_inclusive": [list(core) for core in cores],
        "minimum_frequency_width_margin": 1.25,
        "frequency_margin_scope": "cropped axes; unchanged axes retain their complete input band",
        "renormalised": False,
        "physical_absorption": False,
    }
    return result, record
