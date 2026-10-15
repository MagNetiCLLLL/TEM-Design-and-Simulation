"""Phase operators for the column's existing normal/skew hexapole and Cs laws.

Coefficients and signs follow ray_integrator.canonical_rk4_step and its discrete
Cs kick. These functions add no optics or sources. They retain the full cubic
and quartic phase, including non-quadratic residuals, on an affine wave lattice.
The stationary column_wave development path executes these at physical nodes.
"""
import math

import numpy as np

from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import WaveSamplingError
from temsim.physics.wave_device import array_module, max_abs


# Bound temporary coordinate/action arrays independently of the wave's size.
_MULTIPOLE_TILE_CELLS = 262144


def multipole_working_bytes(shape):
    """CPU allocation bound for the tiled complex128 operator, not total RSS.

    Includes retained input, output, the immutable output copy and full-array
    probability/finite checks. FFT and polynomial scratch are bounded tiles.
    The caller separately reserves other modes and upstream arrays.
    """
    cells = math.prod(shape)
    tile = min(cells, max(_MULTIPOLE_TILE_CELLS, max(shape)))
    return 80*cells+256*tile+128*sum(shape)


def _spectral_marginals(amplitude, *, cancelled=lambda: False):
    """Exact Parseval marginals using bounded one-dimensional FFT batches.

    Sum over the other Fourier axis by Parseval instead of materialising the
    full two-dimensional spectrum. Every original frequency is retained;
    precision, normalisation and the spectral-tail admission rule are unchanged.
    """
    xp = array_module(amplitude)
    result = []
    for axis in (0, 1):
        length = amplitude.shape[axis]
        across = amplitude.shape[1-axis]
        batch = max(1, _MULTIPOLE_TILE_CELLS//length)
        marginal = xp.zeros(length, dtype=xp.float64)
        for start in range(0, across, batch):
            if cancelled():
                raise InterruptedError("Spectral marginal calculation cancelled")
            part = (amplitude[:, start:start+batch] if axis == 0
                    else amplitude[start:start+batch, :])
            if xp is np:
                from scipy.fft import fft
                scratch = np.array(part, dtype=np.complex128, copy=True)
                transformed = fft(scratch, axis=axis, norm="ortho", overwrite_x=True, workers=1)
            else:
                scratch = xp.array(part, dtype=xp.complex128, copy=True)
                transformed = xp.fft.fft(scratch, axis=axis, norm="ortho")
            power = xp.abs(transformed)
            power *= power
            marginal += power.sum(axis=1-axis)
            del scratch, transformed, power
        result.append(marginal)
    return result


def multipole_action(x, y, *, normal_m2=0., skew_m2=0., spherical_m3=0.):
    """Optical action in metres; its gradient is the canonical momentum kick.

    normal_m2/skew_m2 are integrated distributed strengths (h*hn and h*hs),
    whereas spherical_m3 is the existing discrete radial-cubic kick strength.
    """
    return (-normal_m2*(x*x*x/3-x*y*y)
            -skew_m2*(x*x*y-y*y*y/3)
            -spherical_m3*(x*x+y*y)**2/4)


def apply_multipole_phase(wave, wavelength_m, *, normal_m2=0., skew_m2=0., spherical_m3=0.,
                          spectral_tail_fraction=1e-12, sampling_record=None):
    """Apply the complete phase with a declared spectral sampling admission.

    ``spectral_tail_fraction`` changes only the probability quantile used to
    bound each envelope-frequency axis. No coefficients are projected away
    and no physical probability is removed. A relaxed admission is not a
    continuum discretisation error bound. ``sampling_record`` receives the
    actual tails and phase increments, including a rejected sampling trial;
    callers must distinguish rejected trials from executed operators.
    """
    if (isinstance(spectral_tail_fraction, (bool, np.bool_))
            or not math.isfinite(spectral_tail_fraction)
            or not 0 <= spectral_tail_fraction <= 1e-4):
        raise ValueError("Multipole spectral tail fraction must be finite and between zero and 0.0001")
    for value in (wavelength_m, normal_m2, skew_m2, spherical_m3):
        if not math.isfinite(value):
            raise ValueError("Multipole phase inputs must be finite")
    if wavelength_m <= 0:
        raise ValueError("Multipole phase requires positive wavelength")
    if normal_m2 == skew_m2 == spherical_m3 == 0:
        return wave
    xp = array_module(wave.amplitude)
    x0, y0 = wave.origin_m
    hn, hs, cs = normal_m2, skew_m2, spherical_m3
    r2 = x0*x0+y0*y0
    constant = float(multipole_action(x0, y0, normal_m2=hn, skew_m2=hs, spherical_m3=cs))
    gradient = np.array((-hn*(x0*x0-y0*y0)-2*hs*x0*y0-cs*r2*x0,
                          2*hn*x0*y0-hs*(x0*x0-y0*y0)-cs*r2*y0))
    cross = 2*hn*y0-2*hs*x0-2*cs*x0*y0
    hessian = np.array(((-2*hn*x0-2*hs*y0-cs*(3*x0*x0+y0*y0), cross),
                        (cross, 2*hn*x0+2*hs*y0-cs*(x0*x0+3*y0*y0))))
    # Exact Taylor remainder of this degree-four polynomial. Evaluating it
    # directly avoids subtracting nearly equal large off-axis phase carriers.
    amplitude = wave.amplitude
    ny, nx = amplitude.shape
    tile_rows = max(1, _MULTIPOLE_TILE_CELLS//nx)
    ix = (xp.arange(nx)-nx//2)[None, :]
    basis = wave.basis_m

    def phase_rows(start, stop):
        iy = (xp.arange(start, stop)-ny//2)[:, None]
        u = basis[0, 0]*ix+basis[0, 1]*iy
        v = basis[1, 0]*ix+basis[1, 1]*iy
        residual = multipole_action(u, v, normal_m2=hn, skew_m2=hs, spherical_m3=cs)
        residual -= cs*(x0*u+y0*v)*(u*u+v*v)
        return (2*np.pi/wavelength_m)*residual

    occupied_threshold = max(max_abs(amplitude[start:start+tile_rows])
                             for start in range(0, ny, tile_rows))*1e-8
    # Bound represented envelope frequencies by marginal spectral probability.
    # A wrapped neighbour phase is not a frequency bound: zeros and vortices
    # legitimately have pi jumps. No spectrum is removed by this check.
    marginals = _spectral_marginals(amplitude)
    total = float(marginals[0].sum())
    bandwidths, ignored_fractions = [], []
    for axis in (0, 1):
        n = amplitude.shape[axis]
        frequencies = abs(2*np.pi*xp.fft.fftfreq(n))
        marginal = marginals[axis]
        order = xp.argsort(frequencies)
        tail = xp.cumsum(marginal[order][::-1])[::-1]
        support = frequencies[order][tail > total*spectral_tail_fraction]
        bandwidth = max_abs(support)
        bandwidths.append(bandwidth)
        # Direct tail sums avoid cancellation at the strict 1e-12 default.
        ignored = float(marginal[frequencies > bandwidth].sum())
        ignored_fractions.append(ignored/total if total else 0.)
    del marginals
    increments = [0., 0.]
    for start in range(0, ny, tile_rows):
        # One overlapping row checks neighbours across every tile boundary.
        stop = min(ny, start+tile_rows+1)
        phase = phase_rows(start, stop)
        occupied = abs(amplitude[start:stop]) > occupied_threshold
        for axis in (0, 1):
            neighbours = (occupied[:-1] & occupied[1:] if axis == 0
                          else occupied[:, :-1] & occupied[:, 1:])
            added = max_abs(abs(xp.diff(phase, axis=axis))[neighbours])
            increments[axis] = max(increments[axis], added)
    del phase, occupied
    combined = [a+b for a, b in zip(increments, bandwidths)]
    largest = max(combined)
    required_scale = max(1., largest/(.8*np.pi))
    if sampling_record is not None:
        sampling_record.update({
            "schema": "multipole-bandwidth-admission-v1",
            "spectral_tail_fraction_per_axis": float(spectral_tail_fraction),
            "ignored_probability_fraction_yx": ignored_fractions,
            "ignored_spectral_union_fraction_upper_bound": min(1., sum(ignored_fractions)),
            "envelope_bandwidth_rad_per_step_yx": bandwidths,
            "operator_increment_rad_per_step_yx": increments,
            "combined_bandwidth_rad_per_step_yx": combined,
            "budget_rad_per_step": .8*np.pi,
            "occupied_relative_amplitude_threshold": 1e-8,
            "input_probability": total,
            "required_scale": required_scale,
            "admitted": required_scale <= 1.,
            "scope": "Sampling admission only; complete spectrum retained; no physical probability loss or continuum error bound",
        })
    if required_scale > 1:
        raise WaveSamplingError(f"Nonlinear column phase is undersampled (operator increment plus envelope bandwidth "
                                f"{largest:.6g} rad, budget 0.8*pi); refine the physical wave grid", required_scale)
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    output = xp.empty_like(amplitude)
    for start in range(0, ny, tile_rows):
        stop = min(ny, start+tile_rows)
        phase = phase_rows(start, stop)
        output[start:stop] = amplitude[start:stop]*xp.exp(1j*(phase+2*np.pi*constant/wavelength_m))
    del phase
    result = wave._replace_owned(output,
                     curvature_m1=curvature+hessian, tilt_rad=tilt+gradient)
    del output
    check_lossless_norm(wave.probability, result.probability, context="Column multipole phase")
    return result
