"""SI mechanical/canonical boundaries and sampled phase-carrier admission.

Canonical ordering is (x, y, p_x/p_ref, p_y/p_ref). Positions are metres;
momentum ratios and paraxial mechanical slopes are dimensionless. The gauge
is A=(-Bz*y/2, Bz*x/2, 0) at an axial, locally uniform boundary plane.
"""
from __future__ import annotations

import numpy as np

from temsim.physics.wave_device import array_module, to_host

CANONICAL_BASIS = "laboratory-normalized-canonical"
SYMPLECTIC_TOLERANCE = 1e-7
_PHASE_MOMENT_TILE_CELLS = 262144


def phase_carrier_moments_working_bytes(shape):
    """Bound input, norm reduction and tiled complex128 moment scratch.

    The input stays immutable. No full-grid coordinates, Fourier spectrum,
    momentum components or four-component phase-space vectors are allocated.
    Other retained waves belong in the caller's separate memory reservation.
    """
    cells = int(np.prod(shape))
    tile = min(cells, max(_PHASE_MOMENT_TILE_CELLS, max(shape)))
    return 32*cells+192*tile+128*sum(shape)


def phase_carrier_covariance_blocks(wave, wavelength_m, *, cancelled=lambda: False):
    """Return the exact finite-Fourier covariance blocks ``(xx, xp)``.

    Only these blocks select a quadratic phase carrier. A derivative along
    either lattice axis is its complete one-dimensional Fourier derivative;
    the independent lines are processed in bounded batches. This is the same
    derivative as the full two-dimensional transform, without calculating the
    unused momentum-momentum block. Every sample and Fourier bin is retained.

    ``p`` is laboratory canonical momentum divided by the reference momentum,
    including the existing analytic curvature. Constant tilt and translation
    cancel in the centred covariance; no phase or wave samples are changed.
    """
    if cancelled():
        raise InterruptedError("Phase-carrier moments cancelled")
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Phase-carrier moments need a positive finite wavelength")
    norm = wave.probability
    if norm <= 0:
        raise ValueError("An empty wave has no phase-space covariance")
    amplitude = wave.amplitude
    xp = array_module(amplitude)
    ny, nx = amplitude.shape
    basis = wave.basis_m
    first = xp.zeros(2, dtype=xp.float64)
    second = xp.zeros((2, 2), dtype=xp.float64)

    def coordinates(row_start, row_stop, column_start, column_stop):
        ix = (xp.arange(column_start, column_stop)-nx//2)[None, :]
        iy = (xp.arange(row_start, row_stop)-ny//2)[:, None]
        return (basis[0, 0]*ix+basis[0, 1]*iy,
                basis[1, 0]*ix+basis[1, 1]*iy)

    rows = max(1, _PHASE_MOMENT_TILE_CELLS//nx)
    for start in range(0, ny, rows):
        if cancelled():
            raise InterruptedError("Phase-carrier moments cancelled")
        stop = min(ny, start+rows)
        x, y = coordinates(start, stop, 0, nx)
        power = xp.abs(amplitude[start:stop])
        power *= power
        first[0] += xp.sum(x*power)
        first[1] += xp.sum(y*power)
        second[0, 0] += xp.sum(x*x*power)
        second[0, 1] += xp.sum(x*y*power)
        second[1, 1] += xp.sum(y*y*power)
    del x, y, power
    second[1, 0] = second[0, 1]
    mean_x = first/norm
    xx = second/norm-xp.outer(mean_x, mean_x)

    # Columns use the canonical lattice ordering (x frequency, y frequency).
    first_frequency = xp.zeros(2, dtype=xp.float64)
    mixed = xp.zeros((2, 2), dtype=xp.float64)
    for axis, component in ((1, 0), (0, 1)):
        length = amplitude.shape[axis]
        across = amplitude.shape[1-axis]
        batch = max(1, _PHASE_MOMENT_TILE_CELLS//length)
        frequency = xp.fft.fftfreq(length)
        frequency = frequency[None, :] if axis == 1 else frequency[:, None]
        for start in range(0, across, batch):
            if cancelled():
                raise InterruptedError("Phase-carrier moments cancelled")
            stop = min(across, start+batch)
            if axis == 1:
                part = amplitude[start:stop, :]
                x, y = coordinates(start, stop, 0, nx)
            else:
                part = amplitude[:, start:stop]
                x, y = coordinates(0, ny, start, stop)
            scratch = xp.array(part, dtype=xp.complex128, copy=True)
            if xp is np:
                from scipy.fft import fft, ifft
                transformed = fft(scratch, axis=axis, overwrite_x=True, workers=1)
                transformed *= frequency
                momentum = ifft(transformed, axis=axis, overwrite_x=True, workers=1)
            else:
                transformed = xp.fft.fft(scratch, axis=axis)
                transformed *= frequency
                momentum = xp.fft.ifft(transformed, axis=axis)
            correlation = xp.real(part.conj()*momentum)
            first_frequency[component] += xp.sum(correlation)
            mixed[0, component] += xp.sum(x*correlation)
            mixed[1, component] += xp.sum(y*correlation)
            del scratch, transformed, momentum, correlation, x, y
    if cancelled():
        raise InterruptedError("Phase-carrier moments cancelled")
    x_frequency = mixed/norm-xp.outer(mean_x, first_frequency/norm)
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    xp_covariance = (wavelength_m*x_frequency@xp.asarray(np.linalg.inv(basis))
                     +xx@xp.asarray(curvature.T))
    return to_host(xx), to_host(xp_covariance)


def canonical_basis_matrix(*, charge_c: float, bz_t: float,
                           momentum_kg_m_s: float, reference_momentum_kg_m_s: float) -> np.ndarray:
    values = (charge_c, bz_t, momentum_kg_m_s, reference_momentum_kg_m_s)
    if not np.all(np.isfinite(values)) or min(values[2:]) <= 0:
        raise ValueError("Canonical conversion requires finite fields and positive physical momenta")
    matrix = np.eye(4)
    matrix[2:, 2:] *= momentum_kg_m_s / reference_momentum_kg_m_s
    g = charge_c * bz_t / (2 * reference_momentum_kg_m_s)
    matrix[2:, :2] = ((0., -g), (g, 0.))
    return matrix


def validate_canonical_map(matrix, *, basis=CANONICAL_BASIS,
                           tolerance=SYMPLECTIC_TOLERANCE) -> float:
    """Return dimensionless symplectic defect; never repair an invalid map.

    A common position/angle scaling balances B and C before checking M^T J M.
    It is a similarity in a single normalized basis, not a change of momentum
    reference between input and output. Mechanical maps must be converted first.
    """
    if basis != CANONICAL_BASIS:
        raise ValueError("Wave transport needs an explicitly normalized canonical basis")
    m = np.asarray(matrix, float)
    if m.shape != (4, 4) or not np.all(np.isfinite(m)):
        raise ValueError("Canonical map must be a finite 4x4 matrix")
    if not np.isfinite(tolerance) or not 0 < tolerance <= 1e-5:
        raise ValueError("Invalid symplectic admission tolerance")
    b, c = np.linalg.norm(m[:2, 2:], ord=2), np.linalg.norm(m[2:, :2], ord=2)
    length = np.sqrt(b / c) if b > 0 and c > 0 else (b if b > 0 else (1/c if c > 0 else 1.))
    scale = np.diag((1/length, 1/length, 1., 1.))
    scaled = scale @ m @ np.diag((length, length, 1., 1.))
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    defect = float(np.linalg.norm(scaled.T @ j @ scaled - j, ord=np.inf))
    if not np.isfinite(defect) or defect > tolerance:
        raise ValueError(f"Non-symplectic canonical map: residual {defect:.6g} exceeds {tolerance:.6g}")
    return defect


def mechanical_map_to_canonical(matrix, *, input_basis, output_basis):
    """Convert mechanical slopes at BOTH endpoints, then check admission."""
    result = np.asarray(output_basis) @ np.asarray(matrix) @ np.linalg.inv(input_basis)
    validate_canonical_map(result)
    return result


def expanded_phase_amplitude(wave, wavelength_m: float) -> np.ndarray:
    """Materialize the full complex CELL amplitude, including tilt/curvature.

    FFT sign is exp(-2 pi i f.x); a positive phase gradient is positive momentum.
    The envelope spectrum and analytic carrier share one Nyquist budget.
    A nodal sign change is not a measurement of local wave bandwidth.
    No normalization, pupil replacement or resampling occurs here.
    """
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Phase expansion requires a finite positive wavelength")
    from temsim.physics.wave_grid import WaveSamplingError, check_combined_phase_sampling
    xp = array_module(wave.amplitude)
    try:
        # Check the represented envelope spectrum and known UNWRAPPED
        # analytic carrier. Differentiating angle(a1*conj(a0)) at zeros can
        # falsely reject a well-resolved non-Gaussian field: a legitimate
        # cosine sign change already has a pi branch, before adding any tilt.
        # This sufficient spectral test retains every complex cell and uses
        # the same cumulative-tail and 20% Nyquist guard as material stages.
        check_combined_phase_sampling(wave, xp.zeros(wave.amplitude.shape), wavelength_m)
    except WaveSamplingError as error:
        raise WaveSamplingError(f"Full-wave phase carrier is undersampled: {error}",
                                required_scale=error.required_scale) from error
    xy = wave.coordinates_m() - xp.asarray(wave.origin_m)[:, None, None]
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    t = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase = (2*np.pi/wavelength_m) * (
        .5*xp.einsum("iyx,ij,jyx->yx", xy, xp.asarray(q), xy)
        + xp.einsum("i,iyx->yx", xp.asarray(t), xy))
    return wave.amplitude * xp.exp(1j*phase)
