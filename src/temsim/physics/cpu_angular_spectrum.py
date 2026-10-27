"""Bounded CPU execution of the existing conservative Fourier drift.

The caller must already admit the same full-band conservative drift domain as
``multiplane_wave._sampled_angular_spectrum``. This module changes allocation
order only: all complex128 frequency bins and their quadratic phase survive.
It supplies no replacement for non-conservative-domain or CUDA admission.
The caller must also check the physical input/output support: a conservative
Fourier phase increment alone does not prevent finite-window wraparound.
"""
import math

import numpy as np

from temsim.physics.wave_device import array_module
from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_grid import (
    WaveGridNumerics, _cpu_chunked_probability, _inplace_axis_fft,
)


_ANGULAR_TILE_CELLS = 65536


def angular_spectrum_working_bytes(shape):
    """Bound the conservative drift and its caller's result publication.

    80 B/cell covers the immutable input, private complex spectrum/result,
    PlaneWave's immutable copy, a subsequent constant-global-phase result and
    its immutable copy, plus norm/finiteness checks. Those phases do not all
    coexist; the margin includes the caller retaining its previous result.
    Coordinates and quadratic phase are <=65536-cell row tiles, and FFT calls
    use bounded one-axis batches with one worker. The line reserve includes
    native FFT/twiddle/Bluestein scratch for prime axis lengths. Other modes
    and external arrays remain the caller's separate ``retained_bytes``.
    """
    cells = math.prod(shape)
    tile = min(cells, max(_ANGULAR_TILE_CELLS, max(shape)))
    return 80*cells+128*tile+8*1024**2+1024*max(shape)


def conservative_angular_spectrum(wave, drift, wavelength_m, *, grid_numerics=None,
                                 retained_bytes=0, cancelled=lambda: False):
    """Return the exact finite-Fourier drift amplitude in private CPU storage.

    No source fit, filtering, mode truncation or probability correction is
    applied. Curvature, affine lattice and axial/global phases remain owned
    by the caller. A zero drift reuses the original immutable amplitude.
    """
    if cancelled():
        raise InterruptedError("Conservative angular-spectrum drift cancelled")
    if array_module(wave.amplitude) is not np:
        raise ValueError("Bounded conservative angular-spectrum execution requires a CPU wave")
    drift = np.asarray(drift, dtype=float)
    if drift.shape != (2, 2) or not np.isfinite(drift).all():
        raise ValueError("Conservative angular-spectrum drift needs a finite 2 by 2 matrix")
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Conservative angular-spectrum drift needs a positive finite wavelength")
    numerics = grid_numerics or WaveGridNumerics()
    numerics.check_working_memory(wave.amplitude.shape,
        working_bytes=angular_spectrum_working_bytes(wave.amplitude.shape), retained_bytes=retained_bytes)
    if np.all(drift == 0):
        return wave.amplitude
    output = np.array(wave.amplitude, dtype=np.complex128, copy=True)
    _inplace_axis_fft(output, 1, cancelled=cancelled)
    _inplace_axis_fft(output, 0, cancelled=cancelled)
    ny, nx = output.shape
    inverse_basis = np.linalg.inv(wave.basis_m).T
    fx = np.fft.fftfreq(nx)[None, :]
    fy = np.fft.fftfreq(ny)
    rows = max(1, _ANGULAR_TILE_CELLS//nx)
    for first in range(0, ny, rows):
        if cancelled():
            raise InterruptedError("Conservative angular-spectrum drift cancelled")
        last = min(ny, first+rows)
        y = fy[first:last, None]
        u = inverse_basis[0, 0]*fx+inverse_basis[0, 1]*y
        v = inverse_basis[1, 0]*fx+inverse_basis[1, 1]*y
        phase = -np.pi*wavelength_m*(drift[0, 0]*u*u
            +(drift[0, 1]+drift[1, 0])*u*v+drift[1, 1]*v*v)
        output[first:last] *= np.exp(1j*phase)
    del u, v, phase
    _inplace_axis_fft(output, 1, inverse=True, cancelled=cancelled)
    _inplace_axis_fft(output, 0, inverse=True, cancelled=cancelled)
    if cancelled():
        raise InterruptedError("Conservative angular-spectrum drift cancelled")
    check_lossless_norm(_cpu_chunked_probability(wave.amplitude), _cpu_chunked_probability(output),
                        context="Conservative angular-spectrum Fourier drift")
    return output
