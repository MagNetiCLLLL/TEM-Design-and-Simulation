"""Bounded refinement of an executed complex field, never a new source.

The periodic complex Fourier interpolant retains every represented frequency,
including the negative Nyquist bin of an even lattice. It adds no recovered
physical detail and performs no filtering, fitting or probability correction.
Analytical phase carriers stay factored and the physical period stays fixed.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.physics.wave_flux import check_lossless_norm
from temsim.physics.wave_execution import check_available_memory
from temsim.physics.wave_device import array_module, check_device_memory, max_abs, normalise_backend


class WaveSamplingError(ValueError):
    """An operator has not been applied because its grid needs refinement."""

    def __init__(self, message, required_scale=2.):
        super().__init__(message)
        self.required_scale = float(required_scale)


class WaveGridBudgetError(ValueError):
    """A valid numerical request exceeds its declared grid/memory budget."""


class WaveMemoryBudgetError(WaveGridBudgetError):
    """Only the declared working-memory partition is insufficient.

    Unlike a pixel cap or unresolved operator, this may be retried with fewer
    concurrent modes without changing any numerical or physical operator.
    """


def check_combined_phase_sampling(wave, phase, wavelength_m):
    """Preflight the envelope + analytical carrier + added phase together.

    Frequencies are radians per lattice step (also valid on affine grids).
    The marginal envelope support excludes at most 1e-12 probability per
    axis for the CHECK only; no wave data or weight is removed. The 0.8*pi
    budget leaves a 20% Nyquist guard. Check both the unwrapped local phase
    increments and the COMPLEX transmission spectrum (phase modulation has
    harmonics even when its local slope is small). The support-sum condition
    is sufficient for resolved trigonometric interpolants up to the declared
    tail; it is NOT potential-grid convergence certification. Unknown detail
    already absent from the input potential needs an independent finer grid.
    No wrapped phase of the incoming wave is differentiated at its zeros.
    """
    xp = array_module(wave.amplitude)
    phase = xp.asarray(phase, dtype=float)
    if phase.shape != wave.amplitude.shape or not bool(xp.all(xp.isfinite(phase))):
        raise ValueError("Added phase must be finite and match the wave lattice")
    if not math.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("Phase sampling requires a positive finite wavelength")
    delta = wave.coordinates_m()-xp.asarray(wave.origin_m)[:, None, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    carrier = (xp.einsum("iyx,ij,jyx->yx", delta, xp.asarray(curvature), delta)/2
               + xp.einsum("i,iyx->yx", xp.asarray(tilt), delta))*2*np.pi/wavelength_m
    unwrapped = phase+carrier
    if not bool(xp.all(xp.isfinite(unwrapped))):
        raise ValueError("Combined phase is outside the finite numerical range")
    spectrum = abs(xp.fft.fft2(wave.amplitude, norm="ortho"))**2
    # An affine phase has known (possibly non-bin-centred) frequency. FFT of
    # its periodic extension would invent a boundary jump and reject resolved
    # carriers. Only sample the non-affine specimen transmission here; keep
    # analytical carrier increments separate on the physical lattice.
    affine = all(bool(xp.allclose(xp.diff(phase, axis=axis), xp.diff(phase, axis=axis).ravel()[0],
                             rtol=1e-10, atol=1e-10)) for axis in (0, 1))
    transmission_spectrum = (None if affine else
        abs(xp.fft.fft2(xp.exp(1j*phase), norm="ortho"))**2)
    total = float(spectrum.sum())
    transmission_total = 0. if affine else float(transmission_spectrum.sum())
    bounds = []
    for axis in (0, 1):
        frequency = abs(2*np.pi*xp.fft.fftfreq(wave.amplitude.shape[axis]))
        marginal = spectrum.sum(axis=1-axis)
        order = xp.argsort(frequency)
        tail = xp.cumsum(marginal[order][::-1])[::-1]
        support = max_abs(frequency[order][tail > total*1e-12])
        added = max_abs(xp.diff(unwrapped, axis=axis))
        transmission_support = 0.
        if transmission_spectrum is not None:
            transmission_marginal = transmission_spectrum.sum(axis=1-axis)
            transmission_tail = xp.cumsum(transmission_marginal[order][::-1])[::-1]
            transmission_support = max_abs(frequency[order][transmission_tail > transmission_total*1e-12])
            transmission_support += max_abs(xp.diff(carrier, axis=axis))
        bounds.append(support+max(added, transmission_support))
    largest = max(bounds)
    if largest >= .8*np.pi:
        raise WaveSamplingError(
            f"Combined specimen/carrier/envelope bandwidth {largest:.6g} rad exceeds the 0.8*pi sampling budget; "
            "refine the incoming wave and independently regenerate/verify the potential grid before retrying",
            largest/(.8*np.pi))
    return {"schema": "combined-phase-bandwidth-v1", "bounds_rad_per_step_yx": bounds,
            "budget_rad_per_step": .8*np.pi, "spectral_tail_probability_per_axis": 1e-12,
            "potential_grid_convergence": "NOT_ESTABLISHED_BY_THIS_CHECK"}


@dataclass(frozen=True)
class WaveGridNumerics:
    automatic_refinement: bool = True
    maximum_pixels: int = 32768
    maximum_working_bytes: int = 72*1024**3
    specimen_phase_method: str = "galerkin"
    specimen_quadrature_factor: int = 2
    compute_backend: str = "cpu"
    acceleration_enabled: bool = True
    maximum_device_working_bytes: int = 24*1024**3
    # Bounds the omitted non-quadratic magnetic action per travelled metre.
    # A density keeps admission invariant under checkpoint segmentation.
    maximum_posed_lens_phase_error_rad_per_m: float = .01
    # Per discrete Cs event, separately from the distributed magnetic bound.
    maximum_spherical_obliquity_phase_error_rad: float = .01
    # Numerical support compaction only after a nonlinear operator requests
    # refinement. Zero keeps the original full-domain calculation. Physical
    # source parameters and optical strengths are never changed.
    maximum_support_loss_fraction: float = 1e-6
    maximum_spectral_loss_fraction: float = 1e-8
    # Admission-only spectral tail per axis for nonlinear column phases.
    # All coefficients and probability remain present. This is independent
    # of the measured Fourier-projection loss and its cumulative L2 ledger.
    multipole_spectral_tail_fraction: float = 1e-12
    # Absolute per-reference-electron probability across all omitted mutually
    # incoherent modes, applied once after the executed gun. Not a Tip setting.
    maximum_incoherent_mode_loss_probability: float = 1e-8
    maximum_accumulated_support_l2: float = .02

    def validate(self):
        if not isinstance(self.automatic_refinement, bool):
            raise ValueError("Automatic wave refinement must be a boolean")
        if isinstance(self.maximum_pixels, bool) or not isinstance(self.maximum_pixels, int) or not 32 <= self.maximum_pixels <= 65536:
            raise ValueError("Maximum wave grid pixels must be an integer from 32 to 65536")
        if isinstance(self.maximum_working_bytes, bool) or not isinstance(self.maximum_working_bytes, int) or self.maximum_working_bytes <= 0:
            raise ValueError("Wave working memory budget must be a positive integer")
        normalise_backend(self.compute_backend)
        if type(self.acceleration_enabled) is not bool:
            raise ValueError("Wave acceleration participation must be a boolean")
        if (type(self.maximum_device_working_bytes) is not int
                or self.maximum_device_working_bytes <= 0):
            raise ValueError("Wave device working memory budget must be a positive integer")
        if (isinstance(self.maximum_posed_lens_phase_error_rad_per_m, bool)
                or not math.isfinite(self.maximum_posed_lens_phase_error_rad_per_m)
                or self.maximum_posed_lens_phase_error_rad_per_m <= 0):
            raise ValueError("Posed lens phase error budget must be finite and positive")
        if (isinstance(self.maximum_spherical_obliquity_phase_error_rad, bool)
                or not math.isfinite(self.maximum_spherical_obliquity_phase_error_rad)
                or self.maximum_spherical_obliquity_phase_error_rad <= 0):
            raise ValueError("Spherical obliquity phase error budget must be finite and positive")
        if self.specimen_phase_method not in ("sampled", "galerkin"):
            raise ValueError("Specimen phase method must be sampled or galerkin")
        if (type(self.specimen_quadrature_factor) is not int
                or not 2 <= self.specimen_quadrature_factor <= 16):
            raise ValueError("Specimen potential quadrature factor must be an integer from 2 to 16")
        for name in ("maximum_support_loss_fraction", "maximum_spectral_loss_fraction",
                     "maximum_incoherent_mode_loss_probability",
                     "maximum_accumulated_support_l2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= .1:
                raise ValueError(f"{name} must be finite and between zero and 0.1")
        value = self.multipole_spectral_tail_fraction
        if isinstance(value, (bool, np.bool_)) or not math.isfinite(value) or not 0 <= value <= 1e-4:
            raise ValueError("Multipole spectral tail fraction must be finite and between zero and 0.0001")
        return self

    def check(self, shape, *, retained_bytes=0, working_bytes_per_pixel=256):
        if (type(working_bytes_per_pixel) is not int or working_bytes_per_pixel < 256):
            raise ValueError("Wave working estimate must be at least 256 bytes per pixel")
        # Covers immutable input/output copies, FFT scratch, coordinates,
        # polynomial phase arrays and field propagation work, not total RSS.
        return self.check_working_memory(shape,
            working_bytes=working_bytes_per_pixel*math.prod(shape), retained_bytes=retained_bytes)

    def check_working_memory(self, shape, *, working_bytes, retained_bytes=0):
        """Preflight an operator with an explicitly bounded allocation model.

        ``working_bytes`` includes that operator's input, output, immutable
        publication copies and scratch. ``retained_bytes`` counts other live
        fields. Only operators with bounded allocations should use this entry;
        ordinary operators retain the conservative per-pixel ``check`` above.
        """
        self.validate()
        if (isinstance(working_bytes, bool) or not isinstance(working_bytes, (int, np.integer))
                or working_bytes < 0):
            raise ValueError("Wave operator working bytes must be a non-negative integer")
        if (isinstance(retained_bytes, bool) or not isinstance(retained_bytes, (int, np.integer))
                or retained_bytes < 0):
            raise ValueError("Retained wave bytes must be a non-negative integer")
        required = int(retained_bytes)+int(working_bytes)
        if max(shape) > self.maximum_pixels or required > self.maximum_working_bytes:
            # A simultaneous pixel and memory failure is still a pixel-cap
            # failure: serial execution cannot resolve it.
            error = WaveGridBudgetError if max(shape) > self.maximum_pixels else WaveMemoryBudgetError
            raise error(f"Wave refinement budget exceeded: grid={tuple(shape)}, estimated working bytes={required}; "
                        f"maximum_pixels={self.maximum_pixels}, maximum_working_bytes={self.maximum_working_bytes}. "
                        "The unresolved optical operator was not applied; increase the numerical budget.")
        check_available_memory(required-int(retained_bytes))
        check_device_memory(required-int(retained_bytes))
        return required

    def column_identity(self):
        """Consumed column numerics only; specimen quadrature is downstream."""
        self.validate()
        return {"automatic_refinement": self.automatic_refinement,
                "refinement_grid_selection": "fft-friendly-required-scale-1.05-budgeted-min-growth-1.25-v2",
                "cpu_fourier_refinement": "tiled-complex128-unitary-frequency-embedding-v1",
                "posed_lens_model": "main-column-carrier-centred-bounded-electric-residual-v3",
                "posed_spherical_model": "canonical-first-inclination-cs-v1",
                "intrinsic_aberration_model": "intrinsic-polynomial-canonical-screen-v1",
                "maximum_pixels": self.maximum_pixels,
                "maximum_working_bytes": self.maximum_working_bytes,
                "compute_backend": self.compute_backend,
                "acceleration_enabled": self.acceleration_enabled,
                "maximum_device_working_bytes": self.maximum_device_working_bytes,
                "support_compaction_model": "c2-window-and-bounded-fourier-projection-v2",
                "maximum_support_loss_fraction": self.maximum_support_loss_fraction,
                "maximum_spectral_loss_fraction": self.maximum_spectral_loss_fraction,
                "multipole_sampling_model": "marginal-spectral-tail-admission-v1",
                "multipole_spectral_tail_fraction": self.multipole_spectral_tail_fraction,
                "maximum_incoherent_mode_loss_probability": self.maximum_incoherent_mode_loss_probability,
                "maximum_accumulated_support_l2": self.maximum_accumulated_support_l2,
                "maximum_spherical_obliquity_phase_error_rad": self.maximum_spherical_obliquity_phase_error_rad,
                "maximum_posed_lens_phase_error_rad_per_m": self.maximum_posed_lens_phase_error_rad_per_m}


_FOURIER_TILE_CELLS = 65536


def _cpu_refinement_working_bytes(old, shape):
    """Live arrays for the bounded complex128 CPU implementation below.

    The immutable input stays live. During embedding the private input FFT
    and target coexist; during publication target + immutable copy coexist.
    PlaneWave's finiteness check can allocate one boolean target. FFT calls
    see at most 65536 cells (or one longer axis), never the full image: 8 MiB
    covers their tile copies and norm tiles; 1024 B per longest axis reserves
    single-worker FFT line/twiddle/Bluestein scratch, including prime lengths.
    This is a stage allocation bound, not a process-RSS guarantee.
    """
    old_bytes = 16*math.prod(old)
    cells = math.prod(shape)
    live = old_bytes+max(old_bytes+16*cells, 33*cells)
    return live+8*1024**2+1024*max((*old, *shape))


def _check_refinement_memory(wave, shape, numerics, retained_bytes):
    if array_module(wave.amplitude) is np:
        return numerics.check_working_memory(shape,
            working_bytes=_cpu_refinement_working_bytes(wave.amplitude.shape, shape),
            retained_bytes=retained_bytes)
    return numerics.check(shape, retained_bytes=retained_bytes+wave.amplitude.nbytes)


def _check_refinement_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Wave refinement cancelled")


def _inplace_axis_fft(values, axis, *, inverse=False, cancelled=lambda: False):
    """Native SciPy complex128 FFTs on bounded private views, one worker."""
    from scipy import fft
    transform = fft.ifft if inverse else fft.fft
    batch = max(1, _FOURIER_TILE_CELLS//values.shape[axis])
    other = 1-axis
    with fft.set_backend("scipy", only=True):
        for first in range(0, values.shape[other], batch):
            _check_refinement_cancelled(cancelled)
            selection = [slice(None), slice(None)]
            selection[other] = slice(first, first+batch)
            tile = values[tuple(selection)]
            transformed = transform(tile, axis=axis, norm="ortho", overwrite_x=True, workers=1)
            # Native complex-to-complex SciPy reuses the supplied output.
            # A backend/version returning a fresh tile is still bounded by
            # the same tile budget; never adopt a separate full-size result.
            if not np.shares_memory(transformed, tile):
                tile[...] = transformed


def _centred_inverse_phase(count):
    """Fourier multiplier for fftshift, exact signs for even dimensions."""
    frequency = np.arange(count)
    frequency[(count+1)//2:] -= count
    sign = 1-2*(frequency % 2)
    if count % 2 == 0:
        return sign
    # exp(-2*pi*i*k*floor(N/2)/N), with the bounded residual angle kept
    # separate from integer parity to avoid a large-angle rounding error.
    return sign*np.exp(1j*np.pi*frequency/count)


def _cpu_fourier_embedding(amplitude, shape, *, cancelled):
    old = amplitude.shape
    spectrum = np.empty(old, dtype=np.complex128)
    # ifftshift by quadrant copies, with no rolled full-image temporary.
    for y0, y1 in ((slice(old[0]//2, None), slice(0, (old[0]+1)//2)),
                   (slice(0, old[0]//2), slice((old[0]+1)//2, None))):
        for x0, x1 in ((slice(old[1]//2, None), slice(0, (old[1]+1)//2)),
                       (slice(0, old[1]//2), slice((old[1]+1)//2, None))):
            spectrum[y1, x1] = amplitude[y0, x0]
    _inplace_axis_fft(spectrum, 1, cancelled=cancelled)
    _inplace_axis_fft(spectrum, 0, cancelled=cancelled)
    _check_refinement_cancelled(cancelled)
    output = np.zeros(shape, dtype=np.complex128)
    phases = tuple(_centred_inverse_phase(n) for n in shape)
    # Unshifted positive and negative frequency blocks. The even input's
    # Nyquist bin remains negative; it is neither split nor discarded.
    blocks = [((slice(0, (o+1)//2), slice(0, (o+1)//2)),
               (slice((o+1)//2, None), slice(n-o//2, None))) for o, n in zip(old, shape)]
    for sy, dy in blocks[0]:
        for sx, dx in blocks[1]:
            output[dy, dx] = spectrum[sy, sx]
            output[dy, dx] *= phases[0][dy, None]
            output[dy, dx] *= phases[1][None, dx]
    del spectrum, phases
    _inplace_axis_fft(output, 1, inverse=True, cancelled=cancelled)
    _inplace_axis_fft(output, 0, inverse=True, cancelled=cancelled)
    return output


def _cpu_chunked_probability(amplitude):
    batch = max(1, _FOURIER_TILE_CELLS//amplitude.shape[1])
    return math.fsum(float(np.sum(abs(amplitude[first:first+batch])**2))
                     for first in range(0, amplitude.shape[0], batch))


def refine_plane_wave(wave, shape, *, numerics=WaveGridNumerics(), retained_bytes=0,
                      cancelled=lambda: False):
    """Unitary Fourier embedding at fixed physical period and carrier gauge."""
    old = wave.amplitude.shape
    if len(shape) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < o
                              for n, o in zip(shape, old)):
        raise ValueError("Refinement needs two integer dimensions no smaller than the input")
    shape = tuple(int(n) for n in shape)
    _check_refinement_cancelled(cancelled)
    _check_refinement_memory(wave, shape, numerics, retained_bytes)
    if shape == old:
        return wave
    xp = array_module(wave.amplitude)
    if xp is np:
        amplitude = _cpu_fourier_embedding(wave.amplitude, shape, cancelled=cancelled)
        before = _cpu_chunked_probability(wave.amplitude)
        after = _cpu_chunked_probability(amplitude)
        check_lossless_norm(before, after, context="Fourier wave refinement")
    else:
        spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(wave.amplitude), norm="ortho"))
        padded = xp.zeros(shape, dtype=xp.complex128)
        start = tuple(n//2-o//2 for n, o in zip(shape, old))
        padded[start[0]:start[0]+old[0], start[1]:start[1]+old[1]] = spectrum
        amplitude = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(padded), norm="ortho"))
    basis = wave.basis_m@np.diag((old[1]/shape[1], old[0]/shape[0]))
    _check_refinement_cancelled(cancelled)
    result = wave._replace_owned(amplitude, basis_m=basis)
    if xp is np:
        object.__setattr__(result, "_cpu_probability", after)
    else:
        check_lossless_norm(wave.probability, result.probability, context="Fourier wave refinement")
    return result


def apply_resolved_operator(wave, operator, *, numerics=WaveGridNumerics(), retained_bytes=0,
                            cancelled=lambda: False, prepare_retry=None,
                            operator_working_bytes=None):
    """Retry only a typed sampling failure, always from the pre-operator wave.

    An optional caller-owned preparation can compact the failed input within
    its explicit approximation budget. A changed input MUST be checked again
    by the physical operator before selecting a grid: the old required scale
    describes the old field. The callback owns its error/loss accounting.
    """
    initial_amplitude = wave.amplitude
    def external_bytes(candidate):
        # The caller still owns its original input after our local variable
        # advances to a refined field. Count that immutable buffer once.
        return retained_bytes+(0 if candidate.amplitude is initial_amplitude else initial_amplitude.nbytes)
    def check_operator(shape, external):
        if operator_working_bytes is None:
            return numerics.check(shape, retained_bytes=external)
        return numerics.check_working_memory(shape,
            working_bytes=operator_working_bytes(shape), retained_bytes=external)
    check_operator(wave.amplitude.shape, external_bytes(wave))
    rows = []
    while True:
        if cancelled():
            raise InterruptedError("Wave refinement cancelled")
        try:
            return operator(wave), rows
        except WaveSamplingError as error:
            if not numerics.automatic_refinement:
                raise
            if prepare_retry is not None:
                prepared = prepare_retry(wave, error)
                if prepared is not wave:
                    check_operator(prepared.amplitude.shape, external_bytes(prepared))
                    previous = wave.amplitude.shape
                    wave = prepared
                    del prepared
                    rows.append({"from_shape": previous, "to_shape": wave.amplitude.shape,
                                 "reason": str(error), "probability": wave.probability,
                                 "method": "caller-budgeted sampling preparation; physical operator rechecked"})
                    continue
                # The callback may decline a crop by returning this input.
                # Do not retain it after refinement: the next operator's
                # reservation covers its current field and the caller's
                # original, not an obsolete intermediate mesh as well.
                del prepared
            if not math.isfinite(error.required_scale):
                raise ValueError("Wave operator requires an unbounded refinement") from error
            # A power-of-two *factor* can almost quadruple the needed cell
            # count, causing an avoidable memory failure. FFT-friendly sizes
            # need not be powers of two. Keep a modest sampling guard and
            # retry the unchanged physical operator on every candidate grid.
            shape = _refinement_shape(wave.amplitude.shape, error.required_scale)
            # The target physical operator must fit as well as its temporary
            # Fourier embedding, before any refinement allocation takes place.
            budget_limited_growth = False
            try:
                required = check_operator(shape, retained_bytes+initial_amplitude.nbytes)
                required = max(required, _check_refinement_memory(wave, shape, numerics, external_bytes(wave)))
            except WaveGridBudgetError:
                # A 25% minimum growth avoids many small retries, but is not
                # a sampling requirement. Near a budget boundary use the
                # actual requested scale with the same 5% resolution guard.
                bounded = _refinement_shape(wave.amplitude.shape, error.required_scale, minimum_growth=1.)
                if bounded == shape:
                    raise
                required = check_operator(bounded, retained_bytes+initial_amplitude.nbytes)
                required = max(required, _check_refinement_memory(wave, bounded, numerics, external_bytes(wave)))
                shape, budget_limited_growth = bounded, True
            previous = wave.amplitude.shape
            wave = refine_plane_wave(wave, shape, numerics=numerics,
                                     retained_bytes=external_bytes(wave), cancelled=cancelled)
            rows.append({"from_shape": previous, "to_shape": shape, "reason": str(error),
                         "requested_scale": error.required_scale,
                         "grid_selection": "FFT-friendly required scale with 5 percent guard; minimum growth 1.25 when budget permits",
                         "budget_limited_growth": budget_limited_growth,
                         "estimated_working_bytes": required, "probability": wave.probability,
                         "method": "unitary complex Fourier embedding; fixed physical period and analytical carriers"})


def _refinement_shape(shape, required_scale, *, minimum_growth=1.25):
    """Even smooth FFT dimensions covering the requested resolution increase."""
    from scipy.fft import next_fast_len

    factor = max(minimum_growth, float(required_scale)*1.05)
    result = []
    for count in shape:
        target = max(count+1, math.ceil(count*factor))
        target += target % 2
        target = int(next_fast_len(target))
        while target % 2:
            target = int(next_fast_len(target+1))
        result.append(target)
    return tuple(result)
