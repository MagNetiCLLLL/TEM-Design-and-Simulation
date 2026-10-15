"""Bounded local quadrature of an existing field at a circular hard aperture.

The finite Fourier envelope is evaluated before clipping, on nested affine
charts enclosing the aperture. This neither recovers missing input bandwidth
nor creates a replacement source. Only the masked result may be published:
cropping the unmasked incident field would be a different physical operator.
The analytic carrier remains factored, including its exact origin phase.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim.physics.wave_device import array_module
from temsim.physics.wave_grid import WaveGridNumerics, WaveSamplingError, refine_plane_wave


SCHEMA = "local-fourier-circular-aperture-v1"


@dataclass(frozen=True)
class ApertureSamplingNumerics:
    minimum_radius_cells: int = 16
    output_halfwidth_radii: float = 2.
    maximum_relative_current_change: float = .002
    maximum_complex_change_per_input_norm: float = .005

    def validate(self):
        if type(self.minimum_radius_cells) is not int or self.minimum_radius_cells < 8:
            raise ValueError("Aperture radius sampling must be an integer of at least eight")
        if (isinstance(self.output_halfwidth_radii, bool)
                or not math.isfinite(self.output_halfwidth_radii)
                or self.output_halfwidth_radii < 1.5):
            raise ValueError("Aperture output halfwidth must be at least 1.5 radii")
        for name in ("maximum_relative_current_change", "maximum_complex_change_per_input_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < .1:
                raise ValueError(f"{name} must be finite and between zero and 0.1")
        return self


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Circular aperture sampling cancelled before publication")


def _footprint_corners(wave):
    ny, nx = wave.amplitude.shape
    # Include whole cells, not just their centres, in the exact no-op proof.
    x = (-(nx//2)-.5, nx-nx//2-.5)
    y = (-(ny//2)-.5, ny-ny//2-.5)
    return np.asarray([(xx, yy) for xx in x for yy in y])@wave.basis_m.T+wave.origin_m


def _sample_envelope(spectrum, shape, scale, shift, cancelled):
    """Exact finite Fourier series; ZoomFFT avoids powers of rounded chirps."""
    xp = array_module(spectrum)
    if xp is np:
        from scipy.signal import zoom_fft
    else:
        from cupyx.scipy.signal import zoom_fft
    result = spectrum
    original_shape = spectrum.shape
    for axis, coordinate in ((0, 1), (1, 0)):
        _cancel(cancelled)
        count, old = shape[axis], original_shape[axis]
        first = float(shift[coordinate])-scale*(count//2)
        result = zoom_fft(result, (-first/old, -(first+scale*count)/old),
                          m=count, fs=1., endpoint=False, axis=axis)
        t = first+scale*xp.arange(count)
        correction = xp.exp(-2j*np.pi*(old//2)*t/old)
        result *= correction[:, None] if axis == 0 else correction[None, :]
    # Spectrum is Fourier-series coefficients of the original per-cell
    # amplitude. Cell area scales by scale**2 on this proportional lattice.
    return result*scale


def _input_admission(wave, spectrum, grid_numerics):
    """Reject unresolved incoming envelopes; interpolation cannot repair them."""
    xp = array_module(spectrum)
    probability = wave.probability
    spectral_probability = abs(spectrum)**2
    spectral_total = float(spectral_probability.sum())
    tails = []
    for axis, size in enumerate(spectrum.shape):
        frequencies = xp.fft.fftshift(xp.fft.fftfreq(size))
        marginal = spectral_probability.sum(axis=1-axis)
        tails.append(float(marginal[abs(frequencies) >= .4].sum())/spectral_total)
    amplitude = wave.amplitude
    boundary = (float(xp.sum(abs(amplitude[:2])**2))+float(xp.sum(abs(amplitude[-2:])**2))
                +float(xp.sum(abs(amplitude[2:-2, :2])**2))
                +float(xp.sum(abs(amplitude[2:-2, -2:])**2)))
    boundary_fraction = boundary/probability
    if max(tails) > grid_numerics.maximum_spectral_loss_fraction:
        raise WaveSamplingError(
            "Incoming aperture envelope is undersampled: outer 20 percent of Nyquist "
            f"contains {max(tails):.6g} probability fraction. Regenerate the incoming "
            "propagation on a resolved grid; local interpolation cannot recover absent detail.")
    if boundary_fraction > grid_numerics.maximum_support_loss_fraction:
        raise WaveSamplingError(
            f"Incoming aperture field has unresolved physical-domain edges ({boundary_fraction:.6g} "
            "probability fraction); a local periodic interpolant cannot establish the missing exterior.")
    return {"input_nyquist_guard_probability_fraction_yx": tails,
            "input_boundary_probability_fraction": boundary_fraction}


def apply_circular_aperture(wave, *, radius_m, centre_m=(0., 0.), wavelength_m,
                            grid_numerics=WaveGridNumerics(), numerics=ApertureSamplingNumerics(),
                            retained_bytes=0, cancelled=lambda: False, strict_boundary=False):
    """Return ``(masked_wave, evidence)`` only after nested quadrature admission.

    Error is an observed nested-grid L2 change, not a rigorous continuum bound.
    It includes the hard edge: the coarse *masked* field is Fourier-embedded
    onto the fine chart before comparison. No phase alignment, norm matching,
    fitted source or smoothing of the physical aperture is allowed. The caller
    must accumulate ``complex_error_norm`` in its column error ledger.

    Work remains on the input array's CPU/GPU device. A surrounding execution
    scope owns backend selection/fallback. Every trial starts from the original
    unmasked Fourier coefficients. Failure and cancellation leave it unchanged.
    """
    numerics.validate()
    grid_numerics.validate()
    centre = np.asarray(centre_m, dtype=float)
    if (centre.shape != (2,) or not np.all(np.isfinite(centre))
            or isinstance(radius_m, bool) or not math.isfinite(radius_m) or radius_m < 0
            or not math.isfinite(wavelength_m) or wavelength_m <= 0 or type(strict_boundary) is not bool):
        raise ValueError("Circular aperture needs finite centre, nonnegative radius and positive wavelength")
    _cancel(cancelled)
    probability = wave.probability
    base = {"schema": SCHEMA, "radius_m": float(radius_m), "centre_m": centre.tolist(),
            "input_probability": probability, "input_shape": list(wave.amplitude.shape),
            "numerics": asdict(numerics), "strict_boundary": strict_boundary,
            "normalized": False, "phase_fitted": False,
            "complex_error_norm": 0., "relative_current_change": 0., "trials": [],
            "scope": "Nested finite-Fourier-envelope quadrature; incoming physical discretization remains independent"}
    corners = np.linalg.norm(_footprint_corners(wave)-centre, axis=1)
    clears_grid = np.all(corners < radius_m if strict_boundary else corners <= radius_m)
    if probability == 0 or clears_grid:
        return wave, {**base, "status": "exact_noop", "output_probability": probability}
    xp = array_module(wave.amplitude)
    if radius_m == 0:
        grid_numerics.check(wave.amplitude.shape, retained_bytes=retained_bytes+wave.amplitude.nbytes)
        result = wave._replace_owned(xp.zeros_like(wave.amplitude))
        return result, {**base, "status": "closed", "output_probability": 0.}
    if not grid_numerics.automatic_refinement:
        raise WaveSamplingError("A circular aperture requires nested sampling admission; automatic refinement is disabled")
    inverse = np.linalg.inv(wave.basis_m)
    shift = inverse@(centre-wave.origin_m)
    radii_in_cells = radius_m*np.linalg.norm(inverse, axis=1)
    counts_xy = np.asarray(wave.amplitude.shape[::-1])
    lower = -(counts_xy//2)+2
    upper = counts_xy-counts_xy//2-2
    local_chart = not (np.any(shift-radii_in_cells < lower) or np.any(shift+radii_in_cells > upper))
    grid_numerics.check(wave.amplitude.shape, retained_bytes=retained_bytes+wave.amplitude.nbytes,
                        working_bytes_per_pixel=384)
    spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(wave.amplitude)))/wave.amplitude.size
    base.update(_input_admission(wave, spectrum, grid_numerics))
    complex_limit = min(numerics.maximum_complex_change_per_input_norm*math.sqrt(probability),
                        grid_numerics.maximum_accumulated_support_l2)
    if complex_limit <= 0:
        raise WaveSamplingError("Circular aperture quadrature requires a positive declared L2 error budget")
    from scipy.fft import next_fast_len
    if local_chart:
        scale = min(1., radius_m/(numerics.minimum_radius_cells*np.linalg.norm(wave.basis_m, ord=2)))
        shape = tuple(2*next_fast_len(max(16, math.ceil(numerics.output_halfwidth_radii*r/scale)))
                      for r in radii_in_cells[::-1])
        chart_origin = centre
        base["chart"] = "local_aperture_enclosure"
    else:
        # A large circle may clip only the corners of the original rectangle.
        # Refine that same physical window, never extend the unknown exterior
        # with the Fourier period. This branch does not claim domain convergence.
        scale, shape, shift = 1., wave.amplitude.shape, np.zeros(2)
        chart_origin = wave.origin_m
        base["chart"] = "original_physical_window"
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    physical_shift = chart_origin-wave.origin_m
    phase = np.exp(2j*np.pi/wavelength_m*(.5*physical_shift@curvature@physical_shift+tilt@physical_shift))
    new_tilt = tilt+.5*(curvature+curvature.T)@physical_shift
    previous = None
    trials = []
    while True:
        _cancel(cancelled)
        held_bytes = retained_bytes+wave.amplitude.nbytes+spectrum.nbytes
        if previous is not None:
            held_bytes += previous.amplitude.nbytes
        workspace_shape = tuple(max(a, b) for a, b in zip(shape, wave.amplitude.shape))
        grid_numerics.check(workspace_shape, retained_bytes=held_bytes, working_bytes_per_pixel=384)
        amplitude = _sample_envelope(spectrum, shape, scale, shift, cancelled)*phase
        basis = wave.basis_m*scale
        yy, xx = xp.meshgrid(xp.arange(shape[0])-shape[0]//2,
                             xp.arange(shape[1])-shape[1]//2, indexing="ij")
        displacement = chart_origin-centre
        radial_squared = ((basis[0, 0]*xx+basis[0, 1]*yy+displacement[0])**2
                          +(basis[1, 0]*xx+basis[1, 1]*yy+displacement[1])**2)
        mask = radial_squared < radius_m**2 if strict_boundary else radial_squared <= radius_m**2
        amplitude = xp.where(mask, amplitude, 0j)
        current = wave._replace_owned(amplitude, basis_m=basis, origin_m=chart_origin, tilt_rad=new_tilt)
        _cancel(cancelled)
        output_probability = current.probability
        trial = {"shape": list(shape), "basis_m": basis.tolist(), "output_probability": output_probability}
        if previous is not None:
            embedded = refine_plane_wave(previous, shape, numerics=grid_numerics,
                                          retained_bytes=held_bytes+current.amplitude.nbytes)
            complex_change = math.sqrt(float(xp.sum(abs(embedded.amplitude-current.amplitude)**2)))
            relative_current = abs(output_probability-previous.probability)/max(output_probability, 1e-300)
            trial.update(complex_change_norm=complex_change, relative_current_change=relative_current)
            del embedded
            if (complex_change <= complex_limit and relative_current <= numerics.maximum_relative_current_change
                    and output_probability <= probability+max(probability*1e-12, 1e-300)):
                trials.append(trial)
                _cancel(cancelled)
                return current, {**base, "status": "converged", "output_probability": output_probability,
                                 "complex_error_norm": complex_change, "maximum_complex_error_norm": complex_limit,
                                 "relative_current_change": relative_current, "output_shape": list(shape),
                                 "trials": trials, "physical_loss_probability": probability-output_probability}
        trials.append(trial)
        previous = current
        scale *= .5
        shape = tuple(2*n for n in shape)
