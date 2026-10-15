"""Admit the executed Cartesian complex field to a centred m=0 chart.

The Cartesian Fourier interpolant is the input, not a newly fitted source.
No phase alignment, Gaussian fit, angular averaging or norm correction is
performed. Unsupported fields retain their ordinary two-dimensional route.
This representation check cannot establish upstream physical convergence.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import CubicSpline

from temsim.physics.radial_column_wave import RadialWave
from temsim.physics.wave_execution import check_available_memory
from temsim.physics.wave_grid import WaveMemoryBudgetError


class CartesianRadialUnavailable(ValueError):
    """The complete executed input has not qualified for an m=0 chart."""

    def __init__(self, message, *, evidence=None):
        super().__init__(message)
        self.evidence = {} if evidence is None else evidence


@dataclass(frozen=True)
class CartesianRadialNumerics:
    maximum_samples: int = 1_048_576
    maximum_working_bytes: int = 512*1024**2
    complex_tolerance: float = 1e-6
    current_tolerance: float = 1e-8
    angular_spokes: int = 16

    def validate(self):
        for name, minimum in (("maximum_samples", 32), ("maximum_working_bytes", 1),
                              ("angular_spokes", 8)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"Cartesian/radial {name} must be an integer >= {minimum}")
        if self.angular_spokes > 128:
            raise ValueError("Cartesian/radial angular_spokes must not exceed 128")
        for name in ("complex_tolerance", "current_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value < .01:
                raise ValueError(f"Cartesian/radial {name} must be positive and below 0.01")
        return self


def cartesian_to_radial(wave, *, wavelength_m, samples=65536,
                        numerics=CartesianRadialNumerics(), retained_bytes=0,
                        cancelled=lambda: False):
    """Return a checked radial field and evidence, or decline this route.

    Each call evaluates the SAME input Fourier interpolant, including when a
    radial executor requests a larger grid. The reference radial envelope is
    its positive lattice-X ray; other angles are tests, never averaged into
    it. Spatial-domain loss, angular checks at two distinct grids, radial
    midpoint interpolation, full Cartesian roundtrip and absolute probability
    are checked independently. The analytical spherical carrier is retained.
    """
    numerics.validate()
    if (isinstance(wavelength_m, bool) or not math.isfinite(wavelength_m)
            or wavelength_m <= 0):
        raise ValueError("Cartesian/radial handoff needs the executed wavelength")
    if type(samples) is not int or samples < 32:
        raise ValueError("Cartesian/radial samples must be an integer >= 32")
    if samples > numerics.maximum_samples:
        raise CartesianRadialUnavailable("Cartesian/radial sample budget exceeded")
    if type(retained_bytes) is not int or retained_bytes < 0:
        raise ValueError("Cartesian/radial retained bytes must be a nonnegative integer")
    amplitude = np.asarray(wave.amplitude)
    if amplitude.ndim != 2 or min(amplitude.shape) < 8:
        raise CartesianRadialUnavailable("Cartesian/radial handoff needs a finite two-dimensional field")
    ny, nx = amplitude.shape
    basis = np.asarray(wave.basis_m, dtype=float)
    if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
        raise CartesianRadialUnavailable("Cartesian/radial handoff needs a finite lattice")
    gram = basis.T@basis
    pitch = math.sqrt(float(gram[0, 0]))
    if not pitch or np.linalg.norm(gram/(pitch*pitch)-np.eye(2), ord=2) > 1e-12:
        raise CartesianRadialUnavailable("A non-similarity Cartesian lattice needs the two-dimensional route")
    origin = np.asarray(wave.origin_m)
    tilt = np.zeros(2) if wave.tilt_rad is None else np.asarray(wave.tilt_rad)
    if np.any(origin != 0.) or np.any(tilt != 0.):
        raise CartesianRadialUnavailable("An off-axis centre or tilt needs the two-dimensional route")
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else np.asarray(wave.curvature_m1)
    if curvature.shape != (2, 2) or not np.all(np.isfinite(curvature)):
        raise CartesianRadialUnavailable("Cartesian/radial handoff needs a finite curvature")
    scalar_curvature = float(np.trace(curvature)/2)
    outer_radius = (min(nx, ny)//2-2)*pitch
    carrier_error = (np.pi/wavelength_m*outer_radius**2
                     *np.linalg.norm(curvature-np.eye(2)*scalar_curvature, ord=2))
    if carrier_error > numerics.complex_tolerance*.1:
        raise CartesianRadialUnavailable("An anisotropic phase carrier needs the two-dimensional route")
    # Includes FFT/input/roundtrip grids, spline coefficients, radial arrays,
    # and bounded Fourier-evaluation rows. This is not a total process cap.
    chunk = 256
    check_count = min(samples, max(256, 4*max(nx, ny)))
    required = (retained_bytes+96*amplitude.size+192*samples
                +48*chunk*(nx+ny)+32*chunk*max(nx, ny)
                +96*2*numerics.angular_spokes*check_count)
    if required > numerics.maximum_working_bytes:
        raise WaveMemoryBudgetError(f"Cartesian/radial handoff needs approximately {required} bytes; "
                                   f"maximum_working_bytes={numerics.maximum_working_bytes}")
    check_available_memory(required-retained_bytes)
    if cancelled():
        raise InterruptedError("Cartesian/radial handoff cancelled")
    if not np.all(np.isfinite(amplitude)):
        raise CartesianRadialUnavailable("Cartesian/radial handoff needs a finite two-dimensional field")
    norm = float(wave.probability)
    if not math.isfinite(norm) or norm <= 0:
        raise CartesianRadialUnavailable("An empty or nonfinite Cartesian mode has no radial domain to qualify")
    area = abs(float(np.linalg.det(basis)))
    density_scale = math.sqrt(area)
    spectrum = np.fft.fft2(np.fft.ifftshift(amplitude))/(nx*ny*density_scale)
    fx, fy = np.fft.fftfreq(nx), np.fft.fftfreq(ny)
    # y=0 in centred lattice coordinates: exact restriction of the full 2-D
    # Fourier interpolant, not interpolation through a new Gaussian.
    line_spectrum = spectrum.sum(axis=0)

    def line(radius):
        result = np.empty(len(radius), complex)
        for first in range(0, len(radius), chunk):
            if cancelled():
                raise InterruptedError("Cartesian/radial Fourier evaluation cancelled")
            last = min(first+chunk, len(radius))
            result[first:last] = np.exp(2j*np.pi*np.outer(radius[first:last]/pitch, fx))@line_spectrum
        return result

    def polar(radius, theta):
        result = np.empty(len(radius), complex)
        for first in range(0, len(radius), chunk):
            if cancelled():
                raise InterruptedError("Cartesian/radial angular check cancelled")
            last = min(first+chunk, len(radius))
            r = radius[first:last]/pitch
            ex = np.exp(2j*np.pi*np.outer(r*np.cos(theta[first:last]), fx))
            ey = np.exp(2j*np.pi*np.outer(r*np.sin(theta[first:last]), fy))
            result[first:last] = np.sum((ey@spectrum)*ex, axis=1)
        return result

    axis_x = np.arange(nx)-nx//2
    axis_y = np.arange(ny)-ny//2
    radius_native = pitch*np.hypot(axis_y[:, None], axis_x[None, :])
    excluded = float(np.sum(abs(amplitude[radius_native >= outer_radius])**2)/norm)
    tail_budget = min((numerics.complex_tolerance*.25)**2, numerics.current_tolerance*.1)
    if excluded > tail_budget:
        raise CartesianRadialUnavailable(f"Cartesian field reaches the square-domain boundary: "
                                         f"excluded probability fraction {excluded:.6g}")
    inner_radius = pitch*1e-8
    radius = np.geomspace(inner_radius, outer_radius, samples, endpoint=False)
    value = line(radius)
    radial = RadialWave(radius, value, scalar_curvature, 0)
    relative_current_error = abs(radial.probability-norm)/norm
    centre_density = abs(line(np.array([0.]))[0])**2
    inner_fraction = float(np.pi*inner_radius**2*centre_density/norm)
    if inner_fraction > tail_budget:
        raise CartesianRadialUnavailable("Cartesian/radial inner domain omits unresolved central probability")
    spline = CubicSpline(np.log(radius), value, extrapolate=False)
    mid = np.sqrt(radius[:-1]*radius[1:])
    delta = spline(np.log(mid))-line(mid)
    interpolation_error = math.sqrt(float(2*np.pi*radial.dln*np.sum(abs(mid*delta)**2)/norm))
    if interpolation_error > numerics.complex_tolerance*.25:
        raise CartesianRadialUnavailable(f"Radial interpolation is unresolved: {interpolation_error:.6g}")

    # A log-integral controls an entire annulus, not merely a few selected
    # intensity maxima. Two interlaced angular grids detect grid-aligned
    # asymmetry; no angular projection is applied to the returned envelope.
    angular_rows = []
    check_radius = np.geomspace(inner_radius, radius[-1], check_count)
    dln_check = math.log(check_radius[-1]/check_radius[0])/(check_count-1)
    reference = line(check_radius)
    for count, offset in ((numerics.angular_spokes, .317), (2*numerics.angular_spokes, .719)):
        angles = (np.arange(count)+offset)*(2*np.pi/count)
        points = np.tile(check_radius, count)
        observed = polar(points, np.repeat(angles, check_count)).reshape(count, check_count)
        difference = observed-reference
        error = math.sqrt(float(2*np.pi*dln_check*np.sum(
            check_radius**2*np.mean(abs(difference)**2, axis=0))/norm))
        angular_rows.append({"spokes": count, "radial_samples": check_count,
                             "relative_complex_error": error})

    reconstructed = np.zeros_like(amplitude, dtype=complex)
    inside = (radius_native >= radius[0]) & (radius_native <= radius[-1])
    reconstructed[inside] = spline(np.log(radius_native[inside]))*density_scale
    # The axis is a known Fourier sample; a log grid never includes r=0.
    reconstructed[ny//2, nx//2] = line(np.array([0.]))[0]*density_scale
    # Preserve the original scalar carrier. Account explicitly for the tiny
    # anisotropic numerical remainder accepted by the bound above.
    xx = basis[0, 0]*axis_x[None, :]+basis[0, 1]*axis_y[:, None]
    yy = basis[1, 0]*axis_x[None, :]+basis[1, 1]*axis_y[:, None]
    residual = curvature-np.eye(2)*scalar_curvature
    carrier_difference = np.pi/wavelength_m*(residual[0, 0]*xx**2
        +(residual[0, 1]+residual[1, 0])*xx*yy+residual[1, 1]*yy**2)
    roundtrip_error = float(np.linalg.norm(reconstructed-amplitude*np.exp(1j*carrier_difference))/math.sqrt(norm))
    if cancelled():
        raise InterruptedError("Cartesian/radial handoff cancelled before publication")
    evidence = {"schema": "executed-cartesian-radial-handoff-v1", "angular_order": 0,
        "input_norm": norm, "output_norm": radial.probability,
        "relative_current_error": relative_current_error,
        "relative_complex_roundtrip_error": roundtrip_error,
        "relative_radial_interpolation_error": interpolation_error,
        "angular_checks": angular_rows, "outer_numerical_tail_fraction": excluded,
        "inner_numerical_tail_estimate_fraction": inner_fraction,
        "carrier_anisotropy_phase_bound_rad": float(carrier_error),
        "radius_min_m": float(radius[0]), "radius_max_m": float(radius[-1]),
        "samples": samples, "estimated_working_bytes": required,
        "complex_tolerance": numerics.complex_tolerance, "current_tolerance": numerics.current_tolerance,
        "source": "Complete executed Cartesian Fourier interpolant",
        "normalisation_correction": False, "phase_fit": False, "angular_average": False,
        "scope": "Representation admission only; upstream and propagated grid/domain convergence remain required"}
    failures = []
    if relative_current_error > numerics.current_tolerance:
        failures.append(f"probability error {relative_current_error:.6g}")
    if any(row["relative_complex_error"]+carrier_error > numerics.complex_tolerance for row in angular_rows):
        failures.append(f"angular complex error {max(row['relative_complex_error'] for row in angular_rows):.6g}")
    if roundtrip_error > numerics.complex_tolerance:
        failures.append(f"complex roundtrip error {roundtrip_error:.6g}")
    if failures:
        raise CartesianRadialUnavailable("Executed Cartesian field has not qualified as m=0: "+", ".join(failures),
                                         evidence=evidence)
    return radial, evidence
