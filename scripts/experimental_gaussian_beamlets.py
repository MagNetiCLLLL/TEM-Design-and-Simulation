"""Diagnostic-only coherent Gaussian beamlets; not a production wave solver.

This module accepts a decomposition of an actual complex field; it supplies
no fitted source, decomposition, probability correction, or physical result
publication. Each normalized packet is N exp(i k[p.(x-q)+(x-q).C.(x-q)/2]).
Its complex coefficient includes the phase at q. Packets interfere coherently:
their summed intensity is not a sum of individual packet intensities.

All states and operations use complex128 on CPU; Numba readout loops are
sequential. A caller must separately establish complex-field reconstruction,
step/packet convergence, physical apertures, and the complete coherent norm.
The local Taylor bound does not qualify a full microscope calculation. This
module is intentionally outside temsim's production solver and UI routes.
"""
from dataclasses import dataclass
import math
import numpy as np

from temsim.physics.canonical_action import (
    principal_reference_phase, reference_frame,
    validate_reference_phase,
)
from temsim.physics.canonical_phase import validate_canonical_map


DIAGNOSTIC_SCOPE = "experimental-coherent-gaussian-packets; no source substitution or production solver routing"


def append_physical_column_step(combined, step):
    """Compose a fully lifted Hamiltonian path through its Gaussian cocycle.

    The standalone winding of a canonical matrix is not its composition
    cocycle with an existing path. The same positive-reference Gaussian
    continuation used by propagate_affine gives that missing phase. This
    also accepts accurately integrated quadratic paths without retaining all
    their small-matrix integration factors. The input path must already own
    its physically resolved lift and affine scalar action.
    """
    if hasattr(step, "correction"):
        raise ValueError("Experimental replay does not admit posed column paths")
    validate_canonical_map(step.matrix)
    validate_reference_phase(step.matrix, step.reference_length_m, step.reference_phase_rad)
    old = combined.matrix
    precision = (old[2:, :2]+1j*old[2:, 2:]/combined.reference_length_m)@_inverse2(
        reference_frame(old, combined.reference_length_m))
    frame = step.matrix[:2, :2]+step.matrix[:2, 2:]@precision
    relative = _inverse2(reference_frame(step.matrix, step.reference_length_m))@frame
    phase = combined.reference_phase_rad+step.reference_phase_rad-.5*float(_eigenphase_sum2(relative))
    matrix = step.matrix@old
    propagated_offset = step.matrix@combined.offset
    displacement = np.asarray(step.offset, float)
    action = combined.action_m+step.action_m+.5*(displacement[2:]@propagated_offset[:2]
        - displacement[:2]@propagated_offset[2:])
    offset = propagated_offset+displacement
    validate_reference_phase(matrix, combined.reference_length_m, phase)
    if not math.isfinite(action) or not np.isfinite(offset).all():
        raise ValueError("Composed canonical action and displacement must be finite")
    combined.matrix, combined.offset = matrix, offset
    combined.action_m, combined.reference_phase_rad = float(action), phase


def _inverse2(values):
    det = values[..., 0, 0]*values[..., 1, 1]-values[..., 0, 1]*values[..., 1, 0]
    result = np.empty_like(values)
    result[..., 0, 0], result[..., 1, 1] = values[..., 1, 1]/det, values[..., 0, 0]/det
    result[..., 0, 1], result[..., 1, 0] = -values[..., 0, 1]/det, -values[..., 1, 0]/det
    return result


def _eigenphase_sum2(values):
    trace = values[..., 0, 0]+values[..., 1, 1]
    det = values[..., 0, 0]*values[..., 1, 1]-values[..., 0, 1]*values[..., 1, 0]
    root = np.sqrt(trace*trace-4*det)
    plus, minus = (trace+root)/2, (trace-root)/2
    first = np.where(abs(plus) >= abs(minus), plus, minus)
    second = det/first
    return np.angle(first)+np.angle(second)


@dataclass(frozen=True)
class GaussianBeamlets:
    centres_m: np.ndarray
    momenta: np.ndarray
    precision_m1: np.ndarray
    coefficients: np.ndarray
    wavelength_m: float

    def __post_init__(self):
        n = len(self.coefficients)
        shapes = {"centres_m": (n, 2), "momenta": (n, 2),
                  "precision_m1": (n, 2, 2), "coefficients": (n,)}
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=complex if name in
                             ("precision_m1", "coefficients") else float, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid Gaussian beamlet {name}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if not math.isfinite(self.wavelength_m) or self.wavelength_m <= 0:
            raise ValueError("Gaussian beamlets need a positive wavelength")
        c = self.precision_m1
        if n:
            if np.max(abs(c-c.swapaxes(-1, -2))) > 1e-10*max(1., np.max(abs(c))):
                raise ValueError("Gaussian precision must be symmetric with positive imaginary part")
            # Sylvester's criterion is exact for symmetric 2x2 matrices. Scale
            # before products to avoid overflow; match eigvalsh's default lower
            # triangle when symmetry differs within the accepted roundoff.
            a, b, d = c[:, 0, 0].imag, c[:, 1, 0].imag, c[:, 1, 1].imag
            scale = np.maximum(np.maximum(abs(a), abs(b)), abs(d))
            scale = np.where(scale > 0., scale, 1.)
            product, square = (a/scale)*(d/scale), (b/scale)**2
            determinant = product-square
            rounding = 8*np.finfo(float).eps*(abs(product)+square)
            if np.any((a <= 0.) | (d <= 0.) | (determinant < -rounding)):
                raise ValueError("Gaussian precision must be symmetric with positive imaginary part")
            ambiguous = determinant <= rounding
            # Keep the existing numerical decision for nearly singular or
            # extremely anisotropic cases instead of adding an epsilon cutoff.
            if np.any(ambiguous) and np.min(np.linalg.eigvalsh(c.imag[ambiguous])) <= 0:
                raise ValueError("Gaussian precision must be symmetric with positive imaginary part")

    @property
    def count(self):
        return len(self.coefficients)

    @property
    def bytes(self):
        return sum(getattr(self, name).nbytes for name in
                   ("centres_m", "momenta", "precision_m1", "coefficients"))

    def evaluate(self, points_m, *, packet_batch=32, point_batch=2048,
                 maximum_working_bytes=256*1024**2, cancelled=lambda: False):
        """Coherent complex amplitude density, with bounded packet/point tiles."""
        points = np.asarray(points_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError("Evaluation points must be finite XY pairs")
        if packet_batch <= 0 or point_batch <= 0:
            raise ValueError("Evaluation batches must be positive")
        required = (self.bytes+points.nbytes+16*len(points)
                    +128*min(self.count, packet_batch)*min(len(points), point_batch))
        if required > maximum_working_bytes:
            raise MemoryError(f"Gaussian evaluation requires {required} bytes within its explicit tile estimate")
        if cancelled():
            raise InterruptedError("Gaussian evaluation cancelled")
        result = np.zeros(len(points), dtype=np.complex128)
        k = 2*np.pi/self.wavelength_m
        normalizers = (k/np.pi)**.5*np.linalg.det(self.precision_m1.imag)**.25
        for first in range(0, len(points), point_batch):
            stop = min(first+point_batch, len(points))
            for p in range(0, self.count, packet_batch):
                if cancelled():
                    raise InterruptedError("Gaussian evaluation cancelled")
                end = min(p+packet_batch, self.count)
                delta = points[None, first:stop]-self.centres_m[p:end, None]
                action = (.5*np.einsum("bpi,bij,bpj->bp", delta, self.precision_m1[p:end], delta)
                          + np.einsum("bi,bpi->bp", self.momenta[p:end], delta))
                values = np.exp(1j*k*action)
                values *= (self.coefficients[p:end]*normalizers[p:end])[:, None]
                result[first:stop] += values.sum(axis=0)
        return result

    def sample_plane(self, shape, basis_m, origin_m, *, amplitude_tail=1e-12,
                     maximum_working_bytes=256*1024**2, cancelled=lambda: False,
                     return_record=False, method="direct", recurrence_reset=32):
        """Fast affine-lattice cell amplitudes with Gaussian local-box bounds."""
        ny, nx = shape
        basis, origin = np.asarray(basis_m, float), np.asarray(origin_m, float)
        if (ny < 1 or nx < 1 or basis.shape != (2, 2) or origin.shape != (2,)
                or not np.isfinite(basis).all() or not np.isfinite(origin).all()
                or np.linalg.det(basis) == 0):
            raise ValueError("Gaussian readout needs a finite invertible affine lattice")
        if not math.isfinite(amplitude_tail) or not 0 < amplitude_tail < 1:
            raise ValueError("Gaussian evaluation tail must be between zero and one")
        _validate_readout_method(method, recurrence_reset)
        required = self.bytes+16*ny*nx+256*self.count
        if required > maximum_working_bytes:
            raise MemoryError("Gaussian lattice evaluation exceeds its bounded allocation estimate")
        if cancelled():
            raise InterruptedError("Gaussian lattice evaluation cancelled")
        k = 2*np.pi/self.wavelength_m
        inverse = _inverse2(basis)
        covariance = _inverse2(self.precision_m1.imag)/(2*k)
        index_covariance = inverse@covariance@inverse.T
        radius = np.sqrt(-4*np.log(amplitude_tail)*np.diagonal(index_covariance, axis1=1, axis2=2))
        index_centres = (self.centres_m-origin)@inverse.T+np.array((nx//2, ny//2))
        normalizers = (k/np.pi)**.5*np.linalg.det(self.precision_m1.imag)**.25
        area = abs(np.linalg.det(basis))
        result = np.zeros((ny, nx), complex)
        for first in range(0, self.count, 256):
            if cancelled():
                raise InterruptedError("Gaussian lattice evaluation cancelled")
            stop = min(self.count, first+256)
            arguments = (result, basis, origin, self.centres_m[first:stop],
                self.momenta[first:stop], self.precision_m1[first:stop],
                self.coefficients[first:stop]*normalizers[first:stop]*math.sqrt(area),
                index_centres[first:stop], radius[first:stop], k)
            if method == "recurrence":
                _lattice_sum_recurrence(*arguments, recurrence_reset)
            else:
                _lattice_sum(*arguments)
        record = {"scope": "Evaluation tail bound; excludes viewport truncation, decomposition and pixel quadrature",
            "amplitude_tail": amplitude_tail,
            "coherent_absolute_l2_tail_bound": float(abs(self.coefficients).sum()*amplitude_tail),
            "pointwise_cell_amplitude_error_bound": float(np.dot(abs(self.coefficients), normalizers)*amplitude_tail*math.sqrt(area)),
            "packet_count": self.count, "estimated_working_bytes": required,
            "evaluation_method": method, "recurrence_reset_pixels": recurrence_reset if method == "recurrence" else None}
        return (result, record) if return_record else result

    def evaluate_grid(self, x_axis_m, y_axis_m, *, amplitude_tail=1e-12,
                      maximum_working_bytes=256*1024**2, cancelled=lambda: False,
                      method="direct", recurrence_reset=32):
        """Serial bounded local-box Gaussian summation on a uniform XY grid.

        The complex density is returned with a separate analytic omitted-tail
        bound. Output viewport truncation and grid quadrature remain separate.
        No beam probabilities or coefficients are changed by this readout.
        """
        x, y = np.asarray(x_axis_m, float), np.asarray(y_axis_m, float)
        for axis in (x, y):
            if (axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all()
                    or axis[1] <= axis[0] or not np.allclose(np.diff(axis), axis[1]-axis[0], rtol=1e-10, atol=0.)):
                raise ValueError("Gaussian grid axes must be finite, increasing and uniform")
        if not math.isfinite(amplitude_tail) or not 0 < amplitude_tail < 1:
            raise ValueError("Gaussian evaluation tail must be between zero and one")
        _validate_readout_method(method, recurrence_reset)
        required = self.bytes+16*len(x)*len(y)+256*self.count
        if required > maximum_working_bytes:
            raise MemoryError("Gaussian grid evaluation exceeds its bounded allocation estimate")
        if cancelled():
            raise InterruptedError("Gaussian grid evaluation cancelled")
        k = 2*np.pi/self.wavelength_m
        cov = _inverse2(self.precision_m1.imag)/(2*k)
        radius = np.sqrt(-4*np.log(amplitude_tail)*np.diagonal(cov, axis1=1, axis2=2))
        normalizers = (k/np.pi)**.5*np.linalg.det(self.precision_m1.imag)**.25
        result = np.zeros((len(y), len(x)), complex)
        # Batches retain cooperative cancellation without Python per-pixel work.
        for first in range(0, self.count, 256):
            if cancelled():
                raise InterruptedError("Gaussian grid evaluation cancelled")
            stop = min(self.count, first+256)
            arguments = (result, x[0], x[1]-x[0], y[0], y[1]-y[0],
                self.centres_m[first:stop], self.momenta[first:stop], self.precision_m1[first:stop],
                self.coefficients[first:stop]*normalizers[first:stop], radius[first:stop], k)
            if method == "recurrence":
                _grid_sum_recurrence(*arguments, recurrence_reset)
            else:
                _grid_sum(*arguments)
        return result, {
            "scope": "Evaluation tail bound for normalized Gaussian packets; excludes viewport truncation, decomposition and pixel quadrature",
            "amplitude_tail": amplitude_tail,
            "coherent_absolute_l2_tail_bound": float(abs(self.coefficients).sum()*amplitude_tail),
            "pointwise_amplitude_density_error_bound": float(np.dot(abs(self.coefficients), normalizers)*amplitude_tail),
            "packet_count": self.count, "estimated_working_bytes": required,
            "evaluation_method": method, "recurrence_reset_pixels": recurrence_reset if method == "recurrence" else None}


def _validate_readout_method(method, recurrence_reset):
    if method not in ("direct", "recurrence"):
        raise ValueError("Gaussian readout method must be direct or recurrence")
    if type(recurrence_reset) is not int or not 1 <= recurrence_reset <= 256:
        raise ValueError("Gaussian recurrence reset must be an integer from 1 to 256")


def propagate_affine(packets, matrix, offset=None, *, affine_action_m=0.,
                     reference_phase_rad=None, reference_length_m=1.):
    """Exact normalized Gaussian metaplectic/Weyl transport, including lift.

    The supplied lift fixes the reference positive Gaussian. Continuation from
    that reference to each positive complex precision has no singularity;
    summing individual relative eigenphases avoids determinant wrapping.
    """
    validate_canonical_map(matrix)
    matrix = np.asarray(matrix, float)
    offset = np.zeros(4) if offset is None else np.asarray(offset, float)
    if offset.shape != (4,) or not np.isfinite(offset).all() or not math.isfinite(affine_action_m):
        raise ValueError("Canonical displacement and action must be finite")
    if reference_phase_rad is None:
        reference_phase_rad = principal_reference_phase(matrix, reference_length_m)
    validate_reference_phase(matrix, reference_length_m, reference_phase_rad)
    a, b, c, d = matrix[:2, :2], matrix[:2, 2:], matrix[2:, :2], matrix[2:, 2:]
    frame = a+b@packets.precision_m1
    precision = (c+d@packets.precision_m1)@_inverse2(frame)
    relative = _inverse2(reference_frame(matrix, reference_length_m))@frame
    lift = reference_phase_rad-.5*_eigenphase_sum2(relative)
    mapped = np.concatenate((packets.centres_m, packets.momenta), axis=1)@matrix.T
    phase_space = mapped+offset
    action = (affine_action_m+.5*(np.sum(mapped[:, 2:]*mapped[:, :2], axis=1)
                                 -np.sum(packets.momenta*packets.centres_m, axis=1))
              +mapped[:, :2]@offset[2:]+.5*offset[2:]@offset[:2])
    coefficients = packets.coefficients*np.exp(1j*(2*np.pi/packets.wavelength_m*action+lift))
    return GaussianBeamlets(phase_space[:, :2], phase_space[:, 2:], precision,
                            coefficients, packets.wavelength_m)


def _residual_rms_action(packets, hn, hs, cs):
    """Exact Gaussian E[R^2] for the cubic/quartic Taylor remainder."""
    k = 2*np.pi/packets.wavelength_m
    covariance = _inverse2(packets.precision_m1.imag)/(2*k)
    xx, xy, yy = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    cache = {(0, 0): np.ones(packets.count)}
    def moment(a, b):
        if a < 0 or b < 0 or (a+b) % 2:
            return np.zeros(packets.count)
        if (a, b) not in cache:
            cache[a, b] = ((a-1)*xx*moment(a-2, b)+b*xy*moment(a-1, b-1)
                            if a else (b-1)*yy*moment(0, b-2))
        return cache[a, b]
    x, y = packets.centres_m.T
    terms = {(3, 0): -hn/3-cs*x, (2, 1): -hs-cs*y,
             (1, 2): hn-cs*x, (0, 3): hs/3-cs*y,
             (4, 0): -cs/4, (2, 2): -cs/2, (0, 4): -cs/4}
    terms = {key: value for key, value in terms.items() if np.any(value != 0)}
    squared = np.zeros(packets.count)
    for (a, b), first in terms.items():
        for (c, d), second in terms.items():
            squared += first*second*moment(a+c, b+d)
    # Positive moments can cancel at floating precision for special strengths.
    result = np.sqrt(np.maximum(squared, 0.))
    # The recursive local function closes over its own cell and this cache.
    # Break that cycle explicitly: repeated 100k-packet screens must release
    # their moment arrays immediately, independently of cyclic GC timing.
    cache.clear()
    del moment
    return result


def apply_taylor_screen(packets, *, normal_m2=0., skew_m2=0., spherical_m3=0.,
                       include_packet_bounds=False):
    """Local quadratic screen with a conservative coherent L2 residual bound.

    This is an approximation of the shared cubic/quartic phase operator. The
    error is bounded per normalized packet using |exp(i k R)-1|<=|k R|; a
    triangle sum of |coefficient|*packet_error bounds the coherent superposition.
    Packets are not independent particles. No probability is renormalized.
    """
    from temsim.physics.multipole_wave import multipole_action
    hn, hs, cs = normal_m2, skew_m2, spherical_m3
    if not all(math.isfinite(v) for v in (hn, hs, cs)):
        raise ValueError("Gaussian phase-screen strengths must be finite")
    x, y = packets.centres_m.T
    r2 = x*x+y*y
    action = multipole_action(x, y, normal_m2=hn, skew_m2=hs, spherical_m3=cs)
    gradient = np.stack((-hn*(x*x-y*y)-2*hs*x*y-cs*r2*x,
                          2*hn*x*y-hs*(x*x-y*y)-cs*r2*y), axis=1)
    hessian = np.empty_like(packets.precision_m1.real)
    hessian[:, 0, 0] = -2*hn*x-2*hs*y-cs*(3*x*x+y*y)
    hessian[:, 1, 1] = 2*hn*x+2*hs*y-cs*(x*x+3*y*y)
    hessian[:, 0, 1] = hessian[:, 1, 0] = 2*hn*y-2*hs*x-2*cs*x*y
    k = 2*np.pi/packets.wavelength_m
    result = GaussianBeamlets(packets.centres_m, packets.momenta+gradient,
        packets.precision_m1+hessian, packets.coefficients*np.exp(1j*k*action), packets.wavelength_m)
    bound = np.minimum(2., k*_residual_rms_action(packets, hn, hs, cs))
    evidence = {"scope": "Experimental Gaussian Taylor-screen residual only; excludes decomposition and path discretisation errors",
        "maximum_packet_l2_error_upper_bound": float(np.max(bound, initial=0.)),
        "coefficient_l1_norm": float(abs(packets.coefficients).sum()),
        "coherent_absolute_l2_error_upper_bound": float(np.dot(abs(packets.coefficients), bound)),
        "probability_renormalized": False, "packet_count": packets.count}
    if include_packet_bounds:
        evidence["packet_l2_error_upper_bounds"] = bound.tolist()
    return result, evidence


from numba import njit


@njit(cache=True)
def _grid_sum(output, x0, dx, y0, dy, centres, momenta, precisions, coefficients, radius, k):
    ny, nx = output.shape
    for packet in range(len(coefficients)):
        cx, cy = centres[packet, 0], centres[packet, 1]
        low_x = max(0, int(math.ceil((cx-radius[packet, 0]-x0)/dx)))
        high_x = min(nx, int(math.floor((cx+radius[packet, 0]-x0)/dx))+1)
        low_y = max(0, int(math.ceil((cy-radius[packet, 1]-y0)/dy)))
        high_y = min(ny, int(math.floor((cy+radius[packet, 1]-y0)/dy))+1)
        for iy in range(low_y, high_y):
            v = y0+iy*dy-cy
            for ix in range(low_x, high_x):
                u = x0+ix*dx-cx
                action = (momenta[packet, 0]*u+momenta[packet, 1]*v
                    +.5*(precisions[packet, 0, 0]*u*u
                    +(precisions[packet, 0, 1]+precisions[packet, 1, 0])*u*v
                    +precisions[packet, 1, 1]*v*v))
                output[iy, ix] += coefficients[packet]*np.exp(1j*k*action)


@njit(cache=True)
def _lattice_sum(output, basis, origin, centres, momenta, precisions, coefficients, index_centres, radius, k):
    ny, nx = output.shape
    for packet in range(len(coefficients)):
        cx, cy = index_centres[packet, 0], index_centres[packet, 1]
        low_x = max(0, int(math.ceil(cx-radius[packet, 0])))
        high_x = min(nx, int(math.floor(cx+radius[packet, 0]))+1)
        low_y = max(0, int(math.ceil(cy-radius[packet, 1])))
        high_y = min(ny, int(math.floor(cy+radius[packet, 1]))+1)
        for iy in range(low_y, high_y):
            for ix in range(low_x, high_x):
                u = origin[0]+basis[0, 0]*(ix-nx//2)+basis[0, 1]*(iy-ny//2)-centres[packet, 0]
                v = origin[1]+basis[1, 0]*(ix-nx//2)+basis[1, 1]*(iy-ny//2)-centres[packet, 1]
                action = (momenta[packet, 0]*u+momenta[packet, 1]*v
                    +.5*(precisions[packet, 0, 0]*u*u
                    +(precisions[packet, 0, 1]+precisions[packet, 1, 0])*u*v
                    +precisions[packet, 1, 1]*v*v))
                output[iy, ix] += coefficients[packet]*np.exp(1j*k*action)


@njit(inline="always")
def _add_quadratic_row(output, iy, low, high, u0, v0, du, dv,
                       p0, p1, c00, cross, c11, coefficient, k, reset):
    second = c00*du*du+cross*du*dv+c11*dv*dv
    step_exponent = 1j*k*second
    # Strongly anisotropic packets can be underflowed at a bounding-box
    # corner and finite farther along the same row. Fall back to direct
    # exponentials there; multiplying a zero start cannot recover that peak.
    safe_step = abs(step_exponent.real) < 2.
    step_ratio = np.exp(step_exponent) if safe_step else 0j
    for start in range(low, high, reset):
        stop = min(start+reset, high)
        u, v = u0+start*du, v0+start*dv
        action = p0*u+p1*v+.5*(c00*u*u+cross*u*v+c11*v*v)
        derivative = p0*du+p1*dv+c00*u*du+.5*cross*(u*dv+v*du)+c11*v*dv
        exponent = 1j*k*action
        increment = 1j*k*(derivative+.5*second)
        safe = safe_step and exponent.real > -600. and abs(increment.real) < 50.
        if safe:
            value = coefficient*np.exp(exponent)
            ratio = np.exp(increment)
            for ix in range(start, stop):
                output[iy, ix] += value
                value *= ratio
                ratio *= step_ratio
        else:
            for ix in range(start, stop):
                u, v = u0+ix*du, v0+ix*dv
                action = p0*u+p1*v+.5*(c00*u*u+cross*u*v+c11*v*v)
                output[iy, ix] += coefficient*np.exp(1j*k*action)


@njit(cache=True)
def _grid_sum_recurrence(output, x0, dx, y0, dy, centres, momenta, precisions, coefficients, radius, k, reset):
    ny, nx = output.shape
    for packet in range(len(coefficients)):
        cx, cy = centres[packet, 0], centres[packet, 1]
        low_x = max(0, int(math.ceil((cx-radius[packet, 0]-x0)/dx)))
        high_x = min(nx, int(math.floor((cx+radius[packet, 0]-x0)/dx))+1)
        low_y = max(0, int(math.ceil((cy-radius[packet, 1]-y0)/dy)))
        high_y = min(ny, int(math.floor((cy+radius[packet, 1]-y0)/dy))+1)
        for iy in range(low_y, high_y):
            _add_quadratic_row(output, iy, low_x, high_x, x0-cx, y0+iy*dy-cy, dx, 0.,
                momenta[packet, 0], momenta[packet, 1], precisions[packet, 0, 0],
                precisions[packet, 0, 1]+precisions[packet, 1, 0], precisions[packet, 1, 1],
                coefficients[packet], k, reset)


@njit(cache=True)
def _lattice_sum_recurrence(output, basis, origin, centres, momenta, precisions, coefficients, index_centres, radius, k, reset):
    ny, nx = output.shape
    for packet in range(len(coefficients)):
        cx, cy = index_centres[packet, 0], index_centres[packet, 1]
        low_x = max(0, int(math.ceil(cx-radius[packet, 0])))
        high_x = min(nx, int(math.floor(cx+radius[packet, 0]))+1)
        low_y = max(0, int(math.ceil(cy-radius[packet, 1])))
        high_y = min(ny, int(math.floor(cy+radius[packet, 1]))+1)
        for iy in range(low_y, high_y):
            u0 = origin[0]-basis[0, 0]*(nx//2)+basis[0, 1]*(iy-ny//2)-centres[packet, 0]
            v0 = origin[1]-basis[1, 0]*(nx//2)+basis[1, 1]*(iy-ny//2)-centres[packet, 1]
            _add_quadratic_row(output, iy, low_x, high_x, u0, v0, basis[0, 0], basis[1, 0],
                momenta[packet, 0], momenta[packet, 1], precisions[packet, 0, 0],
                precisions[packet, 0, 1]+precisions[packet, 1, 0], precisions[packet, 1, 1],
                coefficients[packet], k, reset)
