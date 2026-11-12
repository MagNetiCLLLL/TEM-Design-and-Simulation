"""Declared mixed-state lift of executed particles near the specimen.

The classical trajectories do not determine a unique mutual intensity. This
boundary assigns the validated minimum-uncertainty Gaussian mixture, followed
by an optional independent source-image translation channel. Original current,
energy and particle ancestry remain explicit; no Tip wave calculation runs.
"""
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.constants import c, e, h, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.optics.aberration_basis import CARTESIAN_NAMES, polynomial_basis
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_checkpoint import EMITTED_ELECTRON_REFERENCE, WaveCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


_PHASE_FIELDS = ("x_m", "y_m", "tx_rad", "ty_rad", "ray_id", "weight",
                 "kinetic_energy_ev", "flight_time_s")
_CANONICAL_SHIFT = "canonical_frequency_shift_inv_angstrom"


def _frozen(value, dtype=float):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class MixedSourceNumerics:
    pixels: int = 512
    width_nm: float = 4.
    maximum_modes: int = 96
    maximum_mode_loss_fraction: float = 1e-5
    maximum_working_bytes: int = 24*1024**3
    source_image_rms_pm: float = 10.
    spatial_quadrature_order: int = 5
    frequency_support_inv_angstrom: float = 1.90

    def validate(self):
        for name, lower, upper in (("pixels", 32, 8192), ("maximum_modes", 1, 4096),
                                   ("spatial_quadrature_order", 1, 32)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.pixels % 2:
            raise ValueError("Mixed-state wave pixels must be even")
        if type(self.maximum_working_bytes) is not int or self.maximum_working_bytes <= 0:
            raise ValueError("Mixed-state workspace bytes must be a positive integer")
        for name in ("width_nm", "frequency_support_inv_angstrom", "maximum_mode_loss_fraction"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.maximum_mode_loss_fraction > 1e-5:
            raise ValueError("Mixed-state mode loss cannot exceed the existing 1e-5 budget")
        if (isinstance(self.source_image_rms_pm, (bool, np.bool_))
                or not np.isfinite(self.source_image_rms_pm) or self.source_image_rms_pm < 0):
            raise ValueError("Source-image RMS must be finite and nonnegative")
        if self.source_image_rms_pm and self.spatial_quadrature_order < 3:
            raise ValueError("Nonzero Gaussian source-image RMS requires at least three quadrature nodes per axis")
        return self


@dataclass(frozen=True)
class ParticleWaveSource:
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    ray_id: np.ndarray
    weight: np.ndarray
    kinetic_energy_ev: np.ndarray
    flight_time_s: np.ndarray
    z_mm: float
    reference_current_a: float
    original_positive_count: int
    input_digest: str
    provenance: object = field(default_factory=dict)
    canonical_frequency_shift_inv_angstrom: np.ndarray | None = None

    def __post_init__(self):
        ids = np.asarray(self.ray_id)
        if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
                or np.any(ids > np.iinfo(np.int64).max)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("Particle wave source requires unique nonnegative original particle IDs")
        shape = ids.shape
        for name in _PHASE_FIELDS:
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Particle wave source {name} must be a matching finite vector")
            object.__setattr__(self, name, _frozen(value, np.int64 if name == "ray_id" else float))
        if np.any(self.weight < 0) or not 0 < math.fsum(self.weight) <= 1.+1e-10:
            raise ValueError("Particle wave source must retain positive absolute source fractions without renormalization")
        if np.any(self.kinetic_energy_ev <= 0) or np.any(self.flight_time_s < 0):
            raise ValueError("Particle wave source requires positive energy and actual nonnegative flight times")
        if (not np.isfinite(self.z_mm) or self.z_mm <= 0 or not np.isfinite(self.reference_current_a)
                or self.reference_current_a <= 0):
            raise ValueError("Particle wave source requires a finite physical plane and positive reference current")
        if (type(self.original_positive_count) is not int or self.original_positive_count < 1000
                or self.original_positive_count < np.count_nonzero(self.weight > 0)):
            raise ValueError("New mixed-state calculations require at least 1000 original positive-current source particles")
        if (not isinstance(self.input_digest, str) or len(self.input_digest) != 64
                or set(self.input_digest)-set("0123456789abcdef")):
            raise ValueError("Particle wave source requires its immutable instrument input digest")
        shift = self.canonical_frequency_shift_inv_angstrom
        if shift is not None:
            shift = np.asarray(shift, dtype=float)
            if shift.shape != (len(ids), 2) or not np.isfinite(shift).all():
                raise ValueError("Canonical frequency shift must be a matching finite Nx2 cycles/angstrom array")
            object.__setattr__(self, _CANONICAL_SHIFT, _frozen(shift))
        object.__setattr__(self, "provenance", freeze_json(self.provenance))

    @property
    def source_fraction(self):
        return math.fsum(self.weight)

    @property
    def mean_energy_ev(self):
        return float(np.average(self.kinetic_energy_ev, weights=self.weight))

    @property
    def digest(self):
        digest = sha256()
        for name in _PHASE_FIELDS:
            value = getattr(self, name)
            digest.update(name.encode())
            digest.update(str((value.dtype.str, value.shape)).encode())
            digest.update(value.tobytes())
        shift = self.canonical_frequency_shift_inv_angstrom
        digest.update(_CANONICAL_SHIFT.encode())
        if shift is None:
            digest.update(b"not-supplied")
        else:
            digest.update(str((shift.dtype.str, shift.shape)).encode())
            digest.update(shift.tobytes())
        return json_digest({"arrays": digest.hexdigest(), "z_mm": self.z_mm,
            "reference_current_a": self.reference_current_a,
            "original_positive_count": self.original_positive_count,
            "input_digest": self.input_digest, "provenance": self.provenance})


def _check_cancelled(cancelled):
    if cancelled():
        raise InterruptedError("Particle-to-wave calculation cancelled; no partial checkpoint published")


def _report(callback, completed, total, text):
    if callback is not None:
        callback(completed, total, text)


def square_root(a, *, inverse=False):
    values, vectors = np.linalg.eigh((a+a.T)*.5)
    if not np.isfinite(values).all() or np.min(values) <= 0:
        raise ValueError("The measured source covariance does not admit the declared Gaussian lift")
    return (vectors * values**(-.5 if inverse else .5)) @ vectors.T


def threefold_basis(u, degree):
    """Value and gradient of the A2 (degree 3) or D4 (degree 5) phase basis."""
    x, y = u.T
    q, r2 = x+1j*y, x*x+y*y
    value, dx, dy = q**3/degree, 3*q**2/degree, 3j*q**2/degree
    if degree == 5:
        dx, dy = 2*x*value+r2*dx, 2*y*value+r2*dy
        value *= r2
    derivative = np.stack((dx, dy), axis=1)
    return np.column_stack((value.real, value.imag)), np.stack((derivative.real, derivative.imag), axis=2)


def angular_phase_model(positions, frequency, weight, ids, wavelength):
    """Separate the irrotational A2/D4 contribution from the finite-source residual."""
    center = np.average(frequency, weights=weight, axis=0)
    offset = frequency-center
    scale = float(np.max(np.linalg.norm(offset, axis=1)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The particle source has no resolved angular support for a mixed-state phase fit")
    u = offset/scale
    gradient = polynomial_basis(u, gradient=True)
    intercept = np.broadcast_to(np.eye(2), (len(u), 2, 2))
    d4_gradient = threefold_basis(u, 5)[1]
    design = np.concatenate((intercept, gradient, d4_gradient), axis=2)
    ordered = np.argsort(ids)
    holdout = np.random.default_rng(7614).permutation(ordered)[:len(u)//5]
    train = np.ones(len(u), dtype=bool)
    train[holdout] = False
    if min(train.sum(), (~train).sum()) < len(CARTESIAN_NAMES):
        raise ValueError("Insufficient original particles for the independent phase-fit split")
    sqrt_weight = np.sqrt(weight)

    def fit(basis, target, subset):
        a = basis[subset] * sqrt_weight[subset, None, None]
        b = target[subset] * sqrt_weight[subset, None]
        values, _, rank, singular = np.linalg.lstsq(a.reshape(-1, a.shape[-1]), b.ravel(), rcond=1e-11)
        if rank != a.shape[-1] or singular[0]/singular[-1] > 1e6:
            raise ValueError("Unresolved angular phase fit")
        return values

    def rms(delta):
        return float(np.sqrt(np.average(np.sum(delta[~train]**2, axis=1), weights=weight[~train])))

    held_out = fit(design, positions, train)
    unrestricted = np.column_stack([u[:, 0]**i * u[:, 1]**(degree-i)
                                   for degree in range(6) for i in range(degree+1)])
    vector_coeff, *_ = np.linalg.lstsq(unrestricted[train]*sqrt_weight[train, None],
                                      positions[train]*sqrt_weight[train, None], rcond=1e-11)
    coefficients = fit(design, positions, np.ones(len(u), dtype=bool))[2:]
    a2 = [CARTESIAN_NAMES.index(name) for name in ("A2_x", "A2_y")]
    fit_info = {
        "held_out_particles": int((~train).sum()),
        "holdout_rms_position_about_centroid_angstrom": rms(positions-np.average(positions[train], weights=weight[train], axis=0)),
        "holdout_rms_curl_free_gradient_angstrom": rms(positions-design @ held_out),
        "holdout_rms_unrestricted_vector_polynomial_angstrom": rms(positions-unrestricted @ vector_coeff),
        "holdout_rms_gradient_vector_disagreement_angstrom": rms(design @ held_out-unrestricted @ vector_coeff),
        "finite_source_fit_A2_quadratures_angstrom": coefficients[a2].tolist(),
        "finite_source_fit_D4_quadratures_angstrom": coefficients[-2:].tolist(),
        "finite_source_gradient_chart_radius_inv_angstrom": scale,
    }
    terms = {3: coefficients[a2], 5: coefficients[-2:]}
    fit_info["applied_phase_source"] = "Held-out-checked finite-source angular regression"
    fit_info.update(applied_phase_frequency_center_inv_angstrom=center.tolist(),
                    applied_phase_chart_radius_inv_angstrom=scale,
                    applied_scaled_A2_angstrom=terms[3].tolist(), applied_scaled_D4_angstrom=terms[5].tolist())

    def phase(freq, *, derivative=False):
        # Retain the measured polynomials within the angular chart and extend
        # their value and radial derivative linearly into the Gaussian tails.
        chart = (freq-center)/scale
        radial = np.linalg.norm(chart, axis=1)
        radial_denominator = np.maximum(1., radial)
        boundary = chart/radial_denominator[:, None]
        total = np.zeros_like(chart) if derivative else np.zeros(len(chart))
        for degree, quadratures in terms.items():
            values, gradients = threefold_basis(boundary, degree)
            p = values @ quadratures
            slope = gradients @ quadratures
            extension = 1.+degree*np.maximum(0., radial-1.)
            if derivative:
                radial_part = (degree*p)[:, None]*boundary
                total += radial_part + (extension/radial_denominator)[:, None]*(slope-radial_part)
            else:
                total += p*extension
        return total if derivative else -2.*np.pi*scale*total

    return phase(frequency, derivative=True), phase, fit_info


def lifted_centers(phase, wavelength, probability):
    weights = probability/probability.sum()
    position = np.column_stack((phase["x_m"], phase["y_m"]))*1e10
    slopes = np.column_stack((phase["tx_rad"], phase["ty_rad"]))
    frequency = slopes / np.sqrt(1.+np.sum(slopes**2, axis=1))[:, None] / wavelength
    shift = phase.get(_CANONICAL_SHIFT)
    if shift is not None:
        shift = np.asarray(shift, dtype=float)
        if shift.shape != frequency.shape or not np.isfinite(shift).all():
            raise ValueError("Canonical frequency shift does not match the lifted particle population")
        frequency += shift
    displacement, phase_function, fit = angular_phase_model(position, frequency, weights, phase["ray_id"], wavelength)
    values = np.column_stack((position-displacement, frequency))
    center = np.average(values, weights=weights, axis=0)
    centered = values-center
    covariance = (centered.T * weights) @ centered
    sqrt_covariance = square_root(covariance)
    zero, identity = np.zeros((2, 2)), np.eye(2)
    symplectic = np.block([[zero, identity], [-identity, zero]])
    skew = sqrt_covariance @ symplectic @ sqrt_covariance
    modulus = square_root(-skew @ skew)
    vacuum_uncertainty = 1./(4.*np.pi)
    packet = vacuum_uncertainty*sqrt_covariance @ np.linalg.inv(modulus) @ sqrt_covariance
    np.testing.assert_allclose(packet @ symplectic @ packet, vacuum_uncertainty**2*symplectic, rtol=0, atol=1e-12)
    centers = centered @ (square_root(covariance-packet) @ np.linalg.inv(sqrt_covariance)).T+center
    real_covariance = packet[:2, :2]
    chirp = packet[2:, :2] @ np.linalg.inv(real_covariance)
    np.testing.assert_allclose(chirp, chirp.T, rtol=0, atol=1e-12)
    quadratic = .25*np.linalg.inv(real_covariance)-1j*np.pi*chirp
    provenance = {
        "method": "Positive mixture of chirped minimum-uncertainty Gaussian packets with covariance-matched residual coordinates, followed by a common A2 and D4 angular phase",
        "coherence": "A ray defines one packet center in the declared lift. The finite-source rays are mutually incoherent; amplitudes interfere within each packet. Complex relative source phases are not inferred.",
        "covariance_units": "Coordinates (x,y) in angstrom and reciprocal coordinates (fx,fy) in cycles/angstrom",
        "deaberrated_geometric_covariance": covariance.tolist(),
        "minimum_uncertainty_packet_covariance": packet.tolist(),
        "phase_fit": fit,
    }
    normalization = np.linalg.det(2.*np.pi*real_covariance)**(-.25) * np.pi/np.sqrt(np.linalg.det(quadratic))
    return centers, normalization, np.linalg.inv(quadratic), phase_function, provenance


def _admit_workspace(rows, columns, numerics, *, retained_bytes=0):
    from temsim.physics.wave_device import check_device_memory
    from temsim.physics.host_memory_budget import check_host_allocation
    # Complex128 factors, Hermitian Gram/eigensolver work and projected modes.
    required = int(retained_bytes+16*(3*rows*columns+4*columns**2)
                   +48*numerics.pixels**2*numerics.maximum_modes)
    if required > numerics.maximum_working_bytes:
        raise MemoryError(f"Particle mixed-state decomposition needs approximately {required} bytes, "
                          f"above its {numerics.maximum_working_bytes}-byte working limit")
    check_device_memory(required)
    check_host_allocation(16*columns**2+32*numerics.pixels**2*numerics.maximum_modes,
                          requested_maximum_bytes=numerics.maximum_working_bytes)
    return required


def _compress_factor(cp, factor, expected_trace, allowed_loss, numerics, cancelled):
    """Diagonalise a positive density factor without restoring omitted power."""
    _check_cancelled(cancelled)
    gram = factor.conj().T @ factor
    trace = float(cp.trace(gram).real.item())
    if not np.isfinite(trace) or abs(trace-expected_trace) > 2e-6*expected_trace:
        raise ValueError("The Gaussian support or source-current trace is unresolved; refine the wave grid")
    eigenvalues, eigenvectors = cp.linalg.eigh(gram)
    _check_cancelled(cancelled)
    values = cp.asnumpy(eigenvalues)[::-1]
    if not np.isfinite(values).all() or values[-1] < -1e-12*expected_trace:
        raise ValueError("The mixed-state density is not numerically positive semidefinite")
    target = max(0., expected_trace-allowed_loss*.99)
    count = int(np.searchsorted(np.cumsum(values), target)+1)
    if not 1 <= count <= min(numerics.maximum_modes, len(values)) or values[count-1] <= 0:
        raise ValueError("The density cannot be represented within the declared mode-count and trace budgets")
    projected = factor @ eigenvectors[:, -1:-count-1:-1]
    powers = cp.sum(cp.abs(projected)**2, axis=0, dtype=cp.float64)
    weights = cp.asnumpy(powers)
    if (not np.isfinite(weights).all() or np.min(weights) <= 0
            or expected_trace-float(weights.sum()) > allowed_loss
            or float(weights.sum()) > expected_trace+1e-10*expected_trace):
        raise ValueError("The actual compressed mode powers exceed the current or omitted-trace budget")
    projected /= cp.sqrt(powers)[None, :]
    return projected, weights, {"input_trace": expected_trace, "discretized_factor_trace": trace,
        "retained_trace": float(weights.sum()), "discarded_trace": max(0., expected_trace-float(weights.sum())),
        "retained_modes": count, "allowed_discarded_trace": allowed_loss,
        "retained_density_eigenvalues_fraction_of_emitted": values[:count].tolist()}


def _unitary_density_rebasis(cp, modes, weights, cancelled):
    """Express the identical density as positive nonorthogonal components.

    If F contains the weighted eigenmodes, G=FQ for a unitary mode-axis DFT
    satisfies GG*=FF*. Each new component receives its own actual column
    norm. Factoring that norm into weight and unit shape restores no current.
    The original density eigenvalues remain the authority for modal purity.
    """
    _check_cancelled(cancelled)
    factor = modes*cp.sqrt(cp.asarray(weights))[None, :]
    before = cp.sum(cp.abs(factor)**2, axis=1)
    transformed = cp.fft.fft(factor, axis=1, norm="ortho")
    recovered = cp.fft.ifft(transformed, axis=1, norm="ortho")
    trace = float(weights.sum())
    roundtrip = math.sqrt(float(cp.sum(cp.abs(recovered-factor)**2).item())/trace)
    diagonal = cp.sum(cp.abs(transformed)**2, axis=1)
    diagonal_error = float(cp.sum(cp.abs(diagonal-before)).item())/trace
    powers = cp.sum(cp.abs(transformed)**2, axis=0, dtype=cp.float64)
    component_weights = cp.asnumpy(powers)
    trace_error = abs(float(component_weights.sum())-trace)/trace
    if (not np.isfinite(component_weights).all() or np.any(component_weights <= 0)
            or roundtrip > 5e-13 or diagonal_error > 5e-13 or trace_error > 5e-13):
        raise ValueError("Unitary mixed-density re-expression failed its factor, intensity or current invariant")
    transformed /= cp.sqrt(powers)[None, :]
    _check_cancelled(cancelled)
    return transformed, component_weights, {
        "method": "Unitary DFT of the weighted density factor along the component axis",
        "density_identity": "G=F*Q with unitary Q; G*G^H=F*F^H",
        "components_orthogonal": False,
        "component_count_unchanged": len(component_weights) == len(weights),
        "factor_roundtrip_relative_l2_error": roundtrip,
        "angular_density_diagonal_relative_l1_error": diagonal_error,
        "trace_relative_error": trace_error,
        "propagation_component_weights_fraction_of_emitted": component_weights.tolist(),
        "component_phase_scope": "A mathematical pure-state component; not a unique phase of the mixed beam",
        "normalization_scope": "Actual column norm factored into component weight and unit shape; no common current renormalization",
        "purity_scope": "Use retained density eigenvalues; squared propagation component weights are not the density purity"}


def _translation_channel(cp, modes, weights, reciprocal, numerics, source_fraction, cancelled):
    """Independent local Gaussian displacement; preserve the angular diagonal."""
    sigma = numerics.source_image_rms_pm/100.  # picometres -> angstrom
    record = {"model": "Independent Gaussian source-image translation at the particle-to-wave boundary",
        "assumed_translation_rms_per_axis_pm": numerics.source_image_rms_pm,
        "cross_spectral_density_multiplier": "exp(-2*pi^2*sigma_angstrom^2*|q-q_prime|^2)",
        "scope": "A local model assumption; not the Tip coherence length or a new particle source"}
    if sigma == 0:
        return modes, weights, record | {"quadrature_order_per_axis": 0, "discarded_trace": 0.}
    order = numerics.spatial_quadrature_order
    nodes, quadrature = hermgauss(order)
    nodes, quadrature = np.sqrt(2.)*sigma*nodes, quadrature/np.sqrt(np.pi)
    columns = len(weights)*order**2
    _admit_workspace(len(reciprocal), columns, numerics, retained_bytes=modes.nbytes)
    weighted = modes*cp.sqrt(cp.asarray(weights))[None, :]
    before = cp.sum(cp.abs(weighted)**2, axis=1)
    factor = cp.empty((len(reciprocal), columns), dtype=cp.complex128)
    frequencies = cp.asarray(reciprocal)
    for x in range(order):
        _check_cancelled(cancelled)
        for y in range(order):
            phase = cp.exp((-2j*np.pi)*(frequencies[:, 0]*nodes[x]+frequencies[:, 1]*nodes[y]))
            start = (x*order+y)*len(weights)
            factor[:, start:start+len(weights)] = weighted*(np.sqrt(quadrature[x]*quadrature[y])*phase[:, None])
    exact_diagonal = cp.sum(cp.abs(factor)**2, axis=1)
    diagonal_error = float(cp.sum(cp.abs(exact_diagonal-before)).item())/float(weights.sum())
    if diagonal_error > 5e-13:
        raise ValueError("Gaussian translation failed to preserve the angular density diagonal")
    remaining = source_fraction*numerics.maximum_mode_loss_fraction-(source_fraction-float(weights.sum()))
    if remaining <= 0:
        raise ValueError("No discarded-trace budget remains for spatial coherence compression")
    output, output_weights, compression = _compress_factor(cp, factor, float(weights.sum()), remaining,
                                                          numerics, cancelled)
    after = cp.sum(cp.abs(output)**2*cp.asarray(output_weights)[None, :], axis=1)
    angular_change = float(cp.sum(cp.abs(after-before)).item())/float(weights.sum())
    if angular_change > remaining/float(weights.sum())+5e-13:
        raise ValueError("Spatial-coherence compression exceeds its angular-diagonal error budget")
    return output, output_weights, record | {"quadrature_order_per_axis": order,
        "uncompressed_angular_diagonal_relative_l1_error": diagonal_error,
        "compressed_angular_diagonal_relative_l1_change": angular_change, **compression}


def _build_particle_wave_checkpoint(source, numerics, *, cancelled, progress_callback):
    """Assign and compress a near-specimen mixed state on verified CUDA.

    Energy-position correlations remain in the actual input trajectories. The
    stationary propagation wavelength is the current-weighted mean, admitted
    only for an actual energy span <=2 eV. Different original particles are
    incoherent; complex relative phases between emitted particles are unknown.
    """
    from temsim.physics.compute_backend import require_coherent_cuda
    from temsim.physics.wave_device import device_scope
    from temsim.physics.backend_execution import record_backend
    if type(source) is not ParticleWaveSource or type(numerics) is not MixedSourceNumerics:
        raise TypeError("Mixed-state construction requires its immutable particle source and numerical settings")
    numerics.validate()
    _check_cancelled(cancelled)
    require_coherent_cuda("Require GPU")
    selected = source.weight > 0
    phase = {name: getattr(source, name)[selected] for name in _PHASE_FIELDS}
    if source.canonical_frequency_shift_inv_angstrom is not None:
        phase[_CANONICAL_SHIFT] = source.canonical_frequency_shift_inv_angstrom[selected]
    probability = phase["weight"]
    energy = phase["kinetic_energy_ev"]
    mean_energy = source.mean_energy_ev
    if float(np.ptp(energy)) > 2.:
        raise ValueError("The single-mean-wavelength approximation exceeds its declared 2 eV energy span")
    rest_ev = m_e*c*c/e
    wavelength = h*c/e*1e10/np.sqrt(mean_energy*(mean_energy+2.*rest_ev))
    _report(progress_callback, 0, 3, "Assigning near-specimen Gaussian mixed state")
    centers, constant, kernel, phase_function, lift = lifted_centers(phase, wavelength, probability)
    origin = np.average(np.column_stack((phase["x_m"], phase["y_m"])), weights=probability, axis=0)
    if np.max(np.abs(np.column_stack((phase["x_m"], phase["y_m"]))-origin)) >= numerics.width_nm*.5e-9:
        raise ValueError("The wave field of view does not contain every positive-current incident particle")
    width_a = numerics.width_nm*10.
    radius = int(np.ceil(numerics.frequency_support_inv_angstrom*width_a))
    if radius >= numerics.pixels//2:
        raise ValueError("The requested reciprocal support reaches wave-grid Nyquist; increase pixels")
    gx, gy = np.meshgrid(np.arange(-radius, radius+1), np.arange(-radius, radius+1))
    keep = np.hypot(gx, gy)/width_a <= numerics.frequency_support_inv_angstrom
    gx, gy = gx[keep], gy[keep]
    reciprocal = np.column_stack((gx, gy))/width_a
    count = len(probability)
    modes_out = []
    with device_scope("Require GPU", maximum_working_bytes=numerics.maximum_working_bytes) as device:
        cp = device.xp
        workspace = _admit_workspace(len(reciprocal), count, numerics)
        factor = cp.empty((len(reciprocal), count), dtype=cp.complex128)
        frequencies, quadratic, angular_phase = cp.asarray(reciprocal), cp.asarray(kernel), cp.asarray(phase_function(reciprocal))
        for start in range(0, count, 128):
            _check_cancelled(cancelled)
            stop = min(count, start+128)
            positions, offsets = cp.asarray(centers[start:stop, :2]), cp.asarray(centers[start:stop, 2:])
            delta = frequencies[:, None, :]-offsets[None, :, :]
            exponent = -np.pi**2*cp.einsum("...i,ij,...j->...", delta, quadratic, delta)
            exponent += -2j*np.pi*(frequencies @ positions.T)+1j*angular_phase[:, None]
            factor[:, start:stop] = cp.exp(exponent)*cp.asarray(constant*np.sqrt(probability[start:stop]/width_a**2))
        base_budget = source.source_fraction*numerics.maximum_mode_loss_fraction*(.99 if numerics.source_image_rms_pm else 1.)
        modes, weights, compression = _compress_factor(cp, factor, source.source_fraction, base_budget,
                                                      numerics, cancelled)
        del factor, frequencies, quadratic, angular_phase
        _report(progress_callback, 1, 3, "Applying local mixed-state coherence parameters")
        modes, weights, coherence = _translation_channel(cp, modes, weights, reciprocal, numerics,
                                                         source.source_fraction, cancelled)
        eigenmode_weights = weights.copy()
        density_eigenvalues = np.asarray(coherence.get("retained_density_eigenvalues_fraction_of_emitted",
            compression["retained_density_eigenvalues_fraction_of_emitted"]))
        modes, weights, propagation_basis = _unitary_density_rebasis(cp, modes, weights, cancelled)
        if source.source_fraction-float(weights.sum()) > source.source_fraction*numerics.maximum_mode_loss_fraction:
            raise ValueError("Combined mixed-state compression exceeds its declared current-loss budget")
        origin_phase = cp.exp((2j*np.pi)*(cp.asarray(reciprocal) @ cp.asarray(origin*1e10)))
        iy, ix = cp.asarray(gy+numerics.pixels//2), cp.asarray(gx+numerics.pixels//2)
        boundary_probability = 0.
        unit_boundaries = []
        strip = max(1, numerics.pixels//32)
        spectrum = cp.zeros((numerics.pixels, numerics.pixels), dtype=cp.complex128)
        for index, weight in enumerate(weights):
            _check_cancelled(cancelled)
            spectrum.fill(0)
            spectrum[iy, ix] = modes[:, index]*origin_phase
            amplitude = cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(spectrum), norm="ortho"))
            norm = float(cp.sum(cp.abs(amplitude)**2).item())
            if abs(norm-1.) > 1e-10:
                raise ValueError("Cartesian mixed-state sampling changed the unit mode norm")
            edge = (cp.sum(cp.abs(amplitude[:strip])**2)+cp.sum(cp.abs(amplitude[-strip:])**2)
                    +cp.sum(cp.abs(amplitude[strip:-strip, :strip])**2)
                    +cp.sum(cp.abs(amplitude[strip:-strip, -strip:])**2))
            boundary_probability += float(weight)*float(edge.item())
            edge_two = (cp.sum(cp.abs(amplitude[:2])**2)+cp.sum(cp.abs(amplitude[-2:])**2)
                +cp.sum(cp.abs(amplitude[2:-2, :2])**2)+cp.sum(cp.abs(amplitude[2:-2, -2:])**2))
            unit_boundaries.append(float(edge_two.item())/norm)
            plane = PlaneWave(cp.asnumpy(amplitude), np.eye(2)*numerics.width_nm*1e-9/numerics.pixels, origin)
            modes_out.append(WaveMode(plane, float(weight), EMITTED_ELECTRON_REFERENCE,
                f"particle-mixed:{source.digest[:20]}:{index}", mean_energy/1000.))
        if boundary_probability > source.source_fraction*numerics.maximum_mode_loss_fraction:
            raise ValueError("The mixed-state wave reaches the numerical boundary; enlarge its field of view")
        if max(unit_boundaries) > 1e-9:
            raise ValueError("A mixed-state propagation component exceeds the existing 1e-9 boundary fraction; enlarge its physical domain at unchanged sampling pitch")
        evidence = device.evidence()
    _check_cancelled(cancelled)
    _report(progress_callback, 3, 3, "Near-specimen mixed-state checkpoint ready")
    record = {"schema": "particle-wave-source-v1", "particle_source_digest": source.digest,
        "input_digest": source.input_digest, "original_positive_count": source.original_positive_count,
        "handoff_particle_count": int(np.count_nonzero(selected)), "handoff_z_mm": source.z_mm,
        "source_fraction": source.source_fraction, "retained_source_fraction": float(weights.sum()),
        "discarded_source_fraction": max(0., source.source_fraction-float(weights.sum())),
        "numerics": asdict(numerics), "lift": lift, "compression": compression, "coherence": coherence,
        "propagation_basis": propagation_basis,
        "retained_density_eigenvalues_fraction_of_emitted": density_eigenvalues.tolist(),
        "retained_eigenmode_norm_weights_fraction_of_emitted": eigenmode_weights.tolist(),
        "density_eigenvalue_statistics": {
            "scope": "Conditional retained compressed density; not statistics of the nonorthogonal propagation component weights",
            "purity": float(np.sum((density_eigenvalues/density_eigenvalues.sum())**2)),
            "coherent_fraction": float(density_eigenvalues.max()/density_eigenvalues.sum()),
            "effective_modes": float(density_eigenvalues.sum()**2/np.sum(density_eigenvalues**2))},
        "canonical_frequency_initialization": {
            "executed": source.canonical_frequency_shift_inv_angstrom is not None,
            "frequency_units": "cycles/angstrom", "shift_formula": "electron q*A_xy/h, q=-e; SI inverse metres multiplied by 1e-10",
            "mechanical_trajectory_slopes_retained": True,
            "maximum_absolute_shift_inv_angstrom": (float(np.max(abs(source.canonical_frequency_shift_inv_angstrom)))
                if source.canonical_frequency_shift_inv_angstrom is not None else 0.),
            "scope": ("Canonical frequencies initialized in the captured vector-potential gauge"
                if source.canonical_frequency_shift_inv_angstrom is not None else
                "No vector-potential shift supplied; explicit free-space boundary assignment")},
        "wave_energy_approximation": {"model": "One current-weighted mean stationary wavelength",
            "mean_energy_ev": mean_energy, "minimum_energy_ev": float(energy.min()),
            "maximum_energy_ev": float(energy.max()), "maximum_allowed_energy_span_ev": 2.,
            "wavelength_angstrom": float(wavelength), "original_particle_energies_retained": True,
            "scope": "Chromatic particle trajectories retained; no temporal interference or energy-resolved wavepacket"},
        "flight_time_s_range": [float(phase["flight_time_s"].min()), float(phase["flight_time_s"].max())],
        "boundary_probability_fraction_of_emitted": boundary_probability,
        "propagation_unit_mode_outer_2cell_fractions": unit_boundaries,
        "maximum_unit_mode_outer_2cell_fraction": max(unit_boundaries),
        "unit_mode_boundary_fraction_limit": 1e-9,
        "estimated_workspace_bytes": workspace, "backend": evidence, "upstream": source.provenance,
        "particle_relative_phases": "Unknown; original particles are mutually incoherent in this declared lift",
        "source_current_reference": "Original emitted electrons; upstream aperture losses are retained",
        "sample_feedback_on_external_fields": False}
    record_backend("particle_wave_source", "Require GPU", "CUDA GPU", **evidence)
    auxiliary = {name: getattr(source, name) for name in _PHASE_FIELDS}
    auxiliary["retained_density_eigenvalues"] = density_eigenvalues
    if source.canonical_frequency_shift_inv_angstrom is not None:
        auxiliary[_CANONICAL_SHIFT] = source.canonical_frequency_shift_inv_angstrom
    return WaveCheckpoint(BeamState(tuple(modes_out), EMITTED_ELECTRON_REFERENCE), source.z_mm,
        source.reference_current_a, record, auxiliary)


def build_particle_wave_checkpoint(source, numerics=MixedSourceNumerics(), *,
                                  cancelled=lambda: False, progress_callback=None):
    """Build one immutable local mixed state with CUDA and bounded resources."""
    from temsim.physics.wave_device import release_device_memory
    try:
        return _build_particle_wave_checkpoint(source, numerics, cancelled=cancelled,
                                              progress_callback=progress_callback)
    finally:
        release_device_memory()
