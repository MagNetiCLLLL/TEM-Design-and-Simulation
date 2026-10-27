"""Bounded scalar positive-branch electromagnetic carrier propagation.

The spinless forward operator retains positive-branch kinetic powers through
explicit order 2, 4, 6 or 8: H=sum(c_n D^n/p^(2n-1))-q*Az,
D=Pi_perp^2+p_axis^2-p_local^2, q=-e. A quadratic reference is advanced by the existing metaplectic
implementation, including its affine scalar action. The remainder acts on the
complex envelope with Pi/p=D/k+grad(S)/p-q*A/p. The quartic term is a fourth
order differential operator, never a position-only aberration screen.

Each retained power is the complete differential operator, including mixed
derivatives. The residual is a Hermitian finite-period spectral discretisation. Its bounded
exponential is not a proof of physical-domain, potential quadrature, spatial or
axial convergence. Electrostatic energy laws and explicitly retarded native
scan slices carry separate energy, clock and approximation receipts; arbitrary
FEM magnetic or time-domain pulse propagation is not inferred. Affine lattice covectors use the complete inverse-transpose metric;
large carriers are not sampled or discarded.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e
from scipy.linalg import expm

from temsim.cache_memory import retained_memory_inventory
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave, reselect_phase_carrier
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.potential_kinetic_expansion import (
    validate_kinetic_order, kinetic_coefficient, kinetic_remainder_phase_bound,
    differential_power_norm_bound,
)
from temsim.physics.wave_grid import (WaveGridNumerics, WaveSamplingError, WaveGridBudgetError,
    WaveResourceProposal, WaveResourceTransaction, apply_resolved_operator)
from temsim.physics.wave_magnetic_residual import _series_plan
from temsim.physics.wave_device import array_module, normalise_backend
from temsim.physics.immutable_wave_checks import ImmutableWaveCheckCache


_J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
_PROVIDERS = (AnalyticMagneticPotential, AnalyticPotentialDomain, HighOrderRoundLensPotential)
_IMMUTABLE_WAVE_CHECKS = ImmutableWaveCheckCache()


class OccupiedCarrierBoundaryError(ValueError):
    """The represented edge is occupied; this is not a field-domain error."""


class ScalarPhaseSamplingError(WaveSamplingError):
    """The full represented domain undersamples an added scalar phase."""


def _scalar_phase_resource_pressure(error):
    """Recognize a scalar-phase cause without parsing diagnostic text."""
    if not isinstance(error, WaveGridBudgetError):
        return False
    seen, pending = set(), [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ScalarPhaseSamplingError):
            return True
        pending.extend((current.__cause__, current.__context__))
    return False


def _positive(value, name, *, zero=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


def _admit(snapshot, *, electric_reference=None):
    from temsim.physics.retarded_wave import RetardedPotentialSlice
    from temsim.physics.dynamic_potential import DynamicDipolePotential
    retarded = type(snapshot) is RetardedPotentialSlice
    if type(snapshot) is not ElectromagneticPotentialSnapshot and not retarded:
        raise ValueError("Carrier propagation requires an immutable potential snapshot")
    for row in snapshot.contributions:
        if row.kind == "electric" and electric_reference is not None:
            from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, electric_rows
            if type(electric_reference) not in (ElectrostaticWaveReference, RetardedPotentialSlice):
                raise ValueError("Accelerating carrier propagation requires its immutable energy reference")
            electric_reference.require(snapshot)
            electric_rows(snapshot.source_snapshot if retarded else snapshot)
            continue
        if retarded and type(row) is DynamicDipolePotential:
            continue  # Explicit time law and its error/clock history are retained.
        if row.kind != "magnetic" or type(row.potential) not in _PROVIDERS:
            raise ValueError("Carrier benchmark requires native analytical static magnetic jets; FEM and electric sources are not admitted")
        provider = row.potential
        meta = provider.to_metadata()
        native = getattr(getattr(provider, "base", provider), "field", None)
        if (meta.get("static") is False or meta.get("dynamic_drive_frozen", False)
                or meta.get("provenance", {}).get("dynamic_drive_frozen", False)
                or bool(getattr(native, "dynamic", False))):
            raise ValueError("Frozen dynamic drives are not qualified static carrier fields")


def _check_boundary(amplitude, boundary_limit):
    return _IMMUTABLE_WAVE_CHECKS.check(amplitude, boundary_limit, _check_boundary_uncached)


def _check_boundary_uncached(amplitude, boundary_limit):
    probability = abs(amplitude)**2
    total = float(probability.sum())
    boundary = math.fsum(float(part.sum()) for part in (probability[:2], probability[-2:],
        probability[2:-2, :2], probability[2:-2, -2:]))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Carrier wave requires finite positive flux")
    if boundary > total*boundary_limit:
        raise OccupiedCarrierBoundaryError("Occupied carrier envelope boundary: enlarge the represented physical domain")
    return boundary


def _check_envelope_sampling(amplitude, tail_limit):
    return _IMMUTABLE_WAVE_CHECKS.check(amplitude, tail_limit, _check_envelope_sampling_uncached)


def _check_envelope_sampling_uncached(amplitude, tail_limit):
    """Keep the full-band gate and suggest a bounded next sampling candidate.

    The cumulative marginal support is read from this same complete FFT only
    after a failure. It removes no coefficients and does not certify unknown
    frequencies beyond the current lattice. The caller still refines the
    original input and reruns every operator/admission gate on that candidate.
    """
    xp = array_module(amplitude)
    spectrum = abs(xp.fft.fft2(amplitude))**2
    total = float(spectrum.sum())
    fractions = [float(xp.sum(spectrum.take(xp.flatnonzero(abs(xp.fft.fftfreq(amplitude.shape[axis])) >= .4),
                                               axis=axis)))/total for axis in (0, 1)]
    if max(fractions) > tail_limit:
        support = 0.
        for axis in (0, 1):
            frequency = abs(xp.fft.fftfreq(amplitude.shape[axis]))
            marginal = spectrum.sum(axis=1-axis)
            order = xp.argsort(frequency)
            tail = xp.cumsum(marginal[order][::-1])[::-1]
            occupied = frequency[order][tail > total*tail_limit]
            if occupied.size:
                support = max(support, float(xp.max(occupied)))
        raise WaveSamplingError("Carrier envelope spectrum reaches the Nyquist guard; refine the physical lattice without filtering the wave",
                                required_scale=max(1., support/.4))
    return max(fractions)


def _residual_grid_resource_checks(numerics, requested):
    """Plan candidate grids against the declared residual device partition.

    This is a numeric preflight, not a CUDA probe or permission to execute.
    Device admission still checks availability and live free memory. Required
    GPU execution has a hard limit; optional acceleration only prefers a
    smaller fully rechecked candidate and retains the original CPU fallback.
    Host field/canonical allocations keep their separate host memory limit.
    """
    numerics.validate()
    requested = normalise_backend(requested)
    from temsim.physics.potential_wave_device import RESIDUAL_DEVICE_BYTES_PER_CELL
    limit = numerics.maximum_device_working_bytes
    def fits(shape):
        return RESIDUAL_DEVICE_BYTES_PER_CELL*math.prod(shape) <= limit
    def require(shape):
        if not fits(shape):
            raise WaveGridBudgetError("Required residual device budget exceeded: "
                f"grid={tuple(shape)}, estimated device working bytes={RESIDUAL_DEVICE_BYTES_PER_CELL*math.prod(shape)}; "
                f"maximum_device_working_bytes={limit}. The optical operator was not applied.")
    if requested == "Require GPU":
        return require, None
    if numerics.acceleration_enabled and requested not in ("CPU", "Numba CPU"):
        return None, fits
    return None, None


def _pressure_resampling_constraints(wave, error, *, numerics, requested,
                                     retained_bytes, stage, pressure=None):
    """Describe mandatory resources without allocating or probing a device.

    Reserve at least one full finite-DFT spectrum's growth into the ordinary
    80-percent Nyquist guard (1 / 0.8), plus the usual refinement margin. A
    larger measured failure scale is also retained. The actual physical
    operator and sampling gates run again; one-step headroom is not a
    certificate that all subsequent propagation fits the smaller lattice.
    """
    if not isinstance(error, WaveGridBudgetError):
        raise ValueError("Pressure resampling requires a typed resource failure")
    from temsim.physics.wave_grid import CanonicalChartResourceError, _refinement_shape
    from temsim.physics.potential_wave_device import residual_host_working_bytes
    requested = normalise_backend(requested)
    required, _ = _residual_grid_resource_checks(numerics, requested)
    scale, seen, pending = 1.25, set(), [error]
    preferred_failure_shape = None
    while pending:
        failure = pending.pop()
        if failure is None or id(failure) in seen:
            continue
        seen.add(id(failure))
        if type(failure) is CanonicalChartResourceError:
            hint = failure.canonical_chart_resource_hint
            scale = max(scale, *(required/source for required, source in
                zip(hint.required_shape, hint.input_shape)))
            preferred_failure_shape = hint.required_shape
            # These are alternative complete representations of one operator.
            # Collins sampling remains in the exception chain for diagnosis;
            # it is not a requirement of the preferred angular-spectrum chart.
            continue
        if isinstance(failure, WaveSamplingError):
            if not math.isfinite(failure.required_scale):
                raise ValueError("Pressure resampling cannot resolve unbounded sampling")
            scale = max(scale, float(failure.required_scale))
        pending.extend((failure.__cause__, failure.__context__))
    if pressure is not None:
        scale = max(scale, pressure.growth_scale)
    original_bytes = wave.amplitude.nbytes

    def admissible(shape):
        if (pressure is not None and pressure.previous_shape is not None
                and (math.prod(shape) >= math.prod(pressure.previous_shape)
                     or any(n > old for n, old in zip(shape, pressure.previous_shape)))):
            return False
        forecast = _refinement_shape(shape, scale, minimum_growth=1.)
        try:
            for selected in (shape, forecast):
                # The unchanged operator entrance and its caller-owned
                # buffers stay alive while a candidate is prepared/retried.
                numerics.check_working_memory(selected,
                    working_bytes=residual_host_working_bytes(selected),
                    retained_bytes=retained_bytes+original_bytes)
                if required is not None:
                    required(selected)
        except WaveGridBudgetError:
            return False
        return True

    context = {"schema": "wave-resource-pressure-v1", "stage": stage,
        "failure_kind": ("WaveGridBudgetError" if type(error) is CanonicalChartResourceError
                         else type(error).__name__), "input_shape": list(wave.amplitude.shape),
        "maximum_host_working_bytes": numerics.maximum_working_bytes,
        "maximum_device_working_bytes": numerics.maximum_device_working_bytes,
        "maximum_pixels": numerics.maximum_pixels,
        "retained_host_bytes": retained_bytes,
        "operator_memory_policy": "carrier-residual-host-512-plus-64MiB/device-448",
        "requested_backend": requested, "growth_guard_scale": scale,
        "candidate_family": "fixed_aspect_even_fft_friendly_v1"}
    if preferred_failure_shape is not None:
        context["source_failure_shape"] = list(preferred_failure_shape)
    return admissible, context


def _validate_wave_lattice(wave):
    if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
        raise ValueError("Carrier benchmark requires a CPU PlaneWave")
    if min(wave.amplitude.shape) < 8:
        raise ValueError("Carrier benchmark requires at least eight samples per axis")
    basis = wave.basis_m
    if np.linalg.det(basis) <= 0 or np.linalg.cond(basis) > 1e8:
        raise ValueError("Carrier benchmark requires a positive-orientation, numerically resolved affine lattice")


def _validate_wave(wave, numerics, boundary_limit, envelope_limit):
    _validate_wave_lattice(wave)
    # Validation owns boundary and FFT diagnostics, not field coefficients
    # or a subsequent Hermitian exponential. Those stages preflight their
    # own distinct workspace before execution.
    numerics.check_working_memory(wave.amplitude.shape, working_bytes=256*wave.amplitude.size, retained_bytes=0)
    boundary = _check_boundary(wave.amplitude, boundary_limit)
    _check_envelope_sampling(wave.amplitude, envelope_limit)
    return boundary


def _additional_wave_bytes(candidate, *retained_amplitudes):
    """Unique outside owners, excluding the active wave's existing storage."""
    active = retained_memory_inventory(candidate.amplitude)
    retained = retained_memory_inventory(*retained_amplitudes)
    return sum(size for identity, size in retained.items() if identity not in active)


def _points(wave, z):
    xy = wave.coordinates_m()
    return np.stack((xy[0], xy[1], np.full(wave.amplitude.shape, z)), axis=-1)


def _corners(wave, z):
    """Four exact corners of the represented affine grid, not its pixel cells."""
    ny, nx = wave.amplitude.shape
    indices = np.array([(x, y) for x in (-(nx//2), nx-1-nx//2)
                        for y in (-(ny//2), ny-1-ny//2)])
    xy = wave.origin_m+indices@wave.basis_m.T
    return np.column_stack((xy, np.full(4, z)))


def _boundary_side(provider, point, interval):
    """Declared one-sided compact Az value at an exactly aligned face only."""
    from temsim.physics.dynamic_potential import DynamicDipolePotential
    if type(provider) is DynamicDipolePotential:
        provider = provider.base
    if interval is None or type(provider) is not AnalyticPotentialDomain:
        return None
    native = provider.base.field
    if type(native) is not FrozenPosedMultipole or np.any(native.registration.rotation_array[:2, 2] != 0.):
        return None
    edges = np.asarray(native.native_support_mm)*.001
    local = native.registration.positions_global_to_local_m(np.asarray(point))
    if local[2] not in edges:
        return None
    faces = np.sort(native.registration.origin_array_m[2]+native.registration.rotation_array[2, 2]*edges)
    lower, upper = interval
    if upper <= faces[0] or lower >= faces[1]:
        return "outside"
    if lower >= faces[0] and upper <= faces[1]:
        return "inside"
    raise ValueError("One-sided compact potential sampling requires an interval on one known support side")


def _jet(snapshot, point, interval=None):
    a, jac, hess = np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3, 3))
    for row in snapshot.contributions:
        if row.kind == "electric":
            continue  # Scalar potential enters the same squared kinetic operator.
        side = _boundary_side(row.potential, point, interval)
        if side == "outside":
            continue
        from temsim.physics.dynamic_potential import DynamicDipolePotential
        if type(row.potential) is DynamicDipolePotential:
            result = row.potential.sample_jet(point, snapshot.absolute_time(np.asarray(point)[..., 2]),
                                             axial_side=side)
        elif side == "inside":
            # This is the analytic interior limit of the DECLARED compact Az
            # model, not a two-sided derivative or an extrapolated FEM field.
            from temsim.physics.posed_lens_wave import vector_potential_jet
            row.potential.sample(point)  # Preserve the exact radial-domain gate.
            values = vector_potential_jet((row.potential.base.field,), np.asarray(point))
            result = dict(zip(("A_t_m", "jac_A_t", "hess_A_t_per_m"), values))
        else:
            result = row.potential.sample_jet(point)
        a += result["A_t_m"]
        jac += result["jac_A_t"]
        hess += result["hess_A_t_per_m"]
    a += snapshot.gauge.vector_potential_shift_t_m(point)
    jac += snapshot.gauge.hessian_t
    if not all(np.isfinite(item).all() for item in (a, jac, hess)):
        raise ValueError("Carrier reference potential jet must be finite")
    return a, jac, hess


def _coverage(snapshot, wave, lower, upper):
    """Check full planes and reject an unresolved hard face inside a step."""
    # Every admitted native radial domain is convex on a smooth side of its
    # support faces. Affine-grid corners give its exact inclusion test. The
    # separate face test below rejects any tilted/interior discontinuity, so
    # a compact slab cannot be skipped just because both end planes are zero.
    # Actual potential coefficients are still sampled on every wave cell.
    end_points = np.concatenate((_corners(wave, lower), _corners(wave, upper)))
    _coverage_points(snapshot, end_points, lower, upper)


def _coverage_points(snapshot, end_points, lower, upper):
    snapshot.sample(end_points)
    for row in snapshot.contributions:
        provider = row.potential
        if type(provider) is HighOrderRoundLensPotential and provider.remote_domain != "strict":
            # The native cylinder plus remote roundoff-qualified Gaussian
            # tails is not convex. Admitted endpoints alone cannot certify a
            # swept box that crosses an unqualified near-lens radial region.
            provider.require_global_box(end_points)
        from temsim.physics.dynamic_potential import DynamicDipolePotential
        if type(provider) is DynamicDipolePotential:
            provider = provider.base
        native = getattr(getattr(provider, "base", provider), "field", None)
        if type(native) is FrozenPosedMultipole:
            rotation = native.registration.rotation_array
            if type(provider) is AnalyticPotentialDomain and np.all(rotation[:2, 2] == 0.):
                # Az is parallel to the exact aligned face normal. The
                # existing compact model has no curl(A) delta sheet there.
                # Midpoint jets lie on a smooth side; its inside endpoint
                # sample is a measure-zero convention, not an extra kick.
                edges = [native.registration.origin_array_m[2]+rotation[2, 2]*edge*.001
                         for edge in native.native_support_mm]
                if any(lower < edge < upper for edge in edges):
                    raise ValueError("Carrier step crosses a hard axial support face; place the face at an axial node")
                continue
            local_z = native.registration.positions_global_to_local_m(end_points)[:, 2]
            start, stop = local_z.min(), local_z.max()
            if any(start <= edge*.001 <= stop for edge in native.native_support_mm):
                raise ValueError("Carrier step touches or crosses a hard axial support face; an interface-aware operator is required")


@dataclass(frozen=True)
class _Reference:
    a: np.ndarray
    jac: np.ndarray
    hess: np.ndarray
    center_xy: np.ndarray
    center_u: np.ndarray
    alpha: float
    path: CanonicalPath
    generator: np.ndarray
    center_force: np.ndarray
    linear_force: np.ndarray
    constant_h: float
    sweep_segments: tuple = ()
    axial_interval: tuple | None = None
    kinetic_scale: float = 1.
    electric_reference: object = None
    chart_energy_ev: float | None = None


def _reference(snapshot, wave, z, distance, momentum, quadrature="midpoint", axial_interval=None,
               electric_reference=None, chart_energy_ev=None):
    """Quadratic Taylor part of H_2/p, including all sources before squaring."""
    from temsim.physics.column_wave import _linear_factor
    center_xy = wave.origin_m
    center_u = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    a, jac, hess = _jet(snapshot, np.r_[center_xy, z], axial_interval)
    alpha = -e/momentum
    beta = 1.
    if electric_reference is not None:
        from temsim.physics.electric_wave_reference import momentum_velocity
        axial_momentum, _ = momentum_velocity(electric_reference.axis_energy(snapshot, z))
        beta = float(momentum/axial_momentum)
    transverse, derivative = a[:2], jac[:2, :2]
    v0 = beta*.5*alpha**2*(transverse@transverse)-alpha*a[2]
    gv = beta*alpha**2*derivative.T@transverse-alpha*jac[2, :2]
    hv = beta*alpha**2*(derivative.T@derivative+np.einsum("i,ijk->jk", transverse, hess[:2, :2, :2]))-alpha*hess[2, :2, :2]
    matrix = np.zeros((4, 4))
    matrix[:2, :2] = hv-beta*alpha*np.einsum("i,ijk->jk", center_u, hess[:2, :2, :2])
    matrix[:2, 2:] = -beta*alpha*derivative.T
    matrix[2:, :2] = matrix[:2, 2:].T
    matrix[2:, 2:] = beta*np.eye(2)
    point = np.r_[center_xy, center_u]
    gradient = np.r_[gv-beta*alpha*derivative.T@center_u, beta*(center_u-alpha*transverse)]
    value = beta*(.5*(center_u@center_u)-alpha*(transverse@center_u))+v0
    linear = gradient-matrix@point
    constant = value-gradient@point+.5*point@matrix@point
    path = CanonicalPath(1e-3)
    _linear_factor(path, _J@matrix, distance, force=_J@linear, constant_h=float(constant))
    reference = _Reference(a, jac, hess, center_xy.copy(), center_u.copy(), alpha, path,
                          _J@matrix, _J@gradient, _J@linear, float(constant), axial_interval=axial_interval,
                          kinetic_scale=beta, electric_reference=electric_reference, chart_energy_ev=chart_energy_ev)
    if quadrature == "cf4":
        # Fourth-order commutator-free integration of this QUADRATIC part.
        # Alvermann and Fehske (2011), doi:10.1016/j.jcp.2011.04.006;
        # the two Gauss nodes and (3 +/- 2 sqrt(3))/12 coefficients retain
        # affine forces and the scalar central action in the same Lie algebra.
        # The nonquadratic midpoint residual remains generally SECOND order.
        lower, upper = (z-distance/2., z+distance/2.) if axial_interval is None else axial_interval
        first_inside, last_inside = np.nextafter(lower, upper), np.nextafter(upper, lower)
        # A two-ULP event interval has one representable interior point.
        # Keep the exact endpoints and distance; round BOTH Gaussian nodes
        # into that open interval instead of evaluating a hard-face value.
        early_z, late_z = z-distance*np.sqrt(3)/6., z+distance*np.sqrt(3)/6.
        if first_inside <= last_inside:
            early_z = np.clip(early_z, first_inside, last_inside)
            late_z = np.clip(late_z, first_inside, last_inside)
        # With no representable interior point, the explicit interval-side
        # contract above supplies only the known compact model's limit.
        early = _reference(snapshot, wave, early_z, 0., momentum, axial_interval=axial_interval,
                           electric_reference=electric_reference, chart_energy_ev=chart_energy_ev)
        late = _reference(snapshot, wave, late_z, 0., momentum, axial_interval=axial_interval,
                          electric_reference=electric_reference, chart_energy_ev=chart_energy_ev)
        low, high = (3-2*np.sqrt(3))/12., (3+2*np.sqrt(3))/12.
        path, segments = CanonicalPath(1e-3), []
        for first, second in ((high, low), (low, high)):
            generator = first*early.generator+second*late.generator
            force = first*early.linear_force+second*late.linear_force
            constant = first*early.constant_h+second*late.constant_h
            _linear_factor(path, generator, distance, force=force, constant_h=constant)
            segments.append((generator, first*early.center_force+second*late.center_force, distance))
        reference = replace(reference, path=path, sweep_segments=tuple(segments))
    return reference


def _swept_coverage(snapshot, wave, lower, upper, reference, wavelength):
    """Conservative represented phase-space box throughout the reference step.

    For y'=G*y+f, |y(s)| <= exp(|G|s)|y(0)| + integral exp(|G|t)|f|dt.
    The nonnegative augmented exponential bounds every intermediate time by
    its final majorant. This is a lattice-ray support admission check, separate
    from metaplectic wave boundary and complex grid-convergence checks.
    """
    xy = _corners(wave, lower)[:, :2].T
    delta = xy-reference.center_xy[:, None]
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    carrier = tilt[:, None]+curvature@(xy-wave.origin_m[:, None])-reference.center_u[:, None]
    inverse = np.linalg.inv(wave.basis_m).T
    frequency_bound = .5*wavelength*np.sum(abs(inverse), axis=1)
    bound = np.r_[np.max(abs(delta), axis=1), np.max(abs(carrier), axis=1)+frequency_bound, 1.]
    swept = bound
    segments = reference.sweep_segments or ((reference.generator, reference.center_force, upper-lower),)
    for generator, center_force, distance in segments:
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = abs(generator)
        augmented[:4, 4] = abs(center_force)
        with np.errstate(over="ignore", invalid="ignore"):
            swept = expm(augmented*distance)@swept
    if not np.isfinite(swept).all():
        raise ValueError("Canonical swept-domain bound is unbounded; refine axial steps")
    from itertools import product
    points = np.array([(reference.center_xy[0]+sx*swept[0],
                        reference.center_xy[1]+sy*swept[1], z)
                      for sx, sy, z in product((-1, 1), (-1, 1), (lower, upper))])
    try:
        _coverage_points(snapshot, points, lower, upper)
    except ValueError as error:
        raise ValueError(f"Canonical swept domain is not admitted; refine axial steps or enlarge the qualified field domain: {error}") from error
    return swept[:2].tolist()


def _coefficients(snapshot, wave, z, reference, wavelength, sampler=None, *, return_energy_terms=False):
    """Stable remainder of the quadratic kinetic Hamiltonian, plus full Pi."""
    sampled = snapshot.sample(_points(wave, z)) if sampler is None else sampler.sample(_points(wave, z))
    # These exact retained providers have already sampled and validated the
    # scalar potential on this same plane. Reusing that value avoids a second
    # full electric-field interpolation, without caching across waves or Z.
    # Unrecognized adapters retain the independent original energy sampler.
    from temsim.physics.electric_wave_reference import ElectrostaticWaveReference
    from temsim.physics.retarded_wave import RetardedPotentialSlice
    from temsim.physics.unified_potential_sampler import UnifiedPotentialSampler
    reuse_phi = (
        type(snapshot) is ElectromagneticPotentialSnapshot
        and type(reference.electric_reference) is ElectrostaticWaveReference
        and (sampler is None or (type(sampler) is UnifiedPotentialSampler
             and sampler.snapshot_digest == snapshot.numerical_digest))
    ) or (type(snapshot) is RetardedPotentialSlice
          and reference.electric_reference is snapshot and sampler is None)
    sampled_phi = sampled["phi_v"] if reuse_phi else None
    if reference.axial_interval is not None:
        point = np.r_[wave.origin_m, z]
        excluded = [row for row in snapshot.contributions
                    if _boundary_side(row.potential, point, reference.axial_interval) == "outside"]
        if excluded and not hasattr(snapshot, "source_snapshot"):
            # At the measure-zero face, ordinary sampling uses the native
            # inside value. Remove precisely those known-zero exterior terms
            # BEFORE forming total-A products for this outside subinterval.
            sampled = dict(sampled)
            points = _points(wave, z)
            total_a = np.zeros_like(sampled["A_t_m"])
            excluded_ids = {row.source_id for row in excluded}
            for row in snapshot.contributions:
                if row.kind == "magnetic" and row.source_id not in excluded_ids:
                    total_a += row.potential.sample(points)["A_t_m"]
            total_a += snapshot.gauge.vector_potential_shift_t_m(points)
            sampled["A_t_m"] = total_a
            del points, total_a
    a = np.moveaxis(sampled["A_t_m"], -1, 0)
    # Only A enters this magnetic remainder. Retain at most the single scalar
    # potential array needed by the electric energy law, not unused E/B grids.
    del sampled
    coordinates = wave.coordinates_m()
    delta = coordinates-reference.center_xy[:, None, None]
    linear_a = np.einsum("ij,jyx->iyx", reference.jac[:, :2], delta)
    quadratic_a = .5*np.einsum("iyx,kij,jyx->kyx", delta, reference.hess[:, :2, :2], delta)
    remainder_a = a-reference.a[:, None, None]-linear_a
    alpha = reference.alpha
    beta = reference.kinetic_scale
    velocity = -beta*alpha*remainder_a[:2]
    scalar = beta*.5*alpha**2*(
        2*np.einsum("i,iyx->yx", reference.a[:2], (remainder_a-quadratic_a)[:2])
        +2*np.sum(linear_a[:2]*remainder_a[:2], axis=0)+np.sum(remainder_a[:2]**2, axis=0))
    scalar -= alpha*(remainder_a[2]-quadratic_a[2])
    scalar += beta*alpha*np.einsum("i,iyx->yx", reference.center_u, quadratic_a[:2])
    del delta, linear_a, quadratic_a, remainder_a
    defect = 0.
    if reference.electric_reference is not None:
        if reuse_phi:
            defect = reference.electric_reference._squared_momentum_defect_from_phi(
                snapshot, sampled_phi, z, reference.chart_energy_ev)
        else:
            defect = reference.electric_reference.squared_momentum_defect(
                snapshot, _points(wave, z), z, reference.chart_energy_ev)
        scalar += .5*beta*defect
    del sampled_phi
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    carrier = tilt[:, None, None]+np.einsum("ij,jyx->iyx", curvature,
        coordinates-wave.origin_m[:, None, None])
    scalar += np.sum(velocity*carrier, axis=0)
    mechanical = carrier-alpha*a[:2]
    del carrier, coordinates, a
    ny, nx = wave.amplitude.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing="ij")
    frequency = wavelength*np.einsum("ij,jyx->iyx", np.linalg.inv(wave.basis_m).T, np.stack((fx, fy)))
    if any(not np.isfinite(value).all() for value in (velocity, scalar, mechanical, frequency)):
        raise ValueError("Carrier residual coefficients must be finite")
    values = (velocity, scalar, mechanical, frequency)
    return (*values, defect, beta) if return_energy_terms else values


def _apply_residual(amplitude, velocity, scalar, mechanical, frequency, kinetic_order,
                    momentum_defect=0., kinetic_scale=1., *, quartic_scalar_splitting=False,
                    higher_order_scalar_splitting=False):
    xp = array_module(amplitude)
    spectrum = xp.fft.fft2(amplitude)
    derivatives = tuple(xp.fft.ifft2(symbol*spectrum) for symbol in frequency)
    divergence = sum(frequency[axis]*xp.fft.fft2(velocity[axis]*amplitude)
                     for axis in range(2))
    result = scalar*amplitude+.5*(xp.fft.ifft2(divergence)
        +sum(velocity[axis]*derivatives[axis] for axis in range(2)))
    # The velocity anticommutator is complete. Keeping its full complex
    # divergence through every high-order kinetic power wastes one lattice.
    del divergence
    if kinetic_order >= 4:
        squared_frequency = xp.sum(frequency**2, axis=0)
        multiplication = xp.sum(mechanical**2, axis=0)+momentum_defect

        def kinetic(value, transformed=None, differentiated=None, *, include_multiplication=True):
            if transformed is None:
                transformed = xp.fft.fft2(value)
            if differentiated is None:
                differentiated = tuple(xp.fft.ifft2(symbol*transformed) for symbol in frequency)
            # Expand (D_i + m_i)^2 without differentiating m_i analytically.
            # Fourier differentiation of each product preserves the finite-grid
            # anticommutator, including variable-coefficient aliasing semantics.
            transformed_total = squared_frequency*transformed
            for axis in range(2):
                transformed_total += frequency[axis]*xp.fft.fft2(mechanical[axis]*value)
            if not include_multiplication:
                return (xp.fft.ifft2(transformed_total)
                        +sum(mechanical[axis]*differentiated[axis] for axis in range(2)))
            # Keep the original arithmetic grouping on the default path.
            return (xp.fft.ifft2(transformed_total)+multiplication*value
                    +sum(mechanical[axis]*differentiated[axis] for axis in range(2)))

        if kinetic_order in (6, 8):
            if higher_order_scalar_splitting:
                # R_n = K^n - U^n without subtracting large powers. The
                # recurrence K R_n + L U^n retains every noncommuting word.
                remainder = kinetic(amplitude, spectrum, derivatives, include_multiplication=False)
                # The first kinetic action consumed these original-input
                # transforms. All subsequent powers transform their own
                # input; releasing three complex lattices changes no maths.
                del spectrum, derivatives
                scalar_power = amplitude
                for power in range(2, kinetic_order//2+1):
                    scalar_power = multiplication*scalar_power
                    # K R + L U^n = L(R + U^n) + U R, with K=L+U.
                    # The same discrete L retains every noncommuting term;
                    # linearity saves one kinetic action (six FFTs) per power.
                    # This changes floating-point grouping, not the operator,
                    # and never subtracts the large full K^n and U^n powers.
                    remainder = (kinetic(remainder+scalar_power, include_multiplication=False)
                        +multiplication*remainder)
                    result += kinetic_coefficient(power)*kinetic_scale**(2*power-1)*remainder
            else:
                kinetic_power = kinetic(amplitude, spectrum, derivatives)
                del spectrum, derivatives
                for power in range(2, kinetic_order//2+1):
                    kinetic_power = kinetic(kinetic_power)
                    result += kinetic_coefficient(power)*kinetic_scale**(2*power-1)*kinetic_power
            return result
        if quartic_scalar_splitting:
            # K = L + U, with real U = m^2 + defect. U^2 is executed as an
            # exact scalar phase outside this operator. Apply L^2 + LU + UL
            # directly: subtracting U^2*psi from K(K*psi) would lose precision.
            linear = kinetic(amplitude, spectrum, derivatives, include_multiplication=False)
            del spectrum, derivatives
            fourth = (kinetic(linear, include_multiplication=False)
                +kinetic(multiplication*amplitude, include_multiplication=False)
                +multiplication*linear)
        else:
            # Applying the complete operator twice retains cross-axis and
            # electric defect terms in Pi^4, not separate fourth derivatives.
            second = kinetic(amplitude, spectrum, derivatives)
            del spectrum, derivatives
            fourth = kinetic(second)
        result += kinetic_scale**3*fourth/8
    return result


def _residual(snapshot, wave, z, distance, reference, wavelength, order, error, remainder_budget,
              boundary_limit, envelope_limit, cancelled, scalar_splitting=False, sampler=None,
              quartic_scalar_splitting=False, higher_order_scalar_splitting=False,
              residual_compute_backend="CPU", maximum_device_working_bytes=24*1024**3,
              acceleration_enabled=True, residual_exponential_method="taylor",
              maximum_chebyshev_group=8, maximum_host_working_bytes=72*1024**3):
    validate_kinetic_order(order)
    if higher_order_scalar_splitting and (order not in (6, 8) or not scalar_splitting or quartic_scalar_splitting):
        raise ValueError("higher_order_scalar_splitting requires scalar_splitting=True, kinetic_order=6 or 8, and quartic_scalar_splitting=False")
    if quartic_scalar_splitting and order != 4:
        raise ValueError("quartic_scalar_splitting requires kinetic_order=4")
    velocity, scalar, mechanical, frequency, defect, beta = _coefficients(
        snapshot, wave, z, reference, wavelength, sampler, return_energy_terms=True)
    frequency_bound = np.max(abs(frequency), axis=(1, 2))
    generator_bound = float((0. if scalar_splitting else np.max(abs(scalar)))
                            +np.max(abs(velocity), axis=(1, 2))@frequency_bound)
    kinetic_bound = float(np.sum((frequency_bound+np.max(abs(mechanical), axis=(1, 2)))**2))
    defect_bound = float(np.max(abs(defect)))
    squared_root_bound = beta**2*(kinetic_bound+defect_bound)
    if squared_root_bound >= .1:
        raise ValueError("Carrier and envelope exceed the forward small-angle kinetic domain")
    k = 2*np.pi/wavelength
    if order in (6, 8):
        if higher_order_scalar_splitting:
            multiplication = np.sum(mechanical**2, axis=0)+defect
            multiplication_bound = float(np.max(abs(multiplication)))
            linear_bound = float(np.sum(frequency_bound**2
                +2*frequency_bound*np.max(abs(mechanical), axis=(1, 2))))
        for power in range(2, order//2+1):
            coefficient = kinetic_coefficient(power)*beta**(2*power-1)
            if higher_order_scalar_splitting:
                generator_bound += coefficient*differential_power_norm_bound(power, linear_bound, multiplication_bound)
                scalar = scalar+coefficient*multiplication**power
            else:
                generator_bound += coefficient*(kinetic_bound+defect_bound)**power
    elif order == 4:
        if quartic_scalar_splitting:
            if not scalar_splitting:
                raise ValueError("quartic_scalar_splitting requires scalar_splitting=True")
            multiplication = np.sum(mechanical**2, axis=0)+defect
            linear_bound = float(np.sum(frequency_bound**2
                +2*frequency_bound*np.max(abs(mechanical), axis=(1, 2))))
            quartic_bound = beta**3*(linear_bound**2
                +2*linear_bound*float(np.max(abs(multiplication))))/8
            scalar = scalar+beta**3*multiplication**2/8
        else:
            quartic_bound = beta**3*(kinetic_bound+defect_bound)**2/8
        generator_bound += quartic_bound
        # Forward admission and the omitted square-root remainder still use
        # the FULL K norm, including U; exact exponentiation does not remove it.
    remainder = kinetic_remainder_phase_bound(order, k, distance, beta, squared_root_bound)
    if not math.isfinite(remainder) or remainder > remainder_budget:
        raise ValueError("Carrier kinetic truncation exceeds its declared phase budget")
    # This is a local scalar-coefficient guard, not a full potential/grid proof.
    phase = k*distance*scalar*(.5 if scalar_splitting else 1.)
    increment = max(float(np.max(abs(np.diff(phase, axis=axis)))) for axis in (0, 1))
    if increment >= .2*np.pi:
        raise ScalarPhaseSamplingError("Carrier residual scalar phase is undersampled; refine the physical envelope grid",
                                increment/(.2*np.pi))
    count, degree, bound = _series_plan(k*abs(distance)*generator_bound, error)
    from temsim.physics.potential_wave_device import execute_residual_exponential
    amplitude, execution = execute_residual_exponential(
        wave.amplitude, velocity, scalar, mechanical, frequency, defect, beta,
        phase=phase, scalar_splitting=scalar_splitting, kinetic_order=order,
        quartic_scalar_splitting=quartic_scalar_splitting,
        higher_order_scalar_splitting=higher_order_scalar_splitting,
        count=count, degree=degree, k_distance=k*distance,
        boundary_limit=boundary_limit, envelope_limit=envelope_limit, cancelled=cancelled,
        requested=residual_compute_backend, maximum_device_working_bytes=maximum_device_working_bytes,
        acceleration_enabled=acceleration_enabled,
        exponential_method=residual_exponential_method, generator_norm_bound=generator_bound,
        # Each executor may use the same declared half-step allowance. The
        # achieved Taylor remainder is not a new, tighter input tolerance for
        # another recurrence's independently certified approximation.
        series_error_budget=error, maximum_host_working_bytes=maximum_host_working_bytes,
        maximum_chebyshev_group=maximum_chebyshev_group)
    selected_bound = bound
    if execution["exponential_method"] == "chebyshev":
        selected_bound = execution["chebyshev_certificate"]["series_error_bound"]
        if (not math.isfinite(selected_bound) or selected_bound < 0.
                or selected_bound > min(error, 1e-12)):
            raise ValueError("Selected residual exponential exceeds its declared series allowance")
    maximum_boundary = execution["maximum_boundary_flux"]
    maximum_spectral_tail = execution["maximum_envelope_spectral_tail"]
    applications = count*degree
    # Keep the ORIGINAL Taylor allowance for either executor. A different
    # recurrence's roundoff diagnostic never increases this flux tolerance.
    rounding = 128*np.finfo(float).eps*max(1, applications)*(1+math.log2(amplitude.size))
    before, after = wave.probability, float(np.sum(abs(amplitude)**2))
    if not math.isfinite(after) or abs(after-before) > before*(2*bound+bound**2+rounding):
        raise ValueError("Carrier residual failed flux accounting; no normalization was applied")
    return replace(wave, amplitude=amplitude), {"series_error_bound": selected_bound,
        "declared_series_error_budget": error, "original_taylor_series_error_bound": bound,
        "residual_execution": execution,
        "generator_norm_bound": generator_bound, "kinetic_slope_squared_bound": kinetic_bound,
        "electrostatic_momentum_defect_bound": defect_bound, "square_root_expansion_bound": squared_root_bound,
        "kinetic_remainder_phase_bound": remainder, "operator_applications": applications,
        "actual_operator_applications": execution["actual_operator_applications"],
        "roundoff_relative_allowance": rounding, "maximum_scalar_phase_increment": increment,
        "maximum_boundary_flux": maximum_boundary, "boundary_check": "each bounded exponential substep",
        "maximum_envelope_spectral_tail": maximum_spectral_tail,
        "scalar_exact_strang": scalar_splitting,
        "quartic_scalar_exact_strang": quartic_scalar_splitting,
        "higher_order_scalar_exact_strang": higher_order_scalar_splitting,
        "scalar_splitting_error_in_series_bound": False}


def propagate_carrier_potential(snapshot, wave, z_m, *, energy_ev, kinetic_order=4,
        exponential_error=1e-10, kinetic_remainder_phase_limit=.01, boundary_fraction_limit=1e-9,
        envelope_tail_limit=1e-10, reselect_carrier=False, scalar_splitting=False,
        quartic_scalar_splitting=False, higher_order_scalar_splitting=False,
        sampling_backend="snapshot", residual_compute_backend="CPU", grid_numerics=None, cancelled=lambda: False, progress_callback=None,
        residual_exponential_method="taylor", maximum_chebyshev_group=8,
        support_ledger=None, support_mode_weight=1., support_projection_policy="sampling_retry",
        segment_start_m=None, representation_estimates=None,
        reference_quadrature="midpoint", kinetic_phase_ledger=None, electric_reference=None,
        electric_forward_error_limit=1e-6):
    """Return (PlaneWave, evidence) without expanding or deleting its carrier.

    Positive forward steps and admitted native magnetic sources are required.
    An electric source needs its immutable energy reference; a dynamic source
    needs an explicit retarded slice with cumulative clock/work error gates.
    Aligned compact Az faces may lie exactly at declared nodes; tilted or
    unresolved faces, magnetic FEM and ill-conditioned lattices are rejected. Source/canonical/quadrature convergence
    are separate from the reported finite-grid exponential truncation bound.
    Optional exact carrier reselection only changes the numerical chart: its
    compensating complex envelope retains the complete physical phase.
    Quartic scalar splitting is a separate opt-in for real U^2 multiplication;
    it requires scalar_splitting and fourth-order kinetics. It leaves full
    kinetic admission unchanged and needs independent axial convergence.
    Higher-order scalar splitting is a separate option for orders 6 and 8.
    Its noncommuting remainder uses K R_n + L U^n without large-power
    subtraction; the same complete spectral bound governs the kinetic tail.
    An explicit grouped Chebyshev exponential can reuse Hermitian polynomial
    actions while retaining every original Taylor substep check. The default
    remains Taylor, and its original flux allowance remains in force.
    Optional controlled numerical support projection is independently budgeted
    and recorded. The default policy only considers sampling retries. Explicit
    step-start projection can also reduce redundant canonical padding. Neither
    policy is physical absorption or a propagated complex-error certificate.
    """
    _admit(snapshot, electric_reference=electric_reference)
    energy = _positive(energy_ev, "Kinetic energy")*e
    error = _positive(exponential_error, "Exponential error budget")
    remainder_limit = _positive(kinetic_remainder_phase_limit, "Kinetic phase budget")
    electric_error_limit = _positive(electric_forward_error_limit, "Added electric forward error budget")
    boundary_limit = _positive(boundary_fraction_limit, "Boundary fraction", zero=True)
    envelope_limit = _positive(envelope_tail_limit, "Envelope spectral tail", zero=True)
    if boundary_limit >= 1 or envelope_limit >= 1:
        raise ValueError("Boundary and envelope spectral tail fractions must be less than one")
    validate_kinetic_order(kinetic_order)
    if type(reselect_carrier) is not bool:
        raise ValueError("reselect_carrier must be an explicit boolean")
    if type(scalar_splitting) is not bool:
        raise ValueError("scalar_splitting must be an explicit boolean")
    if type(quartic_scalar_splitting) is not bool:
        raise ValueError("quartic_scalar_splitting must be an explicit boolean")
    if quartic_scalar_splitting and (not scalar_splitting or kinetic_order != 4):
        raise ValueError("quartic_scalar_splitting requires scalar_splitting=True and kinetic_order=4")
    if type(higher_order_scalar_splitting) is not bool:
        raise ValueError("higher_order_scalar_splitting must be an explicit boolean")
    if higher_order_scalar_splitting and (not scalar_splitting or kinetic_order not in (6, 8) or quartic_scalar_splitting):
        raise ValueError("higher_order_scalar_splitting requires scalar_splitting=True, kinetic_order=6 or 8, and quartic_scalar_splitting=False")
    if type(sampling_backend) is not str or sampling_backend not in ("snapshot", "native_compiled", "native_plane"):
        raise ValueError("sampling_backend must be snapshot, native_compiled or native_plane")
    residual_compute_backend = normalise_backend(residual_compute_backend)
    if type(residual_exponential_method) is not str or residual_exponential_method not in ("taylor", "chebyshev"):
        raise ValueError("residual_exponential_method must be 'taylor' or 'chebyshev'")
    if type(maximum_chebyshev_group) is not int or not 1 <= maximum_chebyshev_group <= 32:
        raise ValueError("maximum_chebyshev_group must be an explicit integer from 1 to 32")
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    if support_ledger is not None and type(support_ledger) is not NumericalSupportLedger:
        raise ValueError("support_ledger must be an immutable NumericalSupportLedger or None")
    if kinetic_phase_ledger is not None and type(kinetic_phase_ledger) is not KineticPhaseLedger:
        raise ValueError("kinetic_phase_ledger must be an immutable KineticPhaseLedger or None")
    support_weight = _positive(support_mode_weight, "Support mode weight", zero=True)
    if support_projection_policy not in ("sampling_retry", "step_start_and_retry", "step_boundaries_and_retry", "weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry"):
        raise ValueError("Unsupported controlled support projection policy")
    phase_preview = support_projection_policy == "preview_phase_segment_and_retry"
    from temsim.physics.potential_representation_estimates import (
        RepresentationEstimateLedger, RepresentationBudgetError,
        require_combined_support_allowance,
    )
    if isinstance(representation_estimates, Mapping):
        representation_estimates = RepresentationEstimateLedger.from_dict(representation_estimates)
    if phase_preview:
        if support_ledger is None:
            raise ValueError("Preview phase preparation requires its original active support ledger")
        if representation_estimates is None:
            representation_estimates = RepresentationEstimateLedger.for_support(support_ledger)
        if type(representation_estimates) is not RepresentationEstimateLedger:
            raise ValueError("Preview phase preparation requires an immutable representation history")
        require_combined_support_allowance(support_ledger, representation_estimates)
    elif representation_estimates is not None:
        raise ValueError("Representation estimates require the explicit preview phase policy")
    if segment_start_m is not None:
        if (support_projection_policy not in ("preview_segment_and_retry", "preview_phase_segment_and_retry")
                or isinstance(segment_start_m, (bool, np.bool_))
                or not np.isscalar(segment_start_m) or not np.isreal(segment_start_m)
                or not math.isfinite(float(segment_start_m))):
            raise ValueError("A preview segment start requires its explicit policy and a finite real plane")
        segment_start_m = float(segment_start_m)
    if reference_quadrature not in ("midpoint", "cf4"):
        raise ValueError("Quadratic reference quadrature must be midpoint or cf4")
    initial_support_count = 0 if support_ledger is None else len(support_ledger.events)
    sampler = None
    if sampling_backend == "native_compiled":
        if electric_reference is not None:
            raise ValueError("Compiled magnetic-only sampling cannot replace the captured scalar potential")
        from temsim.physics.potential_native_sampler import compile_native_sampler
        sampler = compile_native_sampler(snapshot)
        sampler.require_snapshot(snapshot)
    elif sampling_backend == "native_plane":
        if electric_reference is not None:
            raise ValueError("Magnetic plane sampling cannot replace the captured scalar potential")
        from temsim.physics.potential_plane_sampler import compile_plane_sampler
        sampler = compile_plane_sampler(snapshot)
        sampler.require_snapshot(snapshot)
    elif electric_reference is not None and type(snapshot) is ElectromagneticPotentialSnapshot:
        from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
        sampler = compile_unified_potential_sampler(snapshot, prefer_plane=True)
    nodes = np.asarray(z_m)
    if (nodes.dtype.kind not in "iuf" or nodes.ndim != 1 or len(nodes) < 2
            or not np.isfinite(nodes).all() or not np.all(np.diff(nodes) > 0)):
        raise ValueError("Carrier axial nodes must be finite and strictly increasing")
    nodes = np.asarray(nodes, dtype=float)
    if segment_start_m is not None and segment_start_m > nodes[0]:
        raise ValueError("A preview segment start cannot be after the current propagation entrance")
    if kinetic_phase_ledger is not None:
        kinetic_phase_ledger = kinetic_phase_ledger.require(snapshot.numerical_digest,
            energy_ev if electric_reference is None else electric_reference.entrance_energy_ev,
            kinetic_order, float(nodes[0]))
    accepted_kinetic_bounds = []
    numerics = WaveGridNumerics() if grid_numerics is None else grid_numerics
    if type(numerics) is not WaveGridNumerics:
        raise ValueError("Carrier wave requires explicit WaveGridNumerics")
    residual_resource_check, residual_resource_preference = _residual_grid_resource_checks(
        numerics, residual_compute_backend)
    momentum = math.sqrt(energy*(energy+2*m_e*c*c))/c
    wavelength = 2*np.pi*hbar/momentum
    result, records, charts, electric_errors, retarded_steps = wave, [], [], [], []
    from temsim.physics.potential_wave_device import residual_host_working_bytes
    def validate(candidate):
        numerics.check_working_memory(candidate.amplitude.shape,
            working_bytes=256*candidate.amplitude.size,
            retained_bytes=_additional_wave_bytes(candidate, wave.amplitude))
        return _validate_wave(candidate, numerics, boundary_limit, envelope_limit)
    weighted_resampling = support_projection_policy in ("weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry")
    def check_combined_allowance(ledger, history=None):
        if phase_preview:
            require_combined_support_allowance(ledger,
                representation_estimates if history is None else history)
    preview_initial_attempted = False
    def prepare_support(z, lower, upper, stage, retained_amplitudes=(), transaction_entrance=None):
        projection_committed = False
        def prepare(candidate, sampling_error, pressure=None):
            nonlocal support_ledger, projection_committed
            if support_ledger is None or projection_committed:
                return candidate
            if weighted_resampling and not isinstance(sampling_error, WaveGridBudgetError):
                # Preserve the finite field while resources can resolve it.
                # Ordinary sampling retries first use lossless refinement;
                # a segment marker alone does not authorize numerical loss.
                return candidate
            original_ledger = support_ledger
            if weighted_resampling:
                if pressure is None:
                    return candidate
                if transaction_entrance is not None and not transaction_entrance:
                    transaction_entrance.append(candidate.amplitude)
                projection_order = (("spatial", "spectral")
                    if _scalar_phase_resource_pressure(sampling_error)
                    else ("spectral", "spatial"))
            else:
                projection_order = ("spatial",)

            def decline(kind, shape, reason):
                # Plain text survives a final resource failure without retaining
                # an unaccepted array or traceback. Successful runs also keep the
                # same information in their ordinary chart diagnostics.
                charts.append({"method": "support projection proposal declined",
                    "candidate_kind": kind, "stage": stage, "z_m": float(z),
                    "from_shape": candidate.amplitude.shape, "to_shape": shape,
                    "reason": reason, "projection_budget_spent": False})
                if isinstance(sampling_error, WaveGridBudgetError):
                    sampling_error.add_note(
                        f"Support {kind} candidate declined at {stage}, z={float(z):.17g} m, "
                        f"shape {candidate.amplitude.shape} -> {shape}: {reason}")

            # A pending spectral window can need lossless admission growth
            # before the physical operator runs. Measure that separate growth
            # instead of returning a proposal the outer monotone check rejects.
            from temsim.physics.potential_wave_support import SPATIAL_PRESSURE_SHARES
            projection_queue = [(kind, share) for kind in projection_order
                for share in (SPATIAL_PRESSURE_SHARES if weighted_resampling and kind == "spatial" else (None,))]
            spectral_admission_attempts = 0
            admission_growth = 1.
            previous_raw_shape = None
            while projection_queue:
                if cancelled():
                    raise InterruptedError("Support candidate preparation cancelled")
                projection_kind, spatial_share = projection_queue.pop(0)
                prepared, pending_ledger = candidate, original_ledger
                if projection_kind == "spatial":
                    # A fixed-period filter cannot reduce a scalar coefficient's
                    # full-domain excursion. A spatial window can remove empty
                    # edges, and can also avoid ringing from a rejected spectral
                    # proposal. Every pressure share starts from the original
                    # entrance and ledger; ordinary compact() remains unchanged.
                    retained = _additional_wave_bytes(candidate, wave.amplitude, *retained_amplitudes)
                    try:
                        numerics.check_working_memory(candidate.amplitude.shape,
                            working_bytes=256*candidate.amplitude.size,
                            retained_bytes=retained)
                    except WaveGridBudgetError as error:
                        decline(projection_kind, candidate.amplitude.shape,
                            f"{type(error).__name__}: {error}")
                        continue
                    if weighted_resampling:
                        # A cropped physical domain has a different scalar
                        # phase excursion. Do not reuse the old spectral
                        # growth estimate; actual admission and the unchanged
                        # physical operator must resolve the new domain.
                        spatial_context = {"schema": "wave-resource-pressure-v1", "stage": stage,
                            "failure_kind": "WaveMemoryBudgetError" if type(sampling_error).__name__ == "WaveMemoryBudgetError" else "WaveGridBudgetError",
                            "input_shape": list(candidate.amplitude.shape),
                            "maximum_host_working_bytes": numerics.maximum_working_bytes,
                            "maximum_device_working_bytes": numerics.maximum_device_working_bytes,
                            "maximum_pixels": numerics.maximum_pixels, "retained_host_bytes": retained,
                            "operator_memory_policy": "carrier-residual-host-512-plus-64MiB/device-448",
                            "requested_backend": normalise_backend(residual_compute_backend),
                            "growth_guard_scale": 1., "candidate_family": "centered_quintic_crop_remaining_share_v1",
                            "source_failure_shape": list(pressure.required_shape)}
                        if pressure.previous_shape is not None:
                            spatial_context["previous_shape"] = list(pressure.previous_shape)
                        prepared, pending_ledger = original_ledger.compact_pressure(candidate,
                            wavelength_m=wavelength, z_m=z, mode_weight=support_weight,
                            allocation_fraction=spatial_share, resource_context=spatial_context,
                            numerics=numerics, retained_bytes=retained, cancelled=cancelled)
                    else:
                        prepared, pending_ledger = original_ledger.compact(candidate,
                            wavelength_m=wavelength, z_m=z, mode_weight=support_weight)
                    if (weighted_resampling and prepared is not candidate
                            and pressure.previous_shape is not None
                            and (prepared.amplitude.size >= math.prod(pressure.previous_shape)
                                 or any(n > old for n, old in
                                        zip(prepared.amplitude.shape, pressure.previous_shape)))):
                        decline(projection_kind, prepared.amplitude.shape,
                            "Candidate does not strictly reduce the previous resource proposal")
                        prepared, pending_ledger = candidate, original_ledger
                        continue
                else:
                    spectral_admission_attempts += 1
                    retained = _additional_wave_bytes(candidate, wave.amplitude, *retained_amplitudes)
                    admitted_admissible, baseline_context = _pressure_resampling_constraints(candidate,
                        sampling_error, numerics=numerics, requested=residual_compute_backend,
                        retained_bytes=retained, stage=stage, pressure=pressure)
                    adjusted_pressure = replace(pressure,
                        growth_scale=baseline_context["growth_guard_scale"]*admission_growth)
                    resource_admissible, resource_context = _pressure_resampling_constraints(candidate,
                        sampling_error, numerics=numerics, requested=residual_compute_backend,
                        retained_bytes=retained, stage=stage, pressure=adjusted_pressure)
                    def admissible(shape):
                        return (previous_raw_shape is None or (
                            math.prod(shape) < math.prod(previous_raw_shape)
                            and all(n <= old for n, old in zip(shape, previous_raw_shape)))) and resource_admissible(shape)
                    try:
                        prepared, pending_ledger = original_ledger.resample_spectrum_pressure(candidate,
                            z_m=z, mode_weight=support_weight, numerics=numerics,
                            retained_bytes=retained, cancelled=cancelled,
                            admissible_shape=admissible, resource_context=resource_context)
                    except WaveGridBudgetError as error:
                        # Optional projection workspace cannot borrow from the
                        # physical operator or change the selected backend.
                        decline(projection_kind, candidate.amplitude.shape,
                            f"{type(error).__name__}: {error}")
                        continue
                if prepared is candidate:
                    decline(projection_kind, candidate.amplitude.shape,
                        "No smaller candidate satisfies its existing projection and resource limits")
                    continue
                try:
                    check_combined_allowance(pending_ledger)
                except RepresentationBudgetError as error:
                    decline(projection_kind, prepared.amplitude.shape, str(error))
                    prepared, pending_ledger = candidate, original_ledger
                    continue
                raw_projection_shape = prepared.amplitude.shape
                # Windowing changes the represented period and can broaden the
                # retained spectrum. A projection is only a proposal until the
                # ordinary lossless Fourier preparation and every original gate
                # admit it. Failed proposals spend no probability/L2 allowance.
                def admit(projected):
                    validate(projected)
                    _coverage(snapshot, projected, lower, upper)
                    if weighted_resampling and projection_kind == "spatial":
                        numerics.check_working_memory(projected.amplitude.shape,
                            working_bytes=residual_host_working_bytes(projected.amplitude.shape),
                            retained_bytes=_additional_wave_bytes(projected, candidate.amplitude,
                                wave.amplitude, raw_projection.amplitude, *retained_amplitudes))
                    return projected
                # An outer operator retry can retain its original entrance even
                # after Fourier refinement has replaced the current candidate.
                # Count that distinct buffer as well as the kernel entrance;
                # shared arrays/views retain their backing allocation only once.
                # Every optional window starts from this SAME unexecuted
                # spectral wave and ledger. Failed windows cannot accumulate
                # attenuation, factorize the mode weight, or spend a receipt.
                raw_projection, raw_ledger = prepared, pending_ledger
                boundary_width, remaining_widths = 0, []
                forecast_failure_shape = None
                while True:
                    retained = _additional_wave_bytes(prepared, candidate.amplitude, wave.amplitude,
                                                      raw_projection.amplitude, *retained_amplitudes)
                    proposed_shape = prepared.amplitude.shape
                    try:
                        check_combined_allowance(pending_ledger)
                        prepared, refinements = apply_resolved_operator(prepared, admit,
                            numerics=numerics, cancelled=cancelled,
                            retained_bytes=retained,
                            operator_working_bytes=lambda shape: 256*math.prod(shape),
                            operator_resource_check=residual_resource_check,
                            preferred_resource_check=residual_resource_preference)
                    except (WaveSamplingError, WaveGridBudgetError, OccupiedCarrierBoundaryError) as error:
                        # Domain errors and cancellation are deliberately not
                        # swallowed. Only this unexecuted entrance admission can
                        # propose a taper; no failed physical output is windowed.
                        kind = (projection_kind+f" with boundary taper ({boundary_width} cells)"
                                if boundary_width else projection_kind)
                        decline(kind, proposed_shape, f"{type(error).__name__}: {error}")
                        if (projection_kind == "spectral" and not boundary_width
                                and isinstance(error, OccupiedCarrierBoundaryError)):
                            remaining_widths = [2, 4, 8, 16]
                        import traceback
                        failures, seen = [error], set()
                        while failures:
                            failure = failures.pop()
                            if failure is None or id(failure) in seen:
                                continue
                            seen.add(id(failure))
                            traceback.clear_frames(failure.__traceback__)
                            failures.extend((failure.__cause__, failure.__context__))
                        del failures, failure
                    else:
                        if (weighted_resampling and projection_kind == "spatial"
                                and pressure.previous_shape is not None
                                and (prepared.amplitude.size >= math.prod(pressure.previous_shape)
                                     or any(n > old for n, old in zip(prepared.amplitude.shape, pressure.previous_shape)))):
                            decline("spatial admission", prepared.amplitude.shape,
                                f"Remaining share {spatial_share} passes physical entrance gates but does not "
                                "strictly reduce the previous resource proposal after lossless refinement")
                            del refinements
                        elif (weighted_resampling and projection_kind == "spectral"
                                and not admitted_admissible(prepared.amplitude.shape)):
                            shape = prepared.amplitude.shape
                            if (forecast_failure_shape is None
                                    or math.prod(shape) < math.prod(forecast_failure_shape)):
                                forecast_failure_shape = shape
                            decline("spectral boundary admission" if boundary_width else "spectral admission",
                                shape, f"Admitted {boundary_width or 'unwindowed'}-cell candidate does not satisfy "
                                "the unchanged previous-proposal and physical-operator resource forecast")
                            del refinements
                        else:
                            break
                    # Release a failed window and any lossless embedding before
                    # allocating another. The original pending spectral input
                    # remains explicitly reserved throughout this small search.
                    prepared, pending_ledger = candidate, original_ledger
                    while remaining_widths:
                        if cancelled():
                            raise InterruptedError("Support candidate boundary taper cancelled")
                        boundary_width = remaining_widths.pop(0)
                        retained = _additional_wave_bytes(raw_projection, candidate.amplitude,
                            wave.amplitude, *retained_amplitudes)
                        try:
                            numerics.check_working_memory(raw_projection.amplitude.shape,
                                working_bytes=256*raw_projection.amplitude.size, retained_bytes=retained)
                        except WaveGridBudgetError as budget_error:
                            decline("spectral boundary taper", raw_projection_shape,
                                f"{type(budget_error).__name__}: {budget_error}")
                            break
                        tapered, taper_ledger = raw_ledger.taper_boundary(raw_projection,
                            z_m=z, mode_weight=support_weight, transition_cells=boundary_width)
                        if cancelled():
                            raise InterruptedError("Support candidate boundary taper cancelled")
                        if tapered is not raw_projection:
                            prepared, pending_ledger = tapered, taper_ledger
                            del tapered, taper_ledger
                            break
                        del tapered, taper_ledger
                        decline("spectral boundary taper", raw_projection_shape,
                            f"The {boundary_width}-cell window does not satisfy its existing projection limits")
                    if prepared is candidate:
                        break
                del raw_projection, raw_ledger
                if prepared is candidate and forecast_failure_shape is not None:
                    admitted_shape = forecast_failure_shape
                    admission_growth = max(admission_growth,
                        *(n/raw for n, raw in zip(admitted_shape, raw_projection_shape)))
                    decline("spectral admission", admitted_shape,
                        "Lossless candidate admission growth from "
                        f"{raw_projection_shape} to {admitted_shape} does not satisfy "
                        "the strict previous-proposal and physical-operator resource forecast; "
                        f"measured admission growth={admission_growth:.17g}")
                    charts.append({"method": "support candidate admission forecast retry",
                        "stage": stage, "z_m": float(z),
                        "raw_shape": raw_projection_shape, "admitted_shape": admitted_shape,
                        "admission_growth_scale": admission_growth,
                        "operator_growth_guard_scale": baseline_context["growth_guard_scale"],
                        "composite_growth_guard_scale": baseline_context["growth_guard_scale"]*admission_growth,
                        "attempt": spectral_admission_attempts, "maximum_attempts": 6,
                        "projection_budget_spent": False})
                    previous_raw_shape = raw_projection_shape
                    # Release this entire uncommitted pair before allocating
                    # the next proposal from the same original wave and ledger.
                    prepared, pending_ledger = candidate, original_ledger
                    if spectral_admission_attempts < 6:
                        projection_queue.insert(0, ("spectral", None))
                    else:
                        decline("spectral admission", admitted_shape,
                            "Bounded candidate admission retries exhausted after 6 proposals")
                    continue
                if prepared is candidate:
                    continue
                break
            else:
                return candidate
            projection_record = {"method": "budgeted support projection with full candidate admission",
                "stage": stage, "z_m": float(z), "from_shape": candidate.amplitude.shape,
                "to_shape": prepared.amplitude.shape, "probability": prepared.probability,
                "retained_input_bytes": retained,
                "error_scope": "projection only; no propagated complex-error certificate"}
            def commit():
                nonlocal support_ledger, projection_committed
                if projection_committed or support_ledger is not original_ledger:
                    raise ValueError("Support projection transaction lost its original ledger")
                check_combined_allowance(pending_ledger)
                charts.append(projection_record)
                charts.extend({**row, "stage": stage+" Fourier operator preparation"} for row in refinements)
                support_ledger = pending_ledger
                projection_committed = True
            if weighted_resampling:
                return WaveResourceProposal(prepared, commit)
            commit()
            return prepared
        return WaveResourceTransaction(prepare) if weighted_resampling else prepare
    def residual(current, middle, half_distance, reference, lower, upper, plane_z):
        nonlocal support_ledger, representation_estimates, preview_initial_attempted
        evidence = []
        transaction_entrance = []
        def apply(candidate):
            # Fourier embedding retains the period but moves its last sample
            # closer to the period edge. Every retry must admit both physical
            # end planes, including the final residual of the final step.
            _coverage(snapshot, candidate, lower, upper)
            phase_available = (remainder_limit/(2*(len(nodes)-1)) if kinetic_phase_ledger is None else
                kinetic_phase_ledger.maximum_phase_rad-math.fsum([
                    *(value for event in kinetic_phase_ledger.events for value in event["accepted_bounds_rad"]),
                    *accepted_kinetic_bounds]))
            output, record = _residual(snapshot, candidate, middle, half_distance, reference, wavelength,
                kinetic_order, error/(2*(len(nodes)-1)), phase_available,
                boundary_limit, envelope_limit, cancelled, scalar_splitting, sampler,
                quartic_scalar_splitting, higher_order_scalar_splitting,
                residual_compute_backend, numerics.maximum_device_working_bytes, numerics.acceleration_enabled,
                residual_exponential_method=residual_exponential_method,
                maximum_chebyshev_group=maximum_chebyshev_group,
                maximum_host_working_bytes=numerics.maximum_working_bytes
                    -_additional_wave_bytes(candidate, wave.amplitude, current.amplitude,
                                            *transaction_entrance))
            evidence.append(record)
            return output
        output = None
        if (support_projection_policy in ("preview_segment_and_retry", "preview_phase_segment_and_retry")
                and support_ledger is not None and not preview_initial_attempted
                and segment_start_m == float(nodes[0]) == float(plane_z)):
            # Only an already executed segment entrance is eligible. A private
            # proposal must pass the actual residual before its ledger is used;
            # a failed trial cannot spend any source allowance or repair an
            # unresolved physical output. Later halves never repeat this offer.
            from temsim.physics.potential_segment_resampling import prepare_segment_resampling
            preview_initial_attempted = True
            original_ledger = support_ledger
            original_estimates = representation_estimates
            retained = _additional_wave_bytes(current, wave.amplitude)

            def admit_preview(candidate, *, retained_bytes):
                def admission(item):
                    _validate_wave(item, numerics, boundary_limit, envelope_limit)
                    _coverage(snapshot, item, lower, upper)
                    return item
                admitted, _ = apply_resolved_operator(candidate, admission, numerics=numerics,
                    cancelled=cancelled, retained_bytes=retained_bytes,
                    operator_working_bytes=lambda shape: 256*math.prod(shape))
                return admitted

            if phase_preview:
                from temsim.physics.potential_segment_resampling import prepare_phase_segment_resampling
                proposal = prepare_phase_segment_resampling(current, original_ledger, original_estimates,
                    wavelength_m=wavelength, z_m=plane_z, mode_weight=support_weight,
                    numerics=numerics, retained_bytes=retained, admit=admit_preview, cancelled=cancelled)
            else:
                proposal = prepare_segment_resampling(current, original_ledger,
                    z_m=plane_z, mode_weight=support_weight, numerics=numerics,
                    retained_bytes=retained, admit=admit_preview, cancelled=cancelled)
            if proposal.wave is not current:
                class PreviewCandidateNotSmaller(Exception):
                    """A voluntary trial lost its declared grid-size benefit."""

                def preview_resources(shape):
                    if (any(n > old for n, old in zip(shape, current.amplitude.shape))
                            or math.prod(shape) > .95*current.amplitude.size):
                        raise PreviewCandidateNotSmaller("Lossless admission removed the preview grid-size benefit")
                    if residual_resource_check is not None:
                        residual_resource_check(shape)

                # A lossless retry can replace the current candidate while
                # the proposal still owns its original amplitude. The inner
                # exponential planner must reserve that live buffer too.
                transaction_entrance.append(proposal.wave.amplitude)
                try:
                    output, refinements = apply_resolved_operator(proposal.wave, apply,
                        numerics=numerics, cancelled=cancelled,
                        retained_bytes=retained+current.amplitude.nbytes,
                        operator_working_bytes=residual_host_working_bytes,
                        operator_resource_check=preview_resources,
                        preferred_resource_check=residual_resource_preference)
                except (PreviewCandidateNotSmaller, WaveSamplingError,
                        WaveGridBudgetError, OccupiedCarrierBoundaryError) as rejected:
                    charts.append({"method": "segment preview proposal declined by physical residual",
                        "z_m": float(plane_z), "reason": str(rejected),
                        "projection_budget_spent": False})
                    evidence.clear()
                else:
                    if (support_ledger is not original_ledger
                            or representation_estimates is not original_estimates):
                        raise ValueError("Preview proposal lost its unchanged source ledger")
                    if cancelled():
                        raise InterruptedError("Preview projection commit cancelled")
                    if phase_preview:
                        check_combined_allowance(proposal.ledger, proposal.representation_estimates)
                        representation_estimates = proposal.representation_estimates
                    support_ledger = proposal.ledger
                    charts.append({"method": "segment preview projection accepted after physical residual",
                        "z_m": float(plane_z), "from_shape": current.amplitude.shape,
                        "to_shape": output.amplitude.shape, "projection_budget_spent": True,
                        "preparation": proposal.diagnostics,
                        "error_scope": ("strict projection cost plus uncertified representation estimate; not a propagated error bound"
                            if phase_preview else "projection only; subsequent propagation needs independent convergence")})
                finally:
                    del proposal
                    transaction_entrance.clear()
            else:
                charts.append({"method": "segment preview preparation retained original wave",
                    "z_m": float(plane_z), "preparation": proposal.diagnostics,
                    "projection_budget_spent": False})
                del proposal
        if output is None:
            preparation = prepare_support(plane_z, lower, upper, "residual retry", (current.amplitude,),
                                          transaction_entrance)
            output, refinements = apply_resolved_operator(current, apply, numerics=numerics,
                cancelled=cancelled, operator_working_bytes=residual_host_working_bytes,
                retained_bytes=_additional_wave_bytes(current, wave.amplitude),
                prepare_retry=preparation,
                prepare_resource=preparation if weighted_resampling else None,
                operator_resource_check=residual_resource_check,
                preferred_resource_check=residual_resource_preference)
        charts.extend(refinements)
        if kinetic_phase_ledger is not None:
            # Failed candidate refinements have no phase expenditure. Only
            # the final admitted result contributes to the cumulative bound.
            accepted_kinetic_bounds.append(evidence[-1]["kinetic_remainder_phase_bound"])
        return output, evidence[-1]
    # An executed entrance can legitimately occupy high retained DFT bins.
    # Prepare the same finite Fourier state for nonlinear products before
    # applying any operator. This is not missing source-detail recovery or
    # continuous-source convergence evidence. Only typed sampling failures
    # trigger unitary embedding; domain/boundary/type failures remain final.
    if type(result) is not PlaneWave or not isinstance(result.amplitude, np.ndarray):
        raise ValueError("Carrier benchmark requires a CPU PlaneWave")
    entrance_boundary = []
    entrance_projection_input = []
    def admit_entrance(candidate):
        nonlocal support_ledger
        pending_ledger = support_ledger
        try:
            boundary = validate(candidate)
        except OccupiedCarrierBoundaryError:
            # Only a Fourier-prepared candidate can enter this branch. The
            # supplied entrance itself must pass its physical edge check.
            # Refinement can expose tiny edge samples absent from the coarse
            # lattice. An explicit numerical ledger may pay for a SAME-grid
            # taper, never silently zero-extend or repair the source.
            if support_ledger is None or candidate is wave:
                raise
            prepared, pending_ledger = support_ledger.taper_boundary(candidate,
                z_m=float(nodes[0]), mode_weight=support_weight)
            if prepared is candidate:
                raise
            original_probability = candidate.probability
            boundary = validate(prepared)
            candidate = prepared
        else:
            original_probability = None
        _coverage(snapshot, candidate, float(nodes[0]), float(nodes[0]))
        # A failed spectral/domain candidate spends no projection allowance.
        check_combined_allowance(pending_ledger)
        support_ledger = pending_ledger
        if original_probability is not None:
            entrance_projection_input.append(original_probability)
        entrance_boundary.append(boundary)
        return candidate
    entrance_preparation = prepare_support(float(nodes[0]), nodes[0], nodes[0], "segment entrance")
    if weighted_resampling and support_ledger is not None:
        # Resample only the already executed input. No failed optical output
        # may acquire a projection receipt to conceal unresolved propagation.
        _validate_wave_lattice(result)
        _check_boundary(result.amplitude, boundary_limit)
        _coverage(snapshot, result, float(nodes[0]), float(nodes[0]))
        before = result.probability
        result = entrance_preparation(result, None)
        if result is not wave:
            entrance_projection_input.append(before)
    result, initial_refinements = apply_resolved_operator(result, admit_entrance,
        numerics=numerics, cancelled=cancelled,
        retained_bytes=_additional_wave_bytes(result, wave.amplitude),
        prepare_resource=entrance_preparation if weighted_resampling else None,
        operator_working_bytes=lambda shape: 256*math.prod(shape))
    charts.extend({**row, "stage": "entrance Fourier operator preparation",
                   "continuous_source_convergence": "not established by this lossless preparation"}
                  for row in initial_refinements)
    maximum_boundary = entrance_boundary[-1]
    # Resource preparation can now project during entrance admission as well.
    # Report the supplied physical input before every numerical projection.
    initial = wave.probability
    for step_index, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):
        if cancelled():
            raise InterruptedError("Carrier potential propagation cancelled")
        if support_ledger is not None and support_projection_policy in ("step_start_and_retry", "step_boundaries_and_retry", "weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry"):
            result = prepare_support(float(lower), lower, upper, "step-start support")(result, None)
            maximum_boundary = max(maximum_boundary,
                validate(result))
        if reselect_carrier:
            result, selected = reselect_phase_carrier(result, wavelength,
                grid_numerics=numerics, cancelled=cancelled,
                retained_bytes=_additional_wave_bytes(result, wave.amplitude))
            charts.extend(selected)
            maximum_boundary = max(maximum_boundary,
                validate(result))
        middle, distance = .5*(lower+upper), upper-lower
        _coverage(snapshot, result, lower, upper)
        reference = _reference(snapshot, result, middle, distance, momentum, reference_quadrature,
                               axial_interval=(float(lower), float(upper)), electric_reference=electric_reference,
                               chart_energy_ev=float(energy_ev))
        swept_radius = _swept_coverage(snapshot, result, lower, upper, reference, wavelength)
        electric_radius = float(np.linalg.norm(abs(reference.center_xy)+np.asarray(swept_radius)))
        if electric_reference is not None:
            prospective = electric_reference.forward_variation_bound(snapshot, lower, upper, electric_radius)
            if math.expm1(math.fsum([*(math.log1p(x) for x in electric_errors), math.log1p(prospective)])) > electric_error_limit:
                raise ValueError("Electric forward branch approximation exceeds its remaining amplitude budget")
        result, record = residual(result, middle, distance/2, reference, lower, upper, lower)
        records.append(record)
        record["canonical_swept_xy_half_width_m"] = swept_radius
        maximum_boundary = max(maximum_boundary, record["maximum_boundary_flux"])
        maximum_boundary = max(maximum_boundary, validate(result))
        canonical_entrance = result
        canonical_transaction_entrance = []
        def canonical(candidate):
            # Refinement retains the physical period, but adds represented
            # Fourier covectors. Recheck the complete swept momentum box.
            record["canonical_swept_xy_half_width_m"] = _swept_coverage(
                snapshot, candidate, lower, upper, reference, wavelength)
            propagated = propagate_plane_wave(candidate, reference.path.matrix, reference.path.offset, wavelength,
                **reference.path.phase_kwargs(), grid_numerics=numerics, cancelled=cancelled,
                retained_bytes=_additional_wave_bytes(candidate, wave.amplitude, canonical_entrance.amplitude,
                                                      *canonical_transaction_entrance),
                sampling_records=charts)
            # This is a private trial until its OUTPUT spectrum is admitted.
            # A failed canonical output must refine and rerun the unchanged
            # entrance, just like a residual trial; it cannot bypass the
            # retry path or be projected into an apparently valid result.
            _check_envelope_sampling(propagated.amplitude, envelope_limit)
            return propagated
        preparation = prepare_support(lower, lower, upper, "canonical retry", (canonical_entrance.amplitude,),
                                      canonical_transaction_entrance)
        result, refinements = apply_resolved_operator(result, canonical, numerics=numerics,
            cancelled=cancelled, operator_working_bytes=lambda shape: 256*math.prod(shape),
            retained_bytes=_additional_wave_bytes(result, wave.amplitude),
            prepare_retry=preparation,
            prepare_resource=preparation if weighted_resampling else None)
        # The preparation closure also owns the unchanged entrance tuple.
        # Drop it before the next residual reserves its own working memory.
        del preparation, canonical_entrance
        canonical_transaction_entrance.clear()
        charts.extend(refinements)
        if support_ledger is not None and support_projection_policy in ("step_boundaries_and_retry", "weighted_segment_and_retry", "preview_segment_and_retry", "preview_phase_segment_and_retry"):
            # H2 has already executed. Its full output must pass all physical
            # and sampling gates before a numerical projection can be used.
            # The diagnostic/window workspace is smaller than a *subsequent*
            # high-order residual, but still independently
            # admitted, including the retained original entrance buffer.
            _validate_wave_lattice(result)
            numerics.check_working_memory(result.amplitude.shape,
                working_bytes=256*result.amplitude.size,
                retained_bytes=0 if result.amplitude is wave.amplitude else wave.amplitude.nbytes)
            maximum_boundary = max(maximum_boundary, _check_boundary(result.amplitude, boundary_limit))
            _check_envelope_sampling(result.amplitude, envelope_limit)
            _coverage(snapshot, result, lower, upper)
            full_radius = float(np.max(np.linalg.norm(_corners(result, upper)[:, :2], axis=1)))
            electric_radius = max(electric_radius, full_radius)
            records[-1]["unprojected_canonical_output_radius_m"] = full_radius
            result = prepare_support(float(upper), lower, upper, "post-canonical support")(result, None)
        maximum_boundary = max(maximum_boundary, validate(result))
        _coverage(snapshot, result, lower, upper)
        result, record = residual(result, middle, distance/2, reference, lower, upper, upper)
        records.append(record)
        maximum_boundary = max(maximum_boundary, record["maximum_boundary_flux"])
        maximum_boundary = max(maximum_boundary, validate(result))
        if electric_reference is not None:
            final_radius = float(np.max(np.linalg.norm(_corners(result, upper)[:, :2], axis=1)))
            electric_radius = max(electric_radius, final_radius,
                float(np.linalg.norm(abs(reference.center_xy)+np.asarray(
                    records[-2]["canonical_swept_xy_half_width_m"]))))
            actual = electric_reference.forward_variation_bound(snapshot, lower, upper, electric_radius)
            electric_errors.append(actual)
            if math.expm1(math.fsum(math.log1p(x) for x in electric_errors)) > electric_error_limit:
                raise ValueError("Refined electric forward branch exceeds its remaining amplitude budget")
            record["electric_forward_cylinder_radius_m"] = electric_radius
            record["electric_forward_error_bound"] = actual
        if hasattr(snapshot, "source_snapshot"):
            retarded_steps.append({"lower_z_m": float(lower), "upper_z_m": float(upper),
                "radius_m": electric_radius,
                "kinetic_squared_bound": max(records[-2]["square_root_expansion_bound"],
                                             records[-1]["square_root_expansion_bound"])})
            snapshot.advance_adiabatic_state(retarded_steps)  # Reject before continuing an unaffordable path.
        if progress_callback is not None:
            progress_callback({"completed_steps": step_index+1, "total_steps": len(nodes)-1,
                               "z_m": float(upper), "shape": result.amplitude.shape,
                               "probability": result.probability})
    series = math.expm1(sum(math.log1p(row["series_error_bound"]) for row in records))
    if kinetic_phase_ledger is not None:
        kinetic_phase_ledger = kinetic_phase_ledger.append(float(nodes[0]), float(nodes[-1]),
                                                          accepted_kinetic_bounds)
    check_combined_allowance(support_ledger)
    return result, {"method": "quadratic metaplectic reference plus Hermitian carrier residual",
        "snapshot_digest": snapshot.numerical_digest, "compute_backend": "numpy", "kinetic_order": kinetic_order,
        "residual_compute_policy": residual_compute_backend,
        "residual_exponential_method": residual_exponential_method,
        "maximum_chebyshev_group": maximum_chebyshev_group,
        "residual_exponential_methods": sorted({row["residual_execution"]["exponential_method"] for row in records}),
        "residual_backends": sorted({row["residual_execution"]["compute_backend"] for row in records}),
        "field_sampling_backend": getattr(snapshot, "backend", sampling_backend) if sampler is None else sampler.backend,
        "energy_ev": float(energy_ev), "axial_nodes_m": nodes.tolist(), "axial_steps": len(nodes)-1,
        "electric_reference": None if electric_reference is None else electric_reference.to_dict(),
        "electric_forward_error_bound": math.expm1(math.fsum(math.log1p(x) for x in electric_errors)),
        "retarded_steps": retarded_steps,
        "input_flux": initial, "output_flux": result.probability, "normalization_applied": False,
        "support_ledger": None if support_ledger is None else support_ledger.to_dict(),
        **({"representation_estimates": representation_estimates.to_dict(),
            "representation_estimate_scope": "exploratory numerical estimate; not a certified global error bound",
            "high_accuracy_eligible": False} if phase_preview else {}),
        "kinetic_phase_ledger": None if kinetic_phase_ledger is None else kinetic_phase_ledger.to_dict(),
        "kinetic_budget_scope": "accepted residual bounds accumulated without resetting the global phase budget"
            if kinetic_phase_ledger is not None else "per-step uniform allocation within this kernel call",
        "support_projection_policy": support_projection_policy,
        "support_projection_trigger": ("first_residual_at_segment_start_once_or_typed_resource_failure"
            if support_projection_policy in ("preview_segment_and_retry", "preview_phase_segment_and_retry") else "typed_resource_failure_only" if weighted_resampling
            else "historical_policy_timing"),
        "entrance_boundary_projection": "explicit ledger only, after lossless Fourier preparation; original entrance edges must pass",
        "reference_quadrature": reference_quadrature,
        "axial_order_scope": "CF4 is fourth order for the quadratic reference only; general nonquadratic residual splitting remains second order"
            if reference_quadrature == "cf4" else "second-order midpoint reference and residual splitting",
        "new_support_event_count": 0 if support_ledger is None else len(support_ledger.events)-initial_support_count,
        "support_projection_scope": "explicit numerical loss; projection bounds exclude subsequent different-domain propagation error",
        "series_amplitude_error_bound": series,
        "kinetic_remainder_phase_bound": math.fsum(row["kinetic_remainder_phase_bound"] for row in records),
        "maximum_boundary_flux": maximum_boundary, "residual_steps": records, "canonical_sampling": charts,
        "envelope_spectral_guard": {"Nyquist_fraction": .8, "probability_tail_limit": envelope_limit,
            "frequency_filtering_applied": bool(support_ledger is not None and any(
                event["kind"] == "numerical_spectral_projection" for event in support_ledger.events[initial_support_count:])),
            "checked": "initial, each residual substep and canonical output"},
        "longitudinal_carrier_in_amplitude": False, "phase_carrier_expanded": False,
        "exact_carrier_reselection": reselect_carrier,
        "exact_scalar_potential_strang": scalar_splitting,
        "exact_quartic_scalar_strang": quartic_scalar_splitting,
        "exact_higher_order_scalar_strang": higher_order_scalar_splitting,
        "higher_order_scalar_splitting_scope": ("retained pure U powers are exact multiplication; K R_n + L U^n retains all noncommuting differential terms; independent axial convergence required"
            if higher_order_scalar_splitting else "disabled"),
        "quartic_scalar_splitting_scope": ("explicit U^2 multiplication; L^2 + LU + UL remain differential; independent axial convergence required"
            if quartic_scalar_splitting else "disabled; the full squared kinetic operator remains differential"),
        "physical_phase_removed": False, "production_admission": False,
        "scope": ("static scalar electrostatic and magnetic positive-branch flux wave; reflected waves and spin excluded"
                  if electric_reference is not None else
                  "static constant-energy native analytical magnetic benchmark on resolved affine envelope grids"),
        "lattice_metric": "physical covectors = inverse(basis).T @ index covectors; full mixed derivatives retained",
        "interface_scope": "aligned compact normal-A faces only at declared nodes; known one-sided values for floating event slivers; tilted/interior faces rejected",
        "not_qualified": ["complete instrument", "general hard interfaces", "ill-conditioned affine grids", "turning/reflected branches",
            "spin and explicit quantum field corrections", "FEM", "potential quadrature aliasing",
            "physical-domain, transverse-grid and axial-splitting convergence", "longitudinal pulsed/time-sideband wave dynamics"],
        "operator_ordering": "K=Pi_perp^2+p_axis^2-p_local^2; retained powers use the complete same Hermitian K; all mixed differential terms retained"}
