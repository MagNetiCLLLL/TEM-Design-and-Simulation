"""Opt-in scalar forward magnetic propagation on a fixed Cartesian lattice.

The classical-symbol Hamiltonian is Pi_perp^2/(2p) + Pi_perp^4/(8p^3)
- q A_z after removing the constant -p carrier. Spin, explicit hbar-dependent
Dirac corrections, back reflection and electric acceleration are not included.
The amplitude is the flux-normalized forward beam amplitude, not a normalized
image: aperture losses are retained and no renormalization is performed.

Wilson links discretize covariant transverse derivatives. Longitudinal links
parallel-transport both half steps, so a static gauge is respected in z too.
Sources are summed before forming the squared covariant momentum; overlapping
fields therefore retain cross terms. Dirichlet edges replace periodic wrap.
Grid, axial splitting and field-model convergence remain independent checks.
"""
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.immutable_json import freeze_json, thaw_json
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.wave_magnetic_residual import _series_plan


@dataclass(frozen=True)
class PotentialWaveResult:
    amplitude: object
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: float
    metadata: object


def propagate_plane_wave_potential(snapshot, wave, z_m, *, energy_ev, **numerics):
    """Bridge an existing PlaneWave through the bounded benchmark operator.

    The existing carrier sampling guard must admit expansion on this fixed
    lattice. Tilt, curvature and scalar phase are included in the full complex
    amplitude; none is fitted away. This is not an adaptive-carrier full-column
    implementation. Sheared/rotated lattices require a separate metric operator.
    """
    from temsim.physics.multiplane_wave import PlaneWave
    if type(wave) is not PlaneWave:
        raise ValueError("Potential wave bridge requires an existing PlaneWave")
    basis = wave.basis_m
    if basis[0, 1] != 0. or basis[1, 0] != 0. or min(basis[0, 0], basis[1, 1]) <= 0:
        raise ValueError("Potential wave bridge requires a positive axis-aligned Cartesian lattice")
    energy = _positive(energy_ev, "Electron kinetic energy")*e
    momentum = math.sqrt(energy*(energy+2*m_e*c*c))/c
    wavelength = 2*math.pi*hbar/momentum
    ny, nx = wave.amplitude.shape
    x = wave.origin_m[0]+(np.arange(nx)-nx//2)*basis[0, 0]
    y = wave.origin_m[1]+(np.arange(ny)-ny//2)*basis[1, 1]
    result = propagate_static_magnetic_wave(snapshot, wave.full_amplitude(wavelength), x, y, z_m,
                                            energy_ev=energy_ev, **numerics)
    return (PlaneWave(result.amplitude, wave.basis_m, wave.origin_m), result.metadata)


def propagate_wave_mode_potential(snapshot, mode, z_m, **numerics):
    """Preserve the existing mode weight and separate axial phase reference.

    Unit-shape factorization transfers the remaining norm into the mode's
    absolute weight. It does not restore any stopped electrons. Longitudinal
    carrier action advances separately and is not multiplied into amplitude a
    second time. This bridge retains the fixed-grid operator's admission limits.
    """
    from temsim.physics.wave_flux import WaveMode
    if type(mode) is not WaveMode:
        raise ValueError("Potential mode bridge requires an existing WaveMode")
    plane, record = propagate_plane_wave_potential(snapshot, mode.plane, z_m,
                                                   energy_ev=mode.energy_kev*1e3, **numerics)
    norm = plane.probability
    shape = replace(plane, amplitude=plane.amplitude/math.sqrt(norm)) if norm else plane
    reference = mode.axial_reference
    if reference is not None:
        reference = reference.advance(record["axial_flight_time_increment_s"],
                                      record["longitudinal_action_increment_j_s"])
    output = replace(mode, plane=shape, weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                     axial_reference=reference)
    metadata = thaw_json(record)
    metadata.update(input_mode_weight=mode.weight_per_reference_electron,
                    output_mode_weight=output.weight_per_reference_electron,
                    unit_shape_factorization=True,
                    axial_reference=None if reference is None else asdict(reference))
    return output, freeze_json(metadata)


def _positive(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} real number")
    return float(value)


def _axis(values, name):
    values = np.asarray(values)
    if (values.dtype.kind not in "iuf" or values.ndim != 1 or len(values) < 4
            or not np.isfinite(values).all()):
        raise ValueError(f"{name} must contain at least four finite real coordinates")
    values = np.asarray(values, dtype=float)
    spacing = np.diff(values)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0], rtol=1e-11, atol=0):
        raise ValueError(f"{name} must be increasing and uniformly sampled")
    return np.frombuffer(values.tobytes(), dtype=float), float(spacing[0])


def _line_integral(snapshot, starts, ends, component):
    # Four-point Gaussian quadrature integrates the supported polynomial gauge
    # exactly. Native field variation still requires link/grid refinement.
    nodes, weights = np.polynomial.legendre.leggauss(4)
    delta = ends-starts
    result = np.zeros(starts.shape[:-1])
    for node, weight in zip(nodes, weights):
        positions = starts+(.5+.5*node)*delta
        result += .5*weight*snapshot.sample(positions)["A_t_m"][..., component]*delta[..., component]
    if not np.isfinite(result).all():
        raise ValueError("Magnetic link integral exceeds finite numerical range")
    return result


def _lattice_points(x_m, y_m, z_m):
    xx, yy = np.meshgrid(x_m, y_m)
    return np.stack((xx, yy, np.full_like(xx, z_m)), axis=-1)


def _transverse_links(snapshot, points, xp):
    integral_x = _line_integral(snapshot, points[:, :-1], points[:, 1:], 0)
    integral_y = _line_integral(snapshot, points[:-1, :], points[1:, :], 1)
    # U_ij = exp(-i q integral_i^j A / hbar), q=-e.
    return xp.asarray(np.exp(1j*(e/hbar)*integral_x)), xp.asarray(np.exp(1j*(e/hbar)*integral_y))


def _covariant_laplacian(amplitude, links, dx, dy):
    ux, uy = links
    output = (2/dx**2+2/dy**2)*amplitude
    output[:, :-1] -= ux*amplitude[:, 1:]/dx**2
    output[:, 1:] -= ux.conj()*amplitude[:, :-1]/dx**2
    output[:-1, :] -= uy*amplitude[1:, :]/dy**2
    output[1:, :] -= uy.conj()*amplitude[:-1, :]/dy**2
    return output


def _boundary_probability(amplitude, xp):
    # A two-cell guard diagnoses occupied edges; it is never removed from the
    # wave or interpreted as physical absorption.
    probability = xp.abs(amplitude)**2
    total = float(xp.sum(probability))
    interior = float(xp.sum(probability[2:-2, 2:-2]))
    return max(0., total-interior)


def propagate_static_magnetic_wave(snapshot, amplitude, x_m, y_m, z_m, *, energy_ev,
                                   kinetic_order=4, backend="numpy",
                                   exponential_error=1e-10, boundary_fraction_limit=1e-8,
                                   kinetic_remainder_phase_limit=.01, aperture_masks=None,
                                   cancelled=lambda: False):
    """Propagate a selected static magnetic domain without changing its fields.

    Numerical inputs are independent of particle counts. z_m supplies actual
    axial step nodes. x/y are the physical sample coordinates, not display
    pixels. amplitude samples include sqrt(pixel area), so sum(abs(psi)^2)
    measures forward flux/probability in the caller's original units.

    The kinetic truncation bound applies to the entire represented lattice
    spectrum. It does not certify the paraxial magnetic/electric approximation,
    field mesh, physical support, quadrature or axial splitting error. A boundary
    rejection requests a larger domain; no edge clipping or rescaling is used.
    """
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("Wave execution requires a supported immutable potential snapshot")
    if any(row.kind != "magnetic" for row in snapshot.contributions):
        raise ValueError("This static magnetic operator does not admit electric acceleration")
    for row in snapshot.contributions:
        meta = row.potential.to_metadata()
        if (meta.get("static") is False or meta.get("dynamic_drive_frozen", False)
                or meta.get("provenance", {}).get("dynamic_drive_frozen", False)
                or bool(getattr(getattr(row.potential, "field", None), "dynamic", False))):
            raise ValueError("A frozen dynamic drive is not a qualified static wave field")
    if type(kinetic_order) is not int or kinetic_order not in (2, 4):
        raise ValueError("kinetic_order must be 2 or 4")
    if backend not in {"numpy", "cupy"}:
        raise ValueError("Wave backend must be numpy or cupy")
    if backend == "cupy":
        import cupy as xp
        if xp.cuda.runtime.getDeviceCount() < 1:
            raise ValueError("CUDA wave backend has no available device")
    else:
        xp = np
    energy = _positive(energy_ev, "Electron kinetic energy")*e
    error = _positive(exponential_error, "Exponential error budget")
    boundary_limit = _positive(boundary_fraction_limit, "Boundary fraction limit", allow_zero=True)
    if boundary_limit >= 1:
        raise ValueError("Boundary fraction limit must be less than one")
    remainder_limit = _positive(kinetic_remainder_phase_limit, "Kinetic remainder phase limit")
    x, dx = _axis(x_m, "x_m")
    y, dy = _axis(y_m, "y_m")
    nodes = np.asarray(z_m)
    if (nodes.dtype.kind not in "iuf" or nodes.ndim != 1 or len(nodes) < 2
            or not np.isfinite(nodes).all() or not np.all(np.diff(nodes) > 0)):
        raise ValueError("z_m must contain finite increasing axial nodes")
    nodes = np.asarray(nodes, dtype=float)
    psi = xp.asarray(amplitude, dtype=xp.complex128).copy()
    if psi.shape != (len(y), len(x)) or not bool(xp.isfinite(psi).all()):
        raise ValueError("Wave amplitude must be finite and match the physical lattice")
    masks = {} if aperture_masks is None else dict(aperture_masks)
    for index, mask in masks.items():
        if type(index) is not int or index < 0 or index >= len(nodes):
            raise ValueError("Aperture mask must belong to an actual axial node")
        raw = xp.asarray(mask)
        if raw.dtype.kind != "b" or raw.shape != psi.shape:
            raise ValueError("Aperture masks must be boolean arrays matching the wave lattice")
        masks[index] = raw.copy()
    momentum = math.sqrt(energy*(energy+2*m_e*c*c))/c
    wavenumber = momentum/hbar
    laplacian_bound = 4/dx**2+4/dy**2
    ratio = laplacian_bound/wavenumber**2
    if not math.isfinite(ratio) or ratio >= .1:
        raise ValueError("Represented lattice exceeds the forward small-angle kinetic domain")
    operator_bound = laplacian_bound/(2*wavenumber)
    if kinetic_order == 4:
        operator_bound += laplacian_bound**2/(8*wavenumber**3)
        remainder_per_m = wavenumber*ratio**3/(16*(1-ratio)**2.5)
    else:
        remainder_per_m = wavenumber*ratio**2/(8*(1-ratio)**1.5)
    remainder_phase = float(nodes[-1]-nodes[0])*remainder_per_m
    if remainder_phase > remainder_limit:
        raise ValueError("Kinetic-order truncation exceeds its declared phase budget")
    before = float(xp.sum(abs(psi)**2))
    if not math.isfinite(before) or before <= 0:
        raise ValueError("Input wave must have finite positive flux")
    aperture_loss, series_log_bound, applications = 0., 0., 0
    maximum_boundary = 0.

    def check_boundary():
        nonlocal maximum_boundary
        edge = _boundary_probability(psi, xp)
        maximum_boundary = max(maximum_boundary, edge)
        if edge > before*boundary_limit:
            raise ValueError("Occupied Dirichlet boundary: enlarge the physical wave domain")

    def observe(index):
        nonlocal psi, aperture_loss
        # The complete sampled observation plane belongs to the physical
        # domain, including points not visited by quadrature link nodes.
        snapshot.sample(_lattice_points(x, y, nodes[index]))
        check_boundary()
        if index in masks:
            old = float(xp.sum(abs(psi)**2))
            psi = xp.where(masks[index], psi, 0.)
            aperture_loss += old-float(xp.sum(abs(psi)**2))

    observe(0)
    for index, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):
        if cancelled():
            raise InterruptedError("Potential wave propagation cancelled")
        middle, distance = .5*(lower+upper), upper-lower
        points = _lattice_points(x, y, middle)
        links = _transverse_links(snapshot, points, xp)
        def parallel_transport(start, stop):
            left, right = points.copy(), points.copy()
            left[..., 2], right[..., 2] = start, stop
            return xp.asarray(np.exp(-1j*(e/hbar)*_line_integral(snapshot, left, right, 2)))
        psi *= parallel_transport(lower, middle)
        def apply_h(value):
            laplacian = _covariant_laplacian(value, links, dx, dy)
            result = laplacian/(2*wavenumber)
            if kinetic_order == 4:
                result += _covariant_laplacian(laplacian, links, dx, dy)/(8*wavenumber**3)
            return result
        count, order, bound = _series_plan(distance*operator_bound, error/(len(nodes)-1))
        for _ in range(count):
            if cancelled():
                raise InterruptedError("Potential wave propagation cancelled")
            term, total = psi, psi.copy()
            for degree in range(1, order+1):
                term = (-1j*distance/(count*degree))*apply_h(term)
                total += term
            psi = total
            # A packet can reach a Dirichlet edge and reflect within one
            # externally requested step. Check every bounded exponential
            # substep, not only the final observation plane.
            check_boundary()
        applications += count*order
        series_log_bound = float(np.nextafter(series_log_bound+math.log1p(bound), np.inf))
        psi *= parallel_transport(middle, upper)
        observe(index+1)
    after = float(xp.sum(abs(psi)**2))
    series_bound = float(np.nextafter(math.expm1(series_log_bound), np.inf))
    roundoff = 128*np.finfo(float).eps*max(1, applications)
    if not math.isfinite(after) or abs(after+aperture_loss-before) > before*(2*series_bound+series_bound**2+roundoff):
        raise ValueError("Lossless propagation/current accounting failed; no normalization was applied")
    metadata = freeze_json({"method": "static magnetic Wilson-link forward scalar operator",
        "snapshot_digest": snapshot.numerical_digest, "backend": backend, "kinetic_order": kinetic_order,
        "energy_ev": float(energy_ev), "input_flux": before, "output_flux": after,
        "aperture_loss": aperture_loss, "maximum_boundary_flux": maximum_boundary,
        "kinetic_remainder_phase_bound": remainder_phase, "series_amplitude_error_bound": series_bound,
        "roundoff_relative_allowance": roundoff, "operator_applications": applications,
        "axial_steps": len(nodes)-1, "pixel_pitch_m": (dx, dy),
        "longitudinal_action_increment_j_s": momentum*float(nodes[-1]-nodes[0]),
        "axial_flight_time_increment_s": float(nodes[-1]-nodes[0])*(energy+m_e*c*c)/(momentum*c*c),
        "longitudinal_carrier_in_amplitude": False,
        "wave_scope": "static selected magnetic domain; spinless forward classical-symbol approximation",
        "excluded": ("electric acceleration", "spin", "explicit quantum field corrections",
                     "back reflection", "automatic physical-domain or mesh qualification"),
        "normalization_applied": False, "production_admission": False})
    if xp is np:
        psi = np.frombuffer(psi.tobytes(), dtype=psi.dtype).reshape(psi.shape)
    return PotentialWaveResult(psi, x, y, float(nodes[-1]), metadata)
