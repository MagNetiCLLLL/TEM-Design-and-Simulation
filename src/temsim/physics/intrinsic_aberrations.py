"""Additional, user-declared intrinsic round-lens aberrations.

This is an equivalent thin action at the lens reference plane, not a new
magnetic field or a fit to the installed corrector.  S(q)=-W(q/f), with W in
the shared aberration basis, generates both ray momentum and wave phase.
Legacy Cs remains owned by the existing spherical operator; this module
never includes C3 or a second copy of alignment/multipole aberrations.

Tilt uses the same first-inclination Hamiltonian convention as posed Cs:
S=(p/p_ref) S0 + (d grad S0).(u+e A/p_ref). The momentum term is Weyl ordered.
Its bounded small-inclination domain is an approximation, not an exact
oblique wave solution. Both representations reject unsupported geometry.
"""
from dataclasses import dataclass, replace
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.optics.intrinsic_aberrations import INTRINSIC_TERMS
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.wave_device import array_module, max_abs


MODEL_VERSION = "intrinsic-polynomial-canonical-screen-v1"
TERMS = INTRINSIC_TERMS


def _coefficients(values):
    from temsim.optics.intrinsic_aberrations import validated_coefficients
    return tuple(sorted(validated_coefficients(dict(values)).items()))


def _local_terms(coefficients, focal_m):
    """Expand r**(p-m) Re[(x+iy)**m exp(-im phi)] exactly."""
    values, result = dict(coefficients), {}
    for term in TERMS:
        magnitude = values.get(term.field, 0.)
        if not magnitude:
            continue
        power, harmonic = term.power, term.harmonic
        radial = (power-harmonic)//2
        angle = math.radians(values.get(term.azimuth_field, 0.))*harmonic if harmonic else 0.
        scale = -magnitude*1e-3/(power*focal_m**power)
        for j in range(radial+1):
            for k in range(harmonic+1):
                iy = 2*j+k
                value = (scale*math.comb(radial, j)*math.comb(harmonic, k)
                         *((1j**k)*np.exp(-1j*angle)).real)
                index = power-iy, iy
                result[index] = result.get(index, 0.)+value
    return tuple((i, j, float(value)) for (i, j), value in sorted(result.items()) if value)


def _transform_terms(terms, matrix):
    """Substitute local_xy=matrix @ laboratory_xy without fitted coefficients."""
    result = {}
    for ix, iy, value in terms:
        for jx in range(ix+1):
            for jy in range(iy+1):
                j = jx+jy
                coefficient = (value*math.comb(ix, jx)*math.comb(iy, jy)
                    *matrix[0, 0]**(ix-jx)*matrix[0, 1]**jx
                    *matrix[1, 0]**(iy-jy)*matrix[1, 1]**jy)
                index = ix+iy-j, j
                result[index] = result.get(index, 0.)+coefficient
    return tuple((i, j, float(v)) for (i, j), v in sorted(result.items()) if v)


def polynomial_jet(terms, x, y, *, xp=np):
    """S, exact dS/dq and Hessian; derivative axes are the final axes."""
    x, y = xp.broadcast_arrays(xp.asarray(x), xp.asarray(y))
    value = xp.zeros_like(x, dtype=xp.float64)
    gradient = xp.zeros((*x.shape, 2), dtype=xp.float64)
    hessian = xp.zeros((*x.shape, 2, 2), dtype=xp.float64)
    for ix, iy, coefficient in terms:
        value += coefficient*x**ix*y**iy
        if ix:
            gradient[..., 0] += coefficient*ix*x**(ix-1)*y**iy
        if iy:
            gradient[..., 1] += coefficient*iy*x**ix*y**(iy-1)
        if ix > 1:
            hessian[..., 0, 0] += coefficient*ix*(ix-1)*x**(ix-2)*y**iy
        if iy > 1:
            hessian[..., 1, 1] += coefficient*iy*(iy-1)*x**ix*y**(iy-2)
        if ix and iy:
            cross = coefficient*ix*iy*x**(ix-1)*y**(iy-1)
            hessian[..., 0, 1] += cross
            hessian[..., 1, 0] += cross
    return value, gradient, hessian


@dataclass(frozen=True, slots=True)
class FrozenIntrinsicAberrationKick:
    lens_key: str
    local_z_m: float
    focal_m: float
    coefficients: tuple[tuple[str, float], ...]
    registration: CoordinateRegistration = CoordinateRegistration()
    chromatic_reference_energy_ev: float | None = None

    def __post_init__(self):
        if not np.isfinite((self.local_z_m, self.focal_m)).all() or self.focal_m <= 0:
            raise ValueError("Intrinsic aberration plane and positive focal length must be finite")
        object.__setattr__(self, "coefficients", _coefficients(self.coefficients))
        energy = self.chromatic_reference_energy_ev
        if energy is not None:
            if isinstance(energy, (bool, str)) or not math.isfinite(energy) or energy <= 0:
                raise ValueError("Chromatic reference kinetic energy must be finite and positive")
            if any(value for name, value in self.coefficients if name != "c1_mm"):
                raise ValueError("A chromatic screen must contain only its equivalent C1 coefficient")

    def energy_scale(self, momentum, *, xp=np, mass=m_e):
        """Additional Cc action, using the executing solver's mass convention.

        The field solver already owns its momentum-dependent focusing. This
        screen adds only the explicitly declared residual focal perturbation.
        """
        if self.chromatic_reference_energy_ev is None:
            return xp.ones_like(momentum, dtype=xp.float64)
        pc = xp.asarray(momentum)*c
        rest = mass*c*c
        kinetic_ev = pc*pc/(xp.hypot(pc, rest)+rest)/e
        delta = (kinetic_ev-self.chromatic_reference_energy_ev)/self.chromatic_reference_energy_ev
        # Suppress only roundoff in the inverse relativistic conversion.
        return xp.where(abs(delta) <= 8*np.finfo(float).eps, 0., delta)

    @property
    def z_mm(self):
        return float((self.registration.rotation_array @ np.array((0., 0., self.local_z_m))
                      +self.registration.origin_array_m)[2]*1e3)

    @property
    def active(self):
        values = dict(self.coefficients)
        return any(values.get(term.field, 0.) for term in TERMS)

    def geometry(self):
        from temsim.physics.posed_aberration_wave import MAXIMUM_INCLINATION_RAD
        rotation = self.registration.rotation_array
        normal = rotation[:, 2]
        inclination = math.atan2(float(np.linalg.norm(normal[:2])), float(normal[2]))
        if inclination > MAXIMUM_INCLINATION_RAD:
            raise ValueError("Intrinsic screen exceeds the 10 mrad small-inclination model domain")
        centre = (rotation@np.array((0., 0., self.local_z_m))+self.registration.origin_array_m)[:2]
        inclination_vector = normal[:2]/normal[2]
        # Exact projection onto the tilted plane along laboratory +Z;
        # the additional slope dependence is the bounded inclination action.
        matrix = rotation[:2, :2].T-np.outer(rotation[2, :2], inclination_vector)
        terms = _transform_terms(_local_terms(self.coefficients, self.focal_m), matrix)
        return centre, inclination_vector, inclination, terms

    def apply_canonical(self, x, tx, y, ty, **kwargs):
        return apply_intrinsic_rays(self, x, tx, y, ty, **kwargs)


def capture_intrinsic_kicks(state, z0, z1):
    """Capture immutable additional screens, never legacy Cs or field-map Cs."""
    from temsim.simulation_modes import is_thin
    if is_thin(state):
        return ()
    from temsim.lens_pose import lens_pose_registration
    from temsim.optics.lens_focal_length import focal_length_mm
    from temsim.physics.lens_field_provider import active_mapped_providers
    definitions = getattr(state, "lens_intrinsic_aberrations", {})
    if not definitions:
        return ()
    mapped = {provider.lens_key for provider in active_mapped_providers(state)}
    result = []
    for lens in state.lenses:
        row = definitions.get(lens.key, {})
        if not row or not lens.enabled or lens.key in mapped:
            continue
        coefficients = _coefficients(row)
        if not any(dict(coefficients).get(term.field, 0.) for term in TERMS):
            continue
        focal_m = float(focal_length_mm(lens, state.beam_voltage_kv))*1e-3
        if not math.isfinite(focal_m) or focal_m <= 0:
            raise ValueError(f"{lens.key}: nonzero intrinsic aberrations require an active finite-focus lens")
        kick = FrozenIntrinsicAberrationKick(lens.key, float(lens.z_mm)*1e-3,
            focal_m, coefficients, lens_pose_registration(state, lens.key))
        if z0-1e-9 <= kick.z_mm <= z1+1e-9:
            result.append(kick)
    return tuple(result)


def capture_chromatic_kicks(state, z0, z1):
    """Capture declared additional Cc once, sharing ray and wave ownership.

    A missing Cc remains inactive except for the historical objective 2 mm
    fallback. Spatial field maps own their chromatic response and are excluded.
    No coefficient is inferred from, or added to, a field-derived Cc estimate.
    """
    from temsim.simulation_modes import is_thin
    if is_thin(state):
        return ()
    from temsim.lens_pose import lens_pose_registration
    from temsim.optics.lens_focal_length import focal_length_mm
    from temsim.physics.lens_field_provider import active_mapped_providers
    if not bool(getattr(state, "chromatic_aberration_enabled", False)):
        return ()
    mapped = {provider.lens_key for provider in active_mapped_providers(state)}
    result = []
    for lens in state.lenses:
        if not lens.enabled or lens.key in mapped:
            continue
        cc = getattr(lens, "cc_mm", None)
        if cc is None:
            cc = 2. if lens.key == "objective_lens" else 0.
        if isinstance(cc, (bool, str)) or not math.isfinite(cc) or cc < 0:
            raise ValueError(f"{lens.key}: Cc must be finite and nonnegative")
        if not cc:
            continue
        if float(getattr(lens, "percent", 100.)) == 0.:
            continue
        focal = float(focal_length_mm(lens, state.beam_voltage_kv))*1e-3
        if not math.isfinite(focal) or focal <= 0:
            raise ValueError(f"{lens.key}: nonzero Cc requires an active finite-focus lens")
        kick = FrozenIntrinsicAberrationKick(lens.key, float(lens.z_mm)*1e-3,
            focal, (("c1_mm", -float(cc)),), lens_pose_registration(state, lens.key),
            float(state.beam_voltage_kv)*1000.)
        # Use the event-index tolerance at interval endpoints as well:
        # mm -> m -> mm can round a coincident specimen plane by one ulp.
        if z0-1e-9 <= kick.z_mm <= z1+1e-9:
            result.append(kick)
    return tuple(result)


def specimen_chromatic_energy_change(state, x_m, y_m, incoming_energy_ev, outgoing_energy_ev,
                                    *, active_mask=None):
    """Particle-only incremental Cc at a coincident material reference plane.

    The incident trace has already executed Cc at its incoming energy. For
    material reinjection add only the energy change, never its full offset
    from nominal again. Finite material is matched to this same reference
    plane: this is the existing equivalent thin-material approximation, not
    a distributed chromatic treatment inside the specimen. Tilted events are
    rejected because their energy-switch canonical bridge is not implemented.
    """
    x, y = np.broadcast_arrays(np.asarray(x_m, float), np.asarray(y_m, float))
    dx, dy = np.zeros_like(x), np.zeros_like(y)
    if not bool(getattr(state, "chromatic_aberration_enabled", False)):
        return dx, dy
    z = float(state.sample.z_mm)
    kicks = capture_chromatic_kicks(state, z-1e-9, z+1e-9)
    if not kicks:
        return dx, dy
    active = (np.isfinite(x)&np.isfinite(y) if active_mask is None else
              np.broadcast_to(np.asarray(active_mask, bool), x.shape))
    if not np.any(active):
        return dx, dy
    incoming = np.broadcast_to(np.asarray(incoming_energy_ev, float), x.shape)[active]
    outgoing = np.broadcast_to(np.asarray(outgoing_energy_ev, float), x.shape)[active]
    if (not np.isfinite(incoming).all() or not np.isfinite(outgoing).all()
            or np.any(incoming <= 0) or np.any(outgoing <= 0)
            or not np.isfinite(x[active]).all() or not np.isfinite(y[active]).all()):
        raise ValueError("Specimen chromatic update requires known positive incoming and outgoing energies")
    change = outgoing-incoming
    if not np.any(change):
        return dx, dy
    for kick in kicks:
        centre, inclination, _, terms = kick.geometry()
        if np.any(inclination):
            raise ValueError("A tilted Cc screen at the specimen cannot yet bridge a material energy change")
        _, gradient, _ = polynomial_jet(terms, x[active]-centre[0], y[active]-centre[1])
        scale = change/kick.chromatic_reference_energy_ev
        dx[active] += scale*gradient[..., 0]
        dy[active] += scale*gradient[..., 1]
    return dx, dy


def _absolute_bounds(terms, radius):
    action = gradient = hessian = 0.
    for ix, iy, value in terms:
        degree = ix+iy
        action += abs(value)*radius**degree
        gradient += abs(value)*degree*radius**(degree-1)
        if degree > 1:
            hessian += abs(value)*degree*(degree-1)*radius**(degree-2)
    return action, gradient, hessian


def _domain(terms, b, radius, xy_bound, theta, wavelength, p, pref, fields,
            aligned_bz_t, z_m, budget, electric_gradient, electric_hessian):
    from temsim.physics.posed_aberration_wave import _field_bounds, MAXIMUM_MECHANICAL_SLOPE
    bnorm = float(np.linalg.norm(b))
    if not np.isfinite(theta) or theta > MAXIMUM_MECHANICAL_SLOPE:
        raise ValueError("Intrinsic aberration screen exceeds the 0.1 mechanical-slope domain")
    radius = max(float(radius), 1e-30)
    expanded = radius*1.01
    for _ in range(16):
        _, gradient, _ = _absolute_bounds(terms, expanded)
        displacement = bnorm*expanded*gradient
        if displacement <= expanded-radius:
            break
        expanded = radius+1.1*displacement
        if not math.isfinite(expanded) or expanded > 2*radius:
            raise ValueError("Intrinsic inclination action cannot establish bounded screen support")
    else:
        raise ValueError("Intrinsic inclination action cannot establish bounded screen support")
    _, gradient, _ = _absolute_bounds(terms, expanded)
    theta += gradient
    if theta > MAXIMUM_MECHANICAL_SLOPE or bnorm*theta >= .05:
        raise ValueError("Intrinsic aberration screen exceeds the 0.1 mechanical-slope domain")
    depth = bnorm*expanded/(1-bnorm*theta)
    hit_radius = expanded+depth*theta
    action, gradient, hessian = _absolute_bounds(terms, hit_radius)
    _, magnetic = _field_bounds(fields, aligned_bz_t, xy_bound+(expanded-radius)+depth*theta, z_m, depth)
    # Exact straight intersection minus its first slope term, followed by
    # the second-order Taylor remainder of this finite polynomial.
    geometric_action = (gradient*bnorm*expanded*theta*(bnorm*theta)/(1-bnorm*theta)
                        +.5*hessian*(depth*theta)**2)
    magnetic_shift = .5*e*magnetic/p*(1+theta*theta)**1.5*depth*depth
    magnetic_action = _absolute_bounds(terms, hit_radius+magnetic_shift)[1]*magnetic_shift
    eg = np.zeros(3) if electric_gradient is None else np.asarray(electric_gradient, float)
    eh = np.zeros((3, 3)) if electric_hessian is None else np.asarray(electric_hessian, float)
    if eh.shape == (2, 2):
        expanded_hessian = np.zeros((3, 3)); expanded_hessian[:2, :2] = eh; eh = expanded_hessian
    if (eg.shape != (3,) or eh.shape != (3, 3) or not np.isfinite(eg).all()
            or not np.isfinite(eh).all() or not np.allclose(eh, eh.T, rtol=1e-10, atol=1e-12)):
        raise ValueError("Intrinsic screen requires finite symmetric local electric derivatives")
    extent = np.r_[xy_bound+expanded-radius+depth*theta, depth]
    delta_v = float(abs(eg)@extent+.5*extent@abs(eh)@extent)
    energy = math.hypot(p*c, m_e*c*c)
    kinetic = (p*c)**2/(energy+m_e*c*c)
    delta_energy = e*delta_v
    if delta_energy >= kinetic:
        raise ValueError("Intrinsic screen electric variation prevents positive kinetic energy")
    minus = delta_energy*(2*energy-delta_energy)/(p*c)**2
    plus = delta_energy*(2*energy+delta_energy)/(p*c)**2
    epsilon = max(plus/(math.sqrt(1+plus)+1), minus/(1+math.sqrt(1-minus)))
    pmin = p*math.sqrt(1-minus)
    vmin = pmin*c*c/(energy-delta_energy)
    electric_field = float(np.linalg.norm(abs(eg)+abs(eh)@extent))
    electric_shift = depth*theta*epsilon/(1-epsilon)+.5*e*electric_field/(pmin*vmin)*(1+theta*theta)**1.5*depth*depth
    electric_action = epsilon*action+(1+epsilon)*_absolute_bounds(terms, hit_radius+electric_shift)[1]*electric_shift
    factor = (2*math.pi/wavelength)*(p/pref)
    bounds = {"geometric_phase_remainder_bound_rad": factor*geometric_action,
        "magnetic_bridge_phase_bound_rad": factor*magnetic_action,
        "electric_phase_remainder_bound_rad": factor*electric_action,
        "maximum_bridge_depth_m": depth, "maximum_mechanical_slope_bound": theta,
        "electric_relative_momentum_variation_bound": epsilon}
    if factor*(geometric_action+magnetic_action+electric_action) > budget:
        raise ValueError("Intrinsic first-inclination phase remainder exceeds its declared event budget")
    return bounds


def _translated_terms(terms, centre):
    result = {}
    for ix, iy, value in terms:
        for a in range(ix+1):
            for b in range(iy+1):
                coefficient = value*math.comb(ix, a)*math.comb(iy, b)*centre[0]**(ix-a)*centre[1]**(iy-b)
                result[a, b] = result.get((a, b), 0.)+coefficient
    return tuple((a, b, value) for (a, b), value in sorted(result.items()) if value)


def _phase(wave, wavelength, terms, centre, scale):
    from temsim.physics.wave_flux import check_lossless_norm
    from temsim.physics.wave_grid import WaveSamplingError
    xp = array_module(wave.amplitude)
    translated = _translated_terms(terms, wave.origin_m-centre)
    constant, gradient, hessian = polynomial_jet(translated, 0., 0.)
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    if not any(a+b > 2 for a, b, _ in translated):
        # Defocus and twofold astigmatism live exactly in the analytic
        # carrier; neither a physical grid refinement nor an FFT is needed.
        return replace(wave,
            amplitude=wave.amplitude*xp.exp(2j*math.pi*scale*float(constant)/wavelength),
            curvature_m1=curvature+scale*hessian, tilt_rad=tilt+scale*gradient)
    xy = wave.coordinates_m()-xp.asarray(wave.origin_m)[:, None, None]
    residual = xp.zeros_like(wave.amplitude.real)
    for a, b, value in translated:
        if a+b > 2:
            residual += value*xy[0]**a*xy[1]**b
    phase = (2*math.pi*scale/wavelength)*residual
    occupied = abs(wave.amplitude) > max_abs(wave.amplitude)*1e-8
    spectrum = abs(xp.fft.fft2(wave.amplitude, norm="ortho"))**2
    total, required, largest = float(spectrum.sum()), 1., 0.
    for axis in (0, 1):
        n = wave.amplitude.shape[axis]
        mask = xp.take(occupied, xp.arange(n-1), axis=axis)&xp.take(occupied, xp.arange(1, n), axis=axis)
        added = max_abs(abs(xp.diff(phase, axis=axis))[mask])
        frequencies = abs(2*math.pi*xp.fft.fftfreq(n))
        marginal = spectrum.sum(axis=1-axis)
        order = xp.argsort(frequencies)
        tail = xp.cumsum(marginal[order][::-1])[::-1]
        bandwidth = max_abs(frequencies[order][tail > total*1e-12])
        largest = max(largest, added+bandwidth)
        required = max(required, (added+bandwidth)/(.8*math.pi))
    if required > 1:
        raise WaveSamplingError(f"Intrinsic column phase is undersampled ({largest:.6g} rad); refine the physical wave grid", required)
    result = replace(wave, amplitude=wave.amplitude*xp.exp(1j*(phase+2*math.pi*scale*float(constant)/wavelength)),
                     curvature_m1=curvature+scale*hessian, tilt_rad=tilt+scale*gradient)
    check_lossless_norm(wave.probability, result.probability, context="Intrinsic lens phase")
    return result


def apply_intrinsic_wave(wave, wavelength, kick, *, momentum_kg_m_s,
        reference_momentum_kg_m_s, fields=(), aligned_bz_t=0., error_budget=1e-10,
        maximum_obliquity_phase_error_rad=.01, electrostatic_gradient_v_m=None,
        electrostatic_hessian_v_m2=None, cancelled=lambda: False):
    from temsim.physics.posed_aberration_wave import _supported, _potential, _field_bounds
    from temsim.physics.wave_magnetic_residual import apply_magnetic_residual
    from temsim.physics.wave_grid import WaveSamplingError
    p, pref = float(momentum_kg_m_s), float(reference_momentum_kg_m_s)
    if not np.isfinite((p, pref, wavelength, error_budget, maximum_obliquity_phase_error_rad)).all() or min(p, pref, wavelength, error_budget, maximum_obliquity_phase_error_rad) <= 0:
        raise ValueError("Intrinsic screen requires finite positive momenta, wavelength and budgets")
    if cancelled():
        raise InterruptedError("Intrinsic lens screen cancelled")
    record = {"model": MODEL_VERSION, "lens_key": kick.lens_key, "z_mm": kick.z_mm,
        "ownership": "additional intrinsic coefficients only; legacy Cs and physical fields executed separately",
        "input_probability": wave.probability}
    if kick.chromatic_reference_energy_ev is not None:
        scale = float(kick.energy_scale(p))
        record.update(model="additional-chromatic-canonical-screen-v1",
            ownership="declared additional Cc; field momentum dependence remains separate",
            cc_mm=-dict(kick.coefficients)["c1_mm"], relative_energy_offset=scale,
            chromatic_reference_energy_ev=kick.chromatic_reference_energy_ev)
        kick = replace(kick, coefficients=tuple((name, value*scale) for name, value in kick.coefficients),
                       chromatic_reference_energy_ev=None)
    if not kick.active:
        return wave, {**record, "method": "zero intrinsic identity", "output_probability": wave.probability, "split_steps": 0}
    centre, b, inclination, terms = kick.geometry()
    record.update(inclination_rad=inclination, coefficients=dict(kick.coefficients))
    if not np.any(b):
        result = _phase(wave, wavelength, terms, centre, p/pref)
        return result, {**record, "method": "intrinsic polynomial phase", "output_probability": result.probability, "split_steps": 0}
    if not _supported(fields):
        raise ValueError("Tilted intrinsic screens require captured analytic vector potentials")
    xp = array_module(wave.amplitude)
    ny, nx = wave.amplitude.shape
    extent = abs(wave.basis_m)@np.array((nx//2, ny//2))
    radius = float(np.linalg.norm(abs(wave.origin_m-centre)+extent))
    xy_bound = abs(wave.origin_m)+extent
    curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    canonical = abs(tilt)+abs(curvature)@extent+.5*wavelength*abs(np.linalg.inv(wave.basis_m).T)@np.ones(2)
    a_bound, _ = _field_bounds(fields, aligned_bz_t, xy_bound, kick.z_mm*1e-3, 0.)
    theta = (pref*float(np.linalg.norm(canonical))+e*a_bound)/p
    bounds = _domain(terms, b, radius, xy_bound, theta, wavelength, p, pref, fields, aligned_bz_t,
        kick.z_mm*1e-3, maximum_obliquity_phase_error_rad, electrostatic_gradient_v_m, electrostatic_hessian_v_m2)
    xy = wave.coordinates_m()
    q = xy-xp.asarray(centre)[:, None, None]
    _, gradient, _ = polynomial_jet(terms, q[0], q[1], xp=xp)
    gradient = xp.moveaxis(gradient, -1, 0)
    depth = -xp.einsum("i,iyx->yx", xp.asarray(b), q)
    velocity = depth[None]*gradient
    displacement = float(xp.max(abs(xp.einsum("ij,jyx->iyx", xp.asarray(np.linalg.inv(wave.basis_m)), velocity))))
    if displacement > .125*min(ny, nx):
        raise ValueError("Intrinsic inclination displacement exceeds the physical wave-grid domain")
    points = xp.moveaxis(xp.concatenate((xy, xp.full((1, ny, nx), kick.z_mm*1e-3))), 0, -1)
    magnetic_a = xp.moveaxis(_potential(points, fields, aligned_bz_t, xp=xp)[..., :2], -1, 0)
    local = xy-xp.asarray(wave.origin_m)[:, None, None]
    def evolve(steps):
        value, receipts = wave, []
        for _ in range(steps):
            if cancelled():
                raise InterruptedError("Intrinsic lens screen cancelled")
            value = _phase(value, wavelength, terms, centre, p/pref/(2*steps))
            curve = np.zeros((2, 2)) if value.curvature_m1 is None else value.curvature_m1
            carrier_tilt = np.zeros(2) if value.tilt_rad is None else value.tilt_rad
            carrier = xp.asarray(carrier_tilt)[:, None, None]+xp.einsum("ij,jyx->iyx", xp.asarray(curve), local)
            scalar = xp.sum(velocity*(carrier+e/pref*magnetic_a), axis=0)
            phase = 2*math.pi/wavelength/steps*scalar
            increment = max(float(xp.max(abs(xp.diff(phase, axis=axis)))) for axis in (0, 1))
            if increment > .2*math.pi:
                raise WaveSamplingError("Intrinsic inclination action exceeds the phase sampling guard", increment/(.2*math.pi))
            value, receipt = apply_magnetic_residual(value, -velocity, -scalar, wavelength, 1./steps,
                error_budget=error_budget/(16*steps), cancelled=cancelled)
            receipts.append(receipt)
            value = _phase(value, wavelength, terms, centre, p/pref/(2*steps))
        return value, receipts
    previous, _ = evolve(1)
    for steps in (2, 4, 8, 16, 32):
        current, receipts = evolve(steps)
        error = float(xp.linalg.norm(current.amplitude-previous.amplitude))/max(math.sqrt(wave.probability), np.finfo(float).tiny)/3
        if error <= error_budget*.5:
            return current, {**record, **bounds, "method": "polynomial phase / Hermitian inclination / polynomial phase",
                "compute_backend": "numpy" if xp is np else "cupy", "output_probability": current.probability,
                "split_steps": steps, "split_error_estimate": error,
                "series_error_bound": sum(row["series_error_bound"] for row in receipts),
                "maximum_lattice_displacement": displacement,
                "model_scope": "bounded first-inclination canonical action, not an exact oblique wave solver"}
        previous = current
    raise ValueError("Intrinsic inclination split refinement did not satisfy its numerical budget")


def apply_intrinsic_rays(kick, x, tx, y, ty, *, momentum_kg_m_s,
        reference_momentum_kg_m_s=None, fields=(), aligned_bz_t=0.,
        maximum_obliquity_phase_error_rad=.01, electrostatic_gradient_v_m=None,
        electrostatic_hessian_v_m2=None):
    """Apply the shared canonical action on the arrays' existing device.

    NumPy inputs preserve the CPU API. CUDA arrays stay on device through
    the polynomial, magnetic gauge and RK flow; only scalar/two-coordinate
    domain bounds are read on the host for the common physical admission.
    This function never selects a device independently of its caller.
    """
    from temsim.physics.posed_aberration_wave import _supported, _potential
    if not kick.active:
        return x, tx, y, ty
    centre, b, _, terms = kick.geometry()
    inputs = (x, tx, y, ty, momentum_kg_m_s)
    xp = next((array_module(value) for value in (*inputs, reference_momentum_kg_m_s)
               if hasattr(value, "__cuda_array_interface__")), np)
    centre_device, b_device = xp.asarray(centre), xp.asarray(b)
    arrays = xp.broadcast_arrays(*(xp.asarray(value, dtype=xp.float64) for value in inputs))
    x, tx, y, ty, p = (a.copy() for a in arrays)
    valid = xp.isfinite(x)&xp.isfinite(tx)&xp.isfinite(y)&xp.isfinite(ty)&xp.isfinite(p)&(p > 0)
    if not bool(valid.any()):
        return x, tx, y, ty
    p = p[valid]
    # Ray solvers use the project's fixed CODATA mass. The coherent solver
    # uses scipy.constants and passes through the wave action above instead.
    scale = kick.energy_scale(p, xp=xp, mass=9.1093837015e-31)
    if not bool(xp.any(scale)):
        return x, tx, y, ty
    pref = p if reference_momentum_kg_m_s is None else xp.broadcast_to(
        xp.asarray(reference_momentum_kg_m_s, dtype=xp.float64), x.shape)[valid]
    if not bool(xp.isfinite(pref).all()) or bool(xp.any(pref <= 0)):
        raise ValueError("Intrinsic ray reference momentum must be positive and finite")
    q = xp.column_stack((x[valid], y[valid]))
    if not np.any(b):
        _, gradient, _ = polynomial_jet(terms, *(q-centre_device).T, xp=xp)
        tx[valid] += scale*gradient[:, 0]; ty[valid] += scale*gradient[:, 1]
        return x, tx, y, ty
    if not _supported(fields):
        raise ValueError("Tilted intrinsic screens require captured analytic vector potentials")
    radius = float(xp.max(xp.linalg.norm(q-centre_device, axis=1)))
    theta = float(xp.max(xp.hypot(tx[valid], ty[valid])))
    pmin, pmax = float(xp.min(p)), float(xp.max(p))
    xy_bound = xp.max(abs(q), axis=0)
    if xp is not np:
        xy_bound = xp.asnumpy(xy_bound)
    bounded_terms = tuple((i, j, value*float(xp.max(abs(scale)))) for i, j, value in terms)
    _domain(bounded_terms, b, radius, xy_bound, theta, 2*math.pi*hbar/pmax, pmin, pmin,
        fields, aligned_bz_t, kick.z_mm*1e-3, maximum_obliquity_phase_error_rad,
        electrostatic_gradient_v_m, electrostatic_hessian_v_m2)
    points = lambda pos: xp.column_stack((pos, xp.full(len(pos), kick.z_mm*1e-3)))
    a = _potential(points(q), fields, aligned_bz_t, xp=xp)[:, :2]
    u = (p[:, None]*xp.column_stack((tx[valid], ty[valid]))-e*a)/pref[:, None]
    def flow(pos, momentum):
        relative = pos-centre_device
        _, gradient, hessian = polynomial_jet(terms, *relative.T, xp=xp)
        gradient *= scale[:, None]
        hessian *= scale[:, None, None]
        depth = -relative@b_device
        velocity = depth[:, None]*gradient
        derivative = depth[:, None, None]*hessian-gradient[:, :, None]*b_device[None, None, :]
        a, da = _potential(points(pos), fields, aligned_bz_t, gradient=True, xp=xp)
        kinetic = momentum+e/pref[:, None]*a[:, :2]
        du = ((p/pref)[:, None]*gradient+xp.einsum("nij,ni->nj", derivative, kinetic)
              +e/pref[:, None]*xp.einsum("ni,nij->nj", velocity, da[:, :2, :2]))
        return -velocity, du
    def evolve(steps):
        pos, momentum = q.copy(), u.copy()
        h = 1./steps
        for _ in range(steps):
            a1, b1 = flow(pos, momentum)
            a2, b2 = flow(pos+h*a1/2, momentum+h*b1/2)
            a3, b3 = flow(pos+h*a2/2, momentum+h*b2/2)
            a4, b4 = flow(pos+h*a3, momentum+h*b3)
            pos += h*(a1+2*a2+2*a3+a4)/6
            momentum += h*(b1+2*b2+2*b3+b4)/6
        return pos, momentum
    old_q, old_u = evolve(4)
    for steps in (8, 16, 32):
        new_q, new_u = evolve(steps)
        if (float(xp.max(abs(new_q-old_q))) <= 1e-11*max(radius, 1e-15)
                and float(xp.max(abs(new_u-old_u))) <= 1e-11*max(float(xp.max(abs(new_u))), 1e-12)):
            break
        old_q, old_u = new_q, new_u
    else:
        raise ValueError("Intrinsic canonical ray flow did not satisfy its numerical budget")
    a = _potential(points(new_q), fields, aligned_bz_t, xp=xp)[:, :2]
    mechanical = (pref[:, None]*new_u+e*a)/p[:, None]
    x[valid], y[valid] = new_q.T
    tx[valid], ty[valid] = mechanical.T
    return x, tx, y, ty
