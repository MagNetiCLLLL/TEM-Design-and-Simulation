"""Opt-in exact quadratic reduction of a narrowly declared static field domain.

This replaces the second-order forward scalar Hamiltonian, not full Lorentz
or the fourth-order potential operator. Qualification is by native coefficient
identity, never a fit to rays, Cs or a sampled endpoint matrix. No production
plan selects this module automatically. Source, boundary and detector actions
remain outside the reduced interval.
"""
from dataclasses import dataclass, field
from itertools import product
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e
from scipy.linalg import expm

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.canonical_action import CanonicalPath
from temsim.physics.interaction_registry import InteractionRecord, InteractionRegistry
from temsim.physics.multiplane_wave import PlaneWave, propagate_plane_wave
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, _canonical_owner


_J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])


def _array(value, shape, name):
    value = np.asarray(value)
    if value.dtype.kind not in "iuf" or value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite real values with shape {shape}")
    return np.frombuffer(np.asarray(value, float).tobytes(), dtype=float).reshape(shape)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_, str)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f"{name} must be a positive finite real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite real scalar")
    return value


def _momentum(energy_ev):
    energy = energy_ev*e
    return math.sqrt(energy*(energy+2*m_e*c*c))/c


def _coefficients(snapshot):
    """Prove A_perp affine and Az quadratic, constant along global z."""
    constant, linear, az_hessian = np.zeros(3), np.zeros((3, 2)), np.zeros((2, 2))
    for row in snapshot.contributions:
        potential = row.potential
        if row.kind != "magnetic":
            raise ValueError("Quadratic reduction admits static magnetic contributions only")
        metadata = potential.to_metadata()
        if (metadata.get("static") is False or metadata.get("dynamic_drive_frozen", False)
                or metadata.get("provenance", {}).get("dynamic_drive_frozen", False)):
            raise ValueError("Frozen dynamic fields are not static quadratic qualifications")
        if type(potential) is AxisymmetricMagneticPotential:
            rotation = potential.registration.rotation_array
            if not np.array_equal(rotation[:, 2], [0., 0., 1.]):
                raise ValueError("Uniform axial reduction requires an aligned positive z axis")
            coefficient = potential.a_phi_tm[-1, 0]/potential.r_m[-1]
            expected = np.broadcast_to((coefficient*potential.r_m)[:, None], potential.a_phi_tm.shape)
            if not np.array_equal(potential.a_phi_tm, expected):
                raise ValueError("A_phi coefficients do not prove an exact uniform axial field")
            coefficient *= potential.source_scale
            matrix = coefficient*np.array(((0., -1.), (1., 0.)))
            linear[:2] += matrix
            constant[:2] -= matrix@potential.registration.origin_array_m[:2]
            continue
        base = potential.base if type(potential) is AnalyticPotentialDomain else potential
        if type(base) is not AnalyticMagneticPotential or type(base.field) is not FrozenPosedMultipole:
            raise ValueError("Source has no admitted exact quadratic coefficient proof")
        native = base.field
        if native.dynamic or base.provenance.get("dynamic_drive_frozen", False):
            raise ValueError("Frozen dynamic fields are not static quadratic qualifications")
        if native.envelope_kind != "uniform" or any(i+j > 2 for i, j, _ in native.polynomial_terms):
            raise ValueError("Quadratic reduction requires a uniform axial envelope and degree at most two")
        rotation = native.registration.rotation_array
        if not np.array_equal(rotation[:, 2], [0., 0., 1.]):
            raise ValueError("Tilted multipole boundaries require an unreduced field executor")
        gradient, hessian = np.zeros(2), np.zeros((2, 2))
        for i, j, value in native.polynomial_terms:
            if (i, j) == (1, 0): gradient[0] += value
            elif (i, j) == (0, 1): gradient[1] += value
            elif (i, j) == (2, 0): hessian[0, 0] += 2*value
            elif (i, j) == (0, 2): hessian[1, 1] += 2*value
            elif (i, j) == (1, 1): hessian[0, 1] += value; hessian[1, 0] += value
        rotation = rotation[:2, :2]
        offset = native.registration.origin_array_m[:2]
        gradient, hessian = rotation@gradient, rotation@hessian@rotation.T
        constant[2] += -gradient@offset+.5*offset@hessian@offset
        linear[2] += gradient-hessian@offset
        az_hessian += hessian
    gauge = snapshot.gauge
    if np.any(gauge.hessian_t[:, 2]):
        raise ValueError("This exact constant-generator reduction excludes z-dependent gauge coefficients")
    constant += gauge.gradient_t_m
    linear += gauge.hessian_t[:, :2]
    return constant, linear, az_hessian


@dataclass(frozen=True, slots=True, eq=False)
class QualifiedQuadraticReduction:
    snapshot: ElectromagneticPotentialSnapshot
    interval_m: tuple
    position_bounds_m: object
    energy_range_ev: tuple
    maximum_canonical_slope: float
    kinetic_phase_budget: float = .01
    boundary_planes_m: tuple = ()
    constant_a: np.ndarray = field(init=False, repr=False)
    linear_a: np.ndarray = field(init=False, repr=False)
    az_hessian: np.ndarray = field(init=False, repr=False)
    kinetic_remainder_phase_bound: float = field(init=False)
    digest: str = field(init=False)

    def __post_init__(self):
        if type(self.snapshot) is not ElectromagneticPotentialSnapshot:
            raise ValueError("Reduction requires an immutable potential snapshot")
        interval = _array(self.interval_m, (2,), "Reduced interval")
        energy = _array(self.energy_range_ev, (2,), "Energy range")
        bounds = _array(self.position_bounds_m, (2, 2), "Position bounds")
        if interval[1] <= interval[0] or energy[0] <= 0 or energy[1] < energy[0] or np.any(bounds[:, 1] <= bounds[:, 0]):
            raise ValueError("Reduction intervals, position bounds and energy range must be ordered")
        object.__setattr__(self, "interval_m", tuple(interval))
        object.__setattr__(self, "energy_range_ev", tuple(energy))
        object.__setattr__(self, "position_bounds_m", bounds)
        object.__setattr__(self, "maximum_canonical_slope", _positive(self.maximum_canonical_slope, "Canonical slope limit"))
        object.__setattr__(self, "kinetic_phase_budget", _positive(self.kinetic_phase_budget, "Kinetic phase budget"))
        planes = np.asarray(self.boundary_planes_m)
        if planes.dtype.kind not in "iuf" or planes.ndim != 1 or not np.isfinite(planes).all():
            raise ValueError("Boundary planes must be finite real axial positions")
        object.__setattr__(self, "boundary_planes_m", tuple(float(z) for z in planes))
        self._boundaries(planes)
        values = _coefficients(self.snapshot)
        for name, value, shape in zip(("constant_a", "linear_a", "az_hessian"), values, ((3,), (3, 2), (2, 2))):
            object.__setattr__(self, name, _array(value, shape, name))
        # Axial faces are interfaces even for an explicitly zero exterior.
        for row in self.snapshot.contributions:
            potential = row.potential
            base = potential.base if type(potential) is AnalyticPotentialDomain else potential
            if type(base) is AnalyticMagneticPotential:
                native = base.field
                lo, hi = np.asarray(native.native_support_mm)*1e-3+native.registration.origin_array_m[2]
                if not lo < interval[0] < interval[1] < hi:
                    raise ValueError("Reduction must stay strictly inside a single smooth field interval")
        corners = np.array([(x, y, z) for x, y, z in product(*bounds, interval)])
        self.snapshot.sample(corners)
        # A conservative physical forward-expansion omission budget, separate
        # from metaplectic quadrature, sampling and float64 roundoff.
        extent = np.max(abs(bounds), axis=1)
        max_a = abs(self.constant_a[:2])+abs(self.linear_a[:2])@extent
        slope = self.maximum_canonical_slope+e*max_a/_momentum(energy[0])
        ratio = float(slope@slope)
        if ratio >= .1:
            raise ValueError("Reduction exceeds the declared forward kinetic domain")
        remainder = _momentum(energy[1])/hbar*(interval[1]-interval[0])*ratio**2/(8*(1-ratio)**1.5)
        if remainder > self.kinetic_phase_budget:
            raise ValueError("Quadratic kinetic omission exceeds the declared phase budget")
        object.__setattr__(self, "kinetic_remainder_phase_bound", float(remainder))
        object.__setattr__(self, "digest", json_digest(self.metadata()))

    def _boundaries(self, planes):
        if any(self.interval_m[0] <= z <= self.interval_m[1] for z in planes):
            raise ValueError("A reduced interval cannot include an aperture or other boundary action")

    def metadata(self):
        return {"schema": "qualified-exact-quadratic-reduction-v1",
            "snapshot_numerical_digest": self.snapshot.numerical_digest,
            "interval_m": self.interval_m, "position_bounds_m": self.position_bounds_m.tolist(),
            "energy_range_ev": self.energy_range_ev, "maximum_canonical_slope": self.maximum_canonical_slope,
            "kinetic_phase_budget": self.kinetic_phase_budget, "boundary_planes_m": self.boundary_planes_m,
            "kinetic_remainder_phase_bound": self.kinetic_remainder_phase_bound,
            "model": "constant-energy static magnetic second-order forward scalar Hamiltonian",
            "qualification": "exact native coefficient identity; no fitted aberrations",
            "production_admission": False, "nonlinear_surrogate": False,
            "endpoint_convention": "no boundary action at either endpoint or inside the interval"}

    def _validate_current(self, snapshot, energy_ev, boundary_planes_m):
        if type(snapshot) is not ElectromagneticPotentialSnapshot or snapshot.numerical_digest != self.snapshot.numerical_digest:
            raise ValueError("Reduced field snapshot changed; requalification is required")
        energy_ev = _positive(energy_ev, "Electron energy")
        if not self.energy_range_ev[0] <= energy_ev <= self.energy_range_ev[1]:
            raise ValueError("Electron energy is outside the qualified range")
        planes = np.asarray(boundary_planes_m)
        if planes.dtype.kind not in "iuf" or planes.ndim != 1 or not np.isfinite(planes).all():
            raise ValueError("Current boundary planes must be finite real axial positions")
        self._boundaries(planes)
        return energy_ev

    def _hamiltonian(self, energy_ev):
        alpha = -e/_momentum(energy_ev)
        a, d, h = self.constant_a, self.linear_a, self.az_hessian
        matrix = np.zeros((4, 4))
        matrix[:2, :2] = alpha**2*d[:2].T@d[:2]-alpha*h
        matrix[:2, 2:] = -alpha*d[:2].T
        matrix[2:, :2] = matrix[:2, 2:].T
        matrix[2:, 2:] = np.eye(2)
        linear = np.r_[alpha**2*d[:2].T@a[:2]-alpha*d[2], -alpha*a[:2]]
        constant = .5*alpha**2*(a[:2]@a[:2])-alpha*a[2]
        return matrix, linear, float(constant)

    def canonical_path(self, snapshot, *, energy_ev, boundary_planes_m=()):
        from temsim.physics.column_wave import _linear_factor
        energy_ev = self._validate_current(snapshot, energy_ev, boundary_planes_m)
        matrix, linear, constant = self._hamiltonian(energy_ev)
        path = CanonicalPath(1e-3)
        _linear_factor(path, _J@matrix, self.interval_m[1]-self.interval_m[0],
                       force=_J@linear, constant_h=constant)
        return path

    def _support(self, positions, canonical, energy_ev):
        center = self.position_bounds_m.mean(axis=1)
        half = np.diff(self.position_bounds_m, axis=1).ravel()/2
        matrix, linear, _ = self._hamiltonian(energy_ev)
        origin = np.r_[center, (0., 0.)]
        extent = np.r_[np.max(abs(positions-center), axis=0), np.max(abs(canonical), axis=0)]
        augmented = np.zeros((5, 5))
        augmented[:4, :4] = abs(_J@matrix)
        augmented[:4, 4] = abs(_J@(matrix@origin+linear))
        bound = expm(augmented*(self.interval_m[1]-self.interval_m[0]))@np.r_[extent, 1.]
        if (not np.isfinite(bound).all() or np.any(bound[:2] > half)
                or np.any(bound[2:4] > self.maximum_canonical_slope)):
            raise ValueError("Propagation leaves the qualified position or canonical-momentum support")

    def propagate_rays(self, snapshot, positions_m, slopes_rad, *, energy_ev, boundary_planes_m=()):
        """Execute the qualified paraxial map; return mechanical slopes, no recentering."""
        energy_ev = self._validate_current(snapshot, energy_ev, boundary_planes_m)
        positions = np.asarray(positions_m)
        if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
            raise ValueError("Positions must be a nonempty N by 2 array")
        positions = _array(positions, positions.shape, "Ray positions")
        slopes = _array(slopes_rad, positions.shape, "Mechanical slopes")
        alpha = -e/_momentum(energy_ev)
        canonical = slopes+alpha*(self.constant_a[:2]+positions@self.linear_a[:2].T)
        self._support(positions, canonical, energy_ev)
        path = self.canonical_path(snapshot, energy_ev=energy_ev, boundary_planes_m=boundary_planes_m)
        output = np.column_stack((positions, canonical))@path.matrix.T+path.offset
        mechanical = output[:, 2:]-alpha*(self.constant_a[:2]+output[:, :2]@self.linear_a[:2].T)
        return output[:, :2], mechanical, freeze_json(self.metadata())

    def propagate_wave(self, snapshot, wave, *, energy_ev, boundary_planes_m=(), grid_numerics=None):
        """Keep the canonical carrier and continuous phase lift; never fit phase."""
        energy_ev = self._validate_current(snapshot, energy_ev, boundary_planes_m)
        if type(wave) is not PlaneWave or not isinstance(wave.amplitude, np.ndarray):
            raise ValueError("Qualified reduction requires an existing CPU PlaneWave")
        self._wave_boundary(wave)
        basis = wave.basis_m
        if np.linalg.det(basis) <= 0 or abs(basis[:, 0]@basis[:, 1]) > 1e-12*np.linalg.norm(basis[:, 0])*np.linalg.norm(basis[:, 1]):
            raise ValueError("Qualified wave reduction requires an orthogonal Cartesian lattice")
        wavelength = 2*np.pi*hbar/_momentum(energy_ev)
        xy = wave.coordinates_m().reshape(2, -1).T
        curvature = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
        tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
        carrier = tilt+(xy-wave.origin_m)@curvature.T
        frequency = .5*wavelength*np.sum(abs(np.linalg.inv(basis).T), axis=1)
        canonical = np.array(tuple(product(*[(-v, v) for v in np.max(abs(carrier), axis=0)+frequency])))
        self._support(xy, canonical, energy_ev)
        path = self.canonical_path(snapshot, energy_ev=energy_ev, boundary_planes_m=boundary_planes_m)
        output = propagate_plane_wave(wave, path.matrix, path.offset, wavelength,
            **path.phase_kwargs(), grid_numerics=grid_numerics)
        points = output.coordinates_m().reshape(2, -1).T
        if np.any(points < self.position_bounds_m[:, 0]) or np.any(points > self.position_bounds_m[:, 1]):
            raise ValueError("Output wave grid exceeds the qualified position domain")
        self._wave_boundary(output)
        return output, freeze_json({**self.metadata(), "input_flux": wave.probability,
            "output_flux": output.probability, "normalization_applied": False,
            "affine_action_m": path.action_m, "reference_phase_rad": path.reference_phase_rad})

    @staticmethod
    def _wave_boundary(wave):
        probability = abs(wave.amplitude)**2
        total = float(probability.sum())
        if min(probability.shape) < 8 or total <= 0:
            raise ValueError("Qualified wave requires positive flux and at least eight samples per axis")
        edge = total-float(probability[2:-2, 2:-2].sum())
        if edge > total*1e-10:
            raise ValueError("Qualified wave has occupied grid boundaries")

    def native_interactions(self):
        """Owned source records that this standalone reduction can replace."""
        return tuple(InteractionRecord("potential:"+row.source_id, row.source_id,
            "quadratic_magnetic_interval", "field", tuple(z*1e3 for z in self.interval_m),
            tuple("potential:magnetic:"+_canonical_owner(self.snapshot.scene, owner) for owner in row.owner_ids),
            row.potential.digest) for row in self.snapshot.contributions)

    def replace_interactions(self, registry):
        """Replace exact source records, preserving unrelated boundaries/actions."""
        required = self.native_interactions()
        if type(registry) is not InteractionRegistry or any(registry.records.count(row) != 1 for row in required):
            raise ValueError("Reduction requires every exact native source record once")
        remaining = tuple(row for row in registry.records if row not in required)
        lo, hi = (z*1e3 for z in self.interval_m)
        for row in remaining:
            start, stop = row.interval_mm
            if (row.representation != "observation" and start <= hi and stop >= lo):
                raise ValueError("Reduced interval overlaps an unreplaced field or boundary action")
        coverage = tuple(effect for row in required for effect in row.covered_effects)
        if not coverage:
            return registry  # Vacuum carries no magnetic ownership to replace.
        reduced = InteractionRecord("reduced:"+self.digest, "quadratic:"+self.digest,
            "quadratic_magnetic_interval", "reduced", tuple(z*1e3 for z in self.interval_m),
            coverage, self.snapshot.numerical_digest)
        return InteractionRegistry((*remaining, reduced))


def qualify_quadratic_interval(snapshot, **scope):
    """Capture a coefficient-proven optional map; no instrument state is changed."""
    return QualifiedQuadraticReduction(snapshot, **scope)
