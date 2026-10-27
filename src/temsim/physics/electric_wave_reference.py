"""Executed electrostatic energy law for a flux-normalized forward wave.

The input mode declares its kinetic energy on the optical axis at the supplied
entrance. Its conserved stationary total energy determines every later local
momentum from the *same* captured scalar potential. This is not a fit, a new
field solve, or a claim that a one-way scalar wave includes reflected waves.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import math

import numpy as np
from scipy.constants import c, e, hbar, m_e

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


_REST_EV = m_e*c*c/e


def _finite(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0.):
        raise ValueError(f"{name} must be finite"+(" and positive" if positive else ""))
    return value


def momentum_velocity(energy_ev):
    energy = np.asarray(energy_ev, dtype=float)
    if not np.isfinite(energy).all() or np.any(energy <= 0.):
        raise ValueError("The positive forward wave branch requires positive kinetic energy")
    momentum = e*np.sqrt(energy*(energy+2*_REST_EV))/c
    return momentum, c*np.sqrt(energy*(energy+2*_REST_EV))/(energy+_REST_EV)


def electric_rows(snapshot):
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    if type(snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot):
        raise ValueError("Static electric wave energy needs an immutable potential snapshot")
    rows = tuple(row for row in snapshot.contributions if row.kind == "electric")
    if any(type(row.potential) is not ElectricPotentialSnapshot for row in rows):
        raise ValueError("Unsupported scalar-potential interpolation law")
    return rows


def scalar_potential(snapshot, points):
    points = np.asarray(points, dtype=float)
    result = np.zeros(points.shape[:-1], dtype=float)
    for row in electric_rows(snapshot):
        result += row.potential.sample(points)["phi_v"]
    return result


def _validated_sampled_phi(sampled_phi_v):
    """Admit a captured real scalar potential without changing its values.

    The caller must bind this array to the exact snapshot and physical points
    already admitted by the full provider. This helper cannot establish that
    provenance from an unlabelled array; it only prevents invalid values from
    entering the private arithmetic-only reuse path.
    """
    value = np.asarray(sampled_phi_v)
    if value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError("Captured scalar potential must contain finite real voltages")
    return np.asarray(value, dtype=float)


@dataclass(frozen=True, slots=True)
class ElectrostaticWaveReference:
    """Immutable source energy reference; continuation never chooses a new zero."""
    snapshot_digest: str
    entrance_z_m: float
    entrance_energy_ev: float
    entrance_phi_v: float

    def __post_init__(self):
        if type(self.snapshot_digest) is not str or not self.snapshot_digest.strip() or self.snapshot_digest.strip() != self.snapshot_digest:
            raise ValueError("An electrostatic wave reference requires a snapshot identity")
        for key in ("entrance_z_m", "entrance_energy_ev", "entrance_phi_v"):
            object.__setattr__(self, key, _finite(getattr(self, key), key, positive=key == "entrance_energy_ev"))

    @classmethod
    def capture(cls, snapshot, z_m, energy_ev):
        electric_rows(snapshot)
        z = _finite(z_m, "Entrance Z")
        return cls(snapshot.numerical_digest, z, energy_ev, float(scalar_potential(snapshot, np.array([0., 0., z]))))

    @property
    def digest(self):
        return json_digest(self._payload())

    def _payload(self):
        return {"schema": "electrostatic-wave-reference-v1", "snapshot_digest": self.snapshot_digest,
                "entrance_z_m": self.entrance_z_m, "entrance_energy_ev": self.entrance_energy_ev,
                "entrance_phi_v": self.entrance_phi_v}

    def to_dict(self):
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {"schema", "snapshot_digest", "entrance_z_m", "entrance_energy_ev", "entrance_phi_v", "digest"}
                or value["schema"] != "electrostatic-wave-reference-v1"):
            raise ValueError("Invalid electrostatic wave reference")
        result = cls(*(value[name] for name in ("snapshot_digest", "entrance_z_m", "entrance_energy_ev", "entrance_phi_v")))
        if result.digest != value["digest"]:
            raise ValueError("Electrostatic wave energy reference digest mismatch")
        return result

    def require(self, snapshot):
        if snapshot.numerical_digest != self.snapshot_digest:
            raise ValueError("Electrostatic wave energy source changed")
        electric_rows(snapshot)
        actual = float(scalar_potential(snapshot, np.array([0., 0., self.entrance_z_m])))
        if actual != self.entrance_phi_v:
            raise ValueError("Electrostatic wave reference does not match its original scalar potential")
        return self

    def energy(self, snapshot, points):
        self.require(snapshot)
        # Volt differences enter electron kinetic energies in eV. Subtract the
        # original potential BEFORE adding the 300-keV-scale kinetic energy.
        result = self.entrance_energy_ev+(scalar_potential(snapshot, points)-self.entrance_phi_v)
        if not np.isfinite(result).all() or np.any(result <= 0.):
            raise ValueError("Electrostatic wave reached a turning or non-forward region")
        return result

    def axis_energy(self, snapshot, z_m):
        z = np.asarray(z_m, dtype=float)
        return self.energy(snapshot, np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=-1))

    def squared_momentum_defect(self, snapshot, positions, axis_z_m, chart_energy_ev):
        """Return (p_axis^2-p_local^2)/p_chart^2 without subtracting p^2."""
        self.require(snapshot)
        axis_phi = float(scalar_potential(snapshot, np.array([0., 0., axis_z_m])))
        local_phi = scalar_potential(snapshot, positions)
        delta = local_phi-axis_phi
        axis_energy = self.entrance_energy_ev+(axis_phi-self.entrance_phi_v)
        local_energy = axis_energy+delta
        if np.any(local_energy <= 0.):
            raise ValueError("Electrostatic wave reached a local turning region")
        return -delta*(2*(axis_energy+_REST_EV)+delta)/(chart_energy_ev*(chart_energy_ev+2*_REST_EV))

    def _squared_momentum_defect_from_phi(self, snapshot, sampled_phi_v, axis_z_m, chart_energy_ev):
        """Use the same query's already sampled potential; retain every energy gate.

        This private path is for the carrier's exact known sampler dispatch.
        Unknown adapters must use ``squared_momentum_defect`` so their original
        position, domain and provider validation is executed independently.
        The subtraction and multiplication order matches that public method.
        """
        self.require(snapshot)
        axis_phi = float(scalar_potential(snapshot, np.array([0., 0., axis_z_m])))
        local_phi = _validated_sampled_phi(sampled_phi_v)
        delta = local_phi-axis_phi
        axis_energy = self.entrance_energy_ev+(axis_phi-self.entrance_phi_v)
        local_energy = axis_energy+delta
        if np.any(local_energy <= 0.):
            raise ValueError("Electrostatic wave reached a local turning region")
        return -delta*(2*(axis_energy+_REST_EV)+delta)/(chart_energy_ev*(chart_energy_ev+2*_REST_EV))

    def axial_nodes(self, snapshot, lower, upper):
        """All native axial interpolation knots, including real short cells."""
        self.require(snapshot)
        nodes = [float(lower), float(upper)]
        for row in electric_rows(snapshot):
            z = row.potential.arrays["z"]
            nodes.extend(z[(z > lower) & (z < upper)])
        return np.unique(nodes)

    def axial_integrals(self, snapshot, lower, upper, *, quadrature_order=8):
        """Reference action/time; stable electric action correction is separate.

        Gauss quadrature stays inside each original interpolation cell. The
        stable correction is retained separately from the large baseline,
        so validation can observe a weak tail without subtracting large S.
        """
        if type(quadrature_order) is not int or quadrature_order not in (4, 8, 16):
            raise ValueError("Axial energy quadrature order must be 4, 8 or 16")
        nodes = self.axial_nodes(snapshot, lower, upper)
        q, w = np.polynomial.legendre.leggauss(quadrature_order)
        z = (nodes[:-1, None]+nodes[1:, None])/2+np.diff(nodes)[:, None]*q/2
        weights = np.diff(nodes)[:, None]*w/2
        kinetic = self.axis_energy(snapshot, z)
        p, velocity = momentum_velocity(kinetic)
        p0, _ = momentum_velocity(self.entrance_energy_ev)
        delta = scalar_potential(snapshot, np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=-1))-self.entrance_phi_v
        delta_p = (e/c)**2*delta*(2*(self.entrance_energy_ev+_REST_EV)+delta)/(p+p0)
        correction = math.fsum((weights*delta_p).ravel())
        baseline = float(p0)*(upper-lower)
        return {"longitudinal_action_j_s": baseline+correction,
                "baseline_longitudinal_action_j_s": baseline,
                "electric_action_correction_j_s": correction,
                "electric_phase_correction_rad": correction/hbar,
                "axial_flight_time_s": math.fsum((weights/velocity).ravel()),
                "quadrature_order": quadrature_order, "native_cells": len(nodes)-1}

    def cylinder_energy_bounds(self, snapshot, lower, upper, radius_m):
        """Native node extrema/variation over the complete declared cylinder."""
        self.require(snapshot)
        radius = _finite(radius_m, "Qualified electric radius", positive=True)
        total_variation, minimum_phi, maximum_phi = 0., 0., 0.
        for row in electric_rows(snapshot):
            provider = row.potential
            if provider.metadata["basis"] != "axisymmetric-r2-z-bilinear-v1":
                raise ValueError("Accelerating electric wave variation requires the native r-squared/Z bilinear law")
            arrays = provider.arrays
            radial = np.unique(np.r_[0., arrays["r"][(arrays["r"] > 0.) & (arrays["r"] < radius)], radius])
            axial = np.unique(np.r_[lower, arrays["z"][(arrays["z"] > lower) & (arrays["z"] < upper)], upper])
            points = np.zeros((len(radial), len(axial), 3))
            points[..., 0], points[..., 2] = radial[:, None], axial[None, :]
            phi = provider.sample(points)["phi_v"]
            total_variation += math.fsum(np.max(abs(np.diff(phi, axis=1)), axis=0))
            minimum_phi += float(np.min(phi))
            maximum_phi += float(np.max(phi))
        return (self.entrance_energy_ev+(minimum_phi-self.entrance_phi_v),
                self.entrance_energy_ev+(maximum_phi-self.entrance_phi_v), total_variation)

    def forward_variation_bound(self, snapshot, lower, upper, radius_m, *, work_bounds=(0., 0.), work_variation_ev=0.):
        """Conservative added-scalar branch-coupling budget on a cylinder.

        For the admitted r-squared/Z bilinear law, all extrema on this
        cylinder occur on radial/axial interpolation nodes. The sum of the
        largest cell changes bounds the axial total variation for every
        represented radial position. For Q positive, C=Q**(1/4) gives the
        flux variables (C psi +/- C**(-1) pi_z psi)/sqrt(2). Its extra
        connection has norm at most kappa(Q)**(1/4)*||dQ||/(4*lambda_min(Q)),
        by the positive-operator fractional-power Frechet derivative bound.
        Q has the separately enforced [0.9, 1.1] momentum-squared margin.
        This is an additional ELECTRIC adiabatic
        bound; it does not qualify previously omitted magnetic quantum/spin
        terms or replace independent full-wave grid convergence.
        """
        minimum_energy, maximum_energy, total_variation = self.cylinder_energy_bounds(snapshot, lower, upper, radius_m)
        minimum_energy += work_bounds[0]
        maximum_energy += work_bounds[1]
        total_variation += work_variation_ev
        if minimum_energy <= 0.:
            raise ValueError("Electric variation does not bound a positive forward branch")
        # Common (e/c)^2 cancels, avoiding subnormal SI momenta here.
        q_min = .9*minimum_energy*(minimum_energy+2*_REST_EV)
        q_max = 1.1*maximum_energy*(maximum_energy+2*_REST_EV)
        variation_q = 2.*(maximum_energy+_REST_EV)*total_variation
        exponent = (q_max/q_min)**.25*variation_q/(4.*q_min)
        return math.expm1(exponent) if exponent < 700. else math.inf


@dataclass(frozen=True, slots=True)
class ElectricWaveHistory:
    """A fixed source energy and non-resetting added-electric error budget."""
    reference: ElectrostaticWaveReference
    maximum_forward_error: float = 1e-6
    events: tuple = ()
    dynamic_state: object = None

    def __post_init__(self):
        if type(self.reference) is not ElectrostaticWaveReference:
            raise ValueError("Electric wave history needs an immutable energy reference")
        maximum = _finite(self.maximum_forward_error, "Electric forward amplitude budget", positive=True)
        events = freeze_json(self.events)
        if not isinstance(events, tuple):
            raise ValueError("Electric wave events must be a sequence")
        previous = self.reference.entrance_z_m
        for event in events:
            if not isinstance(event, Mapping) or set(event) != {"lower_z_m", "upper_z_m", "forward_error_bound"}:
                raise ValueError("Invalid electric wave event")
            lower, upper = _finite(event["lower_z_m"], "Electric event start"), _finite(event["upper_z_m"], "Electric event end")
            bound = _finite(event["forward_error_bound"], "Electric forward error")
            if lower != previous or upper <= lower or bound < 0.:
                raise ValueError("Electric wave events must continue at exactly the last executed plane")
            previous = upper
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "maximum_forward_error", maximum)
        dynamic = None if self.dynamic_state is None else freeze_json(self.dynamic_state)
        if dynamic is not None:
            names = {"emission_time_s", "flight_time_s", "initial_flight_time_s", "axial_work_ev",
                     "delay_bound_s", "energy_spread_bound_ev", "phase_error_bound_rad", "maximum_phase_error_rad"}
            if not isinstance(dynamic, Mapping) or set(dynamic) != names:
                raise ValueError("Invalid retarded wave timing and error history")
            for key, value in dynamic.items():
                checked = _finite(value, key, positive=key == "maximum_phase_error_rad")
                if key not in ("emission_time_s", "axial_work_ev") and checked < 0.:
                    raise ValueError("Retarded timing and error bounds cannot be negative")
            if dynamic["flight_time_s"] < dynamic["initial_flight_time_s"]:
                raise ValueError("Retarded wave flight time cannot restart")
            if dynamic["phase_error_bound_rad"] > dynamic["maximum_phase_error_rad"]:
                raise ValueError("Retarded wave approximation exceeds its cumulative raw-phase budget")
        object.__setattr__(self, "dynamic_state", dynamic)
        if self.used_forward_error > maximum:
            raise ValueError("Electric forward branch approximation exceeds its cumulative amplitude budget")

    @property
    def last_z_m(self):
        return self.events[-1]["upper_z_m"] if self.events else self.reference.entrance_z_m

    @property
    def used_forward_error(self):
        return math.expm1(math.fsum(math.log1p(row["forward_error_bound"]) for row in self.events))

    @property
    def digest(self):
        return json_digest(self._payload())

    def _payload(self):
        return {"schema": "electric-wave-history-v1", "reference": self.reference.to_dict(),
                "maximum_forward_error": self.maximum_forward_error, "events": thaw_json(self.events),
                "dynamic_state": thaw_json(self.dynamic_state)}

    def to_dict(self):
        return {**self._payload(), "digest": self.digest}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping) or set(value) != {"schema", "reference", "maximum_forward_error", "events", "dynamic_state", "digest"} or value["schema"] != "electric-wave-history-v1":
            raise ValueError("Invalid electric wave history")
        result = cls(ElectrostaticWaveReference.from_dict(value["reference"]), value["maximum_forward_error"], tuple(value["events"]), value["dynamic_state"])
        if result.digest != value["digest"]:
            raise ValueError("Electric wave history digest mismatch")
        return result

    def require(self, snapshot, z_m, energy_ev):
        self.reference.require(snapshot)
        if float(z_m) != self.last_z_m:
            raise ValueError("Electric wave history cannot restart on another plane")
        expected = float(self.reference.axis_energy(snapshot, z_m))
        if self.dynamic_state is not None:
            phi = float(scalar_potential(snapshot, np.array([0., 0., z_m])))
            expected = self.reference.entrance_energy_ev+(phi-self.reference.entrance_phi_v+self.dynamic_state["axial_work_ev"])
        if energy_ev != expected and energy_ev != expected*.001*1000.:
            raise ValueError("Wave kinetic energy differs from its captured electrostatic energy law")
        return self

    def append(self, lower, upper, bound, *, dynamic_state=None):
        if self.dynamic_state is not None and dynamic_state is None:
            raise ValueError("Dynamic electric history requires its advanced clock and error receipt")
        return ElectricWaveHistory(self.reference, self.maximum_forward_error,
            (*self.events, {"lower_z_m": float(lower), "upper_z_m": float(upper), "forward_error_bound": float(bound)}), dynamic_state)
