"""Explicit retarded, adiabatic wavefront sampling of prescribed scan fields.

This temporary integration view is deliberately not a static field snapshot.
Its source identity remains the complete dynamic snapshot. The reference is
an axial stationary-mode clock, not a pulse envelope or an exact sideband
solution. Axial induced work is integrated; neglected transverse clock/work
spread is separately bounded before accepting the one-energy approximation.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from scipy.constants import c, e, hbar, m_e
from scipy.integrate import solve_ivp

from temsim.immutable_json import thaw_json
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
from temsim.physics.electric_wave_reference import (
    ElectricWaveHistory, _validated_sampled_phi, momentum_velocity, scalar_potential,
)


def initial_dynamic_state(emission_time_s, flight_time_s):
    from temsim.physics.emission_timing import validate_emission_time_s
    epoch = validate_emission_time_s(emission_time_s)
    if isinstance(flight_time_s, bool) or not isinstance(flight_time_s, Real) or not math.isfinite(flight_time_s) or flight_time_s < 0.:
        raise ValueError("Retarded wave propagation requires its actual executed axial flight time")
    return {"emission_time_s": epoch, "flight_time_s": float(flight_time_s),
        "initial_flight_time_s": float(flight_time_s), "axial_work_ev": 0.,
        "delay_bound_s": 0., "energy_spread_bound_ev": 0.,
        "phase_error_bound_rad": 0., "maximum_phase_error_rad": .01}


@dataclass(frozen=True)
class RetardedPotentialSlice:
    """Transient dense axial clock plus the unchanged captured dynamic field."""
    source_snapshot: DynamicPotentialSnapshot
    history: ElectricWaveHistory
    lower_z_m: float
    upper_z_m: float

    def __post_init__(self):
        if type(self.source_snapshot) is not DynamicPotentialSnapshot or type(self.history) is not ElectricWaveHistory:
            raise ValueError("Retarded propagation needs exact frozen field and energy history types")
        if self.history.dynamic_state is None or self.history.last_z_m != self.lower_z_m or self.upper_z_m <= self.lower_z_m:
            raise ValueError("Retarded integration must continue the declared clock and plane")
        self.history.reference.require(self.source_snapshot)
        from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
        object.__setattr__(self, "_static_sampler", compile_unified_potential_sampler(
            self.source_snapshot.static_snapshot, prefer_plane=True))
        for coil in self.source_snapshot.dynamic_coils:
            rotation = coil.base.base.field.registration.rotation_array
            if not np.array_equal(rotation[:, 2], (0., 0., 1.)):
                raise ValueError("Retarded scalar wave currently requires globally axial native scan coils")
        state = self.history.dynamic_state
        p0, _ = momentum_velocity(self.entrance_energy_ev)
        rest_ev = m_e*c*c/e
        def rhs(z, values):
            work, elapsed = values[:2]
            phi = float(scalar_potential(self.source_snapshot, np.array([0., 0., z])))
            delta = phi-self.history.reference.entrance_phi_v+work
            energy = self.entrance_energy_ev+delta
            p, velocity = momentum_velocity(energy)
            time = state["emission_time_s"]+elapsed
            work_derivative = 0.
            for coil in self.source_snapshot.dynamic_coils:
                if self._side(coil) != "outside":
                    work_derivative += float(coil.sample(np.array([0., 0., z]), time)["dA_dt_t_m_per_s"][2])
            delta_p = (e/c)**2*delta*(2*(self.entrance_energy_ev+rest_ev)+delta)/(p+p0)
            # H-H_entry = e*work. Spatial p dz remains in AxialWaveReference;
            # the removed entrance-frequency convention ALSO needs -dH dt.
            return (work_derivative, 1./velocity, delta_p, -e*work/velocity)
        solution = solve_ivp(rhs, (self.lower_z_m, self.upper_z_m),
            (state["axial_work_ev"], state["flight_time_s"], 0., 0.), method="DOP853",
            rtol=2e-12, atol=(1e-17, 1e-23, 1e-39, 1e-39), dense_output=True,
            max_step=(self.upper_z_m-self.lower_z_m)/2.)
        if not solution.success or not np.isfinite(solution.y).all():
            raise ValueError("Retarded axial energy/clock integration did not converge")
        self.source_snapshot.require_time_interval(state["emission_time_s"]+state["flight_time_s"],
            state["emission_time_s"]+solution.y[1, -1])
        object.__setattr__(self, "_solution", solution)

    @property
    def numerical_digest(self):
        return self.source_snapshot.numerical_digest

    @property
    def scene(self):
        return self.source_snapshot.scene

    @property
    def gauge(self):
        return self.source_snapshot.gauge

    @property
    def contributions(self):
        return self.source_snapshot.contributions

    @property
    def backend(self):
        return "retarded-"+self._static_sampler.backend

    @property
    def entrance_energy_ev(self):
        return self.history.reference.entrance_energy_ev

    def _side(self, coil):
        native = coil.base.base.field
        faces = np.asarray(native.native_support_mm)*.001+native.registration.origin_array_m[2]
        if self.upper_z_m <= faces[0] or self.lower_z_m >= faces[1]:
            return "outside"
        if self.lower_z_m >= faces[0] and self.upper_z_m <= faces[1]:
            return "inside"
        raise ValueError("Retarded scan step crosses a native coil face; insert the exact face")

    def absolute_time(self, z_m):
        z = np.asarray(z_m, dtype=float)
        if np.any(z < self.lower_z_m) or np.any(z > self.upper_z_m):
            raise ValueError("Retarded sample leaves its integrated clock interval")
        return self.history.dynamic_state["emission_time_s"]+self._values(z)[1]

    def _values(self, z_m):
        z = np.asarray(z_m, dtype=float)
        if z.size and np.all(z == z.reshape(-1)[0]):
            # Plane samples share one exact axial coordinate. Evaluate the
            # same dense solution once, without quantizing Z or the clock.
            values = self._solution.sol(float(z.reshape(-1)[0]))
            return np.broadcast_to(values.reshape((4,)+(1,)*z.ndim), (4, *z.shape))
        return self._solution.sol(z.reshape(-1)).reshape((4, *z.shape))

    def sample(self, positions_m):
        points = np.asarray(positions_m, dtype=float)
        result = self._static_sampler.sample(points)
        times = self.absolute_time(points[..., 2])
        for coil in self.source_snapshot.dynamic_coils:
            if self._side(coil) == "outside":
                # The identical native position/domain check is required even
                # for exact compact exterior zeros. With no active point the
                # original sample only allocates zero A/B outputs; all finite
                # and radial/registration checks have already passed. A face
                # or any active point still uses the full original sampler.
                _, active = coil.base._positions(points, derivatives=False)
                if np.any(active):
                    coil.base.sample(points)
                del active
                continue
            sampled = coil.sample(points, times)
            for key in ("A_t_m", "B_t", "E_v_per_m"):
                result[key] += sampled[key]
            # Overlapping active coils are accumulated in the same order.
            # Their obsolete A/B/E/time-derivative bundle need not remain
            # live while the next independent coil allocates its samples.
            del sampled
        return result

    def require(self, snapshot):
        if snapshot is not self:
            raise ValueError("Retarded energy reference belongs to a different executed clock slice")
        return self

    def axis_energy(self, snapshot, z_m):
        self.require(snapshot)
        z = np.asarray(z_m, dtype=float)
        phi = scalar_potential(self.source_snapshot, np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=-1))
        return self.entrance_energy_ev+(phi-self.history.reference.entrance_phi_v+self._values(z)[0])

    def squared_momentum_defect(self, snapshot, positions, axis_z_m, chart_energy_ev):
        self.require(snapshot)
        phi_axis = float(scalar_potential(self.source_snapshot, np.array([0., 0., axis_z_m])))
        delta = scalar_potential(self.source_snapshot, positions)-phi_axis
        axis_energy = float(self.axis_energy(self, axis_z_m))
        if np.any(axis_energy+delta <= 0.):
            raise ValueError("Retarded wave reached a local turning region")
        rest_ev = m_e*c*c/e
        return -delta*(2*(axis_energy+rest_ev)+delta)/(chart_energy_ev*(chart_energy_ev+2*rest_ev))

    def _squared_momentum_defect_from_phi(self, snapshot, sampled_phi_v, axis_z_m, chart_energy_ev):
        """Reuse this exact slice's captured phi without changing clock or work.

        Only the local full-grid electric query is reused. Source identity,
        the on-axis potential, executed retarded energy and turning-region
        rejection remain identical to the standalone method.
        """
        self.require(snapshot)
        phi_axis = float(scalar_potential(self.source_snapshot, np.array([0., 0., axis_z_m])))
        delta = _validated_sampled_phi(sampled_phi_v)-phi_axis
        axis_energy = float(self.axis_energy(self, axis_z_m))
        if np.any(axis_energy+delta <= 0.):
            raise ValueError("Retarded wave reached a local turning region")
        rest_ev = m_e*c*c/e
        return -delta*(2*(axis_energy+rest_ev)+delta)/(chart_energy_ev*(chart_energy_ev+2*rest_ev))

    def forward_variation_bound(self, snapshot, lower, upper, radius_m):
        self.require(snapshot)
        work_lower, work_upper, variation = self._work_bounds(lower, upper)
        return self.history.reference.forward_variation_bound(self.source_snapshot, lower, upper, radius_m,
            work_bounds=(work_lower, work_upper), work_variation_ev=variation)

    def _work_bounds(self, lower, upper):
        derivative = 0.
        for coil in self.source_snapshot.dynamic_coils:
            if self._side(coil) != "outside":
                offset = np.linalg.norm(coil.base.base.field.registration.origin_array_m[:2])
                derivative += offset*np.linalg.norm(coil.temporal_bounds()["derivatives"][1]["coefficient_abs_bound"])
        initial = self.history.dynamic_state["axial_work_ev"]
        reach = derivative*(upper-self.lower_z_m)
        return initial-reach, initial+reach, derivative*(upper-lower)

    def to_dict(self):
        return {"schema": "retarded-potential-slice-v1", "source_digest": self.numerical_digest,
                "source_energy_reference": self.history.reference.to_dict(),
                "initial_timing": thaw_json(self.history.dynamic_state),
                "lower_z_m": self.lower_z_m, "upper_z_m": self.upper_z_m,
                "ode_nfev": self._solution.nfev,
                "clock_scope": "axial stationary-mode flight reference, not a pulsed temporal envelope"}

    def axial_integrals(self):
        p0, _ = momentum_velocity(self.entrance_energy_ev)
        baseline = float(p0)*(self.upper_z_m-self.lower_z_m)
        correction = float(self._solution.y[2, -1])
        return {"longitudinal_action_j_s": baseline+correction,
                "baseline_longitudinal_action_j_s": baseline,
                "electric_action_correction_j_s": correction,
                "electric_phase_correction_rad": correction/hbar,
                "dynamic_temporal_action_j_s": float(self._solution.y[3, -1]),
                "dynamic_temporal_phase_rad": float(self._solution.y[3, -1])/hbar,
                "axial_flight_time_s": float(self._solution.y[1, -1])-self.history.dynamic_state["flight_time_s"],
                "quadrature_order": "DOP853 axial work/clock/action", "native_cells": 1}

    def advance_adiabatic_state(self, steps):
        """Accumulate full-grid clock/work-spread raw-phase bounds, no weight scaling."""
        state = thaw_json(self.history.dynamic_state)
        for step in steps:
            lower, upper = step["lower_z_m"], step["upper_z_m"]
            radius, theta2 = step["radius_m"], step["kinetic_squared_bound"]
            if not 0. <= theta2 < .1:
                raise ValueError("Retarded wave exceeds its positive longitudinal momentum margin")
            theta = math.sqrt(theta2)
            distance = upper-lower
            first, second = 0., 0.
            for coil in self.source_snapshot.dynamic_coils:
                if self._side(coil) == "outside":
                    continue
                native = coil.base.base.field
                transverse_origin = np.linalg.norm(native.registration.origin_array_m[:2])
                bounds = coil.temporal_bounds()["derivatives"]
                first += (radius+transverse_origin)*np.linalg.norm(bounds[1]["coefficient_abs_bound"])
                second += (radius+transverse_origin)*np.linalg.norm(bounds[2]["coefficient_abs_bound"])
            # The axis work is retained explicitly. A one-energy transverse
            # wave omits the DIFFERENCE between each ray's and axial work;
            # twice the uniform derivative bound covers this difference.
            state["energy_spread_bound_ev"] += 2.*first*(1.+theta)*distance
            low, high, _ = self.history.reference.cylinder_energy_bounds(self.source_snapshot, lower, upper, radius)
            work_low, work_high, _ = self._work_bounds(lower, upper)
            low += work_low-state["energy_spread_bound_ev"]
            high += work_high+state["energy_spread_bound_ev"]
            if low <= 0.:
                raise ValueError("Retarded energy/clock majorant reaches a turning region")
            _, velocities = momentum_velocity(np.array([low, high]))
            v_min, v_max = map(float, velocities)
            # Both geometric path length and transverse electrostatic /
            # induced energy variation affect the laboratory arrival time.
            state["delay_bound_s"] += distance*(1./(v_min*math.sqrt(1.-theta2))-1./v_max)
            delay = state["delay_bound_s"]
            delta_a = first*delay+.5*second*delay*delay
            phase = e/hbar*distance*(
                ((1.+theta)*delta_a+state["energy_spread_bound_ev"]/v_min)/math.sqrt(1.-theta2))
            state["phase_error_bound_rad"] += phase
            if state["phase_error_bound_rad"] > state["maximum_phase_error_rad"]:
                raise ValueError("Retarded scan wave exceeds its cumulative clock/induced-energy raw-phase budget; a time/energy-resolved wave is required")
            self.source_snapshot.require_time_interval(self.absolute_time(lower)-delay,
                                                       self.absolute_time(upper)+delay)
        state["flight_time_s"] = float(self._solution.y[1, -1])
        state["axial_work_ev"] = float(self._solution.y[0, -1])
        return state
