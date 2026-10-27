"""Exact first-crossing observations of the production geometry-field gun.

The observer re-integrates a fraction of an already accepted two-half-step
discrete-gradient trajectory to its requested Z. It never changes that
trajectory, source population, physical stop, or ordinary cached result.
"""
from dataclasses import dataclass
import numpy as np

from temsim.physics.relativistic_lorentz import kinetic_energy_ev_from_momentum


@dataclass(frozen=True)
class GunPlaneObservations:
    z_mm: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    tx_rad: np.ndarray
    ty_rad: np.ndarray
    flight_time_s: np.ndarray
    kinetic_energy_ev: np.ndarray
    reached: np.ndarray
    alive: np.ndarray
    plane_residual_m: np.ndarray
    schema: str = "gun-exact-dg-first-crossings-v1"


class ExactGunPlaneObserver:
    """Collect requested planes without interpolating a display history."""
    def __init__(self, planes_mm, exit_z_mm):
        planes = np.asarray(planes_mm, dtype=float)
        if (planes.ndim != 1 or not 1 <= len(planes) <= 20000
                or not np.isfinite(planes).all() or np.any(np.diff(planes) <= 0)
                or not np.isfinite(exit_z_mm) or planes[0] < 0 or planes[-1] > exit_z_mm):
            raise ValueError("Exact gun planes must increase from emission through the existing exit")
        self.planes = planes.copy()
        self.values = None

    def initialize(self, phase):
        count = len(phase.position_m)
        if len(self.planes)*count*8*8 > 512*1024**2:
            raise ValueError("Exact gun observation arrays exceed the bounded memory budget")
        shape = (len(self.planes), count)
        self.values = [np.full(shape, np.nan) for _ in range(7)]
        self.reached = np.zeros(shape, dtype=bool)
        self.alive = np.zeros(shape, dtype=bool)
        for row, plane in enumerate(self.planes*.001):
            for index in np.flatnonzero(phase.position_m[:, 2] == plane):
                self._store(row, index, plane, phase.position_m[index],
                    phase.momentum_kg_m_per_s[index], phase.time_s, True)

    def _store(self, row, index, plane, position, momentum, time_s, alive):
        residual = float(position[2]-plane)
        if (not np.isfinite((time_s, residual)).all() or abs(residual) > 1e-12
                or not np.isfinite(position).all() or not np.isfinite(momentum).all()
                or momentum[2] <= 0):
            raise ValueError("Exact gun crossing failed its plane, time, or forward-momentum check")
        x, y, tx, ty, times, energy, error = self.values
        x[row, index], y[row, index] = position[:2]
        tx[row, index], ty[row, index] = momentum[:2]/momentum[2]
        times[row, index] = time_s
        energy[row, index] = kinetic_energy_ev_from_momentum(momentum)
        error[row, index] = residual
        self.reached[row, index] = True
        self.alive[row, index] = bool(alive)

    def observe(self, previous_position, previous_momentum, previous_time_s,
                new_position, new_time_s, active_before_step, blocked_time_s,
                electric, magnetic):
        from temsim.optics.electron_gun.tracing import _surface_plane_crossing
        duration = float(new_time_s-previous_time_s)
        if duration <= 0 or not np.isfinite(duration):
            raise ValueError("Exact gun observation requires a positive accepted step duration")
        active = np.asarray(active_before_step, bool)
        if not active.any():
            return
        lower = float(np.min(previous_position[active, 2]))
        upper = float(np.max(new_position[active, 2]))
        planes = self.planes*.001
        for row in np.flatnonzero((planes >= lower) & (planes <= upper)):
            plane = planes[row]
            selected = np.flatnonzero(active & ~self.reached[row]
                & (previous_position[:, 2] <= plane) & (new_position[:, 2] >= plane))
            for index in selected:
                crossing = _surface_plane_crossing(plane, previous_position[index],
                    previous_momentum[index], previous_time_s, duration, electric, magnetic)
                stopped = np.isfinite(blocked_time_s[index])
                if stopped and crossing.time_s > blocked_time_s[index]:
                    continue
                self._store(row, index, plane, crossing.position_m[0],
                    crossing.momentum_kg_m_per_s[0], crossing.time_s,
                    not stopped or crossing.time_s < blocked_time_s[index])

    def result(self):
        if self.values is None:
            raise ValueError("Exact gun observations were not initialized by an executed source")
        return GunPlaneObservations(self.planes.copy(), *[v.copy() for v in self.values[:6]],
            self.reached.copy(), self.alive.copy(), self.values[6].copy())
