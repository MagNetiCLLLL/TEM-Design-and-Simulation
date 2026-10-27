"""Request-local reuse of exactly matching prepared unified trajectories.

Only transport is shared. Material labels, branch probabilities and detector
collection remain owned by each projected response. No material path or
zero-loss label is used as evidence that two transport requests are equal.
"""
from hashlib import sha256

import numpy as np


MAX_RETAINED_BYTES = 16 * 1024**2


class ProjectedTrajectoryReuse:
    """Retain at most one compact transport within one response construction."""

    def __init__(self):
        self._identity = None
        self._stored = None

    def propagate(self, state, start, stop, phase, events, energy_offset, *,
                  save_z_mm, initial_time_s, kinetic_energy_ev, weights,
                  source_ids, source_azimuths, absolute_probability,
                  recording_interceptions):
        from temsim.calculation_manifest import solver_source_identity
        from temsim.physics.core import build_propagation_plan, execute_propagation_plan
        from temsim.physics.optical_tuning import check_tuning_cancelled
        from temsim.physics.unified_field_particle import merge_unified_stops

        check_tuning_cancelled(state)
        if str(getattr(state, "simulation_mode", "")) != "unified" or state.vacuum_map.enabled:
            raise ValueError("Projected trajectory reuse requires unified transport without medium collisions")
        plan = build_propagation_plan(state, start, stop, events, save_z_mm=save_z_mm,
                                      particle_medium=False, medium_energy_ev=kinetic_energy_ev)
        if plan.unified_runtime is None:
            raise ValueError("Projected trajectory reuse requires an executed unified field plan")
        # The exact plan binds the complete captured field/geometry/epoch and
        # every RK node. Backend admission is also part of this request; a hit
        # cannot turn an unsupported Require GPU request into a CPU success.
        digest = sha256(repr(("prepared-projected-trajectory-v1", solver_source_identity(),
            plan.signature, plan.unified_runtime.digest, recording_interceptions,
            getattr(state, "acceleration_backend", None),
            getattr(state, "acceleration_enabled", None), absolute_probability,
            "no-initial-kicks", "defer-nonfinite-until-clipping")).encode("utf-8"))
        for value in (*phase, energy_offset, initial_time_s, kinetic_energy_ev,
                      weights, source_ids, source_azimuths, np.asarray(save_z_mm)):
            array = np.ascontiguousarray(value)
            if array.dtype.kind not in "biuf":
                raise ValueError("Prepared projected transport requires numerical arrays")
            digest.update(repr((array.dtype.str, array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        identity = digest.hexdigest()
        if identity == self._identity and self._stored is not None:
            check_tuning_cancelled(state)
            return self._copy(self._stored), True

        energy = []
        result = execute_propagation_plan(state, plan, *phase, energy_offset,
            include_initial_plane_kicks=False, defer_nonfinite_until_clipping=True,
            initial_time_s=initial_time_s, return_flight_times=True,
            initial_kinetic_energy_ev=kinetic_energy_ev, energy_output=energy)
        z, x, tx, y, ty, clocks = result[:6]
        count = x.shape[1]
        alive, blocked, labels = merge_unified_stops(z, np.ones(count, dtype=bool),
            np.full(count, np.nan), [""] * count)
        values = (z, x, tx, y, ty, clocks, energy[0], alive, blocked, tuple(labels))
        compact = self._compact(values, save_z_mm)
        check_tuning_cancelled(state)
        self._identity, self._stored = None, None
        if sum(value.nbytes for value in compact[:-1]) <= MAX_RETAINED_BYTES:
            self._stored = self._copy(compact)
            self._identity = identity
        return values, False

    @staticmethod
    def _copy(values):
        return (*tuple(value.copy() for value in values[:-1]), tuple(values[-1]))

    @staticmethod
    def _compact(values, saved):
        z, x, tx, y, ty, clocks, energy, alive, blocked, labels = values
        keep = {0, len(z)-1}
        for plane in saved:
            rows = np.flatnonzero(z == plane)
            if not rows.size:
                raise ValueError("Prepared trajectory lacks an exact requested observation plane")
            keep.add(int(rows[-1]))
        # Preserve both original samples bracketing every mechanical contact.
        for location in blocked[np.isfinite(blocked)]:
            hi = min(int(np.searchsorted(z, location)), len(z)-1)
            keep.update((max(0, hi-1), hi))
        # Numerical truncation is classified by the ordinary response builder.
        # Retain its exact first invalid node and preceding valid node as well.
        invalid = ~(np.isfinite(x) & np.isfinite(tx) & np.isfinite(y)
                    & np.isfinite(ty) & np.isfinite(energy))
        for ray in np.flatnonzero(np.any(invalid, axis=0)):
            first = int(np.flatnonzero(invalid[:, ray])[0])
            keep.update((max(0, first-1), first))
        rows = np.array(sorted(keep))
        return (z[rows], x[rows], tx[rows], y[rows], ty[rows], clocks[rows],
                energy[rows], alive, blocked, labels)
