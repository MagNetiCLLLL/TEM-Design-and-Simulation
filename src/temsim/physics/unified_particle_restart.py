"""Complete immutable internal state and fail-closed unified prefix proofs."""
from dataclasses import dataclass

import numpy as np

from temsim.physics.unified_spatial_dependencies import spatial_runtime_identity


SCHEMA = "unified-particle-checkpoint-state-v1"


def _frozen(values, dtype=float):
    value = np.asarray(values, dtype=dtype)
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class UnifiedParticleCheckpointState:
    reference_energy_ev: np.ndarray
    reference_phi_v: np.ndarray
    induction_work_ev: np.ndarray
    active: np.ndarray
    blocked_z_mm: np.ndarray
    blocked_key: tuple
    runtime_digest: str
    implementation_digest: str
    emission_time_s: float
    swept_bounds_m: np.ndarray
    spatial_dependency_identities: tuple
    schema: str = SCHEMA

    def __post_init__(self):
        energy, phi = np.asarray(self.reference_energy_ev), np.asarray(self.reference_phi_v)
        work, active = np.asarray(self.induction_work_ev), np.asarray(self.active)
        stop, box = np.asarray(self.blocked_z_mm), np.asarray(self.swept_bounds_m)
        if self.schema != SCHEMA or energy.ndim != 1 or phi.shape != energy.shape:
            raise ValueError("Invalid unified checkpoint reference arrays")
        if work.ndim != 2 or work.shape[1:] != energy.shape or active.shape != work.shape or active.dtype.kind != "b":
            raise ValueError("Unified checkpoint activity and induction arrays must align")
        if stop.shape != energy.shape or len(self.blocked_key) != energy.size or box.shape != (len(work), 2, 3):
            raise ValueError("Unified checkpoint stop and swept-box arrays must align")
        if (np.isinf(energy).any() or np.isinf(phi).any()
                or not np.isfinite(work).all() or np.isinf(stop).any()
                or not np.isfinite(box).all() or np.any(box[:, 0] > box[:, 1])):
            raise ValueError("Unified checkpoint internal state is not finite and ordered")
        if active.any() and (not np.isfinite(energy[np.any(active, axis=0)]).all()
                             or np.any(energy[np.any(active, axis=0)] <= 0.)
                             or not np.isfinite(phi[np.any(active, axis=0)]).all()):
            raise ValueError("Active unified checkpoints require reference energy and potential")
        if any(not isinstance(value, str) for value in self.blocked_key):
            raise ValueError("Unified checkpoint stop labels must be strings")
        if len(self.spatial_dependency_identities) != len(work):
            raise ValueError("Unified checkpoints require one spatial identity per retained row")
        for digest in (self.runtime_digest, self.implementation_digest, *self.spatial_dependency_identities):
            if not isinstance(digest, str) or len(digest) != 64 or set(digest)-set("0123456789abcdef"):
                raise ValueError("Unified checkpoints require exact runtime and implementation identities")
        if not np.isfinite(self.emission_time_s):
            raise ValueError("Unified checkpoint emission time must be finite")
        for name in ("reference_energy_ev", "reference_phi_v", "induction_work_ev", "blocked_z_mm", "swept_bounds_m"):
            object.__setattr__(self, name, _frozen(getattr(self, name)))
        object.__setattr__(self, "active", _frozen(active, bool))
        object.__setattr__(self, "blocked_key", tuple(self.blocked_key))
        object.__setattr__(self, "spatial_dependency_identities", tuple(self.spatial_dependency_identities))

    def restored(self, row, z_mm):
        if isinstance(row, bool) or int(row) != row or not 0 <= row < len(self.active):
            raise ValueError("Unified restart checkpoint row is invalid")
        row = int(row)
        # A downstream loss in the old execution must not kill the particle at
        # an upstream restart. Activity at the exact checkpoint is authoritative.
        past = np.isfinite(self.blocked_z_mm) & (self.blocked_z_mm <= float(z_mm)) & ~self.active[row]
        return {"reference_energy_ev": self.reference_energy_ev.copy(),
            "reference_phi_v": self.reference_phi_v.copy(),
            "induction_work_ev": self.induction_work_ev[row].copy(),
            "active": self.active[row].copy(),
            "blocked_z_mm": np.where(past, self.blocked_z_mm, np.nan),
            "blocked_key": tuple(key if keep else "" for key, keep in zip(self.blocked_key, past)),
            "swept_bounds_m": self.swept_bounds_m[row].copy(),
            "spatial_dependency_identity": self.spatial_dependency_identities[row]}


def validate_unified_checkpoint(checkpoints, plan=None):
    state = getattr(checkpoints, "unified_state", None)
    if type(state) is not UnifiedParticleCheckpointState or state.active.shape != np.shape(checkpoints.x_m):
        raise ValueError("Unified restart lacks complete retained internal state")
    if plan is not None and state.runtime_digest != plan.unified_runtime.digest:
        raise ValueError("Unified checkpoint runtime differs from its executed plan")
    if (np.any(state.active & ~np.isfinite(checkpoints.kinetic_energy_ev))
            or np.any(state.active & ~np.isfinite(checkpoints.flight_time_s))):
        raise ValueError("Active unified restart lacks actual energy or flight time")
    for values in (checkpoints.x_m, checkpoints.y_m, checkpoints.tx_rad, checkpoints.ty_rad):
        if np.any(state.active & ~np.isfinite(values)):
            raise ValueError("Active unified restart lacks finite phase space")
    bounds = state.swept_bounds_m
    if (np.any(bounds[1:, 0] > bounds[:-1, 0]) or np.any(bounds[1:, 1] < bounds[:-1, 1])):
        raise ValueError("Unified checkpoint swept bounds must retain all previous queries")
    for axis, values in ((0, checkpoints.x_m), (1, checkpoints.y_m),
                         (2, np.broadcast_to(checkpoints.z_mm[:, None]*.001, state.active.shape))):
        if np.any(state.active & ((values < bounds[:, 0, axis, None]) | (values > bounds[:, 1, axis, None]))):
            raise ValueError("Unified checkpoint swept bounds omit an active retained point")
    return state


def unified_prefix_matches(old, new, checkpoints, row, old_index, new_index, *, implementation_digest):
    try:
        retained = validate_unified_checkpoint(checkpoints, old)
        if retained.implementation_digest != implementation_digest:
            return False
        if (old.unified_runtime is None or new.unified_runtime is None or old_index != new_index
                or old.reference_momentum_kg_m_s != new.reference_momentum_kg_m_s
                or not np.array_equal(old.z_mm[:old_index+1], new.z_mm[:new_index+1])
                or not np.array_equal(old.step_m[:old_index], new.step_m[:new_index])):
            return False
        bounds = retained.swept_bounds_m[row]
        return (retained.spatial_dependency_identities[row]
                == spatial_runtime_identity(new.unified_runtime, bounds))
    except (AttributeError, TypeError, ValueError, IndexError):
        return False


def merge_unified_checkpoint_states(previous, suffix, keep):
    old, new = previous.unified_state, suffix.unified_state
    if old is None or new is None:
        return None
    for name in ("reference_energy_ev", "reference_phi_v"):
        if not np.array_equal(getattr(old, name), getattr(new, name), equal_nan=True):
            raise ValueError("Unified restart changed its original energy reference")
    if old.implementation_digest != new.implementation_digest or old.emission_time_s != new.emission_time_s:
        raise ValueError("Unified restart changed its implementation or emission epoch")
    return UnifiedParticleCheckpointState(old.reference_energy_ev, old.reference_phi_v,
        np.concatenate((old.induction_work_ev[keep], new.induction_work_ev)),
        np.concatenate((old.active[keep], new.active)), new.blocked_z_mm, new.blocked_key,
        new.runtime_digest, new.implementation_digest, new.emission_time_s,
        np.concatenate((old.swept_bounds_m[keep], new.swept_bounds_m)),
        tuple(np.asarray(old.spatial_dependency_identities)[keep])+new.spatial_dependency_identities)


STATE_ARRAY_FIELDS = ("reference_energy_ev", "reference_phi_v", "induction_work_ev",
                      "active", "blocked_z_mm", "swept_bounds_m")


def encode_checkpoint_state(checkpoints, arrays, *, prefix="unified."):
    """Add plain restart arrays to a bundle and return JSON-only metadata."""
    if checkpoints.unified_state is None:
        return None
    state = validate_unified_checkpoint(checkpoints)
    for name in STATE_ARRAY_FIELDS:
        arrays[prefix+name] = getattr(state, name)
    return {name: getattr(state, name) for name in ("schema", "blocked_key", "runtime_digest",
        "implementation_digest", "emission_time_s", "spatial_dependency_identities")}


def decode_checkpoint_state(metadata, arrays, *, prefix="unified."):
    """Historical bundles lacking internal restart state remain view-only."""
    present = {name for name in arrays if name.startswith(prefix)}
    if metadata is None:
        if present:
            raise ValueError("Unified restart arrays lack required metadata")
        return None
    expected = {prefix+name for name in STATE_ARRAY_FIELDS}
    if present != expected:
        raise ValueError("Unified restart bundle has incomplete internal arrays")
    try:
        return UnifiedParticleCheckpointState(**dict(metadata),
            **{name: arrays[prefix+name] for name in STATE_ARRAY_FIELDS})
    except (TypeError, KeyError) as exc:
        raise ValueError("Invalid unified restart bundle metadata") from exc


def rebind_unified_stops(source_z, target_z):
    """Carry the executed suffix's full stop receipt across history joining."""
    import weakref
    from temsim.physics.unified_field_particle import _STOPS
    receipt = _STOPS.get(id(source_z))
    if receipt is None or receipt[0]() is not source_z:
        raise ValueError("Unified continuation lacks an executed mechanical receipt")
    _STOPS[id(target_z)] = (weakref.ref(target_z, lambda ref, key=id(target_z): _STOPS.pop(key, None)),
                            receipt[1].copy(), receipt[2].copy(), receipt[3])
