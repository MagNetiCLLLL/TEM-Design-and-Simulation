"""Shared acceleration of retained potential samples, without new field laws."""
from dataclasses import dataclass
from collections import OrderedDict
from threading import RLock

import numpy as np

from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot, _times
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


_CACHE = OrderedDict()
_LOCK = RLock()
_MAXIMUM_CACHED_SNAPSHOTS = 8


@dataclass(frozen=True, slots=True, init=False)
class UnifiedPotentialSampler:
    """Immutable dispatch; all uncompiled providers retain their exact sample."""
    snapshot: object
    static_snapshot: ElectromagneticPotentialSnapshot
    compiled: object
    remaining: tuple
    backend: str

    def __init__(self,*args,**kwargs):
        raise TypeError("Use compile_unified_potential_sampler with a frozen snapshot")

    @property
    def snapshot_digest(self):
        return self.snapshot.numerical_digest

    def sample(self, positions_m, time_s=None):
        dynamic = type(self.snapshot) is DynamicPotentialSnapshot
        if dynamic and time_s is None:
            raise ValueError("Dynamic potential sampling requires absolute laboratory time")
        if time_s is not None:
            np.broadcast_to(_times(time_s),np.shape(positions_m)[:-1])
        if self.compiled is None:
            result = self.static_snapshot.sample(positions_m)
        else:
            result = self.compiled.sample(positions_m)
            for row in self.remaining:
                for key,value in row.potential.sample(positions_m).items():
                    if key in result:
                        result[key] += value
        if dynamic:
            result.update(dA_dt_t_m_per_s=np.zeros_like(result["A_t_m"]),
                          d2A_dt2_t_m_per_s2=np.zeros_like(result["A_t_m"]),
                          dphi_dt_v_per_s=np.zeros_like(result["phi_v"]))
            for coil in self.snapshot.dynamic_coils:
                for key,value in coil.sample(positions_m,time_s).items():
                    result[key] += value
        if any(not np.isfinite(value).all() for value in result.values()):
            raise ValueError("Combined unified potential exceeds the finite numerical range")
        return result

    __call__ = sample


def compile_unified_potential_sampler(snapshot, *, prefer_plane=False):
    """Compile only known native magnetic terms; preserve E, domains and gauge.

    Plane aggregation is optional and retains the existing strict native
    fallback for poses, non-planar queries and points near domain boundaries.
    Electric/FEM providers are not replaced, zeroed, extended or refitted.
    """
    if type(snapshot) not in (ElectromagneticPotentialSnapshot,DynamicPotentialSnapshot):
        raise ValueError("Unified potential sampling requires an exact immutable snapshot")
    if type(prefer_plane) is not bool:
        raise ValueError("Plane sampling preference must be Boolean")
    key = (snapshot.numerical_digest,prefer_plane)
    with _LOCK:
        previous = _CACHE.get(key)
        if previous is not None:
            _CACHE.move_to_end(key)
            return previous
    static = snapshot.static_snapshot if type(snapshot) is DynamicPotentialSnapshot else snapshot
    selected = tuple(row for row in static.contributions
        if type(row.potential) is HighOrderRoundLensPotential
        or (type(row.potential) is AnalyticPotentialDomain and not row.potential._round))
    compiled = None
    remaining = tuple(row for row in static.contributions if row not in selected)
    if selected:
        native = ElectromagneticPotentialSnapshot(static.scene,selected,static.gauge)
        if prefer_plane:
            from temsim.physics.potential_plane_sampler import compile_plane_sampler
            compiled = compile_plane_sampler(native)
        else:
            from temsim.physics.potential_native_sampler import compile_native_sampler
            compiled = compile_native_sampler(native)
    result = object.__new__(UnifiedPotentialSampler)
    for key,value in (("snapshot",snapshot),("static_snapshot",static),("compiled",compiled),
                      ("remaining",remaining),("backend","retained-providers" if compiled is None else compiled.backend)):
        object.__setattr__(result,key,value)
    cache_key = (snapshot.numerical_digest,prefer_plane)
    with _LOCK:
        result = _CACHE.setdefault(cache_key,result)
        _CACHE.move_to_end(cache_key)
        while len(_CACHE)>_MAXIMUM_CACHED_SNAPSHOTS:
            _CACHE.popitem(last=False)
    return result
