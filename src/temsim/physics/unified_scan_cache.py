"""Bounded memory/disk cache for complete shared-field calibration responses.

Only completed response measurements are retained. Entries are private deep
copies, never mutable calibration drives or particle results. Concurrent
requests for one exact dependency key share a single measurement; cancellation
and failure release waiting callers without publishing a partial response.
"""
from collections import OrderedDict
from copy import deepcopy
from threading import Event, RLock

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics import unified_scan_disk as disk


_MAXIMUM_RESPONSES = 16
_RESPONSES = OrderedDict()
_PENDING = {}
_LOCK = RLock()
_GENERATION = 0


def response_dependencies(runtime, bounds_m=None):
    """Identify the actual providers, not unrelated source/UI scene metadata.

    A cached complete set of executed RK queries may prove an unchanged local
    dependency region. Without that receipt, retain every provider and piece;
    a component's nominal axial support is never used to truncate a Gaussian.
    """
    if bounds_m is not None:
        from temsim.physics.unified_spatial_dependencies import spatial_runtime_payload
        bounds = np.asarray(bounds_m, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
            raise ValueError("Scan response requires finite executed-query bounds")
        payload = spatial_runtime_payload(runtime, bounds)
        return {"kind": "executed-query-box", "bounds_m": bounds.tolist(),
                "payload_digest": json_digest(payload)}
    geometry = runtime.geometry.to_dict()
    geometry.pop("source_scene_geometry_digest", None)
    geometry.pop("digest", None)
    snapshot = runtime.snapshot
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        # Only the explicitly static private DC reference can omit the raw
        # source identity. Unknown time laws remain bound in their entirety.
        payload = runtime.to_dict()
    else:
        payload = {"providers": [(row.source_id, row.owner_ids, row.kind, row.potential.digest)
                                  for row in snapshot.contributions],
                   "gauge": snapshot.gauge.digest, "geometry": geometry}
    return {"kind": "complete-frozen-runtime", "payload_digest": json_digest(payload)}


def matching_response_dependencies(value, runtime):
    record = value.get("receipt", {}).get("response_dependencies")
    if not isinstance(record, dict):
        return False
    return response_dependencies(runtime, record.get("bounds_m")) == record


def clear_scan_response_cache(*, include_disk=True):
    """Clear memory and, by default, this storage namespace's disk records.

    ``include_disk=False`` models a fresh process without removing its saved
    responses. Every clear also invalidates outstanding deferred publications.
    """
    global _GENERATION
    with _LOCK:
        _RESPONSES.clear()
        _GENERATION += 1
        if include_disk:
            disk.discard_response(directory=disk.cache_directory())


def discard_scan_response(key):
    """Discard a measured response when its enclosing calibration failed."""
    with _LOCK:
        _RESPONSES.pop((disk.cache_namespace(), key), None)
        disk.discard_response(key, disk.cache_directory())


def _freeze_arrays(value):
    if isinstance(value, np.ndarray):
        return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
    if isinstance(value, dict):
        return {key: _freeze_arrays(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_freeze_arrays(item) for item in value)
    if isinstance(value, list):
        return [_freeze_arrays(item) for item in value]
    return deepcopy(value)


def _returned(value, key, source):
    result = deepcopy(value)
    result["receipt"] = dict(result["receipt"])
    result["receipt"]["response_cache"] = {
        "key": key, "hit": source != "computed", "source": source,
        "scope": "Exact frozen field, geometry and response-numerics dependencies; original measurement receipts retained",
    }
    return result


def commit_scan_response(key, check_cancelled=lambda: None):
    """Publish after an enclosing calibration has validated all its drives."""
    identity = (disk.cache_namespace(), key)
    with _LOCK:
        entry = _RESPONSES.get(identity)
        if entry is None:
            return False
        request_key, value, expected, generation, published = entry
        if generation != _GENERATION or expected is None:
            return False
        try:
            check_cancelled()
            if not published:
                published = disk.write_response(request_key, key, value, expected,
                                                 disk.cache_directory(), check_cancelled)
                _RESPONSES[identity] = request_key, value, expected, generation, published
            check_cancelled()
        except BaseException:
            discard_scan_response(key)
            raise
        return published


def _compatible(value, compatible):
    try:
        return compatible is None or compatible(value)
    except (ValueError, TypeError, KeyError, OverflowError):
        # A stale/corrupt dependency record is not a physics execution error.
        return False


def _retain(identity, request_key, frozen, expected, generation, published=False):
    _RESPONSES[identity] = (request_key, frozen, deepcopy(expected), generation, published)
    _RESPONSES.move_to_end(identity)
    while len(_RESPONSES) > _MAXIMUM_RESPONSES:
        _RESPONSES.popitem(last=False)


def cached_scan_response(key, compute, check_cancelled, *, compatible=None,
                         expected=None, defer_disk=False):
    """Return an independent complete response for a content-addressed key."""
    if not disk._digest(key):
        raise ValueError("Scan response cache requires a complete dependency digest")
    namespace, directory = disk.cache_namespace(), disk.cache_directory()
    request_identity = namespace, key
    while True:
        check_cancelled()
        with _LOCK:
            for identity, (request_key, value, stored_expected, generation, _) in reversed(tuple(_RESPONSES.items())):
                if (identity[0] == namespace and request_key == key and expected == stored_expected
                        and _compatible(value, compatible)):
                    _RESPONSES.move_to_end(identity)
                    check_cancelled()
                    if not defer_disk:
                        commit_scan_response(identity[1], check_cancelled)
                    return _returned(value, identity[1], "memory")
            pending = _PENDING.get(request_identity)
            if pending is None:
                pending = Event()
                _PENDING[request_identity] = pending
                generation = _GENERATION
                break
        # Waiting remains cancellable without cancelling another caller's job.
        pending.wait(.05)
    try:
        if expected is not None:
            for entry_key, value in disk.read_candidates(key, expected, directory):
                check_cancelled()
                if _compatible(value, compatible):
                    frozen = _freeze_arrays(value)
                    with _LOCK:
                        if generation == _GENERATION:
                            _retain((namespace, entry_key), key, frozen, expected, generation, True)
                    check_cancelled()
                    return _returned(frozen, entry_key, "disk")
        value = compute()
        check_cancelled()
        frozen = _freeze_arrays(value)
        dependency = frozen.get("receipt", {}).get("response_dependencies")
        entry_key = key if dependency is None else json_digest((key, dependency))
        with _LOCK:
            if generation == _GENERATION:
                _retain((namespace, entry_key), key, frozen, expected, generation)
                if not defer_disk:
                    try:
                        commit_scan_response(entry_key, check_cancelled)
                    except BaseException:
                        discard_scan_response(entry_key)
                        raise
        return _returned(frozen, entry_key, "computed")
    finally:
        with _LOCK:
            if _PENDING.get(request_identity) is pending:
                _PENDING.pop(request_identity)
            pending.set()
