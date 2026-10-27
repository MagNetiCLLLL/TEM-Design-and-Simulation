"""Optional double-precision device execution for the tip-origin wave chain.

The device is an execution choice, never another source or numerical model.
Public checkpoints remain host arrays. A GPU job owns its allocation pool and
executes one mode at a time; only unavailable hardware or exhausted memory may
fall back, and a failed operator must be rerun from its unchanged input.
"""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import sys
from threading import RLock

import numpy as np

from temsim.physics.compute_backend import (
    GPUExecutionError, WAVE_BACKEND_CUPY, choose_wave_backend, cupy_module,
    gpu_retry_reason, normalise_backend as _normalise_backend,
)


_ACTIVE = ContextVar("coherent_wave_device", default=None)
_REUSABLE = ContextVar("coherent_wave_reusable_device", default=None)
_DEVICE_LOCK = RLock()
_RETIRED_POOLS = []
PRECISION = "complex128 / float64"


def normalise_backend(value):
    from temsim.physics.compute_backend import BACKEND_CHOICES
    aliases = {"cpu": "CPU", "auto": "Auto", "gpu": "CUDA GPU", "cuda": "CUDA GPU",
               "cupy": "CUDA GPU", "numpy": "CPU", "numba": "Numba CPU",
               "prefer_gpu": "Prefer GPU", "require_gpu": "Require GPU",
               "numpy cpu": "CPU", "cupy cuda": "CUDA GPU"}
    aliases.update({name.lower(): name for name in BACKEND_CHOICES})
    name = str(value).strip().lower()
    if name not in aliases:
        raise ValueError(f"Unknown coherent compute policy: {value!r}")
    return _normalise_backend(aliases[name])


def array_module(value):
    if hasattr(value, "__cuda_array_interface__"):
        import cupy
        return cupy
    return np


def to_host(value):
    xp = array_module(value)
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def max_abs(value):
    return float(array_module(value).max(abs(value))) if value.size else 0.


def release_device_memory():
    """Drain completed pools after caller-owned GPU temporaries have expired.

    CuPy allocations reference their pool weakly. In particular, a split
    block cannot be recovered if its pool dies before all its arrays. Retain
    pools with live arrays and release only their unused blocks; live device
    outputs stay valid. Call again at the public job boundary.
    """
    with _DEVICE_LOCK:
        retained = []
        for cp, device_id, pool in _RETIRED_POOLS:
            with cp.cuda.Device(device_id):
                pool.free_all_blocks()
                if pool.used_bytes():
                    retained.append((cp, device_id, pool))
        _RETIRED_POOLS[:] = retained


def _retire_pool(cp, pool):
    pool.free_all_blocks()
    if pool.used_bytes():
        _RETIRED_POOLS.append((cp, int(cp.cuda.runtime.getDevice()), pool))


@dataclass
class WaveDevice:
    xp: object = np
    backend: str = "numpy"
    fallback_reason: str | None = None
    maximum_working_bytes: int = 0
    pool: object = None
    uncached_fft_plans: int = 0
    largest_uncached_fft_plan_bytes: int = 0
    workspace_shape: tuple | None = None

    def evidence(self):
        result = {"compute_backend": self.backend, "numeric_precision": PRECISION,
                  "fallback_reason": self.fallback_reason}
        if self.uncached_fft_plans:
            result.update(uncached_fft_plans=self.uncached_fft_plans,
                          largest_uncached_fft_plan_bytes=self.largest_uncached_fft_plan_bytes)
        return result


@contextmanager
def _temporary_oversized_fft_plans(cp, session, plan_cache):
    """A cache quota limits retention, not an already admitted FFT execution.

    CuPy creates a cuFFT plan before inserting it into its LRU. Its insertion
    raises if that one plan exceeds the cache's retention quota, even though
    the workspace was successfully allocated under our unchanged pool limit.
    Let that plan remain local to the FFT call. All allocation, cuFFT and other
    cache errors still propagate normally. Small plans retain their usual LRU.

    The CuPy hook is process-visible, so only this context's session and actual
    thread-local cache receive the proxy; unrelated callers see the original.
    """
    implementation = cp.fft._fft
    original_get_cache = implementation.get_plan_cache

    class RetentionCache:
        def __getattr__(self, name):
            return getattr(plan_cache, name)

        def __setitem__(self, key, plan):
            try:
                plan_cache[key] = plan
            except RuntimeError as error:
                if str(error) != "the plan memsize is too large":
                    raise
                session.uncached_fft_plans += 1
                work_area = getattr(plan, "work_area", None)
                size = int(getattr(getattr(work_area, "mem", None), "size", 0))
                session.largest_uncached_fft_plan_bytes = max(
                    session.largest_uncached_fft_plan_bytes, size)

    proxy = RetentionCache()

    def get_cache():
        cache = original_get_cache()
        return proxy if _ACTIVE.get() is session and cache is plan_cache else cache

    implementation.get_plan_cache = get_cache
    try:
        yield
    finally:
        if implementation.get_plan_cache is get_cache:
            implementation.get_plan_cache = original_get_cache


def check_device_memory(required_bytes):
    """Check total estimated live device work, including FFT scratch space."""
    active = _ACTIVE.get()
    if active is None or active.backend != "cupy":
        return
    required = int(required_bytes)
    free, _ = active.xp.cuda.runtime.memGetInfo()
    # Owned allocations are already subtracted from CUDA's free count. The
    # estimate describes the whole operation, including those live arrays.
    owned = 0 if active.pool is None else active.pool.total_bytes()
    available = max(0, int(free) + int(owned) - 256 * 1024**2)
    allowed = min(active.maximum_working_bytes, available)
    if required > allowed:
        raise GPUExecutionError("out_of_memory",
            f"Coherent GPU work needs {required} bytes; device working limit "
            f"{active.maximum_working_bytes}, currently available {available} bytes")


class _ReusableDevice:
    """Keep resources, but never leave a device active during host operators."""

    def __init__(self, maximum_working_bytes):
        self.maximum_working_bytes = maximum_working_bytes
        self.stack = ExitStack()
        self.session = None
        self.suspended_token = None

    def acquire(self, requested, *, work_items, acceleration_enabled):
        if self.session is None:
            previous = _ACTIVE.get()
            token = _REUSABLE.set(None)
            try:
                session = self.stack.enter_context(device_scope(requested,
                    maximum_working_bytes=self.maximum_working_bytes,
                    work_items=work_items, acceleration_enabled=acceleration_enabled))
            finally:
                _REUSABLE.reset(token)
            if session.backend != "cupy":
                self.stack.close()
                return session
            self.session = session
            self.suspended_token = _ACTIVE.set(previous)
        return self.session

    def close(self):
        if self.suspended_token is not None:
            _ACTIVE.reset(self.suspended_token)
        self.stack.close()


@contextmanager
def reusable_device_scope(*, maximum_working_bytes=24*1024**3):
    """Lazily reuse one private CUDA pool across a serial host mode segment.

    Each operator still selects its own backend from its actual work size.
    CPU operations and host grid admission see no active CUDA device. The
    owner closes on cancellation or failure; public results remain host arrays.
    Nested operators may tighten, but never enlarge, the owner's hard cap.
    """
    if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0:
        raise ValueError("Reusable device working bytes must be a positive integer")
    if _REUSABLE.get() is not None or _ACTIVE.get() is not None:
        raise ValueError("Reusable device scopes require a non-nested host segment")
    owner = _ReusableDevice(maximum_working_bytes)
    token = _REUSABLE.set(owner)
    try:
        yield owner
    finally:
        _REUSABLE.reset(token)
        primary = sys.exc_info()[1]
        try:
            owner.close()
        except Exception as cleanup:
            if primary is None:
                raise
            primary.add_note(f"Reusable device cleanup also failed: {type(cleanup).__name__}: {cleanup}")


def _prepare_reused_workspace(session, shape, required_bytes):
    """Release old plans and unused blocks without invalidating live arrays."""
    shape = None if shape is None else tuple(shape)
    changed = (shape is not None and session.workspace_shape is not None
               and shape != session.workspace_shape)
    pool = session.pool
    pressure = (pool is not None and pool.total_bytes() > 0
                and (pool.total_bytes() > .8*session.maximum_working_bytes
                or pool.used_bytes()+int(required_bytes) > session.maximum_working_bytes))
    if changed or pressure:
        session.xp.cuda.get_current_stream().synchronize()
        session.xp.fft.config.get_plan_cache().clear()
        if pool is not None:
            pool.free_all_blocks()
    if shape is not None:
        session.workspace_shape = shape


@contextmanager
def _borrow_device(session, *, maximum_working_bytes, required_bytes, shape):
    previous_limit = session.maximum_working_bytes
    nested_limit = min(previous_limit, int(maximum_working_bytes))
    token = _ACTIVE.set(session)
    session.maximum_working_bytes = nested_limit
    try:
        _prepare_reused_workspace(session, shape, required_bytes)
        if session.pool is not None:
            session.pool.set_limit(size=nested_limit)
        check_device_memory(required_bytes)
        yield session
    finally:
        session.maximum_working_bytes = previous_limit
        primary = sys.exc_info()[1]
        try:
            if session.pool is not None:
                session.pool.set_limit(size=previous_limit)
        except Exception as cleanup:
            if primary is None:
                raise
            primary.add_note(f"Device limit restoration also failed: {type(cleanup).__name__}: {cleanup}")
        finally:
            _ACTIVE.reset(token)


@contextmanager
def device_scope(requested="cpu", *, maximum_working_bytes=24*1024**3,
                 work_items=0, required_bytes=0, acceleration_enabled=True,
                 workspace_shape=None):
    """Select and admit a bounded operator. Errors inside it are not hidden.

    The caller owns retry from its original host input if execution (rather
    than admission) exhausts memory. Require GPU is never weakened.
    """
    requested = normalise_backend(requested)
    active = _ACTIVE.get()
    if active is not None and active.backend == "cupy":
        selected, reason = choose_wave_backend(requested, acceleration_enabled=acceleration_enabled,
                                               work_items=int(work_items))
        if selected != WAVE_BACKEND_CUPY:
            # A failed GPU mode retries from its host checkpoint while the
            # serial segment still owns its reusable GPU pool. Honour that
            # explicit CPU scope, including its allocation checks.
            session = WaveDevice(fallback_reason=reason)
            token = _ACTIVE.set(session)
            try:
                yield session
            finally:
                _ACTIVE.reset(token)
            return
        with _borrow_device(active, maximum_working_bytes=maximum_working_bytes,
                required_bytes=required_bytes, shape=workspace_shape) as session:
            yield session
        return
    release_device_memory()
    selected, reason = choose_wave_backend(requested, acceleration_enabled=acceleration_enabled,
                                           work_items=int(work_items))
    if selected != WAVE_BACKEND_CUPY:
        yield WaveDevice(fallback_reason=reason)
        return
    owner = _REUSABLE.get()
    if owner is not None:
        session = owner.acquire(requested, work_items=int(work_items),
                                acceleration_enabled=acceleration_enabled)
        if session.backend != "cupy":
            yield session
            return
        with _borrow_device(session, maximum_working_bytes=maximum_working_bytes,
                required_bytes=required_bytes, shape=workspace_shape) as borrowed:
            yield borrowed
        return
    with _DEVICE_LOCK:
        session = None
        token = None
        plan_cache = None
        plan_limits = None
        try:
            cp = cupy_module()
            pool = cp.cuda.MemoryPool()
            pool.set_limit(size=int(maximum_working_bytes))
            session = WaveDevice(cp, "cupy", None, int(maximum_working_bytes), pool)
            token = _ACTIVE.set(session)
            check_device_memory(required_bytes)
            # cuFFT plans own work buffers beyond an array's lifetime. Keep
            # only this stage's bounded plans; otherwise old private pools
            # remain alive across successive shapes and escape the budget.
            plan_cache = cp.fft.config.get_plan_cache()
            plan_limits = (plan_cache.get_size(), plan_cache.get_memsize())
            plan_cache.clear()
            plan_cache.set_size(4)
            plan_cache.set_memsize(min(2*1024**3, int(maximum_working_bytes)//8))
        except Exception as error:
            if token is not None:
                _ACTIVE.reset(token)
            if plan_cache is not None and plan_limits is not None:
                plan_cache.clear()
                plan_cache.set_size(plan_limits[0])
                plan_cache.set_memsize(plan_limits[1])
            if session is not None and session.pool is not None:
                _retire_pool(session.xp, session.pool)
            reason = gpu_retry_reason(error, requested, stage="wave_device_admission")
            yield WaveDevice(fallback_reason=reason)
            return
        try:
            with cp.cuda.using_allocator(pool.malloc), _temporary_oversized_fft_plans(cp, session, plan_cache):
                yield session
                cp.cuda.get_current_stream().synchronize()
        finally:
            _ACTIVE.reset(token)
            if plan_cache is not None:
                plan_cache.clear()
                plan_cache.set_size(plan_limits[0])
                plan_cache.set_memsize(plan_limits[1])
            _retire_pool(cp, pool)
