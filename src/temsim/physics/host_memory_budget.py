"""Per-worker host stage admission, separate from CUDA workspace budgets.

This is a planning cap, not a Python allocator, RSS reservation, or replacement
for the operating system's available-memory check. Consumers submit the whole
live host stage they are about to allocate; nested workers may only tighten it.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Integral

from temsim.physics.wave_execution import check_available_memory


_CAP = ContextVar("temsim_host_stage_budget_bytes", default=None)


def active_host_memory_budget_bytes():
    """Return this worker's planning cap, or None outside bounded admission."""
    return _CAP.get()


def _bytes(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return int(value)


@contextmanager
def host_memory_budget(maximum_bytes):
    requested = _bytes(maximum_bytes, "Host stage budget")
    inherited = _CAP.get()
    token = _CAP.set(requested if inherited is None else min(requested, inherited))
    try:
        yield
    finally:
        _CAP.reset(token)


def check_host_allocation(additional_bytes, *, requested_maximum_bytes=None):
    required = _bytes(additional_bytes, "Host allocation", zero=True)
    cap = _CAP.get()
    if requested_maximum_bytes is not None:
        requested = _bytes(requested_maximum_bytes, "Requested host stage budget")
        cap = requested if cap is None else min(cap, requested)
    if cap is not None and required > cap:
        raise MemoryError(f"Host stage requires {required} bytes, exceeding the admitted {cap}-byte host budget")
    check_available_memory(required)


__all__ = ("host_memory_budget", "check_host_allocation", "active_host_memory_budget_bytes")
