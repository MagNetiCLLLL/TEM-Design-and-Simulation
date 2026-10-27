"""Scoped optical status messages; never part of physical state or cache keys."""
from contextlib import contextmanager
from contextvars import ContextVar
from operator import index

_CALLBACK = ContextVar("optical_transport_progress", default=None)


@contextmanager
def transport_progress(callback):
    token = _CALLBACK.set(callback)
    try:
        yield
    finally:
        _CALLBACK.reset(token)


def transport_progress_active():
    return _CALLBACK.get() is not None


@contextmanager
def transport_progress_segment(completed_segments, total_segments, *, label):
    """Map one nested solver into a declared sequence of equal work units.

    Fractions are measured child progress, not time estimates. An unknown
    child total reports only already completed units; a successful context
    exit completes this unit. Neither child completion nor this local mapping
    declares a workflow stage complete. Exceptions restore the parent scope
    without publishing fictitious completion.
    """
    from temsim.calculation_progress import LocalProgressLabel
    if isinstance(completed_segments, bool) or isinstance(total_segments, bool):
        raise ValueError("Progress segments need integer counts")
    completed, total = index(completed_segments), index(total_segments)
    if not 0 <= completed < total:
        raise ValueError("Progress segment must be inside a positive declared total")
    parent = _CALLBACK.get()
    # Fixed integer display units preserve monotonically comparable fractions
    # when different traces use different integration grids. Floor rounding
    # cannot present the still-running final trace as complete.
    scale = 1_000_000
    accepted = 0
    prefix = f"{label} | {completed+1}/{total}"
    def publish(fraction, detail):
        if parent is not None:
            parent(LocalProgressLabel(f"{prefix} | {detail}",
                completed*scale+fraction, total*scale))
    def mapped(detail):
        nonlocal accepted
        local = getattr(detail, "local_progress", None)
        if local is not None and local.total is not None:
            accepted = max(accepted, min(scale-1, local.completed*scale//local.total))
        publish(accepted, str(detail))
    publish(0, "Starting")
    token = _CALLBACK.set(mapped if parent is not None else None)
    try:
        yield
    except BaseException:
        raise
    else:
        publish(scale, "Completed")
    finally:
        _CALLBACK.reset(token)


def report_transport_progress(label, *, completed=None, total=None):
    callback = _CALLBACK.get()
    if callback is not None:
        if completed is not None:
            from temsim.calculation_progress import LocalProgressLabel
            label = LocalProgressLabel(label, completed, total)
        callback(label)
