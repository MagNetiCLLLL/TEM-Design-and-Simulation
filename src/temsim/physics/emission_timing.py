"""A shared absolute emission epoch, separate from elapsed electron flight time.

An observation at one epoch is not a raster exposure or a temporal wavepacket.
Executors add their retained ray/mode flight time exactly once when sampling a
time-dependent field. Negative epochs are valid; elapsed flight time is a
different quantity with its own non-negative contract.
"""
import math
from numbers import Real


def validate_emission_time_s(value):
    """Return a finite real epoch without coercing text or Boolean controls."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Tip emission time must be a finite real number of seconds")
    try:
        epoch = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Tip emission time must be a finite real number of seconds") from error
    if not math.isfinite(epoch):
        raise ValueError("Tip emission time must be a finite real number of seconds")
    return epoch


def resolve_emission_time_s(state, explicit_time_s=None):
    """Resolve an explicit wave epoch or the captured instrument's epoch."""
    value = getattr(state, "simulation_time_s", 0.) if explicit_time_s is None else explicit_time_s
    return validate_emission_time_s(value)


def require_matching_emission_time(state, request):
    """Reject a particle/wave pair with different epochs before starting it."""
    epoch = resolve_emission_time_s(state)
    wave_epoch = resolve_emission_time_s(state, getattr(request, "tip_time_s", None))
    if wave_epoch != epoch:
        raise ValueError("Paired particle and coherent calculations require the same tip emission time; "
                         "the explicit wave time differs from the captured instrument time")
    return epoch
