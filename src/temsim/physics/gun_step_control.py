"""Trace-local trial-step feedback, independent of physical acceptance.

The cube-root expression is a deliberately damped feedback gain. It does not
assert a cubic error law for the piecewise production fields or their budgets.
Every proposed step still requires the caller's unchanged DG accuracy gates.
"""
from __future__ import annotations

import math
from numbers import Real
import sys

import numpy as np


SURFACE_STEP_PROPOSAL_SCHEMA = "surface-dg-bounded-feedback-proposal-v2"
_MAXIMUM_FLOAT = sys.float_info.max
_MINIMUM_FLOAT = math.ulp(0.)


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if (not math.isfinite(result) or (positive and result <= 0.)
            or (nonnegative and result < 0.)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValueError(f"{label} must be a finite {qualifier}real number")
    return result


def _mask(active):
    try:
        value = np.asarray(active)
    except (TypeError, ValueError) as error:
        raise ValueError("Active population must be a one-dimensional boolean mask") from error
    if value.ndim != 1 or value.dtype != np.dtype(bool):
        raise ValueError("Active population must be a one-dimensional boolean mask")
    result = value.copy()
    result.setflags(write=False)
    return result


def _clock_after(time_s, dt_s):
    result = time_s+dt_s
    if not math.isfinite(result) or result <= time_s:
        raise ValueError("Step must advance a finite population clock")
    return result


def _feedback_gain(ratio):
    # Bound before exponentiation, including zero and subnormal input ratios.
    # A step that passed the caller's gates must not shrink indefinitely when
    # a piecewise/noise-limited error ratio stays near one independently of h.
    if ratio <= (.9/1.25)**3:
        return 1.25
    if ratio >= .9**3:
        return 1.
    return min(1.25, max(1., .9*math.exp(-math.log(ratio)/3.)))


def _scaled_step(dt_s, factor):
    if factor > 1. and dt_s > _MAXIMUM_FLOAT/factor:
        return _MAXIMUM_FLOAT
    result = dt_s*factor
    # There is no representable positive value below the first subnormal.
    # Saturation is only a suggestion; it cannot accept a failed DG step.
    return max(result, _MINIMUM_FLOAT)


class SurfaceStepPredictor:
    """Remember accepted steps without changing fields, gates or populations.

    Both methods receive the *starting* phase clock. ``accepted`` records the
    expected next clock as that exact binary64 start-plus-duration operation.
    Providers are opaque strong references; neither their equality callbacks
    nor any field evaluator is invoked. Create one instance per actual trace.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Explicitly discard scheduling history for a new trace/context."""
        self._have_context = False
        self._magnetic = self._electric = self._active = None
        self._expected_time_s = None
        self._last_accepted_dt_s = self._suggested_dt_s = None
        self._suppress_growth = False
        self._last_proposal = dict(schema=SURFACE_STEP_PROPOSAL_SCHEMA,
            physics_cap_s=None, trial_dt_s=None, last_accepted_dt_s=None,
            next_suggested_dt_s=None, context_changed=False,
            context_change_reasons=(), active_set_changed=False,
            growth_suppressed=False)

    @property
    def last_proposal(self):
        """Return an independent scalar record of the most recent proposal."""
        return dict(self._last_proposal)

    def _changes(self, time_s, active, magnetic, electric):
        reasons = []
        if not self._have_context:
            reasons.append("new_trace")
        else:
            if magnetic is not self._magnetic or electric is not self._electric:
                reasons.append("providers_changed")
            if time_s != self._expected_time_s:
                reasons.append("clock_discontinuity")
        reset_context = bool(reasons)
        changed_active = (self._have_context and not reset_context
                          and not np.array_equal(active, self._active))
        return tuple(reasons), reset_context, changed_active

    def _adopt(self, time_s, active, magnetic, electric):
        self._have_context = True
        self._expected_time_s = time_s
        self._active = active
        self._magnetic, self._electric = magnetic, electric

    def propose(self, cap_s, *, phase_time_s, active, magnetic, electric):
        """Return a positive trial no larger than the existing physical cap."""
        cap = _number(cap_s, "Physical time-step cap", positive=True)
        time = _number(phase_time_s, "Starting phase clock")
        mask = _mask(active)
        reasons, reset_context, active_changed = self._changes(time, mask, magnetic, electric)
        last = None if reset_context else self._last_accepted_dt_s
        suggestion = None if reset_context else self._suggested_dt_s
        suppress = False if reset_context else self._suppress_growth or active_changed
        trial = cap if suggestion is None else min(cap, suggestion)
        if suppress and last is not None:
            trial = min(trial, last)
        _clock_after(time, trial)
        # Commit only after validating the full proposal; bad input leaves the
        # last valid scheduling context intact.
        if reset_context:
            self._last_accepted_dt_s = self._suggested_dt_s = None
        self._suppress_growth = suppress
        self._adopt(time, mask, magnetic, electric)
        self._last_proposal = dict(schema=SURFACE_STEP_PROPOSAL_SCHEMA,
            physics_cap_s=cap, trial_dt_s=trial, last_accepted_dt_s=last,
            next_suggested_dt_s=suggestion,
            context_changed=reset_context or active_changed,
            context_change_reasons=(*reasons, *(("active_set_changed",) if active_changed else ())),
            active_set_changed=active_changed, growth_suppressed=suppress)
        return trial

    def accepted(self, dt_s, max_normalized_error, *, phase_time_s, active,
                 magnetic, electric, retried=False):
        """Record caller acceptance and guide its next trial; return no verdict."""
        dt = _number(dt_s, "Accepted time step", positive=True)
        ratio = _number(max_normalized_error, "Maximum normalized error", nonnegative=True)
        time = _number(phase_time_s, "Starting phase clock")
        if not isinstance(retried, (bool, np.bool_)):
            raise ValueError("Retry flag must be boolean")
        mask = _mask(active)
        next_time = _clock_after(time, dt)
        _, reset_context, active_changed = self._changes(time, mask, magnetic, electric)
        suppress = bool(retried or active_changed
                        or (self._suppress_growth and not reset_context))
        # Only the existing physical caps and actual gate/implicit rejections
        # reduce accepted durations. This feedback suggests bounded growth.
        gain = 1. if suppress else _feedback_gain(ratio)
        self._last_accepted_dt_s = dt
        self._suggested_dt_s = _scaled_step(dt, gain)
        self._suppress_growth = False
        self._adopt(next_time, mask, magnetic, electric)
