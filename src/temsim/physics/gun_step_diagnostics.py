"""Opt-in, context-local observation of the unchanged gun step decisions.

No observer is stored on State or Gun, and no observation returns a numerical
decision. Snapshots are independent immutable copies of pre-boundary DG states,
not transport restart records. Default transport has no observer allocations.
"""
from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


_ACTIVE = ContextVar("gun_step_diagnostics", default=None)


def current_gun_step_diagnostics():
    return _ACTIVE.get()


def _copy(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass
class GunStepDiagnostics:
    """Bounded scalar observations and selected exact input/output arrays."""
    maximum_attempt_records: int = 20000
    snapshot_stride: int = 1
    maximum_snapshots: int = 64
    counts: Counter = field(default_factory=Counter, init=False)
    static_step_seconds: Counter = field(default_factory=Counter, init=False)
    dispatch: Counter = field(default_factory=Counter, init=False)
    snapshots: list = field(default_factory=list, init=False)
    traces: list = field(default_factory=list, init=False)
    source_arrays: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        for value in (self.maximum_attempt_records, self.snapshot_stride, self.maximum_snapshots):
            if type(value) is not int or value < 1:
                raise ValueError("Gun diagnostic capacities and stride must be positive integers")
        self.attempts = deque(maxlen=self.maximum_attempt_records)
        self.started = perf_counter()
        self._attempt = None
        self._step = None
        self._ray_id = None

    @property
    def accepted_steps(self):
        return int(self.counts["accepted_steps"])

    def begin_trace(self, ray_id, weight, phase, launch_energy_ev):
        self.counts["traces"] += 1
        trace = int(self.counts["traces"])
        self._ray_id = _copy(ray_id)
        values = dict(ray_id=ray_id, weight=weight, launch_position_m=phase.position_m,
            launch_momentum_kg_m_per_s=phase.momentum_kg_m_per_s, launch_energy_ev=launch_energy_ev)
        self.source_arrays.update({f"trace_{trace:04d}_{key}": _copy(value) for key, value in values.items()})
        self.traces.append(dict(trace=trace, original_count=len(ray_id),
            original_positive_count=int(np.count_nonzero(np.asarray(weight) > 0)),
            source_fraction=float(np.sum(weight)), initial_phase_time_s=float(phase.time_s)))

    def begin_step(self, phase, dt, active):
        self.counts["surface_step_calls"] += 1
        indices = np.flatnonzero(active)
        self._active_ids = (self._ray_id[indices] if self._ray_id is not None
            and len(self._ray_id) == len(active) else indices)
        z = phase.position_m[active, 2]
        self._step = dict(surface_step=int(self.counts["surface_step_calls"]),
            trace=int(self.counts["traces"]), initial_dt_s=float(dt),
            phase_time_s=float(phase.time_s), active_count=len(indices),
            z_min_m=float(z.min()) if len(z) else None,
            z_max_m=float(z.max()) if len(z) else None)

    def begin_attempt(self, index, dt, cached_full):
        self.counts["attempts"] += 1
        self.counts["cached_full_uses"] += bool(cached_full)
        self._attempt = dict(self._step, attempt=int(index), dt_s=float(dt),
            cached_full=bool(cached_full), accepted=None, static_calls=[], started=perf_counter())

    def record_step_proposal(self, proposal):
        """Copy numerical guidance; the observer never supplies a step decision."""
        self._step["trial_proposal"] = dict(proposal)

    def time_static_step(self, label, function, *args, **kwargs):
        self.counts["static_step_calls"] += 1
        self.counts["static_step_"+label] += 1
        started, returned = perf_counter(), False
        try:
            result = function(*args, **kwargs)
            returned = True
            return result
        finally:
            seconds = perf_counter()-started
            self.static_step_seconds[label] += seconds
            if self._attempt is not None:
                self._attempt["static_calls"].append(dict(kind=label, seconds=seconds, returned=returned))

    def record_dispatch(self, name, used, reason=""):
        """Optional scalar hook for a dispatcher; it never changes selection."""
        self.dispatch[(str(name), bool(used), str(reason))] += 1

    def _gate(self, error, budget):
        error = np.asarray(error)
        with np.errstate(all="ignore"):
            ratio = error/np.asarray(budget)
        bad = np.flatnonzero(~np.isfinite(ratio))
        index = int(bad[0]) if len(bad) else int(np.argmax(ratio))
        bound = np.asarray(budget)
        bound = bound if bound.ndim == 0 else bound[index]
        return dict(maximum_ratio=float(ratio[index]) if not len(bad) else None,
            limiting_original_id=int(self._active_ids[index]),
            error=float(error[index]) if np.isfinite(error[index]) else None,
            budget=float(bound) if np.isfinite(bound) else None,
            all_finite=not bool(len(bad)))

    def record_gates(self, *, momentum_error, momentum_limit, xyz_error, xyz_budget,
                     transverse_error, transverse_limit, xy_error, xy_budget, finite):
        self._attempt["finite"] = bool(finite)
        self._attempt["gates"] = dict(
            total_momentum=self._gate(momentum_error, momentum_limit),
            xyz_position=self._gate(xyz_error, xyz_budget),
            transverse_momentum=self._gate(transverse_error, transverse_limit),
            xy_position=self._gate(xy_error, xy_budget))

    def _finish_attempt(self, **result):
        self._attempt.update(result)
        self._attempt["seconds"] = perf_counter()-self._attempt.pop("started")
        self.attempts.append(self._attempt)
        self._attempt = None

    def record_exception(self, error):
        retryable = "iteration did not converge" in str(error)
        self.counts["implicit_failures" if retryable else "nonretryable_step_errors"] += 1
        self._finish_attempt(accepted=False, retryable_implicit_failure=retryable,
            error_type=type(error).__name__, error=str(error))

    def rejected(self):
        self.counts["gate_rejections"] += 1
        for name, row in self._attempt["gates"].items():
            if not row["all_finite"] or row["maximum_ratio"] > 1:
                self.counts["failed_gate_"+name] += 1
        self._finish_attempt(accepted=False)

    def accepted(self, before, positions, momenta, active, dt):
        self.counts["accepted_steps"] += 1
        self._finish_attempt(accepted=True)
        number = self.accepted_steps
        if (number-1) % self.snapshot_stride or len(self.snapshots) >= self.maximum_snapshots:
            return
        self.snapshots.append(dict(accepted_step=number, **self._step,
            used_dt_s=float(dt), end_phase_time_s=float(before.time_s+dt),
            start_position_m=_copy(before.position_m),
            start_momentum_kg_m_per_s=_copy(before.momentum_kg_m_per_s),
            end_position_m=_copy(positions), end_momentum_kg_m_per_s=_copy(momenta),
            active=_copy(active)))

    def arrays(self):
        values = dict(self.source_arrays)
        for index, row in enumerate(self.snapshots):
            values.update({f"snapshot_{index:05d}_{key}": value for key, value in row.items()
                           if isinstance(value, np.ndarray)})
        return values

    def report(self):
        snapshots = [{key: value for key, value in row.items() if not isinstance(value, np.ndarray)}
                     for row in self.snapshots]
        return dict(schema="gun-step-observer-v1", counts=dict(self.counts),
            elapsed_seconds=perf_counter()-self.started,
            static_step_seconds=dict(self.static_step_seconds),
            dispatch=[dict(name=name, used=used, reason=reason, count=count)
                for (name, used, reason), count in sorted(self.dispatch.items())],
            traces=list(self.traces), attempts=list(self.attempts), snapshots=snapshots,
            attempt_record_capacity=self.maximum_attempt_records,
            attempt_records_dropped=max(0, self.counts["attempts"]-len(self.attempts)),
            snapshot_stride=self.snapshot_stride, snapshot_capacity=self.maximum_snapshots,
            scope="Read-only timing and copies of actual pre-boundary DG steps. No aperture/stop processing, no restart-state admission, no field replacement, tolerance change or source modification.")


@contextmanager
def observe_gun_steps(diagnostics):
    if type(diagnostics) is not GunStepDiagnostics:
        raise TypeError("Use the closed gun-step diagnostics observer")
    token = _ACTIVE.set(diagnostics)
    try:
        yield diagnostics
    finally:
        _ACTIVE.reset(token)
