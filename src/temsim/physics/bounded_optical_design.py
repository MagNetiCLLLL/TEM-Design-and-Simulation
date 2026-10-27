"""Bounded diagnostic optimization on isolated captured instruments.

The diagnostic receives a private trial state restored from a complete input
snapshot. It must leave that state unchanged. Results are proposals only: no
live setting, profile, fitted aberration coefficient or file is applied.
"""
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import least_squares

from temsim.immutable_json import freeze_json, json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument


@dataclass(frozen=True, slots=True)
class DesignParameter:
    path: tuple[str, ...]
    lower: float
    upper: float
    difference_step: float | None = None

    def __post_init__(self):
        if (not isinstance(self.path, (tuple, list)) or not self.path
                or any(type(name) is not str or not name.isidentifier() or name.startswith("_") for name in self.path)):
            raise ValueError("Design parameters require public attribute paths, not executable expressions")
        if (any(isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value)
                for value in (self.lower, self.upper)) or self.upper <= self.lower):
            raise ValueError("Design parameter bounds must be finite and increasing")
        object.__setattr__(self, "path", tuple(self.path))
        object.__setattr__(self, "lower", float(self.lower))
        object.__setattr__(self, "upper", float(self.upper))
        step = (self.upper-self.lower)*1e-5 if self.difference_step is None else self.difference_step
        if (isinstance(step, bool) or not isinstance(step, Real) or not np.isfinite(step)
                or not 0 < step <= self.upper-self.lower):
            raise ValueError("Design finite-difference step must be positive and within the parameter range")
        object.__setattr__(self, "difference_step", float(step))

    def owner(self, state):
        target = state
        for name in self.path[:-1]:
            target = getattr(target, name)
        if not hasattr(target, self.path[-1]):
            raise ValueError("Design parameter does not exist in the captured instrument")
        return target


def optimize_captured_design(state, parameters, diagnostic, *, maximum_evaluations=30, cancelled=lambda: False):
    """Return bounded candidate values and evidence, without mutating the input.

    ``diagnostic(trial_state)`` returns a finite residual vector in explicitly
    chosen units/scales. Only trusted in-process callables are accepted; no
    script/string evaluation or code loading is provided. Physical validity
    checks and genuine propagation remain the diagnostic's responsibility.
    """
    parameters = tuple(parameters)
    if not parameters or any(type(item) is not DesignParameter for item in parameters):
        raise ValueError("At least one bounded DesignParameter is required")
    if len({item.path for item in parameters}) != len(parameters):
        raise ValueError("Design parameter paths must not repeat")
    if not callable(diagnostic) or type(maximum_evaluations) is not int or maximum_evaluations < 1:
        raise ValueError("A diagnostic callable and positive evaluation budget are required")
    snapshot = capture_instrument_snapshot(state)
    initial = []
    for item in parameters:
        value = getattr(item.owner(state), item.path[-1])
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
            raise ValueError("Design parameters must be existing finite real settings")
        if not item.lower <= value <= item.upper:
            raise ValueError("Initial design is outside the specified bounds")
        initial.append(float(value))
        _difference_value(float(value), item)
    history, best, dimension = [], None, None

    def evaluate(values):
        nonlocal best, dimension
        if cancelled():
            raise InterruptedError("Optical design cancelled")
        if history and np.array_equal(values, history[-1]["parameters"]):
            return np.array(history[-1]["residual"])
        if len(history) >= maximum_evaluations:
            raise _EvaluationBudget
        trial = snapshot.restore()
        for item, value in zip(parameters, values):
            if not item.lower <= value <= item.upper:
                raise ValueError("Optimizer proposed a value outside the declared bounds")
            setattr(item.owner(trial), item.path[-1], float(value))
        before = json_digest(encode_instrument(trial))
        residual = np.asarray(diagnostic(trial))
        if json_digest(encode_instrument(trial)) != before:
            raise ValueError("Design diagnostic mutated its private captured inputs; use a read-only diagnostic")
        if residual.dtype.kind not in "iuf" or residual.ndim != 1 or not len(residual) or not np.isfinite(residual).all():
            raise ValueError("Design diagnostic must return a nonempty finite real residual vector")
        if dimension is not None and len(residual) != dimension:
            raise ValueError("Design diagnostic residual dimension changed")
        dimension = len(residual)
        residual = np.asarray(residual, float)
        cost = float(residual@residual)
        if not np.isfinite(cost):
            raise ValueError("Design residual scale overflows the squared-error objective")
        row = {"parameters": list(values), "residual": residual.tolist(), "squared_residual": cost,
               "trial_input_digest": before}
        history.append(row)
        if best is None or cost < best["squared_residual"]:
            best = row
        return residual

    def jacobian(values):
        baseline = evaluate(values)
        columns = []
        for index, item in enumerate(parameters):
            moved = np.array(values, copy=True)
            moved[index] = _difference_value(values[index], item)
            actual_step = moved[index]-values[index]
            columns.append((evaluate(moved)-baseline)/actual_step)
        return np.column_stack(columns)

    completed = False
    try:
        result = least_squares(evaluate, initial,
            bounds=([p.lower for p in parameters], [p.upper for p in parameters]),
            jac=jacobian,
            max_nfev=maximum_evaluations, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        completed, message = bool(result.success), str(result.message)
    except _EvaluationBudget:
        message = "Diagnostic evaluation budget exhausted; best evaluated candidate retained"
    if best is None:
        raise ValueError("No optical design candidate was evaluated")
    if capture_instrument_snapshot(state).digest != snapshot.digest:
        raise ValueError("Instrument inputs or implementation changed during the design study; candidate was not applied")
    return freeze_json({"schema": "bounded-captured-optical-design-v1", "source_digest": snapshot.digest,
        "parameter_paths": [list(p.path) for p in parameters],
        "bounds": [[p.lower, p.upper] for p in parameters], "initial_values": initial,
        "finite_difference_steps": [p.difference_step for p in parameters],
        "best": best, "evaluations": history, "converged": completed, "message": message,
        "applied": False, "aberration_feedback": False,
        "scope": "bounded numerical diagnostic study; no instrument or surrogate qualification"})


class _EvaluationBudget(Exception):
    pass


def _difference_value(value, parameter):
    """Use a representable bounded change, never a fictitious zero Jacobian."""
    step = min(parameter.difference_step, parameter.upper-value)
    if step < parameter.difference_step/2:
        step = -min(parameter.difference_step, value-parameter.lower)
    moved = float(np.clip(value+step, parameter.lower, parameter.upper))
    if moved == value:
        raise ValueError("Design finite-difference step is not representable at the current parameter value; increase difference_step")
    return moved
