"""Bounded local electrical compensation on a common diagnostic pupil.

Every accepted step is propagated independently. Signed/full versus half-step
responses are checked before use. Route search bounds and electrical headroom
are distinct; exhausting the former does not prove a design uncorrectable.
"""
from dataclasses import asdict

import numpy as np
from scipy.optimize import lsq_linear

from temsim.assembly_error_study import report_payload, report_vector
from temsim import input_io
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.aberration_basis import CARTESIAN_NAMES, polynomial_basis
from temsim.optics.aberration_validation import _control_snapshot
from temsim.optics.probe_service import (
    OUTPUTS, apply_probe_output_deltas, output_reading, preview_probe_output_deltas,
)
from temsim.optics.system_residual_aberrations import measure_system_residual


def combined_weights(intrinsic, learning):
    return [{key: a.get(key, 0.) + b.get(key, 0.) for key in a.keys() | b.keys()}
            for a, b in zip(intrinsic, learning, strict=True)]


def output_deltas(weights, vector):
    result = {}
    for row, value in zip(weights, vector, strict=True):
        for name, gain in row.items():
            result[name] = result.get(name, 0.) + gain * float(value)
    return {key: value for key, value in result.items() if value}


class StudyObserver:
    def __init__(self, reference, options, cancelled, progress):
        self.reference, self.options = reference, options
        self.cancelled, self.progress, self.count = cancelled, progress, 0
        # RMS gradient displacement weights orders by their actual pupil action.
        basis = -polynomial_basis(reference.holdout, gradient=True).reshape(-1, 15) * 1e-3
        basis[:, [i for i, key in enumerate(CARTESIAN_NAMES) if key not in options.target_coefficients]] = 0.
        self.displacement = basis
        self.objective = np.zeros((len(basis) + 4, 19))
        self.objective[:-4, :15] = basis / (options.pupil_rms_tolerance_nm * 1e-9 * np.sqrt(len(basis)))
        self.objective[-4:-2, 15:17] = np.eye(2) / (options.centroid_tolerance_nm * 1e-9)
        self.objective[-2:, 17:] = np.eye(2) / (options.direction_tolerance_mrad * 1e-3)

    def check(self):
        if self.cancelled():
            raise RuntimeError("Assembly study cancelled")

    def validate_state(self, state):
        from temsim.geometry_experiments import validate_active_fields
        with input_io.input_scope(state):
            validate_active_fields(state)

    def measure(self, state):
        self.check()
        self.count += 1
        result = measure_system_residual(state, "probe", reference=self.reference,
            semiangle_mrad=self.options.semiangle_mrad,
            maximum_step_mm=self.options.maximum_step_mm, cancelled=self.cancelled)
        report_vector(result)
        return result

    def errors(self, report, nominal):
        difference = report_vector(report) - report_vector(nominal)
        values = dict(pupil_rms_nm=float(np.sqrt(np.mean((self.displacement @ difference[:15])**2))) * 1e9,
                      centroid_distance_nm=float(np.linalg.norm(difference[15:17])) * 1e9,
                      direction_distance_mrad=float(np.linalg.norm(difference[17:])) * 1e3)
        values["target_met"] = bool(values["pupil_rms_nm"] <= self.options.pupil_rms_tolerance_nm
            and values["centroid_distance_nm"] <= self.options.centroid_tolerance_nm
            and values["direction_distance_mrad"] <= self.options.direction_tolerance_mrad)
        values["coefficient_delta_mm"] = dict(zip(CARTESIAN_NAMES, map(float, difference[:15])))
        return values


def _shifted(state, weights, vector):
    work = _control_snapshot(state)
    changes = apply_probe_output_deltas(work, output_deltas(weights, vector))
    return work, changes


def response_matrix(state, baseline, weights, observer):
    """Nineteen observations per route percentage point; keep cross terms."""
    count = len(weights)
    matrix = np.zeros((19, count))
    steps = np.zeros(count)
    columns = []
    for index, row in enumerate(weights):
        observer.check()
        label = observer.options.routes[index]
        if not any(row.values()):
            columns.append(dict(route=label, status="ZERO_CONFIGURED_GAIN", step_percent=0.))
            continue
        observer.progress(f"Measuring signed response: {label}")
        step = observer.options.calibration_step_percent
        failure = ""
        for _ in range(8):
            trials = []
            try:
                for sign in (1., -1., .5, -.5):
                    vector = np.zeros(count)
                    vector[index] = sign * step
                    trials.append(_shifted(state, weights, vector)[0])
                break
            except ValueError as exc:
                failure = str(exc)
                step *= .5
        else:
            columns.append(dict(route=label, status="NO_SIGNED_HEADROOM", reason=failure, step_percent=0.))
            continue
        plus, minus, half_plus, half_minus = [report_vector(observer.measure(s)) for s in trials]
        full = (plus - minus) / (2 * step)
        half = (half_plus - half_minus) / step
        scaled = observer.objective @ half * step
        denominator = max(float(np.linalg.norm(scaled)), 1e-12)
        error = float(np.linalg.norm(observer.objective @ (full - half) * step) / denominator)
        asymmetry = float(np.linalg.norm(observer.objective @ (
            half_plus + half_minus - 2 * report_vector(baseline))) / denominator)
        good = np.isfinite(half).all() and error <= .1 and asymmetry <= .2
        columns.append(dict(route=label, status="USABLE" if good else "NONLINEAR_OR_UNRESOLVED",
            step_percent=step, signed_half_step_relative_error=error, signed_asymmetry=asymmetry))
        if good:
            matrix[:, index], steps[index] = half, step
    scaled = observer.objective @ matrix * steps
    singular = np.linalg.svd(scaled, compute_uv=False)
    rank = int(np.linalg.matrix_rank(scaled))
    condition = float(singular[0] / singular[-1]) if len(singular) and singular[-1] > 0 else None
    if condition is not None and not np.isfinite(condition):
        condition = None
    evidence = dict(observables=[*CARTESIAN_NAMES, "chief_x", "chief_y", "chief_tx", "chief_ty"],
        units=[*["mm/percentage_point"] * 15, *["m/percentage_point"] * 2, *["rad/percentage_point"] * 2],
        matrix=matrix.tolist(), columns=columns, objective_rank=rank,
        objective_condition_number=condition, singular_values=singular.tolist(),
        output_weights=weights, natural_cross_coupling_retained=True)
    return (matrix, steps), evidence


def _directional_headroom(state, weights, direction):
    weights = {name: gain * direction for name, gain in weights.items() if gain}
    if not weights:
        return None
    cap = float("inf")
    for name, gain in weights.items():
        reading = output_reading(state, name)
        lower = 0. if OUTPUTS[name].unit == "% excitation" else -100.
        upper = 100.
        cap = min(cap, (upper - reading.percent) / gain if gain > 0 else (lower - reading.percent) / gain)
    cap = max(0., cap)
    def valid(value):
        try:
            preview_probe_output_deltas(state, {name: gain * value for name, gain in weights.items()})
            return True
        except ValueError:
            return False
    if cap == 0. or valid(cap):
        return cap
    low, high = 0., cap
    for _ in range(32):
        middle = (low + high) / 2
        if valid(middle):
            low = middle
        else:
            high = middle
    return low


def headroom(state, weights, routes):
    """Directional limits include simultaneous vector and shared AC/DP12 loads."""
    rows = []
    for name in sorted({name for row in weights for name, gain in row.items() if gain}):
        reading = output_reading(state, name)
        rows.append(dict(output=name, component=OUTPUTS[name].component,
            physical_host="ac_deflector" if name.startswith("DP12") else OUTPUTS[name].component,
            maximum=reading.maximum, unit=reading.unit, native=reading.native, percent=reading.percent,
            positive_margin_percent=_directional_headroom(state, {name: 1.}, 1.),
            negative_margin_percent=_directional_headroom(state, {name: 1.}, -1.)))
    return dict(outputs=rows, routes=[dict(route=key,
        positive_margin_percent=_directional_headroom(state, row, 1.),
        negative_margin_percent=_directional_headroom(state, row, -1.))
        for key, row in zip(routes, weights, strict=True)],
        scope="Model electrical limits with other drives held; not a global correctable-aberration range")


def compensate(state, nominal, perturbed, weights, observer):
    options = observer.options
    count = len(weights)
    command = np.zeros(count)
    current, measured = _control_snapshot(state), perturbed
    history, first_response = [], None
    status = "LOCAL_SEARCH_LIMIT"
    for iteration in range(options.maximum_iterations):
        observer.check()
        if observer.errors(measured, nominal)["target_met"]:
            status = "LOCAL_TARGET_MET"
            break
        (jacobian, steps), evidence = response_matrix(current, measured, weights, observer)
        if first_response is None:
            first_response = evidence
        active = np.flatnonzero(steps > 0)
        if not len(active) or evidence["objective_rank"] == 0:
            status = "LOCAL_RESPONSE_UNAVAILABLE"
            break
        delta = report_vector(measured) - report_vector(nominal)
        residual = observer.objective @ delta
        scale = steps[active]
        lower = np.maximum(-.5, (-options.maximum_change_percent - command[active]) / scale)
        upper = np.minimum(.5, (options.maximum_change_percent - command[active]) / scale)
        free = upper - lower > 1e-12
        active, scale, lower, upper = active[free], scale[free], lower[free], upper[free]
        if not len(active):
            break
        matrix = (observer.objective @ jacobian[:, active]) * scale
        solution = lsq_linear(matrix, -residual, bounds=(lower, upper), tol=1e-10, max_iter=100)
        increment = np.zeros(count)
        increment[active] = solution.x * scale
        if not np.isfinite(increment).all():
            raise ValueError("Nonfinite bounded compensation solution")
        accepted, failures = False, []
        for fraction in (1., .5, .25, .125, .0625):
            observer.check()
            try:
                candidate, _ = _shifted(state, weights, command + fraction * increment)
                actual = observer.measure(candidate)
                score = float(np.linalg.norm(observer.objective @ (report_vector(actual) - report_vector(nominal))))
                if score < np.linalg.norm(residual) * (1. - 1e-9):
                    current, measured = candidate, actual
                    command += fraction * increment
                    accepted = True
                    break
                failures.append("Executed candidate did not improve the combined target")
            except ValueError as exc:
                failures.append(str(exc))
        history.append(dict(iteration=iteration + 1, accepted=accepted, failures=failures,
                            route_change_percent=command.tolist(), errors=observer.errors(measured, nominal),
                            response=evidence))
        if not accepted:
            status = "NO_IMPROVING_LOCAL_STEP"
            break
    errors = observer.errors(measured, nominal)
    if errors["target_met"]:
        status = "LOCAL_TARGET_MET"
    # Publish an exact complete-graph candidate identity, not the ray-only clone.
    exact = capture_instrument_snapshot(state).restore()
    changes = apply_probe_output_deltas(exact, output_deltas(weights, command))
    return dict(status=status, report=report_payload(measured), errors=errors,
        route_change_percent=dict(zip(options.routes, map(float, command))),
        output_change_percent=output_deltas(weights, command),
        physical_changes=[asdict(change) for change in changes],
        result_physical_digest=capture_instrument_snapshot(exact).physical_digest,
        response=first_response, iterations=history,
        headroom_before=headroom(state, weights, options.routes),
        headroom_after=headroom(exact, weights, options.routes),
        qualification="Diagnostic target only; local search exhaustion is not proof of insufficient hardware",
        learning_coefficients_fitted=False, coefficients_applied=False,
        finite_source_status="NOT_ASSESSED")
