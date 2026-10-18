"""Pure finite-response solve for downstream position and angle cancellation.

Responses have rows (x, y, tx, ty): the first two rows are metres per radian
of coil kick, and the final two are radians per radian. Incident phase-space
columns are metres and radians per raster factor. This module does not move
hardware, choose observation planes, alter a state, or certify a beam waist.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class DescanPhaseSpaceSolution:
    upper_kick_matrix_mrad: np.ndarray
    lower_kick_matrix_mrad: np.ndarray
    command_matrix_mrad: np.ndarray
    lower_from_upper: np.ndarray
    condition_number: float
    position_residual_m: float
    angle_residual_rad: float
    position_relative_residual: float
    angle_relative_residual: float


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return value


def _phase(value, name):
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real 4x2 matrix") from exc
    if raw.shape != (4, 2) or any(
        isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
        for item in raw.flat
    ):
        raise ValueError(f"{name} must be a finite real 4x2 matrix without booleans")
    try:
        matrix = np.array(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real 4x2 matrix") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite real 4x2 matrix")
    return matrix


def _condition(matrix, limit, label):
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{label} condition could not be determined") from exc
    if not math.isfinite(condition) or condition > limit:
        raise ValueError(f"{label} is singular or ill-conditioned (limit {limit:g})")
    return condition


def _norm(matrix):
    # Scaling avoids avoidable squaring overflow/underflow in tiny residuals.
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0:
        return 0.0
    result = scale * float(np.linalg.norm(matrix/scale))
    if not math.isfinite(result):
        raise ValueError("Descan phase-space residual magnitude exceeds the finite numerical range")
    return result


def _readonly(matrix):
    result = np.array(matrix, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def solve_position_angle_cancellation(
    upper_phase, lower_phase, incident_phase, *, separation_mm, upper_gain,
    condition_limit=1.0e10,
):
    """Cancel one captured incident position/angle map using both physical coils.

    Solve ``[U L] [Ku; Kl] = -incident`` after dividing position rows by the
    coil separation in metres. Ku and Kl are physical radians per raster
    factor. The existing paired-command representation is recovered as
    ``command = Ku / upper_gain`` and ``lower_from_upper @ Ku = Kl``.
    Both the scaled 4x4 solve and this 2x2 factorization must be well-conditioned.
    No inverse or pseudoinverse hides an uncontrollable axis.

    Absolute residuals are Frobenius norms in metres/radians. Each relative
    residual uses the largest incident/upper/lower contribution in that block,
    so a zero incident angle cannot turn cancellation roundoff into an invalid
    division or a misleading zero-baseline claim.
    """
    upper = _phase(upper_phase, "Upper coil response")
    lower = _phase(lower_phase, "Lower coil response")
    incident = _phase(incident_phase, "Incident phase space")
    separation = _scalar(separation_mm, "Coil separation")
    gain = _scalar(upper_gain, "Upper coil gain")
    limit = _scalar(condition_limit, "Condition limit")
    if separation <= 0.0:
        raise ValueError("Coil separation must be positive in millimetres")
    if gain == 0.0:
        raise ValueError("Upper coil gain must be nonzero for paired-command factorization")
    if limit < 1.0:
        raise ValueError("Condition limit must be at least one")
    scale_m = separation*1.0e-3
    if not math.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("Coil separation is outside the finite metre conversion range")
    scaled = np.column_stack((upper, lower))
    rhs = -incident.copy()
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            scaled[:2] /= scale_m
            rhs[:2] /= scale_m
    except FloatingPointError as exc:
        raise ValueError("Scaled descan phase-space problem exceeds the finite numerical range") from exc
    condition = _condition(scaled, limit, "Scaled descan phase-space response")
    try:
        physical = np.linalg.solve(scaled, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Scaled descan phase-space response is not solvable") from exc
    if not np.isfinite(physical).all():
        raise ValueError("Descan phase-space solve produced nonfinite coil kicks")
    ku, kl = physical[:2], physical[2:]
    _condition(ku, limit, "Upper coil command factorization")
    try:
        # Solve Ku.T * R.T = Kl.T, rather than explicitly inverting Ku.
        ratio = np.linalg.solve(ku.T, kl.T).T
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            upper_mrad, lower_mrad = ku*1.0e3, kl*1.0e3
            command_mrad = upper_mrad/gain
            upper_contribution, lower_contribution = upper @ ku, lower @ kl
            residual = incident + upper_contribution + lower_contribution
    except (np.linalg.LinAlgError, FloatingPointError) as exc:
        raise ValueError("Descan paired-command factorization or residual exceeds the finite numerical range") from exc
    if not all(np.isfinite(value).all() for value in (
        ratio, upper_mrad, lower_mrad, command_mrad, upper_contribution, lower_contribution, residual
    )):
        raise ValueError("Descan paired-command factorization produced nonfinite values")
    absolute = tuple(_norm(residual[block]) for block in (slice(0, 2), slice(2, 4)))
    denominators = tuple(max(_norm(value[block]) for value in (
        incident, upper_contribution, lower_contribution
    )) for block in (slice(0, 2), slice(2, 4)))
    relative = tuple(0.0 if denominator == 0.0 else value/denominator
                     for value, denominator in zip(absolute, denominators))
    return DescanPhaseSpaceSolution(
        _readonly(upper_mrad), _readonly(lower_mrad), _readonly(command_mrad), _readonly(ratio),
        condition, absolute[0], absolute[1], relative[0], relative[1],
    )
