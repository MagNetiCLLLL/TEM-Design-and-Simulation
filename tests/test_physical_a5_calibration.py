"""Pure-array checks for physical-drive selection and protected A5 evidence."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_physical_a5 import apply, angular_metrics, protected_response


def test_fixed_axis_drive_cannot_rotate_or_invent_an_auxiliary_supply():
    hp = SimpleNamespace(key="probe_hp1_hexapole", enabled=True, strength_m3=5000.,
                         maximum_strength_m3=1e9, orientation_rad=.23)
    state = SimpleNamespace(corrector_elements=[hp], lenses=[],
        condenser_stigmator=SimpleNamespace(key="condenser_stigmator"),
        objective_lens=SimpleNamespace(key="objective_lens"))
    apply(state, {"HP1": 2.})
    assert hp.strength_m3 == 7000.
    assert hp.orientation_rad == .23
    for bad in ("HP1_skew", "HPC", "UpperOL", "DP12"):
        with pytest.raises(ValueError, match="Unregistered"):
            apply(state, {bad: 1.})
    with pytest.raises(ValueError, match="range"):
        apply(state, {"HP1": -10.})
    with pytest.raises(ValueError, match="Nonfinite"):
        apply(state, {"HP1": np.nan})


def test_independent_fields_can_have_no_protected_a5_response():
    matrix = np.zeros((26, 7))
    protected = [0, 3, 4, 7, 25]
    matrix[protected, :5] = np.eye(5)
    matrix[19, 0] = 10.
    matrix[20, 1] = 20.
    result = protected_response(matrix, np.zeros_like(matrix), np.ones(7))
    assert result["null_dimension"] == 2
    assert result["resolved_target_rank"] == 0


def test_two_protected_a5_directions_need_signals_above_difference_error():
    matrix = np.zeros((26, 7))
    matrix[[0, 3, 4, 7, 25], :5] = np.eye(5)
    matrix[19, 5], matrix[20, 6] = 10., 2.
    exact = protected_response(matrix, np.zeros_like(matrix), np.ones(7))
    assert exact["resolved_target_rank"] == 2
    error = np.zeros_like(matrix)
    error[20, 6] = .25
    noisy = protected_response(matrix, error, np.ones(7))
    assert noisy["resolved_target_rank"] == 1


def test_outer_shell_metric_detects_radial_cancellation_of_sixfold_tails():
    # Six equal angular lobes in each shell, rotated by 30 degrees in the
    # second shell. The total sixth moment cancels but the shells remain lobed.
    phi = np.arange(6)*np.pi/3
    z = np.r_[1.2*np.exp(1j*phi), 1.8*np.exp(1j*(phi+np.pi/6))]
    result = angular_metrics(dict(xy_nm=np.c_[z.real, z.imag], particle_current_a=np.full(12, 1e-9)))
    assert result["outside_1nm"]["absolute_sixth_a"] < 1e-22
    assert result["outer_shell_absolute_sum_a"] == pytest.approx(12e-9)
    assert result["outside_1nm"]["current_a"] == pytest.approx(12e-9)


def test_native_focus_proxy_uses_envelope_covariance_not_chief_position():
    from design_native_a5 import focus_proxy_nm

    angles = np.arange(12)*2*np.pi/12
    slopes = .03*np.c_[np.cos(angles), np.sin(angles)]
    offset_m = 2.3e-9
    phase = np.c_[offset_m*slopes, slopes]
    weights = np.arange(1, 13, dtype=float)
    assert focus_proxy_nm(phase, weights) == pytest.approx(2.3)
    translated = phase+np.array([5e-9, -7e-9, .002, -.003])
    assert focus_proxy_nm(translated, weights) == pytest.approx(2.3)
    # This estimates a local radial envelope zero. It is not substituted for
    # a measured x/y bracket in the actual candidate acceptance.
