"""Array-only checks for current-preserving probe-search observables."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from optimize_joint_probe import proxy_metrics, weighted_quantile


def test_containment_quantile_retains_original_nonuniform_weights():
    radius = np.array([3., 1., 2.])
    weights = np.array([.02, .96, .02])
    assert weighted_quantile(radius, weights, .95) == 1.
    np.testing.assert_array_equal(weights, [.02, .96, .02])


def test_shell_magnitudes_do_not_hide_oppositely_rotated_sixfold_tails():
    phi = np.arange(6)*np.pi/3
    z = np.r_[1.25*np.exp(1j*phi), 1.75*np.exp(1j*(phi+np.pi/6))]
    phase = np.c_[z.real*1e-9, z.imag*1e-9, np.zeros((12, 2))]
    metrics = proxy_metrics(phase, np.ones(12))
    assert metrics[5] < 1e-6
    assert metrics[6] > .999


def test_core_observable_stays_on_commanded_axis_while_rms_is_centroid_based():
    phi = np.arange(12)*np.pi/6
    phase = np.c_[.01e-9*np.cos(phi), .01e-9*np.sin(phi), np.zeros((12, 2))]
    a = proxy_metrics(phase, np.ones(12))
    displaced = phase+np.array([.2e-9, 0., 0., 0.])
    b = proxy_metrics(displaced, np.ones(12))
    assert b[0] == pytest.approx(a[0])
    assert b[1] == pytest.approx(a[1])
    assert b[2] < a[2]*1e-10
