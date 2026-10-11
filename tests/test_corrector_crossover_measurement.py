"""Physical waist evidence uses ray phase space, not a conjugate-plane label."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.corrector_crossovers import (
    detect_corrector_crossovers,
    measured_corrector_crossovers,
)


def drift_waist():
    z = np.arange(3., 8.)
    phi = np.arange(16)*2*np.pi/16
    tx, ty = .002*np.cos(phi), .002*np.sin(phi)
    dz = (z[:, None] - 5.)*1e-3
    # The second angular harmonic supplies 10 nm RMS emittance blur and is
    # uncorrelated with the first-harmonic convergence. The exact waist is 5 mm.
    x = dz*tx + 1e-8*np.cos(2*phi)
    y = dz*ty + 1e-8*np.sin(2*phi)
    cp = SimpleNamespace(z_mm=z, x_m=x, y_m=y,
                         tx_rad=np.broadcast_to(tx, x.shape).copy(),
                         ty_rad=np.broadcast_to(ty, x.shape).copy())
    return cp, np.ones(x.shape, bool), np.ones(phi.size)


def measure(cp, masks, weights, lower=2., upper=8.):
    return measured_corrector_crossovers(cp, masks, weights, lower, upper)


def test_real_waist_has_current_weighted_rotation_invariant_evidence():
    cp, masks, weights = drift_waist()
    row, = measure(cp, masks, weights)
    assert row["z_mm"] == pytest.approx(5.)
    assert row["rms_radius_nm"] == pytest.approx(10.)
    assert row["bracket_mm"] == [4., 6.]
    assert row["bracket_covariance_m_rad"][0] < 0 < row["bracket_covariance_m_rad"][2]
    assert row["ray_count"] == 16
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    cp.x_m, cp.y_m = np.einsum("ab,bzr->azr", rotation, [cp.x_m, cp.y_m])
    cp.tx_rad, cp.ty_rad = np.einsum("ab,bzr->azr", rotation, [cp.tx_rad, cp.ty_rad])
    # A moving off-axis chief ray is not beam convergence or divergence.
    cp.x_m += cp.z_mm[:, None]*1e-5
    cp.tx_rad += .01
    rotated, = measure(cp, masks, weights)
    assert rotated["rms_radius_nm"] == pytest.approx(row["rms_radius_nm"])
    assert rotated["bracket_principal_rms_nm"] == pytest.approx(
        np.asarray(row["bracket_principal_rms_nm"]))


def test_ray_splitting_preserves_current_weighted_crossover():
    cp, masks, weights = drift_waist()
    weights[::2] = 3.
    before, = measure(cp, masks, weights)
    repeats = np.arange(weights.size) % 3 + 1
    for key in ("x_m", "y_m", "tx_rad", "ty_rad"):
        setattr(cp, key, np.repeat(getattr(cp, key), repeats, axis=1))
    after, = measure(cp, np.repeat(masks, repeats, axis=1),
                      np.repeat(weights/repeats, repeats))
    assert after["z_mm"] == before["z_mm"]
    assert after["rms_radius_nm"] == pytest.approx(before["rms_radius_nm"])
    assert after["bracket_covariance_m_rad"] == pytest.approx(before["bracket_covariance_m_rad"])


def test_one_axis_line_focus_is_not_a_two_dimensional_crossover():
    cp, masks, weights = drift_waist()
    cp.y_m[:] = cp.y_m[2]
    cp.ty_rad[:] = 0.
    assert measure(cp, masks, weights) == []


def test_collimated_pupil_or_monotonic_divergence_does_not_make_source_waist():
    cp, masks, weights = drift_waist()
    cp.x_m[:] = cp.x_m[2]
    cp.y_m[:] = cp.y_m[2]
    cp.tx_rad[:] = cp.ty_rad[:] = 0.
    # A transfer-plane conjugacy could still exist, but these rays do not focus.
    assert measure(cp, masks, weights) == []
    cp, masks, weights = drift_waist()
    cp.x_m += .003*cp.tx_rad
    cp.y_m += .003*cp.ty_rad
    assert measure(cp, masks, weights) == []


def test_clipping_outer_convergent_rays_cannot_create_survivor_waist():
    cp, masks, weights = drift_waist()
    # Surviving core diverges throughout. A broad focused halo disappears at
    # 5 mm: measuring different cohorts would falsely produce a local minimum.
    core_x = cp.x_m + .003*cp.tx_rad
    core_y = cp.y_m + .003*cp.ty_rad
    cp.x_m = np.concatenate([core_x, cp.x_m*100], axis=1)
    cp.y_m = np.concatenate([core_y, cp.y_m*100], axis=1)
    cp.tx_rad = np.concatenate([cp.tx_rad, cp.tx_rad*100], axis=1)
    cp.ty_rad = np.concatenate([cp.ty_rad, cp.ty_rad*100], axis=1)
    masks = np.ones(cp.x_m.shape, bool)
    masks[2:, 16:] = False
    assert measure(cp, masks, np.ones(32)) == []


def test_entire_bracket_must_be_inside_clear_hardware_interval():
    cp, masks, weights = drift_waist()
    assert measure(cp, masks, weights, lower=4., upper=8.) == []
    assert measure(cp, masks, weights, lower=2., upper=6.) == []
    assert len(measure(cp, masks, weights, lower=3., upper=7.)) == 1


def test_narrow_waist_between_equal_coarse_samples_requires_an_exact_checkpoint():
    original, _, weights = drift_waist()

    def sample(z):
        dz = (z[:, None] - 5.25)*1e-3
        cp = SimpleNamespace(z_mm=z,
            x_m=original.x_m[2] + dz*original.tx_rad[2],
            y_m=original.y_m[2] + dz*original.ty_rad[2],
            tx_rad=np.broadcast_to(original.tx_rad[2], (z.size, 16)),
            ty_rad=np.broadcast_to(original.ty_rad[2], (z.size, 16)))
        return measure(cp, np.ones(cp.x_m.shape, bool), weights)

    # Convergence changes sign between 5 and 5.5 mm. Neither 500 nm sampled
    # radius is the true 10 nm waist, so do not claim an exact sampled minimum.
    assert sample(np.array([4.5, 5., 5.5, 6.])) == []
    row, = sample(np.array([5.249, 5.25, 5.251]))
    assert row["z_mm"] == pytest.approx(5.25)
    assert row["rms_radius_nm"] == pytest.approx(10.)


def test_invalid_current_or_live_coordinates_fail_explicitly():
    cp, masks, weights = drift_waist()
    weights[0] = -1.
    with pytest.raises(ValueError, match="Invalid measured"):
        measure(cp, masks, weights)
    weights[0] = 1.
    cp.x_m[2, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite surviving"):
        measure(cp, masks, weights)


def test_historical_corrector_markers_honor_current_not_representative_count():
    cp, masks, weights = drift_waist()
    # Zero-current copies with a separate waist must not displace the marker.
    branch = SimpleNamespace(z=cp.z_mm, blocked_z=np.full(32, np.nan),
                             ray_weight=np.r_[weights, np.zeros(16)])
    for old, new in (("x_m", "x"), ("y_m", "y"),
                     ("tx_rad", "tx"), ("ty_rad", "ty")):
        values = getattr(cp, old)
        other = values
        # Position and slope names differ in the checkpoint API.
        if old == "x_m":
            other = values + .001*cp.tx_rad
        elif old == "y_m":
            other = values + .001*cp.ty_rad
        setattr(branch, new, np.concatenate([values, other], axis=1))
    row, = detect_corrector_crossovers(branch, [5.], half_window_mm=3.)
    assert row["z_mm"] == pytest.approx(5.)
    assert row["rms_radius_mm"] == pytest.approx(1e-5)
