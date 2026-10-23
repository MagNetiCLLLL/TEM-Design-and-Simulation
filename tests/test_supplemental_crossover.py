"""Supplemental C2-C3 markers must measure physical current, not ray count."""

from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.all_lens_crossovers import detect_all_lens_crossovers
from temsim.physics.crossovers import first_crossover_after_lens


def bundle():
    z = np.linspace(0., 10., 101)
    angle = np.tile(np.arange(4)*np.pi/2, 2)
    tx, ty = .01*np.cos(angle), .01*np.sin(angle)
    focus = np.repeat([2., 8.], 4)
    dz = (z[:, None]-focus)*1e-3
    return SimpleNamespace(z=z, x=dz*tx, y=dz*ty,
        tx=np.broadcast_to(tx, dz.shape).copy(),
        ty=np.broadcast_to(ty, dz.shape).copy(),
        blocked_z=np.full(8, np.nan), ray_weight=np.repeat([9., 1.], 4))


def marker(branch, **kwargs):
    return first_crossover_after_lens(branch, .5, "C2-C3 crossover", **kwargs)


def test_current_weighted_marker_is_invariant_to_ray_splitting():
    b = bundle()
    # For equal angular spread, the minimum of sum(w*(z-focus)**2) is
    # the current-weighted mean focus: (9*2+1*8)/10 = 2.6 mm.
    expected_z, expected_rms_mm = 2.6, .018
    repeats = np.repeat([1, 31], 4)
    copies = SimpleNamespace(z=b.z,
        **{key: np.repeat(getattr(b, key), repeats, axis=1)
           for key in ("x", "y", "tx", "ty")},
        blocked_z=np.repeat(b.blocked_z, repeats),
        ray_weight=np.repeat(b.ray_weight/repeats, repeats))
    lens = SimpleNamespace(key="condenser_lens_2", name="C2", z_mm=.5, enabled=True)
    for branch in (b, copies):
        before = {key: value.copy() for key, value in vars(branch).items()}
        actual = marker(branch, stop_z_mm=10.)
        assert actual == dict(name="C2-C3 crossover", z_mm=pytest.approx(expected_z),
            rms_radius_mm=pytest.approx(expected_rms_mm), verified=True, source_lens_z_mm=.5)
        main, = detect_all_lens_crossovers([branch], [lens])
        assert actual["z_mm"] == main["z_mm"]
        assert actual["rms_radius_mm"] == pytest.approx(main["rms_radius_mm"])
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(branch, key), value)


def test_zero_current_probes_do_not_move_or_create_a_marker():
    b = bundle()
    expected = marker(b)
    # Add wide, differently focused numerical probes carrying no current.
    for key in ("x", "y", "tx", "ty"):
        values = getattr(b, key)
        probes = values*100.
        if key in ("x", "y"):
            probes = (values + .003*getattr(b, "t"+key))*100.
        setattr(b, key, np.concatenate([values, probes], axis=1))
    b.blocked_z = np.full(16, np.nan)
    b.ray_weight = np.r_[b.ray_weight, np.zeros(8)]
    assert marker(b) == expected
    b.ray_weight[:] = 0.
    assert marker(b) is None


def test_particles_stopped_before_right_bracket_cannot_verify_a_marker():
    b = bundle()
    # The equal-weight waist is at 5 mm, but every ray stops before the
    # next sample. There is no surviving convergence/divergence bracket.
    b.ray_weight = None
    b.blocked_z[:] = 5.05
    assert marker(b) is None


@pytest.mark.parametrize("weight_mode", ["missing", "none", "equal"])
def test_legacy_equal_weight_histories_retain_their_waist(weight_mode):
    b = bundle()
    if weight_mode == "missing":
        del b.ray_weight
    else:
        b.ray_weight = None if weight_mode == "none" else np.ones(8)
    assert marker(b)["z_mm"] == pytest.approx(5.)


@pytest.mark.parametrize("weight", [np.full(8, -1.), np.full(8, np.nan), np.ones(7)])
def test_invalid_weights_are_rejected_like_main_waist_diagnostics(weight):
    b = bundle()
    b.ray_weight = weight
    with pytest.raises(ValueError, match="current-weighted waist history"):
        marker(b)


def test_first_marker_search_limits_and_minimum_population_are_preserved():
    b = bundle()
    b.z = np.arange(9.)
    radius = np.array([5., 4., 3., 4., 5., 3., 1., 2., 3.])*1e-5
    angle = np.arange(8)*2*np.pi/8
    slope = np.gradient(radius, b.z*1e-3)
    b.x, b.y = radius[:, None]*np.cos(angle), radius[:, None]*np.sin(angle)
    b.tx, b.ty = slope[:, None]*np.cos(angle), slope[:, None]*np.sin(angle)
    b.ray_weight = np.ones(8)
    assert marker(b)["z_mm"] == 2.  # First waist, not the smaller one at 6.
    assert marker(b, stop_z_mm=2.) is None
    assert marker(b, start_margin_mm=1.5)["z_mm"] == 6.
    assert marker(b, minimum_rays=8)["z_mm"] == 2.
    assert marker(b, minimum_rays=9) is None
