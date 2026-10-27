"""Reception images preserve actual first stops, capture state and electron dose."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN
from temsim.detector.receiver_image import available_receivers, receiver_image


def _plane(key, z, *, disk=False):
    return SimpleNamespace(
        key=key, name=key, z_mm=z, inserted=True, outer_width_mm=2.0,
        point_spread_model="none", point_spread_sigma_x_mm=0.0,
        point_spread_sigma_y_mm=0.0, point_spread_rotation_deg=0.0,
        point_spread_status="provisional_model_parameter", point_spread_source="unit test",
        hit_mask=(lambda x, y: np.hypot(x, y) <= 1.0) if disk else
                 (lambda x, y: (np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)))


@pytest.fixture
def capture(monkeypatch):
    monkeypatch.setattr("temsim.physics.beam_current.effective_source_current_pa", lambda state: 2.0)
    camera, screen = _plane(CAMERA, 3.0), _plane(FLUORESCENT_SCREEN, 2.0, disk=True)
    weights = np.array([0.2, 0.3, 0.5])
    branch = SimpleNamespace(
        z=np.array([1.0, 2.0, 3.0]), x=np.tile([0.0, 0.0004, -0.0004], (3, 1)),
        y=np.zeros((3, 3)), alive=np.zeros(3, dtype=bool), weight=1.0,
        ray_weight=weights.copy(), blocked_key=[CAMERA, FLUORESCENT_SCREEN, "wall"],
        blocked_z=np.array([3.0, 2.0, 1.5]))
    incident = SimpleNamespace(z=np.array([0.0, 1.0]), alive=np.ones(3, dtype=bool), ray_weight=weights.copy())
    state = SimpleNamespace(recording_planes=[screen, camera],
                            sample=SimpleNamespace(z_mm=1.0, specimen_mode="vacuum", inserted=False),
                            electron_gun=SimpleNamespace(ray_count=3), projector_mode="image")
    return SimpleNamespace(state_snapshot=state, specimen_exit=None, workflow="particle",
                           simulation=SimpleNamespace(incident=incident, branches={"post": branch},
                               metrics={"section_target_z_mm": 3.0}))


def test_receivers_include_retracted_hardware_but_exclude_uninstalled(capture):
    state = capture.state_snapshot
    state.recording_planes[0].inserted = False
    state.recording_planes.append(SimpleNamespace(key="haadf"))
    assert [p.key for p in available_receivers(state)] == [FLUORESCENT_SCREEN, CAMERA]
    state.recording_planes[0]._layout_installed = False
    assert [p.key for p in available_receivers(state)] == [CAMERA]
    state.recording_planes[1].installed = False
    assert available_receivers(state) == ()


def test_first_hit_probability_and_exposure_are_not_numerical_ray_counts(capture):
    before = deepcopy(capture)
    image = receiver_image(capture, CAMERA, pixels=16, exposure_s=0.002)
    assert image.status == "AVAILABLE"
    assert image.ideal_probability.sum() == pytest.approx(0.2)
    assert image.response_probability.sum() == pytest.approx(0.2)
    assert image.current_pa == pytest.approx(0.4)
    assert image.expected_electrons.sum() == pytest.approx(0.4e-12 / 1.602176634e-19 * 0.002)
    assert image.ideal_probability.shape == (16, 16)
    assert image.x_mm[[0, -1]].tolist() == pytest.approx([-0.9375, 0.9375])
    for values in (image.ideal_probability, image.response_probability, image.expected_electrons, image.x_mm):
        assert not values.flags.writeable
    original, after = before.simulation.branches["post"], capture.simulation.branches["post"]
    assert np.array_equal(original.x, after.x)
    assert original.blocked_key == after.blocked_key
    capture.state_snapshot.electron_gun.ray_count = 9000
    repeated = receiver_image(capture, CAMERA, pixels=16, exposure_s=0.002)
    assert np.array_equal(repeated.expected_electrons, image.expected_electrons)


def test_sampling_statistics_use_nonuniform_hit_weights_and_preserve_crop_probability(capture):
    branch = capture.simulation.branches["post"]
    branch.blocked_key = [CAMERA] * 3
    branch.blocked_z[:] = 3.
    branch.x[:] = [-.2e-3, -.1e-3, .6e-3]
    before = branch.x.copy(), branch.ray_weight.copy()
    image = receiver_image(capture, CAMERA, pixels=2, include_response=False)
    s = image.metrics["sampling"]
    assert s["raw_hit_count"] == s["positive_weight_hit_count"] == 3
    assert s["in_view_hit_count"] == s["in_view_positive_weight_hit_count"] == 3
    assert s["occupied_bin_count"] == 2 and s["display_bin_count"] == 4
    assert s["hit_weight_sum"] == pytest.approx(1.)
    assert s["hit_weight_square_sum"] == pytest.approx(.38)
    assert s["effective_hit_count"] == pytest.approx(1./.38)
    effective = s["effective_hits_per_occupied_bin"]
    assert effective["minimum"] == 1.
    assert effective["maximum"] == pytest.approx(.5**2/(.2**2+.3**2))
    assert effective["median"] == pytest.approx((1.+25./13.)/2.)
    assert s["positive_hits_per_occupied_bin"]["median"] == 1.5
    crop = receiver_image(capture, CAMERA, pixels=2, extent_mm=(-.5, 0., -1., 1.), include_response=False)
    c = crop.metrics["sampling"]
    assert c["raw_hit_count"] == 3 and c["in_view_hit_count"] == 2
    assert c["occupied_bin_count"] == 1
    assert c["in_view_weight_sum"] == pytest.approx(.5)
    assert c["in_view_weight_square_sum"] == pytest.approx(.13)
    assert c["in_view_effective_hit_count"] == pytest.approx(25./13.)
    assert crop.ideal_probability.sum() == pytest.approx(.5)
    assert crop.current_pa == image.current_pa
    assert "not independent electron counts" in c["interpretation"]
    np.testing.assert_array_equal(branch.x, before[0])
    np.testing.assert_array_equal(branch.ray_weight, before[1])


def test_sampling_counts_include_edges_and_distinguish_zero_weight_records(capture):
    branch = capture.simulation.branches["post"]
    branch.blocked_key = [CAMERA] * 3
    branch.blocked_z[:] = 3.
    branch.x[:] = [-1.e-3, 1.e-3, 0.]
    branch.ray_weight[:] = [.2, .3, 0.]
    full = receiver_image(capture, CAMERA, pixels=32)
    s = full.metrics["sampling"]
    assert (s["raw_hit_count"], s["positive_weight_hit_count"], s["occupied_bin_count"]) == (3, 2, 2)
    assert full.ideal_probability.sum() == pytest.approx(.5)
    crop = receiver_image(capture, CAMERA, pixels=32, extent_mm=(-1., 0., -1., 1.))
    s = crop.metrics["sampling"]
    assert (s["in_view_hit_count"], s["in_view_positive_weight_hit_count"], s["occupied_bin_count"]) == (2, 1, 1)
    assert crop.ideal_probability.sum() == pytest.approx(.2)
    empty = receiver_image(capture, CAMERA, pixels=32, extent_mm=(.1, .2, -.1, .1))
    s = empty.metrics["sampling"]
    assert s["in_view_hit_count"] == s["occupied_bin_count"] == 0
    assert s["in_view_effective_hit_count"] == 0.
    assert all(value is None for value in s["effective_hits_per_occupied_bin"].values())


@pytest.mark.parametrize("pixels", (32, 64, 128, 256, 512, 1024))
def test_sampling_grid_changes_neither_raw_hits_nor_probability(capture, pixels):
    before = capture.simulation.branches["post"].x.copy()
    image = receiver_image(capture, CAMERA, pixels=pixels, include_response=False)
    assert image.ideal_probability.shape == (pixels, pixels)
    assert image.ideal_probability.sum() == pytest.approx(.2)
    assert image.current_pa == pytest.approx(.4)
    assert image.metrics["sampling"]["positive_weight_hit_count"] == 1
    assert image.metrics["sampling"]["display_bin_count"] == pixels**2
    np.testing.assert_array_equal(capture.simulation.branches["post"].x, before)


def test_receiver_switch_obeys_upstream_first_interception(capture):
    assert receiver_image(capture, FLUORESCENT_SCREEN, pixels=16).ideal_probability.sum() == pytest.approx(0.3)
    branch = capture.simulation.branches["post"]
    branch.blocked_key = [FLUORESCENT_SCREEN] * 3
    branch.blocked_z[:] = 2.0
    camera = receiver_image(capture, CAMERA, pixels=16)
    assert camera.status == "AVAILABLE" and camera.current_pa == 0.0
    assert not camera.expected_electrons.any()


def test_material_exit_is_preferred_over_optical_display_branch(capture):
    branch = deepcopy(capture.simulation.branches["post"])
    branch.ray_weight = np.array([0.4, 0.1, 0.5])
    capture.specimen_exit = SimpleNamespace(branches=(branch,))
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.ideal_probability.sum() == pytest.approx(0.4)
    assert image.metrics["particle_source"] == "specimen_exit"


@pytest.mark.parametrize("mode,status", [
    ("retracted", "NOT_INSERTED"), ("readout_disabled", "READOUT_DISABLED"),
    ("unreached", "NOT_REACHED"), ("optical", "NOT_CALCULATED"),
    ("rays", "NOT_CALCULATED"), ("missing", "NOT_CALCULATED"),
    ("missing_material", "NOT_CALCULATED"), ("missing_branches", "NOT_CALCULATED"),
])
def test_unavailable_is_distinct_from_a_calculated_zero(capture, mode, status):
    camera = capture.state_snapshot.recording_planes[1]
    if mode == "retracted":
        camera.inserted = False
    elif mode == "readout_disabled":
        camera.readout_enabled = False
    elif mode == "unreached":
        capture.simulation.metrics["section_target_z_mm"] = 2.5
    elif mode == "optical":
        capture.simulation.metrics["optical_tuning"] = True
    elif mode == "rays":
        capture.workflow = "rays"
    elif mode == "missing":
        capture.simulation = None
    elif mode == "missing_branches":
        capture.simulation.branches = {}
    elif mode == "missing_material":
        capture.state_snapshot.sample = SimpleNamespace(
            z_mm=1.0, specimen_mode="atomic", inserted=True, thickness_nm=5.0, cif_path="captured.cif")
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == status
    assert image.expected_electrons is None and image.current_pa is None


def test_no_downstream_population_with_no_illumination_is_a_valid_zero(capture):
    capture.simulation.branches = {}
    capture.simulation.incident.alive[:] = False
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == "AVAILABLE" and not image.expected_electrons.any()


def test_incomplete_surviving_branch_cannot_be_extrapolated(capture):
    branch = capture.simulation.branches["post"]
    branch.z = branch.z[:2]
    branch.x, branch.y = branch.x[:2], branch.y[:2]
    branch.alive[0] = True
    image = receiver_image(capture, CAMERA, pixels=16)
    assert image.status == "NOT_REACHED"


def test_physical_plane_coordinates_are_interpolated_only_inside_history(capture):
    branch = capture.simulation.branches["post"]
    branch.z = np.array([1.0, 2.5, 3.5])
    branch.x[:, 0] = [0.0, 0.0, 0.0005]
    image = receiver_image(capture, CAMERA, pixels=16)
    row, col = np.unravel_index(image.ideal_probability.argmax(), image.ideal_probability.shape)
    assert image.x_mm[col] == pytest.approx(0.3125)
    assert image.y_mm[row] == pytest.approx(0.0625)


def test_screen_response_spreads_and_is_masked_to_physical_disk(capture):
    screen = capture.state_snapshot.recording_planes[0]
    screen.point_spread_model = "gaussian"
    screen.point_spread_sigma_x_mm = screen.point_spread_sigma_y_mm = 0.2
    branch = capture.simulation.branches["post"]
    branch.x[:, 1] = 0.00095
    image = receiver_image(capture, FLUORESCENT_SCREEN, pixels=64, exposure_s=None)
    active = screen.hit_mask(image.x_mm[None, :], image.y_mm[:, None])
    assert np.all(image.response_probability[~active] == 0.0)
    assert 0.0 < image.response_probability.sum() < image.ideal_probability.sum()
    assert image.expected_electrons is None


def test_no_psf_preserves_real_hit_in_a_pixel_crossing_screen_edge(capture):
    branch = capture.simulation.branches["post"]
    branch.x[:, 1] = branch.y[:, 1] = 0.0007
    image = receiver_image(capture, FLUORESCENT_SCREEN, pixels=4)
    assert image.response_probability.sum() == pytest.approx(0.3)


@pytest.mark.parametrize("corruption,message", [
    ("wrong_z", "stop position"), ("outside", "outside"), ("nan", "finite positions"),
    ("weights", "exceeds"),
])
def test_corrupt_hit_records_are_not_presented_as_physical_reception(capture, corruption, message):
    branch = capture.simulation.branches["post"]
    if corruption == "wrong_z":
        branch.blocked_z[0] = 2.9
    elif corruption == "outside":
        branch.x[:, 0] = 0.002
    elif corruption == "nan":
        branch.x[:, 0] = np.nan
    elif corruption == "weights":
        branch.ray_weight[0] = 2.0
    with pytest.raises(ValueError, match=message):
        receiver_image(capture, CAMERA, pixels=16)


@pytest.mark.parametrize("kwargs", [{"exposure_s": -1}, {"exposure_s": np.nan},
                                    {"pixels": 1}, {"pixels": 2.5},
                                    {"extent_mm": (1, -1, -1, 1)},
                                    {"extent_mm": (np.nan, 1, -1, 1)},
                                    {"extent_mm": (2, 3, -1, 1)}])
def test_invalid_display_or_dose_requests_fail_explicitly(capture, kwargs):
    with pytest.raises(ValueError):
        receiver_image(capture, CAMERA, **kwargs)


def test_unknown_receiver_cannot_be_created_from_a_virtual_plane(capture):
    with pytest.raises(ValueError, match="Unknown or uninstalled"):
        receiver_image(capture, "virtual_camera")


def test_zoom_rebins_small_physical_hit_motion_instead_of_magnifying_old_pixels(capture):
    branch = capture.simulation.branches["post"]
    branch.x[:, 0] = 10e-6
    before = receiver_image(capture, CAMERA, pixels=64)
    roi = (.009, .013, -.002, .002)
    fine_before = receiver_image(capture, CAMERA, pixels=64, extent_mm=roi)
    branch.x[:, 0] += .1e-6
    after = receiver_image(capture, CAMERA, pixels=64)
    fine_after = receiver_image(capture, CAMERA, pixels=64, extent_mm=roi)
    np.testing.assert_array_equal(before.ideal_probability, after.ideal_probability)
    assert not np.array_equal(fine_before.ideal_probability, fine_after.ideal_probability)
    assert fine_after.ideal_probability.sum() == pytest.approx(.2)
    assert fine_after.current_pa == before.current_pa
    assert fine_after.metrics["physical_area_extent_mm"] == (-1., 1., -1., 1.)
    assert fine_after.metrics["display_extent_mm"] == roi
    assert fine_after.metrics["pixel_size_x_mm"] == pytest.approx(.004/64)


def test_crop_keeps_full_receiver_current_but_only_viewport_pixel_dose(capture):
    branch = capture.simulation.branches["post"]
    branch.blocked_key[1] = CAMERA
    branch.blocked_z[1] = 3.
    image = receiver_image(capture, CAMERA, pixels=64, extent_mm=(-.1, .1, -.1, .1), exposure_s=.5)
    assert image.metrics["intercepted_probability"] == pytest.approx(.5)
    assert image.metrics["displayed_probability"] == pytest.approx(.2)
    assert image.current_pa == pytest.approx(1.)
    assert image.expected_electrons.sum() == pytest.approx(.2*2e-12/1.602176634e-19*.5)
    assert image.metrics["first_hit_bounds_mm"] == (0., .4, 0., 0.)


def test_gaussian_response_includes_hits_outside_crop_without_tail_renormalization(capture):
    from scipy.special import ndtr
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm = camera.point_spread_sigma_y_mm = .2
    capture.simulation.branches["post"].x[:, 0] = .0002
    image = receiver_image(capture, CAMERA, pixels=64, extent_mm=(-.1, .1, -.1, .1))
    expected = .2*(ndtr(-.5)-ndtr(-1.5))*(ndtr(.5)-ndtr(-.5))
    assert image.ideal_probability.sum() == 0.
    assert image.response_probability.sum() == pytest.approx(expected, rel=2e-13)
    assert image.current_pa == pytest.approx(.4)
    assert image.metrics["intercepted_probability"] == pytest.approx(.2)
    # A much finer viewport preserves the configured physical sigma.
    tiny = receiver_image(capture, CAMERA, pixels=64, extent_mm=(-1e-5, 1e-5, -1e-5, 1e-5))
    expected_tiny = .2*(ndtr((1e-5-.2)/.2)-ndtr((-1e-5-.2)/.2))*(ndtr(1e-5/.2)-ndtr(-1e-5/.2))
    assert tiny.response_probability.sum() == pytest.approx(expected_tiny, rel=1e-10)


def test_rotated_psf_roi_matches_same_grid_full_response_including_external_hits(capture):
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm, camera.point_spread_sigma_y_mm = .12, .2
    camera.point_spread_rotation_deg = 37.
    capture.simulation.branches["post"].x[:, 0] = .0003
    full = receiver_image(capture, CAMERA, pixels=128)
    crop = receiver_image(capture, CAMERA, pixels=32, extent_mm=(-.25, .25, -.25, .25))
    assert crop.ideal_probability.sum() == 0.
    assert crop.response_probability.sum() > 0.
    np.testing.assert_allclose(crop.response_probability, full.response_probability[48:80, 48:80], atol=1e-15)


def test_crop_bounds_are_clipped_to_physical_receiver_without_changing_current(capture):
    image = receiver_image(capture, CAMERA, pixels=16, extent_mm=(-2., .5, -.5, 2.))
    assert image.metrics["display_extent_mm"] == (-1., .5, -.5, 1.)
    assert image.metrics["physical_area_extent_mm"] == (-1., 1., -1., 1.)
    assert image.current_pa == pytest.approx(.4)


def test_raw_arrivals_bypass_rotated_psf_in_a_tiny_view(capture, monkeypatch):
    import importlib
    module = importlib.import_module("temsim.detector.receiver_image")
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm, camera.point_spread_sigma_y_mm = .12, .2
    camera.point_spread_rotation_deg = 37.
    def unexpected_response(*args, **kwargs):
        raise AssertionError("An arrivals-only request must not compute a PSF")
    monkeypatch.setattr(module, "_response_from_hits", unexpected_response)
    image = receiver_image(capture, CAMERA, pixels=512,
                           extent_mm=(-1e-7, 1e-7, -1e-7, 1e-7), include_response=False)
    assert image.status == "AVAILABLE" and image.ideal_probability.sum() == pytest.approx(.2)
    assert image.response_probability is None and image.expected_electrons is None
    assert image.current_pa == pytest.approx(.4)
    assert image.metrics["response_computed"] is False
    assert "not computed" in image.metrics["response_scope"]


def test_empty_receiver_tiny_rotated_response_is_a_valid_zero(capture):
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm, camera.point_spread_sigma_y_mm = .12, .2
    camera.point_spread_rotation_deg = 37.
    capture.simulation.branches["post"].blocked_key[0] = "wall"
    image = receiver_image(capture, CAMERA, pixels=512, extent_mm=(-1e-7, 1e-7, -1e-7, 1e-7))
    assert image.status == "AVAILABLE" and image.current_pa == 0.
    assert not image.response_probability.any()
    assert not image.expected_electrons.any()


@pytest.mark.parametrize("rotation", [0., 37.])
def test_large_population_tiny_view_uses_bounded_physical_psf_grid(monkeypatch, rotation):
    import importlib
    module = importlib.import_module("temsim.detector.receiver_image")
    grid_module = importlib.import_module("temsim.detector.gaussian_pixel_response")
    # Exercise the bounded route even though duplicate hits now share exact
    # integrals and may fit inside the default direct-quadrature budget.
    monkeypatch.setattr(grid_module, "DIRECT_EVALUATION_BUDGET", 0)
    from temsim.detector.point_spread import DetectorPointSpread
    count = 100_000
    sigma_x, sigma_y = .12, .2
    psf = DetectorPointSpread("gaussian", sigma_x, sigma_y, rotation,
                             "provisional_model_parameter", "analytic test")
    # All hits are outside the very small view. Their physical response must
    # remain visible, without renormalizing the small response mass to one.
    positions = np.repeat([[.03, -.02], [-.03, .02]], count//2, axis=0)
    # Separated clusters exceed the compact domain and retain fallback coverage.
    weights = np.full(count, .2/count)
    edges = np.linspace(-1e-5, 1e-5, 513)
    calls = []
    apply = grid_module.apply_point_spread
    def bounded(values, *args, **kwargs):
        calls.append(values.shape)
        assert values.size < 120_000  # Independent of requested view pixel pitch.
        return apply(values, *args, **kwargs)
    monkeypatch.setattr(grid_module, "apply_point_spread", bounded)
    response, method, pitch = module._response_from_hits(positions, weights, edges, edges, psf)
    angle = np.deg2rad(rotation)
    centres = positions[[0, -1]]
    px = np.cos(angle)*centres[:, 0]+np.sin(angle)*centres[:, 1]
    py = -np.sin(angle)*centres[:, 0]+np.cos(angle)*centres[:, 1]
    # Here ROI width/sigma < 2e-4, so centre density times area is an
    # independent analytic limit. Deposition/interpolation tolerance is 0.1%.
    expected = .1*np.exp(-.5*((px/sigma_x)**2+(py/sigma_y)**2)).sum()/(2*np.pi*sigma_x*sigma_y)*(2e-5)**2
    assert calls and "Bounded" in method
    assert pitch == pytest.approx((sigma_x/24, sigma_x/24))
    assert response.sum() == pytest.approx(expected, rel=1e-3)
    assert response.sum() < 1e-8


def test_large_population_response_tracks_a_subpixel_displacement():
    from temsim.detector.receiver_image import _response_from_hits
    from temsim.detector.point_spread import DetectorPointSpread
    psf = DetectorPointSpread("gaussian", .03, .05, 0.,
                             "provisional_model_parameter", "centroid test")
    count = 100_000
    positions = np.tile([.001, -.002], (count, 1))
    weights = np.full(count, .2/count)
    edges = np.linspace(-.4, .4, 257)
    before, method, pitch = _response_from_hits(positions, weights, edges, edges, psf)
    shift = .0002
    positions[:, 0] += shift
    after, _, _ = _response_from_hits(positions, weights, edges, edges, psf)
    centres = .5*(edges[:-1]+edges[1:])
    centroid_before = (before*centres[None, :]).sum()/before.sum()
    centroid_after = (after*centres[None, :]).sum()/after.sum()
    assert "Compact" in method and pitch is None
    assert shift < edges[1]-edges[0]
    assert not np.array_equal(before, after)
    assert before.sum() == pytest.approx(.2, rel=1e-10)
    assert after.sum() == pytest.approx(.2, rel=1e-10)
    assert centroid_before == pytest.approx(.001, abs=1e-10)
    assert centroid_after-centroid_before == pytest.approx(shift, abs=1e-10)


def test_compact_response_exports_numeric_bound_and_physical_centroid(capture):
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm = camera.point_spread_sigma_y_mm = .2
    image = receiver_image(capture, CAMERA, pixels=64)
    numeric = image.metrics["point_spread_numerics"]
    assert numeric["method"] == "compact_gaussian_moments"
    assert numeric["order"] == 6
    assert numeric["global_l1_truncation_bound"] >= 0.
    assert numeric["radius_over_sigma_max"] <= .001
    assert numeric["total_weight"] == pytest.approx(image.metrics["intercepted_probability"])
    assert "floating-point" in numeric["bound_scope"]
    assert image.metrics["response_grid_pixel_size_mm"] is None


def test_rotated_receiver_blur_loses_edge_response_without_altering_current(capture):
    from scipy.special import ndtr
    camera = capture.state_snapshot.recording_planes[1]
    camera.point_spread_model = "gaussian"
    camera.point_spread_sigma_x_mm, camera.point_spread_sigma_y_mm = .025, .020
    camera.point_spread_rotation_deg = 45.
    branch = capture.simulation.branches["post"]
    branch.x[:, 0] = .998e-3
    original_positions, original_weights = branch.x.copy(), branch.ray_weight.copy()
    image = receiver_image(capture, CAMERA, pixels=128, exposure_s=.002)
    sigma_x = np.sqrt((.025**2+.020**2)/2)
    expected = .2*ndtr((1.-.998)/sigma_x)
    assert image.ideal_probability.sum() == pytest.approx(.2)
    assert image.response_probability.sum() == pytest.approx(expected, abs=3e-14)
    assert image.current_pa == pytest.approx(.4)
    assert image.expected_electrons.sum() == pytest.approx(expected*2e-12/1.602176634e-19*.002)
    assert image.response_probability.sum() < image.ideal_probability.sum()
    np.testing.assert_array_equal(branch.x, original_positions)
    np.testing.assert_array_equal(branch.ray_weight, original_weights)
