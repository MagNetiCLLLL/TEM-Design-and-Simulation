"""Offscreen section-spot presentation checks using executed-record fixtures.

These are display and publication regressions, not microscope qualification.
No field solve, source sampling or user preference store is involved.
"""
from copy import deepcopy

import numpy as np
import pyqtgraph as pg
import pytest

from test_beam_analysis_modes import (
    displayed_spot_flux, make_result, view,
)
from temsim.gui import beam_analysis


def select(combo, value):
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def asymmetric_result(current=200.):
    result = make_result(4, current=current)
    branch = result.simulation.incident
    branch.x[:] = np.array([.8, 1.3, 1.9, 2.2]) * 1e-6
    branch.y[:] = np.array([-.6, -.2, .4, .7]) * 1e-6
    branch.ray_weight = np.array([.1, .2, .3, .4])
    return result


def show_spot(view, result):
    view.display_result(result)
    select(view.analysis.display_combo, "spot")
    view.plot.setRange(xRange=(-5., 5.), yRange=(-5., 5.), padding=0.)
    view._manual_view_range_changed(None)
    return view.analysis


def raster_centroid(raster):
    x = (raster.x_edges_m[1:] + raster.x_edges_m[:-1]) / 2
    y = (raster.y_edges_m[1:] + raster.y_edges_m[:-1]) / 2
    image = raster.display_fraction
    return np.array(((image * x[None, :]).sum(),
                     (image * y[:, None]).sum())) / image.sum()


def test_direct_particle_spot_switch_is_visible_and_does_not_require_diagnostics(view):
    result = asymmetric_result()
    view.display_result(result)
    analysis = view.analysis
    assert analysis.display_combo.isVisible()
    assert analysis.display_combo.currentData() == "particles"
    assert analysis.tracking_combo.currentData() == "source"
    assert analysis.mode_combo.isHidden()
    select(analysis.display_combo, "spot")
    assert analysis.mode == "intensity"
    assert analysis.tracking_combo.currentData() == "source"
    assert not analysis._advanced
    assert analysis.mode_combo.isHidden()
    assert analysis.spot_grid_combo.isVisible()
    assert analysis.spot_smoothing_combo.isVisible()
    assert analysis.spot_fixed_colour.isVisible()
    assert analysis.spot_fixed_colour.isChecked()
    assert view._scatter is None
    assert sum(isinstance(item, pg.ImageItem) for item in view.plot.items()) == 1

    select(analysis.display_combo, "particles")
    assert analysis.mode == "position"
    assert analysis.tracking_combo.currentData() == "source"
    assert not analysis.spot_grid_combo.isVisible()
    assert view._scatter is not None
    np.testing.assert_allclose(view._scatter.data["x"], result.simulation.incident.x[-1]*1e6)


def test_spot_round_trip_preserves_angular_particle_preset(view):
    view.display_result(asymmetric_result())
    analysis = view.analysis
    select(analysis.tracking_combo, "emission_direction")
    expected_x = view._scatter.data["x"].copy()
    expected_y = view._scatter.data["y"].copy()
    select(analysis.display_combo, "spot")
    assert analysis.mode == "intensity"
    assert analysis.tracking_combo.currentData() == "emission_direction"
    select(analysis.display_combo, "particles")
    assert analysis.mode == "angular"
    assert analysis.colour_combo.currentData() == "emission_direction"
    np.testing.assert_allclose(view._scatter.data["x"], expected_x)
    np.testing.assert_allclose(view._scatter.data["y"], expected_y)


def test_grid_and_smoothing_reuse_plane_and_image_without_mutating_particles(view, monkeypatch):
    result = asymmetric_result()
    before = {key: getattr(result.simulation.incident, key).copy()
              for key in ("x", "y", "tx", "ty", "ray_weight", "blocked_z")}
    analysis = show_spot(view, result)
    cached_plane = analysis.plane_data()
    image = analysis._spot_image

    def forbidden(*_args, **_kwargs):
        pytest.fail("Display controls must reuse the completed plane, without retracing or resampling")

    monkeypatch.setattr(beam_analysis, "sample_beam_plane", forbidden)
    for bins in (32, 128, 512):
        select(analysis.spot_grid_combo, bins)
        assert analysis._spot_image is image
        assert image.image.shape == (bins, bins)
        raw = analysis._spot_raster.raw_fraction.copy()
        select(analysis.spot_smoothing_combo, 2.)
        assert analysis._spot_image is image
        assert analysis.plane_data() is cached_plane
        # Equal-aspect range restoration can move a display edge by a floating
        # ulp; this only changes the bilinear weights at roundoff scale.
        np.testing.assert_allclose(analysis._spot_raster.raw_fraction, raw, rtol=1e-12, atol=1e-14)
        assert not np.array_equal(analysis._spot_raster.display_fraction, raw)
        assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(200., rel=2e-13)
        select(analysis.spot_smoothing_combo, 0.)
        np.testing.assert_allclose(analysis._spot_raster.display_fraction, raw, rtol=1e-12, atol=1e-14)
    for key, original in before.items():
        np.testing.assert_array_equal(getattr(result.simulation.incident, key), original)
    assert sum(isinstance(item, pg.ImageItem) for item in view.plot.items()) == 1


@pytest.mark.parametrize("angle", [0., 90., 31.])
def test_density_preserves_off_axis_centroid_flux_and_projection(view, angle):
    result = asymmetric_result()
    analysis = show_spot(view, result)
    view.set_projection_angle(angle)
    x, y = result.simulation.incident.x[-1], result.simulation.incident.y[-1]
    weights = result.simulation.incident.ray_weight
    theta = np.deg2rad(angle)
    expected = np.array((np.dot(np.cos(theta)*x + np.sin(theta)*y, weights),
                         np.dot(-np.sin(theta)*x + np.cos(theta)*y, weights)))
    assert np.linalg.norm(expected) > 1e-6  # A real displaced beam, not a centred fixture.
    for bins, sigma in ((32, 0.), (64, .5), (128, 2.)):
        select(analysis.spot_grid_combo, bins)
        select(analysis.spot_smoothing_combo, sigma)
        raster = analysis._spot_raster
        np.testing.assert_allclose(raster.centroid_m, expected, rtol=0., atol=1e-18)
        np.testing.assert_allclose(raster_centroid(raster), expected, rtol=0., atol=1e-18)
        assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(200., rel=2e-13)
        np.testing.assert_array_equal(analysis._spot_image.image, analysis._hover_payload[1].T)


def test_crossing_a_stop_preserves_lost_current_and_surviving_centroid(view):
    result = asymmetric_result()
    branch = result.simulation.incident
    branch.blocked_z[:2] = .5
    analysis = show_spot(view, result)
    view.focus_z(.25)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(200.)
    analysis.spot_rescale.click()
    levels = analysis._spot_image.getLevels().copy()
    view.focus_z(.75)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(140.)
    assert analysis._spot_raster.total_source_fraction == pytest.approx(.7)
    surviving = np.isnan(branch.blocked_z)
    expected = np.array((np.dot(branch.x[-1, surviving], branch.ray_weight[surviving]),
                         np.dot(branch.y[-1, surviving], branch.ray_weight[surviving]))) / .7
    np.testing.assert_allclose(analysis._spot_raster.centroid_m, expected, rtol=0., atol=1e-18)
    np.testing.assert_allclose(analysis._spot_image.getLevels(), levels)
    select(analysis.spot_smoothing_combo, 2.)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(140.)


def test_fixed_colour_survives_current_grid_z_and_publication_changes_until_rescale(view):
    result = asymmetric_result()
    analysis = show_spot(view, result)
    analysis.spot_rescale.click()
    levels = analysis._spot_image.getLevels().copy()
    initial = analysis._spot_image.image.copy()
    cached = analysis.plane_data()

    changed = deepcopy(result)
    changed.simulation.metrics["effective_source_current_pa"] = 400.
    view.display_result(changed)
    assert analysis.plane_data() is not cached
    np.testing.assert_allclose(analysis._spot_image.image, initial*2, rtol=1e-13, atol=0.)
    np.testing.assert_array_equal(analysis._spot_image.getLevels(), levels)
    view.focus_z(.2)
    select(analysis.spot_grid_combo, 256)
    select(analysis.spot_smoothing_combo, 1.)
    np.testing.assert_array_equal(analysis._spot_image.getLevels(), levels)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(400.)

    analysis.spot_rescale.click()
    assert analysis._spot_image.getLevels()[1] == pytest.approx(analysis._spot_image.image.max())
    analysis.spot_fixed_colour.setChecked(False)
    changed.simulation.metrics["effective_source_current_pa"] = 600.
    view.display_result(changed)  # Publication is a refresh even for the same result wrapper.
    assert analysis._spot_image.getLevels()[1] == pytest.approx(analysis._spot_image.image.max())
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(600.)


def test_fixed_colour_uses_separate_current_and_source_fraction_units(view):
    analysis = show_spot(view, asymmetric_result())
    analysis.spot_rescale.click()
    current_levels = analysis._spot_image.getLevels().copy()
    view.display_result(asymmetric_result(current=None))
    assert analysis._hover_payload[-1] == "% source / µm²"
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(100.)
    np.testing.assert_allclose(analysis._spot_image.getLevels(), current_levels/2)
    fraction_levels = analysis._spot_image.getLevels().copy()
    view.display_result(asymmetric_result(current=400.))
    assert analysis._hover_payload[-1] == "pA / µm²"
    np.testing.assert_array_equal(analysis._spot_image.getLevels(), current_levels)
    view.display_result(asymmetric_result(current=None))
    np.testing.assert_array_equal(analysis._spot_image.getLevels(), fraction_levels)


def test_republished_geometry_and_clipping_refresh_the_same_result_wrapper(view):
    result = asymmetric_result()
    analysis = show_spot(view, result)
    image = analysis._spot_image
    before = image.image.copy()
    plane = analysis.plane_data()
    levels = image.getLevels().copy()
    branch = result.simulation.incident

    # A newly completed propagation can reuse its result object while changing
    # both the deflected coordinates and which rays reach this plane.
    branch.x[:] += .45e-6
    branch.blocked_z[0] = .3
    view.display_result(result)

    assert analysis.plane_data() is not plane
    assert analysis._spot_image is image
    assert not np.array_equal(image.image, before)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(180.)
    weights = branch.ray_weight[1:]
    expected = np.array((np.dot(branch.x[-1, 1:], weights),
                         np.dot(branch.y[-1, 1:], weights))) / weights.sum()
    np.testing.assert_allclose(analysis._spot_raster.centroid_m, expected, rtol=0., atol=1e-18)
    np.testing.assert_allclose(raster_centroid(analysis._spot_raster), expected, rtol=0., atol=1e-18)
    np.testing.assert_array_equal(image.getLevels(), levels)


def test_initial_zero_current_does_not_lock_an_arbitrary_colour_maximum(view):
    analysis = show_spot(view, asymmetric_result(current=0.))
    assert not np.any(analysis._spot_image.image)
    view.display_result(asymmetric_result(current=200.))
    assert analysis._spot_image.getLevels()[1] == pytest.approx(analysis._spot_image.image.max())
    assert analysis._spot_image.getLevels()[1] > 0.


def test_explicit_spot_fit_includes_smoothing_support_without_recentering_or_rescaling(view):
    result = make_result(1, current=200.)
    result.simulation.incident.x[:] = 1e-6
    result.simulation.incident.y[:] = 1e-6
    analysis = show_spot(view, result)
    view.plot.setRange(xRange=(-1.08, 1.08), yRange=(-1.08, 1.08), padding=0.)
    view._manual_view_range_changed(None)
    manual_ranges = np.asarray(view.plot.viewRange())
    colour_levels = analysis._spot_image.getLevels().copy()

    select(analysis.spot_grid_combo, 32)
    select(analysis.spot_smoothing_combo, 3.)
    np.testing.assert_allclose(view.plot.viewRange(), manual_ranges, rtol=1e-12, atol=1e-12)
    assert analysis._spot_raster.display_visible_fraction < .9

    view.fit_beam.click()
    raster = analysis._spot_raster
    assert raster.raw_visible_fraction == pytest.approx(1., abs=1e-13)
    assert raster.display_visible_fraction == pytest.approx(1., abs=1e-13)
    assert displayed_spot_flux(analysis._hover_payload) == pytest.approx(200., abs=1e-10)
    np.testing.assert_allclose(raster.centroid_m, (1e-6, 1e-6), rtol=0., atol=1e-18)
    np.testing.assert_allclose(raster_centroid(raster), (1e-6, 1e-6), rtol=0., atol=1e-18)
    np.testing.assert_allclose(np.mean(view.plot.viewRange(), axis=1), 0., atol=1e-12)
    np.testing.assert_array_equal(analysis._spot_image.getLevels(), colour_levels)
