"""Selected-plane geometry and probability accounting from cached histories."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from temsim.gui.beam_plane_data import (
    _interpolate,
    angular_histogram,
    sample_beam_plane,
    spatial_histogram,
)
from temsim.physics.interaction_budget import plane_interaction_budget
from temsim.specimen.downstream_transport import GeometricSpecimenExit
from temsim.gui.beam_spot_data import rasterize_beam_spot


def make_branch(name="incident", count=4, z=(0.0, 10.0), **changes):
    x = np.tile(np.linspace(-2.0e-6, 2.0e-6, count), (len(z), 1))
    y = np.tile(np.linspace(1.0e-6, -1.0e-6, count), (len(z), 1))
    values = dict(
        name=name, interaction_kind="incident" if name == "incident" else "transmitted",
        z=np.asarray(z, float), x=x, y=y,
        tx=np.tile(np.linspace(0.0, 0.03, count), (len(z), 1)),
        ty=np.zeros((len(z), count)), blocked_z=np.full(count, np.nan),
        ray_weight=np.full(count, 1.0 / max(count, 1)), weight=1.0,
        source_ray_id=np.arange(count, dtype=np.int64) * 7 + 50,
        source_azimuth_rad=np.linspace(0.0, 2.0 * np.pi, count, endpoint=False),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def make_result(incident=None, branches=None):
    incident = incident if incident is not None else make_branch()
    if branches is None:
        branches = [make_branch("000", count=incident.x.shape[1], z=(10.0, 20.0),
                                blocked_z=incident.blocked_z.copy())]
    return SimpleNamespace(
        simulation=SimpleNamespace(
            incident=incident, branches={str(i): branch for i, branch in enumerate(branches)},
            metrics={"effective_source_current_pa": 20.0, "branch_weights_are_absolute": True},
            real_interactions=None,
        ),
        signatures={"sample_downstream": "current"},
        state_snapshot=SimpleNamespace(sample=SimpleNamespace(z_mm=10.0)),
    )


def detailed_result(branches):
    result = make_result()
    tracked = sum(branch.weight for branch in branches)
    result.specimen_exit = GeometricSpecimenExit(
        tuple(branches),
        {"tracked_downstream_source_probability": tracked,
         "inelastic_absorbed_source_probability": 1.0 - tracked},
        "current",
    )
    return result


def test_incident_interpolation_source_identity_and_absolute_current():
    incident = make_branch(ray_weight=np.array([0.1, 0.2, 0.3, 0.4]))
    incident.x[1] += 4.0e-6
    incident.tx[1] += 0.01
    original = incident.x.copy()
    plane = sample_beam_plane(make_result(incident), 2.5)
    assert plane.provenance == "Incident" and plane.status == "Ready"
    assert plane.weights_valid and plane.ray_count == 4
    np.testing.assert_allclose(plane.x_m, incident.x[0] + 1.0e-6)
    np.testing.assert_allclose(plane.tx, incident.tx[0] + 0.0025)
    np.testing.assert_array_equal(plane.source_ray_id, incident.source_ray_id)
    np.testing.assert_array_equal(plane.source_azimuth_rad, incident.source_azimuth_rad)
    np.testing.assert_array_equal(plane.column_index, np.arange(4))
    np.testing.assert_array_equal(plane.interaction_key, np.full(4, "incident"))
    assert plane.total_column_count == 4
    assert plane.total_source_fraction == pytest.approx(1.0)
    assert plane.current_pa == pytest.approx(20.0)
    for array in (plane.x_m, plane.source_fraction, plane.source_ray_id,
                  plane.interaction_key, plane.interaction_rgb, plane.column_index):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(incident.x, original)


def test_exact_interception_is_included_and_global_indices_do_not_shift():
    incident = make_branch(
        ray_weight=np.array([0.1, 0.0, 0.3, 0.6]),
        blocked_z=np.array([5.0, np.nan, np.nan, 8.0]),
    )
    result = make_result(incident)
    before = sample_beam_plane(result, 4.0)
    intercept = sample_beam_plane(result, 5.0)
    after = sample_beam_plane(result, 5.0001)
    np.testing.assert_array_equal(before.column_index, [0, 2, 3])
    np.testing.assert_array_equal(intercept.column_index, before.column_index)
    np.testing.assert_array_equal(after.column_index, [2, 3])
    assert after.total_column_count == 4
    assert intercept.total_source_fraction == pytest.approx(1.0)
    assert after.total_source_fraction == pytest.approx(0.9)


@pytest.mark.parametrize("absolute", [False, True])
def test_optical_reference_agrees_with_authoritative_budget(absolute):
    incident = make_branch(ray_weight=np.array([0.1, 0.2, 0.3, 0.4]),
                           blocked_z=np.array([np.nan, 5.0, np.nan, np.nan]))
    direct = make_branch("000", z=(10.0, 20.0), weight=0.3 if absolute else 3.0,
                         blocked_z=np.array([np.nan, 5.0, 14.0, np.nan]))
    loss = make_branch("real_plasmon", z=(10.0, 20.0),
                       interaction_kind="real_plasmon", weight=0.4 if absolute else 4.0,
                       blocked_z=incident.blocked_z.copy())
    result = make_result(incident, [direct, loss])
    result.simulation.metrics["branch_weights_are_absolute"] = absolute
    plane = sample_beam_plane(result, 15.0)
    budget = plane_interaction_budget(result, 15.0)
    assert plane.provenance == "Optical reference" and plane.weights_valid
    assert plane.total_source_fraction == pytest.approx(budget.source_fraction_at_plane)
    for channel in budget.channels:
        assert plane.source_fraction[plane.interaction_key == channel.key].sum() == pytest.approx(
            channel.source_fraction_at_plane)
    np.testing.assert_array_equal(plane.column_index, [0, 3, 4, 6, 7])
    assert plane.total_column_count == 8
    assert any("Optical-reference population" in text for text in plane.diagnostics)


def test_reference_cannot_resurrect_incident_rays_stopped_before_sample():
    incident = make_branch(blocked_z=np.array([2.0, np.nan, np.nan, np.nan]))
    result = make_result(incident, [make_branch("000", z=(10.0, 20.0))])
    plane = sample_beam_plane(result, 15.0)
    assert plane.weights_valid and plane.total_source_fraction == pytest.approx(0.75)
    np.testing.assert_array_equal(plane.column_index, [1, 2, 3])


def test_detailed_compact_repeated_ancestors_use_conditional_not_incident_weights():
    elastic = make_branch(
        "specimen_elastic:real_plasmon", count=3, z=(10.0, 20.0), weight=0.4,
        interaction_kind="sample_region_elastic", ray_weight=np.array([0.0, 1.0, 3.0]),
        source_ray_id=np.array([99, 57, 57]), source_azimuth_rad=np.array([0.3, 0.5, 0.5]),
        blocked_z=np.array([np.nan, 15.0, np.nan]),
    )
    primary = make_branch(
        "specimen_primary:000", count=2, z=(10.0, 20.0), weight=0.2,
        interaction_kind="sample_region_primary", ray_weight=np.array([0.75, 0.25]),
    )
    result = detailed_result([elastic, primary])
    # The detailed exit already owns absolute source survival; do not apply this twice.
    result.simulation.incident.ray_weight[:] = [0.01, 0.01, 0.01, 0.01]
    plane = sample_beam_plane(result, 15.0)
    after = sample_beam_plane(result, 15.1)
    assert plane.provenance == "Specimen exit" and plane.weights_valid
    np.testing.assert_allclose(plane.source_fraction, [0.1, 0.3, 0.15, 0.05])
    np.testing.assert_array_equal(plane.source_ray_id[:2], [57, 57])
    np.testing.assert_array_equal(plane.column_index, [1, 2, 3, 4])
    np.testing.assert_array_equal(after.column_index, [2, 3, 4])
    assert plane.total_column_count == 5 and plane.current_pa == pytest.approx(12.0)
    assert after.total_source_fraction == pytest.approx(0.5)
    np.testing.assert_array_equal(plane.interaction_key[:2], ["elastic+real_plasmon"] * 2)
    np.testing.assert_array_equal(plane.interaction_symbol[:2], ["star"] * 2)
    assert any("conditional normalisation" in text for text in plane.diagnostics)
    assert not any("source fractions are ambiguous" in text for text in plane.diagnostics)


def test_incident_boundary_remains_incident_even_when_detailed_exit_is_empty():
    result = detailed_result([])
    at_sample = sample_beam_plane(result, 10.0)
    downstream = sample_beam_plane(result, 10.001)
    assert at_sample.provenance == "Incident" and at_sample.ray_count == 4
    assert downstream.provenance == "Specimen exit" and downstream.weights_valid
    assert downstream.ray_count == 0 and downstream.total_source_fraction == 0.0
    assert downstream.current_pa == 0.0 and downstream.status == "No reaching rays"


def test_stale_detailed_exit_uses_labelled_reference_not_stale_population():
    detailed = make_branch("specimen_primary:000", count=1, z=(10.0, 20.0), weight=0.2)
    result = detailed_result([detailed])
    result.signatures["sample_downstream"] = "changed"
    plane = sample_beam_plane(result, 15.0)
    assert plane.provenance == "Optical reference" and plane.ray_count == 4
    assert plane.total_source_fraction == pytest.approx(1.0)


@pytest.mark.parametrize("selected", [-1.0, 21.0])
def test_no_extrapolation_of_unstopped_histories(selected):
    plane = sample_beam_plane(make_result(), selected)
    assert plane.ray_count == 0 and not plane.weights_valid
    assert plane.total_source_fraction is None and plane.current_pa is None
    assert any("no extrapolation" in text for text in plane.diagnostics)


def test_stopped_bundle_remains_known_zero_beyond_last_history():
    post = make_branch("000", z=(10.0, 20.0), blocked_z=np.full(4, 18.0))
    plane = sample_beam_plane(make_result(branches=[post]), 21.0)
    assert plane.weights_valid and plane.total_source_fraction == 0.0


@pytest.mark.parametrize("invalid", [None, [0.0, 0.0], [0.2], [np.nan, 1.0], [-0.2, 1.2]])
def test_invalid_detailed_weights_keep_geometry_without_inventing_signal(invalid):
    child = make_branch("specimen_primary:000", count=2, z=(10.0, 20.0),
                        ray_weight=invalid, weight=0.5)
    plane = sample_beam_plane(detailed_result([child]), 15.0)
    assert plane.ray_count == (0 if invalid == [0.0, 0.0] else 2)
    assert not plane.weights_valid
    assert np.all(np.isnan(plane.source_fraction))
    assert plane.total_source_fraction is None and plane.current_pa is None
    with pytest.raises(ValueError, match="probabilities"):
        spatial_histogram(plane)
    with pytest.raises(ValueError, match="probabilities"):
        angular_histogram(plane)


@pytest.mark.parametrize("invalid", [np.nan, -0.1, None])
def test_invalid_optical_probability_does_not_get_normalised_to_one(invalid):
    post = make_branch("000", z=(10.0, 20.0), weight=invalid)
    plane = sample_beam_plane(make_result(branches=[post]), 15.0)
    assert plane.ray_count == 4 and not plane.weights_valid
    assert plane.total_source_fraction is None


def test_zero_relative_probabilities_are_not_replaced_by_uniform_probabilities():
    post = make_branch("000", z=(10.0, 20.0), weight=0.0)
    result = make_result(branches=[post])
    result.simulation.metrics["branch_weights_are_absolute"] = False
    assert not sample_beam_plane(result, 15.0).weights_valid
    result.simulation.metrics["branch_weights_are_absolute"] = True
    plane = sample_beam_plane(result, 15.0)
    assert plane.weights_valid and plane.ray_count == 0 and plane.current_pa == 0.0


def test_missing_source_weights_leave_quantitative_readout_unavailable():
    incident = make_branch(ray_weight=None)
    plane = sample_beam_plane(make_result(incident), 5.0)
    assert not plane.weights_valid
    assert plane.ray_count == 4
    assert np.all(np.isnan(plane.source_fraction))
    assert plane.current_pa is None
    assert any("probabilities are unavailable" in text for text in plane.diagnostics)


def test_missing_branch_probability_convention_is_not_assumed_relative():
    result = make_result()
    del result.simulation.metrics["branch_weights_are_absolute"]
    plane = sample_beam_plane(result, 15.0)
    assert not plane.weights_valid and plane.current_pa is None
    assert any("probability convention" in text for text in plane.diagnostics)


@pytest.mark.parametrize("weights", [[0.1, 0.15, 0.25, 0.0], [1.0, 2.0, 3.0, 0.0], [0.0] * 4])
def test_ambiguous_explicit_source_normalisation_is_not_silently_assumed(weights):
    plane = sample_beam_plane(make_result(make_branch(ray_weight=np.array(weights))), 5.0)
    assert not plane.weights_valid and plane.current_pa is None
    assert plane.ray_count == np.count_nonzero(weights)
    assert any("source weights do not sum to one" in text for text in plane.diagnostics)


@pytest.mark.parametrize("invalid", [None, np.nan, np.inf, -2.0, False, np.bool_(True), "bad"])
def test_current_is_never_invented_without_valid_cached_source_metric(invalid):
    result = make_result()
    result.simulation.metrics["effective_source_current_pa"] = invalid
    result.state_snapshot.source_current_pa = 999.0
    plane = sample_beam_plane(result, 5.0)
    assert plane.weights_valid and plane.total_source_fraction == pytest.approx(1.0)
    assert plane.source_current_pa is None and plane.current_pa is None


def test_full_weighted_population_is_not_limited_to_display_representatives():
    incident = make_branch(count=5003)
    incident.ray_weight[:3] = 0.0
    incident.ray_weight[3:] = 1.0 / 5000
    plane = sample_beam_plane(make_result(incident), 5.0)
    assert plane.ray_count == 5000 and plane.total_column_count == 5003
    spatial, _, _ = spatial_histogram(plane)
    angular, _ = angular_histogram(plane)
    assert spatial.shape == (64, 64) and angular.shape == (64,)
    assert spatial.sum() == pytest.approx(plane.total_source_fraction, abs=1e-14)
    assert angular.sum() == pytest.approx(plane.total_source_fraction, abs=1e-14)


def test_histograms_preserve_absolute_mass_and_crop_without_normalising():
    incident = make_branch(ray_weight=np.array([0.1, 0.2, 0.3, 0.4]))
    incident.x[:] = [-2.0e-6, -1.0e-6, 0.0, 2.0e-6]
    incident.y[:] = 0.0
    incident.tx[:] = [0.0, 0.001, 0.002, 0.003]
    plane = sample_beam_plane(make_result(incident), 5.0)
    theta = np.arctan([0.0, 0.001, 0.002, 0.003]) * 1000.0
    np.testing.assert_allclose(plane.theta_mrad, theta)
    hist, x_edges, y_edges = spatial_histogram(plane, bins=8)
    assert x_edges[0] == -x_edges[-1] and y_edges[0] == -y_edges[-1]
    assert hist.sum() == pytest.approx(1.0)
    cropped, _, _ = spatial_histogram(plane, range_m=((-1.0e-6, 1.0e-6), (-1.0e-6, 1.0e-6)))
    assert cropped.sum() == pytest.approx(0.5)
    angular, edges = angular_histogram(plane, range_mrad=(0.0, theta[2]))
    assert angular.sum() == pytest.approx(0.6)
    assert edges[-1] == theta[2]  # The last bin includes its upper boundary.
    for values in (hist, x_edges, y_edges, angular, edges):
        with pytest.raises(ValueError):
            values.setflags(write=True)


def test_projected_spatial_histogram_rotates_coordinates_not_probability():
    incident = make_branch(count=1)
    incident.x[:] = 2.0e-6
    incident.y[:] = 1.0e-6
    plane = sample_beam_plane(make_result(incident), 5.0)
    ranges = ((0.5e-6, 1.5e-6), (-2.5e-6, -1.5e-6))
    unrotated, _, _ = spatial_histogram(plane, range_m=ranges)
    rotated, _, _ = spatial_histogram(plane, projection_angle_deg=90.0, range_m=ranges)
    assert unrotated.sum() == 0.0 and rotated.sum() == pytest.approx(1.0)


def test_float32_history_converts_only_selected_rows_not_entire_cache(monkeypatch):
    branch = make_branch()
    for name in ("x", "y", "tx", "ty"):
        setattr(branch, name, getattr(branch, name).astype(np.float32))
    original_asarray = np.asarray

    def no_full_conversion(values, dtype=None, *args, **kwargs):
        if isinstance(values, np.ndarray) and values.ndim == 2 and dtype is float:
            pytest.fail("Display interpolation must not promote the complete cached history")
        return original_asarray(values, dtype=dtype, *args, **kwargs)

    monkeypatch.setattr(np, "asarray", no_full_conversion)
    points = _interpolate(branch, 4, 5.0)
    assert all(values.dtype == np.dtype(float) for values in points)
    np.testing.assert_allclose(points[0], branch.x[0])


@pytest.mark.parametrize("changes", [
    {"z": np.array([0.0, 0.0])}, {"x": [[0.0], [0.0, 1.0]]},
    {"tx": np.zeros((1, 4))}, {"blocked_z": np.zeros(3)}, {"z": "bad"},
])
def test_malformed_cached_history_returns_diagnostics_not_signal(changes):
    result = make_result()
    for name, value in changes.items():
        setattr(result.simulation.incident, name, value)
    plane = sample_beam_plane(result, 5.0)
    assert not plane.weights_valid and plane.total_source_fraction is None
    assert plane.diagnostics


def test_nonfinite_reaching_point_is_omitted_and_readout_marked_partial():
    incident = make_branch()
    incident.x[:, 1] = np.nan
    plane = sample_beam_plane(make_result(incident), 5.0)
    assert plane.ray_count == 3 and plane.status == "Partial data"
    assert not plane.weights_valid and plane.current_pa is None


@pytest.mark.parametrize("value", [None, np.nan, "bad"])
def test_invalid_plane_is_safe(value):
    plane = sample_beam_plane(make_result(), value)
    assert not plane.weights_valid and plane.ray_count == 0


def test_unavailable_result_is_safe():
    plane = sample_beam_plane(None, 5.0)
    assert plane.provenance == "Unavailable" and not plane.weights_valid


def test_malformed_cached_metrics_do_not_crash_or_invent_current():
    result = make_result()
    result.simulation.metrics = np.array([1.0, 2.0])
    incident = sample_beam_plane(result, 5.0)
    assert incident.weights_valid and incident.current_pa is None
    downstream = sample_beam_plane(result, 15.0)
    assert not downstream.weights_valid and downstream.current_pa is None


@pytest.mark.parametrize("bins", [0, -1, 1025, 1.5, True])
def test_histograms_reject_invalid_bin_counts(bins):
    plane = sample_beam_plane(make_result(), 5.0)
    with pytest.raises(ValueError):
        angular_histogram(plane, bins=bins)
    with pytest.raises(ValueError):
        spatial_histogram(plane, bins=bins)


def spot_plane(x, y, weights):
    """Small physical coordinate/probability fixture, with a 20 pA source."""
    return replace(sample_beam_plane(make_result(), 5.0),
                   x_m=np.asarray(x, float), y_m=np.asarray(y, float),
                   source_fraction=np.asarray(weights, float))


def spot_moments(raster, *, smooth=False):
    values = raster.display_fraction if smooth else raster.raw_fraction
    x = (raster.x_edges_m[:-1] + raster.x_edges_m[1:]) / 2
    y = (raster.y_edges_m[:-1] + raster.y_edges_m[1:]) / 2
    total = values.sum()
    return total, (float(np.sum(values * x[None, :]) / total),
                   float(np.sum(values * y[:, None]) / total))


def test_spot_bilinear_subpixel_weights_and_duplicate_hits_are_conservative():
    plane = spot_plane([0.25e-6] * 3, [0.75e-6] * 3, [0.1, 0.2, 0.3])
    raster = rasterize_beam_spot(plane, bins=4,
                                range_m=((-2e-6, 2e-6), (-2e-6, 2e-6)))
    expected = np.zeros((4, 4))
    expected[2:4, 1:3] = np.array([[3 / 16, 9 / 16], [1 / 16, 3 / 16]]) * 0.6
    np.testing.assert_allclose(raster.raw_fraction, expected, atol=1e-16)
    total, centroid = spot_moments(raster)
    assert total == pytest.approx(0.6, abs=1e-15)
    np.testing.assert_allclose(centroid, [0.25e-6, 0.75e-6], atol=1e-21)
    assert raster.centroid_m == pytest.approx(centroid, abs=1e-21)
    assert raster.current_pa == pytest.approx(12.0)
    assert raster.ray_count == 3
    assert raster.effective_ray_count == pytest.approx(0.6**2 / (0.1**2 + 0.2**2 + 0.3**2))
    assert raster.pixel_area_m2 == pytest.approx(1e-12)
    for array in (raster.raw_fraction, raster.display_fraction,
                  raster.x_edges_m, raster.y_edges_m):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize("bins", [32, 64, 128, 512])
@pytest.mark.parametrize("sigma", [0.0, 0.8, 3.0])
def test_spot_grid_and_display_smoothing_preserve_translated_weighted_beam(bins, sigma):
    plane = spot_plane(np.array([-.35, .12, .61]) * 1e-6,
                       np.array([.22, -.18, .44]) * 1e-6, [.1, .2, .4])
    before = (plane.x_m.copy(), plane.y_m.copy(), plane.source_fraction.copy())
    raster = rasterize_beam_spot(plane, bins=bins, smoothing_sigma_px=sigma)
    expected = np.array([np.dot(plane.x_m, plane.source_fraction),
                         np.dot(plane.y_m, plane.source_fraction)]) / 0.7
    for smooth in (False, True):
        total, centroid = spot_moments(raster, smooth=smooth)
        assert total == pytest.approx(0.7, abs=4e-15)
        np.testing.assert_allclose(centroid, expected, atol=3e-21, rtol=2e-14)
    np.testing.assert_allclose(raster.centroid_m, expected, atol=2e-22)
    assert raster.total_source_fraction == pytest.approx(.7)
    assert raster.viewport_source_fraction == pytest.approx(.7)
    assert raster.current_pa == pytest.approx(14.)
    assert raster.ray_count == 3
    assert raster.effective_ray_count == pytest.approx(.7**2 / (.1**2 + .2**2 + .4**2))
    for previous, current in zip(before, (plane.x_m, plane.y_m, plane.source_fraction)):
        np.testing.assert_array_equal(previous, current)
    # Changing smoothing changes only displayed density, not deposition.
    plain = rasterize_beam_spot(plane, bins=bins,
        range_m=((raster.x_edges_m[0], raster.x_edges_m[-1]),
                 (raster.y_edges_m[0], raster.y_edges_m[-1])))
    np.testing.assert_array_equal(plain.raw_fraction, raster.raw_fraction)


def test_spot_crop_does_not_rescale_and_display_halo_never_reflects_or_wraps():
    # A particle on the left viewport edge has half its linear support outside.
    # Another farther outside cannot be reflected back into the viewport.
    plane = spot_plane([-2e-6, -20e-6], [.5e-6, .5e-6], [.25, .5])
    raster = rasterize_beam_spot(plane, bins=4,
                                range_m=((-2e-6, 2e-6), (-2e-6, 2e-6)))
    assert raster.total_source_fraction == pytest.approx(.75)
    assert raster.viewport_source_fraction == pytest.approx(.25)
    assert raster.raw_visible_fraction == pytest.approx(.125)
    assert raster.display_visible_fraction == pytest.approx(.125)
    assert raster.raw_fraction[2, 0] == pytest.approx(.125)
    assert not np.any(raster.raw_fraction[:, -1])
    assert raster.centroid_m == pytest.approx((-14e-6, .5e-6), abs=1e-21)
    smoothed = rasterize_beam_spot(plane, bins=16, smoothing_sigma_px=1,
                                  range_m=((-2e-6, 2e-6), (-2e-6, 2e-6)))
    assert smoothed.total_source_fraction == pytest.approx(.75)
    assert smoothed.display_visible_fraction == pytest.approx(.125, abs=1e-15)
    assert not np.any(smoothed.display_fraction[:, -1])


def test_spot_smoothing_reads_halo_outside_view_instead_of_smoothing_a_crop():
    # Coordinates coincide with a pixel centre just outside a four-pixel view.
    plane = spot_plane([-2.5e-6], [.5e-6], [.6])
    limits = ((-2e-6, 2e-6), (-2e-6, 2e-6))
    plain = rasterize_beam_spot(plane, bins=4, range_m=limits)
    smooth = rasterize_beam_spot(plane, bins=4, range_m=limits, smoothing_sigma_px=.5)
    assert plain.raw_visible_fraction == pytest.approx(0.)
    assert smooth.viewport_source_fraction == pytest.approx(0.)
    assert 0. < smooth.display_visible_fraction < .6
    assert smooth.total_source_fraction == .6 and smooth.centroid_m == (-2.5e-6, .5e-6)
    assert smooth.display_fraction[:, -1].sum() == 0.
    np.testing.assert_array_equal(smooth.raw_fraction, plain.raw_fraction)


def test_spot_projection_rotates_true_centroid_and_axes_without_changing_flux():
    plane = spot_plane([2e-6], [1e-6], [.4])
    raster = rasterize_beam_spot(plane, bins=32, projection_angle_deg=90.,
                                smoothing_sigma_px=1.)
    np.testing.assert_allclose(raster.centroid_m, [1e-6, -2e-6], atol=1e-21)
    total, centroid = spot_moments(raster, smooth=True)
    assert total == pytest.approx(.4, abs=1e-15)
    np.testing.assert_allclose(centroid, [1e-6, -2e-6], atol=1e-21)


def test_spot_physical_interception_changes_current_without_renormalising():
    branch = make_branch(ray_weight=np.array([.1, .2, .3, .4]),
                         blocked_z=np.array([np.nan, 5., np.nan, np.nan]))
    result = make_result(branch)
    before = rasterize_beam_spot(sample_beam_plane(result, 4.9), smoothing_sigma_px=1.)
    after = rasterize_beam_spot(sample_beam_plane(result, 5.1), smoothing_sigma_px=1.)
    assert before.display_visible_fraction == pytest.approx(1.)
    assert after.display_visible_fraction == pytest.approx(.8)
    assert before.current_pa == pytest.approx(20.)
    assert after.current_pa == pytest.approx(16.)
    assert after.ray_count == 3
    assert after.centroid_m != before.centroid_m


def test_spot_empty_zero_weight_and_subnormal_probability_are_not_invented():
    for plane in (spot_plane([], [], []), spot_plane([1e-6], [1e-6], [0.])):
        raster = rasterize_beam_spot(plane, smoothing_sigma_px=1.)
        assert raster.total_source_fraction == 0.
        assert raster.raw_visible_fraction == raster.display_visible_fraction == 0.
        assert raster.centroid_m is None and raster.effective_ray_count == 0.
    tiny = spot_plane([1e-6, 3e-6], [0., 0.], [1e-300, 3e-300])
    raster = rasterize_beam_spot(tiny)
    assert raster.total_source_fraction == 4e-300
    assert raster.effective_ray_count == pytest.approx(1.6)
    np.testing.assert_allclose(raster.centroid_m, [2.5e-6, 0.], atol=1e-21)


@pytest.mark.parametrize("changes", [
    {"bins": 1}, {"bins": 1025}, {"bins": np.inf}, {"bins": 1.5},
    {"bins": True}, {"bins": np.bool_(True)}, {"bins": None},
    {"smoothing_sigma_px": np.nan}, {"smoothing_sigma_px": -1},
    {"smoothing_sigma_px": 8.1}, {"smoothing_sigma_px": True},
    {"projection_angle_deg": np.inf}, {"projection_angle_deg": None},
    {"range_m": ((0, 0), (0, 1))}, {"range_m": ((0, np.inf), (0, 1))},
    {"range_m": ((0, 1e-300), (0, 1e-300))}, {"range_m": ((0, 1),)},
    {"bins": 2, "smoothing_sigma_px": 3.},
])
def test_spot_rejects_invalid_and_unresolved_display_parameters(changes):
    with pytest.raises(ValueError):
        rasterize_beam_spot(spot_plane([0.], [0.], [1.]), **changes)


@pytest.mark.parametrize("changes", [
    {"weights_valid": False}, {"x_m": np.array([np.nan])},
    {"x_m": np.array([0., 1.])}, {"source_fraction": np.array([-.5])},
    {"source_fraction": np.array([np.nan])}, {"source_fraction": np.array([1.1])},
])
def test_spot_rejects_unavailable_or_invalid_particle_weights(changes):
    with pytest.raises(ValueError):
        rasterize_beam_spot(replace(spot_plane([0.], [0.], [1.]), **changes))


def test_spot_extreme_offscreen_particles_do_not_overflow_integer_deposition():
    plane = spot_plane([0., 1e200, -1e200], [0., 1e200, -1e200], [.2, .3, .5])
    with np.errstate(all="raise"):
        raster = rasterize_beam_spot(plane, bins=128,
                                    range_m=((-1e-12, 1e-12), (-1e-12, 1e-12)))
    assert raster.raw_visible_fraction == pytest.approx(.2)
    assert raster.total_source_fraction == 1.
