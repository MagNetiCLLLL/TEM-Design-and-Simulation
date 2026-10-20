"""Magnitude textures retain vector cancellation, masks, and a fixed scale."""
import numpy as np
import pytest

from temsim.field_density_texture import (
    HATCH_SPACINGS_PX, HATCH_THRESHOLDS, SectionMagnitudeGrid, hatch_mask,
    section_grid_axes, from_longitudinal, to_longitudinal,
)


def _points(axis0=(0., 1.), axis1=(0., 1.), section="XY"):
    first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[section]
    xx, yy = np.meshgrid(axis0, axis1, indexing="ij")
    points = np.zeros((xx.size, 3))
    points[:, first], points[:, second], points[:, normal] = xx.ravel(), yy.ravel(), .25
    return points


def _query(x, y, section="XY"):
    first, second, normal = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}[section]
    point = np.zeros(3)
    point[first], point[second], point[normal] = x, y, .25
    return point


@pytest.mark.parametrize("section", ["XY", "XZ", "YZ"])
def test_axes_are_physical_coordinates_from_complete_unfiltered_grid(section):
    points = _points((-2., 0., 2.), (1., 2., 3., 4.), section)
    axes = section_grid_axes(points[::-1], section)
    np.testing.assert_array_equal(axes[0], [-2., 0., 2.])
    np.testing.assert_array_equal(axes[1], [1., 2., 3., 4.])
    assert section_grid_axes(points[:-1], section) is None
    assert section_grid_axes(points, "3D") is None


def test_duplicate_nonplanar_and_gapped_coordinates_do_not_infer_a_grid():
    points = _points()
    assert section_grid_axes(points[[0, 1, 2, 2]], "XY") is None
    points[-1, 2] += 1.
    assert section_grid_axes(points, "XY") is None
    assert section_grid_axes(_points((0., 1., 3.)), "XY") is None
    assert section_grid_axes(_points((0.,)), "XY") is None


@pytest.mark.parametrize("section", ["XY", "XZ", "YZ"])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_magnitude_includes_all_three_components_without_direction_sign(section, sign):
    points = _points(section=section)
    values = np.tile(np.array([3., 4., 12.]) * sign, (len(points), 1))
    grid = SectionMagnitudeGrid(points, values, np.ones(len(points), dtype=bool), section)
    magnitude, supported, valid = grid.evaluate([_query(.3, .7, section)])
    np.testing.assert_allclose(magnitude, [13.])
    assert supported.tolist() == [True] and valid.tolist() == [True]


def test_vector_interpolation_preserves_cancellation_before_taking_norm():
    points = _points()
    values = np.outer(1. - 2. * points[:, 0], [3., 4., 12.])
    grid = SectionMagnitudeGrid(points, values, np.ones(4, dtype=bool), "XY")
    magnitude, supported, valid = grid.evaluate([
        _query(0., .5), _query(.25, .5), _query(.5, .5), _query(1., .5)])
    np.testing.assert_allclose(magnitude, [13., 6.5, 0., 13.])
    assert supported.all() and valid.all()


def test_bilinear_interpolation_uses_physical_coordinates():
    points = _points((0., 2.), (0., 4.))
    values = np.column_stack((points[:, 0], points[:, 1], 2 * points[:, 0] - points[:, 1]))
    grid = SectionMagnitudeGrid(points, values, np.ones(4, dtype=bool), "XY")
    magnitude, supported, valid = grid.evaluate([_query(.5, 3.)])
    np.testing.assert_allclose(magnitude, [np.linalg.norm([.5, 3., -2.])])
    assert supported.all() and valid.all()


def test_constant_field_is_bitwise_invariant_across_coordinates_and_grid_refinement():
    vector = np.array([.05, .1, .2])
    expected = np.hypot.reduce(vector)
    query = np.array([_query(x, y) for x, y in zip(
        np.linspace(.001, .999, 997), np.linspace(.9987, .003, 997))])
    screen_x = np.arange(len(query), dtype=float) + .5
    screen_y = screen_x * .75
    expected_hatch = hatch_mask(expected, 4. * expected, screen_x, screen_y)
    for count in (3, 9, 31):
        points = _points(np.linspace(0., 1., count), np.linspace(0., 1., count))
        grid = SectionMagnitudeGrid(points, np.tile(vector, (len(points), 1)),
                                    np.ones(len(points), dtype=bool), "XY")
        magnitude, supported, valid = grid.evaluate(query)
        np.testing.assert_array_equal(magnitude, np.full(len(query), expected))
        assert supported.all() and valid.all()
        np.testing.assert_array_equal(
            hatch_mask(magnitude, 4. * expected, screen_x, screen_y), expected_hatch)


@pytest.mark.parametrize("removed_axis", [0, 1])
def test_original_axes_preserve_entire_missing_rows_and_columns(removed_axis):
    points = _points((0., 1., 2., 3.), (0., 1., 2., 3.))
    axes = section_grid_axes(points, "XY")
    keep = points[:, removed_axis] != 1.
    filtered = points[keep]
    values = np.tile([0., 0., 7.], (len(filtered), 1))
    grid = SectionMagnitudeGrid(filtered, values, np.ones(len(filtered), dtype=bool),
                                "XY", grid_axes_m=axes)
    queries = np.array([_query(.5, 2.5), _query(1.5, 2.5), _query(2.5, 2.5)])
    if removed_axis == 1:
        queries[:, [0, 1]] = queries[:, [1, 0]]
    magnitude, supported, valid = grid.evaluate(queries)
    assert supported.tolist() == [False, False, True]
    assert valid.tolist() == [False, False, True]
    assert np.isnan(magnitude[:2]).all() and magnitude[2] == pytest.approx(7.)


def test_missing_corner_cannot_be_inferred_without_original_grid_metadata():
    points = _points()[:-1]
    grid = SectionMagnitudeGrid(points, np.ones((3, 3)), np.ones(3, dtype=bool), "XY")
    magnitude, supported, valid = grid.evaluate([_query(.1, .1)])
    assert np.isnan(magnitude).all() and not supported.any() and not valid.any()


def test_support_and_validity_distinguish_missing_invalid_and_valid_zero():
    points = _points((0., 1., 2., 3., 4.), (0., 1.))
    axes = section_grid_axes(points, "XY")
    keep = points[:, 0] != 1.
    points = points[keep]
    values = np.zeros_like(points)
    validity = points[:, 0] != 2.
    grid = SectionMagnitudeGrid(points, values, validity, "XY", grid_axes_m=axes)
    magnitude, supported, valid = grid.evaluate([
        _query(.5, .5), _query(2.5, .5), _query(3.5, .5)])
    assert supported.tolist() == [False, True, True]
    assert valid.tolist() == [False, False, True]
    assert np.isnan(magnitude[:2]).all() and magnitude[2] == 0.


def test_invalid_or_nonfinite_corner_is_conservative_even_at_opposite_node():
    points = _points()
    values = np.ones_like(points)
    values[-1, 0] = np.nan
    grid = SectionMagnitudeGrid(points, values, np.ones(4, dtype=bool), "XY")
    magnitude, supported, valid = grid.evaluate([_query(0., 0.), _query(.5, .5)])
    assert supported.all() and not valid.any() and np.isnan(magnitude).all()


def test_domain_boundaries_outside_plane_and_nonfinite_queries():
    points = _points()
    grid = SectionMagnitudeGrid(points, np.ones_like(points), np.ones(4, dtype=bool), "XY")
    queries = np.array([_query(0., 0.), _query(1., 1.), _query(-.001, .5),
                        _query(1.001, .5), _query(.5, .5), _query(np.nan, .5)])
    queries[-2, 2] += .001
    magnitude, supported, valid = grid.evaluate(queries)
    assert supported.tolist() == [True, True, False, False, False, False]
    np.testing.assert_array_equal(valid, supported)
    assert np.isnan(magnitude[2:]).all()


def test_3d_and_empty_samples_have_no_interpolated_support():
    points = _points()
    for section, selected in (("3D", points), ("XY", np.empty((0, 3)))):
        grid = SectionMagnitudeGrid(selected, np.zeros_like(selected),
                                    np.ones(len(selected), dtype=bool), section)
        magnitude, supported, valid = grid.evaluate([_query(.5, .5)])
        assert np.isnan(magnitude).all() and not supported.any() and not valid.any()


@pytest.mark.parametrize("angle", [0., 37., 90., 180.])
def test_longitudinal_transforms_preserve_full_v_offset_and_input(angle):
    local = np.array([[-.003, .0007, .01], [.001, -.0002, .04], [0., 1e-20, .02]])
    original = local.copy()
    global_points = from_longitudinal(local, angle)
    recovered = to_longitudinal(global_points, angle)
    np.testing.assert_allclose(recovered, local, rtol=1e-14, atol=1e-18)
    np.testing.assert_array_equal(local, original)
    assert np.all(recovered[:, 1] != 0.)
    np.testing.assert_allclose(from_longitudinal(recovered, angle), global_points,
                               rtol=1e-14, atol=1e-18)


@pytest.mark.parametrize("angle", [0., 37., 90., 180.])
def test_rotated_uz_grid_uses_global_field_components_and_accepts_roundtrip_edges(angle):
    local = _points((-.002, 0., .002), (.01, .03, .05), "XZ")
    local[:, 1] = 0.
    points = from_longitudinal(local, angle)
    axes = section_grid_axes(points, "UZ", angle_deg=angle)
    np.testing.assert_allclose(axes[0], [-.002, 0., .002], rtol=1e-14, atol=1e-18)
    np.testing.assert_array_equal(axes[1], [.01, .03, .05])

    def field(xyz):
        x, y, z = xyz.T
        return np.column_stack((.3 + 2 * x + 3 * y, .4 + x - y, 1.2 + 2 * z))

    grid = SectionMagnitudeGrid(points, field(points), np.ones(len(points), dtype=bool),
                                "UZ", grid_axes_m=axes, angle_deg=angle)
    queries = from_longitudinal(np.array([[0., 0., .02], [.001, 0., .04],
                                         [-.002, 0., .01], [.002, 0., .05]]), angle)
    # Repeated conversions reproduce the canvas's forward/inverse projection
    # roundoff, including queries on the domain boundary.
    queries = from_longitudinal(to_longitudinal(queries, angle), angle)
    queries = from_longitudinal(to_longitudinal(queries, angle), angle)
    magnitude, supported, valid = grid.evaluate(queries)
    assert supported.all() and valid.all()
    np.testing.assert_allclose(magnitude, np.hypot.reduce(field(queries), axis=1), rtol=1e-14)
    # This point has identical global coordinates for every observation angle.
    np.testing.assert_allclose(magnitude[0], np.linalg.norm([.3, .4, 1.24]), rtol=1e-14)


@pytest.mark.parametrize("angle", [0., 37., 90., 180.])
def test_uz_never_flattens_true_off_plane_samples_or_queries(angle):
    local = _points((-.002, 0., .002), (.01, .03), "XZ")
    local[:, 1] = 0.
    points = from_longitudinal(local, angle)
    grid = SectionMagnitudeGrid(points, np.ones_like(points),
                                np.ones(len(points), dtype=bool), "UZ", angle_deg=angle)
    queries = from_longitudinal(np.array([[.001, 0., .02], [.001, 1e-9, .02],
                                         [0., 1e-20, .02]]), angle)
    magnitude, supported, valid = grid.evaluate(queries)
    assert supported.tolist() == [True, False, False]
    np.testing.assert_array_equal(valid, supported)
    assert np.isnan(magnitude[1:]).all()
    off_plane = local.copy()
    off_plane[0, 1] = 1e-9
    off_plane = from_longitudinal(off_plane, angle)
    assert section_grid_axes(off_plane, "UZ", angle_deg=angle) is None
    rejected = SectionMagnitudeGrid(off_plane, np.ones_like(off_plane),
                                    np.ones(len(off_plane), dtype=bool), "UZ", angle_deg=angle)
    assert not rejected.evaluate(queries)[1].any()


@pytest.mark.parametrize("angle", [0., 37., 90., 180.])
@pytest.mark.parametrize("removed_axis", [0, 2])
def test_uz_rotated_grid_retains_missing_rows_invalid_cells_and_valid_zero(angle, removed_axis):
    local = _points(np.arange(5.) * .001, np.arange(5.) * .001, "XZ")
    local[:, 1] = 0.
    points = from_longitudinal(local, angle)
    axes = section_grid_axes(points, "UZ", angle_deg=angle)
    keep = local[:, removed_axis] != .001
    validity = local[keep, removed_axis] != .002
    grid = SectionMagnitudeGrid(points[keep], np.zeros_like(points[keep]), validity,
                                "UZ", grid_axes_m=axes, angle_deg=angle)
    query = np.array([[.0005, 0., .0035], [.0015, 0., .0035],
                      [.0025, 0., .0035], [.0035, 0., .0035]])
    if removed_axis == 2:
        query[:, [0, 2]] = query[:, [2, 0]]
    magnitude, supported, valid = grid.evaluate(from_longitudinal(query, angle))
    assert supported.tolist() == [False, False, True, True]
    assert valid.tolist() == [False, False, False, True]
    assert np.isnan(magnitude[:3]).all() and magnitude[3] == 0.


@pytest.mark.parametrize("angle", [0., 37., 90., 180.])
def test_uz_constant_field_and_hatch_threshold_survive_rotation_and_refinement(angle):
    vector = np.array([.05, .1, .2])
    expected = np.hypot.reduce(vector)
    query = np.column_stack((np.linspace(-.002, .002, 997), np.zeros(997),
                             np.linspace(.01, .05, 997)))
    query = from_longitudinal(query, angle)
    screen_x = np.arange(len(query), dtype=float) + .5
    screen_y = screen_x * .75
    expected_hatch = hatch_mask(expected, 4. * expected, screen_x, screen_y)
    for count in (9, 31):
        local = _points(np.linspace(-.002, .002, count), np.linspace(.01, .05, count), "XZ")
        local[:, 1] = 0.
        points = from_longitudinal(local, angle)
        grid = SectionMagnitudeGrid(points, np.tile(vector, (len(points), 1)),
                                    np.ones(len(points), dtype=bool), "UZ", angle_deg=angle)
        magnitude, supported, valid = grid.evaluate(query)
        assert supported.all() and valid.all()
        np.testing.assert_array_equal(magnitude, np.full(len(query), expected))
        np.testing.assert_array_equal(
            hatch_mask(magnitude, 4. * expected, screen_x, screen_y), expected_hatch)


def test_hatch_density_is_nested_monotonic_and_saturates_at_twice_reference():
    xx, yy = np.meshgrid(np.arange(256), np.arange(256))
    ratios = [0., .01, .125, .25, .5, 1., 2., 100.]
    masks = [hatch_mask(ratio * 7., 7., xx, yy) for ratio in ratios]
    assert not masks[0].any()
    for lower, upper in zip(masks[:-1], masks[1:]):
        assert np.all(~lower | upper)
    counts = [mask.sum() for mask in masks]
    assert all(lower < upper for lower, upper in zip(counts[:-2], counts[1:-1]))
    np.testing.assert_array_equal(masks[-2], masks[-1])
    assert counts[-1] < xx.size / 2
    assert HATCH_SPACINGS_PX[-1] == 4. and HATCH_THRESHOLDS[-1] == 2.


def test_hatch_fixed_reference_thresholds_and_unknown_values():
    x, y = np.meshgrid(np.arange(128), np.arange(128))
    for ratio in HATCH_THRESHOLDS:
        below = hatch_mask(np.nextafter(ratio, 0.), 1., x, y)
        at = hatch_mask(ratio, 1., x, y)
        assert at.sum() > below.sum()
    np.testing.assert_array_equal(hatch_mask(.5, 1., x, y), hatch_mask(5., 10., x, y))
    assert hatch_mask(.5, 1., x, y).sum() > hatch_mask(.5, 2., x, y).sum()
    assert not hatch_mask([0., -1., np.nan, np.inf], 1., 0., 0.).any()
    assert not hatch_mask(1., 1., [np.inf, np.nan], 0.).any()
    for reference in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError):
            hatch_mask(1., reference, 0., 0.)


def test_hatch_is_independent_of_batching_or_buffer_stride():
    x, y = np.meshgrid(np.arange(-64, 128, dtype=float) + .5,
                       np.arange(-32, 96, dtype=float) + .5)
    magnitude = np.linspace(.01, 3., x.size).reshape(x.shape)
    whole = hatch_mask(magnitude, 1., x, y)
    for stride in (1, 2, 3, 8):
        np.testing.assert_array_equal(
            hatch_mask(magnitude[::stride, ::stride], 1., x[::stride, ::stride], y[::stride, ::stride]),
            whole[::stride, ::stride])
    split = 713
    batched = np.concatenate([
        hatch_mask(magnitude.ravel()[:split], 1., x.ravel()[:split], y.ravel()[:split]),
        hatch_mask(magnitude.ravel()[split:], 1., x.ravel()[split:], y.ravel()[split:])])
    np.testing.assert_array_equal(batched, whole.ravel())
