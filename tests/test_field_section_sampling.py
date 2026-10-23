"""Small CPU-only sampling/coverage tests; no field solve or trajectories."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.field_density_texture import SectionMagnitudeGrid, from_longitudinal, to_longitudinal, section_grid_axes
from temsim.field_section_sampling import longitudinal_field_grid, with_source_support
from temsim.field_vector_scene import FieldVectorScene, FieldVectorSource


def uniform_window(u=(-.1, .1), z=(0., 3.), count=15, angle=0.):
    uu, zz = np.meshgrid(np.linspace(*u, count), np.linspace(*z, count), indexing="ij")
    return from_longitudinal(np.column_stack((uu.ravel(), np.zeros(uu.size), zz.ravel())), angle)


def source(key, centre, half=.0002, radius=.0001, offset=0.):
    bounds = np.array([[offset-radius, -radius, centre-half], [offset+radius, radius, centre+half]])
    native = SimpleNamespace(bounds_m=bounds, known_zero=False, field_map=None,
                             analytic_axial_mask=lambda p: np.ones(len(p), bool),
                             contains=lambda p: np.all((p >= bounds[0]) & (p <= bounds[1]), axis=1),
                             provider=SimpleNamespace(field_at_global_positions_t=lambda p:
                                                      np.tile((.001, 0., .1), (len(p), 1))))
    return FieldVectorSource(key, key, "magnetic", (key,), bounds, "synthetic narrow field", _magnetic=(native,))


@pytest.mark.parametrize("angle", [0., 37., 90.])
def test_whole_column_samples_axis_thin_sources_and_nonzero_production_sum(angle):
    sources = (source("lens", 1.234567), source("deflector", 2.345678, half=5e-6))
    original = uniform_window(angle=angle)
    scene = FieldVectorScene(sources, np.array([[-.2, -.2, 0.], [.2, .2, 3.]]), ())
    old = scene.sample(original)
    assert not np.any(old.b_t[old.magnetic_valid])  # Uniform grid misses both devices.
    points, axes = longitudinal_field_grid(original, sources, angle_deg=angle)
    assert len(points) <= 1600
    local = to_longitudinal(points, angle)
    np.testing.assert_allclose(local[:, 1], 0., atol=1e-16)
    assert 0. in axes[0] and np.any((abs(axes[0]) > 0.) & (abs(axes[0]) < .0001))
    assert all(value in axes[1] for value in (1.234567, 2.345678))
    sample = scene.sample(points)
    assert np.max(np.linalg.norm(sample.b_t[sample.magnetic_valid], axis=1)) > .1
    again, again_axes = longitudinal_field_grid(original, sources, angle_deg=angle)
    np.testing.assert_array_equal(points, again)
    for first, second in zip(axes, again_axes):
        np.testing.assert_array_equal(first, second)


def test_joint_source_keeps_individual_lens_centres_and_budget():
    joint = source("joint", 1.5, half=.5)
    object.__setattr__(joint, "component_keys", ("c1", "c2"))
    many = (joint, *(source(f"lens{i}", .01+i*.029) for i in range(90)))
    points, axes = longitudinal_field_grid(uniform_window(count=40), many,
        component_centres=(("c1", 1.1), ("c2", 1.9)), max_points=1600)
    assert len(points) <= 1600 and len(points) == len(axes[0])*len(axes[1])
    assert 0. in axes[0]
    # Small normal assemblies retain every declared member of a joint source.
    _, joint_axes = longitudinal_field_grid(uniform_window(), (joint,),
        component_centres=(("c1", 1.1), ("c2", 1.9)))
    assert 1.1 in joint_axes[1] and 1.9 in joint_axes[1]


def test_visible_window_is_not_expanded_by_axis_or_offscreen_sources():
    original = uniform_window(u=(.002, .003), z=(1., 1.1), angle=37.)
    points, axes = longitudinal_field_grid(original, (source("outside", 2.),), angle_deg=37., max_points=60)
    assert len(points) <= 60 and 0. not in axes[0]
    local = to_longitudinal(points, 37.)
    np.testing.assert_allclose(local[:, 1], 0., atol=1e-16)
    assert np.all(local[:, 0] >= .002-1e-16) and np.all(local[:, 0] <= .003+1e-16)
    assert np.all(local[:, 2] >= 1.) and np.all(local[:, 2] <= 1.1)


def test_nonuniform_density_axes_preserve_physical_mask_gaps():
    points, axes = longitudinal_field_grid(uniform_window(), (source("lens", 1.234567),), angle_deg=0.)
    assert not np.allclose(np.diff(axes[0]), np.diff(axes[0])[0])
    values = np.column_stack((1.+points[:, 0], np.zeros(len(points)), np.zeros(len(points))))
    grid = SectionMagnitudeGrid(points, values, np.ones(len(points), bool), "UZ", axes)
    query = np.array([[0., 0., 1.234567]])
    magnitude, supported, valid = grid.evaluate(query)
    assert supported[0] and valid[0] and magnitude[0] == pytest.approx(1.)
    keep = points[:, 0] != 0.
    masked = SectionMagnitudeGrid(points[keep], values[keep], np.ones(np.count_nonzero(keep), bool), "UZ", axes)
    magnitude, supported, valid = masked.evaluate(query)
    assert not supported[0] and not valid[0] and np.isnan(magnitude[0])


def test_thin_unknown_source_remains_unavailable_in_whole_column():
    bounds = np.array([[-.001, -.001, 2.22220], [.001, .001, 2.22224]])
    missing = FieldVectorSource("missing", "Missing", "unmodelled", ("missing",),
                                bounds, "unavailable", supported=False, _unknown_b=True)
    points, axes = longitudinal_field_grid(uniform_window(), (missing,))
    centre = np.mean(bounds[:, 2])
    assert centre in axes[1]
    scene = FieldVectorScene((missing,), np.array([[-.2, -.2, 0.], [.2, .2, 3.]]), ())
    sample = scene.sample(points)
    at_centre = points[:, 2] == centre
    assert np.any(at_centre) and not np.any(sample.magnetic_valid[at_centre])
    assert np.isnan(sample.b_t[at_centre]).all()


def test_narrow_near_axis_rows_survive_many_offset_sources():
    narrow = source("narrow", 1.234567, radius=1e-5)
    offset = tuple(source(f"offset{i}", 2., radius=.001, offset=i*.002) for i in range(-30, 31))
    points, axes = longitudinal_field_grid(uniform_window(), (narrow, *offset))
    assert len(points) <= 1600
    assert all(value in axes[0] for value in (-5e-6, 0., 5e-6))


def constant_source(key, centre, half, vector, radius=.003):
    result = source(key, centre, half=half, radius=radius)
    result._magnetic[0].provider.field_at_global_positions_t = lambda points: np.tile(vector, (len(points), 1))
    return result


@pytest.mark.parametrize("angle", [0., 37.5, 90.])
def test_finite_sources_keep_gap_exactly_zero_in_rotated_strength_sections(angle):
    sources = (constant_source("upper", .8, .005, (2., -3., 0.)),
               constant_source("lower", .82, .005, (1., 2., 0.)))
    scene = FieldVectorScene(sources, np.array([[-.1, -.1, 0.], [.1, .1, 3.]]), ())
    points, axes = longitudinal_field_grid(uniform_window(angle=angle), sources, angle_deg=angle)
    sampled = scene.sample(points)
    grid = SectionMagnitudeGrid(points, sampled.b_t, sampled.magnetic_valid, "UZ", axes, angle_deg=angle)
    query = from_longitudinal(np.array([[0., 0., z] for z in (.798, .803, .806, .807, .81, .814, .818)]), angle)
    actual = scene.sample(query)
    magnitude, supported, valid = grid.evaluate(query)
    assert supported.all() and valid.all()
    np.testing.assert_array_equal(magnitude, np.hypot.reduce(actual.b_t, axis=1))
    np.testing.assert_array_equal(magnitude[2:6], np.zeros(4))


def test_gap_retains_other_source_field_and_vector_cancellation():
    upper = constant_source("upper", .8, .005, (2., -3., 0.))
    opposite = constant_source("opposite", .8, .005, (-2., 3., 0.))
    background = constant_source("background", 1.5, 1.5, (0., 0., .25))
    sources = (upper, opposite, background)
    scene = FieldVectorScene(sources, np.array([[-.1, -.1, 0.], [.1, .1, 3.]]), ())
    points, axes = longitudinal_field_grid(uniform_window(), sources)
    sampled = scene.sample(points)
    grid = SectionMagnitudeGrid(points, sampled.b_t, sampled.magnetic_valid, "UZ", axes)
    query = np.array([[0., 0., z] for z in (.798, .803, .807, .81, .818)])
    magnitude, supported, valid = grid.evaluate(query)
    assert supported.all() and valid.all()
    np.testing.assert_array_equal(magnitude, np.full(len(query), .25))
    # Sampling positions must remain comparable when contributions are toggled.
    selected_points, selected_axes = longitudinal_field_grid(
        uniform_window(), sources, interpolation_sources=(background,))
    np.testing.assert_array_equal(points, selected_points)
    assert len(selected_axes.support_planes) < len(axes.support_planes)


@pytest.mark.parametrize("unknown", [False, True])
def test_budget_cannot_remove_barrier_and_infer_zero_across_missed_source(unknown):
    narrow = constant_source("narrow", 1.234567, 1e-6, (1., 0., 0.))
    if unknown:
        narrow = FieldVectorSource("missing", "Missing", "unmodelled", ("missing",),
                                   narrow.bounds_m, "unavailable", supported=False, _unknown_b=True)
    scene = FieldVectorScene((narrow,), np.array([[-.1, -.1, 0.], [.1, .1, 3.]]), ())
    points, axes = longitudinal_field_grid(uniform_window(), (narrow,), max_points=6)
    assert len(points) <= 6 and len(axes.support_planes) == 2
    sampled = scene.sample(points)
    assert sampled.magnetic_valid.all() and not sampled.b_t.any()  # Retained points miss the source.
    grid = SectionMagnitudeGrid(points, sampled.b_t, sampled.magnetic_valid, "UZ", axes)
    magnitude, supported, valid = grid.evaluate([[0., 0., 1.234567]])
    assert not supported[0] and not valid[0] and np.isnan(magnitude[0])


@pytest.mark.parametrize("section,angle", [("UZ", 0.), ("UZ", 37.5), ("UZ", 90.),
                                          ("XY", 0.), ("XZ", 0.), ("YZ", 0.)])
def test_posed_native_axial_boundary_blocks_interpolation_inside_global_bounds(section, angle):
    from temsim.magnetic_field_scene import _Source
    from temsim.physics.lens_field_provider import CoordinateRegistration
    theta = np.deg2rad(25.)
    rotation = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.],
                         [-np.sin(theta), 0., np.cos(theta)]])
    registration = CoordinateRegistration(origin_global_m=(0., 0., .8), rotation_local_to_global=rotation)
    bounds = np.array([[-.02, -.02, .78], [.02, .02, .82]])
    provider = SimpleNamespace(registration=registration, native_field_support_mm=lambda: (-5., 5.),
                               field_at_global_positions_t=lambda p: np.tile((1., 0., 0.), (len(p), 1)))
    native = _Source("posed", provider, bounds, .02)
    posed = FieldVectorSource("posed", "Posed", "magnetic", ("posed",), bounds, "test", _magnetic=(native,))
    scene = FieldVectorScene((posed,), bounds, ())
    def section_points(count0, count1):
        first = np.linspace(-.019, .019, count0)
        second = np.linspace(*((-.019, .019) if section == "XY" else (.781, .819)), count1)
        aa, bb = np.meshgrid(first, second, indexing="ij")
        result = np.zeros((aa.size, 3))
        if section == "XY":
            result[:, 2] = .8
        active = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2), "UZ": (0, 2)}[section]
        result[:, active] = np.column_stack((aa.ravel(), bb.ravel()))
        return from_longitudinal(result, angle) if section == "UZ" else result

    points = section_points(31, 31)
    if section == "UZ":
        points, axes = longitudinal_field_grid(points, (posed,), angle_deg=angle)
    else:
        axes = with_source_support(section_grid_axes(points, section), (posed,))
    sampled = scene.sample(points)
    grid = SectionMagnitudeGrid(points, sampled.b_t, sampled.magnetic_valid, section, axes, angle_deg=angle)
    query = section_points(61, 67)
    actual = scene.sample(query)
    magnitude, supported, valid = grid.evaluate(query)
    assert supported.any() and valid.any()
    np.testing.assert_array_equal(magnitude[valid], np.hypot.reduce(actual.b_t[valid], axis=1))
    assert np.any(valid & (magnitude == 0.)) and np.any(valid & (magnitude == 1.))
