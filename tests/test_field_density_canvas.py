"""Bounded CPU hatching obeys physical geometry and the shared view transform."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QWheelEvent

from temsim.field_density_texture import section_grid_axes
from temsim.gui.field_vector_canvas import FieldVectorCanvas


def section_sample(section="XZ", count=9, strength=.25):
    axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[section]
    xy = np.meshgrid(np.linspace(-.001, .001, count),
                     np.linspace(-.001, .001, count), indexing="ij")
    points = np.zeros((count*count, 3))
    points[:, axes] = np.column_stack([item.ravel() for item in xy])
    values = np.zeros_like(points)
    # Field normal to the section must still contribute its full strength.
    values[:, ({0, 1, 2}-set(axes)).pop()] = strength
    valid = np.ones(len(points), dtype=bool)
    return SimpleNamespace(
        points_m=points, b_t=values, e_v_per_m=values*2, force_n=values*3,
        magnetic_valid=valid.copy(), electric_valid=valid.copy(), force_valid=valid.copy(),
        grid_axes_m=section_grid_axes(points, section))


def canvas_fixture(qtbot, section="XZ"):
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(700, 450)
    sample = section_sample(section)
    canvas.set_sample(sample, np.array([[-.001]*3, [.001]*3]), section)
    canvas.set_reference("B", 1.)
    canvas.set_mode("B")
    canvas.set_render_style("density")
    canvas.set_projection_angle(90. if section == "YZ" else 0.)
    return canvas


def rgba(image):
    return np.asarray(image.constBits()).reshape(image.height(), image.bytesPerLine())[:, :image.width()*4].reshape(image.height(), image.width(), 4)

def widget_pixels(canvas):
    image = canvas.grab().toImage()
    return bytes(image.constBits())


@pytest.mark.parametrize("section,angle", [("XY", 0.), ("XZ", 31.), ("YZ", 64.)])
def test_density_uses_full_norm_and_shared_rotated_physical_coordinates(qtbot, section, angle):
    canvas = canvas_fixture(qtbot, section)
    canvas.set_projection_angle(angle)
    canvas._zoom = np.full(2, 1.8)
    canvas._pan = QPointF(12., -6.)
    xyz = np.array([[.0001, -.0002, .0003]])
    normal = {"XY": 2, "XZ": 1, "YZ": 0}[section]
    xyz[:, normal] = 0.
    projected = canvas._project_positions(xyz)[:, :2]
    recovered = canvas._section_positions(projected, canvas._section_inverse(canvas._plot_rect()))
    np.testing.assert_allclose(recovered, xyz, atol=1e-18)
    text = canvas._density_hover_text(QPointF(*projected[0]))
    assert "|B| ≈ 0.25 T" in text
    image = canvas._density_image(canvas._plot_rect())
    assert np.count_nonzero(rgba(image)[..., 3] == 220) > 100
    assert canvas.references["B"] == 1.


def test_texture_cache_is_reused_for_model_changes_and_rebuilt_for_field_scale(qtbot):
    canvas = canvas_fixture(qtbot)
    plot = canvas._plot_rect()
    first = canvas._density_image(plot)
    sample = canvas.sample
    canvas.set_model_opacity(.15)
    canvas.set_selected_component("lens")
    assert canvas._density_image(plot) is first
    canvas.set_reference("B", .25)
    stronger = canvas._density_image(plot)
    assert stronger is not first
    assert np.count_nonzero(rgba(stronger)[..., 3] == 220) > np.count_nonzero(rgba(first)[..., 3] == 220)
    canvas.set_mode("E")
    assert canvas._density_image(plot) is not stronger
    assert canvas.sample is sample
    canvas.clear()
    assert canvas._density_image_cache is None
    assert canvas._density_grid_cache is None


def test_sample_refinement_does_not_change_constant_strength_texture(qtbot):
    canvas = canvas_fixture(qtbot)
    plot = canvas._plot_rect()
    first = rgba(canvas._density_image(plot)).copy()
    canvas.set_sample(section_sample(count=31), canvas._bounds_m, "XZ")
    refined = rgba(canvas._density_image(plot))
    np.testing.assert_array_equal(first, refined)
    assert canvas.references["B"] == 1.


def test_exact_geometric_mask_clips_hatching_and_unavailable_remains_distinct(qtbot):
    canvas = canvas_fixture(qtbot, "XY")
    # Curved boundary crosses cells whose four corner samples all exist.
    regions = SimpleNamespace(mask=lambda points, key=None: np.linalg.norm(points[:, :2], axis=1) < .0008)
    canvas.set_observation_regions(regions, "cylinder")
    image = canvas._density_image(canvas._plot_rect())
    array = rgba(image)
    assert array[0, 0, 3] == 0
    assert array[image.height()//2, image.width()//2, 3] > 0
    assert canvas._density_hover_text(QPointF(*canvas._project_positions([[.0009, .0009, 0.]])[0, :2])) is None
    sample = section_sample("XY", strength=0.)
    canvas.set_sample(sample, canvas._bounds_m, "XY")
    zero = rgba(canvas._density_image(canvas._plot_rect())).copy()
    assert not np.any(zero[..., 3] == 220)
    sample.magnetic_valid[:] = False
    canvas.set_sample(sample, canvas._bounds_m, "XY")
    unknown = rgba(canvas._density_image(canvas._plot_rect()))
    assert np.any(unknown[..., 3] == 160)
    assert not np.array_equal(unknown, zero)


def test_large_view_has_bounded_texture_and_small_mask_batches(qtbot):
    canvas = canvas_fixture(qtbot)
    canvas.resize(6000, 4000)
    batches = []

    def mask(points, _component=None):
        batches.append(len(points))
        return np.ones(len(points), dtype=bool)

    canvas.set_observation_regions(SimpleNamespace(mask=mask))
    image = canvas._density_image(canvas._plot_rect())
    assert image.width()*image.height() <= 524288
    assert image.sizeInBytes() <= 2*1024*1024
    assert max(batches) <= 8192
    before = len(batches)
    assert canvas._density_image(canvas._plot_rect()) is image
    assert len(batches) == before


def test_density_rejects_edge_on_sections_3d_and_mixed_units(qtbot):
    canvas = canvas_fixture(qtbot)
    canvas.set_projection_angle(90.)
    assert canvas._density_image(canvas._plot_rect()) is None
    assert "edge-on" in canvas._density_notice
    canvas.set_projection_angle(0.)
    canvas.set_mode("overlay")
    assert canvas._density_image(canvas._plot_rect()) is None
    assert "own strength scale" in canvas._density_notice
    canvas.set_sample(canvas.sample, canvas._bounds_m, "3D")
    assert canvas._density_image(canvas._plot_rect()) is None
    assert "2D" in canvas._density_notice


def test_switching_back_to_arrows_restores_existing_render_and_clears_on_new_snapshot(qtbot):
    canvas = canvas_fixture(qtbot)
    canvas.set_render_style("arrows")
    before = widget_pixels(canvas)
    canvas.set_render_style("density")
    hatched = widget_pixels(canvas)
    assert hatched != before
    assert len(canvas._visible_sample_indices) == 0
    canvas.set_render_style("arrows")
    assert widget_pixels(canvas) == before
    assert len(canvas._visible_sample_indices) > 0
    canvas.set_sample(section_sample(strength=.5), canvas._bounds_m, "XZ")
    assert canvas._density_image_cache is None


def test_small_off_axis_section_preserves_exact_normal_through_detail_and_empty_view(qtbot):
    canvas = canvas_fixture(qtbot, "XZ")
    sample = section_sample()
    offset = 1.e-10
    sample.points_m[:, 1] = offset
    bounds = canvas._bounds_m.copy()
    bounds[:, 1] += offset
    assert bounds[:, 1].mean() != offset  # Averages lose bits near the axis.
    canvas.set_sample(sample, bounds, "XZ")
    assert canvas._section_centre()[1] == offset
    points, _key = canvas.viewport_sample_grid()
    np.testing.assert_array_equal(points[:, 1], offset)
    assert "|B| ≈ 0.25 T" in canvas._density_hover_text(canvas._plot_rect().center())
    image = canvas._density_image(canvas._plot_rect())
    assert np.any(rgba(image)[..., 3] == 220)
    empty = SimpleNamespace(**{key: value[:0] for key, value in vars(sample).items()
                               if isinstance(value, np.ndarray)}, grid_axes_m=None)
    canvas.set_sample(empty, bounds, "XZ")
    assert canvas._section_centre()[1] == offset


def test_region_culling_matches_exact_union_including_tilt_and_boundary_roundoff(qtbot, monkeypatch):
    from temsim.field_display_regions import FieldDisplayRegion, FieldDisplayRegions
    angle = np.deg2rad(35.)
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])

    def region(key, centre, transform, shape):
        half = np.array([.001, .0008, .0006])
        extent = np.abs(transform) @ half
        return FieldDisplayRegion(key, key, shape, np.array([centre-extent, centre+extent]),
                                  centre, transform.T, half)

    cylinder = region("cylinder", np.zeros(3), np.eye(3), "cylinder")
    tilted = region("tilted", np.array([.0015, 0., .0002]), rotation, "box")
    distant = region("distant", np.array([0., 0., 1.]), np.eye(3), "box")
    regions = FieldDisplayRegions((cylinder, tilted, distant))
    xyz = np.meshgrid(np.linspace(-.002, .003, 12), np.linspace(-.001, .001, 10),
                      np.linspace(-.001, .001, 7), indexing="ij")
    points = np.column_stack([axis.ravel() for axis in xyz])
    points = np.vstack((points, [.001*(1.+1e-15), 0., 0.]))
    expected = regions.mask(points)
    queries = []
    contains = FieldDisplayRegion.contains

    def observed(region, positions):
        queries.append((region.component_key, len(positions)))
        return contains(region, positions)

    monkeypatch.setattr(FieldDisplayRegion, "contains", observed)
    canvas = canvas_fixture(qtbot)
    canvas.set_observation_regions(regions)
    np.testing.assert_array_equal(canvas._observation_mask(points), expected)
    assert {key for key, _count in queries} == {"cylinder", "tilted"}
    assert sum(count for _key, count in queries) < len(points)
    cached_bounds = canvas._observation_bounds_cache
    np.testing.assert_array_equal(canvas._observation_mask(points[::3]), expected[::3])
    assert canvas._observation_bounds_cache is cached_bounds
    # Replacing a snapshot or selecting a different envelope must never reuse
    # the old union, even if the new index contains the same region objects.
    subset = FieldDisplayRegions((cylinder, tilted, distant),
                                 component_region_keys={"tilted": ("tilted",)})
    canvas.set_observation_regions(subset)
    np.testing.assert_array_equal(canvas._observation_mask(points), expected)
    canvas.set_observation_regions(subset, "tilted")
    np.testing.assert_array_equal(canvas._observation_mask(points), tilted.contains(points))
    canvas.set_observation_regions(subset, "absent")
    np.testing.assert_array_equal(canvas._observation_mask(points), False)


def longitudinal_canvas(qtbot, angle):
    from temsim.field_density_texture import from_longitudinal
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(800, 450)
    uu, zz = np.meshgrid(np.linspace(-.001, .001, 15), np.linspace(.1, .12, 15), indexing="ij")
    points = from_longitudinal(np.column_stack((uu.ravel(), np.zeros(uu.size), zz.ravel())), angle)
    # An anisotropic physical field: turning the section really changes values.
    values = np.column_stack((points[:, 0]*2, points[:, 1], np.zeros(len(points))))
    valid = np.ones(len(points), dtype=bool)
    sample = SimpleNamespace(points_m=points, b_t=values, e_v_per_m=values,
                             force_n=values, magnetic_valid=valid, electric_valid=valid,
                             force_valid=valid, section_angle_deg=angle,
                             grid_axes_m=section_grid_axes(points, "UZ", angle_deg=angle))
    canvas.set_projection_angle(angle)
    canvas.set_sample(sample, np.array([[-.001, -.001, .1], [.001, .001, .12]]), "UZ")
    canvas.set_mode("B")
    canvas.set_reference("B", .002)
    canvas.set_render_style("density")
    return canvas


@pytest.mark.parametrize("angle", [0., 37., 90., 180., 270.])
def test_longitudinal_section_keeps_z_horizontal_and_is_invertible_at_every_angle(qtbot, angle):
    from temsim.field_density_texture import from_longitudinal, to_longitudinal
    canvas = longitudinal_canvas(qtbot, angle)
    local = np.array([[.0005, 0., .105], [.0005, 0., .115], [-.0005, 0., .105]])
    world = from_longitudinal(local, angle)
    screen = canvas._project_positions(world)[:, :2]
    assert screen[1, 0] > screen[0, 0]
    assert screen[1, 1] == pytest.approx(screen[0, 1])
    assert screen[2, 0] == pytest.approx(screen[0, 0])
    assert screen[2, 1] > screen[0, 1]
    inverse = canvas._section_inverse(canvas._plot_rect())
    np.testing.assert_allclose(canvas._section_positions(screen, inverse), world, atol=1e-17)
    points, key = canvas.viewport_sample_grid()
    assert key[:2] == ("UZ", angle)
    rotated = to_longitudinal(points, angle)
    np.testing.assert_allclose(rotated[:, 1], 0., atol=1e-18)
    assert len(points) <= 1600
    image = canvas._density_image(canvas._plot_rect())
    assert np.any(rgba(image)[..., 3] == 220)
    assert not canvas._density_notice
    magnitude = np.linalg.norm([world[0, 0]*2, world[0, 1], 0.])
    assert f"|B| ≈ {magnitude:.6g} T" in canvas._density_hover_text(QPointF(*screen[0]))


def test_longitudinal_physical_cut_and_component_bounds_rotate_on_the_same_plane(qtbot):
    from test_field_vector_mesh_canvas import tube_mesh
    from temsim.field_density_texture import to_longitudinal
    canvas = longitudinal_canvas(qtbot, 37.)
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh,))
    cuts = canvas._physical_sections()
    assert len(cuts) == 1
    local = to_longitudinal(cuts[0][1].reshape(-1, 3), 37.)
    np.testing.assert_allclose(local[:, 1], 0., atol=1e-18)
    assert np.abs(local[:, 0]).min() > .00039  # The true bore stays open.
    bounds = np.array([[-.001, -.002, -.003], [.001, .002, .003]])
    outline = canvas._longitudinal_box_cut(bounds)
    assert len(outline) >= 4
    np.testing.assert_allclose(to_longitudinal(outline, 37.)[:, 1], 0., atol=1e-18)
    assert np.all(outline >= bounds[0]-1e-18) and np.all(outline <= bounds[1]+1e-18)
    assert len(canvas._longitudinal_box_cut(bounds + [1., 0., 0.])) == 0


@pytest.mark.parametrize("axis", ["horizontal", "vertical"])
def test_axis_zoom_rebuilds_hatching_with_same_physical_field_and_inverse(qtbot, axis):
    from temsim.field_density_texture import from_longitudinal

    canvas = longitudinal_canvas(qtbot, 37.)
    sample = canvas.sample
    original = {key: value.copy() for key, value in vars(sample).items() if isinstance(value, np.ndarray)}
    refs = canvas.references.copy()
    plot = canvas._plot_rect()
    first = canvas._density_image(plot)
    grid = canvas._magnitude_grid()
    position = (QPointF(plot.center().x(), plot.bottom()+8.) if axis == "horizontal"
                else QPointF(plot.left()-8., plot.center().y()))
    event = QWheelEvent(position, position, QPointF().toPoint(), QPointF(0., 120.).toPoint(),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    canvas.wheelEvent(event)
    assert canvas._density_image_cache is None
    refreshed = canvas._density_image(plot)
    assert refreshed is not first and canvas._density_image(plot) is refreshed
    assert canvas._magnitude_grid() is grid
    assert np.any(rgba(refreshed)[..., 3] == 220)

    world = from_longitudinal(np.array([[.00025, 0., .11]]), 37.)
    projected = canvas._project_positions(world)[:, :2]
    recovered = canvas._section_positions(projected, canvas._section_inverse(plot))
    np.testing.assert_allclose(recovered, world, atol=1e-17)
    magnitude = np.linalg.norm([world[0, 0]*2, world[0, 1], 0.])
    assert f"|B| ≈ {magnitude:.6g} T" in canvas._density_hover_text(QPointF(*projected[0]))
    assert canvas.sample is sample and canvas.references == refs
    for key, value in original.items():
        np.testing.assert_array_equal(getattr(sample, key), value)


def test_longitudinal_left_and_middle_drag_pan_without_requesting_rotation(qtbot):
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QMouseEvent
    canvas = longitudinal_canvas(qtbot, 0.)
    angles = []
    canvas.projection_angle_requested.connect(angles.append)
    start = canvas._plot_rect().center()
    end = start+QPointF(40., 10.)
    for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
        before = QPointF(canvas._pan)
        for kind, point in ((QEvent.Type.MouseButtonPress, start), (QEvent.Type.MouseMove, end),
                            (QEvent.Type.MouseButtonRelease, end)):
            event = QMouseEvent(kind, point, point, button, button, Qt.KeyboardModifier.NoModifier)
            if kind == QEvent.Type.MouseButtonPress:
                canvas.mousePressEvent(event)
            elif kind == QEvent.Type.MouseMove:
                canvas.mouseMoveEvent(event)
            else:
                canvas.mouseReleaseEvent(event)
        assert angles == []
        assert canvas._pan == before+QPointF(40., 10.)


def test_navigation_preview_is_bounded_and_restores_full_texture_without_resampling(qtbot):
    canvas = canvas_fixture(qtbot)
    canvas.resize(1600, 1000)
    plot = canvas._plot_rect()
    sample, refs = canvas.sample, canvas.references.copy()
    full = canvas._density_image(plot)
    grid = canvas._magnitude_grid()
    canvas.set_interaction_active(True)
    preview = canvas._density_image(plot)
    assert preview.width()*preview.height() <= 32768
    assert preview.sizeInBytes() <= 128*1024
    assert preview.width() < full.width()
    assert canvas._density_image(plot) is preview
    canvas._interaction_timer.start(1)
    qtbot.waitUntil(lambda: not canvas._interaction_active, timeout=1000)
    restored = canvas._density_image(plot)
    np.testing.assert_array_equal(rgba(restored), rgba(full))
    assert canvas._magnitude_grid() is grid
    assert canvas.sample is sample and canvas.references == refs


def test_preview_preserves_unavailable_mask_and_current_inverse_transform(qtbot):
    canvas = canvas_fixture(qtbot, "XY")
    canvas.resize(1600, 1000)
    canvas.sample.magnetic_valid[:] = False
    canvas.set_sample(canvas.sample, canvas._bounds_m, "XY")
    canvas.set_interaction_active(True)
    canvas._pan = QPointF(21., -13.)
    canvas._zoom = np.array([1.7, .8])
    batches = []

    def mask(points, _component=None):
        batches.append(points.copy())
        return np.linalg.norm(points[:, :2], axis=1) < .00075

    canvas.set_observation_regions(SimpleNamespace(mask=mask))
    result = rgba(canvas._density_image(canvas._plot_rect()))
    assert np.any(result[..., 3] == 160)
    assert not np.any(result[..., 3] == 220)
    assert max(map(len, batches)) <= 8192
    assert canvas._density_hover_text(QPointF(*canvas._project_positions([[0, 0, 0]])[0, :2])).endswith(
        "B: unavailable / outside model validity (not zero).")
