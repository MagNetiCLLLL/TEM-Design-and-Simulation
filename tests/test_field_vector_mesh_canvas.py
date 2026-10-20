"""Material/field registration and real section checks, without desktop use."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QWheelEvent

from temsim.gui.field_vector_canvas import FieldVectorCanvas
from temsim.part_model_3d import TriangleMesh, revolve_section


def sample_at(points):
    points = np.asarray(points, dtype=float)
    values = np.tile((.2, .4, .8), (len(points), 1))
    valid = np.ones(len(points), dtype=bool)
    return SimpleNamespace(points_m=points, b_t=values, e_v_per_m=values * 2,
                           force_n=values * 3, magnetic_valid=valid,
                           electric_valid=valid, force_valid=valid)


def tube_mesh():
    # The existing mesh constructor is geometrically unit-neutral. Values
    # here are metres, matching the canvas's explicitly documented boundary.
    return revolve_section(((-.002, .0004), (-.002, .0008),
                            (.002, .0008), (.002, .0004)), key="tube",
                           angular_segments=64, material_class="copper_winding")


def canvas_with_sample(qtbot, section="3D"):
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(760, 540)
    bounds = np.array([[-.0002, -.0002, -.0003], [.0002, .0002, .0003]])
    canvas.set_sample(sample_at([[0, 0, 0], [.0001, .0001, .0002]]), bounds, section)
    canvas.set_model_visible(True)
    canvas.show()
    return canvas


def pixels(canvas):
    image = canvas.grab().toImage()
    return bytes(image.constBits())


def test_mesh_and_field_share_projection_under_fit_rotation_pan_and_zoom(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    points = mesh.vertices[::17].copy()
    canvas.set_sample(sample_at(points), canvas._bounds_m, "3D")
    canvas.set_physical_meshes((mesh,))
    canvas.fit_physical_models()
    for angle, zoom, pan in ((0., 1., QPointF()), (37., 2.3, QPointF(37, -23))):
        canvas.set_projection_angle(angle)
        canvas._zoom, canvas._pan = np.full(2, zoom), pan
        canvas.grab()
        actual = canvas._project_positions(mesh.vertices[::17])[:, :2]
        np.testing.assert_allclose(actual, canvas._screen_points, atol=1e-10)
        centre, span, basis, size, origin = canvas._projection()
        expected = (((points - centre) / span) @ basis.T)[:, :2] * size + origin
        np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_fit_models_only_changes_display_bounds_and_reset_restores_sample(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    sample = canvas.sample
    points = sample.points_m.copy()
    bounds = canvas._bounds_m.copy()
    refs = canvas.references.copy()
    canvas.set_physical_meshes((mesh,))
    np.testing.assert_array_equal(canvas._view_bounds_m, bounds)
    canvas.fit_physical_models()
    assert np.all(canvas._view_bounds_m[0] <= mesh.vertices.min(axis=0))
    assert np.all(canvas._view_bounds_m[1] >= mesh.vertices.max(axis=0))
    assert canvas.sample is sample and canvas.references == refs
    np.testing.assert_array_equal(canvas._bounds_m, bounds)
    np.testing.assert_array_equal(sample.points_m, points)
    plot = canvas._plot_rect()
    screen = canvas._project_positions(mesh.vertices)
    assert all(plot.contains(QPointF(*point)) for point in screen[:, :2])
    canvas.reset_view()
    np.testing.assert_array_equal(canvas._view_bounds_m, bounds)
    np.testing.assert_array_equal(canvas._zoom, np.ones(2))
    assert not canvas._fit_models


@pytest.mark.parametrize("section", ["3D", "XY", "XZ", "YZ"])
def test_zero_opacity_is_identical_to_field_only_render(qtbot, section):
    canvas = canvas_with_sample(qtbot, section)
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    canvas.set_model_visible(False)
    field_only = pixels(canvas)
    canvas.set_model_visible(True)
    canvas.set_model_opacity(0.)
    assert not canvas.model_overlay_visible
    assert pixels(canvas) == field_only
    canvas.set_model_opacity(.6)
    assert pixels(canvas) != field_only


@pytest.mark.parametrize("section, normal", [("XY", 2), ("XZ", 1), ("YZ", 0)])
def test_sections_are_true_intersections_and_preserve_the_tube_bore(qtbot, section, normal):
    canvas = canvas_with_sample(qtbot, section)
    mesh = tube_mesh()
    bounds = canvas._bounds_m.copy()
    bounds[:, normal] = 0.
    canvas.set_sample(canvas.sample, bounds, section)
    canvas.set_physical_meshes((mesh,))
    cuts = canvas._physical_sections()
    assert len(cuts) == 1
    segments = cuts[0][1]
    np.testing.assert_array_equal(segments[..., normal], 0.)
    if section == "XY":
        # A slice through the hollow tube contains two separate rings;
        # no false face spans its empty centre.
        radius = np.linalg.norm(segments[..., :2], axis=-1)
        assert radius.min() >= .0004 * np.cos(np.pi / 64) - 1e-12
        assert radius.max() <= .0008 + 1e-12
        mid_radius = np.linalg.norm(segments[..., :2].mean(axis=1), axis=-1)
        assert mid_radius.min() > .00039
    else:
        transverse = 0 if section == "XZ" else 1
        assert np.abs(segments[..., transverse]).min() >= .0004 - 1e-12
        assert not np.any(segments[:, 0, transverse] * segments[:, 1, transverse] < 0)
    # An offset plane missing the solid must have no projected silhouette.
    bounds[:, normal] = .003
    canvas.set_sample(canvas.sample, bounds, section)
    canvas.set_physical_meshes((mesh,))
    assert canvas._physical_sections() == ()


def test_section_follows_sample_plane_even_when_fit_changes_display_centre(qtbot):
    canvas = canvas_with_sample(qtbot, "XY")
    bounds = canvas._bounds_m.copy()
    bounds[:, 2] = .0011
    canvas.set_sample(canvas.sample, bounds, "XY")
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    assert np.mean(canvas._view_bounds_m[:, 2]) != .0011
    segments = canvas._physical_sections()[0][1]
    np.testing.assert_array_equal(segments[..., 2], .0011)


def test_replacement_and_clear_cannot_keep_obsolete_geometry(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh,))
    canvas.fit_physical_models()
    canvas.set_sample(canvas.sample, canvas._bounds_m, "3D")
    assert not canvas._physical_meshes and not canvas.model_overlay_visible
    canvas.set_physical_meshes((mesh,))
    assert canvas._fit_models and canvas._view_bounds_m[1, 2] == .002
    canvas.clear("Stale inputs")
    assert canvas.sample is None and not canvas._physical_meshes
    assert canvas._section_cache is None


@pytest.mark.parametrize("opacity", [-.1, 1.1, float("nan"), float("inf")])
def test_opacity_rejects_invalid_values(qtbot, opacity):
    canvas = canvas_with_sample(qtbot)
    with pytest.raises(ValueError, match="opacity"):
        canvas.set_model_opacity(opacity)


def test_invalid_face_indices_rejected_and_nonmaterial_guides_not_shown(qtbot):
    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    with pytest.raises(ValueError, match="outside"):
        canvas.set_physical_meshes((replace(mesh, faces=np.array([[0, 1, len(mesh.vertices)]])),))
    canvas.set_physical_meshes((replace(mesh, wireframe=True),))
    assert not canvas._physical_meshes


def test_opaque_triangles_are_sorted_back_to_front_across_materials(qtbot):
    canvas = canvas_with_sample(qtbot)
    canvas.set_sample(sample_at([[.0002, .0002, .0003]]), canvas._bounds_m, "3D")
    canvas.set_projection_angle(0.)
    # The Ray-compatible camera has depth +Y, horizontal Z and vertical X. The near blue and far
    # red triangles cover the same screen location; submit near first.
    xz = np.array([[-.0001, -.0002], [.0001, -.0002], [0., .0002]])
    def triangle(y, colour):
        vertices = np.column_stack((xz[:, 0], np.full(3, y), xz[:, 1]))
        return TriangleMesh(vertices, np.array([[0, 1, 2]]), str(y), color=colour)
    canvas.set_physical_meshes((triangle(.0001, (0., 0., 1., 1.)),
                                triangle(-.0001, (1., 0., 0., 1.))))
    canvas.set_model_opacity(1.)
    image = canvas.grab().toImage()
    centre = canvas._project_positions(np.array([[0., 0., 0.]]))[0]
    colour = image.pixelColor(round(centre[0]), round(centre[1]))
    assert colour.blue() > 200 and colour.red() < 30
    assert canvas.pick_component_at(QPointF(*centre[:2])) == "0.0001"


@pytest.mark.parametrize("angle", [0., 90., 37., 217.])
@pytest.mark.parametrize("section", ["3D", "XZ", "YZ"])
def test_longitudinal_projection_matches_ray_coordinates_without_distorting_transverse_angle(qtbot, angle, section):
    canvas = canvas_with_sample(qtbot, section)
    bounds = np.array([[-.001, -.002, 0.], [.001, .002, .01]])
    points = np.array([[0., 0., .004], [.0002, .0003, .006]])
    canvas.set_sample(sample_at(points), bounds, section)
    canvas.set_projection_angle(angle)
    projected = canvas._project_positions(points)
    cosine, sine = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    _centre, span, basis, size, _origin = canvas._projection()
    np.testing.assert_allclose(np.linalg.det(basis), 1.)
    np.testing.assert_allclose(projected[1, :2]-projected[0, :2],
        [.002/span[2]*size[0], (.0002*cosine+.0003*sine)/span[0]*size[1]])
    assert projected[1, 0] > projected[0, 0]  # +Z is always right.
    values = np.array([[.0002, .0003, .002]])
    arrows = canvas._display_vectors(values, np.array([True]), basis, span, size, 10., 1.)
    delta = projected[1, :2]-projected[0, :2]
    np.testing.assert_allclose(arrows[0]/np.linalg.norm(arrows[0]), delta/np.linalg.norm(delta))
    assert "Ray projection" in canvas.projection_caption
    if section != "3D":
        assert f"{section} section (plane unchanged)" in canvas.projection_caption


def _mouse(canvas, kind, position, button, buttons):
    event = QMouseEvent(kind, position, QPointF(canvas.mapToGlobal(position.toPoint())),
                        button, buttons, Qt.KeyboardModifier.NoModifier)
    if kind == QEvent.Type.MouseButtonPress:
        canvas.mousePressEvent(event)
    elif kind == QEvent.Type.MouseButtonRelease:
        canvas.mouseReleaseEvent(event)
    else:
        canvas.mouseMoveEvent(event)


def _drag(canvas, button, start, end):
    _mouse(canvas, QEvent.Type.MouseButtonPress, start, button, button)
    _mouse(canvas, QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, button)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, end, button, Qt.MouseButton.NoButton)


def test_left_and_middle_drag_pan_without_changing_shared_angle_or_physical_data(qtbot):
    canvas = canvas_with_sample(qtbot)
    sample = canvas.sample
    arrays = {name: getattr(sample, name).copy() for name in vars(sample)}
    mesh = tube_mesh()
    vertices = mesh.vertices.copy()
    canvas.set_physical_meshes((mesh,))
    meshes = canvas._physical_meshes
    emitted, selected, contexts = [], [], []
    canvas.projection_angle_requested.connect(emitted.append)
    canvas.component_selected.connect(selected.append)
    canvas.component_context_requested.connect(lambda *args: contexts.append(args))
    canvas.set_projection_angle(20.)
    assert emitted == []  # Programmatic synchronization cannot loop.
    centre = canvas._plot_rect().center()
    _drag(canvas, Qt.MouseButton.LeftButton, centre, centre+QPointF(40., 20.))
    assert canvas.projection_angle_deg == 20. and emitted == []
    _drag(canvas, Qt.MouseButton.MiddleButton, centre, centre+QPointF(20., 10.))
    assert canvas._pan == QPointF(60., 30.)
    assert not selected and not contexts
    canvas.reset_view()
    assert canvas.projection_angle_deg == 20. and emitted == []
    assert canvas.sample is sample and canvas._physical_meshes is meshes
    for name, before in arrays.items():
        np.testing.assert_array_equal(getattr(sample, name), before)
    np.testing.assert_array_equal(mesh.vertices, vertices)


@pytest.mark.parametrize("delta", [QPointF(20., 0.), QPointF(0., 15.), QPointF(-20., 15.)])
def test_right_drag_matches_ray_axis_scales_and_keeps_press_coordinate_fixed(qtbot, delta):
    canvas = canvas_with_sample(qtbot)
    canvas._zoom = np.array([1.3, .7])
    canvas._pan = QPointF(12., -9.)
    canvas.set_projection_angle(37.)
    point = np.array([[.00004, .00006, .00007]])
    anchor = QPointF(*canvas._project_positions(point)[0, :2])
    zoom = canvas._zoom.copy()
    emitted = []
    canvas.projection_angle_requested.connect(emitted.append)
    _drag(canvas, Qt.MouseButton.RightButton, anchor, anchor+delta)
    np.testing.assert_allclose(canvas._zoom, zoom*np.power(1.02, [delta.x(), -delta.y()]))
    np.testing.assert_allclose(canvas._project_positions(point)[0, :2], [anchor.x(), anchor.y()])
    assert canvas.projection_angle_deg == 37. and emitted == []


def test_same_snapshot_detail_retains_cached_sections_but_replacement_clears_them(qtbot):
    canvas = canvas_with_sample(qtbot, "UZ")
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh,))
    cuts = canvas._physical_sections()
    canvas.set_sample(sample_at([[0, 0, 0]]), canvas._bounds_m, "UZ", preserve_geometry=True)
    canvas.set_physical_meshes((mesh,))
    assert canvas._physical_sections() is cuts
    canvas.set_projection_angle(17.)
    assert canvas._physical_sections() is not cuts
    canvas.set_sample(canvas.sample, canvas._bounds_m, "UZ")
    assert not canvas._physical_meshes and canvas._section_cache is None


@pytest.mark.parametrize("section", ["3D", "UZ"])
def test_model_camera_geometry_is_reused_for_pan_and_independent_axis_zoom(qtbot, section):
    canvas = canvas_with_sample(qtbot, section)
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    before = _model_layer_image(canvas)
    cache = canvas._model_triangle_cache[False] if section == "3D" else canvas._model_path_cache
    canvas._zoom *= [1.4, .7]
    canvas._pan += QPointF(13., -7.)
    canvas._view_updated()
    after = _model_layer_image(canvas)
    assert (canvas._model_triangle_cache[False] if section == "3D" else canvas._model_path_cache) is cache
    assert bytes(before.constBits()) != bytes(after.constBits())
    canvas.set_projection_angle(32.)
    _model_layer_image(canvas)
    assert (canvas._model_triangle_cache[False] if section == "3D" else canvas._model_path_cache) is not cache


def test_pending_section_preserves_accepted_frame_and_disables_inspection(qtbot):
    canvas = canvas_with_sample(qtbot, "UZ")
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.set_projection_angle(23.)
    sample = canvas.sample
    before = canvas._project_positions(sample.points_m).copy()
    canvas.set_pending_section_angle(61.)
    assert canvas._pending_section_angle == 61.
    assert canvas.projection_angle_deg == 23. and "23°" in canvas.projection_caption
    np.testing.assert_array_equal(canvas._project_positions(sample.points_m), before)
    assert canvas.pick_component_at(canvas._plot_rect().center()) is None
    assert canvas._density_hover_text(canvas._plot_rect().center()) is None
    canvas.grab()  # Paint the conspicuous accepted/target-angle notice offscreen.
    assert canvas.sample is sample
    canvas.set_pending_section_angle(None)
    assert canvas._pending_section_angle is None
    canvas.set_pending_section_angle(70.)
    canvas.clear()
    assert canvas._pending_section_angle is None


def test_projection_rejects_nonfinite_angles_without_changing_view(qtbot):
    canvas = canvas_with_sample(qtbot)
    canvas.set_projection_angle(37.)
    for angle in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite"):
            canvas.set_projection_angle(angle)
        assert canvas.projection_angle_deg == 37.
    canvas.set_projection_angle(397.)
    assert canvas.projection_angle_deg == 37.


def test_selection_highlights_actual_keys_and_caps_opacity_without_changing_geometry(qtbot):
    canvas = canvas_with_sample(qtbot)
    first = tube_mesh()
    second = replace(first, key="child")
    other = replace(first, key="unrelated")
    canvas.set_physical_meshes((first, second, other))
    canvas.set_model_opacity(1.)
    canvas.set_selected_components(("tube", "child"))
    for mesh in (first, second):
        assert canvas._model_colour(mesh)[3] == .35
        assert not np.allclose(canvas._model_colour(mesh)[:3], mesh.color[:3])
    assert canvas._model_colour(other)[3] == 1.
    canvas.set_selected_component("unrelated")
    assert canvas._model_colour(first)[3] == 1.
    assert canvas._model_colour(other)[3] == .35
    canvas.set_model_opacity(0.)
    assert all(canvas._model_colour(mesh)[3] == 0. for mesh in (first, second, other))
    assert not canvas.model_overlay_visible


def test_right_click_picks_real_surface_and_emits_global_context_without_pan(qtbot):
    canvas = canvas_with_sample(qtbot)
    vertices = np.array([[-.0001, 0., -.0002], [.0001, 0., -.0002], [0., 0., .0002]])
    mesh = TriangleMesh(vertices, np.array([[0, 1, 2]]), "real_coil")
    canvas.set_physical_meshes((mesh,))
    centre = QPointF(*canvas._project_positions(np.array([[0., 0., 0.]]))[0, :2])
    selected, contexts = [], []
    canvas.component_selected.connect(selected.append)
    canvas.component_context_requested.connect(lambda *args: contexts.append(args))
    _mouse(canvas, QEvent.Type.MouseButtonPress, centre, Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)
    _mouse(canvas, QEvent.Type.MouseButtonRelease, centre, Qt.MouseButton.RightButton, Qt.MouseButton.NoButton)
    assert selected == ["real_coil"]
    assert contexts == [("real_coil", canvas.mapToGlobal(centre.toPoint()))]
    assert canvas._pan == QPointF() and canvas.projection_angle_deg == 0.
    empty = canvas._plot_rect().topLeft()+QPointF(5., 5.)
    assert canvas.pick_component_at(empty) is None
    canvas.set_model_visible(False)
    assert canvas.pick_component_at(centre) is None


def test_section_picking_uses_visible_boundary_not_envelope_or_hole(qtbot):
    canvas = canvas_with_sample(qtbot, "XY")
    canvas.set_physical_meshes((tube_mesh(),))
    canvas.fit_physical_models()
    centre = QPointF(*canvas._project_positions(np.array([[0., 0., 0.]]))[0, :2])
    edge = QPointF(*canvas._project_positions(np.array([[.0008, 0., 0.]]))[0, :2])
    assert canvas.pick_component_at(centre) is None
    assert canvas.pick_component_at(edge) == "tube"


def _model_layer_image(canvas):
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    try:
        canvas._draw_physical_models(painter, canvas._plot_rect())
    finally:
        painter.end()
    return image


@pytest.mark.parametrize("opacity", [.12, .35, 1.])
def test_selected_assembly_opacity_is_applied_once_across_overlapping_faces_and_children(qtbot, opacity):
    canvas = canvas_with_sample(qtbot)
    xz = np.array([[-.0001, -.0002], [.0001, -.0002], [0., .0002]])
    meshes = tuple(TriangleMesh(
        np.column_stack((xz[:, 0], np.full(3, index*.00001), xz[:, 1])),
        np.tile(np.array([[0, 1, 2]]), (4, 1)), f"child_{index}", color=(1., .5, .1, 1.))
        for index in range(6))
    canvas.set_model_opacity(opacity)
    canvas.set_selected_components(mesh.key for mesh in meshes)
    centre = canvas._project_positions(np.array([[0., 0., 0.]]))[0]
    alphas = []
    for group in (meshes[:1], meshes):
        canvas.set_physical_meshes(group)
        image = _model_layer_image(canvas)
        alphas.append(image.pixelColor(round(centre[0]), round(centre[1])).alphaF())
    # QPainter's 8-bit global-opacity composition can round down twice.
    assert 0 <= min(opacity, .35)-alphas[0] <= 2/255
    assert alphas[0] == alphas[1]
    assert canvas.pick_component_at(QPointF(*centre[:2])) == "child_5"
    # Global zero still hides the layer and removes it from picking.
    canvas.set_model_opacity(0.)
    assert _model_layer_image(canvas).pixelColor(round(centre[0]), round(centre[1])).alpha() == 0
    assert canvas.pick_component_at(QPointF(*centre[:2])) is None


def test_selected_children_share_one_viewport_buffer_and_leave_unselected_opacity_unchanged(qtbot, monkeypatch):
    import temsim.gui.field_vector_canvas as module

    canvas = canvas_with_sample(qtbot)
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh, replace(mesh, key="child")))
    canvas.set_selected_components(("tube", "child"))
    canvas.fit_physical_models()
    allocations = []
    real_image = module.QImage

    def tracked_image(width, height, image_format):
        allocations.append((width, height))
        return real_image(width, height, image_format)

    tracked_image.Format = real_image.Format
    monkeypatch.setattr(module, "QImage", tracked_image)
    _model_layer_image(canvas)
    plot = canvas._plot_rect().toAlignedRect().intersected(canvas.rect())
    ratio = canvas.devicePixelRatioF()
    assert allocations == [(int(np.ceil(plot.width()*ratio)), int(np.ceil(plot.height()*ratio)))]
    # Ordinary unselected surfaces continue to draw without an extra layer.
    canvas.set_selected_component(None)
    allocations.clear()
    _model_layer_image(canvas)
    assert allocations == []
    assert canvas._model_colour(mesh)[3] == canvas._model_opacity


def test_selected_section_boundaries_do_not_accumulate_opacity(qtbot):
    canvas = canvas_with_sample(qtbot, "XY")
    mesh = tube_mesh()
    meshes = tuple(replace(mesh, key=f"child_{index}") for index in range(6))
    canvas.set_physical_meshes(meshes)
    canvas.set_selected_components(item.key for item in meshes)
    canvas.set_model_opacity(1.)
    canvas.fit_physical_models()
    image = _model_layer_image(canvas)
    alpha = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4)[..., 3]
    assert 0 < alpha.max() <= 90
    centre = QPointF(*canvas._project_positions(np.array([[0., 0., 0.]]))[0, :2])
    assert canvas.pick_component_at(centre) is None  # Hollow bore remains empty.


def _wheel_at(canvas, position, delta=120):
    event = QWheelEvent(position, QPointF(canvas.mapToGlobal(position.toPoint())),
                        QPointF().toPoint(), QPointF(0., delta).toPoint(),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        Qt.ScrollPhase.ScrollUpdate, False)
    canvas.wheelEvent(event)


def _axis_wheel_position(canvas, axis):
    plot = canvas._plot_rect()
    if axis == 0:
        return QPointF(plot.left() + .7 * plot.width(), plot.bottom() + 8.)
    return QPointF(plot.left() - 8., plot.top() + .3 * plot.height())


def _visible_zu_corners(canvas):
    plot = canvas._plot_rect()
    screen = np.array([[plot.left(), plot.top()], [plot.right(), plot.bottom()]])
    positions = canvas._section_positions(screen, canvas._section_inverse(plot))
    theta = np.deg2rad(canvas.projection_angle_deg)
    return np.column_stack((positions[:, 2],
                            positions[:, 0] * np.cos(theta) + positions[:, 1] * np.sin(theta)))


@pytest.mark.parametrize("axis", [0, 1], ids=["horizontal", "vertical"])
@pytest.mark.parametrize("angle", [0., 37.])
@pytest.mark.parametrize("on_line", [False, True], ids=["tick-band", "axis-line"])
def test_axis_wheel_matches_ray_scale_and_preserves_other_axis_and_anchor(qtbot, axis, angle, on_line):
    canvas = canvas_with_sample(qtbot, "UZ")
    canvas.set_projection_angle(angle)
    sample = canvas.sample
    arrays = {name: value.copy() for name, value in vars(sample).items()}
    mesh = tube_mesh()
    vertices = mesh.vertices.copy()
    canvas.set_physical_meshes((mesh,))
    references = canvas.references.copy()
    bounds = canvas._bounds_m.copy()
    plot = canvas._plot_rect()
    cursor = _axis_wheel_position(canvas, axis)
    if on_line:
        if axis == 0:
            cursor.setY(plot.bottom())
        else:
            cursor.setX(plot.left())
    anchor_screen = np.array([[cursor.x() if axis == 0 else plot.center().x(),
                               cursor.y() if axis == 1 else plot.center().y()]])
    anchor = canvas._section_positions(anchor_screen, canvas._section_inverse(plot))
    before = _visible_zu_corners(canvas)
    model_before = canvas._project_positions(mesh.vertices)[:, :2]
    requests, view_changes = [], []
    canvas.projection_angle_requested.connect(requests.append)
    canvas.view_changed.connect(lambda: view_changes.append(True))

    _wheel_at(canvas, cursor)

    # Ray Diagram's ViewBox uses range scale 1.02 ** (-delta / 8).
    scale = 1.02 ** (120. / 8.)
    after = _visible_zu_corners(canvas)
    np.testing.assert_allclose(after[:, 1-axis], before[:, 1-axis], atol=1e-16, rtol=0.)
    assert np.ptp(after[:, axis]) == pytest.approx(np.ptp(before[:, axis]) / scale, rel=1e-12)
    np.testing.assert_allclose(canvas._project_positions(anchor)[:, :2], anchor_screen, atol=1e-10)
    model_after = canvas._project_positions(mesh.vertices)[:, :2]
    np.testing.assert_allclose(model_after[:, 1-axis], model_before[:, 1-axis], atol=1e-10, rtol=0.)
    assert np.ptp(model_after[:, axis]) == pytest.approx(np.ptp(model_before[:, axis]) * scale)
    assert not requests and len(view_changes) == 1
    assert canvas.sample is sample and canvas.projection_angle_deg == angle
    assert canvas.references == references
    np.testing.assert_array_equal(canvas._bounds_m, bounds)
    np.testing.assert_array_equal(mesh.vertices, vertices)
    for name, before_array in arrays.items():
        np.testing.assert_array_equal(getattr(sample, name), before_array)


def test_sequential_axis_wheels_remain_independent_and_plot_zoom_keeps_their_ratio(qtbot):
    canvas = canvas_with_sample(qtbot, "UZ")
    canvas.set_projection_angle(37.)
    initial = _visible_zu_corners(canvas)
    _wheel_at(canvas, _axis_wheel_position(canvas, 0), 240)
    after_horizontal = _visible_zu_corners(canvas)
    _wheel_at(canvas, _axis_wheel_position(canvas, 1), -120)
    after_both = _visible_zu_corners(canvas)
    np.testing.assert_allclose(after_both[:, 0], after_horizontal[:, 0], atol=1e-16, rtol=0.)
    np.testing.assert_allclose(after_horizontal[:, 1], initial[:, 1], atol=1e-16, rtol=0.)
    assert np.ptp(after_both[:, 1]) > np.ptp(initial[:, 1])
    # Inside the plot both dimensions zoom, preserving an intentionally
    # independent aspect ratio rather than making the scales equal again.
    plot = canvas._plot_rect()
    _wheel_at(canvas, QPointF(plot.left() + .65 * plot.width(), plot.top() + .4 * plot.height()))
    after_plot = _visible_zu_corners(canvas)
    np.testing.assert_allclose(np.ptp(after_plot, axis=0),
                               np.ptp(after_both, axis=0) / (1.02 ** 15.), rtol=1e-12)


def test_axis_zoom_reset_and_fit_restore_both_display_dimensions(qtbot):
    canvas = canvas_with_sample(qtbot, "UZ")
    canvas.set_projection_angle(37.)
    mesh = tube_mesh()
    canvas.set_physical_meshes((mesh,))
    initial = canvas._project_positions(mesh.vertices).copy()
    _wheel_at(canvas, _axis_wheel_position(canvas, 0), 240)
    _wheel_at(canvas, _axis_wheel_position(canvas, 1), -120)
    canvas.reset_view()
    np.testing.assert_allclose(canvas._project_positions(mesh.vertices), initial, atol=1e-10, rtol=0.)
    canvas.fit_physical_models()
    fitted = canvas._project_positions(mesh.vertices).copy()
    _wheel_at(canvas, _axis_wheel_position(canvas, 0), -120)
    _wheel_at(canvas, _axis_wheel_position(canvas, 1), 360)
    canvas.fit_physical_models()
    np.testing.assert_allclose(canvas._project_positions(mesh.vertices), fitted, atol=1e-10, rtol=0.)
    assert canvas.projection_angle_deg == 37.
    assert all(canvas._plot_rect().contains(QPointF(*point)) for point in fitted[:, :2])


def test_wheel_outside_plot_and_axis_bands_does_not_change_view(qtbot):
    canvas = canvas_with_sample(qtbot, "UZ")
    points = canvas.sample.points_m.copy()
    before = canvas._project_positions(points).copy()
    changed = []
    canvas.view_changed.connect(lambda: changed.append(True))
    plot = canvas._plot_rect()
    for position in (QPointF(plot.center().x(), 10.),
                     QPointF(8., canvas.height()-8.),
                     QPointF(canvas.width()-8., plot.center().y())):
        _wheel_at(canvas, position)
    assert not changed
    np.testing.assert_array_equal(canvas._project_positions(points), before)
