"""Bounded section refinement and display-only glyph density on a CPU widget."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication, QToolTip

from temsim.gui.field_vector_canvas import FieldVectorCanvas


def test_envelope_cuts_are_reused_for_pan_zoom_but_rebuilt_for_angle(qtbot, monkeypatch):
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    bounds = np.array([[-.001, -.001, .1], [.001, .001, .15]])
    canvas.set_sample(_sample([[0., 0., .125]]), bounds, "UZ")
    canvas.set_source_regions((("coil", bounds),))
    calls = []
    original = canvas._longitudinal_box_cut

    def cut(region):
        calls.append(region.copy())
        return original(region)

    monkeypatch.setattr(canvas, "_longitudinal_box_cut", cut)
    first = canvas._source_outlines()
    canvas._pan = QPointF(19., -11.)
    canvas._zoom = np.array([1.5, .8])
    canvas._view_updated()
    canvas.set_source_regions((("coil", bounds.copy()),))
    assert canvas._source_outlines() is first and len(calls) == 1
    canvas.set_projection_angle(35.)
    assert canvas._source_outlines() is not first and len(calls) == 2
    canvas.set_source_regions((("coil", bounds*1.1),))
    canvas._source_outlines()
    assert len(calls) == 3


def _sample(points, values=None):
    points = np.asarray(points, dtype=float)
    values = np.tile((.2, .4, .8), (len(points), 1)) if values is None else np.asarray(values)
    valid = np.ones(len(points), dtype=bool)
    return SimpleNamespace(points_m=points, b_t=values, e_v_per_m=values*2,
                           force_n=values*3, magnetic_valid=valid,
                           electric_valid=valid, force_valid=valid)


def _canvas(qtbot, section="XZ", angle=0.):
    canvas = FieldVectorCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(900, 600)
    bounds = np.array([[-.001, -.002, .1], [.001, .002, .15]])
    canvas.set_sample(_sample([[0., 0., .125]]), bounds, section)
    canvas.set_projection_angle(angle)
    canvas.show()
    QApplication.processEvents()
    return canvas


def _wheel(canvas, position, steps):
    event = QWheelEvent(position, QPointF(canvas.mapToGlobal(position.toPoint())),
                        QPoint(), QPoint(0, steps*120), Qt.MouseButton.NoButton,
                        Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.NoScrollPhase, False)
    canvas.wheelEvent(event)


@pytest.mark.parametrize("section,angle,normal", [("XY", 0., 2), ("XZ", 31., 1), ("YZ", 62., 0)])
def test_real_zoom_refines_global_coordinates_and_keeps_section_fixed(qtbot, section, angle, normal):
    canvas = _canvas(qtbot, section, angle)
    bounds = canvas._bounds_m.copy()
    original, original_key = canvas.viewport_sample_grid()
    mouse = canvas._plot_rect().center()+QPointF(80., -45.)
    centre, span, basis, size, origin = canvas._projection()
    active = [axis for axis in range(3) if axis != normal]
    reference = bounds.mean(axis=0)
    matrix = size[:, None]*basis[:2, active]/span[active]
    reference[active] += np.linalg.solve(matrix, np.array([mouse.x(), mouse.y()])
                                         -canvas._project_positions(reference[None])[0, :2])
    _wheel(canvas, mouse, 8)
    zoomed, zoomed_key = canvas.viewport_sample_grid()
    np.testing.assert_allclose(canvas._project_positions(reference[None])[0, :2], [mouse.x(), mouse.y()])
    assert np.ptp(zoomed, axis=0)[active].sum() < np.ptp(original, axis=0)[active].sum()
    assert zoomed_key != original_key
    np.testing.assert_array_equal(zoomed[:, normal], bounds.mean(axis=0)[normal])
    assert np.all(zoomed >= bounds[0]) and np.all(zoomed <= bounds[1])
    np.testing.assert_array_equal(canvas._bounds_m, bounds)
    # Coordinates are visible apart from the subpixel outward snap at edges.
    screen = canvas._project_positions(zoomed)[:, :2]
    plot = canvas._plot_rect()
    assert screen[:, 0].min() >= plot.left()-1. and screen[:, 0].max() <= plot.right()+1.
    assert screen[:, 1].min() >= plot.top()-1. and screen[:, 1].max() <= plot.bottom()+1.


@pytest.mark.parametrize("section,angle", [("XZ", 90.), ("XZ", 270.), ("YZ", 0.), ("YZ", 180.)])
def test_edge_on_section_does_not_invent_inverse_coordinates(qtbot, section, angle):
    canvas = _canvas(qtbot, section, angle)
    assert canvas.viewport_sample_grid() is None


def test_pan_beyond_physical_domain_is_empty_instead_of_edge_on(qtbot):
    canvas = _canvas(qtbot, "XZ")
    canvas._pan = QPointF(10_000., 0.)
    points, signature = canvas.viewport_sample_grid()
    assert points.shape == (0, 3)
    canvas._pan = QPointF(20_000., 0.)
    points_again, signature_again = canvas.viewport_sample_grid()
    assert signature == signature_again and points_again.shape == (0, 3)
    assert hash(signature)
    canvas.set_sample(_sample(points), canvas._bounds_m, "XZ")
    canvas.grab()
    assert len(canvas._visible_sample_indices) == 0


def test_grid_budget_and_stable_quantized_cache_coordinates(qtbot):
    canvas = _canvas(qtbot, "XY")
    canvas.resize(6000, 5000)
    points, key = canvas.viewport_sample_grid(count=21)
    assert len(points) <= 1600
    smaller, _key = canvas.viewport_sample_grid(count=21, max_points=100)
    assert len(smaller) <= 100
    assert canvas.viewport_sample_grid(max_points=3) is None
    # Exact same view and tiny subpixel pan reuse an identical physical grid.
    again, same_key = canvas.viewport_sample_grid(count=21)
    assert key == same_key
    np.testing.assert_array_equal(points, again)
    canvas._zoom = np.full(2, 2.37)
    canvas._pan = QPointF(2.123, -6.123)
    first, key = canvas.viewport_sample_grid()
    canvas._pan += QPointF(.00000001, -.00000001)
    second, same_key = canvas.viewport_sample_grid()
    assert key == same_key
    np.testing.assert_array_equal(first, second)


def test_3d_navigation_stays_cached_and_view_events_do_not_publish_samples(qtbot):
    canvas = _canvas(qtbot, "3D")
    sample = canvas.sample
    reference = canvas.references.copy()
    events = []
    canvas.view_changed.connect(lambda: events.append(True))
    _wheel(canvas, canvas._plot_rect().center(), 3)
    canvas.set_projection_angle(47.)
    canvas.fit_physical_models()
    canvas.reset_view()
    assert len(events) == 4
    assert canvas.viewport_sample_grid() is None
    assert canvas.sample is sample and canvas.references == reference
    canvas.set_sample(sample, canvas._bounds_m, "3D")
    assert len(events) == 4  # Publication cannot recursively schedule refinement.
    canvas.resize(700, 500)
    QApplication.processEvents()
    assert len(events) == 5


def test_section_pan_requests_new_visible_bounds_without_touching_field_data(qtbot):
    canvas = _canvas(qtbot, "XZ")
    canvas._zoom = np.full(2, 3.)
    initial, initial_key = canvas.viewport_sample_grid()
    sample = canvas.sample
    signals = []
    canvas.view_changed.connect(lambda: signals.append(True))
    start = canvas._plot_rect().center()
    end = start+QPointF(80., 35.)
    for kind, pos, button, buttons in (
            (QEvent.Type.MouseButtonPress, start, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseMove, end, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton),
            (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton)):
        event = QMouseEvent(kind, pos, QPointF(canvas.mapToGlobal(pos.toPoint())),
                            button, buttons, Qt.KeyboardModifier.NoModifier)
        if kind == QEvent.Type.MouseButtonPress:
            canvas.mousePressEvent(event)
        elif kind == QEvent.Type.MouseMove:
            canvas.mouseMoveEvent(event)
        else:
            canvas.mouseReleaseEvent(event)
    moved, moved_key = canvas.viewport_sample_grid()
    assert moved_key != initial_key and len(signals) == 1
    assert moved[:, 2].mean() < initial[:, 2].mean()
    assert moved[:, 0].mean() > initial[:, 0].mean()
    assert canvas.sample is sample


def test_glyph_declutter_preserves_normal_signs_values_and_references(qtbot):
    canvas = _canvas(qtbot, "XY")
    # Coincident projections include positive/negative normal components.
    points = np.tile((0., 0., .125), (120, 1))
    values = np.tile((.2, .4, 1.), (120, 1))
    values[1::2, 2] = -1.
    sample = _sample(points, values)
    bounds = canvas._bounds_m.copy()
    canvas.set_sample(sample, bounds, "XY")
    references = canvas.references.copy()
    before = {name: array.copy() for name, array in vars(sample).items()}
    canvas.grab()
    indices = canvas._visible_sample_indices
    assert len(indices) == 2
    assert set(np.sign(values[indices, 2])) == {-1., 1.}
    assert canvas.references == references
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(sample, name), value)
    # Dispersed dense samples still obey the hard display budget.
    xx, yy = np.meshgrid(np.linspace(-.001, .001, 100), np.linspace(-.002, .002, 100))
    dense = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, .125)))
    canvas.set_sample(_sample(dense), bounds, "XY")
    drawn = canvas._glyph_indices(canvas._project_positions(dense), max_glyphs=100)
    assert 0 < len(drawn) <= 100
    assert canvas.references == references


def test_hover_only_reports_drawn_samples_and_ignores_outside_plot(qtbot, monkeypatch):
    canvas = _canvas(qtbot, "XY")
    points = np.tile((0., 0., .125), (3, 1))
    points[1, 0] = .000001
    points[2, 0] = .004  # Outside the physical view.
    canvas.set_sample(_sample(points), canvas._bounds_m, "XY")
    canvas.grab()
    assert len(canvas._visible_sample_indices) == 1
    shown = []
    monkeypatch.setattr(QToolTip, "showText", lambda _position, text, _widget: shown.append(text))
    selected = int(canvas._visible_sample_indices[0])
    pos = QPointF(*canvas._screen_points[selected])
    event = QMouseEvent(QEvent.Type.MouseMove, pos, QPointF(canvas.mapToGlobal(pos.toPoint())),
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    canvas.mouseMoveEvent(event)
    assert f"X={points[selected, 0]*1000:.6g}" in shown[-1]
    outside = QPointF(5., 5.)
    event = QMouseEvent(QEvent.Type.MouseMove, outside, outside, Qt.MouseButton.NoButton,
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    canvas.mouseMoveEvent(event)
    assert len(shown) == 1
    # Changing the camera invalidates stale hit locations before repaint.
    _wheel(canvas, pos, 2)
    assert len(canvas._visible_sample_indices) == 0

