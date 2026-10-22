"""Small CPU-only vector displays: scales and symbols retain their meaning."""
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QFont, QImage, QPainter

from temsim.gui.field_vector_canvas import FieldVectorCanvas


def _sample(b, e=None, valid=None):
    b = np.asarray(b, dtype=float)
    count = len(b)
    points = np.column_stack((np.zeros(count), np.zeros(count), np.linspace(.1, .9, count)))
    good = np.ones(count, bool) if valid is None else np.asarray(valid, bool)
    return SimpleNamespace(points_m=points, b_t=b,
                           e_v_per_m=b.copy() if e is None else np.asarray(e, float),
                           force_n=b.copy(), magnetic_valid=good,
                           electric_valid=good.copy(), force_valid=good.copy())


@pytest.fixture
def canvas(qtbot):
    view = FieldVectorCanvas()
    qtbot.addWidget(view)
    view.resize(1000, 420)
    view.set_mode("B")
    return view


BOUNDS = np.array([[-.1, -.1, 0.], [.1, .1, 1.]])


def test_auto_scale_replaces_old_tiny_e_reference_without_changing_samples(canvas):
    canvas.set_sample(_sample([[0., 0., .1]], [[0., 1.5809e-6, 0.]]), BOUNDS, "UZ")
    canvas.set_automatic_reference(True)
    fresh = _sample([[0., 0., .2]], [[0., 4.5502e5, 0.]])
    before = {name: value.copy() for name, value in vars(fresh).items()}
    canvas.set_sample(fresh, BOUNDS, "UZ")
    assert canvas.auto_reference
    assert canvas.references["B"] == .2
    assert canvas.references["E"] == 4.5502e5
    canvas.set_mode("overlay")
    assert "Auto scale" in canvas.legend_text
    assert "above arrow limit" not in canvas.legend_text
    assert "cyan" in canvas.legend_text and "amber" in canvas.legend_text
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(fresh, name), original)


def test_fixed_scale_retains_comparison_and_reports_actual_clipping(canvas):
    canvas.set_sample(_sample([[0., 0., 1.]]), BOUNDS, "UZ")
    canvas.set_automatic_reference(True)
    canvas.set_automatic_reference(False)
    fresh = _sample([[0., 0., 0.], [0., 0., 2.], [0., 0., 100.]])
    canvas.set_sample(fresh, BOUNDS, "UZ")
    assert canvas.references["B"] == 1.
    assert "Fixed scale" in canvas.legend_text
    assert "1/3 above arrow limit" in canvas.legend_text
    assert "|B| = 100 T" in canvas.sample_tooltip(2)
    assert "numeric value unchanged" in canvas.sample_tooltip(2)
    canvas.set_reference("B", 25.)
    assert "above arrow limit" not in canvas.legend_text
    assert canvas.references["B"] == 25.


def test_zero_unknown_normal_and_tiny_nonzero_have_distinct_visible_symbols(canvas, monkeypatch):
    sample = _sample([[0., 0., 0.], [0., 0., 1e-8], [0., 1., 0.],
                      [0., -1., 0.], [0., 0., 1.], [np.nan, np.nan, np.nan]],
                     valid=[True, True, True, True, True, False])
    canvas.set_sample(sample, BOUNDS, "UZ")
    canvas.set_reference("B", 1.)
    kinds = []
    original = canvas._glyph_kind

    def record(value, available, length, normal):
        kind = original(value, available, length, normal)
        kinds.append(kind)
        return kind

    monkeypatch.setattr(canvas, "_glyph_kind", record)
    canvas.grab()
    assert set(kinds) == {"zero", "below_scale", "normal", "arrow", "unavailable"}
    assert set(canvas._visible_sample_indices) == set(range(6))
    assert "zero at this sample" in canvas.sample_tooltip(0)
    assert "|B| = 1e-08 T" in canvas.sample_tooltip(1)
    assert "unavailable" in canvas.sample_tooltip(5)
    assert "not zero" in canvas.sample_tooltip(5)
    assert "○ zero" in canvas.symbol_legend
    assert "◇ below scale" in canvas.symbol_legend
    assert "⊙ +V / ⊗ -V" in canvas.symbol_legend


def test_zero_peak_is_explicitly_limited_to_sampled_values(canvas):
    canvas.set_sample(_sample([[0., 0., 0.]]), BOUNDS, "UZ")
    assert "sampled values are zero" in canvas.legend_text
    assert "not a whole-column guarantee" in canvas.legend_text


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_glyph_budget_keeps_nonzero_field_ahead_of_nearby_zero_and_unknown(canvas, budget):
    values = np.zeros((30, 3))
    values[29, 2] = 1.
    valid = np.ones(30, bool)
    valid[0] = False
    sample = _sample(values, valid=valid)
    canvas.set_sample(sample, BOUNDS, "UZ")
    # All projected points share one display cell. The nonzero field lies
    # farther from its centre than zero/unknown samples, as near-axis probes
    # can do in a compressed whole-column view.
    plot = canvas._plot_rect()
    projected = np.tile([plot.left()+12., plot.top()+12., 0.], (30, 1))
    projected[-1, :2] -= 9.
    selected = canvas._glyph_indices(projected, plot, max_glyphs=budget)
    assert 29 in selected
    assert len(selected) <= budget
    assert canvas.references["B"] == 1.
    np.testing.assert_array_equal(sample.b_t, values)


@pytest.mark.parametrize("angle", [0., 37., 90., 201.])
def test_axis_guide_tracks_actual_axis_under_rotation_pan_and_anisotropic_zoom(canvas, angle):
    bounds = BOUNDS + np.array([.01, .02, 0.])
    canvas.set_sample(_sample([[0., 0., 1.]]), bounds, "UZ")
    canvas.set_projection_angle(angle)
    canvas._zoom = np.array([2.3, .8])
    canvas._pan = QPointF(31., -19.)
    plot = canvas._plot_rect()
    guide = canvas.axis_guide()
    assert guide is not None
    np.testing.assert_allclose(guide[:, 0], [plot.left(), plot.right()])
    # Inverting its screen endpoints must return the actual global X=Y=0
    # axis, not the sample-window centre or a screen-centred decoration.
    points = canvas._section_positions(guide, canvas._section_inverse(plot))
    np.testing.assert_allclose(points[:, :2], 0., atol=1e-14)
    before_y = guide[0, 1]
    canvas._pan += QPointF(7., 11.)
    assert canvas.axis_guide()[0, 1] == pytest.approx(before_y+11.)


@pytest.mark.parametrize("pan", [-10000., 10000.])
def test_axis_guide_is_hidden_when_axis_is_outside_view(canvas, pan):
    canvas.set_sample(_sample([[0., 0., 1.]]), BOUNDS, "UZ")
    canvas._pan = QPointF(0., pan)
    assert canvas.axis_guide() is None


def test_axis_guide_is_only_a_longitudinal_coordinate_guide(canvas):
    assert canvas.axis_guide() is None
    canvas.set_sample(_sample([[0., 0., 1.]]), BOUNDS, "3D")
    assert canvas.axis_guide() is None


def test_native_symbol_legend_uses_ascii_labels_and_fits_laptop_width(canvas):
    class RecordingPainter(QPainter):
        def __init__(self, image):
            super().__init__(image)
            self.labels = []
            self.right_edges = []
            self.shapes = []

        def drawText(self, point, text):
            self.labels.append(text)
            self.right_edges.append(point.x()+self.fontMetrics().horizontalAdvance(text))
            super().drawText(point, text)

        def drawRect(self, *args):
            self.shapes.append("square")
            super().drawRect(*args)

        def drawEllipse(self, *args):
            self.shapes.append("circle")
            super().drawEllipse(*args)

        def drawPolygon(self, *args):
            self.shapes.append("diamond")
            super().drawPolygon(*args)

    canvas.resize(700, 420)
    image = QImage(canvas.size(), QImage.Format.Format_ARGB32)
    image.fill(QColor("black"))
    painter = RecordingPainter(image)
    painter.setFont(QFont("Segoe UI", 10))
    try:
        canvas._draw_symbol_legend(painter, QColor("white"))
        assert painter.labels == ["Unavailable", "zero", "below scale", "+V", "-V", "cap 4x ref"]
        assert all(text.isascii() for text in painter.labels)
        assert max(painter.right_edges) <= canvas.width()-8
        assert painter.shapes.count("square") == 1
        assert painter.shapes.count("circle") == 3
        assert painter.shapes.count("diamond") == 1
    finally:
        painter.end()
