from copy import deepcopy
from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QSignalSpy

from temsim.gui.vacuum_axial_view import VacuumAxialView
from temsim.vacuum_pumping import default_pumping_layout


@pytest.fixture
def diagram(qtbot):
    regions = [
        SimpleNamespace(key=key, name=key.replace("_", " "), start_z_mm=start, end_z_mm=end,
                        radius_mm=None, end_medium=None, medium=SimpleNamespace(phase="gas", pressure_mbar=1e-7))
        for key, start, end in (
            ("gun_tip", 0., 5.), ("gun_accelerator", 5., 450.), ("column", 450., 1599.19),
            ("specimen", 1599.19, 1599.21), ("post_column", 1599.21, 2200.), ("projection", 2200., 3026.4),
        )
    ]
    view = VacuumAxialView()
    qtbot.addWidget(view)
    view.resize(1000, 650)
    view.set_regions(regions, ())
    view.setXRange(0, 3100, padding=0)
    view.set_pumping_layout(default_pumping_layout(regions), {"sample": 1599.2})
    view.show()
    qtbot.wait(30)
    return view


def test_ports_follow_real_z_and_leave_narrow_specimen_unchanged(diagram):
    vb, overlay = diagram.getViewBox(), diagram.pumping_overlay
    for row in diagram.rows:
        actual = overlay.mapToScene(overlay.region_ports[f"region:{row.key}"])
        expected = vb.mapViewToScene(QPointF((row.start_z_mm+row.end_z_mm)/2, 1.0))
        assert actual.x() == pytest.approx(expected.x(), abs=.01)
        box = next(rect for rect, key in diagram.boxes if key == row.key)
        assert box.left() == row.start_z_mm
        assert box.width() == pytest.approx(row.end_z_mm-row.start_z_mm)
    specimen = next(rect for rect, key in diagram.boxes if key == "specimen")
    assert specimen.width() == pytest.approx(.02)
    assert len(overlay.node_rects) == 7 and len(overlay.line_paths) == 10


def test_show_toggle_and_reconnect_never_change_horizontal_scale(diagram, qtbot):
    before_range = diagram.getViewBox().viewRange()[0][:]
    before_pixels = diagram.getViewBox().width()
    boxes = [(key, rect.left(), rect.width()) for rect, key in diagram.boxes]
    diagram.set_pumping_visible(False)
    qtbot.wait(10)
    assert not diagram.pumping_overlay.isVisible()
    diagram.set_pumping_visible(True)
    layout = deepcopy(diagram.pumping_overlay.layout)
    layout["lines"][0]["target"] = "pump:turbo"
    diagram.set_pumping_layout(layout, {"sample": 1599.2})
    qtbot.wait(10)
    assert diagram.getViewBox().viewRange()[0] == before_range
    assert diagram.getViewBox().width() == pytest.approx(before_pixels, abs=.01)
    assert [(key, rect.left(), rect.width()) for rect, key in diagram.boxes] == boxes


def test_zoom_updates_true_ports_and_omits_offscreen_branches(diagram, qtbot):
    overlay = diagram.pumping_overlay
    diagram.setXRange(0, 600, padding=0)
    qtbot.wait(10)
    point = overlay.region_ports["region:projection"]
    assert point.x() > overlay.boundingRect().right()
    assert "projection_turbo" not in overlay.line_paths
    assert "pump:turbo" in overlay.node_rects
    assert overlay._offscreen > 0
    expected = diagram.getViewBox().mapViewToScene(QPointF(227.5, 1.0))
    assert overlay.mapToScene(overlay.region_ports["region:gun_accelerator"]).x() == pytest.approx(expected.x(), abs=.01)


def test_shared_physical_plot_scale_stays_bidirectional(diagram, qtbot):
    physical = pg.PlotWidget()
    qtbot.addWidget(physical)
    physical.setXRange(100, 900, padding=0)
    diagram.bind_physical_plot(physical)
    assert diagram.getViewBox().viewRange()[0] == pytest.approx([100, 900])
    diagram.set_pumping_visible(False)
    diagram.set_pumping_visible(True)
    assert physical.getViewBox().viewRange()[0] == pytest.approx([100, 900])
    diagram.setXRange(1400, 1700, padding=0)
    assert physical.getViewBox().viewRange()[0] == pytest.approx([1400, 1700])


def test_real_clicks_emit_selection_without_deleting_overlay(diagram, qtbot):
    overlay = diagram.pumping_overlay
    pump_spy, line_spy, region_spy = QSignalSpy(diagram.pumping_selected), QSignalSpy(diagram.pumping_line_selected), QSignalSpy(diagram.selected)
    diagram.pumping_selected.connect(lambda _: diagram.set_pumping_layout(deepcopy(overlay.layout), {"sample": 1599.2}))
    node = overlay.node_rects["pump:turbo"]
    qtbot.mouseClick(diagram.viewport(), Qt.MouseButton.LeftButton,
                    pos=diagram.mapFromScene(overlay.mapToScene(node.center())))
    assert pump_spy.count() == 1 and pump_spy.at(0) == ["pump:turbo"]
    assert diagram.pumping_overlay is overlay and overlay.scene() is diagram.scene()
    rect, _ = overlay.line_labels["projection_turbo"]
    qtbot.mouseClick(diagram.viewport(), Qt.MouseButton.LeftButton,
                    pos=diagram.mapFromScene(overlay.mapToScene(rect.center())))
    assert line_spy.count() == 1 and line_spy.at(0) == ["projection_turbo"]
    qtbot.mouseClick(diagram.viewport(), Qt.MouseButton.LeftButton,
                    pos=diagram.mapFromScene(overlay.mapToScene(overlay.region_ports["region:projection"])))
    assert region_spy.count() == 1 and region_spy.at(0) == ["projection"]


def test_original_region_click_is_not_intercepted(diagram, qtbot):
    spy = QSignalSpy(diagram.selected)
    scene = diagram.getViewBox().mapViewToScene(QPointF(2700, 1.35))
    assert not any(diagram.pumping_overlay.hit_at(diagram.pumping_overlay.mapFromScene(scene)))
    qtbot.mouseClick(diagram.viewport(), Qt.MouseButton.LeftButton, pos=diagram.mapFromScene(scene))
    assert spy.count() == 1 and spy.at(0) == ["projection"]


def test_pending_removed_region_reference_does_not_create_false_port(diagram):
    old_layout = deepcopy(diagram.pumping_overlay.layout)
    diagram.set_regions(diagram.rows[:-1], ())
    diagram.set_pumping_layout(old_layout, {"sample": 1599.2})
    assert "region:projection" not in diagram.pumping_overlay.region_ports
    assert "projection_turbo" not in diagram.pumping_overlay.line_paths
    assert "pump:turbo" in diagram.pumping_overlay.node_rects


def test_missing_auxiliary_anchor_is_explicit_and_does_not_change_saved_reference(diagram):
    overlay = diagram.pumping_overlay
    layout = deepcopy(overlay.layout)
    layout["auxiliaries"][0]["anchor"] = "component:missing.center"
    diagram.set_pumping_layout(layout, {"sample": 1599.2})
    assert overlay._node_data["aux:airlock"][2] == "Location unavailable"
    assert layout["auxiliaries"][0]["anchor"] == "component:missing.center"


def test_small_page_keeps_size_and_readable_column_when_editor_opens(qtbot):
    from temsim.gui.vacuum_map_page import VacuumMapPage
    from temsim.optics.column import default_state
    from temsim.vacuum import boundary_anchors
    page = VacuumMapPage()
    qtbot.addWidget(page)
    page.resize(1000, 650)
    state = default_state()
    page.set_state(state)
    page.diagram.setXRange(0, boundary_anchors(state)["column_end"], padding=0)
    page.show()
    qtbot.wait(30)
    assert page.width() == 1000 and page.height() == 650
    page.select_region("projection")
    qtbot.wait(30)
    assert page.width() == 1000 and page.height() == 650
    assert page.diagram.height() >= 420
    assert page.diagram.pumping_overlay._service_rect.top() >= 175
    assert not page.editor.isHidden()
