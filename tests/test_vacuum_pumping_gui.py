"""Displayed vacuum plumbing stays editable without changing particle inputs."""
from copy import deepcopy

import pytest

from temsim.gui.vacuum_map_page import VacuumMapPage
from temsim.gui.vacuum_pumping_editor import VacuumPumpingDialog
from temsim.vacuum import VacuumMap
from temsim.vacuum_pumping import effective_pumping_layout


@pytest.fixture
def page(qtbot):
    from temsim.optics.column import default_state
    widget = VacuumMapPage()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    return widget


def _line_row(dialog, key):
    return next(row for row in range(dialog.lines.rowCount()) if dialog._key(dialog.lines, row) == key)


def test_display_install_preserves_transport_results_and_unsaved_pressure(page, qtbot):
    page.select_region("projection")
    page.pressure.setText("2e-5")  # Still an unapplied region edit.
    page.result_text.setText("Retained particle result")
    before = page.state.vacuum_map.transport_dict()
    physics, display = [], []
    page.changed.connect(physics.append)
    page.display_changed.connect(display.append)
    layout = effective_pumping_layout(page.state.vacuum_map)
    layout["pumps"][0]["name"] = "My gun pump"
    layout["lines"][0]["valve"], layout["lines"][0]["open"] = "Gun isolation", False
    assert page.install_pumping(layout)
    assert page.state.vacuum_map.transport_dict() == before
    assert page.result_text.text() == "Retained particle result"
    assert page.pressure.text() == "2e-5"
    assert physics == [] and display == ["vacuum_pumping"]
    assert not page.state.vacuum_map.enabled
    layout["pumps"][0]["name"] = "Caller mutation"
    assert page.state.vacuum_map.pumping["pumps"][0]["name"] == "My gun pump"
    assert page.show_pumping.isChecked()
    assert "pump:turbo" in page.diagram.pumping_overlay.node_rects


def test_dialog_cancel_is_transactional_and_shared_pump_has_one_identity(page, qtbot):
    previous = deepcopy(page.state.vacuum_map.to_dict())
    dialog = VacuumPumpingDialog(page.state)
    qtbot.addWidget(dialog)
    row = _line_row(dialog, "airlock_turbo")
    dialog.lines.cellWidget(row, 4).setCurrentIndex(0)
    candidate = dialog.candidate_layout()
    assert len([pump for pump in candidate["pumps"] if pump["key"] == "turbo"]) == 1
    assert {line["source"] for line in candidate["lines"] if line["target"] == "pump:turbo"} >= {
        "region:projection", "aux:airlock"}
    dialog.reject()
    assert page.state.vacuum_map.to_dict() == previous
    assert dialog.result_layout is None


def test_add_rename_reconnect_and_delete_pump_roundtrip(page, qtbot, tmp_path):
    dialog = VacuumPumpingDialog(page.state)
    qtbot.addWidget(dialog)
    dialog.add_pump()
    row = dialog.pumps.rowCount()-1
    key = dialog._key(dialog.pumps, row)
    dialog.pumps.item(row, 0).setText("Shared service pump")
    kind = dialog.pumps.cellWidget(row, 1)
    kind.setCurrentIndex(kind.findData("turbo"))
    for line_key in ("projection_turbo", "airlock_turbo"):
        target = dialog.lines.cellWidget(_line_row(dialog, line_key), 1)
        target.setCurrentIndex(target.findData(f"pump:{key}"))
    candidate = dialog.candidate_layout()
    assert page.install_pumping(candidate)
    path = tmp_path/"plumbing.toml"
    page.state.vacuum_map.save(path)
    assert VacuumMap.load(path).pumping == candidate
    dialog.pumps.selectRow(row)
    dialog._remove_node(dialog.pumps, "pump")
    removed = dialog.candidate_layout()
    assert all(line["target"] != f"pump:{key}" and line["source"] != f"pump:{key}"
               for line in removed["lines"])
    assert page.state.vacuum_map.pumping == candidate  # Second draft not yet saved.


def test_invalid_rewiring_and_closed_unvalved_pipe_leave_map_intact(page, qtbot):
    dialog = VacuumPumpingDialog(page.state)
    qtbot.addWidget(dialog)
    previous = deepcopy(page.state.vacuum_map.to_dict())
    row = _line_row(dialog, "turbo_backing")
    dialog.lines.cellWidget(row, 4).setCurrentIndex(1)
    dialog.accept()
    assert "without a valve" in dialog.status.text()
    assert dialog.result_layout is None
    invalid = effective_pumping_layout(page.state.vacuum_map)
    invalid["lines"][0]["target"] = "pump:missing"
    assert not page.install_pumping(invalid)
    assert "missing endpoint" in page.status.text()
    assert page.state.vacuum_map.to_dict() == previous


@pytest.mark.parametrize("duplicate_unrelated", [False, True])
def test_split_and_merge_preserve_shared_services_without_dangling_pipes(page, duplicate_unrelated):
    original = effective_pumping_layout(page.state.vacuum_map)
    if duplicate_unrelated:
        pipe = deepcopy(next(line for line in original["lines"] if line["key"] == "gun_tip_igp"))
        pipe["key"] = "separate_gun_line"
        original["lines"].append(pipe)
        assert page.install_pumping(original)
    page.select_region("column")
    page.split_region()
    new_key = page.current_key
    assert new_key.startswith("region_")
    split = effective_pumping_layout(page.state.vacuum_map)
    original_targets = {(line["target"], line["role"], line["valve"], line["open"])
                        for line in split["lines"] if line["source"] == "region:column"}
    split_targets = {(line["target"], line["role"], line["valve"], line["open"])
                     for line in split["lines"] if line["source"] == f"region:{new_key}"}
    assert original_targets == split_targets and original_targets
    page.remove_region()
    assert effective_pumping_layout(page.state.vacuum_map) == original
    page.state.vacuum_map.validate()


def test_map_open_with_only_plumbing_change_keeps_particle_result(page):
    candidate = deepcopy(page.state.vacuum_map)
    candidate.pumping = effective_pumping_layout(candidate)
    candidate.pumping["pumps"][0]["kind"] = "cryo"
    changed = []
    page.changed.connect(changed.append)
    page.result_text.setText("Current")
    page._install(candidate, "Loaded")
    assert page.result_text.text() == "Current" and changed == []
    assert page.state.vacuum_map.pumping["pumps"][0]["kind"] == "cryo"
    candidate.regions[0].medium.pressure_mbar *= 2
    page._install(candidate, "New pressure")
    assert changed == ["vacuum_map"]
    assert "out of date" in page.result_text.text()


def test_pumps_attach_to_existing_map_without_changing_column_scale(page, qtbot):
    from PySide6.QtCore import QPointF, Qt
    page.resize(1000, 650)
    page.show()
    page.diagram.setXRange(0, 3500, padding=0)
    vb = page.diagram.getViewBox()
    original = {key: (box.left(), box.width()) for box, key in page.diagram.boxes}
    scale = vb.mapViewToScene(QPointF(100, 1.35)).x()-vb.mapViewToScene(QPointF(0, 1.35)).x()
    for enabled in (False, True):
        page.show_pumping.setChecked(enabled)
        qtbot.wait(10)
        assert {key: (box.left(), box.width()) for box, key in page.diagram.boxes} == original
        assert vb.viewRange()[0] == pytest.approx([0, 3500])
        actual = vb.mapViewToScene(QPointF(100, 1.35)).x()-vb.mapViewToScene(QPointF(0, 1.35)).x()
        assert actual == pytest.approx(scale)
    start, width = original["projection"]
    scene = vb.mapViewToScene(QPointF(start+width/2, 1.35))
    qtbot.wait(10)
    qtbot.mouseClick(page.diagram.viewport(), Qt.MouseButton.LeftButton,
                     pos=page.diagram.mapFromScene(scene))
    qtbot.wait(10)
    assert page.current_key == "projection"
    assert "Shared turbo" in page.pump_assignment.text()
    page.diagram.pumping_line_selected.emit("airlock_turbo")
    assert page.editor.isHidden() and page._pumping_line == "airlock_turbo"
    assert vb.viewRange()[0] == pytest.approx([0, 3500])
    page.selection.setCurrentIndex(page.selection.findData("gun_tip"))
    assert page.diagram.pumping_overlay.current == "region:gun_tip"


def test_pumping_edit_does_not_request_main_window_preview(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell
    settings = QSettings(str(tmp_path/"vacuum.ini"), QSettings.Format.IniFormat)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    calls = []
    monkeypatch.setattr(shell.MainWindow, "_runtime_parameter_changed", lambda *_args: calls.append("runtime"))
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    page = window.workspace.vacuum_map
    page.set_state(window.state)
    calls.clear()
    layout = effective_pumping_layout(window.state.vacuum_map)
    layout["pumps"][0]["name"] = "Renamed IGP"
    assert page.install_pumping(layout)
    assert not calls and not window.preview_timer.isActive()
    assert window.state.vacuum_map.pumping["pumps"][0]["name"] == "Renamed IGP"
