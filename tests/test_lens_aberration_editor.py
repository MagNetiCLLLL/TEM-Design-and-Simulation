"""Per-lens editing is transactional and never adjusts operating strengths."""

from copy import deepcopy

import pytest
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QSplitter, QWidget

from temsim.gui.lens_aberration_editor import LensAberrationEditor
from temsim.gui.parameter_panel import ParameterPanel
from temsim.optics.column import default_state
from temsim.runtime_parameters import RuntimeTarget


def _dialog(qtbot):
    state = default_state()
    lens = state.lenses[0]
    dialog = LensAberrationEditor(state, lens)
    qtbot.addWidget(dialog)
    return state, lens, dialog


def _operating(state):
    return tuple((lens.key, lens.percent, lens.polarity, lens.enabled,
                  lens.cs_mm, lens.cc_mm) for lens in state.lenses)


def test_apply_changes_only_selected_lens_and_keeps_optical_strengths(qtbot):
    state, lens, dialog = _dialog(qtbot)
    other = state.lenses[1]
    state.lens_intrinsic_aberrations = {other.key: {"a1_mm": 1.e-4}}
    original_other = deepcopy(state.lens_intrinsic_aberrations[other.key])
    operating = _operating(state)
    dialog.editors["a1_mm"].setText("2.5e-5")
    dialog.editors["a1_azimuth_deg"].setText("37.5")
    dialog.editors["c5_mm"].setText("-0.75")

    with qtbot.waitSignal(dialog.applied, timeout=1000):
        qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)

    values = state.lens_intrinsic_aberrations[lens.key]
    assert values["a1_mm"] == 2.5e-5
    assert values["a1_azimuth_deg"] == 37.5
    assert values["c5_mm"] == -.75
    assert state.lens_intrinsic_aberrations[other.key] == original_other
    assert _operating(state) == operating
    assert "cs_mm" not in dialog.editors and "c3_mm" not in dialog.editors
    assert "cc_mm" not in dialog.editors


@pytest.mark.parametrize("invalid", ["nan", "inf", "-inf", "no number", ""])
def test_invalid_draft_leaves_active_settings_untouched(qtbot, invalid):
    state, lens, dialog = _dialog(qtbot)
    state.lens_intrinsic_aberrations = {lens.key: {"b2_mm": 0.025}}
    before = deepcopy(state.lens_intrinsic_aberrations)
    dialog.editors["b2_mm"].setText("0.8")
    dialog.editors["c1_mm"].setText(invalid)
    assert dialog.apply_draft() is False
    assert state.lens_intrinsic_aberrations == before
    assert "Not applied" in dialog.status.text()


def test_reset_is_a_draft_and_close_does_not_apply(qtbot):
    state, lens, dialog = _dialog(qtbot)
    state.lens_intrinsic_aberrations = {lens.key: {"b2_mm": 0.025}}
    before = deepcopy(state.lens_intrinsic_aberrations)
    dialog.editors["b2_mm"].setText("0.8")
    dialog.reset_draft()
    assert all(float(editor.text()) == 0. for editor in dialog.editors.values())
    dialog.reject()
    assert state.lens_intrinsic_aberrations == before


def test_applying_reset_removes_only_this_lens(qtbot):
    state, lens, dialog = _dialog(qtbot)
    other = state.lenses[1]
    state.lens_intrinsic_aberrations = {
        lens.key: {"c1_mm": 1.e-5}, other.key: {"c5_mm": 2.}
    }
    operating = _operating(state)
    dialog.reset_draft()
    assert dialog.apply_draft()
    assert state.lens_intrinsic_aberrations == {other.key: {"c5_mm": 2.}}
    assert _operating(state) == operating


def test_field_map_explanation_does_not_solve_and_does_not_hide_saved_values(qtbot):
    state = default_state()
    lens = state.lenses[0]
    state.simulation_mode = "custom"
    state.lens_field_map_descriptors[lens.key] = {"solver": "axisymmetric_linear_fem"}
    state.lens_intrinsic_aberrations = {lens.key: {"a2_mm": .125}}
    dialog = LensAberrationEditor(state, lens)
    qtbot.addWidget(dialog)
    assert "not added" in dialog.model_note.text()
    assert "matching field" in dialog.model_note.text()
    assert dialog.editors["a2_mm"].text() == "0.125"
    assert dialog.apply_draft()
    assert state.lens_intrinsic_aberrations[lens.key]["a2_mm"] == .125


def test_parameter_panel_button_opens_editor_and_emits_invalidation(qtbot):
    state = default_state()
    lens = state.lenses[0]
    parent = QWidget()
    parent.state = state
    panel = ParameterPanel(parent)
    splitter = QSplitter(parent)
    splitter.addWidget(panel)
    qtbot.addWidget(parent)
    qtbot.addWidget(panel)
    panel.set_context(lens.name, RuntimeTarget(lens.key, lens.name, lens), None, (), None)
    assert panel.lens_intrinsic_button.isEnabled()
    events = []
    panel.runtime_changed.connect(events.append)

    def edit():
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, LensAberrationEditor)
        dialog.editors["a3_mm"].setText("1.125")
        qtbot.mouseClick(dialog.apply_button, Qt.MouseButton.LeftButton)
        dialog.reject()

    QTimer.singleShot(0, edit)
    qtbot.mouseClick(panel.lens_intrinsic_button, Qt.MouseButton.LeftButton)
    assert state.lens_intrinsic_aberrations[lens.key]["a3_mm"] == 1.125
    assert events == ["lens_intrinsic_aberrations"]
