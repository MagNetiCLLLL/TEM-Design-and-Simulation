"""Detached current-group editing; no physical source or wave execution."""
from copy import deepcopy

import pytest
from PySide6.QtWidgets import QDialogButtonBox

from temsim.gui.ideal_current_segments_dialog import IdealCurrentSegmentsDialog
from temsim.ideal_coil_segments import FIELD, set_current_segments
from temsim.optics.column import default_state


@pytest.fixture
def app(qapp):
    return qapp


@pytest.fixture
def state():
    value = default_state()
    value.simulation_mode = "unified"
    value.round_lens_field_model = "copper_coil"
    return value


def test_dialog_edits_stay_detached_until_applied(app, state):
    before = deepcopy(state.lens_field_map_descriptors)
    parts = state._resolved_assembly.parts
    dialog = IdealCurrentSegmentsDialog(state, "condenser_lens_1")
    dialog.groups.setValue(2)
    dialog._generate()
    assert dialog.table.rowCount() == 2
    dialog.table.item(1, 5).setText("-0.75")
    app.processEvents()
    assert float(dialog.table.item(1, 7).text()) < 0
    assert state.lens_field_map_descriptors == before
    dialog.accept()
    assert state.lens_field_map_descriptors == before
    set_current_segments(state, "condenser_lens_1", dialog.recipe)
    assert FIELD in state.lens_field_map_descriptors["condenser_lens_1"]
    assert state._resolved_assembly.parts is parts
    dialog.close()


def test_invalid_ratio_cannot_commit_and_cancel_is_inert(app, state):
    before = deepcopy(state.lens_field_map_descriptors)
    dialog = IdealCurrentSegmentsDialog(state, "condenser_lens_1")
    dialog.table.item(0, 5).setText("nan")
    app.processEvents()
    assert not dialog.buttons.button(QDialogButtonBox.StandardButton.Ok).isEnabled()
    assert "finite" in dialog.error.text()
    dialog.reject()
    assert state.lens_field_map_descriptors == before
    dialog.close()


def test_negative_lens_polarity_is_reflected_in_current_preview(app, state):
    dialog = IdealCurrentSegmentsDialog(state, "mini_condenser")
    lens = next(lens for lens in state.lenses if lens.key == "mini_condenser")
    assert lens.polarity == -1
    assert float(dialog.table.item(0, 6).text()) < 0
    assert float(dialog.table.item(0, 7).text()) < 0
    dialog.table.item(0, 5).setText("-1")
    app.processEvents()
    assert float(dialog.table.item(0, 7).text()) > 0
    dialog.close()


def test_mode_inspector_enables_editor_only_for_ideal_coils(app, state):
    from temsim.gui.model_inspector import ModelInspectorPage
    panel = ModelInspectorPage()
    panel.set_state(state)
    assert panel.current_segments_button.isEnabled()
    state.round_lens_field_model = "gaussian"
    panel._refresh_round_lens_controls()
    assert not panel.current_segments_button.isEnabled()
    panel.close()
