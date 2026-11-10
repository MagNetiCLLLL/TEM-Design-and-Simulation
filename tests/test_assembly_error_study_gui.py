"""Detached study publication and cancellation in the hardware-tuning panel."""
import pytest

from temsim.gui.assembly_error_study import AssemblyErrorStudyControls
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = AssemblyErrorStudyControls()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    jobs = []
    monkeypatch.setattr(widget.pool, "start", jobs.append)
    monkeypatch.setattr(widget.pool, "clear", lambda: None)
    widget.jobs = jobs
    return widget


def test_open_and_options_preserve_hardware_and_learning_default(view):
    before = capture_instrument_snapshot(view._state).physical_digest
    assert not view.learning.isChecked()
    view.learning.setChecked(True)
    view.editors["calibration_step_percent"].setValue(.02)
    assert not view.jobs
    assert before == capture_instrument_snapshot(view._state).physical_digest
    assert not view.save_button.isEnabled()


def test_worker_has_captured_inputs_and_cancelled_generation_cannot_publish(view):
    view.start()
    assert len(view.jobs) == 1, view.status.text()
    old = view.jobs[0]
    assert not old.request.options.compare_learning
    view.cancel()
    assert old.cancellation.is_set()
    view.start()
    current = view.jobs[-1]
    view._ready(old.generation, {})
    view._finished(old.generation)
    assert view._worker is current and view._saved is None
    view.cancel()


def test_stale_state_without_revision_does_not_install_results(view):
    view.start()
    worker = view.jobs[0]
    view._state.hp1_hexapole.strength_m3 *= .99
    view._ready(worker.generation, {})
    assert view._saved is None
    assert "Inputs changed" in view.status.text()


def test_revision_cancels_active_job(view):
    view.start()
    worker = view.jobs[0]
    view.set_revision(2)
    assert worker.cancellation.is_set() and view._worker is None
    assert "fresh study" in view.status.text()


def test_error_parameters_keep_readable_units_and_separate_rotation_types(view, qtbot):
    view.resize(1000, 1200)
    view.show()
    qtbot.wait(20)
    component = view.errors.cellWidget(0, 0)
    parameter = view.errors.cellWidget(0, 1)
    values = {parameter.itemData(i): parameter.itemText(i) for i in range(parameter.count())}
    assert "field only" in values["field_azimuth_error_deg"]
    assert "mrad" in values["rotation_x_mrad"]
    for combo in (component, parameter):
        assert combo.width() > combo.fontMetrics().horizontalAdvance(combo.currentText()) + 20
