"""Channel UI publication: explicit work, cancellation and stale optical inputs."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.corrector_channels import CorrectorChannelControls
from temsim.optics.column import default_state


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = CorrectorChannelControls()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    jobs = []
    monkeypatch.setattr(widget.pool, "start", jobs.append)
    monkeypatch.setattr(widget.pool, "clear", lambda: None)
    widget.jobs = jobs
    return widget


def _report(a1=0., a2=0., c3=1.):
    return SimpleNamespace(coefficients=SimpleNamespace(a1_mm=a1, a1_azimuth_deg=0.,
        a2_mm=a2, a2_azimuth_deg=0., c3_mm=c3, c1_mm=.001),
        diagnostics={"holdout_rms_m": 2e-9})


def _calibrate(view):
    view.calibrate()
    worker = view.jobs[-1]
    calibration = SimpleNamespace(baseline=_report(), jacobian=np.eye(2)*1e-6,
        steps=np.ones(2), control_specs=(SimpleNamespace(key="condenser_stigmator"),))
    view._ready(worker.generation, calibration, .2)
    view._finished(worker.generation)
    return calibration


def test_open_and_edits_do_not_compute_or_change_hardware(view):
    before = view._state.to_dict()
    view.channel.setCurrentIndex(3)
    view.x.setValue(2.)
    view.y.setValue(-3.)
    view.alpha.setValue(2.)
    assert not view.jobs and not view.apply_button.isEnabled()
    assert view._state.to_dict() == before
    assert view.calibrate_button.property("calculationAction")


def test_length_change_invalidates_response_even_without_gui_revision(view):
    _calibrate(view)
    hp = view._state.hp2_hexapole
    hp.effective_length_mm *= .9
    view.calculate_and_apply()
    assert len(view.jobs) == 1
    assert not view.apply_button.isEnabled()
    assert "stale" in view.status.text()


def test_options_and_revision_invalidate_calibration(view):
    _calibrate(view)
    view.channel.setCurrentIndex(1)
    assert view._calibration is None
    _calibrate(view)
    view.set_revision(1)
    assert view._calibration is None
    _calibrate(view)
    view.step.setValue(.025)
    assert view._calibration is None


def test_cancelled_worker_cannot_publish_or_displace_new_worker(view):
    view.calibrate()
    old = view.jobs[-1]
    view.cancel()
    assert old.cancellation.is_set()
    view.calibrate()
    current = view.jobs[-1]
    view._ready(old.generation, SimpleNamespace(baseline=_report()), .1)
    view._finished(old.generation)
    assert view._worker is current and view._calibration is None
    view.cancel()


def test_inputs_changed_during_measurement_discard_result(view):
    view.calibrate()
    worker = view.jobs[-1]
    view._state.hp2_hexapole.strength_m3 *= .9
    view._ready(worker.generation, SimpleNamespace(baseline=_report()), .1)
    assert view._calibration is None
    assert "Inputs changed" in view.status.text()


def test_apply_converts_nm_and_emits_one_invalidation_with_actual_c3(view, monkeypatch):
    from temsim.optics import corrector_channels as physics
    _calibrate(view)
    calls, signals = [], []
    monkeypatch.setattr(physics, "apply_channel_change", lambda state, candidate, **kw:
                        calls.append((state, candidate)))
    view.runtime_changed.connect(signals.append)
    view.x.setValue(2.)
    view.y.setValue(-3.)
    view.calculate_and_apply()
    worker = view.jobs[-1]
    assert worker.delta == pytest.approx((2e-6, -3e-6))
    candidate = SimpleNamespace(before=_report(c3=1.), after=_report(c3=.9),
        achieved_delta_xy_mm=worker.delta)
    view._ready(worker.generation, candidate, .3)
    assert calls == [(view._state, candidate)]
    assert signals == ["corrector_channels.calibrated_drives"]
    assert "C3 1 mm" in view.measured.text() and "C3 0.9 mm" in view.measured.text()
    assert "pupil-fit check RMS 2 nm" in view.measured.text()
    assert "same captured reference" in view.measured.text()
    assert "Tilt, defocus" in view.measured.text()
    assert not view.apply_button.isEnabled()
    view._ready(worker.generation, candidate, .3)
    assert len(calls) == 1


def test_submission_and_numerical_failure_leave_controls_reusable(view, monkeypatch):
    view.calibrate()
    worker = view.jobs[-1]
    view._failed(worker.generation, "Physical response has insufficient rank")
    view._finished(worker.generation)
    assert "insufficient rank" in view.status.text()
    assert view.calibrate_button.isEnabled()
    monkeypatch.setattr(view.pool, "start", lambda worker: (_ for _ in ()).throw(RuntimeError("queue full")))
    view.calibrate()
    assert view._worker is None and "queue full" in view.status.text()


def test_hardware_panel_exposes_channels_and_refreshes_raw_controls(qtbot):
    from temsim.gui.hardware_tuning_panel import HardwareTuningPanel
    panel = HardwareTuningPanel()
    qtbot.addWidget(panel)
    panel.set_state(default_state())
    panel.select_task("corrector_channels")
    assert not panel.channel_tabs.isHidden()
    assert panel.channel_tabs.currentWidget() is panel.probe_service
    service = panel.probe_service
    service.delta_percent.setValue(.09)
    assert not service.learning_enabled.isChecked()
    service.input.setCurrentIndex(service.input.findData("AT_A2.lower"))
    assert "probe_tl21_lens.percent" in service.preview.toPlainText()
    assert "hexapole" not in service.preview.toPlainText()
    assert "probe_dp22_deflector" not in service.preview.toPlainText()
    signals = []
    panel.runtime_changed.connect(signals.append)
    with qtbot.waitSignal(panel.runtime_changed):
        service.apply.click()
    assert signals == ["corrector_channels.service_drives"]
    service.tool.setCurrentText("TL2SymFoc")
    assert not service.apply.isEnabled()
    service.gains.item(0, 4).setText(".1")
    assert service.apply.isEnabled()
    assert panel._host.isHidden()
    panel.channel_tabs.setCurrentWidget(panel.corrector_channels)
    assert not panel.corrector_channels.isHidden()
    panel.select_task("probe_corrector")
    assert panel.channel_tabs.isHidden()
    assert panel._host.isHidden()
    assert not panel.probe_outputs.isHidden()
    raw = panel.probe_outputs
    raw.output.setCurrentText("QPoly")
    assert raw.maximum.text().startswith("300")
    raw.percent.setValue(10.)
    raw.apply.click()
    assert "10%" in raw.current.text()
    from temsim.optics.probe_service import set_probe_output_percent
    set_probe_output_percent(panel._state, "QPoly", percent=5.)
    raw.refresh()
    assert raw.percent.value() == 5.
    assert not raw.apply.isEnabled()
    assert panel.shutdown()
