"""Scan controls remain responsive and commit only current complete solutions."""
from copy import deepcopy
from threading import Event, get_ident

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QTimer

from temsim.gui import scan_calibration_controller as controller
from temsim.gui.scan_panel import ScanControlView
from temsim.physics import unified_scan_calibration
from temsim.physics.scan_geometry import calibrate_scan_system
from test_unified_scan_calibration import _state, _drift_response


@pytest.fixture(autouse=True)
def stable_solver_identity(monkeypatch):
    # These controlled workers do not execute physics. Give their captured
    # implementation an explicit stable identity, also overridden by drift tests.
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "a"*64)


def drives(state):
    return deepcopy((state.ac_deflector.scan_command_matrix_mrad,
                     state.descan_deflector.scan_command_matrix_mrad,
                     state.ac_deflector.calibration_mode,
                     state.ac_deflector.calibration_record_json))


@pytest.fixture
def view(qtbot, monkeypatch):
    monkeypatch.setattr(unified_scan_calibration, "measure_scan_responses", _drift_response)
    widget = ScanControlView()
    widget.set_state(_state())
    qtbot.addWidget(widget)
    errors, applied = [], []
    widget.error.connect(errors.append)
    widget.scan_calibration.applied.connect(applied.append)
    yield widget, errors, applied
    assert widget.shutdown()


@pytest.mark.parametrize("operation", ["automatic", "hold", "descan"])
def test_public_scan_actions_leave_qt_responsive_and_do_not_install_partial_drives(view, qtbot, monkeypatch, operation):
    widget, errors, applied = view
    state = widget._state
    if operation == "descan":
        calibrate_scan_system(state, force=True, hold=True)
        widget.set_state(state)
    before, host = drives(state), state.descan_deflector._physical_host
    entered, release = Event(), Event()
    worker_threads, beats = [], []

    def measure(candidate, observations, **kwargs):
        worker_threads.append(get_ident())
        entered.set()
        assert release.wait(10)
        return _drift_response(candidate, observations, **kwargs)

    monkeypatch.setattr(unified_scan_calibration, "measure_scan_responses", measure)
    timer = QTimer(widget)
    timer.setInterval(10)
    timer.timeout.connect(lambda: beats.append(True))
    timer.start()
    try:
        if operation == "automatic":
            widget.ac_controls["scan_pixel_size_nm"].setValue(.022)
            assert state.ac_deflector.scan_pixel_size_nm == .022
            assert state.descan_deflector.scan_pixel_size_nm == .022
            assert widget.component_fov_labels["ac"][0].text() == "1.408 nm"
        elif operation == "hold":
            widget.calibrate_hold_button.click()
        else:
            widget.calibrate_descan_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        qtbot.waitUntil(lambda: len(beats) >= 3, timeout=1000)
        assert widget.scan_calibration.busy and widget.cancel_calibration_button.isEnabled()
        assert worker_threads == [worker_threads[0]] and worker_threads[0] != get_ident()
        assert drives(state) == before
        assert not errors and not applied
    finally:
        release.set()
        timer.stop()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert not errors and applied == [operation]
    assert state.descan_deflector._physical_host is host
    assert not widget.cancel_calibration_button.isEnabled()


def test_later_lens_edit_discards_old_background_solution(view, qtbot, monkeypatch):
    widget, errors, applied = view
    state = widget._state
    entered, release = Event(), Event()
    before = drives(state)

    def solve(candidate, **kwargs):
        entered.set()
        assert release.wait(10)
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.123)
        candidate.ac_deflector.calibration_mode = "held"

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        state.objective_lens.percent += .1
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert drives(state) == before and not applied and not errors
    assert "discarded" in widget.calibration_status.text()


def test_latest_edit_wins_even_when_cancelled_solver_returns(view, qtbot, monkeypatch):
    widget, errors, applied = view
    first, release = Event(), Event()
    calls = []

    def solve(candidate, **kwargs):
        calls.append(candidate.ac_deflector.scan_pixel_size_nm)
        if len(calls) == 1:
            first.set()
            assert release.wait(10)
        candidate.ac_deflector.set_scan_command_matrix_mrad(
            np.eye(2)*candidate.ac_deflector.scan_pixel_size_nm)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.ac_controls["scan_pixel_size_nm"].setValue(.022)
        qtbot.waitUntil(first.is_set, timeout=10000)
        widget.ac_controls["scan_pixel_size_nm"].setValue(.024)
        assert widget._state.ac_deflector.scan_pixel_size_nm == .024
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    np.testing.assert_array_equal(widget._state.ac_deflector.scan_command_matrix_mrad, np.eye(2)*.024)
    assert calls == [.022, .024] and applied == ["automatic"] and not errors


@pytest.mark.parametrize("ending", ["failure", "cancel", "shutdown", "replace"])
def test_failed_cancelled_or_replaced_requests_never_install_partial_results(view, qtbot, monkeypatch, ending):
    widget, errors, applied = view
    state = widget._state
    before = drives(state)
    entered, release = Event(), Event()

    def solve(candidate, **kwargs):
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.321)
        candidate.ac_deflector.calibration_mode = "held"
        entered.set()
        assert release.wait(10)
        if ending == "failure":
            raise ValueError("Controlled calibration failure")

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if ending == "cancel":
            widget.cancel_calibration_button.click()
        elif ending == "shutdown":
            assert not widget.shutdown(timeout_ms=1)
        elif ending == "replace":
            widget.set_state(_state())
    finally:
        release.set()
    assert widget.scan_calibration.pool.waitForDone(10000)
    assert drives(state) == before and not applied
    assert bool(errors) == (ending == "failure")
    assert not widget.scan_calibration.busy


def test_page_deletion_cancels_owned_worker_without_queued_callbacks(qtbot, monkeypatch):
    widget = ScanControlView()
    state = _state()
    widget.set_state(state)
    before = drives(state)
    entered, release = Event(), Event()
    cancellation = []

    def solve(candidate, **kwargs):
        cancellation.append(candidate._tuning_cancelled)
        entered.set()
        assert release.wait(10)
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.456)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    pool = widget.scan_calibration.pool
    coordinator = pool.coordinator
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        assert cancellation[0]()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not coordinator.has_owner(pool), timeout=10000)
    assert drives(state) == before


def test_stem_request_waits_for_complete_calibration(view, qtbot, monkeypatch):
    widget, errors, applied = view
    entered, release = Event(), Event()
    requests = []
    widget.calculation_requested.connect(lambda: requests.append(drives(widget._state)))

    def solve(candidate, **kwargs):
        entered.set()
        assert release.wait(10)
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.222)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        widget.calculation_bar.requested.emit()
        assert not requests
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert len(requests) == 1 and applied == ["hold"] and not errors
    np.testing.assert_array_equal(requests[0][0], np.eye(2)*.222)


@pytest.mark.parametrize("disable_descan", [False, True])
def test_disabling_scan_cancels_automatic_frame_and_reenabling_requests_once(view, qtbot, monkeypatch, disable_descan):
    widget, errors, applied = view
    state = widget._state
    state.ac_deflector.scan_enabled = False
    widget.set_state(state)
    entered, release = Event(), Event()
    requests = []
    widget.calculation_requested.connect(lambda: requests.append(
        (state.ac_deflector.scan_enabled, state.descan_deflector.scan_enabled)))

    def measure(candidate, observations, **kwargs):
        entered.set()
        assert release.wait(10)
        return _drift_response(candidate, observations, **kwargs)

    monkeypatch.setattr(unified_scan_calibration, "measure_scan_responses", measure)
    try:
        widget.ac_controls["scan_enabled"].setChecked(True)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        widget.ac_controls["scan_enabled"].setChecked(False)
        if disable_descan:
            widget.descan_controls["scan_enabled"].setChecked(False)
        assert not requests
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert not requests and not errors
    assert applied == ["automatic"]

    widget.ac_controls["scan_enabled"].setChecked(True)
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert requests == [(True, not disable_descan)] and not errors
    assert applied == ["automatic", "automatic"]


@pytest.mark.parametrize("request_timing", ["before_disable", "after_disable"])
def test_explicit_calculate_survives_disabling_both_rasters(view, qtbot, monkeypatch, request_timing):
    widget, errors, applied = view
    state = widget._state
    state.ac_deflector.scan_enabled = False
    widget.set_state(state)
    entered, release = Event(), Event()
    requests = []
    widget.calculation_requested.connect(lambda: requests.append(
        (state.ac_deflector.scan_enabled, state.descan_deflector.scan_enabled)))

    def solve(candidate, **kwargs):
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.ac_controls["scan_enabled"].setChecked(True)
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if request_timing == "before_disable":
            widget.calculation_bar.requested.emit()
        widget.ac_controls["scan_enabled"].setChecked(False)
        widget.descan_controls["scan_enabled"].setChecked(False)
        if request_timing == "after_disable":
            widget.calculation_bar.requested.emit()
        assert not requests
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert requests == [(False, False)] and not errors
    assert applied == ["automatic"]


@pytest.mark.parametrize("disabled", ["ac_host", "ac_raster", "image_readout"])
def test_automatic_frame_rechecks_enabled_state_after_calibration_callbacks(view, qtbot, monkeypatch, disabled):
    widget, errors, applied = view
    state = widget._state
    state.ac_deflector.scan_enabled = False
    widget.set_state(state)
    requests = []
    widget.calculation_requested.connect(lambda: requests.append(True))
    monkeypatch.setattr(controller, "calibrate_scan_system", lambda candidate, **kwargs: None)

    def disable_after_commit(parameter):
        if parameter == "scan_calibration.updated":
            target, field = {
                "ac_host": (state.ac_deflector, "enabled"),
                "ac_raster": (state.ac_deflector, "scan_enabled"),
                "image_readout": (state.sample, "stem_image_enabled"),
            }[disabled]
            setattr(target, field, False)

    widget.parameters_changed.connect(disable_after_commit)
    widget.ac_controls["scan_enabled"].setChecked(True)
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert applied == ["automatic"] and not errors and not requests


@pytest.mark.parametrize("operation", ["automatic", "hold", "descan", "restore_ac"])
def test_enabling_image_readout_restarts_calibration_and_requests_one_frame(view, monkeypatch, operation):
    widget, errors, applied = view
    state = widget._state
    state.sample.stem_image_enabled = False
    widget.set_state(state)
    workers, requests = [], []
    monkeypatch.setattr(widget.scan_calibration.pool, "start", workers.append)
    widget.calculation_requested.connect(lambda: requests.append(drives(state)))
    widget._start_scan_calibration(operation)
    widget.image_enabled.setChecked(True)
    assert len(workers) == 2 and not requests
    old, latest = workers
    assert latest.operation == operation
    old_candidate = old.snapshot.restore()
    old_candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.123)
    widget.scan_calibration._accept_result(old.generation, old_candidate)
    assert widget.scan_calibration.busy and not applied and not requests
    candidate = latest.snapshot.restore()
    assert candidate.sample.stem_image_enabled
    candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.222)
    widget.scan_calibration._accept_result(latest.generation, candidate)
    assert not widget.scan_calibration.busy and not errors
    assert applied == [operation] and len(requests) == 1
    np.testing.assert_array_equal(requests[0][0], np.eye(2)*.222)


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("reenable", [False, True])
def test_image_readout_switch_preserves_only_requested_frames(view, monkeypatch, explicit, reenable):
    widget, errors, applied = view
    state = widget._state
    state.ac_deflector.scan_enabled = False
    widget.set_state(state)
    workers, requests = [], []
    monkeypatch.setattr(widget.scan_calibration.pool, "start", workers.append)
    widget.calculation_requested.connect(lambda: requests.append(state.sample.stem_image_enabled))
    widget.ac_controls["scan_enabled"].setChecked(True)
    if explicit:
        widget.calculation_bar.requested.emit()
    widget.image_enabled.setChecked(False)
    if reenable:
        widget.image_enabled.setChecked(True)
    for worker in workers:
        widget.scan_calibration._accept_result(worker.generation, worker.snapshot.restore())
    assert not widget.scan_calibration.busy and not errors
    assert applied == ["automatic"]
    assert requests == ([reenable] if explicit or reenable else [])


@pytest.mark.parametrize("changed", ["external", "implementation"])
def test_changed_external_content_or_solver_rejects_completed_worker(view, qtbot, monkeypatch, tmp_path, changed):
    widget, errors, applied = view
    source = tmp_path / "field-input.txt"
    source.write_text("original field data", encoding="utf-8")
    widget._state.lens_field_map_descriptors = {"condenser_lens_2": {"source_path": str(source)}}
    implementation = ["a"*64]
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: implementation[0])
    entered, release = Event(), Event()
    before = drives(widget._state)

    def solve(candidate, **kwargs):
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.123)
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        if changed == "external":
            source.write_text("replacement field data", encoding="utf-8")
        else:
            implementation[0] = "b"*64
    finally:
        release.set()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert errors and not applied and drives(widget._state) == before


@pytest.mark.parametrize("changed", ["external", "implementation"])
def test_commit_rechecks_dependencies_after_worker_completion(view, monkeypatch, tmp_path, changed):
    widget, errors, applied = view
    source = tmp_path / "field-input.txt"
    source.write_text("original", encoding="utf-8")
    widget._state.lens_field_map_descriptors = {"condenser_lens_2": {"source_path": str(source)}}
    workers = []
    monkeypatch.setattr(widget.scan_calibration.pool, "start", workers.append)
    widget.calibrate_hold_button.click()
    worker = workers[0]
    candidate = worker.snapshot.restore()
    candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.123)
    before = drives(widget._state)
    if changed == "external":
        source.write_text("replacement", encoding="utf-8")
    else:
        monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "b"*64)
    widget.scan_calibration._accept_result(worker.generation, candidate)
    assert not widget.scan_calibration.busy and not applied and not errors
    assert "discarded" in widget.calibration_status.text()
    assert drives(widget._state) == before


def test_page_remains_usable_when_parent_close_is_postponed(view, qtbot, monkeypatch):
    widget, errors, applied = view
    assert widget.shutdown()
    monkeypatch.setattr(controller, "calibrate_scan_system", lambda candidate, **kwargs: None)
    widget.calibrate_hold_button.click()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert applied == ["hold"] and not errors


def test_capture_failure_resets_calibration_status_and_retains_drives(view, monkeypatch):
    widget, errors, applied = view
    before = drives(widget._state)

    def fail_capture(state):
        raise ValueError("Controlled input capture failure")

    monkeypatch.setattr(controller, "capture_instrument_snapshot", fail_capture)
    widget.calibrate_hold_button.click()
    assert errors == ["Controlled input capture failure"] and not applied
    assert drives(widget._state) == before and not widget.scan_calibration.busy
    assert "could not start" in widget.calibration_status.text()
    assert not widget.cancel_calibration_button.isEnabled()


def test_install_failure_restores_all_scan_values_and_shared_host(view, qtbot, monkeypatch):
    widget, errors, applied = view
    state = widget._state
    before = controller.encode_instrument(state)
    host = state.descan_deflector._physical_host
    monkeypatch.setattr(controller, "calibrate_scan_system", lambda candidate, **kwargs: None)

    def fail_install(candidate, live):
        live.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.987)
        live.ac_deflector.calibration_mode = "held"
        live.descan_deflector.scan_pixels_x += 1
        raise ValueError("Controlled installation failure")

    monkeypatch.setattr(controller, "copy_resolved_scan_drives", fail_install)
    widget.calibrate_hold_button.click()
    qtbot.waitUntil(lambda: not widget.scan_calibration.busy, timeout=10000)
    assert errors == ["Controlled installation failure"] and not applied
    assert controller.encode_instrument(state) == before
    assert state.descan_deflector._physical_host is host


def test_main_window_close_waits_for_scan_cancellation(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation, instrument_configuration_dialog

    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    for module in (shell, interactive_calculation, instrument_configuration_dialog):
        monkeypatch.setattr(module, "QSettings", lambda: settings)
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    widget = window.workspace.scan_control
    widget.set_state(_state())
    before = drives(widget._state)
    entered, release = Event(), Event()
    cancellation, applied = [], []
    widget.scan_calibration.applied.connect(applied.append)

    def solve(candidate, **kwargs):
        cancellation.append(candidate._tuning_cancelled)
        candidate.ac_deflector.set_scan_command_matrix_mrad(np.eye(2)*.654)
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(controller, "calibrate_scan_system", solve)
    shutdown = widget.shutdown
    monkeypatch.setattr(widget, "shutdown", lambda: shutdown(timeout_ms=1))
    try:
        widget.calibrate_hold_button.click()
        qtbot.waitUntil(entered.is_set, timeout=10000)
        assert not window.close()
        assert cancellation[0]()
        assert "Waiting for owned calculations" in window.status_label.text()
        assert drives(widget._state) == before and not applied
    finally:
        release.set()
        assert widget.scan_calibration.pool.waitForDone(10000)
    assert window.close()
    assert drives(widget._state) == before and not applied
