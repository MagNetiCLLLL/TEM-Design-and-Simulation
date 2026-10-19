"""Projector-only switching retains the calibrated incident working point."""
from dataclasses import replace

import pytest

from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.operating_modes import (
    apply_projector_mode, load_operating_mode_catalog, mode_by_key,
)
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets


def _working_point(state):
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    targets = runtime_targets(state)
    for key, value in (("condenser_lens_1", 43.25), ("condenser_lens_2", 29.5),
                       ("condenser_lens_3", 32.1)):
        targets[key].obj.percent = value
    state.mini_condenser.signed_excitation_percent = -37.25
    state.objective_lens.percent = 54.3
    state.electron_gun.electrostatic_lens.voltage_kv = 1.7
    state.fluorescent_screen.inserted = True
    state.camera.inserted = False
    for detector in state.stem_detectors:
        detector.inserted = detector.readout_enabled = False
    ac = state.ac_deflector
    ac.set_pure_shift_coupling(((-1.2, .1), (-.2, -1.3)))
    ac.set_scan_command_matrix_mrad(((.001, .0002), (-.0003, .002)))
    from temsim.physics.scan_calibration import capture_record
    capture_record(state, descan_calibrated=False)
    ac.calibration_mode = "held"
    state.descan_deflector.scan_enabled = False
    state.illumination_mode = "STEM"
    state.projector_mode = "diffraction"
    return state


def _nonprojector(state):
    # Compare the entire encoded graph after normalising only the explicitly
    # permitted projector fields. This catches hidden source/drive resets.
    copied = decode_instrument(encode_instrument(state))
    copied.projector_mode = "diffraction"
    targets = runtime_targets(copied)
    for key in ("diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"):
        targets[key].obj.percent = 0.
    return json_digest(encode_instrument(copied))


def test_projector_roundtrip_preserves_complete_nonprojector_graph():
    state = _working_point(default_state())
    before = _nonprojector(state)
    record = state.ac_deflector.calibration_record_json
    for key in ("imaging", "diffraction"):
        applied = apply_projector_mode(state, key)
        assert _nonprojector(state) == before
        assert state.ac_deflector.calibration_record_json == record
        assert state.ac_deflector.calibration_mode == "held"
        assert set(applied.changed_devices) == set(mode_by_key(key).devices)
        if key == "imaging":
            from temsim.optics.direct_alignment import diffraction_transfers
            from temsim.physics.scan_geometry import classify_sample_plane_transfer
            z = float(state.fluorescent_screen.z_mm)
            validation = decode_instrument(encode_instrument(state))
            validation.step_mm = .025
            validation.acceleration_enabled = True
            validation.acceleration_backend = "Numba CPU"
            assert classify_sample_plane_transfer(diffraction_transfers(validation, [z])[z])[0] == "image"
            assert applied.image_projection is not None
        else:
            for lens_key, values in mode_by_key(key).devices.items():
                assert runtime_targets(state)[lens_key].obj.percent == values["percent"]
            assert "unqualified" in applied.summary and "no image focus" in applied.summary
            assert "conjugate residual" not in applied.summary


@pytest.mark.parametrize("failure", ("upstream", "invalid_strength", "wrong_family", "incompatible"))
def test_projector_switch_rejects_invalid_definition_atomically(failure):
    state = _working_point(default_state())
    before = json_digest(encode_instrument(state))
    catalog = load_operating_mode_catalog()
    mode = mode_by_key("imaging", catalog)
    options = {}
    if failure == "upstream":
        mode = replace(mode, devices={**mode.devices, "objective_lens": {"percent": 1.}})
    elif failure == "invalid_strength":
        mode = replace(mode, devices={**mode.devices, "projector_lens_2": {"percent": -1.}})
    elif failure == "wrong_family":
        mode = replace(mode, family="condenser")
    else:
        options["recording_name"] = "unavailable recording system"
    catalog = replace(catalog, modes=tuple(mode if row.key == mode.key else row for row in catalog.modes))
    with pytest.raises(ValueError):
        apply_projector_mode(state, "imaging", catalog=catalog, **options)
    assert json_digest(encode_instrument(state)) == before


@pytest.fixture
def window(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation as gui, calculation_controller as controller
    settings = QSettings(str(tmp_path/"workspace.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(gui, "QSettings", lambda: settings)
    monkeypatch.setattr(controller, "default_artifact_cache_root", lambda: tmp_path/"artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    # These cases exercise asynchronous publication and GUI invalidation. The
    # physical focus/residual is covered independently in test_image_receiver_focus.
    from temsim.optics.projector_reference import ImageProjection
    def focus(candidate):
        candidate.diffraction_lens.percent += .1
        target = candidate.fluorescent_screen
        return ImageProjection(target.key, target.name, target.z_mm, target.inserted,
                               23., 23., 1.e-9, .025)
    monkeypatch.setattr("temsim.optics.projector_reference.focus_image_projector", focus)
    widget = shell.MainWindow()
    qtbot.addWidget(widget)
    widget.preview_timer.stop()
    yield widget
    widget.preview_timer.stop()
    widget.close()


def test_gui_projector_switch_preserves_working_point_and_invalidates_pair(window, monkeypatch, qtbot):
    from temsim.gui import main_window as shell
    from temsim.gui import operating_preset_controller as worker
    state = _working_point(window.state)
    before = _nonprojector(state)
    page = window.workspace.coherent_beam
    page._stale = False
    page._session_ready = True
    page._observation_session = object()
    page._cache["old-wave"] = object()
    page._pair_context = object()
    invalidated = []
    page.pair_invalidated.connect(lambda: invalidated.append(True))
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(shell, "apply_operating_mode_pair", lambda *_a, **_k:
        pytest.fail("Projector-only selection must not apply the incident preset"))
    monkeypatch.setattr(shell, "apply_physical_layout_to_state", lambda *_a, **_k:
        pytest.fail("Projector-only switch does not change Objective or topology"))
    monkeypatch.setattr(worker, "apply_operating_mode_pair", lambda *_a, **_k:
        pytest.fail("Projector-only worker must not apply the incident preset"))
    monkeypatch.setattr(worker, "apply_physical_layout_to_state", lambda *_a, **_k:
        pytest.fail("Projector-only worker must preserve the captured geometry"))
    window.apply_operating_modes("nano_probe", "imaging")
    assert window.state is state and state.projector_mode == "diffraction"
    qtbot.waitUntil(lambda: window._preset_state_token is None, timeout=90_000)
    window.preview_timer.stop()
    assert not errors
    assert window.state.projector_mode == "image"
    assert _nonprojector(window.state) == before
    assert invalidated and page._pair_context is None
    assert page._stale and not page._session_ready
    assert page._observation_session is None and not page._cache
    status = window.assembly_panel.operating_mode_status.text()
    assert "first-order" in status.lower() and "residual" in status
    assert "same selection again" in window.assembly_panel.apply_operating_mode_button.toolTip()


def test_gui_reapplying_same_keys_still_reapplies_complete_presets(window, monkeypatch):
    state = _working_point(window.state)
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window.apply_operating_modes("nano_probe", "diffraction")
    window.preview_timer.stop()
    assert not errors
    assert runtime_targets(state)["condenser_lens_3"].obj.percent == mode_by_key("nano_probe").devices["condenser_lens_3"]["percent"]
    assert state.objective_lens.percent == mode_by_key("nano_probe").devices["objective_lens"]["percent"]
    assert state.electron_gun.electrostatic_lens.voltage_kv == 1.2
    assert not state.fluorescent_screen.inserted


def test_gui_projector_switch_does_not_run_blankers_condenser_solve(window, monkeypatch, qtbot):
    from temsim.gui import operating_preset_controller as worker
    state = _working_point(window.state)
    state.nanopulser.installed = True
    monkeypatch.setattr(worker, "apply_operating_mode_pair", lambda *_a, **_k:
        pytest.fail("Downstream switch must not start a condenser solve"))
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    window.apply_operating_modes("nano_probe", "imaging")
    qtbot.waitUntil(lambda: window._preset_state_token is None, timeout=90_000)
    window.preview_timer.stop()
    assert not errors and window.state.projector_mode == "image"
    assert window.state.objective_lens.percent == 54.3


def test_late_image_solve_cannot_replace_a_newer_captured_source_edit(window, monkeypatch):
    state = _working_point(window.state)
    submitted = []
    monkeypatch.setattr(window.operating_presets, "submit", lambda *a, **kw: submitted.append(kw))
    window.apply_operating_modes("nano_probe", "imaging")
    assert submitted[0]["projector_only"]
    candidate = decode_instrument(encode_instrument(state))
    candidate.projector_mode = "image"
    # A complete graph identity also covers model fields omitted by the legacy
    # operating-profile serializer; do not publish a result over such an edit.
    state.electron_gun._source_edit_during_projection = 0.5
    window._operating_preset_ready(candidate, window.selection, None, .01)
    window._operating_preset_finished()
    assert window.state is state
    assert state.projector_mode == "diffraction"
    assert "Discarded optical preset" in window.log_output.toPlainText()


def test_explicit_image_refocus_retains_inserted_screen_and_illumination(window, monkeypatch):
    state = _working_point(window.state)
    state.projector_mode = "image"
    window.assembly_panel.set_operating_mode_keys("nano_probe", "imaging")
    window.assembly_panel.set_direct_alignment_state(state)
    before = json_digest(encode_instrument(state))
    submissions = []
    monkeypatch.setattr(window.operating_presets, "submit", lambda *a, **kw: submissions.append((a, kw)))
    assert window.assembly_panel.focus_image_button.isEnabled()
    window.assembly_panel.focus_image_button.click()
    assert submissions[0][1]["projector_only"]
    assert submissions[0][0][3:5] == ("nano_probe", "imaging")
    assert json_digest(encode_instrument(state)) == before
    assert not window.assembly_panel.focus_image_button.isEnabled()
    window._operating_preset_finished()
    assert window.assembly_panel.focus_image_button.isEnabled()
