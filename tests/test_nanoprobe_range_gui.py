"""Nanoprobe range selection is pending until Apply and survives projection changes."""
from dataclasses import replace

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.assembly_panel import AssemblyPanel
from temsim.operating_modes import apply_operating_mode_pair, mode_by_key
from temsim.optics.column import default_state


NORMAL = "nano_probe"
LARGE = "nano_probe_large_angle"


@pytest.fixture
def panel(qtbot):
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    state = default_state()
    catalog.apply(state, selection)
    apply_operating_mode_pair(state, NORMAL, "diffraction")
    widget = AssemblyPanel(catalog, selection)
    widget.set_direct_alignment_state(state)
    qtbot.addWidget(widget)
    return widget, state


def select_range(panel, key):
    index = panel.nanoprobe_range_combo.findData(key)
    assert index >= 0
    panel.nanoprobe_range_combo.setCurrentIndex(index)


def test_range_selector_keeps_two_probe_modes_and_only_emits_on_apply(panel):
    widget, state = panel
    assert [widget.probe_mode.itemText(i) for i in range(widget.probe_mode.count())] == [
        "Microprobe", "Nanoprobe"]
    assert [widget.nanoprobe_range_combo.itemText(i)
            for i in range(widget.nanoprobe_range_combo.count())] == ["Normal", "Large angle"]
    assert widget.selected_condenser_mode_key() == NORMAL
    assert not widget.nanoprobe_range_combo.isHidden()
    requests = []
    widget.operating_mode_requested.connect(lambda *args: requests.append(args))
    before = [(lens.key, lens.percent) for lens in state.lenses]
    select_range(widget, LARGE)
    assert state.nanoprobe_range == "normal"
    assert [(lens.key, lens.percent) for lens in state.lenses] == before
    assert not requests
    widget.apply_operating_mode_button.click()
    assert requests == [(LARGE, "diffraction")]
    assert "30 mrad" in widget.nanoprobe_range_combo.toolTip()
    assert "Large angle" in widget.operating_mode_status.text()


def test_pending_range_disables_every_alignment_and_reverting_restores_it(panel):
    widget, _state = panel
    alignment = widget.direct_alignment_panel
    nano = alignment.controls["nanoprobe_convergence"]
    assert nano.apply_button.isEnabled()
    assert alignment.beam_alignment_editor.apply_button.isEnabled()
    assert alignment.stigmator_alignment_editor.apply_button.isEnabled()
    select_range(widget, LARGE)
    assert "Selection changed" in alignment.mode_status.text()
    assert all(not control.apply_button.isEnabled() for control in alignment.controls.values())
    assert not alignment.beam_alignment_editor.apply_button.isEnabled()
    assert not alignment.stigmator_alignment_editor.apply_button.isEnabled()
    select_range(widget, NORMAL)
    assert "Selection changed" not in alignment.mode_status.text()
    assert nano.apply_button.isEnabled()
    assert alignment.beam_alignment_editor.apply_button.isEnabled()
    assert alignment.stigmator_alignment_editor.apply_button.isEnabled()


def test_applied_large_range_uses_shared_alignment_and_preserves_busy_gating(panel):
    widget, state = panel
    apply_operating_mode_pair(state, LARGE, "diffraction")
    widget.set_operating_mode_keys(LARGE, "diffraction")
    widget.set_direct_alignment_state(state)
    alignment = widget.direct_alignment_panel
    assert "Nanoprobe (Large angle)" in alignment.mode_status.text()
    assert "Selection changed" not in alignment.mode_status.text()
    assert {key for key, control in alignment.controls.items()
            if not control.group.isHidden()} == {"nanoprobe_convergence", "diffraction_camera_length"}
    assert alignment.controls["nanoprobe_convergence"].apply_button.isEnabled()
    assert alignment.beam_alignment_editor.apply_button.isEnabled()
    assert alignment.stigmator_alignment_editor.apply_button.isEnabled()
    alignment.set_busy("nanoprobe_convergence")
    select_range(widget, NORMAL)
    select_range(widget, LARGE)
    assert not alignment.controls["nanoprobe_convergence"].apply_button.isEnabled()
    assert not alignment.beam_alignment_editor.apply_button.isEnabled()
    alignment.set_busy(None)
    assert alignment.controls["nanoprobe_convergence"].apply_button.isEnabled()
    assert alignment.beam_alignment_editor.apply_button.isEnabled()


def test_range_visibility_and_catalog_reload_preserve_pending_branch(panel):
    widget, _state = panel
    select_range(widget, LARGE)
    widget.probe_mode.setCurrentIndex(widget.probe_mode.findData("micro_probe"))
    assert widget.nanoprobe_range_combo.isHidden()
    assert widget.nanoprobe_range_label.isHidden()
    assert widget.selected_condenser_mode_key() == "micro_probe"
    widget.probe_mode.setCurrentIndex(widget.probe_mode.findData(NORMAL))
    assert not widget.nanoprobe_range_combo.isHidden()
    assert widget.selected_condenser_mode_key() == LARGE
    widget.reload_catalog(AssemblyCatalog(), widget.current_selection())
    assert widget.selected_condenser_mode_key() == LARGE
    unsupported = replace(widget.current_selection(), column="C3")
    widget.reload_catalog(AssemblyCatalog(), unsupported)
    assert widget.probe_mode.count() == widget.nanoprobe_range_combo.count() == 0
    assert widget.nanoprobe_range_combo.isHidden()
    assert not widget.apply_operating_mode_button.isEnabled()


@pytest.mark.parametrize("key, value", [(NORMAL, "normal"), (LARGE, "large_angle")])
def test_reloaded_state_restores_the_full_selected_preset(panel, key, value):
    from temsim.gui.main_window import MainWindow
    widget, state = panel
    apply_operating_mode_pair(state, key, "imaging")
    restored = type(state).from_dict(state.to_dict())
    assert restored.nanoprobe_range == value
    selected = MainWindow._state_operating_mode_keys(restored)
    assert selected == (key, "imaging")
    widget.load_operating_modes(widget.current_selection(), *selected)
    widget.set_direct_alignment_state(restored)
    assert widget.selected_condenser_mode_key() == key
    assert "Selection changed" not in widget.direct_alignment_panel.mode_status.text()


def test_microprobe_state_restores_remembered_nanoprobe_range(panel):
    from temsim.gui.main_window import MainWindow
    widget, state = panel
    apply_operating_mode_pair(state, LARGE, "diffraction")
    apply_operating_mode_pair(state, "micro_probe", "diffraction")
    restored = type(state).from_dict(state.to_dict())
    assert restored.nanoprobe_range == "large_angle"
    widget.load_operating_modes(widget.current_selection(),
        *MainWindow._state_operating_mode_keys(restored), nanoprobe_range=restored.nanoprobe_range)
    assert widget.nanoprobe_range_combo.isHidden()
    widget.probe_mode.setCurrentIndex(widget.probe_mode.findData(NORMAL))
    assert widget.selected_condenser_mode_key() == LARGE


def test_window_applies_large_range_then_switches_only_projector(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation, calculation_controller
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(calculation_controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "schedule_preview", lambda *_args, **_kwargs: None)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    # This regression executes the legacy image-projector calibration below.
    # Unified finite-field optics deliberately rejects that calibration route.
    window.set_simulation_mode("custom")
    widget = window.assembly_panel
    select_range(widget, LARGE)
    assert window.state.nanoprobe_range == "normal"
    widget.apply_operating_mode_button.click()
    assert not errors
    assert window.state.nanoprobe_range == "large_angle"
    assert widget.selected_condenser_mode_key() == LARGE
    c3 = next(lens for lens in window.state.lenses if lens.key == "condenser_lens_3")
    assert c3.percent == mode_by_key(LARGE).devices[c3.key]["percent"]
    c3.percent += 0.1
    before = [(lens.key, lens.percent) for lens in window.state.lenses
              if lens.key not in {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}]
    window.state.nanopulser.installed = True
    from temsim.gui import operating_preset_controller as worker
    monkeypatch.setattr(worker, "apply_operating_mode_pair", lambda *_a, **_k:
                        pytest.fail("Projection-only change must not solve a condenser preset"))
    monkeypatch.setattr(shell, "apply_operating_mode_pair", lambda *_a, **_k:
                        pytest.fail("Projection-only change must preserve the Large angle branch"))
    monkeypatch.setattr(shell, "apply_physical_layout_to_state", lambda *_a, **_k:
                        pytest.fail("Projection-only change must not rebuild upstream optics"))
    widget.projector_mode.setCurrentIndex(widget.projector_mode.findData("imaging"))
    widget.apply_operating_mode_button.click()
    qtbot.waitUntil(lambda: window._preset_state_token is None, timeout=90_000)
    assert not errors
    assert window.state.nanoprobe_range == "large_angle"
    assert window.state.projector_mode == "image"
    assert widget.selected_condenser_mode_key() == LARGE
    assert [(lens.key, lens.percent) for lens in window.state.lenses
            if lens.key not in {"diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}] == before
    window.close()
