"""The B-COR extension shares incident presets without claiming projector calibration."""
from dataclasses import replace
from pathlib import Path
import tomllib

import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.gui.assembly_panel import AssemblyPanel
from temsim.operating_modes import apply_condenser_mode, load_mini_condenser_default_excitation_percent
from temsim.optics.column import default_state


COMBINED = "C3 + Probe Corrector + B-COR"
RANGES = (("nano_probe", "normal"), ("nano_probe_large_angle", "large_angle"))
ROOT = Path(__file__).resolve().parents[1]


def combined_state():
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), column=COMBINED)
    state = default_state()
    catalog.apply(state, selection)
    return catalog, selection, state


def test_combined_column_inherits_calibrated_incident_geometry_and_startup():
    def document(name):
        return tomllib.loads((ROOT / "configs/instruments/column" / name).read_text(encoding="utf-8"))

    baseline = document("C3_ProbeCorrector.toml")
    combined = document("C3_ProbeCorrector_BCOR.toml")
    original, extended = ({part["key"]: part for part in doc["parts"]}
                          for doc in (baseline, combined))
    # Display ordering/names and downstream provenance do not change fields,
    # stops, optical references, material dimensions or installation geometry.
    cosmetic = {"name", "order", "bcor_geometry_source", "bcor_geometry_status"}
    for key, part in original.items():
        assert {k: v for k, v in part.items() if k not in cosmetic} == {
            k: v for k, v in extended[key].items() if k not in cosmetic}, key
    sample_exit = original["sample"]["local_end_z_mm"]
    assert all(part["local_start_z_mm"] > sample_exit
               for key, part in extended.items() if key not in original)
    assert baseline["ports"]["entrance"] == combined["ports"]["entrance"]
    assert baseline["subassemblies"] == combined["subassemblies"]
    for key, value in baseline["geometry"].items():
        if key in {"length_mm", "parts_end_z_mm", "magnetic_lens_design_peak_fields_t"}:
            continue
        assert combined["geometry"][key] == value, key
    for key, value in baseline["geometry"]["magnetic_lens_design_peak_fields_t"].items():
        assert combined["geometry"]["magnetic_lens_design_peak_fields_t"][key] == value
    for module in ("column/C3_ProbeCorrector.toml", "column/C3_ProbeCorrector_BCOR.toml"):
        assert load_mini_condenser_default_excitation_percent(module) == pytest.approx(43.88646617222637)
    _, _, state = combined_state()
    assert abs(state.mini_condenser.signed_excitation_percent) == pytest.approx(43.88646617222637)


@pytest.mark.parametrize("key,selected", RANGES)
def test_combined_panel_emits_only_incident_preset_and_keeps_alignment_gate(qtbot, key, selected):
    catalog, selection, state = combined_state()
    apply_condenser_mode(state, "nano_probe", column_name=selection.column,
                         recording_name=selection.recording)
    widget = AssemblyPanel(catalog, selection)
    qtbot.addWidget(widget)
    widget.set_direct_alignment_state(state)
    assert widget.projector_mode.count() == 0
    assert widget.apply_operating_mode_button.isEnabled()
    assert widget.apply_operating_mode_button.text() == "Apply illumination preset"
    requests, paired = [], []
    widget.condenser_mode_requested.connect(requests.append)
    widget.operating_mode_requested.connect(lambda *args: paired.append(args))
    widget.nanoprobe_range_combo.setCurrentIndex(widget.nanoprobe_range_combo.findData(key))
    assert state.nanoprobe_range == "normal"
    control = widget.direct_alignment_panel.controls["nanoprobe_convergence"]
    assert control.apply_button.isEnabled() == (selected == "normal")
    widget.apply_operating_mode_button.click()
    assert requests == [key] and not paired
    assert "current projection retained" in widget.operating_mode_status.text()
    assert state.image_corrector_design == "bcor"


@pytest.mark.parametrize("key,selected", RANGES)
def test_background_worker_accepts_incident_only_preset(qtbot, key, selected):
    from temsim.gui.operating_preset_controller import _Worker

    catalog, selection, state = combined_state()
    state.projector_mode = "image"
    before = {lens.key: lens.percent for lens in state.lenses
              if lens.key.startswith("image_") or lens.key in {
                  "diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}}
    worker = _Worker(1, type(state), state.to_dict(), catalog, selection, key, None, False)
    results, errors = [], []
    worker.signals.result.connect(lambda *args: results.append(args))
    worker.signals.error.connect(lambda *args: errors.append(args))
    worker.run()
    assert not errors and len(results) == 1
    restored, result = results[0][1], results[0][3]
    assert restored.nanoprobe_range == selected
    assert restored.image_corrector_design == "bcor"
    assert restored.projector_mode == "image" and result.projector is None
    assert {lens.key: lens.percent for lens in restored.lenses if lens.key in before} == before


def test_window_routes_combined_illumination_without_projector_preset(qtbot, monkeypatch, tmp_path):
    from PySide6.QtCore import QSettings
    from temsim.gui import main_window as shell, interactive_calculation, calculation_controller

    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    settings.setFallbacksEnabled(False)
    monkeypatch.setattr(shell, "QSettings", lambda: settings)
    monkeypatch.setattr(interactive_calculation, "QSettings", lambda: settings)
    monkeypatch.setattr(calculation_controller, "default_artifact_cache_root", lambda: tmp_path / "artifacts")
    monkeypatch.setattr(shell.MainWindow, "INITIAL_PREVIEW_DELAY_MS", 60000)
    monkeypatch.setattr(shell.MainWindow, "schedule_preview", lambda _self: None)
    window = shell.MainWindow()
    qtbot.addWidget(window)
    window.preview_timer.stop()
    selection = replace(window.selection, column=COMBINED)
    window.catalog.apply(window.state, selection, preserve_operating_parameters=True)
    window.selection = selection
    window.assembly = window.state._resolved_assembly
    window.assembly_panel.set_selection(selection)
    window._refresh_assembly_views()
    before = {lens.key: lens.percent for lens in window.state.lenses
              if lens.key.startswith("image_") or lens.key in {
                  "diffraction_lens", "intermediate_lens", "projector_lens_1", "projector_lens_2"}}
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(shell, "apply_operating_mode_pair", lambda *_a, **_k:
                        pytest.fail("The combined assembly must not apply an unqualified projector preset"))
    widget = window.assembly_panel
    for key, selected in reversed(RANGES):
        widget.nanoprobe_range_combo.setCurrentIndex(widget.nanoprobe_range_combo.findData(key))
        widget.apply_operating_mode_button.click()
        assert not errors
        assert window.state.nanoprobe_range == selected
        assert window.state.image_corrector_design == "bcor"
        assert {lens.key: lens.percent for lens in window.state.lenses if lens.key in before} == before
    window.close()
