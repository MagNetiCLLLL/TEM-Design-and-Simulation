"""Nominal preview restores the user's source sampling draft without editing physics."""
from test_coherent_beam_gui import panel, _enable_unified_numeric_controls
from test_coherent_result_labels import observation, preview_profile
from temsim.gui.coherent_result_labels import coherent_preview_text


def test_nominal_toggle_restores_energy_count_and_preserves_physical_spread(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.tip_energy.setValue(2.)
    panel.tip_energy_spread.setValue(.3)
    panel.energy_samples.setValue(9)
    original = panel._make_request()
    spread = panel.tip_energy_spread.value()
    panel.column_exploratory_preview.setChecked(True)
    assert not panel.source_nominal_energy_preview.isChecked()
    assert panel.source_nominal_energy_preview.isEnabled()
    panel._stale, panel._session_ready = False, True
    panel.source_nominal_energy_preview.setChecked(True)
    nominal = panel._make_request()
    assert nominal.source_nominal_energy_preview and nominal.source.energy_samples == 1
    assert not panel.energy_samples.isEnabled()
    assert panel._stale and not panel._session_ready
    assert panel.tip_energy_spread.value() == spread
    assert panel.tip_energy.value() == 2. and spread == .3
    panel._set_default_energy_samples(7)
    assert panel.energy_samples.value() == 1
    panel.source_nominal_energy_preview.setChecked(False)
    assert panel.energy_samples.value() == 9 and panel.energy_samples.isEnabled()
    assert not panel._make_request().source_nominal_energy_preview
    panel.source_nominal_energy_preview.setChecked(True)
    panel.column_exploratory_preview.setChecked(False)
    assert not panel.source_nominal_energy_preview.isChecked()
    assert not panel.source_nominal_energy_preview.isEnabled()
    assert panel._make_request() == original


def test_leaving_unified_mode_restores_the_nominal_source_draft(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.energy_samples.setValue(5)
    panel.column_exploratory_preview.setChecked(True)
    panel.source_nominal_energy_preview.setChecked(True)
    panel._state.simulation_mode = "ideal"
    panel.update_simulation_mode(panel._state)
    assert not panel._make_request().source_nominal_energy_preview
    assert panel.energy_samples.value() == 5 and panel.energy_samples.isEnabled()


def test_saved_nominal_preview_is_labeled_as_unresolved_energy_distribution():
    profile = {**preview_profile(), "source_nominal_energy_preview": True}
    label = coherent_preview_text(observation(profile))
    assert "Nominal emission energy (preview)" in label
    assert "configured FWHM not integrated" in label
    assert "not accuracy qualified" in label
