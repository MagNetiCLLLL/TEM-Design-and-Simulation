"""Offscreen explicit phase-preparation control and honest estimate labels."""
from dataclasses import replace

from test_coherent_beam_gui import panel, _enable_unified_numeric_controls
from test_coherent_result_labels import observation, preview_profile
from temsim.gui.coherent_result_labels import coherent_preview_text


def test_phase_preparation_is_independent_opt_in_and_off_restores_original(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    original = panel._make_request()
    assert not panel.column_preview_phase_reselection.isChecked()
    assert not panel.column_preview_phase_reselection.isEnabled()
    panel.column_exploratory_preview.setChecked(True)
    preview = panel._make_request()
    assert not preview.column_preview_phase_reselection
    assert panel.column_preview_phase_reselection.isEnabled()
    panel._stale, panel._session_ready = False, True
    panel.column_preview_phase_reselection.setChecked(True)
    assert panel._make_request() == replace(preview, column_preview_phase_reselection=True)
    assert panel._stale and not panel._session_ready
    assert "not a certified bound" in panel.column_preview_phase_reselection.toolTip()
    panel.column_exploratory_preview.setChecked(False)
    assert panel._make_request() == original
    assert not panel.column_preview_phase_reselection.isChecked()
    assert not panel.column_preview_phase_reselection.isEnabled()


def test_inactive_phase_draft_cannot_enter_request_and_mode_change_clears_it(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_exploratory_preview.setChecked(True)
    panel.column_preview_phase_reselection.setChecked(True)
    panel.column_preview_segment_resampling.setChecked(False)
    assert panel.column_preview_phase_reselection.isChecked()
    assert not panel.column_preview_phase_reselection.isEnabled()
    assert not panel._make_request().column_preview_phase_reselection
    panel.column_preview_segment_resampling.setChecked(True)
    assert panel._make_request().column_preview_phase_reselection
    panel._state.simulation_mode = "ideal"
    panel.update_simulation_mode(panel._state)
    assert not panel._make_request().column_preview_phase_reselection
    assert not panel.column_preview_phase_reselection.isChecked()


def test_result_label_separates_estimated_representation_error_from_projection_bound():
    original = preview_profile()
    assert "representation" not in coherent_preview_text(observation(original))
    profile = {**original, "column_preview_phase_reselection": True,
        "phase_preparation_maximum_pixels": 4096,
        "estimated_representation_l2_per_reference": .0002,
        "combined_projection_and_estimated_l2_per_reference": .0012,
        "representation_estimates_certified": False}
    label = coherent_preview_text(observation(profile))
    for text in ("not accuracy qualified", "Column cap 2048", "CPU phase preparation cap 4096",
                 "projection L2 cost 0.001", "representation L2 estimate 0.0002",
                 "combined L2 cost 0.0012", "estimated, not a certified bound"):
        assert text in label
