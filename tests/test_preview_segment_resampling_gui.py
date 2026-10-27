"""Offscreen explicit preview switching; no propagation is asserted here."""
from dataclasses import replace

import pytest

from test_coherent_beam_gui import panel, _enable_unified_numeric_controls


def test_preview_preset_enables_optional_segment_start_and_restores_strict_draft(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    original = panel._make_request()
    assert not original.column_preview_segment_resampling
    assert not panel.column_preview_segment_resampling.isEnabled()
    panel.column_exploratory_preview.setChecked(True)
    request = panel._make_request()
    assert request.column_preview_segment_resampling
    assert panel.column_preview_segment_resampling.isEnabled()
    assert request.wave_grid == original.wave_grid
    panel._stale = False
    panel._session_ready = True
    panel.column_preview_segment_resampling.setChecked(False)
    edited = panel._make_request()
    assert edited == replace(request, column_preview_segment_resampling=False)
    assert panel._stale and not panel._session_ready
    panel.column_exploratory_preview.setChecked(False)
    assert panel._make_request() == original
    assert not panel.column_preview_segment_resampling.isChecked()
    assert not panel.column_preview_segment_resampling.isEnabled()


def test_inactive_segment_draft_cannot_enter_compatibility_request(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_exploratory_preview.setChecked(True)
    panel.column_weighted_resampling.setChecked(False)
    assert panel.column_preview_segment_resampling.isChecked()
    assert not panel.column_preview_segment_resampling.isEnabled()
    with pytest.raises(ValueError, match="requires controlled support"):
        panel._make_request()
    panel.column_weighted_resampling.setChecked(True)
    assert panel._make_request().column_preview_segment_resampling
    panel._state.simulation_mode = "ideal"
    panel.update_simulation_mode(panel._state)
    assert not panel._make_request().column_preview_segment_resampling
    assert not panel.column_preview_segment_resampling.isChecked()

