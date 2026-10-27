"""Offscreen control transitions; no physical propagation is substituted as evidence."""
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from test_coherent_beam_gui import panel, calculate, _enable_unified_numeric_controls


def test_weighted_resampling_default_stays_disabled_and_preserves_existing_numerics(panel, monkeypatch):
    default = TipWaveRequest()
    assert default.column_weighted_resampling is False
    assert not panel.column_weighted_resampling.isChecked()
    assert not panel.column_weighted_resampling.isEnabled()
    baseline = panel._make_request()
    assert baseline.column_step_mm == .5
    _enable_unified_numeric_controls(panel, monkeypatch)
    assert not panel.column_weighted_resampling.isEnabled()
    assert panel._make_request() == baseline
    panel.column_support_projection.setChecked(True)
    assert panel.column_weighted_resampling.isEnabled()
    assert not panel._make_request().column_weighted_resampling
    assert panel.column_kinetic_order.currentData() == 4
    assert panel.column_exponential_method.currentData() == "taylor"
    assert panel.support_l2_budget.text() == repr(default.wave_grid.maximum_accumulated_support_l2)
    assert panel.support_probability_budget.text() == repr(default.wave_grid.maximum_support_loss_fraction)


def test_weighted_resampling_updates_request_and_invalidates_previous_result_only(panel, qtbot, monkeypatch):
    calculate(panel, qtbot)
    previous = panel.result
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_support_projection.setChecked(True)
    baseline = panel._make_request()
    physical = deepcopy(panel._state)
    panel._stale = False
    panel._session_ready = True
    panel.column_weighted_resampling.setChecked(True)
    request = panel._make_request()
    assert request.column_weighted_resampling is True
    assert replace(request, column_weighted_resampling=False) == baseline
    assert json_digest(asdict(request)) != json_digest(asdict(baseline))
    assert panel._state == physical
    assert panel.result is previous and panel._stale and not panel._session_ready
    assert panel._worker is None and panel.pool.activeThreadCount() == 0
    assert "Previous optics remain displayed" in panel.status.text()


def test_support_toggle_preserves_weighted_draft_but_cannot_submit_without_support(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_support_projection.setChecked(True)
    panel.column_weighted_resampling.setChecked(True)
    selected = panel._make_request()
    panel.column_support_projection.setChecked(False)
    assert panel.column_weighted_resampling.isChecked()
    assert not panel.column_weighted_resampling.isEnabled()
    disabled = panel._make_request()
    assert not disabled.column_support_projection and not disabled.column_weighted_resampling
    assert disabled == replace(selected, column_support_projection=False, column_weighted_resampling=False)
    panel.column_support_projection.setChecked(True)
    assert panel.column_weighted_resampling.isEnabled()
    assert panel._make_request() == selected


def test_execution_mode_switch_keeps_weighted_draft_without_running_it_in_legacy_modes(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_support_projection.setChecked(True)
    panel.column_weighted_resampling.setChecked(True)
    selected = panel._make_request()
    for mode in ("ideal", "custom", "analytical", "ideal_thin"):
        panel._state.simulation_mode = mode
        panel.update_simulation_mode(panel._state)
        assert panel.column_weighted_resampling.isChecked()
        assert not panel.column_weighted_resampling.isEnabled()
        if mode == "ideal_thin":
            with pytest.raises(ValueError, match="unavailable in Ideal Optics"):
                panel._make_request()
        else:
            assert panel._make_request() == replace(selected, column_weighted_resampling=False)
    _enable_unified_numeric_controls(panel, monkeypatch)
    assert panel.column_weighted_resampling.isEnabled()
    assert panel._make_request() == selected


def test_zero_budgets_remain_explicit_and_tooltip_describes_complex_wave_limits(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_support_projection.setChecked(True)
    panel.column_weighted_resampling.setChecked(True)
    panel.support_l2_budget.setText("0")
    panel.support_probability_budget.setText("0")
    request = panel._make_request()
    assert request.column_weighted_resampling
    assert request.wave_grid.maximum_accumulated_support_l2 == 0.
    assert request.wave_grid.maximum_support_loss_fraction == 0.
    note = panel.column_weighted_resampling.toolTip()
    for phrase in ("declared resource limit", "largest affordable grid",
                   "amplitudes and phases are preserved", "lost intensity is not restored",
                   "does not itself trigger downsampling", "cumulative source budgets", "Gaussian tails",
                   "separately from physical absorption", "phase convergence"):
        assert phrase in note
    assert panel.column_weighted_resampling.objectName() == "coherent_column_weighted_resampling"


@pytest.mark.parametrize("value", [1, 0, "true", None, np.bool_(True)])
def test_request_rejects_nonboolean_weighted_resampling(value):
    with pytest.raises(ValueError, match="explicit boolean"):
        replace(TipWaveRequest(), column_support_projection=True, column_weighted_resampling=value).validate()


def test_request_requires_controlled_support_for_weighted_resampling():
    with pytest.raises(ValueError, match="requires controlled support"):
        replace(TipWaveRequest(), column_weighted_resampling=True).validate()
    replace(TipWaveRequest(), column_support_projection=True, column_weighted_resampling=True).validate()


def test_exploratory_preview_is_explicit_adjustable_and_restores_previous_strict_settings(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.support_l2_budget.setText("0.015")
    panel.support_probability_budget.setText("5e-7")
    panel.column_step.setValue(.625)
    panel.column_kinetic_order.setCurrentIndex(panel.column_kinetic_order.findData(8))
    baseline = panel._make_request()
    physical = deepcopy(panel._state)
    panel.column_exploratory_preview.setChecked(True)
    preview = panel._make_request()
    assert preview.column_exploratory_preview
    assert preview.column_support_projection and preview.column_weighted_resampling
    assert preview.column_preview_pixels == 2048
    assert preview.column_preview_support_probability == .01
    assert preview.column_preview_support_l2 == .5
    assert preview.wave_grid == baseline.wave_grid
    assert preview.wave_grid.maximum_pixels == baseline.wave_grid.maximum_pixels
    assert preview.column_step_mm == 2.
    for field in ("source", "surface", "gun", "radial_gun", "column_kinetic_order"):
        assert getattr(preview, field) == getattr(baseline, field)
    assert panel._state == physical
    assert panel.column_preview_pixels.isEnabled()
    assert not panel.column_support_projection.isEnabled()
    assert not panel.column_weighted_resampling.isEnabled()
    panel.support_l2_budget.setText("0.08")
    panel.support_probability_budget.setText("0.0005")
    panel.column_preview_pixels.setValue(1024)
    assert panel.column_step.isEnabled()
    panel.column_step.setValue(1.25)
    edited = panel._make_request()
    assert edited.column_preview_support_probability == .0005
    assert edited.column_preview_support_l2 == .08
    assert edited.wave_grid == baseline.wave_grid
    assert edited.column_preview_pixels == 1024
    assert edited.column_step_mm == 1.25
    panel.column_exploratory_preview.setChecked(False)
    assert panel._make_request() == replace(baseline, column_preview_pixels=1024)
    assert not panel.column_preview_pixels.isEnabled()
    assert panel.column_support_projection.isEnabled()
    # A later activation remembers the current strict draft, not the first one.
    panel.column_step.setValue(.375)
    panel.column_exploratory_preview.setChecked(True)
    assert panel.column_step.value() == 2.
    panel.column_exploratory_preview.setChecked(False)
    assert panel.column_step.value() == .375
    for tooltip in (panel.column_exploratory_preview.toolTip(), panel.column_step.toolTip()):
        assert "2 mm" in tooltip and "not converged for the whole column" in tooltip
        assert "Projection budgets do not bound axial-step error" in tooltip


@pytest.mark.parametrize("mode", ["ideal", "custom", "analytical", "ideal_thin"])
def test_leaving_finite_field_preview_restores_strict_budgets_before_request(panel, monkeypatch, mode):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.column_step.setValue(.875)
    baseline = panel._make_request()
    panel.column_exploratory_preview.setChecked(True)
    panel.column_step.setValue(1.5)
    panel._state.simulation_mode = mode
    panel.update_simulation_mode(panel._state)
    assert not panel.column_exploratory_preview.isChecked()
    assert not panel.column_exploratory_preview.isEnabled()
    assert panel.column_step.value() == baseline.column_step_mm
    if mode == "ideal_thin":
        with pytest.raises(ValueError, match="unavailable in Ideal Optics"):
            panel._make_request()
    else:
        assert panel._make_request() == baseline


def test_exploratory_preview_result_label_contains_actual_losses_without_accuracy_claim():
    from types import SimpleNamespace
    from temsim.gui.coherent_beam import _coherent_preview_text
    checkpoint = SimpleNamespace(record={"upstream": {"unified_numerical_profile": {
        "exploratory_preview": True, "effective_column_maximum_pixels": 1024,
        "maximum_support_probability_fraction": .001, "maximum_accumulated_support_l2": .1,
        "numerical_projection_loss_per_reference": 2.5e-5,
        "complex_l2_projection_bound_per_reference": .015}}})
    text = _coherent_preview_text(checkpoint)
    for phrase in ("Exploratory preview", "not accuracy qualified", "1024 pixels/axis",
                   "probability 0.001", "L2 0.1", "projection loss 2.5e-05", "projection L2 cost 0.015"):
        assert phrase in text
    assert _coherent_preview_text(SimpleNamespace(record={})) == ""
