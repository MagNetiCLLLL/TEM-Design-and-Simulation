"""Column-only exploratory allowances never alter the strict gun request."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.tip_wave_pipeline import TipWaveRequest, _execute_tip_wave
from temsim.physics.wave_grid import (
    EXPLORATORY_SUPPORT_L2_MAXIMUM, EXPLORATORY_SUPPORT_PROBABILITY_MAXIMUM,
    STRICT_SUPPORT_BUDGET_MAXIMUM, WaveGridNumerics,
)
from test_coherent_beam_gui import panel, _enable_unified_numeric_controls


def _preview(**options):
    return replace(TipWaveRequest(), column_exploratory_preview=True,
        column_support_projection=True, column_weighted_resampling=True, **options)


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_exploratory_grid_flag_requires_exact_boolean(flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        WaveGridNumerics(exploratory_support_budget=flag).validate()


def test_wider_preview_allowances_are_explicit_and_do_not_raise_other_limits():
    preview = WaveGridNumerics(exploratory_support_budget=True,
        maximum_accumulated_support_l2=EXPLORATORY_SUPPORT_L2_MAXIMUM,
        maximum_support_loss_fraction=EXPLORATORY_SUPPORT_PROBABILITY_MAXIMUM).validate()
    assert preview.column_identity()["exploratory_support_budget"] is True
    assert "exploratory_support_budget" not in WaveGridNumerics().column_identity()
    for name, maximum in (("maximum_accumulated_support_l2", 4.),
                          ("maximum_support_loss_fraction", .5)):
        with pytest.raises(ValueError, match=name):
            replace(preview, **{name: np.nextafter(maximum, np.inf)}).validate()
        with pytest.raises(ValueError, match=name):
            replace(WaveGridNumerics(), **{name: np.nextafter(STRICT_SUPPORT_BUDGET_MAXIMUM, np.inf)}).validate()
    for name in ("maximum_spectral_loss_fraction", "maximum_incoherent_mode_loss_probability"):
        with pytest.raises(ValueError, match=name):
            replace(preview, **{name: np.nextafter(.1, 1.)}).validate()
    for probability in (1., 1.01, float("inf")):
        with pytest.raises(ValueError, match="maximum_support_loss_fraction"):
            replace(preview, maximum_support_loss_fraction=probability).validate()


@pytest.mark.parametrize("field,maximum", [
    ("column_preview_support_probability", .5), ("column_preview_support_l2", 4.)])
@pytest.mark.parametrize("invalid", [True, np.bool_(False), "0.01", float("nan"), float("inf"), -.1])
def test_column_override_requires_explicit_finite_bounded_number(field, maximum, invalid):
    with pytest.raises(ValueError, match=field):
        _preview(**{field: invalid}).validate()
    _preview(**{field: maximum}).validate()
    _preview(**{field: 0.}).validate()
    with pytest.raises(ValueError, match=field):
        _preview(**{field: np.nextafter(maximum, np.inf)}).validate()


def test_probability_one_is_forbidden_while_l2_one_is_a_valid_preview_allowance():
    with pytest.raises(ValueError, match="column_preview_support_probability"):
        _preview(column_preview_support_probability=1.).validate()
    _preview(column_preview_support_l2=1.).validate()


@pytest.mark.parametrize("change", [dict(column_exploratory_preview=False),
    dict(column_support_projection=False), dict(column_weighted_resampling=False),
    dict(column_model="first_order_reference")])
def test_column_overrides_require_the_authorized_physical_preview_path(change):
    with pytest.raises(ValueError, match="requires|require"):
        replace(_preview(column_preview_support_probability=.5, column_preview_support_l2=4.), **change).validate()


def test_gun_grid_remains_strict_even_when_only_column_overrides_are_active():
    original = TipWaveRequest().validate()
    assert original.column_preview_support_probability is None and original.column_preview_support_l2 is None
    old = _preview().validate()
    assert old.wave_grid == original.wave_grid and not old.wave_grid.exploratory_support_budget
    for fields in ({"column_preview_support_probability": .01}, {"column_preview_support_l2": .5},
                   {"column_preview_support_probability": .5, "column_preview_support_l2": 4.}):
        request = _preview(**fields).validate()
        assert request.wave_grid is original.wave_grid
        assert request.source == original.source and request.gun == original.gun
    permissive = replace(original.wave_grid, exploratory_support_budget=True, maximum_accumulated_support_l2=.5)
    for request in (original, _preview(column_preview_support_l2=.5)):
        with pytest.raises(ValueError, match="gun wave_grid must keep strict"):
            replace(request, wave_grid=permissive).validate()


@pytest.mark.parametrize("mode", ["ideal", "custom", "analytical", "ideal_thin"])
def test_nonunified_override_rejection_precedes_any_source_or_cache_access(mode):
    request = _preview(column_preview_support_l2=.5).validate()
    # No gun, specimen, cache, or snapshot fields exist: the mode guard must
    # reject before any such execution prerequisite is read.
    with pytest.raises(ValueError, match="only in unified Finite-Field Optics"):
        _execute_tip_wave(SimpleNamespace(simulation_mode=mode), request, None, "unexecuted",
            use_cache=False, cancelled=lambda: False, progress_callback=None)


def test_gui_preview_edits_only_column_overrides_and_restores_exact_strict_text(panel, monkeypatch):
    _enable_unified_numeric_controls(panel, monkeypatch)
    panel.support_l2_budget.setText("1.2345678912345678e-4")
    panel.support_probability_budget.setText("4.321987654321e-8")
    panel.column_step.setValue(.625)
    panel.column_support_projection.setChecked(True)
    panel.column_weighted_resampling.setChecked(False)
    baseline = panel._make_request()
    strict_text = (panel.support_l2_budget.text(), panel.support_probability_budget.text())
    panel.column_exploratory_preview.setChecked(True)
    request = panel._make_request()
    assert request.column_preview_support_probability == .01 and request.column_preview_support_l2 == .5
    assert request.column_step_mm == 2.
    assert request.wave_grid == baseline.wave_grid and request.wave_grid.exploratory_support_budget is False
    for name in ("source", "surface", "gun", "radial_gun"):
        assert getattr(request, name) == getattr(baseline, name)
    panel.support_l2_budget.setText("0.3141592653589793")
    panel.support_probability_budget.setText("0.02")
    edited = panel._make_request()
    assert edited.column_preview_support_l2 == .3141592653589793
    assert edited.column_preview_support_probability == .02
    assert edited.wave_grid == baseline.wave_grid
    panel.support_l2_budget.setText("4")
    panel.support_probability_budget.setText("0.5")
    wider = panel._make_request().validate()
    assert wider.column_preview_support_l2 == 4.
    assert wider.column_preview_support_probability == .5
    assert wider.wave_grid == baseline.wave_grid
    assert wider.source == baseline.source and wider.gun == baseline.gun
    for widget, value in ((panel.support_l2_budget, "4.000000000000001"),
                          (panel.support_probability_budget, "0.5000000000000001"),
                          (panel.support_probability_budget, "1")):
        previous = widget.text()
        widget.setText(value)
        with pytest.raises(ValueError, match="Preview support"):
            panel._make_request()
        widget.setText(previous)
    panel.column_exploratory_preview.setChecked(False)
    assert (panel.support_l2_budget.text(), panel.support_probability_budget.text()) == strict_text
    assert panel._make_request() == baseline
    assert "not a phase-error certificate" in panel.support_l2_budget.toolTip()
    assert "up to 4" in panel.support_l2_budget.toolTip()
    assert "representation estimates" in panel.support_l2_budget.toolTip()
    assert "up to 0.5" in panel.support_probability_budget.toolTip()
    assert "do not certify accuracy" in panel.column_exploratory_preview.toolTip()
    assert "column-only" in panel.column_exploratory_preview.toolTip()
    panel.support_l2_budget.setText("0.5")
    with pytest.raises(ValueError, match="between zero and 0.1"):
        panel._make_request()


def test_wider_fresh_column_uses_new_limits_without_changing_fields_or_source(tmp_path):
    from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
    from test_exploratory_column_budgets import _preview as request_fixture, _state
    from test_unified_wave_runtime import _runtime
    from test_unified_weighted_resampling import _modes

    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    original_request = request_fixture(tmp_path)
    request = replace(original_request, column_preview_support_probability=.5, column_preview_support_l2=4.).validate()
    original_adapter = UnifiedPotentialWaveAdapter(state, original_request, runtime=runtime)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    for grid in (adapter.column_grid, adapter.material_grid):
        assert grid.maximum_support_loss_fraction == .5
        assert grid.maximum_accumulated_support_l2 == 4.
        assert grid.exploratory_support_budget
    assert request.wave_grid is original_request.wave_grid
    assert adapter.snapshot.numerical_digest == original_adapter.snapshot.numerical_digest
    assert adapter.runtime.geometry.digest == original_adapter.runtime.geometry.digest
    assert adapter.operator_options == original_adapter.operator_options
    assert adapter.identity != original_adapter.identity
    source = _modes(1)
    source_digest = source.digest
    result = adapter.column(source, 1000.0002)
    assert source.digest == source_digest
    before, after = source.beam.modes[0], result.beam.modes[0]
    assert before.mode_id == after.mode_id and before.energy_kev == after.energy_kev
    assert after.weight_per_reference_electron == pytest.approx(before.weight_per_reference_electron, rel=1e-13)
    context = result.record["unified_mode_contexts"][after.mode_id]
    allocation = context["unified_support_allocation"]
    assert allocation["relative_l2_limit"] == 4.
    assert allocation["probability_fraction_limit"] == .5
    assert allocation["maximum_l2_per_reference"] == 4.*np.sqrt(before.weight_per_reference_electron)
    assert allocation["maximum_probability_per_reference"] == .5*before.weight_per_reference_electron
    assert result.record["unified_support_summary"]["normalization_applied"] is False


def test_existing_preview_allocation_cannot_be_reset_to_wider_limits(tmp_path):
    from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
    from test_exploratory_column_budgets import _preview as request_fixture, _state
    from test_unified_wave_runtime import _runtime
    from test_unified_weighted_resampling import _modes

    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    original_request = request_fixture(tmp_path)
    original = UnifiedPotentialWaveAdapter(state, original_request, runtime=runtime)
    source = _modes(1)
    allocated = original._entrance(source, source.beam.modes[0])
    digest = allocated.digest
    for fields in ({"column_preview_support_probability": .5}, {"column_preview_support_l2": 4.},
                   {"column_preview_support_probability": .5, "column_preview_support_l2": 4.}):
        wider = UnifiedPotentialWaveAdapter(state, replace(original_request, **fields), runtime=runtime)
        with pytest.raises(ValueError, match="support limits changed"):
            wider._entrance(allocated, allocated.beam.modes[0])
        assert allocated.digest == digest
