from copy import deepcopy

import pytest

from temsim.gui.model_inspector import ModelInspectorPage
from temsim.optics.column import default_state
from temsim.optics.model import State


def _analytic_state_with_inactive_recipe(mode):
    from temsim.simulation_modes import switch_mode
    state = default_state()
    switch_mode(state, mode)
    state.lens_field_map_descriptors[state.lenses[0].key] = {
        "solver": "axisymmetric_linear_fem", "relative_permeability": 1000.,
        "ampere_turns": 250., "radial_nodes": 16, "axial_nodes": 32,
        "padding_factor": 2.,
    }
    state.probe_aberrations = {"mode": "manual", "c3_mm": .2}
    return state


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
def test_manual_mode_edit_preserves_analytic_mode_and_inactive_maps(qtbot, mode):
    from temsim.physics.lens_field_provider import active_mapped_providers
    state = _analytic_state_with_inactive_recipe(mode)
    recipes = deepcopy(state.lens_field_map_descriptors)
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.fit_angle.setText("12.5")
    with qtbot.waitSignal(page.changed):
        page._apply_mode()
    assert state.simulation_mode == mode
    assert state.probe_aberrations == {"mode": "manual", "c3_mm": .2, "fit_semiangle_mrad": 12.5}
    assert state.lens_field_map_descriptors == recipes
    assert active_mapped_providers(state) == ()


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
def test_field_fit_is_disabled_and_rejected_without_mutating_analytic_state(qtbot, mode):
    state = _analytic_state_with_inactive_recipe(mode)
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    index = page.aberration_mode.findData("field_derived")
    assert not page.aberration_mode.model().item(index).isEnabled()
    before = deepcopy(state.to_dict())
    changed = []
    page.changed.connect(changed.append)
    page.aberration_mode.setCurrentIndex(index)  # Bypass disabled UI to check the mutation boundary.
    with qtbot.waitSignal(page.error) as emitted:
        page._apply_mode()
    assert "requires an active spatial field map" in emitted.args[0]
    assert state.to_dict() == before
    assert not changed


@pytest.mark.parametrize("mode", ["ideal", "analytical"])
def test_coefficient_dialog_preserves_analytic_mode_and_inactive_maps(qtbot, monkeypatch, mode):
    from temsim.gui.aberration_dialog import AberrationSettingsDialog
    from temsim.physics.lens_field_provider import active_mapped_providers
    state = _analytic_state_with_inactive_recipe(mode)
    recipes = deepcopy(state.lens_field_map_descriptors)
    state._effective_aberration_cache = {"probe": "stale"}
    def accepted(dialog):
        dialog.result_options = {"mode": "manual", "c3_mm": .42}
        return dialog.DialogCode.Accepted
    monkeypatch.setattr(AberrationSettingsDialog, "exec", accepted)
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    with qtbot.waitSignal(page.changed):
        page._edit_aberrations()
    assert state.simulation_mode == mode
    assert state.probe_aberrations == {"mode": "manual", "c3_mm": .42}
    assert state.lens_field_map_descriptors == recipes
    assert active_mapped_providers(state) == ()
    assert "probe" not in state._effective_aberration_cache


def test_model_controls_are_explicit_serializable_and_do_not_solve(qtbot, monkeypatch):
    from temsim.physics import axisymmetric_magnetostatics
    monkeypatch.setattr(axisymmetric_magnetostatics, "solve_geometry_field_map", lambda *args: (_ for _ in ()).throw(AssertionError("UI must not solve")))
    state = default_state()
    original_strengths = [lens.percent for lens in state.lenses]
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    page.permeability.setText("1000")
    page.ampere_turns.setText("250")
    with qtbot.waitSignal(page.changed):
        page._apply_field()
    key = page.lens.currentData()
    assert state.lens_field_map_descriptors[key]["solver"] == "axisymmetric_linear_fem"
    assert original_strengths == [lens.percent for lens in state.lenses]
    page.aberration_mode.setCurrentIndex(page.aberration_mode.findData("field_derived"))
    with qtbot.waitSignal(page.changed):
        page._apply_mode()
    restored = State.from_dict(state.to_dict())
    assert restored.probe_aberrations["mode"] == "field_derived"
    assert restored.lens_field_map_descriptors == state.lens_field_map_descriptors
    assert all(page.table.item(row, 1).text() != "Validated" for row in range(page.table.rowCount()))


def test_field_fit_is_reused_by_detached_snapshot_and_display(qtbot, monkeypatch):
    from temsim.optics.column import default_state
    from temsim.optics import aberrations, field_aberrations
    from temsim.physics import lens_field_provider
    from temsim.gui.aberration_view import AberrationComparisonView

    state = default_state()
    # Production workers start from the normalised serialization boundary.
    state = type(state).from_dict(state.to_dict())
    state.image_aberrations = {"mode": "field_derived"}
    coefficient = aberrations.EffectiveAberrationSet("objective image", "field-derived", c3_mm=.42)
    diagnostics = {
        "fit_rms_m": 1e-12, "fit_condition_number": 5,
        "source": "Test-only fit", "diagnostic_scope": "UI cache contract",
        "unmeasured_coefficients": (),
        "coefficient_status": {term: "field ray fit" for term, _, _ in aberrations.SYSTEM_COEFFICIENT_ROWS},
        "correction_comparison_available": False,
    }
    calls = []
    def fit(state, system):
        calls.append(system)
        return coefficient, coefficient, diagnostics
    monkeypatch.setattr(field_aberrations, "derive_field_aberrations", fit)
    monkeypatch.setattr(lens_field_provider, "active_mapped_providers", lambda state: ())
    aberrations.prepare_field_aberration_diagnostics(state)
    snapshot = type(state).from_dict(state.to_dict())
    aberrations.prepare_field_aberration_diagnostics(snapshot, state)
    assert calls == ["image"]
    assert snapshot._effective_aberration_cache is not state._effective_aberration_cache

    view = AberrationComparisonView(fixed_system="image")
    qtbot.addWidget(view)
    view._state = snapshot
    view._refresh()
    view._refresh()
    assert calls == ["image"]
    assert "correction comparison not calculated" in view.summary.text()
    assert view.table.isColumnHidden(2)
    assert view.table.isColumnHidden(4)
    assert view.table.rowCount() == len(aberrations.SYSTEM_COEFFICIENT_ROWS)
