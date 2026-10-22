"""Wire spatial coverage to displayed results, not cursor or requested planes."""
from types import SimpleNamespace

import numpy as np
import pytest
from test_incremental_ray_scene import _result, make_workspace


def completed(target=16., resumable=None):
    # Explicit UI-only data: the physical restart path has separate tests.
    result = _result(end_z=target)
    result.simulation.metrics.update(section_target_z_mm=target,
        section_resumable_through_z_mm=target if resumable is None else resumable)
    result.simulation.section_checkpoint = SimpleNamespace(
        gun_trace=SimpleNamespace(z_mm=np.array([0., 4.])))
    return result


def test_result_cutoff_selection_cursor_and_staleness_stay_separate(make_workspace):
    view = make_workspace()
    page = view.interactive_calculation
    view.display_result(completed(resumable=12.), "Preview")
    bar = view.ray_calculation_extent
    assert bar.extent["completed_z_mm"] == 16.
    assert bar.extent["resumable_z_mm"] == 12.
    assert bar.extent["requested_z_mm"] is None
    assert not bar.extent["stale"]
    assert view.ray_plot_details.layout().indexOf(bar) == 0
    assert view.ray_plot_panel.layout().indexOf(view.ray_plot_details) == view.ray_plot_panel.layout().indexOf(view.plot) + 1
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    assert bar.extent["requested_z_mm"] == 19.
    assert bar.extent["completed_z_mm"] == 16.
    view.jump_to_ray_position(7., activate_tab=False)
    assert bar.extent["completed_z_mm"] == 16.
    # A changed physical input marks the old calculation, not the chosen plane.
    view.mark_ray_stale(SimpleNamespace(electron_gun=SimpleNamespace(
        type_key="fixture", display_name="Electron source"), apertures=()))
    assert bar.extent["stale"]
    view.display_result(completed(target=19.), "Preview")
    assert bar.extent["completed_z_mm"] == 19.
    assert not bar.extent["stale"]
    page.section_group.setChecked(False)
    assert bar.extent["requested_z_mm"] is None
    assert bar.extent["completed_z_mm"] == 19.


def test_request_without_completed_result_never_fills_bar(make_workspace):
    view = make_workspace()
    page = view.interactive_calculation
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    assert view.ray_calculation_extent.extent["requested_z_mm"] == 19.
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None
    # Historical diagram data are still displayed without inventing completion.
    view.display_result(_result(), "Preview")
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None


def test_recapturing_settings_updates_the_requested_marker(make_workspace):
    from temsim.optics.column import default_state
    view = make_workspace()
    page = view.interactive_calculation
    page.section_group.setChecked(True)
    page.section_z.setValue(19.)
    page.set_source(default_state())
    assert view.ray_calculation_extent.extent["requested_z_mm"] == page.section_z.value()
    assert view.ray_calculation_extent.extent["completed_z_mm"] is None


@pytest.mark.parametrize("quality", ["Preview", "Medium", "High accuracy"])
@pytest.mark.parametrize("selection", ["chosen", "dragging", "outside_new_extent"])
def test_calculation_completion_keeps_observation_plane_and_updates_spot(
    make_workspace, qtbot, monkeypatch, quality, selection,
):
    """Exercise the completion handler as well as the real ray/spot views.

    The small retained arrays stand in for completed instrument calculations;
    no field solve, trajectory propagation or background archive is needed.
    """
    from temsim.gui.main_window import MainWindow

    view = make_workspace()
    view.show()
    qtbot.waitUntil(view.transverse_beam.isVisible)

    def ignore(*_args, **_kwargs):
        pass

    monkeypatch.setattr(view.model_inspector, "display_result", ignore)
    monkeypatch.setattr(view.vacuum_map, "set_result", ignore)
    for panel in (view.probe_aberrations, view.image_aberrations, view.optical_transfer):
        monkeypatch.setattr(panel, "display_result", ignore)
    monkeypatch.setattr(view, "_display_ray_scope_products", ignore)
    monkeypatch.setattr("temsim.calculation_cache.state_model_signature", lambda _state: "gui-test")
    shell = SimpleNamespace(
        workspace=view, state=SimpleNamespace(),
        result_files=SimpleNamespace(loading=False, hold_automatic_preview=False,
                                     refresh_actions=ignore),
        paired_beams=SimpleNamespace(token=None),
        _interactive_preview_pending=False, _interactive_preview_in_flight=lambda: False,
        live_beam_refresh=SimpleNamespace(particle_ready=ignore),
        _archive_matching_particle_result=ignore, _schedule_design_explorer_refresh=ignore,
        _selected_component_key=None,
        assembly_panel=SimpleNamespace(update_direct_alignment_metrics=ignore),
        status_label=SimpleNamespace(setText=ignore),
        log_output=SimpleNamespace(appendPlainText=ignore),
    )
    initial = completed(target=2987.15)
    initial.model_signature, initial.workflow = "gui-test", "rays"
    MainWindow._calculation_ready(shell, quality, initial, .1)
    if quality != "High accuracy":
        assert view._selected_z_mm == pytest.approx(2987.15)  # First-result default.
    chosen = 12.345678
    view.jump_to_ray_position(chosen, activate_tab=False)
    expected = chosen
    target = 2987.15
    if selection == "dragging":
        expected = 14.125
        view.axial_cursor_item.setValue(expected)  # Result arrives before release.
        assert view._selected_z_mm == chosen
    elif selection == "outside_new_extent":
        view.jump_to_ray_position(25., activate_tab=False)
        target, expected = 16., 25.
    view.plot.setRange(xRange=(1., 30.), yRange=(-.3, .4), padding=0, disableAutoRange=True)
    before_range = view.plot.viewRange()

    first_x = None
    for offset_m in (3e-6, 6e-6):
        latest = completed(target=target)
        latest.model_signature, latest.workflow = "gui-test", "rays"
        latest.simulation.branches["000"].x += offset_m
        MainWindow._calculation_ready(shell, quality, latest, .1)
        assert view._selected_z_mm == pytest.approx(expected)
        assert view.axial_position.value() == pytest.approx(expected)
        assert view.axial_cursor_item.value() == pytest.approx(expected)
        assert view.selected_plane_readout._selected_z_mm == pytest.approx(expected)
        assert view._transverse_focus_request == ("z", expected)
        spot = view.transverse_beam
        assert spot._result is latest
        assert spot._plane_z_mm == pytest.approx(expected)
        assert view.interactive_calculation._section_result is latest
        assert view.ray_calculation_extent.extent["completed_z_mm"] == target
        np.testing.assert_allclose(view.plot.viewRange(), before_range, rtol=0, atol=1e-12)
        x = spot.analysis.plane_data().x_m.copy()
        if selection == "outside_new_extent":
            assert not x.size and spot._scatter is None
            assert "no calculated ray data" in spot.summary.text()
            assert view.selected_plane_readout._diagnostic.kind == "not_calculated"
            from temsim.gui.visualization import VisualizationWorkspace
            VisualizationWorkspace._update_interaction_detail(view)
            assert "Not calculated" in view.interaction_detail.toPlainText()
            assert "Current reaching" not in view.interaction_detail.toPlainText()
            view._jump_to_position_input()
            assert view._selected_z_mm == expected
            continue
        assert x.size
        if first_x is None:
            first_x = x
        else:
            np.testing.assert_allclose(x - first_x, 3e-6, rtol=0, atol=1e-15)
    if selection == "outside_new_extent":
        restored = completed(target=2987.15)
        restored.model_signature, restored.workflow = "gui-test", "rays"
        MainWindow._calculation_ready(shell, quality, restored, .1)
        assert view._selected_z_mm == view.transverse_beam._plane_z_mm == expected
        assert view.transverse_beam.analysis.plane_data().x_m.size


def test_component_editor_selections_preserve_manual_z_during_reload(make_workspace, qtbot, monkeypatch):
    from temsim.gui.main_window import MainWindow
    from test_incremental_ray_scene import _part

    view = make_workspace()
    view.show()
    qtbot.waitUntil(view.transverse_beam.isVisible)
    result = completed(target=2987.15)
    # Editor selection travels through the same handler for every component.
    result.assembly.parts += (_part("beam_deflector", "Beam deflector", 5.),
                              _part("objective_stigmator", "Objective stigmator", 9.))
    view.display_result(result, "Preview")
    view.jump_to_ray_position(12.345678, activate_tab=False)
    monkeypatch.setattr(view.physical_layout.model_editor, "focus_project_part", lambda _part: None)
    shell = SimpleNamespace(
        state=SimpleNamespace(), workspace=view, assembly=result.assembly,
        _runtime_targets={}, _anchors_by_key={},
        parameter_panel=SimpleNamespace(set_context=lambda *_args, **_kwargs: None,
                                        set_lens_diagnostics=lambda _text: None),
    )
    for part in result.assembly.parts:
        selection = SimpleNamespace(key=part.key, label=part.name, is_module=False, module_path=None)
        MainWindow._select_tree_item(shell, selection)
        assert view._focused_part is part
        assert view._selected_z_mm == view.transverse_beam._plane_z_mm == 12.345678
        assert view._transverse_focus_request == ("z", 12.345678)
    latest = completed(target=2987.15)
    view.display_result(latest, "Preview")
    assert view._selected_z_mm == view.transverse_beam._plane_z_mm == 12.345678
    # Context navigation carries an explicit clicked Z; highlighting its part
    # must not replace that coordinate with the component's mechanical centre.
    view.jump_to_ray_position(13., activate_tab=False)
    view.reveal_component(latest.assembly.part("test_lens"), select_observation_plane=False)
    assert view._selected_z_mm == view.transverse_beam._plane_z_mm == 13.


@pytest.mark.parametrize("dragging", [False, True])
def test_input_refresh_keeps_follow_ray_at_current_observation_before_first_wave_result(
    make_workspace, dragging,
):
    from temsim.optics.column import default_state

    view = make_workspace()
    state = default_state()
    view.display_result(completed(target=2987.15), "Preview")
    view.mark_ray_stale(state)
    view.jump_to_ray_position(1000., activate_tab=False)
    expected = 1001.125 if dragging else 1000.
    if dragging:
        view.axial_cursor_item.setValue(expected)
    assert view.coherent_beam.result is None
    assert view.coherent_beam._target_z_mm == expected
    state.lenses[0].percent += .1
    view.mark_ray_stale(state)
    assert view.axial_cursor_item.value() == expected
    assert view._transverse_focus_request == ("z", expected)
    assert view.coherent_beam._target_z_mm == expected
    assert not view.coherent_beam.timer.isActive()
    assert view.coherent_beam._worker is None
