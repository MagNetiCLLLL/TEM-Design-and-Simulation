"""Nested-population GUI routing uses mock results, never physical transport."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_working_point_restore_gui import window


def seed(count, name):
    return SimpleNamespace(signatures={"request": name},
        simulation=SimpleNamespace(gun_trace=SimpleNamespace(
            exit_bundle=SimpleNamespace(ray_id=tuple(range(count))))))


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
def test_controller_routes_largest_retained_compatible_prefix_across_quality(qapp, monkeypatch, quality):
    from temsim.gui.calculation_controller import CalculationController
    from temsim.physics import incremental_particles

    controller = CalculationController(persistent_cache_enabled=False)
    previous = seed(8, "older-high")
    preview = seed(16, "preview")
    incompatible = seed(24, "different-source")
    controller._high_cache["old"] = previous
    controller._tuning_cache[("Preview", "old")] = preview
    controller._loaded_tuning_sections[("Medium", "different")] = incompatible
    state = SimpleNamespace(step_mm=.025, quality=quality)
    seen = []

    def eligibility(result, captured, *, target_z_mm):
        seen.append((result, captured, target_z_mm))
        return result is not incompatible

    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", eligibility)
    monkeypatch.setattr(controller, "_seed_reuse_score", lambda *_: 0)
    chosen = controller._incremental_particle_seed(state, {}, previous, target_z_mm=123.25)

    assert chosen is preview
    assert len(seen) == 3
    assert all(captured is state and target == 123.25 for _result, captured, target in seen)
    assert state.step_mm == .025
    # Selection shares immutable result references; it does not publish a hit
    # or create an unbounded extra cache holding an otherwise evicted result.
    assert controller._cache_hits == {"high": 0, "tuning": 0}
    assert controller._high_cache["old"] is previous
    assert controller._tuning_cache[("Preview", "old")] is preview


def test_incremental_candidate_does_not_replace_exact_completed_product(qapp, monkeypatch):
    from temsim.gui.calculation_controller import CalculationController
    from temsim.physics import incremental_particles

    controller = CalculationController(persistent_cache_enabled=False)
    complete, lower = seed(32, "complete"), seed(16, "lower")
    controller._tuning_cache[("Preview", "lower")] = lower
    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", lambda result, *_a, **_k: result is lower)
    monkeypatch.setattr(controller, "_seed_reuse_score", lambda result, *_: 4 if result is complete else 1)
    assert controller._incremental_particle_seed(object(), {}, complete) is complete


def test_metadata_rejection_and_cache_eviction_cannot_create_incremental_seed(qapp, monkeypatch):
    from temsim.gui.calculation_controller import CalculationController
    from temsim.physics import incremental_particles

    controller = CalculationController(persistent_cache_enabled=False)
    lower = seed(16, "lower")
    controller._tuning_cache[("Preview", "lower")] = lower
    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", lambda *_a, **_k: False)
    assert controller._incremental_particle_seed(object(), {}) is None
    controller.clear_memory_cache()
    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", lambda *_a, **_k: True)
    assert controller._incremental_particle_seed(object(), {}) is None


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
def test_official_submit_hands_incremental_candidate_to_worker_without_changing_numerics(qapp, monkeypatch, quality):
    from temsim.gui import calculation_controller as module
    from temsim.optics.column import default_state
    from temsim.physics import incremental_particles

    state = default_state()
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    state.step_mm = .025
    before = deepcopy(state.to_dict())
    controller = module.CalculationController(persistent_cache_enabled=False)
    lower = seed(16, "other-quality")
    controller._tuning_cache[("Medium", "lower")] = lower
    monkeypatch.setattr(controller, "_seed_reuse_score", lambda *_: 0)
    seen = []

    def eligible(result, captured, *, target_z_mm):
        seen.append((captured, target_z_mm))
        return result is lower

    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", eligible)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    section = {"target_z_mm": state.sample.upper_surface_z_mm, "component_keys": ()}
    options = ({"particle_tuning": True, "section_request": section} if quality == "Preview" else
               {"workflow": "rays"})

    controller.submit(state, quality, 2048, .025, **options)

    assert len(workers) == 1
    worker = workers[0]
    assert worker.existing_result is lower
    assert worker.state.electron_gun.ray_count == 2048
    assert worker.state.step_mm == .025
    assert worker.state.history_step_mm == (2. if quality == "Preview" else .5)
    assert worker.state.electron_gun.emitter.energy_spread_fwhm_ev == .3
    assert seen == [(worker.state, section["target_z_mm"] if quality == "Preview" else None)]
    assert state.to_dict() == before
    assert worker.particle_tuning == (quality == "Preview")
    assert worker.workflow == ("full" if quality == "Preview" else "rays")
    adapter = worker.job_adapter()
    assert adapter.roots[0] is worker.state and adapter.roots[1] is lower
    assert adapter.claim is worker.resource_claim
    assert controller._cache_hits == {"high": 0, "tuning": 0}
    controller.invalidate_pending(include_explicit=True)


@pytest.mark.parametrize("location", ["cache", "display_after_clear"])
def test_legacy_growth_is_rejected_before_a_worker_and_keeps_old_cross_quality_seed(qapp, monkeypatch, location):
    from temsim.gui import calculation_controller as module
    from temsim.optics.column import default_state

    previous_state = default_state()
    previous_state.electron_gun.emitter.sampling = None
    previous_state.electron_gun.emitter.ray_count = 1024
    state = default_state()
    state.electron_gun.emitter.sampling = None
    state.electron_gun.emitter.ray_count = 2048
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    previous = seed(1024, "legacy-preview")
    previous.state_snapshot = previous_state
    provider = (lambda: (previous,)) if location == "display_after_clear" else None
    controller = module.CalculationController(persistent_cache_enabled=False, displayed_particle_results=provider)
    controller._tuning_cache[("Preview", "legacy")] = previous
    if location == "display_after_clear":
        controller.clear_memory_cache()
    before = deepcopy(state.to_dict())
    errors = []
    controller.failed.connect(lambda quality, message: errors.append((quality, message)))
    monkeypatch.setattr(controller.pool, "start", lambda *_: pytest.fail("Legacy growth must not execute transport"))

    with pytest.raises(ValueError, match="historical batch-dependent source"):
        controller.submit(state, "High accuracy", 2048, .025, workflow="rays")

    assert len(errors) == 1 and errors[0][0] == "High accuracy"
    assert "historical result remains unchanged" in errors[0][1]
    assert not controller.has_pending_requests
    if location == "cache":
        assert controller._tuning_cache[("Preview", "legacy")] is previous
    else:
        assert not controller._tuning_cache
        assert controller._displayed_particle_results()[0] is previous
    assert state.to_dict() == before
    assert state.electron_gun.emitter.sampling is None


@pytest.mark.parametrize("existing", [False, True])
def test_displayed_source_candidate_over_budget_is_checked_but_not_reused(qapp, monkeypatch, existing):
    from temsim.gui import calculation_controller as module
    from temsim.physics import incremental_particles

    previous = seed(16, "displayed")
    controller = module.CalculationController(
        persistent_cache_enabled=False, displayed_particle_results=lambda: (previous,))
    checked = []
    monkeypatch.setattr(incremental_particles, "validate_particle_count_growth", lambda result, _state: checked.append(result))
    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", lambda *_a, **_k: True)
    monkeypatch.setattr(module, "retained_memory_inventory", lambda result: {"array": 65})
    state = object()
    controller._validate_particle_growth(state)
    assert controller._incremental_particle_seed(
        state, {}, previous if existing else None, maximum_seed_bytes=64) is None
    assert checked == [previous]
    assert not controller._high_cache and not controller._tuning_cache
    assert controller._displayed_particle_results()[0] is previous


def test_compatible_displayed_source_still_routes_after_controller_cache_clear(qapp, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim.physics import incremental_particles

    previous = seed(16, "displayed")
    controller = module.CalculationController(
        persistent_cache_enabled=False, displayed_particle_results=lambda: (previous,))
    controller._tuning_cache[("Preview", "old")] = previous
    controller.clear_memory_cache()
    monkeypatch.setattr(incremental_particles, "can_extend_particle_result", lambda *_a, **_k: True)
    monkeypatch.setattr(module, "retained_memory_inventory", lambda result: {"array": 64})
    monkeypatch.setattr(controller, "_seed_reuse_score", lambda *_: 0)
    assert controller._incremental_particle_seed(object(), {}, maximum_seed_bytes=64) is previous


def test_displayed_legacy_growth_is_rejected_before_a_complete_result_cache_hit(qapp, monkeypatch):
    from temsim.gui import calculation_controller as module
    from temsim.optics.column import default_state

    state = default_state()
    state.electron_gun.emitter.sampling = None
    state.electron_gun.emitter.ray_count = 2048
    state.sample.wave_enabled = state.sample.stem_wave_enabled = False
    older = seed(1024, "displayed-legacy")
    older.state_snapshot = default_state()
    older.state_snapshot.electron_gun.emitter.sampling = None
    older.state_snapshot.electron_gun.emitter.ray_count = 1024
    visible = []
    controller = module.CalculationController(
        persistent_cache_enabled=False, displayed_particle_results=lambda: visible)
    workers = []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    # Capture the normal request identity, without executing the worker.
    controller.submit(state, "High accuracy", 2048, .025, workflow="rays")
    key = workers[0].request_signatures["request"]
    controller.invalidate_pending(include_explicit=True)
    controller._high_cache[key] = seed(2048, "old-complete")
    visible.append(older)

    with pytest.raises(ValueError, match="historical batch-dependent source"):
        controller.submit(state, "High accuracy", 2048, .025, workflow="rays")

    assert len(workers) == 1
    assert not controller.has_pending_requests
    assert controller._cache_hits["high"] == 0
    assert visible == [older]


def test_count_edit_keeps_previous_frames_caches_plane_and_physical_controls(window, monkeypatch):
    """An N edit invalidates publication without destroying a reuse candidate."""
    state = window.state
    state.electron_gun.emitter.ray_count = 1024
    window._sync_ray_count_selector()
    before = deepcopy(state.to_dict())
    selected_z = float(state.sample.upper_surface_z_mm)
    workspace = window.workspace
    workspace._selected_z_mm = selected_z
    assert workspace.axial_cursor_item is None
    previous = SimpleNamespace(simulation=SimpleNamespace(metrics={}))
    workspace._last_result = previous
    workspace._preview_result = previous
    workspace._high_accuracy_result = previous
    workspace._high_accuracy_current = True
    controller = window.calculations
    controller._high_cache["previous-high"] = previous
    controller._tuning_cache[("Preview", "previous-preview")] = previous
    controller._tuning_seeds["Preview"] = previous
    monkeypatch.setattr(controller.pool, "start", lambda *_: pytest.fail("A count edit must not run transport"))
    monkeypatch.setattr(workspace, "clear_result", lambda: pytest.fail("A count edit must retain the previous frame"))

    window.high_rays.setValue(2048)

    expected = deepcopy(before)
    expected["electron_gun"]["components"]["feg_tip"]["ray_count"] = 2048
    assert state.to_dict() == expected
    assert state.electron_gun.emitter.energy_spread_fwhm_ev == .3
    assert window.high_step.value() == state.step_mm == before["step_mm"]
    assert workspace._last_result is previous
    assert workspace._preview_result is previous
    assert workspace._high_accuracy_result is previous
    assert not workspace._high_accuracy_current
    assert workspace._selected_z_mm == selected_z
    assert workspace.coherent_beam._target_z_mm == selected_z
    assert controller._high_cache["previous-high"] is previous
    assert controller._tuning_cache[("Preview", "previous-preview")] is previous
    assert controller._tuning_seeds["Preview"] is previous
    assert not window.preview_timer.isActive()
    assert "previous" in workspace.ray_source_status.text().lower()


def test_main_window_can_explicitly_select_nested_source_without_retuning_or_erasing_history(window, monkeypatch):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    from temsim.optics.electron_gun.emitter import HALTON_PREFIX_SAMPLING_SCHEMA

    state = window.state
    gun = state.electron_gun
    gun.emitter.sampling = None
    before = deepcopy(state.to_dict())
    previous = SimpleNamespace(simulation=SimpleNamespace(metrics={}))
    window.workspace._last_result = previous
    window.workspace._high_accuracy_result = previous
    errors = []
    monkeypatch.setattr(window, "_show_error", errors.append)
    monkeypatch.setattr(window.preview_timer, "start", lambda *_: None)
    monkeypatch.setattr(window.calculations.pool, "start", lambda *_: pytest.fail("Source editing must not run transport"))
    monkeypatch.setattr(window, "apply_direct_alignment", lambda *_: pytest.fail("Sampling selection must retain lens settings"))

    def select_nested(dialog):
        dialog.nested_sampling_button.click()
        dialog.accept()
        assert not dialog.match_transport_requested
        return dialog.result()

    monkeypatch.setattr(GunSourceDialog, "exec", select_nested)
    window._open_tip_editor()

    assert not errors
    expected = deepcopy(before)
    expected["electron_gun"]["components"]["feg_tip"]["sampling"] = {
        "schema": HALTON_PREFIX_SAMPLING_SCHEMA, "seed": 0}
    assert state.to_dict() == expected
    assert window.workspace._last_result is previous
    assert window.workspace._high_accuracy_result is previous
    assert gun._trace_cache is None
