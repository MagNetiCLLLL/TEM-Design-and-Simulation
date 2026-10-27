"""Real main-window wiring for three owners; no physical solve is substituted as evidence."""
from dataclasses import replace
from types import SimpleNamespace
from threading import Event

import pytest

from test_background_preview_gui import window
from test_coherent_beam_gui import panel
from test_coherent_state_controller import page, _two_states, _calculate
from temsim.calculation_progress import CalculationProgress, callback_progress


@pytest.fixture
def background_progress(qtbot, monkeypatch):
    """Exercise capture, preparation and worker signals with a bounded solver stub."""
    import temsim.gui.calculation_controller as module
    from temsim.gui.calculation_progress import CalculationProgressWidget
    from temsim.optics.column import default_state
    from temsim.physics.transport_progress import report_transport_progress
    from temsim.simulation_pipeline import CalculationResult

    controller = module.CalculationController(persistent_cache_enabled=False)
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    workers, updates, results, errors, finished = [], [], [], [], []
    monkeypatch.setattr(controller.pool, "start", workers.append)
    controller.progress_detail.connect(updates.append)
    controller.progress_detail.connect(widget.set_progress)
    controller.result_ready.connect(lambda _quality, result, _duration: results.append(result))
    controller.failed.connect(lambda _quality, message: errors.append(message))
    controller.finished.connect(finished.append)

    def simulation():
        return SimpleNamespace(incident=None, branches={}, metrics={})

    def rays(*_args, **_kwargs):
        report_transport_progress("Controlled particle transport", completed=1, total=2)
        return simulation()

    def calculate(state, *, progress_callback, **_kwargs):
        progress_callback(1, 2, "Controlled particle transport")
        return CalculationResult(simulation=simulation(), energy_filter=None, state_snapshot=state)

    monkeypatch.setattr(module, "run_ray_simulation", rays)
    monkeypatch.setattr(module, "calculate", calculate)
    monkeypatch.setattr(module, "detect_all_lens_crossovers", lambda *_args: ())
    monkeypatch.setattr("temsim.detector.particle_readout.measure_particle_detectors", lambda _result: None)
    context = SimpleNamespace(controller=controller, widget=widget, workers=workers,
        updates=updates, results=results, errors=errors, finished=finished, state=default_state())
    yield context
    controller.invalidate_pending(include_explicit=True)
    for worker in workers:
        if isinstance(worker, module.PreparationWorker):
            worker.release_inputs()


def _prepare_background(context, quality="Preview"):
    from temsim.gui.calculation_controller import CalculationWorker, PreparationWorker
    context.controller.submit_background(context.state, quality, 9, 1.)
    preparation = context.workers.pop(0)
    assert isinstance(preparation, PreparationWorker)
    operation_id = context.updates[-1].operation_id
    assert preparation.request_id == operation_id
    preparation.run()
    solver = context.workers.pop(0)
    assert isinstance(solver, CalculationWorker)
    assert solver.request_id == operation_id
    return solver, operation_id


def _assert_no_active_progress(context, status):
    assert not context.controller._requests
    assert all(not update.active for update in context.widget._updates.values())
    assert context.widget.displayed.status == status
    assert context.widget.current.maximum() > 0
    assert context.widget.overall.maximum() > 0
    assert "jobs active" not in context.widget.current_label.text()


@pytest.mark.parametrize("quality", ("Preview", "Medium", "High accuracy"))
def test_background_request_keeps_progress_identity_through_worker_success(background_progress, quality):
    context = background_progress
    worker, operation_id = _prepare_background(context, quality)
    worker.run()
    assert len(context.results) == 1 and not context.errors
    assert context.finished == [quality]
    assert context.updates[0].step == "Preparing captured inputs"
    assert any(update.step == "Controlled particle transport" and update.active for update in context.updates)
    assert {update.operation_id for update in context.updates} == {operation_id}
    assert len(context.widget._updates) == 1
    _assert_no_active_progress(context, "succeeded")


@pytest.mark.parametrize("quality", ("Preview", "Medium"))
def test_background_cache_completion_leaves_no_preparing_job(background_progress, qtbot, quality):
    context = background_progress
    worker, first_id = _prepare_background(context, quality)
    worker.run()
    original = context.results[0]
    context.controller.submit_background(context.state, quality, 9, 1.)
    second_id = context.updates[-1].operation_id
    preparation = context.workers.pop(0)
    assert preparation.request_id == second_id and second_id != first_id
    preparation.run()
    qtbot.waitUntil(lambda: len(context.results) == 2)
    assert not context.workers, "An exact cache hit must not submit another solver"
    assert context.results[-1].cache_hit
    assert context.results[-1].simulation is original.simulation
    assert {update.operation_id for update in context.updates} == {first_id, second_id}
    assert context.finished == [quality, quality]
    _assert_no_active_progress(context, "succeeded")


@pytest.mark.parametrize("phase", ("preparation", "solver"))
def test_background_failure_finishes_original_progress_identity(background_progress, monkeypatch, phase):
    from temsim.gui.calculation_request import CapturedCalculationRequest
    context = background_progress

    def fail(*_args, **_kwargs):
        raise RuntimeError("Controlled lifecycle failure")

    if phase == "preparation":
        monkeypatch.setattr(CapturedCalculationRequest, "prepare", fail)
        context.controller.submit_background(context.state, "Preview", 9, 1.)
        worker = context.workers.pop(0)
    else:
        worker, _operation_id = _prepare_background(context)
        monkeypatch.setattr("temsim.gui.calculation_controller.run_ray_simulation", fail)
    worker.run()
    assert not context.results
    assert context.errors == ["Controlled lifecycle failure"]
    assert context.finished == ["Preview"]
    assert len({update.operation_id for update in context.updates}) == 1
    _assert_no_active_progress(context, "failed")


@pytest.mark.parametrize("phase", ("preparation", "solver"))
def test_background_cancellation_clears_original_progress_and_ignores_late_worker(background_progress, phase):
    context = background_progress
    if phase == "preparation":
        context.controller.submit_background(context.state, "Preview", 9, 1.)
        worker = context.workers.pop(0)
    else:
        worker, _operation_id = _prepare_background(context)
    context.controller.invalidate_pending()
    count = len(context.updates)
    worker.run()
    assert worker.cancel_event.is_set()
    assert not context.results and not context.errors
    assert len(context.updates) == count
    assert len({update.operation_id for update in context.updates}) == 1
    _assert_no_active_progress(context, "cancelled")


def test_background_supersession_cancels_only_previous_operation(background_progress):
    context = background_progress
    old_worker, old_id = _prepare_background(context)
    new_worker, new_id = _prepare_background(context, "Medium")
    assert new_id != old_id
    assert context.widget._updates[old_id].status == "cancelled"
    assert context.widget.displayed.operation_id == new_id
    assert "jobs active" not in context.widget.current_label.text()
    count = len(context.updates)
    old_worker.run()
    assert len(context.updates) == count and not context.results
    new_worker.run()
    assert len(context.results) == 1
    _assert_no_active_progress(context, "succeeded")


def test_background_completion_preserves_concurrent_high_accuracy_progress(background_progress):
    context = background_progress
    high_worker, high_id = _prepare_background(context, "High accuracy")
    preview_worker, preview_id = _prepare_background(context)
    assert "2 jobs active" in context.widget.current_label.text()
    preview_worker.run()
    assert context.widget._updates[preview_id].status == "succeeded"
    assert context.widget.displayed.operation_id == high_id
    assert context.widget.displayed.active and not high_worker.cancel_event.is_set()
    assert "jobs active" not in context.widget.current_label.text()
    high_worker.run()
    assert len(context.results) == 2 and not context.errors
    assert {update.operation_id for update in context.updates} == {high_id, preview_id}
    _assert_no_active_progress(context, "succeeded")


def test_main_window_particle_wave_and_pair_signals_reach_shared_progress(window):
    widget = window.calculation_progress
    particle = CalculationProgress("particle-1", "Particles", "Column", completed=7, total=12,
        stages_completed=0, stages_total=4)
    window.calculations.progress_detail.emit(particle)
    assert widget.displayed == particle
    assert widget.current.value() == 7 and widget.current.maximum() == 12
    unknown = callback_progress(particle, 0, 0, "Adaptive work")
    window.calculations.progress_detail.emit(unknown)
    assert widget.current.maximum() == 0
    assert "total unknown" in widget.current_label.text()
    window.calculations.progress_detail.emit(unknown.terminal("succeeded"))
    assert widget.overall.value() == widget.overall.maximum() == 4
    wave = CalculationProgress("wave-1", "Coherent beam", "Readout",
        stages_completed=1, stages_total=3)
    window.workspace.coherent_beam.progress_detail.emit(wave)
    assert widget.displayed == wave
    assert widget.overall.value() == 1
    window.workspace.coherent_beam.progress_detail.emit(wave.terminal("failed", "Rejected plane"))
    assert widget.displayed.status == "failed"
    assert widget.overall.value() == 1
    pair = CalculationProgress("pair-1", "Paired beams", "Particle transport",
        stages_completed=0, stages_total=4)
    window.paired_beams.progress_detail.emit(pair)
    assert widget.displayed == pair
    assert "0/4 stages complete" in widget.overall_label.text()
    # These owners no longer show the unrelated calibration spinner.
    assert window.progress.isHidden()


def test_main_window_pair_wave_hook_counts_readout_and_terminal_once(window):
    pair = window.paired_beams
    pair.token, pair._stage, pair._material_stage = "paired-request", "wave", False
    pair._pair_progress = CalculationProgress(pair.token, "Paired beams", "Wave transport",
        stages_completed=1, stages_total=4)
    pair.progress_detail.emit(pair._pair_progress)
    wave = CalculationProgress("paired-wave", "Coherent beam", "Readout",
        stages_completed=1, stages_total=3, parent_id=pair.token)
    window.workspace.coherent_beam.progress_detail.emit(wave)
    assert pair._pair_progress.stages_completed == 2
    assert window.calculation_progress.displayed.operation_id == pair.token
    assert "jobs active" not in window.calculation_progress.current_label.text()
    window.workspace.coherent_beam.progress_detail.emit(replace(wave, stages_completed=2))
    assert window.calculation_progress.overall.value() == 3
    window.workspace.coherent_beam.progress_detail.emit(wave.terminal("succeeded"))
    assert pair._stage == "complete"
    assert window.calculation_progress.overall.value() == 4
    assert "Succeeded" in window.calculation_progress.overall_label.text()


def test_coherent_input_rejection_replaces_previous_success_progress(panel, monkeypatch):
    import pytest
    import temsim.gui.coherent_beam as module
    updates = []
    panel.progress_detail.connect(updates.append)
    panel._progress_state = CalculationProgress("old-wave", "Coherent beam", "Done",
        stages_completed=3, stages_total=3, status="succeeded")
    monkeypatch.setattr(module, "prepare_coherent_state", lambda *_args:
                        (_ for _ in ()).throw(ValueError("Unsupported field model")))
    with pytest.raises(ValueError, match="Unsupported field model"):
        panel.capture_calculation_inputs()
    assert updates[-1].status == "failed"
    assert updates[-1].operation_id != "old-wave"
    assert updates[-1].stages_completed == 0
    assert panel._worker is None


def test_saved_state_overall_completes_only_after_all_states_and_overlay_accept(page, qtbot):
    _two_states(page)
    updates = []
    page.progress_detail.connect(updates.append)
    _calculate(page, qtbot)
    owner = [item for item in updates if item.operation == "Saved coherent states"]
    assert owner
    assert {item.stages_total for item in owner} == {7}
    assert any(item.stages_completed == 3 and item.active for item in owner)
    assert any(item.stages_completed == 6 and item.active for item in owner)
    assert owner[-1].status == "succeeded" and owner[-1].stages_completed == 7
    assert all(item.active for item in owner[:-1])


def test_saved_state_failure_retains_completed_member_and_ignores_late_progress(page, qtbot):
    _two_states(page)
    page.execution.fail = lambda energy, _z: energy == 5.
    updates = []
    page.progress_detail.connect(updates.append)
    page.state_list.mode.setCurrentIndex(page.state_list.mode.findData("overlay"))
    page.calculate_button.click()
    qtbot.waitUntil(lambda: bool(updates) and updates[-1].status == "failed", timeout=10000)
    last = updates[-1]
    assert last.stages_completed == 3 and last.stages_total == 7
    page.state_set._progress_detail(page.state_set.generation - 1, replace(last, status="running"))
    assert updates[-1] is last


def test_accepted_wave_retains_pair_identity_failure_in_its_terminal_progress(panel, monkeypatch):
    import temsim.gui.coherent_beam as module
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot",
                        lambda _state: SimpleNamespace(digest="captured-pair"))
    def wrong_identity(_result):
        raise ValueError("Executed wave identity does not match this captured pair")
    context = SimpleNamespace(token="pair-token", instrument_identity="captured-pair",
        verify_particle=lambda: None, verify_wave=wrong_identity)
    request = panel._make_request()
    worker = module._WaveWorker(1, 1, panel._state, request, Event(), pair_context=context)
    panel._generation = panel._session = 1
    panel._request, panel._worker, panel._stale = request, worker, False
    panel._progress_state = worker.progress_state
    updates = []
    panel.progress_detail.connect(updates.append)
    worker.signals.progress_detail.connect(panel._progress_detail)
    worker.signals.solved.connect(panel._solved)
    worker.run()
    assert panel.result is not None and panel.preview is not None
    assert "identity" in panel.preview.comparison_error
    assert updates[-1].status == "succeeded"
    assert updates[-1].pair_error == panel.preview.comparison_error


def test_wave_preparation_progress_uses_structured_counts_and_keeps_overall_incomplete(panel, qtbot, monkeypatch):
    from temsim.calculation_progress import LocalProgressLabel
    from temsim.gui.calculation_progress import CalculationProgressWidget
    import temsim.gui.coherent_beam as module
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    worker = module._WaveWorker(1, 1, panel._state, panel._make_request(), Event())
    messages = []
    worker.signals.progress.connect(lambda _generation, message: messages.append(message))
    worker.signals.progress_detail.connect(lambda _generation, update: widget.set_progress(update))
    clock = iter((1., 2., 3., 3.001, 4.))
    monkeypatch.setattr(module, "monotonic", lambda: next(clock))

    worker._report_progress(0, 0, LocalProgressLabel("Preparing shared fields", 0, None))
    assert messages[-1] == "Preparing shared fields"
    assert widget.current.maximum() == 0
    assert "total unknown" in widget.current_label.text()
    assert widget.overall.value() == 0 and widget.overall.maximum() == 3

    # Text-only transport bridges pass legacy 0/0; their structured local
    # counts remain authoritative for both visible text and the progress bar.
    worker._report_progress(0, 0, LocalProgressLabel("Measuring scan responses", 7, 9))
    assert messages[-1] == "Measuring scan responses (7/9)"
    assert widget.current.value() == 7 and widget.current.maximum() == 9
    worker._report_progress(0, 0, LocalProgressLabel("Measuring scan responses", 9, 9))
    assert widget.current.value() == widget.current.maximum() == 9
    assert widget.overall.value() == 0 and widget.displayed.status == "running"
    before = len(messages)
    worker._report_progress(0, 0, LocalProgressLabel("Transporting wave", 1, 100))
    assert len(messages) == before  # Legacy 0/0 cannot bypass UI throttling.
    worker._report_progress(2, None, "Adaptive boundaries")
    assert messages[-1] == "Adaptive boundaries"
    assert widget.displayed.completed == 2 and widget.displayed.total is None
    widget.set_progress(worker.progress_state.terminal("cancelled"))
    assert widget.overall.value() == 0 and widget.displayed.status == "cancelled"
