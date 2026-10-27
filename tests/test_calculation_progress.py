"""Current work, completed stages and terminal publication are separate facts."""
from dataclasses import FrozenInstanceError, replace
from threading import Event
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QObject, Signal

from temsim.calculation_progress import (
    CalculationProgress, StageProgress, StageProgressLabel, bounded_bar_counts,
    callback_progress,
)
from temsim.gui.calculation_progress import CalculationProgressWidget


def initial(**kwargs):
    return CalculationProgress("request-1", "Particles", "Preparing", **kwargs)


def test_local_completion_does_not_invent_overall_fraction():
    update = callback_progress(initial(), 4000, 4000, "Gun checkpoint")
    assert (update.completed, update.total) == (4000, 4000)
    assert update.stages_total is None
    assert update.active
    next_step = callback_progress(update, 0, 0, "Column solve")
    assert next_step.completed == 0 and next_step.total is None
    assert next_step.stages_total is None
    adaptive = callback_progress(next_step, 42, 0, "Adaptive accepted steps")
    assert adaptive.completed == 42 and adaptive.total is None


def test_declared_stages_preserve_local_counts_and_reserve_publication():
    label = StageProgressLabel("EDS photons", StageProgress(1, 4, "EDS", 25, 100))
    update = callback_progress(initial(), 12500, 40000, label, reserved_stages=1)
    assert isinstance(label, str)
    assert (update.completed, update.total) == (25, 100)
    assert (update.stages_completed, update.stages_total) == (1, 5)
    done = StageProgressLabel("Complete", StageProgress(4, 4, "Publication"))
    update = callback_progress(update, 40000, 40000, done, reserved_stages=1)
    assert update.active and update.stages_completed < update.stages_total
    accepted = update.terminal("succeeded", "Accepted result")
    assert accepted.stages_completed == accepted.stages_total == 5
    with pytest.raises(FrozenInstanceError):
        accepted.status = "running"
    with pytest.raises(AttributeError):
        label.progress = None


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_terminal_failure_never_completes_remaining_stages(status):
    update = initial(stages_completed=2, stages_total=5).terminal(status)
    assert update.stages_completed == 2
    assert update.stages_total == 5
    assert not update.active


def test_arbitrary_integer_counters_do_not_overflow_or_round_to_complete():
    assert bounded_bar_counts(10**100 - 1, 10**100) == (999999, 1000000)
    assert bounded_bar_counts(10**100, 10**100) == (1000000, 1000000)


def test_widget_unknown_step_replaces_old_fraction_and_terminal_stops_animation(qtbot):
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    known = initial(completed=8, total=10, stages_completed=1, stages_total=4)
    widget.set_progress(known)
    assert widget.current.value() == 8
    unknown = callback_progress(known, 0, 0, "Adaptive refinement")
    widget.set_progress(unknown)
    assert widget.current.maximum() == 0
    assert "total unknown" in widget.current_label.text()
    assert widget.overall.value() == 1
    assert "1/4 stages complete" in widget.overall_label.text()
    widget.set_progress(unknown.terminal("failed", "Memory limit"))
    assert widget.current.maximum() == 1
    assert widget.overall.value() == 1
    assert "Failed" in widget.overall_label.text()
    widget.set_progress(known)
    assert widget.displayed.status == "failed"


def test_widget_completed_request_does_not_hide_other_active_request(qtbot):
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    first = initial(stages_completed=1, stages_total=3)
    second = replace(first, operation_id="request-2", step="Other work")
    widget.set_progress(first)
    widget.set_progress(second)
    assert "2 jobs active" in widget.current_label.text()
    widget.set_progress(second.terminal("succeeded"))
    assert widget.displayed.operation_id == first.operation_id
    assert widget.displayed.active


def test_new_request_does_not_reuse_prior_workflow_stage_total(qtbot):
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    old = initial(stages_completed=4, stages_total=4).terminal("succeeded")
    widget.set_progress(old)
    new = CalculationProgress("next-request", "Coherent beam", "Gun")
    new = callback_progress(new, 7, 10, "Source quadrature")
    widget.set_progress(new)
    assert widget.overall.maximum() == 0
    assert "Overall: progress unknown" == widget.overall_label.text()
    assert widget.current.value() == 7
    widget.set_progress(new.terminal("succeeded", "Accepted result"))
    assert widget.overall.value() == widget.overall.maximum() == 1
    assert widget.overall_label.text() == "Overall: Complete | Succeeded"


def test_widget_pair_child_is_not_counted_as_a_second_job(qtbot):
    widget = CalculationProgressWidget()
    qtbot.addWidget(widget)
    pair = initial(stages_completed=1, stages_total=4)
    child = replace(pair, operation_id="wave-child", parent_id=pair.operation_id)
    widget.set_progress(pair)
    widget.set_progress(child)
    assert widget.displayed.operation_id == pair.operation_id
    assert "2 jobs" not in widget.current_label.text()


def test_controller_acceptance_is_the_only_success_boundary(qtbot, monkeypatch):
    from temsim.gui.calculation_controller import CalculationController
    controller = CalculationController()
    updates = []
    controller.progress_detail.connect(updates.append)
    generation = controller._begin_request("High accuracy")
    running = callback_progress(updates[-1], 1, 1, "Gun complete")
    controller._accept_progress_detail(generation, running)
    assert updates[-1].active and updates[-1].stages_total is None
    monkeypatch.setattr(controller, "_cache_result", lambda result: None)
    controller._accept_result(generation, "High accuracy", SimpleNamespace(), 0.)
    assert updates[-1].status == "succeeded"
    controller._accept_finished(generation, "High accuracy")
    assert updates[-1].status == "succeeded"


def test_controller_rejected_capture_reports_failure_without_running_worker(qtbot):
    from temsim.gui.calculation_controller import CalculationController
    controller = CalculationController()
    updates = []
    controller.progress_detail.connect(updates.append)
    controller.report_rejected_request("High accuracy", "Unsupported source")
    assert len(updates) == 1
    assert updates[-1].status == "failed"
    assert updates[-1].stages_completed == 0 and updates[-1].stages_total == 1
    assert not controller._requests


@pytest.mark.parametrize("status", ["failed", "cancelled"])
def test_controller_cancellation_and_failure_suppress_late_detail(qtbot, status):
    from temsim.gui.calculation_controller import CalculationController
    controller = CalculationController()
    updates = []
    controller.progress_detail.connect(updates.append)
    generation = controller._begin_request("High accuracy")
    running = replace(updates[-1], stages_completed=2, stages_total=6)
    controller._accept_progress_detail(generation, running)
    if status == "failed":
        controller._accept_error(generation, "High accuracy", "Rejected field")
    else:
        controller.invalidate_pending(include_explicit=True)
    count = len(updates)
    controller._accept_progress_detail(generation, running)
    controller._accept_finished(generation, "High accuracy")
    assert len(updates) == count
    assert updates[-1].status == status
    assert updates[-1].stages_completed == 2


def test_transport_callback_reports_real_step_counts_without_parsing_text():
    from temsim.physics.transport_progress import transport_progress, report_transport_progress
    from temsim.simulation_pipeline import _StageProgress
    updates = []
    progress = _StageProgress(["Column", "Readout"], lambda *args: updates.append(args))
    with transport_progress(progress.detail):
        report_transport_progress("Physical column", completed=71, total=123)
    done, total, label = updates[-1]
    assert label.progress.completed == 0
    assert label.progress.total == 2
    assert (label.progress.local_completed, label.progress.local_total) == (71, 123)
    assert "71" not in str(label)  # Counters came from metadata, never text parsing.


def test_wave_worker_keeps_readout_and_publication_after_solver_completion(monkeypatch):
    import temsim.gui.coherent_beam as module
    monkeypatch.setattr(module, "_wave_resource_claim", lambda *_args: None)
    request = SimpleNamespace(observation_z_mm=12.)
    worker = module._WaveWorker(1, 1, object(), request, Event())
    updates = []
    worker.signals.progress_detail.connect(lambda _generation, value: updates.append(value))
    worker._report_progress(10, 10, "Gun complete")
    assert updates[-1].completed == updates[-1].total == 10
    assert updates[-1].stages_completed == 0 and updates[-1].stages_total == 3
    worker._report_stage(1, "Readout")
    assert updates[-1].total is None and updates[-1].stages_completed == 1
    worker._report_stage(2, "Publishing")
    assert updates[-1].active and updates[-1].stages_completed == 2


class _Calculations(QObject):
    result_ready = Signal(str, object, float)
    failed = Signal(str, str)
    progress_changed = Signal(str, int, int, str)
    progress_detail = Signal(object)

    def __init__(self):
        super().__init__()
        self.pool = SimpleNamespace(coordinator=SimpleNamespace(register_retained=lambda *_args: None))

    def invalidate_pending(self, **_kwargs):
        pass


@pytest.mark.parametrize("material", [False, True])
def test_pair_maps_wave_readout_publication_and_rejects_other_pair(material):
    from temsim.gui.paired_beam_controller import PairedBeamController
    pair = PairedBeamController(lambda: None, calculations=_Calculations())
    pair.token, pair._stage, pair._material_stage = "pair-1", "wave", material
    base = 1 + int(material)
    pair._pair_progress = CalculationProgress("pair-1", "Paired beams", "Wave",
        stages_completed=base, stages_total=base + 3)
    updates = []
    pair.progress_detail.connect(updates.append)
    wave = CalculationProgress("wave-1", "Wave", "Gun", completed=1, total=1,
        stages_completed=0, stages_total=3, parent_id="pair-1")
    pair.accept_wave_progress(replace(wave, parent_id="old-pair"))
    assert not updates
    pair.accept_wave_progress(wave)
    assert updates[-1].stages_completed == base
    pair.accept_wave_progress(replace(wave, stages_completed=2, step="Publication"))
    assert updates[-1].stages_completed == base + 2
    assert updates[-1].active
    pair.accept_wave_progress(wave.terminal("succeeded"))
    assert updates[-1].stages_completed == updates[-1].stages_total
    assert pair._stage == "complete"
    pair.cancel()
    assert updates[-1].status == "succeeded"


def test_pair_identity_mismatch_keeps_wave_success_but_never_completes_pair():
    from temsim.gui.paired_beam_controller import PairedBeamController
    pair = PairedBeamController(lambda: None, calculations=_Calculations())
    pair.token, pair._stage, pair._material_stage = "pair-1", "wave", False
    pair._pair_progress = CalculationProgress("pair-1", "Paired beams", "Wave",
        stages_completed=1, stages_total=4)
    wave = CalculationProgress("wave-1", "Coherent beam", "Accepted wave",
        stages_completed=3, stages_total=3, parent_id="pair-1", status="succeeded",
        pair_error="Executed wave identity differs from the captured pair")
    pair.accept_wave_progress(wave)
    assert wave.status == "succeeded"
    assert pair._stage == "failed"
    assert pair._pair_progress.stages_completed == 3
    assert pair._pair_progress.stages_total == 4
    assert "coherent result retained" in pair._pair_progress.step
