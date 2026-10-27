"""Honest stage/local progress without running physical calculations."""

import pytest

from temsim.simulation_pipeline import _StageProgress, _QT_PROGRESS_SAFE_MAX


def reporter():
    events = []
    progress = _StageProgress(
        ["Column", "EDS", "STEM", "Diagnostics"],
        lambda *event: events.append(event),
    )
    return progress, events


@pytest.mark.parametrize("histories", [1, 15_000, 1_000_000, 4_000_000_000])
def test_history_count_never_weights_eds_more_than_other_stages(histories):
    progress, events = reporter()
    progress.advance()
    progress.update(histories, histories, "Elastic histories complete")
    completed, total, label = events[-1]
    assert completed / total < 0.5
    assert label == "Stage 2/4 | Elastic histories complete | step progress 100.0%"
    assert total <= _QT_PROGRESS_SAFE_MAX
    progress.advance()
    assert events[-1] == (20_000, 40_000, "Stage 3/4 | STEM")


def test_new_local_step_reports_real_counts_without_outer_regression():
    progress, events = reporter()
    progress.advance()
    progress.update(15_000, 15_000, "Elastic histories")
    progress.update(0, 80, "X-ray collection 0/80")
    progress.update(20, 80, "X-ray collection 20/80")
    assert events[-2][2].endswith("X-ray collection 0/80 | step progress 0.0%")
    assert events[-1][2].endswith("X-ray collection 20/80 | step progress 25.0%")
    positions = [done / total for done, total, _ in events]
    assert positions == sorted(positions)
    assert positions[-1] < 0.5


def test_only_finished_pipeline_reports_complete():
    progress, events = reporter()
    progress.report()
    for _ in range(3):
        progress.advance()
    progress.update(1, 1, "Diagnostics substep")
    assert events[-1][0] < events[-1][1]
    assert events[-1][2].startswith("Stage 4/4")
    progress.advance()
    assert events[-1] == (40_000, 40_000, "Complete")


def test_invalid_nested_counts_are_bounded_and_unknown_totals_clear_local_progress():
    progress, events = reporter()
    progress.update(-10, 2, "Before work")
    assert events[-1] == (0, 40_000, "Stage 1/4 | Before work | step progress 0.0%")
    progress.update(10, 2, "After work")
    assert events[-1] == (9_999, 40_000, "Stage 1/4 | After work | step progress 100.0%")
    count = len(events)
    progress.update(0, 0, "Unknown")
    assert len(events) == count + 1
    assert events[-1][2] == "Stage 1/4 | Unknown"
    assert events[-1][2].progress.local_total is None


def test_callback_exceptions_preserve_cancellation_contract():
    def cancel(*_args):
        raise RuntimeError("Cancelled")

    progress = _StageProgress(["EDS"], cancel)
    with pytest.raises(RuntimeError, match="Cancelled"):
        progress.update(1, 2, "Photons")


def test_no_callback_does_not_require_progress_consumers():
    progress = _StageProgress(["EDS"], None)
    progress.report()
    progress.update(1, 2, "Photons")
    progress.advance()


def test_transport_detail_keeps_stage_position_without_a_time_estimate():
    progress, events = reporter()
    progress.update(1, 4, "Earlier substep")
    position = events[-1][:2]
    progress.detail("Electron gun | Z 123 mm | 15000 active")
    assert events[-1][:2] == position
    assert events[-1][2] == "Stage 1/4 | Electron gun | Z 123 mm | 15000 active"
    assert "%" not in events[-1][2]


def test_transport_progress_scope_cleans_up_on_cancellation():
    from temsim.physics.transport_progress import (
        transport_progress, report_transport_progress, transport_progress_active)
    events = []
    def cancel(label):
        raise InterruptedError(label)
    with transport_progress(events.append):
        with pytest.raises(InterruptedError, match="cancel"):
            with transport_progress(cancel):
                report_transport_progress("cancel")
        report_transport_progress("outer restored")
    assert not transport_progress_active()
    report_transport_progress("must not leak to the next job")
    assert events == ["outer restored"]


def test_nested_transport_traces_keep_one_monotonic_local_progress_without_completing_workflow():
    from temsim.physics.transport_progress import (
        transport_progress, transport_progress_segment, report_transport_progress)
    from temsim.calculation_progress import CalculationProgress, callback_progress
    events = []
    with transport_progress(events.append):
        for trace, steps in enumerate((8, 13, 5)):
            with transport_progress_segment(trace, 3, label="Actual scan response"):
                report_transport_progress("Preparing unknown work")
                for step in range(steps+1):
                    report_transport_progress("Column integration", completed=step, total=steps)
                assert events[-1].local_progress.completed < (trace+1)*1_000_000
    counts = [event.local_progress.completed for event in events]
    assert counts == sorted(counts)
    assert {event.local_progress.total for event in events} == {3_000_000}
    assert counts[-1] == 3_000_000
    assert "1/3" in events[0] and "3/3" in events[-1]
    previous = CalculationProgress("job", "Particles", "Prepare", stages_completed=0,
                                   stages_total=4, stage_name="Prepare")
    final = callback_progress(previous, 0, 0, events[-1])
    assert final.stages_completed == 0 and final.stages_total == 4
    assert final.status == "running"


@pytest.mark.parametrize("exception", [InterruptedError, RuntimeError])
def test_nested_transport_failure_does_not_complete_trace_and_restores_parent(exception):
    from temsim.physics.transport_progress import (
        transport_progress, transport_progress_segment, report_transport_progress)
    events = []
    with transport_progress(events.append):
        with pytest.raises(exception):
            with transport_progress_segment(1, 2, label="Response"):
                report_transport_progress("Half done", completed=5, total=10)
                report_transport_progress("Unknown remainder")
                assert events[-1].local_progress.completed == 1_500_000
                raise exception("Stop this trace")
        assert events[-1].local_progress.completed < events[-1].local_progress.total
        report_transport_progress("Outer callback restored")
    assert events[-1] == "Outer callback restored"
    assert not hasattr(events[-1], "local_progress")


def test_progress_segment_composes_without_overwriting_parent_mapping():
    from temsim.physics.transport_progress import (
        transport_progress, transport_progress_segment, report_transport_progress)
    events = []
    with transport_progress(events.append), transport_progress_segment(0, 2, label="Calibration"):
        with transport_progress_segment(1, 4, label="Trace"):
            report_transport_progress("Actual step", completed=1, total=2)
            assert events[-1].local_progress.completed == 375_000
            assert events[-1].local_progress.total == 2_000_000
    assert events[-1].local_progress.completed == 1_000_000


@pytest.mark.parametrize("cancelled", [False, True])
def test_particle_section_reports_scan_calibration_before_starting_source(monkeypatch, cancelled):
    import temsim.simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    from temsim.physics.transport_progress import report_transport_progress, transport_progress_active

    state = default_state()
    state.simulation_mode = "unified"
    events = []
    class ReachedSourcePreparation(RuntimeError):
        pass

    def calibration(_state, **kwargs):
        assert transport_progress_active()
        report_transport_progress("Finite-coil response", completed=7, total=17)
        assert events[-1][2].progress.local_completed == 7
        assert events[-1][2].progress.local_total == 17
        if cancelled:
            raise InterruptedError("Cancelled in scan calibration")
        report_transport_progress("Finite-coil response complete", completed=17, total=17)

    def captured(_state):
        raise ReachedSourcePreparation("Calibration finished before source capture")

    monkeypatch.setattr("temsim.physics.unified_scan_calibration.calibrate_unified_scan", calibration)
    monkeypatch.setattr(pipeline, "capture_external_input_identities", captured)
    with pytest.raises(InterruptedError if cancelled else ReachedSourcePreparation):
        pipeline.calculate_particle_section(state, target_z_mm=state.sample.z_mm,
            acquire_stem=False, progress_callback=lambda *row: events.append(row))
    assert len(events) == (2 if cancelled else 3)
    assert all(row[2].progress.completed == 0 and row[2].progress.total == 4 for row in events)
    assert events[0][2].progress.local_total is None
    assert events[-1][2].progress.local_completed == (7 if cancelled else 17)
    assert not transport_progress_active()


def test_bank_progress_names_point_and_stage_without_global_percentage(qtbot):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QProgressBar
    from temsim.gui.interactive_calculation import InteractiveCalculationPage

    bar = QProgressBar()
    bar.setRange(0, 1000)
    qtbot.addWidget(bar)
    page = SimpleNamespace(progress=bar)
    InteractiveCalculationPage._progress(
        page, 1, 5, 15_000, 30_000,
        "Stage 2/3 | EDS photons 20/80 | step progress 50.0%",
    )
    assert bar.value() == 300
    assert bar.text().startswith("Point 2/5 | Stage 2/3")
    assert "step progress 50.0%" in bar.text()
    assert "%p" not in bar.format()


@pytest.mark.parametrize("eds_cached", [False, True])
def test_pipeline_cached_elastic_and_eds_omit_completed_stages(monkeypatch, eds_cached):
    from types import SimpleNamespace
    import temsim.simulation_pipeline as pipeline
    from temsim.optics.column import default_state
    from temsim.specimen.interaction_types import SpecimenObservable

    state = default_state()
    state.sample.wave_enabled = False
    state.ac_deflector.scan_enabled = False
    state.descan_deflector.scan_enabled = False
    state.energy_filter.enabled = False
    signatures = pipeline.calculation_signatures(state)
    simulation = pipeline.Simulation(incident=SimpleNamespace(), branches={}, metrics={})
    interactions = SimpleNamespace(
        elastic_transport=object(), eds_spectrum=object() if eds_cached else None,
        inelastic_distribution=object(), metrics={}, completed_observables=frozenset(),
        conservation=(object(),),  # Completed ledger; this fixture tests routing only.
    )
    cached = {key: signatures[key] for key in ("incident", "column", "elastic", "energy_filter")}
    if eds_cached:
        cached["eds"] = signatures["eds"]
    existing = pipeline.CalculationResult(
        simulation=simulation, energy_filter=None, specimen_interactions=interactions,
        signatures=cached,
    )
    monkeypatch.setattr(pipeline, "_eds_point_requested", lambda _state: True)
    monkeypatch.setattr(pipeline, "_geometric_specimen_transport_requested", lambda _state: False)
    monkeypatch.setattr(pipeline, "sample_illumination_absent", lambda *_args: False)
    monkeypatch.setattr(pipeline, "retain_specimen_observables", lambda *_args: interactions)
    monkeypatch.setattr("temsim.specimen.elastic_transport.incident_rays_from_simulation",
                        lambda *_args: SimpleNamespace(original_centroid_nm=(0., 0.)))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Cached ray or energy-filter propagation must not run")

    monkeypatch.setattr(pipeline, "run", forbidden)
    monkeypatch.setattr(pipeline, "simulate_energy_filter", forbidden)
    observed = []

    def fake_interactions(_state, _simulation, request, **kwargs):
        if SpecimenObservable.CHARACTERISTIC_X_RAY in request.observables:
            assert not eds_cached
            assert kwargs["existing_result"].elastic_transport is interactions.elastic_transport
            observed.append("EDS")
            callback = kwargs["progress_callback"]
            callback(0, 2, "EDS photons 0/2")
            callback(1, 2, "EDS photons 1/2")
            callback(2, 2, "EDS photons 2/2")
        return interactions

    monkeypatch.setattr(pipeline, "run_specimen_interactions", fake_interactions)
    events = []
    result = pipeline.calculate(state, existing_result=existing,
                                progress_callback=lambda *event: events.append(event))
    assert observed == ([] if eds_cached else ["EDS"])
    assert "elastic" in result.reused_products
    if eds_cached:
        assert "eds" in result.reused_products
    stages = 2 if eds_cached else 3
    assert events[0][2].startswith(f"Stage 1/{stages}")
    assert all("Transporting electrons" not in label for _, _, label in events)
    fractions = [done / total for done, total, _ in events]
    assert fractions == sorted(fractions)
    assert events[-1] == (stages * 10_000, stages * 10_000, "Complete")
    timings = result.performance["stages"]
    assert len(timings) == stages
    assert sum(item["seconds"] for item in timings) == pytest.approx(result.performance["pipeline_seconds"])
    assert all(item["seconds"] >= 0 for item in timings)
    assert all("Tracing the electron column" != item["stage"] for item in timings)
