"""Steering UI lifecycle: explicit execution, stale/cancel guards, atomic publication."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.beam_steering import BeamSteeringControls
from temsim.optics.column import default_state
from temsim.physics import beam_steering_calibration as physics


@pytest.fixture
def view(qtbot, monkeypatch):
    widget = BeamSteeringControls()
    qtbot.addWidget(widget)
    widget.set_state(default_state())
    widget.resize(680, 420)
    widget.show()
    # The numerical solver is covered by its own tests. This fixture keeps
    # callbacks under deterministic control to reproduce late Qt delivery.
    jobs = []
    monkeypatch.setattr(widget.pool, "start", jobs.append)
    monkeypatch.setattr(widget.pool, "clear", lambda: None)
    widget.jobs = jobs
    return widget


def _complete_calibration(view):
    view.calibrate_button.click()
    worker = view.jobs[-1]
    calibration = SimpleNamespace(condition_number=2., backend="test receipt")
    calibrator = SimpleNamespace(reference=SimpleNamespace(provenance="Recorded electron 7"))
    view._ready(worker.generation, calibrator, calibration, .1)
    view._finished(worker.generation)
    return calibrator


def _solution(state):
    reference = physics.reference_from_values(
        state, z_mm=1019., phase_space=(0.,)*4, kinetic_energy_ev=300000.,
        flight_time_s=0., provenance="GUI lifecycle fixture, not a physical validation",
    )
    return physics.BeamSteeringSolution(
        input_identity=physics.beam_steering_state_identity(state),
        target_z_mm=state.sample.z_mm, baseline=(0.,)*4,
        target=(0., 0., .001, 0.), achieved=(0., 0., .001, 0.), residual=(0.,)*4,
        initial_drives_mrad=tuple(getattr(state.beam_deflector, key) for key in physics.DRIVE_FIELDS),
        drives_mrad=(.1, .2, -.3, .4), condition_number=2., iterations=1,
        backend="test receipt", reference=reference,
        position_tolerance_m=1e-11, slope_tolerance=1e-9,
    )


def test_selection_and_parameter_entry_never_start_a_calculation(view):
    view.set_kind("shift")
    view.x.setValue(2.)
    view.y.setValue(-3.)
    view.reference_z.setValue(view.reference_z.value()-.1)
    assert view.jobs == []
    assert not view.apply_button.isEnabled()
    assert view.x.suffix() == " nm"
    assert view.calibrate_button.property("calculationAction") is True


def test_reference_and_live_optics_edits_invalidate_calibration(view):
    _complete_calibration(view)
    assert view.apply_button.isEnabled()
    view.reference_z.setValue(view.reference_z.value()-.1)
    assert not view.apply_button.isEnabled()
    _complete_calibration(view)
    view.set_revision(1)
    assert view._calibrator is None
    assert not view.apply_button.isEnabled()


def test_cancel_and_newer_job_reject_late_success(view):
    view.calibrate_button.click()
    old = view.jobs[-1]
    view.cancel_button.click()
    assert old.cancellation.is_set()
    view.calibrate_button.click()
    current = view.jobs[-1]
    view._ready(old.generation, object(), object(), .1)
    view._finished(old.generation)
    assert view._worker is current
    assert view._calibrator is None
    view.cancel()


def test_success_applies_four_drives_once_and_publishes_existing_runtime_signal(view):
    calibrator = _complete_calibration(view)
    signals = []
    view.runtime_changed.connect(signals.append)
    dph = next(c for c in view._state.corrector_elements if c.key == "probe_dph2_deflector")
    original_dph = (dph.kick_x_mrad, dph.kick_y_mrad)
    solution = _solution(view._state)
    view.x.setValue(1.)
    view.apply_button.click()
    worker = view.jobs[-1]
    assert worker.target == (.001, 0.)
    view._ready(worker.generation, calibrator, solution, .2)
    view._finished(worker.generation)
    assert tuple(getattr(view._state.beam_deflector, key) for key in physics.DRIVE_FIELDS) == solution.drives_mrad
    assert (dph.kick_x_mrad, dph.kick_y_mrad) == original_dph
    assert signals == ["beam_deflector.calibrated_drives"]
    assert "Applied tilt" in view.status.text()
    assert not view.apply_button.isEnabled()
    view._ready(worker.generation, calibrator, solution, .2)
    assert len(signals) == 1


def test_state_change_during_solve_cannot_apply_old_drives(view):
    calibrator = _complete_calibration(view)
    solution = _solution(view._state)
    view.apply_button.click()
    worker = view.jobs[-1]
    view._state.beam_deflector.upper_x_mrad += .03
    before = tuple(getattr(view._state.beam_deflector, key) for key in physics.DRIVE_FIELDS)
    view._ready(worker.generation, calibrator, solution, .2)
    assert tuple(getattr(view._state.beam_deflector, key) for key in physics.DRIVE_FIELDS) == before
    assert "Inputs changed" in view.status.text()
    assert not view.apply_button.isEnabled()


def test_solver_error_and_submit_error_leave_controls_reusable(view, monkeypatch):
    view.calibrate_button.click()
    worker = view.jobs[-1]
    view._failed(worker.generation, "Reference blocked at aperture")
    view._finished(worker.generation)
    assert "blocked" in view.status.text()
    assert view.calibrate_button.isEnabled()
    assert not view.apply_button.isEnabled()
    monkeypatch.setattr(view.pool, "start", lambda _: (_ for _ in ()).throw(RuntimeError("queue closed")))
    view.calibrate_button.click()
    assert "queue closed" in view.status.text()
    assert view._worker is None
    assert view.calibrate_button.isEnabled()
