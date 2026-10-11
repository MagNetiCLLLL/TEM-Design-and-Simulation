"""Actual retained entrance data, independent affine metrics and backend parity."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.actual_probe_diagnostics import (
    affine_position_residual, calculate_actual_probe_diagnostic,
)


def test_weighted_affine_metric_removes_translation_and_general_linear_transport():
    rng = np.random.default_rng(71)
    entrance = rng.normal(size=(100, 4))*[1e-6, 2e-6, 1e-3, 2e-3]
    weights = rng.uniform(.1, 1., 100)
    matrix = np.array([[2., -.3], [.7, 3.], [.004, -.003], [.002, .001]])
    output = entrance@matrix + [3e-5, -2e-5]
    output[-1] = 1e5  # A zero-current support probe cannot dominate the fit.
    weights[-1] = 0
    fit, residual = affine_position_residual(entrance, output, weights, np.ones(100, bool))
    assert fit["rank"] == 4
    assert fit["rms_m"] < 2e-20
    assert np.isnan(residual[-1]).all()


def test_nonlinear_residual_uses_same_survivors_and_declares_rank():
    x = np.linspace(-1., 1., 51)
    source = np.column_stack((x*1e-6, x*2e-6, x*1e-4, np.zeros(51)))
    target = np.column_stack((x*x*1e-9, x*4e-7))
    mask = np.ones(51, bool)
    mask[:7] = False
    target[:7] = np.nan
    fit, _ = affine_position_residual(source, target, np.ones(51), mask)
    independent = np.polyfit(x[mask], target[mask, 0], 1)
    expected = np.sqrt(np.mean((target[mask, 0]-np.polyval(independent, x[mask]))**2))
    assert fit["rank"] == 1
    assert fit["rms_m"] == pytest.approx(expected, rel=1e-12)
    assert "not an aberration" in fit["status"]


@pytest.fixture(scope="module")
def retained_local_result():
    """A small declared integration fixture, not an alleged physical Tip solve."""
    from temsim.calculation_cache import calculation_signatures
    from temsim.optics.column import default_state
    from temsim.physics.core import propagate
    state = default_state()
    state.acceleration_backend = "CPU"
    state.step_mm = state.history_step_mm = .01
    for item in (*state.lenses, *state.stigmators, *state.corrector_elements, *state.deflectors):
        item.enabled = False
    for index, hp in enumerate((state.hp2_hexapole, state.hp1_hexapole)):
        hp.enabled = True
        hp.z_mm = 1200.+index*2
        hp.effective_length_mm = .1
        hp.strength_m3 = (1 if index == 0 else -1)*1e8
    state.sample.z_mm = 1204.
    rng = np.random.default_rng(391)
    entrance = rng.normal(size=(48, 4))*[2e-5, 1e-5, 2e-4, 1e-4]
    entrance[:, 0] += 7e-6
    energy = 300000.+np.linspace(-.3, .3, 48)
    times = 2e-9+np.arange(48)*1e-14
    start, stop = 1199.5, state.sample.z_mm
    output = propagate(state, start, stop, entrance[:, 0], entrance[:, 2],
        entrance[:, 1], entrance[:, 3], initial_kinetic_energy_ev=energy,
        initial_time_s=times, checkpoint_z_mm=(start, stop), return_checkpoints=True,
        save_z_mm=(start, stop), return_flight_times=True)
    cp = output[-1]
    weights = np.linspace(.5, 1.5, 48)/48
    weights[-1] = 0
    branch = SimpleNamespace(alive=np.ones(48, bool), blocked_z=np.full(48, np.nan),
        blocked_key=[""]*48, source_ray_id=np.arange(48)+100, ray_weight=weights,
        x=cp.x_m, y=cp.y_m, tx=cp.tx_rad, ty=cp.ty_rad)
    return SimpleNamespace(state_snapshot=state,
        simulation=SimpleNamespace(incident=branch, incident_checkpoints=cp),
        signatures=calculation_signatures(state))


def test_actual_continuation_preserves_original_entrance_energy_clock_weights_ids(retained_local_result):
    result = retained_local_result
    original = result.simulation.incident_checkpoints
    diagnostic = calculate_actual_probe_diagnostic(result)
    actual = diagnostic.checkpoints
    executed = result.simulation.incident.ray_weight > 0
    for field in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_array_equal(getattr(actual, field)[0, executed], getattr(original, field)[0, executed])
        assert np.isnan(getattr(actual, field)[:, ~executed]).all()
    np.testing.assert_array_equal(diagnostic.weights, result.simulation.incident.ray_weight)
    np.testing.assert_array_equal(diagnostic.source_ray_id, result.simulation.incident.source_ray_id)
    assert len(diagnostic.planes) == 8
    assert diagnostic.planes[-1]["affine"]["rms_m"] > 1e-12
    assert diagnostic.provenance["source_ray_count"] == 48
    assert "reference ring" in diagnostic.scope
    # Same integration law, with new exact observation nodes only.
    error = np.linalg.norm(actual.x_m[-1, executed]-original.x_m[-1, executed])
    assert error < 1e-6*np.linalg.norm(original.x_m[-1, executed])
    assert diagnostic.provenance["retained_sample_difference_rms_m"] is not None
    assert not hasattr(result.state_snapshot, "_tuning_cancelled")


def test_upstream_stopped_unknown_columns_are_not_replaced_or_propagated(retained_local_result):
    from dataclasses import replace
    result = copy(retained_local_result)
    result.simulation = copy(result.simulation)
    result.simulation.incident = branch = copy(result.simulation.incident)
    branch.alive = branch.alive.copy()
    branch.blocked_z = branch.blocked_z.copy()
    branch.blocked_key = list(branch.blocked_key)
    branch.alive[0] = False
    branch.blocked_z[0] = 1199.
    branch.blocked_key[0] = "upstream aperture"
    original = result.simulation.incident_checkpoints
    fields = {}
    for name in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        values = getattr(original, name).copy()
        values[:, [0, -1]] = np.nan  # One stopped particle, one zero-current support column.
        fields[name] = values
    result.simulation.incident_checkpoints = replace(original, **fields)
    diagnostic = calculate_actual_probe_diagnostic(result)
    assert diagnostic.provenance["executed_positive_current_rays"] == 46
    np.testing.assert_array_equal(diagnostic.source_ray_id, branch.source_ray_id)
    np.testing.assert_array_equal(diagnostic.weights, branch.ray_weight)
    assert not diagnostic.alive[:, [0, -1]].any()
    for name in fields:
        assert np.isnan(getattr(diagnostic.checkpoints, name)[:, [0, -1]]).all()
        np.testing.assert_array_equal(getattr(diagnostic.checkpoints, name)[0, 1:-1],
                                      getattr(original, name)[0, 1:-1])
    assert diagnostic.planes[-1]["affine"]["rays"] == 46
    assert branch.blocked_z[0] == 1199.
    assert branch.blocked_key[0] == "upstream aperture"
    # The same missing energy on a surviving positive-current particle is a
    # genuine incomplete boundary and must still fail, not be silently omitted.
    values = fields["kinetic_energy_ev"].copy()
    values[0, 1] = np.nan
    result.simulation.incident_checkpoints = replace(result.simulation.incident_checkpoints,
                                                    kinetic_energy_ev=values)
    with pytest.raises(ValueError, match="finite retained"):
        calculate_actual_probe_diagnostic(result)


def test_actual_gpu_continuation_matches_cpu_without_replacing_source(retained_local_result):
    from temsim.physics.compute_backend import cuda_capability
    from temsim.calculation_cache import calculation_signatures
    if not cuda_capability().available:
        pytest.skip("CUDA device unavailable")
    cpu = calculate_actual_probe_diagnostic(retained_local_result)
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    gpu_result = copy(retained_local_result)
    gpu_result.state_snapshot = decode_instrument(encode_instrument(retained_local_result.state_snapshot))
    gpu_result.state_snapshot.acceleration_backend = "Require GPU"
    gpu_result.signatures = calculation_signatures(gpu_result.state_snapshot)
    gpu = calculate_actual_probe_diagnostic(gpu_result)
    executed = [r for r in gpu.provenance["backend_receipts"] if r["outcome"] == "returned"]
    assert any(r["actual"] == "CUDA GPU" for r in executed)
    assert all(r["actual"] == "CUDA GPU" for r in executed)
    for field in ("x_m", "y_m", "tx_rad", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_allclose(getattr(gpu.checkpoints, field), getattr(cpu.checkpoints, field),
                                   rtol=3e-11, atol=1e-15)
    np.testing.assert_array_equal(gpu.alive, cpu.alive)


def test_changed_capture_is_rejected_without_source_regeneration(retained_local_result):
    from temsim.instrument_snapshot import encode_instrument, decode_instrument
    result = copy(retained_local_result)
    result.state_snapshot = decode_instrument(encode_instrument(retained_local_result.state_snapshot))
    result.state_snapshot.hp2_hexapole.strength_m3 *= .9
    with pytest.raises(ValueError, match="not bound"):
        calculate_actual_probe_diagnostic(result)


def test_cancellation_prevents_diagnostic_work(retained_local_result):
    from temsim.cpu_resources import NumericalJobCancelled
    with pytest.raises(NumericalJobCancelled):
        calculate_actual_probe_diagnostic(retained_local_result, cancelled=lambda: True)


def test_actual_gui_runs_explicitly_and_new_result_invalidates_report(qtbot, retained_local_result):
    from temsim.gui.aberration_view import AberrationComparisonView
    view = AberrationComparisonView(fixed_system="probe")
    qtbot.addWidget(view)
    view.display_result(retained_local_result)
    assert view._actual_report is None
    assert view.actual_button.property("calculationAction") is True
    view.actual_button.click()
    qtbot.waitUntil(lambda: view._actual_worker is None, timeout=30000)
    assert view._actual_report is not None, view.actual_summary.text()
    assert view.actual_table.rowCount() == 8
    assert "captured" in view.actual_summary.text().lower()
    view.display_result(retained_local_result)
    assert view._actual_report is None
    assert view.actual_table.rowCount() == 0
    assert view.shutdown()


def test_cancel_rejects_queued_success_and_start_failure_restores_controls(qtbot, retained_local_result, monkeypatch):
    from temsim.gui.aberration_view import AberrationComparisonView, _ActualWorker
    view = AberrationComparisonView(fixed_system="probe")
    qtbot.addWidget(view)
    view.display_result(retained_local_result)
    worker = _ActualWorker(view._generation, retained_local_result)
    view._actual_worker = worker
    view.cancel_actual_diagnostic()
    # A success queued immediately before Cancel must never be published.
    view._actual_solved(view._generation, object())
    assert view._actual_report is None
    view._actual_finished(view._generation)
    monkeypatch.setattr(view.pool, "start", lambda _worker: (_ for _ in ()).throw(RuntimeError("closed queue")))
    view._start_actual()
    assert view._actual_worker is None
    assert view.actual_button.isEnabled()
    assert not view.actual_cancel.isEnabled()
    assert "closed queue" in view.actual_summary.text()
    assert view.shutdown()
