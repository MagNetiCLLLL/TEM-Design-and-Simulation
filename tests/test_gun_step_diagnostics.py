"""Pure-array DG observation tests; no finite-source gun or wave execution."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun import tracing
from temsim.physics.gun_step_diagnostics import (
    GunStepDiagnostics, current_gun_step_diagnostics, observe_gun_steps,
)
from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace, momentum_from_kinetic_energy_ev


def fixture():
    initial = RelativisticPhaseSpace(np.array([[0., 0., 0.], [1e-8, 0., 1e-7]]),
        momentum_from_kinetic_energy_ev([100., 100.], [[0., 0., 1.], [.01, 0., 1.]]))
    electric = SimpleNamespace(potential_v_at_global_positions=lambda xyz: -1e3*xyz[..., 2],
        field_at_global_positions_v_per_m=lambda xyz: np.broadcast_to([0., 0., 1e3], np.shape(xyz)).copy())
    magnetic = SimpleNamespace(field_at_global_positions_t=lambda xyz: np.zeros_like(xyz))
    return initial, np.array([True, False]), magnetic, electric


@pytest.mark.parametrize("implicit_failure", [False, True])
def test_observation_preserves_dg_calls_output_and_immutable_inputs(monkeypatch, implicit_failure):
    from temsim.physics import static_energy_lorentz
    original = static_energy_lorentz.static_energy_step
    results, calls = [], []
    for observed in (False, True):
        phase, active, magnetic, electric = fixture()
        before = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
        current_calls, failed = [], []
        def wrapped(local, dt, *args, **kwargs):
            current_calls.append((local.time_s, dt))
            if implicit_failure and not failed:
                failed.append(True)
                raise ValueError("Discrete-gradient Lorentz iteration did not converge")
            return original(local, dt, *args, **kwargs)
        monkeypatch.setattr(static_energy_lorentz, "static_energy_step", wrapped)
        if observed:
            diagnostic = GunStepDiagnostics()
            with observe_gun_steps(diagnostic):
                diagnostic.begin_trace([12, 99], [.5, .5], phase, [100., 100.])
                results.append(tracing._surface_step(phase, 1e-15, active, magnetic, electric))
            assert diagnostic.accepted_steps == 1
            assert diagnostic.counts["implicit_failures"] == int(implicit_failure)
            assert diagnostic.counts["static_step_calls"] == len(current_calls)
            assert set(diagnostic.attempts[-1]["gates"]) == {
                "total_momentum", "xyz_position", "transverse_momentum", "xy_position"}
            assert all(row["limiting_original_id"] == 12 for row in diagnostic.attempts[-1]["gates"].values())
            assert all(not a.flags.writeable for a in diagnostic.arrays().values())
            snapshot = diagnostic.snapshots[0]
            np.testing.assert_array_equal(snapshot["start_position_m"], before[0])
            np.testing.assert_array_equal(snapshot["end_position_m"], results[-1][0].position_m)
            json.dumps(diagnostic.report(), allow_nan=False)
        else:
            results.append(tracing._surface_step(phase, 1e-15, active, magnetic, electric))
        calls.append(current_calls)
        np.testing.assert_array_equal(phase.position_m, before[0])
        np.testing.assert_array_equal(phase.momentum_kg_m_per_s, before[1])
    assert calls[0] == calls[1]
    assert results[0][1] == results[1][1]
    assert results[0][0].time_s == results[1][0].time_s
    np.testing.assert_array_equal(results[0][0].position_m, results[1][0].position_m)
    np.testing.assert_array_equal(results[0][0].momentum_kg_m_per_s, results[1][0].momentum_kg_m_per_s)
    assert current_gun_step_diagnostics() is None


def test_nonretryable_error_is_observed_and_raised_without_replacement(monkeypatch):
    from temsim.physics import static_energy_lorentz
    def refused(*args, **kwargs):
        raise ValueError("Outside the physical field domain")
    monkeypatch.setattr(static_energy_lorentz, "static_energy_step", refused)
    diagnostic = GunStepDiagnostics()
    with pytest.raises(ValueError, match="Outside the physical field domain"):
        with observe_gun_steps(diagnostic):
            phase, active, magnetic, electric = fixture()
            tracing._surface_step(phase, 1e-15, active, magnetic, electric)
    assert current_gun_step_diagnostics() is None
    assert diagnostic.counts["nonretryable_step_errors"] == 1
    assert diagnostic.counts["accepted_steps"] == 0
    assert diagnostic.attempts[0]["retryable_implicit_failure"] is False
    json.dumps(diagnostic.report(), allow_nan=False)
