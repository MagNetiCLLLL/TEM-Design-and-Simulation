"""Mock DG decisions and exact root reuse; no source emission or wave run."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun import tracing
from temsim.physics.gun_step_control import SurfaceStepPredictor
from temsim.physics.gun_step_diagnostics import GunStepDiagnostics, observe_gun_steps
from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace


def initial_phase():
    return RelativisticPhaseSpace(np.zeros((2, 3)), np.tile([0., 0., 1.], (2, 1)))


def install_cubic_operator(monkeypatch, gate):
    from temsim.physics import static_energy_lorentz
    calls = []

    def step(phase, dt, *_fields, **kwargs):
        calls.append((phase.time_s, dt, kwargs["tolerance"]))
        xyz, momentum = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
        xyz[:, 2] += dt
        if gate == "total_momentum":
            momentum[:, 2] += dt**3
        elif gate == "transverse_momentum":
            momentum[:, 0] += dt**3
        elif gate == "xyz_position":
            xyz[:, 2] += dt**3
        elif gate == "xy_position":
            xyz[:, 0] += dt**3
        return RelativisticPhaseSpace(xyz, momentum, phase.time_s+dt)

    monkeypatch.setattr(static_energy_lorentz, "static_energy_step", step)
    return calls


@pytest.mark.parametrize("gate", ["total_momentum", "transverse_momentum", "xyz_position", "xy_position"])
def test_predictor_cannot_bypass_any_original_gate(monkeypatch, gate):
    calls = install_cubic_operator(monkeypatch, gate)
    phase = initial_phase()
    original = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
    diagnostic = GunStepDiagnostics()
    active = np.array([True, False])
    with observe_gun_steps(diagnostic):
        diagnostic.begin_trace([4, 9], [.5, .5], phase, [1., 1.])
        advanced, used = tracing._surface_step(phase, .2, active, object(), object(),
            predictor=SurfaceStepPredictor())
    assert used < .2
    assert diagnostic.counts["failed_gate_"+gate] > 0
    assert diagnostic.accepted_steps == 1
    assert all(row["all_finite"] and row["maximum_ratio"] <= 1.
               for row in diagnostic.attempts[-1]["gates"].values())
    assert diagnostic.attempts[0]["trial_proposal"]["physics_cap_s"] == .2
    assert all(row[2] == 1e-13 for row in calls)
    np.testing.assert_array_equal(phase.position_m, original[0])
    np.testing.assert_array_equal(phase.momentum_kg_m_per_s, original[1])
    np.testing.assert_array_equal(advanced.position_m[1], original[0][1])
    np.testing.assert_array_equal(advanced.momentum_kg_m_per_s[1], original[1][1])


def test_memory_avoids_reset_halving_without_changing_acceptance(monkeypatch):
    install_cubic_operator(monkeypatch, "transverse_momentum")
    summaries = []
    for enabled in (False, True):
        phase, diagnostic = initial_phase(), GunStepDiagnostics()
        predictor = SurfaceStepPredictor() if enabled else None
        active = np.array([True, True])
        magnetic, electric = object(), object()
        with observe_gun_steps(diagnostic):
            diagnostic.begin_trace([4, 9], [.5, .5], phase, [1., 1.])
            for _ in range(60):
                phase, used = tracing._surface_step(phase, .01, active, magnetic, electric,
                    predictor=predictor)
                assert 0. < used <= .01
                assert all(row["maximum_ratio"] <= 1.
                           for row in diagnostic.attempts[-1]["gates"].values())
        summaries.append(diagnostic)
    assert summaries[0].accepted_steps == summaries[1].accepted_steps == 60
    assert summaries[0].counts["gate_rejections"] == 300
    assert summaries[1].counts["gate_rejections"] < 20
    assert summaries[1].counts["static_step_calls"] < summaries[0].counts["static_step_calls"]/2
    # Predictor guidance remains independent of the original halving cache.
    assert summaries[1].counts["cached_full_uses"] == summaries[1].counts["gate_rejections"]


def static_root_fields():
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.instrument_magnetic import InstrumentMagneticField
    request = {"schema": "array-only-root-cache", "high_tension_v": 300000.,
               "domain": {"entrance_m": 0., "exit_m": 1., "gun_exit_m": 1.}}
    electric = ClosedGunField(request, np.array([0., 1.]), np.array([0., 1.]),
                              np.zeros((2, 2)), {})
    return electric, InstrumentMagneticField((), None, None)


def test_root_exact_fraction_memoization_preserves_executed_phase(monkeypatch):
    from temsim.physics import static_energy_lorentz
    electric, magnetic = static_root_fields()
    calls = []

    def drift(start, duration, *_fields, **kwargs):
        calls.append((start.time_s, duration, kwargs["tolerance"]))
        xyz = start.position_m.copy()
        xyz[:, 2] += duration
        return RelativisticPhaseSpace(xyz, start.momentum_kg_m_per_s.copy(), start.time_s+duration)

    monkeypatch.setattr(static_energy_lorentz, "static_energy_step", drift)
    arguments = (.8, np.zeros(3), np.array([0., 0., 1.]), 0., 1., electric, magnetic)
    actual = tracing._surface_plane_crossing(*arguments)
    cached_calls = list(calls)
    calls.clear()
    monkeypatch.setattr(tracing, "_static_surface_root_cache_supported", lambda *_: False)
    reference = tracing._surface_plane_crossing(*arguments)
    np.testing.assert_array_equal(actual.position_m, reference.position_m)
    np.testing.assert_array_equal(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s)
    assert actual.time_s == reference.time_s
    assert len(cached_calls) < len(calls)
    assert all(row[2] == 1e-13 for row in calls+cached_calls)


def test_root_repeated_evaluations_retained_for_unknown_or_overridden_provider():
    electric, magnetic = static_root_fields()
    assert tracing._static_surface_root_cache_supported(electric, magnetic)
    assert not tracing._static_surface_root_cache_supported(electric, SimpleNamespace())
    electric.field_at_global_positions_v_per_m = lambda xyz: np.zeros_like(xyz)
    assert not tracing._static_surface_root_cache_supported(electric, magnetic)


def test_root_unknown_inner_provider_keeps_the_original_repeated_evaluation():
    from temsim.physics.instrument_magnetic import InstrumentMagneticField
    electric, _ = static_root_fields()
    inner = SimpleNamespace(field_at_global_positions_t=lambda xyz: np.zeros_like(xyz))
    source = SimpleNamespace(provider=inner, known_zero=False,
                             bounds_m=np.array([[-1., -1., -1.], [1., 1., 1.]]))
    magnetic = InstrumentMagneticField((source,), None, None)
    assert not tracing._static_surface_root_cache_supported(electric, magnetic)


def test_root_native_inner_callback_override_is_not_statically_admitted():
    from temsim.magnetic_field_scene import _Source
    from temsim.physics.instrument_magnetic import ColumnDipoleField, InstrumentMagneticField
    electric, _ = static_root_fields()
    provider = ColumnDipoleField("array-only", 0., .01, 1e-5, 2e-5, 5., 0., 0., 3e-22)
    source = _Source(provider.key, provider,
                     np.array([[-.01, -.01, 0.], [.01, .01, .01]]), .01)
    magnetic = InstrumentMagneticField((source,), None, None)
    assert tracing._static_surface_root_cache_supported(electric, magnetic)
    object.__setattr__(provider, "field_at_global_positions_t", lambda xyz: np.zeros_like(xyz))
    assert not tracing._static_surface_root_cache_supported(electric, magnetic)
