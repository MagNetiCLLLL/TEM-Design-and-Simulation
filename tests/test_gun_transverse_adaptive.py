"""Active-ray DG accuracy and deterministic retry reuse, with analytical fields."""
from types import SimpleNamespace
import numpy as np
import pytest
from temsim.optics.electron_gun import tracing
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev,
    kinetic_energy_ev_from_momentum, ELEMENTARY_CHARGE_C,
    ELECTRON_MASS_KG, SPEED_OF_LIGHT_M_PER_S,
)


def phase(slopes):
    return RelativisticPhaseSpace(np.zeros((len(slopes), 3)),
        momentum_from_kinetic_energy_ev(np.full(len(slopes), 300000.),
            np.array([[s, 0., 1.] for s in slopes])))


def fields(electric_strength=0., magnetic_strength=0.):
    electric = SimpleNamespace(
        potential_v_at_global_positions=lambda x: electric_strength*x[:, 2],
        field_at_global_positions_v_per_m=lambda x:
            np.tile([0., 0., -electric_strength], (len(x), 1)))
    magnetic = SimpleNamespace(field_at_global_positions_t=lambda x:
        np.tile([0., 0., magnetic_strength], (len(x), 1)))
    return magnetic, electric


def full_and_fine(initial, dt, magnetic, electric):
    from temsim.physics.static_energy_lorentz import static_energy_step
    full = static_energy_step(initial, dt, magnetic, electric)
    fine = static_energy_step(static_energy_step(initial, dt/2, magnetic, electric),
                              dt/2, magnetic, electric)
    return full, fine


def stock_gates_accept(initial, full, fine):
    p = fine.momentum_kg_m_per_s
    pscale = np.maximum(np.linalg.norm(p, axis=1), 1e-30)
    return (np.all(np.linalg.norm(p-full.momentum_kg_m_per_s, axis=1)/pscale <= 1e-6)
        and np.max(np.linalg.norm(fine.position_m-full.position_m, axis=1)) <=
        max(1e-13, np.max(np.linalg.norm(fine.position_m-initial.position_m, axis=1))*1e-5))


def test_small_angle_helix_requires_xy_control_even_when_total_gates_pass():
    initial = phase([1e-6]); magnetic, electric = fields(magnetic_strength=1.)
    p = initial.momentum_kg_m_per_s[0]
    gamma = np.sqrt(1+np.sum(p*p)/(ELECTRON_MASS_KG*SPEED_OF_LIGHT_M_PER_S)**2)
    omega = ELEMENTARY_CHARGE_C/(gamma*ELECTRON_MASS_KG)
    dt = .3/omega
    full, old = full_and_fine(initial, dt, magnetic, electric)
    assert stock_gates_accept(initial, full, old)
    actual, used = tracing._surface_step(initial, dt, np.array([True]), magnetic, electric)
    def error(result, elapsed):
        expected = p[0]*np.array([np.cos(omega*elapsed), np.sin(omega*elapsed)])
        return np.linalg.norm(result.momentum_kg_m_per_s[0, :2]-expected)/abs(p[0])
    assert used == dt/4
    assert error(old, dt) > 1e-4
    assert error(actual, used) < error(old, dt)/20
    np.testing.assert_allclose(kinetic_energy_ev_from_momentum(actual.momentum_kg_m_per_s),
                               300000., rtol=0, atol=1e-7)


def test_an_active_off_axis_ray_cannot_lend_its_xy_allowance_to_the_axis_ray():
    magnetic, electric = fields(electric_strength=1e10)
    initial = phase([1e-4, 1e-4]); initial.position_m[1, 0] = 1e-3
    dt = 1e-13; full, fine = full_and_fine(initial, dt, magnetic, electric)
    error = np.linalg.norm(fine.position_m[:, :2]-full.position_m[:, :2], axis=1)
    budget = 1e-13+1e-6*np.linalg.norm(fine.position_m[:, :2], axis=1)
    assert stock_gates_accept(initial, full, fine)
    assert error[0] > budget[0]
    assert np.max(error) < np.max(budget)  # Incorrect cross-ray max/max would accept.
    actual, used = tracing._surface_step(initial, dt, np.array([True, True]), magnetic, electric)
    alone, alone_dt = tracing._surface_step(phase([1e-4]), dt, np.array([True]), magnetic, electric)
    assert used < dt and used == alone_dt
    np.testing.assert_array_equal(actual.position_m[0], alone.position_m[0])
    np.testing.assert_array_equal(actual.momentum_kg_m_per_s[0], alone.momentum_kg_m_per_s[0])


def test_blocked_ray_cannot_change_the_active_budget_or_readonly_input():
    initial = phase([1e-4, .8]); magnetic, electric = fields(electric_strength=1e10)
    before = initial.position_m.copy(), initial.momentum_kg_m_per_s.copy()
    initial.position_m.flags.writeable = False
    initial.momentum_kg_m_per_s.flags.writeable = False
    actual, used = tracing._surface_step(initial, 1e-13, np.array([True, False]), magnetic, electric)
    alone, alone_dt = tracing._surface_step(phase([1e-4]), 1e-13, np.array([True]), magnetic, electric)
    assert used == alone_dt
    for original, expected, result, single in zip(
            (initial.position_m, initial.momentum_kg_m_per_s), before,
            (actual.position_m, actual.momentum_kg_m_per_s),
            (alone.position_m, alone.momentum_kg_m_per_s)):
        np.testing.assert_array_equal(original, expected)
        np.testing.assert_array_equal(result[0], single[0])
        np.testing.assert_array_equal(result[1], expected[1])
        assert not np.shares_memory(original, result)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('field', ['position_m', 'momentum_kg_m_per_s'])
def test_nonfinite_error_scale_or_budget_never_accepts(monkeypatch, bad_value, field):
    import temsim.physics.static_energy_lorentz as solver
    def invalid(local, dt, *args, **kwargs):
        values = dict(position_m=local.position_m.copy(),
                      momentum_kg_m_per_s=local.momentum_kg_m_per_s.copy())
        values[field][0, 0] = bad_value
        return SimpleNamespace(**values)
    monkeypatch.setattr(solver, 'static_energy_step', invalid)
    magnetic, electric = fields()
    with np.errstate(invalid='ignore', over='ignore'):
        with pytest.raises(ValueError, match='transverse/local error budget'):
            tracing._surface_step(phase([1e-6]), 1e-12, np.array([True]), magnetic, electric)


def uncached_comparable_reference(phase, dt, active, magnetic, electric):
    """Independent no-cache retry reference for finite physical-field fixtures."""
    from temsim.physics.static_energy_lorentz import static_energy_step
    local = RelativisticPhaseSpace(phase.position_m[active], phase.momentum_kg_m_per_s[active], phase.time_s)
    for _ in range(32):
        try:
            full = static_energy_step(local, dt, magnetic, electric, tolerance=tracing.GEOMETRY_IMPLICIT_TOLERANCE)
            half = static_energy_step(local, dt/2, magnetic, electric, tolerance=tracing.GEOMETRY_IMPLICIT_TOLERANCE)
            fine = static_energy_step(half, dt/2, magnetic, electric, tolerance=tracing.GEOMETRY_IMPLICIT_TOLERANCE)
        except ValueError as error:
            if 'iteration did not converge' not in str(error): raise
            dt /= 2
            continue
        p = fine.momentum_kg_m_per_s
        scale = np.maximum(np.linalg.norm(p[:, :2], axis=1), np.linalg.norm(p, axis=1)*1e-5)
        if (stock_gates_accept(local, full, fine)
                and np.all(np.linalg.norm((p-full.momentum_kg_m_per_s)[:, :2], axis=1)/scale <= 1e-5)
                and np.all(np.linalg.norm((fine.position_m-full.position_m)[:, :2], axis=1) <=
                           1e-13+1e-6*np.linalg.norm(fine.position_m[:, :2], axis=1))):
            x, p_all = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
            x[active], p_all[active] = fine.position_m, p
            return RelativisticPhaseSpace(x, p_all, phase.time_s+dt), dt
        dt /= 2
    raise AssertionError('Finite reference fixture did not converge')


@pytest.mark.parametrize('force_iteration_failure', [False, True])
def test_half_retry_reuse_is_bitwise_equal_and_invalidated_by_implicit_failure(monkeypatch, force_iteration_failure):
    import temsim.physics.static_energy_lorentz as solver
    magnetic, electric = fields(electric_strength=1e10)
    original = solver.static_energy_step; results = []; counts = []
    for advance in (uncached_comparable_reference, tracing._surface_step):
        calls = []; failed = []
        def observed(local, dt, *args, **kwargs):
            calls.append((local.time_s, dt))
            if force_iteration_failure and not failed and local.time_s == 0 and dt == 2.5e-14:
                failed.append(True)
                raise ValueError('Discrete-gradient Lorentz iteration did not converge')
            return original(local, dt, *args, **kwargs)
        monkeypatch.setattr(solver, 'static_energy_step', observed)
        results.append(advance(phase([1e-4, .8]), 1e-13, np.array([True, False]), magnetic, electric))
        counts.append(len(calls))
        assert bool(failed) == force_iteration_failure
    assert results[0][1] == results[1][1]
    assert results[0][0].time_s == results[1][0].time_s
    np.testing.assert_array_equal(results[0][0].position_m, results[1][0].position_m)
    np.testing.assert_array_equal(results[0][0].momentum_kg_m_per_s, results[1][0].momentum_kg_m_per_s)
    assert counts[1] < counts[0]


def test_nonconvergence_is_the_only_retryable_implicit_failure(monkeypatch):
    import temsim.physics.static_energy_lorentz as solver
    def outside(*args, **kwargs): raise ValueError('Outside the physical field domain')
    monkeypatch.setattr(solver, 'static_energy_step', outside)
    with pytest.raises(ValueError, match='Outside the physical field domain'):
        tracing._surface_step(phase([1e-6]), 1e-12, np.array([True]), *fields())
