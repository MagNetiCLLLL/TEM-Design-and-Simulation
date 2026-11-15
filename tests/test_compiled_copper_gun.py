"""Synthetic field/DG operator parity; no finite-source or wave calculation."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import textwrap
import json

import numpy as np
import pytest

from temsim.magnetic_field_scene import _Source
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
from temsim.physics.frozen_copper_coil_field import FrozenCopperCoilField
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.compiled_copper_instrument import (
    pack_copper_instrument_sources, copper_instrument_batch, copper_near_axis_domain,
)
from temsim.physics.instrument_magnetic import ColumnDipoleField, InstrumentMagneticField
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.relativistic_lorentz import (
    RelativisticPhaseSpace, momentum_from_kinetic_energy_ev, kinetic_energy_ev_from_momentum,
)
from temsim.physics.static_energy_lorentz import static_energy_step


def _copper(**changes):
    coil = AnnularCoilPotential(**dict(dict(source_id="test-coil", inner_radius_m=.02,
        outer_radius_m=.035, length_m=.04, ampere_turns=2700., center_z_m=.5), **changes))
    reverse = replace(coil, source_id="reverse-coil", ampere_turns=-.25*coil.ampere_turns,
                      center_z_m=.65)
    potential = CopperCoilAssemblyPotential("synthetic-copper", (coil, reverse))
    frozen = FrozenCopperCoilField(potential)
    return _Source("synthetic-copper", frozen, frozen.display_bounds_m, frozen.radial_support_m)


def _native():
    field = ColumnDipoleField("synthetic-dipole", .0001, .01, 2e-5, -3e-5,
                             5., 0., 0., 3e-22)
    return _Source(field.key, field, np.array([[-.01, -.01, .0001], [.01, .01, .01]]), .01)


def _instrument(*, copper_first=True, **changes):
    sources = (_copper(**changes), _native())
    if not copper_first:
        sources = sources[::-1]
    return InstrumentMagneticField(sources, None, None)


def _electric():
    r = np.array([0., .001, .01])
    z = np.array([0., .0004, .0008, .002, .01, .1])
    # A small radial term differentiates the same bilinear r-squared table.
    voltage = 3e6*z[None, :] + 2e6*r[:, None]**2
    request = {"schema": "synthetic-dg-field", "high_tension_v": 300000.,
               "domain": {"entrance_m": 0., "exit_m": .1, "gun_exit_m": .1}}
    return ClosedGunField(request, r, z, voltage, {})


def _phase(count=7, *, z=.00055):
    t = np.linspace(-1., 1., count)
    position = np.column_stack((t*2e-8, t[::-1]*3e-8, z+t*2e-6))
    direction = np.column_stack((1e-5+t*2e-6, -2e-5+t*1e-6, np.ones(count)))
    energy = 3e6*position[:, 2]+.3
    return RelativisticPhaseSpace(position, momentum_from_kinetic_energy_ev(energy, direction))


@pytest.mark.parametrize("copper_first", [False, True])
def test_packed_copper_and_native_fields_keep_source_order_and_full_tails(copper_first):
    magnetic = _instrument(copper_first=copper_first)
    packed = pack_copper_instrument_sources(magnetic._sources)
    assert packed is not None
    points = np.array([[1e-8, -2e-8, -.01], [0., 0., .00055],
                       [1e-7, 2e-7, .005], [2e-6, -3e-6, 1.]])
    admitted, result = copper_instrument_batch(points, packed)
    assert admitted
    np.testing.assert_allclose(result, magnetic.field_at_global_positions_t(points), rtol=3e-14, atol=1e-20)
    for value in packed:
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_pose_and_custom_callback_decline_packing():
    magnetic = _instrument(registration=CoordinateRegistration((1e-6, 0., 0.)))
    assert pack_copper_instrument_sources(magnetic._sources) is None
    fake = _Source("custom", SimpleNamespace(field_at_global_positions_t=lambda p: np.zeros_like(p)),
                   np.array([[-1., -1., 0.], [1., 1., 1.]]), 1.)
    assert pack_copper_instrument_sources((_copper(), fake)) is None


@pytest.mark.parametrize("point", [[.0002, 0., .5], [.02, 0., .5], [0., 0., 1e10],
                                   [np.nan, 0., .5]])
def test_elliptic_branch_copper_surface_and_extreme_trials_decline(point):
    packed = pack_copper_instrument_sources((_copper(),))
    points = np.array([point])
    assert not copper_near_axis_domain(points, packed[3])
    assert not copper_instrument_batch(points, packed)[0]


def test_compiled_copper_dg_matches_same_reference_and_static_energy(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    phase, magnetic, electric, dt = _phase(), _instrument(), _electric(), 1e-14
    actual = compiled.try_step(phase, dt, magnetic, electric, 1e-13, 64)
    assert actual is not None
    with monkeypatch.context() as patch:
        patch.setattr(compiled, "try_step", lambda *_a, **_k: None)
        reference = static_energy_step(phase, dt, magnetic, electric, tolerance=1e-13)
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-14, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=2e-12, atol=1e-36)
    before = kinetic_energy_ev_from_momentum(phase.momentum_kg_m_per_s)-electric.potential_rise_v_at_global_positions(phase.position_m)
    after = kinetic_energy_ev_from_momentum(actual.momentum_kg_m_per_s)-electric.potential_rise_v_at_global_positions(actual.position_m)
    np.testing.assert_allclose(after, before, rtol=0., atol=3e-9)
    assert actual.time_s == reference.time_s == dt


def test_trial_that_leaves_near_axis_uses_complete_native_step(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    magnetic, electric = _instrument(center_z_m=.00055), _electric()
    phase = _phase(1)
    # Initial points are safely in the admitted expansion. A transverse chief
    # direction crosses the branch boundary during the trial; no compiled
    # partial update may be applied.
    phase = RelativisticPhaseSpace(phase.position_m,
        momentum_from_kinetic_energy_ev([1e5], [[1., 0., .01]]))
    assert compiled.try_step(phase, 3e-12, magnetic, electric, 1e-13, 64) is None


def test_1024_synthetic_rows_share_one_global_fixed_point_barrier(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    phase, magnetic, electric, dt = _phase(1024), _instrument(), _electric(), 1e-14
    packed = compiled._packed_instrument_field(magnetic)
    assert packed is not None and packed[2] is not None
    native_sources, native_terms, copper = packed
    data = compiled._closed_field_data(electric)
    args = (data, 300000., np.empty(0), True, True, native_sources, native_terms, copper)
    serial = compiled._step(phase.position_m, phase.momentum_kg_m_per_s, dt, 1e-13, 64, *args)
    monkeypatch.setattr("temsim.cpu_resources.numerical_thread_budget", lambda: 4)
    pool = compiled._StepPool()
    try:
        parallel = pool.step(phase.position_m, phase.momentum_kg_m_per_s, dt, 1e-13, 64, *args)
        assert pool.workers == 4
    finally:
        pool.close()
    np.testing.assert_array_equal(parallel[0], serial[0])
    np.testing.assert_array_equal(parallel[1], serial[1])


def test_1024_synthetic_rows_keep_same_nonconvergence_failure(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    phase, magnetic, electric, dt = _phase(1024), _instrument(), _electric(), 1e-13
    native_sources, native_terms, copper = compiled._packed_instrument_field(magnetic)
    args = (compiled._closed_field_data(electric), 300000., np.empty(0), True, True,
            native_sources, native_terms, copper)
    with pytest.raises(ValueError, match="did not converge"):
        compiled._step(phase.position_m, phase.momentum_kg_m_per_s, dt, 1e-13, 1, *args)
    monkeypatch.setattr("temsim.cpu_resources.numerical_thread_budget", lambda: 4)
    pool = compiled._StepPool()
    try:
        with pytest.raises(ValueError, match="did not converge"):
            pool.step(phase.position_m, phase.momentum_kg_m_per_s, dt, 1e-13, 1, *args)
    finally:
        pool.close()


@pytest.mark.parametrize("z,dt", [(0., 1e-15), (.00055, 1e-13), (.000799, 3e-13)])
def test_step_doubling_keeps_accepted_dt_and_four_error_budgets(monkeypatch, z, dt):
    from temsim.physics import grounded_particle_step as compiled
    from temsim.optics.electron_gun.tracing import _surface_step
    magnetic, electric = _instrument(), _electric()
    # At the cathode all entries must start in the same physical half-space.
    if z == 0.:
        phase = RelativisticPhaseSpace(np.column_stack((np.zeros((7, 2)), np.zeros(7))),
            momentum_from_kinetic_energy_ev(np.full(7, .3), np.tile([1e-3, -2e-3, 1.], (7, 1))))
    else:
        phase = _phase(z=z)
    active = np.ones(7, dtype=bool)
    with monkeypatch.context() as patch:
        patch.setattr(compiled, "try_step", lambda *_a, **_k: None)
        patch.setattr(AnnularCoilPotential, "axial_derivatives", AnnularCoilPotential.axial_derivatives_numpy)
        reference, reference_dt = _surface_step(phase, dt, active, magnetic, electric)
    actual, actual_dt = _surface_step(phase, dt, active, magnetic, electric)
    assert actual_dt == reference_dt
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=3e-14, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=3e-12, atol=1e-36)


def test_implicit_comparison_boundary_is_a_native_miss():
    from temsim.physics.grounded_particle_step import _copper_convergence_needs_native
    assert _copper_convergence_needs_native(1e-13, 1e-13)
    assert not _copper_convergence_needs_native(1e-15, 1e-13)


def test_installed_copper_graph_is_packable_without_source_execution():
    from temsim.optics.column import default_state
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    state = default_state()
    state.simulation_mode = "unified"
    state.round_lens_field_model = "copper_coil"
    state.electron_gun.emitter.ray_count = 1024
    magnetic = capture_instrument_magnetic_field(state)
    packed = pack_copper_instrument_sources(magnetic._sources)
    assert packed is not None
    # Field queries are algebraic probes, not original source particles.
    z = np.linspace(0., .45, 37)
    points = np.column_stack((np.full(len(z), 1e-8), np.full(len(z), -2e-8), z))
    admitted, actual = copper_instrument_batch(points, packed)
    assert admitted
    np.testing.assert_allclose(actual, magnetic.field_at_global_positions_t(points), rtol=5e-14, atol=1e-20)


def test_existing_non_copper_instrument_keeps_compiled_reference_parity(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    phase, electric, dt = _phase(), _electric(), 1e-14
    magnetic = InstrumentMagneticField((_native(),), None, None)
    assert compiled._packed_instrument_field(magnetic)[2] is None
    actual = compiled.try_step(phase, dt, magnetic, electric, 1e-13, 64)
    assert actual is not None
    with monkeypatch.context() as patch:
        patch.setattr(compiled, "try_step", lambda *_a, **_k: None)
        reference = static_energy_step(phase, dt, magnetic, electric, tolerance=1e-13)
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=3e-14, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s, rtol=3e-12, atol=1e-36)


@pytest.mark.parametrize("native_miss", [True, False])
def test_only_explicit_copper_native_miss_is_caught(monkeypatch, native_miss):
    from temsim.physics import grounded_particle_step as compiled
    message = compiled._COPPER_NATIVE_FALLBACK if native_miss else "Explicit independent numerical failure"
    def fail(*_args):
        raise ValueError(message)
    monkeypatch.setattr(compiled, "_step", fail)
    if native_miss:
        assert compiled.try_step(_phase(), 1e-14, _instrument(), _electric(), 1e-13, 64) is None
    else:
        with pytest.raises(ValueError, match="independent numerical failure"):
            compiled.try_step(_phase(), 1e-14, _instrument(), _electric(), 1e-13, 64)


def test_unavailable_optional_compiler_retains_native_path(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    monkeypatch.setattr(compiled, "njit", None)
    assert compiled.try_step(_phase(), 1e-14, _instrument(), _electric(), 1e-13, 64) is None


@pytest.mark.parametrize("posed", [False, True])
def test_context_local_dispatch_observation_never_changes_selection_or_arrays(posed):
    from temsim.physics import grounded_particle_step as compiled
    from temsim.physics.gun_step_diagnostics import GunStepDiagnostics, observe_gun_steps
    phase, electric = _phase(), _electric()
    magnetic = (_instrument(registration=CoordinateRegistration((1e-6, 0., 0.)))
                if posed else _instrument())
    before = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
    reference = compiled.try_step(phase, 1e-14, magnetic, electric, 1e-13, 64)
    observer = GunStepDiagnostics()
    with observe_gun_steps(observer):
        actual = compiled.try_step(phase, 1e-14, magnetic, electric, 1e-13, 64)
    if posed:
        assert reference is actual is None
        assert observer.dispatch[("grounded_dg", False, "unsupported_magnetic_law_or_pose")] == 1
    else:
        np.testing.assert_array_equal(actual.position_m, reference.position_m)
        np.testing.assert_array_equal(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s)
        assert actual.time_s == reference.time_s
        assert observer.dispatch[("grounded_dg", True, "")] == 1
        assert observer.dispatch[("grounded_copper_dg", True, "")] == 1
    np.testing.assert_array_equal(phase.position_m, before[0])
    np.testing.assert_array_equal(phase.momentum_kg_m_per_s, before[1])


def test_dispatch_observation_preserves_original_exception_object(monkeypatch):
    from temsim.physics import grounded_particle_step as compiled
    from temsim.physics.gun_step_diagnostics import GunStepDiagnostics, observe_gun_steps
    failure = RuntimeError("Synthetic compilation dispatch failure")
    def fail(*_args):
        raise failure
    monkeypatch.setattr(compiled, "_step", fail)
    observer = GunStepDiagnostics()
    with pytest.raises(RuntimeError) as caught:
        with observe_gun_steps(observer):
            compiled.try_step(_phase(), 1e-14, _instrument(), _electric(), 1e-13, 64)
    assert caught.value is failure
    assert observer.dispatch[("grounded_dg", False, "compiled_error")] == 1
    assert observer.dispatch[("grounded_copper_dg", False, "compiled_error")] == 1


def test_composed_callers_have_no_persistent_dependency_cache():
    from temsim.physics import grounded_particle_step as compiled
    for kernel in (copper_instrument_batch, compiled._step_iteration, compiled._step):
        assert type(kernel._cache).__name__ == "NullCache"


def test_new_process_composed_caller_obeys_current_scalar_range_guard():
    # Reuse the ordinary on-disk scalar cache in two fresh interpreters. Do
    # not clear caches or select a temporary cache directory: composed callers
    # must consult the current helper even when an old native cache exists.
    script = textwrap.dedent('''
        import json,time
        import numpy as np
        from temsim.magnetic_field_scene import _Source
        from temsim.physics.annular_coil_potential import AnnularCoilPotential
        from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
        from temsim.physics.frozen_copper_coil_field import FrozenCopperCoilField
        from temsim.physics.compiled_copper_instrument import pack_copper_instrument_sources,copper_instrument_batch
        from temsim.physics import grounded_particle_step as dg
        c=AnnularCoilPotential('cache-probe',.02,.035,.04,3000.,.5)
        f=FrozenCopperCoilField(CopperCoilAssemblyPotential('cache-owner',(c,)))
        p=pack_copper_instrument_sources((_Source('cache-owner',f,f.display_bounds_m,f.radial_support_m),))
        kinds=[type(k._cache).__name__ for k in (copper_instrument_batch,dg._step_iteration,dg._step)]
        assert kinds==['NullCache']*3,kinds
        start=time.perf_counter()
        supported,_=copper_instrument_batch(np.array([[0.,0.,1000.]]),p)
        assert not supported,'Composed caller bypassed current |u| range guard'
        seconds=time.perf_counter()-start
        ordinary,_=copper_instrument_batch(np.array([[0.,0.,.00055]]),p)
        assert ordinary
        print(json.dumps({'range_rejected':True,'ordinary_admitted':True,'cache_kinds':kinds,'first_call_seconds':seconds}))
    ''')
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=60, check=True)
        row = json.loads(result.stdout.splitlines()[-1])
        assert row["range_rejected"] and row["ordinary_admitted"]
