"""Synthetic DG/backend operators only; no emitted source or column trial."""
from contextlib import contextmanager
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from temsim.physics import cuda_gun_step as cuda_step
from temsim.physics import grounded_particle_step
from temsim.physics.compute_backend import GPUExecutionError
from temsim.physics.relativistic_lorentz import kinetic_energy_ev_from_momentum
from temsim.physics.static_energy_lorentz import static_energy_step
from test_compiled_copper_gun import _electric, _instrument, _phase


class SyntheticFieldSampler:
    """A CPU field double for dispatch tests, never CUDA execution evidence."""
    def __init__(self, magnetic, identity=None):
        self.magnetic = magnetic
        self.calls = []
        self.execution_identity = dict(identity or {"schema": "synthetic-cuda-field", "device": "fixture"})

    def field_at_global_positions_t(self, points):
        self.calls.append(np.asarray(points).copy())
        return self.magnetic.field_at_global_positions_t(points)

    def execution_receipt(self):
        return {"scope": "Synthetic field double; no CUDA execution", "query_count": len(self.calls)}


def binding(sampler, *, cancelled=None):
    return cuda_step.CudaGunBinding(sampler, "Prefer GPU",
        {"schema": "synthetic-cuda-field-binding"}, cancelled)


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CUDA DG operator checks must not emit or transport a source")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


@pytest.fixture
def reference_only(monkeypatch):
    monkeypatch.setattr(grounded_particle_step, "try_step", lambda *_a, **_k: None)


@pytest.mark.parametrize("count", [1, 7, 256])
def test_cuda_dispatch_reuses_identical_native_formulas_and_one_population(count, reference_only, monkeypatch):
    phase, magnetic, electric, dt = _phase(count), _instrument(), _electric(), 1e-14
    reference = static_energy_step(phase, dt, magnetic, electric, tolerance=1e-13)
    sampler = SyntheticFieldSampler(magnetic)
    monkeypatch.setattr(grounded_particle_step, "try_step",
        lambda *_a, **_k: pytest.fail("Explicit CUDA scope reached the CPU compiled dispatcher"))
    with cuda_step.cuda_gun_execution(binding(sampler)):
        actual = static_energy_step(phase, dt, sampler, electric, tolerance=1e-13)
    np.testing.assert_array_equal(actual.position_m, reference.position_m)
    np.testing.assert_array_equal(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s)
    assert actual.time_s == reference.time_s == dt
    assert sampler.calls and all(len(points) == count for points in sampler.calls)
    assert cuda_step.current_cuda_gun_binding() is None


def test_cuda_iteration_uses_original_tolerance_and_iteration_limit(reference_only):
    phase, magnetic, electric = _phase(7), _instrument(), _electric()
    sampler = SyntheticFieldSampler(magnetic)
    for selected in (magnetic, sampler):
        with cuda_step.cuda_gun_execution(None if selected is magnetic else binding(sampler)):
            with pytest.raises(ValueError, match="iteration did not converge"):
                static_energy_step(phase, 1e-13, selected, electric,
                    tolerance=1e-13, maximum_iterations=1)
    assert len(sampler.calls) == 1


def test_binding_cannot_accelerate_an_unrelated_provider(reference_only):
    phase, magnetic, electric = _phase(), _instrument(), _electric()
    sampler = SyntheticFieldSampler(magnetic)
    reference = static_energy_step(phase, 1e-14, magnetic, electric, tolerance=1e-13)
    with cuda_step.cuda_gun_execution(binding(sampler)):
        actual = static_energy_step(phase, 1e-14, magnetic, electric, tolerance=1e-13)
    assert not sampler.calls
    np.testing.assert_array_equal(actual.position_m, reference.position_m)


def test_nested_cpu_scope_masks_and_restores_outer_cuda_binding(reference_only):
    phase, magnetic, electric = _phase(), _instrument(), _electric()
    sampler = SyntheticFieldSampler(magnetic)
    outer = binding(sampler)
    with cuda_step.cuda_gun_execution(outer):
        assert cuda_step.current_cuda_gun_binding() is outer
        with cuda_step.cuda_gun_execution(None):
            assert cuda_step.current_cuda_gun_binding() is None
            static_energy_step(phase, 1e-14, magnetic, electric, tolerance=1e-13)
        assert cuda_step.current_cuda_gun_binding() is outer
    assert cuda_step.current_cuda_gun_binding() is None and not sampler.calls


def test_cancel_before_gpu_sample_never_updates_phase(reference_only):
    phase, magnetic, electric = _phase(), _instrument(), _electric()
    positions, momenta = phase.position_m.copy(), phase.momentum_kg_m_per_s.copy()
    sampler = SyntheticFieldSampler(magnetic)
    with cuda_step.cuda_gun_execution(binding(sampler, cancelled=lambda: True)):
        with pytest.raises(RuntimeError, match="Superseded optical tuning request"):
            static_energy_step(phase, 1e-14, sampler, electric, tolerance=1e-13)
    assert not sampler.calls and cuda_step.current_cuda_gun_binding() is None
    np.testing.assert_array_equal(phase.position_m, positions)
    np.testing.assert_array_equal(phase.momentum_kg_m_per_s, momenta)


def test_cuda_launch_failure_does_not_retry_cpu(reference_only, monkeypatch):
    phase, magnetic, electric = _phase(), _instrument(), _electric()
    sampler = SyntheticFieldSampler(magnetic)
    def launch_failure(_points):
        raise GPUExecutionError("invalid_configuration", "synthetic CUDA launch failure")
    sampler.field_at_global_positions_t = launch_failure
    monkeypatch.setattr(grounded_particle_step, "try_step",
        lambda *_a, **_k: pytest.fail("CUDA failure retried on the CPU"))
    with cuda_step.cuda_gun_execution(binding(sampler)):
        with pytest.raises(GPUExecutionError, match="synthetic CUDA launch failure"):
            static_energy_step(phase, 1e-14, sampler, electric, tolerance=1e-13)
    assert cuda_step.current_cuda_gun_binding() is None


@pytest.mark.parametrize("policy", ["Auto", "CPU", "Numba CPU"])
def test_default_and_cpu_policies_do_not_import_or_prepare_cuda(monkeypatch, policy):
    core = SimpleNamespace(prepare_cuda_instrument_magnetic=lambda *_a, **_k:
        pytest.fail("CPU Gun policy prepared a CUDA adapter"))
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential", core)
    assert cuda_step.prepare_cuda_gun_binding(object(), execution_backend=policy) is None


def test_prepared_binding_preserves_core_execution_identity_and_policy(monkeypatch):
    sampler = SyntheticFieldSampler(_instrument(), {"schema": "fixture-kernel", "device": 2})
    calls = []
    core = SimpleNamespace(prepare_cuda_instrument_magnetic=lambda provider, **kw:
        calls.append((provider, kw)) or sampler)
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential", core)
    provider = object()
    result = cuda_step.prepare_cuda_gun_binding(provider, execution_backend="Prefer GPU")
    assert calls == [(provider, {"execution_backend": "Prefer GPU"})]
    assert result.sampler is sampler and result.execution_backend == "Prefer GPU"
    assert result.execution_identity["magnetic_execution"]["device"] == 2
    assert result.execution_identity["coordinator_implementation"]


@pytest.fixture
def scoped_environment(monkeypatch):
    from temsim.optics.column import default_state
    from temsim.physics import instrument_magnetic, instrument_electric
    from temsim.physics import gun_field_environment as environment
    state = default_state()
    provider = SimpleNamespace(_sources=(),
        identity_for_axial_range=lambda *_: "b"*64,
        field_at_global_positions_t=lambda p: np.zeros_like(p))
    physical_end_mm = instrument_electric.instrument_electric_end_mm(state)
    monkeypatch.setattr(instrument_magnetic, "capture_instrument_magnetic_field", lambda _: provider)
    monkeypatch.setattr(instrument_electric, "instrument_electric_end_mm", lambda _: physical_end_mm)
    monkeypatch.setattr(instrument_electric, "configure_instrument_electric_domain",
        lambda gun, end: setattr(gun, "_instrument_electric_end_mm", end))
    metadata = {"schema": "fixture-cuda-kernel", "device": 0}
    def prepare(*args, **kwargs):
        return SyntheticFieldSampler(provider, metadata)
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential",
        SimpleNamespace(prepare_cuda_instrument_magnetic=prepare))
    return state, provider, metadata, environment


def test_gun_cache_identity_separates_backend_device_and_kernel(scoped_environment):
    state, _, metadata, environment = scoped_environment
    def key():
        with environment.instrument_gun_field_context(state):
            return state.electron_gun._cache_key(1024)
    cpu = key()
    state.acceleration_backend = "Prefer GPU"
    first_gpu = key()
    assert first_gpu != cpu and first_gpu == key()
    metadata["device"] = 1
    another_device = key()
    assert another_device != first_gpu
    metadata["schema"] = "fixture-new-kernel"
    assert key() != another_device
    state.acceleration_enabled = False
    disabled = key()
    state.acceleration_backend = "CPU"
    assert key() == disabled


def test_gpu_gun_context_restores_inputs_on_success_and_exception(scoped_environment):
    state, provider, _, environment = scoped_environment
    state.acceleration_backend = "Prefer GPU"
    gun = state.electron_gun
    sentinel = object()
    gun._instrument_magnetic_field = sentinel
    gun._instrument_magnetic_identity = "old-identity"
    for fail in (False, True):
        @contextmanager
        def exercise():
            try:
                with environment.instrument_gun_field_context(state) as actual:
                    assert actual is not provider and actual.magnetic is provider
                    assert environment.gun_transport_magnetic_field(gun) is actual
                    assert cuda_step.current_cuda_gun_binding().sampler is actual
                    if fail:
                        raise RuntimeError("fixture trace failure")
                    yield
            finally:
                assert gun._instrument_magnetic_field is sentinel
                assert gun._instrument_magnetic_identity == "old-identity"
                assert cuda_step.current_cuda_gun_binding() is None
        if fail:
            with pytest.raises(RuntimeError, match="fixture trace failure"):
                with exercise():
                    pass
        else:
            with exercise():
                pass


def test_gpu_prepare_failure_restores_electric_and_magnetic_bindings(scoped_environment, monkeypatch):
    state, _, _, environment = scoped_environment
    state.acceleration_backend = "Prefer GPU"
    def failure(*_a, **_k):
        raise GPUExecutionError("unavailable", "fixture CUDA preparation failure")
    monkeypatch.setitem(sys.modules, "temsim.physics.cuda_coil_potential",
        SimpleNamespace(prepare_cuda_instrument_magnetic=failure))
    with pytest.raises(GPUExecutionError, match="fixture CUDA preparation failure"):
        with environment.instrument_gun_field_context(state):
            pytest.fail("Failed setup reached a Gun trace")
    assert not any(name.startswith("_instrument_") for name in vars(state.electron_gun))
    assert cuda_step.current_cuda_gun_binding() is None


def test_actual_cuda_field_dg_preserves_existing_step_and_work_tolerances(reference_only):
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("CUDA field operator unavailable; no physical or wave trial is run")
    phase, magnetic, electric, dt = _phase(256), _instrument(), _electric(), 1e-14
    reference = static_energy_step(phase, dt, magnetic, electric, tolerance=1e-13)
    prepared = cuda_step.prepare_cuda_gun_binding(magnetic, execution_backend="Prefer GPU")
    assert prepared is not None
    gpu_calls_before = prepared.sampler.execution_receipt()["counts"].get("gpu_calls", 0)
    with cuda_step.cuda_gun_execution(prepared):
        actual = static_energy_step(phase, dt, prepared.sampler, electric, tolerance=1e-13)
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-14, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s,
        rtol=2e-12, atol=1e-36)
    before = kinetic_energy_ev_from_momentum(phase.momentum_kg_m_per_s)-electric.potential_rise_v_at_global_positions(phase.position_m)
    after = kinetic_energy_ev_from_momentum(actual.momentum_kg_m_per_s)-electric.potential_rise_v_at_global_positions(actual.position_m)
    np.testing.assert_allclose(after, before, rtol=0., atol=3e-9)
    assert actual.time_s == reference.time_s == dt
    assert prepared.sampler.execution_receipt()["counts"].get("gpu_calls", 0) > gpu_calls_before


@pytest.mark.parametrize("step_doubling", [False, True])
def test_actual_cuda_radial_hexapole_preserves_dg_and_surface_accuracy_gates(reference_only, step_doubling):
    """Synthetic phase arrays exercise closed rectangles without an emitter."""
    from dataclasses import replace
    from temsim.magnetic_field_scene import _Source
    from temsim.physics.compute_backend import cupy_capability
    from temsim.physics.instrument_magnetic import InstrumentMagneticField
    from temsim.physics.relativistic_lorentz import RelativisticPhaseSpace
    from temsim.physics.transverse_coil_field import FrozenTransverseCoilField
    from test_radial_multipole_symmetry import _potential
    if not cupy_capability().available:
        pytest.skip("CUDA field operator unavailable; no physical or wave trial is run")
    # Match the unchanged full default pack/path rule. The explicit 134526 A,
    # two-turn source stresses six-coil cancellation, not a fitted probe.
    potential = replace(_potential(3, 1), pack_order=12, path_order=24)
    field = FrozenTransverseCoilField(potential, potential.source_id, 3e-22)
    row = _Source(field.key, field, field.bounds_m, field.radial_support_m)
    magnetic = InstrumentMagneticField((row,), None, None)
    initial = _phase(256, z=.0045)
    positions = initial.position_m.copy()
    positions[:, :2] *= 10000.
    phase = RelativisticPhaseSpace(positions, initial.momentum_kg_m_per_s)
    electric, dt = _electric(), 1e-14
    def advance(selected):
        if not step_doubling:
            return static_energy_step(phase, dt, selected, electric, tolerance=1e-13), dt
        from temsim.optics.electron_gun.tracing import _surface_step
        return _surface_step(phase, dt, np.ones(len(positions), dtype=bool), selected, electric)
    reference, reference_dt = advance(magnetic)
    prepared = cuda_step.prepare_cuda_gun_binding(magnetic, execution_backend="Prefer GPU")
    before = prepared.sampler.execution_receipt()["counts"].get("gpu_calls", 0)
    analytic_before = prepared.sampler.execution_receipt()["counts"].get("gpu_analytic_rectangle_source_calls", 0)
    with cuda_step.cuda_gun_execution(prepared):
        actual, actual_dt = advance(prepared.sampler)
    assert actual_dt == reference_dt
    np.testing.assert_allclose(actual.position_m, reference.position_m, rtol=2e-14, atol=1e-22)
    np.testing.assert_allclose(actual.momentum_kg_m_per_s, reference.momentum_kg_m_per_s,
        rtol=2e-12, atol=1e-36)
    assert actual.time_s == reference.time_s == actual_dt
    assert prepared.sampler.execution_receipt()["counts"].get("gpu_calls", 0) > before
    assert prepared.sampler.execution_receipt()["counts"].get("gpu_analytic_rectangle_source_calls", 0) > analytic_before
