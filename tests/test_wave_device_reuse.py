"""Serial mode ownership without changing backend policy or physical samples."""
from contextlib import contextmanager

import numpy as np
import pytest

from temsim.physics import wave_device as device
from temsim.physics.compute_backend import GPUExecutionError, WAVE_BACKEND_NUMPY
from test_wave_device import _fake_cuda_scopes


def test_reuse_is_lazy_and_preserves_each_operator_work_size_policy(monkeypatch):
    pools, _ = _fake_cuda_scopes(monkeypatch)
    choose = device.choose_wave_backend
    calls = []
    def select(requested, **kwargs):
        calls.append((requested, kwargs["work_items"]))
        if requested == "Auto" and kwargs["work_items"] < 100:
            return WAVE_BACKEND_NUMPY, "Small workload"
        return choose(requested, **kwargs)
    monkeypatch.setattr(device, "choose_wave_backend", select)
    with device.reusable_device_scope(maximum_working_bytes=256*1024**2) as owner:
        assert not pools and owner.session is None
        for size in (3, 300, 3, 300):
            with device.device_scope("Auto", work_items=size) as session:
                assert session.backend == ("numpy" if size == 3 else "cupy")
            assert device._ACTIVE.get() is None
            # Host canonical/refinement budgets must not inherit the GPU cap.
            device.check_device_memory(10**30)
        assert len(pools) == 1
    assert pools[0].freed == 1 and device._REUSABLE.get() is None
    assert ("Auto", 3) in calls and ("Auto", 300) in calls


@pytest.mark.parametrize("policy", ["CPU", "Auto"])
def test_cpu_only_reusable_segment_never_initializes_cuda(monkeypatch, policy):
    monkeypatch.setattr(device, "choose_wave_backend", lambda *a, **k: (WAVE_BACKEND_NUMPY, None))
    monkeypatch.setattr(device, "cupy_module", lambda: pytest.fail("CPU segment initialized CUDA"))
    with device.reusable_device_scope() as owner:
        with device.device_scope(policy) as session:
            assert session.backend == "numpy"
        assert owner.session is None and device._ACTIVE.get() is None


def test_changed_shape_and_pressure_release_only_unused_storage(monkeypatch):
    pools, plans = _fake_cuda_scopes(monkeypatch)
    limit = 256*1024**2
    with device.reusable_device_scope(maximum_working_bytes=limit) as owner:
        with device.device_scope("Require GPU", workspace_shape=(128, 128)):
            pass
        pool = pools[0]
        cleared, freed = plans.cleared, pool.freed
        # A live caller buffer remains live after freeing unused blocks.
        monkeypatch.setattr(pool, "used_bytes", lambda: 4096)
        monkeypatch.setattr(pool, "total_bytes", lambda: 8192)
        with device.device_scope("Require GPU", workspace_shape=(128, 128)):
            assert plans.cleared == cleared and pool.freed == freed
        with device.device_scope("Require GPU", workspace_shape=(256, 256)):
            assert plans.cleared == cleared+1 and pool.freed == freed+1
            assert pool.used_bytes() == 4096
        monkeypatch.setattr(pool, "total_bytes", lambda: limit)
        with device.device_scope("Require GPU", workspace_shape=(256, 256)):
            assert plans.cleared == cleared+2 and pool.freed == freed+2
        assert owner.session.workspace_shape == (256, 256)
        monkeypatch.setattr(pool, "used_bytes", lambda: 0)


def test_nested_hard_caps_and_failures_preserve_owner_and_cpu_override(monkeypatch):
    pools, plans = _fake_cuda_scopes(monkeypatch)
    limit = 256*1024**2
    limits = []
    with device.reusable_device_scope(maximum_working_bytes=limit) as owner:
        with device.device_scope("Require GPU", workspace_shape=(128, 128)):
            pass
        pool = pools[0]
        monkeypatch.setattr(pool, "set_limit", lambda *, size: limits.append(size))
        with pytest.raises(GPUExecutionError, match="out_of_memory"):
            with device.device_scope("Require GPU", maximum_working_bytes=1,
                                     required_bytes=2):
                pytest.fail("A reused pool bypassed the nested hard cap")
        assert limits == [1, limit]
        assert owner.session.maximum_working_bytes == limit
        with device.device_scope("CPU") as cpu:
            assert cpu.backend == "numpy"
            device.check_device_memory(10**30)
        assert device._ACTIVE.get() is None
        with device.device_scope("Require GPU") as reused:
            assert reused is owner.session
    assert len(pools) == 1 and device._ACTIVE.get() is None
    assert (plans.size, plans.memory) == (16, 3 << 30)


@pytest.mark.parametrize("error", [InterruptedError("Cancelled mode"), ValueError("Physical gate")])
def test_cancel_and_domain_failure_close_owner_without_retry(monkeypatch, error):
    pools, plans = _fake_cuda_scopes(monkeypatch)
    with pytest.raises(type(error), match=str(error)):
        with device.reusable_device_scope(maximum_working_bytes=256*1024**2):
            with device.device_scope("Require GPU"):
                raise error
    assert device._ACTIVE.get() is None and device._REUSABLE.get() is None
    assert len(pools) == 1 and pools[0].freed == 1
    assert (plans.size, plans.memory) == (16, 3 << 30)


def test_cleanup_failure_does_not_replace_cancellation(monkeypatch):
    _fake_cuda_scopes(monkeypatch)
    original_close = device._ReusableDevice.close
    def fail_after_cleanup(owner):
        original_close(owner)
        raise RuntimeError("Injected cleanup failure")
    monkeypatch.setattr(device._ReusableDevice, "close", fail_after_cleanup)
    with pytest.raises(InterruptedError, match="Original cancellation") as caught:
        with device.reusable_device_scope():
            with device.device_scope("Require GPU"):
                raise InterruptedError("Original cancellation")
    assert device._ACTIVE.get() is None and device._REUSABLE.get() is None
    assert any("Injected cleanup failure" in note for note in caught.value.__notes__)


def test_failed_optional_admission_does_not_poison_later_operator(monkeypatch):
    pools, _ = _fake_cuda_scopes(monkeypatch)
    get_cupy = device.cupy_module
    attempts = [0]
    def unavailable_once():
        attempts[0] += 1
        if attempts[0] == 1:
            raise GPUExecutionError("unavailable", "Injected unavailable device")
        return get_cupy()
    monkeypatch.setattr(device, "cupy_module", unavailable_once)
    with device.reusable_device_scope() as owner:
        with device.device_scope("Auto") as cpu:
            assert cpu.backend == "numpy" and "unavailable" in cpu.fallback_reason
        assert owner.session is None
        with device.device_scope("Require GPU") as gpu:
            assert gpu.backend == "cupy"
        assert len(pools) == 1


def test_nested_resource_owners_are_rejected_without_creating_another_pool(monkeypatch):
    pools, _ = _fake_cuda_scopes(monkeypatch)
    with device.reusable_device_scope(maximum_working_bytes=256*1024**2) as owner:
        with device.device_scope("Require GPU"):
            pass
        with pytest.raises(ValueError, match="non-nested host segment"):
            with device.reusable_device_scope(maximum_working_bytes=512*1024**2):
                pytest.fail("A nested owner bypassed the shared resource cap")
        with device.device_scope("Require GPU") as reused:
            assert reused is owner.session
        assert len(pools) == 1
    assert device._ACTIVE.get() is None and device._REUSABLE.get() is None


@pytest.mark.parametrize("cancel", [False, True])
def test_failed_limit_restoration_always_resets_active_context(monkeypatch, cancel):
    pools, _ = _fake_cuda_scopes(monkeypatch)
    limit = 256*1024**2
    with device.reusable_device_scope(maximum_working_bytes=limit):
        with device.device_scope("Require GPU"):
            pass
        def restore_fails(*, size):
            if size == limit:
                raise RuntimeError("Injected limit restoration failure")
        monkeypatch.setattr(pools[0], "set_limit", restore_fails)
        error = InterruptedError if cancel else RuntimeError
        message = "Original cancellation" if cancel else "Injected limit restoration failure"
        with pytest.raises(error, match=message) as caught:
            with device.device_scope("Require GPU", maximum_working_bytes=limit//2):
                if cancel:
                    raise InterruptedError("Original cancellation")
        assert device._ACTIVE.get() is None
        if cancel:
            assert any("limit restoration" in note for note in caught.value.__notes__)


def test_reused_gpu_resource_retry_runs_original_host_input(monkeypatch):
    from temsim.physics import potential_wave_device as executor
    from temsim.physics import potential_wave_carrier as carrier
    from test_potential_wave_device import _arguments
    args, options = _arguments()
    expected, _ = executor.execute_residual_exponential(*args, **options)
    original = args[0].copy()
    pools, _ = _fake_cuda_scopes(monkeypatch)
    actual_scope = executor.device_scope
    entered = []
    @contextmanager
    def numpy_kernels_for_policy_only(requested, **kwargs):
        with actual_scope(requested, **kwargs) as session:
            entered.append(session.backend)
            # Exercise real resource ownership while keeping this test CPU-only.
            yield device.WaveDevice(backend=session.backend,
                maximum_working_bytes=session.maximum_working_bytes)
    monkeypatch.setattr(executor, "device_scope", numpy_kernels_for_policy_only)
    apply = carrier._apply_residual
    calls = [0]
    def fail_after_completed_substep(*a, **k):
        calls[0] += 1
        if calls[0] == 4:
            raise GPUExecutionError("out_of_memory", "Injected completed-substep failure")
        return apply(*a, **k)
    monkeypatch.setattr(carrier, "_apply_residual", fail_after_completed_substep)
    with device.reusable_device_scope(maximum_working_bytes=256*1024**2):
        output, evidence = executor.execute_residual_exponential(*args, **options, requested="Auto")
        assert device._ACTIVE.get() is None
        repeated, _ = executor.execute_residual_exponential(*args, **options, requested="Require GPU")
        np.testing.assert_array_equal(repeated, expected)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(args[0], original)
    assert entered == ["cupy", "numpy", "cupy"] and len(pools) == 1
    assert evidence["retry_input"] == "unchanged host residual entrance"


def test_adapter_owns_one_lazy_scope_per_complete_mode_segment(tmp_path, monkeypatch):
    from dataclasses import replace
    from temsim.optics.column import default_state
    from temsim.physics import unified_wave_runtime as runtime
    from temsim.physics.wave_flux import BeamState
    from test_potential_geometry import _checkpoint
    from test_unified_wave_runtime import _runtime, _request
    state = default_state()
    source = _checkpoint()
    mode = source.beam.modes[0]
    source = replace(source, beam=BeamState((mode, replace(mode, mode_id="second")),
                                           source.beam.reference_plane))
    adapter = runtime.UnifiedPotentialWaveAdapter(state, _request(tmp_path),
        runtime=_runtime(state, 1., 1.000004))
    observe = runtime.observe_potential_geometry_checkpoint
    owners = []
    def wrapped(*args, **kwargs):
        owner = device._REUSABLE.get()
        assert owner is not None and device._ACTIVE.get() is None
        owners.append(owner)
        return observe(*args, **kwargs)
    monkeypatch.setattr(runtime, "observe_potential_geometry_checkpoint", wrapped)
    output = adapter.column(source, 1000.0004)
    # Four steps with a three-step segment limit produce two segments, each
    # with two independently owned incoherent modes.
    assert len(owners) == 4 and len({id(owner) for owner in owners}) == 4
    assert all(owner.session is None for owner in owners)
    assert device._REUSABLE.get() is None
    assert [m.mode_id for m in output.beam.modes] == [m.mode_id for m in source.beam.modes]


def test_real_cuda_reuse_preserves_raw_complex_fields_and_live_arrays():
    import subprocess
    from temsim.physics.compute_backend import cupy_capability
    if not cupy_capability().available:
        pytest.skip("Optional real CUDA device unavailable")
    def check_shared_gpu():
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
            "--format=csv,noheader,nounits"], text=True)
        if max(int(row) for row in used.splitlines()) >= 8192:
            pytest.skip("Concurrent GPU work occupies at least 8 GiB")
    check_shared_gpu()
    import cupy as cp
    from temsim.physics.potential_wave_device import execute_residual_exponential
    from test_potential_wave_residual_memory import _input
    limit = 256*1024**2
    data = {n: _input((n, n)) for n in (128, 256, 350)}
    def run(n, requested, **updates):
        args = data[n]
        options = dict(phase=args[2]*25,
            scalar_splitting=True, kinetic_order=8, quartic_scalar_splitting=False,
            higher_order_scalar_splitting=True, count=1, degree=1, k_distance=50.,
            boundary_limit=1e-9, envelope_limit=1e-10, cancelled=lambda: False,
            maximum_device_working_bytes=limit, requested=requested)
        options.update(updates)
        return execute_residual_exponential(*args, **options)[0]
    expected = {n: run(n, "CPU") for n in data}
    fresh = {n: run(n, "Require GPU") for n in data}
    with device.reusable_device_scope(maximum_working_bytes=limit) as owner:
        for n in (128, 128, 256, 350, 128):
            check_shared_gpu()
            actual = run(n, "Require GPU")
            np.testing.assert_array_equal(actual, fresh[n])
            assert np.linalg.norm(actual-expected[n])/np.linalg.norm(expected[n]) < 2e-14
            assert device._ACTIVE.get() is None
            assert owner.session.pool.total_bytes() <= limit
            assert owner.session.pool.get_limit() == limit
        with pytest.raises(InterruptedError, match="cancelled"):
            run(128, "Require GPU", cancelled=lambda: True)
        with pytest.raises(GPUExecutionError, match="out_of_memory"):
            run(128, "Require GPU", maximum_device_working_bytes=1)
        np.testing.assert_array_equal(run(128, "Require GPU"), fresh[128])
        with device.device_scope("Require GPU", workspace_shape=(128, 128)):
            live = cp.arange(17, dtype=cp.float64)
        with device.device_scope("Require GPU", workspace_shape=(350, 350)):
            np.testing.assert_array_equal(cp.asnumpy(live), np.arange(17))
        del live
    device.release_device_memory()
    assert owner.session.pool.used_bytes() == 0
    assert device._ACTIVE.get() is None and device._REUSABLE.get() is None
