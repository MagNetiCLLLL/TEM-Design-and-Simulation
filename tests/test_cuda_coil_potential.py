"""Bounded pure-field CUDA parity and failure-policy tests; no source or wave."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential
from temsim.physics.cuda_coil_potential import (
    prepare_cuda_coil_potential, prepare_cuda_instrument_magnetic,
    cuda_coil_execution_identity, cuda_coil_cache_info, clear_cuda_coil_cache,
)
from temsim.physics.compute_backend import GPUExecutionError, cupy_capability, gpu_retry_reason
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.instrument_magnetic import InstrumentMagneticField
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge
from temsim.physics.transverse_coil_field import TransverseCoilPotential
from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler


@pytest.fixture(autouse=True)
def no_source_or_wave(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CUDA field operator tests must not emit, trace a source or propagate a wave")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


@pytest.fixture(scope="module")
def cuda():
    capability = cupy_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    return cuda_coil_execution_identity("Prefer GPU")


def round_potential(**changes):
    coil = AnnularCoilPotential(**dict(dict(source_id="round", inner_radius_m=.02,
        outer_radius_m=.035, length_m=.04, ampere_turns=2700., center_z_m=.5), **changes))
    return CopperCoilAssemblyPotential("round-owner", (coil, replace(coil, source_id="round-reverse",
        center_z_m=.65, ampere_turns=-.25*coil.ampere_turns)))


def transverse_potential(**changes):
    from test_compiled_transverse_coil import winding
    return replace(winding().potential, **changes)


def snapshot(*potentials, gauge=None, electric=False):
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    rows = [PotentialContribution("source:"+str(index), ("owner:"+str(index),), potential, "magnetic")
            for index, potential in enumerate(potentials)]
    if electric:
        from test_electric_potential_snapshot import planar, wrapped
        rows.append(PotentialContribution("electric", ("electric",), ElectricPotentialSnapshot.from_provider(wrapped(planar())), "electric"))
    return ElectromagneticPotentialSnapshot(scene, tuple(rows), gauge or StaticPolynomialGauge())


def compare(expected, actual):
    for key, value in expected.items():
        assert actual[key].shape == value.shape
        if key in ("phi_v", "E_v_per_m"):
            np.testing.assert_array_equal(actual[key], value)
        else:
            np.testing.assert_allclose(actual[key], value, rtol=5.e-12,
                atol=2.e-19 if key == "A_t_m" else 2.e-16)


@pytest.mark.parametrize("policy", ["Auto", "CPU", "Numba CPU"])
def test_default_cpu_policy_never_initializes_cuda(policy, monkeypatch):
    import temsim.physics.cuda_coil_potential as module
    monkeypatch.setattr(module, "_device", lambda: (_ for _ in ()).throw(AssertionError("Unexpected CUDA discovery")))
    original = snapshot(round_potential(), transverse_potential())
    assert prepare_cuda_coil_potential(original, execution_backend=policy) is None
    sampler = compile_unified_potential_sampler(original, execution_backend=policy)
    point = np.array([[1.e-8, -2.e-8, .04]])
    for key, value in original.sample(point).items():
        np.testing.assert_array_equal(sampler.sample(point)[key], value)


@pytest.mark.parametrize("mode", ["potential", "magnetic"])
def test_live_outer_callback_guard_works_without_any_cuda_runtime(mode, monkeypatch):
    """Pure mock: no driver, preparation, device data or source calculation."""
    from collections import Counter
    from threading import RLock
    from temsim.physics.cuda_coil_potential import CudaCoilSampler
    original = snapshot(round_potential()) if mode == "potential" else InstrumentMagneticField((), None, None)
    sampler = object.__new__(CudaCoilSampler)
    sampler.original, sampler.mode = original, mode
    sampler._lock, sampler._counts, sampler._reasons = RLock(), Counter(), Counter()
    if mode == "potential":
        def callback(self, p):
            return dict(A_t_m=np.ones_like(p)*9., B_t=np.ones_like(p)*7.,
                        E_v_per_m=np.ones_like(p)*5., phi_v=np.full(np.shape(p)[:-1], 3.))
        monkeypatch.setattr(ElectromagneticPotentialSnapshot, "sample", callback)
    else:
        monkeypatch.setattr(InstrumentMagneticField, "field_at_global_positions_t", lambda self, p: np.ones_like(p)*7.)
    for count in (1, 128):
        points = np.zeros((count, 3))
        actual = sampler.sample(points)["B_t"] if mode == "potential" else sampler.sample_b(points)
        np.testing.assert_array_equal(actual, np.ones_like(points)*7.)
    assert sampler._counts["reference_calls"] == 2 and sampler._counts["gpu_calls"] == 0


@pytest.mark.parametrize("shape", [(3,), (17, 3), (3, 5, 3), (128, 3)])
def test_cuda_potential_keeps_shape_source_order_electric_and_final_gauge(cuda, shape):
    gauge = StaticPolynomialGauge(1.e-15, (1.e-8, -2.e-8, 3.e-8),
        ((1.e-5, 2.e-6, 0.), (2.e-6, -3.e-5, 0.), (0., 0., 2.e-5)))
    original = snapshot(transverse_potential(), round_potential(), gauge=gauge, electric=True)
    points = np.zeros(shape); points[..., 0] = 1.e-6; points[..., 1] = -2.e-6; points[..., 2] = .0005
    before = points.copy()
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    compare(original.sample(points), sampler.sample(points))
    np.testing.assert_array_equal(points, before)
    receipt = sampler.execution_receipt()
    assert receipt["counts"]["gpu_calls"] >= 1
    assert receipt["reference_reasons"]["native_electric_provider"] >= 1
    assert receipt["execution_identity"]["minimum_gpu_batch"] == 1


def test_gpu_and_cpu_sampler_cache_keys_and_mode_identities_are_distinct(cuda):
    original = snapshot(round_potential())
    cpu = compile_unified_potential_sampler(original)
    gpu = compile_unified_potential_sampler(original, execution_backend="Prefer GPU")
    assert cpu is not gpu
    assert gpu is compile_unified_potential_sampler(original, execution_backend="Prefer GPU")
    assert gpu.execution_identity["requested_backend"] == "Prefer GPU"
    assert gpu.snapshot_digest == cpu.snapshot_digest == original.numerical_digest
    assert prepare_cuda_coil_potential(original, execution_backend="Prefer GPU") is not prepare_cuda_coil_potential(original, execution_backend="Require GPU")
    assert cuda_coil_execution_identity("Prefer GPU", mode="magnetic")["minimum_gpu_batch"] == 128
    assert cuda_coil_execution_identity("Prefer GPU", mode="potential")["minimum_gpu_batch"] == 1


def test_poses_elliptic_and_near_winding_vacuum_use_declared_native_reference(cuda):
    posed = round_potential(registration=CoordinateRegistration((1.e-4, 0., 0.)))
    original = snapshot(posed, round_potential(), transverse_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    points = np.array([[.020, 0., 0.]])
    compare(original.sample(points), sampler.sample(points))
    reasons = sampler.execution_receipt()["reference_reasons"]
    assert reasons["unsupported_or_posed_source"]
    assert reasons["round_pose_domain_or_elliptic_branch"]
    assert reasons["transverse_bore_numerical_admission"]


@pytest.mark.parametrize("points", [np.array([[.02, 0., .5]]), np.array([[.020, .01205, 0.]])])
def test_material_queries_keep_the_original_error_before_cuda_work(cuda, points):
    original = snapshot(round_potential(), transverse_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    before = sampler.execution_receipt()["counts"].get("gpu_calls", 0)
    with pytest.raises(ValueError, match="winding"):
        sampler.sample(points)
    assert sampler.execution_receipt()["counts"].get("gpu_calls", 0) == before


def test_extreme_finite_values_and_empty_queries_preserve_reference(cuda):
    original = snapshot(round_potential(), transverse_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    for points in (np.array([[0., 0., 1.e6]]), np.empty((0, 3))):
        for key, value in original.sample(points).items():
            np.testing.assert_array_equal(sampler.sample(points)[key], value)
    with pytest.raises(ValueError, match="finite real"):
        sampler.sample(np.array([[np.nan, 0., 0.]]))


def test_exact_zero_current_and_complete_axis_symmetry_preserve_derivative_owner(cuda):
    from test_compiled_transverse_coil import rectangle
    coils = tuple(rectangle(coil_index=i+1, azimuth_deg=30.+i*60., current_sign=(-1)**i) for i in range(6))
    complete = TransverseCoilPotential("hexapole", coils, 2, (-1.e5, 0.), provenance={"multipole_order": 3})
    original = snapshot(complete, transverse_potential(currents_a=(0., 0.)))
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    points = np.array([[0., 0., 0.], [0., 0., .04], [0., 0., 1.6]])
    actual = sampler.sample(points)
    np.testing.assert_array_equal(actual["A_t_m"], np.zeros_like(points))
    np.testing.assert_array_equal(actual["B_t"], np.zeros_like(points))
    assert sampler.execution_receipt()["counts"]["exact_zero_source_calls"] >= 2
    assert sampler.execution_receipt()["counts"].get("gpu_calls", 0) == 0
    assert np.max(abs(complete.sample_jet(np.array([[0., 0., 0.]]))["hess_A_t_per_m"])) > 0.
    points = np.array([[1.e-20, -2.e-20, 0.], [1.e-8, -2.e-8, 0.]])
    actual = sampler.sample(points); expected = original.sample(points)
    # Independent absolute cancellation allowance; no field/solver tolerance is changed.
    np.testing.assert_allclose(actual["A_t_m"], expected["A_t_m"], rtol=5.e-12, atol=2.e-18)
    # Native near-zero B is itself a highly cancelled ordinary reduction.
    # Check both against an independent accurately summed node reference and
    # the standard gamma_n absolute forward-error bound, rather than enlarging
    # an output-relative tolerance whose denominator is effectively zero.
    import math
    from temsim.physics.compiled_transverse_coil import pack_transverse_potential
    nodes, weighted, _, _ = pack_transverse_potential(complete)
    delta = points[:, None]-nodes[None]
    norm2 = np.einsum("pni,pni->pn", delta, delta)
    terms = np.cross(weighted[None], delta)*((1/np.sqrt(norm2))/norm2)[:, :, None]
    accurate = np.array([[math.fsum(terms[p, :, k]) for k in range(3)] for p in range(len(points))])
    mass = np.sum(abs(terms), axis=1)
    eps = np.finfo(float).eps
    gpu_depth = math.ceil(len(nodes)/32)+5
    gpu_bound = (gpu_depth*eps/(1-gpu_depth*eps)+16*eps)*mass
    native_bound = (len(nodes)*eps/(1-len(nodes)*eps)+16*eps)*mass
    assert np.all(abs(actual["B_t"]-accurate) <= gpu_bound)
    assert np.all(abs(expected["B_t"]-accurate) <= native_bound)
    assert np.all(abs(actual["B_t"]-expected["B_t"]) <= gpu_bound+native_bound)


def test_cuda_allocation_failure_is_fatal_even_under_prefer_gpu(cuda, monkeypatch):
    original = snapshot(round_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    def fail(*_args, **_kwargs):
        raise GPUExecutionError("out_of_memory", "Injected field allocation failure")
    with monkeypatch.context() as patch:
        patch.setattr(sampler._cp, "empty", fail)
        with pytest.raises(GPUExecutionError) as caught:
            sampler.sample(np.array([[1.e-8, 0., .04]]))
    error = caught.value
    assert error.category == "coil_field_execution_failure" and error.original_category == "out_of_memory"
    with pytest.raises(GPUExecutionError) as retried:
        gpu_retry_reason(error, "Prefer GPU")
    assert retried.value is error
    assert sampler.execution_receipt()["failure"]["stage"] == "allocation_launch_or_copy"


def test_nonfinite_device_result_never_becomes_a_reference_success(cuda, monkeypatch):
    original = snapshot(round_potential(ampere_turns=2701.))
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    def invalid(_grid, _block, args):
        args[-1].fill(float("nan"))
    monkeypatch.setattr(sampler, "_round_kernel", invalid)
    with pytest.raises(GPUExecutionError, match="invalid"):
        sampler.sample(np.array([[1.e-8, 0., .04]]))


def test_magnetic_only_api_keeps_small_batch_and_full_batch_policies(cuda):
    from test_compiled_copper_gun import _copper, _native
    from test_compiled_transverse_coil import winding, source, native_field
    magnetic = InstrumentMagneticField((_copper(), source(winding()), _native()), None, None)
    sampler = prepare_cuda_instrument_magnetic(magnetic, execution_backend="Prefer GPU")
    small = np.array([[1.e-8, -2.e-8, .04]])
    np.testing.assert_array_equal(sampler.field_at_global_positions_t(small), magnetic.field_at_global_positions_t(small))
    points = np.repeat(small, 128, axis=0)
    np.testing.assert_allclose(sampler.sample_b(points), native_field(magnetic, points), rtol=5.e-12, atol=2.e-16)
    receipt = sampler.execution_receipt()
    assert receipt["reference_reasons"]["small_magnetic_batch"] >= 1
    assert receipt["counts"]["gpu_calls"] >= 1
    with pytest.raises(ValueError, match="magnetic-only"):
        sampler.sample(points)


@pytest.mark.parametrize("scope", ["class", "instance"])
def test_prepared_magnetic_sampler_never_bypasses_a_later_instrument_callback(cuda, monkeypatch, scope):
    from test_compiled_copper_gun import _copper
    magnetic = InstrumentMagneticField((_copper(),), None, None)
    sampler = prepare_cuda_instrument_magnetic(magnetic, execution_backend="Prefer GPU")
    callback = lambda points: np.ones_like(points)*7.
    if scope == "class":
        monkeypatch.setattr(InstrumentMagneticField, "field_at_global_positions_t", lambda self, points: callback(points))
    else:
        object.__setattr__(magnetic, "field_at_global_positions_t", callback)
    for count in (1, 128):
        points = np.repeat(np.array([[1.e-8, 0., .04]]), count, axis=0)
        np.testing.assert_array_equal(sampler.sample_b(points), callback(points))
    receipt = sampler.execution_receipt()
    assert receipt["counts"].get("gpu_calls", 0) == 0
    assert receipt["reference_reasons"]["overridden_instrument_callback"] == 2


def test_snapshot_callback_is_live_and_owns_its_gauge_exactly_once(cuda, monkeypatch):
    original = snapshot(round_potential(), gauge=StaticPolynomialGauge(0., (1., 2., 3.)))
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    def custom(self, points):
        a = np.ones_like(points)*7.
        return dict(A_t_m=a, B_t=a*2, E_v_per_m=a*3, phi_v=np.full(np.shape(points)[:-1], 5.))
    monkeypatch.setattr(ElectromagneticPotentialSnapshot, "sample", custom)
    for count in (0, 1, 128):
        points = np.zeros((count, 3))
        actual = sampler.sample(points)
        for key, value in custom(original, points).items():
            np.testing.assert_array_equal(actual[key], value)
    assert sampler.execution_receipt()["reference_reasons"]["overridden_snapshot_callback"] == 3
    assert sampler.execution_receipt()["counts"].get("gpu_calls", 0) == 0


def test_magnetic_round_wrapper_is_not_bypassed_by_gpu_rows(cuda, monkeypatch):
    from test_compiled_copper_gun import _copper
    from temsim.physics.frozen_copper_coil_field import FrozenCopperCoilField
    magnetic = InstrumentMagneticField((_copper(),), None, None)
    sampler = prepare_cuda_instrument_magnetic(magnetic, execution_backend="Prefer GPU")
    monkeypatch.setattr(FrozenCopperCoilField, "field_at_global_positions_t", lambda self, p: np.ones_like(p)*11.)
    for count in (1, 128):
        points = np.repeat(np.array([[1.e-8, 0., .04]]), count, axis=0)
        np.testing.assert_array_equal(sampler.sample_b(points), np.ones_like(points)*11.)
    assert sampler.execution_receipt()["reference_reasons"]["overridden_round_field_wrapper"] == 1


def test_equal_digest_clean_instrument_never_reuses_an_overridden_original_wrapper(cuda):
    first = InstrumentMagneticField((), "same-physical", "a"*64)
    old = prepare_cuda_instrument_magnetic(first, execution_backend="Prefer GPU")
    object.__setattr__(first, "field_at_global_positions_t", lambda p: np.ones_like(p)*7.)
    second = InstrumentMagneticField((), "same-physical", "a"*64)
    clean = prepare_cuda_instrument_magnetic(second, execution_backend="Prefer GPU")
    assert old is not clean and old.original is first and clean.original is second
    for count in (1, 128):
        points = np.zeros((count, 3))
        np.testing.assert_array_equal(old.sample_b(points), np.ones_like(points)*7.)
        np.testing.assert_array_equal(clean.sample_b(points), np.zeros_like(points))


def test_equal_snapshot_graphs_keep_their_own_gpu_callback_wrappers(cuda):
    first = snapshot(round_potential())
    second = ElectromagneticPotentialSnapshot.from_dict(first.to_dict())
    assert first.numerical_digest == second.numerical_digest
    old = prepare_cuda_coil_potential(first, execution_backend="Prefer GPU")
    clean = prepare_cuda_coil_potential(second, execution_backend="Prefer GPU")
    assert old is not clean and old.cache_token != clean.cache_token
    assert compile_unified_potential_sampler(first, execution_backend="Prefer GPU") is not compile_unified_potential_sampler(second, execution_backend="Prefer GPU")


def test_duplicate_magnetic_source_labels_keep_every_original_contribution(cuda):
    from temsim.magnetic_field_scene import _Source
    bounds = np.array([[-1., -1., 0.], [1., 1., 1.]])
    first = _Source("same-label", SimpleNamespace(field_at_global_positions_t=lambda p: np.ones_like(p)), bounds, 1.)
    second = _Source("same-label", SimpleNamespace(field_at_global_positions_t=lambda p: np.ones_like(p)*2.), bounds, 1.)
    original = InstrumentMagneticField((first, second), None, None)
    sampler = prepare_cuda_instrument_magnetic(original, execution_backend="Prefer GPU")
    for count in (1, 128):
        points = np.repeat(np.array([[0., 0., .04]]), count, axis=0)
        np.testing.assert_array_equal(sampler.sample_b(points), original.field_at_global_positions_t(points))
        np.testing.assert_array_equal(sampler.sample_b(points), np.ones_like(points)*3.)
    receipt = sampler.execution_receipt()
    assert receipt["reference_reasons"]["duplicate_magnetic_source_keys"] == 4
    assert receipt["counts"].get("gpu_calls", 0) == 0


def test_prepared_device_cache_is_bounded(cuda, monkeypatch):
    import temsim.physics.cuda_coil_potential as module
    clear_cuda_coil_cache()
    monkeypatch.setattr(module, "MAX_CACHE_ENTRIES", 2)
    for current in (2701., 2702., 2703.):
        prepare_cuda_coil_potential(snapshot(round_potential(ampere_turns=current)), execution_backend="Prefer GPU")
    assert cuda_coil_cache_info()["entries"] <= 2
    clear_cuda_coil_cache()


@pytest.mark.parametrize("shape", [(3,), (17, 3), (3, 5, 3), (128, 3)])
def test_real_fields_only_agrees_with_full_fields_without_exposing_A(cuda, shape):
    original = snapshot(round_potential(), transverse_potential(), electric=True,
        gauge=StaticPolynomialGauge(1.e-12, (1.e-5, 2.e-5, -3.e-5), np.eye(3)*.001))
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    points = np.zeros(shape); points[..., 0] = 1.e-6; points[..., 1] = -2.e-6; points[..., 2] = .0005
    full = sampler.sample(points); physical = sampler.sample_fields(points)
    assert set(physical) == {"B_t", "E_v_per_m", "phi_v"}
    for key in physical:
        np.testing.assert_array_equal(physical[key], full[key])
    assert sampler.execution_receipt()["counts"]["physical_fields_calls"] >= 1


def test_fields_only_never_runs_A_shader_or_final_gauge(cuda, monkeypatch):
    original = snapshot(round_potential(), transverse_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    expected = sampler.sample(np.array([[1.e-6, -2.e-6, .0005]]))
    called = []
    round_kernel, segment_kernel = sampler._round_kernel, sampler._segment_kernel
    def round_fields(grid, block, args):
        assert args[-2] == 0 and args[-1].shape[-1] == 3
        called.append("round B"); return round_kernel(grid, block, args)
    def segment_fields(grid, block, args):
        assert args[-2] == 0 and args[-1].shape[-1] == 3
        called.append("rectangle B"); return segment_kernel(grid, block, args)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("An unrequested A evaluator was called")
    monkeypatch.setattr(sampler, "_round_kernel", round_fields)
    monkeypatch.setattr(sampler, "_segment_kernel", segment_fields)
    monkeypatch.setattr(sampler, "_transverse_kernel", forbidden)
    monkeypatch.setattr(StaticPolynomialGauge, "vector_potential_shift_t_m", forbidden)
    result = sampler.sample_fields(np.array([[1.e-6, -2.e-6, .0005]]))
    assert set(called) == {"round B", "rectangle B"}
    assert set(result) == {"B_t", "E_v_per_m", "phi_v"}
    for key in result:
        np.testing.assert_array_equal(result[key], expected[key])


@pytest.mark.parametrize("kind", ["low_order_rectangle", "saddle"])
def test_retained_gauss_paths_use_true_B_only_kernel(cuda, monkeypatch, kind):
    from test_compiled_transverse_coil import winding
    potential = (replace(winding().potential, path_order=23) if kind == "low_order_rectangle"
                 else winding(family="saddle").potential)
    original = snapshot(potential)
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    adapter = sampler._transverse["source:0"]
    assert adapter.ends is None and "Gauss" in adapter.path_representation
    points = np.array([[1.e-6, -2.e-6, .0005]])
    expected = sampler.sample(points)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Full A/B node kernel is not a B-only readout")
    monkeypatch.setattr(sampler, "_transverse_kernel", forbidden)
    actual = sampler.sample_fields(points)
    np.testing.assert_array_equal(actual["B_t"], expected["B_t"])
    assert "A_t_m" not in actual


def test_analytic_rectangle_receipt_distinguishes_pack_and_path(cuda):
    original = snapshot(transverse_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    adapter = sampler._transverse["source:0"]
    assert adapter.ends is not None
    assert len(adapter.nodes) == 2*12*12*4
    assert adapter.original_path_node_count == len(adapter.nodes)*24
    sampler.sample_fields(np.array([[1.e-6, -2.e-6, .0005]]))
    receipt = sampler.execution_receipt()
    assert receipt["transverse_paths"]["source:0"]["retained_pack_order"] == 12
    assert "Analytic" in receipt["transverse_paths"]["source:0"]["representation"]
    assert receipt["counts"]["gpu_analytic_rectangle_source_calls"] >= 1
    assert "Analytic" in receipt["execution_identity"]["radial_rectangle_path"]


def test_fields_only_keeps_complete_callback_and_empty_shape_semantics(cuda, monkeypatch):
    original = snapshot(round_potential())
    sampler = prepare_cuda_coil_potential(original, execution_backend="Prefer GPU")
    empty = sampler.sample_fields(np.empty((0, 3)))
    assert set(empty) == {"B_t", "E_v_per_m", "phi_v"} and empty["phi_v"].shape == (0,)
    def changed(self, points):
        return dict(A_t_m=np.ones_like(points)*100., B_t=np.ones_like(points)*7.,
                    E_v_per_m=np.ones_like(points)*5., phi_v=np.full(np.shape(points)[:-1], 3.))
    monkeypatch.setattr(ElectromagneticPotentialSnapshot, "sample", changed)
    result = sampler.sample_fields(np.zeros((128, 3)))
    assert set(result) == {"B_t", "E_v_per_m", "phi_v"}
    np.testing.assert_array_equal(result["B_t"], np.full((128, 3), 7.))
