"""Scalar check reuse must preserve every physical and numerical admission."""
from dataclasses import replace
import weakref

import numpy as np
import pytest

from temsim.physics.immutable_wave_checks import ImmutableWaveCheckCache


def _frozen(shape=(8, 8)):
    values = np.arange(np.prod(shape), dtype=np.complex128).reshape(shape)
    return np.frombuffer(values.tobytes(), dtype=np.complex128).reshape(shape)


@pytest.mark.parametrize("limit", [0., -0., 1e-10])
def test_success_reuses_exact_result_for_the_same_immutable_wave(limit):
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, threshold):
        calls.append((id(value), threshold))
        return float(value.size)
    result = cache.check(amplitude, limit, checker)
    assert cache.check(amplitude, limit, checker) is result
    assert len(calls) == 1 and len(cache) == 1
    with pytest.raises(ValueError):
        amplitude.setflags(write=True)


def test_exact_limits_checker_identity_and_layout_are_distinct():
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, limit):
        calls.append((value.shape, limit))
        return float(value.shape[0])
    for limit in (0., -0., 1e-10, np.nextafter(1e-10, np.inf)):
        cache.check(amplitude, float(limit), checker)
    assert len(calls) == 4
    other_checker = lambda value, limit: -3.
    assert cache.check(amplitude, 0., other_checker) == -3.
    amplitude.shape = (4, 16)
    assert cache.check(amplitude, 0., checker) == 4.
    assert len(calls) == 5


def test_layout_changed_by_checker_is_not_cached():
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, limit):
        calls.append(value.shape)
        value.shape = (4, 16) if value.shape == (8, 8) else (8, 8)
        return 0.
    cache.check(amplitude, 0., checker)
    cache.check(amplitude, 0., checker)
    assert len(calls) == 2 and len(cache) == 0


def _unsafe_array(kind):
    values = np.zeros((8, 8), dtype=np.complex128)
    if kind == "writeable":
        return values
    if kind == "readonly_owner":
        values.setflags(write=False)
        return values
    if kind == "readonly_view":
        view = values.view()
        view.setflags(write=False)
        return view
    if kind == "bytearray":
        result = np.frombuffer(bytearray(values.tobytes()), dtype=np.complex128).reshape(8, 8)
        result.setflags(write=False)
        return result
    if kind == "memoryview":
        return np.frombuffer(memoryview(values.tobytes()), dtype=np.complex128).reshape(8, 8)
    if kind == "subclass":
        class ArraySubclass(np.ndarray):
            pass
        return _frozen().view(ArraySubclass)
    return np.frombuffer(values.real.tobytes(), dtype=np.float64).reshape(8, 8)


@pytest.mark.parametrize("kind", ["writeable", "readonly_owner", "readonly_view",
    "bytearray", "memoryview", "subclass", "other_dtype"])
def test_potentially_mutable_or_nonstandard_arrays_always_run_checker(kind):
    cache, amplitude, calls = ImmutableWaveCheckCache(), _unsafe_array(kind), []
    def checker(value, limit):
        calls.append(1)
        return 0.
    cache.check(amplitude, 0., checker)
    cache.check(amplitude, 0., checker)
    assert len(calls) == 2 and len(cache) == 0


def test_readonly_view_does_not_hide_a_changed_mutable_base():
    cache = ImmutableWaveCheckCache()
    owner = np.zeros((8, 8), dtype=np.complex128)
    amplitude = owner.view()
    amplitude.setflags(write=False)
    checker = lambda value, limit: float(abs(value).sum())
    assert cache.check(amplitude, 0., checker) == 0.
    owner[0, 0] = 3j
    assert cache.check(amplitude, 0., checker) == 3.
    assert len(cache) == 0


def test_gpu_like_objects_are_never_converted_or_cached():
    class DeviceAmplitude:
        def __array__(self, *args, **kwargs):
            raise AssertionError("A cache must not transfer device data")
    cache, amplitude, calls = ImmutableWaveCheckCache(), DeviceAmplitude(), []
    def checker(value, limit):
        assert value is amplitude
        calls.append(1)
        return True
    cache.check(amplitude, 0., checker)
    cache.check(amplitude, 0., checker)
    assert len(calls) == 2 and len(cache) == 0


@pytest.mark.parametrize("result", [False, None, np.float64(0.), np.array(0.), float("nan"), float("inf")])
def test_non_success_or_non_scalar_builtin_results_are_not_cached(result):
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, limit):
        calls.append(1)
        return result
    assert cache.check(amplitude, 0., checker) is result
    assert cache.check(amplitude, 0., checker) is result
    assert len(calls) == 2 and len(cache) == 0


def test_literal_true_is_a_successful_scalar_receipt():
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, limit):
        calls.append(1)
        return True
    assert cache.check(amplitude, 0., checker) is True
    assert cache.check(amplitude, 0., checker) is True
    assert len(calls) == 1


@pytest.mark.parametrize("limit", [True, 1, np.float64(1e-10), float("nan"), float("inf")])
def test_non_exact_finite_float_limits_preserve_original_checker_semantics(limit):
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, threshold):
        assert threshold is limit
        calls.append(1)
        return 0.
    cache.check(amplitude, limit, checker)
    cache.check(amplitude, limit, checker)
    assert len(calls) == 2 and len(cache) == 0


@pytest.mark.parametrize("error", [ValueError("physical admission"), InterruptedError("cancelled")])
def test_exceptions_are_unchanged_and_never_cached(error):
    cache, amplitude, calls = ImmutableWaveCheckCache(), _frozen(), []
    def checker(value, limit):
        calls.append(1)
        raise error
    for _ in range(2):
        with pytest.raises(type(error)) as caught:
            cache.check(amplitude, 0., checker)
        assert caught.value is error
    assert len(calls) == 2 and len(cache) == 0


@pytest.mark.parametrize("gate", ["boundary", "envelope"])
def test_real_gate_failure_is_not_hidden_by_a_success_at_another_limit(gate):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_grid import WaveSamplingError
    from test_potential_wave_carrier import _wave
    cache, calls = ImmutableWaveCheckCache(), []
    if gate == "boundary":
        wave = PlaneWave(np.ones((32, 32), dtype=complex), np.eye(2), np.zeros(2))
        raw = getattr(carrier, "_check_boundary_uncached", carrier._check_boundary)
        failure = carrier.OccupiedCarrierBoundaryError
    else:
        source = _wave(n=64)
        wave = replace(source, amplitude=source.amplitude*(-1.)**np.arange(64)[None, :])
        raw = getattr(carrier, "_check_envelope_sampling_uncached", carrier._check_envelope_sampling)
        failure = WaveSamplingError
    def checker(value, limit):
        calls.append(limit)
        return raw(value, limit)
    cache.check(wave.amplitude, 1., checker)
    cache.check(wave.amplitude, 1., checker)
    for _ in range(2):
        with pytest.raises(failure):
            cache.check(wave.amplitude, 1e-10, checker)
    assert calls == [1., 1e-10, 1e-10]
    assert len(cache) == 1


def test_cache_does_not_retain_array_or_checker_closure():
    cache = ImmutableWaveCheckCache()
    amplitude = _frozen()
    reference = weakref.ref(amplitude)
    def make_checker(retained):
        return lambda value, limit: float(retained.size)
    checker = make_checker(amplitude)
    checker_reference = weakref.ref(checker)
    cache.check(amplitude, 0., checker)
    del checker, amplitude
    assert reference() is None and checker_reference() is None and len(cache) == 0


def test_array_expiry_removes_all_limits_and_cache_owner_can_expire():
    cache, amplitude = ImmutableWaveCheckCache(), _frozen()
    checker = lambda value, limit: 0.
    for limit in (0., .1, .2):
        cache.check(amplitude, limit, checker)
    assert len(cache) == 3
    reference = weakref.ref(amplitude)
    del amplitude
    assert reference() is None and len(cache) == 0
    amplitude = _frozen()
    cache.check(amplitude, 0., checker)
    cache_reference = weakref.ref(cache)
    del cache
    assert cache_reference() is None


def test_entry_identity_is_checked_even_if_an_id_key_is_reused():
    cache, amplitude, other, calls = ImmutableWaveCheckCache(), _frozen(), _frozen(), []
    def checker(value, limit):
        calls.append(1)
        return float(len(calls))
    assert cache.check(amplitude, 0., checker) == 1.
    key = next(iter(cache._entries))
    cache._entries[key] = replace(cache._entries[key], amplitude=weakref.ref(other))
    assert cache.check(amplitude, 0., checker) == 2.


def test_lru_bound_and_clear_never_keep_evicted_wave_storage():
    cache, arrays, calls = ImmutableWaveCheckCache(2), [_frozen() for _ in range(3)], []
    def checker(value, limit):
        calls.append(id(value))
        return 0.
    for index in (0, 1, 0, 2):
        cache.check(arrays[index], 0., checker)
    assert len(cache) == 2 and len(calls) == 3
    cache.check(arrays[1], 0., checker)
    assert len(cache) == 2 and len(calls) == 4
    cache.clear()
    assert len(cache) == 0


@pytest.mark.parametrize("capacity", [True, 0, -1, 2.5])
def test_capacity_is_a_bounded_positive_exact_integer(capacity):
    with pytest.raises(ValueError, match="maximum_entries"):
        ImmutableWaveCheckCache(capacity)


def test_non_weak_referenceable_checker_runs_unchanged():
    class Checker:
        __slots__ = ("calls",)
        def __init__(self):
            self.calls = 0
        def __call__(self, amplitude, limit):
            self.calls += 1
            return 0.
    cache, amplitude, checker = ImmutableWaveCheckCache(), _frozen(), Checker()
    cache.check(amplitude, 0., checker)
    cache.check(amplitude, 0., checker)
    assert checker.calls == 2 and len(cache) == 0


@pytest.mark.parametrize("providers", ["vacuum", "round_and_hexapole"])
def test_complete_cpu_step_preserves_complex_fields_records_and_other_gates(providers, monkeypatch, record_property):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveGridNumerics
    from test_potential_wave_carrier import _wave, _snapshot, _round, _multipole, ENERGY_EV

    wave = _wave(n=64, curvature=.2, tilt=(2e-7, -1e-7))
    snapshot = _snapshot() if providers == "vacuum" else _snapshot(_round(), _multipole())
    options = dict(energy_ev=ENERGY_EV, kinetic_order=8, scalar_splitting=True,
        higher_order_scalar_splitting=True, residual_compute_backend="CPU",
        support_ledger=NumericalSupportLedger(.02, 1e-6),
        support_projection_policy="weighted_segment_and_retry",
        grid_numerics=WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=128*1024**2))
    original_boundary = getattr(carrier, "_check_boundary_uncached", carrier._check_boundary)
    original_envelope = getattr(carrier, "_check_envelope_sampling_uncached", carrier._check_envelope_sampling)
    original_fft = np.fft.fft2
    original_lattice = carrier._validate_wave_lattice
    original_coverage = carrier._coverage
    original_memory = WaveGridNumerics.check_working_memory
    runs = []
    for reuse in (False, True):
        cache = ImmutableWaveCheckCache()
        counts = dict(fft=0, boundary=0, envelope=0, lattice=0, coverage=0, memory=0, cancellation=0)
        def count_fft(*args, **kwargs):
            counts["fft"] += 1
            return original_fft(*args, **kwargs)
        def boundary(value, limit):
            counts["boundary"] += 1
            return original_boundary(value, limit)
        def envelope(value, limit):
            counts["envelope"] += 1
            return original_envelope(value, limit)
        def lattice(*args, **kwargs):
            counts["lattice"] += 1
            return original_lattice(*args, **kwargs)
        def coverage(*args, **kwargs):
            counts["coverage"] += 1
            return original_coverage(*args, **kwargs)
        def memory(*args, **kwargs):
            counts["memory"] += 1
            return original_memory(*args, **kwargs)
        def cancelled():
            counts["cancellation"] += 1
            return False
        with monkeypatch.context() as patch:
            patch.setattr(np.fft, "fft2", count_fft)
            patch.setattr(carrier, "_check_boundary", (lambda a, b: cache.check(a, b, boundary)) if reuse else boundary)
            patch.setattr(carrier, "_check_envelope_sampling", (lambda a, b: cache.check(a, b, envelope)) if reuse else envelope)
            patch.setattr(carrier, "_validate_wave_lattice", lattice)
            patch.setattr(carrier, "_coverage", coverage)
            patch.setattr(WaveGridNumerics, "check_working_memory", memory)
            output, record = carrier.propagate_carrier_potential(snapshot, wave, (0., 1e-6, 2e-6),
                cancelled=cancelled, **options)
        runs.append((output, record, counts))
    before, after = runs
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        assert getattr(before[0], name).tobytes() == getattr(after[0], name).tobytes()
    assert before[1] == after[1]
    assert after[2]["fft"] < before[2]["fft"]
    assert after[2]["boundary"] < before[2]["boundary"]
    assert after[2]["envelope"] < before[2]["envelope"]
    for gate in ("lattice", "coverage", "memory", "cancellation"):
        assert after[2][gate] == before[2][gate] > 0
    record_property("uncached_checks", str(before[2]))
    record_property("cached_checks", str(after[2]))
    record_property("raw_complex_output", "bitwise identical; all records identical")


def test_reused_scalar_receipts_do_not_bypass_later_memory_admission(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError
    from test_potential_wave_carrier import _wave
    cache, wave = ImmutableWaveCheckCache(), _wave(n=64)
    boundary, envelope = carrier._check_boundary, carrier._check_envelope_sampling
    monkeypatch.setattr(carrier, "_check_boundary", lambda a, b: cache.check(a, b, boundary))
    monkeypatch.setattr(carrier, "_check_envelope_sampling", lambda a, b: cache.check(a, b, envelope))
    carrier._validate_wave(wave, WaveGridNumerics(), 1e-9, 1e-10)
    assert len(cache) == 2
    with pytest.raises(WaveGridBudgetError):
        carrier._validate_wave(wave, WaveGridNumerics(maximum_working_bytes=1024), 1e-9, 1e-10)
