"""Exact finite Fourier interpolation with bounded CPU allocation stages."""
import json
import math
import os
import subprocess
import sys
import weakref

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import (
    WaveGridNumerics, WaveMemoryBudgetError, WaveSamplingError,
    _cpu_refinement_working_bytes, apply_resolved_operator, refine_plane_wave,
)


def fixture_wave(shape):
    rng = np.random.default_rng(17)
    amplitude = rng.standard_normal(shape)+1j*rng.standard_normal(shape)
    amplitude *= np.sqrt(.37/np.sum(abs(amplitude)**2))
    return PlaneWave(amplitude, np.array(((2e-8, .3e-8), (-.2e-8, 3e-8))),
                     np.array((7e-6, -8e-6)), np.array(((.2, -.3), (-.3, .4))), np.array((.01, -.02)))


@pytest.mark.parametrize("old,new", [((32, 34), (48, 52)), ((33, 35), (51, 53)),
    ((32, 35), (53, 48)), ((33, 34), (48, 53)), ((31, 37), (31, 67)), ((32, 32), (47, 47))])
def test_tiled_embedding_matches_independent_finite_fourier_sum(old, new):
    wave = fixture_wave(old)
    result = refine_plane_wave(wave, new)
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude), norm="ortho"))
    fy, fx = np.arange(old[0])-old[0]//2, np.arange(old[1])-old[1]//2
    y, x = np.arange(new[0])-new[0]//2, np.arange(new[1])-new[1]//2
    expected = (np.exp(2j*np.pi*np.outer(y, fy)/new[0])@spectrum
                @np.exp(2j*np.pi*np.outer(fx, x)/new[1]))/np.sqrt(math.prod(new))
    np.testing.assert_allclose(result.amplitude, expected, rtol=3e-12, atol=3e-16)
    assert result.amplitude.dtype == np.complex128
    assert result.probability == pytest.approx(.37, abs=3e-15)
    np.testing.assert_array_equal(result.curvature_m1, wave.curvature_m1)
    np.testing.assert_array_equal(result.tilt_rad, wave.tilt_rad)
    np.testing.assert_array_equal(result.origin_m, wave.origin_m)
    with pytest.raises(ValueError):
        result.amplitude.setflags(write=True)


def test_cpu_refinement_uses_only_bounded_fft_tiles_and_never_rolls_full_arrays(monkeypatch):
    from scipy import fft
    import temsim.physics.wave_grid as grid
    wave = fixture_wave((67, 71))
    original = wave.amplitude.copy()
    monkeypatch.setattr(grid, "_FOURIER_TILE_CELLS", 512)
    calls = []
    for name in ("fft", "ifft"):
        operation = getattr(fft, name)
        def tracked(values, *, axis, norm, overwrite_x, workers, _operation=operation):
            assert values.size <= 512
            assert values.dtype == np.complex128 and values.flags.writeable
            assert not np.shares_memory(values, wave.amplitude)
            assert overwrite_x is True and workers == 1 and norm == "ortho"
            calls.append(values.shape)
            return _operation(values, axis=axis, norm=norm, overwrite_x=overwrite_x, workers=workers)
        monkeypatch.setattr(fft, name, tracked)
    def forbidden(*args, **kwargs):
        pytest.fail("CPU refinement allocated a full shifted image")
    monkeypatch.setattr(np.fft, "fftshift", forbidden)
    monkeypatch.setattr(np.fft, "ifftshift", forbidden)
    result = refine_plane_wave(wave, (97, 101))
    assert len(calls) > 4
    np.testing.assert_array_equal(wave.amplitude, original)
    assert result.probability == pytest.approx(wave.probability, rel=1e-14)


def test_explicit_allocation_budget_does_not_weaken_general_operator_guard():
    numerics = WaveGridNumerics(maximum_working_bytes=100_000)
    assert numerics.check_working_memory((32, 32), working_bytes=60_000, retained_bytes=30_000) == 90_000
    with pytest.raises(WaveMemoryBudgetError):
        numerics.check((32, 32))
    with pytest.raises(WaveMemoryBudgetError):
        numerics.check_working_memory((32, 32), working_bytes=80_000, retained_bytes=30_000)
    for value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            numerics.check_working_memory((32, 32), working_bytes=value)


def test_cpu_refinement_budget_includes_immutable_input_output_copy_and_retained_fields(monkeypatch):
    import temsim.physics.wave_grid as grid
    wave = fixture_wave((64, 65))
    retained = 12_345
    required = retained+_cpu_refinement_working_bytes(wave.amplitude.shape, (101, 107))
    monkeypatch.setattr(grid, "_cpu_fourier_embedding",
                        lambda *args, **kwargs: pytest.fail("Refinement allocated before checking its complete budget"))
    with pytest.raises(WaveMemoryBudgetError):
        refine_plane_wave(wave, (101, 107), retained_bytes=retained,
            numerics=WaveGridNumerics(maximum_working_bytes=required-1))


def test_operator_callback_keeps_callers_original_input_after_multiple_refinements(monkeypatch):
    import temsim.physics.wave_grid as grid
    wave = fixture_wave((32, 32))
    observed = []
    original_check = WaveGridNumerics.check_working_memory
    def check(self, shape, *, working_bytes, retained_bytes=0):
        observed.append((shape, working_bytes, retained_bytes))
        return original_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    visited = []
    def operator(current):
        visited.append(current.amplitude.shape)
        if len(visited) < 3:
            raise WaveSamplingError("resolved after two refinements", 1.01)
        return current
    result, _ = apply_resolved_operator(wave, operator, retained_bytes=123,
                                       operator_working_bytes=lambda shape: 80*math.prod(shape))
    assert len(visited) == 3 and result is not wave
    operator_checks = [(shape, retained) for shape, work, retained in observed if work == 80*math.prod(shape)]
    assert operator_checks[0] == ((32, 32), 123)
    assert all(retained == 123+wave.amplitude.nbytes for _, retained in operator_checks[1:])
    assert any(shape == visited[-1] and retained == 123+wave.amplitude.nbytes
               for shape, _, retained in observed)


def test_target_operator_is_checked_before_fourier_refinement_allocation(monkeypatch):
    import temsim.physics.wave_grid as grid
    wave = fixture_wave((32, 32))
    monkeypatch.setattr(grid, "_cpu_fourier_embedding",
                        lambda *args, **kwargs: pytest.fail("Unusable target operator grid was allocated"))
    def operator(current):
        raise WaveSamplingError("target needs more memory", 2.)
    with pytest.raises(WaveMemoryBudgetError):
        apply_resolved_operator(wave, operator, numerics=WaveGridNumerics(maximum_working_bytes=16*1024**2),
            operator_working_bytes=lambda shape: 1 if shape == (32, 32) else 17*1024**2)


def test_budget_retries_requested_resolution_without_unnecessary_minimum_growth():
    wave = fixture_wave((512, 512))
    visited = []
    def operator(current):
        visited.append(current.amplitude.shape)
        if len(visited) == 1:
            raise WaveSamplingError("needs 1.01273 times sampling", 1.01273)
        assert current.amplitude.shape[0] >= math.ceil(512*1.01273*1.05)
        return current
    result, rows = apply_resolved_operator(wave, operator,
        numerics=WaveGridNumerics(maximum_working_bytes=80*1024**2))
    assert visited == [(512, 512), (550, 550)]
    assert rows[0]["budget_limited_growth"] is True
    assert result.probability == pytest.approx(wave.probability, rel=1e-14)


@pytest.mark.parametrize("prepare_once", [False, True])
def test_retry_preparation_releases_obsolete_grids_before_next_operator(prepare_once):
    """Only the caller's original and the current field may remain resident.

    A callback declining support compaction used to leave its ``prepared``
    local pointing to the previous mesh throughout the next operator. That
    mesh was omitted from the operator's explicit allocation reservation.
    """
    wave = fixture_wave((32, 32))
    references = []
    prepared_once = False
    def operator(current):
        references.append(weakref.ref(current.amplitude))
        if len(references) < 4:
            raise WaveSamplingError("controlled bounded retry", 1.5)
        assert references[0]() is wave.amplitude
        assert all(reference() is None for reference in references[1:-1])
        return current
    def prepare(current, error):
        nonlocal prepared_once
        if prepare_once and not prepared_once:
            prepared_once = True
            return current._replace_owned(current.amplitude.copy())
        return current
    result, rows = apply_resolved_operator(wave, operator,
        numerics=WaveGridNumerics(compute_backend="CPU"), prepare_retry=prepare)
    expected = refine_plane_wave(wave, result.amplitude.shape)
    np.testing.assert_allclose(result.amplitude, expected.amplitude, rtol=3e-12, atol=3e-16)
    assert result.probability == pytest.approx(wave.probability, rel=2e-14)
    assert len(rows) == 3


def test_cancellation_between_fft_tiles_keeps_immutable_input(monkeypatch):
    import temsim.physics.wave_grid as grid
    wave = fixture_wave((64, 64))
    original = wave.amplitude.copy()
    monkeypatch.setattr(grid, "_FOURIER_TILE_CELLS", 256)
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls >= 4
    with pytest.raises(InterruptedError):
        refine_plane_wave(wave, (128, 128), cancelled=cancelled)
    np.testing.assert_array_equal(wave.amplitude, original)


_RSS_SCRIPT = r'''
import gc, json, os, sys
sys.modules["cupy"] = None
import numpy as np
from scipy import fft
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import refine_plane_wave, _cpu_refinement_working_bytes
if os.name == "nt":
    import ctypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)]+[(n, ctypes.c_size_t) for n in
            ("peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
    def peak():
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        handle = ctypes.c_void_p(-1)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise OSError("Unable to measure process peak working set")
        return counters.peak
else:
    import resource
    def peak():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform == "darwin" else 1024)
rng = np.random.default_rng(31)
amplitude = rng.standard_normal((1024,1024))+1j*rng.standard_normal((1024,1024))
amplitude /= np.linalg.norm(amplitude)
wave = PlaneWave(amplitude, np.eye(2), np.zeros(2))
del amplitude
fft.fft(np.zeros(64,complex), workers=1)
gc.collect()
before = peak()
if sys.argv[1] == "new":
    result = refine_plane_wave(wave,(1536,1536))
else:
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(wave.amplitude),norm="ortho"))
    padded = np.zeros((1536,1536),complex)
    padded[256:1280,256:1280] = spectrum
    output = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padded),norm="ortho"))
    result = wave._replace_owned(output,basis_m=np.eye(2)*(2/3))
norm = result.probability
after = peak()
selected = result.amplitude[::173,::179].ravel()
print(json.dumps({"increment":after-before,"before":before,"after":after,"norm":norm,
    "samples":np.stack((selected.real,selected.imag),axis=-1).tolist(),
    "estimate":_cpu_refinement_working_bytes((1024,1024),(1536,1536))}))
'''


def test_cpu_refinement_rss_is_lower_than_old_full_array_fft_path():
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMBA_NUM_THREADS="1", NUMBA_DISABLE_CUDA="1")
    records = {}
    for mode in ("old", "new"):
        run = subprocess.run([sys.executable, "-c", _RSS_SCRIPT, mode], env=env,
                             capture_output=True, text=True, check=True, timeout=60)
        records[mode] = json.loads(run.stdout)
    old, new = records["old"], records["new"]
    np.testing.assert_allclose(new["samples"], old["samples"], atol=3e-17, rtol=3e-12)
    assert new["norm"] == pytest.approx(old["norm"], rel=2e-14)
    # Independent fresh processes include native FFT scratch and immutable
    # publication copies. Leave allocator/import slack outside the stage's
    # explicit array model, while requiring a material observed reduction.
    assert new["increment"] < .8*old["increment"], records
    assert new["increment"] <= new["estimate"]+24*1024**2, records
    print("CPU Fourier refinement RSS:", json.dumps({key: {name: row[name] for name in
          ("increment", "estimate", "before", "after")} for key, row in records.items()}))
