"""Wave storage ownership remains independent of the numerical accelerator."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics import multiplane_wave as plane


def values():
    x = np.linspace(-1., 1., 16)
    return np.exp(-(x[:, None]**2+x[None, :]**2)) * np.exp(.3j*x[None, :])


def wave(amplitude=None):
    return plane.PlaneWave(values() if amplitude is None else amplitude,
                           np.eye(2)*1e-9, np.zeros(2))


def cuda():
    cp = pytest.importorskip("cupy", reason="optional real CUDA verification requires CuPy")
    from temsim.physics.compute_backend import cupy_capability
    status = cupy_capability()
    if not status.available:
        pytest.skip("CuPy CUDA unavailable: " + status.detail)
    return cp


def test_cpu_snapshot_and_cached_norm_are_immutable(monkeypatch):
    source = values()
    initial = wave(source)
    expected = float(np.sum(abs(source)**2))
    assert initial.probability == expected
    source[:] = 0
    with pytest.raises(ValueError):
        initial.amplitude.setflags(write=True)
    monkeypatch.setattr(plane, "_amplitude_probability", lambda *a: pytest.fail("immutable norm should be reused"))
    assert initial.probability == expected
    moved = initial._replace_metadata(origin_m=np.array((1e-9, 2e-9)))
    assert moved.amplitude is initial.amplitude and moved.probability == expected
    np.testing.assert_array_equal(initial.origin_m, np.zeros(2))


def test_cpu_owned_output_still_freezes_caller_buffer():
    initial = wave()
    output = initial.amplitude * 2
    result = initial._replace_owned(output)
    expected = output.copy()
    output[:] = 0
    np.testing.assert_array_equal(result.amplitude, expected)
    assert not result.amplitude.flags.writeable
    assert result.probability == pytest.approx(4*initial.probability, rel=1e-15)


def test_default_replace_preserves_defensive_snapshot_semantics():
    initial = wave()
    output = values()
    replaced = replace(initial, amplitude=output)
    output[:] = 0
    np.testing.assert_array_equal(replaced.amplitude, initial.amplitude)
    assert replaced.amplitude is not initial.amplitude


@pytest.mark.parametrize("metadata", [dict(basis_m=np.zeros((2, 2))),
    dict(origin_m=np.array((np.nan, 0))), dict(curvature_m1=np.array(((0., 2.), (0., 0.)))),
    dict(tilt_rad=np.array((np.inf, 0)))])
def test_fast_metadata_path_retains_validation(metadata):
    initial = wave()
    with pytest.raises(ValueError):
        initial._replace_metadata(**metadata)
    with pytest.raises(ValueError):
        initial._replace_owned(initial.amplitude*1, **metadata)


def test_private_methods_reject_unknown_fields_and_retained_input():
    initial = wave()
    with pytest.raises(TypeError, match="Unknown plane metadata"):
        initial._replace_metadata(amplitude=values())
    with pytest.raises(TypeError, match="Unknown plane metadata"):
        initial._replace_owned(values(), weight=1.)
    with pytest.raises(ValueError, match="retained input"):
        initial._replace_owned(initial.amplitude)
    for bad in (np.zeros((1, 4), complex), np.full((4, 4), np.nan+0j)):
        with pytest.raises(ValueError, match="finite two-dimensional"):
            initial._replace_owned(bad)


def test_real_gpu_public_input_and_default_replace_own_their_storage():
    cp = cuda()
    array = cp.asarray(values())
    initial = wave(array)
    replaced = replace(initial, origin_m=np.ones(2)*1e-9)
    array[:] = 0
    np.testing.assert_array_equal(cp.asnumpy(initial.amplitude), values())
    initial.amplitude[:] = 0
    np.testing.assert_array_equal(cp.asnumpy(replaced.amplitude), values())


def test_real_gpu_owned_operator_output_avoids_copy_and_preserves_input():
    cp = cuda()
    initial = wave(cp.asarray(values()))
    output = initial.amplitude*np.exp(.2j)
    result = initial._replace_owned(output)
    assert result.amplitude is output
    np.testing.assert_array_equal(cp.asnumpy(initial.amplitude), values())
    metadata = result._replace_metadata(origin_m=np.array((3e-9, 0.)))
    assert metadata.amplitude is result.amplitude
    assert result.origin_m[0] == 0.
    with pytest.raises(ValueError, match="alias"):
        initial._replace_owned(initial.amplitude[:, ::-1])
    result.amplitude[0, 0] = cp.nan
    with pytest.raises(ValueError, match="finite two-dimensional"):
        result._replace_metadata(origin_m=np.zeros(2))
    with pytest.raises(ValueError, match="finite two-dimensional"):
        initial._replace_owned(cp.full((4, 4), cp.inf, dtype=cp.complex128))


def test_real_gpu_norm_has_no_stale_cache_and_retains_double_precision():
    cp = cuda()
    initial = wave(cp.asarray(values()))
    expected = float(np.sum(abs(values())**2))
    assert initial.probability == pytest.approx(expected, rel=4e-15)
    initial.amplitude[...] *= 2
    assert initial.probability == pytest.approx(expected*4, rel=4e-15)
    assert "_cpu_probability" not in initial.__dict__
    initial.amplitude[...] = 0
    assert initial.probability == 0.


def test_real_gpu_canonical_chain_matches_cpu_and_keeps_retained_input():
    cp = cuda()
    x = (np.arange(64)-32)*1e-9
    amplitude = np.exp(-(x[:, None]**2+x[None, :]**2)/(4*(3e-9)**2)).astype(complex)
    amplitude /= np.linalg.norm(amplitude)
    cpu = plane.PlaneWave(amplitude, np.eye(2)*1e-9, np.array((1e-10, -2e-10)))
    gpu = replace(cpu, amplitude=cp.asarray(amplitude))
    retained = gpu
    drift = np.eye(4)
    drift[:2, 2:] = np.eye(2)*2e-6
    lens = np.eye(4)
    lens[2:, :2] = np.diag((-100., -120.))
    for matrix in (lens, drift, lens, drift):
        cpu = plane.propagate_plane_wave(cpu, matrix, np.zeros(4), 2.51e-12)
        gpu = plane.propagate_plane_wave(gpu, matrix, np.zeros(4), 2.51e-12)
    np.testing.assert_allclose(cp.asnumpy(gpu.amplitude), cpu.amplitude, atol=3e-14, rtol=3e-12)
    np.testing.assert_array_equal(cp.asnumpy(retained.amplitude), amplitude)
    np.testing.assert_allclose(gpu.basis_m, cpu.basis_m, atol=0, rtol=1e-14)
    np.testing.assert_allclose(gpu.curvature_m1, cpu.curvature_m1, atol=0, rtol=1e-14)
    assert gpu.probability == pytest.approx(cpu.probability, rel=4e-15)
