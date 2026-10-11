"""Complex CPU/CUDA agreement with the public SciPy FFTLog reference."""
import numpy as np
import pytest
from scipy.fft import fht, ifht, fhtoffset

from temsim.physics.radial_fftlog import complex_hankel, hankel_backend


@pytest.mark.parametrize("count", (127, 128, 4096))
@pytest.mark.parametrize("inverse", (False, True))
@pytest.mark.parametrize("order", (0, 1, 2))
def test_cached_cpu_matches_scipy_complex_field_including_nyquist(count, inverse, order):
    rng = np.random.default_rng(837)
    wave = rng.normal(size=count)+1j*rng.normal(size=count)
    spacing = .01
    transform = ifht if inverse else fht
    with hankel_backend("cpu"):
        # Offset changes use the same grid Gamma ratio, not an old offset.
        for initial in (-5., .3, 2.):
            offset = fhtoffset(spacing, order, initial=initial)
            expected = transform(wave.real, spacing, order, offset=offset)+1j*transform(wave.imag, spacing, order, offset=offset)
            actual = complex_hankel(wave, spacing, offset, inverse=inverse, order=order)
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-12


@pytest.mark.parametrize("order", (0, 1, 2))
def test_explicit_cuda_has_same_complex_field_and_no_implicit_fallback(order):
    cupy = pytest.importorskip("cupy")
    try:
        available = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("No accessible CUDA runtime")
    if not available:
        pytest.skip("No CUDA device")
    radius = np.geomspace(1e-8, 100., 131072, endpoint=False)
    wave = radius**(order+1)*np.exp(-radius**2/2-.2j*radius**4)/np.sqrt(np.pi)
    spacing = float(np.log(radius[1]/radius[0]))
    offset = fhtoffset(spacing, order, initial=np.log(radius[0]*radius[-1]))
    scipy_reference = (fht(wave.real, spacing, order, offset=offset)
                       +1j*fht(wave.imag, spacing, order, offset=offset))
    with hankel_backend("cpu"):
        expected = complex_hankel(wave, spacing, offset, order=order)
    with hankel_backend("cuda"):
        actual = complex_hankel(wave, spacing, offset, order=order)
        returned = complex_hankel(actual, spacing, offset, inverse=True, order=order)
    assert np.linalg.norm(actual-scipy_reference)/np.linalg.norm(scipy_reference) < 2e-12
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 2e-12
    assert np.linalg.norm(returned-wave)/np.linalg.norm(wave) < 2e-12


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="cpu or cuda"):
        with hankel_backend("pretend-gpu"):
            pass


def test_kernel_cache_separates_orders_and_default_preserves_order_zero():
    from temsim.physics import radial_fftlog
    rng = np.random.default_rng(315)
    wave = rng.normal(size=128)+1j*rng.normal(size=128)
    spacing = .02
    with hankel_backend("cpu"):
        for order in (0, 1, 2, 0):
            offset = fhtoffset(spacing, order)
            expected = fht(wave.real,spacing,order,offset=offset)+1j*fht(wave.imag,spacing,order,offset=offset)
            actual = complex_hankel(wave,spacing,offset,order=order)
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
        assert len(radial_fftlog._KERNELS.get()) == 3
        np.testing.assert_array_equal(complex_hankel(wave,spacing,offset),actual)


@pytest.mark.parametrize("order", (-1, True, float("nan"), float("inf"), 1j, "1"))
def test_invalid_bessel_orders_are_rejected_before_transform(order):
    with pytest.raises(ValueError, match="Hankel order"):
        complex_hankel(np.ones(8,dtype=complex),.01,0.,order=order)
