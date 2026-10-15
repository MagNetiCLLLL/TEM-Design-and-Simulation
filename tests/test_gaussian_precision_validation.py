"""Fast 2x2 admission retains the existing lower-triangle eigenvalue decision."""
import numpy as np
import pytest

from scripts.experimental_gaussian_beamlets import GaussianBeamlets


def _construct(imaginary):
    return GaussianBeamlets(np.zeros((1, 2)), np.zeros((1, 2)),
        (1j*np.asarray(imaginary))[None], np.ones(1, complex), 2e-12)


def test_random_positive_and_indefinite_precisions_match_eigenvalue_reference():
    rng = np.random.default_rng(6317)
    for index in range(100):
        q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
        eigenvalues = 10.**rng.uniform(-12, 12, size=2)
        if index % 2:
            eigenvalues[0] *= -1
        imaginary = (q*eigenvalues)@q.T
        if np.linalg.eigvalsh(imaginary).min() > 0:
            result = _construct(imaginary)
            np.testing.assert_array_equal(result.precision_m1.imag[0], imaginary)
        else:
            with pytest.raises(ValueError, match='positive imaginary'):
                _construct(imaginary)


@pytest.mark.parametrize('imaginary', [
    [[1e-240, 0.], [0., 1e240]],
    [[1e-150, 0.], [0., 1e150]],
    [[1e-250, 0.], [0., 1e-250]],
    [[1e250, 0.], [0., 1e250]],
    [[1., 1.-1e-15], [1.-1e-15, 1.]],
    [[1., 1.+1e-15], [1.+1e-15, 1.]],
    [[1., 1.], [1., 1.]],
    [[0., 0.], [0., 0.]],
    [[-1., 0.], [0., 1.]],
    # Tiny asymmetry is accepted; eigvalsh historically reads the lower entry.
    [[1., 1.-1e-15], [1.+1e-15, 1.]],
    [[1., 1.+1e-15], [1.-1e-15, 1.]],
])
def test_extreme_or_ambiguous_precisions_keep_previous_admission(imaginary):
    expected = np.linalg.eigvalsh(np.asarray(imaginary)).min() > 0
    if expected:
        _construct(imaginary)
    else:
        with pytest.raises(ValueError, match='positive imaginary'):
            _construct(imaginary)
