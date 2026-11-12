"""Reusable CUDA factorization and bounded sparse-wave execution."""
import numpy as np
import pytest
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from temsim.physics import sparse_wave_device
from test_tip_wave_pipeline import coherent_cuda


def operator(size=144):
    return diags((np.full(size-1, -.6+.2j), np.full(size, 4.-.5j),
                  np.full(size-1, -1.2+.3j)), (-1, 0, 1), format="csc")


def test_cuda_factorization_is_reused_across_vectors_and_rhs_blocks(coherent_cuda, monkeypatch):
    from nvmath.sparse.advanced import DirectSolver
    from cupyx.scipy.sparse import linalg

    matrix = operator()
    reference = splu(matrix)
    counts = dict(plan=0, factorize=0, solve=0)
    for name in counts:
        original = getattr(DirectSolver, name)

        def tracked(self, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(self, **kwargs)

        monkeypatch.setattr(DirectSolver, name, tracked)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A CUDA sparse solve must reuse its GPU factorization")

    monkeypatch.setattr(sparse_wave_device, "splu", forbidden)
    monkeypatch.setattr(linalg, "spsolve", forbidden)
    rng = np.random.default_rng(71)
    inputs = [rng.normal(size=144)+1j*rng.normal(size=144),
              rng.normal(size=(144, 17))+1j*rng.normal(size=(144, 17)),
              rng.normal(size=(144, 3))+1j*rng.normal(size=(144, 3))]
    with sparse_wave_device.sparse_wave_solver(matrix, compute_backend="Require GPU",
            maximum_working_bytes=256*1024**2, rhs_columns=8) as solve:
        for rhs in inputs:
            actual = solve(rhs)
            assert actual.shape == rhs.shape
            np.testing.assert_allclose(actual, reference.solve(rhs), rtol=1e-11, atol=1e-11)
    assert counts == dict(plan=1, factorize=1, solve=5)


def test_cuda_plan_memory_admission_precedes_factorization(coherent_cuda, monkeypatch):
    from nvmath.sparse.advanced import DirectSolver

    def forbidden(*_args, **_kwargs):
        pytest.fail("An oversized plan must fail before factorization")

    monkeypatch.setattr(DirectSolver, "factorize", forbidden)
    with pytest.raises(MemoryError, match="CUDA sparse factorization needs"):
        with sparse_wave_device.sparse_wave_solver(operator(40), compute_backend="Require GPU",
                maximum_working_bytes=64*1024, rhs_columns=4):
            pytest.fail("An oversized CUDA factorization was admitted")


def test_cuda_factor_resources_close_after_caller_failure(coherent_cuda, monkeypatch):
    from nvmath.sparse.advanced import DirectSolver

    closed = []
    original = DirectSolver.free

    def track_free(self):
        closed.append(self)
        return original(self)

    monkeypatch.setattr(DirectSolver, "free", track_free)
    with pytest.raises(InterruptedError, match="Synthetic cancellation"):
        with sparse_wave_device.sparse_wave_solver(operator(), compute_backend="Require GPU",
                maximum_working_bytes=256*1024**2) as solve:
            solve(np.ones(144))
            raise InterruptedError("Synthetic cancellation")
    assert len(closed) == 1
    assert not closed[0].valid_state


def test_cuda_singular_operator_cannot_return_a_wave(coherent_cuda):
    from scipy.sparse import csc_matrix

    values = np.eye(40, dtype=complex)
    values[:2, :2] = 1.
    matrix = csc_matrix(values)
    with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
        with sparse_wave_device.sparse_wave_solver(matrix, compute_backend="Require GPU",
                maximum_working_bytes=256*1024**2) as solve:
            solve(np.ones(40))
