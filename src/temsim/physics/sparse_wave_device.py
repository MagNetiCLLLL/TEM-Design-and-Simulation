"""Complex sparse solves for the driven and coupled tip boundaries."""
from contextlib import contextmanager

import numpy as np
from scipy.sparse.linalg import splu

from temsim.physics.wave_device import check_device_memory, device_scope
from temsim.physics.wave_execution import check_available_memory


CUDA_SPARSE_BACKEND = "CuPy CUDA cuDSS reusable LU"


@contextmanager
def sparse_wave_solver(matrix, *, compute_backend, maximum_working_bytes, rhs_columns=1):
    """Factor one sparse operator once and reuse it for all right-hand sides.

    GPU factors stay in cuDSS for the context lifetime. Its native workspace
    estimate is admitted separately from the owned CuPy input/output buffers.
    """
    if compute_backend == "CPU":
        yield splu(matrix.tocsc()).solve
        return

    if type(rhs_columns) is not int or rhs_columns < 1:
        raise ValueError("Sparse solve RHS capacity must be a positive integer")
    try:
        from nvmath.sparse.advanced import DirectSolver, DirectSolverMatrixType, ExecutionCUDA
    except ImportError as error:
        raise RuntimeError("CUDA sparse wave solves require the project's GPU extra "
                           "with nvmath-python and nvidia-cudss-cu12") from error

    with device_scope("Require GPU", maximum_working_bytes=maximum_working_bytes) as device:
        from cupyx.scipy.sparse import csr_matrix

        cp = device.xp
        canonical = matrix.tocsr(copy=True).astype(np.complex128, copy=False)
        canonical.sum_duplicates()
        canonical.sort_indices()
        device_matrix = csr_matrix(canonical)
        size = matrix.shape[0]
        # A fixed column-major layout lets cuDSS reuse the same factorization
        # when callers alternate between a source vector and boundary blocks.
        right_hand_sides = cp.zeros((size, rhs_columns), dtype=cp.complex128, order="F")
        retained = sum(a.nbytes for a in (device_matrix.data, device_matrix.indices,
                                         device_matrix.indptr, right_hand_sides))
        options = {"blocking": True, "sparse_system_type": DirectSolverMatrixType.GENERAL}
        with DirectSolver(device_matrix, right_hand_sides, options=options,
                          execution=ExecutionCUDA(device_id=cp.cuda.Device().id)) as factor:
            estimates = factor.plan().memory_estimates
            # Native cuDSS allocations use its managed CUDA allocator, which
            # is separate from the private CuPy pool's allocation limit.
            required = retained + int(estimates["peak_device_memory"]) + 2*right_hand_sides.nbytes
            if required > maximum_working_bytes:
                raise MemoryError(f"CUDA sparse factorization needs approximately {required} bytes, "
                                  f"above its {maximum_working_bytes}-byte working limit")
            check_device_memory(required)
            check_available_memory(int(estimates["peak_host_memory"]))
            # A numerically small diagonal must not be replaced with a
            # perturbation of the physical wave operator.
            factor.factorization_config.pivot_eps = 0.
            info = factor.factorize()
            if info.info != 0:
                raise np.linalg.LinAlgError(f"CUDA sparse factorization failed with cuDSS info {info.info}")

            def solve(rhs):
                values = np.asarray(rhs, dtype=np.complex128)
                if values.ndim not in (1, 2) or values.shape[0] != size:
                    raise ValueError("Sparse solve RHS must match the operator row count")
                vector = values.ndim == 1
                columns = values[:, None] if vector else values
                result = np.empty(columns.shape, dtype=np.complex128)
                for start in range(0, columns.shape[1], rhs_columns):
                    stop = min(columns.shape[1], start+rhs_columns)
                    right_hand_sides.fill(0)
                    right_hand_sides[:, :stop-start] = cp.asarray(columns[:, start:stop])
                    solution = factor.solve()
                    result[:, start:stop] = cp.asnumpy(solution[:, :stop-start])
                if not np.all(np.isfinite(result)):
                    raise np.linalg.LinAlgError("CUDA sparse solve returned non-finite values")
                return result[:, 0] if vector else result

            yield solve
