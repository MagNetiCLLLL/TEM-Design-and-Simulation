"""Complex sparse solves for the driven and coupled tip boundaries."""
from contextlib import contextmanager

from scipy.sparse.linalg import splu

from temsim.physics.wave_device import device_scope


@contextmanager
def sparse_wave_solver(matrix, *, compute_backend, maximum_working_bytes):
    """Reuse the uploaded matrix within one solve, with a CPU reference path.

    The allocation pool bounds CuPy arrays. cuSOLVER also uses internal QR
    workspace outside that pool and may reject a matrix that otherwise fits.
    """
    if compute_backend == "CPU":
        yield splu(matrix.tocsc()).solve
        return

    with device_scope("Require GPU", maximum_working_bytes=maximum_working_bytes) as device:
        from cupyx.scipy.sparse import csr_matrix
        from cupyx.scipy.sparse.linalg import spsolve

        cp = device.xp
        device_matrix = csr_matrix(matrix.tocsr())

        def solve(rhs):
            return cp.asnumpy(spsolve(device_matrix, cp.asarray(rhs)))

        yield solve
