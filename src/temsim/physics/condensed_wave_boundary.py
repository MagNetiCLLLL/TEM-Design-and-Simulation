"""Exact static condensation for repeated, fully coupled wave boundary solves.

Only the interior linear response is cached. Every new downstream admittance
is solved against the driven tip again, and all interior complex coefficients
are reconstructed. This is neither a forward-only approximation nor a source.
"""
from hashlib import sha256
import math

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csc_matrix

from temsim.physics.sparse_wave_device import CUDA_SPARSE_BACKEND, sparse_wave_solver
from temsim.physics.wave_device import device_scope
from temsim.physics.wave_execution import check_available_memory


def system_identity(matrix, rhs, boundary_size):
    digest = sha256(str((matrix.shape, matrix.dtype.str, rhs.dtype.str, boundary_size)).encode())
    for value in (matrix.data, matrix.indices, matrix.indptr, rhs):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


class CondensedBoundary:
    def __init__(self, matrix, rhs, boundary_size, *, maximum_working_bytes, cancelled, compute_backend):
        total, inner = len(rhs), len(rhs)-boundary_size
        # Conservative array/factor allowance, not an assurance of LU fill.
        required = 16*(12*matrix.nnz+4*total*boundary_size+128*total)
        if type(maximum_working_bytes) is not int or maximum_working_bytes <= 0 or required > maximum_working_bytes:
            raise MemoryError("Condensed wave boundary exceeds its numerical working-memory budget")
        check_available_memory(required)
        self.required_bytes = required
        self.compute_backend = compute_backend
        if cancelled():
            raise InterruptedError("Condensed wave boundary cancelled")
        interior = matrix[:inner, :inner].tocsc()
        coupling = matrix[:inner, inner:]
        returning = matrix[inner:, :inner]
        self.response = np.empty((inner, boundary_size), dtype=complex)
        with sparse_wave_solver(interior, compute_backend=compute_backend,
                                maximum_working_bytes=maximum_working_bytes,
                                rhs_columns=min(8, boundary_size)) as solve_interior:
            if cancelled():
                raise InterruptedError("Condensed wave boundary cancelled")
            self.driven = solve_interior(rhs[:inner])
            for start in range(0, boundary_size, 8):
                if cancelled():
                    raise InterruptedError("Condensed wave boundary cancelled")
                stop = min(boundary_size, start+8)
                self.response[:, start:stop] = solve_interior(coupling[:, start:stop].toarray(order="F"))
        self.schur = matrix[inner:, inner:].toarray()-returning@self.response
        self.rhs = rhs[inner:]-returning@self.driven
        # No LU factor is retained once the exact interior response is known.
        self.matrix = matrix.copy()
        self.original_rhs = rhs.copy()
        self.boundary_size = boundary_size

    def solve(self, admittance, *, tolerance, cancelled, maximum_working_bytes):
        if cancelled():
            raise InterruptedError("Condensed wave boundary cancelled")
        matrix = self.schur-admittance
        if self.compute_backend == "CPU":
            trace = solve(matrix, self.rhs, check_finite=False)
        else:
            with device_scope("Require GPU", maximum_working_bytes=maximum_working_bytes) as device:
                cp = device.xp
                trace = cp.asnumpy(cp.linalg.solve(cp.asarray(matrix), cp.asarray(self.rhs)))
        field = np.r_[self.driven-self.response@trace, trace]
        defect = self.matrix@field-self.original_rhs
        defect[-self.boundary_size:] -= admittance@trace
        residual = float(np.linalg.norm(defect)/max(np.linalg.norm(self.original_rhs), np.finfo(float).tiny))
        if not np.all(np.isfinite(field)) or not math.isfinite(residual) or residual > tolerance:
            raise ValueError(f"Reconstructed full wave boundary residual {residual:.6g} exceeds {tolerance:.6g}")
        return field, residual


def solve_cached_boundary(matrix, rhs, admittance, cache, *, tolerance=1e-9,
                          maximum_working_bytes=8*1024**3, cancelled=lambda: False, compute_backend="CPU"):
    """A single-entry, exact-matrix cache; changed source/field RHS invalidates it."""
    matrix, rhs, admittance = csc_matrix(matrix, dtype=complex), np.asarray(rhs, complex), np.asarray(admittance, complex)
    count = len(admittance)
    if (rhs.ndim != 1 or matrix.shape != (len(rhs), len(rhs)) or not 0 < count < len(rhs)
            or admittance.shape != (count, count)
            or not all(np.all(np.isfinite(a)) for a in (matrix.data, rhs, admittance))):
        raise ValueError("Condensed wave boundary requires finite matching interior and boundary blocks")
    if not math.isfinite(tolerance) or not 0 < tolerance <= 1e-4:
        raise ValueError("Condensed boundary residual tolerance must be finite and in (0, 1e-4]")
    if cancelled():
        raise InterruptedError("Condensed wave boundary cancelled")
    digest = system_identity(matrix, rhs, count)
    key = (digest, compute_backend)
    system = cache.get(key)
    hit = system is not None
    if system is None:
        cache.clear()  # Release the previous energy's response before allocation.
        system = CondensedBoundary(matrix, rhs, count, maximum_working_bytes=maximum_working_bytes,
                                   cancelled=cancelled, compute_backend=compute_backend)
        cache[key] = system
    elif type(maximum_working_bytes) is not int or system.required_bytes > maximum_working_bytes:
        raise MemoryError("Cached condensed boundary exceeds the requested working-memory budget")
    value, residual = system.solve(admittance, tolerance=tolerance, cancelled=cancelled,
                                  maximum_working_bytes=maximum_working_bytes)
    return value, {"method": "exact-interior-schur-full-reconstruction-v1", "cache_hit": hit,
                   "sparse_backend": "SciPy SuperLU" if compute_backend == "CPU" else CUDA_SPARSE_BACKEND,
                   "matrix_rhs_digest": digest, "linear_residual": residual}
