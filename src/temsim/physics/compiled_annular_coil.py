"""Compiled common factors for the existing annular-coil axis expansion.

All Gauss-Legendre loops, derivative polynomials and near-axis terms remain.
The recurrence changes binary64 evaluation order, not the source law. Native
NumPy remains the strict fallback for unavailable compilation or unsafe range.
No thread count, source support, quadrature order or tolerance is changed here.
"""
from functools import lru_cache

import numpy as np
from scipy.constants import mu_0

try:
    from numba import get_num_threads, njit, prange
except ImportError:
    njit = None
    prange = range


# Accelerator admission only. Outside these deliberately broad microscopy
# scales the original operator still evaluates every loop and full axial tail.
# For order <= 12 these bounds keep a**(n+1), (1+u*u)**(n+1.5), and their
# non-cancelling prefactors away from binary64 underflow/overflow. Polynomial
# near-zero products receive an additional per-term normal-range guard below.
_MIN_RADIUS_M = 1e-6
_MAX_RADIUS_M = 10.
_MAX_ABS_NORMALIZED_Z = 1e4
_MIN_ABS_PREFACTOR_T = 1e-100
_MAX_ABS_PREFACTOR_T = 1e100
_MIN_NORMAL = np.finfo(np.float64).tiny


def _jit(function=None, *, parallel=False):
    def decorate(value):
        return value if njit is None else njit(cache=True, nogil=True, fastmath=False,
                                               parallel=parallel)(value)
    return decorate if function is None else decorate(function)


@_jit
def _pairwise_leaf(values, start, count):
    if count < 8:
        total = -0.
        for index in range(count):
            total += values[start+index]
        return total
    r0, r1, r2, r3 = values[start], values[start+1], values[start+2], values[start+3]
    r4, r5, r6, r7 = values[start+4], values[start+5], values[start+6], values[start+7]
    index = 8
    while index < count-count % 8:
        r0 += values[start+index]
        r1 += values[start+index+1]
        r2 += values[start+index+2]
        r3 += values[start+index+3]
        r4 += values[start+index+4]
        r5 += values[start+index+5]
        r6 += values[start+index+6]
        r7 += values[start+index+7]
        index += 8
    total = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
    while index < count:
        total += values[start+index]
        index += 1
    return total


@_jit
def _pairwise_sum(values, start, count):
    """NumPy reduction tree without recursive cross-module JIT cache symbols."""
    if count <= 128:
        return _pairwise_leaf(values, start, count)
    if count <= 256:
        half = count//2
        half -= half % 8
        return _pairwise_leaf(values, start, half)+_pairwise_leaf(values, start+half, count-half)
    # The declared quadrature maximum is 64**2 loops; eight levels exceed
    # its complete pairwise tree depth. Default 12**2 loops need no stack.
    starts = np.empty(8, dtype=np.int64)
    counts = np.empty(8, dtype=np.int64)
    stages = np.zeros(8, dtype=np.int64)
    left_values = np.empty(8, dtype=np.float64)
    starts[0], counts[0], top, value = start, count, 0, 0.
    while top >= 0:
        size = counts[top]
        if size <= 128:
            value = _pairwise_leaf(values, starts[top], size)
            top -= 1
            continue
        half = size//2
        half -= half % 8
        if stages[top] == 0:
            stages[top] = 1
            starts[top+1], counts[top+1], stages[top+1] = starts[top], half, 0
            top += 1
        elif stages[top] == 1:
            left_values[top], stages[top] = value, 2
            starts[top+1], counts[top+1], stages[top+1] = starts[top]+half, size-half, 0
            top += 1
        else:
            value = left_values[top]+value
            top -= 1
    return value


@_jit
def annular_axis_derivatives_one(z, radii, centres, prefactors, polynomials, lengths,
                                 maximum_order):
    """Return orders 0..maximum_order; nonfinite output requests native fallback.

    ``prefactors`` is the original one-dimensional mu0*I/(2*a) loop array.
    The scalar kernel is serial and can be reused inside a larger parallel
    integrator without creating nested workers or an independent clock.
    """
    count = len(radii)
    output = np.empty(maximum_order+1, dtype=np.float64)
    terms = np.empty((maximum_order+1, count), dtype=np.float64)
    for loop in range(count):
        a = radii[loop]
        amplitude = abs(prefactors[loop])
        if (not _MIN_RADIUS_M <= a <= _MAX_RADIUS_M or not np.isfinite(amplitude)
                or (amplitude != 0. and not _MIN_ABS_PREFACTOR_T <= amplitude <= _MAX_ABS_PREFACTOR_T)):
            output[:] = np.nan
            return output
        u = (z-centres[loop])/a
        if not np.isfinite(u) or abs(u) > _MAX_ABS_NORMALIZED_Z:
            output[:] = np.nan
            return output
        base = 1+u*u
        factor = prefactors[loop]/base**1.5
        increment = (1./a)/base
        for order in range(maximum_order+1):
            # Do not accept premature common-factor underflow, which a later
            # polynomial multiplication could otherwise hide as a false zero.
            if not np.isfinite(factor) or (prefactors[loop] != 0. and abs(factor) < _MIN_NORMAL):
                output[:] = np.nan
                return output
            value = polynomials[order, lengths[order]-1]
            for coefficient in range(lengths[order]-2, -1, -1):
                value = polynomials[order, coefficient]+value*u
            term = factor*value
            if (not np.isfinite(term) or (value != 0. and factor != 0.
                    and abs(term) < _MIN_NORMAL)):
                output[:] = np.nan
                return output
            terms[order, loop] = term
            factor *= increment
    for order in range(maximum_order+1):
        output[order] = _pairwise_sum(terms[order], 0, count)
    return output


@_jit
def annular_near_axis_b_one(x, y, z, radii, centres, prefactors, polynomials, lengths):
    """Evaluate the unchanged six-term near-axis B law after caller admission.

    The caller owns the original geometric near-axis predicate and copper
    exclusion. This routine returns False for nonfinite/unsafe numerical range.
    """
    result = np.zeros(3, dtype=np.float64)
    derivatives = annular_axis_derivatives_one(z, radii, centres, prefactors,
                                               polynomials, lengths, 11)
    if not np.isfinite(derivatives).all():
        return False, result
    s = x*x+y*y
    f, fs, fz = 0., 0., 0.
    factorial = 1.
    for order in range(6):
        if order:
            factorial *= order
        coefficient = (-1.)**order/(2.**(2*order+1)*factorial*factorial*(order+1))
        f += coefficient*s**order*derivatives[2*order]
        if order:
            fs += order*coefficient*s**(order-1)*derivatives[2*order]
        fz += coefficient*s**order*derivatives[2*order+1]
    result[0], result[1], result[2] = -x*fz, -y*fz, 2*f+2*s*fs
    return bool(np.isfinite(result).all()), result


@_jit
def _axis_batch(z, radii, centres, prefactors, polynomials, lengths, maximum_order):
    output = np.empty((maximum_order+1, len(z)), dtype=np.float64)
    for index in range(len(z)):
        output[:, index] = annular_axis_derivatives_one(z[index], radii, centres,
            prefactors, polynomials, lengths, maximum_order)
    return output


@_jit(parallel=True)
def _axis_parallel(z, radii, centres, prefactors, polynomials, lengths, maximum_order):
    output = np.empty((maximum_order+1, len(z)), dtype=np.float64)
    for block in prange((len(z)+15)//16):
        start, stop = block*16, min(len(z), (block+1)*16)
        output[:, start:stop] = _axis_batch(z[start:stop], radii, centres, prefactors,
                                           polynomials, lengths, maximum_order)
    return output


def _readonly(array):
    value = np.ascontiguousarray(array)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _polynomial_key():
    from temsim.physics.annular_coil_potential import _AXIS_POLYNOMIALS
    return tuple(tuple(float(value) for value in row) for row in _AXIS_POLYNOMIALS)


def annular_axis_execution_metadata():
    """Numerical policy only; never part of the physical winding definition."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        numba_version = version("numba") if njit is not None else None
    except PackageNotFoundError:
        numba_version = None
    return {"schema": "annular-axis-common-factor-v1", "compiled_available": njit is not None,
            "numba_version": numba_version, "fastmath": False,
            "reduction": "NumPy float64 eight-lane pairwise tree",
            "accelerator_admission": {"radius_m": [_MIN_RADIUS_M, _MAX_RADIUS_M],
                "abs_nonzero_loop_prefactor_t": [_MIN_ABS_PREFACTOR_T, _MAX_ABS_PREFACTOR_T],
                "max_abs_normalized_z": _MAX_ABS_NORMALIZED_Z,
                "subnormal_terms": "Original NumPy fallback"},
            "fallback": "Original direct-power NumPy for unavailable backend or unsafe numerical range"}


@lru_cache(maxsize=128)
def _pack_axis(loops, polynomial_key):
    radii, centres, current = np.asarray(loops, dtype=np.float64).T
    polynomials = np.zeros((len(polynomial_key), max(map(len, polynomial_key))))
    lengths = np.array([len(row) for row in polynomial_key], dtype=np.int64)
    for index, row in enumerate(polynomial_key):
        polynomials[index, :len(row)] = row
    return tuple(_readonly(array) for array in
        (radii, centres, mu_0*current/(2*radii), polynomials, lengths))


def pack_annular_axis(loops):
    """Immutable radii, centres, mu0*I/(2*a), P_n coefficients and lengths."""
    return _pack_axis(tuple(loops), _polynomial_key())


def _axis_values(loops, polynomial_key, z_bytes, maximum_order, parallel):
    z = np.frombuffer(z_bytes, dtype=np.float64)
    args = _pack_axis(loops, polynomial_key)
    kernel = _axis_parallel if parallel else _axis_batch
    result = kernel(z, *args, maximum_order)
    return _readonly(result) if np.isfinite(result).all() else None


_cached_axis_values = lru_cache(maxsize=64)(_axis_values)


def compiled_axis_derivatives(z, loops, maximum_order):
    """Return an owned array, or None to execute the unchanged NumPy reference.

    Exact repeated Z arrays may reuse immutable results in a 64-entry cache.
    Parallel dispatch uses an already admitted Numba thread mask only; this
    function never changes a process/thread resource policy.
    """
    if njit is None:
        return None
    try:
        unique, inverse = np.unique(z, return_inverse=True)
        from temsim.cpu_resources import numerical_thread_budget
        parallel = (len(unique) >= 256 and 1 < get_num_threads() <= numerical_thread_budget())
        # At most 64 x 256 KiB of derivative values plus bounded exact keys;
        # large requests execute normally without retaining another full copy.
        evaluate = _cached_axis_values if (maximum_order+1)*len(unique) <= 32768 else _axis_values
        result = evaluate(tuple(loops), _polynomial_key(),
            np.ascontiguousarray(unique, dtype=np.float64).tobytes(), maximum_order, parallel)
        return None if result is None else result[:, inverse].reshape((maximum_order+1, *z.shape)).copy()
    except Exception:
        # Compilation/backend/range failure is a strict native miss. Public
        # validation and the native finite-value checks keep their own errors.
        return None
