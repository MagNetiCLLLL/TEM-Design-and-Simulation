"""Packed exact near-axis copper contributors for the native Gun DG step.

This is an execution adapter, not a new field model. It retains every captured
quadrature loop, six radial terms and full axial tails in original owner/coil
order. Unsupported classes, poses, elliptic-branch queries and extreme values
return to the reference evaluator without applying a compiled trial.
The shared derivative reducer is nonrecursive for reusable native caches.
"""
import math

import numpy as np

from .compiled_magnetic_field import (
    njit, compiled_magnetic_field, prepare_compiled_magnetic_sources,
)
from .frozen_copper_coil_field import FrozenCopperCoilField
from .lens_field_provider import _identity_registration


def _readonly(array, dtype=float):
    result = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(result.tobytes(), dtype=result.dtype).reshape(result.shape)


def pack_copper_instrument_sources(source_records):
    """Capture exact supported laws; None means preserve the complete native sum.

    A source row remains one addition to the global field. Each copper owner
    first sums its own windings, matching CopperCoilAssemblyPotential before
    the instrument sum. No grouping by source kind changes that ordering.
    """
    if njit is None:
        return None
    from .compiled_annular_coil import pack_annular_axis
    sources, terms, layout, coil_geometry, coil_offsets = [], [], [], [], [0]
    radii, centers, prefactors = [], [], []
    polynomials = lengths = None
    copper_count = 0
    for source in source_records:
        if source.known_zero:
            continue
        provider = source.provider
        if type(provider) is FrozenCopperCoilField:
            if not provider.implementation_is_supported():
                return None
            start = len(coil_geometry)
            for coil in provider.potential.coils:
                if not _identity_registration(coil.registration):
                    return None
                packed = pack_annular_axis(coil._loops)
                if packed is None:
                    return None
                r, z, factor, polynomial, size = packed
                if polynomials is not None and (not np.array_equal(polynomials, polynomial)
                                               or not np.array_equal(lengths, size)):
                    return None
                polynomials, lengths = polynomial, size
                radii.extend(r.tolist())
                centers.extend(z.tolist())
                prefactors.extend(factor.tolist())
                coil_geometry.append((coil.inner_radius_m, coil.center_z_m, coil.length_m))
                coil_offsets.append(len(radii))
            layout.append((1, start, len(coil_geometry)))
            copper_count += 1
        else:
            packed = prepare_compiled_magnetic_sources((source,))
            if packed is None:
                return None
            native_sources, native_terms = packed
            start = len(sources)
            offset = len(terms)
            for row in native_sources:
                adjusted = row.copy()
                adjusted[8:10] += offset
                sources.append(adjusted)
            terms.extend(native_terms.tolist())
            layout.append((0, start, len(sources)))
    if copper_count == 0:
        return None
    return (_readonly(sources).reshape(-1, 11), _readonly(terms).reshape(-1, 9),
            _readonly(layout, np.int64).reshape(-1, 3),
            _readonly(coil_geometry).reshape(-1, 3), _readonly(coil_offsets, np.int64),
            _readonly(radii), _readonly(centers), _readonly(prefactors),
            _readonly(polynomials), _readonly(lengths, np.int64))


def _jit(function=None, *, cache=True):
    def decorate(value):
        return value if njit is None else njit(cache=cache, nogil=True, fastmath=False)(value)
    return decorate if function is None else decorate(function)


@_jit
def copper_near_axis_domain(points, geometry):
    """The unchanged native near-axis branch, checked at every trial query.

    This deliberately excludes poses and the entire elliptic branch rather
    than extending an expansion to conductor surfaces. Large ratios use the
    reference law so overflow/underflow cannot silently change its admission.
    """
    for point in points:
        x, y, z = point
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        radius = math.hypot(x, y)
        for inner, center, length in geometry:
            distance = max(abs(z-center)-length/2., 0.)
            scale = math.hypot(inner, distance)
            if not (radius < .01*scale*(1.-32.*np.finfo(np.float64).eps)) or abs(z-center)/inner > 1e6:
                return False
    return True


@_jit(cache=False)
def copper_instrument_batch(points, packed):
    """Return (admitted, B), retaining original per-source accumulation order.

    Numba's persistent caller cache does not track imported helper source
    revisions. Recompile this composed kernel once per interpreter so changed
    scalar range/admission rules cannot be hidden in an older cached caller.
    The resulting native function remains resident throughout the process.
    """
    sources, terms, layout, geometry, offsets, radii, centers, factors, polynomials, lengths = packed
    result = np.zeros_like(points)
    if not copper_near_axis_domain(points, geometry):
        return False, result
    for index in range(len(points)):
        point = points[index]
        for kind, first, stop in layout:
            contribution = np.zeros(3)
            if kind == 0:
                _, contribution = compiled_magnetic_field(point, sources[first:stop], terms, False)
            else:
                for coil in range(first, stop):
                    begin, end = offsets[coil], offsets[coil+1]
                    admitted, value = annular_near_axis_b_one(
                        point[0], point[1], point[2], radii[begin:end], centers[begin:end],
                        factors[begin:end], polynomials, lengths)
                    if not admitted:
                        return False, result
                    contribution += value
            result[index] += contribution
        if not np.isfinite(result[index]).all():
            return False, result
    return True, result


# The shared scalar contains the same derivative and six-term expansion used
# by AnnularCoilPotential; keeping it shared prevents divergent field formulas.
from .compiled_annular_coil import annular_near_axis_b_one
