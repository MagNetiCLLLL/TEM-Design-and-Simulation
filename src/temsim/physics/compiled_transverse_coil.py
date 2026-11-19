"""Exact closed-winding B evaluation over the retained Gauss nodes.

This adapter does not interpolate, truncate tails, refit a multipole or change
quadrature. Unsupported poses, overridden laws, material/boundary queries and
unsafe numerical ranges retain the original evaluator. A and its derivatives
remain owned by the reference potential.
"""
from collections import OrderedDict
import json
import math
from threading import RLock

import numpy as np

from .compiled_magnetic_field import njit
from .lens_field_provider import _identity_registration


_PACKED = OrderedDict()
_LOCK = RLock()
_MAX_PACKS = 24


def _readonly(value, dtype=float):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def pack_transverse_source(provider):
    """Capture every original node, in order; None preserves native execution."""
    from .transverse_coil_field import FrozenTransverseCoilField
    if (njit is None or type(provider) is not FrozenTransverseCoilField
            or not provider.implementation_is_supported()):
        return None
    return pack_transverse_potential(provider.potential)


def pack_transverse_potential(potential):
    """Pack a pure winding potential; no fictitious particle/event inputs."""
    from temsim.immutable_json import thaw_json
    from .transverse_coil_field import potential_weighted_nodes, transverse_potential_implementation_is_supported
    if (njit is None or not transverse_potential_implementation_is_supported(potential)
            or not _identity_registration(potential.registration)):
        return None
    with _LOCK:
        previous = _PACKED.get(id(potential))
        if previous is not None and previous[0] is potential:
            _PACKED.move_to_end(id(potential))
            return previous[1]
    nodes, weighted = potential_weighted_nodes(potential)
    active = np.any(weighted != 0., axis=1)
    # Filtering exactly zero current nodes matches the reference operator.
    nodes, weighted = _readonly(nodes[active]), _readonly(weighted[active])
    if not np.isfinite(weighted).all():
        return None
    geometry = []
    for coil in potential.coils:
        angle = math.radians(coil["azimuth_deg"])
        common = [0., coil["center_z_mm"]*.001, math.cos(angle), math.sin(angle)]
        if coil["family"] == "saddle":
            geometry.append(common+[
                coil["start_z_mm"]*.001, coil["end_z_mm"]*.001,
                coil["inner_radius_mm"]*.001, coil["outer_radius_mm"]*.001,
                math.radians(coil["span_deg"]*.5), coil["band_mm"]*.001,
                coil["window_half_angle_rad"]])
        else:
            common[0] = 1.
            geometry.append(common+[
                coil["radial_start_mm"]*.001, coil["radial_end_mm"]*.001,
                coil["outer_width_mm"]*.0005, coil["outer_height_mm"]*.0005,
                coil["window_width_mm"]*.0005, coil["window_height_mm"]*.0005, 0.])
    from .transverse_coil_field import complete_current_layer_axis_symmetry
    axis_zero = complete_current_layer_axis_symmetry(potential)
    packed = (nodes, weighted, _readonly(geometry).reshape(-1, 11), bool(axis_zero))
    with _LOCK:
        _PACKED[id(potential)] = (potential, packed)
        _PACKED.move_to_end(id(potential))
        while len(_PACKED) > _MAX_PACKS:
            _PACKED.popitem(last=False)
    return packed


def _jit(function):
    return function if njit is None else njit(cache=True, nogil=True, fastmath=False)(function)


@_jit
def transverse_vacuum_domain(point, geometry, first, stop):
    """Conservative material/roundoff guard; rejection means native fallback.

    Expanded winding envelopes and contracted open windows ensure a rounded
    boundary query never receives a compiled field in place of native domain
    validation. This guard does not define any axial field cutoff.
    """
    x, y, z = point
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return False
    if max(abs(x), abs(y), abs(z)) > 1.e4:
        return False
    pad = 128.*np.finfo(np.float64).eps*max(abs(x), abs(y), abs(z), 1.)
    for index in range(first, stop):
        row = geometry[index]
        local_z = z-row[1]
        local_x, local_y = x*row[2]+y*row[3], -x*row[3]+y*row[2]
        if row[0] == 1.:
            outer = (row[4]-pad <= local_x <= row[5]+pad
                     and abs(local_y) <= row[6]+pad and abs(local_z) <= row[7]+pad)
            if outer and not (abs(local_y) < row[8]-pad and abs(local_z) < row[9]-pad):
                return False
        else:
            radius = math.hypot(x, y)
            if radius < row[6]-pad or radius > row[7]+pad or z < row[4]-pad or z > row[5]+pad:
                continue
            angle = abs(math.atan2(local_y, local_x))
            angular_pad = pad/max(radius, np.finfo(np.float64).tiny)
            outer = angle <= row[8]+angular_pad
            hole = (angle < row[10]-angular_pad and z > row[4]+row[9]+pad
                    and z < row[5]-row[9]-pad)
            if outer and not hole:
                return False
    return True


@_jit
def transverse_b_one(point, nodes, weighted, geometry, first_node, stop_node,
                     first_coil, stop_coil, axis_zero):
    """Node-order Biot-Savart reduction, without fastmath or field fitting."""
    result = np.zeros(3)
    if not transverse_vacuum_domain(point, geometry, first_coil, stop_coil):
        return False, result
    if axis_zero and point[0] == 0. and point[1] == 0.:
        return True, result
    for index in range(first_node, stop_node):
        dx = point[0]-nodes[index, 0]
        dy = point[1]-nodes[index, 1]
        dz = point[2]-nodes[index, 2]
        # Same three-coordinate reduction used by the native NumPy einsum.
        norm2 = (dx*dx+dz*dz)+dy*dy
        if not math.isfinite(norm2) or norm2 <= np.finfo(np.float64).tiny:
            return False, result
        inv3 = (1./math.sqrt(norm2))/norm2
        wx, wy, wz = weighted[index]
        bx = (wy*dz-wz*dy)*inv3
        by = (wz*dx-wx*dz)*inv3
        bz = (wx*dy-wy*dx)*inv3
        if not (math.isfinite(bx) and math.isfinite(by) and math.isfinite(bz)):
            return False, result
        result[0] += bx
        result[1] += by
        result[2] += bz
    if not np.isfinite(result).all():
        return False, result
    return True, result


@_jit
def transverse_b_batch(points, packed):
    nodes, weighted, geometry, axis_zero = packed
    result = np.zeros_like(points)
    for index in range(len(points)):
        admitted, value = transverse_b_one(points[index], nodes, weighted, geometry,
            0, len(nodes), 0, len(geometry), axis_zero)
        if not admitted:
            return False, result
        result[index] = value
    return True, result


def compiled_transverse_field(provider, positions_m):
    """Return exact B or None; native validation owns every declined query."""
    packed = pack_transverse_source(provider)
    if packed is None:
        return None
    points = np.asarray(positions_m)
    if (points.dtype.kind not in "iuf" or points.shape[-1:] != (3,)
            or not np.isfinite(points).all()):
        return None
    points = np.asarray(points, dtype=float)
    admitted, result = transverse_b_batch(points.reshape(-1, 3), packed)
    return result.reshape(points.shape) if admitted else None
