"""Exact axis identities proved from complete closed radial winding geometry.

At a source-local axis point, each rotated closed circuit rotates A and B by
the same proper axial rotation. For 2m identical circuits with alternating
current signs, the transverse sum is proportional to
sum_j (-1)**j exp(i*j*pi/m), and the axial sum to sum_j (-1)**j. Both vanish
for m=2 or m=3. This is independent of axial position and pack quadrature.

Only an exact descriptor identity is admitted here. A source label, a small
field value, an approximately uniform angle grid, or a near-axis point cannot
prove this identity. Potential derivatives must retain their ordinary values:
quadrupole gradients and hexapole Hessians need not vanish on the axis.
"""
from collections.abc import Mapping
import math
from numbers import Integral, Real


_DESCRIPTOR_FIELDS = frozenset((
    "family", "coil_index", "station_index", "group", "current_sign",
    "azimuth_deg", "center_z_mm", "radial_start_mm", "radial_end_mm",
    "outer_width_mm", "outer_height_mm", "window_width_mm", "window_height_mm",
))
_GEOMETRY_FIELDS = _DESCRIPTOR_FIELDS - {
    "coil_index", "group", "current_sign", "azimuth_deg",
}


def complete_radial_multipole_axis_symmetry(coils, multipole_order):
    """Prove exact on-axis A=B=0 for every populated electrical group.

    Each x/y current channel multiplies one complete group by a scalar; the
    two currents may have arbitrary magnitudes and signs. The caller must
    apply the result only where source-local x == y == 0. Returning False is
    conservative: descriptors whose arithmetic does not reproduce the exact
    declared angle sequence use ordinary field integration without repair.
    """
    if (isinstance(multipole_order, bool)
            or not isinstance(multipole_order, Integral)
            or multipole_order not in (2, 3)
            or not isinstance(coils, (tuple, list)) or not coils):
        return False
    groups = {}
    identifiers = set()
    for coil in coils:
        if not isinstance(coil, Mapping) or set(coil) != _DESCRIPTOR_FIELDS:
            return False
        if coil["family"] != "radial_rectangle" or coil["group"] not in ("x", "y"):
            return False
        index, station, sign = (coil[key] for key in ("coil_index", "station_index", "current_sign"))
        if (isinstance(index, bool) or not isinstance(index, Integral) or index < 1
                or index in identifiers or isinstance(station, bool)
                or not isinstance(station, Integral) or station < 0
                or isinstance(sign, bool) or sign not in (-1, 1)):
            return False
        identifiers.add(index)
        for key in _DESCRIPTOR_FIELDS - {"family", "coil_index", "station_index", "group", "current_sign"}:
            value = coil[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                return False
        groups.setdefault(coil["group"], []).append(coil)
    for rows in groups.values():
        if len(rows) != 2*multipole_order:
            return False
        ordered = sorted(rows, key=lambda row: row["coil_index"])
        first = ordered[0]
        for index, row in enumerate(ordered):
            if (any(row[key] != first[key] for key in _GEOMETRY_FIELDS)
                    or row["current_sign"] != first["current_sign"]*((-1)**index)
                    or row["azimuth_deg"] != first["azimuth_deg"] + index*180/multipole_order):
                return False
    return True
