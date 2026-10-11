"""Native two-axis steering windings, independent of optical field strength.

The four closed saddle-loop envelopes are an explicitly approximate mechanical
reconstruction. They are not a winding diagram or CAD model of a CEOS product.
Coordinates are module-local millimetres, matching the other part meshes.
"""

from dataclasses import replace
import math
from numbers import Real

import numpy as np
from temsim.mechanical_profiles import DEFLECTOR_COIL_INSERT, DEFLECTOR_COIL_PROFILES


PROFILE = DEFLECTOR_COIL_INSERT
WINDING_DIMENSIONS = ("winding_layer_thickness_mm", "winding_angular_span_deg", "winding_band_width_mm")
DESCRIPTION = (
    "Generic two-axis winding reconstruction; not OEM winding/CAD; "
    "field strength from linked DPH/DP channel, not mesh."
)


def winding_dimensions(part):
    """Validate native dimensions; only the old profile supplies legacy defaults.

    The return value is (radial layer thickness, full angular span, band width).
    All linear dimensions use mm, like the normal component editor. These are
    reconstruction dimensions, not a current/field or manufacturer calibration.
    """
    key = str(part.get("key", "winding"))
    if part.get("mechanical_profile") not in DEFLECTOR_COIL_PROFILES:
        raise ValueError(f"{key}: unsupported winding profile")

    def number(field, fallback=None):
        value = part.get(field, fallback)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{key}.{field} must be a finite number")
        return float(value)

    inner = number("mechanical_inner_diameter_mm") / 2.0
    outer = number("mechanical_outer_diameter_mm") / 2.0
    length = number("local_end_z_mm") - number("local_start_z_mm")
    if not (0.0 < inner < outer and length > 0.0):
        raise ValueError(f"{key}: windings require an open bore and positive radial/axial thickness")
    legacy = part.get("mechanical_profile") != PROFILE
    layer = number(WINDING_DIMENSIONS[0], 0.4 * (outer - inner) if legacy else None)
    span = number(WINDING_DIMENSIONS[1], 120.0 if legacy else None)
    band = number(WINDING_DIMENSIONS[2], min(0.08 * length, 0.2 * inner) if legacy else None)
    if not 0.0 < layer < (outer - inner) / 2.0:
        raise ValueError(f"{key}: winding layer thickness must leave separation between both radial layers")
    if not 0.0 < span < 180.0:
        raise ValueError(f"{key}: winding angular span must be between 0 and 180 degrees")
    if not 0.0 < band < min(length / 2.0, math.radians(span / 2.0) * (inner + layer / 2.0)):
        raise ValueError(f"{key}: winding band width must leave an open axial and angular window")
    return layer, span, band


def validate_deflector_coil_part(part):
    if part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES:
        winding_dimensions(part)


def _saddle_loop(*, key, region, inner, outer, start, end, angle, count, color,
                 half_angle, band):
    """Extrude a rectangular frame in cylindrical coordinates into one solid.

    Long sides follow the axis and the end bands follow circular arcs. Shared
    faces are removed so the loop has a single closed, consistently wound
    surface, including its open central window.
    """
    from temsim.part_model_3d import _mesh

    window_half_angle = half_angle - band / ((inner + outer) / 2.0)
    # Polygonal inner arcs must also stay outside the declared inner radius.
    # Bound their chord sagitta before placing the inner vertices outward.
    maximum_step = 2.0 * math.acos(inner / (inner + 0.25 * (outer - inner)))
    intervals = max(16, math.ceil(count / 3),
                    math.ceil(2.0 * window_half_angle / maximum_step))
    if intervals > 4096:
        raise ValueError(f"{key}: winding radial thickness is too small to mesh")
    angles = np.r_[-half_angle,
                   np.linspace(-window_half_angle, window_half_angle, intervals + 1),
                   half_angle] + angle
    step = float(np.diff(angles).max())
    inner_vertex_radius = inner / math.cos(step / 2.0)
    if not inner_vertex_radius < outer:
        raise ValueError(f"{key}: winding bands cannot fit their radial envelope")
    radial = (inner_vertex_radius, outer)
    axial = (start, start + band, end - band, end)
    vertices, lookup, boundary = [], {}, {}

    def vertex(index):
        if index not in lookup:
            r, t, z = index
            lookup[index] = len(vertices)
            vertices.append((radial[r] * math.cos(angles[t]),
                             radial[r] * math.sin(angles[t]), axial[z]))
        return lookup[index]

    # Outward vertex order for each face of a (r, theta, z) cell.
    sides = (
        ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
        ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
        ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    )
    for t in range(len(angles) - 1):
        for z in range(3):
            if z == 1 and t not in (0, len(angles) - 2):
                continue
            for side in sides:
                quad = tuple(vertex((r, t + dt, z + dz)) for r, dt, dz in side)
                identity = tuple(sorted(quad))
                if identity in boundary:
                    del boundary[identity]
                else:
                    boundary[identity] = quad
    faces = [(a, b, c) for a, b, c, _ in boundary.values()]
    faces += [(a, c, d) for a, _, c, d in boundary.values()]
    mesh = _mesh(vertices, faces, key, region, "copper_winding", False, DESCRIPTION)
    return replace(mesh, color=color)


def deflector_coil_meshes(part, angular_segments=32, *, drive_key=None):
    """Return two orthogonal pairs within the declared annular envelope.

    The material bore and axial extent come from the part. The subdivision into
    two winding layers is a display construction, never an excitation model.
    """
    from temsim.part_model_3d import _extent, _number, _segments

    key = str(part["key"])
    layer, span, band = winding_dimensions(part)
    count = _segments(angular_segments)
    start, end = _extent(part)
    inner = _number(part["mechanical_inner_diameter_mm"], "mechanical_inner_diameter_mm") / 2.0
    outer = _number(part["mechanical_outer_diameter_mm"], "mechanical_outer_diameter_mm") / 2.0
    if not (0.0 < inner < outer and end > start):
        raise ValueError(f"{key}: windings require an open bore and positive radial/axial thickness")
    layers = ((inner, inner + layer), (outer - layer, outer))
    colors = ((0.94, 0.46, 0.18, 1.0), (0.24, 0.70, 0.87, 1.0))
    meshes = []
    for axis, (lower, upper) in enumerate(layers):
        # Old files chose this proportion separately for each radial layer.
        # Explicit native dimensions instead define the same material band on
        # both axes, and never change silently when another dimension is edited.
        layer_band = (min(0.08 * (end - start), 0.2 * lower)
                      if part.get("mechanical_profile") != PROFILE
                      and "winding_band_width_mm" not in part else band)
        for side in range(2):
            meshes.append(_saddle_loop(
                key=key, region=f"{'xy'[axis]}_winding_{side + 1}",
                inner=lower, outer=upper, start=start, end=end,
                angle=axis * math.pi / 2.0 + side * math.pi,
                count=count, color=colors[axis],
                half_angle=math.radians(span / 2.0), band=layer_band,
            ))
    linked = str(drive_key or "declared DPH/DP channel")
    notes = (
        f"{key}: {DESCRIPTION} Linked field source: {linked}.",
        str(part.get("axial_relation_note", "")),
        str(part.get("radial_relation_note", "")),
        f"Display construction: each winding layer is {layer:g} mm thick; "
        f"opposing loops span {span:g} degrees, with the second pair "
        f"rotated by 90 degrees. End/side band width is {band:g} mm. "
        "These are reconstruction dimensions, not "
        "manufacturer dimensions, wire gauge, turn count or a current calibration.",
        ("Legacy file: unspecified end/side bands retain the old per-layer minimum of "
         "8% axial extent and 20% layer inner radius."
         if part.get("mechanical_profile") != PROFILE and "winding_band_width_mm" not in part else ""),
    )
    return tuple(meshes), tuple(note for note in notes if note)


# Public Python entry point retained for callers of the initial implementation.
corrector_deflector_meshes = deflector_coil_meshes
