"""Opt-in bulk laminate support for the native two-axis winding inserts.

Coordinates and dimensions are module-local millimetres. The patent supports
flexible conductor carriers, insulating layers and bonded cylindrical stacks;
these continuous annuli are a bulk equivalent, not a winding manufacture
drawing, conformal turn insulation, or a calibrated material/field model.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

from temsim.cad_geometry import SolidSpec, annulus
from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES


PREFIX = "winding_support_"
SUPPORT_PROFILE = "bonded_flexible_pcb_equivalent"
SUPPORT_SOURCE = "https://patents.google.com/patent/US20090079531A1/en"
DIMENSIONS = (PREFIX + "carrier_thickness_mm", PREFIX + "cover_thickness_mm")
NOTE = (
    "Bonded flexible-PCB bulk laminate equivalent; patent Figs. 1, 4 and 6 "
    "support the construction class, not this DPH/DP installation. Continuous "
    "cylindrical carriers, radial covers and an interlayer bond are engineering "
    "reconstruction dimensions, not OEM measurements, conformal sidewall/turn "
    "insulation or an exact PCB manufacturing layout. No calibrated electrical, "
    "magnetic, thermal or vacuum material properties are assigned. Optical "
    "field strength and effective field length remain independently configured."
)


def _number(part, field):
    value = part.get(field)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'winding')}.{field} must be a finite number")
    return float(value)


def deflector_support_dimensions(part):
    """Validate explicit support opt-in, retaining no-support legacy geometry."""
    supplied = {name for name in part if str(name).startswith(PREFIX)}
    if not supplied:
        return None
    key = str(part.get("key", "winding"))
    if part.get(PREFIX + "profile") != SUPPORT_PROFILE:
        raise ValueError(f"{key}: winding support requires profile {SUPPORT_PROFILE}")
    if part.get("mechanical_profile") not in DEFLECTOR_COIL_PROFILES:
        raise ValueError(f"{key}: winding support requires a deflector coil insert")
    # Import lazily: the native winding validator also invokes this function.
    from temsim.corrector_model_3d import winding_axial_extent, winding_dimensions

    layer, _, _ = winding_dimensions(part)
    start, end = winding_axial_extent(part)
    inner = _number(part, "mechanical_inner_diameter_mm") / 2.0
    outer = _number(part, "mechanical_outer_diameter_mm") / 2.0
    carrier, cover = (_number(part, name) for name in DIMENSIONS)
    if not carrier > 0.0 or not cover > 0.0:
        raise ValueError(f"{key}: winding support carrier and cover thickness must be positive")
    clear = _number(part, "mechanical_clear_bore_diameter_mm") / 2.0
    if not 0.0 < clear <= inner - carrier:
        raise ValueError(f"{key}: winding support carrier intrudes into the declared clear bore")
    bond = outer - inner - 2.0 * layer - carrier - cover
    if not bond > 0.0:
        raise ValueError(f"{key}: winding support must leave a positive interlayer bond thickness")
    return dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
                carrier_thickness_mm=carrier, cover_thickness_mm=cover,
                bond_thickness_mm=bond, winding_layer_thickness_mm=layer,
                material_inner_radius_mm=inner-carrier,
                material_outer_radius_mm=outer+cover)


def full_material_radial_extent(part):
    """Return full material inner/outer radii, including any support layers."""
    dimensions = deflector_support_dimensions(part)
    if dimensions is not None:
        return dimensions["material_inner_radius_mm"], dimensions["material_outer_radius_mm"]
    from temsim.corrector_model_3d import winding_dimensions
    winding_dimensions(part)
    return (_number(part, "mechanical_inner_diameter_mm") / 2.0,
            _number(part, "mechanical_outer_diameter_mm") / 2.0)


def validate_support_installations(parts):
    """Keep a nominal coaxial insert inside its declared probe-HP material bore.

    Parent ownership permits axial overlap, never penetration into the pole
    material. This nominal-envelope guard does not replace posed BRep checks.
    """
    from temsim.probe_hexapole_geometry import is_probe_hexapole, probe_hexapole_dimensions
    by_key = {part["key"]: part for part in parts}
    for part in parts:
        dimensions = deflector_support_dimensions(part)
        host = by_key.get(part.get("parent_key"))
        if dimensions is None or host is None or not is_probe_hexapole(host):
            continue
        bore = probe_hexapole_dimensions(host)["inner_radius_mm"]
        if dimensions["material_outer_radius_mm"] > bore + 1e-9:
            raise ValueError(f"{part['key']}: winding support exceeds {host['key']} material bore")


def deflector_support_solids(part):
    """Five separate annuli shared by native viewing and analytic STEP export."""
    d = deflector_support_dimensions(part)
    if d is None:
        return ()
    inner, outer = d["inner_radius_mm"], d["outer_radius_mm"]
    layer, carrier, cover = (d[name] for name in (
        "winding_layer_thickness_mm", "carrier_thickness_mm", "cover_thickness_mm"))
    layers = (
        ("x_carrier", "insulator", inner-carrier, inner),
        ("x_radial_cover", "insulator", inner+layer, inner+layer+cover),
        ("interlayer_bond", "bonding_material_unspecified", inner+layer+cover, outer-layer-carrier),
        ("y_carrier", "insulator", outer-layer-carrier, outer-layer),
        ("y_radial_cover", "insulator", outer, outer+cover),
    )
    parameters = {**d, "construction": SUPPORT_PROFILE,
                  "dimension_status": "engineering_reconstruction_not_oem",
                  "topology_source": "US20090079531A1 Figs. 1, 4 and 6",
                  "bond_thickness_is_derived": True,
                  "optical_field_unchanged": True}
    return tuple(SolidSpec(str(part["key"]), region, material,
                          annulus(d["start_mm"], d["end_mm"], low, high),
                          (SUPPORT_SOURCE,), note=NOTE,
                          parameters={**parameters, "layer_inner_radius_mm": low,
                                      "layer_outer_radius_mm": high})
                 for region, material, low, high in layers)


def support_mesh_segments(solids, angular_segments):
    """Bound radial chord error to a quarter of the thinnest laminate layer.

    This refines only the newly added support display. The existing native
    copper mesh is unchanged, and analytic solids remain the contact authority.
    """
    from temsim.part_model_3d import _segments

    count = _segments(angular_segments)
    for solid in solids:
        inner, outer = (solid.parameters[name] for name in (
            "layer_inner_radius_mm", "layer_outer_radius_mm"))
        half_angle = math.acos(inner / (inner + 0.25 * (outer - inner)))
        if half_angle <= 0.0:
            raise ValueError(f"{solid.key}: winding support layer is too thin to mesh")
        count = max(count, math.ceil(math.pi / half_angle))
    # Keep opposing and orthogonal cylindrical quadrants equally tessellated.
    count = 4 * math.ceil(count / 4)
    if count > 4096:
        raise ValueError("Winding support layer is too thin to mesh within 4096 angular segments")
    return count


def annotate_support_mesh(mesh, part):
    """Expose real support dimensions on selectable faces without changing coils."""
    from temsim.part_model_features import _decorate, _normal_data

    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    groups = np.where(np.abs(normals[:, 2]) > 1.0 - 1e-9,
                      np.where(normals[:, 2] > 0.0, "axial_positive", "axial_negative"),
                      np.where(np.einsum("ij,ij->i", centers[:, :2], normals[:, :2]) < 0.0,
                               "inner", "outer"))
    radial_fields = {
        "x_carrier": ("mechanical_inner_diameter_mm", DIMENSIONS[0]),
        "x_radial_cover": ("mechanical_inner_diameter_mm", "winding_layer_thickness_mm", DIMENSIONS[1]),
        "interlayer_bond": ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm",
                            "winding_layer_thickness_mm", *DIMENSIONS),
        "y_carrier": ("mechanical_outer_diameter_mm", "winding_layer_thickness_mm", DIMENSIONS[0]),
        "y_radial_cover": ("mechanical_outer_diameter_mm", DIMENSIONS[1]),
    }[mesh.region]
    radial_paths = tuple(("parts", part["key"], name) for name in radial_fields if name in part)
    axial_fields = ("winding_axial_length_mm",) if "winding_axial_length_mm" in part else ("length_mm",)
    axial_paths = tuple(("parts", part["key"], name) for name in axial_fields if name in part)
    metadata = {
        "inner": {"label": f"{mesh.region} inner surface", "kind": "inner", "parameter_paths": radial_paths},
        "outer": {"label": f"{mesh.region} outer surface", "kind": "outer", "parameter_paths": radial_paths},
        "axial_negative": {"label": f"{mesh.region} upstream face", "kind": "cap", "parameter_paths": axial_paths},
        "axial_positive": {"label": f"{mesh.region} downstream face", "kind": "cap", "parameter_paths": axial_paths},
    }
    return _decorate(mesh, groups, metadata, normals=normals)
