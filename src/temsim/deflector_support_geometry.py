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

from temsim.cad_geometry import SolidSpec, annulus, difference
from temsim.component_representation import non_material_role
from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES


PREFIX = "winding_support_"
SUPPORT_PROFILE = "bonded_flexible_pcb_equivalent"
SUPPORT_SOURCE = "https://patents.google.com/patent/US20090079531A1/en"
DIMENSIONS = (PREFIX + "carrier_thickness_mm", PREFIX + "cover_thickness_mm")
GAP_FILL_FIELD = PREFIX + "fill_conductor_gaps"
GAP_REGIONS = ("x_conductor_gap_insulation", "y_conductor_gap_insulation")
END_COVER_FIELD = PREFIX + "end_cover_thickness_mm"
END_SIDE_FIELD = PREFIX + "end_cover_side"
END_METADATA_FIELDS = tuple(PREFIX + "end_cover_" + name for name in (
    "geometry_status", "source", "source_note"))
END_REGIONS = ("upstream_end_cover", "downstream_end_cover")
END_STATUS = "engineering_reconstruction_family_inference"
FIELDS = frozenset((*DIMENSIONS, GAP_FILL_FIELD, END_COVER_FIELD, END_SIDE_FIELD, *END_METADATA_FIELDS,
                    PREFIX + "profile", PREFIX + "source", PREFIX + "host_fixation_status"))
NOTE = (
    "Bonded flexible-PCB bulk laminate equivalent; patent Figs. 1, 4 and 6 "
    "support the construction class, not this DPH/DP installation. Continuous "
    "cylindrical carriers, radial covers and an interlayer bond are engineering "
    "reconstruction dimensions, not OEM measurements, conformal sidewall/turn "
    "insulation or an exact PCB manufacturing layout. No calibrated electrical, "
    "magnetic, thermal or vacuum material properties are assigned. Optical "
    "field strength and effective field length remain independently configured."
)
GAP_NOTE = (
    "Bulk conductor-gap insulation equivalent inferred from CEOS "
    "US20090079531A1 Fig. 4 and its insulated-conductor/full-surface bonding "
    "description. This opts into a filled laminate construction, not verified "
    "DPH/DP OEM geometry, individual-turn insulation, vias or external mounting. "
    "The existing copper-layer annulus minus its exact bulk windings defines "
    "the volume; no new thickness, copper shape or field calibration is inferred."
)
END_NOTE = (
    "Optional bulk axial edge seals of the already filled flexible laminate. "
    "CEOS US20090079531A1 Fig. 4 supports conductor cover insulation and "
    "full-surface bonding, not a documented DPH2/DPH1/DP axial end-cap design. The "
    "selected annular covers and their saved thickness are an explicit engineering "
    "construction-family inference, not verified OEM geometry or a chosen "
    "resin/coverlay grade. They cover the model's exposed bulk winding ends "
    "without changing the copper allocation, individual turns, connections, "
    "host fixation, effective field length or magnetic excitation."
)


def _number(part, field):
    value = part.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{part.get('key', 'winding')}.{field} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{part.get('key', 'winding')}.{field} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{part.get('key', 'winding')}.{field} must be a finite number")
    return result


def deflector_support_dimensions(part):
    """Validate explicit support opt-in, retaining no-support legacy geometry."""
    supplied = {name for name in part if str(name).startswith(PREFIX)}
    if not supplied:
        return None
    key = str(part.get("key", "winding"))
    unknown = supplied - FIELDS
    if unknown:
        raise ValueError(f"{key}: unknown winding support fields: " + ", ".join(sorted(unknown)))
    if part.get(PREFIX + "profile") != SUPPORT_PROFILE:
        raise ValueError(f"{key}: winding support requires profile {SUPPORT_PROFILE}")
    if non_material_role(part) or part.get("mechanical_profile") not in DEFLECTOR_COIL_PROFILES:
        raise ValueError(f"{key}: winding support requires a deflector coil insert")
    if type(part.get(GAP_FILL_FIELD, False)) is not bool:
        raise ValueError(f"{key}.{GAP_FILL_FIELD} must be a boolean")
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
    end_cover = 0.
    end_side = "both"
    if any(name in part for name in (END_COVER_FIELD, END_SIDE_FIELD, *END_METADATA_FIELDS)):
        end_cover = _number(part, END_COVER_FIELD)
        if end_cover <= 0.:
            raise ValueError(f"{key}: axial end-cover thickness must be positive")
        if part.get(END_METADATA_FIELDS[0]) != END_STATUS:
            raise ValueError(f"{key}: axial end covers require explicit construction-family inference status")
        if part.get(END_METADATA_FIELDS[1]) != SUPPORT_SOURCE:
            raise ValueError(f"{key}: axial end covers require the declared construction-family source")
        note = part.get(END_METADATA_FIELDS[2])
        if not isinstance(note, str) or not note.strip():
            raise ValueError(f"{key}: axial end covers require a nonempty engineering source note")
        end_side = part.get(END_SIDE_FIELD, "both")
        if not isinstance(end_side, str) or end_side not in ("both", "upstream", "downstream"):
            raise ValueError(f"{key}: axial end-cover side must be both, upstream or downstream")
        if not math.isfinite(start-end_cover) or not math.isfinite(end+end_cover):
            raise ValueError(f"{key}: axial end-cover extent must be finite")
    upstream_cover = end_cover if end_side in ("both", "upstream") else 0.
    downstream_cover = end_cover if end_side in ("both", "downstream") else 0.
    if ((upstream_cover and not start-upstream_cover < start)
            or (downstream_cover and not end < end+downstream_cover)):
        raise ValueError(f"{key}: axial end-cover thickness is too small at its allocated position")
    return dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
                carrier_thickness_mm=carrier, cover_thickness_mm=cover,
                bond_thickness_mm=bond, winding_layer_thickness_mm=layer,
                end_cover_thickness_mm=end_cover,
                end_cover_side=end_side,
                material_start_mm=start-upstream_cover, material_end_mm=end+downstream_cover,
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


def full_material_axial_extent(part):
    """Actual laminate extent, separately from copper/optical allocation."""
    dimensions = deflector_support_dimensions(part)
    if dimensions is not None:
        return dimensions["material_start_mm"], dimensions["material_end_mm"]
    from temsim.corrector_model_3d import winding_axial_extent
    return winding_axial_extent(part)


def validate_support_installations(parts):
    """Keep a nominal coaxial insert inside its declared probe-HP material bore.

    Parent ownership permits axial overlap, never penetration into the pole
    material. This nominal-envelope guard does not replace posed BRep checks.
    """
    from temsim.probe_hexapole_geometry import is_probe_hexapole, probe_hexapole_dimensions
    by_key = {part["key"]: part for part in parts}
    for part in parts:
        if part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
            continue
        dimensions = deflector_support_dimensions(part)
        host = by_key.get(part.get("parent_key"))
        if dimensions is None or host is None or not is_probe_hexapole(host):
            continue
        if host.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
            continue
        host_dimensions = probe_hexapole_dimensions(host)
        bore = host_dimensions["inner_radius_mm"]
        if dimensions["material_outer_radius_mm"] > bore + 1e-9:
            raise ValueError(f"{part['key']}: winding support exceeds {host['key']} material bore")
        start, end = full_material_axial_extent(part)
        if start < host_dimensions["start_mm"]-1.e-9 or end > host_dimensions["end_mm"]+1.e-9:
            raise ValueError(f"{part['key']}: full axial winding support exceeds {host['key']} envelope")


def deflector_support_solids(part, windings=None):
    """Radial laminate, conductor-gap insulation and optional axial edge seals."""
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
    result = tuple(SolidSpec(str(part["key"]), region, material,
                          annulus(d["start_mm"], d["end_mm"], low, high),
                          (SUPPORT_SOURCE,), note=NOTE,
                          parameters={**parameters, "layer_inner_radius_mm": low,
                                      "layer_outer_radius_mm": high})
                 for region, material, low, high in layers)
    # The copper's original local interval stays untouched. End covers use a
    # separately exposed material extent, checked against the actual host.
    if d["end_cover_thickness_mm"]:
        for region, a, b in ((END_REGIONS[0], d["material_start_mm"], d["start_mm"]),
                             (END_REGIONS[1], d["end_mm"], d["material_end_mm"])):
            if ((region == END_REGIONS[0] and d["end_cover_side"] == "downstream")
                    or (region == END_REGIONS[1] and d["end_cover_side"] == "upstream")):
                continue
            result += (SolidSpec(str(part["key"]), region, "insulator",
                annulus(a, b, d["material_inner_radius_mm"], d["material_outer_radius_mm"]),
                (SUPPORT_SOURCE,), status=END_STATUS, note=END_NOTE,
                parameters={**parameters, "construction": "bulk_axial_edge_seal",
                    "topology_status": "construction_family_inference_not_verified_dph_dp_oem",
                    "layer_inner_radius_mm": d["material_inner_radius_mm"],
                    "layer_outer_radius_mm": d["material_outer_radius_mm"],
                    "cover_start_mm": a, "cover_end_mm": b}),)
    if not part.get(GAP_FILL_FIELD, False):
        return result
    if windings is None:
        from temsim.cad_multipoles import deflector_insert_winding_solids
        windings = deflector_insert_winding_solids(part)
    for axis, low, high in (("x", inner, inner+layer), ("y", outer-layer, outer)):
        tools = tuple(s.shape for s in windings if s.region.startswith(axis+"_winding_"))
        if len(tools) != 2:
            raise ValueError(f"{part['key']}: conductor-gap insulation requires both {axis} windings")
        result += (SolidSpec(str(part["key"]), axis+"_conductor_gap_insulation", "insulator",
            difference(annulus(d["start_mm"], d["end_mm"], low, high), *tools),
            (SUPPORT_SOURCE,), status="engineering_reconstruction_family_inference", note=GAP_NOTE,
            parameters={**parameters, "layer_inner_radius_mm": low, "layer_outer_radius_mm": high,
                        "construction": "filled_conductor_layer_bulk_equivalent",
                        "topology_status": "construction_family_inference_not_verified_dph_dp_oem",
                        "subtracted_winding_regions": [axis+"_winding_1", axis+"_winding_2"],
                        "new_dimensions_introduced": False}),)
    return result


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
    if mesh.region in GAP_REGIONS:
        paths = tuple(("parts", part["key"], name) for name in (
            "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm",
            "winding_layer_thickness_mm", "winding_angular_span_deg", "winding_band_width_mm",
            "winding_axial_length_mm" if "winding_axial_length_mm" in part else "length_mm")
            if name in part)
        # Each cavity face is inherited from copper; classify it together with
        # its layer boundaries rather than mislabelling angular cuts as cylinders.
        metadata = {"laminate": {"label": "Conductor-gap insulation / winding boundaries",
                    "kind": "material_boundary", "parameter_paths": paths}}
        return _decorate(mesh, np.full(len(mesh.faces), "laminate"), metadata, normals=normals)
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
        END_REGIONS[0]: ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", *DIMENSIONS),
        END_REGIONS[1]: ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", *DIMENSIONS),
    }[mesh.region]
    radial_paths = tuple(("parts", part["key"], name) for name in radial_fields if name in part)
    axial_fields = ("winding_axial_length_mm",) if "winding_axial_length_mm" in part else ("length_mm",)
    if mesh.region in END_REGIONS:
        axial_fields += (END_COVER_FIELD,)
    axial_paths = tuple(("parts", part["key"], name) for name in axial_fields if name in part)
    metadata = {
        "inner": {"label": f"{mesh.region} inner surface", "kind": "inner", "parameter_paths": radial_paths},
        "outer": {"label": f"{mesh.region} outer surface", "kind": "outer", "parameter_paths": radial_paths},
        "axial_negative": {"label": f"{mesh.region} upstream face", "kind": "cap", "parameter_paths": axial_paths},
        "axial_positive": {"label": f"{mesh.region} downstream face", "kind": "cap", "parameter_paths": axial_paths},
    }
    return _decorate(mesh, groups, metadata, normals=normals)
