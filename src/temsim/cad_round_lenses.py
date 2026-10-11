"""Native round-lens sections as analytic, renderer-neutral material solids.

The shapes are the same module-local millimetre sections as the existing
physical view. No current, optical plane, dimension or material property is
inferred from a patent. Evidence is scoped in docs/references/cad-round-lenses.md.
The caller applies saved CAD features and rigid placement after this base stage.
"""

from copy import deepcopy

from temsim.cad_geometry import SolidSpec, annulus, revolve
from temsim.magnetic_circuits import is_custom_mechanical_part, part_data, radial_profile_mm
from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.mechanical_profiles import (
    C1_C2_POLE_PIECE_CARTRIDGE, MAGNETIC_EXCITATION_COIL,
    MAGNETIC_LENS_ASSEMBLY, MAGNETIC_LENS_HOUSING,
    MAGNETIC_LENS_MECHANICAL_PROFILES, MAGNETIC_LENS_YOKE, MAGNETIC_POLE_PIECE,
)


GENERIC_REFERENCE = "https://patents.google.com/patent/JP2967966B2/en#description"
OBJECTIVE_REFERENCE = "https://patents.google.com/patent/WO2021170762A1/en#description"
MINI_REFERENCE = "https://patents.google.com/patent/US4306149A/en#description"
LEDGER = "docs/references/cad-round-lenses.md"

SUPPORTED_PROFILES = MAGNETIC_LENS_MECHANICAL_PROFILES | {
    MAGNETIC_POLE_PIECE, C1_C2_POLE_PIECE_CARTRIDGE,
}


def _references(row, parent):
    identity = str(row.get("geometry_template_key", row["key"]))
    owner = str(row.get("parent_key", ""))
    if row.get("mechanical_profile") == C1_C2_POLE_PIECE_CARTRIDGE:
        return (LEDGER,)
    references = [LEDGER, GENERIC_REFERENCE]
    if identity.startswith("objective_") or owner == "objective_lens":
        references.append(OBJECTIVE_REFERENCE)
    if identity.startswith("mini_condenser") or owner == "mini_condenser":
        references.append(MINI_REFERENCE)
    for source in (parent, row):
        for name in ("mechanical_pole_mounting_geometry_source_urls", "geometry_source_urls"):
            for url in source.get(name, ()):
                if isinstance(url, str) and url not in references:
                    references.append(url)
    return tuple(references)


def _parameters(row, parent):
    """Expose existing reconstruction inputs without exposing mutable owners."""
    result = {name: deepcopy(value) for name, value in row.items()
              if (name.startswith(("mechanical_", "pole_", "local_"))
                  and name.endswith(("_mm", "_deg")))
              or name in {"length_mm", "magnetic_radial_profile_mm", "material_intervals_mm",
                          "pole_piece_geometry_style", "geometry_template_key"}}
    result["native_material_class"] = row.get("material_class", "Unspecified")
    for name in ("local_start_z_mm", "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
                 "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm",
                 "mechanical_coil_axial_inset_mm"):
        if name in parent:
            result.setdefault("parent_section_parameters", {})[name] = deepcopy(parent[name])
    return result


def _material(row):
    value = str(row.get("material_class", "Unspecified"))
    if "copper" in value or "winding" in value:
        return "copper_winding"
    if value == "soft_magnetic":
        return "soft_magnetic"
    if value == "non_magnetic_structural":
        return "nonmagnetic_housing"
    # In particular, never turn the unresolved mixed C1/C2 carrier into iron.
    return value


def round_lens_solids(part, by_key) -> tuple[SolidSpec, ...] | None:
    """Return native base material, () for assembly references, None otherwise.

    Upper/lower objective intervals and pole orientation come from the same
    helpers as the existing renderer. The returned geometry is unposed. No
    guessed insulation thickness or cooling tube is inserted into an air gap.
    """
    row = part_data(part)
    profile = row.get("mechanical_profile", "")
    if profile in {MAGNETIC_LENS_ASSEMBLY, "integrated_magnetic_lens_channel"}:
        return ()
    if profile not in SUPPORTED_PROFILES:
        return None

    # Local imports prevent a cycle when the native view consumes these specs.
    from temsim.component_representation import non_material_role
    from temsim.part_model_3d import _extent, _number, _pole_section, _radius

    if non_material_role(row):
        return ()
    start, end = _extent(row)
    if end <= start:
        raise ValueError(f"{row['key']}: CAD material requires positive axial extent")
    raw_parent = by_key.get(row.get("parent_key"))
    parent = part_data(raw_parent) if raw_parent is not None else {}
    params = _parameters(row, parent)
    refs = _references(row, parent)
    note = "Native configured material section; public references establish topology, not OEM dimensions."
    status = "engineering_reconstruction"
    if profile == C1_C2_POLE_PIECE_CARTRIDGE:
        status = "mixed_material_cartridge_unresolved"
        note = ("One photo-identified shared C1/C2 carrier; retained contour only. "
                "Its internal material subdivision is unresolved, not homogeneous iron.")

    def emit(shape, region="body", extra=""):
        return SolidSpec(str(row["key"]), region, _material(row), shape,
                         refs, status, note + (" " + extra if extra else ""),
                         deepcopy(params))

    radial = radial_profile_mm(row, end - start)
    if radial is not None:
        section = [(start + float(z), float(outer)) for z, _, outer in radial]
        section += [(start + float(z), float(inner)) for z, inner, _ in radial[::-1]]
        return (emit(revolve(section)),)
    if profile == MAGNETIC_POLE_PIECE:
        section = _pole_section(row, start, end)
        if section is None:
            return None  # Existing unsupported pole style, not a fabricated cylinder.
        notes = []
        if "pole_nose_axial_length_mm" not in row:
            notes.append("Pole nose retains the existing 0.38-times-length reconstruction.")
            params["resolved_pole_nose_axial_length_mm"] = .38 * (end - start)
        if "pole_root_fillet_radius_range_mm" in row:
            notes.append("Fillet range is metadata; configured shoulders remain sharp.")
        return (emit(revolve(section), extra=" ".join(notes)),)

    outer = _radius(row, (("mechanical_outer_diameter_mm", .5),
                          ("mechanical_outer_radius_mm", 1), ("outer_diameter_mm", .5)))
    inner = _radius(row, (("mechanical_inner_diameter_mm", .5),
                          ("mechanical_bore_diameter_mm", .5),
                          ("mechanical_clear_bore_diameter_mm", .5),
                          ("mechanical_bore_radius_mm", 1), ("bore_diameter_mm", .5),
                          ("clear_bore_diameter_mm", .5), ("inner_diameter_mm", .5)))
    if inner is None or outer is None:
        return None
    if not 0 <= inner < outer:
        raise ValueError(f"{row['key']}: material radii require 0 <= inner < outer")
    intervals = ()
    custom = is_custom_mechanical_part(row)
    if custom and "material_intervals_mm" in row:
        intervals = row["material_intervals_mm"]
        if not intervals:
            raise ValueError(f"{row['key']}: material intervals must not be empty")
    elif (not custom
          and profile in {MAGNETIC_EXCITATION_COIL, MAGNETIC_LENS_YOKE}):
        intervals = objective_layer_intervals_mm(parent, parent.get("local_start_z_mm", 0), profile)
    if not intervals and "material_intervals_mm" in row:
        intervals = row["material_intervals_mm"]
        if not intervals:
            raise ValueError(f"{row['key']}: material intervals must not be empty")
    if not intervals:
        return (emit(annulus(start, end, inner, outer)),)
    checked = tuple((_number(a, "interval start"), _number(b, "interval end")) for a, b in intervals)
    if any(b <= a for a, b in checked):
        raise ValueError(f"{row['key']}: material intervals require positive lengths")
    ordered = sorted(checked)
    if any(a < last_end for (_, last_end), (a, _) in zip(ordered, ordered[1:])):
        raise ValueError(f"{row['key']}: material intervals overlap")
    regions = ("upper", "lower") if len(checked) == 2 else tuple(f"interval_{i}" for i in range(len(checked)))
    return tuple(emit(annulus(a, b, inner, outer), region) for region, (a, b) in zip(regions, checked))
