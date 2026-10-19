"""User-requested, open polymer carriers for the objective electronics.

These are passive engineering envelopes, not OEM dimensions, vacuum boundaries
or electromagnetic material assignments. They do not own the enclosed fields.
"""
from __future__ import annotations

import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, annulus


PROFILE = "objective_electronics_sleeve"
MATERIAL = "insulating_polymer_unspecified"
COMPONENTS = {
    "upper": ("condenser_stigmator", "ac_deflector", "mini_condenser"),
    "lower": ("objective_stigmator", "image_diffraction_deflector"),
}


def _number(part, key):
    value = part.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'sleeve')}.{key} must be finite")
    return float(value)


def validate_sleeve(part, by_key):
    """Validate passive geometry without freezing user-editable package placement."""
    if part.get("mechanical_profile") != PROFILE:
        return
    key = part["key"]
    copied = part.get("mechanical_part_role") == "custom_mechanical_copy"
    if (part.get("mechanical_only") is not True or part.get("axial_vacuum_context_only") is not True
            or (not copied and part.get("material_class") != MATERIAL)):
        raise ValueError(f"{key}: sleeve must remain a passive polymer carrier, not a field or vacuum boundary")
    if any(name in part for name in ("field_source_key", "optical_reference_local_z_mm", "magnetic_circuit_id")):
        raise ValueError(f"{key}: sleeve cannot define an optical field or circuit")
    if not copied and part.get("material_regions"):
        raise ValueError(f"{key}: unspecified polymer sleeve cannot acquire a magnetic material assignment")
    start, end, length, centre, inner, outer = (_number(part, name) for name in (
        "local_start_z_mm", "local_end_z_mm", "length_mm", "local_center_z_mm",
        "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    if not (start < end and 0 < inner < outer and abs(length-(end-start)) < 1e-9
            and abs(centre-(start+end)/2) < 1e-9):
        raise ValueError(f"{key}: sleeve needs consistent positive axial extent and an open bore")
    if copied:
        # Existing copy/edit operations deliberately detach hardware from its
        # original optical installation. Preserve the annulus, not its former
        # containment obligation or a fabricated link to the installed fields.
        return
    side = part.get("sleeve_side")
    if (side not in COMPONENTS or tuple(part.get("sleeve_component_keys", ())) != COMPONENTS[side]
            or part.get("parent_key") != "objective_lens"):
        raise ValueError(f"{key}: sleeve requires its declared upper or lower electronics package")


def validate_nominal_sleeve_clearance(part, by_key):
    """Audit default envelopes, not a save constraint on independently edited parts.

    Installed package declarations record the nominal intent. Later placement,
    size or CAD feature edits require a new collision/containment review rather
    than preventing those existing edit operations.
    """
    if part.get("mechanical_profile") != PROFILE:
        return
    validate_sleeve(part, by_key)
    if part.get("mechanical_part_role") == "custom_mechanical_copy":
        return
    key, side = part["key"], part["sleeve_side"]
    start, end, inner, outer = (_number(part, name) for name in (
        "local_start_z_mm", "local_end_z_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    for child_key in COMPONENTS[side]:
        child = by_key.get(child_key)
        if child is None:
            raise ValueError(f"{key}: missing contained electronics {child_key}")
        if (start > _number(child, "local_start_z_mm")+1e-9
                or end < _number(child, "local_end_z_mm")-1e-9):
            raise ValueError(f"{key}: sleeve must cover {child_key} axially")
        if inner <= _number(child, "mechanical_outer_diameter_mm"):
            raise ValueError(f"{key}: sleeve bore must clear {child_key}")
    winding = by_key.get("objective_lens_excitation_coil")
    if winding is None or outer >= _number(winding, "mechanical_inner_diameter_mm"):
        raise ValueError(f"{key}: sleeve must clear the objective excitation winding")
    upper = by_key.get("objective_upper_pole")
    lower = by_key.get("objective_lower_pole")
    if upper is None or lower is None:
        raise ValueError(f"{key}: objective pole geometry is required")
    if side == "upper":
        if end >= _number(upper, "local_start_z_mm"):
            raise ValueError(f"{key}: sleeve must end before the upper objective pole")
        upstream = by_key.get("probe_tl12_lens")
        if upstream is not None and start <= _number(upstream, "local_end_z_mm"):
            raise ValueError(f"{key}: sleeve must clear the upstream transfer lens")
    elif start <= _number(lower, "local_end_z_mm"):
        raise ValueError(f"{key}: sleeve must start after the lower objective pole")


def validate_objective_sleeves(parts):
    """Optional for older saved modules that do not declare these carriers."""
    by_key = {part["key"]: part for part in parts}
    for part in parts:
        validate_sleeve(part, by_key)


def sleeve_solids(part, by_key):
    if part.get("mechanical_profile") != PROFILE:
        return None
    validate_sleeve(part, by_key)
    start, end = (float(part[name]) for name in ("local_start_z_mm", "local_end_z_mm"))
    inner, outer = (float(part[name])/2 for name in
                    ("mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    copied = part.get("mechanical_part_role") == "custom_mechanical_copy"
    material = str(part.get("material_class", MATERIAL)) if copied else MATERIAL
    note = ("Detached mechanical copy of an open-ended electronics sleeve; material assignments "
            "are CAD metadata only and do not create a field, vacuum boundary or magnetic-material model."
            if copied else
            "Open-ended polymer electronics sleeve requested by the user. "
            "Polymer grade and OEM construction are unspecified; configured dimensions are "
            "nominal engineering clearance allocations. Recheck containment and interference after editing. "
            "No end caps, field, vacuum wall, charge "
            "response or magnetic-material model are added.")
    return (SolidSpec(str(part["key"]), "body", material, annulus(start, end, inner, outer), (),
        status="engineering_reconstruction",
        note=note,
        parameters=dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
                        wall_thickness_mm=outer-inner, sleeve_side=part["sleeve_side"],
                        covered_component_keys=[] if copied else list(part["sleeve_component_keys"]),
                        coverage_status=("detached_mechanical_copy" if copied else "nominal_installed_package"),
                        dimension_status="user_requested_engineering_clearance_not_oem")),)
