"""Saved pole-face lands in the existing axisymmetric reconstruction.

The nose is the complete axial interval between its outer shoulder and pole
face. A face land occupies the final part of that interval at the saved tip
radius. This convention uses existing dimensions; it is not an OEM drawing.
"""


def pole_face_at_end(part, *, key=None):
    """Resolve a native pole's face direction for both material and CAD.

    C1/C2 name their two faces by optical channel within one shared cartridge;
    those names have the opposite axial convention to ordinary upper/lower
    pole pairs. A detached mechanical copy retains its source convention.
    ``key`` supplies the identity omitted from a field-map data payload.
    """
    from temsim.magnetic_circuits import is_custom_mechanical_part

    key = part["key"] if key is None else key
    if is_custom_mechanical_part(part):
        key = part.get("geometry_template_key", key)
    if key == "condenser_lens_1_lower_pole":
        return True
    if key == "condenser_lens_2_upper_pole":
        return False
    return "upper" in key


def pole_face_land_mm(part, nose):
    """Return the saved land, preserving the legacy zero-land contour if absent."""
    from temsim.part_model_3d import _number

    field = "pole_face_land_axial_thickness_mm"
    if field not in part:
        return 0.
    land = _number(part[field], field)
    if not 0 < land < nose:
        raise ValueError(f"{part.get('key', 'pole')}: pole face land requires 0 < land < pole nose axial length")
    return land


def validate_pole_face_land_geometry(part):
    """Validate an active native pole recipe before it can enter a saved draft.

    Explicit CAD replacement and an explicit radial profile both supersede the
    native pole contour. Restoring the native contour reactivates this check.
    """
    if (part.get("mechanical_profile") != "magnetic_pole_piece"
            or "pole_face_land_axial_thickness_mm" not in part
            or part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"
            or part.get("magnetic_radial_profile_mm") is not None):
        return None
    from temsim.part_model_3d import _extent, _pole_section

    start, end = _extent(part)
    return _pole_section(part, start, end)
