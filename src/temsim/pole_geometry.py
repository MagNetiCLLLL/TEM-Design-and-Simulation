"""Saved pole-face lands in the existing axisymmetric reconstruction.

The nose is the complete axial interval between its outer shoulder and pole
face. A face land occupies the final part of that interval at the saved tip
radius. This convention uses existing dimensions; it is not an OEM drawing.
"""


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
