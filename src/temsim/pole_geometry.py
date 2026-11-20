"""Axisymmetric pole geometry shared by the material and drawing paths.

The nose is the complete axial interval between its outer shoulder and pole
face. A face land occupies the final part of that interval at the saved tip
radius. This convention uses existing dimensions; it is not an OEM drawing.
"""

from typing import NamedTuple


RECESSED_OBJECTIVE_STYLE = "objective_recessed_back_compound_nose"
RECESSED_AXIAL_FIELDS = (
    "pole_rear_recess_depth_mm", "pole_rear_flange_thickness_mm",
    "pole_barrel_bevel_axial_length_mm", "pole_nose_knee_axial_length_mm",
)
RECESSED_OUTER_DIAMETERS = (
    "pole_barrel_diameter_mm", "pole_shoulder_diameter_mm",
    "pole_nose_root_diameter_mm", "pole_nose_knee_diameter_mm",
)
RECESSED_FIELDS = (*RECESSED_AXIAL_FIELDS, *RECESSED_OUTER_DIAMETERS, "pole_rear_recess_diameter_mm")


def recessed_objective_section(part, start, end, *, key=None):
    """One continuous (z, r) boundary in mm, including both walls of the rear well."""
    from temsim.part_geometry import _dimension

    def mm(name):
        if name not in part:
            raise ValueError(f"Recessed objective pole requires {name}")
        return _dimension(name, part[name])

    bore = mm("mechanical_bore_diameter_mm") / 2
    mouth = mm("pole_mounting_shank_inner_diameter_mm") / 2
    connector = mm("pole_vacuum_connector_outer_diameter_mm") / 2
    stem = mm("pole_stem_outer_diameter_mm") / 2
    well = mm("pole_rear_recess_diameter_mm") / 2
    flange = mm("mechanical_outer_diameter_mm") / 2
    barrel = mm("pole_barrel_diameter_mm") / 2
    shoulder = mm("pole_shoulder_diameter_mm") / 2
    root = mm("pole_nose_root_diameter_mm") / 2
    knee = mm("pole_nose_knee_diameter_mm") / 2
    tip = mm("mechanical_tip_diameter_mm") / 2

    length = end - start
    lip_end = mm("pole_vacuum_connector_axial_length_mm")
    rear = mm("pole_mounting_shank_axial_length_mm")
    flange_end = rear + mm("pole_rear_flange_thickness_mm")
    floor = rear + mm("pole_rear_recess_depth_mm")
    nose_start = length - mm("pole_nose_axial_length_mm")
    bevel_start = nose_start - mm("pole_barrel_bevel_axial_length_mm")
    nose_knee = length - mm("pole_nose_knee_axial_length_mm")
    if not (0 < bore <= mouth < connector < stem < well < shoulder
            and bore < tip < knee < root < shoulder < barrel < flange):
        raise ValueError("Recessed objective pole diameters must leave an open bore, "
                         "a distinct tail and well, and positive nose and flange walls")
    if not (0 < lip_end < rear < flange_end < bevel_start < nose_start < nose_knee < length
            and rear < floor < nose_start):
        raise ValueError("Recessed objective pole axial dimensions must fit the mounting "
                         "shank, rear well, flange, barrel and two nose tapers")
    section = [
        (0., mouth), (0., connector), (lip_end, connector), (lip_end, stem),
        (floor, stem), (floor, well), (rear, well), (rear, flange),
        (flange_end, flange), (flange_end, barrel), (bevel_start, barrel),
        (nose_start, shoulder), (nose_start, root), (nose_knee, knee),
    ]
    land = pole_face_land_mm(part, length - nose_knee)
    if land:
        section.append((length - land, tip))
    section.extend([(length, tip), (length, bore), (lip_end, bore)])
    face_at_end = pole_face_at_end(part, key=key)
    return [(start + z if face_at_end else end - z, r) for z, r in section]


class RadialBand(NamedTuple):
    """Linear inner and outer radii on one positive-length axial interval, in mm."""

    z0: float
    z1: float
    inner0: float
    inner1: float
    outer0: float
    outer1: float


def section_radial_bands(section):
    """Slice a simple meridional polygon, retaining separated rings at the same z."""
    stations = sorted({z for z, _ in section})
    edges = [(a, b) if a[0] < b[0] else (b, a)
             for a, b in zip(section, (*section[1:], section[0])) if a[0] != b[0]]
    bands = []
    for z0, z1 in zip(stations, stations[1:]):
        walls = []
        for (a, ra), (b, rb) in edges:
            if a <= z0 and z1 <= b:
                slope = (rb - ra) / (b - a)
                walls.append((ra + slope * (z0 - a), ra + slope * (z1 - a)))
        walls.sort(key=lambda wall: wall[0] + wall[1])
        for inner, outer in zip(walls[::2], walls[1::2], strict=True):
            bands.append(RadialBand(z0, z1, *inner, *outer))
    return tuple(bands)


def installed_objective_section(part):
    """Native objective section at the installed column coordinates, in mm."""
    from temsim.part_model_3d import _pole_section

    section = _pole_section(dict(part.data, key=part.key), part.start_z_mm, part.end_z_mm)
    return tuple(section or ())


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


def flared_pole_profile_mm(part, length, *, key=None):
    """Build the axial bore and exterior of a pole with a wider return-side bore."""
    from temsim.part_model_3d import _number

    def radius(field):
        return _number(part[field], field) / 2

    bore = radius('mechanical_bore_diameter_mm')
    back = radius('pole_back_bore_diameter_mm')
    tip = radius('mechanical_tip_diameter_mm')
    outer = radius('mechanical_outer_diameter_mm')
    nose = _number(part['pole_nose_axial_length_mm'], 'pole_nose_axial_length_mm')
    if not (0 < bore <= back < outer and bore < tip <= outer and 0 < nose <= length + 1e-9):
        raise ValueError('Flared pole requires a clear tip, a wider or equal back bore, positive walls and a nose inside its length')
    land = pole_face_land_mm(part, nose)
    # Start at the pole face. The bore and exterior taper between the same
    # face land and shoulder, then run parallel to the axis to the return.
    profile = [(0., bore, tip)]
    if land:
        profile.append((land, bore, tip))
    if nose < length - 1e-9:
        profile.append((nose, back, outer))
    profile.append((length, back, outer))
    if pole_face_at_end(part, key=key):
        return [(length-z, inner, outer) for z, inner, outer in reversed(profile)]
    return profile


def validate_pole_face_land_geometry(part):
    """Validate an active native pole recipe before it can enter a saved draft.

    Explicit CAD replacement and an explicit radial profile both supersede the
    native pole contour. Restoring the native contour reactivates this check.
    """
    if (part.get("mechanical_profile") != "magnetic_pole_piece"
            or ("pole_face_land_axial_thickness_mm" not in part
                and part.get("pole_piece_geometry_style") != RECESSED_OBJECTIVE_STYLE)
            or part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"
            or part.get("magnetic_radial_profile_mm") is not None):
        return None
    from temsim.part_model_3d import _extent, _pole_section

    start, end = _extent(part)
    return _pole_section(part, start, end)
