"""Explicit passive details of the upper pack in one shared objective winding.

Material intervals come from the same parent geometry as the original winding;
detached copies use their saved interval snapshot. No optical field is added.
The field solver retains its established homogenized insulated-winding domain.
"""

from copy import deepcopy

from temsim.cad_geometry import SolidSpec, annulus
from temsim.cad_round_lens_end_hardware import FAMILY_PREFIXES, STATUS, _number
from temsim.magnetic_circuits import is_custom_mechanical_part, part_data, radial_profile_mm
from temsim.magnetic_geometry import objective_layer_intervals_mm


PREFIX = "mechanical_objective_upper_end_"
FIELDS = frozenset(PREFIX + name for name in (
    "structure", "insulation_thickness_mm", "inner_insulation_thickness_mm",
    "cooling_plate_thickness_mm", "geometry_status", "geometry_source", "geometry_source_urls"))
LEDGER = "docs/references/cad-objective-upper-hardware.md"


def _intervals(row, by_key):
    """Resolve two material packs; never substitute the old allocation envelope."""
    if is_custom_mechanical_part(row):
        raw = row.get("material_intervals_mm")
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise ValueError(f"{row['key']}: copied Upper Objective winding needs two saved material intervals")
        intervals = []
        for interval in raw:
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError(f"{row['key']}: copied Upper Objective winding needs two saved material intervals")
            values = {"key": row["key"], "start": interval[0], "end": interval[1]}
            intervals.append((_number(values, "start"), _number(values, "end")))
        intervals = tuple(intervals)
    else:
        parent_raw = (by_key or {}).get(row.get("parent_key"))
        if parent_raw is None:
            raise ValueError(f"{row['key']}: Upper Objective winding requires its objective parent geometry")
        parent = part_data(parent_raw)
        for name in ("local_start_z_mm", "upper_yoke_start_local_z_mm", "upper_yoke_end_local_z_mm",
                     "lower_yoke_start_local_z_mm", "lower_yoke_end_local_z_mm", "mechanical_coil_axial_inset_mm"):
            _number(parent, name)
        if parent.get("mechanical_profile") != "magnetic_lens_assembly":
            raise ValueError(f"{row['key']}: Upper Objective winding requires a magnetic lens assembly parent")
        intervals = objective_layer_intervals_mm(parent, parent["local_start_z_mm"], "magnetic_excitation_coil")
    if not (len(intervals) == 2 and intervals[0][0] < intervals[0][1] < intervals[1][0] < intervals[1][1]):
        raise ValueError(f"{row['key']}: Upper Objective winding needs two ordered nonoverlapping material intervals")
    return intervals


def _sections(row, intervals):
    (a, b), (c, d) = intervals
    ri, ro = .5 * row["mechanical_inner_diameter_mm"], .5 * row["mechanical_outer_diameter_mm"]
    axial = row[PREFIX + "insulation_thickness_mm"]
    radial = row[PREFIX + "inner_insulation_thickness_mm"]
    plate = row[PREFIX + "cooling_plate_thickness_mm"]
    return (
        ("upper", a, b, ri + radial, ro, "copper_winding"),
        ("lower", c, d, ri, ro, "copper_winding"),
        ("upper_winding_inner_insulation", a, b, ri, ri + radial, "insulator"),
        ("upper_cooling_plate", a - axial - plate, a - axial, ri + radial, ro, "copper_cooling_plate"),
        ("upper_end_insulation", a - axial, a, ri, ro, "insulator"),
        ("lower_end_insulation", b, b + axial, ri, ro, "insulator"),
        ("lower_cooling_plate", b + axial, b + axial + plate, ri + radial, ro, "copper_cooling_plate"),
    )


def _native_sections(other, parent):
    """Use the same native section precedence as round_lens_solids, without recursion."""
    model = other.get("model_3d", {})
    if model.get("base", {}).get("kind", "existing") != "existing":
        return ()  # Replacement geometry belongs to the actual CAD overlap audit.
    transform = model.get("transform", {})
    if (any(feature.get("enabled", True) for feature in model.get("features", ()))
            or any(tuple(transform.get(name, default)) != default for name, default in (
                ("scale_xy", (1., 1.)), ("offset_mm", (0., 0., 0.)), ("rotation_deg", (0., 0., 0.))))):
        # A cut can remove an old interference; placement can introduce another.
        # Only the final posed CAD geometry may qualify these edited neighbors.
        return ()
    start, end = (_number(other, name) for name in ("local_start_z_mm", "local_end_z_mm"))
    radial = radial_profile_mm(other, end - start)
    if radial is not None:
        outer = [(start + float(z), float(ro)) for z, _, ro in radial]
        inner = [(start + float(z), float(ri)) for z, ri, _ in radial[::-1]]
        return (outer + inner,)
    if other.get("mechanical_profile") == "magnetic_pole_piece":
        from temsim.part_model_3d import _pole_section
        section = _pole_section(other, start, end)
        return (section,) if section is not None else ()
    if is_custom_mechanical_part(other) and "material_intervals_mm" in other:
        intervals = other["material_intervals_mm"]
    else:
        intervals = objective_layer_intervals_mm(parent, parent["local_start_z_mm"], "magnetic_lens_yoke")
    if not intervals:
        intervals = other.get("material_intervals_mm", ((start, end),))
    ri, ro = (_number(other, name) / 2 for name in (
        "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    return tuple([(float(a), ri), (float(a), ro), (float(b), ro), (float(b), ri)] for a, b in intervals)


def _interferes(section, start, end, inner, outer, require_clearance):
    """Clip the actual axial/radial polygon against the added annular section.

    Positive area means material penetration. A positive axial line contact is
    also rejected for copper, while an insulator may touch its support pole.
    Common axial end faces are not mistaken for radial penetration.
    """
    if max(z for z, _ in section) <= start or min(z for z, _ in section) >= end:
        return False
    polygon = list(section)
    for axis, bound, above in ((0, start, True), (0, end, False),
                                (1, inner, True), (1, outer, False)):
        if not polygon:
            return False
        clipped = []
        for previous, current in zip(polygon[-1:] + polygon[:-1], polygon):
            previous_inside = previous[axis] >= bound if above else previous[axis] <= bound
            current_inside = current[axis] >= bound if above else current[axis] <= bound
            if previous_inside != current_inside:
                fraction = (bound - previous[axis]) / (current[axis] - previous[axis])
                clipped.append(tuple(previous[i] + fraction * (current[i] - previous[i]) for i in (0, 1)))
            if current_inside:
                clipped.append(current)
        polygon = clipped
    if len(polygon) < 2:
        return False
    if require_clearance and max(z for z, _ in polygon) - min(z for z, _ in polygon) > 1e-9:
        return True
    # Translate before the area sum so thin films at large absolute Z do not
    # lose precision through subtraction of large products.
    z0, r0 = polygon[0]
    points = [(z - z0, r - r0) for z, r in polygon]
    area = .5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1])))
    return area > 1e-9


def validate_objective_upper_hardware(part, by_key=None):
    """Validate dimensions and actual material interfaces, never nominal row Z."""
    row = part_data(part)
    if row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return
    if not any(name.startswith(PREFIX) for name in row):
        return
    unknown = sorted(name for name in row if name.startswith(PREFIX) and name not in FIELDS)
    if unknown:
        raise ValueError(f"{row['key']}: unknown Upper Objective winding detail fields: {', '.join(unknown)}")
    if any(prefix != PREFIX and any(name.startswith(prefix) for name in row) for prefix in FAMILY_PREFIXES):
        raise ValueError(f"{row['key']}: conflicting round-lens end hardware families")
    if (row.get("mechanical_profile") != "magnetic_excitation_coil"
            or row.get("material_class") != "insulated_copper_winding"
            or row.get(PREFIX + "structure") != "insulated_upper_winding_end_cooling_plates"):
        raise ValueError(f"{row['key']}: Upper Objective winding details require its insulated winding profile and declared structure")
    if row.get(PREFIX + "geometry_status") != STATUS:
        raise ValueError(f"{row['key']}: Upper Objective winding details must declare engineering reconstruction, not OEM")
    source, urls = row.get(PREFIX + "geometry_source"), row.get(PREFIX + "geometry_source_urls")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{row['key']}: Upper Objective winding details geometry_source is required")
    if (not isinstance(urls, (list, tuple)) or not urls
            or any(not isinstance(url, str) or not url.startswith("https://") for url in urls)):
        raise ValueError(f"{row['key']}: Upper Objective winding details geometry_source_urls are required")
    if "magnetic_radial_profile_mm" in row:
        raise ValueError(f"{row['key']}: Upper Objective winding details conflict with another radial material profile")
    inner, outer = (_number(row, name) for name in (
        "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
    axial, radial, plate = (_number(row, PREFIX + name) for name in (
        "insulation_thickness_mm", "inner_insulation_thickness_mm", "cooling_plate_thickness_mm"))
    if not (0 < inner < inner + 2 * radial < outer and axial > 0 and plate > 0):
        raise ValueError(f"{row['key']}: Upper Objective winding requires positive insulation/plate dimensions inside its bore and OD")
    # A context-free field-schema check may precede the full native document.
    # Geometry construction, in contrast, always requires its actual intervals.
    if by_key is None and not is_custom_mechanical_part(row):
        return
    intervals = _intervals(row, by_key)
    (a, b), (c, _) = intervals
    if not (a - axial - plate < a - axial < a < b < b + axial < b + axial + plate < c):
        raise ValueError(f"{row['key']}: Upper Objective winding end layers need representable positive thickness and separated intervals")
    if is_custom_mechanical_part(row):
        return
    parent_key = row.get("parent_key")
    parent = part_data(by_key[parent_key])
    yokes = objective_layer_intervals_mm(parent, parent["local_start_z_mm"], "magnetic_lens_yoke")
    if not (yokes[0][0] <= a - axial - plate and b + axial + plate <= yokes[0][1]):
        raise ValueError(f"{row['key']}: Upper Objective winding details exceed the upper yoke end spaces")
    # The lower plate is alongside the OD60 shank, not the OD96 shoulder.
    # Saved radial profiles take precedence over both pole style and yoke ID.
    for raw in by_key.values():
        other = part_data(raw)
        if other.get("parent_key") != parent_key:
            continue
        profile = other.get("mechanical_profile")
        if profile not in {"magnetic_lens_yoke", "magnetic_pole_piece"}:
            continue
        for section in _native_sections(other, parent):
            for region, lo, hi, ri, ro, material in _sections(row, intervals):
                if region == "lower":
                    continue
                if _interferes(section, lo, hi, ri, ro, material.startswith("copper")):
                    raise ValueError(f"{row['key']}: Upper Objective winding details lack clearance from {other['key']}")


def objective_upper_hardware_solids(part, by_key=None):
    """Two winding regions, upper inner insulation and four passive end layers."""
    row = part_data(part)
    if row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    if not any(name.startswith(PREFIX) for name in row):
        return None
    validate_objective_upper_hardware(row, by_key)
    from temsim.component_representation import non_material_role
    if non_material_role(row):
        return ()
    intervals = _intervals(row, by_key)
    params = {name: deepcopy(value) for name, value in row.items()
              if name in FIELDS or name in {"local_start_z_mm", "local_end_z_mm",
                  "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "material_intervals_mm"}}
    params["resolved_material_intervals_mm"] = deepcopy(intervals)
    evidence = (LEDGER, *row[PREFIX + "geometry_source_urls"])
    note = ("Generic upper-winding inner/end insulation and passive copper cooling plates; sizes are explicit "
            "engineering allocations, not OEM measurements. Plates and insulation are not another current source "
            "or a ferromagnetic return. Upper and lower remain one shared objective field source. "
            "Mechanical CAD subdivides the upper winding pack; the existing field model retains its homogenized "
            "insulated-winding domain. Original pole/yoke, lower winding and sample gap remain unchanged. "
            "Radial magnetic-return connections, fasteners, coolant and thermal/eddy-current response remain unresolved.")
    return tuple(SolidSpec(str(row["key"]), region, material, annulus(lo, hi, ri, ro),
        evidence, STATUS, note, deepcopy(params)) for region, lo, hi, ri, ro, material in _sections(row, intervals))

