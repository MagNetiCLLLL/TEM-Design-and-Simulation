"""Material-only profile and declared strip-entry openings of the column tube.

The electrostatic solver retains its existing axisymmetric equivalent liner.
These engineering material details do not change its ID, fields, or clipping.
"""
import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, box, difference, revolve, transformed


CONTINUOUS_TUBE_KEY = "@vacuum_liner:c1_c2_to_upper_objective"
PREFIX = "mechanical_material_tube_"
PROFILE = PREFIX + "outer_profile_z_od_mm"
SLOT_KEYS = PREFIX + "aperture_slot_keys"
CLEARANCE = PREFIX + "aperture_slot_clearance_mm"
STATUS = PREFIX + "status"
SOURCE = PREFIX + "source"
SLOTS = PREFIX + "aperture_slots"
FIELDS = (PROFILE, SLOT_KEYS, CLEARANCE, STATUS, SOURCE)
TERMINAL_KEY = "objective_terminal_vacuum_tube"


def terminal_profile(geometry, length):
    """Clip the installed OD profile to the selected terminal length (local mm)."""
    start, end, outer = (float(geometry["c1_c2_objective_vacuum_tube_" + k])
                         for k in ("start_z_mm", "end_z_mm", "outer_diameter_mm"))
    length = _finite(length, "terminal_tube_length_mm")
    if not 0 < length <= end - start:
        raise ValueError("The objective terminal tube length must fit the installed tube")
    cut = end - length
    points = geometry.get(PROFILE, ((start, outer), (end, outer)))
    for (a, da), (b, db) in zip(points, points[1:]):
        if a <= cut <= b:
            return [(cut, da + (db - da)*(cut-a)/(b-a)), *((z, d) for z, d in points if z > cut)]
    raise ValueError("Terminal tube lies outside the material wall profile")


def terminal_field_part(assembly):
    """Use the same wall and placement for magnetic and nonmagnetic FEM runs."""
    from temsim.part_materials import material_for_region, nonmagnetic_tube_material
    upper = next((p for p in assembly.parts if p.key == "objective_upper_pole"), None)
    if upper is None or "terminal_tube_length_mm" not in upper.data:
        return None
    module = next(m for m in assembly.modules if m.key == upper.module_key)
    liner = next(s for s in assembly.vacuum_liner_segments if s.key == CONTINUOUS_TUBE_KEY)
    points = terminal_profile(module.geometry, upper.data["terminal_tube_length_mm"])
    start, end = points[0][0], points[-1][0]
    origin = liner.end_z_mm - end
    inner = liner.inner_diameter_mm
    material = material_for_region(upper, "terminal_tube") or nonmagnetic_tube_material()
    data = dict(mechanical_profile="magnetic_lens_yoke", material_class="objective_terminal_wall",
                mechanical_inner_diameter_mm=inner, mechanical_outer_diameter_mm=max(d for _, d in points),
                magnetic_radial_profile_mm=[[z-start, inner/2, d/2] for z, d in points],
                material_regions={"body": material})
    return dict(key=TERMINAL_KEY, parent_key="objective_lens", module_key=upper.module_key,
                start_z_mm=start+origin, end_z_mm=end+origin, center_z_mm=(start+end)/2+origin,
                length_mm=end-start, data=data)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_material_tube_geometry(parts, geometry):
    """Validate only explicitly declared material changes; old files still load."""
    if not any(key in geometry for key in FIELDS):
        return
    missing = [key for key in FIELDS if key not in geometry]
    if missing:
        raise ValueError("Incomplete material tube geometry: " + ", ".join(missing))
    start, end, inner, outer = (float(geometry["c1_c2_objective_vacuum_tube_" + key])
        for key in ("start_z_mm", "end_z_mm", "inner_diameter_mm", "outer_diameter_mm"))
    profile = geometry[PROFILE]
    if not isinstance(profile, (list, tuple)) or len(profile) < 2:
        raise ValueError("Material tube profile requires at least two [local Z, OD] pairs")
    points = []
    for point in profile:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Material tube profile entries must be [local Z, OD] pairs")
        z, od = (_finite(value, PROFILE) for value in point)
        if not inner < od <= outer:
            raise ValueError("Material tube OD must exceed its ID and not exceed the original OD")
        if points and z <= points[-1][0]:
            raise ValueError("Material tube profile Z values must be strictly increasing")
        points.append((z, od))
    if abs(points[0][0]-start) > 1e-9 or abs(points[-1][0]-end) > 1e-9:
        raise ValueError("Material tube profile endpoints must cover exactly the existing tube")
    keys = geometry[SLOT_KEYS]
    if (not isinstance(keys, (list, tuple)) or any(not isinstance(key, str) for key in keys)
            or len(set(keys)) != len(keys)):
        raise ValueError("Material tube aperture slot keys must be a unique list of part keys")
    if _finite(geometry[CLEARANCE], CLEARANCE) <= 0:
        raise ValueError("Material tube aperture slot clearance must be positive")
    by_key = {part["key"]: part for part in parts}
    for key in keys:
        part = by_key.get(key)
        if part is None or part.get("aperture_plate_form") != "perforated_strip":
            raise ValueError(f"Material tube slot {key} must name an existing strip aperture")
        z = float(part.get("optical_reference_local_z_mm", part["local_center_z_mm"]))
        thickness = float(part.get("plate_thickness_mm", part.get("active_length_mm", 0.)))
        if not math.isfinite(thickness) or thickness <= 0 or not start <= z-thickness/2 < z+thickness/2 <= end:
            raise ValueError(f"Material tube slot {key} must lie within the continuous tube span")
    pole = by_key.get("objective_upper_pole", {})
    if "terminal_tube_length_mm" in pole:
        cut = terminal_profile(geometry, pole["terminal_tube_length_mm"])[0][0]
        for key in keys:
            aperture = by_key[key]
            slot_end = (aperture.get("optical_reference_local_z_mm", aperture["local_center_z_mm"])
                        + aperture.get("plate_thickness_mm", aperture.get("active_length_mm", 0.))/2 + geometry[CLEARANCE])
            if slot_end > cut:
                raise ValueError("The terminal tube wall overlaps an aperture side-entry slot")
    # An edited OD profile must not silently restore the known C3 pole collision.
    def diameter_at(z):
        for (a, da), (b, db) in zip(points, points[1:]):
            if a <= z <= b:
                return da + (db-da)*(z-a)/(b-a)
        raise ValueError("C3 pole falls outside the material tube profile")
    for key in ("condenser_lens_3_upper_pole", "condenser_lens_3_lower_pole"):
        if key in by_key:
            pole = by_key[key]
            a, b = float(pole["local_start_z_mm"]), float(pole["local_end_z_mm"])
            samples = [diameter_at(a), diameter_at(b)] + [d for z, d in points if a < z < b]
            if max(samples) >= float(pole["mechanical_bore_diameter_mm"]):
                raise ValueError("Material tube must retain positive radial clearance inside C3 poles")
    if geometry[STATUS] != "engineering_reconstruction_not_oem" or not str(geometry[SOURCE]).strip():
        raise ValueError("Material tube details require explicit engineering, non-OEM provenance")


def decorate_physical_tube_row(row, assembly, *, runtime_values=None, aperture_indices=None):
    """Add self-contained global-mm material geometry to the one declared tube.

    No assembly rows are mutated. Aperture entry boxes follow the same physical
    pose as their strip, including rotation about its module-local reference.
    Runtime opening values are optional and affect the same strip outline rule
    used by the aperture itself; they never create additional slot identities.
    """
    if row.get("key") != CONTINUOUS_TUBE_KEY:
        return row
    modules = [module for module in getattr(assembly, "modules", ()) if PROFILE in module.geometry]
    if not modules:
        return row
    if len(modules) != 1:
        raise ValueError("Material continuous tube must have one authoritative module")
    module = modules[0]
    geometry = module.geometry
    active = {part.key: part for part in assembly.parts if part.module_key == module.key}
    by_key = {part.key: part.data for part in module.parts}
    by_key.update({key: part.data for key, part in active.items()})
    validate_material_tube_geometry(by_key.values(), geometry)
    origin = float(row["local_start_z_mm"]) - float(geometry["c1_c2_objective_vacuum_tube_start_z_mm"])
    if abs(float(row["local_end_z_mm"])-origin-float(geometry["c1_c2_objective_vacuum_tube_end_z_mm"])) > 1e-8:
        raise ValueError("Material tube row does not match its declared module span")
    from temsim.cad_auxiliary import aperture_solids
    from temsim.lens_pose import mechanical_part_transform_mm
    result = dict(row)
    result[PROFILE] = [[float(z)+origin, float(d)] for z, d in geometry[PROFILE]]
    result[STATUS], result[SOURCE] = geometry[STATUS], geometry[SOURCE]
    result[CLEARANCE] = clearance = float(geometry[CLEARANCE])
    result[SLOT_KEYS] = list(geometry[SLOT_KEYS])
    slots = []
    for key in geometry[SLOT_KEYS]:
        if key not in active:
            raise ValueError(f"Declared tube aperture slot has no installed aperture: {key}")
        resolved = active[key]
        part = resolved.data
        if abs(float(resolved.center_z_mm)-float(part["local_center_z_mm"])-origin) > 1e-8:
            raise ValueError("Tube and aperture must share the same module placement")
        # The side-entry slot follows only the plate, not its external rod or
        # fixing screw. Aperture material may now contain several regions.
        spec, = [spec for spec in aperture_solids(
            part, (runtime_values or {}).get(key), (aperture_indices or {}).get(key, 0))
            if spec.region == "body"]
        from temsim.cad_aperture_support import aperture_material_scale_xy
        scale_x, scale_y = aperture_material_scale_xy(part)
        # Native material applies CAD transverse scaling before its shared
        # rigid placement. Clearance is a physical gap in mm, not a scale.
        width = float(spec.parameters["strip_half_width_mm"])*scale_x + clearance
        lower = float(spec.parameters["strip_y_bounds_mm"][0])*scale_y - clearance
        upper = float(row["mechanical_outer_diameter_mm"])/2 + clearance
        thickness = float(spec.parameters["plate_thickness_mm"]) + 2*clearance
        z = float(part.get("optical_reference_local_z_mm", part["local_center_z_mm"]))
        # The tube opening follows the inserted material, not an optical-axis
        # tilt that leaves its mechanical envelope in place.
        rotation, translation = mechanical_part_transform_mm(part, by_key)
        slot = box((0., (lower+upper)/2, z), (2*width, upper-lower, thickness))
        slot = transformed(slot, rotation=rotation.tolist(), offset=translation.tolist())
        slot = transformed(slot, offset=(0., 0., origin))
        slots.append(dict(aperture_key=key, clearance_mm=clearance,
            strip_parameters=dict(spec.parameters), local_bounds_mm=[[-width, lower, z-thickness/2], [width, upper, z+thickness/2]],
            rotation_local_to_global=rotation.tolist(), translation_local_mm=translation.tolist(),
            module_origin_z_mm=origin, shape=slot))
    result[SLOTS] = slots
    pole = active.get("objective_upper_pole")
    if pole is not None and "terminal_tube" in pole.data.get("material_regions", {}):
        from temsim.part_materials import material_for_region
        result["terminal_tube_length_mm"] = pole.data["terminal_tube_length_mm"]
        result["material_regions"] = {"terminal_tube": material_for_region(pole, "terminal_tube")}
    return result


def tube_solids(row):
    """Return analytic material leaves for a decorated globally placed tube."""
    if row.get("key") != CONTINUOUS_TUBE_KEY or PROFILE not in row:
        return None
    points = [(float(z), float(d)/2) for z, d in row[PROFILE]]
    inner = float(row["mechanical_inner_diameter_mm"])/2
    terminal = ()
    if "terminal_tube_length_mm" in row:
        cut = points[-1][0] - row["terminal_tube_length_mm"]
        a, b = next((a, b) for a, b in zip(points, points[1:]) if a[0] <= cut <= b[0])
        radius = a[1] + (b[1]-a[1])*(cut-a[0])/(b[0]-a[0])
        tail = [(cut, radius), *(p for p in points if p[0] > cut)]
        points = [*(p for p in points if p[0] < cut), (cut, radius)]
        material = row["material_regions"]["terminal_tube"]
        terminal = (SolidSpec(row["key"], "terminal_tube", material["material_key"],
                              revolve([*tail, (tail[-1][0], inner), (cut, inner)]),
                              status=row[STATUS], note="Assigned terminal wall of the continuous vacuum tube.",
                              parameters={"material_assignment": material}),)
        if len(points) == 1:
            return terminal
    shape = revolve([*points, (points[-1][0], inner), (points[0][0], inner)])
    slots = row.get(SLOTS, ())
    if slots:
        shape = difference(shape, *(slot["shape"] for slot in slots))
    return (SolidSpec(row["key"], "body", "nonmagnetic_housing", shape,
        status=row[STATUS], note=str(row[SOURCE]) + " The side-entry slots describe the material around the beam passage. The passive tube material is a model assumption. Its alloy and magnetic contact at the upper objective require independent evidence.",
        parameters={PROFILE: row[PROFILE], CLEARANCE: row[CLEARANCE],
            SLOT_KEYS: row[SLOT_KEYS], "aperture_slots": [{key: value for key, value in slot.items() if key != "shape"} for slot in slots],
            "inner_diameter_mm": 2*inner}),) + terminal
