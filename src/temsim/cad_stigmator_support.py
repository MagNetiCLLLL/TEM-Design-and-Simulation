"""Opt-in supports around the unchanged TEM eight-coil stigmator winding.

Module-local mm. The construction principle is adapted from an objective
stigmator, not a measured condenser package. Copper and field calibration stay
owned by the existing winding and runtime definitions respectively.
"""
from __future__ import annotations

import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, annulus, box, difference, transformed, union
from temsim.component_representation import non_material_role


PREFIX = "stigmator_support_"
PROFILE = "slotted_carrier_insulated_formers"
SOURCE = "https://patents.google.com/patent/CN102509689B/en"
DIMENSIONS = tuple(PREFIX+name for name in (
    "insulation_thickness_mm", "former_wall_thickness_mm", "adhesive_thickness_mm"))
FIELDS = frozenset((*DIMENSIONS, *(PREFIX+name for name in (
    "profile", "geometry_status", "source", "source_note"))))
NOTE = (
    "Condenser engineering adaptation of the slotted cylindrical support, coil formers, "
    "insulated wire and adhesive construction in CN102509689B Figs. 5, 6 and 12. "
    "Dimensions and unspecified insulating/bonding materials are saved model assumptions, "
    "not OEM geometry. Bulk insulation does not resolve individual turns. No magnetic "
    "shield, iron core, wiring, flange threads or host fastening is inferred. Existing "
    "copper windings, effective quadrupole length and two field channels are unchanged."
)


def _positive(part, key):
    value = part.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{part.get('key')}.{key} must be finite and positive")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{part.get('key')}.{key} must be finite and positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{part.get('key')}.{key} must be finite and positive")
    return number


def stigmator_support_dimensions(part):
    """Validate active supports; absent or replaced geometry has no native detail."""
    supplied = {str(key) for key in part if str(key).startswith(PREFIX)}
    if not supplied:
        return None
    # An explicit CAD base owns the material geometry. Keep the inactive recipe
    # saved so that restoring 'existing' revalidates it before the draft commits.
    if part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    key = str(part.get("key", "stigmator"))
    if supplied-FIELDS:
        raise ValueError(f"{key}: unknown stigmator support fields: {sorted(supplied-FIELDS)}")
    if (non_material_role(part) or part.get("branch_path_only")
            or part.get("geometry_template_key", key) != "condenser_stigmator"
            or part.get("coil_topology") != "two_interleaved_four_pole_windings"):
        raise ValueError(f"{key}: support requires the material eight-coil condenser stigmator")
    if part.get(PREFIX+"profile") != PROFILE:
        raise ValueError(f"{key}: stigmator support requires profile {PROFILE}")
    if part.get(PREFIX+"geometry_status") != "engineering_reconstruction_not_oem":
        raise ValueError(f"{key}: stigmator support requires explicit non-OEM status")
    if part.get(PREFIX+"source") != SOURCE or not isinstance(part.get(PREFIX+"source_note"), str) or not part[PREFIX+"source_note"].strip():
        raise ValueError(f"{key}: stigmator support requires its construction source and note")
    t, wall, adhesive = (_positive(part, field) for field in DIMENSIONS)
    # Reuse the exact winding allocation, including edited channel basis angles.
    # This private copper-only builder does not call this opt-in support module.
    from temsim.cad_multipoles import _stigmator_windings
    windings = _stigmator_windings(part)
    source = windings[0].parameters
    p = windings[0].shape["shape"]
    centre = p["base"]["center_mm"]
    radial, width, length = p["base"]["size_mm"]
    _, window_width, window_length = p["tools"][0]["size_mm"]
    ri, ro = centre[0]-radial/2, centre[0]+radial/2
    inner, outer = source["inner_radius_mm"], source["outer_radius_mm"]
    start, end, z = source["start_mm"], source["end_mm"], centre[2]
    pocket_inner, pocket_outer = ri-t-wall, ro+t+wall
    pocket_half_width = width/2+t
    pocket_start = z-length/2-t-adhesive
    if not (2*(t+wall) < window_width and 2*(t+wall) < window_length):
        raise ValueError(f"{key}: stigmator winding/former windows must remain open")
    if not (inner < pocket_inner and math.hypot(pocket_outer, pocket_half_width) < outer
            and start < pocket_start and z+length/2+t < end):
        raise ValueError(f"{key}: stigmator support must fit the unchanged bore and body")
    if 2*math.degrees(math.atan2(pocket_half_width, pocket_inner)) >= source["minimum_coil_separation_deg"]:
        raise ValueError(f"{key}: adjacent stigmator supports would overlap")
    return dict(start_mm=start, end_mm=end, centre_mm=z,
        inner_radius_mm=inner, outer_radius_mm=outer,
        coil_inner_radius_mm=ri, coil_outer_radius_mm=ro,
        coil_width_mm=width, coil_length_mm=length,
        window_width_mm=window_width, window_length_mm=window_length,
        insulation_thickness_mm=t, former_wall_thickness_mm=wall, adhesive_thickness_mm=adhesive,
        pocket_inner_mm=pocket_inner, pocket_outer_mm=pocket_outer,
        pocket_half_width_mm=pocket_half_width, pocket_start_mm=pocket_start,
        coil_angles_deg=[spec.parameters["coil_azimuth_deg"] for spec in windings],
        effective_field_length_mm=part.get("effective_length_mm"))


def stigmator_support_solids(part, by_key=None):
    """Return the same eight copper leaves plus 25 passive support regions."""
    d = stigmator_support_dimensions(part)
    if d is None:
        return None
    from temsim.cad_multipoles import _stigmator_windings
    windings = _stigmator_windings(part)
    t, wall, adhesive = (d[name] for name in (
        "insulation_thickness_mm", "former_wall_thickness_mm", "adhesive_thickness_mm"))
    ri, ro, z = d["coil_inner_radius_mm"], d["coil_outer_radius_mm"], d["centre_mm"]
    x, radial = (ri+ro)/2, ro-ri
    width, length = d["coil_width_mm"], d["coil_length_mm"]
    wi, li = d["window_width_mm"], d["window_length_mm"]
    copper = windings[0].shape["shape"]
    coated = difference(box((x, 0., z), (radial+2*t, width+2*t, length+2*t)),
        box((x, 0., z), (2*(radial+2*t), wi-2*t, li-2*t)))
    insulation = difference(coated, copper)
    former = union(box((x, 0., z), (radial+2*t, wi-2*t, li-2*t)),
        box((ri-t-wall/2, 0., z), (wall, width+2*t, length+2*t)),
        box((ro+t+wall/2, 0., z), (wall, width+2*t, length+2*t)))
    former = difference(former, box((x, 0., z),
        (2*(radial+2*t+2*wall), wi-2*(t+wall), li-2*(t+wall))))
    pocket_lo, pocket_hi = d["pocket_start_mm"], d["end_mm"]+length
    pocket_radial = d["pocket_outer_mm"]-d["pocket_inner_mm"]
    pocket_width = 2*d["pocket_half_width_mm"]
    pocket = box((x, 0., (pocket_lo+pocket_hi)/2),
                 (pocket_radial, pocket_width, pocket_hi-pocket_lo))
    seat = box((x, 0., pocket_lo+adhesive/2), (pocket_radial, pocket_width, adhesive))
    parameters = {**d, "geometry_status": "engineering_reconstruction_not_oem",
        "optical_field_unchanged": True, "parameter_fields": sorted(FIELDS)}
    def spec(region, material, shape, extra=None):
        return SolidSpec(str(part["key"]), region, material, shape, (SOURCE,),
            "engineering_reconstruction", NOTE, {**parameters, **(extra or {})})
    supports, slots = [], []
    for index, winding in enumerate(windings, 1):
        rotation = winding.shape["rotation"]
        slots.append(transformed(pocket, rotation=rotation))
        extra = {name: winding.parameters[name] for name in ("coil_group", "basis_angle_deg", "coil_azimuth_deg")}
        for name, material, shape in (("insulation", "insulator", insulation),
            ("former", "insulating_polymer_unspecified", former),
            ("adhesive_seat", "bonding_material_unspecified", seat)):
            supports.append(spec(f"coil_{index}_{name}", material, transformed(shape, rotation=rotation), extra))
    carrier = difference(annulus(d["start_mm"], d["end_mm"],
        d["inner_radius_mm"], d["outer_radius_mm"]), *slots)
    return windings + (spec("slotted_carrier", "insulating_polymer_unspecified", carrier), *supports)
