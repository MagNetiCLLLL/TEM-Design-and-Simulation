"""Explicit engineering adaptation of an eight-coil gun stigmator carrier.

Solids use module-local millimetres and the existing open gun envelope. No
coil-resolved field, turn count, magnetic core, material grade or OEM fit is
inferred. Every internal dimension is supplied by the opt-in TOML declaration.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

from temsim.cad_geometry import SolidSpec, annulus, box, difference, transformed, union
from temsim.component_representation import non_material_role


PREFIX = "gun_stigmator_"
PROFILE = "eight_coil_slotted_carrier"
HOST_KEYS = frozenset({"feg_stigmator", "thermionic_stigmator"})
COIL_SOURCE = "https://www.jeol.co.jp/words/semterms/20201020.111014.html"
SUPPORT_SOURCE = "https://patents.google.com/patent/CN102509689B/en"
FIELD_TOPOLOGY_SOURCE = "https://patents.google.com/patent/US6548816B2/en"
DIMENSIONS = (
    "coil_inner_radius_mm", "coil_outer_radius_mm", "coil_width_mm", "coil_length_mm",
    "window_width_mm", "window_length_mm", "insulation_thickness_mm",
    "former_wall_thickness_mm", "adhesive_thickness_mm", "basis_angle_deg",
)
FIELDS = frozenset(PREFIX+name for name in (
    "geometry_profile", "geometry_status", "coil_source", "support_source", "source_note",
    "coil_count", *DIMENSIONS,
))
NOTE = (
    "Engineering gun adaptation, not OEM CAD: eight bulk windings in separate insulating "
    "formers, an equivalent bulk insulation envelope and adhesive seats in a slotted carrier. "
    "The support principle comes from an objective-stigmator example, not a measured gun. "
    "All internal dimensions are saved model assumptions. No magnetic core, shield, individual "
    "turns, leads, external fixing, material grade or insulation performance is established. "
    "The existing finite quadrupole field remains authoritative; these solids add no field source."
)


def _number(part, field):
    value = part.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{part.get('key', 'gun stigmator')}.{field} must be finite")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{part.get('key', 'gun stigmator')}.{field} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'gun stigmator')}.{field} must be finite")
    return value


def gun_stigmator_dimensions(part):
    """Validate an active recipe; legacy or replaced geometry has no native detail."""
    supplied = {key for key in part if str(key).startswith(PREFIX)}
    if not supplied:
        return None
    # An explicit CAD base owns the material geometry. Keep the inactive recipe
    # saved so that restoring 'existing' revalidates it before the draft commits.
    if part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    key = str(part.get("key", "gun stigmator"))
    if supplied-FIELDS:
        raise ValueError(f"{key}: unknown gun stigmator fields: {sorted(supplied-FIELDS)}")
    if part.get("geometry_template_key", key) not in HOST_KEYS or non_material_role(part):
        raise ValueError(f"{key}: gun stigmator geometry requires a material gun stigmator host")
    if part.get(PREFIX+"geometry_profile") != PROFILE:
        raise ValueError(f"{key}: gun stigmator requires profile {PROFILE}")
    if part.get(PREFIX+"geometry_status") != "engineering_reconstruction_not_oem":
        raise ValueError(f"{key}: gun stigmator requires explicit engineering reconstruction status")
    if (part.get(PREFIX+"coil_source") != COIL_SOURCE
            or part.get(PREFIX+"support_source") != SUPPORT_SOURCE):
        raise ValueError(f"{key}: gun stigmator construction sources must be declared")
    note = part.get(PREFIX+"source_note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{key}: gun stigmator source note is required")
    count = part.get(PREFIX+"coil_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count != 8:
        raise ValueError(f"{key}: the declared stigmator topology requires exactly eight coils")
    d = {name: _number(part, PREFIX+name) for name in DIMENSIONS}
    if any(value <= 0 for name, value in d.items() if name != "basis_angle_deg"):
        raise ValueError(f"{key}: gun stigmator dimensions must be positive")
    from temsim.cad_auxiliary import _radii
    from temsim.part_model_3d import _extent
    start, end = _extent(part)
    bore, outer = _radii(part)
    centre = _number(part, "local_center_z_mm")
    if bore is None or outer is None or not 0 < bore < outer or not start < centre < end:
        raise ValueError(f"{key}: gun stigmator requires a finite open envelope")
    ri, ro = d["coil_inner_radius_mm"], d["coil_outer_radius_mm"]
    width, length = d["coil_width_mm"], d["coil_length_mm"]
    wi, li = d["window_width_mm"], d["window_length_mm"]
    insulation, wall, adhesive = (d[name] for name in (
        "insulation_thickness_mm", "former_wall_thickness_mm", "adhesive_thickness_mm"))
    # The former tube occupies the insulated winding window; flanges extend
    # radially beyond it. Its air window remains open, independently of the beam bore.
    if not (ri < ro and 2*(insulation+wall) < wi < width
            and 2*(insulation+wall) < li < length):
        raise ValueError(f"{key}: winding and former windows must remain open")
    pocket_inner, pocket_outer = ri-insulation-wall, ro+insulation+wall
    pocket_half_width = width/2+insulation
    coil_start, coil_end = centre-length/2, centre+length/2
    pocket_start = coil_start-insulation-adhesive
    if not (bore < pocket_inner and math.hypot(pocket_outer, pocket_half_width) < outer
            and start < pocket_start and coil_end+insulation < end):
        raise ValueError(f"{key}: gun stigmator details must fit the unchanged bore and body envelope")
    # Each cartridge fits strictly inside its 45-degree azimuth allocation.
    if math.atan2(pocket_half_width, pocket_inner) >= math.pi/8:
        raise ValueError(f"{key}: adjacent gun stigmator cartridges would overlap")
    d.update(start_mm=start, end_mm=end, centre_mm=centre, bore_radius_mm=bore,
             outer_radius_mm=outer, pocket_inner_mm=pocket_inner, pocket_outer_mm=pocket_outer,
             pocket_half_width_mm=pocket_half_width, pocket_start_mm=pocket_start,
             coil_start_mm=coil_start, coil_end_mm=coil_end, coil_count=count)
    return d


def _rotation(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]


def gun_stigmator_solids(part, by_key=None):
    """One shared CAD/native construction; no declaration keeps legacy geometry."""
    d = gun_stigmator_dimensions(part)
    if d is None:
        return None
    ri, ro, z = d["coil_inner_radius_mm"], d["coil_outer_radius_mm"], d["centre_mm"]
    width, length = d["coil_width_mm"], d["coil_length_mm"]
    wi, li = d["window_width_mm"], d["window_length_mm"]
    t, wall = d["insulation_thickness_mm"], d["former_wall_thickness_mm"]
    x = (ri+ro)/2
    radial = ro-ri
    # All cut tools extend beyond their target using the already configured
    # radial allocation; these extensions are not additional material dimensions.
    copper = difference(box((x, 0., z), (radial, width, length)),
                        box((x, 0., z), (2*radial, wi, li)))
    covered = difference(box((x, 0., z), (radial+2*t, width+2*t, length+2*t)),
                         box((x, 0., z), (2*(radial+2*t), wi-2*t, li-2*t)))
    coating = difference(covered, copper)
    former = union(
        box((x, 0., z), (radial+2*t, wi-2*t, li-2*t)),
        box((ri-t-wall/2, 0., z), (wall, width+2*t, length+2*t)),
        box((ro+t+wall/2, 0., z), (wall, width+2*t, length+2*t)),
    )
    former = difference(former, box((x, 0., z),
        (2*(radial+2*t+2*wall), wi-2*(t+wall), li-2*(t+wall))))
    pocket_width = 2*d["pocket_half_width_mm"]
    pocket_radial = d["pocket_outer_mm"]-d["pocket_inner_mm"]
    pocket_lo, pocket_hi = d["pocket_start_mm"], d["end_mm"]+length
    pocket = box((x, 0., (pocket_lo+pocket_hi)/2),
                 (pocket_radial, pocket_width, pocket_hi-pocket_lo))
    adhesive = box((x, 0., pocket_lo+d["adhesive_thickness_mm"]/2),
                   (pocket_radial, pocket_width, d["adhesive_thickness_mm"]))
    source_paths = [name for name in sorted(FIELDS) if name in part]
    common = {**d, "dimension_status": "saved_engineering_assumptions_not_oem",
              "parameter_fields": source_paths, "field_source_added": False,
              "effective_field_length_mm": part.get("active_length_mm"),
              "support_topology_status": "objective_stigmator_principle_adapted_for_gun"}
    sources = (COIL_SOURCE, FIELD_TOPOLOGY_SOURCE, SUPPORT_SOURCE)
    def spec(region, material, shape, extra=None):
        return SolidSpec(str(part["key"]), region, material, shape, sources,
                         "engineering_reconstruction", NOTE, {**common, **(extra or {})})
    parts, slots = [], []
    for index in range(8):
        angle = d["basis_angle_deg"]+45*index
        rotation = _rotation(angle)
        slots.append(transformed(pocket, rotation=rotation))
        metadata = {"coil_index": index+1, "coil_azimuth_deg": angle,
                    "quadrupole_basis": "normal" if index % 2 == 0 else "skew",
                    "current_or_turn_calibration": "not_defined"}
        for name, material, shape in (
            ("coil", "copper_winding", copper),
            ("coil_insulation", "insulator", coating),
            ("coil_former", "insulating_polymer_unspecified", former),
            ("adhesive_seat", "bonding_material_unspecified", adhesive),
        ):
            parts.append(spec(f"{name}_{index+1}", material,
                              transformed(shape, rotation=rotation), metadata))
    carrier = difference(annulus(d["start_mm"], d["end_mm"],
                                 d["bore_radius_mm"], d["outer_radius_mm"]), *slots)
    return (spec("slotted_carrier", "insulating_polymer_unspecified", carrier), *parts)
