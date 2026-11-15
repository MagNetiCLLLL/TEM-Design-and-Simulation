"""Optional passive formers and outer jackets for the HP1/HP2 six-pin family.

Coordinates are module-local mm. The generic multipole winding-member and
bonding principle is documented; rectangular flanges, dimensions and material
classes here are explicit engineering adaptations, not CEOS manufacturing data.
"""
from __future__ import annotations

import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, box, difference, transformed, union
from temsim.probe_hexapole_geometry import is_probe_hexapole, probe_hexapole_dimensions

PREFIX = "hexapole_support_"
PROFILE = "bonded_radial_coil_formers"
SOURCE = "https://patents.google.com/patent/JP2004241190A/en"
BASE_DIMENSIONS = tuple(PREFIX+name for name in (
    "former_wall_thickness_mm", "flange_thickness_mm"))
OUTER_INSULATION_FIELD = PREFIX+"outer_insulation_thickness_mm"
DIMENSIONS = (*BASE_DIMENSIONS, OUTER_INSULATION_FIELD)
BASE_FIELDS = frozenset((*BASE_DIMENSIONS, *(PREFIX+name for name in (
    "profile", "geometry_status", "source", "source_note"))))
FIELDS = BASE_FIELDS | {OUTER_INSULATION_FIELD}
NOTE = (
    "Engineering adaptation of the hollow winding member and core bonding in "
    "JP2004241190A paragraph 0005, Fig.7(c,d); paragraph 0055 also permits bobbins. "
    "This is generic multipole construction evidence, not verified CEOS HP1/HP2 hardware. "
    "Rectangular former flanges, unspecified insulating material and bulk core bedding "
    "are saved reconstruction assumptions. No individual turns, screws, cooling or "
    "mounting to the column is inferred. Original copper, magnetic poles, yoke and "
    "all field inputs are unchanged. These passive solids do not add a field solve."
)
OUTER_INSULATION_NOTE = (
    "Optional external winding insulation jacket, explicitly engineered rather than OEM geometry. "
    "Six rectangular sleeves enclose the original radial-axis bulk copper envelope without "
    "changing its window or extending its radial ends. Original poles, yoke, winding formers, "
    "core bedding, vacuum passage and all optical/field inputs remain unchanged. Material grade, "
    "dielectric rating, turns, leads and electrical current are not inferred. This CAD subdivision "
    "does not add a magnetic-material solve or an optical excitation."
)


def _positive(part, field):
    value = part.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{part.get('key')}.{field} must be finite and positive")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{part.get('key')}.{field} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{part.get('key')}.{field} must be finite and positive")
    return value


def hexapole_support_dimensions(part):
    """Validate an active saved recipe; absent or replaced geometry stays legacy."""
    supplied = {str(key) for key in part if str(key).startswith(PREFIX)}
    if not supplied:
        return None
    # A replacement CAD base owns its materials. Inactive reconstruction fields
    # remain saved so that restoring 'existing' can validate them atomically.
    if part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    key = str(part.get("key", "hexapole"))
    if supplied-FIELDS:
        raise ValueError(f"{key}: unknown hexapole support fields: {sorted(supplied-FIELDS)}")
    if (not is_probe_hexapole(part)
            or part.get("geometry_template_key", key) not in {"probe_hp1_hexapole", "probe_hp2_hexapole"}):
        raise ValueError(f"{key}: winding supports require a material HP1/HP2 hexapole")
    if part.get(PREFIX+"profile") != PROFILE:
        raise ValueError(f"{key}: hexapole support requires profile {PROFILE}")
    if part.get(PREFIX+"geometry_status") != "engineering_reconstruction_not_oem":
        raise ValueError(f"{key}: hexapole support requires explicit non-OEM status")
    if (part.get(PREFIX+"source") != SOURCE
            or not isinstance(part.get(PREFIX+"source_note"), str)
            or not part[PREFIX+"source_note"].strip()):
        raise ValueError(f"{key}: hexapole support requires its construction source and note")
    d = probe_hexapole_dimensions(part)
    wall, flange = (_positive(part, field) for field in BASE_DIMENSIONS)
    inner_width = d['coil_window_width_mm']-2*wall
    inner_height = d['coil_window_height_mm']-2*wall
    if not (inner_width > d['root_width_mm'] and inner_height > d['stem_height_mm']):
        raise ValueError(f"{key}: hexapole former must leave positive core bedding thickness")
    ri = d['inner_radius_mm']+d['coil_radial_inset_mm']
    ro = ri+d['coil_radial_length_mm']
    root = d['inner_radius_mm']+d['taper_length_mm']
    yoke_inner = d['outer_radius_mm']-d['yoke_thickness_mm']
    if not (root < ri-flange and math.hypot(ro+flange, d['coil_width_mm']/2) < yoke_inner):
        raise ValueError(f"{key}: hexapole former flanges must clear the pole taper and return yoke")
    if d['coil_width_mm']/2 >= (ri-flange)*math.tan(math.pi/6):
        raise ValueError(f"{key}: neighbouring hexapole formers would overlap")
    values = dict(d, former_wall_thickness_mm=wall, flange_thickness_mm=flange,
        coil_start_radius_mm=ri, coil_end_radius_mm=ro,
        former_inner_width_mm=inner_width, former_inner_height_mm=inner_height,
        core_bedding_tangential_thickness_mm=(inner_width-d['root_width_mm'])/2,
        core_bedding_axial_thickness_mm=(inner_height-d['stem_height_mm'])/2)
    if OUTER_INSULATION_FIELD in part:
        thickness = _positive(part, OUTER_INSULATION_FIELD)
        outer_width = d['coil_width_mm']+2*thickness
        outer_height = d['coil_height_mm']+2*thickness
        centre = (d['start_mm']+d['end_mm'])/2
        copper_lower, copper_upper = centre-d['coil_height_mm']/2, centre+d['coil_height_mm']/2
        outer_lower, outer_upper = centre-outer_height/2, centre+outer_height/2
        if (not all(math.isfinite(value) for value in (outer_width, outer_height, outer_lower, outer_upper))
                or not outer_width > d['coil_width_mm'] or not outer_height > d['coil_height_mm']
                or not outer_lower < copper_lower < copper_upper < outer_upper):
            raise ValueError(f"{key}: outer winding insulation needs a positive representable wall thickness")
        if not d['start_mm'] <= outer_lower < outer_upper <= d['end_mm']:
            raise ValueError(f"{key}: outer winding insulation must fit the original axial envelope")
        if math.hypot(ro, outer_width/2) >= yoke_inner:
            raise ValueError(f"{key}: outer winding insulation intersects the return yoke")
        if outer_width/2 >= ri*math.tan(math.pi/6):
            raise ValueError(f"{key}: neighbouring outer winding insulation jackets would overlap")
        values.update(outer_insulation_thickness_mm=thickness,
                      outer_insulation_width_mm=outer_width,
                      outer_insulation_height_mm=outer_height,
                      outer_insulation_start_mm=outer_lower,
                      outer_insulation_end_mm=outer_upper)
    return values


def validate_hexapole_support(part):
    hexapole_support_dimensions(part)


def hexapole_support_solids(part):
    """Return old formers/bedding plus optional external insulation sleeves."""
    d = hexapole_support_dimensions(part)
    if d is None:
        return ()
    ri, ro, flange = (d[name] for name in (
        'coil_start_radius_mm', 'coil_end_radius_mm', 'flange_thickness_mm'))
    x, z = (ri+ro)/2, (d['start_mm']+d['end_mm'])/2
    radial = ro-ri+2*flange
    # The former tube fills part of the existing copper window; its two radial
    # end flanges contact copper end faces without changing that winding pack.
    former = union(box((x, 0., z),
        (radial, d['coil_window_width_mm'], d['coil_window_height_mm'])),
        box((ri-flange/2, 0., z), (flange, d['coil_width_mm'], d['coil_height_mm'])),
        box((ro+flange/2, 0., z), (flange, d['coil_width_mm'], d['coil_height_mm'])))
    former = difference(former, box((x, 0., z),
        (2*radial, d['former_inner_width_mm'], d['former_inner_height_mm'])))
    bedding = difference(box((x, 0., z),
        (radial, d['former_inner_width_mm'], d['former_inner_height_mm'])),
        box((x, 0., z), (2*radial, d['root_width_mm'], d['stem_height_mm'])))
    # Preserve the pre-jacket support records exactly, including their existing
    # parameter-field list. New metadata belongs only to the added sleeves.
    old_dimensions = {key: value for key, value in d.items() if not key.startswith("outer_insulation_")}
    params = dict(old_dimensions, geometry_status="engineering_reconstruction_not_oem",
        optical_field_unchanged=True, passive_geometry_only=True,
        coil_axis="radial", pole_pitch_deg=60.,
        contact_definition="Copper-window walls and radial end faces contact the former; "
            "bulk bedding joins the former inner wall to the existing rectangular core.",
        parameter_fields=sorted(BASE_FIELDS))
    result = []
    for index in range(6):
        angle = math.radians(60*index)
        c, s = math.cos(angle), math.sin(angle)
        rotation = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        for family, material, shape in (
            ("former", "insulating_polymer_unspecified", former),
            ("core_bedding", "bonding_material_unspecified", bedding)):
            result.append(SolidSpec(str(part['key']), f"coil_{index+1}_{family}",
                material, transformed(shape, rotation=rotation),
                (SOURCE, "docs/references/cad-probe-corrector.md"),
                "engineering_reconstruction", NOTE, dict(params, coil_azimuth_deg=60.*index)))
    if "outer_insulation_thickness_mm" in d:
        # The sleeves grow tangentially and axially only. Their unchanged
        # radial end faces avoid the existing former-flange allocations.
        jacket = difference(box((x, 0., z),
            (ro-ri, d['outer_insulation_width_mm'], d['outer_insulation_height_mm'])),
            box((x, 0., z), (2*(ro-ri), d['coil_width_mm'], d['coil_height_mm'])))
        receipt = dict(d, geometry_status="engineering_reconstruction_not_oem",
            passive_geometry_only=True, optical_field_unchanged=True,
            original_copper_envelope_unchanged=True, coil_axis="radial",
            radial_extent_policy="Original copper start/end radii retained; no radial-end caps",
            parameter_fields=[OUTER_INSULATION_FIELD])
        for index in range(6):
            angle = math.radians(60*index); c, s = math.cos(angle), math.sin(angle)
            rotation = [[c,-s,0.],[s,c,0.],[0.,0.,1.]]
            result.append(SolidSpec(str(part['key']), f"coil_{index+1}_outer_insulation",
                "insulator", transformed(jacket,rotation=rotation),
                ("docs/references/cad-probe-corrector.md",), "engineering_reconstruction",
                OUTER_INSULATION_NOTE,
                dict(receipt,coil_azimuth_deg=60.*index,
                     winding_geometry={"kind":"probe_radial_outer_jacket", "lead_geometry_included":False})))
    return tuple(result)
