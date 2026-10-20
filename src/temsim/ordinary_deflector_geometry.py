"""Opt-in rigid insulating supports for ordinary two-stage magnetic deflectors.

The two-stage optical layout and the existing bulk copper loops are retained.
The added geometry follows a published construction family, not an identified
OEM part: concentric insulating coil carriers, an inter-axis spacer, radial
registration pins and an insulating outer casing. All dimensions are local mm.
"""
from __future__ import annotations

import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, annulus, difference, sector, transformed
from temsim.component_representation import non_material_role


PREFIX = "deflector_support_"
SUPPORT_PROFILE = "rigid_insulating_carriers_pinned"
SUPPORT_SOURCE = "https://patents.google.com/patent/US5994704A/en"
HOST_KEYS = frozenset({"condenser_deflector", "beam_deflector", "ac_deflector",
                       "image_diffraction_deflector"})
DIMENSIONS = (PREFIX + "radial_skin_thickness_mm", PREFIX + "casing_thickness_mm",
              PREFIX + "pin_diameter_mm")
SLOT_SWEEP_FIELD = PREFIX + "registration_slot_sweep_deg"
METADATA_FIELDS = (PREFIX + "profile", PREFIX + "geometry_status", PREFIX + "source",
                   PREFIX + "source_note")
FIELDS = frozenset((*DIMENSIONS, SLOT_SWEEP_FIELD, *METADATA_FIELDS))
NOTE = (
    "Rigid insulating-carrier engineering equivalent based on US5994704A "
    "Figs. 1-4, not verified OEM hardware. Existing saddle-loop copper volumes "
    "are embedded bulk winding allocations, not the patent's precise deposited "
    "conductor pattern or closed, shorted single turns. Carrier cylinders, "
    "inter-axis spacer, registration pins and insulating casing have assumed "
    "dimensions and unspecified material grades. Pins register the internal "
    "carriers; external host mounting, wiring, individual-turn insulation and "
    "calibrated magnetic, electric, thermal or vacuum properties are not "
    "established. These passive solids do not add optical field sources."
)
SLOT_NOTE = (
    "The optional round-ended circumferential registration slots follow "
    "US5994704A Figs. 1-2 and the description of elongated outer-carrier and "
    "spacer knock holes. The inner carrier retains round holes. This model's "
    "saved radial order calls the outer carrier Y, whereas the patent calls "
    "its outer carrier X. The total centreline sweep is an engineering "
    "dimension, not an OEM measurement or a command to rotate either coil. "
    "Copper, registration-pin axes and the calibrated field bases do not move."
)


def _number(part, field, default=None):
    value = part.get(field, default)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'deflector')}.{field} must be a finite number")
    return float(value)


def support_dimensions(part):
    """Validate the explicit opt-in; legacy ordinary coils remain unchanged."""
    supplied = {name for name in part if str(name).startswith(PREFIX)}
    if not supplied:
        return None
    key = str(part.get("key", "deflector"))
    unknown = supplied - FIELDS
    if unknown:
        raise ValueError(f"{key}: unknown ordinary deflector support fields: " + ", ".join(sorted(unknown)))
    if part.get(PREFIX + "profile") != SUPPORT_PROFILE:
        raise ValueError(f"{key}: ordinary deflector support requires profile {SUPPORT_PROFILE}")
    profile = part.get("mechanical_profile", "")
    template = part.get("geometry_template_key", key)
    if (non_material_role(part) or part.get("branch_path_only") or not isinstance(profile, str)
            or not (profile == "magnetic_deflector" or (profile == "" and template in HOST_KEYS))):
        raise ValueError(f"{key}: ordinary deflector support requires a material magnetic deflector host")
    if part.get(PREFIX + "geometry_status") != "engineering_reconstruction_not_oem":
        raise ValueError(f"{key}: ordinary deflector support requires explicit non-OEM engineering status")
    if part.get(PREFIX + "source") != SUPPORT_SOURCE:
        raise ValueError(f"{key}: ordinary deflector support requires the declared construction source")
    source_note = part.get(PREFIX + "source_note")
    if not isinstance(source_note, str) or not source_note.strip():
        raise ValueError(f"{key}: ordinary deflector support requires a nonempty source note")

    start, end = (_number(part, name) for name in ("local_start_z_mm", "local_end_z_mm"))
    inner_field = next((name for name in (
        "mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm",
        "mechanical_clear_bore_diameter_mm", "bore_diameter_mm", "clear_bore_diameter_mm")
        if name in part), None)
    if inner_field is None:
        raise ValueError(f"{key}: ordinary deflector material bore must be declared")
    inner = _number(part, inner_field) / 2.0
    outer_field = "mechanical_outer_diameter_mm" if "mechanical_outer_diameter_mm" in part else "outer_diameter_mm"
    outer = _number(part, outer_field) / 2.0
    if not end > start or not 0.0 < inner < outer:
        raise ValueError(f"{key}: ordinary deflector requires a positive axial extent and open material annulus")
    clear_field = "mechanical_clear_bore_diameter_mm" if "mechanical_clear_bore_diameter_mm" in part else inner_field
    clear = _number(part, clear_field) / 2.0
    centres = part.get("interaction_centers_local_z_mm")
    if not isinstance(centres, (tuple, list)) or len(centres) != 2:
        raise ValueError(f"{key}: supported ordinary deflector requires exactly two finite-coil centres")
    centres = [_number({"key": key, "centre": value}, "centre") for value in centres]
    coil_length = _number(part, "mechanical_coil_length_mm", part.get("effective_thickness_mm",
                         part.get("active_length_mm", (end-start)/2)))
    if coil_length <= 0.0 or any(c-coil_length/2 < start-1e-9 or c+coil_length/2 > end+1e-9 for c in centres):
        raise ValueError(f"{key}: finite coil extent leaves the saved physical envelope")
    if centres[1]-centres[0] < coil_length:
        raise ValueError(f"{key}: finite-coil centres must be ordered without overlapping coils")
    width = outer-inner
    layer, span = 0.10*width, 120.0
    band = min(0.08*coil_length, 0.2*inner)
    x_low, y_low = inner+0.08*width, inner+0.26*width
    x_high, y_high = x_low+layer, y_low+layer
    skin, casing, pin_diameter = (_number(part, field) for field in DIMENSIONS)
    if min(skin, casing, pin_diameter) <= 0.0:
        raise ValueError(f"{key}: ordinary deflector skin, casing and pin dimensions must be positive")
    x_carrier = (x_low-skin, x_high+skin)
    y_carrier = (y_low-skin, y_high+skin)
    if not 0.0 < clear <= inner or x_carrier[0] < max(clear, inner):
        raise ValueError(f"{key}: ordinary deflector carrier intrudes into the declared clear bore or envelope")
    if not x_carrier[1] < y_carrier[0]:
        raise ValueError(f"{key}: ordinary deflector carriers must leave a positive inter-axis spacer")
    if y_carrier[1]+casing > outer:
        raise ValueError(f"{key}: ordinary deflector casing leaves the saved physical envelope")
    pin_radius = pin_diameter/2.0
    if pin_radius >= coil_length/2.0-band:
        raise ValueError(f"{key}: registration pins must remain inside the axial winding windows")
    # Both orthogonal saddle-loop windows include 45 and 225 degrees, but their
    # finite bands reduce the free angle. Use the smallest carrier radius for
    # a conservative cylindrical-pin angular bound over the complete pin.
    angular_clearance = min(60.0-math.degrees(band/((x_low+x_high)/2.0))-45.0,
                            45.0-(30.0+math.degrees(band/((y_low+y_high)/2.0))))
    slot_sweep = _number(part, SLOT_SWEEP_FIELD, 0.0)
    if slot_sweep < 0.0:
        raise ValueError(f"{key}: registration slot sweep must be nonnegative")
    if (pin_radius >= x_carrier[0] or angular_clearance <= 0.0
            or slot_sweep/2.0 + math.degrees(math.asin(pin_radius/x_carrier[0])) >= angular_clearance-1e-8):
        raise ValueError(f"{key}: registration pins and slots must remain inside both angular winding windows")
    return dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
                clear_radius_mm=clear, finite_coil_centres_mm=centres,
                mechanical_coil_length_mm=coil_length, winding_layer_thickness_mm=layer,
                winding_angular_span_deg=span, winding_band_width_mm=band,
                x_winding_radii_mm=(x_low, x_high), y_winding_radii_mm=(y_low, y_high),
                x_carrier_radii_mm=x_carrier, y_carrier_radii_mm=y_carrier,
                interaxis_spacer_radii_mm=(x_carrier[1], y_carrier[0]),
                outer_casing_radii_mm=(y_carrier[1], y_carrier[1]+casing),
                radial_skin_thickness_mm=skin, casing_thickness_mm=casing,
                pin_diameter_mm=pin_diameter, pin_angles_deg=(45.0, 225.0),
                registration_slot_sweep_deg=slot_sweep,
                material_inner_radius_mm=x_carrier[0],
                material_outer_radius_mm=y_carrier[1]+casing)


def validate_ordinary_deflector_part(part):
    """Public transactional validator for manifests and native model editing."""
    support_dimensions(part)


def _radial_pin(start, end, radius, angle_deg, centre, clip_inner, clip_outer):
    """A radial round pin clipped by exact coaxial inner/outer surfaces."""
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    # Local X is tangential, Y axial and Z radial; this is a proper rotation.
    rotation = [[-s, 0.0, c], [c, 0.0, s], [0.0, 1.0, 0.0]]
    cylinder = transformed(annulus(start, end, 0.0, radius), rotation=rotation,
                           offset=(0.0, 0.0, centre))
    # A cylinder cap is planar. Clip its circular cross-section to the actual
    # cylindrical carriers instead of allowing corner material to protrude.
    padding = max(1.0, radius)
    return difference(cylinder,
        annulus(centre-radius-padding, centre+radius+padding, 0.0, clip_inner),
        annulus(centre-radius-padding, centre+radius+padding, clip_outer, clip_outer+padding))


def _registration_slot(inner, outer, radius, angle_deg, centre, sweep_deg):
    """Round-ended angular slot with the existing pin's axial diameter.

    The sector joins two radial round end bores. The cutting tools extend
    beyond both radial carrier faces to avoid coincident Boolean surfaces;
    subtraction clips the result to each carrier. No polygonal approximation
    defines the exported slot, or opens the carrier's axial ends.
    """
    padding = max(1., radius)
    low, high = max(0., inner-padding), outer+padding
    caps = []
    for sign in (-1., 1.):
        angle = math.radians(angle_deg+sign*sweep_deg/2.)
        c, s = math.cos(angle), math.sin(angle)
        caps.append(transformed(annulus(low, high, 0., radius),
            rotation=[[-s, 0., c], [c, 0., s], [0., 1., 0.]], offset=(0., 0., centre)))
    # Keep the three overlapping cutters separate. A pre-fused slot generates
    # unnecessary trimmed-surface seams whose STEP round trip measurably
    # changes the volume integration. Cutting by the same primitive union in
    # one difference preserves the analytic boundary and existing tolerances.
    return (sector(centre-radius, centre+radius, low, high, angle_deg, sweep_deg), *caps)


def ordinary_deflector_support_solids(part, copper_specs):
    """Six passive material regions per stage, shared by native and STEP CAD."""
    d = support_dimensions(part)
    if d is None:
        return ()
    copper = {spec.region: spec for spec in copper_specs}
    expected = {f"coil_{stage}_{axis}_winding_{side}" for stage in (1, 2)
                for axis in "xy" for side in (1, 2)}
    if set(copper) != expected or any(spec.material != "copper_winding" for spec in copper.values()):
        raise ValueError(f"{part['key']}: ordinary support requires the unchanged eight copper winding regions")
    parameters = {**d, "construction": SUPPORT_PROFILE,
                  "dimension_status": "engineering_reconstruction_not_oem",
                  "topology_source": "US5994704A Figs. 1-4",
                  "material_grade": "unspecified", "optical_field_unchanged": True}
    solids = []
    for index, centre in enumerate(d["finite_coil_centres_mm"], 1):
        a, b = centre-d["mechanical_coil_length_mm"]/2, centre+d["mechanical_coil_length_mm"]/2
        inner, outer = d["x_carrier_radii_mm"][0], d["y_carrier_radii_mm"][1]
        pins = [_radial_pin(inner, outer, d["pin_diameter_mm"]/2, angle, centre, inner, outer)
                for angle in d["pin_angles_deg"]]
        sweep = d["registration_slot_sweep_deg"]
        slots = ([tool for angle in d["pin_angles_deg"] for tool in
                  _registration_slot(inner, outer, d["pin_diameter_mm"]/2., angle, centre, sweep)]
                 if sweep else pins)
        stage_parameters = {**parameters, "station_index": index,
                            "station_centre_mm": centre, "station_start_mm": a, "station_end_mm": b}
        for axis in "xy":
            low, high = d[f"{axis}_carrier_radii_mm"]
            coils = [copper[f"coil_{index}_{axis}_winding_{side}"].shape for side in (1, 2)]
            # Preserve the current inner-X / outer-Y winding order. Only the
            # outer carrier and intervening spacer receive adjustment slots.
            tools = slots if axis == "y" else pins
            shape = difference(annulus(a, b, low, high), *coils, *tools)
            solids.append(SolidSpec(str(part["key"]), f"coil_{index}_{axis}_carrier", "insulator",
                shape, (SUPPORT_SOURCE,), note=NOTE + (" " + SLOT_NOTE if sweep and axis == "y" else ""),
                parameters={**stage_parameters, "role": "insulating_coil_carrier", "axis": axis}))
        low, high = d["interaxis_spacer_radii_mm"]
        solids.append(SolidSpec(str(part["key"]), f"coil_{index}_interaxis_spacer", "insulator",
            difference(annulus(a, b, low, high), *slots), (SUPPORT_SOURCE,),
            note=NOTE + (" " + SLOT_NOTE if sweep else ""),
            parameters={**stage_parameters, "role": "insulating_interaxis_spacer"}))
        low, high = d["outer_casing_radii_mm"]
        solids.append(SolidSpec(str(part["key"]), f"coil_{index}_outer_casing", "insulator",
            annulus(a, b, low, high), (SUPPORT_SOURCE,), note=NOTE,
            parameters={**stage_parameters, "role": "insulating_outer_casing"}))
        for pin_index, shape in enumerate(pins, 1):
            solids.append(SolidSpec(str(part["key"]), f"coil_{index}_registration_pin_{pin_index}", "insulator",
                shape, (SUPPORT_SOURCE,), note=NOTE,
                parameters={**stage_parameters, "role": "internal_registration_pin",
                            "pin_angle_deg": d["pin_angles_deg"][pin_index-1]}))
    return tuple(solids)
