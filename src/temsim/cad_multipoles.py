"""Configured multipole windings and explicitly unresolved envelopes.

Dimensions are module-local millimetres. Internal construction ratios are
declared reconstruction parameters, not OEM measurements or a field solve.
"""
from __future__ import annotations

import math
from dataclasses import replace
from numbers import Integral, Real

from temsim.cad_geometry import SolidSpec, annulus, box, difference, sector, transformed, union
from temsim.component_representation import non_material_role
from temsim.corrector_model_3d import PROFILE, winding_axial_extent, winding_dimensions
from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES


COIL_SOURCE = "https://patents.google.com/patent/US20090079531A1/en"
STIGMATOR_SOURCE = "https://patents.google.com/patent/US6548816B2/en"
CESCOR_SOURCE = "https://www.ceos-gmbh.de/en/produkte/stemkorrektoren/dcor-ascor/Mueller2006a.pdf"
CETCOR_SOURCE = "https://doi.org/10.1016/j.phpro.2008.07.093"
BCOR_SOURCE = "https://www.ceos-gmbh.de/user/pages/02.products/Poster-internet.pdf"
BCOR_PATENT_SOURCE = "https://patents.google.com/patent/EP1941531A2/en"
NOTE = ("Generic mechanical reconstruction, not OEM CAD or a magnetic-material solution. "
        "The linked optical component supplies the field; these solids do not add excitations.")


def _number(part, key, default=None):
    value = part.get(key, default)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{part.get('key', 'multipole')}.{key} must be finite")
    return float(value)


def _envelope(part):
    start, end = (_number(part, name) for name in ("local_start_z_mm", "local_end_z_mm"))
    inner_field = next((name for name in ("mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm",
        "mechanical_clear_bore_diameter_mm", "bore_diameter_mm", "clear_bore_diameter_mm") if name in part), None)
    if inner_field is None:
        raise ValueError(f"{part.get('key')}: material bore must be declared")
    inner = .5*_number(part, inner_field)
    outer_field = "mechanical_outer_diameter_mm" if "mechanical_outer_diameter_mm" in part else "outer_diameter_mm"
    outer = .5*_number(part, outer_field)
    if not (end > start and 0 < inner < outer):
        raise ValueError(f"{part.get('key')}: require positive axial extent and an open material annulus")
    clear = part.get("mechanical_clear_bore_diameter_mm")
    if clear is not None and .5*_number(part, "mechanical_clear_bore_diameter_mm") > inner:
        raise ValueError(f"{part.get('key')}: material intrudes into the declared clear bore")
    return start, end, inner, outer


def _solid(part, region, material, shape, parameters, evidence):
    return SolidSpec(str(part["key"]), region, material, shape, tuple(evidence),
                     note=NOTE, parameters=dict(parameters))


def _loop(start, end, inner, outer, angle_deg, span_deg, band):
    """Closed saddle conductor with an axial/angular window and an open bore."""
    window_span = span_deg-2*math.degrees(band/((inner+outer)/2))
    if end-start <= 2*band or window_span <= 0:
        raise ValueError("Winding band closes its central window")
    # Radially through-cut the window. The overlap is only between a boolean
    # tool and its parent; material solids do not overlap one another.
    extension = max(outer-inner, 1e-6)
    return difference(sector(start, end, inner, outer, angle_deg, span_deg),
        sector(start+band, end-band, max(0., inner-extension), outer+extension,
               angle_deg, window_span))


def deflector_insert_solids(part, by_key=None):
    """Exact configured DPH/DP envelope and bands, including legacy defaults."""
    if part.get("mechanical_profile") not in DEFLECTOR_COIL_PROFILES:
        return None
    if non_material_role(part):
        return ()
    from temsim.deflector_support_geometry import deflector_support_solids
    windings = deflector_insert_winding_solids(part, by_key)
    return windings + deflector_support_solids(part, windings)


def deflector_insert_winding_solids(part, by_key=None):
    """Copper only, also used as the exact insulation subtraction tools."""
    layer, span, band = winding_dimensions(part)
    allocated_start, allocated_end, inner, outer = _envelope(part)
    start, end = winding_axial_extent(part)
    linked = sorted(str(key) for key, row in (by_key or {}).items()
                    if row.get("physical_host_key") == part["key"])
    parameters = dict(inner_radius_mm=inner, outer_radius_mm=outer, start_mm=start, end_mm=end,
        allocated_start_mm=allocated_start, allocated_end_mm=allocated_end,
        winding_axial_length_mm=end-start,
        effective_field_length_mm=part.get("effective_thickness_mm"),
        winding_layer_thickness_mm=layer, winding_angular_span_deg=span,
        winding_band_width_mm=band, linked_field_channels=linked,
        construction="Two concentric orthogonal saddle pairs; CEOS patent Figs. 5/6, 10/11")
    solids = []
    for axis, (low, high) in enumerate(((inner, inner+layer), (outer-layer, outer))):
        layer_band = (min(.08*(end-start), .2*low)
                      if part.get("mechanical_profile") != PROFILE and "winding_band_width_mm" not in part else band)
        for side in range(2):
            params = {**parameters, "axis": "xy"[axis], "actual_band_width_mm": layer_band}
            shape = _loop(start, end, low, high, 90*axis+180*side, span, layer_band)
            solids.append(_solid(part, f"{'xy'[axis]}_winding_{side+1}", "copper_winding",
                                 shape, params, (COIL_SOURCE,)))
    return tuple(solids)


def _rotation(angle_deg):
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]


def _configured_envelope(part):
    """Retain the saved component extent without inventing its internals."""
    start, end, inner, outer = _envelope(part)
    return (SolidSpec(str(part["key"]), "body", "unspecified",
        annulus(start, end, inner, outer), (), status="configured_envelope",
        note=("Saved component envelope only. Internal pole/coil count, magnetic cores, "
              "housing, insulation and material distribution are unknown; this closed "
              "envelope does not assert that its whole volume is filled with material."),
        parameters=dict(start_mm=start, end_mm=end, inner_radius_mm=inner,
                        outer_radius_mm=outer, geometry_semantics="envelope_only_not_material_distribution")),)


def _cescor_hexapole_solids(part, *, image_family=False):
    """Source-backed CESCOR HP1/HP2 topology inside this model's saved envelope.

    Mueller et al. (2006), p. 443 and Fig. 3 on p. 444, identify six magnetic
    pole pins with pole pieces, coils and a yoke. The published dimensions are
    not transplanted into the simulator. A pole pin is one joined material
    body; no extra insulation, housing or vacuum tube is inferred here.
    """
    if not image_family:
        return _probe_hexapole_solids(part)
    start, end, inner, outer = _envelope(part)
    length, width, centre = end-start, outer-inner, (start+end)/2
    pole_length = (_number(part, "effective_length_mm") if image_family else .8*length)
    if not 0.0 < pole_length <= length:
        raise ValueError(f"{part['key']}: hexapole pole allocation must fit its saved envelope")
    yoke_inner = inner+.78*width
    shoe_outer, stem_start = inner+.20*width, inner+.16*width
    coil_start, coil_end = inner+.28*width, inner+.69*width
    available = min(2*coil_start*math.tan(math.pi/6), .6*width)
    params = dict(physical_pole_count=6, coil_count=6,
        topology_source="Mueller et al. 2006, p.443; p.444 Fig.3 real hexapole photograph",
        dimension_status="Internal proportions are model reconstruction; published 152 mm OD and pole dimensions are not used",
        start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
        shoe_radial_fraction=.20, stem_start_radial_fraction=.16,
        stem_width_mm=.25*available, shoe_sector_fraction=.65,
        coil_radial_fractions=[.28, .69], coil_axial_fraction=.70,
        coil_window_axial_fraction=.48, coil_tangential_fractions=[.45, .80],
        yoke_inner_radial_fraction=.78,
        construction="Six joined pole pins, six bulk windings, one yoke; no optional supports or individual turns")
    evidence = (CESCOR_SOURCE,)
    if image_family:
        params.update(
            topology_source=("Mueller et al. 2008 CETCOR author text, Figs.5/6: pole pins, coils and yoke; "
                             "six-pin count inferred from the CEOS family documented by Mueller2006 Fig.3"),
            topology_status="cetcor_family_inference_not_oem_internal_geometry",
            primary_image_availability="CETCOR2008 original mechanical figure unavailable; not visually verified",
            dimension_status=("Saved envelope and active length retained; internal proportions are reconstruction; "
                              "neither the published 152 mm nor 154 mm OD or bore dimensions are imported"),
            pole_axial_length_mm=pole_length,
            pole_axial_length_status="Model allocation from the configured effective length, not an OEM measurement",
        )
        evidence = (CETCOR_SOURCE, CESCOR_SOURCE)
    result = [_solid(part, "return_yoke", "soft_magnetic",
                     annulus(start, end, yoke_inner, outer), params, evidence)]
    for index in range(6):
        angle = 60*index
        shoe = sector(centre-pole_length/2, centre+pole_length/2,
                      inner, shoe_outer, 0., .65*60)
        # The short overlap fuses shoe and radial pin into one solid. Trim the
        # outer end to meet the circular yoke with no inter-part volume overlap.
        stem = box(((stem_start+yoke_inner)/2, 0., centre),
                   (yoke_inner-stem_start, .25*available, .40*length))
        stem = difference(stem, annulus(start, end, yoke_inner, outer+width))
        pin = union(shoe, stem)
        coil = difference(box(((coil_start+coil_end)/2, 0., centre),
                              (coil_end-coil_start, .80*available, .70*length)),
                          box(((coil_start+coil_end)/2, 0., centre),
                              (coil_end-coil_start+width, .45*available, .48*length)))
        for region, material, shape in ((f"pole_pin_{index+1}", "soft_magnetic", pin),
                                        (f"coil_{index+1}", "copper_winding", coil)):
            result.append(_solid(part, region, material,
                                 transformed(shape, rotation=_rotation(angle)), params, evidence))
    if image_family:
        result = [replace(s, status="engineering_reconstruction_family_inference",
            note=(NOTE + " CETCOR pin/coil/yoke structure uses a documented CEOS family inference. "
                  "The CETCOR2008 mechanical figure was unavailable; this is not a verified OEM cross-section. "
                  "No DPH winding, QPH, mirror plate or optional support is inferred.")) for s in result]
    return tuple(result)


def _probe_hexapole_solids(part):
    """Six tapered radial pole noses with explicit mechanical winding windows.

    Fig. 3 establishes topology and an inward taper, not the exact pole-face
    contour. A planar taper and rectangular bulk winding are engineering
    sections at the same detail level as the other native material parts.
    Existing box/boolean/rotation primitives serve both mesh and STEP export.
    """
    from temsim.probe_hexapole_geometry import probe_hexapole_dimensions, PREFIX
    d = probe_hexapole_dimensions(part)
    start, end = d['start_mm'], d['end_mm']
    inner, outer = d['inner_radius_mm'], d['outer_radius_mm']
    centre, length = (start+end)/2, end-start
    root = inner+d['taper_length_mm']
    yoke_inner = outer-d['yoke_thickness_mm']
    coil_start = inner+d['coil_radial_inset_mm']
    coil_end = coil_start+d['coil_radial_length_mm']
    tip_half, root_half = d['tip_width_mm']/2, d['root_width_mm']/2
    angle = math.degrees(math.atan2(root_half-tip_half, root-inner))
    # Cut two half spaces from a rectangular prism. Both cuts are oversized
    # construction tools, never extra physical solids. The resulting inward
    # taper is tangent to the saved bore and has no invented hyperbolic contour.
    reach = 8*max(outer, length)
    tools = []
    for sign in (-1, 1):
        tools.append(transformed(box((0., sign*reach/2, 0.), (reach, reach, 2*reach)),
            rotation=_rotation(sign*angle), offset=(inner, sign*tip_half, centre)))
    nose = difference(box(((inner+root)/2, 0., centre),
                          (root-inner, 2*root_half, d['pole_length_mm'])), *tools)
    overlap = min(.01, .01*d['taper_length_mm'])
    stem_start = root-overlap
    stem = box(((stem_start+yoke_inner)/2, 0., centre),
               (yoke_inner-stem_start, 2*root_half, d['stem_height_mm']))
    stem = difference(stem, annulus(start, end, yoke_inner, outer+outer-inner))
    pin = union(nose, stem)
    coil = difference(box(((coil_start+coil_end)/2, 0., centre),
        (coil_end-coil_start, d['coil_width_mm'], d['coil_height_mm'])),
        box(((coil_start+coil_end)/2, 0., centre),
        (2*outer, d['coil_window_width_mm'], d['coil_window_height_mm'])))
    params = dict(d, physical_pole_count=6, coil_count=6,
        topology_source="Mueller et al. 2006, p.443; p.444 Fig.3 real hexapole photograph",
        dimension_status="Engineering reconstruction; published 152 mm OD and pole dimensions are not used",
        geometry_revision="probe_radial_taper_v2",
        pole_face_status="Planar taper reconstruction, not measured OEM pole contour",
        physical_length_source=("explicit mechanical_hexapole_pole_length_mm" if PREFIX+'pole_length_mm' in part
                                else "legacy envelope allocation: 0.8 times mechanical length"),
        effective_field_length_mm=part.get('effective_length_mm'),
        coil_axis="radial", pole_pitch_deg=60.,
        coil_core_clearance_tangential_mm=(d['coil_window_width_mm']-d['root_width_mm'])/2,
        coil_core_clearance_axial_mm=(d['coil_window_height_mm']-d['stem_height_mm'])/2,
        construction="Six joined tapered pole pins, six radial-axis bulk windings, one shared yoke")
    evidence = (CESCOR_SOURCE, "docs/references/cad-probe-corrector.md")
    result = [_solid(part, 'return_yoke', 'soft_magnetic',
                     annulus(start, end, yoke_inner, outer), params, evidence)]
    for index in range(6):
        for region, material, shape in ((f'pole_pin_{index+1}', 'soft_magnetic', pin),
                                       (f'coil_{index+1}', 'copper_winding', coil)):
            result.append(_solid(part, region, material,
                transformed(shape, rotation=_rotation(60*index)), params, evidence))
    from temsim.cad_probe_hexapole_support import hexapole_support_solids
    result.extend(hexapole_support_solids(part))
    return tuple(result)


def _stigmator_windings(part):
    """Keep declared bulk coils; coil count never implies magnetic pole count."""
    if part.get("coil_topology") != "two_interleaved_four_pole_windings":
        return _configured_envelope(part)
    count = part.get("coil_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count != 8:
        raise ValueError(f"{part['key']}: two_interleaved_four_pole_windings requires coil_count=8")
    start, end, inner, outer = _envelope(part)
    length, width, centre = end-start, outer-inner, (start+end)/2
    basis = (_number(part, "channel_x_angle_deg", 0.),
             _number(part, "channel_y_angle_deg", 45.))
    if abs(math.sin(math.radians(2*(basis[1]-basis[0])))) < 1e-6:
        raise ValueError(f"{part['key']}: stigmator channels must be independent quadrupoles")
    # Keep the existing alternating X/Y identities. The two groups occupy
    # one axial station, including when their declared structural basis rotates.
    angles = [basis[index % 2]+90*(index//2) for index in range(count)]
    ordered = sorted(angle % 360 for angle in angles)
    separation = min(b-a for a, b in zip(ordered, ordered[1:]+[ordered[0]+360]))
    # Preserve the bulk winding allocation, but do not invent magnetic cores,
    # shoes, return yokes, housings or insulating supports around those coils.
    coil_start, coil_end = inner+.28*width, inner+.69*width
    # Bound each rectangular winding inside its own angular wedge. A valid
    # non-45-degree field basis must not manufacture intersecting copper loops.
    # This only allocates reconstruction geometry; it does not rescale fields.
    available = min(2*coil_start*math.tan(math.radians(separation/2)), .6*width)
    params = dict(coil_count=count, coil_topology=part["coil_topology"],
        start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
        coil_radial_fractions=[.28, .69], coil_axial_fraction=.70, coil_window_axial_fraction=.48,
        tangential_fractions=[.45, .80], minimum_coil_separation_deg=separation,
        winding_tangential_allocation_mm=available,
        construction="Declared two interleaved four-coil groups represented as bulk windings; dimensions are envelope allocations, not individual turns")
    evidence = tuple(source for source in (part.get("coil_topology_source_url"), STIGMATOR_SOURCE) if source)
    result = []
    for index in range(count):
        coil = difference(box(((coil_start+coil_end)/2, 0., centre),
                              (coil_end-coil_start, .80*available, .70*length)),
                          box(((coil_start+coil_end)/2, 0., centre),
                              (coil_end-coil_start+width, .45*available, .48*length)))
        coil_params = {**params, "coil_group": "xy"[index % 2],
                       "basis_angle_deg": basis[index % 2], "coil_azimuth_deg": angles[index]}
        result.append(_solid(part, f"coil_{index+1}", "copper_winding",
                             transformed(coil, rotation=_rotation(angles[index])), coil_params, evidence))
    return tuple(result)


def _deflector_solids(part):
    from temsim.ordinary_deflector_geometry import ordinary_deflector_support_solids, support_dimensions
    # Validate opt-in supports before a legacy envelope fallback can hide a
    # malformed construction declaration. Parts without support fields retain
    # their existing material geometry and dimensions.
    support_dimensions(part)
    start, end, inner, outer = _envelope(part)
    length, width = end-start, outer-inner
    centres = part.get("interaction_centers_local_z_mm")
    if centres is None:
        return _configured_envelope(part)
    if not isinstance(centres, (tuple, list)) or not 1 <= len(centres) <= 2:
        raise ValueError(f"{part['key']}: one or two declared finite-coil centres are required")
    centres = [_number({"key": part["key"], "centre": value}, "centre") for value in centres]
    coil_length = _number(part, "mechanical_coil_length_mm", part.get("effective_thickness_mm",
                         part.get("active_length_mm", length/len(centres))))
    if coil_length <= 0 or any(c-coil_length/2 < start-1e-9 or c+coil_length/2 > end+1e-9 for c in centres):
        raise ValueError(f"{part['key']}: finite coil extent leaves the saved physical envelope")
    if len(centres) == 2 and abs(centres[1]-centres[0]) < coil_length:
        raise ValueError(f"{part['key']}: axial finite coils overlap")
    layer, span = .10*width, 120.
    band = min(.08*coil_length, .2*inner)
    params = dict(start_mm=start, end_mm=end, inner_radius_mm=inner, outer_radius_mm=outer,
        finite_coil_centres_mm=list(centres), mechanical_coil_length_mm=coil_length,
        winding_layer_thickness_mm=layer, winding_angular_span_deg=span, winding_band_width_mm=band,
        winding_radial_fractions=[[.08, .18], [.26, .36]],
        construction="One two-axis saddle pair at each declared physical interaction centre")
    result = []
    for index, centre in enumerate(centres):
        a, b = centre-coil_length/2, centre+coil_length/2
        # The winding envelope does not establish additional supports or yokes.
        for axis, fraction in enumerate((.08, .26)):
            low = inner+fraction*width
            high = low+layer
            for side in range(2):
                result.append(_solid(part, f"coil_{index+1}_{'xy'[axis]}_winding_{side+1}", "copper_winding",
                    _loop(a, b, low, high, 90*axis+180*side, span, band), params, (COIL_SOURCE,)))
    return tuple(result) + ordinary_deflector_support_solids(part, result)


def multipole_solids(part, by_key=None) -> tuple[SolidSpec, ...] | None:
    """Resolve material hosts only; return None for a different part family."""
    if non_material_role(part):
        return ()
    # Energy-filter parts are placed in their bent branch frame by the filter
    # builder; notably its camera deflector uses electrostatic electrodes.
    if part.get("branch_path_only"):
        return None
    if part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES:
        return deflector_insert_solids(part, by_key)
    key = str(part.get("geometry_template_key", part.get("key", "")))
    profile = str(part.get("mechanical_profile", ""))
    if profile.startswith("electrostatic_"):
        return None
    if key in {"probe_hp1_hexapole", "probe_hp2_hexapole"}:
        return _cescor_hexapole_solids(part)
    if key in {"image_hp1_hexapole", "image_hp2_hexapole", "image_hp3_hexapole"}:
        values = _cescor_hexapole_solids(part, image_family=True)
        if part.get("bcor_geometry_status"):
            # The patent establishes three hexapole fields and permits sixfold
            # multipoles, but supplies no manufacturing dimensions. Preserve
            # the existing family reconstruction without inventing extra coils.
            values = tuple(replace(value,
                evidence=(*value.evidence, BCOR_SOURCE, BCOR_PATENT_SOURCE),
                note=(NOTE + " B-COR optical topology; six-pin internal construction "
                      "uses the existing CEOS family reconstruction, not verified "
                      "B-COR manufacturing geometry. No extra DPH winding is inferred."),
                parameters={**value.parameters,
                    "topology_status": "bcor_optical_topology_generic_six_pole_reconstruction",
                    "bcor_geometry_source": part.get("bcor_geometry_source", "")})
                for value in values)
        return values
    if key.endswith("stigmator") or profile == "magnetic_stigmator":
        return _stigmator_windings(part)
    if key.endswith("quadrupole") or profile == "magnetic_quadrupole":
        return _configured_envelope(part)
    if key.endswith("hexapole") or profile == "magnetic_hexapole":
        return _configured_envelope(part)
    if key.endswith("deflector") or profile == "magnetic_deflector":
        return _deflector_solids(part)
    return None
