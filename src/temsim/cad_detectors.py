"""Detector material reconstructions without redefining detector acceptance.

EDS uses the existing schematic head placement, not an inferred physical active
area. EELS material is behind the existing indirect spectral-strip datum; the
overlapping alignment guide remains a logical readout reference. Coordinates
are owning-module mm for EDS and Cartesian branch mm for the EELS camera.
"""
from dataclasses import replace
import math
from numbers import Real

import numpy as np

from temsim.cad_geometry import SolidSpec, annulus, box, difference, revolve, transformed
from temsim.detector.eds_geometry import EDSDetectorArrayGeometry, resolve_eds_detector_part_data


EDS_PROFILE = "transverse_eds_detector_array"
EELS_PROFILE = "zebra_eels_detector"
STATUS = "engineering_reconstruction_not_oem"
REFERENCE = "docs/references/cad-detectors.md"
EDS_SOURCE = "https://patents.google.com/patent/US8410439B2/en"
SDD_SOURCE = "https://www.ketek.net/wp-content/uploads/2025/06/KETEK-VITUS_Product-Information.pdf"
EELS_SOURCE = "https://documents.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf"
INDIRECT_SOURCE = "https://patents.google.com/patent/US7642513B2/en"
EDS_FIELDS = ("mechanical_sensor_thickness_mm", "mechanical_support_thickness_mm",
              "mechanical_housing_wall_mm", "mechanical_back_wall_mm")
EELS_FIELDS = ("mechanical_scintillator_thickness_mm", "mechanical_optical_coupling_depth_mm",
               "mechanical_frame_margin_mm", "mechanical_housing_depth_mm",
               "mechanical_housing_wall_mm")
PROVENANCE_FIELDS = ("detector_material_geometry_status", "detector_material_geometry_source")


def _positive(part, field):
    value = part[field]
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{part.get('key', 'detector')}: {field} must be finite and positive")
    return float(value)


def _resolved(part):
    # Native module documents retain the shared definition reference, whereas
    # assembled PartRecords already carry its values. Accept either boundary.
    if (part.get("mechanical_profile") == EDS_PROFILE
            and part.get("eds_detector_definition_file") and "eds_system_key" not in part):
        return resolve_eds_detector_part_data(part)
    return part


def eds_head_geometry(part, by_key):
    """One pure definition of the pre-existing 2D schematic head geometry.

    No solid angle is converted to sensor area or distance. The old meridional
    drawing's clearance rule is simply lifted to the saved six azimuths.
    This establishes no three-dimensional mechanical clearance qualification.
    """
    part = _resolved(part)
    angular = EDSDetectorArrayGeometry.from_part_data(part)
    upper, lower = by_key["objective_upper_pole"], by_key["objective_lower_pole"]
    tip = max(_positive(upper, "mechanical_tip_diameter_mm")/2,
              _positive(lower, "mechanical_tip_diameter_mm")/2, 2.5)
    objective_radius = _positive(by_key["objective_lens"], "mechanical_outer_diameter_mm")/2
    pole_radius = max(_positive(upper, "mechanical_outer_diameter_mm"),
                      _positive(lower, "mechanical_outer_diameter_mm"))/2
    face = max(.45*tip, .025*objective_radius)
    length = max(.8*tip, .05*objective_radius)
    front, rear = 1.25*face, .82*face
    elevation = math.radians(angular.takeoff_angle_deg)
    distance = (pole_radius + 1. + front*math.sin(elevation))/math.cos(elevation)
    sample_z = float(part["local_center_z_mm"])
    if not math.isfinite(sample_z):
        raise ValueError("EDS sample datum must be finite")
    heads = []
    for index, azimuth in enumerate(angular.azimuth_centers_deg):
        angle = math.radians(azimuth)
        direction = np.array((math.cos(elevation)*math.cos(angle),
                              math.cos(elevation)*math.sin(angle), -math.sin(elevation)))
        tangent = np.array((-math.sin(angle), math.cos(angle), 0.))
        meridian = np.cross(direction, tangent)
        center = np.array((0., 0., sample_z)) + distance*direction
        heads.append(dict(index=index, azimuth_deg=azimuth, center_mm=center.tolist(),
                          direction=direction.tolist(), rotation=np.column_stack((tangent, meridian, direction)).tolist()))
    return dict(face_half_width_mm=face, housing_length_mm=length,
                front_half_width_mm=front, rear_half_width_mm=rear,
                drawing_distance_mm=distance, pole_display_clearance_mm=1.,
                sample_z_mm=sample_z, heads=heads,
                geometry_status="existing_schematic_head_not_physical_acceptance",
                physical_acceptance="unchanged_aggregate_angular_acceptance_no_sensor_face")


def _declared_fields(part):
    profile = part.get("mechanical_profile")
    fields = EDS_FIELDS if profile == EDS_PROFILE else EELS_FIELDS if profile == EELS_PROFILE else ()
    if not fields or not any(field in part for field in (*fields, *PROVENANCE_FIELDS)):
        return ()
    missing = [field for field in (*fields, *PROVENANCE_FIELDS) if field not in part]
    if missing:
        raise ValueError("Incomplete detector material geometry: " + ", ".join(missing))
    if part[PROVENANCE_FIELDS[0]] != STATUS or not isinstance(part[PROVENANCE_FIELDS[1]], str) or not part[PROVENANCE_FIELDS[1]].strip():
        raise ValueError("Detector material geometry requires explicit non-OEM engineering provenance")
    return fields


def validate_detector_material_geometry(part, by_key):
    """Validate opt-in mechanical fields; legacy definitions keep their guides.

    The angular model, sensor responses and existing readout dimensions are
    immutable inputs. Missing mechanical geometry is not defaulted silently.
    """
    part = _resolved(part)
    fields = _declared_fields(part)
    if not fields:
        return
    values = {field: _positive(part, field) for field in fields}
    if part["mechanical_profile"] == EDS_PROFILE:
        if part.get("windowless") is not True:
            raise ValueError("This EDS material reconstruction is only defined for the saved windowless array")
        geometry = eds_head_geometry(part, by_key)
        face, length = geometry["face_half_width_mm"], geometry["housing_length_mm"]
        front, rear = geometry["front_half_width_mm"], geometry["rear_half_width_mm"]
        wall, back = values["mechanical_housing_wall_mm"], values["mechanical_back_wall_mm"]
        stack = values["mechanical_sensor_thickness_mm"] + values["mechanical_support_thickness_mm"]
        if wall >= rear or stack >= length-back:
            raise ValueError("EDS sensor/support and back wall must fit the retained head depth")
        cavity_at_stack = front+(rear-front)*stack/length-wall
        if cavity_at_stack <= face:
            raise ValueError("EDS sensor/support must fit the retained tapered housing with positive radial clearance")
    else:
        # Existing helper owns count, strip dimensions, pitch and alignment
        # geometry validation. Do not reproduce those readout definitions here.
        from temsim.energy_filter_model_3d import _zebra_meshes, energy_filter_frame_mm
        _zebra_meshes(part)
        origin, rotation = energy_filter_frame_mm(part, by_key)
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            raise ValueError("EELS branch frame must remain finite")
        layer = values["mechanical_scintillator_thickness_mm"]
        coupling = values["mechanical_optical_coupling_depth_mm"]
        depth, wall = values["mechanical_housing_depth_mm"], values["mechanical_housing_wall_mm"]
        if layer+coupling >= depth-wall:
            raise ValueError("EELS scintillator and optical backing must fit ahead of the housing back wall")
        if wall >= values["mechanical_frame_margin_mm"]:
            raise ValueError("EELS housing wall must fit inside the declared frame margin")


def _eds_solids(part, by_key):
    geometry = eds_head_geometry(part, by_key)
    radius, length = geometry["face_half_width_mm"], geometry["housing_length_mm"]
    front, rear = geometry["front_half_width_mm"], geometry["rear_half_width_mm"]
    sensor, support, wall, back = (float(part[field]) for field in EDS_FIELDS)
    inside_back = front+(rear-front)*(length-back)/length-wall
    # Open front, finite side wall, closed rear. No fictitious entrance window.
    shell = revolve([(0., front-wall), (0., front), (length, rear), (length, 0.),
                     (length-back, 0.), (length-back, inside_back)])
    shapes = (("sensor", "silicon", annulus(0., sensor, 0., radius)),
              ("support", "insulator", annulus(sensor, sensor+support, 0., radius)),
              ("cold_shield", "nonmagnetic_housing", shell))
    values = []
    for head in geometry["heads"]:
        for name, material, shape in shapes:
            values.append(SolidSpec(part["key"], f"segment_{head['index']+1}_{name}", material,
                transformed(shape, rotation=head["rotation"], offset=head["center_mm"]),
                (REFERENCE, EDS_SOURCE, SDD_SOURCE), STATUS,
                "Windowless EDS engineering material in the existing schematic head envelope. "
                "The front disk is a mechanical Si die illustration, not a newly inferred active area; "
                "the unchanged angular photon acceptance does not intersect this CAD face. "
                "Cold shield and insulating support are simplified topology; cooling plumbing, shutter and circuits are not reconstructed.",
                {**{field: part[field] for field in EDS_FIELDS}, **head,
                 "face_radius_mm": radius, "housing_length_mm": length,
                 "front_radius_mm": front, "rear_radius_mm": rear,
                 "drawing_distance_mm": geometry["drawing_distance_mm"],
                 "physical_acceptance": geometry["physical_acceptance"]}))
    return tuple(values)


def _eels_solids(part, by_key):
    from temsim.energy_filter_model_3d import _zebra_meshes, energy_filter_frame_mm
    guides = _zebra_meshes(part)
    strips = [mesh for mesh in guides if mesh.region.startswith("active_strip_")]
    layer, coupling, margin, depth, wall = (float(part[field]) for field in EELS_FIELDS)
    corners = np.concatenate([mesh.vertices for mesh in strips])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    width, height = upper[0]-lower[0], upper[1]-lower[1]
    cx, cy = (upper[:2]+lower[:2])/2
    parameters = {field: part[field] for field in EELS_FIELDS}
    parameters.update(active_strip_count=len(strips), readout_datum_mm=0.,
        mechanical_front_width_mm=float(width+2*margin), mechanical_front_height_mm=float(height+2*margin),
        alignment_scope="unchanged_overlapping_reference_guide_not_second_material_sensor")
    values = []
    for guide in strips:
        low, high = guide.vertices.min(axis=0), guide.vertices.max(axis=0)
        center = ((low[0]+high[0])/2, (low[1]+high[1])/2, layer/2)
        values.append(SolidSpec(part["key"], guide.region+"_scintillator", "scintillator_unspecified",
            box(center, (high[0]-low[0], high[1]-low[1], layer)),
            (REFERENCE, EELS_SOURCE, INDIRECT_SOURCE), STATUS,
            "Indirect EELS conversion material begins at the unchanged spectral-strip face. "
            "Thickness is engineering; phosphor composition, coating and quantum efficiency are not inferred.", dict(parameters)))
    values.append(SolidSpec(part["key"], "optical_backing", "optical_coupler_unspecified",
        box((cx, cy, layer+coupling/2), (width, height, coupling)),
        (REFERENCE, EELS_SOURCE, INDIRECT_SOURCE), STATUS,
        "Simplified optical backing behind indirect EELS strips; source supports optical coupling topology, "
        "not the Zebra lens/fibre construction, material or thickness. No internal circuitry is represented.", dict(parameters)))
    outer = box((cx, cy, depth/2), (width+2*margin, height+2*margin, depth))
    # Cut through the front. Interior clears the unchanged strips and backing;
    # the back wall starts beyond both rather than overlapping the sensor.
    cavity = box((cx, cy, (depth-wall)/2-.5),
                 (width+2*(margin-wall), height+2*(margin-wall), depth-wall+1.))
    values.append(SolidSpec(part["key"], "housing", "nonmagnetic_housing", difference(outer, cavity),
        (REFERENCE, INDIRECT_SOURCE), STATUS,
        "Open-front simplified EELS camera frame and rear wall. Its external dimensions were not published; "
        "these newly declared engineering allocations are not an OEM camera enclosure.", dict(parameters)))
    origin, rotation = energy_filter_frame_mm(part, by_key)
    return tuple(replace(spec, shape=transformed(spec.shape, rotation=rotation.tolist(), offset=origin.tolist()))
                 for spec in values)


def detector_solids(part, by_key):
    """Closed leaves for native/STEP consumers; absent fields preserve legacy."""
    part = _resolved(part)
    fields = _declared_fields(part)
    if not fields:
        return None
    validate_detector_material_geometry(part, by_key)
    return _eds_solids(part, by_key) if part["mechanical_profile"] == EDS_PROFILE else _eels_solids(part, by_key)
