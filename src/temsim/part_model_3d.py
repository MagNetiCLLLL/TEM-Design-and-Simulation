"""Read-only, renderer-neutral meshes of existing module-local geometry.

Coordinates are x/y radial and z axial, all in mm. ``is_exact`` means that the
configured axisymmetric section is represented (up to angular tessellation),
not that the configuration is an OEM CAD model. Unsupported mechanisms retain
an explicitly labelled envelope; no holes, slots or material are invented.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import math
from numbers import Integral, Real

import numpy as np

from temsim.component_representation import (
    non_material_role, representation_note, shared_deflector_field_owner,
)
from temsim.magnetic_circuits import is_custom_mechanical_part, radial_profile_mm
from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.mechanical_profiles import MAGNETIC_LENS_MECHANICAL_PROFILES


@dataclass(frozen=True)
class DimensionSpec:
    path: tuple[str | int, ...]
    label: str
    value: float
    unit: str
    editable: bool = True
    reason: str = ""
    meaning: object | None = None

    @property
    def name(self):
        return self.path[2]

    @property
    def units(self):
        return self.unit


@dataclass(frozen=True)
class TriangleMesh:
    vertices: np.ndarray
    faces: np.ndarray
    key: str
    region: str = "body"
    color: tuple[float, ...] = (0.55, 0.61, 0.69, 1.0)
    material_class: str = "Unspecified"
    is_exact: bool = True
    description: str = "Configured axisymmetric body"
    face_groups: np.ndarray | None = None
    surfaces: Mapping = field(default_factory=dict)
    edges: tuple = ()
    # Non-material location guides: outlines overlaid without solid occlusion.
    wireframe: bool = False

    @property
    def part_key(self):
        return self.key

    @property
    def region_key(self):
        return self.region


@dataclass(frozen=True)
class PartModel3D:
    part_key: str
    name: str
    meshes: tuple[TriangleMesh, ...]
    fields: tuple[DimensionSpec, ...]
    notes: tuple[str, ...] = ()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _segments(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 3 <= value <= 4096:
        raise ValueError("angular_segments must be an integer from 3 to 4096")
    return int(value)


def _colour(material, exact):
    if not exact:
        return (0.53, 0.61, 0.70, 0.28)
    if "copper" in material or "winding" in material:
        return (0.83, 0.43, 0.20, 1.0)
    if "soft_magnetic" in material:
        return (0.36, 0.49, 0.64, 1.0)
    return (0.64, 0.70, 0.75, 1.0)


def _mesh(vertices, faces, key, region, material, exact, description):
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces) or not np.all(np.isfinite(vertices)):
        raise ValueError(f"{key}: geometry must produce a finite nonempty mesh")
    triangles = vertices[faces]
    with np.errstate(over="ignore", invalid="ignore"):
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
        raise ValueError(f"{key}: geometry produces degenerate triangles")
    vertices.setflags(write=False)
    faces.setflags(write=False)
    return TriangleMesh(vertices, faces, key, region, _colour(material, exact),
                        material, exact, description)


def revolve_section(section_mm, *, key, region="body", angular_segments=32,
                    material_class="Unspecified", is_exact=True,
                    description="Configured axisymmetric body"):
    """Revolve a simple closed (z, radius) material boundary into outward faces.

    The closing point is optional. Axis points each have one vertex, so solid
    cylinders and cone tips have triangle fans rather than degenerate quads.
    """
    count = _segments(angular_segments)
    points = []
    for row in section_mm:
        if len(row) != 2:
            raise ValueError("A section requires (z, radius) pairs")
        point = (_number(row[0], "z"), _number(row[1], "radius"))
        if point[1] < 0:
            raise ValueError("Section radii must be nonnegative")
        if not points or point != points[-1]:
            points.append(point)
    if points and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise ValueError("A material section needs at least three distinct boundary points")
    # Translation avoids cancellation for short bodies far along the column.
    origin = points[0][0]
    signed_area = sum((a[0] - origin) * b[1] - (b[0] - origin) * a[1]
                      for a, b in zip(points, points[1:] + points[:1]))
    if not math.isfinite(signed_area) or signed_area == 0:
        raise ValueError("A material section must have positive area")
    if signed_area > 0:
        points.reverse()
    angles = np.arange(count) * (2 * np.pi / count)
    vertices, rings, faces = [], [], []
    for z, radius in points:
        start = len(vertices)
        if radius == 0:
            vertices.append((0.0, 0.0, z))
            rings.append([start])
        else:
            vertices.extend(zip(radius * np.cos(angles), radius * np.sin(angles),
                                np.full(count, z)))
            rings.append(list(range(start, start + count)))
    for left, right in zip(rings, rings[1:] + rings[:1]):
        if len(left) == len(right) == 1:
            continue  # An axis edge sweeps zero area.
        for j in range(count):
            following = (j + 1) % count
            if len(left) == 1:
                faces.append((left[0], right[following], right[j]))
            elif len(right) == 1:
                faces.append((left[j], left[following], right[0]))
            else:
                faces.extend(((left[j], left[following], right[following]),
                              (left[j], right[following], right[j])))
    return _mesh(vertices, faces, str(key), str(region), str(material_class),
                 bool(is_exact), description)


def _annulus(start, end, inner, outer):
    if not start < end or not 0 <= inner < outer:
        raise ValueError("Body requires start < end and 0 <= inner radius < outer radius")
    return [(start, inner), (start, outer), (end, outer), (end, inner)]


def _part_index(document):
    parts = document.get("parts", ())
    if not isinstance(parts, (tuple, list)):
        raise ValueError("Module parts must be an array")
    result = {}
    for part in parts:
        if not isinstance(part, Mapping) or not isinstance(part.get("key"), str) or not part["key"]:
            raise ValueError("Every part needs a nonempty key")
        if part["key"] in result:
            raise ValueError(f"Duplicate part key: {part['key']}")
        result[part["key"]] = part
    return result


def _dimensions(part, by_key=None):
    from temsim.parameter_semantics import describe_parameter
    from temsim.cad_detectors import EDS_FIELDS, EDS_PROFILE, EELS_FIELDS, EELS_PROFILE
    fields = []
    detector_fields = (EDS_FIELDS if part.get("mechanical_profile") == EDS_PROFILE else
                       EELS_FIELDS if part.get("mechanical_profile") == EELS_PROFILE else ())
    def append(value, path, label, unit):
        if isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                append(item, (*path, index), f"{label} [{index}]", unit)
        elif not isinstance(value, bool) and isinstance(value, Real):
            fields.append(DimensionSpec(path, label, _number(value, str(path)), unit))

    for name, value in part.items():
        if is_custom_mechanical_part(part) and name == "vacuum_inner_diameter_mm":
            # Required schema context, not a material bore or active cutoff.
            # The complete TOML remains available in All parameters.
            continue
        unit = next((unit for unit in ("mm", "um", "nm", "deg") if name.endswith("_" + unit)), None)
        if unit is None:
            continue
        holder_parameter = (part.get("mechanical_profile") in {"transverse_goniometer", "side_entry_sample_holder"}
                            and name.startswith(("holder_", "stage_")))
        if not holder_parameter and name not in detector_fields and not any(token in name for token in ("diameter", "radius", "length", "width", "height", "thickness", "gap", "inset", "angle", "angular_span", "radial_profile", "material_intervals", "path_center", "path_entrance", "strip_center_pitch")):
            continue
        append(value, ("parts", part["key"], name),
               name.removesuffix("_" + unit).replace("_", " ").capitalize(), unit)
    parent = (by_key or {}).get(part.get("parent_key"), {})
    source_fields = [f"{side}_yoke_{edge}_local_z_mm" for side in ("upper", "lower") for edge in ("start", "end")]
    # Introspection must work even while the draft's dimensions are invalid.
    explicit_intervals = is_custom_mechanical_part(part) and "material_intervals_mm" in part
    split = (not is_custom_mechanical_part(part) and part.get("mechanical_profile") in {"magnetic_excitation_coil", "magnetic_lens_yoke"}
             and all(name in parent for name in source_fields))
    if explicit_intervals:
        reason = "Scale the copied material sections axially about the fixed centre in proportion to the nominal length; radial dimensions stay unchanged."
        fields = [replace(field, reason=reason) if field.path[-1] == "length_mm" else field
                  for field in fields]
    if split:
        reason = "Display-envelope length; physical sections use the parent yoke endpoints and coil inset below."
        fields = [replace(field, editable=False, reason=reason) if field.path[-1] == "length_mm" else field
                  for field in fields]
        if part["mechanical_profile"] == "magnetic_excitation_coil":
            source_fields.append("mechanical_coil_axial_inset_mm")
        for name in source_fields:
            if name in parent:
                append(parent[name], ("parts", parent["key"], name),
                       "Parent: " + name.removesuffix("_mm").replace("_", " "), "mm")
    annotated = []
    for item in fields:
        owner = (by_key or {}).get(item.path[1], part)
        meaning = describe_parameter(owner, item.path, by_key=by_key)
        label = meaning.label if len(item.path) == 3 else item.label
        if item.path[1] != part["key"]:
            label = "Parent: " + label
        if part.get("tip_particle_model") and item.path[-1] == "outer_diameter_mm":
            item = replace(item, editable=False, reason="Derived from the physical cap/cone, shank length and curvature radius.")
        reason = item.reason or meaning.description
        if item.path[-1] in detector_fields:
            reason = ("Engineering detector material dimension, not an OEM measurement. "
                      "Changes the native 3D and STEP material geometry; does not redefine "
                      "EDS angular acceptance, sensor response, or the EELS readout plane.")
        annotated.append(replace(item, label=label, reason=reason, meaning=meaning))
    return tuple(annotated)


def _radius(part, fields):
    for name, factor in fields:
        if name in part:
            return _number(part[name], name) * factor
    return None


def _extent(part):
    start = _number(part["local_start_z_mm"], "local_start_z_mm")
    end = _number(part["local_end_z_mm"], "local_end_z_mm")
    if end < start:
        raise ValueError(f"{part['key']}: axial endpoints are reversed")
    if end == start:
        for name in ("housing_length_mm", "electrode_length_mm", "plate_thickness_mm"):
            if name in part and _number(part[name], name) > 0:
                length = _number(part[name], name)
                return start - length / 2, end + length / 2
    return start, end


def _pole_section(part, start, end):
    get = lambda field: _number(part[field], field)
    length = end - start
    bore, outer, tip = (get(field) / 2 for field in
                        ("mechanical_bore_diameter_mm", "mechanical_outer_diameter_mm", "mechanical_tip_diameter_mm"))
    if not 0 <= bore < tip <= outer:
        raise ValueError(f"{part['key']}: pole requires 0 <= bore < tip <= outer radius")
    nose = get("pole_nose_axial_length_mm") if "pole_nose_axial_length_mm" in part else 0.38 * length
    if not 0 < nose <= length:
        raise ValueError(f"{part['key']}: pole nose must fit its axial length")
    style = part.get("pole_piece_geometry_style", "")
    if style not in {"", "embedded_hourglass_bore", "tapered_bore_pole",
                     "objective_vertical_back_inserted_shank_tapered_nose"}:
        return None
    if style == "objective_vertical_back_inserted_shank_tapered_nose":
        shank = get("pole_mounting_shank_axial_length_mm")
        shank_inner = get("pole_mounting_shank_inner_diameter_mm") / 2
        stem = get("pole_stem_outer_diameter_mm") / 2
        connector = get("pole_vacuum_connector_axial_length_mm") if "pole_vacuum_connector_axial_length_mm" in part else 0
        tail = get("pole_vacuum_connector_outer_diameter_mm") / 2 if connector else stem
        if not (0 <= connector < shank <= length - nose and 0 <= shank_inner < tail <= stem <= outer):
            raise ValueError(f"{part['key']}: invalid pole shank/connector/shoulder dimensions")
        section = [(0, shank_inner), (0, tail)]
        if connector:
            section += [(connector, tail), (connector, stem)]
        section += [(shank, stem), (shank, outer), (length - nose, outer),
                    (length, tip), (length, bore), (shank, bore), (shank, shank_inner)]
    else:
        section = [(0, bore), (0, outer), (length - nose, outer), (length, tip), (length, bore)]
    # The shared C1/C2 cartridge's two pole faces are reversed relative to their
    # optical names, as in the existing physical-layout projection.
    template_key = part.get("geometry_template_key", part["key"]) if is_custom_mechanical_part(part) else part["key"]
    face_at_end = "upper" in template_key
    if template_key == "condenser_lens_1_lower_pole":
        face_at_end = True
    elif template_key == "condenser_lens_2_upper_pole":
        face_at_end = False
    return [(start + z if face_at_end else end - z, radius) for z, radius in section]


def _legacy_part_meshes(part, by_key, count, aperture_index=0, runtime=None):
    from temsim.cad_catalog import native_reconstruction
    solids = native_reconstruction(part, by_key, runtime=runtime, aperture_index=aperture_index)
    if solids is not None:
        from temsim.cad_mesh import meshes_from_solids
        meshes = meshes_from_solids(solids, count)
        if part.get("branch_path_only") or part.get("mechanical_profile") == "transverse_eds_detector_array":
            from temsim.cad_mesh import annotate_material_faces
            meshes = tuple(annotate_material_faces(mesh, part, runtime) for mesh in meshes)
        if part.get("branch_path_only"):
            if part.get("mechanical_profile") == "tapered_sector_prism":
                from temsim.energy_filter_model_3d import _prism_meshes
                meshes += tuple(mesh for mesh in _prism_meshes(part, by_key, count) if mesh.wireframe)
            elif part.get("mechanical_profile") == "zebra_eels_detector":
                from temsim.energy_filter_model_3d import energy_filter_meshes
                guides, _ = energy_filter_meshes(part, by_key, count, runtime, aperture_index)
                # The overlapping alignment rectangle is a logical readout
                # reference, not a second material sensor or a STEP solid.
                meshes += tuple(mesh for mesh in guides if mesh.wireframe)
        notes = [spec.note for spec in solids]
        if part.get("mechanical_profile") == "xo_energy_slit_assembly" and (not runtime or "gap_m" not in runtime):
            notes.append(f"{part['key']}: no runtime slit gap supplied; blade edges show the configured maximum travel opening.")
        return meshes, tuple(dict.fromkeys(notes))
    from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES
    if part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES:
        from temsim.corrector_model_3d import deflector_coil_meshes
        channels = sorted(str(key) for key, row in by_key.items()
                          if row.get("physical_host_key") == part["key"])
        return deflector_coil_meshes(part, count, drive_key=", ".join(channels))
    from temsim.energy_filter_model_3d import energy_filter_meshes, supports_energy_filter_part
    if supports_energy_filter_part(part):
        return energy_filter_meshes(part, by_key, count, runtime, aperture_index)
    from temsim.optics.electron_gun.tip_assembly import is_tip_part
    if is_tip_part(part):
        from temsim.tip_model_3d import tip_meshes
        return tip_meshes(part, count, runtime)
    from temsim.part_model_apertures import is_strip_aperture, strip_meshes
    if is_strip_aperture(part):
        return strip_meshes(part, count, runtime, aperture_index)
    key, profile = part["key"], part.get("mechanical_profile", "")
    material = str(part.get("material_class", "Unspecified"))
    start, end = _extent(part)
    outer = _radius(part, (("mechanical_outer_diameter_mm", .5), ("mechanical_outer_radius_mm", 1),
                           ("outer_diameter_mm", .5), ("outer_width_mm", .5)))
    inner = _radius(part, (("mechanical_inner_diameter_mm", .5), ("mechanical_bore_diameter_mm", .5),
                           ("mechanical_clear_bore_diameter_mm", .5), ("mechanical_bore_radius_mm", 1),
                           ("bore_diameter_mm", .5), ("clear_bore_diameter_mm", .5),
                           ("inner_diameter_mm", .5)))
    emit = lambda section, region="body", exact=True, description="Configured axisymmetric body": revolve_section(
        section, key=key, region=region, angular_segments=count, material_class=material,
        is_exact=exact, description=description)
    radial = radial_profile_mm(part, end - start)
    if radial is not None:
        section = [(start + z, radius) for z, _, radius in radial]
        section += [(start + z, radius) for z, radius, _ in radial[::-1]]
        return (emit(section, description="Configured magnetic radial profile"),), ()
    if profile == "magnetic_pole_piece" and end > start:
        section = _pole_section(part, start, end)
        if section is not None:
            notes = () if "pole_nose_axial_length_mm" in part else (
                f"{key}: nose uses the existing 0.38 × length reconstruction rule.",)
            if "pole_root_fillet_radius_range_mm" in part:
                notes += (f"{key}: fillet radius range is metadata; the existing physical model has sharp shoulders.",)
            return (emit(section, description="Configured pole section with an open bore"),), notes
    hole_field = next((name for name in ("aperture_hole_diameters_mm", "aperture_hole_diameters_um",
                                       "hole_diameters_mm", "hole_diameters_um") if name in part), None)
    if hole_field:
        holes = part[hole_field]
        if (not isinstance(holes, (list, tuple)) or not holes
                or isinstance(aperture_index, bool) or not isinstance(aperture_index, Integral)
                or not 0 <= aperture_index < len(holes)):
            raise ValueError(f"{key}: aperture_index must select an existing aperture")
        hole = _number(holes[aperture_index], hole_field) * (0.001 if hole_field.endswith("_um") else 1) / 2
        if hole <= 0:
            raise ValueError(f"{key}: the selected aperture diameter must be positive")
        if outer is None:
            return (), (f"{key}: selected existing aperture index={aperture_index}; outer plate envelope is undefined.",)
        if "plate_thickness_mm" in part:
            thickness = _number(part["plate_thickness_mm"], "plate_thickness_mm")
            center = _number(part.get("optical_reference_local_z_mm", part.get("local_center_z_mm", (start + end) / 2)), "aperture center")
            start, end = center - thickness / 2, center + thickness / 2
        description = (f"Selected existing aperture index={aperture_index}, diameter={hole * 2:g} mm; "
                       "outer plate envelope only, multi-hole positions are not defined")
        return (emit(_annulus(start, end, hole, outer), exact=False, description=description),), (f"{key}: {description}.",)
    if profile in MAGNETIC_LENS_MECHANICAL_PROFILES and outer is not None and inner is not None:
        parent = by_key.get(part.get("parent_key"), {})
        intervals = ()
        explicit_intervals = is_custom_mechanical_part(part) and "material_intervals_mm" in part
        if explicit_intervals:
            intervals = part["material_intervals_mm"]
        elif not is_custom_mechanical_part(part) and profile in {"magnetic_excitation_coil", "magnetic_lens_yoke"}:
            intervals = objective_layer_intervals_mm(parent, parent.get("local_start_z_mm", 0), profile)
        if intervals and (not explicit_intervals or len(intervals) == 2):
            return tuple(emit(_annulus(a, b, inner, outer), region,
                              description="Copied physical material interval" if explicit_intervals else "Parent-defined physical material interval")
                         for region, (a, b) in zip(("upper", "lower"), intervals)), ()
        if part.get("material_intervals_mm") is not None:
            intervals = part["material_intervals_mm"]
            pieces = [emit(_annulus(_number(a, "interval start"), _number(b, "interval end"), inner, outer))
                      for a, b in intervals]
            if not pieces:
                raise ValueError(f"{key}: material intervals must not be empty")
            return (_join_meshes(pieces),), ()
        return (emit(_annulus(start, end, inner, outer)),), ()
    exact = profile in {"circular_aperture", "vacuum_liner", "electrostatic_bias_tube"}
    if outer is None:
        return (), (f"{key}: no defined outer envelope; only existing dimensions are listed.",)
    if inner is None:
        # A vacuum clearance is not a hole in the material (e.g. a cathode or
        # specimen can occupy it). An unknown construction gets a solid envelope.
        inner = 0
        exact = False
    description = "Configured axisymmetric body" if exact else "Envelope only; internal/non-axisymmetric construction is not defined by this surface"
    if start == end:
        # Preserve a true reference plane; never manufacture a visible thickness.
        if not 0 <= inner < outer:
            return (), (f"{key}: reference plane has no material annulus.",)
        angles = np.arange(count) * (2 * np.pi / count)
        vertices = [(outer * np.cos(t), outer * np.sin(t), start) for t in angles]
        if inner == 0:
            vertices.append((0, 0, start))
            faces = [(count, j, (j + 1) % count) for j in range(count)]
        else:
            vertices += [(inner * np.cos(t), inner * np.sin(t), start) for t in angles]
            faces = [(j, (j + 1) % count, count + (j + 1) % count) for j in range(count)]
            faces += [(j, count + (j + 1) % count, count + j) for j in range(count)]
        return (_mesh(vertices, faces, key, "body", material, False, "Reference/envelope plane; no axial thickness"),), (f"{key}: zero-thickness reference/envelope plane.",)
    return (emit(_annulus(start, end, inner, outer), exact=exact, description=description),), (() if exact else (f"{key}: {description}.",))


def _join_meshes(pieces):
    offsets = np.cumsum([0] + [len(piece.vertices) for piece in pieces[:-1]])
    first = pieces[0]
    return _mesh(np.concatenate([piece.vertices for piece in pieces]),
                 np.concatenate([piece.faces + offset for piece, offset in zip(pieces, offsets)]),
                 first.key, first.region, first.material_class, all(piece.is_exact for piece in pieces),
                 "Configured disjoint material intervals")


def _part_meshes(part, by_key, count, aperture_index=0, runtime=None):
    # Saved channel envelopes (or even an explicit CAD base) do not establish
    # material ownership. Keep their metadata available without making solids.
    if non_material_role(part):
        return (), (f"{part['key']}: {representation_note(part)}",)
    from temsim.part_model_features import apply_model_features, annotate_legacy_mesh, validate_model_3d
    validate_model_3d(part)
    # An explicit new base can model parts whose old display has no usable solid.
    explicit = part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"
    if explicit:
        meshes, notes = (), ()
    else:
        meshes, notes = _legacy_part_meshes(part, by_key, count, aperture_index, runtime)
    meshes = tuple(annotate_legacy_mesh(mesh, part, by_key) for mesh in meshes)
    if "model_3d" in part:
        from temsim.energy_filter_model_3d import (
            energy_filter_component_pose, rigid_mesh, supports_energy_filter_part,
        )
        if supports_energy_filter_part(part):
            # The shared feature schema keeps its existing component-local
            # transverse XY/longitudinal Z convention. Convert branch meshes
            # to those axes before editing, then restore the branch frame.
            origin, rotation = energy_filter_component_pose(part, by_key)
            center = np.array((0., 0., _number(part["local_center_z_mm"], "local_center_z_mm")))
            local = tuple(rigid_mesh(mesh, rotation.T, center - rotation.T @ origin) for mesh in meshes)
            updated = apply_model_features(part, local, angular_segments=count)
            if not explicit and not any(feature.get("enabled", True) for feature in part["model_3d"].get("features", ())):
                # The generic solid annotator does not retain open-boundary or
                # path-guide edges. Carry them through the same CAD transform.
                from temsim.part_model_features import _transform
                matrix, offset = _transform(part)
                preserved = []
                for old, new in zip(local, updated, strict=True):
                    ids = {edge["id"] for edge in new.edges}
                    extra = []
                    for edge in old.edges:
                        if edge["id"] not in ids:
                            points = (edge["vertices"] - center) @ matrix.T + center + offset
                            points.setflags(write=False)
                            extra.append({**edge, "vertices": points})
                    preserved.append(replace(new, edges=new.edges + tuple(extra)))
                updated = tuple(preserved)
            meshes = tuple(rigid_mesh(mesh, rotation, origin - rotation @ center) for mesh in updated)
        else:
            # Keep CAD shape changes local, then apply mechanical placement.
            # Optical-axis tilt is an alignment imperfection, not a rotation
            # of the housing or any of its physical descendants.
            meshes = apply_model_features(lens_shape_part(part), meshes, angular_segments=count)
        notes += (f"{part['key']}: explicit 3D solid and feature geometry; the optical/magnetic solver does not infer a new field model from this mesh.",)
    from temsim.lens_pose import mechanical_part_transform_mm
    rotation, translation = mechanical_part_transform_mm(part, by_key)
    if not (np.array_equal(rotation, np.eye(3)) and np.array_equal(translation, np.zeros(3))):
        from temsim.energy_filter_model_3d import rigid_mesh
        meshes = tuple(rigid_mesh(mesh, rotation, translation) for mesh in meshes)
    return meshes, notes


def lens_shape_part(part):
    """Return CAD geometry with a lens's separately handled pose removed.

    Older files store that pose in ``model_3d.transform``.  It remains readable
    by the pose helpers: offsets place the material, while tilt affects only
    the optical axis. Scale and Boolean features still describe shape only
    and do not rescale a field.
    """
    from temsim.lens_pose import without_lens_rigid_cad
    return without_lens_rigid_cad(part)


def part_dimension_specs(document, part_key, *, runtime_values=None):
    """List existing dimension paths even when the draft cannot produce a mesh."""
    by_key = _part_index(document)
    if part_key not in by_key:
        raise ValueError(f"Unknown part: {part_key}")
    from temsim.part_model_features import feature_dimension_specs
    from temsim.part_model_apertures import operating_dimension_specs
    part = by_key[part_key]
    from temsim.parameter_semantics import describe_parameter
    from temsim.optics.field_hardware_binding import bound_field_owner
    specs = (_dimensions(part, by_key) + feature_dimension_specs(part)
             + operating_dimension_specs(part, (runtime_values or {}).get(part_key)))
    from temsim.part_model_apertures import has_optional_plate_thickness
    if has_optional_plate_thickness(part) and "plate_thickness_mm" not in part:
        path = ("parts", part_key, "plate_thickness_mm")
        meaning = describe_parameter(part, path, by_key=by_key)
        specs += (DimensionSpec(path, meaning.label, 0., "mm", reason=meaning.description, meaning=meaning),)
    from temsim.lens_pose import physical_pose_values, supports_physical_lens_pose, inherits_parent_lens_pose, column_pose_kind, PHYSICAL_POSE_FIELDS
    host_key = bound_field_owner(part, "rotation_x_mrad")
    pose_owner, seen = by_key.get(host_key, part), set()
    while pose_owner is not None and not supports_physical_lens_pose(pose_owner):
        if not inherits_parent_lens_pose(pose_owner):
            pose_owner = None
            break
        if pose_owner["key"] in seen:
            raise ValueError("Cyclic lens pose ownership")
        seen.add(pose_owner["key"])
        pose_owner = by_key.get(pose_owner.get("parent_key"))
    if pose_owner is not None:
        values = physical_pose_values(pose_owner)
        pose_names = set(values)
        # Canonical placement controls exist even for old files with no saved
        # pose; omit the old CAD controls to avoid two competing edit surfaces.
        if pose_owner is part:
            specs = tuple(spec for spec in specs if spec.path[2] not in pose_names and not (
                len(spec.path) > 4 and spec.path[2:4] == ("model_3d", "transform")
                and spec.path[4] in {"offset_mm", "rotation_deg"}))
        prefix = "Physical host: " if host_key else ("Parent lens: " if pose_owner is not part else "")
        specs += tuple(DimensionSpec(("parts", pose_owner["key"], name), prefix +
            ("Position " + name[7].upper() + " offset" if name.startswith("offset_")
             else "Optical tilt about " + name[9].upper()),
            float(value), "mrad" if name.endswith("_mrad") else "mm")
            for name, value in values.items())
    if part.get("tip_particle_model"):
        from temsim.tip_model_3d import tip_dimension_overrides
        specs = tip_dimension_overrides(part, specs, (runtime_values or {}).get(part_key))
    if part.get("branch_path_only"):
        specs = tuple(replace(item, editable=False,
                              reason="Zero main-column packing length; this component uses its branch path coordinate and its separately declared physical dimensions.")
                      if item.path[-1] == "length_mm" else item for item in specs)
    specs = tuple(item if item.meaning is not None else replace(
        item, meaning=describe_parameter(by_key.get(item.path[1], part), item.path, by_key=by_key))
        for item in specs)
    if non_material_role(part):
        # Legacy fields remain inspectable for saved solver settings;
        # their old envelope names must not imply an independently built part.
        reason = (representation_note(part) + " This stored value is not a dimension "
                  "of an independently drawn material body.")
        metadata = []
        for item in specs:
            if item.path[1] != part_key:
                # A routed physical host pose is an actual placement control,
                # not obsolete channel-envelope metadata.
                metadata.append(item)
                continue
            if item.path[2] in PHYSICAL_POSE_FIELDS and column_pose_kind(part) == "field":
                metadata.append(replace(item, label="Field " + item.label))
                continue
            host = (shared_deflector_field_owner(part_key, item.path[2])
                    or bound_field_owner(part, item.path[2]))
            item_reason = reason + (f" Edit {host} to change the shared structure." if host else "")
            label = "Stored " + " / ".join(
                str(piece).removesuffix("_" + item.unit).replace("_", " ") for piece in item.path[2:])
            metadata.append(replace(
                item, label=label, reason=item_reason, editable=item.editable and not host,
                meaning=replace(item.meaning, label=label, category="unknown",
                                category_label="Channel / reference metadata", description=item_reason),
            ))
        specs = tuple(metadata)
    specs = tuple(replace(item, editable=False,
                          reason=f"Derived from {owner}; edit its physical geometry or placement. For an anchored insert, its optical reference offset controls the axial relation.")
                  if (owner := bound_field_owner(by_key.get(item.path[1], part), item.path[2]))
                  else item for item in specs)
    return specs


def part_model_from_document(document, part_key, *, angular_segments=32, include_children=True,
                             aperture_index=0, runtime_values=None):
    """Read one part and optionally its declared descendants/shared bodies.

    Ownership is preserved: a shared cartridge retains its own key and occurs
    once, rather than becoming a filled cylinder for each optical channel.
    """
    by_key = _part_index(document)
    count = _segments(angular_segments)
    if part_key not in by_key:
        raise ValueError(f"Unknown part: {part_key}")
    keys = {part_key}
    if include_children:
        while True:
            more = {key for key, part in by_key.items()
                    if part.get("parent_key") in keys or keys.intersection(part.get("magnetic_lens_keys", ()))}
            if more <= keys:
                break
            keys.update(more)
    meshes, notes = [], []
    for key, part in by_key.items():
        if key in keys:
            items, messages = _part_meshes(part, by_key, count, aperture_index if key == part_key else 0,
                                          (runtime_values or {}).get(key))
            meshes.extend(items)
            notes.extend(messages)
    part = by_key[part_key]
    return PartModel3D(part_key, str(part.get("name", part_key)), tuple(meshes),
                       part_dimension_specs(document, part_key, runtime_values=runtime_values), tuple(notes))


def module_model_from_document(document, *, angular_segments=32, runtime_values=None,
                               aperture_indices=None):
    """Render each declared part once, without duplicating shared ownership."""
    by_key = _part_index(document)
    count = _segments(angular_segments)
    meshes, fields, notes = [], [], []
    for part in by_key.values():
        items, messages = _part_meshes(part, by_key, count,
            aperture_index=(aperture_indices or {}).get(part["key"], 0),
            runtime=(runtime_values or {}).get(part["key"]))
        meshes.extend(items)
        fields.extend(part_dimension_specs(document, part["key"], runtime_values=runtime_values))
        notes.extend(messages)
    module = document.get("module", {})
    return PartModel3D(str(module.get("key", "module")), str(module.get("name", "Module")),
                       tuple(meshes), tuple(fields), tuple(notes))
