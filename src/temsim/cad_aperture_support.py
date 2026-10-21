"""Opt-in C2/C3 strips, single support rod and nominal screw in module-local mm.

The saved user-photo metadata identifies the construction family only. All new
sizes are explicitly stored engineering assumptions; no threads, extra working
holes, vacuum feedthrough or drive are inferred. The optical aperture remains
the existing selected/runtime hole at the existing reference plane.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np

from temsim.cad_geometry import SolidSpec, annulus, box, difference, transformed, union


PREFIX = "aperture_support_"
PROFILE = "single_rod_nominal_screw"
SOURCE = "https://patents.google.com/patent/JP2007329043A/en"
HOST_LABELS = {"condenser_aperture_2": "C2", "condenser_aperture_3": "C3"}
DIMENSIONS = (
    "plate_half_width_mm", "plate_lower_y_mm", "plate_upper_y_mm",
    "rod_start_y_mm", "rod_end_y_mm", "rod_width_mm", "rod_thickness_mm",
    "screw_center_y_mm", "screw_shank_diameter_mm", "screw_head_diameter_mm",
    "screw_head_thickness_mm",
)
FIELDS = frozenset(PREFIX + name for name in (*DIMENSIONS, "profile", "geometry_status", "source", "source_note"))


def _number(part, name):
    value = part.get(name)
    try:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be finite numeric")
        return float(value)
    except (OverflowError, TypeError) as exc:
        raise ValueError(f"{name} must be finite numeric") from exc


def aperture_material_scale_xy(part):
    """Saved CAD transverse scale, before fixed-mm slot clearance is added."""
    values = part.get("model_3d", {}).get("transform", {}).get("scale_xy", (1., 1.))
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError("Aperture CAD scale_xy must contain two positive finite factors")
    result = tuple(_number({"scale": value}, "scale") for value in values)
    if min(result) <= 0:
        raise ValueError("Aperture CAD scale_xy must contain two positive finite factors")
    return result


def _validate_detached_tube_placement(part, by_key, geometry, label, bounds=None):
    """Check actual replacement/copy bounds independently of an inactive recipe.

    No insertion slot is inferred for independent material. Subtractive CAD
    features cannot enlarge this conservative enclosing box. STEP base sizes
    use the same centred asset bounds as the production CAD transform.
    """
    if not geometry or "c1_c2_objective_vacuum_tube_outer_diameter_mm" not in geometry:
        return
    from temsim.lens_pose import mechanical_part_transform_mm
    from temsim.part_model_features import _transform, validate_model_3d
    from temsim.part_model_3d import lens_shape_part
    validate_model_3d(part)
    physical_rotation, translation = mechanical_part_transform_mm(part, by_key or {part["key"]: part})
    center = np.array([0., 0., float(part["local_center_z_mm"])])
    if bounds is None:
        base = part["model_3d"]["base"]
        size = np.array([base[name] for name in ("width_mm", "height_mm", "length_mm")], float)
        lo, hi = -size/2, size/2
    else:
        lo, hi = np.asarray(bounds)-center
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    matrix, offset = _transform(lens_shape_part(part))
    placed = (corners @ matrix.T+center+offset) @ physical_rotation.T+translation
    bounds = np.vstack((placed.min(0), placed.max(0)))
    nearest = np.maximum(np.maximum(bounds[0, :2], -bounds[1, :2]), 0.)
    radius = float(geometry["c1_c2_objective_vacuum_tube_outer_diameter_mm"])/2
    clearance = float(geometry.get("mechanical_material_tube_aperture_slot_clearance_mm", 0.))
    start = float(geometry["c1_c2_objective_vacuum_tube_start_z_mm"])
    end = float(geometry["c1_c2_objective_vacuum_tube_end_z_mm"])
    if (bounds[1, 2] > start and bounds[0, 2] < end
            and np.linalg.norm(nearest) <= radius+clearance):
        raise ValueError(f"Independent or replacement {label} aperture material must remain outside the continuous tube; "
                         "its rotated/scaled envelope has no declared insertion slot")


def validate_aperture_support(part, by_key=None, geometry=None):
    """Return explicit dimensions or None; reject an incomplete opt-in recipe."""
    declared = {name for name in part if name.startswith(PREFIX)}
    if not declared:
        return None
    host = part.get("geometry_template_key", part.get("key"))
    if part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        # Preserve the native recipe for a later restore, without using its rod,
        # screw or optical plane to constrain the active replacement material.
        from temsim.part_model_features import validate_model_3d
        validate_model_3d(part)
        _validate_detached_tube_placement(part, by_key, geometry, HOST_LABELS.get(host, ""))
        return None
    if declared != FIELDS:
        raise ValueError(f"Aperture support fields missing {sorted(FIELDS-declared)} or unknown {sorted(declared-FIELDS)}")
    if (host not in HOST_LABELS
            or part.get("aperture_plate_form") != "perforated_strip"
            or part.get("aperture_plate_attachment") != "screw_to_single_connection_rod"):
        raise ValueError("The single-rod support requires a declared C2/C3 strip/attachment topology")
    label = HOST_LABELS[host]
    if part[PREFIX+"profile"] != PROFILE or part[PREFIX+"geometry_status"] != "engineering_reconstruction_not_oem":
        raise ValueError("Aperture support must identify its non-OEM engineering reconstruction")
    for name in ("source", "source_note"):
        if not isinstance(part[PREFIX+name], str) or not part[PREFIX+name].strip():
            raise ValueError(f"Aperture support {name} is required")
    d = {name: _number(part, PREFIX+name) for name in DIMENSIONS}
    w, a, b = d["plate_half_width_mm"], d["plate_lower_y_mm"], d["plate_upper_y_mm"]
    r0, r1, rw, rt = (d[name] for name in ("rod_start_y_mm", "rod_end_y_mm", "rod_width_mm", "rod_thickness_mm"))
    sy, sr, hr, ht = (d[name] for name in ("screw_center_y_mm", "screw_shank_diameter_mm", "screw_head_diameter_mm", "screw_head_thickness_mm"))
    sr, hr = sr/2, hr/2
    thickness = _number(part, "plate_thickness_mm")
    z = _number(part, "optical_reference_local_z_mm")
    outer = _number(part, "mechanical_outer_diameter_mm")/2
    if not (w > 0 and a < 0 < r0 < b < r1 and rt > 0 and ht > 0 and 0 < sr < hr < rw/2 <= w):
        raise ValueError("Aperture support needs a positive open strip, overlapping rod seat and smaller shank than head")
    if not (r0 < sy-hr and sy+hr < b and thickness > 0):
        raise ValueError("Aperture screw head must lie fully on both strip and rod seat")
    if not (math.hypot(w, max(abs(a), abs(b))) < outer and math.hypot(rw/2, r1) < outer):
        raise ValueError("Aperture support must remain inside the saved radial mechanism envelope")
    if not (_number(part, "local_start_z_mm") <= z-thickness/2-ht
            < z+thickness/2+rt <= _number(part, "local_end_z_mm")):
        raise ValueError("Aperture support must remain inside the saved axial mechanism envelope")
    maximum = _number(part, "maximum_radius_mm")
    if not 0 < maximum < min(w, -a, sy-hr):
        raise ValueError("Aperture strip must retain the full existing maximum working opening")
    if geometry and "c1_c2_objective_vacuum_tube_outer_diameter_mm" in geometry:
        tube_radius = float(geometry["c1_c2_objective_vacuum_tube_outer_diameter_mm"])/2
        clearance = float(geometry.get("mechanical_material_tube_aperture_slot_clearance_mm", 0.))
        from temsim.lens_pose import mechanical_part_transform_mm
        physical_rotation, translation = mechanical_part_transform_mm(part, by_key or {part["key"]: part})
        # Main-column aperture rotations are optical-axis tilt; mechanical
        # translation remains the established physical placement convention.
        tube_start = float(geometry["c1_c2_objective_vacuum_tube_start_z_mm"])
        tube_end = float(geometry["c1_c2_objective_vacuum_tube_end_z_mm"])
        from temsim.magnetic_circuits import is_custom_mechanical_part
        if is_custom_mechanical_part(part):
            # A detached copy has no declared entry slot. Evaluate its actual
            # CAD placement, including independent rotations/scaling, instead
            # of pretending its old optical template proves an insertion path.
            _validate_detached_tube_placement(part, by_key, geometry, label,
                ([-w, a, z-thickness/2-ht], [w, r1, z+thickness/2+rt]))
            return d
        axial_overlap = (z+thickness/2+rt+float(translation[2]) > tube_start
                         and z-thickness/2-ht+float(translation[2]) < tube_end)
        _, scale_y = aperture_material_scale_xy(part)
        if axial_overlap and min(r0, sy-hr)*scale_y+float(translation[1]) <= tube_radius+clearance:
            raise ValueError(f"{label} rod and screw must remain outside the continuous tube with its declared clearance")
    return d


def aperture_support_solids(part, runtime=None, aperture_index=0):
    d = validate_aperture_support(part)
    if d is None:
        return None
    label = HOST_LABELS[part.get("geometry_template_key", part.get("key"))]
    from temsim.part_model_apertures import _opening
    diameter, paths, (dx, dy) = _opening(part, runtime, aperture_index)
    radius = (diameter or 0.)/2
    w, a, b = (d[name] for name in ("plate_half_width_mm", "plate_lower_y_mm", "plate_upper_y_mm"))
    sy, sr, hr = d["screw_center_y_mm"], d["screw_shank_diameter_mm"]/2, d["screw_head_diameter_mm"]/2
    if (abs(dx)+radius >= w or not a < dy-radius <= dy+radius < b
            or math.hypot(dx, dy-sy) <= radius+hr):
        raise ValueError(f"Working opening must fit the declared {label} strip and remain separate from its fixing screw")
    z, t = float(part["optical_reference_local_z_mm"]), float(part["plate_thickness_mm"])
    r0, r1, rw, rt = (d[name] for name in ("rod_start_y_mm", "rod_end_y_mm", "rod_width_mm", "rod_thickness_mm"))
    ht = d["screw_head_thickness_mm"]
    lo, hi = z-t/2, z+t/2
    # The cylindrical shank is the nominal threaded-envelope equivalent. The
    # matching subtraction prevents material overlap; no thread pitch exists.
    fixing_hole = transformed(annulus(lo-ht, hi+rt, 0., sr), offset=(0., sy, 0.))
    plate = difference(box((0., (a+b)/2, z), (2*w, b-a, t)), fixing_hole)
    if radius:
        plate = difference(plate, transformed(annulus(lo-t, hi+t, 0., radius), offset=(dx, dy, 0.)))
    rod = difference(box((0., (r0+r1)/2, hi+rt/2), (rw, r1-r0, rt)), fixing_hole)
    screw = transformed(union(annulus(lo-ht, lo, 0., hr), annulus(lo, hi+rt, 0., sr)), offset=(0., sy, 0.))
    parameters = {name: deepcopy(part[name]) for name in FIELDS}
    parameters.update(plate_thickness_mm=t, working_diameter_mm=diameter,
                      strip_half_width_mm=w, strip_y_bounds_mm=[a, b],
                      working_hole_count="selected_only_total_unknown", fixing_hole_count=1,
                      field_source_added=False)
    note = (f"{label} single-rod strip attachment: saved engineering dimensions, not OEM measurements. "
            "One nominal unthreaded screw and matching fixing bore; no additional working holes. "
            "Only the existing optical aperture clips rays; drive/feedthrough and thread detail are unresolved.")
    evidence = (str(part[PREFIX+"source"]), "docs/references/cad-gun-filter-apertures.md")
    return tuple(SolidSpec(part["key"], region, material, shape, evidence, "engineering_reconstruction", note, deepcopy(parameters))
                 for region, material, shape in (("body", str(part["aperture_plate_material"]), plate),
                     ("connection_rod", "nonmagnetic_housing", rod), ("fixing_screw", "nonmagnetic_housing", screw)))


def aperture_support_meshes(part, count, runtime=None, aperture_index=0):
    from temsim.cad_mesh import meshes_from_solids, annotate_material_faces
    from temsim.part_model_features import _decorate, _normal_data
    from temsim.part_model_apertures import _opening
    specs = aperture_support_solids(part, runtime, aperture_index)
    if specs is None:
        return None
    meshes = list(meshes_from_solids(specs, count))
    diameter, opening_paths, (dx, dy) = _opening(part, runtime, aperture_index)
    mesh = meshes[0]
    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    cap = np.abs(normals[:, 2]) > 1-1e-9
    # A cylindrical bore points toward its own axis; keep fixing and working
    # holes separate so selecting the screw cannot expose an optical control.
    sy = float(part[PREFIX+"screw_center_y_mm"])
    fixing = ((centers[:, 0]**2+(centers[:, 1]-sy)**2) < (float(part[PREFIX+"screw_shank_diameter_mm"])/2+1e-7)**2)
    working = np.einsum("ij,ij->i", centers[:, :2]-[dx, dy], normals[:, :2]) < 0
    groups = np.where(cap, np.where(normals[:, 2] > 0, "plate_positive", "plate_negative"),
                      np.where(fixing, "fixing_bore", np.where(working, "working_opening", "strip_outline")))
    thick = (("parts", part["key"], "plate_thickness_mm"),)
    outline = tuple(("parts", part["key"], PREFIX+name) for name in ("plate_half_width_mm", "plate_lower_y_mm", "plate_upper_y_mm"))
    fix = tuple(("parts", part["key"], PREFIX+name) for name in ("screw_center_y_mm", "screw_shank_diameter_mm"))
    metadata = {
        "plate_positive": dict(label="Aperture plate face (+Z)", kind="cap", parameter_paths=thick+opening_paths+outline),
        "plate_negative": dict(label="Aperture plate face (-Z)", kind="cap", parameter_paths=thick+opening_paths+outline),
        "working_opening": dict(label="Current working aperture opening", kind="inner", parameter_paths=thick+opening_paths),
        "fixing_bore": dict(label="Screw fixing bore; not an optical aperture", kind="inner", parameter_paths=thick+fix),
        "strip_outline": dict(label="Engineering strip outline", kind="outer", parameter_paths=thick+outline),
    }
    meshes[0] = _decorate(mesh, groups, metadata)
    meshes[1:] = [annotate_material_faces(mesh, part) for mesh in meshes[1:]]
    label = HOST_LABELS[part.get("geometry_template_key", part.get("key"))]
    note = (f"{label} transverse strip, single rod and screw are explicit engineering reconstruction, not measured OEM structure. "
            "The selected working hole is not specified offline unless provided; the total working-hole count and positions remain unspecified.")
    return tuple(meshes), (note,)
