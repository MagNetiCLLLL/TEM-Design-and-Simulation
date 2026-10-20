"""Configured recording footprints, without inventing detector packages.

The camera already accepts a square in the particle model; annular STEM
receivers and circular screens retain their configured radii. Millimetres and
module-local Z are shared by native meshes and analytic CAD. The saved finite
layout thickness is a nominal slab allocation, not a measured sensitive depth.
"""
from dataclasses import replace
import math
from numbers import Real

import numpy as np

from temsim.cad_geometry import SolidSpec, annulus, box


PROFILES = frozenset({"camera_sensor_plane", "retractable_detector_plane"})
REFERENCE = "docs/references/cad-detectors.md"


def validate_detector_footprint(part):
    """Validate the shared receiver-shape convention without loading geometry.

    EDS square faces are a user-specified engineering representation, not an
    inference about the OEM sensor or its independent angular acceptance.
    Older documents without the explicit shape field retain their role's shape.
    """
    profile = part.get("mechanical_profile")
    expected = {
        "camera_sensor_plane": "square",
        "eftem_output_reference_plane": "square",
        "transverse_eds_detector_array": "square",
        "zebra_eels_detector": "rectangle",
    }.get(profile)
    if profile in PROFILES:
        width = part.get("outer_width_mm")
        inner = part.get("inner_diameter_mm", 0.)
        if isinstance(width, bool) or not isinstance(width, Real) or not math.isfinite(width) or width <= 0:
            raise ValueError("Detector outer_width_mm must be finite and positive")
        if isinstance(inner, bool) or not isinstance(inner, Real) or not math.isfinite(inner) or inner < 0:
            raise ValueError("Detector inner diameter must be finite and nonnegative")
        if inner >= width:
            raise ValueError("Recording footprint requires 0 <= inner diameter < outer width")
        if profile == "camera_sensor_plane":
            if inner:
                raise ValueError("A square camera footprint cannot have an annular inner diameter")
        else:
            expected = "annulus" if inner else "disk"
    if expected is None:
        return None
    if part.get("mechanical_footprint_shape", expected) != expected:
        raise ValueError(f"{part.get('key', 'detector')}: mechanical_footprint_shape must be {expected}")
    if profile == "zebra_eels_detector":
        for width_key, height_key in (("strip_active_width_mm", "strip_active_height_mm"),
                                      ("alignment_active_width_mm", "alignment_active_height_mm")):
            width, height = part.get(width_key), part.get(height_key)
            for name, value in ((width_key, width), (height_key, height)):
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive")
            if width == height:
                raise ValueError("EELS camera receiving regions must be rectangular with distinct width and height")
    return expected


def supports_recording_plane(part):
    return part.get("mechanical_profile") in PROFILES


def _geometry(part):
    from temsim.part_model_3d import _extent

    shape = validate_detector_footprint(part)
    start, end = _extent(part)
    width = float(part["outer_width_mm"])
    inner = float(part.get("inner_diameter_mm", 0.))
    return start, end, width, inner, shape


def recording_plane_solids(part, by_key=None):
    """One configured slab; a zero-thickness reference supplies no solid."""
    if not supports_recording_plane(part):
        return None
    start, end, width, inner, footprint = _geometry(part)
    if start == end:
        return ()
    shape = (box((0., 0., .5*(start+end)), (width, width, end-start))
             if footprint == "square" else annulus(start, end, inner/2, width/2))
    note = (
        f"Configured {footprint} recording footprint and {end-start:g} mm nominal slab. "
        "The upstream face is the signal-collection surface; rear and rim faces only describe "
        "the saved slab allocation. No sensor stack, housing, insertion mechanism or material "
        "composition is inferred. Geometry does not change acceptance or insertion state."
    )
    return (SolidSpec(
        str(part["key"]), "body", str(part.get("material_class", "Unspecified")), shape,
        (REFERENCE,), "configured_envelope", note,
        {"footprint": footprint, "outer_width_mm": width, "inner_diameter_mm": inner,
         "upstream_surface_z_mm": start, "downstream_surface_z_mm": end,
         "nominal_layout_thickness_mm": end-start},
    ),)


def _annotate(mesh, part, footprint):
    """Identify the real front/back/edge surfaces without extra material."""
    from temsim.part_model_features import _decorate, _normal_data, _paths

    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    groups = np.empty(len(mesh.faces), dtype=object)
    upstream = normals[:, 2] < -1.+1e-9
    downstream = normals[:, 2] > 1.-1e-9
    side = ~(upstream | downstream)
    groups[upstream], groups[downstream] = "collection_surface", "rear_surface"
    axial = _paths(part, ("local_start_z_mm", "local_end_z_mm", "length_mm",
                          "optical_reference_local_z_mm"))
    radial = _paths(part, ("outer_width_mm", "inner_diameter_mm"))
    metadata = {
        "collection_surface": {
            "label": "Upstream signal-collection surface", "kind": "cap",
            "parameter_paths": axial+radial,
        },
        "rear_surface": {
            "label": "Downstream face of nominal slab", "kind": "cap",
            "parameter_paths": axial+radial,
        },
    }
    if footprint == "square":
        for axis, direction, name in ((0, -1, "negative_x"), (0, 1, "positive_x"),
                                      (1, -1, "negative_y"), (1, 1, "positive_y")):
            key = "rim_"+name
            groups[side & (normals[:, axis]*direction > 1.-1e-9)] = key
            metadata[key] = {"label": name.replace("_", " ").title()+" rim",
                             "kind": "outer", "parameter_paths": axial+radial}
    else:
        inward = np.einsum("ij,ij->i", centers[:, :2], normals[:, :2]) < 0.
        groups[side & inward], groups[side & ~inward] = "bore_rim", "outer_rim"
        metadata["bore_rim"] = {"label": "Open inner bore rim", "kind": "inner",
                                "parameter_paths": axial+_paths(part, ("inner_diameter_mm",))}
        metadata["outer_rim"] = {"label": "Outer footprint rim", "kind": "outer",
                                 "parameter_paths": axial+_paths(part, ("outer_width_mm",))}
    return _decorate(mesh, groups, metadata, normals=normals)


def recording_plane_meshes(part, angular_segments):
    """Native view with selectable collection, rear, and rim faces."""
    if not supports_recording_plane(part):
        return None
    from temsim.cad_mesh import meshes_from_solids
    from temsim.part_model_3d import _mesh, _segments

    count = _segments(angular_segments)
    start, end, width, inner, footprint = _geometry(part)
    solids = recording_plane_solids(part)
    if solids:
        return tuple(_annotate(mesh, part, footprint)
                     for mesh in meshes_from_solids(solids, count)), tuple(spec.note for spec in solids)
    # A reference with no thickness is shown as a guide; it never becomes a
    # closed finite detector or a pretend zero-volume STEP material body.
    if footprint == "square":
        vertices = np.array([[-width/2, -width/2, start], [width/2, -width/2, start],
                             [width/2, width/2, start], [-width/2, width/2, start]])
        faces = np.array([[0, 2, 1], [0, 3, 2]])
    else:
        angles = np.arange(count)*2.*math.pi/count
        outer = np.column_stack((width/2*np.cos(angles), width/2*np.sin(angles),
                                 np.full(count, start)))
        if inner:
            inside = np.column_stack((inner/2*np.cos(angles), inner/2*np.sin(angles),
                                      np.full(count, start)))
            vertices = np.vstack((outer, inside))
            faces = np.array([face for i in range(count) for face in
                              ((i, count+(i+1) % count, (i+1) % count),
                               (i, count+i, count+(i+1) % count))])
        else:
            vertices = np.vstack((outer, (0., 0., start)))
            faces = np.array([(count, (i+1) % count, i) for i in range(count)])
    note = f"Configured {footprint} signal-collection reference; no finite thickness or package is defined."
    mesh = _mesh(vertices, faces, str(part["key"]), "body",
                 str(part.get("material_class", "Unspecified")), False, note)
    mesh = _annotate(replace(mesh, wireframe=True), part, footprint)
    return (mesh,), (note,)
