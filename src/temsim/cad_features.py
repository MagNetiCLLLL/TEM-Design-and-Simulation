"""The B-rep counterpart of the native part-local feature convention.

All cuts precede Rz @ Ry @ Rx @ diag(scale_x, scale_y, 1).  Physical
placement is applied separately, once, by the catalog just as by the viewer.
"""
from dataclasses import replace
import math
import numpy as np

from temsim.cad_geometry import SolidSpec, box, difference, transformed, union


def affine(shape, matrix, offset):
    return {"kind": "affine", "shape": shape, "matrix": np.asarray(matrix).tolist(),
            "offset_mm": np.asarray(offset).tolist()}


def map_solid(spec, operation):
    parameters = dict(spec.parameters)
    if "editor_original_shape" in parameters:
        parameters["editor_original_shape"] = operation(parameters["editor_original_shape"])
        parameters["editor_tools_global"] = [operation(tool) for tool in parameters["editor_tools_global"]]
    return replace(spec, shape=operation(spec.shape), parameters=parameters)


def feature_tool(feature):
    depth = float(feature["depth_mm"])
    radius = float(feature["diameter_mm"] if feature["kind"] == "hole" else feature["width_mm"]) / 2
    cylinder = {"kind": "cylinder", "radius_mm": radius, "length_mm": depth, "center_mm": [0., 0., 0.]}
    shape = cylinder
    if feature["kind"] == "slot":
        half = (float(feature["length_mm"]) - 2 * radius) / 2
        if half > 0:
            shape = union(transformed(cylinder, offset=(-half, 0., 0.)),
                          box((0., 0., 0.), (2 * half, 2 * radius, depth)),
                          transformed(cylinder, offset=(half, 0., 0.)))
    angle = math.radians(float(feature.get("rotation_deg", 0)))
    spin = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    orient = {"z": np.eye(3), "x": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
              "y": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])}[feature.get("axis", "z")]
    return transformed(shape, rotation=(orient @ spin).tolist(), offset=feature.get("center_mm", [0, 0, 0]))


def apply_solid_features(part, legacy):
    from temsim.part_model_features import validate_model_3d, _transform
    validate_model_3d(part)
    if "model_3d" not in part:
        return tuple(legacy)
    model = part["model_3d"]
    base = model.get("base", {"kind": "existing"})
    center = np.array([0., 0., float(part["local_center_z_mm"])])
    kind = base.get("kind", "existing")
    if kind == "existing":
        values = tuple(replace(spec, shape=transformed(spec.shape, offset=-center)) for spec in legacy)
    elif kind == "step":
        from temsim.step_assets import source_transform
        asset, matrix, offset = source_transform(base)
        values = tuple(SolidSpec(part["key"], f"step_{i + 1}", "imported_unspecified",
                       affine(leaf["shape"], matrix, offset), (), "imported_geometry",
                       "Imported STEP material geometry; no inferred optical field or OEM qualification.",
                       {"step_name": leaf.get("name", ""), "step_path": leaf.get("path", ""),
                        "step_color_rgba": leaf.get("color_rgba") or [.64, .70, .75, 1.],
                        "step_material_layers": leaf.get("material_layers", [])})
                       for i, leaf in enumerate(asset["leaves"]))
    else:
        size = [float(base[k]) for k in ("width_mm", "height_mm", "length_mm")]
        shape = box([0., 0., 0.], size) if kind == "box" else {
            "kind": "elliptic_cylinder", "width_mm": size[0], "height_mm": size[1],
            "length_mm": size[2], "center_mm": [0., 0., 0.]}
        values = (SolidSpec(part["key"], "body", str(part.get("material_class", "unspecified")),
                  shape, (), "configured_geometry", "Explicit editable material geometry."),)
    if not values:
        raise ValueError(f"{part['key']}: no closed solid to edit")
    tools = [feature_tool(f) for f in model.get("features", []) if f.get("enabled", True)]
    # A tool may miss an individual assembly leaf. Native editing and the
    # geometry worker validate material removal across the whole component.
    if tools:
        values = tuple(replace(s, shape=difference(s.shape, *tools),
                               parameters={**s.parameters, "editor_tools": tools,
                                           "editor_original_shape": s.shape,
                                           "editor_tools_global": tools}) for s in values)
    matrix, offset = _transform(part)
    return tuple(map_solid(s, lambda shape: affine(shape, matrix, center + offset)) for s in values)
