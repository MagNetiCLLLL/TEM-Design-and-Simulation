"""Portable, immutable STEP source bodies and their display tessellation.

STEP units are converted to millimetres by OpenCascade on import.  Imported
coordinates are retained explicitly; a bounding-box registration never creates
or moves an optical reference plane.
"""
from dataclasses import replace
from functools import lru_cache
import base64
import hashlib
import json

import numpy as np


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@lru_cache(maxsize=8)
def read_asset(payload):
    if not isinstance(payload, str) or len(payload) > 256 * 1024 * 1024:
        raise ValueError("STEP asset must be embedded JSON smaller than 256 MiB")
    asset = json.loads(payload)
    if asset.get("schema") != "temsim-cad-brep-asset-v1" or asset.get("units") != "mm":
        raise ValueError("Unsupported STEP asset schema or units (expected millimetres)")
    bounds = np.asarray(asset.get("metrics", {}).get("bounds_mm"), dtype=float)
    if bounds.shape != (6,) or not np.isfinite(bounds).all() or np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("STEP asset requires finite positive three-dimensional bounds")
    leaves = asset.get("leaves")
    if not isinstance(leaves, list) or not leaves or len(leaves) > 10000:
        raise ValueError("STEP asset needs 1–10000 closed material bodies")
    identities = set()
    for leaf in leaves:
        identity = str(leaf.get("id", ""))
        if not identity or identity in identities:
            raise ValueError("STEP body identifiers must be unique")
        identities.add(identity)
        shape = leaf.get("shape", {})
        if shape.get("kind") != "brep":
            raise ValueError("A STEP source must retain its B-rep")
        raw = base64.b64decode(shape.get("brep_base64", ""), validate=True)
        if not raw or hashlib.sha256(raw).hexdigest() != shape.get("sha256"):
            raise ValueError("STEP source B-rep checksum mismatch")
        mesh = leaf.get("mesh", {})
        vertices = np.asarray(mesh.get("vertices_mm"), dtype=float)
        faces = np.asarray(mesh.get("faces"))
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4
                or faces.dtype.kind not in "iu" or np.min(faces) < 0 or np.max(faces) >= len(vertices)):
            raise ValueError("STEP display tessellation is invalid")
        if len(mesh.get("face_ids", [])) != len(faces):
            raise ValueError("STEP tessellation must retain every face identifier")
        color = np.asarray(leaf.get("color_rgba") or [.64, .70, .75, 1.], dtype=float)
        if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
            raise ValueError("STEP colour must be RGBA in [0, 1]")
    return asset


def asset_bounds(asset):
    bounds = np.asarray(asset["metrics"]["bounds_mm"], dtype=float)
    return (bounds[:3] + bounds[3:]) / 2, bounds[3:] - bounds[:3]


def base_from_asset(asset):
    payload = canonical_json(asset)
    read_asset(payload)
    _, size = asset_bounds(asset)
    return dict(kind="step", width_mm=float(size[0]), height_mm=float(size[1]),
                length_mm=float(size[2]), asset_json=payload)


def source_transform(base):
    asset = read_asset(base["asset_json"])
    center, size = asset_bounds(asset)
    requested = np.asarray([base[k] for k in ("width_mm", "height_mm", "length_mm")], float)
    matrix = np.diag(requested / size)
    return asset, matrix, -matrix @ center


def imported_meshes(part):
    """Source tessellation scaled into the part-local editing frame."""
    from temsim.part_model_3d import _mesh
    from temsim.part_model_features import _decorate
    asset, matrix, offset = source_transform(part["model_3d"]["base"])
    dimensions = tuple(("parts", part["key"], "model_3d", "base", name)
                       for name in ("width_mm", "height_mm", "length_mm"))
    result = []
    for i, leaf in enumerate(asset["leaves"]):
        data = leaf["mesh"]
        vertices = np.asarray(data["vertices_mm"], float) @ matrix.T + offset
        faces = np.asarray(data["faces"], np.int64)
        region = f"step_{i + 1}"
        groups = np.asarray([f"{region}:face:{value}" for value in data["face_ids"]])
        metadata = {str(group): {"label": f"{leaf.get('name', region)} / {group.split(':')[-1]}",
                    "kind": "step_face", "parameter_paths": dimensions} for group in set(groups)}
        material = "imported_unspecified"
        mesh = _mesh(vertices, faces, part["key"], region, material, True,
                     "Imported STEP B-rep; display is a tolerance-controlled tessellation")
        mesh = replace(mesh, color=tuple(leaf.get("color_rgba") or mesh.color))
        result.append(_decorate(mesh, groups, metadata))
    return tuple(result)
