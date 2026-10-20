"""Inspect every installed component in downstream Z order, without tracing.

The report describes configured geometry, not OEM accuracy or collision
qualification. Contact sheets use the production assembly meshes; they do not
invent missing hardware. Run with the project interpreter.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def mesh_health(mesh):
    """Report indexed surface topology; open guides are not called solids."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    triangles = vertices[faces]
    area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)
    edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    local = triangles-vertices.mean(axis=0)
    volume = float(np.einsum("ij,ij->i", local[:, 0],
                            np.cross(local[:, 1], local[:, 2])).sum()/6.)
    result = dict(region=mesh.region, material=mesh.material_class, wireframe=bool(mesh.wireframe),
                vertices=len(vertices), triangles=len(faces),
                finite=bool(np.isfinite(vertices).all() and np.isfinite(area2).all()
                            and np.isfinite(volume)),
                degenerate_triangles=int(np.count_nonzero(area2 == 0)),
                boundary_edges=int(np.count_nonzero(counts == 1)),
                nonmanifold_edges=int(np.count_nonzero(counts > 2)),
                signed_volume_mm3=volume,
                bounds_mm=[vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
                configured_shape_exact=bool(mesh.is_exact),
                surfaces=sorted(mesh.surfaces), description=mesh.description,
                geometry_sha256=hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest())
    issues = []
    if not result["finite"]:
        issues.append("nonfinite_geometry")
    if result["degenerate_triangles"]:
        issues.append("degenerate_triangles")
    if result["nonmanifold_edges"]:
        issues.append("nonmanifold_edges")
    if not result["wireframe"]:
        if result["boundary_edges"]:
            issues.append("open_material_boundary")
        if result["signed_volume_mm3"] <= 0:
            issues.append("nonpositive_material_volume")
    result["issues"] = issues
    return result


def inspect(state, *, angular_segments=24):
    from temsim.assembly_model_3d import (
        _contexts, _omission_reason, assembly_model_fingerprint, assembly_model_from_assembly,
    )
    from temsim.cad_catalog import part_solids
    from temsim.component_representation import non_material_role
    from temsim.field_vector_geometry import _runtime_geometry

    assembly, runtime = state._resolved_assembly, _runtime_geometry(state)
    parts, documents = _contexts(assembly)
    fingerprint = assembly_model_fingerprint(assembly, runtime)
    model = assembly_model_from_assembly(assembly, runtime_values=runtime,
                                         angular_segments=angular_segments)
    meshes = defaultdict(list)
    for mesh in model.meshes:
        meshes[mesh.key].append(mesh)
    physical_parents = set()
    for part in parts:
        if (part.data.get("mechanical_profile") and not part.data.get("branch_path_only")
                and not non_material_role(part.data)
                and "reference_plane" not in part.data["mechanical_profile"]):
            if part.parent_key:
                physical_parents.add(part.parent_key)
            physical_parents.update(part.data.get("magnetic_lens_keys", ()))
    rows = []
    for part in sorted(parts, key=lambda p: (p.center_z_mm, p.key)):
        by_key = {row["key"]: row for row in documents[part.module_key]["parts"]}
        source = by_key[part.key]
        cad_error = None
        try:
            specs = part_solids(source, by_key, runtime=runtime.get(part.key)) or ()
        except (ValueError, TypeError, KeyError, RuntimeError) as exc:
            specs, cad_error = (), str(exc)
        health = [mesh_health(mesh) for mesh in meshes[part.key]]
        issues = [{"region": h["region"], "issues": h["issues"]} for h in health if h["issues"]]
        reason = _omission_reason(part, physical_parents)
        status = ("mesh_issue" if issues else "cad_error" if cad_error else "rendered" if health else
                  "omitted_material" if specs else
                  "intentional_nonmaterial" if reason else "unresolved_geometry")
        rows.append(dict(key=part.key, name=part.name, module=part.module_key,
                         parent=part.parent_key, centre_z_mm=part.center_z_mm,
                         extent_z_mm=[part.start_z_mm, part.end_z_mm],
                         profile=source.get("mechanical_profile", ""),
                         status=status, omission_reason=reason, cad_error=cad_error, issues=issues,
                         geometry_statuses=sorted({spec.status for spec in specs}),
                         references=sorted({ref for spec in specs for ref in spec.evidence}),
                         details=list(dict.fromkeys(spec.note for spec in specs)),
                         meshes=health))
    synthesized = [{"key": key, "meshes": [mesh_health(mesh) for mesh in group]}
                   for key, group in meshes.items() if key not in {p.key for p in parts}]
    for row in synthesized:
        row["issues"] = [{"region": h["region"], "issues": h["issues"]}
                         for h in row["meshes"] if h["issues"]]
        row["status"] = "mesh_issue" if row["issues"] else "rendered"
    mesh_issues = [{"key": row["key"], **issue}
                   for row in [*rows, *synthesized] for issue in row["issues"]]
    assert fingerprint == assembly_model_fingerprint(assembly, runtime)
    return dict(scope="Saved selected assembly; configured global XYZ in mm, +Z downstream. "
                      "Geometry inspection only; no trajectory, field solve or OEM qualification.",
                assembly_fingerprint=fingerprint, angular_segments=angular_segments,
                modules=[module.key for module in assembly.modules],
                part_count=len(rows), mesh_count=len(model.meshes),
                statuses=dict(Counter(row["status"] for row in rows)),
                synthesized_statuses=dict(Counter(row["status"] for row in synthesized)),
                mesh_issues=mesh_issues,
                material_omissions=[row["key"] for row in rows if row["status"] == "omitted_material"],
                cad_errors=[{"key": row["key"], "error": row["cad_error"]}
                            for row in rows if row["cad_error"]],
                errors=list(model.errors), parts=rows, synthesized=synthesized), meshes


def _face_colours(triangles, colour):
    """Ambient + directional diagnostic lighting, using camera-space normals."""
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(normals, lengths[:, None], out=np.zeros_like(normals),
                        where=lengths[:, None] > 0)
    light = np.array([.4, .6, 1.])
    light /= np.linalg.norm(light)
    strength = .52+.48*np.maximum(0., normals @ light)
    return np.asarray(colour[:3])*strength[:, None]


def _raster_tile(meshes, basis, size=400):
    """Fit an equal-scale orthographic view through the production depth rasterizer."""
    from temsim.gui.part_model_view import _rasterize
    pixels = np.full((size, size, 3), 255, dtype=np.uint8)
    triangles, colours = [], []
    for mesh in meshes:
        if mesh.wireframe:
            continue
        projected = (mesh.vertices @ basis.T)[mesh.faces]
        triangles.append(projected)
        colours.append(_face_colours(projected, mesh.color))
    if not triangles:
        return pixels
    projected = np.concatenate(triangles)
    colours = np.clip(np.concatenate(colours)*255, 0, 255).astype(np.uint8)
    low, high = projected.min(axis=(0, 1)), projected.max(axis=(0, 1))
    span = max(float(np.max(high[:2]-low[:2])), 1e-12)
    scale = size/(1.16*span)
    projected[:, :, :2] -= .5*(low[:2]+high[:2])
    projected[:, :, 0] = size/2+projected[:, :, 0]*scale
    projected[:, :, 1] = size/2-projected[:, :, 1]*scale
    depth = np.full((size, size), -np.inf)
    ids = np.full((size, size), -1, dtype=np.int32)
    face_ids = np.full((size, size), -1, dtype=np.int32)
    _rasterize(projected, colours, 0, 0, pixels, depth, ids, face_ids,
               max(1e-12, float(np.linalg.norm(high-low))*1e-10))
    return pixels


def contact_sheets(report, meshes, output):
    """Bounded, equal-axis isometric previews, one tile per installed solid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in report["parts"] if row["meshes"]]
    # Orthonormal camera basis; physical length ratios remain unchanged.
    basis = np.array([[1., -1., 0.], [.5, .5, -1.], [1., 1., 1.]])
    basis /= np.linalg.norm(basis, axis=1)[:, None]
    for start in range(0, len(rows), 12):
        fig, axes = plt.subplots(4, 3, figsize=(12, 14), layout="constrained")
        for axis, row in zip(axes.flat, rows[start:start+12]):
            axis.imshow(_raster_tile(meshes[row["key"]], basis), interpolation="nearest")
            axis.set_title(f'{row["key"]}\nZ {row["centre_z_mm"]:.3f} mm', fontsize=8)
            axis.axis("off")
        for axis in axes.flat[len(rows[start:start+12]):]:
            axis.axis("off")
        fig.suptitle("Installed component meshes · downstream order\n"
                     "Each tile fitted independently; configured geometry, not OEM CAD\n"
                     "Diagnostic lighting only; not an occlusion or collision qualification", fontsize=11)
        fig.savefig(output/f"contact-{start//12+1:02}.png", dpi=130)
        plt.close(fig)


def write_markdown(report, output):
    lines = ["# Installed 3D model inspection", "", report["scope"], "",
             f'{report["part_count"]} installed entries; {report["mesh_count"]} meshes. '
             f'Status: `{json.dumps(report["statuses"], sort_keys=True)}`.', "",
             f'{len(report["mesh_issues"])} mesh issues (including synthesized liners); '
             f'{len(report["cad_errors"])} CAD construction errors; '
             f'{len(report["material_omissions"])} defined material components missing from the assembly.', "",
             "An intentional omission is a control channel, reference or parent represented "
             "by its children. Closed surfaces and configured_shape_exact do not certify "
             "a manufacturer's construction or mechanical tolerances.", "",
             "| Z / mm | Component | Status | Regions / materials and limits |",
             "| ---: | --- | --- | --- |"]
    for row in report["parts"]:
        regions = ", ".join(f'{mesh["region"]}: {mesh["material"]}' for mesh in row["meshes"])
        detail = regions or row["omission_reason"] or row["cad_error"] or "No material mesh"
        limits = "; ".join(row["geometry_statuses"])
        lines.append(f'| {row["centre_z_mm"]:.3f} | `{row["key"]}` | {row["status"]} | '
                     f'{detail.replace("|", "/")} {limits} |')
    lines += ["", "Full region metadata, geometry provenance, topology measurements and "
              "unresolved details are in `audit.json`. Contact sheets cover every rendered "
              "installed component; synthetic connecting liners are listed separately in JSON.", ""]
    (output/"README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/column-model-review-20261020"))
    parser.add_argument("--angular-segments", type=int, default=24)
    parser.add_argument("--no-images", action="store_true")
    parser.add_argument("--gun", help="Catalog gun name; defaults to the saved default")
    parser.add_argument("--column", help="Catalog column name; defaults to the saved default")
    parser.add_argument("--recording", help="Catalog recording-system name; defaults to the saved default")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/"src"))
    from temsim.optics.column import default_state
    from temsim.assembly_catalog import AssemblyCatalog
    from dataclasses import replace
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state = default_state()
    selection = {name: getattr(args, name) for name in ("gun", "column", "recording")
                 if getattr(args, name) is not None}
    if selection:
        catalog = AssemblyCatalog()
        catalog.apply(state, replace(catalog.default_selection(), **selection))
    report, meshes = inspect(state, angular_segments=args.angular_segments)
    (output/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                               allow_nan=False), encoding="utf-8")
    write_markdown(report, output)
    if not args.no_images:
        contact_sheets(report, meshes, output)
    print(json.dumps({key: report[key] for key in ("part_count", "mesh_count", "statuses",
                                                  "synthesized_statuses", "mesh_issues", "cad_errors",
                                                  "material_omissions", "errors")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
