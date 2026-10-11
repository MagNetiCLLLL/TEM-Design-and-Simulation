"""Isolated exact CAD operations on portable JSON; no simulator imports.

All coordinates are millimetres. BRep is the source; tessellation is disposable
display data. The optional CadQuery runtime is loaded only when an operation is
requested. No optical property is inferred from imported mechanical geometry.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

ASSET_SCHEMA = "temsim-cad-brep-asset-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _module(filename):
    spec = importlib.util.spec_from_file_location("_isolated_" + filename,
                                                 Path(__file__).with_name(filename + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate(leaves):
    if not leaves:
        raise ValueError("No solid leaves were found")
    rows = [row["metrics"] for row in leaves]
    return dict(valid=True, leaf_count=len(rows), solid_count=sum(r["solid_count"] for r in rows),
                volume_mm3=sum(r["volume_mm3"] for r in rows),
                volume_semantics="Sum of leaf volumes, not a Boolean union",
                bounds_mm=[min(r["bounds_mm"][i] for r in rows) for i in range(3)] +
                          [max(r["bounds_mm"][i] for r in rows) for i in range(3, 6)])


def _mesh(shape, tolerance, angular):
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in (tolerance, angular)):
        raise ValueError("Tessellation tolerances must be finite and positive")
    vertices, triangles, face_ids = [], [], []
    for face_id, face in enumerate(shape.Faces()):
        points, faces = face.tessellate(float(tolerance), float(angular))
        offset = len(vertices)
        vertices.extend([float(p.x), float(p.y), float(p.z)] for p in points)
        triangles.extend([int(v) + offset for v in f] for f in faces)
        face_ids.extend([face_id] * len(faces))
    if not triangles or any(not math.isfinite(v) for p in vertices for v in p):
        raise ValueError("BRep display tessellation is empty or nonfinite")
    return dict(vertices_mm=vertices, faces=triangles, face_ids=face_ids,
                linear_tolerance_mm=float(tolerance), angular_tolerance_rad=float(angular),
                semantics="Display tessellation only; face IDs are local to this BRep revision")


def _leaf(backend, shape, *, identity, name, path, color=None, layers=(),
          linear_tolerance_mm=.05, angular_tolerance_rad=.1, include_mesh=True):
    row = dict(id=identity, name=name, path=path, color_rgba=color,
               material_layers=list(layers), shape=backend.shape_brep_spec(shape),
               metrics=backend._shape_metrics(shape))
    if include_mesh:
        row["mesh"] = _mesh(shape, linear_tolerance_mm, angular_tolerance_rad)
    return row


def import_step(path, *, linear_tolerance_mm=.05, angular_tolerance_rad=.1):
    """Read XCAF instances, preserving names/colors and baking their locations.

    Instance identities are traversal indices rather than names, so repeated
    names cannot silently overwrite one another. Original path segments are
    kept as well as a readable path. STEP unit conversion is performed by OCC.
    """
    backend = _module("cad_step")
    cq = backend._cadquery()
    from cadquery.occ_impl.importers.assembly import _get_name, _get_ref_color, _get_shape_color
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_ChildIterator
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise ValueError("STEP could not be read")
    reader.ChangeReader().SetSystemLengthUnit(1.)
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    if not reader.Transfer(doc):
        raise ValueError("STEP transfer failed")
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    layers = XCAFDoc_DocumentTool.LayerTool_s(doc.Main())
    leaves = []

    def visit(label, location, segments, indices, inherited_color=None):
        target = label
        if tool.IsReference_s(label):
            target = TDF_Label()
            if not tool.GetReferredShape_s(label, target):
                raise ValueError("STEP contains an unresolved component reference")
            location = location * cq.Location(tool.GetLocation_s(label))
        instance_name = _get_name(label)
        name = (_get_name(target) if tool.IsAssembly_s(target) else instance_name) or _get_name(target) or "unnamed"
        segments = segments + [name]
        color = _get_ref_color(label) or _get_ref_color(target) or inherited_color
        if tool.IsAssembly_s(target):
            children = TDF_LabelSequence()
            tool.GetComponents_s(target, children)
            if children.Length() == 0:
                raise ValueError("STEP contains an empty assembly")
            for i in range(1, children.Length() + 1):
                visit(children.Value(i), location, segments, indices + [i], color)
            return
        shape = cq.Shape.cast(tool.GetShape_s(target))
        color = color or _get_shape_color(shape.wrapped, colors)
        shape = shape.moved(location)
        found = set()
        labels = [label, target]
        descendants = TDF_ChildIterator(target, True)
        while descendants.More():
            labels.append(descendants.Value())
            descendants.Next()
        for child in labels:
            values = layers.GetLayers(child)
            if values is not None:
                found.update(values.Value(i).ToExtString() for i in range(1, values.Length() + 1))
        row = _leaf(backend, shape, identity="leaf-" + "-".join(map(str, indices)),
                    name=name, path="/".join(segments), color=list(color.toTuple()) if color else None,
                    layers=sorted(found), linear_tolerance_mm=linear_tolerance_mm,
                    angular_tolerance_rad=angular_tolerance_rad)
        row["path_segments"] = segments
        row["instance_name"] = instance_name
        leaves.append(row)

    roots = TDF_LabelSequence()
    tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        visit(roots.Value(i), cq.Location(), [], [i])
    if path.read_bytes() != raw:
        raise RuntimeError("STEP source changed during import; retry against a stable file")
    return dict(schema=ASSET_SCHEMA, units="mm", source=dict(filename=path.name, format="STEP",
                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)), leaves=leaves,
                metrics=_aggregate(leaves), cadquery_version=cq.__version__,
                scope="Mechanical BRep only; no inferred optical parameters or material density",
                colors_scope="Assembly-instance/leaf colors; individual face colors are not used as material definitions")


def evaluate(solids, *, linear_tolerance_mm=.05, angular_tolerance_rad=.1, include_mesh=True):
    """Evaluate shared exact recipes, returning portable BReps and display data."""
    backend = _module("cad_step")
    normalized = [dict(row, key=str(row.get("key", index)), region=str(row.get("region", "body")),
                       shape=row.get("shape", row)) for index, row in enumerate(solids)]
    shapes, edit_report = backend.build_solid_rows(normalized)
    leaves, ids = [], set()
    for index, spec in enumerate(normalized):
        identity = str(spec.get("id", str(spec.get("key", index)) + "/" + str(spec.get("region", "body"))))
        if identity in ids:
            raise ValueError("Duplicate solid identity: " + identity)
        ids.add(identity)
        shape = shapes.get(spec["key"] + "/" + spec["region"])
        if shape is None:
            continue
        row = _leaf(backend, shape, identity=identity, name=str(spec.get("name", identity)),
                    path=str(spec.get("path", identity)), color=spec.get("color", spec.get("color_rgba")),
                    layers=spec.get("material_layers", []), linear_tolerance_mm=linear_tolerance_mm,
                    angular_tolerance_rad=angular_tolerance_rad, include_mesh=include_mesh)
        row["recipe_sha256"] = digest(spec.get("shape", spec))
        leaves.append(row)
    return dict(schema=ASSET_SCHEMA, units="mm", leaves=leaves, metrics=_aggregate(leaves), edit_report=edit_report,
                source=dict(format="exact_recipe", sha256=digest(solids)))


def collision(first, second, *, linear_tolerance_mm=1e-7, volume_tolerance_mm3=1e-9):
    """Exact occupied-volume intersection and minimum distance; no exemptions."""
    backend = _module("cad_step")
    a = backend.build_shape(first.get("shape", first))
    b = backend.build_shape(second.get("shape", second))
    for value in (linear_tolerance_mm, volume_tolerance_mm3):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError("Collision tolerances must be positive and finite")
    metrics_a, metrics_b = backend._shape_metrics(a), backend._shape_metrics(b)
    common = a.intersect(b)
    if not common.isValid():
        raise RuntimeError("BRep intersection is invalid")
    volume = sum(backend.shape_volume(s) for s in common.Solids())
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    extrema = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    if not extrema.IsDone() or extrema.NbSolution() < 1:
        raise RuntimeError("BRep minimum distance did not converge")
    distance = float(extrema.Value())
    relation = ("overlap" if volume > volume_tolerance_mm3 else "sub_tolerance_overlap" if volume > 0
                else "contact" if distance <= linear_tolerance_mm else "separate")
    return dict(first_id=first.get("id"), second_id=second.get("id"), relation=relation,
                intersection_volume_mm3=volume, distance_mm=distance,
                intersection_bounds_mm=backend._shape_metrics(common)["bounds_mm"] if volume > 0 else None,
                linear_tolerance_mm=linear_tolerance_mm, volume_tolerance_mm3=volume_tolerance_mm3,
                first_metrics=metrics_a, second_metrics=metrics_b)


def validate_catalog(catalog, *, optical_probes=(), binding_report_sha256=None):
    """Audit catalog alternatives separately, or explicit placed assemblies."""
    if catalog.get("units", "mm") != "mm":
        raise ValueError("Catalog must use millimetres")
    audit = _module("cad_assembly_validation")
    backend = _module("cad_step")
    cq = backend._cadquery()
    modules = catalog.get("modules", [catalog] if "solids" in catalog else [])
    by_key = {row["key"]: row for row in modules}
    if not modules or len(by_key) != len(modules):
        raise ValueError("Catalog requires unique modules")
    assemblies = catalog.get("assemblies") or [dict(key=row["key"], placements=[dict(
        module_key=row["key"], offset_mm=[0., 0., 0.])]) for row in modules]
    resolved, edit_reports, module_shapes = [], {}, {}
    for row in modules:
        shapes, edits = backend.build_solid_rows(row["solids"])
        module_shapes[row["key"]] = shapes
        removed = set(edits["removed_leaf_ids"])
        resolved.append(dict(row, solids=[s for s in row["solids"] if s["key"]+"/"+s["region"] not in removed]))
        edit_reports[row["key"]] = edits
    report = audit.audit_document(dict(units="mm", modules=resolved, assemblies=assemblies))
    resolved_by_key = {row["key"]: row for row in resolved}
    assembly_by_key = {row["key"]: row for row in assemblies}
    if len(assembly_by_key) != len(assemblies):
        raise ValueError("Duplicate assembly keys")
    probe_results = []
    for probe in optical_probes:
        module_key = probe.get("module_key")
        if module_key is None and len(modules) == 1:
            module_key = modules[0]["key"]
        if module_key not in by_key:
            raise ValueError("Optical probe must identify an existing module")
        spec = backend.build_shape(probe["shape"])
        assembly_key = probe.get("assembly_key")
        module_offset = (0., 0., 0.)
        coordinate_frame = "module_local_mm"
        if assembly_key is not None:
            if assembly_key not in assembly_by_key:
                raise ValueError("Optical probe references an absent assembly")
            placements = assembly_by_key[assembly_key]["placements"]
            owners = [i for i, placement in enumerate(placements) if placement["module_key"] == module_key
                      and (probe.get("owner_instance_key") is None or
                           placement.get("instance_key", placement["module_key"]) == probe["owner_instance_key"])]
            if len(owners) != 1:
                raise ValueError("Assembly optical probe needs one unambiguous owner module instance")
            owner_index = owners[0]
            module_offset = audit._vector(placements[owner_index].get("offset_mm", (0., 0., 0.)))
            # Probe recipes arrive in the owner's local frame, exactly like
            # its material. Translate once; never stack mutually exclusive
            # module variants or apply another energy-filter branch rotation.
            spec = spec.translate(module_offset)
            coordinate_frame = "global_column_mm"
        else:
            placements = [dict(module_key=module_key, offset_mm=module_offset)]
            owner_index = 0
        metrics = backend._shape_metrics(spec)
        expected_ids = set(probe.get("expected_solid_ids", []))
        ids = {row["key"] + "/" + row["region"] for row in by_key[module_key]["solids"]}
        if not expected_ids.issubset(ids):
            raise ValueError("Coverage probe references an absent solid")
        if probe["kind"] not in ("clearance", "coverage"):
            raise ValueError("Unknown optical probe kind")
        if probe["kind"] == "coverage" and not expected_ids:
            raise ValueError("Coverage probe needs expected material solid IDs")
        covered, blockers, unresolved = [], [], []
        for placement_index, placement in enumerate(placements):
            placed_key = placement["module_key"]
            offset = audit._vector(placement.get("offset_mm", (0., 0., 0.)))
            for row in resolved_by_key[placed_key]["solids"]:
                sid = row["key"] + "/" + row["region"]
                shape = module_shapes[placed_key][sid].translate(offset)
                if audit.bbox_distance(metrics["bounds_mm"], backend._shape_metrics(shape)["bounds_mm"]) > 1e-7:
                    continue
                common = spec.intersect(shape)
                if not common.isValid():
                    raise RuntimeError("Invalid optical probe intersection")
                volume = sum(backend.shape_volume(s) for s in common.Solids())
                unknown = audit.occupancy_kind(row) != "material"
                record = dict(solid_id=sid, volume_mm3=volume, module_key=placed_key,
                              instance_key=placement.get("instance_key", placed_key))
                if volume > 0 and unknown:
                    unresolved.append(record)
                elif placement_index == owner_index and sid in expected_ids:
                    covered.append(shape)
                elif volume > 1e-9:
                    blockers.append(record)
        uncovered = None
        if probe["kind"] == "coverage":
            remainder = spec.cut(*covered) if covered else spec
            if not remainder.isValid():
                raise RuntimeError("Invalid optical coverage Boolean")
            uncovered = sum(backend.shape_volume(s) for s in remainder.Solids())
        probe_results.append(dict(id=probe["id"], module_key=module_key, kind=probe["kind"],
            assembly_key=assembly_key, coordinate_frame=coordinate_frame, module_offset_mm=list(module_offset),
            probe_volume_mm3=metrics["volume_mm3"], uncovered_volume_mm3=uncovered,
            material_blockers=blockers, unresolved_intersections=unresolved,
            status="unresolved" if unresolved else "failed" if blockers or (uncovered or 0) > 1e-9 else "passed",
            scope=probe.get("scope"), note=probe.get("note")))
    result = dict(schema="temsim-cad-geometry-validation-v1", units="mm", catalog_sha256=digest(catalog),
                  binding_report_sha256=binding_report_sha256, geometry_report=report,
                  edit_reports=edit_reports, probe_results=probe_results)
    result["report_sha256"] = digest(result)
    return result


def export_step(solids, path, *, name="edited_geometry"):
    """Export exact recipes under stable IDs; retain original provenance in receipt."""
    backend = _module("cad_step")
    if Path(path).exists():
        raise ValueError("Choose a new STEP path; existing source files are never overwritten")
    cq = backend._cadquery()
    assembly = cq.Assembly(name=backend._label(name))
    materials, source_names = {}, []
    normalized = [dict(row, key=str(row.get("key", index)), region=str(row.get("region", "body")),
                       shape=row.get("shape", row)) for index, row in enumerate(solids)]
    shapes, edit_report = backend.build_solid_rows(normalized)
    for index, row in enumerate(normalized):
        label = "solid_" + str(index)
        shape = shapes.get(row["key"] + "/" + row["region"])
        if shape is None:
            continue
        color = row.get("color", row.get("color_rgba")) or [.64, .70, .75, 1.]
        assembly.add(shape, name=label, color=cq.Color(*color))
        materials[label] = row.get("material", "unspecified")
        source_names.append(dict(exported_name=label, source_name=row.get("name"),
                                 source_path=row.get("path"), material_layers=row.get("material_layers", [])))
    report = backend.export_assembly(assembly, path, materials)
    report["source_names"] = source_names
    report["edit_report"] = edit_report
    return report


def dispatch(request):
    op = request.get("op", request.get("operation"))
    if "op" in request and "operation" in request and request["op"] != request["operation"]:
        raise ValueError("Conflicting operation identifiers")
    options = {key: request[key] for key in ("linear_tolerance_mm", "angular_tolerance_rad") if key in request}
    if op == "import_step":
        return import_step(request["path"], **options)
    if op == "evaluate":
        return evaluate(request["solids"], include_mesh=request.get("include_mesh", True), **options)
    if op == "export_step":
        return export_step(request["solids"], request["path"], name=request.get("name", "edited_geometry"))
    if op == "collision":
        options = {key: request[key] for key in ("linear_tolerance_mm", "volume_tolerance_mm3") if key in request}
        if "pairs" in request:
            return dict(pairs=[dict(id=pair["id"], **collision(pair["first"], pair["second"], **options))
                               for pair in request["pairs"]])
        return collision(request["first"], request["second"], **options)
    if op == "validate_catalog":
        return validate_catalog(request["catalog"], optical_probes=request.get("optical_probes", []),
                                binding_report_sha256=request.get("binding_report_sha256"))
    raise ValueError("Unknown CAD operation: " + str(op))
