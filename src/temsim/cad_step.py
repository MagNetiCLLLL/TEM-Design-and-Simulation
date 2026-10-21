"""Optional analytic STEP export, with no simulator or OCC import at startup.

The catalog is plain JSON; lengths are millimetres. Material strings are
classification labels, not measured density, permeability or B-H data. No
triangle mesh is converted to STEP. CadQuery/OpenCascade is loaded on demand
from the separate CAD environment.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from time import perf_counter

SCHEMA = "temsim-analytic-step-v1"


def _cadquery():
    try:
        import cadquery as cq
    except ImportError as exc:
        raise RuntimeError("STEP export requires the optional CadQuery 2.6.1 CAD environment") from exc
    if cq.__version__ != "2.6.1":
        raise RuntimeError(f"This STEP adapter is validated for CadQuery 2.6.1, found {cq.__version__}")
    return cq


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _vector(values, length, label):
    result = tuple(_number(value, label) for value in values)
    if len(result) != length:
        raise ValueError(f"{label} requires {length} values")
    return result


def _shape_metrics(shape):
    solids = shape.Solids()
    if not shape.isValid() or not solids:
        raise ValueError("Analytic B-rep is invalid or contains no solid")
    volumes = [shape_volume(solid) for solid in solids]
    if any(not math.isfinite(value) or value <= 0 for value in volumes):
        raise ValueError("Every material solid must have positive finite volume")
    box = shape.BoundingBox()
    bounds = [box.xmin, box.ymin, box.zmin, box.xmax, box.ymax, box.zmax]
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("Nonfinite B-rep bounds")
    return dict(valid=True, solid_count=len(solids), volume_mm3=sum(volumes), bounds_mm=bounds)


def shape_volume(shape):
    """Adaptive exact-surface integration, including rational spline spans.

    CQ's default fixed integration can mismeasure an affinely transformed
    ellipse by almost one percent. Never use the display tessellation here.
    """
    if not shape.Solids():
        return 0.
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    properties = GProp_GProps()
    error = BRepGProp.VolumePropertiesGK_s(shape.wrapped, properties, 1e-10, True, True)
    if error < 0 or not math.isfinite(properties.Mass()):
        raise RuntimeError("Exact BRep volume integration did not converge")
    return properties.Mass()


def build_shape(spec, *, allow_empty=False):
    """Build an exact BRep recipe; tessellated meshes are never accepted."""
    cq = _cadquery()
    if not isinstance(spec, dict):
        raise ValueError("Shape specification must be a JSON object")
    kind = spec.get("kind")
    if kind == "brep":
        try:
            data = base64.b64decode(spec["brep_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("BRep source is not valid base64") from exc
        if not data or hashlib.sha256(data).hexdigest() != spec.get("sha256"):
            raise ValueError("BRep source SHA256 mismatch")
        shape = cq.Shape.importBrep(BytesIO(data))
    elif kind == "revolve":
        section = [_vector(point, 2, "section Z/radius") for point in spec["section_zr_mm"]]
        if len(section) < 3 or any(radius < 0 for _, radius in section):
            raise ValueError("Revolved sections need at least three points and nonnegative radii")
        if section[0] == section[-1]:
            section.pop()
        wire = cq.Wire.makePolygon([cq.Vector(radius, 0., z) for z, radius in section], close=True)
        shape = cq.Solid.revolve(wire, [], 360., (0., 0., 0.), (0., 0., 1.))
    elif kind == "round_tip":
        radius = _number(spec["radius_mm"], "Tip radius")
        length = _number(spec["length_mm"], "Tip length")
        angle = math.radians(_number(spec["cone_half_angle_deg"], "Cone half angle"))
        apex = _number(spec.get("apex_z_mm", 0.), "Tip apex Z")
        if not radius > 0 or not length > radius or not 0 < angle < math.pi/2:
            raise ValueError("Tip needs positive radius, length > radius and half angle in (0, 90) degrees")
        # Sphere centre at z=-R; tangent cone extends upstream, as in
        # tip_surface.TipGeometry.radius_m. Keep the arc analytic, not polygonal.
        theta = math.pi/2-angle
        join_r, join_z = radius*math.sin(theta), radius*(math.cos(theta)-1.)
        mid = cq.Vector(radius*math.sin(theta/2), 0., radius*(math.cos(theta/2)-1.))
        tip, join = cq.Vector(0., 0., 0.), cq.Vector(join_r, 0., join_z)
        base = cq.Vector(join_r+(length+join_z)*math.tan(angle), 0., -length)
        axis = cq.Vector(0., 0., -length)
        wire = cq.Wire.assembleEdges([cq.Edge.makeThreePointArc(tip, mid, join),
            cq.Edge.makeLine(join, base), cq.Edge.makeLine(base, axis), cq.Edge.makeLine(axis, tip)])
        shape = cq.Solid.revolve(wire, [], 360., (0., 0., 0.), (0., 0., 1.)).translate((0., 0., apex))
    elif kind == "box":
        centre = _vector(spec["center_mm"], 3, "Box centre")
        size = _vector(spec["size_mm"], 3, "Box size")
        if min(size) <= 0:
            raise ValueError("Box sizes must be positive")
        corner = tuple(c-s/2 for c, s in zip(centre, size))
        shape = cq.Solid.makeBox(*size, cq.Vector(*corner))
    elif kind in ("cylinder", "elliptic_cylinder"):
        centre = _vector(spec.get("center_mm", (0., 0., 0.)), 3, "Cylinder centre")
        length = _number(spec["length_mm"], "Cylinder length")
        if kind == "cylinder":
            rx = ry = _number(spec["radius_mm"], "Cylinder radius")
        else:
            rx = _number(spec["width_mm"], "Ellipse width") / 2.
            ry = _number(spec["height_mm"], "Ellipse height") / 2.
        if min(rx, ry, length) <= 0:
            raise ValueError("Cylinder dimensions must be positive")
        # Ellipse axes are diameters in the native feature editor contract.
        shape = (cq.Solid.makeCylinder(rx, length) if kind == "cylinder" else
                 cq.Workplane("XY").ellipse(rx, ry).extrude(length).val())
        shape = shape.translate((centre[0], centre[1], centre[2]-length/2.))
    elif kind == "annular_sector":
        start, end, inner, outer, angle, span = (
            _number(spec[key], key) for key in
            ("start_mm", "end_mm", "inner_mm", "outer_mm", "angle_deg", "span_deg"))
        if not end > start or not 0 <= inner < outer or not 0 < span <= 360:
            raise ValueError("Annular sector dimensions or angular span are invalid")
        section = [(inner, 0., start), (outer, 0., start), (outer, 0., end), (inner, 0., end)]
        wire = cq.Wire.makePolygon([cq.Vector(*point) for point in section], close=True)
        shape = cq.Solid.revolve(wire, [], span, (0., 0., 0.), (0., 0., 1.))
        # angle_deg is the sector centre angle, matching the native model.
        shape = shape.rotate((0., 0., 0.), (0., 0., 1.), angle-span/2)
    elif kind in ("difference", "strict_difference"):
        shape = build_shape(spec["base"])
        tools = [build_shape(item) for item in spec["tools"]]
        before = shape_volume(shape)
        if tools:
            shape = shape.cut(*tools)
        if kind == "strict_difference":
            threshold = max(1e-12, abs(before)*1e-10)
            if before-shape_volume(shape) <= threshold:
                raise ValueError("Cut does not remove material from the selected solid")
    elif kind == "union":
        children = [build_shape(item) for item in spec["shapes"]]
        if not children:
            raise ValueError("A union requires at least one material shape")
        shape = children[0].fuse(*children[1:]) if len(children) > 1 else children[0]
    elif kind == "affine":
        rows = tuple(_vector(row, 3, "Affine row") for row in spec["matrix"])
        offset = _vector(spec.get("offset_mm", (0., 0., 0.)), 3, "Affine offset")
        if len(rows) != 3:
            raise ValueError("Affine matrix must be 3 by 3")
        determinant = sum(rows[0][j]*(rows[1][(j+1)%3]*rows[2][(j+2)%3]
                                    -rows[1][(j+2)%3]*rows[2][(j+1)%3]) for j in range(3))
        if determinant <= 1e-15:
            raise ValueError("Affine transform must preserve orientation and nonzero volume")
        matrix = cq.Matrix([list(row)+[d] for row, d in zip(rows, offset)] + [[0., 0., 0., 1.]])
        child = build_shape(spec["shape"], allow_empty=allow_empty)
        shape = child.transformGeometry(matrix) if child.Solids() else child
    elif kind == "transform":
        rows = tuple(_vector(row, 3, "Rotation row") for row in spec["rotation"])
        offset = _vector(spec["offset_mm"], 3, "Rigid offset")
        if len(rows) != 3:
            raise ValueError("Rigid rotation must be 3 by 3")
        error = max(abs(sum(rows[i][k]*rows[j][k] for k in range(3))-(i == j))
                    for i in range(3) for j in range(3))
        determinant = sum(rows[0][j]*(rows[1][(j+1)%3]*rows[2][(j+2)%3]
                                    -rows[1][(j+2)%3]*rows[2][(j+1)%3]) for j in range(3))
        if error > 1e-10 or abs(determinant-1.) > 1e-10:
            raise ValueError("Transforms require a proper rotation; scale, shear and reflection are unsupported")
        from OCP.gp import gp_Trsf
        transform = gp_Trsf()
        transform.SetValues(*(value for row, d in zip(rows, offset) for value in (*row, d)))
        child = build_shape(spec["shape"], allow_empty=allow_empty)
        shape = child.moved(cq.Location(transform)) if child.Solids() else child
    else:
        raise ValueError(f"Unknown analytic CAD primitive: {kind!r}")
    if not (allow_empty and kind != "strict_difference" and not shape.Solids() and shape.isValid()):
        _shape_metrics(shape)
    return shape


def solid_identity(row):
    """Keep separately placed material segments distinct within one module."""
    identity = row.get("instance_key", row["key"] + "/" + row["region"])
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("A solid instance requires a nonempty identity")
    return identity


def build_solid_rows(specs):
    """Validate part-wide cuts before allowing deletion of an individual leaf."""
    groups, shapes, removed, edits = {}, {}, [], []
    for row in specs:
        groups.setdefault(str(row["key"]), []).append(row)
    for key, rows in groups.items():
        editing = any("editor_tools" in row.get("parameters", {}) for row in rows)
        if editing:
            if any("editor_original_shape" not in row.get("parameters", {}) or
                   "editor_tools_global" not in row.get("parameters", {}) for row in rows):
                raise ValueError(f"{key}: incomplete original/tool geometry for exact editing")
            tools_spec = rows[0]["parameters"]["editor_tools_global"]
            canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
            if any(canonical(row["parameters"]["editor_tools_global"]) != canonical(tools_spec) for row in rows):
                raise ValueError(f"{key}: material leaves disagree on cutting tools")
            current = [build_shape(row["parameters"]["editor_original_shape"]) for row in rows]
            initial = sum(shape_volume(shape) for shape in current)
            losses = []
            for tool_spec in tools_spec:
                tool = build_shape(tool_spec)
                before = sum(shape_volume(shape) for shape in current)
                current = [shape.cut(tool) if shape.Solids() else shape for shape in current]
                if any(not shape.isValid() for shape in current):
                    raise ValueError(f"{key}: invalid edited BRep")
                after = sum(shape_volume(shape) for shape in current)
                if before-after <= max(1e-12, initial*1e-10):
                    raise ValueError(f"{key}: cutting tool does not remove material")
                losses.append(before-after)
            remaining = sum(shape_volume(shape) for shape in current)
            if remaining <= max(1e-12, initial*1e-10):
                raise ValueError(f"{key}: editing removes the entire part")
            edits.append(dict(key=key, initial_volume_mm3=initial, remaining_volume_mm3=remaining,
                              removed_volume_per_tool_mm3=losses))
        for index, row in enumerate(rows):
            identity = solid_identity(row)
            if identity in shapes or identity in removed:
                raise ValueError("Duplicate solid ID: " + identity)
            shape = build_shape(row["shape"], allow_empty=editing)
            if editing:
                expected = current[index]
                # Check the final recipe against the independently replayed
                # global tools, not just a metadata marker permitting deletion.
                delta = sum(shape_volume(s) for s in shape.cut(expected).Solids()) if shape.Solids() else 0.
                delta += sum(shape_volume(s) for s in expected.cut(shape).Solids()) if expected.Solids() else 0.
                if delta > max(1e-10, initial*1e-8):
                    raise ValueError(f"{identity}: final recipe disagrees with declared edit tools")
            if not shape.Solids():
                removed.append(identity)
            else:
                shapes[identity] = shape
    return shapes, dict(removed_leaf_ids=removed, parts=edits)


def shape_brep_spec(shape):
    """Portable exact source, independent of any display tessellation."""
    _shape_metrics(shape)
    stream = BytesIO()
    if not shape.exportBrep(stream):
        raise RuntimeError("BRep serialization failed")
    data = stream.getvalue()
    return dict(kind="brep", brep_base64=base64.b64encode(data).decode("ascii"),
                sha256=hashlib.sha256(data).hexdigest())


def _label(value):
    value = str(value).strip()
    if not value or "/" in value or "\\" in value or any(ord(char) < 32 for char in value):
        raise ValueError("Assembly names must be nonempty single-level labels")
    return value


def _filename(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", _label(value)).strip(". _")
    if not text or text.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}:
        raise ValueError("Unsafe STEP filename")
    return text


def _normalise_solid(value):
    spec = asdict(value) if is_dataclass(value) else dict(value)
    for field in ("key", "region", "material"):
        spec[field] = _label(spec[field])
    color = spec.get("color", getattr(value, "color", (.64, .70, .75, 1.)))
    color = _vector(color, 4, "RGBA colour")
    if min(color) < 0 or max(color) > 1:
        raise ValueError("RGBA channels must lie within [0, 1]")
    spec["color"] = color
    return spec


def _named_labels(doc):
    from OCP.TDF import TDF_LabelSequence
    from OCP.TDataStd import TDataStd_Name
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    sequence = TDF_LabelSequence()
    XCAFDoc_DocumentTool.ShapeTool_s(doc.Main()).GetShapes(sequence)
    for index in range(1, sequence.Length()+1):
        label = sequence.Value(index)
        attribute = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            yield attribute.Get().ToExtString(), label


def _write_step(assembly, path, materials):
    from cadquery.occ_impl.assembly import toCAF
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import (XCAFDoc_DocumentTool, XCAFDoc_ColorGen,
                            XCAFDoc_ColorSurf, XCAFDoc_ColorCurv)
    from OCP.Quantity import Quantity_ColorRGBA
    from OCP.XSControl import XSControl_WorkSession
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_StepModelType
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    _, doc = toCAF(assembly, True)
    layers = XCAFDoc_DocumentTool.LayerTool_s(doc.Main())
    colors = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    leaf_colors = {name.rsplit("/", 1)[-1]: color
                   for _, name, _, color in assembly if color is not None}
    assigned = set()
    for name, label in _named_labels(doc):
        if name in materials:
            layer = TCollection_ExtendedString("material:"+materials[name])
            layers.SetLayer(label, layer)
            # Boolean results can contain nested compounds. STEP omits an
            # outer compound's layer/color when it flattens that compound;
            # carry both onto its actual solids without altering geometry.
            cq = _cadquery()
            shape = cq.Shape.cast(shape_tool.GetShape_s(label))
            color = leaf_colors.get(name)
            for solid in shape.Solids():
                sublabel = shape_tool.AddSubShape(label, solid.wrapped)
                if not sublabel.IsNull():
                    layers.SetLayer(sublabel, layer)
                    # addSubshape colors already belong to this solid. A leaf
                    # default supplies only missing color; it must not erase a
                    # saved generic, surface or curve override.
                    explicit_color = any(colors.GetColor_s(sublabel, kind, Quantity_ColorRGBA())
                                         for kind in (XCAFDoc_ColorGen, XCAFDoc_ColorSurf,
                                                      XCAFDoc_ColorCurv))
                    if color is not None and not explicit_color:
                        colors.SetColor(sublabel, color.wrapped, XCAFDoc_ColorGen)
            assigned.add(name)
    if assigned != set(materials):
        raise ValueError("Not every CAD leaf received its material classification")
    writer = STEPCAFControl_Writer(XSControl_WorkSession(), False)
    writer.SetColorMode(True)
    writer.SetLayerMode(True)
    writer.SetNameMode(True)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
    Interface_Static.SetIVal_s("write.stepcaf.subshapes.name", 1)
    if not writer.Transfer(doc, STEPControl_StepModelType.STEPControl_AsIs):
        raise RuntimeError("STEP transfer did not complete")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError("STEP writer did not complete")


def _read_step(path):
    cq = _cadquery()
    from cadquery.occ_impl.importers.assembly import _importDoc
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.TDF import TDF_ChildIterator
    reader = STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetLayerMode(True)
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_RetDone:
        raise RuntimeError("Exported STEP could not be read")
    reader.ChangeReader().SetSystemLengthUnit(1.)
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    if not reader.Transfer(doc):
        raise RuntimeError("Exported STEP could not be transferred for validation")
    assembly = cq.Assembly()
    _importDoc(doc, assembly)
    # STEP flattens nested compounds and may attach their uniform color only
    # to the constituent solids. Recover a leaf color only when every actual
    # solid has that same recorded color; never infer it from an expected input
    # or hide missing/mixed material colors.
    for item in assembly.objects.values():
        if item.obj is None or item.color is not None:
            continue
        solids = [solid for shape in item.shapes for solid in shape.Solids()]
        recorded = [item._subshape_colors.get(solid) for solid in solids]
        if len(recorded) > 1 and all(color is not None for color in recorded):
            reference = recorded[0].toTuple()
            if all(max(abs(a-b) for a, b in zip(reference, color.toTuple())) < 1e-12
                   for color in recorded[1:]):
                item.color = recorded[0]
    tool = XCAFDoc_DocumentTool.LayerTool_s(doc.Main())
    layers = {}
    for name, label in _named_labels(doc):
        descendants = [label]
        iterator = TDF_ChildIterator(label, True)
        while iterator.More():
            descendants.append(iterator.Value())
            iterator.Next()
        found = set(layers.get(name, []))
        for child in descendants:
            values = tool.GetLayers(child)
            if values is not None:
                found.update(values.Value(i).ToExtString() for i in range(1, values.Length()+1))
        layers[name] = sorted(found)
    return assembly, layers


def _leaves(assembly):
    return {path: dict(shape=shape.moved(location), color=color.toTuple() if color else None)
            for shape, path, location, color in assembly}


def export_assembly(assembly, path, material_by_leaf):
    """Export and re-import every named leaf; fail on geometric/metadata loss."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = _leaves(assembly)
    if not before:
        raise ValueError("An exported assembly must contain material solids")
    expected = {name: _shape_metrics(row["shape"]) for name, row in before.items()}
    temporary = path.with_name(path.stem+".pending.step")
    _write_step(assembly, temporary, material_by_leaf)
    restored, material_layers = _read_step(temporary)
    after = _leaves(restored)
    if set(before) != set(after):
        raise ValueError(f"STEP hierarchy/name mismatch: expected {sorted(before)}, got {sorted(after)}")
    records = []
    for name, row in before.items():
        actual = _shape_metrics(after[name]["shape"])
        original = expected[name]
        if actual["solid_count"] != original["solid_count"]:
            raise ValueError(f"STEP changed material solid count for {name}")
        volume_error = abs(actual["volume_mm3"]-original["volume_mm3"])
        bound_error = max(abs(a-b) for a, b in zip(actual["bounds_mm"], original["bounds_mm"]))
        if volume_error > max(1e-12, 1e-7*original["volume_mm3"]) or bound_error > 1e-5:
            raise ValueError(f"STEP round-trip changed volume/bounds for {name}: {volume_error}, {bound_error}")
        old_color, new_color = row["color"], after[name]["color"]
        if old_color is not None and (new_color is None or max(abs(a-b) for a, b in zip(old_color, new_color)) > 2e-6):
            raise ValueError(f"STEP changed RGBA colour for {name}")
        leaf = name.rsplit("/", 1)[-1]
        material = material_by_leaf[leaf]
        if "material:"+material not in material_layers.get(leaf, []):
            raise ValueError(f"STEP lost material classification for {name}")
        records.append(dict(path=name, material=material, color_rgba=new_color, **actual,
                            volume_difference_mm3=volume_error, maximum_bound_difference_mm=bound_error))
    data = temporary.read_bytes()
    if b"SI_UNIT(.MILLI.,.METRE.)" not in data:
        raise ValueError("STEP did not declare millimetre units")
    temporary.replace(path)
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data),
                units="mm", leaf_count=len(records), records=records,
                volume_mm3=sum(item["volume_mm3"] for item in records),
                material_semantics="Classification layers; density and electromagnetic material constants are unspecified")


def export_catalog(catalog, output_directory, *, progress=None, manifest_name="step-manifest.json"):
    """Export module and optical-unit assemblies from a simulator-free catalog."""
    if catalog.get("units", "mm") != "mm":
        raise ValueError("CAD catalogs must explicitly use millimetres")
    cq = _cadquery()
    started = perf_counter()
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if Path(manifest_name).name != manifest_name or not manifest_name.endswith(".json"):
        raise ValueError("Manifest name must be a local JSON filename")
    report = dict(schema=SCHEMA, cadquery_version=cq.__version__, units="mm", status="running",
        geometry="Analytic B-rep from canonical SolidSpec; no triangle-to-STEP conversion",
        scope="Geometric validity and STEP metadata/volume/bounds round-trip; not OEM dimensions or manufacturing qualification",
        catalog_sha256=hashlib.sha256(json.dumps(catalog, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        modules=[], exports=[])
    files = set()

    def emit(assembly, filename, materials):
        if filename.casefold() in files:
            raise ValueError("Catalog produces duplicate STEP filenames")
        files.add(filename.casefold())
        if progress:
            progress(filename)
        record = export_assembly(assembly, directory/filename, materials)
        report["exports"].append(record)

    try:
        for module in catalog["modules"]:
            module_key = _label(module["key"])
            root = cq.Assembly(name=module_key)
            specs = [_normalise_solid(row) for row in module["solids"]]
            ids = [solid_identity(row) for row in specs]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate solid IDs in {module_key}")
            shapes, edit_report = build_solid_rows(specs)
            removed = set(edit_report["removed_leaf_ids"])
            specs = [spec for spec in specs if solid_identity(spec) not in removed]
            ids = [solid_identity(row) for row in specs]
            spec_by_id = dict(zip(ids, specs))
            leaf_names = {key: _label(key.replace("/", "__")) for key in ids}
            if len(set(leaf_names.values())) != len(ids):
                raise ValueError("Solid naming collision")
            units = [dict(unit, solid_ids=[key for key in unit["solid_ids"] if key not in removed])
                     for unit in module.get("units", [])]
            units = [unit for unit in units if unit["solid_ids"]]
            assigned = [key for unit in units for key in unit["solid_ids"]]
            if len(assigned) != len(set(assigned)) or not set(assigned).issubset(ids):
                raise ValueError("Optical units must reference existing solids exactly once")
            unassigned = [key for key in ids if key not in assigned]
            if unassigned:
                units.append(dict(key="module_hardware", name="Other module hardware", solid_ids=unassigned))
            materials = {}
            for unit in units:
                unit_key = _label(unit["key"])
                child = cq.Assembly(name=unit_key)
                unit_materials = {}
                for key in unit["solid_ids"]:
                    spec = spec_by_id[key]
                    name = leaf_names[key]
                    child.add(shapes[key], name=name, color=cq.Color(*spec["color"]), metadata=spec)
                    unit_materials[name] = spec["material"]
                if not unit_materials:
                    raise ValueError(f"Optical unit {unit_key} has no material solids")
                root.add(child)
                materials.update(unit_materials)
                emit(child, _filename(module_key)+"__"+_filename(unit_key)+".step", unit_materials)
            emit(root, _filename(module_key)+".step", materials)
            report["modules"].append(dict(key=module_key, name=module.get("name", module_key),
                solid_count=len(specs), units=units, solids=specs, edit_report=edit_report))
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        raise
    finally:
        report["elapsed_s"] = perf_counter()-started
        (directory/manifest_name).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report
