"""Tessellate analytic CAD construction for the native, OCC-free viewer."""
from dataclasses import replace
import math
import numpy as np


def _manifold(shape, count):
    import manifold3d as m
    kind = shape["kind"]
    if kind == "round_tip":
        from temsim.cad_geometry import revolve
        radius, length = float(shape["radius_mm"]), float(shape["length_mm"])
        angle, apex = math.radians(shape["cone_half_angle_deg"]), float(shape["apex_z_mm"])
        theta = np.linspace(0., math.pi/2-angle, 49)
        section = [(apex-2*radius*math.sin(t/2)**2,radius*math.sin(t)) for t in theta]
        join_depth = radius*(1-math.sin(angle))
        base_radius = radius*math.cos(angle)+(length-join_depth)*math.tan(angle)
        section += [(apex-length,base_radius),(apex-length,0.)]
        return _manifold(revolve(section),count)
    if kind == "box":
        size = np.asarray(shape["size_mm"], float)
        if size.shape != (3,) or np.any(size <= 0) or not np.isfinite(size).all():
            raise ValueError("CAD box dimensions must be finite and positive")
        return m.Manifold.cube(tuple(size), center=True).translate(tuple(shape["center_mm"]))
    if kind == "revolve":
        from temsim.part_model_3d import revolve_section
        mesh = revolve_section(shape["section_zr_mm"], key="cad", angular_segments=count)
        return m.Manifold(m.Mesh64(mesh.vertices.copy(), mesh.faces.astype(np.uint64)))
    if kind == "annular_sector":
        a, b = float(shape["start_mm"]), float(shape["end_mm"])
        ri, ro = float(shape["inner_mm"]), float(shape["outer_mm"])
        span = float(shape["span_deg"])
        if not 0 <= ri < ro or not a < b or not 0 < span <= 360:
            raise ValueError("CAD sector requires positive extent and an open radial interval")
        if span == 360:
            from temsim.cad_geometry import annulus
            return _manifold(annulus(a, b, ri, ro), count)
        n = max(3, math.ceil(count * span / 360) + 1)
        angles = np.deg2rad(np.linspace(shape["angle_deg"] - span/2, shape["angle_deg"] + span/2, n))
        xy = [(ro*math.cos(t), ro*math.sin(t)) for t in angles]
        xy += [(ri*math.cos(t), ri*math.sin(t)) for t in angles[::-1]] if ri else [(0., 0.)]
        return m.CrossSection([xy]).extrude(b-a).translate((0., 0., a))
    if kind == "difference":
        solid = _manifold(shape["base"], count)
        for tool in shape["tools"]:
            solid = solid - _manifold(tool, count)
        return solid
    if kind == "union":
        items = [_manifold(item, count) for item in shape["shapes"]]
        if not items:
            raise ValueError("A CAD union needs material")
        solid = items[0]
        for item in items[1:]:
            solid = solid + item
        return solid
    if kind == "transform":
        rotation = np.asarray(shape["rotation"], float)
        offset = np.asarray(shape["offset_mm"], float)
        if rotation.shape != (3,3) or offset.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(offset).all():
            raise ValueError("CAD placement must be finite")
        return _manifold(shape["shape"], count).transform(np.column_stack((rotation, offset)))
    raise ValueError(f"Unsupported analytic CAD shape: {kind}")


def meshes_from_solids(solids, angular_segments=32):
    from temsim.part_model_3d import _mesh
    import manifold3d as m
    meshes = []
    for spec in solids:
        solid = _manifold(spec.shape, angular_segments)
        if solid.status() != m.Error.NoError or solid.is_empty() or solid.volume() <= 0:
            raise ValueError(f"{spec.key}/{spec.region}: CAD construction is not a closed positive solid")
        raw = solid.to_mesh64()
        mesh = _mesh(np.asarray(raw.vert_properties)[:, :3], np.asarray(raw.tri_verts),
                     spec.key, spec.region, spec.material, spec.status == "configured_geometry", spec.note)
        meshes.append(replace(mesh, color=spec.color))
    return tuple(meshes)


def annotate_material_faces(mesh, part, runtime=None):
    """Keep selectable plane normals and actual source paths after CAD builds."""
    from temsim.part_model_features import _decorate, _normal_data
    centers,normals=_normal_data(mesh.vertices,mesh.faces)
    # Split by supporting plane: opposite faces and disconnected planar facets
    # must never share a normal or masquerade as the old zero-thickness guide.
    rows=np.column_stack((normals,np.einsum("ij,ij->i",centers,normals)))
    rounded=np.round(rows,10)
    planes,inverse=np.unique(rounded,axis=0,return_inverse=True)
    groups=np.array([f"material_face_{i}" for i in inverse],dtype=object)
    excluded={"length_mm","local_start_z_mm","local_center_z_mm","local_end_z_mm"} if part.get("branch_path_only") else set()
    fields=tuple(("parts",part["key"],key) for key in part if key.endswith("_mm") and key not in excluded)
    if runtime:
        fields+=tuple(("runtime",part["key"],key) for key in ("gap_m","centre_m") if key in runtime)
    metadata={f"material_face_{i}":{"label":f"{mesh.region} material face {i+1}","kind":"cad_material",
               "normal":plane[:3],"parameter_paths":fields} for i,plane in enumerate(planes)}
    return _decorate(mesh,groups,metadata)
