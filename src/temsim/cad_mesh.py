"""Tessellate analytic CAD construction for the native, OCC-free viewer."""
from dataclasses import replace
import math
from numbers import Integral
import numpy as np


def _winding_profile_segments(angular_segments):
    """Bound only the display wire section; retain the requested major circle."""
    return min(angular_segments, 12)


def _manifold(shape, count):
    import manifold3d as m
    kind = shape["kind"]
    if kind=='closed_wire_loop':
        return _closed_wire_manifold(shape,count)
    if kind == 'torus_arc':
        major,minor,z,start,end=(float(shape[name]) for name in (
            'major_radius_mm','minor_radius_mm','center_z_mm','start_angle_deg','end_angle_deg'))
        if not all(math.isfinite(v) for v in (major,minor,z,start,end)) or not 0<minor<major or not 0<end-start<360:
            raise ValueError('CAD wire arc requires finite positive radii and a span below 360 degrees')
        profile=_winding_profile_segments(count)
        intervals=max(1,math.ceil(count*(end-start)/360))
        tolerance=shape.get('display_chord_tolerance_mm')
        if tolerance is not None:
            tolerance=float(tolerance)
            if not math.isfinite(tolerance) or tolerance<=0:
                raise ValueError('Wire arc chord tolerance must be finite and positive')
            angle=2*math.acos(max(-1.,1-min(tolerance/major,1.)))
            intervals=max(intervals,math.ceil(math.radians(end-start)/angle))
        if intervals>16384:
            raise ValueError('Wire arc display requires excessive angular resolution; no turns were omitted')
        angles=np.deg2rad(np.linspace(start,end,intervals+1))
        cross=np.arange(profile)*2*math.pi/profile
        vertices=np.array([((major+minor*math.cos(u))*math.cos(a),
                            (major+minor*math.cos(u))*math.sin(a),z+minor*math.sin(u))
                           for a in angles for u in cross])
        faces=[]
        for i in range(len(angles)-1):
            for j in range(profile):
                a=i*profile+j;b=(i+1)*profile+j
                c=(i+1)*profile+(j+1)%profile;d=i*profile+(j+1)%profile
                faces.extend(((a,b,c),(a,c,d)))
        for j in range(1,profile-1):
            faces.append((0,j,j+1))
            first=(len(angles)-1)*profile
            faces.append((first,first+j+1,first+j))
        return m.Manifold(m.Mesh64(vertices,np.asarray(faces,dtype=np.uint64)))
    if kind == "torus":
        major, minor, z = (float(shape[name]) for name in (
            "major_radius_mm", "minor_radius_mm", "center_z_mm"))
        if not all(math.isfinite(v) for v in (major, minor, z)) or not 0 < minor < major:
            raise ValueError("CAD torus requires finite major radius > wire radius > 0")
        # Use the same circular section and azimuth grid as the insulation
        # holes, so the display partition has no artificial overlaps or gaps.
        return m.CrossSection.circle(minor, _winding_profile_segments(count)).translate((major, z)).revolve(count)
    if kind == "round_winding_insulation":
        start, end, inner, outer, wire = (float(shape[name]) for name in (
            "start_mm", "end_mm", "inner_mm", "outer_mm", "wire_radius_mm"))
        turns = np.asarray(shape["turns_rz_mm"], float).reshape((-1, 2))
        if (not all(math.isfinite(v) for v in (start, end, inner, outer, wire))
                or not start < end or not 0 < inner < outer or wire <= 0
                or not np.isfinite(turns).all()):
            raise ValueError("CAD winding insulation requires a finite positive annular pack")
        section = m.CrossSection([[(inner, start), (outer, start), (outer, end), (inner, end)]])
        circles = [m.CrossSection.circle(wire, _winding_profile_segments(count)).translate(tuple(point)) for point in turns]
        return m.CrossSection.batch_boolean([section, *circles], m.OpType.Subtract).revolve(count)
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
        tolerance=shape.get('display_chord_tolerance_mm')
        if tolerance is not None:
            tolerance=float(tolerance)
            if not math.isfinite(tolerance) or tolerance<=0:
                raise ValueError('Sector display chord tolerance must be finite and positive')
            angle=2*math.acos(max(-1.,1-min(tolerance/ro,1.)))
            n=max(n,math.ceil(math.radians(span)/angle)+1)
            if n>16384:
                raise ValueError('Sector display requires excessive angular resolution')
        angles = np.deg2rad(np.linspace(shape["angle_deg"] - span/2, shape["angle_deg"] + span/2, n))
        xy = [(ro*math.cos(t), ro*math.sin(t)) for t in angles]
        xy += [(ri*math.cos(t), ri*math.sin(t)) for t in angles[::-1]] if ri else [(0., 0.)]
        return m.CrossSection([xy]).extrude(b-a).translate((0., 0., a))
    if kind == "difference":
        solid = _manifold(shape["base"], count)
        if shape.get('batch_tools') and shape['tools']:
            result=m.Manifold.batch_boolean([solid,*(_manifold(tool,count) for tool in shape['tools'])],m.OpType.Subtract)
            result=result.simplify(0.)
            # Dense wire holes can contain exactly zero-area triangulation
            # remnants. Remove only those primitives, then retain the library's
            # existing merge tolerance to recover the same closed manifold.
            # No positive-area face, wire, analytic CAD or field tolerance is
            # discarded or relaxed. Verify material volume before accepting it.
            raw=result.to_mesh64();vertices=np.asarray(raw.vert_properties)[:,:3]
            faces=np.asarray(raw.tri_verts);triangles=vertices[faces]
            areas=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
            if np.any(areas==0):
                mesh=m.Mesh64(vertices.copy(),faces[areas>0].astype(np.uint64))
                mesh.merge();clean=m.Manifold(mesh)
                if clean.status()!=m.Error.NoError or abs(clean.volume()-result.volume())>128*math.ulp(result.volume()):
                    raise ValueError(f'Dense winding display triangulation cleanup did not preserve its closed material volume: {clean.status()}, {clean.volume()-result.volume():.12g} mm3')
                result=clean
            return result
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
    if kind == "affine":
        matrix = np.asarray(shape["matrix"], float)
        offset = np.asarray(shape.get("offset_mm", (0., 0., 0.)), float)
        if (matrix.shape != (3, 3) or offset.shape != (3,)
                or not np.isfinite(matrix).all() or not np.isfinite(offset).all()):
            raise ValueError("CAD affine placement requires a finite 3 by 3 matrix and 3D offset")
        determinant = float(np.linalg.det(matrix))
        # Match the STEP export contract: positive scaling/shear is permitted,
        # but a collapsed or reflected solid is not an editor placement.
        if not math.isfinite(determinant) or determinant <= 1e-15:
            raise ValueError("CAD affine transform must preserve orientation and nonzero volume")
        return _manifold(shape["shape"], count).transform(np.column_stack((matrix, offset)))
    if kind == "transform":
        rotation = np.asarray(shape["rotation"], float)
        offset = np.asarray(shape["offset_mm"], float)
        if rotation.shape != (3,3) or offset.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(offset).all():
            raise ValueError("CAD placement must be finite")
        return _manifold(shape["shape"], count).transform(np.column_stack((rotation, offset)))
    raise ValueError(f"Unsupported analytic CAD shape: {kind}")


def _closed_wire_manifold(shape,count):
    """Direct display sweep of every wire; analytic STEP keeps circular BRep."""
    import manifold3d as m
    radius=float(shape['wire_radius_mm'])
    if not math.isfinite(radius) or radius<=0:
        raise ValueError('Wire radius must be finite and positive')
    points=[];tangents=[];normals=[]
    if shape['path_family']=='rounded_rectangle':
        x,w,h,d,z=(float(shape[k]) for k in ('radial_x_mm','window_width_mm','window_height_mm','corner_offset_mm','center_z_mm'))
        q=max(2,math.ceil(count/4))
        for cy,cz,theta in ((w/2,-h/2,-math.pi/2),(w/2,h/2,0.),
                            (-w/2,h/2,math.pi/2),(-w/2,-h/2,math.pi)):
            for a in np.linspace(theta,theta+math.pi/2,q+1):
                points.append((x,cy+d*math.cos(a),z+cz+d*math.sin(a)))
                tangents.append((0.,-math.sin(a),math.cos(a)))
                normals.append((1.,0.,0.))
    elif shape['path_family']=='saddle':
        r,phi,lo,hi,tolerance=(float(shape[k]) for k in ('radius_mm','half_angle_rad','lower_z_mm','upper_z_mm','display_chord_tolerance_mm'))
        maxstep=2*math.acos(1-min(tolerance/r,1.))
        q=max(2,math.ceil(count*phi/math.pi),math.ceil(2*phi/maxstep))
        if q>16384:
            raise ValueError('Saddle display chord resolution exceeds its resource budget')
        # Top and bottom arcs include both endpoints. Tangents at their four
        # straight/arc junctions are bisectors for a closed, mitered tube.
        for z,angles,sign in ((hi,np.linspace(-phi,phi,q+1),1.),
                              (lo,np.linspace(phi,-phi,q+1),-1.)):
            for j,a in enumerate(angles):
                tangent=np.array([-sign*math.sin(a),sign*math.cos(a),0.])
                if j==0:tangent[2]=sign
                if j==q:tangent[2]=-sign
                tangent/=np.linalg.norm(tangent)
                points.append((r*math.cos(a),r*math.sin(a),z))
                tangents.append(tangent);normals.append((math.cos(a),math.sin(a),0.))
    else:
        raise ValueError('Unknown closed-wire display path')
    points=np.asarray(points);tangents=np.asarray(tangents);normals=np.asarray(normals)
    bins=np.cross(tangents,normals)
    profile=min(8,count)
    u=np.arange(profile)*2*math.pi/profile
    rings=points[:,None,:]+radius*(normals[:,None,:]*np.cos(u)[None,:,None]+bins[:,None,:]*np.sin(u)[None,:,None])
    angle=math.radians(float(shape['azimuth_deg']));c,s=math.cos(angle),math.sin(angle)
    rings=rings@np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])
    faces=[];n=len(points)
    for i in range(n):
        for j in range(profile):
            a=i*profile+j;b=((i+1)%n)*profile+j
            cc=((i+1)%n)*profile+(j+1)%profile;d=i*profile+(j+1)%profile
            faces.extend(((a,d,cc),(a,cc,b)))
    return m.Manifold(m.Mesh64(rings.reshape(-1,3),np.asarray(faces,dtype=np.uint64)))


def meshes_from_solids(solids, angular_segments=32):
    """Bound complete CAD display detail, with explicit transverse-wire LOD."""
    solids=tuple(solids)
    transverse=any(s.region.startswith(('winding_insulation_transverse_',
                                       'winding_insulation_multipole_')) for s in solids)
    candidates=tuple(dict.fromkeys((angular_segments,min(angular_segments,8),min(angular_segments,6)))) if transverse else (angular_segments,)
    last=None
    for resolution in candidates:
        try:
            meshes=_meshes_at_resolution(solids,resolution)
        except ValueError as error:
            if 'budget' not in str(error):raise
            last=error;continue
        if resolution!=angular_segments:
            family='Multipole' if any(s.region.startswith('winding_insulation_multipole_') for s in solids) else 'Transverse'
            note=(f' {family}-wire display resolution reduced from {angular_segments} to {resolution} '
                  'angular facets to retain every physical turn within the unchanged 4,000,000-triangle display budget. '
                  'Analytic STEP and magnetic-field/source tolerances are unchanged.')
            meshes=tuple(replace(mesh,description=mesh.description+note) for mesh in meshes)
        return meshes
    raise last


def _meshes_at_resolution(solids,angular_segments):
    from temsim.part_model_3d import _mesh
    import manifold3d as m
    solids = tuple(solids)
    _check_winding_mesh_budget(solids, angular_segments)
    meshes = []
    triangles=0
    check_actual=any(spec.region.startswith(('winding_insulation_transverse_',
                                            'winding_insulation_multipole_')) for spec in solids)
    for spec in solids:
        solid = _manifold(spec.shape, angular_segments)
        if solid.status() != m.Error.NoError or solid.is_empty() or solid.volume() <= 0:
            raise ValueError(f"{spec.key}/{spec.region}: CAD construction is not a closed positive solid")
        raw = solid.to_mesh64()
        triangles+=len(raw.tri_verts)
        if check_actual and triangles>4_000_000:
            raise ValueError('Complete winding display exceeds the actual 4,000,000-triangle budget; no turns were omitted')
        note = spec.note
        if spec.region.startswith(("winding_turn_", "winding_insulation_")):
            note += (f" Display tessellation: wire profile {_winding_profile_segments(angular_segments)} facets, "
                     f"azimuth {angular_segments} facets; every turn retained. Exact STEP keeps analytic circles.")
        mesh = _mesh(np.asarray(raw.vert_properties)[:, :3], np.asarray(raw.tri_verts),
                     spec.key, spec.region, spec.material, spec.status == "configured_geometry", note)
        meshes.append(replace(mesh, color=spec.color))
    return tuple(meshes)


def _check_winding_mesh_budget(solids, angular_segments):
    """Reject excessive detail explicitly; never decimate or omit turns."""
    turns, weighted = 0, 0
    for spec in solids:
        if not spec.region.startswith("winding_insulation_"):
            continue
        count=spec.parameters.get("winding_turn_count",0)
        factor=spec.parameters.get("display_complexity_factor",1)
        if (isinstance(count,bool) or not isinstance(count,Integral) or count<0
                or isinstance(factor,bool) or not isinstance(factor,Integral) or factor<1):
            raise ValueError("Winding display count and complexity must be nonnegative/positive integers")
        turns+=int(count);weighted+=int(count)*int(factor)
    estimate = 4 * weighted * _winding_profile_segments(angular_segments) * angular_segments
    if estimate > 4_000_000:
        raise ValueError(f"Detailed winding display requests about {estimate:,} triangles for {turns:,} turns; "
                         "the 4,000,000 triangle budget was exceeded. No turns were omitted; "
                         "use a lower angular display resolution or export the exact CAD separately.")


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
    fields=tuple(("parts",part["key"],key) for key in part
                 if key.endswith(("_mm", "_deg")) and key not in excluded)
    if runtime:
        fields+=tuple(("runtime",part["key"],key) for key in ("gap_m","centre_m") if key in runtime)
    metadata={f"material_face_{i}":{"label":f"{mesh.region} material face {i+1}","kind":"cad_material",
               "normal":plane[:3],"parameter_paths":fields} for i,plane in enumerate(planes)}
    return _decorate(mesh,groups,metadata)
