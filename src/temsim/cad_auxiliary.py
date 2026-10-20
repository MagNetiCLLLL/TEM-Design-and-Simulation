"""Gun, aperture and filter material constructions in millimetres.

Only the configured electrode surfaces are physical solver boundaries. Added
insulators, carriers and pole thicknesses are documented CAD reconstruction.
They do not replace or tune the electron-optical model.
"""
from dataclasses import replace
import math
import numpy as np
from temsim.cad_geometry import SolidSpec, annulus, box, difference, sector, transformed

GUN_SOURCE = "https://patents.google.com/patent/US9548182B2/en"
WIEN_SOURCE = "https://patents.google.com/patent/US6844548B2/en"
FILTER_SOURCE = "https://patents.google.com/patent/US9966219B2/en"
GUN_ELECTRODE_KEYS = frozenset({
    "feg_extractor", "feg_electrostatic_lens",
    "thermionic_gun_lens", "thermionic_wehnelt",
})
_OUTER_RADIUS_FIELDS = (("mechanical_outer_diameter_mm", .5),
                       ("mechanical_outer_radius_mm", 1.),
                       ("outer_diameter_mm", .5), ("outer_width_mm", .5))
_INNER_RADIUS_FIELDS = (("mechanical_inner_diameter_mm", .5),
                       ("mechanical_bore_diameter_mm", .5),
                       ("mechanical_clear_bore_diameter_mm", .5),
                       ("mechanical_bore_radius_mm", 1.),
                       ("bore_diameter_mm", .5), ("clear_bore_diameter_mm", .5),
                       ("inner_diameter_mm", .5))


def _radii(part):
    from temsim.part_model_3d import _radius
    outer = _radius(part, _OUTER_RADIUS_FIELDS)
    inner = _radius(part, _INNER_RADIUS_FIELDS)
    return inner, outer


def gun_electrode_solids(part):
    """Configured open gun electrodes, shared by native display and CAD."""
    if part.get("key") not in GUN_ELECTRODE_KEYS:
        return None
    from temsim.part_model_3d import _extent
    start, end = _extent(part)
    inner, outer = _radii(part)
    if inner is None or outer is None or not (start < end and 0 < inner < outer):
        raise ValueError("Gun electrode requires positive length and 0 < bore < outer radius")
    return (SolidSpec(
        part["key"], "body", "electrode", annulus(start, end, inner, outer),
        (GUN_SOURCE,), "configured_geometry",
        "Configured open annular electrode: upstream/downstream faces and beam bore "
        "use the saved dimensions. OEM nose and support details are not established.",
        parameters={"upstream_face_z_mm": start, "downstream_face_z_mm": end,
                    "bore_radius_mm": inner, "outer_radius_mm": outer},
    ),)


def annotate_gun_electrode_faces(mesh, part):
    """Keep four physical boundaries selectable with their actual dimension paths."""
    if part.get("key") not in GUN_ELECTRODE_KEYS:
        return mesh
    from temsim.part_model_features import _decorate, _normal_data, _paths
    centers, normals = _normal_data(mesh.vertices, mesh.faces)
    groups = np.where(
        np.abs(normals[:, 2]) > 1 - 1e-9,
        np.where(normals[:, 2] > 0, "electrode_downstream", "electrode_upstream"),
        np.where(np.einsum("ij,ij->i", centers[:, :2], normals[:, :2]) < 0,
                 "electrode_bore", "electrode_outer"),
    )
    # Only the first supported radius field is authoritative, as in _radii.
    inner = _paths(part, (next(name for name, _ in _INNER_RADIUS_FIELDS if name in part),))
    outer = _paths(part, (next(name for name, _ in _OUTER_RADIUS_FIELDS if name in part),))
    extent = ()
    if part["local_start_z_mm"] == part["local_end_z_mm"]:
        extent = _paths(part, (next(name for name in
                         ("housing_length_mm", "electrode_length_mm", "plate_thickness_mm")
                         if name in part and part[name] > 0),))
    upstream = (*_paths(part, ("local_start_z_mm",)), *extent, *inner, *outer)
    downstream = (*_paths(part, ("local_end_z_mm",)), *extent, *inner, *outer)
    surfaces = {
        "electrode_upstream": {"label": "Upstream electrode face", "kind": "cap",
                              "parameter_paths": upstream},
        "electrode_downstream": {"label": "Downstream electrode face", "kind": "cap",
                                "parameter_paths": downstream},
        "electrode_bore": {"label": "Open beam bore", "kind": "inner",
                          "parameter_paths": inner},
        "electrode_outer": {"label": "Outer electrode surface", "kind": "outer",
                           "parameter_paths": outer},
    }
    return _decorate(mesh, groups, surfaces, normals=normals)


def accelerator_solids(part):
    if "stage_centers_z_mm" not in part:
        return None
    from temsim.part_model_3d import _extent
    start, end = _extent(part)
    inner, outer = _radii(part)
    if "electrode_thickness_mm" not in part:
        return (SolidSpec(part["key"], "body", "unspecified",
            annulus(start, end, inner, outer), (GUN_SOURCE,), "configured_envelope",
            "Configured accelerator envelope. Electrode thickness is not supplied; no finite electrodes or supports are invented."),)
    thickness = float(part["electrode_thickness_mm"])
    centres = [float(c) for c in part["stage_centers_z_mm"]]
    if not centres or inner is None or outer is None or not 0 < inner < outer or thickness <= 0:
        raise ValueError("Accelerator CAD requires positive configured rings")
    result, previous = [], start
    ceramic_inner = inner + .84 * (outer-inner)
    for index, center in enumerate(centres):
        lo, hi = center-thickness/2, center+thickness/2
        if lo < previous or hi > end:
            raise ValueError("Accelerator electrodes must fit the retained envelope")
        if index > 0 and lo > previous:
            result.append(SolidSpec(part["key"], f"insulator_{index+1}", "insulator",
                annulus(previous, lo, ceramic_inner, outer), (GUN_SOURCE,),
                parameters={"inner_radius_mm":ceramic_inner,"outer_radius_mm":outer},
                note="Insulating inter-stage ring; topology from US9548182 Fig.2, radial thickness is a reconstruction."))
        explicit_thickness="electrode_thickness_mm" in part
        result.append(SolidSpec(part["key"], f"electrode_{index+1}", "electrode",
            annulus(lo, hi, inner, outer), (GUN_SOURCE,),
            "configured_geometry" if explicit_thickness else "engineering_reconstruction",
            "Annular accelerator electrode: configured stage centre, bore and outer radius. " +
            ("Thickness is configured." if explicit_thickness else "Thickness is an explicit 1 mm reconstruction; not supplied in this file."),
            parameters={"electrode_thickness_mm":thickness,
                        "thickness_source":"configured" if explicit_thickness else "reconstruction_default"}))
        previous = hi
    return tuple(result)


def aperture_solids(part, runtime=None, aperture_index=0):
    from temsim.cad_aperture_support import aperture_support_solids
    supported = aperture_support_solids(part, runtime, aperture_index)
    if supported is not None:
        return supported
    from temsim.part_model_apertures import is_strip_aperture, _opening, plate_thickness_field
    if not is_strip_aperture(part):
        return None
    thickness = float(part[plate_thickness_field(part)])
    center = float(part.get("optical_reference_local_z_mm", part["local_center_z_mm"]))
    diameter, paths, (dx,dy) = _opening(part, runtime, aperture_index)
    vacuum = .5*float(part.get("vacuum_inner_diameter_mm",0.))
    carrier, envelope = _radii(part)
    carrier = carrier or 0.
    envelope = max(envelope or 0., vacuum+1.)
    margin = max(.75,.15*max(vacuum,1.))
    reach = max(vacuum+margin,carrier+margin)
    joint = max(reach,min(.60*envelope,vacuum+12.))
    radius = .5*(diameter or 0.)
    width = max(reach,abs(dx)+radius+margin)
    lower,upper = min(-reach,dy-radius-margin),max(joint,dy+radius+margin)
    base = box((0.,.5*(lower+upper),center),(2*width,upper-lower,thickness))
    if radius:
        tool = transformed(annulus(center-thickness,center+thickness,0.,radius),offset=(dx,dy,0.))
        base = difference(base,tool)
    note = "Configured plate thickness and working hole; transverse strip outline follows the existing schematic reach rule."
    if diameter is None:
        note += " Working hole is not specified in this saved file."
    return (SolidSpec(part["key"],"body",str(part.get("material_class",part.get("aperture_plate_material","electrode"))),base,
                     (), note=note, parameters={"plate_thickness_mm":thickness,"working_diameter_mm":diameter,
                                                "strip_half_width_mm":width,"strip_y_bounds_mm":[lower,upper]}),)


def tip_solids(part, runtime=None):
    from temsim.optics.electron_gun.tip_assembly import is_tip_part, model_from_part, tip_apex_z_mm
    if not is_tip_part(part):
        return None
    from temsim.optics.electron_gun.tip_surface import TipSurfaceModel
    model = model_from_part(part)
    if runtime and runtime.get("tip_surface_model") is not None:
        model = TipSurfaceModel.from_dict(runtime["tip_surface_model"])
    g = model.geometry
    return (SolidSpec(part["key"],"body",g.material,
        {"kind":"round_tip","radius_mm":g.apex_radius_nm*1e-6,
         "cone_half_angle_deg":g.cone_half_angle_deg,"length_mm":g.shank_length_um*.001,
         "apex_z_mm":tip_apex_z_mm(part)}, (), "configured_geometry",
         "Configured physical spherical tip and tangent cone; emission FWHM is not the metal radius."),)


def filter_solids(part, by_key, runtime=None, aperture_index=0):
    from temsim.energy_filter_model_3d import (supports_energy_filter_part, energy_filter_frame_mm,
        energy_filter_origin_mm, FILTER_TO_COLUMN_ROTATION, _sector, slit_opening_mm)
    if not supports_energy_filter_part(part):
        return None
    key,profile = part["key"],part.get("mechanical_profile","")
    if key == "energy_filter_entrance_aperture":
        rotation=np.asarray(FILTER_TO_COLUMN_ROTATION).T
        offset=rotation@np.array((0.,0.,-energy_filter_origin_mm(by_key)))
        return tuple(replace(s,shape=transformed(s.shape,rotation=rotation.tolist(),offset=offset))
                     for s in aperture_solids(part,runtime,aperture_index))
    if not math.isclose(float(part["local_center_z_mm"]),energy_filter_origin_mm(by_key),rel_tol=0.,abs_tol=1e-8):
        raise ValueError("Branch-path component must share the Energy Filter entrance datum")
    if profile=="tapered_sector_prism":
        _,radius,angle,origin,_,_=_sector(by_key)
        gap,half=float(part["pole_gap_mm"]),float(part["radial_clear_half_width_mm"])
        if not 0 < half < radius:
            raise ValueError("Prism radial channel must not cross its curvature centre")
        if gap <= 0:
            raise ValueError("Prism pole gap must be positive")
        thickness=float(part.get("cad_pole_thickness_mm",gap*.4))
        # Sector XY rotated into branch XZ, its cylinder axis becomes -Y.
        rotation=[[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]]
        offset=origin+np.array((0.,0.,-radius))
        items=[]
        for name,a,b in (("pole_positive",gap/2,gap/2+thickness),("pole_negative",-gap/2-thickness,-gap/2)):
            shape=sector(a,b,radius-half,radius+half,180.-math.degrees(angle)/2,math.degrees(angle))
            items.append(SolidSpec(key,name,"soft_magnetic",transformed(shape,rotation=rotation,offset=offset),
                (FILTER_SOURCE,),parameters={"pole_thickness_mm":thickness},
                note="Retained reference orbit and pole gap; finite pole thickness reconstructed outside the vacuum gap. External yoke not dimensionally established."))
        return tuple(items)
    if profile=="zebra_eels_detector":
        return () # active surfaces have no specified material thickness
    origin,rotation=energy_filter_frame_mm(part,by_key)
    items=[]
    if profile=="energy_filter_multipole_carrier":
        length=float(part["housing_length_mm"])
        inner,outer=_radii(part)
        items=[SolidSpec(key,"body","unspecified",annulus(-length/2,length/2,inner,outer),
            (FILTER_SOURCE,),"configured_envelope",
            "Configured multipole carrier envelope. Individual pole count and internal material layout are not established; no extra poles are inferred from the optical channel.")]
    elif profile=="electrostatic_quadrupole":
        length=float(part["housing_length_mm"])
        inner,outer=_radii(part)
        for pole in range(4):
            items.append(SolidSpec(key,f"electrode_{pole+1}","electrode",
                sector(-length/2,length/2,inner,inner+.72*(outer-inner),90*pole,55.),
                (WIEN_SOURCE,),parameters={"electrode_sector_deg":55.},
                note="Four separated electrostatic electrodes; bore and housing length retained, angular profile reconstructed."))
    elif profile=="electrostatic_bias_tube":
        length=float(part["housing_length_mm"])
        inner,outer=_radii(part)
        items=[SolidSpec(key,"body","electrode",annulus(-length/2,length/2,inner,outer),(FILTER_SOURCE,),"configured_geometry","Configured open bias electrode tube.")]
    elif profile=="xo_energy_slit_assembly":
        gap,center=slit_opening_mm(part,runtime)
        height,thickness=float(part["clear_height_mm"]),float(part["blade_thickness_mm"])
        width=float(part.get("cad_blade_width_mm",height/2))
        for name,x in (("blade_negative",center-gap/2-width/2),("blade_positive",center+gap/2+width/2)):
            items.append(SolidSpec(key,name,"electrode",box((x,0.,0.),(width,height,thickness)),(FILTER_SOURCE,),
                parameters={"blade_width_mm":width},note="Configured gap, height and blade thickness; blade outer reach reconstructed."))
    elif profile in {"fast_electrostatic_shutter","electrostatic_camera_deflector"}:
        length,gap=float(part["electrode_length_mm"]),float(part["electrode_gap_mm"])
        outer=float(part["mechanical_outer_diameter_mm"])/2
        thickness=min(.1*(outer-gap/2),1.)
        width=2*math.sqrt(outer**2-(gap/2+thickness)**2)*.9
        for name,y in (("electrode_negative",-gap/2-thickness/2),("electrode_positive",gap/2+thickness/2)):
            items.append(SolidSpec(key,name,"electrode",box((0.,y,0.),(width,thickness,length)),(FILTER_SOURCE,),
                parameters={"electrode_thickness_mm":thickness,"electrode_width_mm":width},note="Finite opposed plates inside the retained envelope; configured gap and active length."))
    else:
        return None
    return tuple(replace(s,shape=transformed(s.shape,rotation=rotation.tolist(),offset=origin)) for s in items)


def auxiliary_solids(part, by_key, runtime=None, aperture_index=0):
    if part.get("mechanical_profile")=="electrostatic_parallel_plates":
        gap,length=float(part["plate_gap_mm"]),float(part["plate_length_mm"])
        center=float(part["local_center_z_mm"]); _,outer=_radii(part)
        thickness=min(1.,(outer-gap/2)*.1)
        if min(gap,length,thickness)<=0:
            raise ValueError("Electrostatic plates require positive gap, length and radial space")
        width=1.8*math.sqrt((outer*.9)**2-(gap/2+thickness)**2)
        items=[]
        for name,y in (("electrode_negative",-gap/2-thickness/2),("electrode_positive",gap/2+thickness/2)):
            items.append(SolidSpec(part["key"],name,"electrode",box((0.,y,center),(width,thickness,length)),
                ("https://assets.thermofisher.com/TFS-Assets/MSD/Datasheets/iliad-ultra-datasheet-ds0510.pdf",),
                note="Electrostatic blanker: configured plate gap and length; finite width/thickness reconstructed inside the retained envelope.",
                parameters={"plate_width_mm":width,"plate_thickness_mm":thickness,"plate_gap_mm":gap}))
        return tuple(items)
    for build in (filter_solids,):
        value=build(part,by_key,runtime,aperture_index)
        if value is not None:
            return value
    for value in (accelerator_solids(part),tip_solids(part,runtime),aperture_solids(part,runtime,aperture_index)):
        if value is not None:
            return value
    values = gun_electrode_solids(part)
    if values is not None:
        return values
    if part["key"]=="feg_monochromator_wien":
        from temsim.part_model_3d import _extent
        start,end=_extent(part); inner,outer=_radii(part)
        return (SolidSpec(part["key"],"body","unspecified",annulus(start,end,inner,outer),
            (WIEN_SOURCE,),"configured_envelope",
            "Configured Wien unit envelope. The cited twelve-pole patent is an alternative construction, not evidence of this unit's physical pole count."),)
    return None
