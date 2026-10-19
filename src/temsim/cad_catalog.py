"""Read-only material CAD catalog from resolved native component documents."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from temsim.cad_geometry import SolidSpec, annulus, transformed


def part_solids(part, by_key, *, runtime=None, aperture_index=0):
    """Return module-local material; () means a documented non-material plane."""
    from temsim.component_representation import non_material_role
    from temsim.cad_round_lenses import round_lens_solids
    from temsim.cad_multipoles import multipole_solids
    from temsim.cad_auxiliary import auxiliary_solids, _radii
    from temsim.part_model_3d import _extent
    from temsim.cad_stage_holder import stage_holder_solids, access_port_solids
    from temsim.cad_vacuum_tube import tube_solids
    from temsim.cad_detectors import detector_solids
    from temsim.cad_objective_sleeves import sleeve_solids
    if non_material_role(part):
        return ()
    values = tube_solids(part)
    if values is not None:
        return values
    for build in (sleeve_solids, detector_solids, stage_holder_solids, access_port_solids, round_lens_solids, multipole_solids):
        values=build(part,by_key)
        if values is not None:
            return values
    values=auxiliary_solids(part,by_key,runtime,aperture_index)
    if values is not None:
        return values
    start,end=_extent(part)
    inner,outer=_radii(part)
    if outer is None or start==end:
        return ()
    return (SolidSpec(part["key"],"body",str(part.get("material_class","unspecified")),
        annulus(start,end,inner or 0.,outer), (), "configured_envelope",
        "Retained configured envelope; internal construction and material grade are not established."),)


def native_reconstruction(part, by_key, *, runtime=None, aperture_index=0):
    """Only replace incomplete native envelopes; preserve existing face editing."""
    from temsim.cad_multipoles import multipole_solids
    from temsim.cad_auxiliary import accelerator_solids, filter_solids
    from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES
    from temsim.cad_stage_holder import stage_holder_solids, access_port_solids
    from temsim.cad_vacuum_tube import tube_solids
    from temsim.cad_detectors import detector_solids
    from temsim.cad_objective_sleeves import sleeve_solids
    values = tube_solids(part)
    if values is not None:
        return values
    for build in (sleeve_solids, detector_solids, stage_holder_solids, access_port_solids):
        values = build(part, by_key)
        if values is not None:
            return values
    # Existing saddle-loop face provenance and Tip emission overlays stay native.
    if part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES:
        return None
    values=multipole_solids(part,by_key)
    if values is not None:
        return values or None
    values=accelerator_solids(part)
    if values is not None:
        return values
    if part.get("key")=="feg_monochromator_wien" or part.get("mechanical_profile")=="electrostatic_parallel_plates":
        from temsim.cad_auxiliary import auxiliary_solids
        return auxiliary_solids(part,by_key,runtime,aperture_index)
    if part.get("key") != "energy_filter_entrance_aperture":
        values=filter_solids(part,by_key,runtime,aperture_index)
        if values:
            return values
    return None


def is_reconstruction_region(part, region):
    """Display material categories do not create new FEM material assignments."""
    if part.get("model_3d",{}).get("base",{}).get("kind") == "step" and region.startswith("step_"):
        return True
    if part.get("mechanical_profile") == "objective_electronics_sleeve":
        return region == "body" and part.get("mechanical_part_role") != "custom_mechanical_copy"
    from temsim.cad_multipoles import multipole_solids
    from temsim.cad_auxiliary import accelerator_solids
    from temsim.cad_stage_holder import PROFILES
    if (part.get("mechanical_profile") in {"transverse_eds_detector_array", "zebra_eels_detector"}
            and part.get("detector_material_geometry_status")):
        return True
    if (part.get("mechanical_profile") in PROFILES or "holder_side_port" in part
            or (part.get("mechanical_profile") == "magnetic_lens_housing"
                and part.get("parent_key") == "objective_lens")):
        return True
    if part.get("mechanical_profile")=="electrostatic_parallel_plates":
        return region in {"housing","electrode_negative","electrode_positive"}
    if part.get("key")=="feg_monochromator_wien":
        return region in {"housing","return_yoke"} or region.startswith("pole_")
    if part.get("branch_path_only"):
        return (region in {"pole_positive","pole_negative","blade_negative","blade_positive",
                           "electrode_negative","electrode_positive","housing","return_yoke"}
                or region.startswith("pole_") or region.startswith("electrode_"))
    if "stage_centers_z_mm" in part:
        return region in {s.region for s in accelerator_solids(part)}
    try:
        values=multipole_solids(part,{part.get("key"):part})
    except ValueError:
        return False
    return values is not None and region in {s.region for s in values}


def posed_solids(part,by_key,**kwargs):
    from temsim.lens_pose import mechanical_part_transform_mm
    from temsim.cad_features import map_solid
    from temsim.component_representation import non_material_role
    if non_material_role(part):
        return ()
    explicit=part.get("model_3d",{}).get("base",{}).get("kind","existing") != "existing"
    values=() if explicit else part_solids(part,by_key,**kwargs)
    from temsim.energy_filter_model_3d import supports_energy_filter_part, FILTER_TO_COLUMN_ROTATION, energy_filter_origin_mm
    if "model_3d" in part:
        from temsim.cad_features import apply_solid_features
        if supports_energy_filter_part(part):
            from temsim.energy_filter_model_3d import energy_filter_component_pose
            origin,basis=energy_filter_component_pose(part,by_key)
            center=np.array([0.,0.,part["local_center_z_mm"]])
            values=tuple(replace(s,shape=transformed(s.shape,rotation=basis.T.tolist(),offset=center-basis.T@origin)) for s in values)
            values=apply_solid_features(part,values)
            values=tuple(map_solid(s,lambda shape: transformed(shape,rotation=basis.tolist(),offset=origin-basis@center)) for s in values)
        else:
            from temsim.part_model_3d import lens_shape_part
            values=apply_solid_features(lens_shape_part(part),values)
    if supports_energy_filter_part(part):
        values=tuple(map_solid(s,lambda shape: transformed(shape,rotation=[list(r) for r in FILTER_TO_COLUMN_ROTATION],
            offset=(0.,0.,energy_filter_origin_mm(by_key)))) for s in values)
    rotation,translation=mechanical_part_transform_mm(part,by_key)
    if not (np.array_equal(rotation,np.eye(3)) and np.array_equal(translation,np.zeros(3))):
        values=tuple(map_solid(s,lambda shape: transformed(shape,rotation=rotation.tolist(),offset=translation)) for s in values)
    return values


def module_catalog(document, *, source="", runtime_values=None, aperture_indices=None):
    by_key={part["key"]:part for part in document["parts"]}
    solids,units,omitted=[],[],[]
    groups={}
    for part in by_key.values():
        values=posed_solids(part,by_key,runtime=(runtime_values or {}).get(part["key"]),
                            aperture_index=(aperture_indices or {}).get(part["key"],0))
        if not values:
            omitted.append({"key":part["key"],"reason":"No independent closed material (parent assembly, control channel, reference/active plane, or unspecified envelope)."})
            continue
        owner=part
        visited={owner["key"]}
        while owner.get("parent_key") in by_key and owner.get("parent_key") not in visited:
            if owner.get("cad_export_unit"):
                break
            candidate=by_key[owner["parent_key"]]
            # A complete optical unit owns its pole/coil/yoke, but an adjacent
            # aperture stays a separately editable optical unit.
            if "aperture" in part["key"] or part["key"] == "sample":
                break
            owner=candidate; visited.add(owner["key"])
        group=groups.setdefault(owner["key"],{"key":owner["key"],"name":owner.get("name",owner["key"]),"solid_ids":[]})
        for spec in values:
            row=asdict(spec); row["color"]=list(spec.color)
            if "step_color_rgba" in spec.parameters:
                row["color"]=list(spec.parameters["step_color_rgba"])
            from temsim.part_materials import REGIONS, material_for_region, configured_region_colour
            if spec.region in REGIONS:
                assignment=material_for_region(part,spec.region)
                if assignment is not None:
                    row["material"]=assignment["material_key"]
                    row["color"]=list(configured_region_colour(part,spec.region,spec.color))
            solids.append(row); group["solid_ids"].append(f"{spec.key}/{spec.region}")
    units=list(groups.values())
    module=document["module"]
    import hashlib
    from temsim.step_assets import canonical_json
    return {"key":module["key"],"name":module.get("name",module["key"]),"source":str(source),
            "source_document_sha256":hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest(),
            "solids":solids,"units":units,"omitted":omitted}


def build_catalog(root):
    from temsim.part_model_document import PartModelDocument
    root=Path(root)
    modules=[]
    for folder in ("gun","beam_blanker","column","project_and_recording_system"):
        for path in sorted((root/"configs/instruments"/folder).glob("*.toml")):
            doc=PartModelDocument(path).document
            modules.append(module_catalog(doc,source=str(path.relative_to(root))))
    return {"schema":"temsim-analytic-material-cad-v1","units":"mm","modules":modules,
            "scope":"Configured envelopes and documented internal reconstruction. No OEM mechanical certification or shape-derived field recalibration.",
            "construction":"Analytic solids, not triangular STEP. Material names are categories; no invented density or B-H data."}


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=".")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    result=build_catalog(args.root)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    print(f"{len(result['modules'])} modules; {sum(len(m['units']) for m in result['modules'])} optical units; {sum(len(m['solids']) for m in result['modules'])} material solids")


if __name__=="__main__":
    main()
