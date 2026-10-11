"""Compare one frozen native TOML assembly with and without CAD reconstruction.

The baseline is the current saved TOML, including resolved subassemblies and
Tip definitions, rendered through the original native geometry path. It is not
an old Git revision with different optical presets. No TOML or solver is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@contextmanager
def original_geometry():
    # This hook is process-local, reversible, and does not modify any file.
    with patch("temsim.cad_catalog.native_reconstruction", return_value=None):
        yield


def mesh_summary(mesh):
    vertices = np.asarray(mesh.vertices)
    sha = hashlib.sha256(vertices.tobytes() + np.asarray(mesh.faces).tobytes()).hexdigest()
    return dict(key=mesh.key, region=mesh.region, material=mesh.material_class,
                is_exact=bool(mesh.is_exact), wireframe=bool(mesh.wireframe),
                description=mesh.description, vertices=len(vertices), triangles=len(mesh.faces),
                minimum_mm=vertices.min(axis=0).tolist(), maximum_mm=vertices.max(axis=0).tolist(),
                geometry_sha256=sha)


def bounds(meshes, part=None, by_key=None):
    if not meshes:
        return None
    vertices = np.vstack([mesh.vertices for mesh in meshes])
    frame = "module local XYZ"
    if part is not None:
        from temsim.lens_pose import effective_part_transform_mm
        rotation, translation = effective_part_transform_mm(part, by_key)
        vertices = (vertices - translation) @ rotation
        frame = "unposed module local XYZ"
        if part.get("branch_path_only") and part.get("path_center_mm") is not None:
            from temsim.energy_filter_model_3d import energy_filter_frame_mm
            origin, rotation = energy_filter_frame_mm(part, by_key)
            vertices = (vertices - origin) @ rotation
            frame = "branch component transverse XY and longitudinal Z, centred on path_center_mm"
    radial = np.linalg.norm(vertices[:, :2], axis=1)
    return dict(frame=frame, minimum_mm=vertices.min(axis=0).tolist(),
                maximum_mm=vertices.max(axis=0).tolist(),
                radial_vertex_min_mm=float(radial.min()), radial_vertex_max_mm=float(radial.max()))


def dimensional_checks(part, measured, by_key=None):
    """Explicit applicability: vacuum envelopes are not automatically material holes."""
    if measured is None:
        return dict(applicable=False, reason="No native surface; reference/channel/parent ownership is reported separately.")
    from temsim.component_representation import non_material_role
    from temsim.part_model_apertures import is_strip_aperture
    if non_material_role(part):
        return dict(applicable=False, reason="Non-material channel or reference.")
    if is_strip_aperture(part) or part.get("tip_particle_model"):
        return dict(applicable=False, reason="Aperture mechanism or physical Tip uses its own declared geometry, not a cylindrical packing envelope.")
    if part.get("mechanical_profile") == "tapered_sector_prism":
        return dict(applicable=False, reason="Curved sector: pole-gap and radial-channel checks are reported independently.")
    outer_field = next((name for name in ("mechanical_outer_diameter_mm", "outer_diameter_mm") if name in part), None)
    bore_field = next((name for name in ("mechanical_inner_diameter_mm", "mechanical_bore_diameter_mm",
                                        "mechanical_clear_bore_diameter_mm") if name in part), None)
    checks = {}
    if outer_field is not None:
        checks["outer_envelope"] = dict(field=outer_field, radius_mm=float(part[outer_field])/2,
            measured_vertex_radius_mm=measured["radial_vertex_max_mm"],
            margin_mm=float(part[outer_field])/2-measured["radial_vertex_max_mm"])
    if bore_field is not None:
        checks["material_bore"] = dict(field=bore_field, radius_mm=float(part[bore_field])/2,
            measured_vertex_radius_mm=measured["radial_vertex_min_mm"],
            margin_mm=measured["radial_vertex_min_mm"]-float(part[bore_field])/2,
            limitation="Vertex-radius check; tessellated chords and 3D material collisions are not certified by this scalar metric.")
    if part.get("branch_path_only"):
        field = next((name for name in ("housing_length_mm", "electrode_length_mm", "blade_thickness_mm") if name in part), None)
        limits = (-float(part[field])/2, float(part[field])/2) if field else None
    else:
        from temsim.part_model_3d import _extent
        limits = _extent(part)
        field = "local_start_z_mm / local_end_z_mm (physical-length fallback for zero packing length)"
        parent=(by_key or {}).get(part.get("parent_key"),{})
        intervals=()
        if part.get("mechanical_profile") in {"magnetic_excitation_coil","magnetic_lens_yoke"}:
            from temsim.magnetic_geometry import objective_layer_intervals_mm
            intervals=objective_layer_intervals_mm(parent,parent.get("local_start_z_mm",0),part["mechanical_profile"])
        if intervals:
            checks["axial_packing_reference"]=dict(field=field,minimum_mm=limits[0],maximum_mm=limits[1],
                rendered_minimum_minus_packing_minimum_mm=measured["minimum_mm"][2]-limits[0],
                packing_maximum_minus_rendered_maximum_mm=limits[1]-measured["maximum_mm"][2],
                note="Equivalent packing segment is not the extent of split objective material. Parent-defined physical intervals govern both original and integrated renderers.")
            checks["physical_intervals"]=dict(intervals_mm=[list(value) for value in intervals],
                authority="Parent upper/lower yoke bounds and configured axial coil inset")
            limits=(min(a for a,b in intervals),max(b for a,b in intervals))
            field="parent-defined physical material intervals"
    if limits and limits[1] > limits[0]:
        checks["axial_envelope"] = dict(field=field, minimum_mm=limits[0], maximum_mm=limits[1],
            start_margin_mm=measured["minimum_mm"][2]-limits[0],
            end_margin_mm=limits[1]-measured["maximum_mm"][2])
    deviations = [name for name, record in checks.items()
                  if any(value < -1e-7 for key, value in record.items() if key.endswith("margin_mm"))]
    return dict(applicable=bool(checks), checks=checks, negative_margins=deviations)


def build_models(document, segments):
    from temsim.part_model_3d import module_model_from_document
    frozen = digest(document)
    with original_geometry():
        original = module_model_from_document(document, angular_segments=segments)
    integrated = module_model_from_document(document, angular_segments=segments)
    if digest(document) != frozen:
        raise RuntimeError("A geometry path modified its captured TOML document")
    return original, integrated


def compare_module(document, original, integrated):
    originals, current = defaultdict(list), defaultdict(list)
    for mesh in original.meshes:
        originals[mesh.key].append(mesh)
    for mesh in integrated.meshes:
        current[mesh.key].append(mesh)
    by_key = {row["key"]: row for row in document["parts"]}
    extra = (set(originals) | set(current)) - set(by_key)
    if extra:
        raise RuntimeError(f"Renderer created unowned part identities: {extra}")
    rows = []
    for index, part in enumerate(document["parts"]):
        key = part["key"]
        old, new = originals[key], current[key]
        old_records, new_records = [mesh_summary(m) for m in old], [mesh_summary(m) for m in new]
        old_hashes = sorted(r["geometry_sha256"] for r in old_records)
        new_hashes = sorted(r["geometry_sha256"] for r in new_records)
        old_bounds, new_bounds = bounds(old, part, by_key), bounds(new, part, by_key)
        original_checks = dimensional_checks(part, old_bounds, by_key)
        integrated_checks = dimensional_checks(part, new_bounds, by_key)
        delta = None
        if old_bounds is not None and new_bounds is not None:
            delta = dict(minimum_mm=(np.array(new_bounds["minimum_mm"])-old_bounds["minimum_mm"]).tolist(),
                         maximum_mm=(np.array(new_bounds["maximum_mm"])-old_bounds["maximum_mm"]).tolist(),
                         radial_vertex_min_mm=new_bounds["radial_vertex_min_mm"]-old_bounds["radial_vertex_min_mm"],
                         radial_vertex_max_mm=new_bounds["radial_vertex_max_mm"]-old_bounds["radial_vertex_max_mm"])
        from temsim.component_representation import non_material_role
        role = non_material_role(part)
        classification = ("non_material" if role else "no_surface" if not old and not new else
                          "unchanged_native_geometry" if old_hashes == new_hashes else
                          "display_geometry_or_representation_changed")
        rows.append(dict(key=key, name=part.get("name",key), saved_order=part.get("order"),
            document_index=index, parent_key=part.get("parent_key"), physical_host_key=part.get("physical_host_key"),
            profile=part.get("mechanical_profile"), classification=classification,
            original_description=[m.description for m in old], integrated_description=[m.description for m in new],
            original_meshes=old_records, integrated_meshes=new_records,
            saved_coordinates={name:part[name] for name in ("local_start_z_mm","local_center_z_mm","local_end_z_mm",
                "optical_reference_local_z_mm","path_center_mm","path_reference","length_mm") if name in part},
            original_bounds=old_bounds, integrated_bounds=new_bounds, bounds_delta=delta,
            original_dimensional_checks=original_checks, integrated_dimensional_checks=integrated_checks,
            newly_negative_margins=sorted(set(integrated_checks.get("negative_margins",()))-
                                           set(original_checks.get("negative_margins",())))))
    insert_clearances=[]
    for part in document["parts"]:
        parent=by_key.get(part.get("parent_key"))
        if (part.get("mechanical_profile") not in {"deflector_coil_insert","corrector_dipole_coil_insert"}
                or parent is None or "mechanical_inner_diameter_mm" not in parent or not current[part["key"]]):
            continue
        measured=bounds(current[part["key"]],parent,by_key)
        insert_clearances.append(dict(insert=part["key"],host=parent["key"],
            radial_clearance_to_host_inner_mm=float(parent["mechanical_inner_diameter_mm"])/2-measured["radial_vertex_max_mm"],
            upstream_margin_mm=measured["minimum_mm"][2]-float(parent["local_start_z_mm"]),
            downstream_margin_mm=float(parent["local_end_z_mm"])-measured["maximum_mm"][2],
            scope="Axial extents and coaxial host-bore vertex clearance, not a general collision test."))
    return dict(key=document["module"]["key"], name=document["module"].get("name"),
                resolved_document_sha256=digest(document), parts=rows,
                part_identity_and_parent_order="same frozen document used in both render paths",
                optical_centres="same frozen values, no recalibration or solver execution",
                original_mesh_count=len(original.meshes), integrated_mesh_count=len(integrated.meshes),
                host_insert_clearances=insert_clearances,
                classification_counts=dict(Counter(row["classification"] for row in rows)))


def assembled_models(root, segments):
    from temsim.column.module_assembly import resolve_module_assembly
    from temsim.assembly_model_3d import assembly_model_from_assembly, assembly_model_fingerprint, _document_value
    configuration = SimpleNamespace(electron_gun_type="cold_feg", monochromator_installed=False,
        corrector=SimpleNamespace(value="probe_corrector"), c3_hardware=SimpleNamespace(value="three_condenser"),
        energy_filter_selected=True, nanopulser_installed=False)
    assembly = resolve_module_assembly(configuration, root=root/"configs/instruments")
    before = assembly_model_fingerprint(assembly)
    with original_geometry():
        original = assembly_model_from_assembly(assembly, angular_segments=segments)
    integrated = assembly_model_from_assembly(assembly, angular_segments=segments)
    after = assembly_model_fingerprint(assembly)
    rows = [dict(key=p.key, module=p.module_key, source=p.source_file, parent=p.parent_key,
                 start_z_mm=p.start_z_mm, centre_z_mm=p.center_z_mm, end_z_mm=p.end_z_mm,
                 optical_reference_local_z_mm=p.data.get("optical_reference_local_z_mm"),
                 module_to_column_z_mm=p.center_z_mm-p.data["local_center_z_mm"])
            for p in assembly.parts]
    # Independently derive module-to-column placement from the source local mesh.
    from temsim.part_model_3d import part_model_from_document
    from temsim.energy_filter_model_3d import supports_energy_filter_part, FILTER_TO_COLUMN_ROTATION, energy_filter_origin_mm
    documents = {m.key:dict(parts=[_document_value(p.data) for p in m.parts]) for m in assembly.modules}
    actual = defaultdict(list)
    for mesh in integrated.meshes:
        actual[mesh.key].append(mesh)
    placement = []
    for part in assembly.parts:
        if part.key not in actual:
            continue
        doc = documents[part.module_key]
        local = part_model_from_document(doc,part.key,angular_segments=segments,include_children=False)
        shift = part.center_z_mm-float(part.data["local_center_z_mm"])
        errors = []
        for before_mesh, after_mesh in zip(local.meshes,actual[part.key],strict=True):
            if before_mesh.region != after_mesh.region:
                raise RuntimeError("Assembled region order differs from local material order")
            expected = before_mesh.vertices.copy()
            if supports_energy_filter_part(part.data):
                by_key = {row["key"]:row for row in doc["parts"]}
                expected = expected @ np.asarray(FILTER_TO_COLUMN_ROTATION).T
                expected[:,2] += energy_filter_origin_mm(by_key)
            expected[:,2] += shift
            errors.append(float(np.max(np.abs(expected-after_mesh.vertices))))
        placement.append(dict(key=part.key,maximum_vertex_placement_error_mm=max(errors,default=0.)))
    return original,integrated,dict(selection="FEG + C3 + Probe Corrector + Energy Filter", parts=rows,
        fingerprint_before=before,fingerprint_after=after, immutable_assembly=before==after,
        original_errors=list(original.errors),integrated_errors=list(integrated.errors),
        original_omitted=list(original.omitted_keys),integrated_omitted=list(integrated.omitted_keys),
        placement_checks=placement,
        maximum_vertex_placement_error_mm=max((r["maximum_vertex_placement_error_mm"] for r in placement),default=0.),
        scope="Static saved assembly; no source, particle, wave, FEM or GPU calculation.")


def normalized_palette(mesh):
    from temsim.cad_geometry import MATERIAL_COLORS
    name=mesh.material_class.lower()
    kind=("copper_winding" if "copper" in name or "winding" in name else
          "insulator" if "insulat" in name or "ceramic" in name else
          "nonmagnetic_housing" if "nonmagnetic" in name or "non_magnetic" in name else
          "soft_magnetic" if "magnetic" in name or "iron" in name else
          "electrode" if "electrode" in name else "tungsten" if "tungsten" in name else None)
    return replace(mesh,color=MATERIAL_COLORS.get(kind,(.64,.70,.75,1.)))


def render_pair(output,slug,title,original,integrated,*,column=False,section=True):
    from PySide6.QtGui import QFont,QFontDatabase
    from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QHBoxLayout,QLabel
    from temsim.gui.part_model_view import PartModelView
    app=QApplication.instance() or QApplication([])
    font=Path(os.environ.get("WINDIR","C:/Windows"))/"Fonts/segoeui.ttf"
    if font.exists():
        families=QFontDatabase.applicationFontFamilies(QFontDatabase.addApplicationFont(str(font)))
        if families: app.setFont(QFont(families[0],10))
    window=QWidget()
    window.setStyleSheet("QWidget {background:#101823;color:#e5edf7;} QLabel {background:transparent;}")
    layout=QVBoxLayout(window)
    heading=QLabel(title); heading.setStyleSheet("font-size:20px;font-weight:600;")
    layout.addWidget(heading)
    label=QLabel("Same frozen TOML • Same scale and camera • Native PartModelView • Internals are not OEM-certified")
    layout.addWidget(label)
    row=QHBoxLayout(); layout.addLayout(row)
    views=[]
    groups=[tuple(normalized_palette(m) for m in original),tuple(normalized_palette(m) for m in integrated)]
    for caption,meshes in zip(("Original TOML geometry / envelopes","Integrated CAD material geometry"),groups):
        panel=QWidget(); panel_layout=QVBoxLayout(panel); panel_layout.addWidget(QLabel(caption))
        view=PartModelView(); view.setFixedSize(730,850 if column else 600)
        view.set_meshes(meshes); view.set_section_enabled(section,keep_positive_y=not column)
        if column: view.set_column_view()
        else: view.set_isometric_view()
        view.set_view_labels(title="Section" if section else "Full material")
        panel_layout.addWidget(view); row.addWidget(panel); views.append(view)
    # Fit the UNION to both views: an enlarged/new body cannot be hidden by auto-fit.
    points=[m.vertices for group in groups for m in group]
    for view in views: view._fit_points(points)
    legend=QLabel("Copper = winding | blue = magnetic material | cream = insulator | grey = other/unspecified. "
                 "Identical material-category palette on both sides; unknown material stays unspecified.")
    legend.setWordWrap(True); layout.addWidget(legend)
    window.show(); app.processEvents()
    for view in views: view.repaint()
    app.processEvents()
    path=output/f"{slug}.png"
    if not window.grab().save(str(path)): raise RuntimeError(f"Could not save {path}")
    record=dict(file=path.name,title=title,renderer="PartModelView",platform=app.platformName(),
        desktop_automation=False,section_enabled=section,section_retains="y <= 0" if column else "y >= 0",
        camera_rotation=views[0].camera_rotation.tolist(),camera_centre_mm=views[0]._center.tolist(),
        camera_radius_mm=views[0]._radius,pixels_per_mm=views[0]._scale(),
        equal_camera_and_scale=bool(np.array_equal(views[0].camera_rotation,views[1].camera_rotation)
                                   and views[0]._scale()==views[1]._scale()),
        original_keys=sorted({m.key for m in original}),integrated_keys=sorted({m.key for m in integrated}),
        original_regions=[dict(key=m.key,region=m.region) for m in original],
        integrated_regions=[dict(key=m.key,region=m.region) for m in integrated])
    window.close()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,default=Path("outputs/cad_models_20261011/assembly-comparison"))
    parser.add_argument("--angular-segments",type=int,default=64)
    parser.add_argument("--no-images",action="store_true")
    args=parser.parse_args(); root=args.root.resolve(); output=args.output.resolve()
    sys.path.insert(0,str(root/"src")); os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
    from temsim.part_model_document import PartModelDocument
    from temsim.shared_tip import dependencies
    paths=[p for folder in ("gun","beam_blanker","column","project_and_recording_system")
           for p in sorted((root/"configs/instruments"/folder).glob("*.toml"))]
    if len(paths)!=11: raise RuntimeError(f"Expected 11 saved native modules; found {len(paths)}")
    source_paths={root/"configs/instruments/catalog.toml",*paths}
    for path in paths: source_paths.update(dependencies(path))
    snapshots={path:path.read_bytes() for path in source_paths}
    implementation=list((root/"src/temsim").glob("cad_*.py"))+[
        root/p for p in ("src/temsim/part_model_3d.py","src/temsim/part_model_features.py",
        "src/temsim/energy_filter_model_3d.py","src/temsim/corrector_model_3d.py",
        "src/temsim/gui/part_model_view.py","src/temsim/column/module_assembly.py",
        "src/temsim/assembly_model_3d.py","scripts/compare_cad_assembly.py")]
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation}
    output.mkdir(parents=True,exist_ok=True)
    for path,data in snapshots.items():
        saved=output/"original-toml"/path.relative_to(root)
        saved.parent.mkdir(parents=True,exist_ok=True); saved.write_bytes(data)
    report=dict(schema="frozen-toml-native-versus-integrated-cad-v1",status="RUNNING",
        scope="Current saved TOMLs and their resolved includes; baseline disables only native_reconstruction in this isolated process. Not an older optical preset or claimed OEM CAD.",
        angular_segments=args.angular_segments, source_files=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(data).hexdigest()) for p,data in sorted(snapshots.items())],
        implementation_sha256_before=hashes,modules=[],images=[])
    models={}; documents={}
    for path in paths:
        slug=path.stem; print("Comparing "+slug,flush=True)
        document=PartModelDocument(path).document
        save_json(output/"resolved-originals"/f"{slug}.json",document)
        old,new=build_models(document,args.angular_segments); models[slug]=(old,new); documents[slug]=document
        result=compare_module(document,old,new); result["source"]=str(path.relative_to(root))
        report["modules"].append(result)
        save_json(output/"comparison.json",report)
    print("Comparing installed assembly",flush=True)
    old,new,assembly=assembled_models(root,args.angular_segments); report["installed_assembly"]=assembly
    if not args.no_images:
        for slug,(a,b) in models.items():
            report["images"].append(render_pair(output,slug,slug,a.meshes,b.meshes,column=True))
        report["images"].append(render_pair(output,"installed_assembly","FEG + Probe Corrector + Energy Filter — installed assembly",old.meshes,new.meshes,column=True))
        cases=[("C3_ProbeCorrector","probe_hp2_hexapole"),("C3_ProbeCorrector","ac_deflector"),
               ("C3_ProbeCorrector","objective_lens"),("FEG","feg_accelerator"),
               ("ElectrostaticBeamBlanker","nanopulser_deflector"),("EnergyFilter","energy_filter_tapered_prism")]
        from temsim.part_model_3d import part_model_from_document
        for slug,key in cases:
            document=documents[slug]
            with original_geometry(): a=part_model_from_document(document,key,angular_segments=args.angular_segments)
            b=part_model_from_document(document,key,angular_segments=args.angular_segments)
            report["images"].append(render_pair(output,"detail_"+key,key,a.meshes,b.meshes))
    report["source_files_unchanged"]=all(path.read_bytes()==data for path,data in snapshots.items())
    report["implementation_sha256_after"]={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in implementation}
    report["implementation_unchanged"]=report["implementation_sha256_after"]==hashes
    rows=[part for module in report["modules"] for part in module["parts"]]
    report["summary"]=dict(modules=len(paths),part_occurrences=len(rows),unique_part_keys=len({r["key"] for r in rows}),
        classification_counts=dict(Counter(r["classification"] for r in rows)),
        newly_negative_margin_occurrences=[dict(module=m["key"],key=p["key"],checks=p["newly_negative_margins"])
            for m in report["modules"] for p in m["parts"] if p["newly_negative_margins"]])
    report["status"]=("COMPARISON_COMPLETE_REVIEW_DIFFERENCES" if report["source_files_unchanged"] and report["implementation_unchanged"]
        and assembly["immutable_assembly"] and not assembly["integrated_errors"] and assembly["maximum_vertex_placement_error_mm"]<1e-8
        else "FAILED_IDENTITY_OR_PLACEMENT")
    report["limitations"]=["No optical propagation or field changes were performed.",
        "New material subdivisions are not new optical components; reference/channel omissions remain explicit.",
        "Scalar envelope checks do not establish nonintersecting 3D parts or exact vacuum clearance; inspect recorded margins and finite material models.",
        "No arbitrary unknown mechanism is qualified by a successful renderer. New internal construction must retain its evidence/reconstruction label.",
        "Original schematic envelopes may overlap intentionally. Their volumes are not reference material volumes.",
        "All screenshots use one angular-segment setting and the same material-category palette; tessellation is display precision, not evidence of fine internal structure or STEP tolerance."]
    save_json(output/"comparison.json",report)
    lines=["# Native TOML and integrated CAD comparison","",report["scope"],"",
        f"Status: {report['status']}",f"Saved module variants: {len(paths)}; part occurrences: {len(rows)}; unique keys: {report['summary']['unique_part_keys']}",
        f"Input files unchanged: {report['source_files_unchanged']}; implementation unchanged: {report['implementation_unchanged']}",
        f"Maximum installed placement difference: {assembly['maximum_vertex_placement_error_mm']:.6g} mm","",
        "## Geometry categories",""]
    lines.extend(f"- {key}: {value}" for key,value in report["summary"]["classification_counts"].items())
    lines.extend(["","## Newly negative envelope/bore margins","","These are diagnostic differences requiring interpretation, not silently passed checks.",""])
    lines.extend(f"- {row['module']} / {row['key']}: {', '.join(row['checks'])}" for row in report["summary"]["newly_negative_margin_occurrences"])
    if not report["summary"]["newly_negative_margin_occurrences"]: lines.append("None detected by the stated applicable scalar checks.")
    lines.extend(["","## Same-camera native views",""])
    lines.extend(f"- [{image['title']}]({image['file']})" for image in report["images"])
    lines.extend(["","## Scope limits",""]+report["limitations"])
    (output/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(dict(status=report["status"],summary=report["summary"])),flush=True)


if __name__=="__main__":
    main()
