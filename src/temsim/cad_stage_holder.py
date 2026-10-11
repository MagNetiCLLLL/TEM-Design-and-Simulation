"""Side-entry stage, ordinary holder and the matching material access ports.

Shapes are millimetres in the owning module's coordinates. The X insertion
axis is independent of the beam (+Z) axis. These engineering solids do not
replace the axisymmetric equivalent boundary used by the gun field solver.
"""
from dataclasses import replace

from temsim.cad_geometry import SolidSpec, annulus, box, difference, transformed, union


PROFILES = frozenset({"transverse_goniometer", "side_entry_sample_holder"})
REFERENCE = "docs/references/cad-stage-holder.md"
X_ROTATION = ((0., 0., 1.), (0., 1., 0.), (-1., 0., 0.))


def _x_tube(start, end, inner, outer, z):
    return transformed(annulus(start, end, inner, outer), rotation=X_ROTATION,
                       offset=(0., 0., z))


def _port_tool(stage, *, outer_x=None):
    # Starts on the beam axis: only the +X wall opens. Keep the opposite wall.
    end = float(stage["stage_port_end_x_mm"]) if outer_x is None else outer_x
    return _x_tube(0., end, 0., float(stage["housing_side_port_diameter_mm"])/2,
                   float(stage["local_center_z_mm"]))


def stage_holder_solids(part, by_key, *, runtime=None):
    """Material leaves shared by native rendering, STEP and interference audit."""
    profile = part.get("mechanical_profile")
    if profile not in PROFILES:
        return None
    if profile == "transverse_goniometer" and "stage_port_start_x_mm" not in part:
        # Older documents describe only an allocation rectangle, not material.
        # Preserve their original renderer path without inventing new hardware.
        return None
    z = float(part["local_center_z_mm"])
    parameters = {name: value for name, value in part.items()
                  if name.endswith("_mm") and isinstance(value, (int, float))}
    evidence = (REFERENCE, *part.get("mechanical_geometry_source_urls", ()))
    note = str(part["mechanical_geometry_source"])

    def emit(region, shape):
        return SolidSpec(part["key"], region, str(part["material_class"]), shape,
                         evidence, "engineering_reconstruction_not_oem", note, parameters)

    if profile == "transverse_goniometer":
        return tuple(emit(region, _x_tube(
            float(part[f"stage_{region}_start_x_mm"]),
            float(part[f"stage_{region}_end_x_mm"]),
            float(part[f"stage_{region}_inner_diameter_mm"])/2,
            float(part[f"stage_{region}_outer_diameter_mm"])/2, z))
            for region in ("port", "flange", "body"))

    width = float(part["holder_tip_width_mm"])
    thickness = float(part["holder_tip_thickness_mm"])
    end = float(part["holder_tip_end_x_mm"])
    # A rounded seat at the optical origin joins a flat arm. The round rod
    # begins outside the pole noses, not in the narrow central pole gap.
    tip = union(annulus(z-thickness/2, z+thickness/2, 0., width/2),
                box((end/2, 0., z), (end, width, thickness)))
    tip = difference(tip,
        annulus(z-thickness, z+thickness, 0., float(part["holder_tip_opening_diameter_mm"])/2),
        annulus(z+thickness/2-float(part["holder_tip_seat_depth_mm"]),
                z+thickness, 0., float(part["holder_tip_seat_diameter_mm"])/2))
    values = [emit("tip", tip)]
    for region in ("shaft", "handle"):
        values.append(emit(region, _x_tube(float(part[f"holder_{region}_start_x_mm"]),
            float(part[f"holder_{region}_end_x_mm"]), 0.,
            float(part[f"holder_{region}_diameter_mm"])/2, z)))
    distance = float(part.get("holder_retraction_mm", 0.))
    if distance:
        values = [replace(spec, shape=transformed(spec.shape, offset=(distance, 0., 0.)))
                  for spec in values]
    return tuple(values)


def access_port_solids(part, by_key):
    """Open only the physical objective shell and intersecting liner material."""
    from temsim.cad_round_lenses import round_lens_solids
    from temsim.part_model_3d import _extent
    stage = by_key.get("sample_stage")
    if (part.get("mechanical_profile") == "magnetic_lens_housing"
            and part.get("parent_key") == "objective_lens" and stage is not None
            and "housing_side_port_diameter_mm" in stage):
        values = round_lens_solids(part, by_key)
    elif part.get("mechanical_profile") == "vacuum_liner":
        stage = part.get("holder_side_port")
        start, end = _extent(part)
        values = (SolidSpec(part["key"], "body", "nonmagnetic_housing",
            annulus(start, end, float(part["mechanical_inner_diameter_mm"])/2,
                    float(part["mechanical_outer_diameter_mm"])/2),
            (REFERENCE,), "configured_geometry",
            "Configured material tube, distinct from a component's equivalent solver boundary."),)
        if stage is None:
            return values
    else:
        return None
    return tuple(replace(spec, shape=difference(spec.shape, _port_tool(stage)),
        evidence=tuple(dict.fromkeys((*spec.evidence, REFERENCE))),
        status="engineering_reconstruction_not_oem",
        note=spec.note + " Side-entry holder port is a material opening. The gun solver retains an axisymmetric equivalent grounded boundary; no 3D electric-field qualification is implied.")
        for spec in values)


def liner_material_decisions(assembly):
    """Trace every equivalent segment; component bores do not add a second wall.

    The assembly resolver expands all bore radii by one nominal wall thickness
    for its axisymmetric boundary. Only the explicitly allocated continuous
    tube and inter-component drift tubes identify independent material. Inside
    an electrode, pole, aperture or detector allocation this is a numerical
    boundary, not evidence for a second tube through that component.
    """
    records = []
    for index, segment in enumerate(getattr(assembly, "vacuum_liner_segments", ())):
        generated = segment.key.startswith("@vacuum_liner:")
        independent = (not generated
            or segment.key == "@vacuum_liner:c1_c2_to_upper_objective"
            or segment.key.startswith("@vacuum_liner:@vacuum_drift:"))
        records.append(dict(index=index, key=segment.key, start_z_mm=segment.start_z_mm,
            end_z_mm=segment.end_z_mm, independent_material=independent,
            reason=("Configured independent tube or inter-component drift tube." if independent else
                "Equivalent component bore boundary only; no independently defined tube material. Existing component geometry owns this interval.")))
    return records


def physical_liner_rows(assembly, *, runtime_values=None, aperture_indices=None):
    """CAD tubes, with sample access, from unchanged equivalent solver liners.

    Each row is self-contained because the assembly's synthetic liner meshes
    have no module document. Coordinates here are already global column mm.
    """
    stage = next((part for part in assembly.parts if part.key == "sample_stage"), None)
    port = None
    if stage is not None and "housing_side_port_diameter_mm" in stage.data:
        port = {name: stage.data[name] for name in
                ("stage_port_end_x_mm", "housing_side_port_diameter_mm")}
        port["local_center_z_mm"] = stage.center_z_mm
    rows = []
    for segment, decision in zip(getattr(assembly, "vacuum_liner_segments", ()), liner_material_decisions(assembly)):
        if not decision["independent_material"]:
            continue
        row = dict(key=segment.key, name=segment.name, mechanical_profile="vacuum_liner",
            local_start_z_mm=segment.start_z_mm,
            local_center_z_mm=(segment.start_z_mm + segment.end_z_mm)/2,
            local_end_z_mm=segment.end_z_mm, length_mm=segment.end_z_mm-segment.start_z_mm,
            mechanical_inner_diameter_mm=segment.inner_diameter_mm,
            mechanical_outer_diameter_mm=segment.outer_diameter_mm,
            material_class="nonmagnetic_vacuum_liner")
        if port is not None:
            radius = float(port["housing_side_port_diameter_mm"])/2
            if (segment.start_z_mm < stage.center_z_mm + radius
                    and segment.end_z_mm > stage.center_z_mm - radius):
                row["holder_side_port"] = dict(port)
        from temsim.cad_vacuum_tube import decorate_physical_tube_row
        rows.append(decorate_physical_tube_row(row, assembly,
            runtime_values=runtime_values, aperture_indices=aperture_indices))
    return rows
