"""Identify retained references which do not define an editable material size.

This policy does not rewrite stored optics or invent missing material. Native
dimensions remain available in All parameters when an explicit base is active.
"""


def dimension_reference_reason(part, field):
    template = part.get("geometry_template_key", part.get("key"))
    profile = part.get("mechanical_profile", "")
    native = part.get("model_3d", {}).get("base", {}).get("kind", "existing") == "existing"
    from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES
    if native:
        if field == "length_mm" and "stage_centers_z_mm" in part and "electrode_thickness_mm" in part:
            return ("Accelerator installation allocation only. The material is defined by "
                    "stage_centers_z_mm and electrode_thickness_mm, with insulation between stages. "
                    "Edit those explicit dimensions to change the rings; this reference length does not resize them.")
        if part.get("aperture_plate_form") == "perforated_strip":
            if field == "length_mm":
                thickness = "plate_thickness_mm" if "plate_thickness_mm" in part else "active_length_mm"
                return ("Aperture mechanism allocation only, not material thickness. Edit " + thickness +
                        " for the plate, or the explicit aperture_support_* dimensions for a modelled rod/screw. "
                        "The installation allocation does not resize the plate or change its optical plane.")
            if (field in {"mechanical_outer_diameter_mm", "outer_diameter_mm"}
                    and part.get("aperture_support_profile") == "single_rod_nominal_screw"):
                return ("Mechanism radial allocation only. The material outline is defined by "
                        "aperture_support_plate_half_width_mm, plate Y bounds and rod/screw dimensions. "
                        "This reference diameter only checks that those separate parts fit the allocation.")
        if (field == "length_mm" and part.get("interaction_centers_local_z_mm") is not None
                and (str(template).endswith("deflector") or profile == "magnetic_deflector")
                and not part.get("branch_path_only")
                and any(name in part for name in ("mechanical_coil_length_mm", "effective_thickness_mm", "active_length_mm"))):
            copper_length = next(name for name in ("mechanical_coil_length_mm", "effective_thickness_mm", "active_length_mm")
                                 if name in part)
            return ("Deflector installation allocation only. Actual coils use interaction_centers_local_z_mm "
                    "and " + copper_length + "; edit those established dimensions to change their position or extent. "
                    "The allocation does not stretch copper or recalibrate the linked optical field.")
        if field == "length_mm" and profile == "transverse_goniometer" and "stage_port_start_x_mm" in part:
            return ("Stage beam-axis allocation only. This side-entry structure is built along X from "
                    "stage_port/flange/body_start_x_mm, end_x_mm and their diameters. Edit those material "
                    "dimensions; the Z allocation is constrained by the objective gap.")
        if field == "length_mm" and profile == "side_entry_sample_holder" and "holder_tip_width_mm" in part:
            return ("Holder beam-axis allocation only. The material uses holder_tip_width_mm/thickness_mm "
                    "and the explicit shaft/handle X endpoints and diameters. Edit those dimensions; "
                    "this reference allocation does not scale the transverse holder.")
        if field == "length_mm" and (template == "eds_detector_system" or template == "energy_filter"):
            return ("Zero-thickness assembly/interface reference, not a material body. Edit the EDS detector "
                    "face/housing dimensions or the Energy Filter child components to change their solids. "
                    "Changing this marker length would not construct material.")
    if (field == "length_mm" and part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES
            and "winding_axial_length_mm" in part
            and native):
        return ("Installation/field allocation length only. Explicit winding_axial_length_mm "
                "defines the copper extent about its saved centre; edit Winding axial length "
                "to resize the material without changing the optical allocation.")
    if field == "pole_cone_angle_to_axis_deg" and part.get("mechanical_profile") == "magnetic_pole_piece":
        return ("Reference cone angle only. The material contour is defined by its saved "
                "bore, outer/tip diameters, nose length and face land; this legacy angle "
                "does not define a second, potentially inconsistent contour.")
    if field == "pole_gap_mm" and template in {"condenser_lens_1", "condenser_lens_2"}:
        return ("Reference pole gap only: this C1/C2 model declares one pole face, "
                "not a complete two-pole gap. Edit the explicitly modelled pole dimensions; "
                "the missing return geometry is not inferred from this number.")
    if (field in {"mechanical_bore_diameter_mm", "bore_diameter_mm"}
            and part.get("aperture_plate_form") == "perforated_strip"
            and template in {"condenser_aperture_2", "condenser_aperture_3",
                             "objective_aperture", "selected_area_aperture",
                             "energy_filter_entrance_aperture"}):
        return ("Carrier-bore reference only: the carrier body is not modelled by this "
                "strip/rod geometry. Edit the plate/support dimensions or working opening "
                "to change their corresponding surfaces; this value is not the working hole.")
    return ""


def inactive_native_dimension_reason(part, field):
    if (part.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing"
            and field != "model_3d"
            and not field.startswith(("offset_", "rotation_"))
            and not (field == "length_mm" and part.get("mechanical_part_role")
                     in {"custom_mechanical", "custom_mechanical_copy"})):
        return ("Stored native geometry is inactive while a replacement base is selected. "
                "Edit the base dimensions and cut features below, or restore the native shape. "
                "The original recipe is retained; its optical settings are separate.")
    return ""
