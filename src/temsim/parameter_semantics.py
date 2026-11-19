"""Conservative parameter meanings and declared provenance, without a GUI.

Categories describe what a value means, not whether it was measured. Evidence
about a component's existence or photograph topology does not establish any of
its dimensions. A stored source declaration is reported, not independently
certified by this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re

from temsim.component_keys import CAMERA, FLUORESCENT_SCREEN, STEM_DETECTOR_KEYS
from temsim.mechanical_profiles import DEFLECTOR_COIL_PROFILES
from temsim.pole_geometry import RECESSED_OBJECTIVE_STYLE


CATEGORY_LABELS = {
    "physical": "Physical dimension", "envelope": "Envelope / allocation",
    "vacuum": "Vacuum passage", "operating": "Operating value / limit",
    "placement": "Position / reference", "material": "Material definition",
    "cad": "User CAD geometry", "unknown": "Meaning not classified",
}
SOURCE_LABELS = {
    "measured": "Measured (declared)", "documented": "Documented (declared)",
    "estimated": "Estimated / reconstructed", "unspecified": "Source unspecified",
    "runtime": "Current operating value", "user_defined": "User-defined",
}


@dataclass(frozen=True, slots=True)
class ParameterSemantics:
    label: str
    category: str
    category_label: str
    unit: str
    source_kind: str
    source_label: str
    source_note: str
    description: str


_ANNULAR = {"magnetic_excitation_coil", "magnetic_lens_housing", "magnetic_lens_yoke"}
_POLE_STYLES = {"", "embedded_hourglass_bore", "tapered_bore_pole",
                "objective_vertical_back_inserted_shank_tapered_nose", RECESSED_OBJECTIVE_STYLE}
_ENVELOPE_FIELDS = {"length_mm", "mechanical_outer_diameter_mm", "mechanical_outer_radius_mm",
                    "outer_diameter_mm", "outer_width_mm", "housing_length_mm", "electrode_length_mm"}
_DIMENSION_WORDS = ("diameter", "radius", "length", "width", "height", "thickness", "gap", "inset", "angle")
_SPLIT_ENDPOINTS = {f"{side}_yoke_{edge}_local_z_mm" for side in ("upper", "lower") for edge in ("start", "end")}
_UNITS = ("mrad", "keV", "eV", "kV", "MHz", "kHz", "Hz", "nm", "um", "mm", "deg", "rad", "m", "s")
# Composite dimensions must be resolved before their last suffix (for example
# a magnetic gradient is T/m, not m). Inverse powers below are field strengths,
# not a blanket interpretation of every name ending in m2 or m3.
_FIELD_UNITS = {"maximum_strength_m2": "m⁻²", "max_strength_m2": "m⁻²", "strength_m2": "m⁻²",
                "maximum_strength_m3": "m⁻³", "max_strength_m3": "m⁻³", "strength_m3": "m⁻³",
                "air_core_reference_ampere_turns": "A·turn", "coil_ampere_turn_fraction": "1",
                "winding_turn_count": "turn", "winding_interval_turn_counts": "turn",
                "winding_wire_cross_section_mm2": "mm²",
                "hexapole_winding_turn_count_per_coil": "turn",
                "hexapole_winding_wire_cross_section_mm2": "mm²",
                "transverse_winding_turn_count_per_coil": "turn",
                "transverse_winding_wire_cross_section_mm2": "mm²",
                "transverse_winding_insulation_thickness_mm": "mm",
                "ideal_transverse_current_matrix_a_per_unit": "A/native unit",
                "ideal_transverse_current_bias_a": "A",
                "ideal_transverse_calibration_reference_energy_ev": "eV",
                "multipole_winding_turn_count_per_coil": "turn",
                "multipole_winding_wire_cross_section_mm2": "mm²",
                "multipole_winding_insulation_thickness_mm": "mm",
                "ideal_multipole_current_matrix_a_per_unit": "A/native coefficient",
                "ideal_multipole_current_bias_a": "A",
                "ideal_multipole_calibration_reference_energy_ev": "eV",
                "ideal_coil_current_a_at_100_percent": "A"}
_DETECTOR_UNITS = {"pixels": "pixels", "stem_fourdstem_charge_spread_sigma_px": "pixels",
    "stem_fourdstem_dark_electrons_per_pixel": "electrons/pixel/exposure",
    "stem_fourdstem_read_noise_electrons_rms": "electrons RMS",
    "stem_fourdstem_saturation_electrons": "electrons",
    "stem_fourdstem_gain_counts_per_electron": "counts/electron",
    "stem_fourdstem_offset_counts": "counts"}
_COMPOSITE_UNITS = (("electrons_per_nm2_s", "electrons/(nm² s)"),
                    ("v_per_m2", "V/m²"), ("v_per_m", "V/m"),
                    ("t_per_m", "T/m"), ("nm_inv", "nm⁻¹"),
                    ("na", "nA"), ("pa", "Pa"), ("mt", "mT"),
                    ("t", "T"), ("a", "A"))
_FRIENDLY = {
    "multipole_winding_turn_count_per_coil": "Multipole winding turns per coil",
    "multipole_winding_wire_cross_section_mm2": "Multipole conductor metal area",
    "multipole_winding_insulation_thickness_mm": "Multipole wire insulation radial thickness",
    "multipole_winding_basis_angle_deg": "Multipole winding angular datum",
    "multipole_winding_wire_diameter_mm": "Derived multipole conductor diameter",
    "ideal_multipole_current_matrix_a_per_unit": "Fixed multipole current calibration matrix",
    "ideal_multipole_current_bias_a": "Signed multipole current bias",
    "ideal_multipole_calibration_reference_energy_ev": "Multipole calibration reference energy",
    "ideal_multipole_calibration_reference_orientation_rad": "Multipole calibration reference orientation",
    "transverse_winding_turn_count_per_coil": "Transverse winding turns per coil",
    "transverse_winding_wire_cross_section_mm2": "Transverse conductor metal area",
    "transverse_winding_insulation_thickness_mm": "Wire insulation radial thickness",
    "transverse_winding_wire_diameter_mm": "Derived transverse conductor diameter",
    "ideal_transverse_current_matrix_a_per_unit": "Fixed current calibration matrix",
    "ideal_transverse_current_bias_a": "Signed coil-channel current bias",
    "ideal_transverse_calibration_reference_energy_ev": "Current calibration reference energy",
    "hexapole_winding_turn_count_per_coil": "Radial winding turns per coil",
    "hexapole_winding_wire_cross_section_mm2": "Radial winding conductor metal area",
    "hexapole_winding_wire_diameter_mm": "Derived radial-winding conductor diameter",
    "winding_turn_count": "Conductor turn count",
    "winding_interval_turn_counts": "Fixed turns per occupied axial interval",
    "winding_wire_diameter_mm": "Conductor wire diameter",
    "winding_wire_cross_section_mm2": "Conductor metal cross-section per turn",
    "ideal_coil_current_a_at_100_percent": "Conductor current at 100% excitation",
    "air_core_reference_ampere_turns": "Reference ampere-turns at 100% excitation",
    "coil_ampere_turn_fraction": "Fraction of the owner's ampere-turns",
    "gaussian_reference_coil_length_mm": "Reference winding length for Gaussian scaling",
    "rear_gun_role": "Rear gun component role",
    "rear_voltage_reference": "Rear electrode voltage reference",
    "rear_bias_kv": "Rear electrode voltage relative to reference",
    "counterbore_depth_mm": "Extractor recess depth",
    "counterbore_diameter_mm": "Extractor recess diameter",
    "voltage_start_reference": "Accelerator divider start electrode",
    "mechanical_inner_diameter_mm": "Material inner diameter",
    "mechanical_outer_diameter_mm": "Material outer diameter",
    "mechanical_bore_diameter_mm": "Existing bore diameter",
    "vacuum_inner_diameter_mm": "Beam passage (vacuum)",
    "plate_thickness_mm": "Aperture plate thickness",
    "radial_thickness_mm": "Radial thickness",
    "winding_layer_thickness_mm": "Winding layer radial thickness",
    "winding_angular_span_deg": "Winding loop angular span",
    "winding_band_width_mm": "Winding end and side band width",
    "winding_support_carrier_thickness_mm": "Flexible winding carrier thickness",
    "winding_support_cover_thickness_mm": "Winding radial insulation thickness",
    "winding_support_end_cover_thickness_mm": "Winding axial end insulation thickness",
    "deflector_support_radial_skin_thickness_mm": "Rigid carrier radial insulation thickness",
    "deflector_support_casing_thickness_mm": "Insulating casing radial thickness",
    "deflector_support_pin_diameter_mm": "Insulating registration pin diameter",
    "deflector_support_registration_slot_sweep_deg": "Registration slot centreline sweep",
    "length_mm": "Length", "percent": "Excitation strength",
    "tip_radius_nm": "Apex curvature radius",
    "tip_cone_half_angle_deg": "Tip cone half-angle",
    "emission_cap_half_angle_deg": "Emitting surface cap half-angle",
    "emission_maximum_angle_deg": "Maximum emission angle from local normal",
    "emission_flux_electrons_per_nm2_s": "Emitted electrons per unit area per second",
    "default_voltage_kv": "Default electrode voltage difference",
    "default_high_tension_kv": "Default accelerating voltage",
}


def parameter_unit(path):
    """Unit of an actual field, including nested CAD vector/list elements."""
    words = [word for word in _relative_path(path) if isinstance(word, str)]
    if "scale_xy" in words:
        return "×"
    for word in reversed(words):
        if word in _DETECTOR_UNITS:
            return _DETECTOR_UNITS[word]
        if word in _FIELD_UNITS:
            return _FIELD_UNITS[word]
        for suffix, unit in _COMPOSITE_UNITS:
            if word.endswith("_" + suffix):
                return unit
        if word.endswith("_ev"):
            return "eV"
        if word.endswith("_kv"):
            return "kV"
        for unit in _UNITS:
            if word.endswith("_" + unit):
                return "µm" if unit == "um" else unit
        if word == "percent" or word.endswith("_percent"):
            return "%"
    return ""


def _relative_path(path):
    return path[2:] if path and path[0] in {"parts", "runtime", "derived"} else path


def _label(field, path):
    label = _FRIENDLY.get(field)
    if label is None:
        label = re.sub(r"_(?:mrad|nm|um|mm|deg|rad|m|s)$", "", str(field)).replace("_", " ").capitalize()
    for index in path[3:]:
        if isinstance(index, int) and not isinstance(index, bool):
            label += f" [{index}]"
    return label


def _meaning(part, path, by_key):
    relative = _relative_path(path)
    field = str(relative[0]) if relative else ""
    label = _label(field, ("parts", "", *relative))
    profile = part.get("mechanical_profile", "")
    if (field.startswith('multipole_winding_') or field.startswith('hexapole_winding_')
            and 'multipole_winding_geometry_status' in part):
        return label,'physical',(
            'Explicit engineering multipole winding input with fixed integer turns, conductor metal area, '
            'radial insulation and angular datum. Probe HP retains its original canonical per-coil N and '
            'metal-area records. Every requested closed turn and remaining insulation fits the existing '
            'allocation. Auxiliary winding-only CAD preserves transport clipping and existing cartridge '
            'ownership. At fixed N, current and envelope, wire-size edits do not change the homogenized '
            'ideal source. Turn or angular edits retain numerical current gains and mark the old fit stale.')
    if field.startswith('ideal_multipole_'):
        return label,'physical',(
            'Fixed signed current calibration for a vacuum multipole winding. The matrix maps native '
            'quadrupole coefficients in m^-2 or hexapole coefficients in m^-3 to amperes, plus a separately '
            'entered current bias. Main HP1/HP2 uses one winding channel; auxiliary normal/skew windings '
            'use two independent channels. Reference energy and orientation document the original fit '
            'without rescaling current. No magnetic material, current refit or optical retuning is inferred.')
    if field.startswith('transverse_winding_'):
        return label,'physical',(
            'Explicit physical winding inputs for identified deflector or material stigmator coils. '
            'Fixed integer turns and conductor metal area define closed wire turns and automatically '
            'filled remaining insulation inside the original envelope. Diameter is calculated, and '
            'insulation thickness is radial. Known turns may retain bulk CAD while metal area is unknown. '
            'The ideal current source uses the whole homogenized pack, excluding magnetic cores and '
            'pole pieces. Geometry or turn edits do not automatically refit currents or retune optics.')
    if field.startswith('ideal_transverse_'):
        return label,'physical',(
            'Fixed electrical calibration for the ideal vacuum transverse-coil source. The complete '
            'matrix maps existing column deflector commands in mrad or stigmator commands in percent '
            '(gun deflector: mT; gun stigmator: normal/skew gradients in T/m) to '
            'signed physical channel currents; a separately entered bias adds amperes. '
            'Channels follow physical station order, X then Y. The reference energy records the '
            'original calibration, not a query-time rescaling. A manual edit does not repeat the '
            'field fit, change native optical commands or model magnetic materials and heating.')
    if field.startswith('hexapole_winding_'):
        return label, 'physical', (
            'CAD-only inputs for six radial-axis HP1/HP2 windings. Known per-coil integer turns may '
            'retain bulk CAD while conductor metal area is unknown. Turns stay fixed after authoring; '
            'geometry changes do not re-estimate them. Per-coil turns and '
            'conductor metal area define rounded rectangular turns and automatic remaining insulation inside '
            'the original winding bounds. Diameter is calculated. These are engineering inputs, not measured '
            'or OEM windings; native multipole strength and current calibration remain separate and unchanged.')
    if field == "winding_turn_count":
        return label, "physical", (
            "Fixed integer conductor turn count in the existing round-winding envelope. "
            "Known turns do not require a guessed wire diameter. Detailed wire CAD is available only when wire size is supplied. "
            "Together with the owner's declared conductor current this determines ampere-turns in the ideal prescribed-current field model. "
            "Calibrate current rather than changing a photo-constrained turn count. This is not a current/thermal rating.")
    if field == "winding_interval_turn_counts":
        return label, "physical", (
            "Fixed conductor turns for each original current-bearing axial interval, in increasing Z order. "
            "Positive integer entries sum to the total turn count and remain fixed when interval lengths change. "
            "Each region carries its declared turns times current; no turns are redistributed to fit a field target. "
            "Photo estimates and symmetry assumptions retain their separate provenance.")
    if field == "winding_wire_diameter_mm":
        return label, "physical", (
            "Equivalent circular conductor diameter, excluding insulation. Existing diameter inputs remain readable; "
            "the winding editor derives diameter from the user's conductor metal cross-section. "
            "The turn grid must fit the existing winding inner/outer diameters and current-bearing axial intervals. "
            "Wire size changes conductor/insulation geometry, not that envelope or the ideal field's occupied cross-section. "
            "No resistance, heating or wire current rating is inferred.")
    if field == "winding_wire_cross_section_mm2":
        return label, "physical", (
            "Bare conductor metal cross-sectional area of one circular turn, in square millimetres. "
            "This excludes insulation and is distinct from the winding-pack area allocated per turn. "
            "The diameter is derived from area; remaining allocated volume is filled automatically with insulation. "
            "Turn count and conductor area must fit the unchanged inner/outer and axial envelopes; impossible packing is rejected. "
            "At fixed turns and current this changes CAD packing, not the ideal field. No thermal or current rating is inferred.")
    if field == "ideal_coil_current_a_at_100_percent":
        return label, "operating", (
            "Explicit reference conductor current at 100% lens excitation for Ideal prescribed-current coils in vacuum. Actual signed current scales with "
            "the existing excitation percentage and polarity; configured turns times current determine ampere-turns. "
            "Explicit turn/current configuration replaces the legacy total-ampere-turn fallback without compressing "
            "the ideal current-bearing envelope. Other field models retain their own current/calibration rules. "
            "This is a prescribed-current input, not a solved supply or thermal limit.")
    if field == "air_core_reference_ampere_turns":
        return label, "operating", (
            "Total turns times conductor current at 100% excitation for this round-lens owner. "
            "The ideal prescribed-current coil model multiplies this reference by excitation and polarity, then distributes it among "
            "the owned windings when no explicit turn/current configuration is selected. This legacy fallback "
            "does not specify turns and amperes separately or establish a thermal rating.")
    if field == "coil_ampere_turn_fraction":
        return label, "operating", (
            "Positive share of the independently driven owner's total ampere-turns. Owned winding shares sum to one; "
            "separate axial segments divide that same share by occupied cross-section, without duplicating current. "
            "In the ideal prescribed-current field model this legacy share is inactive when explicit owner current "
            "and winding turn counts define each coil's ampere-turns.")
    if field == "gaussian_reference_coil_length_mm":
        return label, "operating", (
            "Frozen denominator of the optional actual/reference current-bearing winding-length rule for Gaussian a. "
            "It is a model reference, not the current coil dimension or an independently measured field width.")
    if field in {"rear_voltage_reference", "rear_bias_kv"}:
        return label, "operating", (
            "Explicit rear-electrode connection: tip, extractor, gun_lens or ground, plus bias in kV. "
            "Conductors participate in curved-tip fields; the Flat infinite cathode omits them. "
            "EEE wiring is an editable engineering assumption; a contacting holder must stay at tip potential.")
    if field == "rear_gun_role":
        return label, "physical", (
            "Rear axisymmetric hardware at negative Z; the tip apex remains Z=0. "
            "EEE and holder enter curved-tip electrostatics. Insulators provide geometry and interception only.")
    if field in {"counterbore_depth_mm", "counterbore_diameter_mm"}:
        return label, "physical", (
            "Tip-facing extractor recess. The small-hole entrance is start Z plus recess depth; "
            "the remaining thickness is the aperture web. The same contour sets CAD, the conducting "
            "electrostatic boundary and electron interception; no angular cutoff is imposed.")
    if field == "voltage_start_reference":
        return label, "operating", (
            "Electrode connected to the accelerator divider start: gun_lens or extractor. "
            "Stage fractions interpolate from this potential to the final anode at ground; "
            "high tension is the total tip-to-ground voltage, including extraction.")
    from temsim.dimension_edit_policy import dimension_reference_reason
    if reason := dimension_reference_reason(part, field):
        if part.get("geometry_template_key", part.get("key")) == "feg_extractor":
            if field == "optical_reference_local_z_mm":
                return "Extraction plane reference", "placement", reason
            if field == "vacuum_inner_diameter_mm":
                return "Extractor minimum-passage reference", "vacuum", reason
        if field == "length_mm":
            reference_label = ("Mechanism envelope length" if part.get("aperture_plate_form") == "perforated_strip" else
                "Installation/field allocation length" if profile in DEFLECTOR_COIL_PROFILES and "winding_axial_length_mm" in part else
                "Envelope length")
            return reference_label, "envelope", reason
        return ("Reference pole gap" if field == "pole_gap_mm" else
                "Carrier bore reference" if "bore" in field else
                "Mechanism envelope diameter" if field in {"mechanical_outer_diameter_mm", "outer_diameter_mm"} else
                "Reference cone angle"), "envelope" if field in {"mechanical_outer_diameter_mm", "outer_diameter_mm"} else "unknown", reason
    if (field.startswith(("gun_stigmator_", "mechanical_cartridge_", "stigmator_support_", "aperture_support_", "mechanical_c3_end_", "mechanical_adapter_end_", "mechanical_tl22_end_", "mechanical_transfer_end_", "mechanical_objective_upper_end_", "hexapole_support_"))
            and field.endswith(("_mm", "_deg"))):
        return label, "physical", (
            "Explicit engineering dimension of the reconstructed internal material. "
            "Updates the shared native 3D and STEP geometry; it is not an OEM measurement. "
            "These construction details do not recalibrate the existing field, change its "
            "effective length, or replace the separately declared vacuum passage.")
    if field in {"deflector_support_radial_skin_thickness_mm", "deflector_support_casing_thickness_mm",
                 "deflector_support_pin_diameter_mm", "deflector_support_registration_slot_sweep_deg"}:
        return label, "physical", (
            "Dimension of the rigid insulating carrier assembly. Pins register the two coil carriers internally; "
            "external host fastening is not specified. Mechanical edits do not recalibrate optical fields.")
    if path and path[0] not in {"parts", "runtime", "derived"}:
        last_field = next((item for item in reversed(path) if isinstance(item, str)), "Parameter")
        label = _label(last_field, ("parts", "", last_field, *[item for item in path if isinstance(item, int)]))
        if path[0] == "geometry" and last_field in _ENVELOPE_FIELDS:
            return label, "envelope", "Module-level allocation or outer envelope, not the dimensions of its individual material parts."
        if path[0] == "geometry" and "vacuum" in last_field:
            return label, "vacuum", "Module-level vacuum tube/passage constraint, separate from an individual part's material body or working aperture."
        if path[0] == "ports" and last_field.endswith("_z_mm"):
            return label, "placement", "Module-local entrance/exit connection reference coordinate."
        return label, "unknown", "Module/document parameter; its material meaning is not established by the part geometry rules."
    if path and path[0] == "runtime":
        key = str(part.get("key", path[1] if len(path) > 1 else ""))
        if key == "nanopulser_deflector" and field == "azimuth_deg":
            return "Plate / field azimuth about column Z", "placement", (
                "Shared mechanical plate and electric-deflection azimuth. Zero points along column +X; "
                "positive angles turn toward +Y about +Z. Rotates CAD and the integrated field together; "
                "not a field-only error or an optical-axis tilt.")
        if key == "nanopulser_deflector" and field == "voltage_v":
            return "Positive minus negative plate voltage", "operating", (
                "Potential of the local +X plate minus the local -X plate. Positive voltage deflects "
                "electrons toward local +X, rotated by the shared column azimuth.")
        if key == "energy_filter" and field in {"alignment_x_mrad", "alignment_y_mrad"}:
            axis = "X" if field == "alignment_x_mrad" else "Y"
            return f"Entrance alignment: column {axis}", "operating", (
                f"Added entrance slope toward column +{axis}, in mrad. Column +Z is downstream. "
                "The CAD and field solver share a proper branch rotation; exit local X is dispersive "
                "and maps to column -Z after a 90-degree bend, while local Y remains column +Y.")
        return label, "operating", "Current operating state or limit. It is separate from the saved mechanical body dimensions."
    if path and path[0] == "derived" and field == "radial_thickness_mm":
        return "Radial thickness", "physical" if profile in _ANNULAR else "unknown", (
            "Derived radial thickness = (mechanical_outer_diameter_mm − mechanical_inner_diameter_mm) / 2; no independent thickness field is stored.")
    from temsim.part_model_apertures import has_optional_plate_thickness
    if field == "plate_thickness_mm" and has_optional_plate_thickness(part):
        return "Aperture plate thickness (0 = unspecified)", "physical", (
            "Actual material thickness along the aperture's local axis, in mm. "
            "0 means unspecified; no physical thickness is supplied by default. "
            "Enter a positive actual plate thickness for tilted coherent propagation. "
            "The same thickness value controls the nominal 3D plate and effective tilted ray/wave interception. "
            "Optical tilt does not rotate the plate's mechanical envelope; "
            "the zero-length placement reference and schematic display thickness are not substitutes.")
    from temsim.lens_pose import PHYSICAL_POSE_FIELDS, supports_physical_lens_pose, physical_pose_description, pose_control_label
    if field in PHYSICAL_POSE_FIELDS and supports_physical_lens_pose(part):
        label, _unit = pose_control_label(field)
        return label, "placement", physical_pose_description(part)
    if field == "model_3d":
        name = next((str(item) for item in reversed(path[3:]) if isinstance(item, str)), "model")
        if supports_physical_lens_pose(part) and len(relative) >= 3 and relative[1] == "transform" and relative[2] in {"offset_mm", "rotation_deg"}:
            return "Component alignment: " + name.replace("_", " "), "placement", (
                "Legacy alignment input, exposed through the existing scalar controls: "
                "offsets in mm move the mechanical and optical reference positions together; "
                "rotations in mrad tilt only the optical action and reference planes. "
                "Optical tilt does not rotate the CAD material envelope.")
        return "3D model: " + _label(name, ("parts", "", name, *[p for p in path[3:] if isinstance(p, int)])), "cad", (
            "Explicit user CAD base, transform or Boolean feature. This controls the 3D solid; the optical/magnetic solver does not infer a new field law from the mesh.")
    if part.get("key") in {CAMERA, FLUORESCENT_SCREEN, *STEM_DETECTOR_KEYS}:
        # Local import avoids the registry/unit adapter cycle. Structural and
        # runtime editors share explanations without introducing new controls.
        from temsim.parameter_registry import parameter_definition
        definition = parameter_definition(part.get("key"), field)
        if definition is not None:
            category = ("physical" if field in {"outer_width_mm", "inner_diameter_mm"}
                        else "envelope" if field == "external_envelope"
                        else "unknown" if definition.category == "metadata"
                        else "operating" if definition.category == "structural"
                        else definition.category)
            return label, category, definition.description
    if part.get("tip_particle_model") == "curved_surface_particles":
        if field in {"tip_radius_nm", "tip_cone_half_angle_deg", "length_mm", "tip_shape"}:
            return label, "physical", "Actual conducting tip geometry used by the 3D shape, emission surface and extraction-field boundary. Apex curvature is 1/radius; the apex remains at the gun origin."
        if field == "tip_material":
            return label, "material", "Tip material identity. Electron flux is prescribed; this model does not infer tunnelling current from the material."
        if field.startswith("emission_"):
            return label, "operating", "Emission at the physical tip. Flux is per curved surface area; electron direction is relative to each local outward normal. Total current follows area times flux times electron charge."
        if field.startswith("tip_field_"):
            return label, "operating", "Numerical resolution or tolerance of the connected tip-to-anode electrostatic solution."
        if field == "outer_diameter_mm":
            return "Derived shank diameter", "physical", "Computed from curvature radius, cone angle and physical length; edit those independent dimensions."
    if part.get("key") == "feg_accelerator" and field == "electrode_thickness_mm":
        return "Accelerator ring thickness", "physical", "Axial thickness of each annular accelerator electrode, consumed by the connected electrostatic boundary. The overall assembly envelope is separate."
    if field in {"material_class", "material_regions", "bh_material", "relative_permeability"} or field.startswith("material_"):
        if field == "material_intervals_mm":
            return label, "physical", "Explicit axial material intervals, which define occupied material independently of the display envelope."
        return label, "material", "Material identity or constitutive response. Material settings are distinct from geometric dimensions and operating excitation."
    if profile == "magnetic_pole_piece" and field in {
            "pole_vacuum_connector_outer_diameter_mm", "pole_vacuum_connector_axial_length_mm"}:
        return label, "physical", "Outer material dimension of the pole's vacuum connector; this is not the diameter of the vacuum passage inside it."
    if "vacuum" in field or field in {"column_inner_diameter_mm", "clearance_to_vacuum_mm"}:
        return label, "vacuum", "Vacuum/beam-passage geometry or clearance. This is not automatically the material bore or the working aperture opening."
    if field in _SPLIT_ENDPOINTS:
        return label, "physical", "Parent-owned boundary of an actual upper/lower yoke material interval; the associated coil interval also includes its axial inset."
    if field == "mechanical_coil_axial_inset_mm":
        return label, "physical", "Axial inset of the winding from its parent material boundaries; it changes the actual coil material interval."
    if (field.startswith("local_") and field.endswith("_z_mm")) or "reference_local_z" in field or "centers_local_z" in field:
        return label, "placement", "Module-local axial placement or optical reference coordinate. A coordinate is not a measured material thickness."
    if field.startswith("offset_") or field in {"detector_axis_rotation_deg", "point_spread_rotation_deg"}:
        return label, "operating", "Configured operating alignment/response setting, separate from the nominal material envelope."
    if part.get("aperture_plate_form") == "perforated_strip":
        from temsim.part_model_apertures import dimension_semantics
        known = dimension_semantics(part)
        if field in known:
            label, description = known[field]
            category = ("envelope" if field in {"length_mm", "mechanical_outer_diameter_mm", "outer_diameter_mm"}
                        else "operating" if field == "maximum_radius_mm" else "physical")
            return label, category, description
        if field in {"radius_mm", "diameter_mm", "opening_diameter_mm", "aperture_radius_mm", "reference_operating_diameter_mm"}:
            return label, "operating", "Working or reference aperture opening; it is separate from the carrier bore and the unspecified transverse strip outline."
        if "hole_diameters" in field:
            return label, "physical", "Diameter of an explicitly listed aperture hole. The file does not thereby establish the transverse plate outline or hole locations."
    if field.startswith(("effective_", "maximum_", "minimum_", "reference_operating_", "field_", "point_spread_")) or field in {
            "percent", "cs_mm", "cc_mm", "reference_bore_diameter_mm", "reference_bore_radius_mm", "current_a", "voltage_kv"}:
        return label, "operating", "An operating/reference limit, response parameter or field-model quantity; it does not directly specify a material boundary."
    parent = by_key.get(part.get("parent_key"), {})
    split = (profile in {"magnetic_excitation_coil", "magnetic_lens_yoke"}
             and (part.get("parent_key") == "objective_lens" or _SPLIT_ENDPOINTS.intersection(parent)
                  or "material_intervals_mm" in part))
    if profile in _ANNULAR:
        if field == "length_mm" and split:
            return "Display-envelope length", "envelope", "Display allocation only: actual material occupies separate parent-owned or explicit intervals. Edit those interval endpoints to change the solid length."
        if field in {"length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "magnetic_radial_profile_mm"}:
            return label, "physical", "Dimension of the configured material body; its real inner/outer surfaces remain separate from the vacuum-passage constraint."
    if profile in DEFLECTOR_COIL_PROFILES:
        if part.get("winding_support_profile") and field in {"mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"}:
            return label.replace("Material", "Copper"), "physical", (
                "Diameter of the copper winding envelope. The carrier extends inward and the radial cover outward; "
                "the full laminate material envelope includes their separately editable thicknesses.")
        explanations = {
            "winding_axial_length_mm": "Actual copper winding length about the saved winding centre. It does not change the independent effective thickness or analytical field support.",
            "winding_support_carrier_thickness_mm": "Radial thickness of each carrier in the equivalent flexible laminate. The interlayer bond fills the remaining positive gap; host mounting is unverified.",
            "winding_support_cover_thickness_mm": "Radial thickness of each dielectric cover in the bulk laminate equivalent. It increases the full material outer radius; it does not model individual conductor sidewall insulation.",
            "winding_support_end_cover_thickness_mm": "Engineering axial insulation on the selected end(s) of the configured DPH/DP copper winding. It follows the actual copper length and extends the material stack; the complete stack must clear neighbouring parts. It does not change optical length, winding centres or field calibration and is not verified OEM packaging.",
            "winding_layer_thickness_mm": "Radial material thickness of each of the two winding layers. Their remaining radial gap must be positive.",
            "winding_angular_span_deg": "Full angular span of each loop about the local axis. Opposite loops must stay separated; the X/Y pairs are rotated by 90 degrees.",
            "winding_band_width_mm": "Axial end-band width and circumferential side-band width at the layer mean radius. Both must leave an open loop window.",
        }
        if field in explanations or field in {"length_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"}:
            return label, "physical", (explanations.get(field, "Dimension of the configured winding material envelope.") +
                " This changes the reconstructed solid, not the linked analytical field strength; it is not an OEM winding measurement.")
    if profile == "magnetic_pole_piece" and part.get("pole_piece_geometry_style", "") in _POLE_STYLES:
        if field == "pole_rear_recess_depth_mm":
            return label, "physical", "Axial depth of the annular well from the rear flange face to its floor around the tubular stem."
        if field == "pole_root_fillet_radius_range_mm":
            return label, "unknown", "Reference range only. The current pole solid has sharp shoulders; this metadata does not generate a fillet or select a physical fillet radius."
        if field == "magnetic_radial_profile_mm" or any(word in field for word in _DIMENSION_WORDS):
            return label, "physical", "Configured pole material geometry, including its open bore, shoulder, mounting shank or nose. A geometric meaning does not establish measurement provenance."
    if profile == "magnetic_lens_assembly":
        if field in {"pole_gap_mm", "mechanical_coil_radial_thickness_mm"}:
            return label, "physical", "Assembly-owned spacing or material reconstruction dimension, used by its constituent magnetic parts."
        if field in _ENVELOPE_FIELDS or field in {"bore_diameter_mm", "mechanical_clear_bore_diameter_mm"}:
            return label, "envelope", "Assembly allocation/reference envelope; the housing, winding, yokes and pole pieces define the individual material solids."
    if profile in {"circular_aperture", "fixed_differential_pumping_aperture", "vacuum_liner", "electrostatic_bias_tube"}:
        if field == "length_mm" and part.get(field) == 0:
            return "Reference-plane axial extent", "placement", "A zero-thickness reference plane, not an assigned physical plate thickness."
        if any(word in field for word in _DIMENSION_WORDS):
            return label, "physical", "Explicit dimension of the configured aperture/liner/tube body, distinct from the shared vacuum constraint."
    if field in _ENVELOPE_FIELDS:
        return ("Envelope length" if field == "length_mm" else label.replace("Material", "Envelope")), "envelope", (
            "The existing preview uses this value as a component allocation or outer envelope. Its internal/non-axisymmetric material construction is not established by that surface.")
    return label, "unknown", "This field's physical meaning is not sufficiently defined by the supported geometry rules; review its schema and source before treating it as a material dimension."


def _source_from_text(status, note, *, explicit_field):
    status, note = str(status or ""), str(note or "")
    combined = (status + " " + note).lower()
    normalized = combined.replace("_", " ").replace("-", " ")
    topology_only = any(word in normalized for word in ("topology only", "not dimensionally", "uncalibrated", "not calibrated", "photo only", "photograph only"))
    if topology_only:
        return "unspecified", "The stored evidence concerns topology or an uncalibrated image, not this dimension. " + note
    estimated = any(word in combined for word in ("provisional", "reconstruction", "engineering_assumption", "engineering assumption",
                                                   "simulator assumption", "non-oem simulator", "estimated", "schematic", "assumed", "spacing-derived"))
    if estimated:
        return "estimated", note or f"Stored status: {status}."
    if explicit_field and note.strip():
        if any(word in normalized for word in ("not measured", "unmeasured", "measurement unknown", "not documented", "undocumented")):
            return "unspecified", "The stored declaration does not establish a measured/documented dimension. " + note
        if any(word in status.lower() for word in ("measured", "metrology", "calibrated_measurement")):
            return "measured", note
        if any(word in status.lower() for word in ("documented", "published_dimension", "oem_drawing", "manufacturer_dimension")):
            return "documented", note
        if "user_defined" in status.lower():
            return "user_defined", note
    return "unspecified", ("Stored source does not establish this field's dimension provenance. " + note).strip()


def _source(part, path, category):
    if path and path[0] == "runtime":
        return "runtime", "Read from the current operating state, not a mechanical measurement or a new TOML dimension."
    if len(path) > 2 and path[2] == "model_3d":
        return "user_defined", "Explicit CAD geometry authored in model_3d; no independent dimensional verification is implied."
    if path and path[0] == "derived":
        if len(path)>2 and path[2]=='multipole_winding_wire_diameter_mm':
            from temsim.multipole_winding import area_field
            kind,note=_source(part,('parts',part.get('key',''),area_field(part)),category)
            return kind,'Computed from multipole conductor metal area; no independent diameter input. '+note
        if len(path)>2 and path[2]=='transverse_winding_wire_diameter_mm':
            kind,note=_source(part,('parts',part.get('key',''),'transverse_winding_wire_cross_section_mm2'),category)
            return kind,'Computed from transverse conductor metal area; no independent diameter input. '+note
        if len(path)>2 and path[2]=='hexapole_winding_wire_diameter_mm':
            kind,note=_source(part,('parts',part.get('key',''),'hexapole_winding_wire_cross_section_mm2'),category)
            return kind,'Computed from radial-winding conductor metal area; no independent diameter input. '+note
        if (len(path) > 2 and path[2] == "winding_wire_diameter_mm"
                and "winding_wire_cross_section_mm2" in part):
            source_kind, note = _source(part, ("parts",part.get("key",""),"winding_wire_cross_section_mm2"), category)
            return source_kind, "Computed equivalent diameter from the conductor metal cross-section; no independent diameter input. " + note
        return "unspecified", "Derived from mechanical_inner_diameter_mm and mechanical_outer_diameter_mm; no independent thickness measurement or source is declared."
    relative = _relative_path(path)
    field = str(relative[0]) if relative else ""
    metadata = part.get("parameter_metadata", {})
    declared_field = metadata.get(field) if isinstance(metadata, Mapping) else None
    if field.startswith(('multipole_winding_','ideal_multipole_')) and field in part and not isinstance(declared_field,Mapping):
        note_field=('multipole_winding_source_note' if field.startswith('multipole_winding_')
                    else 'ideal_multipole_calibration_source_note')
        return 'user_defined',str(part.get(note_field,'Explicit engineering multipole winding/current input; not OEM or measured data.'))
    if field.startswith(('transverse_winding_','ideal_transverse_')) and field in part and not isinstance(declared_field,Mapping):
        note_field=('transverse_winding_source_note' if field.startswith('transverse_winding_')
                    else 'ideal_transverse_calibration_source_note')
        return 'user_defined',str(part.get(note_field,'Explicit engineering winding/current input; not measured or OEM data.'))
    if (field in {"winding_turn_count", "winding_interval_turn_counts", "winding_wire_diameter_mm",
                  "winding_wire_cross_section_mm2", "ideal_coil_current_a_at_100_percent"}
            and field in part and not isinstance(declared_field, Mapping)):
        return "user_defined", "Explicit winding input supplied by the user; no OEM specification or independent measurement is implied."
    if field.startswith('hexapole_winding_') and field in part and not isinstance(declared_field,Mapping):
        return 'user_defined',str(part.get('hexapole_winding_source_note','Explicit CAD-only radial winding input; not measured or OEM data.'))
    if field == "air_core_reference_ampere_turns" and not isinstance(declared_field, Mapping):
        status, note = part.get("air_core_reference_status", ""), part.get("air_core_reference_source", "")
        if str(status).startswith("engineering_"):
            return "estimated", str(note) or "Explicit engineering reference; not an OEM or measured current calibration."
        return _source_from_text(status, note, explicit_field=True)
    if field == "gaussian_reference_coil_length_mm" and not isinstance(declared_field, Mapping):
        return "estimated", "Frozen engineering winding-length reference for an optional scaling rule; not a measured magnetic-field extent."
    if isinstance(metadata, Mapping):
        candidates = [".".join(map(str, relative)), field]
        for key in dict.fromkeys(candidates):
            declared = metadata.get(key)
            if not isinstance(declared, Mapping):
                continue
            kind, note = declared.get("source_kind"), declared.get("source_note", "")
            if not isinstance(kind, str) or kind not in SOURCE_LABELS or not isinstance(note, str) or (kind != "unspecified" and not note.strip()):
                return "unspecified", "A field-level source declaration is incomplete or unsupported; source_kind and a meaningful source_note are required."
            normalized_note = note.lower().replace("_", " ").replace("-", " ")
            if kind in {"measured", "documented"} and any(word in normalized_note for word in (
                    "topology only", "uncalibrated", "not dimensionally", "not calibrated", "photo only", "photograph only",
                    "not measured", "unmeasured", "not documented", "undocumented")):
                return "unspecified", "Field metadata cites uncalibrated/topology evidence rather than a dimension. " + note
            return kind, note or "Field metadata explicitly leaves its source unspecified."
    if field == "winding_support_end_cover_thickness_mm" and category == "physical":
        return _source_from_text(part.get("winding_support_end_cover_geometry_status"),
                                 part.get("winding_support_end_cover_source_note"), explicit_field=False)
    stem = re.sub(r"_(?:mrad|nm|um|mm|deg|rad|m|s)$", "", field)
    prefixes = list(dict.fromkeys((field, stem, re.sub(r"_(?:diameter|radius|length|width|height|thickness)$", "", stem))))
    if field == "length_mm" and part.get("mechanical_profile") == "fixed_differential_pumping_aperture":
        prefixes.insert(0, "mechanical_axial_thickness")
    for prefix in prefixes:
        status, note = part.get(prefix + "_status"), part.get(prefix + "_source")
        if status is not None or note is not None:
            return _source_from_text(status, note, explicit_field=True)
    if field.startswith("deflector_support_") and category == "physical":
        return _source_from_text(part.get("deflector_support_geometry_status"),
                                 part.get("deflector_support_source_note"), explicit_field=False)
    if field.startswith("gun_stigmator_") and category == "physical":
        return _source_from_text(part.get("gun_stigmator_geometry_status"),
                                 part.get("gun_stigmator_source_note"), explicit_field=False)
    if field.startswith("mechanical_cartridge_") and category == "physical":
        return _source_from_text(part.get("mechanical_cartridge_geometry_status"),
                                 part.get("mechanical_cartridge_geometry_source"), explicit_field=False)
    if field.startswith("stigmator_support_") and category == "physical":
        return _source_from_text(part.get("stigmator_support_geometry_status"),
                                 part.get("stigmator_support_source_note"), explicit_field=False)
    if field.startswith("aperture_support_") and category == "physical":
        return _source_from_text(part.get("aperture_support_geometry_status"),
                                 part.get("aperture_support_source_note"), explicit_field=False)
    if field.startswith("mechanical_c3_end_") and category == "physical":
        return _source_from_text(part.get("mechanical_c3_end_geometry_status"),
                                 part.get("mechanical_c3_end_geometry_source"), explicit_field=False)
    if field.startswith("mechanical_adapter_end_") and category == "physical":
        return _source_from_text(part.get("mechanical_adapter_end_geometry_status"),
                                 part.get("mechanical_adapter_end_geometry_source"), explicit_field=False)
    if field.startswith("mechanical_tl22_end_") and category == "physical":
        return _source_from_text(part.get("mechanical_tl22_end_geometry_status"),
                                 part.get("mechanical_tl22_end_geometry_source"), explicit_field=False)
    if field.startswith("mechanical_transfer_end_") and category == "physical":
        return _source_from_text(part.get("mechanical_transfer_end_geometry_status"),
                                 part.get("mechanical_transfer_end_geometry_source"), explicit_field=False)
    if field.startswith("mechanical_objective_upper_end_") and category == "physical":
        return _source_from_text(part.get("mechanical_objective_upper_end_geometry_status"),
                                 part.get("mechanical_objective_upper_end_geometry_source"), explicit_field=False)
    if field.startswith("hexapole_support_") and category == "physical":
        return _source_from_text(part.get("hexapole_support_geometry_status"),
                                 part.get("hexapole_support_source_note"), explicit_field=False)
    if category in {"physical", "envelope", "placement"}:
        status, note = part.get("mechanical_geometry_status"), part.get("mechanical_geometry_source")
        if status is not None or note is not None:
            return _source_from_text(status, note, explicit_field=False)
    if part.get("mechanical_profile") in DEFLECTOR_COIL_PROFILES and category == "physical":
        return "estimated", ("Generic winding reconstruction dimensions; not OEM winding measurements. " +
                              str(part.get("geometry_source", ""))).strip()
    if part.get("aperture_plate_form") == "perforated_strip" and part.get("aperture_mechanism_evidence_source"):
        return "unspecified", "The aperture evidence identifies a perforated-strip topology, not calibrated dimensions; no field-specific dimension source is declared."
    return "unspecified", "No field-specific dimensional measurement, drawing or estimation source is declared."


def describe_parameter(part: Mapping, path: tuple, *, by_key=None) -> ParameterSemantics:
    """Describe parts/runtime paths, resolving parent-owned paths when supplied."""
    part = part if isinstance(part, Mapping) else {}
    path = tuple(path)
    by_key = by_key if isinstance(by_key, Mapping) else {}
    if len(path) >= 2 and path[0] == "parts" and part.get("key") not in {None, path[1]}:
        part = by_key.get(path[1], {})
    label, category, description = _meaning(part, path, by_key)
    source, note = _source(part, path, category)
    return ParameterSemantics(label, category, CATEGORY_LABELS[category], parameter_unit(path),
                              source, SOURCE_LABELS[source], note, description)
