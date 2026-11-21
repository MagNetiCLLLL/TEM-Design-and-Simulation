"""Persisted material responses for existing mechanical parts and regions.

Assignments select static magnetic constitutive laws, not colours. They do not
change geometry, current settings, vacuum cutoffs, conductivity or temperature.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

from temsim.magnetic_circuits import MAGNETIC_BODIES, is_custom_mechanical_part, optical_owner, part_data
from temsim.magnetic_geometry import objective_layer_intervals_mm
from temsim.magnetic_materials import (
    datasheet_assignment, lens_material_defaults, pinned_datasheet, reference_materials, validate_bh_material,
)
from temsim.mechanical_profiles import MAGNETIC_EXCITATION_COIL, MAGNETIC_LENS_HOUSING
from temsim.component_representation import non_material_role
from temsim.immutable_json import thaw_json


MATERIAL_PROFILES = MAGNETIC_BODIES | {MAGNETIC_EXCITATION_COIL, MAGNETIC_LENS_HOUSING}
REGIONS = frozenset({"body", "upper", "lower"})
POLE_GRADES = ("armco_pure_iron", "vacoflux_50", "permenorm_5000_h2", "mumetall_solid")
NONMAGNETIC_STEEL = "nonmagnetic_stainless_steel"
HEXAPOLE_STAGES = frozenset({"probe_hp1_hexapole", "probe_hp2_hexapole", "image_hp1_hexapole",
                            "image_hp2_hexapole", "image_hp3_hexapole"})
MATERIAL_RESPONSES = {
    **dict.fromkeys(POLE_GRADES, "bh"),
    "femm_pure_iron": "bh", "copper": "nonmagnetic_approximation",
    "aluminum": "nonmagnetic_approximation",
    "nonmagnetic_stainless_steel": "nonmagnetic_approximation", "vacuum": "vacuum",
}


def nonmagnetic_assignment(key, label):
    return dict(schema_version=1, material_key=key, label=label,
                magnetic_response="nonmagnetic_approximation", relative_permeability=1.0,
                source="temsim.physics.nonlinear_magnetostatics: existing nonmagnetic-region law",
                scope="Explicit static nonmagnetic approximation (relative permeability 1), "
                      "using the simulator's existing nonmagnetic-region law. No measured "
                      "susceptibility, electrical conductivity, eddy currents or thermal response.")


def nonmagnetic_tube_material():
    return nonmagnetic_assignment(NONMAGNETIC_STEEL, "Nonmagnetic stainless steel - static approximation")


def material_catalog() -> tuple[dict, ...]:
    """Independent, serializable source snapshots; never infer new properties."""
    defaults = lens_material_defaults()
    iron = dict(schema_version=1, material_key="femm_pure_iron",
                label="Pure iron - FEMM reference", magnetic_response="bh",
                relative_permeability=defaults["linear_material_reference"]["relative_permeability"],
                linear_source=defaults["linear_material_reference"], bh_material=defaults["bh_material"],
                scope="Sourced pure-iron reference for static linear/B-H magnetics; not an OEM or measured-batch assignment. No hysteresis, eddy currents or thermal model.")
    rows = [iron, *(datasheet_assignment(curve) for curve in reference_materials()
                    if curve["key"] in POLE_GRADES)]
    for key, label in (("copper", "Copper - static nonmagnetic approximation"),
                       ("aluminum", "Aluminum - static nonmagnetic approximation"),
                       ("nonmagnetic_stainless_steel", "Nonmagnetic stainless steel - static approximation")):
        rows.append(nonmagnetic_assignment(key, label))
    rows.append(dict(schema_version=1, material_key="vacuum", label="Vacuum - static magnetic response",
                     magnetic_response="vacuum", relative_permeability=1.0,
                     source="Relative permeability of the solver's vacuum reference is one by definition",
                     scope="Vacuum magnetic response for an existing passive region. Does not remove its geometry or change beam-passage cutoffs. Not permitted for driven excitation coils."))
    return tuple(deepcopy(rows))


def _validate_snapshot(snapshot):
    if not isinstance(snapshot, Mapping):
        raise ValueError("Material assignment must contain a material snapshot table")
    result = deepcopy(thaw_json(snapshot))
    if "library_key" in result:
        if set(result) != {"library_key", "library_sha256"} or result["library_key"] not in POLE_GRADES:
            raise ValueError("Invalid pinned magnetic material reference")
        return pinned_datasheet(result["library_key"], result["library_sha256"])
    if type(result.get("schema_version")) is not int or result["schema_version"] != 1:
        raise ValueError("Unsupported part material snapshot schema")
    for field in ("material_key", "label", "scope"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"Material snapshot requires {field}")
    value = result.get("relative_permeability")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("Material relative permeability must be finite and positive")
    response = result.get("magnetic_response")
    if MATERIAL_RESPONSES.get(result["material_key"]) != response:
        raise ValueError("Material key and supported magnetic response must agree")
    if response == "bh":
        result["bh_material"] = validate_bh_material(result.get("bh_material"))
        if result["bh_material"].get("key") != result["material_key"]:
            raise ValueError("Part material key disagrees with its B-H curve")
        source = result.get("linear_source")
        if not isinstance(source, Mapping) or not all(source.get(k) for k in ("source_url", "source_sha256", "source_fields")):
            raise ValueError("Linear part material requires its source reference")
        if source.get("relative_permeability") != value:
            raise ValueError("Linear material response disagrees with its source snapshot")
    elif response in {"nonmagnetic_approximation", "vacuum"}:
        if value != 1.0 or not isinstance(result.get("source"), str) or not result["source"].strip():
            raise ValueError("Nonmagnetic/vacuum assignments require the explicit unit-permeability model and source")
        if "bh_material" in result:
            raise ValueError("A nonmagnetic/vacuum assignment cannot also specify B-H data")
    else:
        raise ValueError("Unsupported part material magnetic response")
    return result


def hardware_regions(row):
    """Independent material domains of the installed native hardware."""
    if row.get("geometry_template_key", row.get("key")) in HEXAPOLE_STAGES:
        if row.get("model_3d", {}).get("base", {}).get("kind", "existing") == "existing":
            return (("pole_pins", "pole_root_sleeves", "return_yoke") if "pole_root_sleeve_depth_mm" in row
                    else ("pole_pins", "return_yoke"))
    if row.get("key") in {"objective_upper_pole", "@vacuum_liner:c1_c2_to_upper_objective"} and "terminal_tube_length_mm" in row:
        return ("terminal_tube",)
    return ()


def material_region_key(row, region):
    """Map the six individual CAD pins or sleeves to their stage's material."""
    domains = hardware_regions(row)
    if region in REGIONS or region in domains:
        return region
    for prefix, group in (("pole_pin_", "pole_pins"), ("pole_root_sleeve_", "pole_root_sleeves")):
        if group in domains and region.startswith(prefix) and region[len(prefix):] in ("1", "2", "3", "4", "5", "6"):
            return group
    return None


def validated_material_regions(part) -> dict:
    row = part_data(part)
    regions = row.get("material_regions", {})
    if not regions and isinstance(regions, Mapping):
        return {}
    domains = set(hardware_regions(row))
    if not isinstance(regions, Mapping) or not set(regions) <= REGIONS | domains:
        raise ValueError(f"{row.get('key', 'Part')}: unsupported material region")
    if row.get("mechanical_profile") not in MATERIAL_PROFILES and set(regions) - {"body"} - domains:
        raise ValueError(f"{row.get('key', 'Part')}: this component supports only a body material definition")
    result = {name: _validate_snapshot(value) for name, value in regions.items()}
    if "terminal_tube" in result and result["terminal_tube"]["magnetic_response"] == "vacuum":
        raise ValueError("The terminal vacuum tube requires a solid wall material")
    if row.get("mechanical_profile") == MAGNETIC_EXCITATION_COIL:
        if any(value["magnetic_response"] != "nonmagnetic_approximation" for value in result.values()):
            raise ValueError(f"{row.get('key', 'Coil')}: excitation coils require a nonmagnetic winding material; vacuum and magnetic coils are unsupported")
    return result


def material_for_region(part, region="body") -> dict | None:
    region_key = material_region_key(part_data(part), region)
    if region_key is None:
        raise ValueError(f"Unsupported material region: {region}")
    regions = validated_material_regions(part)
    if region_key == "terminal_tube":
        return regions.get(region_key)  # The tube is a separate part from the pole.
    return regions.get(region_key, regions.get("body"))


def configured_region_colour(part, region="body", fallback=(0.55, 0.61, 0.69, 1.0)) -> tuple:
    """Shared presentation-only RGBA for saved material-region assignments.

    The palette matches the 3D Parts editor. No assignment means the existing
    mesh colour is retained; this helper does not change any material law.
    """
    if material_region_key(part_data(part), region) is None:
        from temsim.cad_catalog import is_reconstruction_region
        if is_reconstruction_region(part_data(part), region):
            return tuple(fallback)
        # Known Energy Filter patches share their validated body material;
        # display region IDs must not add fictitious solver material domains.
        from temsim.energy_filter_model_3d import energy_filter_material_region
        region = energy_filter_material_region(part_data(part), region)
    assignment = material_for_region(part, region)
    if assignment is None:
        return tuple(fallback)
    palette = {"femm_pure_iron": "8b9bad", "armco_pure_iron": "869bad", "vacoflux_50": "657e9d",
               "permenorm_5000_h2": "8ca99e", "mumetall_solid": "ac99b9", "copper": "c8874e", "aluminum": "b7c5d3",
               "nonmagnetic_stainless_steel": "99aca9", "vacuum": "6db1c4"}
    rgb = palette.get(assignment["material_key"], "87aaa3")
    return (*tuple(int(rgb[index:index + 2], 16) / 255.0 for index in (0, 2, 4)), 1.0)


def material_application_scope(part) -> str:
    """Describe this component's actual solver participation for editors/reports."""
    if non_material_role(part_data(part)):
        return ("Control channel or virtual reference: no independent material body. "
                "Stored material metadata does not create a solid or change its calibrated field.")
    profile = part_data(part).get("mechanical_profile")
    if "pole_pins" in hardware_regions(part_data(part)):
        return ("Assignments identify the materials of the 3D hexapole hardware. "
                "The multipole field model does not solve a nonlinear 3D magnetic core.")
    if is_custom_mechanical_part(part) or profile not in MATERIAL_PROFILES:
        return ("Material definition only: this component has no modeled magnetostatic solid. "
                "The assignment does not change magnetic fields, electrical/thermal response, "
                "beam-passage cutoffs or scattering.")
    if profile == MAGNETIC_EXCITATION_COIL:
        return ("Prescribed-current winding with the explicit static nonmagnetic approximation. "
                "Existing ampere-turns remain unchanged; conductivity, heating and eddy currents are not modeled.")
    return ("Existing body regions participate in static linear/B-H FEM using their assigned response. "
            "Ideal/analytic lens models are unchanged, and earlier imported maps require matching material identity. "
            "Geometry, beam cutoffs, electrical/thermal response and scattering are unchanged.")


def part_material_updates(part, key, region="body") -> dict:
    """Stage one existing region without changing any geometric/operating value."""
    choices = {value["material_key"]: value for value in material_catalog()}
    if key not in choices:
        raise ValueError(f"Unknown material key: {key}")
    return update_material_regions(part, {region: choices[key]})


def update_material_regions(part, changes):
    """Merge edits to one physical part, retaining all untouched source snapshots."""
    row = part_data(part)
    if non_material_role(row):
        raise ValueError("A control channel or virtual reference has no material body to assign")
    regions = deepcopy(thaw_json(row.get("material_regions", {})))
    for region, snapshot in changes.items():
        target = material_region_key(row, region)
        if target is None:
            raise ValueError(f"Unsupported material region: {region}")
        regions[target] = deepcopy(thaw_json(snapshot))
    validated_material_regions({**row, "material_regions": regions})
    return {("parts", row["key"], "material_regions"): regions}


def validate_part_materials(parts) -> None:
    rows = [part_data(part) for part in parts]
    by_key = {row["key"]: row for row in rows}
    for row in rows:
        regions = validated_material_regions(row)
        if not regions:
            continue
        owner = by_key.get(optical_owner(row, by_key), {})
        if owner.get("magnetic_circuit_topology") == "air_core" and any(
            value["magnetic_response"] == "bh" for value in regions.values()
        ):
            raise ValueError(f"{row['key']}: an air-core circuit cannot contain a magnetic material assignment")
        if not (set(regions) & {"upper", "lower"}):
            continue
        parent = by_key.get(row.get("parent_key"), {})
        split = ()
        if row.get("mechanical_profile") in {"magnetic_lens_yoke", MAGNETIC_EXCITATION_COIL}:
            if not is_custom_mechanical_part(row):
                split = objective_layer_intervals_mm(parent, float(parent.get("local_start_z_mm", 0)), row["mechanical_profile"])
            if not split:
                split = row.get("material_intervals_mm", ())
        if len(split) != 2 or "magnetic_radial_profile_mm" in row:
            raise ValueError(f"{row['key']}: upper/lower material assignments require existing split material intervals")


def is_magnetostatic_body(data) -> bool:
    """Include a housing only when an explicit magnetic response is assigned."""
    if non_material_role(data) or is_custom_mechanical_part(data):
        return False
    profile = data.get("mechanical_profile")
    if profile in MAGNETIC_BODIES:
        return True
    return profile == MAGNETIC_LENS_HOUSING and any(
        value["magnetic_response"] == "bh" for value in validated_material_regions(data).values()
    )
