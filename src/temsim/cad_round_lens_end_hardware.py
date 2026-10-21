"""Insulated round-lens end hardware shared by explicitly configured families.

No dimensions or material/current domains are inferred from the family name.
Each housing carries its complete saved recipe, independently of sibling rows.
"""

from copy import deepcopy
import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, annulus
from temsim.magnetic_circuits import is_custom_mechanical_part, part_data


STATUS = "engineering_reconstruction_not_oem"
FAMILY_PREFIXES = ("mechanical_c3_end_", "mechanical_adapter_end_", "mechanical_tl22_end_",
                   "mechanical_transfer_end_")
SUFFIXES = (
    "structure", "upper_contact_local_z_mm", "lower_contact_local_z_mm",
    "inner_diameter_mm", "outer_diameter_mm", "insulation_thickness_mm",
    "cooling_plate_thickness_mm", "coil_key", "geometry_status",
    "geometry_source", "geometry_source_urls",
)


def end_hardware_fields(prefix):
    return frozenset(prefix + suffix for suffix in SUFFIXES)



def _number(row, name):
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{row['key']}: {name} must be finite and numeric")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{row['key']}: {name} must be finite and numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{row['key']}: {name} must be finite and numeric")
    return result


def validate_lens_end_hardware(row, by_key=None, *, prefix, label):
    """Reject contradictory saved hardware; missing profile preserves legacy."""
    row = part_data(row)
    # An explicit box/STEP/etc. replaces the native recipe. Its own validator
    # owns geometry; retained recipe dimensions are checked upon restoration.
    if row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return
    fields = end_hardware_fields(prefix)
    if not any(name.startswith(prefix) for name in row):
        return
    conflicting = [other for other in FAMILY_PREFIXES if other != prefix
                   and any(name.startswith(other) for name in row)]
    if conflicting:
        raise ValueError(f"{row['key']}: conflicting round-lens end hardware families: "
                         f"{prefix}, {', '.join(conflicting)}")
    unknown = sorted(name for name in row if name.startswith(prefix) and name not in fields)
    if unknown:
        raise ValueError(f"{row['key']}: unknown {label} end hardware fields: {', '.join(unknown)}")
    if (row.get("mechanical_profile") != "magnetic_lens_housing"
            or row.get(prefix + "structure") != "insulated_coil_end_cooling_plates"):
        raise ValueError(f"{row['key']}: {label} end hardware requires its housing profile and declared structure")
    if row.get("material_class") != "non_magnetic_structural":
        raise ValueError(f"{row['key']}: {label} end hardware housing must retain non_magnetic_structural material")
    if row.get(prefix + "geometry_status") != STATUS:
        raise ValueError(f"{row['key']}: {label} end hardware must declare engineering reconstruction, not OEM")
    source, urls = row.get(prefix + "geometry_source"), row.get(prefix + "geometry_source_urls")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{row['key']}: {label} end hardware geometry_source is required")
    if (not isinstance(urls, (list, tuple)) or not urls
            or any(not isinstance(url, str) or not url.startswith("https://") for url in urls)):
        raise ValueError(f"{row['key']}: {label} end hardware geometry_source_urls are required")
    coil_key = row.get(prefix + "coil_key")
    if not isinstance(coil_key, str) or not coil_key.strip():
        raise ValueError(f"{row['key']}: {label} end hardware coil_key is required")
    if any(name in row for name in ("magnetic_radial_profile_mm", "material_intervals_mm")):
        raise ValueError(f"{row['key']}: {label} end hardware cannot override another housing material profile")
    start, end = (_number(row, name) for name in ("local_start_z_mm", "local_end_z_mm"))
    upper, lower, inner, outer, insulation, plate = (
        _number(row, prefix + name) for name in (
            "upper_contact_local_z_mm", "lower_contact_local_z_mm", "inner_diameter_mm",
            "outer_diameter_mm", "insulation_thickness_mm", "cooling_plate_thickness_mm"))
    housing_inner, housing_outer, vacuum = (_number(row, name) for name in (
        "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm"))
    # Test the same individual planes used by the solid builder. A positive
    # decimal thickness can round away at a nonzero column Z, even when the
    # combined insulation/plate stack still has a positive total span.
    if not (0 < insulation and 0 < plate
            and start <= upper - insulation - plate < upper - insulation < upper
            < lower < lower + insulation < lower + insulation + plate <= end):
        raise ValueError(f"{row['key']}: {label} end hardware needs positive thicknesses and ordered contacts inside housing")
    if not 0 < vacuum < inner < outer < housing_inner < housing_outer:
        raise ValueError(f"{row['key']}: {label} end hardware requires vacuum < plate ID < plate OD < housing ID < OD")
    if by_key is None or is_custom_mechanical_part(row):
        return
    # A detached recipe and a deliberately empty context remain self-contained.
    # A native assembly context containing its parent or material siblings must
    # supply the referenced coil, so a misspelling cannot bypass contact checks.
    parent_key = row.get("parent_key")
    has_assembly_context = parent_key in by_key or any(
        part_data(raw).get("parent_key") == parent_key
        and part_data(raw).get("mechanical_profile") in {
            "magnetic_excitation_coil", "magnetic_pole_piece", "magnetic_lens_yoke"}
        for raw in by_key.values())
    if coil_key not in by_key and has_assembly_context:
        raise ValueError(f"{row['key']}: {label} end hardware references missing coil {coil_key}; "
                         "synchronize the housing hardware recipe with the native assembly")
    # Missing partial context never substitutes a global/default file's sizes.
    # When supplied, the physical winding must match the saved interfaces.
    if coil_key in by_key:
        coil = part_data(by_key[coil_key])
        expected = (upper, lower, inner, outer)
        actual = tuple(_number(coil, name) for name in (
            "local_start_z_mm", "local_end_z_mm", "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"))
        if (coil.get("mechanical_profile") != "magnetic_excitation_coil"
                or coil.get("parent_key") != row.get("parent_key")
                or any(abs(a - b) > 1e-9 for a, b in zip(expected, actual))):
            raise ValueError(f"{row['key']}: {label} end hardware no longer contacts {coil_key}; "
                             "synchronize the housing hardware recipe with the coil geometry")
    spans = ((upper - insulation - plate, upper), (lower, lower + insulation + plate))
    for raw in by_key.values():
        other = part_data(raw)
        if other.get("parent_key") != row.get("parent_key"):
            continue
        profile = other.get("mechanical_profile")
        if profile not in {"magnetic_pole_piece", "magnetic_lens_yoke"}:
            continue
        lo, hi = _number(other, "local_start_z_mm"), _number(other, "local_end_z_mm")
        if not any(max(lo, a) < min(hi, b) for a, b in spans):
            continue
        if profile == "magnetic_pole_piece" and _number(other, "mechanical_outer_diameter_mm") >= inner:
            raise ValueError(f"{row['key']}: {label} end hardware lacks clearance from pole {other['key']}")
        if profile == "magnetic_lens_yoke" and _number(other, "mechanical_inner_diameter_mm") <= outer:
            raise ValueError(f"{row['key']}: {label} end hardware lacks clearance from yoke {other['key']}")


def lens_end_hardware_solids(part, by_key=None, *, prefix, label):
    """Original housing wall plus four separate, explicitly saved CAD regions."""
    row = part_data(part)
    if row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        return None
    fields = end_hardware_fields(prefix)
    if not any(name.startswith(prefix) for name in row):
        return None
    validate_lens_end_hardware(row, by_key, prefix=prefix, label=label)
    from temsim.component_representation import non_material_role
    if non_material_role(row):
        return ()
    a, b, inner, outer, insulation, plate = (
        float(row[prefix + name]) for name in (
            "upper_contact_local_z_mm", "lower_contact_local_z_mm", "inner_diameter_mm",
            "outer_diameter_mm", "insulation_thickness_mm", "cooling_plate_thickness_mm"))
    params = {name: deepcopy(value) for name, value in row.items()
              if name in fields or name in {"local_start_z_mm", "local_end_z_mm",
                  "mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"}}
    evidence = ("docs/references/cad-round-lenses.md", *row[prefix + "geometry_source_urls"])
    note = (f"Generic insulated coil-end cooling-plate construction adapted to the saved {label} housing. "
            "All internal dimensions are explicit engineering reconstruction, not OEM measurements. "
            "The separate excitation winding, yoke and pole solids are unchanged. Plate copper is "
            "passive thermal material, not another current source. Mounting bolts, hidden water paths, "
            "thermal transport and eddy-current response are not reconstructed or solved.")
    sections = (
        ("upper_cooling_plate", a - insulation - plate, a - insulation, "copper_cooling_plate"),
        ("upper_coil_insulation", a - insulation, a, "insulator"),
        ("lower_coil_insulation", b, b + insulation, "insulator"),
        ("lower_cooling_plate", b + insulation, b + insulation + plate, "copper_cooling_plate"),
    )
    wall = SolidSpec(str(row["key"]), "body", "nonmagnetic_housing",
        annulus(float(row["local_start_z_mm"]), float(row["local_end_z_mm"]),
                .5 * row["mechanical_inner_diameter_mm"], .5 * row["mechanical_outer_diameter_mm"]),
        evidence, "engineering_reconstruction", note, deepcopy(params))
    return (wall, *(SolidSpec(str(row["key"]), region, material,
        annulus(lo, hi, .5 * inner, .5 * outer), evidence, STATUS, note, deepcopy(params))
        for region, lo, hi, material in sections))
