"""Explicit mechanical partition of the shared C1/C2 carrier.

The saved separator is an engineering adaptation of a generic nonmagnetic
pole support. It is not another optical gap or a measured OEM material map.
Existing pole pieces, windings and yokes retain their own material ownership.
"""

from copy import deepcopy
import math
from numbers import Real

from temsim.cad_geometry import SolidSpec, annulus
from temsim.magnetic_circuits import is_custom_mechanical_part, part_data
from temsim.mechanical_profiles import C1_C2_POLE_PIECE_CARTRIDGE


PREFIX = "mechanical_cartridge_"
STRUCTURE = "carrier_separator_carrier"
STATUS = "engineering_reconstruction_not_oem"
LEDGER = "docs/references/cad-round-lenses.md"
FIELDS = frozenset(PREFIX + suffix for suffix in (
    "structure", "geometry_status", "geometry_source", "geometry_source_urls",
    "spacer_material", "spacer_start_local_z_mm", "spacer_end_local_z_mm", "pole_keys",
))


def _number(row, name):
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{row['key']}: {name} must be a finite number")
    return float(value)


def validate_cartridge_geometry(row, by_key=None):
    """Validate opted-in local-mm sections; absent fields preserve legacy CAD.

    Optional context checks the existing physical pole members, without using
    their faces to silently replace the explicitly saved separator positions.
    """
    row = part_data(row)
    if not any(name.startswith(PREFIX) for name in row):
        return
    unknown = sorted(name for name in row if name.startswith(PREFIX) and name not in FIELDS)
    if unknown:
        raise ValueError(f"{row['key']}: unknown cartridge fields: {', '.join(unknown)}")
    if row.get("mechanical_profile") != C1_C2_POLE_PIECE_CARTRIDGE:
        raise ValueError(f"{row['key']}: cartridge details require the cartridge profile")
    if row.get(PREFIX + "structure") != STRUCTURE:
        raise ValueError(f"{row['key']}: unsupported cartridge structure")
    if row.get(PREFIX + "geometry_status") != STATUS:
        raise ValueError(f"{row['key']}: cartridge geometry must identify its engineering reconstruction")
    source = row.get(PREFIX + "geometry_source")
    urls = row.get(PREFIX + "geometry_source_urls")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"{row['key']}: cartridge geometry_source is required")
    if (not isinstance(urls, (tuple, list)) or not urls
            or any(not isinstance(url, str) or not url.startswith("https://") for url in urls)):
        raise ValueError(f"{row['key']}: cartridge geometry_source_urls are required")
    if row.get(PREFIX + "spacer_material") != "non_magnetic_structural":
        raise ValueError(f"{row['key']}: cartridge spacer material must remain non_magnetic_structural")
    if row.get("material_class") != "mixed_material_cartridge_user_photo_unresolved":
        raise ValueError(f"{row['key']}: cartridge carriers must retain unresolved mixed material")
    start = _number(row, "local_start_z_mm")
    end = _number(row, "local_end_z_mm")
    a = _number(row, PREFIX + "spacer_start_local_z_mm")
    b = _number(row, PREFIX + "spacer_end_local_z_mm")
    if not start < a < b < end:
        raise ValueError(f"{row['key']}: cartridge requires start < spacer start < spacer end < end")
    inner = _number(row, "mechanical_inner_diameter_mm")
    outer = _number(row, "mechanical_outer_diameter_mm")
    vacuum = _number(row, "vacuum_inner_diameter_mm")
    if not 0 < vacuum <= inner < outer:
        raise ValueError(f"{row['key']}: cartridge requires 0 < vacuum bore <= material ID < OD")
    if "magnetic_radial_profile_mm" in row or "material_intervals_mm" in row:
        raise ValueError(f"{row['key']}: cartridge sections cannot compete with another material profile")
    members = row.get(PREFIX + "pole_keys")
    if (not isinstance(members, (tuple, list)) or len(members) != 2
            or any(not isinstance(key, str) or not key for key in members)
            or len(set(members)) != 2):
        raise ValueError(f"{row['key']}: cartridge pole_keys must identify two distinct existing poles")
    # A mechanical-only copied carrier has no optical pole children. References
    # retained in its recipe document the source, not new field ownership.
    if by_key is not None and not is_custom_mechanical_part(row):
        for key in members:
            if key not in by_key:
                raise ValueError(f"{row['key']}: cartridge pole {key} is missing")
            pole = part_data(by_key[key])
            if (pole.get("mechanical_profile") != "magnetic_pole_piece"
                    or pole.get("mechanical_container_key") != row["key"]):
                raise ValueError(f"{row['key']}: cartridge pole {key} must remain a separate contained pole")
            if not (start <= _number(pole, "local_start_z_mm")
                    < _number(pole, "local_end_z_mm") <= end):
                raise ValueError(f"{row['key']}: cartridge pole {key} lies outside its carrier")
            if _number(pole, "mechanical_outer_diameter_mm") > inner:
                raise ValueError(f"{row['key']}: cartridge pole {key} overlaps carrier material")


def cartridge_solids(part, by_key=None):
    """Return three disjoint material solids, or None for unchanged legacy CAD."""
    row = part_data(part)
    if not any(name.startswith(PREFIX) for name in row):
        return None
    validate_cartridge_geometry(row, by_key)
    from temsim.component_representation import non_material_role
    if non_material_role(row):
        return ()
    start, end = float(row["local_start_z_mm"]), float(row["local_end_z_mm"])
    a, b = float(row[PREFIX + "spacer_start_local_z_mm"]), float(row[PREFIX + "spacer_end_local_z_mm"])
    inner, outer = .5 * row["mechanical_inner_diameter_mm"], .5 * row["mechanical_outer_diameter_mm"]
    parameters = {name: deepcopy(value) for name, value in row.items()
                  if name.startswith(PREFIX) or name in {
                      "local_start_z_mm", "local_end_z_mm", "mechanical_inner_diameter_mm",
                      "mechanical_outer_diameter_mm", "vacuum_inner_diameter_mm",
                      "magnetic_lens_keys", "material_class"}}
    evidence = (LEDGER, *row[PREFIX + "geometry_source_urls"])
    sections = (
        ("c1_carrier_unresolved", start, a, row["material_class"]),
        ("nonmagnetic_separator_assumed", a, b, "nonmagnetic_housing"),
        ("c2_carrier_unresolved", b, end, row["material_class"]),
    )
    note = ("Three sections partition the existing shared carrier envelope. The nonmagnetic "
            "separator is a generic construction-family engineering assumption, not a known OEM "
            "material or an optical gap. Carrier material subdivision remains unresolved. "
            "Existing poles, yokes, windings and analytic lens fields are separate and unchanged; "
            "these display/CAD sections are not an added magnetic-material solver input.")
    return tuple(SolidSpec(str(row["key"]), region, material,
                           annulus(lo, hi, inner, outer), evidence, STATUS, note,
                           deepcopy(parameters))
                 for region, lo, hi, material in sections)
