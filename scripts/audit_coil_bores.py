"""Inspect nominal winding bores and objective inserts in the installed instrument catalog."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from temsim.cad_catalog import part_solids
from temsim.cad_objective_sleeves import COMPONENTS, validate_sleeve
from temsim.cad_objective_upper_hardware import objective_upper_hardware_solids
from temsim.cad_round_winding import round_winding_envelope
from temsim.component_representation import non_material_role
from temsim.deflector_support_geometry import full_material_radial_extent
from temsim.module_manifest import read_document
from temsim.multipole_winding import multipole_winding_envelopes, supported_multipole_winding
from temsim.ordinary_deflector_geometry import support_dimensions
from temsim.transverse_winding import transverse_winding_envelopes

INSTRUMENTS = ROOT / "configs" / "instruments"


def interval(part):
    return part["local_start_z_mm"], part["local_end_z_mm"]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def round_coils(parts):
    records = {}
    for part in parts.values():
        if part.get("mechanical_profile") != "magnetic_excitation_coil":
            continue
        envelope = round_winding_envelope(part, parts)
        owner = part["parent_key"]
        poles = {}
        for pole in parts.values():
            if pole.get("parent_key") == owner and pole.get("mechanical_profile") == "magnetic_pole_piece":
                solid, = part_solids(pole, parts)
                radii = [r for _, r in solid.shape["section_zr_mm"]]
                poles[pole["key"]] = {"min_bore_mm": 2 * min(radii), "max_od_mm": 2 * max(radii)}
        records[part["key"]] = {"owner": owner, **envelope, "poles": poles}
    return records


def off_axis_windings(parts):
    """Inner diameter tangent to the nearest winding pack, before any support or tube."""
    records = {}
    for part in parts.values():
        key = part["key"]
        coils = transverse_winding_envelopes(part)
        if not coils and supported_multipole_winding(part):
            if non_material_role(part) and "multipole_winding_turn_count_per_coil" not in part:
                continue
            coils = multipole_winding_envelopes(part, parts)
        if coils:
            inner = 2 * min(c["radial_start_mm"] if c["family"] == "radial_rectangle"
                            else c["inner_radius_mm"] for c in coils)
        elif key == "image_hp3_hexapole":
            # B-COR's third hexapole retains six native bulk windings.
            copper = next(s for s in part_solids(part, parts) if s.material == "copper_winding")
            p = copper.parameters
            inner = 2 * (p["inner_radius_mm"] + (p["outer_radius_mm"] - p["inner_radius_mm"])
                         * p["coil_radial_fractions"][0])
        else:
            continue
        clear = part.get("mechanical_clear_bore_diameter_mm", part.get("bore_diameter_mm"))
        record = {"pack_inner_tangent_diameter_mm": inner, "declared_clear_bore_mm": clear}
        if part.get("mechanical_profile") == "deflector_coil_insert":
            ri, ro = full_material_radial_extent(part)
            record["insert_id_od_with_support_mm"] = [2 * ri, 2 * ro]
        if non_material_role(part):
            record["material_scope"] = "winding only at an existing control reference"
        records[key] = record
    return records


def objective_fits(parts, windings):
    coil_key = "objective_lens_excitation_coil"
    if coil_key not in windings:
        return {}
    reference = (parts["objective_upper_pole"]["local_end_z_mm"]
                 + parts["objective_lower_pole"]["local_start_z_mm"]) / 2
    winding = windings[coil_key]
    detail = objective_upper_hardware_solids(parts[coil_key], parts) or ()
    copper_ids = {s.region: 2 * min(r for _, r in s.shape["section_zr_mm"])
                  for s in detail if s.material == "copper_winding"}
    result = {}
    for side, winding_span in zip(("upper", "lower"), winding["intervals_mm"], strict=True):
        sleeve = parts[f"objective_{side}_electronics_sleeve"]
        validate_sleeve(sleeve, parts)
        sleeve_span = interval(sleeve)
        sleeve_id = sleeve["mechanical_inner_diameter_mm"]
        sleeve_od = sleeve["mechanical_outer_diameter_mm"]
        packages = {}
        for key in COMPONENTS[side]:
            part = parts[key]
            packages[key] = {
                "z_from_gap_midplane_mm": [z - reference for z in interval(part)],
                "package_od_mm": part["mechanical_outer_diameter_mm"],
                "sleeve_radial_clearance_mm": (sleeve_id - part["mechanical_outer_diameter_mm"]) / 2,
                "sleeve_axial_overlap_mm": overlap(interval(part), sleeve_span),
                "winding_axial_overlap_mm": overlap(interval(part), winding_span),
            }
        result[side] = {
            "geometry_source": sleeve.get("mechanical_geometry_source"),
            "winding_z_from_gap_midplane_mm": [z - reference for z in winding_span],
            "copper_pack_inner_diameter_mm": copper_ids.get(side, winding["inner_diameter_mm"]),
            "winding_pole_axial_overlap_mm": overlap(winding_span, interval(parts[f"objective_{side}_pole"])),
            "sleeve_z_from_gap_midplane_mm": [z - reference for z in sleeve_span],
            "sleeve_id_od_mm": [sleeve_id, sleeve_od],
            "winding_to_sleeve_radial_clearance_mm": (winding["inner_diameter_mm"] - sleeve_od) / 2,
            "winding_sleeve_axial_overlap_mm": overlap(winding_span, sleeve_span),
            "packages": packages,
        }
    shift_tilt = parts["beam_deflector"]
    shift_span = interval(shift_tilt)
    shift_packs = transverse_winding_envelopes(shift_tilt)
    support = support_dimensions(shift_tilt)
    outer_radius = (support["material_outer_radius_mm"] if support
                    else max(pack["outer_radius_mm"] for pack in shift_packs))
    result["upper"]["separate_beam_shift_tilt"] = {
        "key": shift_tilt["key"],
        "z_from_gap_midplane_mm": [z - reference for z in shift_span],
        "distance_to_sleeve_start_mm": interval(parts["objective_upper_electronics_sleeve"])[0] - shift_span[1],
        "declared_outer_diameter_mm": shift_tilt["mechanical_outer_diameter_mm"],
        "modeled_windings_and_supports_od_mm": 2 * outer_radius,
    }
    return result


def audit():
    catalog = tomllib.loads((INSTRUMENTS / "catalog.toml").read_text(encoding="utf-8"))
    results = {}
    for kind, entries in catalog.items():
        if not kind.endswith("_variants"):
            continue
        for entry in entries:
            source = entry["file"]
            document = read_document(INSTRUMENTS / source)
            parts = {part["key"]: part for part in document["parts"]}
            windings = round_coils(parts)
            crossings = []
            for a, b in combinations(windings.values(), 2):
                length = sum(overlap(x, y) for x in a["intervals_mm"] for y in b["intervals_mm"])
                if length > 1e-8:
                    crossings.append({"owners": [a["owner"], b["owner"]], "axial_overlap_mm": length})
            results[source] = {
                "round_windings": windings, "overlapping_round_windings": crossings,
                "other_winding_packs": off_axis_windings(parts),
                "objective_installation": objective_fits(parts, windings),
            }
            geometry = document["geometry"]
            if "mechanical_material_tube_outer_profile_z_od_mm" in geometry:
                results[source]["continuous_tube"] = {
                    "dimension_source": geometry.get("c1_c2_objective_vacuum_tube_geometry_source"),
                    "inner_diameter_mm": geometry["c1_c2_objective_vacuum_tube_inner_diameter_mm"],
                    "outer_profile_z_od_mm": geometry["mechanical_material_tube_outer_profile_z_od_mm"],
                }
    return results


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
