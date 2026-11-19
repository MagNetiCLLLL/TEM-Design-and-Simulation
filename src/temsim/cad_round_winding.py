"""Optional closed round-wire turns and the remaining pack insulation.

This is mechanical CAD. It does not replace the prescribed-current field's
homogenized original envelope with discrete wires or alter its ampere-turns.
Existing passive end layers remain separately identified material regions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from temsim.cad_geometry import SolidSpec, torus, winding_insulation
from temsim.coil_winding import (
    WINDING_FIELDS, WindingLayout, resolve_allocated_winding_layout,
    resolve_wire_cross_section_mm2, resolve_wire_diameter_mm,
    validate_interval_turn_counts, validate_winding_definition,
)
from temsim.magnetic_circuits import is_custom_mechanical_part, part_data, radial_profile_mm
from temsim.magnetic_geometry import objective_layer_intervals_mm


@dataclass(frozen=True)
class RoundWindingGeometry:
    """One layout and the distinct field and conductor allocation witnesses."""

    key: str
    layout: WindingLayout
    field_envelope: dict
    conductor_allocations: tuple[dict, ...]
    passive_solids: tuple[SolidSpec, ...]
    wire_cross_section_mm2: float | None = None

    def to_metadata(self) -> dict:
        return dict(schema="temsim-round-wire-cad-v1", key=self.key,
                    packing=self.layout.to_metadata(),
                    field_envelope=deepcopy(self.field_envelope),
                    conductor_allocations=deepcopy(list(self.conductor_allocations)),
                    passive_regions=[spec.region for spec in self.passive_solids],
                    wire_cross_section_mm2=self.wire_cross_section_mm2,
                    field_policy="Homogenize the original full winding envelope; do not shrink it to the wire allocation",
                    leads_included=False)


def _envelope(row, by_key):
    from temsim.part_model_3d import _extent, _number

    start, end = _extent(row)
    if not {"mechanical_inner_diameter_mm", "mechanical_outer_diameter_mm"}.issubset(row):
        raise ValueError(f"{row['key']}: winding inner and outer diameters must be explicit")
    inner = _number(row["mechanical_inner_diameter_mm"], "Winding inner diameter")
    outer = _number(row["mechanical_outer_diameter_mm"], "Winding outer diameter")
    if not 0 < inner < outer or not start < end:
        raise ValueError(f"{row['key']}: round winding requires positive length and 0 < ID < OD")
    parent_raw = (by_key or {}).get(row.get("parent_key"))
    parent = part_data(parent_raw) if parent_raw is not None else {}
    intervals = ()
    if is_custom_mechanical_part(row) and "material_intervals_mm" in row:
        intervals = row["material_intervals_mm"]
    elif not is_custom_mechanical_part(row):
        intervals = objective_layer_intervals_mm(parent, parent.get("local_start_z_mm", 0),
                                                "magnetic_excitation_coil")
    if not intervals:
        intervals = row.get("material_intervals_mm", ((start, end),))
    intervals = tuple(tuple(_number(value, "Winding interval") for value in pair) for pair in intervals)
    if (not intervals or any(len(pair) != 2 or pair[1] <= pair[0] for pair in intervals)
            or any(a[1] > b[0] for a, b in zip(intervals, intervals[1:]))):
        raise ValueError(f"{row['key']}: winding intervals must be ordered, disjoint and nonempty")
    return dict(inner_diameter_mm=inner, outer_diameter_mm=outer,
                intervals_mm=[list(pair) for pair in intervals])


def round_winding_envelope(part, by_key=None) -> dict:
    """Read the original annular envelope without inventing wire parameters."""
    row = part_data(part)
    if row.get("mechanical_profile") != "magnetic_excitation_coil":
        raise ValueError(f"{row['key']}: detailed round winding is only available for a round excitation coil")
    if row.get("model_3d", {}).get("base", {}).get("kind", "existing") != "existing":
        raise ValueError(f"{row['key']}: round winding inputs require the native annular base geometry")
    from temsim.part_model_3d import _extent
    start, end = _extent(row)
    if radial_profile_mm(row, end-start) is not None:
        raise ValueError(f"{row['key']}: round wire layout requires a rectangular annular winding envelope")
    return _envelope(row, by_key)


def resolve_round_winding_geometry(part, by_key=None) -> RoundWindingGeometry | None:
    """Validate the exact layout used by CAD without constructing a mesh/BRep.

    An N-only electrical definition keeps its original envelope CAD without
    guessing a wire diameter. Fixed interval counts are checked even in that
    case. Coordinates and lengths are module-local millimetres.
    """
    row = part_data(part)
    if not WINDING_FIELDS.intersection(row):
        return None
    validate_winding_definition(row)
    envelope = round_winding_envelope(row, by_key)
    fixed_counts = validate_interval_turn_counts(row, len(envelope["intervals_mm"]))
    diameter = resolve_wire_diameter_mm(row)
    if diameter is None:
        return None
    from temsim.cad_objective_upper_hardware import objective_upper_hardware_solids
    detailed = objective_upper_hardware_solids(row, by_key)
    passive = tuple(spec for spec in (detailed or ()) if spec.material != "copper_winding")
    allocations = []
    if detailed is not None:
        for spec in detailed:
            if spec.material == "copper_winding":
                points = spec.shape["section_zr_mm"]
                allocations.append(dict(region=spec.region,
                    start_mm=min(point[0] for point in points), end_mm=max(point[0] for point in points),
                    inner_mm=min(point[1] for point in points), outer_mm=max(point[1] for point in points)))
    else:
        intervals = envelope["intervals_mm"]
        names = ("upper", "lower") if len(intervals) == 2 else tuple(f"interval_{i}" for i in range(len(intervals)))
        allocations = [dict(region=region, start_mm=a, end_mm=b,
                            inner_mm=envelope["inner_diameter_mm"]/2,
                            outer_mm=envelope["outer_diameter_mm"]/2)
                       for region, (a, b) in zip(names, intervals)]
    allocation_counts = None
    if fixed_counts is not None:
        # Supported native packs have one conductor allocation per original
        # source interval. Reject an ambiguous future subdivision explicitly;
        # a CAD packing heuristic must not silently reassign physical turns.
        allocation_counts = [None]*len(allocations)
        for (a, b), count in zip(envelope["intervals_mm"], fixed_counts):
            matches = [index for index, item in enumerate(allocations)
                       if a <= item["start_mm"] < item["end_mm"] <= b]
            if len(matches) != 1 or allocation_counts[matches[0]] is not None:
                raise ValueError(f"{row['key']}: fixed interval counts require one unambiguous conductor allocation per field interval")
            allocation_counts[matches[0]] = count
        if any(count is None for count in allocation_counts):
            raise ValueError(f"{row['key']}: conductor allocations do not preserve every fixed winding interval")
    layout = resolve_allocated_winding_layout(
        tuple((item["inner_mm"]*2, item["outer_mm"]*2, item["start_mm"], item["end_mm"])
              for item in allocations), row["winding_turn_count"], diameter,
        allocation_turn_counts=allocation_counts)
    return RoundWindingGeometry(str(row["key"]), layout, envelope, tuple(allocations), passive,
                                resolve_wire_cross_section_mm2(row))


def round_winding_solids(part, by_key=None) -> tuple[SolidSpec, ...] | None:
    """Retain every configured turn, all remaining insulation and passive layers."""
    geometry = resolve_round_winding_geometry(part, by_key)
    if geometry is None:
        return None
    row = part_data(part)
    radius = geometry.layout.wire_diameter_mm/2
    receipt = geometry.to_metadata()
    note = ("Configured round-wire closed turns, without leads. The remaining allocated winding volume is "
            "uniform unspecified insulation. Existing passive layers are retained. The prescribed-current field "
            "continues to homogenize the original complete winding envelope; this CAD is not a discrete-wire field solve. "
            "Display wire sections use up to 12 facets; STEP wire sections remain analytic circles.")
    specs = []
    turns = tuple(geometry.layout.turns)
    for index, turn in enumerate(turns):
        specs.append(SolidSpec(geometry.key, f"winding_turn_{index+1:06d}", "copper_winding",
                     torus(turn.radius_mm, radius, turn.z_mm), (), "configured_geometry", note,
                     dict(winding_turn_index=index+1, winding_wire_diameter_mm=radius*2,
                          winding_wire_cross_section_mm2=geometry.wire_cross_section_mm2,
                          display_wire_profile_max_facets=12)))
    assigned = 0
    for allocation_index, allocation in enumerate(geometry.conductor_allocations):
        a, b = allocation["start_mm"], allocation["end_mm"]
        local_turns = tuple((turn.radius_mm, turn.z_mm) for turn in turns if turn.interval_index == allocation_index)
        assigned += len(local_turns)
        shape = winding_insulation(a, b, allocation["inner_mm"], allocation["outer_mm"], radius, local_turns)
        specs.append(SolidSpec(geometry.key, "winding_insulation_"+allocation["region"], "insulator",
                     shape, (), "configured_geometry", note,
                     dict(winding_turn_count=len(local_turns), winding_geometry=deepcopy(receipt),
                          display_wire_profile_max_facets=12)))
    if assigned != len(turns):
        raise ValueError(f"{geometry.key}: every round wire turn must belong to exactly one allocation")
    return tuple(specs) + geometry.passive_solids
