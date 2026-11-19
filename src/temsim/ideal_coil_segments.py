"""Electrical partitions inside an unchanged ideal annular winding.

Fractions refer to the original radial-axial current-bearing rectangle, not
to CAD wire radii or new mechanical parts. Every cell consists of complete
azimuthal current loops. The vacuum calculation has no R or power constraint.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral, Real
import sys


FIELD = "ideal_current_segments"
CURRENT_RECIPE_FIELDS = (FIELD, "ideal_transverse_current_segments")
SCHEMA = "ideal-coil-current-segments-v1"
MAXIMUM_SEGMENTS_PER_WINDING = 1024
_CELL_KEYS = frozenset({"interval_index", "turn_count", "radial_fraction",
                        "axial_fraction", "current_ratio"})


def _integer(value, name, *, positive=True):
    if (isinstance(value, bool) or not isinstance(value, Integral)
            or value < (1 if positive else 0)):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return int(value)


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _bounds(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} requires two normalized bounds")
    lower, upper = (_real(item, name) for item in value)
    if not 0. <= lower < upper <= 1.:
        raise ValueError(f"{name} must stay within the original winding allocation [0, 1]")
    return lower, upper


@dataclass(frozen=True, slots=True)
class CurrentSegment:
    interval_index: int
    turn_count: int
    radial_fraction: tuple[float, float]
    axial_fraction: tuple[float, float]
    current_ratio: float

    def to_dict(self):
        return dict(interval_index=self.interval_index, turn_count=self.turn_count,
                    radial_fraction=list(self.radial_fraction),
                    axial_fraction=list(self.axial_fraction), current_ratio=self.current_ratio)


def normalize_recipe(recipe):
    """Validate portable electrical syntax without resolving a mechanical scene."""
    if (not isinstance(recipe, Mapping) or set(recipe) != {"schema", "windings"}
            or recipe["schema"] != SCHEMA or not isinstance(recipe["windings"], Mapping)):
        raise ValueError("Invalid ideal-coil internal-current recipe")
    output = {}
    for key, raw in recipe["windings"].items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("A current partition requires a physical winding key")
        if (not isinstance(raw, (tuple, list)) or not raw
                or len(raw) > MAXIMUM_SEGMENTS_PER_WINDING):
            raise ValueError("A winding requires a nonempty bounded list of current segments")
        cells = []
        for row in raw:
            if not isinstance(row, Mapping) or set(row) != _CELL_KEYS:
                raise ValueError("A current segment must declare its interval, turns, bounds and ratio")
            cell = CurrentSegment(
                _integer(row["interval_index"], "Interval index", positive=False),
                _integer(row["turn_count"], "Segment turns"),
                _bounds(row["radial_fraction"], "Radial fraction"),
                _bounds(row["axial_fraction"], "Axial fraction"),
                _real(row["current_ratio"], "Current ratio"))
            cells.append(cell.to_dict())
        output[key] = cells
    return dict(schema=SCHEMA, windings=output)


def _tile(cells):
    radial = sorted({0., 1., *(edge for cell in cells for edge in cell.radial_fraction)})
    for lower, upper in zip(radial[:-1], radial[1:]):
        active = sorted((cell.axial_fraction for cell in cells
                         if cell.radial_fraction[0] <= lower and cell.radial_fraction[1] >= upper))
        previous = 0.
        for start, stop in active:
            if start != previous:
                raise ValueError("Current segments must tile the original allocation without gaps or overlaps")
            previous = stop
        if previous != 1.:
            raise ValueError("Current segments must cover the complete original allocation")


def bind_segments(recipe, winding_key, interval_turn_counts):
    """Bind current groups to exact physical counts, with unchanged density.

    The only tolerance below covers binary64 arithmetic in rational partition
    bounds. It is not a magnetic-field, probe, packing or solver tolerance.
    """
    canonical = normalize_recipe(recipe)
    rows = canonical["windings"].get(winding_key)
    if rows is None:
        return None
    counts = tuple(_integer(value, "Physical interval turns") for value in interval_turn_counts)
    if not counts:
        raise ValueError("Internal currents require declared physical turns in every winding interval")
    cells = tuple(CurrentSegment(row["interval_index"], row["turn_count"],
        tuple(row["radial_fraction"]), tuple(row["axial_fraction"]), row["current_ratio"]) for row in rows)
    if any(cell.interval_index >= len(counts) for cell in cells):
        raise ValueError("Current segment refers to an absent physical winding interval")
    for index, count in enumerate(counts):
        selected = tuple(cell for cell in cells if cell.interval_index == index)
        if not selected or sum(cell.turn_count for cell in selected) != count:
            raise ValueError("Current segments must preserve the physical turns in each original interval")
        _tile(selected)
        for cell in selected:
            area = ((cell.radial_fraction[1]-cell.radial_fraction[0])
                    *(cell.axial_fraction[1]-cell.axial_fraction[0]))
            # Subtracting normalized endpoints near one can lose relative
            # digits for a thin cell. Bound that coordinate-roundoff in area
            # units instead of imposing a relative test on the thin fraction.
            rounding = 16.*sys.float_info.epsilon
            if not math.isclose(area, cell.turn_count/count, rel_tol=0., abs_tol=rounding):
                raise ValueError("Segment area must retain the original homogenized turn density")
    return cells


def axial_recipe(winding_key, interval_turn_counts, groups_per_interval, *, current_ratios=None):
    """Partition by integer turns; original interval endpoints stay unchanged."""
    groups = _integer(groups_per_interval, "Groups per interval")
    counts = tuple(_integer(count, "Physical interval turns") for count in interval_turn_counts)
    rows = []
    for index, count in enumerate(counts):
        if groups > count:
            raise ValueError("A current group must contain at least one original physical turn")
        allocation = [count//groups+(position < count%groups) for position in range(groups)]
        cumulative = 0
        for number in allocation:
            lower = cumulative/count
            cumulative += number
            rows.append(dict(interval_index=index, turn_count=number,
                radial_fraction=[0., 1.], axial_fraction=[lower, cumulative/count], current_ratio=1.))
    if current_ratios is not None:
        if len(current_ratios) != len(rows):
            raise ValueError("Supply one current ratio per generated segment")
        for row, ratio in zip(rows, current_ratios):
            row["current_ratio"] = _real(ratio, "Current ratio")
    return normalize_recipe(dict(schema=SCHEMA, windings={winding_key: rows}))


def require_ideal_mode(state):
    from temsim.simulation_modes import mode_key
    if mode_key(state) != "unified" or getattr(state, "round_lens_field_model", "gaussian") != "copper_coil":
        raise ValueError("Internal forward/reverse currents are available only in Ideal prescribed-current coils in vacuum")


def set_current_segments(state, owner_key, recipe):
    """Save a validated electrical recipe, without touching drives or geometry."""
    require_ideal_mode(state)
    canonical = normalize_recipe(recipe)
    assembly = getattr(state, "_resolved_assembly", None)
    if assembly is None:
        raise ValueError("Internal current groups require a resolved physical assembly")
    winding_parts = {part.key: part for part in assembly.parts
                     if part.data.get("mechanical_profile") == "magnetic_excitation_coil"
                     and part.data.get("field_source_key") == owner_key}
    unknown = set(canonical["windings"])-set(winding_parts)
    if unknown:
        raise ValueError("Current groups refer to an absent or differently owned winding: "+", ".join(sorted(unknown)))
    if not winding_parts:
        raise ValueError("Select an installed round-lens winding")
    from temsim.coil_winding import INTERVAL_TURNS_FIELD
    from temsim.coil_winding import OWNER_CURRENT_FIELD
    master = assembly.part(owner_key).data.get(OWNER_CURRENT_FIELD)
    for key in canonical["windings"]:
        data = winding_parts[key].data
        intervals = data.get("material_intervals_mm")
        counts = data.get(INTERVAL_TURNS_FIELD)
        if counts is None:
            if intervals is not None and len(intervals) != 1:
                raise ValueError("Define fixed physical interval turns before adding internal current groups")
            counts = (data.get("winding_turn_count"),)
        cells = bind_segments(canonical, key, counts)
        if master is not None:
            for cell in cells:
                try:
                    ni = master*cell.current_ratio*cell.turn_count
                except OverflowError as error:
                    raise ValueError("Segment ampere-turns exceed the finite numerical range") from error
                if not math.isfinite(ni):
                    raise ValueError("Segment ampere-turns exceed the finite numerical range")
    descriptors = deepcopy(state.lens_field_map_descriptors)
    row = dict(descriptors.get(owner_key, {}))
    if canonical["windings"]:
        row[FIELD] = canonical
    else:
        row.pop(FIELD, None)
    if row:
        descriptors[owner_key] = row
    else:
        descriptors.pop(owner_key, None)
    state.lens_field_map_descriptors = descriptors
    # The potential digest also changes. Drop cached executable objects now so
    # no caller can retain an unqualified runtime after an electrical edit.
    for name in ("_runtime_copper_coil_provider_cache", "_unified_field_runtime_cache"):
        cache = getattr(state, name, None)
        if isinstance(cache, dict):
            cache.clear()
    return canonical
