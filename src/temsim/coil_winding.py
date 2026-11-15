"""Pure, explicit round-wire winding definitions and mechanical packing.

Lengths are millimetres.  Each closed turn is a circular wire cross-section
in the R-Z plane, revolved around the column axis.  This is a deterministic
orthogonal packing, not a claim of maximally dense circle packing.  Its common
centre pitch is at least the wire diameter; unoccupied allocation is insulation.
No current, magnetic field, default turn count or historical conversion is
inferred here.  Field homogenization may use a different, original envelope.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
import sys


OWNER_CURRENT_FIELD = "ideal_coil_current_a_at_100_percent"
INTERVAL_TURNS_FIELD = "winding_interval_turn_counts"
WIRE_AREA_FIELD = "winding_wire_cross_section_mm2"
WINDING_FIELDS = frozenset({"winding_turn_count", "winding_wire_diameter_mm", WIRE_AREA_FIELD, INTERVAL_TURNS_FIELD})
WINDING_LAYOUT_SCHEMA = "temsim-round-winding-layout-v1"
# A synchronous document validator must not materialize an unbounded turn table.
# This is an allocation budget, not a physical turn-count or ampere-turn limit.
# CAD mesh/STEP operations retain their own, potentially smaller, budgets.
MAXIMUM_LAYOUT_TURNS = 65_536
_EPSILON = sys.float_info.epsilon


def _number(value, label, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if (not math.isfinite(result) or (positive and result <= 0.)
            or (nonnegative and result < 0.)):
        qualifier = "positive " if positive else "non-negative " if nonnegative else ""
        raise ValueError(f"{label} must be a finite {qualifier}real number")
    return result


def _turn_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Winding turn count must be a positive integer")
    return int(value)


def _wire_size(data: Mapping) -> tuple[float | None, float | None]:
    """Resolve round conductor size without changing its declared inputs.

    Metal area is the canonical user input. Legacy diameter remains readable;
    simultaneous declarations must describe the same round cross-section.
    The small consistency guard covers binary64 conversion only, not CAD
    clearance, physical dimensions or a relaxed field tolerance.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Winding definition must be a mapping")
    diameter = (_number(data["winding_wire_diameter_mm"], "Winding wire diameter", positive=True)
                if "winding_wire_diameter_mm" in data else None)
    area = (_number(data[WIRE_AREA_FIELD], "Winding conductor metal cross-section", positive=True)
            if WIRE_AREA_FIELD in data else None)
    if area is not None:
        # Taking the square root first avoids overflow in 4*A and underflow
        # in A/pi at the positive subnormal boundary.
        area_diameter = math.sqrt(area)*(2./math.sqrt(math.pi))
        if not math.isfinite(area_diameter) or area_diameter <= 0:
            raise ValueError("Winding conductor cross-section has no finite positive round diameter")
        if diameter is not None and not math.isclose(diameter, area_diameter, rel_tol=8.*_EPSILON, abs_tol=0.):
            raise ValueError("Winding conductor metal cross-section and legacy wire diameter are inconsistent")
        return area_diameter if diameter is None else diameter, area
    if diameter is None:
        return None, None
    try:
        derived_area = math.pi*(diameter/2.)**2
    except OverflowError:
        derived_area = math.inf
    return diameter, derived_area


def resolve_wire_diameter_mm(data: Mapping) -> float | None:
    """Return a round diameter from metal area or a legacy diameter declaration.

    Unknown conductor size remains None. This geometry conversion neither
    infers insulation/fill nor changes turn count, envelope or electrical current.
    """
    return _wire_size(data)[0]


def resolve_wire_cross_section_mm2(data: Mapping) -> float | None:
    """Return exact declared metal area, or derive it from a legacy round wire."""
    area = _wire_size(data)[1]
    if area is not None and (not math.isfinite(area) or area <= 0):
        raise ValueError("Winding conductor metal cross-section is outside the positive finite range")
    return area


def has_winding_definition(data: Mapping) -> bool:
    """Return whether any optional winding field is explicitly present.

    A partial declaration returns True so that callers cannot silently treat it
    as legacy geometry.  Use :func:`validate_winding_definition` for completeness.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Winding definition must be a mapping")
    return bool(WINDING_FIELDS.intersection(data))


def validate_interval_turn_counts(data: Mapping, interval_count=None) -> tuple[int, ...] | None:
    """Validate optional fixed physical counts, without length redistribution.

    Geometry owners supply ``interval_count`` when their authoritative field
    intervals are available. A bare electrical definition may validate the
    exact integer sum before that geometry context is resolved.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Winding definition must be a mapping")
    if INTERVAL_TURNS_FIELD not in data:
        return None
    if "winding_turn_count" not in data:
        raise ValueError(f"{INTERVAL_TURNS_FIELD} requires winding_turn_count")
    count = _turn_count(data["winding_turn_count"])
    raw = data[INTERVAL_TURNS_FIELD]
    if not isinstance(raw, (tuple, list)) or not raw:
        raise ValueError(f"{INTERVAL_TURNS_FIELD} must be a nonempty list of positive integers")
    counts = tuple(_turn_count(value) for value in raw)
    if sum(counts) != count:
        raise ValueError(f"{INTERVAL_TURNS_FIELD} must sum to winding_turn_count")
    if interval_count is not None and len(counts) != interval_count:
        raise ValueError(f"{INTERVAL_TURNS_FIELD} must have one count per actual winding interval")
    return counts


def validate_winding_definition(data: Mapping) -> None:
    """Validate optional explicit fields without mutating or supplying values.

    Turn count is sufficient for electrical excitation. Conductor metal area
    or a legacy bare diameter is optional and only needed for detailed CAD. Interval
    counts, when present, preserve physical turns in each source region. An
    empty legacy record remains valid without inferred values.
    """
    if not isinstance(data, Mapping):
        raise ValueError("Winding definition must be a mapping")
    if "winding_turn_count" in data:
        _turn_count(data["winding_turn_count"])
    if "winding_wire_diameter_mm" in data or WIRE_AREA_FIELD in data:
        if "winding_turn_count" not in data:
            raise ValueError("Winding conductor size requires winding_turn_count")
        resolve_wire_diameter_mm(data)
    validate_interval_turn_counts(data)
    if OWNER_CURRENT_FIELD in data:
        _number(data[OWNER_CURRENT_FIELD], "Coil current at 100% excitation", nonnegative=True)


@dataclass(frozen=True, slots=True)
class WindingTurn:
    radius_mm: float
    z_mm: float
    interval_index: int
    axial_row: int
    radial_column: int


@dataclass(frozen=True, slots=True)
class WindingAllocation:
    """One unchanged rectangular annular conductor allocation."""

    inner_diameter_mm: float
    outer_diameter_mm: float
    start_mm: float
    end_mm: float
    turn_count: int
    radial_rows: int
    axial_rows: int
    radial_turn_counts: tuple[int, ...]
    radial_boundary_gap_mm: float
    axial_boundary_gap_mm: float

    def to_metadata(self) -> dict:
        return dict(inner_diameter_mm=self.inner_diameter_mm,
                    outer_diameter_mm=self.outer_diameter_mm,
                    start_mm=self.start_mm, end_mm=self.end_mm,
                    turn_count=self.turn_count, radial_rows=self.radial_rows,
                    axial_rows=self.axial_rows,
                    radial_turn_counts=list(self.radial_turn_counts),
                    radial_boundary_gap_mm=self.radial_boundary_gap_mm,
                    axial_boundary_gap_mm=self.axial_boundary_gap_mm)


@dataclass(frozen=True, slots=True)
class WindingAreaWitness:
    """Actual wire area/volume and remaining unspecified insulation."""

    cross_section_area_mm2: float
    conductor_cross_section_area_mm2: float
    insulation_cross_section_area_mm2: float
    allocation_volume_mm3: float
    conductor_volume_mm3: float
    insulation_volume_mm3: float

    @property
    def area_fill_fraction(self) -> float:
        return self.conductor_cross_section_area_mm2/self.cross_section_area_mm2

    @property
    def volume_fill_fraction(self) -> float:
        return self.conductor_volume_mm3/self.allocation_volume_mm3

    def to_metadata(self) -> dict:
        return dict(cross_section_area_mm2=self.cross_section_area_mm2,
                    conductor_cross_section_area_mm2=self.conductor_cross_section_area_mm2,
                    insulation_cross_section_area_mm2=self.insulation_cross_section_area_mm2,
                    allocation_volume_mm3=self.allocation_volume_mm3,
                    conductor_volume_mm3=self.conductor_volume_mm3,
                    insulation_volume_mm3=self.insulation_volume_mm3,
                    area_fill_fraction=self.area_fill_fraction,
                    volume_fill_fraction=self.volume_fill_fraction)


@dataclass(frozen=True, slots=True)
class WindingLayout:
    """Immutable exact turns, retained allocations and packing witnesses."""

    turn_count: int
    wire_diameter_mm: float
    centre_pitch_mm: float
    turns: tuple[WindingTurn, ...]
    allocations: tuple[WindingAllocation, ...]
    area: WindingAreaWitness

    @property
    def interwire_gap_mm(self) -> float:
        return max(0., self.centre_pitch_mm-self.wire_diameter_mm)

    @property
    def interval_turn_counts(self) -> tuple[int, ...]:
        return tuple(item.turn_count for item in self.allocations)

    @property
    def intervals_mm(self) -> tuple[tuple[float, float], ...]:
        return tuple((item.start_mm, item.end_mm) for item in self.allocations)

    def to_metadata(self) -> dict:
        """Return a fresh JSON-compatible summary, without a per-turn payload."""
        return dict(schema=WINDING_LAYOUT_SCHEMA, units="mm",
                    packing="balanced-orthogonal-common-pitch",
                    turn_count=self.turn_count, wire_diameter_mm=self.wire_diameter_mm,
                    centre_pitch_mm=self.centre_pitch_mm,
                    interwire_gap_mm=self.interwire_gap_mm,
                    interval_turn_counts=list(self.interval_turn_counts),
                    allocations=[item.to_metadata() for item in self.allocations],
                    **self.area.to_metadata())


def _sequence(value, label):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise ValueError(f"{label} must be a sequence") from error


def _geometry(allocations):
    result = []
    for index, entry in enumerate(_sequence(allocations, "Winding allocations")):
        values = _sequence(entry, f"Winding allocation {index}")
        if len(values) != 4:
            raise ValueError("Each winding allocation requires (ID, OD, start, end) in millimetres")
        inner, outer, start, end = tuple(_number(value, "Winding allocation coordinate") for value in values)
        if not 0. < inner < outer or not start < end:
            raise ValueError("Winding allocations require 0 < ID < OD and start < end")
        if result and start < result[-1][3]:
            raise ValueError("Winding allocations must have ordered, disjoint axial intervals")
        width, length = (outer-inner)/2., end-start
        if not math.isfinite(width) or not math.isfinite(length) or width <= 0.:
            raise ValueError("Winding allocation dimensions must be finite and positive")
        result.append((inner, outer, start, end))
    if not result:
        raise ValueError("At least one winding allocation is required")
    return tuple(result)


def _axis_capacity(extent, diameter, pitch):
    # Only a few ulps compensate coordinate subtraction at an exact tangency.
    # This is a floating-point geometry guard, not a physical clearance budget.
    difference = extent-diameter
    guard = 8.*_EPSILON*max(extent, diameter)
    if difference < -guard:
        return 0
    ratio = max(0., difference)/pitch
    if not math.isfinite(ratio):
        raise ValueError("Winding grid capacity is outside the finite numerical range")
    nearest = round(ratio)
    if abs(ratio-nearest) <= 8.*_EPSILON*max(1., ratio):
        ratio = nearest
    return math.floor(ratio)+1


def _capacities(geometry, diameter, pitch):
    return tuple((_axis_capacity((outer-inner)/2., diameter, pitch),
                  _axis_capacity(end-start, diameter, pitch))
                 for inner, outer, start, end in geometry)


def _common_pitch(geometry, diameter, count, fixed_counts=None):
    capacities = _capacities(geometry, diameter, diameter)
    capacity = sum(radial*axial for radial, axial in capacities)
    if count > capacity:
        raise ValueError(f"Winding needs {count} turns but orthogonal packing capacity is {capacity}")
    if fixed_counts is not None:
        for index, (required, (radial, axial)) in enumerate(zip(fixed_counts, capacities)):
            if required > radial*axial:
                raise ValueError(f"Fixed winding allocation {index} needs {required} turns but its capacity is {radial*axial}; "
                                 "physical interval turns cannot be moved to another allocation")
    low = diameter
    high = max(diameter, max(max((outer-inner)/2., end-start)
                             for inner, outer, start, end in geometry))
    # The finite upper bound is just a convention for singleton allocations;
    # with two or more centres it also exceeds every possible occupied span.
    for _ in range(80):
        mid = low+(high-low)/2.
        if mid == low or mid == high:
            break
        trial = _capacities(geometry, diameter, mid)
        feasible = (sum(radial*axial for radial, axial in trial) >= count if fixed_counts is None else
                    all(radial*axial >= required for (radial, axial), required in zip(trial, fixed_counts)))
        if feasible:
            low = mid
        else:
            high = mid
    return low, _capacities(geometry, diameter, low)


def _integer_allocation(count, capacities):
    total = sum(capacities)
    quotients = [divmod(count*capacity, total) for capacity in capacities]
    result = [quotient for quotient, _ in quotients]
    residual = count-sum(result)
    for index in sorted(range(len(result)), key=lambda i: (-quotients[i][1], i))[:residual]:
        result[index] += 1
    return tuple(result)


def _fixed_allocation_counts(values, count, allocation_count):
    if values is None:
        return None
    raw = _sequence(values, "Fixed winding allocation counts")
    if len(raw) != allocation_count:
        raise ValueError("Fixed winding allocation counts require one entry per allocation")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError("Fixed winding allocation counts must be nonnegative integers")
        result.append(int(value))
    if sum(result) != count:
        raise ValueError("Fixed winding allocation counts must sum to the total turn count")
    return tuple(result)


def _row_counts(count, rows):
    base, extra = divmod(count, rows)
    result = [base]*rows
    # Spread the residual rows over the span rather than accumulating them at
    # either axial end.  Row counts differ by at most one.
    for index in range(extra):
        result[((2*index+1)*rows)//(2*extra)] += 1
    return tuple(result)


def _finite_positive(value, label):
    if not math.isfinite(value) or value <= 0.:
        raise ValueError(f"{label} is outside the finite positive numerical range")
    return value


def _positive_sum(values, label):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} is outside the finite positive numerical range") from error
    return _finite_positive(result, label)


def resolve_allocated_winding_layout(allocations, turn_count, wire_diameter_mm, *,
                                     allocation_turn_counts=None,
                                     maximum_layout_turns=MAXIMUM_LAYOUT_TURNS) -> WindingLayout:
    """Pack explicit turns into ordered (ID, OD, z0, z1) allocations in mm.

    All intervals use a common square-grid centre pitch, chosen as large as the
    requested count permits.  Counts use capacity-proportional largest
    remainders, with input order breaking ties, unless explicit physical
    ``allocation_turn_counts`` are supplied. Those counts constrain every
    allocation and are never moved to another region. Each allocation has centred
    rows with balanced integer radial counts.  The envelopes are never resized.
    A count beyond this declared orthogonal packing capacity is rejected;
    hexagonal packing, leads, bends and wire enamel are not inferred.
    ``maximum_layout_turns`` is an explicit preallocation resource budget.  It
    does not restrict mathematical definitions validated without a turn table.
    """
    count = _turn_count(turn_count)
    budget = _turn_count(maximum_layout_turns)
    if count > budget:
        raise ValueError(f"Winding layout needs {count} turns but its preallocation resource budget is {budget}; "
                         "no turns were generated or dropped")
    diameter = _number(wire_diameter_mm, "Winding wire diameter", positive=True)
    geometry = _geometry(allocations)
    wire_area = _finite_positive(math.pi*(diameter/2.)*(diameter/2.), "Wire cross-section area")
    fixed_counts = _fixed_allocation_counts(allocation_turn_counts, count, len(geometry))
    pitch, capacities = _common_pitch(geometry, diameter, count, fixed_counts)
    counts = (fixed_counts if fixed_counts is not None else
              _integer_allocation(count, tuple(radial*axial for radial, axial in capacities)))
    turns, resolved = [], []
    for index, ((inner, outer, start, end), (radial_capacity, _), local_count) in enumerate(
            zip(geometry, capacities, counts)):
        radial_mid = inner/2.+(outer-inner)/4.
        axial_mid = start+(end-start)/2.
        axial_rows = (local_count+radial_capacity-1)//radial_capacity if local_count else 0
        rows = _row_counts(local_count, axial_rows) if axial_rows else ()
        for row, row_count in enumerate(rows):
            z = axial_mid+(row-(axial_rows-1)/2.)*pitch
            for column in range(row_count):
                radius = radial_mid+(column-(row_count-1)/2.)*pitch
                turns.append(WindingTurn(radius, z, index, row, column))
        radial_rows = max(rows, default=0)
        radial_gap = ((outer-inner)/2.-diameter-(radial_rows-1)*pitch)/2. if rows else (outer-inner)/4.
        axial_gap = ((end-start)-diameter-(axial_rows-1)*pitch)/2. if rows else (end-start)/2.
        # Tangencies may carry a few negative ulps after subtracting spans.
        resolved.append(WindingAllocation(inner, outer, start, end, local_count,
                        radial_rows, axial_rows, rows, max(0., radial_gap), max(0., axial_gap)))
    area = _positive_sum(((outer-inner)/2.*(end-start)
                          for inner, outer, start, end in geometry), "Winding area")
    conductor_area = _finite_positive(count*wire_area, "Conductor cross-section area")
    volume = _positive_sum((math.pi*((outer-inner)/2.)*(inner+(outer-inner)/2.)*(end-start)
                            for inner, outer, start, end in geometry), "Allocation volume")
    conductor_volume = _positive_sum((2.*math.pi*turn.radius_mm*wire_area for turn in turns),
                                     "Conductor volume")
    if conductor_area > area or conductor_volume > volume:
        raise ValueError("Winding conductor exceeds its allocation")
    witness = WindingAreaWitness(area, conductor_area, area-conductor_area,
                                volume, conductor_volume, volume-conductor_volume)
    return WindingLayout(count, diameter, pitch, tuple(turns), tuple(resolved), witness)


def resolve_winding_layout(inner_diameter_mm, outer_diameter_mm, intervals_mm,
                           turn_count, wire_diameter_mm, *,
                           interval_turn_counts=None,
                           maximum_layout_turns=MAXIMUM_LAYOUT_TURNS) -> WindingLayout:
    """Pack a constant-ID/OD winding without changing any axial interval."""
    # Check before consuming a possibly large/generator interval collection.
    count = _turn_count(turn_count)
    budget = _turn_count(maximum_layout_turns)
    if count > budget:
        raise ValueError(f"Winding layout needs {count} turns but its preallocation resource budget is {budget}; "
                         "no turns were generated or dropped")
    intervals = _sequence(intervals_mm, "Winding intervals")
    allocations = []
    for interval in intervals:
        pair = _sequence(interval, "Winding interval")
        if len(pair) != 2:
            raise ValueError("Each winding interval requires (start, end) in millimetres")
        allocations.append((inner_diameter_mm, outer_diameter_mm, *pair))
    if interval_turn_counts is not None:
        interval_turn_counts = validate_interval_turn_counts(
            {"winding_turn_count": count, INTERVAL_TURNS_FIELD: interval_turn_counts}, len(allocations))
    return resolve_allocated_winding_layout(allocations, count, wire_diameter_mm,
                                           allocation_turn_counts=interval_turn_counts,
                                           maximum_layout_turns=budget)
