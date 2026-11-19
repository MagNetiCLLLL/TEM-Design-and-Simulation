"""Explicit engineering estimates of fixed winding turns from R-Z area.

This optional authoring helper assumes the reference and target have the same
turn density in the winding-pack cross-section. It is not a measurement or an
automatic geometry binding. Persist the returned integers as fixed physical
inputs; later geometry edits must not silently invoke a new estimate.
"""
from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import math
from numbers import Integral, Real


def _finite(value: Real, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _envelope(envelope: Mapping, label: str) -> tuple[float, tuple[float, ...]]:
    if not isinstance(envelope, Mapping):
        raise ValueError(f"{label} must be an envelope mapping")
    try:
        inner = _finite(envelope['inner_diameter_mm'], f"{label} inner diameter")
        outer = _finite(envelope['outer_diameter_mm'], f"{label} outer diameter")
        raw = envelope['intervals_mm']
    except KeyError as error:
        raise ValueError(f"{label} requires inner/outer diameters and axial intervals") from error
    if not 0 < inner < outer:
        raise ValueError(f"{label} requires 0 < inner diameter < outer diameter")
    if not isinstance(raw, (tuple, list)) or not raw:
        raise ValueError(f"{label} requires nonempty ordered axial intervals")
    previous = -math.inf
    areas = []
    radial_thickness = (outer-inner)/2
    for interval in raw:
        if not isinstance(interval, (tuple, list)) or len(interval) != 2:
            raise ValueError(f"{label} axial intervals require two endpoints")
        start = _finite(interval[0], f"{label} interval start")
        end = _finite(interval[1], f"{label} interval end")
        if not start < end or start < previous:
            raise ValueError(f"{label} axial intervals must be positive, ordered and disjoint")
        area = radial_thickness*(end-start)
        if not math.isfinite(area) or area <= 0:
            raise ValueError(f"{label} cross-sectional area must be finite and positive")
        areas.append(area)
        previous = end
    try:
        total = math.fsum(areas)
    except OverflowError as error:
        raise ValueError(f"{label} cross-sectional area exceeds the finite range") from error
    if not math.isfinite(total):
        raise ValueError(f"{label} cross-sectional area exceeds the finite range")
    return total, tuple(areas)


def _allocate(total: int, interval_areas: tuple[float, ...]) -> tuple[int, ...]:
    # Ordinary largest remainders start from the floor of each quota. Enforce
    # one turn per occupied interval, then resolve the remaining integer budget
    # by largest quota deficit/excess. Equal priorities use increasing Z order.
    # Exact fractions of the supplied binary64 areas keep the integer budget
    # bounded even when a caller supplies a very large finite turn count.
    exact_areas = tuple(Fraction(value) for value in interval_areas)
    exact_area = sum(exact_areas)
    quotas = tuple(total*value/exact_area for value in exact_areas)
    counts = [max(1, math.floor(value)) for value in quotas]
    difference = total-sum(counts)
    while difference > 0:
        index = max(range(len(counts)), key=lambda i: (quotas[i]-counts[i], -i))
        counts[index] += 1
        difference -= 1
    while difference < 0:
        eligible = [i for i, count in enumerate(counts) if count > 1]
        index = max(eligible, key=lambda i: (counts[i]-quotas[i], -i))
        counts[index] -= 1
        difference += 1
    return tuple(counts)


def estimate_winding_turns(
    envelope: Mapping,
    reference_envelope: Mapping,
    reference_turn_count: int,
) -> dict:
    """Return a deterministic, explicitly estimated fixed turn definition.

    Both envelopes contain ``inner_diameter_mm``, ``outer_diameter_mm`` and
    nonempty increasing-Z ``intervals_mm``. The estimate is reference N times
    target/reference R-Z area, rounded to nearest integer with halves upward.
    Total N is at least the number of occupied intervals, ensuring one positive
    integer count per interval. Fractional allocations use constrained largest
    remainders with increasing-Z tie order. Mean radius/turn circumference do
    not enter this assumption. No conductor diameter or current is inferred.
    Neither input is mutated, and this function does not persist its result.
    """
    if (isinstance(reference_turn_count, bool) or not isinstance(reference_turn_count, Integral)
            or reference_turn_count <= 0):
        raise ValueError("Reference turn count must be a positive integer")
    reference_turn_count = int(reference_turn_count)
    area, interval_areas = _envelope(envelope, 'Target winding')
    reference_area, _ = _envelope(reference_envelope, 'Reference winding')
    ratio = area/reference_area
    try:
        unrounded = reference_turn_count*ratio
    except OverflowError as error:
        raise ValueError("Estimated turn count exceeds the finite numerical range") from error
    if not math.isfinite(ratio) or not math.isfinite(unrounded):
        raise ValueError("Estimated turn count exceeds the finite numerical range")
    if ratio <= 0 or unrounded <= 0:
        raise ValueError("Estimated turn count underflowed the positive numerical range")
    whole = math.floor(unrounded)
    rounded = whole + int(unrounded-whole >= .5)
    total = max(len(interval_areas), rounded)
    counts = _allocate(total, interval_areas)
    return {
        'winding_turn_count': total,
        'winding_interval_turn_counts': counts,
        'source_kind': 'estimated',
        'method': 'reference_winding_rz_area_ratio_v1',
        'source_note': 'Engineering estimate assuming equal turns per winding-pack R-Z area; not measured or OEM data.',
        'reference_turn_count': reference_turn_count,
        'reference_cross_section_mm2': reference_area,
        'cross_section_mm2': area,
        'interval_cross_sections_mm2': interval_areas,
        'cross_section_ratio': ratio,
        'unrounded_turn_count': unrounded,
        'rounding': 'nearest_half_up; minimum one turn per occupied interval',
        'interval_allocation': 'positive_largest_remainders; increasing_Z_ties',
        'minimum_interval_count_applied': total != rounded,
        'wire_diameter_mm': None,
        'automatic_geometry_binding': False,
    }


def estimate_winding_turns_from_allocated_area(envelope: Mapping, allocated_area_mm2: Real) -> dict:
    """Estimate fixed N using an explicit nominal pack allocation per turn.

    This area includes insulation and unused pack space. It is not conductor
    metal area, a wire diameter, or a minimum CAD cell size. Integer rounding
    can leave actual envelope area/N above or below the nominal allocation.
    Photo-constrained counts must be preserved by the caller rather than
    replaced with this engineering estimate. Neither geometry nor current is
    changed by this pure authoring calculation.
    """
    nominal = _finite(allocated_area_mm2, 'Nominal allocated area per turn')
    if nominal <= 0:
        raise ValueError('Nominal allocated area per turn must be positive')
    area, interval_areas = _envelope(envelope, 'Target winding')
    unrounded = area/nominal
    if not math.isfinite(unrounded) or unrounded <= 0:
        raise ValueError('Estimated turn count is outside the positive finite numerical range')
    whole = math.floor(unrounded)
    rounded = whole + int(unrounded-whole >= .5)
    total = max(len(interval_areas), rounded)
    return dict(winding_turn_count=total,
                winding_interval_turn_counts=_allocate(total, interval_areas),
                source_kind='estimated', method='nominal_allocated_area_per_turn_v1',
                nominal_allocated_cross_section_mm2_per_turn=nominal,
                actual_allocated_cross_section_mm2_per_turn=area/total,
                cross_section_mm2=area, interval_cross_sections_mm2=interval_areas,
                unrounded_turn_count=unrounded,
                rounding='nearest_half_up; minimum one turn per occupied interval',
                interval_allocation='positive_largest_remainders; increasing_Z_ties',
                minimum_interval_count_applied=total != rounded,
                wire_cross_section_mm2=None, wire_diameter_mm=None,
                automatic_geometry_binding=False)
