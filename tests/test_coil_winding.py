"""Pure winding-definition and circle-packing tests; no field or transport."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import math

import numpy as np
import pytest

from temsim.coil_winding import (
    INTERVAL_TURNS_FIELD, MAXIMUM_LAYOUT_TURNS, OWNER_CURRENT_FIELD, WIRE_AREA_FIELD, WINDING_FIELDS, WINDING_LAYOUT_SCHEMA,
    has_winding_definition, resolve_allocated_winding_layout,
    resolve_wire_cross_section_mm2, resolve_wire_diameter_mm,
    resolve_winding_layout, validate_interval_turn_counts, validate_winding_definition,
)


def _layout(count=100, diameter=.2):
    return resolve_winding_layout(10., 14., ((-2., 2.),), count, diameter)


def _check_circles(layout):
    radius = layout.wire_diameter_mm/2.
    for turn in layout.turns:
        allocation = layout.allocations[turn.interval_index]
        assert turn.radius_mm-radius >= allocation.inner_diameter_mm/2.-1e-13
        assert turn.radius_mm+radius <= allocation.outer_diameter_mm/2.+1e-13
        assert turn.z_mm-radius >= allocation.start_mm-1e-13
        assert turn.z_mm+radius <= allocation.end_mm+1e-13
    for index, first in enumerate(layout.turns):
        for second in layout.turns[index+1:]:
            distance = math.hypot(first.radius_mm-second.radius_mm, first.z_mm-second.z_mm)
            assert distance >= layout.wire_diameter_mm-1e-13


def test_legacy_fields_remain_absent_and_electrical_turns_need_no_invented_diameter():
    record = {"key": "legacy", "air_core_reference_ampere_turns": 123.}
    before = deepcopy(record)
    assert not has_winding_definition(record)
    assert validate_winding_definition(record) is None
    assert record == before
    electrical = {"winding_turn_count": 586}
    assert has_winding_definition(electrical)
    assert validate_winding_definition(electrical) is None
    assert "winding_wire_diameter_mm" not in electrical
    for partial in ({"winding_wire_diameter_mm": .2}, {INTERVAL_TURNS_FIELD: [586]}):
        assert has_winding_definition(partial)
        with pytest.raises(ValueError, match="requires winding_turn_count"):
            validate_winding_definition(partial)


def test_canonical_metal_area_resolves_round_wire_without_mutating_inputs():
    record = dict(winding_turn_count=12, winding_wire_cross_section_mm2=.2)
    before = deepcopy(record)
    assert WIRE_AREA_FIELD in WINDING_FIELDS
    assert validate_winding_definition(record) is None
    assert resolve_wire_diameter_mm(record) == pytest.approx(math.sqrt(.8/math.pi), rel=2e-16)
    assert resolve_wire_cross_section_mm2(record) == .2
    assert record == before and 'winding_wire_diameter_mm' not in record
    assert resolve_wire_diameter_mm({'winding_turn_count':586}) is None
    assert resolve_wire_cross_section_mm2({'winding_turn_count':586}) is None


def test_legacy_wire_diameter_and_consistent_dual_declaration_remain_readable():
    legacy = dict(winding_turn_count=12, winding_wire_diameter_mm=.4)
    area = math.pi*(.4/2)**2
    assert resolve_wire_diameter_mm(legacy) == .4
    assert resolve_wire_cross_section_mm2(legacy) == area
    both = dict(legacy, winding_wire_cross_section_mm2=area)
    assert validate_winding_definition(both) is None
    assert resolve_wire_diameter_mm(both) == .4
    assert resolve_wire_cross_section_mm2(both) == area
    with pytest.raises(ValueError,match='inconsistent'):
        validate_winding_definition(dict(legacy,winding_wire_cross_section_mm2=area*1.01))


@pytest.mark.parametrize('area', [True, np.bool_(True), 0., -1., '0.2', None,
                               math.nan, math.inf, -math.inf, 10**400])
def test_invalid_conductor_metal_area_is_rejected(area):
    with pytest.raises(ValueError):
        validate_winding_definition(dict(winding_turn_count=12, winding_wire_cross_section_mm2=area))


def test_metal_area_without_turn_count_is_not_ignored_as_legacy_data():
    record = {WIRE_AREA_FIELD:.2}
    assert has_winding_definition(record)
    with pytest.raises(ValueError,match='requires winding_turn_count'):
        validate_winding_definition(record)


@pytest.mark.parametrize('area',[np.nextafter(0.,1.),1e-300,1e300,np.finfo(float).max])
def test_area_to_diameter_avoids_intermediate_overflow_or_underflow(area):
    record = {WIRE_AREA_FIELD:area}
    diameter = resolve_wire_diameter_mm(record)
    assert math.isfinite(diameter) and diameter > 0
    assert resolve_wire_cross_section_mm2(record) == area


def test_unrepresentable_legacy_area_is_rejected_by_area_readback():
    for diameter in (1e-300,1e200):
        with pytest.raises(ValueError,match='positive finite range'):
            resolve_wire_cross_section_mm2({'winding_wire_diameter_mm':diameter})


def test_fixed_physical_interval_counts_validate_without_wire_or_redistribution():
    record = dict(winding_turn_count=1172, winding_interval_turn_counts=[586, 586])
    before = deepcopy(record)
    assert validate_winding_definition(record) is None
    assert validate_interval_turn_counts(record, 2) == (586, 586)
    assert record == before
    with pytest.raises(ValueError, match="one count per actual winding interval"):
        validate_interval_turn_counts(record, 1)
    assert validate_interval_turn_counts({"winding_turn_count": 1172}) is None


@pytest.mark.parametrize("counts", [[], [586], [585, 586], [587, 586], [0, 1172],
                                    [-1, 1173], [True, 1171], [586., 586.], "586,586", None])
def test_invalid_fixed_interval_definitions_are_rejected(counts):
    with pytest.raises(ValueError):
        validate_winding_definition(dict(winding_turn_count=1172,
                                        winding_interval_turn_counts=counts))


@pytest.mark.parametrize("count", [1, 100, np.int64(100)])
def test_definition_accepts_only_explicit_integer_count_without_mutation(count):
    record = dict(winding_turn_count=count, winding_wire_diameter_mm=.2)
    before = deepcopy(record)
    assert validate_winding_definition(record) is None
    assert record == before


@pytest.mark.parametrize("count", [True, np.bool_(True), 0, -1, 1., 1.5, "100", None, math.nan, math.inf])
def test_invalid_turn_count_is_rejected(count):
    with pytest.raises(ValueError, match="positive integer"):
        _layout(count)


@pytest.mark.parametrize("diameter", [True, np.bool_(True), 0., -1., "0.2", None, math.nan, math.inf, -math.inf])
def test_invalid_wire_diameter_is_rejected(diameter):
    with pytest.raises(ValueError, match="diameter"):
        _layout(diameter=diameter)


@pytest.mark.parametrize("current", [0., 2., np.float64(2.)])
def test_independent_owner_current_is_nonnegative(current):
    record = {OWNER_CURRENT_FIELD: current}
    assert validate_winding_definition(record) is None
    assert not has_winding_definition(record)


@pytest.mark.parametrize("current", [True, np.bool_(True), -.1, math.nan, math.inf, "2", None])
def test_invalid_owner_current_is_rejected(current):
    with pytest.raises(ValueError, match="current"):
        validate_winding_definition({OWNER_CURRENT_FIELD: current})


@pytest.mark.parametrize("record", [None, (), 1, "invalid"])
def test_definition_requires_mapping(record):
    with pytest.raises(ValueError, match="mapping"):
        validate_winding_definition(record)
    with pytest.raises(ValueError, match="mapping"):
        has_winding_definition(record)


def test_layout_is_deterministic_frozen_and_metadata_is_an_independent_summary():
    first = _layout()
    assert first == _layout()
    assert len(first.turns) == first.turn_count == 100
    assert first.centre_pitch_mm >= first.wire_diameter_mm
    _check_circles(first)
    with pytest.raises(FrozenInstanceError):
        first.turn_count = 9
    with pytest.raises(FrozenInstanceError):
        first.turns[0].z_mm = 9
    metadata = first.to_metadata()
    assert metadata["schema"] == WINDING_LAYOUT_SCHEMA
    assert "turns" not in metadata
    assert json.loads(json.dumps(metadata)) == metadata
    metadata["allocations"][0]["radial_turn_counts"].clear()
    assert first.allocations[0].radial_turn_counts


def test_original_envelope_and_balanced_rows_are_preserved():
    layout = resolve_winding_layout(20., 24., ((-3., -1.), (2., 6.)), 37, .25)
    assert layout.intervals_mm == ((-3., -1.), (2., 6.))
    assert sum(layout.interval_turn_counts) == 37
    for allocation in layout.allocations:
        assert allocation.inner_diameter_mm == 20.
        assert allocation.outer_diameter_mm == 24.
        counts = allocation.radial_turn_counts
        assert sum(counts) == allocation.turn_count
        assert max(counts)-min(counts) <= 1
    _check_circles(layout)


def test_different_inner_diameters_share_pitch_and_never_enter_passive_allocation():
    allocations = ((62., 70., -4., -2.), (60., 70., 2., 5.))
    layout = resolve_allocated_winding_layout(allocations, 61, .3)
    assert layout.allocations[0].inner_diameter_mm == 62.
    assert layout.allocations[1].inner_diameter_mm == 60.
    _check_circles(layout)
    for allocation in layout.allocations:
        local = [turn for turn in layout.turns if turn.interval_index == layout.allocations.index(allocation)]
        for row in range(allocation.axial_rows):
            radii = [turn.radius_mm for turn in local if turn.axial_row == row]
            for left, right in zip(radii, radii[1:]):
                assert right-left == pytest.approx(layout.centre_pitch_mm, abs=1e-14)


def test_capacity_proportional_integer_residual_is_deterministic_and_conserved():
    layout = resolve_winding_layout(10., 12., ((0., 1.), (2., 3.)), 5, .2)
    assert layout.interval_turn_counts == (3, 2)
    assert len(layout.turns) == 5
    _check_circles(layout)


def test_fixed_interval_counts_override_capacity_shares_without_resizing():
    geometry = ((-2., -1.), (2., 6.))
    layout = resolve_winding_layout(10., 14., geometry, 12, .25,
                                    interval_turn_counts=[9, 3])
    assert layout.interval_turn_counts == (9, 3)
    assert layout.intervals_mm == geometry
    assert len(layout.turns) == 12
    assert layout == resolve_winding_layout(10., 14., geometry, 12, .25,
                                            interval_turn_counts=[9, 3])
    _check_circles(layout)


def test_fixed_counts_reject_local_overcapacity_instead_of_moving_turns():
    # The second region has ample spare room; it cannot absorb the first's
    # explicitly fixed physical turns to make an impossible layout pass.
    allocations = ((10., 12., 0., 1.), (10., 12., 2., 12.))
    with pytest.raises(ValueError, match="physical interval turns cannot be moved"):
        resolve_allocated_winding_layout(allocations, 30, .25,
                                         allocation_turn_counts=[29, 1])


@pytest.mark.parametrize("counts", [[1], [1, 1], [True, 11], [-1, 13], [1., 11.]])
def test_invalid_fixed_allocation_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        resolve_allocated_winding_layout(((10., 14., 0., 2.), (10., 14., 3., 5.)),
                                         12, .25, allocation_turn_counts=counts)


def test_explicit_empty_conductor_allocation_keeps_insulation_without_losing_turns():
    layout = resolve_allocated_winding_layout(((10., 14., 0., 2.), (10., 14., 3., 5.)),
                                              3, .25, allocation_turn_counts=[0, 3])
    assert layout.interval_turn_counts == (0, 3)
    assert len(layout.turns) == 3
    assert all(turn.interval_index == 1 for turn in layout.turns)
    assert layout.allocations[0].radial_turn_counts == ()


def test_zero_turn_allocation_retains_insulation_and_does_not_invent_turns():
    layout = resolve_winding_layout(10., 12., ((0., 1.), (2., 3.)), 1, .2)
    assert layout.interval_turn_counts == (1, 0)
    assert layout.allocations[1].radial_turn_counts == ()
    assert layout.area.cross_section_area_mm2 == 2.
    assert len(layout.turns) == 1


def test_area_and_torus_volume_close_without_discrete_field_assumptions():
    layout = _layout(8, .25)
    wire_area = math.pi*(.25/2.)**2
    assert layout.area.cross_section_area_mm2 == 8.
    assert layout.area.conductor_cross_section_area_mm2 == pytest.approx(8*wire_area)
    assert layout.area.conductor_volume_mm3 == pytest.approx(
        sum(2*math.pi*turn.radius_mm*wire_area for turn in layout.turns))
    assert layout.area.allocation_volume_mm3 == pytest.approx(math.pi*(7**2-5**2)*4)
    assert (layout.area.conductor_cross_section_area_mm2+layout.area.insulation_cross_section_area_mm2
            == pytest.approx(layout.area.cross_section_area_mm2))
    assert (layout.area.conductor_volume_mm3+layout.area.insulation_volume_mm3
            == pytest.approx(layout.area.allocation_volume_mm3))
    assert 0 < layout.area.area_fill_fraction < 1
    assert 0 < layout.area.volume_fill_fraction < 1


def test_exact_tangent_full_grid_and_small_decimal_dimensions():
    layout = resolve_winding_layout(10., 12., ((0., 1.),), 16, .25)
    assert layout.centre_pitch_mm == pytest.approx(.25)
    assert layout.interwire_gap_mm == pytest.approx(0., abs=1e-14)
    _check_circles(layout)
    decimal = resolve_winding_layout(10., 10.6, ((0., .3),), 9, .1)
    assert len(decimal.turns) == 9
    _check_circles(decimal)


@pytest.mark.parametrize("args", [
    (10., 12., ((0., 1.),), 17, .25),
    (10., 12., ((0., 1.),), 1, 1.01),
    (10., 12., ((0., .1),), 1, .2),
])
def test_overcapacity_is_explicit_and_does_not_resize_envelope(args):
    with pytest.raises(ValueError, match="capacity"):
        resolve_winding_layout(*args)


@pytest.mark.parametrize("allocations", [
    (), ((10., 12., 1., 0.),), ((10., 10., 0., 1.),), ((-1., 12., 0., 1.),),
    ((0., 12., 0., 1.),), ((10., math.inf, 0., 1.),),
    ((10., 12., math.nan, 1.),), ((10., 12., 0., 1.), (10., 12., .5, 2.)),
    ((10., 12., 2., 3.), (10., 12., 0., 1.)), ((10., 12., 0.),), "bad",
])
def test_bad_or_overlapping_allocations_are_rejected(allocations):
    with pytest.raises(ValueError):
        resolve_allocated_winding_layout(allocations, 1, .1)


@pytest.mark.parametrize("intervals", [None, "bad", ((0.,),), ((0., 1., 2.),)])
def test_invalid_interval_sequences_are_rejected(intervals):
    with pytest.raises(ValueError):
        resolve_winding_layout(10., 12., intervals, 1, .1)


@pytest.mark.parametrize("diameter", [1e-300, 1e200])
def test_unrepresentable_area_is_rejected_clearly(diameter):
    with pytest.raises(ValueError, match="area"):
        _layout(1, diameter)


def test_inputs_are_not_modified_and_generator_intervals_are_supported():
    allocations = [[10., 12., 0., 1.], [10., 12., 2., 3.]]
    before = deepcopy(allocations)
    layout = resolve_allocated_winding_layout(allocations, 8, .1)
    assert allocations == before
    other = resolve_winding_layout(10., 12., ((a, b) for a, b in ((0., 1.), (2., 3.))), 8, .1)
    assert layout == other


def test_huge_layout_rejected_before_geometry_or_turn_generation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("The resource gate must precede geometry and turn materialization")

    monkeypatch.setattr("temsim.coil_winding._geometry", forbidden)
    monkeypatch.setattr("temsim.coil_winding.WindingTurn", forbidden)
    with pytest.raises(ValueError, match="preallocation resource budget"):
        resolve_allocated_winding_layout(((10., 12., 0., 1.),), 10**9, 1e-9)

    def intervals():
        raise AssertionError("The resource gate must precede consuming intervals")
        yield (0., 1.)

    with pytest.raises(ValueError, match="preallocation resource budget"):
        resolve_winding_layout(10., 12., intervals(), 10**9, 1e-9)


def test_layout_budget_is_explicit_and_does_not_restrict_mathematical_definition():
    record = dict(winding_turn_count=10**9, winding_wire_diameter_mm=1e-9,
                  ideal_coil_current_a_at_100_percent=2.)
    assert validate_winding_definition(record) is None
    assert MAXIMUM_LAYOUT_TURNS == 65_536
    with pytest.raises(ValueError, match="budget is 4"):
        resolve_winding_layout(10., 12., ((0., 1.),), 5, .1, maximum_layout_turns=4)
    layout = resolve_allocated_winding_layout(((10., 12., 0., 1.),), 5, .1, maximum_layout_turns=5)
    assert len(layout.turns) == 5


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, math.inf, None])
def test_invalid_layout_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        resolve_winding_layout(10., 12., ((0., 1.),), 1, .1, maximum_layout_turns=budget)
