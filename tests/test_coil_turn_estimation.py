"""Physical-size turn estimates are explicit pure authoring calculations."""
from copy import deepcopy
import math

import numpy as np
import pytest

from temsim.coil_turn_estimation import estimate_winding_turns
from temsim.coil_turn_estimation import estimate_winding_turns_from_allocated_area


REFERENCE = dict(inner_diameter_mm=60., outer_diameter_mm=132.,
                 intervals_mm=((1396.5, 1541.5), (1656.9, 1801.9)))


@pytest.mark.parametrize(('area','turns'),[(2700.,300),(4860.,540),(4590.,510),(153.,17),
                                         (208.3536,23),(174.201315,19),(5832.,648),(6480.,720)])
def test_nominal_integer_pack_area_has_distinct_unknown_metal_area(area,turns):
    target=dict(inner_diameter_mm=10.,outer_diameter_mm=12.,intervals_mm=((0.,area),))
    result=estimate_winding_turns_from_allocated_area(target,9.)
    assert result['winding_turn_count']==turns
    assert result['nominal_allocated_cross_section_mm2_per_turn']==9.
    assert result['actual_allocated_cross_section_mm2_per_turn']==area/turns
    assert result['wire_cross_section_mm2'] is None and result['wire_diameter_mm'] is None


@pytest.mark.parametrize('nominal',[0.,-1.,True,float('nan'),float('inf'),10**400])
def test_invalid_nominal_pack_area_rejects(nominal):
    with pytest.raises(ValueError):
        estimate_winding_turns_from_allocated_area(REFERENCE,nominal)


def envelope(area, *, intervals=None, inner=10.):
    return dict(inner_diameter_mm=inner, outer_diameter_mm=inner+2,
                intervals_mm=intervals or ((0., area),))


def test_reference_reproduces_exact_fixed_objective_counts_without_input_mutation():
    target = deepcopy(REFERENCE)
    reference = deepcopy(REFERENCE)
    result = estimate_winding_turns(target, reference, 1172)
    assert result['winding_turn_count'] == 1172
    assert result['winding_interval_turn_counts'] == (586, 586)
    assert result['cross_section_mm2'] == 10440.
    assert result['reference_cross_section_mm2'] == 10440.
    assert result['interval_cross_sections_mm2'] == (5220., 5220.)
    assert result['source_kind'] == 'estimated'
    assert result['wire_diameter_mm'] is None
    assert result['automatic_geometry_binding'] is False
    assert target == reference == REFERENCE


def test_single_interval_area_ratio_gives_expected_condenser_estimate():
    target = dict(inner_diameter_mm=96., outer_diameter_mm=156., intervals_mm=((455., 545.),))
    result = estimate_winding_turns(target, REFERENCE, 1172)
    assert result['cross_section_mm2'] == 2700.
    assert result['unrounded_turn_count'] == pytest.approx(303.1034482758621)
    assert result['winding_turn_count'] == 303
    assert result['winding_interval_turn_counts'] == (303,)


@pytest.mark.parametrize(('area', 'expected'), [(1.49, 1), (1.5, 2), (2.5, 3), (.01, 1)])
def test_rounding_is_nearest_half_up_with_positive_minimum(area, expected):
    result = estimate_winding_turns(envelope(area), envelope(1.), 1)
    assert result['winding_turn_count'] == expected
    assert result['minimum_interval_count_applied'] is (area < .5)


def test_mean_radius_changes_conductor_length_but_not_turn_area_estimate():
    first = estimate_winding_turns(envelope(3., inner=10.), envelope(1.), 10)
    second = estimate_winding_turns(envelope(3., inner=100.), envelope(1.), 10)
    assert first == second


def test_positive_largest_remainders_preserve_integer_budget_and_interval_order():
    target = envelope(1., intervals=((0., 1.), (2., 3.), (4., 5.)))
    result = estimate_winding_turns(target, envelope(3.), 8)
    assert result['winding_turn_count'] == 8
    assert result['winding_interval_turn_counts'] == (3, 3, 2)


def test_small_interval_has_one_turn_without_rescaling_total_count():
    target = envelope(1., intervals=((0., .01), (1., 10.99)))
    result = estimate_winding_turns(target, envelope(10.), 10)
    assert result['winding_turn_count'] == 10
    assert result['winding_interval_turn_counts'] == (1, 9)


def test_minimum_count_keeps_every_occupied_interval_positive():
    target = envelope(1., intervals=((0., .001), (1., 1.001), (2., 2.001)))
    result = estimate_winding_turns(target, envelope(10.), 1)
    assert result['winding_turn_count'] == 3
    assert result['winding_interval_turn_counts'] == (1, 1, 1)
    assert result['minimum_interval_count_applied']


def test_saved_estimate_is_not_an_automatic_geometry_relationship():
    target = envelope(10.)
    result = estimate_winding_turns(target, envelope(1.), 10)
    target['intervals_mm'] = ((0., 20.),)
    assert result['winding_turn_count'] == 100
    assert estimate_winding_turns(target, envelope(1.), 10)['winding_turn_count'] == 200


@pytest.mark.parametrize('count', [True, np.bool_(True), 0, -1, 1., '1', None, math.nan, math.inf])
def test_invalid_reference_turn_count_is_rejected(count):
    with pytest.raises(ValueError, match='positive integer'):
        estimate_winding_turns(envelope(1.), envelope(1.), count)


@pytest.mark.parametrize('invalid', [
    None, {}, dict(inner_diameter_mm=True, outer_diameter_mm=4., intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=2., outer_diameter_mm=2., intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=2., outer_diameter_mm=1., intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=0., outer_diameter_mm=2., intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=math.nan, outer_diameter_mm=2., intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=2., outer_diameter_mm=math.inf, intervals_mm=((0., 1.),)),
    dict(inner_diameter_mm=2., outer_diameter_mm=4., intervals_mm=[]),
    envelope(1., intervals=((0., 0.),)), envelope(1., intervals=((1., 0.),)),
    envelope(1., intervals=((0., 2.), (1., 3.))), envelope(1., intervals=((2., 3.), (0., 1.))),
    envelope(1., intervals=((0., True),)), envelope(1., intervals=((0., math.nan),)),
    envelope(1., intervals=((0., math.inf),)), envelope(1., intervals=((0., 1., 2.),)),
])
@pytest.mark.parametrize('side', ['target', 'reference'])
def test_invalid_geometry_is_rejected_for_either_envelope(invalid, side):
    target, reference = envelope(1.), envelope(1.)
    if side == 'target':
        target = invalid
    else:
        reference = invalid
    with pytest.raises(ValueError):
        estimate_winding_turns(target, reference, 1)


def test_integer_reference_accepts_numpy_integral():
    assert estimate_winding_turns(envelope(1.), envelope(1.), np.int64(12))['winding_turn_count'] == 12


def test_large_finite_count_allocation_has_exact_integer_budget():
    target = envelope(1., intervals=((0., 1.), (2., 3.), (4., 5.)))
    result = estimate_winding_turns(target, target, 10**100)
    counts = result['winding_interval_turn_counts']
    assert sum(counts) == result['winding_turn_count']
    assert max(counts)-min(counts) <= 1


def test_overflow_and_underflow_are_reported_without_nonfinite_witness():
    with pytest.raises(ValueError, match='finite numerical range'):
        estimate_winding_turns(envelope(1.), envelope(1.), 10**400)
    with pytest.raises(ValueError, match='underflowed'):
        estimate_winding_turns(envelope(1e-300), envelope(1e100), 1)
