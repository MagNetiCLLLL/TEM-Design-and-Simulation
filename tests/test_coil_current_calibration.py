"""Fixed-N single-plane field calibration, without emission or transport."""
from copy import copy, deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.coil_current_calibration import current_for_field_match, calibrate_round_lens_current
from temsim.coil_winding import OWNER_CURRENT_FIELD


@pytest.fixture(autouse=True)
def forbid_source_transport(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Current calibration must not emit, transport or focus particles')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit', forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit', forbidden)


@pytest.fixture(scope='module')
def baseline():
    from temsim.optics.column import default_state
    return default_state()


def snapshot(baseline):
    state = copy(baseline)
    state.lenses = deepcopy(baseline.lenses)
    return state


def edit(state, key, updates=None, remove=()):
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(p, data={**{k: v for k, v in p.data.items() if k not in remove}, **(updates or {})})
        if p.key == key else p for p in state._resolved_assembly.parts))


def known_turns(state, key, count=100):
    coils = [p for p in state._resolved_assembly.parts if p.data.get('mechanical_profile') == 'magnetic_excitation_coil'
             and p.data.get('field_source_key') == key]
    assert len(coils) == 1
    edit(state, coils[0].key, {'winding_turn_count': count},
         remove=('winding_interval_turn_counts', 'winding_wire_diameter_mm'))
    return coils[0].key


@pytest.mark.parametrize('polarity', [1, -1])
def test_scalar_solution_keeps_fixed_polarity_and_percent(polarity):
    result = current_for_field_match(polarity*.2, .001, operating_percent=25., polarity=polarity)
    assert result['current_at_100_percent_a'] == 800.
    assert result['current_a'] == polarity*200.


def test_scalar_zero_target_and_large_finite_current_are_valid():
    assert current_for_field_match(0., .1)['current_at_100_percent_a'] == 0.
    assert current_for_field_match(1., 1e-100)['current_at_100_percent_a'] == 1e100


@pytest.mark.parametrize('changes', [
    {'unit_current_bz_t': 0.}, {'target_bz_t': float('nan')}, {'unit_current_bz_t': float('inf')},
    {'target_bz_t': True}, {'operating_percent': 0.}, {'operating_percent': 101.},
    {'operating_percent': True}, {'operating_percent': float('nan')}, {'enabled': False},
    {'enabled': 1}, {'polarity': 0}, {'polarity': True}, {'target_bz_t': -.2},
    {'target_bz_t': 1e308, 'unit_current_bz_t': 1e-300},
    {'target_bz_t': 1e-300, 'unit_current_bz_t': 1e308},
    {'target_bz_t': 10**400},
    {'operating_percent': np.nextafter(0., 1.)},
])
def test_scalar_unidentifiable_invalid_or_unrepresentable_cases_reject(changes):
    values = dict(target_bz_t=.2, unit_current_bz_t=.001)
    values.update(changes)
    with pytest.raises(ValueError):
        current_for_field_match(**values)


@pytest.mark.parametrize('historical',[False,True],ids=['integer_area_default_580','historical_photo_586'])
def test_selected_objective_surface_recipe_preserves_current_controls_and_live_caches(baseline,historical):
    state = snapshot(baseline)
    region_turns=586 if historical else 580
    expected_reference=1575.2330285262728*586/region_turns
    if historical:
        edit(state,'objective_lens_excitation_coil',{'winding_turn_count':1172,'winding_interval_turn_counts':[586,586]})
        edit(state,'objective_lens',{OWNER_CURRENT_FIELD:1575.2330285262728})
    else:
        assert state._resolved_assembly.part('objective_lens_excitation_coil').data['winding_turn_count']==1160
    lens = next(l for l in state.lenses if l.key == 'objective_lens')
    lens.percent = 68.97810600833917
    sentinel = object()
    state._runtime_lens_field_provider_cache = {'original': sentinel}
    state._runtime_copper_coil_provider_cache = {'original': sentinel}
    state._field_provider_diagnostics = {'original': sentinel}
    before = deepcopy(state.to_dict()); assembly = state._resolved_assembly
    result = calibrate_round_lens_current(state, lens.key, matching_plane_z_mm=1599.1999975)
    np.testing.assert_allclose(result['current_at_100_percent_a'], expected_reference, rtol=2e-14, atol=0)
    np.testing.assert_allclose(result['current_a'], 1086.565908295224*586/region_turns, rtol=2e-14, atol=0)
    np.testing.assert_allclose(result['calibrated_operating_bz_t'], result['original_bz_t'], rtol=2e-14, atol=0)
    assert result['total_turn_count'] == region_turns*2
    assert state.to_dict() == before and state._resolved_assembly is assembly
    assert state._runtime_lens_field_provider_cache == {'original': sentinel}
    assert state._runtime_copper_coil_provider_cache == {'original': sentinel}
    assert state._field_provider_diagnostics == {'original': sentinel}


@pytest.mark.parametrize('key,polarity', [('condenser_lens_2', 1), ('probe_tl22_lens', -1)])
def test_nonobjective_peak_plane_matches_signed_field_and_scales_inverse_to_turns(baseline, key, polarity):
    state = snapshot(baseline)
    coil = known_turns(state, key)
    before = deepcopy(state.to_dict())
    first = calibrate_round_lens_current(state, key)
    assert first['matching_condition'] == 'native_gaussian_absolute_peak_plane'
    assert first['reference_polarity'] == polarity
    assert first['current_at_100_percent_a'] > 0
    np.testing.assert_allclose(first['calibrated_operating_bz_t'], first['original_bz_t'], rtol=2e-14, atol=0)
    assert state.to_dict() == before
    edit(state, coil, {'winding_turn_count': 200})
    second = calibrate_round_lens_current(state, key, matching_plane_z_mm=first['matching_plane_z_mm'])
    np.testing.assert_allclose(second['current_at_100_percent_a'], first['current_at_100_percent_a']/2, rtol=2e-14, atol=0)


@pytest.mark.parametrize('percent,enabled', [(0., True), (37., False)])
def test_zero_or_disabled_workpoint_gets_explicit_reference_without_enabling_live_lens(baseline, percent, enabled):
    state = snapshot(baseline); known_turns(state, 'intermediate_lens')
    lens = next(l for l in state.lenses if l.key == 'intermediate_lens')
    lens.percent = percent; lens.enabled = enabled
    original = deepcopy(state.to_dict())
    result = calibrate_round_lens_current(state, lens.key)
    assert result['current_at_100_percent_a'] > 0
    assert result['current_a'] == result['original_bz_t'] == result['calibrated_operating_bz_t'] == 0.
    assert result['reference_excitation_percent'] == 100. and result['reference_enabled']
    assert result['zero_operating_excitation']
    assert result['original_excitation_percent'] == percent and result['original_enabled'] == enabled
    assert state.to_dict() == original


def test_unknown_turns_and_invalid_explicit_plane_reject_without_inference(baseline):
    state = snapshot(baseline)
    edit(state, 'intermediate_lens_excitation_coil', remove=('winding_turn_count', 'winding_interval_turn_counts', 'winding_wire_diameter_mm'))
    with pytest.raises(ValueError, match='define turns'):
        calibrate_round_lens_current(state, 'intermediate_lens')
    known_turns(state, 'intermediate_lens')
    with pytest.raises(ValueError, match='finite'):
        calibrate_round_lens_current(state, 'intermediate_lens', matching_plane_z_mm=float('nan'))
    with pytest.raises(ValueError, match='nonzero'):
        calibrate_round_lens_current(state, 'intermediate_lens', matching_plane_z_mm=1e10)


def test_map_recipe_and_posed_source_reject_before_replacement_or_solve(baseline):
    state = snapshot(baseline); known_turns(state, 'intermediate_lens')
    state.lens_field_map_descriptors = {'intermediate_lens': {'solver': 'axisymmetric_nonlinear_fem'}}
    with pytest.raises(ValueError, match='mapped or FEM'):
        calibrate_round_lens_current(state, 'intermediate_lens')
    state.lens_field_map_descriptors = {}
    edit(state, 'intermediate_lens', {'offset_x_mm': .01})
    with pytest.raises(ValueError, match='aligned'):
        calibrate_round_lens_current(state, 'intermediate_lens')


def test_overridden_nonlinear_native_field_is_not_extrapolated(baseline, monkeypatch):
    state = snapshot(baseline); known_turns(state, 'intermediate_lens')
    lens = next(item for item in state.lenses if item.key == 'intermediate_lens')
    native_type = type(lens)
    linear = native_type.magnetic_field_t
    monkeypatch.setattr(native_type, 'magnetic_field_t', lambda self, z: linear(self, z)*(self.percent*.01))
    with pytest.raises(ValueError, match='not linear'):
        calibrate_round_lens_current(state, lens.key)
