"""Geometry DG criteria bind the gun and every dependent transport product."""
import json
import pytest
from temsim.calculation_cache import calculation_signatures, _state_payload
from temsim.optics.column import default_state
from temsim.optics.electron_gun import tracing


@pytest.mark.parametrize('name,value', [
    ('GEOMETRY_STEP_SCHEMA', 'test-different-geometry-criteria'),
    ('GEOMETRY_IMPLICIT_TOLERANCE', 2e-13),
    ('GEOMETRY_TOTAL_RELATIVE_MOMENTUM', 5e-7),
    ('GEOMETRY_TRANSVERSE_RELATIVE_MOMENTUM', 5e-6),
    ('GEOMETRY_TRANSVERSE_MOMENTUM_FLOOR_FRACTION', 5e-6),
    ('GEOMETRY_MOMENTUM_NORM_FLOOR', 5e-31),
    ('GEOMETRY_XY_ABSOLUTE_POSITION_M', 5e-14),
    ('GEOMETRY_XY_RELATIVE_POSITION', 5e-7),
    ('GEOMETRY_XYZ_ABSOLUTE_POSITION_M', 5e-14),
    ('GEOMETRY_XYZ_RELATIVE_STEP', 5e-6),
    ('GEOMETRY_STEP_MAXIMUM_RETRIES', 24),
])
def test_geometry_criterion_changes_invalidate_gun_and_dependent_products(monkeypatch, name, value):
    state = default_state(); gun = state.electron_gun
    before_gun = gun._cache_key(9)
    before = calculation_signatures(state)
    monkeypatch.setattr(tracing, name, value)
    after = calculation_signatures(state)
    assert gun._cache_key(9) != before_gun
    assert json.loads(gun._cache_key(9))['geometry_step_criteria'] == tracing.geometry_step_criteria()
    assert _state_payload(state)['_gun_geometry_step_criteria'] == tracing.geometry_step_criteria()
    for product in ('request', 'incident', 'column', 'elastic', 'eds', 'stem'):
        assert after[product] != before[product], product


def test_geometry_cache_metadata_is_an_owned_value():
    criteria = tracing.geometry_step_criteria()
    criteria['xy_absolute_position_m'] = 1.
    assert tracing.geometry_step_criteria()['xy_absolute_position_m'] == 1e-13
