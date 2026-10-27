"""Thermionic validation rejects invalid source data before physical evaluation."""

from copy import deepcopy
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.optics.column import default_state
from temsim.optics.model import State
from temsim.optics.electron_gun.thermionic import (
    ThermionicEmitter,
    ThermionicGun,
    thermionic_gun_from_dict,
)
from temsim.profile_io import apply_profile_values


@pytest.fixture
def emitter():
    return ThermionicEmitter(0.0, 1.0, 1.0)


@pytest.mark.parametrize("name", [
    "cathode_temperature_k", "work_function_ev",
    "richardson_constant_a_cm2_k2", "vacuum_pa", "emitting_radius_um",
    "cathode_anode_gap_mm", "extraction_field_scale",
    "mechanical_center_from_tip_mm", "mechanical_length_mm",
    "mechanical_outer_diameter_mm",
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_source_parameters_are_rejected(emitter, name, value):
    setattr(emitter, name, value)
    with pytest.raises(ValueError, match=name):
        emitter.validate()


@pytest.mark.parametrize("name,value", [
    ("cathode_temperature_k", "1800"),
    ("work_function_ev", True),
    ("emitting_radius_um", np.bool_(True)),
    ("vacuum_pa", None),
])
def test_source_numbers_are_not_coerced_from_other_types(emitter, name, value):
    setattr(emitter, name, value)
    with pytest.raises(ValueError, match=name):
        emitter.validate()


@pytest.mark.parametrize("value", [
    8, 0, -1, 9.0, 9.9, True, False, np.bool_(True), "9",
    None, float("nan"), float("inf"), np.float64(9),
])
def test_configured_ray_count_requires_an_integer_of_at_least_nine(emitter, value):
    emitter.ray_count = value
    with pytest.raises(ValueError, match="ray count"):
        emitter.validate()
    with pytest.raises(ValueError, match="ray count"):
        emitter.emit(9)


@pytest.mark.parametrize("value", [
    8, -1, 9.0, 9.9, True, np.bool_(True), "9", float("nan"), float("inf"),
])
def test_emit_override_does_not_truncate_or_coerce_count(emitter, value):
    with pytest.raises(ValueError, match="ray count"):
        emitter.emit(value)


@pytest.mark.parametrize("name,value", [
    ("work_function_ev", float("nan")),
    ("cathode_temperature_k", float("inf")),
    ("emitting_radius_um", float("nan")),
    ("richardson_constant_a_cm2_k2", float("nan")),
    ("extraction_field_scale", float("nan")),
])
def test_emit_and_current_diagnostics_reject_invalid_live_source(name, value):
    gun = ThermionicGun()
    setattr(gun.emitter, name, value)
    with pytest.raises(ValueError, match=name):
        gun.emit(9)
    with pytest.raises(ValueError, match=name):
        _ = gun.emitted_current_a
    with pytest.raises(ValueError, match=name):
        gun.emitter.temperature_limited_current_density_a_m2(8e5)


@pytest.fixture(scope="module")
def gun_payload():
    return ThermionicGun().to_dict()


@pytest.mark.parametrize("name,value", [
    ("work_function_ev", float("nan")),
    ("cathode_temperature_k", float("inf")),
    ("ray_count", 9.9), ("ray_count", True), ("ray_count", "9"),
])
def test_gun_import_rejects_invalid_source_without_changing_payload(gun_payload, name, value):
    payload = deepcopy(gun_payload)
    row = payload["components"]["thermionic_cathode"]
    row[name] = value
    before = repr(payload)
    with pytest.raises(ValueError):
        thermionic_gun_from_dict(payload)
    assert repr(payload) == before


@pytest.fixture(scope="module")
def thermionic_state():
    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, replace(catalog.default_selection(), gun="Thermionic"))
    return state


@pytest.mark.parametrize("name,value", [
    ("work_function_ev", float("nan")),
    ("emitting_radius_um", float("inf")),
    ("ray_count", 9.9), ("ray_count", True), ("ray_count", "9"),
])
def test_state_import_rejects_invalid_thermionic_source(thermionic_state, name, value):
    payload = thermionic_state.to_dict()
    payload["electron_gun"]["components"]["thermionic_cathode"][name] = value
    with pytest.raises(ValueError):
        State.from_dict(payload)


@pytest.mark.parametrize("name,value", [
    ("work_function_ev", float("nan")), ("ray_count", 9.9),
    ("ray_count", True), ("ray_count", "9"),
])
def test_profile_boundary_remains_strict_and_transactional(thermionic_state, name, value):
    source = thermionic_state.electron_gun.emitter
    before = dict(vars(source))
    with pytest.raises(ValueError):
        apply_profile_values(thermionic_state, {source.key: {name: value}})
    assert vars(source) == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_scalar_current_helpers_reject_nonfinite_external_fields(emitter, value):
    with pytest.raises(ValueError):
        emitter.schottky_lowering_ev(value)
    with pytest.raises(ValueError):
        emitter.child_langmuir_current_density_a_m2(value, 1e-3)
    with pytest.raises(ValueError):
        emitter.child_langmuir_current_density_a_m2(800, value)


def test_zero_boundaries_and_numpy_numeric_inputs_remain_supported(emitter):
    emitter.work_function_ev = np.float64(0.0)
    emitter.vacuum_pa = 0
    emitter.extraction_field_scale = 0.0
    emitter.ray_count = np.int64(9)
    assert emitter.validate() is emitter
    bundle = emitter.emit(np.int64(9))
    assert len(bundle.x_m) == 9
    np.testing.assert_array_equal(bundle.weight, np.full(9, 1 / 9))


def test_default_current_and_thermal_energy_formulas_are_unchanged():
    gun = ThermionicGun()
    source = gun.emitter
    # An independently written scalar RLD/Schottky/CL calculation in SI units.
    charge, mass = 1.602176634e-19, 9.1093837015e-31
    permittivity, boltzmann = 8.8541878128e-12, 1.380649e-23
    temperature, voltage, gap = 1800.0, 800.0, 1e-3
    lowering = math.sqrt(charge * (voltage / gap) / (4 * math.pi * permittivity))
    supply = 29.0e4 * temperature**2 * math.exp(-(2.7 - lowering) / (boltzmann * temperature / charge))
    ceiling = 4 / 9 * permittivity * math.sqrt(2 * charge / mass) * voltage**1.5 / gap**2
    expected_current = math.pi * (10e-6)**2 * min(supply, ceiling)
    assert gun.emitted_current_a == pytest.approx(expected_current, rel=2e-12)
    assert source.emission_energy_ev == pytest.approx(2 * boltzmann * temperature / charge, rel=1e-15)
    before = source.emit(17)
    restored = thermionic_gun_from_dict(gun.to_dict())
    after = restored.emit(17)
    for name in vars(before):
        np.testing.assert_array_equal(getattr(before, name), getattr(after, name))
    assert source.ray_count == 3000

