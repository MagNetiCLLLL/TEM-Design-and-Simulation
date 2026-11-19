"""Copper consumer configuration/field checks; no source or wave propagation."""
from copy import copy, deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.magnetic_field_scene import prepare_magnetic_scene
from temsim.optics.column import default_state
from temsim.physics.annular_coil_potential import AnnularCoilPotential
from temsim.physics.copper_coil_field_provider import CopperCoilAssemblyPotential, CopperCoilFieldProvider
from temsim.physics.frozen_copper_coil_field import FrozenCopperCoilField
from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
from temsim.physics.lens_field_provider import CoordinateRegistration, LensGeometryBinding, freeze_vector_provider


def _provider(*, z=.1, current=3000., registration=CoordinateRegistration()):
    coil = AnnularCoilPotential("test:coil", .006, .01, .01, current, z, registration)
    potential = CopperCoilAssemblyPotential("test-lens", (coil,))
    return CopperCoilFieldProvider("test-lens", potential,
                                  LensGeometryBinding("test-lens", "geometry", "assembly", "{}"))


def _scene(monkeypatch, provider, **kwargs):
    lens = SimpleNamespace(key=provider.lens_key, enabled=True, name="Synthetic copper lens")
    state = SimpleNamespace(lenses=(lens,), _field_provider_diagnostics={})
    monkeypatch.setattr("temsim.magnetic_field_scene._resolved_provider", lambda *_: provider)
    return prepare_magnetic_scene(state, **kwargs)


@pytest.fixture(scope="module")
def installed():
    state = default_state()
    state.simulation_mode = "unified"
    state.round_lens_field_model = "copper_coil"
    return state


def _copy_state(state):
    result = copy(state)
    result.lenses = deepcopy(state.lenses)
    result._runtime_copper_coil_provider_cache = {}
    result._runtime_lens_field_provider_cache = {}
    return result


def test_frozen_adapter_keeps_exact_field_and_all_axial_tails():
    provider = _provider()
    frozen = freeze_vector_provider(provider)
    assert type(frozen) is FrozenCopperCoilField
    assert frozen.axial_domain_all_z
    points = np.array([[.001, -.002, -.2], [0., 0., .1], [0., 0., .6]])
    np.testing.assert_array_equal(frozen.field_at_global_positions_t(points),
                                  provider.field_at_global_positions_t(points))
    assert np.all(np.linalg.norm(frozen.field_at_global_positions_t(points), axis=1) > 0.)
    assert frozen.field_support_mm == pytest.approx((95., 105.))
    assert 0 < frozen.maximum_step_mm(100., 3e-22) <= 2.
    assert frozen.implementation_is_supported()


def test_scene_explicit_window_outside_winding_retains_exact_field(monkeypatch):
    provider = _provider()
    scene = _scene(monkeypatch, provider, z_limits_mm=(0., 500.))
    points = np.array([[.001, 0., .001], [.001, 0., .4]])
    assert scene.contains(points).all()
    expected = provider.field_at_global_positions_t(points)
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), expected)
    for point, field in zip(points, expected):
        np.testing.assert_array_equal(scene.diagnostic_field_at_global_position_t(point), field)
    assert scene.physical_identity and scene.numerical_identity
    assert scene.support_metadata[0].model == "air_core_annular_copper_coil"
    assert not scene.contains([[0., 0., .6]])[0]


def test_scene_includes_tails_from_every_coil_in_overlapping_display(monkeypatch):
    provider = _provider()
    second = replace(provider.potential.coils[0], source_id="test:second", center_z_m=.3)
    provider = replace(provider, potential=CopperCoilAssemblyPotential("test-lens", (provider.potential.coils[0], second)))
    scene = _scene(monkeypatch, provider)
    point = np.array([[0., 0., .2]])
    assert scene.contains(point).all()
    np.testing.assert_array_equal(scene.field_at_global_positions_t(point), provider.field_at_global_positions_t(point))


def test_scene_posed_bore_uses_registered_cylinder_and_rejects_copper(monkeypatch):
    angle = .03
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    registration = CoordinateRegistration((.02, -.01, 0.), tuple(map(tuple, rotation)))
    provider = _provider(registration=registration)
    scene = _scene(monkeypatch, provider, z_limits_mm=(0., 400.))
    local = np.array([[0., 0., .1], [.003, 0., .2]])
    points = local@rotation.T+registration.origin_array_m
    # The explicit finite display may clip a tilted axis; use an enclosing
    # bounded view to isolate the registered-cylinder contract.
    scene = replace(scene, bounds_m=np.array([[-.1, -.1, 0.], [.1, .1, .4]]))
    assert scene.contains(points).all()
    np.testing.assert_array_equal(scene.field_at_global_positions_t(points), provider.field_at_global_positions_t(points))
    copper = np.array([[.006, 0., .1]])@rotation.T+registration.origin_array_m
    assert not scene.contains(copper)[0]


def test_instrument_capture_has_full_tails_and_range_dependency(monkeypatch):
    from temsim.physics.instrument_magnetic import InstrumentMagneticField
    provider = _provider()
    scene = _scene(monkeypatch, provider)
    captured = InstrumentMagneticField(scene._sources, scene.physical_identity, scene.numerical_identity)
    points = np.array([[0., 0., -.2], [0., 0., .6]])
    np.testing.assert_array_equal(captured.field_at_global_positions_t(points), provider.field_at_global_positions_t(points))
    before = captured.identity_for_axial_range(0., .01)
    changed = _scene(monkeypatch, _provider(current=1500.))
    changed = InstrumentMagneticField(changed._sources, changed.physical_identity, changed.numerical_identity)
    assert before and changed.identity_for_axial_range(0., .01) != before


def test_custom_copper_callback_cannot_claim_native_identity(monkeypatch):
    provider = _provider()
    monkeypatch.setattr(CopperCoilAssemblyPotential, "field_at_global_positions_t", lambda self, p: np.zeros_like(p))
    scene = _scene(monkeypatch, provider)
    assert scene.physical_identity is None and scene.numerical_identity is None


def test_installed_copper_graph_and_gun_dependency_capture_without_transport(installed):
    from temsim.physics.gun_field_environment import instrument_gun_field_context
    from temsim.physics.transverse_coil_field import FrozenTransverseCoilField
    state = _copy_state(installed)
    field = capture_instrument_magnetic_field(state)
    assert field.physical_identity and field.numerical_identity
    point = np.array([[0., 0., -0.01]])
    expected = sum((source.provider.potential.field_at_global_positions_t(point)
                    for source in field._sources
                    if isinstance(source.provider, (FrozenCopperCoilField, FrozenTransverseCoilField))),
                   start=np.zeros_like(point))
    np.testing.assert_array_equal(field.field_at_global_positions_t(point), expected)
    assert np.linalg.norm(expected) > 0.
    with instrument_gun_field_context(state):
        assert state.electron_gun._instrument_magnetic_identity
        assert (-np.inf, np.inf) in state.electron_gun._instrument_magnetic_supports_mm
        assert state.electron_gun.integration_step_mm_at(np.array([0.])) > 0.
    assert not hasattr(state.electron_gun, "_instrument_magnetic_field")


def test_far_downstream_copper_edit_invalidates_gun_dependency_without_transport(installed, monkeypatch):
    from temsim.physics.particle_sections import gun_dependency_signature
    state = _copy_state(installed)
    monkeypatch.setattr(type(state.electron_gun), "trace_to_exit",
                        lambda *_a, **_k: pytest.fail("Dependency inspection must not transport particles"))
    before = gun_dependency_signature(state)
    next(lens for lens in state.lenses if lens.key == "projector_lens_2").percent *= .5
    after = gun_dependency_signature(state)
    assert before and after and before != after


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
@pytest.mark.parametrize("pose", [{"offset_x_mm": .001}, {"rotation_y_mrad": .1}])
def test_posed_copper_memory_admission_without_calculation(installed, quality, pose):
    from temsim.gui.calculation_controller import estimate_calculation_memory_bytes
    state = _copy_state(installed)
    state._resolved_assembly = replace(state._resolved_assembly, parts=tuple(
        replace(part, data={**part.data, **pose}) if part.key == "objective_lens" else part
        for part in state._resolved_assembly.parts))
    before = [(lens.key, lens.percent, lens.polarity) for lens in state.lenses]
    estimate = estimate_calculation_memory_bytes(state, quality, 1024, .2, workflow="rays")
    assert estimate > 0
    assert [(lens.key, lens.percent, lens.polarity) for lens in state.lenses] == before


def test_saved_map_recipe_does_not_override_selected_copper_scene(installed):
    state = _copy_state(installed)
    state.lens_field_map_descriptors = {"objective_lens": {"solver": "axisymmetric_linear_fem"}}
    scene = prepare_magnetic_scene(state)
    row = next(source for source in scene._sources if source.key == "objective_lens")
    assert type(row.provider) is FrozenCopperCoilField


def test_field_vector_scene_preserves_selected_copper_tails(monkeypatch):
    from temsim.field_vector_scene import _magnetic_sample
    provider = _provider()
    scene = _scene(monkeypatch, provider)
    points = np.array([[0., 0., -.2], [0., 0., .6]])
    valid = np.ones(len(points), dtype=bool)
    actual = _magnetic_sample(scene._sources, points, valid)
    assert valid.all()
    np.testing.assert_array_equal(actual, provider.field_at_global_positions_t(points))
