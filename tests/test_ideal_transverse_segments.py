"""Source-free HP current segments: physical count, identity and CUDA parity."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

from temsim.ideal_transverse_segments import FIELD, bind_recipe, closed_layers, radial_recipe, set_transverse_current_segments
from temsim.physics.transverse_coil_field import TransverseCoilPotential, FrozenTransverseCoilField


@pytest.fixture(autouse=True)
def forbid_physical_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Electrical field checks must not emit, transport a source or construct a wave")
    monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)
    monkeypatch.setattr("temsim.physics.particle_wave_source.build_particle_wave_checkpoint", forbidden)


def hp(**changes):
    from test_radial_multipole_symmetry import _coils
    return TransverseCoilPotential(**dict(dict(source_id="probe_hp1_hexapole", coils=_coils(3, 2),
        turn_count_per_coil=2, currents_a=(7.25, -3.5), pack_order=4, path_order=24,
        provenance={"multipole_order": 3}), **changes))


def points():
    return np.array(((0., 0., .004), (.001, -.002, .007), (2e-5, 1e-5, -.3)))


def test_all_one_is_bitwise_legacy_field_jet_and_analytic_pack():
    from temsim.physics.analytic_rectangle_coil import pack_analytic_rectangle_potential
    original = hp()
    changed = replace(original, current_segments=radial_recipe(2))
    for method in ("sample", "sample_jet"):
        for key, value in getattr(original, method)(points()).items():
            np.testing.assert_array_equal(value, getattr(changed, method)(points())[key])
    a, b = map(pack_analytic_rectangle_potential, (original, changed))
    for name in ("starts_m", "ends_m", "weights_t_m"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert original.digest != changed.digest
    assert changed.turn_count_per_coil == 2 and changed.coils == original.coils


def test_signed_zero_NI_is_not_zero_field_and_drives_are_not_layers():
    value = replace(hp(), current_segments=radial_recipe(2, current_ratios=(1., -1.)))
    layers = closed_layers(value)
    assert [layer.turn_count_per_coil for layer in layers] == [1, 1]
    assert layers[0].currents_a == value.currents_a
    assert layers[1].currents_a == tuple(-i for i in value.currents_a)
    assert sum(layer.turn_count_per_coil*layer.currents_a[0] for layer in layers) == 0
    assert np.linalg.norm(value.sample(points())["B_t"][1]) > 0
    for method in ("sample", "sample_jet"):
        for key, actual in getattr(value, method)(points()).items():
            expected = sum(getattr(layer, method)(points())[key] for layer in layers)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-17)


@pytest.mark.parametrize("ratios", ((0., 0.), (0., -2.), (1., -1.)))
def test_full_parent_material_domain_and_strict_axis_survive_signed_currents(ratios):
    value = replace(hp(), current_segments=radial_recipe(2, current_ratios=ratios))
    coil = value.coils[0]
    angle = np.deg2rad(coil["azimuth_deg"])
    material = np.array((.02*np.cos(angle), .02*np.sin(angle), .01105))
    for method in ("sample", "sample_jet"):
        with pytest.raises(ValueError, match="occupied winding pack"):
            getattr(value, method)(material)
    for key in ("A_t_m", "B_t"):
        np.testing.assert_array_equal(value.sample(points()[:1])[key], np.zeros((1, 3)))


def test_v2_snapshot_roundtrip_detachment_and_tamper_identity():
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    value = replace(hp(), current_segments=radial_recipe(2, current_ratios=(2., -.5)))
    restored = TransverseCoilPotential.from_dict(json.loads(json.dumps(value.to_dict())))
    assert restored.digest == value.digest
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    snapshot = ElectromagneticPotentialSnapshot(scene, (PotentialContribution(value.source_id, (value.source_id,), value, "magnetic"),))
    again = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert again.numerical_digest == snapshot.numerical_digest
    tampered = value.to_dict()
    tampered["payload"]["current_segments"]["segments"][0]["current_ratio"] = 3.
    with pytest.raises(ValueError, match="identity mismatch"):
        TransverseCoilPotential.from_dict(tampered)
    with pytest.raises(TypeError):
        value.current_segments["segments"][0]["current_ratio"] = 3.


@pytest.mark.parametrize("case", ("schema", "extra", "turns", "density", "gap", "overlap", "bool", "nan", "huge", "axial"))
def test_invalid_recipes_are_rejected(case):
    recipe = radial_recipe(2)
    if case == "schema": recipe["schema"] = "unknown"
    elif case == "extra": recipe["extra"] = 1
    elif case == "turns": recipe["segments"][0]["turn_count"] = 2
    elif case == "density":
        recipe["segments"][0]["radial_fraction"] = [0., .25]
        recipe["segments"][1]["radial_fraction"] = [.25, 1.]
    elif case == "gap": recipe["segments"][1]["radial_fraction"] = [.6, 1.]
    elif case == "overlap": recipe["segments"][1]["radial_fraction"] = [.4, 1.]
    elif case == "bool": recipe["segments"][0]["current_ratio"] = True
    elif case == "nan": recipe["segments"][0]["current_ratio"] = float("nan")
    elif case == "huge": recipe["segments"][0]["current_ratio"] = 10**1000
    else: recipe["segments"][0]["axial_fraction"] = recipe["segments"][0].pop("nested_fraction")
    with pytest.raises(ValueError):
        bind_recipe(recipe, 2)


def test_compiled_field_and_complete_Gun_pack_preserve_parent_domain_and_tails():
    from temsim.physics.compiled_transverse_coil import pack_transverse_source, transverse_b_batch
    from temsim.physics.compiled_copper_instrument import pack_copper_instrument_sources, copper_instrument_batch
    from temsim.magnetic_field_scene import _Source
    value = replace(hp(), current_segments=radial_recipe(2, current_ratios=(1., -1.)))
    field = FrozenTransverseCoilField(value, value.source_id, 3e-22)
    packed = pack_transverse_source(field)
    assert packed is not None and len(packed[2]) == len(value.coils)
    admitted, actual = transverse_b_batch(points(), packed)
    assert admitted
    np.testing.assert_allclose(actual, value.sample(points())["B_t"], rtol=5e-12, atol=2e-18)
    source = _Source(field.key, field, field.bounds_m, field.radial_support_m)
    combined = pack_copper_instrument_sources((source,))
    assert combined is not None
    admitted, actual = copper_instrument_batch(points(), combined)
    assert admitted
    np.testing.assert_allclose(actual, value.sample(points())["B_t"], rtol=5e-12, atol=2e-18)


@pytest.fixture
def state():
    from temsim.optics.column import default_state
    result = default_state()
    result.simulation_mode = "unified"
    result.round_lens_field_model = "copper_coil"
    return result


def test_setter_capture_preserves_inputs_and_rejects_overflow_atomically(state):
    from temsim.physics.multipole_coil_field import capture_multipole_coil_fields
    key = "probe_hp1_hexapole"
    original = capture_multipole_coil_fields(state, component_keys={key})[0].potential
    before = deepcopy(state.to_dict())
    set_transverse_current_segments(state, key, radial_recipe(2, current_ratios=(1., -1.)))
    actual = capture_multipole_coil_fields(state, component_keys={key})[0].potential
    assert actual.coils == original.coils and actual.turn_count_per_coil == 2
    assert actual.currents_a == original.currents_a and actual.registration == original.registration
    assert actual.digest != original.digest and before["lenses"] == state.to_dict()["lenses"]
    saved = deepcopy(state.lens_field_map_descriptors)
    with pytest.raises(ValueError, match="finite numerical range"):
        set_transverse_current_segments(state, key, radial_recipe(2, current_ratios=(1e308, 1.)))
    assert state.lens_field_map_descriptors == saved
    set_transverse_current_segments(state, key, None)
    assert FIELD not in state.lens_field_map_descriptors.get(key, {})


def test_state_and_TOML_preserve_detached_HP_electrical_inputs(state, tmp_path):
    from temsim.optics.model import State
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    key = "probe_hp2_hexapole"
    recipe = radial_recipe(2, current_ratios=(2., -1.))
    set_transverse_current_segments(state, key, recipe)
    payload = state.to_dict()
    restored = State.from_dict(payload)
    payload["lens_field_map_descriptors"][key][FIELD]["segments"][0]["current_ratio"] = 7.
    assert restored.lens_field_map_descriptors[key][FIELD] == recipe
    path = tmp_path/"hp-segments.toml"
    selection = AssemblyCatalog().default_selection()
    save_profile(path, state, selection)
    loaded_selection, values = read_profile(path)
    fresh = State.from_dict(state.to_dict())
    apply_profile_values(fresh, values)
    assert loaded_selection == selection and fresh.lens_field_map_descriptors[key][FIELD] == recipe


@pytest.mark.parametrize("mode,model", (("ideal", "copper_coil"), ("nonlinear_material", "copper_coil"), ("unified", "gaussian")))
def test_other_modes_ignore_and_retain_loaded_HP_recipe(state, mode, model):
    from temsim.ideal_transverse_segments import apply_to_potential
    state.simulation_mode = mode
    state.round_lens_field_model = model
    state.lens_field_map_descriptors["probe_hp1_hexapole"] = {FIELD: {"invalid": "dormant"}}
    original = hp()
    assert apply_to_potential(state, original) is original
    assert state.lens_field_map_descriptors["probe_hp1_hexapole"][FIELD] == {"invalid": "dormant"}
    with pytest.raises(ValueError):
        set_transverse_current_segments(state, "probe_hp1_hexapole", radial_recipe(2))


def test_actual_CUDA_signed_layers_full_fields_identity_and_cache():
    from temsim.physics.compute_backend import cupy_capability
    from temsim.physics.cuda_coil_potential import prepare_cuda_coil_potential
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    capability = cupy_capability()
    if not capability.available:
        pytest.skip(capability.detail)
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    def prepare(value):
        snapshot = ElectromagneticPotentialSnapshot(scene, (PotentialContribution(value.source_id, (value.source_id,), value, "magnetic"),))
        return prepare_cuda_coil_potential(snapshot, execution_backend="Require GPU")
    value = replace(hp(), current_segments=radial_recipe(2, current_ratios=(1., -1.)))
    sampler = prepare(value)
    actual = sampler.sample(points())
    for key, expected in value.sample(points()).items():
        np.testing.assert_allclose(actual[key], expected, rtol=5e-12, atol=2e-18)
    np.testing.assert_array_equal(actual["A_t_m"][0], np.zeros(3))
    np.testing.assert_array_equal(actual["B_t"][0], np.zeros(3))
    fields = sampler.sample_fields(points())
    assert "A_t_m" not in fields
    np.testing.assert_array_equal(fields["B_t"], actual["B_t"])
    receipt = sampler.execution_receipt()
    assert receipt["counts"]["gpu_analytic_rectangle_source_calls"] > 0
    path = receipt["transverse_paths"][value.source_id]
    assert path["physical_turn_count_per_radial_coil"] == 2
    assert path["internal_closed_current_recipe"]["segments"][1]["current_ratio"] == -1.
    other = prepare(replace(value, current_segments=radial_recipe(2, current_ratios=(1., -.5))))
    assert other is not sampler and other.original.numerical_digest != sampler.original.numerical_digest
    assert not np.array_equal(other.sample(points())["B_t"], actual["B_t"])
    # The explicit magnetic/Gun API must use the same signed-layer pack.
    from temsim.magnetic_field_scene import _Source
    from temsim.physics.instrument_magnetic import InstrumentMagneticField
    from temsim.physics.cuda_coil_potential import prepare_cuda_instrument_magnetic
    field = FrozenTransverseCoilField(value, value.source_id, 3e-22)
    magnetic = InstrumentMagneticField((_Source(field.key, field, field.bounds_m, field.radial_support_m),), None, None)
    magnetic_gpu = prepare_cuda_instrument_magnetic(magnetic, execution_backend="Require GPU")
    query = np.repeat(points()[1:2], 128, axis=0)
    np.testing.assert_allclose(magnetic_gpu.field_at_global_positions_t(query),
        value.sample(query)["B_t"], rtol=5e-12, atol=2e-18)
    assert magnetic_gpu.execution_receipt()["counts"]["gpu_analytic_rectangle_source_calls"] > 0
    # Identity segmentation preserves CUDA reductions bitwise as well as CPU.
    raw = hp()
    raw_gpu = prepare(raw).sample(points())
    identity_gpu = prepare(replace(raw, current_segments=radial_recipe(2))).sample(points())
    for key in raw_gpu:
        np.testing.assert_array_equal(identity_gpu[key], raw_gpu[key])
