"""Selectable vector diagnostics: synthetic fixtures test contracts, not hardware."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.field_vector_scene import prepare_field_vector_scene
from temsim.magnetic_field_scene import MagneticSceneField, _Source


class UniformField:
    def __init__(self, magnetic=(0., 0., 1.), electric=(0., 0., 0.), *, drives=()):
        self.magnetic = magnetic
        self.electric = electric
        self.drive_keys = drives

    def field_at_global_positions_t(self, points):
        return np.broadcast_to(self.magnetic, np.shape(points)).copy()

    def field_at_global_positions_v_per_m(self, points):
        return np.broadcast_to(self.electric, np.shape(points)).copy()


def source(key="a", magnetic=(0., 0., 1.), *, radius=.001, lower=.01, upper=.02,
           electric=(0., 0., 0.), category="lens", drives=(), zero=False):
    return _Source(key, UniformField(magnetic, electric, drives=drives),
                   np.array(((-radius, -radius, lower), (radius, radius, upper))),
                   radius, category=category, label=key, known_zero=zero)


def magnetic_scene(*sources):
    bounds = np.array(((-.002, -.002, 0.), (.002, .002, .04)))
    return MagneticSceneField(bounds, .002, tuple(s.key for s in sources), (), (), sources, bounds)


def state(**kwargs):
    return SimpleNamespace(lenses=(), stigmators=(), corrector_elements=(), deflectors=(),
                           _field_provider_diagnostics={}, **kwargs)


def fake_electric(monkeypatch, *, base=None, wien=None, material=False):
    provider = base or UniformField(electric=(12., 0., 0.))
    if material:
        provider.conductor_material_mask = lambda points: points[:, 0] > .0002
    captured = SimpleNamespace(base_field=provider,
                               provider=SimpleNamespace(base_field=provider, wien_field=wien),
                               bounds_m=np.array(((-.001, -.001, 0.), (.001, .001, .04))),
                               gun_snapshot=SimpleNamespace(bore_components=()), notes=())
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", lambda _: captured)
    return captured


def test_selection_sum_duplicate_zero_and_immutable_data():
    a, b = source("a"), source("b", (0., 2., 0.))
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(a, b))
    points = np.array(((0., 0., .015),))
    av = scene.sample(points, ("a",))
    bv = scene.sample(points, ("b",))
    both = scene.sample(points, ("a", "a", "b"))
    np.testing.assert_allclose(both.b_t, av.b_t+bv.b_t)
    np.testing.assert_array_equal(scene.sample(points, ()).b_t, 0.)
    assert both.magnetic_valid.all() and both.electric_valid.all()
    for array in (scene.bounds_m, scene.sources[0].bounds_m, both.b_t, both.points_m, both.force_valid):
        assert not array.flags.writeable


def test_equal_opposite_fields_cancel_and_disabled_does_not_make_unknown():
    disabled = SimpleNamespace(key="off", name="Off", enabled=False, z_mm=15., length_mm=2.)
    inputs = state()
    inputs.lenses = (disabled,)
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source("a"), source("b", (0., 0., -1.))))
    sample = scene.sample(((0., 0., .015),))
    np.testing.assert_array_equal(sample.b_t, 0.)
    assert sample.force_valid.all()
    assert next(item for item in scene.sources if item.key == "off").model == "disabled_zero"


def test_unknown_radial_overlap_is_never_partial_field_or_zero():
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(
        source("narrow", radius=.0001), source("broad", (0., 2., 0.), radius=.001)))
    points = ((.00005, 0., .015), (.0002, 0., .015))
    sample = scene.sample(points)
    assert sample.magnetic_valid.tolist() == [True, False]
    assert sample.electric_valid.tolist() == [True, True]
    assert np.isnan(sample.b_t[1]).all() and np.isnan(sample.force_n[1]).all()
    assert scene.sample(points, ("broad",)).magnetic_valid.all()


def test_map_outside_native_domain_is_unknown_even_when_another_source_contains_it():
    class NativeMap:
        map_type = "cartesian_xyz"
        axes_m = (np.array((-.0001, .0001)), np.array((-.0001, .0001)), np.array((.01, .02)))
        registration = SimpleNamespace(positions_global_to_local_m=lambda points: points)
    mapped = replace(source("map"), field_map=NativeMap())
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(mapped, source("broad")))
    sample = scene.sample(((.0002, 0., .015),))
    assert not sample.magnetic_valid[0] and np.isnan(sample.b_t).all()


def test_known_zero_multipole_does_not_truncate_active_broad_source():
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(
        source("off", radius=.0001, zero=True), source("on", radius=.001)))
    sample = scene.sample(((.0002, 0., .015),))
    assert sample.force_valid.all()
    np.testing.assert_array_equal(sample.b_t, ((0., 0., 1.),))


def test_shared_coil_stages_and_drives_appear_once():
    drives = ("image_diffraction_deflector", "descan_deflector")
    upper = source("image_diffraction_deflector:0", (1., 0., 0.), drives=drives, upper=.014)
    lower = source("image_diffraction_deflector:1", (2., 0., 0.), drives=drives, lower=.016)
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(upper, lower))
    assert len(scene.sources) == 1
    entry = scene.sources[0]
    assert entry.key == "image_diffraction_deflector"
    assert set(entry.component_keys) == set(drives)
    np.testing.assert_array_equal(scene.sample(((0., 0., .012), (0., 0., .018))).b_t,
                                  ((1., 0., 0.), (2., 0., 0.)))
    assert any("shared" in note or "share" in note for note in entry.notes)


def test_nonlinear_joint_members_are_one_inseparable_source():
    inputs = state()
    inputs._field_provider_diagnostics = {
        "a": {"joint_field_owner": "a", "channels": ("a", "b")},
        "b": {"joint_field_owner": "a", "channels": ("a", "b")},
    }
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source("a")))
    assert scene.sources[0].component_keys == ("a", "b")
    assert scene.sources[0].model == "joint_nonlinear"
    assert "joint" in scene.sources[0].label
    np.testing.assert_array_equal(scene.sample(((0., 0., .015),), ("a", "a")).b_t, ((0., 0., 1.),))


def test_electric_domain_and_conductor_interior_are_not_known_zero(monkeypatch):
    fake_electric(monkeypatch, material=True)
    scene = prepare_field_vector_scene(state(electron_gun=object()), magnetic_scene=magnetic_scene(source()))
    points = ((0., 0., .015), (.0005, 0., .015), (.001, .001, .015))
    sample = scene.sample(points, ("gun_electric_field",))
    assert sample.electric_valid.tolist() == [True, False, False]
    assert np.isnan(sample.e_v_per_m[1:]).all()
    assert sample.magnetic_valid.all()


def test_wien_e_and_b_are_one_group_and_e_not_counted_twice(monkeypatch):
    wien = source("wien", (0., .02, 0.), electric=(100., 0., 0.), category="crossed_field")
    fake_electric(monkeypatch, base=UniformField(electric=(10., 0., 0.)), wien=wien.provider)
    scene = prepare_field_vector_scene(state(electron_gun=object()), magnetic_scene=magnetic_scene(wien))
    points = ((0., 0., .015),)
    mixed = scene.sample(points)
    only_wien = scene.sample(points, ("wien",))
    np.testing.assert_allclose(mixed.e_v_per_m, ((110., 0., 0.),))
    np.testing.assert_allclose(only_wien.e_v_per_m, ((100., 0., 0.),))
    assert next(item for item in scene.sources if item.key == "wien").category == "mixed"


def test_lorentz_signed_charge_local_speed_and_direction(monkeypatch):
    fake_electric(monkeypatch, base=UniformField(electric=(10., 0., 0.)))
    scene = prepare_field_vector_scene(state(electron_gun=object()), magnetic_scene=magnetic_scene(source(magnetic=(0., 2e-5, 0.))))
    points = ((0., 0., .015),)
    energy = 100000.
    gamma = 1.+energy/510998.9499961642
    speed = 299792458.*np.sqrt(1.-gamma**-2)
    expected = -1.602176634e-19*np.array(((10.-speed*2e-5, 0., 0.),))
    forward = scene.sample(points, kinetic_energy_ev=energy)
    reverse = scene.sample(points, kinetic_energy_ev=energy, direction=(0., 0., -2.))
    np.testing.assert_allclose(forward.force_n, expected, rtol=2e-12)
    np.testing.assert_allclose((forward.force_n+reverse.force_n)/2., ((-1.602176634e-18, 0., 0.),), rtol=2e-13)
    np.testing.assert_array_equal(forward.b_t, reverse.b_t)
    parallel = scene.sample(points, ("a",), direction=(0., 1., 0.))
    np.testing.assert_array_equal(parallel.force_n, 0.)
    stopped = scene.sample(points, kinetic_energy_ev=0.)
    np.testing.assert_allclose(stopped.force_n, ((-1.602176634e-18, 0., 0.),))


def test_electric_failure_keeps_magnetic_selection_usable(monkeypatch):
    def unavailable(_):
        raise ValueError("missing gun boundary")
    monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", unavailable)
    scene = prepare_field_vector_scene(state(electron_gun=object()), magnetic_scene=magnetic_scene(source()))
    total = scene.sample(((0., 0., .015),))
    assert total.magnetic_valid.all() and not total.electric_valid.any()
    assert np.isnan(total.force_n).all()
    assert scene.sample(((0., 0., .015),), ("a",)).force_valid.all()


def test_magnetic_failure_keeps_electric_selection_usable(monkeypatch):
    fake_electric(monkeypatch)
    def unavailable(_):
        raise ValueError("captured magnetic cache missing")
    monkeypatch.setattr("temsim.field_vector_scene.prepare_magnetic_scene", unavailable)
    scene = prepare_field_vector_scene(state(electron_gun=object()))
    total = scene.sample(((0., 0., .015),))
    assert total.electric_valid.all() and not total.magnetic_valid.any()
    assert np.isnan(total.force_n).all()
    assert scene.sample(((0., 0., .015),), ("gun_electric_field",)).force_valid.all()


def test_unsupported_blanker_marks_downstream_e_unknown_not_b(monkeypatch):
    fake_electric(monkeypatch)
    pulser = SimpleNamespace(installed=True, blanked=True, z_mm=15., plate_length_mm=2.)
    scene = prepare_field_vector_scene(state(electron_gun=object(), nanopulser=pulser), magnetic_scene=magnetic_scene(source()))
    sample = scene.sample(((0., 0., .013), (0., 0., .016)))
    assert sample.magnetic_valid.tolist() == [True, True]
    assert sample.electric_valid.tolist() == [True, False]
    assert np.isnan(sample.force_n[1]).all()
    assert not next(item for item in scene.sources if item.key == "electrostatic_blanker").supported


@pytest.mark.parametrize("kwargs", [
    {"points_m": [[0., float('nan'), 0.]]}, {"points_m": [0., 0., 0.]},
    {"kinetic_energy_ev": -1.}, {"kinetic_energy_ev": float('inf')},
    {"direction": (0., 0., 0.)}, {"direction": (0., float('nan'), 1.)},
    {"source_keys": ("missing",)},
])
def test_rejects_invalid_sample_inputs(kwargs):
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(source()))
    values = {"points_m": ((0., 0., .015),), **kwargs}
    with pytest.raises(ValueError):
        scene.sample(**values)


def test_view_limits_do_not_change_captured_fields_and_outside_is_unknown():
    magnetic = magnetic_scene(source())
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic, z_limits_mm=(12., 18.))
    sample = scene.sample(((0., 0., .015), (0., 0., .019)))
    assert sample.magnetic_valid.tolist() == [True, False]
    assert np.isnan(sample.b_t[1]).all()
    np.testing.assert_array_equal(sample.b_t[:1], magnetic._sources[0].provider.field_at_global_positions_t(np.array(((0., 0., .015),))))


def test_unmodelled_enabled_component_is_not_silently_removed():
    inputs = state()
    inputs.deflectors = (SimpleNamespace(key="custom_deflector", name="Custom", enabled=True, z_mm=15.),)
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source()))
    total = scene.sample(((0., 0., .015),))
    assert not total.magnetic_valid.any() and np.isnan(total.b_t).all()
    entry = next(item for item in scene.sources if item.key == "custom_deflector")
    assert not entry.supported
    assert scene.sample(((0., 0., .015),), ("a",)).magnetic_valid.all()


def test_layout_reference_is_a_declared_zero_and_not_missing_magnet():
    inputs = state()
    inputs.deflectors = (SimpleNamespace(key="virtual", name="Reference", enabled=True, INTERACTION_KIND="none"),)
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source()))
    assert scene.sample(((0., 0., .015),)).force_valid.all()
    assert next(item for item in scene.sources if item.key == "virtual").model == "reference_zero"


def test_extreme_finite_direction_normalizes_without_overflow():
    scene = prepare_field_vector_scene(state(), magnetic_scene=magnetic_scene(source(magnetic=(0., 1., 0.))))
    first = scene.sample(((0., 0., .015),), direction=(0., 0., 1e300))
    second = scene.sample(((0., 0., .015),), direction=(0., 0., 1.))
    np.testing.assert_array_equal(first.force_n, second.force_n)
    with pytest.raises(ValueError, match="finite numerical momentum"):
        scene.sample(((0., 0., .015),), kinetic_energy_ev=1e300)


def test_electrostatic_boundary_membership_does_not_hide_missing_magnet(monkeypatch):
    base = UniformField(electric=(1., 0., 0.))
    base.request = {"rings": ({"key": "custom_magnet"},)}
    fake_electric(monkeypatch, base=base)
    inputs = state(electron_gun=object())
    inputs.stigmators = (SimpleNamespace(key="custom_magnet", name="Custom", enabled=True),)
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source()))
    sample = scene.sample(((0., 0., .015),))
    assert sample.electric_valid.all() and not sample.magnetic_valid.any()
    assert not next(entry for entry in scene.sources if entry.key == "custom_magnet").supported


def _observation_index(inputs, scene, rows):
    """Small captured mechanical fixture; no assembly files or field solves."""
    from temsim.field_display_regions import build_field_display_regions
    parts = []
    for key, changes in rows:
        row = dict(key=key, name=key, local_start_z_mm=14., local_center_z_mm=15.,
                   local_end_z_mm=16., length_mm=2., mechanical_outer_diameter_mm=4.,
                   mechanical_profile="magnetic_lens_assembly")
        row.update(changes)
        parts.append(SimpleNamespace(key=key, name=key, module_key="captured",
                                     center_z_mm=15., data=row))
    inputs._resolved_assembly = SimpleNamespace(parts=tuple(parts), modules=())
    return build_field_display_regions(inputs, scene.sources)


def test_failed_magnetic_capture_preserves_single_component_unknown_and_passive_zero(monkeypatch):
    def unavailable(_):
        raise ValueError("captured joint magnetic cache missing")
    monkeypatch.setattr("temsim.field_vector_scene.prepare_magnetic_scene", unavailable)
    inputs = state()
    inputs.lenses = (SimpleNamespace(key="lens", enabled=True),
                     SimpleNamespace(key="off", enabled=False, z_mm=15., length_mm=2.))
    inputs.stigmators = (SimpleNamespace(key="stigmator", enabled=True),)
    inputs.deflectors = (SimpleNamespace(key="image_diffraction_deflector", enabled=True),
                        SimpleNamespace(key="descan_deflector", enabled=True),
                        SimpleNamespace(key="reference", enabled=True, INTERACTION_KIND="none"))
    scene = prepare_field_vector_scene(inputs)
    index = _observation_index(inputs, scene, [
        ("lens", {}), ("off", {}), ("stigmator", {}),
        ("lens_coil", dict(parent_key="lens", mechanical_profile="magnetic_excitation_coil")),
        ("image_diffraction_deflector", {}),
        ("descan_deflector", {}),
        ("reference", dict(layout_role="virtual_reference")),
        ("passive", dict(parent_key="lens", mechanical_profile="aperture_plate", mechanical_only=True)),
    ])
    points = ((0., 0., .015),)
    for key in ("lens", "lens_coil", "stigmator", "image_diffraction_deflector", "descan_deflector"):
        keys = index.source_keys_for_component(key)
        assert keys, f"{key} must retain its failed field identity"
        sample = scene.sample(points, keys)
        assert not sample.magnetic_valid.any()
        assert np.isnan(sample.b_t).all() and np.isnan(sample.force_n).all()
        assert sample.electric_valid.all()
    assert index.source_keys_for_component("image_diffraction_deflector") == index.source_keys_for_component("descan_deflector")
    assert index.source_keys_for_component("reference") == ()
    assert index.source_keys_for_component("passive") == ()
    for key in ("off", "passive"):
        sample = scene.sample(points, index.source_keys_for_component(key))
        assert sample.force_valid.all()
        np.testing.assert_array_equal(sample.b_t, 0.)
    assert "captured joint magnetic cache missing" in " ".join(
        next(entry for entry in scene.sources if entry.key == "lens").notes)


@pytest.mark.parametrize("failed", [False, True])
def test_joint_electrode_identity_is_retained_when_capture_fails(monkeypatch, failed):
    gun = SimpleNamespace(**{name: SimpleNamespace(key=key) for name, key in (
        ("extractor", "extractor"), ("electrostatic_lens", "gun_lens"),
        ("accelerator", "accelerator"), ("emitter", "feg_tip"))})
    if failed:
        def unavailable(_):
            raise ValueError("missing gun boundary")
        monkeypatch.setattr("temsim.physics.instrument_electric.capture_instrument_electric_field", unavailable)
    else:
        captured = fake_electric(monkeypatch)
        captured.gun_snapshot = gun
    inputs = state(electron_gun=gun)
    scene = prepare_field_vector_scene(inputs, magnetic_scene=magnetic_scene(source()))
    index = _observation_index(inputs, scene, [
        *((key, dict(mechanical_profile="electrode")) for key in
          ("extractor", "gun_lens", "accelerator", "feg_tip")),
        ("passive", dict(parent_key="extractor", mechanical_profile="aperture_plate", mechanical_only=True)),
    ])
    entry, = (item for item in scene.sources if item.key == "gun_electric_field")
    assert set(entry.component_keys) == {"gun_electric_field", "extractor", "gun_lens", "accelerator", "feg_tip"}
    points = ((0., 0., .015),)
    for key in ("extractor", "gun_lens", "accelerator", "feg_tip"):
        keys = index.source_keys_for_component(key)
        assert keys == ("gun_electric_field",)
        sample = scene.sample(points, keys)
        assert sample.magnetic_valid.all()
        if failed:
            assert not sample.electric_valid.any()
            assert np.isnan(sample.e_v_per_m).all() and np.isnan(sample.force_n).all()
        else:
            assert sample.force_valid.all()
            np.testing.assert_array_equal(sample.e_v_per_m, ((12., 0., 0.),))
    assert index.source_keys_for_component("passive") == ()
    passive = scene.sample(points, ())
    assert passive.force_valid.all()
    np.testing.assert_array_equal(passive.e_v_per_m, 0.)
