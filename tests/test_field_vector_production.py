"""Selected diagnostic vectors against production fields and analytic invariants.

These checks use actual stigmator, hexapole and finite-coil providers.  The gun
fixture uses a small exactly linear scalar grid through the real electric
capture/cache path; it is deliberately not a solved microscope electrostatic
field or a qualification of the displayed geometry.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.field_vector_scene import prepare_field_vector_scene
from temsim.optics.column import default_state
from temsim.optics.hexapole import HexapoleComponent
from temsim.optics.model import DeflectorPair, Stigmator
from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field


# Independent SI constants/formulas: no force or momentum helper from the
# display implementation is used to establish its numerical expectations.
E_CHARGE = 1.602176634e-19
E_MASS = 9.1093837015e-31
LIGHT_SPEED = 299792458.0


def momentum(energy_ev):
    joules = energy_ev * E_CHARGE
    rest_joules = E_MASS * LIGHT_SPEED**2
    return np.sqrt(joules * (joules + 2.0 * rest_joules)) / LIGHT_SPEED


def speed(energy_ev):
    gamma = 1.0 + energy_ev * E_CHARGE / (E_MASS * LIGHT_SPEED**2)
    return LIGHT_SPEED * np.sqrt(1.0 - gamma**-2)


def bare_state(**changes):
    values = dict(lenses=(), stigmators=(), corrector_elements=(), deflectors=(),
                  apertures=(), beam_voltage_kv=300., simulation_time_s=.237,
                  simulation_mode="analytical", sample=SimpleNamespace(z_mm=0.))
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture(scope="module")
def instrument_template():
    return default_state()


@pytest.fixture
def tiny_electric_grid(monkeypatch):
    """Actual ClosedGunField interpolation/cache with an unsolved linear grid."""
    from temsim.physics import closed_gun_field as module
    from temsim.physics.planar_gun_field import request_digest
    calls = []

    def build(request, directory):
        calls.append(request)
        radius = np.linspace(0., request["domain"]["outer_radius_m"], 3)
        z = np.linspace(request["domain"]["entrance_m"], request["domain"]["exit_m"], 5)
        gradient = request["extraction_v"] / 4.
        potential = np.broadcast_to(gradient * z, (len(radius), len(z))).copy()
        return module.ClosedGunField(request, radius, z, potential,
                                     {"request_sha256": request_digest(request)})

    monkeypatch.setattr(module, "_MEMORY_FIELDS", OrderedDict())
    monkeypatch.setattr(module, "_build_request_field", build)
    monkeypatch.setattr(module, "_trim_disk_cache", lambda *args: None)
    return calls


def test_actual_stigmator_two_channels_and_gaussian_width():
    stigmator = Stigmator("Diagnostic stigmator", "s", 25., length_mm=6.,
                         strength_x_percent=20., strength_y_percent=-10.)
    state = bare_state(stigmators=(stigmator,))
    captured = prepare_field_vector_scene(state)
    points = np.array(((2e-5, -3e-5, .025), (-1e-5, 2e-5, .028), (0., 0., .025)))
    sample = captured.sample(points, ("s",))
    envelope = np.exp(-.5 * ((points[:, 2] * 1000. - 25.) / (6. / 2.355))**2)
    a, b = 60. * envelope, -30. * envelope
    pq = momentum(300000.) / -E_CHARGE
    expected = pq * np.column_stack((a * points[:, 1] - b * points[:, 0],
                                     a * points[:, 0] + b * points[:, 1],
                                     np.zeros(len(points))))
    assert sample.magnetic_valid.all() and sample.force_valid.all()
    np.testing.assert_allclose(sample.b_t, expected, rtol=3e-15, atol=1e-20)
    np.testing.assert_array_equal(sample.b_t[-1], 0.)
    np.testing.assert_array_equal(sample.e_v_per_m, 0.)

    stigmator.length_mm = 3.
    shorter = prepare_field_vector_scene(state).sample(points, ("s",))
    np.testing.assert_allclose(shorter.b_t[0], sample.b_t[0], rtol=3e-15)
    ratio = np.exp(-.5 * (3. / (3. / 2.355))**2 + .5 * (3. / (6. / 2.355))**2)
    np.testing.assert_allclose(shorter.b_t[1], sample.b_t[1] * ratio, rtol=5e-15)
    np.testing.assert_array_equal(captured.sample(points, ("s",)).b_t, sample.b_t)


def test_stigmator_strength_voltage_and_position_edits_refresh_only_new_capture():
    stigmator = Stigmator("Stigmator", "s", 25., strength_x_percent=15.)
    state = bare_state(stigmators=(stigmator,))
    points = np.array(((2e-5, 1e-5, .025), (2e-5, 1e-5, .026)))
    old_scene = prepare_field_vector_scene(state)
    old = old_scene.sample(points, ("s",)).b_t
    stigmator.strength_x_percent *= 2.
    doubled = prepare_field_vector_scene(state).sample(points, ("s",)).b_t
    np.testing.assert_allclose(doubled, 2. * old, rtol=3e-15)
    state.beam_voltage_kv = 80.
    low_energy = prepare_field_vector_scene(state).sample(points, ("s",)).b_t
    np.testing.assert_allclose(low_energy, doubled * momentum(80000.) / momentum(300000.), rtol=3e-15)
    stigmator.z_mm += 1.
    moved = prepare_field_vector_scene(state).sample(points, ("s",)).b_t
    np.testing.assert_allclose(moved[1], low_energy[0], rtol=3e-15)
    assert not np.array_equal(moved[0], low_energy[0])
    np.testing.assert_array_equal(old_scene.sample(points, ("s",)).b_t, old)


def test_actual_hexapole_quadratic_scaling_and_threefold_orientation():
    hexapole = HexapoleComponent("Hexapole", "h", 25., 3e7, 4e7, 4., True, "white",
                                25., 8., 20., 4., 25., orientation_rad=.23)
    state = bare_state(corrector_elements=(hexapole,))
    points = np.array(((1e-5, 2e-5, .025), (2e-5, 4e-5, .025), (0., 0., .025)))
    scene = prepare_field_vector_scene(state)
    sample = scene.sample(points, ("h",))
    hn, hs = 3e7 * np.cos(3. * .23), 3e7 * np.sin(3. * .23)
    u, v = points[:, 0]**2 - points[:, 1]**2, 2. * points[:, 0] * points[:, 1]
    expected = momentum(300000.) / -E_CHARGE * np.column_stack((hn*v - hs*u, hn*u + hs*v, np.zeros(3)))
    np.testing.assert_allclose(sample.b_t, expected, rtol=5e-15, atol=1e-21)
    np.testing.assert_allclose(sample.b_t[1], 4. * sample.b_t[0], rtol=3e-15)
    np.testing.assert_array_equal(sample.b_t[2], 0.)
    hexapole.orientation_rad += np.pi / 3.
    rotated = prepare_field_vector_scene(state).sample(points, ("h",))
    # Rotation adds pi to the field phase; a small Cartesian component is a
    # cancellation of two larger terms, so use a vector-norm roundoff bound.
    assert np.linalg.norm(rotated.b_t + sample.b_t) < 5e-15 * np.linalg.norm(sample.b_t)
    np.testing.assert_array_equal(scene.sample(points, ("h",)).b_t, sample.b_t)


def test_finite_deflector_length_changes_peak_but_preserves_signed_kick_integral():
    component = DeflectorPair("Deflector", "d", 25., 40., 2., -3., thickness_mm=4.)
    state = bare_state(deflectors=(component,))
    first = prepare_field_vector_scene(state)
    center = np.array(((0., 0., .025),))
    first_peak = first.sample(center, ("d",)).b_t[0]
    pq = momentum(300000.) / -E_CHARGE
    np.testing.assert_allclose(first_peak, (pq * -.003/.004, -pq * .002/.004, 0.), rtol=4e-15)
    for width in (4., 8.):
        component.thickness_mm = width
        scene = prepare_field_vector_scene(state)
        z = np.linspace((25.-width/2.)*1e-3, (25.+width/2.)*1e-3, 129)
        points = np.column_stack((np.zeros((len(z), 2)), z))
        sample = scene.sample(points, ("d",))
        assert sample.magnetic_valid.all()
        integral = np.trapezoid(sample.b_t, z, axis=0)
        np.testing.assert_allclose((E_CHARGE * integral[1]/momentum(300000.),
                                   -E_CHARGE * integral[0]/momentum(300000.)), (.002, -.003), rtol=3e-14)
    np.testing.assert_allclose(scene.sample(center, ("d",)).b_t[0], first_peak / 2., rtol=3e-15)
    np.testing.assert_array_equal(first.sample(center, ("d",)).b_t[0], first_peak)


def test_actual_descan_host_has_one_selectable_physical_field():
    from temsim.optics.descan_deflector import create_descan_deflector
    from temsim.optics.image_diffraction_deflector import create_image_diffraction_deflector
    host, channel = create_image_diffraction_deflector(), create_descan_deflector()
    host.upper_x_mrad, host.upper_y_mrad = .4, -.2
    host.lower_x_mrad, host.lower_y_mrad = -.1, .3
    channel.kick_x_mrad, channel.kick_y_mrad = .6, -.4
    channel.scan_enabled = channel.wobble_enabled = False
    state = bare_state(deflectors=(host,), corrector_elements=(channel,))
    scene = prepare_field_vector_scene(state)
    assert len(scene.sources) == 1
    source, = scene.sources
    assert source.key == host.key
    assert set(source.component_keys) == {host.key, channel.key}
    points = np.array(((0., 0., host.upper_z_mm*1e-3), (0., 0., host.lower_z_mm*1e-3)))
    sample = scene.sample(points, (host.key, host.key))
    pq_over_length = momentum(300000.) / (-E_CHARGE * host.thickness_mm * 1e-3)
    expected = pq_over_length * np.array(((-.4e-3, -.7e-3, 0.), (.1e-3, -.2e-3, 0.)))
    np.testing.assert_allclose(sample.b_t, expected, rtol=4e-15, atol=1e-18)
    np.testing.assert_array_equal(scene.sample(points, (host.key,)).b_t, sample.b_t)
    np.testing.assert_allclose(sample.b_t, capture_instrument_magnetic_field(state).field_at_global_positions_t(points), rtol=3e-15)


def test_actual_stigmator_rigid_placement_rotates_vectors_and_translates_axis(instrument_template):
    native = instrument_template
    component = deepcopy(native.stigmators[0])
    component.strength_x_percent, component.strength_y_percent = 17., -9.
    state = bare_state(stigmators=(component,), _resolved_assembly=native._resolved_assembly)
    old_scene = prepare_field_vector_scene(state)
    local_points = np.array(((2e-5, -3e-5, component.z_mm*1e-3),
                             (-1e-5, 2e-5, (component.z_mm+1.)*1e-3)))
    old = old_scene.sample(local_points, (component.key,)).b_t
    part = native._resolved_assembly.part(component.key)
    angle = .008
    rotation = np.array(((np.cos(angle), 0., np.sin(angle)), (0., 1., 0.),
                         (-np.sin(angle), 0., np.cos(angle))))
    center = np.array((0., 0., part.center_z_mm*1e-3))
    moved_points = (local_points-center) @ rotation.T + center + (4e-5, 0., 0.)
    state._resolved_assembly = replace(native._resolved_assembly, parts=tuple(
        replace(row, data={**row.data, "rotation_y_mrad": 8., "offset_x_mm": .04})
        if row.key == component.key else row for row in native._resolved_assembly.parts))
    moved = prepare_field_vector_scene(state).sample(moved_points, (component.key,))
    assert moved.magnetic_valid.all()
    np.testing.assert_allclose(moved.b_t, old @ rotation.T, rtol=2e-10, atol=1e-18)
    assert np.linalg.norm(moved.b_t[:, 2]) > 0.
    np.testing.assert_array_equal(old_scene.sample(local_points, (component.key,)).b_t, old)


def test_default_instrument_vectors_equal_production_e_and_b(tiny_electric_grid):
    from temsim.physics.instrument_electric import capture_instrument_electric_field
    state = default_state()
    state.stigmators[0].strength_x_percent = 13.
    state.stigmators[0].strength_y_percent = -8.
    state.deflectors[0].upper_x_mrad = .3
    vector_scene = prepare_field_vector_scene(state)
    points = np.array([(2e-8, -3e-8, z*1e-3) for z in (
        state.lenses[0].z_mm, state.stigmators[0].z_mm, state.deflectors[0].upper_z_mm)])
    sample = vector_scene.sample(points)
    assert sample.magnetic_valid.all() and sample.electric_valid.all()
    expected_b = capture_instrument_magnetic_field(state).field_at_global_positions_t(points)
    expected_e = capture_instrument_electric_field(state).field_at_global_positions_v_per_m(points)
    np.testing.assert_allclose(sample.b_t, expected_b, rtol=3e-15, atol=1e-18)
    np.testing.assert_array_equal(sample.e_v_per_m, expected_e)
    np.testing.assert_allclose(sample.e_v_per_m, np.tile((0., 0., -1000.), (len(points), 1)), rtol=1e-14, atol=1e-11)
    assert len(tiny_electric_grid) == 1


def test_electrode_voltage_refreshes_joint_e_and_slice_does_not_resolve(tiny_electric_grid):
    state = default_state()
    scene = prepare_field_vector_scene(state)
    points = np.array(((0., 0., .7), (2e-7, -1e-7, .71)))
    original = scene.sample(points, ("gun_electric_field",))
    assert original.electric_valid.all()
    for energy in (80000., 300000.):
        np.testing.assert_array_equal(scene.sample(points, ("gun_electric_field",), kinetic_energy_ev=energy).e_v_per_m,
                                      original.e_v_per_m)
    cropped = prepare_field_vector_scene(state, z_limits_mm=(690., 720.))
    np.testing.assert_array_equal(cropped.sample(points, ("gun_electric_field",)).e_v_per_m, original.e_v_per_m)
    assert len(tiny_electric_grid) == 1
    state.electron_gun.extractor.voltage_kv *= 1.1
    refreshed = prepare_field_vector_scene(state).sample(points, ("gun_electric_field",))
    np.testing.assert_allclose(refreshed.e_v_per_m, original.e_v_per_m * 1.1, rtol=2e-14, atol=1e-11)
    np.testing.assert_array_equal(scene.sample(points, ("gun_electric_field",)).e_v_per_m, original.e_v_per_m)
    assert len(tiny_electric_grid) == 2


@pytest.mark.parametrize("previous_view", ["cropped", "stale"])
def test_provided_magnetic_view_cannot_omit_or_preserve_obsolete_production_fields(previous_view, tiny_electric_grid):
    from temsim.magnetic_field_scene import prepare_magnetic_scene
    state = default_state()
    lens = state.lenses[0]
    points = np.array(((1e-8, -2e-8, lens.z_mm*1e-3),))
    if previous_view == "cropped":
        previous = prepare_magnetic_scene(state, z_limits_mm=(1000., 1700.))
        assert lens.key not in previous.source_keys
    else:
        previous = prepare_magnetic_scene(state)
        old_value = previous.field_at_global_positions_t(points).copy()
        lens.percent *= .6
    scene = prepare_field_vector_scene(state, magnetic_scene=previous)
    sample = scene.sample(points)
    assert lens.key in {source.key for source in scene.sources}
    assert sample.magnetic_valid.all() and sample.electric_valid.all()
    expected = capture_instrument_magnetic_field(state).field_at_global_positions_t(points)
    np.testing.assert_allclose(sample.b_t, expected, rtol=3e-15, atol=1e-18)
    assert abs(sample.b_t[0, 2]) > 0.
    if previous_view == "stale":
        assert not np.array_equal(sample.b_t, old_value)
        np.testing.assert_array_equal(previous.field_at_global_positions_t(points), old_value)
    assert len(tiny_electric_grid) == 1


@pytest.mark.parametrize("energy_ev", [0., 80000., 300000.])
def test_force_uses_signed_electron_charge_and_relativistic_velocity(energy_ev):
    component = DeflectorPair("Deflector", "d", 25., 40., 2., 0., thickness_mm=4.)
    scene = prepare_field_vector_scene(bare_state(deflectors=(component,)))
    points = np.array(((0., 0., .025),))
    forward = scene.sample(points, ("d",), kinetic_energy_ev=energy_ev)
    reverse = scene.sample(points, ("d",), kinetic_energy_ev=energy_ev, direction=(0., 0., -2.))
    parallel = scene.sample(points, ("d",), kinetic_energy_ev=energy_ev, direction=(0., 1., 0.))
    expected = -E_CHARGE * np.cross((0., 0., speed(energy_ev)), forward.b_t)
    np.testing.assert_allclose(forward.force_n, expected, rtol=3e-15, atol=1e-30)
    np.testing.assert_allclose(reverse.force_n, -forward.force_n, rtol=3e-15, atol=1e-30)
    np.testing.assert_array_equal(parallel.force_n, 0.)
    np.testing.assert_array_equal(reverse.b_t, forward.b_t)
    if energy_ev:
        assert forward.force_n[0, 0] > 0.  # +Z electron, +By coil: positive X force.


def test_actual_wien_e_and_b_are_selected_once_with_independent_lorentz_sum(tiny_electric_grid):
    state = default_state()
    state.electron_gun.install_monochromator()
    wien = state.electron_gun.monochromator.wien
    points = np.array(((0., 0., wien.optical_reference_from_tip_mm*1e-3),))
    scene = prepare_field_vector_scene(state)
    source = next(source for source in scene.sources if wien.key in source.component_keys)
    assert source.category == "mixed"
    sample = scene.sample(points, (source.key, "gun_electric_field"), kinetic_energy_ev=4000.)
    assert sample.force_valid.all()
    expected_e = np.array(((wien.electric_field_v_per_m, 0., -1000.),))
    expected_b = np.array(((0., wien.magnetic_field_mt*1e-3, 0.),))
    expected_force = -E_CHARGE * (expected_e + np.cross((0., 0., speed(4000.)), expected_b))
    np.testing.assert_allclose(sample.e_v_per_m, expected_e, rtol=3e-14, atol=1e-11)
    np.testing.assert_array_equal(sample.b_t, expected_b)
    np.testing.assert_allclose(sample.force_n, expected_force, rtol=3e-14, atol=1e-25)
    # A velocity selector's transverse Lorentz contributions cancel at its
    # design speed; the independently supplied axial gun electric force stays.
    assert abs(sample.force_n[0, 0]) < 1e-24
    assert sample.force_n[0, 2] == pytest.approx(E_CHARGE * 1000., rel=3e-14)
    reverse = scene.sample(points, (source.key, "gun_electric_field"), kinetic_energy_ev=4000., direction=(0., 0., -1.))
    np.testing.assert_allclose((sample.force_n + reverse.force_n) / 2., -E_CHARGE * expected_e, rtol=3e-14, atol=1e-25)
