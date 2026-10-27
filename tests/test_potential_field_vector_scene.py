"""Same-potential diagnostic fields and offscreen existing-page handoff."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e

from temsim.optics.column import default_state
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge
from temsim.physics.relativistic_lorentz import ELECTRON, momentum_from_kinetic_energy_ev, relativistic_gamma
from temsim.potential_field_vector_scene import prepare_potential_field_vector_scene


def _snapshot(state=None):
    state = SimpleNamespace(lenses=[]) if state is None else state
    r, z = np.array([0., 1e-4]), np.array([-.01, .01])
    potential = AxisymmetricMagneticPotential(r, z, np.broadcast_to(.125*r[:, None], (2, 2)))
    return ElectromagneticPotentialSnapshot(capture_physical_scene(state),
        (PotentialContribution("joint_coil", ("coil_a", "coil_b"), potential, "magnetic"),))


def test_selected_vectors_and_force_use_the_exact_potential_with_atomic_joint_ownership():
    snapshot = _snapshot()
    scene = prepare_potential_field_vector_scene(snapshot, ((-2e-4, -2e-4, -.02), (2e-4, 2e-4, .02)))
    points = np.array([[0., 0., 0.], [1e-5, -2e-5, .003]])
    direction = np.array([1., 0., 1.])/np.sqrt(2.)
    sample = scene.sample(points, ("joint_coil", "joint_coil"), kinetic_energy_ev=300000., direction=direction)
    authoritative = snapshot.sample(points)
    np.testing.assert_array_equal(sample.b_t, authoritative["B_t"])
    np.testing.assert_array_equal(sample.e_v_per_m, authoritative["E_v_per_m"])
    momentum = momentum_from_kinetic_energy_ev(300000., direction, ELECTRON)
    velocity = momentum/(float(relativistic_gamma(momentum, ELECTRON))*ELECTRON.rest_mass_kg)
    np.testing.assert_allclose(sample.force_n, -e*np.cross(velocity, authoritative["B_t"]), rtol=3e-15)
    assert len(scene.sources) == 1 and scene.sources[0].component_keys == ("coil_a", "coil_b")
    assert scene.snapshot is snapshot and np.all(sample.force_valid)
    assert not sample.b_t.flags.writeable


def test_fem_exterior_stays_unknown_and_display_crop_does_not_extend_the_field():
    scene = prepare_potential_field_vector_scene(_snapshot(), ((-2e-4, -2e-4, -.02), (2e-4, 2e-4, .02)))
    points = [[0., 0., 0.], [1.5e-4, 0., 0.], [0., 0., .015], [0., 0., .03]]
    sample = scene.sample(points)
    assert sample.magnetic_valid.tolist() == [True, False, False, False]
    assert np.isnan(sample.b_t[1:]).all() and np.isnan(sample.force_n[1:]).all()
    empty = scene.sample(points, ())
    assert empty.magnetic_valid.tolist() == [True, True, True, False]
    np.testing.assert_array_equal(empty.b_t[:3], 0.)
    assert any("unavailable" in note for note in sample.notes)
    with pytest.raises(ValueError, match="Unknown"):
        scene.sample(points, ("coil_a",))  # A logical owner is not another solved source.


def test_electric_magnetic_superposition_retains_each_domain_and_actual_force():
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    magnetic = _snapshot()
    z, radial = np.array([-.005, .005]), np.array([0., 1e-4])
    provider = PlanarGunField({}, radial, z, np.broadcast_to(7.+4e4*z, (2, 2)))
    electric = PotentialContribution("electrodes", ("electrodes",), ElectricPotentialSnapshot.from_provider(provider), "electric")
    snapshot = replace(magnetic, contributions=(*magnetic.contributions, electric))
    scene = prepare_potential_field_vector_scene(snapshot, ((-2e-4, -2e-4, -.02), (2e-4, 2e-4, .02)))
    points = np.array([[2e-5, 0., .002], [0., 0., .008]])
    result = scene.sample(points, direction=(1., 0., 0.))
    expected = snapshot.sample(points[:1])
    np.testing.assert_array_equal(result.b_t[0], expected["B_t"][0])
    np.testing.assert_array_equal(result.e_v_per_m[0], expected["E_v_per_m"][0])
    assert result.magnetic_valid.tolist() == [True, True]
    assert result.electric_valid.tolist() == result.force_valid.tolist() == [True, False]
    assert np.isnan(result.force_n[1]).all()
    magnetic_only = scene.sample(points, ("joint_coil",), direction=(1., 0., 0.))
    np.testing.assert_allclose(result.force_n[0]-magnetic_only.force_n[0], -e*expected["E_v_per_m"][0], atol=1e-28)


def test_explicit_gaussian_all_z_field_is_not_truncated_to_legacy_display_support():
    from temsim.physics.potential_capture import capture_round_lens_potential
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    state = default_state()
    row = capture_round_lens_potential(state, state.condenser_lens_2.key)
    potential = HighOrderRoundLensPotential(row.potential.field, radial_order=3, axial_domain="gaussian_all_z")
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(state),
        (PotentialContribution(row.source_id, row.owner_ids, potential, "magnetic"),))
    lower, upper = row.potential.field.native_support_mm
    z = upper*.001+1e-6
    point = np.array([[0., 0., z]])
    scene = prepare_potential_field_vector_scene(snapshot, ((-1e-5, -1e-5, lower*.001), (1e-5, 1e-5, z+1e-5)))
    result = scene.sample(point)
    assert result.magnetic_valid[0] and result.b_t[0, 2] != 0.
    np.testing.assert_array_equal(result.b_t, snapshot.sample(point)["B_t"])


def test_pure_gauge_does_not_change_displayed_fields_or_absolute_forces():
    snapshot = _snapshot()
    transformed = replace(snapshot, gauge=StaticPolynomialGauge(.1, (2e-7, -3e-7, 4e-7)))
    box = ((-1e-4, -1e-4, -.01), (1e-4, 1e-4, .01))
    points = np.array([[2e-5, 1e-5, .002], [0., 0., 0.]])
    original = prepare_potential_field_vector_scene(snapshot, box).sample(points)
    shifted = prepare_potential_field_vector_scene(transformed, box).sample(points)
    np.testing.assert_array_equal(original.b_t, shifted.b_t)
    np.testing.assert_array_equal(original.e_v_per_m, shifted.e_v_per_m)
    np.testing.assert_array_equal(original.force_n, shifted.force_n)


def test_existing_field_page_worker_accepts_only_matching_retained_potential_inputs(monkeypatch, qapp):
    from temsim.gui.field_vector_page import _PrepareWorker
    import temsim.field_vector_scene as legacy

    state = default_state()
    scene = prepare_potential_field_vector_scene(_snapshot(state), ((-1e-4, -1e-4, -.01), (1e-4, 1e-4, .01)))
    monkeypatch.setattr(legacy, "prepare_field_vector_scene", lambda *a, **k: pytest.fail("No legacy field recapture or solve"))
    rows = []
    worker = _PrepareWorker(1, state, scene, None)
    worker.signals.finished.connect(lambda *values: rows.append(values))
    worker.run()
    assert rows[0][2] is None and rows[0][1][0] is scene
    state.condenser_stigmator.strength_x_percent += 1.
    stale = _PrepareWorker(2, state, scene, None)
    stale.signals.finished.connect(lambda *values: rows.append(values))
    stale.run()
    assert rows[-1][1] is None and "inputs differ" in rows[-1][2]
    assert qapp is not None


def _native_scene(*, rotation=None, origin=(0., 0., 0.), terms=((1, 0, .2),),
                  support_mm=(-50., 50.), radius=.003, display=None):
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.lens_field_provider import CoordinateRegistration
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    registration = CoordinateRegistration(origin, np.eye(3) if rotation is None else rotation)
    native = FrozenPosedMultipole("coil", "coil", registration, terms, "uniform", 0., .01,
                                  support_mm, radius, 1e-22)
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        (PotentialContribution("coil", ("coil",), AnalyticPotentialDomain(AnalyticMagneticPotential(native)), "magnetic"),))
    return prepare_potential_field_vector_scene(snapshot,
        ((-.1, -.1, -.2), (.1, .1, .2)) if display is None else display)


def _section_points(section, first, second, *, angle=0., fixed_z=0.):
    from temsim.field_density_texture import from_longitudinal
    aa, bb = np.meshgrid(first, second, indexing="ij")
    points = np.zeros((aa.size, 3))
    points[:, (0, 1) if section == "XY" else (0, 2)] = np.c_[aa.ravel(), bb.ravel()]
    if section == "XY":
        points[:, 2] = fixed_z
    return from_longitudinal(points, angle) if section == "UZ" else points


def test_compact_potential_metadata_prevents_a_false_field_ramp_in_the_zero_exterior():
    from temsim.field_density_texture import SectionMagnitudeGrid
    from temsim.field_section_sampling import with_source_support
    scene = _native_scene()
    axes = (np.array([-.0001, .0001]), np.array([0., .1]))
    points = _section_points("UZ", *axes)
    sample = scene.sample(points)
    query = np.array([[0., 0., .075]])
    np.testing.assert_array_equal(scene.sample(query).b_t, 0.)
    # Without real finite-face metadata this same cell invents 0.05 T.
    grid = SectionMagnitudeGrid(points, sample.b_t, sample.magnetic_valid, "UZ",
                                with_source_support(axes, scene.sources))
    magnitude, supported, valid = grid.evaluate(query)
    assert not supported[0] and not valid[0] and np.isnan(magnitude[0])


def test_declared_constant_electric_continuation_keeps_its_mesh_transition_plane():
    from temsim.field_density_texture import SectionMagnitudeGrid
    from temsim.field_section_sampling import with_source_support
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    r, z = np.array([0., .001]), np.array([0., .01])
    native = ClosedGunField({}, r, z, np.broadcast_to(4e4*z, (2, 2)))
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        (PotentialContribution("gun", ("gun",), ElectricPotentialSnapshot.from_provider(native), "electric"),))
    scene = prepare_potential_field_vector_scene(snapshot, ((-.001, -.001, -.02), (.001, .001, .02)))
    axes = (np.array([-.0001, .0001]), np.array([-.01, .005]))
    points = _section_points("UZ", *axes)
    sample = scene.sample(points)
    query = np.array([[0., 0., -.0025]])
    np.testing.assert_array_equal(scene.sample(query).e_v_per_m, 0.)
    grid = SectionMagnitudeGrid(points, sample.e_v_per_m, sample.electric_valid, "UZ",
                                with_source_support(axes, scene.sources))
    magnitude, supported, valid = grid.evaluate(query)
    assert not supported[0] and not valid[0] and np.isnan(magnitude[0])


@pytest.mark.parametrize("section,angle", [("UZ", 0.), ("UZ", 37.5), ("XY", 0.)])
def test_posed_compact_potential_sampler_and_texture_preserve_real_zero_regions(section, angle):
    from temsim.field_density_texture import SectionMagnitudeGrid, section_grid_axes
    from temsim.field_section_sampling import longitudinal_field_grid, with_source_support
    theta = np.deg2rad(25.)
    rotation = np.array(((np.cos(theta), 0., np.sin(theta)), (0., 1., 0.),
                         (-np.sin(theta), 0., np.cos(theta))))
    scene = _native_scene(rotation=rotation, origin=(0., 0., .8), support_mm=(-5., 5.), radius=.03,
                          display=((-.03, -.03, .77), (.03, .03, .83)))
    first = np.linspace(-.019, .019, 19)
    second = first if section == "XY" else np.linspace(.781, .819, 19)
    points = _section_points(section, first, second, angle=angle, fixed_z=.8)
    if section == "UZ":
        points, axes = longitudinal_field_grid(points, scene.sources, angle_deg=angle)
    else:
        axes = with_source_support(section_grid_axes(points, section), scene.sources)
    sample = scene.sample(points)
    grid = SectionMagnitudeGrid(points, sample.b_t, sample.magnetic_valid, section, axes, angle_deg=angle)
    query = _section_points(section, np.linspace(first[0], first[-1], 61),
                            np.linspace(second[0], second[-1], 67), angle=angle, fixed_z=.8)
    actual = scene.sample(query)
    magnitude, supported, valid = grid.evaluate(query)
    assert np.any(valid & (magnitude == 0.)) and np.any(valid & (magnitude > .1))
    np.testing.assert_allclose(magnitude[valid], np.linalg.norm(actual.b_t[valid], axis=1), rtol=0., atol=1e-15)
    normal = rotation[:, 2]
    np.testing.assert_allclose(axes.support_planes,
        [(*normal, -.005+.8*normal[2]), (*normal, .005+.8*normal[2])], rtol=0., atol=1e-16)


def test_actual_near_axis_hexapole_is_not_missed_by_a_whole_column_sampling_grid():
    from temsim.field_section_sampling import longitudinal_field_grid
    scene = _native_scene(terms=((3, 0, 1e6), (1, 2, -3e6)))
    initial = _section_points("UZ", np.linspace(-.1, .1, 15), np.linspace(-.2, .2, 15))
    points, axes = longitudinal_field_grid(initial, scene.sources)
    assert len(points) <= 1600
    assert np.any((abs(axes[0]) > 0.) & (abs(axes[0]) < .003))
    sample = scene.sample(points)
    # The physical polynomial vanishes on axis, so centre-only sampling cannot
    # establish zero. Off-axis anchors recover its independently known curl.
    active = sample.magnetic_valid & (abs(points[:, 2]) < .05)
    np.testing.assert_allclose(sample.b_t[active, 1], -3e6*points[active, 0]**2, rtol=3e-15, atol=0.)
    assert np.max(np.linalg.norm(sample.b_t[sample.magnetic_valid], axis=1)) > .1
    np.testing.assert_array_equal(scene.sample([[.001, 0., 0.]]).b_t, [[0., -3., 0.]])


@pytest.mark.parametrize("high_order", [False, True])
def test_gaussian_all_z_sampling_hints_do_not_create_a_texture_cut_at_the_native_window(high_order):
    from temsim.field_density_texture import SectionMagnitudeGrid
    from temsim.field_section_sampling import longitudinal_field_grid
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
    native = FrozenAnalyticField("lens", ((.2, 0., .02),), CoordinateRegistration(), (-50., 50.), .003)
    potential = (HighOrderRoundLensPotential(native, radial_order=1, axial_domain="gaussian_all_z")
                 if high_order else AnalyticPotentialDomain(AnalyticMagneticPotential(native)))
    snapshot = ElectromagneticPotentialSnapshot(capture_physical_scene(SimpleNamespace(lenses=[])),
        (PotentialContribution("lens", ("lens",), potential, "magnetic"),))
    scene = prepare_potential_field_vector_scene(snapshot, ((-.01, -.01, -.1), (.01, .01, .1)))
    points = _section_points("UZ", np.linspace(-.0001, .0001, 9), np.linspace(.048, .052, 9))
    points, axes = longitudinal_field_grid(points, scene.sources)
    assert axes.support_planes.shape == (0, 4)
    sample = scene.sample(points)
    grid = SectionMagnitudeGrid(points, sample.b_t, sample.magnetic_valid, "UZ", axes)
    query = np.array([[0., 0., .05], [0., 0., .05025]])
    magnitude, supported, valid = grid.evaluate(query)
    assert supported.all() and valid.all() and np.all(magnitude > 0.)
    np.testing.assert_allclose(scene.sample(query).b_t[:, 2], .2*np.exp(-.5*(query[:, 2]/.02)**2), rtol=3e-15)
