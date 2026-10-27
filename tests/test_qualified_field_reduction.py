"""Independent exact-quadratic limits; not qualification of a whole column."""
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c, e, hbar, m_e

from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge
from temsim.physics.qualified_field_reduction import qualify_quadratic_interval
from temsim.physics.interaction_registry import InteractionRegistry, InteractionRecord
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave import propagate_static_magnetic_wave


ENERGY = 300000.
P = np.sqrt(ENERGY*e*(ENERGY*e+2*m_e*c*c))/c
WAVELENGTH = 2*np.pi*hbar/P


def snapshot(*sources, gauge=None):
    scene = capture_physical_scene(SimpleNamespace(lenses=[]))
    return ElectromagneticPotentialSnapshot(scene, tuple(PotentialContribution(
        f"source:{i}", (f"coil:{i}",), source, "magnetic") for i, source in enumerate(sources)),
        StaticPolynomialGauge() if gauge is None else gauge)


def uniform(b=.125):
    r, z = np.array([0., 1e-4]), np.array([-.1, .1])
    return AxisymmetricMagneticPotential(r, z, np.broadcast_to(.5*b*r[:, None], (2, 2)))


def multipole(terms=((1, 0, -1e-4),), **changes):
    field = FrozenPosedMultipole("coil", "coil", CoordinateRegistration(), terms,
        "uniform", 0., .01, (-10., 10.), 1e-4, P)
    return AnalyticPotentialDomain(AnalyticMagneticPotential(replace(field, **changes)))


def qualify(source, **changes):
    values = dict(interval_m=(0., 1e-6), position_bounds_m=((-1e-5, 1e-5), (-1e-5, 1e-5)),
                  energy_range_ev=(290000., 310000.), maximum_canonical_slope=.005)
    values.update(changes)
    return qualify_quadratic_interval(source, **values)


def test_uniform_axial_map_matches_independent_helical_paraxial_solution():
    source = snapshot(uniform())
    reduction = qualify(source, interval_m=(0., 1e-5), maximum_canonical_slope=.001)
    positions = np.array([[1e-7, -2e-7], [-3e-7, 1e-7]])
    slopes = np.array([[2e-4, -1e-4], [-1e-4, 3e-4]])
    omega = e*.125/P
    angle = omega*1e-5
    sine, cosine = np.sin(angle), np.cos(angle)
    expected_slopes = slopes@np.array([[cosine, sine], [-sine, cosine]])
    displacement = slopes@np.array([[sine, 1-cosine], [-(1-cosine), sine]])/omega
    output, final_slopes, _ = reduction.propagate_rays(source, positions, slopes, energy_ev=ENERGY)
    np.testing.assert_allclose(output, positions+displacement, rtol=2e-12, atol=2e-19)
    np.testing.assert_allclose(final_slopes, expected_slopes, rtol=2e-13, atol=1e-18)


def test_uniform_force_retains_affine_action_and_energy_specific_response():
    source = snapshot(multipole())
    reduction = qualify(source)
    position, slopes = np.array([[1e-7, -2e-7]]), np.array([[2e-4, -1e-4]])
    for energy in (290000., 300000., 310000.):
        momentum = np.sqrt(energy*e*(energy*e+2*m_e*c*c))/c
        force = e*1e-4/momentum
        actual, angles, _ = reduction.propagate_rays(source, position, slopes, energy_ev=energy)
        np.testing.assert_allclose(actual, position+slopes*1e-6+[[force*1e-12/2, 0]], atol=1e-21)
        np.testing.assert_allclose(angles, slopes+[[force*1e-6, 0]], atol=1e-18)
        path = reduction.canonical_path(source, energy_ev=energy)
        assert path.action_m == pytest.approx(force**2*1e-18/12, rel=2e-13)
        assert np.isfinite(path.reference_phase_rad)


def test_wave_complex_phase_matches_wilson_reference_under_grid_refinement():
    source = snapshot(gauge=StaticPolynomialGauge(0., (0., 0., 1e-10)))
    reduction = qualify(source)
    errors = []
    for count in (33, 65, 129):
        axis = np.linspace(-32e-9, 32e-9, count)
        x, y = np.meshgrid(axis, axis)
        amplitude = np.exp(-(x*x+y*y)/(2*(4e-9)**2)+.37j)
        amplitude *= np.sqrt(.73/np.sum(abs(amplitude)**2))
        wave = PlaneWave(amplitude, np.eye(2)*(axis[1]-axis[0]), np.zeros(2))
        reduced, record = reduction.propagate_wave(source, wave, energy_ev=ENERGY)
        finite = propagate_static_magnetic_wave(source, amplitude, axis, axis, (0., 1e-6),
            energy_ev=ENERGY, kinetic_order=2)
        assert reduced.amplitude.shape == amplitude.shape
        np.testing.assert_allclose(reduced.basis_m, wave.basis_m, atol=1e-23)
        errors.append(np.linalg.norm(reduced.full_amplitude(WAVELENGTH)-finite.amplitude))
        assert record["normalization_applied"] is False
        assert record["affine_action_m"] == pytest.approx(-e/P*1e-16)
        assert reduced.probability == pytest.approx(.73, abs=3e-12)
    assert errors[1] < .3*errors[0] and errors[2] < .3*errors[1]
    assert errors[2] < 2e-4


def test_qualification_is_immutable_and_rejects_changed_sources_energy_and_support():
    source = snapshot(uniform())
    reduction = qualify(source, maximum_canonical_slope=.001)
    with pytest.raises(FrozenInstanceError):
        reduction.interval_m = (0., 3e-6)
    for array in (reduction.position_bounds_m, reduction.constant_a, reduction.linear_a):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="snapshot changed"):
        reduction.canonical_path(snapshot(uniform(.25)), energy_ev=ENERGY)
    with pytest.raises(ValueError, match="energy.*range"):
        reduction.canonical_path(source, energy_ev=320000.)
    with pytest.raises(ValueError, match="qualified.*support"):
        reduction.propagate_rays(source, [[0., 0.]], [[.002, 0.]], energy_ev=ENERGY)
    with pytest.raises(ValueError, match="qualified.*support"):
        reduction.propagate_rays(source, [[2e-5, 0.]], [[0., 0.]], energy_ev=ENERGY)
    assert reduction.metadata()["production_admission"] is False


@pytest.mark.parametrize("source", [
    snapshot(multipole(((3, 0, 1e6),))),
    snapshot(multipole(envelope_kind="gaussian")),
    snapshot(multipole(dynamic=True, captured_time_s=0.)),
    snapshot(replace(uniform(), a_phi_tm=np.array([[0., 0.], [1e-5, 2e-5]]))),
    snapshot(gauge=StaticPolynomialGauge(hessian_t=((0., 0., .1), (0., 0., 0.), (.1, 0., 0.)))),
])
def test_nonquadratic_dynamic_or_unsupported_gauge_never_becomes_qualified(source):
    with pytest.raises(ValueError):
        qualify(source)


def test_aperture_boundaries_and_native_field_faces_cannot_be_crossed():
    source = snapshot(multipole())
    for plane in (0., .5e-6, 1e-6):
        with pytest.raises(ValueError, match="boundary action"):
            qualify(source, boundary_planes_m=(plane,))
    reduction = qualify(source)
    with pytest.raises(ValueError, match="boundary action"):
        reduction.canonical_path(source, energy_ev=ENERGY, boundary_planes_m=(.5e-6,))
    with pytest.raises(ValueError, match="smooth field interval"):
        qualify(source, interval_m=(-.02, .02))


def test_reduced_interval_replaces_all_sources_and_cannot_double_count():
    source = snapshot(uniform(), multipole())
    reduction = qualify(source)
    native = reduction.native_interactions()
    observation = InteractionRecord("observe", "screen", "observation", "observation", (2., 2.))
    registry = reduction.replace_interactions(InteractionRegistry((*native, observation)))
    assert len(registry.records) == 2 and observation in registry.records
    reduced = next(row for row in registry.records if row.representation == "reduced")
    assert set(reduced.covered_effects) == {effect for row in native for effect in row.covered_effects}
    with pytest.raises(ValueError, match="executed twice"):
        InteractionRegistry((*registry.records, native[0]))
    with pytest.raises(ValueError, match="every exact native"):
        reduction.replace_interactions(InteractionRegistry(native[:1]))
    for representation in ("geometry_boundary", "material", "detector", "field"):
        extra = InteractionRecord("extra", "other", "extra", representation,
            (.0005, .0005), ("another-effect",), "another-source")
        with pytest.raises(ValueError, match="unreplaced field or boundary"):
            reduction.replace_interactions(InteractionRegistry((*native, extra)))
