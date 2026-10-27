"""Explicit analytical axial policies keep unknown exteriors out of composition."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution


def _base(kind="round", *, posed=False, center=0., dynamic=False):
    registration = CoordinateRegistration()
    if posed:
        registration = CoordinateRegistration((.001, -.002, .1), tuple(map(tuple,
            Rotation.from_rotvec((.08, -.03, .13)).as_matrix())))
    if kind == "round":
        field = FrozenAnalyticField("round", ((.3, center, .006),), registration,
                                   ((center-.012)*1000, (center+.012)*1000), .003)
    else:
        polynomial = {"dipole": ((1, 0, -.3), (0, 1, .2)),
                      "quadrupole": ((2, 0, -1.), (0, 2, 1.)),
                      "hexapole": ((3, 0, -2./3), (1, 2, 2.))}[kind]
        field = FrozenPosedMultipole(kind, kind, registration, polynomial,
            "uniform" if kind == "dipole" else "gaussian", center, .006,
            ((center-.012)*1000, (center+.012)*1000), .003, 3e-22, dynamic=dynamic)
    return AnalyticMagneticPotential(field)


def _global(base, points):
    return np.asarray(points)@base.field.registration.rotation_array.T+base.field.registration.origin_array_m


def _scene():
    return capture_physical_scene(SimpleNamespace(lenses=[]))


@pytest.mark.parametrize("posed", [False, True])
def test_gaussian_extension_preserves_tail_instead_of_reusing_legacy_support_mask(posed):
    base = _base(posed=posed)
    domain = AnalyticPotentialDomain(base)
    local = np.array(((.0002, -.0003, -.014), (0., 0., .012), (.0001, .0002, .019)))
    points = _global(base, local)
    with pytest.raises(ValueError, match="outside"):
        base.sample(points)
    sample = domain.sample(points)
    np.testing.assert_allclose(sample["B_t"], base.field.field_at_global_positions_t(points), rtol=2e-14, atol=1e-17)
    assert np.linalg.norm(sample["B_t"][-1]) > 0
    assert domain.to_metadata()["axial_policy"] == "gaussian_all_z"
    assert domain.to_metadata()["known_zero_exterior"] is False
    # Curl of the continued potential agrees independently beyond old support.
    point, step = points[-1], 1e-8
    derivative = np.column_stack([(domain.sample(point+d)["A_t_m"]-
        domain.sample(point-d)["A_t_m"])/(2*step) for d in np.eye(3)*step])
    curl = (derivative[2, 1]-derivative[1, 2], derivative[0, 2]-derivative[2, 0],
            derivative[1, 0]-derivative[0, 1])
    np.testing.assert_allclose(curl, sample["B_t"][-1], rtol=2e-8, atol=1e-11)


@pytest.mark.parametrize("kind", ["dipole", "quadrupole", "hexapole"])
@pytest.mark.parametrize("posed", [False, True])
def test_compact_Az_has_explicit_model_zero_outside_slab_and_strict_radial_domain_inside(kind, posed):
    base = _base(kind, posed=posed)
    domain = AnalyticPotentialDomain(base)
    local = np.array(((.0001, -.0002, 0.), (.1, -.2, -.013), (-.2, .1, .015)))
    sample = domain.sample(_global(base, local))
    np.testing.assert_allclose(sample["B_t"][0], base.sample(_global(base, local[0]))["B_t"], atol=1e-17)
    np.testing.assert_array_equal(sample["B_t"][1:], 0.)
    np.testing.assert_array_equal(sample["A_t_m"][1:], 0.)
    for value in domain.sample_jet(_global(base, local[1:])).values():
        np.testing.assert_array_equal(value, 0.)
    with pytest.raises(ValueError, match="radial domain"):
        domain.sample(_global(base, [.0031, 0., 0.]))
    assert domain.to_metadata()["known_zero_exterior"] is True


@pytest.mark.parametrize("kind", ["dipole", "quadrupole", "hexapole"])
def test_compact_edge_values_match_native_but_two_sided_derivatives_are_rejected(kind):
    base = _base(kind)
    domain = AnalyticPotentialDomain(base)
    points = np.array(((.0001, -.0002, -.012), (.0001, -.0002, .012)))
    np.testing.assert_allclose(domain.sample(points)["B_t"], base.field.field_at_global_positions_t(points), atol=1e-17)
    with pytest.raises(ValueError, match="hard axial face"):
        domain.sample_jet(points)


@pytest.mark.parametrize("kind", ["round", "dipole", "hexapole"])
def test_domains_preserve_interior_jets_arbitrary_batch_shapes_and_empty_input(kind):
    base = _base(kind)
    domain = AnalyticPotentialDomain(base)
    points = np.array(((.0001, .0002, -.002), (.0003, .0001, .004)))
    for name, value in base.sample_jet(points).items():
        np.testing.assert_array_equal(domain.sample_jet(points)[name], value)
    for points in (points[0], points.reshape(1, 2, 3), np.empty((0, 3))):
        assert domain.sample(points)["B_t"].shape == points.shape
        assert domain.sample_jet(points)["hess_A_t_per_m"].shape == (*points.shape[:-1], 3, 3, 3)


@pytest.mark.parametrize("point", [[.0031, 0., 0.], [.0031, 0., 3.], [0., 0., np.nan],
                                   [0., 1.], [0j, 0j, 0j], [False, True, False]])
def test_gaussian_domain_never_removes_radial_or_coordinate_validation(point):
    with pytest.raises(ValueError):
        AnalyticPotentialDomain(_base()).sample(point)


def test_inactive_polynomial_is_not_evaluated_and_active_overflow_is_rejected():
    compact = AnalyticPotentialDomain(_base("hexapole"))
    np.testing.assert_array_equal(compact.sample((1e150, -1e150, 1e150))["B_t"], 0.)
    overflowing = AnalyticMagneticPotential(replace(_base().field,
        terms_t_m=((1e308, 0., .006), (1e308, 0., .006))))
    with pytest.raises(ValueError, match="finite numerical range"):
        AnalyticPotentialDomain(overflowing).sample((0., 0., 0.))


def test_snapshot_roundtrip_is_detached_and_policy_tampering_is_rejected():
    base = _base()
    domain = AnalyticPotentialDomain(base)
    assert domain.base is not base and domain.base.field is not base.field
    restored = AnalyticPotentialDomain.from_dict(json.loads(json.dumps(domain.to_dict())))
    assert restored.digest == domain.digest and deepcopy(domain) is domain
    np.testing.assert_array_equal(restored.sample((.0001, .0002, .02))["B_t"], domain.sample((.0001, .0002, .02))["B_t"])
    with pytest.raises(FrozenInstanceError):
        domain.base = _base("dipole")
    with pytest.raises(TypeError):
        domain.base.provenance["new"] = 1
    record = domain.to_dict()
    record["metadata"]["known_zero_exterior"] = True
    with pytest.raises(ValueError, match="identity mismatch"):
        AnalyticPotentialDomain.from_dict(record)


def test_fem_unknown_exterior_and_custom_subclasses_cannot_request_native_extension():
    fem = AxisymmetricMagneticPotential((0., .001), (-.001, .001), ((0., 0.), (.001, .001)))
    class Custom(AnalyticMagneticPotential):
        pass
    for base in (fem, Custom(_base().field), SimpleNamespace(sample=lambda p: {})):
        with pytest.raises(ValueError, match="exact native analytical"):
            AnalyticPotentialDomain(base)


@pytest.mark.parametrize("native_dynamic,provenance_dynamic", [(False, False), (True, False), (False, True)])
def test_static_metadata_preserves_frozen_dynamic_drive_state(native_dynamic, provenance_dynamic):
    base = _base("dipole", dynamic=native_dynamic)
    base = replace(base, provenance={"dynamic_drive_frozen": provenance_dynamic})
    metadata = AnalyticPotentialDomain(base).to_metadata()
    assert metadata["dynamic_drive_frozen"] is (native_dynamic or provenance_dynamic)
    assert metadata["static"] is not (native_dynamic or provenance_dynamic)


def test_disjoint_native_sources_can_sum_without_filling_unknown_FEM_exterior():
    a = AnalyticPotentialDomain(_base("dipole", center=-.02))
    b = AnalyticPotentialDomain(_base("quadrupole", center=.02))
    combined = ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("a", ("a",), a, "magnetic"),
        PotentialContribution("b", ("b",), b, "magnetic")))
    points = np.array(((.0001, -.0002, -.02), (.0001, -.0002, 0.), (.0001, -.0002, .02)))
    actual = combined.sample(points)
    for name in ("A_t_m", "B_t"):
        np.testing.assert_array_equal(actual[name], a.sample(points)[name]+b.sample(points)[name])
        np.testing.assert_array_equal(actual[name][1], 0.)
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(combined.to_dict())))
    assert restored.digest == combined.digest
    np.testing.assert_array_equal(restored.sample(points)["B_t"], actual["B_t"])
    fem = AxisymmetricMagneticPotential((0., .001), (-.001, .001), ((0., 0.), (.001, .001)))
    with_fem = replace(combined, contributions=(*combined.contributions,
        PotentialContribution("unknown", ("fem",), fem, "magnetic")))
    with pytest.raises(ValueError, match="outside"):
        with_fem.sample(points)


@pytest.mark.parametrize("axial_domain", ["native_support", "gaussian_all_z"])
def test_high_order_exact_provider_composes_with_explicit_domain_and_json_identity(axial_domain):
    provider = HighOrderRoundLensPotential(_base().field, radial_order=3, axial_domain=axial_domain)
    snapshot = ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("round", ("round",), provider, "magnetic"),))
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.numerical_digest == snapshot.numerical_digest
    points = np.array(((.0001, .0002, .001), (.0001, .0002, .02)))
    if axial_domain == "native_support":
        with pytest.raises(ValueError, match="outside"):
            restored.sample(points)
    else:
        np.testing.assert_array_equal(restored.sample(points)["B_t"], provider.sample(points)["B_t"])


@pytest.mark.parametrize("mode", ["nano_probe", "nano_probe_large_angle"])
def test_actual_C3_probe_modes_admit_explicit_column_corridor_without_changing_presets(mode):
    from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
    from temsim.operating_modes import apply_condenser_mode
    from temsim.optics.column import default_state
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from test_potential_capture import _all_supported_keys
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", "C3 + Probe Corrector", "No Energy Filter"))
    apply_condenser_mode(state, mode, column_name="C3 + Probe Corrector")
    lenses, auxiliary = _all_supported_keys(state)
    captured = capture_selected_column_potentials(state, round_lens_keys=lenses, auxiliary_keys=auxiliary)
    prior = captured.to_dict()
    extended = replace(captured, contributions=tuple(replace(row,
        potential=AnalyticPotentialDomain(row.potential)) for row in captured.contributions))
    z = np.linspace(min(row.potential.field.field_support_mm[0] for row in captured.contributions)*.001,
                    max(row.potential.field.field_support_mm[1] for row in captured.contributions)*.001, 37)
    points = np.column_stack((np.full_like(z, 2e-7), np.full_like(z, -3e-7), z))
    with pytest.raises(ValueError, match="outside"):
        captured.sample(points)
    actual = extended.sample(points)
    expected = sum(row.potential.field.field_at_global_positions_t(points) for row in captured.contributions)
    np.testing.assert_allclose(actual["B_t"], expected, rtol=2e-13, atol=1e-16)
    assert np.isfinite(actual["A_t_m"]).all() and len(extended.contributions) > 20
    # Same captured coefficients and hardware state; the explicit domain
    # identity changes, not optical settings or legacy field masking.
    assert captured.to_dict() == prior
    assert capture_selected_column_potentials(state, round_lens_keys=lenses, auxiliary_keys=auxiliary).digest == captured.digest
    assert extended.scene.digest == captured.scene.digest
    assert extended.physical_digest == captured.physical_digest
    assert extended.numerical_digest != captured.numerical_digest


def test_regular_FEM_representation_composes_without_gaining_an_exterior_domain():
    from temsim.physics.regular_axisymmetric_magnetic_potential import RegularAxisymmetricMagneticPotential
    radius, z = np.linspace(0., .004, 9), np.linspace(-.01, .01, 7)
    raw = AxisymmetricMagneticPotential(radius, z, np.broadcast_to(.35*radius[:, None], (9, 7)))
    regular = RegularAxisymmetricMagneticPotential.from_potential(raw, fit_radius_m=.004,
        valid_radius_m=.003, vacuum_domain_note="Manufactured source-free uniform-field cylinder")
    snapshot = ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("regular", ("regular",), regular, "magnetic"),))
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert restored.digest == snapshot.digest
    np.testing.assert_allclose(restored.sample((.001, .001, .003))["B_t"], (0., 0., .7), atol=1e-13)
    with pytest.raises(ValueError, match="outside"):
        restored.sample((0., 0., .011))
