"""Static potential composition and gauge contracts, without propagation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import e, hbar

from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.planar_gun_field import PlanarGunField
from temsim.physics.potential_snapshot import (
    ElectromagneticPotentialSnapshot, PotentialContribution, StaticPolynomialGauge,
)


def _scene():
    return capture_physical_scene(SimpleNamespace(lenses=[
        SimpleNamespace(key=key, z_mm=float(index)) for index, key in enumerate(("a", "b", "gun"))]))


def _magnetic(bz=1.3, **kwargs):
    r, z = np.array((0., .002, .004)), np.array((-.003, .003))
    return AxisymmetricMagneticPotential(r, z, np.broadcast_to(.5*bz*r[:, None], (3, 2)), **kwargs)


def _electric(slope=4e4, offset=17.):
    r, z = np.array((0., .003)), np.array((0., .002))
    provider = PlanarGunField({}, r, z, np.broadcast_to(offset+slope*z, (2, 2)))
    return ElectricPotentialSnapshot.from_provider(provider)


def _combined():
    return ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("mag:a", ("a",), _magnetic(), "magnetic"),
        PotentialContribution("electric:gun", ("gun",), _electric(), "electric")))


def _points():
    return np.array(((.0002, -.0001, .0005), (0., 0., .001)))


def test_common_snapshot_preserves_analytic_uniform_electric_and_magnetic_values():
    snapshot, points = _combined(), _points()
    sample = snapshot.sample(points)
    np.testing.assert_allclose(sample["phi_v"], 17.+4e4*points[:, 2], atol=1e-14)
    np.testing.assert_allclose(sample["E_v_per_m"], np.tile((0., 0., -4e4), (2, 1)), atol=1e-10)
    np.testing.assert_allclose(sample["A_t_m"], .65*np.column_stack((-points[:, 1], points[:, 0], np.zeros(2))), atol=1e-18)
    np.testing.assert_allclose(sample["B_t"], np.tile((0., 0., 1.3), (2, 1)), atol=1e-15)
    metadata = snapshot.to_metadata()
    assert metadata["static"] is True and metadata["wave_admission"] is False
    electric, = [row for row in metadata["sources"] if row["kind"] == "electric"]
    assert electric["provider_metadata"]["potential_reference"] == "rise_relative_to_original_tip_v"
    assert electric["provider_metadata"]["ground_reference_offset_v"] is None
    for shape in ((3,), (1, 2, 3), (0, 3)):
        query = points[0] if shape == (3,) else points.reshape(shape) if shape != (0, 3) else np.empty(shape)
        result = snapshot.sample(query)
        assert result["phi_v"].shape == query.shape[:-1]
        assert result["B_t"].shape == query.shape


def test_linear_independent_channel_bases_sum_once_in_preserved_order():
    first = PotentialContribution("a", ("a",), _magnetic(), "magnetic")
    second = PotentialContribution("b", ("b",), _magnetic(bz=-.3), "magnetic")
    snapshot = ElectromagneticPotentialSnapshot(_scene(), (first, second))
    total = snapshot.sample(_points())
    for name in ("A_t_m", "B_t"):
        expected = first.potential.sample(_points())[name]+second.potential.sample(_points())[name]
        np.testing.assert_array_equal(total[name], expected)
    np.testing.assert_allclose(total["B_t"], np.tile((0., 0., 1.), (2, 1)), atol=1e-15)
    np.testing.assert_array_equal(snapshot.sample(_points(), ["b"])["B_t"], second.potential.sample(_points())["B_t"])


def test_two_declared_scalar_contributions_keep_their_references_without_ground_conversion():
    snapshot = ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("e:a", ("a",), _electric(4e4, 17.), "electric"),
        PotentialContribution("e:b", ("b",), _electric(-1e4, -3.), "electric")))
    sampled = snapshot.sample(_points())
    np.testing.assert_allclose(sampled["phi_v"], 14.+3e4*_points()[:, 2], atol=1e-14)
    np.testing.assert_allclose(sampled["E_v_per_m"], np.tile((0., 0., -3e4), (2, 1)), atol=1e-10)
    assert "no ground-reference conversion" in snapshot.to_metadata()["scalar_potential_reference"]


def test_native_analytical_adapter_composes_and_roundtrips_with_other_known_adapters():
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
    field = FrozenAnalyticField("b", ((.7, 0., .004),), CoordinateRegistration(), (-3., 3.), .004)
    analytic = AnalyticMagneticPotential(field, {"model": "native first-order field"})
    base = _combined()
    combined = replace(base, contributions=(*base.contributions,
        PotentialContribution("mag:b", ("b",), analytic, "magnetic")))
    expected, extra = base.sample(_points()), analytic.sample(_points())
    sampled = combined.sample(_points())
    for name in ("A_t_m", "B_t"):
        np.testing.assert_array_equal(sampled[name], expected[name]+extra[name])
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(combined.to_dict())))
    assert restored.digest == combined.digest
    for name, value in sampled.items():
        np.testing.assert_array_equal(restored.sample(_points())[name], value)


def test_electric_and_magnetic_effects_can_share_one_atomic_owner():
    snapshot = ElectromagneticPotentialSnapshot(_scene(), (
        PotentialContribution("a:electric", ("a",), _electric(), "electric"),
        PotentialContribution("a:magnetic", ("a",), _magnetic(), "magnetic")))
    assert np.any(snapshot.sample(_points())["E_v_per_m"])
    assert np.any(snapshot.sample(_points())["B_t"])


def test_total_joint_circuit_and_any_member_cannot_both_execute():
    joint = PotentialContribution("joint", ("b", "a"), _magnetic(), "magnetic")
    member = PotentialContribution("member", ("b",), _magnetic(source_scale=0.), "magnetic")
    assert joint.owner_ids == ("a", "b")
    with pytest.raises(ValueError, match="Duplicate magnetic potential ownership for b"):
        ElectromagneticPotentialSnapshot(_scene(), (joint, member))
    duplicate_id = replace(member, source_id="joint", owner_ids=("gun",))
    with pytest.raises(ValueError, match="Duplicate potential source_id"):
        ElectromagneticPotentialSnapshot(_scene(), (joint, duplicate_id))


@pytest.mark.parametrize("suffix", ["", ":0", ":1"])
def test_declared_shared_host_alias_cannot_be_reapplied_as_another_source(suffix):
    scene = capture_physical_scene(SimpleNamespace(deflectors=[
        SimpleNamespace(key="image_diffraction_deflector", enabled=True),
        SimpleNamespace(key="descan_deflector", enabled=True)]))
    host = PotentialContribution("host", ("image_diffraction_deflector"+suffix,), _magnetic(), "magnetic")
    alias = PotentialContribution("alias", ("descan_deflector"+suffix,), _magnetic(), "magnetic")
    with pytest.raises(ValueError, match="Duplicate magnetic potential ownership"):
        ElectromagneticPotentialSnapshot(scene, (host, alias))


def test_unknown_regions_remain_errors_and_selection_does_not_invent_zero_extension():
    snapshot = _combined()
    points = ((0., 0., -.001),)
    with pytest.raises(ValueError, match="outside"):
        snapshot.sample(points)
    selected = snapshot.sample(points, source_ids=["mag:a"])
    assert selected["B_t"][0, 2] == pytest.approx(1.3)
    np.testing.assert_array_equal(selected["E_v_per_m"], np.zeros((1, 3)))
    zero = PotentialContribution("zero", ("b",), _magnetic(source_scale=0.), "magnetic")
    only_zero = ElectromagneticPotentialSnapshot(_scene(), (zero,))
    with pytest.raises(ValueError, match="outside"):
        only_zero.sample(((.0041, 0., 0.),))


@pytest.mark.parametrize("selected", [["missing"], ["mag:a", "mag:a"], "mag:a", [True]])
def test_invalid_source_selection_is_rejected(selected):
    with pytest.raises(ValueError):
        _combined().sample(_points(), selected)


def test_static_gauge_changes_only_potential_once_and_has_correct_electron_phase():
    scale = hbar/e
    gradient = scale*np.array((100., -200., 0.))
    hessian = scale*np.diag((2e4, 0., -1e4))
    gauge = StaticPolynomialGauge(np.pi*scale, gradient, hessian)
    original = _combined()
    changed = replace(original, gauge=gauge)
    points = _points()
    first, second = original.sample(points), changed.sample(points)
    for name in ("phi_v", "E_v_per_m", "B_t"):
        np.testing.assert_array_equal(first[name], second[name])
    np.testing.assert_allclose(second["A_t_m"], first["A_t_m"]+gradient+points@hessian, atol=1e-20)
    # Analytic phase for these chosen coefficients checks the electron sign
    # and the constant source phase, without fitting away any wave phase.
    expected_angle = -(np.pi+100*points[:, 0]-200*points[:, 1]
                       +1e4*points[:, 0]**2-.5e4*points[:, 2]**2)
    np.testing.assert_allclose(gauge.electron_phase_factor(points), np.exp(1j*expected_angle), atol=8e-16)
    np.testing.assert_allclose(abs(gauge.electron_phase_factor(points)), 1., atol=2e-16)
    assert changed.physical_digest == original.physical_digest
    assert changed.numerical_digest != original.numerical_digest
    # Per-source readout excludes the final gauge. Otherwise summing displayed
    # components would incorrectly apply this gauge once for every source.
    for source in original.contributions:
        for key, value in original.sample_source(source.source_id, points).items():
            np.testing.assert_array_equal(changed.sample_source(source.source_id, points)[key], value)
    empty = changed.sample(points, source_ids=[])
    np.testing.assert_array_equal(empty["A_t_m"], gauge.vector_potential_shift_t_m(points))
    assert not np.any(empty["B_t"])


def test_polynomial_gauge_shift_has_zero_curl_by_independent_finite_difference():
    gauge = StaticPolynomialGauge(.2, (1., -2., 3.), ((2., 4., 6.), (4., -1., .5), (6., .5, 3.)))
    points = _points()
    derivatives = np.empty((len(points), 3, 3))
    for axis in range(3):
        delta = np.eye(3)[axis]*1e-5
        plus = gauge.vector_potential_shift_t_m(points+delta)
        minus = gauge.vector_potential_shift_t_m(points-delta)
        derivatives[:, :, axis] = (plus-minus)/(2e-5)
        phi_derivative = (gauge.chi_t_m2(points+delta)-gauge.chi_t_m2(points-delta))/(2e-5)
        np.testing.assert_allclose(phi_derivative, gauge.vector_potential_shift_t_m(points)[:, axis], atol=3e-12)
    np.testing.assert_allclose(derivatives, derivatives.transpose(0, 2, 1), atol=3e-11)


def test_immutable_snapshot_roundtrip_and_field_identity_are_detached():
    source = _combined()
    contributions = list(source.contributions)
    gradient = np.array((1e-14, 0., 0.))
    gauge = StaticPolynomialGauge(gradient_t_m=gradient)
    snapshot = ElectromagneticPotentialSnapshot(source.scene, contributions, gauge)
    before = snapshot.to_dict()
    gradient[:] = 100.
    contributions.clear()
    assert snapshot.to_dict() == before
    with pytest.raises(ValueError):
        snapshot.gauge.gradient_t_m.setflags(write=True)
    with pytest.raises(ValueError):
        snapshot.gauge.hessian_t.setflags(write=True)
    with pytest.raises(TypeError):
        snapshot.scene.payload["components"]["a"]["inputs"] = {}
    with pytest.raises(FrozenInstanceError):
        snapshot.contributions = ()
    assert deepcopy(snapshot) is snapshot
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(before, allow_nan=False)))
    assert restored.to_dict() == before and restored.digest == snapshot.digest
    for name, expected in snapshot.sample(_points()).items():
        np.testing.assert_array_equal(restored.sample(_points())[name], expected)
    detached = restored.to_metadata()
    detached["gauge"]["gradient_t_m"][0] = 100.
    assert restored.to_dict() == before
    stronger = replace(source.contributions[0], potential=_magnetic(bz=1.7))
    new_field = replace(snapshot, contributions=(stronger, source.contributions[1]))
    # The explicitly limited physical digest cannot establish field equality.
    assert new_field.physical_digest == snapshot.physical_digest
    assert new_field.numerical_digest != snapshot.numerical_digest
    assert "not field equivalence" in snapshot.to_metadata()["physical_digest_scope"]


@pytest.mark.parametrize("part", ["physical_digest", "numerical_digest", "gauge", "owner"])
def test_restoration_rejects_changed_identity_gauge_or_ownership(part):
    record = _combined().to_dict()
    if part == "gauge":
        record["gauge"]["constant_t_m2"] = 1.
    elif part == "owner":
        record["contributions"][0]["owner_ids"] = ["other"]
    else:
        record[part] = "changed"
    with pytest.raises(ValueError, match="identity mismatch"):
        ElectromagneticPotentialSnapshot.from_dict(record)


@pytest.mark.parametrize("changes", [
    {"kind": "mixed"}, {"kind": "electric"}, {"source_id": ""},
    {"owner_ids": ()}, {"owner_ids": ("a", "a")}, {"owner_ids": "a"},
    {"potential": SimpleNamespace(sample=lambda p: {})},
])
def test_unsupported_or_ambiguous_contributions_are_rejected(changes):
    values = dict(source_id="a", owner_ids=("a",), potential=_magnetic(), kind="magnetic")
    with pytest.raises(ValueError):
        PotentialContribution(**dict(values, **changes))


@pytest.mark.parametrize("changes", [
    {"constant_t_m2": True}, {"constant_t_m2": np.inf},
    {"gradient_t_m": (0., np.nan, 0.)}, {"gradient_t_m": (0., 0.)},
    {"hessian_t": ((1., 0., 0.), (1e-15, 0., 0.), (0., 0., 0.))},
])
def test_invalid_gauges_do_not_silently_change_the_field(changes):
    with pytest.raises(ValueError):
        StaticPolynomialGauge(**changes)
