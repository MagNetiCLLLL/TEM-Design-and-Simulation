"""Native analytical gauges, independent curls, derivatives and persistence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.posed_column_fields import FrozenPosedMultipole


KINDS = ("round", "stigmator", "quadrupole", "hexapole", "dipole")


def native(kind, *, posed=True, envelope="gaussian"):
    rotation = Rotation.from_rotvec((.017, -.023, .12)).as_matrix() if posed else np.eye(3)
    registration = CoordinateRegistration((2e-5, -3e-5, .4) if posed else (0., 0., 0.),
                                           tuple(map(tuple, rotation)))
    if kind == "round":
        return FrozenAnalyticField("round", ((.3, 0., .006), (-.08, .008, .004)),
                                   registration, (-25., 25.), .003)
    terms = {
        "stigmator": ((2, 0, -1.), (0, 2, 1.), (1, 1, -.7)),
        "quadrupole": ((2, 0, -1.), (0, 2, 1.)),
        "hexapole": ((3, 0, -2./3), (1, 2, 2.), (2, 1, -.7), (0, 3, .7/3)),
        "dipole": ((1, 0, -.3), (0, 1, .2)),
    }[kind]
    return FrozenPosedMultipole(kind, kind, registration, terms, envelope, 0., .006,
                                (-25., 25.), .003, 3e-22)


def global_points(field, local=None):
    local = np.array([[0., 0., 0.], [.2e-3, -.3e-3, .007], [-.4e-3, .1e-3, -.012]]) if local is None else np.asarray(local)
    return local@field.registration.rotation_array.T+field.registration.origin_array_m


def independent_local_field(kind, local, envelope):
    x, y, z = np.moveaxis(local, -1, 0)
    if kind == "round":
        b = np.zeros_like(z)
        db = np.zeros_like(z)
        for amplitude, centre, sigma in ((.3, 0., .006), (-.08, .008, .004)):
            value = amplitude*np.exp(-.5*((z-centre)/sigma)**2)
            b += value
            db += -value*(z-centre)/sigma**2
        return np.stack((-.5*x*db, -.5*y*db, b), axis=-1)
    f = np.exp(-.5*(z/.006)**2) if envelope == "gaussian" else np.ones_like(z)
    if kind == "stigmator":
        bx, by = 2*y-.7*x, 2*x+.7*y
    elif kind == "quadrupole":
        bx, by = 2*y, 2*x
    elif kind == "hexapole":
        bx, by = 4*x*y+.7*(y*y-x*x), 2*(x*x-y*y)+1.4*x*y
    else:
        bx, by = np.full_like(x, .2), np.full_like(x, .3)
    return np.stack((bx*f, by*f, np.zeros_like(z)), axis=-1)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("posed", [False, True])
@pytest.mark.parametrize("envelope", ["gaussian", "uniform"])
def test_curl_matches_native_particle_field_and_independent_polynomials(kind, posed, envelope):
    field = native(kind, posed=posed, envelope=envelope)
    potential = AnalyticMagneticPotential(field)
    points = global_points(field)
    sampled = potential.sample(points)
    np.testing.assert_allclose(sampled["B_t"], field.field_at_global_positions_t(points),
                               rtol=2e-13, atol=1e-17)
    local = field.registration.positions_global_to_local_m(points)
    expected = independent_local_field(kind, local, envelope)@field.registration.rotation_array.T
    np.testing.assert_allclose(sampled["B_t"], expected, rtol=2e-13, atol=1e-17)
    # A finite-difference curl tests the gauge independently of its analytic jet.
    point, step = points[1], 1e-7
    jacobian = np.column_stack([(potential.sample(point+delta)["A_t_m"]-
        potential.sample(point-delta)["A_t_m"])/(2*step) for delta in np.eye(3)*step])
    curl = np.array([jacobian[2, 1]-jacobian[1, 2], jacobian[0, 2]-jacobian[2, 0],
                     jacobian[1, 0]-jacobian[0, 1]])
    np.testing.assert_allclose(curl, sampled["B_t"][1], rtol=2e-7, atol=1e-12)


@pytest.mark.parametrize("kind", KINDS)
def test_translated_rotated_potential_transforms_as_a_vector(kind):
    original = native(kind, posed=False)
    placed = native(kind)
    local = global_points(original)
    points = global_points(placed, local)
    first = AnalyticMagneticPotential(original).sample(local)
    second = AnalyticMagneticPotential(placed).sample(points)
    for name in ("A_t_m", "B_t"):
        np.testing.assert_allclose(second[name], first[name]@placed.registration.rotation_array.T,
                                   rtol=2e-13, atol=1e-17)


@pytest.mark.parametrize("kind", KINDS)
def test_cartesian_jacobian_and_hessian_match_finite_differences(kind):
    field = native(kind)
    potential = AnalyticMagneticPotential(field)
    point = global_points(field, [.18e-3, -.21e-3, .005])
    actual = potential.sample_jet(point)
    step = 1e-7
    for axis, delta in enumerate(np.eye(3)*step):
        plus, minus = potential.sample_jet(point+delta), potential.sample_jet(point-delta)
        np.testing.assert_allclose(actual["jac_A_t"][:, axis],
            (plus["A_t_m"]-minus["A_t_m"])/(2*step), rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(actual["hess_A_t_per_m"][:, :, axis],
            (plus["jac_A_t"]-minus["jac_A_t"])/(2*step), rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(actual["hess_A_t_per_m"],
                               actual["hess_A_t_per_m"].swapaxes(-1, -2), atol=1e-14)


@pytest.mark.parametrize("kind", KINDS)
def test_scalar_batch_grid_and_empty_shapes(kind):
    field = native(kind)
    potential = AnalyticMagneticPotential(field)
    points = global_points(field)
    for positions in (points[0], points, points.reshape(1, 3, 3), np.empty((0, 3))):
        prefix = positions.shape[:-1]
        jet = potential.sample_jet(positions)
        assert jet["A_t_m"].shape == (*prefix, 3)
        assert jet["jac_A_t"].shape == (*prefix, 3, 3)
        assert jet["hess_A_t_per_m"].shape == (*prefix, 3, 3, 3)
        assert potential.sample(positions)["B_t"].shape == (*prefix, 3)


@pytest.mark.parametrize("kind", ["round", "dipole", "hexapole"])
@pytest.mark.parametrize("local", [[0., 0., -.025], [0., 0., .025], [0., 0., .026],
                                  [.004, 0., 0.], [0., 0., np.nan], [0., 1.]])
def test_strict_native_domain_and_invalid_positions_are_rejected(kind, local):
    potential = AnalyticMagneticPotential(native(kind, posed=False))
    with pytest.raises(ValueError):
        potential.sample(local)


@pytest.mark.parametrize("kind", KINDS)
def test_placed_domain_is_checked_in_local_coordinates(kind):
    field = native(kind)
    potential = AnalyticMagneticPotential(field)
    for local in ([.0031, 0., 0.], [0., 0., -.026], [0., 0., .026]):
        with pytest.raises(ValueError, match="outside"):
            potential.sample(global_points(field, local))
    assert np.isfinite(potential.sample(global_points(field, [0., 0., 0.]))["B_t"]).all()


@pytest.mark.parametrize("kind", KINDS)
def test_immutable_provenance_and_detached_field_roundtrip(kind):
    field = native(kind)
    provenance = {"source": {"host": "coil", "drives": ["x", "y"]}}
    potential = AnalyticMagneticPotential(field, provenance)
    original = potential.sample(global_points(field))
    fingerprint = potential.digest
    provenance["source"]["drives"].append("changed")
    assert potential.provenance["source"]["drives"] == ("x", "y")
    with pytest.raises(TypeError):
        potential.provenance["source"]["host"] = "changed"
    with pytest.raises(FrozenInstanceError):
        potential.field = field
    # The adapter owns a detached native record even if upstream code replaces
    # a frozen source's attributes through an explicit low-level escape hatch.
    object.__setattr__(field, "lens_key", "caller-replaced")
    assert potential.source_id == kind and potential.digest == fingerprint
    restored = AnalyticMagneticPotential.from_dict(json.loads(json.dumps(potential.to_dict())))
    assert restored.digest == fingerprint
    for key, expected in original.items():
        np.testing.assert_array_equal(restored.sample(global_points(potential.field))[key], expected)
    metadata = restored.to_metadata()
    assert metadata["derivative_order"] == 2 and metadata["wave_admission"] is False
    assert "no new Maxwell" in metadata["physical_model"]


@pytest.mark.parametrize("kind", ["round", "stigmator", "dipole"])
def test_identity_changes_with_coefficients_pose_and_provenance(kind):
    field = native(kind)
    original = AnalyticMagneticPotential(field)
    registration = replace(field.registration, origin_global_m=(.001, 0., .4))
    assert AnalyticMagneticPotential(replace(field, registration=registration)).digest != original.digest
    if kind == "round":
        changed = replace(field, terms_t_m=((.4, 0., .006),))
    else:
        changed = replace(field, polynomial_terms=((1, 0, .3),))
    assert AnalyticMagneticPotential(changed).digest != original.digest
    assert AnalyticMagneticPotential(field, {"scope": "new"}).digest != original.digest


@pytest.mark.parametrize("kind", ["round", "stigmator", "hexapole"])
def test_tampered_identity_coefficients_registration_and_metadata_fail(kind):
    record = AnalyticMagneticPotential(native(kind)).to_dict()
    changed = deepcopy(record)
    changed["field"]["radial_support_m"] *= 2
    with pytest.raises(ValueError, match="identity mismatch"):
        AnalyticMagneticPotential.from_dict(changed)
    changed = deepcopy(record)
    changed["field"]["registration"]["origin_global_m"] = (.001, 0., .4)
    with pytest.raises(ValueError, match="identity mismatch"):
        AnalyticMagneticPotential.from_dict(changed)
    changed = deepcopy(record)
    changed["metadata"]["gauge"] = "another gauge"
    with pytest.raises(ValueError, match="identity mismatch"):
        AnalyticMagneticPotential.from_dict(changed)


@pytest.mark.parametrize("key,value", [("metadata", None), ("metadata", []), ("field", None),
                                      ("field", []), ("digest", "bad")])
def test_malformed_records_fail_with_value_error(key, value):
    record = AnalyticMagneticPotential(native("round")).to_dict()
    record[key] = value
    with pytest.raises(ValueError):
        AnalyticMagneticPotential.from_dict(record)


def test_unknown_mutable_registration_and_overridden_field_methods_are_rejected():
    with pytest.raises(ValueError, match="known frozen native"):
        AnalyticMagneticPotential(SimpleNamespace())
    field = native("round")
    object.__setattr__(field, "registration", SimpleNamespace())
    with pytest.raises(ValueError, match="registration"):
        AnalyticMagneticPotential(field)
    field = native("dipole")
    field.__dict__["vector_potential_jet"] = MethodType(lambda *_: None, field)
    with pytest.raises(ValueError, match="Overridden"):
        AnalyticMagneticPotential(field)


@pytest.mark.parametrize("bad", [[], {}, None, "", "   ", " leading", "trailing "])
@pytest.mark.parametrize("name", ["lens_key", "component_key", "drive_keys"])
def test_source_component_and_drive_ids_cannot_hold_mutable_or_untrimmed_values(name, bad):
    field = native("dipole")
    object.__setattr__(field, name, (bad,) if name == "drive_keys" else bad)
    with pytest.raises(ValueError, match="identity|identities"):
        AnalyticMagneticPotential(field)


def test_round_source_id_and_registration_method_override_fail_at_capture(monkeypatch):
    field = native("round")
    object.__setattr__(field, "lens_key", ["mutable"])
    with pytest.raises(ValueError, match="identity"):
        AnalyticMagneticPotential(field)
    field = native("round")
    monkeypatch.setattr(CoordinateRegistration, "positions_global_to_local_m", lambda *_: np.zeros(3))
    with pytest.raises(ValueError, match="Overridden coordinate"):
        AnalyticMagneticPotential(field)


def test_malformed_basis_metadata_fails_with_value_error():
    record = AnalyticMagneticPotential(native("round")).to_dict()
    record["metadata"]["kind"] = []
    with pytest.raises(ValueError, match="basis"):
        AnalyticMagneticPotential.from_dict(record)


@pytest.mark.parametrize("name,value", [("event_z_mm", []), ("captured_time_s", []),
    ("event_dx_rad", [0.]), ("event_dy_rad", float("nan")), ("dynamic", "false")])
def test_optional_event_metadata_cannot_leave_mutable_or_ambiguous_state(name, value):
    field = native("dipole")
    object.__setattr__(field, name, value)
    with pytest.raises(ValueError, match="finite scalar|Boolean"):
        AnalyticMagneticPotential(field)


def test_overridden_native_polynomial_is_not_silently_replaced():
    field = native("hexapole")
    field.__dict__["_polynomial"] = MethodType(lambda *_: np.zeros(1), field)
    with pytest.raises(ValueError, match="Overridden"):
        AnalyticMagneticPotential(field)
