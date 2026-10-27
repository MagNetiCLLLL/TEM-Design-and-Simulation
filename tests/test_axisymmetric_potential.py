"""Potential-preservation tests, not a qualified column or OEM lens model."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import numpy as np
import pytest

from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.lens_field_provider import CoordinateRegistration


def _uniform(bz=1.3, **kwargs):
    r, z = np.array((0., .001, .003)), np.array((-.004, .001, .006))
    return AxisymmetricMagneticPotential(r, z, np.broadcast_to(.5*bz*r[:, None], (3, 3)), **kwargs)


def test_uniform_field_has_exact_potential_curl_and_regular_axis():
    field = _uniform()
    points = np.array(((0., 0., -.004), (0., 0., 0.), (0., 0., .001),
                       (0., 0., .006), (.001, .002, -.002)))
    sample = field.sample(points)
    np.testing.assert_allclose(sample["A_t_m"], .65*np.column_stack((-points[:, 1], points[:, 0], np.zeros(5))), atol=1e-18)
    np.testing.assert_allclose(sample["B_t"], np.tile((0., 0., 1.3), (5, 1)), atol=1e-15)
    assert field.sample(points[0])["A_t_m"].shape == (3,)
    assert field.sample(points.reshape(1, 5, 3))["B_t"].shape == (1, 5, 3)
    assert field.sample(np.empty((0, 3)))["B_t"].shape == (0, 3)


def test_triangle_basis_matches_independent_barycentric_potential_and_curl():
    r, z = np.array((0., .002, .005)), np.array((-.003, .001, .007))
    a = np.array(((0., 0., 0.), (.003, .006, .002), (.004, -.001, .008)))
    field = AxisymmetricMagneticPotential(r, z, a)
    points, exact_a, exact_b = [], [], []
    for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
        for offsets in (((0, 0), (1, 0), (1, 1)), ((0, 0), (1, 1), (0, 1))):
            indices = [(i+di, j+dj) for di, dj in offsets]
            vertices = np.array([(r[ii], z[jj]) for ii, jj in indices])
            values = np.array([a[ii, jj] for ii, jj in indices])
            coefficients = np.linalg.solve(np.column_stack((np.ones(3), vertices)), values)
            radius, axial = np.array((.23, .31, .46))@vertices
            phi = np.array((.23, .31, .46))@values
            theta = .43
            points.append((radius*np.cos(theta), radius*np.sin(theta), axial))
            exact_a.append((-np.sin(theta)*phi, np.cos(theta)*phi, 0.))
            exact_b.append((-np.cos(theta)*coefficients[2], -np.sin(theta)*coefficients[2], coefficients[1]+phi/radius))
    points = np.array(points)
    sample = field.sample(points)
    np.testing.assert_allclose(sample["A_t_m"], exact_a, rtol=2e-14, atol=1e-17)
    np.testing.assert_allclose(sample["B_t"], exact_b, rtol=2e-14, atol=1e-14)
    derivative = np.empty((len(points), 3, 3))
    for axis in range(3):
        delta = np.eye(3)[axis]*1e-8
        derivative[:, :, axis] = (field.sample(points+delta)["A_t_m"]
                                  -field.sample(points-delta)["A_t_m"])/(2e-8)
    curl = np.column_stack((derivative[:, 2, 1]-derivative[:, 1, 2],
                            derivative[:, 0, 2]-derivative[:, 2, 0],
                            derivative[:, 1, 0]-derivative[:, 0, 1]))
    np.testing.assert_allclose(curl, sample["B_t"], rtol=2e-8, atol=2e-8)


def test_axis_vertices_use_declared_one_sided_regular_limit():
    field = AxisymmetricMagneticPotential((0., 1.), (0., 1., 2.), ((0., 0., 0.), (1., 2., 4.)))
    result = field.sample(((0., 0., 0.), (0., 0., .5), (0., 0., 1.), (0., 0., 2.)))
    np.testing.assert_array_equal(result["A_t_m"], np.zeros((4, 3)))
    np.testing.assert_array_equal(result["B_t"], ((0., 0., 4.), (0., 0., 4.), (0., 0., 8.), (0., 0., 8.)))
    assert "no global Hessian" in field.to_metadata()["derivative_capability"]
    assert field.to_metadata()["wave_admission"] is False


def test_potential_is_continuous_across_diagonal_but_derivatives_are_not_invented():
    field = AxisymmetricMagneticPotential((0., 1., 2.), (0., 1.), ((0., 0.), (1., 3.), (2., 2.)))
    epsilon = 1e-10
    sides = field.sample(((1.5, 0., .5-epsilon), (1.5, 0., .5+epsilon)))
    np.testing.assert_allclose(sides["A_t_m"][0], sides["A_t_m"][1], atol=1e-9)
    assert not np.allclose(sides["B_t"][0], sides["B_t"][1])


def test_rigid_pose_rotates_both_potential_and_field_without_recalibration():
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec((.31, -.24, .19)).as_matrix()
    registration = CoordinateRegistration((.003, -.002, .02), tuple(map(tuple, rotation)))
    original = _uniform(source_scale=-.7)
    posed = replace(original, registration=registration)
    local = np.array(((.0005, .001, -.002), (-.001, .0015, .004)))
    global_points = local@rotation.T+registration.origin_array_m
    for name, values in original.sample(local).items():
        np.testing.assert_allclose(posed.sample(global_points)[name], values@rotation.T, rtol=1e-13, atol=2e-17)
    assert posed.digest != original.digest


def test_registration_is_detached_and_overridden_registration_is_rejected():
    registration = CoordinateRegistration((.001, .002, .003))
    field = _uniform(registration=registration)
    assert field.registration is not registration
    before = field.sample((.001, .002, .003))
    object.__setattr__(registration, "origin_global_m", (1., 2., 3.))
    for name, value in before.items():
        np.testing.assert_array_equal(field.sample((.001, .002, .003))[name], value)
    class MutableRegistration(CoordinateRegistration):
        @property
        def rotation_array(self):
            return np.eye(3)*2
    with pytest.raises(ValueError, match="exact CoordinateRegistration"):
        _uniform(registration=MutableRegistration())


def test_numerically_unusable_grids_and_overflow_are_rejected():
    with pytest.raises(ValueError, match="Potential axes"):
        AxisymmetricMagneticPotential((0., 1e-320), (0., 1.), ((0., 0.), (1., 1.)))
    with pytest.raises(ValueError, match="Potential axes"):
        AxisymmetricMagneticPotential((0., 1.), (-1e308, 1e308), ((0., 0.), (1., 1.)))
    field = _uniform(source_scale=1.7e308)
    with pytest.raises(ValueError, match="finite numerical range"):
        field.sample((0., 0., 0.))


def test_overridden_registration_class_cannot_change_captured_transform(monkeypatch):
    field = _uniform()
    before = field.sample((.001, .001, .001))
    monkeypatch.setattr(CoordinateRegistration, "rotation_array", property(lambda self: np.eye(3)*2.))
    with pytest.raises(ValueError, match="Overridden coordinate registration"):
        _uniform()
    # Evaluation uses the detached tuple coefficients rather than a live
    # registration property whose implementation could be changed externally.
    for name, value in before.items():
        np.testing.assert_array_equal(field.sample((.001, .001, .001))[name], value)


@pytest.mark.parametrize("point", [(.003000000001, 0., 0.), (0., 0., -.004000000001),
                                    (0., 0., .006000000001), (np.nan, 0., 0.),
                                    (0., np.inf, 0.), (0., 0.), (0j, 0., 0.)])
def test_unknown_domain_and_invalid_queries_are_rejected_even_at_zero_scale(point):
    with pytest.raises(ValueError):
        _uniform(source_scale=0.).sample(point)


def test_exact_unposed_boundary_is_admitted_but_next_float_is_outside():
    field = _uniform()
    field.sample(((.003, 0., -.004), (0., .003, .006)))
    for point in ((np.nextafter(.003, np.inf), 0., 0.),
                  (0., 0., np.nextafter(.006, np.inf)),
                  (0., 0., np.nextafter(-.004, -np.inf))):
        with pytest.raises(ValueError, match="outside"):
            field.sample(point)


def test_snapshot_detaches_inputs_and_roundtrips_complete_immutable_identity():
    r, z = np.array((0., .001)), np.array((-.001, .001))
    a = np.array(((0., 0.), (.001, .002)))
    provenance = {"circuit": {"channels": ["a", "b"]}, "solver": "synthetic"}
    field = AxisymmetricMagneticPotential(r, z, a, provenance=provenance)
    before = field.to_dict()
    r[-1], z[-1], a[1, 1] = 100., 100., 100.
    provenance["circuit"]["channels"].append("c")
    assert field.to_dict() == before
    for array in (field.r_m, field.z_m, field.a_phi_tm):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        field.provenance["circuit"]["channels"] = ()
    with pytest.raises(FrozenInstanceError):
        field.source_scale = 2.
    clone = deepcopy(field)
    assert clone is field
    restored = AxisymmetricMagneticPotential.from_dict(json.loads(json.dumps(field.to_dict(), allow_nan=False)))
    assert restored.digest == field.digest
    assert restored.to_dict() == before
    payload = restored.to_dict()
    payload["metadata"]["provenance"]["circuit"]["channels"].clear()
    assert restored.to_dict() == before
    assert replace(field, source_scale=-1.).digest != field.digest


@pytest.mark.parametrize("change", ["coefficient", "gauge", "basis", "units", "digest", "extra"])
def test_restoration_rejects_changed_coefficients_or_claims(change):
    payload = _uniform().to_dict()
    if change == "coefficient":
        payload["coefficients"]["a_phi_tm"][1][0] *= 2.
    else:
        payload["metadata"][change] = "changed"
    with pytest.raises(ValueError, match="identity or representation"):
        AxisymmetricMagneticPotential.from_dict(payload)


@pytest.mark.parametrize("changes", [
    {"r_m": (.001, .002)}, {"r_m": (0., 0.)}, {"z_m": (1., 0.)},
    {"z_m": (0., np.inf)}, {"a_phi_tm": ((1e-30, 0.), (1., 2.))},
    {"a_phi_tm": ((0., 0.),)}, {"a_phi_tm": ((0., 0.), (1., np.nan))},
    {"source_scale": True}, {"source_scale": np.inf}, {"source_id": " "},
    {"provenance": []}, {"registration": {}},
])
def test_invalid_potential_snapshots_are_rejected(changes):
    values = dict(r_m=(0., .001), z_m=(-.001, .001), a_phi_tm=((0., 0.), (.001, .002)))
    with pytest.raises(ValueError):
        AxisymmetricMagneticPotential(**dict(values, **changes))


def test_existing_cached_fem_solution_can_be_captured_without_new_solve(monkeypatch):
    from test_magnetic_circuits import _state, _shared_rows
    from temsim.physics import axisymmetric_magnetostatics as fem
    from temsim.physics.lens_field_provider import lens_geometry_binding
    state = _state(_shared_rows())
    binding = lens_geometry_binding(state, "a", state.lenses[0])
    legacy, solution = fem.solve_geometry_field_details(binding, state.lens_field_map_descriptors["a"])
    components = tuple(array.copy() for array in legacy.components_t)
    def unexpected_solve(*args, **kwargs):
        raise AssertionError("Capturing an existing solution must not solve another field")
    monkeypatch.setattr(fem, "solve_geometry_field_details", unexpected_solve)
    captured = AxisymmetricMagneticPotential.from_solution(solution,
        source_id="synthetic-shared:a", provenance={"source_sha256": legacy.provenance.source_sha256})
    np.testing.assert_array_equal(captured.a_phi_tm, solution.a_phi_tm)
    assert not np.shares_memory(captured.a_phi_tm, solution.a_phi_tm)
    assert np.any(captured.sample(((.0003, .0004, -.005),))["B_t"])
    for original, unchanged in zip(components, legacy.components_t):
        np.testing.assert_array_equal(original, unchanged)


def test_manufactured_fem_potential_and_same_basis_curl_converge_independently():
    from temsim.physics.axisymmetric_magnetostatics import solve_axisymmetric, MU0
    # Prespecified gates for a smooth manufactured vacuum solution: P1
    # potential and its piecewise derivative converge separately. This is not
    # a tolerance against the legacy nodal-gradient/interpolated B evaluator.
    radius, length, k = .05, .05, 1e5
    random = np.random.default_rng(7185)
    radial = random.uniform(.08, .88, 128)*radius
    axial = random.uniform(-.85, .85, 128)*length
    points = np.column_stack((radial, np.zeros_like(radial), axial))
    exact_a = k*radial*(radius**2-radial**2)*(length**2-axial**2)
    exact_b = np.column_stack((2*k*radial*(radius**2-radial**2)*axial,
        np.zeros_like(radial), 2*k*(radius**2-2*radial**2)*(length**2-axial**2)))
    errors = []
    for n in (16, 32, 64):
        r, z = np.linspace(0., radius, n), np.linspace(-length, length, 2*n)
        current = lambda r, z: 2*k*r*((radius**2-r*r)+4*(length**2-z*z))/MU0
        solution = solve_axisymmetric(r, z, lambda r, z: np.ones_like(r), current)
        assert solution.relative_residual < 1e-7
        sample = AxisymmetricMagneticPotential.from_solution(solution).sample(points)
        errors.append((np.linalg.norm(sample["A_t_m"][:, 1]-exact_a)/np.linalg.norm(exact_a),
                       np.linalg.norm(sample["B_t"]-exact_b)/np.linalg.norm(exact_b)))
    errors = np.asarray(errors)
    assert np.all(errors[1:, 0] < .4*errors[:-1, 0])
    assert np.all(errors[1:, 1] < .75*errors[:-1, 1])
    assert errors[-1, 0] < .007
    assert errors[-1, 1] < .08
