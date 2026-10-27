"""Representation, curl, identity and strict-domain checks for regular A_phi."""
import copy
import json

import numpy as np
import pytest

from temsim.physics.axisymmetric_potential import AxisymmetricMagneticPotential
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.regular_axisymmetric_magnetic_potential import RegularAxisymmetricMagneticPotential


def polynomial_source(registration=None, scale=1.):
    r, z = np.linspace(0., .006, 25), np.linspace(-.01, .01, 17)
    s, zz = r[:, None]**2, z[None, :]
    f = .2+7*zz+90*zz**2+(300+1e4*zz)*s-2e7*s*s
    return AxisymmetricMagneticPotential(r, z, r[:, None]*f,
        registration=registration or CoordinateRegistration(), source_scale=scale)


def reconstructed(source=None, **options):
    return RegularAxisymmetricMagneticPotential.from_potential(source or polynomial_source(),
        fit_radius_m=.004, radial_degree=2, axial_degree=3,
        vacuum_domain_note="Synthetic polynomial test cylinder.", **options)


def analytic(points, scale=1.):
    x, y, z = np.asarray(points).T
    s = x*x+y*y
    f = .2+7*z+90*z*z+(300+1e4*z)*s-2e7*s*s
    fz = 7+180*z+1e4*s
    fs = 300+1e4*z-4e7*s
    return {"A_t_m": scale*np.c_[-y*f, x*f, x*0],
            "B_t": scale*np.c_[-x*fz, -y*fz, 2*f+2*s*fs]}


def test_polynomial_reconstruction_axis_and_source_scale_are_exact():
    points = np.array([[0., 0., -.01], [.0003, -.0004, -.003], [.001, .002, .007], [0., 0., .01]])
    potential = reconstructed(polynomial_source(scale=-1.7))
    for name, expected in analytic(points, scale=-1.7).items():
        np.testing.assert_allclose(potential.sample(points)[name], expected, rtol=3e-13, atol=3e-16)
    assert potential.fit_diagnostics["A_phi_over_r_fit_rms_t"] < 1e-15
    assert potential.sample(points.reshape(2, 2, 3))["B_t"].shape == (2, 2, 3)


def test_rotated_translated_curl_is_the_same_scalar_potential():
    angle = .43
    rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                         [-np.sin(angle), 0., np.cos(angle)]])
    registration = CoordinateRegistration((.01, -.02, .04), tuple(map(tuple, rotation)))
    potential = reconstructed(polynomial_source(registration=registration))
    local = np.array([[.0007, -.0002, -.003], [-.001, .0008, .004]])
    points = local@rotation.T+registration.origin_array_m
    expected = analytic(local)
    for name in expected:
        np.testing.assert_allclose(potential.sample(points)[name], expected[name]@rotation.T,
                                   rtol=3e-13, atol=2e-16)
    epsilon = 2e-8
    derivative = np.stack([(potential.sample(points+epsilon*np.eye(3)[i])["A_t_m"]
                           -potential.sample(points-epsilon*np.eye(3)[i])["A_t_m"])/(2*epsilon)
                           for i in range(3)], axis=-1)
    curl = np.c_[derivative[:, 2, 1]-derivative[:, 1, 2],
                 derivative[:, 0, 2]-derivative[:, 2, 0],
                 derivative[:, 1, 0]-derivative[:, 0, 1]]
    np.testing.assert_allclose(curl, potential.sample(points)["B_t"], rtol=3e-9, atol=1e-10)


def test_reconstruction_is_immutable_explicit_and_has_verified_json_roundtrip():
    source = polynomial_source()
    before = source.to_dict()
    potential = reconstructed(source)
    assert source.to_dict() == before
    assert potential.to_metadata()["source_digest"] == source.digest
    assert potential.to_metadata()["production_admission"] is False
    assert potential.to_metadata()["wave_admission"] is False
    for values in (potential.knots_m, potential.coefficients_t, potential.source.a_phi_tm):
        with pytest.raises(ValueError):
            values.setflags(write=True)
    record = json.loads(json.dumps(potential.to_dict()))
    restored = RegularAxisymmetricMagneticPotential.from_dict(record)
    assert restored.digest == potential.digest
    np.testing.assert_array_equal(restored.sample((.001, .0005, 0.))["B_t"],
                                  potential.sample((.001, .0005, 0.))["B_t"])
    for name in ("metadata", "coefficients", "source"):
        corrupted = copy.deepcopy(record)
        if name == "metadata":
            corrupted[name]["radial_degree"] = 3
        elif name == "coefficients":
            corrupted[name]["coefficients_t"][0][0] += .01
        else:
            corrupted[name]["coefficients"]["a_phi_tm"][1][1] += .001
        with pytest.raises(ValueError):
            RegularAxisymmetricMagneticPotential.from_dict(corrupted)


@pytest.mark.parametrize("point", [(0.00401, 0., 0.), (0., 0., .010001),
                                    (0., 0., -.010001), (np.nan, 0., 0.), (1j, 0., 0.)])
def test_unsupported_points_are_not_zero_filled(point):
    with pytest.raises(ValueError):
        reconstructed().sample(point)


@pytest.mark.parametrize("options", [
    {"fit_radius_m": True}, {"fit_radius_m": .007}, {"fit_radius_m": .0002},
    {"valid_radius_m": .005}, {"valid_radius_m": np.inf},
    {"radial_degree": True}, {"radial_degree": 0}, {"radial_degree": 7},
    {"axial_degree": 4}, {"axial_degree": True},
    {"vacuum_domain_note": ""}, {"vacuum_domain_note": " unstated "},
])
def test_invalid_reconstruction_contract_is_rejected(options):
    settings = dict(fit_radius_m=.004, radial_degree=2, axial_degree=3,
                    vacuum_domain_note="Synthetic source-free cylinder.")
    settings.update(options)
    with pytest.raises(ValueError):
        RegularAxisymmetricMagneticPotential(polynomial_source(), **settings)


def test_fit_window_degrees_and_valid_domain_participate_in_identity():
    base = reconstructed()
    alternatives = [RegularAxisymmetricMagneticPotential(polynomial_source(), radius, degree, axial,
        valid_radius_m=valid, vacuum_domain_note="Synthetic polynomial test cylinder.")
        for radius, degree, axial, valid in ((.003, 2, 3, .003), (.004, 3, 3, .004),
                                           (.004, 2, 5, .004), (.004, 2, 3, .003))]
    assert len({base.digest, *(item.digest for item in alternatives)}) == 5
