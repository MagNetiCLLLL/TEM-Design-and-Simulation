"""Analytical fields and exact arithmetic for the planar sampling accelerator.

These bounded operator checks do not solve an electrode field, propagate a
particle or wave, or establish full-column numerical convergence.
"""
import numpy as np
import pytest

from temsim.physics import electric_potential_snapshot as module
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.grounded_tip_field import GroundedTipField
from temsim.physics.instrument_electric import InstrumentElectricField
from temsim.physics.planar_gun_field import PlanarGunField


@pytest.fixture
def native():
    implementation = module._bilinear_native
    if implementation is None:
        pytest.skip("Optional Numba planar accelerator is unavailable")
    return implementation


def _exact_float64(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype == np.dtype(np.float64)
    assert actual.shape == expected.shape
    # Ordinary equality hides changes to the sign bit of zero.
    np.testing.assert_array_equal(
        np.ascontiguousarray(actual).view(np.uint64),
        np.ascontiguousarray(expected).view(np.uint64),
    )


def _planar(kind=PlanarGunField):
    r = np.array([0., .13e-3, .7e-3, 1.e-3])
    z = np.array([0., .2e-3, .9e-3, 2.e-3])
    voltage = 17. + 2.e4*z[None, :] + 3.e7*r[:, None]**2 + 1.e9*r[:, None]**2*z[None, :]
    return kind({"scope": "Prepared analytic scalar-law fixture"}, r, z, voltage)


def _forbidden_reference(*args, **kwargs):
    raise AssertionError("The public planar route must not fall back to the NumPy reference")


def test_mixed_radial_axial_potential_and_gradient_have_analytic_units_and_symmetry(native, monkeypatch):
    snapshot = module.ElectricPotentialSnapshot.from_provider(_planar())
    points = np.array([[.1e-3, -.2e-3, .6e-3], [-.2e-3, -.1e-3, .6e-3],
                       [0., 0., 1.4e-3], [.4e-3, .3e-3, 1.2e-3]])
    squared_radius = np.sum(points[:, :2]**2, axis=1)
    axial = points[:, 2]
    expected_phi = 17. + 2.e4*axial + 3.e7*squared_radius + 1.e9*squared_radius*axial
    expected_e = np.column_stack((
        -2*points[:, 0]*(3.e7 + 1.e9*axial),
        -2*points[:, 1]*(3.e7 + 1.e9*axial),
        -2.e4 - 1.e9*squared_radius,
    ))
    monkeypatch.setattr(module, "_bilinear", _forbidden_reference)
    actual = snapshot.sample(points)
    np.testing.assert_allclose(actual["phi_v"], expected_phi, rtol=3e-15, atol=1e-13)
    np.testing.assert_allclose(actual["E_v_per_m"], expected_e, rtol=3e-15, atol=2e-10)
    assert actual["phi_v"][0] == actual["phi_v"][1]
    np.testing.assert_array_equal(actual["E_v_per_m"][2, :2], np.zeros(2))
    # A line integral in Z must recover the potential difference, including
    # the r^2 Z cross term. This checks E = -grad(Phi), not just sample parity.
    end = points.copy()
    end[:, 2] += .1e-3
    change = snapshot.sample(end)["phi_v"] - actual["phi_v"]
    np.testing.assert_allclose(change, -actual["E_v_per_m"][:, 2]*.1e-3,
                               rtol=3e-14, atol=5e-14)


def test_cell_faces_select_the_right_derivative_without_smoothing(native):
    r, z = np.array([0., 1., 3.]), np.array([-2., 1., 4.])
    # Radial derivatives dPhi/d(r^2) are 2 and 7; axial derivatives are 3
    # and 5. Phi is continuous, but the declared cell derivatives jump.
    voltage = np.array([0., 2., 58.])[:, None] + np.array([-9., 0., 15.])[None, :]
    lower, upper = np.nextafter(1., -np.inf), np.nextafter(1., np.inf)
    points = np.array([[lower, 0., lower], [1., 0., 1.], [upper, 0., upper], [0., 0., 1.]])
    phi, electric = native(points, np.hypot(points[:, 0], points[:, 1]), r, z, voltage)
    np.testing.assert_allclose(electric[:, 0], [-4*lower, -14., -14*upper, 0.], rtol=0., atol=0.)
    np.testing.assert_allclose(electric[:, 2], [-3., -5., -5., -5.], rtol=2e-15, atol=0.)
    assert phi[1] == 2.
    assert phi[3] == 0.
    expected = module._bilinear(points, r, z, voltage)
    _exact_float64(phi, expected[0])
    _exact_float64(electric, expected[1])


@pytest.mark.parametrize("scale", [1e-100, 1e-6, .1, 1e100])
@pytest.mark.parametrize("voltage_kind", ["mixed", "zero", "negative_zero"])
def test_nonuniform_cells_faces_nextafter_and_signed_zero_match_reference_bytes(native, scale, voltage_kind):
    rng = np.random.default_rng(742103)
    r = np.r_[0., np.cumsum(rng.uniform(.1, 1., 6))]*scale
    z = np.cumsum(rng.uniform(.1, 1., 6))*scale
    voltage = rng.normal(size=(len(r), len(z)))*1e5
    if voltage_kind != "mixed":
        voltage.fill(-0. if voltage_kind == "negative_zero" else 0.)
    radii = rng.uniform(0., r[-1], 257)
    angles = rng.uniform(-np.pi, np.pi, len(radii))
    points = np.column_stack((radii*np.cos(angles), radii*np.sin(angles),
                              rng.uniform(z[0], z[-1], len(radii))))
    rr, zz = np.meshgrid(r, z)
    faces = np.column_stack((rr.ravel(), np.zeros(rr.size), zz.ravel()))
    signed_axis = np.array([[0., -0., z[0]], [-0., 0., z[-1]],
                            [-0., -0., z[1]], [0., 0., z[2]]])
    points = np.vstack((points, faces, np.nextafter(faces, np.inf),
                        np.nextafter(faces, -np.inf), signed_axis))
    originals = tuple(value.copy() for value in (points, r, z, voltage))
    radius = np.hypot(points[:, 0], points[:, 1])
    expected = module._bilinear(points, r, z, voltage)
    actual = native(points, radius, r, z, voltage)
    for observed, reference in zip(actual, expected, strict=True):
        assert np.isfinite(observed).all()
        _exact_float64(observed, reference)
    for observed, original in zip((points, r, z, voltage), originals, strict=True):
        _exact_float64(observed, original)


@pytest.mark.parametrize("shape", [(3,), (2, 2, 3), (0, 3), (2, 0, 3)])
def test_public_sample_preserves_shape_exact_values_and_input_bytes(native, shape, monkeypatch):
    snapshot = module.ElectricPotentialSnapshot.from_provider(_planar())
    points = np.array([[0., -0., 0.], [.1e-3, -.2e-3, .6e-3],
                       [0., .7e-3, .9e-3], [1.e-3, 0., 2.e-3]])
    if shape == (3,):
        points = points[1]
    elif 0 in shape:
        points = np.empty(shape, dtype=np.float64)
    else:
        points = points.reshape(shape)[:, ::-1, :]
        assert not points.flags.c_contiguous
    original = points.copy()
    arrays = snapshot.arrays
    phi, electric = module._bilinear(points.reshape(-1, 3), arrays["r"], arrays["z"], arrays["voltage"])
    monkeypatch.setattr(module, "_bilinear", _forbidden_reference)
    actual = snapshot.sample(points)
    _exact_float64(actual["phi_v"], phi.reshape(shape[:-1]))
    _exact_float64(actual["E_v_per_m"], electric.reshape(shape))
    _exact_float64(points, original)


def test_flat_cathode_boundary_keeps_declared_zero_extension_only(native, monkeypatch):
    snapshot = module.ElectricPotentialSnapshot.from_provider(_planar(ClosedGunField))
    arrays = snapshot.arrays
    points = np.array([[.1e-3, -.2e-3, np.nextafter(0., -np.inf)],
                       [.1e-3, -.2e-3, -0.], [.1e-3, -.2e-3, 0.],
                       [.1e-3, -.2e-3, np.nextafter(0., np.inf)]])
    clamped = points.copy()
    clamped[:, 2] = np.maximum(clamped[:, 2], 0.)
    phi, electric = module._bilinear(clamped, arrays["r"], arrays["z"], arrays["voltage"])
    phi[0], electric[0] = 0., 0.
    monkeypatch.setattr(module, "_bilinear", _forbidden_reference)
    actual = snapshot.sample(points)
    _exact_float64(actual["phi_v"], phi)
    _exact_float64(actual["E_v_per_m"], electric)
    assert actual["phi_v"][1] > 0.
    assert actual["E_v_per_m"][1, 2] < 0.


def test_optional_accelerator_absence_preserves_numpy_reference_bytes(monkeypatch):
    snapshot = module.ElectricPotentialSnapshot.from_provider(_planar())
    points = np.array([[[0., -0., 0.], [.1e-3, -.2e-3, .6e-3]],
                       [[0., .7e-3, .9e-3], [1.e-3, 0., 2.e-3]]])
    arrays = snapshot.arrays
    phi, electric = module._bilinear(points.reshape(-1, 3), arrays["r"], arrays["z"], arrays["voltage"])
    monkeypatch.setattr(module, "_bilinear_native", None)
    actual = snapshot.sample(points)
    _exact_float64(actual["phi_v"], phi.reshape(points.shape[:-1]))
    _exact_float64(actual["E_v_per_m"], electric.reshape(points.shape))


def _grounded():
    # Prepared uncut FEM buffers exercise the retained non-planar dispatch;
    # no electrostatic solver or fitted field is involved.
    planar = _planar()
    fem = object.__new__(AxisymmetricCutField)
    fem.r, fem.z, fem.nodal_voltage = planar.r, planar.z, planar.voltage
    cells = (len(fem.r)-1)*(len(fem.z)-1)
    fem.cut_cells = np.zeros((len(fem.r)-1, len(fem.z)-1), dtype=bool)
    fem.lookup = np.full((2*cells, 2), -1, dtype=np.int64)
    fem.origin, fem.inverse = np.empty((0, 2)), np.empty((0, 2, 2))
    fem.gradient, fem.phi0 = np.empty((0, 2)), np.empty(0)
    provider = object.__new__(GroundedTipField)
    provider._fem, provider._regular = fem, None
    provider.r, provider.z = fem.r, fem.z
    provider.request, provider.high_tension_v = {}, 200000.
    return provider


def test_grounded_tail_and_mixed_cut_basis_queries_never_use_planar_accelerator(monkeypatch):
    snapshot = module.ElectricPotentialSnapshot.from_provider(_grounded())
    edge = snapshot.arrays["z"][-1]
    points = np.array([[.1e-3, 0., np.nextafter(edge, -np.inf)],
                       [.1e-3, 0., edge], [.1e-3, 0., np.nextafter(edge, np.inf)]])
    monkeypatch.setattr(module, "_bilinear_native", _forbidden_reference)
    result = snapshot.sample(points)
    assert result["phi_v"][0] < 200000.
    assert result["E_v_per_m"][0, 2] < 0.
    _exact_float64(result["phi_v"][1:], np.full(2, 200000.))
    _exact_float64(result["E_v_per_m"][1:], np.zeros((2, 3)))
    assert snapshot.metadata["ground_reference_offset_v"] == -200000.
    # The all-tail shortcut must still avoid any interpolation, including for
    # an empty batch; this is an explicitly constant declared continuation.
    monkeypatch.setattr(module, "_evaluate", _forbidden_reference)
    for tail in (points[1:], np.empty((2, 0, 3))):
        result = snapshot.sample(tail)
        _exact_float64(result["phi_v"], np.full(tail.shape[:-1], 200000.))
        _exact_float64(result["E_v_per_m"], np.zeros(tail.shape))


@pytest.mark.parametrize("kind,point", [
    (PlanarGunField, [np.nextafter(.001, np.inf), 0., .001]),
    (PlanarGunField, [0., 0., np.nextafter(0., -np.inf)]),
    (PlanarGunField, [0., 0., np.nextafter(.002, np.inf)]),
    (PlanarGunField, [0., np.nan, .001]),
    (PlanarGunField, [0., 0., np.inf]),
    (PlanarGunField, [0., 0.]),
    (ClosedGunField, [np.nextafter(.001, np.inf), 0., -.001]),
    (ClosedGunField, [0., 0., np.nextafter(.002, np.inf)]),
    (GroundedTipField, [np.nextafter(.001, np.inf), 0., .003]),
    (GroundedTipField, [0., 0., np.nextafter(0., -np.inf)]),
])
def test_invalid_domains_are_rejected_before_any_interpolation(kind, point, monkeypatch):
    provider = _grounded() if kind is GroundedTipField else _planar(kind)
    snapshot = module.ElectricPotentialSnapshot.from_provider(provider)
    monkeypatch.setattr(module, "_bilinear_native", _forbidden_reference)
    monkeypatch.setattr(module, "_evaluate", _forbidden_reference)
    with pytest.raises(ValueError, match="finite XYZ|no extrapolation"):
        snapshot.sample(point)


def test_fixed_instrument_radius_still_rejects_rectangular_corner_before_kernel(monkeypatch):
    provider = _planar()
    wrapped = InstrumentElectricField(provider, provider, None,
        np.array([[-.0005, -.0005, .0001], [.0005, .0005, .001]]),
        "physical", "numerical", "request", ())
    snapshot = module.ElectricPotentialSnapshot.from_provider(wrapped)
    monkeypatch.setattr(module, "_bilinear_native", _forbidden_reference)
    with pytest.raises(ValueError, match="fixed instrument"):
        snapshot.sample([.0004, .0004, .0005])
