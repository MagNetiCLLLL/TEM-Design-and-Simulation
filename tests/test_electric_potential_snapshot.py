"""Exact scalar-law snapshots, independent gradients and strict admission."""
from dataclasses import FrozenInstanceError
import json
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from temsim.physics.axis_regular_potential import AxisRegularPotential
from temsim.physics.axisymmetric_cut_field import AxisymmetricCutField
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.continuous_gun_field import ContinuousGunField
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.grounded_tip_field import GroundedTipField
from temsim.physics.instrument_electric import InstrumentElectricField
from temsim.physics.planar_gun_field import PlanarGunField


def planar(cls=PlanarGunField):
    r = np.array([0., .3e-3, 1e-3])
    z = np.array([0., .7e-3, 2e-3])
    voltage = 12.+2e4*z[None, :]+3e7*r[:, None]**2+1e9*r[:, None]**2*z[None, :]
    return cls({"schema": "snapshot-analytic-fixture", "electrode_v": 100.}, r, z, voltage)


def points():
    return np.array([[0., 0., 0.], [1e-4, -2e-4, .5e-3], [0., 8e-4, 1e-3],
                     [1e-3, 0., 2e-3], [.3e-3, 0., .7e-3]])


def cut_fixture(*, regular=False):
    """Prepared interpolation buffers; no Laplace solve is used by this suite.

    One cell contains a clipped triangular vacuum region. The rest retain the
    native bilinear law. This exercises the existing evaluator, not FEM
    convergence or a claimed physical cathode geometry.
    """
    fem = object.__new__(AxisymmetricCutField)
    fem.r = np.array([0., .5e-3, 1e-3])
    fem.z = np.array([0., .7e-3, 2e-3])
    fem.nodal_voltage = 10.+3e7*fem.r[:, None]**2+2e4*fem.z[None, :]
    fem.nodal_voltage[1, :] += [2., 4., 9.]
    fem.cut_cells = np.array([[True, False], [False, False]])
    vertices = np.array([[.05e-6, .1e-3], [.25e-6, .1e-3], [.25e-6, .7e-3]])
    matrix = np.column_stack((vertices[1]-vertices[0], vertices[2]-vertices[0]))
    fem.origin = vertices[:1]
    fem.inverse = np.linalg.inv(matrix)[None]
    fem.gradient = np.array([[2e7, 3e4]])
    fem.phi0 = np.array([7.])
    fem.lookup = np.full((8, 2), -1, dtype=np.int64)
    fem.lookup[:2, 0] = 0
    if not regular:
        return fem
    core = object.__new__(AxisRegularPotential)
    core.field = fem
    core.compiled = False
    core.fraction = .01
    core.s = np.array([0., .5e-6, .8e-6])
    core.slope = np.array([0., 1.2e7, 1.8e7])
    core.radius_m = np.sqrt(core.s)
    return core


def cut_points():
    return np.array([[0., 0., 0.], [np.sqrt(.2e-6), 0., .3e-3],
        [0., 0., 1e-3], [.2e-3, -.1e-3, 1.4e-3], [.8e-3, 0., .5e-3],
        [1e-3, 0., 2e-3]])


def wrapped(provider, bounds=None):
    if bounds is None:
        bounds = [[-1e-3, -1e-3, 0.], [1e-3, 1e-3, 2e-3]]
    return InstrumentElectricField(provider, provider, None, np.asarray(bounds),
                                   "physical", "numerical", "request", ())


def test_uniform_electric_field_and_cathode_reference():
    r = np.array([0., 1e-3])
    z = np.array([0., 2e-3])
    provider = PlanarGunField({}, r, z, np.broadcast_to(17.+4e4*z, (2, 2)))
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    sampled = snapshot.sample(points())
    np.testing.assert_allclose(sampled["phi_v"], 17.+4e4*points()[:, 2], rtol=1e-15)
    np.testing.assert_allclose(sampled["E_v_per_m"], np.tile([0., 0., -4e4], (5, 1)))
    assert snapshot.metadata["potential_reference"] == "rise_relative_to_original_tip_v"
    assert snapshot.metadata["ground_reference_offset_v"] is None
    assert snapshot.metadata["derivative_order"] == 1
    assert snapshot.metadata["domain"]["radius_m"] == .001
    assert snapshot.metadata["domain"]["z_max_m"] == .002


def test_radial_quadratic_and_axial_mixed_derivatives_are_same_scalar_law():
    snapshot = ElectricPotentialSnapshot.from_provider(planar())
    p = points()
    s, z = np.sum(p[:, :2]**2, axis=1), p[:, 2]
    expected_phi = 12.+2e4*z+3e7*s+1e9*s*z
    expected_e = np.column_stack((-2*p[:, :2]*(3e7+1e9*z)[:, None], -2e4-1e9*s))
    sampled = snapshot.sample(p)
    np.testing.assert_allclose(sampled["phi_v"], expected_phi, rtol=2e-15)
    np.testing.assert_allclose(sampled["E_v_per_m"], expected_e, rtol=2e-15, atol=1e-10)
    # Independent central differences at interior points, away from cell faces.
    p = np.array([[1e-4, 1.4e-4, 1.1e-3]])
    derivative = []
    for axis in range(3):
        delta = np.zeros_like(p)
        delta[:, axis] = 1e-8
        derivative.append(-(snapshot.sample(p+delta)["phi_v"]-snapshot.sample(p-delta)["phi_v"])/2e-8)
    np.testing.assert_allclose(np.array(derivative).T, snapshot.sample(p)["E_v_per_m"], rtol=3e-10)


@pytest.mark.parametrize("cls", [PlanarGunField, ClosedGunField])
@pytest.mark.parametrize("shape", [(3,), (5, 3), (1, 5, 3), (0, 3)])
def test_native_planar_closed_parity_and_shapes(cls, shape):
    provider = planar(cls)
    p = points()[1] if shape == (3,) else points().reshape(shape) if shape != (0, 3) else np.empty(shape)
    expected = provider.interpolate(p)
    result = ElectricPotentialSnapshot.from_provider(provider).sample(p)
    np.testing.assert_array_equal(result["phi_v"], expected[0])
    np.testing.assert_array_equal(result["E_v_per_m"], expected[1])
    assert result["phi_v"].shape == p.shape[:-1]
    assert result["E_v_per_m"].shape == p.shape


@pytest.mark.parametrize("p", [[0., 0., -.01], [0., 0., .003], [0., .002, .001],
                              [1., 2.], [0., np.nan, 0.]])
def test_planar_domain_and_invalid_coordinates_fail(p):
    with pytest.raises(ValueError):
        ElectricPotentialSnapshot.from_provider(planar()).sample(p)


def test_closed_cathode_continuation_and_instrument_domain():
    provider = planar(ClosedGunField)
    p = np.array([[.2e-3, 0., -.01], [0., 0., 0.]])
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    result = snapshot.sample(p)
    expected = provider.interpolate(p)
    np.testing.assert_array_equal(result["phi_v"], expected[0])
    np.testing.assert_array_equal(result["E_v_per_m"], expected[1])
    assert result["phi_v"][0] == 0
    assert snapshot.metadata["domain"]["z_min_m"] is None
    assert not result["E_v_per_m"][0].any()
    with pytest.raises(ValueError, match="no extrapolation"):
        snapshot.sample([0., 0., .003])
    instrument = ElectricPotentialSnapshot.from_provider(wrapped(provider,
        [[-1e-3, -1e-3, -.002], [1e-3, 1e-3, .002]]))
    assert instrument.sample([0., 0., -.001])["phi_v"] == 0
    with pytest.raises(ValueError, match="fixed instrument"):
        instrument.sample(p)


def test_snapshot_detaches_requests_arrays_and_roundtrips_losslessly():
    provider = planar()
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    original = snapshot.sample(points())
    provider.voltage.setflags(write=True)
    provider.voltage[:] = 0
    provider.request["electrode_v"] = -999
    restored = ElectricPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    assert snapshot.digest == restored.digest
    assert restored.metadata["request"]["electrode_v"] == 100.
    for key, value in original.items():
        np.testing.assert_array_equal(restored.sample(points())[key], value)
    with pytest.raises(TypeError):
        snapshot.metadata["request"]["electrode_v"] = 5
    with pytest.raises(FrozenInstanceError):
        snapshot.digest = "modified"
    for value in snapshot.arrays.values():
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert ElectricPotentialSnapshot.from_provider(provider).digest != snapshot.digest


def test_persistence_rejects_tampered_arrays_metadata_and_schema():
    snapshot = ElectricPotentialSnapshot.from_provider(planar())
    record = snapshot.to_dict()
    record["metadata"]["request"]["electrode_v"] = 200.
    with pytest.raises(ValueError, match="digest mismatch"):
        ElectricPotentialSnapshot.from_dict(record)
    record = snapshot.to_dict()
    record["arrays"]["voltage"]["base64"] = "AAAA"
    with pytest.raises(ValueError, match="array representation"):
        ElectricPotentialSnapshot.from_dict(record)
    record = snapshot.to_dict()
    record["schema"] = "future-model"
    with pytest.raises(ValueError, match="schema"):
        ElectricPotentialSnapshot.from_dict(record)


@pytest.mark.parametrize("regular", [False, True])
def test_cut_cells_and_regular_axis_retain_original_basis(regular):
    provider = cut_fixture(regular=regular)
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    p = cut_points()
    expected = provider.interpolate(p)
    sampled = snapshot.sample(p)
    np.testing.assert_allclose(sampled["phi_v"], expected[0], rtol=5e-15, atol=1e-14)
    np.testing.assert_allclose(sampled["E_v_per_m"], expected[1], rtol=5e-15, atol=1e-10)
    assert sampled["phi_v"][0] == 0.  # Outside the clipped vacuum triangle.
    assert sampled["E_v_per_m"][1, 2] == -3e4
    restored = ElectricPotentialSnapshot.from_dict(snapshot.to_dict())
    for name, value in sampled.items():
        np.testing.assert_array_equal(restored.sample(p)[name], value)
    fem = provider.field if regular else provider
    fem.gradient[:] = 0
    np.testing.assert_array_equal(snapshot.sample(p)["E_v_per_m"], sampled["E_v_per_m"])


@pytest.mark.parametrize("kind", [ContinuousGunField, GroundedTipField])
def test_native_gun_cut_field_parity_and_explicit_grounded_continuation(kind):
    regular = cut_fixture(regular=True)
    provider = object.__new__(kind)
    provider._regular, provider._fem = regular, regular.field
    provider.r, provider.z = regular.field.r, regular.field.z
    provider.request = {"schema": "prepared-cut-field-test"}
    provider.high_tension_v = 200000.
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    p = cut_points()
    actual = snapshot.sample(p)
    np.testing.assert_allclose(actual["phi_v"], provider.potential_rise_v_at_global_positions(p), rtol=5e-15)
    np.testing.assert_allclose(actual["E_v_per_m"], provider.field_at_global_positions_v_per_m(p), rtol=5e-15, atol=1e-10)
    if kind is GroundedTipField:
        assert snapshot.metadata["ground_reference_offset_v"] == -200000.
        assert snapshot.metadata["domain"]["z_max_m"] is None
        np.testing.assert_allclose(actual["phi_v"]-200000., provider.potential_v_at_global_positions(p))
        assert snapshot.sample([0., 0., .003])["phi_v"] == 200000.
        np.testing.assert_array_equal(snapshot.sample([0., 0., .003])["E_v_per_m"], np.zeros(3))
        instrument = ElectricPotentialSnapshot.from_provider(wrapped(provider,
            [[-1e-3, -1e-3, 0.], [1e-3, 1e-3, .004]]))
        assert instrument.sample([0., 0., .003])["phi_v"] == 200000.
        assert instrument.metadata["domain"]["z_max_m"] == .004
        with pytest.raises(ValueError, match="fixed instrument"):
            instrument.sample([0., 0., .005])
    else:
        assert snapshot.metadata["ground_reference_offset_v"] is None
        with pytest.raises(ValueError, match="no extrapolation"):
            snapshot.sample([0., 0., .003])


def test_unknown_composed_and_overridden_providers_fail_without_querying():
    class Unknown:
        def interpolate(self, _):
            raise AssertionError("No provider sampling is allowed during capture")
    class CustomPlanar(PlanarGunField):
        pass
    for provider in (Unknown(), planar(CustomPlanar)):
        with pytest.raises(ValueError, match="Unsupported electric potential provider"):
            ElectricPotentialSnapshot.from_provider(provider)
    provider = planar()
    composite = InstrumentElectricField(SimpleNamespace(base_field=provider), provider, None,
        np.array([[-.001, -.001, 0], [.001, .001, .002]]), None, None, None, ())
    with pytest.raises(ValueError, match="Composed"):
        ElectricPotentialSnapshot.from_provider(composite)
    provider.wien_field = object()
    with pytest.raises(ValueError, match="Composed"):
        ElectricPotentialSnapshot.from_provider(wrapped(provider))
    provider.interpolate = MethodType(lambda *_: None, provider)
    with pytest.raises(ValueError, match="Overridden"):
        ElectricPotentialSnapshot.from_provider(provider)


def test_capture_never_calls_a_solver_or_provider_interpolation(monkeypatch):
    provider = planar()
    # Replacing constructors, rather than the interpolation law, keeps the
    # admitted type contract while proving that capture cannot rebuild a solve.
    def fail(*_, **__):
        raise AssertionError("Snapshot capture must not solve or reconstruct a provider")
    monkeypatch.setattr(PlanarGunField, "__init__", fail)
    monkeypatch.setattr(AxisymmetricCutField, "__init__", fail)
    assert ElectricPotentialSnapshot.from_provider(provider).sample([0., 0., 0.])["phi_v"] == 12.
    assert ElectricPotentialSnapshot.from_provider(cut_fixture()).sample([0., 0., 0.])["phi_v"] == 0.


def test_instrument_snapshot_preserves_known_bounds_and_source_identities():
    provider = wrapped(planar(), [[-.0005, -.0005, .0001], [.0005, .0005, .001]])
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    p = np.array([[.0002, -.0001, .0005]])
    phi, electric = provider.interpolate(p)
    np.testing.assert_array_equal(snapshot.sample(p)["phi_v"], phi)
    np.testing.assert_array_equal(snapshot.sample(p)["E_v_per_m"], electric)
    assert snapshot.metadata["source_identities"]["physical"] == "physical"
    with pytest.raises(ValueError, match="fixed instrument"):
        snapshot.sample([.0004, .0004, .0005])
    with pytest.raises(ValueError, match="domain exceeds"):
        ElectricPotentialSnapshot.from_provider(wrapped(planar(),
            [[-.001, -.001, 0.], [.001, .001, .003]]))


def test_cut_and_regular_core_gradients_match_scalar_finite_differences():
    snapshot = ElectricPotentialSnapshot.from_provider(cut_fixture(regular=True))
    # A cut-triangle interior, a regular-axis core, and an uncut outer cell.
    p = np.array([[np.sqrt(.2e-6), 0., .3e-3], [.2e-3, .1e-3, 1.1e-3],
                  [.9e-3, 0., 1.2e-3]])
    derivative = []
    for axis in range(3):
        delta = np.zeros_like(p)
        delta[:, axis] = 1e-9
        derivative.append(-(snapshot.sample(p+delta)["phi_v"]-snapshot.sample(p-delta)["phi_v"])/2e-9)
    np.testing.assert_allclose(np.array(derivative).T, snapshot.sample(p)["E_v_per_m"],
                               rtol=5e-9, atol=1e-5)


def test_existing_compiled_evaluator_parity_without_resampling():
    from temsim.physics.axis_field_interpolation import compiled_evaluate
    if compiled_evaluate is None:
        pytest.skip("Optional Numba evaluator is unavailable")
    provider = cut_fixture(regular=True)
    provider.compiled = True
    snapshot = ElectricPotentialSnapshot.from_provider(provider)
    p = cut_points()
    phi, electric = provider.interpolate(p)
    result = snapshot.sample(p)
    np.testing.assert_allclose(result["phi_v"], phi, rtol=5e-15, atol=1e-14)
    np.testing.assert_allclose(result["E_v_per_m"], electric, rtol=5e-15, atol=1e-10)
