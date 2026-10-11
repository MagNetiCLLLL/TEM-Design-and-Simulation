"""Native particle device parity, including thin-screen checkpoint semantics."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from temsim.physics.electrostatic_column_transport import (
    _grid_electric, _grid_electric_arrays, _momentum_speed, electrostatic_column_rk4,
)
from temsim.physics.intrinsic_aberrations import FrozenIntrinsicAberrationKick
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.particle_gpu_reference import gpu_reference_transport
from temsim.physics.posed_aberrations import FrozenLensAberrationKick
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.vector_field_transport import vector_map_rk4


@pytest.fixture
def cp():
    from temsim.physics.compute_backend import cupy_capability, cupy_module
    if not cupy_capability().available:
        pytest.skip("Real CuPy CUDA device is unavailable")
    return cupy_module()


def inputs_fixture(*, nodes=9):
    z = np.linspace(499., 501., nodes) if nodes > 1 else np.array([500.])
    n = 2*nodes-1
    energy = np.array([270000., 300000., 330000.])
    momentum = np.array([_momentum_speed(value)[0] for value in energy])
    inverse_speed = np.array([1./_momentum_speed(value)[1] for value in energy])
    actions = [np.zeros(nodes) for _ in range(5)]
    middle = nodes//2
    actions[0][middle], actions[1][middle], actions[2][middle] = 1e8, 10., .01
    actions[3][middle], actions[4][middle] = 1e-4, -2e-4
    values = (np.full(n, 4.), np.full(n, 5.), np.full(n, 30.), np.full(n, -20.),
        np.full(n, 1.602176634e-19*.003/2.), 1./momentum,
        *actions[:3], np.diff(z)*1e-3,
        np.array([2e-7, -4e-7, 5e-7]), np.array([1e-3, -1.5e-3, 2e-3]),
        np.array([-3e-7, 2e-7, 4e-7]), np.array([-.5e-3, 1e-3, -1e-3]),
        *actions[3:], np.arange(nodes), np.arange(nodes), np.full(n, .5),
        np.full(3*(nodes-1), 1e-5), np.full(3*(nodes-1), -2e-5),
        np.array([_momentum_speed(300000.)[0]]))
    return values, z, energy, inverse_speed


def native_optics():
    pose = CoordinateRegistration((1e-7, -2e-7, .5),
        tuple(map(tuple, Rotation.from_rotvec((.0008, -.0004, .01)).as_matrix())))
    fields = (
        FrozenAnalyticField("lens", ((.015, 0., .006),), pose, (-20., 20.), .01),
        FrozenPosedMultipole("stigmator", "stigmator", pose,
            ((2, 0, .02), (0, 2, -.02), (1, 0, 1e-5)), "gaussian", 0., .006,
            (-20., 20.), .01, _momentum_speed(300000.)[0]),
    )
    kicks = (FrozenLensAberrationKick("Cs", 0., 1e8, pose),
        FrozenIntrinsicAberrationKick("intrinsic", 0., .01,
            (("c1_mm", .01), ("a1_mm", .002), ("a1_azimuth_deg", 23.),
             ("b2_mm", .02), ("b2_azimuth_deg", -13.)), pose))
    return fields, kicks


def electric_fixture():
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.instrument_electric import InstrumentElectricField
    r, z = np.array([0., .001, .01]), np.array([.49, .5, .51])
    voltage = 2e6*(z[None, :]-.49)+2e7*r[:, None]**2
    base = ClosedGunField({}, r, z, voltage)
    return InstrumentElectricField(base, base, SimpleNamespace(exit_plane_z_mm=490.),
        np.array([[-.01, -.01, .49], [.01, .01, .51]]), "grid", "grid", "grid", ())


def assert_transport_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, wanted in zip(actual, expected):
        assert isinstance(got, np.ndarray)
        np.testing.assert_allclose(got, wanted, rtol=3e-11, atol=2e-20, equal_nan=True)


def electric_boundary_cases():
    r = np.array([0., .001, .002, .008])
    z = np.array([.1, .12, .19])
    voltage = np.array([[0., 4., -2.], [1., 7., 13.], [8., -3., 1.], [2., 6., 9.]])
    radii = np.r_[r, np.nextafter(r[1:], 0.), np.nextafter(r, np.inf), np.nan]
    axial = np.r_[z, np.nextafter(z, 0.), np.nextafter(z, np.inf), np.nan]
    rr, zz = np.meshgrid(radii, axial)
    return (r, z, voltage), rr.ravel(), np.zeros(rr.size), zz.ravel()


def test_array_electric_interpolation_matches_scalar_cells_and_boundaries():
    data, x, y, z = electric_boundary_cases()
    expected = np.array([_grid_electric(a, b, c, data) for a, b, c in zip(x, y, z)]).T
    actual = np.array(_grid_electric_arrays(x, y, z, data))
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-12, equal_nan=True)


def test_gpu_electric_interpolation_matches_scalar_cells_and_boundaries(cp):
    data, x, y, z = electric_boundary_cases()
    expected = np.array([_grid_electric(a, b, c, data) for a, b, c in zip(x, y, z)]).T
    actual = _grid_electric_arrays(cp.asarray(x), cp.asarray(y), cp.asarray(z),
        tuple(cp.asarray(value) for value in data), xp=cp)
    np.testing.assert_allclose(cp.asnumpy(cp.stack(actual)), expected,
        rtol=2e-14, atol=1e-11, equal_nan=True)


@pytest.mark.parametrize("timed", [False, True])
def test_native_magnetic_gpu_matches_reference_and_keeps_inputs(cp, timed, monkeypatch):
    values, z, energy, inv_speed = inputs_fixture()
    original = tuple(value.copy() for value in values)
    fields, kicks = native_optics()
    actions = {len(z)//2: kicks}
    times = np.array([0., 1e-10, 2e-10]) if timed else None
    expected = vector_map_rk4(*values, z_mm=z, mapped_fields=fields,
        posed_spherical_kicks=actions, initial_time_s=times, inverse_speed=inv_speed)
    # Prove both field queries and the new screen receive device arrays.
    seen = []
    for cls, name in ((FrozenAnalyticField, "slope_derivative"),
                      (FrozenIntrinsicAberrationKick, "apply_canonical")):
        old = getattr(cls, name)
        def watch(self, first, *args, _old=old, _name=name, **kwargs):
            assert isinstance(first, cp.ndarray)
            seen.append(_name)
            return _old(self, first, *args, **kwargs)
        monkeypatch.setattr(cls, name, watch)
    actual = gpu_reference_transport(values, z_mm=z, initial_kinetic_energy_ev=energy,
        mapped_fields=fields, posed_spherical_kicks=actions, initial_time_s=times,
        inverse_speed=inv_speed)
    assert_transport_equal(actual, expected)
    assert set(seen) == {"slope_derivative", "apply_canonical"}
    assert len(actual) == (10 if timed else 8)
    for got, before in zip(values, original):
        np.testing.assert_array_equal(got, before)


@pytest.mark.parametrize("optical_invariant", [None, 300000.])
def test_native_electric_gpu_matches_reference_energy_and_flight_time(cp, optical_invariant):
    values, z, energy, _ = inputs_fixture()
    fields, kicks = native_optics()
    options = dict(z_mm=z, electric_field=electric_fixture(),
        initial_kinetic_energy_ev=energy, initial_time_s=np.array([0., 1e-10, 2e-10]),
        mapped_fields=fields, posed_spherical_kicks={len(z)//2: kicks},
        optical_reference_invariant_ev=optical_invariant)
    expected, _, _ = electrostatic_column_rk4(values, **options)
    actual = gpu_reference_transport(values, **options)
    assert len(actual) == 12
    assert_transport_equal(actual, expected)


def test_aligned_magnetic_intervals_keep_timing_and_stopped_nan_rays(cp):
    values, z, energy, inverse_speed = inputs_fixture()
    values = tuple(value.copy() for value in values)
    for index in (10, 11, 12, 13):
        values[index][-1] = np.nan
    _, kicks = native_optics()
    # No active field provider: use the existing scalar/array canonical RK
    # stage law while the thin screen still executes on the GPU.
    options = dict(z_mm=z, mapped_fields=(), posed_spherical_kicks={len(z)//2: kicks},
        initial_time_s=np.zeros(3), inverse_speed=inverse_speed,
        defer_nonfinite_until_clipping=True)
    expected = vector_map_rk4(*values, **options)
    actual = gpu_reference_transport(values, initial_kinetic_energy_ev=energy, **options)
    assert_transport_equal(actual, expected)
    assert np.isnan(actual[0][:, -1]).all()


@pytest.mark.parametrize("electric", [False, True])
def test_zero_interval_records_actions_once_before_checkpoint(cp, electric):
    values, z, energy, inv_speed = inputs_fixture(nodes=1)
    fields, kicks = native_optics()
    options = dict(z_mm=z, mapped_fields=fields, posed_spherical_kicks={0: kicks},
        initial_time_s=np.array([0., 1e-10, 2e-10]))
    if electric:
        options.update(electric_field=electric_fixture(), initial_kinetic_energy_ev=energy)
        expected, _, _ = electrostatic_column_rk4(values, **options)
    else:
        expected = vector_map_rk4(*values, **options, inverse_speed=inv_speed)
    actual = gpu_reference_transport(values, **options)
    assert_transport_equal(actual, expected)
    # Checkpoint is after the thin actions, not the unchanged entrance state.
    assert np.max(abs(actual[5][0]-values[11])) > 1e-5


def test_gpu_cancellation_after_device_actions_preserves_original_inputs(cp):
    values, z, energy, _ = inputs_fixture(nodes=1)
    original = tuple(value.copy() for value in values)
    _, kicks = native_optics()
    calls = []
    def cancel():
        calls.append(True)
        if len(calls) == 3:
            raise InterruptedError("cancel before result publication")
    with pytest.raises(InterruptedError, match="publication"):
        gpu_reference_transport(values, z_mm=z, initial_kinetic_energy_ev=energy,
            posed_spherical_kicks={0: kicks}, cancel_check=cancel)
    for got, before in zip(values, original):
        np.testing.assert_array_equal(got, before)


def test_unaudited_field_is_rejected_before_device_allocation(monkeypatch):
    from temsim.physics import particle_gpu_reference as transport
    from temsim.physics.compute_backend import GPUExecutionError
    def forbidden():
        raise AssertionError("Device must not be allocated for unsupported field")
    monkeypatch.setattr(transport, "cupy_module", forbidden)
    values, z, _, _ = inputs_fixture()
    with pytest.raises(GPUExecutionError, match="Imported magnetic"):
        gpu_reference_transport(values, z_mm=z, mapped_fields=(object(),))
