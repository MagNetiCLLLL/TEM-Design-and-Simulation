"""Independent magnetic thin-event laws and local-energy backend regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.optics.column import default_state
from temsim.optics.equivalent_image_lenses import equivalent_image_calibrations
from temsim.physics import ray_integrator
from temsim.physics.compute_backend import cuda_capability
from temsim.physics.electrostatic_column_transport import electrostatic_column_rk4
from temsim.physics.vector_field_transport import vector_map_rk4


MASS = 9.1093837015e-31
CHARGE = 1.602176634e-19
LIGHT = 299792458.0


def momentum(energy_ev):
    kinetic = np.asarray(energy_ev) * CHARGE
    return np.sqrt(kinetic * (kinetic + 2 * MASS * LIGHT**2)) / LIGHT


def inputs(energy, z_m=None, event=1, *, zero_slopes=False):
    """The established 22-array ABI, with no other fields or optical actions."""
    energy = np.asarray(energy, dtype=float)
    z = np.array([0.0, .001, .002]) if z_m is None else z_m
    nodes, count = len(z), len(energy)
    profile = np.zeros(2 * nodes - 1)
    power, rotation = np.zeros(nodes), np.zeros(nodes)
    power[event], rotation[event] = 120., -.27
    x = np.linspace(-2e-6, 4e-6, count)
    y = np.linspace(3e-6, -1e-6, count)
    tx = np.zeros(count) if zero_slopes else np.linspace(1e-4, -3e-4, count)
    ty = np.zeros(count) if zero_slopes else np.linspace(-2e-4, 1e-4, count)
    forcing = np.zeros(3 * (nodes - 1))
    return (
        profile.copy(), profile.copy(), profile.copy(), profile.copy(), profile.copy(),
        1 / momentum(energy), np.zeros(nodes), power, rotation, np.diff(z),
        x, tx, y, ty, np.zeros(nodes), np.zeros(nodes),
        np.arange(nodes, dtype=np.int64), np.arange(nodes, dtype=np.int64),
        profile.copy(), forcing.copy(), forcing.copy(), np.array([momentum(300000.)]),
    )


def event_oracle(coordinates, energy, *, power=120., rotation=-.27):
    """Lorentz thin-lens law: focusing ~ 1/p^2 and Larmor rotation ~ 1/p."""
    x, tx, y, ty = np.asarray(coordinates).copy()
    scale = momentum(300000.) / momentum(energy)
    tx -= power * scale**2 * x
    ty -= power * scale**2 * y
    cosine, sine = np.cos(rotation * scale), np.sin(rotation * scale)
    return np.array((cosine*x-sine*y, cosine*tx-sine*ty,
                     sine*x+cosine*y, sine*tx+cosine*ty))


def drift(values, distance):
    x, tx, y, ty = np.asarray(values).copy()
    return np.array((x+distance*tx, tx, y+distance*ty, ty))


def require_backend(name):
    if name.startswith("numba") and not ray_integrator.NUMBA_AVAILABLE:
        pytest.skip("Numba is unavailable")
    if name in {"cuda", "vector_cupy", "CUDA GPU"} and not cuda_capability().available:
        pytest.skip("CUDA hardware is unavailable")


@pytest.mark.parametrize("timed", [False, True])
@pytest.mark.parametrize("backend", ["numpy", "numba_serial", "numba_parallel", "vector_numpy", "cuda", "vector_cupy"])
def test_mixed_energy_thin_event_matches_independent_oracle(backend, timed):
    require_backend(backend)
    energy = np.array([300000., 299000., 350000.])
    arguments = inputs(energy)
    originals = tuple(value.copy() for value in arguments)
    timing = (dict(initial_time_s=np.zeros(3), inverse_speed=np.full(3, 5e-9))
              if timed else {})
    if backend == "numpy":
        result = ray_integrator.vectorised_rk4(*arguments, **timing)
    elif backend.startswith("numba"):
        call = ray_integrator.serial_rk4 if backend == "numba_serial" else ray_integrator.parallel_rk4
        result = call(*arguments, **timing)
    elif backend == "cuda":
        result = ray_integrator.cuda_rk4(*arguments, **timing)
    elif backend == "vector_cupy":
        cp = pytest.importorskip("cupy")
        # Axial coefficients remain host arrays, as in the production GPU wrapper.
        device = tuple(cp.asarray(value) if index in (5, 10, 11, 12, 13) else value
                       for index, value in enumerate(arguments))
        device_timing = {key: cp.asarray(value) for key, value in timing.items()}
        result = vector_map_rk4(*device, z_mm=np.array([0., 1., 2.]),
                               mapped_fields=(), xp=cp, **device_timing)
        result = tuple(cp.asnumpy(value) for value in result)
    else:
        result = vector_map_rk4(*arguments, z_mm=np.array([0., 1., 2.]),
                               mapped_fields=(), **timing)
    at_event = event_oracle(drift(arguments[10:14], .001), energy)
    expected = drift(at_event, .001)
    checkpoints = np.asarray(result[4:8])
    np.testing.assert_allclose(checkpoints[:, 1], at_event, rtol=3e-13, atol=1e-20)
    np.testing.assert_allclose(checkpoints[:, -1], expected, rtol=3e-13, atol=1e-20)
    for before, after in zip(originals, arguments, strict=True):
        np.testing.assert_array_equal(after, before)


def electric_fixture(z):
    from temsim.physics.closed_gun_field import ClosedGunField
    from temsim.physics.instrument_electric import InstrumentElectricField
    radial = np.array([0., .01, .02])
    axes = np.array([z[0], .5*(z[0]+z[-1]), z[-1]])
    gradient = 2e7
    voltage = np.broadcast_to(gradient*axes, (len(radial), len(axes))).copy()
    base = ClosedGunField({}, radial, axes, voltage)
    field = InstrumentElectricField(base, base, SimpleNamespace(exit_plane_z_mm=0.),
        np.array([[-.02, -.02, z[0]], [.02, .02, z[-1]]]),
        "thin-energy-fixture", "thin-energy-fixture", "thin-energy-fixture", ())
    return field, gradient


@pytest.mark.parametrize("backend", ["CPU", "Numba CPU", "CUDA GPU"])
@pytest.mark.parametrize("optical_invariant", [None, 300000.])
def test_electric_event_uses_actual_local_momentum(backend, optical_invariant):
    require_backend("numba_serial" if backend == "Numba CPU" else backend)
    energy = np.array([300000., 280000., 340000.])
    z = np.linspace(0., .002, 101)
    event = 50
    arguments = inputs(energy, z, event, zero_slopes=True)
    field, gradient = electric_fixture(z)
    actual, selected, reason = electrostatic_column_rk4(arguments, z_mm=z*1e3,
        electric_field=field, initial_kinetic_energy_ev=energy,
        optical_reference_invariant_ev=optical_invariant, initial_time_s=np.zeros(3),
        backend=backend, policy="require_gpu" if backend == "CUDA GPU" else "auto", serial=True)
    assert selected == backend and reason is None
    local_energy = energy + gradient*z[event]
    expected_event = event_oracle(arguments[10:14], local_energy)
    checkpoints = np.asarray(actual[4:8])
    np.testing.assert_allclose(checkpoints[:, event], expected_event, rtol=3e-12, atol=1e-20)
    np.testing.assert_allclose(actual[11], energy[None, :]+gradient*z[:, None], rtol=2e-15)
    if optical_invariant is None:
        # Uniform axial E conserves each transverse mechanical momentum after the event.
        end_energy = energy + gradient*z[-1]
        end = expected_event.copy()
        for ray in range(len(energy)):
            distance_factor = momentum(local_energy[ray])*quad(
                lambda zz: 1/momentum(energy[ray]+gradient*zz), z[event], z[-1],
                epsabs=1e-6, epsrel=2e-13)[0]
            end[0, ray] += expected_event[1, ray]*distance_factor
            end[2, ray] += expected_event[3, ray]*distance_factor
        end[[1, 3]] *= momentum(local_energy)/momentum(end_energy)
        np.testing.assert_allclose(checkpoints[:, -1], end, rtol=2e-11, atol=1e-20)


@pytest.fixture
def calibration_state():
    return default_state()


@pytest.mark.parametrize("change", ["lower_center", "upper_center", "normalise_peak", "term_partition"])
def test_calibration_cache_tracks_every_native_profile_boundary(calibration_state, change):
    state = calibration_state
    lower, upper = float(state.sample.z_mm), 3000.
    original = equivalent_image_calibrations(state, lower, upper)
    assert equivalent_image_calibrations(state, lower, upper) is original
    if change == "lower_center":
        state.objective_lens.lower_field_center_z_mm += 1.
    elif change == "upper_center":
        state.objective_lens.upper_field_center_z_mm += 1.
    elif change == "normalise_peak":
        state.projector_lens_p1.normalise_profile_peak = not state.projector_lens_p1.normalise_profile_peak
    else:
        objective = state.objective_lens
        # Same flattened coefficients; different upper/lower physical profiles.
        objective.lower_gaussian = objective.upper_gaussian[-1:] + objective.lower_gaussian
        objective.upper_gaussian = objective.upper_gaussian[:-1]
    warm = equivalent_image_calibrations(state, lower, upper)
    assert warm is not original
    assert warm != original
    del state._equivalent_image_calibration_cache
    fresh = equivalent_image_calibrations(state, lower, upper)
    assert warm == fresh


def test_production_image_event_uses_each_actual_particle_energy(calibration_state):
    from temsim.optics.equivalent_image_lenses import equivalent_image_events
    from temsim.physics.core import build_propagation_plan, execute_propagation_plan
    state = calibration_state
    state._propagation_energy_kev = state.beam_voltage_kv
    state.electron_gun = None
    state.projector_mode = "image"
    state.equivalent_image_lenses_enabled = True
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.stigmators, state.corrector_elements, state.deflectors = [], [], []
    state.step_mm = state.history_step_mm = .1
    start, end = state.projector_lens_p1.z_mm-1, state.projector_lens_p1.z_mm+1
    plan = build_propagation_plan(state, start, end, checkpoint_z_mm=(end,),
        include_spherical_aberration=False, include_hexapole=False)
    event, = equivalent_image_events(state, start, end)
    assert np.max(abs(plan.magnetic_t)) == 0.
    energy = np.array([300000., 299000.])
    initial = np.array([[1e-6, 1e-6], [0., 0.], [2e-6, 2e-6], [0., 0.]])
    result = execute_propagation_plan(state, plan, *initial, initial_kinetic_energy_ev=energy)[-1]
    expected = event_oracle(drift(initial, (event.z_mm-start)*1e-3), energy,
                            power=event.power_m1, rotation=event.rotation_rad)
    expected = drift(expected, (end-event.z_mm)*1e-3)
    actual = np.array([result.x_m[-1], result.tx_rad[-1], result.y_m[-1], result.ty_rad[-1]])
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-20)
