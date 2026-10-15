"""Diagnostic packet mathematics; never full microscope/source qualification."""
import gc
import weakref

import numpy as np
import pytest

from scripts import experimental_gaussian_beamlets as beam
from scripts import validate_experimental_gaussian_beamlets as validation


def test_exact_affine_field_and_single_axis_caustic_lift():
    evidence = validation.canonical_checks()
    assert max(evidence["canonical_complex_relative_errors"]) < 2e-10
    assert abs(evidence["single_axis_full_cycle_phase"]) == pytest.approx(np.pi)


def test_noncommuting_path_keeps_affine_action_and_supplied_metaplectic_winding():
    evidence = validation.noncommuting_lift_checks()
    assert evidence["noncommuting_path_complex_relative_error"] < 5e-11
    assert evidence["two_axis_half_cycle_phase_rad"] == pytest.approx(-np.pi)


@pytest.mark.parametrize("driven", [False, True])
def test_replay_composes_hamiltonian_lift_across_multiple_focuses(driven):
    """Endpoint winding addition previously flips this field's whole sign.

    Compare the batched path with separate fully lifted physical operations,
    including noncommuting astigmatic axes and their affine scalar actions.
    """
    from temsim.physics.canonical_action import CanonicalPath
    from temsim.physics.column_wave import _ColumnPath, _linear_factor

    physical_steps = (
        ([[8593076.09185783, -1175669.411661549],
          [-1175669.4116615485, 8053848.015797882]], .0018689993256033165),
        ([[2887292.130962625, 241106.76921570717],
          [241106.7692157071, 3424347.6583390087]], .0011894062485220498),
    )
    source = validation.packet(centre=(1e-7, -2e-7), real=[[10., 20.], [20., -50.]])
    sequential = source
    combined = CanonicalPath(1e-3)
    for index, (stiffness, distance) in enumerate(physical_steps):
        generator = np.block([[np.zeros((2, 2)), np.eye(2)],
                              [-np.asarray(stiffness), np.zeros((2, 2))]])
        force = np.array([0., 0., .001, -.002])*(index+1) if driven else np.zeros(4)
        step = _ColumnPath(1e-3, generator, force, distance)
        _linear_factor(step, generator, distance, force=force)
        sequential = beam.propagate_affine(sequential, step.matrix, step.offset,
                                          **step.phase_kwargs())
        beam.append_physical_column_step(combined, step)
    actual = beam.propagate_affine(source, combined.matrix, combined.offset,
                                  **combined.phase_kwargs())
    coordinates = validation.points((51, 43), np.eye(2)*1e-8, actual.centres_m[0])
    expected = sequential.evaluate(coordinates)
    assert np.linalg.norm(actual.evaluate(coordinates)-expected)/np.linalg.norm(expected) < 2e-11
    np.testing.assert_allclose(actual.coefficients, sequential.coefficients, rtol=2e-11, atol=2e-13)


def test_replay_rejects_unimplemented_posed_path_action():
    from types import SimpleNamespace
    from temsim.physics.canonical_action import CanonicalPath
    with pytest.raises(ValueError, match="posed"):
        beam.append_physical_column_step(CanonicalPath(1e-3), SimpleNamespace(correction=object()))


def test_cubic_quartic_screen_error_is_bounded_and_converges_with_packet_width():
    evidence = validation.screen_checks()
    errors, bounds = evidence["screen_complex_l2_errors"], evidence["screen_l2_upper_bounds"]
    assert all(error <= bound for error, bound in zip(errors, bounds))
    assert errors[1] < errors[0]/6 and errors[2] < errors[1]/6


def test_local_box_readout_keeps_complex_field_within_declared_tail_bound():
    evidence = validation.readout_checks()
    assert evidence["affine_readout_max_cell_error"] <= evidence["affine_readout_bound"]


def random_packets(count=17):
    rng = np.random.default_rng(583)
    centres = rng.normal(size=(count, 2))*2e-7
    momenta = rng.normal(size=(count, 2))*2e-6
    real = rng.normal(size=(count, 2, 2))*100
    real = (real+real.swapaxes(1, 2))/2
    root = rng.normal(size=(count, 2, 2))
    imaginary = root@root.swapaxes(1, 2)*10+np.eye(2)*10
    weights = (rng.normal(size=count)+1j*rng.normal(size=count))/count
    return beam.GaussianBeamlets(centres, momenta, real+1j*imaginary, weights, 2e-12)


@pytest.mark.parametrize("reset", [1, 32, 64])
@pytest.mark.parametrize("basis", [np.array([[2e-8, .4e-8], [-.2e-8, 2.1e-8]]),
                                   np.array([[-2.2e-8, .1e-8], [.2e-8, 2e-8]])])
def test_affine_recurrence_preserves_full_coherent_complex_sum(reset, basis):
    packets = random_packets()
    origin = np.array([2e-8, -1e-8])
    expected = packets.sample_plane((67, 95), basis, origin)
    actual, evidence = packets.sample_plane((67, 95), basis, origin, method="recurrence",
                                            recurrence_reset=reset, return_record=True)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 3e-12
    assert actual.dtype == np.complex128
    assert evidence["evaluation_method"] == "recurrence"
    assert evidence["recurrence_reset_pixels"] == reset
    # No fitted global phase, intensity normalization or norm correction.


def test_recurrence_recovers_finite_peak_after_underflowed_anisotropic_box_corner():
    angle = .71
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    precision = 1j*rotation@np.diag([1e7, 10.])@rotation.T
    packets = beam.GaussianBeamlets(np.zeros((1, 2)), np.zeros((1, 2)), precision[None], np.ones(1), 2e-12)
    x, y = np.linspace(-1e-6, 1e-6, 95), np.linspace(-.8e-6, .8e-6, 67)
    expected, _ = packets.evaluate_grid(x, y)
    actual, _ = packets.evaluate_grid(x, y, method="recurrence")
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8)


def test_packets_are_coherent_amplitudes_and_not_independent_intensities():
    source = validation.packet()
    packets = beam.GaussianBeamlets(np.repeat(source.centres_m, 2, axis=0),
        np.repeat(source.momenta, 2, axis=0), np.repeat(source.precision_m1, 2, axis=0),
        np.array([1.+0j, -1.+0j]), source.wavelength_m)
    actual = packets.sample_plane((65, 65), np.eye(2)*2e-8, np.zeros(2), method="recurrence")
    np.testing.assert_array_equal(actual, np.zeros_like(actual))
    changed, evidence = beam.apply_taylor_screen(packets, normal_m2=2e6)
    np.testing.assert_allclose(changed.coefficients[0], -changed.coefficients[1])
    assert evidence["probability_renormalized"] is False


def test_recursive_moment_storage_is_released_without_cyclic_gc(monkeypatch):
    """Observe actual covariance lifetime, not allocator-dependent process RSS."""
    source = random_packets(32)
    original_inverse = beam._inverse2
    references = []
    class ObservedCovarianceInput(np.ndarray):
        def __truediv__(self, other):
            result = np.asarray(self)/other
            references.append(weakref.ref(result))
            return result
    def observed_inverse(value):
        return original_inverse(value).view(ObservedCovarianceInput)
    expected = beam._residual_rms_action(source, 2e6, -1e6, 2e12)
    monkeypatch.setattr(beam, "_inverse2", observed_inverse)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(5):
            actual = beam._residual_rms_action(source, 2e6, -1e6, 2e12)
            np.testing.assert_array_equal(actual, expected)
            assert references and all(reference() is None for reference in references)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("method", ["direct", "recurrence"])
def test_readout_budget_and_cancellation_precede_output_allocation(monkeypatch, method):
    source = validation.packet()
    before = source.coefficients.copy()
    monkeypatch.setattr(beam.np, "zeros", lambda *a, **k: pytest.fail("Output allocated before resource check"))
    axis = np.linspace(-1e-6, 1e-6, 32)
    with pytest.raises(MemoryError):
        source.evaluate_grid(axis, axis, maximum_working_bytes=1, method=method)
    with pytest.raises(InterruptedError):
        source.evaluate_grid(axis, axis, cancelled=lambda: True, method=method)
    np.testing.assert_array_equal(source.coefficients, before)


@pytest.mark.parametrize("kwargs", [{"method": "unknown"}, {"recurrence_reset": 0},
                                    {"recurrence_reset": 257}, {"amplitude_tail": 0.}])
def test_readout_rejects_invalid_numerical_policy(kwargs):
    axis = np.linspace(-1e-6, 1e-6, 32)
    with pytest.raises(ValueError):
        validation.packet().evaluate_grid(axis, axis, **kwargs)


def _stage_plan(z_mm, reference_momentum=1.):
    """Only the fields consumed by the aligned diagnostic stage builder."""
    from dataclasses import make_dataclass
    z = np.asarray(z_mm, float)
    n = len(z)-1
    values = dict(z_mm=z, step_m=np.diff(z)*1e-3,
        mapped_fields=(), posed_spherical_kicks=(), intrinsic_aberration_kicks=(),
        electric_field=None, reference_momentum_kg_m_s=reference_momentum)
    for key in ("midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
                "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        values[key] = np.zeros(n)
    for key in ("cs_kick_m3", "kick_x_rad", "kick_y_rad"):
        values[key] = np.zeros(n+1)
    for key in ("dipole_bx_t", "dipole_by_t"):
        values[key] = np.zeros(3*n)
    return make_dataclass("DiagnosticStagePlan", [(key, object) for key in values])(**values)


def test_fourth_order_stages_sample_actual_midpoints_and_keep_events_once(monkeypatch):
    from scripts import gaussian_column_stages as stages
    plan = _stage_plan([10., 12., 15.])
    plan.cs_kick_m3[:] = [11., 12., 13.]
    plan.kick_x_rad[:] = [21., 22., 23.]
    plan.kick_y_rad[:] = [31., 32., 33.]
    plan.dipole_bx_t[:] = np.repeat([.1, .2], 3)
    plan.dipole_by_t[:] = np.repeat([.3, .4], 3)
    sampled = []
    def physical_fields(z, state):
        sampled.append(z.copy())
        return z**2, np.sin(z), np.cos(z)
    monkeypatch.setattr(stages, "fields", physical_fields)
    monkeypatch.setattr(stages, "skew_quadrupole_field", lambda z, state: z**3)
    monkeypatch.setattr(stages, "hexapole_field_components", lambda z, state: (np.exp(z/100), -z**4))
    actual, electric = stages.fourth_order_column_stages(object(), plan)
    midpoint = .5*(actual.z_mm[:-1]+actual.z_mm[1:])
    np.testing.assert_allclose(sampled[0], midpoint, rtol=0., atol=5e-15)
    np.testing.assert_allclose(actual.midpoint_magnetic_t, midpoint**2)
    np.testing.assert_allclose(actual.midpoint_hex_normal_m3, np.exp(midpoint/100))
    np.testing.assert_allclose(actual.midpoint_hex_skew_m3, -midpoint**4)
    np.testing.assert_allclose(actual.step_m, np.diff(actual.z_mm)*1e-3, rtol=3e-15)
    assert np.all(actual.step_m[1::3] < 0)
    np.testing.assert_allclose(actual.z_mm[::3], plan.z_mm, rtol=0., atol=0.)
    # Real hardware events remain at the real macro boundaries, never the
    # virtual intermediate backtracking locations.
    for name in ("cs_kick_m3", "kick_x_rad", "kick_y_rad"):
        np.testing.assert_array_equal(getattr(actual, name)[::3], getattr(plan, name))
        np.testing.assert_array_equal(getattr(actual, name)[1::3], np.zeros(2))
        np.testing.assert_array_equal(getattr(actual, name)[2::3], np.zeros(2))
    np.testing.assert_array_equal(actual.dipole_bx_t[1::3], np.repeat([.1, .2], 3))
    np.testing.assert_array_equal(actual.dipole_by_t[1::3], np.repeat([.3, .4], 3))
    assert electric[0].shape == (7,)
    np.testing.assert_array_equal(plan.cs_kick_m3, [11., 12., 13.])


def test_signed_column_stages_keep_one_entrance_momentum_reference(monkeypatch):
    from scripts import gaussian_column_stages as stages
    from temsim.physics.column_wave import _column_transports
    from temsim.physics.tip_gun_wave import _momentum_velocity
    energy_kev = 300.
    p0, _ = _momentum_velocity(energy_kev*1000.)
    plan = _stage_plan([10., 12., 15.], float(p0))
    monkeypatch.setattr(stages, "fields", lambda z, state: (np.zeros_like(z),)*3)
    monkeypatch.setattr(stages, "skew_quadrupole_field", lambda z, state: np.zeros_like(z))
    monkeypatch.setattr(stages, "hexapole_field_components", lambda z, state: (np.zeros_like(z),)*2)
    def electric_coefficients(value):
        z = value.z_mm
        return (100.+10.*(z-z[0]), 100.+10.*(.5*(z[:-1]+z[1:])-z[0]),
                np.zeros((len(z)-1, 2)), np.zeros((len(z)-1, 2, 2)))
    monkeypatch.setattr(stages, "_electric_coefficients", electric_coefficients)
    actual, electric = stages.fourth_order_column_stages(object(), plan)
    paths = tuple(_column_transports(actual, energy_kev, electric=electric))
    momentum, _ = _momentum_velocity(energy_kev*1000.+electric[1]-electric[0][0])
    for path, p, distance in zip(paths, momentum, actual.step_m):
        np.testing.assert_allclose(path.generator[:2, 2:], np.eye(2)*p0/p, rtol=2e-14, atol=0.)
        np.testing.assert_allclose(path.matrix[:2, 2:], np.eye(2)*distance*p0/p, rtol=2e-14, atol=0.)
    assert electric[0][0] == 100.
    assert np.ptp(momentum) > 0  # A reset-per-stage reference would hide this.


def test_nonautonomous_fourth_order_stage_fields_match_independent_gaussian_ode(monkeypatch):
    """Order of the actual stage builder and column maps, including phase.

    This dimensioned mathematical fixture is not a physical source model.
    The independent ODE follows q,p,Riccati precision and central/Gouy phase.
    """
    from scipy.constants import e
    from scipy.integrate import solve_ivp
    from scripts import gaussian_column_stages as stages
    from temsim.physics.column_wave import _column_transports, _electric_coefficients
    from temsim.physics.tip_gun_wave import _momentum_velocity
    p0 = float(_momentum_velocity(300000.)[0])
    wavelength = .03
    k = 2*np.pi/wavelength
    j = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
    q0, momentum0 = np.array([.17, -.13]), np.array([.11, .07])
    c0 = np.array([[.3+1.5j, .08+.04j], [.08+.04j, -.2+1.2j]])
    force = np.array([0., 0., .08, .06])
    def coefficients(z):
        g = .23*np.sin(.9*z)
        rotation = np.array([[0., g], [-g, 0.]])
        stiffness = np.array([[.8+.2*np.cos(1.3*z), .13*np.sin(.7*z)],
                              [.13*np.sin(.7*z), .6+.1*np.cos(1.7*z)]])
        generator = np.block([[rotation, np.eye(2)], [-stiffness, rotation]])
        return generator, .7+.4*np.sin(1.4*z), .2*np.cos(1.1*z)
    def physical_fields(z_mm, state):
        z = z_mm*1e-3
        g = .23*np.sin(.9*z)
        return -2*p0*g/e, .8+.2*np.cos(1.3*z)-g*g, .6+.1*np.cos(1.7*z)-g*g
    monkeypatch.setattr(stages, "fields", physical_fields)
    monkeypatch.setattr(stages, "skew_quadrupole_field", lambda z, state: .13*np.sin(.7*z*1e-3))
    monkeypatch.setattr(stages, "hexapole_field_components", lambda z, state: (
        .7+.4*np.sin(1.4*z*1e-3), .2*np.cos(1.1*z*1e-3)))
    def rhs(z, state):
        q, p = state[:2].real, state[2:4].real
        c = state[4:8].reshape(2, 2)
        generator, hn, hs = coefficients(z)
        hessian, linear = -j@generator, -j@force
        x, y = q
        potential = hn*(x**3-3*x*y*y)/3+hs*(3*x*x*y-y**3)/3
        gradient = np.array([hn*(x*x-y*y)+2*hs*x*y, -2*hn*x*y+hs*(x*x-y*y)])
        nonlinear_hessian = np.array([[2*hn*x+2*hs*y, -2*hn*y+2*hs*x],
                                      [-2*hn*y+2*hs*x, -2*hn*x-2*hs*y]])
        r = np.r_[q, p]
        hamiltonian = .5*r@hessian@r+linear@r+potential
        derivative = generator@r+force
        derivative[2:] -= gradient
        c_dot = -(hessian[:2, :2]+nonlinear_hessian)-hessian[:2, 2:]@c-c@hessian[2:, :2]-c@hessian[2:, 2:]@c
        phase_dot = k*(p@derivative[:2]-hamiltonian)-.5*np.trace(hessian[2:, 2:]@c.imag)
        return np.r_[derivative, c_dot.ravel(), phase_dot]
    solution = solve_ivp(rhs, (0., 1.), np.r_[q0, momentum0, c0.ravel(), 0.],
                         method="DOP853", rtol=2.3e-14, atol=2.3e-14)
    assert solution.success
    final = solution.y[:, -1]
    reference = beam.GaussianBeamlets(final[:2].real[None], final[2:4].real[None],
        final[4:8].reshape(1, 2, 2), np.exp(1j*final[8].real)[None], wavelength)
    source = beam.GaussianBeamlets(q0[None], momentum0[None], c0[None], np.ones(1), wavelength)
    coordinates = validation.points((51, 49), np.eye(2)*.02, np.zeros(2))
    expected = reference.evaluate(coordinates)
    def integrate(n, fourth, reverse=False, packet=source):
        plan = _stage_plan(np.linspace(1000. if reverse else 0., 0. if reverse else 1000., n+1), p0)
        plan.dipole_bx_t[:] = -force[3]*p0/e
        plan.dipole_by_t[:] = force[2]*p0/e
        if fourth:
            plan, electric = stages.fourth_order_column_stages(object(), plan)
        else:
            z = .5*(plan.z_mm[:-1]+plan.z_mm[1:])
            plan.midpoint_magnetic_t, plan.midpoint_sx_m2, plan.midpoint_sy_m2 = physical_fields(z, None)
            plan.midpoint_sxy_m2 = stages.skew_quadrupole_field(z, None)
            plan.midpoint_hex_normal_m3, plan.midpoint_hex_skew_m3 = stages.hexapole_field_components(z, None)
            electric = _electric_coefficients(plan)
        for index, path in enumerate(_column_transports(plan, 300., electric=electric)):
            hn = plan.midpoint_hex_normal_m3[index]*plan.step_m[index]/2
            hs = plan.midpoint_hex_skew_m3[index]*plan.step_m[index]/2
            packet, _ = beam.apply_taylor_screen(packet, normal_m2=hn, skew_m2=hs)
            packet = beam.propagate_affine(packet, path.matrix, path.offset, **path.phase_kwargs())
            packet, _ = beam.apply_taylor_screen(packet, normal_m2=hn, skew_m2=hs)
        return packet
    for fourth, lower_order, upper_order in ((False, 1.9, 2.1), (True, 3.8, 4.2)):
        errors = [np.linalg.norm(integrate(n, fourth).evaluate(coordinates)-expected)/np.linalg.norm(expected)
                  for n in (8, 16, 32)]
        orders = np.log2(np.asarray(errors[:-1])/errors[1:])
        assert np.all(orders > lower_order), (errors, orders)
        assert np.all(orders < upper_order), (errors, orders)
    reverse = integrate(16, True, reverse=True, packet=integrate(16, True))
    initial_field = source.evaluate(coordinates)
    assert np.linalg.norm(reverse.evaluate(coordinates)-initial_field)/np.linalg.norm(initial_field) < 2e-11


def _prepared_packet_input():
    """Metadata only: no real source, stored field, or allocation is needed."""
    from temsim.immutable_json import json_digest
    state = {"schema": 80, "nodes": [{"key": "C3", "percent": 9.2079848}]}
    manifest = dict(key="executed-case-key", dependency="actual-upstream-identity",
        record=dict(case="baseline", state=state), reference_plane="physical",
        plane_z_mm=1083., modes=[dict(arrays=dict(amplitude=dict(shape=[48, 80])))])
    manifest["manifest_digest"] = json_digest(manifest)
    prepared = dict(status="computed", case="baseline", storage={
        key: manifest[key] for key in ("key", "dependency", "manifest_digest")})
    metadata = dict(case="baseline", state_sha256=json_digest(state),
        upstream450_to_c2_changes=[], auto_refocus=False)
    return manifest, prepared, state, metadata


def test_prepared_packet_input_binds_executed_case_and_declared_grid():
    from scripts.replay_gaussian_corrector import validate_prepared_input
    inputs = _prepared_packet_input()
    # The original rectangular grid is never silently downsampled for a smaller
    # packet request. Equality at the declared allocation bound is permitted.
    expected = (64, 80)
    required = 2048*64*80
    assert validate_prepared_input(*inputs, 0, 64, required) == expected
    with pytest.raises(MemoryError, match="working-memory budget"):
        validate_prepared_input(*inputs, 0, 64, required-1)


@pytest.mark.parametrize("fault", ["state", "manifest", "outcome_digest", "outcome_key",
    "outcome_dependency", "record_case", "metadata_case", "metadata_state", "missing_mode",
    "invalid_shape", "large_grid", "budget", "failed_outcome", "changed_upstream", "refocus"])
def test_replay_rejects_swapped_or_over_budget_input_before_loading_arrays(tmp_path, monkeypatch, fault):
    import copy
    import json
    from scripts import replay_gaussian_corrector as replay
    from temsim.immutable_json import json_digest
    manifest, prepared, state, metadata = copy.deepcopy(_prepared_packet_input())
    mode_index, memory_mib = 0, 32
    if fault == "state":
        # A mutually consistent state.json and metadata file still cannot be
        # substituted for the instrument captured in the executed manifest.
        state = copy.deepcopy(state)
        state["nodes"][0]["percent"] += .1
        metadata["state_sha256"] = json_digest(state)
    elif fault == "manifest":
        manifest["record"]["case"] = "c3_plus"  # stale checksum
    elif fault.startswith("outcome_"):
        key = dict(outcome_digest="manifest_digest", outcome_key="key",
                   outcome_dependency="dependency")[fault]
        prepared["storage"][key] = "another-executed-checkpoint"
    elif fault == "record_case":
        # Re-signing a real alternate manifest cannot make its case identity
        # agree with an unrelated prepared outcome.
        manifest["record"]["case"] = "c3_plus"
    elif fault == "metadata_case":
        metadata["case"] = "c3_plus"
    elif fault == "metadata_state":
        metadata["state_sha256"] = "another-state-digest"
    elif fault == "missing_mode":
        mode_index = 1
    elif fault == "invalid_shape":
        manifest["modes"][0]["arrays"]["amplitude"]["shape"] = [48, True]
    elif fault == "large_grid":
        manifest["modes"][0]["arrays"]["amplitude"]["shape"] = [48, 1025]
    elif fault == "budget":
        memory_mib = 1
    elif fault == "failed_outcome":
        prepared["status"] = "failed"
    elif fault == "changed_upstream":
        metadata["upstream450_to_c2_changes"] = ["C2.percent"]
    elif fault == "refocus":
        metadata["auto_refocus"] = True
    if fault in ("record_case", "invalid_shape", "large_grid"):
        manifest["manifest_digest"] = json_digest({key: value for key, value in manifest.items()
                                                   if key != "manifest_digest"})
        prepared["storage"]["manifest_digest"] = manifest["manifest_digest"]
    case_directory = tmp_path/"case"
    stored_directory = tmp_path/"stored"
    case_directory.mkdir()
    stored_directory.mkdir()
    prepared["storage"]["directory"] = str(stored_directory)
    for path, data in ((case_directory/"state.json", state),
                       (case_directory/"metadata.json", metadata),
                       (case_directory/"outcome-all.json", prepared),
                       (stored_directory/"manifest.json", manifest)):
        path.write_text(json.dumps(data), encoding="utf-8")
    def arrays_forbidden(*args, **kwargs):
        pytest.fail("Instrument/field arrays were loaded before rejecting invalid input")
    monkeypatch.setattr(replay, "decode_instrument", arrays_forbidden)
    monkeypatch.setattr(replay, "_StoredModes", arrays_forbidden)
    monkeypatch.setattr(replay, "refine_plane_wave", arrays_forbidden)
    output = tmp_path/"output"
    monkeypatch.setattr(replay.sys, "argv", ["replay", "--case-directory", str(case_directory),
        "--output", str(output), "--pixels", "64", "--mode-index", str(mode_index),
        "--maximum-working-mib", str(memory_mib)])
    exception = MemoryError if fault in ("large_grid", "budget") else ValueError
    with pytest.raises(exception):
        replay.main()
    assert not output.exists()


def test_analytic_two_by_two_containment_eigenvalue_matches_batched_reference():
    from scripts.replay_gaussian_corrector import largest_symmetric_eigenvalue2
    rng = np.random.default_rng(312)
    rotations, _ = np.linalg.qr(rng.normal(size=(3, 7, 2, 2)))
    eigenvalues = 10.**rng.uniform(-12, 12, size=(3, 7, 2))
    matrices = (rotations*eigenvalues[..., None, :])@rotations.swapaxes(-1, -2)
    matrices = np.concatenate((matrices.reshape(-1, 2, 2), np.array([
        [[0., 0.], [0., 0.]], [[1e-100, 0.], [0., 2e-100]],
        [[2e100, -1e100], [-1e100, 2e100]], [[-3., 1.], [1., -2.]]])))
    expected = np.linalg.eigvalsh(matrices)[..., -1]
    np.testing.assert_allclose(largest_symmetric_eigenvalue2(matrices), expected, rtol=8e-15, atol=0.)
    # Matching eigvalsh's lower-triangle convention avoids a hidden dependency
    # on roundoff differences between independently accumulated covariance axes.
    matrices[:, 0, 1] += 99.
    np.testing.assert_allclose(largest_symmetric_eigenvalue2(matrices), expected, rtol=8e-15, atol=0.)
    assert largest_symmetric_eigenvalue2(np.empty((0, 2, 2))).shape == (0,)


def test_replay_checks_actual_mmap_shape_before_planewave_copy(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import replay_gaussian_corrector as replay
    from temsim.immutable_json import json_digest
    from temsim.physics import wave_checkpoint_store as store
    manifest, prepared, state, metadata = _prepared_packet_input()
    case_directory, stored_directory = tmp_path/"case", tmp_path/"stored"
    case_directory.mkdir()
    stored_directory.mkdir()
    amplitude_path = stored_directory/"amplitude.npy"
    # An honestly hashed payload can still disagree with a malformed manifest's
    # smaller dimensions. Keep this adversarial fixture itself small.
    np.save(amplitude_path, np.ones((96, 80), dtype=np.complex128))
    amplitude = manifest["modes"][0]["arrays"]["amplitude"]
    amplitude.update(file=amplitude_path.name,
        sha256=hashlib.sha256(amplitude_path.read_bytes()).hexdigest())
    manifest["manifest_digest"] = json_digest({key: value for key, value in manifest.items()
                                               if key != "manifest_digest"})
    prepared["storage"].update(directory=str(stored_directory),
                               manifest_digest=manifest["manifest_digest"])
    for path, data in ((case_directory/"state.json", state),
                       (case_directory/"metadata.json", metadata),
                       (case_directory/"outcome-all.json", prepared),
                       (stored_directory/"manifest.json", manifest)):
        path.write_text(json.dumps(data), encoding="utf-8")
    reads = []
    original_load = store.np.load
    def observed_load(*args, **kwargs):
        reads.append(kwargs.get("mmap_mode"))
        return original_load(*args, **kwargs)
    def copies_forbidden(*args, **kwargs):
        pytest.fail("Instrument or immutable PlaneWave loaded before the NPY header was admitted")
    monkeypatch.setattr(store.np, "load", observed_load)
    monkeypatch.setattr(store, "PlaneWave", copies_forbidden)
    monkeypatch.setattr(replay, "decode_instrument", copies_forbidden)
    monkeypatch.setattr(replay.sys, "argv", ["replay", "--case-directory", str(case_directory),
        "--output", str(tmp_path/"output"), "--pixels", "64", "--maximum-working-mib", "32"])
    with pytest.raises(ValueError, match="Stored amplitude shape"):
        replay.main()
    assert reads == ["r"]
    assert not (tmp_path/"output").exists()


def test_stored_amplitude_preflight_preserves_hash_checks_and_accepts_actual_shape(tmp_path):
    import hashlib
    from scripts.replay_gaussian_corrector import preflight_stored_amplitude
    from temsim.physics.wave_checkpoint_store import _StoredModes
    path = tmp_path/"amplitude.npy"
    np.save(path, np.ones((31, 42), dtype=np.complex128))
    item = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=[31, 42])
    row = dict(arrays=dict(amplitude=item))
    reader = _StoredModes(tmp_path, (row,), "physical")
    preflight_stored_amplitude(reader, row)
    item["sha256"] = "incorrect-payload-checksum"
    with pytest.raises(ValueError, match="checksum/path"):
        preflight_stored_amplitude(reader, row)


def test_importing_replay_does_not_change_cuda_or_thread_policy():
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    # A fresh interpreter observes the import itself; reusing pytest's already
    # cached module could hide an accidental top-level environment mutation.
    # temsim has its own established startup BLAS limit. Initialize that policy
    # first, then ensure importing this diagnostic adds no CLI-only mutation.
    code = """
import importlib, json, os, sys
import temsim
keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
        'NUMBA_NUM_THREADS', 'NUMBA_DISABLE_CUDA')
before = {key: os.environ.get(key) for key in keys}
sentinel = object()
cupy_before = sys.modules.get('cupy', sentinel)
importlib.import_module('scripts.replay_gaussian_corrector')
assert {key: os.environ.get(key) for key in keys} == before
assert sys.modules.get('cupy', sentinel) is cupy_before
print(json.dumps({'preserved': True}))
"""
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[name] = "2"
    env["NUMBA_DISABLE_CUDA"] = "0"
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["preserved"] is True


def _plot_numerical_receipt():
    return dict(axial_order=4, axial_method="Yoshida", maximum_column_step_mm=.5,
        quadratic_path_integration=dict(method="nested quadratic", maximum_quadratic_step_mm=.0625),
        method="Gaussian complex collocation", shape=[32, 32], packet_count=1024,
        sigma_amplitude_m=1e-6, sigma_intensity_m=1e-6/np.sqrt(2), sigma_cells=.8,
        implementation_sha256={name: "unchanged-"+name for name in
            ("experimental_gaussian_beamlets.py", "gaussian_column_stages.py",
             "gaussian_quadratic_paths.py", "replay_gaussian_corrector.py")},
        current_production_source_sha256="unchanged-source")


@pytest.mark.parametrize("dimension", ["width", "axial", "quadratic"])
def test_phase_comparison_refines_only_its_declared_numerical_dimension(monkeypatch, dimension):
    import copy
    from scripts import plot_gaussian_phase_comparison as plotting
    monkeypatch.setattr(plotting, "decomposition_sigma_cells", lambda receipt: receipt["sigma_cells"])
    coarse = _plot_numerical_receipt()
    fine = copy.deepcopy(coarse)
    if dimension == "width":
        fine.update(shape=[64, 64], packet_count=4096,
                    sigma_amplitude_m=.5e-6, sigma_intensity_m=.5e-6/np.sqrt(2))
    elif dimension == "axial":
        fine["maximum_column_step_mm"] = .25
    else:
        fine["quadratic_path_integration"]["maximum_quadratic_step_mm"] = .03125
    evidence = plotting.require_shared_numerics([coarse, fine], dimension)
    assert evidence["refinement_dimension"] == dimension
    with pytest.raises(ValueError, match="Uncontrolled numerical change"):
        plotting.require_shared_numerics([coarse, fine])
    fine["axial_order"] = 2
    with pytest.raises(ValueError, match="axial_order"):
        plotting.require_shared_numerics([coarse, fine], dimension)


def test_lens_case_comparison_preserves_dimensionless_width_on_changed_physical_lattice(monkeypatch):
    import copy
    from scripts import plot_gaussian_phase_comparison as plotting
    monkeypatch.setattr(plotting, "decomposition_sigma_cells", lambda receipt: receipt["sigma_cells"])
    baseline = _plot_numerical_receipt()
    changed_lens = copy.deepcopy(baseline)
    changed_lens["sigma_amplitude_m"] *= 1.01
    changed_lens["sigma_intensity_m"] *= 1.01
    plotting.require_shared_numerics([baseline, changed_lens])
    changed_lens["sigma_cells"] = .81
    with pytest.raises(ValueError, match="sigma_cells"):
        plotting.require_shared_numerics([baseline, changed_lens])


def test_phase_comparison_requires_exact_evidence_bound_nonnumerical_version_audit(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from scripts import plot_gaussian_phase_comparison as plotting
    monkeypatch.setattr(plotting, "decomposition_sigma_cells", lambda receipt: receipt["sigma_cells"])
    baseline = _plot_numerical_receipt()
    changed = copy.deepcopy(baseline)
    changed["implementation_sha256"]["replay_gaussian_corrector.py"] = "new-admission-only-version"
    with pytest.raises(ValueError, match="Unreviewed version difference"):
        plotting.require_shared_numerics([baseline, changed])
    proof = tmp_path/"proof.json"
    proof.write_text('{"numerical_operations_unchanged": true}', encoding="utf-8")
    audit = tmp_path/"audit.json"
    audit.write_text(json.dumps(dict(schema="experimental-nonnumerical-version-audit-v1", changes=[dict(
        category="replay-script", old_sha256=baseline["implementation_sha256"]["replay_gaussian_corrector.py"],
        new_sha256=changed["implementation_sha256"]["replay_gaussian_corrector.py"],
        numerics_unchanged=True, reason="Header admission only", evidence=str(proof),
        evidence_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())])), encoding="utf-8")
    evidence = plotting.require_shared_numerics([baseline, changed], version_audit=audit)
    assert len(evidence["approved_nonnumerical_differences"]) == 1
    changed["implementation_sha256"]["gaussian_quadratic_paths.py"] = "different-mathematics"
    with pytest.raises(ValueError, match="Hybrid numerical implementation changed"):
        plotting.require_shared_numerics([baseline, changed], version_audit=audit)
    changed["implementation_sha256"]["gaussian_quadratic_paths.py"] = baseline["implementation_sha256"]["gaussian_quadratic_paths.py"]
    proof.write_text('{"unreviewed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="audit evidence changed"):
        plotting.require_shared_numerics([baseline, changed], version_audit=audit)
