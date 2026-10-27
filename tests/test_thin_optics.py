"""Thin-model analytical limits and column integration, not instrument calibration."""
from dataclasses import replace
from copy import deepcopy

import numpy as np
import pytest
from scipy.integrate import quad

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.instrument_magnetic import active_column_events
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.thin_optics import ThinOpticalAction, gaussian_integrals, validate_thin_plan


VARIANTS = ("C2", "C3", "C3_ImageCorrector", "C3_ProbeCorrector",
            "C3_ProbeCorrector_ImageCorrector", "C3_BCOR", "C3_ProbeCorrector_BCOR")


def _state(variant=None):
    state = default_state()
    if variant:
        catalog = AssemblyCatalog()
        column = next(c for c in catalog.columns if c.file == f"column/{variant}.toml")
        catalog.apply(state, AssemblySelection("FEG", column.name, "No Energy Filter"))
    state.simulation_mode = "ideal_thin"
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.history_step_mm = 20.
    return state


def test_gaussian_integrals_match_independent_quadrature_with_overlap():
    terms = ((.2, .003, .002), (-.07, -.001, .001), (.03, .004, .0003))
    def b(z):
        return sum(a*np.exp(-.5*((z-c)/s)**2) for a, c, s in terms)
    actual = gaussian_integrals(terms)
    expected = tuple(quad(lambda z: b(z)**power, -.03, .03, epsabs=1e-15)[0] for power in (1, 2))
    np.testing.assert_allclose(actual, expected, rtol=2e-12)


def test_round_action_matches_focusing_rotation_and_relativistic_energy_scaling():
    action = ThinOpticalAction("lens", "lens", .1, CoordinateRegistration(), 40., .7)
    initial = np.array([[1e-6, 3e-4, -2e-6, 4e-4], [4e-6, -2e-4, 1e-6, -3e-4]])
    scales = np.array([1., .8])
    actual = np.column_stack(action.apply(*initial.T, scales))
    expected = initial.copy()
    expected[:, 1] -= 40*scales**2*initial[:, 0]
    expected[:, 3] -= 40*scales**2*initial[:, 2]
    for i, scale in enumerate(scales):
        theta = .7*scale
        matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        expected[i, [0, 2]] = matrix @ expected[i, [0, 2]]
        expected[i, [1, 3]] = matrix @ expected[i, [1, 3]]
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-20)


def test_quadrupole_and_hexapole_are_symmetric_gradient_actions():
    action = ThinOpticalAction("multi", "multi", .2, CoordinateRegistration(),
        polynomial=((2, 0, -3.), (0, 2, 3.), (1, 1, .5), (3, 0, -2e4/3), (1, 2, 2e4)))
    x, y = np.array([2e-5, -1e-5]), np.array([1e-5, 3e-5])
    zero = np.zeros(2)
    actual = action.apply(x, zero, y, zero, np.ones(2))
    np.testing.assert_allclose(actual[1], -6*x+.5*y-2e4*(x*x-y*y), rtol=1e-14)
    np.testing.assert_allclose(actual[3], 6*y+.5*x+4e4*x*y, rtol=1e-14)
    # Hamiltonian impulse is symplectic: cross derivatives must be equal.
    h = 1e-8
    dtydx = (action.apply(x+h, zero, y, zero, np.ones(2))[3]
              -action.apply(x-h, zero, y, zero, np.ones(2))[3])/(2*h)
    dtxdy = (action.apply(x, zero, y+h, zero, np.ones(2))[1]
              -action.apply(x, zero, y-h, zero, np.ones(2))[1])/(2*h)
    np.testing.assert_allclose(dtydx, dtxdy, rtol=1e-10)


@pytest.mark.parametrize("variant", VARIANTS)
def test_seven_columns_capture_once_and_preserve_settings(variant):
    state = _state(variant)
    before = deepcopy(state.to_dict())
    events = active_column_events(state)
    plan = build_propagation_plan(state, 600., state.sample.z_mm, events,
        checkpoint_z_mm=(state.sample.z_mm,))
    validate_thin_plan(plan)
    keys = [a.key for a in plan.thin_actions]
    assert len(keys) == len(set(keys))
    assert plan.physics_contract["mode"] == "ideal_thin"
    assert plan.mapped_fields == plan.intrinsic_aberration_kicks == plan.posed_spherical_kicks == ()
    assert state.to_dict() == before
    assert not any("Descan" in key for key in keys)  # Only actual upstream hosts are present.
    result = execute_propagation_plan(state, plan, np.array([0., 1e-9]), np.zeros(2),
                                      np.zeros(2), np.zeros(2))
    assert np.isfinite(result[-1].x_m).all()


def test_step_size_does_not_create_rk_steps_or_change_thin_output():
    state = _state("C3_ProbeCorrector")
    state.step_mm = .001
    a = build_propagation_plan(state, 600., 1800., active_column_events(state), checkpoint_z_mm=(1800.,))
    state.step_mm = 5.
    b = build_propagation_plan(state, 600., 1800., active_column_events(state), checkpoint_z_mm=(1800.,))
    assert a.solver_signature == b.solver_signature
    np.testing.assert_array_equal(a.z_mm, b.z_mm)
    assert len(a.z_mm) < 200
    values = (np.array([0., 1e-8]), np.array([1e-8, -2e-8]), np.zeros(2), np.zeros(2))
    for left, right in zip(execute_propagation_plan(state, a, *values)[:5],
                           execute_propagation_plan(state, b, *values)[:5]):
        np.testing.assert_array_equal(left, right)


def test_resume_is_after_action_and_keeps_energy_and_clock():
    state = _state()
    plane, stop = state.lenses[0].z_mm, 1000.
    # Choose a known in-range action, independently of source emission.
    lower = state.electron_gun.exit_plane_z_mm
    plan = build_propagation_plan(state, lower, stop, checkpoint_z_mm=(plane, stop))
    initial = (np.array([1e-8]), np.array([0.]), np.array([0.]), np.array([0.]))
    energy = np.array([290000.])
    whole = execute_propagation_plan(state, plan, *initial, initial_kinetic_energy_ev=energy,
        initial_time_s=np.array([2e-9]), return_flight_times=True)
    cp = whole[-1]
    index = int(np.searchsorted(plan.z_mm, plane))
    resumed = execute_propagation_plan(state, plan, cp.x_m[0], cp.tx_rad[0], cp.y_m[0], cp.ty_rad[0],
        start_index=index, include_initial_plane_kicks=False, initial_kinetic_energy_ev=cp.kinetic_energy_ev[0],
        initial_time_s=cp.flight_time_s[0], return_flight_times=True)
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "kinetic_energy_ev", "flight_time_s"):
        np.testing.assert_array_equal(getattr(resumed[-1], name)[-1], getattr(cp, name)[-1])


def test_changed_lens_invalidates_identity_and_missing_events_reject():
    state = _state()
    lower = state.electron_gun.exit_plane_z_mm
    plan = build_propagation_plan(state, lower, 1000.)
    state.lenses[0].percent *= .9
    updated = build_propagation_plan(state, lower, 1000.)
    assert plan.solver_signature != updated.solver_signature
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_thin_plan(replace(plan, thin_actions=()))
    with pytest.raises(ValueError, match="additional numerical actions"):
        validate_thin_plan(replace(plan, magnetic_t=np.ones_like(plan.magnetic_t)))


def test_optional_physics_and_wave_are_explicit_not_silently_discarded():
    state = _state()
    with pytest.raises(ValueError, match="gas collisions"):
        build_propagation_plan(state, state.electron_gun.exit_plane_z_mm, 1000., particle_medium=True)
    with pytest.raises(ValueError, match="gun source"):
        build_propagation_plan(state, 0., 1000.)
    from temsim.physics.tip_wave_pipeline import simulate_tip_wave
    with pytest.raises(ValueError, match="Finite-Field"):
        simulate_tip_wave(state)


def test_thin_deflector_uses_integrated_kick_not_effective_thickness():
    state = _state("C3_ProbeCorrector")
    from temsim.physics.thin_optics import capture_thin_actions
    component = next(c for c in state.deflectors if hasattr(c, "upper_x_mrad"))
    component.upper_x_mrad = .2
    old = capture_thin_actions(state, active_column_events(state))
    component.thickness_mm *= .8
    new = capture_thin_actions(state, active_column_events(state))
    before = next(a for a in old if a.key == component.key+":0")
    after = next(a for a in new if a.key == component.key+":0")
    np.testing.assert_allclose(np.array(before.polynomial)[:, 2], np.array(after.polynomial)[:, 2], rtol=2e-14)


def test_archived_section_graph_preserves_thin_actions():
    from temsim.particle_section_io import _pack, _unpack, _data_classes
    state = _state()
    plan = build_propagation_plan(state, state.electron_gun.exit_plane_z_mm, 1000.)
    arrays = {}
    payload = _pack(plan, arrays, {}, _data_classes())
    restored = _unpack(payload, arrays, _data_classes())
    validate_thin_plan(restored)
    assert restored.thin_actions == plan.thin_actions


def test_specimen_energy_change_does_not_reintroduce_empirical_cc_in_thin_mode():
    from test_shared_chromatic_screens import fixture
    from temsim.physics.intrinsic_aberrations import specimen_chromatic_energy_change, capture_chromatic_kicks
    from types import SimpleNamespace
    state = fixture(cc=2.)
    state.sample = SimpleNamespace(z_mm=0.)
    values = (np.array([1e-6]), np.array([2e-6]), np.array([300000.]), np.array([299000.]))
    assert np.any(specimen_chromatic_energy_change(state, *values)[0])
    state.simulation_mode = "ideal_thin"
    assert capture_chromatic_kicks(state, -1., 1.) == ()
    for kick in specimen_chromatic_energy_change(state, *values):
        np.testing.assert_array_equal(kick, np.zeros(1))


def test_thin_rotation_observer_uses_executed_events_not_distributed_profiles():
    from temsim.physics.simulation import _sample_to_stop_larmor_rotation_rad
    state = _state()
    stop = state.projector_lens_p1.z_mm+10.
    plan = build_propagation_plan(state, state.sample.z_mm, stop)
    expected = sum(action.rotation_rad for action in plan.thin_actions
                   if action.z_mm > state.sample.z_mm)
    assert _sample_to_stop_larmor_rotation_rad(state, stop) == pytest.approx(expected, rel=1e-14)
    state.step_mm *= 10
    assert _sample_to_stop_larmor_rotation_rad(state, stop) == pytest.approx(expected, rel=1e-14)


def test_authoritative_mm_plane_is_not_moved_across_adjacent_observations():
    state = _state()
    plane = float(state.sample.z_mm)
    assert (plane*1e-3)*1e3 != plane  # This fixture exposes an SI round-trip ULP.
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements, *state.deflectors):
        component.enabled = component is state.objective_lens
    lower, upper = plane-1., plane+1.
    initial = (np.array([1e-8]), np.array([2e-5]), np.array([-2e-8]), np.array([-1e-5]))
    before, after = np.nextafter(plane, -np.inf), np.nextafter(plane, np.inf)
    whole_plan = build_propagation_plan(state, lower, upper, checkpoint_z_mm=(before, plane, after, upper))
    action, = whole_plan.thin_actions
    assert action.z_mm == plane and action.native_reference_z_mm == plane
    assert not build_propagation_plan(state, lower, before).thin_actions
    assert build_propagation_plan(state, lower, plane).thin_actions == (action,)
    assert not build_propagation_plan(state, after, upper).thin_actions
    whole = execute_propagation_plan(state, whole_plan, *initial)[-1]
    np.testing.assert_array_equal(whole.z_mm, (before, plane, after, upper))
    assert abs(whole.tx_rad[1, 0]-whole.tx_rad[0, 0]) > 1e-8
    np.testing.assert_array_equal(whole.tx_rad[2], whole.tx_rad[1])
    prefix = build_propagation_plan(state, lower, plane, checkpoint_z_mm=(plane,))
    point = execute_propagation_plan(state, prefix, *initial)[-1]
    suffix = build_propagation_plan(state, plane, upper, checkpoint_z_mm=(upper,))
    resumed = execute_propagation_plan(state, suffix, point.x_m[-1], point.tx_rad[-1],
        point.y_m[-1], point.ty_rad[-1], include_initial_plane_kicks=False)[-1]
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "kinetic_energy_ev"):
        np.testing.assert_allclose(getattr(resumed, name)[-1], getattr(whole, name)[-1], rtol=2e-14, atol=1e-20)


def test_native_plane_cannot_disagree_with_its_si_geometry():
    with pytest.raises(ValueError, match="reference plane"):
        ThinOpticalAction("plane", "plane", .1, CoordinateRegistration(), native_reference_z_mm=101.)

