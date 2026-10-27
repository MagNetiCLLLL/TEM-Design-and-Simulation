"""Thin/local-material ownership and unchanged finite-field specimen transport.

These synthetic and small executed fixtures qualify the stated routing only,
not specimen accuracy or a manufacturer's microscope.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.relativistic_lorentz import (
    ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, SPEED_OF_LIGHT_M_PER_S,
)
from temsim.physics import thin_optics
from temsim.specimen import elastic_transport as elastic
from temsim.specimen import vector_field_transport as vector
from temsim.specimen.axial_field_transport import sample_axial_field_diagnostic


def _state(*, optics=True, matter=False):
    state = default_state()
    state.simulation_mode = "ideal_thin"
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.sample.eds_support_material_key = "vacuum"
    if not optics:
        for collection in (state.lenses, state.stigmators, state.corrector_elements, state.deflectors):
            for component in collection:
                component.enabled = False
    if matter:
        state.sample.inserted = True
        state.sample.specimen_mode = "atomic"
        state.sample.cif_path = str(Path(__file__).parent / "fixtures" / "cif" / "Si.cif")
        state.sample.thickness_nm = 100.
        state.sample.size_x_nm = state.sample.size_y_nm = 1000.
    return state


def _speed(energy):
    gamma = 1 + energy * ELEMENTARY_CHARGE_C / (ELECTRON_MASS_KG * SPEED_OF_LIGHT_M_PER_S**2)
    return SPEED_OF_LIGHT_M_PER_S * np.sqrt(1 - gamma**-2)


def _forbidden(*_args, **_kwargs):
    raise AssertionError("The thin specimen must not sample or integrate continuous optics")


@pytest.mark.parametrize("energy", [120_000., 300_000.])
@pytest.mark.parametrize("length", [-100., 100.])
def test_thin_flight_has_zero_continuous_field_exact_drift_and_signed_clock(monkeypatch, energy, length):
    state = _state()
    monkeypatch.setattr(vector, "capture_instrument_magnetic_field", _forbidden)
    monkeypatch.setattr(vector, "boris_step", _forbidden)
    transport = vector.SpecimenFieldTransport(state)
    np.testing.assert_array_equal(transport.field_at_global_positions_t(np.zeros((7, 3))), 0.)
    position = np.array((10., -20., 500.))
    direction = np.array((.01, .003, 1.))
    direction /= np.linalg.norm(direction)
    endpoint, final, elapsed = transport.advance(
        position, direction, length, energy_ev=energy, return_elapsed_time=True)
    np.testing.assert_allclose(endpoint, position + length * direction, rtol=0, atol=3e-14)
    np.testing.assert_allclose(final, direction, rtol=0, atol=2e-16)
    assert elapsed == pytest.approx(length * 1e-9 / _speed(energy), rel=3e-15)


def test_thin_diagnostic_never_queries_finite_field(monkeypatch):
    import temsim.specimen.axial_field_transport as axial
    monkeypatch.setattr(axial, "fields", _forbidden)
    diagnostic = axial.sample_axial_field_diagnostic(_state())
    assert diagnostic.total_field_t == diagnostic.objective_field_t == 0.
    assert diagnostic.face_fields_t == (0., 0.)
    assert diagnostic.active_lens_contributions_t == ()
    assert diagnostic.transport_model == "thin_optics_straight_specimen_flights"


def test_nonzero_objective_rejects_finite_material_before_any_scatter(monkeypatch):
    state = _state(matter=True)
    monkeypatch.setattr(elastic, "elastic_scattering_rates_nm_inverse", _forbidden)
    ray = elastic.IncidentElectronRay(0, (0., 0.), (0., 0., 1.), 300_000., 1.)
    with pytest.raises(ValueError, match="objective_lens.*finite specimen envelope.*Finite-Field Optics"):
        elastic.simulate_elastic_point_transport(state, incident_rays=(ray,))


@pytest.mark.parametrize("start,target", [(100., 0.), (0., -100.), (-100., 100.)])
def test_local_matching_cannot_silently_replay_or_omit_an_executed_action(start, target):
    transport = vector.SpecimenFieldTransport(_state())
    with pytest.raises(ValueError, match="reference-matching flight.*Finite-Field Optics"):
        transport.to_plane((0., 0., start), (0., 0., 1.), target, energy_ev=300_000.)


def test_zero_length_reference_matching_is_allowed_on_nonzero_action():
    transport = vector.SpecimenFieldTransport(_state())
    position = np.array((10., -20., 0.))
    end, final, elapsed = transport.to_plane(
        position, (.01, .02, 1.), 0., energy_ev=200_000., return_elapsed_time=True)
    np.testing.assert_array_equal(end, position)
    assert final[0] / final[2] == pytest.approx(.01)
    assert elapsed == 0.


def _box_geometry(*, centre=(0., 0.), size=(100., 100.), thickness=10., support=False, matter=True):
    return SimpleNamespace(
        sample_material=object() if matter else None,
        sample_centre_xy_nm=centre, sample_size_xy_nm=size, sample_thickness_nm=thickness,
        support_material=object() if support else None,
        support_offset_xy_um=(0., 0.), support_top_nm=5., support_bottom_nm=15.,
        support_grid=SimpleNamespace(outer_diameter_mm=.001),
    )


def _posed_context(monkeypatch, *, power=1., z_nm=20., tilt=np.pi/4):
    state = _state(optics=False)
    co, si = np.cos(tilt), np.sin(tilt)
    registration = CoordinateRegistration(
        origin_global_m=(0., 0., state.sample.z_mm * 1e-3 + z_nm * 1e-9),
        rotation_local_to_global=((co, 0., si), (0., 1., 0.), (-si, 0., co)),
    )
    action = thin_optics.ThinOpticalAction("posed", "posed", 0., registration, power_m1=power)
    monkeypatch.setattr(thin_optics, "capture_all_thin_actions", lambda *_: (action,))
    return vector.SpecimenFieldTransport(state)


def test_tilted_plane_intersection_uses_xyz_not_only_its_centre_z(monkeypatch):
    context = _posed_context(monkeypatch)
    # The action centre is 20 nm downstream, outside the +/-5 nm specimen;
    # its tilted physical plane still intersects the finite lateral envelope.
    with pytest.raises(ValueError, match="posed.*finite specimen envelope"):
        context.validate_material_geometry(_box_geometry())
    # Translate the actual material away from that plane; both physical and
    # scheduled-Z ownership planes are now demonstrably disjoint.
    context.validate_material_geometry(_box_geometry(centre=(-100., 0.), size=(10., 10.)))
    context.validate_material_geometry(_box_geometry(matter=False))


def test_tilted_local_flight_cannot_cross_plane_with_unintersected_z_interval(monkeypatch):
    context = _posed_context(monkeypatch)
    with pytest.raises(ValueError, match="reference-matching"):
        context.advance((15., 0., 0.), (1., 0., 0.), 10., energy_ev=300_000.)


@pytest.mark.parametrize("power", [0., 1.])
def test_actual_support_envelope_and_zero_excitation_policy(monkeypatch, power):
    context = _posed_context(monkeypatch, power=power, z_nm=10., tilt=0.)
    geometry = _box_geometry(matter=False, support=True)
    if power:
        with pytest.raises(ValueError, match="physical support envelope"):
            context.validate_material_geometry(geometry)
    else:
        context.validate_material_geometry(geometry)
        context.advance((0., 0., 0.), (0., 0., 1.), 20., energy_ev=300_000.)


@pytest.mark.parametrize("blanked,voltage,rejected", [(False, 500., False), (True, 0., False), (True, 500., True)])
def test_static_blanker_uses_shared_action_capture_for_material_gate(blanked, voltage, rejected):
    state = _state(optics=False)
    state.nanopulser.installed = True
    state.nanopulser.blanked = blanked
    state.nanopulser.voltage_v = voltage
    state.nanopulser.z_mm = state.sample.z_mm
    state.nanopulser.stop_z_mm = state.sample.z_mm + 20.
    context = vector.SpecimenFieldTransport(state)
    if rejected:
        assert any(action.key == "blanker:0" for action in context.thin_actions)
        with pytest.raises(ValueError, match="blanker:0.*Finite-Field Optics"):
            context.validate_material_geometry(_box_geometry())
    else:
        context.validate_material_geometry(_box_geometry())


def _rays():
    direction = np.array((.02, -.01, 1.))
    direction /= np.linalg.norm(direction)
    return tuple(elastic.IncidentElectronRay(i, (0., 0.), tuple(direction), energy, weight)
                 for i, (energy, weight) in enumerate(((200_000., .2), (300_000., .8))))


def test_vacuum_retains_executed_reference_without_artificial_inverse_action():
    state = _state()
    state.sample.thickness_nm = 100.
    result = elastic.simulate_elastic_point_transport(state, incident_rays=_rays())
    terminal = result.terminal_electrons
    np.testing.assert_array_equal(terminal.position_nm, np.zeros((2, 3)))
    np.testing.assert_array_equal(terminal.kinetic_energy_ev, [200_000., 300_000.])
    np.testing.assert_array_equal(terminal.weight, [.2, .8])
    np.testing.assert_array_equal(terminal.reference_time_offset_s, [0., 0.])
    assert result.metrics["sample_field_transport_model"] == "thin_optics_straight_specimen_flights"


def test_disjoint_material_retains_scattering_energy_weights_and_signed_times(monkeypatch):
    state = _state(optics=False, matter=True)
    monkeypatch.setattr(elastic, "elastic_scattering_rates_nm_inverse", lambda *_: ((14, .05),))
    monkeypatch.setattr(elastic, "sample_screened_rutherford_angle", lambda *_: (.04, .3))
    finite_state = _state(optics=False, matter=True)
    finite_state.simulation_mode = "ideal"
    thin = elastic.simulate_elastic_point_transport(
        state, incident_rays=_rays(), seed=12, stored_trajectory_count=2)
    finite = elastic.simulate_elastic_point_transport(
        finite_state, incident_rays=_rays(), seed=12, stored_trajectory_count=2)
    left, right = thin.terminal_electrons, finite.terminal_electrons
    assert np.all(left.event_count > 0)
    np.testing.assert_array_equal(left.event_count, right.event_count)
    np.testing.assert_array_equal(left.kinetic_energy_ev, [200_000., 300_000.])
    np.testing.assert_array_equal(left.weight, [.2, .8])
    np.testing.assert_allclose(left.position_nm, right.position_nm, atol=2e-12, rtol=2e-13)
    np.testing.assert_allclose(left.direction, right.direction, atol=2e-15)
    np.testing.assert_allclose(left.reference_time_offset_s, right.reference_time_offset_s, atol=1e-27, rtol=2e-12)
    assert finite.metrics["sample_field_transport_model"] == "shared_vector_field_relativistic_boris"


def test_old_finite_field_mode_still_bends_locally():
    state = _state()
    state.simulation_mode = "ideal"
    context = vector.SpecimenFieldTransport(state)
    point = np.array((10., 0., 0.))
    direction = np.array((.01, .003, 1.))
    direction /= np.linalg.norm(direction)
    end, final = context.advance(point, direction, 100., energy_ev=300_000.)
    assert np.linalg.norm(context.field_at_global_positions_t(np.zeros(3))) > .1
    assert np.linalg.norm(end - point - 100. * direction) > 1e-6
    assert np.linalg.norm(final - direction) > 1e-8
    assert sample_axial_field_diagnostic(state).transport_model == "shared_vector_field_relativistic_boris"


@pytest.mark.parametrize("mode", ["ideal_thin", "ideal"])
def test_projected_atom_response_at_reference_remains_available_without_reapplying_objective(mode):
    from test_projected_response import _simulation, _projection
    from temsim.detector.projected_response import build_projected_responses
    state = _state(optics=False)
    state.simulation_mode = mode
    state.objective_lens.enabled = mode == "ideal_thin"
    state.sample.thickness_nm = 5.
    state.camera.inserted = True
    original_camera = state.camera
    original_planes = tuple(state.recording_planes)
    original_inserted = tuple(plane.inserted for plane in original_planes)
    simulation = _simulation()
    simulation.incident.z[:] = state.sample.z_mm
    responses = build_projected_responses(
        state, simulation, _projection(state, simulation), stop_z_mm=state.sample.z_mm + .1)
    direct = responses["direct"].branches[0]
    # Incident coordinates already include the objective action. Projection
    # adds its independent angular statistics at that same reference plane.
    expected_x = simulation.incident.x[0] + .1e-3 * simulation.incident.tx[0]
    np.testing.assert_allclose(direct.x[-1], expected_x, rtol=5e-8, atol=1e-13)
    np.testing.assert_array_equal(direct.kinetic_energy_ev[0], simulation.incident.kinetic_energy_ev[0])
    scattered = responses["Z14"]
    assert 0. < scattered.metrics["elastic_nontransmitted_conditional_probability"] < 1.
    assert (sum(branch.weight for branch in scattered.branches)
            + scattered.metrics["elastic_nontransmitted_conditional_probability"]) == pytest.approx(1.)
    assert not scattered.checkpoints and not scattered.segments
    assert state.camera is original_camera
    assert all(left is right for left, right in zip(state.recording_planes, original_planes, strict=True))
    assert tuple(plane.inserted for plane in state.recording_planes) == original_inserted
    assert state.camera.inserted
    assert responses["direct"].metrics["terminal_state_projection_preserves_free_flight_line"] == (mode == "ideal_thin")


def test_actual_terminal_matching_rejects_unowned_objective_crossing():
    from test_downstream_transport import _test_transport, _test_simulation
    from temsim.specimen.downstream_transport import build_geometric_specimen_exit
    state = _state()
    simulation = _test_simulation()
    simulation.incident.z[:] = state.sample.z_mm
    with pytest.raises(ValueError, match="reference-matching.*Finite-Field Optics"):
        build_geometric_specimen_exit(state, simulation, _test_transport(),
                                     stop_z_mm=state.sample.z_mm + .1, capture_sections=False)
