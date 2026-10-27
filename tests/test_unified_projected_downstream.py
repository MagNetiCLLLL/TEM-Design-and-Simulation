"""Bounded projected-observable handoff; no gun, full column or wave solve."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.core import propagate
from temsim.physics.lens_field_provider import CoordinateRegistration
from temsim.physics.posed_wave_aperture import PosedWaveAperture
from temsim.physics.potential_snapshot import PotentialContribution
from temsim.specimen.downstream_transport import build_geometric_specimen_exit
from temsim.specimen.elastic_transport import ElasticTerminalBundle, ElasticTransportResult
from test_unified_field_particle import controlled
from test_unified_field_runtime import parked_state


def fixture(monkeypatch, *, dynamic=False, aperture=False, aperture_start=450.5):
    from temsim.physics.planar_gun_field import PlanarGunField
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    import temsim.physics.unified_field_particle as particle
    state = parked_state()
    state.sample.z_mm = 450.
    state.simulation_time_s = .21e-9
    state.step_mm = .05
    state.recording_planes = []
    state.apertures = []
    radial, axial = np.array([0., .02]), np.array([.449, .453])
    potential = np.broadcast_to(200000. + 2e6 * (axial - .45), (2, 2)).copy()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({}, radial, axial, potential))
    pieces = (PosedWaveAperture("bounded_iris", aperture_start, aperture_start + .1, .015, CoordinateRegistration()),) if aperture else ()
    runtime = controlled(state, (PotentialContribution("electric", ("electric",), electric, "electric"),), pieces)
    if dynamic:
        from test_dynamic_potential import coil
        from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
        runtime = replace(runtime, snapshot=DynamicPotentialSnapshot(runtime.snapshot, (coil(),)))
    runtime = replace(runtime, metadata={"emission_time_s": state.simulation_time_s})
    from temsim.instrument_snapshot import capture_instrument_snapshot
    # Receiver deferral creates an owned state view. Bind this declared tiny
    # field fixture to that view without changing its field or mechanical law.
    monkeypatch.setattr(particle, "get_unified_field_runtime", lambda current:
        replace(runtime, source_identity=capture_instrument_snapshot(current).digest))
    simulation = SimpleNamespace(incident=SimpleNamespace(
        z=np.array([450.]), x=np.array([[1e-5, 2e-5, 0.]]), y=np.zeros((1, 3)),
        tx=np.array([[.001, -.002, 0.]]), ty=np.array([[.002, -.001, 0.]]),
        alive=np.array([True, True, False]), blocked_z=np.array([np.nan, np.nan, 449.]),
        ray_weight=np.array([.2, .3, .5]), source_ray_id=np.array([7, 13, 31]),
        source_azimuth_rad=np.array([.1, .2, .3]),
        flight_time_s=np.array([[2e-9, 3e-9, np.nan]]),
        kinetic_energy_ev=np.array([[200000., 280000., np.nan]])))
    directions = np.c_[simulation.incident.tx[0, :2], simulation.incident.ty[0, :2], np.ones(2)]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    terminal = ElasticTerminalBundle(source_ray_index=np.array([0, 1]),
        position_nm=np.array([[1e4, 0., 0.], [2e4, 0., 0.]]), direction=directions,
        kinetic_energy_ev=np.array([200000., 280000.]), weight=np.array([.4, .6]),
        outcome=("transmitted", "transmitted"), event_count=np.zeros(2, int),
        has_scattered=np.zeros(2, bool), material_path_nm=np.full(2, 5.),
        reference_time_offset_s=np.zeros(2))
    elastic = ElasticTransportResult(eds_tracks=(), trajectories=(), metrics={}, terminal_electrons=terminal)
    return state, simulation, elastic, runtime


def build(state, simulation, elastic, **kwargs):
    return build_geometric_specimen_exit(state, simulation, elastic,
        stop_z_mm=452., save_z_mm=(450., 452.), capture_sections=False,
        recording_interceptions=False, projected_reference_plane=True, **kwargs)


@pytest.mark.parametrize("dynamic", [False, True])
def test_projected_handoff_matches_actual_field_transport_energy_clock_and_identity(monkeypatch, dynamic):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=dynamic)
    import temsim.specimen.downstream_transport as downstream
    import temsim.physics.intrinsic_aberrations as aberrations
    def forbidden(*args, **kwargs):
        raise AssertionError("Projected unified transport must not use legacy fields or chromatic kicks")
    monkeypatch.setattr(downstream, "SpecimenFieldTransport", forbidden)
    monkeypatch.setattr(downstream, "sample_axial_field_diagnostic", forbidden)
    monkeypatch.setattr(aberrations, "specimen_chromatic_energy_change", forbidden)
    result = build(state, simulation, elastic)
    branch, = result.branches
    energy = []
    reference = propagate(state, 450., 452., simulation.incident.x[0, :2],
        simulation.incident.tx[0, :2], simulation.incident.y[0, :2],
        simulation.incident.ty[0, :2], save_z_mm=(450., 452.),
        initial_time_s=simulation.incident.flight_time_s[0, :2], return_flight_times=True,
        initial_kinetic_energy_ev=elastic.terminal_electrons.kinetic_energy_ev,
        energy_output=energy)
    for name, value in zip(("z", "x", "tx", "y", "ty", "flight_time_s"), reference):
        np.testing.assert_allclose(getattr(branch, name), value, rtol=3e-15, atol=1e-22)
    np.testing.assert_allclose(branch.kinetic_energy_ev, energy[0], rtol=0., atol=1e-9)
    np.testing.assert_array_equal(branch.source_ray_id, [7, 13])
    np.testing.assert_array_equal(branch.ray_weight, [.4, .6])
    assert branch.weight == pytest.approx(.5)
    assert result.metrics["source_probability_conserved"]
    assert not result.checkpoints and not result.segments
    assert np.all(branch.kinetic_energy_ev[-1] > branch.kinetic_energy_ev[0])


def test_projected_unified_mechanics_keep_actual_stop_without_legacy_reclip(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch, aperture=True)
    import temsim.specimen.downstream_transport as downstream
    monkeypatch.setattr(downstream, "clip_column_wall", lambda *a, **k: pytest.fail("Legacy wall reclip"))
    branch, = build(state, simulation, elastic).branches
    np.testing.assert_array_equal(branch.alive, [True, False])
    assert branch.blocked_key == ["", "bounded_iris"]
    assert branch.blocked_z[1] == pytest.approx(450.5)
    assert np.isnan(branch.kinetic_energy_ev[-1, 1])
    assert branch.weight == pytest.approx(.5)


def test_projected_entrance_mechanical_absorption_keeps_its_receipt(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch, aperture=True, aperture_start=450.)
    branch, = build(state, simulation, elastic).branches
    np.testing.assert_array_equal(branch.alive, [True, False])
    assert branch.blocked_z[1] == 450.
    assert branch.blocked_key[1] == "bounded_iris"
    assert branch.kinetic_energy_ev[0, 1] == 280000.
    assert np.isnan(branch.kinetic_energy_ev[1:, 1]).all()
    assert "projected_field_domain" not in branch.blocked_key


def test_projected_mode_does_not_admit_finite_material_or_missing_dynamic_clocks(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=True)
    with pytest.raises(ValueError, match="explicitly labelled projected"):
        build_geometric_specimen_exit(state, simulation, elastic, stop_z_mm=452.)
    shifted = elastic.terminal_electrons.position_nm.copy()
    shifted[0, 2] = 1.
    with pytest.raises(ValueError, match="reference plane"):
        build(state, simulation, replace(elastic, terminal_electrons=replace(elastic.terminal_electrons, position_nm=shifted)))
    simulation.incident.flight_time_s[0, 0] = np.nan
    with pytest.raises(ValueError, match="actual finite entrance flight times"):
        build(state, simulation, elastic)


def test_projected_positive_loss_does_not_fabricate_dynamic_event_time(monkeypatch):
    state, simulation, elastic, _ = fixture(monkeypatch, dynamic=True)
    import temsim.specimen.downstream_transport as downstream
    monkeypatch.setattr(downstream, "_inelastic_specs", lambda *args:
        ((("loss", 0., 0., 1., "real_inelastic", 20.),), 1., 0.))
    with pytest.raises(ValueError, match="actual finite entrance flight times"):
        build(state, simulation, elastic)
