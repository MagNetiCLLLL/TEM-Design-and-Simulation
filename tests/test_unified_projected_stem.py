"""Unified raster wiring with real CIF projection and analytic response fixtures.

These bounded tests qualify ownership, coordinates and current accounting,
not full-column physical convergence or quantitative atomic imaging.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.detector import projected_response, projected_stem, stem_signal
from temsim.detector import unified_particle_stem as unified
from test_projected_stem_integration import _fcc_cif, _state, _incident, _install_drift_responses


@pytest.fixture
def setup(tmp_path, monkeypatch):
    state = _state(_fcc_cif(tmp_path))
    state.simulation_mode = "unified"
    state.step_mm = .05
    state.electron_gun.emitter.ray_count = 1000
    state.ac_deflector.scan_pixels_x = 5
    state.ac_deflector.scan_lines = 3
    state.ac_deflector.set_scan_command_matrix_mrad(np.eye(2), 0.)
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight",
                        lambda _state: {"issues": (), "supported": True})
    _install_drift_responses(monkeypatch)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unified particle STEM invoked legacy equivalent optics")

    monkeypatch.setattr(stem_signal, "paired_kick_response", forbidden)
    monkeypatch.setattr(stem_signal, "calibrate_scan_system", forbidden)
    monkeypatch.setattr(stem_signal, "collection_angle", forbidden)
    monkeypatch.setattr(projected_stem, "paired_kick_response", forbidden)
    calls = []

    def measure(local, targets, **kwargs):
        calls.append((tuple(targets), kwargs))
        targets = sorted(set(targets))
        upper, lower = map(np.asarray, local.ac_deflector.coil_kick_matrices())
        rows = np.zeros((len(targets), 4, 2))
        # Deliberately specified independent affine fixture, in m/rad.
        # A 1 mrad command moves the probe by 0.1 nm at the sample.
        rows[targets.index(local.sample.z_mm), :2] = 1e-7 * np.linalg.inv(upper)
        return {"z_mm": targets,
                "phase": {"ac_deflector:0": rows, "ac_deflector:1": np.zeros_like(rows)},
                "receipt": {"model": "analytic_coil_derivative_fixture", "reference_kinetic_energy_ev": 300000.}}

    monkeypatch.setattr(unified, "measure_scan_responses", measure)
    return state, _incident(.025, .7), calls


def test_shared_response_drives_non_square_full_raster_with_exact_current_budget(setup):
    state, simulation, calls = setup
    frame = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    assert frame.scan_x_um.shape == (3, 5)
    # The fixture probe is centred at zero at the reference epoch. That epoch
    # already includes a -1 mrad scan command, removed exactly once here.
    np.testing.assert_allclose(frame.scan_x_um[0], (np.linspace(-.8, .8, 5)+1.)*1e-4,
                               rtol=0, atol=2e-19)
    times = np.asarray(frame.metrics["scan_times_s"])
    np.testing.assert_allclose(times, (np.arange(15).reshape(3, 5)+.5)/15)
    total = sum(frame.fractions.values()) + frame.uncollected_fraction + frame.absorbed_fraction + frame.truncated_fraction
    np.testing.assert_allclose(total, 1., rtol=0, atol=2e-14)
    assert frame.metrics["unified_potential_stem"] is True
    assert frame.metrics["per_pixel_nonlinear_column_retrace"] is False
    assert frame.metrics["source_particle_count"] == 1000
    assert frame.metrics["projected_direct_ray_limit"] == 64
    assert frame.metrics["projected_angular_samples_per_element_nominal"] == 512
    assert frame.metrics["sample_incident_source_probability"] == pytest.approx(.7)
    assert calls[0][1]["maximum_step_mm"] == .05
    assert all(signal.collection_angle is None for signal in frame.detector_signals.values())


def test_unread_inserted_detector_still_absorbs_before_downstream(setup):
    state, simulation, _ = setup
    haadf = next(plane for plane in state.stem_detectors if plane.key == "haadf")
    haadf.geometry = "disk"
    haadf.inner_diameter_mm = 0.
    haadf.outer_width_mm = 100.
    haadf.readout_enabled = False
    frame = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    assert "haadf" not in frame.fractions
    assert frame.metrics["unreadout_intercepted_mean_fraction"]["haadf"] == pytest.approx(.7)
    np.testing.assert_array_equal(frame.fractions["bf"], 0.)
    np.testing.assert_array_equal(frame.fractions["df"], 0.)
    np.testing.assert_allclose(frame.uncollected_fraction, .3, atol=2e-14)


def test_fully_clipped_probe_retains_zero_frame_without_replacement_source(setup, monkeypatch):
    state, simulation, _ = setup
    simulation.incident.alive[:] = False
    monkeypatch.setattr(unified, "measure_scan_responses",
                        lambda *_a, **_k: pytest.fail("A missing probe must not be recreated"))
    frame = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    assert frame.probe_state is None
    assert frame.metrics["zero_illumination"] is True
    assert frame.metrics["sample_incident_source_probability"] == 0.
    assert frame.scan_x_um.shape == (3, 5)
    for values in (*frame.fractions.values(), *frame.current_pa.values(), *frame.expected_electrons.values()):
        np.testing.assert_array_equal(values, 0.)
    np.testing.assert_array_equal(frame.uncollected_fraction, 1.)


def test_physical_shared_host_and_both_coil_wiring_matrices_are_respected():
    upper = np.array(((1., 2.), (0., 3.)))
    lower = np.array(((4., 0.), (5., -1.)))
    component = SimpleNamespace(key="descan_deflector", enabled=True, _physical_host_key="image_diffraction_deflector",
                                coil_kick_matrices=lambda: (upper, lower), upper_z_mm=1., lower_z_mm=2.)
    up = np.arange(8, dtype=float).reshape(1, 4, 2)
    low = np.arange(8, 16, dtype=float).reshape(1, 4, 2)
    measured = {"z_mm": [3.], "phase": {"image_diffraction_deflector:0": up,
                                          "image_diffraction_deflector:1": low}}
    actual = unified._paired_position_response(measured, component, 3.)
    np.testing.assert_array_equal(actual, 1e3*(up[0, :2]@upper+low[0, :2]@lower))


def test_shared_projection_never_falls_back_to_legacy_on_missing_measured_map(setup):
    state, simulation, _ = setup
    with pytest.raises(ValueError, match="requires measured optical responses"):
        projected_stem.acquire_projected_stem_scan(
            simulation, state, state.stem_detectors,
            scan_x_um=np.zeros((2, 2)), scan_y_um=np.zeros((2, 2)),
            kick_grid_mrad=np.zeros((2, 2, 2)), baseline_scan_mrad=np.zeros(2),
            baseline_descan_scan_mrad=np.zeros(2), sample_response=np.eye(2), scan_times_s=np.zeros((2, 2)))


def test_zero_depth_terminal_preserves_energy_and_source_rows_without_fabricated_material_clock():
    rays = (SimpleNamespace(source_ray_index=7, position_xy_nm=(2., -3.),
                            direction=(0., 0., 1.), kinetic_energy_ev=299999.7),)
    terminal = projected_response._terminal(rays, np.array([0]), np.array([1.]), 5.,
                                            projected_reference_plane=True)
    np.testing.assert_array_equal(terminal.reference_time_offset_s, 0.)
    np.testing.assert_array_equal(terminal.position_nm[:, 2], 0.)
    np.testing.assert_array_equal(terminal.source_ray_index, [7])
    np.testing.assert_array_equal(terminal.kinetic_energy_ev, [299999.7])
    np.testing.assert_array_equal(terminal.material_path_nm, 5.)


def test_actual_arrival_clock_invalidates_projected_response_cache(setup, monkeypatch):
    state, _, _ = setup
    from test_projected_response import _simulation, _projection
    from temsim.specimen.elastic_transport import incident_rays_from_simulation
    simulation = _simulation()
    simulation.incident.z[:] = state.sample.z_mm
    simulation.incident.flight_time_s = np.full((1, 4), 1e-8)
    projection = _projection(state, simulation)
    incident = incident_rays_from_simulation(state, simulation)
    monkeypatch.setattr("temsim.calculation_cache.calculation_signatures", lambda *_a: {"sample_downstream": "same"})
    monkeypatch.setattr("temsim.physics.unified_field_runtime.get_unified_field_runtime", lambda *_a: SimpleNamespace(digest="same"))
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "same")
    identity = projected_response._response_identity(state, incident, simulation, projection, None, 1., (1.,))
    simulation.incident.flight_time_s[0, 0] += 1e-10
    changed = projected_response._response_identity(state, incident, simulation, projection, None, 1., (1.,))
    assert identity != changed


def test_explicit_absorption_is_resolved_and_passed_without_material_history(setup, monkeypatch):
    state, simulation, _ = setup
    state.sample.real_absorption_mean_free_path_nm = 2.
    state.sample.real_inelastic_enabled = True
    expected = 1.-np.exp(-state.sample.thickness_nm/2.)
    _install_drift_responses(monkeypatch, absorbed=expected)
    original = projected_response.build_projected_responses
    captured = []
    def build(local, incident, projection, distribution=None, **kwargs):
        captured.append(distribution)
        return original(local, incident, projection, distribution, **kwargs)
    monkeypatch.setattr(projected_response, "build_projected_responses", build)
    frame = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    assert captured[0].absorbed_probability == pytest.approx(expected)
    assert frame.metrics["inelastic_distribution"]["absorbed_probability"] == pytest.approx(expected)
    assert np.max(frame.absorbed_fraction) > 0.
    np.testing.assert_allclose(sum(frame.fractions.values()) + frame.uncollected_fraction
                               + frame.absorbed_fraction + frame.truncated_fraction, 1., atol=2e-14)


def test_positive_loss_override_is_not_silently_dropped(setup, monkeypatch):
    state, simulation, _ = setup
    state.sample.real_other_inelastic_mean_free_path_nm = 2.
    state.sample.real_inelastic_enabled = True
    monkeypatch.setattr(unified, "measure_scan_responses",
                        lambda *_a, **_k: pytest.fail("Unsupported time history must fail before optical work"))
    with pytest.raises(ValueError, match="aggregate positive-energy-loss"):
        stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)


def test_changed_sample_and_downstream_response_maps_reach_physical_image(setup, monkeypatch):
    state, simulation, _ = setup
    original = unified.measure_scan_responses
    base = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    def sample_changed(local, targets, **kwargs):
        measured = original(local, targets, **kwargs)
        row = measured["z_mm"].index(local.sample.z_mm)
        measured["phase"]["ac_deflector:0"][row, :2] *= 3.
        return measured
    monkeypatch.setattr(unified, "measure_scan_responses", sample_changed)
    changed_sample = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    assert not np.array_equal(base.fractions["haadf"], changed_sample.fractions["haadf"])
    def downstream_changed(local, targets, **kwargs):
        measured = original(local, targets, **kwargs)
        upper = np.asarray(local.ac_deflector.coil_kick_matrices()[0])
        for index, z in enumerate(measured["z_mm"]):
            if z != local.sample.z_mm:
                measured["phase"]["ac_deflector:0"][index, :2] = 100.*np.linalg.inv(upper)
        return measured
    monkeypatch.setattr(unified, "measure_scan_responses", downstream_changed)
    changed_downstream = stem_signal.acquire_stem_scan(simulation, state, scan_calibrated=True)
    np.testing.assert_array_equal(base.scan_x_um, changed_downstream.scan_x_um)
    assert not np.array_equal(base.fractions["haadf"], changed_downstream.fractions["haadf"])
