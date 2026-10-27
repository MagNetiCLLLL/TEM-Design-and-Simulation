"""Analytical physical-stop boundaries, angular geometry and acceptance gates."""
import numpy as np
import pytest

from temsim.optics.pupil_acceptance import (
    direction_angles, measure_radial_boundary, summarize_boundary,
)


def _mapped_observations(boundary, *, slope_gain=1., distance_m=1.):
    rows = boundary["sectors"]
    azimuth = np.array([row["azimuth_rad"] for row in rows])
    slope = np.array([row["accepted_launch_slope"] for row in rows])
    xy = distance_m * slope[:, None] * np.c_[np.cos(azimuth), np.sin(azimuth)]
    phase = np.c_[xy, xy / distance_m * slope_gain][None, :, :]
    common = dict(z_mm=np.array([distance_m * 1000]), snapshot_digest="analytic-drift",
                  reference_emission_energy_ev=300000.)
    rim = dict(**common, xy_slopes=phase, alive=np.ones((1, len(rows)), bool))
    chief = dict(**common, xy_slopes=np.zeros((1, 1, 4)), alive=np.ones((1, 1), bool))
    return rim, chief


def test_circular_physical_stop_resolves_30_mrad_without_source_quantiles():
    distance_m, aperture_radius_m = .2, .2 * np.tan(.03)

    def drift_to_stop(slopes, azimuths):
        radius = distance_m * slopes
        live = radius <= aperture_radius_m
        return dict(transmitted=live, blocked_key=np.where(live, "", "physical_aperture"))

    boundary = measure_radial_boundary(drift_to_stop, azimuth_count=16,
        radial_slopes=[0., .01, .02, .04, .08], refinements=4)
    assert boundary["status"] == "BRACKETED"
    for row in boundary["sectors"]:
        lower, upper = row["accepted_launch_slope"], row["rejected_launch_slope"]
        assert lower <= aperture_radius_m / distance_m < upper
        assert upper - lower <= .02 / 8**4 * (1 + 1e-9)
        assert row["limiting_component"] == "physical_aperture"
    rim, chief = _mapped_observations(boundary)
    summary = summarize_boundary(boundary, rim, chief)
    np.testing.assert_allclose(summary["specimen"]["alpha_edge_by_source_azimuth_mrad"],
                               30., rtol=0, atol=.005)
    assert summary["status"] == "WITHIN_TARGET_PENDING_CONVERGENCE"
    assert not summary["physical_qualification"] and not summary["alpha95_used_for_target"]


def test_elliptic_and_circular_stops_select_actual_first_limiting_component():
    # A circular aperture at 0.1 m admits slopes <= 0.03. The later elliptical
    # aperture at 0.2 m admits 0.04 along X but only 0.02 along Y.
    def two_stops(slopes, phi):
        x, y = slopes * np.cos(phi), slopes * np.sin(phi)
        circle_blocked = slopes > .003 / .1
        ellipse_blocked = (x / (.008 / .2))**2 + (y / (.004 / .2))**2 > 1
        labels = np.where(circle_blocked, "upstream_circle",
                          np.where(ellipse_blocked, "downstream_ellipse", ""))
        return dict(transmitted=labels == "", blocked_key=labels)

    boundary = measure_radial_boundary(two_stops, azimuth_count=16,
        radial_slopes=[0., .01, .025, .05, .1], refinements=4)
    assert boundary["status"] == "BRACKETED"
    for row in boundary["sectors"]:
        phi = row["azimuth_rad"]
        elliptical_limit = 1 / np.sqrt((np.cos(phi)/.04)**2 + (np.sin(phi)/.02)**2)
        expected = min(.03, elliptical_limit)
        assert row["accepted_launch_slope"] <= expected + 1e-14
        assert row["rejected_launch_slope"] >= expected - 1e-14
    assert boundary["sectors"][0]["limiting_component"] == "upstream_circle"
    assert boundary["sectors"][4]["limiting_component"] == "downstream_ellipse"
    rim, chief = _mapped_observations(boundary)
    summary = summarize_boundary(boundary, rim, chief)
    assert summary["status"] == "OUTSIDE_TARGET"
    np.testing.assert_allclose(summary["specimen"]["alpha_edge_min_mrad"],
                               np.arctan(.02)*1000, atol=.01)
    np.testing.assert_allclose(summary["specimen"]["alpha_edge_max_mrad"],
                               np.arctan(.03)*1000, atol=.01)


def test_missing_blocked_and_disconnected_sectors_cannot_be_promoted():
    def partial_acceptance(slopes, phi):
        sector = np.rint(phi / (2*np.pi/8)).astype(int)
        live = slopes <= .025
        live[sector == 0] = True  # No physical stop was reached in this search.
        live[sector == 1] = False  # The chief is intercepted.
        island = sector == 2
        live[island] = (slopes[island] < .015) | ((slopes[island] > .025) & (slopes[island] < .035))
        return dict(transmitted=live, blocked_key=np.where(live, "", "absorbing_material"))

    boundary = measure_radial_boundary(partial_acceptance, azimuth_count=8,
        radial_slopes=[0., .01, .02, .03, .04], refinements=2)
    assert boundary["status"] == "UNRESOLVED"
    assert [row["reason"] for row in boundary["sectors"][:3]] == [
        "no_mechanical_boundary_in_searched_launch_directions", "chief_ray_blocked",
        "disconnected_acceptance_detected"]
    assert all(row["accepted_launch_slope"] is None for row in boundary["sectors"][:3])
    # Even an alpha95 of exactly 30 must not manufacture the absent pupil.
    result = summarize_boundary(boundary, {"alpha95_mrad": 30.}, {})
    assert result["status"] == "NOT_MEASURED" and result["alpha_edge_mrad"] is None


def test_spherical_angle_about_a_tilted_chief_is_not_a_slope_difference():
    chief_angle = .3
    offsets = np.array([-.03, 0., .03])
    slopes = np.c_[np.tan(chief_angle + offsets), np.zeros(3)]
    angles = direction_angles(slopes, [np.tan(chief_angle), 0.])
    np.testing.assert_allclose(angles, np.abs(offsets), atol=2e-16, rtol=0)
    assert abs(slopes[-1, 0] - np.tan(chief_angle) - .03) > .002
    np.testing.assert_allclose(direction_angles([[1., 1.]], [0., 0.]),
                               np.arctan(np.sqrt(2.)), atol=2e-16, rtol=0)


def test_transport_must_identify_physical_interception_and_refinement_remain_stable():
    with pytest.raises(ValueError, match="physical stops"):
        measure_radial_boundary(lambda r, p: dict(transmitted=r < .02,
                                blocked_key=np.full(len(r), "")), azimuth_count=8)
    calls = 0

    def moving_boundary(slopes, phi):
        nonlocal calls
        calls += 1
        # A changing transport/geometry cannot qualify a bracket frozen earlier.
        live = slopes <= (.025 if calls == 1 else .005)
        return dict(transmitted=live, blocked_key=np.where(live, "", "aperture"))

    boundary = measure_radial_boundary(moving_boundary, azimuth_count=8,
        radial_slopes=[0., .01, .02, .03], refinements=2)
    assert boundary["status"] == "UNRESOLVED"
    assert all(row["reason"] == "unstable_or_disconnected_boundary" for row in boundary["sectors"])


def test_rim_and_chief_require_same_planes_and_unintercepted_transport():
    boundary = measure_radial_boundary(lambda r, p: dict(transmitted=r <= .02,
        blocked_key=np.where(r <= .02, "", "aperture")), azimuth_count=8,
        radial_slopes=[0., .01, .03], refinements=1)
    rim, chief = _mapped_observations(boundary)
    wrong_plane = dict(chief, z_mm=chief["z_mm"] + 1.)
    with pytest.raises(ValueError, match="matching planes"):
        summarize_boundary(boundary, rim, wrong_plane)
    blocked_rim = dict(rim, alive=rim["alive"].copy())
    blocked_rim["alive"][0, 0] = False
    with pytest.raises(ValueError, match="surviving rays"):
        summarize_boundary(boundary, blocked_rim, chief)
    for key, value in (("snapshot_digest", "another-column"), ("reference_emission_energy_ev", 100000.)):
        with pytest.raises(ValueError, match="snapshot and reference emission energy"):
            summarize_boundary(boundary, rim, dict(chief, **{key: value}))
    for target in (0., -30., np.nan):
        with pytest.raises(ValueError, match="target"):
            summarize_boundary(boundary, rim, chief, target_mrad=target)


def test_zero_current_tip_to_column_handoff_preserves_energy_clock_and_gun_stops(monkeypatch):
    """Only the expensive executors are mocked; snapshot/launch routing is real."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.column import default_state
    from temsim.optics.electron_gun.base import GunExitBundle
    from temsim.optics.pupil_acceptance import trace_tip_probes

    state = default_state()
    state.simulation_mode = "unified"
    state.electron_gun.emitter.curvature_nm_inv = 0.
    state.electron_gun.emitter.surface_model = None
    state.electron_gun.emitter.coherence = None
    state.vacuum_map.enabled = False
    before = capture_instrument_snapshot(state)
    launch = np.array([[0., 0., 0., 0.], [0., 0., .001, 0.], [0., 0., .004, 0.]])
    planes = np.array([500., 700.])
    calls = {}

    def gun_trace(gun, bundle, **kwargs):
        assert gun is not state.electron_gun
        np.testing.assert_array_equal(bundle.weight, np.zeros(3))
        np.testing.assert_array_equal(bundle.ray_id, np.arange(3))
        np.testing.assert_allclose(bundle.energy_offset_ev,
                                   .4-state.electron_gun.emitter.emission_energy_ev)
        calls["nominal_exit_energy_ev"] = gun.nominal_exit_energy_ev
        exit_bundle = GunExitBundle(x_m=np.array([np.nan, 1e-6, 2e-6]),
            y_m=np.array([np.nan, -1e-6, -2e-6]), tx_rad=np.array([np.nan, .002, .003]),
            ty_rad=np.array([np.nan, -.002, -.003]), energy_offset_ev=np.array([np.nan, .75, -.25]),
            weight=np.zeros(3), ray_id=np.arange(3), alive=np.array([False, True, True]),
            flight_time_s=np.array([np.nan, 2e-8, 3e-8]))
        return SimpleNamespace(exit_bundle=exit_bundle, blocked_z_mm=np.array([400., np.nan, np.nan]),
                               blocked_key=("gun_stop", "", ""), plane_arrivals=())

    def column_trace(current, initial, z0, targets, **kwargs):
        assert current is not state
        np.testing.assert_array_equal(initial, [[1e-6, -1e-6, .002, -.002],
                                                [2e-6, -2e-6, .003, -.003]])
        np.testing.assert_allclose(kwargs["energy_ev"],
                                   calls["nominal_exit_energy_ev"]+np.array([.75, -.25]))
        np.testing.assert_array_equal(kwargs["flight_time_s"], [2e-8, 3e-8])
        np.testing.assert_allclose(targets, planes*.001)
        assert kwargs["maximum_step_m"] == pytest.approx(.00005)
        return dict(xy_slopes=np.broadcast_to(initial, (2, 2, 4)).copy(),
                    kinetic_energy_ev=np.array([[300001., 300002.], [300003., 300004.]]),
                    flight_time_s=np.array([[4e-8, 5e-8], [6e-8, 7e-8]]),
                    blocked_z_m=np.array([np.nan, .65]), blocked_key=("", "column_stop"),
                    receipt={"scope": "Mock transport; no physical qualification"})

    monkeypatch.setattr("temsim.physics.instrument_magnetic.column_dipole_fields", lambda current: [])
    monkeypatch.setattr("temsim.vacuum.bind_gun_environment", lambda current: None)
    monkeypatch.setattr("temsim.physics.gun_field_environment.instrument_gun_field_context",
                        lambda current: nullcontext())
    monkeypatch.setattr("temsim.optics.electron_gun.tracing.trace_feg_diagnostic_to_exit", gun_trace)
    monkeypatch.setattr("temsim.physics.unified_field_particle.trace_unified_response", column_trace)
    result = trace_tip_probes(state, launch, planes, step_mm=.05, emission_energy_ev=.4)
    assert result["zero_current"] and result["reference_emission_energy_ev"] == .4
    assert result["blocked_key"] == ["gun_stop", "", "column_stop"]
    np.testing.assert_array_equal(result["alive"], [[False, True, True], [False, True, False]])
    assert np.isnan(result["xy_slopes"][:, 0]).all()
    np.testing.assert_array_equal(result["kinetic_energy_ev"][:, 1:], [[300001., 300002.], [300003., 300004.]])
    np.testing.assert_array_equal(result["source_direction_supported"], [True, True, False])
    np.testing.assert_array_equal(result["launch_xy_slopes"], launch)
    assert capture_instrument_snapshot(state).digest == before.digest == result["snapshot_digest"]
