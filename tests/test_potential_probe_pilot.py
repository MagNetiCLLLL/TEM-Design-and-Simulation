"""Bounded static C3 + Probe Corrector pilot using real assembly providers.

The physical FEG and legacy column supply 25 weighted particles at z=600 mm.
Only the downstream segment uses the new shared-potential Lorentz operator.
AC scan and wobble are explicitly parked; no lens/current calibration is
changed. This hybrid entrance, small source sample and static scan are NOT
full-gun, source-sampling, dynamic-scan, coherent-phase or production
qualification. The ray action remains a ray action, not a phase plate.
"""
from functools import lru_cache
import json

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import _pre_sample_kick_events
from temsim.optics.electron_gun.source import trace_source_to_exit
from temsim.optics.surface_probe_focus import surface_z_mm
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.aperture_clipping import clip_segment
from temsim.physics.column_wall import clip_column_wall
from temsim.physics.core import propagate
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.potential_capture import capture_selected_column_potentials
from temsim.physics.potential_ray import trace_potential_rays
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution


ENTRANCE_MM = 600.
MODES = ("nano_probe", "nano_probe_large_angle")


def _state(mode):
    state, catalog = default_state(), AssemblyCatalog()
    option = next(option for option in catalog.columns if option.file == "column/C3_ProbeCorrector.toml")
    catalog.apply(state, AssemblySelection("FEG", option.name, "No Energy Filter"))
    apply_operating_mode_pair(state, mode, "diffraction")
    state.acceleration_backend = "CPU"
    for item in (state.ac_deflector, state.descan_deflector):
        item.scan_enabled = False
        if hasattr(item, "wobble_enabled"):
            item.wobble_enabled = False
    return state


def _capture(state, order=3):
    end = surface_z_mm(state)
    lenses = tuple(lens.key for lens in state.lenses if lens.enabled and lens.z_mm <= end+1.)
    auxiliaries = tuple(item.key for item in (*state.corrector_elements, *state.stigmators, *state.deflectors)
        if item.enabled and item.key not in ("probe_dp12_scan_deflector", "descan_deflector", "image_diffraction_deflector")
        and (item.key.startswith("probe_") or item.key in
             ("condenser_stigmator", "beam_deflector", "condenser_deflector", "ac_deflector")))
    native = capture_selected_column_potentials(state, round_lens_keys=lenses, auxiliary_keys=auxiliaries)
    rows, support_faces = [], []
    for row in native.contributions:
        if row.source_id in lenses:
            potential = HighOrderRoundLensPotential(row.potential.field, radial_order=order,
                                                     axial_domain="gaussian_all_z")
        else:
            potential = AnalyticPotentialDomain(row.potential)
            support_faces.extend(row.potential.field.native_support_mm)
        rows.append(PotentialContribution(row.source_id, row.owner_ids, potential, row.kind))
    snapshot = ElectromagneticPotentialSnapshot(native.scene, tuple(rows))
    apertures = tuple(aperture for aperture in state.apertures if aperture.enabled
        and getattr(aperture, "installed", True) and getattr(aperture, "inserted", True)
        and ENTRANCE_MM <= aperture.z_mm <= end)
    # Every compact support face and aperture is an integration boundary.
    # Additional regular observation planes permit weighted crossover checks.
    nodes_mm = np.unique(np.r_[ENTRANCE_MM, end, np.arange(ENTRANCE_MM, end, 2.),
        [value for value in support_faces if ENTRANCE_MM < value < end],
        [aperture.z_mm for aperture in apertures]])
    stops = tuple((int(np.flatnonzero(nodes_mm == aperture.z_mm)[0]),
                   aperture.offset_x_mm*.001, aperture.offset_y_mm*.001, aperture.radius_mm*.001)
                  for aperture in apertures)
    return snapshot, nodes_mm*.001, stops


@lru_cache(maxsize=2)
def _entrance(mode):
    state = _state(mode)
    gun = trace_source_to_exit(state, count=25)
    bundle = gun.exit_bundle
    z, x, tx, y, ty = propagate(state, state.electron_gun.exit_plane_z_mm, ENTRANCE_MM,
        bundle.x_m, bundle.tx_rad, bundle.y_m, bundle.ty_rad,
        events=_pre_sample_kick_events(state), energy_offset_ev=bundle.energy_offset_ev,
        maximum_step_mm=.1, save_z_mm=(ENTRANCE_MM,))
    alive, stopped_z, stopped_key = clip_segment(state, z, x, y, np.asarray(bundle.alive).copy(),
        np.asarray(gun.blocked_z_mm).copy(), list(gun.blocked_key))
    alive, _, _ = clip_column_wall(state, z, x, y, alive, stopped_z, stopped_key)
    selected = np.flatnonzero(alive)
    return state, np.c_[x[-1, selected], y[-1, selected], tx[-1, selected], ty[-1, selected]], \
        bundle.weight[selected], state.electron_gun.nominal_exit_energy_ev+bundle.energy_offset_ev[selected]


def _run(state, initial, weights, energy, step=.00025, order=3):
    snapshot, nodes, stops = _capture(state, order)
    result = trace_potential_rays(snapshot, initial, nodes, energy_ev=energy, weights=weights,
        maximum_step_m=step, method="RK4", stop_planes=stops, sampling_backend="native_compiled")
    return snapshot, result


def _weighted_covariance(result):
    covariances, rms, centres = [], [], []
    for rays, alive in zip(result.xy_slopes, result.alive):
        weights, rays = result.weights[alive], rays[alive]
        centre = np.average(rays, axis=0, weights=weights)
        delta = rays-centre
        covariances.append(np.average(delta[:, 0]*delta[:, 2]+delta[:, 1]*delta[:, 3], weights=weights))
        rms.append(np.sqrt(np.average(np.sum(delta[:, :2]**2, axis=1), weights=weights)))
        centres.append(centre)
    return np.asarray(covariances), np.asarray(rms), np.asarray(centres)


def _crossovers(result, lower, upper):
    covariance, _, _ = _weighted_covariance(result)
    indices = np.flatnonzero((result.z_m[:-1] >= lower) & (result.z_m[1:] <= upper)
        & (covariance[:-1] < 0.) & (covariance[1:] >= 0.))
    return [float(result.z_m[i]-covariance[i]*(result.z_m[i+1]-result.z_m[i])
                  /(covariance[i+1]-covariance[i])) for i in indices]


@lru_cache(maxsize=2)
def _baseline(mode):
    state, initial, weights, energy = _entrance(mode)
    snapshot, result = _run(state, initial, weights, energy)
    return mode, state, snapshot, result


@pytest.fixture(scope="module", params=MODES)
def pilot(request):
    return _baseline(request.param)


def test_real_native_nanoprobe_pilot_preserves_population_and_expected_upstream_topology(pilot, record_testsuite_property):
    mode, state, snapshot, result = pilot
    first = _crossovers(result, .650, .915)
    second = _crossovers(result, .915, 1.0882)
    _, rms, centroid = _weighted_covariance(result)
    evidence = {"mode": mode, "scope": __doc__, "source_count": 25,
        "represented_sources": len(snapshot.contributions), "snapshot_digest": snapshot.numerical_digest,
        "sampling_backend": result.metadata["potential_sampling_backend"],
        "input_weight": result.metadata["input_weight"], "output_weight": result.metadata["output_weight"],
        "c2_c3_crossovers_m": first, "c3_adl_crossovers_m": second,
        "sample_rms_m": float(rms[-1]), "sample_centroid_xy_slopes": centroid[-1].tolist(),
        "ray_action_range_m": [float(np.nanmin(result.reduced_action_m[-1])),
                               float(np.nanmax(result.reduced_action_m[-1]))]}
    record_testsuite_property("actual_probe_pilot_"+mode, json.dumps(evidence, sort_keys=True))
    assert len(first) == 1, evidence
    assert len(second) == (1 if mode == "nano_probe_large_angle" else 0), evidence
    assert 0 < result.metadata["output_weight"] <= result.metadata["input_weight"]
    assert result.metadata["normalization_applied"] is False
    assert result.metadata["empirical_aberrations_applied"] is False
    assert result.metadata["production_admission"] is False
    assert "not a coherent wave" in result.metadata["action_scope"]
    assert len(snapshot.contributions) == len({row.source_id for row in snapshot.contributions})
    assert "probe_dp12_scan_deflector" not in {row.source_id for row in snapshot.contributions}
    # All losses are accounted for without rescaling surviving source weights.
    assert result.metadata["input_weight"] == pytest.approx(
        result.metadata["output_weight"]+result.metadata["stopped_weight"])


def test_actual_native_B0_is_unchanged_by_field_order_and_dynamic_scan_is_rejected():
    state = _state("nano_probe")
    low, nodes, _ = _capture(state, 1)
    high, _, _ = _capture(state, 5)
    for first, second in zip(low.contributions, high.contributions):
        if type(first.potential) is HighOrderRoundLensPotential:
            points = np.c_[np.zeros(9), np.zeros(9), np.linspace(nodes[0], nodes[-1], 9)]
            np.testing.assert_array_equal(first.potential.sample(points)["B_t"], second.potential.sample(points)["B_t"])
    state.ac_deflector.scan_enabled = True
    dynamic, nodes, _ = _capture(state)
    with pytest.raises(ValueError, match="dynamic drive"):
        trace_potential_rays(dynamic, [[0., 0., 0., 0.]], nodes[:2],
                             energy_ev=300000., maximum_step_m=.00025)


@pytest.mark.parametrize("channel,key,attribute,amount", [
    ("a1_coarse", "probe_dph2_deflector", "kick_x_mrad", .001),
    ("a1_fine", "condenser_stigmator", "strength_x_percent", .01),
    ("a2_coarse", "probe_hp2_hexapole", "strength_m3", 100.),
    ("a2_fine", "probe_hpc_hexapole", "strength_m3", 100.),
    ("hp2_thickness", "probe_hp2_hexapole", "effective_length_mm", 1.2),
])
def test_actual_hardware_channel_and_hexapole_thickness_change_raw_downstream_rays_and_action(
        channel, key, attribute, amount, record_testsuite_property):
    _, state, original, baseline = _baseline("nano_probe")
    changed = capture_instrument_snapshot(state).restore()
    components = {item.key: item for item in (*changed.corrector_elements, *changed.stigmators)}
    component = components[key]
    old_value = getattr(component, attribute)
    setattr(component, attribute, old_value+amount)
    perturbed, nodes, stops = _capture(changed)
    assert perturbed.numerical_digest != original.numerical_digest
    before_sources = {row.source_id: row.potential.digest for row in original.contributions}
    altered_sources = {row.source_id for row in perturbed.contributions
                       if row.potential.digest != before_sources[row.source_id]}
    assert altered_sources and all(source == key or source.startswith(key+":") for source in altered_sources)
    # Native auxiliary fields are exactly zero outside their declared support.
    # Reuse only the unchanged causal prefix, before BOTH support intervals.
    def lower_face(snapshot):
        fields = [row.potential.base.field for row in snapshot.contributions
                  if row.source_id == key or row.source_id.startswith(key+":")]
        assert fields
        return min(field.native_support_mm[0]*.001 for field in fields)
    first_effect = min(lower_face(original), lower_face(perturbed))
    index = int(np.flatnonzero(baseline.z_m < first_effect)[-1])
    start = baseline.z_m[index]
    selected = np.flatnonzero(baseline.alive[index])
    _, _, _, energies = _entrance("nano_probe")
    run_nodes = np.unique(np.r_[start, nodes[nodes > start]])
    run_stops = []
    for node, x, y, radius in stops:
        if nodes[node] >= start:
            run_stops.append((int(np.flatnonzero(run_nodes == nodes[node])[0]), x, y, radius))
    result = trace_potential_rays(perturbed, baseline.xy_slopes[index, selected], run_nodes,
        energy_ev=energies[selected], weights=baseline.weights[selected],
        maximum_step_m=.00025, method="RK4", stop_planes=tuple(run_stops), sampling_backend="native_compiled")
    common = result.alive[-1] & baseline.alive[-1, selected]
    raw_difference = result.xy_slopes[-1, common]-baseline.xy_slopes[-1, selected[common]]
    # The continuation starts its action at zero. Compare the unchanged-prefix
    # subtraction explicitly; do not call this quantity a coherent phase.
    old_action = baseline.reduced_action_m[-1, selected[common]]-baseline.reduced_action_m[index, selected[common]]
    action_difference = result.reduced_action_m[-1, common]-old_action
    position_change = float(np.max(np.linalg.norm(raw_difference[:, :2], axis=1)))
    direction_change = float(np.max(np.linalg.norm(raw_difference[:, 2:], axis=1)))
    action_change = float(np.max(np.abs(action_difference)))
    record_testsuite_property("actual_probe_control_"+channel, json.dumps({
        "component_key": key, "attribute": attribute, "before": old_value,
        "after": getattr(component, attribute), "unchanged_prefix_end_z_m": float(start),
        "max_raw_position_change_m": position_change,
        "max_raw_direction_change_rad": direction_change, "max_ray_action_change_m": action_change,
        "snapshot_before": original.numerical_digest, "snapshot_after": perturbed.numerical_digest,
        "scope": "raw response of one physical drive; not calibrated A1/A2 purity or coherent phase"}))
    assert position_change > 1e-14 and direction_change > 1e-12 and action_change > 1e-20
    assert result.metadata["normalization_applied"] is False


def test_actual_normal_pilot_separates_trajectory_step_and_radial_field_order(record_testsuite_property):
    _, state, nominal_snapshot, nominal = _baseline("nano_probe")
    _, initial, weights, energy = _entrance("nano_probe")
    _, fine = _run(state, initial, weights, energy, step=.000125)
    _, finest = _run(state, initial, weights, energy, step=.0000625)
    np.testing.assert_array_equal(fine.alive, nominal.alive)
    np.testing.assert_array_equal(finest.alive, nominal.alive)
    live = nominal.alive[-1]
    first = float(np.max(np.linalg.norm(nominal.xy_slopes[-1, live, :2]-fine.xy_slopes[-1, live, :2], axis=1)))
    second = float(np.max(np.linalg.norm(fine.xy_slopes[-1, live, :2]-finest.xy_slopes[-1, live, :2], axis=1)))
    # Field expansion is examined on the SAME admitted physical trajectories,
    # separately from changing the numerical integration step. This sampled
    # field comparison does not claim a complete higher-order ray re-fit.
    observed = nominal.xy_slopes[:, :, :2]
    z = np.broadcast_to(nominal.z_m[:, None], nominal.alive.shape)
    points = np.dstack((observed, z))[nominal.alive]
    # Bound the diagnostic point count without choosing points by field error.
    points = points[::max(1, len(points)//1000)]
    fields = [snapshot.sample(points)["B_t"] for snapshot in
              (nominal_snapshot, _capture(state, 5)[0], _capture(state, 7)[0])]
    scale = float(np.linalg.norm(fields[-1]))
    order_change = [float(np.linalg.norm(a-b)/scale) for a, b in zip(fields, fields[1:])]
    record_testsuite_property("actual_probe_normal_independent_numerical_axes", json.dumps({
        "step_m": [.00025, .000125, .0000625], "successive_image_max_changes_m": [first, second],
        "field_radial_orders": [3, 5, 7], "successive_sampled_B_relative_changes": order_change,
        "field_sample_count": len(points), "field_order_scope": "same actual trajectories, not re-fitted Cs/Cc"}))
    assert second < .2*first and second < 2e-10
    assert order_change[1] < .2*order_change[0] and order_change[1] < 1e-5
