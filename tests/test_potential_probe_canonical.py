"""Actual static C3 + Probe canonical diagnostics on a controlled source sheet.

This is not a physical gun-population or complete electric-column qualification.
All 29 captured magnetic sources retain their native excitation. The synthetic
Lagrangian entrance at 600 mm is constructed once to focus in the nominal linear
map. No output chief, tilt or defocus is removed; perturbations reuse that fixed
sheet, physical apertures, vector-potential gauge and reference momentum.
"""
from functools import lru_cache
import json

import numpy as np
import pytest

from temsim.cpu_resources import numerical_job
from temsim.instrument_snapshot import encode_instrument, decode_instrument
from temsim.immutable_json import json_digest
from temsim.optics.aberration_basis import CARTESIAN_NAMES, predict_displacement_m
from temsim.optics.corrector_channels import CHANNELS, channel_control_specs, _control_values, _set_controls
from temsim.physics.potential_aberrations import (
    canonical_coordinates, capture_canonical_pupil, measure_canonical_aberrations,
    fit_canonical_intercepts,
)
from temsim.physics.potential_ray import trace_potential_rays
from test_potential_probe_pilot import MODES, _state, _capture


@pytest.fixture(scope="module", autouse=True)
def bounded_numerical_resources():
    # A small DOP853 state must not spawn a large BLAS team at every stage.
    with numerical_job(2):
        yield


@lru_cache(maxsize=2)
def _qualified(mode):
    state = _state(mode)
    original = encode_instrument(state)
    snapshot, nodes, stops = _capture(state)
    reference = capture_canonical_pupil(snapshot, nodes, energy_ev=300000.,
        maximum_step_m=.00025, stop_planes=stops, sampling_backend="native_compiled")
    report = measure_canonical_aberrations(snapshot, reference, retain_nominal_trajectory=True)
    assert json_digest(encode_instrument(state)) == json_digest(original)
    return state, snapshot, reference, report


def _evidence(report):
    return {key: report[key] for key in (
        "snapshot_digest", "reference_digest", "input_convention", "coefficient_names",
        "coefficients_mm", "constant_gradient_intercept_m", "cs_response_mm", "cc_response_mm",
        "energy_relative_steps", "gradient_training_rms_m", "gradient_holdout_rms_m",
        "finite_basis_gradient_qualification", "chromatic_qualification", "cc_convention")}


@pytest.mark.parametrize("mode", MODES)
def test_actual_canonical_gradient_and_frozen_field_energy_scan(mode, record_testsuite_property):
    state, snapshot, reference, report = _qualified(mode)
    evidence = _evidence(report)
    evidence.update({key: reference[key] for key in (
        "scaled_symplectic_defect", "input_lagrangian_relative_defect", "relative_transfer_change",
        "entrance_axis_B_t", "exit_chief_B_t", "source_canonical_curvature_m1")})
    record_testsuite_property("actual_canonical_"+mode, json.dumps(evidence))
    assert report["finite_basis_gradient_qualification"]["status"] == "PASS", evidence
    assert report["chromatic_qualification"]["status"] == "PASS", evidence
    assert report["chromatic_qualification"]["observable_budget_pass"] is True
    changes = np.abs(np.diff(report["cc_response_mm"]))
    assert changes[-1] < .4*changes[-2]
    assert report["chief_tilt_defocus_removed"] is False
    assert report["same_field_as_reference"] is True
    assert report["production_admission"] is False
    assert report["coherent_wave_admission"] is False
    assert reference["reference_was_symplectified"] is False
    assert len(snapshot.contributions) == 29
    for delta, mechanical in zip(report["relative_energy_deltas"], report["raw_input_mechanical"]):
        observed = canonical_coordinates(snapshot, mechanical, reference["z_m"][0],
            reference["energy_ev"]*(1+delta), reference["reference_momentum_kg_m_s"])
        np.testing.assert_allclose(observed, reference["input_canonical"], atol=1e-16, rtol=2e-14)


@pytest.mark.parametrize("mode", MODES)
def test_actual_fixed_mechanical_chromatic_comparison_is_separately_labelled(mode, record_testsuite_property):
    _, snapshot, reference, canonical = _qualified(mode)
    mechanical = measure_canonical_aberrations(snapshot, reference, input_convention="fixed_mechanical")
    record_testsuite_property("actual_fixed_mechanical_"+mode, json.dumps(_evidence(mechanical)))
    assert mechanical["finite_basis_gradient_qualification"]["status"] == "REJECTED"
    assert mechanical["chromatic_qualification"]["status"] == "REJECTED"
    assert mechanical["scalar_family_eligible"] is False
    for row in mechanical["raw_input_mechanical"]:
        np.testing.assert_array_equal(row, mechanical["raw_input_mechanical"][0])
    np.testing.assert_allclose(mechanical["raw_output_mechanical"][0],
                               canonical["raw_output_mechanical"][0], atol=1e-14, rtol=1e-9)


@pytest.mark.parametrize("mode", MODES)
def test_actual_canonical_field_order_and_integration_step_are_independent(mode, record_testsuite_property):
    state, snapshot, reference, nominal = _qualified(mode)
    step_reports = [nominal]+[measure_canonical_aberrations(snapshot, reference,
        maximum_step_m=step, energy_relative_steps=(.0005,)) for step in (.000125, .0000625)]
    field_reports = [nominal]+[measure_canonical_aberrations(_capture(state, order)[0], reference,
        energy_relative_steps=(.0005,)) for order in (5, 7)]
    def changes(reports):
        return [float(np.max(np.linalg.norm(a["raw_output_mechanical"][0, :, :2]
                        -b["raw_output_mechanical"][0, :, :2], axis=1)))
                for a, b in zip(reports, reports[1:])]
    step_changes, field_changes = changes(step_reports), changes(field_reports)
    evidence = {"step_m": [.00025, .000125, .0000625], "step_image_changes_m": step_changes,
        "step_cs_response_mm": [r["cs_response_mm"] for r in step_reports],
        "step_cc_response_mm": [r["cc_response_mm"][-1] for r in step_reports],
        "field_radial_orders": [3, 5, 7], "field_image_changes_m": field_changes,
        "field_cs_response_mm": [r["cs_response_mm"] for r in field_reports],
        "field_cc_response_mm": [r["cc_response_mm"][-1] for r in field_reports],
        "refinement_energy_derivative_step": .0005,
        "fixed_input_and_reference": True, "B0_renormalized": False}
    record_testsuite_property("actual_canonical_independent_axes_"+mode, json.dumps(evidence))
    # Strict adaptive integration can reach its error floor before max-step
    # changes become asymptotic; do not manufacture a convergence ratio there.
    assert max(step_changes) < 2e-10, evidence
    assert field_changes[-1] < .2*field_changes[0] and field_changes[-1] < 2e-10, evidence
    # Energy-derivative refinement fixes the same +/-0.0005 perturbation at
    # each numerical resolution. The nominal three-step scan establishes the
    # independent energy-step gate; a single pair never claims that alone.
    cc_step_error = abs(step_reports[-1]["cc_response_mm"][-1]-step_reports[-2]["cc_response_mm"][-1])
    cc_field_error = abs(field_reports[-1]["cc_response_mm"][-1]-field_reports[-2]["cc_response_mm"][-1])
    assert max(cc_step_error, cc_field_error)*1e-3*.01*.002 < 1e-9, evidence
    assert abs(field_reports[-1]["cs_response_mm"]-field_reports[-2]["cs_response_mm"]) < \
        1e-6+.01*abs(field_reports[-1]["cs_response_mm"]), evidence


def _continued_candidate(state, original, reference, baseline, changed_keys):
    snapshot, nodes, stops = _capture(state)
    before = {row.source_id: row.potential.digest for row in original.contributions}
    altered = {row.source_id for row in snapshot.contributions if row.potential.digest != before[row.source_id]}
    assert altered and all(any(source == key or source.startswith(key+":") for key in changed_keys)
                           for source in altered), altered
    def faces(captured):
        return [row.potential.base.field.native_support_mm[0]*.001 for row in captured.contributions
                if row.source_id in altered]
    first_effect = min(faces(original)+faces(snapshot))
    saved_z = baseline["nominal_trajectory_z_m"]
    index = int(np.flatnonzero(saved_z < first_effect)[-1])
    start = saved_z[index]
    run_nodes = np.unique(np.r_[start, nodes[nodes > start]])
    run_stops = tuple((int(np.flatnonzero(run_nodes == nodes[node])[0]), x, y, radius)
                     for node, x, y, radius in stops if nodes[node] >= start)
    result = trace_potential_rays(snapshot, baseline["nominal_trajectory_xy_slopes"][index], run_nodes,
        energy_ev=reference["energy_ev"], maximum_step_m=.00025, method="DOP853", stop_planes=run_stops,
        relative_tolerance=2e-11, position_tolerance_m=1e-15, slope_tolerance_rad=1e-13,
        sampling_backend="native_compiled")
    assert result.alive[-1].all()
    fitted = fit_canonical_intercepts(snapshot, reference, result.xy_slopes[-1])
    fitted["raw_output_mechanical"] = result.xy_slopes[-1]
    fitted["raw_action_difference_m"] = result.reduced_action_m[-1]-(
        baseline["nominal_trajectory_ray_action_m"][-1]-baseline["nominal_trajectory_ray_action_m"][index])
    fitted["unchanged_prefix_end_z_m"] = float(start)
    fitted["snapshot_digest"] = snapshot.numerical_digest
    fitted["changed_source_ids"] = sorted(altered)
    return fitted


@lru_cache(maxsize=2)
def _response(mode):
    state, snapshot, reference, baseline = _qualified(mode)
    # This is a private in-process fixture clone, not archived-run restoration.
    # Compare the complete input graph so unrelated concurrent source editing
    # cannot impersonate a changed physical control; production archive source
    # identity checks remain fully enabled and are tested independently.
    archive = encode_instrument(state)
    specs = tuple(spec for channel in CHANNELS for spec in channel_control_specs(state, "probe", channel))
    base = _control_values(state, specs)
    steps = np.array([spec.step for spec in specs])
    rows = []
    for column, spec in enumerate(specs):
        variants = []
        for fraction in (1., -1., .5, -.5):
            candidate = decode_instrument(archive)
            values = base.copy()
            values[column] += fraction*steps[column]
            _set_controls(candidate, specs, values)
            variants.append(_continued_candidate(candidate, snapshot, reference, baseline, (spec.key,)))
        rows.append(variants)
    full = np.column_stack([(row[0]["coefficients_mm"]-row[1]["coefficients_mm"])/(2*step)
                            for row, step in zip(rows, steps)])
    half = np.column_stack([(row[2]["coefficients_mm"]-row[3]["coefficients_mm"])/step
                            for row, step in zip(rows, steps)])
    intercept = np.column_stack([(row[2]["constant_gradient_intercept_m"]-row[3]["constant_gradient_intercept_m"])/step
                                for row, step in zip(rows, steps)])
    assert json_digest(archive) == json_digest(encode_instrument(state))
    return specs, base, steps, rows, full, half, intercept


@pytest.mark.parametrize("mode", MODES)
def test_actual_four_channels_keep_all_fourteen_drives_and_cross_coupling(mode, record_testsuite_property):
    state, snapshot, reference, baseline = _qualified(mode)
    specs, base, steps, rows, full, half, intercept = _response(mode)
    assert len(specs) == 14 and half.shape == (15, 14)
    assert np.isfinite(half).all()
    pupil = np.asarray(reference["holdout_reference_pupil"])
    change = np.array([np.linalg.norm(predict_displacement_m(pupil, column))
                      for column in ((full-half)*steps).T])
    response = np.array([np.linalg.norm(predict_displacement_m(pupil, column))
                        for column in (half*steps).T])
    channels, offset = {}, 0
    for channel in CHANNELS:
        count = len(channel_control_specs(state, "probe", channel))
        target = [CARTESIAN_NAMES.index(name) for name in
                  (("A1_x", "A1_y") if channel.startswith("a1") else ("A2_x", "A2_y"))]
        matrix = half[np.ix_(target, range(offset, offset+count))]*steps[offset:offset+count]
        singular = np.linalg.svd(matrix, compute_uv=False)
        uncertainty = np.linalg.norm((full-half)[np.ix_(target, range(offset, offset+count))]
                                    *steps[offset:offset+count], ord=2)
        channels[channel] = {"target_scaled_jacobian_mm": matrix.tolist(),
            "singular_values_mm": singular.tolist(), "half_step_uncertainty_mm": float(uncertainty),
            "two_axes_resolved": bool(singular[-1] > 10*uncertainty),
            "condition": float(singular[0]/singular[-1])}
        offset += count
    evidence = {"controls": [spec.__dict__ for spec in specs], "base_control_values": base.tolist(),
        "all_coefficients_jacobian": half.tolist(), "constant_intercept_jacobian_m": intercept.tolist(),
        "full_half_step_image_change_norm_m": change.tolist(),
        "per_step_image_response_norm_m": response.tolist(), "channels": channels,
        "max_gradient_holdout_rms_m": max(v["gradient_holdout_rms_m"] for row in rows for v in row),
        "calibration_applied_to_hardware": False, "OEM_assignment": False}
    record_testsuite_property("actual_canonical_14_drive_response_"+mode, json.dumps(evidence))
    assert np.all(response > 0.), evidence
    assert np.max(change) < 1e-9, evidence
    assert evidence["max_gradient_holdout_rms_m"] < 1e-9, evidence
    # Rank is evidence, never forced by regularization or an invented mapping.
    assert all(channel["two_axes_resolved"] for channel in channels.values()), evidence


@pytest.mark.parametrize("mode", MODES)
def test_actual_joint_small_control_holdout_predicts_raw_intercepts_without_recentering(mode, record_testsuite_property):
    state, snapshot, reference, baseline = _qualified(mode)
    specs, base, steps, _, _, jacobian, intercept_jacobian = _response(mode)
    fractions = .1*np.cos(np.arange(len(specs))*.7)
    delta = steps*fractions
    candidate = decode_instrument(encode_instrument(state))
    _set_controls(candidate, specs, base+delta)
    actual = _continued_candidate(candidate, snapshot, reference, baseline, {spec.key for spec in specs})
    coefficient = np.asarray(baseline["coefficients_mm"][0])+jacobian@delta
    intercept = np.asarray(baseline["constant_gradient_intercept_m"][0])+intercept_jacobian@delta
    holdout = actual["raw_output_canonical"][1+reference["training_count"]:]
    predicted = predict_displacement_m(holdout[:, 2:], coefficient)+intercept
    error = float(np.sqrt(np.mean((predicted-holdout[:, :2])**2)))
    raw_change = actual["raw_output_mechanical"]-baseline["raw_output_mechanical"][0]
    evidence = {"control_fractions": fractions.tolist(), "holdout_rms_m": error,
        "raw_max_position_change_m": float(np.max(np.linalg.norm(raw_change[:, :2], axis=1))),
        "raw_max_direction_change_rad": float(np.max(np.linalg.norm(raw_change[:, 2:], axis=1))),
        "raw_max_action_change_m": float(np.max(np.abs(actual["raw_action_difference_m"]))),
        "predicted_all_coefficients_mm": coefficient.tolist(),
        "observed_all_coefficients_mm": actual["coefficients_mm"].tolist(),
        "constant_intercept_m": actual["constant_gradient_intercept_m"].tolist(),
        "chief_tilt_defocus_removed": False, "wave_phase_qualification": False}
    record_testsuite_property("actual_canonical_joint_control_holdout_"+mode, json.dumps(evidence))
    assert error < 1e-9, evidence
    assert evidence["raw_max_position_change_m"] > 1e-12
    assert evidence["raw_max_action_change_m"] > 1e-20


@pytest.mark.parametrize("mode", MODES)
def test_actual_hexapole_thickness_changes_canonical_aberrations_at_fixed_pupil(mode, record_testsuite_property):
    state, snapshot, reference, baseline = _qualified(mode)
    candidate = decode_instrument(encode_instrument(state))
    element = next(item for item in candidate.corrector_elements if item.key == "probe_hp2_hexapole")
    old_length = element.effective_length_mm
    element.effective_length_mm = old_length+1.2
    actual = _continued_candidate(candidate, snapshot, reference, baseline, (element.key,))
    difference = actual["coefficients_mm"]-np.asarray(baseline["coefficients_mm"][0])
    evidence = {"before_length_mm": old_length, "after_length_mm": element.effective_length_mm,
        "coefficient_changes_mm": difference.tolist(), "coefficient_names": list(CARTESIAN_NAMES),
        "gradient_holdout_rms_m": actual["gradient_holdout_rms_m"],
        "raw_max_action_change_m": float(np.max(np.abs(actual["raw_action_difference_m"]))),
        "fixed_hardware_excitation": True, "refocused": False}
    record_testsuite_property("actual_canonical_hp2_thickness_"+mode, json.dumps(evidence))
    assert abs(difference[CARTESIAN_NAMES.index("C3")]) > 1e-4, evidence
    assert evidence["raw_max_action_change_m"] > 1e-20
