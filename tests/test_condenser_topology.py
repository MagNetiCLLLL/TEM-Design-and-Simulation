"""Condenser topology is physical ray evidence, not a mode-name assumption."""
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.condenser_topology import (
    LARGE_ANGLE_REQUIREMENT,
    REQUIREMENT,
    evaluate,
    is_applicable,
    requirement_for_state,
    requested_capture_planes,
    validate,
)


def state(shift=0., *, nanoprobe_range="normal"):
    return SimpleNamespace(
        nanoprobe_range=nanoprobe_range,
        lenses=[SimpleNamespace(key=key, z_mm=z+shift, enabled=True)
                for key, z in (("condenser_lens_2", 10.),
                               ("condenser_lens_3", 20.), ("adapter_lens", 30.))],
        electron_gun=SimpleNamespace(exit_plane_z_mm=shift),
        sample=SimpleNamespace(upper_surface_z_mm=40.+shift))


def phase_space(z, *, root=15., second=None, rays=16):
    z = np.asarray(z, float)
    phi = np.arange(rays)*2*np.pi/rays
    position = (z-root)*1e-6
    slope = np.full(z.size, .001)
    if second is not None:
        position *= z-second
        slope = (2*z-root-second)*.001
    cp = SimpleNamespace(z_mm=z,
        x_m=position[:, None]*np.cos(phi) + 1e-8*np.cos(2*phi),
        y_m=position[:, None]*np.sin(phi) + 1e-8*np.sin(2*phi),
        tx_rad=slope[:, None]*np.cos(phi), ty_rad=slope[:, None]*np.sin(phi))
    return cp, np.ones(cp.x_m.shape, bool), np.ones(rays)


def measure(*, root=15., second=None, rays=16, nanoprobe_range="normal"):
    s = state(nanoprobe_range=nanoprobe_range)
    cp, masks, weights = phase_space(requested_capture_planes(s), root=root,
                                     second=second, rays=rays)
    return s, cp, masks, weights


def test_capture_planes_follow_live_centres_and_bound_the_spacing():
    planes = np.array(requested_capture_planes(state(), maximum_spacing_mm=.7))
    assert all(z in planes for z in (10., 20., 30.))
    assert planes[0] < 10. and planes[-1] > 30.
    assert np.max(np.diff(planes)) <= .7 + 1e-12
    shifted = requested_capture_planes(state(43.25), maximum_spacing_mm=.7)
    assert shifted == pytest.approx(planes+43.25)


def test_applicable_invalid_order_is_rejected_instead_of_bypassed():
    s, cp, masks, weights = measure()
    s.lenses[1].z_mm = 8.
    assert is_applicable(s)
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["failures"] == ["invalid_geometry_or_checkpoints"]
    s.lenses[2].enabled = False
    assert not is_applicable(s)


def test_ordinary_path_needs_an_executed_c2_c3_waist():
    s, cp, masks, weights = measure()
    result = evaluate(s, cp, masks, weights)
    assert result["status"] == "PASS"
    assert result["passed"]
    assert result["intervals"]["c2_c3"]["waists"][0]["z_mm"] == pytest.approx(15.)
    assert result["intervals"]["c3_adl"]["candidate_count"] == 0
    assert result["cohort"]["ray_count"] == 16


def test_extra_c3_adl_waist_rejects_otherwise_valid_c2_c3_waist():
    s, cp, masks, weights = measure(second=25.)
    result = evaluate(s, cp, masks, weights)
    assert result["status"] == "FAIL"
    assert "forbidden_c3_adl_waist" in result["failures"]
    assert result["intervals"]["c2_c3"]["waists"]
    assert result["intervals"]["c3_adl"]["candidate_count"] == 1


@pytest.mark.parametrize("nanoprobe_range,second,passed,failure", [
    ("normal", None, True, None),
    ("normal", 25., False, "forbidden_c3_adl_waist"),
    ("large_angle", None, False, "missing_c3_adl_waist"),
    ("large_angle", 25., True, None),
])
def test_range_selects_physical_topology_without_disabling_the_gate(
    nanoprobe_range, second, passed, failure,
):
    s, cp, masks, weights = measure(second=second, nanoprobe_range=nanoprobe_range)
    result = evaluate(s, cp, masks, weights)
    assert result["passed"] is passed
    assert result["requirement"] == (LARGE_ANGLE_REQUIREMENT
        if nanoprobe_range == "large_angle" else REQUIREMENT)
    assert result["nanoprobe_range"] == nanoprobe_range
    assert result["failures"] == ([] if passed else [failure])
    if passed and nanoprobe_range == "large_angle":
        assert result["intervals"]["c3_adl"]["waists"][0]["z_mm"] == pytest.approx(25.)
        assert result["intervals"]["c3_adl"]["waists"][0]["status"] == "EXECUTED_AT_FIXED_SAMPLING"


def test_large_angle_still_requires_the_first_waist():
    s, cp, masks, weights = measure(root=25., nanoprobe_range="large_angle")
    assert evaluate(s, cp, masks, weights)["failures"] == ["missing_c2_c3_waist"]


def test_unknown_range_is_not_silently_treated_as_normal():
    s, cp, masks, weights = measure(nanoprobe_range="unknown")
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["requirement"] is None
    assert result["failures"] == ["invalid_geometry_or_checkpoints"]


def test_legacy_state_without_a_range_uses_normal():
    s, cp, masks, weights = measure()
    del s.nanoprobe_range
    result = evaluate(s, cp, masks, weights)
    assert result["passed"]
    assert result["requirement"] == REQUIREMENT


def test_explicit_large_angle_legacy_observer_matches_the_mode_helper():
    from temsim.operating_modes import condenser_mode_key

    s, cp, masks, weights = measure(second=25.)
    del s.nanoprobe_range
    s.probe_mode = "nano_probe_large_angle"
    assert condenser_mode_key(s) == "nano_probe_large_angle"
    assert requirement_for_state(s) == LARGE_ANGLE_REQUIREMENT
    result = evaluate(s, cp, masks, weights)
    assert result["passed"]
    assert result["nanoprobe_range"] == "large_angle"


def test_conflicting_explicit_large_angle_observer_is_rejected():
    s, cp, masks, weights = measure(second=25.)
    s.probe_mode = "nano_probe_large_angle"
    with pytest.raises(ValueError, match="Conflicting"):
        requirement_for_state(s)
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["failures"] == ["invalid_geometry_or_checkpoints"]


def test_large_angle_line_focus_in_second_interval_is_not_a_2d_waist():
    s, cp, masks, weights = measure(second=25., nanoprobe_range="large_angle")
    second = cp.z_mm >= 20.
    cp.y_m[second] = 1e-7*np.sin(np.arange(weights.size)*2*np.pi/weights.size)
    cp.ty_rad[second] = 0.
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["intervals"]["c2_c3"]["waists"]
    assert "unexecuted_c3_adl_waist" in result["failures"]


@pytest.mark.parametrize("nanoprobe_range", ["normal", "large_angle"])
@pytest.mark.parametrize("root", [10., 20., 30., 20.+1e-8, 20.-1e-8])
def test_boundary_waist_never_disappears_between_open_intervals(root, nanoprobe_range):
    s, cp, masks, weights = measure(root=root, nanoprobe_range=nanoprobe_range)
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["status"] == "UNRESOLVED"
    assert "crossover_on_component_boundary" in result["failures"]


def test_narrow_between_node_waist_requests_actual_planes_before_acceptance():
    s, cp, masks, weights = measure(root=15.5)
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert "unexecuted_c2_c3_waist" in result["failures"]
    assert result["refinement_planes_mm"] == pytest.approx([15.49, 15.5, 15.51])
    planes = np.unique(np.r_[cp.z_mm, result["refinement_planes_mm"]])
    refined, masks, weights = phase_space(planes, root=15.5)
    assert evaluate(s, refined, masks, weights)["passed"]


@pytest.mark.parametrize("angle", [0., .63])
def test_a_line_focus_is_not_accepted_as_a_two_dimensional_waist(angle):
    s, cp, masks, weights = measure()
    cp.y_m[:] = 1e-7*np.sin(np.arange(weights.size)*2*np.pi/weights.size)
    cp.ty_rad[:] = 0.
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    cp.x_m, cp.y_m = np.einsum("ab,bzr->azr", rotation, [cp.x_m, cp.y_m])
    cp.tx_rad, cp.ty_rad = np.einsum("ab,bzr->azr", rotation, [cp.tx_rad, cp.ty_rad])
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert "unexecuted_c2_c3_waist" in result["failures"]


def test_shared_candidate_solver_retains_waists_on_executed_bracket_endpoints():
    from temsim.optics.beam_path_audit import crossover_candidates
    # Both exact roots have tiny finite-source covariances. Their sign must be
    # evaluated consistently between selecting and solving the bracket.
    _, cp, masks, weights = measure(second=25.)
    roots = crossover_candidates(cp, masks, weights)
    assert [row["z_mm"] for row in roots] == pytest.approx([15., 25.])


def test_rotation_chief_ray_and_weighted_ray_splitting_preserve_evidence():
    s, cp, masks, weights = measure()
    weights[::2] = 2.
    reference = evaluate(s, cp, masks, weights)
    rotation = np.array([[np.cos(.63), -np.sin(.63)],
                         [np.sin(.63), np.cos(.63)]])
    cp.x_m, cp.y_m = np.einsum("ab,bzr->azr", rotation, [cp.x_m, cp.y_m])
    cp.tx_rad, cp.ty_rad = np.einsum("ab,bzr->azr", rotation, [cp.tx_rad, cp.ty_rad])
    cp.x_m += cp.z_mm[:, None]*.001
    cp.tx_rad += 1.
    repeats = np.arange(weights.size) % 3 + 1
    for key in ("x_m", "y_m", "tx_rad", "ty_rad"):
        setattr(cp, key, np.repeat(getattr(cp, key), repeats, axis=1))
    result = evaluate(s, cp, np.repeat(masks, repeats, axis=1),
                      np.repeat(weights/repeats, repeats))
    assert result["passed"]
    before = reference["intervals"]["c2_c3"]["waists"][0]
    after = result["intervals"]["c2_c3"]["waists"][0]
    assert after["z_mm"] == pytest.approx(before["z_mm"])
    assert after["rms_radius_nm"] == pytest.approx(before["rms_radius_nm"])


def test_changing_clipped_population_cannot_manufacture_required_waist():
    s, core, masks, weights = measure(root=5.)
    halo, _, _ = phase_space(core.z_mm, root=15.)
    for key in ("x_m", "y_m", "tx_rad", "ty_rad"):
        setattr(core, key, np.c_[getattr(core, key), getattr(halo, key)*100])
    masks = np.ones(core.x_m.shape, bool)
    masks[core.z_mm >= 15., 16:] = False
    result = evaluate(s, core, masks, np.ones(32))
    assert result["failures"] == ["missing_c2_c3_waist"]
    assert result["cohort"]["ray_count"] == 16


@pytest.mark.parametrize("nanoprobe_range", ["normal", "large_angle"])
@pytest.mark.parametrize("damage, expected", [
    ("missing_boundary", "missing_boundary_checkpoints"),
    ("sparse", "insufficient_axial_sampling"),
    ("few_rays", "insufficient_surviving_rays"),
    ("nonfinite", "nonfinite_surviving_coordinates"),
    ("negative_weight", "invalid_checkpoint_arrays"),
    ("resurrection", "inconsistent_survivor_masks"),
])
def test_missing_or_invalid_evidence_fails_closed(damage, expected, nanoprobe_range):
    s, cp, masks, weights = measure(nanoprobe_range=nanoprobe_range,
                                  second=25. if nanoprobe_range == "large_angle" else None)
    if damage in {"missing_boundary", "sparse"}:
        keep = (cp.z_mm != 20.) if damage == "missing_boundary" else (cp.z_mm != 16.)
        cp.z_mm = cp.z_mm[keep]
        for key in ("x_m", "y_m", "tx_rad", "ty_rad"):
            setattr(cp, key, getattr(cp, key)[keep])
        masks = masks[keep]
    elif damage == "few_rays":
        weights[4:] = 0.
    elif damage == "nonfinite":
        cp.x_m[3, 0] = np.nan
    elif damage == "negative_weight":
        weights[0] = -1.
    else:
        masks[3, 0] = False
    result = evaluate(s, cp, masks, weights)
    assert not result["passed"]
    assert result["status"] == "UNRESOLVED"
    assert result["failures"] == [expected]


def test_later_specimen_losses_do_not_erase_upstream_condenser_evidence():
    s = state()
    planes = np.r_[requested_capture_planes(s), 39.]
    cp, masks, weights = phase_space(planes)
    masks[-1, :] = False
    assert evaluate(s, cp, masks, weights)["passed"]


def test_validate_executes_requested_refinement_with_the_same_backend_state(monkeypatch):
    from temsim.optics import beam_path_audit
    s = state()
    calls = []

    def execute(candidate, planes, *, step_mm):
        assert candidate is s
        assert step_mm == .1
        calls.append(planes)
        cp, masks, weights = phase_space(planes, root=15.5)
        return SimpleNamespace(exit_bundle=SimpleNamespace(weight=weights)), cp, masks

    monkeypatch.setattr(beam_path_audit, "incident_checkpoints", execute)
    result = validate(s, step_mm=.1)
    assert result["passed"]
    assert result["executions"] == 2
    assert len(calls) == 2
    assert 15.5 not in calls[0]
    assert 15.5 in calls[1]


def test_large_angle_refines_both_required_waists_with_a_bounded_execution_budget(monkeypatch):
    from temsim.optics import beam_path_audit
    s = state(nanoprobe_range="large_angle")
    calls = []

    def execute(candidate, planes, *, step_mm):
        assert candidate is s
        calls.append(planes)
        cp, masks, weights = phase_space(planes, root=15.5, second=25.5)
        # Equal minima on neighbouring coarse nodes require real refinement
        # in both intervals; the quadratic fixture has asymmetric brackets.
        phi = np.arange(weights.size)*2*np.pi/weights.size
        phase = (cp.z_mm-15.5)*np.pi/10
        position = np.sin(phase)*1e-6
        slope = np.cos(phase)*np.pi/10*.001
        cp.x_m = position[:, None]*np.cos(phi) + 1e-8*np.cos(2*phi)
        cp.y_m = position[:, None]*np.sin(phi) + 1e-8*np.sin(2*phi)
        cp.tx_rad = slope[:, None]*np.cos(phi)
        cp.ty_rad = slope[:, None]*np.sin(phi)
        return SimpleNamespace(exit_bundle=SimpleNamespace(weight=weights)), cp, masks

    monkeypatch.setattr(beam_path_audit, "incident_checkpoints", execute)
    result = validate(s, step_mm=.1)
    assert result["passed"]
    assert result["requirement"] == LARGE_ANGLE_REQUIREMENT
    assert 1 < result["executions"] <= 3
    assert any(15.4 < z < 15.6 for z in calls[-1])
    assert any(25.4 < z < 25.6 for z in calls[-1])
    assert all(result["intervals"][key]["waists"] for key in ("c2_c3", "c3_adl"))


def test_validate_does_not_accept_execution_failure(monkeypatch):
    from temsim.optics import beam_path_audit

    def execute(*args, **kwargs):
        raise ValueError("Missing actual checkpoint")

    monkeypatch.setattr(beam_path_audit, "incident_checkpoints", execute)
    result = validate(state())
    assert not result["passed"]
    assert result["failures"] == ["execution_failed"]
