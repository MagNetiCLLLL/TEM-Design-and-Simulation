"""Bounded analytic checks for the actual-crossover qualification script."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.surface_probe_focus import SurfaceFocusMeasurement
from temsim.physics.beam_statistics import transverse_beam_statistics


_spec = spec_from_file_location("finite_field_nanoprobe_qualification",
    Path(__file__).resolve().parents[1] / "scripts" / "qualify_finite_field_nanoprobe.py")
qualification = module_from_spec(_spec)
_spec.loader.exec_module(qualification)
KEYS = ("x_m", "y_m", "tx_rad", "ty_rad")


def regions():
    return dict(central=dict(bounds_mm=[2., 8.], maximum_spacing_mm=1., geometric_midplane_mm=5.),
                ac=dict(bounds_mm=[18., 22.], maximum_spacing_mm=.1, geometric_midplane_mm=20.))


def drift_pair(*, central_root=4., ac_ratio=1., rays=16):
    z = np.asarray(qualification.requested_corrector_crossover_planes(regions()))
    phi = np.arange(rays)*2*np.pi/rays
    root = np.where(z < 15., central_root, 20.)
    angle = np.where(z < 15., .002, .002*ac_ratio)
    tx = angle[:, None]*np.cos(phi)
    ty = angle[:, None]*np.sin(phi)
    dz = (z-root)[:, None]*1e-3
    cp = SimpleNamespace(z_mm=z,
        x_m=dz*(tx+.004)+1e-8*np.cos(2*phi)+1e-6,
        y_m=dz*(ty-.003)+1e-8*np.sin(2*phi)-2e-6,
        tx_rad=tx+.004, ty_rad=ty-.003)
    return cp, np.ones(cp.x_m.shape, bool), np.ones(rays), np.arange(rays)+100


def evaluate(cp, masks, weights, identities, **kwargs):
    return qualification.evaluate_corrector_crossovers(cp, masks, weights, regions(),
        source_current_a=5e-9, source_ids=identities, **kwargs)


def test_capture_uses_live_lens_and_physical_coil_centers(monkeypatch):
    state = SimpleNamespace(lenses=[SimpleNamespace(key=key, z_mm=z, enabled=True)
        for key, z in (("probe_tl22_lens", 2.), ("probe_tl21_lens", 8.))],
        electron_gun=SimpleNamespace(exit_plane_z_mm=0.), sample=SimpleNamespace(upper_surface_z_mm=30.))
    coils = [SimpleNamespace(key=f"ac_deflector:{i}", arrival_z_mm=z, dynamic=False)
             for i, z in enumerate((18., 22.))]
    monkeypatch.setattr("temsim.physics.instrument_magnetic.column_dipole_fields", lambda state: coils)
    found = qualification.corrector_crossover_regions(state)
    assert found == regions()
    requested = np.asarray(qualification.requested_corrector_crossover_planes(found))
    for row in found.values():
        low, high = row["bounds_mm"]
        selected = requested[(requested >= low) & (requested <= high)]
        assert selected[[0, -1]] == pytest.approx([low, high])
        assert np.diff(selected).max() <= row["maximum_spacing_mm"]+1e-12
    coils[0].dynamic = True
    with pytest.raises(ValueError, match="static physical AC coils"):
        qualification.corrector_crossover_regions(state)


def test_two_executed_drift_waists_match_without_forcing_geometric_midpoint():
    cp, masks, weights, identities = drift_pair()
    originals = [getattr(cp, key).copy() for key in KEYS]
    report = evaluate(cp, masks, weights, identities)
    assert report["passed"] and report["status"] == "PASS"
    central, ac = (report["intervals"][key]["executed_waists"][0] for key in ("central", "ac"))
    assert central["z_mm"] == 4. and central["bracket_mm"] == [3., 5.]
    assert central["geometric_midplane_offset_mm"] == -1.
    assert ac["z_mm"] == 20.
    assert report["common"]["ac_over_central_ratio"] == pytest.approx(1.)
    assert central["actual"]["mean_x_m"] == pytest.approx(1e-6)
    assert central["actual"]["mean_y_m"] == pytest.approx(-2e-6)
    # Mean direction averages normalized 3D directions, with a small finite-
    # cone correction relative to the arithmetic mean of mechanical slopes.
    assert central["actual"]["mean_tx_rad"] == pytest.approx(.004, abs=1e-8)
    assert central["current_a"] == pytest.approx(5e-9)
    assert report["common_cohort"]["source_ids"] == identities.tolist()
    for key, original in zip(KEYS, originals):
        np.testing.assert_array_equal(getattr(cp, key), original)
    np.testing.assert_array_equal(weights, np.ones(16))


@pytest.mark.parametrize("ratio,tolerance,passed", [(1.005, .01, True), (1.005, .001, False), (.5, .01, False)])
def test_angle_tolerance_controls_both_measured_populations(ratio, tolerance, passed):
    report = evaluate(*drift_pair(ac_ratio=ratio), angle_relative_tolerance=tolerance,
                      require_equal_crossover_angles=True)
    assert report["passed"] is passed
    assert report["actual"]["passed"] is passed
    assert report["common"]["passed"] is passed


def test_interpolated_candidate_without_executed_minimum_cannot_pass():
    report = evaluate(*drift_pair(central_root=4.5))
    assert not report["passed"]
    central = report["intervals"]["central"]
    assert central["interpolated_candidates_diagnostic_only"]
    assert central["executed_waists"] == []
    assert "missing_or_ambiguous_executed_central_2d_waist" in report["failures"]


def test_line_focus_cannot_replace_two_dimensional_waist():
    cp, masks, weights, identities = drift_pair()
    selected = cp.z_mm < 15.
    cp.y_m[selected] = -2e-6 + 1e-8*np.sin(2*np.arange(16)*2*np.pi/16)
    cp.ty_rad[selected] = 0.
    report = evaluate(cp, masks, weights, identities)
    assert not report["passed"]
    assert report["intervals"]["central"]["executed_waists"] == []


@pytest.mark.parametrize("missing", [2., 6., 18., 20.5])
def test_missing_boundary_or_excessive_spacing_fails_closed(missing):
    cp, masks, weights, identities = drift_pair()
    keep = ~np.isclose(cp.z_mm, missing, rtol=0, atol=1e-12)
    cp = SimpleNamespace(z_mm=cp.z_mm[keep], **{key: getattr(cp, key)[keep] for key in KEYS})
    report = evaluate(cp, masks[keep], weights, identities)
    assert not report["passed"]
    assert any("checkpoint_coverage" in failure for failure in report["failures"])


def test_clipping_does_not_hide_actual_current_or_angle_change():
    cp, _, _, _ = drift_pair()
    for key in KEYS:
        a = getattr(cp, key)
        chief = a.mean(axis=1, keepdims=True)
        setattr(cp, key, np.concatenate((a, chief+2*(a-chief)), axis=1))
    masks = np.ones(cp.x_m.shape, bool)
    masks[cp.z_mm >= 18., 16:] = False
    weights = np.r_[np.full(16, .02), np.full(16, .08)]
    original_weights = weights.copy()
    report = evaluate(cp, masks, weights, np.arange(32), require_equal_crossover_angles=True)
    assert not report["passed"]
    assert report["common"]["passed"]
    assert not report["actual"]["passed"]
    assert report["common_cohort"]["source_current_fraction"] == pytest.approx(.2)
    assert report["common_cohort"]["original_weights"] == pytest.approx(weights[:16])
    central, ac = (report["intervals"][key]["executed_waists"][0] for key in ("central", "ac"))
    assert central["current_a"] == pytest.approx(5e-9)
    assert ac["current_a"] == pytest.approx(1e-9)
    assert central["common_current_a"] == ac["common_current_a"] == pytest.approx(1e-9)
    np.testing.assert_array_equal(weights, original_weights)


def test_clipped_focused_halo_cannot_create_common_cohort_waist():
    core, _, _, _ = drift_pair(central_root=-1.)
    halo, _, _, _ = drift_pair()
    for key in KEYS:
        setattr(core, key, np.concatenate((getattr(core, key), 100*getattr(halo, key)), axis=1))
    masks = np.ones(core.x_m.shape, bool)
    masks[core.z_mm >= 4., 16:] = False
    report = evaluate(core, masks, np.ones(32), np.arange(32))
    assert not report["passed"]
    assert report["intervals"]["central"]["executed_waists"] == []
    assert report["common_cohort"]["ray_count"] == 16


def test_ray_splitting_preserves_weighted_measurement_and_original_total():
    cp, masks, weights, identities = drift_pair()
    weights[::2] = 3.
    before = evaluate(cp, masks, weights, identities)
    repeat = np.arange(16) % 3 + 1
    split = SimpleNamespace(z_mm=cp.z_mm, **{key: np.repeat(getattr(cp, key), repeat, axis=1) for key in KEYS})
    after = evaluate(split, np.repeat(masks, repeat, axis=1),
        np.repeat(weights/repeat, repeat), np.arange(repeat.sum()))
    assert before["passed"] and after["passed"]
    assert after["common"]["alpha95_mrad"] == pytest.approx(before["common"]["alpha95_mrad"])
    assert after["common_cohort"]["source_weight"] == pytest.approx(weights.sum())


@pytest.mark.parametrize("damage", ["negative_weight", "nonfinite_alive", "reappearing_ray", "few_rays"])
def test_invalid_or_unresolved_particle_evidence_cannot_pass(damage):
    cp, masks, weights, identities = drift_pair(rays=4 if damage == "few_rays" else 16)
    if damage == "negative_weight":
        weights[0] = -1.
    elif damage == "nonfinite_alive":
        cp.x_m[2, 0] = np.nan
    elif damage == "reappearing_ray":
        masks[2, 0] = False
    report = evaluate(cp, masks, weights, identities)
    assert not report["passed"] and report["failures"]


def test_small_probe_and_independent_direct_focus_and_corrector_gates():
    phi = np.arange(16)*2*np.pi/16
    statistics = transverse_beam_statistics(1e-10*np.cos(2*phi), 1e-10*np.sin(2*phi),
        np.tan(.03)*np.cos(phi), np.tan(.03)*np.sin(phi), weights=np.ones(16))
    measurement = SurfaceFocusMeasurement(100., statistics, 0., 1., (2e-10, 1.5e-10, 1e-10), .05, 16.)
    assert 2*statistics.radius_95_m*1e9 < .7
    corrector = {"waist_evidence_passed": True, "angle_matching_passed": False}
    topology = {"passed": True, "direct_focus": {"status": "PASS"}}
    assert qualification.qualification_failures(topology, measurement, corrector, 30.) == []
    assert qualification.qualification_failures({"passed": True}, measurement, corrector, 30.) == [
        "Direct Mini Condenser/objective focus topology failed or is unresolved"]
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        require_equal_crossover_angles=True) == ["Explicit equal-angle crossover calibration constraint failed"]
    failures = qualification.qualification_failures(topology, measurement, {"waist_evidence_passed": False}, 30.)
    assert failures == ["Executed internal and AC crossover evidence failed or is unresolved"]


def finite_direct_focus_inputs():
    measurement = SimpleNamespace(accepts=lambda _: True,
        statistics=SimpleNamespace(radius_95_m=.2e-9))
    corrector = dict(waist_detector="finite_region", waist_evidence_passed=True)
    topology = dict(passed=True, direct_focus=dict(status="PASS"))
    chain = dict(required_topology_satisfied_at_sampling=True,
        unattributed_population_change_count=0,
        direct_focus=dict(schema="direct-focus-finite-region-topology-v2",
            requirement="direct_focus_without_extra_pre_specimen_waist",
            status="PASS", passed=True, failures=[]))
    return topology, measurement, corrector, chain


def test_finite_direct_focus_is_independent_of_unresolved_condenser_summary():
    topology, measurement, corrector, chain = finite_direct_focus_inputs()
    topology = dict(passed=False, status="UNRESOLVED")
    chain["required_topology_satisfied_at_sampling"] = False
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain) == [
            "Required source-image regions failed or have incomplete whole-capture evidence",
            "Condenser topology failed or is unresolved"]


@pytest.mark.parametrize("evidence", [None, {}, "PASS",
    dict(schema="direct-focus-topology-v1", status="PASS", passed=True, failures=[])])
def test_finite_direct_focus_missing_or_unknown_authority_cannot_fall_back(evidence):
    topology, measurement, corrector, chain = finite_direct_focus_inputs()
    chain["direct_focus"] = evidence
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain) == [
            "Direct Mini Condenser/objective focus topology failed or is unresolved"]
    del chain["direct_focus"]
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain)


@pytest.mark.parametrize("key,value", [("passed", False), ("passed", 1),
    ("status", "UNRESOLVED"), ("status", "FAIL"), ("status", "UNKNOWN"),
    ("schema", None), ("requirement", None), ("failures", ["unresolved"]), ("failures", None)])
def test_finite_direct_focus_contradiction_or_unresolved_cannot_pass(key, value):
    topology, measurement, corrector, chain = finite_direct_focus_inputs()
    chain["direct_focus"][key] = value
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain) == [
            "Direct Mini Condenser/objective focus topology failed or is unresolved"]


def test_finite_authoritative_focus_pass_preserves_other_physical_failures():
    topology, measurement, corrector, chain = finite_direct_focus_inputs()
    del topology["direct_focus"]
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain) == []
    measurement.statistics.radius_95_m = .35e-9
    measurement.accepts = lambda _: False
    corrector["waist_evidence_passed"] = False
    chain["unattributed_population_change_count"] = 1
    assert qualification.qualification_failures(topology, measurement, corrector, 30.,
        crossover_chain=chain) == [
            "Unattributed physical interception remains unresolved in the incident capture",
            "Surface focus, convergence angle or effective sample support failed",
            "Surface D95 must be strictly below the declared diameter limit",
            "Executed internal and AC crossover evidence failed or is unresolved"]


@pytest.mark.parametrize("region,spacing", [("central", 1.01), ("ac", .101)])
def test_declared_spacing_cannot_relax_required_coverage(region, spacing):
    declared = regions()
    declared[region]["maximum_spacing_mm"] = spacing
    with pytest.raises(ValueError, match="observation intervals"):
        qualification.requested_corrector_crossover_planes(declared)


@pytest.mark.parametrize("tolerance", [float("nan"), -.01, 1.])
def test_invalid_angle_tolerance_fails_before_any_acceptance(tolerance):
    report = evaluate(*drift_pair(), angle_relative_tolerance=tolerance)
    assert not report["passed"]
    assert report["failures"] == ["invalid_checkpoint_arrays_or_parameters"]


def test_equal_angles_cannot_hide_a_diverging_full_beam_at_the_common_waist():
    phi = np.arange(8)*np.pi/4
    sx, sy = np.cos(phi)*.001, np.sin(phi)*.001
    z = np.r_[np.arange(5.), np.linspace(10., 10.4, 5)]
    cp = SimpleNamespace(z_mm=z, x_m=np.empty((10, 16)), y_m=np.empty((10, 16)),
        tx_rad=np.tile(np.r_[sx, sx], (10, 1)), ty_rad=np.tile(np.r_[sy, sy], (10, 1)))
    for row, plane in enumerate(z):
        center = 2. if plane < 5 else 10.2
        cp.x_m[row] = np.r_[(plane-center)*1e-3*sx, (plane+5)*1e-3*sx]
        cp.y_m[row] = np.r_[(plane-center)*1e-3*sy, (plane+5)*1e-3*sy]
    masks = np.ones((10, 16), bool)
    masks[z > 5, 8:] = False
    weights = np.full(16, 1/16)
    declared = dict(central=dict(bounds_mm=[0., 4.], maximum_spacing_mm=1., geometric_midplane_mm=2.),
                    ac=dict(bounds_mm=[10., 10.4], maximum_spacing_mm=.1, geometric_midplane_mm=10.2))
    report = qualification.evaluate_corrector_crossovers(cp, masks, weights, declared, source_current_a=1e-9)
    assert not report["passed"]
    waist, = report["intervals"]["central"]["executed_waists"]
    assert waist["actual_waist_support"]["positive_current_membership_stable"]
    assert waist["actual_waist_support"]["status"] == "FAIL"
    assert waist["actual"]["radial_position_angle_covariance_m_rad"] == pytest.approx(3.5e-9)
    assert waist["actual"]["convergence_95_rad"] == pytest.approx(waist["common"]["convergence_95_rad"])
    assert "missing_central_actual_2d_waist_at_common_bracket" in report["failures"]


def test_changing_actual_membership_inside_waist_bracket_is_unresolved():
    cp, _, _, _ = drift_pair()
    for key in KEYS:
        setattr(cp, key, np.tile(getattr(cp, key), (1, 2)))
    masks = np.ones(cp.x_m.shape, bool)
    masks[cp.z_mm >= 4., 16:] = False
    weights = np.ones(32)
    report = evaluate(cp, masks, weights, np.arange(32))
    assert not report["passed"] and report["status"] == "UNRESOLVED"
    assert "unresolved_central_actual_waist_cohort_changed" in report["failures"]
    waist, = report["intervals"]["central"]["executed_waists"]
    assert not waist["actual_waist_support"]["positive_current_membership_stable"]
    assert [row["current_a"] for row in waist["executed_bracket"]] == pytest.approx([5e-9, 2.5e-9, 2.5e-9])
    np.testing.assert_array_equal(weights, np.ones(32))


@pytest.mark.parametrize("required,passed", [(False, True), (True, False)])
def test_nonunit_angular_magnification_is_diagnostic_unless_explicitly_required(required, passed):
    report = evaluate(*drift_pair(ac_ratio=.5), require_equal_crossover_angles=required)
    assert report["waist_evidence_passed"]
    assert not report["angle_matching_passed"]
    assert report["passed"] is passed
    assert report["require_equal_crossover_angles"] is required
    assert report["diagnostic_findings"]
    assert bool(report["failures"]) is required
    assert report["actual"]["ac_over_central_ratio"] == pytest.approx(.5, abs=1e-5)


def test_optional_calibration_requirement_cannot_be_a_truthy_string():
    report = evaluate(*drift_pair(), require_equal_crossover_angles="false")
    assert not report["passed"] and not report["waist_evidence_passed"]
    assert report["failures"] == ["invalid_checkpoint_arrays_or_parameters"]


@pytest.fixture
def capture_observer(monkeypatch):
    """Run report assembly on synthetic checkpoints, without any transport."""
    import temsim.instrument_snapshot as snapshots
    import temsim.calculation_manifest as manifest
    import temsim.optics.beam_path_audit as beam_audit
    import temsim.optics.condenser_topology as condenser
    import temsim.optics.crossover_acceptance as acceptance
    import temsim.optics.waist_evidence as waists
    import temsim.physics.beam_current as current
    state = SimpleNamespace(simulation_mode="unified", step_mm=.05,
        electron_gun=SimpleNamespace(exit_plane_z_mm=0., emitted_current_a=1e-9,
            emitter=SimpleNamespace(ray_count=16, emission_energy_ev=.5, energy_spread_fwhm_ev=.3)),
        sample=SimpleNamespace(upper_surface_z_mm=10.),
        lenses=[SimpleNamespace(key=key, enabled=True, z_mm=z, percent=1.,
            mechanical_center_from_tip_mm=z, mechanical_length_mm=2.)
            for key, z in (("adapter_lens", 3.), ("probe_tl12_lens", 7.))])
    snapshot = SimpleNamespace(digest="unchanged-state")
    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", lambda _: snapshot)
    monkeypatch.setattr(manifest, "solver_source_identity", lambda: {"source": "synthetic"})
    monkeypatch.setattr(current, "effective_source_current_a", lambda _: 1e-9)
    monkeypatch.setattr(current, "column_current_limit_percent", lambda _: 100.)
    monkeypatch.setattr(condenser, "requested_capture_planes", lambda *a, **k: (0., 2., 4., 6., 8., 10.))
    monkeypatch.setattr(qualification, "corrector_crossover_regions", lambda _: {})
    monkeypatch.setattr(qualification, "requested_corrector_crossover_planes", lambda _: (4., 6.))
    monkeypatch.setattr(qualification, "corrector_planes", lambda _: (3., 7.))
    monkeypatch.setattr(qualification, "evaluate_corrector_crossovers", lambda *a, **k:
        dict(waist_detector="finite_region", waist_evidence_passed=True, angle_matching_role="diagnostic"))
    monkeypatch.setattr(waists, "detector_for_state", lambda _: "finite_region")
    monkeypatch.setattr(waists, "observed_aperture_losses", lambda *a: [])
    monkeypatch.setattr(acceptance, "regions_for_state", lambda _: {"full": {"bounds_mm": (0., 10.)}})
    chain = finite_direct_focus_inputs()[-1]
    chain["status"] = "OBSERVED_AT_BOUNDED_SAMPLING"
    monkeypatch.setattr(acceptance, "evaluate_crossover_chain", lambda *a, **k: chain)
    topology_calls = []
    def topology(*args, **kwargs):
        topology_calls.append(args[1].z_mm.copy())
        index = len(topology_calls)
        return dict(passed=index == 3, status="PASS" if index == 3 else "UNRESOLVED",
            refinement_planes_mm=[1.1] if index == 1 else [1.2] if index == 2 else [])
    monkeypatch.setattr(condenser, "evaluate", topology)
    calls = []
    phi = np.arange(16) * 2*np.pi/16
    x, y, tx, ty = 1e-10*np.cos(phi), 1e-10*np.sin(phi), .03*np.cos(phi), .03*np.sin(phi)
    emitted = SimpleNamespace(x_m=x, y_m=y, tx_rad=tx, ty_rad=ty,
        flight_time_s=np.zeros(16), energy_offset_ev=np.zeros(16), weight=np.ones(16),
        alive=np.ones(16, bool), ray_id=np.arange(16))
    def capture(state, planes, **kwargs):
        z = np.asarray(planes)
        calls.append(z.copy())
        cp = SimpleNamespace(z_mm=z, x_m=np.tile(x, (len(z), 1)), y_m=np.tile(y, (len(z), 1)),
            tx_rad=np.tile(tx, (len(z), 1)), ty_rad=np.tile(ty, (len(z), 1)),
            flight_time_s=np.zeros((len(z), 16)), kinetic_energy_ev=np.full((len(z), 16), 300000.))
        return SimpleNamespace(exit_bundle=emitted), cp, np.ones((len(z), 16), bool)
    monkeypatch.setattr(beam_audit, "incident_checkpoints", capture)
    return state, calls


def test_extra_capture_planes_are_exact_in_initial_capture_and_every_refinement(capture_observer):
    state, calls = capture_observer
    extras = (7.651, 7.649, 7.651)
    report = qualification.observe(state, extra_capture_planes_mm=extras)
    assert len(calls) == 3
    assert report["extra_capture_planes_mm"] == [7.649, 7.651]
    for planes in calls:
        assert np.all(np.diff(planes) > 0)
        assert all(np.count_nonzero(planes == value) == 1 for value in set(extras))
    assert 1.1 in calls[1] and 1.1 in calls[2] and 1.2 in calls[2]
    assert report["snapshot_digest"] == "unchanged-state"


def test_default_capture_planes_and_refinement_sets_are_unchanged(capture_observer):
    state, calls = capture_observer
    report = qualification.observe(state)
    expected = np.unique(np.r_[np.arange(0., 10.1, .5), 9.9998, 9.9999])
    np.testing.assert_array_equal(calls[0], expected)
    np.testing.assert_array_equal(calls[1], np.array([0., 1.1, 2., 4., 6., 8., 10.]))
    np.testing.assert_array_equal(calls[2], np.array([0., 1.1, 1.2, 2., 4., 6., 8., 10.]))
    assert report["extra_capture_planes_mm"] == []


@pytest.mark.parametrize("values", [None, [True], [np.bool_(True)], ["1"], [[1.]],
    [float("nan")], [float("inf")], [-.01], [10.01]])
def test_invalid_extra_capture_planes_cannot_start_transport(capture_observer, values):
    state, calls = capture_observer
    with pytest.raises(ValueError, match="Extra capture planes"):
        qualification.observe(state, extra_capture_planes_mm=values)
    assert not calls


def test_extra_capture_normalization_preserves_exact_boundary_values(capture_observer):
    state, calls = capture_observer
    below = np.nextafter(7.65, 0.)
    above = np.nextafter(7.65, np.inf)
    assert qualification.validated_extra_capture_planes(state,
        [10., 0., above, below, 7.65, below]) == (0., below, 7.65, above, 10.)
    assert not calls
