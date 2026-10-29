"""Analytical observation/report tests; no particle or coherent propagation."""
from copy import deepcopy
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest

from temsim import performance_accuracy as report


@pytest.fixture
def inputs(monkeypatch):
    def capture(state):
        return NS(graph={"nodes": [{"type": "test:State", "attributes": {
            "step_mm": state.step_mm, "excitation": state.excitation}}],
            "root": {"ref": 0}}, external_inputs=(), implementation=state.implementation)
    monkeypatch.setattr(report, "capture_instrument_snapshot", capture)
    monkeypatch.setattr(report, "physical_identity_graph", lambda graph: graph)
    monkeypatch.setattr("temsim.physics.beam_current.effective_source_current_a", lambda state: 8e-12)
    monkeypatch.setattr("temsim.optics.condenser_topology.is_applicable", lambda state: False)
    state = NS(step_mm=.05, excitation=42., implementation="code-a",
               sample=NS(upper_surface_z_mm=10.))
    z = np.array([9., 10.-.0002, 10.-.0001, 10.])
    phi = np.arange(32)*2*np.pi/32
    tx = np.broadcast_to(.002*np.cos(phi), (len(z), len(phi))).copy()
    ty = np.broadcast_to(.002*np.sin(phi), tx.shape).copy()
    x = (z-10.)[:, None]*1e-3*tx+2e-10*np.cos(2*phi)
    y = (z-10.)[:, None]*1e-3*ty+2e-10*np.sin(2*phi)
    weights = np.ones(len(phi))
    alive = np.ones(len(phi), bool)
    ids = np.arange(100, 100+len(phi))
    branch = NS(z=z, x=x, y=y, tx=tx, ty=ty, alive=alive,
        ray_weight=weights, source_ray_id=ids, blocked_z=np.full(len(phi), np.nan),
        blocked_key=[None]*len(phi))
    source = NS(x_m=x[0].copy(), y_m=y[0].copy(), tx_rad=tx[0].copy(), ty_rad=ty[0].copy(),
        weight=weights.copy(), ray_id=ids.copy(), alive=alive.copy(), energy_offset_ev=np.zeros(len(phi)))
    cp = NS(z_mm=z.copy(), x_m=x.copy(), y_m=y.copy(), tx_rad=tx.copy(), ty_rad=ty.copy())
    simulation = NS(incident=branch, gun_trace=NS(exit_bundle=source), incident_checkpoints=cp, metrics={})
    result = NS(simulation=simulation, cache_hit=False, calculated_products={"column"}, reused_products=set(),
        calculation_manifest=NS(instrument_snapshot=capture(state)),
        performance={"stages": [{"stage": "Column", "seconds": .25}], "pipeline_seconds": .3,
            "backend_stages": [{"stage": "Column", "actual": "NumPy", "requested": "CPU", "outcome": "executed"}]})
    return state, result


@pytest.fixture
def section_inputs(inputs):
    from temsim.physics.particle_sections import ParticleSectionCheckpoint, ParticleSectionSegment, SECTION_SCHEMA
    state, result = inputs
    simulation = result.simulation
    column, cp, gun = simulation.incident, simulation.incident_checkpoints, simulation.gun_trace
    column.flight_time_s = np.broadcast_to((np.arange(4)+1)[:, None]*1e-9, (4, 32)).copy()
    column.kinetic_energy_ev = np.full((4, 32), 300000.)
    column.energy_offset_ev = np.zeros(32)
    column.source_azimuth_rad = np.zeros(32)
    cp.flight_time_s = column.flight_time_s.copy()
    cp.kinetic_energy_ev = column.kinetic_energy_ev.copy()
    gun.exit_bundle.flight_time_s = column.flight_time_s[0].copy()
    gun.z_mm = np.array([8., 9.])
    for name, key in (("x", "x_m"), ("y", "y_m"), ("tx", "tx_rad"), ("ty", "ty_rad")):
        setattr(gun, key, np.stack([getattr(column, name)[0]]*2))
    gun.flight_time_s = np.stack([np.zeros(32), column.flight_time_s[0]])
    plan = NS(z_mm=column.z.copy(), checkpoint_index=np.arange(4))
    simulation.incident_plan = plan
    segment = ParticleSectionSegment("incident", column, plan, cp, "executed-dependency")
    simulation.section_checkpoint = ParticleSectionCheckpoint(SECTION_SCHEMA, "executed-gun", gun, (segment,))
    display = deepcopy(column)
    display.z = np.r_[8., column.z]
    for name, key in (("x", "x_m"), ("y", "y_m"), ("tx", "tx_rad"),
                      ("ty", "ty_rad"), ("flight_time_s", "flight_time_s")):
        setattr(display, name, np.vstack([getattr(gun, key)[0], getattr(column, name)]))
    display.kinetic_energy_ev = np.vstack([np.full(32, np.nan), column.kinetic_energy_ev])
    simulation.incident = display
    simulation.metrics["section_target_z_mm"] = 10.
    return state, result


def run(observation, phase="warm", seconds=1., **extra):
    return dict(case_id="nano", phase=phase, elapsed_seconds=seconds,
                observation=observation, memory={"peak_rss_bytes": 1000}, **extra)


def test_observer_measures_retained_absolute_current_and_explicit_units(inputs):
    state, result = inputs
    before = deepcopy((state, result))
    observed = report.observe_calculation(state, result)
    assert observed["execution"]["status"] == "COMPLETE"
    assert observed["population"]["status"] == "VALID"
    assert observed["identity"]["status"] == "CAPTURED"
    values = observed["observables"]
    assert values["status"] == "MEASURED"
    assert values["alpha95_mrad"] == pytest.approx(1e3*np.arctan(.002))
    assert values["d95_nm"] == pytest.approx(.4)
    assert values["rms_radius_nm"] == pytest.approx(.2)
    assert values["current_a"] == pytest.approx(8e-12)
    assert observed["qualification"]["physical"]["focus"]["status"] == "MEASURED"
    assert observed["qualification"]["physical"]["topology"]["status"] == "NOT_EVALUATED"
    assert observed["qualification"]["physical"]["status"] == "UNRESOLVED"
    assert observed["qualification"]["numerical"]["status"] == "NOT_RUN"
    json.dumps(observed, allow_nan=False)
    for key in ("x", "y", "tx", "ty", "ray_weight", "alive", "blocked_z"):
        np.testing.assert_array_equal(getattr(result.simulation.incident, key), getattr(before[1].simulation.incident, key))
    assert vars(state) == vars(before[0])


def test_observer_preserves_aperture_loss_denominator(inputs):
    state, result = inputs
    result.simulation.incident.alive[:8] = False
    result.simulation.incident.blocked_z[:8] = 9.5
    observed = report.observe_calculation(state, result)
    assert observed["population"]["source_current_denominator"] == 32.
    assert observed["observables"]["current_a"] == pytest.approx(6e-12)
    assert observed["observables"]["transmission"] == .75


@pytest.fixture
def downstream_stop_inputs(inputs):
    state, result = inputs
    branch = result.simulation.incident
    branch.z = np.r_[branch.z, 10.0001]
    for name in ("x", "y", "tx", "ty"):
        values = getattr(branch, name)
        setattr(branch, name, np.vstack([values, values[-1]]))
    branch.alive[:8] = False
    branch.blocked_z[:8] = 10.00005
    branch.blocked_key[:8] = ["downstream-aperture"] * 8
    return state, result


@pytest.mark.parametrize("checkpoint", [True, False])
def test_exact_surface_observables_include_particles_stopped_later(downstream_stop_inputs, checkpoint):
    state, result = downstream_stop_inputs
    if not checkpoint:
        result.simulation.incident_checkpoints = None
    observed = report.observe_calculation(state, result)
    values = observed["observables"]
    assert observed["population"]["status"] == "VALID"
    assert observed["population"]["surviving_positive_current_rays"] == 24
    assert values["status"] == "MEASURED"
    assert values["plane_kind"] == "specimen_upper_surface"
    assert values["plane_z_mm"] == 10.
    assert values["current_a"] == pytest.approx(8e-12)
    assert values["transmission"] == 1.
    assert values["alpha95_mrad"] == pytest.approx(1e3*np.arctan(.002))
    assert values["d95_nm"] == pytest.approx(.4)
    assert values["rms_radius_nm"] == pytest.approx(.2)


@pytest.mark.parametrize("damage", ["missing_stop", "future_stop", "alive_after_stop",
                                    "nonfinite_before_stop", "checkpoint_before_stop",
                                    "clock_before_stop", "energy_before_stop"])
def test_plane_observation_rejects_invalid_downstream_stop_evidence(downstream_stop_inputs, damage):
    state, result = downstream_stop_inputs
    branch = result.simulation.incident
    if damage == "missing_stop":
        branch.blocked_z[0] = np.nan
    elif damage == "future_stop":
        branch.blocked_z[0] = 10.0002
    elif damage == "alive_after_stop":
        branch.alive[0] = True
    elif damage == "nonfinite_before_stop":
        branch.x[-2, 0] = np.nan
    elif damage == "checkpoint_before_stop":
        result.simulation.incident_checkpoints.x_m[-1, 0] = np.nan
    else:
        name = "flight_time_s" if damage == "clock_before_stop" else "kinetic_energy_ev"
        values = np.full(branch.x.shape, 1e-9 if name == "flight_time_s" else 300000.)
        values[-2, 0] = -1. if name == "flight_time_s" else 0.
        setattr(branch, name, values)
    observed = report.observe_calculation(state, result)
    assert observed["population"]["status"] == "UNRESOLVED"
    assert observed["observables"]["status"] == "NOT_EVALUATED"
    assert report.compare_observations(observed, observed)["status"] != "PASS"


@pytest.mark.parametrize("damage", ["missing_ids", "bad_ids", "weights", "nan_alive", "missing_source", "resurrection"])
def test_incomplete_or_invalid_source_never_passes_repeatability(inputs, damage):
    state, result = inputs
    branch = result.simulation.incident
    if damage == "missing_ids": branch.source_ray_id = None
    elif damage == "bad_ids": branch.source_ray_id[0] = branch.source_ray_id[1]
    elif damage == "weights": branch.ray_weight[0] = -1.
    elif damage == "nan_alive": branch.x[-1, 0] = np.nan
    elif damage == "missing_source": result.simulation.gun_trace = None
    else: result.simulation.gun_trace.exit_bundle.alive[0] = False
    observed = report.observe_calculation(state, result)
    assert observed["population"]["status"] == "UNRESOLVED"
    assert report.compare_observations(observed, observed)["status"] != "PASS"
    json.dumps(observed, allow_nan=False)


def test_stopped_nonfinite_phase_is_retained_in_fingerprint_without_fabrication(inputs):
    state, result = inputs
    branch = result.simulation.incident
    branch.alive[0] = False
    branch.blocked_z[0] = 9.5
    branch.x[1:, 0] = np.nan
    result.simulation.incident_checkpoints.x_m[1:, 0] = np.nan
    observed = report.observe_calculation(state, result)
    assert observed["population"]["status"] == "VALID"
    assert observed["population"]["fingerprints"]["x_m"]["nonfinite_count"] == 3
    assert observed["observables"]["status"] == "MEASURED"


@pytest.mark.parametrize("owner_name", ["incident", "incident_checkpoints", "source"])
@pytest.mark.parametrize("name,value", [("kinetic_energy_ev", np.nan), ("kinetic_energy_ev", 0.),
                                       ("flight_time_s", -1.), ("flight_time_s", np.nan)])
def test_present_invalid_time_or_energy_evidence_cannot_pass(inputs, owner_name, name, value):
    state, result = inputs
    owner = (result.simulation.gun_trace.exit_bundle if owner_name == "source"
             else getattr(result.simulation, owner_name))
    shape = (32,) if owner_name == "source" else (4, 32)
    setattr(owner, name, np.full(shape, value))
    observed = report.observe_calculation(state, result)
    assert observed["population"]["status"] == "UNRESOLVED"
    assert report.compare_observations(observed, observed)["status"] != "PASS"


def test_valid_checkpoint_clock_and_energy_are_fingerprinted(inputs):
    state, result = inputs
    cp = result.simulation.incident_checkpoints
    cp.flight_time_s = np.broadcast_to(np.arange(4)[:, None]*1e-9, (4, 32))
    cp.kinetic_energy_ev = np.full((4, 32), 300000.)
    observed = report.observe_calculation(state, result)
    assert observed["population"]["status"] == "VALID"
    assert "flight_time_s" in observed["population"]["checkpoint_fingerprints"]
    assert "kinetic_energy_ev" in observed["population"]["checkpoint_fingerprints"]


def test_proven_production_gun_prefix_unknown_energy_is_explicit_and_exact(section_inputs):
    state, result = section_inputs
    observation = report.observe_calculation(state, result)
    assert observation["population"]["status"] == "VALID"
    coverage = observation["population"]["energy_coverage"]
    assert coverage["status"] == "PARTIAL_RETAINED"
    assert coverage["unknown_prefix_rows"] == 1
    assert coverage["column_start_z_mm"] == 9.
    assert observation["population"]["fingerprints"]["kinetic_energy_ev"]["nonfinite_count"] == 32
    assert observation["observables"]["plane_kind"] == "specimen_upper_surface"
    assert report.compare_observations(observation, observation)["status"] == "PASS"


@pytest.mark.parametrize("damage", ["prefix_z", "prefix_phase", "prefix_clock", "missing_plan",
    "column_nan", "checkpoint_nan", "prefix_infinity", "prefix_zero", "section_ids", "section_phase"])
def test_gun_prefix_exemption_requires_exact_provenance_and_valid_column(section_inputs, damage):
    state, result = section_inputs
    sim, branch = result.simulation, result.simulation.incident
    if damage == "prefix_z": branch.z[0] -= .01
    elif damage == "prefix_phase": branch.x[0, 0] += 1e-9
    elif damage == "prefix_clock": branch.flight_time_s[0, 0] += 1e-12
    elif damage == "missing_plan": sim.incident_plan = None
    elif damage == "column_nan": branch.kinetic_energy_ev[1, 0] = np.nan
    elif damage == "checkpoint_nan": sim.incident_checkpoints.kinetic_energy_ev[-1, 0] = np.nan
    elif damage == "prefix_infinity": branch.kinetic_energy_ev[0, 0] = np.inf
    elif damage == "prefix_zero": branch.kinetic_energy_ev[0, 0] = 0.
    elif damage == "section_ids": sim.section_checkpoint.segments[0].branch.source_ray_id[0] += 100
    else: sim.section_checkpoint.segments[0].branch.x[1, 0] += 1e-9
    observation = report.observe_calculation(state, result)
    assert observation["population"]["status"] == "UNRESOLVED"
    assert report.compare_observations(observation, observation)["status"] != "PASS"


def test_gun_prefix_proof_also_supports_full_column_without_section_cache(section_inputs):
    state, result = section_inputs
    result.simulation.section_checkpoint = None
    observation = report.observe_calculation(state, result)
    assert observation["population"]["status"] == "VALID"
    assert observation["population"]["energy_coverage"]["unknown_prefix_rows"] == 1
    result.simulation.incident.kinetic_energy_ev[1, 0] = np.nan
    assert report.observe_calculation(state, result)["population"]["status"] == "UNRESOLVED"


def test_bounded_section_measures_local_exact_terminal_without_probe_qualification(section_inputs):
    state, result = section_inputs
    state.sample.upper_surface_z_mm = 12.
    observed = report.observe_calculation(state, result)
    assert observed["observables"]["status"] == "MEASURED"
    assert observed["observables"]["plane_kind"] == "section_terminal"
    assert observed["observables"]["plane_z_mm"] == 10.
    assert observed["observables"]["current_a"] == pytest.approx(8e-12)
    assert observed["observables"]["d95_nm"] == pytest.approx(.4)
    for name in ("focus", "topology"):
        assert observed["qualification"]["physical"][name]["status"] == "NOT_EVALUATED"
        assert "has not reached the specimen" in observed["qualification"]["physical"][name]["reason"]
    assert report.compare_observations(observed, observed)["status"] == "PASS"
    assert "Plane: section_terminal; z_mm=10" in report.format_report(report.summarize_runs([run(observed)]))


@pytest.mark.parametrize("damage", ["missing_section", "target", "past_specimen", "missing_terminal"])
def test_bounded_terminal_requires_valid_exact_completed_section(section_inputs, damage):
    state, result = section_inputs
    state.sample.upper_surface_z_mm = 12.
    if damage == "missing_section": result.simulation.section_checkpoint = None
    elif damage == "target": result.simulation.metrics["section_target_z_mm"] = 9.99
    elif damage == "past_specimen": state.sample.upper_surface_z_mm = 9.95
    else: result.simulation.incident_checkpoints.z_mm[-1] -= 1e-8
    observed = report.observe_calculation(state, result)
    assert observed["observables"]["status"] != "MEASURED"


@pytest.mark.parametrize("kind,z", [("section_terminal", 10.), ("specimen_upper_surface", 9.), (None, 10.), ("specimen_upper_surface", None)])
def test_plane_identity_cannot_be_hidden_by_equal_scalar_metrics(inputs, kind, z):
    original = report.observe_calculation(*inputs)
    changed = deepcopy(original)
    changed["observables"].update(plane_kind=kind, plane_z_mm=z)
    comparison = report.compare_observations(original, changed)
    assert comparison["status"] == "FAIL"
    assert any("observation plane" in issue for issue in comparison["failures"])


def test_historical_manifest_implementation_is_separate_from_current_observer(inputs):
    state, result = inputs
    state.implementation = "new-observer-code"
    observation = report.observe_calculation(state, result)
    assert observation["identity"]["implementation"] == "code-a"
    assert observation["identity"]["observer_implementation"] == "new-observer-code"
    result.calculation_manifest = None
    result.simulation.incident_checkpoints.unified_state = NS(implementation_digest="saved-executed-code")
    observation = report.observe_calculation(state, result)
    assert observation["identity"]["implementation"] == "saved-executed-code"
    assert observation["identity"]["implementation_basis"] == "Executed unified incident checkpoint"
    result.simulation.incident_checkpoints.unified_state = None
    assert report.observe_calculation(state, result)["identity"]["implementation"] is None


def test_missing_exact_surface_does_not_interpolate(inputs):
    state, result = inputs
    state.sample.upper_surface_z_mm = 9.95
    observed = report.observe_calculation(state, result)
    assert observed["observables"]["status"] == "NOT_EVALUATED"
    assert observed["observables"]["d95_nm"] is None


def test_exact_sample_reference_is_measured_without_surface_or_scattering_claims(inputs):
    state, result = inputs
    state.sample.upper_surface_z_mm = 10.-2.5e-6
    state.sample.z_mm = 10.
    result.simulation.incident_plan = NS(z_mm=result.simulation.incident.z.copy())
    observation = report.observe_calculation(state, result)
    values = observation["observables"]
    assert values["status"] == "MEASURED"
    assert values["plane_kind"] == "sample_reference_plane"
    assert values["plane_z_mm"] == 10.
    assert values["d95_nm"] == pytest.approx(.4)
    assert values["current_a"] == pytest.approx(8e-12)
    assert "not a specimen upper-surface focus or specimen-scattering measurement" in values["basis"]
    for name in ("focus", "topology"):
        assert observation["qualification"]["physical"][name]["status"] == "NOT_EVALUATED"
        assert "incident optical reference" in observation["qualification"]["physical"][name]["reason"]
    assert report.compare_observations(observation, observation)["status"] == "PASS"
    changed = deepcopy(observation)
    changed["observables"]["plane_kind"] = "specimen_upper_surface"
    assert report.compare_observations(observation, changed)["status"] == "FAIL"
    assert "Plane: sample_reference_plane; z_mm=10" in report.format_report(report.summarize_runs([run(observation)]))


@pytest.mark.parametrize("damage", ["missing_checkpoint", "nearby_reference", "missing_plan", "plan_terminal", "history_terminal"])
def test_sample_reference_requires_exact_original_checkpoint_and_matching_endpoints(inputs, damage):
    state, result = inputs
    state.sample.upper_surface_z_mm = 10.-2.5e-6
    state.sample.z_mm = 10.
    result.simulation.incident_plan = NS(z_mm=result.simulation.incident.z.copy())
    if damage == "missing_checkpoint": result.simulation.incident_checkpoints = None
    elif damage == "nearby_reference": state.sample.z_mm = np.nextafter(10., np.inf)
    elif damage == "missing_plan": result.simulation.incident_plan = None
    elif damage == "plan_terminal": result.simulation.incident_plan.z_mm[-1] += .01
    else: result.simulation.incident.z[-1] += .01
    observation = report.observe_calculation(state, result)
    assert observation["observables"]["status"] != "MEASURED"
    assert observation["observables"]["d95_nm"] is None


def test_upper_surface_keeps_priority_and_old_plane_reports_remain_comparable(section_inputs):
    state, result = section_inputs
    state.sample.z_mm = 10.
    surface = report.observe_calculation(state, result)
    assert surface["observables"]["plane_kind"] == "specimen_upper_surface"
    section_state, section_result = section_inputs
    section_state.sample.upper_surface_z_mm = 12.
    section_state.sample.z_mm = 12.0000025
    terminal = report.observe_calculation(section_state, section_result)
    assert terminal["observables"]["plane_kind"] == "section_terminal"
    for observation in (surface, terminal):
        old = deepcopy(observation)
        old["identity"]["observer"] = "temsim-performance-accuracy-observer-v2"
        old["identity"]["observer_implementation"] = "previous-observer-code"
        assert report.compare_observations(old, observation)["status"] == "PASS"


def test_full_reuse_excludes_inherited_stage_times_and_backend_claims(inputs):
    state, result = inputs
    result.cache_hit = True
    observed = report.observe_calculation(state, result)
    performance = observed["performance"]
    assert performance["pipeline_seconds"] is None
    assert performance["stages"] == performance["backend_stages"] == []
    assert performance["calculated_products"] == []
    assert performance["reused_products"] == ["column"]
    assert performance["inherited_timing_ignored"]


def test_exact_repeat_and_before_after_implementation_change(inputs):
    state, result = inputs
    before = report.observe_calculation(state, result)
    state.implementation = "code-b"
    result.calculation_manifest.instrument_snapshot.implementation = "code-b"
    after = report.observe_calculation(state, result)
    assert before["identity"]["inputs_digest"] == after["identity"]["inputs_digest"]
    comparison = report.compare_observations(before, after)
    assert comparison["status"] == "PASS"
    assert comparison["implementations"] == ["code-a", "code-b"]
    left = report.summarize_runs([run(before, seconds=2.), run(before, seconds=2.)])
    right = report.summarize_runs([run(after), run(after)])
    assert report.compare_reports(left, right)["status"] == "PASS"
    assert report.compare_reports(left, right)["performance"][0]["speedup"] == 2.


@pytest.mark.parametrize("field", ["excitation", "step_mm"])
def test_physical_and_numerical_identity_changes_are_incomparable(inputs, field):
    state, result = inputs
    before = report.observe_calculation(state, result)
    setattr(state, field, getattr(state, field)*2)
    after = report.observe_calculation(state, result)
    assert report.compare_observations(before, after)["status"] == "FAIL"
    if field == "step_mm":
        assert before["identity"]["physical_inputs_digest"] == after["identity"]["physical_inputs_digest"]
        assert before["identity"]["numerical_inputs_digest"] != after["identity"]["numerical_inputs_digest"]


def test_hash_changes_cannot_be_waived_by_loose_scalar_tolerances(inputs):
    state, result = inputs
    before = report.observe_calculation(state, result)
    result.simulation.incident.x[-1] += 1e-9  # Chief shift can preserve centered spot statistics.
    result.simulation.incident_checkpoints.x_m[-1] += 1e-9
    after = report.observe_calculation(state, result)
    comparison = report.compare_observations(before, after,
        metric_tolerances={key: {"atol": 100., "rtol": 100.} for key in report.METRICS})
    assert comparison["status"] == "FAIL"
    assert any("fingerprints" in message for message in comparison["failures"])


def test_summary_groups_phases_and_keeps_missing_memory_unknown(inputs):
    observation = report.observe_calculation(*inputs)
    summary = report.summarize_runs([run(observation, "cold", 5.), run(observation, seconds=1.),
                                    run(observation, seconds=3.), run(observation, "reuse", .1)])
    warm = next(row for row in summary["groups"] if row["phase"] == "warm")
    assert warm["elapsed_seconds"]["median"] == 2.
    assert warm["elapsed_seconds"]["standard_deviation"] == 1.
    assert warm["peak_vram_bytes"]["median"] is None
    assert warm["peak_vram_bytes"]["missing_count"] == 2
    assert summary["repeatability"]["status"] == "PASS"
    assert summary["physical_qualification"] == "UNRESOLVED"
    assert "Physical qualification: UNRESOLVED" in report.format_report(summary)


def test_repeatability_never_hides_a_retained_physical_failure(inputs):
    observation = report.observe_calculation(*inputs)
    observation["qualification"]["physical"]["status"] = "FAIL"
    summary = report.summarize_runs([run(observation), run(observation)])
    assert summary["repeatability"]["status"] == "PASS"
    assert summary["physical_qualification"] == "FAIL"
    assert len(summary["physical_failures"]) == 2
    assert "Physical qualification: FAIL" in report.format_report(summary)


def test_text_report_displays_single_run_measurements_and_qualification_limits(inputs):
    observation = report.observe_calculation(*inputs)
    summary = report.summarize_runs([run(observation)])
    original = deepcopy(summary)
    rendered = report.format_report(summary)
    assert "nano measurements: MEASURED (representative retained run 1, phase warm)" in rendered
    assert "alpha95_mrad=1.9999973" in rendered
    assert "D95_nm=0.4; RMS_radius_nm=0.2; current_a=8e-12" in rendered
    assert "Focus: MEASURED; local_waist_offset_nm=" in rendered
    assert "no declared target-angle or size qualification" in rendered
    assert "Topology: NOT_EVALUATED; Required C2/C3/ADL geometry is not applicable" in rendered
    assert "nano repeatability: NOT_EVALUATED" in rendered
    assert summary == original


def test_text_report_keeps_unavailable_measurements_and_topology_reason_visible(inputs):
    observation = report.observe_calculation(*inputs)
    observation["qualification"]["physical"]["topology"] = {
        "status": "UNRESOLVED", "message": "Checkpoint spacing exceeds the declared limit"}
    observation["observables"] = {"status": "NOT_EVALUATED", "reason": "Exact surface plane is missing"}
    rendered = report.format_report(report.summarize_runs([
        run(observation), {**run(None, execution_status="FAILED"), "case_id": "missing"}]))
    assert "alpha95_mrad=unknown; D95_nm=unknown; RMS_radius_nm=unknown; current_a=unknown" in rendered
    assert "Measurement reason: Exact surface plane is missing" in rendered
    assert "Topology: UNRESOLVED; Checkpoint spacing exceeds the declared limit" in rendered
    assert "missing measurements: NOT_EVALUATED; no retained observation" in rendered
    assert "missing repeatability: FAIL" in rendered


def test_failed_runs_are_not_dropped_from_summary_or_before_after(inputs):
    observation = report.observe_calculation(*inputs)
    good = report.summarize_runs([run(observation)])
    assert good["repeatability"]["status"] == "NOT_EVALUATED"
    failed = report.summarize_runs([run(None, execution_status="FAILED"), run(observation)])
    assert failed["repeatability"]["status"] == "FAIL"
    assert failed["repeatability"]["cases"][0]["failed_run_indices"] == [0]
    assert report.compare_reports(failed, good)["status"] == "FAIL"
    assert report.compare_reports(good, failed)["status"] == "FAIL"
    assert failed["groups"][0]["complete_runs"] == 1
    assert failed["groups"][0]["failed_or_incomplete_runs"] == 1
    assert failed["groups"][0]["elapsed_seconds"]["count"] == 1


def test_fast_failure_cannot_improve_performance_median(inputs):
    observation = report.observe_calculation(*inputs)
    summary = report.summarize_runs([run(observation, seconds=4.),
        run(None, seconds=.001, execution_status="FAILED"), run(observation, seconds=6.)])
    assert summary["groups"][0]["elapsed_seconds"]["median"] == 5.
    assert summary["repeatability"]["status"] == "FAIL"


def test_stop_labels_are_part_of_exact_particle_identity(inputs):
    state, result = inputs
    before = report.observe_calculation(state, result)
    result.simulation.incident.blocked_key[0] = "changed-stop"
    after = report.observe_calculation(state, result)
    assert report.compare_observations(before, after)["status"] == "FAIL"


def test_before_after_requires_matching_phases_counts_and_internal_repeats(inputs):
    observation = report.observe_calculation(*inputs)
    single = report.summarize_runs([run(observation)])
    repeated = report.summarize_runs([run(observation), run(observation)])
    changed_phase = report.summarize_runs([run(observation, "cold"), run(observation)])
    changed_count = report.summarize_runs([run(observation)]*3)
    assert report.compare_reports(single, single)["status"] == "UNRESOLVED"
    assert report.compare_reports(repeated, changed_phase)["status"] == "INCOMPARABLE"
    assert report.compare_reports(repeated, changed_count)["status"] == "INCOMPARABLE"


@pytest.mark.parametrize("limits", [{"d95_nm": {"atol": -1., "rtol": 0.}},
    {"d95_nm": {"atol": 0.}}, {"d95_nm": {"atol": float("nan"), "rtol": 0.}},
    {"d95_nm": {"atol": True, "rtol": 0.}}, {"unknown": {"atol": 0., "rtol": 0.}}])
def test_tolerances_must_be_explicit_valid_and_known(limits):
    with pytest.raises(ValueError):
        report.summarize_runs([], metric_tolerances=limits)


def test_budget_change_is_rejected_and_json_is_finite(inputs, tmp_path):
    observation = report.observe_calculation(*inputs)
    before = report.summarize_runs([run(observation)])
    after = report.summarize_runs([run(observation)], metric_tolerances={"d95_nm": {"atol": .01, "rtol": 0.}})
    assert report.compare_reports(before, after)["status"] == "INCOMPARABLE"
    destination = report.write_report(before, tmp_path/"report.json")
    decoded = json.loads(destination.read_text(encoding="utf-8"))
    assert decoded == before
