"""Analytic and mocked acceptance checks; no source or column propagation."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


_spec = spec_from_file_location("validate_joint_probe",
    Path(__file__).resolve().parents[1] / "scripts" / "validate_joint_probe.py")
validator = module_from_spec(_spec)
_spec.loader.exec_module(validator)


def observation():
    return dict(surface=dict(surface_z_mm=100., local_waist_offset_nm=.2,
        variance_second_derivative=.1, effective_rays=32., source_population_identity="source-a",
        statistics=dict(convergence_95_rad=.03, radius_95_m=.3e-9, radius_rms_m=1e-9,
            waist_offset_m=.3e-9, surviving_rays=32, surviving_fraction=.0001)),
        topology=dict(passed=True, status="PASS"),
        corrector=dict(waist_detector="finite_region", waist_evidence_passed=True),
        crossover_chain=dict(required_topology_satisfied_at_sampling=True,
            direct_focus=dict(schema="direct-focus-finite-region-topology-v2",
                requirement="direct_focus_without_extra_pre_specimen_waist",
                status="PASS", passed=True, failures=[]),
            unattributed_population_change_count=0), sample_current_a=1e-12,
        fixed_sampling_pass=True, failures=[])


def run_row():
    return dict(execution_status="COMPLETE", fixed_candidate_identity="candidate-a",
        solver_identity={"sha256": "solver-a"}, external_input_identity="external-a",
        source_identity="source-a", physical=validator.physical_gates(observation(), surface_z_mm=100.))


def test_actual_current_quantile_not_pupil_and_no_rms_or_transmission_floor():
    report = validator.physical_gates(observation(), surface_z_mm=100.)
    assert report["passed"]
    assert report["metrics"]["rms_radius_nm"] == 1.
    assert report["metrics"]["source_current_fraction"] == .0001
    assert report["physical_pupil_edge"]["status"] == "NOT_EVALUATED"


@pytest.mark.parametrize("field,value", [("radius_95_m", .35e-9),
    ("radius_95_m", float("nan")), ("convergence_95_rad", .03031),
    ("waist_offset_m", 1.01e-9), ("surviving_rays", 15), ("surviving_fraction", 0.)])
def test_physical_limits_fail_closed(field, value):
    record = observation()
    record["surface"]["statistics"][field] = value
    assert not validator.physical_gates(record, surface_z_mm=100.)["passed"]


@pytest.mark.parametrize("key,value", [("local_waist_offset_nm", 1.01),
    ("variance_second_derivative", 0.), ("effective_rays", 15.99),
    ("source_population_identity", None), ("surface_z_mm", 100.0000001)])
def test_exact_plane_focus_support_and_identity_are_required(key, value):
    record = observation()
    record["surface"][key] = value
    assert not validator.physical_gates(record, surface_z_mm=100.)["passed"]


def test_focus_boundary_is_inclusive_and_topology_is_independent():
    record = observation()
    record["surface"]["local_waist_offset_nm"] = -1.
    record["surface"]["statistics"]["waist_offset_m"] = 1e-9
    assert validator.physical_gates(record, surface_z_mm=100.)["passed"]
    record["crossover_chain"]["direct_focus"]["status"] = "UNRESOLVED"
    report = validator.physical_gates(record, surface_z_mm=100.)
    assert report["status"] == "UNRESOLVED" and not report["passed"]


def test_direct_focus_uses_independent_whole_capture_evidence():
    record = observation()
    record["topology"] = dict(passed=False, status="UNRESOLVED",
        failures=["unresolved_finite_condenser_evidence"])
    record["crossover_chain"]["required_topology_satisfied_at_sampling"] = False
    record["fixed_sampling_pass"] = False
    record["failures"] = ["Condenser topology failed or is unresolved"]
    report = validator.physical_gates(record, surface_z_mm=100.)
    assert report["checks"]["direct_focus_topology"]
    assert report["direct_focus_evidence"]["source"] == "crossover_chain.direct_focus"
    assert report["status"] == "UNRESOLVED" and not report["passed"]
    assert not report["checks"]["condenser_topology"]
    assert not report["checks"]["required_whole_capture_topology"]
    assert not report["checks"]["observer_acceptance"]


@pytest.mark.parametrize("evidence", [None, {}, "PASS",
    dict(schema="direct-focus-topology-v1", status="PASS", passed=True, failures=[]),
    dict(schema="direct-focus-finite-region-topology-v2", status="UNRESOLVED", passed=False, failures=[]),
])
def test_missing_or_unsupported_authoritative_direct_focus_never_falls_back(evidence):
    record = observation()
    record["topology"]["direct_focus"] = deepcopy(record["crossover_chain"]["direct_focus"])
    record["crossover_chain"]["direct_focus"] = evidence
    report = validator.physical_gates(record, surface_z_mm=100.)
    assert not report["checks"]["direct_focus_topology"]
    assert report["status"] == "UNRESOLVED" and not report["passed"]
    del record["crossover_chain"]["direct_focus"]
    assert not validator.physical_gates(record, surface_z_mm=100.)["passed"]


@pytest.mark.parametrize("key,value", [("passed", False), ("passed", 1),
    ("status", "UNKNOWN"), ("schema", None), ("requirement", None),
    ("failures", ["unresolved evidence"]), ("failures", None)])
def test_inconsistent_direct_focus_evidence_is_unresolved(key, value):
    record = observation()
    record["crossover_chain"]["direct_focus"][key] = value
    report = validator.physical_gates(record, surface_z_mm=100.)
    assert not report["checks"]["direct_focus_topology"]
    assert report["status"] == "UNRESOLVED" and not report["passed"]


def test_authoritative_direct_focus_failure_cannot_be_overridden_by_summary_pass():
    record = observation()
    record["topology"]["direct_focus"] = deepcopy(record["crossover_chain"]["direct_focus"])
    record["crossover_chain"]["direct_focus"].update(
        status="FAIL", passed=False, failures=["forbidden_pre_specimen_waist"])
    report = validator.physical_gates(record, surface_z_mm=100.)
    assert report["status"] == "FAIL" and not report["passed"]
    assert not report["checks"]["direct_focus_topology"]


def test_summary_direct_focus_cannot_replace_authoritative_pass():
    record = observation()
    record["topology"]["direct_focus"] = dict(status="UNRESOLVED", passed=False)
    assert validator.physical_gates(record, surface_z_mm=100.)["passed"]


def test_missing_loss_attribution_cannot_pass():
    record = observation()
    del record["crossover_chain"]["unattributed_population_change_count"]
    assert not validator.physical_gates(record, surface_z_mm=100.)["passed"]


@pytest.mark.parametrize("key", ["fixed_candidate_identity", "solver_identity", "external_input_identity", "source_identity"])
def test_step_comparison_requires_complete_matching_identities(key):
    left, right = run_row(), run_row()
    right[key] = "different"
    assert not validator.compare_observations(left, right)["passed"]
    del right[key]
    assert not validator.compare_observations(left, right)["passed"]


@pytest.mark.parametrize("key,value", [("alpha95_mrad", 31.), ("d95_nm", .65),
    ("rms_radius_nm", 1.1), ("current_a", 1.1e-12),
    ("local_focus_offset_nm", .451), ("covariance_focus_offset_nm", .551)])
def test_step_error_budgets_are_independently_applied(key, value):
    left, right = run_row(), run_row()
    right["physical"]["metrics"][key] = value
    assert not validator.compare_observations(left, right)["passed"]


def test_complete_execution_is_not_physical_acceptance():
    left, right = run_row(), run_row()
    left["physical"]["passed"] = right["physical"]["passed"] = False
    assert validator.compare_observations(left, right)["passed"]
    right["execution_status"] = "FAILED"
    assert not validator.compare_observations(left, right)["passed"]


class MockState:
    def __init__(self):
        self.step_mm = .04
        self.electron_gun = SimpleNamespace(exit_plane_z_mm=0., emitter=SimpleNamespace(ray_count=193))
        self.sample = SimpleNamespace(upper_surface_z_mm=100.)
        self.control = 12.34
        self.hidden_quality = "High accuracy"

    def to_dict(self):
        return dict(step=self.step_mm, rays=self.electron_gun.emitter.ray_count, control=self.control)


class MockSnapshot:
    def __init__(self, state):
        self._state = deepcopy(state)
        self.graph = dict(state=state.to_dict(), hidden_quality=state.hidden_quality)
        self.digest = validator._digest(self.graph)

    def restore(self):
        return deepcopy(self._state)

    def to_dict(self):
        return dict(graph=self.graph, implementation={"sha256": "solver-a"}, external_inputs=[{"sha256": "data-a"}])


@pytest.fixture
def execution(monkeypatch):
    import temsim.instrument_snapshot as snapshots
    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", MockSnapshot)
    calls = []
    def observer(state, **kwargs):
        calls.append((deepcopy(state.to_dict()), kwargs))
        result = observation()
        result.update(snapshot_digest=MockSnapshot(state).digest,
            solver_identity={"sha256": "solver-a"}, elapsed_s=.1,
            extra_capture_planes_mm=list(kwargs.get("extra_capture_planes_mm", ())),
            rays=state.electron_gun.emitter.ray_count, step_mm=state.step_mm)
        return result
    return calls, observer, lambda state: dict(status="PASS", passed=True)


def test_runner_preserves_original_and_only_changes_declared_discretization(tmp_path, execution):
    calls, observer, audit = execution
    state = MockState()
    original = state.to_dict()
    report = validator.validate_state(state, tmp_path / "validation", rays=769,
        observer=observer, audit=audit)
    assert state.to_dict() == original and report["original_state_unchanged"]
    assert [row[0]["step"] for row in calls] == [.05, .025]
    assert all(row[0]["rays"] == 769 and row[0]["control"] == original["control"] for row in calls)
    assert report["passed"] and report["step_comparison"]["passed"]
    assert report["source_count_convergence"]["status"] == "NOT_EVALUATED"
    assert report["qualification"].startswith("NOT_QUALIFIED")
    assert json.loads((tmp_path / "validation" / "report.json").read_text())["passed"]


def test_single_step_does_not_claim_numerical_acceptance(tmp_path, execution):
    calls, observer, audit = execution
    report = validator.validate_state(MockState(), tmp_path / "single", steps=(.025,), observer=observer, audit=audit)
    assert len(calls) == 1 and report["fixed_sampling_physical_pass"]
    assert not report["passed"] and report["step_comparison"]["status"] == "NOT_EVALUATED"


def test_explicit_extra_planes_are_identical_at_both_steps_and_recorded(tmp_path, execution):
    calls, observer, audit = execution
    report = validator.validate_state(MockState(), tmp_path / "extra", observer=observer,
        audit=audit, extra_capture_planes_mm=(76.51, 76.49, 76.51))
    assert report["passed"]
    assert report["extra_capture_planes_mm"] == [76.49, 76.51]
    assert all(row[1]["extra_capture_planes_mm"] == (76.49, 76.51) for row in calls)
    assert all(row["extra_capture_planes_mm"] == [76.49, 76.51] for row in report["runs"])
    assert json.loads((tmp_path / "extra" / "report.json").read_text())["extra_capture_planes_mm"] == [76.49, 76.51]


def test_default_extra_planes_leave_observer_invocation_unchanged(tmp_path, execution):
    calls, observer, audit = execution
    validator.validate_state(MockState(), tmp_path / "default", observer=observer, audit=audit)
    assert all("extra_capture_planes_mm" not in kwargs for _, kwargs in calls)


@pytest.mark.parametrize("planes", [None, [True], ["1"], [[1.]], [float("nan")],
    [float("inf")], [-.01], [100.01]])
def test_invalid_extra_planes_are_rejected_before_observation(tmp_path, execution, planes):
    calls, observer, audit = execution
    with pytest.raises(ValueError, match="Extra capture planes"):
        validator.validate_state(MockState(), tmp_path / "invalid", observer=observer,
            audit=audit, extra_capture_planes_mm=planes)
    assert not calls and not (tmp_path / "invalid").exists()


def test_missing_extra_plane_receipt_cannot_pass(tmp_path, execution):
    calls, observer, audit = execution
    def missing_receipt(state, **kwargs):
        record = observer(state, **kwargs)
        del record["extra_capture_planes_mm"]
        return record
    report = validator.validate_state(MockState(), tmp_path / "missing-extra",
        observer=missing_receipt, audit=audit, extra_capture_planes_mm=(76.5,))
    assert report["execution_status"] == "FAILED"
    assert "extra capture planes" in report["errors"][0]["message"]


def test_step_comparison_requires_identical_extra_observations():
    left, right = run_row(), run_row()
    left["extra_capture_planes_mm"] = [76.5]
    assert not validator.compare_observations(left, right)["passed"]


def test_cli_repeated_extra_capture_coordinates_are_forwarded(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(validator, "load_candidate", lambda _: (MockState(), {}))
    def validate(state, output, **kwargs):
        seen.update(kwargs)
        return dict(execution_status="COMPLETE", passed=False, qualification="NOT_QUALIFIED")
    monkeypatch.setattr(validator, "validate_state", validate)
    assert validator.main(["--candidate", "unused.json", "--output", str(tmp_path / "cli"),
        "--single-step", "--extra-capture-z-mm", "76.49", "--extra-capture-z-mm", "76.51"]) == 0
    assert seen["extra_capture_planes_mm"] == [76.49, 76.51]


@pytest.mark.parametrize("error", [ValueError("mock failure"), KeyboardInterrupt()])
def test_failure_and_cancellation_are_saved(tmp_path, execution, error):
    calls, observer, audit = execution
    def fail(state, **kwargs):
        raise error
    report = validator.validate_state(MockState(), tmp_path / "failed", observer=fail, audit=audit)
    assert not report["passed"]
    assert report["execution_status"] == ("CANCELLED" if isinstance(error, KeyboardInterrupt) else "FAILED")
    saved = json.loads((tmp_path / "failed" / "report.json").read_text())
    assert saved["runs"][0]["execution_status"] == report["execution_status"]


def test_audit_failure_prevents_propagation(tmp_path, execution):
    calls, observer, audit = execution
    report = validator.validate_state(MockState(), tmp_path / "audit-failed", observer=observer,
        audit=lambda state: dict(status="FAIL", passed=False))
    assert not calls and report["execution_status"] == "FAILED"


def test_input_mutation_and_mismatched_receipts_fail_closed(tmp_path, execution):
    calls, observer, audit = execution
    def mutated(state, **kwargs):
        record = observer(state, **kwargs)
        state.control = 42.
        return record
    state = MockState()
    report = validator.validate_state(state, tmp_path / "mutated", observer=mutated, audit=audit)
    assert report["execution_status"] == "FAILED" and state.control == 12.34
    def wrong(state, **kwargs):
        record = observer(state, **kwargs)
        record["snapshot_digest"] = "different"
        return record
    report = validator.validate_state(state, tmp_path / "receipt", observer=wrong, audit=audit)
    assert report["execution_status"] == "FAILED"


def test_existing_output_is_never_overwritten(tmp_path, execution):
    calls, observer, audit = execution
    with pytest.raises(FileExistsError):
        validator.validate_state(MockState(), tmp_path, observer=observer, audit=audit)
    assert not calls


@pytest.fixture
def saved_candidate(tmp_path, monkeypatch):
    import temsim.optics.column as column
    import temsim.instrument_snapshot as snapshots
    class SavedState:
        def __init__(self, public, hidden=None):
            self.public = deepcopy(public)
            if "sample" in self.public:
                for key in ("in_plane_axis_uvw", "specimen_orientation_quaternion_wxyz", "zone_axis_uvw"):
                    self.public["sample"][key] = tuple(self.public["sample"][key])
            self.hidden_quality = hidden

        def to_dict(self):
            return self.public

        @staticmethod
        def from_dict(value):
            return SavedState(value)
    monkeypatch.setattr(column, "State", SavedState)
    class SavedSnapshot:
        @staticmethod
        def from_dict(value):
            restored = SavedState(value["graph"]["public"], value["graph"]["hidden_quality"])
            return SimpleNamespace(digest=value["digest"], restore=lambda: restored)
    monkeypatch.setattr(snapshots, "InstrumentSnapshot", SavedSnapshot)
    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", lambda state: SimpleNamespace(digest="snapshot-a"))
    directory = tmp_path / "candidates" / "trial-000001"
    directory.mkdir(parents=True)
    payload = dict(lenses=[{"key": "objective", "percent": 29.3}],
        c2_radius_mm=.05, multipoles={"hp1": 123.4}, source="unchanged",
        sample=dict(in_plane_axis_uvw=[1, 0, 0], specimen_orientation_quaternion_wxyz=[1., 0., 0., 0.], zone_axis_uvw=[0, 0, 1]))
    (directory / "state.json").write_text(json.dumps(payload))
    (directory / "candidate.json").write_text(json.dumps({"proposal_only": True}))
    snapshot = dict(graph=dict(public=payload, hidden_quality="High accuracy"),
        external_inputs=[], implementation="solver-a", digest="snapshot-a")
    (directory / "inputs.json").write_text(json.dumps(snapshot))
    best = tmp_path / "best.json"
    best.write_text(json.dumps({"directory": "candidates/trial-000001", "files_sha256": {
        name: validator.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("state.json", "inputs.json", "candidate.json")}}))
    return directory, best, payload


@pytest.mark.parametrize("entry", ["state.json", "inputs.json", "candidate.json", "best.json", "directory"])
def test_solver_candidate_resolution_keeps_full_state(saved_candidate, entry):
    directory, best, payload = saved_candidate
    path = best if entry == "best.json" else directory if entry == "directory" else directory / entry
    state, identity = validator.load_candidate(path)
    assert validator._digest(state.to_dict()) == validator._digest(payload)
    assert isinstance(state.to_dict()["sample"]["zone_axis_uvw"], tuple)
    assert state.hidden_quality == "High accuracy"
    assert identity["path"] == str((directory / "inputs.json").resolve())
    assert identity["restoration_kind"] == "complete_snapshot"
    assert identity["snapshot_identity"] == "snapshot-a"


@pytest.mark.parametrize("change", ["metadata", "c2", "multipole"])
def test_solver_pointer_rejects_changed_candidate_fields(saved_candidate, change):
    directory, best, payload = saved_candidate
    path = directory / "inputs.json"
    snapshot = json.loads(path.read_text())
    if change == "metadata":
        snapshot["graph"]["hidden_quality"] = "Preview"
    elif change == "c2":
        snapshot["graph"]["public"]["c2_radius_mm"] = .1
    else:
        snapshot["graph"]["public"]["multipoles"]["hp1"] = 0.
    path.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validator.load_candidate(best)


@pytest.mark.parametrize("change", ["c2", "multipole", "orientation", "boolean_for_number"])
def test_explicit_state_must_agree_with_paired_complete_snapshot(saved_candidate, change):
    directory, best, payload = saved_candidate
    modified = deepcopy(payload)
    if change == "c2":
        modified["c2_radius_mm"] = .1
    elif change == "multipole":
        modified["multipoles"]["hp1"] = 0.
    elif change == "orientation":
        modified["sample"]["zone_axis_uvw"] = [1, 0, 0]
    else:
        modified["sample"]["zone_axis_uvw"] = [0, 0, True]
    path = directory / "state.json"
    path.write_text(json.dumps(modified))
    with pytest.raises(ValueError, match="disagrees"):
        validator.load_candidate(path)


def test_solver_pointer_requires_pinned_complete_inputs(saved_candidate):
    directory, best, payload = saved_candidate
    best.write_text(json.dumps({"directory": "candidates/trial-000001"}))
    with pytest.raises(ValueError, match="pin the complete snapshot"):
        validator.load_candidate(best)


def test_complete_snapshot_cannot_normalize_fields_while_restoring(saved_candidate, monkeypatch):
    directory, best, payload = saved_candidate
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot",
        lambda state: SimpleNamespace(digest="changed-input-graph"))
    with pytest.raises(ValueError, match="changed its exact input graph"):
        validator.load_candidate(directory / "inputs.json")


def test_snapshot_restoration_supports_immutable_mapping_inputs(monkeypatch):
    restored = SimpleNamespace(inputs=MappingProxyType({"quality": "High accuracy"}))
    snapshot = SimpleNamespace(digest="exact", restore=lambda: restored)
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot",
        lambda state: SimpleNamespace(digest="exact"))
    assert validator._restore_exact(snapshot) is restored


def test_paired_state_does_not_sanitize_nonfinite_values(saved_candidate):
    directory, best, payload = saved_candidate
    payload["multipoles"]["hp1"] = float("nan")
    path = directory / "state.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        validator.load_candidate(path)


def test_candidate_identity_binds_hidden_metadata_but_excludes_only_step_and_count(execution):
    state = MockState()
    identity = validator._candidate_identity(state)
    state.step_mm = .1
    state.electron_gun.emitter.ray_count = 769
    assert validator._candidate_identity(state) == identity
    state.hidden_quality = "Preview"
    assert validator._candidate_identity(state) != identity
    state.hidden_quality = "High accuracy"
    state.control += .1
    assert validator._candidate_identity(state) != identity


def test_loading_a_state_that_is_silently_normalized_is_rejected(tmp_path, monkeypatch):
    import temsim.optics.column as column
    class ChangedState:
        @staticmethod
        def from_dict(value):
            return SimpleNamespace(to_dict=lambda: {"changed": True})
    monkeypatch.setattr(column, "State", ChangedState)
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"unchanged": True}))
    with pytest.raises(ValueError, match="changed"):
        validator.load_candidate(path)
