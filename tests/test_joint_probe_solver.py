"""Synthetic solver contracts only; these tests execute no physical transport."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("joint_probe_solver_script",
    Path(__file__).resolve().parents[1] / "scripts" / "solve_joint_probe.py")
solver = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = solver
_spec.loader.exec_module(solver)


def fake_state():
    parts = {}
    for _, key, field, _, _ in solver.DEFINITIONS:
        part = parts.setdefault(key, SimpleNamespace(key=key, enabled=True, z_mm=2.,
                                                    max_percent=100., maximum_strength_m3=1e7,
                                                    maximum_strength_m2=300.))
        setattr(part, field, 20. if field == "percent" else 0.)
    state = SimpleNamespace(lenses=[parts[key] for key in solver.LENSES],
        corrector_elements=[item for key, item in parts.items() if "hexapole" in key or "quadrupole" in key],
        stigmators=[parts["condenser_stigmator"]], deflectors=[],
        sample=SimpleNamespace(upper_surface_z_mm=10., z_mm=10.1),
        condenser_aperture_2=SimpleNamespace(radius_mm=.05, enabled=True, z_mm=3., offset_x_mm=0., offset_y_mm=0.))
    state.to_dict = lambda: {"synthetic_fixture": True}
    return state


def observation(state=None, count=20, offset=0.):
    x = np.linspace(-.1e-9, .1e-9, count) + offset
    arrays = {"x_m": np.tile(x, (3, 1)), "y_m": np.tile(x[::-1], (3, 1)),
              "source_ray_id": np.arange(count, dtype=np.int64), "weights": np.full(count, .25),
              "alive": np.ones(count, dtype=bool)}
    return solver.Observation({"emitted_identity_digest": "fixed-original-source", "effective_points": float(count),
        "local_waist_nm": 0., "covariance_waist_nm": 0., "variance_second_derivative": 1.,
        "alpha95_mrad": 30., "d95_nm": .2}, arrays, fake_state() if state is None else state)


class SyntheticEvaluator:
    """No physical solver is called; only bookkeeping and optimizer contracts."""
    def __init__(self, state=None):
        self.state = fake_state() if state is None else state
        self.calls = []
        self.cancelled = lambda: False
        self.fail = False

    def evaluate(self, values):
        self.calls.append(np.array(values).copy())
        if self.fail:
            raise RuntimeError("Synthetic evaluation failure")
        result = observation(self.state)
        result.arrays["x_m"] *= 1 + (values[0]-21.)**2
        return result


@pytest.fixture
def fake_snapshot(monkeypatch):
    import temsim.instrument_snapshot
    monkeypatch.setattr(temsim.instrument_snapshot, "capture_instrument_snapshot",
                        lambda state: SimpleNamespace(to_dict=lambda: {"synthetic_snapshot": True}))


def search(tmp_path, fake_snapshot, **kwargs):
    evaluator = SyntheticEvaluator()
    controls = solver.control_space(evaluator.state, ("c1",))
    return solver.Search(evaluator, controls, tmp_path / "run", **kwargs)


@pytest.mark.parametrize("value", [True, None, "1", float("nan"), float("inf")])
def test_finite_rejects_ambiguous_values(value):
    with pytest.raises(ValueError):
        solver.finite(value, "value")


@pytest.mark.parametrize("text", ['{"a":NaN}', '{"a":Infinity}', '{"a":1e999}', '{"a":1,"a":2}'])
def test_strict_seed_json(tmp_path, text):
    path = tmp_path / "seed.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        solver.read_json(path)


def test_real_fixture_is_static_and_preserves_aperture_and_source_budget():
    state = solver.make_state(193, .2)
    assert state.condenser_aperture_2.radius_mm == .05
    assert state.electron_gun.emitter.ray_count == 193
    assert state.step_mm == .2
    assert len(solver.control_space(state)) == 18
    assert {item.key: item for item in state.lenses}["objective_lens"].percent == 29.293344328471793
    payload = state.to_dict()
    restored = type(state).from_dict(payload)
    assert restored.to_dict() == payload
    assert solver.seed_controls(restored) == solver.seed_controls(state)
    from temsim.instrument_snapshot import capture_instrument_snapshot
    snapshot = capture_instrument_snapshot(state)
    restored = snapshot.restore()
    assert capture_instrument_snapshot(restored).digest == snapshot.digest
    assert solver.seed_controls(restored) == solver.seed_controls(state)


@pytest.mark.parametrize("rays,step", [(True, .2), (15, .2), (193, .3), (193, 0), (193, float("nan"))])
def test_unbounded_fixture_rejected_before_source_execution(rays, step):
    with pytest.raises(ValueError):
        solver.make_state(rays, step)


def test_objective_preparation_records_only_proved_derived_planes_and_stabilizes_gun_signature():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.particle_sections import gun_dependency_signature
    state = solver.make_state(193, .2, seed=None)
    solver.load_seed(state, solver.DEFAULT_SEED)
    controls = solver.seed_controls(state)
    report = solver.prepare_objective_observations(state)
    changes = report["derived_observation_changes"]
    assert len(changes) == 4
    assert {item["path"].rsplit("/", 1)[-1] for item in changes} == {
        "objective_back_focal_plane_z_mm", "objective_image_plane_z_mm",
        "_back_focal_plane_z_mm", "_image_plane_z_mm"}
    assert solver.seed_controls(state) == controls
    assert any(item["before"] == 1605.1936767812895 and item["after"] == 1614.0726662366578 for item in changes)
    assert any(item["before"] == 1612.4780373979831 and item["after"] is None for item in changes)
    before = capture_instrument_snapshot(state).digest
    gun_dependency_signature(state)  # Captures static providers only; no gun or column propagation.
    assert capture_instrument_snapshot(state).digest == before
    assert solver.prepare_objective_observations(state)["derived_observation_changes"] == []


def test_objective_preparation_rejects_any_physical_control_mutation(monkeypatch):
    state = solver.make_state(193, .2)
    original = type(state).sync_objective
    def changed(instance):
        original(instance)
        instance.lenses[0].percent += .1
    monkeypatch.setattr(type(state), "sync_objective", changed)
    with pytest.raises(ValueError, match="changed physical inputs"):
        solver.prepare_objective_observations(state)


def test_controls_are_lossless_at_seed_and_do_not_wrap_angles():
    state = fake_state()
    parts = solver.components(state)
    parts["probe_hp1_hexapole"].orientation_rad = 17.123456789012345
    controls = solver.control_space(state)
    values = solver.physical_values(controls, np.zeros(len(controls)))
    assert np.array_equal(values, [c.initial for c in controls])
    solver.set_values(state, controls, values)
    assert parts["probe_hp1_hexapole"].orientation_rad == 17.123456789012345
    assert all(c.lower <= c.initial <= c.upper for c in controls)


@pytest.mark.parametrize("names,bounds", [(("c1", "c1"), {}), (("unknown",), {}),
    (("c1",), {"c2": [1, 30]}), (("c1",), {"c1": [-1, 100]}),
    (("c1",), {"c1": [21, 30]}), (("hp1",), {"hp1": [-2e7, 2e7]})])
def test_explicit_bounds_reject_invalid_controls(names, bounds):
    with pytest.raises(ValueError):
        solver.control_space(fake_state(), names, bounds)


def test_seed_only_changes_declared_physical_values(tmp_path):
    state = fake_state()
    data = solver.seed_controls(state)
    data["controls_percent"]["condenser_lens_1"] = 45.123456789
    data["probe_field_controls"]["probe_hp1_hexapole"]["strength_m3"] = 1234567.89
    data["old_result"] = {"d95_nm": .1, "status": "PASS"}
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(data))
    provenance = solver.load_seed(state, path)
    assert state.lenses[0].percent == 45.123456789
    assert solver.components(state)["probe_hp1_hexapole"].strength_m3 == 1234567.89
    assert "historical observations are not reused" in provenance["scope"]
    data["probe_field_controls"]["probe_hp1_hexapole"]["fake_cs"] = 2.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsupported"):
        solver.load_seed(state, path)


def test_local_objective_centers_without_rescaling_saved_arrays():
    baseline = observation()
    cohort = solver.LocalCohort(baseline)
    moved = observation(offset=1e-6)
    before = {key: value.copy() for key, value in moved.arrays.items()}
    residual = cohort.residual(moved, solver.Objective())
    assert np.allclose(residual, cohort.residual(baseline, solver.Objective()), atol=1e-12, rtol=0)
    assert all(np.array_equal(moved.arrays[key], value) for key, value in before.items())
    assert np.all(moved.arrays["weights"] == .25)


def test_actual_loss_is_rejected_for_local_fit_not_reweighted():
    base = observation()
    cohort = solver.LocalCohort(base)
    changed = deepcopy(base)
    changed.arrays["alive"][0] = False
    with pytest.raises(solver.CohortRejected, match="physical stop"):
        cohort.residual(changed, solver.Objective())


def test_total_physical_loss_returns_fixed_local_penalty_without_aborting(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    initial = run.residual([0.])
    empty = observation()
    empty.arrays["alive"][:] = False
    empty.report.update(alpha95_mrad=None, d95_nm=None, local_waist_nm=None, covariance_waist_nm=None)
    run.evaluator.evaluate = lambda values: empty
    penalty = run.residual([.1])
    assert penalty.shape == initial.shape
    assert np.all(penalty == 1e6)
    assert solver.read_json(run.output / "trials/000002.json")["status"] == "REJECTED_LOCAL_COHORT"
    assert run.best["trial"] == 1


def test_new_surviving_particles_are_allowed_without_changing_residual_dimension():
    base = observation(count=21)
    base.arrays["alive"][-1] = False
    cohort = solver.LocalCohort(base)
    changed = deepcopy(base)
    changed.arrays["alive"][-1] = True
    assert cohort.residual(changed, solver.Objective()).shape == (43,)


@pytest.mark.parametrize("change", ["weights", "source_ray_id", "identity", "concave"])
def test_source_or_invalid_focus_cannot_improve_local_cost(change):
    base = observation()
    cohort = solver.LocalCohort(base)
    changed = deepcopy(base)
    if change in {"weights", "source_ray_id"}:
        changed.arrays[change][0] += 1
    elif change == "identity":
        changed.report["emitted_identity_digest"] = "changed"
    else:
        changed.report["variance_second_derivative"] = -1
    with pytest.raises(ValueError):
        cohort.residual(changed, solver.Objective())


def test_absolute_jacobian_probes_zero_parameter_and_obeys_bounds():
    calls = []
    def function(values):
        calls.append(values.copy())
        return np.array((2*values[0]+3*values[1], values[0]-values[1]))
    matrix = solver.absolute_jacobian(function, np.array((0., 1.)), (np.array((-1., -1.)), np.ones(2)), 1e-4)
    assert np.allclose(matrix, ((2., 3.), (1., -1.)), rtol=1e-11, atol=1e-11)
    assert calls[1][0] == 1e-4
    assert calls[2][1] == 1.-1e-4


def test_actual_budget_counts_jacobian_and_preserves_best(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot, max_evaluations=2)
    result = run.run("fit")
    assert result["status"] == "STOPPED"
    assert len(run.evaluator.calls) == result["actual_evaluations"] == 2
    assert len(result["trials"]) == 2
    assert result["best"] is not None
    assert all(item["status"] == "COMPLETE" for item in result["trials"])
    for row in (run.output / "trials").glob("*.json"):
        assert solver.read_json(row)["status"] != "RUNNING"


def test_cached_exact_vector_does_not_consume_actual_budget(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot, max_evaluations=1)
    first = run.residual([0.])
    second = run.residual([0.])
    assert np.array_equal(first, second)
    assert run.evaluations == len(run.evaluator.calls) == 1
    with pytest.raises(solver.SearchStopped):
        run.residual([.1])


def test_failure_receipt_clears_running_trial(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    run.evaluator.fail = True
    result = run.run("measure")
    assert result["status"] == "FAILED"
    assert "active_trial" not in result
    assert solver.read_json(run.output / "trials/000001.json")["status"] == "FAILED"
    assert not (run.output / "best.json").exists()


def test_measure_does_not_apply_local_cohort_acceptance(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    run.evaluator.evaluate = lambda values: observation(count=3)
    result = run.run("measure")
    assert result["status"] == "COMPLETE"
    assert result["best"] is not None
    trial = solver.read_json(run.output / "trials/000001.json")
    assert trial["objective_status"] == "NOT_EVALUATED"
    assert trial["objective_squared_norm"] is None


def test_completed_best_files_are_immutable_and_pointer_hashes_match(tmp_path, fake_snapshot):
    import hashlib
    run = search(tmp_path, fake_snapshot)
    run.run("measure")
    pointer = solver.read_json(run.output / "best.json")
    folder = solver.resolve_candidate_directory(run.output)
    for name, digest in pointer["files_sha256"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    with np.load(folder / "particles.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["weights"], np.full(20, .25))
    with pytest.raises(FileExistsError):
        solver.Search(run.evaluator, run.controls, run.output)


def test_cancel_before_first_trace_is_durable(tmp_path, fake_snapshot):
    cancel = tmp_path / "cancel"
    cancel.touch()
    run = search(tmp_path, fake_snapshot, cancel_file=cancel)
    assert run.run("fit")["status"] == "STOPPED"
    assert run.evaluations == 0
    assert not run.evaluator.calls


def test_best_pointer_cannot_escape_run(tmp_path):
    (tmp_path / "best.json").write_text('{"directory":"../elsewhere"}')
    with pytest.raises(ValueError, match="within"):
        solver.resolve_candidate_directory(tmp_path)


def test_exact_c2_plane_keeps_contact_loss_and_absolute_current():
    state = fake_state()
    cp = SimpleNamespace(z_mm=np.array([3.]), unified_state=SimpleNamespace(active=np.array([[True, False, False]])))
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        setattr(cp, key, np.array([[1e-5, 6e-5, np.nan]]))
    from temsim.physics.simulation import Branch
    branch = Branch("incident", (), cp.z_mm, cp.x_m, cp.y_m, cp.tx_rad, cp.ty_rad,
        np.array([True, False, False]), np.array([np.nan, 3., 2.]), ["", "c2", "upstream"], 1., np.zeros(3))
    segment = SimpleNamespace(checkpoints=cp, branch=branch)
    report, arrays = solver.aperture_observation(state, segment, np.ones(3, bool), np.array([.2, .3, .5]), np.arange(3), 1e-5)
    assert report["diameter_um"] == 100.
    assert report["incoming"]["source_ids"] == [0, 1]
    assert report["outgoing"]["source_ids"] == [0]
    assert report["incoming"]["current_a"] == 5e-6
    assert report["outgoing"]["current_a"] == pytest.approx(2e-6)
    assert np.array_equal(arrays["c2_incoming_alive"], [True, True, False])
    cp.z_mm[0] = np.nextafter(3., 4.)
    with pytest.raises(ValueError, match="exact"):
        solver.aperture_observation(state, segment, np.ones(3, bool), np.ones(3), np.arange(3), 1e-5)


def test_production_adapter_preserves_clock_energy_stops_and_retraces_changed_gun_dependencies(monkeypatch):
    """Real result data classes, synthetic transport seam; no field integration."""
    from temsim.optics.electron_gun.base import GunTraceResult, GunExitBundle
    from temsim.optics.electron_gun import source
    from temsim.physics import particle_sections, simulation
    from temsim.physics.core import PropagationCheckpoints
    from temsim.physics.unified_particle_restart import UnifiedParticleCheckpointState
    from temsim.instrument_snapshot import capture_instrument_snapshot

    state = solver.make_state(193, .2)
    controls = solver.control_space(state, ("c1",))
    initial_digest = capture_instrument_snapshot(state).digest
    n = 20
    angle = np.arange(n)*2*np.pi/n
    x, y = np.cos(angle)*1e-10, np.sin(angle)*1e-10
    tx, ty = np.cos(angle)*.03, np.sin(angle)*.03
    weights, ids = np.full(n, .05), np.arange(n, dtype=np.int64)
    times = np.linspace(2e-9, 3e-9, n)
    alive = np.ones(n, dtype=bool)
    alive[0], times[0] = False, np.nan
    offsets = np.linspace(-.1, .1, n)
    c2 = float(state.condenser_aperture_2.z_mm)
    calls = {"guns": [], "segments": [], "lose_all": False, "drift": False}

    def gun_trace(candidate):
        shift = (candidate.lenses[0].percent-controls[0].initial)*1e-7
        emitted = GunExitBundle(x+shift, y.copy(), tx.copy(), ty.copy(), offsets.copy(), weights.copy(),
            ids.copy(), alive.copy(), times.copy())
        history = [np.tile(v, (2, 1)) for v in (emitted.x_m, y, tx, ty)]
        gun = GunTraceResult(np.array([0., candidate.electron_gun.exit_plane_z_mm]), *history, emitted,
            np.r_[10., np.full(n-1, np.nan)], tuple(["gun_stop"]+[""]*(n-1)), 1e-5, None, None,
            emission_reference={"ray_id": ids.copy(), "position_m": np.c_[x, y, np.zeros(n)],
                "direction": np.c_[np.zeros(n), np.zeros(n), np.ones(n)]},
            flight_time_s=np.tile(times, (2, 1)))
        calls["guns"].append(gun)
        return gun

    def segment(candidate, name, start, end, initial, energy, weight, identity, survival,
                events, previous, boundary, **options):
        gun = calls["guns"][-1]
        assert name == "incident" and boundary == np.inf
        assert candidate is not state
        assert options["initial_kicks"] is True
        for actual, expected in zip(initial, (gun.exit_bundle.x_m, tx, y, ty, times)):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(energy, offsets)
        np.testing.assert_array_equal(options["initial_kinetic_energy_ev"], candidate.electron_gun.nominal_exit_energy_ev+offsets)
        np.testing.assert_array_equal(weight, weights)
        np.testing.assert_array_equal(survival[0], alive)
        assert survival[2][0] == "gun_stop"
        planes = np.sort(np.asarray(options["save_z_mm"]))
        assert len(planes) == 4 and c2 in planes and planes[-1] == end
        dz = (planes-end)[:, None]*1e-3
        xx, yy = x + dz*tx, y + dz*ty
        xx[0], yy[0] = x*1e5, y*1e5
        phase_shape = xx.shape
        final_alive = alive.copy()
        final_alive[1] = False
        if calls["lose_all"]:
            final_alive[:] = False
        stops = np.where(final_alive, np.nan, c2)
        stops[0] = 10.
        internal = UnifiedParticleCheckpointState(candidate.electron_gun.nominal_exit_energy_ev+offsets,
            np.zeros(n), np.zeros((4, n)), np.tile(final_alive, (4, 1)), stops,
            tuple("stop" if not a else "" for a in final_alive), "a"*64, "b"*64, 0.,
            np.zeros((4, 2, 3)), ("c"*64,)*4)
        cp = PropagationCheckpoints(planes, xx, np.broadcast_to(tx, phase_shape), yy,
            np.broadcast_to(ty, phase_shape), np.tile(times, (4, 1)),
            np.tile(candidate.electron_gun.nominal_exit_energy_ev+offsets, (4, 1)),
            internal)
        branch = simulation.Branch("incident", (), planes, cp.x_m, cp.y_m, cp.tx_rad, cp.ty_rad,
            final_alive, stops, ["stop" if not a else "" for a in final_alive], 1., offsets.copy(), weights.copy(),
            source_ray_id=ids.copy(), source_azimuth_rad=identity[1],
            flight_time_s=cp.flight_time_s, kinetic_energy_ev=cp.kinetic_energy_ev)
        result = particle_sections.ParticleSectionSegment("incident", branch, object(), cp, "synthetic")
        calls["segments"].append((result, previous))
        if calls["drift"]:
            candidate.lenses[0].percent += .1
        return result, start, previous is not None

    monkeypatch.setattr(source, "trace_source_to_exit", gun_trace)
    monkeypatch.setattr(particle_sections, "gun_dependency_signature", lambda s: "physical-c1-"+repr(s.lenses[0].percent))
    monkeypatch.setattr(particle_sections, "_events", lambda s: ())
    monkeypatch.setattr(particle_sections, "_trace_segment", segment)
    monkeypatch.setattr(simulation, "_validate_gun_flight_times", lambda gun: any(gun is item for item in calls["guns"]))
    monkeypatch.setitem(sys.modules, "audit_joint_probe_fields", SimpleNamespace(audit=lambda s, plan: {"passed": True}))
    evaluator = solver.JointEvaluator(state, controls)
    first = evaluator.evaluate([controls[0].initial])
    second = evaluator.evaluate([controls[0].initial])
    assert len(calls["guns"]) == 1
    assert second.report["gun_reused"] and second.report["column_prefix_reused"]
    assert calls["segments"][1][1] is calls["segments"][0][0]
    third = evaluator.evaluate([controls[0].initial+.001])
    assert len(calls["guns"]) == 2 and calls["segments"][2][1] is None
    assert third.report["emitted_identity_digest"] == first.report["emitted_identity_digest"]
    assert third.report["gun_exit_population_digest"] != first.report["gun_exit_population_digest"]
    np.testing.assert_array_equal(third.arrays["weights"], weights)
    np.testing.assert_array_equal(third.arrays["gun_exit_flight_time_s"], times)
    assert third.arrays["blocked_z_mm"][0] == 10.
    assert third.arrays["blocked_z_mm"][1] == c2
    assert third.report["c2_aperture"]["incoming"]["points"] == 19
    assert third.report["c2_aperture"]["outgoing"]["points"] == 18
    assert 0 <= third.report["d90_nm"] <= third.report["d95_nm"]
    assert capture_instrument_snapshot(state).digest == initial_digest
    calls["lose_all"] = True
    empty = evaluator.evaluate([controls[0].initial+.002])
    assert empty.report["population_status"] == "NO_TRANSMISSION"
    assert empty.report["d95_nm"] is None and empty.report["current_a"] == 0.
    assert empty.report["d90_nm"] is None
    assert not empty.arrays["alive"].any()
    calls["drift"] = True
    with pytest.raises(ValueError, match="Inputs changed during physical evaluation"):
        evaluator.evaluate([controls[0].initial+.003])


def test_source_identity_failure_aborts_instead_of_becoming_physical_loss_penalty(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    run.residual([0.])
    invalid = observation()
    invalid.report["emitted_identity_digest"] = "changed-original-source"
    run.evaluator.evaluate = lambda values: invalid
    with pytest.raises(ValueError, match="emission"):
        run.residual([.1])
    assert solver.read_json(run.output / "trials/000002.json")["status"] == "FAILED"


@pytest.mark.parametrize("changes", [
    {"d95_target_nm": .65}, {"d95_scale_nm": .2},
    {"d95_target_nm": 0., "d95_scale_nm": .2},
    {"d95_target_nm": .65, "d95_scale_nm": -1.},
    {"d95_target_nm": float("nan"), "d95_scale_nm": .2},
    {"d95_target_nm": .65, "d95_scale_nm": float("inf")},
    {"d95_target_nm": True, "d95_scale_nm": .2},
    {"position_scale_nm": None},
])
def test_d95_options_are_paired_positive_finite_controls(changes):
    with pytest.raises(ValueError):
        solver.Objective(**changes).validate()


@pytest.mark.parametrize("diameter,expected", [(.4, 0.), (.65, 0.), (.85, 1.), (2.25, 8.)])
def test_d95_hinge_adds_only_the_exact_full_current_diameter_term(diameter, expected):
    measured = observation()
    measured.report["d95_nm"] = diameter
    before = {key: value.copy() for key, value in measured.arrays.items()}
    baseline = solver.LocalCohort(measured).residual(measured, solver.Objective())
    objective = solver.Objective(d95_target_nm=.65, d95_scale_nm=.2).validate()
    cohort = solver.LocalCohort(measured, d95_enabled=True)
    residual = cohort.residual(measured, objective)
    np.testing.assert_array_equal(residual[:-1], baseline)
    assert residual[-1] == pytest.approx(expected)
    assert len(residual) == len(baseline)+1 == cohort.dimension
    assert all(np.array_equal(measured.arrays[key], value) for key, value in before.items())
    assert measured.report["d95_nm"] == diameter


def test_d95_disabled_preserves_original_dimension_and_ignores_size_metric():
    measured = observation()
    cohort = solver.LocalCohort(measured)
    first = cohort.residual(measured, solver.Objective())
    measured.report["d95_nm"] = 1000.
    second = cohort.residual(measured, solver.Objective())
    np.testing.assert_array_equal(first, second)
    assert len(first) == 2*20+3


def test_d95_uses_full_current_metric_when_new_particle_enters_local_fit():
    base = observation(count=21)
    base.arrays["alive"][-1] = False
    cohort = solver.LocalCohort(base, d95_enabled=True)
    changed = deepcopy(base)
    changed.arrays["alive"][-1] = True
    changed.arrays["x_m"][-1, -1] = 1e-7
    changed.report["d95_nm"] = 1.25
    result = cohort.residual(changed, solver.Objective(d95_target_nm=.65, d95_scale_nm=.2))
    assert result.shape == (44,)
    assert result[-1] == pytest.approx(3.)
    np.testing.assert_array_equal(changed.arrays["weights"], np.full(21, .25))


def test_d95_rejection_penalty_has_same_dimension_and_cannot_replace_best(tmp_path, fake_snapshot):
    objective = solver.Objective(d95_target_nm=.65, d95_scale_nm=.2)
    run = search(tmp_path, fake_snapshot, objective=objective)
    base = run.residual([0.])
    lost = observation()
    lost.arrays["alive"][0] = False
    lost.report["d95_nm"] = .01  # A smaller clipped probe must never improve this local fit.
    run.evaluator.evaluate = lambda values: lost
    rejected = run.residual([.1])
    assert rejected.shape == base.shape == (44,)
    assert float(rejected @ rejected) > float(base @ base)
    assert run.best["trial"] == 1
    assert solver.read_json(run.output / "trials/000002.json")["status"] == "REJECTED_LOCAL_COHORT"
    saved = solver.read_json(run.output / "run.json")["objective"]
    assert saved["d95_target_nm"] == .65 and saved["d95_scale_nm"] == .2


def test_d95_dimension_cannot_change_midfit():
    measured = observation()
    with pytest.raises(ValueError, match="dimension"):
        solver.LocalCohort(measured).residual(measured, solver.Objective(d95_target_nm=.65, d95_scale_nm=.2))


@pytest.mark.parametrize("options", [
    ["--d95-target-nm", ".65"], ["--d95-scale-nm", ".2"],
    ["--d95-target-nm", "0", "--d95-scale-nm", ".2"],
    ["--d95-target-nm", ".65", "--d95-scale-nm", "nan"],
])
def test_invalid_cli_d95_options_fail_before_preparation_or_output(monkeypatch, tmp_path, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid objective must not prepare physical inputs")
    monkeypatch.setattr(solver, "make_state", forbidden)
    output = tmp_path / "new-run"
    with pytest.raises(SystemExit) as error:
        solver.main(["fit", "--output", str(output), *options])
    assert error.value.code == 2
    assert not output.exists()


def test_existing_quadrupoles_are_opt_in_signed_amplitudes_and_roundtrip_exactly():
    state = solver.make_state(193, .2)
    names = ("qph2", "qpc", "qph1", "qpol")
    assert len(solver.control_space(state)) == len(solver.DEFAULT_CONTROLS) == 18
    assert not set(names) & set(solver.DEFAULT_CONTROLS)
    controls = solver.control_space(state, names)
    assert len(controls) == 4
    assert all(c.scale == 1. and c.lower == -30. and c.upper == 30. and c.field == "strength_m2" for c in controls)
    parts = solver.components(state)
    assert all(parts[c.key].maximum_strength_m2 == 300. for c in controls)
    assert all(not hasattr(parts[c.key], "orientation_rad") for c in controls)
    values = (12.3456789012345, -23.456789012345, 7.123456789012345, -.123456789012345)
    solver.set_values(state, controls, values)
    payload = state.to_dict()
    restored = type(state).from_dict(payload)
    assert restored.to_dict() == payload
    restored_parts = solver.components(restored)
    assert tuple(getattr(restored_parts[c.key], c.field) for c in controls) == values
    saved = solver.seed_controls(state)["probe_field_controls"]
    assert all(saved[c.key] == {"strength_m2": value} for c, value in zip(controls, values))


@pytest.mark.parametrize("name", ["qph2", "qpc", "qph1", "qpol"])
def test_existing_quadrupole_hardware_bounds_are_not_expanded(name):
    state = fake_state()
    control, = solver.control_space(state, (name,), {name: [-300., 300.]})
    assert (control.lower, control.upper) == (-300., 300.)
    with pytest.raises(ValueError, match="physical limits"):
        solver.control_space(state, (name,), {name: [-301., 300.]})


def test_smoothed_d95_point_and_ring_have_known_logistic_floor():
    tau = .05
    zero = np.zeros(4)
    alive = np.ones(4, bool)
    weights = np.array([1., 1., 4., 4.])
    floor = 2*tau*np.log(19.)
    assert solver.smoothed_current_d95_nm(zero, zero, alive, weights, tau) == pytest.approx(floor, rel=1e-13)
    x, y = np.array([1., -1., 0., 0.])*1e-9, np.array([0., 0., 1., -1.])*1e-9
    assert solver.smoothed_current_d95_nm(x, y, alive, weights, tau) == pytest.approx(2.+floor, rel=1e-13)


def test_smoothed_d95_translation_weight_scale_and_original_arrays_are_preserved():
    x = np.array([-.7, .4, .1, 1.2])*1e-9
    y = np.array([.2, -.5, .3, .8])*1e-9
    alive, weights = np.ones(4, bool), np.array([1., 2., 4., 3.])
    before = [v.copy() for v in (x, y, alive, weights)]
    value = solver.smoothed_current_d95_nm(x, y, alive, weights, .05)
    shifted = solver.smoothed_current_d95_nm(x+1e-6, y-2e-6, alive, weights*11., .05)
    assert shifted == pytest.approx(value, abs=1e-10)
    for actual, expected in zip((x, y, alive, weights), before):
        np.testing.assert_array_equal(actual, expected)
    from scipy.special import expit
    w = weights/weights.sum()
    radii = np.hypot(x-np.sum(w*x), y-np.sum(w*y))*1e9
    assert np.sum(w*expit((value/2-radii)/.05)) == pytest.approx(.95, abs=1e-14)


def test_smoothed_d95_uses_actual_full_positive_current_and_new_arrivals():
    measured = observation(count=21)
    measured.arrays["alive"][-1] = False
    cohort = solver.LocalCohort(measured, d95_enabled=True)
    arrived = deepcopy(measured)
    arrived.arrays["alive"][-1] = True
    arrived.arrays["x_m"][-1, -1] = 10e-9
    arrived.report["d95_nm"] = 1.25
    before = deepcopy(arrived)
    objective = solver.Objective(d95_target_nm=.65, d95_scale_nm=.2, d95_smoothing_nm=.05)
    diagnostics = {}
    actual = cohort.residual(arrived, objective, diagnostics=diagnostics)
    full = solver.smoothed_current_d95_nm(arrived.arrays["x_m"][-1], arrived.arrays["y_m"][-1],
        arrived.arrays["alive"], arrived.arrays["weights"], .05)
    subset = solver.smoothed_current_d95_nm(arrived.arrays["x_m"][-1], arrived.arrays["y_m"][-1],
        measured.arrays["alive"], arrived.arrays["weights"], .05)
    assert full > subset
    assert actual[-1] == pytest.approx(max(full-.65, 0)/.2)
    assert diagnostics["smoothed_d95_nm"] == full
    assert diagnostics["native_d95_nm"] == 1.25
    assert diagnostics["status"] == "EVALUATED"
    hard = cohort.residual(arrived, solver.Objective(d95_target_nm=.65, d95_scale_nm=.2))
    np.testing.assert_array_equal(actual[:-1], hard[:-1])
    assert arrived.report == before.report
    for key in before.arrays:
        np.testing.assert_array_equal(arrived.arrays[key], before.arrays[key])


def test_smoothed_d95_ignores_only_zero_current_or_dead_coordinates():
    x = np.array([1e-9, -1e-9, np.nan, np.inf])
    y = np.array([0., 0., np.nan, np.inf])
    result = solver.smoothed_current_d95_nm(x, y, np.array([True, True, False, True]),
        np.array([1., 1., 2., 0.]), .05)
    assert result == pytest.approx(2.+.1*np.log(19.))


def test_smoothed_d95_small_bandwidth_converges_without_claiming_an_empirical_upper_bound():
    # At a discrete 95% boundary the logistic surrogate is not a universal
    # upper bound on the empirical order statistic, despite its point-spot floor.
    x, y = np.array([0., -1., 1.])*1e-9, np.zeros(3)
    alive, weights = np.ones(3, bool), np.array([.94, .03, .03])
    smoothed = solver.smoothed_current_d95_nm(x, y, alive, weights, .05)
    assert 0 < smoothed < 2.
    close = solver.smoothed_current_d95_nm(x, y, alive, weights, 1e-8)
    assert close == pytest.approx(2., abs=4e-8)
    # A tiny positive bandwidth must not get rounded to a zero-radius root.
    point = solver.smoothed_current_d95_nm(y, y, alive, weights, 1e-15)
    assert point == pytest.approx(2e-15*np.log(19.), rel=1e-12, abs=0.)


@pytest.mark.parametrize("bandwidth", [0., -1., float("nan"), float("inf"), True, None])
def test_smoothed_d95_requires_a_positive_finite_bandwidth(bandwidth):
    with pytest.raises(ValueError):
        solver.smoothed_current_d95_nm(np.zeros(2), np.zeros(2), np.ones(2, bool), np.ones(2), bandwidth)


@pytest.mark.parametrize("case", ["negative_weight", "infinite_weight", "no_current", "no_live", "nan_live", "shape", "mask"])
def test_smoothed_d95_rejects_invalid_population(case):
    x, y, alive, weights = np.zeros(2), np.zeros(2), np.ones(2, bool), np.ones(2)
    if case == "negative_weight": weights[0] = -1
    elif case == "infinite_weight": weights[0] = np.inf
    elif case == "no_current": weights[:] = 0
    elif case == "no_live": alive[:] = False
    elif case == "nan_live": x[0] = np.nan
    elif case == "shape": y = np.zeros(3)
    else: alive = np.ones(2, int)
    with pytest.raises(ValueError):
        solver.smoothed_current_d95_nm(x, y, alive, weights, .05)


@pytest.mark.parametrize("value", [-1., float("nan"), float("inf"), True, None])
def test_objective_rejects_invalid_smoothing(value):
    with pytest.raises(ValueError):
        solver.Objective(d95_target_nm=.65, d95_scale_nm=.2, d95_smoothing_nm=value).validate()


def test_smoothing_disabled_retains_exact_hinge_and_needs_pair_when_enabled():
    solver.Objective(d95_smoothing_nm=0.).validate()
    with pytest.raises(ValueError, match="requires"):
        solver.Objective(d95_smoothing_nm=.05).validate()
    measured = observation()
    cohort = solver.LocalCohort(measured, d95_enabled=True)
    old = cohort.residual(measured, solver.Objective(d95_target_nm=.1, d95_scale_nm=.2))
    new = cohort.residual(measured, solver.Objective(d95_target_nm=.1, d95_scale_nm=.2, d95_smoothing_nm=0.))
    np.testing.assert_array_equal(new, old)


def test_smoothed_trial_records_surrogate_separately_from_native_measurement(tmp_path, fake_snapshot):
    objective = solver.Objective(d95_target_nm=.1, d95_scale_nm=.2, d95_smoothing_nm=.05)
    run = search(tmp_path, fake_snapshot, objective=objective)
    residual = run.residual([0.])
    row = solver.read_json(run.output / "trials/000001.json")
    diag = row["objective_diagnostics"]
    assert row["observation"]["d95_nm"] == .2 == diag["native_d95_nm"]
    assert diag["smoothed_d95_nm"] > .2
    assert diag["hinge_residual"] == residual[-1]
    assert diag["d95_smoothing_nm"] == .05
    assert solver.read_json(run.output / "run.json")["objective"]["d95_smoothing_nm"] == .05


@pytest.mark.parametrize("options", [
    ["--d95-smoothing-nm", ".05"],
    ["--d95-target-nm", ".65", "--d95-smoothing-nm", ".05"],
    ["--d95-target-nm", ".65", "--d95-scale-nm", ".2", "--d95-smoothing-nm", "-1"],
    ["--d95-target-nm", ".65", "--d95-scale-nm", ".2", "--d95-smoothing-nm", "nan"],
])
def test_invalid_cli_smoothing_is_rejected_before_any_preparation(monkeypatch, tmp_path, options):
    monkeypatch.setattr(solver, "make_state", lambda *a, **k: pytest.fail("Unexpected physical preparation"))
    output = tmp_path / "new-run"
    with pytest.raises(SystemExit) as error:
        solver.main(["fit", "--output", str(output), *options])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("translation,weight_scale", [(0., 1.), (1e-6, 3.7)])
def test_native_full_current_d90_uses_original_nonuniform_weights_without_mutation(translation, weight_scale):
    from temsim.physics.beam_statistics import transverse_beam_statistics
    # The inner pair carries .88 current, the middle pair raises that to .94,
    # and the outer pair completes it: native D90=4 nm while native D95=6 nm.
    x = np.array([-1., 1., -2., 2., -3., 3., 100., 200.])*1e-9 + translation
    y = np.full(8, -translation)
    weights = np.array([.44, .44, .03, .03, .03, .03, 4., 0.])*weight_scale
    alive = np.array([True]*6+[False, True])
    before = [v.copy() for v in (x, y, weights, alive)]
    selected = alive & (weights > 0)
    stats = transverse_beam_statistics(x, y, np.zeros(8), np.zeros(8), alive=selected, weights=weights)
    d90 = solver.native_current_d90_nm(x, y, alive, weights, stats)
    assert d90 == pytest.approx(4., abs=1e-10)
    assert 2e9*stats.radius_95_m == pytest.approx(6., abs=1e-10)
    for actual, original in zip((x, y, weights, alive), before):
        np.testing.assert_array_equal(actual, original)
