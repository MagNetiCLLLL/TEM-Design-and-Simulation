"""Analytical and mocked D95 search contracts; no physical transport runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_waist_solver_script", ROOT/"scripts"/"solve_probe_waist.py")
solver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = solver
spec.loader.exec_module(solver)


def observation(*, count=64, radius_nm=.2, angle_mrad=30., value=0.):
    angles = np.arange(count)*2*np.pi/count
    offsets = np.asarray(solver.OFFSETS_NM)
    radius = radius_nm+.0001*offsets**2
    arrays = dict(z_mm=10.+offsets*1e-6,
        x_m=radius[:, None]*np.cos(angles)[None, :]*1e-9,
        y_m=radius[:, None]*np.sin(angles)[None, :]*1e-9,
        tx_rad=np.tile(np.tan(angle_mrad*.001)*np.cos(angles), (7, 1)),
        ty_rad=np.tile(np.tan(angle_mrad*.001)*np.sin(angles), (7, 1)),
        flight_time_s=np.full((7, count), 1e-6), kinetic_energy_ev=np.full((7, count), 300000.),
        alive=np.ones((7, count), dtype=bool), source_id=np.arange(count, dtype=np.int64),
        weights=np.full(count, .25), emission_x_m=np.cos(angles)*1e-9)
    state = SimpleNamespace(value=value)
    state.to_dict = lambda: {"synthetic_value": value}
    report = dict(surface={"z_mm": 10., "d95_nm": 2*radius_nm}, source_current_a=1e-5,
        input_digest=f"snapshot-{value}", implementation={"synthetic": "v1"})
    return SimpleNamespace(arrays=arrays, report=report, state=state)


@pytest.mark.parametrize("tau", [0., .05])
def test_known_ring_d95_and_algorithmic_cost_terms(tau):
    obs = observation()
    before = {key: value.copy() for key, value in obs.arrays.items()}
    q = np.array([.1, -.3, .2])
    objective = solver.Objective(d95_smoothing_nm=tau)
    residual, report = solver.objective_residual(obs, q, objective)
    assert residual.shape == (11,)
    assert report["native_profile"][3]["d90_nm"] == pytest.approx(.4)
    assert report["native_profile"][3]["d95_nm"] == pytest.approx(.4)
    assert report["guiding_d95_nm"][3] == pytest.approx(.4+2*tau*np.log(19))
    assert residual[0] == pytest.approx((.4+2*tau*np.log(19))/.2)
    assert residual[1] == pytest.approx(0., abs=1e-10)
    np.testing.assert_allclose(residual[2:8], 0., atol=1e-12)
    assert report["squared_cost_terms"]["control_regularizer"] == pytest.approx(1e-4*np.mean(q*q))
    assert sum(report["squared_cost_terms"].values()) == pytest.approx(residual@residual)
    assert report["native_screen"]["passed"] is True
    assert report["native_screen"]["qualification"] == "UNQUALIFIED_PROPOSAL"
    for key, value in before.items():
        np.testing.assert_array_equal(obs.arrays[key], value)


def test_default_direct_size_minimization_has_no_cap_or_hidden_target():
    obs = observation(radius_nm=2.)
    plain, report = solver.objective_residual(obs, [0.], solver.Objective(d95_smoothing_nm=0.))
    assert plain[0] == pytest.approx(4./.2)
    assert report["native_screen"]["passed"] is True
    assert report["native_screen"]["maximum_d95_nm"] is None
    assert report["native_screen"]["size_cap_applied"] is False
    assert "d95_target_nm" not in solver.Objective.__dataclass_fields__
    capped, cap_report = solver.objective_residual(obs, [0.],
        solver.Objective(d95_smoothing_nm=0., max_d95_nm=1.))
    np.testing.assert_array_equal(plain, capped)
    assert cap_report["native_screen"]["checks"]["explicit_native_d95_cap"] is False
    assert cap_report["native_screen"]["passed"] is False


def test_explicit_search_cap_is_strict_at_native_diameter():
    obs = observation()
    _, uncapped = solver.objective_residual(obs, [0.], solver.Objective())
    diameter = uncapped["native_profile"][3]["d95_nm"]
    _, capped = solver.objective_residual(obs, [0.], solver.Objective(max_d95_nm=diameter))
    assert capped["native_screen"]["checks"]["explicit_native_d95_cap"] is False


def test_rms_tail_and_legacy_focus_do_not_enter_objective():
    original = observation()
    changed = deepcopy(original)
    # A symmetric 3.125% tail raises RMS while leaving centroid/native D95
    # and all angles unchanged. Native unsmoothed objective must be identical.
    changed.arrays["x_m"][:, [0, 32]] *= 1000
    changed.arrays["y_m"][:, [0, 32]] = 0.
    changed.report.update(local_waist_nm=1e12, covariance_waist_nm=-1e12,
        variance_second_derivative=-1., rms_radius_nm=1e12)
    objective = solver.Objective(d95_smoothing_nm=0.)
    first, a = solver.objective_residual(original, [0.], objective)
    second, b = solver.objective_residual(changed, [0.], objective)
    np.testing.assert_allclose(first, second, atol=1e-10)
    assert b["native_profile"][3]["rms_radius_nm"] > 100*a["native_profile"][3]["rms_radius_nm"]


def test_profile_penalties_use_each_actual_plane_and_no_parabolic_focus():
    obs = observation()
    obs.arrays["x_m"][1] *= .5
    obs.arrays["y_m"][1] *= .5
    residual, report = solver.objective_residual(obs, [0.], solver.Objective(d95_smoothing_nm=0.))
    assert residual[3] == pytest.approx((.4-.2025)/.01)
    assert report["native_screen"]["checks"]["strict_sampled_surface_minimum"] is False
    assert not report["native_screen"]["passed"]


def test_real_clipping_and_new_arrivals_between_trials_are_allowed():
    obs = observation()
    obs.arrays["alive"][:, :8] = False
    residual, first = solver.objective_residual(obs, [0.], solver.Objective())
    assert np.isfinite(residual).all()
    assert first["native_profile"][3]["surviving_rays"] == 56
    assert first["native_profile"][3]["current_a"] == pytest.approx(1e-5*56/64)
    obs.arrays["alive"][:, :8] = True
    _, second = solver.objective_residual(obs, [0.], solver.Objective())
    assert second["native_profile"][3]["surviving_rays"] == 64
    obs.arrays["alive"][4:, :4] = False
    _, clipped = solver.objective_residual(obs, [0.], solver.Objective())
    assert clipped["native_profile"][3]["surviving_rays"] == 64
    assert clipped["native_profile"][4]["surviving_rays"] == 60
    assert clipped["native_screen"]["checks"]["constant_profile_population"] is False


@pytest.mark.parametrize("kind", ["count", "effective", "nonfinite", "revival"])
def test_invalid_numerical_support_is_rejected(kind):
    obs = observation()
    if kind == "count":
        obs.arrays["alive"][:, :50] = False
    elif kind == "effective":
        obs.arrays["weights"][0] = 1000
    elif kind == "nonfinite":
        obs.arrays["x_m"][3, 1] = np.nan
    else:
        obs.arrays["alive"][0, 0] = False
    with pytest.raises(solver.NumericalSupportRejected):
        solver.objective_residual(obs, [0.], solver.Objective())


def test_translation_and_uniform_weight_scale_are_statistics_only():
    obs = observation()
    first, _ = solver.objective_residual(obs, [0., 0.], solver.Objective())
    obs.arrays["x_m"] += 1e-6
    obs.arrays["y_m"] -= 2e-6
    obs.arrays["weights"] *= 3.7
    before = deepcopy(obs.arrays)
    second, _ = solver.objective_residual(obs, [0., 0.], solver.Objective())
    np.testing.assert_allclose(first, second, atol=2e-10)
    for key in before:
        np.testing.assert_array_equal(obs.arrays[key], before[key])


@pytest.mark.parametrize("field", list(solver.Objective.__dataclass_fields__))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1., True])
def test_objective_invalid_options_are_rejected(field, value):
    with pytest.raises(ValueError):
        solver.Objective(**{field: value}).validate()


class SyntheticEvaluator:
    def __init__(self):
        self.calls = []
        self.cancelled = lambda: False
        self.failure = None
        self.mutate_source = False

    def evaluate(self, values):
        self.calls.append(values.copy())
        if self.failure:
            raise self.failure
        value = float(values[0])
        result = observation(value=value, radius_nm=.2-.02*value, angle_mrad=30.+.2*value)
        result.report.update(actual_trace_batches=2, gun_reused=len(self.calls) > 1)
        if self.mutate_source:
            result.arrays["emission_x_m"][0] += 1e-12
        return result


@pytest.fixture
def fake_snapshot(monkeypatch):
    import temsim.instrument_snapshot
    monkeypatch.setattr(temsim.instrument_snapshot, "capture_instrument_snapshot", lambda state:
        SimpleNamespace(digest=f"snapshot-{state.value}",
            to_dict=lambda: {"digest": f"snapshot-{state.value}", "implementation": {"synthetic": "v1"}}))
    monkeypatch.setattr(solver, "seed_controls", lambda state: {"synthetic_control": state.value})


def search(tmp_path, fake_snapshot, **options):
    from solve_joint_probe import Control
    return solver.Search(SyntheticEvaluator(), (Control("test", "test", "value", 0., -2., 2., 1.),),
        tmp_path/"new-run", **options)


def test_fixed_residual_dimension_cached_calls_and_separate_native_pointer(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    first = run.residual(np.array([0.]))
    cached = run.residual(np.array([0.]))
    second = run.residual(np.array([1.]))
    assert first.shape == second.shape == (9,)
    np.testing.assert_array_equal(first, cached)
    assert len(run.evaluator.calls) == run.evaluations == 2
    assert run.document["completed_column_trace_batches"] == 4
    assert run.document["completed_gun_traces"] == 1
    assert run.best["trial"] == 1
    assert run.native_best["trial"] == 2  # Native D95 improves despite larger angle cost.
    assert run.native_best["qualification"] == "UNQUALIFIED_PROPOSAL"
    for pointer in (run.best, run.native_best):
        for name, expected in pointer["files_sha256"].items():
            assert solver.file_hash(run.output/pointer["directory"]/name) == expected
    with np.load(run.output/run.best["directory"]/"particles.npz") as raw:
        assert raw["alive"].shape == (7, 64)
        np.testing.assert_array_equal(raw["weights"], np.full(64, .25))
    with pytest.raises(FileExistsError):
        search(tmp_path, fake_snapshot)


def test_source_identity_drift_is_fatal_and_old_trial_remains(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    run.residual([0.])
    old = (run.output/run.best["directory"]/"particles.npz").read_bytes()
    run.evaluator.mutate_source = True
    with pytest.raises(ValueError, match="Original emission"):
        run.residual([1.])
    row = json.loads((run.output/"trials/trial-000002/trial.json").read_text())
    assert row["status"] == "FAILED"
    assert (run.output/run.best["directory"]/"particles.npz").read_bytes() == old


def test_budget_counts_real_calls_including_numerical_jacobian(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot, max_evaluations=1)
    report = run.run("fit")
    assert report["status"] == "STOPPED"
    assert report["actual_evaluations"] == len(run.evaluator.calls) == 1
    assert "budget" in report["reason"]
    assert run.best is not None


@pytest.mark.parametrize("failure,status", [(RuntimeError("broken executor"), "FAILED"),
    (KeyboardInterrupt(), "STOPPED"), (InterruptedError("cancelled"), "STOPPED")])
def test_failure_and_cancellation_preserve_completed_trials(tmp_path, fake_snapshot, failure, status):
    run = search(tmp_path, fake_snapshot)
    run.evaluator.failure = failure
    report = run.run("measure")
    assert report["status"] == status
    assert report["actual_evaluations"] == 1
    assert run.best is None
    row = json.loads((run.output/"trials/trial-000001/trial.json").read_text())
    assert row["status"] in {"FAILED", "CANCELLED"}
    assert not (run.output/"best.json").exists()


def test_preexisting_cancel_file_prevents_executor_call(tmp_path, fake_snapshot):
    flag = tmp_path/"cancel"
    flag.touch()
    run = search(tmp_path, fake_snapshot, cancel_file=flag)
    report = run.run("measure")
    assert report["status"] == "STOPPED"
    assert report["actual_evaluations"] == 0


def test_zero_transmission_trial_is_persisted_as_fixed_width_rejection(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    run.residual([0.])
    previous = run.best.copy()
    original = run.evaluator.evaluate
    def stopped(values):
        result = original(values)
        result.arrays["alive"][:] = False
        result.report["surface"]["d95_nm"] = None
        return result
    run.evaluator.evaluate = stopped
    residual = run.residual([1.])
    assert residual.shape == (9,) and np.all(residual == 1e6)
    assert run.best == previous
    folder = run.output/"trials/trial-000002"
    row = json.loads((folder/"trial.json").read_text())
    assert row["status"] == "REJECTED_NUMERICAL_SUPPORT"
    with np.load(folder/"particles.npz") as saved:
        assert not saved["alive"].any()
        np.testing.assert_array_equal(saved["source_id"], np.arange(64))
        np.testing.assert_array_equal(saved["weights"], np.full(64, .25))


def test_saved_input_receipt_mismatch_fails_closed(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot)
    original = run.evaluator.evaluate
    def changed(values):
        result = original(values)
        result.report["input_digest"] = "different-input"
        return result
    run.evaluator.evaluate = changed
    report = run.run("measure")
    assert report["status"] == "FAILED"
    assert "complete unchanged instrument inputs" in report["reason"]
    assert run.best is None


@pytest.mark.parametrize("failure,status", [(ValueError("invalid thread count"), "FAILED"),
    (KeyboardInterrupt(), "STOPPED")])
def test_cli_initialization_failure_cannot_leave_running_record(tmp_path, fake_snapshot, monkeypatch, failure, status):
    import measure_probe_waist
    import numba
    from solve_joint_probe import Control
    state = observation().state
    monkeypatch.setattr(solver, "load_warm_candidate", lambda *a, **k: (state, {}))
    monkeypatch.setattr(solver, "control_space", lambda *a, **k: (Control("test", "test", "value", 0., -2., 2., 1.),))
    monkeypatch.setattr(measure_probe_waist, "ProfileEvaluator", lambda *a, **k: SyntheticEvaluator())
    monkeypatch.setattr(numba, "get_num_threads", lambda: 1)
    def configure(count):
        if count != 1:
            raise failure
    monkeypatch.setattr(numba, "set_num_threads", configure)
    output = tmp_path/"new"
    with pytest.raises(SystemExit):
        solver.main(["measure", "--candidate", str(tmp_path/"candidate"), "--output", str(output), "--threads", "2"])
    report = json.loads((output/"run.json").read_text())
    assert report["status"] == status
    assert report["actual_evaluations"] == 0
    assert "reason" in report


def test_minimal_cli_measure_persists_complete_mocked_lifecycle(tmp_path, fake_snapshot, monkeypatch):
    import measure_probe_waist
    from solve_joint_probe import Control
    state = observation().state
    monkeypatch.setattr(solver, "load_warm_candidate", lambda *a, **k: (state, {"preserved_full_snapshot": True}))
    monkeypatch.setattr(solver, "control_space", lambda *a, **k: (Control("test", "test", "value", 0., -2., 2., 1.),))
    evaluator = SyntheticEvaluator()
    monkeypatch.setattr(measure_probe_waist, "ProfileEvaluator", lambda *a, **k: evaluator)
    output = tmp_path/"new"
    code = solver.main(["measure", "--candidate", str(tmp_path/"candidate"), "--output", str(output),
        "--threads", "1", "--max-evaluations", "1"])
    assert code == 0
    assert len(evaluator.calls) == 1
    report = json.loads((output/"run.json").read_text())
    assert report["status"] == "COMPLETE" and report["qualification"] == "UNQUALIFIED_PROPOSAL"
    assert report["completed_column_trace_batches"] == 2
    assert report["warm_start"]["preserved_full_snapshot"] is True
    assert (output/"initial-inputs.json").exists()
    assert (output/"native-proposal-best.json").exists()


def test_bounded_synthetic_fit_uses_d95_objective_and_exact_call_budget(tmp_path, fake_snapshot):
    run = search(tmp_path, fake_snapshot, max_evaluations=30)
    report = run.run("fit")
    assert report["status"] == "COMPLETE"
    assert report["optimizer"]["success"] is True
    assert report["actual_evaluations"] == len(run.evaluator.calls) <= 30
    assert report["objective_dimension"] == 9
    first = json.loads((run.output/"trials/trial-000001/trial.json").read_text())
    assert run.best_cost <= first["objective_diagnostics"]["squared_cost"]


def test_complete_snapshot_warm_start_preserves_hidden_state(tmp_path):
    from solve_joint_probe import make_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state = make_state(49, .2)
    state.history_step_mm = 1.23
    folder = tmp_path/"candidate"
    folder.mkdir()
    solver.atomic_json(folder/"inputs.json", capture_instrument_snapshot(state).to_dict())
    solver.atomic_json(folder/"state.json", state.to_dict())
    exact, metadata = solver.load_warm_candidate(folder)
    assert capture_instrument_snapshot(exact).digest == capture_instrument_snapshot(state).digest
    assert metadata["changes"] == []
    changed, metadata = solver.load_warm_candidate(folder, rays=193, step=.1)
    assert changed.electron_gun.emitter.ray_count == 193
    assert changed.step_mm == .1
    assert changed.history_step_mm == 1.23
    assert changed._tuning_quality == state._tuning_quality
    assert {item["path"].split("/")[-1] for item in metadata["changes"]} == {"ray_count", "step_mm"}


@pytest.mark.parametrize("option,value", [("--d95-smoothing-nm", "nan"), ("--profile-scale-nm", "0"),
    ("--control-regularizer", "-1"), ("--angle-scale-mrad", "inf")])
def test_invalid_cli_fails_before_candidate_or_output(tmp_path, option, value):
    output = tmp_path/"new"
    with pytest.raises(SystemExit) as error:
        solver.main(["fit", "--candidate", str(tmp_path/"missing"), "--output", str(output), option, value])
    assert error.value.code == 2
    assert not output.exists()


def test_help_in_real_subprocess_is_headless():
    result = subprocess.run([sys.executable, str(ROOT/"scripts/solve_probe_waist.py"), "--help"],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    assert "--control-regularizer" in result.stdout
    assert "--position-scale" not in result.stdout
    assert "--focus-scale" not in result.stdout
    assert "--d95-target-nm" not in result.stdout
    assert "--max-d95-nm" in result.stdout
    code = ("import sys; sys.path.insert(0, 'scripts'); import solve_probe_waist; "
            "from PySide6.QtWidgets import QApplication; assert QApplication.instance() is None")
    imported = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert imported.returncode == 0, imported.stderr
