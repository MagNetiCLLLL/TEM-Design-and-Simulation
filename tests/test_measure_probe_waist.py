"""Native D95 waist evidence on synthetic arrays; no physical propagation."""
from copy import deepcopy
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = spec_from_file_location("probe_waist", Path(__file__).resolve().parents[1]/"scripts"/"measure_probe_waist.py")
waist = module_from_spec(spec)
spec.loader.exec_module(waist)


def synthetic(offsets=np.arange(-5., 6.), *, shift=0., halo=True):
    offsets = np.asarray(offsets)
    z = 100.+offsets*1e-6
    core = np.sqrt(.2**2+(.03*(offsets-shift))**2)*1e-9
    radius = np.tile(core[:, None], (1, 100))
    if halo:
        radius[:, 96:] = (5.+.1*offsets[:, None])*1e-9
    sign = np.tile([1., -1.], 50)
    x = radius*sign
    arrays = dict(z_mm=z, x_m=x, y_m=np.zeros_like(x), tx_rad=np.tile(.03*sign, (len(z), 1)),
        ty_rad=np.zeros_like(x), flight_time_s=np.ones_like(x), kinetic_energy_ev=np.full_like(x, 300000.),
        alive=np.ones(x.shape, bool), weights=np.ones(100), source_id=np.arange(100),
        blocked_z_mm=np.full(100, np.nan), blocked_key=np.full(100, ""), terminal_alive=np.ones(100, bool))
    return arrays, z


def analyze(arrays, planes, **kwargs):
    return waist.analyze_scan(arrays, planes, surface_z_mm=100., source_current_a=1e-9, **kwargs)


def test_d95_waist_acceptance_does_not_require_rms_minimum_at_surface():
    arrays, planes = synthetic()
    before = deepcopy(arrays)
    result = analyze(arrays, planes)
    assert result["surface_gate"]["status"] == "BOUNDED_D95_WAIST_AT_SURFACE"
    assert result["surface"]["d95_nm"] == pytest.approx(.4)
    assert result["d95_waist"]["observed_minimum_planes_mm"] == [100.]
    assert result["rms_waist_diagnostic"]["observed_minimum_planes_mm"] == [planes[0]]
    assert result["rms_waist_diagnostic"]["status"] == "UNRESOLVED_BOUNDARY_MINIMUM"
    assert result["surface"]["current_a"] == 1e-9
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


def test_default_has_no_size_cap_and_keeps_native_surface_waist_requirement():
    arrays, planes = synthetic(halo=False)
    arrays["x_m"] *= 10
    result = analyze(arrays, planes)
    assert result["surface"]["d95_nm"] == pytest.approx(4.)
    assert result["surface_gate"]["passed"]
    assert result["surface_gate"]["maximum_d95_nm"] is None
    assert result["surface_gate"]["strict_surface_d95_passed"] is None
    assert result["surface_gate"]["size_cap_applied"] is False


def test_explicit_size_cap_is_strict_and_does_not_change_measurements():
    arrays, planes = synthetic(halo=False)
    before = deepcopy(arrays)
    original = analyze(arrays, planes)
    diameter = original["surface"]["d95_nm"]
    capped = analyze(arrays, planes, max_d95_nm=diameter)
    assert capped["surface_gate"]["size_cap_applied"] is True
    assert capped["surface_gate"]["strict_surface_d95_passed"] is False
    assert capped["surface_gate"]["passed"] is False
    assert capped["surface"] == original["surface"]
    assert analyze(arrays, planes, max_d95_nm=np.nextafter(diameter, np.inf))["surface_gate"]["passed"]
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], before[key])


@pytest.mark.parametrize("limit", [0., -1., np.nan, np.inf, True, "1"])
def test_optional_size_cap_rejects_invalid_values(limit):
    arrays, planes = synthetic()
    with pytest.raises(ValueError, match="D95"):
        analyze(arrays, planes, max_d95_nm=limit)


@pytest.mark.parametrize("shift,status", [(5., "UNRESOLVED_BOUNDARY_MINIMUM"), (3., "BOUNDED_OBSERVED_MINIMUM")])
def test_boundary_and_displaced_minima_cannot_pass(shift, status):
    arrays, planes = synthetic(shift=shift, halo=False)
    result = analyze(arrays, planes)
    assert result["d95_waist"]["status"] == status
    assert not result["surface_gate"]["passed"]
    assert result["surface_gate"]["status"] == ("UNRESOLVED" if shift == 5. else "FAIL")


def test_coarse_bracket_crossing_tolerance_requires_actual_refinement():
    arrays, planes = synthetic(np.arange(-4., 5., 2.))
    result = analyze(arrays, planes)
    assert result["d95_waist"]["observed_minimum_planes_mm"] == [100.]
    assert not result["surface_gate"]["placement_passed"]
    assert result["surface_gate"]["status"] == "UNRESOLVED"


def test_plateau_and_disjoint_equal_minima_are_reported_without_interpolation():
    z = np.arange(5.)
    plateau = waist.observed_minimum(z, [3., 1., 1., 1., 3.], 2.)
    assert plateau["plateau_intervals_mm"] == [[1., 3.]]
    assert plateau["neighbour_bracket_mm"] == [0., 4.]
    assert plateau["observed_minimum_planes_mm"] == [1., 2., 3.]
    separated = waist.observed_minimum(z, [3., 1., 2., 1., 3.], 2.)
    assert separated["status"] == "UNRESOLVED_MULTIPLE_MINIMA"


def test_actual_population_changes_remain_visible_and_cannot_qualify():
    arrays, planes = synthetic()
    arrays["alive"][planes > 100., -2:] = False
    result = analyze(arrays, planes)
    assert result["planes"][0]["surviving_rays"] == 100
    assert result["planes"][-1]["surviving_rays"] == 98
    assert result["planes"][-1]["current_a"] == pytest.approx(.98e-9)
    assert result["planes"][0]["source_ids"] == list(range(100))
    assert result["surface_gate"]["status"] == "UNRESOLVED"
    assert not result["positive_current_membership_stable"]


@pytest.mark.parametrize("field,value", [(key, np.nan) for key in waist.PHASE]
    + [("kinetic_energy_ev", 0.), ("flight_time_s", -.1)])
def test_invalid_active_positive_particle_is_never_silently_filtered(field, value):
    arrays, planes = synthetic()
    arrays[field][5, 0] = value
    with pytest.raises(ValueError, match="positive-current|energy"):
        analyze(arrays, planes)


def test_zero_current_and_physically_stopped_placeholders_remain_raw_but_not_current():
    arrays, planes = synthetic()
    arrays["weights"][0] = 0.
    arrays["alive"][:, 1] = False
    for key in waist.PHASE:
        arrays[key][:, :2] = np.nan
    result = analyze(arrays, planes)
    assert all(row["surviving_rays"] == 98 for row in result["planes"])
    assert result["surface"]["current_a"] == pytest.approx(98/99*1e-9)
    assert all(0 not in row["source_ids"] and 1 not in row["source_ids"] for row in result["planes"])


def test_original_nonuniform_weights_use_native_left_quantile():
    from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile
    arrays, planes = synthetic()
    arrays["weights"] = np.arange(1., 101.)
    result = analyze(arrays, planes)
    i = 5
    native = transverse_beam_statistics(*(arrays[key][i] for key in waist.PHASE[:4]),
        alive=arrays["alive"][i], weights=arrays["weights"])
    r = np.hypot(arrays["x_m"][i]-native.mean_x_m, arrays["y_m"][i]-native.mean_y_m)
    w = arrays["weights"]/arrays["weights"].sum()
    assert result["surface"]["d95_nm"] == 2e9*native.radius_95_m
    assert result["surface"]["d50_nm"] == 2e9*_weighted_quantile(r, w, .5)
    assert result["surface"]["d90_nm"] == 2e9*_weighted_quantile(r, w, .9)


def test_reappearing_particle_and_nearest_instead_of_exact_plane_are_rejected():
    arrays, planes = synthetic()
    arrays["alive"][0, 0] = False
    with pytest.raises(ValueError, match="reappeared"):
        analyze(arrays, planes)
    arrays["alive"][0, 0] = True
    near = planes.copy()
    near[5] = np.nextafter(100., np.inf)
    with pytest.raises(ValueError, match="exact"):
        analyze(arrays, near)


def state():
    return SimpleNamespace(sample=SimpleNamespace(upper_surface_z_mm=100., z_mm=100.+2.5e-6,
        lower_surface_z_mm=100.+5e-6), electron_gun=SimpleNamespace(exit_plane_z_mm=0.),
        optical_control=13.4, step_mm=.05)


@pytest.fixture
def bounded_state(monkeypatch):
    monkeypatch.setattr("temsim.physics.particle_sections.section_limits", lambda _: (0., 200.))
    return state()


def test_default_planes_include_all_exact_physical_references(bounded_state):
    planes = waist.capture_planes(bounded_state)
    assert planes[0] == 100.-100e-6 and planes[-1] == 100.+100e-6
    for value in (100., bounded_state.sample.z_mm, bounded_state.sample.lower_surface_z_mm):
        assert np.count_nonzero(planes == value) == 1
    np.testing.assert_array_equal(waist.capture_planes(bounded_state, extra_z_offset_nm=()), planes)
    near = waist.capture_planes(bounded_state, extra_z_offset_nm=(-.1, .1, -.1))
    assert np.count_nonzero(near == 100.-.1e-6) == 1
    assert np.count_nonzero(near == 100.+.1e-6) == 1


@pytest.mark.parametrize("kwargs", [dict(window_nm=0), dict(spacing_nm=0), dict(window_nm=True),
    dict(window_nm=np.inf), dict(spacing_nm=.0001), dict(extra_z_offset_nm=[np.nan]),
    dict(extra_z_offset_nm=[True]), dict(extra_z_offset_nm=[101.]), dict(extra_z_offset_nm=None),
    dict(window_nm=1., spacing_nm=.1)])
def test_invalid_or_excessive_scan_is_rejected(bounded_state, kwargs):
    with pytest.raises(ValueError):
        waist.capture_planes(bounded_state, **kwargs)


class Snapshot:
    def __init__(self, value):
        self.value = deepcopy(value)
        self.digest = sha256(json.dumps(vars(value), default=lambda x: vars(x), sort_keys=True).encode()).hexdigest()

    def restore(self):
        return deepcopy(self.value)

    def to_dict(self):
        return dict(digest=self.digest, graph=json.loads(json.dumps(vars(self.value), default=lambda x: vars(x))),
            implementation={"source": "synthetic"}, external_inputs=[])


@pytest.fixture
def execution(bounded_state, monkeypatch):
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot", Snapshot)
    calls = []
    def execute(value, planes, cancelled, **kwargs):
        calls.append((value, planes.copy(), kwargs))
        arrays, _ = synthetic((planes-100.)*1e6)
        arrays["z_mm"] = planes.copy()
        return arrays, dict(status="COMPLETE", input_digest=Snapshot(value).digest,
            implementation={"source": "synthetic"}, source_current_a=1e-9,
            batches=[dict(field_audit=dict(status="PASS", passed=True))])
    return bounded_state, calls, execute


def test_success_preserves_state_raw_data_full_snapshot_and_provenance(tmp_path, execution):
    value, calls, execute = execution
    before = Snapshot(value).digest
    report = waist.measure(value, tmp_path/"scan", executor=execute, threads=1)
    assert report["execution_status"] == "COMPLETE", report["errors"]
    assert len(calls) == 1 and calls[0][0] is not value
    assert Snapshot(value).digest == before and report["original_state_unchanged"]
    assert not report["passed"] and report["qualification"] == "NOT_QUALIFIED"
    assert report["numerical_convergence"] == "NOT_EVALUATED"
    assert report["parameters"]["max_d95_nm"] is None
    checksums = json.loads((tmp_path/"scan"/"checksums.json").read_text())
    for filename, expected in checksums.items():
        assert waist._hash(tmp_path/"scan"/filename) == expected
    with np.load(tmp_path/"scan"/"particles.npz", allow_pickle=False) as saved:
        assert saved["weights"].shape == (100,)
        assert saved["alive"].shape[1] == 100
        assert np.isnan(saved["blocked_z_mm"]).all()
    assert json.loads((tmp_path/"scan"/"inputs.json").read_text())["digest"] == before


def test_explicit_cap_is_recorded_by_measurement_without_changing_raw_output(tmp_path, execution):
    value, calls, execute = execution
    report = waist.measure(value, tmp_path/"scan", executor=execute, threads=1, max_d95_nm=.1)
    assert report["execution_status"] == "COMPLETE"
    assert report["parameters"]["max_d95_nm"] == .1
    assert report["analysis"]["surface_gate"]["maximum_d95_nm"] == .1
    assert report["analysis"]["surface_gate"]["strict_surface_d95_passed"] is False
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["receipt", "mutation", "missing_audit", "nonfinite", "cancel"])
def test_failure_evidence_is_saved_without_claiming_qualification(tmp_path, execution, mode):
    value, calls, execute = execution
    def damaged(value, planes, cancelled, **kwargs):
        if mode == "cancel":
            raise KeyboardInterrupt()
        arrays, receipt = execute(value, planes, cancelled, **kwargs)
        if mode == "receipt":
            receipt["input_digest"] = "wrong"
        elif mode == "mutation":
            value.optical_control += 1.
        elif mode == "missing_audit":
            receipt["batches"] = []
        else:
            arrays["x_m"][5, 0] = np.nan
        return arrays, receipt
    report = waist.measure(value, tmp_path/mode, executor=damaged, threads=1)
    assert report["execution_status"] == ("CANCELLED" if mode == "cancel" else "FAILED")
    assert not report["passed"] and report["errors"]
    assert report["original_state_unchanged"] and value.optical_control == 13.4
    assert json.loads((tmp_path/mode/"report.json").read_text())["execution_status"] == report["execution_status"]
    if mode == "nonfinite":
        assert (tmp_path/mode/"particles.npz").exists()


def test_new_output_only_and_precancelled_job_never_executes(tmp_path, execution):
    value, calls, execute = execution
    with pytest.raises(FileExistsError):
        waist.measure(value, tmp_path, executor=execute)
    cancel = tmp_path/"cancel"
    cancel.touch()
    report = waist.measure(value, tmp_path/"stopped", executor=execute, cancel_file=cancel)
    assert report["execution_status"] == "CANCELLED" and not calls


def test_cli_requires_complete_snapshot_before_execution(tmp_path, monkeypatch):
    monkeypatch.setattr("validate_joint_probe.load_candidate", lambda _: (state(), dict(restoration_kind="standalone_serialized_state")))
    with pytest.raises(SystemExit) as exc:
        waist.main(["--candidate", "state.json", "--output", str(tmp_path/"no-run")])
    assert exc.value.code == 2 and not (tmp_path/"no-run").exists()


@pytest.fixture
def production_adapter(monkeypatch):
    import temsim.physics.particle_sections as sections
    value = state()
    value.electron_gun.source_representation = "classical_particles"
    value.electron_gun.nominal_exit_energy_ev = 300000.
    monkeypatch.setattr("temsim.simulation_modes.is_unified", lambda _: True)
    monkeypatch.setattr("temsim.specimen.source.specimen_is_vacuum", lambda _: True)
    monkeypatch.setattr("temsim.optics.electron_gun.source_policy.require_physical_gun_source", lambda _: None)
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight", lambda _: {"issues": []})
    class StaticSnapshot:
        digest = "original"
        def to_dict(self):
            return {"implementation": {"solver": "synthetic-adapter"}}
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot", lambda _: StaticSnapshot())
    monkeypatch.setattr(sections, "gun_dependency_signature", lambda _: "source-signature")
    monkeypatch.setattr(sections, "_events", lambda _: ())
    n = 7
    original = np.arange(n, dtype=float)
    e = SimpleNamespace(x_m=original*1e-9, y_m=-original*1e-9, tx_rad=original*.001,
        ty_rad=-original*.001, flight_time_s=original*1e-12, energy_offset_ev=original*.01,
        weight=np.array([.1, .2, .3, .4, .5, .6, .7]), alive=np.array([True]*6+[False]))
    gun = SimpleNamespace(exit_bundle=e, blocked_z_mm=np.array([np.nan]*6+[0.]),
        blocked_key=[""]*6+["gun-stop"], emission_reference={"ray_id": np.arange(10, 17)})
    for key in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s"):
        setattr(gun, key, np.vstack([getattr(e, key)]*2))
    gun.z_mm = np.array([-.1, 0.])
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _: gun)
    monkeypatch.setattr("temsim.physics.simulation._validate_gun_flight_times", lambda _: True)
    monkeypatch.setattr("temsim.physics.ray_identity.emitted_source_identity", lambda _: (np.arange(10, 17), original*.2))
    monkeypatch.setattr("temsim.physics.beam_current.effective_source_current_a", lambda _: 2e-9)
    monkeypatch.setattr(waist, "audit_extended_plan", lambda *a: {"passed": True, "status": "PASS"})
    calls = []
    def trace(state, name, start, stop, initial, offsets, weights, identity, survival,
              events, previous, boundary, **kwargs):
        calls.append(dict(start=start, stop=stop, weights=weights.copy(), ids=identity[0].copy(),
            initial=tuple(a.copy() for a in initial), offsets=offsets.copy(), kwargs=kwargs))
        z = np.unique(np.r_[start, kwargs["save_z_mm"], stop])
        cp = SimpleNamespace(z_mm=z,
            unified_state=SimpleNamespace(active=np.tile(survival[0], (len(z), 1))))
        for key, a in zip(("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"), initial):
            setattr(cp, key, np.tile(a, (len(z), 1)))
        cp.kinetic_energy_ev = np.tile(kwargs["initial_kinetic_energy_ev"], (len(z), 1))
        branch = SimpleNamespace(source_ray_id=identity[0].copy(), ray_weight=weights.copy(),
            blocked_z=np.asarray(survival[1]).copy(), blocked_key=survival[2], alive=survival[0].copy())
        plan = SimpleNamespace(z_mm=z.copy(), signature="same-full-source-mesh")
        return SimpleNamespace(checkpoints=cp, branch=branch, plan=plan), start, False
    monkeypatch.setattr(sections, "_trace_segment", trace)
    return value, e, gun, calls, trace


def test_production_adapter_preserves_complete_source_batches_and_exact_mesh(production_adapter):
    value, e, gun, calls, _ = production_adapter
    planes = np.array([99.9999, 100., 100.0001])
    arrays, receipt = waist.execute_scan(value, planes, lambda: False, batch_size=3)
    assert len(calls) == 3
    assert all(call["kwargs"]["checkpoint_ray_count"] == 7 for call in calls)
    assert all(call["stop"] == planes[-1] and call["kwargs"]["save_z_mm"] == tuple(planes) for call in calls)
    np.testing.assert_array_equal(np.concatenate([call["weights"] for call in calls]), e.weight)
    np.testing.assert_array_equal(arrays["weights"], e.weight)
    np.testing.assert_array_equal(arrays["source_id"], np.arange(10, 17))
    np.testing.assert_array_equal(arrays["gun_exit_x_m"], e.x_m)
    np.testing.assert_array_equal(arrays["x_m"][1], e.x_m)
    np.testing.assert_array_equal(arrays["kinetic_energy_ev"][1], 300000.+e.energy_offset_ev)
    assert not arrays["alive"][:, -1].any() and arrays["blocked_key"][-1] == "gun-stop"
    assert receipt["source_points"] == 7 and receipt["source_current_a"] == 2e-9
    assert receipt["integration_mesh_sha256"] == waist._array_hash(arrays["integration_z_mm"])


@pytest.mark.parametrize("damage", ["weights", "ids", "mesh", "missing_plane"])
def test_production_adapter_rejects_changed_source_or_inexact_batch_mesh(production_adapter, monkeypatch, damage):
    import temsim.physics.particle_sections as sections
    value, e, gun, calls, trace = production_adapter
    def broken(*args, **kwargs):
        segment, resume, reused = trace(*args, **kwargs)
        if damage == "weights":
            segment.branch.ray_weight[0] *= .5
        elif damage == "ids":
            segment.branch.source_ray_id[0] += 100
        elif damage == "mesh" and len(calls) == 2:
            segment.plan.z_mm[1] = np.nextafter(segment.plan.z_mm[1], np.inf)
        elif damage == "missing_plane":
            segment.checkpoints.z_mm[1] = np.nextafter(segment.checkpoints.z_mm[1], np.inf)
        return segment, resume, reused
    monkeypatch.setattr(sections, "_trace_segment", broken)
    with pytest.raises(ValueError, match="identity|meshes|exact"):
        waist.execute_scan(value, np.array([99.9999, 100., 100.0001]), lambda: False, batch_size=3)


def test_production_adapter_does_not_disable_coherent_or_material_inputs(production_adapter, monkeypatch):
    value, _, _, calls, _ = production_adapter
    value.sample.wave_enabled = True
    with pytest.raises(ValueError, match="Coherent"):
        waist.execute_scan(value, np.array([99.9999, 100., 100.0001]), lambda: False)
    assert value.sample.wave_enabled and not calls
    value.sample.wave_enabled = False
    monkeypatch.setattr("temsim.specimen.source.specimen_is_vacuum", lambda _: False)
    with pytest.raises(ValueError, match="already-vacuum"):
        waist.execute_scan(value, np.array([99.9999, 100., 100.0001]), lambda: False)
    assert not calls


def test_source_and_prefix_cache_uses_production_signature_and_prior_segments(production_adapter, monkeypatch):
    import temsim.physics.particle_sections as sections
    value, _, gun, calls, trace = production_adapter
    source_calls, previous_segments = [], []
    monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", lambda _: source_calls.append(1) or gun)
    signature = ["a"]
    monkeypatch.setattr(sections, "gun_dependency_signature", lambda _: signature[0])
    def resumable(*args, **kwargs):
        previous = args[10]
        previous_segments.append(previous)
        segment, _, _ = trace(*args, **kwargs)
        return segment, 100. if previous is not None else 0., previous is not None
    monkeypatch.setattr(sections, "_trace_segment", resumable)
    cache, planes = {}, np.array([99.9999, 100., 100.0001])
    first_arrays, first = waist.execute_scan(value, planes, lambda: False, batch_size=3, _cache=cache)
    second_arrays, second = waist.execute_scan(value, planes, lambda: False, batch_size=3, _cache=cache)
    assert source_calls == [1]
    assert not first["gun_reused"] and not first["column_prefix_reused"]
    assert second["gun_reused"] and second["column_prefix_reused"]
    assert all(item is None for item in previous_segments[:3])
    assert all(item is not None for item in previous_segments[3:])
    assert first["emitted_identity_digest"] == second["emitted_identity_digest"]
    for key in first_arrays:
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
    signature[0] = "changed-physical-source-dependency"
    _, third = waist.execute_scan(value, planes, lambda: False, batch_size=3, _cache=cache)
    assert len(source_calls) == 2 and not third["gun_reused"] and not third["column_prefix_reused"]


def test_profile_evaluator_executes_one_actual_profile_and_preserves_seed(execution, monkeypatch):
    import solve_joint_probe as solver
    value, _, _ = execution
    controls = (SimpleNamespace(name="objective", lower=0., upper=100.),)
    monkeypatch.setattr(solver, "control_space", lambda *a: controls)
    monkeypatch.setattr(solver, "set_values", lambda state, controls, values: setattr(state, "optical_control", float(values[0])))
    monkeypatch.setattr(solver, "prepare_objective_observations", lambda _: {"changed_derived_planes": []})
    calls = []
    def execute(state, planes, cancelled, **kwargs):
        calls.append((state.optical_control, planes.copy(), kwargs["_cache"]))
        arrays, _ = synthetic((planes-100.)*1e6)
        arrays["z_mm"] = planes.copy()
        return arrays, dict(status="COMPLETE", input_digest=Snapshot(state).digest,
            implementation={"source": "synthetic"}, source_current_a=1e-9,
            emitted_identity_digest="same-original-emission", batches=[])
    monkeypatch.setattr(waist, "execute_scan", execute)
    evaluator = waist.ProfileEvaluator(value, controls)
    a = evaluator.evaluate([12.])
    b = evaluator.evaluate([14.])
    assert len(calls) == 2 and calls[0][0] == 12. and calls[1][0] == 14.
    assert calls[0][2] is calls[1][2]
    assert len(a.report["profile"]) == 7 and a.report["surface"]["z_mm"] == 100.
    assert a.report["qualification"] == "PROPOSAL_ONLY"
    assert value.optical_control == 13.4 and a.state.optical_control == 12. and b.state.optical_control == 14.
    np.testing.assert_array_equal(calls[0][1], 100.+np.array([-20., -5., -1., 0., 1., 5., 20.])*1e-6)


def test_extended_audit_checks_the_actual_extended_plan_not_surface_proxy(monkeypatch):
    import audit_joint_probe_fields as audit
    import temsim.physics.unified_field_runtime as runtimes
    value = state()
    monkeypatch.setattr("temsim.instrument_snapshot.capture_instrument_snapshot", lambda _: SimpleNamespace(digest="same"))
    called = []
    class Runtime:
        source_identity = "same"
        digest = "runtime"
        snapshot = SimpleNamespace(numerical_digest="field")
        geometry = SimpleNamespace(digest="geometry")
        def require_state(self, state):
            assert state is value
    monkeypatch.setattr(runtimes, "UnifiedFieldRuntime", Runtime)
    runtime = Runtime()
    plan = SimpleNamespace(unified_runtime=runtime, z_mm=np.array([0., 100., 100.0001]),
        checkpoint_index=np.array([0, 1, 2]), signature="extended-plan")
    monkeypatch.setattr(audit, "_inputs", lambda state: ("physical-aperture", 0., 100.))
    monkeypatch.setattr(audit, "_plan_actions", lambda p, r, start, stop:
        called.append((p, r, start, stop)) or {"interval_mm": [start, stop]})
    monkeypatch.setattr(audit, "_field_ownership", lambda s, r: {"actual_runtime": r.digest})
    monkeypatch.setattr(audit, "_c2_geometry", lambda a, r, start, stop: {"actual_boundary_radius_mm": .05})
    result = waist.audit_extended_plan(value, plan, 100.0001)
    assert result["passed"] and called == [(plan, runtime, 0., 100.0001)]
    assert result["identities"]["plan_signature"] == "extended-plan"
    plan.checkpoint_index = np.array([0, 2])
    with pytest.raises(ValueError, match="exact surface"):
        waist.audit_extended_plan(value, plan, 100.0001)
    plan.checkpoint_index = np.array([0, 1, 2])
    runtime.source_identity = "different-physical-state"
    with pytest.raises(ValueError, match="original inputs"):
        waist.audit_extended_plan(value, plan, 100.0001)


@pytest.mark.parametrize("damage", ["all_stopped", "late_stopped", "live_nan", "live_bad_energy", "duplicate_id"])
def test_profile_zero_current_trial_keeps_raw_evidence_without_hiding_invalid_inputs(execution, monkeypatch, damage):
    import solve_joint_probe as solver
    value, _, _ = execution
    controls = (SimpleNamespace(name="objective", lower=0., upper=100.),)
    monkeypatch.setattr(solver, "control_space", lambda *a: controls)
    monkeypatch.setattr(solver, "set_values", lambda *a: None)
    monkeypatch.setattr(solver, "prepare_objective_observations", lambda _: {})
    raw = {}
    def execute(state, planes, cancelled, **kwargs):
        arrays, _ = synthetic((planes-100.)*1e6)
        arrays["z_mm"] = planes.copy()
        arrays["alive"][4:] = False
        if damage == "all_stopped":
            arrays["alive"][:] = False
        elif damage == "live_nan":
            arrays["x_m"][0, 0] = np.nan
        elif damage == "live_bad_energy":
            arrays["kinetic_energy_ev"][0, 0] = -1.
        elif damage == "duplicate_id":
            arrays["source_id"][0] = arrays["source_id"][1]
        raw.update(arrays)
        return arrays, dict(status="COMPLETE", input_digest=Snapshot(state).digest,
            implementation={"source": "synthetic"}, source_current_a=1e-9, batches=[])
    monkeypatch.setattr(waist, "execute_scan", execute)
    evaluator = waist.ProfileEvaluator(value, controls)
    if damage in {"live_nan", "live_bad_energy", "duplicate_id"}:
        with pytest.raises(ValueError):
            evaluator.evaluate([12.])
    else:
        result = evaluator.evaluate([12.])
        assert result.report["status"] == "NO_NUMERICAL_SUPPORT"
        assert result.report["surface"] == {"z_mm": 100., "d95_nm": None}
        assert result.report["qualification"] == "PROPOSAL_ONLY"
        assert result.arrays["alive"] is raw["alive"]
        assert result.report["input_digest"] == Snapshot(value).digest
