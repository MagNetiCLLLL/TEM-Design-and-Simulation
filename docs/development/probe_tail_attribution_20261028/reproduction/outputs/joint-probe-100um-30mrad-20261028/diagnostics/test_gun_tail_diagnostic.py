"""Bounded orchestration checks: synthetic crossings, no physical propagation."""
from contextlib import contextmanager
from dataclasses import fields, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).with_name("run_gun_tail_diagnostic.py")
SPEC = importlib.util.spec_from_file_location("gun_tail_diagnostic", PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)

from temsim.optics.electron_gun.base import GunTraceResult, GunExitBundle
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.observations import GunPlaneObservations
from temsim.physics.closed_gun_field import ClosedGunField
from temsim.physics.ray_identity import emission_reference
from temsim.instrument_snapshot import capture_instrument_snapshot
from validate_joint_probe import load_candidate, _restore_exact


@pytest.fixture(scope="module")
def original():
    state, provenance = load_candidate(diagnostic.DEFAULT_CANDIDATE)
    return capture_instrument_snapshot(state), provenance


@pytest.fixture
def synthetic(monkeypatch, original):
    snapshot, provenance = original
    state = _restore_exact(snapshot)
    exit_m = state.electron_gun.exit_plane_z_mm * 1e-3
    request = dict(numerics=dict(cells_per_bore=16), rings=[dict(key=key, start_m=start, stop_m=stop)
        for key, start, stop in (("extractor", .001, .002), ("electrostatic_lens", .003, .004),
                                 ("accelerator:0", .005, .006))])
    electric = ClosedGunField(request, [0., .001], [0., exit_m], np.zeros((2, 2)))
    def requested(gun):
        return dict(request, numerics=dict(cells_per_bore=gun._gun_field_cells_per_bore))
    def axes(request):
        n = (request["numerics"]["cells_per_bore"]-16)//4+2
        return np.linspace(0., .001, n), np.array([0., exit_m])
    def field_for(gun):
        if gun._gun_field_cells_per_bore == 16:
            return electric
        row = requested(gun)
        r, z = axes(row)
        return ClosedGunField(row, r, z, np.zeros((len(r), len(z))))
    calls = []
    @contextmanager
    def context(state):
        yield SimpleNamespace(physical_identity="synthetic-field", numerical_identity="synthetic-numerics")
    @contextmanager
    def cpu(requested, cancelled):
        assert requested == 4
        yield SimpleNamespace(to_dict=lambda: dict(numerical_thread_budget=4, scope="mock_no_execution"))
    def trace(gun, planes, *, cancelled):
        assert not cancelled()
        calls.append(gun.trace_step_mm)
        launch = gun.emit()
        count, rows = len(launch.weight), len(planes)
        phase = [np.broadcast_to(getattr(launch, name), (rows, count)).copy()
                 for name in diagnostic.PHASE[:4]]
        clock = np.broadcast_to(planes[:, None] * 1e-11, (rows, count)).copy()
        energy = np.broadcast_to(gun.emitter.emission_energy_ev + launch.energy_offset_ev, (rows, count)).copy()
        observed = GunPlaneObservations(np.asarray(planes), *phase, clock, energy,
            np.ones((rows, count), bool), np.ones((rows, count), bool), np.zeros((rows, count)))
        bundle = GunExitBundle(*[getattr(launch, name) for name in
            ("x_m", "y_m", "tx_rad", "ty_rad", "energy_offset_ev", "weight", "ray_id")],
            np.ones(count, bool), clock[-1])
        result = GunTraceResult(np.asarray(planes), *phase, bundle, np.full(count, np.nan),
            tuple("" for _ in range(count)), 1e-9, 1e-9, 1e-9,
            emission_reference=emission_reference(launch), flight_time_s=clock,
            electrostatic_model_report=dict(scope="synthetic_no_propagation"))
        return result, observed
    monkeypatch.setattr("validate_joint_probe.load_candidate", lambda _: (_restore_exact(snapshot), provenance))
    monkeypatch.setattr("temsim.cpu_resources.numerical_job", cpu)
    monkeypatch.setattr("temsim.vacuum.bind_gun_environment", lambda _: None)
    monkeypatch.setattr("temsim.physics.gun_field_environment.instrument_gun_field_context", context)
    monkeypatch.setattr(FieldEmissionGun, "base_electric_field", property(field_for))
    monkeypatch.setattr("temsim.physics.closed_gun_field.closed_field_request", requested)
    monkeypatch.setattr("temsim.physics.closed_gun_field.mesh_axes", axes)
    monkeypatch.setattr(diagnostic, "memory_snapshot", lambda: dict(available_physical_bytes=32*1024**3,
        total_physical_bytes=64*1024**3, process_memory_bytes=dict(rss=1024)))
    monkeypatch.setattr("temsim.optics.electron_gun.tracing.trace_feg_with_observations", trace)
    return SimpleNamespace(trace=trace, electric=electric, calls=calls, snapshot=snapshot)


def test_complete_mock_retains_raw_source_and_only_step_changes(tmp_path, synthetic):
    result = diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run")
    assert result["status"] == "COMPLETE"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert synthetic.calls == [.2, .1]
    left, right = result["runs"]
    assert left["source_array_fingerprints"] == right["source_array_fingerprints"]
    assert left["actual_launch_fingerprints"] == right["actual_launch_fingerprints"]
    assert left["field_identity"] == right["field_identity"]
    assert left["explicit_input_changes"] == []
    assert len(right["explicit_input_changes"]) == 1
    assert right["explicit_input_changes"][0]["before"] == .2
    assert right["explicit_input_changes"][0]["after"] == .1
    raw = np.load(tmp_path / "run/step-2/particles.npz", allow_pickle=False)
    assert raw["observed_x_m"].shape == (8, 769)
    assert raw["launch_weight"].shape == raw["blocked_key"].shape == (769,)
    assert "observed_kinetic_energy_ev" in raw and "exit_flight_time_s" in raw
    assert all(row["status"] == "MEASURED" for row in right["measurements"])
    assert json.loads((tmp_path / "run/report.json").read_text())["comparison"]["status"] == "MEASURED_NOT_QUALIFIED"


def test_existing_directory_never_overwritten(tmp_path, synthetic):
    (tmp_path / "sentinel").write_text("unchanged")
    with pytest.raises(FileExistsError):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path)
    assert (tmp_path / "sentinel").read_text() == "unchanged"
    assert synthetic.calls == []


@pytest.mark.parametrize("threads", [0, 9, True, 1.5])
def test_invalid_thread_budget_rejected_before_output(tmp_path, threads):
    with pytest.raises(ValueError, match="Thread budget"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run", threads=threads)
    assert not (tmp_path / "run").exists()


def test_cancelled_before_trace_is_persisted(tmp_path, synthetic):
    token = tmp_path / "cancel"
    token.touch()
    with pytest.raises(InterruptedError):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run", cancel_file=token)
    assert json.loads((tmp_path / "run/report.json").read_text())["status"] == "CANCELLED"
    assert synthetic.calls == []


def test_nonfinite_actual_phase_rejects_without_discarding_raw(tmp_path, monkeypatch, synthetic):
    def invalid(*args, **kwargs):
        result, observed = synthetic.trace(*args, **kwargs)
        observed.x_m[-1, 306] = np.nan
        return result, observed
    monkeypatch.setattr("temsim.optics.electron_gun.tracing.trace_feg_with_observations", invalid)
    with pytest.raises(ValueError, match="positive-current"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run")
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert report["status"] == report["runs"][0]["status"] == "FAILED"
    raw = np.load(tmp_path / "run/step-1/particles.npz", allow_pickle=False)
    assert np.isnan(raw["observed_x_m"][-1, 306])


def test_changed_field_grid_fails_before_second_trace(tmp_path, monkeypatch, synthetic):
    original_identity = diagnostic.field_identity
    count = 0
    def changed(*args):
        nonlocal count
        count += 1
        identity = original_identity(*args)
        if count >= 3:
            identity["radial_grid_sha256"] = "changed"
        return identity
    monkeypatch.setattr(diagnostic, "field_identity", changed)
    with pytest.raises(ValueError, match="electric grid"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run")
    assert synthetic.calls == [.2]
    report = json.loads((tmp_path / "run/report.json").read_text())
    assert [r["status"] for r in report["runs"]] == ["COMPLETE", "FAILED"]


def test_changed_raw_source_fails_before_second_trace(tmp_path, monkeypatch, synthetic):
    original_emit = FieldEmissionGun.emit
    def emit(gun, count=None):
        bundle = original_emit(gun, count)
        if gun.trace_step_mm == .1:
            bundle.weight[0] *= 2.
        return bundle
    monkeypatch.setattr(FieldEmissionGun, "emit", emit)
    with pytest.raises(ValueError, match="original spatial/angular/energy source"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run")
    assert synthetic.calls == [.2]


def test_posttrace_physical_input_mutation_fails(tmp_path, monkeypatch, synthetic):
    def changed(gun, *args, **kwargs):
        result = synthetic.trace(gun, *args, **kwargs)
        gun.extractor.voltage_kv += .001
        return result
    monkeypatch.setattr("temsim.optics.electron_gun.tracing.trace_feg_with_observations", changed)
    with pytest.raises(ValueError, match="input graph"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run")
    assert json.loads((tmp_path / "run/report.json").read_text())["status"] == "FAILED"


def test_empty_and_invalid_energy_cannot_be_measured(synthetic):
    state = _restore_exact(synthetic.snapshot)
    planes, labels = diagnostic.exact_planes(synthetic.electric, state.electron_gun.exit_plane_z_mm)
    result, observed = synthetic.trace(state.electron_gun, planes, cancelled=lambda: False)
    observed.alive[-1] = False
    records = diagnostic.metrics(observed, result.exit_bundle.weight, result.exit_bundle.ray_id, 1e-9, labels)
    assert records[-1]["status"] == "NOT_EVALUATED"
    assert "d95_nm" not in records[-1]
    observed.kinetic_energy_ev[0, 306] = -1.
    with pytest.raises(ValueError, match="positive-current"):
        diagnostic.metrics(observed, result.exit_bundle.weight, result.exit_bundle.ray_id, 1e-9, labels)


def test_grid_mode_keeps_fixed_step_and_source_with_explicit_native_grids(tmp_path, synthetic):
    result = diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run", mode="grid")
    assert synthetic.calls == [.1, .1, .1]
    assert [row["cells_per_bore"] for row in result["runs"]] == [16, 20, 24]
    assert result["comparison"]["same_source"]
    assert result["comparison"]["same_physical_field_boundaries"]
    assert not result["comparison"]["same_field_grid_and_voltages"]
    assert len(result["comparison"]["transitions"]) == 2
    assert len({row["field_identity"]["physical_request_sha256"] for row in result["runs"]}) == 1
    assert len({row["field_identity"]["radial_grid_sha256"] for row in result["runs"]}) == 3
    assert all((tmp_path / f"run/grid-{n}/particles.npz").exists() for n in (16, 20, 24))


def test_grid_memory_headroom_fails_before_field_or_trace(tmp_path, monkeypatch, synthetic):
    monkeypatch.setattr(diagnostic, "memory_snapshot", lambda: dict(available_physical_bytes=1024))
    with pytest.raises(MemoryError, match="headroom"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run", mode="grid")
    assert synthetic.calls == []
    assert json.loads((tmp_path / "run/report.json").read_text())["status"] == "FAILED"


def test_grid_physical_boundary_change_is_rejected(tmp_path, monkeypatch, synthetic):
    original_identity = diagnostic.field_identity
    count = 0
    def changed(*args):
        nonlocal count
        count += 1
        identity = original_identity(*args)
        if count >= 3:
            identity["physical_request_sha256"] = "changed"
        return identity
    monkeypatch.setattr(diagnostic, "field_identity", changed)
    with pytest.raises(ValueError, match="physical field boundary"):
        diagnostic.run(diagnostic.DEFAULT_CANDIDATE, tmp_path / "run", mode="grid")
    assert synthetic.calls == [.1]
