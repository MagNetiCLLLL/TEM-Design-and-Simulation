"""Mocked CLI orchestration and real safe-array export; no physical propagation."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim import mixed_wave_cli as cli


@pytest.fixture
def pipeline(monkeypatch):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics import compute_backend, mixed_wave_particles, mixed_wave_pipeline
    import temsim.instrument_snapshot as snapshots
    import temsim.particle_section_io as codec
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.wave_checkpoint import EMITTED_ELECTRON_REFERENCE, WaveCheckpoint
    from temsim.physics.wave_flux import BeamState, WaveMode

    calls = []
    state = SimpleNamespace(simulation_mode="unified", step_mm=.025,
        sample=SimpleNamespace(z_mm=1605., thickness_nm=10.),
        electron_gun=SimpleNamespace(emitter=SimpleNamespace(ray_count=16384)),
        lenses=[SimpleNamespace(strength=42.)])
    original_ids = np.arange(1024)
    gun = SimpleNamespace(exit_bundle=SimpleNamespace(ray_id=original_ids,
        weight=np.full(1024, 1 / 1024), alive=np.arange(1024) < 3),
        blocked_z_mm=np.full(1024, np.nan), blocked_key=[""] * 1024)
    particles = SimpleNamespace(state_snapshot=state, simulation=SimpleNamespace(gun_trace=gun))
    source = SimpleNamespace(ray_id=np.array([3, 7, 11]), weight=np.array([.1, .05, .05]),
        x_m=np.array([-1., 0., 1.]) * 1e-10, y_m=np.zeros(3), tx_rad=np.zeros(3), ty_rad=np.zeros(3),
        kinetic_energy_ev=np.full(3, 300000.), flight_time_s=np.full(3, 1e-8),
        digest="a" * 64, input_digest="b" * 64, z_mm=1605. - 5e-6,
        reference_current_a=1e-8, original_positive_count=1024, source_fraction=.2,
        provenance={"scope": "Mocked orchestration fixture, not a physical calculation"})

    class Inputs:
        digest = "b" * 64

        def to_dict(self):
            return {"fixture": "no physical propagation", "step_mm": state.step_mm,
                "ray_count": state.electron_gun.emitter.ray_count, "lens_strength": state.lenses[0].strength}

    monkeypatch.setattr(snapshots, "capture_instrument_snapshot", lambda *_: Inputs())
    monkeypatch.setattr(compute_backend, "require_coherent_cuda", lambda *_a, **_k: calls.append("cuda") or "Require GPU")
    monkeypatch.setattr(codec, "load_section_result", lambda path: calls.append(("load", path)) or particles)
    monkeypatch.setattr(mixed_wave_particles, "particle_source_from_result",
        lambda result, **kwargs: calls.append(("source", result)) or source)

    class Session:
        def __init__(self, captured, request, supplier):
            calls.append(("session", captured, request))
            self.request, self.supplier = request, supplier

        def observe(self, z_mm, **kwargs):
            calls.append(("observe", z_mm))
            assert self.supplier() is source
            plane = PlaneWave(np.ones((4, 4), complex) / 4, np.eye(2) * 1e-9, np.zeros(2))
            mode = WaveMode(plane, .2, EMITTED_ELECTRON_REFERENCE, "fixture:0", 300.)
            checkpoint = WaveCheckpoint(BeamState((mode,), EMITTED_ELECTRON_REFERENCE), z_mm,
                source.reference_current_a, {"schema": "mocked-pipeline-not-physical"})
            return SimpleNamespace(checkpoint=checkpoint, request=self.request, instrument_digest=Inputs.digest)

        def close(self):
            calls.append("close")

    monkeypatch.setattr(mixed_wave_pipeline, "MixedWaveObservationSession", Session)
    return SimpleNamespace(calls=calls, state=state, result=particles, source=source)


def test_cuda_failure_precedes_archive_load_and_output_creation(monkeypatch, tmp_path):
    from temsim.physics import compute_backend

    def unavailable(*_args, **_kwargs):
        raise compute_backend.GPUExecutionError("unavailable", "synthetic CUDA unavailable")

    monkeypatch.setattr(compute_backend, "require_coherent_cuda", unavailable)
    monkeypatch.setattr(cli, "_particle_result", lambda *_: pytest.fail("CUDA failure loaded particle inputs"))
    output = tmp_path / "unstarted"
    with pytest.raises(compute_backend.GPUExecutionError, match="synthetic CUDA unavailable"):
        cli.main(["--result", str(tmp_path / "missing.temresult"), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("plane, expected", [("upper", 1605. - 5e-6), ("exit", 1605. + 5e-6)])
def test_checked_result_codec_supplies_one_boundary_and_streams_safe_outputs(pipeline, tmp_path, plane, expected):
    output = tmp_path / "mixed"
    archive = tmp_path / "saved.temresult"
    assert cli.main(["--result", str(archive), "--output", str(output), "--plane", plane]) == 0
    assert pipeline.calls[0] == "cuda"
    assert ("load", archive) in pipeline.calls
    assert ("observe", expected) in pipeline.calls
    assert sum(isinstance(call, tuple) and call[0] == "source" for call in pipeline.calls) == 1
    assert pipeline.calls[-1] == "close"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED"
    assert report["original_positive_particles"] == 1024 and report["handoff_particle_count"] == 3
    assert report["handoff_absolute_weight"] == .2
    with np.load(output / "mixed_wave_arrays.npz", allow_pickle=False) as arrays:
        for name in arrays.files:
            assert not arrays[name].dtype.hasobject
        np.testing.assert_array_equal(arrays["source_ray_id"], [3, 7, 11])
        np.testing.assert_array_equal(arrays["source_weight"], pipeline.source.weight)
        np.testing.assert_array_equal(arrays["original_gun_ray_id"], np.arange(1024))
        assert arrays["mode_00000_diffraction_probability"].sum() == pytest.approx(.2)
        assert arrays["mode_00000_intensity_per_m2"].sum() * 1e-18 == pytest.approx(.2)
    assert json.loads((output / "inputs.json").read_text())["lens_strength"] == 42.


@pytest.mark.parametrize("quality", ["Preview", "High accuracy"])
def test_profile_particle_calculation_preserves_optics_and_saved_step(pipeline, monkeypatch, tmp_path, quality):
    import temsim.simulation_pipeline as simulation
    from temsim.working_point import WorkingPointCheckpoint

    monkeypatch.setattr(WorkingPointCheckpoint, "read_package",
        lambda path: SimpleNamespace(compatible_state=lambda: pipeline.state))

    def calculate(state, **kwargs):
        pipeline.calls.append(("calculate", state._tuning_quality, state.step_mm, kwargs["workflow"]))
        assert state.electron_gun.emitter.ray_count == 8192
        assert state.lenses[0].strength == 42.
        return pipeline.result

    monkeypatch.setattr(simulation, "calculate", calculate)
    output = tmp_path / "profile-mixed"
    assert cli.main(["--profile", str(tmp_path / "inputs.temwp"), "--particles", "8192",
        "--quality", quality, "--z-mm", "1605.1", "--output", str(output)]) == 0
    assert ("calculate", quality, .025, "rays") in pipeline.calls
    assert ("observe", 1605.1) in pipeline.calls
    assert pipeline.state.step_mm == .025


def test_result_does_not_accept_particle_setting_overrides(pipeline, tmp_path):
    with pytest.raises(SystemExit) as rejected:
        cli.main(["--result", "saved.temresult", "--particles", "1024", "--output", str(tmp_path / "new")])
    assert rejected.value.code == 2
    assert pipeline.calls == ["cuda"]


def test_profile_rejects_insufficient_original_population_before_loading(tmp_path):
    with pytest.raises(SystemExit) as rejected:
        cli.main(["--profile", "inputs.temwp", "--particles", "999", "--output", str(tmp_path / "new")])
    assert rejected.value.code == 2


def test_existing_output_is_not_overwritten_or_loaded(pipeline, tmp_path):
    marker = tmp_path / "existing.json"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="already exists"):
        cli.main(["--result", "saved.temresult", "--output", str(tmp_path)])
    assert marker.read_text() == "preserve"
    assert pipeline.calls == ["cuda"]


def test_invalid_wave_numerics_reject_before_particle_load_or_calculation(pipeline, tmp_path):
    with pytest.raises(ValueError, match="pixels must be even"):
        cli.main(["--result", "saved.temresult", "--pixels", "513", "--output", str(tmp_path / "new")])
    assert pipeline.calls == ["cuda"]


@pytest.mark.parametrize("shift", [None, np.array([[.01, -.02], [.03, .04], [-.05, .06]])])
def test_optional_canonical_shift_export_and_device_budget_match_declared_inputs(pipeline, tmp_path, shift):
    pipeline.source.canonical_frequency_shift_inv_angstrom = shift
    mechanical_slopes = np.stack((pipeline.source.tx_rad, pipeline.source.ty_rad)).copy()
    output = tmp_path / "canonical-export"
    assert cli.main(["--result", "saved.temresult", "--working-gib", "8", "--output", str(output)]) == 0
    request = next(call[2] for call in pipeline.calls if isinstance(call, tuple) and call[0] == "session")
    assert request.source.maximum_working_bytes == 8 * 1024 ** 3
    assert request.wave_grid.maximum_working_bytes == 8 * 1024 ** 3
    assert request.wave_grid.maximum_device_working_bytes == 8 * 1024 ** 3
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    canonical = report["canonical_frequency_initialization"]
    assert canonical["shift_supplied"] is (shift is not None)
    assert canonical["mechanical_trajectory_slopes_retained"] is True
    assert "q*A_xy/h" in canonical["shift_formula"]
    with np.load(output / "mixed_wave_arrays.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(np.stack((arrays["source_tx_rad"], arrays["source_ty_rad"])), mechanical_slopes)
        if shift is None:
            assert canonical["array"] is None
            assert "source_canonical_frequency_shift_inv_angstrom" not in arrays.files
        else:
            np.testing.assert_array_equal(arrays[canonical["array"]], shift)
