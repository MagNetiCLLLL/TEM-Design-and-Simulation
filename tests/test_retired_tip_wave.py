"""Retired execution rejects before preparation; stored mathematics stays usable."""
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics.retired_tip_wave import RetiredTipCoherentWorkflow


def _entry_call(name, state):
    from temsim.physics import tip_wave_pipeline as pipeline

    if name == "session":
        session = object.__new__(pipeline.TipWaveObservationSession)
        return session.observe(1605.)
    if name == "private_pipeline":
        return pipeline._execute_tip_wave(state, None, None, "cached", use_cache=True,
            cancelled=lambda: False, progress_callback=None)
    if name == "ideal":
        return pipeline.simulate_ideal_reference(state, None)
    if name == "pipeline":
        return pipeline.simulate_tip_wave(state, use_cache=True)
    if name == "scan":
        from temsim.physics.tip_wave_scan import simulate_tip_scan
        return simulate_tip_scan(state, use_cache=True)
    if name == "hybrid":
        from temsim.physics.hybrid_pipeline import simulate_hybrid
        return simulate_hybrid(state, None)
    if name in {"quantum_emission", "quantum_boundary"}:
        from temsim.optics.electron_gun import tip_coherence
        function = (tip_coherence.generate_tip_emission if name == "quantum_emission"
                    else tip_coherence.generate_tip_boundary_emission)
        return function(None)
    module, function, args = {
        "tip_gun": ("tip_gun_wave", "build_tip_gun_checkpoint", (None,)),
        "surface_gun": ("surface_gun_wave", "build_surface_gun_checkpoint", (None,)),
        "surface": ("surface_wave", "compute_surface_wave", (None,)),
        "planar": ("planar_tip_boundary", "build_planar_tip_boundary", (state,)),
        "accelerating": ("accelerating_tip_boundary", "execute_accelerating_tip", (state, None, None)),
    }[name]
    return getattr(import_module("temsim.physics." + module), function)(*args)


@pytest.mark.parametrize("policy", ["CPU", "Require GPU"])
@pytest.mark.parametrize("entry", ["pipeline", "session", "private_pipeline", "ideal", "scan",
    "hybrid", "tip_gun", "surface_gun", "surface", "planar", "accelerating",
    "quantum_emission", "quantum_boundary"])
def test_retired_entries_precede_cuda_preparation_and_existing_cache(monkeypatch, entry, policy):
    from temsim.physics import compute_backend, tip_wave_pipeline as pipeline

    def forbidden(*_args, **_kwargs):
        pytest.fail("A retired execution consulted CUDA or prepared physical inputs")

    monkeypatch.setattr(compute_backend, "require_coherent_cuda", forbidden)
    monkeypatch.setattr(compute_backend, "validate_coherent_cuda_policy", forbidden)
    monkeypatch.setattr(pipeline, "capture_instrument_snapshot", forbidden)
    marker = object()
    monkeypatch.setattr(pipeline, "_STAGES", {"existing-result": (marker, 1)})
    state = SimpleNamespace(acceleration_backend=policy, acceleration_enabled=policy != "CPU")
    with pytest.raises(RetiredTipCoherentWorkflow, match="temsim.mixed_wave_cli") as rejected:
        _entry_call(entry, state)
    assert rejected.value.code == "TIP_COHERENT_WORKFLOW_RETIRED"
    assert pipeline._STAGES == {"existing-result": (marker, 1)}


@pytest.mark.parametrize("module, arguments", [
    ("temsim.hybrid_cli", ["--profile", "missing.toml", "--detector", "camera", "--sample-width-nm", "10"]),
    ("scripts.trace_tip_wave", ["--profile", "missing.toml"]),
    ("scripts.inspect_surface_column", []),
    ("scripts.audit_gun_axial_steps", ["missing.json"]),
    ("scripts.validate_ac_wave_scan", []),
    ("scripts.validate_physical_hybrid", ["--hybrid"]),
    ("scripts.cache_high_accuracy", ["--gun-wave-only", "--startup-defaults"]),
    ("scripts.validate_ac_wave_handoff", ["--producer", "missing", "--cache-key", "saved"]),
    ("scripts.validate_ac_wave_downstream", ["--baseline", "missing", "--cache-key", "saved",
        "--labels", "centre", "corner", "--selection-reason", "fixture"]),
])
def test_retired_clis_reject_before_missing_inputs_cuda_or_output_creation(
        monkeypatch, tmp_path, module, arguments):
    from temsim.physics import compute_backend

    def forbidden(*_args, **_kwargs):
        pytest.fail("A retired CLI consulted CUDA")

    monkeypatch.setattr(compute_backend, "require_coherent_cuda", forbidden)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    destination = tmp_path / "must-not-be-created"
    with pytest.raises(RetiredTipCoherentWorkflow, match="near the specimen"):
        import_module(module).main([*arguments, "--output", str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize("module, arguments", [
    ("scripts.validate_hybrid_workflow", ["--output"]),
    ("scripts.audit_current_gun_integrators", ["--output"]),
    ("scripts.freeze_physics_run", ["--driver", "trace_tip_wave", "--snapshot"]),
    ("scripts.replay_gaussian_corrector", ["--case-directory", "missing", "--output"]),
])
def test_argument_only_drivers_cannot_launch_retired_work(monkeypatch, tmp_path, module, arguments):
    destination = tmp_path / "must-not-be-created"
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    monkeypatch.setattr("sys.argv", [module, *arguments, str(destination)])
    with pytest.raises(RetiredTipCoherentWorkflow):
        import_module(module).main()
    assert not destination.exists()


def test_historical_wave_cache_remains_readable_without_execution(tmp_path, monkeypatch):
    from temsim.physics import compute_backend
    from temsim.physics.multiplane_wave import PlaneWave
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState, WaveMode

    monkeypatch.setattr(compute_backend, "require_coherent_cuda",
        lambda *_a, **_k: pytest.fail("Reading an archive requested CUDA"))
    amplitude = np.ones((4, 4), dtype=complex) / 4
    plane = PlaneWave(amplitude, np.eye(2) * 1e-9, np.zeros(2))
    reference = "physical-feg-tip-emitted-electron"
    mode = WaveMode(plane, .25, reference, "historical:0", 300.)
    original = TipGunCheckpoint(BeamState((mode,), reference), 450., 1e-8,
        {"schema": "executed-tip-gun-v1", "historical_fixture": True})
    store = ExecutedWaveStore(tmp_path, "historical-format-fixture", 1 << 20)
    key = store.key("stored-result")
    store.put(key, original)
    restored = store.get(key)
    np.testing.assert_array_equal(restored.beam.modes[0].plane.amplitude, amplitude)
    assert restored.reference_current_a == original.reference_current_a
    assert restored.beam.total_weight == .25


def test_retirement_preserves_particle_source_energy_weights_and_identity():
    from temsim.optics.electron_gun.field_emission import FieldEmissionGun
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, tip_particle_samples

    gun = FieldEmissionGun()
    gun.emitter.coherence = TipCoherence()
    gun.emitter.emission_energy_ev = 4.
    gun.emitter.minimum_kinetic_energy_ev = 3.
    gun.emitter.energy_spread_fwhm_ev = 0.
    emitted = tip_particle_samples(gun.emitter, 1024)
    np.testing.assert_array_equal(emitted.ray_id, np.arange(1024))
    np.testing.assert_array_equal(emitted.energy_offset_ev, np.zeros(1024))
    np.testing.assert_array_equal(emitted.weight, np.full(1024, 1 / 1024))
    assert np.isfinite([emitted.x_m, emitted.y_m, emitted.tx_rad, emitted.ty_rad]).all()
