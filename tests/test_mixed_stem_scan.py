"""Mocked scan orchestration, policy and input protection; no physical trial."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import mixed_stem_scan as scan
from temsim.physics.particle_wave_source import ParticleWaveSource
from temsim.physics.stem_wave_imaging import AngularDetector
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_checkpoint import EMITTED_ELECTRON_REFERENCE, WaveCheckpoint
from temsim.physics.wave_flux import BeamState, WaveMode


@pytest.fixture
def mocked_scan(monkeypatch):
    from temsim.specimen.scene import SpecimenScene
    from temsim.specimen import inelastic
    calls = []
    state = SimpleNamespace(token="a" * 64, simulation_mode="unified",
        electron_gun=SimpleNamespace(emitter=SimpleNamespace(energy_spread_fwhm_ev=.3)),
        sample=SimpleNamespace(z_mm=100., thickness_nm=10., wave_frozen_phonon_enabled=False))
    source = ParticleWaveSource(x_m=np.array([-1., 0., 1.])*1e-10, y_m=np.zeros(3),
        tx_rad=np.zeros(3), ty_rad=np.zeros(3), ray_id=np.array([7, 12, 19]),
        weight=np.array([.1, .07, .03]), kinetic_energy_ev=np.array([300000.1, 300000.2, 300000.3]),
        flight_time_s=np.full(3, 1e-8), z_mm=100.-5e-6,
        reference_current_a=1e-8, original_positive_count=1024, input_digest=state.token,
        provenance={"scope": "Synthetic array fixture; no finite-source propagation"})
    mode = WaveMode(PlaneWave(np.ones((2, 2), complex)/2, np.eye(2)*1e-10, np.zeros(2)),
        .2, EMITTED_ELECTRON_REFERENCE, "mocked-source", source.mean_energy_ev/1000.)
    checkpoint = WaveCheckpoint(BeamState((mode,), EMITTED_ELECTRON_REFERENCE), source.z_mm,
        source.reference_current_a, {"particle_source_digest": source.digest, "input_digest": state.token,
            "numerics": asdict(scan.MixedStemScanRequest().source),
            "original_positive_count": source.original_positive_count,
            "handoff_particle_count": len(source.ray_id), "source_fraction": source.source_fraction})
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *a, **k: calls.append(("cuda", a, k)) or "Require GPU")
    monkeypatch.setattr(scan, "capture_instrument_snapshot", lambda value: SimpleNamespace(digest=value.token))
    monkeypatch.setattr(SpecimenScene, "from_state", staticmethod(lambda *_:
        SimpleNamespace(is_vacuum=False, structure_available=True, thickness_nm=10.)))
    monkeypatch.setattr(inelastic, "real_inelastic_distribution", lambda _: SimpleNamespace(
        total_inelastic_mean_free_path_nm=np.inf, absorption_mean_free_path_nm=np.inf,
        model="Static elastic array fixture", material_key="mock", warnings=()))
    monkeypatch.setattr(scan, "cupy_module", lambda: object())
    monkeypatch.setattr(scan, "_rectangular_crystal", lambda _:
        (np.array([2.,3.,4.]),np.zeros((1,3)),np.array([14]),{"scope":"Mocked quarter-lattice proof"}))

    def execute(cp, captured, original_source, original_checkpoint, request, positions, crystal, cancelled, progress):
        calls.append(("execute", positions.copy()))
        count = len(positions)
        return {"bf": np.linspace(.02, .08, count), "adf": np.full(count, .01)}, \
            np.ones((2, 2))*.03, np.array([-.1, .1]), np.array([-.1, .1]), {"scope": "Mocked transport only"}

    monkeypatch.setattr(scan, "_execute_scan", execute, raising=False)
    monkeypatch.setattr(scan, "build_particle_wave_checkpoint",
        lambda *_a, **_k: pytest.fail("A compatible supplied checkpoint was regenerated"))
    return SimpleNamespace(state=state, source=source, checkpoint=checkpoint, calls=calls)


def test_cuda_unavailable_rejects_before_source_capture_or_cached_checkpoint(monkeypatch):
    from temsim.physics.compute_backend import GPUExecutionError

    def unavailable(*_a, **_k):
        raise GPUExecutionError("unavailable", "mock missing CUDA")

    monkeypatch.setattr(scan, "require_coherent_cuda", unavailable)
    monkeypatch.setattr(scan, "capture_instrument_snapshot",
        lambda *_: pytest.fail("Unavailable CUDA reached source capture"))
    with pytest.raises(GPUExecutionError, match="mock missing CUDA"):
        scan.simulate_mixed_stem_scan(object(), object(), checkpoint=object())


def test_unsupported_cif_is_rejected_before_local_source_lift(mocked_scan,monkeypatch):
    monkeypatch.setattr(scan,"_rectangular_crystal",lambda _:None)
    with pytest.raises(ValueError,match="rectangular CIF"):
        scan.simulate_mixed_stem_scan(mocked_scan.state,mocked_scan.source)
    assert not any(call[0]=='execute' for call in mocked_scan.calls)


def test_existing_checkpoint_still_uses_strict_cuda_policy(mocked_scan, monkeypatch):
    from temsim.physics import compute_backend
    admitted = []
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability",
        lambda: compute_backend.BackendCapability(True, "Synthetic policy capability only"))

    def require(*args, **kwargs):
        policy = compute_backend.require_coherent_cuda(*args, **kwargs)
        admitted.append(policy)
        return policy

    monkeypatch.setattr(scan, "require_coherent_cuda", require)
    item = mocked_scan
    scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint,
        positions_nm=np.array([[0., 0.]]))
    assert admitted == ["Require GPU"]


@pytest.mark.parametrize("changes", [{"pixels": True}, {"pixels": 0}, {"batch_size": 0},
    {"maximum_working_bytes": 0}, {"pitch_nm": 0.}, {"pitch_nm": np.nan},
    {"maximum_modulation_error": np.nextafter(.05, 1.)},
    {"bandwidth_fraction": np.nextafter(2/3, 1.)}, {"save_diffraction": 1}])
def test_request_rejects_invalid_controls_and_relaxed_existing_budgets(changes):
    with pytest.raises(ValueError):
        replace(scan.MixedStemScanRequest(), **changes).validate()


@pytest.mark.parametrize("detectors", [(),
    (AngularDetector("same", 0., 15.), AngularDetector("same", 50., 150.)),
    (AngularDetector("bf", 0., 15.), AngularDetector("adf", 14., 150.)),
    (AngularDetector("adf", 50., 500.),)])
def test_request_requires_unique_nonoverlapping_finite_detector_annuli(detectors):
    with pytest.raises(ValueError):
        replace(scan.MixedStemScanRequest(), detectors=detectors).validate()


def test_absolute_detector_current_and_source_identity_are_preserved(mocked_scan):
    item = mocked_scan
    before = {name: getattr(item.source, name).copy() for name in
        ("x_m", "y_m", "tx_rad", "ty_rad", "weight", "ray_id", "kinetic_energy_ev", "flight_time_s")}
    positions = np.array([[0., 0.], [.1, -.2]])
    result = scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint, positions_nm=positions)
    np.testing.assert_array_equal(result.fractions["bf"], [.02, .08])
    np.testing.assert_array_equal(result.current_a["bf"], np.array([.02, .08])*1e-8)
    assert result.source_checkpoint is item.checkpoint
    assert result.record["original_positive_count"] == 1024
    assert result.record["surviving_positive_count"] == 3
    assert result.record["tip_coherent_executed"] is False
    np.testing.assert_allclose(result.record["positions_nm"], positions, rtol=0, atol=3e-17)
    np.testing.assert_array_equal(item.calls[1][1], positions*10.)
    for name, original in before.items():
        np.testing.assert_array_equal(getattr(item.source, name), original)


def test_default_raster_is_centered_and_restored_to_two_dimensional_shape(mocked_scan):
    item = mocked_scan
    request = replace(scan.MixedStemScanRequest(), pixels=3, pitch_nm=.02)
    result = scan.simulate_mixed_stem_scan(item.state, item.source, request, checkpoint=item.checkpoint)
    assert result.fractions["bf"].shape == (3, 3)
    np.testing.assert_allclose(result.scan_x_nm[1], [-.02, 0., .02], rtol=0, atol=0)
    np.testing.assert_allclose(result.scan_y_nm[:, 1], [-.02, 0., .02], rtol=0, atol=0)


def test_changed_particle_inputs_reject_before_wave_or_material_execution(mocked_scan, monkeypatch):
    item = mocked_scan
    item.state.token = "b" * 64
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Changed inputs reached material"))
    with pytest.raises(ValueError, match="inputs changed after particle transport"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint)


def test_exact_upper_surface_gate_does_not_accept_a_nearby_source_plane(mocked_scan):
    item = mocked_scan
    changed = replace(item.source, z_mm=item.source.z_mm+16*abs(np.spacing(item.source.z_mm)))
    with pytest.raises(ValueError, match="upper-surface"):
        scan.simulate_mixed_stem_scan(item.state, changed, checkpoint=item.checkpoint)


def test_supplied_checkpoint_cannot_reuse_different_coherence_numerics(mocked_scan, monkeypatch):
    item = mocked_scan
    # Matching input and particle IDs cannot turn a zero-pm mixed state into
    # the currently requested ten-pm translation model.
    changed = replace(item.checkpoint, record={**item.checkpoint.record,
        "numerics": {**item.checkpoint.record["numerics"], "source_image_rms_pm": 0.}})
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Mismatched coherence reached material"))
    with pytest.raises(ValueError, match="numerics|numerical|coherence|source settings"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=changed)


@pytest.mark.parametrize("field,value", [("original_positive_count", 1000),
    ("handoff_particle_count", 2), ("source_fraction", .3)])
def test_supplied_checkpoint_cannot_relabel_the_emitted_current_ledger(mocked_scan, monkeypatch, field, value):
    item = mocked_scan
    changed = replace(item.checkpoint, record={**item.checkpoint.record, field: value})
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Relabelled source reached material"))
    with pytest.raises(ValueError, match="current, original population or stationary energy"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=changed)


@pytest.mark.parametrize("mismatch", ["current", "energy"])
def test_supplied_checkpoint_cannot_change_reference_current_or_wave_energy(mocked_scan, monkeypatch, mismatch):
    item = mocked_scan
    if mismatch == "current":
        changed = replace(item.checkpoint, reference_current_a=item.source.reference_current_a*2)
    else:
        mode = replace(item.checkpoint.beam.modes[0], energy_kev=301.)
        changed = replace(item.checkpoint, beam=BeamState((mode,), EMITTED_ELECTRON_REFERENCE))
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Changed source current/energy reached material"))
    with pytest.raises(ValueError, match="current, original population or stationary energy"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=changed)


@pytest.mark.parametrize("setting,value,message", [
    ("wave_multislice_enabled", False, "actual multislice"),
    ("real_high_angle_tail_enabled", True, "high-angle material channel"),
    ("wave_frozen_phonon_enabled", True, "frozen-phonon histories")])
def test_unsupported_selected_material_channel_is_not_silently_discarded(mocked_scan, monkeypatch, setting, value, message):
    item = mocked_scan
    setattr(item.state.sample, setting, value)
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Unsupported material reached propagation"))
    with pytest.raises(ValueError, match=message):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint)


@pytest.mark.parametrize("channel", ["total_inelastic_mean_free_path_nm", "absorption_mean_free_path_nm"])
def test_selected_finite_material_loss_history_is_not_silently_discarded(mocked_scan, monkeypatch, channel):
    from temsim.specimen import inelastic
    item = mocked_scan
    distribution = SimpleNamespace(total_inelastic_mean_free_path_nm=np.inf,
        absorption_mean_free_path_nm=np.inf)
    setattr(distribution, channel, 50.)
    monkeypatch.setattr(inelastic, "real_inelastic_distribution", lambda _: distribution)
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Unsupported material loss reached propagation"))
    with pytest.raises(ValueError, match="inelastic/absorption channels"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint)


def test_stationary_energy_approximation_refuses_an_excessive_actual_energy_span(mocked_scan, monkeypatch):
    item = mocked_scan
    source = replace(item.source, kinetic_energy_ev=np.array([299999., 300000., 300002.]))
    monkeypatch.setattr(scan, "_execute_scan", lambda *_a, **_k: pytest.fail("Rejected energy span reached material"))
    with pytest.raises(ValueError, match="2 eV span guard"):
        scan.simulate_mixed_stem_scan(item.state, source, checkpoint=item.checkpoint)


def test_input_mutation_during_scan_prevents_result_publication(mocked_scan, monkeypatch):
    item = mocked_scan
    original = scan._execute_scan

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        item.state.token = "b" * 64
        return result

    monkeypatch.setattr(scan, "_execute_scan", mutate)
    with pytest.raises(ValueError, match="inputs changed during acquisition"):
        scan.simulate_mixed_stem_scan(item.state, item.source, checkpoint=item.checkpoint,
            positions_nm=np.array([[0., 0.]]))


def test_cancelled_scan_never_checks_cuda_or_publishes_partial_image(monkeypatch):
    monkeypatch.setattr(scan, "require_coherent_cuda", lambda *_a, **_k: pytest.fail("Cancelled scan probed CUDA"))
    with pytest.raises(InterruptedError, match="no partial image published"):
        scan.simulate_mixed_stem_scan(object(), object(), cancelled=lambda: True)
