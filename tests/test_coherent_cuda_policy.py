"""User wave admission with fake capabilities; never execute a coherent job."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import compute_backend as backend
from temsim.physics import tip_wave_pipeline as pipeline


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: backend.BackendCapability(False, "synthetic missing CuPy CUDA"))


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: backend.BackendCapability(True, "synthetic CUDA capability"))


@pytest.mark.parametrize("policy", ["Auto", "Prefer GPU", "CUDA GPU", "Require GPU"])
def test_user_wave_policies_require_cuda_and_never_fall_back(no_cuda, policy):
    with pytest.raises(backend.GPUExecutionError, match="Coherent calculations currently require.*CPU fallback"):
        backend.require_coherent_cuda(policy)


@pytest.mark.parametrize("policy,enabled", [("CPU", True), ("cpu", True),
    ("Numba CPU", True), ("numpy", True), ("Auto", False)])
def test_cpu_or_disabled_user_requests_are_rejected_without_probing(monkeypatch, policy, enabled):
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("An explicitly disabled request must not load CUDA"))
    with pytest.raises(backend.GPUExecutionError, match="Enable acceleration"):
        backend.require_coherent_cuda(policy, acceleration_enabled=enabled)


@pytest.mark.parametrize("policy", ["Auto", "Prefer GPU", "CUDA GPU", "Require GPU"])
def test_admitted_user_policy_is_strict_without_mutating_controls(fake_cuda, policy):
    original = SimpleNamespace(acceleration_backend=policy, acceleration_enabled=True)
    working = backend.coherent_execution_state(original)
    assert working is not original
    assert working.acceleration_backend == "Require GPU"
    assert original.acceleration_backend == policy


def test_enumeration_alone_cannot_admit_a_broken_cuda_fft(monkeypatch):
    class DeviceArray(np.ndarray):
        __cuda_array_interface__ = {}

    def broken_fft(_values):
        raise RuntimeError("synthetic cuFFT loading failure")

    fake = SimpleNamespace(asarray=lambda values: values.view(DeviceArray),
        fft=SimpleNamespace(fft2=broken_fft), asnumpy=np.asarray)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(backend, "cupy_capability",
        lambda: backend.BackendCapability(True, "enumerated only"))
    assert not backend.coherent_cuda_capability().available
    with pytest.raises(backend.GPUExecutionError, match="synthetic cuFFT loading failure"):
        backend.require_coherent_cuda()


def test_capability_probe_checks_device_array_and_fft_result(monkeypatch):
    class DeviceArray(np.ndarray):
        __cuda_array_interface__ = {}

    calls = []
    def fft(values):
        calls.append(values.shape)
        return np.fft.fft2(values)
    fake = SimpleNamespace(asarray=lambda values: values.view(DeviceArray),
        fft=SimpleNamespace(fft2=fft), asnumpy=np.asarray)
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(backend, "cupy_capability",
        lambda: backend.BackendCapability(True, "test device"))
    assert backend.coherent_cuda_capability().available
    assert calls == [(2, 2)]
    fake.asarray = np.asarray
    assert not backend.coherent_cuda_capability().available


@pytest.mark.parametrize("entry", ["single", "observation", "scan", "reference"])
def test_tip_entries_reject_before_snapshot_or_preparation(no_cuda, monkeypatch, entry):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Missing CUDA must reject before instrument capture or preparation")
    monkeypatch.setattr(pipeline, "capture_instrument_snapshot", forbidden)
    monkeypatch.setattr(pipeline, "_execute_tip_wave", forbidden)
    state = object()
    with pytest.raises(backend.GPUExecutionError, match="synthetic missing CuPy CUDA"):
        if entry == "single":
            pipeline.simulate_tip_wave(state)
        elif entry == "observation":
            session = pipeline.TipWaveObservationSession(state,
                pipeline.TipWaveRequest(stop="plane", observation_z_mm=1.))
            session.observe(2.)
        elif entry == "scan":
            from temsim.physics import tip_wave_scan
            monkeypatch.setattr(tip_wave_scan, "capture_instrument_snapshot", forbidden)
            next(tip_wave_scan.simulate_tip_scan(state))
        else:
            actual = pipeline.TipWaveResult(SimpleNamespace(plane_z_mm=1.), None,
                pipeline.TipWaveRequest(stop="plane", observation_z_mm=1.), False, "synthetic")
            pipeline.simulate_ideal_reference(state, actual)


def test_tip_entry_passes_strict_policy_to_substituted_execution(fake_cuda, monkeypatch):
    working = object()
    snapshot = SimpleNamespace(restore=lambda: working, digest="synthetic")
    monkeypatch.setattr(pipeline, "capture_instrument_snapshot", lambda state: snapshot)
    from temsim.physics import wave_device
    monkeypatch.setattr(wave_device, "release_device_memory", lambda: None)
    def execute(state, request, *_args, **_kwargs):
        assert state is working
        return request
    monkeypatch.setattr(pipeline, "_execute_tip_wave", execute)
    original = pipeline.TipWaveRequest(wave_grid=replace(
        pipeline.TipWaveRequest().wave_grid, compute_backend="Prefer GPU"))
    executed = pipeline.simulate_tip_wave(object(), original)
    assert executed.wave_grid.compute_backend == "Require GPU"
    assert executed.radial_column.backend == "cuda"
    assert original.wave_grid.compute_backend == "Prefer GPU"
    assert original.radial_column.backend == "cuda"
    with pytest.raises(backend.GPUExecutionError, match="out_of_memory"):
        backend.gpu_retry_reason(backend.GPUExecutionError("out_of_memory", "synthetic"),
            executed.wave_grid.compute_backend)


def test_explicit_cpu_radial_request_is_not_silently_rewritten(monkeypatch):
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("An explicit CPU radial request must not probe CUDA"))
    request = pipeline.TipWaveRequest(radial_column=replace(
        pipeline.TipWaveRequest().radial_column, backend="cpu"))
    with pytest.raises(backend.GPUExecutionError, match="CUDA radial transforms"):
        pipeline.require_tip_wave_cuda(request)
    assert request.radial_column.backend == "cpu"


@pytest.mark.parametrize("entry", ["tem", "stem", "reprojection"])
def test_tem_stem_entries_reject_before_numerical_operators(no_cuda, monkeypatch, entry):
    from temsim.physics import source_admission, wave_imaging, stem_wave_imaging, illumination
    monkeypatch.setattr(source_admission, "require_gun_wave_source", lambda *a, **kw: None)
    monkeypatch.setattr(illumination, "require_production_illumination", lambda *a, **kw: None)
    for module in (wave_imaging, stem_wave_imaging):
        monkeypatch.setattr(module, "illumination_config", lambda *_: None)
    def forbidden(*_args, **_kwargs):
        pytest.fail("CUDA must be checked before numerical operators")
    monkeypatch.setattr(wave_imaging, "_simulate_wave_image", forbidden)
    monkeypatch.setattr(stem_wave_imaging, "_simulate_angle_resolved_stem", forbidden)
    state = SimpleNamespace(acceleration_backend="Auto", acceleration_enabled=True)
    with pytest.raises(backend.GPUExecutionError, match="synthetic missing CuPy CUDA"):
        if entry == "tem":
            wave_imaging.simulate_wave_image(state, object())
        elif entry == "stem":
            stem_wave_imaging.simulate_angle_resolved_stem(state, object(), (), [], [])
        else:
            wave_imaging.reproject_wave_image(state, SimpleNamespace(metrics={}))


def test_classical_cpu_and_small_cpu_wave_operators_remain_available(monkeypatch):
    monkeypatch.setattr(backend, "cupy_capability", lambda: pytest.fail("CPU reference must not probe CUDA"))
    monkeypatch.setattr(backend, "cuda_capability", lambda: pytest.fail("Classical CPU must not probe CUDA"))
    assert backend.choose_ray_backend("CPU", acceleration_enabled=False, ray_count=8) == ("CPU", None)
    assert backend.choose_wave_backend("CPU", acceleration_enabled=False, work_items=16) == ("NumPy CPU", None)


def test_completed_wave_readout_does_not_require_or_probe_cuda(monkeypatch):
    from temsim.detector.wave_readout import read_wave_detector
    from test_wave_detector_readout import checkpoint, detector
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("Reading a completed checkpoint must not probe CUDA"))
    completed = checkpoint()
    readout = read_wave_detector(completed, detector())
    assert readout.optical_probability.sum() == pytest.approx(.6)
    assert readout.modes[0].plane is completed.beam.modes[0].plane


def test_gui_checks_cuda_before_capture_but_configuration_capture_is_allowed(no_cuda, monkeypatch):
    from temsim.gui import coherent_beam
    calls = []
    state = SimpleNamespace()
    monkeypatch.setattr(coherent_beam, "prepare_coherent_state",
        lambda state, settings: calls.append("capture") or state)
    monkeypatch.setattr(coherent_beam, "wave_input_summary",
        lambda *_: {"status": "SOURCE_READY"})
    owner = SimpleNamespace(_settings=lambda: object(), state_provider=None, _state=state,
        _make_request=lambda *_: pipeline.TipWaveRequest(), result=object(),
        _show_source_check=lambda *a: None,
        progress_detail=SimpleNamespace(emit=lambda *_: None),
        status=SimpleNamespace(setStyleSheet=lambda *_: None, setText=lambda *_: None))
    with pytest.raises(backend.GPUExecutionError):
        coherent_beam.CoherentBeamPage.capture_calculation_inputs(owner)
    assert calls == []
    captured, _, _ = coherent_beam.CoherentBeamPage.capture_calculation_inputs(owner, for_execution=False)
    assert captured is state and calls == ["capture"]
    assert owner.result is not None


def test_cli_requires_cuda_before_loading_an_instrument(no_cuda, monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("cuda_admission_cli",
        Path(__file__).parents[1]/"scripts"/"trace_tip_wave.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "default_state", lambda: pytest.fail("No instrument should be prepared"))
    output = tmp_path/"unstarted"
    with pytest.raises(backend.GPUExecutionError, match="synthetic missing CuPy CUDA"):
        cli.main(["--profile", str(tmp_path/"unused.toml"), "--output", str(output)])
    assert not output.exists()
