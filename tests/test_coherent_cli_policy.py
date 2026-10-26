"""Coherent diagnostic CLI admission only; no physical propagation is executed."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from temsim.physics import compute_backend as backend


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_cli(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"coherent_policy_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def arguments(name, root):
    output = str(root / "output")
    if name == "inspect_surface_column":
        return ["--output", output]
    if name == "audit_gun_axial_steps":
        return [str(root / "report.json"), "--output", output]
    if name == "validate_ac_wave_downstream":
        return ["--baseline", str(root / "baseline"), "--cache-key", "saved-key",
                "--output", output, "--labels", "centre", "corner", "--selection-reason", "Test policy"]
    if name == "validate_ac_wave_handoff":
        return ["--producer", str(root / "producer"), "--cache-key", "saved-key", "--output", output]
    if name == "validate_ac_wave_scan":
        return ["--output", output, "--previous", str(root / "previous")]
    if name == "benchmark_wave_hotpaths":
        return ["--source-root", str(SCRIPTS.parent / "src"), "--fixture", str(root / "fixture"),
                "--instrument", str(root / "instrument.json"), "--output", output]
    if name == "benchmark_stem_wave":
        return ["--repeat", "1"]
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["inspect_surface_column", "audit_gun_axial_steps",
    "validate_ac_wave_downstream", "validate_ac_wave_handoff", "validate_ac_wave_scan",
    "benchmark_wave_hotpaths", "benchmark_stem_wave"])
def test_diagnostic_execution_rejects_missing_cuda_before_inputs_or_output(name, monkeypatch, tmp_path):
    cli = load_cli(name, monkeypatch)
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: backend.BackendCapability(False, "synthetic unavailable CUDA"))

    def forbidden(*_args, **_kwargs):
        pytest.fail("No input preparation or wave propagation is permitted without CUDA")

    for function in ("default_state", "prepare", "load_inputs", "benchmark_inputs"):
        if hasattr(cli, function):
            monkeypatch.setattr(cli, function, forbidden)
    with pytest.raises(backend.GPUExecutionError, match="synthetic unavailable CUDA"):
        cli.main(arguments(name, tmp_path))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("name,option", [
    ("inspect_surface_column", ["--radial-backend", "cpu"]),
    ("benchmark_stem_wave", ["--backends", "CPU"]),
    ("benchmark_stem_wave", ["--backends", "Numba CPU"]),
])
def test_explicit_cpu_diagnostics_reject_before_loading_cuda(name, option, monkeypatch, tmp_path):
    cli = load_cli(name, monkeypatch)
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("Explicit CPU rejection must not probe CUDA"))
    with pytest.raises(backend.GPUExecutionError, match="CPU execution.*disabled"):
        cli.main(arguments(name, tmp_path) + option)
    assert not (tmp_path / "output").exists()


class PreparationReached(Exception):
    """Stop at substituted metadata preparation, before any physical work."""


@pytest.mark.parametrize("name,entry", [
    ("validate_ac_wave_downstream", "load_inputs"),
    ("validate_ac_wave_handoff", "prepare"),
    ("validate_ac_wave_scan", "prepare"),
])
def test_prepare_only_can_inspect_inputs_without_cuda(name, entry, monkeypatch, tmp_path):
    cli = load_cli(name, monkeypatch)
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("Metadata preparation must not probe CUDA"))

    def prepare(*_args):
        raise PreparationReached()

    monkeypatch.setattr(cli, entry, prepare)
    with pytest.raises(PreparationReached):
        cli.main(arguments(name, tmp_path) + ["--prepare-only"])


def test_handoff_patch_pinning_does_not_probe_cuda(monkeypatch, tmp_path):
    cli = load_cli("validate_ac_wave_handoff", monkeypatch)
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: pytest.fail("Pinning existing evidence must not probe CUDA"))
    calls = []
    monkeypatch.setattr(cli, "pin_changes", lambda *args: calls.append(args) or {"status": "PINNED"})
    cli.main(["--producer", str(tmp_path / "producer"), "--pin-patches",
              str(tmp_path / "first.patch"), str(tmp_path / "second.patch")])
    assert len(calls) == 1


@pytest.mark.parametrize("policy", [None, "Auto", "CUDA GPU", "Prefer GPU", "Require GPU"])
def test_stem_benchmark_passes_strict_policy_to_substituted_execution(policy, monkeypatch, tmp_path):
    cli = load_cli("benchmark_stem_wave", monkeypatch)
    monkeypatch.setattr(backend, "coherent_cuda_capability",
        lambda: backend.BackendCapability(True, "synthetic CUDA"))
    state = SimpleNamespace()
    monkeypatch.setattr(cli, "benchmark_inputs", lambda *_: (state, object(), (), (), (), object()))

    def execute(*_args, **_kwargs):
        assert state.acceleration_backend == "Require GPU"
        assert state.acceleration_enabled
        raise PreparationReached()

    monkeypatch.setattr(cli, "simulate_angle_resolved_stem", execute)
    args = arguments("benchmark_stem_wave", tmp_path)
    if policy is not None:
        args += ["--backends", policy]
    with pytest.raises(PreparationReached):
        cli.main(args)
