"""Cheap CLI checks; no source launch or ray propagation."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_nanoprobe_condenser.py"
    spec = spec_from_file_location("validate_nanoprobe_condenser_cli", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_defaults_match_calibrated_budget_and_keep_small_budget_explicit(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_nanoprobe_condenser.py"])
    args = cli._arguments()
    assert (args.rays, args.threads, args.step, args.target) == (3000, 2, .05, 30.)
    assert args.range == "normal"
    monkeypatch.setattr(sys, "argv", ["validate_nanoprobe_condenser.py", "--rays", "512"])
    assert cli._arguments().rays == 512


def test_large_angle_is_an_explicit_branch_with_unchanged_acceptance(cli, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["validate_nanoprobe_condenser.py", "--range", "large_angle"])
    args = cli._arguments()
    assert args.range == "large_angle"
    assert (args.rays, args.target, args.step) == (3000, 30., .05)
    monkeypatch.setattr(sys, "argv", ["validate_nanoprobe_condenser.py", "--range", "unknown"])
    with pytest.raises(SystemExit):
        cli._arguments()


def test_candidate_applies_only_supported_controls(cli, tmp_path):
    lens = SimpleNamespace(key="condenser_lens_3", percent=20., max_percent=75.)
    hp = SimpleNamespace(key="probe_hp1_hexapole", strength_m3=1., orientation_rad=0.)
    state = SimpleNamespace(lenses=[lens], corrector_elements=[hp])
    path = tmp_path / "candidate.json"
    values = dict(lenses={lens.key: 33.},
                  hexapoles={hp.key: dict(strength_m3=-3., orientation_rad=.5)})
    text = json.dumps(values)
    path.write_text(text, encoding="utf-8")
    cli._load_candidate(state, path)
    assert (lens.percent, hp.strength_m3, hp.orientation_rad) == (33., -3., .5)
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("bad_lenses", [
    {"condenser_lens_3": 76.}, {"condenser_lens_3": float("nan")},
    {"condenser_lens_3": True}, {"missing_lens": 10.},
])
def test_invalid_candidate_control_is_rejected(cli, tmp_path, bad_lenses):
    lens = SimpleNamespace(key="condenser_lens_3", percent=20., max_percent=75.)
    state = SimpleNamespace(lenses=[lens], corrector_elements=[])
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(dict(lenses=bad_lenses, hexapoles={})), encoding="utf-8")
    with pytest.raises(ValueError):
        cli._load_candidate(state, path)
    assert lens.percent == 20.


def test_execution_error_saves_failure_diagnostic_and_returns_nonzero(cli, tmp_path, monkeypatch):
    output = tmp_path / "diagnostic.json"
    monkeypatch.setattr(sys, "argv", ["validate_nanoprobe_condenser.py", "--output", str(output)])

    def fail(args, report):
        raise ValueError("candidate is unavailable")

    monkeypatch.setattr(cli, "_execute", fail)
    assert cli.main() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "FAIL"
    assert report["errors"] == [{"type": "ValueError", "message": "candidate is unavailable"}]
    assert report["angle_relative_tolerance"] == .01
    assert report["focus_tolerance_nm"] == 1.
    assert report["sampling_convergence"] == "NOT_TESTED_FIXED_PARTICLE_BUDGET"
    assert report["actual_backends"] == []
