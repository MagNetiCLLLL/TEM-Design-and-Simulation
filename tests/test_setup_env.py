"""Installer compatibility, offline policy, and real local-wheel resolution."""
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace
import zipfile

from packaging.specifiers import SpecifierSet
import pytest

import setup_env


@pytest.mark.parametrize("minor", [10, 11, 12, 13, 14])
def test_runtime_policy_matches_project_metadata(minor):
    project = tomllib.loads((setup_env.PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    supported = f"3.{minor}" in SpecifierSet(project["project"]["requires-python"])
    info = {"version": [3, minor, 0], "bits": 64, "implementation": "cpython"}
    if supported:
        setup_env.validate_runtime(info)
    else:
        with pytest.raises(ValueError, match="64-bit CPython"):
            setup_env.validate_runtime(info)


@pytest.mark.parametrize("change", [{"bits": 32}, {"implementation": "pypy"}])
def test_rejects_unsupported_runtime_architecture(change):
    with pytest.raises(ValueError, match="64-bit CPython"):
        setup_env.validate_runtime({"version": [3, 12, 4], "bits": 64, "implementation": "cpython", **change})


def test_existing_incompatible_venv_is_not_modified(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"existing interpreter")
    wheels.mkdir()
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {"version": [3, 14, 0], "bits": 64, "implementation": "cpython"})
    monkeypatch.setattr(setup_env, "run", lambda _: pytest.fail("must not modify unsupported existing environment"))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels)]) == 1
    assert python.read_bytes() == b"existing interpreter"


def test_offline_gpu_policy_applies_to_every_install_step(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "local wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.touch()
    wheels.mkdir()
    nested = wheels / "cp313"
    nested.mkdir()
    (nested / "candidate.whl").touch()
    commands, verification = [], []
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {"version": [3, 13, 1], "bits": 64, "implementation": "cpython"})
    monkeypatch.setattr(setup_env, "run", commands.append)
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, json.dumps(["setuptools>=77", "wheel"])))
    monkeypatch.setattr(setup_env, "verify", lambda p, **k: verification.append((p, k)))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels), "--offline", "--gpu"]) == 0
    assert len(commands) == 2
    for cmd in commands:
        assert cmd[0] == str(python)
        assert "--no-index" in cmd and "--disable-pip-version-check" in cmd
        assert str(wheels) in cmd and str(nested) in cmd
        assert "--upgrade-deps" not in cmd
    assert "--no-build-isolation" in commands[-1]
    assert all("--constraint" not in cmd and "--requirement" not in cmd for cmd in commands)
    assert commands[-1][-1] == f"{setup_env.PROJECT_ROOT}[dev,gpu]"
    assert verification == [(python, {"gpu": True})]


@pytest.mark.parametrize("minor, platform, locked", [
    (12, "win32", True), (11, "win32", False), (13, "win32", False),
    (12, "linux", False), (12, "darwin", False),
])
def test_lock_applies_only_to_validated_runtime(minor, platform, locked):
    info = {"version": [3, minor, 1], "bits": 64,
            "implementation": "cpython", "platform": platform}
    assert setup_env.validation_lock(info) == (setup_env.CPU_LOCK if locked else None)


@pytest.mark.parametrize("offline, gpu", [(False, False), (True, True)])
def test_validated_setup_synchronises_full_lock_without_losing_install_policy(tmp_path, monkeypatch, offline, gpu):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.touch()
    wheels.mkdir()
    commands, verification = [], []
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {
        "version": [3, 12, 4], "bits": 64, "implementation": "cpython", "platform": "win32"})
    monkeypatch.setattr(setup_env, "run", commands.append)
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, json.dumps(["setuptools>=77", "wheel"])))
    monkeypatch.setattr(setup_env, "verify", lambda p, **k: verification.append((p, k)))
    args = ["--venv", str(env), "--wheel-dir", str(wheels)]
    args += ["--offline"] if offline else []
    args += ["--gpu"] if gpu else []
    assert setup_env.main(args) == 0
    assert len(commands) == 2
    for command in commands:
        assert command[0] == str(python)
        assert command[command.index("--constraint") + 1] == str(setup_env.CPU_LOCK)
        assert ("--no-index" in command) is offline
        assert command[command.index("--find-links") + 1] == str(wheels)
    assert "--requirement" not in commands[0]
    assert commands[1][commands[1].index("--requirement") + 1] == str(setup_env.CPU_LOCK)
    assert commands[1][-1] == f"{setup_env.PROJECT_ROOT}[{'dev,gpu' if gpu else 'dev'}]"
    assert verification == [(python, {"gpu": gpu})]


def test_missing_validation_lock_fails_before_install(tmp_path, monkeypatch, capsys):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    python = setup_env.venv_python(env)
    python.parent.mkdir(parents=True)
    python.touch()
    wheels.mkdir()
    monkeypatch.setattr(setup_env, "CPU_LOCK", tmp_path / "missing-lock.txt")
    monkeypatch.setattr(setup_env, "runtime_info", lambda _: {
        "version": [3, 12, 4], "bits": 64, "implementation": "cpython", "platform": "win32"})
    monkeypatch.setattr(setup_env, "run", lambda _: pytest.fail("must not install without the lock"))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels)]) == 1
    assert "validated dependency lock is missing" in capsys.readouterr().err


@pytest.fixture
def installed_project(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fixture-app"\nversion = "2.0"\n'
        '[project.scripts]\nfixture-app = "fixture.app:run"\n'
        'fixture-recorder = "fixture.recorder:run"\n', encoding="utf-8")
    monkeypatch.setattr(setup_env, "PROJECT_ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    monkeypatch.setattr(setup_env.sysconfig, "get_path", lambda _: str(scripts))
    suffix = ".exe" if setup_env.os.name == "nt" else ""
    for name in ("fixture-app", "fixture-recorder"):
        (scripts / f"{name}{suffix}").touch()
    distribution = SimpleNamespace(version="2.0", entry_points=[
        setup_env.metadata.EntryPoint("fixture-app", "fixture.app:run", "console_scripts"),
        setup_env.metadata.EntryPoint("fixture-recorder", "fixture.recorder:run", "console_scripts"),
    ])
    def find_distribution(name):
        assert name == "fixture-app"
        return distribution
    monkeypatch.setattr(setup_env.metadata, "distribution", find_distribution)
    return distribution, scripts, suffix


def test_installation_verifies_current_metadata_and_all_launchers(installed_project):
    setup_env.verify_installation()


@pytest.mark.parametrize("problem, message", [
    ("missing_distribution", "not installed"), ("wrong_version", "does not match"),
    ("missing_entrypoint", "does not point"), ("wrong_entrypoint", "does not point"),
    ("missing_launcher", "launcher is missing"),
])
def test_installation_rejects_missing_or_stale_project_install(installed_project, monkeypatch, problem, message):
    distribution, scripts, suffix = installed_project
    if problem == "missing_distribution":
        def missing(_):
            raise setup_env.metadata.PackageNotFoundError("fixture-app")
        monkeypatch.setattr(setup_env.metadata, "distribution", missing)
    elif problem == "wrong_version":
        distribution.version = "1.0"
    elif problem == "missing_entrypoint":
        distribution.entry_points.pop()
    elif problem == "wrong_entrypoint":
        distribution.entry_points[-1] = setup_env.metadata.EntryPoint(
            "fixture-recorder", "fixture.legacy:run", "console_scripts")
    else:
        (scripts / f"fixture-recorder{suffix}").unlink()
    with pytest.raises(ValueError, match=message):
        setup_env.verify_installation()


def test_verify_checks_installation_in_selected_interpreter(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(setup_env, "run", commands.append)
    python = setup_env.venv_python(tmp_path)
    setup_env.verify(python, gpu=True)
    assert all(command[0] == str(python) for command in commands)
    assert commands[0][-1] == "check"
    assert commands[1][-1] == "from setup_env import verify_installation; verify_installation()"
    assert "import cupy;" in commands[2][-1]


def test_nonempty_non_environment_directory_is_preserved(tmp_path, monkeypatch):
    env, wheels = tmp_path / "env", tmp_path / "wheels"
    env.mkdir()
    wheels.mkdir()
    (env / "notes.txt").write_text("keep me")
    monkeypatch.setattr(setup_env, "run", lambda _: pytest.fail("must not replace directory"))
    assert setup_env.main(["--venv", str(env), "--wheel-dir", str(wheels)]) == 1
    assert (env / "notes.txt").read_text() == "keep me"


def _wheel(directory, name, version, tag="py3-none-any"):
    stem = f"{name}-{version}"
    dist = stem + ".dist-info"
    entries = {
        f"{name}/__init__.py": f"VERSION = {version!r}\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\nRequires-Python: >=3.11\n",
        f"{dist}/WHEEL": f"Wheel-Version: 1.0\nGenerator: installer-test\nRoot-Is-Purelib: true\nTag: {tag}\n",
    }
    entries[f"{dist}/RECORD"] = "\n".join(f"{p},," for p in [*entries, f"{dist}/RECORD"])
    path = directory / f"{stem}-{tag}.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return path


def test_pip_installs_compatible_local_wheel_from_nested_folder_offline(tmp_path):
    root = tmp_path / "mixed wheels with spaces"
    nested = root / "other directory"
    nested.mkdir(parents=True)
    _wheel(nested, "temsim_setup_fixture", "1.0")
    _wheel(root, "temsim_setup_fixture", "9.0", "cp399-cp399-win_amd64")
    _wheel(root, "temsim_unrelated_fixture", "1.0")
    target = tmp_path / "installed"
    result = subprocess.run([sys.executable, "-m", "pip", "--disable-pip-version-check", "install",
        *setup_env.wheel_sources(root, offline=True), "--ignore-installed", "--target", str(target),
        "temsim_setup_fixture>=1"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target / "temsim_setup_fixture" / "__init__.py").read_text() == "VERSION = '1.0'\n"
    assert not (target / "temsim_unrelated_fixture").exists()
    assert "http" not in result.stdout
