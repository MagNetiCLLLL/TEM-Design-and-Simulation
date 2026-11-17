"""Bounded input, field, CUDA-operator and GUI checks; no physical trial.

Run with the project virtual environment. Source emission and Gun transport
are forbidden in every selected test. CUDA tests exercise field operators on
synthetic arrays; they do not qualify a finite-source probe or a scan.
"""
from pathlib import Path
from hashlib import sha256
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TESTS = (
    "test_analytic_rectangle_coil.py",
    "test_compiled_transverse_coil.py",
    "test_cuda_coil_potential.py",
    "test_cuda_gun_step.py",
    "test_radial_multipole_symmetry.py",
    "test_unified_field_execution_identity.py",
    "test_unified_field_products.py",
    "test_particle_wave_covariance_admission.py",
    "test_ideal_current_segments.py",
    "test_ideal_current_segments_gui.py",
    "test_ideal_current_segment_plumbing.py",
    "test_ideal_transverse_segments.py",
    "test_mixed_stem_model_admission.py",
    "test_mixed_stem_gui.py",
)


class SourceFreePolicy:
    @pytest.fixture(autouse=True)
    def forbid_physical_source(self, monkeypatch):
        def forbidden(*_args, **_kwargs):
            pytest.fail("This validation must not emit or transport a finite source")

        monkeypatch.setattr("temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit", forbidden)
        monkeypatch.setattr("temsim.optics.electron_gun.source.trace_source_to_exit", forbidden)


def summarize(output):
    suites = list(ET.parse(output).getroot().iter("testsuite"))
    counts = {name: sum(int(suite.get(name, "0")) for suite in suites)
              for name in ("tests", "failures", "errors", "skipped")}
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
               for path in sorted((ROOT / "src/temsim").rglob("*.py"))}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, check=False)
        gpu_description = gpu.stdout.strip() if gpu.returncode == 0 else None
    except OSError:
        gpu_description = None
    baseline = "71566c5dc97733e55a2d8b30c31af4d1a74f19fe"
    configuration = subprocess.run(["git", "diff", "--quiet", baseline, "--", "configs"], cwd=ROOT, check=False)
    receipt = dict(schema="ideal-current-fit-sourcefree-validation-v1", counts=counts,
        status="PASSED" if counts["failures"] == counts["errors"] == 0 else "FAILED",
        scope="Input, field, actual CUDA operators and offscreen GUI only; no finite source, full-column qualification or coherent propagation",
        test_files=list(TESTS), python=sys.version,
        source_manifest_sha256=sha256(json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        source_file_count=len(sources), junit_sha256=sha256(output.read_bytes()).hexdigest(),
        observed_gpu=gpu_description, configuration_baseline=baseline,
        unchanged_default_configuration_verified=configuration.returncode == 0,
        intermediate_attempts="Earlier collection during live editing and HP admission failures were corrected; final results are not summed with previous runs")
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.chdir(ROOT)
    if len(sys.argv) > 1 and sys.argv[1] == "--summarize":
        summarize(Path(sys.argv[2]))
        raise SystemExit(0)
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "validation" / "field-tests.xml"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pytest.main([
        *(str(ROOT / "tests" / name) for name in TESTS),
        "-q", "--tb=short", f"--junitxml={output}",
    ], plugins=[SourceFreePolicy()])
    if output.is_file():
        summarize(output)
    raise SystemExit(result)
