"""Seal the completed high-angle evidence after final software/input checks."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path/"agent.md").is_file())
ARCHIVE = ROOT/"docs/development/two_condenser_10mrad_20261112"


def main():
    if (ARCHIVE/"artifact_manifest.json").exists():
        raise ValueError("Refusing to overwrite a sealed scientific archive")
    summary = json.loads((HERE/"validation-summary.json").read_text())
    catalog = ROOT/"configs/operating_modes/catalog.toml"
    if sha256(catalog.read_bytes()).hexdigest() != summary["catalog_sha256"]:
        raise ValueError("The final catalog differs from the software validation receipt")
    if summary["source_sampling_convergence"] != "NOT_CONVERGED":
        raise ValueError("The measured source-count limitation must remain explicit")
    subprocess.run([sys.executable, str(HERE/"write_readme.py")], check=True, cwd=ROOT)
    trials = {
        "initial-strong-1024": HERE/"initial-n1024-s010/trial-00",
        "initial-weak-1024": HERE/"initial-n1024-s010/trial-01",
        "initial-transmission-1024": HERE/"weak-n1024-s010/trial-00",
        "proposal-focus-2048": HERE/"focus-n2048-s010/trial-00",
        "primary-4096": HERE/"focused4096-s005",
        "step-4096": HERE/"final4096-s010",
        "source-reference-2048": HERE/"reference2048-s005",
    }
    command = [sys.executable, str(HERE/"curate_study.py"), "--archive", str(ARCHIVE),
               "--selected", "primary-4096", "--primary", "primary-4096"]
    for label, source in trials.items():
        command += ["--trial", f"{label}={source}"]
    for label in ("primary-4096", "step-4096", "source-reference-2048"):
        command += ["--focus-trial", label]
    for path in (
        ROOT/"scripts/study_two_condenser_nanoprobe.py", ROOT/"scripts/measure_probe_waist.py",
        ROOT/"scripts/audit_joint_probe_fields.py", HERE/"run_persistent_trials.py",
        HERE/"observer-before-fine-b0036f605d02.py", HERE/"observer-initial-ab71d3c9d09f.py",
        HERE/"audit_initial_source_tail.py", HERE/"audit_crossover_statistics.py",
        HERE/"audit_controlled_final.py", HERE/"validate_saved_inputs.py",
        HERE/"summarize_validation.py", HERE/"write_readme.py", Path(__file__),
    ):
        command += ["--driver", str(path)]
    names = (
        "API_AUDIT.md", "seed-regressions.xml", "final-mode-regressions.xml",
        "validation-summary.json", "saved-input-validation.json", "portable-focused4096-inputs.json",
        "initial-source-tail-audit.json", "final-source-tail-audit.json", "final-crossover-statistics.json",
        "final-controlled-comparison.json", "final-scene-count-diagnostic.json", "numerical-comparison.json",
        "final-upstream-controls.json",
    )
    for name in names:
        command += ["--supplement", f"{name}={HERE/name}"]
    # Preserve original failed batches and any target reruns exactly as the
    # final de-duplicated validation receipt records them.
    already_included = {(HERE/name).resolve() for name in names}
    already_included.add((HERE.parent/"two-condenser-20261112/explicit-aperture-regression.xml").resolve())
    for inventory in summary["junit"]:
        source = (ROOT/inventory["path"]).resolve()
        if source not in already_included:
            command += ["--supplement", f"regressions/{source.name}={source}"]
            already_included.add(source)
    command += ["--supplement", f"default-promotion/catalog.toml={catalog}",
                "--supplement", f"default-promotion/C2.toml={ROOT/'configs/instruments/column/C2.toml'}",
                "--supplement", f"regressions/explicit-aperture-regression.xml={HERE.parent/'two-condenser-20261112/explicit-aperture-regression.xml'}"]
    for path in sorted((HERE/"projector").rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            command += ["--supplement", f"projector/{path.relative_to(HERE/'projector').as_posix()}={path}"]
    subprocess.run(command, check=True, cwd=ROOT)
    manifest = json.loads((ARCHIVE/"artifact_manifest.json").read_text())
    for name, identity in manifest["files"].items():
        data = (ARCHIVE/name).read_bytes()
        if len(data) != identity["bytes"] or sha256(data).hexdigest() != identity["sha256"]:
            raise ValueError(f"Sealed artifact hash mismatch: {name}")
    print(json.dumps({"archive": str(ARCHIVE), "verified_registered_files": len(manifest["files"]),
        "registered_bytes": sum(row["bytes"] for row in manifest["files"].values()),
        "trials": len(trials), "source_sampling_convergence": "NOT_CONVERGED"}, indent=2))


if __name__ == "__main__":
    main()
