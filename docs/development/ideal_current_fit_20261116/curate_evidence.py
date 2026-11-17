"""Preserve selected completed calculations without publishing particle caches.

This only copies existing evidence. It does not rerun or qualify a calculation.
The original executed scripts and results remain in the output directory.
"""
from hashlib import sha256
import gzip
import json
from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "outputs/ideal-coil-si110-scan-20261116"


def preserve(source, target=None):
    source = SOURCE / source
    target = HERE / (target or source.relative_to(SOURCE))
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".json" and source.stat().st_size > 500000:
        target.with_suffix(".json.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        # Only remove a generated plaintext copy in this exact evidence folder.
        assert target.resolve().is_relative_to(HERE.resolve())
        if target.is_file():
            target.unlink()
    else:
        shutil.copy2(source, target)
    return target


def main():
    for name in ("captured-field-coefficients.json", "candidate-currents.csv",
                 "results.json", "README.md", "all-lens-field-comparison.png"):
        preserve(Path("round-lens-field-match") / name)
    for name in ("fit.py", "executed-fit-v1.py", "joint.py", "audit_precision.py",
                 "report.py", "README.md", "results.json", "summary.csv",
                 "candidate-library.json", "joint-results.json",
                 "joint-candidate-library.json", "precision-audit.json",
                 "profiles.npz", "joint-profiles.npz", "per-owner-field-comparison.png",
                 "joint-field-comparison.png"):
        preserve(Path("fixed-envelope-segmented-current/round-study") / name)
    for name in ("fixed-envelope-reachability-evidence.json",
                 "fixed-envelope-two-current-study.json",
                 "fixed-envelope-two-current-comparison.png"):
        preserve(Path("hp-fixed-envelope-v1") / name)
    for name in ("README.md", "final-conclusion.json", "metrics.json", "recipes.json", "candidate-v2.json",
                 "design_transfer.py", "design_transfer_v2.py", "finalize_v2.py"):
        preserve(Path("fixed-envelope-segmented-current/round-transfer") / name)
    for study in ("round-study", "round-transfer"):
        source = Path("fixed-envelope-segmented-current") / study / "checksums.json"
        preserve(source, source.with_name("executed-checksums.json"))
    for name in ("entrance-summary.json", "input-snapshot.json", "backend-evidence.json"):
        preserve(Path("mixed-run-v3") / name)
    preserve("historical-gaussian-target-provenance.json")
    preserve("validation-exclusions.json")
    # Only the source-root discovery changes in this portable copy. Preserve
    # executed-fit-v1.py byte for byte as historical execution evidence.
    fit = HERE / "fixed-envelope-segmented-current/round-study/fit.py"
    text = fit.read_text(encoding="utf-8")
    text = text.replace("PROJECT = HERE.parents[3]",
        'PROJECT = next(parent for parent in HERE.parents if (parent / "src/temsim").is_dir())')
    fit.write_text(text, encoding="utf-8")
    transfer = HERE / "fixed-envelope-segmented-current/round-transfer/design_transfer.py"
    text = transfer.read_text(encoding="utf-8").replace("PROJECT=HERE.parents[3]",
        'PROJECT=next(parent for parent in HERE.parents if (parent/"src/temsim").is_dir())')
    transfer.write_text(text, encoding="utf-8")
    readme = transfer.parent / "README.md"
    text = readme.read_text(encoding="utf-8").replace(
        "`checksums.json` covers every authored script and serialized result.",
        "`executed-checksums.json` preserves the original complete output manifest. "
        "The top-level `checksums.json` covers the curated files; only source-root "
        "discovery is changed in the portable script copies.")
    readme.write_text(text, encoding="utf-8")
    manifest = {str(path.relative_to(HERE)).replace("\\", "/"): sha256(path.read_bytes()).hexdigest()
                for path in HERE.rglob("*") if path.is_file()
                and "__pycache__" not in path.parts and path.name != "checksums.json"}
    (HERE / "checksums.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
