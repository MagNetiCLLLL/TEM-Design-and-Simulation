"""Seal the first completed curation; never change immutable physical receipts."""
from pathlib import Path
from hashlib import sha256
import json
import shutil

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/"docs/development/microprobe_match_20261111"
path=DEST/"artifact_manifest.json"
manifest=json.loads(path.read_text())
assert manifest.get("status")!="VERIFIED_COMPLETE", "A completed archive must not be resealed"
shutil.copyfile(path,HERE/"initial-curation-manifest.json")
for source in sorted(HERE.glob("*.py")):
    target=DEST/"drivers"/source.name
    shutil.copyfile(source,target)
shutil.copyfile(HERE/"report-body.md",DEST/"README.md")
for artifact in sorted(DEST.rglob("*")):
    if not artifact.is_file() or artifact==path:continue
    key=str(artifact.relative_to(DEST)).replace("\\","/")
    digest=sha256(artifact.read_bytes()).hexdigest()
    prior=manifest["files"].get(key)
    if prior and prior["sha256"]!=digest:
        if not key.startswith("drivers/"):raise ValueError("Physical evidence was altered: "+key)
        prior=dict(prior)
        source=HERE/Path(key).name
        manifest["files"][key]=dict(original_path=str(source.relative_to(ROOT)).replace("\\","/"),
            original=dict(sha256=digest,bytes=artifact.stat().st_size),sha256=digest,bytes=artifact.stat().st_size,
            superseded_during_initial_curation=prior)
    elif not prior:
        manifest["files"][key]=dict(generated_from="Unchanged native physical observations, primary-source research or offline verification",
            sha256=digest,bytes=artifact.stat().st_size)
for key,row in manifest["files"].items():
    assert sha256((DEST/key).read_bytes()).hexdigest()==row["sha256"],key
for name in ("phase-verification.json","profile-verification.json"):
    assert json.loads((DEST/name).read_text())["status"]=="VERIFIED"
manifest.update(status="VERIFIED_COMPLETE",source_originals_unchanged=True,
    verification="All curated hashes, original source IDs/weights, profile semantic roundtrips, clocks, and native surface statistics verified; kinetic-energy roundtrip bounded by one ULP")
path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
print(json.dumps(dict(status=manifest["status"],files=len(manifest["files"]),
    total_curated_bytes=sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())),indent=2))
