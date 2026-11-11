"""Finish sequential evidence checks once the last physical batch completes."""
from pathlib import Path
from hashlib import sha256
import json
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
DEST=ROOT/"docs/development/microprobe_match_20261111"
PYTHON=str(ROOT/".venv/Scripts/python.exe")
report=HERE/"column/n2048-final-hp010-endpoints/report.json"
deadline=time.monotonic()+900
while True:
    status=json.loads(report.read_text())["status"]
    if status=="COMPLETE":break
    if status in {"ERROR","CANCELLED"} or time.monotonic()>deadline:
        raise RuntimeError("Final endpoint batch did not complete")
    time.sleep(5)


def run(name,*args):
    print("Evidence step:",name,flush=True)
    subprocess.run([PYTHON,str(HERE/name),*map(str,args)],cwd=ROOT,check=True)


run("audit_case.py",HERE/"column/n2048-final-hp010-endpoints/trial-00",
    HERE/"column/n2048-final-hp010-endpoints/trial-01",HERE/"column/n1024-hp010-nominal/trial-00")
run("final_comparisons.py")
run("archive_match.py")
shutil.copyfile(HERE/"report-body.md",DEST/"README.md")
run("extract_gun_exit.py")
run("verify_archived_family.py")
run("verify_phase.py")
run("finish_report.py")
run("curate_status.py")
manifest=json.loads((DEST/"artifact_manifest.json").read_text())
for path in sorted(DEST.rglob("*")):
    if not path.is_file() or path.name=="artifact_manifest.json":continue
    key=str(path.relative_to(DEST)).replace("\\","/")
    digest=sha256(path.read_bytes()).hexdigest()
    old=manifest["files"].get(key)
    if old and old["sha256"]!=digest:
        raise ValueError("An already archived artifact was altered: "+key)
    manifest["files"].setdefault(key,dict(generated_from="Complete unchanged physical observations and offline verification",
        sha256=digest,bytes=path.stat().st_size))
(DEST/"artifact_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
for key,row in manifest["files"].items():
    assert sha256((DEST/key).read_bytes()).hexdigest()==row["sha256"],key
print("VERIFIED all curated files, source identity, profile roundtrips and measured finite-source metrics",flush=True)
