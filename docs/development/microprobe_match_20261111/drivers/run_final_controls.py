"""Run final full-source controls sequentially, retaining the numerical budget."""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
parent=HERE/"column/n2048-hp010-family/report.json"
deadline=time.monotonic()+600
while True:
    status=json.loads(parent.read_text())["status"]
    if status=="COMPLETE":break
    if status in {"ERROR","CANCELLED"} or time.monotonic()>deadline:
        raise RuntimeError("Parent family did not complete")
    time.sleep(5)
base=[str(ROOT/".venv/Scripts/python.exe"),str(HERE/"column/run_fine.py")]
jobs=[
    ["--gun-cache","outputs/microprobe-20261111/extractor/new-n1024",
     "--output",str(HERE/"column/n1024-hp010-nominal"),
     "--controls",str(HERE/"column/nominal-hp010-controls.json"),"--trials","1"],
    ["--gun-cache","outputs/microprobe-20261111/extractor/new-n2048",
     "--output",str(HERE/"column/n2048-final-hp010-endpoints"),
     "--controls",str(HERE/"column/final-low-hp010-controls.json"),
     "--inbox",str(HERE/"column/inbox-final-hp010"),"--trials","2"],
]
for options in jobs:
    print("Starting final source control:",options[options.index("--output")+1],flush=True)
    subprocess.run(base+options+["--step-mm","0.1","--topology-spacing-mm","0.5",
        "--topology","--threads","2","--budget-s","1200"],cwd=ROOT,check=True)
print("All final full-source controls completed",flush=True)
