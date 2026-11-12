"""Summarize bounded software checks and measured physical limitations."""
from pathlib import Path
from hashlib import sha256
import json
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
files = [HERE.parent/"two-condenser-20261112/explicit-aperture-regression.xml",
         HERE/"final-mode-regressions.xml", HERE/"final-mode-targeted.xml"]
cases, inventories = {}, []
for path in files:
    tree = ET.parse(path)
    rows = []
    for item in tree.iter("testcase"):
        key = (item.attrib.get("classname", ""), item.attrib["name"])
        status = next((name for name in ("failure", "error", "skipped") if item.find(name) is not None), "passed")
        cases[key] = status
        rows.append(dict(classname=key[0], name=key[1], status=status))
    inventories.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path.read_bytes()).hexdigest(), cases=rows))
assert all(status == "passed" for status in cases.values()), cases
trials = []
for label in ("focused4096-s005", "final4096-s010", "reference2048-s005"):
    data = json.loads((HERE/label/"report.json").read_text())
    trials.append(dict(trial=label, emitted_positive=data["execution"]["emitted_positive_count"],
        surviving_positive=data["surviving_positive_count"], effective=data["effective_count"],
        alpha95_mrad=data["alpha95_mrad"], d95_nm=data["d95_nm"], rms_radius_nm=data["rms_radius_nm"],
        alpha95_requirement=data["alpha95_requirement"], direct_surface_focus=data["direct_surface_focus"],
        implementation=data["execution"]["implementation"]))
summary = dict(current_unique_test_counts={"passed": len(cases)}, junit=inventories,
    physical_trials=trials, saved_inputs=json.loads((HERE/"saved-input-validation.json").read_text()),
    source_sampling_convergence="NOT_CONVERGED",
    focus_scope="Fixed 4096-particle D95 surface placement checked at 0.05/0.1 mm; 2048-particle reference fails surface placement by 16 nm. No continuous-source nanometre error bound.",
    calculation_scope="Classical particles only. No coherent execution. Static/mocked tests and zero-current projector pencils do not qualify a finite source.",
    source_weights="Original IDs and absolute weights retained; C2 interception is physical current loss.",
    catalog_sha256=sha256((ROOT/"configs/operating_modes/catalog.toml").read_bytes()).hexdigest(),
    regression_follow_up="The broad batch retained one obsolete C2 Microprobe-seed expectation. The same testcase was corrected to require side-effect-free rejection without a C2 Microprobe calibration; the targeted repeat supersedes it. No solver or physical control changed.",
    additional_checks={"git_diff_check": "PASS", "driver_audit_waist_compile": "PASS"},
    observer_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/"validation-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n", encoding="utf-8")
print(json.dumps({"passed": len(cases), "files": [str(path) for path in files]}))
