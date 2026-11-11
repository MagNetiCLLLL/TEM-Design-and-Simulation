"""Keep rejected candidates and preparation failures explicit in the evidence."""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DEST=ROOT/"docs/development/microprobe_match_20261111"


def main():
    accepted={(entry["run"],entry["index"]) for entry in json.loads((DEST/"results.json").read_text())["family"]}
    candidates=[];failures=[]
    for path in sorted((HERE/"column").glob("*/report.json")):
        batch=json.loads(path.read_text())
        if batch["status"]=="ERROR":
            target=DEST/"preparation-failures"/path.parent.name/"report.json"
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            failures.append(dict(run=path.parent.name,error=batch.get("error"),receipt=str(target.relative_to(DEST)).replace("\\","/")))
        for case in sorted(path.parent.glob("trial-??/report.json")):
            row=json.loads(case.read_text());s=row["surface"]
            unresolved={key:value["actual_population"]["unresolved"] for key,value in row["topology"].items()
                        if value["actual_population"]["unresolved"]}
            candidates.append(dict(run=path.parent.name,index=row["index"],selected=(path.parent.name,row["index"]) in accepted,
                d95_um=s["d95_um"],alpha95_mrad=s["alpha95_mrad"],size_angle_in_domain=.75<=s["d95_um"]<=2.2 and s["alpha95_mrad"]<=.3,
                unresolved_actual_population=unresolved,original_positive_count=row["execution"]["emitted_positive_current_points"],
                source_identity=row["execution"]["source_identity_sha256"],input_digest=row["execution"]["input_digest"]))
    report=dict(scope="Final selection is explicit; rejected exploratory points and failed preparations are not qualification",candidates=candidates,preparation_failures=failures)
    (DEST/"candidate-status.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(f"Retained {len(candidates)} candidate statuses and {len(failures)} failed preparation receipts")


if __name__=="__main__":main()
