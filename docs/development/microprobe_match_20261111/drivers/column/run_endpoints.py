"""Refine the two final endpoints with the full unchanged original source."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("micro_fine_driver",HERE/"run_fine.py")
fine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fine)
driver=fine.driver


def endpoint_planes(state,spacing):
    nearby,intervals,planes=fine.match.full_planes(state,spacing)
    excitation={part.key:part.percent for part in state.lenses}
    references=[]
    for case in (HERE/"n1024-first").glob("trial-??/report.json"):
        # The saturated exploratory point intentionally missed the requested
        # parallel-input correlation and is not a waist-location seed.
        if case.parent.name=="trial-03":
            continue
        row=json.loads(case.read_text())
        distance=sum((row["controls"][key]["percent"]-excitation[key])**2
                     for key in ("condenser_lens_2","condenser_lens_3"))
        references.append((distance,row))
    reference=min(references,key=lambda item:item[0])[1]
    dense=[]
    for item in reference["topology_summary"].values():
        for center in item["actual_population"]["centers_mm"]:
            dense.append(np.arange(center-.75,center+.7501,.005))
    return nearby,intervals,np.unique(np.r_[planes,np.concatenate(dense)])


if __name__=="__main__":
    driver.observation_planes=endpoint_planes
    driver.focus_driver.set_optics=fine.set_controls
    out=Path(sys.argv[sys.argv.index("--output")+1])
    proof=out.parent/(out.name+"-wrapper.json")
    proof.write_text(json.dumps(dict(wrapper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        parent_sha256=sha256((HERE/"run_fine.py").read_bytes()).hexdigest(),
        scope="Full original population and native endpoint waist observations"),indent=2)+"\n",encoding="utf8")
    driver.main()
