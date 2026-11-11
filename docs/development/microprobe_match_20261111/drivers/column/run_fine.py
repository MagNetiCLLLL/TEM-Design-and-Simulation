"""Resolve all measured Microprobe waists and optional physical AC commands."""
from pathlib import Path
from hashlib import sha256
import importlib.util
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("micro_match_driver",HERE/"run_match.py")
match=importlib.util.module_from_spec(spec)
spec.loader.exec_module(match)
driver=match.driver
original_set=driver.focus_driver.set_optics


def fine_planes(state,spacing):
    nearby,intervals,planes=match.full_planes(state,spacing)
    references=[]
    excitation={part.key:part.percent for part in state.lenses}
    for case in (HERE/"n1024-first").glob("trial-??/report.json"):
        row=json.loads(case.read_text())
        difference=sum((row["controls"][key]["percent"]-excitation[key])**2
                       for key in ("condenser_lens_2","condenser_lens_3"))
        references.append((difference,row))
    if not references:
        raise ValueError("A complete coarse full-source survey is required first")
    reference=min(references,key=lambda item:item[0])[1]
    dense=[]
    for label,populations in reference["topology_summary"].items():
        for center in populations["actual_population"]["centers_mm"]:
            dense.append(np.arange(center-.75,center+.7501,.005))
    return nearby,intervals,np.unique(np.r_[planes,np.concatenate(dense)])


def set_controls(state,controls):
    names={"ac_deflector","probe_dp12_scan_deflector","beam_deflector","condenser_deflector"}
    ordinary={key:fields for key,fields in controls.items() if key not in names}
    changes=original_set(state,ordinary)
    parts={part.key:part for part in (*state.deflectors,*state.corrector_elements)}
    allowed={"ac_deflector":{"kick_x_mrad","kick_y_mrad"},
        "probe_dp12_scan_deflector":{"alignment_x_mrad","alignment_y_mrad"},
        "beam_deflector":{"upper_x_mrad","upper_y_mrad","lower_x_mrad","lower_y_mrad"},
        "condenser_deflector":{"upper_x_mrad","upper_y_mrad","lower_x_mrad","lower_y_mrad"}}
    for key,fields in controls.items():
        if key not in names:
            continue
        item=parts[key]
        for field,value in fields.items():
            if (field not in allowed[key] or type(value) not in (float,int)
                    or not math.isfinite(value) or abs(value)>item.maximum_kick_mrad):
                raise ValueError("Invalid installed physical steering drive")
            before=float(getattr(item,field))
            setattr(item,field,float(value))
            changes.append(dict(key=key,field=field,before=before,after=float(value)))
        if hasattr(item,"validate"):
            item.validate()
    state.ac_deflector.validate_coil_limits(parts["probe_dp12_scan_deflector"].coil_bias_mrad(state.ac_deflector))
    return changes


if __name__=="__main__":
    driver.observation_planes=fine_planes
    driver.focus_driver.set_optics=set_controls
    if "--output" in sys.argv:
        out=Path(sys.argv[sys.argv.index("--output")+1])
        proof=out.parent/(out.name+"-wrapper.json")
        proof.parent.mkdir(parents=True,exist_ok=True)
        proof.write_text(json.dumps(dict(wrapper_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
            parent_sha256=sha256((HERE/"run_match.py").read_bytes()).hexdigest(),
            source="Verified original gun source; no filtering or current normalization",
            scope="Full-source native waist refinement and shared physical coil commands"),indent=2)+"\n",encoding="utf8")
    driver.main()
