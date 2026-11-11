"""Compare complete physical runs without changing their inputs or particles."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def read(relative):
    return json.loads((HERE/relative).read_text())


def compare(old,new):
    return {key:dict(before=old[key],after=new[key],relative_change=float(new[key]/old[key]-1.))
            for key in ("d95_um","radial_rms_um","alpha95_mrad","alpha_rms_mrad","current_a")}


coarse=read("column/n1024-first/report.json")
fine=read("column/n1024-step010-v2/report.json")
full=read("column/n2048-step010/report.json")
if any(item["status"]!="COMPLETE" for item in (coarse,fine,full)):
    raise ValueError("Comparison requires completed physical batches")
baseline=full["trials"][0]["surface"]
report=dict(scope="Native complete-source comparisons; no retracing or current renormalization",
    step_oneum=dict(source_count=1024,old_step_mm=.2,new_step_mm=.1,
        metrics=compare(coarse["trials"][0]["surface"],fine["trials"][0]["surface"])),
    sampling_oneum=dict(step_mm=.1,old_positive_count=1024,new_positive_count=2048,nested_source=False,
        metrics=compare(fine["trials"][0]["surface"],baseline)),
    combined_source_and_step=dict(old_count=1024,new_count=2048,old_step_mm=.2,new_step_mm=.1,
        low=compare(coarse["trials"][4]["surface"],full["trials"][1]["surface"]),
        middle=compare(coarse["trials"][1]["surface"],full["trials"][2]["surface"]),
        high=compare(coarse["trials"][5]["surface"],full["trials"][3]["surface"])),
    finite_source_steering=[])
for index,name in ((4,"shift_x_50nm"),(5,"shift_y_50nm")):
    surface=full["trials"][index]["surface"]
    displacement=(np.array(surface["centroid_xy_um"])-baseline["centroid_xy_um"])*1e3
    angular_change=np.array(surface["chief_slope_xy_mrad"])-baseline["chief_slope_xy_mrad"]
    target=np.array((50.,0.)) if index==4 else np.array((0.,50.))
    report["finite_source_steering"].append(dict(command=name,original_positive_count=2048,
        actual_displacement_nm=displacement.tolist(),target_displacement_nm=target.tolist(),
        displacement_error_nm=(displacement-target).tolist(),
        mechanical_mean_angle_change_mrad=angular_change.tolist(),
        mechanical_angle_change_norm_mrad=float(np.linalg.norm(angular_change)),
        width_and_angle_comparison=compare(baseline,surface),source_arrivals=surface["count"],
        current_a=surface["current_a"],beam_statistics_about_actual_weighted_chief=True))
(HERE/"comparisons.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf8")
print(json.dumps(report,indent=2),flush=True)
