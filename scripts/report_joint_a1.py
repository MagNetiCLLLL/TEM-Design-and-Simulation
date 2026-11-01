"""Qualify and archive the measured A1 trim without replacing its input point."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from correct_joint_probe_a1 import (
    ARCHIVE, AXES, LIMITS, PARENT, RESPONSE_CASES, SAMPLE_ANGLE,
    apply_trim, at, configure, physical, sha, verified_focus,
)


def protected_changes(row, reference):
    c3_m = (row["C3_mm"]-reference["C3_mm"])*.001
    a5_m = (np.array(row["signed_A5_mm"])-reference["signed_A5_mm"])*.001
    return dict(C3_mm=c3_m*1000.,A5_vector_change_mm=(a5_m*1000.).tolist(),
                C3_edge_pm=abs(c3_m)*SAMPLE_ANGLE**3*1e12,
                A5_edge_pm=np.linalg.norm(a5_m)*SAMPLE_ANGLE**5*1e12)


def publish_verified_images():
    style = physical.read_json(physical.INPUTS/"baseline.json")["render"]
    receipts = {}
    for path in sorted(ARCHIVE.glob("*/result.json")):
        folder = path.parent
        rec = physical.read_json(path)
        checks = verified_focus(folder,rec)
        with np.load(folder/"current.npz") as data, TemporaryDirectory(dir=folder) as temporary:
            image_path = Path(temporary)/"incident-probe.png"
            physical.render_current(data,style,image_path)
            if sha(image_path) != rec["image_sha256"]:
                raise ValueError("The focus-verified image differs from its current artifact")
            image_path.replace(folder/image_path.name)
        receipts[folder.name] = dict(checks, input_digest=rec["input_digest"],
            raw_current_sha256=sha(folder/"current.npz"), image_sha256=rec["image_sha256"])
    physical._json(ARCHIVE/"image-verification.json",receipts)


def comparison_figure(folders, target, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig,axes = plt.subplots(len(folders),2,squeeze=False,figsize=(7.6,3.15*len(folders)),layout="constrained")
    style = physical.read_json(physical.INPUTS/"baseline.json")["render"]
    peak = style["color_max_nA"]
    for _,folder in folders:
        with np.load(folder/"current.npz") as data:
            peak = max(peak,float(data["current_a_yx"].max()*1e9))
    maximum = 50.*np.ceil(peak/50.)
    for i,(label,folder) in enumerate(folders):
        rec = physical.read_json(folder/"result.json")
        verified_focus(folder,rec)
        with np.load(folder/"current.npz") as current:
            edges = [current["x_edges_nm"][0],current["x_edges_nm"][-1],
                     current["y_edges_nm"][0],current["y_edges_nm"][-1]]
            frame = current["current_a_yx"]*1e9
        for j,hw in enumerate((2.56,.22)):
            ax = axes[i,j]
            im = ax.imshow(frame,origin="lower",extent=edges,cmap="magma",interpolation="nearest",
                           norm=SymLogNorm(linthresh=style["symlog_linthresh_nA"],vmin=0,vmax=maximum))
            ax.set(xlim=(-hw,hw),ylim=(-hw,hw),xlabel="x (nm)",ylabel="y (nm)",
                   title=f"{label}   |   {2*hw:.2f} nm field")
    bar = fig.colorbar(im,ax=axes,shrink=.63)
    ticks = sorted({v for v in (0,5,10,20,50,100,200,maximum) if v <= maximum})
    bar.set_ticks(ticks,labels=list(map(str,ticks)))
    bar.set_label("Current per 20 pm pixel (nA)")
    fig.suptitle(title)
    fig.savefig(target,dpi=155)
    plt.close(fig)


def export_package(folder):
    from temsim.working_point import WorkingPointCheckpoint
    configure()
    state,_,_,_ = physical.load_baseline(physical.INPUTS)
    rec = physical.read_json(folder/"result.json")
    apply_trim(state,rec["controls"])
    state.step_mm = rec["maximum_step_mm"]
    snapshot = physical.capture_instrument_snapshot(state)
    if snapshot.digest != rec["input_digest"]:
        raise ValueError("The trial does not restore exactly")
    cp = WorkingPointCheckpoint(snapshot,{},state.sample.upper_surface_z_mm,
        "inputs-only:"+snapshot.digest,dict(label="Focused 30 mrad A1 trim",
        package_kind="INSTRUMENT_INPUTS_ONLY",validation_status="INPUTS_ONLY",
        phase_status="NOT_COMPUTED",quality="NOT_COMPUTED",source_representation="classical tip particles"))
    target = folder/"a1-trim-inputs.temwp"
    if not target.exists():
        cp.write_package(target)
    restored = WorkingPointCheckpoint.read_package(target)
    if restored.digest != cp.digest or physical.capture_instrument_snapshot(restored.compatible_state()).digest != snapshot.digest:
        raise ValueError("Exported input package did not preserve the measured settings")
    physical._json(folder/"scan-input.json",dict(verified_focus(folder,rec),input_digest=snapshot.digest,
                   surface_sha256=sha(folder/"surface.npz"),current_sha256=sha(folder/"current.npz")))
    return dict(package_sha256=sha(target), input_digest=snapshot.digest,
                final_percent=[getattr(state.condenser_stigmator,k) for k in AXES])


def main():
    reference = at(PARENT)
    folders = [("Reference",PARENT),("Half trim",ARCHIVE/"a1-half"),("Full trim",ARCHIVE/"a1-corrected")]
    rows = [at(folder) for _,folder in folders]
    fine = at(ARCHIVE/"a1-corrected-fine")
    for row in [*rows,fine]:
        row["protected_changes"] = protected_changes(row,reference)
        row["focused_beam_pass"] = bool(row["focus_pass"]
            and row["current"]["surviving_rays"]==2000
            and abs(row["current"]["current_a"]-reference["current"]["current_a"])<1e-15
            and abs(row["current"]["alpha95_mrad"]-reference["current"]["alpha95_mrad"])<=LIMITS["maximum_angle_change_mrad"]
            and max(row["protected_changes"][k] for k in ("C3_edge_pm","A5_edge_pm"))<=LIMITS["maximum_protected_edge_change_pm"])
    final = rows[-1]
    def passes_null(row):
        return max(row["A1_nm"],np.linalg.norm(row["quarter_A1_nm"]))<=LIMITS["maximum_A1_nm"]
    consistency = {key:fine["current"][key]-final["current"][key] for key in
                   ("d95_nm","rms_radius_nm","alpha95_mrad","core_50pm_a","current_a")}
    consistency.update(waist_offset_nm=fine["waist_offset_nm"]-final["waist_offset_nm"],
                       A1_vector_nm=(np.array(fine["signed_A1_nm"])-final["signed_A1_nm"]).tolist())
    consistent = (abs(consistency["d95_nm"])<.001 and abs(consistency["rms_radius_nm"])<.001
                  and abs(consistency["alpha95_mrad"])<.001 and abs(consistency["waist_offset_nm"])<=.25
                  and abs(consistency["core_50pm_a"])<=5.001e-9
                  and np.linalg.norm(consistency["A1_vector_nm"])<.01)
    topology = physical.read_json(folders[-1][1]/"topology.json")
    passed = bool(topology["passed"] and topology["input_digest"]==final["input_digest"]
                  and all(row["focused_beam_pass"] for row in [*rows,fine]) and passes_null(final)
                  and passes_null(fine) and consistent)
    result = dict(passed=passed,limits=LIMITS,rounds=rows,fine=fine,step_comparison=consistency,
                  topology=topology)
    publish_verified_images()
    comparison_figure(folders,ARCHIVE/"a1-rounds.png","Sample-plane probe current  |  30 mrad A1 trim")
    comparison_figure([(name,ARCHIVE/name) for name,_,_ in RESPONSE_CASES],ARCHIVE/"response-probes.png",
                      "Sample-plane probe current  |  signed channel measurements")
    if passed:
        result["restore_verified"] = export_package(folders[-1][1])
    physical._json(ARCHIVE/"comparison.json",result)
    print("A1 qualification",passed,flush=True)
    if not passed:
        raise ValueError("Check the measured focus, protected channels and numerical repeat")


if __name__=="__main__":
    main()
