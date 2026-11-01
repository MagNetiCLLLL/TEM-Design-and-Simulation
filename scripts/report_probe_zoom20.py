"""Compare source-weighted harmonics, fitted coefficients and measured zoom conjugates."""
import argparse

import numpy as np

from check_probe_zoom20 import ARCHIVE, PARENT, p
from scan_joint_probe_si110 import sha, verified_focus


def harmonics(folder):
    with np.load(folder/"current.npz") as f:
        xy = f["xy_nm"] @ np.array([1., 1.j])
        current = f["particle_current_a"]
    result = {}
    for label,centre in (("axis",0.),("centroid",np.average(xy,weights=current))):
        r = abs(xy-centre)
        angles = np.angle(xy-centre)
        bins = [(0.,.1),(.1,1.),(1.,np.inf)]
        shells = []
        for lo,hi in bins:
            mask = (r>lo)&(r<=hi)
            flux = current[mask].sum()
            moments = {}
            for m in (2,3,4,6):
                c = np.sum(current[mask]*np.exp(1.j*m*angles[mask])) / flux
                moments[str(m)] = dict(fraction=abs(c), real=c.real, imag=c.imag,
                    orientation_deg=np.rad2deg(np.angle(c))/m)
            shells.append(dict(inner_nm=lo, outer_nm=hi if np.isfinite(hi) else None,
                               count=int(mask.sum()),current_a=flux,moments=moments))
        result[label] = shells
    return result


def common_pupil(folder, cut=.020):
    # Preserve distinct training and holdout families. The cut is on geometric
    # pencil slopes only; no physical particles are removed from any image.
    with np.load(folder/"optical-pencil.npz") as f:
        full,_ = np.split(f["rays"],2)
    inside = np.linalg.norm(full[:,2:]-full[0,2:],axis=1) <= cut
    training_count = int(inside[1:193].sum())
    fit = p.geometric_coefficients(full[inside],training_count,degree=9)
    fit.update(train=training_count,holdout=int(inside[193:].sum()),cut_slope_rad=cut)
    return fit


def inspect_state(folder):
    report = p.read_json(folder/"result.json")
    fit = common_pupil(folder)
    quarter = report["fits"][1]
    alpha = report["at_specimen"]["alpha95_mrad"]*.001
    contributions = {}
    for name,order in (("A1",1),("A3",3),("C54",5),("C3",3),("C5",5)):
        c = np.array(fit["terms_m"][name])
        contributions[name] = dict(coefficient_m=c.tolist(),coefficient_magnitude_m=float(np.linalg.norm(c)),
            characteristic_edge_displacement_pm=float(np.linalg.norm(c)*alpha**order*1e12),
            quarter_pupil_coefficient_m=quarter["terms_m"][name])
    return dict(input_digest=report["input_digest"], source_digest=report["source_digest"],
                controls=report["controls"], fixed_pupil_terms=contributions, common_fit=fit,
                native_harmonics=harmonics(folder), source_surface=report["at_specimen"],
                crossover=report["at_crossover"], d95_best=report["d95_best"],
                core_best=report["core_best"], rms_best=report["rms_best"])


def compare(name,off):
    result = {}
    aimed = [("reference",PARENT,ARCHIVE/"baseline-hp-zero","baseline"),
             ("zoom",ARCHIVE/name,ARCHIVE/off,name)]
    for label,folder,zero_folder,chain_name in aimed:
        result[label] = row = inspect_state(folder)
        off_report = p.read_json(zero_folder/"result.json")
        on_report = p.read_json(folder/"result.json")
        def changed_fields(left,right):
            def hashes(report):
                return {r["source_id"]:r["potential_digest"] for r in report["field_audit"]["field_ownership"]["sources"]}
            a,b = hashes(left),hashes(right)
            assert a.keys() == b.keys()
            return sorted(k for k in a if a[k] != b[k])
        row["on_off_changed_fields"] = changed_fields(on_report,off_report)
        if row["on_off_changed_fields"] != ["probe_hp1_hexapole","probe_hp2_hexapole"]:
            raise ValueError("The field change is not confined to both main hexapoles")
        row["zoom_changed_fields"] = changed_fields(p.read_json(PARENT/"result.json"),on_report)
        if label == "zoom" and row["zoom_changed_fields"] != ["condenser_lens_2","condenser_lens_3","objective_lens"]:
            raise ValueError("The focused zoom changed a field outside C2, C3 and the shared Objective")
        for key in ("source_digest","source_hashes","maximum_step_mm"):
            if off_report[key] != on_report[key]:
                raise ValueError(f"Unmatched HP on/off pair: {key}")
        assert off_report["controls"]["hexapoles_zero"]
        for key in set(off_report["controls"]) | set(on_report["controls"]):
            if key != "hexapoles_zero" and off_report["controls"].get(key,0.) != on_report["controls"].get(key,0.):
                raise ValueError("Another drive changed in the hexapole on/off comparison")
        off_fit = common_pupil(zero_folder)
        def pair(on, muted):
            return dict(on_mm=on["terms_m"]["C3"][0]*1e3,off_mm=muted["terms_m"]["C3"][0]*1e3,
                on_minus_off_mm=(on["terms_m"]["C3"][0]-muted["terms_m"]["C3"][0])*1e3)
        row["hp_effect"] = dict(common_cut=pair(row["common_fit"],off_fit),
            full=pair(on_report["fits"][0],off_report["fits"][0]),
            quarter=pair(on_report["fits"][1],off_report["fits"][1]), off_common_fit=off_fit,
            off_survivors=off_report["at_specimen"]["surviving_rays"],off_input=off_report["input_digest"])
        row["focus_check"] = verified_focus(folder,on_report)
        chain = p.read_json(ARCHIVE/(chain_name+"-chain.json"))
        if chain["input_digest"] != row["input_digest"]:
            raise ValueError("Different input used for intermediate and specimen measurements")
        row["crossover_chain_mm"] = {key:[v["z_mm"] for v in val["regions"]] for key,val in chain["intervals"].items()}
        assert all(not val["unresolved"] for val in chain["intervals"].values())
    ref, zoom = result["reference"],result["zoom"]
    result["chain_shifts_mm"] = {key:[b-a for a,b in zip(values,zoom["crossover_chain_mm"][key],strict=True)]
                                 for key,values in ref["crossover_chain_mm"].items()}
    delta_ref = ref["hp_effect"]["common_cut"]["on_minus_off_mm"]
    delta_zoom = zoom["hp_effect"]["common_cut"]["on_minus_off_mm"]
    result["relative_change_in_hexapole_effect"] = (delta_zoom-delta_ref)/delta_ref
    p._json(ARCHIVE/"comparison.json",result)
    plot_probes(ARCHIVE/name)
    return result


def plot_probes(zoom_folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    fig, axes = plt.subplots(2,2,figsize=(7.9,7.1),layout="constrained")
    norm = SymLogNorm(linthresh=5.,vmin=0.,vmax=250.)
    for row,folder in enumerate((PARENT,zoom_folder)):
        rec = p.read_json(folder/"result.json")
        verified_focus(folder,rec)
        with np.load(folder/"current.npz") as f:
            frame = f["current_a_yx"]*1e9
            edges = [f["x_edges_nm"][0],f["x_edges_nm"][-1],f["y_edges_nm"][0],f["y_edges_nm"][-1]]
        for j,half_width in enumerate((2.56,.22)):
            ax = axes[row,j]
            im = ax.imshow(frame,extent=edges,origin="lower",interpolation="nearest",norm=norm,cmap="magma")
            ax.set(xlim=(-half_width,half_width),ylim=(-half_width,half_width),
                   xlabel="x (nm)",ylabel="y (nm)",
                   title=f'{rec["at_specimen"]["alpha95_mrad"]:.2f} mrad   |   {2*half_width:.2f} nm field')
    bar = fig.colorbar(im,ax=axes,shrink=.8,extend="max")
    bar.set_label("Absolute current per 20 pm pixel (nA)")
    ticks = [0,5,10,20,50,100,200]
    bar.set_ticks(ticks,labels=[str(t) for t in ticks])
    fig.suptitle("Incident current at the verified specimen entrance",fontsize=14)
    fig.savefig(ARCHIVE/"focused-probes.png",dpi=165)
    plt.close(fig)


def admit(name):
    folder = ARCHIVE/name
    report = p.read_json(folder/"result.json")
    evidence = verified_focus(folder,report)
    if not 19.8 <= report["at_specimen"]["alpha95_mrad"] <= 20.2:
        raise ValueError("The 20 mrad zoom is outside the 1% comparison band")
    p._json(folder/"scan-input.json",dict(evidence,input_digest=report["input_digest"],
            surface_sha256=sha(folder/"surface.npz"),current_sha256=sha(folder/"current.npz")))


def verify_delivery(name):
    from check_probe_zoom20 import configure, apply_drives, write_restoration_recipe
    from temsim.working_point import WorkingPointCheckpoint
    configure()
    folder = ARCHIVE/name
    ref = p.read_json(folder/"result.json")
    fine = p.read_json(ARCHIVE/(name+"-fine")/"result.json")
    assert ref["controls"] == fine["controls"] and ref["source_hashes"] == fine["source_hashes"]
    assert ref["maximum_step_mm"] == 2*fine["maximum_step_mm"]
    for case in (folder,ARCHIVE/(name+"-fine")):
        verified_focus(case,p.read_json(case/"result.json"))
    state,_,_,source = p.load_baseline(p.INPUTS)
    apply_drives(state,ref["controls"])
    state.step_mm = ref["maximum_step_mm"]
    snapshot = p.capture_instrument_snapshot(state)
    assert snapshot.digest == ref["input_digest"]
    package = WorkingPointCheckpoint(snapshot,{},state.sample.upper_surface_z_mm,
        "inputs-only:"+snapshot.digest,dict(label="Focused 20 mrad C2/C3 zoom comparison",
        package_kind="INSTRUMENT_INPUTS_ONLY",validation_status="INPUTS_ONLY",
        phase_status="NOT_COMPUTED",quality="NOT_COMPUTED",source_representation="classical tip particles"))
    target = folder/"zoom-inputs.temwp"
    if not target.exists():
        package.write_package(target)
    readback = WorkingPointCheckpoint.read_package(target)
    assert readback.digest == package.digest
    assert p.capture_instrument_snapshot(readback.compatible_state()).digest == snapshot.digest
    for path in ARCHIVE.glob("*/result.json"):
        with np.load(path.parent/"surface.npz") as f:
            for key in ("source_id","weights"):
                np.testing.assert_array_equal(f[key],source[key])
            assert f["alive"].all()
        write_restoration_recipe(path.parent)
    delta = {k:fine["at_specimen"][k]-ref["at_specimen"][k]
             for k in ("d95_nm","rms_radius_nm","alpha95_mrad","core_50pm_a","current_a")}
    delta["crossover_offset_nm"] = fine["at_crossover"]["offset_from_surface_nm"]-ref["at_crossover"]["offset_from_surface_nm"]
    limits = dict(d95_nm=.001,rms_radius_nm=.001,alpha95_mrad=.001,core_50pm_a=5e-9,
                  current_a=1e-15,crossover_offset_nm=.25)
    assert all(abs(delta[key]) <= limit for key,limit in limits.items())
    p._json(ARCHIVE/"verification.json",dict(package_sha256=sha(target),restored_input_digest=snapshot.digest,
            checked_trials=sorted(path.parent.name for path in ARCHIVE.glob("*/result.json")),
            original_source_ids_weights_and_survival_verified=True, fine_minus_coarse=delta,limits=limits))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("admit","compare","verify-delivery"))
    parser.add_argument("--name",default="zoom-focused")
    parser.add_argument("--off",default="zoom-hp-zero")
    args = parser.parse_args()
    if args.action == "admit":
        admit(args.name)
    elif args.action == "verify-delivery":
        verify_delivery(args.name)
    else:
        compare(args.name,args.off)
