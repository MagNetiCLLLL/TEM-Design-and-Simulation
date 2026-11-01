"""Compare verified zoom trajectories against the actual archived STEM illumination."""
import numpy as np

from audit_zoom_probe_focus import OUT, IMAGED_20, expanded_cases, physical


def inspect_case(folder):
    report = physical.read_json(folder/"result.json")
    obs = physical.read_json(folder/"through-focus.json")
    field = report["field_audit"]
    if (obs["status"] != "MEASURED" or not obs["membership_stable"]
            or len(obs["coarse_region"]["regions"]) != 1 or obs["coarse_region"]["unresolved"]
            or obs["input_digest"] != report["input_digest"] or not field["passed"]
            or obs["executed_field_digest"] != field["identities"]["potential_digest"]):
        raise ValueError(f"Invalid last-stage through-focus evidence in {folder}")
    with np.load(folder/"surface.npz") as rays,np.load(physical.INPUTS/"source-2000.npz") as src:
        for k in ("source_id","weights"):
            np.testing.assert_array_equal(rays[k],src[k])
        if len(src["source_id"])!=2000 or not (rays["alive"]&(rays["weights"]>0)).all():
            raise ValueError("Incomplete original specimen cohort")
        w = rays["weights"]
        r = np.hypot(rays["x_m"],rays["y_m"])
        core = report["original_current_a"]*w[r<=50e-12].sum()/w.sum()
        np.testing.assert_allclose(core,obs["at_specimen"]["core_50pm_a"],rtol=0,atol=1e-16)
    surface = obs["surface_z_mm"]
    profile = obs["profile"]
    def extrema(key,lowest):
        values = np.array([r[key] for r in profile])
        best = (np.min if lowest else np.max)(values)
        delta = 1e-16 if key=="core_50pm_a" else 1e-12
        selected = np.flatnonzero(abs(values-best)<=delta)
        assert 0 < selected.min() <= selected.max() < len(profile)-1
        return [round(profile[i]["offset_from_surface_nm"],4) for i in selected]
    fields = {s["source_id"]:s["potential_digest"] for s in field["field_ownership"]["sources"]}
    return dict(input_digest=report["input_digest"],source_digest=obs["source_identity"],
        controls=report["controls"],field_digests=fields,entrance_z_mm=surface,
        search_interval_mm=obs["search_interval_mm"],surface=obs["at_specimen"],
        crossover=obs["at_crossover"],minimum_D95=obs["d95_best"],peak_core=obs["core_best"],
        core_maxima_nm=extrema("core_50pm_a",False),d95_minima_nm=extrema("d95_nm",True),
        rms_minima_nm=extrema("rms_radius_nm",True),
        envelope_bracket_nm=((np.array(obs["waist"]["bracket_mm"])-surface)*1e6).tolist(),
        directional_crossing_brackets_nm=[((np.array(t["transition_brackets_mm"])-surface)*1e6).tolist()
            for t in obs["waist"]["directional_sign_transitions"]],
        sample_core_fraction_of_observed_peak=core/obs["core_best"]["core_50pm_a"])


def matched_pair(a,b,allowed):
    if a["source_digest"] != b["source_digest"] or a["entrance_z_mm"] != b["entrance_z_mm"]:
        raise ValueError("Mismatched source or sample plane")
    fields,other = a["field_digests"],b["field_digests"]
    changed = {k for k in fields if fields[k]!=other[k]}
    if fields.keys()!=other.keys() or changed!=set(allowed):
        raise ValueError(f"Unexpected change to physical field providers {changed}")
    return dict(changed_fields=sorted(changed),
        crossover_displacement_nm=(b["crossover"]["z_mm"]-a["crossover"]["z_mm"])*1e6,
        minimum_D95_displacement_nm=(b["minimum_D95"]["z_mm"]-a["minimum_D95"]["z_mm"])*1e6,
        example_core_peak_displacement_nm=(b["peak_core"]["z_mm"]-a["peak_core"]["z_mm"])*1e6)


def plot(cases,dense):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(12.2,7.2),layout="constrained")
    grid = fig.add_gridspec(2,6,height_ratios=(1.,1.06))
    ax = fig.add_subplot(grid[0,:3])
    names = ("old-30","zoom-only","imaged-20")
    styles = [("o","#087997","2-D waist"),("s","#66599a","D95 minimum"),("D","#b7541d","Core maximum")]
    for j,k in enumerate(names):
        row = cases[k]
        offsets = ([row["crossover"]["offset_from_surface_nm"]],row["d95_minima_nm"],row["core_maxima_nm"])
        for vals,(marker,color,_) in zip(offsets,styles):
            ax.scatter(vals,[2-j]*len(vals),s=55,color=color,marker=marker,zorder=3)
    ax.axvline(0,color="0.35",ls="--",lw=1)
    ax.set(yticks=[2,1,0],yticklabels=["30 mrad reference","20 mrad, C2/C3 only","20 mrad, imaged"],
           xlim=(-35,35),ylim=(-.45,2.5),xlabel="Axial offset from specimen entrance (nm)",
           title="Which focus moved?")
    ax.grid(axis="x",alpha=.15)
    ax.legend(handles=[Line2D([],[],marker=m,color=c,ls="",label=l) for m,c,l in styles],
              loc="lower center",bbox_to_anchor=(.5,-.43),ncols=3,fontsize=8,frameon=False)
    prof = fig.add_subplot(grid[0,3:])
    with np.load(OUT/"imaged-20-dense/axial-profile.npz") as f:
        prof.plot(f["offset_nm"],f["core_50pm_a"]*1e6,color=styles[2][1],lw=1.6)
    prof.axvline(0,ls="--",lw=1,color="0.35")
    for point,marker,color in [(dense["example_core_plane"],"D",styles[2][1]),
                               (dense["surface"],"o","#323c46"),(dense["minimum_d95"],"s",styles[1][1])]:
        prof.scatter(point["offset_from_surface_nm"],point["core_50pm_a"]*1e6,marker=marker,color=color,zorder=4)
    prof.set(xlabel="Axial offset from specimen entrance (nm)",ylabel="Current in 50 pm radius (µA)",
             title="Through the 20 mrad probe used for the Si scan",xlim=(-40,40),ylim=(0,4.7))
    prof.grid(alpha=.15)
    axes = []
    maps = []
    edges = np.arange(-20,21)*.02
    with np.load(OUT/"imaged-20-dense/planes.npz") as f:
        for key in ("core","specimen","d95"):
            x,y = f[key+"_xy_nm"].T
            maps.append(np.histogram2d(y,x,bins=(edges,edges),weights=f["particle_current_a"]*1e9)[0])
    norm = SymLogNorm(linthresh=5.,vmin=0.,vmax=np.ceil(max(m.max() for m in maps)/50)*50)
    for j,(frame,row,name) in enumerate(zip(maps,[dense["example_core_plane"],dense["surface"],dense["minimum_d95"]],
                                     ["Peak core","Specimen entrance","Minimum D95"])):
        axis = fig.add_subplot(grid[1,2*j:2*j+2])
        axes.append(axis)
        im=axis.imshow(frame,extent=[edges[0],edges[-1]]*2,origin="lower",cmap="magma",norm=norm,interpolation="nearest")
        axis.set(title=f"{name}  |  {row['offset_from_surface_nm']:+.2f} nm\n"
                       f"{row['core_50pm_a']*1e6:.3f} µA in core  ·  D95 {row['d95_nm']:.2f} nm",
                 xlabel="x (nm)",ylabel="y (nm)")
    bar=fig.colorbar(im,ax=axes,shrink=.83)
    ticks=[v for v in (0,5,10,20,50,100,200,500) if v<=norm.vmax]
    bar.set_ticks(ticks,labels=list(map(str,ticks)))
    bar.set_label("Current per 20 pm pixel (nA)")
    fig.suptitle("Final Mini Condenser / Objective focus  |  300 kV, 2000 original particles",fontsize=15)
    fig.savefig(OUT/"axial-focus-audit.png",dpi=160)
    plt.close(fig)


def main():
    cases={label:inspect_case(path) for label,path in expanded_cases().items()}
    dense=physical.read_json(OUT/"imaged-20-dense/measurement.json")
    scan=physical.read_json(IMAGED_20/"si110-scan/metrics.json")
    assert scan["input_digest"]==cases["imaged-20"]["input_digest"]==dense["input_digest"]
    condensers=["condenser_lens_2","condenser_lens_3"]
    matches = {label:matched_pair(cases[before],cases[after],changes) for label,before,after,changes in (
        ("historical_zoom","old-30","zoom-only",condensers),
        ("historical_objective_refocus","zoom-only","imaged-20",["objective_lens"]),
        ("current_a1_zoom","current-30","current-20",condensers),
        ("historical_core_control","zoom-only","core-control",["objective_lens"]))}
    physical._json(OUT/"comparison.json",dict(cases=cases,matched_changes=matches,
        imaged_probe=scan["probe"],zoom_scan_input_digest=scan["input_digest"],
        coordinate="Physical specimen upper face is zero. Positive is downstream into the specimen."))
    plot(cases,dense)
    for name,v in cases.items():
        print(name,round(v["crossover"]["offset_from_surface_nm"],2),v["d95_minima_nm"],
              v["core_maxima_nm"],round(v["surface"]["core_50pm_a"]*1e6,3))


if __name__=="__main__":
    main()
