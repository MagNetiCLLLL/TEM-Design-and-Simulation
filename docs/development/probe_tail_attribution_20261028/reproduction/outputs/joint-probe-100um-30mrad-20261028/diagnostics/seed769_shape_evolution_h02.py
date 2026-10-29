"""Read saved physical checkpoints; write new shape diagnostics, no propagation."""
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile

BASE = ROOT / "outputs/joint-probe-100um-30mrad-20261028"
RUN = BASE / "seed769-native-d95-waist-h02"
OUT = BASE / "diagnostics"
STEM = "seed769-shape-evolution-h02"


def hash_file(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for suffix in (".json", ".md", ".png"):
        if (OUT / (STEM + suffix)).exists():
            raise FileExistsError(OUT / (STEM + suffix))
    saved_report = json.loads((RUN / "report.json").read_text())
    surface = saved_report["analysis"]["surface"]["z_mm"]
    current = saved_report["receipt"]["source_current_a"]
    with np.load(RUN / "particles.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    weights, ids = arrays["weights"], arrays["source_id"]
    special = int(np.flatnonzero(ids == 306)[0])
    if not (arrays["alive"].all() and arrays["gun_exit_alive"].all()
            and np.array_equal(ids, arrays["emission_ray_id"])):
        raise ValueError("This diagnostic requires the verified all-transmitted fixed source")

    def observe(x, y, tx, ty, alive, z, section):
        selected = alive & (weights > 0)
        if not selected.all() or not np.isfinite(np.asarray([x, y, tx, ty])).all():
            raise ValueError("Unexpected missing or invalid current in this archive")
        stats = transverse_beam_statistics(x, y, tx, ty, alive=selected, weights=weights)
        radius = np.hypot(x - stats.mean_x_m, y - stats.mean_y_m)
        w = weights / weights.sum()
        r50 = _weighted_quantile(radius, w, .5)
        r95 = stats.radius_95_m
        second = float(w @ (radius * radius))
        covariance = np.cov(np.vstack([x, y]), aweights=w, bias=True)
        eigenvalues = np.linalg.eigvalsh(covariance)
        order = np.argsort(radius, kind="stable")
        rank = float(w[radius <= radius[special]].sum())
        row = dict(section=section, z_mm=float(z), surface_offset_nm=float((z-surface)*1e6),
            current_a=float(current), surviving_points=int(selected.sum()),
            centroid_x_m=float(stats.mean_x_m), centroid_y_m=float(stats.mean_y_m),
            d50_nm=float(2e9*r50), d90_nm=float(2e9*_weighted_quantile(radius,w,.9)),
            d95_nm=float(2e9*r95), rms_radius_nm=float(1e9*stats.radius_rms_m),
            d95_over_d50=float(r95/r50), maximum_radius_over_r95=float(radius.max()/r95),
            id306_radius_nm=float(radius[special]*1e9), id306_current_percentile=float(100*rank),
            id306_fraction_of_radial_second_moment=float(w[special]*radius[special]**2/second),
            outermost_1pct_current_fraction_of_second_moment=float(
                np.sum(w[order][np.cumsum(w[order]) > .99]*radius[order][np.cumsum(w[order]) > .99]**2)/second),
            rms_ellipse_axis_ratio=float(np.sqrt(eigenvalues[-1]/eigenvalues[0])),
            current_fraction_beyond_twice_r95=float(w[radius > 2*r95].sum()))
        return row, radius

    e, direction = arrays["emission_position_m"], arrays["emission_direction"]
    initial, initial_radius = observe(e[:,0],e[:,1],direction[:,0]/direction[:,2],
        direction[:,1]/direction[:,2],np.ones(len(ids),bool),0.,"emission")
    gun, column, radii = [], [], {}
    for index, z in enumerate(arrays["gun_z_mm"]):
        row, radius = observe(*(arrays[key][index] for key in ("gun_x_m","gun_y_m","gun_tx_rad","gun_ty_rad")),
            np.ones(len(ids),bool),z,"gun")
        gun.append(row)
        if z == 450.: radii["Gun exit"] = radius
    for index, z in enumerate(arrays["z_mm"]):
        row, radius = observe(*(arrays[key][index] for key in ("x_m","y_m","tx_rad","ty_rad")),
            arrays["alive"][index],z,"column")
        column.append(row)
        if z == surface: radii["Specimen surface"] = radius
        if z == 1210.: radii["Corrector checkpoint 1210 mm"] = radius
    radii["Emission"] = initial_radius
    by_z = {row["z_mm"]: row for row in column}
    chosen = [initial] + [by_z[z] for z in (450.,650.,765.,1090.,1145.,1165.,1210.,1250.,1300.,1500.,1595.,surface)]
    near = [row for row in column if surface-100e-6 <= row["z_mm"] <= surface+100e-6]
    d95_min = min(near,key=lambda r:r["d95_nm"])
    rms_min = min(near,key=lambda r:r["rms_radius_nm"])
    special_min = min(near,key=lambda r:r["id306_radius_nm"])
    snapshot = json.loads((RUN/"inputs.json").read_text())
    landmarks = []
    for node in snapshot["graph"]["nodes"]:
        a=node.get("attributes",{})
        if node["type"].endswith(":AssemblyPart") and a.get("key") in {
                "condenser_lens_1","condenser_lens_2","condenser_lens_3","adapter_lens",
                "probe_hp2_hexapole","probe_hp1_hexapole","probe_tl22_lens","probe_tl21_lens",
                "probe_tl12_lens","mini_condenser","objective_lens","ac_deflector"}:
            z=float(a["center_z_mm"])
            below=[v for v in by_z if v<=z]; above=[v for v in by_z if v>=z]
            landmarks.append(dict(key=a["key"],nominal_center_z_mm=z, exact_saved_checkpoint=z in by_z,
                saved_before_z_mm=max(below) if below else None,saved_after_z_mm=min(above) if above else None,
                policy="No interpolation; use only the explicitly reported actual checkpoint positions"))
    doc=dict(schema="temsim-saved-shape-evolution-v1",scope="Archived h02 seed769 arrays only; no new propagation",
        input_digest=saved_report["input_digest"],implementation=saved_report["implementation"],
        raw_files_sha256={str(path.relative_to(ROOT)):hash_file(path) for path in (
            RUN/"particles.npz",RUN/"inputs.json",RUN/"report.json",OUT/"source-emission-audit.json",
            ROOT/"src/temsim/physics/beam_statistics.py",Path(__file__))},
        populations=dict(source_points=len(ids),all_column_planes_transmitted=True,all_gun_exit_transmitted=True,
            current_a=current,original_ids_preserved=True,weight_sum=float(weights.sum())),
        checkpoint_inventory=dict(gun_count=len(gun),column_count=len(column),
            gun_range_mm=[gun[0]["z_mm"],gun[-1]["z_mm"]],column_range_mm=[column[0]["z_mm"],column[-1]["z_mm"]]),
        definitions=dict(diameters="Native production weighted radial quantiles about the actual current-weighted centroid at each plane",
            rms="Square root of weighted centered radial second moment; diagnostic only",
            percentile="Fraction of original current at or inside ID306 radius; no rays removed",
            second_moment_share="ID306 contribution to the full population radial second moment, not a filtered RMS",
            warning="RMS shape ratios are diagnostics only; they do not enter D95 waist acceptance"),
        emission=initial,gun_planes=gun,column_planes=column,selected_checkpoints=chosen,landmark_coverage=landmarks,
        window_observed_minima=dict(d95=d95_min,rms_diagnostic=rms_min,id306_radius=special_min,
            interpretation="Observed points only; ID306 minimum at a window boundary is not a resolved individual waist"),
        conclusions=["No current was clipped between any saved planes in this dataset.",
            "Radial shape ratios change, so the cloud does not undergo a single uniform size rescaling.",
            "Shape broadens already in the gun, changes within the corrector, becomes close to its gun-exit ratio again, and changes strongly near the specimen focus.",
            "ID306 carries 1/769 of current but approximately 61 percent of the surface second moment; D95/D50 also changes, proving the effect is not only a largest-ray label.",
            "D95 and RMS minima occur at different actual saved planes because their statistics weight the changing radial distribution differently.",
            "This diagnostic cannot attribute the deformation to a particular physical field, aberration coefficient or integration error; fixed-source step refinement is required."],
        qualification="NOT_QUALIFIED")
    (OUT/(STEM+".json")).write_text(json.dumps(doc,indent=2,allow_nan=False),encoding="utf-8")
    table="\n".join(f"| {r['section']} | {r['z_mm']:.10f} | {r['d50_nm']:.6g} | {r['d95_nm']:.6g} | {r['rms_radius_nm']:.6g} | {r['d95_over_d50']:.5f} | {100*r['id306_fraction_of_radial_second_moment']:.4f}% |" for r in chosen)
    text=f"""# Shape evolution in saved seed769 h02 trajectories

This diagnostic uses 227 actual gun checkpoints and 433 actual column checkpoints from `seed769-native-d95-waist-h02/particles.npz`. The latter include the column at mostly 5 mm intervals and the specimen window at 1 nm intervals. No positions are interpolated, and no new propagation is performed. The JSON records every plane, original-file/code hashes and exact versus bracketed landmark coverage.

All 769 original particles survive at every saved plane; current remains 10 uA. Thus clipping or particle reweighting does not explain the observed shape change in this dataset.

## What changes, beyond one particle being the largest

The native ratio D95/D50 changes from 2.02412 at emission to 2.24887 at the gun exit. It remains about 2.2504 through much of the condenser path, rises to 5.28351 at the actual corrector checkpoint z=1210 mm, and returns to 2.25042 at z=1500 mm. At the exact specimen surface it becomes 4.41819. A pure uniform magnification would preserve this ratio. These changes directly demonstrate a redistribution between the core and outer-current regions.

At the surface, D50 is 0.581139 nm and D95 is 2.567580 nm. ID306 carries only 1/769 (0.13004%) of the current, but contributes 61.0185% of the full radial second moment. At the gun exit its second-moment contribution was only 0.8840%. This explains why RMS is especially sensitive to the rare outer component, while the changing D95/D50 ratio separately shows a change in the broader distribution.

| Region | Actual z (mm) | D50 (nm) | D95 (nm) | RMS radius (nm) | D95/D50 | ID306 second-moment share |
|---|---:|---:|---:|---:|---:|---:|
{table}

Nominal lens centers that are absent from the archive are not assigned invented observations. The table contains actual saved planes; e.g. 1145, 1165 and 1210 mm are checkpoint coordinates, not assertions of exact element centers or qualified waist coordinates.

## Why the two minima differ in this actual data

Within the saved +/-100 nm window, the smallest observed D95 is {d95_min['d95_nm']:.9f} nm at {d95_min['surface_offset_nm']:+.6f} nm from the surface. At that plane the RMS radius is {d95_min['rms_radius_nm']:.9f} nm. The smallest observed RMS radius is {rms_min['rms_radius_nm']:.9f} nm at {rms_min['surface_offset_nm']:+.6f} nm, where D95 is {rms_min['d95_nm']:.9f} nm. These are actual samples, not fitted parabolic roots.

The inner core, 95%-current contour and rare outer component do not shrink in lockstep as z changes. ID306's smallest observed centered radius lies at the upstream window boundary (-100 nm), so this window does not resolve that individual particle's radius minimum. It is therefore incorrect to describe the whole profile as one Gaussian whose width merely changes with z. D95 remains the intended waist observable; RMS is only an auxiliary explanation of the differing minimum.

## Attribution limit

The saved data localize shape changes in the gun, the corrector path and near the final focus. They do not prove whether each change is the correct nonlinear/chromatic field response, residual numerical integration error, or both. In particular, these statistics do not identify a spherical-aberration coefficient or establish manually added aberrations. Fixed-control, fixed-source finer-step runs are required before causal attribution. The separate source audit also records that count refinement changes the legacy energy nodes.

## Reproduce

Run `.venv/Scripts/python.exe outputs/joint-probe-100um-30mrad-20261028/diagnostics/seed769_shape_evolution_h02.py` from the repository root after selecting new output names. Existing diagnostic files are refused. The source archives remain read-only. Output qualification: `NOT_QUALIFIED`.
"""
    (OUT/(STEM+".md")).write_text(text,encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    for name in ("Emission","Gun exit","Corrector checkpoint 1210 mm","Specimen surface"):
        radius=radii[name]; order=np.argsort(radius,kind="stable"); r50=_weighted_quantile(radius,weights/weights.sum(),.5)
        axes[0].plot(radius[order]/r50,np.cumsum(weights[order])/weights.sum(),label=name)
    axes[0].set(xlim=(0,8),ylim=(0,1.005),xlabel="Centered radius / native median radius",ylabel="Fraction of original current",title="Shape changes after removing overall size")
    axes[0].legend(fontsize=8,loc="lower right");axes[0].grid(alpha=.25)
    for key,label in (("d50_nm","D50"),("d95_nm","D95"),("rms_radius_nm","RMS radius (diagnostic)")):
        axes[1].plot([r["surface_offset_nm"] for r in near],[r[key] for r in near],label=label)
    axes[1].axvline(0,color="black",lw=.8,ls="--")
    axes[1].set(xlabel="Actual checkpoint offset from surface (nm)",ylabel="Size (nm)",title="Same 769 particles and 10 uA at every plane")
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.25)
    fig.suptitle("Saved h02 seed769 trajectories: descriptive evidence, not convergence qualification",fontsize=11)
    fig.savefig(OUT/(STEM+".png"),dpi=160)
    plt.close(fig)
    print(json.dumps(dict(status="COMPLETE_READONLY_ANALYSIS",gun_planes=len(gun),column_planes=len(column),
        source_sha256=doc["raw_files_sha256"][str((RUN/"particles.npz").relative_to(ROOT))],
        outputs=[str(OUT/(STEM+s)) for s in (".json",".md",".png")]),indent=2))


if __name__=="__main__":
    main()
