"""Render absolute specimen current from the archived source packets on a CPU."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Circle
import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "evidence"


def render():
    record = json.loads((EVIDENCE/"normal-objective-probe.json").read_text(encoding="utf-8"))
    with np.load(EVIDENCE/"normal-objective-landings.npz", allow_pickle=False) as f:
        edges = f["pixel_edges_um"]
        stages = []
        for key in ("before","after"):
            m = record[key]
            live = f[key+"_loss"] == -1
            xy = f[key+"_phase"][live,:2]*1e6
            current = f["weight"][live]*record["source_current_A"]*1e9
            hist = np.histogram2d(*xy.T,bins=(edges,edges),weights=current)[0]
            np.testing.assert_allclose(hist, f[key+"_nA_per_pixel"], atol=1e-10)
            np.testing.assert_allclose(hist.sum()*.001,m["transmitted_uA"],rtol=1e-12)
            stages.append((m, xy, current, hist.T/(edges[1]-edges[0])**2))
    palette = LinearSegmentedColormap.from_list("probe", plt.get_cmap("magma")(np.linspace(.18,1.,200)))
    palette.set_bad("#080610")
    present = np.concatenate([d[d>0] for _,_,_,d in stages])
    norm = LogNorm(vmin=present.min()*.90,vmax=present.max())
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"axes.titleweight":"bold","savefig.facecolor":"white"})

    def plane(ax, stage, title):
        row, xy, current, density = stage
        im = ax.imshow(np.ma.masked_less_equal(density,0),origin="lower",interpolation="nearest",
                       extent=(edges[0],edges[-1],edges[0],edges[-1]),cmap=palette,norm=norm)
        ax.add_patch(Circle(np.array(row["centroid_nm"])*.001,row["D95_um"]/2,fill=False,
                            linestyle=(0,(4,4)),linewidth=1.05,edgecolor="#8fe4d8"))
        ax.text(.024,.032,"95% current within dashed circle",transform=ax.transAxes,color="#b8ece4",fontsize=8)
        ax.set(xlabel="x (µm)",ylabel="y (µm)",aspect="equal")
        ax.set_title(title,loc="left",fontsize=12,pad=8)
        return im

    fig = plt.figure(figsize=(9.9,6.3),layout="constrained")
    layout = fig.add_gridspec(2,2,width_ratios=[1.55,1.0],height_ratios=[1.05,1.0])
    whole, core, enclosed = fig.add_subplot(layout[:,0]),fig.add_subplot(layout[0,1]),fig.add_subplot(layout[1,1])
    m, xy, current, _ = stages[1]
    fig.suptitle("Nanoprobe Normal   |   Specimen upper surface",fontsize=16,fontweight="bold")
    im = plane(whole,stages[1],f"Incident current density   ·   D95 {m['D95_um']:.3f} µm")
    cb = fig.colorbar(im,ax=whole,orientation="horizontal",pad=.035,shrink=.88,aspect=36)
    cb.set_label("Current density (nA/µm²)   ·   logarithmic scale")
    side = .18
    within = np.max(abs(xy),axis=1)<=side
    core.scatter(*(xy[within]*1000).T,s=3,color="#263f58",alpha=.70,linewidths=0,rasterized=True)
    core.add_patch(Circle((0,0),50,fill=False,lw=1.35,edgecolor="#c65235"))
    core.plot(0,0,"+",color="#bd482e",ms=6)
    core.set(xlim=(-side*1000,side*1000),ylim=(-side*1000,side*1000),aspect="equal",xlabel="x (nm)",ylabel="y (nm)")
    core.set_title(f"r ≤ 50 nm   ·   {m['axis_50nm_nA']:.2f} nA",loc="left",fontsize=10)
    for (r, positions, currents, _), label, color, ls in zip(stages,("Before","Final"),
                ("#938883","#146a77"),("--","-")):
        radial = np.linalg.norm(positions,axis=1)
        order = np.argsort(radial)
        enclosed.step(radial[order],.001*np.cumsum(currents[order]),where="post",color=color,
                      linestyle=ls,lw=1.5,label=label)
    enclosed.set(xlabel="Radius about the optical axis (µm)",ylabel="Enclosed current (µA)",xscale="log",
                 xlim=(.012,edges[-1]*1.15),ylim=(0,m["transmitted_uA"]*1.08))
    enclosed.grid(axis="y",color="#e3e7e9",lw=.6)
    enclosed.legend(frameon=False,loc="upper left",fontsize=8)
    enclosed.set_title("Encircled specimen current",loc="left",fontsize=10.5)
    fig.savefig(HERE/"normal-objective-final-probe.png",dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1,2,figsize=(9.5,5.7),layout="constrained")
    fig.suptitle("Objective-only focus   |   Matched specimen landings",fontsize=15,fontweight="bold")
    for ax, stage, label in zip(axes,stages,("Before","Final")):
        r = stage[0]
        im = plane(ax,stage,f"{label}   ·   D95 {r['D95_um']:.3f} µm")
    cb = fig.colorbar(im,ax=axes,orientation="horizontal",shrink=.62,aspect=45,pad=.035)
    cb.set_label("Incident current density (nA/µm²)   ·   same absolute scale")
    fig.savefig(HERE/"normal-objective-probe-comparison.png",dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    render()
