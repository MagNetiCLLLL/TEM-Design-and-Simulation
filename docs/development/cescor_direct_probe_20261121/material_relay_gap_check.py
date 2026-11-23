"""Compare the same original rays on either side of the zero-scan relay plane."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_source_fields import write
from column_current import FACTOR
from material_relay_column import Column, Field, source_and_electrodes
from material_relay_fields import SAVE, WORK
from relay_transfer import advance
from trace_normal_joint_field import beam_moments


def measure(label):
    folder = SAVE / label
    r = json.loads((folder / "tracking.json").read_text())
    front = r["isolated_pair_fronts"][0]["z_mm"]
    waists = [w for w in r["waists_mm"] if 1420. < w < 1512.5]
    ref = min(waists, key=lambda w: abs(w-front)) if waists else front
    inside = min(5., .8*(ref-1420.), .8*(1512.5-ref))
    if inside <= 0:
        raise ValueError("The comparison plane must be in the open scan-to-Mini gap")
    offsets = np.unique([-5., -inside, -1., -.1, 0., .1, 1., inside, 5.])
    dc = Field(WORK / r["field"]["source"], step=r["field"]["step_mm"])
    model = Column(source_and_electrodes(), dc, Path(r["gun_source"]), r["count"],
                   r["step_mm"], observation_mm=ref+offsets)
    stops = model.indices((ref+offsets)*.001)
    np.testing.assert_allclose(model.m.z[stops]*1000, ref+offsets, rtol=0., atol=1e-8)
    states = advance(model.t, model.full, model.t.input, model.en, model.phi0, model.loss, 0, stops)
    mask = states[0][1] == -1
    weights = model.t.weights[mask]
    if not np.all(weights > 0) or len(np.unique(model.t.ids[mask])) < 1000:
        raise ValueError("Insufficient original particles")
    centered, rows = [], []
    for (phase, loss), offset in zip(states, offsets, strict=True):
        np.testing.assert_array_equal(loss == -1, mask)
        stats = beam_moments(phase, mask, model.t.weights)
        xy = phase[mask, :2]
        centered.append(xy - np.average(xy, axis=0, weights=weights))
        rows.append(dict(z_mm=ref+offset, offset_mm=offset, rms_nm=stats[0]*1000,
                         directional_half_variance_slopes_nm2_per_mm=(1e6*np.array(stats[3:5])).tolist()))
    def exchanged(delta):
        left, right = (centered[np.argmin(abs(offsets-o))] for o in (-delta, delta))
        opposed = np.sum(left*right, axis=1) < 0
        return float(weights[opposed].sum()/weights.sum())
    bz = np.interp(np.array([1387.5, 1412.5, ref, front])*.001, dc.z, dc.c[:, 0, 2])
    pair = Field(WORK / r["pair"]["source"], step=r["pair"]["step_mm"])
    pair_bz = np.interp(np.array([1387.5, 1412.5, ref, front])*.001, pair.z, pair.c[:, 0, 2])
    interval = (ref + np.linspace(-inside, inside, 101))*.001
    rotation = FACTOR*np.trapezoid(np.interp(interval, dc.z, dc.c[:,0,2]), interval)
    return dict(label=label, field_sha256=dc.receipt["solution_sha256"], original=r["count"],
                surviving=int(mask.sum()), positive_effective_count=float(weights.sum()**2/(weights@weights)),
                gap_waists_mm=waists, front_mm=front, checked_plane_mm=ref, planes=rows,
                opposite_half_plane_current_fraction=exchanged(5.),
                open_gap_half_span_mm=inside, open_gap_opposite_half_plane_current_fraction=exchanged(inside),
                open_gap_larmor_rotation_deg_at_300_kV=float(np.rad2deg(rotation)),
                opposite_half_plane_definition="Negative dot product of the same ray's centroid-relative x,y positions 5 mm before and after the reference plane, in laboratory coordinates.",
                field_stations_mm=[1387.5, 1412.5, ref, front], full_Bz_mT=(bz*1000).tolist(),
                isolated_pair_Bz_mT=(pair_bz*1000).tolist())


def draw(comparison, label):
    folder = SAVE / label
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.7), sharex=True, sharey="row", layout="constrained")
    fig.suptitle("Zero scan between the lower deflector and Mini Condenser", fontsize=16, weight="bold")
    for i, (case, title) in enumerate(zip(comparison, ("Previous setting", "Restored crossover"), strict=True)):
        with np.load(SAVE / case["label"] / "column-and-landings.npz") as d:
            z, paths, rms = d["z_mm"], d["paths_m"][:, :, :2]*1e9, d["stats"][:, 0]*1000
        front = case["front_mm"]
        w = case["gap_waists_mm"]
        keep = (z >= 1411.9) & (z <= 1516.1)
        idx = np.argmin(abs(z-(w[-1] if w else front)+5.))
        colors = plt.cm.coolwarm(np.argsort(np.argsort(paths[idx, :, 0]))/(paths.shape[1]-1))
        top, bottom = axes[:, i]
        for j in np.argsort(paths[idx, :, 0])[::2]:
            top.plot(z[keep], paths[keep, j, 0], lw=.72, alpha=.8, color=colors[j])
        top.set_title(title, loc="left", fontsize=12, weight="bold")
        top.set_yscale("symlog", linthresh=100.)
        top.set_yticks([-1000, -100, 0, 100, 1000], ["−1000", "−100", "0", "100", "1000"])
        bottom.plot(z[keep], rms[keep], color="#164c63", lw=1.8, label="Radial RMS")
        if w:
            point = next(row for row in case["planes"] if row["offset_mm"] == 0.)
            bottom.plot(w[-1], point["rms_nm"], "o", ms=4, color="#00785d", zorder=4)
        bottom.set_yscale("log")
        for ax in (top, bottom):
            ax.axvspan(1412., 1420., color="#9ba8b5", alpha=.35, lw=0)
            ax.axvspan(1512.5, 1516., color="#9ba8b5", alpha=.35, lw=0)
            ax.axvline(front, color="#b45427", ls="--", lw=1.5, label="Lens-pair front")
            if w:
                ax.axvline(w[-1], color="#00785d", ls="-.", lw=1.4, label="Static waist")
            ax.spines[["right", "top"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
        bottom.set(xlim=(1412, 1516), xlabel="Distance from the effective source (mm)")
        bottom.legend(loc="lower right", fontsize=8, frameon=True)
    axes[0, 0].set_ylabel("Individual x positions (nm)")
    axes[1, 0].set_ylabel("Current-weighted RMS radius (nm)")
    comparison_file = folder / "post-scan-crossover.png"
    fig.savefig(comparison_file, dpi=165)
    plt.close(fig)
    current = comparison[-1]
    if not current["gap_waists_mm"]:
        return
    w = current["checked_plane_mm"]
    lo, hi = w-6.5, w+6.5
    fig, detail = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True, layout="constrained")
    fig.suptitle(f"Static beam crossover at {w:.3f} mm", weight="bold", fontsize=15)
    with np.load(folder / "column-and-landings.npz") as data:
        z = data["z_mm"]
        keep = (z >= lo-.5) & (z <= hi+.5)
        paths = data["paths_m"][:, :, :2] * 1e9
    idx = np.argmin(abs(z-w+3.))
    colors = plt.cm.twilight_shifted((np.arctan2(paths[idx,:,1],paths[idx,:,0])+np.pi)/(2*np.pi))
    for j, ax in enumerate(detail):
        for i in range(0, paths.shape[1], 2):
            ax.plot(z[keep], paths[keep,i,j], lw=1.05, color=colors[i], alpha=.83)
        if lo < 1420.:
            ax.axvspan(lo, 1420., color="#c1ccda", alpha=.5, lw=0)
            ax.axvline(1420., color="#536b84", lw=.85)
            if j == 0:
                ax.text((lo+1420.)/2,.92,"Lower scan coil",ha="center",transform=ax.get_xaxis_transform(),color="#3a5267")
        ax.axvline(w,color="#00785d",ls="-.",lw=1.4,label=f"Beam waist  {w:.3f} mm")
        front=current["front_mm"]
        if lo <= front <= hi:
            ax.axvline(front,color="#a74925",ls=":",lw=1.65,label=f"Lens-pair front  {front:.3f} mm")
        ax.set(xlim=(lo,hi),ylabel=("x" if j==0 else "y")+" (nm)")
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y",alpha=.15)
    detail[0].legend(loc="upper right",framealpha=.9,fontsize=8)
    detail[1].set_xlabel("Distance from the effective source (mm)")
    fig.savefig(folder/"post-scan-rays-detail.png",dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--previous", default="relay-final")
    args = parser.parse_args()
    comparison = [measure(args.previous), measure(args.label)]
    write(SAVE / args.label / "post-scan-check.json", dict(previous=comparison[0], current=comparison[1]))
    draw(comparison, args.label)
