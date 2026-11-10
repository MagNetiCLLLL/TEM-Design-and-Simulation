"""Plot saved field/geometry transport with actual and common original cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Do not overwrite a saved result image")
    report = json.loads((args.trial/"report.json").read_text())
    if report["status"] != "COMPLETED_DECLARED_SCOPE":
        raise ValueError("A completed physical trial is required")
    entrance = report["planes"][0]["z_mm"]
    with np.load(args.trial/"classical_column.npz", allow_pickle=False) as data:
        z, weight, stop = data["z_mm"], data["weight"], data["blocked_z_mm"]
        indices = np.flatnonzero(z <= entrance)
        x, y = data["x_m"][indices], data["y_m"][indices]
        z = z[indices]
        common = ~np.isfinite(stop) | (stop >= entrance)
        curves = [[], []]
        current = []
        for i, plane in enumerate(z):
            actual = ~np.isfinite(stop) | (stop >= plane)
            current.append(float(weight[actual].sum()*report["reference_current_a"]*1e6))
            for series, live in zip(curves, (actual, common)):
                w = weight[live]
                xy = np.c_[x[i, live], y[i, live]]
                centre = np.average(xy, axis=0, weights=w)
                radius = np.linalg.norm(xy-centre, axis=1)
                order = np.argsort(radius)
                q = np.searchsorted(np.cumsum(w[order]), .95*w.sum())
                series.append(2*radius[order[q]]*1e6)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})
    for values, label in zip(curves, ("Actual transmitted population", "Same specimen-surviving original cohort")):
        axes[0].semilogy(z, values, label=label)
    axes[0].set(ylabel="Current-weighted D95 (um)",
        title=f"Physical FEG and incident column | {report['source_counts']['positive_current_count']} originals")
    axes[0].legend(loc="lower left", fontsize=8)
    topology = json.loads((args.trial/"topology.json").read_text())
    for name, values in topology["intervals"].items():
        for region in values["specimen_surviving_cohort"]["regions"]:
            place = region["z_mm"]
            axes[0].axvline(place, color="gray", linestyle=":", alpha=.6)
            axes[0].text(place+3, max(curves[0])*.8, f"{place:g} mm", rotation=90,
                ha="left", va="top", fontsize=8)
    axes[1].plot(z, current, color="tab:green")
    axes[1].set(ylabel="Current (uA)", xlabel="Tip-referenced Z (mm)", ylim=(0, 11))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(entrance, color="black", linestyle="--", alpha=.5)
    fig.suptitle("Normal branch: C2-C3, internal TL and AC waists; no extra Mini-to-specimen crossover")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
