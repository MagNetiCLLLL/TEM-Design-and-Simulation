"""Compare completed physical-source nanoprobe trials without changing them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for folder in args.trials:
        report = json.loads((folder/"report.json").read_text())
        if report["status"] != "COMPLETED_DECLARED_SCOPE":
            raise ValueError("A probe comparison requires completed trials")
        plane = next(row for row in report["planes"] if row["key"] == "specimen_entrance")
        profile = report["focus_profile"]
        d95 = min(profile, key=lambda row: row["d95_m"])
        rms = min(profile, key=lambda row: row["radial_rms_m"])
        row = dict(directory=str(folder.resolve()), source_count=report["source_counts"]["positive_current_count"],
            specimen_count=plane["surviving_positive_count"], current_a=plane["current_a"],
            d95_nm=plane["phase_space"]["d95_m"]*1e9,
            rms_nm=plane["phase_space"]["radial_rms_m"]*1e9,
            alpha95_mrad=plane["phase_space"]["alpha95_rad"]*1000,
            minimum_observed_d95_nm=d95["d95_m"]*1e9, d95_minimum_offset_nm=d95["offset_nm"],
            rms_minimum_offset_nm=rms["offset_nm"],
            centroid_core50pm_current_a=plane["phase_space"].get("centroid_core50pm_original_weight", 0.)*report["reference_current_a"],
            centroid_core500pm_current_a=plane["phase_space"].get("centroid_core500pm_original_weight", 0.)*report["reference_current_a"],
            absolute_weight_balance_residual=report["absolute_weight_balance_residual"])
        if (folder/"topology.json").exists():
            topology = json.loads((folder/"topology.json").read_text())
            row["topology"] = {name: {cohort: dict(status=value["status"],
                waists_z_mm=[r["z_mm"] for r in value["regions"]], unresolved=value["unresolved"])
                for cohort, value in records.items()} for name, records in topology["intervals"].items()}
        rows.append(row)
    comparison = {key: (rows[-1][key]-rows[-2][key])/rows[-2][key] for key in
        ("current_a", "d95_nm", "rms_nm", "alpha95_mrad")} if len(rows) > 1 else {}
    result = dict(schema="physical-hybrid-probe-comparison-v1", trials=rows,
        last_pair_relative_changes=comparison,
        scope="Executed geometric reference; source coherence and full wave propagation require independent evaluation",
        notes=["Every trial retains original absolute source weights and physical interception",
               "D95 and RMS minima are distinct sampled observations, not interpolated exact foci",
               "50 pm and 500 pm core currents use apertures centered on the measured centroid; no bright-core fit or refocus",
               "20 nm tip coherence is an explicit unmeasured source assumption"])
    (args.output/"comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), constrained_layout=True)
    final = args.trials[-1]
    report = json.loads((final/"report.json").read_text())
    plane = report["planes"][0]
    with np.load(final/"classical_column.npz") as data:
        z = data["z_mm"]
        i = int(np.argmin(abs(z-plane["z_mm"])))
        live = ~np.isfinite(data["blocked_z_mm"]) | (data["blocked_z_mm"] >= z[i])
        x, y, w = data["x_m"][i, live]*1e9, data["y_m"][i, live]*1e9, data["weight"][live]
        histogram = axes[0].hist2d(x, y, bins=120, range=((-15, 15), (-15, 15)),
            weights=w*report["reference_current_a"]*1e9, norm=LogNorm(), cmap="magma")
    axes[0].set(title="Physical rays at Si entrance", xlabel="Barrel X (nm)", ylabel="Barrel Y (nm)", aspect="equal")
    fig.colorbar(histogram[3], ax=axes[0], label="Current per bin (nA)", shrink=.8)
    for folder, row in zip(args.trials, rows):
        profile = json.loads((folder/"report.json").read_text())["focus_profile"]
        z = [v["offset_nm"] for v in profile]
        axes[1].plot(z, [v["d95_m"]*1e9 for v in profile], label=f"N={row['source_count']}")
        axes[2].plot(z, [v["radial_rms_m"]*1e9 for v in profile], label=f"N={row['source_count']}")
    for ax, title, ylabel in zip(axes[1:], ("Native D95 through focus", "Radial RMS through focus"), ("D95 (nm)", "Radial RMS (nm)")):
        ax.axvline(0., color="black", linestyle="--", alpha=.6, label="Si entrance")
        ax.set(title=title, xlabel="Z relative to Si entrance (nm)", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Current physical FEG model | 20 nm assumed tip coherence | absolute current and aperture losses retained")
    fig.savefig(args.output/"probe_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
