"""Plot archived, unmodified Microprobe particles and native statistics."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "docs/development/microprobe_20261111"
data = json.loads((FOLDER / "results.json").read_text(encoding="utf-8"))
fig, axes = plt.subplots(2, 3, figsize=(14., 8.2), layout="constrained")
colors = ("#1b7a88", "#8c5bab", "#c66a29")
for i, (case, slug, index) in enumerate(zip(data["family"], ("075um", "100um", "220um"), (2, 0, 1))):
    surface = case["surface"]
    path = FOLDER / "column/n2048-step010" / f"trial-{index:02d}/sample-neighborhood.npz"
    with np.load(path, allow_pickle=False) as phase:
        row = np.searchsorted(phase["z_mm"], surface["z_mm"])
        keep = phase["alive"][row] & (phase["weights"] > 0.)
        x, y = phase["x_m"][row, keep] * 1e6, phase["y_m"][row, keep] * 1e6
        ax = axes[0, i]
        ax.scatter(x, y, s=6., alpha=.65, color=colors[i], linewidths=0., rasterized=True)
        ax.add_patch(Circle(surface["centroid_xy_um"], .5 * surface["d95_um"],
                            fill=False, color="#222222", lw=1.1, ls="--"))
        ax.axhline(0., color="#dddddd", lw=.6, zorder=0)
        ax.axvline(0., color="#dddddd", lw=.6, zorder=0)
        ax.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), aspect="equal", xlabel="x (µm)", ylabel="y (µm)")
        ax.set_title(f"D95 = {surface['d95_um']:.4f} µm\n"
                     f"α95 = {surface['alpha95_mrad']:.5f} mrad; I = {surface['current_a']*1e6:.3f} µA")
        ax.text(.03, .03, f"{surface['count']} arrivals / 2048 emitted", transform=ax.transAxes, fontsize=9.)
    profile = case["axial_profile"]
    axes[1, 0].plot([row["offset_from_surface_um"] for row in profile],
                   [100 * (row["d95_um"] / surface["d95_um"] - 1) for row in profile],
                   color=colors[i], marker=".", label=f"{surface['d95_um']:.3f} µm")
diameters = [case["surface"]["d95_um"] for case in data["family"]]
axes[1, 0].axhline(0., color="#999999", lw=.6)
axes[1, 0].set(xlabel="Offset from specimen upper surface (µm)", ylabel="D95 change from surface (%)")
axes[1, 0].set_title("Executed local illumination stability")
axes[1, 0].legend(frameon=False)
for key, label, style in (("alpha95_mrad", "Mechanical α95", "o-"), ("alpha_rms_mrad", "Mechanical angular RMS", "s--")):
    axes[1, 1].plot(diameters, [case["surface"][key] for case in data["family"]], style, label=label)
axes[1, 1].axhline(.3, color="#a04444", ls=":", label="Initial α95 screen: 0.3 mrad")
axes[1, 1].set(xlabel="Measured specimen D95 (µm)", ylabel="Angle (mrad)", ylim=(0., .325))
axes[1, 1].set_title("Measured angle and illuminated size")
axes[1, 1].legend(frameon=False, fontsize=8.)
left, right = axes[1, 2], axes[1, 2].twinx()
left.plot(diameters, [case["controls"]["condenser_lens_2"]["percent"] for case in data["family"]],
          "o-", color="#286599")
right.plot(diameters, [case["controls"]["condenser_lens_3"]["percent"] for case in data["family"]],
           "s-", color="#bc6735")
left.set(xlabel="Measured specimen D95 (µm)", ylabel="C2 excitation (%)")
right.set_ylabel("C3 excitation (%)")
left.yaxis.label.set_color("#286599")
right.yaxis.label.set_color("#bc6735")
left.set_title("C2 / C3 change; other optics fixed")
for ax in axes[1]:
    ax.grid(alpha=.2)
fig.suptitle("Microprobe incident illumination from classical particles\n"
             "Mini +26.67834%; Objective 29.32640%; fixed corrector; 100 µm C2 aperture", fontsize=14.)
fig.savefig(FOLDER / "illumination-family.png", dpi=175, facecolor="white")
plt.close(fig)
print(FOLDER / "illumination-family.png")
