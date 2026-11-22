"""Specimen landing maps and held-out comparisons for the threefold fit."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter
import numpy as np

from plot_specimen_probe import ANNULUS_UM, HERE, landing_current_density


def colour_scale(grids):
    positive = [g[g > 0] for g in grids if np.any(g > 0)]
    norm = LogNorm(vmin=min(s.min() for s in positive), vmax=max(s.max() for s in positive))
    cmap = LinearSegmentedColormap.from_list("current", plt.colormaps["inferno"](np.linspace(.16, .99, 256)))
    cmap.set_bad("#090b13")
    return norm, cmap


def show_map(ax, grid, half_width, ticks, colours):
    norm, cmap = colours
    im = ax.imshow(np.ma.masked_equal(grid, 0), origin="lower", cmap=cmap, norm=norm,
                   interpolation="nearest", extent=(-half_width, half_width, -half_width, half_width))
    ax.set(xticks=ticks, yticks=ticks, aspect="equal")
    ax.tick_params(labelsize=9, length=3, color="#84909b")
    for spine in ax.spines.values():
        spine.set_color("#596372")
    if not np.any(grid):
        ax.text(.5, .53, "No landings\nin this window", transform=ax.transAxes,
                color="#cdd2df", ha="center", va="center", fontsize=10)
    return im


def colour_bar(fig, image, ax):
    cb = fig.colorbar(image, cax=ax)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Landing current density (nA/µm²)", fontsize=10)


def plot_groups(sample, groups, result, *, filename="aperture-angle-spots.png", setting="D95 focus"):
    # The two resolutions are fixed across angular groups. No per-panel scaling
    # of particle weights, colour limits or intercept coordinates is applied.
    windows = [(80, 1., [-80, -40, 0, 40, 80]), (8, .25, [-8, -4, 0, 4, 8])]
    maps = [[landing_current_density(sample.xy_um[g], sample.current_na[g], extent, pitch)[0]
             for g in groups] for extent, pitch, _ in windows]
    colours = colour_scale([g for row in maps for g in row])
    fig = plt.figure(figsize=(13.2, 7.6), facecolor="white")
    grid = fig.add_gridspec(2, 5, width_ratios=(1, 1, 1, 1, .052), left=.061, right=.921,
                            top=.790, bottom=.122, hspace=.21, wspace=.18)
    fig.text(.061, .961, "Specimen landings by aperture semi-angle", fontsize=19,
             fontweight="bold", va="top", color="#17212c")
    fig.text(.061, .905, f"{setting}   ·   C2 aperture Ø 100 µm   ·   specimen upper surface at "
             f"{result['specimen_z_m']*1000:.3f} mm   ·   angles relative to the incident chief ray",
             fontsize=11, va="top", color="#465362")
    for row, (extent, _, ticks) in enumerate(windows):
        for col, group in enumerate(result["angle_groups"]):
            ax = fig.add_subplot(grid[row, col])
            image = show_map(ax, maps[row][col], extent, ticks, colours)
            if row == 0:
                high = group["upper_mrad_exclusive"]
                angle = (f"{group['lower_mrad']:g} ≤ α < {high:g} mrad" if high is not None else
                         f"30 ≤ α ≤ {result['max_angle_mrad']:.2f} mrad")
                ax.set_title(angle + f"\n{group['count']:,} hits   ·   {group['current_uA']:.3f} µA",
                             fontsize=10.8, linespacing=1.6, pad=8)
            else:
                ax.set_xlabel("x (µm)")
            if col == 0:
                ax.set_ylabel("y (µm)")
            else:
                ax.tick_params(labelleft=False)
    colour_bar(fig, image, fig.add_subplot(grid[:, 4]))
    fig.text(.061, .032, "Top  ±80 µm, 1 µm bins.     Bottom  ±8 µm, 0.25 µm bins.     "
             "All panels use the same absolute current-density scale.", fontsize=10, color="#465362")
    fig.savefig(HERE / filename, dpi=175)
    plt.close(fig)


def plot_current_adjustment(samples, records):
    titles = ("Original D95 setting", "Half main differential current", "A₂-adjusted setting")
    windows = ((55, .5, [-50, -25, 0, 25, 50]), (8, .20, [-8, -4, 0, 4, 8]))
    grids = [[landing_current_density(s.xy_um, s.current_na, width, pitch)[0] for s in samples]
             for width, pitch, _ in windows]
    colours = colour_scale([g for row in grids for g in row])
    fig = plt.figure(figsize=(11.2, 7.9), facecolor="white")
    spec = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, .05), left=.074, right=.91,
                            top=.786, bottom=.123, hspace=.23, wspace=.17)
    fig.text(.074, .96, "Specimen probe after physical A₂ adjustment", fontsize=18, fontweight="bold",
             color="#17212c", va="top")
    fig.text(.074, .907, "Native trajectories   ·   same gun, 100 µm C2 aperture and round-lens currents"
             "   ·   fixed specimen plane", fontsize=10.2, color="#465362", va="top")
    for row, (extent, _, ticks) in enumerate(windows):
        for col, (title, stats) in enumerate(zip(titles, records)):
            ax = fig.add_subplot(spec[row, col])
            img = show_map(ax, grids[row][col], extent, ticks, colours)
            if row == 0:
                ax.set_title(f"{title}\nD95 {stats['surface']['D95_nm']/1000:.2f} µm"
                             f"   ·   H₃ {stats['ring']['H3']:.3f}", fontsize=10.4, pad=8, linespacing=1.6)
            else:
                ax.set_xlabel("x (µm)")
            if col == 0:
                ax.set_ylabel("y (µm)")
            else:
                ax.tick_params(labelleft=False)
    colour_bar(fig, img, fig.add_subplot(spec[:, 3]))
    fig.text(.074, .035, "Top  ±55 µm.     Bottom  ±8 µm.     H₃ measures the threefold current contrast "
             "at 2–6 µm radius.", fontsize=10, color="#465362")
    fig.savefig(HERE / "a2-current-adjustment.png", dpi=175)
    plt.close(fig)


def plot_ablation(sample, removed, result):
    xy = [sample.xy_um, removed["A2"], removed["D4"]]
    titles = ("Observed landings", "Subtract fitted A₂", "Subtract fitted D₄")
    rings = [result["observed_ring"], *(result["ring_after_subtracting_held_out_term"][t] for t in ("A2", "D4"))]
    grids = [landing_current_density(points, sample.current_na, 8, .25)[0] for points in xy]
    colours = colour_scale(grids)
    fig = plt.figure(figsize=(11.8, 7.8), facecolor="white")
    fig.text(.066, .962, "Three-arm probe at fixed focus", fontsize=19, fontweight="bold",
             va="top", color="#17212c")
    fig.text(.066, .912, "Same electrons and current in every map. Each subtraction uses a fit trained on other rays.",
             fontsize=11, va="top", color="#465362")
    top = fig.add_gridspec(1, 4, width_ratios=(1, 1, 1, .053), left=.067, right=.914,
                           bottom=.469, top=.808, wspace=.17)
    for j, (title, ring, g) in enumerate(zip(titles, rings, grids)):
        ax = fig.add_subplot(top[0, j])
        image = show_map(ax, g, 8, [-8, -4, 0, 4, 8], colours)
        ax.set_title(f"{title}\nH₃ = {ring['H3']:.3f}", fontsize=11, linespacing=1.5, pad=6)
        ax.set_xlabel("x (µm)")
        if j == 0:
            ax.set_ylabel("y (µm)")
        else:
            ax.tick_params(labelleft=False)
    colour_bar(fig, image, fig.add_subplot(top[:, 3]))

    bottom = fig.add_gridspec(1, 2, width_ratios=(1.25, 1.), left=.08, right=.923,
                              bottom=.093, top=.33, wspace=.29)
    profile = fig.add_subplot(bottom[0, 0])
    edges = np.arange(0, 361, 5)
    for points, label, colour in zip(xy, ("Observed", "Subtract A₂", "Subtract D₄"),
                                     ("#283d52", "#bf5932", "#188a82")):
        r = np.hypot(*points.T)
        inside = (r >= ANNULUS_UM[0]) & (r < ANNULUS_UM[1])
        azimuth = np.degrees(np.arctan2(points[inside, 1], points[inside, 0])) % 360
        counts, _ = np.histogram(azimuth, bins=edges, weights=sample.current_na[inside])
        profile.stairs(counts / np.diff(edges), edges, baseline=None, color=colour,
                       linewidth=1.6, linestyle="--" if label == "Subtract D₄" else "-", label=label)
    profile.set(xlim=(0, 360), xticks=[0, 90, 180, 270, 360], ylim=(0, None),
                xlabel="Landing azimuth from +x toward +y (°)", ylabel="Current (nA/degree)")
    profile.set_ylim(top=profile.get_ylim()[1]*1.18)
    profile.set_title("Current in the 2–6 µm annulus", loc="left", fontsize=12, fontweight="bold")
    profile.legend(loc="upper center", ncol=3, fontsize=8, frameon=False, handlelength=1.6, columnspacing=.8)

    residual = fig.add_subplot(bottom[0, 1])
    labels = ("Background", "+ A₂", "+ D₄", "+ both")
    rms = [result["held_out_models"][m]["rms_um"] for m in ("background", "A2", "D4", "both")]
    bars = residual.barh(labels, rms, color=["#91a0ac", "#238578", "#ad703e", "#354c61"], height=.52)
    residual.invert_yaxis()
    residual.bar_label(bars, labels=[f"{v:.2f}" for v in rms], padding=4, fontsize=9)
    residual.set(xlabel="Held-out vector RMS error (µm)", xlim=(0, max(rms)*1.18))
    residual.set_title("Landing prediction error", loc="left", fontsize=12, fontweight="bold")
    for ax in (profile, residual):
        ax.spines[["right", "top"]].set_visible(False)
        ax.tick_params(labelsize=9)
    fig.savefig(HERE / "threefold-attribution.png", dpi=175)
    plt.close(fig)
