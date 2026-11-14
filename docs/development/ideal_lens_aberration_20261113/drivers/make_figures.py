"""Publication-style figures from completed ideal-lens scalar observations only."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def make_figures(destination, reference, cases, primary_name="fixed-b0.3-s6"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": "ideal-lens-20261113"})
    products = []

    def save(fig, name):
        for extension in ("png", "pdf", "svg"):
            path = destination/(name+"."+extension)
            if extension == "svg":
                fig.savefig(path, metadata={"Date": None})
            elif extension == "pdf":
                fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
            else:
                fig.savefig(path, dpi=180)
            products.append(path.name)
        plt.close(fig)

    def grid(rows, quantity):
        peaks = sorted({float(row["peak_t"]) for row in reference["cases"]})
        sigmas = sorted({float(row["sigma_mm"]) for row in reference["cases"]})
        values = np.full((len(sigmas), len(peaks)), np.nan)
        for row in rows:
            values[sigmas.index(float(row["sigma_mm"])), peaks.index(float(row["peak_t"]))] = quantity(row)
        return peaks, sigmas, values

    def heatmap(ax, rows, quantity, label, logarithmic=False):
        peaks, sigmas, values = grid(rows, quantity)
        finite = values[np.isfinite(values)]
        if not finite.size:
            ax.set_visible(False)
            return
        norm = LogNorm(vmin=float(finite.min()), vmax=float(finite.max())) if logarithmic and np.all(finite > 0) and np.ptp(finite) else Normalize(vmin=float(finite.min()), vmax=float(finite.max()) or 1.)
        rendered = ax.imshow(np.ma.masked_invalid(values), origin="lower", aspect="auto", cmap="cividis", norm=norm)
        ax.set_xticks(range(len(peaks)), [f"{value:g}" for value in peaks])
        ax.set_yticks(range(len(sigmas)), [f"{value:g}" for value in sigmas])
        ax.set_xlabel("Peak on-axis field B₀ (T)")
        ax.set_ylabel("Gaussian standard deviation σ (mm)")
        ax.set_title(label)
        for index, value in np.ndenumerate(values):
            text = f"{value:.4g}" if np.isfinite(value) else "—"
            row = next((row for row in rows if float(row["sigma_mm"]) == sigmas[index[0]] and float(row["peak_t"]) == peaks[index[1]]), {})
            if row.get("paraxial_edge_slope_mrad", 0.) > 100.:
                text += "†"
            color = "white" if np.isfinite(value) and norm(value) < .53 else "black"
            ax.text(index[1], index[0], text, ha="center", va="center", color=color, fontsize=8)
        plt.colorbar(rendered, ax=ax, shrink=.8)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
    heatmap(axes[0], reference["cases"], lambda row: row["variants"]["refined8"]["coefficients"]["cs_mm"], "Intrinsic third-order Cs (mm)", True)
    heatmap(axes[1], reference["cases"], lambda row: row["variants"]["refined8"]["coefficients"]["cc_mm"], "Frozen-hardware chromatic Cc (mm)", True)
    fig.suptitle("300 keV single Gaussian lens: independent variational reference", fontsize=11)
    save(fig, "intrinsic-coefficients")

    def population(report, count=2048):
        return next(item for item in report["populations"] if item["original_positive_count"] == count)

    def view(report, diameter=5.76, count=2048):
        return next(item for item in population(report, count)["tube_views"] if item["tube_diameter_mm"] == diameter)

    principal = [row for row in cases if not row["request"].get("refinement_parent")]
    fixed = [row for row in principal if row["request"].get("family") == "fixed_source"]
    if fixed:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), constrained_layout=True)
        rows = [{**row, "peak_t": row["request"]["peak_t"], "sigma_mm": row["request"]["sigma_mm"]} for row in fixed]
        heatmap(axes[0], rows, lambda row: view(row)["at_paraxial_image"].get("d95_nm", np.nan)*1e-3, "Actual D95 at paraxial image (µm)", True)
        heatmap(axes[1], rows, lambda row: view(row)["at_paraxial_image"].get("alpha95_mrad", np.nan), "Actual α95 at paraxial image (mrad)")
        fig.suptitle("Fixed 0.95 mm source radius · 2048 original packets · 5.76 mm tube ID\n† Declared paraxial edge exceeds 100 mrad: separate high-angle exploration", fontsize=10)
        save(fig, "fixed-source-grid")

    primary = next((row for row in cases if row["request"]["name"] == primary_name), None)
    if primary:
        full = population(primary)
        tube_views = sorted(full["tube_views"], key=lambda row: row["tube_diameter_mm"])
        diameter = [row["tube_diameter_mm"] for row in tube_views]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
        for ax in axes:
            ax.set_xlabel("Passive tube inner diameter (mm)")
            ax.set_xticks(diameter, [f"{value:g}" for value in diameter], rotation=35, ha="right")
            ax.grid(alpha=.25)
        axes[0].plot(diameter, [100*row["at_paraxial_image"]["transmission"] for row in tube_views], "o-")
        axes[0].set_ylabel("Actual current / declared source (%)")
        axes[0].set_ylim(0, 105)
        axes[1].plot(diameter, [row["at_paraxial_image"].get("d95_nm", np.nan)*1e-3 for row in tube_views], "o-", label="At paraxial image")
        axes[1].plot(diameter, [row["native_d95_minimum"].get("d95_nm", np.nan)*1e-3 for row in tube_views], "s--", label="Lowest sampled D95")
        axes[1].set_ylabel("Actual D95 diameter (µm)")
        axes[1].legend()
        axes[2].plot(diameter, [primary["intrinsic_cs_mm"]]*len(diameter), "o-", label="Independent intrinsic Cs")
        axes[2].set_ylabel("Intrinsic Cs (mm)")
        axes[2].legend()
        fig.suptitle("B₀ = 0.3 T, σ = 6 mm · identical field and source, physical interception", fontsize=11)
        save(fig, "tube-current-size-intrinsic")

        fig, axes = plt.subplots(2, 1, figsize=(7.4, 5.6), sharex=True, constrained_layout=True)
        for row in tube_views:
            planes = row["focus_planes"]
            offsets = [p["offset_from_paraxial_nm"]*1e-3 for p in planes]
            label = f"Tube ID {row['tube_diameter_mm']:g} mm"
            for ax, key in zip(axes, ("d95_nm", "rms_radius_nm")):
                ax.plot(offsets, [p.get(key, np.nan)*1e-3 for p in planes], label=label)
            minimum = row["native_d95_minimum"]
            axes[0].plot(minimum["offset_from_paraxial_nm"]*1e-3, minimum.get("d95_nm", np.nan)*1e-3, "o", ms=4)
        for ax in axes:
            ax.axvline(0, ls="--", lw=.8, color="black", label="Fixed paraxial image")
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Native D95 diameter (µm)")
        axes[1].set_ylabel("Actual RMS radius (µm)")
        axes[1].set_xlabel("Native plane offset from fixed paraxial image (µm)")
        axes[0].legend()
        axes[0].set_title("D95 and RMS through focus · 129 sampled planes\nMinimum markers do not locate an exact continuous optimum", fontsize=10)
        save(fig, "native-focus")

    matched = sorted((row for row in principal if row["request"].get("family") == "matched_paraxial_edge"), key=lambda row: row["request"]["peak_t"])
    if matched:
        x = [row["request"]["peak_t"] for row in matched]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3.1), constrained_layout=True)
        axes[0].plot(x, [row["source_radius_mm"] for row in matched], "o-")
        axes[0].set_ylabel("Declared source radius (mm)")
        axes[1].plot(x, [view(row)["at_paraxial_image"]["d95_nm"] for row in matched], "o-")
        axes[1].set_ylabel("Actual D95 at paraxial image (nm)")
        axes[1].set_yscale("log")
        axes[2].plot(x, [view(row)["at_paraxial_image"]["alpha95_mrad"] for row in matched], "o-", label="Actual α95")
        axes[2].axhline(10, color="black", ls="--", lw=.8, label="Declared paraxial edge slope")
        axes[2].set_ylabel("Angular statistic (mrad)")
        axes[2].legend()
        for ax in axes:
            ax.set_xlabel("Peak on-axis field B₀ (T)")
            ax.set_xticks(x)
            ax.grid(alpha=.25)
        fig.suptitle("σ = 6 mm · 10 mrad nominal paraxial edge · different prepared source radii", fontsize=11)
        save(fig, "matched-paraxial-edge")

    filled = [row for row in principal if row["request"].get("family") == "matched_tube_filling"]
    if primary:
        filled.append(primary)
    if len(filled) > 1:
        filled.sort(key=lambda row: row["source_radius_mm"])
        def filling_view(row):
            diameter = 2. if row["request"]["name"] == primary_name else row["request"]["tube_diameters_mm"][0]
            return view(row, diameter)
        x = [filling_view(row)["tube_diameter_mm"] for row in filled]
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), constrained_layout=True)
        axes[0].plot(x, [filling_view(row)["at_paraxial_image"]["d95_nm"]*1e-3 for row in filled], "o-")
        axes[0].set_ylabel("Actual D95 at paraxial image (µm)")
        axes[0].set_yscale("log")
        axes[1].plot(x, [100*(population(row)["vacuum_fit"]["cs_fit_mm"]/row["intrinsic_cs_mm"]-1) for row in filled], "x-", color="black", label="Recorded Cartesian fit")
        for row, label, marker in ((5, "Radial fit through fifth order", "s"), (7, "Radial fit through seventh order", "^")):
            values = []
            for case in filled:
                fits = population(case)["vacuum_fit"]["radial_geometric_fits"]
                fitted = next(fit["cs_mm"] for fit in fits if max(fit["powers"]) == row)
                values.append(100*(fitted/case["intrinsic_cs_mm"]-1))
            axes[1].plot(x, values, marker+"-", label=label)
        axes[1].axhline(0, color="black", ls="--", lw=.8)
        axes[1].set_ylabel("Finite-pupil Cs error (%)")
        axes[1].axhspan(-.1, .1, color="#46794c", alpha=.06, label="Declared ±0.1% C3 recovery gate")
        axes[1].legend()
        for ax in axes:
            ax.set_xlabel("Passive tube inner diameter (mm)")
            ax.set_xticks(x)
            ax.grid(alpha=.25)
        fig.suptitle("B₀ = 0.3 T, σ = 6 mm · different source radii\nSource radius = 95% of tube radius; intrinsic Cs remains fixed", fontsize=10)
        save(fig, "filled-pupil-sensitivity")

    parents = sorted({row["request"]["refinement_parent"] for row in cases if row["request"].get("refinement_parent")})
    if parents:
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.9), constrained_layout=True)
        for ax, parent in zip(axes.ravel(), parents):
            base = next(row for row in cases if row["request"]["name"] == parent)
            refinements = [row for row in cases if row["request"].get("refinement_parent") == parent]
            arranged = [next(row for row in refinements if row["request"]["refinement"] == key) for key in ("order5", "order7")]
            arranged += [base, next(row for row in refinements if row["request"]["refinement"] == "half-step")]
            baseline = view(base)["at_paraxial_image"]["d95_nm"]
            changes = [100*(view(row)["at_paraxial_image"]["d95_nm"]/baseline-1.) for row in arranged]
            ax.plot(range(4), changes, "o-")
            ax.axhline(0, color="black", ls="--", lw=.7)
            ax.set_xticks(range(4), ("Order 5", "Order 7", "Order 9", "Order 9\nhalf step"))
            ax.set_yscale("symlog", linthresh=1e-6)
            ax.set_title(parent)
            ax.set_ylabel("D95 relative to order 9 base (%)")
            ax.grid(alpha=.25)
        for ax in axes.ravel()[len(parents):]:
            ax.set_visible(False)
        fig.suptitle("Completed numerical refinements · 2048 packets, tube ID 5.76 mm\nSigned sensitivity; original per-case acceptance gates remain separate", fontsize=10)
        save(fig, "numerical-refinements")
    return products
