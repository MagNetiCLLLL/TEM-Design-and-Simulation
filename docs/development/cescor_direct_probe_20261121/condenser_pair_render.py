"""Publish the complete rays, specimen current density and reproducible working point."""

import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
import numpy as np

from condenser_pair_study import EV, OUT, fem_dependencies
from condenser_source_fields import EV as PARENT, FEM, sha, write
from relay_transfer import PARTS, START


def load(path):
    return json.loads(path.read_text())


def draw_rays(gun, data, record, *, output_dir=EV, full_only=False, field_label=""):
    zg = gun["gun_z_mm"]
    keep = zg < 450 - 1e-7
    z = np.r_[zg[keep], data["z_mm"]]
    tracks = np.concatenate([gun["gun_paths_m"][keep], data["paths_m"]]) * 1e6
    np.testing.assert_array_equal(data["selected_ids"], gun["selected_ids"])
    # Color the same source identities consistently in every panel.
    n = record["count"]
    index = np.linspace(0, n - 1, 64, dtype=int)
    angle = np.arctan2(gun["launch"][index, 3], gun["launch"][index, 2])
    colors = plt.cm.twilight_shifted((angle + np.pi) / (2 * np.pi))

    def draw(ax, lo, hi, coord):
        sel = (z >= lo - 1e-8) & (z <= hi + 1e-8)
        for i in range(tracks.shape[1]):
            ax.plot(z[sel], tracks[sel, i, coord], color=colors[i], alpha=0.69, lw=0.77)
        ymax = np.nanmax(np.abs(tracks[sel, :, coord])) * 1.18
        ax.set(xlim=(lo, hi), ylim=(-ymax, ymax), ylabel=("x" if coord == 0 else "y") + " (µm)")
        ax.axhline(0, c="#717b83", lw=0.6, zorder=-1)
        ax.grid(axis="y", alpha=0.16)
        ax.spines[["right", "top"]].set_visible(False)
        ax.yaxis.set_major_locator(MaxNLocator(5, symmetric=True))

    def spans(ax, elements):
        for key, label, y in elements:
            row = PARTS[key]
            left = START * 1000 + row["local_start_z_mm"]
            right = START * 1000 + row["local_end_z_mm"]
            ax.axvspan(left, right, color="#7f9dab", alpha=0.13, lw=0, zorder=-2)
            ax.text(
                (left + right) / 2,
                y,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#233f4d",
            )

    def markers(ax, items):
        for where, label, height, color in items:
            ax.axvline(where, ls="--", c=color, lw=0.85)
            ax.text(
                where,
                height,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1),
            )

    top = record["surface_z_mm"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), layout="constrained", sharex=True)
    fig.suptitle("Effective source to specimen surface  |  Nanoprobe Normal", fontsize=17, weight="bold")
    for j, ax in enumerate(axes):
        draw(ax, 0, top, j)
        spans(
            ax,
            [
                ("condenser_lens_1", "C1 lens", 0.99),
                ("condenser_lens_2", "C2 lens", 0.99),
                ("condenser_lens_3", "C3 lens", 0.99),
                ("adapter_lens", "ADL", 0.99),
                ("ac_deflector", "Scan", 0.99),
                ("mini_condenser", "Mini", 0.99),
                ("objective_upper_pole", "Obj. pole", 0.90),
            ],
        )
        first = START * 1000 + PARTS["probe_hp2_hexapole"]["local_start_z_mm"]
        last = START * 1000 + PARTS["probe_tl12_lens"]["local_end_z_mm"]
        ax.axvspan(first, last, color="#7f9dab", alpha=0.13, lw=0, zorder=-2)
        ax.text(
            (first + last) / 2,
            0.99,
            "Corrector · HP2 to TL12",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#233f4d",
        )
        ax.axvline(top, c="#a84132", lw=1.5)
        ax.text(0.995, 0.06, "Specimen", transform=ax.transAxes, ha="right", fontsize=8, color="#a84132")
    axes[0].set_title(
        f"64 original source IDs displayed in both projections  ·  {n:,} weighted particles tracked"
        + (f"  ·  {field_label}" if field_label else ""),
        fontsize=10,
        loc="left",
    )
    axes[1].set_xlabel("Distance from the effective source  z (mm)")
    axes[1].set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400, 1599.2])
    fig.savefig(output_dir / "full-column-rays.png", dpi=165)
    plt.close(fig)
    if full_only:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.6), layout="constrained")
    fig.suptitle("Ray diagram by column section  |  same source IDs", fontsize=17, weight="bold")
    image = record["source_images"][0][0]["z_mm"]
    waist = record["radial_waists_mm"][2][0]
    draw(axes[0], 425, 1071, 0)
    spans(
        axes[0],
        [
            ("condenser_lens_1", "C1 lens", 0.99),
            ("condenser_lens_2", "C2 lens", 0.99),
            ("condenser_lens_3", "C3 lens", 0.99),
        ],
    )
    markers(
        axes[0],
        [
            (image, f"Source image {image:.2f} mm", 0.13, "#9d402e"),
            (765.0, "C2 aperture", 0.98, "#785621"),
            (waist, f"Waist {waist:.2f} mm", 0.13, "#9d402e"),
        ],
    )
    axes[0].set_title("Condenser system to ADL entrance", fontsize=11, loc="left", weight="bold")
    draw(axes[1], 1070, 1425, 0)
    spans(
        axes[1],
        [
            ("adapter_lens", "ADL", 0.99),
            ("probe_hp2_hexapole", "HP2", 0.99),
            ("probe_tl22_lens", "TL22", 0.90),
            ("probe_tl21_lens", "TL21", 0.99),
            ("probe_hp1_hexapole", "HP1", 0.99),
            ("probe_tl12_lens", "TL12", 0.99),
        ],
    )
    upper, lower = START * 1000 + np.array(PARTS["ac_deflector"]["interaction_centers_local_z_mm"])
    markers(axes[1], [(upper, "Upper scan", 0.83, "#395b72"), (lower, "Lower scan", 0.97, "#395b72")])
    axes[1].set_title("Corrector and scan coils", fontsize=11, loc="left", weight="bold")
    draw(axes[2], 1420, top, 0)
    spans(
        axes[2],
        [("mini_condenser", "Mini Condenser", 0.99), ("objective_upper_pole", "Upper objective pole", 0.99)],
    )
    axes[2].axvline(top, c="#a84132", lw=1.5)
    axes[2].text(
        0.996,
        0.13,
        f"Surface {top:.2f} mm",
        transform=axes[2].transAxes,
        ha="right",
        color="#a84132",
        fontsize=9,
    )
    axes[2].set_title("Scan exit to the specimen upper surface", fontsize=11, loc="left", weight="bold")
    axes[2].set_xlabel("Distance from the effective source  z (mm)")
    fig.savefig(output_dir / "ray-sections.png", dpi=165)
    plt.close(fig)


def draw_probe(data, rec, *, output_dir=EV, field_label=""):
    live = (data["weights"] > 0) & (data["loss"] == -1)
    xy = data["surface_phase"][live, :2] * 1e6
    currents = data["weights"][live] * float(data["source_A"]) * 1e6
    s = rec["surface"]
    center = np.array(s["centroid_um"])
    radius = np.linalg.norm(xy - center, axis=1)
    edge = np.ceil(np.max(abs(xy)) * 10) / 10 + 0.1
    edges = np.linspace(-edge, edge, 81)
    pitch = np.diff(edges)[0]
    count, _, _ = np.histogram2d(*xy.T, bins=(edges, edges), weights=currents)
    np.testing.assert_allclose(count.sum(), s["specimen_uA"], rtol=1e-12)
    density = count.T / pitch**2
    cmap = LinearSegmentedColormap.from_list(
        "current_density", ["#f8fafb", "#c0dfdf", "#59a7aa", "#146175", "#10283d"]
    )
    fig, (ax, cum) = plt.subplots(1, 2, figsize=(10.7, 4.9), width_ratios=[1.17, 1], layout="constrained")
    fig.suptitle(f"Specimen upper surface  |  z = {rec['surface_z_mm']:.4f} mm", fontsize=15, weight="bold")
    im = ax.pcolormesh(edges, edges, density, cmap=cmap, shading="flat", rasterized=True)
    ax.add_patch(Circle(center, s["D95_um"] / 2, fill=False, color="#c2562f", ls="--", lw=1.3))
    ax.plot(0, 0, marker="+", color="#e69339", mew=1.2, ms=8)
    ax.set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
    fig.colorbar(im, ax=ax, shrink=0.83, label="Current density (µA/µm²)")
    ax.set_title(f"Particle landings · {pitch*1000:.0f} nm pixels"
                 + (f" · {field_label}" if field_label else ""), fontsize=10, loc="left")
    ix = np.argsort(radius)
    cum.step(radius[ix], np.cumsum(currents[ix]), where="post", c="#007a81", lw=2.2)
    cum.axvline(s["D95_um"] / 2, ls="--", c="#c2562f", lw=1.1)
    cum.axhline(s["specimen_uA"] * 0.95, ls=":", c="#c2562f", lw=1)
    cum.annotate(
        f"D95 = {s['D95_um']:.3f} µm",
        xy=(s["D95_um"] / 2, s["specimen_uA"] * 0.95),
        xytext=(0.61, 0.67),
        textcoords="axes fraction",
        color="#a74226",
        fontsize=10,
        arrowprops=dict(arrowstyle="-", color="#a74226", lw=0.8),
    )
    cum.set(
        xlabel="Radius about the current centroid (µm)",
        ylabel="Enclosed current (µA)",
        xlim=(0, max(radius) * 1.025),
        ylim=(0, 1.06 * sum(currents)),
    )
    cum.grid(alpha=0.18)
    cum.spines[["right", "top"]].set_visible(False)
    cum.set_title(f"{s['count']:,} particles  ·  {s['specimen_uA']:.3f} µA total", fontsize=10, loc="left")
    fig.savefig(output_dir / "specimen-probe.png", dpi=180)
    plt.close(fig)
    return pitch * 1000


def run():
    stem = "column-32768-0.05-7"
    result = load(OUT / (stem + ".json"))
    small = load(OUT / "column-4096-0.05-7.json")
    fine = load(OUT / "column-4096-0.025-5.json")
    refined = load(OUT / "column-32768-0.025-5.json")
    point = load(OUT / "point.json")
    if [len(v) for v in result["radial_waists_mm"]] != [1, 0, 1, 0, 0]:
        raise ValueError("Actual source does not satisfy requested topology")
    for axes in result["principal_waists_mm"]:
        if [len(v) for v in axes] != [1, 0, 1, 0, 0]:
            raise ValueError("Principal-axis growth disagrees with the radial topology")
    if any(len(group) != 1 or max(group[0]["magnifications"]) >= 1 for group in result["source_images"]):
        raise ValueError("The source did not form a unique reduced image at both perturbation amplitudes")
    with np.load(OUT / (stem + ".npz")) as s, np.load(OUT / "tip-32768.npz") as g:
        data, gun = dict(s), dict(g)
    np.testing.assert_array_equal(data["source_ids"], gun["ids"])
    np.testing.assert_array_equal(data["weights"], gun["weight"])
    assert str(data["signature"]) == str(gun["signature"])
    with np.load(OUT / "column-32768-0.025-5.npz") as comparison:
        np.testing.assert_array_equal(comparison["source_ids"], data["source_ids"])
        np.testing.assert_array_equal(comparison["loss"] == -1, data["loss"] == -1)
        live = (data["loss"] == -1) & (data["weights"] > 0)
        difference = comparison["surface_phase"][live, :2] - data["surface_phase"][live, :2]
        max_displacement_nm = float(1e9 * np.max(np.linalg.norm(difference, axis=1)))
    np.testing.assert_allclose(
        refined["all_column_waists_mm"], result["all_column_waists_mm"], atol=0.002, rtol=0
    )
    draw_rays(gun, data, result)
    pitch = draw_probe(data, result)
    for src, dst in (
        ("tip-32768.npz", "fresh-emission-and-exit.npz"),
        (stem + ".npz", "column-and-landings.npz"),
        (stem + ".json", "tracking.json"),
    ):
        shutil.copyfile(OUT / src, EV / dst)
    parent = PARENT / "condenser-source-working-point.json"
    prev = load(parent)
    q = data["series_A"]
    np.testing.assert_array_equal(q[8:], prev["after_group_A"][8:])
    geom = load(EV / "axial-comparison.json")
    manifest = dict(
        mode="Three-condenser Nanoprobe Normal; condenser-pair material working point",
        parent_file=parent.name,
        parent_sha256=sha(parent),
        operating_point=point,
        signed_current_A=q.tolist(),
        winding_owners=prev["owner_per_group"],
        physical_turns=prev["group_turns"],
        corrector_A=prev["corrector_A"],
        objective_A=prev["objective_A"],
        objective_At=prev["objective_At"],
        downstream_tracking_field_sha256=sha(FEM),
        shared_fem_implementation_sha256=fem_dependencies(),
        scan_calibration_sha256=prev["scan_calibration_sha256"],
        C2_aperture_diameter_um=100.0,
        source_A=result["source_A"],
        source_count=result["count"],
        surface=result["surface"],
        foil=result["C2_aperture"],
        material_assignments=geom["parts"],
        bore_field_assumptions=geom["scope"],
        probe_density_pixel_nm=pitch,
        validation=dict(
            smaller_source_4096=small["surface"],
            step_and_order_refinement_4096=fine["surface"],
            step_and_order_refinement_32768=refined["surface"],
            max_matched_particle_displacement_nm=max_displacement_nm,
            fine_step_mm=0.025,
            tracking_step_mm=0.05,
        ),
        evidence_sha256={
            path.name: sha(path)
            for path in sorted(EV.iterdir())
            if path.is_file() and path.name != "working-point.json"
        },
    )
    write(EV / "working-point.json", manifest)
    print(
        json.dumps(
            dict(
                surface=result["surface"],
                foil=result["C2_aperture"],
                current_ledger=result["ledger"],
                waists=result["all_column_waists_mm"],
                source_images=result["source_images"],
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    run()
