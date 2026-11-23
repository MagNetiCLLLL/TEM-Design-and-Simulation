"""Draw the current Nanoprobe Normal rays, or reproduce the earlier vacuum-control figures."""

import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = HERE / "evidence"
TRACE = ROOT / "outputs/cescor-relay-20261121/normal-core-fixed-0.0375-capture"
COLORS = ("#2768a4", "#188372", "#b17a1a", "#bd473a")
INK = "#25384b"
WAIST = "#9c3f30"
FOCUS = "#146779"
PIVOT = "#814598"
FOIL = "#475662"
LOST = "#7a8490"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_tracks():
    probe = read_json(EVIDENCE / "normal-objective-probe.json")
    final = probe["after"]
    capture = read_json(TRACE.with_name(TRACE.name + ".json"))
    if (capture["corrector_sha256"] != final["corrector_table_sha256"]
            or capture["objective_At"] != final["objective_At"]
            or capture["source_digest"] != probe["source_digest"]):
        raise ValueError("The native trajectory file does not describe the final probe")
    with np.load(str(TRACE) + ".npz", allow_pickle=False) as native, np.load(
            EVIDENCE / "normal-objective-landings.npz", allow_pickle=False) as landings:
        for name in ("source_id", "weight", "gun_exit_phase", "energy_offset_eV"):
            np.testing.assert_array_equal(native[name], landings[name])
        np.testing.assert_array_equal(native["specimen_phase"], landings["after_phase"])
        np.testing.assert_array_equal(native["specimen_loss"], landings["after_loss"])
        ids, selected = native["source_id"], native["track_source_id"]
        order = np.argsort(ids)
        indices = order[np.searchsorted(ids[order], selected)]
        np.testing.assert_array_equal(ids[indices], selected)
        z = native["track_z_m"] * 1000
        paths, bands = native["tracks"], native["track_band"]
        surface = probe["specimen_top_mm"]
        stop, = np.flatnonzero(np.isclose(z, surface, atol=1e-9, rtol=0))
        if np.unique(selected).size != selected.size or z[0] != capture["gun_z_mm"]:
            raise ValueError("Invalid trajectory identifiers or gun-exit coordinate")
        reached = native["specimen_loss"][indices] == -1
        np.testing.assert_array_equal(reached, bands >= 0)
        np.testing.assert_array_equal(paths[0], native["gun_exit_phase"][indices, :4])
        np.testing.assert_array_equal(paths[stop, reached], native["specimen_phase"][indices[reached], :4])
        if not np.isnan(paths[stop, ~reached]).all():
            raise ValueError("An intercepted ray continues to the specimen")
        z, x = z[:stop+1], paths[:stop+1, :, 0] * 1e6
    return probe, capture, z, x, bands


def line_paths(ax, z, x, bands, limits, *, scale=1, origin=0, zscale=1):
    visible = (z >= limits[0]-1e-8) & (z <= limits[1]+1e-8)
    horizontal = (z[visible] - origin) * zscale
    values = x[visible] * scale
    for group, color in enumerate(COLORS):
        ax.plot(horizontal, values[:, bands == group], color=color, lw=.75,
                alpha=.68, solid_capstyle="round", zorder=2)
    ax.plot(horizontal, values[:, bands < 0], color=LOST, lw=.76, alpha=.68,
            linestyle=(0, (2, 2.4)), zorder=1)
    ax.set_xlim((np.asarray(limits) - origin) * zscale)
    ax.axhline(0, color="#73828a", lw=.55, zorder=.7)
    ax.grid(axis="y", color="#e1e6e8", lw=.55, zorder=0)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#a2adb4")
    ax.tick_params(length=3, color="#8796a0", labelsize=9)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))


def instrument_parts():
    comparison = read_json(HERE.parent / "probe_corrector_cescor_20261121/evidence/comparison.json")
    filename = comparison["fields"]["material"]["source_file"]
    parts = tomllib.loads((ROOT / "configs/instruments/column" / filename).read_text(encoding="utf-8"))["parts"]
    return {part["key"]: part for part in parts}


def shade(ax, part, origin, label=None):
    left = origin + part["local_start_z_mm"]
    right = origin + part["local_end_z_mm"]
    ax.axvspan(left, right, facecolor="#e8edf1", edgecolor="#d3dce1", alpha=.72, lw=.55, zorder=-1)
    if label:
        station_label(ax, (left+right)/2, label)


def station_label(ax, z, label, height=.99, color=INK):
    ax.text(z, height, label, transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=8.5, color=color, zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=.7) if height < .99 else None)


def station(ax, z, color, style, *, ymax=.91):
    ax.axvline(z, ymax=ymax, color=color, linestyle=style, lw=1.05, zorder=3)


def section(ax, name, ylabel):
    ax.set_title(name, loc="left", fontsize=12, color=INK, weight="bold", pad=9)
    ax.set_ylabel(ylabel)


def header(fig, title, subtitle):
    fig.text(.073, .964, title, fontsize=18, color=INK, weight="bold", va="top")
    fig.text(.074, .914, subtitle, fontsize=10, color="#536573", va="top")


def legend(fig, *, clipped):
    labels = ("0–0.35 R", "0.35–0.65 R", "0.65–0.90 R", "0.90–1.00 R")
    handles = [Line2D([], [], color=c, lw=2.2, label=s) for c, s in zip(COLORS, labels)]
    if clipped:
        handles.append(Line2D([], [], color=LOST, lw=1.5, ls="--", label="Intercepted at C2 aperture"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.065, .008),
               title="Entry radius at the C2 aperture (R = 50 µm)", title_fontsize=9,
               fontsize=9, ncol=5, frameon=False, handlelength=1.7, columnspacing=1.8,
               alignment="left")


def overview(probe, capture, z, x, bands, parts):
    fig = plt.figure(figsize=(13, 8))
    grid = fig.add_gridspec(2, 2, left=.074, right=.98, bottom=.165, top=.815,
                            width_ratios=(1.58, 1), height_ratios=(1.12, 1), hspace=.55, wspace=.19)
    full, condenser, tip = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    start, end = capture["gun_z_mm"], probe["specimen_top_mm"]
    final = probe["after"]
    header(fig, "Gun exit to specimen", f"Nanoprobe Normal    /    300 kV    /    100 µm C2 aperture"
           f"    /    Objective {final['objective_A']:+.6f} A    /    Specimen z = {end:.3f} mm")
    line_paths(full, z, x, bands, (start, end), scale=.001)
    full.set_ylim(-1.85, 2.1)
    section(full, "Full illumination column", "x (mm)")
    full.set_xlabel("Column z (mm)")
    full.set_xticks([start, 600, 800, 1000, 1200, 1400, end],
                    [f"{start:.0f}", "600", "800", "1000", "1200", "1400", f"{end:.2f}"])
    full.get_xticklabels()[-1].set_ha("right")
    for key, label in (("condenser_lens_1","C1 lens"), ("condenser_lens_2","C2 lens"),
                       ("condenser_lens_3","C3 lens"), ("adapter_lens","ADL"),
                       ("ac_deflector","AC scan"), ("mini_condenser","Mini"),
                       ("objective_upper_pole",None)):
        shade(full, parts[key], start, label)
    pole = start + parts["objective_upper_pole"]["local_center_z_mm"]
    station_label(full, pole, "Upper pole", .84)
    corrector = ("probe_hp2_hexapole", "probe_tl22_lens", "probe_tl21_lens",
                 "probe_hp1_hexapole", "probe_tl12_lens")
    for key in corrector:
        shade(full, parts[key], start)
    station_label(full, start + parts["probe_tl21_lens"]["local_center_z_mm"], "Probe corrector")
    station(full, end, INK, "-", ymax=1)
    foil = start + parts["condenser_aperture_2"]["local_center_z_mm"]
    station(full, foil, FOIL, ":")
    station_label(full, foil, "C2 aperture", .82)

    c2, c3, adl = [parts[key] for key in ("condenser_lens_2","condenser_lens_3","adapter_lens")]
    limits = start+c2["local_start_z_mm"], start+adl["local_end_z_mm"]
    line_paths(condenser, z, x, bands, limits)
    condenser.set_ylim(-225, 255)
    section(condenser, "Condenser gaps", "x (µm)")
    condenser.set_xlabel("Column z (mm)")
    for part, label in ((c2, "C2 lens"), (c3, "C3 lens"), (adl, "ADL")):
        shade(condenser, part, start, label)
    station(condenser, foil, FOIL, ":")
    station_label(condenser, foil-4, "C2 aperture", .78)
    waist = final["source_waists_mm"]["C2_C3_gap"][0][0]
    station(condenser, np.mean(waist), WAIST, "--")
    station_label(condenser, np.mean(waist), f"{waist[0]:.2f}–{waist[1]:.2f} mm", .15, WAIST)

    limits = end-.30, end
    line_paths(tip, z, x, bands, limits, origin=end, zscale=1000)
    tip.set_ylim(-10.2, 10.2)
    section(tip, "Final 300 µm of approach", "x (µm)")
    tip.set_xlabel("Distance from specimen upper surface (µm)")
    tip.set_xticks([-300, -200, -100, 0])
    tip.scatter(np.zeros((bands >= 0).sum()), x[-1, bands >= 0], s=6, color=INK, clip_on=False, zorder=3)
    station(tip, 0, INK, "-", ymax=1)
    tip.text(.96, .97, f"Surface D95   {final['D95_um']:.3f} µm", ha="right", va="top",
             transform=tip.transAxes, fontsize=9, color=INK)
    legend(fig, clipped=True)
    fig.savefig(HERE / "normal-objective-ray-diagram.png", dpi=175, facecolor="white")
    plt.close(fig)


def downstream(probe, capture, z, x, bands, parts):
    fig, (relay, approach) = plt.subplots(2, 1, figsize=(13, 8.1))
    fig.subplots_adjust(left=.074, right=.98, top=.815, bottom=.165, hspace=.53)
    start, end = capture["gun_z_mm"], probe["specimen_top_mm"]
    final = probe["after"]
    header(fig, "Probe corrector and scanning optics",
           "Static trajectories in the laboratory x–z plane    /    Scan pivot measured separately with ±1 µrad commands")
    ac = parts["ac_deflector"]
    line_paths(relay, z, x, bands, (1120, 1422))
    relay.set_ylim(-310, 340)
    section(relay, "Hexapoles, transfer lenses and unpowered scan windings", "x (µm)")
    relay.set_xlabel("Column z (mm)")
    for key, label in (("probe_hp2_hexapole","HP2"), ("probe_tl22_lens","TL22"),
                       ("probe_tl21_lens","TL21"), ("probe_hp1_hexapole","HP1"),
                       ("probe_tl12_lens","TL12")):
        shade(relay, parts[key], start, label)
    shade(relay, ac, start)
    for zz, label in zip(ac["interaction_centers_local_z_mm"], ("Upper scan\n0 A", "Lower scan\n0 A")):
        station(relay, start+zz, "#667789", ":")
        station_label(relay, start+zz, label)
    waists = final["downstream_radial_waists_mm"]
    for lo, hi in waists:
        mid = (lo+hi)/2
        station(relay, mid, WAIST, "--")
        station_label(relay, mid, f"{lo:.0f}–{hi:.0f}", .13, WAIST)
    relay.text(.015, .06, "Dashed red lines mark finite-source radial waists (mm)", color=WAIST,
               transform=relay.transAxes, fontsize=8.5)

    line_paths(approach, z, x, bands, (1345, end), scale=.001)
    approach.set_ylim(-2.15, 2.22)
    section(approach, "Last TL12 waist, two-lens front focus and virtual scan pivot", "x (mm)")
    approach.set_xlabel("Column z (mm)")
    for key, label in (("probe_tl12_lens","TL12"), ("ac_deflector","AC scan pair"),
                       ("mini_condenser","Mini Condenser"), ("objective_upper_pole","Upper objective pole")):
        shade(approach, parts[key], start, label)
    station(approach, end, INK, "-", ymax=1)
    lo, hi = waists[-1]
    station(approach, (lo+hi)/2, WAIST, "--")
    station_label(approach, (lo+hi)/2, f"Static waist\n{lo:.0f}–{hi:.0f} mm", .22, WAIST)
    calibration = final["downstream_calibration"]
    front, pivot = calibration["isolated_front_focal_plane_mm"], calibration["closest_common_pivot_mm"]
    station(approach, front, FOCUS, (0, (5, 2)))
    station(approach, pivot, PIVOT, (0, (4, 2, 1, 2)))
    for plane, shift, title, color in ((front,-5,"Front focus",FOCUS),(pivot,5,"AC pivot",PIVOT)):
        approach.annotate(f"{title}\n{plane:.3f} mm", xy=(plane,-1.72),
                          xytext=(plane+shift,-1.86), va="center", ha="right" if shift<0 else "left",
                          color=color, fontsize=8.5, arrowprops=dict(arrowstyle="-",color=color,lw=.8))
    legend(fig, clipped=False)
    fig.savefig(HERE / "normal-objective-ray-detail.png", dpi=175, facecolor="white")
    plt.close(fig)


def scan_waist_origin():
    result = read_json(EVIDENCE / "normal-scan-waist-origin.json")
    field = result["field"]
    blue, copper, gray = "#09646d", "#ac501e", "#586779"
    fig, (magnetic, radius) = plt.subplots(2, 1, figsize=(10.8, 7.4), sharex=True,
                                           gridspec_kw={"height_ratios": [1, 1.36]})
    fig.subplots_adjust(left=.102, right=.972, bottom=.096, top=.80, hspace=.19)
    fig.text(.102, .96, "TL12 focusing through the scan-coil region", color=INK,
             fontsize=18, weight="bold", va="top")
    fig.text(.103, .908, f"Nanoprobe Normal at 300 kV   ·   100 µm C2 aperture   ·   "
             f"{result['cohort_count']:,} matched particles   ·   Scan drive 0 A",
             color=gray, fontsize=10, va="top")
    field_z = np.asarray(field["z_mm"])
    for ax in (magnetic, radius):
        ax.axvspan(*result["TL12_span_mm"], fc="#eef1f3", zorder=-2)
        ax.axvspan(*result["AC_span_mm"], fc="#eef5f6", zorder=-2)
        for center in result["scan_centers_mm"]:
            ax.axvline(center, color="#7c9097", lw=1, ls=(0, (2, 3)), zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#9aa6aa")
        ax.tick_params(length=3, color="#89959c")
        ax.grid(axis="y", color="#cdd8db", lw=.65, alpha=.7)
    for z, label in zip(result["scan_centers_mm"], ("Upper scan", "Lower scan"), strict=True):
        magnetic.text(z, 1.032, f"{label}  {z:g} mm", transform=magnetic.get_xaxis_transform(),
                      ha="center", color=gray, fontsize=9)
    magnetic.text(np.mean(result["TL12_span_mm"]), 1.032, "TL12 housing",
                  transform=magnetic.get_xaxis_transform(), ha="center", color=gray, fontsize=9)
    magnetic.plot(field_z, field["TL12_Bz_T"], color=blue, lw=2.2, label="TL12 windings")
    magnetic.plot(field_z, field["total_Bz_T"], color=gray, lw=1.6, ls="--", label="Total installed field")
    magnetic.set_ylim(0, .57)
    magnetic.set_ylabel("Axial field Bz (T)")
    magnetic.legend(loc="upper left", frameon=False, fontsize=9)

    for key, label, color, style in (
            ("scan_off_installed", "Installed TL12", blue, "-"),
            ("TL12_off_from_gun", "TL12 off from gun exit", gray, "--"),
            ("TL12_off_from_AC_entrance", "Same beam at 1380 mm, TL12 field removed downstream", copper, "-")):
        data = result[key]
        radius.plot(data["z_mm"], data["rms_radius_um"], color=color, ls=style,
                    lw=2 if style == "-" else 1.6, label=label)
    original = result["scan_off_installed"]
    for number, (lo, hi) in enumerate(original["waist_brackets_mm"], start=1):
        z = np.asarray(original["z_mm"])
        w = np.asarray(original["rms_radius_um"])
        near = np.flatnonzero((z >= lo) & (z <= hi))
        i = near[np.argmin(w[near])]
        radius.scatter(z[i], w[i], s=24, color=blue, zorder=5)
        radius.annotate(f"Waist {number}", xy=(z[i], w[i]), xytext=(6, 5),
                        textcoords="offset points", color=blue, fontsize=9)
    radius.set_yscale("log")
    radius.set_ylim(.29, 1600)
    ticks = [.5, 2, 10, 50, 200, 800]
    radius.set_yticks(ticks, labels=[f"{t:g}" for t in ticks])
    radius.minorticks_off()
    radius.set_ylabel("RMS radius about centroid (µm)")
    radius.set_xlabel("Column z (mm)")
    radius.set_xlim(field_z[0], field_z[-1])
    radius.legend(loc="upper left", fontsize=8.6, frameon=True, edgecolor="none",
                  facecolor="white", framealpha=.9, labelspacing=.35)
    fig.savefig(HERE / "normal-scan-waist-origin.png", dpi=175, facecolor="white")
    plt.close(fig)


def plot_joint():
    from hashlib import sha256

    summary = read_json(EVIDENCE / "normal-joint-rays.json")
    source = EVIDENCE / "normal-joint-rays.npz"
    if sha256(source.read_bytes()).hexdigest() != summary["ray_archive_sha256"]:
        raise ValueError("The tracked particles differ from the comparison receipt")
    parts = instrument_parts()
    start, end = summary["gun_exit_mm"], summary["specimen_top_mm"]
    centers = summary["scan_centers_mm"]
    old_waists = np.asarray(summary["old"]["all_live"]["radial_waist_brackets_mm"])
    local = summary["local_interval_mm"]
    with np.load(source, allow_pickle=False) as data:
        order = np.argsort(data["source_id"])
        chosen = order[np.searchsorted(data["source_id"][order], data["track_source_id"])]
        np.testing.assert_array_equal(data["source_id"][chosen], data["track_source_id"])
        rays = {}
        for case in ("old", "joint"):
            z = data[case+"_track_z_mm"]
            tracks = data[case+"_tracks"]
            bands = data[case+"_track_band"]
            live = data[case+"_surface_loss"][chosen] == -1
            np.testing.assert_allclose(z[[0, -1]], [start, end], rtol=0, atol=3e-10)
            np.testing.assert_array_equal(tracks[0], data["gun_exit_phase"][chosen, :4])
            np.testing.assert_array_equal(tracks[-1, live], data[case+"_surface_phase"][chosen[live], :4])
            if not np.all(np.isnan(tracks[-1, ~live])) or not np.array_equal(bands >= 0, live):
                raise ValueError("Displayed trajectories do not agree with recorded contacts")
            rays[case] = (z, tracks[:, :, 0]*1e6, bands)
        local_z = {case: data[case+"_z_mm"] for case in rays}
        radius = {case: data[case+"_live_moments"][:, 0] for case in rays}

    names = {"old": "Archived vacuum windings", "joint": "Installed joint FEM"}
    fig, axes = plt.subplots(2, 1, figsize=(12.6, 7.5), sharex=True)
    fig.subplots_adjust(left=.08, right=.965, top=.805, bottom=.17, hspace=.38)
    header(fig, "Nanoprobe Normal  |  Gun exit to specimen",
           f"300 kV    /    C2 aperture Ø{summary['aperture_diameter_um']:.0f} µm    /    "
           f"Objective {summary['objective_A']:.4f} A    /    zero scan command    /    same {len(chosen)} source IDs")
    amplitude = max(np.nanmax(abs(x)) for _, x, _ in rays.values())*.001
    for ax, case in zip(axes, rays, strict=True):
        line_paths(ax, *rays[case], (start, end), scale=.001)
        section(ax, names[case], "x (mm)")
        ax.set_ylim(-amplitude*1.09, amplitude*1.39)
        for key, label in (("condenser_lens_1", "C1"), ("condenser_lens_2", "C2"),
                           ("condenser_lens_3", "C3"), ("adapter_lens", "ADL"),
                           ("ac_deflector", "AC scan"), ("mini_condenser", "Mini"),
                           ("objective_upper_pole", "Obj.")):
            shade(ax, parts[key], start, label)
        for key in ("probe_hp2_hexapole", "probe_tl22_lens", "probe_tl21_lens",
                    "probe_hp1_hexapole", "probe_tl12_lens"):
            shade(ax, parts[key], start)
        station_label(ax, start+parts["probe_tl21_lens"]["local_center_z_mm"], "Probe corrector")
        foil = start+parts["condenser_aperture_2"]["local_center_z_mm"]
        station(ax, foil, FOIL, ":")
        station_label(ax, foil, "C2 aperture", .78)
        ax.axvline(end, color=INK, lw=1.5)
    axes[-1].set_xlabel("Column z (mm)")
    axes[-1].set_xticks([start, 600, 800, 1000, 1200, 1400, end],
                        [f"{start:.0f}\nGun exit", "600", "800", "1000", "1200", "1400", f"{end:.1f}\nSpecimen"])
    axes[-1].get_xticklabels()[-1].set_ha("right")
    legend(fig, clipped=True)
    fig.savefig(HERE / "normal-joint-column-rays.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, (old, new, rms) = plt.subplots(3, 1, figsize=(10.4, 8.0), sharex=True,
                                       gridspec_kw={"height_ratios": [1, 1, .73]})
    fig.subplots_adjust(left=.11, right=.96, top=.805, bottom=.102, hspace=.49)
    header(fig, "TL12 to the scan deflectors",
           "Upper and lower scan windings at zero drive    /    actual projected rays above, "
           "all transmitted particles below")
    for ax, case in ((old, "old"), (new, "joint")):
        line_paths(ax, *rays[case], local)
        section(ax, names[case], "x (µm)")
    local_max = max(np.nanmax(abs(x[(z >= local[0]) & (z <= local[1])])) for z, x, _ in rays.values())
    for ax in (old, new):
        ax.set_ylim(-local_max*1.10, local_max*1.10)
    for case, color in (("old", WAIST), ("joint", FOCUS)):
        rms.plot(local_z[case], radius[case], color=color, lw=1.8, label=names[case])
    rms.set_yscale("log")
    rms.set_ylabel("RMS radius (µm)")
    rms.set_xlabel("Column z (mm)")
    rms.set_title("Two-dimensional beam envelope", loc="left", color=INK, fontsize=12,
                  weight="bold", pad=7)
    rms.spines[["top", "right"]].set_visible(False)
    rms.legend(loc="lower left", ncol=2, frameon=False, fontsize=8.5)
    for ax in (old, new, rms):
        shade(ax, parts["probe_tl12_lens"], start)
        left = start+parts["ac_deflector"]["local_start_z_mm"]
        right = start+parts["ac_deflector"]["local_end_z_mm"]
        ax.axvspan(left, right, color="#cce1e9", alpha=.27, zorder=-2, lw=0)
        for center in centers:
            ax.axvline(center, ls=":", color="#4e6d80", lw=1, zorder=1)
        ax.set_xlim(local)
    for ax in (old, rms):
        for lo, hi in old_waists:
            ax.axvline((lo+hi)/2, ls="--", color=WAIST, lw=.85, zorder=1)
    for lo, hi in old_waists:
        station_label(old, (lo+hi)/2, f"{(lo+hi)/2:.1f} mm", .18, WAIST)
    for z, label in ((start+parts["probe_tl12_lens"]["local_center_z_mm"], "TL12"),
                     (centers[0], "Upper scan"), (centers[1], "Lower scan")):
        station_label(old, z, label, .97)
    fig.savefig(HERE / "normal-joint-scan-rays.png", dpi=175, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--archived", action="store_true", help="Redraw the earlier vacuum-control figures")
    group.add_argument("--retrace", action="store_true", help="Recompute both particle captures before plotting")
    args = parser.parse_args()
    plt.rcParams.update({"font.family":"DejaVu Sans", "axes.labelcolor":INK, "xtick.color":INK,
                         "ytick.color":INK, "axes.labelsize":10, "path.simplify":False})
    if args.archived:
        data = load_tracks()
        hardware = instrument_parts()
        overview(*data, hardware)
        downstream(*data, hardware)
        scan_waist_origin()
    else:
        if args.retrace:
            from trace_normal_joint_field import run
            run()
        plot_joint()
