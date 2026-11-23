"""Count shared transverse beam waists over the physical probe-corrector intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_source_fields import write
from joint_relay_search import mono_roots
from material_relay_column import Column, Field, source_and_electrodes
from material_relay_fields import SAVE, WORK
from relay_transfer import PARTS, advance
from trace_normal_joint_field import beam_moments


MAIN = (("HP2", "probe_hp2_hexapole"), ("TL22", "probe_tl22_lens"),
        ("TL21", "probe_tl21_lens"), ("HP1", "probe_hp1_hexapole"), ("TL12", "probe_tl12_lens"))
TRIMS = (("QPH2", "probe_qph2_quadrupole"), ("HPC", "probe_hpc_hexapole"),
         ("QPC", "probe_qpc_quadrupole"), ("QPH1", "probe_qph1_quadrupole"),
         ("HPOL", "probe_hpol_hexapole"), ("QPOL", "probe_qpol_quadrupole"))
DEFLECTORS = (("DPH2", "probe_dph2_deflector"), ("DP22", "probe_dp22_deflector"),
              ("DP21", "probe_dp21_deflector"), ("DPH1", "probe_dph1_deflector"),
              ("DP11", "probe_dp11_deflector"))


def extent(key):
    part = PARTS[key]
    return [450. + part["local_start_z_mm"], 450. + part["local_end_z_mm"]]


def measure(folder, pitch=.125):
    record = json.loads((folder / "tracking.json").read_text())
    if any(record["AC_command_static"]):
        raise ValueError("The static crossover count requires zero scan drive")
    first, last = extent("probe_hp2_hexapole")[0], extent("probe_tl12_lens")[1]
    upstream, downstream = extent("adapter_lens")[0], extent("ac_deflector")[1]
    reference = np.array([z for z in record["waists_mm"] if upstream < z < downstream])
    close_stations = (reference[:, None] + [-2., -.02, 0., .02, 2.]).ravel()
    stations = np.unique(np.r_[np.arange(upstream, downstream, pitch), first, last, downstream,
                               close_stations[(close_stations > upstream) & (close_stations < downstream)]])
    field = Field(WORK / record["field"]["source"], step=record["field"]["step_mm"])
    column = Column(source_and_electrodes(), field, Path(record["gun_source"]), record["count"],
                    record["step_mm"], observation_mm=stations)
    stops = column.indices(stations * .001)
    z = column.m.z[stops] * 1000.
    np.testing.assert_allclose(z, stations, rtol=0., atol=1e-8)
    with np.load(folder / "column-and-landings.npz") as archived:
        np.testing.assert_array_equal(column.t.ids, archived["source_ids"])
        np.testing.assert_array_equal(column.t.weights, archived["weights"])
        selected = np.searchsorted(column.t.ids, archived["selected_ids"])
        np.testing.assert_array_equal(column.t.ids[selected], archived["selected_ids"])
    weights = column.t.weights
    if np.any(weights <= 0.) or len(np.unique(column.t.ids)) < 1000:
        raise ValueError("At least 1000 distinct positive-current source particles are required")
    phase, lost, start = column.t.input, column.loss, 0
    moments, paths = [], []
    live_counts = []
    for group in np.array_split(stops, (len(stops) + 23) // 24):
        states = advance(column.t, column.full, phase, column.en, column.phi0, lost, start, group)
        for phase, lost in states:
            live = lost == -1
            if live.sum() < 1000:
                raise ValueError("Insufficient original particles for the transverse analysis")
            live_counts.append(int(live.sum()))
            moments.append(beam_moments(phase, live, weights))
            paths.append(np.where(live[selected, None], phase[selected, :2], np.nan))
        start = group[-1]
    stats = np.array(moments)
    rates = (stats[:, 1] + stats[:, 2], stats[:, 3], stats[:, 4])
    radial, lower, upper = [mono_roots(z, rate) for rate in rates]
    shared = []
    for root in radial:
        small = lower[abs(lower - root) <= pitch]
        large = upper[abs(upper - root) <= pitch]
        before, after = np.searchsorted(z, [root - pitch, root + pitch])
        if len(small) != 1 or len(large) != 1 or before == 0 or after >= len(z):
            continue
        if stats[before-1, 4] >= 0. or stats[after, 3] <= 0.:
            continue
        ownership = [name for name, key in MAIN + TRIMS + DEFLECTORS if extent(key)[0] <= root <= extent(key)[1]]
        around = []
        for delta in (-2., 0., 2.):
            i = np.argmin(abs(z - (root+delta)))
            around.append(dict(z_mm=float(z[i]), rms_radius_nm=float(stats[i, 0]*1e3)))
        shared.append(dict(z_mm=float(root), directional_roots_mm=[small.item(), large.item()],
                           components=ownership, nearby_direct_observations=around))
    intervals = {"HP2 entrance to HP1 exit": [first, extent("probe_hp1_hexapole")[1]],
                 "HP2 entrance to TL12 exit (shaded corrector)": [first, last],
                 "ADL entrance to TL12 exit": [upstream, last],
                 "ADL entrance through lower scan exit": [upstream, downstream]}
    counts = {name: dict(bounds_mm=bounds, count=sum(bounds[0] <= row["z_mm"] <= bounds[1] for row in shared))
              for name, bounds in intervals.items()}
    result = dict(original_particles=len(weights), minimum_surviving=min(live_counts),
                  static_scan_A=record["AC_command_static"], source_field_sha256=field.receipt["solution_sha256"],
                  observation_pitch_mm=pitch, integration_step_limit_mm=record["step_mm"],
                  definition="Radial variance has a minimum and both eigenvalues of half the derivative of the 2D position covariance change from negative to positive within one observation interval of that minimum.",
                  dense_checked_bounds_mm=[upstream, downstream], recorded_planes=len(z),
                  method="Native propagation from the saved same-field effective-source handoff, with physical contacts and original weights.",
                  radial_minima_mm=radial.tolist(), lower_eigenvalue_roots_mm=lower.tolist(),
                  upper_eigenvalue_roots_mm=upper.tolist(), shared_crossovers=shared, intervals=counts,
                  post_scan_waists_from_full_trace_mm=[w for w in record["waists_mm"]
                      if downstream < w < extent("mini_condenser")[0]])
    write(folder / "corrector-crossover-count.json", result)
    return z, np.asarray(paths), stats, shared, first, last


def draw(folder, z, paths, stats, shared, first, last):
    internal = [row for row in shared if first <= row["z_mm"] <= last]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.1), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [.68, 1, 1, 1.18]})
    count = len(internal)
    fig.suptitle(f"Probe corrector  |  {count} shared crossover" + ("s" if count != 1 else ""),
                 weight="bold", fontsize=17)
    strip, ax_x, ax_y, envelope = axes
    for y_band, parts, band_color in ((1.55, MAIN, "#9fbac6"), (.53, TRIMS, "#d4dfe1"),
                                     (0., DEFLECTORS, "#e5e8e9")):
        for name, key in parts:
            a, b = extent(key)
            color = "#e9d6a8" if name == "QPC" else band_color
            # HP2 and TL22 overlap axially. Stagger their bands to show both intervals.
            y = 1.04 if name == "TL22" else y_band
            strip.broken_barh([(a,b-a)],(y,.45),facecolors=color,edgecolors="white",linewidth=.6)
            strip.text((a+b)/2,y+.22,name,ha="center",va="center",fontsize=8)
    strip.set(ylim=(-.13,2.08), yticks=[])
    strip.axis("off")
    ref = np.argmin(abs(z - (internal[0]["z_mm"] - 12. if internal else first)))
    colors = plt.cm.twilight_shifted((np.arctan2(paths[ref,:,1],paths[ref,:,0])+np.pi)/(2*np.pi))
    keep = (z >= first-11.) & (z <= last+3.)
    for dim, ax in enumerate((ax_x, ax_y)):
        for index in range(0, paths.shape[1], 2):
            ax.plot(z[keep],paths[keep,index,dim]*1e6,color=colors[index],lw=.72,alpha=.78)
        ax.set_ylabel(("x" if dim == 0 else "y") + "  (µm)")
    envelope.semilogy(z[keep],stats[keep,0]*1e3,color="#184d5e",lw=1.9)
    for row in internal:
        root = row["z_mm"]
        point = row["nearby_direct_observations"][1]
        envelope.scatter(point["z_mm"],point["rms_radius_nm"],color="#a84230",s=30,zorder=3)
        envelope.annotate(f"{root:.3f} mm", (point["z_mm"], point["rms_radius_nm"]),
                          xytext=(12,11), textcoords="offset points",color="#943b2c",weight="bold")
        for ax in axes[1:]:
            ax.axvline(root,color="#a84230",lw=1,ls="--")
    for ax in axes[1:]:
        ax.axvline(first,color="#63818c",lw=.9,ls=":")
        ax.axvline(last,color="#63818c",lw=.9,ls=":")
        ax.grid(axis="y",alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    envelope.set(xlabel="Distance from the effective source (mm)",ylabel="RMS radius  (nm)",xlim=(first-11.,last+3.))
    fig.savefig(folder / "corrector-crossover.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    args = parser.parse_args()
    output = SAVE / args.label
    draw(output, *measure(output))
