"""Compare the two gun-exit populations and draw the original ray paths."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from gun_focus import BASELINE_KV, CASE, HERE, INPUTS, prepare, set_voltage, write_json
from temsim.physics.beam_statistics import transverse_beam_statistics, _weighted_quantile


def exit_metrics(phase, weight, alive, current):
    stats = transverse_beam_statistics(*phase[:, :4].T, alive=alive, weights=weight)
    passed = weight[alive]
    return dict(original_count=len(weight), surviving_count=len(passed),
        effective_surviving_count=float(passed.sum()**2/np.dot(passed, passed)),
        exit_current_uA=float(current*passed.sum()*1e6), alpha95_urad=stats.convergence_95_rad*1e6,
        alpha_rms_urad=stats.convergence_rms_rad*1e6,
        largest_sampled_angle_urad=stats.convergence_edge_rad*1e6,
        rms_radius_um=stats.radius_rms_m*1e6,
        rms_envelope_slope_urad=stats.radial_position_angle_covariance_m_rad/stats.radius_rms_m*1e6)


def measure(parent, tuned):
    for key in ("ids", "weight", "launch", "selected_ids", "gun_z_mm"):
        np.testing.assert_array_equal(parent[key], tuned[key])
    assert float(parent["emitted_A"]) == float(tuned["emitted_A"])
    assert np.isfinite(parent["phase"]).all() and np.all(parent["phase"][:, 4] > 0)
    before_alive = np.ones(len(parent["ids"]), dtype=bool)  # launch_in_field rejects gun losses.
    metrics = {}
    for label, data, alive in (("before", parent, before_alive), ("after", tuned, tuned["alive"])):
        metrics[label] = {}
        for n in (1024, 8192, 16384):
            metrics[label][str(n)] = exit_metrics(data["phase"][:n], data["weight"][:n],
                                                   alive[:n], float(data["emitted_A"]))
    xy = tuned["dpa_xy_m"][tuned["dpa_passed"]]
    weight = tuned["weight"][tuned["dpa_passed"]]
    radius = np.linalg.norm(xy-np.average(xy, axis=0, weights=weight), axis=1)
    metrics["foil"] = dict(D95_nm=_weighted_quantile(radius, weight, .95)*2e9,
                           rms_radius_nm=np.sqrt(np.average(radius**2, weights=weight))*1e9,
                           current_uA=float(tuned["emitted_A"])*weight.sum()*1e6,
                           aperture_interceptions=int(np.count_nonzero(~tuned["dpa_passed"])))
    return metrics


def ray_diagram(parent, tuned, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.5), sharex="col",
                             gridspec_kw={"width_ratios": [1, 1.18]})
    fig.subplots_adjust(left=.087, right=.98, top=.81, bottom=.16, hspace=.45, wspace=.23)
    fig.text(.087, .943, "First source image at the gun aperture", fontsize=23, weight="bold")
    fig.text(.087, .882, "Nanoprobe Normal   |   Matched source particles   |   Same material magnetic field",
             fontsize=12, color="#365264")
    z = parent["gun_z_mm"]
    local = z <= 50.
    common_limit = 1.09*max(np.abs(a["gun_paths_m"][local]).max() for a in (parent, tuned))*1e6
    colors = ("#216f91", "#bb6729")
    for row, (label, data, kv) in enumerate((("before", parent, BASELINE_KV),
                                              ("after", tuned, float(tuned["lens_kV"])))):
        values = data["gun_paths_m"]*1e6
        stat = metrics[label]["16384"]
        for column, ax in enumerate(axes[row]):
            ax.axvspan(14, 22, color="#dde8ec", zorder=-3)
            ax.axvspan(30, 370, color="#f6f1e6", zorder=-3)
            ax.axhline(0, color="#738996", lw=.65)
            # Identical IDs across both cases and both projections.
            for d, color in enumerate(colors):
                ax.plot(z, values[:, ::3, d], color=color, lw=.62, alpha=.46)
            ax.axvline(24, color="#853442", ls=(0,(4,3)), lw=1.35)
            ax.set_xlim((-1,50) if column==0 else (-5,452))
            if column == 0:
                ax.set_ylim(-common_limit*1.14, common_limit*1.14)
            else:
                limit = np.nanmax(np.abs(values))*1.11
                ax.set_ylim(-limit,limit)
                ax.set_xticks([0,100,200,300,400,450])
                ax.axvline(450, color="#314958", lw=1.)
            ax.grid(axis="y", color="#c3cfd4", alpha=.54, lw=.65)
            if row == 1:
                ax.set_xlabel("Distance from effective source (mm)")
        axes[row, 0].set_ylabel("Transverse position (µm)")
        heading = "Before" if row==0 else "After"
        axes[row, 0].set_title(f"{heading}   +{kv:g} kV   |   first 50 mm", loc="left",
                               fontsize=12, weight="bold", pad=6)
        axes[row, 1].set_title(f"Complete gun   |   exit α95 = {stat['alpha95_urad']:.2f} µrad",
                               loc="left", fontsize=12, weight="bold", pad=6)
    axes[0, 0].text(24.7,common_limit*.97,"24 mm",fontsize=9,color="#853442")
    fig.legend(handles=[Line2D([],[],color=colors[0],label="x projection"),
        Line2D([],[],color=colors[1],label="y projection"),
        Patch(facecolor="#dde8ec",label="Gun-lens electrode"),
        Line2D([],[],color="#853442",ls="--",label="Gun aperture"),
        Patch(facecolor="#f6f1e6",label="Acceleration tube")],
        loc="lower center",bbox_to_anchor=(.53,.025),frameon=False,ncol=5,columnspacing=1.5,fontsize=10)
    fig.savefig(HERE / "gun-ray-diagram.png",dpi=180,facecolor="white")
    plt.close(fig)


def capture_potentials(kv):
    gun, _ = prepare()
    z = np.unique(np.r_[np.linspace(0, 50, 1001), 370, 445, 450])
    points = np.column_stack([np.zeros_like(z),np.zeros_like(z),z*.001])
    voltages = [BASELINE_KV, kv]
    potentials, identities = [], []
    for voltage in voltages:
        identities.append(set_voltage(gun, voltage))
        potentials.append(gun.base_electric_field.potential_rise_v_at_global_positions(points))
    common_b = gun._instrument_magnetic_field.field_at_global_positions_t(points)
    np.savez_compressed(HERE / "gun-axis-potentials.npz", z_mm=z, voltages_kV=voltages,
                        axis_potential_from_tip_V=potentials, axis_magnetic_T=common_b,
                        field_sha256=identities)


def source_figure(kv):
    fig, (left, right) = plt.subplots(1,2,figsize=(12.5,4.8))
    fig.subplots_adjust(left=.078,right=.98,top=.78,bottom=.17,wspace=.29)
    fig.text(.078,.907,"How the gun lens moves the source image",fontsize=21,weight="bold")
    for case, style, color, name in (("before","-","#475d6d","+5.2 kV"),
                                      ("after","-","#007773",f"+{kv:g} kV")):
        with np.load(HERE / f"{case}-mesh-16-step-1.npz") as r:
            i = np.flatnonzero(r["amplitudes_urad"]==1.).item()
            z, b = r["z_mm"], r["B_m_per_rad"][i]
            left.plot(z, b[:,0,0]*2e3, style, color=color, lw=2, label=name)
    left.axhline(0, color="#4f626c", lw=.8)
    left.axvline(24, color="#853442", ls="--", lw=1)
    left.scatter([24],[0],c="#007773",s=35,zorder=5)
    left.annotate("First real crossover",xy=(24,0),xytext=(12,-.28),
        color="#005a55",arrowprops=dict(arrowstyle="->",color="#005a55"))
    left.set(xlim=(0.,46.),ylim=(-.70,.68),xlabel="Distance from effective source (mm)",
             ylabel="Signed x separation of the ±1 µrad rays (nm)")
    left.legend(frameon=False,loc="lower left")
    left.set_title("Same-point angular test",loc="left",fontsize=12,weight="bold")
    with np.load(HERE / "gun-axis-potentials.npz") as g:
        z = g["z_mm"]
        for v,p,c in zip(g["voltages_kV"],g["axis_potential_from_tip_V"],("#475d6d","#007773")):
            right.plot(z, p/1000, color=c, lw=2, label=f"+{v:g} kV")
    right.axvspan(14.,22.,color="#dde8ec",zorder=-3)
    right.axvline(24.,color="#853442",ls="--",lw=1)
    right.axhline(0.,color="#4f626c",lw=.8)
    right.set(xlim=(0,26),ylim=(-.25,12.5),xlabel="Distance from effective source (mm)",
              ylabel="On-axis potential relative to tip (kV)")
    right.legend(frameon=False,loc="upper left")
    right.set_title("Electric potential on the gun axis",loc="left",fontsize=12,weight="bold")
    for ax in (left,right):
        ax.grid(axis="y",lw=.55,alpha=.5)
    fig.savefig(HERE / "source-image-and-field.png",dpi=170,facecolor="white")
    plt.close(fig)


def main():
    record = json.loads((INPUTS / "index.json").read_text())["trials"][CASE]
    with np.load(INPUTS / record["particle_source"]["file"]) as parent, np.load(HERE / "gun-exit-particles.npz") as tuned:
        metrics = measure(parent, tuned)
        controls = dict(parent_case=CASE, parent_transport_index="../c1-zoom/transport-inputs/index.json",
            gun_lens_voltage_reference="tip", gun_lens_before_kV=BASELINE_KV,
            gun_lens_after_kV=float(tuned["lens_kV"]), extractor_kV=4.,
            gun_aperture_diameter_mm=4., gun_aperture_z_mm=24., gun_exit_z_mm=450.,
            accelerating_voltage_kV=300., accelerator_divider_start="gun_lens",
            unchanged_magnetic_field_sha256=str(tuned["magnet_id"]),
            source_input_sha256=str(tuned["source_id"]), tuned_electric_field_sha256=str(tuned["electric_id"]),
            tuned_gun_exit="gun-exit-particles.npz")
        verification = {}
        for fn in sorted(HERE.glob("*-mesh-*-step-*.json")):
            v = json.loads(fn.read_text())
            one = next(r for r in v["angular_checks"] if r["perturbation_urad"]==1.)
            verification[fn.stem] = dict(X_mm=[r["z_mm"] for r in one["roots_x"]],
                Y_mm=[r["z_mm"] for r in one["roots_y"]], aperture_pair_separation_nm=one["foil_separation_nm"])
        field_check = dict(max_ray_radius_um=float(tuned["maximum_radius_m"])*1e6,
                           field_valid_radius_um=float(tuned["magnetic_validity_radius_m"])*1e6,
                           max_exit_energy_error_eV=float(tuned["max_energy_error_eV"]),
                           max_matched_exit_energy_difference_eV=float(np.max(
                               np.abs(parent["energy_offset_ev"][tuned["alive"]]
                                      - tuned["energy_offset_ev"][tuned["alive"]]))))
        refined = json.loads((HERE / "matched-refinement.json").read_text())
        assert refined["field_sha256"] == json.loads((HERE / "after-mesh-22-step-0.5.json").read_text())["field_sha256"]
        write_json(HERE/"operating-point.json", dict(controls=controls, results=metrics,
            angular_verification=verification, transport_checks=field_check, paired_refinement=refined))
        ray_diagram(parent,tuned,metrics)
    capture_potentials(controls["gun_lens_after_kV"])
    source_figure(controls["gun_lens_after_kV"])


if __name__ == "__main__":
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#60727d",
        "axes.labelcolor":"#273d4b","text.color":"#1b3342",
        "xtick.color":"#415663","ytick.color":"#415663"})
    main()
