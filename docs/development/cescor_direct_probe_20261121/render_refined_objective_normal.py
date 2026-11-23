"""Verify the full-source objective replay and render its column, scan region and specimen."""

import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from condenser_pair_render import draw_probe, draw_rays
from condenser_source_fields import FEM, check_file, sha, write
from joint_relay_search import mono_roots
from refined_objective_normal import (
    COUNT, EVIDENCE, OBJECTIVE_SHA, PARENT, PARENT_SHA, WORK, RefinedObjectiveFields, objective_path,
)
from relay_transfer import PARTS, START


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case(pitch, step):
    folder = WORK / f"mesh-{pitch:g}mm"
    stem = f"column-{COUNT}-{step:g}-7"
    with np.load(folder / (stem + ".npz")) as result:
        rays = dict(result)
    return rays, read(folder / (stem + ".json"))


def summary(data, rec):
    z = data["z_mm"]
    stats = data["stats"]
    cov = stats[:, 1] + stats[:, 2]
    minima = mono_roots(z, cov)
    maxima = mono_roots(z, -cov)
    recorded = rec["all_column_waists_mm"]
    np.testing.assert_allclose(minima, recorded, rtol=0, atol=1e-9)
    old = 1474.7
    i = int(np.argmin(np.abs(z - old)))
    if abs(z[i] - old) > 1e-8:
        raise ValueError("Missing exact observation at the former waist")
    scan = PARTS["ac_deflector"]
    entrance, exit_z = 1000 * START + np.array([scan["local_start_z_mm"], scan["local_end_z_mm"]])
    rear = (z >= exit_z) & (z <= rec["surface_z_mm"])
    local = (minima >= entrance) & (minima <= exit_z)
    return dict(
        all_radial_waists_mm=minima.tolist(),
        scan_housing_waists_mm=minima[local].tolist(),
        all_extrema_after_TL12_mm={"minima": minima[minima > 1353.6].tolist(),
                                   "maxima": maxima[maxima > 1353.6].tolist()},
        scan_exit_to_surface_principal_waists_mm=[
            mono_roots(z[rear], stats[rear, j]).tolist() for j in (3, 4)
        ],
        at_old_waist=dict(z_mm=z[i], rms_radius_um=stats[i, 0],
                          rms_slope_um_per_mm=cov[i] / stats[i, 0]),
        surface=rec["surface"], near_surface_0p4mm_D95_um=[rec["near_surface"][0]["D95_um"],
                                                          rec["near_surface"][-1]["D95_um"]],
    )


def validate_surface(data, rec):
    live = (data["loss"] == -1) & (data["weights"] > 0)
    xy = 1e6 * data["surface_phase"][live, :2]
    amps = data["weights"][live] * float(data["source_A"])
    centre = np.average(xy, weights=amps, axis=0)
    radius = np.linalg.norm(xy - centre, axis=1)
    order = np.argsort(radius)
    i = np.searchsorted(np.cumsum(amps[order]), .95 * amps.sum())
    s = rec["surface"]
    measured = [amps.sum() * 1e6, 2 * radius[order[i]],
                np.sqrt(np.average(radius**2, weights=amps)),
                amps[np.linalg.norm(xy, axis=1) <= .05].sum() * 1e9]
    np.testing.assert_allclose(measured, [s[k] for k in ("specimen_uA","D95_um","RMS_radius_um","axis_50nm_nA")],
                               rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(centre, s["centroid_um"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(rec["ledger"]["specimen_A"] + sum(rec["ledger"]["intercepted_A"].values()),
                               data["source_A"], rtol=1e-12, atol=1e-18)
    if np.count_nonzero(data["weights"] > 0) != COUNT or len(np.unique(data["source_ids"][live])) != s["count"]:
        raise ValueError("Source identities or transmitted sample count changed")
    return float(amps.sum()**2 / (amps @ amps))


def comparisons(primary, control):
    for key in ("source_ids","weights","series_A"):
        np.testing.assert_array_equal(primary[key], control[key])
    common = (primary["weights"] > 0) & (primary["loss"] == -1) & (control["loss"] == -1)
    shifts = np.linalg.norm(primary["surface_phase"][common, :2] - control["surface_phase"][common, :2], axis=1)
    return dict(common_count=int(common.sum()),
                max_matched_gun_exit_energy_change_eV=float(np.max(np.abs(
                    primary["energy_offset_eV"] - control["energy_offset_eV"]))),
                changed_transmissions=int(np.count_nonzero((primary["loss"] == -1) != (control["loss"] == -1))),
                max_matched_specimen_displacement_nm=float(1e9*shifts.max()))


def scan_objective_samples():
    prior = read(PARENT.parent / "condenser-scan-region.json")["on_axis"]
    stations = np.array([1380., 1387.5, 1408., 1412.5, 1420.])
    samples = {}
    for pitch in OBJECTIVE_SHA:
        field = RefinedObjectiveFields(pitch)
        values = field.objective(stations * .001)[:, 0] * 1000
        reference = [prior[str(z)][f"objective_{pitch:g}mm_mT"] for z in stations]
        np.testing.assert_allclose(values, reference, atol=1e-8, rtol=1e-10)
        samples[f"{pitch:g}mm"] = dict(zip(map(str, stations), values))
    return samples


def scan_detail(gun, cur, previous, record, measurements, old_waist):
    z, old_z = cur["z_mm"], previous["z_mm"]
    new_roots = measurements["all_extrema_after_TL12_mm"]["minima"]
    nearest = min(new_roots, key=lambda x: abs(x - old_waist)) if new_roots else None
    lo, hi = 1340., min(record["surface_z_mm"], max(1497., (nearest or old_waist) + 11))
    observed = (z >= lo) & (z <= hi)
    shown = np.arange(0, 64, 2)
    selected = np.linspace(0, COUNT - 1, 64, dtype=int)[shown]
    np.testing.assert_array_equal(gun["ids"][selected], cur["selected_ids"][shown])
    angle = np.arctan2(gun["launch"][selected, 3], gun["launch"][selected, 2])
    colors = plt.cm.twilight_shifted((angle + np.pi) / (2*np.pi))
    tracks = cur["paths_m"][observed][:, shown] * 1e6
    scale = max(np.nanmax(abs(tracks)), .001) * 1.20
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11.1, 7.7), layout="constrained",
                             gridspec_kw={"height_ratios":[1.,1.,.86]})
    fig.suptitle("TL12 through the scan windings and the downstream waist", fontsize=15, weight="bold")
    ac, tl = PARTS["ac_deflector"], PARTS["probe_tl12_lens"]
    def interval(part):
        return 1000 * START + np.array([part["local_start_z_mm"], part["local_end_z_mm"]])
    upper, lower = 1000 * START + np.array(ac["interaction_centers_local_z_mm"])
    for ax in axes:
        ax.axvspan(*interval(tl), color="#eae6ee", zorder=-4)
        ax.axvspan(*interval(ac), color="#e4f1f1", zorder=-4)
        for where in (upper, lower):
            ax.axvline(where, color="#759196", ls=":", lw=.85, zorder=-2)
        ax.set_xlim(lo, hi)
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    for j, label in enumerate(("x","y")):
        ax = axes[j]
        for k, c in enumerate(colors):
            ax.plot(z[observed], tracks[:,k,j], color=c, lw=.82, alpha=.78)
        ax.axhline(0, color="#6a747a", lw=.5, zorder=-3)
        ax.set(ylim=(-scale,scale), ylabel=label+" (µm)")
    for where, label in ((np.mean(interval(tl)),"TL12"), (upper,"Upper scan"), (lower,"Lower scan")):
        axes[0].text(where,.985,label,transform=axes[0].get_xaxis_transform(),ha="center",va="top",
                     fontsize=9,color="#274553")
    axes[0].set_title("Same 32 original ray IDs in x and y  ·  AC scan drive off",loc="left",fontsize=10)
    ax = axes[-1]
    ax.plot(old_z,previous["stats"][:,0],c="#b3643a",ls="--",lw=1.7,
            label=f"Original objective mesh · waist {old_waist:.2f} mm")
    desc = f" · waist {nearest:.2f} mm" if nearest is not None else ""
    ax.plot(z,cur["stats"][:,0],c="#006f71",lw=2,label="Refined objective mesh"+desc)
    for where, col in ((old_waist, "#b3643a"),(nearest,"#006f71")):
        if where is not None:
            ax.axvline(where,c=col,ls=":",lw=1.)
    ax.set_ylim(0, max(np.max(cur["stats"][observed,0]),
                        np.max(previous["stats"][(old_z >= lo)&(old_z <= hi),0]))*1.34)
    ax.set(ylabel="RMS radius (µm)", xlabel="Distance from effective source  z (mm)")
    ax.set_title("Current-weighted beam envelope  ·  all 32,768 original particles", loc="left",fontsize=10)
    ax.legend(fontsize=8,loc="upper right")
    fig.savefig(EVIDENCE / "tl12-scan-rays.png",dpi=175)
    plt.close(fig)


def render():
    check_file(PARENT / "working-point.json", PARENT_SHA)
    parent = read(PARENT / "working-point.json")
    folder = WORK / "mesh-0.5mm"
    old_record = read(PARENT / "tracking.json")
    for name in ("column-and-landings.npz","fresh-emission-and-exit.npz","tracking.json"):
        check_file(PARENT / name, parent["evidence_sha256"][name])
    for pitch, digest in OBJECTIVE_SHA.items():
        check_file(objective_path(pitch), digest)
    with np.load(folder / f"tip-{COUNT}.npz") as src, np.load(PARENT / "column-and-landings.npz") as old:
        gun, old_data = dict(src), dict(old)
    primary, rec = case(.5, .05)
    same_fem, half_step = case(.5, .025)
    coarser, other_mesh = case(1., .05)
    binding = read(folder / "objective-binding.json")
    mesh_binding = read(WORK / "mesh-1mm/objective-binding.json")
    assert binding["objective_sha256"] == OBJECTIVE_SHA[.5] and mesh_binding["objective_sha256"] == OBJECTIVE_SHA[1.]
    assert binding["corrector_sha256"] == parent["downstream_tracking_field_sha256"] == sha(FEM)
    effective_count = validate_surface(primary, rec)
    validate_surface(same_fem, half_step)
    validate_surface(coarser, other_mesh)
    np.testing.assert_array_equal(primary["series_A"], parent["signed_current_A"])
    np.testing.assert_array_equal(primary["source_ids"], gun["ids"])
    np.testing.assert_array_equal(primary["weights"], gun["weight"])
    np.testing.assert_array_equal(primary["energy_offset_eV"], gun["energy_offset_ev"])
    np.testing.assert_array_equal(same_fem["energy_offset_eV"], primary["energy_offset_eV"])
    with np.load(WORK / f"mesh-1mm/tip-{COUNT}.npz") as control_gun:
        np.testing.assert_array_equal(coarser["energy_offset_eV"], control_gun["energy_offset_ev"])
    assert str(gun["signature"]) == rec["gun_signature"] == half_step["gun_signature"]
    assert len({rec["gun_signature"], other_mesh["gun_signature"], old_record["gun_signature"]}) == 3
    assert rec["surface_z_mm"] == old_record["surface_z_mm"]
    stats = summary(primary,rec)
    check = summary(same_fem,half_step)
    slower = summary(coarser,other_mesh)
    step_difference = comparisons(primary,same_fem)
    if step_difference["changed_transmissions"]:
        raise ValueError("Halving the step changed particle transmission")
    np.testing.assert_allclose(rec["all_column_waists_mm"], half_step["all_column_waists_mm"],
                               atol=.02, rtol=0)
    np.testing.assert_allclose(rec["surface"]["D95_um"],half_step["surface"]["D95_um"],atol=.001,rtol=0)
    label = "0.5 mm objective refinement"
    draw_rays(gun,primary,rec,output_dir=EVIDENCE,full_only=True,field_label=label)
    pitch_nm = draw_probe(primary,rec,output_dir=EVIDENCE,field_label=label)
    old_waist = min(old_record["all_column_waists_mm"],key=lambda v:abs(v-1474.7))
    scan_detail(gun,primary,old_data,rec,stats,old_waist)
    for src,dst in ((f"tip-{COUNT}.npz","fresh-emission-and-exit.npz"),
                    (f"column-{COUNT}-0.05-7.npz","column-and-landings.npz"),
                    (f"column-{COUNT}-0.05-7.json","tracking.json")):
        shutil.copyfile(folder / src,EVIDENCE / dst)
    checks = dict(original_objective=dict(surface=old_record["surface"], waists_mm=old_record["all_column_waists_mm"],
                                          **comparisons(primary,old_data)),
                  half_tracking_step=dict(step_mm=.025, **check, **step_difference),
                  one_mm_objective=dict(mesh_mm=1., **slower, **comparisons(primary,coarser)))
    write(EVIDENCE / "matched-controls.json",checks)
    info = dict(mode="Three-condenser Nanoprobe Normal",parent_file="../condenser-pair/working-point.json",
                launch="Fresh physical effective source at z=0", **binding,
                source_A=rec["source_A"], emitted_positive_count=COUNT,
                specimen_count=rec["surface"]["count"], effective_specimen_count=effective_count,
                objective_field_file=objective_path(.5).name,
                objective_mesh_refined_z_mm=[1280.,1440.], objective_binding_scope="Gun and complete downstream trajectory",
                focus_control="Inherited currents, fixed specimen upper surface",
                signed_current_A=primary["series_A"].tolist(), winding_owners=parent["winding_owners"],
                physical_turns=parent["physical_turns"], corrector_A=parent["corrector_A"],
                corrector_field_sha256=sha(FEM), condenser_material_sha256=parent["operating_point"]["material_sha256"],
                scan_calibration_sha256=parent["scan_calibration_sha256"],
                C2_aperture_diameter_um=100., C2_foil=rec["C2_aperture"], current_ledger=rec["ledger"],
                objective_axial_mT_by_mesh_and_station_mm=scan_objective_samples(),
                source_images=rec["source_images"], source_signature=rec["gun_signature"],
                probe_density_pixel_nm=pitch_nm, **stats)
    assets = ("objective-0.5mm.npz","objective-1mm.npz","fresh-emission-and-exit.npz","column-and-landings.npz",
              "tracking.json","matched-controls.json","full-column-rays.png","tl12-scan-rays.png","specimen-probe.png")
    info["evidence_sha256"] = {name:sha(EVIDENCE/name) for name in assets}
    write(EVIDENCE/"working-point.json",info)
    print(json.dumps(dict(surface=stats["surface"], waist=stats["all_extrema_after_TL12_mm"],
                          old_plane=stats["at_old_waist"], step=check["surface"], mesh=slower["surface"]),indent=2))


if __name__ == "__main__":
    render()
