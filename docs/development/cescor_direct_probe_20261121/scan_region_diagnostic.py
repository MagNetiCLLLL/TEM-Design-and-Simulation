"""Audit the idle scan station in the November 22 two-condenser working point."""

import argparse
from hashlib import sha256
import json

import numpy as np
from scipy.interpolate import make_interp_spline

from condenser_source_fields import EV, FEM, FINAL, HERE, OBJ, OWNER, check_file, sha, write
from condenser_pair_study import EV as PAIR_EVIDENCE


OUT = HERE.parents[2] / "outputs/condenser-pair-scan-audit-20261122"
OUT.mkdir(parents=True, exist_ok=True)
ENTRY, EXIT = 1380., 1420.
N = 32768


def bore_coefficients(radius, potential):
    bore = .00288
    use = (radius > 0) & (radius <= bore * (1 + 1e-14))
    design = np.polynomial.polynomial.polyvander((radius[use] / bore) ** 2, 3)
    coeff, _, rank, _ = np.linalg.lstsq(design, potential[use] / radius[use, None], rcond=None)
    assert rank == 4
    return coeff.T


def refine():
    from temsim.optics.column import default_state
    from temsim.part_materials import material_catalog
    from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
    from temsim.physics.lens_field_provider import lens_geometry_binding

    receipt = json.loads((EV / "joint-objective-material-field.json").read_text())
    binding = lens_geometry_binding(default_state(), "objective_lens")
    material = next(row["bh_material"] for row in material_catalog() if row["material_key"] == "armco_pure_iron")
    assert sha256(binding.canonical_geometry_json.encode()).hexdigest() == receipt["installed_geometry_digest"]
    assert material["source_sha256"] == receipt["curve_sha256"]
    assert abs(FINAL["objective_At"]) == receipt["ampere_turns"]
    with np.load(OBJ) as original:
        radius, z0 = original["r"], original["z"]
    # Retain the complete original mesh. Insert axial planes across the upstream
    # gap, the yoke entrance, the first winding edge and the entire AC housing.
    for pitch in (1., .5):
        added = np.arange(1280., 1440. + pitch / 2, pitch) * .001
        z = np.unique(np.round(np.r_[z0, added], 13))
        _, sol = solve_geometry_field_details(binding, dict(
            solver="axisymmetric_nonlinear_fem", bh_material=material,
            channel_ampere_turns={"objective_lens": receipt["ampere_turns"]},
            padding_factor=receipt["padding"], validation_grid_axes_m=[radius.tolist(), z.tolist()],
        ))
        np.testing.assert_allclose(radius, sol.r_m, rtol=0, atol=1e-13)
        np.savez_compressed(OUT / f"objective-{pitch:g}mm.npz", z=sol.z_m,
                            coeff=bore_coefficients(sol.r_m, sol.a_phi_tm), residual=sol.relative_residual,
                            source_sha=sha(OBJ), geometry=receipt["installed_geometry_digest"],
                            curve=material["source_sha256"], objective_At=FINAL["objective_At"])
        print("Objective mesh (mm), shape, residual", pitch, sol.a_phi_tm.shape, sol.relative_residual, flush=True)


def refined_terms(f, z, pitch):
    with np.load(OUT / f"objective-{pitch:g}mm.npz") as data:
        assert str(data["source_sha"]) == sha(OBJ) and data["objective_At"] == FINAL["objective_At"]
        spline = make_interp_spline(data["z"], data["coeff"], k=3)
    result = np.zeros((len(z), 12))
    result[:, :8:2] = spline(z) * f.factors
    result[:, 1:8:2] = spline(z, nu=1) * f.factors
    return result


def field_profiles(f, q, material):
    zmm = np.arange(1200., 1480.01, .25)
    z = zmm * .001
    vacuum = q[:, None] * f.round_axis(z)
    mini = vacuum[OWNER == "mini_condenser"].sum(0)
    others = vacuum.sum(0) - mini + make_interp_spline(material["z"], material["induced_T"], k=7)(z)
    joint = np.interp(z, f.zjoint, f.bjoint[:, 2])
    obj = f.objective(z)[:, 0]
    with np.load(OBJ) as old:
        old_nodes = old["z"]
        old_axis = bore_coefficients(old["r"], old["a_phi"])[:, 0] * f.factors[0]
    np.savez_compressed(OUT / "axial.npz", z_mm=zmm, joint=joint, archive=obj, mini=mini, others=others,
                        one=refined_terms(f, z, 1.)[:, 0], half=refined_terms(f, z, .5)[:, 0],
                        old_z=old_nodes * 1000, old_b=old_axis)


def trace():
    import cupy as cp
    from column_current import ApertureTracker
    from condenser_material_binding import check_materials
    from condenser_pair_column import column_mesh, round_field
    from condenser_pair_ray_density import packet
    from relay_transfer import advance, electric_reference
    from scan_waist_audit import archived_scan
    from trace_normal_joint_field import beam_moments

    point = json.loads((PAIR_EVIDENCE / "working-point.json").read_text())
    check_file(PAIR_EVIDENCE / "column-and-landings.npz", point["evidence_sha256"]["column-and-landings.npz"])
    check_file(FEM, point["downstream_tracking_field_sha256"])
    f, q, material, g = packet()
    check_materials(f.state)
    scan = archived_scan()
    with np.load(FEM) as table:
        raw = dict(table)
    sources = json.loads(str(raw["report"]))["currents_A"]
    assert sources == point["corrector_A"] and not any("deflector" in key for key in sources)
    assert len(np.unique(g["ids"][:N])) == N and np.all(g["weight"][:N] > 0) and not g["weight"][N:].any()
    np.testing.assert_array_equal(q, point["signed_current_A"])
    field_profiles(f, q, material)
    m, _ = column_mesh(f, .05)
    tracker = ApertureTracker(m, raw, "cubic", source=(g["phase"], g["weight"], g["ids"],
                              g["energy_offset_ev"], float(g["emitted_A"])))
    field = round_field(f, m.nodes, q, material, 7)
    fine = field.copy()
    fine[:, :12] += cp.asarray(refined_terms(f, m.nodes, .5) - f.objective(m.nodes))
    no_obj = field.copy()
    no_obj[:, :12] -= cp.asarray(f.objective(m.nodes))
    en, ref = tracker.de + 300000., electric_reference(tracker)
    indices = np.linspace(0, N - 1, 64, dtype=int)
    weights = g["weight"][:N]
    loss0 = np.full(len(g["phase"]), -1, np.int32)
    # Start sufficiently upstream of TL12. Branch the explicitly field-free
    # continuation only at the actual AC entrance, with the same kinetic slopes.
    stations = np.arange(1330., 1478.01, .5)
    iz = np.searchsorted(m.z, np.round(stations * .001, 13))
    np.testing.assert_allclose(m.z[iz] * 1000, stations, rtol=0, atol=1e-9)
    (p0, lost0), = advance(tracker, field, tracker.input, en, ref, loss0, 0, iz[:1])
    assert np.all(lost0 == -1)
    zero = np.zeros((len(stations), 5))
    traces = {}
    entrance_phases = {}
    registered = tracker.tables[0]
    for name, active in (("archive", field), ("refined", fine),
                         ("no_objective", no_obj), ("no_corrector", field)):
        tracker.tables[0] = cp.zeros_like(registered) if name == "no_corrector" else registered
        points, lost, start = p0.copy(), lost0.copy(), iz[0]
        moments, paths, means = zero.copy(), [], []
        for j, (z, end) in enumerate(zip(stations, iz, strict=True)):
            if end != start:
                (points, lost), = advance(tracker, active, points, en, ref, lost, start, [end])
            start = end
            assert (lost == -1).all()
            moments[j] = beam_moments(points[:N], lost[:N] == -1, weights)
            means.append(np.average(points[:N, :4], axis=0, weights=weights))
            paths.append(points[indices, :4].copy())
            if z == ENTRY and name in ("archive", "refined"):
                entrance_phases[name] = points.copy()
        traces[name + "_moments"] = moments
        traces[name + "_paths"] = np.stack(paths)
        traces[name + "_means"] = np.stack(means)
        print("Matched original particles", name, N, flush=True)
    tracker.tables[0] = registered
    from_entry = stations >= ENTRY
    idx = np.flatnonzero(from_entry)[0]
    for case, key in (("archive","drift"), ("refined","refined_drift")):
        moments, paths, means = [], [], []
        for z in stations[from_entry]:
            p = entrance_phases[case][:N, :4].copy()
            p[:, :2] += (z - ENTRY) * .001 * p[:, 2:]
            moments.append(beam_moments(p, lost0[:N] == -1, weights))
            paths.append(p[indices])
            means.append(np.average(p, axis=0, weights=weights))
        traces[key + "_moments"] = np.stack(moments)
        traces[key + "_paths"] = np.stack(paths)
        traces[key + "_means"] = np.stack(means)
        np.testing.assert_array_equal(traces[key + "_paths"][0], traces[case + "_paths"][idx])
    # The published positions, IDs and all-population radial moments verify the
    # registration and the exact RT field used for the original diagram.
    with np.load(PAIR_EVIDENCE / "column-and-landings.npz") as saved:
        where = np.abs(saved["z_mm"][:, None] - stations).argmin(0)
        np.testing.assert_allclose(saved["z_mm"][where], stations, atol=1e-9, rtol=0)
        np.testing.assert_array_equal(saved["selected_ids"], g["ids"][indices])
        np.testing.assert_array_equal(traces["archive_paths"][:, :, :2], saved["paths_m"][where])
        np.testing.assert_allclose(traces["archive_moments"], saved["stats"][where], rtol=2e-11, atol=1e-12)
    np.savez_compressed(OUT / "rays.npz", z_mm=stations, ids=g["ids"][indices],
                        weights=weights, remaining=N, **traces)
    write(OUT / "drives.json", dict(scan=scan, corrector_A=sources, objective_A=FINAL["objective_A"],
          effective_source_count=N, comparison_start_mm=stations[0], drift_start_mm=ENTRY,
          scan_field_table_sha256=sha(FEM), retained_scan_command_mrad=[0.,0.]))


def render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from temsim.optics.column import default_state
    from temsim.physics.lens_field_provider import lens_geometry_binding
    from normal_working_point import CACHE
    from relay_transfer import PARTS, START

    with np.load(OUT / "axial.npz") as data:
        b = dict(data)
    with np.load(OUT / "rays.npz") as data:
        r = dict(data)
    registered = json.loads((OUT / "drives.json").read_text())
    older = json.loads((EV / "normal-current-working-point.json").read_text())
    older_file = CACHE / "normal-joint-field.npz"
    check_file(older_file, older["field_table_sha256"])
    with np.load(older_file) as old_table, np.load(FEM) as new_table:
        old_recipe, new_recipe = [json.loads(str(t["report"])) for t in (old_table, new_table)]
        for key in ("circuit_sha256", "mesh_id", "turns"):
            assert old_recipe[key] == new_recipe[key]
        expected = {**new_recipe["currents_A"], "probe_tl12_lens": 10.}
        assert old_recipe["currents_A"] == expected
        old_mid = len(old_table["xy_m"]) // 2
        old_joint = np.interp(b["z_mm"], (START + old_table["z_local_m"]) * 1000,
                              old_table["B_zyx_T"][:, old_mid, old_mid, 2])
    geom = lens_geometry_binding(default_state(), "objective_lens")
    parts = {s["key"]: s for s in json.loads(geom.canonical_geometry_json)["lens_assembly"]["parts"]}
    coil = parts["objective_lens_excitation_coil"]["data"]
    yoke = parts["objective_lens_yoke"]["data"]
    z, zf = r["z_mm"], b["z_mm"]
    zz = z[z >= ENTRY]
    scan = PARTS["ac_deflector"]
    upper, lower = START * 1000 + np.asarray(scan["interaction_centers_local_z_mm"])
    assert np.allclose(START * 1000 + np.array([scan["local_start_z_mm"],scan["local_end_z_mm"]]), [ENTRY, EXIT])
    stations = np.r_[1360., 1366., 1375., ENTRY, 1382., upper, 1400., 1408., lower, EXIT]

    def sample(x, y, where):
        return float(np.interp(where, x, y))

    def extrema(name, lo, hi):
        mom = r[name + "_moments"]
        zs = zz if name.endswith("drift") else z
        v = mom[:, 1] + mom[:, 2]
        rows = []
        for i in np.flatnonzero(np.signbit(v[1:]) != np.signbit(v[:-1])):
            at = zs[i] - v[i] * (zs[i+1] - zs[i]) / (v[i+1] - v[i])
            if lo <= at <= hi:
                rows.append(dict(kind="maximum" if v[i] > 0 else "minimum", z_mm=float(at),
                                 rms_um=sample(zs, mom[:, 0], at)))
        return rows

    cases = ("archive", "refined", "no_objective", "no_corrector", "drift", "refined_drift")
    reg_field = b["archive"] + b["joint"] + b["mini"] + b["others"]
    new_field = b["half"] + b["joint"] + b["mini"] + b["others"]
    field_record = {
        float(loc): {key: sample(zf, curve * 1000, loc) for key, curve in (
            ("corrector_mT", b["joint"]), ("corrector_TL12_10_A_mT", old_joint),
            ("objective_original_mT", b["archive"]),
            ("objective_1mm_mT", b["one"]), ("objective_0.5mm_mT", b["half"]), ("mini_mT", b["mini"]),
            ("total_original_mT", reg_field), ("total_0.5mm_mT", new_field))}
        for loc in stations
    }
    local = (zf >= ENTRY) & (zf <= EXIT)
    old_at, old_b = b["old_z"], b["old_b"]
    keep = (old_at >= 1343.9) & (old_at <= 1390.)
    report = dict(
        operating_point_sha256=sha(PAIR_EVIDENCE / "working-point.json"), **registered,
        lower_TL12_control_field_sha256=sha(older_file),
        scan_centres_mm=[upper,lower], objective_upper_winding_mm=coil["material_intervals_mm"][0],
        objective_winding_radius_mm=[coil["mechanical_inner_diameter_mm"]/2, coil["mechanical_outer_diameter_mm"]/2],
        objective_upper_yoke_mm=yoke["material_intervals_mm"][0],
        objective_upper_pole_mm=[parts["objective_upper_pole"][v] for v in ("start_z_mm","end_z_mm")],
        on_axis=field_record, original_objective_nodes_mm_mT=np.column_stack((old_at[keep], 1000*old_b[keep])).tolist(),
        scan_maximum_objective_1mm_to_0_5mm_change_mT=float(1000*np.max(abs(b["half"][local]-b["one"][local]))),
        extrema_1330_through_1478={k:extrema(k, 1330., 1478.) for k in cases},
        radius_at_AC_exit_um={k:sample(zz if k.endswith("drift") else z, r[k+"_moments"][:,0], EXIT) for k in cases},
        radius_at_AC_entrance_um=sample(z, r["archive_moments"][:,0], ENTRY),
        centroids_at_AC_stations_nm_and_urad={k:{
            float(s): (r[k+"_means"][np.argmin(abs(z - s))] * np.array([1e9,1e9,1e6,1e6])).tolist()
            for s in (ENTRY,upper,lower,EXIT)} for k in ("archive", "refined")},
        trajectory_controls="All lens-field controls continue the same 32768 incident particles from 1330 mm; each drift continues its own 1380 mm positions and kinetic directions in zero force. Original full-weight RMS and 64 archived trajectories were checked.",
    )
    write(EV / "condenser-scan-region.json", report)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.4, 8.0), layout="constrained",
                             gridspec_kw={"height_ratios":[1.2,1.,1.]})
    fig.suptitle("Why the beam bends with the scan drive off", fontsize=16, weight="bold")
    azul, amber, plum, gray, green = "#156789", "#a94417", "#794780", "#525d66", "#25785d"
    lo, hi = 1340., 1432.
    for ax in axes:
        ax.axvspan(ENTRY, EXIT, color="#e8f1f3", zorder=-3)
        for loc in (upper, lower):
            ax.axvline(loc, color="#67858d", ls=":", lw=1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_xlim(lo, hi)
    ax = axes[0]
    ax.set_title("Axial field along the beam axis", loc="left", fontsize=11, pad=12)
    ax.axhline(0, color="#6f7c81", lw=.65)
    ax.plot(zf, 1000*b["joint"], color=azul, label="Corrector FEM (includes TL12)")
    ax.plot(zf, 1000*b["archive"], color=amber, ls="--", label="Objective, original")
    ax.plot(zf, 1000*b["half"], color=amber, lw=2, label="Objective, refined")
    show=(old_at>=lo)&(old_at<=hi)
    ax.scatter(old_at[show], 1000*old_b[show], s=13, edgecolors=amber, marker="o", facecolors="none",
               label="Original objective FEM nodes")
    for loc, tag in ((1366.3,"TL12"),(upper,"Upper scan"),(lower,"Lower scan")):
        ax.text(loc,.995,tag,transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=9)
    ax.set(ylim=(-163,170),ylabel="Axial field (mT)")
    ax.legend(loc="lower left", ncol=2, fontsize=8, framealpha=.92)

    # Select by entrance radius, independently of the size of the subsequently
    # observed ablation effects. Keep the same original ID in both projections.
    idx = np.flatnonzero(z >= ENTRY)[0]
    radii = np.linalg.norm(r["archive_paths"][idx,:,:2], axis=1)
    sel = np.argsort(radii)[3*len(radii)//4]
    ax = axes[1]
    ax.set_title(f"One original electron in laboratory coordinates  ·  source ID {r['ids'][sel]}", loc="left", fontsize=11)
    for coord,col in ((0,azul),(1,plum)):
        ax.plot(z, 1e6*r["archive_paths"][:,sel,coord], c=col, lw=1.9)
        ax.plot(zz, 1e6*r["drift_paths"][:,sel,coord], c=col, ls="--", lw=1.5)
    ax.set_ylabel("Position (µm)")
    ax.set_ylim(-1.1,1.7)
    ax.legend(handles=[Line2D([],[],color=azul,label="x"), Line2D([],[],color=plum,label="y"),
                       Line2D([],[],color=gray,label="Original field"), Line2D([],[],color=gray,ls="--",
                       label="Straight drift from AC entrance")], loc="lower left", ncol=2, fontsize=8)
    ax = axes[2]
    ax.set_title("Beam radius about the current centroid  ·  all 32,768 particles", loc="left",fontsize=11)
    ax.plot(z,r["archive_moments"][:,0],c=gray,lw=2.0,label="Original field")
    ax.plot(z,r["refined_moments"][:,0],c=green,lw=1.8,label="Refined objective from 1330 mm")
    ax.plot(zz,r["drift_moments"][:,0],c=gray,ls="--",lw=1.4,label="Drift from original AC entrance")
    ax.plot(zz,r["refined_drift_moments"][:,0],c=green,ls="--",lw=1.4,label="Drift from refined AC entrance")
    ax.set(ylim=(.39,1.53),ylabel="RMS radius (µm)",xlabel="Distance from effective source  z (mm)")
    ax.legend(loc="upper left",ncol=2,fontsize=8)
    fig.savefig(EV / "condenser-scan-region.png",dpi=165)
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ("extrema_1330_through_1478", "radius_at_AC_exit_um",
          "scan_maximum_objective_1mm_to_0_5mm_change_mT")}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("refine", "trace", "render"))
    args = parser.parse_args()
    {"refine": refine, "trace": trace, "render": render}[args.action]()
