"""Audit the native specimen focus, angle-dependent errors and a measured vacuum pupil."""

import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

from fit_threefold import EVIDENCE, HERE, additional_gradient, design, fit, read_sample, validation_folds
from temsim.physics.beam_statistics import _weighted_quantile


OUT = HERE.parents[2] / "outputs/cescor-a2-20261121"
RUNS = OUT / "spherical"
SPECIMEN = json.loads((EVIDENCE / "comparison.json").read_text(encoding="utf-8"))
RECORD = EVIDENCE / "focus-and-pupil.json"
VACUUM = EVIDENCE / "louiset-vacuum-roi.npz"
STANDARD = "focus-audit-native"
PREVIOUS = "a2-cubic-both-fine"


def run_record(name):
    return json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))


def spread(xy_um, weight):
    relative = xy_um - np.average(xy_um, axis=0, weights=weight)
    radius = np.linalg.norm(relative, axis=1)
    return {
        "D50_nm": float(2000 * _weighted_quantile(radius, weight, .5)),
        "D95_nm": float(2000 * _weighted_quantile(radius, weight, .95)),
        "rms_radius_nm": float(1000 * np.sqrt(np.average(radius**2, weights=weight))),
    }


def search_core_lens():
    """Vary only the physical Mini excitation to maximize sampled current within 50 nm."""
    from column_current import SURFACE, field_array, metrics, source_digest
    from finite_probe import bundle

    model = run_record(STANDARD)
    with np.load(RUNS / model["phase_file"]) as d:
        obs = d["z_m"]
    mesh, tracker, bg, coils = bundle(.075, obs, model["objective_At"], model["emitted"],
                                       SPECIMEN["gun"]["gun_exit_A"], OUT/"a2-cubic-both-fine-field.npz",
                                       transverse_interpolation="cubic")
    assert source_digest(tracker) == model["source_digest"]
    mesh.saved = np.array([np.flatnonzero(mesh.z == SURFACE).item()])
    ref = json.loads((EVIDENCE/"refocused-working-point.json").read_text(encoding="utf-8"))
    mini = np.asarray(ref["owner_per_group"]) == "mini_condenser"
    original = np.asarray(model["currents_A"])
    assert mini.sum() == 3
    trials = {}

    def measure(change):
        if change not in trials:
            drive = original.copy()
            drive[mini] += change * np.asarray(ref["currents_A"])[mini]
            phase, loss, _, _ = tracker.run(field_array(mesh,bg,coils,drive), keep_history=False)
            live = loss == -1
            p, w = phase[-1,live], tracker.weights[live]
            core = np.hypot(*p[:, :2].T) <= 50e-9
            trials[change] = {"mini_gain": model["mini_gain"] + change,
                              "axis_50nm_uA": float(tracker.current * 1e6 * w[core].sum()),
                              **metrics(p,w,tracker.current)}
        return trials[change]

    for step in np.arange(-10, 3)*.000035:
        measure(float(step))
    def best_change():
        return max((k for k,v in trials.items() if abs(v["alpha95_mrad"]-30) < .1),
                   key=lambda k:trials[k]["axis_50nm_uA"])
    for pitch in (5e-6, 1e-6):
        centre = best_change()
        for step in range(-6,7):
            measure(centre+step*pitch)
    chosen = best_change()
    assert min(trials) < chosen < max(trials)
    result = {"starting_point": STANDARD, "selected": trials[chosen], "trials": list(trials.values())}
    (RUNS/"focus-audit-core-search.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(trials[chosen], flush=True)


def native_focus(name):
    from column_current import SURFACE, field_array, source_digest
    from finite_probe import bundle

    run = run_record(name)
    raw = RUNS / run["phase_file"]
    assert sha256(raw.read_bytes()).hexdigest() == run["trajectory_sha256"]
    # Scan the same column broadly and then sample both finite-radius current maxima.
    offsets = np.unique(np.r_[np.arange(-300000, 300001, 5000), np.arange(-40000, 10001, 1000),
                              np.arange(-27500, -13500 + 1, 100), np.arange(-2000, 10001, 100), [-775, 0]])
    planes = SURFACE + offsets * 1e-9
    with np.load(raw) as data:
        original_loss = data["loss_owner"]
        original_phase = data["phase"]
        original_ids = data["ids"]
        obs = np.unique(np.r_[data["z_m"], planes])
    table = OUT / "a2-cubic-both-fine-field.npz"
    assert sha256(table.read_bytes()).hexdigest() == run["corrector_sha256"]
    mesh, tracker, bg, coils = bundle(.075, obs, run["objective_At"], run["emitted"],
                                       SPECIMEN["gun"]["gun_exit_A"], table, transverse_interpolation="cubic")
    assert source_digest(tracker) == run["source_digest"]
    np.testing.assert_array_equal(tracker.ids, original_ids)
    mesh.saved = np.searchsorted(mesh.z, planes)
    np.testing.assert_array_equal(mesh.z[mesh.saved], planes)
    field = field_array(mesh, bg, coils, run["currents_A"])
    phases, end_loss, _, _ = tracker.run(field, keep_history=True, record_probe=True,
                                         surface_index=int(mesh.saved[offsets == 0].item()))
    live = original_loss == -1
    np.testing.assert_array_equal(live, tracker.surface_loss == -1)
    np.testing.assert_array_equal(live, end_loss == -1)
    np.testing.assert_array_equal(original_loss[~live], mesh.owner[end_loss[~live]])
    np.testing.assert_allclose(tracker.moments[:, 0], tracker.weights[live].sum(), atol=2e-15, rtol=0)
    np.testing.assert_allclose(tracker.surface_phase[live, :2], original_phase[live, :2], atol=2e-13, rtol=0)
    current_uA = tracker.current * 1e6 * tracker.weights[live]
    radii = np.array([.05, 50., 100.])
    disk_current = []
    for phase in phases:
        r_nm = np.linalg.norm(phase[live, :2], axis=1) * 1e9
        disk_current.append([float(current_uA[r_nm <= radius].sum()) for radius in radii])
    disk_current = np.asarray(disk_current)
    extrema = {}
    for col, radius in enumerate(radii):
        best = np.flatnonzero(disk_current[:, col] == disk_current[:, col].max())
        extrema[str(radius)] = {"max_uA": float(disk_current[best[0], col]),
                                "offsets_um": (offsets[best] * .001).tolist(),
                                "surface_uA": float(disk_current[offsets == 0, col].item()),
                                "D95_at_first_max_nm": float(tracker.probes[best[0], 0])}
    return {"offset_um": (offsets * .001).tolist(), "disk_radii_nm": radii.tolist(),
            "current_uA": disk_current.tolist(), "extrema": extrema,
            "D95_nm": tracker.probes[:, 0].tolist(), "rms_radius_nm": tracker.probes[:, 1].tolist()}


def hexapole_relay(name):
    """Measure the position response at HP1 to a change of slope at the HP2 midplane."""
    import tomllib
    import cupy as cp
    from column_current import FACTOR, START, field_array, source_digest
    from finite_probe import bundle

    run = run_record(name)
    table_name = {STANDARD: "a2-cubic-both", "focus-audit-tl4-native": "focus-audit-tl4",
                  "focus-audit-tl6-native": "focus-audit-tl6"}[name]
    table = OUT / f"{table_name}-fine-field.npz"
    assert sha256(table.read_bytes()).hexdigest() == run["corrector_sha256"]
    config = HERE.parents[2] / "configs/instruments/column" / run["corrector_field"]["source_file"]
    parts = {part["key"]: part for part in tomllib.loads(config.read_text(encoding="utf-8"))["parts"]}
    planes = START + .001 * np.array([parts[key]["local_center_z_mm"]
                                     for key in ("probe_hp2_hexapole", "probe_hp1_hexapole")])
    assert planes[0] < planes[1]
    with np.load(RUNS / run["phase_file"]) as d:
        obs = np.unique(np.r_[d["z_m"], planes])
        survivors = d["loss"] == -1
    mesh, tracker, background, coils = bundle(.075, obs, run["objective_At"], run["emitted"],
                                              SPECIMEN["gun"]["gun_exit_A"], table,
                                              transverse_interpolation="cubic")
    assert source_digest(tracker) == run["source_digest"]
    first, last = mesh.saved = np.searchsorted(mesh.z, planes)
    np.testing.assert_array_equal(mesh.z[mesh.saved], planes)
    full_field = field_array(mesh, background, coils, run["currents_A"])
    phases, final_loss, _, _ = tracker.run(full_field, surface_index=first)
    field = cp.asarray(full_field)
    energy = cp.asarray(300000. + tracker.de)
    # Keep the absolute electrostatic energy reference of every original gun particle on restart.
    radius2 = (tracker.input[:, :2]**2).sum(axis=1)
    radial_bin = np.clip(np.searchsorted(tracker.r2, radius2, side="right") - 1, 0, len(tracker.r2)-2)
    phi0 = cp.asarray(tracker.e0[radial_bin, 0] +
                      (radius2-tracker.r2[radial_bin]) * tracker.e0[radial_bin, 1])
    n = len(tracker.ids)
    assert np.all(final_loss[survivors] == -1)

    def replay(axis, delta):
        phase = cp.asarray(phases[0])
        loss = cp.asarray(tracker.surface_loss)
        phase[:, 2+axis] += delta
        peak, margin = cp.zeros(n), cp.full(n, cp.inf)
        for k in range(first, last, 160):
            tracker.kernel(((n+127)//128,), (128,), (
                phase, energy, phi0, field, tracker.ep, tracker.er, np.int32(len(tracker.r2)),
                tracker.h, tracker.rad, loss, peak, margin, np.int32(k), np.int32(min(k+160, last)),
                np.int32(n), tracker.rest, FACTOR, tracker.pc))
        np.testing.assert_array_equal(loss.get(), final_loss)
        return phase.get()[survivors, :2]

    np.testing.assert_allclose(replay(0, 0), phases[-1, survivors, :2], rtol=0, atol=1e-14)
    jacobians = []
    for delta in (2e-6, 1e-6):
        columns = [(replay(axis, delta) - replay(axis, -delta)) / (2*delta) for axis in (0, 1)]
        jacobians.append(1000*np.stack(columns, axis=2))  # millimetres per radian
    large, small = jacobians
    singular = np.linalg.svd(small, compute_uv=False)
    r_hp2 = np.linalg.norm(phases[0, survivors, :2], axis=1)
    inner = r_hp2 <= np.quantile(r_hp2, .1)
    return {"run": name, "HP2_HP1_z_mm": (1000*planes).tolist(), "common_survivors": int(survivors.sum()),
            "slope_steps_urad": [2., 1.], "definition": "d(x,y)_HP1 / d(tx,ty)_HP2 in mm/rad",
            "mean_matrix_mm_per_rad": small.mean(axis=0).tolist(),
            "singular_value_percentiles_5_50_95_mm_per_rad": np.percentile(singular,[5,50,95],axis=0).tolist(),
            "inner_tenth_median_singular_values_mm_per_rad": np.median(singular[inner],axis=0).tolist(),
            "maximum_step_change_mm_per_rad": float(np.max(np.abs(large-small)))}


def capture_experiment(path):
    """Average the 64 original vacuum frames identified in the authors' preprocessing notebook."""
    import h5py

    with h5py.File(path) as f:
        cube = f["datacube_root/datacube/data"]
        assert cube.shape == (170, 150, 256, 256)
        frames = cube[150:158, 4:12]
        cal = f["datacube_root/metadatabundle/calibration"]
        assert cal["Q_pixel_units"][()] == b"mrad"
        pixel = cal["Q_pixel_size"][()]
        centre = np.array([cal["qx0_mean"][()], cal["qy0_mean"][()]])
        edge = pixel * cal["probe_semiangle"][()]
    assert frames.max() < np.iinfo(frames.dtype).max
    # The split retains count totals for checking scan-to-scan repeatability.
    totals = frames.reshape(64, 256, 256).astype(np.uint32)
    np.savez_compressed(VACUUM, sum_even=totals[::2].sum(0, dtype=np.uint32),
                        sum_odd=totals[1::2].sum(0, dtype=np.uint32), pixel_mrad=pixel,
                        centre_pixels=centre, edge_mrad=edge, frames=64,
                        source_doi="10.5281/zenodo.19238708", license="CC BY 4.0")


def angular_data(angle, weight, reference, axis_bins):
    # u is angle divided by each beam's stated reference. The model has no hard 30 mrad edge.
    u = angle / reference
    histogram = np.histogram(u, bins=axis_bins, weights=weight)[0]
    edges_rad = axis_bins * reference * 1e-3
    solid_angle = 4*np.pi*np.sin((edges_rad[1:] + edges_rad[:-1])/2)*np.sin(np.diff(edges_rad)/2)
    disk_omega = 4*np.pi*np.sin(reference * .0005)**2
    groups = np.histogram(u, bins=[0., 1/3, 2/3, 1., np.inf], weights=weight)[0]
    np.testing.assert_allclose(groups.sum(), weight.sum(), rtol=1e-10)
    order = np.argsort(angle)
    csum = np.r_[0.,np.cumsum(weight[order])]
    cumulative = csum[np.searchsorted(angle[order], axis_bins*reference, side="right")]
    return {"reference_mrad": float(reference), "group_pct": (100*groups/weight.sum()).tolist(),
            "group_signal": groups.tolist(),
            "radial_density": (histogram/weight.sum()*disk_omega/solid_angle).tolist(),
            "enclosed_pct": (100*cumulative/weight.sum()).tolist(),
            "alpha50_mrad": float(_weighted_quantile(angle, weight, .5)),
            "alpha95_mrad": float(_weighted_quantile(angle, weight, .95))}


def measured_pupil(bins):
    with np.load(VACUUM) as d:
        even, odd = d["sum_even"].astype(float), d["sum_odd"].astype(float)
        pixel, ref, centre = float(d["pixel_mrad"]), float(d["edge_mrad"]), d["centre_pixels"]
        count = int(d["frames"])
    q = (np.indices(even.shape)-centre[:, None, None]) * pixel
    angles = np.hypot(*q)
    raw = (even + odd) / count
    # Keep all raw detector values including the dark pedestal in reported fractions.
    summary = angular_data(angles.ravel(), raw.ravel(), ref, bins)
    split = [angular_data(angles.ravel(), split_img.ravel(), ref, bins)["group_pct"]
             for split_img in (even, odd)]
    # This alternative subtraction is deliberately signed to avoid positively biasing a weak tail.
    background = raw[(angles >= 50) & (angles < 70)].mean()
    corrected = np.histogram(angles/ref, [0, 1/3, 2/3, 1, np.inf], weights=raw-background)[0]
    # Use each annulus's actual detector pixel population at finite angular pitch.
    pixel_area_sr = (pixel*.001)**2 * np.sinc(angles*.001/np.pi)
    solid_angle = np.histogram(angles/ref, bins, weights=pixel_area_sr)[0]
    counts = np.histogram(angles/ref,bins,weights=raw)[0]
    density = (counts/solid_angle) * (4*np.pi*np.sin(ref*.0005)**2/raw.sum())
    annulus = np.digitize(angles / ref, [1/3, 2/3, 1])
    summary.update(pixel_mrad=pixel, frames=count, origin_pixels=centre.tolist(), split_group_pct=split,
                   detector_counts=int(even.sum() + odd.sum()),
                   annular_mean_counts_per_pixel_frame=[float(raw[annulus == k].mean()) for k in range(4)],
                   background_per_pixel=float(background), estimated_background_group_pct=(100*corrected/corrected.sum()).tolist(),
                   radial_density=density.tolist(),
                   frequency_scale="Raw detector counts, normalized to the whole recorded detector")
    return summary


def ideal_disk():
    """Analytical circular pupil convention, separate from transported particle metrics."""
    from scipy.optimize import brentq
    from scipy.special import j0, j1
    from temsim.optics.electron_gun.effective_source import wavelength_m

    wavelength = wavelength_m(300000.)
    x_half = brentq(lambda x:(2*j1(x)/x)**2-.5,1.,2.)
    x95 = brentq(lambda x:1 - j0(x)**2 - j1(x)**2 - .95,10.,16.)
    scale = wavelength * 1e12/(np.pi*np.sin(.030))
    return {"energy_keV":300., "edge_mrad":30., "wavelength_pm":wavelength*1e12,
            "FWHM_diameter_pm":x_half*scale, "D95_diameter_pm":x95*scale}


def angular_residuals(sample, delta_e):
    from fit_threefold import REFERENCE_RAD

    folds = validation_folds(sample.ids)
    basis, names = design(sample)
    u = sample.angles_rad / REFERENCE_RAD
    fifth = basis
    group_masks = {"all": np.ones(len(u), dtype=bool), "inside_30_mrad": sample.alpha_mrad <= 30}
    results = {}
    for label, mask in group_masks.items():
        xy, w, fold, local = sample.xy_um[mask], sample.current_na[mask], folds[mask], u[mask]
        row = {"observed": spread(xy,w), "current_uA": float(w.sum()*.001), "count": int(mask.sum())}
        basis = fifth[mask]
        for order in (3, 5, 7, 9):
            if order == 3:
                keep = np.array([key.split("_")[0] in ("offset","C1","A1","A2","B2","C3","A3","S3") for key in names])
                tested = basis[:, :, keep]
            elif order == 5:
                tested = basis
            else:
                for power in (order, order+1):
                    higher = [additional_gradient(local,power,m)[:, :, :2 if m else 1]
                              for m in range(power % 2, power+1, 2)]
                    extra = -np.linalg.inv(sample.frame_xy) @ np.concatenate(higher,axis=2)
                    basis = np.concatenate((basis, extra), axis=2)
                tested = basis
            residual = np.empty_like(xy)
            for k in range(5):
                train = fold != k
                coeff, _ = fit(tested[train], xy[train], w[train])
                residual[~train] = xy[~train] - tested[~train] @ coeff
            row[f"order_{order}"] = spread(residual, w)
        # This two-component fit relaxes the gradient constraint. It is a conditional
        # diagnostic of energy-dependent scatter, without an equivalent coil drive.
        x, y = local.T
        monomials = np.column_stack([x**i*y**(n-i) for n in range(8) for i in range(n+1)])
        for use_energy in (False,True):
            a = monomials
            if use_energy:
                a = np.column_stack((a, (delta_e[mask]-.30)[:,None] * monomials[:,:21]))
            residual = np.empty_like(xy)
            for k in range(5):
                train = fold != k
                wi = np.sqrt(w[train, None])
                c, _, rank, _ = np.linalg.lstsq(a[train]*wi,xy[train]*wi,rcond=1e-12)
                assert rank == a.shape[1]
                residual[~train] = xy[~train] - a[~train] @ c
            row["vector_7_with_energy" if use_energy else "vector_7"] = spread(residual,w)
        results[label] = row
    return results


def analyze():
    summary = {"model_runs": {}, "source_current_uA": 15., "real_reference_doi": "10.5281/zenodo.19238708",
               "specimen_upper_surface_z_m": SPECIMEN["specimen_upper_surface_z_m"]}
    bins = np.linspace(0,1.7,35)
    summary["radial_bin_edges_normalized"] = bins.tolist()
    summary["measured_vacuum"] = measured_pupil(bins)
    summary["measured_vacuum"].update(
        source_article_doi="10.1088/2515-7639/ae6e71",
        raw_archive_member="swcnt_4DSTEM/large_angle/ptycho_exp/swcnt_dataset_150x170.h5",
        raw_crc32="8da613c9", roi="data[150:158, 4:12, :, :]", license="CC BY 4.0",
        summed_frames_sha256=sha256(VACUUM.read_bytes()).hexdigest())
    summary["ideal_uniform_coherent_disk"] = ideal_disk()
    names = [PREVIOUS, STANDARD, "focus-audit-halfstep", "focus-audit-bright-core"]
    names += sorted(p.stem for p in RUNS.glob("focus-audit-tl*-native.json"))
    with np.load(RUNS / run_record(PREVIOUS)["phase_file"]) as baseline:
        shared_source = {k: baseline[k] for k in ("ids", "weights", "energy_offset_eV", "loss_owner")}
    for name in names:
        row = run_record(name)
        file = RUNS / row["phase_file"]
        assert sha256(file.read_bytes()).hexdigest() == row["trajectory_sha256"]
        sample = read_sample(file, row, SPECIMEN)
        with np.load(file) as d:
            for key, reference in shared_source.items():
                np.testing.assert_array_equal(d[key], reference)
            energy = d["energy_offset_eV"][d["loss"] == -1]
            offsets = d["offsets_nm"]
            planes = SPECIMEN["specimen_upper_surface_z_m"] + offsets * 1e-9
            idx = np.searchsorted(d["z_m"], planes)
            np.testing.assert_array_equal(d["z_m"][idx], planes)
            shown = np.abs(offsets) <= 40000
            axial_d95 = {"offset_um": (offsets[shown] * .001).tolist(),
                         "diameter_nm": d["profiles"][idx[shown], 0].tolist()}
            core = d["profiles"][idx,2]
            core_max = {"radius_pm": 50., "max_nA": float(core.max()),
                        "offset_um": (d["offsets_nm"][core==core.max()] *.001).tolist()}
        record = {key:row[key] for key in ("surface","source_digest","emitted","survived","trajectory_sha256",
                                           "corrector_sha256","objective_At","tl12_gain","mini_gain","currents_A",
                                           "C2_foil","current_ledger","through_focus","waists_mm","common_waists_mm",
                                           "max_step_mm","transverse_interpolation","fit","fit_inside_30mrad",
                                           "D95_minimum_bracket_nm","phase_file","objective_solution")}
        field_file = OUT / (row["field_file"] if name != PREVIOUS else "a2-cubic-both-fine-field.npz")
        assert sha256(field_file.read_bytes()).hexdigest() == row["corrector_sha256"]
        record["field_file"] = field_file.name
        with np.load(field_file) as field:
            record["transverse_pitch_mm"] = float(1000 * np.diff(field["xy_m"][:2])[0])
            record["axial_pitch_mm"] = float(1000 * np.diff(field["z_local_m"][:2])[0])
        record["corrector_windings_A"] = row["corrector_field"]["currents_A"]
        r = np.linalg.norm(sample.xy_um,axis=1)*1000
        record["fixed_axis_core_50nm_uA"] = float(sample.current_na[r<=50].sum()*.001)
        record["sampled_50pm_peak"] = core_max
        record["pupil"] = angular_data(sample.alpha_mrad,sample.current_na,30.,bins)
        record["pupil"]["max_surviving_angle_mrad"] = float(sample.alpha_mrad.max())
        record["energy_std_eV"] = float(np.sqrt(np.average(
            (energy - np.average(energy, weights=sample.current_na))**2, weights=sample.current_na)))
        record["axial_D95"] = axial_d95
        if name in (STANDARD, "focus-audit-tl4-native", "focus-audit-tl6-native"):
            record["hexapole_relay"] = json.loads((RUNS/f"{name}-relay.json").read_text(encoding="utf-8"))
        if name in (STANDARD, "focus-audit-bright-core"):
            edges_nm = np.linspace(-1000, 1000, 201)
            image, _, _ = np.histogram2d(*(sample.xy_um*1000).T, bins=(edges_nm, edges_nm),
                                         weights=sample.current_na)
            flat = image.T.ravel()
            occupied = np.flatnonzero(flat)
            record["landing_map"] = {
                "edges_nm": edges_nm.tolist(), "pixel_indices_yx": occupied.tolist(),
                "pixel_current_nA": flat[occupied].tolist(),
                "current_pct_in_window": float(100*flat.sum()/sample.current_na.sum())}
        record["angle_landing_samples"] = []
        for lower,upper in ((1,3),(10,12),(18,20),(20,22),(29,30),(30,31),(32,34),(36,38)):
            group = (sample.alpha_mrad>=lower) & (sample.alpha_mrad<upper)
            record["angle_landing_samples"].append({"range_mrad":[lower,upper], "count":int(group.sum()),
                "radius_quantiles_10_50_90_nm":[float(_weighted_quantile(r[group],sample.current_na[group],q))
                                                for q in (.1,.5,.9)]})
        if name in (PREVIOUS,STANDARD,"focus-audit-bright-core"):
            focus = RUNS / f"{name}-core-profile.json"
            record["core_focus"] = json.loads(focus.read_text(encoding="utf-8"))
        if name in (STANDARD,"focus-audit-halfstep"):
            record["held_out_residuals"] = angular_residuals(sample,energy)
        summary["model_runs"][name] = record
    RECORD.write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads(RECORD.read_text(encoding="utf-8"))
    runs = summary["model_runs"]
    ink, teal, red = "#223b54", "#007b79", "#b7542d"
    plt.rcParams.update({"axes.spines.top":False, "axes.spines.right":False, "axes.labelsize":10,
                         "axes.titlesize":12, "legend.fontsize":9})
    edges = np.asarray(summary["radial_bin_edges_normalized"])
    fig, (left, right) = plt.subplots(1,2,figsize=(10.5,4.5))
    fig.subplots_adjust(left=.078,right=.979,top=.765,bottom=.238,wspace=.27)
    fig.text(.078,.945,"Angular illumination at the specimen",fontsize=18,weight="bold",color=ink)
    fig.text(.078,.858,"Each beam is shown relative to its own reference angle. All recorded current or counts are included.",
             fontsize=10,color="#465461")
    for item,label,color in ((summary["measured_vacuum"],"Measured vacuum disk · 26.12 mrad",teal),
                             (runs[STANDARD]["pupil"],"Simulated beam · 30.00 mrad",red)):
        left.stairs(item["radial_density"],edges,baseline=None,color=color,label=label,linewidth=1.9)
        right.plot(edges,item["enclosed_pct"],color=color,linewidth=2)
    for ax in (left,right):
        ax.axvline(1,color="#606975",linestyle=":",linewidth=1)
        ax.set(xlim=(0,1.65),xlabel="Incident semi-angle / reference angle")
        ax.grid(axis="y",alpha=.2)
    left.set(ylabel="Signal per solid angle / uniform disk level",ylim=(0,None))
    right.set(ylabel="Enclosed fraction (%)",ylim=(0,104))
    left.set_title("Angular current density",loc="left")
    right.set_title("Current accumulated from the axis",loc="left")
    fig.legend(*left.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.52,.022),
               frameon=False,ncols=2)
    fig.savefig(HERE/"focus-audit-angular-illumination.png",dpi=155)
    plt.close(fig)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.5,4.5))
    fig.subplots_adjust(left=.078,right=.979,top=.765,bottom=.238,wspace=.27)
    fig.text(.078,.945,"Where the probe focuses",fontsize=18,weight="bold",color=ink)
    fig.text(.078,.858,"Native trajectories at fixed physical planes. Positive distance follows the beam.",
             fontsize=10,color="#465461")
    for name,label,color in ((STANDARD,"D95 focused at the surface",teal),
                              ("focus-audit-bright-core","Maximum surface current within 50 nm",red)):
        scan=runs[name]["core_focus"]
        axial=runs[name]["axial_D95"]
        ax1.plot(axial["offset_um"],np.asarray(axial["diameter_nm"])*.001,color=color,label=label,linewidth=1.8)
        ax2.plot(scan["offset_um"],np.asarray(scan["current_uA"])[:,1],color=color,linewidth=1.8)
        ax1.scatter(0,runs[name]["surface"]["D95_nm"]*.001,color=color,s=20,zorder=3)
        ax2.scatter(0,runs[name]["fixed_axis_core_50nm_uA"],color=color,s=20,zorder=3)
    for ax in (ax1,ax2):
        ax.axvline(0,color="#606975",linestyle=":",linewidth=1)
        ax.set(xlim=(-40,40),xlabel="Distance from specimen upper surface (µm)")
        ax.grid(axis="y",alpha=.2)
    ax1.set(ylim=(0,3.5),ylabel="D95 diameter (µm)")
    ax2.set(ylim=(0,8),ylabel="Current within 50 nm radius (µA)")
    ax1.set_title("Diameter containing 95% of the current",loc="left")
    ax2.set_title("Bright central core",loc="left")
    fig.legend(*ax1.get_legend_handles_labels(),loc="lower center",bbox_to_anchor=(.52,.022),
               frameon=False,ncols=2)
    fig.savefig(HERE/"focus-audit-axial.png",dpi=155)
    plt.close(fig)

    from matplotlib.colors import LogNorm
    from matplotlib.patches import Circle
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,4.8))
    fig.subplots_adjust(left=.084,right=.849,top=.79,bottom=.16,wspace=.26)
    fig.text(.084,.94,"Probe at the specimen upper surface",fontsize=18,weight="bold",color=ink)
    fig.text(.084,.86,"Absolute landing current in 10 nm pixels. The cyan circle marks the 50 nm core.",
             fontsize=10,color="#465461")
    panels = [(ax1,STANDARD,"Surface D95 focus"),
              (ax2,"focus-audit-bright-core","Surface bright-core focus")]
    density_maps=[]
    for _,name,_ in panels:
        data=runs[name]["landing_map"]
        edges_nm=np.asarray(data["edges_nm"])
        size=len(edges_nm)-1
        density=np.zeros(size*size)
        density[data["pixel_indices_yx"]]=np.asarray(data["pixel_current_nA"])*1000/np.diff(edges_nm)[0]**2
        density_maps.append(density.reshape(size,size))
    norm=LogNorm(.5*min(a[a>0].min() for a in density_maps),max(a.max() for a in density_maps))
    for (ax,name,title),density in zip(panels,density_maps):
        ax.set_facecolor("black")
        im=ax.imshow(np.ma.masked_equal(density,0),origin="lower",interpolation="nearest",
                     extent=[edges_nm[0],edges_nm[-1],edges_nm[0],edges_nm[-1]],cmap="magma",norm=norm)
        ax.add_patch(Circle((0,0),50,color="#65dfd9",fill=False,lw=1.25))
        ax.set(xlabel="x (nm)",ylabel="y (nm)",title=title,xticks=[-800,0,800],yticks=[-800,0,800])
        fraction=runs[name]["landing_map"]["current_pct_in_window"]
        ax.text(.04,.04,f"{fraction:.1f}% of specimen current",transform=ax.transAxes,color="white",fontsize=9)
    cax=fig.add_axes([.888,.194,.018,.555])
    fig.colorbar(im,cax=cax,label="Current density (pA/nm²)")
    fig.savefig(HERE/"focus-audit-specimen.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="action",required=True)
    operations.add_parser("experiment").add_argument("raw_h5",type=Path)
    operations.add_parser("search-core")
    operations.add_parser("core-focus").add_argument("run",choices=(PREVIOUS,STANDARD,"focus-audit-bright-core"))
    operations.add_parser("relay").add_argument("run",choices=(STANDARD,"focus-audit-tl4-native","focus-audit-tl6-native"))
    operations.add_parser("analyze")
    operations.add_parser("figures")
    opts = parser.parse_args()
    if opts.action == "experiment":
        capture_experiment(opts.raw_h5)
    elif opts.action == "search-core":
        search_core_lens()
    elif opts.action == "core-focus":
        result = native_focus(opts.run)
        (RUNS / f"{opts.run}-core-profile.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    elif opts.action == "relay":
        result = hexapole_relay(opts.run)
        (RUNS / f"{opts.run}-relay.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    elif opts.action == "analyze":
        analyze()
    else:
        figures()
