"""Scientific plots of actual experimental single-mode Gaussian case exports.

This is deliberately separate from the production-wave comparison publisher.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path("outputs/beamlet-phase-comparison-20261015")
BASELINE = Path("outputs/beamlet-prototype-20261015/specimen")
NAMES = ("baseline", "c3_minus", "c3_plus", "objective_minus", "objective_plus")
LABELS = ("Baseline", "C3 -0.1 pp", "C3 +0.1 pp", "Objective -0.01 pp", "Objective +0.01 pp")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hybrid_policy(receipt):
    quadratic = receipt.get("quadratic_path_integration") or {}
    return {"axial_order": receipt.get("axial_order"), "axial_method": receipt.get("axial_method"),
        "maximum_column_step_mm": receipt.get("maximum_column_step_mm"),
        "quadratic_method": quadratic.get("method"),
        "maximum_quadratic_step_mm": quadratic.get("maximum_quadratic_step_mm")}


def decomposition_sigma_cells(receipt):
    """Recover the width policy on the verified, physically varying lattice."""
    source_path = Path(receipt["input_manifest"])
    if sha(source_path) != receipt["input_manifest_sha256"]:
        raise ValueError("Executed source manifest changed before numerical-policy verification")
    source = read_json(source_path)
    modes = [row for row in source["modes"] if row["mode_id"] == receipt["mode_id"]]
    if len(modes) != 1:
        raise ValueError("Numerical policy needs one identified executed coherent mode")
    original = modes[0]["arrays"]["amplitude"]["shape"]
    item = modes[0]["arrays"]["basis_m"]
    basis_path = source_path.parent/item["file"]
    if basis_path.parent != source_path.parent or sha(basis_path) != item["sha256"]:
        raise ValueError("Executed affine lattice changed before numerical-policy verification")
    basis = np.load(basis_path, allow_pickle=False)
    shape = receipt["shape"]
    refined_basis = basis@np.diag((original[1]/shape[1], original[0]/shape[0]))
    return float(receipt["sigma_amplitude_m"] / np.linalg.svd(refined_basis, compute_uv=False).min())


def require_shared_numerics(receipts, dimension=None, version_audit=None):
    """Permit only the stated refinement and reviewed nonnumerical changes."""
    # Lens changes can change the physical upstream lattice and hence sigma in
    # metres. Its dimensionless sampling policy remains fixed and is verified.
    allowed = {None: {"sigma_amplitude_m", "sigma_intensity_m"}, "axial": {"maximum_column_step_mm"},
               "width": {"shape", "packet_count", "sigma_amplitude_m", "sigma_intensity_m"},
               "quadratic": {"maximum_quadratic_step_mm"}}[dimension]
    configurations = [{**hybrid_policy(r), "sigma_cells": decomposition_sigma_cells(r), **{key: r.get(key) for key in
        ("method", "shape", "packet_count", "sigma_amplitude_m", "sigma_intensity_m")}} for r in receipts]
    implementations = [{Path(key).name: value for key, value in receipt.get("implementation_sha256", {}).items()}
                       for receipt in receipts]
    approved_changes = []
    audit = read_json(version_audit) if version_audit else {}
    if audit and audit.get("schema") != "experimental-nonnumerical-version-audit-v1":
        raise ValueError("Unknown nonnumerical version-audit schema")
    for index in range(1, len(receipts)):
        for key, value in configurations[0].items():
            if key == "sigma_cells" and np.isclose(configurations[index][key], value, rtol=0., atol=2e-14):
                continue
            if key not in allowed and configurations[index][key] != value:
                raise ValueError(f"Uncontrolled numerical change in {dimension or 'lens-case'} comparison: {key}")
        for filename in ("experimental_gaussian_beamlets.py", "gaussian_column_stages.py", "gaussian_quadratic_paths.py"):
            if implementations[0].get(filename) != implementations[index].get(filename):
                raise ValueError(f"Hybrid numerical implementation changed: {filename}")
        for category, before, after in (
                ("replay-script", implementations[0].get("replay_gaussian_corrector.py"), implementations[index].get("replay_gaussian_corrector.py")),
                ("production-source", receipts[0].get("current_production_source_sha256"), receipts[index].get("current_production_source_sha256"))):
            if before == after:
                continue
            matches = [entry for entry in audit.get("changes", [])
                       if entry.get("category") == category
                       and {entry.get("old_sha256"), entry.get("new_sha256")} == {before, after}
                       and entry.get("numerics_unchanged") is True
                       and entry.get("reason") and entry.get("evidence")]
            if len(matches) != 1:
                raise ValueError(f"Unreviewed version difference: {category}; supply its exact nonnumerical version audit")
            if not matches[0].get("evidence_sha256") or sha(Path(matches[0]["evidence"])) != matches[0]["evidence_sha256"]:
                raise ValueError("Nonnumerical version-audit evidence changed")
            approved_changes.append(matches[0])
    if dimension and not any(configurations[0][key] != configurations[1][key] for key in allowed):
        raise ValueError("Refinement evidence did not change its declared numerical dimension")
    return {"refinement_dimension": dimension, "configurations": configurations,
            "version_audit": str(version_audit) if version_audit else None,
            "version_audit_sha256": sha(version_audit) if version_audit else None,
            "approved_nonnumerical_differences": approved_changes}


def validate_convergence_receipt(path, baseline_packet, kind, threshold=.01, version_audit=None):
    """Qualify only a hash-bound, matched-mode empirical hybrid comparison."""
    result = {"kind": kind, "threshold_relative_complex_l2": threshold,
              "status": "PENDING", "receipt": str(path) if path else None}
    if path is None or not path.exists():
        return result
    data = read_json(path)
    if data.get("schema") != "experimental-gaussian-global-cross-v1":
        raise ValueError("Expected a whole-space coherent complex comparison receipt")
    result["receipt_sha256"] = sha(path)
    last = data["radius_results"][-1]
    if "provenance_a" not in data or "provenance_b" not in data:
        # Historical figures remain diagnostic; old receipts without packet
        # hashes cannot promote a newly computed baseline to qualified status.
        result.update(status="UNVERIFIED_LEGACY_RECEIPT",
                      relative_complex_l2=last["global_relative_complex_l2_vs_a"])
        return result
    provenance = [data["provenance_a"], data["provenance_b"]]
    receipts = []
    for observed in provenance:
        packet_path = Path(observed["packet_file"])
        receipt_path = packet_path.with_name(packet_path.name.replace("-packets.npz", "-receipt.json"))
        if receipt_path.resolve() != Path(observed["receipt_file"]).resolve():
            raise ValueError("Convergence evidence refers to a different replay receipt")
        if sha(packet_path) != observed["packet_sha256"] or sha(receipt_path) != observed["receipt_sha256"]:
            raise ValueError("Convergence packet or replay receipt changed after comparison")
        receipts.append(read_json(receipt_path))
    matches = [index for index, observed in enumerate(provenance)
               if Path(observed["packet_file"]).resolve() == baseline_packet.resolve()
               and observed["packet_sha256"] == sha(baseline_packet)]
    if len(matches) != 1:
        raise ValueError("Convergence evidence does not identify the plotted baseline packet exactly once")
    baseline_index = matches[0]
    for key in ("mode_id", "mode_index", "energy_kev", "mode_weight", "wavelength_m",
                "output_plane_z_mm", "executed_input_mode_identity", "state_file_sha256"):
        if provenance[0].get(key) is None or provenance[0].get(key) != provenance[1].get(key):
            raise ValueError(f"Convergence coherent input differs: {key}")
    for key in ("input_manifest_sha256", "start_z_mm", "stop_z_mm", "lens_percent", "state_file_sha256", "method"):
        if receipts[0].get(key) is None or receipts[0].get(key) != receipts[1].get(key):
            raise ValueError(f"Convergence physical replay differs: {key}")
    configuration_audit = require_shared_numerics(receipts, kind, version_audit)
    policies = [hybrid_policy(receipt) for receipt in receipts]
    for policy in policies:
        if (policy["axial_order"] != 4 or not policy["quadratic_method"]
                or not isinstance(policy["maximum_quadratic_step_mm"], (float, int))
                or not 0 < policy["maximum_quadratic_step_mm"] < float("inf")):
            raise ValueError("Final empirical qualification requires the declared hybrid quadratic integration")
    for key in ("axial_order", "axial_method", "quadratic_method", "maximum_quadratic_step_mm"):
        if policies[0][key] != policies[1][key]:
            raise ValueError(f"Convergence hybrid settings differ: {key}")
    for filename in ("experimental_gaussian_beamlets.py", "gaussian_column_stages.py", "gaussian_quadratic_paths.py"):
        identities = [{Path(key).name: value for key, value in receipt["implementation_sha256"].items()}
                      for receipt in receipts]
        if not identities[0].get(filename) or identities[0].get(filename) != identities[1].get(filename):
            raise ValueError(f"Convergence changed the hybrid numerical implementation: {filename}")
    if kind == "axial":
        if (receipts[0]["packet_count"] != receipts[1]["packet_count"]
                or receipts[0]["sigma_amplitude_m"] != receipts[1]["sigma_amplitude_m"]
                or policies[0]["maximum_column_step_mm"] == policies[1]["maximum_column_step_mm"]):
            raise ValueError("Axial evidence must vary only the outer step, retaining the packet decomposition")
    elif kind == "width":
        if (receipts[0]["packet_count"] == receipts[1]["packet_count"]
                or receipts[0]["sigma_amplitude_m"] == receipts[1]["sigma_amplitude_m"]
                or policies[0]["maximum_column_step_mm"] != policies[1]["maximum_column_step_mm"]):
            raise ValueError("Width evidence must vary the packet decomposition at the same hybrid axial steps")
    else:
        raise ValueError("Unknown convergence dimension")
    suffix = "a" if baseline_index == 0 else "b"
    difference = float(last["global_relative_complex_l2_vs_"+suffix])
    tail = float(last["analytic_omitted_pair_absolute_cross_bound"])
    norm = float(data["norm_"+suffix])
    if (last.get("piston_fit_applied") is not False or not np.isfinite(difference)
            or difference < 0 or not np.isfinite(tail) or tail < 0 or not np.isfinite(norm) or norm <= 0):
        raise ValueError("Invalid or phase-fitted convergence measurement")
    # Include the reported cross-pair truncation uncertainty in the empirical
    # threshold decision; this remains separate from roundoff and model error.
    with_cross_tail = float(np.sqrt(difference*difference+2*tail/norm))
    result.update(status="PASS" if with_cross_tail <= threshold else "FAIL",
        relative_complex_l2=difference, relative_l2_including_cross_tail=with_cross_tail,
        analytic_omitted_cross_bound=tail, baseline_packet_sha256=provenance[baseline_index]["packet_sha256"],
        comparison_packet_sha256=provenance[1-baseline_index]["packet_sha256"], hybrid_policies=policies,
        production_source_digests=[receipt.get("current_production_source_sha256") for receipt in receipts],
        configuration_audit=configuration_audit,
        full_wave_validation="NOT_ESTABLISHED", piston_removed=False)
    return result


def settings(case, inputs_root):
    path = inputs_root/case/"state.json"
    nodes = read_json(path)["nodes"]
    values = {n["attributes"]["key"]: n["attributes"]["percent"] for n in nodes
              if "percent" in n.get("attributes", {}) and "key" in n["attributes"]}
    return {"state_file": str(path), "state_sha256": sha(path), "lens_percent": values}


def main():
    global ROOT, BASELINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--inputs-root", type=Path, default=Path("outputs/beamlet-case-inputs-20261015"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--packet-stem", default="mode0-n360")
    parser.add_argument("--readout-tag", default="core256")
    parser.add_argument("--intensity-log-decades", type=float, default=8.,
                        help="Shared absolute intensity display range below the largest case peak")
    parser.add_argument("--axial-validation", type=Path, default=Path("outputs/beamlet-prototype-20261015/axial-refinement/global-cross-step05-step025.json"))
    parser.add_argument("--width-validation", type=Path,
                        help="Hash-bound whole-space comparison of two packet widths at the plotted baseline")
    parser.add_argument("--version-audit", type=Path,
                        help="Explicit audit of exact nonnumerical replay/source digest changes")
    args = parser.parse_args()
    if not np.isfinite(args.intensity_log_decades) or not 1 <= args.intensity_log_decades <= 16:
        raise ValueError("Use a finite shared intensity range between 1 and 16 decades")
    ROOT, BASELINE = args.root, args.baseline
    rows = []
    for name, label in zip(NAMES, LABELS):
        folder = BASELINE if name == "baseline" else ROOT/name
        receipt_path = folder/(args.packet_stem+"-receipt.json")
        receipt = read_json(receipt_path)
        packet_path = folder/(args.packet_stem+"-packets.npz")
        packet_sha = sha(packet_path)
        readout_path = folder/(args.packet_stem+"-"+args.readout_tag+".npz")
        readout = read_json(readout_path.with_suffix(".json"))
        if readout["source_receipt_sha256"] != sha(receipt_path):
            raise ValueError(f"{name}: readout was generated from a different replay receipt")
        if readout.get("source_packet_sha256") != packet_sha:
            raise ValueError(f"{name}: readout is not bound to these packet bytes; regenerate it")
        with np.load(readout_path, allow_pickle=False) as d:
            axis, field = d["axis_m"], d["complex_amplitude_density"]
        profile_path = folder/(args.packet_stem+("-profiles-wide.npz" if name.startswith("objective") else "-profiles.npz"))
        with np.load(profile_path, allow_pickle=False) as d:
            profiles = {key: d[key] for key in d.files}
        profile_record = read_json(profile_path.with_suffix(".json"))
        if profile_record["source_receipt_sha256"] != sha(receipt_path):
            raise ValueError(f"{name}: line profiles were generated from a different replay receipt")
        if profile_record.get("source_packet_sha256") != packet_sha:
            raise ValueError(f"{name}: line profiles are not bound to these packet bytes; regenerate them")
        state = settings(name, args.inputs_root)
        if "lens_percent" in receipt:
            for key, value in receipt["lens_percent"].items():
                if state["lens_percent"].get(key) != value:
                    raise ValueError(f"{name}: packet receipt and state disagree on {key}")
        manifest_path = Path(receipt["input_manifest"])
        if sha(manifest_path) != receipt["input_manifest_sha256"]:
            raise ValueError("Changed upstream executed manifest")
        if len(read_json(manifest_path)["modes"]) != 9:
            raise ValueError("Upstream nine-mode provenance is incomplete")
        norm_path = folder/(args.packet_stem+"-global-norm.json")
        norm_record = read_json(norm_path) if norm_path.exists() else None
        norm = norm_record["radius_results"][-1]["coherent_norm"] if norm_record else None
        moments_path = folder/(args.packet_stem+"-global-moments.json")
        moments = read_json(moments_path) if moments_path.exists() else None
        for derived in (norm_record, moments):
            if derived is not None:
                provenance = derived.get("provenance", {})
                if provenance.get("packet_sha256") != packet_sha or provenance.get("receipt_sha256") != sha(receipt_path):
                    raise ValueError(f"{name}: global norm/moments refer to different packet or receipt bytes")
        if norm is None and moments:
            norm, norm_path = moments["coherent_norm"], moments_path
        intensity = abs(field)**2*receipt["weight"]
        record = {"case": name, "label": label, **state, "energy_kev": receipt["energy_kev"],
            "mode_id": receipt["mode_id"], "mode_weight": receipt["weight"],
            "sample_entrance_z_mm": receipt["stop_z_mm"],
            "upstream_all_nine_modes_present": True, "upstream_manifest": str(manifest_path),
            "upstream_manifest_sha256": receipt["input_manifest_sha256"],
            "receipt": str(receipt_path), "receipt_sha256": sha(receipt_path),
            "packet": str(packet_path), "packet_sha256": packet_sha,
            "hybrid_policy": hybrid_policy(receipt),
            "readout": str(readout_path), "readout_sha256": sha(readout_path),
            "line_profiles": str(profile_path), "line_profiles_sha256": sha(profile_path),
            "global_coherent_norm": norm, "global_norm_receipt": str(norm_path) if norm is not None else None,
            "global_moments_receipt": str(moments_path) if moments else None,
            "global_rms_radius_nm": moments["rms_radius_nm"] if moments else None,
            "global_sigma_xy_nm": moments["sigma_xy_nm"] if moments else None,
            "global_mode_probability": norm*receipt["weight"] if norm is not None else None,
            "roi_conditional_probability": readout["viewport_probability_midpoint_quadrature"],
            "roi_fraction_of_mode_current": readout["viewport_probability_midpoint_quadrature"]/norm if norm is not None else None,
            "max_packet_phase_cycles_per_pixel": readout["conservative_max_packet_phase_cycles_per_pixel"],
            "initial_reconstruction_relative_complex_l2": receipt["initial_reconstruction"]["relative_complex_l2"],
            "accumulated_screen_triangle_l2_bound": receipt["triangle_l2_bound"],
            "bore_omission_l2_bound": receipt["bore_omission_l2_bound"]}
        rows.append(dict(name=name, label=label, axis=axis, field=field, intensity=intensity,
                         receipt=receipt, readout=readout, profiles=profiles, record=record))
    first = rows[0]
    case_configuration_audit = require_shared_numerics([row["receipt"] for row in rows], version_audit=args.version_audit)
    for row in rows[1:]:
        if not np.array_equal(row["axis"], first["axis"]):
            raise ValueError("Case physical lab grids differ")
        for key in ("mode_id", "energy_kev", "start_z_mm", "stop_z_mm"):
            if row["receipt"][key] != first["receipt"][key]:
                raise ValueError(f"Case physical coherent reference differs: {key}")
        if row["record"]["hybrid_policy"] != first["record"]["hybrid_policy"]:
            raise ValueError("Lens cases must use the same numerical integration policy")
        changes = {k: v-first["record"]["lens_percent"][k] for k, v in row["record"]["lens_percent"].items()
                   if v != first["record"]["lens_percent"].get(k)}
        expected_key = "condenser_lens_3" if row["name"].startswith("c3") else "objective_lens"
        expected_delta = (.1 if expected_key == "condenser_lens_3" else .01)*(-1 if row["name"].endswith("minus") else 1)
        if set(changes) != {expected_key} or not np.isclose(changes[expected_key], expected_delta, rtol=0., atol=1e-12):
            raise ValueError(f"Unexpected lens controls for {row['name']}: {changes}")
        row["record"]["lens_changes_percentage_points"] = changes
        a = first["field"]*np.sqrt(first["receipt"]["weight"])
        b = row["field"]*np.sqrt(row["receipt"]["weight"])
        ia, ib = abs(a)**2, abs(b)**2
        valid = (ia >= 1e-4*ia.max()) & (ib >= 1e-4*ib.max())
        phase = np.angle(b*np.conj(a))
        weights = np.sqrt(ia[valid]*ib[valid])
        row["phase_difference"], row["difference_mask"] = phase, valid
        row["record"]["relative_to_baseline_in_roi"] = {
            "relative_complex_l2": float(np.linalg.norm(b-a)/np.linalg.norm(a)),
            "relative_intensity_l1": float(np.sum(abs(ib-ia))/ia.sum()),
            "relative_intensity_l2": float(np.linalg.norm(ib-ia)/np.linalg.norm(ia)),
            "masked_weighted_phase_rms_rad": float(np.sqrt(np.dot(weights, phase[valid]**2)/weights.sum())),
            "masked_phase_max_rad": float(abs(phase[valid]).max()),
            "phase_mask_reference_current_fraction": float(ia[valid].sum()/ia.sum()),
            "phase_mask_case_current_fraction": float(ib[valid].sum()/ib.sum()),
            "phase_difference_is_wrapped": True, "piston_removed": False, "normalization_fitted": False}
    output = args.output or ROOT/"figures"
    output.mkdir(parents=True, exist_ok=True)
    z, energy = first["receipt"]["stop_z_mm"], first["receipt"]["energy_kev"]
    c3 = first["record"]["lens_percent"]["condenser_lens_3"]
    obj = first["record"]["lens_percent"]["objective_lens"]
    baseline_packet = BASELINE/(args.packet_stem+"-packets.npz")
    axial = validate_convergence_receipt(args.axial_validation, baseline_packet, "axial", version_audit=args.version_audit)
    width = validate_convergence_receipt(args.width_validation, baseline_packet, "width", version_audit=args.version_audit)
    empirical_pass = axial["status"] == width["status"] == "PASS"
    if empirical_pass:
        qualification = "BASELINE AXIAL + WIDTH CHECKS PASSED (1% L2)"
    elif "FAIL" in (axial["status"], width["status"]):
        qualification = "UNQUALIFIED: NUMERICAL REFINEMENT FAILED"
    else:
        qualification = "NUMERICAL QUALIFICATION INCOMPLETE"
    banner = ("EXPERIMENTAL — " if empirical_pass else "DIAGNOSTIC ONLY — ")+qualification
    max_i = max(float(row["intensity"].max()) for row in rows)*1e-18
    if not np.isfinite(max_i) or max_i <= 0:
        raise ValueError("No finite positive coherent intensity is available for a logarithmic display")
    min_i = max_i*10.**(-args.intensity_log_decades)
    summary = {"scope": "EXPERIMENTAL semiclassical coherent Gaussian calculation after an actual historical executed upstream wave; not certified full-wave accuracy",
        "numerical_qualification_status": qualification,
        "empirical_baseline_numerical_checks": {"axial": axial, "packet_width": width},
        "full_wave_model_validation": "NOT_ESTABLISHED",
        "lens_case_configuration_audit": case_configuration_audit,
        "mode_scope": "One fixed coherent E0 component. All nine upstream energy modes were retained; the displayed phase and intensity are not the incoherent mixture.",
        "sample_entrance_z_mm": z, "energy_kev": energy,
        "control_definition": "C3 +/-0.1 percentage points; shared whole-objective excitation +/-0.01 percentage points. CM, other lens settings and specimen entrance are fixed; no autofocus.",
        "roi_width_nm": first["readout"]["width_nm"], "roi_pixels": first["readout"]["pixels"], "roi_pitch_pm": first["readout"]["pitch_nm"]*1000,
        "line_profiles": "Baseline/C3: 40nm X/Y cuts at8pm; objective cases:500nm cuts at10pm. Not projected marginals or a2D whole-beam image.",
        "phase_dark_mask_fraction_of_each_peak": 1e-4, "normalization_fitted": False, "piston_removed": False,
        "intensity_display": {"scale": "shared absolute logarithmic", "unit": "E0 probability per reference electron per nm^2",
            "minimum": min_i, "maximum": max_i, "zero_or_below_range_color": "black", "raw_arrays_are_linear": True},
        "readout_evidence": {"global_width_comparison": str(args.width_validation) if args.width_validation and args.width_validation.exists() else None,
            "global_axial_comparison": str(args.axial_validation) if args.axial_validation and args.axial_validation.exists() else None,
            "baseline_pixel_quadrature": str(BASELINE/"core-quadrature-256-320.json") if (BASELINE/"core-quadrature-256-320.json").exists() else None,
            "readout_implementation_checks": "tests/test_experimental_gaussian_beamlets.py"},
        "limitations": ["The recorded refinement checks apply to this plotted baseline, not automatically to every perturbed lens setting.",
            "Passing empirical numerical checks does not certify full-wave physics or real-instrument calibration.",
            "The accumulated screen triangle bounds are conservative and not sufficiently small to certify full-wave accuracy.",
            "Width convergence is empirical and does not cover all possible model/discretization errors.",
            "Central images show only part of the current; full-space Gaussian Gram norms are reported separately.",
            "Line profiles sample only the central X/Y axes. No two-dimensional whole-beam profile is inferred."],
        "cases": [row["record"] for row in rows]}
    (output/"comparison-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    half_width = first["readout"]["width_nm"]/2
    extent = (-half_width, half_width, -half_width, half_width)
    intensity_norm = LogNorm(vmin=min_i, vmax=max_i)
    intensity_colors = plt.get_cmap("magma").copy()
    intensity_colors.set_bad("black")
    intensity_colors.set_under("black")
    fig, axes = plt.subplots(2, 5, figsize=(17.6, 7.6), constrained_layout=True)
    for col, row in enumerate(rows):
        im_i = axes[0, col].imshow(row["intensity"]*1e-18, origin="lower", extent=extent,
                                 cmap=intensity_colors, norm=intensity_norm)
        axes[0, col].set_title(row["label"], fontsize=12)
        phase = np.ma.masked_where(row["intensity"] < 1e-4*row["intensity"].max(), np.angle(row["field"]))
        im_p = axes[1, col].imshow(phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, col].set_facecolor("#e5e5e5")
        fraction = row["record"]["roi_fraction_of_mode_current"]
        axes[1, col].set_title(f"ROI current {fraction:.3%}" if fraction is not None else "ROI current: norm pending", fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel("Lab X (nm)"); ax.set_ylabel("Lab Y (nm)")
    fig.colorbar(im_i, ax=axes[0, :], shrink=.85, label="Log scale: E0 contribution / (reference electron nm$^2$)")
    fig.colorbar(im_p, ax=axes[1, :], shrink=.85, label="Physical wrapped phase (rad)")
    fig.suptitle(banner+"\nSpecimen coherent component: intensity (top) and phase (bottom)\n"
                 f"Z={z:.7f} mm | E0={energy:.9f} keV | baseline C3={c3:.7f}%, whole objective={obj:.5f}%", fontsize=14)
    fig.supxlabel("Common physical grid and shared absolute logarithmic intensity scale. pp = percentage points. No autofocus or phase-piston removal.\n"
                  "Central 2 nm crops are partial-current views; gray masks hide dark phase. The nine-mode incoherent mixture has no single phase.", fontsize=10)
    fig.savefig(output/"specimen-intensity-phase.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.6), constrained_layout=True)
    diff_scale = max(float(abs(row["intensity"]-first["intensity"]).max()) for row in rows[1:])*1e-18
    for col, row in enumerate(rows[1:]):
        im_i = axes[0, col].imshow((row["intensity"]-first["intensity"])*1e-18, origin="lower", extent=extent,
                                  cmap="RdBu_r", vmin=-diff_scale, vmax=diff_scale)
        axes[0, col].set_title(row["label"]+" minus baseline", fontsize=11)
        phase = np.ma.masked_where(~row["difference_mask"], row["phase_difference"])
        im_p = axes[1, col].imshow(phase, origin="lower", extent=extent, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1, col].set_facecolor("#e5e5e5")
        metric = row["record"]["relative_to_baseline_in_roi"]
        axes[1, col].set_title(f"Phase RMS {metric['masked_weighted_phase_rms_rad']:.4g} rad", fontsize=10)
    for ax in axes.flat:
        ax.set_xlabel("Lab X (nm)"); ax.set_ylabel("Lab Y (nm)")
    fig.colorbar(im_i, ax=axes[0, :], shrink=.85, label="Change / (reference electron nm$^2$)")
    fig.colorbar(im_p, ax=axes[1, :], shrink=.85, label="Wrapped phase difference (rad)")
    fig.suptitle(banner+"\nLens sensitivity: intensity difference (top); physical phase difference (bottom)", fontsize=14)
    fig.supxlabel("Each case keeps the same source and all other lens controls. Objective changes use the shared control for both halves.\n"
                  "No phase piston removed. Bright-mask phase RMS is over this 2 nm crop; phase is not an incoherent-mixture observable.", fontsize=10)
    fig.savefig(output/"specimen-lens-differences.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.), constrained_layout=True)
    peak = max(float(abs(row["profiles"][axis+"_amplitude_density"]).max()**2*row["receipt"]["weight"]*1e-18)
               for row in rows for axis in ("x", "y"))
    for group, selected, limit in ((0, rows[:3], 20.), (1, rows[3:], 250.)):
        for col, axis in enumerate(("x", "y")):
            ax = axes[group, col]
            for row in selected:
                density = abs(row["profiles"][axis+"_amplitude_density"])**2*row["receipt"]["weight"]*1e-18
                rms = row["record"]["global_rms_radius_nm"]
                label = row["label"]+(f" (RMS radius {rms:.3f}nm)" if rms is not None else "")
                ax.semilogy(row["profiles"]["axis_m"]*1e9, density, label=label, lw=1.1)
            ax.axvspan(-half_width, half_width, color="gray", alpha=.12, label="Central image crop")
            ax.set_xlim(-limit, limit); ax.set_ylim(peak*1e-10, peak*1.5)
            ax.set_xlabel(f"Lab {axis.upper()} (nm)")
            ax.set_ylabel("E0 contribution / (reference electron nm$^2$)")
            ax.set_title(f"{'Baseline/C3' if group == 0 else 'Whole-objective'}: {axis.upper()} cut, {'Y' if axis == 'x' else 'X'}=0")
            ax.grid(alpha=.2)
            if col == 0: ax.legend(fontsize=8, loc="lower center")
    fig.suptitle(banner+"\nIntensity line profiles: top 40 nm at 8 pm; bottom 500 nm at 10 pm. Cuts, not marginals.", fontsize=13)
    fig.supxlabel("Common absolute scale; no curve normalization. Gray band is the 2 nm region shown in the phase figures.\n"
                  "Global current and moments come from the independent continuous-space coherent Gaussian overlap calculation.", fontsize=10)
    fig.savefig(output/"specimen-wide-line-profiles.png", dpi=160); plt.close(fig)
    print(json.dumps({"output": str(output), "cases": [{"case": row["name"], "roi_fraction": row["record"]["roi_fraction_of_mode_current"],
        "metrics": row["record"].get("relative_to_baseline_in_roi")} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
