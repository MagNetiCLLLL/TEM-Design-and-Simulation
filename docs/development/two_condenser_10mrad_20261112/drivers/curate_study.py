"""Curate the independent 70 um, alpha95 >= 10 mrad C1+C2 study.

This is an archival/postprocessing driver, not a particle calculation. Complete
source populations are retained at the selected native checkpoint rows.
"""
from __future__ import annotations

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path, PureWindowsPath
import shutil
import sys

import numpy as np

ROOT = next((path for path in Path(__file__).resolve().parents if (path/"agent.md").is_file()), Path.cwd())
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")


def digest_bytes(data):
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def file_identity(path):
    return digest_bytes(Path(path).read_bytes())


def study_contract(report, request):
    """Check recorded mechanics and sampling without relabelling focus failures."""
    execution = report["execution"]
    aperture = execution["field_audit"]["c2_aperture"]
    if (execution.get("column_hardware") != "two_condenser"
            or execution.get("column_mode") != "two_condenser"
            or any(key in execution.get("active_lens_owners", ())
                   for key in ("condenser_lens_3", "adapter_lens"))):
        raise ValueError("This archive requires the independent physical C1+C2 column")
    if (request.get("expected_c2_diameter_mm") != .070
            or execution.get("expected_c2_diameter_mm") != .070
            or aperture.get("expected_diameter_mm") != .070
            or aperture.get("diameter_mm") != .070
            or aperture.get("actual_boundary_radius_mm") != .035
            or not all(aperture.get(key) is True for key in ("installed", "enabled", "inserted"))):
        raise ValueError("Recorded operating opening and captured C2 plate must both be 70 um")
    if execution["emitted_positive_count"] < 1000 or execution["emitted_count"] < 1000:
        raise ValueError("A new physical trial requires at least 1000 original positive-current particles")
    return {"expected_c2_diameter_mm": .070, "minimum_alpha95_mrad": 10.,
            "observed_alpha95_mrad": report["alpha95_mrad"],
            "angle_requirement_passed": report["alpha95_mrad"] >= 10.,
            "focus_requirement_passed": report.get("direct_surface_focus", {}).get("passed") is True,
            "joint_qualification": "Read the independent native-focus, topology and numerical checks; archiving does not turn an exploratory trial into a passing operating point."}


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False)+"\n", encoding="utf-8")


def relative(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path).resolve())


def copy_exact(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    assert destination.read_bytes() == source.read_bytes(), str(destination)


def embedded_manifests(snapshot, archive, label):
    records = []
    for item in snapshot["external_inputs"]:
        row = {key: value for key, value in item.items() if key != "content_hex"}
        content = item.get("content_hex")
        if content is None:
            records.append({**row, "available": False})
            continue
        data = bytes.fromhex(content)
        if sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Snapshot embedded input hash mismatch: "+item["path"])
        name = PureWindowsPath(item["path"]).name
        target = archive/"manifest-inputs"/item["sha256"]/name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Conflicting embedded manifest bytes: "+str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({**row, "available": True, "trial": label,
                        "archive_path": target.relative_to(archive).as_posix(), **digest_bytes(data)})
    return records


def topology_planes(value):
    """Read only fields explicitly declared to contain executed coordinates."""
    result = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("z_mm", "representative_z_mm") and isinstance(item, (int, float)):
                result.append(float(item))
            elif key in ("bounds_mm", "bracket_mm", "radial_bracket_mm", "observation_bounds_mm",
                         "observed_minimum_planes_mm", "neighbour_bracket_mm"):
                result.extend(float(z) for z in np.asarray(item).ravel())
            else:
                result.extend(topology_planes(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(topology_planes(item))
    return result


def compact_observations(source, report, request, destination):
    with np.load(source, allow_pickle=False) as data:
        z, weights, ids = (np.asarray(data[key]) for key in ("z_mm", "weights", "source_id"))
        positive = weights > 0
        if (np.count_nonzero(positive) < 1000 or np.unique(ids[positive]).size != positive.sum()
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Archive requires the complete distinct original source population")
        if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
            raise ValueError("Invalid original checkpoint coordinates")
        focus = np.asarray(request["focus_planes_mm"], dtype=float)
        if focus.ndim != 1 or not len(focus) or not np.isfinite(focus).all():
            raise ValueError("The executed native focus grid is missing")
        # A dense window is needed to observe separated uncorrected D95/RMS
        # minima. Preserve its scalar report in full, not every phase row.
        surface = float(report["surface_z_mm"])
        target_offsets_nm = (-100., -75., -50., -25., -10., -5., 0., 5., 10., 25., 50., 75., 100.)
        landmarks = list(np.linspace(float(focus.min()), float(focus.max()), 5))
        landmarks += [surface+value*1e-6 for value in target_offsets_nm]
        requested = [float(focus[int(np.argmin(np.abs(focus-plane)))]) for plane in landmarks]
        for key in ("d95_waist", "rms_waist_diagnostic"):
            waist = report.get("native_focus", {}).get(key, {})
            for field in ("observed_minimum_planes_mm", "neighbour_bracket_mm"):
                requested.extend(float(value) for value in waist.get(field, ()))
        requested += [float(report["surface_z_mm"]), float(z[0]), float(z[-1])]
        requested += topology_planes(report.get("topology", {}))
        indexes = []
        for plane in requested:
            matches = np.flatnonzero(z == plane)
            if len(matches) != 1:
                raise ValueError(f"No unique exact original checkpoint for {plane!r}")
            indexes.append(int(matches[0]))
        selected = np.unique(indexes)
        alive = np.asarray(data["alive"])
        if alive.shape != (len(z), len(weights)) or np.any(~alive[:-1] & alive[1:] & positive):
            raise ValueError("Malformed or resurrected original population")
        surface_row = int(np.flatnonzero(z == report["surface_z_mm"])[0])
        surface_cohort = alive[surface_row] & positive
        arrays = {key: np.asarray(data[key])[selected] for key in PHASE}
        arrays.update(z_mm=z[selected], alive=alive[selected], original_checkpoint_index=selected)
        for key in data.files:
            if key not in (*PHASE, "z_mm", "alive"):
                arrays[key] = np.asarray(data[key])
        np.savez_compressed(destination/"native-phase.npz", **arrays)

        full = [np.asarray(data[key]) for key in PHASE[:4]]
        moments = {"z_mm": z, "source_id": ids, "weights": weights}
        for name in ("actual", "sample_survivors"):
            counts = np.zeros(len(z), dtype=np.int64)
            sums = np.zeros(len(z))
            means = np.full((len(z), 4), np.nan)
            covariances = np.full((len(z), 4, 4), np.nan)
            for row in range(len(z)):
                keep = alive[row] & positive
                if name == "sample_survivors":
                    keep &= surface_cohort
                counts[row], sums[row] = int(keep.sum()), float(weights[keep].sum())
                if not keep.any():
                    continue
                phase = np.column_stack([value[row, keep] for value in full])
                if not np.isfinite(phase).all():
                    raise ValueError("Nonfinite actual live topology phase")
                w = weights[keep]/sums[row]
                means[row] = np.sum(w[:, None]*phase, axis=0)
                centered = phase-means[row]
                covariances[row] = centered.T@(w[:, None]*centered)
            moments.update({name+"_count": counts, name+"_original_weight": sums,
                            name+"_mean_xy_tx_ty": means, name+"_covariance_xy_tx_ty": covariances})
        np.savez_compressed(destination/"topology-moments.npz", **moments)
    write_json(destination/"phase-selection.json", {
        "original_observations": {"path": relative(source), **file_identity(source)},
        "original_checkpoint_count": len(z), "selected_checkpoint_count": len(selected),
        "selected_original_indexes": selected.tolist(), "selected_planes_mm": z[selected].tolist(),
        "original_focus_checkpoint_count": len(focus),
        "selected_focus_checkpoint_count": int(np.isin(focus, z[selected]).sum()),
        "focus_selection": "Executed D95/RMS minimum and neighbour-bracket rows, exact surface, original endpoints and representative defocus rows; all native scalar focus observations remain in the byte-preserved report",
        "source_particle_count": len(weights), "original_positive_count": int(positive.sum()),
        "weights_and_ids_unchanged": True,
        "scope": "All original particles at selected exact native observations; topology moments cover every original checkpoint; this is not the complete original trajectory array.",
        "topology_moment_order": "x_m,y_m,tx_rad,ty_rad; population-centered covariance with original current weights normalized only for statistics"
    })


def measured_table(records):
    lines = ["# Executed observations", "", "A completed observation is not automatically a focus qualification.", "",
        "| Label | Original positive count | Step (mm) | D95 (nm) | RMS radius (nm) | alpha95 (mrad) | Angular RMS (mrad) | Surface current (uA) | Native D95 placement |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in records:
        r, q = row["report"], row["request"]
        gate = r.get("native_focus", {}).get("surface_gate", {})
        step = float(q.get("step_mm", q.get("step")))
        lines.append(f"| {row['label']} | {r['execution']['emitted_positive_count']} | {step:.8g} | {r.get('d95_nm', float('nan')):.9g} | {r.get('rms_radius_nm', float('nan')):.9g} | {r.get('alpha95_mrad', float('nan')):.9g} | {r.get('angular_rms_mrad', float('nan')):.9g} | {r.get('sample_current_a', 0.)*1e6:.9g} | {gate.get('status', 'NOT_MEASURED')} |")
    return "\n".join(lines)+"\n"


def figures(archive, records, chosen, focus_labels=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.hashsalt": "two-condenser-10mrad-20261112"})
    fine = [row for row in records if row["report"].get("native_focus")
            and max(row["request"]["focus_planes_mm"])-min(row["request"]["focus_planes_mm"]) < .001]
    if focus_labels:
        fine = [row for row in fine if row["label"] in focus_labels]
        if {row["label"] for row in fine} != set(focus_labels):
            raise ValueError("Each explicitly plotted focus trial must have a complete bounded native window")
    if fine:
        fig, axes = plt.subplots(2, 1, figsize=(7.8, 6), sharex=True, constrained_layout=True)
        for row in fine:
            # Retain the full native focus window: an uncorrected high-angle
            # bundle may have separated D95 and RMS minima.
            points = row["report"]["native_focus"]["planes"]
            offset = [point["offset_nm"] for point in points]
            for ax, key in zip(axes, ("d95_nm", "rms_radius_nm")):
                ax.plot(offset, [point[key] for point in points], label=row["label"], linewidth=1.2)
        for ax in axes:
            ax.axvspan(-1., 1., color="#2c8d62", alpha=.08)
            ax.axvline(0., color="black", linestyle="--", linewidth=.8, label="Specimen upper surface")
            ax.grid(alpha=.25)
        axes[0].set_ylabel("Native D95 diameter (nm)")
        axes[1].set_ylabel("RMS radius (nm)")
        axes[1].set_xlabel("Executed plane offset from specimen upper surface (nm)")
        axes[0].set_title("Local geometric through-focus observations; D95 and RMS are separate statistics")
        axes[0].legend(fontsize=7)
        fig.savefig(archive/"native-focus.png", dpi=160)
        fig.savefig(archive/"native-focus.svg", metadata={"Date": None})
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), constrained_layout=True)
    labels = [row["label"] for row in records]
    for ax, key, title in zip(axes, ("d95_nm", "rms_radius_nm", "alpha95_mrad"),
                             ("Surface D95 diameter (nm)", "Surface RMS radius (nm)", "Surface alpha95 (mrad)")):
        values = [row["report"].get(key, np.nan) for row in records]
        ax.bar(np.arange(len(records)), values, color="#2563a5")
        ax.set_xticks(np.arange(len(records)), labels, rotation=60, ha="right", fontsize=7)
        if key == "alpha95_mrad":
            ax.set_ylim(0., max(values)*1.15)
            ax.axhline(10., color="#c73b32", linestyle="--", linewidth=.8, label="Required alpha95 >= 10 mrad")
            ax.legend(fontsize=6)
        else:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.25)
    fig.suptitle("70 um C2 opening: classical geometric candidates; no diffraction-resolution claim")
    fig.savefig(archive/"baseline-comparison.png", dpi=160)
    fig.savefig(archive/"baseline-comparison.svg", metadata={"Date": None})
    plt.close(fig)
    row = next((item for item in records if item["label"] == chosen), records[-1])
    with np.load(archive/"trials"/row["label"]/"topology-moments.npz", allow_pickle=False) as moments:
        z = moments["z_mm"]
        fig, ax = plt.subplots(figsize=(9, 3.8), constrained_layout=True)
        for prefix, label, style in (("actual", "Actual current at each plane", "-"),
                                     ("sample_survivors", "Common surface-surviving cohort", "--")):
            covariance = moments[prefix+"_covariance_xy_tx_ty"]
            radius = np.sqrt(np.maximum(0., covariance[:, 0, 0]+covariance[:, 1, 1]))*1e6
            ax.semilogy(z, radius, style, label=label, linewidth=1.1)
        surface = row["report"]["surface_z_mm"]
        markers = {}
        for label, cohorts in row["report"].get("topology", {}).items():
            if " to " not in label:
                continue
            left, right = label.split(" to ", 1)
            bounds = cohorts.get("actual", {}).get("bounds_mm", ())
            if len(bounds) == 2:
                markers.update({name: value for name, value in zip((left, right), bounds)
                                if name in ("C1", "C2", "Mini")})
        for name, position in markers.items():
            ax.axvline(position, color="#888888", linestyle="--", linewidth=.65)
            ax.text(position, .64, " "+name, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="center", fontsize=8, color="#444444")
        ax.axvline(surface, color="black", linestyle=":", label="Specimen upper surface")
        ax.set_xlabel("Executed axial coordinate (mm)")
        ax.set_ylabel("RMS radius (um)")
        ax.set_title(row["label"]+": physical C1+C2 column envelope; no C3 or corrector")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.savefig(archive/"column-envelope.png", dpi=160)
        fig.savefig(archive/"column-envelope.svg", metadata={"Date": None})
        plt.close(fig)
        counts = moments["actual_count"]
        original_weight = moments["actual_original_weight"]
        current = original_weight*row["report"]["execution"]["reference_current_a"]*1e6
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, constrained_layout=True)
        axes[0].plot(z, current, linewidth=1.1)
        axes[0].set_ylabel("Absolute surviving current (uA)")
        axes[1].plot(z, counts, linewidth=1.1)
        axes[1].set_ylabel("Surviving original particles")
        axes[1].set_xlabel("Executed axial coordinate (mm)")
        plate_faces = row["report"]["execution"]["field_audit"]["c2_aperture"]["event_z_mm"]
        plate_z = .5*(min(plate_faces)+max(plate_faces))
        for ax in axes:
            ax.axvline(plate_z, color="#b45030", linestyle="--", linewidth=.8,
                       label=f"Captured C2 plate centre ({plate_z:g} mm)")
            ax.axvline(surface, color="black", linestyle=":", label="Specimen upper surface")
            ax.grid(alpha=.25)
            ax.set_ylim(bottom=0.)
        axes[0].set_title(row["label"]+": physical interception; original weights are not restored")
        axes[0].legend(fontsize=8)
        fig.savefig(archive/"current-retention.png", dpi=160)
        fig.savefig(archive/"current-retention.svg", metadata={"Date": None})
        plt.close(fig)


def main():
    from verify_compact import verify_trial
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--trial", action="append", required=True, help="LABEL=complete original trial directory")
    parser.add_argument("--selected", help="Label used for the column-envelope illustration; not a qualification assertion")
    parser.add_argument("--primary", help="Label whose measured direct-focus input is copied as the primary diagnostic profile")
    parser.add_argument("--focus-trial", action="append", default=[], help="Explicit native-focus illustration labels; inclusion never implies a passing gate")
    parser.add_argument("--driver", type=Path, action="append", default=[])
    parser.add_argument("--supplement", action="append", default=[], help="ARCHIVE_RELATIVE_PATH=original supplementary observation")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if (archive/"artifact_manifest.json").exists():
        raise ValueError("Refusing to overwrite an already sealed scientific archive")
    if archive.exists() and any(path.name != "README.md" for path in archive.iterdir()):
        raise ValueError("Use a new archive directory or the README-only study skeleton")
    archive.mkdir(parents=True, exist_ok=True)
    records, external, originals, labels = [], {}, {}, set()
    selected_phase_verification = {}
    supplements = {}
    for specification in args.trial:
        label, raw_path = specification.split("=", 1)
        if not label or label in labels or Path(label).name != label or label in (".", ".."):
            raise ValueError("Use unique simple trial labels")
        labels.add(label)
        source = Path(raw_path).resolve()
        report = json.loads((source/"report.json").read_text(encoding="utf-8"))
        request = json.loads((source/"request.json").read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETED_CLASSICAL_TWO_CONDENSER_OBSERVATION":
            raise ValueError("Only complete, explicitly classical observations can be archived")
        if file_identity(source/"observations.npz")["sha256"] != report["observations_sha256"]:
            raise ValueError("Original observation bytes do not match the executed report")
        snapshot = json.loads((source/"inputs.json").read_text(encoding="utf-8"))
        if snapshot["digest"] != report["execution"]["input_digest"]:
            raise ValueError("Snapshot/report input identities differ")
        contract = study_contract(report, request)
        destination = archive/"trials"/label
        destination.mkdir(parents=True)
        for name in ("report.json", "request.json", "input.toml"):
            copy_exact(source/name, destination/name)
        original_input = (source/"inputs.json").read_bytes()
        compressed = gzip.compress(original_input, mtime=0)
        (destination/"inputs.json.gz").write_bytes(compressed)
        assert gzip.decompress(compressed) == original_input
        external[label] = embedded_manifests(snapshot, archive, label)
        compact_observations(source/"observations.npz", report, request, destination)
        selected_phase_verification[label] = verify_trial(source, destination)
        originals[label] = {name: {"path": relative(source/name), **file_identity(source/name)}
                            for name in ("report.json", "request.json", "input.toml", "inputs.json", "observations.npz")}
        records.append({"label": label, "original_directory": relative(source), "report": report,
                        "request": request, "study_contract": contract})
    if args.selected is not None and args.selected not in labels:
        raise ValueError("The selected label was not archived")
    primary = None
    if args.primary is not None:
        chosen = next((item for item in records if item["label"] == args.primary), None)
        if chosen is None or chosen["report"].get("direct_surface_focus", {}).get("passed") is not True:
            raise ValueError("The primary input requires an archived, measured direct-surface-focus result")
        if not chosen["study_contract"]["angle_requirement_passed"]:
            raise ValueError("The primary input must meet alpha95 >= 10 mrad with the captured 70 um opening")
        source = archive/"trials"/args.primary/"input.toml"
        target = archive/"two-condenser-nanoprobe-diffraction.toml"
        copy_exact(source, target)
        primary = {"trial": args.primary, "profile": target.name,
                   "snapshot": "trials/"+args.primary+"/inputs.json.gz",
                   "profile_scope": "Complete measured vacuum diagnostic input; source, specimen insertion and raster flags are not an application-wide settings replacement"}
    write_json(archive/"manifest-inputs"/"index.json", external)
    write_json(archive/"selected-phase-verification.json", {
        "schema": "two-condenser-selected-phase-verification-v1",
        "trials": selected_phase_verification,
        "scope": "Every selected original array was checked byte for byte; retained native minima, brackets and surface statistics were recomputed from completed phase without tracing any source"})
    write_json(archive/"results.json", {"schema": "two-condenser-observation-archive-v1",
        "scope": "Independent 70 um C2 aperture and alpha95 >= 10 mrad classical incident-probe study; no automatic final qualification",
        "requirements": {"c2_diameter_um": 70., "minimum_alpha95_mrad": 10.,
                         "focus_reference": "Actual specimen upper surface; D95 and RMS native minima observed separately",
                         "extra_pre_specimen_crossover": "Rejected in the final Mini-Objective combined field"},
        "primary_input": primary,
        "trials": [{"label": item["label"], "original_directory": item["original_directory"],
                    "study_contract": item["study_contract"],
                    "measurements": {key: value for key, value in item["report"].items()
                        if key not in ("execution", "native_focus", "topology")},
                    "native_focus": {key: value for key, value in item["report"].get("native_focus", {}).items() if key != "planes"},
                    "report_path": "trials/"+item["label"]+"/report.json"} for item in records]})
    (archive/"measured-results.md").write_text(measured_table(records), encoding="utf-8")
    copy_exact(Path(__file__).resolve(), archive/"drivers"/"curate_study.py")
    copy_exact(Path(__file__).with_name("verify_compact.py"), archive/"drivers"/"verify_compact.py")
    for driver in args.driver:
        copy_exact(driver.resolve(), archive/"drivers"/driver.name)
    for specification in args.supplement:
        name, raw_path = specification.split("=", 1)
        destination = (archive/name).resolve()
        source = Path(raw_path).resolve()
        if not destination.is_relative_to(archive) or destination.exists():
            raise ValueError("Supplement must have a new archive-local destination")
        copy_exact(source, destination)
        supplements[destination.relative_to(archive).as_posix()] = {"original_path": relative(source), **file_identity(source)}
    figures(archive, records, args.selected, args.focus_trial)
    files = {path.relative_to(archive).as_posix(): file_identity(path)
             for path in sorted(archive.rglob("*")) if path.is_file()}
    manifest = {"schema": "two-condenser-scientific-artifact-manifest-v1",
        "scope": "Byte-preserved executed reports and manifests; complete original populations at selected native observations",
        "files": files, "original_full_files": originals,
        "original_supplements": supplements,
        "primary_input": primary,
        "verification": "Every original raw observation hash, snapshot/report identity, embedded input hash and uncompressed snapshot bytes were checked before sealing; no original observation was modified.",
        "selection_is_complete_trajectory": False,
        "qualification": "Read original per-trial focus/topology gates and numerical comparisons; sealing does not imply a PASS."}
    write_json(archive/"artifact_manifest.json", manifest)
    print(json.dumps({"archive": str(archive), "files": len(files), "trials": len(records),
                      "total_bytes": sum(row["bytes"] for row in files.values())}, indent=2))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
