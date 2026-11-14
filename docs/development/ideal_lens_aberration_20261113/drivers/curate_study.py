"""Seal completed Gaussian-lens research without modifying original evidence.

This driver performs only archival copies, array statistics and figure
generation. It never emits or propagates a source or solves a field.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(path for path in HERE.parents if (path/"agent.md").is_file())
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts"), str(HERE)]
from ideal_lens_beam import array_hash, phase_statistics
from make_figures import make_figures


def identity(path):
    data = Path(path).read_bytes()
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def copy_exact(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"Archive destination already exists: {destination}")
    shutil.copyfile(source, destination)
    if source.read_bytes() != destination.read_bytes():
        raise ValueError("An archival byte copy changed")


def finalize_prepared(archive, late_supplements=(), late_drivers=()):
    """Verify a prepared copy, add late receipts and seal once, without reruns."""
    archive = Path(archive).resolve()
    if (archive/"artifact_manifest.json").exists():
        raise ValueError("Refusing to reseal an existing scientific archive")
    if not (archive/"README.md").is_file():
        raise ValueError("Write the final evidence-based README before sealing")
    receipt = json.loads((archive/"curation-preparation.json").read_text())
    for name, expected in receipt["files"].items():
        if identity(archive/name) != expected:
            raise ValueError(f"Prepared scientific artifact changed before sealing: {name}")
    for specification in late_supplements:
        name, raw = specification.split("=", 1)
        target = (archive/name).resolve()
        if not target.is_relative_to(archive):
            raise ValueError("Late supplement must stay inside the new archive")
        copy_exact(raw, target)
        receipt["original_supplements"][name] = {"original_path": str(Path(raw).resolve()), **identity(raw)}
    for source in late_drivers:
        copy_exact(source, archive/"drivers"/Path(source).name)
    files = {path.relative_to(archive).as_posix(): identity(path) for path in sorted(archive.rglob("*")) if path.is_file()}
    receipt["files"] = files
    receipt["qualification"] = "Read each original numerical/field/pupil/source gate. High-angle exploration and retained fit failures are not promoted to PASS by archiving."
    receipt["preparation_receipt"] = "curation-preparation.json"
    write_json(archive/"artifact_manifest.json", receipt)
    return {"archive": str(archive), "registered_files": len(files), "registered_bytes": sum(row["bytes"] for row in files.values())}


def assert_statistics(actual, expected, context):
    if set(actual) != set(expected):
        raise ValueError(f"Statistic schema differs: {context}")
    for key, value in expected.items():
        computed = actual[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isclose(computed, value, rtol=2e-13, atol=1e-24):
                raise ValueError(f"Native statistic differs: {context}.{key}")
        elif isinstance(value, list):
            if not np.allclose(computed, value, rtol=2e-13, atol=1e-24):
                raise ValueError(f"Native list statistic differs: {context}.{key}")
        elif computed != value:
            raise ValueError(f"Native statistic differs: {context}.{key}")


def select_rows(report, z):
    image = int(report["observations"]["image_index"])
    focus = np.asarray(report["observations"]["focus_indices"], int)
    selected = {0, len(z)-1, image}
    # Preserve native spatial minima and their immediate original neighbours
    # independently for both source populations and each physical tube.
    for population in report["populations"]:
        for view in population["tube_views"]:
            planes = view["focus_planes"]
            for metric in ("d95_nm", "rms_radius_nm"):
                values = [row.get(metric, np.inf) for row in planes]
                minimum = int(np.argmin(values))
                for local in (max(0, minimum-1), minimum, min(len(focus)-1, minimum+1)):
                    selected.add(int(focus[local]))
            for local in np.linspace(0, len(focus)-1, 9).round().astype(int):
                selected.add(int(focus[local]))
    # A few executed whole-path landmarks preserve an interpretable envelope;
    # continuous wall-contact events remain in their separate complete ledger.
    for local in np.linspace(0, len(z)-1, 9).round().astype(int):
        selected.add(int(local))
    return np.asarray(sorted(selected), dtype=np.int64)


def compact_case(original, destination, reference_sha):
    report = json.loads((original/"report.json").read_text())
    if report.get("schema") != "ideal-lens-finite-source-study-v1":
        raise ValueError("Only complete declared finite-source observations may be curated")
    if report["request"]["reference_sha256"] != reference_sha:
        raise ValueError("The original case used another intrinsic reference file")
    for name, digest in report["request"]["observer_sha256"].items():
        if identity(ROOT/name)["sha256"] != digest:
            raise ValueError(f"Current research driver differs from its executed receipt: {name}; archive its exact historical version")
    ledger = json.loads((original/"files.json").read_text())
    for name, digest in ledger.items():
        if Path(name).name != name or identity(original/name)["sha256"] != digest:
            raise ValueError(f"Raw scientific receipt mismatch: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("request.json", "field.json", "source.npz", "report.json", "files.json"):
        copy_exact(original/name, destination/name)
    with np.load(original/"source.npz", allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    ids = source["source_id"]
    if (ids.ndim != 1 or len(ids) != 2048 or np.unique(ids).size != 2048
            or not np.array_equal(ids, np.arange(2048))
            or not np.isfinite(source["initial_phase"]).all()):
        raise ValueError("The complete declared original source is missing")
    z = np.asarray(report["observations"]["z_m"], float)
    if z.ndim != 1 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0):
        raise ValueError("Executed observation planes are malformed")
    chosen = select_rows(report, z)
    parts, byte_checks = [], []
    for part in (0, 1):
        name = f"characteristics-{part}.npz"
        with np.load(original/name, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        phase = arrays["phase"]
        if phase.shape != (len(z), 1024, 5) or not np.isfinite(phase).all():
            raise ValueError("A 1024-original-particle phase partition is malformed")
        if not np.array_equal(arrays["z_m"], z):
            raise ValueError("The phase observation coordinates differ from the completed report")
        section = slice(part*1024, (part+1)*1024)
        if (not np.array_equal(arrays["source_id"], ids[section])
                or not np.array_equal(arrays["kinetic_energy_ev"], source["kinetic_energy_ev"][section])
                or not np.array_equal(phase[0], source["initial_phase"][section])):
            raise ValueError("The phase partition changed its original source identity")
        if array_hash(phase) != report["partitions"][part]["retained_phase_sha256"]:
            raise ValueError("The reported complete raw partition identity differs")
        expected_weight = 1/1024 if part == 0 else 1/2048
        if not np.array_equal(arrays["historical_population_weights"], np.full(1024, expected_weight)):
            raise ValueError("Historical source normalization was silently rewritten")
        compact = {**arrays, "z_m": z[chosen], "phase": phase[chosen], "original_checkpoint_index": chosen}
        path = destination/f"characteristics-{part}-selected.npz"
        np.savez_compressed(path, **compact)
        with np.load(path, allow_pickle=False) as saved:
            for key in arrays:
                expected = arrays[key][chosen] if key in ("phase", "z_m") else arrays[key]
                actual = np.asarray(saved[key])
                if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
                    raise ValueError(f"Selected original phase bytes differ: {part}:{key}")
        byte_checks.append({"partition": part, "original": str(original/name), **identity(original/name),
                            "checked_original_arrays": list(arrays), "selected_original_rows": chosen.tolist()})
        parts.append(phase)
    complete_phase = np.concatenate(parts, axis=1)
    walls = {}
    for diameter in report["request"]["tube_diameters_mm"]:
        wall_parts = []
        for part in (0, 1):
            name = f"wall-{diameter:g}mm-part{part}.npz"
            copy_exact(original/name, destination/name)
            with np.load(original/name, allow_pickle=False) as archive:
                values = {key: np.asarray(archive[key]) for key in archive.files}
            if (not np.array_equal(values["source_ids"], ids[part*1024:(part+1)*1024])
                    or np.any(values["unresolved_mask"])):
                raise ValueError("The complete physical wall classification is missing or unresolved")
            wall_parts.append(values)
        walls[float(diameter)] = {key: np.concatenate([row[key] for row in wall_parts], axis=0) for key in wall_parts[0]}
    image = int(report["observations"]["image_index"])
    scalar_checks, moments = [], {"z_m": z, "source_id": ids}
    for population in report["populations"]:
        count = int(population["original_positive_count"])
        if count not in (1024, 2048) or population["positive_weight_count"] != count:
            raise ValueError("Original positive-current population does not match its declaration")
        weights = np.full(count, 1/count)
        moments[f"population_{count}_weights"] = weights
        if array_hash(source["initial_phase"][:count]) != population["source_phase_hash"]:
            raise ValueError("An original source prefix changed")
        for view in population["tube_views"]:
            diameter = float(view["tube_diameter_mm"])
            hit = walls[diameter]["hit_z_m"][:count]
            hit = np.where(np.isnan(hit), np.inf, hit)
            actual = phase_statistics(complete_phase[image, :count], z[image] < hit, weights, report["request"]["source_current_a"])
            assert_statistics(actual, view["at_paraxial_image"], f"N={count}, tube={diameter}: image")
            for row, expected in zip(report["observations"]["focus_indices"], view["focus_planes"]):
                actual = phase_statistics(complete_phase[row, :count], z[row] < hit, weights, report["request"]["source_current_a"])
                expected = {key: value for key, value in expected.items() if key not in ("z_m", "offset_from_paraxial_nm")}
                assert_statistics(actual, expected, f"N={count}, tube={diameter}: focus")
            before = np.isfinite(hit) & (hit <= z[image])
            if (int(before.sum()) != view["intercepted_before_image"]
                    or not np.isclose(weights[before].sum()*report["request"]["source_current_a"], view["intercepted_current_a"], rtol=2e-13, atol=1e-24)):
                raise ValueError("The original absolute interception ledger differs")
            prefix = f"n{count}_tube{diameter:g}"
            means, covariance, mass, counts = [], [], [], []
            for row, plane in enumerate(z):
                alive = plane < hit
                current = float(weights[alive].sum())
                counts.append(int(alive.sum()))
                mass.append(current)
                phase = complete_phase[row, :count, :4][alive]
                if not len(phase):
                    means.append(np.full(4, np.nan)); covariance.append(np.full((4, 4), np.nan))
                    continue
                w = weights[alive]/current
                mean = np.sum(w[:, None]*phase, axis=0)
                centered = phase-mean
                means.append(mean); covariance.append(centered.T@(w[:, None]*centered))
            moments.update({prefix+"_mean": np.asarray(means), prefix+"_covariance": np.asarray(covariance),
                            prefix+"_original_weight": np.asarray(mass), prefix+"_count": np.asarray(counts)})
            scalar_checks.append({"population": count, "tube_diameter_mm": diameter,
                "all_original_focus_scalar_statistics_reproduced": True,
                "absolute_current_and_interception_reproduced": True})
    np.savez_compressed(destination/"all-checkpoint-moments.npz", **moments)
    selection = {"schema": "ideal-lens-complete-population-phase-selection-v1",
        "original_directory": str(original), "original_checkpoint_count": len(z),
        "selected_checkpoint_count": len(chosen), "selected_original_rows": chosen.tolist(),
        "selected_z_m": z[chosen].tolist(), "raw_partition_byte_checks": byte_checks,
        "scalar_reproduction_relative_tolerance": 2e-13, "scalar_checks": scalar_checks,
        "historical_1024_weights_preserved": True,
        "combined_2048_weights": "Explicit new population normalization of 1/2048; complete original IDs, no restoration of blocked current",
        "phase_order": "x_m,y_m,dx_dz,dy_dz,flight_time_s; kinetic energy is stored separately",
        "physical_scope": "Raw and selected characteristics after first tube contact are counterfactual vacuum continuation. Only the prefix before each original wall-ledger contact is physical.",
        "selection_is_complete_trajectory": False,
        "selection_scope": "All original particles at selected native D95/RMS minimum/neighbour, image, representative focus and whole-path rows. Complete source and wall ledgers are copied byte for byte; full focus scalars and all-checkpoint moments are retained."}
    write_json(destination/"phase-selection.json", selection)
    return report, {name: {"original_path": str(original/name), **identity(original/name)} for name in ledger}, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--requests", type=Path, action="append", default=[])
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--primary", default="fixed-b0.3-s6")
    parser.add_argument("--driver", type=Path, action="append", default=[])
    parser.add_argument("--supplement", action="append", default=[], help="ARCHIVE_PATH=original file")
    parser.add_argument("--prepare-only", action="store_true", help="Keep verified prepared artifacts open for final README/audit receipts")
    parser.add_argument("--finalize-existing", action="store_true", help="Add late receipts and seal a previously prepared archive")
    args = parser.parse_args()
    archive = args.archive.resolve()
    if args.finalize_existing:
        print(json.dumps(finalize_prepared(archive, args.supplement, args.driver), indent=2))
        return
    if not args.source_root or not args.requests or not args.reference:
        parser.error("Preparation requires --source-root, explicit --requests and --reference")
    if (archive/"artifact_manifest.json").exists() or (archive.exists() and any(p.name != "README.md" for p in archive.iterdir())):
        raise ValueError("Use a new archive or a README-only final study skeleton")
    requests = [request for path in args.requests for request in json.loads(path.read_text())]
    reference = json.loads(args.reference.read_text())
    if not requests or len({r["name"] for r in requests}) != len(requests):
        raise ValueError("A complete unique declared request inventory is required")
    for request in requests:
        if Path(request["name"]).name != request["name"] or not (args.source_root/request["name"]/"report.json").is_file():
            raise ValueError("Every explicitly declared case must have its completed report before sealing")
    archive.mkdir(parents=True, exist_ok=True)
    for path in args.requests:
        copy_exact(path, archive/"declared-requests"/path.name)
    write_json(archive/"case-inventory.json", requests)
    copy_exact(args.reference, archive/"variational-reference"/"report.json")
    cases, originals, checks = [], {}, {}
    reference_sha = identity(args.reference)["sha256"]
    for request in requests:
        name = request["name"]
        report, original_files, checked = compact_case(args.source_root/name, archive/"finite-source"/name, reference_sha)
        if any(report["request"].get(key) != value for key, value in request.items()):
            raise ValueError("A completed case differs from its predeclared request")
        cases.append(report); originals[name] = original_files; checks[name] = checked
        print(f"CURATED {name}: {checked['selected_checkpoint_count']} / {checked['original_checkpoint_count']} native rows; complete original population retained", flush=True)
    for source in [Path(__file__), HERE/"make_figures.py", *args.driver]:
        copy_exact(source.resolve(), archive/"drivers"/source.name)
    supplements = {}
    for specification in args.supplement:
        name, raw = specification.split("=", 1)
        destination = (archive/name).resolve()
        if not destination.is_relative_to(archive):
            raise ValueError("Supplement destination must stay inside the new archive")
        copy_exact(Path(raw), destination)
        supplements[name] = {"original_path": str(Path(raw).resolve()), **identity(raw)}
    figures = make_figures(archive, reference, cases, args.primary)
    write_json(archive/"results.json", {"schema": "ideal-lens-aberration-archive-results-v1",
        "scope": "Synthetic single Gaussian magnetic lens; separate intrinsic reference and finite-source/tube observations, not a microscope preset or FEM material design",
        "case_inventory": [{"name": row["request"]["name"], "family": row["request"]["family"],
            "report": f"finite-source/{row['request']['name']}/report.json",
            "intrinsic_cs_mm": row["intrinsic_cs_mm"], "source_count_comparison": row["source_count_comparison"],
            "legacy_100_mrad_edge_scope_exceeded": row["paraxial_edge_slope_mrad"] > 100.,
            "paraxial_edge_slope_mrad": row["paraxial_edge_slope_mrad"],
            "field_order_evidence": row["field_order_evidence"]} for row in cases], "figures": figures})
    write_json(archive/"selected-phase-verification.json", {"schema": "ideal-lens-selected-phase-verification-v1", "cases": checks})
    files = {path.relative_to(archive).as_posix(): identity(path) for path in sorted(archive.rglob("*")) if path.is_file()}
    write_json(archive/"curation-preparation.json", {"schema": "ideal-lens-scientific-artifact-manifest-v1",
        "files": files, "original_full_files": originals, "original_reference": {"path": str(args.reference.resolve()), **identity(args.reference)},
        "original_supplements": supplements,
        "verification": "Original raw file SHA256, complete source/partition IDs, historical weights, selected raw-array bytes, physical wall ledgers, all focus scalar statistics and absolute loss/current balances checked before sealing",
        "selection_is_complete_trajectory": False,
        "qualification": "PREPARED_NOT_SEALED; await final narrative and audit receipts. Read original numerical/field/pupil/source gates per case; preserved failures remain failures."})
    if not args.prepare_only:
        finalize_prepared(archive)
    print(json.dumps({"archive": str(archive), "cases": len(cases), "registered_files": len(files), "registered_bytes": sum(row["bytes"] for row in files.values())}, indent=2))


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        main()
