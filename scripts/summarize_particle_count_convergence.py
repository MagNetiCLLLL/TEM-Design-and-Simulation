"""Combine completed frozen-source observations without rerunning propagation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from validate_particle_count_convergence import compare_counts, physical_input_signature, execution_identity


def read_observation(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    inputs = json.loads(path.with_suffix(".inputs.json").read_text(encoding="utf-8"))
    signature = physical_input_signature(inputs)
    if record.get("schema") == "particle-count-observation-v1":
        if signature != record["physical_input_signature"]:
            raise ValueError(f"Recorded physical identity disagrees with inputs: {path}")
        result = dict(record)
    elif record.get("schema") in {"unified-nanoprobe-qualification-v1", "unified-nanoprobe-qualification-v2"}:
        surface = record["surface"]
        stats = surface["statistics"]
        result = dict(
            rays=record["rays"], step_mm=record["step_mm"],
            elapsed_s=record["elapsed_s"], physical_input_signature=signature,
            source_population_identity=surface["source_population_identity"],
            specimen=dict(
                alpha95_mrad=stats["convergence_95_rad"]*1000,
                alpha99_mrad=stats["convergence_99_rad"]*1000,
                rms_radius_nm=stats["radius_rms_m"]*1e9,
                diameter95_nm=2*stats["radius_95_m"]*1e9,
                current_a=record["sample_current_a"],
                local_focus_offset_nm=surface["local_waist_offset_nm"],
                statistics=stats,
            ),
            sampling=dict(maximum_spacing_mm=record["maximum_checkpoint_spacing_mm"]),
        )
    else:
        raise ValueError(f"Unsupported completed observation schema: {path}")
    result["evidence_path"] = path.as_posix()
    result["input_path"] = path.with_suffix(".inputs.json").as_posix()
    if "execution_identity" not in result:
        # Recover metadata only when the complete historical digest can be
        # reconstructed. Never assign today's solver to an old calculation.
        from temsim.instrument_snapshot import capture_instrument_snapshot, InstrumentSnapshot
        from temsim.optics.model import State
        snapshot_path = path.with_suffix(".snapshot.json")
        if snapshot_path.exists():
            saved = InstrumentSnapshot.from_dict(json.loads(snapshot_path.read_text(encoding="utf-8")))
            if saved.digest != record["snapshot_digest"]:
                raise ValueError("Saved snapshot does not belong to the completed observation")
            result["execution_identity"] = execution_identity(saved)
            result["execution_identity_evidence"] = "Complete saved snapshot matches observation digest"
            return result
        current = capture_instrument_snapshot(State.from_dict(inputs))
        historical = InstrumentSnapshot(current.graph, current.external_inputs,
                                         record.get("solver_identity", current.implementation))
        if historical.digest == record["snapshot_digest"]:
            result["execution_identity"] = execution_identity(historical)
            result["execution_identity_evidence"] = "Reconstructed complete recorded snapshot digest"
        else:
            result["execution_identity_evidence"] = "Historical identity cannot be reconstructed; not promoted"
    return result


def summarize(paths, output):
    records = sorted((read_observation(path) for path in paths), key=lambda row: row["rays"])
    if len(records) < 3 or len({row["rays"] for row in records}) != len(records):
        raise ValueError("At least three distinct complete particle counts are required")
    comparisons = [compare_counts(a, b) for a, b in zip(records, records[1:])]
    last_two_pass = all(row["passed"] for row in comparisons[-2:])
    summary = dict(
        schema="particle-count-convergence-summary-v1",
        status="TWO_SUCCESSIVE_REFINEMENTS_WITHIN_BUDGET" if last_two_pass else "NOT_CONVERGED",
        observations=records,
        comparisons=comparisons,
        scope=("Frozen installed working point, regenerated complete source at each N. "
               "Observation meshes and solver identities are retained in original receipts. "
               "This is not a retuned 30-mrad experiment or continuous crossover proof."),
        numerical_budgets=dict(alpha95_relative=.0025, rms_diameter_current_relative=.02,
                               local_focus_absolute_nm=.25),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows = ["# Complete-source particle-count observations", "", f"Status: **{summary['status']}**.", "",
            "All physical-input signatures agree after excluding only numerical particle counts and the column step. "
            "Every source is regenerated and retraced; no particle weights or optical parameters are fitted.", "",
            "| Particles | alpha95 (mrad) | RMS radius (nm) | D95 (nm) | Local radial focus offset (nm) | Current (microampere) | Seconds |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def formatted(value, digits=6, *, percent=False):
        if value is None:
            return "unavailable"
        return f"{value:.{digits}%}" if percent else f"{value:.{digits}f}"
    for record in records:
        s = record["specimen"]
        current = None if s['current_a'] is None else s['current_a']*1e6
        rows.append(f"| {record['rays']} | {formatted(s['alpha95_mrad'],9)} | {formatted(s['rms_radius_nm'],9)} | "
                    f"{formatted(s['diameter95_nm'],9)} | {formatted(s['local_focus_offset_nm'])} | "
                    f"{formatted(current,9)} | {formatted(record['elapsed_s'],2)} |")
    rows += ["", "| Refinement | alpha95 change | RMS change | D95 change | Focus change (nm) | Current change | Gate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for pair in comparisons:
        m = pair["metrics"]
        rows.append(f"| {pair['first_count']} to {pair['second_count']} | "
                    f"{formatted(m['alpha95_mrad'].get('relative_change'),4,percent=True)} | "
                    f"{formatted(m['rms_radius_nm'].get('relative_change'),4,percent=True)} | "
                    f"{formatted(m['diameter95_nm'].get('relative_change'),4,percent=True)} | "
                    f"{formatted(m['local_focus_offset_nm'].get('absolute_change_nm'))} | "
                    f"{formatted(m['current_a'].get('relative_change'),4,percent=True)} | "
                    f"{'PASS' if pair['passed'] else 'FAIL'} |")
    rows += ["", "Budgets are 0.25% for alpha95, 2% for RMS/D95/current, and 0.25 nm absolute local radial focus. "
             "The nonzero finer-count angle is the relative scale, not the unrelated 30-mrad pupil target. "
             "Two successive complete-count refinements must pass; one stable quantile does not certify the probe. "
             "A passing gate also requires matching solver and external-input identities. Cross-revision historical "
             "metrics remain informative but cannot independently establish strict same-implementation convergence.", "",
             "The local focus uses three executed upstream radii and a variance derivative. "
             "It is neither proof of two-direction focus nor an independently executed downstream minimum. "
             "The existing 193-particle column half-step observation is independent evidence of step sensitivity; "
             "it does not establish high-N step convergence. Earlier sources use the saved observation mesh, "
             "while the larger runs add fixed 0.05-mm waist windows. No exact trajectory identity across meshes is claimed.", "",
             "## Reused evidence", ""]
    rows.extend(f"- `{record['evidence_path']}`" for record in records)
    output.with_suffix(".md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    return summary


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = [row["rays"] for row in summary["observations"]]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key, title in zip(axes.flat,
            ("alpha95_mrad", "rms_radius_nm", "diameter95_nm", "local_focus_offset_nm"),
            ("Current-contained alpha95 (mrad)", "RMS radius (nm)",
             "95%-current diameter (nm)", "Local radial focus offset (nm)")):
        axis.plot(n, [row["specimen"][key] for row in summary["observations"]], "o-", color="#2269a4")
        axis.set_xscale("log", base=4)
        axis.set_xticks(n, [str(value) for value in n])
        axis.set_title(title)
        axis.set_xlabel("Complete emitted population")
        axis.grid(alpha=.25)
    fig.suptitle("Frozen Nanoprobe Normal: particle-count sensitivity\n"+summary["status"], fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.observations, args.output)
    plot_summary(summary, args.output.with_suffix(".png"))
    print(json.dumps(dict(status=summary["status"], counts=[r["rays"] for r in summary["observations"]])))


if __name__ == "__main__":
    main()
