"""Track the actual Mini Condenser / Objective two-dimensional crossover.

Continue an executed original-source packet through its own frozen field,
bracket the waist over the complete lens block and refine the physical
observation grid around that waist. All positions are real checkpoints.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from measure_crossover_cs import component_plane, trace_checkpoints
from measure_probe_waist import PHASE, _array_hash, _hash, _json, analyze_scan, observed_minimum
from temsim.cpu_resources import numerical_job
from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot, decode_instrument
from temsim.optics.waist_evidence import finite_interval
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
from temsim.physics.unified_field_particle import build_unified_plan
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.simulation_modes import is_unified
from temsim.specimen.source import specimen_is_vacuum


def _bounded_grid(lo, hi, spacing):
    return np.linspace(lo, hi, int(np.ceil((hi - lo) / spacing)) + 1)


def _local_grid(center, surface, half_width_nm, spacing_nm):
    # Use integer physical offsets so the same plane requested by two windows
    # has exactly one coordinate in the integration plan.
    offset_nm = (center - surface) * 1e6
    left = int(np.floor((offset_nm - half_width_nm) / spacing_nm))
    right = int(np.ceil((offset_nm + half_width_nm) / spacing_nm))
    return surface + np.arange(left, right + 1) * spacing_nm * 1e-6


def _continue(state, cp_z, phase, energies, times, alive, blocked, keys, nodes):
    if nodes[0] != cp_z or np.any(np.diff(nodes) <= 0):
        raise ValueError("Continuation requires the exact input checkpoint and ordered observation planes")
    plan = build_unified_plan(state, cp_z, nodes[-1], checkpoint_z_mm=nodes, maximum_step_mm=state.step_mm)
    return trace_checkpoints(
        state,
        plan,
        np.where(alive[:, None], phase, np.nan),
        energies,
        times,
        alive,
        np.where(alive, np.nan, blocked),
        np.where(alive, "", keys),
    )


def threefold_tail(x_m, y_m, active, weights, source_current):
    """Measure the third angular moment in the fixed-axis 0.1 to 1.0 nm annulus."""
    z = x_m + 1j * y_m
    radius = np.abs(z)
    live = active & (weights > 0)
    annulus = live & (radius > 0.1e-9) & (radius <= 1.0e-9)
    currents = source_current * weights / weights.sum()
    annular_current = float(currents[annulus].sum())
    third = np.sum(currents[annulus] * (z[annulus] / radius[annulus])**3)
    shell_edges_nm = np.array([0.1, 0.25, 0.5, 0.75, 1.0])
    radii = np.searchsorted(shell_edges_nm[1:]*1e-9, radius[annulus], side="left")
    moment = currents[annulus] * (z[annulus] / radius[annulus])**3
    shell_moments = [float(abs(moment[radii == index].sum())) for index in range(len(shell_edges_nm)-1)]
    return dict(annulus_nm=[0.1, 1.0], annular_current_a=annular_current,
                harmonic_current_a=float(abs(third)),
                normalized_m3=float(abs(third) / annular_current) if annular_current else None,
                shell_edges_nm=shell_edges_nm.tolist(), shell_harmonic_currents_a=shell_moments,
                shell_harmonic_sum_a=sum(shell_moments),
                outside_1nm_current_a=float(currents[live & (radius > 1.0e-9)].sum()))


def _native_profile(checkpoints, active, weights, ids, planes, reference, surface, source_current):
    arrays = {key: getattr(checkpoints, key) for key in PHASE}
    arrays.update(alive=active, weights=weights, source_id=ids, z_mm=checkpoints.z_mm)
    stats = analyze_scan(arrays, planes, surface_z_mm=reference, source_current_a=source_current)
    rows = []
    total_weight = weights.sum()
    for raw in stats["planes"]:
        i = np.searchsorted(checkpoints.z_mm, raw["z_mm"])
        distance = np.hypot(checkpoints.x_m[i], checkpoints.y_m[i])
        core = active[i] & (weights > 0) & (distance <= 50e-12)
        row = {key: value for key, value in raw.items() if key not in ("source_ids", "offset_nm")}
        row.update(
            offset_from_surface_nm=(raw["z_mm"] - surface) * 1e6,
            core_50pm_a=float(source_current * weights[core].sum() / total_weight),
        )
        if raw["z_mm"] in (reference, surface):
            row["threefold_tail"] = threefold_tail(checkpoints.x_m[i], checkpoints.y_m[i], active[i],
                                                  weights, source_current)
        rows.append(row)
    return stats, rows


def measure(folder, *, spacing_nm=0.1):
    if not np.isfinite(spacing_nm) or not 0.05 <= spacing_nm <= 1.0:
        raise ValueError("Local observation spacing must be between 0.05 and 1 nm")
    receipt = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    original = InstrumentSnapshot.from_dict(
        json.loads((folder / "snapshot.json").read_text(encoding="utf-8"))
    )
    state = decode_instrument(original.graph)
    original._verify_external_inputs(state)
    current_input = capture_instrument_snapshot(state)
    if (
        current_input.digest != original.digest
        or original.digest != receipt["input_digest"]
        or receipt["status"] != "COMPLETE"
    ):
        raise ValueError("The saved full-source execution does not match its optical inputs")
    runtime = get_unified_field_runtime(state)
    audit = receipt["field_audit"]
    if (
        not is_unified(state)
        or not specimen_is_vacuum(state.sample)
        or isinstance(runtime.snapshot, DynamicPotentialSnapshot)
        or not audit["passed"]
        or runtime.snapshot.numerical_digest != audit["identities"]["potential_digest"]
        or runtime.geometry.digest != audit["identities"]["geometry_digest"]
    ):
        raise ValueError(
            "A completed static unified vacuum trace with matching physical fields and stops is required"
        )
    mini = next((lens for lens in state.lenses if lens.key == "mini_condenser" and lens.enabled), None)
    objective = state.objective_lens
    if mini is None or not objective.enabled:
        raise ValueError("The combined-field measurement requires an enabled Mini Condenser and Objective")
    lo = component_plane(state, mini)
    surface = float(state.sample.upper_surface_z_mm)
    # A vacuum extension past both objective field centres explicitly observes
    # the divergent side without treating the specimen as an endpoint.
    objective_centres = (
        component_plane(state, objective, z - objective.z_mm)
        for z in (objective.upper_field_center_z_mm, objective.lower_field_center_z_mm)
    )
    hi = max(surface, *objective_centres) + 2.0
    with np.load(folder / "arrays.npz", allow_pickle=False) as original_arrays:
        weights, ids = (original_arrays[k] for k in ("weights", "source_id"))
        if (
            np.count_nonzero(weights > 0) < 1000
            or np.count_nonzero(weights > 0) != receipt["emitted_positive_current_points"]
            or not {"source_id", "weights"}.issubset(receipt["source_sha256"])
            or len(np.unique(ids)) != len(ids)
            or any(
                _array_hash(original_arrays[key]) != digest
                for key, digest in receipt["source_sha256"].items()
            )
        ):
            raise ValueError("Missing or insufficient original source lineage")
        z = original_arrays["z_mm"]
        before_mini = np.flatnonzero(z <= lo)
        if not len(before_mini):
            raise ValueError("No executed checkpoint at or before the Mini Condenser")
        start = before_mini[-1]
        old_surface = np.flatnonzero(z == surface).item()
        old_positions = np.column_stack([original_arrays[k][old_surface] for k in ("x_m", "y_m")])
        old_alive = original_arrays["alive"][old_surface]
        phase = np.column_stack([original_arrays[k][start] for k in ("x_m", "y_m", "tx_rad", "ty_rad")])
        nodes = np.unique(np.r_[z[start], _bounded_grid(lo, hi, 0.5), lo, surface, hi])
        cp, live, stops, labels = _continue(
            state,
            z[start],
            phase,
            original_arrays["kinetic_energy_ev"][start],
            original_arrays["flight_time_s"][start],
            original_arrays["alive"][start],
            original_arrays["blocked_z_mm"],
            original_arrays["blocked_key"],
            nodes,
        )
    i = np.flatnonzero(cp.z_mm == surface).item()
    if not np.array_equal(live[i], old_alive):
        raise ValueError("Re-observing the original column changed surface membership")
    difference = np.column_stack((cp.x_m[i], cp.y_m[i]))[old_alive] - old_positions[old_alive]
    coarse = finite_interval(cp, live, weights, lo, hi, maximum_spacing_mm=0.5, source_ids=ids)
    specimen = None
    if np.any(live[i] & (weights > 0)):
        _, surface_rows = _native_profile(
            cp, live, weights, ids, [surface], surface, surface, receipt["source_current_a"]
        )
        specimen = surface_rows[0]
    report = dict(
        schema="unified-mini-objective-crossover-v1",
        input_digest=original.digest,
        executed_field_digest=runtime.snapshot.numerical_digest,
        source_identity=receipt["emitted_identity_digest"],
        original_positive_count=int(np.count_nonzero(weights > 0)),
        source_current_a=receipt["source_current_a"],
        surface_z_mm=surface,
        search_interval_mm=[lo, hi],
        at_specimen=specimen,
        coarse_region=coarse,
        refinement=[],
        status="INCONCLUSIVE",
        surface_repeat_max_position_change_nm=(
            float(np.max(np.linalg.norm(difference, axis=1)) * 1e9) if difference.size else None
        ),
        particle_file_sha256=_hash(folder / "arrays.npz"),
        original_report=str((folder / "report.json").resolve()),
    )
    if len(coarse["regions"]) != 1 or coarse["unresolved"]:
        report["reason"] = "The combined field interval did not isolate one bounded 2D crossover"
        return report
    if specimen is None:
        report["reason"] = "No positive current reached the physical specimen entrance"
        return report
    candidate = coarse["regions"][0]
    left, right = np.searchsorted(cp.z_mm, candidate["radial_bracket_mm"])
    left = max(0, left - 2)
    right = min(len(cp.z_mm) - 1, right + 2)
    a, b = float(cp.z_mm[left]), float(cp.z_mm[right])
    seeds = [candidate["radial_root_interpolated_mm"]]
    local_nodes = _bounded_grid(a, b, 0.01)
    source_phase = np.column_stack([getattr(cp, k)[left] for k in ("x_m", "y_m", "tx_rad", "ty_rad")])
    for round_number in (1, 2):
        extra = []
        for center in seeds:
            if round_number == 1:
                extra.extend(
                    (_local_grid(center, surface, 10000.0, 100.0), _local_grid(center, surface, 700.0, 2.0))
                )
            else:
                extra.append(_local_grid(center, surface, 30.0, spacing_nm))
        local_nodes = np.unique(np.concatenate([local_nodes, np.asarray([a, b, surface]), *extra]))
        local_nodes = local_nodes[(local_nodes >= a) & (local_nodes <= b)]
        observed, active, _, _ = _continue(
            state,
            a,
            source_phase,
            cp.kinetic_energy_ev[left],
            cp.flight_time_s[left],
            live[left],
            stops,
            np.asarray(labels),
            local_nodes,
        )
        region = finite_interval(
            observed, active, weights, a, b, maximum_spacing_mm=0.010000001, source_ids=ids
        )
        if len(region["regions"]) != 1 or region["unresolved"]:
            report["refinement"].append(region)
            report["reason"] = "The refinement did not resolve a single through-going two-dimensional waist"
            return report
        waist = region["regions"][0]
        waist_z = waist["representative_z_mm"]
        stats, rows = _native_profile(
            observed, active, weights, ids, local_nodes, waist_z, surface, receipt["source_current_a"]
        )
        peak = max(rows, key=lambda row: row["core_50pm_a"])
        d95 = min(rows, key=lambda row: row["d95_nm"])
        seeds = [waist_z, peak["z_mm"], d95["z_mm"]]
        report["refinement"].append(
            dict(
                round=round_number,
                waist_z_mm=waist_z,
                radial_bracket_mm=waist["radial_bracket_mm"],
                finite_2d_bracket_mm=waist["bracket_mm"],
                observed_planes=len(local_nodes),
            )
        )
    report.update(
        status="MEASURED",
        waist=waist,
        at_crossover=next(row for row in rows if row["z_mm"] == waist_z),
        at_specimen=next((row for row in rows if row["z_mm"] == surface), specimen),
        source_count=int(len(weights)),
        d95_best=d95,
        rms_best=min(rows, key=lambda row: row["rms_radius_nm"]),
        core_best=peak,
        core_peak_planes=observed_minimum(local_nodes, [-row["core_50pm_a"] for row in rows], surface)[
            "observed_minimum_planes_mm"
        ],
        separation=stats["waist_separation"],
        membership_stable=stats["positive_current_membership_stable"],
        target_in_2d_bracket=waist["bracket_mm"][0] <= surface <= waist["bracket_mm"][1],
        profile=rows,
    )
    if capture_instrument_snapshot(state).digest != current_input.digest:
        raise ValueError("Observation changed an optical or source input")
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--spacing-nm", type=float, default=0.1)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Choose a new report file")
    with numerical_job(3):
        result = measure(args.trial, spacing_nm=args.spacing_nm)
    _json(args.report, result)
    print(result["status"], result.get("at_crossover", {}), flush=True)


if __name__ == "__main__":
    main()
