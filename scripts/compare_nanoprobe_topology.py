"""Audit executed vacuum-column crossovers and compare their optical assignments."""
from argparse import ArgumentParser
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from temsim.optics.beam_path_audit import (
    crossover_intervals,
    gun_planes_coordinates,
    optical_component_planes,
)
from temsim.optics.waist_evidence import finite_interval, observed_aperture_losses
from temsim.simulation_modes import is_unified
from temsim.specimen.source import specimen_is_vacuum


SCHEMA = "nanoprobe-crossover-topology-v1"
ROUND_LENS_KINDS = {
    "electrostatic_lens", "extractor_electrode", "magnetic_lens", "round_lens",
}
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad")


def _axis(start, stop, spacing):
    count = max(1, math.ceil((stop - start) / spacing))
    return np.linspace(start, stop, count + 1)


def _lens_planes(state):
    enabled = {lens.key for lens in state.lenses if lens.enabled}
    centres = [
        (item.key, float(item.optical_reference_plane_z_mm))
        for item in state._resolved_optics_layout
        if item.kind in ROUND_LENS_KINDS
        and item.optical_reference_plane_z_mm is not None
        and (item.key in enabled or item.kind in {"electrostatic_lens", "extractor_electrode"})
    ]
    return sorted(centres, key=lambda item: (item[1], item[0]))


def _components(state):
    rows = optical_component_planes(state, full_path=True)
    rows.append(("specimen_entrance", float(state.sample.upper_surface_z_mm)))
    rows.append(("specimen_exit", float(state.sample.lower_surface_z_mm)))
    if state.ac_deflector.enabled:
        rows.append(("ac_upper", float(state.ac_deflector.upper_z_mm)))
        rows.append(("ac_lower", float(state.ac_deflector.lower_z_mm)))
    return sorted(rows, key=lambda item: (item[1], item[0]))


def _receiver_end(state):
    planes = [z for _, z in _components(state)]
    return max(planes)


def _assignment_nodes(state):
    lens_z = np.unique([z for _, z in _lens_planes(state)])
    midpoints = (lens_z[:-1] + lens_z[1:]) / 2
    return np.r_[[z for _, z in _components(state)], lens_z, midpoints]


def observation_planes(state, additional=(), *, spacing_mm=1.):
    """Keep user focus nodes and include exact lens and receiver references."""
    if not is_unified(state) or not specimen_is_vacuum(state.sample):
        raise ValueError("The whole-column reference requires a unified vacuum instrument")
    if not math.isfinite(spacing_mm) or not 0 < spacing_mm <= 1.:
        raise ValueError("Whole-column spacing must be positive and at most 1 mm")
    start, stop = float(state.electron_gun.exit_plane_z_mm), _receiver_end(state)
    additional = np.asarray(additional, dtype=float)
    if (additional.ndim != 1 or not np.isfinite(additional).all()
            or np.any((additional < start) | (additional > stop))):
        raise ValueError("Additional observations must lie inside the gun-exit to receiver path")
    special = _assignment_nodes(state)
    surface = float(state.sample.upper_surface_z_mm)
    local_focus = surface + _axis(-20., 20., .125) * 1e-6
    ac = state.ac_deflector
    coil = _axis(float(ac.upper_z_mm), float(ac.lower_z_mm), .1) if ac.enabled else []
    return np.unique(np.r_[_axis(start, stop, spacing_mm),
                           special[(start <= special) & (special <= stop)],
                           local_focus, coil, additional])


def _nearest_lens(z_mm, lenses):
    distances = np.array([abs(z_mm - z) for _, z in lenses])
    error = 64 * np.finfo(float).eps * max(1., abs(z_mm))
    nearest = np.flatnonzero(distances <= distances.min() + error)
    planes = [lenses[index][1] for index in nearest]
    delta = z_mm - planes[0]
    side = "on" if abs(delta) <= error else "upstream" if delta < 0 else "downstream"
    return {
        "lens_keys": [lenses[index][0] for index in nearest],
        "side": side if len(set(planes)) == 1 else "equidistant",
        "offset_mm": float(delta) if len(set(planes)) == 1 else None,
    }


def _assign_region(row, components, lenses, surface):
    z_mm = row["radial_root_interpolated_mm"]
    left, right = row["radial_bracket_mm"]
    nearest = _nearest_lens(z_mm, lenses)
    lens_z = np.unique([z for _, z in lenses])
    boundaries = np.r_[lens_z, (lens_z[:-1] + lens_z[1:]) / 2]
    bracket_assignment_stable = (
        nearest["side"] not in {"on", "equidistant"}
        and not np.any((left < boundaries) & (boundaries < right))
    )
    component_interval, = crossover_intervals([z_mm], components)
    return dict(row, nearest_lens=nearest, nearest_lens_bracket_resolved=bracket_assignment_stable,
                component_interval=component_interval,
                offset_from_specimen_entrance_nm=float((z_mm - surface) * 1e6))


def _population(weights, ids, selected):
    weight = weights[selected]
    return {
        "positive": int(selected.sum()),
        "effective": float(weight.sum() ** 2 / (weight @ weight)) if weight.size else 0.,
        "original_weight_fraction": float(weight.sum() / weights.sum()),
        "source_id_sha256": hashlib.sha256(np.ascontiguousarray(ids[selected], dtype="<i8")).hexdigest(),
    }


def _loss_evidence(state, cp, live, weights, ids, stop_z, stop_key):
    """Record transitions owned by the executed mechanics or receiver clipping."""
    keys = np.asarray(stop_key, str)
    poses = {row.key for row in state._resolved_optics_layout}
    receiver_z = {item.key: float(item.z_mm)
                  for item in state.recording_planes if item.inserted}
    rows = {}
    if hasattr(state, "apertures"):
        rows = {tuple(row["bracket_mm"]): row for row in
                observed_aperture_losses(state, cp, live, weights, ids)}
    z = cp.z_mm
    results = []
    for j in np.flatnonzero(np.any(live[:-1] != live[1:], axis=1)):
        lost = live[j] & ~live[j + 1]
        reappeared = live[j + 1] & ~live[j]
        bracket = float(z[j]), float(z[j + 1])
        tolerance = 64 * np.finfo(float).eps * max(1., abs(bracket[1]))
        supported_stop = (
            np.isfinite(stop_z[lost]).all()
            and np.all(stop_z[lost] >= bracket[0] - tolerance)
            and np.all(stop_z[lost] <= bracket[1] + tolerance)
        )
        labels = set(keys[lost])
        exact_receivers = {
            name for name in labels if name in receiver_z
            and np.all(np.abs(stop_z[lost & (keys == name)] - receiver_z[name]) <= tolerance)
        }
        prior = rows.get(bracket, {})
        if lost.any() and not reappeared.any() and supported_stop and labels <= poses | exact_receivers:
            status = "EXECUTED_FIRST_CONTACT"
        else:
            status = "UNRESOLVED_MEMBERSHIP_CHANGE"
        results.append({
            "bracket_mm": list(bracket),
            "status": status,
            "stop_counts": {name: int(np.count_nonzero(lost & (keys == name))) for name in sorted(labels)},
            "lost": _population(weights, ids, lost),
            "reappeared": int(np.count_nonzero(reappeared)),
            "opening_evidence": prior.get("status"),
        })
    return results


def _audit_cohort(cp, live, weights, ids, components, lenses, surface, losses):
    """Measure finite regions inside stable-current spans separated by stops."""
    z = np.asarray(cp.z_mm)
    cuts = np.r_[0, np.flatnonzero(np.any(live[:-1] != live[1:], axis=1)) + 1, len(z)]
    regions, unresolved, supports = [], [], []
    for first, stop in zip(cuts[:-1], cuts[1:]):
        population = _population(weights, ids, live[first])
        bounds = [float(z[first]), float(z[stop - 1])]
        supports.append(dict(population, bounds_mm=bounds, checkpoints=int(stop - first)))
        if not population["positive"]:
            continue
        if population["positive"] < 5 or stop - first < 3:
            unresolved.append({"bounds_mm": bounds, "reason": "insufficient_stable_cohort_support"})
            continue
        segment = slice(first, stop)
        view = SimpleNamespace(z_mm=z[segment], **{key: getattr(cp, key)[segment] for key in PHASE})
        report = finite_interval(view, live[segment], weights, *bounds,
                                 maximum_spacing_mm=float(np.max(np.diff(view.z_mm))), source_ids=ids)
        regions.extend(_assign_region(row, components, lenses, surface) for row in report["regions"])
        unresolved.extend(report["unresolved"])
    unresolved.extend(row for row in losses if row["status"] != "EXECUTED_FIRST_CONTACT")
    unresolved.extend({"reason": "nearest_lens_relation_not_resolved", "region": row}
                      for row in regions if not row["nearest_lens_bracket_resolved"])
    return {
        "status": "UNRESOLVED" if unresolved else "RESOLVED_AT_BOUNDED_SAMPLING",
        "regions": regions, "unresolved": unresolved, "stable_cohorts": supports, "stops": losses,
    }


def _validate_population(cp, live, weights, ids, stop_z, stop_key):
    z, live, weights, ids, stop_z = map(np.asarray, (cp.z_mm, live, weights, ids, stop_z))
    if (z.ndim != 1 or len(z) < 3 or not np.isfinite(z).all() or np.any(np.diff(z) <= 0)
            or weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0)
            or np.count_nonzero(weights > 0) < 1000 or not np.isfinite(weights.sum())
            or ids.shape != weights.shape or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(weights) or stop_z.shape != weights.shape
            or len(stop_key) != len(weights) or live.dtype != bool
            or live.shape != (len(z), len(weights))
            or any(np.shape(getattr(cp, key)) != live.shape for key in PHASE)):
        raise ValueError("Valid executed checkpoints and at least 1000 original current-carrying IDs are required")
    return z, live & (weights > 0)[None, :], weights, ids, stop_z


def make_column_report(state, cp, live, weights, ids, stop_z, stop_key):
    """Summarize actual and constant-membership waists without saving ray histories."""
    z, live, weights, ids, stop_z = _validate_population(cp, live, weights, ids, stop_z, stop_key)
    start, stop = float(state.electron_gun.exit_plane_z_mm), _receiver_end(state)
    if z[0] != start or z[-1] != stop or np.max(np.diff(z)) > 1. + 1e-10:
        raise ValueError("Whole-column audit requires gun exit to receiver at at most 1 mm spacing")
    surface = float(state.sample.upper_surface_z_mm)
    surface_index = np.searchsorted(z, surface)
    if surface_index >= len(z) or z[surface_index] != surface:
        raise ValueError("The specimen entrance must be an exact executed checkpoint")
    surviving_planes = np.flatnonzero(live.any(axis=1))
    if not len(surviving_planes) or surviving_planes[-1] <= surface_index:
        raise ValueError("No positive current continues past the specimen entrance")
    before_receiver = int(surviving_planes[-1])
    components, lenses = _components(state), _lens_planes(state)
    loss = _loss_evidence(state, cp, live, weights, ids, stop_z, stop_key)
    full = _audit_cohort(cp, live, weights, ids, components, lenses, surface, loss)
    common = {}
    incident_selection = live[surface_index]
    local_exit = np.searchsorted(z, surface + 20e-6, side="right")
    first_loss = np.flatnonzero(np.any(incident_selection & ~live[surface_index:local_exit], axis=1))
    incident_stop = surface_index + int(first_loss[0]) if len(first_loss) else local_exit
    for name, part, selected in (
        ("incident", slice(0, incident_stop), incident_selection),
        ("downstream", slice(surface_index, before_receiver + 1), live[before_receiver]),
    ):
        phase = SimpleNamespace(z_mm=z[part], **{key: getattr(cp, key)[part] for key in PHASE})
        stable = np.broadcast_to(selected, (len(phase.z_mm), len(weights)))
        if not np.all(~stable | live[part]):
            raise ValueError("A current-carrying ID reappeared in the common cohort")
        common[name] = _audit_cohort(phase, stable, weights, ids, components, lenses, surface, ())
    return {
        "schema": SCHEMA, "scope": "Executed unified vacuum optical path; separate gun evidence",
        "z_bounds_mm": [start, stop], "maximum_gap_mm": float(np.max(np.diff(z))),
        "specimen_entrance_z_mm": surface, "round_lenses": lenses, "component_planes": components,
        "emitted": _population(weights, ids, weights > 0),
        "specimen_survivors": _population(weights, ids, live[surface_index]),
        "last_survivors": _population(weights, ids, live[before_receiver]),
        "last_surviving_z_mm": float(z[before_receiver]),
        "cohorts": {"actual": full, **common},
    }


def make_gun_report(state, trace):
    """Summarize the recorded first-forward-crossing gun history."""
    stop = float(state.electron_gun.exit_plane_z_mm)
    start = max(0., float(trace.z_mm[0]))
    sampling = [
        (1e-3, 1e-5), (1e-2, 5e-5), (.05, .001), (2., .05),
        (30., .25), (stop, 1.),
    ]
    special = _assignment_nodes(state)
    grids = []
    lower = start
    for upper, gap in sampling:
        upper = min(upper, stop)
        if upper > lower:
            grids.append(_axis(lower, upper, gap))
            lower = upper
    z = np.unique(np.r_[*grids, special[(start <= special) & (special <= stop)]])
    phase, live = gun_planes_coordinates(trace, z)
    cp = SimpleNamespace(z_mm=z, **dict(zip(PHASE, phase.transpose(1, 0, 2))))
    exit_bundle = trace.exit_bundle
    z, live, weights, ids, stop_z = _validate_population(
        cp, live, exit_bundle.weight, exit_bundle.ray_id, trace.blocked_z_mm, trace.blocked_key
    )
    components, lenses = _components(state), _lens_planes(state)
    surface = float(state.sample.upper_surface_z_mm)
    loss = _loss_evidence(state, cp, live, weights, ids, stop_z, trace.blocked_key)
    actual = _audit_cohort(cp, live, weights, ids, components, lenses, surface, loss)
    stable = np.broadcast_to(live[-1], live.shape)
    common = _audit_cohort(cp, stable, weights, ids, components, lenses, surface, ())
    return {
        "schema": SCHEMA, "scope": "Recorded first-forward-crossing gun history with axial interpolation",
        "z_bounds_mm": z[[0, -1]].tolist(), "maximum_gap_mm": float(np.max(np.diff(z))),
        "requested_sampling_mm": [[upper, gap] for upper, gap in sampling if upper <= stop],
        "round_lenses": lenses, "component_planes": components,
        "cohorts": {"actual": actual, "gun_exit": common},
    }


def compare_reports(reference, candidate):
    """Require the same ordered multiplicity, component interval and nearest side."""
    if any(report.get("schema") != SCHEMA for report in (reference, candidate)):
        raise ValueError("Unknown crossover evidence schema")

    def signatures(rows):
        return [{
            "component_interval": [row["component_interval"][0],
                                   *[list(keys) for keys in row["component_interval"][1:]]],
            "lens_keys": row["nearest_lens"]["lens_keys"],
            "side": row["nearest_lens"]["side"],
        } for row in rows]

    geometry = all(
        [tuple(row) for row in reference[key]] == [tuple(row) for row in candidate[key]]
        for key in ("round_lenses", "component_planes")
    ) and tuple(reference["z_bounds_mm"]) == tuple(candidate["z_bounds_mm"])
    results = {}
    for name in sorted(reference["cohorts"].keys() | candidate["cohorts"].keys()):
        a, b = reference["cohorts"].get(name), candidate["cohorts"].get(name)
        if a is None or b is None:
            results[name] = {"status": "UNRESOLVED", "reason": "missing_cohort"}
            continue
        sig_a, sig_b = signatures(a["regions"]), signatures(b["regions"])
        matched = sig_a == sig_b
        if a["status"] != "RESOLVED_AT_BOUNDED_SAMPLING" or b["status"] != "RESOLVED_AT_BOUNDED_SAMPLING":
            status = "UNRESOLVED"
        elif not matched:
            status = "CHANGED"
        else:
            status = "PRESERVED_AT_BOUNDED_SAMPLING"
        results[name] = {"status": status, "same_resolved_relations": matched,
                         "reference": sig_a, "candidate": sig_b,
                         "reference_unresolved": a["unresolved"], "candidate_unresolved": b["unresolved"]}
    statuses = {row["status"] for row in results.values()}
    if not geometry or "CHANGED" in statuses:
        status = "CHANGED"
    elif not results or "UNRESOLVED" in statuses:
        status = "UNRESOLVED"
    else:
        status = "PRESERVED_AT_BOUNDED_SAMPLING"
    return {
        "schema": SCHEMA, "status": status, "same_reference_geometry": geometry,
        "same_resolved_relations": bool(results) and geometry and all(
            row.get("same_resolved_relations", False) for row in results.values()
        ),
        "cohorts": results,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    reference, candidate = [json.loads(path.read_text(encoding="utf-8"))
                            for path in (args.reference, args.candidate)]
    result = compare_reports(reference, candidate)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"])
    return 0 if result["status"] == "PRESERVED_AT_BOUNDED_SAMPLING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
