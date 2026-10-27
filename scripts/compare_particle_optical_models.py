"""Compare frozen particle column models with one verified gun-exit population.

No working point is fitted or installed. Historical-control cases replay only
the recorded incident lens/hexapole controls on CURRENT geometry, source and
implementation. They are not a replay of the complete historical instrument.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from temsim.calculation_manifest import solver_source_identity
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.crossover_acceptance import evaluate_crossover_chain, regions_for_state
from temsim.optics.model import State
from temsim.physics.backend_execution import backend_receipts
from temsim.physics.beam_current import effective_source_current_a
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.particle_sections import _digest_arrays, _events, _trace_segment


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "outputs/optical-acceptance-contract-20261027"
INCIDENT_KEYS = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3", "adapter_lens",
                 "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens", "mini_condenser", "objective_lens")
HEX_KEYS = ("probe_hp2_hexapole", "probe_hp1_hexapole")
CASES = ("current_custom", "current_unified", "historical_controls_custom", "historical_controls_unified")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def differences(first, second, path=""):
    if isinstance(first, dict) and isinstance(second, dict):
        return [item for key in sorted(set(first) | set(second))
                for item in differences(first.get(key), second.get(key), f"{path}/{key}")]
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second):
        return [item for index, (a, b) in enumerate(zip(first, second))
                for item in differences(a, b, f"{path}/{index}")]
    return [] if first == second else [dict(path=path, before=first, after=second)]


def file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_crossover_chains(reference, candidate, *, maximum_size_ratio=10.):
    """Compare measured topology without changing the waist detector or its limits.

    RMS size is the requested amplification gate. D95 is independently reported.
    Common-cohort identities must match; real losses and unresolved candidates
    remain explicit and cannot be hidden by pairing only accepted minima.
    """
    factor = float(maximum_size_ratio)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("The crossover size ratio must be finite and positive")
    first, last = reference["executed_waists"], candidate["executed_waists"]
    old_sequence = [row["region"] for row in first]
    new_sequence = [row["region"] for row in last]
    same_cohort = reference["common_cohort"] == candidate["common_cohort"]
    same_intervals = ({key: row["bounds_mm"] for key, row in reference["intervals"].items()}
                      == {key: row["bounds_mm"] for key, row in candidate["intervals"].items()})
    topology = old_sequence == new_sequence and same_intervals
    count_differences = []
    for region in sorted(set(old_sequence) | set(new_sequence)):
        old_count, new_count = old_sequence.count(region), new_sequence.count(region)
        if old_count == new_count:
            continue
        lower = reference if old_count < new_count else candidate
        interval = lower["intervals"].get(region)
        unknown = bool(interval is not None and (
            interval["status"] == "UNRESOLVED_COVERAGE" or any(
                interval["bounds_mm"][0] < row["z_mm"] < interval["bounds_mm"][1]
                for row in lower["unresolved_candidates"])))
        count_differences.append(dict(region=region, reference=old_count, candidate=new_count,
            status="UNRESOLVED" if unknown else "CONFIRMED_COUNT_MISMATCH_AT_SAMPLING"))
    def sizes(old, new):
        a, b = old.get("statistics", {}), new.get("statistics", {})
        rms_a, rms_b = a.get("radius_rms_m"), b.get("radius_rms_m")
        q_a, q_b = a.get("radius_95_m"), b.get("radius_95_m")
        def measurement(left, right, scale):
            valid = (left is not None and right is not None
                     and np.isfinite([left, right]).all() and left >= 0 and right >= 0)
            return dict(reference=None if left is None else left*scale,
                candidate=None if right is None else right*scale,
                ratio=right/left if valid and left > 0 else None,
                within_limit=bool(valid and right <= factor*left))
        return dict(rms_radius_nm=measurement(rms_a, rms_b, 1e9),
                    diameter95_nm=measurement(q_a, q_b, 2e9))
    matches = []
    # Positional pairing is valid only when the complete ordered region list
    # agrees. Missing/extra waists must not shift later pairs to hide a failure.
    if topology and same_cohort:
        for index, (a, b) in enumerate(zip(first, last)):
            matches.append(dict(index=index, region=a["region"], reference_z_mm=a["z_mm"],
                candidate_z_mm=b["z_mm"], axial_shift_mm=b["z_mm"]-a["z_mm"],
                reference_bracket_mm=a["bracket_mm"], candidate_bracket_mm=b["bracket_mm"],
                reference_common_2d_minimum=a["common_2d_waist_at_same_bracket"],
                candidate_common_2d_minimum=b["common_2d_waist_at_same_bracket"],
                common=sizes(a["common"], b["common"]),
                actual=sizes(a["actual"], b["actual"]),
                same_actual_source_ids=a["actual_source_ids"] == b["actual_source_ids"]))
    endpoint = sizes(reference["specimen"]["common"], candidate["specimen"]["common"])
    size_pass = bool(topology and same_cohort
        and all(row["common"]["rms_radius_nm"]["within_limit"] for row in matches)
        and endpoint["rms_radius_nm"]["within_limit"])
    unresolved = dict(reference=reference["unresolved_candidates"],
                      candidate=candidate["unresolved_candidates"])
    complete = all(row["coverage"]["complete_incident_interval"] and row["coverage"]["spacing_satisfied"]
                   for row in (reference, candidate))
    # Interception-transition brackets and candidates lacking a two-direction
    # minimum keep the qualification incomplete even if accepted counts match.
    resolved = (topology and same_cohort and size_pass and complete and not any(unresolved.values())
                and all(row["status"] == "OBSERVED_AT_BOUNDED_SAMPLING" for row in (reference, candidate)))
    size_failure = (not endpoint["rms_radius_nm"]["within_limit"] or any(
        not row["common"]["rms_radius_nm"]["within_limit"] for row in matches))
    confirmed_topology_failure = (not same_intervals or any(
        row["status"] != "UNRESOLVED" for row in count_differences)
        or (not topology and not count_differences))
    failed = confirmed_topology_failure or not same_cohort or size_failure
    return dict(status="FAIL" if failed else "PASS_AT_BOUNDED_SAMPLING" if resolved else "UNRESOLVED",
        maximum_rms_size_ratio=factor, reference_waist_count=len(first), candidate_waist_count=len(last),
        reference_region_sequence=old_sequence, candidate_region_sequence=new_sequence,
        observed_count_differences=count_differences,
        same_optical_interval_boundaries=same_intervals, ordered_topology_matches=topology,
        same_common_cohort=same_cohort, rms_size_limit_satisfied=size_pass,
        matched_waists=matches, specimen_endpoint=endpoint, unresolved_candidates=unresolved,
        accepted_waists=dict(reference=[dict(z_mm=row["z_mm"], region=row["region"],
            common=row["common"], actual=row["actual"]) for row in first],
            candidate=[dict(z_mm=row["z_mm"], region=row["region"],
            common=row["common"], actual=row["actual"]) for row in last]),
        complete_bounded_sampling=complete, fully_resolved_at_sampling=resolved,
        size_definition="Current-weighted RMS radius on the unchanged-ID common cohort; D95 reported separately",
        scope="Observed incident-column topology, not all possible continuous roots, diffraction edge, or optical qualification")


def without_added_aberrations(inputs):
    """Private legacy diagnostic; native field strengths and geometry stay fixed."""
    requested = deepcopy(inputs)
    for lens in requested["lenses"]:
        lens["cs_mm"] = 0.
    for key in ("probe_aberrations", "image_aberrations", "lens_intrinsic_aberrations"):
        requested[key] = {}
    requested["chromatic_aberration_enabled"] = False
    return requested


def compare_saved_pair(reference_stem, candidate_stem, *, maximum_size_ratio=10., maximum_spacing_mm=.5,
                       legacy_no_added_diagnostic=False):
    """Read saved states only; never regenerate or propagate an electron source."""
    from types import SimpleNamespace
    captures = []
    for prefix in (Path(reference_stem), Path(candidate_stem)):
        inputs = json.loads(prefix.with_suffix(".inputs.json").read_text(encoding="utf-8"))
        state = State.from_dict(inputs)
        if state.to_dict() != inputs:
            raise ValueError("Stored comparison inputs changed during reconstruction")
        with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        identity = _digest_arrays(*(arrays[key][0] for key in
            ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s")), arrays["source_energy_offset_ev"],
            arrays["weight"], arrays["alive"][0], arrays["source_id"])
        cp = SimpleNamespace(**{key: arrays[key] for key in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad")})
        chain = evaluate_crossover_chain(cp, arrays["alive"], arrays["weight"], arrays["source_id"],
            source_current_a=effective_source_current_a(state), regions=regions_for_state(state),
            maximum_spacing_mm=maximum_spacing_mm, source_identity=identity)
        captures.append(dict(inputs=inputs, arrays=arrays, chain=chain, identity=identity,
            array_sha256=file_digest(prefix.with_suffix(".npz"))))
    a, b = captures
    if a["identity"] != b["identity"]:
        raise ValueError("A model comparison requires identical executed gun-exit states, IDs and weights")
    if not np.array_equal(a["arrays"]["z_mm"], b["arrays"]["z_mm"]):
        raise ValueError("A model comparison requires identical exact observation planes")
    result = compare_crossover_chains(a["chain"], b["chain"], maximum_size_ratio=maximum_size_ratio)
    changes = differences(a["inputs"], b["inputs"])
    other_changes = [row for row in changes if row["path"] not in ("/simulation_mode", "/active_backend")]
    permitted_diagnostic = False
    if legacy_no_added_diagnostic:
        expected = without_added_aberrations(b["inputs"])
        permitted_diagnostic = (a["inputs"]["simulation_mode"] == "custom"
            and b["inputs"]["simulation_mode"] == "unified" and not any(
                row["path"] not in ("/simulation_mode", "/active_backend")
                for row in differences(expected, a["inputs"])))
    survival_equal = np.array_equal(a["arrays"]["alive"], b["arrays"]["alive"])
    if not survival_equal or (other_changes and not permitted_diagnostic):
        result["status"] = "FAIL"
    result.update(reference_arrays=str(Path(reference_stem).with_suffix(".npz")),
        candidate_arrays=str(Path(candidate_stem).with_suffix(".npz")),
        reference_arrays_sha256=a["array_sha256"], candidate_arrays_sha256=b["array_sha256"],
        source_population_identity=a["identity"], input_changes=changes,
        same_inputs_except_model=not other_changes, same_survival_at_observations=bool(survival_equal),
        legacy_no_added_diagnostic=legacy_no_added_diagnostic,
        declared_diagnostic_input_changes_verified=permitted_diagnostic,
        observation_count=len(a["arrays"]["z_mm"]), gun_recomputed=False, column_recomputed=False,
        gun_scope="Identical archived gun exit; internal gun propagation was not repeated",
        downstream_scope="Not compared: a matching legacy downstream capture is required")
    return clean(result)


def compare_existing_report(path, output, *, maximum_size_ratio=10.):
    """Reassess the recorded four-case arrays with the declared size criterion."""
    recorded = json.loads(path.read_text(encoding="utf-8"))
    cases = {row["case"]: row for row in recorded["cases"]}
    pairs = {}
    for key in ("current", "historical_controls"):
        a, b = (cases[key+suffix] for suffix in ("_custom", "_unified"))
        def stem(row):
            raw = Path(row["arrays"])
            return (raw if raw.exists() else path.parent/raw.name).with_suffix("")
        pairs[key] = compare_saved_pair(stem(a), stem(b), maximum_size_ratio=maximum_size_ratio)
    report = dict(schema="legacy-crossover-size-comparison-v1", utc=datetime.now(timezone.utc).isoformat(),
        archive=str(path), archive_sha256=sha256(path.read_bytes()).hexdigest(), pairs=pairs,
        legacy_aberration_scope="Custom retains its archived empirical Cs law; unified applies only its owned fields",
        no_new_transport=True, no_optical_parameter_changes=True,
        historical_scope="Historical controls on current geometry/source; separate explanatory reference, not the same current working point")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Legacy particle topology and size comparison", "", "Saved-state analysis only; no propagation, source regeneration or fitting.", "",
        f"The size gate is a maximum factor of {maximum_size_ratio:g} in current-weighted RMS radius. D95 is reported independently. All waist detection rules are unchanged.", ""]
    for name, item in pairs.items():
        lines += ["## "+name.replace("_", " ").title(), "", "Status: **"+item["status"]+"**.", "",
            f"Accepted waist counts: {item['reference_waist_count']} old / {item['candidate_waist_count']} unified. Unresolved candidates: {len(item['unresolved_candidates']['reference'])} / {len(item['unresolved_candidates']['candidate'])}.", "",
            "| Interval | Old Z (mm) | Unified Z (mm) | Old RMS (nm) | Unified RMS (nm) | RMS ratio | D95 ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in item["matched_waists"]:
            rms, q = row["common"]["rms_radius_nm"], row["common"]["diameter95_nm"]
            lines.append(f"| {row['region']} | {row['reference_z_mm']:.7g} | {row['candidate_z_mm']:.7g} | {rms['reference']:.8g} | {rms['candidate']:.8g} | {rms['ratio']:.8g} | {q['ratio']:.8g} |")
        sample = item["specimen_endpoint"]["rms_radius_nm"]
        lines += ["", f"Specimen endpoint RMS: {sample['reference']:.9g} / {sample['candidate']:.9g} nm; ratio {sample['ratio']:.9g}.", "",
            "A count mismatch disables positional waist pairing; all accepted and unresolved counts remain visible. An endpoint radius is not proof of two-sided focus.", ""]
        if not item["ordered_topology_matches"]:
            lines += ["| Model | Interval | Z (mm) | RMS radius (nm) | D95 (nm) |",
                      "| --- | --- | ---: | ---: | ---: |"]
            for model, rows in item["accepted_waists"].items():
                for row in rows:
                    stats = row["common"]["statistics"]
                    lines.append(f"| {model} | {row['region']} | {row['z_mm']:.9g} | {stats['radius_rms_m']*1e9:.9g} | {stats['radius_95_m']*2e9:.9g} |")
            lines.append("")
    lines += ["## Limits", "", "These are 193-particle observations on 5348 exact planes, with at most 0.5 mm spacing and local refinement. The old custom model retains its recorded empirical Cs treatment; this is a comparison with that model, not proof that the two field laws are equivalent.", "",
        "The high-count source and the downstream screen need separately matched captures. Existing unified-only downstream results cannot establish legacy/unified downstream consistency. Historical-control cases are not substituted for the current working point, and no threshold was changed to hide an extra waist."]
    output.with_suffix(".md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def read_source_result(prefix, source_cache):
    """Reuse a verified executed source; no gun preparation or resampling."""
    inputs = json.loads(prefix.with_suffix(".inputs.json").read_text(encoding="utf-8"))
    report = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    state = State.from_dict(inputs)
    if state.to_dict() != inputs:
        raise ValueError("Stored source inputs changed during reconstruction")
    with np.load(prefix.with_suffix(".npz"), allow_pickle=False) as saved:
        planes = saved["z_mm"].copy()
        expected_ids, expected_weights, expected_offsets = (
            saved[key].copy() for key in ("source_id", "weight", "source_energy_offset_ev"))
        entrance_kinetic = saved["kinetic_energy_ev"][0].copy()
    with np.load(source_cache, allow_pickle=False) as saved:
        gun = {key: saved[key].copy() for key in saved.files}
    initial = tuple(gun[key] for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    ids, weights, offsets, alive = (gun[key] for key in ("ray_id", "weight", "energy_offset_ev", "alive"))
    source_hash = _digest_arrays(*initial, offsets, weights, alive, ids)
    if source_hash != report["source_population_identity"]:
        raise ValueError("The cached gun-exit states do not match the archived source fingerprint")
    if not all(np.array_equal(a, b) for a, b in
               ((ids, expected_ids), (weights, expected_weights), (offsets, expected_offsets))):
        raise ValueError("Cached source identities, weights or energies changed")
    if planes[0] != state.electron_gun.exit_plane_z_mm or not alive.all():
        raise ValueError("This replay requires the archived wholly transmitted gun-exit population")
    if np.isfinite(gun["blocked_z_mm"]).any() or any(gun["blocked_key"]):
        raise ValueError("The cached source contains undeclared gun interception")
    # Recreate the original initialization exactly. Saved checkpoint kinetic
    # energies may carry one ULP from phi subtraction and are not reused here.
    kinetic = float(state.electron_gun.nominal_exit_energy_ev) + offsets
    if not np.allclose(entrance_kinetic, kinetic, rtol=0,
                       atol=2*float(np.spacing(np.max(kinetic)))):
        raise ValueError("Archived entrance kinetic energies disagree with nominal exit plus source offsets")
    source = dict(initial=initial, weights=weights, offsets=offsets, kinetic=kinetic,
        survival=(alive, gun["blocked_z_mm"], gun["blocked_key"].tolist()),
        identity=(ids, np.arctan2(initial[2], initial[0])), source_hash=source_hash,
        current_a=effective_source_current_a(state),
        checkpoint_energy_roundoff_ev=float(np.max(np.abs(entrance_kinetic-kinetic))))
    return inputs, dict(z_mm=planes), source


def replay_cached_source(prefix, source_cache, output, *, stop_mm=None, no_added=False):
    """Run only the legacy column on the exact existing source and planes."""
    inputs, archived, source = read_source_result(prefix, source_cache)
    requested = without_added_aberrations(inputs) if no_added else deepcopy(inputs)
    if stop_mm is not None:
        start, end = archived["z_mm"][[0, -1]]
        if not math.isfinite(stop_mm) or not start < stop_mm <= end:
            raise ValueError("A pilot endpoint must lie after gun exit and before the specimen")
        archived["z_mm"] = np.unique(np.r_[archived["z_mm"][archived["z_mm"] <= stop_mm], stop_mm])
    # execute_case writes the same standard arrays and metadata used by the
    # four-case comparison. The result filename explicitly retains its case.
    result = execute_case(requested, {}, archived, source, "current_custom", output.with_suffix(""))
    result.update(schema="cached-source-legacy-replay-v1", source_archive=str(prefix),
        source_cache=str(source_cache), source_cache_sha256=file_digest(source_cache),
        original_inputs_sha256=file_digest(prefix.with_suffix(".inputs.json")),
        diagnostic_input_changes=differences(inputs, requested), gun_recomputed=False,
        initial_kinetic_energy_definition="nominal_exit_energy_ev + original source_energy_offset_ev",
        archived_entrance_checkpoint_energy_roundoff_ev=source["checkpoint_energy_roundoff_ev"],
        source_azimuth_scope="Display-only azimuth from cached gun-exit x/y; phase, weights, IDs, clock and energies are unchanged",
        no_added_aberrations_diagnostic=no_added,
        partial_pilot=bool(stop_mm is not None and stop_mm < State.from_dict(inputs).sample.upper_surface_z_mm),
        observation_planes=len(archived["z_mm"]),
        scope="Incident column only; no specimen scattering or downstream propagation")
    output.write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return result


def read_source(archive):
    stem = archive / "full-column-baseline-n193-h005"
    inputs = json.loads(stem.with_suffix(".inputs.json").read_text(encoding="utf-8"))
    report = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(stem.with_suffix(".npz"), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    with np.load(stem.with_suffix(".gun.npz"), allow_pickle=False) as saved:
        gun = {key: saved[key].copy() for key in saved.files}
    state = State.from_dict(inputs)
    if state.to_dict() != inputs or arrays["z_mm"][0] != state.electron_gun.exit_plane_z_mm:
        raise ValueError("Comparison must retain the complete archived input and exact gun exit")
    initial = tuple(arrays[key][0].copy() for key in
                    ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
    weights, offsets, ids = (arrays[key] for key in ("weight", "source_energy_offset_ev", "source_id"))
    alive = arrays["alive"][0].copy()
    source_hash = _digest_arrays(*initial, offsets, weights, alive, ids)
    if source_hash != report["source_population_identity"]:
        raise ValueError("Archived gun-exit phase, clock, energy, weights or identity changed")
    if not np.array_equal(ids, gun["source_id"]) or not np.array_equal(weights, gun["weight"]):
        raise ValueError("Gun and column archives must retain the same source IDs and weights")
    if not alive.all() or np.isfinite(gun["blocked_z_mm"]).any():
        raise ValueError("This fixed-source comparison currently requires the wholly transmitted archive")
    kinetic = state.electron_gun.nominal_exit_energy_ev + offsets
    energy_roundoff = float(np.max(np.abs(arrays["kinetic_energy_ev"][0] - kinetic)))
    if not np.allclose(arrays["kinetic_energy_ev"][0], kinetic, rtol=0,
                       atol=2 * float(np.spacing(np.max(kinetic)))):
        raise ValueError("The archived column entrance must retain actual gun-exit kinetic energy")
    identity = (ids.copy(), np.arctan2(gun["equal_time_y_m"][0], gun["equal_time_x_m"][0]))
    source = dict(initial=initial, weights=weights, offsets=offsets, kinetic=kinetic,
                  survival=(alive, gun["blocked_z_mm"], [""] * len(ids)), identity=identity,
                  source_hash=source_hash, current_a=effective_source_current_a(state),
                  checkpoint_energy_roundoff_ev=energy_roundoff)
    return inputs, report, arrays, source


def configured_case(inputs, historical, name):
    requested = deepcopy(inputs)
    requested["simulation_mode"] = "unified" if name.endswith("_unified") else "custom"
    if name.startswith("historical_controls"):
        for lens in requested["lenses"]:
            if lens["key"] in INCIDENT_KEYS:
                lens["percent"] = historical["lenses"][lens["key"]]
        for component in requested["corrector_elements"]:
            if component["key"] in HEX_KEYS:
                component.update(historical["hexapoles"][component["key"]])
    state = State.from_dict(requested)
    if state.to_dict() != requested:
        raise ValueError("Constructing a diagnostic model changed undeclared instrument inputs")
    changes = differences(inputs, requested)
    for change in changes:
        path = change["path"]
        allowed = path == "/simulation_mode"
        if name.startswith("historical_controls"):
            parts = path.split("/")
            if len(parts) == 4 and parts[1] == "lenses" and parts[3] == "percent":
                allowed = requested["lenses"][int(parts[2])]["key"] in INCIDENT_KEYS
            if len(parts) == 4 and parts[1] == "corrector_elements" and parts[3] in {"strength_m3", "orientation_rad"}:
                allowed = requested["corrector_elements"][int(parts[2])]["key"] in HEX_KEYS
        if not allowed:
            raise ValueError(f"Undeclared diagnostic input change: {path}")
    return state, requested, changes


def execute_case(inputs, historical, archived, source, name, output_prefix):
    started = perf_counter()
    output = output_prefix.with_name(output_prefix.name + "-" + name)
    if any(output.with_suffix(suffix).exists() for suffix in (".npz", ".inputs.json")):
        raise FileExistsError("Preserve previous case arrays; choose a new output prefix")
    state, requested, changes = configured_case(inputs, historical, name)
    snapshot = capture_instrument_snapshot(state)
    planes = archived["z_mm"]
    start, stop = float(planes[0]), float(planes[-1])
    receipts = []
    print(f"Starting {name}: {len(source['weights'])} unchanged gun-exit particles", flush=True)
    transport_started = perf_counter()
    with backend_receipts(receipts):
        segment, _, reused = _trace_segment(state, "incident", start, stop,
            source["initial"], source["offsets"], source["weights"], source["identity"],
            source["survival"], _events(state), None, math.inf, initial_kicks=True,
            save_z_mm=tuple(planes), initial_kinetic_energy_ev=source["kinetic"],
            checkpoint_ray_count=len(source["weights"]))
    transport_elapsed = perf_counter()-transport_started
    analysis_started = perf_counter()
    changes_during_execution = differences(requested, state.to_dict())
    if any(row["path"] != "/active_backend" for row in changes_during_execution):
        raise ValueError("Column execution changed physical input controls")
    branch, cp = segment.branch, segment.checkpoints
    indices = np.searchsorted(cp.z_mm, planes)
    if not np.array_equal(cp.z_mm[indices], planes):
        raise ValueError("A diagnostic model lost an exact requested observation plane")
    from temsim.physics.core import PropagationCheckpoints
    selected = PropagationCheckpoints(planes.copy(), *(np.asarray(getattr(cp, key))[indices].copy()
        for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")))
    stops = np.asarray(branch.blocked_z)
    masks = source["survival"][0][None, :] & (~np.isfinite(stops)[None, :] | (stops[None, :] > planes[:, None]))
    statistics = transverse_beam_statistics(*(getattr(selected, key)[-1] for key in
        ("x_m", "y_m", "tx_rad", "ty_rad")), weights=source["weights"], alive=masks[-1])
    chain = evaluate_crossover_chain(selected, masks, source["weights"], source["identity"][0],
        source_current_a=source["current_a"], regions=regions_for_state(state), maximum_spacing_mm=.5,
        source_identity=source["source_hash"])
    labels, counts = np.unique(np.asarray(branch.blocked_key)[np.isfinite(stops)], return_counts=True)
    analysis_elapsed = perf_counter()-analysis_started
    serialization_started = perf_counter()
    np.savez_compressed(output.with_suffix(".npz"), z_mm=planes,
        **{key: np.asarray(getattr(selected, key)) for key in
           ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")},
        alive=masks, weight=source["weights"], source_id=source["identity"][0],
        source_energy_offset_ev=source["offsets"], blocked_z_mm=stops,
        blocked_key=np.asarray(branch.blocked_key, dtype=str))
    output.with_suffix(".inputs.json").write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
    serialization_elapsed = perf_counter()-serialization_started
    comparison = None
    if name == "current_unified":
        comparison = {key: float(np.nanmax(np.abs(np.asarray(getattr(selected, key)) - archived[key])))
            for key in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")}
    result = dict(case=name, elapsed_s=perf_counter()-started, simulation_mode=state.simulation_mode,
        column_transport_elapsed_s=transport_elapsed, analysis_elapsed_s=analysis_elapsed,
        serialization_elapsed_s=serialization_elapsed,
        input_snapshot_digest=snapshot.digest, input_changes=changes, runtime_only_changes=changes_during_execution,
        source_population_identity=source["source_hash"], source_current_a=source["current_a"],
        source_rays=len(source["weights"]), column_prefix_reused=reused,
        physical_inputs_changed_during_execution=False, optical_working_point_installed=False,
        historical_scope=("Recorded incident lens and hexapole controls on current geometry/source/solver; not full historical replay"
                          if name.startswith("historical") else "Current frozen physical working point"),
        sample_z_mm=stop, statistics=asdict(statistics), alpha95_mrad=statistics.convergence_95_mrad,
        alpha99_mrad=statistics.convergence_99_mrad, sampled_max_angle_mrad=statistics.convergence_edge_rad*1000,
        physical_pupil_edge_mrad=None, rms_radius_nm=statistics.radius_rms_m*1e9,
        covariance_drift_waist_offset_nm=statistics.waist_offset_m*1e9,
        focus_scope="Covariance drift extrapolation at the sample; not exact finite-field best focus",
        current_a=source["current_a"]*statistics.surviving_fraction,
        physical_stop_counts=dict(zip(labels.tolist(), counts.tolist())), crossover_chain=chain,
        solver_signature=segment.plan.solver_signature, backend_receipts=receipts,
        archived_same_case_max_absolute_differences=comparison,
        arrays=str(output.with_suffix(".npz")), input_file=str(output.with_suffix(".inputs.json")))
    print(f"Finished {name}: alpha95={result['alpha95_mrad']:.8g} mrad, RMS={result['rms_radius_nm']:.8g} nm, {result['elapsed_s']:.1f} s", flush=True)
    return clean(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--compare-existing", type=Path,
        help="Analyze the saved four-case report; do not run propagation")
    actions.add_argument("--source-result", type=Path,
        help="Replay only the custom column using this saved result prefix")
    parser.add_argument("--source-cache", type=Path, help="Verified executed gun-exit NPZ for --source-result")
    parser.add_argument("--stop-mm", type=float, help="Optional bounded replay pilot endpoint")
    parser.add_argument("--legacy-no-added-aberrations", action="store_true",
        help="Diagnostic custom replay disabling added aberrations; never changes the unified working point")
    parser.add_argument("--output", type=Path,
        default=ROOT / "outputs/extended-optical-validation-20261027/controlled-model-comparison.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous comparison; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.compare_existing:
        compare_existing_report(args.compare_existing, args.output)
        return 0
    if args.source_result:
        if args.source_cache is None:
            parser.error("--source-result requires --source-cache")
        replay_cached_source(args.source_result, args.source_cache, args.output,
            stop_mm=args.stop_mm, no_added=args.legacy_no_added_aberrations)
        return 0
    if args.source_cache or args.stop_mm is not None or args.legacy_no_added_aberrations:
        parser.error("Replay-specific options require --source-result")
    inputs, evidence, archived, source = read_source(args.archive)
    historical_path = ROOT / "docs/development/nanoprobe_normal_20261014.json"
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    report = dict(schema="fixed-source-particle-model-comparison-v1", utc=datetime.now(timezone.utc).isoformat(),
        status="RUNNING", solver_identity=solver_source_identity(), archived_source=args.archive.as_posix(),
        archived_source_identity=source["source_hash"], original_source_verified=True,
        archived_entrance_checkpoint_energy_roundoff_ev=source["checkpoint_energy_roundoff_ev"],
        column_only=True, gun_rerun=False, geometry_source_and_current_held_fixed=True,
        historical_reference=str(historical_path), historical_reference_sha256=sha256(historical_path.read_bytes()).hexdigest(),
        source_mean_energy_ev=inputs["electron_gun"]["components"]["feg_tip"]["emission_energy_ev"],
        source_energy_fwhm_ev=inputs["electron_gun"]["components"]["feg_tip"]["energy_spread_fwhm_ev"],
        maximum_step_mm=inputs["step_mm"], observation_planes=len(archived["z_mm"]),
        target_definition="30 mrad physical pupil edge; alpha95 below is descriptive, not target acceptance",
        model_distinction="Both models have finite axial fields. Custom uses parameterized paraxial/canonical transport with configured empirical aberrations; unified uses shared three-dimensional potentials and natural field aberrations without empirical duplication.",
        cases=[], errors=[])
    for name in args.cases:
        try:
            report["cases"].append(execute_case(inputs, historical, archived, source, name, args.output.with_suffix("")))
        except Exception as error:
            report["errors"].append(dict(case=name, type=type(error).__name__, message=str(error)))
            report["status"] = "FAILED"
            args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
            raise
        args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if solver_source_identity() != report["solver_identity"]:
        raise ValueError("The solver implementation changed during the controlled comparison")
    report["status"] = "EXECUTED_NOT_OPTICALLY_QUALIFIED"
    args.output.write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
