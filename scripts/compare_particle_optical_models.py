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
    state, requested, changes = configured_case(inputs, historical, name)
    snapshot = capture_instrument_snapshot(state)
    planes = archived["z_mm"]
    start, stop = float(planes[0]), float(planes[-1])
    receipts = []
    print(f"Starting {name}: {len(source['weights'])} unchanged gun-exit particles", flush=True)
    with backend_receipts(receipts):
        segment, _, reused = _trace_segment(state, "incident", start, stop,
            source["initial"], source["offsets"], source["weights"], source["identity"],
            source["survival"], _events(state), None, math.inf, initial_kicks=True,
            save_z_mm=tuple(planes), initial_kinetic_energy_ev=source["kinetic"],
            checkpoint_ray_count=len(source["weights"]))
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
    output = output_prefix.with_name(output_prefix.name + "-" + name)
    np.savez_compressed(output.with_suffix(".npz"), z_mm=planes,
        **{key: np.asarray(getattr(selected, key)) for key in
           ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev")},
        alive=masks, weight=source["weights"], source_id=source["identity"][0],
        source_energy_offset_ev=source["offsets"], blocked_z_mm=stops,
        blocked_key=np.asarray(branch.blocked_key, dtype=str))
    output.with_suffix(".inputs.json").write_text(json.dumps(requested, indent=2) + "\n", encoding="utf-8")
    comparison = None
    if name == "current_unified":
        comparison = {key: float(np.nanmax(np.abs(np.asarray(getattr(selected, key)) - archived[key])))
            for key in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")}
    result = dict(case=name, elapsed_s=perf_counter()-started, simulation_mode=state.simulation_mode,
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
    parser.add_argument("--output", type=Path,
        default=ROOT / "outputs/extended-optical-validation-20261027/controlled-model-comparison.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous comparison; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
