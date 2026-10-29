"""Explicit controlled column-entry energy diagnostics; never a real gun result.

No emission, gun trace, source resampling, lens tuning, or empirical aberration
is introduced. A complete saved gun-exit population is replayed in production
unified fields. Energy interventions are recorded separately from its donor.
Only run after the supervising workflow releases the physical-job slot.
"""
from dataclasses import dataclass
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/"agent.md").is_file())
for path in (ROOT/"src", ROOT/"scripts"):
    sys.path.insert(0, str(path))

SCHEMA = "temsim-controlled-column-energy-v1"
PHASE = ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")
INITIAL = ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s")
DEFAULT_DONOR = ROOT/"outputs/joint-probe-100um-30mrad-20261028/seed769-native-d95-waist-h005"


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    if value.dtype.hasobject:
        raise ValueError("Object arrays cannot supply source evidence")
    return sha256(str((value.dtype.str, value.shape)).encode()+value.tobytes()).hexdigest()


def finite(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.number)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    return float(value)


@dataclass
class Donor:
    directory: Path
    state: object
    snapshot: object
    arrays: dict
    report: dict
    planes: np.ndarray
    field_identity: dict
    files_sha256: dict


def validate_arrays(arrays, planes):
    ids, weights = arrays["source_id"], arrays["weights"]
    n = len(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0) or len(np.unique(ids)) != n
            or weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not 0 < weights.sum() < math.inf or not np.array_equal(ids, arrays["emission_ray_id"])):
        raise ValueError("Invalid complete original source IDs or weights")
    z, mesh, alive = arrays["z_mm"], arrays["integration_z_mm"], arrays["alive"]
    for grid in (z, mesh, planes):
        if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
            raise ValueError("Donor grids must be finite and strictly increasing")
    if any(np.count_nonzero(z == p) != 1 for p in planes) or any(p not in mesh for p in z):
        raise ValueError("Donor is missing exact requested checkpoints or integration nodes")
    if alive.shape != (len(z), n) or alive.dtype.kind != "b" or np.any(~alive[:-1] & alive[1:]):
        raise ValueError("Invalid or revived donor checkpoint population")
    live = arrays["gun_exit_alive"]
    if live.shape != (n,) or live.dtype.kind != "b":
        raise ValueError("Invalid donor gun survival mask")
    for key in (*INITIAL, "kinetic_energy_ev"):
        value = arrays["gun_exit_"+key]
        if value.shape != (n,) or not np.isfinite(value[live]).all():
            raise ValueError("Invalid live original gun-exit phase")
    if np.any(arrays["gun_exit_kinetic_energy_ev"][live] <= 0) or np.any(arrays["gun_exit_flight_time_s"][live] < 0):
        raise ValueError("Invalid gun-exit energies or flight clocks")
    for key in PHASE:
        value = arrays[key]
        if value.shape != alive.shape or not np.isfinite(value[alive]).all():
            raise ValueError("Invalid original live checkpoint phase")
    if np.any(arrays["kinetic_energy_ev"][alive] <= 0) or np.any(arrays["flight_time_s"][alive] < 0):
        raise ValueError("Invalid original live checkpoint energy or clock")
    for key in ("gun_blocked_z_mm", "gun_blocked_key", "source_azimuth_rad", "source_energy_offset_ev"):
        if arrays[key].shape != (n,):
            raise ValueError("Incomplete original source metadata")
    if not np.array_equal(alive[0], live):
        raise ValueError("Donor initial checkpoint does not preserve gun survival")
    for key in INITIAL:
        if not np.array_equal(arrays[key][0, live], arrays["gun_exit_"+key][live]):
            raise ValueError("Donor initial checkpoint differs from its original gun exit")


def verify_initial_energy(saved, reference, phi, active):
    """Check the production K+phi-phi+0 operation order, without a tolerance."""
    expected = reference[active]+phi[active]-phi[active]+np.zeros(int(active.sum()))
    if not np.array_equal(saved[active], expected):
        raise ValueError("Initial checkpoint energy differs from the exact production potential-work expression")


def load_donor(path):
    """Check a completed archive, including its static entrance field; never propagate."""
    from temsim.instrument_snapshot import InstrumentSnapshot, capture_instrument_snapshot
    from validate_joint_probe import _restore_exact
    directory = Path(path).resolve()
    if directory.is_file():
        directory = directory.parent
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    if report.get("execution_status") != "COMPLETE" or report.get("receipt", {}).get("status") != "COMPLETE":
        raise ValueError("A completed native-profile donor is required")
    hashes = {name: file_hash(directory/name) for name in ("inputs.json", "particles.npz", "report.json")}
    for name in ("inputs.json", "particles.npz"):
        if hashes[name] != report.get("files_sha256", {}).get(name):
            raise ValueError(f"Donor file checksum mismatch: {name}")
    snapshot = InstrumentSnapshot.from_dict(json.loads((directory/"inputs.json").read_text(encoding="utf-8")))
    state = _restore_exact(snapshot)
    if snapshot.digest != report["input_digest"] or snapshot.to_dict()["implementation"] != report["implementation"]:
        raise ValueError("Donor receipt and full input snapshot differ")
    if capture_instrument_snapshot(state).digest != snapshot.digest:
        raise ValueError("Donor restoration changed the complete inputs")
    with np.load(directory/"particles.npz", allow_pickle=False) as raw:
        arrays = {key: raw[key].copy() for key in raw.files}
    planes = np.asarray(report["parameters"]["requested_planes_mm"], dtype=float)
    validate_arrays(arrays, planes)
    if float(arrays["z_mm"][0]) != float(state.electron_gun.exit_plane_z_mm):
        raise ValueError("The donor does not begin at the exact physical gun exit")
    nominal = float(state.beam_voltage_kv)*1000.
    if (float(state.electron_gun.nominal_exit_energy_ev) != nominal or not np.array_equal(
            nominal+arrays["source_energy_offset_ev"], arrays["gun_exit_kinetic_energy_ev"], equal_nan=True)):
        raise ValueError("Original energy offsets do not bind the saved gun-exit kinetic energies")
    if finite(report["receipt"]["source_current_a"], "Original emitted current") <= 0:
        raise ValueError("Original current must be positive")
    batches = report["receipt"]["batches"]
    identity = batches[0]["field_audit"]["identities"]
    if any(row["field_audit"]["identities"] != identity for row in batches):
        raise ValueError("Donor batches did not share one captured field plan")
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.unified_field_particle import _sampler
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    runtime = get_unified_field_runtime(state)
    if (runtime.digest != identity["runtime_digest"]
            or runtime.snapshot.numerical_digest != identity["potential_digest"]
            or type(runtime.snapshot) is DynamicPotentialSnapshot):
        raise ValueError("Donor requires its exact static captured fields")
    live=arrays["gun_exit_alive"]
    phi=np.full(len(live),np.nan)
    phi[live]=_sampler(runtime.snapshot)(np.column_stack((arrays["gun_exit_x_m"][live],
        arrays["gun_exit_y_m"][live],np.full(int(live.sum()),arrays["z_mm"][0]*.001))))["phi_v"]
    verify_initial_energy(arrays["kinetic_energy_ev"][0],arrays["gun_exit_kinetic_energy_ev"],phi,live)
    return Donor(directory, state, snapshot, arrays, report, planes, identity, hashes)


def prepare_energy(donor, mode, *, constant_ev=300000.3, energy_donor=None):
    original = donor.arrays["gun_exit_kinetic_energy_ev"]
    if mode == "original":
        if energy_donor is not None:
            raise ValueError("Original replay cannot use another energy donor")
        kinetic = original.copy()
        offsets = donor.arrays["source_energy_offset_ev"].copy()
    elif mode == "constant":
        if energy_donor is not None or finite(constant_ev,"Column kinetic energy") <= 0:
            raise ValueError("Constant injection needs positive energy and no energy donor")
        kinetic = np.full(original.shape, float(constant_ev))
        offsets = kinetic-float(donor.state.beam_voltage_kv)*1000.
    elif mode == "donor-energy":
        if energy_donor is None:
            raise ValueError("The crossed case requires an explicit complete energy donor")
        for key in ("source_id", "weights", "emission_ray_id", "emission_position_m", "emission_direction", "gun_exit_alive"):
            if not np.array_equal(donor.arrays[key], energy_donor.arrays[key], equal_nan=True):
                raise ValueError(f"Crossed donors differ in original source identity or admission: {key}")
        for key in ("potential_digest", "geometry_digest"):
            if donor.field_identity[key] != energy_donor.field_identity[key]:
                raise ValueError("Crossed donors used different actual fields or mechanics")
        if float(donor.state.beam_voltage_kv) != float(energy_donor.state.beam_voltage_kv):
            raise ValueError("Crossed donor nominal beam energies differ")
        kinetic = energy_donor.arrays["gun_exit_kinetic_energy_ev"].copy()
        offsets = energy_donor.arrays["source_energy_offset_ev"].copy()
    else:
        raise ValueError("Unsupported column energy intervention")
    active = donor.arrays["gun_exit_alive"]
    if kinetic.shape != original.shape or np.any(~np.isfinite(kinetic[active])) or np.any(kinetic[active] <= 0):
        raise ValueError("Injected live kinetic energies must be positive finite and complete")
    return kinetic, offsets, dict(mode=mode, constant_kinetic_ev=float(constant_ev) if mode=="constant" else None,
        original_kinetic_sha256=array_hash(original), injected_kinetic_sha256=array_hash(kinetic),
        injected_offsets_sha256=array_hash(offsets),
        energy_donor_files_sha256=None if energy_donor is None else energy_donor.files_sha256,
        identity_scope="All original donor positions, slopes, clocks, IDs, weights and gun survival remain fixed; only the declared initial column kinetic energy changes",
        scientific_scope="Full downstream E+B response conditional on saved entry phase; not a physical monochromator or an executed modified gun")


def compare_replay(donor, arrays, start, stop):
    checks = {"z_mm": np.array_equal(arrays["z_mm"], donor.arrays["z_mm"]),
              "integration_z_mm": np.array_equal(arrays["integration_z_mm"], donor.arrays["integration_z_mm"])}
    checks.update({key:np.array_equal(arrays[key],donor.arrays[key][:,start:stop],equal_nan=True)
                   for key in (*PHASE,"alive")})
    checks.update({key:np.array_equal(arrays[key],donor.arrays[key][start:stop],equal_nan=True)
                   for key in ("source_id","weights","blocked_z_mm","terminal_alive")})
    checks["blocked_key"] = np.array_equal(arrays["blocked_key"],donor.arrays["blocked_key"][start:stop])
    return dict(status="EXACT_REPLAY" if all(checks.values()) else "REPLAY_MISMATCH",
                source_slice=[start,stop],checks={k:bool(v) for k,v in checks.items()})


def execute_injection(donor, kinetic, offsets, cancelled, *, max_batches=None,
                      batch_callback=lambda row, arrays:None, trace_segment=None, audit_plan=None):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.particle_sections import _trace_segment, _events
    from measure_probe_waist import audit_extended_plan
    trace_segment = _trace_segment if trace_segment is None else trace_segment
    audit_plan = audit_extended_plan if audit_plan is None else audit_plan
    state, source = donor.state, donor.arrays
    state._tuning_cancelled = cancelled
    n = len(source["source_id"])
    size = int(donor.report["parameters"]["batch_size"])
    if not 1 <= size <= 128:
        raise ValueError("Unsupported original donor batch size")
    if max_batches is not None and (type(max_batches) is not int or max_batches<1):
        raise ValueError("Replay batch limit must be a positive integer")
    batches, pieces = [], []
    for number,start in enumerate(range(0,n,size),1):
        if max_batches is not None and number>max_batches: break
        if cancelled(): raise InterruptedError("Controlled-injection diagnostic cancelled")
        stop = min(n,start+size); sel=slice(start,stop)
        segment,resume,reused = trace_segment(state,"incident",float(source["z_mm"][0]),float(donor.planes[-1]),
            tuple(source["gun_exit_"+key][sel].copy() for key in INITIAL), offsets[sel].copy(),source["weights"][sel].copy(),
            (source["source_id"][sel].copy(),source["source_azimuth_rad"][sel].copy()),
            (source["gun_exit_alive"][sel].copy(),source["gun_blocked_z_mm"][sel].copy(),source["gun_blocked_key"][sel].tolist()),
            _events(state),None,math.inf,initial_kicks=True,save_z_mm=tuple(donor.planes),
            initial_kinetic_energy_ev=kinetic[sel].copy(),checkpoint_ray_count=n)
        cp,branch=segment.checkpoints,segment.branch
        audit=audit_plan(state,segment.plan,float(donor.planes[-1]))
        if audit["identities"] != donor.field_identity:
            raise ValueError("Controlled injection changed the actual field, geometry or integration plan identity")
        if (reused or resume!=float(source["z_mm"][0]) or not np.array_equal(segment.plan.z_mm,source["integration_z_mm"])
                or not np.array_equal(cp.z_mm,source["z_mm"])):
            raise ValueError("Replay must execute the complete original mesh and checkpoints without prefix reuse")
        if not np.array_equal(branch.source_ray_id,source["source_id"][sel]) or not np.array_equal(branch.ray_weight,source["weights"][sel]):
            raise ValueError("Original source IDs or absolute weights changed")
        raw={key:np.asarray(getattr(cp,key)).copy() for key in PHASE}
        raw.update(alive=np.asarray(cp.unified_state.active).copy(),z_mm=cp.z_mm.copy(),integration_z_mm=segment.plan.z_mm.copy(),
            source_id=source["source_id"][sel].copy(),weights=source["weights"][sel].copy(),
            blocked_z_mm=np.asarray(branch.blocked_z).copy(),blocked_key=np.asarray(branch.blocked_key,dtype=str),
            terminal_alive=np.asarray(branch.alive).copy())
        if any(not np.array_equal(raw[key][0,source["gun_exit_alive"][sel]],
                source["gun_exit_"+key][sel][source["gun_exit_alive"][sel]],equal_nan=True) for key in INITIAL):
            raise ValueError("Controlled injection changed the original entry phase or flight clocks")
        if not np.array_equal(cp.unified_state.reference_energy_ev,kinetic[sel],equal_nan=True):
            raise ValueError("The physical executor did not use the declared initial kinetic energies")
        verify_initial_energy(raw["kinetic_energy_ev"][0],kinetic[sel],cp.unified_state.reference_phi_v,
                              source["gun_exit_alive"][sel])
        row=dict(batch=number,source_slice=[start,stop],status="COMPLETE",field_audit=audit,
                 actual_reference_kinetic_sha256=array_hash(cp.unified_state.reference_energy_ev),
                 initial_checkpoint_kinetic_sha256=array_hash(raw["kinetic_energy_ev"][0]),
                 initial_energy_check="Exact K+phi-phi+zero-work operation order; reference K is separately exact")
        batch_callback(row,raw)
        batches.append(row);pieces.append(raw)
    if capture_instrument_snapshot(state).digest != donor.snapshot.digest:
        raise ValueError("Controlled injection mutated frozen physical inputs")
    result={key:np.concatenate([p[key] for p in pieces],axis=1) for key in (*PHASE,"alive")}
    result.update({key:np.concatenate([p[key] for p in pieces]) for key in ("source_id","weights","blocked_z_mm","blocked_key","terminal_alive")})
    result.update(z_mm=source["z_mm"].copy(),integration_z_mm=source["integration_z_mm"].copy())
    return result,batches


def run(donor_path,output,*,mode="constant",constant_ev=300000.3,energy_donor_path=None,
        threads=4,budget_s=1800.,cancel_file=None,replay_batches=None):
    from temsim.cpu_resources import numerical_job
    from solve_joint_probe import atomic_json,atomic_npz
    from measure_probe_waist import analyze_scan
    if type(threads) is not int or threads<1 or finite(budget_s,"Budget")<=0:
        raise ValueError("Positive thread and time budgets are required")
    if replay_batches is not None and mode!="original":
        raise ValueError("Partial execution is only allowed for original baseline replay")
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=perf_counter()
    def cancelled():return perf_counter()-started>=budget_s or bool(cancel_file and Path(cancel_file).exists())
    report=dict(schema=SCHEMA,status="RUNNING",qualification="CONTROLLED_DIAGNOSTIC_NOT_QUALIFIED",mode=mode,
        source_scope="Saved gun-exit controlled injection; no gun execution or source generation",
        no_size_acceptance_threshold=True,threads_requested=threads,budget_s=budget_s,
        cancellation="Cooperative executor checks; no hard deadline",batches=[])
    atomic_json(output/"report.json",report)
    try:
        donor=load_donor(donor_path)
        other=None if energy_donor_path is None else load_donor(energy_donor_path)
        kinetic,offsets,manifest=prepare_energy(donor,mode,constant_ev=constant_ev,energy_donor=other)
        report.update(donor_path=str(donor.directory),donor_files_sha256=donor.files_sha256,
            donor_input_digest=donor.snapshot.digest,implementation=donor.snapshot.to_dict()["implementation"],
            intervention=manifest,requested_planes_mm=donor.planes.tolist(),
            source_current_a=donor.report["receipt"]["source_current_a"],source_points=len(kinetic),
            source_array_sha256={k:array_hash(v) for k,v in donor.arrays.items() if k.startswith(("gun_exit_","emission_")) or k in ("source_id","weights")},
            script_sha256=file_hash(__file__))
        atomic_json(output/"donor-inputs.json",donor.snapshot.to_dict())
        atomic_npz(output/"injection.npz",dict(original_kinetic_energy_ev=donor.arrays["gun_exit_kinetic_energy_ev"],
            injected_kinetic_energy_ev=kinetic,injected_energy_offset_ev=offsets,source_id=donor.arrays["source_id"],weights=donor.arrays["weights"]))
        (output/"driver-source.py").write_bytes(Path(__file__).read_bytes())
        atomic_json(output/"report.json",report)
        def batch_done(row,arrays):
            folder=output/f"batch-{row['batch']:03d}";folder.mkdir()
            atomic_npz(folder/"particles.npz",arrays)
            if mode=="original":
                row["replay"]=compare_replay(donor,arrays,*row["source_slice"])
            atomic_json(folder/"receipt.json",row)
            report["batches"].append(row);atomic_json(output/"report.json",report)
            if mode=="original" and row["replay"]["status"]!="EXACT_REPLAY":
                raise ValueError("Original baseline replay differs; do not interpret any energy intervention")
        with numerical_job(threads,cancelled=cancelled) as cpu:
            report["cpu_resources"]=cpu.to_dict()
            arrays,batches=execute_injection(donor,kinetic,offsets,cancelled,max_batches=replay_batches,batch_callback=batch_done)
        atomic_npz(output/"particles.npz",arrays)
        full=len(arrays["source_id"])==len(kinetic)
        report.update(status="COMPLETE",full_original_population=full,processed_source_points=len(arrays["source_id"]),
            measurement_scope="FULL_ORIGINAL_SOURCE" if full else "PARTIAL_REPLAY_NO_FULL_SOURCE_STATISTICS")
        if full:
            analyzed=analyze_scan(arrays,donor.planes,surface_z_mm=float(donor.state.sample.upper_surface_z_mm),
                source_current_a=report["source_current_a"])
            report["observations"]={k:analyzed[k] for k in ("planes","surface","d95_waist","rms_waist_diagnostic","positive_current_membership_stable")}
            # All size/focus values are measurements. No legacy .7 nm gate is retained.
            c2=batches[0]["field_audit"]["c2_aperture"]
            z=.5*(c2["local_start_z_mm"]+c2["local_end_z_mm"])+1000*c2["origin_global_m"][2]
            where=np.flatnonzero(arrays["z_mm"]==z)
            if len(where)==1:
                i=int(where[0]);live=arrays["alive"][i] & (arrays["weights"]>0)
                report["c2_reference_current"]=dict(z_mm=z,diameter_mm=c2["diameter_mm"],points=int(live.sum()),
                    current_a=report["source_current_a"]*float(arrays["weights"][live].sum()/donor.arrays["weights"].sum()),
                    maximum_radius_um=float(np.hypot(arrays["x_m"][i,live],arrays["y_m"][i,live]).max()*1e6) if live.any() else None,
                    scope="Exact central reference-plane survivors; not a separate finite-plate entrance/exit measurement")
        if any(file_hash(donor.directory/name)!=expected for name,expected in donor.files_sha256.items()):
            raise ValueError("Original donor files changed during the diagnostic")
        report["donor_files_unchanged"]=True
    except BaseException as error:
        report.update(status="CANCELLED" if cancelled() or isinstance(error,(KeyboardInterrupt,InterruptedError)) else "FAILED",
                      error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_s"]=perf_counter()-started
        atomic_json(output/"report.json",report)
        atomic_json(output/"checksums.json",{str(p.relative_to(output)):file_hash(p) for p in output.rglob('*') if p.is_file() and p.name!='checksums.json'})
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor",type=Path,default=DEFAULT_DONOR)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("original","constant","donor-energy"),default="constant")
    parser.add_argument("--constant-ev",type=float,default=300000.3)
    parser.add_argument("--energy-donor",type=Path)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--budget-s",type=float,default=1800.)
    parser.add_argument("--cancel-file",type=Path)
    parser.add_argument("--replay-batches",type=int,help="Original-mode baseline verification only; omitted means full population")
    args=parser.parse_args(argv)
    report=run(args.donor,args.output,mode=args.mode,constant_ev=args.constant_ev,energy_donor_path=args.energy_donor,
        threads=args.threads,budget_s=args.budget_s,cancel_file=args.cancel_file,replay_batches=args.replay_batches)
    print(json.dumps(dict(status=report["status"],qualification=report["qualification"],report=str(args.output/"report.json"))))
    return 0


if __name__=="__main__":raise SystemExit(main())
