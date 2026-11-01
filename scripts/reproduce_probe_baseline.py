"""Replay a frozen original gun outlet without changing optics or particle count."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from measure_crossover_cs import component_plane, target_current_kernel, trace_checkpoints
from measure_probe_crossover import measure
from measure_probe_waist import PHASE, _array_hash, _json, _npz, audit_extended_plan
from temsim import input_io
from temsim.cpu_resources import numerical_job
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.particle_sections import gun_dependency_signature
from temsim.physics.unified_field_particle import build_unified_plan
from temsim.physics.unified_field_runtime import get_unified_field_runtime
from temsim.simulation_modes import is_unified
from temsim.specimen.source import specimen_is_vacuum
from temsim.working_point import WorkingPointCheckpoint


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_solver_sources(manifest, root=ROOT / "src/temsim"):
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*.py")}
    if set(actual) != set(manifest["files"]):
        raise ValueError("The accepted solver source inventory changed")
    changed_endings = []
    for name, path in actual.items():
        data = path.read_bytes()
        expected = manifest["files"][name]
        if sha256(data.replace(b"\r\n", b"\n")).hexdigest() != expected["lf_sha256"]:
            raise ValueError(f"The accepted solver code changed in {name}")
        if sha256(data).hexdigest() != expected["raw_sha256"]:
            changed_endings.append(name)
    return sorted(changed_endings)


def field_content_digest(record):
    """Hash all physical provider contents while separating raw-code receipts."""
    record = json.loads(json.dumps(record))
    record.pop("numerical_digest")
    for contribution in record["contributions"]:
        if contribution["source_id"] == "instrument_electric":
            potential = contribution["potential"]
            potential.pop("digest")
            metadata = potential["metadata"]
            metadata["request"].pop("implementation_sha256")
            metadata["source_identities"].pop("request")
            metadata["source_identities"].pop("numerical")
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_baseline(folder, *, portable=True):
    metadata = read_json(folder / "baseline.json")
    for name, digest in metadata["files"].items():
        if sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed frozen input file {name}")
    name = "portable-probe.temwp" if portable else "calibrated-probe.temwp"
    checkpoint = WorkingPointCheckpoint.read_package(folder / name)
    ending_changes = verify_solver_sources(read_json(folder / "solver-sources.json"))
    # Inputs-only restoration is followed by a new physical execution.
    # Git newline conversion is checked against each archived solver file.
    state = checkpoint.compatible_state() if ending_changes else checkpoint.snapshot.restore()
    identity = capture_instrument_snapshot(state)
    if (identity.graph != checkpoint.snapshot.graph
            or identity.external_inputs != checkpoint.snapshot.external_inputs):
        raise ValueError("Restoration changed a physical input")
    metadata["solver_newline_only_changes"] = ending_changes
    runtime = get_unified_field_runtime(state)
    field_identity_changed = runtime.snapshot.numerical_digest != metadata["potential_digest"]
    if (field_content_digest(runtime.snapshot.to_dict()) != metadata["field_content_sha256"]
            or runtime.geometry.digest != metadata["geometry_digest"]
            or (field_identity_changed and not ending_changes)):
        raise ValueError("The accepted physical fields or geometry changed")
    with input_io.input_scope(state, inherit=False):
        signature = gun_dependency_signature(state)
    if signature != metadata["gun_dependency_signature"] and not ending_changes:
        raise ValueError("The frozen source is incompatible with the gun inputs")
    metadata["current_gun_dependency_signature"] = signature
    metadata["field_identity_changed_by_code_receipts"] = field_identity_changed
    with np.load(folder / "source-2000.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    for key, digest in metadata["source_sha256"].items():
        if _array_hash(arrays[key]) != digest:
            raise ValueError(f"Changed original source field {key}")
    if (len(arrays["weights"]) != 2000 or np.count_nonzero(arrays["weights"] > 0) != 2000
            or len(np.unique(arrays["source_id"])) != 2000
            or not arrays["gun_exit_alive"].all()
            or np.isfinite(arrays["gun_blocked_z_mm"]).any()):
        raise ValueError("The accepted original population is incomplete")
    if not is_unified(state) or not specimen_is_vacuum(state.sample):
        raise ValueError("This input must describe the unified incident vacuum probe")
    if capture_instrument_snapshot(state).digest != identity.digest:
        raise ValueError("Input verification changed the restored state")
    return state, identity, metadata, arrays


def render_current(kernel, settings, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.ticker import FuncFormatter
    current = kernel["current_a_yx"] * 1e9
    edges, ye = kernel["x_edges_nm"], kernel["y_edges_nm"]
    high = settings["color_max_nA"]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 4.1), layout="constrained")
    for axis, zoom, title in zip(axes, (None, settings["central_halfwidth_nm"]),
                                 ("Full field · 5.12 nm", "Central field · 0.44 nm")):
        im = axis.imshow(current, extent=[edges[0], edges[-1], ye[0], ye[-1]], origin="lower",
                         interpolation="nearest", cmap="magma", aspect="equal",
                         norm=SymLogNorm(linthresh=settings["symlog_linthresh_nA"], vmin=0, vmax=high))
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("X (nm)")
        axis.set_ylabel("Y (nm)")
        if zoom:
            axis.set_xlim(-zoom, zoom)
            axis.set_ylim(-zoom, zoom)
            axis.set_xticks([-.2, -.1, 0., .1, .2])
            axis.set_yticks([-.2, -.1, 0., .1, .2])
    bar = figure.colorbar(im, ax=axes, shrink=.78, pad=.025,
                         extend="max" if current.max() > high+1e-6 else "neither")
    bar.set_label("Current per 20 pm pixel (nA)")
    bar.set_ticks([v for v in (0, 5, 10, 20, 50, 100, 200, 500) if v <= high])
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    figure.suptitle("Incident probe current at the specimen surface", fontsize=15)
    figure.savefig(output, dpi=175)
    plt.close(figure)


def reproduce(input_folder, output, *, portable=True):
    state, identity, reference, source = load_baseline(input_folder, portable=portable)
    output.mkdir(parents=True, exist_ok=False)
    target = state.sample.upper_surface_z_mm
    start = state.electron_gun.exit_plane_z_mm
    mini = next(v for v in state.lenses if v.key == "mini_condenser")
    nodes = np.unique(np.r_[start, state.condenser_aperture_2.z_mm, component_plane(state, mini),
                            target + np.array(reference["observation_offsets_nm"])*1e-6, target])
    plan = build_unified_plan(state, start, nodes[-1], checkpoint_z_mm=nodes,
                              maximum_step_mm=state.step_mm)
    audit = audit_extended_plan(state, plan, nodes[-1])
    if not audit["passed"] or state.step_mm != reference["maximum_step_mm"]:
        raise ValueError("The replay does not preserve the physical integration contract")
    initial = np.column_stack([source["gun_exit_"+key] for key in PHASE[:4]])
    with numerical_job(3):
        cp, alive, stops, keys = trace_checkpoints(
            state, plan, initial, source["gun_exit_kinetic_energy_ev"], source["gun_exit_flight_time_s"],
            source["gun_exit_alive"], source["gun_blocked_z_mm"], source["gun_blocked_key"])
    if capture_instrument_snapshot(state).digest != identity.digest:
        raise ValueError("Propagation changed the instrument input")
    arrays = dict(source)
    arrays.update({key: getattr(cp, key) for key in PHASE})
    arrays.update(z_mm=cp.z_mm, alive=alive, blocked_z_mm=stops, blocked_key=np.asarray(keys, str))
    _npz(output / "arrays.npz", arrays)
    _json(output / "snapshot.json", identity.to_dict())
    report = {key: reference[key] for key in ("source_current_a", "source_sha256",
        "gun_dependency_signature", "emitted_identity_digest", "emitted_positive_current_points")}
    report.update(status="COMPLETE", input_digest=identity.digest, implementation=identity.implementation,
                  field_audit=audit, gun_dependency_signature=reference["current_gun_dependency_signature"],
                  original_gun_dependency_signature=reference["gun_dependency_signature"],
                  scope="Unchanged fixed-gun-outlet classical column replay")
    _json(output / "report.json", report)
    with numerical_job(3):
        observed = measure(output, spacing_nm=.1)
    _json(output / "crossover.json", observed)
    if observed["status"] != "MEASURED":
        raise ValueError("The repeated two-dimensional waist was not resolved")
    kernel = target_current_kernel(arrays, target, reference["source_current_a"],
                                   pitch_nm=reference["render"]["pitch_nm"],
                                   pixels=reference["render"]["pixels"])
    _npz(output / "current.npz", kernel)
    render_current(kernel, reference["render"], output / "incident-probe.png")
    at = int(np.flatnonzero(cp.z_mm == target).item())
    with np.load(input_folder / "reference-surface.npz", allow_pickle=False) as prior:
        difference = {key: float(np.max(abs(arrays[key][at]-prior[key]))) for key in PHASE}
        membership_exact = np.array_equal(alive[at], prior["alive"])
    with np.load(input_folder / "current.npz", allow_pickle=False) as prior:
        current_exact = np.array_equal(kernel["current_a_yx"], prior["current_a_yx"])
        ids_exact = np.array_equal(kernel["surviving_source_ids"], prior["surviving_source_ids"])
    summary = dict(schema="unchanged-probe-replay-v1", utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        input_digest=identity.digest, original_input_digest=reference["original_input_digest"],
        source_digest=reference["emitted_identity_digest"], portable_archive=portable,
        solver_newline_only_changes=reference["solver_newline_only_changes"],
        field_identity_changed_by_code_receipts=reference["field_identity_changed_by_code_receipts"],
        original_gun_dependency_signature=reference["gun_dependency_signature"],
        current_gun_dependency_signature=reference["current_gun_dependency_signature"],
        current_pixels_exact=current_exact, source_ids_exact=ids_exact, membership_exact=membership_exact,
        max_absolute_surface_differences=difference, at_specimen=observed["at_specimen"],
        at_crossover=observed["at_crossover"], original_specimen=reference["at_specimen"],
        radial_bracket_mm=observed["waist"]["radial_bracket_mm"],
        field_content_preserved=True,
        raw_field_digest_preserved=not reference["field_identity_changed_by_code_receipts"],
        full_source_reemission=False, controls_changed=False,
        source_count=2000, image="incident-probe.png",
        image_sha256=sha256((output / "incident-probe.png").read_bytes()).hexdigest())
    _json(output / "reproduction.json", summary)
    if not all((current_exact, ids_exact, membership_exact)):
        raise ValueError("The original current image or surviving population was not reproduced exactly")
    print(json.dumps({k: summary[k] for k in ("current_pixels_exact", "source_ids_exact",
                                            "max_absolute_surface_differences")}), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
        default=ROOT / "docs/development/probe_stage12_20261101/inputs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-package", action="store_true")
    args = parser.parse_args()
    reproduce(args.inputs.resolve(), args.output.resolve(), portable=not args.original_package)
