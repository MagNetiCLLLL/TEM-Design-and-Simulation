"""Bounded A5 study using fixed-axis, installed scalar optical excitations.

Every execution retains the original 2,000 current-carrying particles. The
archived zero-current optical pencils are separate diagnostics. No current
observable includes a diagnostic ray or a rescaled survivor weight.
"""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from audit_probe_drives import objects, unchanged_interception_geometry, normalized_rank
from measure_crossover_cs import component_plane, target_current_kernel, trace_checkpoints
from measure_probe_crossover import measure, _native_profile
from measure_probe_waist import PHASE, _array_hash, _json, _npz, audit_extended_plan
from probe_channel_response import TERMS, geometric_coefficients
from reproduce_probe_baseline import load_baseline, render_current, read_json
from temsim.cpu_resources import numerical_job
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.unified_field_particle import build_unified_plan
from temsim.physics.unified_field_runtime import get_unified_field_runtime

INPUTS = ROOT / "docs/development/probe_stage12_20261101/inputs"
WORK = ROOT / "outputs/probe-a5-physical-20261101"
ARCHIVE = ROOT / "docs/development/probe_a5_physical_20261101"
# A diagnostic step is not a supply current calibration. All installation
# angles and field-profile parameters remain frozen at the accepted baseline.
DRIVES = (
    ("HP1", "probe_hp1_hexapole", "strength_m3", 1000., "m^-3"),
    ("HP2", "probe_hp2_hexapole", "strength_m3", 1000., "m^-3"),
    ("TL22", "probe_tl22_lens", "percent", .001, "percentage point"),
    ("TL21", "probe_tl21_lens", "percent", .001, "percentage point"),
    ("TL12", "probe_tl12_lens", "percent", .001, "percentage point"),
    ("ADL", "adapter_lens", "percent", .001, "percentage point"),
    ("Objective", "objective_lens", "percent", .00002, "percentage point"),
    ("Mini", "mini_condenser", "percent", .0001, "percentage point"),
)


def apply(state, changes):
    """Only the registered scalar coordinates may change, with finite limits."""
    allowed = {row[0] for row in DRIVES}
    if set(changes) - allowed:
        raise ValueError("Unregistered physical drive")
    parts = objects(state)
    for label, key, attr, step, _ in DRIVES:
        increment = float(changes.get(label, 0.)) * step
        if not np.isfinite(increment):
            raise ValueError("Nonfinite drive increment")
        if not increment:
            continue
        part = parts[key]
        if not part.enabled:
            raise ValueError("Disabled drive")
        value = getattr(part, attr) + increment
        limit = part.maximum_strength_m3 if attr == "strength_m3" else part.max_percent
        if not 0 < value <= limit:
            raise ValueError("Scalar drive exceeds its declared model range")
        setattr(part, attr, value)


def driver_audit():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    state, identity, reference, _ = load_baseline(INPUTS)
    base = get_unified_field_runtime(state)
    parts = {part.key: part for part in state._resolved_assembly.parts}
    providers = {part.source_id: part.potential.digest for part in base.snapshot.contributions}
    z_samples = [component_plane(state, objects(state)[key]) for _, key, *_ in DRIVES]
    z_samples.append(state.objective_lens.lower_field_center_z_mm)
    phi = np.arange(12)*2*np.pi/12
    transverse = np.vstack([radius*np.c_[np.cos(phi), np.sin(phi)] for radius in (10e-6, 40e-6)])
    points = np.vstack([np.c_[transverse, np.full(len(transverse), z*.001)] for z in z_samples])
    source_ids = [key for _, key, *_ in DRIVES]
    field = base.snapshot.sample(points, source_ids=source_ids)["B_t"]
    rows, columns = [], []
    for label, key, attr, step, unit in DRIVES:
        candidate, _, _, _ = load_baseline(INPUTS)
        apply(candidate, {label: 1.})
        runtime = get_unified_field_runtime(candidate)
        now = {part.source_id: part.potential.digest for part in runtime.snapshot.contributions}
        changed = [name for name in providers if providers[name] != now[name]]
        if (providers.keys() != now.keys() or changed != [key]
                or not unchanged_interception_geometry(base.geometry, runtime.geometry)):
            raise ValueError("A scalar excitation changed an undeclared source or opening")
        original, altered = objects(state)[key], objects(candidate)[key]
        columns.append((runtime.snapshot.sample(points, source_ids=source_ids)["B_t"]-field).ravel())
        if attr == "strength_m3" and original.orientation_rad != altered.orientation_rad:
            raise ValueError("A scalar excitation rotated a hexapole")
        if key == "objective_lens":
            for name in ("upper_b0_t", "lower_b0_t", "upper_gaussian", "lower_gaussian"):
                if getattr(original, name) != getattr(altered, name):
                    raise ValueError("The shared objective changed a field-profile parameter")
        rows.append(dict(channel=label, component=key, coordinate=attr, step=step, unit=unit,
                         value=getattr(original, attr), changed_sources=changed,
                         host_status=parts[key].data.get("physical_host_status", "installed model body"),
                         availability="SUPPORTED_SCALAR_MODEL_EXCITATION",
                         winding_current_calibration="UNVERIFIED"))
    result = dict(schema="conservative-scalar-drive-audit-v1", baseline=identity.digest,
        original_baseline=reference["original_input_digest"], drivers=rows,
        sampled_field_rank=normalized_rank(columns),
        frozen=["all hexapole orientations", "HPC", "HPol", "all quadrupoles", "all dipoles",
                "C2/C3", "Mini Condenser mode and polarity", "upper/lower objective shape and relative amplitude"],
        excluded=["independent HP normal/skew", "unverified auxiliary hosts", "upper-only supply", "virtual DP12"],
        qualification="Installed model excitation audit; not OEM electrical or magnetostatic qualification",
        baseline_limitation="The accepted auxiliary fields remain present and frozen, not hardware-certified")
    _json(ARCHIVE / "driver-audit.json", result)
    _npz(ARCHIVE/"driver-fields.npz", dict(points_m=points, field_response_t=np.column_stack(columns),
                                         labels=np.asarray([row[0] for row in DRIVES])))
    return result


def angular_metrics(kernel):
    x, y = kernel["xy_nm"].T
    r, angle = np.hypot(x, y), np.arctan2(y, x)
    current = kernel["particle_current_a"]
    rows = {}
    masks = {"inner_0p1_1nm": (r > .1) & (r <= 1), "outside_1nm": r > 1,
             "all": r >= 0, "displayed": (abs(x) <= 2.56) & (abs(y) <= 2.56)}
    for lo, hi in ((1, 1.5), (1.5, 2), (2, 3), (3, np.inf)):
        masks[f"shell_{lo}_{hi}nm"] = (r > lo) & (r <= hi)
    for name, mask in masks.items():
        flux = float(current[mask].sum())
        sixth = np.sum(current[mask] * np.exp(6j*angle[mask]))
        rows[name] = dict(current_a=flux, sixth_a=[float(sixth.real), float(sixth.imag)],
                          absolute_sixth_a=float(abs(sixth)),
                          normalized_sixth=float(abs(sixth)/flux) if flux else None)
    rows["outer_shell_absolute_sum_a"] = sum(v["absolute_sixth_a"] for k, v in rows.items()
                                                if k.startswith("shell_"))
    return rows


def coefficient_vector(fit):
    values = []
    for name, order, symmetry in TERMS:
        values.extend(np.asarray(fit["terms_m"][name])[:2 if symmetry else 1] * .03**order * 1e12)
    return np.asarray(values)


def run_trial(name, changes, *, step_mm=.05, focus=True, image_focus_tolerance_nm=None):
    if image_focus_tolerance_nm is not None and (
        not focus or not np.isfinite(image_focus_tolerance_nm) or image_focus_tolerance_nm < 0
    ):
        raise ValueError("A focus-gated image requires through-focus measurement and a finite tolerance")
    started = time.monotonic()
    folder, archive = WORK/name, ARCHIVE/name
    folder.mkdir(parents=True, exist_ok=False)
    archive.mkdir(parents=True, exist_ok=False)
    state, baseline, reference, source = load_baseline(INPUTS)
    original_runtime = get_unified_field_runtime(state)
    apply(state, changes)
    state.step_mm = step_mm
    identity = capture_instrument_snapshot(state)
    runtime = get_unified_field_runtime(state)
    if not unchanged_interception_geometry(original_runtime.geometry, runtime.geometry):
        raise ValueError("Physical interception geometry changed")
    target, start = state.sample.upper_surface_z_mm, state.electron_gun.exit_plane_z_mm
    mini = objects(state)["mini_condenser"]
    nodes = np.unique(np.r_[start, state.condenser_aperture_2.z_mm, component_plane(state, mini),
                            target + np.arange(-40, 41)*1e-6, target])
    plan = build_unified_plan(state, start, nodes[-1], checkpoint_z_mm=nodes, maximum_step_mm=step_mm)
    field_audit = audit_extended_plan(state, plan, nodes[-1])
    if not field_audit["passed"]:
        raise ValueError("Unified field ownership failed")
    with np.load(INPUTS / "reference-pencil.npz", allow_pickle=False) as f:
        pupil = f["pupil"].copy()
    phase = np.vstack((np.column_stack([source["gun_exit_"+key] for key in PHASE[:4]]), pupil))
    n = len(source["weights"])
    energy = np.r_[source["gun_exit_kinetic_energy_ev"], np.full(len(pupil), 300000.)]
    times = np.r_[source["gun_exit_flight_time_s"], np.full(len(pupil), 3.13e-9)]
    active = np.r_[source["gun_exit_alive"], np.ones(len(pupil), bool)]
    stops = np.r_[source["gun_blocked_z_mm"], np.full(len(pupil), np.nan)]
    keys = np.r_[source["gun_blocked_key"], np.full(len(pupil), "", dtype=str)]
    print(name, "propagating 2000 original-current particles and archived zero-current diagnostics", flush=True)
    with numerical_job(2):
        cp, alive, stops, keys = trace_checkpoints(state, plan, phase, energy, times, active, stops, keys)
    if capture_instrument_snapshot(state).digest != identity.digest:
        raise ValueError("Propagation changed a drive")
    at = int(np.flatnonzero(cp.z_mm == target).item())
    arrays = dict(source)
    arrays.update({key: getattr(cp, key)[:, :n] for key in PHASE})
    arrays.update(z_mm=cp.z_mm, alive=alive[:, :n], blocked_z_mm=stops[:n],
                  blocked_key=np.asarray(keys[:n], str))
    _npz(folder/"arrays.npz", arrays)
    _json(folder/"snapshot.json", identity.to_dict())
    report = {key: reference[key] for key in ("source_current_a", "source_sha256",
        "gun_dependency_signature", "emitted_identity_digest", "emitted_positive_current_points")}
    report.update(status="COMPLETE", input_digest=identity.digest, implementation=identity.implementation,
                  field_audit=field_audit, controls=changes,
                  gun_dependency_signature=reference["current_gun_dependency_signature"])
    _json(folder/"report.json", report)
    kernel = target_current_kernel(arrays, target, reference["source_current_a"], **{
        key: reference["render"][key] for key in ("pitch_nm", "pixels")})
    _npz(archive/"current.npz", kernel)
    _npz(archive/"surface.npz", dict({key: arrays[key][at] for key in PHASE},
        alive=arrays["alive"][at], source_id=source["source_id"], weights=source["weights"]))
    diagnostic = np.column_stack([getattr(cp, key)[at, n:] for key in PHASE[:4]])
    if not alive[at, n:].all():
        raise ValueError("Optical pencil is clipped; coefficients are not measurable")
    fits = [geometric_coefficients(rays, 192, degree=9) for rays in np.split(diagnostic, 2)]
    _npz(archive/"optical-pencil.npz", dict(rays=diagnostic, pupil=pupil))
    if focus:
        with numerical_job(2):
            observed = measure(folder, spacing_nm=.1)
        _json(folder/"crossover.json", observed)
    else:
        from types import SimpleNamespace
        physical = SimpleNamespace(**{key: arrays[key] for key in (*PHASE, "z_mm")})
        _, rows = _native_profile(physical, arrays["alive"], source["weights"], source["source_id"],
                                  [target], target, target, reference["source_current_a"])
        observed = dict(status="NOT_REQUESTED_FOR_ISOLATED_RESPONSE", at_specimen=rows[0])
    if image_focus_tolerance_nm is not None:
        if (observed["status"] != "MEASURED" or not observed["target_in_2d_bracket"]
                or abs(observed["at_crossover"]["offset_from_surface_nm"]) > image_focus_tolerance_nm):
            raise ValueError(f"Specimen image withheld at {name}: the measured waist is outside the focus gate")
    render_current(kernel, reference["render"], archive/"incident-probe.png")
    summary = dict(schema="fixed-axis-a5-trial-v1", name=name, utc=datetime.now(timezone.utc).isoformat(),
        baseline_input=baseline.digest, input_digest=identity.digest, controls=changes,
        source_digest=reference["emitted_identity_digest"], source_hashes=reference["source_sha256"],
        original_positive_particles=n, original_current_a=reference["source_current_a"],
        zero_current_diagnostics=len(pupil), maximum_step_mm=step_mm,
        field_audit=field_audit, fits=fits, tails=angular_metrics(kernel),
        crossover_status=observed["status"], at_specimen=observed["at_specimen"],
        at_crossover=observed.get("at_crossover"), waist=observed.get("waist"),
        separation=observed.get("separation"), d95_best=observed.get("d95_best"),
        rms_best=observed.get("rms_best"), core_best=observed.get("core_best"),
        elapsed_s=time.monotonic()-started,
        image_sha256=sha256((archive/"incident-probe.png").read_bytes()).hexdigest())
    _json(archive/"result.json", summary)
    _json(archive/"input-recipe.json", dict(
        baseline_package="docs/development/probe_stage12_20261101/inputs/portable-probe.temwp",
        baseline_input=baseline.digest, expected_input=identity.digest,
        changes=changes, maximum_step_mm=step_mm,
        restoration="load_baseline followed by the registered scalar apply function",
        full_snapshot_local=str((folder/"snapshot.json").relative_to(ROOT))))
    if focus:
        _json(archive/"through-focus.json", observed)
    print(name, "COMPLETE", round(summary["elapsed_s"], 1), "s; A5 mm", round(np.linalg.norm(fits[0]["terms_m"]["A5"])*1000, 6),
          "D95 nm", round(observed["at_specimen"]["d95_nm"], 6), flush=True)
    return summary


def _signed_trial(job):
    name, changes = job
    return run_trial(name, changes, focus=False)["name"]


def calibrate(half=False):
    factor, suffix = (.5, "half") if half else (1., "full")
    jobs = [(f"response-{name}-{suffix}-{sign}", {name: value*factor})
            for name, *_ in DRIVES for sign, value in (("plus", 1.), ("minus", -1.))]
    jobs = [job for job in jobs if not (ARCHIVE/job[0]/"result.json").exists()]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_signed_trial, job) for job in jobs]
        for future in as_completed(futures):
            print("Saved", future.result(), flush=True)


def response_rows(record):
    return np.r_[coefficient_vector(record["fits"][0]), record["at_specimen"]["alpha95_mrad"]]


def protected_response(matrix, derivative_error, drive_ranges):
    """Resolve two-axis A5 reach after exact first-order primary protection.

    Coefficients are pm at 30 mrad; the final row is measured alpha95 in mrad.
    Compare the projected finite-difference error with each singular value.
    Current is a separate discrete membership constraint, not a fake smooth row.
    """
    matrix = np.asarray(matrix, float)
    error = np.asarray(derivative_error, float)
    scaled = matrix * np.asarray(drive_ranges)[None, :]
    protected = [0, 3, 4, 7, 25]  # C1, both A2 components, C3, alpha95
    g = scaled[protected] / np.array([2., 5., 5., 5., .03])[:, None]
    _, singular, vh = np.linalg.svd(g, full_matrices=True)
    rank = int(np.count_nonzero(singular > singular[0]*1e-9)) if len(singular) else 0
    null = vh[rank:].T
    target = scaled[19:21] @ null
    values = np.linalg.svd(target, compute_uv=False)
    error_bound = float(np.linalg.norm((error*np.asarray(drive_ranges)[None, :])[19:21]@null, 2)) if null.size else 0.
    threshold = max(1e-6, 10*error_bound, (values[0]*1e-8 if len(values) else 0.))
    return dict(protected_rows=protected, protected_rank=rank, null_dimension=null.shape[1],
        target_singular_values_pm=values.tolist(), target_error_bound_pm=error_bound,
        resolved_target_rank=int(np.count_nonzero(values > threshold)), target_rank_threshold_pm=threshold,
        normalized_null_basis=null.tolist(), drive_ranges=list(drive_ranges),
        full_step_primary_leakage=(error[protected] * np.asarray(drive_ranges)[None, :])@null)


def analyze_responses():
    base = read_json(ARCHIVE/"baseline/result.json")
    zero = response_rows(base)
    columns = {"full": [], "half": []}
    even, records = {}, []
    for label, *_ in DRIVES:
        for size, h in (("full", 1.), ("half", .5)):
            plus = read_json(ARCHIVE/f"response-{label}-{size}-plus/result.json")
            minus = read_json(ARCHIVE/f"response-{label}-{size}-minus/result.json")
            a, b = response_rows(plus), response_rows(minus)
            columns[size].append((a-b)/(2*h))
            even[label+"_"+size] = ((a+b)/2-zero).tolist()
            records.extend([plus, minus])
    full, half = (np.column_stack(columns[key]) for key in ("full", "half"))
    error = full-half
    rank = protected_response(half, error, [40. if name == "ADL" else 100. for name, *_ in DRIVES])
    rank["full_step_primary_leakage"] = rank["full_step_primary_leakage"].tolist()
    out = dict(schema="protected-physical-a5-response-v1", baseline=base["input_digest"],
        source=base["source_digest"], drives=[row[0] for row in DRIVES],
        row_names=[name+axis for name, _, m in TERMS for axis in (("_x", "_y") if m else ("",))]+["alpha95_mrad"],
        baseline_vector=zero.tolist(), full=full.tolist(), half=half.tolist(),
        derivative_difference=error.tolist(), even_response=even, protected_response=rank,
        current_preserved_in_all_probes=all(r["at_specimen"]["surviving_rays"] == 2000 and
            abs(r["at_specimen"]["current_a"]-1e-5) < 1e-17 for r in records),
        scope="Local scalar model response; nonlinear candidate and current-tail validation still required")
    _json(ARCHIVE/"protected-response.json", out)
    print(json.dumps(rank, indent=2), flush=True)
    return out


def propose(name, span=100., tail_weight=0., protection="all"):
    """Bounded linear proposal. Only a fresh physical replay can accept it."""
    from scipy.optimize import minimize
    data = read_json(ARCHIVE/"protected-response.json")
    base, jac = np.asarray(data["baseline_vector"]), np.asarray(data["half"])
    ranges = np.array([min(span, 40.) if name == "ADL" else span for name, *_ in DRIVES])
    matrix = jac*ranges[None, :]
    with np.load(ARCHIVE/"baseline/current.npz", allow_pickle=False) as f:
        positions = f["xy_nm"].copy()
        current = f["particle_current_a"].copy()
        ids = f["surviving_source_ids"].copy()
    derivatives = []
    for label, *_ in DRIVES:
        pair = []
        for sign in ("plus", "minus"):
            with np.load(ARCHIVE/f"response-{label}-half-{sign}/current.npz", allow_pickle=False) as f:
                if not np.array_equal(f["surviving_source_ids"], ids) or not np.array_equal(f["particle_current_a"], current):
                    raise ValueError("Linear position prediction requires the identical surviving cohort")
                pair.append(f["xy_nm"].copy())
        derivatives.append(pair[0]-pair[1])
    position_jac = np.stack(derivatives, axis=-1)*ranges[None, None, :]
    def smooth_tail(xy):
        z = xy[:, 0]+1j*xy[:, 1]
        radius = abs(z)
        outer = 1/(1+np.exp(np.clip((1.-radius)/.06, -60, 60)))
        sixth = np.sum(current*outer*(z/np.maximum(radius, 1e-30))**6)
        return float(abs(sixth))
    initial_tail = smooth_tail(positions)
    def profile_proxy(xy):
        radius = np.linalg.norm(xy, axis=1)
        centre = np.average(xy, weights=current, axis=0)
        rms = np.sqrt(np.average(np.sum((xy-centre)**2, axis=1), weights=current))
        core = np.sum(current/(1+np.exp(np.clip((radius-.05)/.004, -60, 60))))
        outer = np.sum(current/(1+np.exp(np.clip((1.-radius)/.06, -60, 60))))
        return np.array([rms, core, outer])
    initial_profile = profile_proxy(positions)
    targets = [19, 20]
    groups = []
    i = 0
    for label, _, m in TERMS:
        count = 2 if m else 1
        if label != "A5" and not (protection == "requested" and label in ("D4", "C5")):
            groups.append((list(range(i, i+count)), 2. if label == "C1" else 5.))
        i += count
    def objective(x):
        v = base[targets] + matrix[targets]@x
        proxy_tail = smooth_tail(positions+position_jac@x)
        return float(v@v/(base[targets]@base[targets]) + tail_weight*(proxy_tail/initial_tail)**2 + 1e-8*(x@x))
    def guards(x):
        delta = matrix@x
        alpha = base[-1]+delta[-1]
        result = np.r_[[1.-np.linalg.norm(delta[ix])/tol for ix, tol in groups],
                        (alpha-29.7)/.03, (30.3-alpha)/.03]
        if protection == "requested":
            profile = profile_proxy(positions+position_jac@x)/initial_profile
            result = np.r_[result, 1.-profile[0], profile[1]-1., 1.-profile[2]]
        return result
    result = minimize(objective, np.zeros(len(DRIVES)), method="SLSQP", bounds=[(-1., 1.)]*len(DRIVES),
        constraints=[dict(type="ineq", fun=guards)], options=dict(ftol=1e-12, maxiter=2000))
    v = result.x*ranges
    pred = base+jac@v
    controls = dict(zip([row[0] for row in DRIVES], v.tolist()))
    out = dict(status="LINEAR_PROPOSAL_ONLY", optimizer_success=bool(result.success), message=str(result.message),
        controls=controls, predicted_vector=pred.tolist(), predicted_a5_fraction=float(np.linalg.norm(pred[19:21])/np.linalg.norm(base[19:21])),
        minimum_guard=float(guards(result.x).min()), span_steps=span, tail_weight=tail_weight,
        protection=protection, predicted_profile_fraction=(profile_proxy(positions+position_jac@result.x)/initial_profile).tolist(),
        predicted_smooth_outer_sixth_fraction=smooth_tail(positions+position_jac@result.x)/initial_tail,
        proxy_scope="First-order position prediction with 0.06 nm radial gate smoothing; never rendered or accepted as physical current")
    _json(ARCHIVE/(name+"-proposal.json"), out)
    _json(ARCHIVE/(name+"-controls.json"), controls)
    print(json.dumps(out, indent=2), flush=True)
    return out


def gallery():
    """Archive all native images and publish a bounded diagnostic contact sheet."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    records = []
    for file in sorted(ARCHIVE.glob("*/result.json")):
        rec = read_json(file)
        folder = file.parent
        duplicate = folder/"snapshot.json"
        if duplicate.exists():
            local = WORK/folder.name/"snapshot.json"
            if duplicate.read_bytes() != local.read_bytes():
                raise ValueError("An archived snapshot differs from its local original")
            _json(folder/"input-recipe.json", dict(
                baseline_package="docs/development/probe_stage12_20261101/inputs/portable-probe.temwp",
                baseline_input=rec["baseline_input"], expected_input=rec["input_digest"],
                changes=rec["controls"], maximum_step_mm=rec["maximum_step_mm"],
                restoration="load_baseline followed by the registered scalar apply function",
                full_snapshot_local=str(local.relative_to(ROOT))))
            duplicate.unlink()
        if sha256((folder/"incident-probe.png").read_bytes()).hexdigest() != rec["image_sha256"]:
            raise ValueError("Archived current image changed")
        records.append(dict(name=rec["name"], input_digest=rec["input_digest"],
            source_digest=rec["source_digest"], image=f"{folder.name}/incident-probe.png",
            image_sha256=rec["image_sha256"], controls=rec["controls"],
            metrics=f"{folder.name}/result.json", current=f"{folder.name}/current.npz"))
    for size in ("full", "half"):
        names = [f"response-{label}-{size}-{sign}" for label, *_ in DRIVES for sign in ("plus", "minus")]
        names = [name for name in names if (ARCHIVE/name/"result.json").exists()]
        sheet = ARCHIVE/f"response-{size}-gallery-{len(names):02d}.png"
        if not names or sheet.exists():
            continue
        count = (len(names)+1)//2
        figure, axes = plt.subplots(count, 2, figsize=(8., 3.15*count), layout="constrained", squeeze=False)
        for axis, name in zip(axes.flat, names):
            with np.load(ARCHIVE/name/"current.npz", allow_pickle=False) as kernel:
                axis.imshow(kernel["current_a_yx"]*1e9, origin="lower", extent=[-2.56,2.56,-2.56,2.56],
                    cmap="magma", norm=SymLogNorm(linthresh=5., vmin=0., vmax=310.), interpolation="nearest")
            axis.set_title(name.removeprefix("response-")+" (uncompensated)", fontsize=9)
            axis.set_xlabel("X (nm)")
            axis.set_ylabel("Y (nm)")
        figure.suptitle("Specimen current: diagnostic scalar perturbations\nSame 2000 particles; fixed absolute color scale", fontsize=13)
        for axis in list(axes.flat)[len(names):]:
            axis.set_visible(False)
        figure.savefig(sheet, dpi=120)
        plt.close(figure)
    _json(ARCHIVE/"image-history.json", dict(schema="probe-image-history-v1", trials=records))


def assess_trial(name, native_focus=False):
    base = read_json(ARCHIVE/"baseline/result.json")
    trial = read_json(ARCHIVE/name/"result.json")
    reference, actual = base["at_specimen"], trial["at_specimen"]
    deltas = {}
    for label, order, _ in TERMS:
        deltas[label] = float(np.linalg.norm(np.asarray(trial["fits"][0]["terms_m"][label]) -
                              base["fits"][0]["terms_m"][label])*.03**order*1e12)
    ratios = [float(np.linalg.norm(trial["fits"][i]["terms_m"]["A5"])/
                    np.linalg.norm(base["fits"][i]["terms_m"]["A5"])) for i in (0, 1)]
    exempt = ("A5", "C1") if native_focus else ("A5",)
    protected_max = max(value for label, value in deltas.items() if label not in exempt)
    before_tail, tail = base["tails"], trial["tails"]
    gates = dict(
        fixed_population=trial["original_positive_particles"] == 2000 and trial["source_hashes"] == base["source_hashes"],
        original_transmission=actual["surviving_rays"] == 2000 and abs(actual["current_a"]-reference["current_a"]) < 1e-17,
        angle=29.7 <= actual["alpha95_mrad"] <= 30.3,
        a2_protected=deltas["A2"] <= 5., c3_protected=deltas["C3"] <= 5.,
        other_aberrations_protected=protected_max <= 5.,
        a5_improvement=max(ratios) <= .99,
        outer_sixfold_improvement=tail["outside_1nm"]["absolute_sixth_a"] <= .99*before_tail["outside_1nm"]["absolute_sixth_a"],
        outer_current=tail["outside_1nm"]["current_a"] <= before_tail["outside_1nm"]["current_a"]+1e-17,
        outer_shells=tail["outer_shell_absolute_sum_a"] <= before_tail["outer_shell_absolute_sum_a"]+1e-17,
        d95=actual["d95_nm"] <= reference["d95_nm"], rms=actual["rms_radius_nm"] <= reference["rms_radius_nm"],
        core_current=actual["core_50pm_a"] >= reference["core_50pm_a"]-5e-9-1e-17,
        native_focus=trial["crossover_status"] == "MEASURED" and abs(trial["at_crossover"]["offset_from_surface_nm"]) <= .25)
    if native_focus:
        gates["inner_threefold_preserved"] = (actual["threefold_tail"]["harmonic_current_a"] <=
            reference["threefold_tail"]["harmonic_current_a"]+5e-9)
    result = dict(name=name, gates=gates, all_scalar_and_probe_gates=all(gates.values()),
        focus_compensation_scope="C1 may change through explicit physical drives" if native_focus else "initial strict all-coefficient pilot",
        a5_fraction=ratios, collateral_vector_pm=deltas, max_collateral_pm=protected_max,
        final_qualification="PENDING_INTEGRATION_AND_TOPOLOGY", measured=actual,
        waist_offset_nm=trial["at_crossover"]["offset_from_surface_nm"] if trial["at_crossover"] else None)
    _json(ARCHIVE/name/"gates.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def topology(name):
    from measure_probe_crossover import _continue, _bounded_grid
    from temsim.optics.waist_evidence import finite_interval
    rec = read_json(ARCHIVE/name/"result.json")
    state, _, _, source = load_baseline(INPUTS)
    apply(state, rec["controls"])
    state.step_mm = rec["maximum_step_mm"]
    if capture_instrument_snapshot(state).digest != rec["input_digest"]:
        raise ValueError("The optical input recipe is not exact")
    desired = [("condenser_lens_2", "condenser_lens_3", 1),
               ("condenser_lens_3", "adapter_lens", 0),
               ("probe_tl22_lens", "probe_tl21_lens", 1), ("ac_upper", "ac_lower", 1)]
    lenses = {lens.key: component_plane(state, lens) for lens in state.lenses}
    lenses["ac_upper"] = component_plane(state, state.ac_deflector, state.ac_deflector.upper_z_mm-state.ac_deflector.z_mm)
    lenses["ac_lower"] = component_plane(state, state.ac_deflector, state.ac_deflector.lower_z_mm-state.ac_deflector.z_mm)
    start = state.electron_gun.exit_plane_z_mm
    nodes = np.unique(np.concatenate(([start], *[_bounded_grid(lenses[a], lenses[b], .5) for a, b, _ in desired])))
    phase = np.column_stack([source["gun_exit_"+key] for key in PHASE[:4]])
    with numerical_job(3):
        cp, live, _, _ = _continue(state, start, phase, source["gun_exit_kinetic_energy_ev"],
            source["gun_exit_flight_time_s"], source["gun_exit_alive"], source["gun_blocked_z_mm"], source["gun_blocked_key"], nodes)
    records = {}
    for a, b, count in desired:
        result = finite_interval(cp, live, source["weights"], lenses[a], lenses[b],
                                 maximum_spacing_mm=.500000001, source_ids=source["source_id"])
        records[a+" to "+b] = dict(expected=count, actual=len(result["regions"]),
            status=result["status"], unresolved=result["unresolved"],
            waist_z_mm=[row["z_mm"] for row in result["regions"]])
    passed = all(row["expected"] == row["actual"] and not row["unresolved"] for row in records.values())
    out = dict(input_digest=rec["input_digest"], source=rec["source_digest"], original_particles=2000,
               intervals=records, passed=passed)
    _json(ARCHIVE/name/"topology.json", out)
    print(json.dumps(out, indent=2), flush=True)
    return out


def verify_archive():
    """Read-only input-recipe and original-current checks for completed trials."""
    from temsim.instrument_snapshot import decode_instrument
    _, baseline, reference, source = load_baseline(INPUTS)
    reports = []
    currents = reference["source_current_a"]*source["weights"]/source["weights"].sum()
    for file in sorted(ARCHIVE.glob("*/result.json")):
        rec = read_json(file)
        state = decode_instrument(baseline.graph)
        apply(state, rec["controls"])
        state.step_mm = rec["maximum_step_mm"]
        if capture_instrument_snapshot(state).digest != rec["input_digest"]:
            raise ValueError(f"Input recipe mismatch: {rec['name']}")
        if rec["source_hashes"] != reference["source_sha256"] or rec["original_positive_particles"] != 2000:
            raise ValueError("Original source lineage changed")
        with np.load(file.parent/"surface.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["source_id"], source["source_id"])
            np.testing.assert_array_equal(f["weights"], source["weights"])
            active = f["alive"]
        with np.load(file.parent/"current.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["surviving_source_ids"], source["source_id"][active])
            np.testing.assert_array_equal(f["particle_current_a"], currents[active])
        if sha256((file.parent/"incident-probe.png").read_bytes()).hexdigest() != rec["image_sha256"]:
            raise ValueError("Current image checksum changed")
        reports.append(dict(trial=rec["name"], input_recipe_exact=True, current_weights_exact=True,
                            original_population_exact=True, input_digest=rec["input_digest"]))
    with np.load(ARCHIVE/"baseline/surface.npz", allow_pickle=False) as current, np.load(INPUTS/"reference-surface.npz", allow_pickle=False) as original:
        equality = {key: bool(np.array_equal(current[key], original[key])) for key in original.files}
    if not all(equality.values()):
        raise ValueError("The unmodified baseline was not reproduced exactly")
    result = dict(schema="fixed-source-recipe-verification-v1", baseline_surface_exact=equality,
                  completed_trials=len(reports), trials=reports)
    _json(ARCHIVE/"archive-verification.json", result)
    print("Verified", len(reports), "completed trial recipes and original current arrays", flush=True)
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "run", "calibrate", "analyze", "propose", "gallery", "assess", "topology", "verify"))
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--step-mm", type=float, default=.05)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--span", type=float, default=100.)
    parser.add_argument("--tail-weight", type=float, default=0.)
    parser.add_argument("--protection", choices=("all", "requested"), default="all")
    parser.add_argument("--native-focus", action="store_true")
    args = parser.parse_args()
    if args.action == "audit":
        driver_audit()
    elif args.action == "calibrate":
        calibrate(args.half)
    elif args.action == "analyze":
        analyze_responses()
    elif args.action == "propose":
        propose(args.name, args.span, args.tail_weight, args.protection)
    elif args.action == "gallery":
        gallery()
    elif args.action == "assess":
        assess_trial(args.name, args.native_focus)
    elif args.action == "topology":
        topology(args.name)
    elif args.action == "verify":
        verify_archive()
    else:
        controls = read_json(args.controls) if args.controls else {}
        run_trial(args.name, controls, step_mm=args.step_mm)


if __name__ == "__main__":
    main()
