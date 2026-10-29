"""Propose physical controls using one consistent native-D95 waist definition.

The seven-plane search uses D95 (optionally logistic-smoothed), convergence
angle and an explicit small control regularizer. It has no RMS, per-particle
position or covariance-focus objective. Seven observed planes cannot qualify a
continuous waist: independently measure the native profile over +/-100 nm,
refine its competing minima and check integration/source sampling and topology.
All files are new, immutable trial evidence. No physical source is reweighted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from solve_joint_probe import (absolute_jacobian, atomic_json, atomic_npz,
    control_space, finite, physical_values, read_json, seed_controls,
    smoothed_current_d95_nm, snapshot_changes)

SCHEMA = "temsim-d95-waist-search-v1"
OFFSETS_NM = (-20., -5., -1., 0., 1., 5., 20.)
DEFAULT_CONTROLS = ("c1", "c2", "c3", "adl", "tl22", "tl21", "tl12", "mini", "objective",
    "hp2", "hp1", "hp1_angle", "hpc", "hpc_angle", "hpol", "hpol_angle", "stig_x", "stig_y",
    "qph2", "qpc", "qph1", "qpol")
VALIDATION_REQUIRED = ["Independent native D95 profile over +/-100 nm and actual-plane refinement",
    "Integration-step convergence", "Original source sampling convergence",
    "Condenser/corrector crossover topology and absence of an additional pre-specimen waist"]


class SearchStopped(RuntimeError):
    pass


class NumericalSupportRejected(ValueError):
    pass


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(arrays):
    """Pin emission, IDs and absolute weights, allowing physical exit changes."""
    keys = sorted(key for key in arrays if key.startswith("emission_"))
    if not keys:
        raise ValueError("Original emission reference arrays are required")
    digest = hashlib.sha256()
    for key in ("source_id", "weights", *keys):
        value = np.asarray(arrays[key])
        if value.dtype.hasobject:
            raise ValueError("Original source arrays must not contain Python objects")
        digest.update(key.encode())
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Objective:
    d95_scale_nm: float = .2
    target_mrad: float = 30.
    angle_scale_mrad: float = .03
    profile_scale_nm: float = .01
    d95_smoothing_nm: float = .05
    control_regularizer: float = 1e-4
    max_d95_nm: float | None = None

    def validate(self):
        for key, value in asdict(self).items():
            if key == "max_d95_nm" and value is None:
                continue
            number = finite(value, key)
            if number < 0 or (key not in {"d95_smoothing_nm", "control_regularizer"} and number == 0):
                raise ValueError("Objective scales/targets must be positive; smoothing and regularization may be zero")
        return self


def objective_residual(observation, normalized, objective):
    """Return fixed-width residuals; every plane uses its full actual current.

    D_tau is an optimization surrogate, not a conservative empirical quantile.
    Its zero-radius floor is 2*tau*log(19). Native D95 remains the measured
    diameter; no seven-point result is a continuous-waist qualification.
    """
    from measure_probe_waist import analyze_scan
    objective.validate()
    q = np.asarray(normalized, dtype=float)
    if q.ndim != 1 or not len(q) or not np.isfinite(q).all():
        raise ValueError("A finite nonempty normalized control vector is required")
    arrays = observation.arrays
    surface_z = finite(observation.report["surface"]["z_mm"], "Exact surface coordinate")
    planes = surface_z + np.asarray(OFFSETS_NM)*1e-6
    try:
        measured = analyze_scan(arrays, planes, surface_z_mm=surface_z,
            source_current_a=observation.report["source_current_a"])
    except ValueError as error:
        raise NumericalSupportRejected(str(error)) from error
    profile = measured["planes"]
    if any(row["surviving_rays"] < 16 or row["effective_rays"] < 16*(1-32*np.finfo(float).eps)
            for row in profile):
        raise NumericalSupportRejected("Every objective plane requires at least 16 effective actual particles")
    diameters = np.array([row["d95_nm"] for row in profile])
    guiding = diameters.copy()
    if objective.d95_smoothing_nm > 0:
        for index, plane in enumerate(planes):
            row = int(np.flatnonzero(arrays["z_mm"] == plane)[0])
            guiding[index] = smoothed_current_d95_nm(arrays["x_m"][row], arrays["y_m"][row],
                arrays["alive"][row], arrays["weights"], objective.d95_smoothing_nm)
    center = OFFSETS_NM.index(0.)
    others = np.arange(len(planes)) != center
    width = guiding[center]/objective.d95_scale_nm
    angle = (profile[center]["alpha95_mrad"]-objective.target_mrad)/objective.angle_scale_mrad
    contrast = np.maximum(guiding[center]-guiding[others], 0.)/objective.profile_scale_nm
    regularizer = math.sqrt(objective.control_regularizer/len(q))*q
    residual = np.r_[width, angle, contrast, regularizer]
    if not np.isfinite(residual).all():
        raise NumericalSupportRejected("Nonfinite D95 objective")
    # These are screen conditions at seven exact samples only. A flat or
    # population-changing profile is not promoted as an observed optical waist.
    checks = {"native_alpha95": abs(profile[center]["alpha95_mrad"]/30.-1.) <= .01,
        "strict_sampled_surface_minimum": bool(np.all(diameters[center] < diameters[others])),
        "constant_profile_population": measured["positive_current_membership_stable"],
        "numerical_support": True}
    if objective.max_d95_nm is not None:
        checks["explicit_native_d95_cap"] = bool(0 <= diameters[center] < objective.max_d95_nm)
    screen = dict(status="NATIVE_SEVEN_POINT_SCREENED_PROPOSAL" if all(checks.values()) else "NOT_SCREENED",
        passed=all(checks.values()), checks=checks,
        qualification="UNQUALIFIED_PROPOSAL", maximum_d95_nm=objective.max_d95_nm,
        size_cap_applied=objective.max_d95_nm is not None, independent_validation_required=VALIDATION_REQUIRED,
        scope="Seven sampled native values only; neither an interpolated nor a continuous waist certificate")
    terms = dict(width=float(width*width), angle=float(angle*angle),
        profile=float(contrast@contrast), control_regularizer=float(regularizer@regularizer))
    diagnostic = dict(native_profile=profile, guiding_d95_nm=guiding.tolist(),
        d95_smoothing_nm=objective.d95_smoothing_nm,
        residuals=dict(width=float(width), angle=float(angle), profile=contrast.tolist(),
            control_regularizer=regularizer.tolist()), squared_cost_terms=terms,
        squared_cost=float(residual@residual), native_screen=screen,
        width_objective="Direct D95/scale minimization; a supplied native size cap affects screening only",
        profile_scale_role="Algorithmic contrast weight in nm; not a physical acceptance tolerance",
        regularizer_role="Small normalized control displacement from this declared seed; not a physical measurement",
        focus_definition="Native current D95(z) minimum; RMS and D90 are diagnostics only")
    return residual, diagnostic


def load_warm_candidate(path, *, rays=None, step=None):
    """Preserve the full graph; only explicitly declared count/step may change."""
    from validate_joint_probe import load_candidate
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state, provenance = load_candidate(path)
    if provenance["restoration_kind"] != "complete_snapshot":
        raise ValueError("A complete candidate inputs snapshot is required; controls-only seeds are not admitted")
    before = capture_instrument_snapshot(state)
    before_data = before.to_dict()
    allowed = set()
    if rays is not None:
        if type(rays) is not int or rays < 16:
            raise ValueError("Explicit source count must be an integer of at least 16")
        emitter = state.electron_gun.emitter
        model = type(emitter).__module__+":"+type(emitter).__name__
        nodes = [i for i, node in enumerate(before_data["graph"]["nodes"])
                 if node.get("type") == model and "ray_count" in node.get("attributes", {})]
        if len(nodes) != 1:
            raise ValueError("The original emitter count is not uniquely represented in the snapshot")
        allowed.add(f"/graph/nodes/{nodes[0]}/attributes/ray_count")
        emitter.ray_count = rays
    if step is not None:
        if not 0 < finite(step, "Explicit integration step") <= .2:
            raise ValueError("Explicit integration step must lie in (0, 0.2] mm")
        root = before_data["graph"]["root"]["ref"]
        allowed.add(f"/graph/nodes/{root}/attributes/step_mm")
        state.step_mm = step
    after = capture_instrument_snapshot(state)
    changes = snapshot_changes({k: v for k, v in before_data.items() if k != "digest"},
        {k: v for k, v in after.to_dict().items() if k != "digest"})
    if any(row["path"] not in allowed for row in changes):
        raise ValueError("Explicit count/step overrides changed undeclared instrument inputs")
    return state, dict(candidate=provenance, original_input_digest=before.digest,
        search_input_digest=after.digest, explicit_overrides=dict(rays=rays, step_mm=step), changes=changes)


class Search:
    """Bounded immutable proposals, with separate smooth and native screening."""
    def __init__(self, evaluator, controls, output, *, objective=Objective(),
                 max_evaluations=80, time_budget=7200., jacobian_step=1e-6, cancel_file=None):
        if type(max_evaluations) is not int or max_evaluations < 1:
            raise ValueError("Actual-evaluation budget must be a positive integer")
        if finite(time_budget, "Time budget") <= 0 or finite(jacobian_step, "Jacobian step") <= 0:
            raise ValueError("Time budget and Jacobian step must be positive")
        objective.validate()
        self.controls = tuple(controls)
        if not self.controls:
            raise ValueError("At least one declared physical control is required")
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/"trials").mkdir()
        self.evaluator, self.objective = evaluator, objective
        self.maximum, self.budget, self.jacobian_step = max_evaluations, time_budget, jacobian_step
        self.cancel_file = None if cancel_file is None else Path(cancel_file)
        self.started, self.evaluations = perf_counter(), 0
        self.best = self.native_best = None
        self.best_cost, self.native_best_d95 = math.inf, math.inf
        self.source = None
        self.cache = {}
        self.evaluator.cancelled = self.cancelled
        scripts = ("solve_probe_waist.py", "measure_probe_waist.py", "solve_joint_probe.py", "audit_joint_probe_fields.py")
        self.document = dict(schema=SCHEMA, status="RUNNING", qualification="UNQUALIFIED_PROPOSAL",
            controls=[asdict(c) for c in self.controls], objective=asdict(objective),
            profile_offsets_nm=list(OFFSETS_NM), objective_dimension=8+len(self.controls),
            max_actual_evaluations=max_evaluations, time_budget_s=time_budget,
            absolute_normalized_jacobian_step=jacobian_step,
            driver_sha256={name: file_hash(ROOT/"scripts"/name) for name in scripts},
            independent_validation_required=VALIDATION_REQUIRED,
            cancellation="Cooperative physical executor checks; no hard process deadline",
            trace_count_scope="Profile attempts include failed/cancelled evaluations; batch/gun counts cover completed receipts only",
            population_policy="Real clipping allowed; original emission/IDs/absolute weights retained; no seed-survivor constraint",
            trials=[])
        # Keep the exact search driver alongside all hashes; do not change it
        # after an output directory becomes an execution record.
        (self.output/"driver-source.py").write_bytes(Path(__file__).read_bytes())
        self.save()

    def cancelled(self):
        return perf_counter()-self.started >= self.budget or bool(self.cancel_file and self.cancel_file.exists())

    def save(self):
        self.document.update(actual_evaluations=self.evaluations, elapsed_s=perf_counter()-self.started,
            smooth_best=self.best, native_screened_best=self.native_best)
        self.document["completed_column_trace_batches"] = sum(row.get("completed_column_trace_batches", 0)
            for row in self.document["trials"])
        self.document["completed_gun_traces"] = sum(row.get("completed_gun_traces", 0)
            for row in self.document["trials"])
        atomic_json(self.output/"run.json", self.document)

    def _persist(self, observation, folder, row):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        snapshot = capture_instrument_snapshot(observation.state)
        if (observation.report.get("input_digest") != snapshot.digest
                or observation.report.get("implementation") != snapshot.to_dict()["implementation"]):
            raise ValueError("Trial receipt does not match the complete unchanged instrument inputs")
        atomic_json(folder/"state.json", observation.state.to_dict())
        atomic_json(folder/"inputs.json", snapshot.to_dict())
        atomic_npz(folder/"particles.npz", observation.arrays)
        atomic_json(folder/"candidate.json", dict(schema=SCHEMA, trial=row["trial"],
            qualification="UNQUALIFIED_PROPOSAL", **seed_controls(observation.state),
            observation=observation.report, objective_diagnostics=row.get("objective_diagnostics")))
        names = ("state.json", "inputs.json", "particles.npz", "candidate.json")
        return dict(schema=SCHEMA, directory=folder.relative_to(self.output).as_posix(),
            trial=row["trial"], files_sha256={name: file_hash(folder/name) for name in names},
            qualification="UNQUALIFIED_PROPOSAL")

    def residual(self, normalized):
        if self.cancelled():
            raise SearchStopped("Cooperative cancellation or wall-time budget reached")
        q = np.asarray(normalized, dtype=float)
        values = physical_values(self.controls, q)
        key = values.tobytes()
        if key in self.cache:
            return self.cache[key].copy()
        if self.evaluations >= self.maximum:
            raise SearchStopped("Actual evaluation budget reached, including finite differences")
        self.evaluations += 1
        number, started = self.evaluations, perf_counter()
        folder = self.output/"trials"/f"trial-{number:06d}"
        folder.mkdir()
        row = dict(trial=number, status="RUNNING", normalized=q.tolist(),
            values=dict(zip((c.name for c in self.controls), values.tolist())))
        atomic_json(folder/"trial.json", row)
        self.document["active_trial"] = number
        self.save()
        try:
            observation = self.evaluator.evaluate(values)
            row["observation"] = observation.report
            row["completed_column_trace_batches"] = observation.report.get("actual_trace_batches", 0)
            row["completed_gun_traces"] = int(observation.report.get("gun_reused") is False)
            identity = source_identity(observation.arrays)
            if self.source is None:
                self.source = identity
            elif identity != self.source:
                raise ValueError("Original emission, particle IDs or absolute source weights changed")
            row["original_source_sha256"] = identity
            try:
                result, diagnostic = objective_residual(observation, q, self.objective)
            except NumericalSupportRejected as error:
                row.update(status="REJECTED_NUMERICAL_SUPPORT", error=str(error))
                result = np.full(8+len(self.controls), 1e6)
                self._persist(observation, folder, row)
            else:
                row.update(status="COMPLETE", objective_diagnostics=diagnostic)
                pointer = self._persist(observation, folder, row)
                cost = diagnostic["squared_cost"]
                if cost < self.best_cost:
                    self.best, self.best_cost = {**pointer, "squared_cost": cost}, cost
                    atomic_json(self.output/"best.json", self.best)
                diameter = diagnostic["native_profile"][OFFSETS_NM.index(0.)]["d95_nm"]
                if diagnostic["native_screen"]["passed"] and diameter < self.native_best_d95:
                    self.native_best, self.native_best_d95 = {**pointer, "native_d95_nm": diameter,
                        "scope": "Seven-point screened proposal; independent native waist validation still required"}, diameter
                    atomic_json(self.output/"native-proposal-best.json", self.native_best)
            self.cache[key] = result.copy()
            return result
        except BaseException as error:
            row.update(status="CANCELLED" if self.cancelled() or isinstance(error,
                (KeyboardInterrupt, InterruptedError, SearchStopped)) else "FAILED",
                error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["elapsed_s"] = perf_counter()-started
            atomic_json(folder/"trial.json", row)
            self.document["trials"].append(dict(trial=number, status=row["status"], elapsed_s=row["elapsed_s"],
                completed_column_trace_batches=row.get("completed_column_trace_batches", 0),
                completed_gun_traces=row.get("completed_gun_traces", 0)))
            self.document.pop("active_trial", None)
            self.save()
            print(json.dumps(dict(trial=number, status=row["status"], elapsed_s=row["elapsed_s"],
                native_d95_nm=row.get("observation", {}).get("surface", {}).get("d95_nm"))), flush=True)

    def run(self, mode="fit"):
        if mode not in {"measure", "fit"}:
            raise ValueError("Mode must be measure or fit")
        self.document["mode"] = mode
        try:
            initial = np.zeros(len(self.controls))
            self.residual(initial)
            if self.best is None:
                raise ValueError("The seed has no valid numerical objective support")
            if mode == "fit":
                from scipy.optimize import least_squares
                bounds = tuple(np.array([(getattr(c, side)-c.initial)/c.scale for c in self.controls])
                    for side in ("lower", "upper"))
                fitted = least_squares(self.residual, initial, bounds=bounds,
                    jac=lambda q: absolute_jacobian(self.residual, q, bounds, self.jacobian_step),
                    max_nfev=self.maximum, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
                self.document["optimizer"] = dict(success=bool(fitted.success), message=str(fitted.message),
                    nfev=int(fitted.nfev), njev=int(fitted.njev), normalized=fitted.x.tolist(),
                    qualification="Optimizer termination only; no physical qualification")
                if not fitted.success:
                    self.document["status"] = "STOPPED"
                    self.document["reason"] = str(fitted.message)
                    return self.document
            self.document["status"] = "COMPLETE"
        except (SearchStopped, KeyboardInterrupt, InterruptedError) as error:
            self.document.update(status="STOPPED", reason=str(error) or "Interrupted")
        except Exception as error:
            self.document.update(status="STOPPED" if self.cancelled() else "FAILED",
                reason=f"{type(error).__name__}: {error}")
        finally:
            self.save()
        return self.document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "fit"))
    parser.add_argument("--candidate", type=Path, required=True, help="Complete candidate snapshot or checked best.json pointer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rays", type=int, help="Explicitly change only the original emitter sample count")
    parser.add_argument("--step", type=float, help="Explicitly change only the maximum integration step in mm")
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS))
    parser.add_argument("--bounds", type=Path)
    parser.add_argument("--width-scale", type=float, default=1.)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--time-budget", type=float, default=7200.)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--jacobian-step", type=float, default=1e-6)
    parser.add_argument("--d95-scale-nm", type=float, default=.2)
    parser.add_argument("--max-d95-nm", type=float,
        help="Optional strict native diameter cap for screening; no cap or width target by default")
    parser.add_argument("--d95-smoothing-nm", type=float, default=.05,
        help="Radial logistic bandwidth in nm; 0 disables. Native D95 is always reported separately")
    parser.add_argument("--angle-scale-mrad", type=float, default=.03)
    parser.add_argument("--profile-scale-nm", type=float, default=.01, help="Algorithmic profile contrast scale, not a physical tolerance")
    parser.add_argument("--control-regularizer", type=float, default=1e-4,
        help="Lambda for mean squared normalized control displacement from this seed; 0 disables")
    args = parser.parse_args(argv)
    if args.threads < 1 or not 1 <= args.batch_size <= 128:
        parser.error("Require positive threads and batch size in [1,128]")
    if args.output.exists():
        parser.error("--output must be a new directory")
    search = None
    try:
        objective = Objective(d95_scale_nm=args.d95_scale_nm, target_mrad=30., angle_scale_mrad=args.angle_scale_mrad,
            profile_scale_nm=args.profile_scale_nm, d95_smoothing_nm=args.d95_smoothing_nm,
            control_regularizer=args.control_regularizer, max_d95_nm=args.max_d95_nm).validate()
        state, provenance = load_warm_candidate(args.candidate, rays=args.rays, step=args.step)
        controls = control_space(state, args.controls.split(","),
            None if args.bounds is None else read_json(args.bounds), args.width_scale)
        from measure_probe_waist import ProfileEvaluator
        evaluator = ProfileEvaluator(state, controls, offsets_nm=OFFSETS_NM, batch_size=args.batch_size)
        search = Search(evaluator, controls, args.output, objective=objective,
            max_evaluations=args.max_evaluations, time_budget=args.time_budget,
            jacobian_step=args.jacobian_step, cancel_file=args.cancel_file)
        from temsim.instrument_snapshot import capture_instrument_snapshot
        atomic_json(args.output/"initial-inputs.json", capture_instrument_snapshot(state).to_dict())
        search.document.update(warm_start=provenance, threads=args.threads, batch_size=args.batch_size)
        search.save()
        from numba import get_num_threads, set_num_threads
        from threadpoolctl import threadpool_limits
        previous = get_num_threads()
        try:
            set_num_threads(args.threads)
            with threadpool_limits(limits=args.threads):
                result = search.run(args.mode)
        finally:
            set_num_threads(previous)
    except (Exception, KeyboardInterrupt) as error:
        if search is not None and search.document["status"] == "RUNNING":
            search.document.update(status="STOPPED" if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "FAILED", reason=f"{type(error).__name__}: {error}")
            search.save()
        parser.exit(2, f"D95 waist proposal failed: {type(error).__name__}: {error}\n")
    print(json.dumps(dict(status=result["status"], qualification=result["qualification"],
        actual_evaluations=result["actual_evaluations"], report=str(args.output.resolve()/"run.json"))))
    return 0 if result["status"] == "COMPLETE" else 130 if result["status"] == "STOPPED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
