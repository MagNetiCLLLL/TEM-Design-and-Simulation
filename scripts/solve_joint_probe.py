"""Bounded joint physical-field probe search; every result is a proposal.

The production gun and unified Lorentz executor retain the complete source,
physical interception, clocks and kinetic energy. No empirical aberrations,
coordinate rescaling, preset installation or automatic fine-step qualification
are performed. A local fit retains its seed's transmitted calibration cohort;
this is a smooth-search constraint, not a global minimum-current requirement.

Optional --d95-smoothing-nm replaces only the added D95 hinge's optimization
value with a logistic-CDF diameter. Its bandwidth is a radial distance in nm;
the native measured D95 remains unchanged and is the acceptance statistic.
Even a point spot has a positive surrogate floor of 2*tau*log(19). This supplies
a conservative point-spot margin, not a universal upper bound on empirical D95.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SCHEMA = "temsim-joint-probe-search-v1"
DEFAULT_SEED = ROOT / "docs/development/finite_field_direct_focus_20261028.json"
LENSES = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3", "adapter_lens",
          "probe_tl22_lens", "probe_tl21_lens", "probe_tl12_lens", "mini_condenser", "objective_lens")
# (short name, physical component, attribute, local trust half-width, normalization scale)
DEFINITIONS = tuple(zip(("c1", "c2", "c3", "adl", "tl22", "tl21", "tl12", "mini", "objective"),
    LENSES, ("percent",) * 9, (10., 5., 3., 10., 10., 10., 10., 5., 3.), (1.,) * 9)) + (
    ("hp2", "probe_hp2_hexapole", "strength_m3", 2e6, 1e6),
    ("hp1", "probe_hp1_hexapole", "strength_m3", 2e6, 1e6),
    ("hp1_angle", "probe_hp1_hexapole", "orientation_rad", math.pi / 6, .1),
    ("hpc", "probe_hpc_hexapole", "strength_m3", 1e6, 1e6),
    ("hpc_angle", "probe_hpc_hexapole", "orientation_rad", math.pi / 6, .1),
    ("hpol", "probe_hpol_hexapole", "strength_m3", 1e6, 1e6),
    ("hpol_angle", "probe_hpol_hexapole", "orientation_rad", math.pi / 6, .1),
    ("stig_x", "condenser_stigmator", "strength_x_percent", .05, .01),
    ("stig_y", "condenser_stigmator", "strength_y_percent", .05, .01))
DEFAULT_CONTROLS = tuple(row[0] for row in DEFINITIONS)
# Existing physical quadrupole amplitudes are opt-in, without fabricated angles.
DEFINITIONS += (
    ("qph2", "probe_qph2_quadrupole", "strength_m2", 30., 1.),
    ("qpc", "probe_qpc_quadrupole", "strength_m2", 30., 1.),
    ("qph1", "probe_qph1_quadrupole", "strength_m2", 30., 1.),
    ("qpol", "probe_qpol_quadrupole", "strength_m2", 30., 1.))
SEED_FIELDS = {"strength_m3", "orientation_rad", "strength_m2", "kick_x_mrad", "kick_y_mrad",
               "strength_x_percent", "strength_y_percent"}


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Invalid JSON number: {value}")))
    # Also reject overflowed numeric literals such as 1e999.
    json.dumps(data, allow_nan=False)
    return data


def atomic_json(path, data):
    from temsim.benchmark_suite import atomic_json as publish
    publish(path, data)


def atomic_npz(path, arrays):
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".particles-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def components(state):
    return {item.key: item for collection in (state.lenses, state.corrector_elements, state.stigmators,
                                             state.deflectors) for item in collection}


def resolve_candidate_directory(path):
    path = Path(path)
    if path.is_dir():
        path = path / "best.json"
    if path.name == "best.json":
        relative = Path(read_json(path)["directory"])
        candidate = (path.parent / relative).resolve()
        if relative.is_absolute() or not candidate.is_relative_to(path.parent.resolve()):
            raise ValueError("Best-candidate pointer must remain within its run directory")
        return candidate
    return path.parent


def seed_controls(state):
    return {"controls_percent": {item.key: float(item.percent) for item in state.lenses},
            "probe_field_controls": {key: {name: float(getattr(item, name)) for name in sorted(SEED_FIELDS)
                if hasattr(item, name)} for key, item in components(state).items()
                if key.startswith("probe_") and any(hasattr(item, f) for f in SEED_FIELDS)
                or key == "condenser_stigmator"}}


def load_seed(state, path):
    """Apply explicit physical control values only; historical metrics are ignored."""
    path = Path(path)
    if path.is_dir() or path.name == "best.json":
        path = resolve_candidate_directory(path) / "candidate.json"
    data = read_json(path)
    parts = components(state)
    if not isinstance(data, dict) or not isinstance(data.get("controls_percent"), dict) or not isinstance(data.get("probe_field_controls"), dict):
        raise ValueError("Seed requires controls_percent and probe_field_controls objects")
    for key, value in data["controls_percent"].items():
        if key not in {item.key for item in state.lenses}:
            raise ValueError(f"Unknown seed lens: {key}")
        value = finite(value, key)
        if not 0 <= value <= parts[key].max_percent:
            raise ValueError(f"Seed lens exceeds its physical bounds: {key}")
        parts[key].percent = value
    for key, fields in data["probe_field_controls"].items():
        if key not in parts or not (key.startswith("probe_") or key == "condenser_stigmator") or not isinstance(fields, dict):
            raise ValueError(f"Unknown seed physical field: {key}")
        for name, value in fields.items():
            if name not in SEED_FIELDS or not hasattr(parts[key], name):
                raise ValueError(f"Unsupported seed control: {key}.{name}")
            value = finite(value, f"{key}.{name}")
            maximum = getattr(parts[key], "maximum_" + name, None)
            if maximum is not None and abs(value) > float(maximum):
                raise ValueError(f"Seed field exceeds its physical bounds: {key}.{name}")
            if name in {"strength_x_percent", "strength_y_percent"} and abs(value) > 100:
                raise ValueError("Stigmator control exceeds its physical percentage range")
            setattr(parts[key], name, value)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scope": "Physical controls reconstructed as an unqualified seed; historical observations are not reused"}


def snapshot_changes(before, after, path=""):
    """Describe exact JSON snapshot differences without dropping unknown inputs."""
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after}]
    if isinstance(before, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            child = path + "/" + str(key)
            if key not in before or key not in after:
                result.append({"path": child, "before": before.get(key, {"missing": True}),
                               "after": after.get(key, {"missing": True})})
            else:
                result.extend(snapshot_changes(before[key], after[key], child))
        return result
    if isinstance(before, list):
        if len(before) != len(after):
            return [{"path": path, "before": before, "after": after}]
        return [change for i, (a, b) in enumerate(zip(before, after))
                for change in snapshot_changes(a, b, path + "/" + str(i))]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def prepare_objective_observations(state):
    """Synchronize only production-derived objective observation coordinates.

    Physical objective edits invalidate four saved observation values. The
    production gun-field capture already calls sync_objective; do that explicit
    preparation before freezing trial inputs, and reject any unrelated change.
    The post-execution guard still compares the complete unfiltered snapshot.
    """
    from temsim.instrument_snapshot import capture_instrument_snapshot
    before = capture_instrument_snapshot(state)
    before_data = before.to_dict()
    state.sync_objective()
    after = capture_instrument_snapshot(state)
    after_data = after.to_dict()
    root = before_data["graph"]["root"]["ref"]
    allowed = {f"/graph/nodes/{root}/attributes/{name}" for name in
               ("objective_back_focal_plane_z_mm", "objective_image_plane_z_mm")}
    objective_type = type(state.objective_lens).__module__ + ":" + type(state.objective_lens).__name__
    for index, node in enumerate(before_data["graph"]["nodes"]):
        if node.get("type") == objective_type and node.get("attributes", {}).get("key") == "objective_lens":
            allowed.update(f"/graph/nodes/{index}/attributes/{name}" for name in
                           ("_back_focal_plane_z_mm", "_image_plane_z_mm"))
    changes = snapshot_changes({k: v for k, v in before_data.items() if k != "digest"},
                               {k: v for k, v in after_data.items() if k != "digest"})
    unexpected = [change for change in changes if change["path"] not in allowed]
    if unexpected:
        raise ValueError("Objective observation preparation changed physical inputs: " +
                         json.dumps(unexpected, sort_keys=True, allow_nan=False))
    return {"operation": "production State.sync_objective", "before_digest": before.digest,
            "after_digest": after.digest, "derived_observation_changes": changes,
            "scope": "Only the four derived objective observation coordinates may change during preparation"}


def make_state(rays=193, step=.2, seed=DEFAULT_SEED):
    from temsim.benchmark_suite import BenchmarkConfig, configured_state
    if type(rays) is not int or rays < 16 or not 0 < finite(step, "step") <= .2:
        raise ValueError("Use at least 16 source points and a positive step no larger than 0.2 mm")
    state = configured_state(BenchmarkConfig(rays=rays, step_mm=step))
    if seed is not None:
        load_seed(state, seed)
    prepare_objective_observations(state)
    aperture = state.condenser_aperture_2
    if not aperture.enabled or float(aperture.radius_mm) != .05:
        raise ValueError("Joint search requires the existing physical C2 diameter of exactly 100 um")
    return state


@dataclass(frozen=True)
class Control:
    name: str
    key: str
    field: str
    initial: float
    lower: float
    upper: float
    scale: float


def control_space(state, names=None, bounds=None, width_scale=1.):
    width_scale = finite(width_scale, "width scale")
    if width_scale <= 0:
        raise ValueError("Trust-width scale must be positive")
    names = DEFAULT_CONTROLS if names is None else tuple(names)
    known, parts = {row[0]: row for row in DEFINITIONS}, components(state)
    if not names or len(set(names)) != len(names) or set(names) - known.keys():
        raise ValueError("Controls must be distinct supported physical control names")
    bounds = {} if bounds is None else bounds
    if not isinstance(bounds, dict) or set(bounds) - set(names):
        raise ValueError("Explicit bounds may only name selected controls")
    result = []
    for name in names:
        _, key, field, width, scale = known[name]
        part = parts[key]
        # The objective's nominal centre is the sample centre; its upstream
        # physical field focuses at the entrance surface before that centre.
        if not part.enabled or float(part.z_mm) > float(state.sample.z_mm):
            raise ValueError(f"Control must be an enabled pre-sample physical component: {name}")
        value = finite(float(getattr(part, field)), name)
        if field == "percent":
            physical = (0., float(part.max_percent))
        elif field in {"strength_m3", "strength_m2"}:
            maximum = finite(float(getattr(part, "maximum_" + field)), name + " maximum")
            physical = (-maximum, maximum)
        elif field.startswith("strength_"):
            physical = (-100., 100.)
        else:
            physical = (-math.inf, math.inf)  # An angle is periodic; do not wrap physical inputs.
        lower, upper = max(physical[0], value-width*width_scale), min(physical[1], value+width*width_scale)
        if name in bounds:
            row = bounds[name]
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"Explicit bounds require [lower, upper]: {name}")
            lower, upper = (finite(v, name + " bound") for v in row)
        if not physical[0] <= lower <= value <= upper <= physical[1] or lower == upper:
            raise ValueError(f"Bounds must contain the seed and remain within physical limits: {name}")
        result.append(Control(name, key, field, value, lower, upper, scale))
    return tuple(result)


def physical_values(controls, normalized):
    q = np.asarray(normalized, dtype=float)
    if q.shape != (len(controls),) or not np.isfinite(q).all():
        raise ValueError("Control vector has an invalid shape or nonfinite values")
    values = np.array([c.initial if v == 0 else
        c.lower if v == (c.lower-c.initial)/c.scale else
        c.upper if v == (c.upper-c.initial)/c.scale else c.initial + float(v)*c.scale
        for c, v in zip(controls, q)])
    if any(not c.lower <= v <= c.upper for c, v in zip(controls, values)):
        raise ValueError("Control vector exceeds its declared bounds")
    return values


def set_values(state, controls, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(controls),) or not np.isfinite(values).all():
        raise ValueError("Physical control vector is invalid")
    parts = components(state)
    for control, value in zip(controls, values):
        if not control.lower <= value <= control.upper:
            raise ValueError("Physical control value exceeds its declared bounds")
        setattr(parts[control.key], control.field, float(value))


def aperture_observation(state, segment, source_alive, weights, ids, current):
    """Observe the exact aperture plane; never filter by angle or redraw a pupil."""
    aperture, cp, branch = state.condenser_aperture_2, segment.checkpoints, segment.branch
    plane = float(aperture.z_mm)
    found = np.flatnonzero(np.asarray(cp.z_mm) == plane)
    if len(found) != 1:
        raise ValueError("The exact physical C2 aperture checkpoint is missing")
    index = int(found[0])
    stopped = np.asarray(branch.blocked_z)
    incoming = np.asarray(source_alive, bool) & (~np.isfinite(stopped) | (stopped >= plane)) & (weights > 0)
    outgoing = np.asarray(cp.unified_state.active[index], bool) & (weights > 0)
    if np.any(outgoing & ~incoming):
        raise ValueError("C2 checkpoint would revive a stopped source particle")
    arrays = {"c2_" + key: np.asarray(getattr(cp, key)[index]).copy()
              for key in ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")}
    arrays.update(c2_incoming_alive=incoming.copy(), c2_outgoing_alive=outgoing.copy())
    radius = np.hypot(arrays["c2_x_m"]-float(aperture.offset_x_mm)*1e-3,
                      arrays["c2_y_m"]-float(aperture.offset_y_mm)*1e-3)
    report = {"z_mm": plane, "diameter_um": float(aperture.radius_mm)*2000,
        "scope": "Exact aperture reference plane before/after contacts at that plane; earlier finite-thickness stops remain excluded"}
    for name, selected in (("incoming", incoming), ("outgoing", outgoing)):
        total = float(weights[selected].sum())
        finite_radius = bool(np.isfinite(radius[selected]).all())
        report[name] = {"source_ids": ids[selected].tolist(), "points": int(selected.sum()),
            "absolute_weight": total, "current_a": current*total/float(weights.sum()),
            "maximum_radius_um": float(np.max(radius[selected])*1e6) if total > 0 and finite_radius else None,
            "radius_status": "MEASURED" if total > 0 and finite_radius else "NO_FINITE_COMPLETE_PLANE_POPULATION"}
    return report, arrays


@dataclass
class Observation:
    report: dict
    arrays: dict
    state: object


class JointEvaluator:
    """Production execution with exact source dependency and prefix proofs."""
    def __init__(self, state, controls, *, upstream_spacing_nm=100., cancelled=lambda: False):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.simulation_modes import is_unified
        from temsim.physics.unified_field_runtime import unified_runtime_preflight
        if not is_unified(state) or state.electron_gun.source_representation != "classical_particles":
            raise ValueError("Joint probe search requires classical unified physical fields")
        issues = unified_runtime_preflight(state)["issues"]
        if issues:
            raise ValueError("Unsupported physical inputs: " + "; ".join(issues))
        if state.condenser_aperture_2.radius_mm != .05 or not state.condenser_aperture_2.enabled:
            raise ValueError("C2 aperture must retain its physical 100 um diameter")
        self.spacing_nm = finite(upstream_spacing_nm, "surface spacing")
        if self.spacing_nm <= 0:
            raise ValueError("Surface observation spacing must be positive")
        controls = tuple(controls)
        checked = control_space(state, tuple(c.name for c in controls),
                                {c.name: [c.lower, c.upper] for c in controls})
        if controls != checked:
            raise ValueError("Evaluator controls must match the declared installed physical control contract")
        self.controls, self.cancelled = controls, cancelled
        self.snapshot = capture_instrument_snapshot(state)
        self.gun = self.gun_signature = self.segment = None

    def evaluate(self, values):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        from temsim.optics.surface_probe_focus import surface_z_mm, local_waist_from_radii
        from temsim.physics.beam_statistics import transverse_beam_statistics
        from temsim.physics.beam_current import effective_source_current_a
        from temsim.physics.particle_sections import _trace_segment, _events, _digest_arrays, gun_dependency_signature
        from temsim.physics.ray_identity import emitted_source_identity
        from temsim.physics.simulation import _validate_gun_flight_times
        from temsim.optics.electron_gun.source import trace_source_to_exit
        from temsim.optics.electron_gun.source_policy import require_physical_gun_source
        from audit_joint_probe_fields import audit

        state = self.snapshot.restore()
        set_values(state, self.controls, values)
        preparation = prepare_objective_observations(state)
        original = capture_instrument_snapshot(state)
        state._tuning_cancelled = self.cancelled
        if self.cancelled():
            raise SearchStopped("Cancellation or wall-time budget reached")
        require_physical_gun_source(state.electron_gun)
        current = float(effective_source_current_a(state))
        if not math.isfinite(current) or current <= 0:
            raise ValueError("The physical source current must be finite and positive")
        start, end = float(state.electron_gun.exit_plane_z_mm), surface_z_mm(state)
        h = self.spacing_nm * 1e-6
        planes = np.array((end-2*h, end-h, end))
        if planes[0] <= start:
            raise ValueError("The three surface planes must follow the physical gun exit")
        signature = gun_dependency_signature(state)
        gun_reused = bool(signature and signature == self.gun_signature and self.gun is not None)
        if not gun_reused:
            self.gun = trace_source_to_exit(state)
            self.gun_signature, self.segment = signature, None
        if not _validate_gun_flight_times(self.gun):
            raise ValueError("The executed physical gun must retain its flight-time history")
        gun, emitted = self.gun, self.gun.exit_bundle
        identity = emitted_source_identity(gun)
        initial = tuple(np.asarray(getattr(emitted, key)) for key in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"))
        offsets, weights = np.asarray(emitted.energy_offset_ev), np.asarray(emitted.weight)
        kinetic = float(state.electron_gun.nominal_exit_energy_ev) + offsets
        survival = (np.asarray(emitted.alive, bool), np.asarray(gun.blocked_z_mm), list(gun.blocked_key))
        ids = np.asarray(identity[0])
        if (weights.shape != ids.shape or weights.ndim != 1 or not np.isfinite(weights).all()
                or np.any(weights < 0) or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
                or not np.issubdtype(ids.dtype, np.integer)):
            raise ValueError("Physical source IDs and absolute weights must be finite, unique and retained")
        segment, resume, reused = _trace_segment(state, "incident", start, end, initial, offsets,
            weights, identity, survival, _events(state), self.segment, math.inf,
            initial_kicks=True, save_z_mm=(*planes, float(state.condenser_aperture_2.z_mm)), initial_kinetic_energy_ev=kinetic)
        self.segment = segment
        field_audit = audit(state, plan=segment.plan)
        if field_audit.get("passed") is not True:
            raise ValueError("The actual physical-field action audit did not pass")
        branch, cp = segment.branch, segment.checkpoints
        if not np.array_equal(branch.source_ray_id, ids) or not np.array_equal(branch.ray_weight, weights):
            raise ValueError("Column transport changed the original source identity or absolute weights")
        aperture_report, aperture_arrays = aperture_observation(state, segment, survival[0], weights, ids, current)
        rows = []
        for plane in planes:
            found = np.flatnonzero(np.asarray(cp.z_mm) == plane)
            if len(found) != 1:
                raise ValueError("An exact surface-triple checkpoint is missing")
            rows.append(int(found[0]))
        alive = np.asarray(branch.alive, bool) & (weights > 0)
        arrays = {key: np.asarray(getattr(cp, key))[rows].copy() for key in
                  ("x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev")}
        if any(not np.isfinite(value[:, alive]).all() for value in arrays.values()):
            raise ValueError("Live surface phase space, energy and clocks must be finite")
        if np.any(arrays["flight_time_s"][:, alive] < 0) or np.any(arrays["kinetic_energy_ev"][:, alive] <= 0):
            raise ValueError("Live surface kinetic energy must be positive and clocks nonnegative")
        stats = ([transverse_beam_statistics(*(arrays[key][i] for key in ("x_m", "y_m", "tx_rad", "ty_rad")),
                    alive=alive, weights=weights) for i in range(3)] if np.any(alive) else [])
        radii = tuple(s.radius_rms_m for s in stats)
        waist, second = local_waist_from_radii(radii, h*1e-3) if stats else (math.nan, 0.)
        source_population = _digest_arrays(*initial, offsets, kinetic, weights, *survival[:2],
            np.asarray(survival[2], dtype=str), *identity)
        arrays.update(z_mm=planes, alive=alive.copy(), source_ray_id=ids.copy(), weights=weights.copy(),
            source_azimuth_rad=np.asarray(identity[1]).copy(), blocked_z_mm=np.asarray(branch.blocked_z).copy(),
            blocked_key=np.asarray(branch.blocked_key, dtype=str), gun_exit_energy_offset_ev=offsets.copy(),
            gun_exit_kinetic_energy_ev=kinetic.copy(), gun_exit_alive=survival[0].copy(),
            gun_blocked_z_mm=survival[1].copy(), gun_blocked_key=np.asarray(survival[2], dtype=str))
        for key, value in zip(("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"), initial):
            arrays["gun_exit_" + key] = value.copy()
        for key in ("z_mm", "x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s"):
            arrays["gun_" + key] = np.asarray(getattr(gun, key)).copy()
        for key in ("z_mm", "x_m", "y_m", "tx_rad", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
            arrays["checkpoint_" + key] = np.asarray(getattr(cp, key)).copy()
        arrays["checkpoint_alive"] = np.asarray(cp.unified_state.active).copy()
        arrays.update(aperture_arrays)
        for key, value in gun.emission_reference.items():
            arrays["emission_" + key] = np.asarray(value).copy()
        if capture_instrument_snapshot(state).digest != original.digest:
            raise ValueError("Inputs changed during physical evaluation")
        s = stats[-1] if stats else None
        transmitted = weights[alive]
        report = {"input_digest": original.digest, "implementation": original.implementation,
            "input_preparation": preparation,
            "gun_dependency_signature": signature, "gun_exit_population_digest": source_population,
            "emitted_identity_digest": _digest_arrays(ids, weights, np.asarray(identity[1]),
                *(np.asarray(gun.emission_reference[key]) for key in sorted(gun.emission_reference))),
            "gun_reused": gun_reused, "column_prefix_reused": bool(reused), "resume_z_mm": float(resume),
            "field_audit": field_audit, "source_points": len(weights), "surviving_points": int(alive.sum()),
            "effective_points": float(transmitted.sum()**2 / (transmitted @ transmitted)) if np.any(transmitted) else 0.,
            "source_current_a": current, "transmitted_weight": float(transmitted.sum()),
            "source_weight": float(weights.sum()), "current_a": current*float(transmitted.sum()/weights.sum()),
            "surface_z_mm": end, "surface_spacing_nm": self.spacing_nm, "step_mm": float(state.step_mm),
            "c2_aperture": aperture_report,
            "source_identity_scope": "Fixed original emission law, quadrature, energy spread, IDs and absolute weights; physical gun-exit phase may change under changed overlapping fields and is retraced by the production dependency signature",
            "population_status": "TRANSMITTED" if s is not None else "NO_TRANSMISSION",
            "alpha95_mrad": s.convergence_95_mrad if s is not None else None,
            "d90_nm": native_current_d90_nm(arrays["x_m"][-1], arrays["y_m"][-1],
                alive, weights, s) if s is not None else None,
            "d95_nm": 2e9*s.radius_95_m if s is not None else None,
            "rms_radius_nm": 1e9*s.radius_rms_m if s is not None else None,
            "centroid_x_m": s.mean_x_m if s is not None else None,
            "centroid_y_m": s.mean_y_m if s is not None else None,
            "local_waist_nm": waist if math.isfinite(waist) else None,
            "variance_second_derivative": second, "covariance_waist_nm": s.waist_offset_m*1e9 if s is not None else None,
            "statistics": asdict(s) if s is not None else None, "qualification": "PROPOSAL_ONLY",
            "unperformed_checks": ["Independent integration-step refinement", "Source sampling convergence",
                "C2-C3 and C3-ADL topology", "Corrector crossover topology", "No extra pre-specimen waist"]}
        return Observation(report, arrays, state)


class CohortRejected(ValueError):
    pass


class SearchStopped(RuntimeError):
    pass


def native_current_d90_nm(x_m, y_m, alive, weights, statistics):
    """Report D90 using the validated full-current population and native centroid.

    ``statistics`` must be the native surface statistic of these same arrays,
    as already calculated by JointEvaluator. No fit cohort, smoothing, gate,
    physical source weight or coordinate is changed by this extra readout.
    """
    from temsim.physics.beam_statistics import _weighted_quantile
    raw = np.asarray(weights)
    selected = np.asarray(alive, dtype=bool) & (raw > 0)
    current = np.asarray(raw[selected], dtype=float)
    current /= np.sum(current)
    radius = np.hypot(np.asarray(x_m)[selected]-statistics.mean_x_m,
        np.asarray(y_m)[selected]-statistics.mean_y_m)
    return 2e9*_weighted_quantile(radius, current, .9)


def smoothed_current_d95_nm(x_m, y_m, alive, weights, tau_nm):
    """Return twice the positive radius where the full-current smooth CDF is .95.

    F(q) = sum(w * expit((q - r_nm) / tau_nm)) / sum(w), with radii about
    the same full positive-current centroid used by the native statistic.
    All actually surviving positive-weight particles participate, including new
    arrivals outside a fit's initial calibration cohort. Input arrays and raw
    weights are never changed. Centering/normalization are statistics only.

    A zero-radius spot gives 2*tau*log(19); a radius-R ring gives that floor
    plus 2*R. Smoothing is not a universal empirical-quantile upper bound. As
    tau tends to zero, an exact .95 cumulative-weight plateau can select an
    interior point instead of the native quantile's left endpoint. Native D95
    must therefore remain the independent reported and accepted measurement.
    """
    from scipy.optimize import brentq
    from scipy.special import expit
    tau = finite(tau_nm, "D95 smoothing radius")
    if tau <= 0:
        raise ValueError("D95 smoothing radius must be positive")
    x, y, mask, raw = (np.asarray(value) for value in (x_m, y_m, alive, weights))
    if (x.ndim != 1 or x.shape != y.shape or x.shape != mask.shape or x.shape != raw.shape
            or mask.dtype.kind != "b" or any(v.dtype.kind not in "iuf" for v in (x, y, raw))):
        raise ValueError("Smoothed D95 requires aligned real vectors and a boolean alive mask")
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Smoothed D95 requires finite nonnegative original weights")
    selected = mask & (raw > 0)
    if not np.any(selected):
        raise ValueError("Smoothed D95 requires positive surviving current")
    x, y, w = (np.asarray(v[selected], dtype=float) for v in (x, y, raw))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Smoothed D95 requires finite positive-current positions")
    total = float(np.sum(w))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Smoothed D95 current sum must be finite and positive")
    w = w/total
    # Preserve native weighted-centroid arithmetic in metres before converting.
    radii = np.hypot(x-np.sum(w*x), y-np.sum(w*y))*1e9
    if not np.isfinite(radii).all():
        raise ValueError("Smoothed D95 radii must be finite in nm")
    # F(0) <= .5. At this upper bound every logistic term is at least .975,
    # safely beyond .95, including a constant-radius ring. nextafter handles
    # a small positive bandwidth whose margin rounds to the largest radius.
    upper = float(np.nextafter(float(np.max(radii))+tau*math.log(39.), math.inf))
    if not math.isfinite(upper):
        raise ValueError("Smoothed D95 root bracket exceeds finite floating-point range")

    def equation(q):
        with np.errstate(over="ignore", under="ignore"):
            return float(np.sum(w*expit((q-radii)/tau))-.95)

    q, solved = brentq(equation, 0., upper, xtol=float(np.nextafter(0., 1.)),
        rtol=8*np.finfo(float).eps, maxiter=256, full_output=True)
    diameter = 2*float(q)
    if not solved.converged or not math.isfinite(diameter) or q < 0 or abs(equation(q)) > 1e-8:
        raise ValueError("Smoothed D95 CDF root is not resolved at floating-point precision")
    return diameter


@dataclass(frozen=True)
class Objective:
    target_mrad: float = 30.
    position_scale_nm: float = .2
    angle_scale_mrad: float = .03
    focus_scale_nm: float = 1.
    d95_target_nm: float | None = None
    d95_scale_nm: float | None = None
    d95_smoothing_nm: float = 0.

    @property
    def uses_d95_hinge(self):
        return self.d95_target_nm is not None

    def validate(self):
        if (self.d95_target_nm is None) != (self.d95_scale_nm is None):
            raise ValueError("D95 target and scale must be supplied together")
        values = asdict(self)
        smoothing = finite(values.pop("d95_smoothing_nm"), "D95 smoothing radius")
        if smoothing < 0:
            raise ValueError("D95 smoothing radius must be nonnegative; zero disables smoothing")
        if smoothing > 0 and not self.uses_d95_hinge:
            raise ValueError("D95 smoothing requires both D95 target and scale")
        if not self.uses_d95_hinge:
            values.pop("d95_target_nm")
            values.pop("d95_scale_nm")
        if any(finite(v, key) <= 0 for key, v in values.items()):
            raise ValueError("Objective targets and scales must be positive")
        return self


class LocalCohort:
    def __init__(self, observation, *, d95_enabled=False):
        a = observation.arrays
        self.ids, self.weights = a["source_ray_id"].copy(), a["weights"].copy()
        self.source_identity = observation.report["emitted_identity_digest"]
        self.selected = np.asarray(a["alive"], bool).copy() & (self.weights > 0)
        if self.selected.sum() < 16 or observation.report["effective_points"] < 16*(1-32*np.finfo(float).eps):
            raise ValueError("A local fit requires at least 16 effective transmitted seed particles")
        self.d95_enabled = bool(d95_enabled)
        self.dimension = int(self.selected.sum())*2 + 3 + int(self.d95_enabled)

    def residual(self, observation, objective, *, diagnostics=None):
        objective.validate()
        if objective.uses_d95_hinge != self.d95_enabled:
            raise ValueError("The local objective dimension cannot change during a fit")
        a, r = observation.arrays, observation.report
        if (r["emitted_identity_digest"] != self.source_identity
                or not np.array_equal(a["source_ray_id"], self.ids)
                or not np.array_equal(a["weights"], self.weights)):
            raise ValueError("Original emission IDs or absolute weights changed")
        if np.any(self.selected & ~a["alive"]):
            raise CohortRejected("A physical stop removed a local calibration particle; trial rejected")
        if r["local_waist_nm"] is None or r["variance_second_derivative"] <= 0:
            raise CohortRejected("Surface triple does not resolve a convex local waist")
        w = self.weights[self.selected]
        w = w / w.sum()  # Objective normalization only; stored absolute weights are untouched.
        xy = [a[key][-1, self.selected] for key in ("x_m", "y_m")]
        centered = [(value - np.sum(w*value))*1e9*np.sqrt(w)/objective.position_scale_nm for value in xy]
        result = np.r_[*centered, (r["alpha95_mrad"]-objective.target_mrad)/objective.angle_scale_mrad,
            r["local_waist_nm"]/objective.focus_scale_nm, r["covariance_waist_nm"]/objective.focus_scale_nm]
        if self.d95_enabled:
            # This is the measured full-current diameter, including any newly
            # transmitted source rays. Do not recompute it on the local cohort,
            # normalize it by RMS/angle, or rescale the stored phase arrays.
            diameter = finite(r["d95_nm"], "Measured full-current D95")
            if diameter < 0:
                raise ValueError("Measured full-current D95 must be nonnegative")
            native_diameter = diameter
            if objective.d95_smoothing_nm > 0:
                diameter = smoothed_current_d95_nm(a["x_m"][-1], a["y_m"][-1],
                    a["alive"], a["weights"], objective.d95_smoothing_nm)
            hinge = max(diameter-objective.d95_target_nm, 0.)/objective.d95_scale_nm
            if diagnostics is not None:
                diagnostics.update(status="EVALUATED", native_d95_nm=native_diameter,
                    smoothed_d95_nm=diameter if objective.d95_smoothing_nm > 0 else None,
                    d95_smoothing_nm=objective.d95_smoothing_nm, hinge_diameter_nm=diameter,
                    hinge_residual=hinge, scope="Optimization only; native full-current D95 remains the acceptance statistic")
            result = np.r_[result, hinge]
        if result.shape != (self.dimension,) or not np.isfinite(result).all():
            raise CohortRejected("Nonfinite physical objective")
        return result


def absolute_jacobian(function, point, bounds, step):
    """Bounded forward differences with an absolute normalized increment."""
    point = np.asarray(point, dtype=float)
    base = function(point)
    lower, upper = bounds
    columns = []
    for index in range(len(point)):
        direction = 1 if upper[index]-point[index] >= point[index]-lower[index] else -1
        distance = min(step, upper[index]-point[index] if direction > 0 else point[index]-lower[index])
        if distance <= 0:
            raise ValueError("No finite-difference room inside a physical bound")
        trial = point.copy()
        trial[index] += direction*distance
        actual = trial[index]-point[index]
        columns.append((function(trial)-base)/actual)
    return np.column_stack(columns)


class Search:
    def __init__(self, evaluator, controls, output, *, objective=Objective(), max_evaluations=80,
                 time_budget=7200., cancel_file=None, jacobian_step=1e-4):
        if type(max_evaluations) is not int or max_evaluations < 1:
            raise ValueError("Actual-evaluation budget must be a positive integer")
        if finite(time_budget, "time budget") <= 0 or finite(jacobian_step, "Jacobian step") <= 0:
            raise ValueError("Time budget and absolute Jacobian step must be positive")
        objective.validate()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "trials").mkdir()
        (self.output / "candidates").mkdir()
        self.evaluator, self.controls, self.objective = evaluator, controls, objective.validate()
        self.maximum, self.budget, self.jacobian_step = max_evaluations, time_budget, jacobian_step
        self.cancel_file = None if cancel_file is None else Path(cancel_file)
        self.started, self.evaluations, self.best_cost = perf_counter(), 0, math.inf
        self.cache, self.cohort, self.best = {}, None, None
        self.evaluator.cancelled = self.cancelled
        self.document = {"schema": SCHEMA, "status": "RUNNING", "qualification": "PROPOSAL_ONLY",
            "controls": [asdict(c) for c in controls], "objective": asdict(objective),
            "max_actual_evaluations": max_evaluations, "time_budget_seconds": time_budget,
            "absolute_normalized_jacobian_step": jacobian_step, "trials": [],
            "optimizer_variable_scaling": "Offset/physical-scale variables with Jacobian-based trust-region scaling",
            "budget_scope": "Cooperative wall-time cancellation is checked by production transport; no hard process deadline",
            "local_cohort_scope": "Retain the seed's transmitted cohort during this fit; no global transmission floor"}
        self.save()

    def cancelled(self):
        return perf_counter()-self.started >= self.budget or bool(self.cancel_file and self.cancel_file.exists())

    def save(self):
        self.document.update(actual_evaluations=self.evaluations, elapsed_seconds=perf_counter()-self.started)
        atomic_json(self.output / "run.json", self.document)

    def _best(self, observation, trial, cost):
        from temsim.instrument_snapshot import capture_instrument_snapshot
        relative = Path("candidates") / f"trial-{trial:06d}"
        folder = self.output / relative
        folder.mkdir()
        atomic_json(folder / "state.json", observation.state.to_dict())
        atomic_json(folder / "inputs.json", capture_instrument_snapshot(observation.state).to_dict())
        atomic_npz(folder / "particles.npz", observation.arrays)
        record = {"schema": SCHEMA, "trial": trial, "cost": cost, "qualification": "PROPOSAL_ONLY",
            **seed_controls(observation.state), "observation": observation.report}
        atomic_json(folder / "candidate.json", record)
        files = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                 for name in ("state.json", "inputs.json", "particles.npz", "candidate.json")}
        pointer = {"schema": SCHEMA, "directory": relative.as_posix(), "trial": trial,
                   "cost": cost, "files_sha256": files, "qualification": "PROPOSAL_ONLY"}
        atomic_json(self.output / "best.json", pointer)
        self.best, self.best_cost = pointer, math.inf if cost is None else cost

    def residual(self, normalized):
        if self.cancelled():
            raise SearchStopped("Cancellation file or cooperative wall-time budget reached")
        values = physical_values(self.controls, normalized)
        key = values.tobytes()
        if key in self.cache:
            return self.cache[key].copy()
        if self.evaluations >= self.maximum:
            raise SearchStopped("Maximum actual evaluations reached, including numerical Jacobian calls")
        self.evaluations += 1
        number, started = self.evaluations, perf_counter()
        row = {"trial": number, "status": "RUNNING", "values": dict(zip((c.name for c in self.controls), values.tolist())),
               "normalized": np.asarray(normalized).tolist()}
        self.document["active_trial"] = number
        atomic_json(self.output / "trials" / f"{number:06d}.json", row)
        self.save()
        try:
            observation = self.evaluator.evaluate(values)
            row["observation"] = observation.report
            diagnostics = None
            if self.objective.uses_d95_hinge:
                diagnostics = {"status": "NOT_EVALUATED", "native_d95_nm": observation.report.get("d95_nm"),
                    "smoothed_d95_nm": None, "d95_smoothing_nm": self.objective.d95_smoothing_nm}
                row["objective_diagnostics"] = diagnostics
            if getattr(self, "mode", "fit") == "measure":
                # A fresh physical seed may have a different transmitted cohort.
                # Measurement never treats the local fit constraint as acceptance.
                try:
                    residual = LocalCohort(observation, d95_enabled=self.objective.uses_d95_hinge).residual(
                        observation, self.objective, diagnostics=diagnostics)
                    cost = float(residual @ residual)
                    row["objective_status"] = "MEASURED"
                except (ValueError, CohortRejected) as error:
                    residual, cost = np.zeros(1), None
                    row.update(objective_status="NOT_EVALUATED", objective_reason=str(error))
                row.update(status="COMPLETE", objective_squared_norm=cost)
                self._best(observation, number, cost)
                self.cache[key] = residual.copy()
                return residual
            if self.cohort is None:
                self.cohort = LocalCohort(observation, d95_enabled=self.objective.uses_d95_hinge)
            try:
                residual = self.cohort.residual(observation, self.objective, diagnostics=diagnostics)
            except CohortRejected as error:
                row.update(status="REJECTED_LOCAL_COHORT", error=str(error))
                # A rejected physical loss must not undercut the seed's cost.
                penalty = max(1e6, 100*math.sqrt(getattr(self, "seed_cost", 0.)/self.cohort.dimension))
                residual = np.full(self.cohort.dimension, penalty)
            else:
                cost = float(residual @ residual)
                if not hasattr(self, "seed_cost"):
                    self.seed_cost = cost
                row.update(status="COMPLETE", objective_squared_norm=cost)
                if cost < self.best_cost:
                    self._best(observation, number, cost)
            self.cache[key] = residual.copy()
            return residual
        except BaseException as error:
            row.update(status="CANCELLED" if self.cancelled() or isinstance(error, (KeyboardInterrupt, SearchStopped)) else "FAILED",
                       error=f"{type(error).__name__}: {error}")
            raise
        finally:
            row["elapsed_seconds"] = perf_counter()-started
            atomic_json(self.output / "trials" / f"{number:06d}.json", row)
            self.document["trials"].append({"trial": number, "status": row["status"], "elapsed_seconds": row["elapsed_seconds"]})
            self.document.pop("active_trial", None)
            self.save()
            print(json.dumps({"trial": number, "status": row["status"], "elapsed_seconds": row["elapsed_seconds"],
                              "d95_nm": row.get("observation", {}).get("d95_nm")}), flush=True)

    def run(self, mode):
        if mode not in {"measure", "fit"}:
            raise ValueError("Search mode must be measure or fit")
        self.mode = mode
        try:
            initial = np.zeros(len(self.controls))
            self.residual(initial)
            if mode == "fit":
                from scipy.optimize import least_squares
                bounds = tuple(np.array([(getattr(c, side)-c.initial)/c.scale for c in self.controls])
                               for side in ("lower", "upper"))
                fitted = least_squares(self.residual, initial, bounds=bounds,
                    jac=lambda q: absolute_jacobian(self.residual, q, bounds, self.jacobian_step),
                    max_nfev=self.maximum, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
                self.document["optimizer"] = {"success": bool(fitted.success), "message": str(fitted.message),
                    "nfev": int(fitted.nfev), "njev": int(fitted.njev), "normalized": fitted.x.tolist()}
            self.document["status"] = "COMPLETE"
        except (SearchStopped, KeyboardInterrupt) as error:
            self.document.update(status="STOPPED", reason=str(error) or "Keyboard interrupt")
        except Exception as error:
            self.document.update(status="STOPPED" if self.cancelled() else "FAILED", reason=f"{type(error).__name__}: {error}")
        finally:
            self.document["best"] = self.best
            self.save()
        return self.document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "fit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--rays", type=int, default=193)
    parser.add_argument("--step", type=float, default=.2, help="Maximum physical integration step in mm; proposal only")
    parser.add_argument("--controls", default=",".join(DEFAULT_CONTROLS),
                        help="Comma-separated controls; default 18. Optional existing quadrupoles: qph2,qpc,qph1,qpol")
    parser.add_argument("--bounds", type=Path, help="JSON {control:[physical lower,physical upper]}")
    parser.add_argument("--width-scale", type=float, default=1.)
    parser.add_argument("--max-evaluations", type=int, default=80)
    parser.add_argument("--time-budget", type=float, default=7200.)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--jacobian-step", type=float, default=1e-4)
    parser.add_argument("--position-scale-nm", type=float, default=.2)
    parser.add_argument("--angle-scale-mrad", type=float, default=.03)
    parser.add_argument("--focus-scale-nm", type=float, default=1.)
    parser.add_argument("--d95-target-nm", type=float,
                        help="Optional full-current D95 hinge target in nm; requires --d95-scale-nm")
    parser.add_argument("--d95-scale-nm", type=float,
                        help="Positive scale for max(D95-target,0)/scale; disabled unless both D95 options are supplied")
    parser.add_argument("--d95-smoothing-nm", type=float, default=0.,
                        help="Optional positive radial logistic bandwidth in nm (0 disables); requires both D95 options. Native D95 remains unchanged")
    parser.add_argument("--surface-spacing-nm", type=float, default=100.)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.exists():
        parser.error("--output must be a new directory")
    try:
        objective = Objective(30., args.position_scale_nm, args.angle_scale_mrad, args.focus_scale_nm,
                              args.d95_target_nm, args.d95_scale_nm, args.d95_smoothing_nm).validate()
    except ValueError as error:
        parser.error(str(error))
    search = None
    try:
        state = make_state(args.rays, args.step, args.seed)
        controls = control_space(state, args.controls.split(","),
            None if args.bounds is None else read_json(args.bounds), args.width_scale)
        evaluator = JointEvaluator(state, controls, upstream_spacing_nm=args.surface_spacing_nm)
        search = Search(evaluator, controls, args.output,
            objective=objective,
            max_evaluations=args.max_evaluations, time_budget=args.time_budget,
            cancel_file=args.cancel_file, jacobian_step=args.jacobian_step)
        from temsim.instrument_snapshot import capture_instrument_snapshot
        atomic_json(args.output / "initial-inputs.json", capture_instrument_snapshot(state).to_dict())
        search.document.update(seed={"path": str(args.seed.resolve()),
            "interpretation": "Physical controls only; historical results are not current evidence"}, threads=args.threads,
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), mode=args.mode)
        search.save()
        from threadpoolctl import threadpool_limits
        from numba import set_num_threads
        set_num_threads(args.threads)
        with threadpool_limits(limits=args.threads):
            result = search.run(args.mode)
        return 0 if result["status"] == "COMPLETE" and result["best"] is not None else 1
    except Exception as error:
        if search is not None and search.document["status"] == "RUNNING":
            search.document.update(status="FAILED", reason=f"{type(error).__name__}: {error}")
            search.save()
        print(f"Joint probe search failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
