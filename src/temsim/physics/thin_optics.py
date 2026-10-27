"""Fast, explicit thin-element column approximation.

Drifts are exact straight lines; each optical element acts once at its placed
reference plane. Native field integrals calibrate the nominal kick, but no
trajectory is integrated through its thickness, overlap or material circuit.
The physical gun source, specimen and receiver remain separate executors.
This is not a qualified reduction of the new high-order potential solver.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from temsim.physics.lens_field_provider import CoordinateRegistration


@dataclass(frozen=True, slots=True)
class ThinOpticalAction:
    key: str
    component_key: str
    local_z_m: float
    registration: CoordinateRegistration
    power_m1: float = 0.
    rotation_rad: float = 0.
    # Integrated q*A_z/p_ref polynomial: its transverse gradient is a kick.
    polynomial: tuple = ()
    magnetic: bool = True
    # Preserve the authoritative native millimetre plane. An SI round trip can
    # move an action across an exact checkpoint by one floating-point step.
    native_reference_z_mm: float | None = None

    def __post_init__(self):
        if any(type(v) is not str or not v or v != v.strip() for v in (self.key, self.component_key)):
            raise ValueError("Thin action identities must be nonempty strings")
        if type(self.registration) is not CoordinateRegistration or type(self.magnetic) is not bool:
            raise ValueError("Thin actions require a rigid registration and Boolean magnetic policy")
        if any(isinstance(v, (bool, str)) or not np.isfinite(v)
               for v in (self.local_z_m, self.power_m1, self.rotation_rad)):
            raise ValueError("Thin action plane, power and rotation must be finite")
        if self.native_reference_z_mm is not None:
            value = self.native_reference_z_mm
            if (isinstance(value, (bool, np.bool_, str)) or not np.isfinite(value)
                    or float(value)*1e-3 != self.local_z_m):
                raise ValueError("Thin native reference plane must match its captured SI coordinate")
            object.__setattr__(self, "native_reference_z_mm", float(value))
        rows = []
        for i, j, value in self.polynomial:
            if (type(i) is not int or type(j) is not int or min(i, j) < 0
                    or not 1 <= i+j <= 3 or isinstance(value, (bool, str)) or not np.isfinite(value)):
                raise ValueError("Thin action polynomials require finite dipole, quadrupole or hexapole terms")
            rows.append((i, j, float(value)))
        if len({(i, j) for i, j, _ in rows}) != len(rows):
            raise ValueError("Thin action polynomial powers must be unique")
        object.__setattr__(self, "polynomial", tuple(rows))

    @property
    def z_mm(self):
        if self.native_reference_z_mm is not None:
            return float(self.registration.rotation_array[2, 2]*self.native_reference_z_mm
                         + self.registration.origin_global_m[2]*1e3)
        return float((self.registration.rotation_array @ (0., 0., self.local_z_m)
                      + self.registration.origin_array_m)[2]*1e3)

    def apply(self, x, tx, y, ty, momentum_scale):
        rotation = self.registration.rotation_array
        points = np.column_stack((x, y, np.full_like(x, self.z_mm*1e-3)))
        local = (points-self.registration.origin_array_m) @ rotation
        direction = np.column_stack((tx, ty, np.ones_like(x))) @ rotation
        active = np.isfinite(points).all(axis=1) & np.isfinite(direction).all(axis=1)
        if np.any(active & (direction[:, 2] <= 1e-10)):
            raise ValueError(f"{self.key}: thin plane leaves the forward local-axis domain")
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            direction /= direction[:, 2, None]
            plane = local + (self.local_z_m-local[:, 2, None])*direction
            scale = momentum_scale if self.magnetic else np.ones_like(momentum_scale)
            direction[:, :2] -= (self.power_m1*scale**2)[:, None]*plane[:, :2]
            for i, j, coefficient in self.polynomial:
                if i:
                    direction[:, 0] += scale*coefficient*i*plane[:, 0]**(i-1)*plane[:, 1]**j
                if j:
                    direction[:, 1] += scale*coefficient*j*plane[:, 0]**i*plane[:, 1]**(j-1)
            if self.rotation_rad:
                angle = self.rotation_rad*scale
                co, si = np.cos(angle), np.sin(angle)
                for pair in (plane, direction):
                    a, b = pair[:, 0].copy(), pair[:, 1].copy()
                    pair[:, 0], pair[:, 1] = co*a-si*b, si*a+co*b
            outgoing = direction @ rotation.T
            if np.any(active & (outgoing[:, 2] <= 1e-10)):
                raise ValueError(f"{self.key}: thin action leaves the forward global-axis domain")
            outgoing /= outgoing[:, 2, None]
            global_plane = plane @ rotation.T + self.registration.origin_array_m
            global_plane += (self.z_mm*1e-3-global_plane[:, 2, None])*outgoing
        return global_plane[:, 0], outgoing[:, 0], global_plane[:, 1], outgoing[:, 1]


def gaussian_integrals(terms):
    """Exact all-axis integrals of B and B squared, including within-lens cross terms."""
    first = sum(a*s*math.sqrt(2*math.pi) for a, _, s in terms)
    second = 0.
    for a, c, s in terms:
        for b, d, t in terms:
            width2 = s*s+t*t
            second += a*b*math.sqrt(2*math.pi)*s*t/math.sqrt(width2)*math.exp(-.5*(c-d)**2/width2)
    return first, second


def capture_thin_actions(state, events=(), *, include_hexapole=True, diagnostic_dipoles=None):
    from temsim.component_keys import CONDENSER_LENS_KEYS
    from temsim.lens_pose import lens_pose_registration
    from temsim.physics.core import electron
    from temsim.physics.lens_field_provider import _analytic_gaussian_terms
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    from temsim.physics.plan_contract import reject_repeated_physical_events

    charge, momentum, _ = electron(state)
    actions = []
    if hasattr(state, "sync_objective"):
        state.sync_objective()
    for lens in state.lenses:
        if not lens.enabled:
            continue
        native = state.condenser_system[lens.key] if lens.key in CONDENSER_LENS_KEYS else lens
        integral, square = gaussian_integrals(_analytic_gaussian_terms(native))
        actions.append(ThinOpticalAction(lens.key, lens.key, float(lens.z_mm)*1e-3,
            lens_pose_registration(state, lens.key), (charge/(2*momentum))**2*square,
            -charge/(2*momentum)*integral, native_reference_z_mm=float(lens.z_mm)))
    remaining = list(events)
    coils = []
    for field in (column_dipole_fields(state) if diagnostic_dipoles is None else diagnostic_dipoles):
        event = (field.event_z_mm, field.event_dx_rad, field.event_dy_rad)
        for index, row in enumerate(remaining):
            if tuple(row) == event:
                remaining.pop(index)
                coils.append(field)
                break
    reject_repeated_physical_events(coils, remaining)
    components = {item.key: item for item in (*getattr(state, "stigmators", ()),
                  *getattr(state, "corrector_elements", ()), *getattr(state, "deflectors", ()))}
    for field in capture_posed_column_fields(state, include_aligned=True,
                                             include_hexapole=include_hexapole, dipoles=coils):
        if field.envelope_kind == "gaussian":
            lo, hi = np.asarray(field.native_support_mm)*1e-3
            scale = field.sigma_m*math.sqrt(math.pi/2)*(math.erf((hi-field.center_z_m)/(math.sqrt(2)*field.sigma_m))
                -math.erf((lo-field.center_z_m)/(math.sqrt(2)*field.sigma_m)))
        else:
            scale = np.ptp(field.native_support_mm)*1e-3
        actions.append(ThinOpticalAction(field.lens_key, field.component_key,
            field.center_z_m, field.registration, polynomial=tuple(
                (i, j, charge/momentum*scale*value) for i, j, value in field.polynomial_terms),
            native_reference_z_mm=float(field.event_z_mm if field.event_z_mm is not None
                                        else components[field.component_key].z_mm)))
    for index, (z, dx, dy) in enumerate(remaining):
        actions.append(ThinOpticalAction(f"external:{index}", f"external:{index}", float(z)*1e-3,
            CoordinateRegistration(), polynomial=((1, 0, float(dx)), (0, 1, float(dy))), magnetic=False,
            native_reference_z_mm=float(z)))
    return tuple(actions)


def capture_all_thin_actions(state, events=(), *, include_hexapole=True,
                             diagnostic_dipoles=None):
    """Capture column and static blanker actions with their shared ownership."""
    actions = list(capture_thin_actions(
        state, events, include_hexapole=include_hexapole,
        diagnostic_dipoles=diagnostic_dipoles))
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None and nanopulser.installed:
        for index, (z, dx, dy) in enumerate(nanopulser.kick_events(state.beam_voltage_kv)):
            actions.append(ThinOpticalAction(f"blanker:{index}", "nanopulser", z*1e-3,
                CoordinateRegistration(), polynomial=((1, 0, dx), (0, 1, dy)), magnetic=False,
                native_reference_z_mm=float(z)))
    return tuple(actions)


def build_thin_plan(state, z0, z1, events=(), *, include_hexapole=True,
                    save_z_mm=(), checkpoint_z_mm=(), particle_medium=False,
                    diagnostic_dipoles=None):
    from temsim.physics.core import AxialPropagationPlan, _frozen_array, electron
    from temsim.physics.physical_scene import capture_physical_scene
    from temsim.physics.interaction_registry import InteractionRecord, InteractionRegistry
    from temsim.physics.plan_contract import SCHEMA, _seal
    from temsim.immutable_json import json_digest

    if not np.isfinite((z0, z1)).all() or z1 <= z0:
        raise ValueError("Thin propagation requires finite increasing planes")
    gun = getattr(state, "electron_gun", None)
    if gun is not None and z0 < float(gun.exit_plane_z_mm):
        raise ValueError("Ideal Optics column transport starts at the gun exit; use the physical gun source executor upstream")
    if particle_medium:
        raise ValueError("Ideal Optics does not model gas collisions; select Finite-Field Optics")
    actions = capture_all_thin_actions(state, events, include_hexapole=include_hexapole,
                                       diagnostic_dipoles=diagnostic_dipoles)
    nanopulser = getattr(state, "nanopulser", None)
    if nanopulser is not None and nanopulser.installed:
        # The shared capture owns the static kicks; retain the physical stop.
        save_z_mm = (*save_z_mm, nanopulser.stop_z_mm)
    actions = tuple(sorted((a for a in actions if z0 <= a.z_mm <= z1), key=lambda a: a.z_mm))
    history = float(getattr(state, "history_step_mm", 2.))
    if not math.isfinite(history) or history <= 0:
        raise ValueError("Thin plotting-history spacing must be positive")
    count = max(1, math.ceil((z1-z0)/history))
    if count > 1_000_000:
        raise ValueError("Thin plotting history exceeds the node budget")
    geometry = []
    for aperture in getattr(state, "apertures", ()):
        if getattr(aperture, "enabled", True) and getattr(aperture, "installed", True):
            geometry.append(aperture.z_mm)
    for segment in getattr(getattr(state, "_resolved_assembly", None), "vacuum_bore_segments", ()):
        geometry.extend((segment.start_z_mm, segment.end_z_mm))
    observation = getattr(state, "observation_plane_z_mm", None)
    requested = (*save_z_mm, *checkpoint_z_mm, *geometry,
                 *((observation,) if observation is not None else ()))
    if not np.isfinite(requested).all():
        raise ValueError("Thin observation and boundary planes must be finite")
    z = np.unique(np.r_[np.linspace(z0, z1, count+1), [a.z_mm for a in actions],
                         [v for v in requested if z0 <= v <= z1]])
    checkpoint_index = np.searchsorted(z, sorted(set(v for v in checkpoint_z_mm if z0 <= v <= z1)))
    source = json_digest([asdict(a) for a in actions])
    solver = "ideal-thin-events-v1:"+source
    numerical = json_digest({"solver": solver, "z": z.tolist(), "checkpoints": checkpoint_index.tolist(),
                             "reference_momentum": electron(state)[1]})
    scene = capture_physical_scene(state)
    records = tuple(InteractionRecord(
        interaction_id="thin:"+a.key, owner_id=scene.owner_for(a.component_key) or "external:"+a.key,
        effect_family="thin_optical_action", representation="reduced", interval_mm=(a.z_mm, a.z_mm),
        covered_effects=(a.key+":thin_action",), source_digest=source,
        logical_drive_ids=(a.component_key,)) for a in actions)
    contract = _seal({"schema": SCHEMA, "mode": "ideal_thin", "interval_mm": (float(z0), float(z1)),
        "scene": scene.to_dict(), "interactions": InteractionRegistry(records).to_dict(),
        "reference_planes_mm": {"thin:"+a.key: a.z_mm for a in actions}, "numerical_digest": numerical,
        "scope": "Thin column optics; native integrated excitation, exact drifts and existing physical clipping",
        "limitations": ("No finite-thickness or overlap dynamics", "No material field solve or empirical Cs/Cc screens",
                        "Gun emission/acceleration, specimen and receiver use their separate existing models",
                        "Not a qualified surrogate of the high-order field solver")})
    nodes = {name: _frozen_array(np.zeros(len(z))) for name in (
        "magnetic_t", "sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3",
        "cs_kick_m3", "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")}
    midpoint = {"midpoint_"+name: _frozen_array(np.zeros(len(z)-1)) for name in (
        "magnetic_t", "sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3")}
    return AxialPropagationPlan(z_mm=_frozen_array(z), step_m=_frozen_array(np.diff(z)*1e-3),
        **nodes, **midpoint, save_index=_frozen_array(np.arange(len(z)), np.int64),
        checkpoint_index=_frozen_array(checkpoint_index, np.int64), solver_signature=solver,
        signature=json_digest({"numerical": numerical, "contract": contract["digest"]}),
        dipole_bx_t=_frozen_array(np.zeros(3*(len(z)-1))), dipole_by_t=_frozen_array(np.zeros(3*(len(z)-1))),
        reference_momentum_kg_m_s=electron(state)[1], physics_contract=contract, thin_actions=actions)


def execute_thin_plan(state, plan, x, tx, y, ty, energy_offset_ev=None, *, start_index=0,
                      include_initial_plane_kicks=True, defer_nonfinite_until_clipping=False,
                      medium_transport=None, initial_time_s=None, return_flight_times=False,
                      initial_kinetic_energy_ev=None, energy_output=None):
    from temsim.physics.core import PropagationCheckpoints, E, M, C, _record_active_backend
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.physics.compute_backend import normalise_backend, GPUExecutionError
    validate_thin_plan(plan)
    if medium_transport is not None:
        raise ValueError("Ideal Optics does not execute residual-medium collisions")
    if normalise_backend(getattr(state, "acceleration_backend", "Auto")) == "Require GPU":
        raise GPUExecutionError("unsupported_stage", "Ideal Optics uses the CPU event executor")
    arrays = [np.array(a, dtype=float, copy=True) for a in (x, tx, y, ty)]
    if any(a.ndim != 1 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Thin input phase-space arrays must be matching vectors")
    if not defer_nonfinite_until_clipping and not np.isfinite(arrays).all():
        raise ValueError("Thin input phase space must be finite")
    if isinstance(start_index, bool) or int(start_index) != start_index or not 0 <= start_index < len(plan.z_mm):
        raise ValueError("Thin resume index is out of range")
    energy = np.broadcast_to(np.asarray(state.beam_voltage_kv*1000.+np.asarray(
        0. if energy_offset_ev is None else energy_offset_ev) if initial_kinetic_energy_ev is None
        else initial_kinetic_energy_ev, dtype=float), arrays[0].shape).copy()
    active = np.isfinite(arrays).all(axis=0)
    if np.any((~np.isfinite(energy) | (energy <= 0)) & (active | (not defer_nonfinite_until_clipping))):
        raise ValueError("Thin entrance kinetic energy must be positive and finite")
    momentum = np.sqrt((energy*E)**2+2*energy*E*M*C*C)/C
    scale = plan.reference_momentum_kg_m_s/momentum
    inv_speed = np.sqrt(M*M+(momentum/C)**2)/momentum
    time = np.full_like(energy, np.nan) if initial_time_s is None else np.array(initial_time_s, dtype=float, copy=True)
    if time.shape != energy.shape or np.isinf(time).any() or np.any(time < 0):
        raise ValueError("Thin flight times must be nonnegative or unknown")
    save = np.asarray(plan.save_index); save = save[save >= start_index]
    checkpoints = np.asarray(plan.checkpoint_index); checkpoints = checkpoints[checkpoints >= start_index]
    # Match the existing compact display history; resumable states stay float64.
    histories = [np.empty((len(save), len(energy)), dtype=np.float32) for _ in range(4)]
    captures = [np.empty((len(checkpoints), len(energy))) for _ in range(4)]
    times, ct = np.empty((len(save), len(energy))), np.empty((len(checkpoints), len(energy)))
    grouped = {}
    for action in plan.thin_actions:
        grouped.setdefault(int(np.searchsorted(plan.z_mm, action.z_mm)), []).append(action)
    si = ci = 0
    for j in range(int(start_index), len(plan.z_mm)):
        check_tuning_cancelled(state)
        if j != start_index or include_initial_plane_kicks:
            for action in grouped.get(j, ()):
                arrays = list(action.apply(*arrays, scale))
        if not defer_nonfinite_until_clipping and not np.isfinite(arrays).all():
            raise ValueError("Thin propagation overflowed; reduce the accepted ray support")
        if si < len(save) and j == save[si]:
            for out, a in zip(histories, arrays): out[si] = a
            times[si] = time
            si += 1
        if ci < len(checkpoints) and j == checkpoints[ci]:
            for out, a in zip(captures, arrays): out[ci] = a
            ct[ci] = time
            ci += 1
        if j < len(plan.step_m):
            h = plan.step_m[j]
            arrays[0] += h*arrays[1]
            arrays[2] += h*arrays[3]
            time += h*np.sqrt(1+arrays[1]**2+arrays[3]**2)*inv_speed
    energy_rows = np.broadcast_to(energy, (len(save), len(energy))).copy()
    if energy_output is not None: energy_output.append(energy_rows)
    cp = PropagationCheckpoints(plan.z_mm[checkpoints], *captures,
        flight_time_s=ct if return_flight_times else None,
        kinetic_energy_ev=np.broadcast_to(energy, (len(checkpoints), len(energy))))
    _record_active_backend(state, "CPU", "Exact drifts and thin optical events")
    from temsim.physics.backend_execution import record_backend
    record_backend("column", getattr(state, "acceleration_backend", "Auto"), "CPU",
                   reason="Exact drifts and thin optical events", retried=False)
    state._last_ray_device_receipt = None
    result = (plan.z_mm[save], *histories)
    return (*result, times, cp) if return_flight_times else (*result, cp)


def validate_thin_plan(plan):
    """A damaged archive cannot silently replace missing events by drifts."""
    from temsim.immutable_json import json_digest
    expected = "ideal-thin-events-v1:"+json_digest([asdict(a) for a in plan.thin_actions])
    if plan.solver_signature != expected:
        raise ValueError("Thin plan action identity mismatch")
    if plan.mapped_fields or plan.electric_field is not None or plan.posed_spherical_kicks or plan.intrinsic_aberration_kicks:
        raise ValueError("Thin events cannot be combined with distributed or empirical actions")
    if any(np.any(getattr(plan, name)) for name in (
        "magnetic_t", "sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3",
        "midpoint_magnetic_t", "midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
        "midpoint_hex_normal_m3", "midpoint_hex_skew_m3", "cs_kick_m3", "thin_power_m1",
        "thin_rotation_rad", "kick_x_rad", "kick_y_rad", "dipole_bx_t", "dipole_by_t")):
        raise ValueError("Thin plan has unexpected additional numerical actions")
