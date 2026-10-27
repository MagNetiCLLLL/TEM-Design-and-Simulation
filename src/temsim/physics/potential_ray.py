"""Opt-in forward Lorentz rays consuming the same immutable potentials as waves.

This bounded transport API has no empirical aberration screens, focusing
substitutions, material scattering, or hidden source normalization. Static
potential differences set each ray's relativistic momentum. Turning particles
require time-domain transport and are rejected here. Explicit stop planes
retain lost population and incident weights; a caller must supply geometry.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
from scipy.constants import c, e, m_e
from scipy.integrate import solve_ivp

from temsim.immutable_json import freeze_json
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


@dataclass(frozen=True)
class PotentialRayResult:
    z_m: np.ndarray
    xy_slopes: np.ndarray
    alive: np.ndarray
    weights: np.ndarray
    reduced_action_m: np.ndarray
    metadata: object


def _frozen(value):
    array = np.asarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def trace_potential_rays(snapshot, initial_xy_slopes, z_m, *, energy_ev,
                         maximum_step_m, weights=None, stop_planes=(), method="DOP853",
                         relative_tolerance=1e-10, position_tolerance_m=1e-13,
                         slope_tolerance_rad=1e-11, cancelled=lambda: False,
                         sampling_backend="snapshot"):
    """Trace N rows (x,y,dx/dz,dy/dz) to explicit increasing axial planes.

    Energy is initial kinetic energy per ray, in eV. stop_planes entries are
    (node index, centre_x_m, centre_y_m, radius_m); these circular openings are
    applied at their declared planes, with no post-stop recentring. Detector
    synthesis, source preparation and additional material surfaces are separate
    consumers. The reduced canonical action is integral(P.dr)/p_initial minus
    geometric on-axis distance, in metres; it is not a coherent wave solution.
    """
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("Ray execution requires an immutable potential snapshot")
    if sampling_backend not in {"snapshot", "native_compiled"}:
        raise ValueError("Potential sampling backend must be snapshot or native_compiled")
    for row in snapshot.contributions:
        meta = row.potential.to_metadata() if hasattr(row.potential, "to_metadata") else row.potential.metadata
        if (meta.get("static") is False or meta.get("dynamic_drive_frozen", False)
                or meta.get("provenance", {}).get("dynamic_drive_frozen", False)
                or bool(getattr(getattr(row.potential, "field", None), "dynamic", False))):
            raise ValueError("A frozen dynamic drive is not a static Lorentz field")
    sample = snapshot.sample
    actual_sampling_backend = "snapshot"
    if sampling_backend == "native_compiled":
        from temsim.physics.potential_native_sampler import compile_native_sampler
        sampler = compile_native_sampler(snapshot)
        sampler.require_snapshot(snapshot)
        sample = sampler.sample
        actual_sampling_backend = sampler.backend
    raw = np.asarray(initial_xy_slopes)
    nodes = np.asarray(z_m)
    if raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1] != 4 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Initial rays must be finite N by 4 real coordinates and slopes")
    if (nodes.dtype.kind not in "iuf" or nodes.ndim != 1 or len(nodes) < 2
            or not np.isfinite(nodes).all() or not np.all(np.diff(nodes) > 0)):
        raise ValueError("Observation planes must be finite and increasing")
    nodes = np.asarray(nodes, dtype=float)
    for value in (maximum_step_m, relative_tolerance, position_tolerance_m, slope_tolerance_rad):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
            raise ValueError("Integration step and tolerances must be positive finite scalars")
    if method not in {"RK4", "DOP853"}:
        raise ValueError("Potential ray method must be RK4 or DOP853")
    energy_input = np.asarray(energy_ev)
    if energy_input.dtype.kind not in "iuf":
        raise ValueError("Initial kinetic energies must be real numerical values")
    energies = np.broadcast_to(np.asarray(energy_input, dtype=float), (len(raw),)).copy()
    current = np.asarray(raw, dtype=float).copy()
    weight_input = np.ones(len(raw)) if weights is None else np.asarray(weights)
    if weight_input.dtype.kind not in "iuf":
        raise ValueError("Incident weights must be real numerical values")
    weight = np.asarray(weight_input, dtype=float).copy()
    if not np.isfinite(energies).all() or np.any(energies <= 0):
        raise ValueError("Initial kinetic energies must be finite and positive")
    if weight.shape != (len(raw),) or not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("Incident ray weights must be finite and nonnegative")
    openings = {}
    for row in stop_planes:
        if not isinstance(row, (tuple, list)) or len(row) != 4 or type(row[0]) is not int or not 0 <= row[0] < len(nodes):
            raise ValueError("Circular opening must belong to a declared observation plane")
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in row[1:])
                or not np.isfinite(row[1:]).all() or row[3] <= 0):
            raise ValueError("Circular opening requires a finite centre and positive radius")
        openings.setdefault(row[0], []).append(tuple(row[1:]))
    start = np.column_stack((current[:, :2], np.full(len(current), nodes[0])))
    initial_phi = sample(start)["phi_v"]
    kinetic = energies*e
    p_initial = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
    alive = np.ones(len(raw), dtype=bool)
    action = np.zeros(len(raw))
    history = np.full((len(nodes), len(raw), 4), np.nan)
    action_history = np.full((len(nodes), len(raw)), np.nan)
    live_history = np.zeros((len(nodes), len(raw)), dtype=bool)
    evaluations = 0

    def observe(index):
        nonlocal alive
        for cx, cy, radius in openings.get(index, ()):
            alive &= np.hypot(current[:, 0]-cx, current[:, 1]-cy) <= radius
        history[index, alive] = current[alive]
        action_history[index, alive] = action[alive]
        live_history[index] = alive

    observe(0)
    for index, (lower, upper) in enumerate(zip(nodes[:-1], nodes[1:])):
        if cancelled():
            raise InterruptedError("Potential ray propagation cancelled")
        selected = np.flatnonzero(alive)
        if not len(selected):
            observe(index+1)
            continue
        def derivative(z, flat):
            nonlocal evaluations
            if cancelled():
                raise InterruptedError("Potential ray propagation cancelled")
            values = flat.reshape(len(selected), 5)
            # Compact native Az fields use the inside value exactly on a
            # support face. An endpoint shared by two integration intervals
            # must use each interval's own one-sided limit; otherwise RK4
            # counts a finite endpoint kick in the exterior interval too.
            # Observation coordinates remain at the exact requested plane.
            field_z = np.nextafter(z, upper if z == lower else lower) if z in (lower, upper) else z
            points = np.column_stack((values[:, :2], np.full(len(selected), field_z)))
            field = sample(points)
            energy = (energies[selected]+field["phi_v"]-initial_phi[selected])*e
            if not np.isfinite(energy).all() or np.any(energy <= 0):
                raise ValueError("Trajectory leaves the positive-energy forward domain")
            p = np.sqrt(energy*(energy+2*m_e*c*c))/c
            tx, ty = values[:, 2:4].T
            direction_norm = np.sqrt(1+tx*tx+ty*ty)
            pz = p/direction_norm
            gamma_mass = (energy+m_e*c*c)/c**2
            bx, by, bz = field["B_t"].T
            ex, ey, ez = field["E_v_per_m"].T
            magnetic = -e/pz
            electric = -e*gamma_mass/pz**2
            dtx = magnetic*(ty*bz-(1+tx*tx)*by+tx*ty*bx)+electric*(ex-tx*ez)
            dty = magnetic*((1+ty*ty)*bx-tx*bz-tx*ty*by)+electric*(ey-ty*ez)
            # Stable longitudinal excess for weak slopes. The momentum ratio
            # also retains real electric energy gain, without refocusing rays.
            excess = (tx*tx+ty*ty)/(direction_norm+1)
            pratio = p/p_initial[selected]
            canonical = -e*np.sum(field["A_t_m"]*np.column_stack((tx, ty, np.ones(len(tx)))), axis=1)
            ds = (pratio-1)+pratio*excess+canonical/p_initial[selected]
            evaluations += 1
            return np.column_stack((tx, ty, dtx, dty, ds)).ravel()
        initial = np.column_stack((current[selected], action[selected])).ravel()
        if method == "DOP853":
            tolerances = np.tile((position_tolerance_m, position_tolerance_m,
                                 slope_tolerance_rad, slope_tolerance_rad, position_tolerance_m), len(selected))
            solution = solve_ivp(derivative, (lower, upper), initial, method="DOP853",
                                 max_step=maximum_step_m, rtol=relative_tolerance, atol=tolerances)
            if not solution.success:
                raise ValueError("Potential ray integration failed: "+solution.message)
            final = solution.y[:, -1].reshape(len(selected), 5)
        else:
            intervals = max(1, math.ceil((upper-lower)/maximum_step_m))
            axial = np.linspace(lower, upper, intervals+1)
            final_flat = initial
            for a, b in zip(axial[:-1], axial[1:]):
                h = b-a
                first = derivative(a, final_flat)
                second = derivative(a+h/2, final_flat+h*first/2)
                third = derivative(a+h/2, final_flat+h*second/2)
                fourth = derivative(b, final_flat+h*third)
                final_flat = final_flat+h*(first+2*second+2*third+fourth)/6
            final = final_flat.reshape(len(selected), 5)
        current[selected], action[selected] = final[:, :4], final[:, 4]
        if not np.isfinite(final).all():
            raise ValueError("Potential ray integration produced non-finite state")
        observe(index+1)
    metadata = freeze_json({"snapshot_digest": snapshot.numerical_digest, "method": method,
        "input_weight": float(weight.sum()), "output_weight": float(weight[alive].sum()),
        "stopped_weight": float(weight[~alive].sum()), "field_evaluations": evaluations,
        "maximum_step_m": float(maximum_step_m), "relative_tolerance": float(relative_tolerance),
        "position_tolerance_m": float(position_tolerance_m), "slope_tolerance_rad": float(slope_tolerance_rad),
        "scope": "static forward Lorentz rays with explicit circular stop planes",
        "interval_endpoint_field_policy": "one-sided interior limit at declared integration boundaries",
        "potential_sampling_backend": actual_sampling_backend,
        "empirical_aberrations_applied": False, "normalization_applied": False,
        "action_scope": "reduced canonical ray action; not a coherent wave or caustic phase",
        "production_admission": False})
    return PotentialRayResult(_frozen(nodes), _frozen(history), _frozen(live_history), _frozen(weight),
                              _frozen(action_history), metadata)
