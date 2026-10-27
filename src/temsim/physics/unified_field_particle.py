"""Production forward Lorentz transport on a shared immutable runtime.

The gun and specimen keep their existing source/material responsibilities.
This column integrator owns finite magnetic/electric action and captured
absorbing mechanics exactly once. It never applies empirical Cs/Cc or an
equivalent Image map. RK4 accuracy and chord contact require step refinement.
"""
from dataclasses import dataclass
import math
import weakref

import numpy as np
from scipy.constants import c, e, m_e

from temsim.immutable_json import freeze_json, json_digest
from temsim.physics.unified_field_runtime import get_unified_field_runtime


_STOPS = {}
MAX_UNIFIED_HISTORY_BYTES = 2*1024**3


def _sampler(snapshot):
    """Use the shared retained-potential sampler with legacy native dispatch."""
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    return compile_unified_potential_sampler(snapshot)


def build_unified_plan(state, z0, z1, events=(), *, save_z_mm=(), checkpoint_z_mm=(),
                       maximum_step_mm=None, particle_medium=False, diagnostic_dipoles=None,
                       _captured_runtime=None):
    from temsim.physics.core import AxialPropagationPlan, _frozen_array, electron
    from temsim.physics.plan_contract import SCHEMA, _seal
    from temsim.physics.interaction_registry import InteractionRecord, InteractionRegistry
    from temsim.physics.instrument_magnetic import active_column_events
    if particle_medium or diagnostic_dipoles is not None:
        raise ValueError("Unified fields do not accept residual-medium or synthetic dipole overrides")
    actual = list(active_column_events(state))
    for event in events:
        event = tuple(event)
        if event in actual:
            actual.remove(event)
        elif event[1] or event[2]:
            raise ValueError("Unified fields require actual physical drives, not extra thin steering events")
    runtime = get_unified_field_runtime(state) if _captured_runtime is None else _captured_runtime
    runtime.require_state(state)
    if not np.isfinite((z0, z1)).all() or z1 <= z0 or z0*.001 < runtime.start_z_m or z1*.001 > runtime.stop_z_m:
        raise ValueError("Unified column interval is outside the captured physical source/receiver span")
    step = float(state.step_mm)
    if maximum_step_mm is not None:
        step = min(step, float(maximum_step_mm))
    if not math.isfinite(step) or step <= 0:
        raise ValueError("Unified particle step must be positive and finite")
    count = max(1, math.ceil((z1-z0)/step))
    if count > 1_000_000:
        raise ValueError("Unified particle interval exceeds the integration node budget")
    faces = []
    for row in runtime.snapshot.contributions:
        base = getattr(row.potential, "base", None)
        field = getattr(base, "field", None)
        if field is None:
            field = getattr(getattr(base,"base",None),"field",None)
        if field is not None:
            registration = field.registration
            if np.array_equal(registration.rotation_array, np.eye(3)):
                faces.extend(float(v)+registration.origin_array_m[2]*1000 for v in field.native_support_mm)
    for piece in runtime.geometry.pieces:
        faces.extend(piece.event_z_mm())
    faces.extend(float(plane.z_mm) for plane in getattr(state,"recording_planes",()) if plane.inserted)
    requested = (*save_z_mm, *checkpoint_z_mm, *faces)
    if not np.isfinite(requested).all():
        raise ValueError("Unified observation planes must be finite")
    z = np.unique(np.r_[np.linspace(z0, z1, count+1), [v for v in requested if z0 <= v <= z1]])
    # Keep every integration node for exact mechanical stop and detector views.
    cp = np.searchsorted(z, sorted(set((z0, z1, *(v for v in checkpoint_z_mm if z0 <= v <= z1)))))
    numerical = json_digest({"runtime": runtime.digest, "z_mm": z.tolist(), "checkpoints": cp.tolist(), "integrator": "Lorentz-RK4"})
    records = tuple(InteractionRecord(interaction_id=row.source_id, owner_id=row.owner_ids[0],
        effect_family="electromagnetic_potential", representation="field", interval_mm=(z0,z1),
        covered_effects=tuple(owner+":"+row.kind for owner in row.owner_ids), source_digest=row.potential.digest)
        for row in runtime.snapshot.contributions)
    contract = _seal({"schema": SCHEMA, "mode": "unified", "interval_mm": (float(z0),float(z1)),
        "scene": runtime.snapshot.scene.to_dict(), "interactions": InteractionRegistry(records).to_dict(),
        "unified_capture": {"runtime":runtime.digest,"field":runtime.snapshot.numerical_digest,"geometry":runtime.geometry.digest},
        "reference_planes_mm": {}, "numerical_digest": numerical,
        "scope": "Shared captured potentials and finite absorbing mechanics from the executed gun exit",
        "limitations": ("Forward Lorentz RK4 with actual time and induced electric work", "No empirical aberration or equivalent image action",
                        "Separate physical gun, specimen and receiver responsibilities")})
    nodes = {name: _frozen_array(np.zeros(len(z))) for name in (
        "magnetic_t", "sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3",
        "cs_kick_m3", "thin_power_m1", "thin_rotation_rad", "kick_x_rad", "kick_y_rad")}
    mid = {"midpoint_"+name:_frozen_array(np.zeros(len(z)-1)) for name in (
        "magnetic_t", "sx_m2", "sy_m2", "sxy_m2", "hex_normal_m3", "hex_skew_m3")}
    return AxialPropagationPlan(z_mm=_frozen_array(z), step_m=_frozen_array(np.diff(z)*.001),
        **nodes, **mid, save_index=_frozen_array(np.arange(len(z)),np.int64), checkpoint_index=_frozen_array(cp,np.int64),
        solver_signature="unified-potential-rk4-v1:"+runtime.digest, signature=numerical,
        dipole_bx_t=_frozen_array(np.zeros(3*(len(z)-1))), dipole_by_t=_frozen_array(np.zeros(3*(len(z)-1))),
        reference_momentum_kg_m_s=electron(state)[1], physics_contract=contract, unified_runtime=runtime)


def _active_pieces(geometry, lower, upper):
    result = []
    for host in geometry.boundaries:
        for piece in host.pieces:
            r = piece.registration.rotation_array
            if np.array_equal(r[2], (0.,0.,1.)) and np.array_equal(r[:,2], (0.,0.,1.)):
                offset = piece.registration.origin_array_m[2]
                if upper < piece.local_start_z_mm*.001+offset or lower > piece.local_end_z_mm*.001+offset:
                    continue
            result.append((host.host_key,piece))
    return result


def _contact(pieces, before, after):
    from temsim.physics.unified_particle_contacts import contact
    return contact(pieces, before, after)


def execute_unified_plan(state, plan, x, tx, y, ty, energy_offset_ev=None, *, start_index=0,
        include_initial_plane_kicks=True, defer_nonfinite_until_clipping=False, medium_transport=None,
        initial_time_s=None, return_flight_times=False, initial_kinetic_energy_ev=None, energy_output=None,
        unified_resume=None):
    from temsim.physics.core import PropagationCheckpoints, _frozen_array, _record_active_backend
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.physics.transport_progress import report_transport_progress
    from temsim.physics.compute_backend import normalise_backend, GPUExecutionError
    from temsim.calculation_manifest import solver_source_identity
    from temsim.physics.unified_particle_restart import UnifiedParticleCheckpointState, validate_unified_checkpoint
    from temsim.physics.unified_spatial_dependencies import spatial_runtime_identity
    from temsim.physics.unified_particle_contacts import compile_contact_geometry
    if medium_transport is not None:
        raise ValueError("Unified field transport does not execute residual-medium collisions")
    if normalise_backend(getattr(state,"acceleration_backend","Auto")) == "Require GPU":
        raise GPUExecutionError("unsupported_stage", "Unified potential Lorentz integration currently uses CPU")
    runtime = plan.unified_runtime
    if runtime is None:
        raise ValueError("Unified plan lacks its captured field and geometry")
    expected = {"runtime":runtime.digest,"field":runtime.snapshot.numerical_digest,"geometry":runtime.geometry.digest}
    if plan.physics_contract["unified_capture"] != expected:
        raise ValueError("Unified plan ownership does not match its executable runtime")
    runtime.require_state(state)
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    from temsim.physics.emission_timing import resolve_emission_time_s
    dynamic = type(runtime.snapshot) is DynamicPotentialSnapshot
    epoch = resolve_emission_time_s(state)
    if runtime.metadata.get("emission_time_s",epoch) != epoch:
        raise ValueError("Unified runtime emission epoch differs from the execution input")
    if isinstance(start_index,bool) or int(start_index)!=start_index or not 0 <= start_index < len(plan.z_mm):
        raise ValueError("Unified resume index is invalid")
    raw = [np.asarray(value,dtype=float) for value in (x,y,tx,ty)]
    if any(value.ndim!=1 or value.shape!=raw[0].shape for value in raw):
        raise ValueError("Unified rays require matching phase-space vectors")
    current = np.column_stack(raw)
    active = np.isfinite(current).all(axis=1)
    if not defer_nonfinite_until_clipping and not active.all():
        raise ValueError("Unified input phase space must be finite")
    energy = np.broadcast_to(np.asarray(state.beam_voltage_kv*1000+np.asarray(0 if energy_offset_ev is None else energy_offset_ev)
        if initial_kinetic_energy_ev is None else initial_kinetic_energy_ev,dtype=float),(len(current),)).copy()
    times = np.full(len(current),np.nan) if initial_time_s is None else np.asarray(initial_time_s,dtype=float).copy()
    if times.shape!=energy.shape or np.isinf(times).any() or np.any(times<0):
        raise ValueError("Unified flight times must be nonnegative or unknown")
    z = np.asarray(plan.z_mm[start_index:],dtype=float)
    implementation_digest = solver_source_identity()
    retained = None
    if unified_resume is not None:
        previous, row = unified_resume
        checkpoint_state = validate_unified_checkpoint(previous)
        if (checkpoint_state.implementation_digest != implementation_digest
                or checkpoint_state.emission_time_s != epoch):
            raise ValueError("Unified restart implementation or emission epoch changed")
        retained = checkpoint_state.restored(row, z[0])
        if spatial_runtime_identity(runtime, retained["swept_bounds_m"]) != retained["spatial_dependency_identity"]:
            raise ValueError("Unified restart upstream fields or geometry changed")
        if previous.z_mm[row] != z[0] or retained["active"].shape != active.shape:
            raise ValueError("Unified restart plane or particle population changed")
        for supplied, saved in zip(raw, (previous.x_m, previous.y_m, previous.tx_rad, previous.ty_rad)):
            if not np.array_equal(supplied, saved[row], equal_nan=True):
                raise ValueError("Unified restart must retain exact particle phase space")
        if (not np.array_equal(times, previous.flight_time_s[row], equal_nan=True)
                or not np.array_equal(energy, previous.kinetic_energy_ev[row], equal_nan=True)):
            raise ValueError("Unified restart must retain actual kinetic energy and flight time")
        active = retained["active"]
        energy = retained["reference_energy_ev"]
    if np.any(active & (~np.isfinite(energy) | (energy<=0))):
        raise ValueError("Unified entrance kinetic energies must be positive and finite")
    if dynamic and np.any(active & ~np.isfinite(times)):
        raise ValueError("Time-dependent fields require actual finite entrance flight times")
    required_bytes = len(z)*len(current)*8*6 + len(plan.checkpoint_index)*len(current)*8*12
    if required_bytes > MAX_UNIFIED_HISTORY_BYTES:
        raise ValueError(f"Unified particle history requires {required_bytes:,} bytes, exceeding the "
                         f"{MAX_UNIFIED_HISTORY_BYTES:,}-byte numerical budget; reduce the particle count or requested span")
    raw_sample = _sampler(runtime.snapshot)
    contacts = compile_contact_geometry(runtime.geometry)
    swept_bounds = np.array(((0., 0., z[0]*.001), (0., 0., z[0]*.001))) if retained is None else retained["swept_bounds_m"]
    def include_points(points):
        finite = np.asarray(points)[np.isfinite(points).all(axis=1)]
        if len(finite):
            np.minimum(swept_bounds[0], finite.min(axis=0), out=swept_bounds[0])
            np.maximum(swept_bounds[1], finite.max(axis=0), out=swept_bounds[1])
    def sample(points, elapsed):
        include_points(points)
        return raw_sample(points,epoch+elapsed) if dynamic else raw_sample(points)
    if retained is None:
        induction_work_ev = np.zeros(len(current))
        blocked, labels = np.full(len(current),np.nan), np.full(len(current),"",dtype=object)
        entrance_ids = np.flatnonzero(active)
        entrance = np.c_[current[active,:2],np.full(active.sum(),z[0]*.001)]
        include_points(entrance)
        contact,names = _contact(contacts.active(z[0]*.001,z[0]*.001),entrance,entrance)
        lost = entrance_ids[np.isfinite(contact)]
        active[lost],blocked[lost],labels[lost] = False,z[0],names[np.isfinite(contact)]
        initial_phi = np.full(len(current),np.nan)
        if active.any():
            initial_phi[active] = sample(np.c_[current[active,:2],np.full(active.sum(),z[0]*.001)],times[active])["phi_v"]
    else:
        induction_work_ev = retained["induction_work_ev"]
        blocked, labels = retained["blocked_z_mm"], np.asarray(retained["blocked_key"],dtype=object)
        initial_phi = retained["reference_phi_v"]
    histories = np.full((len(z),len(current),4),np.nan)
    clocks = np.full((len(z),len(current)),np.nan)
    energies = np.full_like(clocks,np.nan)
    energies[0] = energy if retained is None else previous.kinetic_energy_ev[row]
    cp_index = np.asarray(plan.checkpoint_index)
    cp_index = cp_index[cp_index>=start_index]-start_index
    cp_rows = {int(value): index for index,value in enumerate(cp_index)}
    cp_work = np.empty((len(cp_index),len(current)))
    cp_active = np.empty(cp_work.shape,dtype=bool)
    cp_bounds = np.empty((len(cp_index),2,3))
    evaluations = 0
    for index, plane in enumerate(z):
        check_tuning_cancelled(state)
        histories[index], clocks[index] = current, times
        if active.any():
            phi = sample(np.c_[current[active,:2],np.full(active.sum(),plane*.001)],times[active])["phi_v"]
            energies[index,active] = energy[active]+phi-initial_phi[active]+induction_work_ev[active]
        if index in cp_rows:
            cp_row = cp_rows[index]
            cp_work[cp_row], cp_active[cp_row], cp_bounds[cp_row] = induction_work_ev, active, swept_bounds
        if index == len(z)-1:
            break
        lower, upper = plane*.001,z[index+1]*.001
        pieces = contacts.active(lower,upper)
        ids = np.flatnonzero(active)
        if not len(ids):
            continue
        before = current[ids].copy()
        start = np.c_[before[:,:2],np.full(len(ids),lower)]
        def derivative(position, values):
            nonlocal evaluations
            field_z = np.nextafter(position,upper if position==lower else lower) if position in (lower,upper) else position
            points = np.c_[values[:,:2],np.full(len(ids),field_z)]
            include_points(points)
            fraction,_ = _contact(pieces,start,points)
            # A contact exactly at the stage endpoint has zero preceding
            # material path; retain its interior one-sided force and clock.
            valid = ~np.isfinite(fraction) | (fraction == 1.)
            out = np.zeros((len(ids),6))
            out[:,:2] = values[:,2:4]
            if not valid.any():
                return out
            elapsed = times[ids[valid]]+values[valid,4]
            if dynamic:
                runtime.snapshot.require_time_interval(epoch+times[ids[valid]],epoch+elapsed)
            fields = sample(points[valid],elapsed)
            kinetic = (energy[ids[valid]]+fields["phi_v"]-initial_phi[ids[valid]]+values[valid,5])*e
            if np.any(kinetic<=0) or not np.isfinite(kinetic).all():
                raise ValueError("Unified ray left the positive-energy forward domain")
            sx,sy = values[valid,2:4].T
            norm = np.sqrt(1+sx*sx+sy*sy)
            p = np.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
            pz = p/norm
            gamma_mass = (kinetic+m_e*c*c)/(c*c)
            bx,by,bz = fields["B_t"].T
            ex,ey,ez = fields["E_v_per_m"].T
            magnetic,electric = -e/pz,-e*gamma_mass/(pz*pz)
            work = np.sum(fields.get("dA_dt_t_m_per_s",np.zeros((valid.sum(),3)))*np.c_[sx,sy,np.ones(valid.sum())],axis=1)
            out[valid] = np.c_[sx,sy,
                magnetic*(sy*bz-(1+sx*sx)*by+sx*sy*bx)+electric*(ex-sx*ez),
                magnetic*((1+sy*sy)*bx-sx*bz-sx*sy*by)+electric*(ey-sy*ez),norm*gamma_mass/p,work]
            evaluations += 1
            return out
        values = np.c_[before,np.zeros(len(ids)),induction_work_ev[ids]]
        h = upper-lower
        a = derivative(lower,values)
        b = derivative(lower+h/2,values+h*a/2)
        cc = derivative(lower+h/2,values+h*b/2)
        d = derivative(upper,values+h*cc)
        after = values+h*(a+2*b+2*cc+d)/6
        if dynamic:
            runtime.snapshot.require_time_interval(epoch+times[ids],epoch+times[ids]+after[:,4])
        end = np.c_[after[:,:2],np.full(len(ids),upper)]
        include_points(end)
        fraction,hit_labels = _contact(pieces,start,end)
        hit = np.isfinite(fraction)
        current[ids] = after[:,:4]
        times[ids] += after[:,4]
        induction_work_ev[ids] = after[:,5]
        if hit.any():
            lost = ids[hit]
            blocked[lost] = plane+(z[index+1]-plane)*fraction[hit]
            labels[lost] = hit_labels[hit]
            active[lost] = False
            at_plane=ids[hit & (fraction==1.)]
            if len(at_plane):
                phi=sample(np.c_[current[at_plane,:2],np.full(len(at_plane),upper)],times[at_plane])["phi_v"]
                energies[index+1,at_plane]=energy[at_plane]+phi-initial_phi[at_plane]+induction_work_ev[at_plane]
            # Retain the last finite state for inspection; downstream views use
            # the independently recorded contact, never reinterpret it as flux.
        if index % max(1,(len(z)-1)//100) == 0:
            report_transport_progress(f"Unified column | CPU | {len(current):,} electrons | Z {plane:.6g} mm",
                                      completed=index,total=len(z)-1)
    output_z = z.copy()
    _STOPS[id(output_z)] = (weakref.ref(output_z,lambda ref,key=id(output_z):_STOPS.pop(key,None)),
                           blocked.copy(),labels.copy(),runtime.digest)
    # Descending boxes allow the dependency proof to reuse an exact-zero
    # certificate only for smaller, contained upstream regions.
    spatial_ids = tuple(reversed([spatial_runtime_identity(runtime, bounds) for bounds in cp_bounds[::-1]]))
    internal_state = UnifiedParticleCheckpointState(energy,initial_phi,cp_work,cp_active,blocked,
        tuple(str(value) for value in labels),runtime.digest,implementation_digest,epoch,cp_bounds,
        spatial_ids)
    cp = PropagationCheckpoints(z[cp_index],histories[cp_index,:,0],histories[cp_index,:,2],
        histories[cp_index,:,1],histories[cp_index,:,3],clocks[cp_index],energies[cp_index],unified_state=internal_state)
    if energy_output is not None:
        energy_output.append(_frozen_array(energies))
    _record_active_backend(state,"CPU")
    from temsim.physics.backend_execution import record_backend
    record_backend("column",getattr(state,"acceleration_backend","Auto"),"CPU",
                   reason="Shared-potential Lorentz RK4 with captured mechanical boundaries",retried=False)
    state._last_ray_device_receipt=None
    result = (output_z,histories[:,:,0],histories[:,:,2],histories[:,:,1],histories[:,:,3])
    return (*result,clocks,cp) if return_flight_times else (*result,cp)


def merge_unified_stops(z, alive, blocked, labels, selection=slice(None)):
    """Merge one executed geometry receipt without repeating aperture action."""
    receipt = _STOPS.get(id(z))
    if receipt is None or receipt[0]() is not z:
        raise ValueError("Unified trajectory has no executed mechanical stop receipt")
    stop, names = receipt[1][selection],receipt[2][selection]
    alive,blocked,labels = np.asarray(alive,dtype=bool).copy(),np.asarray(blocked,dtype=float).copy(),list(labels)
    replace = np.isfinite(stop) & (~np.isfinite(blocked) | (stop<blocked))
    blocked[replace] = stop[replace]
    for index in np.flatnonzero(replace):
        labels[index] = str(names[index])
    alive &= ~np.isfinite(stop)
    return alive,blocked,labels


def trace_unified_response(state, phase_space_xy_slopes, z0_m, observation_z_m, *,
        energy_ev, flight_time_s, maximum_step_m, coil_perturbations_rad=None):
    """Measure explicit finite-coil responses without replacing a beam source.

    Input columns are x, y, dx/dz, dy/dz in SI. Diagnostic increments change
    the retained uniform Az coefficients of an existing physical coil, never
    a zero-thickness kick. The state, its cache, and mechanical geometry are
    unchanged. Observations contain actual A for canonical momentum analysis.
    """
    from collections.abc import Mapping
    from dataclasses import replace
    from temsim.physics.analytic_potential import AnalyticMagneticPotential
    from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
    from temsim.physics.posed_column_fields import FrozenPosedMultipole
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
    initial = np.asarray(phase_space_xy_slopes, dtype=float)
    targets = np.asarray(observation_z_m, dtype=float)
    if (initial.ndim != 2 or initial.shape[1] != 4 or not len(initial) or not np.isfinite(initial).all()
            or targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all()
            or not np.isfinite(z0_m) or targets[0] <= z0_m or np.any(np.diff(targets) <= 0)
            or not np.isfinite(maximum_step_m) or maximum_step_m <= 0):
        raise ValueError("Unified response requires finite phase space, increasing planes and a positive step")
    runtime = get_unified_field_runtime(state)
    original_digest = runtime.digest
    perturbations = {} if coil_perturbations_rad is None else coil_perturbations_rad
    if not isinstance(perturbations, Mapping):
        raise ValueError("Finite-coil diagnostic increments must map physical coil keys to two angles")
    normalized = {}
    for key, value in perturbations.items():
        angles = np.asarray(value)
        if not isinstance(key, str) or angles.shape != (2,) or angles.dtype.kind not in "iuf" or not np.isfinite(angles).all():
            raise ValueError("Finite-coil diagnostic increments need finite two-angle values in radians")
        normalized[key] = tuple(float(v) for v in angles)
    if normalized:
        if type(runtime.snapshot) is not ElectromagneticPotentialSnapshot:
            raise ValueError("Diagnostic coil responses require an explicitly static private working point")
        rows, found = [], set()
        for row in runtime.snapshot.contributions:
            potential = row.potential
            if row.source_id in normalized:
                if (type(potential) is not AnalyticPotentialDomain
                        or type(potential.base.field) is not FrozenPosedMultipole
                        or potential.base.field.envelope_kind != "uniform"
                        or any(x+y != 1 for x,y,_ in potential.base.field.polynomial_terms)):
                    raise ValueError(f"{row.source_id}: diagnostic increment requires a retained native uniform dipole")
                field = potential.base.field
                length = (field.native_support_mm[1]-field.native_support_mm[0])*.001
                conversion = field.reference_momentum/(-e)/length
                coefficients = {(x,y): value for x,y,value in field.polynomial_terms}
                dx, dy = normalized[row.source_id]
                coefficients[1,0] = coefficients.get((1,0),0.)+conversion*dx
                coefficients[0,1] = coefficients.get((0,1),0.)+conversion*dy
                field = replace(field, polynomial_terms=tuple((x,y,value) for (x,y),value in coefficients.items()))
                potential = AnalyticPotentialDomain(AnalyticMagneticPotential(field, {
                    **dict(potential.base.provenance), "diagnostic_increment_rad": [dx,dy],
                    "diagnostic_scope": "DC finite-coil response; no source or hardware mutation"}))
                found.add(row.source_id)
            rows.append(PotentialContribution(row.source_id,row.owner_ids,potential,row.kind))
        if found != set(normalized):
            raise ValueError("Unknown physical diagnostic coil: "+", ".join(sorted(set(normalized)-found)))
        snapshot = ElectromagneticPotentialSnapshot(runtime.snapshot.scene,tuple(rows),runtime.snapshot.gauge)
        runtime = replace(runtime,snapshot=snapshot,metadata={**dict(runtime.metadata),
            "diagnostic_parent_runtime_digest":original_digest,"diagnostic_coil_increments_rad":normalized})
    target_mm = targets*1000.
    plan = build_unified_plan(state,float(z0_m)*1000.,float(target_mm[-1]),
        checkpoint_z_mm=target_mm,maximum_step_mm=float(maximum_step_m)*1000.,_captured_runtime=runtime)
    energies = []
    result = execute_unified_plan(state,plan,initial[:,0],initial[:,2],initial[:,1],initial[:,3],
        initial_kinetic_energy_ev=energy_ev,initial_time_s=np.broadcast_to(flight_time_s,(len(initial),)),
        return_flight_times=True,energy_output=energies)
    indexes = np.searchsorted(result[0],target_mm)
    if not np.array_equal(result[0][indexes],target_mm):
        raise ValueError("Unified response observations lost their exact plane identity")
    alive,blocked,labels = merge_unified_stops(result[0],np.ones(len(initial),bool),
        np.full(len(initial),np.nan),[""]*len(initial))
    phase = np.stack((result[1][indexes],result[3][indexes],result[2][indexes],result[4][indexes]),axis=-1)
    vector = np.full((*phase.shape[:-1],3),np.nan)
    sampler = _sampler(runtime.snapshot)
    for index, plane in enumerate(targets):
        valid = ~np.isfinite(blocked) | (blocked > target_mm[index])
        if valid.any():
            points = np.c_[phase[index,valid,:2],np.full(valid.sum(),plane)]
            from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
            from temsim.physics.emission_timing import resolve_emission_time_s
            values = (sampler(points,resolve_emission_time_s(state)+result[5][indexes[index],valid])
                      if type(runtime.snapshot) is DynamicPotentialSnapshot else sampler(points))
            vector[index,valid] = values["A_t_m"]
    return {"z_m":targets.copy(),"xy_slopes":phase,"flight_time_s":result[5][indexes],
        "kinetic_energy_ev":energies[0][indexes],"vector_potential_t_m":vector,
        "alive":alive,"blocked_z_m":blocked*.001,"blocked_key":tuple(labels),
        "receipt":freeze_json({"source_identity":runtime.source_identity,"runtime_digest":runtime.digest,
            "swept_bounds_m":result[-1].unified_state.swept_bounds_m[-1].tolist(),
            "parent_runtime_digest":original_digest,"field_digest":runtime.snapshot.numerical_digest,
            "geometry_digest":runtime.geometry.digest,"coil_perturbations_rad":normalized,
            "scope":"Supplied differential phase-space probes; not an electron source",
            "empirical_aberrations_applied":False,"maximum_step_m":float(maximum_step_m)})}
