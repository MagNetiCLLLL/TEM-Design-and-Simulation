"""Opt-in physical-gun -> semiclassical column -> material wave -> ray detector.

This development workflow does not change the production wave admission gate.
It retains the actual gun source, static imposed fields, physical interceptions,
absolute mode current and numerical errors. Exit particles represent a declared
coarse-grained wave measurement, not original tip-particle identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.physics.hybrid_semiclassical import SemiclassicalNumerics, propagate_mode
from temsim.physics.hybrid_exit import sample_exit
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.wave_flux import BeamState


@dataclass(frozen=True)
class HybridGrid:
    """Numerical observation lattice; no lens, beam or current adjustment."""
    z_mm: float
    pixels: int
    width_nm: float
    centre_x_nm: float = 0.
    centre_y_nm: float = 0.

    def plane(self):
        if (type(self.pixels) is not int or self.pixels < 16
                or not np.isfinite((self.z_mm, self.width_nm, self.centre_x_nm, self.centre_y_nm)).all()
                or self.z_mm <= 0 or self.width_nm <= 0):
            raise ValueError("Hybrid observation grids require finite SI-compatible dimensions")
        return PlaneWave(np.zeros((self.pixels, self.pixels), complex),
            np.eye(2)*self.width_nm*1e-9/self.pixels,
            np.array((self.centre_x_nm, self.centre_y_nm))*1e-9)


@dataclass(frozen=True)
class HybridRequest:
    detector_key: str
    specimen_grid: HybridGrid
    source_particles: int = 1024
    exit_particles: int = 4096
    seed: int = 0
    semiclassical: SemiclassicalNumerics = field(default_factory=SemiclassicalNumerics)
    wave: TipWaveRequest = field(default_factory=lambda: TipWaveRequest(stop="gun_exit"))
    # Optional later physical checkpoint computed by the SAME gun wave chain.
    # This bounds a semiclassical study without substituting a downstream source.
    coherent_start_z_mm: float | None = None
    intermediate_grids: tuple[HybridGrid, ...] = ()
    geometry_step_mm: float = .1
    maximum_geometry_steps: int = 10000
    exit_window_sigma_nm: float = .5

    def validate(self):
        if not isinstance(self.detector_key, str) or not self.detector_key.strip():
            raise ValueError("Select a physical detector")
        for name, minimum in (("source_particles", 1000), ("exit_particles", 1), ("seed", 0), ("maximum_geometry_steps", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.geometry_step_mm) or self.geometry_step_mm <= 0:
            raise ValueError("Geometry step must be positive")
        if not math.isfinite(self.exit_window_sigma_nm) or self.exit_window_sigma_nm <= 0:
            raise ValueError("Exit analysis window must have a positive finite physical width")
        if self.coherent_start_z_mm is not None and (not math.isfinite(self.coherent_start_z_mm) or self.coherent_start_z_mm <= 0):
            raise ValueError("Coherent start must be downstream of the physical tip")
        self.semiclassical.validate()
        for grid in (*self.intermediate_grids, self.specimen_grid):
            if type(grid.pixels) is int and 32*grid.pixels**2 > self.semiclassical.maximum_working_bytes:
                raise MemoryError("Hybrid observation grid exceeds its declared memory budget")
            grid.plane()
        self.wave.validate()
        return self


@dataclass(frozen=True)
class HybridResult:
    specimen_incident: TipGunCheckpoint
    specimen_exit: TipGunCheckpoint
    exit_particles: object
    diffraction: tuple
    downstream: dict
    report: dict
    original_source: dict


def _source_count(trace):
    reference = trace.emission_reference
    if reference is None or "ray_id" not in reference:
        raise ValueError("Hybrid trials require actual original emission identity")
    ids = np.asarray(reference["ray_id"])
    weights = np.asarray(trace.exit_bundle.weight)
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Original source weights must be finite and nonnegative")
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids < 0)
            or ids.shape != weights.shape or len(np.unique(ids)) != len(ids)):
        raise ValueError("Original source identities cannot be replaced by duplicate rays")
    current = trace.emitted_current_a
    if not math.isfinite(current) or current < 0:
        raise ValueError("Original source current must be finite and nonnegative")
    positive = int(np.count_nonzero(weights > 0)) if current > 0 else 0
    if positive < 1000:
        raise ValueError(f"Hybrid physical trial emitted only {positive} original positive-current particles; require >=1000")
    surviving = np.asarray(trace.exit_bundle.alive, dtype=bool)
    transmitted = weights[surviving]
    return {"emitted_count": len(ids), "positive_current_count": positive,
        "gun_exit_surviving_count": int(np.count_nonzero(trace.exit_bundle.alive & (weights > 0))),
        "effective_emitted_count": float(weights.sum()**2/np.dot(weights, weights)),
        "emitted_weight": float(weights.sum()),
        "gun_exit_weight": float(transmitted.sum()),
        "effective_gun_exit_count": (float(transmitted.sum()**2/np.dot(transmitted, transmitted))
            if np.any(transmitted > 0) else 0.),
        "scope": "Executed classical gun reference; coherent quadrature packets and exit samples are separate populations"}


def _grid_schedule(checkpoint, request, geometry):
    start, stop = checkpoint.plane_z_mm, request.specimen_grid.z_mm
    grids = tuple(request.intermediate_grids)+(request.specimen_grid,)
    if any(not start < grid.z_mm <= stop for grid in grids) or any(a.z_mm >= b.z_mm for a, b in zip(grids[:-1], grids[1:])):
        raise ValueError("Hybrid observation grids must be ordered after the coherent source and end at the specimen entrance")
    # Include slab faces and resolve both thickness and transverse motion.
    # The envelope covers every incoming mode and the requested output grid.
    output, previous = [], start
    current_planes = [mode.plane for mode in checkpoint.beam.modes]
    for grid in grids:
        end = grid.z_mm
        cuts = {end}
        target = grid.plane()
        planes = [*current_planes, target]
        bounds = []
        for plane in planes:
            ny, nx = plane.amplitude.shape
            corners = np.array([(x, y) for x in (-nx//2, nx-1-nx//2)
                                for y in (-ny//2, ny-1-ny//2)])
            bounds.extend(corners@plane.basis_m.T+plane.origin_m)
        lower, upper = np.min(bounds, axis=0), np.max(bounds, axis=0)
        envelope = PlaneWave(np.zeros((2, 2), complex), np.diag(upper-lower), upper)
        for piece in geometry.pieces:
            if piece.clears_grid(envelope.amplitude.shape, envelope.basis_m, envelope.origin_m,
                                 end, previous_z_mm=previous):
                continue
            cuts.update(float(z) for z in piece.event_z_mm() if previous < z <= end)
            left, right = piece.grid_axial_bounds_mm(envelope.amplitude.shape, envelope.basis_m, envelope.origin_m)
            left, right = max(previous, left), min(end, right)
            if left > right:
                continue
            step = min(request.geometry_step_mm, *(piece.maximum_step_mm(p.basis_m) for p in planes))
            axial = abs(piece.registration.rotation_array[2, 2])
            if axial > 1e-14:
                step = min(step, (piece.local_end_z_mm-piece.local_start_z_mm)/axial/8)
            if step <= 0:
                raise ValueError("Grazing geometry requires a different propagation chart")
            count = max(1, math.ceil((right-left)/step))
            if count+len(cuts)+len(output) > request.maximum_geometry_steps:
                raise MemoryError("Hybrid geometry exceeds its declared step budget")
            cuts.update(float(z) for z in np.linspace(left, right, count+1) if z > previous)
        for z in sorted(cuts):
            output.append((float(z), target if z == end else current_planes[0]))
        if len(output) > request.maximum_geometry_steps:
            raise MemoryError("Hybrid geometry exceeds its declared step budget")
        current_planes = [target]
        previous = end
    return output


def _column(checkpoint, runtime, request, *, xp, cancelled, progress_callback):
    from temsim.physics.potential_geometry import apply_potential_geometry
    current, receipts, physical_loss, numerical_change = checkpoint, [], 0., 0.
    schedule = _grid_schedule(checkpoint, request, runtime.geometry)
    for step, (end, grid) in enumerate(schedule):
        modes = []
        for index, mode in enumerate(current.beam.modes):
            if cancelled():
                raise InterruptedError("Hybrid column cancelled")
            if progress_callback:
                progress_callback(step*len(current.beam.modes)+index, len(schedule)*len(current.beam.modes),
                    f"Semiclassical mode {index+1}, plane {end:g} mm")
            if mode.weight_per_reference_electron == 0:
                modes.append(mode)
                continue
            output, receipt = propagate_mode(mode, runtime.snapshot, current.plane_z_mm*.001, end*.001,
                grid, request.semiclassical, xp=xp, cancelled=cancelled)
            numerical_change += receipt["signed_numerical_probability_change"]
            clipped, boundary = apply_potential_geometry(output.plane, runtime.geometry,
                current.plane_z_mm*.001, end*.001, grid_numerics=request.wave.wave_grid, cancelled=cancelled)
            # A mask may refine the incident lattice. Recheck the actual new
            # geometry step instead of silently keeping a coarser plan.
            for piece in runtime.geometry.pieces:
                if piece.clears_grid(clipped.amplitude.shape, clipped.basis_m, clipped.origin_m,
                                     end, previous_z_mm=current.plane_z_mm):
                    continue
                left, right = piece.grid_axial_bounds_mm(clipped.amplitude.shape, clipped.basis_m, clipped.origin_m)
                active = max(0., min(end, right)-max(current.plane_z_mm, left))
                guide = piece.maximum_step_mm(clipped.basis_m)
                if active > guide+8*np.spacing(max(abs(end), 1.)):
                    raise ValueError("Refined boundary grid requires a smaller geometry_step_mm or explicit intermediate grids")
            norm = clipped.probability
            physical_loss += output.weight_per_reference_electron*boundary["physical_loss_probability"]
            output = replace(output, plane=replace(clipped, amplitude=clipped.amplitude/math.sqrt(norm)
                if norm > 0 else clipped.amplitude), weight_per_reference_electron=output.weight_per_reference_electron*norm)
            modes.append(output)
            receipts.append({**receipt, "start_z_mm": current.plane_z_mm, "stop_z_mm": end, "geometry": boundary})
        current = TipGunCheckpoint(BeamState(tuple(modes), checkpoint.beam.reference_plane), end,
            checkpoint.reference_current_a, {"schema": "executed-hybrid-column-v1", "source_digest": checkpoint.digest,
                "upstream": checkpoint.record, "snapshot_digest": runtime.snapshot.numerical_digest,
                "intervals": receipts, "physical_loss_per_reference": physical_loss,
                "signed_numerical_probability_change": numerical_change,
                "sample_feedback_on_external_fields": False, "qualification": "DEVELOPMENT_SEMICLASSICAL_APPROXIMATION"})
    return current


def _material(state, checkpoint, runtime, request, *, cancelled, progress_callback):
    from temsim.physics.potential_specimen_transport import propagate_potential_specimen
    from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.specimen.source import specimen_is_vacuum
    if specimen_is_vacuum(state.sample):
        return checkpoint
    if checkpoint.beam.total_weight == 0:
        return replace(checkpoint, plane_z_mm=state.sample.z_mm+state.sample.thickness_nm*.5e-6,
            record={"schema": "hybrid-no-material-illumination-v1", "upstream": checkpoint.record,
                    "sample_feedback_on_external_fields": False})
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    store = ExecutedWaveStore(request.wave.execution.cache_directory,
        {"workflow": "hybrid-material-v1", "runtime": runtime.digest, "upstream": checkpoint.digest,
         "request": asdict(request)}, request.wave.execution.maximum_disk_cache_bytes)
    outputs, histories = [], []
    for mode in checkpoint.beam.modes:
        if mode.weight_per_reference_electron == 0:
            continue
        # This is a NEW numerical operator boundary. Upstream HK errors remain
        # in the immutable parent, never claimed as full-wave kinetic bounds.
        history = ElectricWaveHistory(ElectrostaticWaveReference.capture(runtime.snapshot,
            checkpoint.plane_z_mm*.001, mode.energy_kev*1000))
        kinetic = KineticPhaseLedger(.01).require(runtime.snapshot.numerical_digest,
            mode.energy_kev*1000, request.wave.column_kinetic_order, checkpoint.plane_z_mm*.001)
        entry = replace(checkpoint, beam=BeamState((mode,), checkpoint.beam.reference_plane), record={
            "schema": "hybrid-material-entrance-v1", "upstream": checkpoint.record,
            "upstream_digest": checkpoint.digest, "snapshot_digest": runtime.snapshot.numerical_digest,
            "electric_wave_history": history.to_dict(), "kinetic_phase_ledger": kinetic.to_dict(),
            "budget_scope": "Wave discretisation begins here; upstream semiclassical error retained separately"})
        result = propagate_potential_specimen(state, entry, runtime.snapshot, store=store, use_cache=False,
            inelastic_numerics=request.wave.inelastic, grid_numerics=request.wave.wave_grid,
            maximum_step_mm=request.wave.column_step_mm,
            operator_options={"kinetic_order": request.wave.column_kinetic_order,
                "residual_compute_backend": request.wave.wave_grid.compute_backend},
            geometry=runtime.geometry, cancelled=cancelled, progress_callback=progress_callback)
        outputs.extend(result.beam.modes)
        histories.append(thaw_json(result.record))
    if not outputs:
        raise ValueError("No coherent current reached the specimen")
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), result.plane_z_mm,
        checkpoint.reference_current_a, {"schema": "hybrid-material-exit-v1", "upstream": checkpoint.record,
            "upstream_digest": checkpoint.digest, "material_histories": histories,
            "sample_feedback_on_external_fields": False, "external_fields_retained": True})


def _downstream(state, runtime, particles, stop_mm, request, *, start_z_mm=None):
    from temsim.physics.unified_field_particle import build_unified_plan, execute_unified_plan, merge_unified_stops
    from temsim.physics.recording_clipping import clip_recording_planes
    if len(particles.weight) == 0:
        if start_z_mm is None:
            raise ValueError("An empty exit requires its physical observation plane")
        empty = np.empty((2, 0))
        return {"z_mm": np.array((start_z_mm, stop_mm)),
            **{key: empty.copy() for key in ("x_m", "y_m", "tx_rad", "ty_rad", "elapsed_from_handoff_s", "energy_ev")},
            "alive": np.empty(0, bool), "blocked_z_mm": np.empty(0), "blocked_key": np.empty(0, dtype="U1"),
            "weights": particles.weight, "mode_index": particles.mode_index,
            "detected_or_absorbed_weight_by_key": {}, "remaining_weight": 0., "balance_residual": 0.}
    start = float(particles.position_m[0, 2])*1000
    planes = tuple(float(p.z_mm) for p in state.recording_planes if p.inserted and start <= p.z_mm <= stop_mm)
    plan = build_unified_plan(state, start, stop_mm, save_z_mm=planes, _captured_runtime=runtime,
        maximum_step_mm=request.semiclassical.maximum_step_m*1000)
    slope = particles.direction[:, :2]/particles.direction[:, 2, None]
    energies = []
    z, x, tx, y, ty, elapsed, checkpoints = execute_unified_plan(state, plan,
        particles.position_m[:, 0], slope[:, 0], particles.position_m[:, 1], slope[:, 1],
        initial_kinetic_energy_ev=particles.energy_ev, initial_time_s=np.zeros(len(slope)),
        return_flight_times=True, energy_output=energies, history_policy="sampled")
    alive, blocked, keys = merge_unified_stops(z, np.ones(len(slope), bool),
        np.full(len(slope), np.nan), [""]*len(slope))
    alive, blocked, keys = clip_recording_planes(state, z, x, y, alive, blocked, keys)
    weights = particles.weight
    by_key = {str(key): float(weights[np.array(keys) == key].sum()) for key in sorted(set(keys)) if key}
    return {"z_mm": z, "x_m": x, "y_m": y, "tx_rad": tx, "ty_rad": ty,
        "elapsed_from_handoff_s": elapsed, "energy_ev": energies[0],
        "alive": alive, "blocked_z_mm": blocked, "blocked_key": np.asarray(keys),
        "weights": weights, "mode_index": particles.mode_index,
        "detected_or_absorbed_weight_by_key": by_key,
        "remaining_weight": float(weights[alive].sum()),
        "balance_residual": float(weights.sum()-weights[alive].sum()-sum(by_key.values()))}


def simulate_hybrid(state, request: HybridRequest, *, cancelled=lambda: False, progress_callback=None):
    """Public CUDA-gated workflow. No arbitrary source arrays or phase guesses."""
    from temsim.physics.tip_wave_pipeline import require_tip_wave_cuda, simulate_tip_wave

    wave_request = replace(request.wave, stop="gun_exit" if request.coherent_start_z_mm is None else "plane",
        observation_z_mm=request.coherent_start_z_mm)
    request = replace(request, wave=require_tip_wave_cuda(wave_request, state=state))
    request.validate()
    from temsim.simulation_modes import is_unified
    if not is_unified(state):
        raise ValueError("Hybrid calculation requires the explicitly selected Finite-Field Optics mode")
    current = state.electron_gun.emitted_current_a
    if not math.isfinite(current) or current <= 0:
        raise ValueError("Hybrid physical trials require a finite positive source current")
    import cupy as cp
    from temsim.instrument_snapshot import capture_instrument_snapshot
    # The full resolved assembly contains immutable mapping proxies. Restore
    # the supported input graph instead of relying on Python pickle copying.
    working = capture_instrument_snapshot(state).restore()
    working._tuning_cancelled = cancelled
    # Separate declared executors: CUDA wave operations and CPU Lorentz rays.
    working.acceleration_backend = "Auto"
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.detector.recording_system import ensure_recording_system
    apply_physical_layout_to_state(working)
    ensure_recording_system(working)
    # Gun tracing publishes its derived vacuum environment on the gun model.
    # Bind it BEFORE the shared immutable capture, not during source execution.
    from temsim.vacuum import bind_gun_environment
    bind_gun_environment(working)
    candidates = [p for p in working.recording_planes if p.key == request.detector_key and p.inserted]
    if len(candidates) != 1:
        raise ValueError("The selected detector must be installed and inserted")
    detector = candidates[0]
    if working.energy_filter.enabled:
        raise ValueError("Hybrid straight-column transport does not admit the bent energy filter")
    entrance = working.sample.z_mm-working.sample.thickness_nm*.5e-6
    if not math.isclose(request.specimen_grid.z_mm, entrance, rel_tol=0., abs_tol=1e-9):
        raise ValueError(f"Specimen numerical grid must be at the actual upper face {entrance:g} mm")
    if detector.z_mm <= working.sample.z_mm+working.sample.thickness_nm*.5e-6:
        raise ValueError("Select a detector downstream of the specimen")
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    runtime = capture_unified_field_runtime(working, stop_z_m=detector.z_mm*.001)
    if type(runtime.snapshot) is not ElectromagneticPotentialSnapshot:
        active = sorted({law.key for coil in runtime.snapshot.dynamic_coils for law in coil.drives})
        raise ValueError("Hybrid phase transport currently requires static fields; active drives: "
            +", ".join(active)+". Select a separate stationary acquisition or a time-dependent solver; "
            "dynamic drives are not frozen automatically")
    if cancelled():
        raise InterruptedError("Hybrid calculation cancelled")
    from temsim.optics.electron_gun.source import trace_source_to_exit
    if progress_callback:
        progress_callback(0, 1, "Tracing the original positive-current gun population")
    gun_trace = trace_source_to_exit(working, count=request.source_particles)
    source = _source_count(gun_trace)
    initial = simulate_tip_wave(working, request.wave, cancelled=cancelled, progress_callback=progress_callback).checkpoint
    runtime.require_state(working)
    incident = _column(initial, runtime, request, xp=cp, cancelled=cancelled, progress_callback=progress_callback)
    outgoing = _material(working, incident, runtime, request, cancelled=cancelled, progress_callback=progress_callback)
    particles, diffraction = sample_exit(outgoing, runtime.snapshot, request.exit_particles,
        request.semiclassical, xp=cp, seed=request.seed, cancelled=cancelled,
        window_sigma_m=request.exit_window_sigma_nm*1e-9, progress_callback=progress_callback)
    if cancelled():
        raise InterruptedError("Hybrid calculation cancelled")
    downstream = _downstream(working, runtime, particles, detector.z_mm, request, start_z_mm=outgoing.plane_z_mm)
    detected = downstream["detected_or_absorbed_weight_by_key"].get(request.detector_key, 0.)
    numerical = incident.record["signed_numerical_probability_change"]
    report = {"schema": "hybrid-particle-wave-result-v1", "status": "EXECUTED_DEVELOPMENT_MODEL",
        "request": asdict(request), "runtime_digest": runtime.digest, "source": source,
        "source_checkpoint_digest": initial.digest, "specimen_checkpoint_digest": outgoing.digest,
        "initial_wave_plane_z_mm": initial.plane_z_mm,
        "initial_wave_weight": initial.beam.total_weight,
        "source_current_a": initial.reference_current_a, "incident_weight": incident.beam.total_weight,
        "specimen_exit_weight": outgoing.beam.total_weight,
        "upstream_physical_loss": incident.record["physical_loss_per_reference"],
        "upstream_loss_scope": "Coherent start to specimen; earlier physical gun losses remain in the source checkpoint ancestry",
        "upstream_signed_numerical_probability_change": numerical,
        "upstream_balance_residual": (initial.beam.total_weight
            -incident.record["physical_loss_per_reference"]+numerical-incident.beam.total_weight),
        "material_removed_weight": incident.beam.total_weight-outgoing.beam.total_weight,
        "detector_key": request.detector_key, "detector_weight": detected,
        "detector_current_a": initial.reference_current_a*detected,
        "downstream_balance_residual": downstream["balance_residual"],
        "sample_feedback_on_external_fields": False,
        "coherent_backend": "CUDA", "classical_backend": "CPU",
        "scope": "Semiclassical HK column; real-material waves; finite-resolution gauge-covariant exit; incoherent static downstream rays",
        "limitations": ["Frame, tangent, axial step and output-grid convergence are required",
            "Exit windows coarse-grain both position and momentum",
            "Downstream interference and absolute emission times are not reconstructed",
            "Material channel limitations are retained in specimen_exit.record.material_histories",
            "Not a full-column physical or OEM qualification"]}
    original = {**gun_trace.emission_reference, "weight": np.asarray(gun_trace.exit_bundle.weight),
        "gun_exit_alive": np.asarray(gun_trace.exit_bundle.alive),
        "gun_blocked_z_mm": np.asarray(gun_trace.blocked_z_mm),
        "gun_blocked_key": np.asarray(gun_trace.blocked_key, dtype=str)}
    return HybridResult(incident, outgoing, particles, diffraction, downstream, report, original)
