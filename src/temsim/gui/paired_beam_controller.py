"""Own one captured particle/wave pair, without starting another physics engine.

The existing particle controller executes the optical reference and, when
needed, specimen continuation. Its finished result is paired with a wave
session from the very same snapshot. This owner never edits source or optics.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
import math
from uuid import uuid4

from PySide6.QtCore import QObject, Signal

from temsim.gui.calculation_controller import CalculationController
from temsim.calculation_progress import CalculationProgress
from temsim.calculation_cache import state_model_signature
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.emission_timing import (
    require_matching_emission_time, resolve_emission_time_s, validate_emission_time_s,
)
from temsim.specimen.source import specimen_interactions_active
from temsim.physics.scan_preparation import (
    copy_resolved_scan_drives, remember_resolved_scan_drives, scan_drive_identity,
)


def _wave_execution_record(checkpoint, *, require_plane=False):
    """Read inherited execution evidence through known, same-plane receivers.

    A receiver is a physical loss, not a new field propagation. Its immutable
    upstream record retains the original clock and field identity. Arbitrary
    dictionaries with an ``upstream`` key are not execution wrappers.
    """
    from temsim.physics.wave_flux import FLUX_RTOL, FLUX_ATOL
    record = getattr(checkpoint, "record", {})
    seen, wrapped = set(), False
    expected = None
    while isinstance(record, Mapping) and record.get("schema") == "executed-recording-stop-v1":
        if id(record) in seen:
            raise ValueError("Executed wave recording history is cyclic")
        seen.add(id(record))
        rows, upstream = record.get("modes"), record.get("upstream")
        if not isinstance(rows, (tuple, list)) or not isinstance(upstream, Mapping):
            raise ValueError("Executed wave recording history is incomplete")
        if expected is None:
            expected = {mode.mode_id: mode.weight_per_reference_electron for mode in checkpoint.beam.modes}
        transmitted, incoming = {}, {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("mode_id"), str):
                raise ValueError("Executed wave recording modes are invalid")
            key = row["mode_id"]
            if key in transmitted:
                raise ValueError("Executed wave recording mode is duplicated")
            weights = row.get("received_weight"), row.get("transmitted_weight")
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in weights):
                raise ValueError("Executed wave recording weights are invalid")
            received, output = weights
            # WaveMode's unit-shape representation has an explicit norm
            # tolerance. A receiver that misses that shape can therefore
            # retain a tiny negative subtraction residual. Preserve it; do
            # not clamp the stored loss or alter the actual output weight.
            if output < 0 or received < -(FLUX_RTOL+FLUX_ATOL)*output:
                raise ValueError("Executed wave recording weights are invalid")
            transmitted[key], incoming[key] = output, math.fsum(weights)
        if set(transmitted) != set(expected) or any(
                output != expected[key] if not wrapped else
                not math.isclose(output, expected[key], rel_tol=1e-12, abs_tol=0.)
                for key, output in transmitted.items()):
            # The outer values are exact stored mode weights. Reconstructing
            # an earlier incoming weight adds a rounded received/output pair.
            raise ValueError("Executed wave recording weights do not match the retained beam")
        expected, record, wrapped = incoming, upstream, True
    if not isinstance(record, Mapping):
        raise ValueError("Executed wave execution record is invalid")
    if wrapped and (require_plane or record.get("dynamic_snapshot_digest") is not None):
        plane = record.get("unified_executed_plane_z_m")
        if (isinstance(plane, bool) or not isinstance(plane, (int, float))
                or not math.isfinite(plane) or float(checkpoint.plane_z_mm) != plane*1000.):
            raise ValueError("Executed wave recording plane does not match its retained field history")
    return record


@dataclass(frozen=True)
class BeamPairContext:
    token: str
    physical_identity: str
    particle_result: object
    specimen_active: bool
    specimen_z_mm: float
    instrument_identity: str
    particle_model_identity: str
    particle_ray_count: int
    particle_step_mm: float
    executed_scan_identity: str
    unified_execution_identity: tuple[str, str, str] | None = None
    emission_time_s: float | None = None

    def _verify_epoch_receipt(self, record, label):
        if self.emission_time_s is None:
            return  # Compatibility contexts predate explicit epoch receipts.
        required = record.get("dynamic_snapshot_digest") is not None
        if "emission_time_s" not in record:
            if required:
                raise ValueError(f"Executed {label} lost its dynamic-field emission time receipt")
            return
        if validate_emission_time_s(record["emission_time_s"]) != self.emission_time_s:
            raise ValueError(f"Executed {label} emission time does not match this pair")

    def verify_particle(self):
        state = getattr(self.particle_result, "state_snapshot", None)
        if state is None or state_model_signature(state) != self.particle_model_identity:
            raise ValueError("Executed particle source or optics do not match this pair")
        emitter = state.electron_gun.emitter
        if (int(emitter.ray_count) != self.particle_ray_count
                or float(state.step_mm) != self.particle_step_mm):
            raise ValueError("Executed particle numerical inputs do not match this pair")
        if scan_drive_identity(state) != self.executed_scan_identity:
            raise ValueError("Executed particle scan drives do not match this pair")
        if _unified_particle_identity(self.particle_result) != self.unified_execution_identity:
            raise ValueError("Executed particle potentials or geometry do not match this pair")
        if self.emission_time_s is not None and resolve_emission_time_s(state) != self.emission_time_s:
            raise ValueError("Executed particle emission time does not match this pair")
        self._verify_epoch_receipt(getattr(getattr(self.particle_result, "simulation", None), "metrics", {}), "particle")

    def verify_wave(self, result):
        if result.instrument_digest != self.instrument_identity:
            raise ValueError("Executed wave identity does not match this captured pair")
        if getattr(result, "resolved_scan_identity", None) != self.executed_scan_identity:
            raise ValueError("Executed wave scan drives do not match this pair")
        record = _wave_execution_record(getattr(result, "checkpoint", None),
                                        require_plane=self.unified_execution_identity is not None)
        if self.emission_time_s is not None:
            request_epoch = getattr(getattr(result, "request", None), "tip_time_s", None)
            if request_epoch is not None and validate_emission_time_s(request_epoch) != self.emission_time_s:
                raise ValueError("Executed wave request emission time does not match this pair")
            self._verify_epoch_receipt(record, "wave")
        if self.unified_execution_identity is not None:
            identity = tuple(record.get(key) for key in (
                "unified_runtime_digest", "unified_field_digest", "unified_geometry_digest"))
            if identity != self.unified_execution_identity:
                raise ValueError("Executed wave potentials or geometry do not match the particle calculation")


def _unified_particle_identity(result):
    """A pair requires executed ownership evidence, not just equal UI values."""
    from temsim.simulation_modes import is_unified
    state = getattr(result, "state_snapshot", None)
    if state is None or not is_unified(state):
        return None
    metrics = getattr(getattr(result, "simulation", None), "metrics", {})
    identity = tuple(metrics.get(key) for key in (
        "unified_runtime_digest", "unified_field_digest", "unified_geometry_digest"))
    if not all(isinstance(value, str) and len(value) == 64 for value in identity):
        raise ValueError("Particle result lacks executed shared-potential and geometry evidence")
    return identity


class PairedBeamController(QObject):
    """Sequential, cancellable work with isolated publication signals."""

    progress = Signal(str)
    progress_detail = Signal(object)
    ready = Signal(object, object, object, object, float)
    failed = Signal(str)

    def __init__(self, state_provider, parent=None, *, calculations=None, artifact_store=None):
        super().__init__(parent)
        self.state_provider = state_provider
        self.calculations = calculations or CalculationController(
            self, high_cache_limit=1, high_cache_budget_bytes=0,
            tuning_cache_limit=1, tuning_cache_budget_bytes=0,
            artifact_store=artifact_store, persistent_cache_enabled=artifact_store is not None)
        self.calculations.result_ready.connect(self._particle_ready)
        self.calculations.failed.connect(self._failed)
        self.calculations.progress_changed.connect(self._progress)
        self.calculations.progress_detail.connect(self._particle_progress_detail)
        self.token = None
        self.context = None
        self._snapshot = self._state = self._request = self._summary = None
        self._stage = None
        self._duration = 0.
        self.calculations.pool.coordinator.register_retained(self, "retained_roots")

    def retained_roots(self):
        return self._snapshot, self._state, self.context

    def start(self, state, request, summary, ray_count, step_mm):
        # Capture first: malformed requests must not discard a valid old pair.
        epoch = require_matching_emission_time(state, request)
        snapshot = capture_instrument_snapshot(state)
        frozen = snapshot.restore()
        self.cancel()
        self.token = uuid4().hex
        self._snapshot, self._state = snapshot, frozen
        self._request, self._summary = request, summary
        self._emission_time_s = epoch
        self._ray_count, self._step_mm = int(ray_count), float(step_mm)
        self._model_identity = state_model_signature(frozen)
        self._duration = 0.
        self._stage = "rays"
        self._material_stage = bool(specimen_interactions_active(frozen.sample))
        self._pair_progress = CalculationProgress(self.token, "Paired beams", "Particle transport",
            stages_completed=0, stages_total=4 + int(self._material_stage))
        self.progress_detail.emit(self._pair_progress)
        self.progress.emit(f"Paired beams: calculating particle transport from the captured tip at emission time {epoch!r} s.")
        try:
            self._submit("rays")
        except Exception:
            self.cancel()
            raise
        return self.token

    def _submit(self, workflow, existing_result=None):
        self.calculations.submit_background(
            self._snapshot.restore(), "High accuracy", self._ray_count,
            self._step_mm, workflow=workflow, existing_result=existing_result)

    def is_current(self):
        if self.token is None or self._snapshot is None:
            return False
        try:
            return (capture_instrument_snapshot(self.state_provider()).physical_digest
                    == self._snapshot.physical_digest)
        except (ValueError, OSError, RuntimeError):
            return False

    def _particle_ready(self, quality, result, duration):
        if self.token is None or self._stage not in {"rays", "sample"}:
            return
        if not self.is_current():
            self._failed(quality, "Instrument inputs changed; the captured pair is no longer current.")
            return
        if getattr(result, "workflow", None) != self._stage:
            self._failed(quality, "Particle workflow does not match the captured pair request.")
            return
        self._duration += float(duration)
        sample_active = specimen_interactions_active(self._state.sample)
        if self._stage == "rays" and sample_active:
            self._stage = "sample"
            self._pair_progress = replace(self._pair_progress, step="Particle specimen interactions",
                completed=None, total=None, stages_completed=1)
            self.progress_detail.emit(self._pair_progress)
            self.progress.emit("Paired beams: continuing particle specimen interactions from the saved incident beam.")
            try:
                self._submit("sample", result)
            except Exception as error:
                self._failed(quality, str(error))
            return
        self._stage = "wave"
        self._pair_progress = replace(self._pair_progress, step="Coherent wave transport",
            completed=None, total=None, stages_completed=1 + int(self._material_stage))
        self.progress_detail.emit(self._pair_progress)
        # Particle preparation resolved automatic/held drives in its private
        # worker. Carry those actual fields into the wave, not the pre-solve
        # matrices retained by the immutable original request. The original
        # snapshot remains authoritative for is_current and cancellation.
        executed = getattr(result, "state_snapshot", None)
        if executed is None or state_model_signature(executed) != self._model_identity:
            self._failed(quality, "Executed particle source or optics do not match this pair")
            return
        # The particle worker may resolve its own count, step and history
        # budget. Capture that verified execution so the common runtime has
        # exactly the same input identity. Wave numerical controls stay in the
        # independently captured request; no particle count becomes a wave grid.
        self._state = capture_instrument_snapshot(executed).restore()
        copy_resolved_scan_drives(executed, self._state)
        execution_snapshot = capture_instrument_snapshot(self._state)
        if scan_drive_identity(self._state) != scan_drive_identity(executed):
            self._failed(quality, "Executed particle scan controls do not match the captured wave")
            return
        remember_resolved_scan_drives(self._state, execution_snapshot.digest)
        try:
            self.context = BeamPairContext(self.token, self._snapshot.physical_digest,
                result, sample_active, float(self._state.sample.z_mm), execution_snapshot.digest,
                self._model_identity, self._ray_count, self._step_mm, scan_drive_identity(executed),
                _unified_particle_identity(result), self._emission_time_s)
            self.context.verify_particle()
        except (ValueError, AttributeError) as error:
            self._failed(quality, str(error))
            return
        self.progress.emit("Paired beams: particle result complete; calculating the coherent field from the same captured inputs.")
        self.ready.emit(self._state, self._request, self._summary,
                        self.context, self._duration)

    def _progress(self, _quality, done, total, message):
        if self.token is not None and self._stage in {"rays", "sample"}:
            self.progress.emit(f"Paired particles: {message} ({done}/{total})")

    def _particle_progress_detail(self, update):
        if self.token is None or self._stage not in {"rays", "sample"}:
            return
        if not update.active:
            # The result_ready/failed lifecycle decides whether this particle
            # phase is accepted. A particle's completion is not pair completion.
            return
        self._pair_progress = replace(self._pair_progress, step=update.step,
            completed=update.completed, total=update.total, stage_name=update.stage_name)
        self.progress_detail.emit(self._pair_progress)

    def accept_wave_progress(self, update):
        """Consume only this captured pair's wave, including readout/publication."""
        if (self.token is None or self._stage != "wave" or update.parent_id != self.token):
            return
        base = 1 + int(self._material_stage)
        pair_failed = update.status == "succeeded" and update.pair_error is not None
        self._pair_progress = replace(self._pair_progress, step=update.step,
            completed=update.completed, total=update.total, stage_name=update.stage_name,
            stages_completed=base + (2 if pair_failed else update.stages_completed or 0),
            status="failed" if pair_failed else update.status,
            pair_error=update.pair_error)
        if pair_failed:
            self._pair_progress = replace(self._pair_progress,
                step="Pair verification failed; coherent result retained: " + update.pair_error)
        self.progress_detail.emit(self._pair_progress)
        if not update.active:
            self._stage = ("complete" if self._pair_progress.status == "succeeded"
                           else self._pair_progress.status)

    def _failed(self, _quality, message):
        if self.token is None:
            return
        self._terminal_progress("failed", str(message))
        self.cancel()
        self.failed.emit(str(message))

    def report_rejected_request(self, message):
        """Report a preflight failure even when no pair token was admitted."""
        progress = getattr(self, "_pair_progress", None)
        if progress is not None and progress.active:
            self._terminal_progress("failed", str(message))
        elif progress is None or progress.status != "failed" or progress.step != str(message):
            self._pair_progress = CalculationProgress(f"pair-check:{uuid4().hex}",
                "Paired beams", "Checking captured inputs", stages_completed=0, stages_total=1
                ).terminal("failed", str(message))
            self.progress_detail.emit(self._pair_progress)

    def _terminal_progress(self, status, message):
        progress = getattr(self, "_pair_progress", None)
        if progress is not None and progress.active:
            self._pair_progress = progress.terminal(status, message)
            self.progress_detail.emit(self._pair_progress)

    def cancel(self):
        self._terminal_progress("cancelled", "Paired calculation cancelled")
        self.calculations.invalidate_pending(include_explicit=True)
        self.token = self.context = None
        self._snapshot = self._state = self._request = self._summary = None
        self._stage = None

    def shutdown(self, msecs=3000):
        self.cancel()
        return (self.calculations.pool.waitForDone(msecs)
                and self.calculations.section_file_pool.waitForDone(msecs))
