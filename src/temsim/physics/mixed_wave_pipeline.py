"""Particle-to-specimen mixed coherence and downstream observations on CUDA.

The wave boundary is assigned once from the actual executed particle phase
space. Subsequent observations propagate that same mixed state; no Tip wave,
replacement pupil, implicit refocus or restoration of lost current is used.
"""
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import math
from threading import Lock

import numpy as np

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics.compute_backend import require_coherent_cuda
from temsim.physics.gun_wave_transport import specimen_entrance_z_mm
from temsim.physics.particle_wave_source import (
    MixedSourceNumerics, ParticleWaveSource, build_particle_wave_checkpoint,
)
from temsim.physics.wave_execution import InelasticWaveNumerics, WaveExecutionOptions
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.detector.wave_readout import WaveReadoutOptions


@dataclass(frozen=True)
class MixedWaveRequest:
    stop: str = "plane"
    observation_z_mm: float | None = None
    source: MixedSourceNumerics = field(default_factory=MixedSourceNumerics)
    wave_grid: WaveGridNumerics = field(default_factory=lambda: WaveGridNumerics(compute_backend="Require GPU"))
    readout: WaveReadoutOptions = field(default_factory=WaveReadoutOptions)
    execution: WaveExecutionOptions = field(default_factory=WaveExecutionOptions)
    inelastic: InelasticWaveNumerics = field(default_factory=lambda: InelasticWaveNumerics(method="zero_loss"))
    column_step_mm: float = .5
    maximum_checkpoint_bytes: int = 8*1024**3
    maximum_readout_bytes: int = 72*1024**3

    # Fixed policies for the shared downstream electromagnetic executor.
    # The retired Tip/gun and exploratory source controls are not exposed.
    column_kinetic_order = 4
    column_support_projection = False
    column_weighted_resampling = False
    column_model = "physical"
    column_exponential_method = "taylor"
    column_chebyshev_group = 8
    source_nominal_energy_preview = False
    column_preview_segment_resampling = False
    column_preview_phase_reselection = False
    column_exploratory_preview = False
    column_preview_pixels = 2048
    column_preview_support_probability = None
    column_preview_support_l2 = None
    tip_time_s = None

    def validate(self):
        if self.stop not in ("plane", "specimen_entrance", "specimen_exit"):
            raise ValueError("Mixed coherence stops at a physical plane or specimen face")
        if self.observation_z_mm is not None and (isinstance(self.observation_z_mm, (bool, np.bool_))
                or not np.isscalar(self.observation_z_mm) or not np.isreal(self.observation_z_mm)
                or not math.isfinite(self.observation_z_mm)):
            raise ValueError("Mixed coherent observation Z must be a finite physical plane")
        for name in ("maximum_checkpoint_bytes", "maximum_readout_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.column_step_mm, bool) or not math.isfinite(self.column_step_mm) or self.column_step_mm <= 0:
            raise ValueError("Mixed coherent propagation step must be finite and positive")
        self.source.validate()
        self.wave_grid.validate()
        self.readout.validate()
        self.execution.validate()
        self.inelastic.validate()
        return self


@dataclass(frozen=True)
class MixedWaveResult:
    checkpoint: object
    request: MixedWaveRequest
    propagation_cache_hit: bool
    instrument_digest: str
    detector: object | None = None
    detector_readouts: tuple = ()
    resolved_scan_identity: str | None = None


def _cancel(cancelled):
    if cancelled():
        raise InterruptedError("Mixed coherent propagation cancelled; no partial result published")


class MixedWaveObservationSession:
    """One immutable local mixed state, with lazy construction in the worker.

    ``source`` is a ParticleWaveSource or a zero-argument supplier. A supplier
    normally owns an accepted particle result and continues its original
    checkpoint only when the first observation actually runs.
    """

    def __init__(self, state, request=MixedWaveRequest(), source=None):
        from temsim.simulation_modes import require_coherent_mode
        require_coherent_mode(state)
        request.validate()
        self.inputs = capture_instrument_snapshot(state)
        self.state = self.inputs.restore()
        self.request = request
        self._source_supplier = source
        self.source = None
        self.entrance = None
        self._completed = OrderedDict()
        self._closed = False
        self._lock = Lock()

    def retained_roots(self):
        return self.source, self.entrance, self._completed

    def close(self):
        self._closed = True
        self._completed.clear()
        self.source = self.entrance = self._source_supplier = None

    def _boundary(self, cancelled, progress_callback):
        if self.entrance is not None:
            return self.entrance
        _cancel(cancelled)
        source = self._source_supplier() if callable(self._source_supplier) else self._source_supplier
        if not isinstance(source, ParticleWaveSource):
            raise ValueError("Mixed coherence requires an executed particle source at the specimen entrance")
        if source.input_digest != self.inputs.digest:
            raise ValueError("Particle handoff belongs to different captured instrument inputs")
        upper = specimen_entrance_z_mm(self.state)
        if abs(source.z_mm-upper) > 4.*abs(np.spacing(upper)):
            raise ValueError("The mixed coherent boundary must be the exact specimen upper surface")
        entry = build_particle_wave_checkpoint(source, self.request.source,
            cancelled=cancelled, progress_callback=progress_callback)
        _cancel(cancelled)
        self.source, self.entrance = source, entry
        return entry

    def observe(self, z_mm, *, cancelled=lambda: False, progress_callback=None, **unused):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Only one observation may execute in a mixed coherent session")
        caller_cancelled = cancelled
        try:
            return self._observe(z_mm, cancelled=lambda: self._closed or caller_cancelled(),
                                 progress_callback=progress_callback)
        finally:
            self._lock.release()

    def _observe(self, z_mm, *, cancelled, progress_callback):
        if self._closed:
            raise ValueError("This mixed coherent session is closed")
        self.request.validate()
        require_coherent_cuda(self.request.wave_grid.compute_backend,
            acceleration_enabled=self.request.wave_grid.acceleration_enabled)
        _cancel(cancelled)
        if isinstance(z_mm, (bool, np.bool_)) or not np.isscalar(z_mm) or not np.isreal(z_mm) or not math.isfinite(z_mm):
            raise ValueError("Mixed coherent observation Z must be finite")
        z_mm = float(z_mm)
        upper = specimen_entrance_z_mm(self.state)
        if z_mm < upper:
            raise ValueError(f"Mixed coherence begins at the specimen upper surface ({upper:.12g} mm). Use the particle view upstream.")
        self.inputs.verify_current_inputs(self.state)
        if z_mm in self._completed:
            result = self._completed.pop(z_mm)
            self._completed[z_mm] = result
            return replace(result, propagation_cache_hit=True)
        entry = self._boundary(cancelled, progress_callback)
        checkpoint = entry if z_mm == upper else self._propagate(entry, z_mm, cancelled, progress_callback)
        _cancel(cancelled)
        self.inputs.verify_current_inputs(self.state)
        result = MixedWaveResult(checkpoint,
            replace(self.request, stop="plane", observation_z_mm=z_mm), False, self.inputs.digest)
        self._completed[z_mm] = result
        # A single published target plus the original boundary bounds memory.
        while len(self._completed) > 1:
            self._completed.popitem(last=False)
        return result

    def _propagate(self, entry, z_mm, cancelled, progress_callback):
        from temsim.simulation_modes import is_unified
        if not is_unified(self.state):
            raise ValueError("Downstream mixed coherent propagation requires Finite-Field Optics with shared potentials; the particle model was not changed")
        from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
        from temsim.physics.unified_field_runtime import capture_unified_field_runtime
        from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
        from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
        from temsim.physics.particle_sections import section_limits
        lower, maximum = section_limits(self.state)
        if z_mm > maximum:
            raise ValueError("The selected plane exceeds the admitted straight-column boundary; an energy-filter operator cannot be skipped")
        bottom = float(self.state.sample.z_mm)+float(self.state.sample.thickness_nm)*.5e-6
        # Capture only the local-to-observation interval. No gun field operator
        # or Tip near-field approximation is admitted or evaluated here.
        stop = max(z_mm, bottom, entry.plane_z_mm+1e-9)
        runtime = capture_unified_field_runtime(self.state, entry.plane_z_mm*.001, stop*.001)
        if type(runtime.snapshot) is DynamicPotentialSnapshot:
            raise ValueError("A local stationary mixed-state lift requires static downstream fields; dynamic time-energy coherence is not inferred from particle clocks")
        adapter = UnifiedPotentialWaveAdapter(self.state, self.request, runtime=runtime, stop_z_mm=z_mm)
        current = entry
        inserted = self._recording_stops(entry.plane_z_mm, z_mm)
        if not self._is_vacuum() and any(entry.plane_z_mm < float(d.z_mm) < bottom for d in inserted):
            raise ValueError("A recording device inside the specimen requires a joint material/interception operator")
        if z_mm <= bottom and not self._is_vacuum():
            current = adapter.specimen(current, store=self._store(entry), use_cache=False,
                cancelled=cancelled, progress_callback=progress_callback,
                observation_z_mm=z_mm)
        else:
            if not self._is_vacuum() and bottom > entry.plane_z_mm:
                current = adapter.specimen(current, store=self._store(entry), use_cache=False,
                    cancelled=cancelled, progress_callback=progress_callback)
            if z_mm > current.plane_z_mm:
                from temsim.detector.wave_readout import _apply_recording_stop
                for detector in inserted:
                    plane = float(detector.z_mm)
                    if plane < current.plane_z_mm:
                        continue
                    if plane > current.plane_z_mm:
                        current = adapter.column(current, plane, store=None, use_cache=False,
                            cancelled=cancelled, progress_callback=progress_callback)
                    current = _apply_recording_stop(current, detector)
                    if current.beam.total_weight == 0.:
                        from temsim.immutable_json import thaw_json
                        current = replace(current, plane_z_mm=z_mm, record={
                            "schema": "absorbed-mixed-wave-v1", "upstream": thaw_json(current.record),
                            "physical_absorption_complete": True,
                            "propagation_skipped": "The complete coherent branch was physically absorbed"})
                        break
                if z_mm > current.plane_z_mm:
                    current = adapter.column(current, z_mm, store=None, use_cache=False,
                        cancelled=cancelled, progress_callback=progress_callback)
        _cancel(cancelled)
        from temsim.immutable_json import thaw_json
        return replace(current, record={**thaw_json(current.record),
            "mixed_particle_source_digest": self.source.digest,
            "mixed_handoff_z_mm": entry.plane_z_mm,
            "source_scope": "Executed particles followed by a declared local mixed-state lift; no Tip coherent propagation",
            "tip_coherent_executed": False})

    def _recording_stops(self, start, stop):
        inserted = [d for d in getattr(self.state, "recording_planes", ()) if bool(d.inserted)]
        if any(not math.isfinite(float(d.z_mm)) for d in inserted):
            raise ValueError("Physical recording planes require finite axial positions")
        devices = [d for d in inserted if start < float(d.z_mm) < stop]
        if len({d.key for d in devices}) != len(devices):
            raise ValueError("Physical recording plane identities must be unique")
        devices.sort(key=lambda d: float(d.z_mm))
        if any(a.z_mm == b.z_mm for a, b in zip(devices, devices[1:])):
            raise ValueError("Coincident recording devices need an explicit absorption order")
        return devices

    def _is_vacuum(self):
        from temsim.specimen.scene import SpecimenScene
        return SpecimenScene.from_state(self.state).is_vacuum

    def _store(self, entry):
        from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
        return ExecutedWaveStore(self.request.execution.cache_directory,
            entry.digest, self.request.execution.maximum_disk_cache_bytes)


def simulate_mixed_wave(state, source, request=MixedWaveRequest(), *, cancelled=lambda: False,
                        progress_callback=None, **unused):
    """Execute only particle-to-near-specimen mixed coherence on CUDA."""
    request.validate()
    require_coherent_cuda(request.wave_grid.compute_backend,
        acceleration_enabled=request.wave_grid.acceleration_enabled)
    upper = specimen_entrance_z_mm(state)
    z = (upper if request.stop == "specimen_entrance" else
         float(state.sample.z_mm)+float(state.sample.thickness_nm)*.5e-6
         if request.stop == "specimen_exit" else request.observation_z_mm)
    if z is None:
        raise ValueError("Select a physical observation plane for mixed coherence")
    return MixedWaveObservationSession(state, request, source).observe(z,
        cancelled=cancelled, progress_callback=progress_callback)


__all__ = ("MixedWaveRequest", "MixedWaveResult", "MixedWaveObservationSession", "simulate_mixed_wave")
