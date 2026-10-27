"""Explicit development request: physical FEG tip -> column -> specimen -> sensor.

Observables are lazy views of the same executed complex state. A cache entry
is tied to the captured instrument, external model contents, source code and
numerics. There is no API for supplying a replacement downstream source.
"""
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

from temsim.detector.wave_readout import WaveReadoutOptions, read_wave_detector, _apply_recording_stop
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.electron_gun.tip_coherence import TipWaveNumerics
from temsim.physics.column_wave import _prepare_column, _propagate_column, _component_events, _propagate_column_segmented
from temsim.physics.gun_wave_transport import specimen_entrance_z_mm
from temsim.physics.specimen_wave_transport import _propagate_specimen
from temsim.physics.tip_gun_wave import GunWaveNumerics, build_tip_gun_checkpoint
from temsim.physics.wave_grid import (
    EXPLORATORY_SUPPORT_L2_MAXIMUM, EXPLORATORY_SUPPORT_PROBABILITY_MAXIMUM,
    WaveGridNumerics,
)
from temsim.physics.wave_execution import WaveExecutionOptions, InelasticWaveNumerics
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.surface_wave import SurfaceWaveNumerics
from temsim.physics.radial_gun_wave import RadialGunNumerics
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics
from temsim.physics.column_acceptance_window import AcceptanceWindowNumerics


@dataclass(frozen=True)
class TipWaveRequest:
    stop: str = "detector"
    source: TipWaveNumerics = TipWaveNumerics()
    gun: GunWaveNumerics = GunWaveNumerics()
    column_step_mm: float = .5
    column_kinetic_order: int = 4
    column_support_projection: bool = False
    wave_grid: WaveGridNumerics = WaveGridNumerics(compute_backend="Require GPU")
    readout: WaveReadoutOptions = WaveReadoutOptions()
    detector_pixels: int = 512
    detector_key: str | None = None
    maximum_readout_bytes: int = 72*1024**3
    execution: WaveExecutionOptions = WaveExecutionOptions()
    inelastic: InelasticWaveNumerics = InelasticWaveNumerics()
    tip_time_s: float | None = None
    detector_keys: tuple[str, ...] = ()
    surface: SurfaceWaveNumerics = SurfaceWaveNumerics.joint_gun()
    radial_gun: RadialGunNumerics = RadialGunNumerics()
    radial_column: RadialColumnNumerics = RadialColumnNumerics(backend="cuda")
    acceptance_window: AcceptanceWindowNumerics = AcceptanceWindowNumerics()
    observation_z_mm: float | None = None
    column_model: str = "physical"
    column_exponential_method: str = "taylor"
    column_chebyshev_group: int = 8
    column_weighted_resampling: bool = False
    column_exploratory_preview: bool = False
    column_preview_segment_resampling: bool = False
    column_preview_phase_reselection: bool = False
    source_nominal_energy_preview: bool = False
    column_preview_pixels: int = 2048
    column_preview_support_probability: float | None = None
    column_preview_support_l2: float | None = None

    def validate(self):
        from temsim.physics.potential_kinetic_expansion import validate_kinetic_order
        validate_kinetic_order(self.column_kinetic_order)
        from temsim.physics.potential_wave_policy import potential_wave_operator_policy
        potential_wave_operator_policy({"residual_exponential_method": self.column_exponential_method,
                                        "maximum_chebyshev_group": self.column_chebyshev_group})
        if type(self.column_support_projection) is not bool:
            raise ValueError("column_support_projection must be an explicit boolean")
        if type(self.column_weighted_resampling) is not bool:
            raise ValueError("column_weighted_resampling must be an explicit boolean")
        if self.column_weighted_resampling and not self.column_support_projection:
            raise ValueError("Weighted coherent resampling requires controlled support projection")
        if type(self.column_exploratory_preview) is not bool:
            raise ValueError("column_exploratory_preview must be an explicit boolean")
        if type(self.column_preview_segment_resampling) is not bool:
            raise ValueError("column_preview_segment_resampling must be an explicit boolean")
        if self.column_preview_segment_resampling and not (
                self.column_exploratory_preview and self.column_support_projection
                and self.column_weighted_resampling and self.column_model == "physical"):
            raise ValueError("Segment-start resampling requires physical exploratory preview with controlled support and weighted resampling")
        if type(self.column_preview_phase_reselection) is not bool:
            raise ValueError("column_preview_phase_reselection must be an explicit boolean")
        if self.column_preview_phase_reselection and not self.column_preview_segment_resampling:
            raise ValueError("Preview phase reselection requires explicit segment-start resampling")
        if type(self.source_nominal_energy_preview) is not bool:
            raise ValueError("source_nominal_energy_preview must be an explicit boolean")
        if self.source_nominal_energy_preview and not (
                self.column_exploratory_preview and self.column_model == "physical"
                and self.source.energy_samples == 1):
            raise ValueError("Nominal-energy source preview requires physical exploratory preview and exactly one energy sample")
        if type(self.column_preview_pixels) is not int or not 32 <= self.column_preview_pixels <= 8192:
            raise ValueError("column_preview_pixels must be an integer from 32 to 8192")
        if self.column_exploratory_preview and not (self.column_support_projection and self.column_weighted_resampling):
            raise ValueError("Exploratory preview requires controlled support and weighted coherent resampling")
        preview_overrides = (self.column_preview_support_probability, self.column_preview_support_l2)
        if any(value is not None for value in preview_overrides):
            if not (self.column_exploratory_preview and self.column_support_projection
                    and self.column_weighted_resampling and self.column_model == "physical"):
                raise ValueError("Column preview support overrides require physical exploratory preview with controlled support and weighted resampling")
            for name, value, maximum in (
                    ("column_preview_support_probability", preview_overrides[0], EXPLORATORY_SUPPORT_PROBABILITY_MAXIMUM),
                    ("column_preview_support_l2", preview_overrides[1], EXPLORATORY_SUPPORT_L2_MAXIMUM)):
                if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                        or not math.isfinite(value) or not 0 <= value <= maximum):
                    raise ValueError(f"{name} must be finite and between zero and {maximum:g}, or None")
        from temsim.physics.ideal_column_reference import validate_column_model
        validate_column_model(self.column_model)
        if self.column_model != "physical" and self.stop == "tip_near_field":
            raise ValueError("An ideal column reference starts at the executed gun exit")
        if self.stop not in ("tip_near_field", "gun_exit", "specimen_entrance", "specimen_exit", "detector", "plane"):
            raise ValueError("Unknown tip wave stopping stage")
        if self.stop == "plane":
            if (isinstance(self.observation_z_mm, bool)
                    or not isinstance(self.observation_z_mm, Real) or not math.isfinite(self.observation_z_mm)):
                raise ValueError("A virtual observation plane needs a finite numeric axial Z in millimetres")
            if self.detector_key is not None or self.detector_keys:
                raise ValueError("A virtual observation plane cannot also select detector readouts")
        elif self.observation_z_mm is not None:
            raise ValueError("Observation Z is only used with stop='plane'")
        self.surface.validate()
        self.radial_gun.validate()
        self.radial_column.validate()
        self.acceptance_window.validate()
        self.source.validate(); self.gun.validate(); self.readout.validate(); self.wave_grid.validate()
        if self.wave_grid.exploratory_support_budget:
            raise ValueError("The gun wave_grid must keep strict support budgets; use column preview overrides")
        if self.column_preview_phase_reselection:
            probability = (self.wave_grid.maximum_support_loss_fraction if self.column_preview_support_probability is None
                           else self.column_preview_support_probability)
            l2 = (self.wave_grid.maximum_accumulated_support_l2 if self.column_preview_support_l2 is None
                  else self.column_preview_support_l2)
            if probability <= 0 or l2 <= 0:
                raise ValueError("Preview phase reselection requires positive effective probability and complex L2 allowances")
        self.execution.validate(); self.inelastic.validate()
        if self.tip_time_s is not None and (isinstance(self.tip_time_s, bool) or not math.isfinite(self.tip_time_s)):
            raise ValueError("Tip emission time must be finite")
        if self.detector_key is not None and (not isinstance(self.detector_key, str) or not self.detector_key):
            raise ValueError("Detector selection must be an installed component key")
        if (not isinstance(self.detector_keys, tuple)
                or any(not isinstance(key, str) or not key.strip() for key in self.detector_keys)):
            raise ValueError("Detector selections must be a tuple of installed component keys")
        if len(set(self.detector_keys)) != len(self.detector_keys):
            raise ValueError("A detector cannot be requested twice")
        if self.detector_keys and (self.detector_key is not None or self.stop != "detector"):
            raise ValueError("Multiple detectors require stop='detector' and no single detector_key")
        if isinstance(self.column_step_mm, bool) or not math.isfinite(self.column_step_mm) or self.column_step_mm <= 0:
            raise ValueError("Column step must be positive and finite")
        if isinstance(self.detector_pixels, bool) or not isinstance(self.detector_pixels, int) or not 2 <= self.detector_pixels <= 8192:
            raise ValueError("Detector pixels must be an integer from 2 to 8192")
        if isinstance(self.maximum_readout_bytes, bool) or not isinstance(self.maximum_readout_bytes, int) or self.maximum_readout_bytes <= 0:
            raise ValueError("Readout memory budget must be a positive integer")
        return self


@dataclass(frozen=True)
class TipWaveResult:
    checkpoint: object
    detector: object | None
    request: TipWaveRequest
    propagation_cache_hit: bool
    instrument_digest: str
    # Physical Z order, not request order. `detector` retains the final-plane
    # readout for old consumers; it is never a sum of different detectors.
    detector_readouts: tuple = ()
    # instrument_digest remains the immutable request identity. Automatic/held
    # preparation can resolve private scan matrices before field execution.
    resolved_scan_identity: str | None = None


_STAGES = OrderedDict()
_MAXIMUM_CACHE_BYTES = 8*1024**3
# Session-owned references to completed virtual observations and committed
# intermediate column planes. Disk references reuse existing stage files;
# they add no duplicate wave arrays or file format.
_OBSERVATION_PREFIXES = OrderedDict()
_MAXIMUM_OBSERVATION_PREFIXES = 512


def require_tip_wave_cuda(request):
    """Check user execution before capture and forbid downstream CPU fallback."""
    from temsim.physics.compute_backend import GPUExecutionError, require_coherent_cuda
    request.validate()
    if request.radial_column.backend != "cuda":
        raise GPUExecutionError("required",
            "Coherent calculations currently require CUDA radial transforms; "
            "set radial_column.backend='cuda'. CPU execution is disabled for user wave jobs.")
    policy = require_coherent_cuda(request.wave_grid.compute_backend,
        acceleration_enabled=request.wave_grid.acceleration_enabled)
    if request.wave_grid.compute_backend == policy:
        return request
    return replace(request, wave_grid=replace(request.wave_grid, compute_backend=policy))


def _find_observation_prefix(identity, minimum_z_mm, target_z_mm, store, *, segmented, cancelled,
                             completed_validator=None):
    durable = (store.find_completed_plane(identity, minimum_z_mm, target_z_mm,
        validate=completed_validator, cancelled=cancelled)
        if segmented and completed_validator is not None else None)
    candidates = sorted((key for key in _OBSERVATION_PREFIXES
        if key[0] == identity and minimum_z_mm <= key[1] <= target_z_mm),
        key=lambda key: key[1], reverse=True)
    for key in candidates:
        if durable is not None and durable.plane_z_mm >= key[1]:
            return durable
        if cancelled():
            raise InterruptedError("Virtual-plane prefix lookup cancelled")
        cache_key, digest = _OBSERVATION_PREFIXES[key]
        if segmented:
            checkpoint = store.get(cache_key)
        else:
            entry = _STAGES.get(cache_key)
            checkpoint = None if entry is None else entry[0]
        if checkpoint is None:
            del _OBSERVATION_PREFIXES[key]
            continue
        if checkpoint.plane_z_mm != key[1] or checkpoint.digest != digest:
            raise ValueError("Executed virtual-plane prefix identity changed")
        # Store.get validates the manifest. Load one immutable mode at a time
        # to verify the existing array checksums even for an exact-Z readout.
        if segmented:
            for mode in checkpoint.beam.modes:
                if cancelled():
                    raise InterruptedError("Virtual-plane prefix verification cancelled")
                del mode
        if cancelled():
            raise InterruptedError("Virtual-plane prefix verification cancelled")
        _OBSERVATION_PREFIXES.move_to_end(key)
        if not segmented:
            _STAGES.move_to_end(cache_key)
        return checkpoint
    return durable


def _register_observation_prefix(identity, checkpoint, store, request, *, cancelled,
                                 completed_validator=None):
    key = (identity, checkpoint.plane_z_mm)
    digest = checkpoint.digest
    if cancelled():
        raise InterruptedError("Virtual-plane request cancelled before prefix registration")
    if request.execution.segmented:
        storage = checkpoint.record.get("storage", {})
        if storage.get("dependency") != store.dependency or not storage.get("key"):
            # Only a real committed execution can become a continuation.
            return
        cache_key = storage["key"]
        if (completed_validator is not None
                and checkpoint.record.get("schema") == "unified-source-column-v1"):
            store.register_completed_plane(identity, checkpoint,
                expected_mode_ids=tuple(row["mode_id"] for row in checkpoint.beam.modes.rows),
                validate=completed_validator, cancelled=cancelled)
    else:
        cache_key = (store.dependency, "observation_plane", json_digest(key))
        _retain(cache_key, checkpoint, request.execution.maximum_ram_cache_bytes)
        if cache_key not in _STAGES:
            return
    _OBSERVATION_PREFIXES[key] = (cache_key, digest)
    _OBSERVATION_PREFIXES.move_to_end(key)
    while len(_OBSERVATION_PREFIXES) > _MAXIMUM_OBSERVATION_PREFIXES:
        _OBSERVATION_PREFIXES.popitem(last=False)


def _surface_segment(state, request, snapshot, *, use_cache, cancelled, progress_callback,
                     instrument_digest):
    """Near-tip development stage; never mislabelled as a gun-exit BeamState."""
    import numpy as np
    from temsim.physics.surface_wave import compute_surface_wave, _CACHE
    model = state.electron_gun.emitter.surface_model
    if model is None or model.coherence is None:
        raise ValueError("Configure coherent surface emission before requesting tip_near_field")
    model.validate()
    n = request.surface
    from temsim.physics.surface_wave import guard_surface_column
    guard_signature = guard_surface_column(state, n, prepare_column=_prepare_column)
    cached = tuple(_CACHE.values()) if use_cache else ()
    result = compute_surface_wave(state.electron_gun, n, use_cache=use_cache,
        cancelled=cancelled, progress_callback=progress_callback)
    hit = any(result is old for old in cached)
    from temsim.immutable_json import freeze_json
    result = replace(result, record=freeze_json({**result.record,
        "instrument_snapshot": instrument_digest, "column_guard_plan": guard_signature,
        "column_guard": "No active field, stop or intersecting wall in this near-field domain"}))
    snapshot.verify_current_inputs(state)
    if cancelled():
        raise InterruptedError("Tip wave request cancelled")
    return TipWaveResult(result, None, request, hit, instrument_digest)


def _recording_planes(state, key, keys=()):
    candidates = [d for d in (*getattr(state, "stem_detectors", ()),
                  getattr(state, "fluorescent_screen", None), getattr(state, "camera", None))
                  if d is not None and bool(d.inserted) and d.z_mm > state.sample.z_mm+state.sample.thickness_nm*.5e-6]
    candidates.sort(key=lambda d: d.z_mm)
    if len({d.key for d in candidates}) != len(candidates):
        raise ValueError("Duplicate physical detector identity")
    requested = set(keys) if keys else ({key} if key is not None else None)
    targets = [d for d in candidates if bool(getattr(d, "readout_enabled", True))
               and (requested is None or d.key in requested)]
    if requested is not None:
        missing = requested - {d.key for d in targets}
        if missing:
            raise ValueError("Requested detectors are missing, upstream, retracted or readout-disabled: "
                             + ", ".join(sorted(missing)))
    if not targets:
        raise ValueError("Insert and enable the selected physical detector to record a wave result")
    if requested is None:
        targets = targets[:1]
    planes = tuple(d for d in candidates if d.z_mm <= targets[-1].z_mm)
    if any(math.isclose(a.z_mm, b.z_mm, rel_tol=0., abs_tol=1e-9)
           for a, b in zip(planes, planes[1:])):
        raise ValueError("Coincident inserted detector planes need an explicit physical absorption order")
    return planes, tuple(targets)


def _observation_stops(state, gun_exit_z_mm, target_z_mm, entrance_z_mm, exit_z_mm, vacuum):
    """All physical stops before a virtual plane, including unobserved ones.

    A coincident virtual plane denotes the incident state at that physical
    device. Absorption is included only when continuing strictly beyond it.
    The current gun/specimen producers cannot insert a detector internally.
    """
    inserted = tuple(d for d in getattr(state, "recording_planes", ()) if bool(d.inserted))
    if any(not math.isfinite(float(d.z_mm)) for d in inserted):
        raise ValueError("Physical detector axial positions must be finite")
    candidates = tuple(d for d in inserted if float(d.z_mm) < target_z_mm)
    if len({d.key for d in candidates}) != len(candidates):
        raise ValueError("Duplicate physical detector identity")
    if any(float(d.z_mm) < gun_exit_z_mm for d in candidates):
        raise ValueError("A detector inside the gun requires a jointly truncated coherent gun operator")
    if not vacuum and any(entrance_z_mm < float(d.z_mm) < exit_z_mm for d in candidates):
        raise ValueError("A detector inside the specimen requires a truncated specimen wave operator")
    ordered = tuple(sorted(candidates, key=lambda d: float(d.z_mm)))
    if any(math.isclose(a.z_mm, b.z_mm, rel_tol=0., abs_tol=1e-9)
           for a, b in zip(ordered, ordered[1:])):
        raise ValueError("Coincident inserted detector planes need an explicit physical absorption order")
    return ordered


def _recording_stop_already_executed(checkpoint, detector_key):
    """Distinguish a cached detector exit from its same-Z incident field.

    Detector keys are unique in the unchanged captured instrument. Both
    storage forms and their unified propagation stamps retain the executed
    absorption receipt; a plane coordinate alone does not establish its side.
    """
    from temsim.physics.unified_wave_runtime import _recording_stop_ancestry
    ancestor, stops = _recording_stop_ancestry(checkpoint.record)
    return any(stop.get("detector") == detector_key for stop in (
        *stops, *ancestor.get("unified_recording_stops", ())))


def _retain(key, result, maximum_bytes=_MAXIMUM_CACHE_BYTES):
    size = sum(m.plane.amplitude.nbytes for m in result.beam.modes)
    if size > maximum_bytes:
        return
    _STAGES[key] = (result, size)
    _STAGES.move_to_end(key)
    while len(_STAGES) > 8 or sum(v[1] for v in _STAGES.values()) > maximum_bytes:
        _STAGES.popitem(last=False)


class TipWaveObservationSession:
    """Observe multiple exact Z planes from one privately captured instrument.

    Supply the detached instrument controls captured at the user's Calculate
    command. Snapshot capture and the first restore run lazily inside the first
    observation worker. Later requests change only Z; no downstream wave/source
    state is accepted. A caller editing physical inputs must create a new session.
    """

    def __init__(self, detached_state, request, *, use_cache=True):
        import threading
        request.validate()
        if request.stop != "plane":
            raise ValueError("A live observation session requires stop='plane'")
        self.__pending_state = detached_state
        self.__request = request
        self.__use_cache = use_cache
        self.__snapshot = self.__working = self.__instrument_digest = None
        self.__scan_preparation = {}
        self.__lock = threading.Lock()

    def observe(self, z_mm, *, cancelled=lambda: False, progress_callback=None):
        request = replace(self.__request, observation_z_mm=z_mm).validate()
        if cancelled():
            raise InterruptedError("Tip wave observation cancelled")
        request = require_tip_wave_cuda(request)
        if not self.__lock.acquire(blocking=False):
            raise RuntimeError("Only one observation may execute in a coherent session at a time")
        try:
            if self.__snapshot is None:
                snapshot = capture_instrument_snapshot(self.__pending_state)
                working = snapshot.restore()
                identity = snapshot.digest
                self.__snapshot, self.__working = snapshot, working
                self.__instrument_digest = identity
                self.__pending_state = None
            else:
                self.__snapshot.verify_current_inputs(self.__working)
            if cancelled():
                raise InterruptedError("Tip wave observation cancelled before execution")
            return _execute_tip_wave(self.__working, request, self.__snapshot, self.__instrument_digest,
                use_cache=self.__use_cache, cancelled=cancelled, progress_callback=progress_callback,
                scan_preparation_cache=self.__scan_preparation)
        finally:
            self.__lock.release()
            from temsim.physics.wave_device import release_device_memory
            release_device_memory()


def simulate_tip_wave(state, request=TipWaveRequest(), *, use_cache=True,
                      cancelled=lambda: False, progress_callback=None, _scan_preparation_cache=None):
    """Compute selected stages and observations, preserving mandatory state.

This development product supports arrival-time scan coils and conditional
material-model inelastic waves. Existing ray, EDS and resolved EELS products
retain their established workflows and are not replaced by this entry point.
"""
    from temsim.simulation_modes import require_coherent_mode
    require_coherent_mode(state)
    request.validate()
    if cancelled():
        raise InterruptedError("Tip wave request cancelled")
    request = require_tip_wave_cuda(request)
    snapshot = capture_instrument_snapshot(state)
    working = snapshot.restore()
    try:
        return _execute_tip_wave(working, request, snapshot, snapshot.digest, use_cache=use_cache,
                                 cancelled=cancelled, progress_callback=progress_callback,
                                 scan_preparation_cache=_scan_preparation_cache)
    finally:
        from temsim.physics.wave_device import release_device_memory
        release_device_memory()


def _executed_gun_digest(checkpoint):
    """Find the actual immutable source checkpoint in the executed ancestry."""
    from collections.abc import Mapping
    record, digest = checkpoint.record, checkpoint.digest
    seen = set()
    while isinstance(record, Mapping) and id(record) not in seen:
        seen.add(id(record))
        if record.get("schema") in {
            "executed-tip-gun-v1", "executed-grounded-round-surface-gun-v1",
        }:
            # The committed manifest binds the source metadata and every
            # payload checksum, independently of its random storage directory.
            # Re-executing without cache must not turn identical source bytes
            # into an unrelated source merely by publishing them elsewhere.
            storage = record.get("storage", {})
            manifest = storage.get("manifest_digest") if isinstance(storage, Mapping) else None
            if isinstance(manifest, str) and len(manifest) == 64:
                return "stored:" + manifest
            if isinstance(digest, str) and digest:
                return digest
            break
        digest, record = record.get("upstream_digest"), record.get("upstream")
    raise ValueError("Wavefront comparison requires verified executed gun ancestry; recalculate the beam")


def simulate_ideal_reference(state, actual_result, *, use_cache=True,
                             cancelled=lambda: False, progress_callback=None):
    """Compare lenses from the same source, never replace a downstream wave.

    The physical instrument is captured unchanged. Only downstream column
    operators use the explicitly recorded reference policy. Each energy mode
    keeps its focusing/rotation and source phase; this is not an achromatic
    mixture or a replacement for material scattering. Shared gun cache keys
    and verified source ancestry prevent two unrelated fields being compared.
    """
    if cancelled():
        raise InterruptedError("Ideal wavefront reference cancelled")
    if not isinstance(actual_result, TipWaveResult):
        raise TypeError("An ideal reference needs one executed Tip wave result")
    actual_request = actual_result.request.validate()
    if actual_request.stop != "plane" or actual_request.column_model != "physical":
        raise ValueError("Calculate a physical beam at an exact observation plane first")
    if actual_result.checkpoint.plane_z_mm != actual_request.observation_z_mm:
        raise ValueError("Actual wave does not belong to its requested exact observation plane")
    actual_request = require_tip_wave_cuda(actual_request)
    snapshot = capture_instrument_snapshot(state)
    if snapshot.digest != actual_result.instrument_digest:
        raise ValueError("Source or instrument changed since the actual beam; recalculate it before comparison")
    source_digest = _executed_gun_digest(actual_result.checkpoint)
    ideal_request = replace(actual_request, column_model="first_order_reference")
    reference = simulate_tip_wave(state, ideal_request, use_cache=use_cache,
        cancelled=cancelled, progress_callback=progress_callback)
    if cancelled():
        raise InterruptedError("Ideal wavefront reference cancelled before publication")
    snapshot.verify_current_inputs(state)
    if capture_instrument_snapshot(state).digest != snapshot.digest:
        raise ValueError("Source or instrument changed during reference calculation; recalculate the beam")
    if (reference.instrument_digest != actual_result.instrument_digest
            or reference.checkpoint.plane_z_mm != actual_result.checkpoint.plane_z_mm
            or reference.resolved_scan_identity != actual_result.resolved_scan_identity
            or _executed_gun_digest(reference.checkpoint) != source_digest):
        raise ValueError("Actual and ideal wavefronts do not share the same executed source and plane")
    return reference


def _execute_tip_wave(working, request, snapshot, instrument_digest, *, use_cache,
                      cancelled, progress_callback, scan_preparation_cache=None):
    """Execute from privately captured controls, never a replacement beam."""
    def verify():
        snapshot.verify_current_inputs(working)

    # Admission and execution use the same immutable captured controls. Keep
    # plans only within this request and only reuse their exact endpoint grid;
    # a detector/material split must never borrow a larger interval's plan.
    from temsim.simulation_modes import is_unified
    unified = is_unified(working)
    if request.source_nominal_energy_preview:
        if not unified:
            raise ValueError("Nominal-energy source preview is available only in unified Finite-Field Optics")
        source_model = getattr(working.electron_gun.emitter, "surface_model", None)
        if source_model is not None and source_model.coherence is not None:
            raise ValueError("Nominal-energy preview is not implemented for the joint surface/gun source executor")
    if (request.column_preview_segment_resampling or request.column_preview_phase_reselection) and not unified:
        raise ValueError("Preview segment resampling is available only in unified Finite-Field Optics")
    if (any(value is not None for value in
            (request.column_preview_support_probability, request.column_preview_support_l2)) and not unified):
        raise ValueError("Column preview support overrides are available only in unified Finite-Field Optics")
    if request.column_weighted_resampling and not unified:
        raise ValueError("Weighted coherent resampling is available only in unified Finite-Field Optics")
    unified_wave = None
    prepared_columns = {}
    column_options = ({} if request.column_model == "physical"
                      else {"column_model": request.column_model})

    def admit_column(start, stop):
        if unified_wave is not None:
            return unified_wave.admit(start, stop)
        key = (float(start), float(stop))
        prepared = _prepare_column(working, start, stop, request.column_step_mm, **column_options)
        prepared_columns[key] = prepared
        return prepared

    if not unified and bool(getattr(working, "equivalent_image_lenses_enabled", False)):
        raise ValueError("Only executed upstream caches may replace distributed optics")
    nano = getattr(working, "nanopulser", None)
    if nano is not None and nano.installed:
        raise ValueError("Installed electrostatic beam blanker needs time-energy wavepacket propagation")
    if request.stop == "tip_near_field":
        return _surface_segment(working, request, snapshot, use_cache=use_cache,
            cancelled=cancelled, progress_callback=progress_callback, instrument_digest=instrument_digest)
    # Every discrete upstream column event must have an owner, including a
    # component moved into the gun region by a custom assembly.
    gun_end = working.electron_gun.exit_plane_z_mm
    if _component_events(working, -1e-12, gun_end)[0]:
        raise ValueError("Column deflector events inside the gun need the joint accelerating operator")
    entrance = specimen_entrance_z_mm(working)
    exit_z = float(working.sample.z_mm)+float(working.sample.thickness_nm)*.5e-6
    planes, targets = _recording_planes(working, request.detector_key, request.detector_keys) if request.stop == "detector" else ((), ())
    target = targets[-1] if targets else None
    end = {"gun_exit": gun_end, "specimen_entrance": entrance,
           "specimen_exit": exit_z,
           "detector": None if target is None else target.z_mm,
           "plane": request.observation_z_mm}[request.stop]
    observation_stops, observation_vacuum = (), None
    if request.stop == "plane":
        end = float(end)
        if end < gun_end:
            raise ValueError("Virtual observation inside the gun needs a truncated coherent gun operator; this stage starts at the executed gun exit")
        from temsim.specimen.scene import SpecimenScene
        observation_vacuum = SpecimenScene.from_state(working).is_vacuum
        if not observation_vacuum and entrance < gun_end and end > entrance:
            raise ValueError("A specimen inside the gun requires a joint coherent gun/specimen operator")
        if not observation_vacuum and entrance < end < exit_z:
            raise ValueError("Virtual observation inside the specimen needs a truncated specimen wave operator")
        observation_stops = _observation_stops(working, gun_end, end, entrance, exit_z, observation_vacuum)
    if bool(getattr(working, "energy_filter_installed", False)):
        boundary = float(working.energy_filter.entrance_z_mm)
        if not math.isfinite(boundary):
            raise ValueError("Installed energy filter needs a finite physical entrance boundary")
        if end >= boundary:
            raise ValueError(f"The installed energy filter requires its coherent field operator at z={boundary:.9g} mm; "
                             f"the requested path to z={end:.9g} mm cannot skip it")
    if unified:
        from temsim.physics.unified_field_runtime import unified_runtime_preflight
        admission = unified_runtime_preflight(working)
        if admission["issues"]:
            raise ValueError("Finite-Field Optics: "+"; ".join(admission["issues"]))
    # Both routes resolve derived drives before capturing fields. The unified
    # mode measures the actual shared-potential response; it never invokes the
    # historical canonical/equivalent-image calibration. A paired execution
    # reuses the verified particle calibration through the preparation cache.
    from temsim.physics.scan_preparation import prepare_scan_drives
    from temsim.physics.transport_progress import transport_progress
    from temsim.calculation_progress import LocalProgressLabel
    def scan_progress(label):
        if cancelled():
            raise InterruptedError("Coherent scan calibration cancelled")
        if progress_callback is not None:
            local = getattr(label, "local_progress", None)
            progress_callback(0 if local is None else local.completed,
                0 if local is None or local.total is None else local.total, label)
    def scan_cancelled():
        if cancelled():
            raise InterruptedError("Coherent scan calibration cancelled")
        return False
    original_cancelled = getattr(working, "_tuning_cancelled", None)
    working._tuning_cancelled = scan_cancelled
    try:
        scan_progress(LocalProgressLabel("Resolving shared-potential scan calibration" if unified
            else "Resolving scan controls", 0, None))
        with transport_progress(scan_progress):
            prepare_scan_drives(working, instrument_digest, observation_stop_z_mm=end,
                                session_cache=scan_preparation_cache)
    finally:
        if original_cancelled is None:
            vars(working).pop("_tuning_cancelled", None)
        else:
            working._tuning_cancelled = original_cancelled
    if unified and request.stop != "gun_exit":
        # Admit the real captured column before executing or reading a costly
        # gun cache. Unsupported fields never switch to a replacement source
        # or silently run the historical empirical column executor.
        from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
        unified_wave = UnifiedPotentialWaveAdapter(working, request, stop_z_mm=end)
    if request.stop not in ("plane", "gun_exit"):
        admit_column(gun_end, entrance)
        if target is not None:
            admit_column(working.sample.z_mm+working.sample.thickness_nm*.5e-6, target.z_mm)
    propagation_id = json_digest({"instrument": instrument_digest, "source": asdict(request.source),
                                 "gun": asdict(request.gun), "surface": asdict(request.surface),
                                 "radial_gun": asdict(request.radial_gun),
                                 **({"source_nominal_energy_preview": True} if request.source_nominal_energy_preview else {})})
    store = ExecutedWaveStore(request.execution.cache_directory, propagation_id, request.execution.maximum_disk_cache_bytes)
    if request.execution.segmented:
        _STAGES.clear()
        from temsim.physics.tip_gun_wave import _CACHE
        _CACHE.clear()
    cache_hit = False
    epoch = float(getattr(working, "simulation_time_s", 0.) if request.tip_time_s is None else request.tip_time_s)
    observation_identity = None
    if request.stop == "plane":
        numerics = asdict(request)
        numerics.pop("observation_z_mm")
        if not request.column_preview_phase_reselection:
            numerics.pop("column_preview_phase_reselection")
        if not request.source_nominal_energy_preview:
            numerics.pop("source_nominal_energy_preview")
        observation_identity = json_digest({"tip_execution": propagation_id,
            "request_without_observation_z": numerics, "tip_epoch_s": epoch,
            "unified_executor": None if unified_wave is None else unified_wave.identity})
    def remember_observation(checkpoint):
        if use_cache and observation_identity is not None:
            _register_observation_prefix(observation_identity, checkpoint, store, request,
                cancelled=cancelled, completed_validator=completed_validator)
    completed_validator = (lambda checkpoint: unified_wave.column(checkpoint,
        checkpoint.plane_z_mm, cancelled=cancelled)) if (
            unified_wave is not None and request.column_weighted_resampling
            and request.execution.segmented) else None
    def stage(name, producer, *, inputs=()):
        nonlocal cache_hit
        # Both branches consume the identical executed source. All downstream
        # caches are separately bound to the explicit idealisation policy.
        if request.column_model != "physical" and name != "gun_exit":
            from temsim.physics.ideal_column_reference import column_model_record
            inputs = (*inputs, column_model_record(request.column_model))
        if unified_wave is not None and name != "gun_exit":
            inputs = (*inputs, unified_wave.identity)
        key = (propagation_id, name, json_digest(inputs))
        if cancelled():
            raise InterruptedError("Tip wave request cancelled")
        if use_cache and request.execution.segmented:
            cached = store.get(store.key(*key))
            if cached is not None:
                cache_hit = True
                return cached
        if use_cache and not request.execution.segmented and key in _STAGES:
            _STAGES.move_to_end(key)
            cache_hit = True
            return _STAGES[key][0]
        result = producer()
        verify()  # Recheck source implementation and external bytes before caching.
        if cancelled():
            raise InterruptedError("Tip wave request cancelled")
        if request.execution.segmented:
            result = store.put(store.key(*key), result)
        elif use_cache:
            _retain(key, result, request.execution.maximum_ram_cache_bytes)
        return result
    round_attempted_inputs = set()
    def column(upstream, stop, *, allow_acceptance=True):
        nonlocal cache_hit
        from temsim.physics.wave_mode_support import (
            MODE_SUPPORT_SCHEMA, incoherent_mode_approximation, select_incoherent_modes,
        )
        # Preview changes the column lattice, never removes source modes.
        mode_budget = (0. if request.column_exploratory_preview else
                       request.wave_grid.maximum_incoherent_mode_loss_probability)
        if incoherent_mode_approximation(upstream.record) is None:
            original = upstream
            upstream = stage("incoherent_mode_support", lambda: select_incoherent_modes(original,
                maximum_loss_probability=mode_budget),
                inputs=(original.digest, MODE_SUPPORT_SCHEMA, mode_budget))
        else:
            # Includes resumed prefixes and material branches. The budget is
            # spent once on the complete gun mixture, not once per segment.
            upstream = select_incoherent_modes(upstream, maximum_loss_probability=mode_budget)
        if unified_wave is not None:
            result = stage("unified_column", lambda: unified_wave.column(upstream, stop,
                cancelled=cancelled, progress_callback=progress_callback,
                **({"store": store, "use_cache": use_cache, "verify": verify,
                    "checkpoint_callback": remember_observation}
                   if request.execution.segmented and request.column_weighted_resampling else {})),
                inputs=(upstream.digest, float(stop), request.column_step_mm,
                        request.wave_grid.column_identity()))
            remember_observation(result)
            return result
        # The executed gun is the only Cartesian-to-radial entry point.
        # Qualify all modes before the optional aperture-conditioned path;
        # nested acceptance calls must not retry a declined input mixture.
        has_radial = "radial_output_modes" in upstream.record
        at_gun_exit = float(upstream.plane_z_mm) == float(gun_end)
        if (has_radial or at_gun_exit) and upstream.digest not in round_attempted_inputs:
            from temsim.physics.round_column_checkpoint import execute_round_prefix
            original = upstream
            round_attempted_inputs.add(original.digest)
            upstream = stage("round_column_prefix", lambda: execute_round_prefix(working, original, stop,
                maximum_step_mm=request.column_step_mm, numerics=request.radial_column,
                grid_numerics=request.wave_grid, cancelled=cancelled, progress_callback=progress_callback,
                allow_cartesian=at_gun_exit, **column_options),
                inputs=("executed-cartesian-round-routing-v2", original.digest, stop,
                        request.column_step_mm, asdict(request.radial_column),
                        request.wave_grid.column_identity()))
            if upstream.plane_z_mm != original.plane_z_mm:
                remember_observation(upstream)
            else:
                # Stored wrappers have a different digest from their input;
                # mark that unchanged physical input as already attempted too.
                round_attempted_inputs.add(upstream.digest)
            if upstream.plane_z_mm == stop:
                return upstream
        if request.acceptance_window.enabled and allow_acceptance:
            from temsim.physics.column_acceptance_window import (
                SCHEMA, AcceptanceWindowUnavailable, find_acceptance_span, execute_acceptance_span,
            )
            prepared = prepared_columns.pop((float(upstream.plane_z_mm), float(stop)), None)
            if prepared is None:
                prepared = _prepare_column(working, upstream.plane_z_mm, stop, request.column_step_mm, **column_options)
            span = find_acceptance_span(prepared)
            if span is not None:
                if span["start_z_mm"] > upstream.plane_z_mm:
                    upstream = column(upstream, span["start_z_mm"], allow_acceptance=False)
                original = upstream
                try:
                    upstream = stage("acceptance_conditioned_aperture", lambda: execute_acceptance_span(
                        working, original, span, numerics=request.acceptance_window,
                        grid_numerics=request.wave_grid, maximum_step_mm=request.column_step_mm,
                        cancelled=cancelled, progress_callback=progress_callback),
                        inputs=(original.digest, SCHEMA, span["identity"], asdict(request.acceptance_window),
                                request.wave_grid.column_identity(), request.column_step_mm))
                except AcceptanceWindowUnavailable:
                    # Eligibility failure never changes the physical fallback.
                    return column(original, stop, allow_acceptance=False)
                remember_observation(upstream)  # only AFTER its real receiving aperture
                if upstream.plane_z_mm == stop:
                    return upstream
                return column(upstream, stop, allow_acceptance=False)
            prepared_columns[(float(upstream.plane_z_mm), float(stop))] = prepared
        prepared = prepared_columns.pop((float(upstream.plane_z_mm), float(stop)), None)
        if request.execution.segmented:
            result, hit = _propagate_column_segmented(working, upstream, stop, store=store,
                segment_steps=request.execution.segment_steps, maximum_step_mm=request.column_step_mm,
                grid_numerics=request.wave_grid, tip_time_s=epoch, verify=verify,
                cancelled=cancelled, progress_callback=progress_callback, use_cache=use_cache,
                checkpoint_callback=remember_observation if observation_identity is not None else None,
                _prepared=prepared, **column_options)
            cache_hit |= hit
            return result
        return _propagate_column(working, upstream, stop, maximum_step_mm=request.column_step_mm,
            grid_numerics=request.wave_grid, tip_time_s=epoch, cancelled=cancelled,
            progress_callback=progress_callback, _prepared=prepared, **column_options)
    def compute_gun():
        model = getattr(working.electron_gun.emitter, "surface_model", None)
        if model is not None and model.coherence is not None:
            from temsim.physics.surface_gun_wave import build_surface_gun_checkpoint
            radial = replace(request.radial_gun, field_step_mm=request.gun.field_step_mm,
                bore_step_mm=request.gun.bore_step_mm, maximum_steps=request.gun.max_steps,
                maximum_working_bytes=request.gun.maximum_checkpoint_bytes)
            checkpoint, near = build_surface_gun_checkpoint(working.electron_gun, surface=request.surface,
                radial=radial, grid_pixels=request.source.grid_pixels, column_state=working,
                cancelled=cancelled, progress_callback=progress_callback,
                _energy_cache=store, _reuse_energy_cache=use_cache)
            side = float(checkpoint.record["unresolved_side_fraction"])
            if side > request.surface.flux_tolerance:
                raise ValueError(f"The joint tip/gun solve has an unresolved side-wave fraction of {side:.6g}. "
                    "This is not physical absorption; the full-state pipeline cannot discard it. "
                    "Extend and converge the joint domain or transport its lateral complex channel before TEM/STEM readout.")
            return checkpoint
        return build_tip_gun_checkpoint(working.electron_gun, source_numerics=request.source,
            numerics=request.gun, _column_state=working, use_cache=use_cache and not request.execution.segmented,
            cancelled=cancelled, progress_callback=progress_callback,
            **({"nominal_energy_preview": True} if request.source_nominal_energy_preview else {}))
    prefix = (_find_observation_prefix(observation_identity, gun_end, end, store,
        segmented=request.execution.segmented, cancelled=cancelled,
        completed_validator=completed_validator)
        if use_cache and request.stop == "plane" else None)
    if request.stop == "plane":
        # A verified executed prefix already contains all upstream fields and
        # physical stops. Admit only its unexecuted continuation: moving Z
        # must not rebuild the complete gun-to-observation field plan.
        from temsim.physics.wave_field_admission import require_supported_column_wave_fields
        start = gun_end if prefix is None else prefix.plane_z_mm
        spans = ([(start, end)] if observation_vacuum else [
            (start, min(end, entrance)),
            (max(start, entrance), min(end, exit_z)),
            (max(start, exit_z), end)])
        for lower, upper in spans:
            if upper > lower:
                prepared = admit_column(lower, upper)
                if unified_wave is None:
                    require_supported_column_wave_fields(working, lower, upper, prepared[0])
    result = stage("gun_exit", compute_gun) if prefix is None else prefix
    if prefix is not None:
        cache_hit = True
    resume_z_mm = result.plane_z_mm
    del prefix
    if request.stop == "plane":
        remember_observation(result)
        def advance_column(upstream, stop):
            if float(stop) == upstream.plane_z_mm:
                return (unified_wave.column(upstream, float(stop), cancelled=cancelled,
                            progress_callback=progress_callback) if unified_wave is not None else upstream)
            if request.execution.segmented:
                result = column(upstream, float(stop))
            else:
                # This outer cache can bypass column()'s acceptance admission.
                # Include its policy even when the previous span needed no window.
                result = stage("observation_column", lambda: column(upstream, float(stop)),
                    inputs=(upstream.digest, float(stop), request.column_step_mm,
                            request.wave_grid.column_identity(), asdict(request.radial_column),
                            asdict(request.acceptance_window), epoch))
            remember_observation(result)
            return result

        def advance_observation(upstream, stop):
            if observation_vacuum or stop <= entrance:
                return advance_column(upstream, stop)
            if upstream.plane_z_mm < entrance:
                upstream = advance_column(upstream, entrance)
            if upstream.plane_z_mm < exit_z:
                if unified_wave is not None:
                    incident = upstream
                    upstream = stage("unified_specimen_exit", lambda: unified_wave.specimen(incident,
                        store=store, use_cache=use_cache, cancelled=cancelled,
                        progress_callback=progress_callback),
                        inputs=(incident.digest, asdict(request.inelastic),
                                request.wave_grid.column_identity(), request.column_step_mm))
                elif request.inelastic.method == "trajectories":
                    from temsim.physics.inelastic_wave import _propagate_inelastic_specimen
                    upstream = _propagate_inelastic_specimen(working, upstream, numerics=request.inelastic,
                        store=store, maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid,
                        tip_time_s=epoch, cancelled=cancelled, progress_callback=progress_callback,
                        verify=verify, use_cache=use_cache, **column_options)
                else:
                    incident = upstream
                    upstream = stage("specimen_exit", lambda: _propagate_specimen(working, incident,
                        maximum_checkpoint_bytes=request.gun.maximum_checkpoint_bytes,
                        maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid, tip_time_s=epoch,
                        cancelled=cancelled, progress_callback=progress_callback, **column_options),
                        inputs=(incident.digest, asdict(request.inelastic), request.column_step_mm,
                                asdict(request.wave_grid), epoch))
                remember_observation(upstream)
            return advance_column(upstream, stop)

        for plane in observation_stops:
            if float(plane.z_mm) < resume_z_mm:
                continue  # Absorption at these devices is already executed.
            if (float(plane.z_mm) == resume_z_mm
                    and _recording_stop_already_executed(result, plane.key)):
                continue  # Same-Z prefix is the transmitted side, not incident.
            result = advance_observation(result, float(plane.z_mm))
            incident = result
            if request.execution.segmented:
                from temsim.detector.wave_readout import _recording_stop_streamed
                result = _recording_stop_streamed(incident, plane, store, verify=verify,
                    cancelled=cancelled, use_cache=use_cache)
            else:
                result = stage("detector_transmitted:"+plane.key,
                    lambda: _apply_recording_stop(incident, plane), inputs=(incident.digest,))
            del incident
        result = advance_observation(result, end)
        verify()
        if cancelled():
            raise InterruptedError("Virtual-plane wave request cancelled before publication")
        remember_observation(result)
        from temsim.physics.scan_preparation import scan_drive_identity
        return TipWaveResult(result, None, request, cache_hit, instrument_digest,
                             resolved_scan_identity=scan_drive_identity(working, emission_time_s=epoch))
    if request.stop != "gun_exit":
        result = column(result, entrance)
    if request.stop in ("specimen_exit", "detector"):
        upstream = result
        from temsim.specimen.scene import SpecimenScene
        if SpecimenScene.from_state(working).is_vacuum:
            stop = working.sample.z_mm+working.sample.thickness_nm*.5e-6
            result = column(upstream, stop) if stop > upstream.plane_z_mm else upstream
        elif unified_wave is not None:
            result = stage("unified_specimen_exit", lambda: unified_wave.specimen(upstream,
                store=store, use_cache=use_cache, cancelled=cancelled,
                progress_callback=progress_callback),
                inputs=(upstream.digest, asdict(request.inelastic),
                        request.wave_grid.column_identity(), request.column_step_mm))
        elif request.inelastic.method == "trajectories":
            from temsim.physics.inelastic_wave import _propagate_inelastic_specimen
            result = _propagate_inelastic_specimen(working, upstream, numerics=request.inelastic,
                store=store, maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid,
                tip_time_s=epoch, cancelled=cancelled, progress_callback=progress_callback,
                verify=verify, use_cache=use_cache, **column_options)
        else:
            result = stage("specimen_exit", lambda: _propagate_specimen(working, upstream,
                maximum_checkpoint_bytes=request.gun.maximum_checkpoint_bytes,
                maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid, tip_time_s=epoch,
                cancelled=cancelled, progress_callback=progress_callback, **column_options),
                inputs=(upstream.digest, asdict(request.inelastic), request.column_step_mm, asdict(request.wave_grid), epoch))
        del upstream
    readouts = []
    if target is not None:
        from temsim.detector.wave_readout import read_wave_detector_streamed
        reader = read_wave_detector_streamed if request.execution.segmented else read_wave_detector
        selected = {d.key for d in targets}
        # All readouts coexist until the consumer saves/reduces this dwell.
        # Divide the numerical output budget, never electron probability.
        readout_budget = request.maximum_readout_bytes // len(targets)
        for plane in planes:
            upstream = result
            result = column(upstream, float(plane.z_mm))
            del upstream
            if plane.key in selected:
                readouts.append(reader(result, plane, request.readout, pixels=request.detector_pixels,
                                       maximum_bytes=readout_budget))
            if plane.key != target.key:
                incident = result
                if request.execution.segmented:
                    from temsim.detector.wave_readout import _recording_stop_streamed
                    result = _recording_stop_streamed(incident, plane, store, verify=verify,
                        cancelled=cancelled, use_cache=use_cache)
                else:
                    result = stage("detector_transmitted:"+plane.key, lambda: _apply_recording_stop(incident, plane), inputs=(incident.digest,))
                del incident
    verify()
    from temsim.physics.scan_preparation import scan_drive_identity
    return TipWaveResult(result, readouts[-1] if readouts else None, request, cache_hit,
                         instrument_digest, tuple(readouts),
                         scan_drive_identity(working, emission_time_s=epoch))
