"""Actual executed-gun handoff to the admitted common-potential wave solver.

This adapter changes the column executor, never the electron source. Constant
electrostatic continuation is removed only by an exact provider contract, using
the incoming mode's actual kinetic energy. Its original domain is retained by
a zero-field mathematical guard. A weak accelerating tail is not constant.
"""
from dataclasses import dataclass, replace
import math

import numpy as np

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.gun_magnetic_potential import GunMagneticPotential
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.potential_geometry import observe_potential_geometry_checkpoint
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
from temsim.physics.potential_wave_carrier import _admit
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_wave_policy import potential_wave_operator_policy, potential_support_policy
from temsim.physics.unified_wave_support import (
    prepare_unified_support, stamp_unified_support, unified_support_summary,
)
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_plane_observables import ColumnPlaneMagneticGauge
from temsim.physics.wave_device import reusable_device_scope


@dataclass(frozen=True)
class UnifiedPlaneMagneticGauge(ColumnPlaneMagneticGauge):
    """Probability-current A from the exact executed common potential."""
    snapshot: object = None
    executed_z_m: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if type(self.snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot) or self.fields or self.aligned_bz_t != 0.:
            raise ValueError("Unified probability flow requires only its executed potential snapshot")
        if self.executed_z_m is None or not math.isfinite(self.executed_z_m):
            raise ValueError("Unified probability flow requires its exact executed SI plane")

    @property
    def fingerprint(self):
        return json_digest(("unified-plane-gauge-v1", self.snapshot.numerical_digest, self.plane_z_mm, self.executed_z_m))

    def require_plane(self, plane_z_mm):
        if plane_z_mm != self.plane_z_mm:
            raise ValueError("Unified probability flow requires the exact executed observation plane")

    def vector_potential_xy_t_m(self, coordinates_m):
        xy = np.asarray(coordinates_m, float)
        if xy.ndim < 2 or xy.shape[0] != 2 or not np.isfinite(xy).all():
            raise ValueError("Unified magnetic gauge coordinates must be finite global XY arrays")
        xyz = np.moveaxis(np.stack((xy[0], xy[1], np.full_like(xy[0], self.executed_z_m))), 0, -1)
        source = self.snapshot
        if type(source) is DynamicPotentialSnapshot:
            if any(not np.array_equal(coil.base.base.field.registration.rotation_array[:, 2], (0., 0., 1.))
                   for coil in source.dynamic_coils):
                raise ValueError("Time-dependent transverse readout gauge requires a mode-specific clock")
            # Admitted axial coils have A=(0,0,Az) at EVERY time. Their exact
            # transverse contribution is zero, including offset/rotated XY.
            source = source.static_snapshot
        return np.moveaxis(source.sample(xyz)["A_t_m"][..., :2], -1, 0)


def unified_checkpoint_magnetic_gauge(checkpoint):
    """Return no gauge when the actual new executor did not preserve one."""
    if not checkpoint.record.get("unified_column_executed", False):
        return None
    raw = checkpoint.record.get("unified_wave_snapshot")
    if raw is None:
        return None
    from temsim.physics.dynamic_potential import SCHEMA as DYNAMIC_SCHEMA
    snapshot = (DynamicPotentialSnapshot if raw.get("schema") == DYNAMIC_SCHEMA else ElectromagneticPotentialSnapshot).from_dict(raw)
    if snapshot.numerical_digest != checkpoint.record.get("snapshot_digest"):
        raise ValueError("Unified wave readout potential identity does not match its executed checkpoint")
    return UnifiedPlaneMagneticGauge(checkpoint.plane_z_mm, 0., snapshot=snapshot,
                                     executed_z_m=checkpoint.record.get("unified_executed_plane_z_m"))


def _static_wave_snapshot(runtime):
    """Prove each excluded action exactly; retain the original bundle identity."""
    rows, reductions = [], []
    original = (runtime.snapshot.static_snapshot if type(runtime.snapshot) is DynamicPotentialSnapshot else runtime.snapshot)
    for row in original.contributions:
        provider = row.potential
        if type(provider) is ElectricPotentialSnapshot:
            metadata, arrays = provider.metadata, provider.arrays
            continuation = metadata["continuation"]
            if (continuation["kind"] != "grounded_downstream"
                    or runtime.start_z_m < float(arrays["z"][-1])):
                if metadata["basis"] != "axisymmetric-r2-z-bilinear-v1":
                    raise ValueError("Accelerating coherent propagation requires the qualified native r-squared/Z scalar basis")
                rows.append(row)
                continue  # Retain actual phi; the executed mode establishes its energy reference.
            domain = metadata["domain"]
            if ((domain["z_min_m"] is not None and runtime.start_z_m < domain["z_min_m"])
                    or (domain["z_max_m"] is not None and runtime.stop_z_m > domain["z_max_m"])):
                raise ValueError("Unified coherent interval exceeds the constant electric provider domain")
            radius = min(float(arrays["r"][-1]), float(domain["radius_m"]))
            guard_key = "constant-electric-domain:"+row.source_id
            native = FrozenAnalyticField(guard_key, ((0., 0., 1.),), CoordinateRegistration(),
                                        (-1000., 1000.), radius)
            guard = HighOrderRoundLensPotential(native, radial_order=3, axial_domain="gaussian_all_z")
            rows.append(PotentialContribution(guard_key, (guard_key,), guard, "magnetic"))
            reductions.append({"source_id": row.source_id, "provider_digest": provider.digest,
                "proof": "native grounded_downstream constant scalar potential throughout the complete interval",
                "constant_potential_rise_v": continuation["value_v"],
                "reference": "incoming actual kinetic energy; constant electrostatic gauge offset is not added twice",
                "domain_guard_source_id": guard_key, "radius_m": radius})
        elif type(provider) is GunMagneticPotential:
            # Native soft windows have exact compact support; no threshold is
            # applied to a nonzero field or fringe tail.
            if not all(runtime.start_z_m > term[0]+term[1]+term[2]
                       or runtime.stop_z_m < term[0]-term[1]-term[2]
                       for term in provider.terms):
                raise ValueError("Unified coherent propagation does not admit overlapping gun alignment fields")
            reductions.append({"source_id": row.source_id, "provider_digest": provider.digest,
                               "proof": "complete interval is outside native compact gun alignment support"})
        else:
            rows.append(row)
    snapshot = ElectromagneticPotentialSnapshot(runtime.snapshot.scene, tuple(rows), runtime.snapshot.gauge)
    _admit(ElectromagneticPotentialSnapshot(snapshot.scene,
        tuple(row for row in snapshot.contributions if row.kind == "magnetic"), snapshot.gauge))
    if type(runtime.snapshot) is DynamicPotentialSnapshot:
        snapshot = DynamicPotentialSnapshot(snapshot, runtime.snapshot.dynamic_coils)
    return snapshot, freeze_json(reductions)


def _require_material_history(record, *, required=False):
    """Keep ordinary downstream/readout paths behind the material receipt gate."""
    known = record.get("schema") == "potential-material-slice-v1"
    present = any(key in record for key in (
        "material_domain_policy", "material_execution_identity", "column_transport_identity"))
    if required and not all(key in record for key in ("material_domain_policy", "material_execution_identity")):
        raise ValueError("Material history lost its domain policy; restart from the material entrance")
    if required or known or present:
        from temsim.physics.potential_specimen_transport import PotentialMaterialTransport
        PotentialMaterialTransport.context(record)


def _recording_stop_ancestry(record):
    """Return the propagation ancestor and chronological physical stop receipts.

    Streaming changes only payload storage. It must not hide an already
    executed field history or turn a detector exit into a fresh source.
    Older streamed receipts omit readout metadata; their physical losses
    and upstream identity remain complete and must not be synthesized anew.
    """
    stops = []
    while record.get("schema") in ("executed-recording-stop-v1", "executed-recording-stop-stream-v1"):
        stops.append(record)
        record = record["upstream"]
    return record, tuple(reversed(stops))


def _mode_context(checkpoint, mode):
    """Split aggregate geometry bookkeeping without duplicating shared losses."""
    ancestor, recording = _recording_stop_ancestry(checkpoint.record)
    stops = [row for stop in recording for row in stop["modes"] if row["mode_id"] == mode.mode_id]
    contexts = ancestor.get("unified_mode_contexts")
    if contexts is not None:
        if mode.mode_id not in contexts:
            raise ValueError("Unified material mode lost its physical propagation history")
        record = thaw_json(contexts[mode.mode_id])
        _require_material_history(record, required="material_histories" in ancestor)
        if stops:
            record.setdefault("external_recording_losses", []).extend(thaw_json(stops))
            geometry = record.get("potential_geometry")
            if geometry is not None:
                received = math.fsum(row["received_weight"] for row in stops)
                geometry["entry_weight_per_reference"] -= received
                geometry["output_weight_per_reference"] = mode.weight_per_reference_electron
        return record
    record = thaw_json(checkpoint.record)
    _require_material_history(record, required="material_histories" in ancestor)
    geometry = record.get("potential_geometry")
    if geometry is not None and len(checkpoint.beam.modes) > 1:
        losses = [row for row in geometry["physical_losses"] if row["mode_id"] == mode.mode_id]
        lost = math.fsum(row["lost_probability_per_reference"] for row in losses)
        geometry.update(physical_losses=losses, physical_loss_per_reference=lost,
            output_weight_per_reference=mode.weight_per_reference_electron,
            entry_weight_per_reference=mode.weight_per_reference_electron+lost)
    return record


def _source_provenance_only(record):
    """Recognize an initial gun record with no downstream execution history.

    The complete immutable producer record belongs to the aggregate upstream
    receipt. Repeating it in every mode and every optical step adds no evidence.
    Unknown records and any existing continuation ledger retain their original
    path; this is never a way to discard an invalid or incomplete history.
    """
    original = record
    if record.get("schema") == "executed-incoherent-mode-support-v1":
        record = record.get("upstream", {})
    if record.get("schema") != "executed-tip-gun-v1":
        return False
    execution_keys = ("unified_wave_identity", "unified_mode_contexts", "potential_geometry",
        "kinetic_phase_ledger", "electric_wave_history", "support_ledger",
        "material_kinetic_history", "material_kinetic_native_plane", "material_domain_policy", "material_execution_identity",
        "material_histories", "column_transport_identity",
        "inherited_potential_support", "support_prefix_digest", "applied_event_ids",
        "potential_operator_policy", "potential_support_policy", "unified_support_allocation", "unified_support_ledger_digest",
        "representation_estimates")
    return not any(key in value for value in (original, record) for key in execution_keys)


_PLANE_ARRAYS = ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad")


def _host_allocation(array):
    """Count one immutable backing allocation, including shared array views."""
    owner = array
    while isinstance(owner, np.ndarray) and owner.base is not None:
        owner = owner.base
    while isinstance(owner, memoryview):
        owner = owner.obj
    size = owner.nbytes if isinstance(owner, np.ndarray) else memoryview(owner).nbytes
    return ("host", id(owner)), int(size)


def _plane_allocations(plane):
    return dict(_host_allocation(value) for name in _PLANE_ARRAYS
                if (value := getattr(plane, name)) is not None)


def _stored_allocations(root, arrays):
    # The committed store already validates the manifest. Reserve hidden
    # payloads by that manifest; do not load modes or near-tip traces here.
    return {("stored", str(root), row["file"], row["sha256"]): int(row["nbytes"])
            for row in arrays.values()}


def _checkpoint_allocations(checkpoint):
    from temsim.physics.wave_checkpoint_store import _StoredBeam, _StoredAuxiliary
    result = {}
    if isinstance(checkpoint.beam, _StoredBeam):
        for row in checkpoint.beam.modes.rows:
            result.update(_stored_allocations(checkpoint.beam.modes.root, row["arrays"]))
    else:
        for mode in checkpoint.beam.modes:
            result.update(_plane_allocations(mode.plane))
    auxiliary = checkpoint.auxiliary_arrays
    if isinstance(auxiliary, _StoredAuxiliary):
        result.update(_stored_allocations(auxiliary.reader.root, auxiliary.rows))
    else:
        result.update(_host_allocation(value) for value in auxiliary.values())
    return result


def _column_mode_grid(numerics, source, current, outputs, index):
    """Reserve retained modes once, outside the active operator's workspace.

    The original input, current chunk input and completed outputs remain live.
    Lazy input payloads are conservatively reserved by manifest byte counts,
    without paging them in. The active mode is already included in the
    operator's workspace estimate, so its shared buffers are excluded here.
    """
    from temsim.physics.wave_checkpoint_store import _StoredBeam
    from temsim.physics.wave_grid import WaveMemoryBudgetError
    allocations = _checkpoint_allocations(source)
    allocations.update(_checkpoint_allocations(current))
    for mode in outputs:
        allocations.update(_plane_allocations(mode.plane))
    if isinstance(current.beam, _StoredBeam):
        arrays = current.beam.modes.rows[index]["arrays"]
        active = _stored_allocations(current.beam.modes.root, arrays)
        shape = tuple(arrays["amplitude"]["shape"])
    else:
        plane = current.beam.modes[index].plane
        active, shape = _plane_allocations(plane), plane.amplitude.shape
    retained = sum(size for key, size in allocations.items() if key not in active)
    available = numerics.maximum_working_bytes-retained
    if available <= 0:
        raise WaveMemoryBudgetError("Unified retained source/current/output modes need "
            f"{retained} bytes before the active operator, above the declared host "
            f"working limit {numerics.maximum_working_bytes}; no mode was discarded")
    local = replace(numerics, maximum_working_bytes=available)
    # Admit one immutable load and validation before a lazy mode is read.
    # The actual field/canonical operators retain their larger, original checks.
    local.check_working_memory(shape, working_bytes=64*math.prod(shape))
    return local, {"declared_host_working_bytes": numerics.maximum_working_bytes,
        "retained_external_bytes": retained, "active_operator_working_bytes": available,
        "scope": "unique original source, current input and completed output buffers; lazy payloads reserved from verified manifests; active mode counted by the operator"}


class UnifiedPotentialWaveAdapter:
    """One fixed bundle, actual source, and non-resetting per-history budgets."""

    def __init__(self, state, request, *, runtime=None, stop_z_mm=None):
        from temsim.physics.unified_field_runtime import get_unified_field_runtime
        from temsim.physics.wave_device import normalise_backend
        from temsim.physics.potential_kinetic_expansion import validate_kinetic_order
        request.validate()
        self.kinetic_order = validate_kinetic_order(request.column_kinetic_order)
        if type(request.column_support_projection) is not bool:
            raise ValueError("column_support_projection must be an explicit boolean")
        self.support_projection = request.column_support_projection
        if type(request.column_weighted_resampling) is not bool:
            raise ValueError("column_weighted_resampling must be an explicit boolean")
        if request.column_weighted_resampling and not self.support_projection:
            raise ValueError("Weighted coherent resampling requires controlled support projection")
        from temsim.simulation_modes import is_unified
        if request.source_nominal_energy_preview and not is_unified(state):
            raise ValueError("Nominal-energy source preview is available only in unified Finite-Field Optics")
        if (request.column_preview_segment_resampling or request.column_preview_phase_reselection) and not is_unified(state):
            raise ValueError("Preview segment resampling is available only in unified Finite-Field Optics")
        self.support_projection_policy = ("preview_phase_segment_and_retry" if request.column_preview_phase_reselection else
                                          "preview_segment_and_retry" if request.column_preview_segment_resampling else
                                          "weighted_segment_and_retry" if request.column_weighted_resampling else
                                          "step_boundaries_and_retry" if self.support_projection else None)
        self.operator_options = {"kinetic_order": self.kinetic_order, "reference_quadrature": "cf4",
            "scalar_splitting": self.kinetic_order in (6, 8),
            "higher_order_scalar_splitting": self.kinetic_order in (6, 8),
            "residual_exponential_method": request.column_exponential_method,
            "maximum_chebyshev_group": request.column_chebyshev_group}
        if request.column_model != "physical":
            raise ValueError("Unified potentials require the physical column; the legacy fitted first-order reference is not an equivalent executor")
        preview_overrides = {
            "maximum_support_loss_fraction": request.column_preview_support_probability,
            "maximum_accumulated_support_l2": request.column_preview_support_l2,
        }
        preview_overrides = {name: value for name, value in preview_overrides.items() if value is not None}
        if preview_overrides:
            from temsim.simulation_modes import is_unified
            if not (request.column_exploratory_preview and request.column_weighted_resampling
                    and self.support_projection and is_unified(state)):
                raise ValueError("Column preview support budgets require explicit weighted exploratory preview in unified Finite-Field Optics")
            preview_overrides["exploratory_support_budget"] = True
        requested_backend = normalise_backend(request.wave_grid.compute_backend)
        effective_pixels = (min(request.wave_grid.maximum_pixels, request.column_preview_pixels)
                            if request.column_exploratory_preview else request.wave_grid.maximum_pixels)
        self.column_grid = replace(request.wave_grid, compute_backend="CPU", maximum_pixels=effective_pixels,
            maximum_incoherent_mode_loss_probability=(0. if request.column_exploratory_preview else
                request.wave_grid.maximum_incoherent_mode_loss_probability),
            angular_domain_padding="minimum_full_band", **preview_overrides).validate()
        # Material descendants inherit the same original support allowance.
        # Preserve their separate backend, domain and quadrature controls.
        self.material_grid = (replace(request.wave_grid, **preview_overrides).validate()
                              if preview_overrides else request.wave_grid)
        self.numerical_profile = freeze_json({
            "schema": "unified-wave-numerical-profile-v1",
            "exploratory_preview": request.column_exploratory_preview,
            "label": "Exploratory preview" if request.column_exploratory_preview else "Controlled calculation",
            "accuracy_qualified": False,
            **({"source_nominal_energy_preview": True,
                "source_energy_approximation": "one declared nominal emission energy; configured source spread remains unchanged; not an energy-distribution convergence result"}
               if request.source_nominal_energy_preview else {}),
            "requested_preview_pixels": request.column_preview_pixels,
            "global_maximum_pixels": request.wave_grid.maximum_pixels,
            "effective_column_maximum_pixels": effective_pixels,
            "maximum_support_probability_fraction": self.column_grid.maximum_support_loss_fraction,
            "maximum_accumulated_support_l2": self.column_grid.maximum_accumulated_support_l2,
            "maximum_incoherent_mode_loss_probability": self.column_grid.maximum_incoherent_mode_loss_probability,
            **({"column_preview_segment_resampling": True,
                "segment_resampling_scope": "one voluntary projection at the first residual of each actual segment; original one-percent event limits; no budget renewal"}
               if request.column_preview_segment_resampling else {}),
            **({"column_preview_phase_reselection": True,
                "phase_preparation_backend": "CPU",
                "phase_preparation_maximum_pixels": min(65536, 2*effective_pixels),
                "phase_preparation_memory": "unchanged host working limit including retained owners",
                "phase_representation_error": "estimated, not a certified bound; shares the original complex L2 allowance",
                "phase_representation_estimator": "two_dense_grids_shifted_relative_5percent_absolute_1e-7_norm_charged_v2"}
               if request.column_preview_phase_reselection else {}),
            **({"support_budget_scope": "explicit post-gun exploratory override; original gun numerics unchanged",
                "support_budget_overrides": {
                    "probability_fraction": request.column_preview_support_probability,
                    "relative_complex_l2": request.column_preview_support_l2},
                "support_budget_meaning": "probability limits cumulative numerical intensity loss relative to the original gun mode; complex L2 bounds cumulative projection error, not intensity loss or pointwise phase error"}
               if preview_overrides else {}),
            "scope": ("column numerical and declared nominal-emission-energy source approximations; unchanged physical fields and physical gun parameters; configured FWHM not integrated; no normalization repair; independent source, axial, intensity and phase convergence still required"
                      if request.source_nominal_energy_preview else
                      "column numerical approximation only; unchanged physical fields and gun source; no normalization repair; independent source, axial, intensity and phase convergence still required")})
        self.requested_backend = requested_backend
        if stop_z_mm is not None:
            from temsim.specimen.scene import SpecimenScene
            scene = SpecimenScene.from_state(state)
            entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
            if stop_z_mm > entrance and not scene.is_vacuum:
                if not scene.support_grid.material.is_vacuum:
                    raise ValueError("Unified coherent propagation requires a separately admitted non-vacuum holder executor")
                if not np.array_equal(abs(np.asarray(scene.orientation_quaternion_wxyz)), [1., 0., 0., 0.]):
                    raise ValueError("Unified coherent material propagation currently requires an untilted axial specimen slab")
        self.runtime = get_unified_field_runtime(state) if runtime is None else runtime
        self.runtime.require_state(state)
        from temsim.physics.emission_timing import require_matching_emission_time
        self.emission_time_s = require_matching_emission_time(state, request)
        if self.runtime.metadata.get("emission_time_s", self.emission_time_s) != self.emission_time_s:
            raise ValueError("Unified wave runtime belongs to another emission epoch")
        self.snapshot, self.reductions = _static_wave_snapshot(self.runtime)
        if type(self.snapshot) is DynamicPotentialSnapshot and stop_z_mm is not None:
            from temsim.specimen.scene import SpecimenScene
            entrance = state.sample.z_mm-state.sample.thickness_nm*.5e-6
            if stop_z_mm > entrance and not SpecimenScene.from_state(state).is_vacuum:
                raise ValueError("Dynamic coherent material scattering requires a time/energy-resolved material executor; retarded vacuum wavefront propagation cannot qualify it")
        self.state, self.request = state, request
        self.identity = json_digest({"schema": "unified-wave-runtime-v2", "runtime": self.runtime.digest,
            "wave_snapshot": self.snapshot.numerical_digest, "reductions": self.reductions,
            "operator": potential_wave_operator_policy(self.operator_options),
            "canonical_domain_policies": {"column": "minimum_full_band", "material_field_substeps": "doubling"},
            "residual_compute_backend": requested_backend,
            "support_projection": self.support_projection,
            "support_projection_policy": self.support_projection_policy,
            "numerical_profile": self.numerical_profile,
            "column_segment_policy": ("captured-field-boundaries-v2" if request.column_weighted_resampling else
                                      "uniform-bounded-chunks"),
            "support_limits": {"relative_l2": self.column_grid.maximum_accumulated_support_l2,
                "probability_fraction": self.column_grid.maximum_support_loss_fraction,
                "allocation": "fixed original mode weight; epsilon*sqrt(weight), delta*weight"},
            "emission_time_s": self.emission_time_s})

    def admit(self, start_mm, stop_mm):
        if not np.isfinite((start_mm, stop_mm)).all() or stop_mm < start_mm:
            raise ValueError("Unified coherent interval must be finite and forward")
        def exact_native(value):
            for bound in (self.runtime.start_z_m, self.runtime.stop_z_m):
                if value == bound*1000.:
                    return bound
            return value*.001
        if not self.runtime.start_z_m <= exact_native(start_mm) <= exact_native(stop_mm) <= self.runtime.stop_z_m:
            raise ValueError("Unified coherent interval leaves the fixed captured column")
        return None

    def _options(self, start_mm, stop_mm, *, exact_interval_m=None):
        self.admit(start_mm, stop_mm)
        distance = ((stop_mm-start_mm)*.001 if exact_interval_m is None
                    else exact_interval_m[1]-exact_interval_m[0])
        ratio = distance/(self.runtime.stop_z_m-self.runtime.start_z_m)
        return {**self.operator_options,
                "residual_compute_backend": self.requested_backend,
                "sampling_backend": ("snapshot" if type(self.snapshot) is DynamicPotentialSnapshot or any(row.kind == "electric" for row in self.snapshot.contributions)
                                     else "native_compiled"), "exponential_error": 1e-10*ratio,
                "kinetic_remainder_phase_limit": .01*ratio,
                **({"emission_time_s": self.emission_time_s} if type(self.snapshot) is DynamicPotentialSnapshot else {})}

    def _stamp(self, output, upstream, *, contexts=None):
        record = thaw_json(output.record)
        ancestor, stops = _recording_stop_ancestry(upstream.record)
        recording = [{key: thaw_json(stop[key]) for key in ("detector", "modes", "readout_enabled")
                      if key in stop} for stop in stops]
        source_digest = (ancestor["unified_source_checkpoint_digest"]
                         if "unified_source_checkpoint_digest" in ancestor else upstream.digest)
        source_record = (ancestor["upstream"]
                         if "unified_source_checkpoint_digest" in ancestor else upstream.record)
        for name in ("material_histories", "inherited_potential_support", "support_prefix_digest",
                     "common_support_accounting", "common_kinetic_prefix", "kinetic_phase_histories"):
            if name in ancestor and name not in record:
                record[name] = thaw_json(ancestor[name])
        prior_recording = thaw_json(ancestor.get("unified_recording_stops", ()))
        if prior_recording or recording:
            record["unified_recording_stops"] = [*prior_recording, *recording]
        record.update(unified_runtime_digest=self.runtime.digest, unified_wave_identity=self.identity,
            unified_full_snapshot_digest=self.runtime.snapshot.numerical_digest,
            unified_field_digest=self.runtime.snapshot.numerical_digest,
            unified_geometry_digest=self.runtime.geometry.digest, unified_column_executed=True,
            unified_wave_snapshot=self.snapshot.to_dict(),
            snapshot_digest=self.snapshot.numerical_digest, unified_exact_reductions=self.reductions,
            upstream_digest=source_digest, upstream=source_record,
            unified_source_checkpoint_digest=source_digest,
            source_scope="original supplied checkpoint retained; the UI producer is the actual configured gun, with no source replacement",
            empirical_aberrations_applied=False, legacy_column_executed=False,
            unified_operator_scope=f"electromagnetic positive-branch scalar H{self.kinetic_order}; added electric branch and numerical budgets retained",
            potential_operator_policy=potential_wave_operator_policy(self.operator_options),
            unified_canonical_domain_policies={"column": "minimum_full_band", "material_field_substeps": "doubling"},
            unified_support_projection_policy=self.support_projection_policy,
            unified_numerical_profile=self.numerical_profile,
            scan_execution_receipt=getattr(getattr(self.state, "ac_deflector", None), "_scan_execution_receipt", None),
            emission_time_s=self.emission_time_s,
            dynamic_snapshot_digest=self.runtime.snapshot.numerical_digest if type(self.runtime.snapshot) is DynamicPotentialSnapshot else None,
            dynamic_wave_scope=("retarded adiabatic reference wavefront; cumulative column-added timing/work phase budget; "
                "gun-exit common wavefront is an explicit assumption, no longitudinal pulse or complete source time distribution")
                if type(self.snapshot) is DynamicPotentialSnapshot else None)
        record["longitudinal_phase_convention"] = (
            "original entrance-energy free longitudinal carrier remains factored; stable electric spatial and dynamic temporal "
            "corrections are already in the actual complex envelope; axial action records must not be reapplied")
        record["unified_backend"] = {"requested": self.requested_backend,
            "column_carrier": "CPU", "source_policy": "unchanged gun-stage request",
            "residual_requested": self.requested_backend,
            "material_policy": self.request.wave_grid.compute_backend,
            "selection": "stage capability; no state or global policy mutation"}
        if contexts is not None:
            contexts = {key: stamp_unified_support(value) for key, value in contexts.items()}
            residual_backends = set(ancestor.get("unified_backend", {}).get("residual_actual", ()))
            for context in contexts.values():
                for mode_record in context.get("modes", ()):
                    for stage in mode_record.get("stages", ()):
                        residual_backends.update(stage.get("residual_backends", ()))
            record["unified_backend"]["residual_actual"] = sorted(residual_backends)
            planes = {KineticPhaseLedger.from_dict(context["kinetic_phase_ledger"]).last_z_m
                      for context in contexts.values()}
            if len(planes) != 1:
                raise ValueError("Unified modes disagree about their executed observation plane")
            record["unified_executed_plane_z_m"] = planes.pop()
            for context in contexts.values():
                context["unified_wave_identity"] = self.identity
            record["unified_mode_contexts"] = contexts
            record["unified_support_summary"] = unified_support_summary(contexts)
            summary = record["unified_support_summary"]
            record["unified_numerical_profile"] = {**self.numerical_profile,
                "input_weight_per_reference": summary["input_weight_per_reference"],
                "numerical_projection_loss_per_reference": summary["used_probability_per_reference"],
                "complex_l2_projection_bound_per_reference": summary["used_l2_per_reference"],
                "normalization_applied": summary["normalization_applied"]}
            if self.request.column_preview_phase_reselection:
                from temsim.physics.potential_wave_policy import representation_estimate_history
                unique = {}
                for context in contexts.values():
                    history = representation_estimate_history(context)
                    key = context["unified_support_allocation"]["digest"]
                    estimate = history.used_l2_estimate
                    evidence = (json_digest(history.to_dict()), estimate)
                    if key in unique and unique[key] != evidence:
                        raise ValueError("Unified descendants disagree about their representation estimate history")
                    unique[key] = evidence
                estimates = {key: value[1] for key, value in unique.items()}
                rows = summary["allocations"]
                record["unified_numerical_profile"].update(
                    representation_estimates_certified=False,
                    estimated_representation_l2_per_reference=math.sqrt(math.fsum(
                        value**2 for value in estimates.values())),
                    combined_projection_and_estimated_l2_per_reference=math.sqrt(math.fsum(
                        (row["used_l2_per_reference"]+estimates[row["allocation_digest"]])**2
                        for row in rows)))
        return replace(output, record=record)

    def _entrance(self, checkpoint, mode, *, source_digest=None):
        if _source_provenance_only(checkpoint.record):
            context = {"schema": "unified-executed-source-mode-v1",
                "source_checkpoint_digest": checkpoint.digest if source_digest is None else source_digest,
                "source_mode_id": mode.mode_id,
                "source_mode_weight_per_reference": mode.weight_per_reference_electron,
                "source_provenance_location": "complete unchanged aggregate record.upstream"}
        else:
            context = _mode_context(checkpoint, mode)
        if context.get("unified_wave_identity", self.identity) != self.identity:
            raise ValueError("Unified mode history belongs to another field executor")
        can_allocate = not any(key in context for key in ("unified_wave_identity", "potential_geometry",
            "material_kinetic_history", "electric_wave_history", "inherited_potential_support", "snapshot_digest"))
        context["potential_operator_policy"] = potential_wave_operator_policy(self.operator_options, context)
        support_source = source_digest
        if support_source is None:
            support_source = context.get("source_checkpoint_digest")
        if support_source is None:
            ancestor, _ = _recording_stop_ancestry(checkpoint.record)
            support_source = ancestor.get("unified_source_checkpoint_digest")
        if support_source is None:
            support_source = checkpoint.digest
        context = prepare_unified_support(context, mode, enabled=self.support_projection,
            l2_limit=self.column_grid.maximum_accumulated_support_l2,
            probability_limit=self.column_grid.maximum_support_loss_fraction,
            source_digest=support_source,
            allow_allocate=can_allocate)
        if self.request.column_preview_phase_reselection:
            from temsim.physics.potential_wave_policy import representation_estimate_history
            context["representation_estimates"] = representation_estimate_history(
                context, initialize=can_allocate).to_dict()
        support_options = ({"support_projection_policy": self.support_projection_policy}
                           if context.get("support_ledger") is not None else {})
        context["potential_support_policy"] = potential_support_policy(support_options, context)
        if context.get("kinetic_phase_ledger") is None:
            if "unified_wave_identity" in context:
                raise ValueError("Unified continuation lost its cumulative kinetic phase budget")
            context["kinetic_phase_ledger"] = KineticPhaseLedger(.01).to_dict()
        # Near-tip traces belong to the unchanged original source checkpoint.
        # Field propagation consumes this mode, not copies of those traces.
        return replace(checkpoint, beam=BeamState((mode,), checkpoint.beam.reference_plane),
                       record=context, auxiliary_arrays={})

    def column(self, checkpoint, stop_mm, *, cancelled=lambda: False, progress_callback=None,
               store=None, use_cache=True, verify=lambda: None, checkpoint_callback=None):
        self.runtime.require_state(self.state)
        if type(checkpoint) is not TipGunCheckpoint:
            raise ValueError("Unified propagation requires an actual executed wave checkpoint")
        self.admit(checkpoint.plane_z_mm, stop_mm)
        if checkpoint.record.get("unified_wave_identity", self.identity) != self.identity:
            raise ValueError("Unified field inputs changed; restart from the actual gun source")
        if float(stop_mm) == checkpoint.plane_z_mm:
            for mode in checkpoint.beam.modes:
                context = _mode_context(checkpoint, mode)
                if context.get("unified_wave_identity", self.identity) != self.identity:
                    raise ValueError("Unified mode history belongs to another field executor")
                if "unified_wave_identity" in context:
                    context = self._entrance(checkpoint, mode).record
                    ledger = KineticPhaseLedger.from_dict(context.get("kinetic_phase_ledger"))
                    if ledger.binding is None:
                        raise ValueError("Unified observation lost its executed phase history")
                    if "material_kinetic_native_plane" in context:
                        from temsim.physics.potential_material_phase_budget import exact_material_kinetic_plane
                        plane = exact_material_kinetic_plane(context, checkpoint.plane_z_mm)
                    elif checkpoint.plane_z_mm == ledger.last_z_m*1000.:
                        plane = ledger.last_z_m
                    else:
                        raise ValueError("Unified observation differs from its exact executed plane")
                    energy = mode.energy_kev*1000.
                    if context.get("electric_wave_history") is not None:
                        from temsim.physics.electric_wave_reference import ElectricWaveHistory
                        electric = ElectricWaveHistory.from_dict(context["electric_wave_history"])
                        electric.require(self.snapshot, plane, energy)
                        energy = electric.reference.entrance_energy_ev
                    ledger.require(self.snapshot.numerical_digest, energy, self.kinetic_order, plane)
            return checkpoint
        current = checkpoint
        expected_mode_ids = (tuple(row["mode_id"] for row in checkpoint.beam.modes.rows)
            if hasattr(checkpoint.beam.modes, "rows") else
            tuple(mode.mode_id for mode in checkpoint.beam.modes))
        initial_source_digest = checkpoint.digest if _source_provenance_only(checkpoint.record) else None
        if progress_callback is not None:
            progress_callback(0, 0, "Unified coherent column; adaptive step count")
        # Bounded chunks provide cancellation/progress and do not restart the
        # distance-owned exponential budget, cumulative kinetic phase budget,
        # or physical boundary histories.
        chunk = max(self.request.column_step_mm, self.request.column_step_mm*self.request.execution.segment_steps)
        target_m = float(stop_mm)*.001
        z_m = checkpoint.plane_z_mm*.001
        ancestor, _ = _recording_stop_ancestry(checkpoint.record)
        if ancestor.get("unified_mode_contexts"):
            first_context = next(iter(ancestor["unified_mode_contexts"].values()))
            if first_context.get("kinetic_phase_ledger") is not None:
                z_m = KineticPhaseLedger.from_dict(first_context["kinetic_phase_ledger"]).last_z_m
        plan = None
        if self.request.column_weighted_resampling:
            from temsim.physics.potential_column_segments import plan_potential_column_segments
            plan = plan_potential_column_segments(self.snapshot, z_m, target_m, maximum_segment_m=chunk*.001)
            stops = iter(plan.stops_m)
        while z_m < target_m:
            count = max(1, math.ceil((target_m-z_m)/(chunk*.001)))
            upper_m = next(stops) if plan is not None else target_m if count == 1 else z_m+(target_m-z_m)/count
            upper = upper_m*1000.
            if upper_m <= z_m:
                raise ValueError("Unified wave step cannot advance the represented plane")
            options = self._options(current.plane_z_mm, upper, exact_interval_m=(z_m, upper_m))
            modes, contexts = [], {}
            for index in range(len(current.beam.modes)):
                mode_grid, memory = _column_mode_grid(self.column_grid, checkpoint, current, modes, index)
                mode = current.beam.modes[index]
                entrance = self._entrance(current, mode, source_digest=initial_source_digest)
                local_options = dict(options)
                if entrance.record.get("support_ledger") is not None:
                    local_options["support_projection_policy"] = self.support_projection_policy
                    if self.request.column_preview_segment_resampling:
                        local_options["segment_start_m"] = z_m
                    if self.request.column_preview_phase_reselection:
                        local_options["representation_estimates"] = entrance.record["representation_estimates"]
                mode_key = (None if store is None else store.key("unified-mode-segment-v1",
                    self.identity, entrance.digest, z_m, upper_m, self.request.column_step_mm,
                    mode_grid.column_identity(), self.support_projection_policy))
                output = None if store is None or not use_cache else store.get(mode_key)
                if output is None:
                    with reusable_device_scope(maximum_working_bytes=mode_grid.maximum_device_working_bytes):
                        output, = observe_potential_geometry_checkpoint(self.snapshot, self.runtime.geometry,
                            entrance, (upper_m,), maximum_step_m=self.request.column_step_mm*.001,
                            grid_numerics=mode_grid, cancelled=cancelled,
                            progress_callback=progress_callback, **local_options)
                    verify()
                    if cancelled():
                        raise InterruptedError("Unified coherent segment publication cancelled")
                    if store is not None:
                        output = store.put(mode_key, output)
                modes.extend(output.beam.modes)
                contexts[mode.mode_id] = thaw_json(output.record)
                contexts[mode.mode_id]["unified_host_memory_reservation"] = memory
                del entrance, output, mode
            output = TipGunCheckpoint(BeamState(tuple(modes), current.beam.reference_plane),
                upper, current.reference_current_a, {"schema": "unified-source-column-v1"})
            current = self._stamp(output, current, contexts=contexts)
            if plan is not None:
                current = replace(current, record={**thaw_json(current.record),
                    "unified_segment_plan": plan.to_dict()})
            if store is not None:
                verify()
                if cancelled():
                    raise InterruptedError("Unified coherent segment publication cancelled")
                segment_key = store.key("unified-column-segment-v1", self.identity,
                    checkpoint.digest, upper_m, self.request.column_step_mm, self.column_grid.column_identity(),
                    self.request.execution.segment_steps, None if plan is None else plan.digest)
                stored_segment = store.get(segment_key) if use_cache else None
                current = stored_segment if stored_segment is not None else store.put(segment_key, current)
            if checkpoint_callback is not None:
                actual_ids = (tuple(row["mode_id"] for row in current.beam.modes.rows)
                    if hasattr(current.beam.modes, "rows") else
                    tuple(mode.mode_id for mode in current.beam.modes))
                if actual_ids != expected_mode_ids:
                    raise ValueError("Unified completed segment changed the source mode mixture")
                verify()
                if cancelled():
                    raise InterruptedError("Unified completed segment registration cancelled")
                checkpoint_callback(current)
            z_m = upper_m
        return current

    def specimen(self, checkpoint, *, store, use_cache, cancelled=lambda: False, progress_callback=None):
        self.runtime.require_state(self.state)
        from temsim.physics.potential_specimen_transport import propagate_potential_specimen
        stop = self.state.sample.z_mm+self.state.sample.thickness_nm*.5e-6
        options = self._options(checkpoint.plane_z_mm, stop)
        modes, contexts, receipts = [], {}, []
        for mode in checkpoint.beam.modes:
            entrance = self._entrance(checkpoint, mode)
            result = propagate_potential_specimen(self.state, entrance, self.snapshot, store=store,
                inelastic_numerics=self.request.inelastic, grid_numerics=self.material_grid,
                maximum_step_mm=self.request.column_step_mm, operator_options=options,
                geometry=self.runtime.geometry, use_cache=use_cache,
                cancelled=cancelled, progress_callback=progress_callback)
            modes.extend(result.beam.modes)
            receipts.append(thaw_json(result.record))
            for row in result.record["modes"]:
                contexts[row["mode_id"]] = thaw_json(row["column_transport_context"])
        output = TipGunCheckpoint(BeamState(tuple(modes), checkpoint.beam.reference_plane),
            stop, checkpoint.reference_current_a,
            {"schema": "unified-actual-source-material-v1", "material_histories": receipts})
        return self._stamp(output, checkpoint, contexts=contexts)


__all__ = ("UnifiedPotentialWaveAdapter", "UnifiedPlaneMagneticGauge", "unified_checkpoint_magnetic_gauge")
