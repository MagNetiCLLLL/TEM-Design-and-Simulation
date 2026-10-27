"""Opt-in finite-material handoff to the admitted static electromagnetic operator.

The existing atomistic potential, conditional material channels and absolute
weights remain authoritative. Only distributed column transport is replaced.
Material phase / field transport / material phase is a Strang composition;
its slice and spatial errors require independent convergence. This does not
admit time-dependent material scattering, arbitrary thick supports or a physical gun source.
"""
from dataclasses import dataclass, field, replace
import math

import numpy as np

from temsim import input_io
from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot, encode_instrument
from temsim.physics.column_wave import _previous_support_error
from temsim.physics.potential_wave_carrier import _admit
from temsim.physics.potential_wave_observation import observe_potential_mode
from temsim.physics.tip_gun_wave import TipGunCheckpoint
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_grid import WaveGridNumerics
from temsim.physics.wave_execution import InelasticWaveNumerics
from temsim.physics.potential_support_lineage import (
    capture_inherited_support, validate_inherited_support, common_support_summary,
)
from temsim.physics.potential_material_phase_budget import (
    capture_material_kinetic, kinetic_context, prepare_material_kinetic,
    require_contractive_collision, material_kinetic_summary, stamp_material_kinetic_plane,
)


def _admit_material_snapshot(snapshot):
    """Validate field types without inventing a source energy for inspection."""
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
    from temsim.physics.electric_wave_reference import electric_rows
    if type(snapshot) is DynamicPotentialSnapshot:
        raise ValueError("Dynamic material histories require a separately qualified time-energy scattering operator")
    if type(snapshot) is not ElectromagneticPotentialSnapshot:
        raise ValueError("Material propagation requires its immutable potential snapshot")
    electric_rows(snapshot)
    _admit(ElectromagneticPotentialSnapshot(snapshot.scene,
        tuple(row for row in snapshot.contributions if row.kind != "electric"), snapshot.gauge))


@dataclass(frozen=True)
class PotentialMaterialTransport:
    """Captured replacement for column transport inside material slices.

    One declared complete interval owns the exponential/kinetic budgets. Each
    slice receives only its distance share, never a restarted full budget.
    Numerical support projection is deliberately not admitted in this bridge;
    existing material-bandwidth and inherited support losses remain separate.
    """
    snapshot: object
    interval_mm: tuple
    input_identity: str
    operator_options: object = field(default_factory=dict)
    geometry: object = None
    digest: str = field(init=False)

    def __post_init__(self):
        _admit_material_snapshot(self.snapshot)
        if not isinstance(self.input_identity, str) or not self.input_identity.strip():
            raise ValueError("Material transport requires a captured input identity")
        interval = np.asarray(self.interval_mm)
        if (interval.dtype.kind not in "iuf" or interval.shape != (2,)
                or not np.isfinite(interval).all() or interval[1] <= interval[0]):
            raise ValueError("Material transport interval must be finite and increasing")
        options = dict(self.operator_options)
        allowed = {"kinetic_order", "exponential_error", "kinetic_remainder_phase_limit",
            "boundary_fraction_limit", "envelope_tail_limit", "reselect_carrier",
            "scalar_splitting", "quartic_scalar_splitting", "higher_order_scalar_splitting",
            "sampling_backend", "reference_quadrature", "residual_compute_backend",
            "residual_exponential_method", "maximum_chebyshev_group"}
        if set(options)-allowed:
            raise ValueError("Unsupported material potential option; support projection and source overrides are not admitted")
        from temsim.physics.potential_wave_policy import potential_wave_operator_policy
        potential_wave_operator_policy(options)
        for key, default in (("exponential_error", 1e-10), ("kinetic_remainder_phase_limit", .01)):
            value = options.get(key, default)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
                raise ValueError("Material operator budgets must be positive finite scalars")
            options[key] = float(value)
            if not math.isfinite(options[key]) or options[key] <= 0:
                raise ValueError("Material operator budgets must be positive finite scalars")
        object.__setattr__(self, "interval_mm", tuple(map(float, interval)))
        object.__setattr__(self, "operator_options", freeze_json(options))
        object.__setattr__(self, "digest", json_digest({
            "schema": "potential-material-transport-v1", "snapshot": self.snapshot.numerical_digest,
            "angular_domain_padding": "doubling",
            "interval_mm": self.interval_mm, "inputs": self.input_identity, "options": options,
            "geometry": None if self.geometry is None else self.geometry.digest}))

    def domain_policy(self):
        """The field-step reserve is independent of the captured material grid."""
        body = {"angular_domain_padding": "doubling", "transport_identity": self.digest,
                "scope": "field substeps within the unchanged captured material ROI and phonon realization"}
        return {"schema": "potential-material-domain-policy-v1", **body, "digest": json_digest(body)}

    def options_for(self, start_mm, stop_mm):
        start, stop = self.interval_mm
        if not start <= start_mm < stop_mm <= stop:
            raise ValueError("Material transport leaves its captured interval")
        options = thaw_json(self.operator_options)
        ratio = (stop_mm-start_mm)/(stop-start)
        for key in ("exponential_error", "kinetic_remainder_phase_limit"):
            options[key] *= ratio
        return options

    @staticmethod
    def context(record, factor=1.):
        """Retain execution identity across material half-steps and restarts."""
        executed = (record.get("schema") == "potential-material-slice-v1"
                    or "column_transport_identity" in record
                    or "material_execution_identity" in record or "material_domain_policy" in record)
        if executed and ("material_domain_policy" not in record or "material_execution_identity" not in record):
            raise ValueError("Historical material domain policy is unknown; restart from the material entrance")
        if "material_domain_policy" in record:
            receipt = thaw_json(record["material_domain_policy"])
            if not isinstance(receipt, dict):
                raise ValueError("Invalid material domain policy receipt")
            body = {key: value for key, value in receipt.items() if key not in ("schema", "digest")}
            if (set(receipt) != {"schema", "angular_domain_padding", "transport_identity", "scope", "digest"}
                    or receipt["schema"] != "potential-material-domain-policy-v1"
                    or receipt["angular_domain_padding"] != "doubling"
                    or type(receipt["transport_identity"]) is not str or not receipt["transport_identity"]
                    or receipt["scope"] != "field substeps within the unchanged captured material ROI and phonon realization"
                    or receipt["digest"] != json_digest(body)
                    or type(record["material_execution_identity"]) is not str
                    or record["material_execution_identity"] != receipt["transport_identity"]):
                raise ValueError("Invalid material domain policy receipt")
        validate_inherited_support(record)
        result = {key: thaw_json(record[key]) for key in (
            "snapshot_digest", "potential_geometry", "inherited_potential_support", "support_prefix_digest") if key in record}
        if "potential_geometry" in result and factor != 1.:
            geometry = result["potential_geometry"]
            for key in ("physical_loss_per_reference", "entry_weight_per_reference", "output_weight_per_reference"):
                geometry[key] *= factor
            for row in geometry["physical_losses"]:
                for key in ("lost_probability_per_reference", "input_weight_per_reference", "output_weight_per_reference"):
                    row[key] *= factor
        result.update(kinetic_context(record))
        return result

    @staticmethod
    def validate_collision(record, inside, distribution):
        require_contractive_collision(record, inside, distribution)

    def __call__(self, state, checkpoint, stop_z_mm, *, maximum_step_mm,
                 grid_numerics, cancelled, tip_time_s=None, retained_bytes=0,
                 progress_callback=None):
        options = self.options_for(checkpoint.plane_z_mm, stop_z_mm)
        self.context(checkpoint.record)  # Validate lineage before any field work.
        domain_receipt = self.domain_policy()
        if "material_domain_policy" in checkpoint.record:
            if thaw_json(checkpoint.record["material_domain_policy"]) != domain_receipt:
                raise ValueError("Material domain policy changed or is invalid; restart from the material entrance")
        elif checkpoint.record.get("schema") == "potential-material-slice-v1":
            raise ValueError("Historical material domain policy is unknown; restart from the material entrance")
        from temsim.physics.potential_wave_policy import potential_wave_operator_policy
        operator_policy = potential_wave_operator_policy(options, checkpoint.record)
        from temsim.physics.potential_wave_policy import potential_support_policy
        potential_support_policy(options, checkpoint.record)
        from temsim.physics.electric_wave_reference import ElectricWaveHistory, electric_rows
        has_electric = bool(electric_rows(self.snapshot))
        if has_electric and (checkpoint.record.get("kinetic_phase_ledger") is None
                             or checkpoint.record.get("electric_wave_history") is None):
            raise ValueError("Electric material propagation requires an executed energy and kinetic budget history")
        from temsim.physics.wave_device import normalise_backend
        requested_backend = normalise_backend(grid_numerics.compute_backend)
        carrier_grid = replace(grid_numerics, compute_backend="CPU", angular_domain_padding="doubling")
        if checkpoint.record.get("kinetic_phase_ledger") is not None:
            if len(checkpoint.beam.modes) != 1:
                raise ValueError("A cumulative kinetic ledger requires one material history at a time")
            captured = capture_material_kinetic(checkpoint, self.snapshot.numerical_digest, snapshot=self.snapshot)
            checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record), **captured})
            prepared, active = prepare_material_kinetic(checkpoint.record, checkpoint.beam.modes[0],
                checkpoint.plane_z_mm, self.snapshot.numerical_digest, options.get("kinetic_order", 4), snapshot=self.snapshot)
            checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record), **prepared})
            options["kinetic_phase_ledger"] = active
        if has_electric:
            options["electric_wave_history"] = ElectricWaveHistory.from_dict(checkpoint.record["electric_wave_history"])
        outputs, rows, ledgers = [], [], {}
        context = self.context(checkpoint.record)
        context["potential_operator_policy"] = thaw_json(operator_policy)
        geometry_outputs, geometry_evidence = {}, {}
        if self.geometry is not None:
            from temsim.physics.potential_geometry import observe_potential_geometry_checkpoint
            # Geometry owns the complete beam's shared history. Executing each
            # mode with the same aggregate prefix would duplicate prior losses
            # and let the final mode overwrite the current aggregate totals.
            prior_bytes = 0
            for mode in checkpoint.beam.modes:
                grid_numerics.check_working_memory(mode.plane.amplitude.shape,
                    working_bytes=640*mode.plane.amplitude.size,
                    retained_bytes=retained_bytes+prior_bytes)
                prior_bytes += mode.plane.amplitude.nbytes
            observed, = observe_potential_geometry_checkpoint(self.snapshot, self.geometry,
                checkpoint, (stop_z_mm*.001,), maximum_step_m=maximum_step_mm*.001,
                grid_numerics=carrier_grid, cancelled=cancelled, **options)
            context = self.context(observed.record)
            geometry = context["potential_geometry"]
            for output, row in zip(observed.beam.modes, observed.record["modes"], strict=True):
                geometry_outputs[output.mode_id] = output
                losses = [loss for loss in geometry["physical_losses"] if loss["mode_id"] == output.mode_id]
                geometry_evidence[output.mode_id] = {
                    **thaw_json(row), "snapshot_digest": self.snapshot.numerical_digest,
                    "geometry_mode": {"geometry_digest": geometry["geometry_digest"],
                        "mode_id": output.mode_id, "physical_losses": losses,
                        "physical_loss_per_reference": math.fsum(loss["lost_probability_per_reference"] for loss in losses),
                        "output_weight_per_reference": output.weight_per_reference_electron,
                        "accounting_scope": "this mode only; the complete shared history is stored once on the checkpoint"}}
        for mode in checkpoint.beam.modes:
            grid_numerics.check_working_memory(mode.plane.amplitude.shape,
                working_bytes=640*mode.plane.amplitude.size,
                retained_bytes=retained_bytes+sum(row.plane.amplitude.nbytes for row in outputs))
            if self.geometry is None:
                observation, = observe_potential_mode(self.snapshot, mode,
                    checkpoint.plane_z_mm*.001, (stop_z_mm*.001,),
                    maximum_step_m=maximum_step_mm*.001, grid_numerics=carrier_grid,
                    cancelled=cancelled, **options)
                output, evidence = observation.mode, thaw_json(observation.metadata)
                if evidence.get("kinetic_phase_ledger") is not None:
                    context["kinetic_phase_ledger"] = evidence["kinetic_phase_ledger"]
                if evidence.get("electric_wave_history") is not None:
                    context["electric_wave_history"] = evidence["electric_wave_history"]
            else:
                output, evidence = geometry_outputs[mode.mode_id], geometry_evidence[mode.mode_id]
            inherited = _previous_support_error(checkpoint.record, mode.mode_id)
            if inherited is None:
                inherited = {"reference_mode_weight": mode.weight_per_reference_electron,
                    "accumulated_relative_l2_bound": 0., "probability_loss_per_tip_electron": 0.}
            ledgers[mode.mode_id] = inherited
            rows.append({"mode_id": mode.mode_id, "energy_kev": output.energy_kev,
                "input_weight": mode.weight_per_reference_electron,
                "output_weight": output.weight_per_reference_electron,
                "losses": [], "support_approximation": inherited,
                "potential_execution": evidence})
            outputs.append(output)
        context = stamp_material_kinetic_plane(context, float(stop_z_mm))
        return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane),
            float(stop_z_mm), checkpoint.reference_current_a,
            {"schema": "potential-material-slice-v1", "upstream_digest": checkpoint.digest,
             "upstream": checkpoint.record, "snapshot_digest": self.snapshot.numerical_digest,
             **context,
             "transport_identity": self.digest, "modes": rows, "support_error_by_mode": ledgers,
             "material_domain_policy": domain_receipt, "material_execution_identity": self.digest,
             "backend_policy": {"requested": requested_backend, "field_carrier": "CPU",
                                "material_policy": grid_numerics.compute_backend},
             "empirical_aberrations_applied": False, "legacy_column_executed": False,
             "normalization_applied": False, "production_admission": False})


@input_io.using_state_inputs
def propagate_potential_specimen(state, checkpoint, snapshot, *, store,
        inelastic_numerics=InelasticWaveNumerics(), grid_numerics=WaveGridNumerics(),
        maximum_step_mm=.001, operator_options=None, cancelled=lambda: False,
        progress_callback=None, use_cache=True, geometry=None):
    """Execute existing finite multislice/material histories with new fields.

    Static electric and magnetic fields may overlap the finite material and are executed
    between its real half-phase actions at each history's actual energy. The
    current material model is an axial slab with no non-vacuum holder support.
    Dynamic and unknown fields fail before potential preparation. Static
    acceleration retains its source energy law and collision-budget lineage.
    """
    from temsim.specimen.scene import SpecimenScene
    from temsim.physics.specimen_wave_transport import _propagate_specimen
    from temsim.physics.inelastic_wave import _propagate_inelastic_specimen
    if type(checkpoint) is not TipGunCheckpoint:
        raise ValueError("Material handoff requires an executed wave checkpoint")
    _admit_material_snapshot(snapshot)
    from temsim.physics.electric_wave_reference import electric_rows
    if electric_rows(snapshot) and (checkpoint.record.get("electric_wave_history") is None
                                    or checkpoint.record.get("kinetic_phase_ledger") is None):
        raise ValueError("Electric material propagation requires an executed energy and kinetic budget history")
    kinetic_prefix = capture_material_kinetic(checkpoint, snapshot.numerical_digest, snapshot=snapshot)
    if kinetic_prefix:
        checkpoint = replace(checkpoint, record={**thaw_json(checkpoint.record), **kinetic_prefix})
    from temsim.physics.wave_device import normalise_backend
    normalise_backend(grid_numerics.compute_backend)
    if (isinstance(maximum_step_mm, (bool, np.bool_)) or not np.isscalar(maximum_step_mm)
            or not np.isreal(maximum_step_mm) or not math.isfinite(maximum_step_mm) or maximum_step_mm <= 0):
        raise ValueError("Material propagation step must be a positive finite scalar")
    if checkpoint.record.get("potential_geometry") is not None and len(checkpoint.beam.modes) != 1:
        raise ValueError("Inherited geometry histories must be supplied one incident mode at a time")
    scene = SpecimenScene.from_state(state)
    if scene.is_vacuum or scene.thickness_nm <= 0:
        raise ValueError("Finite material handoff requires an inserted nonzero-thickness specimen")
    if not scene.support_grid.material.is_vacuum:
        raise ValueError("Non-vacuum holder support requires a separately admitted material boundary executor")
    quaternion = np.asarray(scene.orientation_quaternion_wxyz)
    if not np.array_equal(abs(quaternion), [1., 0., 0., 0.]):
        raise ValueError("Potential material handoff currently requires an untilted axial specimen slab")
    start = state.sample.z_mm-state.sample.thickness_nm*.5e-6
    stop = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    if kinetic_prefix and checkpoint.plane_z_mm == (start*.001)*1000.:
        record = stamp_material_kinetic_plane(thaw_json(checkpoint.record), start)
        checkpoint = replace(checkpoint, plane_z_mm=start, record=record)
    if checkpoint.plane_z_mm != start:
        raise ValueError("Potential material handoff needs the exact physical entrance face")
    if geometry is not None:
        from temsim.physics.potential_geometry import PotentialGeometrySnapshot
        if (type(geometry) is not PotentialGeometrySnapshot
                or not geometry.start_z_m <= start*.001 < stop*.001 <= geometry.stop_z_m):
            raise ValueError("Captured geometry must cover the complete finite specimen")
    previous = checkpoint.record.get("snapshot_digest")
    if previous is not None and previous != snapshot.numerical_digest:
        raise ValueError("Potential snapshot changed at the specimen entrance")
    prefix = capture_inherited_support(checkpoint, snapshot.numerical_digest)
    if prefix is not None:
        parent = checkpoint
        metadata = thaw_json(parent.record)
        metadata.pop("support_ledger", None)
        metadata.update(inherited_potential_support=prefix, support_prefix_digest=prefix["digest"],
            upstream_digest=parent.digest, upstream=parent.record)
        checkpoint = replace(parent, record=metadata)
        validate_inherited_support(checkpoint.record, {} if operator_options is None else operator_options)
    inelastic_numerics.validate()
    grid_numerics.validate()
    inputs = capture_instrument_snapshot(state)
    initial_graph = json_digest(encode_instrument(state))
    def verify():
        if cancelled():
            raise InterruptedError("Potential material propagation cancelled")
        if json_digest(encode_instrument(state)) != initial_graph:
            raise ValueError("Specimen inputs changed during potential propagation")
        inputs._verify_external_inputs(state)
    transport = PotentialMaterialTransport(snapshot, (start, stop), inputs.digest,
        {} if operator_options is None else operator_options, geometry)
    # Query the actual incoming grid and both faces before expensive atomistic
    # preparation. Every subsequently refined slice/grid is checked again by
    # the field executor; this is not a corner-only whole-path admission.
    for mode in checkpoint.beam.modes:
        xy = mode.plane.coordinates_m().reshape(2, -1).T
        for z in (start, stop):
            snapshot.sample(np.c_[xy, np.full(len(xy), z*.001)])
    options = dict(maximum_step_mm=maximum_step_mm, grid_numerics=grid_numerics,
        tip_time_s=None, cancelled=cancelled, progress_callback=progress_callback,
        column_propagator=transport)
    if inelastic_numerics.method == "trajectories":
        result = _propagate_inelastic_specimen(state, checkpoint, numerics=inelastic_numerics,
            store=store, verify=verify, use_cache=use_cache, **options)
    else:
        result = _propagate_specimen(state, checkpoint, **options)
    verify()
    record = thaw_json(result.record)
    if kinetic_prefix:
        by_id = {mode.mode_id: mode for mode in result.beam.modes}
        summaries = {}
        for row in record["modes"]:
            context = row["column_transport_context"]
            updated, _ = prepare_material_kinetic(context, by_id[row["mode_id"]], result.plane_z_mm,
                snapshot.numerical_digest, transport.operator_options.get("kinetic_order", 4), snapshot=snapshot)
            context.update(updated)
            summaries[row["mode_id"]] = material_kinetic_summary(context)
        record.update(common_kinetic_prefix=kinetic_prefix["material_kinetic_history"]["prefix"],
                      kinetic_phase_histories=summaries)
    record.update(snapshot_digest=snapshot.numerical_digest, input_identity=inputs.digest,
        material_domain_policy=transport.domain_policy(), material_execution_identity=transport.digest,
        potential_material_scope="finite axial atomic slab; static electromagnetic overlap; independent conditional histories",
        empirical_aberrations_applied=False, legacy_column_executed=False,
        production_admission=False,
        geometry_scope="caller-declared no column boundaries" if geometry is None else "captured physical geometry throughout material",
        convergence_scope="material slice, potential grid, wave lattice and field split require independent convergence")
    if prefix is not None:
        record.update(inherited_potential_support=prefix, support_prefix_digest=prefix["digest"])
        record["common_support_accounting"] = common_support_summary(record)
    return replace(result, record=record)


@dataclass(frozen=True)
class PotentialMaterialReadout:
    specimen: TipGunCheckpoint
    detector_checkpoint: TipGunCheckpoint
    readout: object
    metadata: object

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@input_io.using_state_inputs
def propagate_potential_specimen_to_detector(state, checkpoint, snapshot, geometry,
        detector, *, store, readout_options=None, maximum_step_mm=.001,
        operator_options=None, grid_numerics=WaveGridNumerics(),
        inelastic_numerics=InelasticWaveNumerics(), cancelled=lambda: False,
        progress_callback=None, use_cache=True, maximum_readout_bytes=512*1024**2):
    """Material histories, physical downstream boundaries and absolute readout.

    A geometry snapshot covering the entire requested interval is mandatory.
    No fitted/legacy image pupil, chromatic screen or reconstruction from rays
    is called. Each conditional energy remains a separate mode at detection.
    This is an explicit controlled-entrance workflow, not default admission.
    """
    from temsim.detector.wave_readout import WaveReadoutOptions, read_wave_detector
    from temsim.physics.potential_geometry import PotentialGeometrySnapshot, observe_potential_geometry_checkpoint
    if type(geometry) is not PotentialGeometrySnapshot:
        raise ValueError("Material-to-detector execution requires captured physical geometry")
    if type(maximum_readout_bytes) is not int or maximum_readout_bytes <= 0:
        raise ValueError("Detector readout memory budget must be a positive integer")
    stop = float(detector.z_mm)
    specimen_exit = state.sample.z_mm+state.sample.thickness_nm*.5e-6
    if not math.isfinite(stop) or stop <= specimen_exit or not bool(getattr(detector, "inserted", False)):
        raise ValueError("An inserted detector must be strictly downstream of the finite specimen")
    if not geometry.start_z_m <= checkpoint.plane_z_mm*.001 < stop*.001 <= geometry.stop_z_m:
        raise ValueError("Captured geometry must cover the full specimen-to-detector interval")
    options = {} if operator_options is None else dict(operator_options)
    # Allocate the same total operator budget across material and downstream,
    # independently for every environmental history, without resetting it.
    allocation = PotentialMaterialTransport(snapshot, (checkpoint.plane_z_mm, stop),
        "complete-controlled-material-readout", options, geometry)
    material_options = allocation.options_for(checkpoint.plane_z_mm, specimen_exit)
    specimen = propagate_potential_specimen(state, checkpoint, snapshot, store=store,
        maximum_step_mm=maximum_step_mm, operator_options=material_options,
        grid_numerics=grid_numerics, inelastic_numerics=inelastic_numerics,
        geometry=geometry, cancelled=cancelled, progress_callback=progress_callback, use_cache=use_cache)
    downstream_options = allocation.options_for(specimen.plane_z_mm, stop)
    outputs, records = [], []
    material_rows = {row["mode_id"]: row for row in specimen.record["modes"]}
    for mode in specimen.beam.modes:
        if cancelled():
            raise InterruptedError("Potential detector propagation cancelled")
        context = material_rows[mode.mode_id].get("column_transport_context", {})
        entrance = TipGunCheckpoint(BeamState((mode,), specimen.beam.reference_plane),
            specimen.plane_z_mm, specimen.reference_current_a,
            {"upstream_digest": specimen.digest, "snapshot_digest": snapshot.numerical_digest, **context})
        observed, = observe_potential_geometry_checkpoint(snapshot, geometry, entrance,
            (stop*.001,), maximum_step_m=maximum_step_mm*.001, grid_numerics=grid_numerics,
            cancelled=cancelled, **downstream_options)
        outputs.append(observed.beam.modes[0])
        records.append({"mode_id": mode.mode_id, "energy_kev": observed.beam.modes[0].energy_kev,
            "execution": thaw_json(observed.record)})
    terminal = TipGunCheckpoint(BeamState(tuple(outputs), specimen.beam.reference_plane),
        stop, specimen.reference_current_a,
        {"schema": "potential-material-detector-checkpoint-v1", "upstream_digest": specimen.digest,
         "upstream": specimen.record, "snapshot_digest": snapshot.numerical_digest,
         "geometry_digest": geometry.digest, "modes": records, "production_admission": False})
    if specimen.record.get("common_kinetic_prefix") is not None:
        terminal = replace(terminal, record={**thaw_json(terminal.record),
            "common_kinetic_prefix": specimen.record["common_kinetic_prefix"],
            "kinetic_phase_histories": {row["mode_id"]: material_kinetic_summary(row["execution"]) for row in records}})
    prefix = validate_inherited_support(specimen.record)
    if prefix is not None:
        terminal = replace(terminal, record={**thaw_json(terminal.record),
            "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"],
            "common_support_accounting": common_support_summary(specimen.record)})
    readout = read_wave_detector(terminal, detector,
        WaveReadoutOptions() if readout_options is None else readout_options,
        maximum_bytes=maximum_readout_bytes)
    if prefix is not None:
        readout = replace(readout, record={**thaw_json(readout.record),
            "inherited_potential_support": prefix, "support_prefix_digest": prefix["digest"],
            "common_support_accounting": common_support_summary(terminal.record)})
    if terminal.record.get("common_kinetic_prefix") is not None:
        readout = replace(readout, record={**thaw_json(readout.record),
            "common_kinetic_prefix": terminal.record["common_kinetic_prefix"],
            "kinetic_phase_histories": terminal.record["kinetic_phase_histories"]})
    return PotentialMaterialReadout(specimen, terminal, readout, {
        "schema": "potential-material-detector-v1", "snapshot_digest": snapshot.numerical_digest,
        "geometry_digest": geometry.digest, "source_scope": "executed caller-supplied entrance",
        "material_model": "existing atomistic multislice and declared material channels",
        "magnetic_overlap": "same potential at each slice and actual conditional energy",
        "electric_overlap": "stationary total energy between collisions; actual collision energy rebases only the remaining budget",
        "mixed_phase": "UNDEFINED; phases belong to individual environmental histories",
        "empirical_aberrations_applied": False, "legacy_pupil_applied": False,
        "normalization_applied": False, "production_admission": False})
