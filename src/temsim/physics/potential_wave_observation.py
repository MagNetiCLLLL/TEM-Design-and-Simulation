"""Observation and checkpoint bridge for an executed potential-wave input.

The caller supplies the actual entrance mode and immutable potential snapshot;
this module never constructs a replacement source. Static electric acceleration
uses a preserved energy reference. Admitted dynamic coils use a bounded,
retarded stationary reference-wavefront approximation, not a full time-domain
wave packet. Separate wrappers own thick boundaries and specimen scattering.
Unlisted legacy aberration actions are not inferred.
"""
from dataclasses import asdict, dataclass, replace
from collections.abc import Mapping
import math

import numpy as np
from scipy.constants import c, e, m_e

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.posed_column_fields import FrozenPosedMultipole
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot
from temsim.physics.potential_wave_carrier import propagate_carrier_potential
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.potential_phase_budget import KineticPhaseLedger
from temsim.physics.potential_wave_policy import potential_wave_operator_policy, potential_support_policy
from temsim.physics.wave_flux import WaveMode


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isreal(value):
        raise ValueError(f"{name} must be a finite real scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite"+(" and positive" if positive else ""))
    return value


@dataclass(frozen=True, slots=True)
class PotentialAperturePlane:
    """An explicitly declared thin circular absorbing geometry boundary."""
    event_id: str
    z_m: float
    radius_m: float
    center_xy_m: tuple = (0., 0.)

    def __post_init__(self):
        if type(self.event_id) is not str or not self.event_id or self.event_id.strip() != self.event_id:
            raise ValueError("Aperture event_id must be a nonempty trimmed identifier")
        center = np.asarray(self.center_xy_m)
        if center.dtype.kind not in "iuf" or center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("Aperture centre must be finite XY metres")
        object.__setattr__(self, "z_m", _number(self.z_m, "Aperture Z"))
        object.__setattr__(self, "radius_m", _number(self.radius_m, "Aperture radius", positive=True))
        object.__setattr__(self, "center_xy_m", tuple(map(float, center)))

    def mask(self, wave):
        delta = wave.coordinates_m()-np.asarray(self.center_xy_m)[:, None, None]
        return np.hypot(delta[0], delta[1]) <= self.radius_m


@dataclass(frozen=True, slots=True)
class PotentialPlaneObservation:
    plane: PlaneWave
    z_m: float
    metadata: object

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True, slots=True)
class PotentialModeObservation:
    mode: WaveMode
    z_m: float
    metadata: object

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


def potential_observation_nodes(snapshot, start_z_m, observation_z_m, *, maximum_step_m, apertures=()):
    """Insert exact observations, aperture planes and aligned compact faces."""
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot, DynamicDipolePotential
    if type(snapshot) not in (ElectromagneticPotentialSnapshot, DynamicPotentialSnapshot):
        raise ValueError("Observation requires an immutable potential snapshot")
    start = _number(start_z_m, "Entrance Z")
    step = _number(maximum_step_m, "Maximum axial step", positive=True)
    raw = np.asarray(observation_z_m)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Observation planes must be a nonempty finite sequence")
    observations = np.asarray(raw, dtype=float)
    if np.any(np.diff(observations) <= 0) or observations[0] < start:
        raise ValueError("Observation planes must increase from the entrance")
    if any(type(event) is not PotentialAperturePlane for event in apertures):
        raise ValueError("Only declared immutable aperture planes are supported")
    if len({event.event_id for event in apertures}) != len(apertures):
        raise ValueError("Aperture event identities must be unique")
    stop = observations[-1]
    if any(not start <= event.z_m <= stop for event in apertures):
        raise ValueError("Selected aperture planes must lie in the executed interval")
    cuts = [start, *observations, *(event.z_m for event in apertures)]
    for row in snapshot.contributions:
        provider = row.potential
        if type(provider) is DynamicDipolePotential:
            provider = provider.base
        if row.kind == "electric":
            from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
            if type(provider) is not ElectricPotentialSnapshot:
                raise ValueError("Unsupported electrostatic observation field")
            cuts.extend(float(z) for z in provider.arrays["z"] if start < z < stop)
            continue
        if type(provider) is not AnalyticPotentialDomain or type(provider.base.field) is not FrozenPosedMultipole:
            continue
        native = provider.base.field
        rotation = native.registration.rotation_array
        if np.any(rotation[:2, 2] != 0.):
            continue  # The carrier operator rejects a tilted face intersecting its grid.
        for value in native.native_support_mm:
            z = native.registration.origin_array_m[2]+rotation[2, 2]*value*.001
            if start < z < stop:
                cuts.append(z)
    cuts = np.unique(cuts)
    nodes = [start]
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        count = max(1, math.ceil((upper-lower)/step))
        if count > 100000 or len(nodes)+count > 100000:
            raise ValueError("Observation plan exceeds its 100000-node budget")
        nodes.extend(np.linspace(lower, upper, count+1)[1:])
    return np.frombuffer(np.asarray(nodes, dtype=float).tobytes(), dtype=float)


def observe_potential_plane(snapshot, wave, start_z_m, observation_z_m, *, energy_ev,
        maximum_step_m=.0001, apertures=(), already_applied_event_ids=(), cancelled=lambda: False,
        support_ledger=None, support_mode_weight=1., kinetic_phase_ledger=None,
        representation_estimates=None,
        electric_wave_history=None, emission_time_s=None, initial_flight_time_s=None, **numerics):
    """Propagate a controlled entrance wave to requested planes after local masks.

    Raw phase carriers and absolute cell probability are retained. An aperture
    is applied once; a resumed caller must pass previously applied identities.
    The exponential budget is distributed by the actual operator-step count;
    the kinetic expansion budget is distributed by distance. Distinct nearby
    observation and event planes are retained, never rounded into one plane.
    """
    if type(wave) is not PlaneWave:
        raise ValueError("Observation requires a PlaneWave entrance")
    operator_policy = potential_wave_operator_policy(numerics)
    support_policy = potential_support_policy({**numerics, "support_ledger": support_ledger})
    energy_ev = _number(energy_ev, "Kinetic energy", positive=True)
    from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory, electric_rows
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    from temsim.physics.retarded_wave import RetardedPotentialSlice, initial_dynamic_state
    dynamic = type(snapshot) is DynamicPotentialSnapshot
    has_electric = bool(electric_rows(snapshot))
    if electric_wave_history is not None and type(electric_wave_history) is not ElectricWaveHistory:
        raise ValueError("Electric wave continuation needs an immutable ElectricWaveHistory")
    if (has_electric or dynamic) and electric_wave_history is None:
        if dynamic and (emission_time_s is None or initial_flight_time_s is None):
            raise ValueError("Dynamic coherent propagation requires an explicit emission epoch and executed axial flight reference")
        state = initial_dynamic_state(emission_time_s, initial_flight_time_s) if dynamic else None
        electric_wave_history = ElectricWaveHistory(ElectrostaticWaveReference.capture(snapshot, start_z_m, energy_ev), dynamic_state=state)
    if electric_wave_history is not None:
        electric_wave_history.require(snapshot, start_z_m, energy_ev)
        if dynamic:
            state = electric_wave_history.dynamic_state
            if state is None or (emission_time_s is not None and emission_time_s != state["emission_time_s"]):
                raise ValueError("Dynamic coherent history lost or changed its emission clock")
            if initial_flight_time_s is not None and initial_flight_time_s != state["flight_time_s"]:
                raise ValueError("Dynamic coherent axial reference differs from its executed clock")
    electric_reference = None if electric_wave_history is None else electric_wave_history.reference
    if support_ledger is not None and type(support_ledger) is not NumericalSupportLedger:
        raise ValueError("support_ledger must be an immutable NumericalSupportLedger or None")
    from temsim.physics.potential_representation_estimates import (
        RepresentationEstimateLedger, require_combined_support_allowance,
    )
    if isinstance(representation_estimates, Mapping):
        representation_estimates = RepresentationEstimateLedger.from_dict(representation_estimates)
    if numerics.get("support_projection_policy") == "preview_phase_segment_and_retry":
        if support_ledger is None:
            raise ValueError("Preview phase observation requires its original active support ledger")
        if representation_estimates is None:
            representation_estimates = RepresentationEstimateLedger.for_support(support_ledger)
        require_combined_support_allowance(support_ledger, representation_estimates)
    elif representation_estimates is not None:
        raise ValueError("Active representation estimates require the explicit preview phase policy")
    if kinetic_phase_ledger is not None:
        if type(kinetic_phase_ledger) is not KineticPhaseLedger:
            raise ValueError("kinetic_phase_ledger must be an immutable KineticPhaseLedger or None")
        kinetic_phase_ledger = kinetic_phase_ledger.require(snapshot.numerical_digest,
            energy_ev if electric_reference is None else electric_reference.entrance_energy_ev,
            numerics.get("kinetic_order", 4), float(start_z_m))
    initial_support_count = 0 if support_ledger is None else len(support_ledger.events)
    apertures = tuple(apertures)
    nodes = potential_observation_nodes(snapshot, start_z_m, observation_z_m,
        maximum_step_m=maximum_step_m, apertures=apertures)
    observations = tuple(map(float, observation_z_m))
    previous = tuple(already_applied_event_ids)
    applied = set(previous)
    if any(type(key) is not str or not key for key in applied):
        raise ValueError("Previously applied event identities must be nonempty strings")
    if len(applied) != len(previous):
        raise ValueError("Previously applied event identities must not repeat")
    initial, current, z = wave.probability, wave, float(start_z_m)
    cuts = np.unique([*(nodes if dynamic else observations), *(event.z_m for event in apertures)])
    stages, losses, outputs, dynamic_integrals = [], [], [], []
    total_distance = observations[-1]-z
    exponential_budget = _number(numerics.pop("exponential_error", 1e-10), "Exponential error budget", positive=True)
    kinetic_budget = _number(numerics.pop("kinetic_remainder_phase_limit", .01), "Kinetic phase budget", positive=True)
    for stop in cuts:
        if cancelled():
            raise InterruptedError("Potential observation cancelled")
        if stop > z and current.probability > 0:
            selected = nodes[(nodes >= z) & (nodes <= stop)]
            ratio = (stop-z)/total_distance
            executed_snapshot = (RetardedPotentialSlice(snapshot, electric_wave_history, z, float(stop)) if dynamic else snapshot)
            executed_reference = executed_snapshot if dynamic else electric_reference
            current, record = propagate_carrier_potential(executed_snapshot, current, selected, energy_ev=energy_ev,
                exponential_error=exponential_budget*(len(selected)-1)/(len(nodes)-1),
                kinetic_remainder_phase_limit=kinetic_budget*ratio, cancelled=cancelled,
                support_ledger=support_ledger, support_mode_weight=support_mode_weight,
                representation_estimates=representation_estimates,
                kinetic_phase_ledger=kinetic_phase_ledger, electric_reference=executed_reference,
                electric_forward_error_limit=(1e-6 if electric_wave_history is None else
                    (electric_wave_history.maximum_forward_error-electric_wave_history.used_forward_error)
                    /(1+electric_wave_history.used_forward_error)), **numerics)
            stages.append(record)
            if electric_wave_history is not None:
                state = executed_snapshot.advance_adiabatic_state(record["retarded_steps"]) if dynamic else None
                axial = (executed_snapshot.axial_integrals() if dynamic else
                         electric_reference.axial_integrals(snapshot, z, float(stop)))
                # The ordinary phase-plate/readout consumes PlaneWave directly.
                # Put the STABLE added longitudinal phase into that actual
                # complex wave once, retaining only the entrance-energy free
                # carrier outside it. AxialWaveReference remains an action
                # record and must not be applied a second time.
                added_phase = axial["electric_phase_correction_rad"]+axial.get("dynamic_temporal_phase_rad", 0.)
                current = replace(current, amplitude=current.amplitude*np.exp(1j*added_phase))
                record["electric_longitudinal_phase_applied_rad"] = axial["electric_phase_correction_rad"]
                if dynamic:
                    dynamic_integrals.append(axial)
                    record["dynamic_temporal_action_j_s"] = axial["dynamic_temporal_action_j_s"]
                    record["dynamic_temporal_phase_rad"] = axial["dynamic_temporal_phase_rad"]
                electric_wave_history = electric_wave_history.append(z, float(stop), record["electric_forward_error_bound"], dynamic_state=state)
            if record["support_ledger"] is not None:
                support_ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
            if record.get("representation_estimates") is not None:
                representation_estimates = RepresentationEstimateLedger.from_dict(record["representation_estimates"])
                require_combined_support_allowance(support_ledger, representation_estimates)
            if record["kinetic_phase_ledger"] is not None:
                kinetic_phase_ledger = KineticPhaseLedger.from_dict(record["kinetic_phase_ledger"])
        elif stop > z:
            # A fully absorbed mode advances its observation plane without
            # spending phase budget; its history cannot restart at the old Z.
            if kinetic_phase_ledger is not None:
                kinetic_phase_ledger = kinetic_phase_ledger.append(z, float(stop), (0.,))
            if electric_wave_history is not None:
                if dynamic:
                    clock = RetardedPotentialSlice(snapshot, electric_wave_history, z, float(stop))
                    dynamic_integrals.append(clock.axial_integrals())
                    state = clock.advance_adiabatic_state(())
                else:
                    state = None
                electric_wave_history = electric_wave_history.append(z, float(stop), 0., dynamic_state=state)
        z = float(stop)
        for event in apertures:
            if event.z_m == z and event.event_id not in applied:
                before = current.probability
                mask = event.mask(current)
                lost = float(np.sum(abs(current.amplitude[~mask])**2))
                current = replace(current, amplitude=np.where(mask, current.amplitude, 0.))
                losses.append({**asdict(event), "input_probability": before,
                    "output_probability": current.probability, "lost_probability": lost})
                applied.add(event.event_id)
        if z in observations:
            distance = z-float(start_z_m)
            kinetic = energy_ev*e
            momentum = math.sqrt(kinetic*(kinetic+2*m_e*c*c))/c
            velocity = momentum*c*c/(kinetic+m_e*c*c)
            electric_integrals = (None if electric_reference is None else
                electric_reference.axial_integrals(snapshot, float(start_z_m), z))
            if dynamic:
                keys = ("longitudinal_action_j_s", "baseline_longitudinal_action_j_s", "electric_action_correction_j_s",
                        "electric_phase_correction_rad", "axial_flight_time_s", "dynamic_temporal_action_j_s", "dynamic_temporal_phase_rad")
                electric_integrals = {key: math.fsum(row[key] for row in dynamic_integrals) for key in keys}
                electric_integrals.update(quadrature_order="DOP853 axial work/clock/action", native_cells=len(dynamic_integrals))
            output_energy = energy_ev if electric_reference is None else float(electric_reference.axis_energy(snapshot, z))
            if dynamic:
                from temsim.physics.electric_wave_reference import scalar_potential
                phi = float(scalar_potential(snapshot, np.array([0., 0., z])))
                output_energy = electric_reference.entrance_energy_ev+(phi-electric_reference.entrance_phi_v
                    +electric_wave_history.dynamic_state["axial_work_ev"])
            metadata = {"schema": "controlled-potential-observation-v1", "snapshot_digest": snapshot.numerical_digest,
                "potential_operator_policy": operator_policy,
                "potential_support_policy": support_policy,
                "input_z_m": float(start_z_m), "output_z_m": z, "energy_ev": energy_ev,
                "input_probability": initial, "output_probability": current.probability,
                "aperture_loss": sum(row["lost_probability"] for row in losses), "aperture_events": losses,
                "applied_event_ids": sorted(applied), "stages": stages,
                "support_ledger": None if support_ledger is None else support_ledger.to_dict(),
                **({"representation_estimates": representation_estimates.to_dict(),
                    "representation_estimate_scope": "exploratory numerical estimate; not a certified global error bound",
                    "high_accuracy_eligible": False} if representation_estimates is not None else {}),
                "kinetic_phase_ledger": None if kinetic_phase_ledger is None else kinetic_phase_ledger.to_dict(),
                "electric_wave_history": None if electric_wave_history is None else electric_wave_history.to_dict(),
                "output_energy_ev": output_energy,
                "electric_axial_integrals": electric_integrals,
                "dynamic_clock": None if not dynamic else thaw_json(electric_wave_history.dynamic_state),
                "dynamic_scope": ("retarded adiabatic stationary-mode wavefront with column-added clock/work bounds; "
                    "the supplied gun-exit plane defines one reference wavefront, not a longitudinal pulse or all source TOFs") if dynamic else None,
                "numerical_support_loss_per_reference": 0. if support_ledger is None else math.fsum(
                    event["probability_per_reference"] for event in support_ledger.events[initial_support_count:]),
                "axial_flight_time_increment_s": distance/velocity if electric_integrals is None else electric_integrals["axial_flight_time_s"],
                "longitudinal_action_increment_j_s": distance*momentum if electric_integrals is None else electric_integrals["longitudinal_action_j_s"],
                "longitudinal_carrier_in_amplitude": False, "normalization_applied": False,
                "longitudinal_phase_convention": ("entrance-energy free carrier factored out; stable added electric spatial "
                    "and dynamic temporal phase already included in the complex envelope; AxialWaveReference is bookkeeping only")
                    if electric_reference is not None else "unchanged stationary magnetic reduced-phase convention",
                "source_scope": "caller-supplied controlled entrance; no physical gun-source qualification",
                "geometry_scope": "explicit thin absorbing aperture planes only; thick walls/material not inferred",
                "production_admission": False, "all_stopped": current.probability == 0.,
                "numerical_bound_scope": "kernel stage bounds retained; field/grid/splitting convergence independent"}
            outputs.append(PotentialPlaneObservation(current, z, metadata))
    return tuple(outputs)


def observe_potential_mode(snapshot, mode, start_z_m, observation_z_m, **options):
    """Keep absolute mode weight and its separate axial action/time reference."""
    if type(mode) is not WaveMode:
        raise ValueError("Potential mode observation requires an existing WaveMode")
    if "support_mode_weight" in options:
        raise ValueError("WaveMode support accounting uses its actual absolute weight")
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    if type(snapshot) is DynamicPotentialSnapshot:
        if mode.axial_reference is None:
            raise ValueError("Dynamic wave propagation requires its actual executed axial reference")
        options["initial_flight_time_s"] = mode.axial_reference.flight_time_s
    outputs = []
    for observation in observe_potential_plane(snapshot, mode.plane, start_z_m, observation_z_m,
            energy_ev=mode.energy_kev*1000., support_mode_weight=mode.weight_per_reference_electron, **options):
        norm = observation.plane.probability
        plane = replace(observation.plane, amplitude=observation.plane.amplitude/math.sqrt(norm)) if norm else observation.plane
        output_energy = observation.metadata["output_energy_ev"]
        if observation.metadata["electric_wave_history"] is not None:
            from temsim.physics.electric_wave_reference import momentum_velocity
            ratio = float(momentum_velocity(mode.energy_kev*1000.)[0]/momentum_velocity(output_energy)[0])
            plane = replace(plane, curvature_m1=None if plane.curvature_m1 is None else plane.curvature_m1*ratio,
                            tilt_rad=None if plane.tilt_rad is None else plane.tilt_rad*ratio)
        reference = mode.axial_reference
        if reference is not None:
            reference = reference.advance(observation.metadata["axial_flight_time_increment_s"],
                                           observation.metadata["longitudinal_action_increment_j_s"])
            if observation.metadata["dynamic_clock"] is not None:
                # Preserve the independently integrated absolute clock exactly;
                # repeated subtract/add roundtrips must not drift on resume.
                reference = replace(reference, flight_time_s=observation.metadata["dynamic_clock"]["flight_time_s"])
        output = replace(mode, plane=plane, weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         axial_reference=reference, energy_kev=(mode.energy_kev
                            if observation.metadata["electric_wave_history"] is None else output_energy*.001))
        metadata = thaw_json(observation.metadata)
        metadata.update(input_mode_weight=mode.weight_per_reference_electron,
                        output_mode_weight=output.weight_per_reference_electron,
                        aperture_loss_per_reference=mode.weight_per_reference_electron*metadata["aperture_loss"],
                        probability_units={"aperture_loss": "input plane cell-probability units",
                            "aperture_loss_per_reference": "electrons per original reference electron",
                            "numerical_support_loss_per_reference": "numerical probability removed per original reference electron"},
                        unit_shape_factorization=True,
                        axial_reference=None if reference is None else asdict(reference))
        outputs.append(PotentialModeObservation(output, observation.z_m, metadata))
    return tuple(outputs)


def observe_potential_checkpoint(snapshot, checkpoint, observation_z_m, **options):
    """Create existing TipGunCheckpoint objects, directly usable by its store.

    The name of the existing checkpoint class does not qualify this controlled
    entrance as a newly unified gun calculation. No checkpoint is saved here.
    """
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    if type(checkpoint) is not TipGunCheckpoint:
        raise ValueError("Potential checkpoint observation requires an existing TipGunCheckpoint")
    previous_snapshot = checkpoint.record.get("snapshot_digest")
    if previous_snapshot is not None and previous_snapshot != snapshot.numerical_digest:
        raise ValueError("Potential snapshot changed; restart from the original controlled entrance rather than continuing an obsolete field history")
    options = dict(options)
    operator_policy = potential_wave_operator_policy(options, checkpoint.record)
    from temsim.physics.electric_wave_reference import ElectricWaveHistory, electric_rows
    stored_electric = checkpoint.record.get("electric_wave_history")
    if stored_electric is not None:
        restored_electric = ElectricWaveHistory.from_dict(stored_electric)
        if "electric_wave_history" in options and options["electric_wave_history"] != restored_electric:
            raise ValueError("Saved electric wave energy/error history cannot be reset")
        options["electric_wave_history"] = restored_electric
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    if (electric_rows(snapshot) or type(snapshot) is DynamicPotentialSnapshot or options.get("electric_wave_history") is not None) and len(checkpoint.beam.modes) != 1:
        raise ValueError("Electric wave checkpoints require one independently tracked energy history")
    from temsim.physics.potential_material_phase_budget import validate_material_kinetic_record
    material_phase_history = validate_material_kinetic_record(checkpoint.record, options)
    stored_phase = checkpoint.record.get("kinetic_phase_ledger")
    if stored_phase is not None:
        restored_phase = KineticPhaseLedger.from_dict(stored_phase)
        if "kinetic_phase_ledger" in options and options["kinetic_phase_ledger"] != restored_phase:
            raise ValueError("Saved kinetic phase history and budget cannot be reset or replaced")
        options["kinetic_phase_ledger"] = restored_phase
        if (restored_phase.events and options.get("electric_wave_history") is None
                and (electric_rows(snapshot) or type(snapshot) is DynamicPotentialSnapshot)):
            raise ValueError("An executed electric/dynamic phase history cannot lose its original energy and timing reference")
    if options.get("kinetic_phase_ledger") is not None and len(checkpoint.beam.modes) != 1:
        raise ValueError("Controlled kinetic phase accounting is currently admitted for one mode only")
    start_z_m = checkpoint.plane_z_mm*.001
    electric_state = options.get("electric_wave_history")
    if electric_state is not None:
        material_native = "material_kinetic_native_plane" in checkpoint.record
        if material_native:
            from temsim.physics.potential_material_phase_budget import exact_material_kinetic_plane
            material_plane = exact_material_kinetic_plane(checkpoint.record, checkpoint.plane_z_mm)
        if ((material_native and material_plane != electric_state.last_z_m)
                or (not material_native and checkpoint.plane_z_mm != electric_state.last_z_m*1000.)):
            raise ValueError("Checkpoint plane differs from its electric wave history")
        start_z_m = electric_state.last_z_m
    phase_state = options.get("kinetic_phase_ledger")
    if type(phase_state) is KineticPhaseLedger and phase_state.last_z_m is not None:
        # Checkpoints expose mm; the ledger retains the exact executed metre
        # endpoint. Restore it only after the authoritative mm identity check.
        # General low-level continuation remains strictly equal in metres.
        native_material_plane = "material_kinetic_native_plane" in checkpoint.record
        if native_material_plane or checkpoint.plane_z_mm != phase_state.last_z_m*1000.:
            if material_phase_history is None:
                raise ValueError("Checkpoint plane differs from its kinetic phase history")
            from temsim.physics.potential_material_phase_budget import exact_material_kinetic_plane
            if exact_material_kinetic_plane(checkpoint.record, checkpoint.plane_z_mm) != phase_state.last_z_m:
                raise ValueError("Material native plane differs from its kinetic phase history")
        start_z_m = phase_state.last_z_m
    from temsim.physics.potential_support_lineage import validate_inherited_support
    inherited_support = validate_inherited_support(checkpoint.record, options)
    stored_support = checkpoint.record.get("support_ledger")
    if stored_support is not None:
        restored = NumericalSupportLedger.from_dict(stored_support)
        if "support_ledger" in options and options["support_ledger"] != restored:
            raise ValueError("Saved numerical support history and budgets cannot be reset or replaced")
        options["support_ledger"] = restored
    if options.get("support_ledger") is not None and len(checkpoint.beam.modes) != 1:
        raise ValueError("Controlled numerical support is currently admitted for one mode only")
    stored_estimates = checkpoint.record.get("representation_estimates")
    if stored_estimates is not None and stored_support is not None:
        from temsim.physics.potential_representation_estimates import RepresentationEstimateLedger
        restored_estimates = RepresentationEstimateLedger.from_dict(stored_estimates)
        supplied_estimates = options.get("representation_estimates")
        if isinstance(supplied_estimates, Mapping):
            supplied_estimates = RepresentationEstimateLedger.from_dict(supplied_estimates)
        if supplied_estimates is not None and supplied_estimates != restored_estimates:
            raise ValueError("Saved representation estimates cannot be reset or replaced")
        options["representation_estimates"] = restored_estimates
    support_policy = potential_support_policy(options, checkpoint.record)
    options.setdefault("already_applied_event_ids", tuple(checkpoint.record.get("applied_event_ids", ())))
    definitions = {row["event_id"]: thaw_json(row) for row in checkpoint.record.get("aperture_definitions", ())}
    for event in options.get("apertures", ()):
        if type(event) is not PotentialAperturePlane:
            raise ValueError("Only declared immutable aperture planes are supported")
        definition = asdict(event)
        if event.event_id in definitions and json_digest(definitions[event.event_id]) != json_digest(definition):
            raise ValueError("Previously applied aperture geometry changed; restart from the original entrance")
        definitions[event.event_id] = definition
    rows = [observe_potential_mode(snapshot, mode, start_z_m, observation_z_m, **options)
            for mode in checkpoint.beam.modes]
    result = []
    for index, z in enumerate(observation_z_m):
        observations = [row[index] for row in rows]
        metadata = {"schema": "controlled-potential-checkpoint-v1", "upstream_digest": checkpoint.digest,
            "potential_operator_policy": operator_policy,
            "potential_support_policy": support_policy,
            "snapshot_digest": snapshot.numerical_digest, "modes": [thaw_json(row.metadata) for row in observations],
            "applied_event_ids": sorted(set().union(*(row.metadata["applied_event_ids"] for row in observations))),
            "aperture_definitions": [definitions[key] for key in sorted(definitions)],
            "support_ledger": observations[0].metadata["support_ledger"] if len(observations) == 1 else None,
            "kinetic_phase_ledger": observations[0].metadata["kinetic_phase_ledger"] if len(observations) == 1 else None,
            "electric_wave_history": observations[0].metadata["electric_wave_history"] if len(observations) == 1 else None,
            "source_scope": "original entrance reference retained; controlled local potential propagation",
            "production_admission": False}
        if len(observations) == 1 and observations[0].metadata.get("representation_estimates") is not None:
            metadata["representation_estimates"] = thaw_json(observations[0].metadata["representation_estimates"])
            metadata["high_accuracy_eligible"] = False
        elif stored_estimates is not None:
            # A material prefix can be nonspendable while its approximation
            # evidence must still survive downstream observation and caching.
            metadata["representation_estimates"] = thaw_json(stored_estimates)
            metadata["high_accuracy_eligible"] = False
        if checkpoint.record.get("support_policy_transition") is not None:
            # An explicitly migrated policy and its immutable source receipt
            # must survive this rebuilt checkpoint together. Accepted support
            # events advance, while the original allocation is never reissued.
            for key in ("support_policy_transition", "unified_support_allocation", "unified_wave_identity",
                        "source_checkpoint_digest", "source_mode_id", "source_mode_weight_per_reference"):
                if key in checkpoint.record:
                    metadata[key] = thaw_json(checkpoint.record[key])
            raw = metadata["support_ledger"] if inherited_support is None else inherited_support["ledger"]
            metadata["unified_support_ledger_digest"] = None if raw is None else json_digest(raw)
        if inherited_support is not None:
            metadata.update(inherited_potential_support=inherited_support,
                            support_prefix_digest=inherited_support["digest"])
        if material_phase_history is not None:
            metadata.update(material_kinetic_history=material_phase_history,
                material_kinetic_prefix_digest=material_phase_history["prefix"]["digest"])
        result.append(TipGunCheckpoint(BeamState(tuple(row.mode for row in observations), checkpoint.beam.reference_plane),
            float(z)*1000., checkpoint.reference_current_a, metadata))
    return tuple(result)
