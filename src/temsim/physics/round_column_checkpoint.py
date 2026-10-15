"""Execute and cache the round prefix before the ordinary 2-D column stage."""
from dataclasses import asdict, replace
import math

from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.radial_cartesian_handoff import RadialColumnNumerics, radial_to_cartesian
from temsim.physics.radial_column_wave import round_column_prefix
from temsim.physics.tip_gun_wave import TipGunCheckpoint, _momentum_velocity
from temsim.physics.wave_flux import BeamState
from temsim.physics.wave_grid import WaveGridNumerics


def execute_round_prefix(state, checkpoint, stop_z_mm, *, maximum_step_mm=.5,
                         numerics=RadialColumnNumerics(), grid_numerics=WaveGridNumerics(),
                         cancelled=lambda: False, progress_callback=None, column_model="physical",
                         allow_cartesian=False):
    """Consume the executed gun's radial expansion or checked Cartesian field.

    The original mode identity, source weight, scattering history and phase
    references survive; electric work updates energy and the clock. A remaining non-axisymmetric operation is
    owned by the subsequent 2-D propagation, never approximated here.
    """
    numerics.validate(); grid_numerics.validate()
    from temsim.physics.ideal_column_reference import column_model_record, validate_column_model
    from temsim.physics.wave_checkpoint_store import resident_wave_bytes
    validate_column_model(column_model)
    arguments = {} if column_model == "physical" else {"column_model": column_model}
    if any(row.get("angular_order", 0) != 0 for row in checkpoint.record.get("radial_output_modes", ())):
        # The current Cartesian handoff represents m=0 only. Keep the entire
        # executed 2-D field rather than lose exp(i*m*theta) on reconstruction.
        return checkpoint
    def resident_input():
        auxiliary = (0 if hasattr(checkpoint.beam, "content_identity") else
                     sum(value.nbytes for value in checkpoint.auxiliary_arrays.values()))
        return resident_wave_bytes(checkpoint.beam)+auxiliary
    if "radial_output_modes" not in checkpoint.record and allow_cartesian:
        gun = getattr(state, "electron_gun", None)
        if gun is None or checkpoint.plane_z_mm != gun.exit_plane_z_mm:
            return checkpoint
        from temsim.physics.cartesian_radial_handoff import (
            CartesianRadialNumerics, CartesianRadialUnavailable, cartesian_to_radial,
        )
        conversion = CartesianRadialNumerics(maximum_samples=numerics.maximum_samples,
            maximum_working_bytes=grid_numerics.maximum_working_bytes,
            complex_tolerance=numerics.complex_tolerance, current_tolerance=numerics.current_tolerance)
        admission = []
        # Qualify the COMPLETE mixture before applying a single column
        # operator. A rejected mode cannot be dropped to enable this route.
        for mode in checkpoint.beam.modes:
            try:
                radial, evidence = cartesian_to_radial(mode.plane,
                    wavelength_m=float(wavelength_m(mode.energy_kev*1000)),
                    samples=numerics.initial_samples, numerics=conversion,
                    retained_bytes=resident_input()+mode.plane.amplitude.nbytes, cancelled=cancelled)
            except CartesianRadialUnavailable as error:
                if progress_callback:
                    progress_callback(0, len(checkpoint.beam.modes),
                        f"Executed Cartesian gun remains in 2-D ({mode.mode_id}): {error}")
                return checkpoint
            del radial
            admission.append({"mode_id": mode.mode_id, "evidence": evidence})
        arguments.update(cartesian_numerics=conversion, cartesian_admission=tuple(admission))
    z, radial_modes, record = round_column_prefix(state, checkpoint, stop_z_mm,
        maximum_step_mm=maximum_step_mm, initial_samples=numerics.initial_samples,
        maximum_samples=numerics.maximum_samples, backend=numerics.backend,
        maximum_working_bytes=grid_numerics.maximum_working_bytes, retained_bytes=resident_input(),
        cancelled=cancelled, progress_callback=progress_callback, **arguments)
    if not radial_modes:
        if z != checkpoint.plane_z_mm:
            raise ValueError("An empty round prefix cannot advance a physical checkpoint")
        return checkpoint
    if not checkpoint.plane_z_mm < z <= stop_z_mm:
        raise ValueError("Round prefix returned an invalid downstream plane")
    if tuple(mode.mode_id for mode, _ in radial_modes) != tuple(mode.mode_id for mode in checkpoint.beam.modes):
        raise ValueError("Round prefix changed the executed source mode identities")
    outputs, rows = [], []
    retained = resident_input()+sum(mode.plane.amplitude.nbytes+wave.radius_m.nbytes+wave.amplitude.nbytes
                                   for mode, wave in radial_modes)
    length_m = (z-checkpoint.plane_z_mm)*1e-3
    transport_rows = {row["mode_id"]: row for row in record.get("modes", ())}
    for mode, radial in radial_modes:
        if cancelled():
            raise InterruptedError("Round prefix checkpoint cancelled")
        if radial.angular_order != 0:
            raise ValueError("Round checkpoint Cartesian reconstruction requires angular order zero")
        transition = transport_rows.get(mode.mode_id)
        if transition is None:
            # Compatibility with the former constant-energy prefix contract.
            momentum, velocity = _momentum_velocity(mode.energy_kev*1000)
            exit_energy = mode.energy_kev
            flight_increment, action_increment = float(length_m/velocity), float(length_m*momentum)
        else:
            exit_energy = float(transition["exit_energy_kev"])
            flight_increment = float(transition["reference_flight_time_increment_s"])
            action_increment = float(transition["reference_longitudinal_action_increment_j_s"])
        if (not all(math.isfinite(value) and value > 0. for value in
                    (exit_energy, flight_increment, action_increment))):
            raise ValueError("Round prefix needs positive finite executed energy, time and action")
        plane, handoff = radial_to_cartesian(radial, wavelength_m=float(wavelength_m(exit_energy*1000)),
            numerics=numerics, grid_numerics=grid_numerics, retained_bytes=retained,
            cancelled=cancelled, progress_callback=progress_callback)
        norm = plane.probability
        reference = (None if mode.axial_reference is None else
                     mode.axial_reference.advance(flight_increment, action_increment))
        # Factor a unit shape and absolute weight; their product is exactly
        # the returned field. This does NOT restore the lost/numerical norm.
        output = replace(mode, plane=replace(plane, amplitude=plane.amplitude/math.sqrt(norm) if norm else plane.amplitude),
                         weight_per_reference_electron=mode.weight_per_reference_electron*norm,
                         energy_kev=exit_energy, axial_reference=reference)
        outputs.append(output)
        retained += output.plane.amplitude.nbytes
        rows.append({"mode_id": mode.mode_id, "handoff": handoff,
            "input_weight": mode.weight_per_reference_electron,
            "output_weight": output.weight_per_reference_electron,
            "exit_energy_kev": exit_energy,
            "reference_flight_time_increment_s": flight_increment,
            "reference_longitudinal_action_increment_j_s": action_increment,
            "axial_reference": None if reference is None else asdict(reference)})
    if cancelled():
        raise InterruptedError("Round prefix checkpoint cancelled before publication")
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), z,
        checkpoint.reference_current_a, {"schema": "executed-round-column-checkpoint-v1",
            **column_model_record(column_model),
            "upstream_digest": checkpoint.digest, "upstream": checkpoint.record,
            "round_execution": record, "modes": rows, "numerics": asdict(numerics),
            "grid_numerics": grid_numerics.column_identity(),
            "validation_status": "DEVELOPMENT_NOT_SOURCE_ADMISSION"})
