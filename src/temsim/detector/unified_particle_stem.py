"""Projected particle STEM with shared-potential local optical responses.

This is the selected frozen-probe, thin independent-atom projection model.
The executed source is retained; finite-field derivatives move its reference
response across the raster. It is not a nonlinear per-pixel column retrace.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from temsim.optics.shared_deflectors import shared_channel_enabled
from temsim.physics.beam_current import sample_illumination_absent
from temsim.physics.optical_tuning import check_tuning_cancelled
from temsim.physics.scan_geometry import raster_sample_grid
from temsim.physics.unified_scan_calibration import (
    calibrate_unified_scan, measure_scan_responses,
)


def _paired_position_response(measured, component, z_mm):
    """Combine physical coil derivatives using the current logical wiring."""
    if not shared_channel_enabled(component):
        return np.zeros((2, 2))
    index = measured["z_mm"].index(float(z_mm))
    host = str(getattr(component, "_physical_host_key", component.key))
    result = np.zeros((2, 2))
    for coil, matrix in enumerate(component.coil_kick_matrices()):
        key = f"{host}:{coil}"
        if key not in measured["phase"]:
            # A coil downstream of every requested plane is absent from the
            # experiment. Missing upstream coil evidence is an error.
            entrance = float(component.upper_z_mm if coil == 0 else component.lower_z_mm)
            if entrance <= float(z_mm):
                raise ValueError(f"Shared-potential STEM response is missing physical coil {key}")
            continue
        result += np.asarray(measured["phase"][key][index], dtype=float)[:2] @ np.asarray(matrix)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError("Shared-potential STEM response contains invalid position derivatives")
    return result * 1e3  # Executor metres/radian -> physical detector mm/radian.


def _scope(state):
    from temsim.detector.projected_response import DIRECT_RAY_LIMIT, POLAR_BIN_COUNT, AZIMUTH_SAMPLE_COUNT
    return {
        "unified_potential_stem": True,
        "optical_model": "shared_potential_reference_trajectories_and_local_affine_raster",
        "raster_transport_model": "frozen_reference_probe_affine_response",
        "raster_time_model": "pixel midpoint drive commands; reference trajectories retain their executed clocks, without per-pixel flight-time retracing",
        "per_pixel_nonlinear_column_retrace": False,
        "coherent_imaging": False,
        "quantitative_model": False,
        "projected_reference_plane": True,
        "scattering_time_model": "instantaneous zero-depth projected screen; executed incident arrival clock retained",
        "source_particle_count": int(state.electron_gun.ray_count),
        "projected_direct_ray_limit": DIRECT_RAY_LIMIT,
        "projected_angular_samples_per_element_nominal": POLAR_BIN_COUNT * AZIMUTH_SAMPLE_COUNT,
        "projected_angular_sampling_scope": "bounded auxiliary quadrature with detector-boundary hints; source population is unchanged",
        "scan_response_scope": "local derivative at nominal beam energy about the DC chief, measured from the same finite potentials and geometry",
        "detector_angle_label_scope": "physical-plane hit masks; no legacy equivalent collection-angle conversion",
        "sample_state_scope": "auxiliary intensity observable, not a finite-volume material checkpoint",
    }


def acquire_unified_projected_stem(
    simulation, state, *, detector_keys=None, pixels_x=None, pixels_y=None,
    specimen_interactions=None, observation_stop_z_mm=None, scan_calibrated=False,
    progress_callback=None, diffraction_sink=None,
):
    """Acquire the admitted projected model without invoking legacy optics."""
    from temsim.detector.stem_signal import DetectorSignal, _current_values, _readout_view, _stem_result
    from temsim.detector.projected_stem import acquire_projected_stem_scan
    from temsim.calculation_workflow import admit_unified_product

    sample, ac, descan = state.sample, state.ac_deflector, state.descan_deflector
    check_tuning_cancelled(state)
    admit_unified_product(state, "stem")
    if bool(sample.stem_wave_enabled):
        raise ValueError("Shared-potential coherent STEM acquisition is not supported by the projected particle model")
    if bool(sample.stem_fourdstem_enabled) or diffraction_sink is not None:
        raise ValueError("Projected particle STEM does not produce 4D-STEM diffraction data")
    if str(sample.stem_particle_model) != "projected_atoms":
        raise ValueError("Shared-potential material-path STEM requires a separate per-pixel transport implementation")
    if not bool(ac.enabled and ac.scan_enabled):
        raise ValueError("AC Scan Coil and its raster drive must both be enabled before STEM signal acquisition")
    if any(bool(getattr(part, "wobble_enabled", False)) for part in (ac, descan)):
        raise ValueError("Shared-potential projected STEM does not support simultaneous scan-coil wobble")
    inserted = [plane for plane in state.stem_detectors if bool(plane.inserted)]
    if not inserted:
        raise ValueError("Shared-potential STEM requires an inserted scanning detector")
    stop = max(float(plane.z_mm) for plane in inserted)
    if observation_stop_z_mm is not None:
        observed = float(observation_stop_z_mm)
        if not math.isfinite(observed) or observed <= float(sample.z_mm) or stop > observed:
            raise ValueError("The selected section has not reached every inserted scanning detector")
    requested = ({plane.key for plane in inserted if bool(plane.readout_enabled)}
                 if detector_keys is None else {str(key) for key in detector_keys})
    missing = requested - {plane.key for plane in inserted}
    if missing:
        raise ValueError("STEM readout requires physically inserted detector(s): " + ", ".join(sorted(missing)))
    px = int(ac.scan_pixels_x if pixels_x is None else pixels_x)
    py = int(ac.scan_lines if pixels_y is None else pixels_y)
    x_factors, y_factors, times = raster_sample_grid(ac, pixels_x=px, pixels_y=py, maximum_count=None)
    metadata = _scope(state)
    metadata["executed_incident_particle_count"] = int(np.size(simulation.incident.alive))
    metadata["scan_times_s"] = times.tolist()
    metadata["reference_emission_time_s"] = float(getattr(state, "simulation_time_s", 0.0))
    if sample_illumination_absent(simulation, state):
        # A fully blocked source has no probe centroid or measurable field
        # footprint. Retain the requested raster and emitted-current ledger.
        x_um = (x_factors * ac.scan_field_of_view_x_nm * .5 + sample.scan_origin_x_nm) * 1e-3
        y_um = (y_factors * ac.scan_field_of_view_y_nm * .5 + sample.scan_origin_y_nm) * 1e-3
        images = {plane.key: np.zeros_like(x_um) for plane in inserted}
        signals = {plane.key: DetectorSignal(plane.key, plane.name, 0., *_current_values(state, 0.), None)
                   for plane in inserted}
        metadata.update(zero_illumination=True, sample_incident_source_probability=0.,
                        coordinate_scope="requested raster; no surviving specimen illumination",
                        source_probability_conservation_error=0., scan_pixels_x=px, scan_pixels_y=py,
                        model="projected_atomic_scattering", upstream_uncollected_fraction=1.)
        result = _stem_result(state, simulation, x_um, y_um, images, signals, metadata,
                              uncollected_fraction=np.ones_like(x_um),
                              absorbed_fraction=np.zeros_like(x_um), truncated_fraction=np.zeros_like(x_um))
        if progress_callback is not None:
            progress_callback(1, 1, "No incident current at the specimen; zero STEM frame retained")
        return _readout_view(result, inserted_keys=tuple(images), readout_keys=requested)

    simulation_inelastic = getattr(simulation, "real_interactions", None)
    supplied = getattr(specimen_interactions, "inelastic_distribution", None)
    if simulation_inelastic is not None and supplied is not None and simulation_inelastic is not supplied:
        raise ValueError("STEM received an inelastic population from a different column calculation")
    distribution = supplied if supplied is not None else simulation_inelastic
    if distribution is None:
        from temsim.specimen.inelastic import real_inelastic_distribution
        distribution = real_inelastic_distribution(state)
    metadata["inelastic_distribution"] = distribution.metrics()
    if any(channel.probability > 0. and channel.energy_loss_ev > 0. for channel in distribution.channels):
        raise ValueError("Shared-potential projected STEM cannot assign an executed flight clock to aggregate positive-energy-loss channels; no loss was silently discarded")
    maximum_step_mm = min(.2, float(state.step_mm))
    if not scan_calibrated:
        calibrate_unified_scan(state, observation_stop_z_mm=stop, maximum_step_mm=maximum_step_mm)
    planes = sorted((plane for plane in state.recording_planes
                     if plane.inserted and float(sample.z_mm) <= float(plane.z_mm) <= stop),
                    key=lambda plane: float(plane.z_mm))
    if progress_callback is not None:
        progress_callback(0, 1, "Measuring shared-potential raster responses")
    measured = measure_scan_responses(state, (float(sample.z_mm), *(float(plane.z_mm) for plane in planes)),
                                      maximum_step_mm=maximum_step_mm)
    sample_response = _paired_position_response(measured, ac, sample.z_mm)
    plane_responses = {plane.key: (_paired_position_response(measured, ac, plane.z_mm),
                                  _paired_position_response(measured, descan, plane.z_mm))
                       for plane in planes}
    metadata["scan_response_receipt"] = deepcopy(measured["receipt"])
    metadata["scan_response_maximum_step_mm"] = maximum_step_mm
    metadata["descan_physical_host_key"] = str(getattr(descan, "_physical_host_key", descan.key))
    factors = np.stack((x_factors, y_factors), axis=-1)
    kicks = np.einsum("ij,...j->...i", np.asarray(ac.scan_command_matrix_mrad), factors)
    epoch = float(getattr(state, "simulation_time_s", 0.0))
    baseline = np.asarray(ac.scan_kick_mrad(epoch))
    baseline_descan = np.asarray(descan.scan_kick_mrad(epoch) if shared_channel_enabled(descan) else (0., 0.))
    offsets = (kicks * 1e-3) @ sample_response.T
    x_um = offsets[..., 0] * 1e3 + float(sample.scan_origin_x_nm) * 1e-3
    y_um = offsets[..., 1] * 1e3 + float(sample.scan_origin_y_nm) * 1e-3
    result = acquire_projected_stem_scan(
        simulation, state, inserted, scan_x_um=x_um, scan_y_um=y_um,
        kick_grid_mrad=kicks, baseline_scan_mrad=baseline, scan_times_s=times,
        baseline_descan_scan_mrad=baseline_descan, sample_response=sample_response,
        real_interactions=distribution,
        optical_response={"plane_responses": plane_responses, "metadata": metadata},
        progress_callback=progress_callback,
    )
    check_tuning_cancelled(state)
    return _readout_view(result, inserted_keys=[plane.key for plane in inserted], readout_keys=requested)
