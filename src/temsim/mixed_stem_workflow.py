"""Captured particle-to-mixed-wave acquisition for the Scanning Image page.

The raster translates the executed entrance density in the specimen plane.
It does not retrace scan-coil waveforms or claim downstream detector conjugacy.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext

import numpy as np

from temsim import input_io


@dataclass(frozen=True)
class MixedStemAcquisition:
    state_snapshot: object
    particle_result: object
    scan: object
    particle_boundary: object = None


def specimen_scan_positions(state):
    """Return the full selected rectangular raster in nm, shaped (ny,nx,2)."""
    scan = state.ac_deflector
    nx, ny = int(scan.scan_pixels_x), int(scan.scan_lines)
    pitch = float(scan.scan_pixel_size_nm)
    origin = np.asarray((state.sample.scan_origin_x_nm,
                         state.sample.scan_origin_y_nm), dtype=float)
    if (nx < 2 or ny < 2 or nx > 4096 or ny > 4096
            or not np.isfinite(pitch) or pitch <= 0
            or not np.isfinite(origin).all()):
        raise ValueError("A mixed STEM raster requires 2 to 4096 pixels per axis, a positive pitch and finite origins")
    x = (np.arange(nx)-(nx-1)*.5)*pitch+origin[0]
    y = (np.arange(ny)-(ny-1)*.5)*pitch+origin[1]
    xx, yy = np.meshgrid(x, y)
    return np.stack((xx, yy), axis=-1)


def _check(cancelled):
    if cancelled():
        raise InterruptedError("Mixed STEM acquisition cancelled")


@input_io.using_state_inputs
def calculate_mixed_stem(state, *, request=None, existing_result=None,
                         cancelled=lambda: False, progress_callback=None):
    """Prepare actual particles if necessary, then acquire a CUDA mixed raster.

    The source count, source law, energy width, lens excitations and integration
    step are retained. This worker entry works with either displayed quality;
    it never substitutes the old small optical-preview population. Reuse is
    limited to an exactly matching captured input, not a displayed image.
    """
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.compute_backend import require_coherent_cuda
    from temsim.physics.mixed_stem_scan import (
        MixedStemScanRequest, simulate_mixed_stem_scan, preflight_mixed_stem_material,
    )
    from temsim.physics.mixed_wave_particles import particle_source_from_result
    from temsim.simulation_modes import require_coherent_mode
    from temsim.simulation_pipeline import calculate
    from temsim.specimen.source import validate_sample_source, specimen_is_vacuum

    _check(cancelled)
    require_coherent_cuda("Require GPU")
    request = (MixedStemScanRequest() if request is None else request).validate()
    require_coherent_mode(state)
    inputs = capture_instrument_snapshot(state)
    captured = inputs.restore()
    count = int(captured.electron_gun.emitter.ray_count)
    if count < 1000:
        raise ValueError("Mixed STEM requires at least 1000 original positive-current source particles; increase the selected particle count")
    validate_sample_source(captured.sample)
    if specimen_is_vacuum(captured.sample):
        raise ValueError("Insert an atomic specimen before calculating a mixed coherent STEM image")
    positions = specimen_scan_positions(captured)
    if bool(getattr(captured.sample, "stem_wave_enabled", False)):
        raise ValueError("The saved Tip wave STEM request is retired; select the mixed coherent CUDA acquisition")
    preflight_mixed_stem_material(captured)
    captured._tuning_cancelled = cancelled

    def progress(done, total, message):
        _check(cancelled)
        if progress_callback is not None:
            progress_callback(done, total, message)

    previous_state = getattr(existing_result, "state_snapshot", None)
    reusable = (previous_state is not None
                and capture_instrument_snapshot(previous_state).digest == inputs.digest)
    if reusable:
        progress(0, 1, "Checking the executed particle entrance")
        particles = existing_result
    else:
        progress(0, 1, "Preparing the selected original particles")
        particles = calculate(captured, workflow="rays", progress_callback=progress)
    _check(cancelled)
    source = particle_source_from_result(particles, cancelled=cancelled,
                                        progress_callback=progress)
    _check(cancelled)
    from temsim.cache_memory import retained_memory_inventory
    from temsim.physics.host_memory_budget import (
        active_host_memory_budget_bytes, host_memory_budget,
    )
    cap = active_host_memory_budget_bytes()
    scope = nullcontext()
    if cap is not None:
        # Accepted inputs are already counted by shared admission. Histories
        # created inside this worker remain live throughout the wave stages.
        # Count shared buffers once, then reserve the actual additional owners.
        before = retained_memory_inventory(captured, existing_result)
        after = retained_memory_inventory(captured, existing_result, particles, source)
        growth = sum(size for owner, size in after.items() if owner not in before)
        if growth >= cap:
            raise MemoryError("New particle history exhausts the admitted mixed STEM host budget")
        scope = host_memory_budget(cap-growth)
    with scope:
        scan = simulate_mixed_stem_scan(particles.state_snapshot, source, request,
                                       positions_nm=positions, cancelled=cancelled,
                                       progress_callback=progress)
    _check(cancelled)
    return MixedStemAcquisition(particles.state_snapshot, particles, scan)


def calculate_mixed_stem_boundary(boundary, *, request=None, cancelled=lambda: False,
                                  progress_callback=None):
    """Consume a checked completed classical boundary without a solver restart."""
    from temsim.physics.compute_backend import require_coherent_cuda
    from temsim.physics.particle_wave_source_io import ValidatedParticleBoundary
    from temsim.physics.mixed_stem_scan import MixedStemScanRequest, simulate_mixed_stem_scan
    _check(cancelled)
    require_coherent_cuda("Require GPU")
    if type(boundary) is not ValidatedParticleBoundary:
        raise TypeError("A mixed STEM boundary acquisition requires a validated immutable particle boundary")
    request = (MixedStemScanRequest() if request is None else request).validate()
    state = boundary.restore_inputs_for_wave()
    positions = specimen_scan_positions(state)
    scan = simulate_mixed_stem_scan(state, boundary.source, request,
        validated_boundary=boundary, positions_nm=positions, cancelled=cancelled,
        progress_callback=progress_callback)
    _check(cancelled)
    return MixedStemAcquisition(state, None, scan, boundary)
