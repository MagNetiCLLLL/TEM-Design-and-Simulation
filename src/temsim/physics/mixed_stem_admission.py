"""Purpose-specific admission for the local particle-to-wave STEM raster.

This does not admit ideal-current providers to the general column wave kernel.
The local specimen operator retains its independent full-field remainder and
numerical qualification checks after an exact executed particle handoff.
"""
from __future__ import annotations

import numpy as np


def require_local_mixed_stem_mode(state):
    """Admit a field model without changing drives, source or optical inputs.

    Ideal-current windings require an actual static near-specimen snapshot in
    the native zero-axis gauge before any new source transport. A raster here
    translates the executed entrance density; it does not freeze a live coil
    waveform. Gaussian field admission retains its existing later local-field
    checks. No provider is replaced by a native or Gaussian calibration.
    """
    from temsim.simulation_modes import is_thin, is_unified

    if is_thin(state):
        raise ValueError(
            "Local mixed particle-wave STEM is unavailable in Ideal Optics "
            "(thin elements); select Finite-Field Optics explicitly"
        )
    if not is_unified(state):
        raise ValueError("Mixed STEM requires the executed Finite-Field Optics particle source")
    if getattr(state, "round_lens_field_model", "gaussian") != "copper_coil":
        return

    sample = state.sample
    thickness_nm = float(sample.thickness_nm)
    upper_mm = float(sample.z_mm)-thickness_nm*.5e-6
    start = upper_mm*1e-3
    stop = start+thickness_nm*1e-9
    if not np.isfinite((start, stop)).all() or stop <= start:
        raise ValueError("Local mixed STEM needs a finite positive specimen thickness")
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot

    runtime = capture_unified_field_runtime(state, start, stop)
    if type(runtime.snapshot) is DynamicPotentialSnapshot:
        raise ValueError(
            "A stationary local mixed STEM scan cannot consume dynamic AC/Descan "
            "raster or wobble fields. Explicitly prepare stationary virtual-raster "
            "inputs before generating the particle source; no drive was disabled "
            "or frozen automatically"
        )
    snapshot = runtime.snapshot
    a0 = snapshot.sample(np.array([[0., 0., start]]))["A_t_m"][0]
    if (np.linalg.norm(a0) > 1e-20 or np.any(snapshot.gauge.gradient_t_m)
            or np.any(snapshot.gauge.hessian_t)):
        raise ValueError(
            "Mixed STEM split currently requires the native zero-axis magnetic "
            "gauge; a canonical spectrum is not a mechanical angle chart"
        )
