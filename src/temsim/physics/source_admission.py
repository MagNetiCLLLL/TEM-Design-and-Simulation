"""Honest admission for the gun-to-specimen coherent source-chain migration.

Tip mutual intensity and a development quadratic gun operator are available.
Column/specimen/detector, timed scan and conditional material-wave development
stages now exist; general-field and full-chain qualification remain open. Neither
a tip parameter label nor a development checkpoint qualifies a full image.
"""
from __future__ import annotations

import numpy as np

from temsim.immutable_json import freeze_json

PLANCK_J_S = 6.62607015e-34  # Exact SI definition.


class UnsupportedWaveSource(ValueError):
    code = "GUN_PHASE_CHAIN_UNAVAILABLE"


class UnsupportedIntrinsicLensLegacy(UnsupportedWaveSource):
    code = "INTRINSIC_LENS_REQUIRES_COLUMN_WAVE"


def require_legacy_aberration_compatibility(state, *, product,
                                           start_z_mm=None, stop_z_mm=None):
    """Legacy equivalent pupils/LCTs cannot carry arbitrary local lens actions.

    Only an enabled native lens in the consumed interval blocks this route.
    Matching field-map models already own their aberrations and deliberately
    leave the native coefficient inputs inactive. The empty/default path
    avoids provider construction or any numerical work.
    """
    from temsim.optics.intrinsic_aberrations import has_intrinsic_coefficients

    definitions = getattr(state, "lens_intrinsic_aberrations", {})
    if not definitions:
        return
    candidates = []
    for lens in getattr(state, "lenses", ()):
        row = definitions.get(lens.key, {})
        if not lens.enabled or not row or not has_intrinsic_coefficients(row):
            continue
        if start_z_mm is not None or stop_z_mm is not None:
            from temsim.lens_pose import lens_pose_registration
            pose = lens_pose_registration(state, lens.key)
            z_mm = float((pose.rotation_array @ np.array((0., 0., float(lens.z_mm)*1e-3))
                          + pose.origin_array_m)[2])*1e3
            if start_z_mm is not None and z_mm < float(start_z_mm):
                continue
            if stop_z_mm is not None and z_mm > float(stop_z_mm):
                continue
        candidates.append(lens.key)
    if not candidates:
        return
    from temsim.physics.lens_field_provider import active_mapped_providers
    mapped = {provider.lens_key for provider in active_mapped_providers(state)}
    active = [key for key in candidates if key not in mapped]
    if active:
        raise UnsupportedIntrinsicLensLegacy(
            f"{product}: the equivalent-pupil / first-order wave model cannot "
            f"apply per-lens intrinsic aberrations ({', '.join(active)}). "
            "Use Electron beam / Calculate beam for full-column coherent "
            "propagation. The coefficients were not ignored or replaced by "
            "fitted system residuals; all lens settings remain unchanged."
        )


def gun_phase_readiness(state):
    gun = state.electron_gun
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source, UnsupportedSourceModel
    try:
        require_physical_gun_source(gun)
    except UnsupportedSourceModel as error:
        return freeze_json({"status": "UNAVAILABLE", "code": error.code,
            "gun_model": str(gun.type_key), "reason": str(error),
            "coherent_source": "PROHIBITED_CUSTOM_EXIT_SOURCE",
            "gun_phase_transfer": "NOT_COMPUTED", "validation_status": "NOT_RUN"})
    from temsim.optics.electron_gun.tip_coherence import TipCoherence
    configured = isinstance(getattr(gun.emitter, "coherence", None), TipCoherence)
    if getattr(gun.emitter, "surface_model", None) is not None:
        if gun.emitter.surface_model.coherence is not None:
            return freeze_json({"status": "UNAVAILABLE", "code": UnsupportedWaveSource.code,
                "gun_model": str(gun.type_key), "coherent_source": "COHERENT_SURFACE_NEAR_FIELD_AVAILABLE",
                "gun_phase_transfer": "DEVELOPMENT_TWO_WAY_ROUND_GUN_AVAILABLE_NOT_QUALIFIED",
                "validation_status": "FULL_IMAGE_NOT_VALIDATED",
                "reason": "The coherent surface and development two-way relativistic round-gun solvers are available. Full-state boundary, basis and column convergence are not yet qualified for TEM/STEM imaging."})
        return freeze_json({"status": "UNAVAILABLE", "code": UnsupportedWaveSource.code,
            "gun_model": str(gun.type_key), "coherent_source": "SURFACE_CLASSICAL_FLUX_ONLY",
            "gun_phase_transfer": "NOT_IMPLEMENTED_FOR_GROUNDED_SURFACE",
            "validation_status": "NOT_RUN",
            "reason": "Classical surface emission is selected; no coherent phase is defined. Coherent near-field work is available separately, but full TEM/STEM imaging is not yet qualified."})
    return freeze_json({
        "status": "NOT_COMPUTED", "code": UnsupportedWaveSource.code,
        "gun_model": str(gun.type_key), "particle_source": "gun emission and relativistic Lorentz transport",
        "coherent_source": "TIP_GAUSSIAN_SCHELL_CONFIGURED" if configured else "TIP_COHERENCE_NOT_CONFIGURED",
        "gun_phase_transfer": "DEVELOPMENT_QUADRATIC_OPERATOR_AVAILABLE_NOT_COMPUTED",
        "stationary_zero_loss_pipeline": "DEVELOPMENT_AVAILABLE_NOT_FULL_PRODUCT_ACCEPTANCE",
        "dynamic_scan_and_inelastic_waves": "DEVELOPMENT_ARRIVAL_TIME_COILS_AND_CONDITIONAL_MATERIAL_TRAJECTORIES",
        "validation_status": "NOT_RUN",
        "missing": (["Explicit tip coherence parameters"] if not configured else []) +
                   ["General non-paraxial/imported-field gun phase operators",
                    "Converged physical-tip column/specimen/scanned-detector benchmarks",
                    "Longitudinal pulse wavepackets and atomic inelastic transition potentials",
                    "Energy-filter wave integration after upstream completion"],
    })


def require_gun_wave_source(state, *, product):
    """Closed production boundary until the full requested product is qualified.

    Do not accept arbitrary private attributes, metadata or caller-supplied
    arrays as proof. This deliberately cannot be enabled with origin='gun'.
    Ray tracing, EDS particle transport and historical viewing remain available.
    """
    require_legacy_aberration_compatibility(state, product=product)
    readiness = gun_phase_readiness(state)
    if readiness["code"] == "TIP_ORIGIN_REQUIRED":
        raise UnsupportedWaveSource(f"{product}: {readiness['reason']}")
    if readiness.get("coherent_source") in {"SURFACE_CLASSICAL_FLUX_ONLY", "COHERENT_SURFACE_NEAR_FIELD_AVAILABLE"}:
        raise UnsupportedWaveSource(f"{product}: {readiness['reason']}")
    raise UnsupportedWaveSource(
        f"{product}: gun-to-specimen coherent phase is unavailable for {readiness['gun_model']}. "
        "The existing ray-conditioned pupil is historical/reduced-order only. "
        "Tip coherence and a development quadratic gun operator are available, but are not full image admission. "
        "Development timed scan coils and conditional material-model inelastic waves are connected; full image qualification remains open. "
        "An effective exit source is not permitted."
    )


def admit_requested_wave_products(state):
    """Reject unsupported new wave work before heavy preparation/cache reuse."""
    from temsim.physics.wave_imaging import tem_wave_imaging_enabled
    from temsim.optics.electron_gun.source_policy import require_physical_gun_source
    require_physical_gun_source(state.electron_gun)
    sample = state.sample
    scan = getattr(state, "ac_deflector", None)
    if tem_wave_imaging_enabled(state) or (scan is not None and scan.enabled and scan.scan_enabled
                                         and bool(getattr(sample, "stem_wave_enabled", False))):
        require_gun_wave_source(state, product="Wave imaging")


def launch_emittance_audit(gun, *, count=4096):
    """Audit a possible literal quantum interpretation of the CLASSICAL source.

    This is not a coherence reconstruction. For each Cartesian pair, a quantum
    covariance must satisfy sqrt(var(x)var(px)-cov(x,px)^2) >= hbar/2.
    The ratio below is a necessary test, not sufficient quantum validation.
    The launch plane is z=0 of tracing.trace_feg_to_exit. No lens is adjusted.
    """
    from temsim.physics.relativistic_lorentz import momentum_from_kinetic_energy_ev
    emitted = gun.emit(count)
    energy = float(gun.emitter.emission_energy_ev) + emitted.energy_offset_ev
    direction = np.column_stack((emitted.tx_rad, emitted.ty_rad, np.ones(energy.size)))
    if getattr(gun.emitter, "surface_model", None) is not None:
        energy, direction = emitted.surface_energy_ev, emitted.surface_direction
    momentum = momentum_from_kinetic_energy_ev(energy, direction)
    weight = np.asarray(emitted.weight, float)
    weight = weight / weight.sum()
    values = np.column_stack((emitted.x_m, emitted.y_m, momentum[:, 0], momentum[:, 1]))
    mean = weight @ values
    centred = values - mean
    covariance = (centred.T * weight) @ centred
    minimum_action = PLANCK_J_S / (4*np.pi)
    action = [float(np.sqrt(max(0., covariance[k, k]*covariance[k+2, k+2]-covariance[k, k+2]**2))) for k in (0, 1)]
    surface = getattr(gun.emitter, "surface_model", None) is not None
    return {"schema": "classical-launch-emittance-audit-v1", "sample_count": int(count),
            "reference_plane": "curved tip surface; Cartesian projection" if surface else "gun launch z=0 mm",
            "energy_min_ev": float(energy.min()), "energy_max_ev": float(energy.max()),
            "action_rms_j_s": action, "quantum_minimum_action_j_s": minimum_action,
            "quantum_bound_ratio_xy": [v/minimum_action for v in action],
            "necessary_quantum_covariance_condition": ("NOT_APPLICABLE_CURVED_CLASSICAL_SURFACE" if surface
                                                       else "PASS" if min(action) >= minimum_action else "FAIL"),
            "scope": "Classical model interpretation audit only; does not create or validate coherent modes"}
