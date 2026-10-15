"""Atomic aperture-conditioned column approximation, never a replacement source.

The full captured Cartesian field is interpolated, smoothly windowed before a
strong spherical kick, and propagated by the ordinary column operator. Only a
checkpoint AFTER the receiving physical aperture can be published. The large
discarded incident norm is not a unitary L2 error bound or physical absorption.
Guard-window convergence and the separately reported empirical wall estimate
qualify only the transmitted field of this explicit development method.
"""
from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from temsim.immutable_json import json_digest
from temsim.physics.wave_device import array_module, to_host


SCHEMA = "aperture-conditioned-cartesian-window-v1"


class AcceptanceWindowUnavailable(ValueError):
    """This span is outside the optional method's eligibility domain."""


@dataclass(frozen=True)
class AcceptanceWindowNumerics:
    enabled: bool = True
    stationary_guard_factor: float = 12.
    guard_scales: tuple[float, ...] = (.5, 1., 2.)
    plateau_fraction: float = .7
    minimum_taper_cells: int = 8
    aperture_radius_cells: int = 128
    output_halfwidth_radii: float = 4.
    maximum_relative_complex_change: float = .005
    maximum_relative_current_change: float = .001
    maximum_empirical_wall_intensity_error: float = .01

    def __post_init__(self):
        object.__setattr__(self, "guard_scales", tuple(self.guard_scales))

    def validate(self):
        if type(self.enabled) is not bool:
            raise ValueError("Acceptance window enabled must be boolean")
        for key in ("stationary_guard_factor", "output_halfwidth_radii"):
            value = getattr(self, key)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 1:
                raise ValueError(f"{key} must be finite and greater than one")
        if not 0 < self.plateau_fraction < 1:
            raise ValueError("Acceptance plateau fraction must lie between zero and one")
        if (len(self.guard_scales) < 3 or tuple(sorted(set(self.guard_scales))) != self.guard_scales
                or any(not math.isfinite(x) or x <= 0 for x in self.guard_scales)
                or 1. not in self.guard_scales):
            raise ValueError("Acceptance guards require increasing positive scales including one")
        for key in ("minimum_taper_cells", "aperture_radius_cells"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 8:
                raise ValueError(f"{key} must be an integer of at least eight")
        for key in ("maximum_relative_complex_change", "maximum_relative_current_change",
                    "maximum_empirical_wall_intensity_error"):
            if isinstance(getattr(self, key), bool) or not 0 < getattr(self, key) < .1:
                raise ValueError(f"{key} must be positive and below 0.1")
        return self


def find_acceptance_span(prepared):
    """Find one complete Cs-to-next-aperture interval, using actual plan nodes."""
    from temsim.physics.column_wave import _slice_prepared, _PlacedAperture, _ParallelBore
    from temsim.physics.posed_wave_hardware import PosedWaveBore
    from temsim.physics.posed_wave_aperture import PosedWaveAperture
    from temsim.optics.condenser_aperture import ContinuousApertureComponent
    from temsim.optics.model import Aperture
    plan, radii, stops, owners = prepared
    for kick in np.flatnonzero(plan.cs_kick_m3):
        if kick < 1:
            continue
        following = sorted(i for i in stops if i > kick)
        if not following:
            continue
        last = following[0]
        if len(stops[last]) != 1:
            continue
        aperture = stops[last][0]
        if type(aperture) not in (Aperture, ContinuousApertureComponent):
            continue
        if isinstance(aperture, (_PlacedAperture, _ParallelBore, PosedWaveBore, PosedWaveAperture)):
            continue
        radius = float(getattr(aperture, "radius_mm", 0.))*1e-3
        if (not math.isfinite(radius) or radius <= 0 or
                getattr(aperture, "offset_x_mm", 0.) or getattr(aperture, "offset_y_mm", 0.)):
            continue
        if np.any(plan.cs_kick_m3[kick+1:last+1]):
            continue
        first = int(kick)-1
        span = _slice_prepared(prepared, first, last)
        sub = span[0]
        if (getattr(sub, "mapped_fields", ()) or getattr(sub, "posed_spherical_kicks", ())
                or getattr(sub, "intrinsic_aberration_kicks", ())
                or any(row.get("dynamic", False) for row in span[3])):
            continue
        if any(np.any(getattr(sub, name)) for name in (
                "midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
                "midpoint_hex_normal_m3", "midpoint_hex_skew_m3", "kick_x_rad", "kick_y_rad")):
            continue
        if any(np.any(getattr(sub, name, ())) for name in ("dipole_bx_t", "dipole_by_t")):
            continue
        return {"first": first, "last": last, "kick_index": 1,
                "prepared": span, "aperture_radius_m": radius, "aperture_key": aperture.key,
                "start_z_mm": float(sub.z_mm[0]), "kick_z_mm": float(sub.z_mm[1]),
                "stop_z_mm": float(sub.z_mm[-1]),
                "identity": json_digest((SCHEMA, sub.signature, radii[first:last+1].tolist(),
                                         aperture.key, radius))}
    return None


def stationary_input_radius(wave, paths, spherical_m3, radius_m, wavelength_m):
    """Monotone round cubic mapping; no arbitrary choice among real roots."""
    from scipy.optimize import brentq
    from temsim.physics.radial_column_wave import round_map
    matrix = np.eye(4)
    for path in paths:
        if np.any(path.offset):
            raise AcceptanceWindowUnavailable("A shifted path needs the ordinary column operator")
        round_map(path.matrix)
        matrix = path.matrix@matrix
    (a, b, _, _), _ = round_map(matrix, with_rotation=True)
    corners = np.array(((-.5, -.5), (-.5, .5), (.5, -.5), (.5, .5)))
    bandwidth = float(np.max(np.linalg.norm(corners@np.linalg.inv(wave.basis_m), axis=1)))
    envelope_displacement = abs(b)*wavelength_m*bandwidth
    if envelope_displacement > .01*radius_m:
        raise AcceptanceWindowUnavailable("The represented envelope bandwidth exceeds the receiving-aperture window domain")
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    curvature = float(np.trace(q)/2)
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    if (not np.allclose(q, np.eye(2)*curvature, rtol=1e-8, atol=1e-8)
            or np.linalg.norm(tilt-q@wave.origin_m) > 1e-8):
        raise AcceptanceWindowUnavailable("Acceptance window requires a centred round phase carrier")
    linear, cubic = a+b*curvature, -b*spherical_m3
    if not all(math.isfinite(x) for x in (linear, cubic)) or linear == 0 or linear*cubic < 0:
        raise AcceptanceWindowUnavailable("The aperture has multiple or degenerate stationary branches")
    effective_radius = radius_m+envelope_displacement
    root = brentq(lambda r: abs(linear)*r+abs(cubic)*r**3-effective_radius,
                  0., effective_radius/abs(linear), xtol=1e-18, rtol=1e-13)
    return float(root), {"matrix": matrix.tolist(), "linear": linear, "cubic_m2": cubic,
                         "stationary_input_radius_m": float(root),
                         "maximum_envelope_displacement_m": envelope_displacement,
                         "envelope_bandwidth_policy": "Complete original-lattice Nyquist box; at most 1 percent of aperture radius"}


def crop_acceptance_window(wave, *, radius_m, wavelength_m, numerics, grid_numerics,
                           cancelled=lambda: False, retained_bytes=0):
    """Original Fourier interpolant, radial C2 window and exact carrier shift."""
    from temsim.physics.wave_grid import refine_plane_wave
    numerics.validate()
    if cancelled():
        raise InterruptedError("Acceptance window cancelled")
    pitch = float(np.linalg.norm(wave.basis_m, ord=2))
    target = radius_m*(1.-numerics.plateau_fraction)/numerics.minimum_taper_cells
    factor = max(1, math.ceil(pitch/target))
    if factor > 1:
        wave = refine_plane_wave(wave, tuple(factor*n for n in wave.amplitude.shape),
                                 numerics=grid_numerics, retained_bytes=retained_bytes)
    xp = array_module(wave.amplitude)
    xy = wave.coordinates_m()
    r = xp.sqrt(xp.sum(xy*xy, axis=0))
    # The guard must fit inside the actual represented spatial window.
    inverse = np.linalg.inv(wave.basis_m)
    centre = -inverse@wave.origin_m+np.array(wave.amplitude.shape[::-1])//2
    extent = radius_m*np.linalg.norm(inverse, axis=1)
    if np.any(centre-extent < 2) or np.any(centre+extent > np.array(wave.amplitude.shape[::-1])-3):
        raise AcceptanceWindowUnavailable("Acceptance guard extends beyond the executed input field")
    active = r < radius_m
    ys, xs = xp.nonzero(active)
    y0, y1 = max(0, int(ys.min())-2), min(wave.amplitude.shape[0], int(ys.max())+3)
    x0, x1 = max(0, int(xs.min())-2), min(wave.amplitude.shape[1], int(xs.max())+3)
    rho = r[y0:y1, x0:x1]
    t = xp.clip((rho/radius_m-numerics.plateau_fraction)/(1.-numerics.plateau_fraction), 0., 1.)
    window = 1.-t**3*(10.+t*(-15.+6.*t))
    shape = (y1-y0, x1-x0)
    shift = wave.basis_m@np.array((x0+shape[1]//2-wave.amplitude.shape[1]//2,
                                  y0+shape[0]//2-wave.amplitude.shape[0]//2))
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    action = float(.5*shift@q@shift+tilt@shift)
    value = wave.amplitude[y0:y1, x0:x1]*window*np.exp(2j*np.pi*action/wavelength_m)
    result = replace(wave, amplitude=value, origin_m=wave.origin_m+shift,
                     tilt_rad=tilt+.5*(q+q.T)@shift)
    return result, {"schema": SCHEMA, "guard_radius_m": radius_m,
        "input_probability": wave.probability, "window_probability": result.probability,
        "preinterpolation_factor": factor, "origin_shift_m": shift.tolist(),
        "carrier_constant_action_m": action, "output_shape": list(shape),
        "scope": "Numerical receiving-aperture window; discarded input probability is NOT physical absorption or a global L2 error bound"}


def sample_aperture_plane(wave, *, radius_m, wavelength_m, curvature, numerics,
                          grid_numerics, cancelled=lambda: False, retained_bytes=0):
    """Resolve the UNMASKED field first, then put it on a common SI lattice.

    Quadratic carrier differences are evaluated analytically. Cubic complex
    interpolation on the already refined envelope does not fit a phase, shape
    or norm. Its refinement check is made separately from the hard aperture.
    """
    from temsim.physics.wave_device import device_scope
    count = int(np.prod(wave.amplitude.shape))
    with device_scope(grid_numerics.compute_backend,
            acceleration_enabled=grid_numerics.acceleration_enabled,
            maximum_working_bytes=grid_numerics.maximum_device_working_bytes,
            work_items=count, required_bytes=256*count) as device:
        return _sample_aperture_plane(wave, radius_m=radius_m, wavelength_m=wavelength_m,
            curvature=curvature, numerics=numerics, grid_numerics=grid_numerics,
            cancelled=cancelled, device=device, retained_bytes=retained_bytes)


def _sample_aperture_plane(wave, *, radius_m, wavelength_m, curvature, numerics,
                           grid_numerics, cancelled, device, retained_bytes):
    from scipy.ndimage import map_coordinates as cpu_interpolation
    from temsim.physics.multiplane_wave import PlaneWave
    if cancelled():
        raise InterruptedError("Acceptance aperture reconstruction cancelled")
    xp = device.xp
    input_probability = wave.probability
    pitch = radius_m/numerics.aperture_radius_cells
    factor = max(1, math.ceil(float(np.linalg.norm(wave.basis_m, ord=2))/pitch))
    sampling_basis = wave.basis_m/factor
    refined_shape = np.asarray(wave.amplitude.shape)*factor
    half = math.ceil(numerics.output_halfwidth_radii*numerics.aperture_radius_cells)
    count = 2*half
    workspace_shape = tuple(max(count, n) for n in wave.amplitude.shape)
    grid_numerics.check(workspace_shape, retained_bytes=retained_bytes+wave.amplitude.nbytes,
                        working_bytes_per_pixel=384)
    axis = (np.arange(count)-half)*pitch
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    xy = np.stack((xx, yy))
    # Only this neighbourhood can reach the immediately following physical
    # aperture. Retain a guard beyond its edge; the real component clips it.
    inside = xx*xx+yy*yy <= (1.05*radius_m)**2
    positions = xy[:, inside]
    delta = positions-wave.origin_m[:, None]
    indices = np.linalg.inv(sampling_basis)@delta
    indices += refined_shape[::-1, None]//2
    if np.any(indices < 2) or np.any(indices > refined_shape[::-1, None]-3):
        raise ValueError("The executed output field does not cover the receiving aperture")
    sampling_amplitude = xp.asarray(wave.amplitude)
    patch_guard = 0
    if factor > 1:
        from temsim.physics.wave_aperture_sampling import _sample_envelope
        # This is the SAME finite Fourier interpolant as full-period zero
        # padding, evaluated only where the common output needs it. A guard
        # keeps the cubic/quintic spline prefilter's exponentially decaying
        # boundary influence away from the receiving disk. At a physical
        # window edge, preserve the original full-grid boundary instead.
        patch_guard = 32
        lower = np.maximum(0, np.floor(indices.min(axis=1)).astype(int)-patch_guard)
        upper = np.minimum(refined_shape[::-1], np.ceil(indices.max(axis=1)).astype(int)+patch_guard+1)
        patch_shape = tuple(int(n) for n in (upper-lower)[::-1])
        workspace_shape = tuple(max(old, patch) for old, patch in zip(workspace_shape, patch_shape))
        grid_numerics.check(workspace_shape,
            retained_bytes=retained_bytes+2*wave.amplitude.nbytes, working_bytes_per_pixel=384)
        if cancelled():
            raise InterruptedError("Acceptance aperture reconstruction cancelled")
        spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(sampling_amplitude)))/sampling_amplitude.size
        # Align the local patch with original refined cell centres, including
        # odd grids. Do not shift/rephase the physical field's origin/carrier.
        shift = (lower+np.asarray(patch_shape[::-1])//2-refined_shape[::-1]//2)/factor
        sampling_amplitude = _sample_envelope(spectrum, patch_shape, 1./factor, shift, cancelled)
        del spectrum
        indices -= lower[:, None]
    if device.backend == "cupy":
        from cupyx.scipy.ndimage import map_coordinates as interpolate
    else:
        interpolate = cpu_interpolation
    index_array = xp.asarray(indices[::-1])
    values = interpolate(sampling_amplitude, index_array, order=3, mode="constant", cval=0.)
    if cancelled():
        raise InterruptedError("Acceptance aperture reconstruction cancelled")
    alternative = interpolate(sampling_amplitude, index_array, order=5, mode="constant", cval=0.)
    core = np.sum(positions*positions, axis=0) <= radius_m*radius_m
    denominator = float(xp.linalg.norm(values[xp.asarray(core)]))
    interpolation_change = float(xp.linalg.norm((alternative-values)[xp.asarray(core)]))/max(denominator, 1e-300)
    if interpolation_change > numerics.maximum_relative_complex_change*.1:
        raise ValueError("Receiving-aperture complex reconstruction has not converged")
    values = to_host(values)
    q = np.zeros((2, 2)) if wave.curvature_m1 is None else wave.curvature_m1
    tilt = np.zeros(2) if wave.tilt_rad is None else wave.tilt_rad
    phase_slope = 1.05*radius_m*np.linalg.norm(q-curvature, axis=1)+abs(tilt-q@wave.origin_m)
    phase_step = float(2*np.pi/wavelength_m*pitch*np.max(phase_slope))
    if phase_step > .5*np.pi:
        raise ValueError("Receiving-aperture residual carrier exceeds the common-grid sampling budget")
    action = (.5*np.einsum('in,ij,jn->n', delta, q, delta)+tilt@delta
              -.5*np.einsum('in,ij,jn->n', positions, curvature, positions))
    value = np.zeros((count, count), complex)
    value[inside] = values*np.exp(2j*np.pi*action/wavelength_m)*pitch/math.sqrt(abs(np.linalg.det(sampling_basis)))
    if cancelled():
        raise InterruptedError("Acceptance aperture reconstruction cancelled")
    result = PlaneWave(value, np.eye(2)*pitch, np.zeros(2), curvature, np.zeros(2))
    return result, {"pre_aperture_refinement_factor": factor, "aperture_pitch_m": pitch,
        "output_period_m": count*pitch, "output_shape": [count, count],
        "input_unmasked_probability": input_probability,
        "interpolation_domain": "local_finite_fourier_patch" if factor > 1 else "original_input_grid",
        "sampling_patch_shape": list(sampling_amplitude.shape),
        "sampling_patch_spline_guard_cells": patch_guard,
        "virtual_full_refined_shape": refined_shape.tolist(),
        "complex_interpolation_relative_change": interpolation_change,
        "maximum_residual_carrier_step_rad": phase_step,
        "compute_backend": device.evidence(),
        "physical_mask": "pending actual component clip; no aperture taper", "phase_fit": False}


def execute_acceptance_span(state, checkpoint, span, *, numerics, grid_numerics,
                            maximum_step_mm, cancelled=lambda: False, progress_callback=None):
    """Run and qualify all guards/modes before returning one atomic checkpoint."""
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    from temsim.physics.column_wave import _propagate_column, _column_transports, _clip, _electric_coefficients
    from temsim.physics.tip_gun_wave import TipGunCheckpoint
    from temsim.physics.wave_flux import BeamState
    numerics.validate()
    if checkpoint.plane_z_mm != span["start_z_mm"]:
        raise ValueError("Acceptance stage must start at its captured pre-Cs node")
    prepared = span["prepared"]
    plan, radii, stops, owners = prepared
    if float(plan.z_mm[-1]) != span["stop_z_mm"]:
        raise ValueError("Acceptance stage cannot publish inside the conditioned interval")
    nodes, midpoints, _, _ = _electric_coefficients(plan)
    potential_variation = max(float(np.max(abs(nodes-nodes[0]))),float(np.max(abs(midpoints-nodes[0]))))
    # Grounded column solutions retain their large absolute potential baseline;
    # tolerate its floating-point residue, not a physical accelerating field.
    potential_roundoff = 512*np.finfo(float).eps*max(1.,float(np.max(abs(nodes))),float(np.max(abs(midpoints))))
    if potential_variation > potential_roundoff:
        raise AcceptanceWindowUnavailable("Acceptance window currently requires constant axial kinetic energy")
    # Defer only the final physical aperture until its incoming field has been
    # refined. Every intermediate and final physical bore stays in force.
    unmasked = (plan, radii, {i: a for i, a in stops.items() if i != len(plan.z_mm)-1}, owners)
    outputs, records, edge_rows = [], [], []
    transport_cache = {}
    for index, mode in enumerate(checkpoint.beam.modes):
        if cancelled():
            raise InterruptedError("Acceptance stage cancelled before mode")
        wavelength = float(wavelength_m(mode.energy_kev*1000.))
        if mode.energy_kev not in transport_cache:
            transport_cache.clear()
            transport_cache[mode.energy_kev] = tuple(_column_transports(plan, mode.energy_kev))
        paths = transport_cache[mode.energy_kev]
        for path in paths:
            from temsim.physics.radial_column_wave import round_map
            round_map(path.matrix)
            if np.any(path.offset):
                raise AcceptanceWindowUnavailable("A forced path needs ordinary column propagation")
        base = TipGunCheckpoint(BeamState((mode,), checkpoint.beam.reference_plane), checkpoint.plane_z_mm,
            checkpoint.reference_current_a, checkpoint.record)
        fields, rows = [], []
        reference_curvature = None
        actual_input = None
        actual_strength = None
        for scale in numerics.guard_scales:
            retained = sum(x.plane.amplitude.nbytes for x in outputs)+sum(x.amplitude.nbytes for x in fields)
            crop_rows = []
            def before_cs(wave, *, z_mm, wavelength_m, spherical_m3, mode_id):
                nonlocal actual_input, actual_strength
                if z_mm != span["kick_z_mm"] or mode_id != mode.mode_id:
                    raise ValueError("Acceptance crop must consume its exact Cs event once")
                if crop_rows:
                    raise ValueError("Acceptance crop cannot be applied twice")
                root, mapping = stationary_input_radius(wave, paths[span["kick_index"]:],
                    spherical_m3, span["aperture_radius_m"], wavelength_m)
                guard = root*numerics.stationary_guard_factor*scale
                if (2*np.pi/wavelength_m)*abs(spherical_m3)*guard**4/4 < .1:
                    raise AcceptanceWindowUnavailable("No strongly oscillatory Cs event needs this approximation")
                actual_input = replace(wave, amplitude=to_host(wave.amplitude))
                actual_strength = float(spherical_m3)
                cropped, record = crop_acceptance_window(wave, radius_m=guard, wavelength_m=wavelength_m,
                    numerics=numerics, grid_numerics=grid_numerics, cancelled=cancelled,
                    retained_bytes=retained+actual_input.amplitude.nbytes)
                crop_rows.append({**record, "mapping": mapping})
                return cropped
            transported = _propagate_column(state, base, span["stop_z_mm"], maximum_step_mm=maximum_step_mm,
                grid_numerics=grid_numerics, _prepared=unmasked, _before_spherical=before_cs,
                _transport_cache=transport_cache,
                retained_bytes=retained,
                cancelled=cancelled, progress_callback=progress_callback)
            if len(crop_rows) != 1:
                raise ValueError("Acceptance stage did not execute its Cs crop")
            output = transported.beam.modes[0]
            # Restore the actual (unnormalised) cell field relative to the
            # original input mode, rather than comparing unit-normalised shapes.
            scale_weight = output.weight_per_reference_electron/mode.weight_per_reference_electron if mode.weight_per_reference_electron else 0.
            field = replace(output.plane, amplitude=output.plane.amplitude*math.sqrt(scale_weight))
            if reference_curvature is None:
                reference_curvature = np.zeros((2, 2)) if field.curvature_m1 is None else field.curvature_m1
            accepted, reconstruction = sample_aperture_plane(field, radius_m=span["aperture_radius_m"],
                wavelength_m=float(wavelength_m(output.energy_kev*1000.)), curvature=reference_curvature, numerics=numerics,
                grid_numerics=grid_numerics, cancelled=cancelled,
                retained_bytes=retained+actual_input.amplitude.nbytes)
            accepted, physical_losses = _clip(accepted, radii[-1], stops[len(plan.z_mm)-1],
                span["stop_z_mm"], mode.weight_per_reference_electron,
                wavelength_m=float(wavelength_m(output.energy_kev*1000.)), grid_numerics=grid_numerics,
                retained_bytes=retained+actual_input.amplitude.nbytes, cancelled=cancelled)
            # This receiving chart is independently reconstructed on a shared
            # grid for guard comparison. A new local chart here would invalidate
            # that comparison rather than certify its acceptance approximation.
            if any(row.get("sampling", {}).get("status") == "converged" for row in physical_losses):
                raise AcceptanceWindowUnavailable("Receiving aperture reconstruction needs a finer common grid before comparison")
            reconstruction.update(accepted_probability=accepted.probability,
                physical_losses=physical_losses,
                represented_branch_aperture_input_probability=field.probability,
                represented_branch_aperture_loss_probability=field.probability-accepted.probability,
                loss_scope="Only the propagated numerically retained branch; the large pre-Cs numerical discard is not included as aperture absorption")
            fields.append(accepted)
            rows.append({"scale": scale, "crop": crop_rows[0], "reconstruction": reconstruction,
                         "column_execution": transported.record["modes"][0]})
        reference = fields[-1]
        checks = []
        for scale, field in zip(numerics.guard_scales[:-1], fields[:-1]):
            difference = float(np.linalg.norm(field.amplitude-reference.amplitude))
            relative = difference/math.sqrt(reference.probability) if reference.probability else (0. if difference == 0 else math.inf)
            current = abs(field.probability-reference.probability)/reference.probability if reference.probability else 0.
            checks.append({"scale": scale, "relative_complex_change": relative, "relative_current_change": current})
            if relative > numerics.maximum_relative_complex_change or current > numerics.maximum_relative_current_change:
                raise ValueError(f"Receiving-aperture guard convergence failed for {mode.mode_id}: complex={relative:.6g}, current={current:.6g}")
        chosen = fields[numerics.guard_scales.index(1.)]
        norm = chosen.probability
        outputs.append(replace(mode, plane=replace(chosen, amplitude=chosen.amplitude/math.sqrt(norm) if norm else chosen.amplitude),
            weight_per_reference_electron=mode.weight_per_reference_electron*norm,
            axial_reference=output.axial_reference, energy_kev=output.energy_kev))
        from temsim.physics.wave_aperture_edge import estimate_aperture_wall_edges
        edge = estimate_aperture_wall_edges(actual_input, wavelength_m=wavelength,
            spherical_m3=actual_strength, paths=paths[span["kick_index"]:],
            plane_z_mm=plan.z_mm[span["kick_index"]:], bore_radii_mm=radii[span["kick_index"]:],
            aperture_radius_m=span["aperture_radius_m"], accepted_probability=norm)
        edge_rows.append((mode.weight_per_reference_electron, edge))
        records.append({"mode_id": mode.mode_id, "input_weight": mode.weight_per_reference_electron,
            "output_weight": outputs[-1].weight_per_reference_electron,
            "support_approximation": rows[numerics.guard_scales.index(1.)]["column_execution"].get("support_approximation"),
            "guards": rows, "guard_checks": checks, "wall_estimate": edge})
        if progress_callback:
            progress_callback(index+1, len(checkpoint.beam.modes), "Receiving-aperture complex field qualified")
    from temsim.physics.wave_aperture_edge import aggregate_aperture_edge_estimates
    edge_summary = aggregate_aperture_edge_estimates(edge_rows,
        maximum_relative_error=numerics.maximum_empirical_wall_intensity_error)
    if not edge_summary["within_declared_accuracy"]:
        raise ValueError("Receiving-aperture empirical wall estimate exceeds its declared accuracy")
    if cancelled():
        raise InterruptedError("Acceptance stage cancelled before publication")
    return TipGunCheckpoint(BeamState(tuple(outputs), checkpoint.beam.reference_plane), span["stop_z_mm"],
        checkpoint.reference_current_a, {"schema": SCHEMA, "upstream_digest": checkpoint.digest,
            "upstream": checkpoint.record, "span_identity": span["identity"], "numerics": asdict(numerics),
            "start_z_mm": span["start_z_mm"], "stop_z_mm": span["stop_z_mm"],
            "constant_energy_check": {"maximum_potential_change_v": potential_variation,
                "floating_point_tolerance_v": potential_roundoff,
                "coefficients_modified": False},
            "modes": records, "wall_estimate": edge_summary,
            "scope": "Aperture-conditioned numerical approximation. Guard comparison and wall estimate are empirical, not a full hard-wall PDE error bound. No observation inside this span is represented.",
            "validation_status": "DEVELOPMENT_APERTURE_CONDITIONED"})
