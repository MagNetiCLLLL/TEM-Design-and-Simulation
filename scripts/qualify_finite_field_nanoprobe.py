"""Reproducible actual-source Nanoprobe observations; no specimen scattering.

Execute the shared-potential particle path with original weights, clocks and
mechanical losses. Raster drives are disabled on a detached static diagnostic
state. Neither the source distribution nor the apertures are fitted here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np


def configured_state(rays, step, candidate=None):
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    state = default_state()
    catalog = AssemblyCatalog()
    selection = catalog.default_selection()
    catalog.apply(state, selection)
    switch_mode(state, "unified")
    apply_operating_mode_pair(state, "nano_probe", "diffraction",
        column_name=selection.column, recording_name=selection.recording)
    state.electron_gun.emitter.ray_count = int(rays)
    state.step_mm = float(step)
    state.acceleration_enabled = True
    state.acceleration_backend = "Numba CPU"
    for part in (state.ac_deflector, state.descan_deflector):
        part.scan_enabled = False
        if hasattr(part, "wobble_enabled"):
            part.wobble_enabled = False
    if candidate is not None:
        from validate_nanoprobe_condenser import _load_candidate
        _load_candidate(state, Path(candidate))
    return state


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def corrector_planes(state, sigma_margin=8.):
    """Supplementary field references, not crossover-matching acceptance planes.

    Gaussian fields are still evaluated everywhere. Eight standard deviations
    beyond the outer ADL/TL1 Gaussian centres define reproducible observations.
    AC fields overlap the TL1 fringe, so raster drives must remain disabled.
    """
    lenses = {x.key: x for x in state.lenses}
    entrance, exit = lenses['adapter_lens'], lenses['probe_tl12_lens']
    lo = min(entrance.z_mm + entrance.a_mm*(g.offset-sigma_margin*g.sigma) for g in entrance.gaussian)
    hi = max(exit.z_mm + exit.a_mm*(g.offset+sigma_margin*g.sigma) for g in exit.gaussian)
    return (float(lo), float(hi))


def corrector_crossover_regions(state):
    """Use live transfer-lens and physical AC-coil centres, not target waists."""
    from temsim.physics.instrument_magnetic import column_dipole_fields
    lenses = {lens.key: lens for lens in state.lenses
              if lens.enabled and getattr(lens, "installed", True)}
    central = tuple(float(lenses[key].z_mm) for key in ("probe_tl22_lens", "probe_tl21_lens"))
    coils = sorted((coil for coil in column_dipole_fields(state)
                    if coil.key.startswith("ac_deflector:")), key=lambda coil: coil.arrival_z_mm)
    if len(coils) != 2 or any(coil.dynamic for coil in coils):
        raise ValueError("Corrector qualification requires two static physical AC coils")
    ac = tuple(float(coil.arrival_z_mm) for coil in coils)
    if (not np.isfinite((*central, *ac)).all()
            or not state.electron_gun.exit_plane_z_mm <= central[0] < central[1] < ac[0] < ac[1]
            <= state.sample.upper_surface_z_mm):
        raise ValueError("Corrector crossover intervals must be ordered inside the incident column")
    return {name: dict(bounds_mm=list(bounds), maximum_spacing_mm=spacing,
                       geometric_midplane_mm=sum(bounds)/2)
            for name, bounds, spacing in (("central", central, 1.), ("ac", ac, .1))}


def requested_corrector_crossover_planes(regions):
    """Bound observation spacing; the geometric midpoint is only a reference."""
    pieces = []
    for name in ("central", "ac"):
        left, right = map(float, regions[name]["bounds_mm"])
        spacing = float(regions[name]["maximum_spacing_mm"])
        if (not np.isfinite((left, right, spacing)).all() or left >= right
                or not 0 < spacing <= (1. if name == "central" else .1)):
            raise ValueError("Corrector observation intervals require finite bounds and positive spacing")
        count = math.ceil((right-left)/spacing)
        if count > 20000:
            raise ValueError("Corrector observation requests too many checkpoints")
        pieces.append(np.linspace(left, right, count+1))
    return tuple(map(float, np.unique(np.r_[*pieces])))


def evaluate_corrector_crossovers(checkpoints, masks, weights, regions, *, source_current_a,
                                  source_ids=None, angle_relative_tolerance=.01,
                                  require_equal_crossover_angles=False,waist_detector='legacy_point'):
    """Observe two executed 2D waists on original particles and current weights.

    A common cohort must survive the complete central-to-AC observation span.
    Interpolated candidates never replace the three executed planes required
    for each waist. The full actual population must also form a two-dimensional
    waist on those same three planes, with unchanged membership throughout the
    bracket. Angle ratios are diagnostic by default: equal angles are not a
    universal optical invariant. An explicit calibration experiment can require
    both actual-population and common-cohort angles to match. Physical clipping
    and chief-ray displacements remain explicitly reported.
    """
    from temsim.optics.beam_path_audit import crossover_candidates
    from temsim.optics.condenser_topology import _strict_spatial_minima
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.corrector_crossovers import measured_corrector_crossovers
    report = dict(schema="executed-corrector-crossover-observation-v2", status="UNRESOLVED", passed=False,
                  waist_evidence_passed=False, angle_matching_passed=False,
                  require_equal_crossover_angles=require_equal_crossover_angles,
                  failures=[], intervals={}, angle_relative_tolerance=angle_relative_tolerance,
                  angle_matching_role=("Explicit equal-angle calibration requirement" if require_equal_crossover_angles
                                       else "Diagnostic comparison; non-unit angular magnification is not rejected"),
                  scope="Executed sampled 2D waist brackets; independent sampling and step convergence required")

    def reject(reason):
        report["failures"].append(reason)
        return report

    try:
        from temsim.optics.waist_evidence import (FINITE,validate_detector,finite_interval,
            explicit_bracket_indices,support_on_bracket)
        validate_detector(waist_detector)
        report['waist_detector']=waist_detector
        if waist_detector==FINITE:
            report.update(schema='executed-corrector-finite-region-observation-v3',
                scope='Executed finite source-image regions with explicit sides; no exact common stationary plane is asserted')
        requested_corrector_crossover_planes(regions)
        tolerance = float(angle_relative_tolerance)
        current = float(source_current_a)
        z = np.asarray(checkpoints.z_mm, float)
        weight = np.asarray(weights, float)
        alive = np.asarray(masks)
        keys = ("x_m", "y_m", "tx_rad", "ty_rad")
        arrays = [np.asarray(getattr(checkpoints, key), float) for key in keys]
        identities = np.arange(weight.size) if source_ids is None else np.asarray(source_ids)
        if (type(require_equal_crossover_angles) is not bool
                or not np.isfinite((tolerance, current)).all() or not 0 <= tolerance < 1 or current <= 0
                or z.ndim != 1 or z.size < 3 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0)
                or weight.ndim != 1 or np.any(~np.isfinite(weight)) or np.any(weight < 0)
                or not np.isfinite(weight.sum()) or weight.sum() <= 0
                or alive.dtype.kind != "b" or alive.shape != (z.size, weight.size)
                or any(a.shape != alive.shape for a in arrays)
                or identities.shape != weight.shape or identities.dtype.kind not in "iu"
                or np.unique(identities).size != identities.size):
            return reject("invalid_checkpoint_arrays_or_parameters")
        bounds = [regions[key]["bounds_mm"] for key in ("central", "ac")]
        if not bounds[0][1] < bounds[1][0]:
            return reject("unordered_crossover_intervals")
        selected = {}
        for name, region in regions.items():
            left, right = region["bounds_mm"]
            indices = np.flatnonzero((z >= left-1e-9) & (z <= right+1e-9))
            if (indices.size < 5 or any(not np.any(np.isclose(z, edge, rtol=0, atol=1e-9))
                                        for edge in (left, right))
                    or np.max(np.diff(z[indices])) > region["maximum_spacing_mm"]+1e-9):
                return reject(f"insufficient_{name}_checkpoint_coverage")
            selected[name] = indices
        span = np.flatnonzero((z >= bounds[0][0]-1e-9) & (z <= bounds[1][1]+1e-9))
        if np.any((~alive[span[:-1]] & alive[span[1:]]) & (weight > 0)):
            return reject("stopped_particle_reappears")
        if any(np.any(~np.isfinite(a[span][alive[span]])) for a in arrays):
            return reject("nonfinite_surviving_coordinates")
        common = np.all(alive[span], axis=0) & (weight > 0)
        if np.count_nonzero(common) < 5:
            return reject("insufficient_common_positive_current_particles")
        source_weight = float(weight.sum())
        common_weight = float(weight[common].sum())
        report["common_cohort"] = dict(ray_count=int(common.sum()), source_ids=identities[common].tolist(),
            original_weights=weight[common].tolist(), source_weight=common_weight,
            source_current_fraction=common_weight/source_weight, current_a=current*common_weight/source_weight,
            definition="Same original positive-current particles surviving all central-to-AC checkpoints")

        def statistics(index, live):
            return transverse_beam_statistics(*(a[index] for a in arrays), alive=live, weights=weight)

        for name in ("central", "ac"):
            indices, region = selected[name], regions[name]
            left, right = region["bounds_mm"]
            local = SimpleNamespace(z_mm=z[indices], **dict(zip(keys,
                (a[np.ix_(indices, common)] for a in arrays))))
            fixed = np.ones(local.x_m.shape, bool)
            candidates = crossover_candidates(local, fixed, weight[common])
            waists = _strict_spatial_minima(local, weight[common],
                measured_corrector_crossovers(local, fixed, weight[common], left, right))
            legacy_waists=waists
            finite_evidence=None
            if waist_detector==FINITE:
                finite_evidence=finite_interval(checkpoints,np.broadcast_to(common,alive.shape),weight,
                    left,right,maximum_spacing_mm=region['maximum_spacing_mm'],source_ids=identities)
                waists=finite_evidence['regions']
                if finite_evidence['status']=='UNRESOLVED':
                    report['failures'].append(f'unresolved_{name}_finite_region_evidence')
            rows = []
            for waist in waists:
                index = int(np.searchsorted(z, waist["z_mm"]))
                actual, matched = statistics(index, alive[index]), statistics(index, common)
                bracket = []
                bracket_planes = (waist["bracket_mm"][0], waist["z_mm"], waist["bracket_mm"][1])
                bracket_indices = np.searchsorted(z, bracket_planes)
                support_indices=np.arange(bracket_indices[0],bracket_indices[-1]+1) if waist_detector==FINITE else bracket_indices
                actual_masks = alive[support_indices] & (weight > 0)
                actual_support = dict(status="UNRESOLVED", executed_waists=[],
                    positive_current_membership_stable=bool(np.all(actual_masks == actual_masks[0])))
                if not actual_support["positive_current_membership_stable"]:
                    report["failures"].append(f"unresolved_{name}_actual_waist_cohort_changed")
                else:
                    actual_cohort = actual_masks[0]
                    if waist_detector==FINITE:
                        actual_support=support_on_bracket(checkpoints,alive,weight,waist,
                            maximum_spacing_mm=region['maximum_spacing_mm'],source_ids=identities)
                        actual_waists=actual_support['executed_waists']
                        supported=actual_support['status']=='PASS'
                    else:
                        actual_cp = SimpleNamespace(z_mm=z[bracket_indices], **dict(zip(keys,
                            (a[np.ix_(bracket_indices, actual_cohort)] for a in arrays))))
                        actual_fixed = np.ones(actual_cp.x_m.shape, bool)
                        actual_waists = _strict_spatial_minima(actual_cp, weight[actual_cohort],
                            measured_corrector_crossovers(actual_cp, actual_fixed, weight[actual_cohort], left, right))
                        supported = len(actual_waists) == 1 and actual_waists[0]["z_mm"] == waist["z_mm"]
                    actual_support.update(status="PASS" if supported else "FAIL", executed_waists=actual_waists,
                        source_ids=identities[actual_cohort].tolist(), original_weights=weight[actual_cohort].tolist())
                    if not supported:
                        report["failures"].append(f"missing_{name}_actual_2d_waist_at_common_bracket")
                for plane in bracket_planes:
                    j = int(np.searchsorted(z, plane))
                    bracket.append(dict(z_mm=float(z[j]), actual=asdict(statistics(j, alive[j])),
                        common=asdict(statistics(j, common)),
                        current_a=current*float(weight[alive[j]].sum())/source_weight))
                rows.append(dict(waist, actual=asdict(actual), common=asdict(matched),
                    actual_source_weight=float(weight[alive[index]].sum()),
                    current_a=current*float(weight[alive[index]].sum())/source_weight,
                    common_current_a=current*common_weight/source_weight,
                    source_current_fraction=common_weight/source_weight,
                    geometric_midplane_offset_mm=waist["z_mm"]-region["geometric_midplane_mm"],
                    executed_bracket=bracket, actual_waist_support=actual_support))
            report["intervals"][name] = dict(region, checkpoint_count=int(indices.size),
                actual_maximum_spacing_mm=float(np.max(np.diff(z[indices]))),
                interpolated_candidates_diagnostic_only=candidates, executed_waists=rows,
                legacy_point_executed_waists=legacy_waists if waist_detector==FINITE else None,
                finite_region_evidence=finite_evidence)
            if len(rows) != 1:
                report["failures"].append(f"missing_or_ambiguous_executed_{name}_2d_waist")
        if report["failures"]:
            return report
        central, ac = (report["intervals"][key]["executed_waists"][0] for key in ("central", "ac"))
        angle_findings = []
        for population in ("actual", "common"):
            angles = [row[population]["convergence_95_rad"] for row in (central, ac)]
            if not np.isfinite(angles).all() or min(angles) <= 0:
                return reject(f"invalid_{population}_waist_angles")
            ratio = angles[1]/angles[0]
            report[population] = dict(alpha95_mrad=[x*1e3 for x in angles], ac_over_central_ratio=ratio,
                                      relative_error=abs(ratio-1), passed=abs(ratio-1) <= tolerance)
            if not report[population]["passed"]:
                angle_findings.append(f"{population}_crossover_angles_do_not_match")
        report.update(waist_evidence_passed=True, angle_matching_passed=not angle_findings,
                      diagnostic_findings=angle_findings)
        if require_equal_crossover_angles:
            report["failures"].extend(angle_findings)
        report.update(status="FAIL" if report["failures"] else "PASS", passed=not report["failures"])
        return report
    except (AttributeError, KeyError, TypeError, ValueError, FloatingPointError) as error:
        report["error"] = str(error)
        return reject("unresolved_corrector_crossover_evidence")


def validated_diameter_limit_nm(value):
    """Validate a declared strict size bound before any expensive observation."""
    try:
        limit = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Probe diameter limit must be finite and positive") from error
    if isinstance(value, bool) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("Probe diameter limit must be finite and positive")
    return limit


def qualification_failures(topology, measurement, corrector, target_mrad, *,
                           require_equal_crossover_angles=False, maximum_diameter95_nm=.7,
                           crossover_chain=None):
    """Apply declared topology, focus, angle, strict size and crossover gates."""
    diameter_limit = validated_diameter_limit_nm(maximum_diameter95_nm)
    failures = []
    if (corrector.get('waist_detector')=='finite_region'
            and not (crossover_chain or {}).get('required_topology_satisfied_at_sampling',False)):
        failures.append('Required source-image regions failed or have incomplete whole-capture evidence')
    if (corrector.get('waist_detector')=='finite_region'
            and (crossover_chain or {}).get('unattributed_population_change_count',0)):
        failures.append('Unattributed physical interception remains unresolved in the incident capture')
    if not topology["passed"]:
        failures.append("Condenser topology failed or is unresolved")
    if corrector.get('waist_detector') == 'finite_region':
        # An unresolved upstream condenser interval can omit its convenience
        # direct-focus summary. Only the whole-capture finite proof is binding.
        direct_focus = (crossover_chain or {}).get('direct_focus')
        direct_focus = direct_focus if isinstance(direct_focus, dict) else {}
        direct_focus_passed = (
            direct_focus.get('schema') == 'direct-focus-finite-region-topology-v2'
            and direct_focus.get('requirement') == 'direct_focus_without_extra_pre_specimen_waist'
            and direct_focus.get('status') == 'PASS'
            and direct_focus.get('passed') is True
            and direct_focus.get('failures') == [])
    else:
        direct_focus_passed = topology.get("direct_focus", {}).get("status") == "PASS"
    if not direct_focus_passed:
        failures.append("Direct Mini Condenser/objective focus topology failed or is unresolved")
    if not measurement.accepts(target_mrad):
        failures.append("Surface focus, convergence angle or effective sample support failed")
    diameter = 2*measurement.statistics.radius_95_m*1e9
    if not math.isfinite(diameter) or not 0 <= diameter < diameter_limit:
        failures.append("Surface D95 must be strictly below the declared diameter limit")
    if not corrector.get("waist_evidence_passed", False):
        failures.append("Executed internal and AC crossover evidence failed or is unresolved")
    elif require_equal_crossover_angles and not corrector.get("angle_matching_passed", False):
        failures.append("Explicit equal-angle crossover calibration constraint failed")
    return failures


def validated_extra_capture_planes(state, values):
    """Normalize explicit physical observation coordinates without moving them."""
    try:
        values = tuple(values)
    except TypeError as error:
        raise ValueError("Extra capture planes must be finite physical coordinates") from error
    if any(isinstance(value, (bool, np.bool_))
           or not isinstance(value, (int, float, np.integer, np.floating))
           or not np.isfinite(value) for value in values):
        raise ValueError("Extra capture planes must be finite physical coordinates")
    lower, upper = float(state.electron_gun.exit_plane_z_mm), float(state.sample.upper_surface_z_mm)
    if any(not lower <= value <= upper for value in values):
        raise ValueError("Extra capture planes must lie between gun exit and specimen upper surface")
    return tuple(sorted(set(map(float, values))))


def observe(state, *, spacing_mm=1., target_mrad=30., corrector_angle_tolerance=.01,
            require_equal_crossover_angles=False, maximum_diameter95_nm=.7,
            extra_capture_planes_mm=(), output=None):
    from temsim.calculation_manifest import solver_source_identity
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.optics.beam_path_audit import incident_checkpoints
    from temsim.optics import condenser_topology
    from temsim.optics.surface_probe_focus import local_waist_from_radii, SurfaceFocusMeasurement
    from temsim.physics.beam_current import effective_source_current_a, column_current_limit_percent
    from temsim.physics.beam_statistics import transverse_beam_statistics
    from temsim.physics.particle_sections import _digest_arrays
    from temsim.optics.crossover_acceptance import regions_for_state,evaluate_crossover_chain
    from temsim.optics.waist_evidence import detector_for_state,observed_aperture_losses
    if not np.isfinite(corrector_angle_tolerance) or not 0 <= corrector_angle_tolerance < 1:
        raise ValueError("Corrector angle tolerance must be finite and between zero and one")
    if type(require_equal_crossover_angles) is not bool:
        raise ValueError("The optional equal-angle calibration requirement must be boolean")
    maximum_diameter95_nm = validated_diameter_limit_nm(maximum_diameter95_nm)
    extra_capture_planes_mm = validated_extra_capture_planes(state, extra_capture_planes_mm)
    start = perf_counter()
    snapshot = capture_instrument_snapshot(state)
    topology_planes = condenser_topology.requested_capture_planes(state, maximum_spacing_mm=spacing_mm)
    corrector_regions = corrector_crossover_regions(state)
    crossover_planes = requested_corrector_crossover_planes(corrector_regions)
    end = float(state.sample.upper_surface_z_mm)
    focus_planes = (end - .0002, end - .0001, end)
    component_planes = tuple(float(x.z_mm) for x in state.lenses
                             if x.enabled and state.electron_gun.exit_plane_z_mm <= x.z_mm <= end)
    lenses = {x.key:x for x in state.lenses}
    adl, tl1 = lenses['adapter_lens'], lenses['probe_tl12_lens']
    mechanical_ports = (adl.mechanical_center_from_tip_mm-adl.mechanical_length_mm/2,
                        tl1.mechanical_center_from_tip_mm+tl1.mechanical_length_mm/2)
    optical_ports = corrector_planes(state)
    detector=detector_for_state(state)
    chain_regions=regions_for_state(state)
    chain_spacing=min(float(spacing_mm),.5)
    chain_bounds=sorted({p for row in chain_regions.values() for p in row['bounds_mm']})
    chain_planes=np.unique(np.r_[*[np.linspace(a,b,int(np.ceil((b-a)/chain_spacing))+1)
        for a,b in zip(chain_bounds[:-1],chain_bounds[1:])]])
    planes = np.unique(np.r_[state.electron_gun.exit_plane_z_mm, topology_planes,
                             crossover_planes, component_planes, focus_planes, mechanical_ports, optical_ports,
                             chain_planes, extra_capture_planes_mm])
    print(f"Executing {state.electron_gun.emitter.ray_count} emitted particles, step {state.step_mm:g} mm", flush=True)
    gun, cp, masks = incident_checkpoints(state, planes, step_mm=state.step_mm)
    emitted = gun.exit_bundle
    weights = np.asarray(emitted.weight)
    source_weight = float(weights.sum())
    source_current = effective_source_current_a(state)
    corrector = evaluate_corrector_crossovers(cp, masks, weights, corrector_regions,
        source_current_a=source_current, source_ids=np.asarray(emitted.ray_id),
        angle_relative_tolerance=corrector_angle_tolerance,
        require_equal_crossover_angles=require_equal_crossover_angles,waist_detector=detector)
    port_reports = {}
    for name, ports in [('mechanical_boundaries', mechanical_ports), ('static_eight_sigma_reference', optical_ports)]:
        rows = np.searchsorted(cp.z_mm, ports)
        common = masks[rows[0]] & masks[rows[1]]
        actual = [transverse_beam_statistics(*(getattr(cp, key)[i] for key in ('x_m','y_m','tx_rad','ty_rad')),
            alive=masks[i], weights=weights) for i in rows]
        matched = [transverse_beam_statistics(*(getattr(cp, key)[i] for key in ('x_m','y_m','tx_rad','ty_rad')),
            alive=common, weights=weights) for i in rows]
        port_reports[name] = dict(role="Supplementary reference only; not crossover angle matching",
            z_mm=ports, actual=[asdict(row) for row in actual],
            common_survivors=[asdict(row) for row in matched],
            current_a=[float(source_current*weights[masks[i]].sum()/source_weight) for i in rows],
            alpha95_ratio=actual[1].convergence_95_mrad/actual[0].convergence_95_mrad,
            common_alpha95_ratio=matched[1].convergence_95_mrad/matched[0].convergence_95_mrad)
    topology = condenser_topology.evaluate(state, cp, masks, weights, maximum_spacing_mm=spacing_mm,
        source_ids=np.asarray(emitted.ray_id))
    topology_history = [topology]
    for attempt in range(2):
        extra = topology["refinement_planes_mm"]
        if topology["passed"] or not extra:
            break
        topology_planes = tuple(sorted(set(topology_planes).union(extra, extra_capture_planes_mm)))
        _, topo_cp, topo_masks = incident_checkpoints(state, topology_planes, step_mm=state.step_mm)
        topology = condenser_topology.evaluate(state, topo_cp, topo_masks, weights, maximum_spacing_mm=spacing_mm,
            source_ids=np.asarray(emitted.ray_id))
        topology_history.append(topology)
    surface_mask = masks[-1]
    indices = np.searchsorted(cp.z_mm, focus_planes)
    stats = [transverse_beam_statistics(*(getattr(cp, name)[i] for name in ("x_m", "y_m", "tx_rad", "ty_rad")),
        alive=surface_mask, weights=weights) for i in indices]
    radii = tuple(row.radius_rms_m for row in stats)
    offset, second = local_waist_from_radii(radii, 1e-7)
    transmitted = weights[surface_mask]
    effective = float(transmitted.sum()**2 / np.sum(transmitted**2)) if transmitted.size else 0.
    source_identity = _digest_arrays(*(np.asarray(getattr(emitted, key)) for key in
        ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "energy_offset_ev", "weight", "alive", "ray_id")))
    measurement = SurfaceFocusMeasurement(end, stats[-1], offset, second, radii,
        float(state.step_mm), effective, source_identity,
        tuple(map(int, np.asarray(emitted.ray_id)[surface_mask])), float(transmitted.sum()))
    envelope = []
    for i, z in enumerate(cp.z_mm):
        row = transverse_beam_statistics(*(getattr(cp, name)[i] for name in ("x_m", "y_m", "tx_rad", "ty_rad")),
            alive=masks[i], weights=weights)
        envelope.append(dict(z_mm=float(z), rms_radius_nm=row.radius_rms_m*1e9,
            alpha95_mrad=row.convergence_95_mrad, waist_offset_mm=row.waist_offset_m*1e3,
            current_a=source_current*weights[masks[i]].sum()/source_weight,
            mean_x_nm=row.mean_x_m*1e9, mean_y_nm=row.mean_y_m*1e9,
            ray_count=int(masks[i].sum())))
    chain=evaluate_crossover_chain(cp,masks,weights,np.asarray(emitted.ray_id),
        source_current_a=source_current,regions=chain_regions,maximum_spacing_mm=chain_spacing,
        source_identity=source_identity,waist_detector=detector,
        known_interceptions=observed_aperture_losses(state,cp,masks,weights,np.asarray(emitted.ray_id)))
    failures = qualification_failures(topology, measurement, corrector, target_mrad,
        require_equal_crossover_angles=require_equal_crossover_angles,
        maximum_diameter95_nm=maximum_diameter95_nm,crossover_chain=chain)
    result = dict(schema="unified-nanoprobe-qualification-v5", utc=datetime.now(timezone.utc).isoformat(),
        elapsed_s=perf_counter()-start, simulation_mode=state.simulation_mode,
        snapshot_digest=snapshot.digest, solver_identity=solver_source_identity(),
        rays=int(state.electron_gun.emitter.ray_count), step_mm=float(state.step_mm),
        target_mrad=target_mrad, maximum_checkpoint_spacing_mm=spacing_mm,
        extra_capture_planes_mm=list(extra_capture_planes_mm),
        requirements=dict(executed_corrector_waists=True,
            equal_crossover_angles=require_equal_crossover_angles,
            angle_matching_role=corrector["angle_matching_role"],
            fixed_surface_size_threshold=True,
            maximum_diameter95_nm=maximum_diameter95_nm, size_comparison="strictly_less_than"),
        source=dict(emission_mean_ev=state.electron_gun.emitter.emission_energy_ev,
            energy_spread_fwhm_ev=state.electron_gun.emitter.energy_spread_fwhm_ev,
            emitted_current_a=state.electron_gun.emitted_current_a,
            column_current_limit_percent=column_current_limit_percent(state),
            total_weight=source_weight, gun_exit_weight=float(weights[np.asarray(emitted.alive)].sum())),
        controls={x.key: float(x.percent) for x in state.lenses if x.enabled},
        topology=topology, topology_history=topology_history, surface=asdict(measurement),
        corrector=corrector,crossover_chain=chain, supplementary_corrector_ports=port_reports,
        surface_size=dict(rms_radius_nm=stats[-1].radius_rms_m*1e9,
            current_diameter_95_nm=2*stats[-1].radius_95_m*1e9,
            maximum_diameter95_nm=maximum_diameter95_nm,
            comparison="strictly_less_than",
            passed=bool(np.isfinite(stats[-1].radius_95_m) and 0 <= 2*stats[-1].radius_95_m*1e9 < maximum_diameter95_nm),
            role="Declared strict specimen-current D95 bound; numerical convergence remains separate"),
        sample_current_a=float(source_current*transmitted.sum()/source_weight),
        envelope=envelope, failures=failures, fixed_sampling_pass=not failures,
        qualification_scope='Declared required source-image intervals and specimen metrics; full inventory resolution is reported separately and no minimum transmission is imposed',
        inventory_completely_resolved=chain['status']=='OBSERVED_AT_BOUNDED_SAMPLING',
        scope="Actual source and finite fields at the stated sampling; independent count and step convergence required")
    if capture_instrument_snapshot(state).digest != snapshot.digest:
        raise ValueError("Observation changed physical inputs")
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(clean(result), indent=2, allow_nan=False)+"\n", encoding="utf-8")
        np.savez_compressed(path.with_suffix(".npz"), z_mm=cp.z_mm, x_m=cp.x_m, y_m=cp.y_m,
            tx_rad=cp.tx_rad, ty_rad=cp.ty_rad, flight_time_s=cp.flight_time_s,
            kinetic_energy_ev=cp.kinetic_energy_ev, alive=masks, weight=weights,
            source_id=np.asarray(emitted.ray_id), source_energy_offset_ev=np.asarray(emitted.energy_offset_ev))
        path.with_suffix(".inputs.json").write_text(json.dumps(clean(state.to_dict()), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(clean({k: result[k] for k in ("rays", "step_mm", "elapsed_s", "sample_current_a", "failures")})), flush=True)
    print(json.dumps(clean({"topology":topology, "surface":result["surface"]})), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rays", type=int, default=1024,
        help="Original source particles; new physical trials require at least 1000")
    parser.add_argument("--step", type=float, default=.05)
    parser.add_argument("--target", type=float, default=30.,
        help="Legacy alpha95 target in mrad, NOT the physical pupil edge; use measure_optical_acceptance.py for edge acceptance")
    parser.add_argument("--spacing", type=float, default=1.)
    parser.add_argument("--extra-capture-z-mm", type=float, action="append", default=[],
        help="Additional exact physical checkpoint in mm; repeat for multiple coordinates")
    parser.add_argument("--max-d95-nm", type=float, default=.7,
        help="Strict specimen-current D95 upper bound in nm; default requires D95 < 0.7 nm")
    parser.add_argument("--corrector-angle-tolerance", type=float, default=.01,
                        help="Relative alpha95 tolerance between executed internal and AC crossovers")
    parser.add_argument("--require-equal-crossover-angles", action="store_true",
                        help="Opt into an equal-angle calibration experiment; angle ratios are diagnostic by default")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rays < 16 or not np.isfinite((args.step, args.target, args.spacing)).all() or min(args.step, args.target, args.spacing) <= 0:
        parser.error("Use at least 16 rays and finite positive steps, target and spacing")
    if not np.isfinite(args.corrector_angle_tolerance) or not 0 <= args.corrector_angle_tolerance < 1:
        parser.error("Corrector angle tolerance must be finite and between zero and one")
    try:
        maximum_diameter95_nm = validated_diameter_limit_nm(args.max_d95_nm)
    except ValueError as error:
        parser.error(str(error))
    observe(configured_state(args.rays, args.step, args.candidate), spacing_mm=args.spacing,
            target_mrad=args.target, corrector_angle_tolerance=args.corrector_angle_tolerance,
            require_equal_crossover_angles=args.require_equal_crossover_angles,
            maximum_diameter95_nm=maximum_diameter95_nm,
            extra_capture_planes_mm=args.extra_capture_z_mm, output=args.output)


if __name__ == "__main__":
    main()
