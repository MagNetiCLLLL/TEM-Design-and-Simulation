"""Executed incident-column waists and same-particle angular relationships.

These are finite-current observations, not a substitute for nominal pupil
boundary propagation, a canonical transfer map, or a calibrated scan pivot.
No input state, particle weight, or optical parameter is changed here.
"""
from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from temsim.optics.beam_path_audit import crossover_candidates
from temsim.optics.condenser_topology import _strict_spatial_minima
from temsim.physics.beam_statistics import transverse_beam_statistics
from temsim.physics.corrector_crossovers import measured_corrector_crossovers


_COORDINATES = ("x_m", "y_m", "tx_rad", "ty_rad")


def regions_for_state(state):
    """Physical interval labels; no historical waist coordinate is imposed."""
    from temsim.component_keys import MINI_CONDENSER
    from temsim.physics.instrument_magnetic import column_dipole_fields
    lenses = {part.key: part for part in state.lenses
              if part.enabled and getattr(part, "installed", True)}
    names = ("condenser_lens_1", "condenser_lens_2", "condenser_lens_3",
             "adapter_lens", "probe_tl22_lens", "probe_tl21_lens", MINI_CONDENSER)
    try:
        c1, c2, c3, adl, tl22, tl21, mini = [float(lenses[key].z_mm) for key in names]
    except KeyError as error:
        raise ValueError("Crossover chain requires the C3 + Probe Corrector assembly") from error
    coils = sorted((part for part in column_dipole_fields(state)
                    if part.key.startswith("ac_deflector:")), key=lambda part: part.arrival_z_mm)
    if len(coils) != 2 or any(part.dynamic for part in coils):
        raise ValueError("Crossover chain requires two static physical AC coils")
    ac0, ac1 = [float(part.arrival_z_mm) for part in coils]
    start, end = float(state.electron_gun.exit_plane_z_mm), float(state.sample.upper_surface_z_mm)
    boundaries = np.asarray([start, c1, c2, c3, adl, tl22, tl21, ac0, ac1, mini, end])
    if np.any(~np.isfinite(boundaries)) or np.any(np.diff(boundaries) <= 0):
        raise ValueError("Crossover chain intervals must be ordered in the incident column")
    labels = ("gun_exit_to_c1", "c1_to_c2", "c2_to_c3", "c3_to_adl", "adl_to_tl22",
              "corrector_internal", "tl21_to_ac", "ac_pair", "ac_to_mini", "mini_to_specimen")
    from temsim.operating_modes import validate_nanoprobe_range
    large_angle = validate_nanoprobe_range(getattr(state, "nanoprobe_range", "normal")) == "large_angle"
    regions = {key: dict(bounds_mm=[float(left), float(right)], requirement=(
        "required_2d_waist" if key == "c2_to_c3" or (key == "c3_to_adl" and large_angle) else
        "forbidden_converging_to_diverging_waist" if key == "c3_to_adl" else "record_all_waists"))
        for key, left, right in zip(labels, boundaries[:-1], boundaries[1:])}
    from temsim.optics.direct_focus_topology import (required_for_state, bounds_for_state,
        REQUIREMENT, SURFACE_FOCUS_TOLERANCE_NM)
    if required_for_state(state):
        for key in ('corrector_internal','ac_pair'):
            regions[key]['requirement']='exactly_one_2d_waist'
        direct_start, direct_end = bounds_for_state(state)
        regions["mini_to_specimen"].update(requirement=REQUIREMENT,
            direct_focus_bounds_mm=[direct_start, direct_end],
            surface_focus_tolerance_nm=SURFACE_FOCUS_TOLERANCE_NM)
    return regions


def _statistics(cp, index, cohort, weights, current):
    if not np.any(cohort):
        return dict(status="NO_POSITIVE_CURRENT", current_a=0., source_current_fraction=0.)
    stats = transverse_beam_statistics(*(getattr(cp, key)[index] for key in _COORDINATES),
                                      alive=cohort, weights=weights)
    return dict(status="MEASURED", z_mm=float(cp.z_mm[index]), statistics=asdict(stats),
                alpha95_mrad=stats.convergence_95_mrad, alpha99_mrad=stats.convergence_99_mrad,
                sampled_max_angle_mrad=1e3*stats.convergence_edge_rad,
                nominal_pupil_edge_mrad=None, rms_radius_nm=1e9*stats.radius_rms_m,
                current_a=current*stats.surviving_fraction,
                source_current_fraction=stats.surviving_fraction)


def _strict_waists(cp, cohort, weights, selected):
    if selected.size < 3 or np.count_nonzero(cohort) < 5:
        return []
    local = SimpleNamespace(z_mm=cp.z_mm[selected], **{
        key: getattr(cp, key)[np.ix_(selected, cohort)] for key in _COORDINATES})
    fixed = np.ones(local.x_m.shape, bool)
    margin = max(1., abs(float(local.z_mm[0])), abs(float(local.z_mm[-1])))*1e-10
    measured = measured_corrector_crossovers(local, fixed, weights[cohort],
        float(local.z_mm[0])-margin, float(local.z_mm[-1])+margin)
    return _strict_spatial_minima(local, weights[cohort], measured)


def _phase_fit(cp, first, last, cohort, weights):
    """Empirical affine fit, explicitly refusing a rank-deficient source."""
    report = dict(status="UNDERDETERMINED", coordinate_order=["x_m", "y_m", "tx_rad", "ty_rad"],
        role="Empirical weighted mechanical-slope fit of this finite population; not a differential or canonical map",
        energy_scope="Executed source energy mixture; no monoenergetic transfer map is inferred",
        matrix=None, intercept=None, rank=0, scaled_condition=None)
    if np.count_nonzero(cohort) < 5:
        return report
    w = weights[cohort]/weights[cohort].sum()
    first_values = np.stack([getattr(cp, key)[first, cohort] for key in _COORDINATES], axis=1)
    last_values = np.stack([getattr(cp, key)[last, cohort] for key in _COORDINATES], axis=1)
    first_mean, last_mean = w@first_values, w@last_values
    x, y = first_values-first_mean, last_values-last_mean
    scale = np.sqrt(w@(x*x))
    if np.any(scale <= np.finfo(float).tiny):
        return report
    design = (x/scale)*np.sqrt(w)[:, None]
    singular = np.linalg.svd(design, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0]*1e-8))
    condition = float(singular[0]/singular[-1]) if singular[-1] > 0 else None
    report.update(rank=rank, scaled_condition=condition)
    if rank < 4 or condition is None or condition > 1e8:
        return report
    coefficients = np.linalg.lstsq(design, y*np.sqrt(w)[:, None], rcond=1e-8)[0]/scale[:, None]
    residual = y-x@coefficients
    report.update(status="EMPIRICAL_FULL_RANK", matrix=coefficients.T.tolist(),
                  intercept=(last_mean-first_mean@coefficients).tolist(),
                  weighted_residual_rms=np.sqrt(w@(residual*residual)).tolist())
    return report


def evaluate_crossover_chain(checkpoints, masks, weights, source_ids, *, source_current_a,
                             regions, maximum_spacing_mm=.5, source_identity=None,
                             waist_detector='legacy_point', known_interceptions=()):
    """Inspect every sampled incident interval without erasing pre-loss waists.

    All positive-current particles present on a stable three-plane bracket are
    used for actual waist detection. A separate, unchanged-ID cohort surviving
    the entire capture supplies angular comparisons. Interception transitions
    cannot manufacture waists, nor can downstream losses erase an earlier one.
    The caller must supply executed checkpoints, not interpolated plot history.
    """
    from temsim.optics.waist_evidence import (FINITE, validate_detector, finite_interval,
        explicit_bracket_indices, support_on_bracket)
    validate_detector(waist_detector)
    finite_mode = waist_detector == FINITE
    z = np.asarray(checkpoints.z_mm, float)
    arrays = [np.asarray(getattr(checkpoints, key), float) for key in _COORDINATES]
    live, weight, identities = np.asarray(masks), np.asarray(weights, float), np.asarray(source_ids)
    current, spacing = float(source_current_a), float(maximum_spacing_mm)
    if (z.ndim != 1 or z.size < 3 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0)
            or weight.ndim != 1 or np.any(~np.isfinite(weight)) or np.any(weight < 0)
            or not np.isfinite(weight.sum()) or weight.sum() <= 0
            or live.dtype.kind != "b" or live.shape != (z.size, weight.size)
            or any(a.shape != live.shape for a in arrays)
            or identities.shape != weight.shape or identities.dtype.kind not in "iu"
            or np.unique(identities).size != identities.size
            or not np.isfinite([current, spacing]).all() or current <= 0 or spacing <= 0):
        raise ValueError("Invalid crossover chain arrays, source identities, current, or spacing")
    positive = weight > 0
    live = live & positive[None, :]
    if np.any(~live[:-1] & live[1:]):
        raise ValueError("A stopped positive-current ray cannot reappear")
    if any(np.any(~np.isfinite(a[live])) for a in arrays):
        raise ValueError("Nonfinite coordinates of a current-carrying particle")
    cp = SimpleNamespace(z_mm=z, **dict(zip(_COORDINATES, arrays)))
    bounds = sorted((float(row["bounds_mm"][0]), float(row["bounds_mm"][1]), key)
                    for key, row in regions.items())
    if (not bounds or any(not np.isfinite([a, b]).all() or a >= b for a, b, _ in bounds)
            or any(abs(left[1]-right[0]) > 1e-9 for left, right in zip(bounds, bounds[1:]))):
        raise ValueError("Crossover regions must form finite adjacent intervals")
    expected_start, expected_end = bounds[0][0], bounds[-1][1]
    complete = bool(abs(z[0]-expected_start) <= 1e-9 and abs(z[-1]-expected_end) <= 1e-9)
    gap_ok = float(np.max(np.diff(z))) <= spacing+1e-9
    common = np.all(live, axis=0)
    changes = np.flatnonzero(np.any(live[1:] != live[:-1], axis=1))+1
    segments = np.r_[0, changes, len(z)]
    actual_waists = []
    for low, high in zip(segments[:-1], segments[1:]):
        actual_waists.extend(_strict_waists(cp, live[low], weight, np.arange(low, high)))
    legacy_waists=actual_waists
    finite_reports={}
    if finite_mode:
        actual_waists=[]
        for left,right,key in bounds:
            evidence=finite_interval(cp,live,weight,left,right,maximum_spacing_mm=spacing,
                source_ids=identities,known_interceptions=known_interceptions)
            finite_reports[key]=evidence
            actual_waists.extend(evidence['regions'])
    candidates = crossover_candidates(cp, live, weight)
    unresolved = []
    for candidate in candidates:
        if not any(row["bracket_mm"][0]-1e-9 <= candidate["z_mm"] <= row["bracket_mm"][1]+1e-9
                   for row in actual_waists):
            unresolved.append(candidate)
    intervals = {}
    direct_focus = None
    for left, right, key in bounds:
        rows = [row for row in actual_waists if left < row["z_mm"] < right]
        roots = [row for row in candidates if left < row["z_mm"] < right]
        selected = z[(z >= left) & (z <= right)]
        # Include straddling checkpoints in the spacing test; an unobserved
        # interval boundary must not be mistaken for complete axial coverage.
        cover = z[max(0, np.searchsorted(z, left)-1):min(len(z), np.searchsorted(z, right)+1)]
        covered = complete and cover.size >= 3 and cover[0] <= left and cover[-1] >= right
        covered = covered and np.max(np.diff(cover)) <= spacing+1e-9
        if finite_mode:
            covered = covered and finite_reports[key]['status']=='RESOLVED_AT_BOUNDED_SAMPLING'
        requirement = regions[key].get("requirement", "record_all_waists")
        status = "OBSERVED_AT_BOUNDED_SAMPLING" if covered else "UNRESOLVED_COVERAGE"
        if covered and requirement == "required_2d_waist":
            status = "PASS_AT_BOUNDED_SAMPLING" if rows else "FAIL_MISSING_EXECUTED_2D_WAIST"
        elif covered and requirement == 'exactly_one_2d_waist':
            status = 'PASS_AT_BOUNDED_SAMPLING' if len(rows)==1 else 'FAIL_MISSING_OR_AMBIGUOUS_2D_WAIST'
        elif covered and requirement == "forbidden_converging_to_diverging_waist":
            status = "FAIL_FORBIDDEN_WAIST" if roots else "PASS_AT_BOUNDED_SAMPLING"
        if requirement == "direct_focus_without_extra_pre_specimen_waist":
            from temsim.optics.direct_focus_topology import evaluate as evaluate_direct_focus
            direct_bounds = regions[key]["direct_focus_bounds_mm"]
            direct_focus = evaluate_direct_focus(cp, live, weight,
                lower_z_mm=direct_bounds[0], surface_z_mm=direct_bounds[1],
                surface_focus_tolerance_nm=regions[key]["surface_focus_tolerance_nm"],
                maximum_spacing_mm=min(spacing, .5),waist_detector=waist_detector,source_ids=identities)
            status = ("PASS_AT_BOUNDED_SAMPLING" if direct_focus["passed"] else
                      "FAIL_FORBIDDEN_WAIST" if direct_focus["status"] == "FAIL" else
                      "UNRESOLVED_DIRECT_FOCUS")
        intervals[key] = dict(bounds_mm=[left, right], requirement=requirement, status=status,
                              checkpoint_count=int(selected.size), waist_z_mm=[row["z_mm"] for row in rows])
        if finite_mode:
            intervals[key]['finite_region_evidence']=finite_reports[key]
    nodes = []
    for waist in actual_waists:
        index = int(np.searchsorted(z, waist["z_mm"]))
        if finite_mode:
            sides,inside=explicit_bracket_indices(cp,waist)
            common_report=support_on_bracket(cp,np.broadcast_to(common,live.shape),weight,waist,
                maximum_spacing_mm=spacing,source_ids=identities)
            common_waist=common_report['status']=='PASS'
        else:
            sides=np.array([index-1,index,index+1])
            common_waist = _strict_waists(cp, common, weight, np.arange(index-1, index+2))
        labels = [key for left, right, key in bounds if left < waist["z_mm"] < right]
        node = dict(waist, region=labels[0] if labels else "component_boundary", index=index,
                    actual=_statistics(cp, index, live[index], weight, current),
                    common=_statistics(cp, index, common, weight, current),
                    actual_source_ids=identities[live[index]].tolist(),
                    common_2d_waist_at_same_bracket=bool(common_waist))
        node.update(source_current_fraction=node["actual"]["source_current_fraction"],
                    cohort="All positive-current particles present throughout the stable local bracket")
        node["actual_three_planes"] = [_statistics(cp, i, live[i], weight, current) for i in sides]
        if finite_mode:
            node['actual_bracket_planes']=node['actual_three_planes']
            node['common_finite_region_support']=common_report
            node['explicit_bracket_checkpoint_count']=len(inside)
        nodes.append(node)
    specimen = dict(region="specimen_entrance", z_mm=float(z[-1]), index=len(z)-1,
                    status="ENDPOINT_OBSERVATION_NOT_TWO_SIDED_WAIST",
                    actual=_statistics(cp, len(z)-1, live[-1], weight, current),
                    common=_statistics(cp, len(z)-1, common, weight, current))
    relationships = []
    for first, last in zip(nodes, nodes[1:]+[specimen]):
        relation = dict(from_region=first["region"], to_region=last["region"],
                        from_z_mm=first["z_mm"], to_z_mm=last["z_mm"],
                        interpretation="Observed finite-current angular magnification; no equal-angle requirement",
                        empirical_phase_fit=_phase_fit(cp, first["index"], last["index"], common, weight))
        for label in ("actual", "common"):
            a, b = first[label], last[label]
            ratios = {}
            for quantity in ("alpha95_mrad", "alpha99_mrad", "sampled_max_angle_mrad"):
                numerator, denominator = b.get(quantity), a.get(quantity)
                ratios[quantity.replace("_mrad", "_ratio")] = (
                    numerator/denominator if numerator is not None and denominator is not None and denominator > 0 else None)
            relation[label] = ratios
        relationships.append(relation)
    resolved = (complete and gap_ok and not unresolved and not len(changes)
                and (direct_focus is None or direct_focus["status"] != "UNRESOLVED"))
    if finite_mode:
        resolved = resolved and all(row['status']=='RESOLVED_AT_BOUNDED_SAMPLING' for row in finite_reports.values())
    required_ok=bool((complete and gap_ok if finite_mode else resolved) and all(
        row['status']=='PASS_AT_BOUNDED_SAMPLING' for row in intervals.values()
        if row['requirement']!='record_all_waists'))
    loss_evidence=[]
    if finite_mode:
        for i in changes:
            bracket=[float(z[i-1]),float(z[i])]
            loss_evidence.append(next((row for row in known_interceptions
                if row['bracket_mm']==bracket and row.get('status')=='CAPTURED_LOSS_SUPPORTED_BY_OPENING_GEOMETRY'),
                dict(bracket_mm=bracket,status='UNATTRIBUTED_CAPTURED_LOSS')))
    return dict(schema="incident-finite-region-chain-v2" if finite_mode else "incident-crossover-chain-v1", source_population_identity=source_identity,
                waist_detector=waist_detector,
                status="OBSERVED_AT_BOUNDED_SAMPLING" if resolved else "UNRESOLVED",
                coverage=dict(start_mm=float(z[0]), end_mm=float(z[-1]), checkpoint_count=len(z),
                              maximum_gap_mm=float(np.max(np.diff(z))), requested_maximum_gap_mm=spacing,
                              complete_incident_interval=complete, spacing_satisfied=gap_ok,
                              population_change_intervals_mm=[[float(z[i-1]), float(z[i])] for i in changes]),
                intervals=intervals, executed_waists=nodes, specimen=specimen, angular_relationships=relationships,
                direct_focus=direct_focus,
                # Keep observed pre-loss waists above, but an unresolved chain
                # cannot qualify topology through a separate optimistic flag.
                required_topology_satisfied_at_sampling=required_ok,
                qualification_scope=('Required physical regions only; full inventory can remain unresolved after recorded interception. No minimum transmission is imposed.' if finite_mode else 'Legacy whole-inventory resolution'),
                population_change_evidence=loss_evidence,
                unattributed_population_change_count=sum(row['status']=='UNATTRIBUTED_CAPTURED_LOSS' for row in loss_evidence),
                legacy_point_executed_waists=legacy_waists if finite_mode else None,
                common_cohort=dict(source_ids=identities[common].tolist(), original_weights=weight[common].tolist(),
                                   current_a=current*float(weight[common].sum()/weight.sum())),
                interpolated_candidates_diagnostic_only=candidates,
                unresolved_candidates=unresolved,
                scope=dict(gun_internal="NOT_MEASURED_EXACT_PLANE_BRACKETS_REQUIRED",
                           post_specimen="NOT_MEASURED_OUTSIDE_INCIDENT_CAPTURE",
                           scan_pivot="NOT_MEASURED_STATIC_SOURCE_WAIST_IS_NOT_DRIVEN_PIVOT",
                           relay_conjugacy="NOT_QUALIFIED_EMPIRICAL_PARTICLE_FIT_IS_NOT_A_DIFFERENTIAL_MAP",
                           nominal_pupil_edge="SEPARATE_BOUNDARY_RAY_MEASUREMENT_REQUIRED",
                           particle_count_and_step_convergence="NOT_ESTABLISHED_BY_ONE_CAPTURE"))
