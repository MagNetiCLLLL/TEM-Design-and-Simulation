"""Independent, pure-array audit of the frozen ideal magnetic-lens study.

No source is transported, no field solver is invoked, and no raw trial is
modified. Statistical and Gaussian-series formulas are implemented here rather
than imported from the producer. Run after every expected trial has files.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
PLAN_SHA = "da3221b5f6553a873509db53b57d4aa1fc8496fdb5aab13fe25a9af7d6bc45da"


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def bit_equal(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def inverse_digits(ids, base):
    # Independent per-ID base expansion verifies the recorded sampling law.
    result = []
    for item in ids:
        integer, fraction, scale = int(item), 0., 1./base
        while integer:
            integer, digit = divmod(integer, base)
            fraction += digit*scale
            scale /= base
        result.append(fraction)
    return np.asarray(result)


def quantile(values, weights, fraction=.95):
    order = np.argsort(values)
    w = np.asarray(weights)[order]
    target = math.nextafter(fraction*math.fsum(w), -math.inf)
    low, high = 0, len(w)-1
    while low < high:
        middle = (low+high)//2
        if math.fsum(w[:middle+1]) >= target:
            high = middle
        else:
            low = middle+1
    return float(np.asarray(values)[order[low]])


def statistics(phase, alive, weights, current):
    alive = np.asarray(alive, bool) & (np.asarray(weights) > 0)
    if not np.any(alive):
        return dict(status="NO_TRANSMISSION", surviving_positive=0, effective=0., current_a=0.)
    p, absolute = np.asarray(phase)[alive], np.asarray(weights)[alive]
    total = float(np.sum(absolute))
    w = absolute/total
    centroid = np.sum(w[:, None]*p[:, :2], axis=0)
    radius = np.linalg.norm(p[:, :2]-centroid, axis=1)
    directions = np.column_stack((p[:, 2:4], np.ones(len(p))))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    chief = np.sum(w[:, None]*directions, axis=0)
    chief /= np.linalg.norm(chief)
    angles = np.arctan2(np.linalg.norm(np.cross(directions, chief), axis=1),
        np.clip(directions@chief, -1., 1.))
    return dict(status="MEASURED", surviving_positive=int(alive.sum()),
        effective=total**2/float(np.sum(absolute**2)), current_a=current*total,
        transmission=total, d95_nm=2e9*quantile(radius, w),
        rms_radius_nm=1e9*math.sqrt(float(np.sum(w*radius**2))),
        alpha95_mrad=1e3*quantile(angles, w),
        angular_rms_mrad=1e3*math.sqrt(float(np.sum(w*angles**2))),
        centroid_m=centroid.tolist())


def agreement(measured, reported):
    differences = {}
    good = True
    for key, value in measured.items():
        if key in ("status", "surviving_positive"):
            passed = value == reported.get(key)
            if not passed:
                differences[key] = dict(measured=value, reported=reported.get(key))
        else:
            a, b = np.asarray(value), np.asarray(reported.get(key, np.nan))
            delta = float(np.max(np.abs(a-b)))
            # Relative tolerance only covers independent summation rounding.
            passed = bool(np.allclose(a, b, rtol=5e-12, atol=1e-22))
            if not passed:
                differences[key] = dict(maximum_absolute_difference=delta,
                    measured=value, reported=reported.get(key))
        good &= passed
    return bool(good), differences


def gaussian_series(peak, sigma, z, radius, order):
    """B_r, B_z and curl_phi from A_phi through r**order, independently."""
    x = np.asarray(z)/sigma
    r = np.asarray(radius)
    hermite = [np.ones_like(x), x]
    for n in range(2, order+2):
        hermite.append(x*hermite[-1]-(n-1)*hermite[-2])
    envelope = peak*np.exp(-.5*x*x)
    derivative = [(-1.)**n*hermite[n]*envelope/sigma**n for n in range(order+2)]
    br, bz = np.zeros(np.broadcast(x, r).shape), np.zeros(np.broadcast(x, r).shape)
    for k in range((order+1)//2):
        c = (-1.)**k/(2.**(2*k+1)*math.factorial(k)*math.factorial(k+1))
        br -= c*r**(2*k+1)*derivative[2*k+1]
        bz += (2*k+2)*c*r**(2*k)*derivative[2*k]
    # All lower powers cancel in dB_r/dz-dB_z/dr.
    curl = -c*r**order*derivative[order+1]
    return br, bz, curl


def field_checks(report, request, phase, z, physical_alive):
    peak, sigma = request["peak_t"], request["sigma_mm"]*1e-3
    evidence = report["field_order_evidence"]
    nr, nz = evidence["grid_shape"]
    zz, rr = np.meshgrid(np.linspace(-8*sigma, 8*sigma, nz),
        np.linspace(0., evidence["tested_radius_m"], nr))
    values = {order: gaussian_series(peak, sigma, zz, rr, order) for order in (3, 5, 7, 9)}
    rows = []
    for order, (br, bz, curl) in values.items():
        difference = float(np.max(np.hypot(br-values[9][0], bz-values[9][1]))/peak)
        normalized_curl = float(sigma*np.max(np.abs(curl))/peak)
        recorded = next(item for item in evidence["orders"] if item["order"] == order)
        rows.append(dict(order=order, relative_B_difference_to_order9=difference,
            normalized_curl_residual=normalized_curl,
            reported_values_agree=bool(np.isclose(difference, recorded["relative_B_difference_to_order9"],
                rtol=1e-8, atol=1e-15) and np.isclose(normalized_curl,
                    recorded["normalized_curl_residual"], rtol=1e-8, atol=1e-15))))
    # Also inspect actual saved physical points, including Gaussian tails.
    path_r = np.linalg.norm(phase[:, :, :2], axis=2)
    path_z = np.broadcast_to(z[:, None], path_r.shape)[physical_alive]
    path_r = path_r[physical_alive]
    b7 = gaussian_series(peak, sigma, path_z, path_r, 7)
    b9 = gaussian_series(peak, sigma, path_z, path_r, 9)
    path_difference = float(np.max(np.hypot(b7[0]-b9[0], b7[1]-b9[1]))/peak)
    path_curl = float(sigma*np.max(np.abs(b9[2]))/peak)
    envelope_m = report["dense_characteristic_radius_upper_bound_mm"]*1e-3
    gate = rows[2]["relative_B_difference_to_order9"] <= 1e-4 and rows[3]["normalized_curl_residual"] <= 1e-3
    return dict(independent_formula="Aphi Gaussian Hermite series; analytical curl residual",
        identical_report_grid=rows, sampled_field_gate=bool(gate),
        reported_gate_agrees=gate == evidence["sampled_field_gates_passed"],
        tested_radius_covers_dense_counterfactual_bound=evidence["tested_radius_m"] >= envelope_m*(1-1e-14),
        saved_physical_path_points=int(path_r.size),
        saved_physical_path_max_radius_m=float(path_r.max()),
        saved_physical_path_order7_vs9_B_over_peak=path_difference,
        saved_physical_path_sigma_curl_order9_over_peak=path_curl,
        saved_physical_path_gate=path_difference <= 1e-4 and path_curl <= 1e-3,
        scope="Finite cylindrical grid plus retained physical observation points; not a rigorous continuous Maxwell bound")


def independent_fits(initial, image_phase, training, outward, edge, reported):
    """Refit all predeclared vacuum rays using an independent polar gradient."""
    pupil = initial[:, :2]@outward.T
    normalized = pupil/edge
    radius = np.linalg.norm(normalized, axis=1)
    phi = np.arctan2(normalized[:, 1], normalized[:, 0])
    columns, powers = [], []
    for power, harmonic in ((2, 0), (2, 2), (3, 1), (3, 3), (4, 0),
                            (4, 2), (4, 4), (6, 0), (6, 6)):
        r = radius**(power-1)
        c, s, cm, sm = np.cos(phi), np.sin(phi), np.cos(harmonic*phi), np.sin(harmonic*phi)
        columns.append(np.column_stack((r*(c*cm+harmonic/power*s*sm),
                                         r*(s*cm-harmonic/power*c*sm))))
        powers.append(power)
        if harmonic:
            columns.append(np.column_stack((r*(c*sm-harmonic/power*s*cm),
                                             r*(s*sm+harmonic/power*c*cm))))
            powers.append(power)
    gradients = np.stack(columns, axis=-1)
    design = np.concatenate((gradients[training, 0], gradients[training, 1]))
    target = -image_phase[training, :2].T.ravel()*edge
    coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=1e-10)
    coefficients_mm = coefficients/edge**np.asarray(powers)*1e3
    prediction = -(gradients[~training]@coefficients)/edge
    residual = prediction-image_phase[~training, :2]
    radial = -np.sum(image_phase[:, :2]*pupil/np.linalg.norm(pupil, axis=1)[:, None], axis=1)
    q = np.linalg.norm(pupil, axis=1)/edge
    geometric = []
    for orders in ((1, 3, 5), (1, 3, 5, 7)):
        matrix = np.column_stack([q**order for order in orders])
        c, _, radial_rank, _ = np.linalg.lstsq(matrix[training], radial[training], rcond=1e-12)
        original = next(row for row in reported["radial_geometric_fits"] if row["powers"] == list(orders))
        cs = float(c[1]/edge**3*1e3)
        geometric.append(dict(powers=list(orders), cs_mm=cs,
            rank=int(radial_rank), reported_cs_agrees=bool(np.isclose(cs, original["cs_mm"], rtol=1e-10, atol=1e-12))))
    cs = float(coefficients_mm[7])
    holdout = float(np.sqrt(np.mean(residual**2)))
    return dict(cartesian_cs_mm=cs, cartesian_rank=int(rank),
        cartesian_reported_cs_agrees=bool(np.isclose(cs, reported["cs_fit_mm"], rtol=1e-10, atol=1e-12)),
        cartesian_holdout_rms_m=holdout,
        cartesian_holdout_report_agrees=bool(np.isclose(holdout, reported["holdout_rms_m"], rtol=1e-8, atol=1e-18)),
        radial_geometric=geometric,
        fit_population="All retained original vacuum rays in declared training mask; no wall mask is applied")


def audit_case(directory, plan_sha, reference):
    manifest_path = directory/"files.json"
    manifest = load_json(manifest_path)
    hashes = {name: file_hash(directory/name) for name in manifest}
    hash_ok = hashes == manifest
    if not hash_ok:
        raise ValueError(f"Raw file hash mismatch: {directory.name}")
    report, request = load_json(directory/"report.json"), load_json(directory/"request.json")
    source = load_npz(directory/"source.npz")
    parts = [load_npz(directory/f"characteristics-{part}.npz") for part in (0, 1)]
    z = parts[0]["z_m"]
    phase = np.concatenate([part["phase"] for part in parts], axis=1)
    ids = source["source_id"]
    checks = {"raw_manifest_exact": hash_ok,
        "frozen_plan_hash": request["study_plan_sha256"] == plan_sha == PLAN_SHA,
        "embedded_request_exact": report["request"] == request,
        "source_schema": request["source_schema"] == "ideal-uniform-disk-halton235-v1",
        "original_ids_unique_0_to_2047": bit_equal(ids, np.arange(2048, dtype=ids.dtype)),
        "source_phase_finite": bool(np.isfinite(source["initial_phase"]).all()),
        "all_original_weights_positive": bool(np.all(source["combined_population_weights"] > 0)),
        "combined_weight_exact_1_over_2048": bool(np.all(source["combined_population_weights"] == 1/2048)),
        "mono_300keV": bool(np.all(source["kinetic_energy_ev"] == 300000.)),
        "emission_time_zero": bool(np.all(source["emission_time_s"] == 0.)),
        "initial_clock_matches_emission_time": bit_equal(source["initial_phase"][:, 4], source["emission_time_s"]),
        "all_phase_values_finite": bool(np.isfinite(phase).all()),
        "observations_equal_between_partitions": bit_equal(z, parts[1]["z_m"]),
        "reported_observations_exact": bit_equal(z, np.asarray(report["observations"]["z_m"])),
        "partition_shape": phase.shape == (len(z), 2048, 5),
        "paraxial_plane_exact": z[report["observations"]["image_index"]] == report["paraxial_image_z_m"],
        "source_interval_exact": z[0] == request["start_m"] and z[-1] == request["end_m"],
        "independent_initial_phase_ids_unique": np.unique(source["initial_phase"][:, :2], axis=0).shape[0] == 2048,
    }
    ref_case = next(case for case in reference["cases"]
        if case["peak_t"] == request["peak_t"] and case["sigma_mm"] == request["sigma_mm"])
    pupil = ref_case["variants"]["refined8"]["pupil"]
    ref_cs = ref_case["variants"]["refined8"]["coefficients"]["cs_mm"]
    checks["intrinsic_cs_matches_independent_reference"] = report["intrinsic_cs_mm"] == ref_cs
    checks["reference_input_hash_exact"] = request["reference_sha256"] == file_hash(HERE/"variational-reference/report.json")
    matrix = np.asarray(pupil["initial_mechanical_slopes_per_position_m1"])
    radius = request["source_radius_mm"]*1e-3
    radial = radius*np.sqrt(inverse_digits(ids+1, 2))
    phi = 2*np.pi*inverse_digits(ids+1, 3)
    expected_xy = radial[:, None]*np.column_stack((np.cos(phi), np.sin(phi)))
    directions = np.column_stack((source["initial_phase"][:, 2:4], np.ones(2048)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    checks.update(halton_disk_exact=bit_equal(expected_xy, source["initial_phase"][:, :2]),
        source_boundary_respected=bool(np.all(np.linalg.norm(expected_xy, axis=1) < radius)),
        paraxial_zero_larmor_slopes_exact=bit_equal(source["initial_phase"][:, :2]@matrix.T,
            source["initial_phase"][:, 2:4]),
        emission_direction_exact=bit_equal(directions, source["emission_direction"]),
        halton5_training_exact=bit_equal(inverse_digits(ids+1, 5) < .5, source["training"]))
    for part, raw in enumerate(parts):
        sl = slice(1024*part, 1024*(part+1))
        receipt = report["partitions"][part]
        checks.update({f"part{part}_source_ids_exact": bit_equal(ids[sl], raw["source_id"]),
            f"part{part}_initial_phase_exact": bit_equal(raw["phase"][0], source["initial_phase"][sl]),
            f"part{part}_energy_exact": bit_equal(raw["kinetic_energy_ev"], source["kinetic_energy_ev"][sl]),
            f"part{part}_historical_weights_exact": bool(np.all(raw["historical_population_weights"] == (1/1024 if part == 0 else 1/2048))),
            f"part{part}_phase_hash_exact": array_hash(raw["phase"]) == receipt["retained_phase_sha256"],
            f"part{part}_initial_hash_exact": array_hash(source["initial_phase"][sl]) == receipt["initial_phase_sha256"],
            f"part{part}_clock_monotone": bool(np.all(np.diff(raw["phase"][:, :, 4], axis=0) > 0)),
            f"part{part}_integration_interval_exact": raw["integration_z_m"][0] == z[0] and raw["integration_z_m"][-1] == z[-1],
            f"part{part}_new_original_count": receipt["newly_transported_positive_count"] == 1024})
    image_index = report["observations"]["image_index"]
    image = z[image_index]
    wall_arrays = {}
    physical_alive = np.zeros((len(z), 2048), dtype=bool)
    wall_records = []
    for diameter in request["tube_diameters_mm"]:
        walls = [load_npz(directory/f"wall-{diameter:g}mm-part{part}.npz") for part in (0, 1)]
        combined = {key: np.concatenate([wall[key] for wall in walls]) for key in walls[0]}
        wall_arrays[diameter] = combined
        hit = np.where(np.isnan(combined["hit_z_m"]), np.inf, combined["hit_z_m"])
        alive = z[:, None] < hit[None, :]
        physical_alive |= alive
        loss = np.isfinite(hit)
        local = dict(ids_exact=bit_equal(combined["source_ids"], ids),
            no_unresolved_contacts=not bool(combined["unresolved_mask"].any()),
            loss_mask_equals_finite_contact=bool(np.array_equal(loss, combined["physical_loss_mask"])),
            end_alive_matches_contact=bool(np.array_equal(alive[-1], combined["alive_at_end"])),
            no_contact_before_source=bool(np.all(hit >= z[0])),
            contact_not_after_transport=bool(np.all(hit[loss] <= z[-1])),
            all_contact_phase_finite=bool(np.isfinite(combined["hit_phase"][loss]).all()),
            no_loss_restored=bool(np.all(np.diff(alive.astype(np.int8), axis=0) <= 0)))
        checks.update({f"tube{diameter:g}_{key}": value for key, value in local.items()})
        wall_records.append(dict(tube_diameter_mm=diameter, **local,
            contact_count=int(loss.sum()),
            max_wall_contact_radial_residual_m=float(np.max(np.abs(combined["hit_radial_residual_m"][loss]))) if loss.any() else None))
    population_records, fit_records, stats_by_count = [], [], {}
    for population in report["populations"]:
        count = population["original_positive_count"]
        weights = np.full(count, 1/count)
        checks[f"n{count}_source_hash_exact"] = population["source_phase_hash"] == array_hash(source["initial_phase"][:count])
        fit = population["vacuum_fit"]
        refit = independent_fits(source["initial_phase"][:count], phase[image_index, :count],
            source["training"][:count], np.asarray(pupil["outward_pupil_slopes_per_initial_position_m1"]),
            report["paraxial_edge_slope_mrad"]*1e-3, fit)
        checks[f"n{count}_independent_complete_vacuum_fit"] = (refit["cartesian_rank"] == 15 and
            refit["cartesian_reported_cs_agrees"] and refit["cartesian_holdout_report_agrees"] and
            all(item["reported_cs_agrees"] for item in refit["radial_geometric"]))
        cart_relative = abs(fit["cs_fit_mm"]/report["intrinsic_cs_mm"]-1)
        radial7 = next(row for row in fit["radial_geometric_fits"] if row["powers"] == [1, 3, 5, 7])
        radial_relative = abs(radial7["cs_mm"]/report["intrinsic_cs_mm"]-1)
        train_count = int(source["training"][:count].sum())
        checks[f"n{count}_training_count_exact"] = fit["training_count"] == train_count and fit["holdout_count"] == count-train_count
        checks[f"n{count}_fit_scope_complete_vacuum"] = fit["population"] == "Complete declared counterfactual vacuum source; no intercepted-ray deletion"
        checks[f"n{count}_cartesian_gate_not_relabelled"] = fit["cs_reference_agreement"] == (cart_relative <= .001)
        fit_records.append(dict(original_positive_count=count, training_count=train_count,
            holdout_count=count-train_count, cs_reference_mm=report["intrinsic_cs_mm"],
            cartesian_cs_mm=fit["cs_fit_mm"], cartesian_relative_difference=cart_relative,
            cartesian_gate_passed=cart_relative <= .001,
            radial7_cs_mm=radial7["cs_mm"], radial7_relative_difference=radial_relative,
            radial7_gate_passed=radial_relative <= .001,
            independent_full_vacuum_refit=refit,
            gate_scope="Complete counterfactual vacuum pupil; physical tube eligibility is listed per view"))
        views, stats_by_count[count] = [], {}
        for view in population["tube_views"]:
            diameter = view["tube_diameter_mm"]
            wall = wall_arrays[diameter]
            hit = np.where(np.isnan(wall["hit_z_m"][:count]), np.inf, wall["hit_z_m"][:count])
            alive = image < hit
            measured = statistics(phase[image_index, :count], alive, weights, request["source_current_a"])
            good, delta = agreement(measured, view["at_paraxial_image"])
            blocked = ~alive
            loss_current = request["source_current_a"]*float(weights[blocked].sum())
            balance = measured["current_a"]+loss_current-request["source_current_a"]
            checks[f"n{count}_tube{diameter:g}_primary_statistics"] = good
            checks[f"n{count}_tube{diameter:g}_absolute_current_ledger"] = abs(balance) <= 1e-18 and loss_current == view["intercepted_current_a"] and int(blocked.sum()) == view["intercepted_before_image"]
            checks[f"n{count}_tube{diameter:g}_no_clipped_fit"] = view["sphere_fit_from_clipped_population"] is False and view["intrinsic_cs_reference_mm"] == report["intrinsic_cs_mm"]
            checks[f"n{count}_tube{diameter:g}_full_pupil_eligibility"] = view["full_declared_pupil_physically_admitted"] == (not blocked.any())
            # Native-plane checks remain observations, not a refocus or a new gate.
            focus_stats = [statistics(phase[index, :count], z[index] < hit, weights, request["source_current_a"])
                for index in report["observations"]["focus_indices"]]
            native = min(range(len(focus_stats)), key=lambda item: focus_stats[item].get("d95_nm", math.inf))
            native_z = float(z[report["observations"]["focus_indices"][native]])
            native_rms = min(range(len(focus_stats)), key=lambda item: focus_stats[item].get("rms_radius_nm", math.inf))
            checks[f"n{count}_tube{diameter:g}_native_d95_minimum"] = native_z == view["native_d95_minimum"]["z_m"]
            status = wall["status"][:count]
            max_radius_id = None
            if alive.any():
                centroid = np.asarray(measured["centroid_m"])
                radii = np.linalg.norm(phase[image_index, :count, :2]-centroid, axis=1)
                max_radius_id = int(np.flatnonzero(alive)[np.argmax(radii[alive])])
            views.append(dict(tube_diameter_mm=diameter, at_first_paraxial_image=measured,
                reported_statistics_agree=good, disagreement=delta,
                intercepted_before_image=int(blocked.sum()), intercepted_current_a=loss_current,
                absolute_current_balance_residual_a=balance,
                entered_outside_count=int(np.count_nonzero(status == "outside_at_entrance")),
                downstream_wall_contact_count=int(np.count_nonzero(status == "wall_contact")),
                intrinsic_cs_qualification="ELIGIBLE" if not blocked.any() else "INELIGIBLE_CENSORED_PUPIL",
                cartesian_cs_gate="PASS" if not blocked.any() and cart_relative <= .001 else
                    ("FAIL" if not blocked.any() else "INELIGIBLE_CENSORED_PUPIL"),
                radial7_cs_gate="PASS" if not blocked.any() and radial_relative <= .001 else
                    ("FAIL" if not blocked.any() else "INELIGIBLE_CENSORED_PUPIL"),
                native_observed_d95_minimum_z_m=native_z,
                native_observed_d95_minimum_nm=focus_stats[native].get("d95_nm"),
                native_observed_rms_minimum_z_m=float(z[report["observations"]["focus_indices"][native_rms]]),
                native_observed_rms_minimum_nm=focus_stats[native_rms].get("rms_radius_nm"),
                maximum_centred_radius_source_id=max_radius_id))
            stats_by_count[count][diameter] = measured
        population_records.append(dict(original_positive_count=count,
            positive_source_packet_count=int(np.count_nonzero(weights > 0)),
            total_original_current_a=request["source_current_a"], tube_views=views))
    count_checks = []
    for check in report["source_count_comparison"]["tube_checks"]:
        diameter = check["tube_diameter_mm"]
        relative = stats_by_count[2048][diameter]["d95_nm"]/stats_by_count[1024][diameter]["d95_nm"]-1
        gate = abs(relative) <= .01
        checks[f"tube{diameter:g}_source_d95_gate_truthful"] = gate == check["d95_count_gate_passed"] and np.isclose(relative, check["relative_d95_change"], atol=1e-14, rtol=1e-12)
        count_checks.append(dict(tube_diameter_mm=diameter, relative_d95_2048_over_1024_minus1=relative,
            d95_count_gate_passed=gate))
    relative_cs = fit_records[1]["cartesian_cs_mm"]/fit_records[0]["cartesian_cs_mm"]-1
    checks["source_cs_count_gate_truthful"] = (abs(relative_cs) <= .001) == report["source_count_comparison"]["cs_fit_count_gate_passed"]
    field = field_checks(report, request, phase, z, physical_alive)
    checks["independent_field_grid_values_agree"] = all(row["reported_values_agree"] for row in field["identical_report_grid"])
    checks["independent_field_gate_agrees"] = field["reported_gate_agrees"]
    checks["field_envelope_coverage"] = field["tested_radius_covers_dense_counterfactual_bound"]
    record = dict(name=directory.name, request=request, raw_file_sha256=hashes,
        manifest_sha256=file_hash(manifest_path), audit_integrity_passed=all(checks.values()),
        checks=checks, failed_integrity_checks=[key for key, value in checks.items() if not value],
        source_prefix=dict(original=1024, appended=1024, total=2048,
            first_partition_is_identical_array_in_both_population_views=True,
            complete_prefix_fields=["ID", "position", "slope/direction", "energy", "emission_time", "flight_clock"],
            immutable_historical_weights="part0 1/1024; combined population 1/2048; intercepted current not restored",
            provenance_limit="Current manifests and shared retained arrays establish internal identity; they do not independently prove historical file chronology"),
        walls=wall_records, populations=population_records, complete_vacuum_fits=fit_records,
        source_count_checks=count_checks, source_cs_relative_2048_over_1024_minus1=relative_cs,
        source_cs_count_gate_passed=abs(relative_cs) <= .001, field=field)
    edge_slope_mrad = report["paraxial_edge_slope_mrad"]
    edge_angle_mrad = 1e3*math.atan(edge_slope_mrad*1e-3)
    maximum_alpha95 = max(view["at_paraxial_image"].get("alpha95_mrad", 0.)
        for population in report["populations"] for view in population["tube_views"])
    record["historical_benchmark_scope"] = dict(benchmark_maximum_declared_pupil_mrad=100.,
        declared_paraxial_edge_slope_mrad=edge_slope_mrad,
        declared_paraxial_edge_angle_mrad=edge_angle_mrad,
        maximum_observed_current_alpha95_mrad=maximum_alpha95,
        declared_pupil_exceeds_100mrad=edge_slope_mrad > 100.,
        observed_alpha95_exceeds_100mrad=maximum_alpha95 > 100.,
        interpretation="New exploratory extension; no inherited historical <=100 mrad qualification"
            if edge_slope_mrad > 100. or maximum_alpha95 > 100. else
            "Within the historical numerical pupil range; does not inherit qualification without the declared study gates")
    cache = dict(source=source, phase=phase, z=z, walls=wall_arrays, report=report,
        request=request, stats=stats_by_count)
    return record, cache


def compare_refinement(name, record, cache, parent_record, parent):
    source_equal = {key: bit_equal(value, parent["source"][key]) for key, value in cache["source"].items()}
    planes_equal = bit_equal(cache["z"], parent["z"])
    checks = dict(all_initial_source_arrays_bit_exact=all(source_equal.values()),
        observation_planes_bit_exact=planes_equal)
    report, reference = cache["report"], parent["report"]
    comparisons = []
    if planes_equal:
        indices = {"first_paraxial_image": report["observations"]["image_index"], "transport_end": -1}
        for label, index in indices.items():
            delta = np.linalg.norm(cache["phase"][index, :, :2]-parent["phase"][index, :, :2], axis=1)
            row = dict(plane=label, z_m=float(cache["z"][index]),
                maximum_paired_all_original_xy_difference_nm=float(delta.max()*1e9),
                maximum_difference_source_id=int(np.argmax(delta)),
                all_original_paired_position_gate_1nm=bool(delta.max() <= 1e-9), physical_views=[])
            for diameter, wall in cache["walls"].items():
                hit = np.where(np.isnan(wall["hit_z_m"]), np.inf, wall["hit_z_m"])
                other = parent["walls"][diameter]["hit_z_m"]
                other = np.where(np.isnan(other), np.inf, other)
                alive, old_alive = cache["z"][index] < hit, cache["z"][index] < other
                common = alive & old_alive
                row["physical_views"].append(dict(tube_diameter_mm=diameter,
                    alive_membership_bit_exact=bit_equal(alive, old_alive),
                    changed_membership_count=int(np.count_nonzero(alive != old_alive)),
                    common_alive_count=int(common.sum()),
                    maximum_paired_common_alive_xy_difference_nm=float(delta[common].max()*1e9) if common.any() else None))
            comparisons.append(row)
    d95_checks = []
    for diameter in cache["stats"][2048]:
        relative = cache["stats"][2048][diameter]["d95_nm"]/parent["stats"][2048][diameter]["d95_nm"]-1
        d95_checks.append(dict(tube_diameter_mm=diameter,
            relative_primary_d95_refined_over_parent_minus1=relative,
            step_d95_gate_passed=abs(relative) <= .001 if record["request"]["refinement"] == "half-step" else None))
    cs_changes = []
    for fit, old_fit in zip(record["complete_vacuum_fits"], parent_record["complete_vacuum_fits"]):
        cs_changes.append(dict(original_positive_count=fit["original_positive_count"],
            cartesian_cs_relative_refined_over_parent_minus1=fit["cartesian_cs_mm"]/old_fit["cartesian_cs_mm"]-1,
            radial7_cs_relative_refined_over_parent_minus1=fit["radial7_cs_mm"]/old_fit["radial7_cs_mm"]-1,
            cartesian_reference_gate_passed=fit["cartesian_gate_passed"],
            radial7_reference_gate_passed=fit["radial7_gate_passed"]))
    half_step = record["request"]["refinement"] == "half-step"
    return dict(name=name, parent=record["request"]["refinement_parent"],
        refinement=record["request"]["refinement"], source_array_bit_equality=source_equal,
        compatibility=checks, paired_plane_comparisons=comparisons, d95_comparisons=d95_checks,
        cs_fit_comparisons=cs_changes,
        step_qualification_passed=(all(checks.values()) and bool(comparisons) and
            all(row["all_original_paired_position_gate_1nm"] for row in comparisons) and
            all(row["step_d95_gate_passed"] for row in d95_checks)) if half_step else None,
        scope="Order changes are forward-transport/coefficient diagnostics; they do not relabel an independent-Cs reference failure")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HERE/"finite-source")
    parser.add_argument("--output", type=Path, default=HERE/"audit-summary.json")
    args = parser.parse_args()
    plan_sha = file_hash(HERE/"study-plan.md")
    reference = load_json(HERE/"variational-reference/report.json")
    requested = {}
    for path in (HERE/"requests.json", HERE/"requests-remaining.json", HERE/"requests-refinement.json"):
        for request in load_json(path):
            requested[request["name"]] = request
    missing = [name for name in requested if not (args.data_root/name/"files.json").is_file()]
    if missing:
        raise ValueError(f"Incomplete raw trials, refusing partial aggregate: {missing}")
    records, caches = {}, {}
    # Retain only four representative caches for cross-run comparisons.
    parents = {request["refinement_parent"] for request in requested.values() if "refinement_parent" in request}
    comparisons = []
    for name in sorted(requested, key=lambda item: ("refinement_parent" in requested[item], item)):
        print(f"AUDIT {name}", flush=True)
        record, cache = audit_case(args.data_root/name, plan_sha, reference)
        records[name] = record
        if name in parents:
            caches[name] = cache
        elif "refinement_parent" in requested[name]:
            parent_name = requested[name]["refinement_parent"]
            comparisons.append(compare_refinement(name, record, cache, records[parent_name], caches[parent_name]))
        del cache
    primary = [row for row in records.values() if "refinement_parent" not in row["request"]]
    failed_cs = [dict(case=row["name"], original_positive_count=fit["original_positive_count"],
        cartesian_relative_difference=fit["cartesian_relative_difference"],
        radial7_relative_difference=fit["radial7_relative_difference"])
        for row in primary for fit in row["complete_vacuum_fits"]
        if not fit["cartesian_gate_passed"] or not fit["radial7_gate_passed"]]
    summary = dict(schema="ideal-lens-independent-array-audit-v1",
        generated_utc=datetime.now(timezone.utc).isoformat(),
        audit_script_sha256=file_hash(Path(__file__)), frozen_plan_sha256=plan_sha,
        reference_sha256=file_hash(HERE/"variational-reference/report.json"),
        expected_primary_trials=14, expected_refinement_trials=12,
        actual_primary_trials=len(primary), actual_refinement_trials=len(comparisons),
        integrity_passed=all(row["audit_integrity_passed"] for row in records.values()),
        primary_source_count_d95_gates_passed=all(check["d95_count_gate_passed"] for row in primary for check in row["source_count_checks"]),
        primary_source_count_cs_gates_passed=all(row["source_cs_count_gate_passed"] for row in primary),
        primary_field_grid_gates_passed=all(row["field"]["sampled_field_gate"] for row in primary),
        primary_saved_physical_path_field_gates_passed=all(row["field"]["saved_physical_path_gate"] for row in primary),
        all_refinement_sources_and_observations_bit_exact=all(all(row["compatibility"].values()) for row in comparisons),
        all_representative_step_gates_passed=all(row["step_qualification_passed"] for row in comparisons if row["refinement"] == "half-step"),
        failed_primary_complete_vacuum_cs_reference_gates=failed_cs,
        primary_cases_exceeding_historical_100mrad_scope=[dict(case=row["name"], **row["historical_benchmark_scope"])
            for row in primary if row["historical_benchmark_scope"]["declared_pupil_exceeds_100mrad"] or
                row["historical_benchmark_scope"]["observed_alpha95_exceeds_100mrad"]],
        cases=list(records.values()), refinement_comparisons=comparisons,
        interpretation=[
            "Audit integrity is distinct from physical/numerical qualification; failed coefficient gates remain explicit.",
            "Primary observations are at the first paraxial reference image plane; this synthetic study has no specimen.",
            "D95 contains 95% of actually retained current about its current centroid; RMS is a second moment and responds differently to tails.",
            "Angular statistics use angles to the weighted chief direction, not a paraxial edge slope or a diameter.",
            "All source current is absolute; lost current remains lost. No tube-selected population is used to fit Cs.",
            "Counterfactual post-wall characteristics are retained for complete-vacuum fits but excluded from physical views.",
            "Field checks are independent analytic evaluation at finite grids/saved physical points, not a global continuous error proof.",
            "Step paired-position checks list both the fixed reference image and transport end, and do not hide changed membership.",
            "Native minima are observed grid minima and do not establish precision finer than their saved neighbouring planes.",
            "Source-count agreement only covers 1024 to 2048 for the declared synthetic source; no general continuum convergence is implied.",
            "The monochromatic zero-emittance ideal source is not an FEG calibration or a coherent-wave resolution prediction."
        ])
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False, default=json_scalar)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key not in ("cases", "refinement_comparisons", "interpretation")}, indent=2, default=json_scalar))
    return 0 if summary["integrity_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
