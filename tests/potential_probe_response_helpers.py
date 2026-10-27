"""Predeclared numerical sensitivity and functional-drive qualifications."""
import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
import test_potential_probe_whole_wave as m


def _raw_phase_on(reference, changed):
    a, b = reference.beam.modes[0], changed.beam.modes[0]
    pa, pb = a.plane, b.plane
    refined = m.refine_plane_wave(pb, tuple(n*2 for n in pb.amplitude.shape),
        numerics=m.WaveGridNumerics(maximum_pixels=8192, maximum_working_bytes=8*1024**3))
    xy = pa.coordinates_m()
    coordinates = np.einsum("ij,jyx->iyx", np.linalg.inv(refined.basis_m), xy-refined.origin_m[:, None, None])
    coordinates += np.array((refined.amplitude.shape[1]//2, refined.amplitude.shape[0]//2))[:, None, None]
    sampled = sum(multiplier*map_coordinates(spline_filter(values, order=5, mode="constant"),
        coordinates[::-1], order=5, mode="constant", cval=0., prefilter=False)
        for multiplier, values in ((1., refined.amplitude.real), (1j, refined.amplitude.imag)))
    sampled *= np.exp(2j*np.pi/float(m.wavelength_m(a.energy_kev*1000.))*(m._carrier(pb, xy)-m._carrier(pa, xy)))
    mask = (abs(pa.amplitude)**2 > 1e-6*np.max(abs(pa.amplitude)**2)) & (abs(sampled)**2 > 1e-6*np.max(abs(sampled)**2))
    return np.angle(sampled*np.conjugate(pa.amplitude)), mask


def _sample_on(reference, other, factor=2):
    a, b = reference.beam.modes[0], other.beam.modes[0]
    pa, pb = a.plane, b.plane
    if reference.digest == other.digest:
        return pa.amplitude*np.sqrt(a.weight_per_reference_electron), 0.
    refined = m.refine_plane_wave(pb, tuple(n*factor for n in pb.amplitude.shape),
        numerics=m.WaveGridNumerics(maximum_pixels=8192, maximum_working_bytes=8*1024**3))
    xy = pa.coordinates_m()
    coordinates = np.einsum("ij,jyx->iyx", np.linalg.inv(refined.basis_m), xy-refined.origin_m[:, None, None])
    coordinates += np.array((refined.amplitude.shape[1]//2, refined.amplitude.shape[0]//2))[:, None, None]
    sampled = sum(multiplier*map_coordinates(spline_filter(values, order=5, mode="constant"),
        coordinates[::-1], order=5, mode="constant", cval=0., prefilter=False)
        for multiplier, values in ((1., refined.amplitude.real), (1j, refined.amplitude.imag)))
    sampled *= np.sqrt(b.weight_per_reference_electron*np.linalg.det(pa.basis_m)/np.linalg.det(refined.basis_m))
    sampled *= np.exp(2j*np.pi/float(m.wavelength_m(a.energy_kev*1000.))*(m._carrier(pb, xy)-m._carrier(pa, xy)))
    back = np.einsum("ij,jyx->iyx", np.linalg.inv(pa.basis_m), pb.coordinates_m()-pa.origin_m[:, None, None])
    low = np.array((pa.amplitude.shape[1]//2, pa.amplitude.shape[0]//2))[:, None, None]
    high = np.array((pa.amplitude.shape[1]-pa.amplitude.shape[1]//2,
        pa.amplitude.shape[0]-pa.amplitude.shape[0]//2))[:, None, None]
    outside = ~np.all((back >= -low) & (back < high), axis=0)
    probability = float(np.sum(abs(pb.amplitude[outside])**2))*b.weight_per_reference_electron
    return sampled, probability


def _paired(zero_a, plus_a, zero_b, plus_b, incident_weight):
    records = (zero_a, plus_a, zero_b, plus_b)
    anchor = zero_a.beam.modes[0]
    for record in records:
        assert len(record.beam.modes) == 1 and record.plane_z_mm == zero_a.plane_z_mm
        mode = record.beam.modes[0]
        assert (mode.mode_id, mode.energy_kev, mode.axial_reference) == (anchor.mode_id, anchor.energy_kev, anchor.axial_reference)
        assert record.beam.reference_plane == zero_a.beam.reference_plane
        assert record.reference_current_a == zero_a.reference_current_a
    assert incident_weight > 0
    result = {}
    for name, reference in (("forward", zero_a), ("reverse", zero_b)):
        variants, previous = [], None
        for factor in (2, 4):
            sampled = [_sample_on(reference, record, factor) for record in records]
            a0, a1, b0, b1 = (item[0] for item in sampled)
            scale = np.sqrt(incident_weight)
            tail = sum(np.sqrt(item[1]) for item in sampled)/scale
            delta = (a1-a0)-(b1-b0)
            error = float(np.sqrt(np.sum(abs(delta)**2)))/scale+tail
            mask = np.logical_and.reduce([abs(value)**2 > 1e-6*np.max(abs(value)**2)
                for value in (a0, a1, b0, b1)])
            assert np.sum(mask) > 100
            phase = np.angle(a1*np.conjugate(a0))-np.angle(b1*np.conjugate(b0))
            phase = np.angle(np.exp(1j*phase))
            variants.append({"relative_complex_L2": error,
                "raw_phase_rms_rad": float(np.sqrt(np.mean(phase[mask]**2))),
                "outside_L2_triangle_bound": tail, "phase_valid_cells": int(np.sum(mask)),
                "comparison_refinement": factor})
            if previous is not None:
                common = mask & previous[2]
                assert np.sum(common) > 100
                refinement = {"relative_complex_L2": float(np.linalg.norm(delta-previous[0]))/scale,
                    "raw_phase_rms_rad": float(np.sqrt(np.mean(np.angle(np.exp(1j*(phase[common]-previous[1][common])))**2)))}
            previous = (delta, phase, mask)
        result[name] = {key: max(row[key] for row in (*variants, refinement))
            for key in ("relative_complex_L2", "raw_phase_rms_rad")}
        result[name].update(comparison_refinements=variants, comparison_refinement_difference=refinement,
            fixed_incident_weight=incident_weight, output_weights=[record.beam.total_weight for record in records])
    return result


def _signal(zero, plus, incident_weight):
    result = {}
    for name, reference in (("forward", plus), ("reverse", zero)):
        variants, previous = [], None
        for factor in (2, 4):
            a, _ = _sample_on(reference, zero, factor)
            b, _ = _sample_on(reference, plus, factor)
            mask = (abs(a)**2 > 1e-6*np.max(abs(a)**2)) & (abs(b)**2 > 1e-6*np.max(abs(b)**2))
            assert np.sum(mask) > 100
            delta, phase = b-a, np.angle(b*np.conjugate(a))
            variants.append({"relative_complex_L2": float(np.linalg.norm(delta))/np.sqrt(incident_weight),
                "raw_phase_rms_rad": float(np.sqrt(np.mean(phase[mask]**2)))})
            if previous is not None:
                common = mask & previous[2]
                assert np.sum(common) > 100
                interpolation = {"relative_complex_L2": float(np.linalg.norm(delta-previous[0]))/np.sqrt(incident_weight),
                    "raw_phase_rms_rad": float(np.sqrt(np.mean(np.angle(np.exp(1j*(phase[common]-previous[1][common])))**2)))}
            previous = delta, phase, mask
        result[name] = {key: min(row[key] for row in variants)
            for key in ("relative_complex_L2", "raw_phase_rms_rad")}
        result[name].update(interpolation_refinement_error=interpolation, comparison_refinements=variants,
            scope="in-window response norm in fixed incident units; omitted exterior cannot inflate the signal")
    return result


def qualify(mode, case, budget, paired=False):
    options = {"support_budget": budget}
    runs = [m._whole(mode, step, case=case, **options) for step in m.STEPS[1:]]
    baseline = [m._whole(mode, step, **options)[0] for step in m.STEPS[1:]]
    incident_weight = m._initial(mode).beam.total_weight
    assert incident_weight == .25
    gauge = m._hardware(mode)[0].gauge.to_dict()
    for source_case in ("baseline", case):
        for order in (3, 5):
            assert m._hardware(mode, source_case, order)[0].gauge.to_dict() == gauge
    for run in runs:
        assert tuple(run[1]["input_width_curvature"]) == m.INPUT[mode]
        assert run[1]["paired_prefix_equivalence"] is not None
    responses = [m._compare(run[0], nominal) for run, nominal in zip(runs, baseline)]
    uncertainty = m._compare(runs[0][0], runs[1][0])
    controls = {"axial": m._compare(*baseline)}
    perturbed_controls = {"axial": uncertainty}
    paired_controls = {"axial": _paired(baseline[1], runs[1][0], baseline[0], runs[0][0], incident_weight)} if paired else {}
    for name, extra in (("grid", {"grid": "refined"}), ("window", {"grid": "wider"}),
            ("field_order", {"order": 5}), ("support_looser", {"support_budget": 1e-6 if budget == 3e-7 else 3e-6}),
            ("support_tighter", {"support_budget": 1e-7 if budget == 3e-7 else 3e-7})):
        other = m._whole(mode, .00025, **(options | extra))[0]
        controls[name] = m._compare(baseline[0], other)
        if paired:
            changed = m._whole(mode, .00025, case=case, **(options | extra))[0]
            perturbed_controls[name] = m._compare(runs[0][0], changed)
            paired_controls[name] = _paired(baseline[0], runs[0][0], other, changed, incident_weight)
    for row in (uncertainty, *controls.values(), *perturbed_controls.values()):
        m._accept(row)
    signals = _signal(baseline[-1], runs[-1][0], incident_weight) if paired else responses[-1]
    if paired:
        paired_controls["signal_interpolation"] = {direction: signals[direction]["interpolation_refinement_error"]
            for direction in ("forward", "reverse")}
    floor = tuple(paired_controls.values()) if paired else (uncertainty, *controls.values())
    for direction in ("forward", "reverse"):
        error = max(row[direction]["relative_complex_L2"] for row in floor)
        phase_error = max(row[direction]["raw_phase_rms_rad"] for row in floor)
        assert signals[direction]["relative_complex_L2"] > 5*error
        assert signals[direction]["raw_phase_rms_rad"] > phase_error
    return {"runs": [run[1] for run in runs], "perturbation": m.RESOLVABLE_A2_CASES[case],
        "responses": responses, "step_uncertainty": uncertainty,
        "baseline_step_uncertainty": controls["axial"],
        "independent_grid_window_field_support_uncertainty": controls,
        "perturbed_grid_window_field_support_refinements": perturbed_controls,
        "paired_response_refinements": paired_controls,
        "fixed_incident_response": signals, "fixed_incident_weight": incident_weight,
        "common_gauge": gauge,
        "numerical_floor_scope": "Maximum empirical refinement difference, not a rigorous combined bound or statistical sigma",
        "resolution": ("exceeds_five_times_paired_empirical_refinement_floor" if paired
            else "exceeds_five_times_independent_numerical_uncertainty"),
        "support_budget": budget, "physical_input_or_hardware_fit": False}, baseline[-1], runs[-1][0]


def functional_stress_evidence(mode):
    case = "A2_fine_HPC_1e6"
    m.RESOLVABLE_A2_CASES[case] = ("probe_hpc_hexapole", "strength_m3", 1., 1e6)
    data, baseline, full = qualify(mode, case, 1e-6, paired=True)
    small = m._whole(mode, .000125, case="A2_fine_HPC_1e5")[0]
    small_response = m._compare(small, baseline)
    phase_small, mask_small = _raw_phase_on(baseline, small)
    phase_full, mask_full = _raw_phase_on(baseline, full)
    mask = mask_small & mask_full
    correlation = float(np.dot(phase_small[mask], phase_full[mask])/
        np.sqrt(np.sum(phase_small[mask]**2)*np.sum(phase_full[mask]**2)))
    ratio = data["responses"][-1]["forward"]["relative_complex_L2"]/small_response["forward"]["relative_complex_L2"]
    assert 5. < ratio < 20.
    assert correlation > .8
    data["retained_small_candidate"] = {"increment_m3": 1e5, "response": small_response,
        "large_to_small_complete_complex_ratio": ratio, "raw_phase_direction_correlation": correlation,
        "scope": "Finite-grid direction/scaling evidence only; the original +1e5 joint resolution failure remains preserved"}
    data["scope"] = "Actual +1e6 m^-3 engineering-model functional stress drive, not an OEM fine-step calibration"
    return data
