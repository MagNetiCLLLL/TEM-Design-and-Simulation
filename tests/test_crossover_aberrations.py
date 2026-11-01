import numpy as np
import pytest


def test_threefold_terms_separate_quadratic_and_quartic_multipoles(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from measure_crossover_cs import geometric_threefold_terms
    from temsim.optics.crossover_aberrations import measure_geometric_cs, reference_pencil

    launch, n = reference_pencil(1., 0.)
    u = .03 * (launch[:, 0] + 1j*launch[:, 1])
    a2, fourth_minus, fourth_plus = (.7+1.2j)*1e-6, (.4-.8j)*1e-3, (-.1+.2j)*1e-3
    z = a2*u.conj()**2 + fourth_minus*abs(u)**2*u.conj()**2 + fourth_plus*u**4
    z += .2e-3*abs(u)**2*u + .5e-6*u**2
    raw = np.column_stack((z.real, z.imag, u.real, u.imag))
    fit = measure_geometric_cs(launch[:, :2], raw, n, holdout_budget_m=1e-12)
    result = geometric_threefold_terms(fit)
    np.testing.assert_allclose(result["a2_xy_um"], [.7, 1.2])
    np.testing.assert_allclose(result["fourth_neg2_xy_mm"], [.4, -.8])
    np.testing.assert_allclose(result["fourth_pos4_xy_mm"], [-.1, .2])
    assert geometric_threefold_terms({"status": "INCONCLUSIVE", "c3_mm": None}) is None


def test_annular_cancellation_retains_threefold_current_in_individual_shells(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from measure_probe_crossover import threefold_tail

    # Two 120-degree triplets with opposite third moments and one blocked ray.
    angles = np.tile(np.arange(3)*2*np.pi/3, 2) + np.repeat([0., np.pi/3], 3)
    z = np.r_[np.repeat([.2, .8], 3)*np.exp(1j*angles), 1.5, .3] * 1e-9
    live = np.arange(8) != 7
    output = threefold_tail(z.real, z.imag, live, np.ones(8), 8e-6)
    assert output["normalized_m3"] == pytest.approx(0., abs=1e-12)
    assert output["annular_current_a"] == pytest.approx(6e-6)
    assert output["shell_harmonic_sum_a"] == pytest.approx(6e-6)
    assert output["outside_1nm_current_a"] == pytest.approx(1e-6)

from temsim.optics.crossover_aberrations import measure_geometric_cs, reference_pencil


def test_positive_marginal_overfocus_with_astigmatism_rotation_and_fifth_order():
    launch, n = reference_pencil(2.e-6, .55)
    u = launch[:, :2] @ np.array([[0., 1.], [-1., 0.]]) * 1.5e4
    q = np.sum(u*u, axis=1)
    # Rays cross the axis upstream by Cs*tan(theta)^2. At z=0 they have
    # traveled that positive extra distance with the outgoing slope.
    position = (.12e-6 + .002*q - .07*q*q)[:,None]*u
    position += np.column_stack((.0003*u[:,0]**3, -.0003*u[:,1]**3))
    chief = np.array([.4e-9, -.2e-9, .01, -.02])
    result = measure_geometric_cs(launch[:,:2], np.column_stack((position,u))+chief, n, holdout_budget_m=1e-12)
    assert result["sign_at_fixed_step"] == "positive"
    assert result["c3_mm"] == pytest.approx(2.)
    assert result["fit"]["c5_m"] == pytest.approx(-.07)
    assert result["fit"]["c1_m"] == pytest.approx(.12e-6)
    np.testing.assert_allclose(result["chief_xy_slopes"], chief)
    # Moving the observation plane by free drift changes the linear term only.
    position += .05*u
    shifted = measure_geometric_cs(launch[:,:2], np.column_stack((position,u))+chief, n, holdout_budget_m=1e-12)
    assert shifted["c3_mm"] == pytest.approx(result["c3_mm"], abs=1e-8)
    assert shifted["fit"]["c1_m"]-result["fit"]["c1_m"] == pytest.approx(.05)


def test_holdout_error_clipping_and_line_chart_suppress_cs():
    launch, n = reference_pencil(1., 0.)
    u = launch[:,:2]*.02
    raw = np.column_stack((.03*np.sum(u*u,axis=1)[:,None]*u, u))
    raw[n+1:, 0] += 1e-7
    assert measure_geometric_cs(launch[:,:2],raw,n,holdout_budget_m=1e-10)["c3_mm"] is None
    raw[::10,0] = np.nan
    assert measure_geometric_cs(launch[:,:2],raw,n,holdout_budget_m=1e-10)["c3_mm"] is None
    line = np.column_stack((u,u[:,0],np.zeros(len(u))))
    assert measure_geometric_cs(launch[:,:2],line,n,holdout_budget_m=1e-10)["c3_mm"] is None


def test_unified_compact_ring_cannot_infer_corrector_effect(monkeypatch):
    from temsim.optics import aberrations
    from temsim.optics.column import default_state
    state = default_state()
    state.simulation_mode = "unified"
    monkeypatch.setattr(aberrations, "_corrector_trace_ratio", lambda *args: pytest.fail("legacy flag comparison"))
    _, _, diagnostic = aberrations.effective_aberration_comparison(state, "probe")
    assert diagnostic["c3_residual_ratio"] is None
    assert diagnostic["inferred_coefficients"] == ()


def test_target_kernel_retains_absolute_current_and_outside_charge():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from measure_crossover_cs import target_current_kernel
    arrays = dict(z_mm=np.array([12.]), weights=np.array([1.,2.,3.,4.]),
                  source_id=np.array([10,11,12,13]), alive=np.array([[True,True,True,False]]),
                  x_m=np.array([[1.,-1.,6.,np.nan]])*1e-9,
                  y_m=np.array([[-1.,1.,0.,np.nan]])*1e-9)
    kernel = target_current_kernel(arrays, 12., 10e-9, pitch_nm=1., pixels=4)
    assert kernel["current_a_yx"][1,3] == pytest.approx(1e-9,abs=1e-20)
    assert kernel["current_a_yx"][3,1] == pytest.approx(2e-9,abs=1e-20)
    assert kernel["outside_current_a"] == pytest.approx(3e-9,abs=1e-20)
    assert kernel["transmitted_current_a"] == pytest.approx(6e-9,abs=1e-20)
    np.testing.assert_array_equal(kernel["surviving_source_ids"], [10,11,12])


def test_saved_gui_survey_keeps_unresolved_rows(qtbot, tmp_path):
    import json
    from temsim.gui.crossover_survey_view import CrossoverSurveyView
    target = {"z_mm": 4., "pencils":[{"c3_mm": None, "status": "INCONCLUSIVE"},
                                      {"c3_mm": -.3, "status": "MEASURED_AT_FIXED_STEP"}]}
    report = dict(schema="unified-crossover-geometric-cs-v1", input_digest="archived",
        pupil={"radii_um": [.5,.125]}, target_aberrations=target,
        lens_intervals=[{"upstream_reference": "C3", "regions":[], "status":"RESOLVED_AT_BOUNDED_SAMPLING"},
                        {"upstream_reference": "TL21", "regions":[], "status":"UNRESOLVED"}])
    path=tmp_path/"survey.json"
    path.write_text(json.dumps(report))
    view = CrossoverSurveyView()
    qtbot.addWidget(view)
    view.load_report(path)
    assert view.table.rowCount()==3
    assert view.table.item(0,5).text().startswith("No bounded")
    assert view.table.item(1,5).text()=="UNRESOLVED"
    assert view.table.item(2,3).text()=="—"
    assert view.table.item(2,4).text()=="-0.3"
    path.write_text("{}")
    view.load_report(path)
    assert view.table.rowCount()==0 and "Unable to load" in view.summary.text()


def test_target_comparison_rejects_single_plane_focus_and_missing_refinements():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
    from probe_correction_assessment import fixed_target_assessment
    base=dict(z_mm=1., current_a=1., d50_nm=1., d95_nm=2., core_50pm_a=.1, alpha95_mrad=30.,source_ids=[1])
    candidate=dict(base,d50_nm=.7,d95_nm=1.9,core_50pm_a=.2)
    one=fixed_target_assessment(base,candidate,[candidate])
    assert one["fixed_target_performance"]=="PASS"
    assert one["focus_status"]=="INCONCLUSIVE" and one["step_refinement"]["status"]=="NOT_EVALUATED"
    outside=dict(candidate,core_50pm_a=.1,d50_nm=.9)
    result=fixed_target_assessment(base,candidate,[dict(outside,z_mm=.9),candidate,dict(outside,z_mm=1.1)])
    assert result["focus_status"]=="PASS" and result["full_qualification"]=="NOT_QUALIFIED"


def test_d95_focus_requires_both_observed_extrema_at_the_same_plane():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from probe_correction_assessment import fixed_target_assessment
    baseline = dict(z_mm=2., current_a=1., d50_nm=.1, d95_nm=2., core_50pm_a=.1,
                    alpha95_mrad=30., source_ids=[1, 2])
    target = dict(baseline, d50_nm=.2, d95_nm=1.1, core_50pm_a=.3)
    outer = dict(target, d50_nm=.18, core_50pm_a=.21, d95_nm=1.2)
    rows = [dict(outer, z_mm=0.), dict(outer, z_mm=1.), target,
            dict(outer, z_mm=3.), dict(outer, z_mm=4.)]
    both = fixed_target_assessment(baseline, target, rows, diameter="d95")
    assert both["fixed_target_performance"] == both["focus_status"] == "PASS"
    assert both["both_observed_extrema_at_target"] and both["best_d95_planes_mm"] == [2.]
    # Near-optimal differs from a native shared extremum, even with stable transmission.
    rows[1]["d95_nm"] = 1.095
    split = fixed_target_assessment(baseline, target, rows, diameter="d95")
    assert split["within_focus_allowance"] and split["focus_status"] == "FAIL"
    assert not split["both_observed_extrema_at_target"]
    rows[1]["source_ids"] = [2]
    assert fixed_target_assessment(baseline, target, rows, diameter="d95")["focus_status"] == "INCONCLUSIVE"


def test_linear_drift_separates_d95_from_rms_without_spherical_aberration():
    # Two symmetric cohorts cross at 0 and 1 mm with equal absolute slopes.
    # Their weighted squared RMS is proportional to .96*z**2 + .04*(z-1)**2.
    from temsim.physics.beam_statistics import _weighted_quantile
    focus_mm = np.r_[np.zeros(1920), np.ones(80)]
    slopes = np.tile([.01, -.01], 1000)
    weights = np.full(2000, 1/2000)
    z = np.linspace(-.1, 1.1, 241)
    x_mm = (z[:,None] - focus_mm) * slopes
    radii = abs(x_mm - (x_mm@weights)[:,None])
    d95 = [2*_weighted_quantile(row, weights, .95) for row in radii]
    rms = np.sqrt((radii**2)@weights)
    assert z[np.argmin(d95)] == pytest.approx(0., abs=1e-12)
    assert z[np.argmin(rms)] == pytest.approx(.04, abs=1e-12)


def test_nonzero_rms_d95_separation_does_not_veto_joint_d95_focus():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from probe_correction_assessment import fixed_target_assessment
    reference = dict(z_mm=3., current_a=1., core_50pm_a=.1, d50_nm=.3, d95_nm=2.,
                     alpha95_mrad=30., surviving_rays=20, effective_rays=20.,
                     source_ids=list(range(20)))
    rows = [dict(reference, z_mm=float(z), core_50pm_a=.2 if z==3 else .15,
                 d95_nm=1.+.05*(z-3)**2, rms_radius_nm=.6+.05*(z-1)**2)
            for z in range(5)]
    report = fixed_target_assessment(reference, rows[3], rows, diameter="d95")
    assert report["focus_status"] == "PASS"
    assert report["axial_waists"]["separation"]["resolved_delta_z_nm"] == -2e6
    rows[1]["effective_rays"] = 2.
    assert fixed_target_assessment(reference, rows[3], rows, diameter="d95")["axial_waists"]["separation"]["resolved_delta_z_nm"] is None
