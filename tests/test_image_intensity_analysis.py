"""Scientific gates for offline intensity sensitivity, without propagation."""
import json

import numpy as np
import pytest

from scripts.analyse_image_intensity_convergence import (
    CASES, Cloud, Run, aggregate_probability, analyse, bilinear, compare_response,
    density_refinement, direct_gaussian, gaussian_probability,
    gaussian_remainder_bound, histogram, load_runs, population_summary,
    translation_moments,
    validate_point_spread, validate_protocol_receipt, canonical_digest,
    PROJECTOR_KEYS,
    survival_components, survival_response_summary,
)


def cloud(xy, weights=None):
    xy = np.asarray(xy, dtype=float)
    n = len(xy)
    return Cloud(xy, np.full(n, 1/n) if weights is None else np.asarray(weights),
                 np.ones(n, dtype=bool), np.arange(n))


def test_nested_hard_bins_conserve_probability_but_do_not_prove_density_convergence():
    particles = cloud([[.13, .21], [-.31, .27]], [.2, .3])
    coarse = histogram(particles, np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    fine = histogram(particles, np.linspace(-1, 1, 9), np.linspace(-1, 1, 9))
    np.testing.assert_allclose(aggregate_probability(fine, 2), coarse, atol=0)
    assert fine.sum() == pytest.approx(.5)
    comparison = density_refinement(coarse, fine)
    assert comparison['error_l1'] > 0
    assert comparison['status'] == 'FAIL_DECLARED_5_PERCENT'


def test_uniform_piecewise_density_has_zero_refinement_error():
    coarse = np.array([[.1, -.1], [-.2, .2]])
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)/4
    result = density_refinement(coarse, fine)
    assert result['signal_l1'] == pytest.approx(.6)
    assert result['status'] == 'PASS_DECLARED_5_PERCENT'
    assert result['error_l1'] == 0


@pytest.mark.parametrize('coarse', [np.zeros((2, 2)), np.ones((2, 2))])
def test_zero_fine_signal_is_never_a_convergence_pass(coarse):
    result = compare_response(coarse, np.zeros((2, 2)))
    assert result['status'] == 'UNRESOLVED_ZERO_FINE_SIGNAL'
    assert result['relative_error'] is None


def test_tiny_signal_below_declared_numerical_bound_is_unresolved():
    result = compare_response(np.array([1e-14]), np.array([1e-14]), numerical_error_bound=1e-15)
    assert result['status'] == 'UNRESOLVED_NUMERICAL_BOUND'


def test_bilinear_display_preserves_weight_centroid_and_subpixel_motion():
    particles = cloud([[.13, .21], [-.31, .27]], [.2, .3])
    edges = np.linspace(-1, 1, 17)
    image = bilinear(particles, edges, edges)
    centres = .5*(edges[:-1]+edges[1:])
    centroid = np.array([(image*centres[None, :]).sum(), (image*centres[:, None]).sum()])/.5
    np.testing.assert_allclose(centroid, particles.weight@particles.xy_mm/.5, atol=1e-15)
    shifted = cloud(particles.xy_mm+[.0001, 0.], particles.weight)
    assert not np.array_equal(image, bilinear(shifted, edges, edges))
    assert image.sum() == pytest.approx(.5)


def test_compact_gaussian_integrals_preserve_tilt_and_weak_variance_response():
    base = cloud([[-.00010, .00002], [.00015, -.00003], [.00004, .00008]], [.2, .3, .1])
    changed = cloud(base.xy_mm*1.00001+[2e-9, -1e-9], base.weight)
    edges = np.linspace(-2.4, 2.4, 33)
    reference = gaussian_probability(translation_moments(changed, base), edges, edges)
    direct = direct_gaussian(changed, edges, edges)-direct_gaussian(base, edges, edges)
    assert np.abs(reference).sum() > 0
    assert np.abs(reference-direct).sum()/np.abs(reference).sum() < 1e-5
    bound = gaussian_remainder_bound(base, (-2.4, 2.4, -2.4, 2.4))
    assert bound < 1e-17
    assert np.abs(reference.sum()) < 1e-9  # Same total input; only finite ROI tails differ.


def test_gaussian_reference_rejects_noncompact_or_unbounded_expansion_domain():
    with pytest.raises(ValueError, match='domain'):
        gaussian_remainder_bound(cloud([[.001, 0.]]), (-2.4, 2.4, -2.4, 2.4))
    with pytest.raises(ValueError, match='domain'):
        gaussian_remainder_bound(cloud([[.0001, 0.]]), (-40., 40., -40., 40.))


def test_population_metadata_exposes_weight_quantum_and_effective_population():
    particles = cloud([[0., 0.], [.1, 0.], [0., .1]], [.1, .2, .3])
    particles.survivor[-1] = False
    summary = population_summary(particles)
    assert summary['received_weight_total'] == pytest.approx(.3)
    assert summary['received_count'] == 2
    assert summary['effective_received_count'] == pytest.approx(1.8)
    assert summary['single_particle_two_bin_l1_min'] == pytest.approx(.2)
    assert summary['single_particle_two_bin_l1_max'] == pytest.approx(.4)


def test_manifest_loader_groups_complete_runs_without_reusing_ids_across_populations(tmp_path):
    rows = {}
    controls = dict.fromkeys(PROJECTOR_KEYS, 20.)
    for n in (2, 3):
        for case in CASES:
            filename = f'{n}_{case}.npz'
            np.savez(tmp_path/filename, xy_m=np.zeros((n, 2)), weight=np.full(n, 1/n),
                     survivor=np.ones(n, dtype=bool), source_id=np.arange(n))
            rows[filename] = dict(rays=n, step_mm=.025, case=case, status='COMPLETE',
                                  particles_path=filename, source_digest=f'different source {n}',
                                  projector_controls=controls.copy(), receiver_z_mm=2700.)
    rows['pending'] = dict(status='RUNNING')
    metadata = dict(schema='image-intensity-convergence-v1', runs=rows,
                    projector_controls=controls,
                    point_spread=dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=0.))
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    runs, loaded_metadata = load_runs(tmp_path)
    assert [r.rays for r in runs] == [2, 3]
    assert loaded_metadata['pending'][0]['status'] == 'RUNNING'
    rows['2_baseline.npz']['source_digest'] = 'wrong source'
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='source digest'):
        load_runs(tmp_path)
    rows['2_baseline.npz']['source_digest'] = 'different source 2'
    rows['2_baseline.npz']['particles_sha256'] = 'wrong checksum'
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='checksum'):
        load_runs(tmp_path)
    del rows['2_baseline.npz']['particles_sha256']
    rows['2_baseline.npz']['projector_controls']['diffraction_lens'] += .1
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='Projector controls changed'):
        load_runs(tmp_path)


def test_dead_or_zero_weight_nan_positions_do_not_contaminate_received_intensity():
    particles = Cloud(np.array([[.0001, 0.], [np.nan, np.nan], [np.nan, np.nan]]),
                      np.array([.2, .3, 0.]), np.array([True, False, True]), np.arange(3))
    edges = np.linspace(-.001, .001, 33)
    assert histogram(particles, edges, edges).sum() == pytest.approx(.2)
    assert bilinear(particles, edges, edges).sum() == pytest.approx(.2)
    assert all(np.isfinite(v) for v in translation_moments(particles).values())
    assert direct_gaussian(particles, edges, edges).sum() > 0
    with pytest.raises(ValueError, match='Invalid'):
        Cloud(particles.xy_mm, np.ones(3), np.ones(3, dtype=bool), np.arange(3))


@pytest.mark.parametrize('changed', ['source_digest', 'weight', 'source_id', 'receiver_z_mm'])
def test_step_refinement_rejects_changed_physical_source_or_observation_plane(tmp_path, changed):
    controls = dict.fromkeys(PROJECTOR_KEYS, 20.)
    rows = {}
    for h in (.05, .025):
        for case in CASES:
            filename = f'{h}_{case}.npz'
            weights, ids = np.full(2, .5), np.arange(2)
            if h == .025 and changed == 'weight':
                weights = np.array([.4, .6])
            if h == .025 and changed == 'source_id':
                ids = ids[::-1]
            np.savez(tmp_path/filename, xy_m=np.zeros((2, 2)), weight=weights,
                     survivor=np.ones(2, dtype=bool), source_id=ids)
            rows[filename] = dict(rays=2, step_mm=h, case=case, status='COMPLETE',
                particles_path=filename, source_digest='full source A',
                projector_controls=controls, receiver_z_mm=2700.)
            if h == .025 and changed == 'source_digest':
                # Same IDs and weights can conceal changed physical energies.
                rows[filename]['source_digest'] = 'full source with different energies'
            if h == .025 and changed == 'receiver_z_mm':
                rows[filename]['receiver_z_mm'] += 1.
    metadata = dict(schema='image-intensity-convergence-v1', runs=rows,
        projector_controls=controls,
        point_spread=dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=0.))
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    expected = {'source_digest': 'Full source digest', 'weight': 'source weights',
                'source_id': 'source ID order', 'receiver_z_mm': 'Observation plane'}
    with pytest.raises(ValueError, match=expected[changed]):
        load_runs(tmp_path)


def test_manifest_reports_planned_but_not_started_matrix_groups(tmp_path):
    metadata = dict(schema='image-intensity-convergence-v1', runs={},
        matrix=[dict(rays=3000, step_mm=.05)],
        projector_controls=dict.fromkeys(PROJECTOR_KEYS, 20.),
        point_spread=dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=0.))
    (tmp_path/'manifest.json').write_text(json.dumps(metadata))
    runs, info = load_runs(tmp_path)
    assert runs == []
    assert info['pending'] == [dict(rays=3000, step_mm=.05, status='PLANNED_GROUP_NOT_COMPLETE')]


def test_protocol_receipt_rejects_mixed_collection_row_and_snapshot(tmp_path):
    from hashlib import sha256
    controls = dict.fromkeys(PROJECTOR_KEYS, 20.)
    protocol = dict(projector_controls=controls, receiver_z_mm=2700., implementation='solver version A')
    document = dict(protocol=protocol, protocol_identity=canonical_digest(protocol), projector_controls=controls)
    snapshot = tmp_path/'state.json'
    snapshot.write_text(json.dumps(dict(digest='instrument state identity')))
    label = 'N3000_h0p025_baseline'
    row = dict(receiver_z_mm=2700., implementation_before='solver version A', implementation_after='solver version A',
               snapshot_path=snapshot.name, snapshot_sha256=sha256(snapshot.read_bytes()).hexdigest(),
               request_identity=canonical_digest(dict(protocol=document['protocol_identity'],
                   state='instrument state identity', target_z_mm=2700., run_id=label)))
    validate_protocol_receipt(document, row, label, tmp_path)
    with pytest.raises(ValueError, match='different protocol'):
        validate_protocol_receipt(document, dict(row, request_identity='other collection request'), label, tmp_path)
    with pytest.raises(ValueError, match='implementation'):
        validate_protocol_receipt(document, dict(row, implementation_after='changed during run'), label, tmp_path)
    snapshot.write_text(json.dumps(dict(digest='different instrument state')))
    with pytest.raises(ValueError, match='snapshot checksum'):
        validate_protocol_receipt(document, row, label, tmp_path)


@pytest.mark.parametrize('change', [dict(sigma_x_mm=.5), dict(model='none'), dict(rotation_deg=np.nan)])
def test_reference_never_substitutes_a_different_captured_psf(change):
    psf = dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=37.)
    assert validate_point_spread(psf) is psf  # Isotropic PSF is rotation invariant.
    with pytest.raises(ValueError, match='captured'):
        validate_point_spread(dict(psf, **change))


def test_single_population_reports_population_axis_not_run_and_keeps_zero_signals_unresolved():
    base = cloud([[-.0001, .00002], [.0001, -.00002]])
    runs = [Run('one', 2, .025, {name: base for name in CASES}, {'emitted_current_a': 2e-12})]
    result = analyse(runs, [32, 64], include_bilinear=False)
    assert result['axis_availability']['particle_count'] == 'NOT_RUN_NO_INDEPENDENT_AXIS_PAIR'
    assert result['axis_availability']['integration_step'] == 'NOT_RUN_NO_INDEPENDENT_AXIS_PAIR'
    assert all(row['status'] == 'UNRESOLVED_ZERO_FINE_SIGNAL' for row in result['axes'])
    for row in result['runs'][0]['cases'].values():
        assert row['emitted_current_a'] == 2e-12
        assert row['received_current_pa'] == pytest.approx(2.)


@pytest.mark.parametrize('format', ['manifest', 'legacy'])
def test_adjacent_step_values_keep_distinct_signals_and_do_not_false_pass(tmp_path, format):
    """Roundoff-close h values remain distinct even if their measured fields differ."""
    steps = [.025, float(np.nextafter(.025, np.inf))]
    controls = dict.fromkeys(PROJECTOR_KEYS, 20.)
    psf = dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=0.)
    rows, legacy = {}, {'source_id': np.arange(2)}
    for index, step in enumerate(steps):
        for case in CASES:
            xy_mm = np.array([[-.0001, .00002], [.0001, -.00002]])
            if case != 'baseline':
                xy_mm[:, 0] += (index+1)*1e-6
            arrays = dict(xy_m=xy_mm*1e-3, weight=np.full(2, .5),
                          survivor=np.ones(2, dtype=bool), source_id=np.arange(2))
            if format == 'manifest':
                filename = f'{index}_{case}.npz'
                np.savez(tmp_path/filename, **arrays)
                rows[filename] = dict(rays=2, step_mm=step, case=case, status='COMPLETE',
                    particles_path=filename, source_digest='same executed source',
                    projector_controls=controls, receiver_z_mm=2700.)
            else:
                legacy.update({f'step{index}_{case}_imaging_{key}': value
                               for key, value in arrays.items() if key != 'source_id'})
    if format == 'manifest':
        document = dict(schema='image-intensity-convergence-v1', runs=rows,
                        projector_controls=controls, point_spread=psf)
        (tmp_path/'manifest.json').write_text(json.dumps(document))
    else:
        np.savez(tmp_path/'particles.npz', **legacy)
        document = dict(schema='matched-c2-screen-particles-v1', integration_steps_mm=steps,
                        projector_controls=controls, point_spread=psf)
        (tmp_path/'comparison.json').write_text(json.dumps(document))
    runs, _ = load_runs(tmp_path)
    assert {run.label for run in runs} == {'N2_h0.025', 'N2_h0.025000000000000005'}
    result = analyse(runs, [32], include_bilinear=False)
    gaussian = [row for row in result['axes'] if row['axis'] == 'integration_step'
                and row['method'] == 'fixed_physical_gaussian']
    assert len(gaussian) == 3
    assert all(row['coarse_run'] != row['fine_run'] for row in gaussian)
    assert all(row['relative_error'] > .9 for row in gaussian)
    assert all(row['status'] == 'FAIL_DECLARED_5_PERCENT' for row in gaussian)


def test_direct_analysis_rejects_duplicate_labels_before_signal_cache_overwrite():
    base = cloud([[-.0001, .00002], [.0001, -.00002]])
    runs = [Run('rounded label', 2, h, {name: base for name in CASES}, {})
            for h in [.025, float(np.nextafter(.025, np.inf))]]
    with pytest.raises(ValueError, match='Run labels must uniquely'):
        analyse(runs, [32])


def test_four_particle_survival_decomposition_matches_independent_pixel_integrals_and_centroid():
    """Nonuniform probabilities, real displacement, one loss and one gain coexist."""
    from scipy.special import ndtr
    weights = np.array([.1, .2, .3, .4])
    ids = np.array([10, 20, 30, 40])
    base_xy = np.array([[-.0001, .00004], [.00012, -.00008], [.00018, .00011], [np.nan, np.nan]])
    case_xy = np.array([[-.00009, .00006], [.00013, -.00006], [np.nan, np.nan], [-.00016, .0001]])
    base = Cloud(base_xy.copy(), weights.copy(), np.array([True, True, True, False]), ids,
                 np.array(['screen', 'screen', 'screen', 'aperture']), np.array([2700., 2700., 2700., 765.]))
    case = Cloud(case_xy.copy(), weights.copy(), np.array([True, True, False, True]), ids,
                 np.array(['screen', 'screen', 'aperture', 'screen']), np.array([2700., 2700., 765., 2700.]))
    components, summary = survival_components(case, base)
    xe, ye = np.linspace(-.9, .6, 18), np.linspace(-.5, .8, 14)
    def point_image(point, weight):
        return weight*np.outer(np.diff(ndtr((ye-point[1])/.6)), np.diff(ndtr((xe-point[0])/.6)))
    independent = dict(
        common_displacement=sum(point_image(case_xy[i], weights[i])-point_image(base_xy[i], weights[i]) for i in (0, 1)),
        lost_negative=-point_image(base_xy[2], weights[2]),
        gained_positive=point_image(case_xy[3], weights[3]))
    direct_total = sum(point_image(case_xy[i], weights[i]) for i in (0, 1, 3))-sum(point_image(base_xy[i], weights[i]) for i in (0, 1, 2))
    np.testing.assert_allclose(sum(independent.values()), direct_total, rtol=0, atol=2e-17)
    result = survival_response_summary(components, translation_moments(case, base), xe, ye)
    for name, actual in independent.items():
        assert result['components'][name]['absolute_l1'] == pytest.approx(np.abs(actual).sum(), rel=2e-10, abs=2e-16)
        assert result['components'][name]['signed_viewport_probability'] == pytest.approx(actual.sum(), abs=2e-16)
    assert result['total_absolute_l1'] == pytest.approx(np.abs(direct_total).sum(), abs=2e-16)
    assert result['reconstruction_l1_error'] < 2e-16
    assert result['cancelled_component_l1'] > 0
    assert summary['signed_full_plane_probability_change'] == pytest.approx(.1)
    assert result['total_signed_viewport_probability'] < .1  # Cropped PSF tails are not renormalised.
    expected_centroid = weights[[0, 1, 3]]@case_xy[[0, 1, 3]]/.7-weights[:3]@base_xy[:3]/.6
    np.testing.assert_allclose(summary['actual_centroid_change_mm'], expected_centroid, atol=1e-19)
    np.testing.assert_allclose(sum(np.asarray(c['centroid_change_contribution_mm']) for c in summary['components'].values()), expected_centroid, atol=1e-19)
    assert summary['components']['lost_negative']['source_ids'] == [30]
    assert summary['components']['gained_positive']['source_ids'] == [40]
    assert summary['components']['lost_negative']['case_stops']['by_key']['aperture']['z_mm_min'] == 765.
    np.testing.assert_array_equal(base.xy_mm, base_xy)
    np.testing.assert_array_equal(case.xy_mm, case_xy)
    np.testing.assert_array_equal(case.weight, weights)


def test_equal_loss_and_gain_cancel_but_remain_reported_and_empty_centroid_is_undefined():
    xy = np.array([[.0001, -.00005], [.0001, -.00005]])
    base = Cloud(xy, np.array([.2, .2]), np.array([True, False]), np.arange(2))
    case = Cloud(xy, base.weight, np.array([False, True]), base.source_id)
    moments, summary = survival_components(case, base)
    edges = np.linspace(-2.4, 2.4, 33)
    response = survival_response_summary(moments, translation_moments(case, base), edges, edges)
    assert response['total_absolute_l1'] == 0
    assert response['sum_component_absolute_l1'] > .39
    assert response['cancelled_fraction'] == 1.
    assert summary['components']['lost_negative']['count'] == 1
    assert summary['components']['gained_positive']['count'] == 1
    assert summary['components']['lost_negative']['case_stops']['status'] == 'NOT_RECORDED'
    empty = Cloud(np.full((2, 2), np.nan), base.weight, np.zeros(2, dtype=bool), base.source_id)
    _, all_lost = survival_components(empty, base)
    assert all_lost['case_centroid_mm'] is None
    assert all_lost['actual_centroid_change_mm'] is None
    assert all_lost['components']['lost_negative']['centroid_change_contribution_mm'] is None


def test_manifest_retains_actual_stop_and_source_summaries_and_rejects_tampered_source(tmp_path):
    from hashlib import sha256
    weights, ids = np.array([.3, .7]), np.array([8, 9])
    controls = dict.fromkeys(PROJECTOR_KEYS, 20.)
    rows = {}
    for case in CASES:
        path = tmp_path/f'{case}.npz'
        np.savez(path, xy_m=np.array([[1e-7, 0.], [np.nan, np.nan]]), weight=weights,
                 survivor=np.array([True, False]), source_id=ids,
                 blocked_key=np.array(['screen', 'tip']), blocked_z_mm=np.array([2700., 0.]),
                 final_alive=np.array([False, False]))
        rows[case] = dict(rays=2, step_mm=.025, case=case, status='COMPLETE', particles_path=path.name,
                         source_digest='physical source', projector_controls=controls, receiver_z_mm=2700.)
    source = tmp_path/'source.npz'
    np.savez(source, gun_exit_ray_id=ids, gun_exit_weight=weights, gun_exit_alive=np.array([True, False]),
             gun_blocked_key=np.array(['', 'tip']), gun_blocked_z_mm=np.array([np.nan, 0.]))
    source_receipt = dict(particles_path=source.name, particles_sha256=sha256(source.read_bytes()).hexdigest(),
                          source_digest='physical source', sampler_diagnostics={'schema': 'recorded source'})
    document = dict(schema='image-intensity-convergence-v1', runs=rows, sources={'2': source_receipt},
                    projector_controls=controls, point_spread=dict(model='gaussian', sigma_x_mm=.6, sigma_y_mm=.6, rotation_deg=0.))
    (tmp_path/'manifest.json').write_text(json.dumps(document))
    runs, info = load_runs(tmp_path)
    assert runs[0].clouds['baseline'].blocked_key.tolist() == ['screen', 'tip']
    assert not runs[0].clouds['baseline'].final_alive.any()
    source_summary = info['retained_source_summaries']['2']
    assert source_summary['gun_exit_probability'] == .3
    assert source_summary['stops']['by_key']['tip']['source_probability'] == .7
    assert source_summary['stops']['by_key']['']['nonfinite_z_count'] == 1
    source.write_bytes(b'tampered source')
    with pytest.raises(ValueError, match='source file checksum'):
        load_runs(tmp_path)
