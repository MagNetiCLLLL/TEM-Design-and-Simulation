import numpy as np
import pytest
from scripts.source_radial_quadrature import (solve_radial_partitions,gauss_table,integrate_table,
                              validate_node_membership,BoundaryResolutionError)


def test_all_base_case_sign_roots_are_global_batches_and_small_flux_is_not_one_over_n():
    threshold=np.array([.271828,.381966,.618034,.731234])
    epsilon=2e-8
    batches=[]
    def margin(base,u):
        batches.append((base.copy(),u.copy()))
        return np.column_stack((threshold[base]-u,threshold[base]+epsilon-u,
                                threshold[base]-.01-u,threshold[base]+.01-u))
    solution=solve_radial_partitions(margin,4,('base+','case+','base-','case-'),u_tol=1e-11)
    table=gauss_table(solution,order=2)
    actual=validate_node_membership(solution,table)>=0
    flux=integrate_table(table,actual)
    assert flux[1]-flux[0]==pytest.approx(epsilon,abs=2e-11)
    np.testing.assert_allclose(flux,[(threshold).mean(),(threshold+epsilon).mean(),(threshold-.01).mean(),(threshold+.01).mean()],atol=2e-11)
    np.testing.assert_allclose(np.bincount(table.base_index,weights=table.weight),.25,atol=1e-16)
    assert len(batches)<=8  # Count does not multiply by16 independent roots.
    assert all(len(set(b))==4 for b,u in batches)
    assert all(r['bracket_width']<=1e-11*(1+1e-12) for r in solution.roots)
    assert solution.diagnostics['completeness']=='SAMPLED_BRACKETS_ONLY_NOT_CERTIFIED'


def test_shifted_circle_has_two_roots_and_keeps_both_outside_intervals():
    # Ray point=(sqrt(u),0); circular aperture centered at(.6,0), radius.15.
    def margin(base,u):return (.15**2-(np.sqrt(u)-.6)**2)[:,None]
    solution=solve_radial_partitions(margin,1,('shifted',),scan_intervals=16,u_tol=1e-11)
    np.testing.assert_allclose(solution.partitions[0].edges,[0,.45**2,.75**2,1],atol=1e-11)
    table=gauss_table(solution,order=4);passed=validate_node_membership(solution,table)[:,0]>=0
    assert integrate_table(table,passed)==pytest.approx(.75**2-.45**2,abs=2e-11)
    assert solution.partitions[0].diagnostics['sampled_transition_count']==[2]
    assert np.all(table.weight>0)  # Blocked cells also retain full source support.


def test_elliptic_aperture_known_probability_and_cs_like_radial_moment():
    angles=np.array([.2,.7,1.1]); ax,ay=.8,.5
    def margin(base,u):
        theta=angles[base]
        return (1-u*((np.cos(theta)/ax)**2+(np.sin(theta)/ay)**2))[:,None]
    solution=solve_radial_partitions(margin,3,('ellipse',),u_tol=1e-11)
    threshold=1/((np.cos(angles)/ax)**2+(np.sin(angles)/ay)**2)
    for order in (2,4):
        table=gauss_table(solution,order=order);passed=validate_node_membership(solution,table)[:,0]>=0
        u=table.angular_cdf;c=.3
        # Cs-like radial map rho=sqrt(u)*(1+c*u), so rho² is a cubic.
        second_moment=integrate_table(table,passed*(u+2*c*u*u+c*c*u**3))
        exact=np.mean(threshold**2/2+2*c*threshold**3/3+c*c*threshold**4/4)
        assert second_moment==pytest.approx(exact,abs=2e-11)
        assert integrate_table(table,passed)==pytest.approx(threshold.mean(),abs=2e-11)
        # A physical translation remains a first moment; no centroid alignment.
        first=integrate_table(table,passed*(.12+.2*u))
        assert first==pytest.approx(np.mean(.12*threshold+.1*threshold**2),abs=2e-11)


def test_multiple_pass_bands_are_not_reduced_to_one_threshold():
    roots=np.array([.11,.28,.57,.83])
    def margin(base,u):return (-np.prod(u[:,None]-roots,axis=1))[:,None]
    solution=solve_radial_partitions(margin,1,('multi',),scan_intervals=32,u_tol=1e-10)
    table=gauss_table(solution,order=4);passed=validate_node_membership(solution,table)[:,0]>=0
    assert integrate_table(table,passed)==pytest.approx((.28-.11)+(.83-.57),abs=4e-10)
    assert solution.partitions[0].diagnostics['multiple_pass_intervals']
    assert solution.partitions[0].diagnostics['pass_intervals_by_channel']==[2]


def test_duplicate_and_endpoint_roots_do_not_create_zero_measure_cells():
    def margin(base,u):return np.column_stack((u,1-u,.5-u,.5-u))
    solution=solve_radial_partitions(margin,1,('left','right','a','duplicate'),u_tol=1e-10)
    np.testing.assert_array_equal(solution.partitions[0].edges,[0,.5,1])
    table=gauss_table(solution)
    assert len(table.weight)==4
    assert table.weight.sum()==1


def test_unseen_narrow_pass_band_cannot_be_claimed_certified():
    def margin(base,u):return ((u-.37123)*(u-.37124))[:,None]
    solution=solve_radial_partitions(margin,1,('narrow',),scan_intervals=4,scan_refinements=0)
    assert solution.diagnostics['completeness']=='SAMPLED_BRACKETS_ONLY_NOT_CERTIFIED'
    assert solution.diagnostics['max_scan_gap']==.25
    with pytest.raises(BoundaryResolutionError,match='cannot certify'):
        solve_radial_partitions(margin,1,('narrow',),require_certified=True)


def test_actual_node_check_detects_a_missing_scan_interval():
    # A thin blocked interval contains a2point Gauss node but no scan node.
    node=(1-1/np.sqrt(3))/2
    def margin(base,u):return ((u-(node-.001))*(u-(node+.001)))[:,None]
    solution=solve_radial_partitions(margin,1,('hidden',),scan_intervals=2,scan_refinements=0)
    with pytest.raises(BoundaryResolutionError,match='missed boundary'):
        validate_node_membership(solution,gauss_table(solution))


def test_failed_roots_and_invalid_callbacks_are_rejected_without_silent_extra_iterations():
    def nonlinear(base,u):return (np.exp(12*u)-10)[:,None]
    with pytest.raises(BoundaryResolutionError,match='root solve failed'):
        solve_radial_partitions(nonlinear,1,('case',),scan_intervals=2,scan_refinements=0,maxiter=1,u_tol=1e-12)
    with pytest.raises(BoundaryResolutionError,match='nonfinite'):
        solve_radial_partitions(lambda b,u:np.full((len(u),1),np.nan),1,('case',))
    with pytest.raises(BoundaryResolutionError,match='Non-isolated'):
        solve_radial_partitions(lambda b,u:np.zeros((len(u),1)),1,('case',))


def test_order_four_integrates_degree_seven_per_cell_without_loss_renormalisation():
    def margin(base,u):return np.column_stack((.3-u,.7-u))
    solution=solve_radial_partitions(margin,2,('base','case'))
    table=gauss_table(solution,order=4)
    value=table.angular_cdf**7
    assert integrate_table(table,value)==pytest.approx(1/8,abs=1e-16)
    passed=validate_node_membership(solution,table)>=0
    weighted=integrate_table(table,passed*value[:,None])
    np.testing.assert_allclose(weighted,np.array([.3,.7])**8/8,atol=1e-13)


def test_heterogeneous_vector_root_convergence_preserves_active_base_and_channel_binding():
    roots=np.array([[.1234567,.4271828,.8123456],
                    [.2312345,.5182736,.7234567],
                    [.3412345,.6281736,.6345678]])
    calls=[]
    def margin(base,u):
        calls.append(list(zip(base.tolist(),u.tolist())))
        offset=u[:,None]-roots[base]
        return np.column_stack((offset[:,0],np.expm1(6*offset[:,1]),np.tanh(30*offset[:,2])))
    solution=solve_radial_partitions(margin,3,('linear','exponential','steep'),u_tol=1e-10,maxiter=12)
    assert len(solution.roots)==9
    for found in solution.roots:
        assert found['u']==pytest.approx(roots[found['base_index'],found['channel_index']],abs=1e-10)
        assert np.isfinite(found['actual_margin_residual'])
        assert found['bracket'][1]-found['bracket'][0]<=1e-10*(1+1e-12)
    assert len({found['iterations'] for found in solution.roots})>1
    assert len({len(call) for call in calls})>=3
    all_requested=[key for call in calls for key in call]
    assert len(all_requested)==len(set(all_requested))  # Actual callback never repeats cached endpoints.
    table=gauss_table(solution,order=4)
    passed=validate_node_membership(solution,table)>=0
    np.testing.assert_allclose(integrate_table(table,passed),1-roots.mean(axis=0),atol=3e-10)


def test_exact_scan_roots_and_overlapping_channels_keep_the_true_shared_partition():
    def margin(base,u):return np.column_stack((u,1-u,.25-u,u-.25,.75-u))
    solution=solve_radial_partitions(margin,2,('zero','one','left','left_reversed','right'),u_tol=1e-10)
    for part in solution.partitions:
        np.testing.assert_array_equal(part.edges,[0.,.25,.75,1.])
    assert all(r['actual_margin_residual']==0 for r in solution.roots)
    endpoint=[r for r in solution.roots if r['u'] in (0.,1.)]
    assert all(r['bracket_width']==0 for r in endpoint)
    table=gauss_table(solution)
    assert len(table.weight)==2*3*2
    np.testing.assert_allclose(integrate_table(table,validate_node_membership(solution,table)>=0),[1,1,.25,.75,.75],atol=1e-15)


@pytest.mark.parametrize('kwargs',[
    {'u_tol':True},{'u_tol':False},{'u_tol':np.nan},{'u_tol':'1e-8'},
    {'maxiter':True},{'maxiter':2.0},{'maxiter':0},
    {'scan_intervals':True},{'scan_refinements':False},
])
def test_invalid_boundary_controls_do_not_silently_coerce_booleans_or_counts(kwargs):
    with pytest.raises(ValueError,match='parameters'):
        solve_radial_partitions(lambda b,u:(.3-u)[:,None],1,('case',),**kwargs)


def test_gauss_order_and_callback_base_ids_are_strict():
    solution=solve_radial_partitions(lambda b,u:(.3-u)[:,None],1,('case',))
    for order in (True,2.,3):
        with pytest.raises(ValueError,match='Gauss nodes'):
            gauss_table(solution,order=order)
    with pytest.raises(ValueError,match='base IDs'):
        solution.evaluator(np.array([.5]),np.array([.25]))
