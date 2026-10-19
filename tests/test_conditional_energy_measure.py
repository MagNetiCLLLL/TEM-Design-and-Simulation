import numpy as np
import pytest
from scripts.analyse_conditional_source_convergence import expected_energy_orbit,expected_source_table,verify_source_table,weighted_root_probability


def test_conditional_cdf_full_measure_and_zero_strata_preserve_original_base_support():
    base=np.array([[.2,.3,.4,.5,.25],[.7,.8,.9,.1,.75]])
    cuts=np.array([[0.,.1,.4,1.],[0.,0.,1.,1.]])
    e=expected_energy_orbit(base,cuts,8)
    assert len(e['points'])==16
    np.testing.assert_allclose(np.bincount(e['independent_base_index'],weights=e['base_probability']),[.5,.5],atol=1e-16)
    np.testing.assert_array_equal(e['points'][:,3],base[e['independent_base_index'],3])
    expected=e['interval_lower']+(e['interval_upper']-e['interval_lower'])*base[e['independent_base_index'],4]
    np.testing.assert_array_equal(e['points'][:,4],expected)
    prefix=expected_energy_orbit(base[:1],cuts[:1],8)
    np.testing.assert_array_equal(prefix['points'],e['points'][:12])
    np.testing.assert_allclose(prefix['base_probability']/2,e['base_probability'][:12],rtol=0,atol=0)


def test_weighted_root_bound_assigns_small_stratum_actual_mass_not_uniform_count():
    roots=[dict(base_index=0,channel_key='baseline:+1',bracket=[.2,.2001]),
           dict(base_index=1,channel_key='A1:-1',bracket=[.5,.5001])]
    result=weighted_root_probability(roots,[.001,.999],['baseline','A1'])
    assert result['per_case']['baseline']==pytest.approx(.001*.0001/2)
    assert result['per_case']['A1']==pytest.approx(.999*.0001/2)
    assert result['case_minus_baseline']['A1']==pytest.approx(.0001/2)


@pytest.mark.parametrize('order',[2,4])
def test_source_table_preserves_nonuniform_probability_and_analytic_radial_integrals(order):
    points=np.full((2,5),.25);probability=np.array([.03,.97])
    parts=[dict(base_index=0,edges=[0.,.1,.9,1.]),dict(base_index=1,edges=[0.,.5,1.])]
    table=expected_source_table(parts,points,probability,order)
    received=table['angular_cdf']<=np.where(table['base_index']==0,.1,.5)
    assert table['weights']@received==pytest.approx(.03*.1+.97*.5,abs=2e-16)
    assert table['weights']@(table['angular_cdf']**3)==pytest.approx(.25,abs=2e-16)
    np.testing.assert_allclose(np.bincount(table['base_index'],weights=table['weights']),probability,atol=2e-16)
    assert verify_source_table(table,table)['zero_probe_nodes']==8
    changed={key:value.copy() for key,value in table.items()};changed['weights'][0]*=2
    with pytest.raises(ValueError,match='source measure: weights'):verify_source_table(changed,table)


@pytest.mark.parametrize('badcuts',[[[0.,.8,.2,1.]],[[.01,.2,.8,1.]],[[0.,.2,.8,.99]]])
def test_noncomplete_or_reversed_source_partition_rejected(badcuts):
    with pytest.raises(ValueError):expected_energy_orbit(np.full((1,5),.5),badcuts,8)


from dataclasses import asdict
from hashlib import sha256
import numpy as np
import pytest
from scipy.stats import qmc
from scripts.source_energy_strata import expand_energy_strata
from scripts.validate_conditional_source_convergence import compose_energy_orbit,weighted_root_budget
from scripts.analyse_conditional_source_convergence import expected_source_table
from scripts.analyse_conditional_source_convergence import (ENERGY_FIELDS,load_energy_measure,source_method,
    validate_energy_source_table,validate_weighted_roots,require_same_refinement_method)
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from scripts.analyse_image_intensity_convergence import CASES
from scripts.validate_image_intensity_convergence import array_digest


def archive(tmp_path):
    emitter=ColdFieldEmitter(0.,1.,1.)
    bases=qmc.Sobol(5,scramble=True,seed=3,bits=30).random_base2(1)
    plan=compose_energy_orbit(emitter,bases,8,(.143,.149))
    d=dict(azimuth_orbit_size=8,expanded_base_count=len(plan['points']),
        energy_stratification=plan['energy_metadata'],energy_cuts_metadata=plan['cuts_metadata'],
        physical_protocol=dict(energy_stratification=dict(schema=plan['energy_metadata']['schema'],
            band_ev=[.143,.149],cuts_metadata=plan['cuts_metadata'])))
    for prefix,values in [('energy_cuts',plan['cuts']),('source_base_measure',plan['mapping'])]:
        path=tmp_path/(prefix+'.npz');np.savez(path,**values)
        d[prefix+'_path']=path.name;d[prefix+'_sha256']=sha256(path.read_bytes()).hexdigest();d[prefix+'_digest']=array_digest(values)
    snapshot=dict(graph=dict(nodes=[dict(type='temsim.optics.electron_gun.emitter:ColdFieldEmitter',
        attributes={name:getattr(emitter,name) for name in ENERGY_FIELDS})]))
    return d,bases,plan,snapshot


def test_energy_archive_reconstructs_captured_source_measure_and_nonuniform_node_weights(tmp_path):
    d,bases,plan,snapshot=archive(tmp_path)
    result=load_energy_measure(tmp_path,d,bases,plan['points'],snapshot)
    p=plan['mapping']['base_probability'];r=len(p)
    parts=[dict(base_index=i,edges=[0.,.25,.8,1.]) for i in range(r)]
    table=expected_source_table(parts,plan['points'],p,4)
    table.update({key:value[table['base_index']] for key,value in plan['mapping'].items()})
    receipt=validate_energy_source_table(table,dict(partitions=parts),plan['points'],plan['mapping'],4)
    assert receipt['zero_probe_nodes']==4*r
    assert result['receipt']['probability']==pytest.approx(1.)
    assert np.ptp(p)>0
    # Explicit unequal changes preserve total one, so this tests per-node
    # provenance rather than merely a sum-of-weights smoke check.
    table['weights'][0]+=1e-8;table['weights'][2]-=1e-8
    with pytest.raises(ValueError,match='source measure: weights'):
        validate_energy_source_table(table,dict(partitions=parts),plan['points'],plan['mapping'],4)


def test_rehashed_cut_tampering_still_rejected_against_actual_energy_law(tmp_path):
    d,bases,plan,snapshot=archive(tmp_path)
    cuts={key:value.copy() for key,value in plan['cuts'].items()}
    cuts['edges'][0,1]+=1e-6;cuts['interval_probability']=np.diff(cuts['edges'],axis=1)
    path=tmp_path/d['energy_cuts_path'];np.savez(path,**cuts)
    d['energy_cuts_sha256']=sha256(path.read_bytes()).hexdigest();d['energy_cuts_digest']=array_digest(cuts)
    with pytest.raises(ValueError,match='captured physical source law'):
        load_energy_measure(tmp_path,d,bases,plan['points'],snapshot)


def test_actual_emitter_parameter_change_cannot_hide_behind_archived_calibration(tmp_path):
    d,bases,plan,snapshot=archive(tmp_path)
    snapshot['graph']['nodes'][0]['attributes']['emission_energy_ev']+=.001
    with pytest.raises(ValueError,match='actual captured emitter calibration'):
        load_energy_measure(tmp_path,d,bases,plan['points'],snapshot)


def test_weighted_root_budget_retains_existing_convergence_checks(tmp_path):
    d,bases,plan,snapshot=archive(tmp_path)
    lo=.25;hi=.25+1e-12
    root=dict(base_index=0,channel_key='A1_plus0.03mm:+1',u=lo,bracket=[lo,hi],
        bracket_width=hi-lo,actual_margin_residual=0.)
    channels=[f'{case}:{sign:+d}' for case in CASES for sign in (1,-1)]
    partition=dict(roots=[root],weighted_root_budget=weighted_root_budget([root],plan['mapping']['base_probability'],channels))
    diagnostics=dict(all_located_roots_converged=True,root_count=1,angular_cdf_tolerance=2e-12)
    result=validate_weighted_roots(partition,diagnostics,plan['mapping'])
    assert result['case_minus_baseline']['A1_plus0.03mm']==plan['mapping']['base_probability'][0]*(hi-lo)/2
    diagnostics['all_located_roots_converged']=False
    with pytest.raises(ValueError,match='all converge'):validate_weighted_roots(partition,diagnostics,plan['mapping'])


def test_refinement_method_separates_uniform_and_distinct_bands(tmp_path):
    d,_,_,_=archive(tmp_path);energy=source_method(d);uniform=source_method({})
    with pytest.raises(ValueError,match='cannot mix'):require_same_refinement_method(energy,uniform)
    same=source_method(d);require_same_refinement_method(energy,same)
    changed=dict(energy,identity=energy['identity']+' changed fixed band')
    with pytest.raises(ValueError,match='different fixed energy'):require_same_refinement_method(energy,changed)
