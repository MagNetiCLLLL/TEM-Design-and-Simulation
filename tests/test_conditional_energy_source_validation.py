"""Pure source-measure/contract tests only; no actual gun or column execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import qmc

from scripts import validate_conditional_source_convergence as collector
from scripts import source_energy_strata
ROOT=collector.ROOT
from scripts.source_radial_quadrature import solve_radial_partitions,gauss_table
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.tip_joint_sampling import phase_space_from_quantiles


@pytest.fixture(scope='module')
def emitter():
    return ColdFieldEmitter(0.,1.,1.,emission_energy_ev=.3,
        energy_spread_fwhm_ev=.3,young_decay_width_ev=.2,boersch_sigma_ev=.1,
        energy_half_range_ev=1.,angular_rms_mrad=1.,angular_cutoff_mrad=3.)


@pytest.mark.parametrize('k',[2,8,16])
def test_actual_source_measure_and_prefix(emitter,k):
    base=qmc.Sobol(5,scramble=True,seed=0,bits=30).random_base2(6)
    small=collector.compose_energy_orbit(emitter,base[:16],k,(.143,.149))
    large=collector.compose_energy_orbit(emitter,base,k,(.143,.149))
    n=len(small['points'])
    np.testing.assert_array_equal(small['points'],large['points'][:n])
    for key in small['mapping']:
        a=small['mapping'][key]; b=large['mapping'][key][:n]
        np.testing.assert_array_equal(a*16 if key=='base_probability' else a,
                                      b*64 if key=='base_probability' else b)
    p=large['mapping']['base_probability']
    np.testing.assert_allclose(np.bincount(large['mapping']['independent_base_index'],weights=p),1/64,
                               rtol=8*np.finfo(float).eps,atol=0)
    # Apply the unchanged production physical source map, independently of cuts.
    coords=np.column_stack((large['points'][:,:2],np.full(len(p),.5),large['points'][:,2:]))
    energy=emitter.emission_energy_ev+phase_space_from_quantiles(emitter,coords)[-1]
    layer=large['mapping']['energy_stratum_index']
    assert np.all(energy[layer==0]<.143)
    assert np.all((energy[layer==1]>=.143)&(energy[layer==1]<=.149))
    assert np.all(energy[layer==2]>.149)
    assert large['orbit_metadata']['independent_sobol_base_count']==64
    assert large['orbit_metadata']['expanded_radial_base_count']==len(p)


def test_gauss_reweight_is_full_original_measure_and_signs(emitter):
    base=qmc.Sobol(5,scramble=True,seed=2,bits=30).random_base2(2)
    plan=collector.compose_energy_orbit(emitter,base,8,(.143,.149))
    count=len(plan['points']); p=plan['mapping']['base_probability']
    cut=np.linspace(.2,.8,count)
    callback=lambda b,u: np.repeat((cut[b]-u)[:,None],8,axis=1)
    solution=solve_radial_partitions(callback,count,collector.CHANNELS,u_tol=1e-10)
    uniform=gauss_table(solution,order=2)
    weighted=collector.reweight_radial_table(uniform,p)
    np.testing.assert_array_equal(weighted.weight,uniform.weight*(count*p[uniform.base_index]))
    arrays=collector.quadrature_arrays(plan['points'],weighted)
    positive=len(weighted.weight)*2
    assert arrays['weights'][positive:].sum()==0
    np.testing.assert_array_equal(arrays['weights'][:positive:2],weighted.weight/2)
    np.testing.assert_array_equal(arrays['weights'][1:positive:2],weighted.weight/2)
    np.testing.assert_allclose(arrays['weights'].sum(),1.,rtol=0,atol=1e-15)
    np.testing.assert_allclose(np.bincount(arrays['base_index'],weights=arrays['weights']),p,rtol=0,atol=1e-16)
    got=np.sum(weighted.weight[weighted.expected_pass[:,0]])
    assert got==pytest.approx(np.dot(p,cut),abs=1e-10)
    assert abs(got-cut.mean())>1e-4


def test_containment_reports_actual_bad_base_channel():
    mapping=dict(energy_stratum_index=np.array([0,1,2]),independent_base_index=np.array([4,4,4]))
    anchors=np.ones((6,8)); anchors[5]=-1
    assert collector.sampled_energy_containment(anchors,mapping,collector.CHANNELS)['status']=='PASS_SAMPLED'
    anchors[1,2]=-.001
    anchors[5,6]=.002
    report=collector.sampled_energy_containment(anchors,mapping,collector.CHANNELS)
    assert report['status']=='FAIL_SAMPLED' and report['failure_count']==2
    assert report['failures'][0]['independent_base_index']==4
    assert report['failures'][0]['channel']==collector.CHANNELS[2]
    with pytest.raises(ValueError,match='every actual'):
        collector.sampled_energy_containment(anchors[:-1],mapping,collector.CHANNELS)


def test_weighted_root_budget_does_not_use_uniform_count():
    roots=[dict(base_index=0,channel_key='baseline:+1',bracket_width=1e-9),
           dict(base_index=1,channel_key='Cs_plus20mm:+1',bracket_width=2e-9)]
    result=collector.weighted_root_budget(roots,np.array([.99,.01]),collector.CHANNELS)
    assert result['difference_bound_by_case']['Cs_plus20mm']==pytest.approx((.99e-9+.02e-9)/2,rel=1e-15)


@pytest.mark.parametrize('within',[True,False])
def test_weighted_case_anchor_gate_occurs_before_screen(within,monkeypatch):
    mapping=dict(energy_stratum_index=np.array([0,1,2]),independent_base_index=np.array([0,0,0]))
    reference=np.ones((6,8)); reference[5]=-1.
    observed=[]
    tracer=SimpleNamespace(anchor_reference=reference,base_plan={'mapping':mapping},
        observe_anchors=lambda a:observed.append(a.copy()))
    monkeypatch.setattr(collector,'boundary_uncertainty',lambda solution,tracer:dict(
        anchor_drift_within_requested_u_tol=within,maximum_anchor_cdf_uncertainty_estimate=1e-13))
    anchors=reference[:,4:6].copy(); anchors[0]+=.01
    if within:
        report=collector.validate_weighted_case_anchors(tracer,object(),anchors,2,collector.CHANNELS)
        assert report['status']=='PASS_SAMPLED'
    else:
        with pytest.raises(ValueError,match='before screen transport'):
            collector.validate_weighted_case_anchors(tracer,object(),anchors,2,collector.CHANNELS)
    assert len(observed)==1
    np.testing.assert_array_equal(observed[0][:,:4],reference[:,:4])
    np.testing.assert_array_equal(observed[0][:,4:6],anchors)
    np.testing.assert_array_equal(observed[0][:,6:],reference[:,6:])


@pytest.mark.parametrize('args',[['--energy-band','nan','.149'],['--energy-band','.149','.143'],['--energy-band','.143','.143']])
def test_bad_energy_band_rejected_before_source_allocation(args):
    with pytest.raises(SystemExit):
        collector.arguments(args)


def test_distinct_groups_and_explicit_source_cap():
    assert collector.group_label(32,8,4,True)=='M32_E3_K8_G4'
    assert collector.group_label(32,8,4)=='M32_K8_G4'
    with pytest.raises(SystemExit):
        collector.arguments(['--base-count','16384','--azimuth-orbit-size','16','--energy-band','.143','.149'])


@pytest.mark.parametrize('scale',[0.,-1.,float('nan'),float('inf')])
def test_invalid_gun_scale_has_no_mutation(scale):
    gun=SimpleNamespace(trace_step_mm=.2,drift_step_mm=2.)
    with pytest.raises(ValueError,match='positive and finite'):
        collector.apply_gun_step_scale(gun,scale)
    assert (gun.trace_step_mm,gun.drift_step_mm)==(.2,2.)
    with pytest.raises(SystemExit):
        collector.arguments(['--gun-step-scale',str(scale)])


def test_gun_step_scale_default_and_actual_snapshot_roundtrip():
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state=default_state()
    before=capture_instrument_snapshot(state)
    unchanged=collector.apply_gun_step_scale(state.electron_gun,1.)
    assert unchanged['captured']==unchanged['applied']
    assert capture_instrument_snapshot(state).digest==before.digest
    column_step=state.step_mm
    scaled=collector.apply_gun_step_scale(state.electron_gun,.25)
    snapshot=capture_instrument_snapshot(state)
    restored=snapshot.restore()
    for key,value in scaled['applied'].items():
        assert getattr(restored.electron_gun,key)==value==scaled['captured'][key]*.25
    assert restored.step_mm==column_step
    assert snapshot.digest!=before.digest


def test_prepare_archives_exact_energy_measure_gun_steps_and_rejects_band_relabel(tmp_path,monkeypatch):
    import os
    # main() owns CLI process settings; register every overwritten key so a
    # mocked prepare does not change the already initialized test-runner pool.
    for name in ('TEMSIM_CPU_THREADS','NUMBA_NUM_THREADS','OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','QT_QPA_PLATFORM'):
        monkeypatch.setenv(name,os.environ.get(name,'offscreen' if name=='QT_QPA_PLATFORM' else '1'))
    from temsim.optics.column import default_state
    from temsim.instrument_snapshot import capture_instrument_snapshot,InstrumentSnapshot
    state=default_state()
    original=capture_instrument_snapshot(state)
    scratch=tmp_path/'isolated-prepare'; (scratch/'scripts').mkdir(parents=True)
    for name in ('source_radial_quadrature.py','source_azimuth_orbit.py','validate_image_intensity_convergence.py'):
        (scratch/'scripts'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    (scratch/'scripts/source_energy_strata.py').write_bytes(Path(source_energy_strata.__file__).read_bytes())
    monkeypatch.setattr(collector,'ROOT',scratch)
    monkeypatch.setattr(collector,'configure_inputs',lambda *_args:(original.restore(),{},8.))
    output=scratch/'evidence'
    parse=collector.arguments
    cli=['--stage','prepare','--base-count','4','--azimuth-orbit-size','8','--energy-band','.143','.149',
         '--gun-step-scale','.5','--backend','CPU','--threads','1','--output',str(output)]
    monkeypatch.setattr(collector,'arguments',lambda:parse(cli))
    collector.main()
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['expanded_base_count']==48 and manifest['base_count']==4
    assert manifest['protocol']['gauss_validation_execution']=='fresh-weighted-gun-C2-membership-and-anchors-before-screen-v1'
    assert manifest['physical_protocol']['gun_integration']['scale']==.5
    saved=InstrumentSnapshot.from_dict(json.loads((output/'base-input.snapshot.json').read_text())).restore()
    assert saved.electron_gun.trace_step_mm==state.electron_gun.trace_step_mm*.5
    with np.load(output/manifest['source_base_measure_path']) as data:
        assert set(data.files)=={'base_probability','independent_base_index','energy_stratum_index','half_orbit_index','interval_lower','interval_upper'}
        np.testing.assert_allclose(data['base_probability'].sum(),1.,rtol=0,atol=1e-15)
    cli.append('--resume')
    collector.main()
    manifest=json.loads((output/'manifest.json').read_text())
    manifest['energy_stratification']['band_ev']=[.142,.150]
    (output/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='metadata differs'):
        collector.main()
