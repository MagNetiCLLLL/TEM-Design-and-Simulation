"""Real explicit source emission, mocked transport: ordering and archive contracts only."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import qmc

from scripts import validate_conditional_source_convergence as collector
from scripts.source_radial_quadrature import solve_radial_partitions,gauss_table
from temsim.optics.electron_gun.emitter import ColdFieldEmitter
from temsim.optics.electron_gun.tip_joint_sampling import phase_space_from_quantiles


@pytest.fixture
def fixture(monkeypatch,tmp_path):
    import temsim.instrument_snapshot as snapshots
    import temsim.particle_section_io as section_io
    import temsim.simulation_pipeline as pipeline
    emitter=ColdFieldEmitter(0.,1.,1.)
    base=qmc.Sobol(5,scramble=True,seed=0,bits=30).random_base2(2)
    plan=collector.compose_energy_orbit(emitter,base,2,(.143,.149))
    layer=plan['mapping']['energy_stratum_index']
    cut=np.choose(layer,[1.5,.7,-.5])
    callback=lambda b,u:np.repeat((.02*(cut[b]-u))[:,None],8,axis=1)
    solution=solve_radial_partitions(callback,len(layer),collector.CHANNELS,u_tol=1e-10)
    table=collector.reweight_radial_table(gauss_table(solution,order=2),plan['mapping']['base_probability'])
    state=SimpleNamespace(electron_gun=SimpleNamespace(emitter=emitter,exit_plane_z_mm=450.),
        lenses=[SimpleNamespace(key='condenser_lens_2',cs_mm=8.,percent=20.)],
        lens_intrinsic_aberrations={},fluorescent_screen=SimpleNamespace(z_mm=1500.),
        condenser_aperture_2=SimpleNamespace(key='condenser_aperture_2',z_mm=765.,radius_mm=.05,
            offset_x_mm=0.,offset_y_mm=0.))

    class Snapshot:
        def __init__(self,s):
            self.s=deepcopy(s)
            t=s.electron_gun.emitter.cdf_quadrature
            self.digest=collector.canonical_digest(dict(table=None if t is None else t.digest,
                cs=s.lenses[0].cs_mm,intrinsic=s.lens_intrinsic_aberrations))
        def restore(self): return deepcopy(self.s)
        def to_dict(self): return dict(mock_transport_only=True,digest=self.digest)

    monkeypatch.setattr(snapshots,'capture_instrument_snapshot',Snapshot)
    calls=[]; archives={}; damage={'kind':None}

    def calculate(s,target,existing_result=None,**kwargs):
        calls.append(float(target))
        e=s.electron_gun.emitter; emitted=e.emit()
        coords=np.array(e.cdf_quadrature.coordinates)
        energy=e.emission_energy_ev+phase_space_from_quantiles(e,coords)[-1]
        strata=np.where(energy<.143,0,np.where(energy<=.149,1,2))
        margin=.02*(np.choose(strata,[1.5,.7,-.5])-coords[:,2])
        if target==765. and damage['kind']=='membership':
            index=np.flatnonzero((emitted.weight>0)&(margin>=0))[0]; margin[index]=-abs(margin[index])
        if target==765. and damage['kind']=='containment':
            index=np.flatnonzero((emitted.weight==0)&(strata==0)&(coords[:,2]==1.))[0]
            margin[index]=-abs(margin[index])
        passed=margin>=0; alive=passed.copy(); blocked=np.where(passed,'','condenser_aperture_2')
        if target==1500.:
            alive[0]=False; blocked[0]='downstream_stop'
        end=dict(x_m=(.05-margin)*1e-3,y_m=np.zeros(len(coords)),tx_rad=emitted.tx_rad,ty_rad=emitted.ty_rad,
            kinetic_energy_ev=np.full(len(coords),300000.),flight_time_s=np.full(len(coords),1e-8),
            source_id=emitted.ray_id,weight=emitted.weight,alive=alive,blocked_key=blocked,
            blocked_z_mm=np.where(passed,np.nan,765.))
        original=dict(gun_exit_ray_id=emitted.ray_id,gun_exit_weight=emitted.weight,gun_exit_alive=np.ones(len(coords),bool))
        checkpoint=SimpleNamespace(z_mm=np.array([target]),**{k:v[None,:] for k,v in end.items()
            if k in ('x_m','y_m','tx_rad','ty_rad','kinetic_energy_ev','flight_time_s')})
        branch=SimpleNamespace(source_ray_id=emitted.ray_id,ray_weight=emitted.weight,alive=alive,
            blocked_key=blocked,blocked_z=end['blocked_z_mm'])
        simulation=SimpleNamespace(gun_trace=SimpleNamespace(emitted_current_a=1e-5),
            metrics={'section_gun_reused':existing_result is not None},
            section_checkpoint=SimpleNamespace(segments=[SimpleNamespace(checkpoints=checkpoint,branch=branch)]))
        return SimpleNamespace(source=original,end=end,simulation=simulation)

    def save(result,path,**kwargs):
        path.write_bytes(b'Mocked transport archive only'); archives[str(path)]=result
    monkeypatch.setattr(pipeline,'calculate_particle_section',calculate)
    monkeypatch.setattr(section_io,'save_section_result',save)
    monkeypatch.setattr(section_io,'load_section_result',lambda p:archives[str(p)])
    monkeypatch.setattr(collector,'source_arrays',lambda r:r.source)
    monkeypatch.setattr(collector,'screen_arrays',lambda r,*a:dict(r.end,survivor=r.end['alive']))
    anchors=callback(np.repeat(np.arange(len(layer)),2),np.tile([0.,1.],len(layer)))
    tracer=SimpleNamespace(snapshot=Snapshot(state),base_points=plan['points'],base_plan=plan,output=tmp_path,
        manifest=dict(protocol_digest='mock-only',implementation='mock-only',sources={},runs={},base_count=4,
            expanded_base_count=len(layer),azimuth_orbit_size=2,energy_stratification=plan['energy_metadata']),
        controls={},c2_cs=8.,guard=lambda:None,anchor_reference=anchors,anchor_drift=np.zeros_like(anchors))
    tracer.observe_anchors=lambda a:collector.PhysicalBoundaryTracer.observe_anchors(tracer,a)
    return tracer,solution,table,calls,damage


def test_one_fresh_weighted_gun_checks_c2_before_every_screen_and_resumes(fixture):
    tracer,solution,table,calls,_=fixture
    collector.collect_screen_group(tracer,solution,table,2,'mock-partition')
    assert calls==[450.01]+[765.,1500.]*4
    rows=tracer.manifest['runs']; assert len(rows)==4
    for row in rows.values():
        assert row['weighted_source_containment']['status']=='PASS_SAMPLED'
    stored=tracer.manifest['sources']['M4_E3_K2_G2']
    with np.load(tracer.output/stored['table_path']) as data:
        for name,values in tracer.base_plan['mapping'].items():
            np.testing.assert_array_equal(data[name],values[data['base_index']])
        expected=collector.quadrature_arrays(tracer.base_points,table)
        np.testing.assert_array_equal(data['weights'],expected['weights'])
    collector.collect_screen_group(tracer,solution,table,2,'mock-partition')
    assert len(calls)==9


@pytest.mark.parametrize('damage_kind,match',[('membership','membership'),('containment','energy band'),('drift','before screen transport')])
def test_failed_actual_c2_gate_never_starts_screen(fixture,monkeypatch,damage_kind,match):
    tracer,solution,table,calls,damage=fixture
    damage['kind']=damage_kind
    if damage_kind=='drift':
        monkeypatch.setattr(collector,'boundary_uncertainty',lambda *a:dict(anchor_drift_within_requested_u_tol=False))
    with pytest.raises(ValueError,match=match):
        collector.collect_screen_group(tracer,solution,table,2,'mock-partition')
    assert calls==[450.01,765.]
