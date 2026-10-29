"""Synthetic acceptance checks only; no source or field propagation."""
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pytest

from temsim.optics.direct_focus_topology import evaluate


def drift(root, *, extra=(), fixed_y=False):
    end=10.
    z=np.unique(np.r_[np.arange(-.25,end,.25),0.,end-1e-6,end,
                      end-np.array([.8,.6,.4,.2,.1])*1e-6,extra])
    phi=np.arange(16)*2*np.pi/16
    tx=np.broadcast_to(.002*np.cos(phi),(len(z),len(phi))).copy()
    ty=np.broadcast_to(.002*np.sin(phi),(len(z),len(phi))).copy()
    delta=(z-root)[:,None]*1e-3
    cp=SimpleNamespace(z_mm=z,x_m=delta*tx+3e-6,y_m=delta*ty-4e-6,tx_rad=tx,ty_rad=ty)
    if fixed_y:
        cp.y_m[:]=np.linspace(-1e-6,1e-6,len(phi));cp.ty_rad[:]=0
    return cp,np.ones(tx.shape,bool),np.linspace(1.,2.,len(phi))


def gate(data, **kwargs):
    return evaluate(*data,lower_z_mm=0.,surface_z_mm=10.,
                    surface_focus_tolerance_nm=1.,maximum_spacing_mm=.5,**kwargs)


def test_extra_executed_waist_rejects_even_with_an_acceptable_final_probe_elsewhere():
    # This isolates the forbidden intermediate crossing of the old 69-percent
    # branch. The gate must not depend on a diameter/current/focus success flag.
    data=drift(7.,extra=[6.99,7.,7.01])
    report=gate(data)
    assert report['status']=='FAIL' and not report['passed']
    assert report['failures']==['executed_pre_specimen_waist_outside_focus_window']
    assert report['forbidden_waists'][0]['z_mm']==7.


def test_final_endpoint_is_separate_from_absence_of_an_extra_waist():
    report=gate(drift(10.))
    assert report['passed']
    assert report['surface_focus_qualified'] is False


def test_terminal_candidate_requires_entire_bracket_inside_explicit_one_nm_window():
    root=10.-.5e-6
    report=gate(drift(root))
    assert report['passed'] and report['terminal_candidates']
    data=drift(root)
    keep=(data[0].z_mm<10.-1e-6)|(data[0].z_mm==10.-1e-6)|(data[0].z_mm==10.)
    sparse=SimpleNamespace(**{key:value[keep] for key,value in vars(data[0]).items()})
    # The whole 1 nm bracket still fits the explicitly allowed interval.
    assert gate((sparse,data[1][keep],data[2]))['passed']


def finite_terminal_drift(depth_nm, split_nm):
    """Analytical astigmatic drift with independently placed X/Y waists."""
    end = 10.
    z = np.unique(np.r_[np.arange(-.25, end, .25),
        end-np.arange(0., 2.1, .1)*1e-6, end-.0002, end-.0001, end])
    phi = np.arange(32)*2*np.pi/32
    tx, ty = .002*np.cos(phi), .002*np.sin(phi)
    root, split = end-depth_nm*1e-6, split_nm*1e-6
    x = (z-root+split/2)[:, None]*1e-3*tx+1e-14*np.cos(2*phi)
    y = (z-root-split/2)[:, None]*1e-3*ty+1e-14*np.sin(2*phi)
    cp = SimpleNamespace(z_mm=z, x_m=x, y_m=y,
        tx_rad=np.broadcast_to(tx, x.shape), ty_rad=np.broadcast_to(ty, x.shape))
    state = SimpleNamespace(simulation_mode='unified', probe_mode='nano_probe',
        mini_condenser=SimpleNamespace(enabled=True, installed=True,
            mechanical_center_from_tip_mm=1., mechanical_length_mm=2.),
        electron_gun=SimpleNamespace(exit_plane_z_mm=-1.),
        sample=SimpleNamespace(upper_surface_z_mm=end))
    return state, cp, np.ones(x.shape, bool), np.ones(len(phi))


def test_default_finite_terminal_gate_rejects_wide_region_despite_surface_focus():
    from temsim.optics.direct_focus_topology import evaluate_for_state
    from temsim.optics.surface_probe_focus import SurfaceFocusMeasurement, local_waist_from_radii
    from temsim.physics.beam_statistics import transverse_beam_statistics

    state, cp, masks, weights = finite_terminal_drift(.9, .8)
    before = deepcopy((cp, masks, weights))
    planes = [10.-.0002, 10.-.0001, 10.]
    rows = [transverse_beam_statistics(
        *(getattr(cp, key)[i] for key in ('x_m', 'y_m', 'tx_rad', 'ty_rad')),
        alive=masks[i], weights=weights) for i in np.searchsorted(cp.z_mm, planes)]
    radii = tuple(row.radius_rms_m for row in rows)
    offset, curvature = local_waist_from_radii(radii, 1e-7)
    measurement = SurfaceFocusMeasurement(10., rows[-1], offset, curvature, radii, .05, 32.)
    assert measurement.accepts(2.)
    assert offset == pytest.approx(-.9, abs=1e-6)

    report = evaluate_for_state(state, cp, masks, weights)
    assert report['waist_detector'] == 'finite_region'
    region, = report['executed_waists']
    np.testing.assert_allclose(region['bracket_mm'], [9.9999983, 9.9999999], rtol=0, atol=1e-12)
    assert report['terminal_candidates']  # The narrower radial bracket alone passes.
    assert report['status'] == 'UNRESOLVED' and not report['passed']
    assert region in report['unresolved_candidates']
    assert report['refinement_planes_mm']
    assert all(region['bracket_mm'][0] < z < region['bracket_mm'][1]
               for z in report['refinement_planes_mm'])
    for key, value in vars(cp).items():
        np.testing.assert_array_equal(value, getattr(before[0], key))
    np.testing.assert_array_equal(masks, before[1])
    np.testing.assert_array_equal(weights, before[2])


@pytest.mark.parametrize('depth_nm,split_nm,touches_boundary', [(.5, .2, False), (.6, .3, True)])
def test_contained_finite_terminal_region_remains_accepted(depth_nm, split_nm, touches_boundary):
    from temsim.optics.direct_focus_topology import evaluate_for_state
    report = evaluate_for_state(*finite_terminal_drift(depth_nm, split_nm))
    region, = report['executed_waists']
    left, right = region['bracket_mm']
    start, end = report['terminal_window_mm']
    assert start <= left < right <= end
    if touches_boundary:
        assert left == start
    assert report['status'] == 'PASS' and report['passed']
    assert not report['unresolved_candidates']


def test_finite_terminal_endpoint_remains_a_separate_one_sided_observation():
    from temsim.optics.direct_focus_topology import evaluate_for_state
    report = evaluate_for_state(*finite_terminal_drift(0., 0.))
    assert not report['executed_waists']
    assert report['finite_region_evidence']['status'] == 'UNRESOLVED'
    assert report['passed'] and report['surface_focus_qualified'] is False


@pytest.mark.parametrize('damage',['missing_end','missing_guard','missing_window','wide_gap'])
def test_missing_or_sparse_execution_cannot_claim_absence(damage):
    cp,masks,weights=drift(10.)
    keep=np.ones(len(cp.z_mm),bool)
    if damage=='missing_end':keep[-1]=False
    elif damage=='missing_guard':keep[cp.z_mm<0]=False
    elif damage=='missing_window':keep[cp.z_mm==10.-1e-6]=False
    else:keep[(cp.z_mm>4)&(cp.z_mm<6)]=False
    cp=SimpleNamespace(**{key:value[keep] for key,value in vars(cp).items()})
    report=gate((cp,masks[keep],weights))
    assert report['status']=='UNRESOLVED' and not report['passed']


def test_outside_tolerance_radial_candidate_is_not_hidden_by_a_500_nm_search_zone():
    root=10.-100e-6
    report=gate(drift(root,fixed_y=True))
    assert not report['passed'] and report['status']=='UNRESOLVED'
    assert report['unresolved_candidates'] and report['refinement_planes_mm']


def test_downstream_loss_does_not_erase_an_upstream_forbidden_waist():
    cp,masks,weights=drift(7.,extra=[6.99,7.,7.01])
    masks[cp.z_mm>8,0]=False
    report=gate((cp,masks,weights))
    assert report['status']=='FAIL' and report['forbidden_waists']
    assert report['forbidden_waists'][0]['source_current_fraction']==pytest.approx(1.)


def test_population_change_without_a_resolved_waist_remains_unresolved():
    cp,masks,weights=drift(10.)
    masks[cp.z_mm>8,0]=False
    assert gate((cp,masks,weights))['status']=='UNRESOLVED'


def test_qualification_observer_does_not_refocus_recenter_or_reweight():
    data=drift(7.,extra=[6.99,7.,7.01]);before=deepcopy(data)
    gate(data)
    for key,value in vars(data[0]).items():np.testing.assert_array_equal(value,getattr(before[0],key))
    np.testing.assert_array_equal(data[1],before[1]);np.testing.assert_array_equal(data[2],before[2])


def test_no_new_hard_constraint_on_internal_directional_split():
    # Only the Mini-body-to-surface interval belongs to this gate. A mixed
    # directional phase upstream of its entrance cannot create a rejection.
    cp,masks,weights=drift(10.)
    cp.x_m[0]=np.linspace(-1e-3,1e-3,len(weights));cp.y_m[0]=cp.x_m[0]*.5
    # The entrance guard is used for coverage; an entirely upstream feature
    # is not counted as a Mini-to-surface crossover.
    report=gate((cp,masks,weights))
    assert report['passed']


def test_existing_condenser_gate_rejects_old_branch_for_joint_alignment_callers(monkeypatch):
    from temsim.optics import condenser_topology
    lens=lambda key,z:SimpleNamespace(key=key,z_mm=z,enabled=True,installed=True)
    state=SimpleNamespace(simulation_mode='unified',probe_mode='nano_probe',nanoprobe_range='normal',
        lenses=[lens('condenser_lens_1',.2),lens('condenser_lens_2',1.),lens('condenser_lens_3',2.),lens('adapter_lens',3.),
                lens('probe_tl22_lens',3.5),lens('probe_tl21_lens',4.5),lens('mini_condenser',7.)],
        mini_condenser=SimpleNamespace(enabled=True,installed=True,mechanical_center_from_tip_mm=7.,mechanical_length_mm=1.),
        electron_gun=SimpleNamespace(exit_plane_z_mm=0.),sample=SimpleNamespace(upper_surface_z_mm=10.))
    monkeypatch.setattr('temsim.physics.instrument_magnetic.column_dipole_fields',lambda state:[
        SimpleNamespace(key=f'ac_deflector:{i}',arrival_z_mm=z,dynamic=False) for i,z in enumerate((5.,6.))])
    z=np.unique(np.r_[condenser_topology.requested_capture_planes(state,maximum_spacing_mm=.5),
                      1.49,1.5,1.51,7.99,8.,8.01])
    polynomial=np.poly1d(np.poly([1.5,8.]))
    r=polynomial(z)*1e-8;t=np.polyder(polynomial)(z)*1e-5
    phi=np.arange(16)*2*np.pi/16
    cp=SimpleNamespace(z_mm=z,x_m=r[:,None]*np.cos(phi),y_m=r[:,None]*np.sin(phi),
        tx_rad=t[:,None]*np.cos(phi),ty_rad=t[:,None]*np.sin(phi))
    report=condenser_topology.evaluate(state,cp,np.ones(cp.x_m.shape,bool),np.ones(16),maximum_spacing_mm=.5)
    assert report['intervals']['c2_c3']['waists']
    assert not report['intervals']['c3_adl']['waists']
    assert report['direct_focus']['status']=='FAIL'
    assert report['status']=='FAIL' and report['passed'] is False
    # Both final joint-alignment refinement and final direct-alignment forward
    # validation already require this very report's status to equal PASS.
