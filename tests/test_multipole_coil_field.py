"""Pure static field/current checks; no source particles or coherent work."""
from copy import copy
from dataclasses import replace
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import mu_0

from temsim.multipole_current import (
    MATRIX_FIELD, BIAS_FIELD, REFERENCE_ORIENTATION_FIELD, validate_multipole_current_inputs,
)
from temsim.multipole_winding import (
    BASIS_FIELD, STATUS_FIELD, NOTE_FIELD, STATUS_VALUE, SOURCE_NOTE,
    multipole_winding_envelopes, turn_field,
)
from temsim.physics.transverse_coil_field import TransverseCoilPotential, FrozenTransverseCoilField
from temsim.physics.multipole_coil_field import (
    magnetic_field_hessian_t_per_m2, multipole_field_coefficients,
    recommended_basis_angle_deg, calibrate_multipole_current_matrix,
    capture_multipole_coil_fields,
)


@pytest.fixture(autouse=True)
def no_particle_or_wave(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('Multipole field tests must not emit or transport particles')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


def rectangle(**changes):
    return dict(family='radial_rectangle',coil_index=1,station_index=0,group='x',current_sign=1,
        azimuth_deg=0.,center_z_mm=0.,radial_start_mm=19.9,radial_end_mm=20.1,
        outer_width_mm=24.2,outer_height_mm=24.2,window_width_mm=24.,window_height_mm=24.,**changes)


def test_second_B_derivative_matches_independent_square_loop_formula():
    potential=TransverseCoilPotential('square',(rectangle(),),7,(1.,0.),pack_order=12,path_order=24)
    point=np.array([[.020,0.,0.]])
    actual=magnetic_field_hessian_t_per_m2(potential,point)[0,0,0,0]
    # Exact square-loop axis field, differentiated analytically and averaged
    # by an independent 48x48 rule over the source pack.
    x,w=np.polynomial.legendre.leggauss(48)
    offsets=.020-(.020+.0001*x[:,None])
    a=(24.+.2*(x[None,:]+1)/2)*.0005
    a2,t2=a*a,offsets*offsets
    value=7*2*mu_0*a2/(np.pi*(a2+t2)*np.sqrt(2*a2+t2))
    logarithmic_first=-2*offsets/(a2+t2)-offsets/(2*a2+t2)
    logarithmic_second=-2*(a2-t2)/(a2+t2)**2-(2*a2-t2)/(2*a2+t2)**2
    expected=np.sum(value*(logarithmic_first**2+logarithmic_second)*w[:,None]*w[None,:]/4)
    np.testing.assert_allclose(actual,expected,rtol=5e-12,atol=1e-13)


def test_B_hessian_obeys_symmetry_vacuum_laplacian_and_rigid_rotation():
    from temsim.physics.lens_field_provider import CoordinateRegistration
    source=TransverseCoilPotential('square',(rectangle(),),7,(.7,0.))
    points=np.array([[.0195,.001,.002],[.0205,-.002,.001]])
    derivative=magnetic_field_hessian_t_per_m2(source,points)
    np.testing.assert_array_equal(derivative,derivative.swapaxes(-1,-2))
    np.testing.assert_allclose(np.trace(derivative,axis1=-2,axis2=-1),0.,atol=1e-10)
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    registration=CoordinateRegistration(origin_global_m=(.01,.02,.03),rotation_local_to_global=tuple(map(tuple,rotation)))
    posed=replace(source,registration=registration)
    actual=magnetic_field_hessian_t_per_m2(posed,points@rotation.T+registration.origin_array_m)
    expected=np.einsum('ai,pijk,bj,ck->pabc',rotation,derivative,rotation,rotation)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-11)


@pytest.fixture(scope='module')
def installed_state():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    state=default_state();catalog=AssemblyCatalog();catalog.apply(state,catalog.default_selection())
    switch_mode(state,'unified')
    return state


@pytest.fixture(scope='module')
def image_state():
    from temsim.assembly_catalog import AssemblyCatalog,AssemblySelection
    from temsim.optics.column import default_state
    from temsim.simulation_modes import switch_mode
    state=default_state();catalog=AssemblyCatalog();selection=catalog.default_selection()
    column=next(c.name for c in catalog.columns if c.file=='column/C3_ProbeCorrector_ImageCorrector.toml')
    catalog.apply(state,AssemblySelection(selection.gun,column,selection.recording))
    switch_mode(state,'unified')
    return state


def configured_state(state,key):
    """Detached geometry recipe with unchanged runtime optical commands."""
    from temsim.multipole_winding import supported_multipole_winding, electrical_channel_count
    from temsim.physics.transverse_coil_field import _components
    current=copy(state)
    current.corrector_elements=[copy(item) for item in state.corrector_elements]
    component=_components(current)[key]
    updates=[]
    for record in current._resolved_assembly.parts:
        part=dict(record.data)
        # Keep other installed recipes untouched. The requested component can
        # be initialized independently of migration state on the workstation.
        if record.key==key:
            assert supported_multipole_winding(part)
            for name in tuple(part):
                if name.startswith('ideal_multipole_'):part.pop(name)
            part.update({STATUS_FIELD:STATUS_VALUE,NOTE_FIELD:SOURCE_NOTE,
                BASIS_FIELD:(recommended_basis_angle_deg(component) if electrical_channel_count(part)==1 else 0.)})
            part.setdefault(turn_field(part),2 if electrical_channel_count(part)==1 else 3)
            updates.append(replace(record,data=part))
        else:updates.append(record)
    current._resolved_assembly=replace(current._resolved_assembly,parts=tuple(updates))
    return current


def install_reference(state,key,recipe):
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(
        replace(part,data={**part.data,**recipe['fields']}) if part.key==key else part
        for part in state._resolved_assembly.parts))


@pytest.mark.parametrize('key',('probe_hp1_hexapole','probe_hp2_hexapole',
    'probe_qpol_quadrupole','probe_qpc_quadrupole','probe_qph1_quadrupole','probe_qph2_quadrupole',
    'probe_hpol_hexapole','probe_hpc_hexapole'))
def test_native_signed_coefficient_calibration_preserves_commands_and_fixed_N(installed_state,key):
    from temsim.physics.core import electron
    state=configured_state(installed_state,key)
    before=json.dumps(state.to_dict(),sort_keys=True)
    recipe=calibrate_multipole_current_matrix(state,key)
    response=np.array(recipe['unit_current_response']);matrix=np.array(recipe['matrix_a_per_unit'])
    target=np.array(recipe['native_target_per_unit'])
    np.testing.assert_allclose(response@matrix,target,rtol=1e-10,atol=1e-14)
    charge,momentum,_=electron(state)
    order=recipe['multipole_order'];angle=recipe['reference_orientation_rad']
    expected=(momentum/charge)*np.array([[math.cos(order*angle),-math.sin(order*angle)],
                                       [math.sin(order*angle),math.cos(order*angle)]])[:,:matrix.shape[1]]
    np.testing.assert_array_equal(target,expected)
    assert json.dumps(state.to_dict(),sort_keys=True)==before
    if key in ('probe_hp1_hexapole','probe_hp2_hexapole'):
        assert recipe['turn_count_per_coil']==2 and matrix.shape==(1,1)


def test_primary_six_coil_zero_angle_cannot_fit_native_normal_without_angular_datum(installed_state):
    state=configured_state(installed_state,'probe_hp2_hexapole')
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(
        replace(p,data={**p.data,BASIS_FIELD:0.}) if p.key=='probe_hp2_hexapole' else p
        for p in state._resolved_assembly.parts))
    with pytest.raises(ValueError,match='fixed winding phase cannot reproduce'):
        calibrate_multipole_current_matrix(state,'probe_hp2_hexapole')


@pytest.mark.parametrize('key',('image_hp1_hexapole','image_hp2_hexapole',
                              'image_qpol_quadrupole','image_hpol_hexapole'))
def test_image_family_uses_its_native_signed_field_without_replacing_other_controls(image_state,key):
    state=configured_state(image_state,key)
    before=json.dumps(state.to_dict(),sort_keys=True)
    recipe=calibrate_multipole_current_matrix(state,key)
    install_reference(state,key,recipe)
    state.round_lens_field_model='copper_coil'
    rows=capture_multipole_coil_fields(state,component_keys={key})
    assert len(rows)==1 and rows[0].component_key==key
    # Only the explicitly selected field model changes in operating state.
    state.round_lens_field_model=image_state.round_lens_field_model
    assert json.dumps(state.to_dict(),sort_keys=True)==before
    np.testing.assert_allclose(np.asarray(recipe['unit_current_response'])@recipe['matrix_a_per_unit'],
                               recipe['native_target_per_unit'],rtol=1e-10,atol=1e-14)


def test_hexapole_axis_coefficient_agrees_with_independent_small_ring_limit(installed_state):
    state=configured_state(installed_state,'probe_hp2_hexapole')
    part=next(p for p in state._resolved_assembly.parts if p.key=='probe_hp2_hexapole')
    source=TransverseCoilPotential('hp2',multipole_winding_envelopes(part.data),2,(1.,0.))
    z=part.data['optical_reference_local_z_mm']*.001
    exact=multipole_field_coefficients(source,np.array([[0.,0.,z]]),3)[0]
    phase=np.arange(48)*2*np.pi/48
    def ring(radius):
        points=np.column_stack((radius*np.cos(phase),radius*np.sin(phase),np.full(48,z)))
        field=source.field_at_global_positions_t(points)
        value=np.mean((field[:,1]+1j*field[:,0])*np.exp(-2j*phase))/radius**2
        return np.array([value.real,-value.imag])
    extrapolated=(4*ring(.00005)-ring(.0001))/3
    np.testing.assert_allclose(extrapolated,exact,rtol=2e-7,atol=2e-11)


def test_captured_currents_are_energy_independent_and_replace_native_owner_once(installed_state):
    from temsim.physics.transverse_coil_field import capture_transverse_coil_fields
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    state=configured_state(installed_state,'probe_qpol_quadrupole')
    recipe=calibrate_multipole_current_matrix(state,'probe_qpol_quadrupole')
    install_reference(state,'probe_qpol_quadrupole',recipe)
    component=next(c for c in state.corrector_elements if c.key=='probe_qpol_quadrupole')
    component.strength_m2=1.;component.skew_strength_m2=-2.
    state.round_lens_field_model='copper_coil'
    first=capture_multipole_coil_fields(state,component_keys={component.key})
    state._propagation_energy_kev=150.
    second=capture_multipole_coil_fields(state,component_keys={component.key})
    assert len(first)==len(second)==1
    assert first[0].potential.currents_a==second[0].potential.currents_a
    assert first[0].potential.digest==second[0].potential.digest
    for rows in (capture_transverse_coil_fields(state,component_keys={component.key}),
                 capture_posed_column_fields(state,include_aligned=True,component_keys={component.key})):
        assert len(rows)==1 and type(rows[0]) is FrozenTransverseCoilField
        assert rows[0].component_key==component.key
    state.round_lens_field_model='gaussian'
    rows=capture_posed_column_fields(state,include_aligned=True,component_keys={component.key})
    assert len(rows)==1 and type(rows[0]) is not FrozenTransverseCoilField


def test_primary_installation_angle_is_a_real_rigid_rotation_without_current_refitting(installed_state):
    state=configured_state(installed_state,'probe_hp1_hexapole')
    recipe=calibrate_multipole_current_matrix(state,'probe_hp1_hexapole')
    install_reference(state,'probe_hp1_hexapole',recipe)
    component=next(c for c in state.corrector_elements if c.key=='probe_hp1_hexapole')
    state.round_lens_field_model='copper_coil'
    first=capture_multipole_coil_fields(state,component_keys={component.key})[0]
    angle=.123
    component.orientation_rad+=angle
    second=capture_multipole_coil_fields(state,component_keys={component.key})[0]
    assert first.potential.currents_a==second.potential.currents_a
    assert first.potential.provenance['electrical']==second.potential.provenance['electrical']
    rotation=np.array([[math.cos(angle),-math.sin(angle),0.],
                       [math.sin(angle),math.cos(angle),0.],[0.,0.,1.]])
    # Keep the covariance probe well conditioned against cancellation of the
    # large individual line contributions; analytic axis derivatives are
    # independently checked above without a finite-radius approximation.
    points=np.array([[.0005,-.001,component.z_mm*.001+.002]])
    np.testing.assert_allclose(second.field_at_global_positions_t(points@rotation.T),
        first.field_at_global_positions_t(points)@rotation.T,rtol=1e-10,atol=1e-16)
    assert second.potential.provenance['runtime_basis_delta_rad']==pytest.approx(angle)
    # An explicit re-calibration at another installation angle must retain
    # the old winding datum; otherwise the physical pose would jump after fit.
    updated=calibrate_multipole_current_matrix(state,component.key)
    assert updated['fields'][REFERENCE_ORIENTATION_FIELD]==recipe['fields'][REFERENCE_ORIENTATION_FIELD]
    assert updated['calibration_native_orientation_rad']==component.orientation_rad
    install_reference(state,component.key,updated)
    third=capture_multipole_coil_fields(state,component_keys={component.key})[0]
    assert third.potential.coils==second.potential.coils
    np.testing.assert_allclose(third.potential.currents_a,second.potential.currents_a,rtol=1e-12)
    component.skew_strength_m3=1.
    with pytest.raises(ValueError,match='one current basis'):
        capture_multipole_coil_fields(state,component_keys={component.key})


def test_saved_primary_current_reference_supports_generic_default_and_prior_installation_angles(installed_state):
    from temsim.physics.core import electron
    state=copy(installed_state)
    state.round_lens_field_model='copper_coil'
    rows=capture_multipole_coil_fields(state,component_keys={'probe_hp1_hexapole','probe_hp2_hexapole'})
    assert len(rows)==2
    charge,momentum,_=electron(state)
    components={c.key:c for c in state.corrector_elements}
    for row in rows:
        component=components[row.component_key]
        point=np.array([[0.,0.,component.z_mm*.001]])
        native=component.hexapole_strength_components_m3(np.asarray(component.z_mm))
        np.testing.assert_allclose(multipole_field_coefficients(row.potential,point,3)[0],
            (momentum/charge)*np.asarray(native),rtol=1e-10,atol=1e-12)


def test_dual_hexapole_orientation_changes_mix_currents_without_refitting(installed_state):
    state=configured_state(installed_state,'probe_hpol_hexapole')
    recipe=calibrate_multipole_current_matrix(state,'probe_hpol_hexapole')
    install_reference(state,'probe_hpol_hexapole',recipe)
    component=next(c for c in state.corrector_elements if c.key=='probe_hpol_hexapole')
    component.strength_m3=1.;component.skew_strength_m3=0.
    state.round_lens_field_model='copper_coil'
    first=capture_multipole_coil_fields(state,component_keys={component.key})[0]
    component.orientation_rad+=np.pi/6
    second=capture_multipole_coil_fields(state,component_keys={component.key})[0]
    matrix=np.asarray(recipe['matrix_a_per_unit'])
    np.testing.assert_allclose(first.potential.currents_a,matrix@[1.,0.],atol=1e-15)
    np.testing.assert_allclose(second.potential.currents_a,matrix@[0.,1.],atol=1e-15)
    assert second.potential.provenance['electrical']['matrix_a_per_unit']==first.potential.provenance['electrical']['matrix_a_per_unit']
    assert len(second.potential.coils)==12


def test_hexapole_disable_filter_preserves_quadrupole_only(installed_state):
    state=configured_state(installed_state,'probe_hpol_hexapole')
    recipe=calibrate_multipole_current_matrix(state,'probe_hpol_hexapole')
    install_reference(state,'probe_hpol_hexapole',recipe)
    state.round_lens_field_model='copper_coil'
    from temsim.physics.posed_column_fields import capture_posed_column_fields
    assert capture_posed_column_fields(state,include_aligned=True,include_hexapole=False,
        component_keys={'probe_hpol_hexapole'})==()


def test_potential_roundtrip_and_wave_admission_never_substitute_native(installed_state):
    from temsim.physics.transverse_coil_wave_policy import require_transverse_coil_wave_admission
    state=configured_state(installed_state,'probe_hpc_hexapole')
    recipe=calibrate_multipole_current_matrix(state,'probe_hpc_hexapole')
    install_reference(state,'probe_hpc_hexapole',recipe)
    state.round_lens_field_model='copper_coil'
    source=capture_multipole_coil_fields(state,component_keys={'probe_hpc_hexapole'})[0].potential
    restored=TransverseCoilPotential.from_dict(json.loads(json.dumps(source.to_dict())))
    assert restored.digest==source.digest
    with pytest.raises(ValueError,match='classical particles only'):
        require_transverse_coil_wave_admission(SimpleNamespace(contributions=(SimpleNamespace(potential=source),)))


def test_actual_field_and_unified_potential_graph_keep_one_full_tail_source(installed_state):
    from temsim.physics.potential_capture import capture_selected_column_potentials
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    from temsim.physics.unified_field_runtime import capture_unified_field_runtime,UnifiedFieldRuntime
    state=configured_state(installed_state,'probe_qpol_quadrupole')
    recipe=calibrate_multipole_current_matrix(state,'probe_qpol_quadrupole')
    recipe['fields'][BIAS_FIELD]=[.002,-.003]
    install_reference(state,'probe_qpol_quadrupole',recipe)
    state.round_lens_field_model='copper_coil'
    selected=capture_selected_column_potentials(state,auxiliary_keys=('probe_qpol_quadrupole',))
    assert len(selected.contributions)==1
    assert type(selected.contributions[0].potential) is TransverseCoilPotential
    instrument=capture_instrument_magnetic_field(state)
    rows=[r for r in instrument._sources if r.key=='probe_qpol_quadrupole']
    assert len(rows)==1 and type(rows[0].provider) is FrozenTransverseCoilField
    assert rows[0].identity is not None
    point=np.array([[1e-6,-2e-6,.5]])
    expected=selected.contributions[0].potential.field_at_global_positions_t(point)
    assert np.linalg.norm(expected)>0
    np.testing.assert_array_equal(rows[0].provider.field_at_global_positions_t(point),expected)
    runtime=capture_unified_field_runtime(state)
    restored=UnifiedFieldRuntime.from_dict(json.loads(json.dumps(runtime.to_dict())))
    assert restored.digest==runtime.digest
    snapshot=getattr(runtime.snapshot,'static_snapshot',runtime.snapshot)
    owned=[r for r in snapshot.contributions if 'probe_qpol_quadrupole' in r.owner_ids]
    assert len(owned)==1 and type(owned[0].potential) is TransverseCoilPotential


@pytest.mark.parametrize('value',(True,complex(1,2),float('inf')))
def test_derivatives_reject_nonreal_or_nonfinite_positions(value):
    source=TransverseCoilPotential('square',(rectangle(),),7,(1.,0.))
    with pytest.raises(ValueError,match='finite real XYZ'):
        magnetic_field_hessian_t_per_m2(source,[[value,value,value]])
