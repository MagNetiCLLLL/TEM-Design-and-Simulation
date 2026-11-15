"""Fixed physical turns and current-only field calibration, without transport."""
from copy import copy,deepcopy
from dataclasses import replace

import numpy as np
import pytest

from temsim.coil_winding import OWNER_CURRENT_FIELD,INTERVAL_TURNS_FIELD
from temsim.physics.potential_capture import capture_round_lens_potential

HISTORICAL_REGION_TURNS = 586
HISTORICAL_CURRENT_AT_100_PERCENT_A = 1575.2330285262728
DEFAULT_REGION_TURNS = 580
DEFAULT_CURRENT_AT_100_PERCENT_A = HISTORICAL_CURRENT_AT_100_PERCENT_A*HISTORICAL_REGION_TURNS/DEFAULT_REGION_TURNS


@pytest.fixture(autouse=True)
def forbid_source(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Fixed-winding component checks must not emit or transport particles')
    monkeypatch.setattr('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden)
    monkeypatch.setattr('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden)


@pytest.fixture(scope='module')
def baseline():
    from temsim.optics.column import default_state
    return default_state()


def state_copy(baseline):
    state=copy(baseline);state.lenses=deepcopy(baseline.lenses)
    state.simulation_mode='unified';state.round_lens_field_model='copper_coil'
    state._runtime_copper_coil_provider_cache={};state._runtime_lens_field_provider_cache={}
    return state


def edit(state,key,updates=None,remove=()):
    parts=[]
    for part in state._resolved_assembly.parts:
        if part.key==key:
            data={k:v for k,v in part.data.items() if k not in remove}
            data.update(updates or {});part=replace(part,data=data)
        parts.append(part)
    state._resolved_assembly=replace(state._resolved_assembly,parts=tuple(parts))


def test_default_counts_retain_authorized_integer_area_and_photo_reference_without_guessing_wire(baseline):
    coil=baseline._resolved_assembly.part('objective_lens_excitation_coil')
    assert coil.data['winding_turn_count']==1160
    assert coil.data[INTERVAL_TURNS_FIELD]==(580,580)
    assert 10440/coil.data['winding_turn_count']==9.
    assert abs(DEFAULT_REGION_TURNS-HISTORICAL_REGION_TURNS)/HISTORICAL_REGION_TURNS <= .20
    assert 'winding_wire_diameter_mm' not in coil.data
    assert 'winding_wire_cross_section_mm2' not in coil.data
    owner=baseline._resolved_assembly.part('objective_lens')
    assert owner.data[OWNER_CURRENT_FIELD]==DEFAULT_CURRENT_AT_100_PERCENT_A
    assert 'specimen upper surface' in owner.data['parameter_metadata'][OWNER_CURRENT_FIELD]['source_note']
    note=coil.data['parameter_metadata']['winding_turn_count']['source_note']
    assert 'photo' in note and '586' in note
    assert '20%' in note or '20 percent' in note


def test_known_turns_alone_preserve_the_exact_old_legacy_field_coefficients(baseline):
    state=state_copy(baseline)
    edit(state,'objective_lens',remove=(OWNER_CURRENT_FIELD,))
    known=capture_round_lens_potential(state,'objective_lens').potential
    edit(state,'objective_lens_excitation_coil',remove=('winding_turn_count',INTERVAL_TURNS_FIELD,'parameter_metadata'))
    legacy=capture_round_lens_potential(state,'objective_lens').potential
    assert [c.to_metadata() for c in known.coils]==[c.to_metadata() for c in legacy.coils]


def test_large_current_with_unknown_wire_keeps_586_turns_in_each_region(baseline):
    # Preserve the earlier 586/586 illustration as an explicit historical
    # input instead of relying on today's user-authorized integer-area default.
    state=state_copy(baseline)
    edit(state,'objective_lens_excitation_coil',{'winding_turn_count':1172,INTERVAL_TURNS_FIELD:[586,586]})
    edit(state,'objective_lens',{OWNER_CURRENT_FIELD:HISTORICAL_CURRENT_AT_100_PERCENT_A})
    potential=capture_round_lens_potential(state,'objective_lens').potential
    current=potential.provenance['winding_current_definition']['current_a']
    np.testing.assert_allclose([c.ampere_turns for c in potential.coils],[586*current]*2,rtol=0,atol=0)
    witness=potential.provenance['winding_current_definition']['windings'][0]
    assert witness['interval_turn_counts']==(586,586)
    assert witness['wire_diameter_mm'] is None and witness['conductor_cross_section_mm2'] is None
    assert witness['turn_count']==1172
    assert np.isfinite(potential.sample([[0,0,1.5992]])['B_t']).all()


def test_length_edit_does_not_redistribute_fixed_physical_turns(baseline):
    state=state_copy(baseline);edit(state,'objective_lens',{OWNER_CURRENT_FIELD:1000.})
    first=capture_round_lens_potential(state,'objective_lens').potential
    parent=state._resolved_assembly.part('objective_lens')
    edit(state,'objective_lens',{'upper_yoke_end_local_z_mm':parent.data['upper_yoke_end_local_z_mm']-10})
    changed=capture_round_lens_potential(state,'objective_lens').potential
    assert changed.coils[0].length_m!=changed.coils[1].length_m
    assert [c.ampere_turns for c in changed.coils]==[c.ampere_turns for c in first.coils]
    assert changed.coils[0].ampere_turns==changed.coils[1].ampere_turns


def test_fixed_regional_counts_invalidate_cache_and_change_the_field(baseline):
    from temsim.round_lens_models import round_lens_cache_identity
    state=state_copy(baseline);edit(state,'objective_lens',{OWNER_CURRENT_FIELD:1000.})
    first=capture_round_lens_potential(state,'objective_lens').potential
    identity=round_lens_cache_identity(state)
    edit(state,'objective_lens_excitation_coil',{INTERVAL_TURNS_FIELD:[600,560]})
    changed=capture_round_lens_potential(state,'objective_lens').potential
    assert round_lens_cache_identity(state)!=identity
    assert changed.digest!=first.digest
    assert sum(c.ampere_turns for c in changed.coils)==sum(c.ampere_turns for c in first.coils)
    assert not np.array_equal(changed.sample([[0,0,1.47]])['B_t'],first.sample([[0,0,1.47]])['B_t'])


def test_geometry_mismatch_is_rejected_before_field_sampling(baseline):
    state=state_copy(baseline)
    edit(state,'objective_lens_excitation_coil',{INTERVAL_TURNS_FIELD:[1160]})
    with pytest.raises(ValueError,match='one count per actual winding interval'):
        capture_round_lens_potential(state,'objective_lens')


def test_selected_current_matches_original_specimen_field_without_changing_controls():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.column.state_layout import apply_physical_layout_to_state
    from temsim.optics.column import default_state
    from temsim.operating_modes import apply_operating_mode_pair
    from temsim.physics.lens_field_provider import resolve_runtime_lens_field_provider
    from temsim.simulation_modes import switch_mode
    catalog=AssemblyCatalog();selection=catalog.default_selection();state=default_state()
    catalog.apply(state,selection);switch_mode(state,'unified')
    apply_operating_mode_pair(state,'nano_probe','diffraction',column_name=selection.column,
                             recording_name=selection.recording)
    apply_physical_layout_to_state(state);state.step_mm=.05
    controls=deepcopy(state.to_dict())
    lens=next(l for l in state.lenses if l.key=='objective_lens')
    plane=state.sample.upper_surface_z_mm
    gaussian=resolve_runtime_lens_field_provider(state,lens.key,lens)
    ideal=state_copy(state)
    potential=capture_round_lens_potential(ideal,lens.key).potential
    current=potential.provenance['winding_current_definition']['current_a']
    assert current==DEFAULT_CURRENT_AT_100_PERCENT_A*(lens.percent*.01)
    assert [c.ampere_turns for c in potential.coils]==[580*current]*2
    np.testing.assert_allclose(580*current,586*1086.565908295224,rtol=2e-15,atol=0)
    observed=potential.sample([[0.,0.,plane*.001]])['B_t'][0]
    np.testing.assert_allclose(observed,[0.,0.,float(gaussian.magnetic_field_t(plane))],rtol=2e-14,atol=0)
    assert all(c.quadrature_order==12 for c in potential.coils)
    assert state.to_dict()==controls
    assert state.round_lens_field_model=='gaussian'
