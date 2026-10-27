"""Display metadata only; no optical propagation or pooled numerical budget."""
from types import SimpleNamespace

import pytest

from temsim.gui.coherent_result_labels import coherent_preview_text
from temsim.immutable_json import freeze_json


def preview_profile(probability=.001, l2=.1, pixels=2048):
    return {'exploratory_preview': True, 'effective_column_maximum_pixels': pixels,
            'maximum_support_probability_fraction': probability,
            'maximum_accumulated_support_l2': l2,
            'numerical_projection_loss_per_reference': 1e-7,
            'complex_l2_projection_bound_per_reference': .001}


def observation(profile=None):
    record = {} if profile is None else {'upstream': {'unified_numerical_profile': profile}}
    return SimpleNamespace(checkpoint=SimpleNamespace(record=freeze_json(record)))


def test_original_and_single_saved_member_keep_same_preview_metadata():
    original = observation(preview_profile())
    ensemble = SimpleNamespace(members=(original,))
    assert coherent_preview_text(original) == coherent_preview_text(original.checkpoint)
    assert coherent_preview_text(original) == coherent_preview_text(original.checkpoint.record)
    assert coherent_preview_text(ensemble) == coherent_preview_text(original)
    assert 'not accuracy qualified' in coherent_preview_text(ensemble)
    assert 'projection loss 1e-07' in coherent_preview_text(ensemble)


def test_mixed_and_selected_member_labels_keep_independent_budgets():
    first = observation(preview_profile())
    second = observation(preview_profile(.002, .2, 1024))
    strict = observation({'exploratory_preview': False})
    combined = SimpleNamespace(members=(first, second, strict), relative_weights=(.2, .3, .5))
    text = coherent_preview_text(combined)
    for phrase in ('2 of 3 states use preview', 'member budgets and losses are separate',
                   'State 1: Column cap 2048', 'probability 0.001, L2 0.1',
                   'State 2: Column cap 1024', 'probability 0.002, L2 0.2',
                   'State 3: preview disabled'):
        assert phrase in text
    selected = coherent_preview_text(combined, member_index=2)
    assert 'Exploratory preview' in selected and 'Inspecting state 3: preview disabled' in selected
    assert 'Relative budgets' not in selected
    assert combined.relative_weights == (.2, .3, .5)
    with pytest.raises(ValueError, match='selected state index'):
        coherent_preview_text(combined, member_index=4)


def test_nested_members_missing_records_and_cycles_do_not_hide_preview():
    strict = observation()
    nested = SimpleNamespace(members=(SimpleNamespace(members=(observation(preview_profile()),)), strict))
    assert '1 of 2 states use preview' in coherent_preview_text(nested)
    assert coherent_preview_text(SimpleNamespace(members=(strict, strict))) == ''
    assert 'preview status unavailable' in coherent_preview_text(nested)
    record = {}; record['upstream'] = record
    assert coherent_preview_text(SimpleNamespace(record=record)) == ''
    cyclic = SimpleNamespace(); cyclic.members = (cyclic, observation(preview_profile()))
    assert 'Exploratory preview' in coherent_preview_text(cyclic)


def test_selected_state_index_refers_to_members_not_flattened_descendants():
    nested = SimpleNamespace(members=(SimpleNamespace(members=(
        observation(preview_profile()), observation(preview_profile(.002, .2)))),
        observation({'exploratory_preview': False})))
    text = coherent_preview_text(nested, member_index=1)
    assert '2 of 3 states use preview' in text
    assert 'Inspecting state 2: preview disabled' in text
    assert 'Relative budgets' not in text
