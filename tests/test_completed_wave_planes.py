"""Durable complete-mixture references; controlled fields, not tip qualification."""
from dataclasses import replace
import json

import numpy as np
import pytest

from temsim.immutable_json import freeze_json, json_digest, thaw_json
from temsim.physics.tip_wave_pipeline import simulate_tip_wave
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from test_unified_wave_runtime import _request, _runtime
from test_unified_weighted_resampling import _lens_runtime, _modes
from test_tip_wave_pipeline import _quiet_observation_state, coherent_cuda


@pytest.fixture
def completed(tmp_path):
    state = _quiet_observation_state()
    source = _modes(2)
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))
    store = ExecutedWaveStore(tmp_path/'completed', 'controlled-source', 2**28)
    identity = json_digest(('complete-request', adapter.identity, source.digest, request.wave_grid.column_identity()))
    mode_ids = tuple(mode.mode_id for mode in source.beam.modes)
    validate = lambda checkpoint: adapter.column(checkpoint, checkpoint.plane_z_mm)
    saved = []

    def register(checkpoint):
        store.register_completed_plane(identity, checkpoint, expected_mode_ids=mode_ids, validate=validate)
        saved.append(checkpoint)

    adapter.column(source, 1000.0004, store=store, checkpoint_callback=register)
    assert len(saved) >= 2
    return store, identity, saved, validate, mode_ids, adapter


def test_durable_complete_planes_survive_restart_and_keep_older_sections(completed):
    store, identity, saved, validate, mode_ids, _ = completed
    arrays = set(store.root.rglob('*.npy'))
    references = set(store.root.rglob('completed_planes/**/*.json'))
    store.register_completed_plane(identity, saved[-1], expected_mode_ids=mode_ids, validate=validate)
    assert set(store.root.rglob('*.npy')) == arrays
    assert set(store.root.rglob('completed_planes/**/*.json')) == references
    reopened = ExecutedWaveStore(store.root, store.dependency, store.maximum_bytes)
    assert reopened.find_completed_plane(identity, 1000., 1000.0001, validate=validate) is None
    for expected in saved:
        actual = reopened.find_completed_plane(identity, 1000., expected.plane_z_mm, validate=validate)
        assert actual.digest == expected.digest
        assert actual.beam.total_weight == expected.beam.total_weight
        assert actual.record['unified_mode_contexts'] == expected.record['unified_mode_contexts']
    between = (saved[0].plane_z_mm + saved[-1].plane_z_mm)/2
    assert reopened.find_completed_plane(identity, 1000., between, validate=validate).plane_z_mm == saved[0].plane_z_mm


@pytest.mark.parametrize('representation', ['list', 'tuple', 'top_level_tuple'])
def test_nested_json_dependency_survives_manifest_and_completed_index(completed, tmp_path, representation):
    _, identity, saved, validate, mode_ids, _ = completed
    dependency = {'selected_mode_ids': list(mode_ids),
                  'configuration': [{'limits': [2048, 4096]}, ['preview', True]]}
    if representation == 'tuple':
        dependency = freeze_json(dependency)
    elif representation == 'top_level_tuple':
        dependency = ('source', dependency, (2048, 4096))
    store = ExecutedWaveStore(tmp_path/'nested', dependency, 2**28)
    key = store.key('complete', saved[-1].digest)
    stored = store.put(key, saved[-1])
    store.register_completed_plane(identity, stored, expected_mode_ids=mode_ids, validate=validate)
    if representation == 'tuple':
        # Historical writers may use another mapping order/format. Equivalent
        # JSON must register idempotently without rewriting the old reference.
        reference = next(store._completed_directory(identity).glob('*.json'))
        payload = json.loads(reference.read_text(encoding='utf-8'))
        reference.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding='utf-8')
    before = {path: path.read_bytes() for path in store.root.rglob('*.json')}
    # JSON decoding, immutable checkpoint records and callers can each use
    # different list/tuple representations of exactly the same dependency.
    for equivalent in (thaw_json(freeze_json(dependency)), freeze_json(dependency)):
        reopened = ExecutedWaveStore(store.root, equivalent, store.maximum_bytes)
        assert reopened.key('complete', saved[-1].digest) == key
        restored = reopened.get(key)
        assert restored.digest == stored.digest
        reopened.register_completed_plane(identity, restored,
                                          expected_mode_ids=mode_ids, validate=validate)
        observed = reopened.find_completed_plane(identity, 1000., stored.plane_z_mm, validate=validate)
        assert observed.digest == stored.digest
        for actual, expected in zip(observed.beam.modes, stored.beam.modes, strict=True):
            np.testing.assert_array_equal(actual.plane.amplitude, expected.plane.amplitude)
            assert actual.weight_per_reference_electron == expected.weight_per_reference_electron
    assert {path: path.read_bytes() for path in store.root.rglob('*.json')} == before


def test_store_snapshots_nested_dependency_before_caller_mutation(completed, tmp_path):
    _, identity, saved, validate, mode_ids, _ = completed
    dependency = {'selected_mode_ids': list(mode_ids), 'configuration': [{'limits': [2048, 4096]}]}
    expected = freeze_json(dependency)
    store = ExecutedWaveStore(tmp_path/'snapshot', dependency, 2**28)
    key = store.key('complete', saved[-1].digest)
    dependency['selected_mode_ids'].reverse()
    dependency['configuration'][0]['limits'].append(8192)
    dependency['new_unrelated_field'] = 'late mutation'
    assert json_digest(store.dependency) == json_digest(expected)
    assert store.key('complete', saved[-1].digest) == key
    with pytest.raises(TypeError):
        store.dependency['configuration'][0]['limits'][0] = 1
    stored = store.put(key, saved[-1])
    store.register_completed_plane(identity, stored, expected_mode_ids=mode_ids, validate=validate)
    reopened = ExecutedWaveStore(store.root, expected, store.maximum_bytes)
    assert reopened.find_completed_plane(identity, 1000., stored.plane_z_mm, validate=validate).digest == stored.digest
    changed = ExecutedWaveStore(store.root, dependency, store.maximum_bytes)
    with pytest.raises(ValueError, match='identity/dependency changed'):
        changed.get(key)
    assert changed.find_completed_plane(identity, 1000., stored.plane_z_mm, validate=validate) is None


@pytest.mark.parametrize('location', ['manifest', 'registration', 'reference'])
def test_json_equivalence_does_not_accept_changed_dependency(completed, tmp_path, location):
    _, identity, saved, validate, mode_ids, _ = completed
    dependency = {'selected_mode_ids': list(mode_ids), 'configuration': [{'enabled': True}]}
    store = ExecutedWaveStore(tmp_path/'tampered', dependency, 2**28)
    key = store.key('complete', saved[-1].digest)
    stored = store.put(key, saved[-1])
    store.register_completed_plane(identity, stored, expected_mode_ids=mode_ids, validate=validate)
    if location == 'registration':
        record = thaw_json(stored.record)
        # True and 1 compare equal in Python, but are distinct JSON identities.
        record['storage']['dependency']['configuration'][0]['enabled'] = 1
        with pytest.raises(ValueError, match="this store's committed checkpoint"):
            store.register_completed_plane(identity, replace(stored, record=record),
                                           expected_mode_ids=mode_ids, validate=validate)
        return
    if location == 'manifest':
        path = stored.beam.modes.root/'manifest.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload.pop('manifest_digest')
        payload['dependency']['configuration'][0]['enabled'] = 1
        digest = json_digest(payload)
        path.write_text(json.dumps({**payload, 'manifest_digest': digest}), encoding='utf-8')
        index = store.root/(key+'.json')
        pointer = json.loads(index.read_text(encoding='utf-8'))
        pointer['digest'] = digest
        index.write_text(json.dumps(pointer), encoding='utf-8')
        with pytest.raises(ValueError, match='identity/dependency changed'):
            store.get(key)
    else:
        path = next(store._completed_directory(identity).glob('*.json'))
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload.pop('digest')
        payload['dependency']['configuration'][0]['enabled'] = 1
        digest = json_digest(payload)
        path.unlink()
        path.with_name(digest+'.json').write_text(json.dumps({**payload, 'digest': digest}), encoding='utf-8')
        with pytest.raises(ValueError, match='reference identity changed'):
            store.find_completed_plane(identity, 1000., stored.plane_z_mm, validate=validate)


@pytest.mark.parametrize('change', ['source', 'numerics', 'adapter', 'dependency'])
def test_durable_reference_requires_exact_request_and_store_identity(completed, change):
    store, identity, saved, validate, _, _ = completed
    if change == 'dependency':
        store = ExecutedWaveStore(store.root, 'different-dependency', store.maximum_bytes)
    else:
        identity = json_digest((identity, change))
    assert store.find_completed_plane(identity, 1000., saved[-1].plane_z_mm, validate=validate) is None


def test_partial_modes_and_incomplete_mode_sets_cannot_register(completed):
    store, identity, saved, validate, mode_ids, _ = completed
    before = set(store.root.rglob('completed_planes/**/*.json'))
    with pytest.raises(ValueError, match='entire committed mode mixture'):
        store.register_completed_plane(identity, saved[-1], expected_mode_ids=mode_ids[:1], validate=validate)
    for index in store.root.glob('*.json'):
        partial = store.get(index.stem)
        if len(partial.beam.modes) == 1:
            break
    with pytest.raises(ValueError, match='entire committed mode mixture'):
        store.register_completed_plane(identity, partial, expected_mode_ids=mode_ids, validate=validate)
    assert set(store.root.rglob('completed_planes/**/*.json')) == before


@pytest.mark.parametrize('damage', ['reference', 'manifest', 'amplitude', 'missing_index'])
def test_corrupted_complete_reference_fails_closed_without_older_fallback(completed, damage):
    store, identity, saved, validate, _, _ = completed
    last = saved[-1]
    storage = last.record['storage']
    if damage == 'reference':
        reference = next(path for path in store.root.rglob('completed_planes/**/*.json')
                         if json.loads(path.read_text(encoding='utf-8'))['key'] == storage['key'])
        data = json.loads(reference.read_text(encoding='utf-8'))
        data['plane_z_mm'] -= .00001
        reference.write_text(json.dumps(data), encoding='utf-8')
    elif damage == 'missing_index':
        (store.root/(storage['key']+'.json')).unlink()
    elif damage == 'manifest':
        path = last.beam.modes.root/'manifest.json'
        data = json.loads(path.read_text(encoding='utf-8'))
        data['modes'][-1]['weight'] *= .5
        path.write_text(json.dumps(data), encoding='utf-8')
    else:
        path = last.beam.modes.root/last.beam.modes.rows[-1]['arrays']['amplitude']['file']
        path.write_bytes(path.read_bytes()+b'corrupt final mode')
    with pytest.raises(ValueError, match='identity|checksum|lost or changed'):
        store.find_completed_plane(identity, 1000., last.plane_z_mm, validate=validate)


def test_cancelled_or_invalid_phase_registration_publishes_no_reference(completed):
    store, _, saved, validate, mode_ids, adapter = completed
    identity = json_digest('cancelled-reference')
    with pytest.raises(InterruptedError):
        store.register_completed_plane(identity, saved[-1], expected_mode_ids=mode_ids,
                                       validate=validate, cancelled=lambda: True)
    assert not store._completed_directory(identity).exists()
    record = thaw_json(saved[-1].record)
    record['unified_mode_contexts'][mode_ids[-1]].pop('kinetic_phase_ledger')
    bad = store.put(store.key('lost-phase-history'), replace(saved[-1], record=record))
    with pytest.raises(ValueError, match='phase ledger|phase budget'):
        store.register_completed_plane(identity, bad, expected_mode_ids=mode_ids, validate=validate)
    assert not store._completed_directory(identity).exists()


def test_public_changed_target_reuses_committed_intermediate_after_session_restart(monkeypatch, tmp_path, coherent_cuda):
    """Actual bounded two-mode vacuum transport, with an explicit source double."""
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.unified_field_runtime as runtime_module
    import temsim.physics.unified_wave_runtime as wave_module

    state = _quiet_observation_state()
    state.simulation_mode = 'unified'
    source = replace(_modes(2), plane_z_mm=state.electron_gun.exit_plane_z_mm)
    start = source.plane_z_mm
    gun_calls, column_calls = [], []
    monkeypatch.setattr(runtime_module, 'get_unified_field_runtime',
                        lambda working: _runtime(working, start*.001, (start+.002)*.001))

    def gun(*_args, **_kwargs):
        gun_calls.append('executed-input-double')
        return source

    execute = wave_module.observe_potential_geometry_checkpoint

    def transport(snapshot, geometry, checkpoint, stops, **kwargs):
        column_calls.append((checkpoint.plane_z_mm, stops[-1]*1000.))
        return execute(snapshot, geometry, checkpoint, stops, **kwargs)

    monkeypatch.setattr(pipeline, 'build_tip_gun_checkpoint', gun)
    monkeypatch.setattr(wave_module, 'observe_potential_geometry_checkpoint', transport)
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True,
                       stop='plane', observation_z_mm=start+.0008)
    request = replace(request, execution=replace(request.execution, segmented=True, segment_steps=2),
        wave_grid=replace(request.wave_grid, compute_backend="Require GPU"))
    result = simulate_tip_wave(state, request)
    references = list((tmp_path/'completed_planes').rglob('*.json'))
    planes = sorted({json.loads(path.read_text(encoding='utf-8'))['plane_z_mm'] for path in references})
    assert len(planes) >= 3
    earlier = (planes[0]+planes[1])/2
    pipeline._OBSERVATION_PREFIXES.clear()
    pipeline._STAGES.clear()
    column_calls.clear()
    previous = simulate_tip_wave(state, replace(request, observation_z_mm=earlier))
    assert previous.propagation_cache_hit and column_calls[0][0] == planes[0]
    assert all(lower < upper <= (earlier*.001)*1000. for lower, upper in column_calls)
    assert len(gun_calls) == 1
    pipeline._OBSERVATION_PREFIXES.clear()
    pipeline._STAGES.clear()
    column_calls.clear()
    same = simulate_tip_wave(state, request)
    assert same.propagation_cache_hit and column_calls == []
    for actual, expected in zip(same.checkpoint.beam.modes, result.checkpoint.beam.modes, strict=True):
        np.testing.assert_array_equal(actual.plane.amplitude, expected.plane.amplitude)
        assert actual.weight_per_reference_electron == expected.weight_per_reference_electron
    column_calls.clear()
    later = simulate_tip_wave(state, replace(request, observation_z_mm=start+.0011))
    assert later.propagation_cache_hit and column_calls[0][0] == result.checkpoint.plane_z_mm
    assert len(gun_calls) == 1
