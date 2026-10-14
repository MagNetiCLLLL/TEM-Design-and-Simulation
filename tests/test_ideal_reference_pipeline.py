"""Paired-source and cache contracts; the gun here is an explicit test fixture."""
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pytest

from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.physics import column_wave, tip_wave_pipeline as pipeline
from temsim.physics.ideal_column_reference import column_plan_for_model
from test_tip_wave_pipeline import (
    _fixture_tip_checkpoint, _observation_request,
    _quiet_observation_state, _quiet_prepared_column,
)


def paired_fixture(monkeypatch):
    state = _quiet_observation_state()
    start = state.electron_gun.exit_plane_z_mm
    gun = replace(_fixture_tip_checkpoint(start), record={
        "schema": "executed-tip-gun-v1", "scope": "PAIRED PIPELINE FIXTURE, NOT PHYSICAL GUN ACCEPTANCE"})
    gun_calls = []

    def source(*_args, **_kwargs):
        gun_calls.append(1)
        return gun

    def prepare(*args, column_model="physical"):
        plan, radii, stops, owners = _quiet_prepared_column(*args)
        return column_plan_for_model(plan, column_model), radii, stops, owners

    monkeypatch.setattr(pipeline, "_STAGES", OrderedDict())
    monkeypatch.setattr(pipeline, "_OBSERVATION_PREFIXES", OrderedDict())
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", source)
    monkeypatch.setattr(pipeline, "_prepare_column", prepare)
    monkeypatch.setattr(column_wave, "_prepare_column", prepare)
    return state, gun, gun_calls


@pytest.mark.parametrize("segmented", (False, True))
def test_reference_shares_gun_but_never_reuses_physical_plane_prefix(monkeypatch, tmp_path, segmented):
    state, gun, gun_calls = paired_fixture(monkeypatch)
    before = capture_instrument_snapshot(state).digest
    request = _observation_request(tmp_path, gun.plane_z_mm+.1, segmented=segmented)
    actual = pipeline.simulate_tip_wave(state, request)
    ideal = pipeline.simulate_ideal_reference(state, actual)
    assert gun_calls == [1]
    assert ideal.request.column_model == "first_order_reference"
    assert actual.request.column_model == "physical"
    assert ideal.instrument_digest == actual.instrument_digest == before
    assert capture_instrument_snapshot(state).digest == before
    actual_source = pipeline._executed_gun_digest(actual.checkpoint)
    assert pipeline._executed_gun_digest(ideal.checkpoint) == actual_source
    if not segmented:
        assert actual_source == gun.digest
    # Even this aberration-free drift has two distinct execution policies.
    assert actual.checkpoint.digest != ideal.checkpoint.digest
    a, b = actual.checkpoint.beam.modes[0], ideal.checkpoint.beam.modes[0]
    np.testing.assert_allclose(a.plane.amplitude, b.plane.amplitude, atol=1e-14, rtol=0)
    np.testing.assert_allclose(a.plane.curvature_m1, b.plane.curvature_m1, atol=1e-12, rtol=0)
    assert a.weight_per_reference_electron == b.weight_per_reference_electron
    repeated = pipeline.simulate_ideal_reference(state, actual)
    assert repeated.propagation_cache_hit
    assert repeated.checkpoint.digest == ideal.checkpoint.digest
    assert pipeline.simulate_tip_wave(state, request).checkpoint.digest == actual.checkpoint.digest


def test_reference_rejects_changed_instrument_without_starting_a_second_solve(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state, _observation_request(tmp_path, gun.plane_z_mm+.1))
    state.step_mm *= .5
    monkeypatch.setattr(pipeline, "simulate_tip_wave", lambda *a, **k: pytest.fail("Changed inputs must reject first"))
    with pytest.raises(ValueError, match="changed since"):
        pipeline.simulate_ideal_reference(state, actual)


@pytest.mark.parametrize("segmented", (False, True))
def test_recomputed_identical_gun_is_independent_of_storage_location(monkeypatch, tmp_path, segmented):
    state, gun, gun_calls = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state,
        _observation_request(tmp_path, gun.plane_z_mm+.1, segmented=segmented))
    ideal = pipeline.simulate_ideal_reference(state, actual, use_cache=False)
    assert len(gun_calls) == 2
    assert pipeline._executed_gun_digest(actual.checkpoint) == pipeline._executed_gun_digest(ideal.checkpoint)


def test_reference_rejects_controls_changed_during_calculation(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state, _observation_request(tmp_path, gun.plane_z_mm))

    def changed(*_args, **_kwargs):
        state.step_mm *= .5
        return replace(actual, request=replace(actual.request, column_model="first_order_reference"))

    monkeypatch.setattr(pipeline, "simulate_tip_wave", changed)
    with pytest.raises(ValueError, match="changed during"):
        pipeline.simulate_ideal_reference(state, actual)


def test_reference_rejects_wrong_source_even_when_plane_and_instrument_match(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state, _observation_request(tmp_path, gun.plane_z_mm+.1))
    unrelated = replace(actual.checkpoint, record={"schema": "executed-tip-gun-v1", "other_source": True})
    wrong = replace(actual, checkpoint=unrelated,
                    request=replace(actual.request, column_model="first_order_reference"))
    monkeypatch.setattr(pipeline, "simulate_tip_wave", lambda *a, **k: wrong)
    with pytest.raises(ValueError, match="same executed source"):
        pipeline.simulate_ideal_reference(state, actual)


def test_reference_cancel_and_missing_ancestry_do_not_start_propagation(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    request = _observation_request(tmp_path, gun.plane_z_mm)
    actual = pipeline.simulate_tip_wave(state, request)
    monkeypatch.setattr(pipeline, "simulate_tip_wave", lambda *a, **k: pytest.fail("Must not propagate"))
    with pytest.raises(InterruptedError):
        pipeline.simulate_ideal_reference(state, actual, cancelled=lambda: True)
    unknown = replace(actual, checkpoint=replace(gun, record={"historical": True}))
    with pytest.raises(ValueError, match="gun ancestry"):
        pipeline.simulate_ideal_reference(state, unknown)


def test_reference_requires_physical_exact_plane_and_explicit_policy(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state, _observation_request(tmp_path, gun.plane_z_mm))
    with pytest.raises(ValueError, match="physical beam"):
        pipeline.simulate_ideal_reference(state, replace(actual,
            request=replace(actual.request, column_model="first_order_reference")))
    with pytest.raises(ValueError, match="exact observation plane"):
        pipeline.simulate_ideal_reference(state, replace(actual, checkpoint=replace(gun, plane_z_mm=gun.plane_z_mm+.1)))
    with pytest.raises(ValueError, match="Column model"):
        replace(actual.request, column_model="vacuum").validate()


def test_reference_late_cancellation_never_publishes(monkeypatch, tmp_path):
    state, gun, _ = paired_fixture(monkeypatch)
    actual = pipeline.simulate_tip_wave(state, _observation_request(tmp_path, gun.plane_z_mm))
    cancelled = [False]

    def completed(*_args, **_kwargs):
        cancelled[0] = True
        return actual

    monkeypatch.setattr(pipeline, "simulate_tip_wave", completed)
    with pytest.raises(InterruptedError, match="before publication"):
        pipeline.simulate_ideal_reference(state, actual, cancelled=lambda: cancelled[0])
