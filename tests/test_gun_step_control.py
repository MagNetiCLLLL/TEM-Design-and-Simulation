"""Pure mask/clock feedback tests; no source, fields or particle propagation."""
import gc
import math
import sys
import weakref

import numpy as np
import pytest

from temsim.physics.gun_step_control import (
    SURFACE_STEP_PROPOSAL_SCHEMA, SurfaceStepPredictor,
)


class _OpaqueProvider:
    def __init__(self):
        self.tolerance = 1e-13
        self.parameters = {"energy_fwhm_ev": .3, "boundary": (0., .45)}

    def __eq__(self, other):
        raise AssertionError("Scheduling must not compare providers by value")

    def field_at_global_positions_t(self, points):
        raise AssertionError("The predictor must never evaluate a field")

    def field_at_global_positions_v_per_m(self, points):
        raise AssertionError("The predictor must never evaluate a field")


@pytest.fixture
def context():
    return dict(phase_time_s=0., active=np.array([True, True, False, True]),
                magnetic=_OpaqueProvider(), electric=_OpaqueProvider())


def _at(context, time, **changes):
    return dict(context, phase_time_s=time, **changes)


def test_fresh_trace_returns_cap_and_independent_record(context):
    predictor = SurfaceStepPredictor()
    assert predictor.propose(.25, **context) == .25
    record = predictor.last_proposal
    assert record["schema"] == SURFACE_STEP_PROPOSAL_SCHEMA
    assert record["physics_cap_s"] == record["trial_dt_s"] == .25
    assert record["last_accepted_dt_s"] is None
    assert record["context_changed"]
    assert record["context_change_reasons"] == ("new_trace",)
    record["trial_dt_s"] = 99.
    assert predictor.last_proposal["trial_dt_s"] == .25


@pytest.mark.parametrize("ratio,gain", [(0., 1.25), (1e-200, 1.25), (.5, .9*2**(1/3)),
                                        (1., 1.), (8., 1.), (sys.float_info.max, 1.)])
def test_accepted_feedback_is_bounded_and_returns_no_acceptance_decision(context, ratio, gain):
    predictor = SurfaceStepPredictor()
    predictor.propose(10., **context)
    assert predictor.accepted(2., ratio, **context) is None
    trial = predictor.propose(20., **_at(context, 2.))
    assert trial == pytest.approx(2.*gain, rel=3e-15)
    assert 2. <= trial <= 2.5
    assert predictor.last_proposal["last_accepted_dt_s"] == 2.
    # Ratios above one do not introduce an extra acceptance/rejection gate;
    # the caller owns acceptance and the predictor cannot change its tolerance.
    assert context["magnetic"].tolerance == context["electric"].tolerance == 1e-13
    assert context["magnetic"].parameters["energy_fwhm_ev"] == .3


def test_every_existing_physical_cap_remains_authoritative(context):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.2, 0., **context)
    for cap in (.01, .1, .24, .25, 10.):
        assert predictor.propose(cap, **_at(context, .2)) == min(cap, .25)
    assert predictor.last_proposal["physics_cap_s"] == 10.


def test_retry_suppresses_growth_for_one_feedback_interval(context):
    predictor = SurfaceStepPredictor()
    predictor.propose(1., **context)
    predictor.accepted(.1, 0., **context, retried=True)
    assert predictor.propose(1., **_at(context, .1)) == .1
    predictor.accepted(.1, 0., **_at(context, .1))
    assert predictor.propose(1., **_at(context, .2)) == .125


@pytest.mark.parametrize("new_mask", [np.array([True, False, True, True]),
                                     np.array([True, False, False, True]),
                                     np.array([True, True, True]),
                                     np.zeros(4, bool)])
def test_changed_active_set_keeps_last_step_instead_of_returning_to_large_cap(context, new_mask):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 0., **context)
    changed = _at(context, .1, active=new_mask)
    assert predictor.propose(100., **changed) == .1
    record = predictor.last_proposal
    assert record["active_set_changed"] and record["growth_suppressed"]
    assert record["last_accepted_dt_s"] == .1
    assert record["context_change_reasons"] == ("active_set_changed",)
    predictor.accepted(.1, 0., **changed)
    assert predictor.propose(100., **_at(changed, .2)) == .1
    predictor.accepted(.1, 0., **_at(changed, .2))
    assert predictor.propose(100., **_at(changed, .2+.1)) == .125


def test_active_transition_stays_latched_across_repeated_proposals(context):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 0., **context)
    changed = _at(context, .1, active=np.array([True, False, False, True]))
    assert predictor.propose(100., **changed) == .1
    assert predictor.propose(50., **changed) == .1
    assert predictor.last_proposal["growth_suppressed"]
    predictor.accepted(.1, 0., **changed)
    assert predictor.propose(100., **_at(changed, .2)) == .1


def test_mask_storage_is_owned_and_input_remains_writable(context):
    predictor = SurfaceStepPredictor()
    active = context["active"]
    before = active.copy()
    predictor.accepted(.1, 0., **context)
    assert active.flags.writeable
    np.testing.assert_array_equal(active, before)
    active[1] = False
    assert predictor.propose(100., **_at(context, .1)) == .1
    assert predictor.last_proposal["active_set_changed"]


@pytest.mark.parametrize("name", ["magnetic", "electric"])
def test_changed_provider_restarts_scheduling_with_the_physical_cap(context, name):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 1., **context)
    assert predictor.propose(10., **_at(context, .1)) == .1
    changed = _at(context, .1, **{name: _OpaqueProvider()})
    assert predictor.propose(10., **changed) == 10.
    assert predictor.last_proposal["last_accepted_dt_s"] is None
    assert predictor.last_proposal["context_change_reasons"] == ("providers_changed",)


def test_strong_provider_references_prevent_identity_reuse_and_reset_releases_them():
    magnetic, electric = _OpaqueProvider(), _OpaqueProvider()
    references = weakref.ref(magnetic), weakref.ref(electric)
    predictor = SurfaceStepPredictor()
    predictor.propose(.1, phase_time_s=0., active=np.ones(4, bool), magnetic=magnetic, electric=electric)
    del magnetic, electric
    gc.collect()
    assert all(reference() is not None for reference in references)
    predictor.reset()
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("time", [.099, .101, -1., 2.])
def test_discontinuous_starting_clock_restarts_with_cap(context, time):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 0., **context)
    assert predictor.propose(5., **_at(context, time)) == 5.
    assert predictor.last_proposal["context_change_reasons"] == ("clock_discontinuity",)


def test_clock_continuity_uses_the_executed_float_addition(context):
    predictor = SurfaceStepPredictor()
    context = _at(context, .1)
    predictor.accepted(.2, 1., **context)
    assert predictor.propose(5., **_at(context, .1+.2)) == .2
    assert predictor.propose(5., **_at(context, .3)) == 5.


def test_explicit_reset_does_not_preserve_a_previous_trace_suggestion(context):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 1., **context)
    predictor.reset()
    assert predictor.last_proposal["trial_dt_s"] is None
    assert predictor.propose(7., **context) == 7.


@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, -np.inf, True, "1", 1+2j])
def test_invalid_caps_and_accepted_durations_are_rejected(context, bad):
    predictor = SurfaceStepPredictor()
    with pytest.raises(ValueError):
        predictor.propose(bad, **context)
    with pytest.raises(ValueError):
        predictor.accepted(bad, .5, **context)


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf, -np.inf, True, "1", 1+2j])
def test_invalid_error_ratio_is_not_used_as_scheduling_history(context, bad):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 0., **context)
    with pytest.raises(ValueError, match="Maximum normalized error"):
        predictor.accepted(.1, bad, **_at(context, .1))
    assert predictor.propose(1., **_at(context, .1)) == .125


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, True, "0"])
def test_nonfinite_or_nonreal_clock_is_rejected(context, bad):
    predictor = SurfaceStepPredictor()
    changed = _at(context, bad)
    with pytest.raises(ValueError, match="Starting phase clock"):
        predictor.propose(.1, **changed)
    with pytest.raises(ValueError, match="Starting phase clock"):
        predictor.accepted(.1, .5, **changed)


@pytest.mark.parametrize("bad", [np.ones((2, 2), bool), np.ones(4), [1, 0, 1], True])
def test_active_population_must_be_an_explicit_boolean_vector(context, bad):
    predictor = SurfaceStepPredictor()
    changed = dict(context, active=bad)
    with pytest.raises(ValueError, match="boolean mask"):
        predictor.propose(.1, **changed)
    with pytest.raises(ValueError, match="boolean mask"):
        predictor.accepted(.1, .5, **changed)


def test_invalid_retry_flag_does_not_mutate_valid_history(context):
    predictor = SurfaceStepPredictor()
    predictor.accepted(.1, 0., **context)
    with pytest.raises(ValueError, match="Retry flag"):
        predictor.accepted(.1, 0., **_at(context, .1), retried=1)
    assert predictor.propose(1., **_at(context, .1)) == .125


def test_smallest_positive_duration_and_error_ratio_do_not_underflow_to_zero(context):
    smallest = math.ulp(0.)
    predictor = SurfaceStepPredictor()
    assert predictor.propose(smallest, **context) == smallest
    predictor.accepted(smallest, sys.float_info.max, **context)
    assert predictor.propose(smallest, **_at(context, smallest)) == smallest
    predictor.accepted(smallest, smallest, **_at(context, smallest))
    assert predictor.propose(smallest, **_at(context, smallest+smallest)) == smallest


def test_multiplication_overflow_saturates_to_a_finite_suggestion(context):
    largest = sys.float_info.max
    predictor = SurfaceStepPredictor()
    predictor.accepted(largest, 0., **_at(context, -largest))
    assert predictor.propose(largest, **context) == largest
    assert math.isfinite(predictor.last_proposal["next_suggested_dt_s"])


@pytest.mark.parametrize("time,dt", [(sys.float_info.max, sys.float_info.max), (1., math.ulp(0.))])
def test_unrepresentable_population_clock_is_rejected_clearly(context, time, dt):
    predictor = SurfaceStepPredictor()
    predictor.propose(.1, **context)
    before = predictor.last_proposal
    changed = _at(context, time)
    with pytest.raises(ValueError, match="advance a finite population clock"):
        predictor.propose(dt, **changed)
    with pytest.raises(ValueError, match="advance a finite population clock"):
        predictor.accepted(dt, 0., **changed)
    assert predictor.last_proposal == before


def test_gradual_cap_and_error_sequence_keeps_one_clock_and_all_caps(context):
    predictor = SurfaceStepPredictor()
    time = 0.
    last = None
    for index, cap in enumerate(np.geomspace(1e-8, 1e-4, 40)):
        current = _at(context, time)
        trial = predictor.propose(cap, **current)
        assert 0. < trial <= cap
        if last is not None:
            assert trial <= 1.25*last
        # Mock acceptance only: no field, particle or physical gate is run.
        assert predictor.accepted(trial, .2+.015*index, **current) is None
        time += trial
        last = trial
    assert time > 0.


def test_near_limit_accepted_error_plateau_never_shrinks_or_stalls_the_clock(context):
    predictor = SurfaceStepPredictor()
    time = 0.
    duration = .125
    for index in range(400):
        current = _at(context, time)
        trial = predictor.propose(duration if index == 0 else 100., **current)
        assert trial == duration
        predictor.accepted(trial, .99, **current)
        following = time+trial
        assert following > time
        time = following
    assert time == 50.
    assert predictor.last_proposal["schema"] == "surface-dg-bounded-feedback-proposal-v2"


def test_zero_error_growth_remains_bounded_and_later_physical_cap_can_reduce_it(context):
    predictor = SurfaceStepPredictor()
    assert predictor.propose(.125, **context) == .125
    predictor.accepted(.125, 0., **context)
    next_context = _at(context, .125)
    assert predictor.propose(1., **next_context) == .15625
    assert predictor.propose(.03125, **next_context) == .03125
    predictor.accepted(.03125, .99, **next_context)
    assert predictor.propose(1., **_at(context, .15625)) == .03125
