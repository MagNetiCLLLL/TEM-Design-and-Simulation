"""Explicit resource recovery projects only an unexecuted complex entrance."""
from dataclasses import replace
import math
import weakref

import numpy as np
import pytest

from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.wave_grid import (
    WaveGridBudgetError, WaveMemoryBudgetError, WaveSamplingError,
    WaveGridNumerics, apply_resolved_operator, refine_plane_wave,
    WaveResourcePressure, WaveResourceProposal, WaveResourceTransaction,
)


def _source():
    n=128
    y,x=np.meshgrid(np.arange(n)-n//2,np.arange(n)-n//2,indexing="ij")
    value=np.cos(np.pi*x/n)**8*np.cos(np.pi*y/n)**8
    value=value*(1+5e-5*np.exp(2j*np.pi*36*x/n))*np.exp(.37j)
    value/=np.linalg.norm(value)
    return PlaneWave(value,np.eye(2)*1e-7,np.zeros(2),np.eye(2)*.2,np.array((1e-7,-2e-7)))


def _projection(numerics):
    state={"ledger":NumericalSupportLedger(.02,1e-6),"calls":[]}
    def prepare(wave,error):
        state["calls"].append((wave,error))
        result,state["ledger"]=state["ledger"].resample_spectrum(wave,z_m=.5,
            mode_weight=.25,numerics=numerics)
        return result
    return prepare,state


def test_pressure_target_reserves_failed_sampling_growth_and_original_buffers():
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    source=_source()
    error=WaveGridBudgetError("The required refinement does not fit")
    error.__context__=WaveSamplingError("Measured unresolved bandwidth",required_scale=1.5)
    numeric=WaveGridNumerics(maximum_pixels=128,maximum_working_bytes=256*1024**2,
        maximum_device_working_bytes=448*128**2)
    fits,context=_pressure_resampling_constraints(source,error,numerics=numeric,
        requested="Require GPU",retained_bytes=1234,stage="residual retry")
    assert fits((80,80)) is True
    assert fits((90,90)) is False
    assert context["input_shape"]==[128,128]
    assert context["growth_guard_scale"]==1.5
    assert context["retained_host_bytes"]==1234
    assert context["requested_backend"]=="Require GPU"


def test_pressure_target_retains_optional_cpu_fallback_and_separate_host_limit():
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    source=_source()
    numeric=WaveGridNumerics(maximum_working_bytes=256*1024**2,
        maximum_device_working_bytes=1)
    error=WaveMemoryBudgetError("Host pressure")
    fits,_=_pressure_resampling_constraints(source,error,numerics=numeric,
        requested="Auto",retained_bytes=0,stage="residual retry")
    assert fits((100,100)) is True
    required,_=_pressure_resampling_constraints(source,error,numerics=numeric,
        requested="Require GPU",retained_bytes=0,stage="residual retry")
    assert required((100,100)) is False
    bound=64*1024**2+512*90**2+source.amplitude.nbytes+1234
    bounded,_=_pressure_resampling_constraints(source,error,
        numerics=replace(numeric,maximum_working_bytes=bound),
        requested="CPU",retained_bytes=1234,stage="residual retry")
    assert bounded((64,64)) is True
    assert bounded((70,70)) is False


def test_pressure_target_always_reserves_a_complete_nyquist_guard_retry():
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    fits,context=_pressure_resampling_constraints(_source(),WaveGridBudgetError("pixel cap"),
        numerics=WaveGridNumerics(maximum_pixels=128,maximum_working_bytes=256*1024**2),
        requested="CPU",retained_bytes=0,stage="residual retry")
    assert context["growth_guard_scale"]==1.25
    assert fits((96,96)) is True
    assert fits((100,100)) is False


def test_pressure_target_rejects_untyped_or_unbounded_failure():
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    arguments=dict(numerics=WaveGridNumerics(),requested="CPU",retained_bytes=0,stage="residual retry")
    with pytest.raises(ValueError,match="typed resource"):
        _pressure_resampling_constraints(_source(),ValueError("not a resource failure"),**arguments)
    error=WaveGridBudgetError("Limit")
    error.__context__=WaveSamplingError("Unbounded",required_scale=math.inf)
    with pytest.raises(ValueError,match="unbounded"):
        _pressure_resampling_constraints(_source(),error,**arguments)


def test_initial_host_preflight_projects_with_original_one_percent_limits():
    source=_source()
    numeric=WaveGridNumerics(maximum_working_bytes=70*1024**2)
    prepare,state=_projection(numeric)
    visited=[]
    def operator(wave):
        visited.append(wave)
        return wave
    result,rows=apply_resolved_operator(source,operator,numerics=numeric,
        operator_working_bytes=lambda shape:512*math.prod(shape)+64*1024**2,
        prepare_resource=prepare)
    assert [wave.amplitude.shape for wave in visited]==[(32,32)]
    assert state["calls"][0][0] is source
    assert rows[0]["resource_stage"]=="operator resource preflight"
    ledger=state["ledger"]
    assert len(ledger.events)==1
    assert ledger.used_l2<=ledger.maximum_l2/100
    assert ledger.used_probability<=ledger.maximum_probability/100
    common=refine_plane_wave(result,source.amplitude.shape)
    assert np.linalg.norm(common.amplitude-source.amplitude)==pytest.approx(
        ledger.events[0]["absolute_l2_error"],rel=2e-10)
    for field in ("origin_m","curvature_m1","tilt_rad"):
        np.testing.assert_array_equal(getattr(result,field),getattr(source,field))


@pytest.mark.parametrize("location",["operator","refinement"])
def test_resource_failure_retries_current_entrance_and_recomputes_sampling(location):
    source=_source();numeric=WaveGridNumerics(maximum_working_bytes=64*1024**2)
    prepare,state=_projection(numeric);visited=[]
    def resource(shape):
        if max(shape)>160:raise WaveGridBudgetError("refinement device partition")
    def operator(wave):
        visited.append(wave)
        if wave.amplitude.shape==(128,128):
            if location=="operator":raise WaveMemoryBudgetError("operator working partition")
            raise WaveSamplingError("unchanged input needs more resolution",required_scale=2.)
        return wave
    result,rows=apply_resolved_operator(source,operator,numerics=numeric,
        operator_resource_check=resource,prepare_resource=prepare)
    assert [wave.amplitude.shape for wave in visited]==[(128,128),(32,32)]
    assert state["calls"][0][0] is source
    assert len(state["calls"])==1 and len(state["ledger"].events)==1
    assert rows[0]["resource_stage"]==("physical operator" if location=="operator" else "Fourier refinement")
    assert result.amplitude.shape==(32,32)


@pytest.mark.parametrize("proposal",["unchanged","metadata_copy","same_grid_copy","still_over_budget"])
def test_declined_or_inadequate_proposal_keeps_original_resource_error(proposal):
    source=_source();failure=WaveMemoryBudgetError("original failure");calls=[]
    def operator(wave):raise failure
    def prepare(wave,error):
        calls.append(error)
        if proposal=="unchanged":return wave
        if proposal=="metadata_copy":return wave._replace_metadata()
        if proposal=="same_grid_copy":return replace(wave,amplitude=wave.amplitude.copy())
        raise WaveGridBudgetError("proposal cannot fit")
    with pytest.raises(WaveMemoryBudgetError) as captured:
        apply_resolved_operator(source,operator,prepare_resource=prepare)
    assert captured.value is failure and calls==[failure]


def test_resource_preparation_is_not_reissued_after_a_successful_projection():
    source=_source();numeric=WaveGridNumerics(maximum_working_bytes=64*1024**2)
    prepare,state=_projection(numeric);failure=WaveMemoryBudgetError("still cannot fit")
    with pytest.raises(WaveMemoryBudgetError) as captured:
        apply_resolved_operator(source,lambda wave:(_ for _ in ()).throw(failure),
            numerics=numeric,prepare_resource=prepare)
    assert captured.value is failure
    assert len(state["calls"])==1 and len(state["ledger"].events)==1


def test_failed_operator_output_is_released_and_never_projected():
    source=_source();numeric=WaveGridNumerics(maximum_working_bytes=64*1024**2)
    project,state=_projection(numeric);abandoned=[]
    def operator(wave):
        if wave is source:
            failed_output=wave.amplitude.copy()*1j
            abandoned.append(weakref.ref(failed_output))
            raise WaveMemoryBudgetError("private output exhausted workspace")
        return wave
    def prepare(wave,error):
        assert wave is source
        assert abandoned and all(value() is None for value in abandoned)
        return project(wave,error)
    result,_=apply_resolved_operator(source,operator,numerics=numeric,prepare_resource=prepare)
    assert result.amplitude.shape==(32,32)


def test_failed_output_is_released_before_ordinary_lossless_refinement(monkeypatch):
    import temsim.physics.wave_grid as grid
    source = _source()
    abandoned, visited, requests = [], [], []
    failure = WaveSamplingError("ordinary refinement preserves this diagnostic", 1.01)
    def operator(wave):
        visited.append(wave.amplitude.shape)
        if wave is source:
            failed_output = wave.amplitude.copy()*1j
            abandoned.append(weakref.ref(failed_output))
            raise failure
        return wave
    original_refine = grid.refine_plane_wave
    def refine(wave, shape, **kwargs):
        assert abandoned and all(value() is None for value in abandoned)
        assert wave is source
        assert str(failure) == "ordinary refinement preserves this diagnostic"
        assert failure.required_scale == 1.01
        requests.append(shape)
        return original_refine(wave, shape, **kwargs)
    monkeypatch.setattr(grid, "refine_plane_wave", refine)
    output, rows = grid.apply_resolved_operator(source, operator)
    assert visited == [(128, 128), (160, 160)]
    assert requests == [(160, 160)]
    assert rows[0]["requested_scale"] == 1.01
    np.testing.assert_array_equal(output.amplitude, original_refine(source, (160, 160)).amplitude)


@pytest.mark.parametrize("failure",[ValueError("unknown field domain"),InterruptedError("cancelled")])
def test_nonresource_failures_never_prepare_or_change_error(failure):
    def prepare(*args):pytest.fail("physical failure or cancellation invoked a projection")
    with pytest.raises(type(failure)) as captured:
        apply_resolved_operator(_source(),lambda wave:(_ for _ in ()).throw(failure),prepare_resource=prepare)
    assert captured.value is failure


def test_disabled_refinement_and_missing_optin_keep_original_failure():
    failure=WaveMemoryBudgetError("explicit resource limit")
    def prepare(*args):pytest.fail("disabled retry invoked a projection")
    for callback,numeric in ((None,WaveGridNumerics()),(prepare,WaveGridNumerics(automatic_refinement=False))):
        with pytest.raises(WaveMemoryBudgetError) as captured:
            apply_resolved_operator(_source(),lambda wave:(_ for _ in ()).throw(failure),
                numerics=numeric,prepare_resource=callback)
        assert captured.value is failure


def test_cancellation_during_preparation_is_not_resource_fallback():
    source=_source();cancelled=[False]
    def prepare(wave,error):
        cancelled[0]=True
        return wave
    with pytest.raises(InterruptedError):
        apply_resolved_operator(source,lambda wave:(_ for _ in ()).throw(WaveMemoryBudgetError("budget")),
            prepare_resource=prepare,cancelled=lambda:cancelled[0])


def test_invalid_resource_callback_rejected_before_operator():
    with pytest.raises(ValueError,match="callable"):
        apply_resolved_operator(_source(),lambda wave:pytest.fail("invalid callback executed"),prepare_resource=True)


def test_carrier_weighted_resource_and_sampling_retry_share_one_ledger_allocation(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot,ENERGY_EV
    source=_source();initial=NumericalSupportLedger(.02,1e-6)
    original_resample=NumericalSupportLedger.resample_spectrum_pressure
    original_residual=carrier._residual
    state={"calls":0,"pressure":False,"projected":False}
    def resample(self,wave,**options):
        if not state["pressure"] or state["projected"]:return wave,self
        fit=options["admissible_shape"]
        options["admissible_shape"]=lambda shape: max(shape)<=80 and fit(shape)
        output,ledger=original_resample(self,wave,**options)
        state["projected"]=output is not wave
        return output,ledger
    def residual(snapshot,wave,*args,**kwargs):
        state["calls"]+=1
        if state["calls"]==1:
            state["pressure"]=True
            raise WaveMemoryBudgetError("synthetic residual resource pressure")
        if state["calls"]==2:
            raise WaveSamplingError("projected entrance needs lossless refinement",1.01)
        return original_residual(snapshot,wave,*args,**kwargs)
    monkeypatch.setattr(NumericalSupportLedger,"resample_spectrum_pressure",resample)
    monkeypatch.setattr(NumericalSupportLedger,"compact_pressure",lambda self,wave,**options:(wave,self))
    monkeypatch.setattr(carrier,"_residual",residual)
    _,record=carrier.propagate_carrier_potential(_snapshot(),source,(0.,1e-9),energy_ev=ENERGY_EV,
        kinetic_order=2,support_ledger=initial,support_mode_weight=.25,
        support_projection_policy="weighted_segment_and_retry")
    ledger=NumericalSupportLedger.from_dict(record["support_ledger"])
    assert state["calls"]>=3 and len(ledger.events)==1 and initial.events==()
    assert ledger.used_probability<=initial.maximum_probability/100
    assert ledger.used_l2<=initial.maximum_l2/100
    assert any("resource preparation" in row["method"] for row in record["canonical_sampling"])


@pytest.mark.parametrize("policy",["sampling_retry","step_start_and_retry","step_boundaries_and_retry"])
def test_other_carrier_policies_do_not_enable_resource_projection(monkeypatch,policy):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot,ENERGY_EV
    failure=WaveMemoryBudgetError("unchanged other-policy failure")
    monkeypatch.setattr(carrier,"_residual",lambda *a,**k:(_ for _ in ()).throw(failure))
    with pytest.raises(WaveMemoryBudgetError) as captured:
        carrier.propagate_carrier_potential(_snapshot(),_source(),(0.,1e-9),energy_ev=ENERGY_EV,
            kinetic_order=2,support_ledger=NumericalSupportLedger(.02,1e-6),
            support_mode_weight=.25,support_projection_policy=policy)
    assert captured.value is failure


@pytest.mark.parametrize("sampling_retry", [False, True])
def test_weighted_policy_preserves_budget_until_actual_resource_pressure(monkeypatch, sampling_retry):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    original_residual = carrier._residual
    calls = []
    def residual(snapshot, wave, *args, **kwargs):
        calls.append(wave.amplitude.shape)
        if sampling_retry and len(calls) == 1:
            raise WaveSamplingError("ordinary resolvable operator sampling", 1.01)
        return original_residual(snapshot, wave, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("A resource-sufficient step spent projection budget")
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum", forbidden)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", forbidden)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", forbidden)
    source = _source()
    result, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9, 2e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=NumericalSupportLedger(.02, 1e-6),
        support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    assert record["support_ledger"]["events"] == []
    assert record["support_projection_trigger"] == "typed_resource_failure_only"
    assert record["input_flux"] == source.probability
    assert result.probability == pytest.approx(source.probability, abs=1e-13)
    if sampling_retry:
        assert any(shape[0] > source.amplitude.shape[0] for shape in calls)


def test_entrance_resource_projection_retains_original_input_flux(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    original = carrier._validate_wave
    first = [True]
    def validate(wave, *args):
        if first[0]:
            first[0] = False
            raise WaveMemoryBudgetError("entrance validation resource pressure")
        return original(wave, *args)
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    original_constraints = carrier._pressure_resampling_constraints
    def constraints(*args, **kwargs):
        fit, context = original_constraints(*args, **kwargs)
        # The synthetic validation failure represents a stage that can admit
        # at most this shape; keep the real host/device limits in force too.
        return lambda shape: max(shape) <= 80 and fit(shape), context
    monkeypatch.setattr(carrier, "_pressure_resampling_constraints", constraints)
    source = _source()
    result, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=NumericalSupportLedger(.02, 1e-6),
        support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert len(ledger.events) == 1
    assert record["input_flux"] == source.probability
    assert result.probability < source.probability
    assert .25*(source.probability-result.probability) == pytest.approx(ledger.used_probability, abs=2e-15)


def _transaction(numerics, *, maximum_attempts=6):
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    initial = NumericalSupportLedger(.02, 1e-6)
    state = {"ledger": initial, "proposals": [], "commits": [], "weak": []}
    def prepare(anchor, error, pressure):
        assert pressure is not None
        assert state["ledger"] is initial
        # Check that failed private outputs and old proposal storage have
        # gone before another Fourier transform can allocate its workspace.
        assert all(value() is None for value in state["weak"])
        target = 96 if pressure.attempt == 1 else 64
        _, context = _pressure_resampling_constraints(anchor, error,
            numerics=numerics, requested="CPU", retained_bytes=0,
            stage="transaction test", pressure=pressure)
        proposed, pending = initial.resample_spectrum_pressure(anchor, z_m=.5,
            mode_weight=.25, numerics=numerics,
            admissible_shape=lambda shape: max(shape) <= target, resource_context=context)
        state["proposals"].append((anchor, pressure, proposed.amplitude.shape, pending))
        state["weak"].append(weakref.ref(proposed.amplitude))
        def commit():
            assert state["ledger"] is initial
            state["ledger"] = pending
            state["commits"].append(pressure.attempt)
        return WaveResourceProposal(proposed, commit)
    return WaveResourceTransaction(prepare, maximum_attempts), state, initial


def test_transaction_retries_original_complex_entrance_after_multiple_sampling_grows():
    source = _source()
    original = source.amplitude.copy()
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    transaction, state, initial = _transaction(numeric)
    visits, failed_outputs = [], []
    def operator(wave):
        n = wave.amplitude.shape[0]
        visits.append(n)
        assert state["ledger"] is initial
        if n == 64:
            assert all(value() is None for value in failed_outputs)
            return wave._replace_owned(wave.amplitude*np.exp(.3j))
        output = wave.amplitude.copy()*1j
        failed_outputs.append(weakref.ref(output))
        raise WaveSamplingError("private physical output needs finer sampling",
            1.01 if n == 96 else 1.1)
    result, rows = apply_resolved_operator(source, operator, numerics=numeric,
        prepare_resource=transaction)
    assert visits == [128, 96, 120, 64]
    assert state["commits"] == [2]
    assert len(state["ledger"].events) == 1 and initial.events == ()
    assert all(entry[0] is source for entry in state["proposals"])
    first, second = (entry[1] for entry in state["proposals"])
    assert first.previous_shape is None
    assert second.previous_shape == (96, 96)
    assert second.required_shape == (140, 140)
    assert second.growth_scale == pytest.approx(140/96)
    assert len(rows) == 1 and rows[0]["to_shape"] == (64, 64)
    np.testing.assert_array_equal(source.amplitude, original)
    common = refine_plane_wave(result, source.amplitude.shape)
    error = np.linalg.norm(common.amplitude-source.amplitude*np.exp(.3j))
    assert error**2 == pytest.approx(
        state["ledger"].events[0]["absolute_probability_loss"], rel=2e-9)


@pytest.mark.parametrize("failure", ["budget", "domain", "cancel_inside", "cancel_after"])
def test_transaction_failure_never_publishes_an_allowance(failure):
    source = _source()
    numeric = WaveGridNumerics(maximum_working_bytes=256*1024**2)
    transaction, state, initial = _transaction(numeric, maximum_attempts=2)
    cancelled = [False]
    def operator(wave):
        if wave is source:
            raise WaveMemoryBudgetError("initial pressure")
        if failure == "budget":
            raise WaveMemoryBudgetError("bounded candidate remains too expensive")
        if failure == "domain":
            raise ValueError("unknown physical domain")
        if failure == "cancel_inside":
            raise InterruptedError("cancelled private physical operator")
        cancelled[0] = True
        return wave
    expected = WaveGridBudgetError if failure == "budget" else (
        ValueError if failure == "domain" else InterruptedError)
    with pytest.raises(expected):
        apply_resolved_operator(source, operator, numerics=numeric,
            prepare_resource=transaction, cancelled=lambda: cancelled[0])
    assert state["ledger"] is initial and state["commits"] == []
    assert len(state["proposals"]) == (2 if failure == "budget" else 1)


def test_transaction_counts_a_distinct_refined_anchor_and_outer_input(monkeypatch):
    source = _source()
    numeric = WaveGridNumerics(maximum_working_bytes=256*1024**2)
    transaction, state, _ = _transaction(numeric)
    checks = []
    original_check = WaveGridNumerics.check_working_memory
    def check(self, shape, *, working_bytes, retained_bytes=0):
        checks.append((shape, retained_bytes))
        return original_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    def operator(wave):
        if wave is source:
            raise WaveSamplingError("first lossless preparation", 1.01)
        if wave.amplitude.shape == (160, 160):
            raise WaveMemoryBudgetError("pressure on a distinct unexecuted refined input")
        return wave
    result, _ = apply_resolved_operator(source, operator, numerics=numeric,
        retained_bytes=123, operator_working_bytes=lambda shape: 256*math.prod(shape),
        prepare_resource=transaction)
    assert result.amplitude.shape == (96, 96)
    anchor = state["proposals"][0][0]
    assert anchor is not source and anchor.amplitude.shape == (160, 160)
    from temsim.cache_memory import retained_memory_inventory
    expected = 123+sum(retained_memory_inventory(source.amplitude, anchor.amplitude).values())
    assert ((96, 96), expected) in checks


@pytest.mark.parametrize("target", [96, 100])
def test_transaction_rejects_same_or_larger_replacement_without_commit(target):
    source = _source()
    commits, calls = [], []
    def prepare(anchor, error, pressure):
        calls.append(pressure)
        size = 96 if pressure.attempt == 1 else target
        proposal = PlaneWave(np.ones((size, size), complex)/size, np.eye(2), np.zeros(2))
        return WaveResourceProposal(proposal, lambda: commits.append(size))
    with pytest.raises(WaveMemoryBudgetError, match="pressure"):
        apply_resolved_operator(source,
            lambda wave: (_ for _ in ()).throw(WaveMemoryBudgetError("pressure")),
            prepare_resource=WaveResourceTransaction(prepare))
    assert len(calls) == 2 and commits == []


def test_pressure_forecast_uses_cumulative_measured_growth_and_monotone_target():
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    pressure = WaveResourcePressure(2, (96, 96), (140, 140), 140/96)
    fits, context = _pressure_resampling_constraints(_source(), WaveGridBudgetError("cap"),
        numerics=WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2),
        requested="CPU", retained_bytes=0, stage="residual retry", pressure=pressure)
    assert context["growth_guard_scale"] == 140/96
    assert fits((80, 80)) and not fits((90, 90)) and not fits((96, 96))


def test_transaction_declined_second_proposal_does_not_commit_the_first():
    source = _source()
    calls, commits = [], []
    def prepare(anchor, error, pressure):
        calls.append(anchor)
        if pressure.attempt == 2:
            return anchor
        proposed = PlaneWave(np.ones((96, 96), complex)/96, np.eye(2), np.zeros(2))
        return WaveResourceProposal(proposed, lambda: commits.append(1))
    with pytest.raises(WaveMemoryBudgetError, match="pressure"):
        apply_resolved_operator(source,
            lambda wave: (_ for _ in ()).throw(WaveMemoryBudgetError("pressure")),
            prepare_resource=WaveResourceTransaction(prepare))
    assert calls == [source, source] and commits == []


def test_carrier_second_pressure_candidate_spends_only_its_original_input_projection(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source = _source()
    initial = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_residual = carrier._residual
    real_projection = NumericalSupportLedger.resample_spectrum_pressure
    proposals, calls = [], []
    def resample(self, wave, **options):
        output, ledger = real_projection(self, wave, **options)
        proposals.append((self, wave, output.amplitude.shape, ledger))
        return output, ledger
    def residual(snapshot, wave, *args, **options):
        n = wave.amplitude.shape[0]
        calls.append(n)
        if len(calls) == 1:
            raise WaveMemoryBudgetError("initial actual residual pressure")
        if n in (96, 120):
            raise WaveSamplingError("measured projected residual bandwidth", 1.01 if n == 96 else 1.1)
        return real_residual(snapshot, wave, *args, **options)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", resample)
    monkeypatch.setattr(carrier, "_residual", residual)
    output, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric, support_ledger=initial,
        support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    assert calls[:4] == [128, 96, 120, 80]
    assert len(proposals) == 2
    assert all(ledger is initial and wave is source for ledger, wave, _, _ in proposals)
    accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert accepted == proposals[-1][3] and len(accepted.events) == 1
    assert accepted.events[-1]["to_shape"] == (80, 80)
    assert initial.events == ()
    assert len([row for row in record["canonical_sampling"]
        if row["method"] == "budgeted support projection with full candidate admission"]) == 1
    from temsim.physics.potential_wave_carrier import _pressure_resampling_constraints
    _, context = _pressure_resampling_constraints(source, WaveGridBudgetError("reference"),
        numerics=numeric, requested="CPU", retained_bytes=0, stage="reference")
    projected, _ = real_projection(initial, source, z_m=0., mode_weight=.25,
        numerics=numeric, admissible_shape=lambda shape: max(shape) <= 80, resource_context=context)
    monkeypatch.setattr(carrier, "_residual", real_residual)
    expected, _ = carrier.propagate_carrier_potential(_snapshot(), projected, (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric)
    np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=0, atol=2e-14)
    assert output.probability == pytest.approx(expected.probability, abs=2e-14)


def test_canonical_transaction_releases_its_entrance_before_the_next_residual(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    real_canonical, real_residual = carrier.propagate_plane_wave, carrier._residual
    entrance = []
    def canonical(wave, *args, **options):
        entrance.append(weakref.ref(wave.amplitude))
        output = real_canonical(wave, *args, **options)
        return output._replace_owned(output.amplitude.copy())
    def residual(snapshot, wave, *args, **options):
        if entrance:
            assert all(value() is None for value in entrance)
        return real_residual(snapshot, wave, *args, **options)
    monkeypatch.setattr(carrier, "propagate_plane_wave", canonical)
    monkeypatch.setattr(carrier, "_residual", residual)
    carrier.propagate_carrier_potential(_snapshot(), _source(), (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, support_ledger=NumericalSupportLedger(.02, 1e-6),
        support_mode_weight=.25, support_projection_policy="weighted_segment_and_retry")
    assert entrance and all(value() is None for value in entrance)


def _narrow_source():
    source = _source()
    y, x = np.indices(source.amplitude.shape, dtype=float)
    amplitude = np.exp(-((x-64)**2+(y-64)**2)/64.)*np.exp(.37j)
    amplitude /= np.linalg.norm(amplitude)
    return source._replace_owned(amplitude)


@pytest.mark.parametrize("decline", [None, "no_crop", "admission"])
def test_scalar_phase_resource_pressure_prefers_existing_spatial_window(monkeypatch, decline):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source = _narrow_source()
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    initial = NumericalSupportLedger(.02, 1e-6)
    real_compact, real_spectral = NumericalSupportLedger.compact_pressure, NumericalSupportLedger.resample_spectrum_pressure
    real_residual, real_validate = carrier._residual, carrier._validate_wave
    calls, windows, residuals = [], [], []
    def compact(self, wave, **options):
        assert self is initial and wave is source
        calls.append("spatial")
        if decline == "no_crop":
            return wave, self
        output, ledger = real_compact(self, wave, **options)
        for field in ("basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(output, field), getattr(source, field))
        windows.append(weakref.ref(output.amplitude))
        return output, ledger
    def spectral(self, wave, **options):
        assert decline is not None
        assert self is initial and wave is source
        assert all(value() is None for value in windows)
        calls.append("spectral")
        return real_spectral(self, wave, **options)
    def validate(wave, *args):
        if (decline == "admission" and windows and windows[-1]() is wave.amplitude):
            raise carrier.OccupiedCarrierBoundaryError("synthetic unadmitted proposed window")
        return real_validate(wave, *args)
    def residual(snapshot, wave, *args, **options):
        residuals.append(wave.amplitude.shape)
        if len(residuals) == 1:
            raise carrier.ScalarPhaseSamplingError("full-domain scalar phase needs refinement", 2.)
        return real_residual(snapshot, wave, *args, **options)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", compact)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    result, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=initial, support_mode_weight=.25,
        support_projection_policy="weighted_segment_and_retry")
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert calls == (["spatial"] if decline is None else ["spatial"]*7+["spectral"])
    assert len(ledger.events) == 1 and initial.events == ()
    assert ledger.events[0]["kind"] == (
        "numerical_support_projection" if decline is None else "numerical_spectral_projection")
    if decline is None:
        assert ledger.used_probability <= initial.maximum_probability/100
        assert ledger.used_l2 <= initial.maximum_l2/100
        assert result.probability == pytest.approx(source.probability-ledger.used_probability/.25, abs=2e-14)


def test_scalar_pressure_detection_requires_the_typed_cause_and_handles_cycles():
    from temsim.physics.potential_wave_carrier import ScalarPhaseSamplingError, _scalar_phase_resource_pressure
    failure = WaveGridBudgetError("limit")
    ordinary = WaveSamplingError("Carrier residual scalar phase is undersampled", 2.)
    failure.__context__ = ordinary
    assert not _scalar_phase_resource_pressure(failure)
    scalar = ScalarPhaseSamplingError("scalar", 2.)
    ordinary.__cause__ = scalar
    scalar.__context__ = failure
    assert _scalar_phase_resource_pressure(failure)
    assert not _scalar_phase_resource_pressure(scalar)


def test_scalar_crop_preflight_declines_without_allocating_or_changing_ledger(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source = _narrow_source()
    initial = NumericalSupportLedger(.02, 1e-6)
    original_check = WaveGridNumerics.check_working_memory
    pressure = [False]
    def check(self, shape, *, working_bytes, retained_bytes=0):
        if pressure[0] and tuple(shape) == source.amplitude.shape and working_bytes == 256*source.amplitude.size:
            raise WaveMemoryBudgetError("crop workspace cannot fit")
        return original_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)
    def residual(*args, **options):
        pressure[0] = True
        raise carrier.ScalarPhaseSamplingError("scalar pressure", 2.)
    def compact(*args, **options):
        pytest.fail("An unaffordable spatial window was allocated")
    def spectral(self, wave, **options):
        assert self is initial and wave is source
        return wave, self
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", compact)
    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    with pytest.raises(WaveGridBudgetError):
        carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
            energy_ev=ENERGY_EV, kinetic_order=2,
            grid_numerics=WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2),
            support_ledger=initial, support_mode_weight=.25,
            support_projection_policy="weighted_segment_and_retry")
    assert initial.events == ()


@pytest.mark.parametrize("spectral_decline", ["boundary", "sampling", "no_candidate", "workspace"])
def test_generic_pressure_retries_spatial_from_original_after_spectral_decline(monkeypatch, spectral_decline):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source = _narrow_source()
    original_amplitude = source.amplitude.copy()
    initial = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_spectral = NumericalSupportLedger.resample_spectrum_pressure
    real_compact = NumericalSupportLedger.compact_pressure
    real_residual, real_validate = carrier._residual, carrier._validate_wave
    calls, failed_arrays, residuals, accepted_windows = [], [], [], []

    def spectral(self, wave, **options):
        assert self is initial and wave is source
        calls.append("spectral")
        if spectral_decline == "no_candidate":
            return wave, self
        if spectral_decline == "workspace":
            raise WaveMemoryBudgetError("controlled spectral workspace failure")
        output, ledger = real_spectral(self, wave, **options)
        assert output is not wave
        failed_arrays.append(weakref.ref(output.amplitude))
        return output, ledger

    def compact(self, wave, **options):
        assert self is initial and wave is source
        assert all(value() is None for value in failed_arrays)
        calls.append("spatial")
        output, ledger = real_compact(self, wave, **options)
        accepted_windows.append((output, ledger))
        return output, ledger

    def validate(wave, *args):
        if failed_arrays and failed_arrays[-1]() is wave.amplitude:
            if spectral_decline == "sampling":
                raise WaveSamplingError("controlled unresolvable spectral admission", 3.)
            raise carrier.OccupiedCarrierBoundaryError("controlled spectral boundary ringing")
        return real_validate(wave, *args)

    def residual(snapshot, wave, *args, **options):
        residuals.append(wave.amplitude.shape)
        if len(residuals) == 1:
            raise WaveSamplingError("controlled physical envelope resource pressure", 2.)
        return real_residual(snapshot, wave, *args, **options)

    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", compact)
    # Exercise the independent spatial alternative when no optional taper is
    # available; the combined spectral/taper transaction has separate coverage.
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", lambda self, wave, **options: (wave, self))
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    output, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=initial, support_mode_weight=.25,
        support_projection_policy="weighted_segment_and_retry")
    assert calls == ["spectral", "spatial"]
    accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
    assert initial.events == () and len(accepted.events) == 1
    assert accepted == accepted_windows[0][1]
    assert accepted.events[0]["kind"] == "numerical_support_projection"
    np.testing.assert_array_equal(source.amplitude, original_amplitude)
    assert any(row.get("candidate_kind") == "spectral" and not row["projection_budget_spent"]
        for row in record["canonical_sampling"]
        if row["method"] == "support projection proposal declined")
    monkeypatch.setattr(carrier, "_residual", real_residual)
    expected, _ = carrier.propagate_carrier_potential(_snapshot(), accepted_windows[0][0], (0., 1e-9),
        energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric)
    np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=0, atol=2e-14)
    assert output.probability == pytest.approx(expected.probability, abs=2e-14)


@pytest.mark.parametrize("failure", ["both_rejected", "both_unavailable", "spatial_unavailable",
    "spatial_workspace", "cancelled", "domain"])
def test_generic_pressure_failed_candidates_preserve_ledger_and_readable_reason(monkeypatch, failure):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source = _narrow_source()
    original_amplitude = source.amplitude.copy()
    initial = NumericalSupportLedger(.02, 1e-6)
    real_spectral = NumericalSupportLedger.resample_spectrum_pressure
    real_compact = NumericalSupportLedger.compact_pressure
    real_validate = carrier._validate_wave
    real_check = WaveGridNumerics.check_working_memory
    calls, proposed, pressure = [], [], []

    def spectral(self, wave, **options):
        assert self is initial and wave is source
        calls.append("spectral")
        if failure == "both_unavailable":
            return wave, self
        output, ledger = real_spectral(self, wave, **options)
        proposed.append(("spectral", weakref.ref(output.amplitude)))
        return output, ledger

    def compact(self, wave, **options):
        assert failure != "spatial_workspace", "An unaffordable window was allocated"
        assert self is initial and wave is source
        assert all(value() is None for _, value in proposed)
        calls.append("spatial")
        if failure in ("both_unavailable", "spatial_unavailable"):
            return wave, self
        output, ledger = real_compact(self, wave, **options)
        proposed.append(("spatial", weakref.ref(output.amplitude)))
        return output, ledger

    def validate(wave, *args):
        if proposed and proposed[-1][1]() is wave.amplitude:
            if failure == "cancelled":
                raise InterruptedError("controlled cancellation during candidate admission")
            if failure == "domain":
                raise ValueError("controlled physical domain mismatch")
            raise carrier.OccupiedCarrierBoundaryError(f"controlled {proposed[-1][0]} boundary rejection")
        return real_validate(wave, *args)

    def check(self, shape, *, working_bytes, retained_bytes=0):
        if (failure == "spatial_workspace" and proposed and pressure
                and tuple(shape) == source.amplitude.shape and working_bytes == 256*source.amplitude.size):
            raise WaveMemoryBudgetError("controlled spatial workspace failure")
        return real_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)

    def residual(*args, **options):
        pressure.append(True)
        raise WaveSamplingError("controlled physical envelope resource pressure", 2.)

    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", compact)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", lambda self, wave, **options: (wave, self))
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    expected_error = (InterruptedError if failure == "cancelled"
        else ValueError if failure == "domain" else WaveGridBudgetError)
    with pytest.raises(expected_error) as caught:
        carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9),
            energy_ev=ENERGY_EV, kinetic_order=2,
            grid_numerics=WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2),
            support_ledger=initial, support_mode_weight=.25,
            support_projection_policy="weighted_segment_and_retry")
    assert initial.events == ()
    np.testing.assert_array_equal(source.amplitude, original_amplitude)
    assert calls == (["spectral"]+["spatial"]*7 if failure in
        ("both_rejected", "both_unavailable", "spatial_unavailable") else ["spectral"])
    if failure in ("both_rejected", "both_unavailable", "spatial_unavailable", "spatial_workspace"):
        notes = getattr(caught.value, "__notes__", ())
        assert len(notes) == (8 if failure == "both_unavailable" else 12)
        if failure != "both_unavailable":
            for width in (2, 4, 8, 16):
                assert any(f"The {width}-cell window" in note for note in notes)
        assert "Support spectral candidate declined" in notes[0]
        assert ("No smaller candidate" if failure == "both_unavailable"
            else "OccupiedCarrierBoundaryError") in notes[0]
        assert "Support spatial candidate declined" in notes[-1]
        assert all(value() is None for _, value in proposed)


@pytest.mark.parametrize("outcome", ["success", "cancelled", "workspace", "exhausted", "domain", "physical_failure"])
def test_spectral_admission_growth_reselects_from_original_without_spending_failed_pairs(monkeypatch, outcome):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source, weight = _narrow_source(), .25
    original_amplitude = source.amplitude.copy()
    initial = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_spectral, real_taper = NumericalSupportLedger.resample_spectrum_pressure, NumericalSupportLedger.taper_boundary
    real_validate, real_residual = carrier._validate_wave, carrier._residual
    real_proposal = carrier.WaveResourceProposal
    state = {"raw": None, "taper": None, "pending": None, "combined": None,
        "calls": 0, "residuals": 0, "commits": 0, "cancelled": False,
        "raw_shapes": [], "growth": [], "failed_arrays": [], "accepted_input": None}

    def spectral(self, wave, **options):
        assert self is initial and wave is source
        assert options["mode_weight"] == weight
        assert all(reference() is None for reference in state["failed_arrays"])
        state["calls"] += 1
        if state["calls"] > 1:
            if outcome == "cancelled":
                state["cancelled"] = True
                raise InterruptedError("controlled cancellation before replacement projection")
            if outcome == "workspace":
                raise WaveMemoryBudgetError("controlled replacement projection workspace")
            if outcome == "exhausted":
                return wave, self
        output, pending = real_spectral(self, wave, **options)
        assert output is not wave
        state["raw"] = weakref.ref(output.amplitude)
        state["failed_arrays"].append(state["raw"])
        state["pending"] = pending
        state["raw_shapes"].append(output.amplitude.shape)
        state["growth"].append(options["resource_context"]["growth_guard_scale"])
        return output, pending

    def taper(self, wave, **options):
        assert self is state["pending"] and state["raw"]() is wave.amplitude
        assert options["mode_weight"] == weight
        output, combined = real_taper(self, wave, **options)
        assert output is not wave
        state["taper"] = weakref.ref(output.amplitude)
        state["failed_arrays"].append(state["taper"])
        state["combined"] = combined
        return output, combined

    def validate(wave, *args):
        if state["raw"] is not None and state["raw"]() is wave.amplitude:
            raise carrier.OccupiedCarrierBoundaryError("controlled pending boundary ringing")
        if state["taper"] is not None and state["taper"]() is wave.amplitude:
            if outcome == "domain" and state["calls"] > 1:
                raise ValueError("controlled candidate field domain mismatch")
            # This is an admission-only need for exact Fourier embedding. The
            # raw forecast fits, but the first admitted lattice does not leave
            # the original physical-operator growth reserve.
            raise WaveSamplingError("controlled pending taper bandwidth", 1.5)
        return real_validate(wave, *args)

    def residual(snapshot, wave, *args, **options):
        state["residuals"] += 1
        if state["residuals"] == 1:
            raise WaveSamplingError("controlled physical envelope resource pressure", 2.)
        assert state["calls"] >= 2
        if state["residuals"] == 2:
            assert state["commits"] == 0
            state["accepted_input"] = PlaneWave(wave.amplitude.copy(), wave.basis_m.copy(),
                wave.origin_m.copy(), wave.curvature_m1.copy(), wave.tilt_rad.copy())
        if outcome == "physical_failure":
            raise ValueError("controlled physical failure after reselected admission")
        return real_residual(snapshot, wave, *args, **options)

    def proposal(wave, commit):
        def counted_commit():
            state["commits"] += 1
            commit()
        return real_proposal(wave, counted_commit)

    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", lambda self, wave, **options: (wave, self))
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(carrier, "WaveResourceProposal", proposal)
    options = dict(energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=initial, support_mode_weight=weight,
        support_projection_policy="weighted_segment_and_retry", cancelled=lambda: state["cancelled"])
    if outcome == "success":
        output, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9), **options)
        accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
        assert accepted == state["combined"] and len(accepted.events) == 2
        assert state["commits"] == 1
        assert [event["kind"] for event in accepted.events] == [
            "numerical_spectral_projection", "numerical_boundary_taper"]
        assert all(event["mode_weight"] == weight for event in accepted.events)
        assert len(state["raw_shapes"]) >= 2
        assert all(b[0] < a[0] for a, b in zip(state["raw_shapes"], state["raw_shapes"][1:]))
        assert state["growth"][1] > state["growth"][0]
        retries = [row for row in record["canonical_sampling"]
            if row["method"] == "support candidate admission forecast retry"]
        assert retries and all(not row["projection_budget_spent"] for row in retries)
        for row in retries:
            assert row["composite_growth_guard_scale"] == pytest.approx(
                row["operator_growth_guard_scale"]*row["admission_growth_scale"])
        monkeypatch.setattr(carrier, "_residual", real_residual)
        monkeypatch.setattr(carrier, "_validate_wave", real_validate)
        expected, _ = carrier.propagate_carrier_potential(_snapshot(), state["accepted_input"], (0., 1e-9),
            energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric)
        np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=0., atol=2e-14)
        assert output.probability == pytest.approx(source.probability-accepted.used_probability/weight, abs=2e-14)
    else:
        expected_error = InterruptedError if outcome == "cancelled" else (
            ValueError if outcome in ("domain", "physical_failure") else WaveGridBudgetError)
        with pytest.raises(expected_error):
            carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9), **options)
        assert state["commits"] == 0
        assert all(reference() is None for reference in state["failed_arrays"])
    assert 2 <= state["calls"] <= 6
    assert initial.events == ()
    np.testing.assert_array_equal(source.amplitude, original_amplitude)


@pytest.mark.parametrize("outcome", ["success", "no_taper", "taper_alias", "workspace",
    "cancel_before", "cancel_after", "physical_failure", "domain"])
def test_pending_spectral_taper_is_one_budgeted_original_weight_transaction(monkeypatch, outcome):
    import temsim.physics.potential_wave_carrier as carrier
    from test_potential_wave_carrier import _snapshot, ENERGY_EV
    source, weight = _narrow_source(), .25
    original_amplitude = source.amplitude.copy()
    initial = NumericalSupportLedger(.02, 1e-6)
    numeric = WaveGridNumerics(maximum_pixels=128, maximum_working_bytes=256*1024**2)
    real_spectral = NumericalSupportLedger.resample_spectrum_pressure
    real_taper = NumericalSupportLedger.taper_boundary
    real_validate, real_residual = carrier._validate_wave, carrier._residual
    real_check, real_proposal = WaveGridNumerics.check_working_memory, carrier.WaveResourceProposal
    state = {"spectral": None, "taper": None, "pending": None, "combined": None,
        "cancelled": False, "boundary_rejected": False, "residual_calls": 0,
        "taper_calls": 0, "commits": 0, "reservations": [], "options": None}

    def spectral(self, wave, **options):
        assert self is initial and wave is source
        output, pending = real_spectral(self, wave, **options)
        assert output is not wave
        state.update(spectral=weakref.ref(output.amplitude), pending=pending, options=options)
        return output, pending

    def taper(self, wave, **options):
        assert self is state["pending"] and len(self.events) == 1
        assert state["spectral"]() is wave.amplitude
        assert options["mode_weight"] == weight  # The raw candidate is not factorized.
        assert options["z_m"] == 0.
        assert state["reservations"]
        state["taper_calls"] += 1
        if outcome == "no_taper":
            return wave, self
        output, ledger = real_taper(self, wave, **options)
        assert output is not wave and len(ledger.events) == 2
        state.update(taper=weakref.ref(output.amplitude), combined=ledger)
        if outcome == "cancel_after":
            state["cancelled"] = True
        return output, ledger

    def validate(wave, *args):
        if state["spectral"] is not None and state["spectral"]() is wave.amplitude:
            if outcome == "domain":
                raise ValueError("controlled source domain mismatch")
            state["boundary_rejected"] = True
            if outcome == "cancel_before":
                state["cancelled"] = True
            raise carrier.OccupiedCarrierBoundaryError("controlled pending spectral boundary")
        if (outcome == "taper_alias" and state["taper"] is not None
                and state["taper"]() is wave.amplitude):
            raise WaveSamplingError("controlled tapered spectrum exceeds the available grid", 3.)
        return real_validate(wave, *args)

    def check(self, shape, *, working_bytes, retained_bytes=0):
        if (state["boundary_rejected"] and state["taper_calls"] == 0
                and state["spectral"] is not None and state["spectral"]() is not None
                and tuple(shape) == state["spectral"]().shape and working_bytes == 256*math.prod(shape)):
            state["reservations"].append((tuple(shape), working_bytes, retained_bytes))
            assert retained_bytes >= source.amplitude.nbytes
            if outcome == "workspace":
                raise WaveMemoryBudgetError("controlled pending taper workspace failure")
        return real_check(self, shape, working_bytes=working_bytes, retained_bytes=retained_bytes)

    def residual(snapshot, wave, *args, **options):
        state["residual_calls"] += 1
        if state["residual_calls"] == 1:
            raise WaveSamplingError("controlled physical envelope resource pressure", 2.)
        assert state["combined"] is not None
        if state["residual_calls"] == 2:
            assert state["commits"] == 0
        if outcome == "physical_failure":
            raise ValueError("controlled physical operator failure after candidate admission")
        return real_residual(snapshot, wave, *args, **options)

    def proposal(wave, commit):
        def counted_commit():
            state["commits"] += 1
            commit()
        return real_proposal(wave, counted_commit)

    monkeypatch.setattr(NumericalSupportLedger, "resample_spectrum_pressure", spectral)
    monkeypatch.setattr(NumericalSupportLedger, "taper_boundary", taper)
    monkeypatch.setattr(NumericalSupportLedger, "compact_pressure", lambda self, wave, **options: (wave, self))
    monkeypatch.setattr(carrier, "_validate_wave", validate)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(carrier, "WaveResourceProposal", proposal)
    monkeypatch.setattr(WaveGridNumerics, "check_working_memory", check)
    options = dict(energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric,
        support_ledger=initial, support_mode_weight=weight,
        support_projection_policy="weighted_segment_and_retry", cancelled=lambda: state["cancelled"])
    if outcome == "success":
        output, record = carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9), **options)
        accepted = NumericalSupportLedger.from_dict(record["support_ledger"])
        assert accepted == state["combined"] and state["commits"] == 1
        assert [row["kind"] for row in accepted.events] == [
            "numerical_spectral_projection", "numerical_boundary_taper"]
        assert all(row["mode_weight"] == weight for row in accepted.events)
        assert accepted.used_l2 == pytest.approx(sum(row["l2_per_reference"] for row in accepted.events), abs=1e-25)
        assert accepted.used_probability == pytest.approx(
            sum(row["probability_per_reference"] for row in accepted.events), abs=1e-25)
        assert accepted.events[-1]["l2_per_reference"] <= initial.maximum_l2/100
        assert accepted.events[-1]["probability_per_reference"] <= initial.maximum_probability/100
        monkeypatch.setattr(carrier, "_residual", real_residual)
        monkeypatch.setattr(carrier, "_validate_wave", real_validate)
        monkeypatch.setattr(WaveGridNumerics, "check_working_memory", real_check)
        projected, pending = real_spectral(initial, source, **state["options"])
        tapered, expected_ledger = real_taper(pending, projected, z_m=0., mode_weight=weight)
        expected, _ = carrier.propagate_carrier_potential(_snapshot(), tapered, (0., 1e-9),
            energy_ev=ENERGY_EV, kinetic_order=2, grid_numerics=numeric)
        assert accepted == expected_ledger
        np.testing.assert_allclose(output.amplitude, expected.amplitude, rtol=0., atol=2e-14)
        assert output.probability == pytest.approx(
            source.probability-accepted.used_probability/weight, abs=2e-14)
    else:
        expected_error = (InterruptedError if outcome.startswith("cancel_") else
            ValueError if outcome in ("domain", "physical_failure") else WaveGridBudgetError)
        with pytest.raises(expected_error):
            carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-9), **options)
        assert state["commits"] == 0
        if outcome in ("workspace", "cancel_before", "domain"):
            assert state["taper_calls"] == 0
        if outcome in ("workspace", "taper_alias", "no_taper"):
            assert state["spectral"]() is None
            if state["taper"] is not None:
                assert state["taper"]() is None
    assert initial.events == ()
    np.testing.assert_array_equal(source.amplitude, original_amplitude)
