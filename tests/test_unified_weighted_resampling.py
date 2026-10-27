"""Controlled multimode execution checks; not default full-column qualification."""
from dataclasses import replace
import math

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
from temsim.physics.lens_field_provider import CoordinateRegistration, FrozenAnalyticField
from temsim.physics.potential_snapshot import PotentialContribution
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_flux import BeamState
from test_unified_wave_runtime import _request, _resolved_support_checkpoint, _runtime


def _lens_runtime(state):
    native = FrozenAnalyticField("controlled-round", ((.05, 1.00000015, .006),),
        CoordinateRegistration(), (999., 1000.0002), .003)
    potential = HighOrderRoundLensPotential(native, axial_domain="gaussian_all_z")
    row = PotentialContribution(potential.source_id, (potential.source_id,), potential, "magnetic")
    return _runtime(state, 1., 1.0000004, (row,))


def _modes(count):
    source = _resolved_support_checkpoint()
    first = source.beam.modes[0]
    modes = []
    for index in range(count):
        plane = replace(first.plane,
            amplitude=first.plane.amplitude*np.exp(1j*.17*index),
            origin_m=first.plane.origin_m+np.array((index*1e-9, -index*2e-9)),
            tilt_rad=first.plane.tilt_rad+np.array((index*1e-8, -index*3e-8)))
        modes.append(replace(first, plane=plane, mode_id=f"controlled:{index}",
            energy_kev=300.+index*.001,
            weight_per_reference_electron=.8*(index+1)/(count*(count+1))))
    return replace(source, beam=BeamState(tuple(modes), source.beam.reference_plane))


@pytest.mark.parametrize("support,value", [(False, True), (True, 1), (True, "true"), (True, None)])
def test_request_and_executor_reject_invalid_weighted_resampling_before_execution(tmp_path, support, value):
    request = _request(tmp_path, column_support_projection=support, column_weighted_resampling=value)
    with pytest.raises(ValueError, match="explicit boolean|requires controlled support"):
        request.validate()
    state = default_state()
    with pytest.raises(ValueError, match="explicit boolean|requires controlled support"):
        UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state))


def test_weighted_request_binds_executor_identity_and_rejects_same_plane_policy_change(tmp_path):
    state = default_state()
    runtime = _lens_runtime(state)
    regular = _request(tmp_path, column_support_projection=True)
    weighted = replace(regular, column_weighted_resampling=True)
    old = UnifiedPotentialWaveAdapter(state, regular, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, weighted, runtime=runtime)
    assert old.identity != new.identity
    assert old.snapshot.numerical_digest == new.snapshot.numerical_digest
    assert old.support_projection_policy == "step_boundaries_and_retry"
    assert new.support_projection_policy == "weighted_segment_and_retry"
    source = _resolved_support_checkpoint()
    output = new.column(source, 1000.0002)
    assert new.column(output, output.plane_z_mm) is output
    with pytest.raises(ValueError, match="inputs changed"):
        old.column(output, output.plane_z_mm)
    disabled = UnifiedPotentialWaveAdapter(state, replace(regular, column_support_projection=False), runtime=runtime)
    with pytest.raises(ValueError, match="inputs changed"):
        disabled.column(output, output.plane_z_mm)


@pytest.mark.parametrize("count", [2, 9])
def test_actual_multimode_complex_weights_and_histories_survive_lens_boundary_resume(tmp_path, count, record_property):
    state = default_state()
    runtime = _lens_runtime(state)
    source = _modes(count)
    original_digest = source.digest
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True,
                       column_kinetic_order=8, column_exponential_method="chebyshev")
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    direct = adapter.column(source, 1000.0004)
    first = adapter.column(source, 1000.0002)
    store = ExecutedWaveStore(tmp_path/f"weighted-{count}", "controlled-weighted-resume", 2**28)
    key = store.key("lens-exit", first.digest)
    store.put(key, first)
    split = adapter.column(store.get(key), 1000.0004)
    assert source.digest == original_digest
    assert len(split.beam.modes) == count
    assert split.record["upstream_digest"] == original_digest
    assert split.record["unified_support_projection_policy"] == "weighted_segment_and_retry"
    plan = direct.record["unified_segment_plan"]
    assert plan["snapshot_digest"] == runtime.snapshot.numerical_digest
    assert plan["stop_z_m"] == 1000.0004*.001
    assert any(row["kind"] == "nominal_gaussian_exit" and row["z_m"] == 1000.0002*.001
               for row in plan["markers"])
    assert "no field removal" in plan["field_policy"]
    assert "no budget renewal" in plan["history_policy"]
    assert np.linalg.norm(runtime.snapshot.sample([[0., 0., 1000.0004*.001]])["B_t"]) > 0.
    input_weight = math.fsum(mode.weight_per_reference_electron for mode in source.beam.modes)
    assert input_weight == pytest.approx(.4, abs=1e-16)
    relative_errors, phase_errors = [], []
    for initial, resumed, uninterrupted in zip(source.beam.modes, split.beam.modes, direct.beam.modes, strict=True):
        wavelength = float(wavelength_m(initial.energy_kev*1000.))
        # Compare complete complex waves and absolute weights: no fitted phase,
        # recentering, mean removal, intensity normalization or phase flattening.
        from temsim.physics.wave_grid import refine_plane_wave

        # The compressed lattice resolves its envelope; an explicit dense
        # phase-plate readout also needs the analytical carrier's bandwidth.
        # Lossless Fourier embedding preserves that represented wave before
        # the unchanged full-carrier sampling gate is applied.
        readout_shape = tuple(max(160, a, b) for a, b in zip(
            resumed.plane.amplitude.shape, uninterrupted.plane.amplitude.shape))
        a = refine_plane_wave(resumed.plane, readout_shape).full_amplitude(wavelength)*math.sqrt(
            resumed.weight_per_reference_electron)
        b = refine_plane_wave(uninterrupted.plane, readout_shape).full_amplitude(wavelength)*math.sqrt(
            uninterrupted.weight_per_reference_electron)
        assert a.shape == b.shape
        np.testing.assert_allclose(a, b, atol=2e-13, rtol=0.)
        relative_errors.append(float(np.linalg.norm(a-b)/np.linalg.norm(b)))
        occupied = abs(b) > 1e-6*float(abs(b).max())
        phase_errors.append(float(np.max(abs(np.angle(a[occupied]*b[occupied].conj())))))
        assert resumed.weight_per_reference_electron == pytest.approx(
            uninterrupted.weight_per_reference_electron, rel=1e-13, abs=0.)
        assert resumed.weight_per_reference_electron <= initial.weight_per_reference_electron*(1.+1e-13)
        assert resumed.energy_kev == initial.energy_kev
        assert resumed.axial_reference == uninterrupted.axial_reference
        for name in ("origin_m", "basis_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_allclose(getattr(resumed.plane, name), getattr(uninterrupted.plane, name),
                                       atol=1e-14, rtol=1e-13)
        before = first.record["unified_mode_contexts"][initial.mode_id]
        after = split.record["unified_mode_contexts"][initial.mode_id]
        assert before["unified_support_allocation"] == after["unified_support_allocation"]
        allocation = after["unified_support_allocation"]
        assert allocation["input_weight_per_reference"] == initial.weight_per_reference_electron
        assert allocation["maximum_l2_per_reference"] == (
            request.wave_grid.maximum_accumulated_support_l2*math.sqrt(initial.weight_per_reference_electron))
        assert allocation["maximum_probability_per_reference"] == (
            request.wave_grid.maximum_support_loss_fraction*initial.weight_per_reference_electron)
        old_ledger = NumericalSupportLedger.from_dict(before["support_ledger"])
        ledger = NumericalSupportLedger.from_dict(after["support_ledger"])
        assert ledger.events[:len(old_ledger.events)] == old_ledger.events
        assert ledger.maximum_l2 == old_ledger.maximum_l2
        assert ledger.maximum_probability == old_ledger.maximum_probability
        assert ledger.used_l2 >= old_ledger.used_l2
        assert ledger.used_probability >= old_ledger.used_probability
        assert after["potential_support_policy"]["policy"] == "weighted_segment_and_retry"
        assert after["kinetic_phase_ledger"]["events"][:len(before["kinetic_phase_ledger"]["events"])] == (
            before["kinetic_phase_ledger"]["events"])
    summary = split.record["unified_support_summary"]
    assert summary["input_weight_per_reference"] == input_weight
    assert len(summary["allocations"]) == count
    assert summary["normalization_applied"] is False
    assert summary["used_probability_per_reference"] == pytest.approx(
        input_weight-split.beam.total_weight, abs=2e-14, rel=0.)
    record_property("controlled_mode_count", count)
    record_property("maximum_relative_complex_l2_difference", max(relative_errors))
    record_property("maximum_raw_phase_difference_rad_above_1e_minus_12_peak_intensity", max(phase_errors))
    record_property("absolute_input_weight_per_reference", input_weight)
    record_property("absolute_output_weight_per_reference", split.beam.total_weight)
    record_property("numerical_projection_loss_per_reference", summary["used_probability_per_reference"])


def test_same_plane_cannot_drop_weighted_support_policy_receipt(tmp_path):
    state = default_state()
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))
    output = adapter.column(_resolved_support_checkpoint(), 1000.0002)
    record = thaw_json(output.record)
    context = record["unified_mode_contexts"][output.beam.modes[0].mode_id]
    context["potential_support_policy"]["policy"] = "step_boundaries_and_retry"
    with pytest.raises(ValueError, match="support"):
        adapter.column(replace(output, record=record), output.plane_z_mm)


def test_default_request_has_no_implicit_weighted_approximation():
    request = TipWaveRequest().validate()
    assert request.column_weighted_resampling is False
    assert request.column_support_projection is False
    assert request.column_kinetic_order == 4
    assert request.column_exponential_method == "taylor"


def _cache_files(store):
    """Compare actual publication bytes, including payload manifests."""
    import hashlib

    return {str(path.relative_to(store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in store.root.rglob("*") if path.is_file()}


def test_completed_mode_survives_second_mode_failure_and_is_reused_without_reemission(tmp_path, monkeypatch):
    import temsim.physics.unified_wave_runtime as module

    state, source = default_state(), _modes(2)
    source_digest = source.digest
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))
    store = ExecutedWaveStore(tmp_path/"interrupted-mode", "weighted-mode-publication", 2**28)
    execute, calls = module.observe_potential_geometry_checkpoint, []

    def interrupted(snapshot, geometry, entrance, *args, **kwargs):
        mode_id = entrance.beam.modes[0].mode_id
        calls.append(mode_id)
        if mode_id == "controlled:1":
            raise ValueError("controlled second-mode failure")
        return execute(snapshot, geometry, entrance, *args, **kwargs)

    monkeypatch.setattr(module, "observe_potential_geometry_checkpoint", interrupted)
    with pytest.raises(ValueError, match="second-mode failure"):
        adapter.column(source, 1000.0002, store=store)
    assert calls == ["controlled:0", "controlled:1"]
    indexes = tuple(store.root.glob("*.json"))
    assert len(indexes) == 1
    retained_key = indexes[0].stem
    retained = store.get(retained_key)
    retained_record = thaw_json(retained.record)
    retained_amplitude = retained.beam.modes[0].plane.amplitude.copy()
    retained_weight = retained.beam.modes[0].weight_per_reference_electron
    committed = _cache_files(store)
    calls.clear()
    with pytest.raises(ValueError, match="second-mode failure"):
        adapter.column(source, 1000.0002, store=store)
    assert calls == ["controlled:1"]
    assert _cache_files(store) == committed

    def resumed(snapshot, geometry, entrance, *args, **kwargs):
        calls.append(entrance.beam.modes[0].mode_id)
        return execute(snapshot, geometry, entrance, *args, **kwargs)

    def no_new_source(*args, **kwargs):
        raise AssertionError("A resumed column must not re-emit or replace its executed source")

    monkeypatch.setattr("temsim.physics.tip_wave_pipeline.build_tip_gun_checkpoint", no_new_source)
    monkeypatch.setattr(module, "observe_potential_geometry_checkpoint", resumed)
    calls.clear()
    output = adapter.column(source, 1000.0002, store=store)
    assert calls == ["controlled:1"]
    assert source.digest == source_digest
    np.testing.assert_array_equal(output.beam.modes[0].plane.amplitude, retained_amplitude)
    assert output.beam.modes[0].weight_per_reference_electron == retained_weight
    for key in ("support_ledger", "kinetic_phase_ledger", "potential_geometry", "unified_support_allocation"):
        assert thaw_json(output.record["unified_mode_contexts"]["controlled:0"][key]) == retained_record[key]
    for path, digest in committed.items():
        assert _cache_files(store)[path] == digest
    assert len(tuple(store.root.glob("*.json"))) == 3
    warm_cache = _cache_files(store)
    calls.clear()
    repeated = adapter.column(source, 1000.0002, store=store)
    assert calls == []
    assert _cache_files(store) == warm_cache
    assert repeated.beam.total_weight == output.beam.total_weight
    for actual, expected in zip(repeated.beam.modes, output.beam.modes, strict=True):
        np.testing.assert_array_equal(actual.plane.amplitude, expected.plane.amplitude)


@pytest.mark.parametrize("change", ["source", "optics", "policy", "step"])
def test_mode_cache_cannot_reuse_changed_source_optics_or_numerical_execution(tmp_path, monkeypatch, change):
    import temsim.physics.unified_wave_runtime as module

    state, source = default_state(), _modes(2)
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))
    store = ExecutedWaveStore(tmp_path/"bound-mode", "controlled-executor-identity", 2**28)
    adapter.column(source, 1000.0002, store=store)
    committed = _cache_files(store)
    if change == "source":
        first = source.beam.modes[0]
        changed = replace(first, plane=replace(first.plane, amplitude=first.plane.amplitude*np.exp(.1j)))
        source = replace(source, beam=BeamState((changed, *source.beam.modes[1:]), source.beam.reference_plane))
    elif change == "optics":
        state.lenses[0].percent += .1
    elif change == "policy":
        request = replace(request, column_weighted_resampling=False)
    else:
        request = replace(request, column_step_mm=request.column_step_mm/2)
    alternate = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))

    def cache_miss(*args, **kwargs):
        raise ValueError("required fresh mode execution")

    monkeypatch.setattr(module, "observe_potential_geometry_checkpoint", cache_miss)
    with pytest.raises(ValueError, match="fresh mode execution"):
        alternate.column(source, 1000.0002, store=store)
    assert _cache_files(store) == committed


@pytest.mark.parametrize("failure", ["cancelled", "verification"])
def test_cancelled_or_unverified_mode_is_never_published(tmp_path, failure):
    state, source = default_state(), _modes(2)
    source_digest = source.digest
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_lens_runtime(state))
    store = ExecutedWaveStore(tmp_path/"unpublished", "controlled-publication-gate", 2**28)

    def verify():
        if failure == "verification":
            raise ValueError("controlled inputs verification failure")

    with pytest.raises((ValueError, InterruptedError), match="cancelled|verification failure"):
        adapter.column(source, 1000.0002, store=store,
                       cancelled=lambda: failure == "cancelled", verify=verify)
    assert _cache_files(store) == {}
    assert source.digest == source_digest
