"""UI-route admission and executed-source handoff, with explicit fixture scope."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics.physical_scene import capture_physical_scene
from temsim.physics.potential_geometry import PotentialGeometrySnapshot, capture_potential_geometry
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot, PotentialContribution
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.unified_field_runtime import UnifiedFieldRuntime
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter, _static_wave_snapshot
from temsim.physics.wave_execution import WaveExecutionOptions, InelasticWaveNumerics
from temsim.physics.wave_grid import WaveGridNumerics


def _runtime(state, start=.45, stop=.450001, contributions=(), *, physical_geometry=False):
    scene = capture_physical_scene(state)
    snapshot = ElectromagneticPotentialSnapshot(scene, tuple(contributions))
    geometry = (capture_potential_geometry(state, start, stop) if physical_geometry else
        PotentialGeometrySnapshot(start, stop, (), source_scene_geometry_digest=scene.geometry_digest))
    return UnifiedFieldRuntime(snapshot, geometry, capture_instrument_snapshot(state).digest,
        {"scope": "explicit post-gun analytical test domain; not the default complete instrument field"})


def _request(tmp_path, **kwargs):
    return TipWaveRequest(column_step_mm=.0001,
        execution=WaveExecutionOptions(segmented=False, segment_steps=3, cache_directory=str(tmp_path)),
        wave_grid=WaveGridNumerics(maximum_pixels=4096, maximum_working_bytes=4*2**30), **kwargs)


def _resolved_support_checkpoint():
    from test_potential_geometry import _checkpoint
    from test_potential_wave_carrier import _wave
    # This fixture materializes the full carrier for direct phase comparison;
    # use a declared resolved curvature rather than disabling its Nyquist gate.
    return _checkpoint(_wave(n=160, pitch=2.5e-8, width=2e-7,
                             curvature=-1., tilt=(1e-6, -2e-6)))


def test_retained_column_budget_counts_shared_buffers_once_and_excludes_active_mode():
    from test_potential_geometry import _checkpoint
    from temsim.physics.unified_wave_runtime import _column_mode_grid
    from temsim.physics.wave_flux import BeamState
    source = _checkpoint()
    first = source.beam.modes[0]
    shared = replace(first, mode_id="shared", weight_per_reference_electron=.13)
    source = replace(source, beam=BeamState((first, shared), source.beam.reference_plane),
                     auxiliary_arrays={"near_trace": np.arange(91, dtype=float)})
    # The same analytical carrier field can belong to two incoherent modes.
    # It occupies one immutable allocation, not one allocation per weight.
    numerics = WaveGridNumerics(maximum_working_bytes=2**26)
    local, receipt = _column_mode_grid(numerics, source, source, (), 0)
    expected = source.auxiliary_arrays["near_trace"].nbytes
    assert receipt["retained_external_bytes"] == expected
    assert local.maximum_working_bytes == numerics.maximum_working_bytes-expected
    assert numerics.maximum_working_bytes == 2**26
    distinct = replace(first, plane=replace(first.plane), mode_id="completed")
    arrays = (distinct.plane.amplitude, distinct.plane.basis_m, distinct.plane.origin_m,
              distinct.plane.curvature_m1, distinct.plane.tilt_rad)
    expected += sum(value.nbytes for value in arrays if value is not None)
    _, receipt = _column_mode_grid(numerics, source, source, (distinct, distinct), 0)
    assert receipt["retained_external_bytes"] == expected


def test_lazy_column_budget_uses_manifest_without_loading_other_modes_or_auxiliary(tmp_path, monkeypatch):
    from test_potential_geometry import _checkpoint
    from temsim.physics.unified_wave_runtime import _column_mode_grid
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore, _StoredModes, _StoredAuxiliary
    from temsim.physics.wave_flux import BeamState
    source = _checkpoint()
    first = source.beam.modes[0]
    source = replace(source, beam=BeamState((first,
        replace(first, mode_id="second", weight_per_reference_electron=.13)), source.beam.reference_plane),
        auxiliary_arrays={"near_trace": np.arange(91, dtype=float)})
    store = ExecutedWaveStore(tmp_path/"lazy-memory", "retained-column-memory", 2**26)
    stored = store.put(store.key("source"), source)
    expected = sum(item["nbytes"] for item in stored.beam.modes.rows[1]["arrays"].values())
    expected += sum(item["nbytes"] for item in stored.auxiliary_arrays.rows.values())
    def forbidden(*_args):
        raise AssertionError("Memory planning must not load a hidden source payload")
    monkeypatch.setattr(_StoredModes, "__getitem__", forbidden)
    monkeypatch.setattr(_StoredModes, "_array", forbidden)
    monkeypatch.setattr(_StoredAuxiliary, "__getitem__", forbidden)
    numerics = WaveGridNumerics(maximum_working_bytes=2**26)
    local, receipt = _column_mode_grid(numerics, stored, stored, (), 0)
    assert receipt["retained_external_bytes"] == expected
    assert local.maximum_working_bytes == numerics.maximum_working_bytes-expected


def test_retained_source_budget_rejects_before_field_execution(tmp_path, monkeypatch):
    from test_potential_geometry import _checkpoint
    from temsim.physics.wave_grid import WaveMemoryBudgetError
    import temsim.physics.unified_wave_runtime as module
    source = replace(_checkpoint(), auxiliary_arrays={"near_trace": np.zeros(2**17)})
    original = source.digest
    state = default_state()
    request = replace(_request(tmp_path), wave_grid=WaveGridNumerics(maximum_working_bytes=2**20))
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, 1., 1.000004))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("A rejected retained-memory request must not execute the field")
    monkeypatch.setattr(module, "observe_potential_geometry_checkpoint", forbidden)
    with pytest.raises(WaveMemoryBudgetError, match="retained source/current/output"):
        adapter.column(source, 1000.0001)
    assert source.digest == original
    assert adapter.request.wave_grid.maximum_working_bytes == 2**20


def test_multimode_memory_partition_preserves_actual_fields_weights_and_source_auxiliary(tmp_path):
    from test_potential_geometry import _checkpoint
    from temsim.physics.wave_flux import BeamState
    from temsim.physics.potential_wave_device import RESIDUAL_HOST_PREPARATION_RESERVE
    source = _checkpoint()
    first = source.beam.modes[0]
    second = replace(first, plane=first.plane._replace_metadata(origin_m=np.array((1e-9, -2e-9))),
                     mode_id="second", weight_per_reference_electron=.13)
    source = replace(source, beam=BeamState((first, second), source.beam.reference_plane),
                     auxiliary_arrays={"near_trace": np.arange(91, dtype=float)})
    original = source.digest
    auxiliary = source.auxiliary_arrays["near_trace"].copy()
    state = default_state()
    # Preserve the bounded 16 MiB fixture workspace while admitting the
    # carrier's independently reserved cold field-provider preparation.
    budget = RESIDUAL_HOST_PREPARATION_RESERVE+2**24
    request = replace(_request(tmp_path), wave_grid=WaveGridNumerics(maximum_working_bytes=budget))
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, 1., 1.000004))
    single_mode_input = adapter._entrance(source, first)
    assert not single_mode_input.auxiliary_arrays
    output = adapter.column(source, 1000.0004)
    for initial, actual in zip(source.beam.modes, output.beam.modes):
        single = replace(source, beam=BeamState((initial,), source.beam.reference_plane))
        expected = adapter.column(single, 1000.0004).beam.modes[0]
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(actual.plane, name), getattr(expected.plane, name))
        assert actual.weight_per_reference_electron == expected.weight_per_reference_electron
        assert actual.energy_kev == initial.energy_kev
        assert actual.axial_reference == expected.axial_reference
        row = output.record["unified_mode_contexts"][actual.mode_id]["unified_host_memory_reservation"]
        assert row["retained_external_bytes"] > 0
        assert row["active_operator_working_bytes"] + row["retained_external_bytes"] == budget
    assert output.record["upstream_digest"] == original
    assert source.digest == original
    np.testing.assert_array_equal(source.auxiliary_arrays["near_trace"], auxiliary)
    assert not source.auxiliary_arrays["near_trace"].flags.writeable
    assert request.wave_grid.maximum_working_bytes == budget


def test_stored_multimode_column_loads_each_active_mode_once_and_never_copies_gun_auxiliary(tmp_path, monkeypatch):
    from test_potential_geometry import _checkpoint
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore, _StoredModes, _StoredAuxiliary
    from temsim.physics.wave_flux import BeamState
    source = _checkpoint()
    first = source.beam.modes[0]
    source = replace(source, beam=BeamState((first,
        replace(first, mode_id="second", weight_per_reference_electron=.13)), source.beam.reference_plane),
        auxiliary_arrays={"near_trace": np.arange(91, dtype=float)})
    store = ExecutedWaveStore(tmp_path/"streaming-source", "retained-column-memory", 2**26)
    stored = store.put(store.key("source"), source)
    original = stored.digest
    load, indices = _StoredModes.__getitem__, []
    def counted(reader, index):
        indices.append(index)
        return load(reader, index)
    def forbidden(*_args):
        raise AssertionError("Column fields do not consume near-tip auxiliary arrays")
    monkeypatch.setattr(_StoredModes, "__getitem__", counted)
    monkeypatch.setattr(_StoredAuxiliary, "__getitem__", forbidden)
    state = default_state()
    adapter = UnifiedPotentialWaveAdapter(state, _request(tmp_path), runtime=_runtime(state, 1., 1.000004))
    output = adapter.column(stored, 1000.0001)
    assert indices == [0, 1]
    assert output.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-13)
    assert [mode.mode_id for mode in output.beam.modes] == [mode.mode_id for mode in source.beam.modes]
    assert output.record["upstream_digest"] == original
    assert stored.digest == original


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_explicit_column_order_and_fixed_source_support_survive_chunks_and_stops(tmp_path, order):
    import math
    from test_potential_geometry import _checkpoint
    from temsim.physics.wave_flux import BeamState
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.detector.wave_readout import _apply_recording_stop
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    state = default_state()
    source = _resolved_support_checkpoint()
    first_mode = source.beam.modes[0]
    source = replace(source, beam=BeamState((first_mode,
        replace(first_mode, mode_id="controlled:1", weight_per_reference_electron=.13),
        replace(first_mode, mode_id="controlled:zero", weight_per_reference_electron=0.)),
        source.beam.reference_plane))
    request = _request(tmp_path, column_kinetic_order=order, column_support_projection=True)
    runtime = _runtime(state, 1., 1.000004)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    first = adapter.column(source, 1000.001)
    assert adapter.column(first, first.plane_z_mm) is first
    assert first.record["unified_support_projection_policy"] == "step_boundaries_and_retry"
    epsilon = request.wave_grid.maximum_accumulated_support_l2
    delta = request.wave_grid.maximum_support_loss_fraction
    summary = first.record["unified_support_summary"]
    assert summary["input_weight_per_reference"] == .4
    assert summary["maximum_l2_per_reference"] == pytest.approx(epsilon*math.sqrt(.4), rel=2e-16)
    assert summary["maximum_probability_per_reference"] == pytest.approx(delta*.4, rel=2e-16)
    assert summary["normalization_applied"] is False
    for mode in source.beam.modes:
        context = first.record["unified_mode_contexts"][mode.mode_id]
        allocation = context["unified_support_allocation"]
        assert allocation["input_weight_per_reference"] == mode.weight_per_reference_electron
        assert allocation["maximum_l2_per_reference"] == epsilon*math.sqrt(mode.weight_per_reference_electron)
        assert allocation["maximum_probability_per_reference"] == delta*mode.weight_per_reference_electron
        assert allocation["budget_issued"] == (mode.weight_per_reference_electron > 0)
        assert context["potential_operator_policy"]["policy"]["kinetic_order"] == order
        phase = KineticPhaseLedger.from_dict(context["kinetic_phase_ledger"])
        assert phase.binding["kinetic_order"] == order and phase.maximum_phase_rad == .01
    store = ExecutedWaveStore(tmp_path/"support-resume", "fixed-source-support-test", 2**28)
    key = store.key("first", first.digest)
    store.put(key, first)
    split = adapter.column(store.get(key), 1000.002)
    direct = adapter.column(source, 1000.002)
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    for a, b in zip(split.beam.modes, direct.beam.modes):
        wavelength = wavelength_m(a.energy_kev*1000.)
        np.testing.assert_allclose(a.plane.full_amplitude(wavelength), b.plane.full_amplitude(wavelength), atol=2e-13, rtol=0.)
        assert a.weight_per_reference_electron == pytest.approx(b.weight_per_reference_electron, rel=1e-13)
    stopped = _apply_recording_stop(first, state.fluorescent_screen)
    terminal = adapter.column(stopped, 1000.003)
    assert terminal.beam.total_weight == 0.
    for mode in source.beam.modes:
        before = first.record["unified_mode_contexts"][mode.mode_id]
        after = terminal.record["unified_mode_contexts"][mode.mode_id]
        assert after["unified_support_allocation"] == before["unified_support_allocation"]
        assert after["support_ledger"] == before["support_ledger"]
    assert terminal.record["unified_support_summary"]["input_weight_per_reference"] == .4


@pytest.mark.parametrize("change", ["allocation", "ledger", "receipt", "cap", "source", "policy"])
def test_support_history_cannot_be_lost_or_reissued_at_the_same_plane(tmp_path, change):
    from test_potential_geometry import _checkpoint
    state = default_state()
    request = _request(tmp_path, column_kinetic_order=6, column_support_projection=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(_checkpoint(), 1000.001)
    record = thaw_json(first.record)
    context = record["unified_mode_contexts"][first.beam.modes[0].mode_id]
    if change == "allocation":
        del context["unified_support_allocation"]
    elif change == "ledger":
        context["support_ledger"] = None
    elif change == "receipt":
        del context["unified_support_ledger_digest"]
    elif change == "cap":
        context["support_ledger"]["maximum_l2"] *= 2
    elif change == "source":
        from temsim.immutable_json import json_digest
        allocation = context["unified_support_allocation"]
        allocation["source_checkpoint_digest"] = "another-original-source"
        allocation["digest"] = json_digest({key: value for key, value in allocation.items() if key != "digest"})
    else:
        from temsim.immutable_json import json_digest
        context["potential_support_policy"] = {"schema": "potential-support-policy-v1",
            "policy": "step_start_and_retry", "digest": json_digest("step_start_and_retry")}
    with pytest.raises(ValueError, match="support|Support"):
        adapter.column(replace(first, record=record), first.plane_z_mm)


def test_numeric_request_changes_bind_executor_and_same_plane_cache_identity(tmp_path):
    from test_potential_geometry import _checkpoint
    state = default_state()
    runtime = _runtime(state, 1., 1.000004)
    request = _request(tmp_path, column_kinetic_order=6, column_support_projection=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    first = adapter.column(_checkpoint(), 1000.001)
    variants = (replace(request, column_kinetic_order=8),
        replace(request, column_exponential_method="chebyshev"),
        replace(request, column_chebyshev_group=16),
        replace(request, column_support_projection=False),
        replace(request, wave_grid=replace(request.wave_grid,
            maximum_accumulated_support_l2=request.wave_grid.maximum_accumulated_support_l2/2)),
        replace(request, wave_grid=replace(request.wave_grid,
            maximum_support_loss_fraction=request.wave_grid.maximum_support_loss_fraction/2)))
    for variant in variants:
        alternate = UnifiedPotentialWaveAdapter(state, variant, runtime=runtime)
        assert alternate.identity != adapter.identity
        with pytest.raises(ValueError, match="inputs changed"):
            alternate.column(first, first.plane_z_mm)
    original_policy = adapter.support_projection_policy
    assert original_policy == "step_boundaries_and_retry"


@pytest.mark.parametrize("name,value", [("column_kinetic_order", True), ("column_kinetic_order", 6.),
    ("column_kinetic_order", 3), ("column_support_projection", 1),
    ("column_exponential_method", "auto"), ("column_exponential_method", True),
    ("column_chebyshev_group", True), ("column_chebyshev_group", 0),
    ("column_chebyshev_group", 33), ("column_chebyshev_group", 8.)])
def test_column_numeric_controls_require_explicit_types(tmp_path, name, value):
    with pytest.raises(ValueError):
        _request(tmp_path, **{name: value}).validate()


def test_material_descendants_share_one_support_prefix_without_new_allowance(tmp_path):
    from test_potential_geometry import _checkpoint
    from temsim.physics.potential_support_lineage import capture_inherited_support
    from temsim.physics.unified_wave_support import prepare_unified_support, unified_support_summary
    state = default_state()
    request = _request(tmp_path, column_kinetic_order=6, column_support_projection=True)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(_checkpoint(), 1000.001)
    mode = first.beam.modes[0]
    parent = adapter._entrance(first, mode)
    prefix = capture_inherited_support(parent, adapter.snapshot.numerical_digest)
    context = thaw_json(parent.record)
    context.pop("support_ledger")
    context.update(inherited_potential_support=prefix, support_prefix_digest=prefix["digest"])
    contexts = {}
    for index, weight in enumerate((.1, .05)):
        child = replace(mode, mode_id=mode.mode_id+f"/branch:{index}", weight_per_reference_electron=weight)
        contexts[child.mode_id] = prepare_unified_support(context, child, enabled=True,
            l2_limit=request.wave_grid.maximum_accumulated_support_l2,
            probability_limit=request.wave_grid.maximum_support_loss_fraction,
            source_digest=context["unified_support_allocation"]["source_checkpoint_digest"], allow_allocate=False)
        assert contexts[child.mode_id].get("support_ledger") is None
    summary = unified_support_summary(contexts)
    original = first.record["unified_support_summary"]
    assert len(summary["allocations"]) == 1
    assert summary["input_weight_per_reference"] == .27
    for key in ("used_l2_per_reference", "used_probability_per_reference",
                "maximum_l2_per_reference", "maximum_probability_per_reference"):
        assert summary[key] == original[key]
    assert summary["allocations"][0]["further_projection_allowed"] is False


def test_nonconstant_electric_is_retained_even_when_its_amplitude_is_tiny():
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.planar_gun_field import PlanarGunField
    state = default_state()
    electric = ElectricPotentialSnapshot.from_provider(PlanarGunField({},
        np.array([0., .001]), np.array([.4, .5]), np.array([[0., 1e-20], [0., 1e-20]])))
    runtime = _runtime(state, contributions=(PotentialContribution("gun-E", ("gun-E",), electric, "electric"),))
    captured, proofs = _static_wave_snapshot(runtime)
    assert captured.numerical_digest == runtime.snapshot.numerical_digest
    assert not proofs
    point = np.array([0., 0., .45])
    assert captured.sample(point)["E_v_per_m"][2] != 0.


def test_exact_grounded_continuation_retains_its_domain_and_kinetic_reference():
    from test_electric_potential_snapshot import cut_fixture
    from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
    from temsim.physics.grounded_tip_field import GroundedTipField
    field = object.__new__(GroundedTipField)
    field._fem, field._regular = cut_fixture(), None
    field.r, field.z = field._fem.r, field._fem.z
    field.high_tension_v, field.request = 300000., {}
    electric = ElectricPotentialSnapshot.from_provider(field)
    state = default_state()
    runtime = _runtime(state, contributions=(PotentialContribution("gun-E", ("gun-E",), electric, "electric"),))
    snapshot, proofs = _static_wave_snapshot(runtime)
    inside = np.array([[0., 0., .45], [.0009, 0., .450001]])
    np.testing.assert_array_equal(electric.sample(inside)["E_v_per_m"], np.zeros((2, 3)))
    np.testing.assert_array_equal(snapshot.sample(inside)["B_t"], np.zeros((2, 3)))
    assert proofs[0]["constant_potential_rise_v"] == 300000.
    assert snapshot.numerical_digest != runtime.snapshot.numerical_digest
    with pytest.raises(ValueError, match="validity domain"):
        snapshot.sample(np.array([[.0011, 0., .45]]))


def test_backend_policy_keeps_host_transport_and_changed_inputs_fail_before_propagation(tmp_path):
    state = default_state()
    runtime = _runtime(state)
    request = _request(tmp_path)
    for policy in ("gpu", "Require GPU"):
        selected = UnifiedPotentialWaveAdapter(state, replace(request,
            wave_grid=replace(request.wave_grid, compute_backend=policy)), runtime=runtime)
        assert selected.column_grid.compute_backend == "CPU"
        assert selected._options(450., 450.001)["residual_compute_backend"] == selected.requested_backend
    auto = replace(request, wave_grid=replace(request.wave_grid, compute_backend="Auto"))
    selected = UnifiedPotentialWaveAdapter(state, auto, runtime=runtime)
    assert selected.column_grid.compute_backend == "CPU"
    assert selected.request.wave_grid.compute_backend == "Auto"
    state.lenses[0].percent += .1
    with pytest.raises(ValueError, match="inputs or implementation changed"):
        UnifiedPotentialWaveAdapter(state, request, runtime=runtime)


def test_actual_cuda_residual_adapter_preserves_absolute_complex_wave(tmp_path):
    from test_potential_geometry import _checkpoint
    from temsim.physics.compute_backend import cupy_capability
    from temsim.optics.electron_gun.tip_coherence import wavelength_m
    if not cupy_capability().available:
        pytest.skip("A CUDA device is required for the actual residual parity")
    state = default_state()
    runtime = _runtime(state, 1., 1.000004)
    request = _request(tmp_path, column_kinetic_order=8, column_support_projection=True)
    cpu = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    gpu = UnifiedPotentialWaveAdapter(state, replace(request,
        wave_grid=replace(request.wave_grid, compute_backend="Require GPU")), runtime=runtime)
    source = _resolved_support_checkpoint()
    a, b = cpu.column(source, 1000.001), gpu.column(source, 1000.001)
    first, second = a.beam.modes[0], b.beam.modes[0]
    wavelength = wavelength_m(first.energy_kev*1000.)
    np.testing.assert_allclose(first.plane.full_amplitude(wavelength), second.plane.full_amplitude(wavelength), atol=2e-13, rtol=0.)
    assert first.weight_per_reference_electron == pytest.approx(second.weight_per_reference_electron, rel=1e-13)
    assert b.record["unified_backend"]["column_carrier"] == "CPU"
    assert b.record["unified_backend"]["residual_requested"] == "Require GPU"
    assert b.record["unified_backend"]["residual_actual"] == ("cupy",)
    assert b.record["potential_operator_policy"] == a.record["potential_operator_policy"]


def test_continuation_retains_phase_budget_and_recording_stop_history(tmp_path):
    from test_potential_geometry import _checkpoint
    from temsim.detector.wave_readout import _apply_recording_stop
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state = default_state()
    source = _checkpoint()
    runtime = _runtime(state, 1., 1.000004)
    adapter = UnifiedPotentialWaveAdapter(state, _request(tmp_path), runtime=runtime)
    first = adapter.column(source, 1000.001)
    assert adapter.column(first, first.plane_z_mm) is first
    with pytest.raises(ValueError, match="inputs changed"):
        adapter.column(replace(first, record={**thaw_json(first.record),
            "unified_wave_identity": "other"}), first.plane_z_mm)
    with pytest.raises(ValueError, match="fixed captured column"):
        adapter.column(replace(first, plane_z_mm=2000.), 2000.)
    store = ExecutedWaveStore(tmp_path/"resume", "immutable-unified-wave-test", 2**28)
    key = store.key("prefix", first.digest)
    store.put(key, first)
    restored = store.get(key)
    split = adapter.column(restored, 1000.002)
    direct = adapter.column(source, 1000.002)
    a, b = split.beam.modes[0], direct.beam.modes[0]
    np.testing.assert_allclose(a.plane.amplitude, b.plane.amplitude, rtol=0., atol=2e-13)
    np.testing.assert_allclose(a.plane.curvature_m1, b.plane.curvature_m1, rtol=0., atol=1e-12)
    assert a.weight_per_reference_electron == pytest.approx(b.weight_per_reference_electron, rel=1e-13)
    ledger = KineticPhaseLedger.from_dict(split.record["unified_mode_contexts"][a.mode_id]["kinetic_phase_ledger"])
    assert ledger.maximum_phase_rad == .01 and ledger.used_phase_rad > 0
    context = thaw_json(first.record)
    del context["unified_mode_contexts"][a.mode_id]["kinetic_phase_ledger"]
    with pytest.raises(ValueError, match="lost its cumulative"):
        adapter.column(replace(first, record=context), 1000.002)
    # The actual inserted screen intercepts the complete narrow fixture. A
    # downstream zero-flux continuation keeps its physical loss and old budget.
    screen = state.fluorescent_screen
    intercepted = _apply_recording_stop(first, screen)
    terminal = adapter.column(intercepted, 1000.003)
    assert terminal.beam.total_weight == 0.
    assert terminal.record["unified_recording_stops"][0]["detector"] == screen.key
    final_context = terminal.record["unified_mode_contexts"][a.mode_id]
    phase = KineticPhaseLedger.from_dict(final_context["kinetic_phase_ledger"])
    old = KineticPhaseLedger.from_dict(first.record["unified_mode_contexts"][a.mode_id]["kinetic_phase_ledger"])
    assert phase.used_phase_rad == old.used_phase_rad
    assert phase.events[:len(old.events)] == old.events
    assert final_context["external_recording_losses"][0]["received_weight"] == first.beam.total_weight
    auto = UnifiedPotentialWaveAdapter(state, replace(adapter.request,
        wave_grid=replace(adapter.request.wave_grid, compute_backend="Auto")), runtime=runtime)
    automatic = auto.column(source, 1000.001)
    # Auto may select the admitted CUDA residual. Its FFT roundoff need not be
    # bitwise identical to NumPy; retain the same full-complex numeric budget.
    np.testing.assert_allclose(automatic.beam.modes[0].plane.amplitude,
                               first.beam.modes[0].plane.amplitude, atol=2e-13, rtol=0.)
    assert automatic.record["unified_backend"]["requested"] == "Auto"
    assert automatic.record["unified_backend"]["column_carrier"] == "CPU"


def test_streamed_inserted_detector_preserves_source_and_budgets_during_continuation(tmp_path):
    from types import SimpleNamespace
    from temsim.detector.wave_readout import _apply_recording_stop, _recording_stop_streamed
    from temsim.physics.wave_flux import BeamState
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    state = default_state()
    source = _resolved_support_checkpoint()
    mode = source.beam.modes[0]
    shifted = replace(mode, mode_id="controlled:transmitted", weight_per_reference_electron=.13,
        plane=mode.plane._replace_metadata(origin_m=np.array((8e-6, 0.))))
    source = replace(source, beam=BeamState((mode, shifted), source.beam.reference_plane))
    adapter = UnifiedPotentialWaveAdapter(state,
        _request(tmp_path, column_kinetic_order=6, column_support_projection=True),
        runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(source, 1000.0001)
    # This controlled disk intercepts one separated coherent mode completely
    # and leaves the other intact: real total-beam clipping without an
    # unresolved sharp edge through either mode's occupied support.
    detector = SimpleNamespace(key="inserted_disk", geometry="disk", outer_width_mm=.006,
        inner_diameter_mm=0., readout_enabled=False)
    expected_stop = _apply_recording_stop(first, detector)
    store = ExecutedWaveStore(tmp_path/"streamed-stop", "streamed-stop-lineage", 2**28)
    stopped = _recording_stop_streamed(first, detector, store,
        verify=lambda: None, cancelled=lambda: False)
    assert stopped.record["schema"] == "executed-recording-stop-stream-v1"
    assert hasattr(stopped.beam.modes, "rows")
    assert stopped.beam.total_weight == pytest.approx(first.beam.modes[1].weight_per_reference_electron)
    assert stopped.beam.total_weight < first.beam.total_weight
    assert adapter.column(stopped, stopped.plane_z_mm) is stopped
    actual = adapter.column(stopped, 1000.0002)
    expected = adapter.column(expected_stop, 1000.0002)
    assert actual.record["unified_source_checkpoint_digest"] == source.digest
    assert actual.record["upstream_digest"] == source.digest
    assert actual.record["upstream"] == source.record
    assert [row["detector"] for row in actual.record["unified_recording_stops"]] == [detector.key]
    for current, reference in zip(actual.beam.modes, expected.beam.modes):
        for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
            np.testing.assert_array_equal(getattr(current.plane, name), getattr(reference.plane, name))
        assert current.weight_per_reference_electron == reference.weight_per_reference_electron
        before = first.record["unified_mode_contexts"][current.mode_id]
        after = actual.record["unified_mode_contexts"][current.mode_id]
        assert after["unified_support_allocation"] == before["unified_support_allocation"]
        old_support = NumericalSupportLedger.from_dict(before["support_ledger"])
        new_support = NumericalSupportLedger.from_dict(after["support_ledger"])
        assert old_support.events and old_support.used_probability > 0
        assert new_support.events[:len(old_support.events)] == old_support.events
        assert new_support.maximum_probability == old_support.maximum_probability
        assert new_support.maximum_l2 == old_support.maximum_l2
        old_phase = KineticPhaseLedger.from_dict(before["kinetic_phase_ledger"])
        new_phase = KineticPhaseLedger.from_dict(after["kinetic_phase_ledger"])
        assert new_phase.events[:len(old_phase.events)] == old_phase.events
        assert new_phase.maximum_phase_rad == old_phase.maximum_phase_rad
        assert new_phase.used_phase_rad >= old_phase.used_phase_rad > 0
        losses = after["external_recording_losses"]
        expected_loss = next(row for row in expected_stop.record["modes"] if row["mode_id"] == current.mode_id)
        assert len(losses) == 1 and losses[0] == expected_loss


@pytest.mark.parametrize("missing", ("kinetic_phase_ledger", "support_ledger", "unified_support_allocation"))
def test_streamed_stop_cannot_hide_missing_cumulative_history(tmp_path, missing):
    from temsim.detector.wave_readout import _recording_stop_streamed
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state = default_state()
    adapter = UnifiedPotentialWaveAdapter(state,
        _request(tmp_path, column_kinetic_order=6, column_support_projection=True),
        runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(_resolved_support_checkpoint(), 1000.0001)
    record = thaw_json(first.record)
    del record["unified_mode_contexts"][first.beam.modes[0].mode_id][missing]
    broken = replace(first, record=record)
    store = ExecutedWaveStore(tmp_path/"missing-history", "streamed-history-admission", 2**28)
    stopped = _recording_stop_streamed(broken, state.fluorescent_screen, store,
        verify=lambda: None, cancelled=lambda: False)
    with pytest.raises(ValueError, match="lost|missing|allocation|ledger"):
        adapter.column(stopped, stopped.plane_z_mm)
    with pytest.raises(ValueError, match="lost|missing|allocation|ledger"):
        adapter.column(stopped, 1000.0002)


def test_mixed_streamed_and_dense_stops_keep_chronological_loss_history(tmp_path):
    from types import SimpleNamespace
    from temsim.detector.wave_readout import _apply_recording_stop, _recording_stop_streamed
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    state = default_state()
    adapter = UnifiedPotentialWaveAdapter(state,
        _request(tmp_path, column_kinetic_order=6, column_support_projection=True),
        runtime=_runtime(state, 1., 1.000004))
    first = adapter.column(_resolved_support_checkpoint(), 1000.0001)
    store = ExecutedWaveStore(tmp_path/"mixed-stops", "mixed-stop-lineage", 2**28)
    # First detector has no overlap; the next detector absorbs this complete
    # fixture. Chronology is tested independently of an unresolved edge.
    first_detector = SimpleNamespace(key="outer_annulus", geometry="annulus", outer_width_mm=2.,
        inner_diameter_mm=1., readout_enabled=False)
    streamed = _recording_stop_streamed(first, first_detector, store,
        verify=lambda: None, cancelled=lambda: False)
    dense = _apply_recording_stop(streamed, state.fluorescent_screen)
    result = adapter.column(dense, 1000.0002)
    assert result.beam.total_weight == 0.
    assert [row["detector"] for row in result.record["unified_recording_stops"]] == [
        first_detector.key, state.fluorescent_screen.key]
    context = result.record["unified_mode_contexts"][first.beam.modes[0].mode_id]
    losses = context["external_recording_losses"]
    assert len(losses) == 2
    assert losses[0] == streamed.record["modes"][0]
    assert losses[1] == dense.record["modes"][0]
    assert losses[0]["received_weight"] == pytest.approx(0., abs=1e-15)
    assert losses[1]["received_weight"] == streamed.beam.total_weight
    old = first.record["unified_mode_contexts"][first.beam.modes[0].mode_id]
    assert context["support_ledger"] == old["support_ledger"]
    assert context["unified_support_allocation"] == old["unified_support_allocation"]


def test_initial_gun_provenance_is_retained_once_without_losing_mode_history(tmp_path):
    from test_potential_geometry import _checkpoint
    from temsim.physics.unified_wave_runtime import _source_provenance_only
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.potential_phase_budget import KineticPhaseLedger
    state = default_state()
    source = replace(_checkpoint(), record={"schema": "executed-tip-gun-v1",
        "source_scope": "Schema-shaped provenance fixture; not a physical gun calculation",
        "producer_evidence": [dict(index=i, value=float(i)) for i in range(1000)]})
    runtime = _runtime(state, 1., 1.000004)
    adapter = UnifiedPotentialWaveAdapter(state, _request(tmp_path), runtime=runtime)
    first = adapter.column(source, 1000.001)
    mode_id = source.beam.modes[0].mode_id
    context = first.record["unified_mode_contexts"][mode_id]
    assert first.record["upstream"] == source.record
    assert context["source_checkpoint_digest"] == source.digest
    assert context["source_mode_id"] == mode_id
    assert "producer_evidence" not in context
    assert context["potential_geometry"]["last_z_m"] == 1.000001
    for key in ("kinetic_phase_ledger", "electric_wave_history", "potential_geometry", "support_ledger"):
        assert not _source_provenance_only({**thaw_json(source.record), key: None})
    store = ExecutedWaveStore(tmp_path/"provenance", "source-context-roundtrip", 2**28)
    key = store.key(first.digest); store.put(key, first)
    resumed = adapter.column(store.get(key), 1000.002)
    direct = adapter.column(source, 1000.002)
    np.testing.assert_allclose(resumed.beam.modes[0].plane.amplitude,
                               direct.beam.modes[0].plane.amplitude, rtol=0., atol=2e-13)
    assert resumed.record["upstream"] == source.record
    continuation = resumed.record["unified_mode_contexts"][mode_id]
    assert continuation["source_checkpoint_digest"] == source.digest
    before = KineticPhaseLedger.from_dict(context["kinetic_phase_ledger"])
    after = KineticPhaseLedger.from_dict(continuation["kinetic_phase_ledger"])
    assert after.used_phase_rad >= before.used_phase_rad
    assert after.events[:len(before.events)] == before.events


def test_actual_gun_checkpoint_reaches_observable_without_substituting_source(tmp_path):
    """Real tip/gun producer followed by a declared vacuum continuation fixture.

    The post-gun vacuum is an independent propagation fixture, explicitly not
    the default instrument's nonconstant electric tail or a whole-column claim.
    No exit amplitude, fitted pupil, or synthetic source replaces the producer.
    """
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics
    from temsim.physics.tip_gun_wave import build_tip_gun_checkpoint, GunWaveNumerics
    state = default_state()
    gun = state.electron_gun
    gun.emitter.coherence = TipCoherence()
    gun.emitter.virtual_source_fwhm_nm = 50.
    gun.emitter.energy_spread_fwhm_ev = 0.
    source = build_tip_gun_checkpoint(gun, source_numerics=TipWaveNumerics(grid_pixels=128, energy_samples=1),
        numerics=GunWaveNumerics(field_step_mm=.2, bore_step_mm=10.), use_cache=False)
    runtime = _runtime(state, physical_geometry=True)
    adapter = UnifiedPotentialWaveAdapter(state, _request(tmp_path), runtime=runtime)
    seen = []
    output = adapter.column(source, 450.001, progress_callback=lambda *row: seen.append(row))
    assert output.plane_z_mm == 450.001
    assert output.record["upstream"]["schema"] == "executed-tip-gun-v1"
    assert output.record["unified_source_checkpoint_digest"] == source.digest
    assert output.record["unified_runtime_digest"] == runtime.digest
    assert output.beam.total_weight == pytest.approx(source.beam.total_weight, rel=2e-13)
    assert output.beam.modes[0].mode_id == source.beam.modes[0].mode_id
    assert output.beam.modes[0].axial_reference.flight_time_s > source.beam.modes[0].axial_reference.flight_time_s
    assert np.isfinite(output.beam.modes[0].plane.amplitude).all()
    assert seen[-1][0] == pytest.approx(seen[-1][1])
    assert not output.record["empirical_aberrations_applied"]


def test_actual_gun_wave_responds_to_captured_c1_excitation_and_uses_its_exact_gauge(tmp_path, record_property):
    """Real source + selected real C1; excludes other fields explicitly.

    This bounded static-magnetic response is not default full-column admission.
    The complete phase/absolute flux is compared without fitting a piston.
    """
    from temsim.optics.electron_gun.tip_coherence import TipCoherence, TipWaveNumerics, wavelength_m
    from temsim.physics.tip_gun_wave import build_tip_gun_checkpoint, GunWaveNumerics
    from temsim.physics.potential_capture import capture_round_lens_potential
    from temsim.physics.high_order_round_lens import HighOrderRoundLensPotential
    from temsim.physics.unified_wave_runtime import unified_checkpoint_magnetic_gauge
    state = default_state()
    gun = state.electron_gun
    gun.emitter.coherence = TipCoherence()
    gun.emitter.virtual_source_fwhm_nm = 50.
    gun.emitter.energy_spread_fwhm_ev = 0.
    source = build_tip_gun_checkpoint(gun, source_numerics=TipWaveNumerics(grid_pixels=128, energy_samples=1),
        numerics=GunWaveNumerics(field_step_mm=.2, bore_step_mm=10.), use_cache=False)
    lens = next(item for item in state.lenses if item.key == "condenser_lens_1")
    outputs, identities, samplers = [], [], []
    for percent, step in ((1., 1.), (2., 1.), (2., .5)):
        lens.percent = percent
        native = capture_round_lens_potential(state, lens.key)
        potential = HighOrderRoundLensPotential(native.potential.field, radial_order=3, axial_domain="gaussian_all_z")
        row = PotentialContribution(native.source_id, native.owner_ids, potential, "magnetic")
        runtime = _runtime(state, .45, .505, (row,))
        request = replace(_request(tmp_path), column_step_mm=step)
        adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
        output = adapter.column(source, 505.)
        outputs.append(output)
        identities.append(runtime.digest)
        samplers.append(adapter.snapshot)
    assert identities[0] != identities[1]
    assert outputs[0].record["unified_field_digest"] != outputs[1].record["unified_field_digest"]
    first, second, refined = [output.beam.modes[0] for output in outputs]
    assert first.weight_per_reference_electron == pytest.approx(second.weight_per_reference_electron, rel=2e-12)
    # Field changes alter the carrier and/or sampled complex envelope. Retain
    # both factors instead of comparing only a cosmetically flattened phase.
    assert np.linalg.norm(first.plane.curvature_m1-second.plane.curvature_m1) > 1e-6
    from scipy.ndimage import map_coordinates
    sample_xy = np.array([[1e-6, -1e-6, 0.], [0., 0., 1e-6]])
    def physical_amplitude(mode):
        wave = mode.plane
        delta = sample_xy-wave.origin_m[:, None]
        coordinate = np.linalg.solve(wave.basis_m, delta)
        coordinate[0] += wave.amplitude.shape[1]//2
        coordinate[1] += wave.amplitude.shape[0]//2
        envelope = (map_coordinates(wave.amplitude.real, coordinate[::-1], order=3)
                    +1j*map_coordinates(wave.amplitude.imag, coordinate[::-1], order=3))
        carrier = (.5*np.einsum("in,ij,jn->n", delta, wave.curvature_m1, delta)+wave.tilt_rad@delta)
        return envelope*np.exp(2j*np.pi*carrier/float(wavelength_m(mode.energy_kev*1000)))\
            *np.sqrt(mode.weight_per_reference_electron/abs(np.linalg.det(wave.basis_m)))
    raw = [physical_amplitude(mode) for mode in (first, second, refined)]
    signal = np.max(abs(np.angle(raw[1]*raw[0].conj())))
    refinement = np.max(abs(np.angle(raw[2]*raw[1].conj())))
    assert signal > 1e-7 and signal > 5*refinement
    record_property("raw_phase_response_rad", float(signal))
    record_property("half_step_phase_difference_rad", float(refinement))
    record_property("curvature_response_m_inverse", float(np.linalg.norm(first.plane.curvature_m1-second.plane.curvature_m1)))
    gauge = unified_checkpoint_magnetic_gauge(outputs[-1])
    xy = np.array([[1e-4, -2e-4], [2e-4, 1e-4]])
    expected = samplers[-1].sample(np.c_[xy.T, np.full(2, .505)])["A_t_m"][:, :2].T
    np.testing.assert_array_equal(gauge.vector_potential_xy_t_m(xy), expected)
    assert not np.array_equal(expected, .5*samplers[-1].sample([[0., 0., .505]])["B_t"][0, 2]*np.stack((-xy[1], xy[0])))
    assert gauge.fingerprint
    with pytest.raises(ValueError, match="exact executed"):
        gauge.require_plane(np.nextafter(505., np.inf))


@pytest.mark.parametrize("segmented", [False, True])
def test_unified_preflight_cannot_be_bypassed_by_a_gun_cache(monkeypatch, tmp_path, segmented):
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.unified_wave_runtime as adapter_module
    state = default_state()
    state.simulation_mode = "unified"
    for item in (*state.deflectors, *state.stigmators, *state.corrector_elements):
        if hasattr(item, "scan_enabled"):
            item.scan_enabled = False
        if hasattr(item, "wobble_enabled"):
            item.wobble_enabled = False
    def fail_admission(*args, **kwargs):
        raise ValueError("explicit unsupported accelerating electric field")
    monkeypatch.setattr(adapter_module, "UnifiedPotentialWaveAdapter", fail_admission)
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", lambda *a, **k: pytest.fail("Source must not execute"))
    monkeypatch.setattr(pipeline.ExecutedWaveStore, "get", lambda *a: pytest.fail("A cache cannot bypass admission"))
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system",
        lambda *a, **k: pytest.fail("Unified admission must not run legacy scan calibration"))
    request = replace(_request(tmp_path, stop="plane", observation_z_mm=451.),
                      execution=replace(_request(tmp_path).execution, segmented=segmented))
    request = replace(request, wave_grid=replace(request.wave_grid, compute_backend="Require GPU"))
    # Exercise physical admission after the independently tested CUDA policy.
    # The substituted adapter rejects before any source or propagation runs.
    from temsim.physics import compute_backend
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(True, "Substituted unified admission"))
    with pytest.raises(ValueError, match="unsupported accelerating"):
        pipeline.simulate_tip_wave(state, request)


@pytest.mark.parametrize("support", ["disabled", "enabled", "zero-budget", "enabled-chebyshev"])
def test_material_modes_keep_separate_geometry_histories_and_actual_energy(tmp_path, support):
    from test_potential_specimen_transport import _fixture
    from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
    from temsim.physics.wave_flux import BeamState
    state, source = _fixture()
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_sigma_angstrom = .02
    first = source.beam.modes[0]
    source = replace(source, beam=BeamState((replace(first, mode_id="first", weight_per_reference_electron=.27),
        replace(first, mode_id="second", energy_kev=299.9, weight_per_reference_electron=.13)), source.beam.reference_plane))
    source = replace(source, plane_z_mm=source.plane_z_mm-.000001)
    lower, upper = source.plane_z_mm*.001, (state.sample.z_mm+.00001)*.001
    runtime = _runtime(state, lower, upper)
    # This material's occupied angular spectrum needs the explicitly selected
    # higher-order branch after projection. The unchanged/no-projection cases
    # retain their original order and the same .01 rad cumulative budget.
    selected_order = 8 if support.startswith("enabled") else 4
    selected_method = "chebyshev" if support == "enabled-chebyshev" else "taylor"
    request = _request(tmp_path, column_kinetic_order=selected_order, column_support_projection=support != "disabled",
        column_exponential_method=selected_method,
        inelastic=InelasticWaveNumerics(method="zero_loss"))
    if support == "zero-budget":
        request = replace(request, wave_grid=replace(request.wave_grid, maximum_accumulated_support_l2=0.))
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    source = adapter.column(source, state.sample.z_mm-state.sample.thickness_nm*.5e-6)
    store = ExecutedWaveStore(tmp_path/"material", "unified-material-fixture", 2**28)
    material = adapter.specimen(source, store=store, use_cache=False)
    assert len(material.beam.modes) == 4
    assert adapter.column_grid.angular_domain_padding == "minimum_full_band"
    assert material.record["unified_canonical_domain_policies"] == {
        "column": "minimum_full_band", "material_field_substeps": "doubling"}
    for history in material.record["material_histories"]:
        assert history["material_domain_policy"]["angular_domain_padding"] == "doubling"
    assert len(material.record["unified_mode_contexts"]) == 4
    output = adapter.column(material, state.sample.z_mm+.000001)
    assert output.beam.total_weight <= .4
    assert {mode.energy_kev for mode in output.beam.modes} == {300., 299.9}
    assert len(output.record["unified_mode_contexts"]) == 4
    for context in output.record["unified_mode_contexts"].values():
        assert context["potential_geometry"]["geometry_digest"] == runtime.geometry.digest
        assert context["potential_operator_policy"]["policy"]["kinetic_order"] == selected_order
        assert context["potential_operator_policy"]["policy"]["residual_exponential_method"] == selected_method
        if support != "disabled":
            assert context.get("support_ledger") is None
            assert context["unified_support_allocation"]["input_weight_per_reference"] in (.27, .13)
    if support != "disabled":
        summary = output.record["unified_support_summary"]
        assert len(summary["allocations"]) == 2
        assert summary["input_weight_per_reference"] == .4
        assert summary["used_l2_per_reference"] == source.record["unified_support_summary"]["used_l2_per_reference"]
        assert summary["used_probability_per_reference"] == source.record["unified_support_summary"]["used_probability_per_reference"]


def test_auto_material_transport_selects_only_the_field_substep_on_cpu():
    from test_potential_geometry import _checkpoint
    from temsim.physics.potential_specimen_transport import PotentialMaterialTransport
    state = default_state()
    source = _checkpoint()
    runtime = _runtime(state, 1., 1.000004)
    transport = PotentialMaterialTransport(runtime.snapshot, (1000., 1000.004), "auto-material-field-stage",
                                          geometry=runtime.geometry)
    numeric = WaveGridNumerics(compute_backend="Auto", angular_domain_padding="minimum_full_band", maximum_working_bytes=2**30)
    output = transport(state, source, 1000.00001, maximum_step_mm=.001,
                       grid_numerics=numeric, cancelled=lambda: False)
    assert numeric.compute_backend == "Auto"
    assert numeric.angular_domain_padding == "minimum_full_band"
    assert output.record["material_domain_policy"] == transport.domain_policy()
    assert transport.context(output.record)["material_domain_policy"] == transport.domain_policy()
    for invalid in (None, {**transport.domain_policy(), "angular_domain_padding": "minimum_full_band"}):
        with pytest.raises(ValueError, match="domain policy"):
            transport.context({**dict(output.record), "material_domain_policy": invalid})
    legacy = {key: value for key, value in output.record.items() if key != "material_domain_policy"}
    with pytest.raises(ValueError, match="restart from the material entrance"):
        transport.context(legacy)
    with pytest.raises(ValueError, match="restart from the material entrance"):
        transport(state, replace(output, record=legacy), 1000.00002, maximum_step_mm=.001,
                  grid_numerics=numeric, cancelled=lambda: False)
    resumed = transport(state, output, 1000.00002, maximum_step_mm=.001,
                        grid_numerics=numeric, cancelled=lambda: False)
    assert resumed.record["material_domain_policy"] == output.record["material_domain_policy"]
    context = transport.context(output.record)
    assert context["material_execution_identity"] == transport.digest
    for key in ("material_domain_policy", "material_execution_identity"):
        stripped = {name: value for name, value in context.items() if name != key}
        with pytest.raises(ValueError, match="restart from the material entrance"):
            transport.context(stripped)
        with pytest.raises(ValueError, match="restart from the material entrance"):
            transport(state, replace(output, record=stripped), 1000.00002,
                maximum_step_mm=.001, grid_numerics=numeric, cancelled=lambda: False)
    from temsim.physics.unified_wave_runtime import _source_provenance_only
    for key in ("material_domain_policy", "material_execution_identity", "material_histories", "column_transport_identity"):
        assert not _source_provenance_only({"schema": "executed-tip-gun-v1", key: None})
    assert output.record["backend_policy"] == {"requested": "Auto", "field_carrier": "CPU", "material_policy": "Auto"}
    assert output.beam.total_weight == pytest.approx(source.beam.total_weight, rel=1e-13)



def test_material_domain_policy_does_not_regenerate_the_roi_or_phonon_realization():
    from test_potential_specimen_transport import _fixture
    from temsim.physics.specimen_wave_transport import _prepare_material_grid
    state, source = _fixture()
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_sigma_angstrom = .02
    double = WaveGridNumerics(angular_domain_padding="doubling")
    minimum = replace(double, angular_domain_padding="minimum_full_band")
    old = _prepare_material_grid(state, source, grid_numerics=double)
    new = _prepare_material_grid(state, source, grid_numerics=minimum)
    for a, b in zip(old[2:], new[2:], strict=True):
        np.testing.assert_array_equal(a, b)
    transient = {"prepared_specimen_cache_hit", "preparation_seconds", "preparation_build_seconds"}
    assert {k: v for k, v in old[1].metrics.items() if k not in transient} == {
        k: v for k, v in new[1].metrics.items() if k not in transient}
    for a, b in zip(old[1].potential_configurations_v_angstrom,
                    new[1].potential_configurations_v_angstrom, strict=True):
        np.testing.assert_array_equal(a, b)
    assert state.sample.wave_grid_pixels == 256
    assert state.sample.wave_field_of_view_angstrom == 30.


@pytest.mark.parametrize("defect", ["missing", "none", "tampered", "missing_both", "legal"])
def test_material_domain_receipt_is_required_by_downstream_and_same_plane(tmp_path, defect):
    from test_potential_geometry import _checkpoint
    from temsim.physics.potential_specimen_transport import PotentialMaterialTransport
    from temsim.immutable_json import thaw_json
    state = default_state()
    runtime = _runtime(state, 1., 1.000004)
    request = _request(tmp_path)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    transport = PotentialMaterialTransport(runtime.snapshot, (1000., 1000.004),
        "actual material callback downstream gate", adapter.operator_options, runtime.geometry)
    output = transport(state, _checkpoint(), 1000.00001,
        maximum_step_mm=request.column_step_mm, grid_numerics=request.wave_grid,
        cancelled=lambda: False)
    context = transport.context(output.record)
    if defect == "missing":
        del context["material_domain_policy"]
    elif defect == "none":
        context["material_domain_policy"] = None
    elif defect == "tampered":
        context["material_domain_policy"]["digest"] = "not-the-executed-policy"
    elif defect == "missing_both":
        del context["material_domain_policy"]
        del context["material_execution_identity"]
    mode_id = output.beam.modes[0].mode_id
    # Ordinary per-history context and the aggregate material result both
    # route through the same strict gate, including zero-distance reads.
    records = [{"schema": "unified-actual-source-material-v1", "material_histories": [thaw_json(output.record)],
                "unified_mode_contexts": {mode_id: context}}]
    if defect != "missing_both":
        records.append(context)
    for record in records:
        candidate = replace(output, record=record)
        for stop in (output.plane_z_mm, 1000.00002):
            if defect == "legal":
                result = adapter.column(candidate, stop)
                assert result.beam.total_weight == pytest.approx(output.beam.total_weight, rel=2e-13)
                if stop != output.plane_z_mm:
                    assert result.record["unified_mode_contexts"][mode_id]["material_execution_identity"] == transport.digest
            else:
                with pytest.raises(ValueError, match="domain policy|material entrance"):
                    adapter.column(candidate, stop)
