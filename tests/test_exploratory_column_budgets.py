"""Column-only exploratory allowances preserve source and historical budgets."""
from dataclasses import asdict, replace
import math
from types import SimpleNamespace

import pytest

from temsim.gui.coherent_result_labels import coherent_preview_text
from temsim.immutable_json import json_digest, thaw_json
from temsim.optics.column import default_state
from temsim.optics.electron_gun.tip_coherence import wavelength_m
from temsim.physics.potential_wave_support import NumericalSupportLedger
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from temsim.physics.wave_grid import WaveGridNumerics
from test_unified_wave_runtime import _request, _runtime
from test_unified_weighted_resampling import _modes


def _state():
    state = default_state()
    state.simulation_mode = "unified"
    return state


def _preview(tmp_path, **changes):
    request = _request(tmp_path, column_support_projection=True,
        column_weighted_resampling=True, column_exploratory_preview=True,
        column_preview_pixels=256, column_kinetic_order=8,
        column_preview_support_probability=.01, column_preview_support_l2=.5)
    return replace(request, **changes)


@pytest.mark.parametrize("probability,l2", [(.01, .5), (.01, None), (None, .5)])
def test_explicit_overrides_change_only_effective_post_gun_numerics(tmp_path, probability, l2):
    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    request = _preview(tmp_path, column_preview_support_probability=probability,
                       column_preview_support_l2=l2)
    before = asdict(request)
    old = replace(request, column_preview_support_probability=None, column_preview_support_l2=None)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    legacy = UnifiedPotentialWaveAdapter(state, old, runtime=runtime)
    expected_p = request.wave_grid.maximum_support_loss_fraction if probability is None else probability
    expected_l2 = request.wave_grid.maximum_accumulated_support_l2 if l2 is None else l2
    for grid in (adapter.column_grid, adapter.material_grid):
        assert grid.maximum_support_loss_fraction == expected_p
        assert grid.maximum_accumulated_support_l2 == expected_l2
        assert grid.exploratory_support_budget is True
        grid.validate()
    assert adapter.column_grid.maximum_pixels == 256
    assert adapter.material_grid.maximum_pixels == request.wave_grid.maximum_pixels == 4096
    assert adapter.material_grid.compute_backend == request.wave_grid.compute_backend
    assert adapter.material_grid.angular_domain_padding == request.wave_grid.angular_domain_padding
    assert adapter.material_grid.specimen_quadrature_factor == request.wave_grid.specimen_quadrature_factor
    assert adapter.column_grid.angular_domain_padding == "minimum_full_band"
    assert not request.wave_grid.exploratory_support_budget
    assert asdict(request) == before
    assert legacy.material_grid is request.wave_grid
    assert not legacy.column_grid.exploratory_support_budget
    assert "support_budget_overrides" not in legacy.numerical_profile
    assert adapter.snapshot.numerical_digest == legacy.snapshot.numerical_digest
    assert adapter.runtime.geometry.digest == legacy.runtime.geometry.digest
    assert adapter.operator_options == legacy.operator_options
    assert adapter.identity != legacy.identity
    assert adapter.numerical_profile["maximum_support_probability_fraction"] == expected_p
    assert adapter.numerical_profile["maximum_accumulated_support_l2"] == expected_l2
    assert adapter.numerical_profile["support_budget_overrides"] == {
        "probability_fraction": probability, "relative_complex_l2": l2}
    assert "original gun" in adapter.numerical_profile["support_budget_meaning"]
    for name in ("source", "surface", "gun", "radial_gun", "wave_grid"):
        assert getattr(request, name) is getattr(old, name)


@pytest.mark.parametrize("mode", ["ideal", "analytical", "custom", "ideal_thin"])
def test_direct_adapter_cannot_use_override_outside_unified_mode(tmp_path, mode):
    state = _state()
    state.simulation_mode = mode
    with pytest.raises(ValueError, match="unified Finite-Field"):
        UnifiedPotentialWaveAdapter(state, _preview(tmp_path), runtime=object())


def test_large_l2_cannot_enter_ordinary_or_gun_grid(tmp_path):
    with pytest.raises(ValueError, match="maximum_accumulated_support_l2"):
        WaveGridNumerics(maximum_accumulated_support_l2=.5).validate()
    explicit = WaveGridNumerics(maximum_accumulated_support_l2=.5, exploratory_support_budget=True)
    explicit.validate()
    with pytest.raises(ValueError, match="wave_grid|gun|column"):
        replace(_preview(tmp_path), wave_grid=explicit).validate()


def test_nine_mode_execution_allocation_cache_and_readout_use_effective_budgets(tmp_path):
    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    request = _preview(tmp_path)
    source = _modes(9)
    original = source.digest
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    output = adapter.column(source, 1000.0002)
    assert source.digest == original
    assert len(output.beam.modes) == 9
    for before, after in zip(source.beam.modes, output.beam.modes, strict=True):
        assert after.mode_id == before.mode_id and after.energy_kev == before.energy_kev
        assert after.weight_per_reference_electron == pytest.approx(before.weight_per_reference_electron, rel=1e-13)
        allocation = output.record["unified_mode_contexts"][after.mode_id]["unified_support_allocation"]
        assert allocation["relative_l2_limit"] == .5
        assert allocation["probability_fraction_limit"] == .01
        assert allocation["maximum_l2_per_reference"] == .5*math.sqrt(before.weight_per_reference_electron)
        assert allocation["maximum_probability_per_reference"] == .01*before.weight_per_reference_electron
    summary = output.record["unified_support_summary"]
    assert len(summary["allocations"]) == 9
    assert summary["maximum_l2_per_reference"] == pytest.approx(.5*math.sqrt(source.beam.total_weight))
    assert summary["maximum_probability_per_reference"] == pytest.approx(.01*source.beam.total_weight)
    assert summary["normalization_applied"] is False
    store = ExecutedWaveStore(tmp_path/"preview", "column-preview-budgets", 2**28)
    restored = store.put(store.key("completed", output.digest), output)
    assert {key: value for key, value in restored.record.items() if key != "storage"} == output.record
    assert adapter.column(restored, restored.plane_z_mm) is restored
    text = coherent_preview_text(restored)
    assert "not accuracy qualified" in text and "probability 0.01, L2 0.5" in text
    assert "projection loss" in text
    for changes in ({"column_preview_support_probability": .005},
                    {"column_preview_support_l2": .25},
                    {"column_preview_support_probability": None, "column_preview_support_l2": None}):
        other = UnifiedPotentialWaveAdapter(state, replace(request, **changes), runtime=runtime)
        assert other.identity != adapter.identity
        with pytest.raises(ValueError, match="inputs changed"):
            other.column(restored, restored.plane_z_mm)
    assert restored.record["unified_support_summary"] == output.record["unified_support_summary"]


def test_legacy_executed_allocation_cannot_acquire_larger_allowance(tmp_path):
    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    new_request = _preview(tmp_path)
    old_request = replace(new_request, column_preview_support_probability=None, column_preview_support_l2=None)
    old = UnifiedPotentialWaveAdapter(state, old_request, runtime=runtime)
    new = UnifiedPotentialWaveAdapter(state, new_request, runtime=runtime)
    source = _modes(1)
    old_entrance = old._entrance(source, source.beam.modes[0])
    original = old_entrance.digest
    # Even without a published executor identity, the original allocation
    # binds its limits. A request cannot manufacture a fresh source budget.
    with pytest.raises(ValueError, match="support limits changed"):
        new._entrance(old_entrance, old_entrance.beam.modes[0])
    assert old_entrance.digest == original


def test_spent_preview_prefix_stays_nonspendable_across_conditional_descendants(tmp_path):
    from temsim.physics.potential_support_lineage import capture_inherited_support
    from temsim.physics.unified_wave_support import prepare_unified_support, stamp_unified_support, unified_support_summary
    from temsim.physics.wave_flux import BeamState
    state = _state()
    adapter = UnifiedPotentialWaveAdapter(state, _preview(tmp_path), runtime=_runtime(state, 1., 1.0000004))
    source = _modes(1)
    parent = adapter._entrance(source, source.beam.modes[0])
    mode = parent.beam.modes[0]
    record = thaw_json(parent.record)
    ledger = NumericalSupportLedger.from_dict(record["support_ledger"])
    projected, spent = ledger.compact(mode.plane, wavelength_m=float(wavelength_m(mode.energy_kev*1000.)),
        z_m=parent.plane_z_mm*.001, mode_weight=mode.weight_per_reference_electron)
    assert spent.used_l2 > 0 and spent.used_probability > 0
    norm = projected.probability
    mode = replace(mode, plane=replace(projected, amplitude=projected.amplitude/math.sqrt(norm)),
                   weight_per_reference_electron=mode.weight_per_reference_electron*norm)
    record.update(support_ledger=spent.to_dict(), snapshot_digest=adapter.snapshot.numerical_digest)
    record = stamp_unified_support(record)
    parent = replace(parent, beam=BeamState((mode,), parent.beam.reference_plane), record=record)
    prefix = capture_inherited_support(parent, adapter.snapshot.numerical_digest)
    inherited = thaw_json(parent.record)
    inherited.pop("support_ledger")
    inherited.update(inherited_potential_support=prefix, support_prefix_digest=prefix["digest"])
    contexts = {}
    for index in range(3):
        child = replace(mode, mode_id=f"{mode.mode_id}/conditional:{index}",
            weight_per_reference_electron=mode.weight_per_reference_electron/3)
        context = prepare_unified_support(inherited, child, enabled=True, l2_limit=.5,
            probability_limit=.01, source_digest=source.digest, allow_allocate=False)
        assert context.get("support_ledger") is None
        assert context["inherited_potential_support"]["ledger"] == spent.to_dict()
        contexts[child.mode_id] = context
    summary = unified_support_summary(contexts)
    assert len(summary["allocations"]) == 1
    assert summary["used_l2_per_reference"] == spent.used_l2
    assert summary["used_probability_per_reference"] == spent.used_probability
    assert summary["maximum_l2_per_reference"] == ledger.maximum_l2
    assert summary["maximum_probability_per_reference"] == ledger.maximum_probability
    assert summary["allocations"][0]["further_projection_allowed"] is False
    assert not summary["normalization_applied"]


def test_mixed_executed_preview_labels_do_not_pool_independent_allowances(tmp_path):
    state = _state()
    runtime = _runtime(state, 1., 1.0000004)
    source = _modes(1)
    large = UnifiedPotentialWaveAdapter(state, _preview(tmp_path), runtime=runtime).column(source, 1000.0001)
    small = UnifiedPotentialWaveAdapter(state, _preview(tmp_path,
        column_preview_support_probability=.001, column_preview_support_l2=.1), runtime=runtime).column(source, 1000.0001)
    ensemble = SimpleNamespace(members=(SimpleNamespace(checkpoint=large), SimpleNamespace(checkpoint=small)),
        relative_weights=(.25, .75))
    text = coherent_preview_text(ensemble)
    assert "member budgets and losses are separate" in text
    assert "probability 0.01, L2 0.5" in text and "probability 0.001, L2 0.1" in text
    selected = coherent_preview_text(ensemble, member_index=1)
    assert "Inspecting state 2" in selected and "probability 0.001, L2 0.1" in selected
    assert "probability 0.01, L2 0.5" not in selected
    assert ensemble.relative_weights == (.25, .75)


def test_original_gun_cache_is_reused_across_column_only_budget_changes(tmp_path, monkeypatch):
    from collections import OrderedDict
    from temsim.physics import compute_backend
    import temsim.physics.tip_wave_pipeline as pipeline
    import temsim.physics.scan_preparation as scan_preparation
    from temsim.physics.tip_wave_pipeline import simulate_tip_wave
    state = _state()
    source = replace(_modes(1), plane_z_mm=state.electron_gun.exit_plane_z_mm)
    calls = []
    monkeypatch.setattr(pipeline, "_STAGES", OrderedDict())
    monkeypatch.setattr(pipeline, "_OBSERVATION_PREFIXES", OrderedDict())
    monkeypatch.setattr(scan_preparation, "prepare_scan_drives", lambda *_a, **_k: None)
    # The source producer is replaced below and this request stops at its exit.
    monkeypatch.setattr(compute_backend, "coherent_cuda_capability", lambda:
        compute_backend.BackendCapability(True, "Substituted gun-cache dispatch"))
    def source_fixture(*args, **kwargs):
        calls.append((kwargs["source_numerics"], kwargs["numerics"]))
        return source
    monkeypatch.setattr(pipeline, "build_tip_gun_checkpoint", source_fixture)
    request = _preview(tmp_path, stop="gun_exit")
    request = replace(request, wave_grid=replace(request.wave_grid, compute_backend="Require GPU"))
    original_request = asdict(request)
    first = simulate_tip_wave(state, request)
    changed = replace(request, column_preview_support_l2=.25, column_preview_support_probability=.005)
    second = simulate_tip_wave(state, changed)
    assert len(calls) == 1
    assert second.propagation_cache_hit
    assert first.checkpoint.digest == second.checkpoint.digest == source.digest
    assert calls == [(request.source, request.gun)]
    assert asdict(request) == original_request
    assert json_digest(asdict(request)) != json_digest(asdict(changed))


def test_actual_material_slices_keep_preview_allocation_without_reissuing_branches(tmp_path, monkeypatch):
    from test_potential_specimen_transport import _fixture
    from temsim.physics.wave_execution import InelasticWaveNumerics
    import temsim.physics.potential_specimen_transport as material_module
    state, source = _fixture()
    state.simulation_mode = "unified"
    state.sample.wave_frozen_phonon_enabled = True
    state.sample.wave_frozen_phonon_configurations = 2
    state.sample.wave_frozen_phonon_sigma_angstrom = .02
    source = replace(source, plane_z_mm=source.plane_z_mm-.000001)
    runtime = _runtime(state, source.plane_z_mm*.001, (state.sample.z_mm+.00001)*.001)
    # The real atomistic projection resolves onto 1024 squared samples;
    # unlike the small vacuum fixtures, it needs the normal preview cap.
    request = _preview(tmp_path, column_preview_pixels=2048,
        inelastic=InelasticWaveNumerics(method="zero_loss"))
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    source = adapter.column(source, state.sample.z_mm-state.sample.thickness_nm*.5e-6)
    before = source.record["unified_support_summary"]
    original_call = material_module.propagate_potential_specimen
    grids = []
    def recorded_material(*args, **kwargs):
        grids.append(kwargs["grid_numerics"])
        return original_call(*args, **kwargs)
    monkeypatch.setattr(material_module, "propagate_potential_specimen", recorded_material)
    store = ExecutedWaveStore(tmp_path/"material", "preview-material-budget-hand-off", 2**28)
    material = adapter.specimen(source, store=store, use_cache=False)
    output = adapter.column(material, state.sample.z_mm+.000001)
    assert len(grids) == 1 and grids[0] is adapter.material_grid
    assert grids[0].maximum_pixels == request.wave_grid.maximum_pixels
    assert grids[0].maximum_accumulated_support_l2 == .5
    assert grids[0].maximum_support_loss_fraction == .01
    assert len(output.beam.modes) == 2
    assert output.beam.total_weight <= source.beam.total_weight
    for context in output.record["unified_mode_contexts"].values():
        assert context.get("support_ledger") is None
        assert context["unified_support_allocation"]["relative_l2_limit"] == .5
        assert context["unified_support_allocation"]["probability_fraction_limit"] == .01
    after = output.record["unified_support_summary"]
    assert len(after["allocations"]) == 1
    for key in ("maximum_l2_per_reference", "maximum_probability_per_reference",
                "used_l2_per_reference", "used_probability_per_reference"):
        assert after[key] == before[key]
    assert not after["allocations"][0]["further_projection_allowed"]
    assert "probability 0.01, L2 0.5" in coherent_preview_text(output)
