"""Public projected-STEM admission and routing; controlled solvers are not qualification."""
from types import SimpleNamespace

import numpy as np
import pytest

from specimen_inputs import imported_sample
from test_simulation_workflows import case
from temsim import simulation_pipeline as pipeline
from temsim.calculation_workflow import admit_workflow
from temsim.optics.column import default_state


@pytest.fixture
def projected_state(monkeypatch):
    state = default_state()
    state.simulation_mode = "unified"
    state.acceleration_backend = "CPU"
    imported_sample(state)
    state.sample.stem_particle_model = "projected_atoms"
    state.sample.stem_wave_enabled = False
    state.sample.stem_fourdstem_enabled = False
    state.sample.wave_enabled = False
    state.energy_filter.enabled = False
    # This fixture tests workflow ownership, not the independent field-domain
    # preflight or the physical gun/column propagators.
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight",
                        lambda _state: {"issues": (), "supported": True})
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.calibrate_unified_scan",
                        lambda *_args, **_kwargs: None)
    return state


def _physical_controls(state):
    return (
        state.simulation_mode,
        tuple((lens.key, lens.percent, lens.polarity, lens.enabled) for lens in state.lenses),
        state.sample.cif_path, state.sample.thickness_nm, state.sample.inserted,
        state.sample.stem_particle_model, state.sample.stem_wave_enabled,
        state.sample.stem_fourdstem_enabled, state.energy_filter.enabled,
        tuple((detector.key, detector.inserted, detector.readout_enabled)
              for detector in state.stem_detectors),
    )


def test_unified_projected_stem_admission_preserves_selected_physics(projected_state):
    state = projected_state
    before = _physical_controls(state)
    admit_workflow(state, "stem")
    assert _physical_controls(state) == before


@pytest.mark.parametrize(("owner", "field", "value", "message"), (
    ("sample", "stem_wave_enabled", True, "coherent STEM"),
    ("sample", "stem_fourdstem_enabled", True, "4D-STEM"),
    ("sample", "stem_particle_model", "material_paths", "material-path"),
    ("energy_filter", "enabled", True, "Energy Filter"),
))
@pytest.mark.parametrize("entry", ("workflow", "direct_detector"))
def test_unified_stem_unsupported_products_fail_before_work_and_do_not_switch_model(
        projected_state, monkeypatch, owner, field, value, message, entry):
    state = projected_state
    setattr(getattr(state, owner), field, value)
    before = _physical_controls(state)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported acquisition must be rejected before executing a workflow")
    monkeypatch.setattr("temsim.simulation_workflow.calculate_workflow", forbidden)
    with pytest.raises(ValueError, match=message):
        if entry == "workflow":
            pipeline.calculate(state, workflow="stem")
        else:
            from temsim.detector.stem_signal import acquire_stem_scan
            acquire_stem_scan(SimpleNamespace(), state)
    assert _physical_controls(state) == before


@pytest.mark.parametrize("workflow", ("full", "sample", "receiver", "eds"))
def test_projected_stem_admission_does_not_enable_unmigrated_material_workflows(
        projected_state, monkeypatch, workflow):
    state = projected_state
    before = _physical_controls(state)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Projected-STEM support cannot admit a generic material calculation")
    monkeypatch.setattr(pipeline, "run", forbidden)
    monkeypatch.setattr("temsim.simulation_workflow.calculate_workflow", forbidden)
    with pytest.raises(ValueError, match="finite-volume particle specimen"):
        pipeline.calculate(state, workflow=workflow)
    assert _physical_controls(state) == before


def test_unified_stem_checks_shared_field_support_before_route(projected_state, monkeypatch):
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight",
                        lambda _state: {"issues": ("Unsupported B-only field",), "supported": False})
    monkeypatch.setattr("temsim.simulation_workflow.calculate_workflow",
                        lambda *_args, **_kwargs: pytest.fail("Invalid shared fields reached the workflow"))
    with pytest.raises(ValueError, match="Unsupported B-only field"):
        pipeline.calculate(projected_state, workflow="stem")


def test_unified_stem_requires_executed_source_instead_of_creating_sample_source(projected_state, monkeypatch):
    monkeypatch.setattr(pipeline, "run",
                        lambda *_args, **_kwargs: pytest.fail("A missing incident result retraced the gun"))
    with pytest.raises(ValueError, match="Run Ray Diagram"):
        pipeline.calculate(projected_state, workflow="stem")


@pytest.fixture
def controlled_workflow(case, monkeypatch):
    """Use the real public route with explicit synthetic completed products."""
    case.state.simulation_mode = "unified"
    case.state.sample.stem_particle_model = "projected_atoms"
    case.state.sample.stem_wave_enabled = False
    case.state.sample.stem_fourdstem_enabled = False
    case.state.ac_deflector.enabled = case.state.ac_deflector.scan_enabled = True
    monkeypatch.setattr("temsim.physics.unified_field_runtime.unified_runtime_preflight",
                        lambda _state: {"issues": (), "supported": True})
    case.order = []
    def calibrate(*_args, **_kwargs):
        case.order.append("unified_calibration")
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.calibrate_unified_scan", calibrate)
    original_signatures = pipeline.calculation_signatures
    def signatures(state):
        case.order.append("signatures")
        return original_signatures(state)
    monkeypatch.setattr(pipeline, "calculation_signatures", signatures)
    from temsim import simulation_workflow
    original_requirement = simulation_workflow.require_incident_result
    def require_incident(*args, **kwargs):
        case.order.append("incident_validation")
        return original_requirement(*args, **kwargs)
    monkeypatch.setattr(simulation_workflow, "require_incident_result", require_incident)
    def forbidden(*_args, **_kwargs):
        pytest.fail("A projected intensity workflow executed an unmigrated physical product")
    for name in ("run_specimen_interactions", "build_geometric_specimen_exit",
                 "calculate_scan_geometry", "calculate_scan_ray_paths"):
        monkeypatch.setattr(pipeline, name, forbidden)
    monkeypatch.setattr("temsim.physics.scan_geometry.calibrate_scan_system", forbidden)
    monkeypatch.setattr("temsim.physics.scan_geometry.paired_kick_response", forbidden)
    def acquire(state, simulation, **kwargs):
        case.calls.append(("unified_stem", simulation, kwargs))
        assert state is case.state
        return SimpleNamespace(metrics={"unified_potential_stem": True})
    monkeypatch.setattr(pipeline, "calculate_stem_scan_frame", acquire)
    return case


def _executed_reference(case):
    rays = pipeline.calculate(case.state, workflow="rays")
    case.calls.clear()
    case.order.clear()
    return rays


def test_public_unified_stem_retains_executed_reference_without_fake_material_checkpoint(controlled_workflow):
    context = controlled_workflow
    rays = _executed_reference(context)
    original_metrics = dict(rays.simulation.metrics)
    checkpoints = rays.simulation.incident_checkpoints
    clocks, energies = checkpoints.flight_time_s.copy(), checkpoints.kinetic_energy_ev.copy()
    result = pipeline.calculate(context.state, workflow="stem", existing_result=rays)
    assert [call[0] for call in context.calls] == ["unified_stem"]
    assert context.calls[0][1].incident is rays.simulation.incident
    assert result.simulation.incident_checkpoints is checkpoints
    np.testing.assert_array_equal(checkpoints.flight_time_s, clocks)
    np.testing.assert_array_equal(checkpoints.kinetic_energy_ev, energies)
    assert result.stem_scan.metrics["unified_potential_stem"]
    assert result.specimen_exit is None and result.specimen_interactions is None
    assert result.sample_region is None and result.energy_filter is None
    assert result.simulation.metrics["sample_scattering_applied"] is False
    assert result.simulation.metrics["sample_scattering_model"] == "projected_detector_response_only"
    assert rays.simulation.metrics == original_metrics
    assert "stem" in result.calculated_products
    assert {"incident", "column"} <= result.reused_products


def test_unified_scan_calibration_precedes_signature_and_incident_validation(controlled_workflow):
    context = controlled_workflow
    rays = _executed_reference(context)
    pipeline.calculate(context.state, workflow="stem", existing_result=rays)
    assert context.order.index("unified_calibration") < context.order.index("signatures")
    assert context.order.index("signatures") < context.order.index("incident_validation")


def test_public_unified_stem_exact_frame_reuse_requires_field_owned_metadata(controlled_workflow):
    context = controlled_workflow
    rays = _executed_reference(context)
    first = pipeline.calculate(context.state, workflow="stem", existing_result=rays)
    context.calls.clear()
    reused = pipeline.calculate(context.state, workflow="stem", existing_result=first)
    assert not context.calls
    assert reused.stem_scan is first.stem_scan
    assert "stem" in reused.reused_products and "stem" not in reused.calculated_products


@pytest.mark.parametrize("metadata", (None, {}, {"unified_potential_stem": False}))
def test_public_unified_stem_rejects_legacy_frame_cache_without_field_receipt(controlled_workflow, metadata):
    context = controlled_workflow
    rays = _executed_reference(context)
    first = pipeline.calculate(context.state, workflow="stem", existing_result=rays)
    old_frame = first.stem_scan
    old_frame.metrics = metadata
    context.calls.clear()
    result = pipeline.calculate(context.state, workflow="stem", existing_result=first)
    assert [call[0] for call in context.calls] == ["unified_stem"]
    assert result.stem_scan is not old_frame
    assert old_frame.metrics is metadata
    assert "stem" in result.calculated_products and "stem" not in result.reused_products


def test_public_unified_stem_rejects_changed_source_before_projected_response(controlled_workflow):
    context = controlled_workflow
    rays = _executed_reference(context)
    context.state.lenses[0].percent *= 1.03
    with pytest.raises(ValueError, match="Run Ray Diagram"):
        pipeline.calculate(context.state, workflow="stem", existing_result=rays)
    assert not context.calls


def test_public_unified_stem_cannot_return_a_wave_diffraction_sink(controlled_workflow):
    context = controlled_workflow
    rays = _executed_reference(context)
    with pytest.raises(ValueError, match="diffraction-wave sink"):
        pipeline.calculate(context.state, workflow="stem", existing_result=rays,
                           diffraction_sink=object())
    assert not context.calls


def test_unified_workflow_downstream_preserves_exact_incident_energy_and_clock_without_cc(case, monkeypatch):
    from temsim.simulation_workflow import rebuild_optical_downstream
    state = case.state
    state.simulation_mode = "unified"
    cp = case.simulation.incident_checkpoints
    for name in ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s", "kinetic_energy_ev"):
        setattr(cp, name, getattr(cp, name).copy())
    cp.x_m[-1] = [1.7e-7, -2.1e-7]
    cp.y_m[-1] = [-3.6e-7, 4.2e-7]
    cp.kinetic_energy_ev[-1] = [299998.25, 300001.5]
    cp.flight_time_s[-1] = [8.3e-9, 9.1e-9]
    cp.tx_rad[-1] = [2e-4, -3e-4]
    cp.ty_rad[-1] = [-4e-4, 5e-4]
    def forbidden(*_args, **_kwargs):
        pytest.fail("An empirical Cc correction was added to shared-field transport")
    monkeypatch.setattr("temsim.physics.chromatic.objective_chromatic_kick_from_state", forbidden)
    monkeypatch.setattr("temsim.physics.chromatic.configured_objective_chromatic_focal_mm", forbidden)
    monkeypatch.setattr("temsim.physics.optical_tuning.tuning_metrics", lambda *_args: {})
    captured = []
    def trace(*args, **kwargs):
        phase, offsets = args[4], args[5]
        for actual, name in zip(phase, ("x_m", "tx_rad", "y_m", "ty_rad", "flight_time_s")):
            np.testing.assert_array_equal(actual, getattr(cp, name)[-1])
        np.testing.assert_array_equal(kwargs["initial_kinetic_energy_ev"], cp.kinetic_energy_ev[-1])
        np.testing.assert_array_equal(offsets, cp.kinetic_energy_ev[-1] - state.beam_voltage_kv * 1000.)
        assert kwargs["initial_kicks"] is False
        captured.append(True)
        return SimpleNamespace(branch=SimpleNamespace()), None, None
    monkeypatch.setattr("temsim.physics.particle_sections._trace_segment", trace)
    result = rebuild_optical_downstream(state, case.simulation)
    assert captured == [True]
    assert result.incident is case.simulation.incident
    assert result.incident_checkpoints is cp
