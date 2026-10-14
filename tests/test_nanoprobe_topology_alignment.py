"""Both Nanoprobe ranges must satisfy condenser topology and specimen focus."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.operating_modes import direct_alignment_by_key
import temsim.optics.direct_alignment as alignment


def _definition():
    definition = direct_alignment_by_key(alignment.NANOPROBE_CONVERGENCE)
    return replace(definition, applies_to_modes=("nano_probe", "nano_probe_large_angle"), targets={
        **definition.targets,
        "condenser_topology": "c2_c3_waist_no_c3_adl_waist",
        "large_angle_condenser_topology": "c2_c3_waist_and_c3_adl_waist",
        "optimiser_step_mm": 0.1,
        "validation_step_mm": 0.05,
    })


def _state(*, corrected=True, nanoprobe_range="normal"):
    lenses = [SimpleNamespace(key=key, percent=value, max_percent=100.0)
              for key, value in zip(alignment.CONDENSER_KEYS, (25.0, 20.0))]
    if corrected:
        lenses.append(SimpleNamespace(key="adapter_lens", percent=40.0,
                                      max_percent=100.0))
    return SimpleNamespace(
        lenses=lenses, illumination_mode="STEM", projector_mode="imaging",
        nanoprobe_range=nanoprobe_range,
        probe_corrector_installed=corrected,
        equivalent_image_lenses_enabled=False,
        condenser_aperture_2=SimpleNamespace(diameter_um=100.0),
        step_mm=0.2, acceleration_enabled=True, acceleration_backend="Auto",
        active_backend="original", apertures=[],
        electron_gun=SimpleNamespace(exit_plane_z_mm=0.0),
        sample=SimpleNamespace(z_mm=10.0),
    )


def _measurement(status="PASS", *, waist=0.0, value=30.0):
    return alignment.DirectAlignmentMeasurement(
        key=alignment.NANOPROBE_CONVERGENCE, value=value, unit="mrad",
        constraint_value=waist, constraint_unit="mm",
        condenser_topology_status=status,
        condenser_topology_message=f"Nanoprobe topology: {status}.",
    )


@pytest.mark.parametrize("nanoprobe_range,token", (
    ("normal", "condenser_topology"),
    ("large_angle", "large_angle_condenser_topology"),
))
@pytest.mark.parametrize("bad_value", (None, "wrong_branch"))
def test_missing_or_wrong_range_declaration_cannot_disable_the_gate(nanoprobe_range, token, bad_value):
    state = _state(nanoprobe_range=nanoprobe_range)
    definition = _definition()
    targets = dict(definition.targets)
    if bad_value is None:
        targets.pop(token)
    else:
        targets[token] = bad_value
    with pytest.raises(ValueError, match=token):
        alignment._nanoprobe_topology_required(state, replace(definition, targets=targets))


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
def test_both_ranges_share_the_registered_alignment_control(nanoprobe_range):
    state = _state(nanoprobe_range=nanoprobe_range)
    assert alignment._mode_matches(state, _definition())
    assert alignment._nanoprobe_topology_required(state, _definition())
    assert not alignment._mode_matches(state, replace(_definition(), applies_to_modes=("micro_probe",)))


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
def test_optimizer_topology_helper_preserves_selected_range(monkeypatch, nanoprobe_range):
    from temsim.optics import condenser_topology

    model = alignment._CondenserMeasurementModel.__new__(alignment._CondenserMeasurementModel)
    model.state = _state(nanoprobe_range=nanoprobe_range)
    model.topology_planes = (2., 4.)
    model.apertures = ()
    model.source_rays = np.zeros((4, 6))
    model.source_alive = np.ones(6, bool)
    model.source_weights = np.ones(6)
    model.sample_model = SimpleNamespace(full_field_transfer=False,
        matrices_at=lambda vector, planes: np.repeat(np.eye(4)[None, :, :], len(planes), axis=0))

    def evaluate(state, *_args):
        assert state is model.state
        return dict(status="PASS", nanoprobe_range=state.nanoprobe_range)

    monkeypatch.setattr(condenser_topology, "evaluate", evaluate)
    assert model.topology((25., 20.))["nanoprobe_range"] == nanoprobe_range


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
@pytest.mark.parametrize("status", ("FAIL", "UNRESOLVED", None))
def test_focused_candidate_with_wrong_or_unresolved_topology_restores_controls(
    monkeypatch, status, nanoprobe_range,
):
    state = _state(nanoprobe_range=nanoprobe_range)
    initial = alignment._get_vector(state, alignment.CONDENSER_KEYS)
    candidate = np.asarray((26.0, 18.0))
    optimisations = []

    def optimise(*_args):
        optimisations.append(True)
        return candidate.copy(), 1

    monkeypatch.setattr(alignment, "_optimise_condenser", optimise)
    monkeypatch.setattr(alignment, "_refine_nanoprobe_production_focus",
                        lambda _state, _definition, _target, vector, _step: (vector, 0))
    monkeypatch.setattr(alignment, "_validate_condenser_production",
                        lambda *_args: _measurement(status))

    result = alignment._solve_direct_alignment(
        state, alignment.NANOPROBE_CONVERGENCE, 30.0, definition=_definition(),
    )

    assert optimisations, "A focused but wrong-branch current point cannot bypass search"
    assert not result.success
    np.testing.assert_array_equal(
        alignment._get_vector(state, alignment.CONDENSER_KEYS), initial,
    )
    assert dict(result.candidate_strengths) == dict(zip(alignment.CONDENSER_KEYS, candidate))
    assert "previous lens values were restored" in result.message


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
@pytest.mark.parametrize("coarse_status,fine_status,waist,expected", (
    ("PASS", "PASS", 0.0, True),
    ("FAIL", "PASS", 0.0, False),
    ("PASS", "UNRESOLVED", 0.0, False),
    ("PASS", "PASS", 1.0, False),
))
def test_topology_does_not_replace_sample_focus_or_two_step_validation(
    monkeypatch, coarse_status, fine_status, waist, expected, nanoprobe_range,
):
    state = _state(nanoprobe_range=nanoprobe_range)
    definition = _definition()
    candidate = np.asarray((26.0, 18.0))
    initial = alignment._get_vector(state, alignment.CONDENSER_KEYS)

    def validate(_state, _definition, vector, step):
        if np.array_equal(vector, initial):
            return _measurement("FAIL")
        status = (fine_status if step == definition.targets["validation_step_mm"]
                  else coarse_status)
        return _measurement(status, waist=waist)

    monkeypatch.setattr(alignment, "_validate_condenser_production", validate)
    monkeypatch.setattr(alignment, "_optimise_condenser",
                        lambda *_args: (candidate.copy(), 1))
    monkeypatch.setattr(alignment, "_refine_nanoprobe_production_focus",
                        lambda _state, _definition, _target, vector, _step: (vector, 0))

    result = alignment._solve_direct_alignment(
        state, alignment.NANOPROBE_CONVERGENCE, 30.0, definition=definition,
    )

    assert result.success is expected
    np.testing.assert_array_equal(
        alignment._get_vector(state, alignment.CONDENSER_KEYS),
        candidate if expected else initial,
    )


def test_column_without_adl_keeps_its_existing_focus_control(monkeypatch):
    state = _state(corrected=False)
    monkeypatch.setattr(alignment, "_validate_condenser_production",
                        lambda *_args: _measurement(None))
    monkeypatch.setattr(alignment, "_optimise_condenser",
                        lambda *_args: pytest.fail("The current focus already satisfies this column"))

    result = alignment._solve_direct_alignment(
        state, alignment.NANOPROBE_CONVERGENCE, 30.0, definition=_definition(),
    )

    assert result.success
    assert result.iterations == 0


def test_optimizer_continues_after_a_focused_wrong_topology_branch(monkeypatch):
    state = _state()
    seeds = (np.asarray((25.0, 40.0)), np.asarray((25.0, 15.0)))
    screened = []

    class Model:
        upper = np.full(2, 100.0)

        def __init__(self, _state, *, step_mm, topology):
            assert topology

        def measure(self, vector):
            return vector

        def topology(self, vector):
            screened.append(tuple(vector))
            return dict(
                status="PASS" if vector[1] == 15.0 else "FAIL",
                refinement_planes_mm=(),
            )

    monkeypatch.setattr(alignment, "_CondenserMeasurementModel", Model)
    monkeypatch.setattr(alignment, "_condenser_measurement",
                        lambda *_args: _measurement())
    monkeypatch.setattr(alignment, "_deterministic_seeds", lambda *_args, **_kwargs: seeds)
    monkeypatch.setattr(alignment, "interpolated_nanoprobe_seed", lambda *_args: None)
    monkeypatch.setattr(alignment, "least_squares",
                        lambda _residual, seed, **_kwargs: SimpleNamespace(x=seed, nfev=1))

    candidate, iterations = alignment._optimise_condenser(state, _definition(), 30.0)

    np.testing.assert_array_equal(candidate, seeds[1])
    assert screened == [tuple(seed) for seed in seeds]
    assert iterations == 2


@pytest.mark.parametrize("refine,missing_weights", (
    (False, False), (True, False), ("always", False), (False, True),
))
def test_production_topology_uses_executed_checkpoints_and_upstream_survivors(
    monkeypatch, refine, missing_weights,
):
    from temsim.optics import condenser_topology
    from temsim.optics.electron_gun import source

    state = _state()
    count = 6
    weights = None if missing_weights else np.ones(count)
    emitted = SimpleNamespace(
        alive=np.ones(count, dtype=bool), weight=weights,
        x_m=np.linspace(-1.0e-6, 1.0e-6, count), y_m=np.zeros(count),
        tx_rad=np.linspace(-0.01, 0.01, count), ty_rad=np.zeros(count),
        energy_offset_ev=np.zeros(count),
    )
    gun_trace = SimpleNamespace(exit_bundle=emitted,
                               blocked_z_mm=np.full(count, np.nan),
                               blocked_key=[None] * count)
    monkeypatch.setattr(source, "trace_source_to_exit", lambda _state: gun_trace)
    monkeypatch.setattr(alignment, "_pre_sample_kick_events", lambda _state: ())
    monkeypatch.setattr(condenser_topology, "requested_capture_planes",
                        lambda _state: (2.0, 4.0))
    produced = []

    def propagate(*_args, **kwargs):
        assert kwargs["return_checkpoints"] is True
        assert set(kwargs["checkpoint_z_mm"]).issubset(kwargs["save_z_mm"])
        planes = np.asarray(kwargs["checkpoint_z_mm"])
        checkpoints = SimpleNamespace(z_mm=planes, **{
            key: np.tile(getattr(emitted, key), (len(planes), 1))
            for key in ("x_m", "y_m", "tx_rad", "ty_rad")
        })
        produced.append(checkpoints)
        history = [np.tile(getattr(emitted, key), (2, 1))
                   for key in ("x_m", "tx_rad", "y_m", "ty_rad")]
        return np.asarray((0.0, 10.0)), *history, checkpoints

    def clip(_state, _z, _x, _y, alive, stops, reasons):
        # This later specimen-side stop must not remove ray 0 from either
        # condenser checkpoint; ray 1 is stopped between the checkpoints.
        alive[:2] = False
        stops[:2] = (8.0, 3.0)
        return alive, stops, reasons

    def evaluate(_state, checkpoints, masks, supplied_weights):
        assert checkpoints is produced[-1]
        assert supplied_weights is weights
        assert masks[:, 0].all()
        np.testing.assert_array_equal(masks[:, 1], checkpoints.z_mm < 3.0)
        unresolved = missing_weights or refine == "always" or (refine and len(produced) == 1)
        next_plane = 3.0 + len(produced) * 0.1 if refine == "always" else 3.0
        return dict(
            status="UNRESOLVED" if unresolved else "PASS",
            message="Executed topology verified.",
            refinement_planes_mm=(next_plane,) if unresolved and not missing_weights else (),
        )

    monkeypatch.setattr(alignment, "propagate", propagate)
    monkeypatch.setattr(alignment, "clip_segment", clip)
    monkeypatch.setattr(alignment, "clip_column_wall",
                        lambda _state, _z, _x, _y, alive, stops, reasons: (alive, stops, reasons))
    monkeypatch.setattr(condenser_topology, "evaluate", evaluate)

    measured = alignment._validate_condenser_production(
        state, _definition(), np.asarray((26.0, 18.0)), 0.1,
    )

    assert measured.condenser_topology_status == (
        "UNRESOLVED" if missing_weights or refine == "always" else "PASS"
    )
    assert len(produced) == (3 if refine == "always" else 2 if refine else 1)
    assert state.active_backend == "original"
    assert state.acceleration_enabled
    np.testing.assert_array_equal(alignment._get_vector(state, alignment.CONDENSER_KEYS), (25.0, 20.0))


def test_production_topology_exception_restores_trial_controls_and_backend(monkeypatch):
    from temsim.optics import condenser_topology
    from temsim.optics.electron_gun import source

    state = _state()
    emitted = SimpleNamespace(x_m=np.zeros(6), y_m=np.zeros(6), tx_rad=np.zeros(6),
                              ty_rad=np.zeros(6), energy_offset_ev=np.zeros(6))
    monkeypatch.setattr(source, "trace_source_to_exit",
                        lambda _state: SimpleNamespace(exit_bundle=emitted))
    monkeypatch.setattr(condenser_topology, "requested_capture_planes", lambda _state: (2.0, 4.0))
    monkeypatch.setattr(alignment, "_pre_sample_kick_events", lambda _state: ())

    def fail(*_args, **_kwargs):
        assert state.active_backend == "CPU"
        np.testing.assert_array_equal(alignment._get_vector(state, alignment.CONDENSER_KEYS), (26.0, 18.0))
        raise RuntimeError("Checkpoint integration failed")

    monkeypatch.setattr(alignment, "propagate", fail)
    with pytest.raises(RuntimeError, match="Checkpoint integration failed"):
        alignment._validate_condenser_production(state, _definition(), np.asarray((26.0, 18.0)), 0.1)

    assert state.active_backend == "original"
    assert state.acceleration_enabled
    assert state.acceleration_backend == "Auto"
    assert state.step_mm == 0.2
    np.testing.assert_array_equal(alignment._get_vector(state, alignment.CONDENSER_KEYS), (25.0, 20.0))


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
@pytest.mark.parametrize("status", ("PASS", "FAIL", "UNRESOLVED"))
def test_final_topology_reuses_sufficient_checkpoints_and_audits_missing_evidence(
    monkeypatch, status, nanoprobe_range,
):
    from temsim import alignment_transaction as transaction
    from temsim.optics import condenser_topology

    cp = SimpleNamespace(z_mm=np.asarray((2.0, 4.0)))
    emitted = SimpleNamespace(alive=np.ones(3, dtype=bool), weight=np.ones(3))
    simulation = SimpleNamespace(
        incident_checkpoints=cp, gun_trace=SimpleNamespace(exit_bundle=emitted),
        incident=SimpleNamespace(blocked_z=np.asarray((8.0, 3.0, np.nan))),
    )
    fallback_calls = []

    def evaluate(_state, checkpoints, masks, weights):
        assert _state.nanoprobe_range == nanoprobe_range
        assert checkpoints is cp
        assert weights is emitted.weight
        np.testing.assert_array_equal(masks, ((True, True, True), (True, False, True)))
        return dict(status=status, message="Retained checkpoint evidence")

    def validate(state, *, step_mm):
        assert state.nanoprobe_range == nanoprobe_range
        fallback_calls.append(step_mm)
        return dict(status="PASS", message="New exact checkpoints")

    monkeypatch.setattr(condenser_topology, "evaluate", evaluate)
    monkeypatch.setattr(condenser_topology, "validate", validate)
    report = transaction._validate_condenser_topology_forward(
        SimpleNamespace(step_mm=0.025, nanoprobe_range=nanoprobe_range), simulation,
    )

    assert fallback_calls == ([0.025] if status == "UNRESOLVED" else [])
    assert report["status"] == ("PASS" if status == "UNRESOLVED" else status)


@pytest.fixture
def physical_state():
    from temsim.assembly_catalog import AssemblyCatalog
    from temsim.optics.column import default_state

    state = default_state()
    catalog = AssemblyCatalog()
    catalog.apply(state, catalog.default_selection())
    state.illumination_mode = "STEM"
    return state


def test_uncorrected_assembly_disabled_adl_template_does_not_enable_gate(physical_state):
    from temsim.assembly_catalog import AssemblyCatalog

    catalog = AssemblyCatalog()
    uncorrected = next(option for option in catalog.columns
                      if option.properties["c3_lens"] and not option.properties["probe_corrector"])
    catalog.apply(physical_state, replace(catalog.default_selection(), column=uncorrected.name))

    adl = next(lens for lens in physical_state.lenses if lens.key == "adapter_lens")
    assert not adl.enabled
    assert not physical_state.probe_corrector_installed
    assert not alignment._nanoprobe_topology_required(physical_state, _definition())


@pytest.mark.parametrize("missing", (False, True))
def test_installed_corrector_broken_adl_cannot_disable_gate(physical_state, missing):
    adl = next(lens for lens in physical_state.lenses if lens.key == "adapter_lens")
    if missing:
        physical_state.lenses.remove(adl)
    else:
        adl.enabled = False
    assert physical_state.probe_corrector_installed
    assert alignment._nanoprobe_topology_required(physical_state, _definition())


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
@pytest.mark.parametrize("waist", (0., 1.))
@pytest.mark.parametrize("status", ("PASS", "FAIL", "UNRESOLVED"))
def test_independent_forward_topology_can_reject_a_solver_success(
    physical_state, monkeypatch, status, waist, nanoprobe_range,
):
    from temsim import alignment_transaction as transaction
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.working_point import WorkingPointCheckpoint

    state = physical_state
    state.nanoprobe_range = nanoprobe_range
    before = capture_instrument_snapshot(state).digest
    definition = _definition()
    monkeypatch.setattr(transaction, "direct_alignment_by_key", lambda _key: definition)
    request = transaction.AlignmentRequest.capture(
        state, alignment.NANOPROBE_CONVERGENCE, 30.0, revision=0,
    )
    strengths = {lens.key: lens.percent for lens in state.lenses
                 if lens.key in alignment.CONDENSER_KEYS}
    solved = alignment.DirectAlignmentResult(
        definition.key, True, 30.0, 30.0, "mrad", 0.0, "mm", strengths,
        1, 0.05, 0.0, "Synthetic successful solver result",
    )
    monkeypatch.setattr(alignment, "_solve_direct_alignment", lambda *_args, **_kwargs: solved)
    monkeypatch.setattr("temsim.physics.simulation.run",
                        lambda *_args, **_kwargs: SimpleNamespace(incident=object()))
    monkeypatch.setattr("temsim.physics.beam_statistics.branch_sample_statistics", lambda _: object())
    monkeypatch.setattr(alignment, "_condenser_measurement", lambda *_args: _measurement(None, waist=waist))
    def final_topology(candidate, *_args):
        assert candidate.nanoprobe_range == nanoprobe_range
        return dict(status=status, passed=status == "PASS", message=f"Final topology evidence: {status}")
    monkeypatch.setattr(transaction, "_validate_condenser_topology_forward",
                        final_topology)

    def checkpoint(cls, result, **kwargs):
        return cls(result.calculation_manifest.instrument_snapshot, {}, state.sample.z_mm,
                   "transaction fixture", {"validation_status": "NOT_RUN"})

    monkeypatch.setattr(WorkingPointCheckpoint, "from_result", classmethod(checkpoint))
    candidate = transaction.solve_alignment_candidate(request)

    assert candidate.result.success is (status == "PASS" and waist == 0.)
    assert candidate.validation["condenser_topology"]["status"] == status
    assert f"Final topology evidence: {status}" in candidate.result.message
    assert capture_instrument_snapshot(state).digest == before
    if status != "PASS" or waist != 0.:
        with pytest.raises(ValueError, match="complete"):
            transaction.AlignmentCommitGate().apply(state, candidate, revision=0)


@pytest.mark.parametrize("nanoprobe_range", ("normal", "large_angle"))
@pytest.mark.parametrize("status", ("PASS", "FAIL", "UNRESOLVED", None))
def test_joint_alignment_cannot_omit_or_bypass_nanoprobe_topology(
    physical_state, monkeypatch, status, nanoprobe_range,
):
    from temsim import alignment_transaction as transaction
    from temsim.alignment_constraints import ConstrainedAlignment
    from temsim.instrument_snapshot import capture_instrument_snapshot

    state = physical_state
    state.nanoprobe_range = nanoprobe_range
    before = capture_instrument_snapshot(state).digest
    definition = _definition()
    monkeypatch.setattr(transaction, "direct_alignment_by_key", lambda _key: definition)
    monkeypatch.setattr("temsim.operating_modes.direct_alignment_by_key", lambda *_args: definition)
    initial = {lens.key: lens.percent for lens in state.lenses}
    refinement_requests = []

    def evaluate(scratch, **kwargs):
        assert scratch.nanoprobe_range == nanoprobe_range
        controls = {lens.key: lens.percent for lens in scratch.lenses}
        metrics = dict(alpha95_mrad=30.0 + 0.1 * (controls[alignment.CONDENSER_KEYS[0]]
                       - initial[alignment.CONDENSER_KEYS[0]]),
            alpha99_mrad=31.0, diameter95_um=0.001, current_pa=10.0,
            centroid_x_um=0.0, centroid_y_um=0.0, curvature_per_m=0.0,
            waist_offset_mm=1.0e-4 * (controls[alignment.CONDENSER_KEYS[1]]
                             - initial[alignment.CONDENSER_KEYS[1]]), effective_samples=32.0)
        arrays = dict(x_m=np.zeros(32), y_m=np.zeros(32), tx_rad=np.zeros(32),
                      ty_rad=np.zeros(32), weight=np.ones(32) / 32, alive=np.ones(32, bool))
        summary = dict(plane_z_mm=scratch.sample.upper_surface_z_mm, source_current_a=1.0e-11)
        if kwargs.get("condenser_topology"):
            refinement_requests.append(True)
            if status is not None:
                summary["condenser_topology"] = dict(status=status, message=f"Joint topology: {status}")
        return metrics, arrays, summary, None

    monkeypatch.setattr("temsim.alignment_constraints.evaluate_incident", evaluate)
    options = ConstrainedAlignment((), {key: (0.0, 100.0) for key in alignment.CONDENSER_KEYS},
                                   maximum_evaluations=8)
    request = transaction.AlignmentRequest.capture(
        state, alignment.NANOPROBE_CONVERGENCE, 30.0, revision=0, options=options,
    )
    candidate = transaction.solve_alignment_candidate(request)

    assert refinement_requests == [True, True, True]
    assert candidate.result.success is (status == "PASS")
    assert all(row["passed"] is (status == "PASS") for row in candidate.validation["refinements"])
    assert capture_instrument_snapshot(state).digest == before
