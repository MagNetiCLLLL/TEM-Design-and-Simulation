"""Thin optical diagnostics use the actual event executor, not finite fields."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.operating_modes import direct_alignment_by_key
from temsim.optics.column import default_state
from temsim.optics.direct_alignment import (
    IMAGE_KEYS, IMAGE_MAGNIFICATION, _LiveFirstOrderModel,
    _ProjectorMeasurementModel, canonical_source_basis, canonical_transfers,
)
from temsim.physics.core import build_propagation_plan, execute_propagation_plan
from temsim.physics.first_order import trace_transverse_transfer
from temsim.physics.recording_stop import tem_projection_reference_plane


@pytest.fixture
def state():
    result = default_state()
    result.simulation_mode = "ideal_thin"
    result.acceleration_enabled = False
    result.acceleration_backend = "CPU"
    result.history_step_mm = 20.
    return result


def _executed_jacobian(state, source, target):
    """Central differences from an executed after-action source checkpoint."""
    steps = np.array((3e-9, 5e-9, 4e-7, 7e-7))
    rays = np.column_stack((np.zeros(4), np.diag(steps), -np.diag(steps)))
    plan = build_propagation_plan(state, source, target, checkpoint_z_mm=(target,))
    checkpoint = execute_propagation_plan(
        state, plan, rays[0], rays[2], rays[1], rays[3],
        include_initial_plane_kicks=False)[-1]
    output = np.array([getattr(checkpoint, name)[-1] for name in
                       ("x_m", "y_m", "tx_rad", "ty_rad")])
    return (output[:, 1:5]-output[:, 5:9])/(2*steps), output[:, 0]


def _settings(state):
    result = deepcopy(state.to_dict())
    result.pop("active_backend", None)  # Execution receipts are allowed to update.
    return result


def test_thin_source_basis_has_no_finite_field_gauge_even_at_lens_center(state):
    lens = state.lenses[1]
    np.testing.assert_array_equal(canonical_source_basis(state, lens.z_mm), np.eye(4))
    state.simulation_mode = "ideal"
    finite = canonical_source_basis(state, lens.z_mm)
    assert np.linalg.norm(finite[2:, :2]) > 1.


@pytest.mark.parametrize("kwargs", [
    {"source_z_mm": float("nan")},
    {"position_xy_m": (0., float("inf"))},
    {"momentum_kg_m_s": 0.},
    {"momentum_kg_m_s": float("nan")},
])
def test_thin_identity_basis_still_validates_inputs(state, kwargs):
    options = {"source_z_mm": 650., **kwargs}
    with pytest.raises(ValueError, match="finite"):
        canonical_source_basis(state, **options)


def test_thin_canonical_observer_never_uses_axisymmetric_finite_fast_branch(state, monkeypatch):
    import temsim.optics.direct_alignment as alignment
    source, target = 649., 651.
    expected, affine = _executed_jacobian(state, source, target)

    def forbidden(*args, **kwargs):
        raise AssertionError("The thin observer sampled the finite magnetic field")

    monkeypatch.setattr(alignment, "fields", forbidden)
    observed = canonical_transfers(state, source, (source, 649.5, target))
    np.testing.assert_array_equal(observed[source].matrix, np.eye(4))
    np.testing.assert_allclose(observed[target].matrix, expected, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose((*observed[target].position_offset_m,
                                *observed[target].angle_offset_rad), affine, atol=1e-20)
    explicit = canonical_transfers(state, source, (target,), stable_axisymmetric=False)[target]
    np.testing.assert_array_equal(observed[target].matrix, explicit.matrix)


def test_live_thin_alignment_tracks_candidate_excitation_and_particle_jacobian(state):
    lens = state.lenses[1]
    source, target = lens.z_mm-1., lens.z_mm+1.
    model = _LiveFirstOrderModel(state, source, target, (lens.key,),
                                step_mm=.2, capture_z_mm=(lens.z_mm,))
    assert model.full_field_transfer
    original = float(lens.percent)
    before = _settings(state)
    matrices = []
    for value in (original, original*.8):
        observed = model.matrix((value,))
        a, b = model.canonical_position_blocks((value,))
        np.testing.assert_array_equal(observed[:2, :2], a)
        np.testing.assert_array_equal(observed[:2, 2:], b)
        assert _settings(state) == before
        lens.percent = value
        expected, _ = _executed_jacobian(state, source, target)
        lens.percent = original
        np.testing.assert_allclose(observed, expected, rtol=2e-12, atol=1e-12)
        matrices.append(observed)
    assert np.linalg.norm(matrices[0]-matrices[1]) > 1.
    np.testing.assert_array_equal(model.matrices_at((original,), (target,))[0], matrices[0])


@pytest.mark.parametrize("equivalent_enabled", [False, True])
def test_thin_image_alignment_uses_same_whole_lens_calibration_as_particles(state, equivalent_enabled):
    state.equivalent_image_lenses_enabled = equivalent_enabled
    for detector in state.stem_detectors:
        detector.inserted = False
    before = _settings(state)
    model = _ProjectorMeasurementModel(state, direct_alignment_by_key(IMAGE_MAGNIFICATION), step_mm=1.)
    assert isinstance(model.sample_model, _LiveFirstOrderModel)
    assert model.sample_model.full_field_transfer
    vector = tuple(next(lens.percent for lens in state.lenses if lens.key == key) for key in IMAGE_KEYS)
    expected, _ = _executed_jacobian(state, state.sample.z_mm,
                                    tem_projection_reference_plane(state).z_mm)
    np.testing.assert_allclose(model.sample_model.matrix(vector), expected, rtol=5e-11, atol=1e-8)
    measurement, _ = model.measure(vector)
    assert measurement.value == pytest.approx(np.sqrt(abs(np.linalg.det(expected[:2, :2]))), rel=1e-10)
    assert measurement.constraint_value == pytest.approx(np.linalg.norm(expected[:2, 2:], 2)*1e6, rel=1e-10)
    assert _settings(state) == before


def test_existing_finite_lens_model_keeps_electric_full_field_branch(state):
    state.simulation_mode = "ideal"
    lens = state.lenses[1]
    source, target = lens.z_mm-1., lens.z_mm+1.
    model = _LiveFirstOrderModel(state, source, target, (lens.key,), step_mm=.01)
    assert model.full_field_transfer and model._electric_field is not None
    actual = model.matrix((lens.percent,))
    expected = trace_transverse_transfer(state, source, target, maximum_step_mm=.01).matrix
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)


def test_thin_sample_observer_starts_after_objective_and_has_identity_at_source(state):
    from temsim.physics.first_order import trace_transverse_transfers, transverse_position_response_path
    source = float(state.sample.z_mm)
    target = source + .1
    expected = np.eye(4)
    expected[:2, 2:] = np.eye(2) * (target-source) * 1e-3
    observed = trace_transverse_transfers(state, source, (source, target))
    np.testing.assert_array_equal(observed[source].matrix, np.eye(4))
    np.testing.assert_allclose(observed[target].matrix, expected, rtol=5e-15, atol=2e-20)
    canonical = canonical_transfers(state, source, (source, target))
    np.testing.assert_allclose(canonical[target].matrix, expected, rtol=5e-15, atol=2e-20)
    z, position = transverse_position_response_path(state, source, target, save_z_mm=(source, target))
    np.testing.assert_array_equal(position[0], np.zeros((2, 2)))
    np.testing.assert_allclose(position[-1], expected[:2, 2:], rtol=5e-15, atol=2e-20)


def test_live_thin_rays_start_from_same_after_action_reference_as_matrix(state):
    source, target = state.sample.z_mm, state.sample.z_mm + .1
    lens = state.objective_lens
    model = _LiveFirstOrderModel(state, source, target, (lens.key,), step_mm=.2)
    rays = np.array(((2e-8, -3e-8), (4e-8, 1e-8), (.001, -.002), (.003, .004)))
    result = model.rays_at((lens.percent,), rays, (source, target))
    expected = rays.copy()
    expected[:2] += (target-source)*1e-3*rays[2:]
    np.testing.assert_allclose(result[0], rays, rtol=5e-8, atol=1e-14)
    np.testing.assert_allclose(result[-1], expected, rtol=5e-8, atol=1e-14)
    # Source-plane strength has already acted on these supplied coordinates.
    changed = model.rays_at((lens.percent*.5,), rays, (target,))
    np.testing.assert_array_equal(changed[0], result[-1])
