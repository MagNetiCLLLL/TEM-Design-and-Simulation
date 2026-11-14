"""Pure-array and mocked-driver checks for isolated ideal-lens research.

No test transports a finite source through an electromagnetic field. The
source fixtures exercise sampling only; the integration tests replace SciPy's
solver and the magnetic acceleration operator explicitly.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import OdeSolution
from scipy.integrate._ivp.rk import Dop853DenseOutput
from scipy.stats import qmc

from scripts import ideal_lens_beam as beam
from temsim.optics.aberration_basis import CARTESIAN_NAMES, predict_displacement_m


def test_source_growth_preserves_complete_prefix_without_refitting():
    matrix = np.array([[2., -3.], [3., 2.]])
    original = beam.source_population(1024, .00095, matrix, 300000.)
    grown = beam.source_population(2048, .00095, matrix, 300000.)
    for key, value in original.items():
        np.testing.assert_array_equal(value, grown[key][:1024], err_msg=key)
    np.testing.assert_array_equal(grown["source_id"], np.arange(2048))
    assert np.unique(grown["initial_phase"][:, :2], axis=0).shape == (2048, 2)
    assert np.all(np.linalg.norm(grown["initial_phase"][:, :2], axis=1) < .00095)
    np.testing.assert_allclose(grown["initial_phase"][:, 2:4], grown["initial_phase"][:, :2] @ matrix.T)
    np.testing.assert_array_equal(grown["kinetic_energy_ev"], np.full(2048, 300000.))
    np.testing.assert_array_equal(grown["emission_time_s"], np.zeros(2048))
    np.testing.assert_array_equal(grown["initial_phase"][:, 4], grown["emission_time_s"])
    np.testing.assert_allclose(np.linalg.norm(grown["emission_direction"], axis=1), 1., atol=2e-16)
    np.testing.assert_allclose(grown["emission_direction"][:, :2] / grown["emission_direction"][:, 2, None],
                               grown["initial_phase"][:, 2:4], atol=2e-18)


def test_uniform_area_law_and_predeclared_independent_holdout_coordinate():
    source = beam.source_population(1024, .00095, np.zeros((2, 2)))
    # Independent library implementation; omit the all-zero Halton origin.
    reference = qmc.Halton(3, scramble=False).random(1025)[1:]
    radius = .00095*np.sqrt(reference[:, 0])
    angle = 2*np.pi*reference[:, 1]
    np.testing.assert_allclose(source["initial_phase"][:, :2],
                               np.column_stack((radius*np.cos(angle), radius*np.sin(angle))), atol=2e-18)
    np.testing.assert_array_equal(source["training"], reference[:, 2] < .5)
    # The split is neither a radius cut nor a half-pupil/angular split.
    for selected in (source["training"], ~source["training"]):
        assert 480 < np.count_nonzero(selected) < 545
        assert np.min(radius[selected]) < .0001
        assert np.max(radius[selected]) > .00094
        for quadrant in range(4):
            assert np.count_nonzero(selected & (angle >= quadrant*np.pi/2) & (angle < (quadrant+1)*np.pi/2)) > 100


@pytest.mark.parametrize("count", [0, 193, 999, 1024., True])
def test_source_rejects_subminimum_or_noninteger_count(count):
    with pytest.raises(ValueError, match="1000"):
        beam.source_population(count, .00095, np.zeros((2, 2)))


@pytest.mark.parametrize("radius", [0., -1., np.nan, np.inf])
def test_source_rejects_nonphysical_radius(radius):
    with pytest.raises(ValueError):
        beam.source_population(1024, radius, np.zeros((2, 2)))


@pytest.mark.parametrize("matrix", [np.zeros((1, 2)), np.zeros((2, 3)), [[np.nan, 0], [0, 0]]])
def test_source_rejects_invalid_slope_matrix(matrix):
    with pytest.raises(ValueError):
        beam.source_population(1024, .00095, matrix)


@pytest.mark.parametrize("energy", [0., -1., np.nan, np.inf])
def test_source_rejects_nonphysical_energy(energy):
    with pytest.raises(ValueError):
        beam.source_population(1024, .00095, np.zeros((2, 2)), energy)


def test_position_spread_does_not_create_a_convergence_angle():
    phase = np.zeros((4, 5))
    phase[:, 0] = [-2e-9, -2e-9, 2e-9, 2e-9]
    result = beam.phase_statistics(phase, np.ones(4, bool), np.full(4, .25), 1e-6)
    assert result["d95_nm"] == pytest.approx(4.)
    assert result["rms_radius_nm"] == pytest.approx(2.)
    assert result["alpha95_mrad"] == 0.
    assert result["angular_rms_mrad"] == 0.


def test_pure_slope_spread_uses_true_angle_and_does_not_create_probe_size():
    phase = np.zeros((4, 5))
    phase[:, 3] = [-.03, -.03, .03, .03]
    result = beam.phase_statistics(phase, np.ones(4, bool), np.full(4, .25), 1e-6)
    assert result["d95_nm"] == 0.
    assert result["rms_radius_nm"] == 0.
    assert result["alpha95_mrad"] == pytest.approx(np.arctan(.03)*1e3)
    assert result["angular_rms_mrad"] == pytest.approx(np.arctan(.03)*1e3)


def test_interception_preserves_absolute_weight_and_effective_population():
    phase = np.zeros((4, 5))
    phase[:, 0] = [1e-9, 3e-9, 100., -100.]
    weights = np.array([.1, .2, .3, .4])
    result = beam.phase_statistics(phase, [True, True, False, False], weights, 2e-6)
    assert result["surviving_positive"] == 2
    assert result["transmission"] == pytest.approx(.3)
    assert result["current_a"] == pytest.approx(.6e-6)
    assert result["effective"] == pytest.approx(.3**2/(.1**2+.2**2))
    assert result["centroid_m"] == pytest.approx([7e-9/3, 0.])
    assert result["d95_nm"] == pytest.approx(8/3)
    np.testing.assert_array_equal(weights, [.1, .2, .3, .4])
    empty = beam.phase_statistics(phase, np.zeros(4, bool), weights, 2e-6)
    assert empty["status"] == "NO_TRANSMISSION"
    assert empty["surviving_positive"] == 0 and empty["current_a"] == 0.


def test_gaussian_field_si_units_symmetry_and_nonzero_far_tail():
    peak, sigma = .3, .006
    field = beam.gaussian_field(peak, sigma, order=9, valid_radius_m=.002)
    z = sigma*np.array([-9., -1., 0., 1., 9.])
    points = np.column_stack((np.zeros(5), np.zeros(5), z))
    values = field.sample(points)
    np.testing.assert_allclose(values["B_t"][:, 2], peak*np.exp(-.5*(z/sigma)**2), rtol=2e-15)
    np.testing.assert_array_equal(values["B_t"][:, :2], np.zeros((5, 2)))
    assert values["B_t"][-1, 2] > 0.  # All-z Gaussian, not an 8-sigma cutoff.
    pair = field.sample([[.0004, 0., -sigma], [.0004, 0., sigma]])["B_t"]
    assert pair[0, 0] == pytest.approx(-pair[1, 0])
    assert pair[0, 2] == pytest.approx(pair[1, 2])
    wider = beam.gaussian_field(peak, sigma, order=9, valid_radius_m=.003)
    np.testing.assert_array_equal(wider.sample(points)["B_t"], values["B_t"])
    with pytest.raises(ValueError, match="domain"):
        field.sample([[.00201, 0., 0.]])


def test_spherical_fit_units_sign_and_independent_holdout():
    source = beam.source_population(1024, .00095, np.zeros((2, 2)))
    outward = -np.eye(2)*(0.02/.00095)
    pupil = source["initial_phase"][:, :2] @ outward.T
    coefficients = np.zeros(len(CARTESIAN_NAMES))
    coefficients[CARTESIAN_NAMES.index("C3")] = 3.7
    image = np.zeros((1024, 5))
    image[:, :2] = predict_displacement_m(pupil, coefficients)
    result = beam.spherical_fit(source["initial_phase"], image, source["training"], outward, .02, 3.7)
    assert result["cs_fit_mm"] == pytest.approx(3.7, abs=1e-10)
    assert result["holdout_rms_m"] < 1e-18
    assert result["cubic_edge_transverse_error_nm"] == pytest.approx(3.7e-3*.02**3*1e9)
    assert result["paraxial_edge_angle_mrad"] == pytest.approx(np.arctan(.02)*1e3)
    assert result["higher_order_departure_max_nm"] < 1e-10
    for fit in result["radial_geometric_fits"]:
        assert fit["cs_mm"] == pytest.approx(3.7, abs=1e-10)
    perturbed = image.copy()
    perturbed[~source["training"], 0] += 1e-8
    held = beam.spherical_fit(source["initial_phase"], perturbed, source["training"], outward, .02, 3.7)
    np.testing.assert_array_equal(held["coefficients_mm"], result["coefficients_mm"])
    assert held["holdout_rms_m"] == pytest.approx(1e-8/np.sqrt(2))
    assert held["holdout_max_m"] == pytest.approx(1e-8)


def test_mocked_transport_splits_tail_keeps_five_state_columns_and_clock(monkeypatch):
    calls, operator_calls = [], []
    initial = beam.source_population(1024, .00095, np.zeros((2, 2)))["initial_phase"]
    initial[:, 2] = .03
    initial[:, 3] = -.04
    initial[:, 4] = np.linspace(0., 1e-12, 1024)

    def operator(fields, xyz, slopes, charge_over_p):
        operator_calls.append((xyz.copy(), slopes.copy(), charge_over_p))
        return np.zeros_like(slopes)

    def solver(rhs, span, state, **kwargs):
        calls.append((span, state.copy(), kwargs))
        derivative = rhs(span[0], state)
        stages = np.zeros((7, len(state)))
        stages[0] = (span[1]-span[0])*derivative
        local = Dop853DenseOutput(*span, state, stages)
        final = local(span[1])
        return SimpleNamespace(success=True, message="Mock linear array operator", nfev=1,
            t=np.array(span), y=np.column_stack((state, final)),
            sol=OdeSolution(span, [local]))

    monkeypatch.setattr(beam, "solve_ivp", solver)
    monkeypatch.setattr(beam, "magnetic_slope_derivative", operator)
    solution, receipt = beam.propagate(SimpleNamespace(digest="mock-field"), initial,
        energy_ev=300000., start_m=-.008, end_m=.012, sigma_m=.001)
    assert [call[0] for call in calls] == [(-.008, .008), (.008, .012)]
    assert [call[2]["max_step"] for call in calls] == [.001/16, .004]
    assert [segment["far_tail"] for segment in receipt["segments"]] == [False, True]
    np.testing.assert_array_equal(solution(-.008).reshape(1024, 5), initial)
    final = solution(.012).reshape(1024, 5)
    np.testing.assert_allclose(final[:, :2], initial[:, :2]+.02*initial[:, 2:4], atol=2e-18)
    np.testing.assert_array_equal(final[:, 2:4], initial[:, 2:4])
    np.testing.assert_allclose(final[:, 4]-initial[:, 4],
        .02*np.sqrt(1+.03**2+.04**2)/receipt["constant_speed_m_per_s"], rtol=3e-15)
    assert all(call[2] < 0. for call in operator_calls)
    assert receipt["field_digest"] == "mock-field"
    assert receipt["counterfactual_after_contact"] is True


def test_transport_rejects_small_partition_before_any_solver_call(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The solver must not run for a subminimum source")
    monkeypatch.setattr(beam, "solve_ivp", forbidden)
    with pytest.raises(ValueError, match="1000"):
        beam.propagate(None, np.zeros((999, 5)), energy_ev=300000.,
                       start_m=0., end_m=1., sigma_m=.006)


def test_array_hash_distinguishes_shape_dtype_and_content():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert beam.array_hash(a) == beam.array_hash(a.copy(order="F"))
    assert len({beam.array_hash(a), beam.array_hash(a.ravel()),
                beam.array_hash(a.astype(np.float32)), beam.array_hash(a+1)}) == 4
