"""Analytic phase/current checks for the round-column reduction."""
from dataclasses import replace
import math
import numpy as np
import pytest

from temsim.physics.radial_column_wave import RadialWave, propagate_radial, radial_cs, refine_radial
from temsim.physics.canonical_action import CanonicalPath


def gaussian():
    # Both reciprocal endpoints must be negligible, independently of the
    # phase/current test. FFTLog is periodic in log radius.
    radius = np.geomspace(1e-10, 100., 32768, endpoint=False)
    return RadialWave(radius, np.exp(-radius**2/2)/np.sqrt(np.pi))


@pytest.mark.parametrize("distance", (.2, -.2, 2.))
def test_complex_free_gaussian_has_absolute_phase_and_conserved_current(distance):
    wave = gaussian()
    m = np.eye(4); m[:2, 2:] = np.eye(2)*distance
    path = CanonicalPath(1.)
    path.append(m)
    result = propagate_radial(wave, m, 2*np.pi, reference_phase_rad=path.reference_phase_rad,
                              reference_length_m=1.)
    exact = np.exp(-result.radius_m**2/(2*(1+1j*distance)))/(np.sqrt(np.pi)*(1+1j*distance))
    # Global weighted error; no fitted phase or output normalisation.
    full = result.amplitude*np.exp(.5j*result.curvature_m1*result.radius_m**2)
    error = np.sqrt(2*np.pi*result.dln*np.sum(abs(result.radius_m*(full-exact))**2))
    assert error < 1e-6
    assert result.probability == pytest.approx(wave.probability, rel=1e-12)


def test_spherical_phase_is_kept_not_fitted_into_quadratic_curvature():
    wave = gaussian()
    result = radial_cs(wave, 2*np.pi, .1)
    np.testing.assert_allclose(result.amplitude, wave.amplitude*np.exp(-.025j*wave.radius_m**4))
    assert result.curvature_m1 == wave.curvature_m1
    assert result.probability == pytest.approx(wave.probability, rel=1e-12)


def test_ideal_round_prefix_uses_each_actual_mode_energy_in_analytic_lens(monkeypatch):
    """Constant B has independent Q_out=-g*tan(g*L), g=qB/(2p)."""
    from types import SimpleNamespace
    from scipy.constants import e
    from temsim.physics import radial_column_wave as radial
    from temsim.physics.tip_gun_wave import _momentum_velocity
    length, field = .001, .2
    plan = SimpleNamespace(z_mm=np.array((0., 1.)), step_m=np.array((length,)),
        mapped_fields=(), signature="constant-B-local-fixture",
        midpoint_magnetic_t=np.array((field,)), cs_kick_m3=np.zeros(2),
        kick_x_rad=np.zeros(2), kick_y_rad=np.zeros(2))
    for name in ("midpoint_sx_m2", "midpoint_sy_m2", "midpoint_sxy_m2",
                 "midpoint_hex_normal_m3", "midpoint_hex_skew_m3"):
        setattr(plan, name, np.zeros(1))
    monkeypatch.setattr("temsim.physics.column_wave._prepare_column",
        lambda *args: (plan, np.full(2, np.inf), {}, ()))
    radius = np.geomspace(1e-12, 1e-4, 32768, endpoint=False)
    sigma = 1e-6
    source = RadialWave(radius, np.exp(-radius**2/(2*sigma*sigma))/(np.sqrt(np.pi)*sigma))
    monkeypatch.setattr(radial, "from_executed_gun", lambda *args: source)
    modes = tuple(SimpleNamespace(mode_id=f"energy-{energy}", energy_kev=energy,
        weight_per_reference_electron=weight) for energy, weight in ((280., .4), (320., .6)))
    checkpoint = SimpleNamespace(plane_z_mm=0., digest="isolated-radial-fixture",
        beam=SimpleNamespace(modes=modes), record={"radial_output_modes":
            tuple({"mode_id": mode.mode_id} for mode in modes)})
    state = SimpleNamespace(simulation_mode="ideal", beam_voltage_kv=300.)
    stop, output, receipt = radial._round_column_prefix(state, checkpoint, 1.)
    assert stop == 1.
    for (mode, wave), original in zip(output, modes):
        assert mode is original
        momentum = float(_momentum_velocity(mode.energy_kev*1000)[0])
        g = -e*field/(2*momentum)
        assert wave.curvature_m1 == pytest.approx(-g*np.tan(g*length), rel=1e-12)
        assert wave.probability == pytest.approx(source.probability, rel=2e-12)
    assert output[0][1].curvature_m1 != output[1][1].curvature_m1
    assert [row["input_fraction"] for row in receipt["modes"]] == [.4, .6]


def test_log_grid_refinement_preserves_complex_wave_not_just_current():
    wave = gaussian()
    finer = refine_radial(wave, 1.5)
    exact = np.exp(-finer.radius_m**2/2)/np.sqrt(np.pi)
    assert finer.probability == pytest.approx(wave.probability, rel=1e-12)
    weighted = np.sqrt(2*np.pi*finer.dln*np.sum(abs(finer.radius_m*(finer.amplitude-exact))**2))
    assert weighted < 1e-8


def test_refinement_replays_hard_stops_from_executed_field_and_replaces_loss_ledger():
    from temsim.physics.radial_column_wave import RadialExecution, _radial_stop
    from temsim.physics.wave_grid import WaveSamplingError
    loaded = []
    def load(samples):
        loaded.append(samples)
        r = np.geomspace(1e-8, 10., samples, endpoint=False)
        return RadialWave(r, np.exp(-r*r/2)/np.sqrt(np.pi))
    execution = RadialExecution(load, 64, 4096)
    execution.apply(lambda w: _radial_stop(w, 1.), metadata={"component": "aperture"})
    old_loss = execution.ledger[0].copy()
    def fine_operator(wave):
        if len(wave.radius_m) < 512:
            raise WaveSamplingError("Independent resolution check", 512/len(wave.radius_m))
        return wave
    result = execution.apply(fine_operator)
    assert len(loaded) == 2
    assert execution.refinements[-1]["kind"] == "upstream_replay"
    # No Fourier ringing past the physical hard stop; no normalization.
    assert np.all(result.amplitude[result.radius_m > 1.] == 0)
    assert len(execution.ledger) == 1
    assert execution.ledger[0]["outgoing"] == result.probability
    assert execution.ledger[0]["outgoing"] != old_loss["outgoing"]
    exact = np.where(result.radius_m <= 1., np.exp(-result.radius_m**2/2)/np.sqrt(np.pi), 0j)
    np.testing.assert_array_equal(result.amplitude, exact)


def test_replay_cannot_silently_exceed_its_declared_budget():
    from temsim.physics.radial_column_wave import RadialExecution
    from temsim.physics.wave_grid import WaveSamplingError
    execution = RadialExecution(lambda samples: gaussian(), 64, 128)
    def unresolved(wave):
        raise WaveSamplingError("Unresolved physical phase", 100.)
    with pytest.raises(ValueError, match="above its 128 budget"):
        execution.apply(unresolved)
    assert not execution.operations


def test_quartic_field_hankel_transform_against_direct_bessel_integral():
    from scipy.special import j0
    from temsim.physics.radial_column_wave import _hankel_grid, _complex_fht
    radius = np.geomspace(1e-10, 100., 524288, endpoint=False)
    wave = RadialWave(radius, np.exp(-radius**2/2-5j*radius**4)/np.sqrt(np.pi))
    offset, frequency = _hankel_grid(wave)
    transformed = _complex_fht(radius*wave.amplitude, wave.dln, offset)/frequency
    requested = np.array([.1, 1., 3., 10., 30., 100.])
    nodes, weights = np.polynomial.legendre.leggauss(8)
    edges = np.linspace(0., 80., 80001)
    half = np.diff(edges)/2
    u = ((edges[:-1]+edges[1:])[:, None]/2+half[:, None]*nodes).ravel()
    integrand = np.exp(-u/2-5j*u*u)/(2*np.sqrt(np.pi))*(half[:, None]*weights).ravel()
    # Independent quadrature in u=r^2. No fitted global phase or normalisation.
    exact = np.array([np.dot(integrand, j0(k*np.sqrt(u))) for k in requested])
    actual = np.interp(np.log(requested), np.log(frequency), transformed)
    np.testing.assert_allclose(actual, exact, atol=2e-8, rtol=0.)


def _angular_lg(order, curvature=0.):
    radius = np.geomspace(1e-10, 100., 32768, endpoint=False)
    amplitude = (radius**abs(order)*np.exp(-radius**2/2+.37j)
                 /math.sqrt(np.pi*math.factorial(abs(order))))
    return RadialWave(radius, amplitude, curvature, angular_order=order)


def _scalar_map(a, b, c, d):
    return np.block([[a*np.eye(2),b*np.eye(2)],[c*np.eye(2),d*np.eye(2)]])


@pytest.mark.parametrize("order", (-2, -1, 1, 2))
@pytest.mark.parametrize("case", ("drift+", "drift-", "fourier+", "fourier-",
                                  "fractional+", "fractional-", "rotation", "negative_A",
                                  "negative_effective_A"))
def test_angular_lg_full_complex_phase_against_analytic_round_canonical_solution(order,case):
    path = CanonicalPath(1.)
    rotation_angle = 0.
    curvature = 0.
    if case.startswith("drift"):
        a,b,c,d = 1.,(.2 if case.endswith('+') else -.2),0.,1.
        path.append(_scalar_map(a,b,c,d))
    elif case.startswith(("fourier","fractional")):
        angle = (np.pi/2 if case.startswith("fourier") else 2.2)*(1 if case.endswith('+') else -1)
        a,b,c,d = np.cos(angle),np.sin(angle),-np.sin(angle),np.cos(angle)
        for _ in range(16):
            step=angle/16
            path.append(_scalar_map(np.cos(step),np.sin(step),-np.sin(step),np.cos(step)))
    elif case == "negative_A":
        for _ in range(16):
            step=np.pi/16
            path.append(_scalar_map(np.cos(step),np.sin(step),-np.sin(step),np.cos(step)))
        path.append(_scalar_map(2.,0.,0.,.5))
        a,b,c,d = -2.,0.,0.,-.5
    elif case == "negative_effective_A":
        # The positive A map with a converging input carrier requires the
        # negative scaled-angular-spectrum chart, including its angular parity.
        path.append(_scalar_map(1.,-.05,0.,1.))
        path.append(_scalar_map(2.,0.,0.,.5))
        path.append(_scalar_map(1.,0.,.5,1.))
        a,b,c,d,curvature = 2.,-.1,1.,.45,30.
    else:
        a,b,c,d = 1.,0.,0.,1.
        rotation_angle = .7
        rotation=np.array(((np.cos(rotation_angle),-np.sin(rotation_angle)),
                           (np.sin(rotation_angle),np.cos(rotation_angle))))
        path.append(np.block([[rotation,np.zeros((2,2))],[np.zeros((2,2)),rotation]]))
    wave = _angular_lg(order,curvature)
    result = propagate_radial(wave,path.matrix,2*np.pi,reference_phase_rad=path.reference_phase_rad,
                              reference_length_m=1.)
    factor = a+b*(curvature+1j)
    precision = (c+d*(curvature+1j))/factor
    branch = np.exp(1j*(path.reference_phase_rad+np.angle(a+1j*b)))
    exact = (branch*np.exp(.37j-1j*order*rotation_angle)*result.radius_m**abs(order)
             /math.sqrt(np.pi*math.factorial(abs(order)))/factor**(abs(order)+1)
             *np.exp(.5j*precision*result.radius_m**2))
    full = result.amplitude*np.exp(.5j*result.curvature_m1*result.radius_m**2)
    # Absolute complex L2, with the prescribed continuous lift. No fitted
    # scalar phase, angular rotation, intensity matching or normalisation.
    error = np.sqrt(2*np.pi*result.dln*np.sum(abs(result.radius_m*(full-exact))**2))
    assert error < 1e-6
    assert result.angular_order == order
    assert result.probability == pytest.approx(wave.probability,rel=1e-12)


def test_angular_order_survives_numerical_grids_aperture_and_complete_cs():
    from temsim.physics.radial_column_wave import expand_radial_domain, _radial_stop
    wave = _angular_lg(-2)
    for changed in (expand_radial_domain(wave), refine_radial(wave,1.25),
                    _radial_stop(wave,1.3), radial_cs(wave,2*np.pi,.1)):
        assert changed.angular_order == -2
    phased = radial_cs(wave,2*np.pi,.1)
    np.testing.assert_allclose(phased.amplitude,wave.amplitude*np.exp(-.025j*wave.radius_m**4))


@pytest.mark.parametrize("order", (True,1.5,float("nan"),"1"))
def test_noninteger_angular_index_is_rejected(order):
    with pytest.raises(ValueError,match="signed integer"):
        replace(gaussian(),angular_order=order)


def test_dense_log_spacing_uses_full_represented_span_without_large_allocation():
    # Expose only the points used by the spacing calculation. This represents
    # a 64M-point uniform grid without allocating a 64M array in a unit test.
    class UniformLogGridEndpoints:
        def __init__(self, count, start, spacing):
            self.count, self.start, self.spacing = count, start, spacing

        def __len__(self):
            return self.count

        def __getitem__(self, index):
            if index == -1:
                index = self.count-1
            if index not in (0,1,self.count-1):
                raise AssertionError("The spacing calculation must use grid endpoints only")
            # Grid constructors evaluate a uniform log coordinate before
            # exponentiation; its first increment is small compared with
            # log(start), reproducing the dense-grid cancellation.
            return math.exp(math.log(self.start)+index*self.spacing)

    count, spacing = 2**26, 1.82862984022e-7
    grid = UniformLogGridEndpoints(count, 1e-9, spacing)
    near = math.log(grid[1]/grid[0])
    assert abs(near/spacing-1.) > 1e-10  # The cancellation exists in this fixture.
    wave = RadialWave(grid, np.empty(0))
    assert wave.dln == pytest.approx(spacing, rel=2e-15)
    padding = 16_000_000
    expanded = UniformLogGridEndpoints(count+2*padding,
        grid[0]*math.exp(-padding*spacing), spacing)
    wider = replace(wave, radius_m=expanded)
    assert wider.dln == pytest.approx(wave.dln, rel=2e-15)
    # Probability carries dln as its quadrature factor: unchanged retained
    # cells must not acquire a spurious norm shift just from zero extension.
    assert wider.dln/wave.dln == pytest.approx(1., abs=2e-15)


def test_full_span_spacing_respects_endpoint_false_interval_count():
    radius = np.geomspace(1e-10,100.,32768,endpoint=False)
    wave = RadialWave(radius,np.empty(0))
    expected = math.log(100./1e-10)/len(radius)
    assert wave.dln == pytest.approx(expected,rel=2e-15)
