"""Reuse only admitted scalar samples and exact compact exterior zeros."""
from dataclasses import replace

import numpy as np
import pytest

import temsim.physics.potential_wave_carrier as carrier
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from temsim.physics.dynamic_potential import DynamicDipolePotential, DynamicPotentialSnapshot
from temsim.physics.electric_potential_snapshot import ElectricPotentialSnapshot
from temsim.physics.electric_wave_reference import ElectrostaticWaveReference, ElectricWaveHistory
from temsim.physics.potential_wave_observation import observe_potential_mode
from temsim.physics.retarded_wave import RetardedPotentialSlice, initial_dynamic_state
from temsim.physics.wave_grid import WaveGridNumerics
from test_electric_wave_reference import _electric, _mode
from test_potential_wave_carrier import _wave
from test_potential_wave_coefficient_memory import _case
from test_retarded_wave import _source


def _dynamic_source(*, radial_support=.005):
    static = _electric(radial_coefficient=2e8, length=.6, axial_rise=1000.)
    old = _source(offset=1e-5).dynamic_coils[0]
    native = replace(old.base.base.field, radial_support_m=radial_support)
    coil = DynamicDipolePotential(AnalyticPotentialDomain(AnalyticMagneticPotential(native)),
        old.owner_ids, old.drives, old.reference_time_s)
    return DynamicPotentialSnapshot(static, (coil,))


def _slice(lower=.46, upper=.46002, *, radial_support=.005):
    source = _dynamic_source(radial_support=radial_support)
    history = ElectricWaveHistory(ElectrostaticWaveReference.capture(source, lower, 300000.),
        dynamic_state=initial_dynamic_state(.1, 3e-9))
    return RetardedPotentialSlice(source, history, lower, upper)


def _legacy_retarded_sample(self, positions_m):
    """Independent frozen pre-optimization dispatch for the equivalence tests."""
    points = np.asarray(positions_m, dtype=float)
    result = self._static_sampler.sample(points)
    times = self.absolute_time(points[..., 2])
    for coil in self.source_snapshot.dynamic_coils:
        if self._side(coil) == "outside":
            coil.base.sample(points)
            continue
        sampled = coil.sample(points, times)
        for key in ("A_t_m", "B_t", "E_v_per_m"):
            result[key] += sampled[key]
        del sampled
    return result


@pytest.mark.parametrize("kind", ["electric", "retarded_active_electric", "retarded_outside_electric"])
def test_captured_phi_defect_is_bitwise_standalone_and_avoids_a_full_grid_resample(kind, monkeypatch):
    if kind == "retarded_outside_electric":
        snapshot, wave, z = _slice(), _wave(n=64), .46001
        reference = snapshot
    else:
        snapshot, wave, z, quadratic, _, _ = _case(kind, n=64)
        reference = quadratic.electric_reference
    points = carrier._points(wave, z)
    sampled = snapshot.sample(points)["phi_v"]
    before = sampled.tobytes()
    original, queries = ElectricPotentialSnapshot.sample, []
    def inspect(self, positions):
        queries.append(np.shape(positions))
        return original(self, positions)
    monkeypatch.setattr(ElectricPotentialSnapshot, "sample", inspect)
    expected = reference.squared_momentum_defect(snapshot, points, z, 300000.)
    assert sum(shape == points.shape for shape in queries) == 1
    queries.clear()
    actual = reference._squared_momentum_defect_from_phi(snapshot, sampled, z, 300000.)
    assert not any(shape == points.shape for shape in queries)
    assert queries  # Original axis/reference energy validation still executes.
    assert actual.tobytes() == expected.tobytes()
    assert sampled.tobytes() == before


@pytest.mark.parametrize("retarded", [False, True])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, 1j, True, "voltage"])
def test_private_captured_phi_rejects_nonfinite_or_nonreal_values(retarded, invalid):
    snapshot = _slice() if retarded else _electric()
    reference = snapshot if retarded else ElectrostaticWaveReference.capture(snapshot, 0., 300000.)
    z = .46001 if retarded else 1e-5
    with pytest.raises(ValueError, match="finite real voltages"):
        reference._squared_momentum_defect_from_phi(snapshot, np.array([invalid]), z, 300000.)


@pytest.mark.parametrize("retarded", [False, True])
def test_captured_phi_retains_source_identity_and_turning_region_rejection(retarded):
    snapshot = _slice() if retarded else _electric()
    reference = snapshot if retarded else ElectrostaticWaveReference.capture(snapshot, 0., 300000.)
    z = .46001 if retarded else 1e-5
    changed = _slice() if retarded else _electric(offset=1.)
    with pytest.raises(ValueError, match="changed|different executed clock"):
        reference._squared_momentum_defect_from_phi(changed, np.array([0.]), z, 300000.)
    with pytest.raises(ValueError, match="turning"):
        reference._squared_momentum_defect_from_phi(snapshot, np.array([-1e9]), z, 300000.)
    with pytest.raises(ValueError, match="domain"):
        reference.squared_momentum_defect(snapshot, np.array([[1., 0., z]]), z, 300000.)


@pytest.mark.parametrize("case", ["outside", "face", "active"])
def test_retarded_outside_shortcut_retains_exact_values_and_face_fallback(case, monkeypatch):
    bounds = {"outside": (.46, .46002), "face": (.452, .45202), "active": (.45001, .45003)}
    view = _slice(*bounds[case])
    z = bounds[case][0] if case == "face" else sum(bounds[case])/2
    points = carrier._points(_wave(n=64), z)
    expected = _legacy_retarded_sample(view, points)
    base = view.source_snapshot.dynamic_coils[0].base
    original, calls = AnalyticPotentialDomain.sample, []
    def inspect(self, coordinates):
        if self is base:
            calls.append(np.shape(coordinates))
        return original(self, coordinates)
    monkeypatch.setattr(AnalyticPotentialDomain, "sample", inspect)
    actual = view.sample(points)
    for name in expected:
        assert actual[name].tobytes() == expected[name].tobytes()
    assert len(calls) == (1 if case == "face" else 0)


@pytest.mark.parametrize("invalid", ["radial_face", "nonfinite", "outside_clock"])
def test_retarded_shortcut_preserves_invalid_domain_rejections(invalid):
    view = _slice(.452, .45202, radial_support=5e-5)
    points = np.array([[0., 0., .452]])
    if invalid == "radial_face":
        points[0, 0] = 1e-4
    elif invalid == "nonfinite":
        points[0, 0] = np.nan
    else:
        points[0, 2] = .46
    for sample in (view.sample, lambda p: _legacy_retarded_sample(view, p)):
        with pytest.raises(ValueError):
            sample(points)


@pytest.mark.parametrize("kind", ["static", "dynamic_active", "dynamic_outside"])
def test_complete_wave_and_energy_clock_histories_are_bitwise_unchanged(kind, monkeypatch):
    snapshot = (_electric(radial_coefficient=2e8, length=.6, axial_rise=1000.)
                if kind == "static" else _dynamic_source())
    lower = .45001 if kind == "dynamic_active" else .46
    upper = lower+2e-5
    mode = _mode()
    options = dict(maximum_step_m=1e-5, kinetic_order=8,
        scalar_splitting=True, higher_order_scalar_splitting=True,
        grid_numerics=WaveGridNumerics(maximum_pixels=256, maximum_working_bytes=128*1024**2))
    if kind != "static":
        options["emission_time_s"] = .1
    actual, = observe_potential_mode(snapshot, mode, lower, (upper,), **options)
    raw_points, current = carrier._points, {}
    def points(wave, z):
        value = raw_points(wave, z)
        current["points"] = value
        return value
    def standalone(self, source, sampled_phi, z, energy):
        assert current["points"].shape[:-1] == sampled_phi.shape
        assert np.all(current["points"][..., 2] == z)
        return self.squared_momentum_defect(source, current["points"], z, energy)
    with monkeypatch.context() as patch:
        patch.setattr(carrier, "_points", points)
        patch.setattr(ElectrostaticWaveReference, "_squared_momentum_defect_from_phi", standalone)
        patch.setattr(RetardedPotentialSlice, "_squared_momentum_defect_from_phi", standalone)
        patch.setattr(RetardedPotentialSlice, "sample", _legacy_retarded_sample)
        expected, = observe_potential_mode(snapshot, mode, lower, (upper,), **options)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        assert getattr(actual.mode.plane, name).tobytes() == getattr(expected.mode.plane, name).tobytes()
    assert actual.mode.weight_per_reference_electron == expected.mode.weight_per_reference_electron
    assert actual.mode.energy_kev == expected.mode.energy_kev
    assert actual.mode.axial_reference == expected.mode.axial_reference
    assert actual.metadata == expected.metadata


@pytest.mark.parametrize("known", [True, False])
def test_coefficients_reuse_phi_only_from_a_known_snapshot_bound_sampler(known, monkeypatch):
    from temsim.physics.unified_potential_sampler import compile_unified_potential_sampler
    arguments = _case("electric", n=64)
    snapshot = arguments[0]
    expected = carrier._coefficients(*arguments, return_energy_terms=True)

    class UnknownSampler:
        def sample(self, points):
            values = snapshot.sample(points)
            # Its A remains valid, but this unrecognized scalar result must
            # not replace the independently captured electric energy law.
            values["phi_v"] = values["phi_v"]+1234.
            return values

    sampler = compile_unified_potential_sampler(snapshot, prefer_plane=True) if known else UnknownSampler()
    original = ElectrostaticWaveReference._squared_momentum_defect_from_phi
    calls = []

    def inspect(self, *args):
        calls.append(True)
        return original(self, *args)

    monkeypatch.setattr(ElectrostaticWaveReference, "_squared_momentum_defect_from_phi", inspect)
    actual = carrier._coefficients(*arguments[:-1], sampler, return_energy_terms=True)
    assert len(calls) == int(known)
    for first, second in zip(expected, actual):
        assert np.asarray(first).tobytes() == np.asarray(second).tobytes()
