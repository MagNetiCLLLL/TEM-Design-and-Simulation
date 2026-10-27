"""Opt-in carrier wiring retains old physical and numerical admission gates."""
from dataclasses import replace

import numpy as np
import pytest

from temsim.physics import potential_wave_carrier as carrier
from temsim.physics import potential_wave_device as device
from temsim.physics.multiplane_wave import PlaneWave
from temsim.physics.wave_grid import WaveGridNumerics
from test_potential_wave_carrier import _wave, _snapshot, _round, _multipole, ENERGY_EV


@pytest.mark.parametrize("options", [dict(residual_exponential_method="unknown"),
    dict(residual_exponential_method=True), dict(residual_exponential_method=None),
    dict(maximum_chebyshev_group=True), dict(maximum_chebyshev_group=0),
    dict(maximum_chebyshev_group=33), dict(maximum_chebyshev_group=8.)])
def test_public_residual_method_requires_explicit_known_values(options):
    with pytest.raises(ValueError, match="residual_exponential_method|maximum_chebyshev_group"):
        carrier.propagate_carrier_potential(_snapshot(), _wave(), (0., 1e-7),
            energy_ev=ENERGY_EV, **options)


def test_default_taylor_is_bitwise_equal_to_explicit_taylor():
    snapshot = _snapshot(_round(), _multipole())
    wave = _wave(n=64, curvature=.2, tilt=(2e-7, -1e-7))
    args = (snapshot, wave, (0., 1e-6, 2e-6))
    options = dict(energy_ev=ENERGY_EV, kinetic_order=8,
                   scalar_splitting=True, higher_order_scalar_splitting=True)
    old, a = carrier.propagate_carrier_potential(*args, **options)
    explicit, b = carrier.propagate_carrier_potential(*args, **options,
        residual_exponential_method="taylor", maximum_chebyshev_group=8)
    for name in ("amplitude", "basis_m", "origin_m", "curvature_m1", "tilt_rad"):
        np.testing.assert_array_equal(getattr(old, name), getattr(explicit, name))
    assert a == b
    assert a["residual_exponential_method"] == "taylor"
    assert a["residual_exponential_methods"] == ["taylor"]


def test_public_path_passes_exact_bounds_and_host_device_budgets(monkeypatch):
    calls = []
    execute = device.execute_residual_exponential
    def inspect(*args, **options):
        calls.append(dict(options))
        return execute(*args, **options)
    monkeypatch.setattr(device, "execute_residual_exponential", inspect)
    # The carrier reserves cold field-provider preparation separately from
    # the recurrence's per-cell workspace.
    numerics = WaveGridNumerics(maximum_working_bytes=80*1024**2,
        maximum_device_working_bytes=48*1024**2)
    result, record = carrier.propagate_carrier_potential(_snapshot(_round()),
        _wave(n=48), (0., 1e-6), energy_ev=ENERGY_EV, kinetic_order=8,
        scalar_splitting=True, higher_order_scalar_splitting=True,
        residual_exponential_method="chebyshev", maximum_chebyshev_group=4,
        grid_numerics=numerics)
    assert record["residual_exponential_method"] == "chebyshev"
    assert record["maximum_chebyshev_group"] == 4
    assert len(calls) == len(record["residual_steps"]) == 2
    for call, row in zip(calls, record["residual_steps"]):
        assert call["exponential_method"] == "chebyshev"
        assert call["generator_norm_bound"] == row["generator_norm_bound"]
        assert call["series_error_budget"] == row["declared_series_error_budget"]
        assert row["series_error_bound"] <= call["series_error_budget"]
        assert 0 < call["maximum_host_working_bytes"] <= numerics.maximum_working_bytes
        assert call["maximum_device_working_bytes"] == numerics.maximum_device_working_bytes
        assert call["maximum_chebyshev_group"] == 4
        assert row["operator_applications"] == call["count"]*call["degree"]
    assert result.probability == pytest.approx(1., abs=2e-13)
    assert calls[0]["maximum_host_working_bytes"] == numerics.maximum_working_bytes
    assert calls[1]["maximum_host_working_bytes"] < numerics.maximum_working_bytes


def _periodic_residual(monkeypatch):
    # Synthetic finite-period operator, not a physical beam-domain fixture.
    # Its complete kinetic norm remains inside the unchanged small-angle gate.
    rng = np.random.default_rng(871)
    shape = (7, 9)
    amplitude = rng.normal(size=shape)+1j*rng.normal(size=shape)
    amplitude /= np.linalg.norm(amplitude)
    fy, fx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing="ij")
    frequency = .1*np.stack((fx, fy))
    velocity = np.stack((np.full(shape, .01), np.full(shape, -.003)))
    mechanical = np.stack((np.full(shape, .03), np.full(shape, .01)))
    scalar = np.zeros(shape)
    defect = rng.normal(scale=1e-5, size=shape)
    monkeypatch.setattr(carrier, "_coefficients", lambda *a, **k:
        (velocity, scalar, mechanical, frequency, defect, .91))
    wave = PlaneWave(amplitude, np.eye(2), np.zeros(2))
    args = (None, wave, 0., 40000., None, 2*np.pi, 8, 1e-12, 1., .99, .99, lambda: False)
    options = dict(scalar_splitting=True, higher_order_scalar_splitting=True,
                   maximum_host_working_bytes=2**26, maximum_device_working_bytes=2**26)
    return args, options


def test_actual_chebyshev_reduction_keeps_original_taylor_flux_and_kinetic_allowance(monkeypatch):
    args, options = _periodic_residual(monkeypatch)
    old, a = carrier._residual(*args, **options)
    new, b = carrier._residual(*args, **options, residual_exponential_method="chebyshev")
    assert b["residual_execution"]["exponential_method"] == "chebyshev"
    assert b["actual_operator_applications"] < b["operator_applications"]
    for key in ("operator_applications", "roundoff_relative_allowance", "original_taylor_series_error_bound",
                "kinetic_remainder_phase_bound", "square_root_expansion_bound"):
        assert b[key] == a[key]
    assert b["series_error_bound"] == b["residual_execution"]["chebyshev_certificate"]["series_error_bound"]
    assert np.linalg.norm(new.amplitude-old.amplitude) < 3e-13


def test_chebyshev_roundoff_diagnostic_cannot_relax_original_flux_guard(monkeypatch):
    args, options = _periodic_residual(monkeypatch)
    execute = device.execute_residual_exponential
    def inflate(*a, **k):
        output, record = execute(*a, **k)
        record["chebyshev_certificate"]["recurrence_roundoff_amplification"] = 1e100
        return output*(1+1e-6), record
    monkeypatch.setattr(device, "execute_residual_exponential", inflate)
    with pytest.raises(ValueError, match="flux accounting"):
        carrier._residual(*args, **options, residual_exponential_method="chebyshev")
