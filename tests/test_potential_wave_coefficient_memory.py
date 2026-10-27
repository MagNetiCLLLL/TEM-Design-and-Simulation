"""Coefficient liveness bounds for retained field providers; no physics fit."""
import gc
import tracemalloc
from dataclasses import replace

import numpy as np
import pytest
from scipy.constants import hbar

from temsim.physics.potential_wave_carrier import _coefficients, _reference
from temsim.physics.electric_wave_reference import (
    ElectrostaticWaveReference, ElectricWaveHistory, momentum_velocity,
)
from temsim.physics.potential_plane_sampler import compile_plane_sampler
from temsim.physics.retarded_wave import RetardedPotentialSlice, initial_dynamic_state
from temsim.physics.dynamic_potential import DynamicPotentialSnapshot, DynamicDipolePotential
from temsim.physics.analytic_potential import AnalyticMagneticPotential
from temsim.physics.analytic_potential_domain import AnalyticPotentialDomain
from test_potential_wave_carrier import _wave, _snapshot, _round, _multipole
from test_electric_wave_reference import _electric
from test_retarded_wave import _source


def _case(kind, n=256, active_coils=1):
    wave = _wave(n=n, pitch=5e-8, width=8e-7, curvature=.2, tilt=(2e-7, -1e-7))
    sampler, energy_reference = None, None
    lower, upper = 0., 2e-5
    if kind in ("generic_magnetic", "native_plane_magnetic"):
        snapshot = _snapshot(_round(), _multipole())
        if kind == "native_plane_magnetic":
            sampler = compile_plane_sampler(snapshot)
    elif kind == "electric":
        snapshot = _electric(radial_coefficient=2e8)
        energy_reference = ElectrostaticWaveReference.capture(snapshot, lower, 300000.)
    else:
        static = _electric(radial_coefficient=2e8, length=.6, axial_rise=1000.)
        first = _source().dynamic_coils[0]
        coils = []
        for index in range(active_coils):
            native = replace(first.base.base.field, lens_key=f"independent_coil_{index}")
            coils.append(DynamicDipolePotential(AnalyticPotentialDomain(AnalyticMagneticPotential(native)),
                (f"independent_owner_{index}",), first.drives, first.reference_time_s))
        dynamic = DynamicPotentialSnapshot(static, tuple(coils))
        lower, upper = .45, .45002
        history = ElectricWaveHistory(ElectrostaticWaveReference.capture(dynamic, lower, 300000.),
            dynamic_state=initial_dynamic_state(.1, 3e-9))
        snapshot = RetardedPotentialSlice(dynamic, history, lower, upper)
        energy_reference = snapshot
    momentum, _ = momentum_velocity(300000.)
    middle = (lower+upper)/2
    reference = _reference(snapshot, wave, middle, upper-lower, momentum, "cf4",
        axial_interval=(lower, upper), electric_reference=energy_reference, chart_energy_ev=300000.)
    return snapshot, wave, middle, reference, 2*np.pi*hbar/momentum, sampler


@pytest.mark.parametrize("kind,bound", [
    ("generic_magnetic", 336), ("native_plane_magnetic", 208),
    ("electric", 352), ("retarded_active_electric", 416),
])
def test_coefficient_stage_releases_unused_field_and_geometry_arrays(kind, bound):
    arguments = _case(kind)
    # Provider compilation and immutable caches are outside stage workspace.
    expected = _coefficients(*arguments, return_energy_terms=True)
    gc.collect()
    tracemalloc.start()
    try:
        actual = _coefficients(*arguments, return_energy_terms=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    for first, second in zip(expected, actual):
        np.testing.assert_array_equal(first, second)
    # 512 KiB reserves Python dispatch/allocator and line-sized scratch on
    # this bounded fixture. The live full-grid coefficient arrays dominate.
    assert peak < bound*arguments[1].amplitude.size+512*1024
    assert not arguments[1].amplitude.flags.writeable


def test_generic_and_compiled_coefficients_keep_same_covariant_field_terms():
    arguments = _case("generic_magnetic", n=128)
    generic = _coefficients(*arguments, return_energy_terms=True)
    compiled = _coefficients(*arguments[:-1], compile_plane_sampler(arguments[0]), return_energy_terms=True)
    for expected, actual in zip(generic, compiled):
        np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-20)


@pytest.mark.parametrize("count", [2, 3])
def test_overlapping_active_coils_release_each_sample_and_keep_exact_sum(count):
    arguments = _case("retarded_active_electric", active_coils=count)
    view, wave, z, _, _, _ = arguments
    coordinates = wave.coordinates_m()
    points = np.moveaxis(np.concatenate((coordinates,
        np.full((1, *wave.amplitude.shape), z)), axis=0), 0, -1)
    expected = view._static_sampler.sample(points)
    time = view.absolute_time(points[..., 2])
    for coil in view.source_snapshot.dynamic_coils:
        contribution = coil.sample(points, time)
        for key in ("A_t_m", "B_t", "E_v_per_m"):
            expected[key] += contribution[key]
    actual = view.sample(points)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    del coordinates, points, expected, actual, contribution
    _coefficients(*arguments, return_energy_terms=True)
    gc.collect()
    tracemalloc.start()
    try:
        _coefficients(*arguments, return_energy_terms=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # The bound is independent of how many native coils overlap: only one
    # temporary contribution survives while the final field sum is built.
    assert peak < 416*wave.amplitude.size+512*1024


def test_validation_and_residual_preflight_have_distinct_stage_budgets(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.potential_wave_device import residual_host_working_bytes
    from temsim.physics.wave_grid import WaveGridNumerics, WaveGridBudgetError
    wave = _wave(n=64)
    # Boundary/spectrum diagnostics do not reserve unused coefficient arrays.
    carrier._validate_wave(wave, WaveGridNumerics(maximum_working_bytes=256*wave.amplitude.size), 1e-9, 1e-10)
    def no_coefficients(*args, **kwargs):
        pytest.fail("Field coefficients were allocated before their stage budget")
    monkeypatch.setattr(carrier, "_coefficients", no_coefficients)
    budget = residual_host_working_bytes(wave.amplitude.shape)-1
    with pytest.raises(WaveGridBudgetError, match="working bytes"):
        carrier.propagate_carrier_potential(_snapshot(), wave, (0., 1e-6), energy_ev=200000.,
            kinetic_order=2, grid_numerics=WaveGridNumerics(maximum_working_bytes=budget))


def test_host_stage_estimate_contains_fixed_cold_preparation_allowance():
    from temsim.physics.potential_wave_device import (
        residual_host_working_bytes, RESIDUAL_HOST_PREPARATION_RESERVE,
    )
    assert RESIDUAL_HOST_PREPARATION_RESERVE == 64*1024**2
    for shape in ((32, 35), (256, 256), (7546, 7546)):
        assert residual_host_working_bytes(shape) == 512*np.prod(shape)+64*1024**2


def test_outer_wave_storage_is_charged_once_and_shared_views_are_not_charged_twice():
    from types import SimpleNamespace
    from temsim.cache_memory import estimate_result_cache_bytes
    from temsim.physics.potential_wave_carrier import _additional_wave_bytes
    original = _wave(n=128)
    active = _wave(n=32)
    expected = estimate_result_cache_bytes(original.amplitude)
    assert _additional_wave_bytes(active, original.amplitude, original.amplitude, active.amplitude) == expected
    assert _additional_wave_bytes(original, original.amplitude) == 0
    view = SimpleNamespace(amplitude=original.amplitude[::2, ::2])
    # At most another ndarray owner header is new; the backing pixels are
    # already charged by the active view and must not be counted again.
    assert _additional_wave_bytes(view, original.amplitude) < 512


def test_canonical_internal_padding_and_residual_executor_receive_retained_original_wave(monkeypatch):
    import temsim.physics.potential_wave_carrier as carrier
    from temsim.physics.potential_wave_support import NumericalSupportLedger
    from temsim.physics.wave_grid import WaveGridNumerics
    source = _wave(n=128, width=8e-7)
    numerics = WaveGridNumerics(angular_domain_padding="minimum_full_band")
    canonical_calls, residual_calls, outer_calls = [], [], []
    original_canonical = carrier.propagate_plane_wave
    original_residual = carrier._residual
    original_resolver = carrier.apply_resolved_operator
    def canonical(candidate, *args, **kwargs):
        retained = carrier._additional_wave_bytes(candidate, source.amplitude)
        assert kwargs["retained_bytes"] >= retained > 0
        canonical_calls.append(kwargs["retained_bytes"])
        return original_canonical(candidate, *args, **kwargs)
    def residual(snapshot, candidate, *args, **kwargs):
        retained = carrier._additional_wave_bytes(candidate, source.amplitude)
        assert kwargs["maximum_host_working_bytes"] <= numerics.maximum_working_bytes-retained
        residual_calls.append(retained)
        return original_residual(snapshot, candidate, *args, **kwargs)
    def resolve(candidate, operator, **kwargs):
        if operator.__name__ == "canonical":
            assert kwargs["operator_working_bytes"](candidate.amplitude.shape) == 256*candidate.amplitude.size
            assert kwargs["retained_bytes"] == carrier._additional_wave_bytes(candidate, source.amplitude)
            outer_calls.append(True)
        return original_resolver(candidate, operator, **kwargs)
    monkeypatch.setattr(carrier, "propagate_plane_wave", canonical)
    monkeypatch.setattr(carrier, "_residual", residual)
    monkeypatch.setattr(carrier, "apply_resolved_operator", resolve)
    carrier.propagate_carrier_potential(_snapshot(), source, (0., 1e-7), energy_ev=200000., kinetic_order=2,
        support_ledger=NumericalSupportLedger(.02, 1e-6), support_mode_weight=.25,
        support_projection_policy="weighted_segment_and_retry", grid_numerics=numerics)
    assert canonical_calls and residual_calls and outer_calls
