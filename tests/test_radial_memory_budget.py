"""Radial budget failures precede allocations and preserve executed inputs."""
import numpy as np
import pytest

from temsim.physics import radial_column_wave as radial
from temsim.physics.wave_grid import WaveMemoryBudgetError, WaveSamplingError


def field(samples=32):
    radius = np.geomspace(1e-8, 1., samples, endpoint=False)
    return radial.RadialWave(radius, np.exp(-radius*radius).astype(complex))


def test_initial_load_reserves_scratch_and_retained_checkpoints_before_loading(monkeypatch):
    loaded = []
    monkeypatch.setattr(radial, "check_available_memory", lambda _: pytest.fail("budget failed first"))
    with pytest.raises(WaveMemoryBudgetError, match="samples=32"):
        radial.RadialExecution(lambda n: loaded.append(n), 32, 1024,
                               maximum_working_bytes=256*32, retained_bytes=1)
    assert loaded == []


def test_available_memory_reserves_only_additional_scratch(monkeypatch):
    available_checks = []
    monkeypatch.setattr(radial, "check_available_memory", available_checks.append)
    original = field()
    execution = radial.RadialExecution(lambda n: original, 32, 1024,
        maximum_working_bytes=256*32+100, retained_bytes=100)
    assert execution.wave is original
    assert available_checks == [256*32]


def test_replay_budget_failure_keeps_original_field_and_does_not_reload(monkeypatch):
    monkeypatch.setattr(radial, "check_available_memory", lambda _: None)
    loaded, reports = [], []
    original = field()
    def load(samples):
        loaded.append(samples)
        return original
    execution = radial.RadialExecution(load, 32, 1024,
        maximum_working_bytes=256*32, report=lambda *args: reports.append(args))
    def unresolved(wave):
        raise WaveSamplingError("Requires real upstream replay", 4.)
    with pytest.raises(WaveMemoryBudgetError, match="not allocated"):
        execution.apply(unresolved)
    assert loaded == [32]
    assert execution.wave is original
    assert execution.samples == 32
    assert execution.operations == execution.refinements == execution.ledger == []
    assert reports == []


def test_domain_expansion_checks_before_allocating_arrays(monkeypatch):
    monkeypatch.setattr(radial, "check_available_memory", lambda _: None)
    original = field()
    before = original.amplitude.copy()
    def allocation_forbidden(*args, **kwargs):
        pytest.fail("Domain array allocated before the budget check")
    monkeypatch.setattr(radial.np, "arange", allocation_forbidden)
    with pytest.raises(WaveMemoryBudgetError):
        radial.expand_radial_domain(original, maximum_working_bytes=256*32)
    np.testing.assert_array_equal(original.amplitude, before)


def test_execution_forwards_budget_to_hankel_domain_retry(monkeypatch):
    monkeypatch.setattr(radial, "check_available_memory", lambda _: None)
    original = field()
    execution = radial.RadialExecution(lambda n: original, 32, 1024,
                                       maximum_working_bytes=256*32)
    def wrapping(wave):
        raise radial.RadialDomainError("Independent reciprocal-domain check")
    with pytest.raises(WaveMemoryBudgetError):
        execution.apply(wrapping)
    assert execution.wave is original
    assert execution.operations == execution.refinements == []


def test_os_memory_failure_prevents_initial_load(monkeypatch):
    def unavailable(_):
        raise MemoryError("Independent physical-memory probe")
    monkeypatch.setattr(radial, "check_available_memory", unavailable)
    loaded = []
    with pytest.raises(MemoryError, match="physical-memory"):
        radial.RadialExecution(lambda n: loaded.append(n), 32, 1024)
    assert loaded == []


@pytest.mark.parametrize("kwargs", [
    {"maximum_working_bytes": 0}, {"maximum_working_bytes": True},
    {"retained_bytes": -1}, {"retained_bytes": 1.5}, {"retained_bytes": True},
])
def test_invalid_memory_limits_fail_before_loading(kwargs):
    with pytest.raises(ValueError, match="Radial"):
        radial.RadialExecution(lambda n: pytest.fail("load called"), 32, 1024, **kwargs)
