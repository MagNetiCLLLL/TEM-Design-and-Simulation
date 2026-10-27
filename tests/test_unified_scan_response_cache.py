"""Exact response reuse, failure ownership and dependency-scoped invalidation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from temsim.physics import unified_scan_cache as cache


@pytest.fixture(autouse=True)
def empty_response_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(tmp_path/"scan-cache"))
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    cache.clear_scan_response_cache()
    yield
    cache.clear_scan_response_cache()


def _value(value=1.):
    return {"phase": {"coil": np.array([[value]])}, "receipt": {"original": ["executed"]}}


def test_returned_arrays_and_receipts_cannot_mutate_the_retained_response():
    calls = []
    def execute():
        calls.append(True)
        return _value()
    first = cache.cached_scan_response("a"*64, execute, lambda: None)
    assert first["receipt"]["response_cache"]["hit"] is False
    first["phase"]["coil"][:] = 99.
    first["receipt"]["original"].append("caller annotation")
    again = cache.cached_scan_response("a"*64, execute, lambda: None)
    np.testing.assert_array_equal(again["phase"]["coil"], [[1.]])
    assert again["receipt"]["original"] == ["executed"]
    assert again["receipt"]["response_cache"]["hit"] is True
    assert calls == [True]


def test_failure_and_post_execution_cancellation_do_not_publish():
    calls = []
    def failure():
        calls.append("failed")
        raise ValueError("Incomplete derivative")
    with pytest.raises(ValueError, match="Incomplete"):
        cache.cached_scan_response("b"*64, failure, lambda: None)
    checks = []
    def cancel_after_execution():
        checks.append(True)
        if len(checks) == 2:
            raise InterruptedError("Discard completed but cancelled work")
    with pytest.raises(InterruptedError, match="cancelled"):
        cache.cached_scan_response("b"*64, lambda: _value(2.), cancel_after_execution)
    actual = cache.cached_scan_response("b"*64, lambda: _value(3.), lambda: None)
    assert actual["receipt"]["response_cache"]["hit"] is False
    np.testing.assert_array_equal(actual["phase"]["coil"], [[3.]])


def test_parallel_equal_requests_measure_once_and_return_independent_results():
    entered, release = Event(), Event()
    calls = []
    def execute():
        calls.append(True)
        entered.set()
        assert release.wait(3.)
        return _value()
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.cached_scan_response, "c"*64, execute, lambda: None)
        assert entered.wait(3.)
        second = pool.submit(cache.cached_scan_response, "c"*64, execute, lambda: None)
        release.set()
        values = first.result(timeout=3.), second.result(timeout=3.)
    assert calls == [True]
    assert sorted(row["receipt"]["response_cache"]["hit"] for row in values) == [False, True]
    assert not np.shares_memory(values[0]["phase"]["coil"], values[1]["phase"]["coil"])


def test_failed_owner_releases_a_waiting_request():
    entered, release = Event(), Event()
    def failure():
        entered.set()
        assert release.wait(3.)
        raise ValueError("owner failed")
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.cached_scan_response, "d"*64, failure, lambda: None)
        assert entered.wait(3.)
        second = pool.submit(cache.cached_scan_response, "d"*64, lambda: _value(2.), lambda: None)
        release.set()
        with pytest.raises(ValueError, match="owner failed"):
            first.result(timeout=3.)
        result = second.result(timeout=3.)
    assert result["receipt"]["response_cache"]["hit"] is False
    np.testing.assert_array_equal(result["phase"]["coil"], [[2.]])


def test_clear_during_measurement_prevents_republication_and_lru_is_bounded(monkeypatch):
    def execute():
        cache.clear_scan_response_cache()
        return _value()
    cache.cached_scan_response("e"*64, execute, lambda: None)
    assert not cache._RESPONSES
    monkeypatch.setattr(cache, "_MAXIMUM_RESPONSES", 2)
    for key in ("a", "b", "c"):
        cache.cached_scan_response(key*64, lambda: _value(), lambda: None)
    assert len(cache._RESPONSES) == 2
    assert cache.cached_scan_response("a"*64, lambda: _value(), lambda: None)["receipt"]["response_cache"]["hit"] is False


def test_same_request_keeps_multiple_exact_dependency_versions():
    def value(version):
        result = _value(version)
        result["receipt"]["response_dependencies"] = {"version": version}
        return result
    def take(version):
        return cache.cached_scan_response("f"*64, lambda: value(version), lambda: None,
            compatible=lambda row: row["receipt"]["response_dependencies"]["version"] == version)
    assert take(1)["receipt"]["response_cache"]["hit"] is False
    assert take(2)["receipt"]["response_cache"]["hit"] is False
    again = take(1)
    assert again["receipt"]["response_cache"]["hit"] is True
    cache.discard_scan_response(again["receipt"]["response_cache"]["key"])
    assert take(1)["receipt"]["response_cache"]["hit"] is False
    assert take(2)["receipt"]["response_cache"]["hit"] is True


@pytest.fixture
def substituted_responses(monkeypatch):
    """Real captures and dependency checks, with no particle integration."""
    from test_unified_scan_calibration import _state
    calls = []
    def trace(private, initial, source, targets, **kwargs):
        calls.append(kwargs.get("coil_perturbations_rad"))
        phase = np.zeros((len(targets), 1, 4))
        for dx, dy in (kwargs.get("coil_perturbations_rad") or {}).values():
            phase[:, 0, :] = (dx, dy, dx, dy)
        return {"xy_slopes": phase, "kinetic_energy_ev": np.full((len(targets), 1), kwargs["energy_ev"]),
                "vector_potential_t_m": np.zeros((len(targets), 1, 3)), "alive": np.array([True]),
                "receipt": {"scope": "Substituted derivative; not physical calibration evidence",
                    "swept_bounds_m": [[-1e-4, -1e-4, source], [1e-4, 1e-4, targets[-1]]]}}
    monkeypatch.setattr("temsim.physics.unified_field_particle.trace_unified_response", trace)
    monkeypatch.setattr("temsim.calculation_manifest.solver_source_identity", lambda: "1"*64)
    return _state(), calls


def test_response_reuses_after_population_raster_and_certified_remote_c2_edits(substituted_responses):
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    state, calls = substituted_responses
    first = measure_scan_responses(state, (state.sample.z_mm,))
    assert len(calls) == 9
    state.electron_gun.emitter.ray_count = 49
    state.ac_deflector.scan_pixels_x += 1
    state.ac_deflector.scan_pixel_size_nm *= 1.1
    state.step_mm *= .5
    next(lens for lens in state.lenses if lens.key == "condenser_lens_2").percent += .1
    again = measure_scan_responses(state, (state.sample.z_mm,))
    assert len(calls) == 9
    assert again["receipt"]["response_cache"]["hit"] is True
    for key in first["phase"]:
        np.testing.assert_array_equal(again["phase"][key], first["phase"][key])
    # Actual lens effects inside the measured interval invalidate the response.
    state.objective_lens.percent += .1
    changed = measure_scan_responses(state, (state.sample.z_mm,))
    assert len(calls) == 18
    assert changed["receipt"]["response_cache"]["hit"] is False


@pytest.mark.parametrize("change", ["energy", "target", "step", "effective_step", "perturbation", "dc", "coil", "geometry"])
def test_response_invalidates_consumed_reference_numerics_and_drives(substituted_responses, change):
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    state, calls = substituted_responses
    target, options = state.sample.z_mm, {}
    measure_scan_responses(state, (target,))
    if change == "energy":
        state.electron_gun.accelerator.high_tension_kv -= 1.
    elif change == "target":
        target += .1
    elif change == "step":
        options["maximum_step_mm"] = .1
    elif change == "effective_step":
        state.step_mm = .1
    elif change == "perturbation":
        options["perturbation_rad"] = 2e-6
    elif change == "dc":
        state.ac_deflector.kick_x_mrad += .001
    elif change == "geometry":
        state.ac_deflector.mechanical_clear_bore_diameter_mm *= .9
    else:
        state.ac_deflector.effective_thickness_mm *= .9
    result = measure_scan_responses(state, (target,), **options)
    assert len(calls) == 18
    assert result["receipt"]["response_cache"]["hit"] is False


def test_cancelled_hit_does_not_return_cached_work(substituted_responses):
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    state, calls = substituted_responses
    measure_scan_responses(state, (state.sample.z_mm,))
    state._tuning_cancelled = lambda: True
    with pytest.raises(RuntimeError, match="Superseded"):
        measure_scan_responses(state, (state.sample.z_mm,))
    assert len(calls) == 9


def test_descan_response_reuses_raster_changes_and_tracks_its_shared_host(substituted_responses):
    from temsim.physics.scan_geometry import _descan_target
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    state, calls = substituted_responses
    targets = (state.sample.z_mm, _descan_target(state)[2])
    first = measure_scan_responses(state, targets)
    assert len(calls) == 17
    host = state.descan_deflector._physical_host
    assert f"{host.key}:0" in first["phase"]
    assert not any(key.startswith("descan_deflector:") for key in first["phase"])
    state.ac_deflector.scan_pixels_x += 1
    state.ac_deflector.scan_pixel_size_nm *= 1.1
    result = measure_scan_responses(state, targets)
    assert len(calls) == 17
    assert result["receipt"]["response_cache"]["hit"] is True
    host.upper_y_mrad += .001
    changed = measure_scan_responses(state, targets)
    assert len(calls) == 34
    assert changed["receipt"]["response_cache"]["hit"] is False


def test_failed_drive_solution_discards_its_measured_response(monkeypatch):
    from test_unified_scan_calibration import _state, _drift_response
    from temsim.physics.unified_scan_calibration import calibrate_unified_scan
    state = _state()
    def singular(work, targets, **kwargs):
        def execute():
            measured = _drift_response(work, targets, **kwargs)
            measured["phase"][work.descan_deflector._physical_host_key+":1"][:] = 0.
            return measured
        return cache.cached_scan_response("a"*64, execute, lambda: None)
    monkeypatch.setattr("temsim.physics.unified_scan_calibration.measure_scan_responses", singular)
    with pytest.raises(ValueError, match="singular"):
        calibrate_unified_scan(state)
    assert not cache._RESPONSES


def test_real_local_response_reuse_matches_fresh_after_remote_upstream_edit():
    """Bounded real 1 mm finite-coil propagation, not full scan qualification."""
    from test_unified_scan_calibration import _state
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    state = _state()
    source = min(coil.field_support_mm[0] for coil in column_dipole_fields(state)
                 if coil.key.startswith("ac_deflector:"))
    target = source+1.
    first = measure_scan_responses(state, (target,))
    assert first["receipt"]["response_cache"]["hit"] is False
    assert first["receipt"]["response_dependencies"]["kind"] == "executed-query-box"
    next(lens for lens in state.lenses if lens.key == "condenser_lens_2").percent += .1
    reused = measure_scan_responses(state, (target,))
    assert reused["receipt"]["response_cache"]["hit"] is True
    cache.clear_scan_response_cache()
    fresh = measure_scan_responses(state, (target,))
    assert fresh["receipt"]["response_cache"]["hit"] is False
    assert fresh["receipt"]["trace_receipts"][0]["runtime_digest"] != first["receipt"]["trace_receipts"][0]["runtime_digest"]
    for group in ("phase", "canonical"):
        for key in fresh[group]:
            np.testing.assert_array_equal(reused[group][key], fresh[group][key])
    np.testing.assert_array_equal(reused["reference"], fresh["reference"])
