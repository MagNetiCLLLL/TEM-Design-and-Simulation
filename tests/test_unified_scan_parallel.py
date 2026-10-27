"""Unchanged scalar physics under bounded, isolated process scheduling."""
from io import BytesIO
from types import SimpleNamespace
import time

import numpy as np
import pytest

from temsim.physics import unified_scan_parallel as parallel


@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, tmp_path):
    from temsim.physics.unified_scan_cache import clear_scan_response_cache
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(tmp_path/"scan-cache"))
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    clear_scan_response_cache()
    yield
    clear_scan_response_cache()


@pytest.fixture
def captured_reference():
    from test_unified_scan_calibration import _state
    from temsim.physics.finite_scan_response import _private_reference
    from temsim.physics.instrument_magnetic import column_dipole_fields
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    state, _ = _private_reference(_state(), "ac_deflector")
    state.step_mm = .05
    coils = [row for row in column_dipole_fields(state)
             if row.key.startswith(("ac_deflector:", state.descan_deflector._physical_host_key+":"))]
    source = min(row.field_support_mm[0] for row in coils if row.key.startswith("ac_deflector:"))
    runtime = get_unified_field_runtime(state)
    increments = [None]
    for coil in coils:
        for axis in range(2):
            kick = np.eye(2)[axis]*1e-6
            increments.extend(({coil.key: tuple(kick)}, {coil.key: tuple(-kick)}))
    return state, runtime, source, increments


def test_parallel_policy_is_bounded_and_keeps_small_or_frozen_runs_serial(monkeypatch):
    state = SimpleNamespace(step_mm=.05)
    monkeypatch.delenv("TEMSIM_SCAN_RESPONSE_WORKERS")
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    assert parallel.response_worker_count(state, 0., (1.,), .05, 17) == 1
    assert parallel.response_worker_count(state, 0., (1000.,), .05, 17) == 4
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "2")
    assert parallel.response_worker_count(state, 0., (1.,), .05, 17) == 2
    monkeypatch.setattr(parallel.sys, "frozen", True, raising=False)
    assert parallel.response_worker_count(state, 0., (1000.,), .05, 17) == 1


@pytest.mark.parametrize("value", ["0", "5", "-1", "2.5", "unknown"])
def test_invalid_parallel_policy_is_explicit(monkeypatch, value):
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", value)
    with pytest.raises(ValueError, match="integer from 1 to 4"):
        parallel.response_worker_count(SimpleNamespace(step_mm=.05), 0., (1.,), .05, 17)


def test_worker_messages_reject_partial_oversized_and_non_mapping_payloads(monkeypatch):
    monkeypatch.setattr(parallel, "MAX_MESSAGE_BYTES", 32)
    for data in (b'{"kind":"result"}', b"[]\n", b" "*33+b"\n"):
        with pytest.raises(ValueError):
            parallel._read_message(BytesIO(data))
    stream = BytesIO()
    parallel._write_message(stream, {"kind": "finish"})
    stream.seek(0)
    assert parallel._read_message(stream) == {"kind": "finish"}


def test_startup_error_tail_is_bounded_and_retains_the_actual_failure(tmp_path):
    path = tmp_path/"worker.log"
    path.write_bytes(b"old context\n"*2000+b"ImportError: missing worker dependency\n")
    diagnostic = parallel._stderr_tail(path)
    assert len(diagnostic.encode()) <= 8192
    assert diagnostic.endswith("ImportError: missing worker dependency")
    assert parallel._stderr_tail(tmp_path/"absent.log") == ""


def _assert_row_equal(actual, expected):
    from temsim.immutable_json import thaw_json
    assert set(actual) == set(expected)
    for name in expected:
        if isinstance(expected[name], np.ndarray):
            np.testing.assert_array_equal(actual[name], expected[name], err_msg=name)
        elif name == "receipt":
            assert actual[name] == thaw_json(expected[name])
        else:
            assert actual[name] == expected[name]


def test_all_17_process_responses_preserve_exact_fields_clock_energy_order_and_receipts(
        captured_reference, monkeypatch, record_property):
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.unified_field_particle import trace_unified_response
    from temsim.physics.transport_progress import transport_progress
    state, runtime, source, increments = captured_reference
    assert len(increments) == 17
    targets = (source+.35, source+1.)
    before = capture_instrument_snapshot(state).digest
    started = time.perf_counter()
    serial = [trace_unified_response(state, np.zeros((1, 4)), source*1e-3,
              tuple(z*1e-3 for z in targets), energy_ev=300000., flight_time_s=0.,
              maximum_step_m=.05e-3, coil_perturbations_rad=kick) for kick in increments]
    serial_seconds = time.perf_counter()-started
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "4")
    commands, progress = [], []
    real_popen = parallel.subprocess.Popen
    def launch(command, **kwargs):
        commands.append(command)
        return real_popen(command, **kwargs)
    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    with transport_progress(progress.append):
        rows, receipt = parallel.parallel_scan_responses(state, runtime, targets, source,
                                                          300000., .05, increments)
    for actual, expected in zip(rows, serial):
        _assert_row_equal(actual, expected)
    assert np.any(rows[1]["xy_slopes"] != rows[0]["xy_slopes"])
    assert capture_instrument_snapshot(state).digest == before
    assert receipt["workers"] == 4 and receipt["trace_count"] == 17
    assert all(command[1:4] == ["-m", parallel.__name__, "--worker"] for command in commands)
    fractions = [row.local_progress.completed/row.local_progress.total for row in progress]
    assert fractions == sorted(fractions) and fractions[-1] == 1.
    record_property("serial_seconds", serial_seconds)
    record_property("parallel_seconds", receipt["wall_seconds"])


def test_parallel_cancellation_keeps_no_partial_response_in_memory_or_disk(captured_reference, monkeypatch, tmp_path):
    from temsim.physics import unified_scan_cache as cache
    from temsim.physics.transport_progress import transport_progress
    state, runtime, source, increments = captured_reference
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "2")
    seen = []
    def cancelled():
        if seen:
            raise InterruptedError("Caller cancelled the parallel response")
        return False
    state._tuning_cancelled = cancelled
    def execute():
        rows, receipt = parallel.parallel_scan_responses(state, runtime, (source+5.,), source,
                                                         300000., .005, increments)
        return {"receipt": receipt, "rows": rows}
    with transport_progress(seen.append), pytest.raises(InterruptedError, match="Caller cancelled"):
        cache.cached_scan_response("a"*64, execute, lambda: None)
    assert seen and not cache._RESPONSES and not cache._PENDING
    assert not list((tmp_path/"scan-cache").glob("*.scan"))


def test_worker_rejects_changed_runtime_before_propagation(captured_reference, monkeypatch):
    state, runtime, source, increments = captured_reference
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "2")
    wrong = SimpleNamespace(source_identity=runtime.source_identity, digest="f"*64)
    with pytest.raises(RuntimeError, match="differs from captured runtime"):
        parallel.parallel_scan_responses(state, wrong, (source+1.,), source, 300000., .05, increments[:2])


def test_public_response_assembly_and_cache_are_independent_of_worker_count(captured_reference, monkeypatch):
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    from temsim.physics.unified_scan_cache import clear_scan_response_cache
    state, _, source, _ = captured_reference
    targets = (source+.2,)
    serial = measure_scan_responses(state, targets, maximum_step_mm=.05)
    clear_scan_response_cache()
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "2")
    actual = measure_scan_responses(state, targets, maximum_step_mm=.05)
    assert actual["receipt"]["execution"]["mode"] == "independent_processes"
    for group in ("phase", "canonical"):
        assert actual[group].keys() == serial[group].keys()
        for key in actual[group]:
            np.testing.assert_array_equal(actual[group][key], serial[group][key])
    np.testing.assert_array_equal(actual["reference"], serial["reference"])
    assert actual["receipt"]["trace_receipts"] == serial["receipt"]["trace_receipts"]
    assert actual["receipt"]["response_dependencies"] == serial["receipt"]["response_dependencies"]
    assert actual["receipt"]["response_cache"]["key"] == serial["receipt"]["response_cache"]["key"]
    clear_scan_response_cache(include_disk=False)
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    restored = measure_scan_responses(state, targets, maximum_step_mm=.05)
    assert restored["receipt"]["response_cache"]["source"] == "disk"
    assert restored["receipt"]["trace_receipts"] == serial["receipt"]["trace_receipts"]
