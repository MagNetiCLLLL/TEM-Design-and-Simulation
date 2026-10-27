"""Cross-process response reuse, strict provenance and transactional publication."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.immutable_json import freeze_json, json_digest
from temsim.physics import unified_scan_cache as cache
from temsim.physics import unified_scan_disk as disk


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(tmp_path/"scan-cache"))
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_WORKERS", "1")
    cache.clear_scan_response_cache()
    yield
    cache.clear_scan_response_cache()


def _fixture_response():
    expected = {"z_mm": (1401.,), "coil_keys": ("ac_deflector:0",),
        "model": "shared-potential-finite-coil-response-v1", "implementation": "1"*64,
        "source_z_mm": 1400., "perturbation_rad": 1e-6, "maximum_step_mm": .2,
        "reference_kinetic_energy_ev": 300000.}
    bounds = [[-1e-4, -2e-4, 1.4], [1e-4, 2e-4, 1.401]]
    increments = [{}]
    for axis in range(2):
        kick = np.eye(2)[axis]*expected["perturbation_rad"]
        increments.extend(({"ac_deflector:0": kick.tolist()}, {"ac_deflector:0": (-kick).tolist()}))
    traces = [{"source_identity": "2"*64, "runtime_digest": "3"*64,
        "parent_runtime_digest": "3"*64, "field_digest": "4"*64, "geometry_digest": "5"*64,
        "swept_bounds_m": bounds, "coil_perturbations_rad": increment,
        "empirical_aberrations_applied": False, "maximum_step_m": .2*.001,
        "scope": "Synthetic persistence fixture; not physical validation"} for increment in increments]
    value = {"z_mm": (1401.,), "phase": {"ac_deflector:0": np.arange(8., dtype=np.float64).reshape(1, 4, 2)},
        "canonical": {"ac_deflector:0": np.arange(4., dtype=np.float64).reshape(1, 2, 2)},
        "reference": np.array([[-0., np.nextafter(1., 2.), 1e-300, -1e200]]),
        "receipt": {**{key: expected[key] for key in ("model", "implementation", "source_z_mm",
            "perturbation_rad", "maximum_step_mm", "reference_kinetic_energy_ev")},
            "response_dependencies": {"kind": "executed-query-box", "bounds_m": bounds,
                                       "payload_digest": "6"*64}, "trace_receipts": traces}}
    return json_digest(expected), expected, value


def _take(key=None, expected=None, value=None, **kwargs):
    default_key, default_expected, default_value = _fixture_response()
    return cache.cached_scan_response(key or default_key, lambda: deepcopy(value or default_value), lambda: None,
        expected=expected or default_expected, compatible=lambda row: row["receipt"]["response_dependencies"]["payload_digest"] == "6"*64,
        **kwargs)


def test_lossless_cross_process_disk_hit_preserves_original_receipts():
    key, expected, value = _fixture_response()
    result = _take()
    assert result["receipt"]["response_cache"]["source"] == "computed"
    assert len(list(disk.cache_directory().glob("*.scan"))) == 1
    script = """
import json
from test_unified_scan_persistence import _fixture_response
from temsim.physics import unified_scan_cache as cache
key, expected, original = _fixture_response()
def forbidden():
    raise AssertionError('A second process must reuse the completed disk response')
result = cache.cached_scan_response(key, forbidden, lambda: None, expected=expected,
    compatible=lambda row: row['receipt']['response_dependencies'] == original['receipt']['response_dependencies'])
assert result['reference'].tobytes() == original['reference'].tobytes()
assert result['receipt']['trace_receipts'] == original['receipt']['trace_receipts']
assert result['z_mm'] == original['z_mm']
print(json.dumps(result['receipt']['response_cache']))
"""
    environment = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join((str(root/"tests"), str(root/"src"), environment.get("PYTHONPATH", "")))
    completed = subprocess.run([sys.executable, "-X", "utf8", "-c", script],
        env=environment, check=True, capture_output=True, text=True, timeout=40)
    assert json.loads(completed.stdout)["source"] == "disk"
    cache.clear_scan_response_cache(include_disk=False)
    from_disk = _take()
    assert from_disk["receipt"]["response_cache"]["source"] == "disk"
    from_disk["reference"][:] = 99.
    in_memory = _take()
    assert in_memory["receipt"]["response_cache"]["source"] == "memory"
    assert in_memory["reference"].tobytes() == value["reference"].tobytes()


def test_concurrent_process_publishers_leave_one_complete_atomic_record():
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(root/"tests"), str(root/"src"), environment.get("PYTHONPATH", "")))
    script = "from test_unified_scan_persistence import _take; _take()"
    def publish():
        return subprocess.run([sys.executable, "-X", "utf8", "-c", script], env=environment,
            check=True, capture_output=True, text=True, timeout=40).returncode
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: publish(), range(2))) == [0, 0]
    assert len(list(disk.cache_directory().glob("*.scan"))) == 1
    assert not list(disk.cache_directory().glob(".scan-*.tmp"))
    assert _take()["receipt"]["response_cache"]["source"] == "disk"


@pytest.mark.parametrize("dtype, shape", (("O", [1]), ("<f8", [2**60])))
def test_codec_rejects_object_arrays_and_impossible_allocations(dtype, shape):
    with pytest.raises(ValueError):
        disk._decode({"array": [dtype, shape, ""]})


@pytest.mark.parametrize("damage", ("truncated", "checksum", "shape", "trace_count", "bounds", "implementation", "oversized"))
def test_corrupt_or_incomplete_persistence_is_a_miss(damage):
    _take()
    path = next(disk.cache_directory().glob("*.scan"))
    data = path.read_bytes()
    if damage == "truncated":
        path.write_bytes(data[:120])
    elif damage == "checksum":
        path.write_bytes(data[:-2]+b"XX")
    elif damage == "oversized":
        with path.open("r+b") as stream:
            stream.truncate(disk.MAX_ENTRY_BYTES+1)
    else:
        magic, request, entry, _, payload = data.split(b"\n", 4)
        value = disk._decode(json.loads(payload))
        if damage == "shape":
            value["phase"]["ac_deflector:0"] = np.zeros((1, 2, 2))
        elif damage == "trace_count":
            value["receipt"]["trace_receipts"].pop()
        elif damage == "bounds":
            value["receipt"]["response_dependencies"]["bounds_m"] = [[0., 0., 1.4], [0., 0., 1.401]]
        else:
            value["receipt"]["implementation"] = "a"*64
        payload = json.dumps(disk._encode(value), allow_nan=False, separators=(",", ":")).encode()
        path.write_bytes(b"\n".join((magic, request, entry, hashlib.sha256(payload).hexdigest().encode(), payload)))
    cache.clear_scan_response_cache(include_disk=False)
    assert _take()["receipt"]["response_cache"]["source"] == "computed"


def test_clear_deferred_failure_and_cancellation_never_publish(monkeypatch):
    first = _take(defer_disk=True)
    assert not list(disk.cache_directory().glob("*.scan"))
    cache.clear_scan_response_cache(include_disk=False)
    assert cache.commit_scan_response(first["receipt"]["response_cache"]["key"]) is False
    assert not list(disk.cache_directory().glob("*.scan"))
    second = _take(defer_disk=True)
    key = second["receipt"]["response_cache"]["key"]
    def cancelled():
        raise InterruptedError("Publication was cancelled")
    with pytest.raises(InterruptedError, match="cancelled"):
        cache.commit_scan_response(key, cancelled)
    assert not list(disk.cache_directory().glob("*.scan"))
    assert not cache._RESPONSES
    cache.discard_scan_response(key)
    assert not cache._RESPONSES
    third = _take()
    cache.discard_scan_response(third["receipt"]["response_cache"]["key"])
    cache.clear_scan_response_cache(include_disk=False)
    assert _take()["receipt"]["response_cache"]["source"] == "computed"


@pytest.mark.parametrize("error", (InterruptedError, ValueError))
def test_cancellation_at_atomic_publication_boundary_is_not_swallowed(error):
    request, expected, value = _fixture_response()
    entry = json_digest((request, value["receipt"]["response_dependencies"]))
    def cancelled():
        raise error("Cancel exactly once at publication")
    with pytest.raises(error, match="exactly once"):
        disk.write_response(request, entry, value, expected, disk.cache_directory(), cancelled)
    assert not list(disk.cache_directory().glob("*.scan"))
    assert not list(disk.cache_directory().glob(".scan-*.tmp"))


def test_storage_namespaces_unavailable_disk_and_bounded_footprint(monkeypatch, tmp_path):
    _take()
    first_directory = disk.cache_directory()
    second_directory = tmp_path/"other-cache"
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(second_directory))
    assert _take()["receipt"]["response_cache"]["source"] == "computed"
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(first_directory))
    assert _take()["receipt"]["response_cache"]["source"] == "memory"
    blocked = tmp_path/"not-a-directory"
    blocked.write_text("A regular file cannot be a cache directory", encoding="utf-8")
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(blocked))
    assert _take()["receipt"]["response_cache"]["source"] == "computed"
    monkeypatch.setenv("TEMSIM_SCAN_RESPONSE_CACHE_DIR", str(second_directory))
    monkeypatch.setattr(disk, "MAX_ENTRIES", 2)
    for index in range(4):
        request, expected, value = _fixture_response()
        expected["implementation"] = value["receipt"]["implementation"] = str(index+1)*64
        _take(key=json_digest(expected), expected=expected, value=value)
    files = list(second_directory.glob("*.scan"))
    assert len(files) <= 2
    assert sum(path.stat().st_size for path in files) <= disk.MAX_DISK_BYTES


@pytest.fixture
def captured_response_stub(monkeypatch):
    """Use production receipt shapes with deliberately substituted derivatives."""
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    calls = []
    def trace(private, initial, source, targets, **kwargs):
        calls.append(True)
        runtime = get_unified_field_runtime(private)
        phase = np.zeros((len(targets), 1, 4))
        for dx, dy in (kwargs.get("coil_perturbations_rad") or {}).values():
            phase[:, 0, :] = (dx, dy, dx, dy)
        return {"xy_slopes": phase, "kinetic_energy_ev": np.full((len(targets), 1), kwargs["energy_ev"]),
            "vector_potential_t_m": np.zeros((len(targets), 1, 3)), "alive": np.array([True]),
            "receipt": freeze_json({"scope": "Substituted derivative; not physical calibration evidence",
                "source_identity": runtime.source_identity, "runtime_digest": runtime.digest,
                "parent_runtime_digest": runtime.digest, "field_digest": runtime.snapshot.numerical_digest,
                "geometry_digest": runtime.geometry.digest, "empirical_aberrations_applied": False,
                "maximum_step_m": kwargs["maximum_step_m"],
                "coil_perturbations_rad": kwargs.get("coil_perturbations_rad") or {},
                "swept_bounds_m": [[-1e-4, -1e-4, source], [1e-4, 1e-4, targets[-1]]]} )}
    monkeypatch.setattr("temsim.physics.unified_field_particle.trace_unified_response", trace)
    return calls


@pytest.mark.parametrize("column", [option.name for option in AssemblyCatalog().columns])
def test_seven_columns_disk_roundtrip_preserves_real_capture_and_shared_host(column, captured_response_stub):
    """Real capture/dependencies; substituted derivatives are not physics evidence."""
    from test_unified_field_runtime import parked_state
    from temsim.physics.unified_scan_calibration import measure_scan_responses
    from temsim.physics.scan_geometry import _descan_target
    calls = captured_response_stub
    state = parked_state(column)
    targets = (state.sample.z_mm, _descan_target(state)[2])
    first = measure_scan_responses(state, targets)
    count = len(calls)
    assert count == 17
    host = state.descan_deflector._physical_host_key
    assert f"{host}:0" in first["phase"]
    assert not any(key.startswith("descan_deflector:") for key in first["phase"])
    cached_value = next(iter(cache._RESPONSES.values()))
    assert disk.validate_response(cached_value[1], cached_value[2])
    assert len(list(disk.cache_directory().glob("*.scan"))) == 1
    cache.clear_scan_response_cache(include_disk=False)
    again = measure_scan_responses(state, targets)
    assert len(calls) == count
    assert again["receipt"]["response_cache"]["source"] == "disk"
    assert again["receipt"]["trace_receipts"] == first["receipt"]["trace_receipts"]
    for key in first["phase"]:
        assert again["phase"][key].tobytes() == first["phase"][key].tobytes()
    state.descan_deflector._physical_host.upper_y_mrad += .001
    cache.clear_scan_response_cache(include_disk=False)
    changed = measure_scan_responses(state, targets)
    assert len(calls) == 2*count
    assert changed["receipt"]["response_cache"]["source"] == "computed"


def test_numerical_runtime_version_change_invalidates_production_request(captured_response_stub, monkeypatch):
    from test_unified_field_runtime import parked_state
    from temsim.physics import unified_scan_calibration as calibration
    state = parked_state()
    first = calibration.measure_scan_responses(state, (state.sample.z_mm,))
    cache.clear_scan_response_cache(include_disk=False)
    environment = calibration._execution_environment
    def changed(private):
        result = environment(private)
        result["packages"]["numpy"] = "test-only-different-numerical-version"
        return result
    monkeypatch.setattr(calibration, "_execution_environment", changed)
    again = calibration.measure_scan_responses(state, (state.sample.z_mm,))
    assert len(captured_response_stub) == 18
    assert first["receipt"]["response_cache"]["key"] != again["receipt"]["response_cache"]["key"]
    assert again["receipt"]["response_cache"]["source"] == "computed"


def test_failed_enclosing_calibration_discards_previously_published_contract(captured_response_stub):
    from test_unified_scan_calibration import _state
    from temsim.physics.scan_geometry import _descan_target
    from temsim.physics.unified_scan_calibration import measure_scan_responses, calibrate_unified_scan
    state = _state()
    # The substituted equal position derivatives make the pure-shift specimen
    # solve singular, while the complete measured-response contract is valid.
    first = measure_scan_responses(state, (state.sample.z_mm, _descan_target(state)[2]))
    assert first["receipt"]["response_cache"]["source"] == "computed"
    assert len(list(disk.cache_directory().glob("*.scan"))) == 1
    with pytest.raises(ValueError, match="singular"):
        calibrate_unified_scan(state)
    assert len(captured_response_stub) == 17
    assert not cache._RESPONSES
    assert not list(disk.cache_directory().glob("*.scan"))
