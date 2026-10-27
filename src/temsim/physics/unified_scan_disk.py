"""Disposable, bounded and lossless storage for complete scan responses.

The typed JSON codec never loads pickle, imports stored classes or executes
stored expressions. A checksum detects incomplete/corrupt files; the caller's
current request and spatial dependencies remain the authority for reuse.
"""
from contextlib import contextmanager
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from temsim.immutable_json import json_digest


_MAGIC = b"TEMFORGE_SCAN_RESPONSE_V1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[0-9a-f]{64}-[0-9a-f]{64}\.scan\Z")
MAX_ENTRY_BYTES = 8*1024**2
MAX_DISK_BYTES = 64*1024**2
MAX_ENTRIES = 32


def cache_directory():
    """Resolve disposable application storage; an empty override disables it."""
    override = os.environ.get("TEMSIM_SCAN_RESPONSE_CACHE_DIR")
    if override is not None:
        return Path(override).absolute() if override.strip() else None
    from temsim.branding import application_cache_root
    return application_cache_root("scan_response_cache_v1").absolute()


def cache_namespace():
    directory = cache_directory()
    return "disabled" if directory is None else os.path.normcase(str(directory))


def _digest(value):
    return type(value) is str and _HEX.fullmatch(value) is not None


def validate_response(value, expected):
    """Reject partial arrays, changed requests and incomplete provenance."""
    if not isinstance(value, dict) or not isinstance(expected, dict):
        return False
    try:
        targets = tuple(value["z_mm"])
        if targets != tuple(expected["z_mm"]) or not targets or not np.isfinite(targets).all():
            return False
        if np.any(np.diff(targets) <= 0):
            return False
        coils = tuple(expected["coil_keys"])
        if not coils or len(set(coils)) != len(coils):
            return False
        for group, shape in (("phase", (len(targets), 4, 2)), ("canonical", (len(targets), 2, 2))):
            if not isinstance(value[group], dict) or set(value[group]) != set(coils):
                return False
            for array in value[group].values():
                if (type(array) is not np.ndarray or array.dtype != np.dtype("float64")
                        or array.shape != shape or not np.isfinite(array).all()):
                    return False
        reference = value["reference"]
        if (type(reference) is not np.ndarray or reference.dtype != np.dtype("float64")
                or reference.shape != (len(targets), 4) or not np.isfinite(reference).all()):
            return False
        receipt = value["receipt"]
        for key in ("model", "implementation", "source_z_mm", "perturbation_rad",
                    "maximum_step_mm", "reference_kinetic_energy_ev"):
            if receipt[key] != expected[key]:
                return False
        if not _digest(receipt["implementation"]):
            return False
        dependencies = receipt["response_dependencies"]
        if not _digest(dependencies["payload_digest"]):
            return False
        if dependencies["kind"] == "executed-query-box":
            bounds = np.asarray(dependencies["bounds_m"], dtype=float)
            if (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1])
                    or bounds[0, 2] > expected["source_z_mm"]*.001 or bounds[1, 2] < targets[-1]*.001):
                return False
        elif dependencies["kind"] != "complete-frozen-runtime":
            return False
        traces = receipt["trace_receipts"]
        if len(traces) != 1+4*len(coils):
            return False
        increments = [{}]
        for coil in coils:
            for axis in range(2):
                kick = np.eye(2)[axis]*expected["perturbation_rad"]
                increments.extend(({coil: kick.tolist()}, {coil: (-kick).tolist()}))
        geometry_ids, parent_ids, source_ids = set(), set(), set()
        trace_boxes = []
        for trace, increment in zip(traces, increments):
            if any(not _digest(trace[key]) for key in
                   ("source_identity", "runtime_digest", "parent_runtime_digest", "field_digest", "geometry_digest")):
                return False
            if trace["empirical_aberrations_applied"] is not False:
                return False
            if trace["maximum_step_m"] != expected["maximum_step_mm"]*.001:
                return False
            if json_digest(trace["coil_perturbations_rad"]) != json_digest(increment):
                return False
            geometry_ids.add(trace["geometry_digest"])
            parent_ids.add(trace["parent_runtime_digest"])
            source_ids.add(trace["source_identity"])
            if dependencies["kind"] == "executed-query-box":
                box = np.asarray(trace["swept_bounds_m"], dtype=float)
                if (box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[0] > box[1])
                        or box[0, 2] > expected["source_z_mm"]*.001 or box[1, 2] < targets[-1]*.001):
                    return False
                trace_boxes.append(box)
        if trace_boxes:
            union = np.stack((np.min([box[0] for box in trace_boxes], axis=0),
                              np.max([box[1] for box in trace_boxes], axis=0)))
            if np.any(bounds[0] > union[0]) or np.any(bounds[1] < union[1]):
                return False
        return len(geometry_ids) == len(parent_ids) == len(source_ids) == 1
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _encode(value):
    if type(value) is np.ndarray:
        if value.dtype.kind not in "biufc" or value.dtype.hasobject:
            raise ValueError("Unsupported scan cache array type")
        return {"array": [value.dtype.str, list(value.shape), base64.b64encode(value.tobytes()).decode("ascii")]}
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("Scan cache mappings require string keys")
        return {"mapping": [[key, _encode(item)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"tuple" if isinstance(value, tuple) else "list": [_encode(item) for item in value]}
    if value is None or type(value) in (str, bool, int, float):
        return value
    raise ValueError("Unsupported scan cache value")


def _decode(value, depth=0):
    if depth > 40:
        raise ValueError("Scan cache nesting limit exceeded")
    if value is None or type(value) in (str, bool, int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Nonfinite scan cache value")
        return value
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("Invalid scan cache typed value")
    kind, items = next(iter(value.items()))
    if kind in ("list", "tuple") and isinstance(items, list):
        decoded = [_decode(item, depth+1) for item in items]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "mapping" and isinstance(items, list):
        result = {}
        for key, item in items:
            if type(key) is not str or key in result:
                raise ValueError("Invalid scan cache mapping key")
            result[key] = _decode(item, depth+1)
        return result
    if kind == "array" and isinstance(items, list) and len(items) == 3:
        dtype = np.dtype(items[0])
        shape = items[1]
        if (dtype.kind not in "biufc" or dtype.hasobject or not isinstance(shape, list)
                or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape)):
            raise ValueError("Invalid scan cache array")
        count = math.prod(shape)
        if count*dtype.itemsize > MAX_ENTRY_BYTES:
            raise ValueError("Scan cache array exceeds its byte budget")
        raw = base64.b64decode(items[2], validate=True)
        if len(raw) != count*dtype.itemsize:
            raise ValueError("Incomplete scan cache array")
        return np.frombuffer(raw, dtype=dtype).reshape(shape)
    raise ValueError("Unknown scan cache typed value")


@contextmanager
def _locked(directory, *, create=False):
    """Bounded OS lock protects publication/pruning across application runs."""
    stream, acquired = None, False
    try:
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        stream = (directory/".scan-cache.lock").open("a+b")
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic()+.5
        while True:
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(.01)
    except OSError:
        pass
    try:
        yield acquired
    finally:
        if stream is not None:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _files(directory):
    return [path for path in directory.iterdir() if _NAME.fullmatch(path.name) and not path.is_symlink()]


def read_candidates(request_key, expected, directory):
    """Unavailable, oversized, unknown or corrupt files are ordinary misses."""
    if directory is None:
        return ()
    results = []
    bytes_read = 0
    try:
        with _locked(directory) as acquired:
            if not acquired:
                return ()
            paths = sorted(_files(directory), key=lambda path: path.stat().st_mtime_ns, reverse=True)
            for path in [p for p in paths if p.name.startswith(request_key+"-")][:MAX_ENTRIES]:
                try:
                    if path.stat().st_size > MAX_ENTRY_BYTES:
                        continue
                    if bytes_read+path.stat().st_size > MAX_DISK_BYTES:
                        break
                    with path.open("rb") as stream:
                        data = stream.read(MAX_ENTRY_BYTES+1)
                    bytes_read += len(data)
                    magic, request, entry, checksum, payload = data.split(b"\n", 4)
                    if (len(data) > MAX_ENTRY_BYTES or magic != _MAGIC or request.decode() != request_key
                            or path.name != request_key+"-"+entry.decode()+".scan"
                            or hashlib.sha256(payload).hexdigest() != checksum.decode()):
                        continue
                    value = _decode(json.loads(payload))
                    if not validate_response(value, expected):
                        continue
                    if json_digest((request_key, value["receipt"]["response_dependencies"])) != entry.decode():
                        continue
                    results.append((entry.decode(), value))
                except (OSError, ValueError, TypeError, KeyError, UnicodeError, RecursionError, OverflowError):
                    continue
    except OSError:
        pass
    return tuple(results)


def write_response(request_key, entry_key, value, expected, directory, check_cancelled):
    """Atomically install only a fully validated complete response."""
    if directory is None or not validate_response(value, expected):
        return False
    temporary, checking_cancelled = None, False
    try:
        payload = json.dumps(_encode(value), ensure_ascii=True, allow_nan=False,
                             separators=(",", ":")).encode("utf-8")
        data = b"\n".join((_MAGIC, request_key.encode(), entry_key.encode(),
                           hashlib.sha256(payload).hexdigest().encode(), payload))
        if len(data) > MAX_ENTRY_BYTES:
            return False
        with _locked(directory, create=True) as acquired:
            if not acquired:
                return False
            # A prior interrupted writer may have left an unpublished staging
            # file. All publishers hold this lock, so these cannot be live.
            for orphan in directory.glob(".scan-*.tmp"):
                if not orphan.is_symlink():
                    orphan.unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".scan-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            checking_cancelled = True
            check_cancelled()
            checking_cancelled = False
            os.replace(temporary, directory/f"{request_key}-{entry_key}.scan")
            temporary = None
            files = sorted(_files(directory), key=lambda path: path.stat().st_mtime_ns, reverse=True)
            total = 0
            for index, path in enumerate(files):
                size = path.stat().st_size
                if index >= MAX_ENTRIES or total+size > MAX_DISK_BYTES:
                    path.unlink(missing_ok=True)
                else:
                    total += size
            return True
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        if checking_cancelled:
            raise
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def discard_response(entry_key=None, directory=None):
    """Delete only recognized cache records; scientific result stores differ."""
    if directory is None:
        return
    try:
        with _locked(directory) as acquired:
            if acquired:
                for path in _files(directory):
                    if entry_key is None or path.name.endswith("-"+entry_key+".scan"):
                        path.unlink(missing_ok=True)
    except OSError:
        pass
