"""Bounded process scheduling for unchanged, independent scan response traces.

Workers start this module, never the caller's main script or Qt application.
The captured instrument is restored once per worker and its complete runtime
identity must match the parent. Only scheduling changes: every task calls the
ordinary scalar response executor with its original numerical arguments.
"""
from contextlib import ExitStack, redirect_stdout
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread
import time


MAX_WORKERS = 4
MIN_PARALLEL_NODES = 2000
MAX_MESSAGE_BYTES = 16*1024**2
MAX_INPUT_BYTES = 128*1024**2
_SCALE = 1_000_000


def response_worker_count(state, source, targets, step, count):
    """An explicit one-worker override retains the scalar reference path."""
    requested = os.environ.get("TEMSIM_SCAN_RESPONSE_WORKERS")
    if requested is not None:
        try:
            workers = int(requested)
        except ValueError as exc:
            raise ValueError("TEMSIM_SCAN_RESPONSE_WORKERS must be an integer from 1 to 4") from exc
        if not 1 <= workers <= MAX_WORKERS:
            raise ValueError("TEMSIM_SCAN_RESPONSE_WORKERS must be an integer from 1 to 4")
    else:
        workers = min(MAX_WORKERS, max(1, (os.cpu_count() or 1)-1))
        nodes = math.ceil((targets[-1]-source)/min(float(state.step_mm), step))
        if nodes < MIN_PARALLEL_NODES:
            workers = 1
    # Frozen executables need a dedicated packaged worker entry point. Do not
    # relaunch an application executable with Python interpreter arguments.
    if getattr(sys, "frozen", False) or not sys.executable:
        workers = 1
    return min(workers, count)


def _write_message(stream, message):
    data = json.dumps(message, separators=(",", ":"), allow_nan=False).encode()+b"\n"
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Scan worker message exceeds the bounded transfer budget")
    # Raw/unbuffered pipes may accept only part of a frame. In particular, do
    # not report a successful send when its terminating newline was not sent.
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0:
            raise BrokenPipeError("Scan worker pipe closed before the complete message was written")
        remaining = remaining[written:]
    stream.flush()


def _read_message(stream):
    data = stream.readline(MAX_MESSAGE_BYTES+1)
    if not data:
        return None
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError(f"Oversized scan worker message ({len(data)} bytes read; "
                         f"limit {MAX_MESSAGE_BYTES} bytes)")
    if not data.endswith(b"\n"):
        raise ValueError(f"Incomplete scan worker message ({len(data)} bytes before EOF; missing newline)")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError("Invalid scan worker message")
    return value


def _read_worker(process, messages, worker_index):
    try:
        while (message := _read_message(process.stdout)) is not None:
            messages.put((worker_index, message))
    except BaseException as exc:
        messages.put((worker_index, {"kind": "error", "message": str(exc)}))
    finally:
        messages.put((worker_index, {"kind": "closed"}))


def _stderr_tail(path):
    """Keep a bounded startup-failure diagnostic before temporary cleanup."""
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell()-8192))
            return stream.read(8192).decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def parallel_scan_responses(state, runtime, targets, source, energy, step, increments):
    """Return ordered complete rows, or None for the scalar reference policy.

    Partial worker results are local to this call. Failure or cancellation
    stops the remaining workers before it can reach the reusable cache.
    """
    count = len(increments)
    workers = response_worker_count(state, source, targets, step, count)
    if workers == 1:
        return None
    from temsim.instrument_snapshot import capture_instrument_snapshot
    from temsim.physics.optical_tuning import check_tuning_cancelled
    from temsim.physics.transport_progress import report_transport_progress
    from temsim.physics.unified_scan_disk import _decode
    check_tuning_cancelled(state)
    snapshot = capture_instrument_snapshot(state)
    if snapshot.digest != runtime.source_identity:
        raise ValueError("Instrument changed before parallel scan response capture")
    request = {"snapshot": snapshot.to_dict(), "runtime_digest": runtime.digest,
               "source": source, "targets": targets, "energy": energy, "step": step}
    encoded = json.dumps(request, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > MAX_INPUT_BYTES:
        # A very large pinned-input graph remains supported by the existing
        # serial executor, without copying it into multiple process heaps.
        return None
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(
        [str(Path(__file__).resolve().parents[2]), *[p for p in sys.path if p]]))
    environment["TEMSIM_SCAN_RESPONSE_WORKERS"] = "1"
    processes, readers, logs, results = [], [], [], [None]*count
    messages, fractions, completed = Queue(), [0]*count, set()
    started = time.perf_counter()
    success = False
    with TemporaryDirectory(prefix="temforge-scan-workers-") as folder, ExitStack() as stack:
        input_path = Path(folder)/"captured-inputs.json"
        input_path.write_bytes(encoded)
        try:
            for worker_index in range(workers):
                check_tuning_cancelled(state)
                logs.append(Path(folder)/f"worker-{worker_index}.log")
                stderr = stack.enter_context(logs[-1].open("w+b"))
                process = subprocess.Popen(
                    [sys.executable, "-m", __name__, "--worker", str(input_path)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    env=environment, creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
                processes.append(process)
                reader = Thread(target=_read_worker, args=(process, messages, worker_index), daemon=True)
                reader.start()
                readers.append(reader)
                for index in range(worker_index, count, workers):
                    _write_message(process.stdin, {"kind": "trace", "index": index,
                                                   "increments": increments[index]})
                _write_message(process.stdin, {"kind": "finish"})
            while len(completed) < count:
                check_tuning_cancelled(state)
                try:
                    worker_index, message = messages.get(timeout=.1)
                except Empty:
                    continue
                kind = message.get("kind")
                if kind in {"error", "closed"}:
                    # A worker may exit successfully while other workers are
                    # still running, after delivering all of its own rows.
                    expected = set(range(worker_index, count, workers))
                    if kind == "closed" and expected <= completed:
                        continue
                    detail = str(message.get("message", "worker exited before completion"))
                    returncode = processes[worker_index].poll()
                    if returncode is not None:
                        detail += f" (worker {worker_index+1}, exit code {returncode})"
                    diagnostic = _stderr_tail(logs[worker_index])
                    raise RuntimeError("Parallel scan response failed: "+detail+
                                       ("\nWorker diagnostic: "+diagnostic if diagnostic else ""))
                index = message.get("index")
                if type(index) is not int or not 0 <= index < count or index % workers != worker_index:
                    raise ValueError("Scan worker returned a mismatched trace identity")
                if kind == "progress":
                    fraction = message.get("fraction")
                    if type(fraction) is not int or not 0 <= fraction < _SCALE:
                        raise ValueError("Invalid scan worker progress")
                    fractions[index] = max(fractions[index], fraction)
                elif kind == "result":
                    if index in completed:
                        raise ValueError("Scan worker returned a duplicate response")
                    results[index] = _decode(message["row"])
                    completed.add(index)
                    fractions[index] = _SCALE
                else:
                    raise ValueError("Unknown scan worker response")
                report_transport_progress(
                    f"Calibrating shared-potential scan response | {workers} CPU workers | {len(completed)}/{count} traces complete",
                    completed=sum(fractions), total=count*_SCALE)
            check_tuning_cancelled(state)
            success = True
        finally:
            # stdin is consumed by a separate worker control thread, so a
            # cancellation event can interrupt the existing per-RK-step hook.
            if not success:
                for process in processes:
                    try:
                        _write_message(process.stdin, {"kind": "cancel"})
                    except (OSError, ValueError):
                        pass
            deadline = time.monotonic()+2.
            cleanup_errors = []
            for process in processes:
                try:
                    try:
                        process.wait(timeout=max(.01, deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try:
                            process.wait(timeout=2.)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5.)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    cleanup_errors.append(str(exc))
                finally:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
            for reader in readers:
                reader.join(timeout=1.)
            for process in processes:
                process.stdout.close()
            if success and (cleanup_errors or any(process.returncode != 0 for process in processes)):
                raise RuntimeError("Scan worker did not exit cleanly after completing its response")
    return results, {"mode": "independent_processes", "workers": workers,
                     "trace_count": count, "wall_seconds": time.perf_counter()-started,
                     "algorithm": "unchanged scalar Lorentz RK4; ordered independent physical-coil perturbations"}


def _worker_main(path):
    """Private fixed entry point; never imports a caller's application main."""
    # Reserve the protocol pipe before numerical/provider imports can emit
    # diagnostics. Redirect fd 1 as well as Python stdout, so native writes
    # cannot mix with JSON frames. Leave fd 1 on stderr through process exit
    # to keep late native-library flushes out of the protocol too.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0) as output:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with redirect_stdout(sys.stderr):
            return _worker_loop(path, output)


def _worker_loop(path, output):
    """Execute captured traces with a dedicated, isolated protocol stream."""
    from temsim import input_io
    from temsim.immutable_json import thaw_json
    from temsim.instrument_snapshot import InstrumentSnapshot
    from temsim.physics.transport_progress import transport_progress
    from temsim.physics.unified_field_particle import trace_unified_response
    from temsim.physics.unified_field_runtime import get_unified_field_runtime
    from temsim.physics.unified_scan_disk import _encode
    import numpy as np
    cancelled, tasks, output_lock = Event(), Queue(), Lock()

    def send(message):
        with output_lock:
            _write_message(output, message)

    def receive():
        try:
            # An unbuffered duplicate avoids a daemon thread retaining the
            # interpreter's buffered-stdin lock during process shutdown.
            control = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
            while (message := _read_message(control)) is not None:
                if message.get("kind") == "cancel":
                    cancelled.set()
                else:
                    tasks.put(message)
        except BaseException:
            cancelled.set()
        finally:
            tasks.put({"kind": "finish"})

    Thread(target=receive, daemon=True).start()
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(MAX_INPUT_BYTES+1)
        if len(data) > MAX_INPUT_BYTES:
            raise ValueError("Scan worker input exceeds its memory budget")
        request = json.loads(data)
        state = InstrumentSnapshot.from_dict(request["snapshot"]).restore()
        state._tuning_cancelled = cancelled.is_set
        with input_io.input_scope(state, inherit=False):
            runtime = get_unified_field_runtime(state)
            if runtime.digest != request["runtime_digest"]:
                raise ValueError("Restored scan worker field or geometry differs from captured runtime")
            while True:
                task = tasks.get()
                if task.get("kind") == "finish":
                    break
                if task.get("kind") != "trace" or cancelled.is_set():
                    raise RuntimeError("Cancelled or invalid scan worker request")
                index = task["index"]

                def progress(label):
                    local = getattr(label, "local_progress", None)
                    if local is not None and local.total:
                        send({"kind": "progress", "index": index,
                              "fraction": min(_SCALE-1, local.completed*_SCALE//local.total)})

                with transport_progress(progress):
                    row = trace_unified_response(state, np.zeros((1, 4)), request["source"]*1e-3,
                        tuple(z*1e-3 for z in request["targets"]), energy_ev=request["energy"],
                        flight_time_s=0., maximum_step_m=request["step"]*1e-3,
                        coil_perturbations_rad=task["increments"])
                row["receipt"] = thaw_json(row["receipt"])
                send({"kind": "result", "index": index, "row": _encode(row)})
    except BaseException as exc:
        send({"kind": "error", "message": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module is an internal scan-response worker")
    raise SystemExit(_worker_main(sys.argv[2]))
