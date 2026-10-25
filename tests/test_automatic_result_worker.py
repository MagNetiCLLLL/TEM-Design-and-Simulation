"""Automatic-startup IO routing; no transport or scientific arrays are computed."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QThreadPool, Qt

from temsim.automatic_result import AutomaticResult
from temsim.gui.calculation_controller import CalculationController, SectionFileWorker
from temsim import particle_section_io as codec


def _result(name):
    return SimpleNamespace(identity=sha256(name.encode()).hexdigest(), state_snapshot=None)


def _summary(result):
    return dict(identity=result.identity, quality="Preview", target_z_mm=1200.,
                resumable_through_z_mm=1200., saved_at_utc="2026-10-25T12:00:00+00:00")


def _receipt(result, path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = b"IO routing fixture only; not a serialized scientific result. " + result.identity.encode()
    path.write_bytes(content)
    return dict(_summary(result), path=str(path), _package_digest=sha256(content).hexdigest(),
                _file_fingerprint=codec.section_file_fingerprint(path))


@pytest.fixture
def routing(qapp, monkeypatch, tmp_path):
    store = AutomaticResult(tmp_path / "startup")
    owner = CalculationController(persistent_cache_enabled=False, artifact_cache_root=tmp_path / "artifacts",
                                  startup_result_store=store)
    workers, states, writes = [], [], []
    monkeypatch.setattr(owner.section_file_pool, "start", workers.append)
    owner.section_archive_changed.connect(states.append)
    monkeypatch.setattr(codec, "section_archive_summary", _summary)
    monkeypatch.setattr(codec, "capture_result_for_save", lambda result: result)

    def automatic(result, directory, **kwargs):
        writes.append(result.identity)
        return _receipt(result, directory / f"{result.identity}.temsection")

    monkeypatch.setattr(codec, "archive_section_result", automatic)
    monkeypatch.setattr(codec, "save_section_result", lambda *args, **kwargs: SimpleNamespace(digest="a" * 64))
    monkeypatch.setattr(codec, "checked_section_archive_info", lambda result, path, **kwargs: _receipt(result, path))
    monkeypatch.setattr("temsim.gui.calculation_controller.calculate",
                        lambda *args, **kwargs: pytest.fail("Automatic-result persistence must not compute transport"))
    return SimpleNamespace(owner=owner, store=store, workers=workers, states=states, writes=writes)


def _save(routing, name):
    result = _result(name)
    routing.owner.archive_completed_section(result)
    routing.workers[-1].run()
    assert routing.states[-1]["status"] == "saved", routing.states
    return result


def test_automatic_save_commits_existing_archive_without_rewriting_it(routing):
    first = _save(routing, "first")
    entry = routing.store.entry()
    original = Path(entry["path"]).read_bytes()
    _save(routing, "second")
    worker_count, writes = len(routing.workers), list(routing.writes)
    routing.owner.archive_completed_section(first)
    assert routing.store.entry() == entry
    assert len(routing.workers) == worker_count and routing.writes == writes
    assert Path(entry["path"]).read_bytes() == original
    assert routing.states[-1]["reused"] and routing.states[-1]["auto_startup_saved"]


def test_duplicate_pending_request_does_not_create_an_unowned_intent(routing):
    first = _result("first")
    routing.owner.archive_completed_section(first)
    token = routing.workers[-1].startup_result_token
    for _ in range(3):
        routing.owner.archive_completed_section(first)
    assert len(routing.workers) == 1
    assert routing.workers[0].startup_result_token == token
    routing.workers[0].run()
    assert routing.store.entry()["result_identity"] == first.identity
    assert routing.states[-1]["auto_startup_saved"]


def test_manual_export_never_supersedes_pending_automatic_save(routing, tmp_path):
    _save(routing, "old")
    automatic = _result("automatic")
    routing.owner.archive_completed_section(automatic)
    pending = routing.workers[-1]
    routing.owner.archive_completed_section(_result("manual"), path=tmp_path / "manual.temresult")
    manual = routing.workers[-1]
    assert manual.startup_result_store is None and manual.startup_result_token is None
    manual.run()
    pending.run()
    assert routing.store.entry()["result_identity"] == automatic.identity


def test_later_save_intent_wins_when_workers_finish_out_of_order(routing):
    old = _save(routing, "old")
    routing.owner.archive_completed_section(_result("earlier"))
    earlier = routing.workers[-1]
    latest = _result("latest")
    routing.owner.archive_completed_section(latest)
    later = routing.workers[-1]
    assert routing.store.entry()["result_identity"] == old.identity
    later.run()
    earlier.run()
    assert routing.store.entry()["result_identity"] == latest.identity
    assert routing.states[-1]["auto_startup_saved"] is False


@pytest.mark.parametrize("failure", ["cancel_before", "cancel_after", "save_error", "dispatch_error"])
def test_cancelled_or_failed_save_retains_previous_completed_entry(routing, monkeypatch, failure):
    _save(routing, "old")
    previous = routing.store.entry()
    new = _result("new")
    if failure == "dispatch_error":
        monkeypatch.setattr(routing.owner.section_file_pool, "start",
                            lambda worker: (_ for _ in ()).throw(RuntimeError("Queue rejected")))
        with pytest.raises(RuntimeError, match="Queue rejected"):
            routing.owner.archive_completed_section(new)
        assert not routing.owner._section_file_pending
    else:
        routing.owner.archive_completed_section(new)
        worker = routing.workers[-1]
        if failure == "cancel_before":
            worker.cancel_event.set()
        elif failure == "cancel_after":
            def cancelled_save(result, directory, **kwargs):
                info = _receipt(result, directory / "cancelled.temsection")
                worker.cancel_event.set()
                return info
            monkeypatch.setattr(codec, "archive_section_result", cancelled_save)
        else:
            monkeypatch.setattr(codec, "archive_section_result",
                                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("Disk full")))
        token = worker.startup_result_token
        worker.run()
        assert routing.store.commit_save(token, {}) is False
    assert routing.store.entry() == previous


@pytest.mark.parametrize("operation", ["begin_save", "commit_save"])
def test_index_errors_do_not_turn_successful_file_saves_into_failures(routing, monkeypatch, operation):
    _save(routing, "old")
    previous = routing.store.entry()
    monkeypatch.setattr(routing.store, operation,
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("Index is read-only")))
    current = _save(routing, "new")
    info = routing.states[-1]
    assert info["identity"] == current.identity and Path(info["path"]).is_file()
    assert "Index is read-only" in info["auto_startup_error"]
    assert routing.store.entry() == previous


def test_disabled_restore_saves_data_without_creating_a_startup_intent(routing):
    _save(routing, "old")
    previous = routing.store.entry()
    routing.store.set_enabled(False)
    _save(routing, "new")
    assert routing.store.entry() == previous
    assert "auto_startup_saved" not in routing.states[-1]


def test_queued_cancellation_releases_intent_without_running_worker(routing):
    _save(routing, "old")
    previous = routing.store.entry()
    routing.owner.archive_completed_section(_result("queued"))
    worker = routing.workers[-1]
    token = worker.startup_result_token
    routing.owner._section_file_finished(worker.token)
    assert routing.store.commit_save(token, {}) is False
    assert routing.store.entry() == previous
    assert not routing.owner._section_file_pending


def test_worker_commits_before_queued_gui_callbacks_are_dispatched(qapp, monkeypatch, tmp_path):
    store = AutomaticResult(tmp_path / "startup")
    result = _result("threaded")
    token = store.begin_save(result.identity)
    worker = SectionFileWorker("worker", "save", tmp_path / "data", result, automatic=True,
                               maximum_unpacked_bytes=1024**2,
                               startup_result_store=store, startup_result_token=token)
    monkeypatch.setattr(codec, "archive_section_result",
                        lambda result, directory, **kwargs: _receipt(result, directory / "completed.temsection"))
    delivered, failures = [], []
    worker.signals.result.connect(lambda *args: delivered.append(args), Qt.ConnectionType.QueuedConnection)
    worker.signals.error.connect(failures.append, Qt.ConnectionType.QueuedConnection)
    pool = QThreadPool()
    pool.start(worker)
    assert pool.waitForDone(5000)
    # Deliberately do not pump events before checking disk, as in a blocking close.
    assert not delivered and not failures
    assert AutomaticResult(store.root).entry()["result_identity"] == result.identity
    QCoreApplication.processEvents()
    assert len(delivered) == 1 and not failures
