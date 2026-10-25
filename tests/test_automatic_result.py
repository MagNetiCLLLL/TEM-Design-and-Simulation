"""Last completed automatic saves remain stable across failures and races."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

import pytest

from temsim import automatic_result
from temsim.automatic_result import AutomaticResult, AutomaticSaveToken
from temsim.result_library import ResultLibrary


def _info(root, name="saved.temresult", **changes):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Completed opaque codec package; no array decoding in this test.")
    return dict({"path": str(path.resolve()), "identity": "a"*64,
        "_package_digest": "b"*64, "quality": "High accuracy", "target_z_mm": 1599.25,
        "resumable_through_z_mm": 1598.5, "saved_at_utc": "2026-10-25T12:34:56.123456+00:00"}, **changes)


def _save(store, info):
    token = store.begin_save(info["identity"])
    assert store.commit_save(token, info)
    return store.entry()


def _process_competing_save(root, info, ready, release, output):
    """Spawn-safe storage-only worker; no Qt, transport or array imports."""
    try:
        store = AutomaticResult(root)
        token = store.begin_save(info["identity"])
        ready.put((token.intent_id, token.result_identity))
        if not release.wait(10):
            raise TimeoutError("Competing save was not released")
        output.put((token.intent_id, store.commit_save(token, info)))
    except BaseException as error:
        output.put(("error", repr(error)))


def _process_crash_with_lock(root, ready):
    import os
    with AutomaticResult(root)._transaction(create=True):
        ready.set()
        os._exit(0)


def test_missing_index_is_enabled_empty_and_does_not_create_storage(tmp_path):
    store = AutomaticResult(tmp_path/"automatic")
    assert store.enabled() is True
    assert store.entry() is None
    assert not store.root.exists()


@pytest.mark.parametrize("extension", [".temresult", ".temsection", ".TEMRESULT"])
def test_roundtrip_references_complete_package_without_copy_or_extra_metadata(tmp_path, extension):
    store = AutomaticResult(tmp_path/"automatic")
    info = _info(tmp_path/"external", "saved"+extension)
    info.update(arrays=object(), result=object(), size_bytes=999)
    original = Path(info["path"]).read_bytes()
    entry = _save(store, info)
    assert entry == {
        "path": info["path"], "result_identity": info["identity"], "package_digest": info["_package_digest"],
        "quality": info["quality"], "target_z_mm": info["target_z_mm"],
        "resumable_through_z_mm": info["resumable_through_z_mm"], "saved_at_utc": info["saved_at_utc"],
        "size_bytes": len(original)}
    assert AutomaticResult(store.root).entry() == entry
    assert {p.name for p in store.root.iterdir()} == {"automatic-result.json", ".automatic-result.lock"}
    assert Path(info["path"]).read_bytes() == original
    entry["path"] = "mutated"
    info["quality"] = "mutated"
    assert store.entry()["path"] != "mutated" and store.entry()["quality"] != "mutated"


def test_pending_failure_and_cancellation_keep_previous_completed_pointer(tmp_path):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    token = store.begin_save("c"*64)
    # No failed worker notification is needed to keep a completed pointer.
    assert AutomaticResult(store.root).entry() == previous
    document = json.loads(store.index_path.read_text())
    assert document["entry"] == previous
    assert "c"*64 not in store.index_path.read_text()  # No partial export metadata.
    with pytest.raises(ValueError, match="completed export metadata"):
        store.commit_save(token, {})
    assert store.entry() == previous
    assert store.cancel_save(token)
    assert not store.cancel_save(token)
    assert not store.commit_save(token, None)
    assert store.entry() == previous


def test_only_newest_intent_can_commit_even_for_same_result_identity(tmp_path):
    first, second = AutomaticResult(tmp_path/"automatic"), AutomaticResult(tmp_path/"automatic")
    previous = _save(first, _info(tmp_path))
    old = first.begin_save("a"*64)
    current = second.begin_save("a"*64)
    assert old != current
    assert not first.commit_save(old, None)  # Stale completions do not touch package I/O.
    assert not first.cancel_save(old)
    assert first.entry() == previous
    info = _info(tmp_path, "new.temresult", _package_digest="d"*64)
    assert first.commit_save(current, info)
    assert not second.commit_save(current, info)
    assert second.entry()["package_digest"] == "d"*64


def test_enable_toggle_invalidates_pending_work_but_never_discards_completed_entry(tmp_path):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    token = store.begin_save("c"*64)
    other = AutomaticResult(store.root)
    other.set_enabled(False)
    assert not store.enabled() and store.entry() == previous
    assert store.begin_save("d"*64) is None
    assert not store.commit_save(token, None)
    assert not store.commit_save(None, None)
    other.set_enabled(True)
    assert store.enabled() and store.entry() == previous
    assert not store.commit_save(token, None)
    new = store.begin_save("a"*64)
    other.set_enabled(True)  # Idempotent enable does not invalidate live work.
    assert store.commit_save(new, _info(tmp_path, "new.temresult"))


def test_named_startup_is_independent_even_when_both_indexes_share_root(tmp_path):
    root = tmp_path/"library"
    named = ResultLibrary(root)
    chosen = named.remember("Favourite", _info(tmp_path, "named.temresult"))
    named.set_startup(chosen["id"])
    original = named.index_path.read_bytes()
    automatic = AutomaticResult(root)
    _save(automatic, _info(tmp_path, "automatic.temsection", identity="c"*64))
    automatic.set_enabled(False)
    assert named.index_path.read_bytes() == original
    assert named.startup_entry() == chosen


def test_entry_reads_never_open_exported_payload_and_preserve_missing_file_metadata(tmp_path, monkeypatch):
    store = AutomaticResult(tmp_path/"automatic")
    info = _info(tmp_path)
    previous = _save(store, info)
    Path(info["path"]).unlink()
    original = Path.open
    def only_index(path, *args, **kwargs):
        assert path == store.index_path
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", only_index)
    assert AutomaticResult(store.root).entry() == previous
    assert store.entry() == previous


@pytest.mark.parametrize("invalid", [None, 0, 1, "false", [], {}])
def test_enable_requires_exact_boolean_without_changing_document(tmp_path, invalid):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    original = store.index_path.read_bytes()
    with pytest.raises(ValueError, match="explicit boolean"):
        store.set_enabled(invalid)
    assert store.entry() == previous and store.index_path.read_bytes() == original


@pytest.mark.parametrize("identity", [None, 1, "a"*63, "A"*64, "g"*64, " a"*32])
def test_invalid_begin_identity_does_not_replace_current_intent(tmp_path, identity):
    store = AutomaticResult(tmp_path/"automatic")
    token = store.begin_save("a"*64)
    original = store.index_path.read_bytes()
    with pytest.raises(ValueError, match="identity"):
        store.begin_save(identity)
    assert store.index_path.read_bytes() == original
    assert store.commit_save(token, _info(tmp_path))


@pytest.mark.parametrize("changes", [
    {"identity": "c"*64}, {"_package_digest": "bad"}, {"quality": ""}, {"quality": " space "},
    {"quality": "a\nb"}, {"target_z_mm": None}, {"target_z_mm": True}, {"target_z_mm": float("nan")},
    {"resumable_through_z_mm": float("inf")}, {"target_z_mm": 2**10000},
    {"resumable_through_z_mm": "10"}, {"saved_at_utc": None},
    {"saved_at_utc": "2026-10-25T12:34:56"}, {"saved_at_utc": "2026-10-25T12:34:56+01:00"},
    {"path": "relative.temresult"}, {"path": ""}, {"path": None},
])
def test_invalid_completed_metadata_never_changes_saved_pointer_or_intent(tmp_path, changes):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    token = store.begin_save("a"*64)
    info = _info(tmp_path, "invalid.temresult", **changes)
    original = store.index_path.read_bytes()
    with pytest.raises(ValueError):
        store.commit_save(token, info)
    assert store.entry() == previous and store.index_path.read_bytes() == original


@pytest.mark.parametrize("kind", ["missing", "empty", "directory", "wrong_suffix"])
def test_unfinished_or_invalid_package_path_is_not_published(tmp_path, kind):
    store = AutomaticResult(tmp_path/"automatic")
    token = store.begin_save("a"*64)
    info = _info(tmp_path)
    path = Path(info["path"])
    if kind == "missing":
        path.unlink()
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "directory":
        path.unlink(); path.mkdir()
    else:
        wrong = path.with_suffix(".tmp"); path.rename(wrong); info["path"] = str(wrong)
    original = store.index_path.read_bytes()
    with pytest.raises(ValueError):
        store.commit_save(token, info)
    assert store.entry() is None and store.index_path.read_bytes() == original


@pytest.mark.parametrize("corrupt", [b"{", b"\xff", b"[]", b'{"schema":"unknown"}',
    b'{"schema":"temsim-automatic-result-v1","schema":"temsim-automatic-result-v1","enabled":true,"intent_id":null,"entry":null}',
    b'{"schema":"temsim-automatic-result-v1","enabled":NaN,"intent_id":null,"entry":null}',
    b" "*(automatic_result.MAX_INDEX_BYTES+1)],
    ids=["broken-json", "invalid-utf8", "array-document", "wrong-schema", "duplicate-key", "nonfinite", "oversized"])
def test_corrupt_document_is_reported_and_preserved_by_every_operation(tmp_path, corrupt):
    store = AutomaticResult(tmp_path/"automatic")
    token = store.begin_save("a"*64)
    info = _info(tmp_path)
    store.index_path.write_bytes(corrupt)
    actions = (lambda: AutomaticResult(store.root), store.enabled, store.entry,
        lambda: store.set_enabled(False), lambda: store.begin_save("c"*64),
        lambda: store.cancel_save(token), lambda: store.commit_save(token, info))
    for action in actions:
        with pytest.raises(ValueError):
            action()
        assert store.index_path.read_bytes() == corrupt


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(extra=0), lambda d: d.update(enabled=1),
    lambda d: d.update(intent_id="bad"), lambda d: d.update(entry={}),
    lambda d: d["entry"].update(size_bytes=True), lambda d: d["entry"].update(extra=1),
    lambda d: d["entry"].update(path="relative.temresult"),
    lambda d: d.update(enabled=False, intent_id="a"*32),
])
def test_structurally_invalid_index_is_not_repaired_or_overwritten(tmp_path, mutation):
    store = AutomaticResult(tmp_path/"automatic")
    _save(store, _info(tmp_path))
    document = json.loads(store.index_path.read_text()); mutation(document)
    payload = json.dumps(document).encode(); store.index_path.write_bytes(payload)
    with pytest.raises(ValueError):
        store.set_enabled(True)
    assert store.index_path.read_bytes() == payload


@pytest.mark.parametrize("failure", ["fsync", "replace"])
@pytest.mark.parametrize("operation", ["begin", "commit", "disable", "cancel"])
def test_atomic_failure_preserves_previous_document_and_cleans_temporary_files(tmp_path, monkeypatch, failure, operation):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    token = store.begin_save("a"*64)
    info = _info(tmp_path, "next.temresult")
    original = store.index_path.read_bytes()
    def fail(*_args):
        raise OSError("Injected persistence failure")
    with monkeypatch.context() as patch:
        patch.setattr(automatic_result.os, failure, fail)
        action = {"begin": lambda: store.begin_save("c"*64), "commit": lambda: store.commit_save(token, info),
                  "disable": lambda: store.set_enabled(False), "cancel": lambda: store.cancel_save(token)}[operation]
        with pytest.raises(OSError, match="Injected"):
            action()
    assert store.index_path.read_bytes() == original and store.entry() == previous
    assert not list(store.root.glob(".automatic-result-*.tmp"))
    assert store.commit_save(token, info)


def test_wrong_owner_and_invalid_tokens_cannot_publish_or_cancel(tmp_path):
    first, second = AutomaticResult(tmp_path/"first"), AutomaticResult(tmp_path/"second")
    token = first.begin_save("a"*64)
    second.begin_save("a"*64)
    assert not second.commit_save(token, None) and not second.cancel_save(token)
    for invalid in ({}, "token", replace(token, intent_id="bad"), replace(token, result_identity="bad")):
        with pytest.raises(ValueError):
            first.commit_save(invalid, None)
        with pytest.raises(ValueError):
            first.cancel_save(invalid)
    assert first.commit_save(token, _info(tmp_path))


def test_concurrent_instances_publish_only_current_intent(tmp_path):
    stores = [AutomaticResult(tmp_path/"automatic") for _ in range(12)]
    info = _info(tmp_path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        tokens = list(pool.map(lambda store: store.begin_save(info["identity"]), stores))
    current = json.loads(stores[0].index_path.read_text())["intent_id"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(lambda pair: pair[0].commit_save(pair[1], info), zip(stores, tokens)))
    assert outcomes == [token.intent_id == current for token in tokens]
    assert stores[-1].entry()["result_identity"] == info["identity"]


def test_competing_processes_read_current_intent_before_commit(tmp_path):
    root = tmp_path/"automatic"
    context = multiprocessing.get_context("spawn")
    ready, output, release = context.Queue(), context.Queue(), context.Event()
    infos = [_info(tmp_path, "first.temresult"), _info(tmp_path, "second.temsection", identity="c"*64)]
    children = [context.Process(target=_process_competing_save, args=(root, info, ready, release, output)) for info in infos]
    try:
        for child in children:
            child.start()
        intents = [ready.get(timeout=15) for _ in children]
        store = AutomaticResult(root)
        current = json.loads(store.index_path.read_text())["intent_id"]
        release.set()
        results = [output.get(timeout=15) for _ in children]
        assert sorted(results) == sorted((intent, intent == current) for intent, _ in intents)
        assert store.entry()["result_identity"] == dict(intents)[current]
    finally:
        release.set()
        for child in children:
            child.join(15)
            if child.is_alive():
                child.terminate(); child.join(5)
    assert all(child.exitcode == 0 for child in children)


def test_process_crash_releases_lock_and_retains_completed_entry(tmp_path):
    store = AutomaticResult(tmp_path/"automatic")
    previous = _save(store, _info(tmp_path))
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child = context.Process(target=_process_crash_with_lock, args=(store.root, ready))
    child.start()
    try:
        assert ready.wait(10)
        child.join(10)
        assert child.exitcode == 0
        assert store.entry() == previous
        store.set_enabled(False)
        assert store.entry() == previous
    finally:
        if child.is_alive():
            child.terminate(); child.join(5)


def test_invalid_root_index_directory_and_bounded_lock_timeout(tmp_path, monkeypatch):
    file = tmp_path/"file"; file.write_text("not a directory")
    with pytest.raises(ValueError, match="root"):
        AutomaticResult(file)
    store = AutomaticResult(tmp_path/"automatic")
    store.root.mkdir(); store.index_path.mkdir()
    with pytest.raises(ValueError, match="JSON file"):
        store.enabled()
    store.index_path.rmdir()
    other = AutomaticResult(store.root)
    monkeypatch.setattr(automatic_result, "LOCK_TIMEOUT_SECONDS", .05)
    with store._transaction(create=True):
        with pytest.raises(TimeoutError):
            other.set_enabled(False)
    assert other.enabled() is True


@pytest.mark.parametrize("target", ["index", "lock"])
def test_symbolic_link_index_and_lock_are_not_followed(tmp_path, target):
    store = AutomaticResult(tmp_path/"automatic")
    store.root.mkdir()
    external = tmp_path/"external"; external.write_text("preserve")
    link = store.index_path if target == "index" else store.root/".automatic-result.lock"
    try:
        link.symlink_to(external)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this host")
    with pytest.raises(ValueError, match="owned"):
        store.set_enabled(False)
    assert external.read_text() == "preserve"
