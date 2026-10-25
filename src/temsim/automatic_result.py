"""Atomic navigation metadata for the last completed automatic result save.

The result codec owns complete ``.temresult``/``.temsection`` packages and validates their
payloads on restoration. This independent index never copies or decodes array
data, never selects a named-library startup entry, and never publishes an
unfinished save. Its opaque intent ID coordinates competing save workers.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import errno
import json
import math
import os
from pathlib import Path
import re
import stat
from tempfile import NamedTemporaryFile
from threading import RLock
from time import monotonic, sleep
from uuid import uuid4


AUTOMATIC_RESULT_SCHEMA = "temsim-automatic-result-v1"
MAX_INDEX_BYTES = 64 * 1024
LOCK_TIMEOUT_SECONDS = 5.0
_ENTRY_FIELDS = frozenset({"path", "result_identity", "package_digest", "quality",
    "target_z_mm", "resumable_through_z_mm", "saved_at_utc", "size_bytes"})
_PACKAGE_SUFFIXES = frozenset({".temresult", ".temsection"})


def _identity(value, label, length=64):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise ValueError(f"Automatic result {label} must be a {length}-character hexadecimal identity")
    return value


def _path(value):
    if (not isinstance(value, str) or not value or len(value) > 32767
            or not value.isprintable() or not Path(value).is_absolute()
            or Path(value).suffix.lower() not in _PACKAGE_SUFFIXES):
        raise ValueError("Automatic result path must name an absolute .temresult or .temsection package")
    return Path(value)


def _validate_entry(entry):
    if not isinstance(entry, dict) or set(entry) != _ENTRY_FIELDS:
        raise ValueError("Automatic result entry fields do not match the current schema")
    _path(entry["path"])
    for key in ("result_identity", "package_digest"):
        _identity(entry[key], key)
    quality = entry["quality"]
    if (not isinstance(quality, str) or not quality.strip() or quality != quality.strip()
            or len(quality) > 128 or not quality.isprintable()):
        raise ValueError("Automatic result quality must be nonempty printable text")
    for key in ("target_z_mm", "resumable_through_z_mm"):
        value = entry[key]
        try:
            finite = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"Automatic result {key} must be a finite axial coordinate in mm")
    timestamp = entry["saved_at_utc"]
    if not isinstance(timestamp, str) or len(timestamp) > 64:
        raise ValueError("Automatic result save time must be an ISO UTC timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Automatic result save time must be an ISO UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Automatic result save time must include the UTC timezone")
    if type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] < 2**63:
        raise ValueError("Automatic result package size must be a positive integer")
    return entry


def _validate_index(document):
    if (not isinstance(document, dict)
            or set(document) != {"schema", "enabled", "intent_id", "entry"}
            or document["schema"] != AUTOMATIC_RESULT_SCHEMA):
        raise ValueError("Unsupported automatic result index schema")
    if type(document["enabled"]) is not bool:
        raise ValueError("Automatic result enabled setting must be an explicit boolean")
    if document["intent_id"] is not None:
        _identity(document["intent_id"], "save intent", 32)
        if not document["enabled"]:
            raise ValueError("A disabled automatic result index cannot have a save intent")
    if document["entry"] is not None:
        _validate_entry(document["entry"])
    return document


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key in automatic result index")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"Nonfinite JSON value in automatic result index: {value}")


@dataclass(frozen=True)
class AutomaticSaveToken:
    """Opaque identity of one save request, bound to its owning index."""
    intent_id: str
    result_identity: str
    index_path: str


class AutomaticResult:
    """Persist one completed result independently of the named result library.

    Begin before dispatching a save worker. Only its still-current token can
    commit the codec's completed export metadata. Disabling or cancelling
    invalidates in-flight work without deleting the previous completed entry.
    Reads return detached metadata and tolerate a moved package; loading and
    package-digest verification remain the result worker's responsibility.
    """

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("Automatic result root must be a directory")
        self.index_path = self.root / "automatic-result.json"
        self._lock = RLock()
        with self._transaction():
            self._read_index()

    @contextmanager
    def _transaction(self, *, create=False):
        deadline = monotonic() + LOCK_TIMEOUT_SECONDS
        message = f"Timed out waiting for automatic result index: {self.index_path}"
        if not self._lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            raise TimeoutError(message)
        descriptor = None
        try:
            if create:
                self.root.mkdir(parents=True, exist_ok=True)
            elif not self.root.exists():
                yield
                return
            lock_path = self.root / ".automatic-result.lock"
            if lock_path.is_symlink():
                raise ValueError("Automatic result lock must be an owned file")
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            if os.name == "nt":
                import msvcrt

                def acquire():
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                def acquire():
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                try:
                    acquire()
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError(message) from exc
                    sleep(min(.02, remaining))
            yield
        finally:
            try:
                if descriptor is not None:
                    # Never unlink: another process must lock the same inode.
                    os.close(descriptor)
            finally:
                self._lock.release()

    def _read_index(self):
        if self.index_path.is_symlink():
            raise ValueError("Automatic result index must be an owned JSON file, not a symbolic link")
        try:
            with self.index_path.open("rb") as stream:
                payload = stream.read(MAX_INDEX_BYTES + 1)
        except FileNotFoundError:
            return {"schema": AUTOMATIC_RESULT_SCHEMA, "enabled": True, "intent_id": None, "entry": None}
        except OSError as exc:
            if self.index_path.is_dir():
                raise ValueError("Automatic result index must be a JSON file") from exc
            raise
        if len(payload) > MAX_INDEX_BYTES:
            raise ValueError("Automatic result index exceeds its size limit")
        try:
            document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Malformed automatic result index") from exc
        return _validate_index(document)

    def _write_index(self, document):
        _validate_index(document)
        payload = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if len(payload) > MAX_INDEX_BYTES:
            raise ValueError("Automatic result index exceeds its size limit")
        temporary = None
        try:
            with NamedTemporaryFile(dir=self.root, prefix=".automatic-result-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.index_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def enabled(self) -> bool:
        with self._transaction():
            return self._read_index()["enabled"]

    def set_enabled(self, enabled: bool) -> None:
        if type(enabled) is not bool:
            raise ValueError("Automatic result enabled setting must be an explicit boolean")
        with self._transaction(create=True):
            document = self._read_index()
            document["enabled"] = enabled
            if not enabled:
                document["intent_id"] = None
            self._write_index(document)

    def begin_save(self, identity: str) -> AutomaticSaveToken | None:
        """Claim the newest intent without changing any completed entry."""
        _identity(identity, "result identity")
        with self._transaction(create=True):
            document = self._read_index()
            if not document["enabled"]:
                return None
            token = AutomaticSaveToken(uuid4().hex, identity, str(self.index_path))
            document["intent_id"] = token.intent_id
            self._write_index(document)
            return token

    def _matches(self, token, document):
        if token is None:
            return False
        if type(token) is not AutomaticSaveToken:
            raise ValueError("Automatic result save requires its original token")
        _identity(token.intent_id, "save intent", 32)
        _identity(token.result_identity, "result identity")
        return (token.index_path == str(self.index_path) and document["enabled"]
                and token.intent_id == document["intent_id"])

    def commit_save(self, token: AutomaticSaveToken | None, info: dict) -> bool:
        """Index a completed codec save only if its intent remains current."""
        with self._transaction(create=True):
            document = self._read_index()
            if not self._matches(token, document):
                return False
            required = {"path", "identity", "_package_digest", "quality", "target_z_mm",
                        "resumable_through_z_mm", "saved_at_utc"}
            if not isinstance(info, dict) or not required <= set(info):
                raise ValueError("Automatic result requires completed export metadata")
            if info["identity"] != token.result_identity:
                raise ValueError("Automatic result export does not match its requested identity")
            if not isinstance(info["path"], (str, Path)):
                raise ValueError("Automatic result requires an exported package path")
            path = _path(str(info["path"])).resolve()
            _path(str(path))
            try:
                file = path.stat()
            except FileNotFoundError as exc:
                raise ValueError("Automatic result exported package does not exist") from exc
            if not stat.S_ISREG(file.st_mode):
                raise ValueError("Automatic result exported package must be a regular file")
            entry = {"path": str(path), "result_identity": info["identity"],
                "package_digest": info["_package_digest"], "quality": info["quality"],
                "target_z_mm": info["target_z_mm"], "resumable_through_z_mm": info["resumable_through_z_mm"],
                "saved_at_utc": info["saved_at_utc"], "size_bytes": file.st_size}
            _validate_entry(entry)
            document.update(entry=entry, intent_id=None)
            self._write_index(document)
            return True

    def cancel_save(self, token: AutomaticSaveToken | None) -> bool:
        """Invalidate only this pending save, retaining the completed result."""
        with self._transaction(create=True):
            document = self._read_index()
            if not self._matches(token, document):
                return False
            document["intent_id"] = None
            self._write_index(document)
            return True

    def entry(self) -> dict | None:
        with self._transaction():
            entry = self._read_index()["entry"]
            return None if entry is None else dict(entry)
