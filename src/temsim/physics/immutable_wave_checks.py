"""Bounded scalar-check reuse for genuinely immutable CPU wave amplitudes.

This cache is only for deterministic checks of ``checker(amplitude, limit)``.
It does not replace lattice, memory, field-coverage or cancellation checks.
GPU arrays and CPU arrays backed by potentially mutable storage always run
the original checker. No field, spectrum, exception or strong array reference
is retained.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import struct
from threading import RLock
from typing import Any, Callable
import weakref

import numpy as np


def _immutable_layout(amplitude: object) -> tuple | None:
    """Admit only a complex 2-D ndarray with an entirely read-only bytes base.

    ``writeable=False`` alone is insufficient: another view can still change
    an owning array, bytearray, memory map or memoryview. Even a read-only
    memoryview is deliberately excluded. Layout metadata can change without
    changing data bytes, so it also belongs to each cache key.
    """
    if (type(amplitude) is not np.ndarray or amplitude.ndim != 2
            or amplitude.dtype != np.dtype(np.complex128)):
        return None
    base = amplitude
    while type(base) is np.ndarray:
        if base.flags.writeable:
            return None
        base = base.base
    if type(base) is not bytes:
        return None
    return (amplitude.shape, amplitude.strides, amplitude.dtype.str,
            amplitude.__array_interface__["data"][0])


@dataclass(frozen=True, slots=True)
class _Entry:
    amplitude: weakref.ReferenceType
    checker: weakref.ReferenceType
    value: float | bool


class ImmutableWaveCheckCache:
    """Reuse successful scalar checks without extending array lifetimes.

    ``checker`` must be a stable, deterministic callable whose complete
    numerical inputs are the amplitude and the exact Python-float limit.
    Different checker identities and different float bit patterns never
    share a result. Non-weak-referenceable callables bypass the cache.

    Only finite built-in floats and literal ``True`` are stored. Exceptions,
    false results, NumPy scalars and other return values are passed through
    unchanged and never cached. Concurrent calls may compute the same check
    twice; the checker itself is never run while holding the cache lock.
    """

    def __init__(self, maximum_entries: int = 32) -> None:
        if type(maximum_entries) is not int or maximum_entries < 1:
            raise ValueError("maximum_entries must be a positive integer")
        self._maximum_entries = maximum_entries
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._lock = RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        """Forget scalar receipts without touching any wave storage."""
        with self._lock:
            self._entries.clear()

    def check(self, amplitude: object, limit: float,
              checker: Callable[[Any, float], Any]) -> Any:
        layout = _immutable_layout(amplitude)
        if layout is None or type(limit) is not float or not math.isfinite(limit):
            return checker(amplitude, limit)
        try:
            weakref.ref(checker)
        except TypeError:
            return checker(amplitude, limit)
        key = (id(amplitude), id(checker), struct.pack("!d", limit), layout)
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                if entry.amplitude() is amplitude and entry.checker() is checker:
                    self._entries.move_to_end(key)
                    return entry.value
                # Identity is checked independently of ids, including when
                # an expired object's id has been reused by the interpreter.
                del self._entries[key]

        value = checker(amplitude, limit)
        if not (value is True or (type(value) is float and math.isfinite(value))):
            return value
        if _immutable_layout(amplitude) != layout:
            return value

        owner = weakref.ref(self)

        def discard(reference: weakref.ReferenceType) -> None:
            cache = owner()
            if cache is not None:
                with cache._lock:
                    current = cache._entries.get(key)
                    if (current is not None
                            and (current.amplitude is reference or current.checker is reference)):
                        del cache._entries[key]

        entry = _Entry(weakref.ref(amplitude, discard), weakref.ref(checker, discard), value)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._maximum_entries:
                self._entries.popitem(last=False)
        return value
