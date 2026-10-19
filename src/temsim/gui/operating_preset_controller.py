"""Background assembly/preset calculations on isolated microscope states."""

from __future__ import annotations

from time import perf_counter
from threading import Event
from temsim import input_io

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.operating_modes import (
    apply_condenser_mode, apply_operating_mode_pair, apply_projector_mode, compatible_modes,
)


class _Signals(QObject):
    result = Signal(int, object, object, object, float)
    error = Signal(int, str)
    finished = Signal(int)


class _Worker(QRunnable):
    def __init__(self, generation, state_type, payload, catalog, selection,
                 condenser_key, projector_key, load_assembly, projector_only=False,
                 resolved_assembly=None):
        super().__init__()
        self.generation = generation
        self.state_type = state_type
        self.payload = payload
        self._archive_inputs = payload.get("archived_inputs") if projector_only else None
        self.catalog = catalog
        self.selection = selection
        self.condenser_key = condenser_key
        self.projector_key = projector_key
        self.load_assembly = load_assembly
        self.projector_only = projector_only
        self.resolved_assembly = resolved_assembly
        self.cancel_event = Event()
        self.signals = _Signals()

    @input_io.using_state_inputs
    def run(self):
        started = perf_counter()
        try:
            if self.cancel_event.is_set():
                return
            if self.projector_only:
                from temsim.instrument_snapshot import decode_instrument
                # Preserve the complete incident graph, including held drives,
                # edited geometry and source quadrature. A projector switch must
                # not rebuild the column from assembly defaults.
                state = decode_instrument(self.payload)
            else:
                if self.resolved_assembly is None:
                    state = self.state_type.from_dict(self.payload)
                else:
                    state = self.state_type.from_dict(
                        self.payload, resolved_assembly=self.resolved_assembly)
            state._tuning_cancelled = self.cancel_event.is_set
            if self.load_assembly:
                self.catalog.apply(state, self.selection)
            elif not self.projector_only:
                # This is a restored operating state, not a live TOML edit.
                # Rebinding its selected catalog must retain the saved field
                # widths even when that catalog has different defaults.
                for component in getattr(state, "stigmators", ()):
                    vars(component).pop("_manifest_effective_length_mm", None)
                apply_physical_layout_to_state(
                    state, assembly_root=self.catalog.root,
                    preserve_operating_parameters=True,
                )
            condenser_modes = compatible_modes(
                "condenser", self.selection.column, self.selection.recording
            )
            projector_modes = compatible_modes(
                "projector", self.selection.column, self.selection.recording
            )
            available = (
                (self.projector_only or self.condenser_key in {mode.key for mode in condenser_modes})
                and (self.projector_key is None
                     or self.projector_key in {mode.key for mode in projector_modes})
            )
            if not available and not self.load_assembly:
                raise ValueError("Selected assembly has no matching optical preset")
            result = None
            if available:
                arguments = dict(column_name=self.selection.column,
                                 recording_name=self.selection.recording)
                if self.projector_only:
                    result = apply_projector_mode(state, self.projector_key, **arguments)
                elif self.projector_key is None:
                    result = apply_condenser_mode(state, self.condenser_key, **arguments)
                else:
                    result = apply_operating_mode_pair(
                        state, self.condenser_key, self.projector_key, **arguments)
            if not self.projector_only:
                apply_physical_layout_to_state(state, assembly_root=self.catalog.root)
            if self.cancel_event.is_set():
                return
            if hasattr(state, "_tuning_cancelled"):
                del state._tuning_cancelled
            self.signals.result.emit(
                self.generation, state, self.selection, result,
                perf_counter() - started,
            )
        except Exception as exc:
            self.signals.error.emit(self.generation, str(exc))
        finally:
            self.signals.finished.emit(self.generation)


class OperatingPresetController(QObject):
    result_ready = Signal(object, object, object, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        from temsim.gui.job_coordinator import CoordinatedPool
        self.pool = CoordinatedPool(self)
        self.pool.setMaxThreadCount(1)
        self._generation = 0

    def submit(self, state, catalog, selection, condenser_key, projector_key,
               *, load_assembly=False, projector_only=False):
        if projector_only and (load_assembly or projector_key is None):
            raise ValueError("A projector-only calculation requires a projection and the current assembly")
        if projector_only:
            from temsim.instrument_snapshot import encode_instrument
            payload = encode_instrument(state)
        else:
            payload = state.to_dict()
        self.invalidate_pending()
        worker = _Worker(
            self._generation, type(state), payload, catalog, selection,
            condenser_key, projector_key, load_assembly, projector_only,
            resolved_assembly=None if projector_only else getattr(state, "_resolved_assembly", None),
        )
        from temsim.immutable_json import json_digest
        worker.job_input_identity = json_digest(dict(payload=worker.payload, selection=selection,
            condenser=condenser_key, projector=projector_key, load_assembly=load_assembly,
            projector_only=projector_only, resolved_assembly=worker.resolved_assembly))
        self._active_cancel = worker.cancel_event
        worker.signals.result.connect(self._accept_result)
        worker.signals.error.connect(self._accept_error)
        worker.signals.finished.connect(self._accept_finished)
        self.pool.start(worker)

    def invalidate_pending(self):
        if getattr(self, "_active_cancel", None) is not None:
            self._active_cancel.set()
        self._generation += 1
        self.pool.clear()

    def _accept_result(self, generation, state, selection, result, duration):
        if generation == self._generation:
            self.result_ready.emit(state, selection, result, duration)

    def _accept_error(self, generation, message):
        if generation == self._generation:
            self.failed.emit(message)

    def _accept_finished(self, generation):
        if generation == self._generation:
            self.finished.emit()
