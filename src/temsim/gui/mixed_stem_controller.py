"""Generation-guarded mixed STEM acquisition in the shared numerical queue."""
from dataclasses import asdict
from threading import Event
from time import monotonic
import math
from uuid import uuid4

from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from temsim.calculation_progress import CalculationProgress, callback_progress
from temsim.gui.job_coordinator import CoordinatedPool, ResourceClaim
from temsim.gui.mixed_stem_adapter import adapt_mixed_stem
from temsim.immutable_json import json_digest
from temsim.instrument_snapshot import capture_instrument_snapshot


def estimate_mixed_stem_host_bytes(state, request, existing_result=None):
    """Reserve host allocations, independently of each stage's GPU work limit.

    GPU source factors and resident FFT workspace keep their runtime admission
    checks. The physics planner owns the material grid and thickness ladder.
    New particle histories remain live during construction and scanning.
    """
    from temsim.physics.recording_stop import determine_tem_stop_z
    from temsim.physics.mixed_stem_scan import estimate_mixed_stem_host_bytes as material_host_plan
    count = int(state.electron_gun.emitter.ray_count)
    source = request.source
    source_host = 16*count**2+32*source.pixels**2*source.maximum_modes
    scan = state.ac_deflector
    half_x = (int(scan.scan_pixels_x)-1)*float(scan.scan_pixel_size_nm)*.5
    half_y = (int(scan.scan_lines)-1)*float(scan.scan_pixel_size_nm)*.5
    corners = ((float(state.sample.scan_origin_x_nm)-half_x,
                float(state.sample.scan_origin_y_nm)-half_y),
               (float(state.sample.scan_origin_x_nm)+half_x,
                float(state.sample.scan_origin_y_nm)+half_y))
    material_host = material_host_plan(state, request, positions_nm=corners)["estimated_host_peak_bytes"]
    reused = (getattr(existing_result, "state_snapshot", None) is not None
        and capture_instrument_snapshot(existing_result.state_snapshot).digest
            == capture_instrument_snapshot(state).digest)
    retained_growth = 0
    if not reused:
        step = max(float(state.step_mm), float(state.history_step_mm))
        span = max(0., float(determine_tem_stop_z(state))-float(state.electron_gun.exit_plane_z_mm))
        rows = math.ceil(span/step)+4
        # Completed display coordinates, exact section coordinates and clocks;
        # the bounded particle solver has its separate 2 GiB per-batch gate.
        retained_growth = rows*count*80+count*2048*6*8
    particle_stage = retained_growth+2*1024**3
    return int(max(particle_stage, retained_growth+source_host,
                   retained_growth+material_host)+256*1024**2)


class _Signals(QObject):
    ready=Signal(int,object,object,object,float)
    failed=Signal(int,str)
    progress=Signal(int,object)
    finished=Signal(int)


class MixedStemWorker(QRunnable):
    def __init__(self,generation,state,request,existing_result=None,*,maximum_host_bytes=None):
        super().__init__()
        self.generation,self.state,self.request=generation,state,request
        self.existing_result=existing_result
        self.cancel_event=Event()
        self.signals=_Signals()
        self.job_input_identity="mixed-stem:"+capture_instrument_snapshot(state).digest+":"+json_digest(asdict(request))
        estimated=estimate_mixed_stem_host_bytes(state,request,existing_result)
        self.resource_claim=ResourceClaim(estimated if maximum_host_bytes is None
            else min(estimated,maximum_host_bytes))
        self.progress_state=CalculationProgress("mixed-stem:"+uuid4().hex,"Mixed coherent STEM",
            "Preparing actual particles and CUDA acquisition",stages_completed=0,stages_total=1)
        self._last_progress=0.

    def _progress(self,done,total,message):
        if self.cancel_event.is_set():
            return
        update=callback_progress(self.progress_state,done,total,message)
        now=monotonic()
        if now-self._last_progress>=.05 or update.completed==0 or update.completed==update.total:
            self._last_progress=now
            self.progress_state=update
            self.signals.progress.emit(self.generation,update)

    def run(self):
        started=monotonic()
        try:
            from temsim.mixed_stem_workflow import calculate_mixed_stem
            from temsim.physics.host_memory_budget import host_memory_budget
            with host_memory_budget(self.resource_claim.working_bytes):
                acquisition=calculate_mixed_stem(self.state,request=self.request,existing_result=self.existing_result,
                    cancelled=self.cancel_event.is_set,progress_callback=self._progress)
            if self.cancel_event.is_set():
                return
            frame,diffraction=adapt_mixed_stem(acquisition,self.request)
            if not self.cancel_event.is_set():
                self.signals.ready.emit(self.generation,acquisition,frame,diffraction,monotonic()-started)
        except Exception as error:
            if not self.cancel_event.is_set():
                self.signals.failed.emit(self.generation,str(error))
        finally:
            from temsim.physics.wave_device import release_device_memory
            try:
                release_device_memory()
            finally:
                self.signals.finished.emit(self.generation)


class MixedStemScanController(QObject):
    started=Signal()
    ready=Signal(object,object,object,float)
    failed=Signal(str)
    progress_detail=Signal(object)
    finished=Signal()

    def __init__(self,state_provider,request_provider,parent=None):
        super().__init__(parent)
        self.state_provider,self.request_provider=state_provider,request_provider
        self.pool=CoordinatedPool(self)
        self.pool.coordinator.register_retained(self,"retained_roots")
        self.generation=0
        self.worker=None
        self.acquisition=self.frame=self.diffraction=None
        self._input_digest=self._request_digest=None
        self._draft_digest=self._draft_request_digest=None
        self._progress_state=None
        self.closed=False

    @property
    def busy(self):
        return self.worker is not None

    def retained_roots(self):
        return self.acquisition,self.frame,self.diffraction

    def _is_current(self):
        return (not self.closed and capture_instrument_snapshot(self.state_provider()).digest==self._draft_digest
            and json_digest(asdict(self.request_provider()))==self._draft_request_digest)

    def start(self,state,request,*,existing_result=None):
        if self.closed:
            return
        request.validate()
        captured=capture_instrument_snapshot(state)
        if int(state.electron_gun.emitter.ray_count)<1000:
            raise ValueError("Mixed coherent STEM needs at least 1000 original particles; increase the selected particle count")
        self.cancel()
        self._input_digest=captured.digest
        self._request_digest=json_digest(asdict(request))
        self._draft_digest=capture_instrument_snapshot(self.state_provider()).digest
        self._draft_request_digest=json_digest(asdict(self.request_provider()))
        self.generation+=1
        coordinator=self.pool.coordinator
        headroom=coordinator.ram_budget_bytes-coordinator.statistics()["retained_bytes"]
        # Input inventories may be registered after capture. Reserve room for
        # that overhead; runtime source/material admissions enforce this cap.
        maximum_host_bytes=headroom-256*1024**2
        if maximum_host_bytes<=0:
            raise ValueError("Shared job memory is occupied by retained results; clear data or adjust Performance and cache explicitly")
        worker=MixedStemWorker(self.generation,captured.restore(),request,existing_result,
            maximum_host_bytes=maximum_host_bytes)
        worker.signals.ready.connect(self._ready)
        worker.signals.failed.connect(self._failed)
        worker.signals.progress.connect(self._progress)
        worker.signals.finished.connect(self._finished)
        self.worker=worker
        self._progress(self.generation,worker.progress_state)
        self.started.emit()
        if self.worker is worker:
            self.pool.start(worker)

    def _terminal(self,status,message):
        if self._progress_state is not None and self._progress_state.active:
            self._progress_state=self._progress_state.terminal(status,message)
            self.progress_detail.emit(self._progress_state)

    def cancel(self,*_args):
        self.generation+=1
        old,self.worker=self.worker,None
        if old is not None:
            old.cancel_event.set()
        self.pool.clear()
        self._terminal("cancelled","Mixed coherent STEM request cancelled; previous complete image retained")
        if old is not None:
            self.finished.emit()

    @Slot(int,object,object,object,float)
    def _ready(self,generation,acquisition,frame,diffraction,seconds):
        if self.closed or self.worker is None or generation!=self.generation:
            return
        try:
            current=self._is_current()
        except Exception:
            current=False
        if not current:
            self._failed(generation,"Captured mixed scan inputs changed; no obsolete image was published")
            return
        self.acquisition,self.frame,self.diffraction=acquisition,frame,diffraction
        self._terminal("succeeded","Accepted mixed coherent STEM image and exit diffraction")
        self.ready.emit(acquisition,frame,diffraction,seconds)

    @Slot(int,str)
    def _failed(self,generation,message):
        if not self.closed and self.worker is not None and generation==self.generation:
            self._terminal("failed",message)
            self.failed.emit(message)

    @Slot(int,object)
    def _progress(self,generation,update):
        if not self.closed and self.worker is not None and generation==self.generation:
            if self._progress_state is not None and self._progress_state.operation_id==update.operation_id and not self._progress_state.active:
                return
            self._progress_state=update
            self.progress_detail.emit(update)

    @Slot(int)
    def _finished(self,generation):
        if self.worker is not None and generation==self.generation:
            self.worker=None
            self._terminal("cancelled","Request ended without a new accepted mixed scan")
            self.finished.emit()

    def shutdown(self,msecs=3000):
        if not self.closed:
            self.cancel()
            self.closed=True
        return self.pool.waitForDone(msecs)
