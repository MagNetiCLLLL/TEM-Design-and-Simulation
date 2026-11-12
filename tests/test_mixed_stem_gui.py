"""Mixed STEM GUI ownership with pure arrays and substituted numerical execution."""
from dataclasses import asdict,replace
from threading import Event
from types import SimpleNamespace
from hashlib import sha256
import json

import numpy as np
import pytest

from temsim.gui.job_coordinator import GIB
from temsim.gui.mixed_stem_adapter import adapt_mixed_stem,wrap_mixed_stem_result
from temsim.gui.mixed_stem_controller import MixedStemScanController
from temsim.gui.scan_panel import ScanControlView
from temsim.immutable_json import freeze_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state
from temsim.physics.mixed_stem_scan import MixedStemScanRequest
from temsim.simulation_pipeline import CalculationResult


def state():
    value=default_state()
    value.electron_gun.emitter.ray_count=16384
    value.ac_deflector.scan_pixels_x=3
    value.ac_deflector.scan_lines=2
    value.ac_deflector.scan_pixel_size_nm=.02
    return value


def acquisition(inputs,request):
    nx,ny=inputs.ac_deflector.scan_pixels_x,inputs.ac_deflector.scan_lines
    pitch=inputs.ac_deflector.scan_pixel_size_nm
    x,y=np.meshgrid((np.arange(nx)-(nx-1)*.5)*pitch,
                    (np.arange(ny)-(ny-1)*.5)*pitch)
    bf=np.linspace(.1,.105,nx*ny).reshape(ny,nx)
    adf=np.full_like(bf,.02)
    checkpoint=SimpleNamespace(reference_current_a=1e-5,record={"original_positive_count":16384,
        "source_fraction":.25},beam=SimpleNamespace(total_weight=.249999),digest="c"*64)
    fractions={"bf":bf,"adf":adf,"other":.25-bf-adf,"numerical_band_loss":np.full_like(bf,1e-6)}
    q=np.linspace(-1.,1.,4)
    scan=SimpleNamespace(scan_x_nm=x,scan_y_nm=y,fractions=fractions,
        current_a={key:values*checkpoint.reference_current_a for key,values in fractions.items()},
        diffraction_probability=np.eye(4)*.01,qx_inv_angstrom=q,qy_inv_angstrom=q,
        source_checkpoint=checkpoint,record=freeze_json({"request":asdict(request),
        "diffraction_position_nm":[0.,.01],"input_digest":capture_instrument_snapshot(inputs).digest}))
    result=CalculationResult(simulation=SimpleNamespace(),energy_filter=None,state_snapshot=inputs)
    return SimpleNamespace(state_snapshot=inputs,particle_result=result,scan=scan)


def saved_acquisition(inputs,request):
    from temsim.mixed_stem_artifact import SavedSourceSummary,IntensityReference
    actual=acquisition(inputs,request)
    original=actual.scan.source_checkpoint
    actual.scan.source_checkpoint=SavedSourceSummary(original.reference_current_a,original.record,
        original.digest,IntensityReference(original.beam.total_weight),{})
    actual.particle_result=None
    actual.read_only=True
    actual.saved_directory="checked-export"
    return actual


def write_mock_saved_acquisition(directory,inputs,*,historical=False):
    """Pure-array artifact fixture; no particle or coherent physics is run."""
    from temsim.immutable_json import thaw_json
    directory.mkdir()
    actual=acquisition(inputs,MixedStemScanRequest())
    scan=actual.scan
    scan.fractions['other']-=scan.fractions['numerical_band_loss']
    scan.fractions['source_mode_loss']=scan.fractions['numerical_band_loss'].copy()
    scan.fractions['numerical_band_loss']=np.zeros_like(scan.scan_x_nm)
    scan.fractions['source_representation_loss']=np.zeros_like(scan.scan_x_nm)
    scan.fractions['upstream_physical_loss']=np.full_like(scan.scan_x_nm,.75)
    scan.current_a={key:value*1e-5 for key,value in scan.fractions.items()}
    arrays=dict(scan_x_nm=scan.scan_x_nm,scan_y_nm=scan.scan_y_nm,
        qx_inv_angstrom=scan.qx_inv_angstrom,qy_inv_angstrom=scan.qy_inv_angstrom,
        diffraction_probability=scan.diffraction_probability)
    arrays.update({"fraction_"+key:value for key,value in scan.fractions.items()})
    arrays.update({"current_a_"+key:value for key,value in scan.current_a.items()})
    np.savez_compressed(directory/"scan-and-diffraction.npz",**arrays)
    count=16384
    ids=np.arange(count,dtype=np.int64)
    weights=np.full(count,1/count)
    gun=dict(ray_id=ids,weight=weights,alive=np.ones(count,bool),
        x_m=np.zeros(count),y_m=np.zeros(count),tx_rad=np.zeros(count),ty_rad=np.zeros(count),
        energy_offset_ev=np.zeros(count),flight_time_s=np.full(count,1e-8),
        blocked_z_mm=np.full(count,np.nan),blocked_key=np.full(count,"",dtype=str))
    np.savez_compressed(directory/"original-gun.npz",**gun)
    surviving=count//4
    source={key:np.asarray(gun[key][:surviving]) for key in
        ("ray_id","weight","x_m","y_m","tx_rad","ty_rad","flight_time_s")}
    source.update(kinetic_energy_ev=np.full(surviving,300000.),
        canonical_frequency_shift_inv_angstrom=np.zeros((surviving,2)))
    np.savez_compressed(directory/"entrance-source-arrays.npz",**source)
    report=thaw_json(scan.record)
    snapshot=capture_instrument_snapshot(inputs)
    if historical:
        from temsim.instrument_snapshot import InstrumentSnapshot
        graph=thaw_json(snapshot.graph)
        for node in graph['nodes']:
            if node.get('type')=='temsim.optics.model:Sample':
                for key in ('upper_surface_reference_z_mm','lattice_reference_thickness_nm'):
                    node['attributes'].pop(key,None)
                    node['fields']=[name for name in node['fields'] if name!=key]
        snapshot=InstrumentSnapshot(graph,snapshot.external_inputs,'historical-ui-fixture')
    report.update(schema="mixed-particle-stem-scan-v1",reference_current_a=1e-5,
        original_positive_count=count,checkpoint_digest="c"*64,input_digest=snapshot.digest)
    source_record=dict(scan.source_checkpoint.record,retained_source_fraction=.249999)
    for filename,value in (("report.json",report),("source-checkpoint.json",source_record),
            ("inputs.json",snapshot.to_dict())):
        (directory/filename).write_text(json.dumps(value),encoding="utf-8")
    (directory/"checksums.json").write_text(json.dumps({path.name:sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()}),encoding="utf-8")
    return actual


@pytest.fixture
def panel(qtbot):
    result=ScanControlView()
    qtbot.addWidget(result)
    result.set_state(state())
    result.resize(1200,850)
    result.show()
    return result


def test_method_selection_never_enables_retired_wave_or_changes_inputs(panel):
    before=panel._state.to_dict()
    panel.stem_scan_method.setCurrentIndex(panel.stem_scan_method.findData("mixed_coherent_cuda"))
    assert panel.mixed_coherent_selected()
    assert not panel._state.sample.stem_wave_enabled
    assert panel._state.to_dict()==before
    assert not panel.stem_particle_model.isEnabled()
    panel.stem_scan_method.setCurrentIndex(0)
    assert panel.stem_particle_model.isEnabled()


def test_mixed_method_clears_a_retired_request_only_after_user_action(panel):
    panel._state.sample.stem_wave_enabled=True
    panel.set_state(panel._state)
    panel.stem_scan_method.setCurrentIndex(1)
    before=panel._state.to_dict()
    assert panel._state.sample.stem_wave_enabled
    assert panel.clear_retired_wave_request.text()=="Clear retired wave request"
    changed=[]
    panel.parameters_changed.connect(changed.append)
    panel.clear_retired_wave_request.click()
    expected=before
    expected['sample']['stem_wave_enabled']=False
    assert panel._state.to_dict()==expected
    assert changed==["sample.stem_wave_enabled"]
    assert panel.mixed_coherent_selected()


def test_adapter_preserves_absolute_reference_and_current_and_raster():
    inputs=state()
    request=MixedStemScanRequest()
    actual=acquisition(inputs,request)
    frame,diffraction=adapt_mixed_stem(actual)
    np.testing.assert_array_equal(frame.fractions["bf"],actual.scan.fractions["bf"])
    np.testing.assert_array_equal(frame.current_pa["haadf"],actual.scan.current_a["adf"]*1e12)
    assert frame.metrics["ideal_exit_angle_detectors_mrad"]=={"bf":[0.,15.],"haadf":[50.,150.]}
    assert frame.expected_electrons["bf"].shape==(2,3)
    assert diffraction.position_nm==[0.,.01]
    assert frame.metrics["mixed_exit_diffraction"]["probability"].shape==(4,4)
    wrapped=wrap_mixed_stem_result(actual,frame)
    assert wrapped.stem_scan is frame and wrapped.workflow=="stem"
    assert wrapped.signatures["stem"]!=actual.particle_result.signatures.get("stem")
    assert not wrapped.cache_hit


def test_adapter_does_not_mask_relative_error_at_tiny_current():
    inputs=state()
    request=MixedStemScanRequest()
    actual=acquisition(inputs,request)
    actual.scan.source_checkpoint.reference_current_a=1e-30
    actual.scan.current_a={key:values*1e-30 for key,values in actual.scan.fractions.items()}
    adapt_mixed_stem(actual,request)
    actual.scan.current_a["bf"]*=1.01
    with pytest.raises(ValueError,match="current and original"):
        adapt_mixed_stem(actual,request)


def test_adapter_rejects_a_changed_display_request():
    request=MixedStemScanRequest()
    actual=acquisition(state(),request)
    changed=replace(request,source=replace(request.source,source_image_rms_pm=12.))
    with pytest.raises(ValueError,match="executed numerical metadata"):
        adapt_mixed_stem(actual,changed)


def test_adapter_rejects_an_incomplete_selected_raster():
    request=MixedStemScanRequest()
    actual=acquisition(state(),request)
    actual.scan.scan_x_nm=actual.scan.scan_x_nm[:1]
    actual.scan.scan_y_nm=actual.scan.scan_y_nm[:1]
    with pytest.raises(ValueError,match="complete selected"):
        adapt_mixed_stem(actual)


@pytest.mark.parametrize("shape",[(1,1),(4,4)])
def test_invalid_saved_frequency_axes_are_rejected_before_display(panel,shape):
    actual=saved_acquisition(panel._state,MixedStemScanRequest())
    actual.scan.diffraction_probability=np.zeros(shape)
    actual.scan.qx_inv_angstrom=np.zeros(shape[1])
    actual.scan.qy_inv_angstrom=np.zeros(shape[0])
    previous=panel._stem_frame
    with pytest.raises(ValueError,match="frequency"):
        adapt_mixed_stem(actual)
    assert panel._stem_frame is previous and panel._mixed_diffraction is None


def test_panel_renders_ideal_labels_and_selected_exit_pattern_without_calculation(panel):
    request=MixedStemScanRequest()
    actual=acquisition(panel._state,request)
    frame,diffraction=adapt_mixed_stem(actual,request)
    panel._set_stem_frame(frame,state_snapshot=actual.state_snapshot)
    np.testing.assert_array_equal(panel.detector_image_items["bf"].image,frame.fractions["bf"].T)
    assert panel.detector_image_items["df"].image is None
    assert panel.detector_title_labels["haadf"].text()=="ADF"
    assert "Ideal exit-angle window 50–150" in panel.detector_geometry_labels["haadf"].text()
    assert "Z " not in panel.detector_geometry_labels["haadf"].text()
    assert "0, 0.01 nm" in panel.exit_diffraction_summary.text()
    assert "not a raster average" in panel.exit_diffraction_summary.text()
    source=diffraction.probability.copy()
    panel.exit_diffraction_scale.setCurrentIndex(1)
    np.testing.assert_array_equal(panel.exit_diffraction_image.image,source.T)
    np.testing.assert_array_equal(diffraction.probability,source)
    panel.mark_stem_frame_stale()
    assert "Previous pattern | inputs changed" in panel.exit_diffraction_summary.text()
    panel.display_result(None,complete=True)
    assert panel.exit_diffraction_image.image is None
    assert panel._mixed_diffraction is None


def test_saved_mixed_view_keeps_live_controls_and_never_creates_ray_history(panel,monkeypatch):
    inputs=state()
    inputs.step_mm=.03125
    inputs.sample.stem_poisson_enabled=True
    actual=saved_acquisition(inputs,MixedStemScanRequest())
    monkeypatch.setattr(np.random,"default_rng",lambda *_:pytest.fail("Loading cannot generate new Poisson samples"))
    frame,diffraction=adapt_mixed_stem(actual)
    assert frame.poisson_counts is None
    live=capture_instrument_snapshot(panel._state).digest
    previous=panel._result
    panel.display_saved_mixed(frame,diffraction,state_snapshot=inputs,directory="checked-export")
    assert capture_instrument_snapshot(panel._state).digest==live
    assert panel._result is previous
    assert panel.image_source.currentText()=="Saved mixed scan (read-only)"
    assert "read-only" in panel.saved_mixed_notice.text()
    assert "working points remain unchanged" in panel.saved_mixed_notice.text()
    assert "Saved read-only pattern" in panel.exit_diffraction_summary.text()
    assert not panel.export_mixed_scan.isEnabled()
    np.testing.assert_array_equal(panel.detector_image_items["bf"].image,actual.scan.fractions["bf"].T)
    with pytest.raises(ValueError,match="ray history"):
        wrap_mixed_stem_result(actual,frame)
    fresh=acquisition(panel._state,MixedStemScanRequest())
    new_frame,_=adapt_mixed_stem(fresh)
    panel._set_stem_frame(new_frame,state_snapshot=panel._state)
    assert panel.saved_mixed_notice.isHidden()
    assert panel.image_source.currentText()=="Current calculation"


def test_loading_saved_mixed_frame_replaces_the_empty_playback_banner(panel,qtbot):
    assert "No STEM frame calculated" in panel.detector_playback_summary.text()
    actual=saved_acquisition(panel._state,MixedStemScanRequest())
    frame,diffraction=adapt_mixed_stem(actual)
    panel.display_saved_mixed(frame,diffraction,state_snapshot=actual.state_snapshot,
                             directory="checked-export")
    qtbot.wait(1)
    summary=panel.detector_playback_summary.text()
    assert "Saved mixed frame loaded; read-only" in summary
    assert "No STEM frame" not in summary
    assert "Click Calculate STEM" not in summary
    assert "BF mean fraction" in summary
    assert not panel._playback_timer.isActive()


@pytest.mark.parametrize("saved",[False,True])
def test_mixed_frame_summary_names_bf_adf_without_relabeling_particle_haadf(panel,saved):
    actual=(saved_acquisition if saved else acquisition)(panel._state,MixedStemScanRequest())
    frame,diffraction=adapt_mixed_stem(actual)
    if saved:
        panel.display_saved_mixed(frame,diffraction,state_snapshot=actual.state_snapshot,
                                 directory="checked-export")
    else:
        panel._set_stem_frame(frame,state_snapshot=actual.state_snapshot)
        panel._set_playback_active(False)
    summary=panel.detector_playback_summary.text()
    assert "Mixed coherent BF/ADF" in summary
    assert "near_specimen_mixed_coherent_cuda" not in summary
    assert "HAADF" not in summary
    signal=frame.detector_signals["haadf"]
    assert f"ADF mean fraction {signal.fraction:.5g} ({signal.current_pa:.5g} pA)" in summary
    particle=replace(frame,metrics={**frame.metrics,"model":"material_particle_paths"})
    particle_summary=panel._stem_frame_summary("Particle frame",frame=particle)
    assert "material_particle_paths" in particle_summary
    assert "HAADF mean fraction" in particle_summary
    assert "Mixed coherent BF/ADF" not in particle_summary


def test_load_mixed_button_requests_inspection_without_editing_state(panel):
    requested=[]
    edits=[]
    before=panel._state.to_dict()
    panel.load_mixed_scan_requested.connect(lambda:requested.append(True))
    panel.parameters_changed.connect(edits.append)
    panel.load_mixed_scan.click()
    assert requested==[True] and not edits
    assert panel._state.to_dict()==before


@pytest.fixture
def controller(qtbot):
    inputs=state()
    request=MixedStemScanRequest()
    owner=MixedStemScanController(lambda:inputs,lambda:request)
    yield owner,inputs,request
    assert owner.shutdown(5000)


def test_default_request_uses_real_coordinated_admission_without_lowering_budgets(controller,qtbot,monkeypatch):
    from temsim import mixed_stem_workflow
    owner,inputs,request=controller
    inputs.sample.thickness_nm=10.
    inputs.ac_deflector.scan_pixels_x=inputs.ac_deflector.scan_lines=64
    owner.pool.coordinator.configure(ram_budget_bytes=24*GIB)
    entered=[]
    def execute(captured,**kwargs):
        entered.append((captured.electron_gun.emitter.ray_count,captured.step_mm,kwargs['request']))
        from temsim.physics.host_memory_budget import check_host_allocation
        assert owner.worker.resource_claim.working_bytes<24*GIB
        with pytest.raises(MemoryError,match="admitted"):
            check_host_allocation(owner.worker.resource_claim.working_bytes+1)
        return acquisition(captured,kwargs['request'])
    monkeypatch.setattr(mixed_stem_workflow,"calculate_mixed_stem",execute)
    ready=[]
    owner.ready.connect(lambda *args:ready.append(args))
    owner.start(inputs,request)
    qtbot.waitUntil(lambda:bool(ready),timeout=10000)
    qtbot.waitUntil(lambda:not owner.busy)
    assert entered[0][:2]==(16384,inputs.step_mm)
    assert entered[0][2].maximum_working_bytes==24*GIB
    assert entered[0][2].source.maximum_working_bytes==24*GIB


def test_explicit_normalized_numerics_are_not_confused_with_live_draft(controller,qtbot,monkeypatch):
    from temsim import mixed_stem_workflow
    owner,inputs,request=controller
    captured=capture_instrument_snapshot(inputs).restore()
    captured.history_step_mm=.5
    captured.step_mm=.03125
    captured._tuning_quality="Preview"
    observed=[]
    def execute(snapshot,**kwargs):
        observed.append((snapshot.electron_gun.emitter.ray_count,snapshot.step_mm,snapshot.history_step_mm,snapshot._tuning_quality))
        return acquisition(snapshot,request)
    monkeypatch.setattr(mixed_stem_workflow,"calculate_mixed_stem",execute)
    ready=[]
    owner.ready.connect(lambda *args:ready.append(args))
    owner.start(captured,request)
    qtbot.waitUntil(lambda:bool(ready),timeout=10000)
    assert observed==[(16384,.03125,.5,"Preview")]
    assert inputs.step_mm!=.03125


def test_cancel_and_late_callbacks_cannot_publish_or_resurrect_progress(controller,qtbot,monkeypatch):
    from temsim import mixed_stem_workflow
    owner,inputs,request=controller
    started,release=Event(),Event()
    def execute(snapshot,**kwargs):
        started.set()
        assert release.wait(5)
        return acquisition(snapshot,request)
    monkeypatch.setattr(mixed_stem_workflow,"calculate_mixed_stem",execute)
    ready,progress=[],[]
    owner.ready.connect(lambda *args:ready.append(args))
    owner.progress_detail.connect(progress.append)
    owner.start(inputs,request)
    try:
        qtbot.waitUntil(started.is_set,timeout=10000)
        worker=owner.worker
        owner.cancel()
        assert worker.cancel_event.is_set()
        worker.signals.ready.emit(worker.generation,acquisition(inputs,request),None,None,1.)
        worker.signals.progress.emit(worker.generation,worker.progress_state)
        assert not ready and progress[-1].status=="cancelled"
    finally:
        release.set()
    assert owner.pool.waitForDone(5000)
    assert not ready


def test_changed_live_inputs_reject_completed_old_frame(controller,qtbot,monkeypatch):
    from temsim import mixed_stem_workflow
    owner,inputs,request=controller
    pending=[]
    monkeypatch.setattr(owner.pool,"start",pending.append)
    errors=[]
    owner.failed.connect(errors.append)
    owner.start(inputs,request)
    worker=owner.worker
    inputs.sample.thickness_nm+=1.
    worker.signals.ready.emit(worker.generation,acquisition(worker.state,request),None,None,1.)
    assert errors and "inputs changed" in errors[-1]
    assert owner.acquisition is None


def test_worker_cuda_rejection_precedes_particle_work_and_preserves_previous_frame(controller,qtbot,monkeypatch):
    from temsim.physics import compute_backend
    from temsim import simulation_pipeline
    owner,inputs,request=controller
    monkeypatch.setattr(compute_backend,"coherent_cuda_capability",lambda:
        compute_backend.BackendCapability(False,"Mocked unavailable device"))
    monkeypatch.setattr(simulation_pipeline,"calculate",lambda *a,**k:pytest.fail("CUDA guard must precede particles"))
    previous=object()
    owner.frame=previous
    errors=[]
    owner.failed.connect(errors.append)
    owner.start(inputs,request)
    qtbot.waitUntil(lambda:bool(errors),timeout=10000)
    assert "CUDA" in errors[-1]
    assert owner.frame is previous


def test_low_particle_count_rejected_without_a_worker(controller):
    owner,inputs,request=controller
    inputs.electron_gun.emitter.ray_count=49
    with pytest.raises(ValueError,match="1000 original"):
        owner.start(inputs,request)
    assert owner.worker is None


def test_window_request_uses_shared_source_and_clears_loaded_hold(window,monkeypatch):
    page=window.workspace.coherent_beam
    source=page._source_controls["source_image_rms_pm"]
    source.setValue(12.)
    window.high_rays.lineEdit().setText("16384")
    window.high_rays.lineEdit().setModified(True)
    window.high_step.lineEdit().setText("0.03125")
    window.high_step.lineEdit().setModified(True)
    window.state._tuning_quality="Preview"
    window.result_files.hold_automatic_preview=True
    calls=[]
    monkeypatch.setattr(window.mixed_stem,"start",lambda *args,**kwargs:calls.append((args,kwargs)))
    window.workspace.scan_control.stem_scan_method.setCurrentIndex(1)
    window.run_page_calculation("stem")
    assert not window.result_files.hold_automatic_preview
    assert len(calls)==1
    inputs,request=calls[0][0]
    assert inputs.electron_gun.emitter.ray_count==16384 and inputs.step_mm==.03125
    assert inputs._tuning_quality=="Preview"
    assert request.source.source_image_rms_pm==12.
    assert not window.state.sample.stem_wave_enabled


def test_window_saved_load_does_not_publish_particles_or_require_cuda(window,monkeypatch):
    from temsim import mixed_stem_artifact
    from temsim.gui import main_window
    from temsim.physics import compute_backend,mixed_wave_particles
    inputs=capture_instrument_snapshot(window.state).restore()
    inputs.ac_deflector.scan_pixels_x=3
    inputs.ac_deflector.scan_lines=2
    actual=saved_acquisition(inputs,MixedStemScanRequest())
    before=capture_instrument_snapshot(window.state).digest
    previous=window.workspace._last_result
    high=window.workspace._high_accuracy_result
    monkeypatch.setattr(main_window.QFileDialog,"getExistingDirectory",lambda *_:"checked-export")
    monkeypatch.setattr(mixed_stem_artifact,"load_saved_acquisition",lambda path:actual)
    monkeypatch.setattr(compute_backend,"require_coherent_cuda",lambda *_:pytest.fail("Inspection needs no CUDA"))
    monkeypatch.setattr(mixed_wave_particles,"particle_source_from_result",lambda *_a,**_k:pytest.fail("Inspection cannot create particles"))
    errors=[]
    monkeypatch.setattr(window,"_show_error",errors.append)
    window._load_mixed_stem()
    assert not errors
    assert capture_instrument_snapshot(window.state).digest==before
    assert window.workspace._last_result is previous
    assert window.workspace._high_accuracy_result is high
    assert window.workspace._saved_mixed_acquisition is actual
    assert window.workspace.coherent_beam.particle_result_provider() is previous
    assert window.workspace.scan_control._stem_frame.metrics["read_only_saved_acquisition"]
    assert "no ray history loaded" in window.workspace.scan_control.calculation_bar.status.text()


@pytest.mark.parametrize("historical",[False,True])
def test_window_uses_checked_artifact_loader_for_a_read_only_scan(window,monkeypatch,tmp_path,historical):
    from temsim.gui import main_window
    from temsim.physics import compute_backend,mixed_wave_particles
    from temsim.mixed_stem_artifact import SavedSourceSummary
    inputs=state()
    actual=write_mock_saved_acquisition(tmp_path/"saved",inputs,historical=historical)
    original_files={path.name:path.read_bytes() for path in (tmp_path/"saved").iterdir()}
    before=capture_instrument_snapshot(window.state).digest
    previous=window.workspace._last_result
    high=window.workspace._high_accuracy_result
    monkeypatch.setattr(main_window.QFileDialog,"getExistingDirectory",lambda *_:str(tmp_path/"saved"))
    monkeypatch.setattr(compute_backend,"require_coherent_cuda",lambda *_:pytest.fail("Reading needs no CUDA"))
    monkeypatch.setattr(mixed_wave_particles,"particle_source_from_result",lambda *_a,**_k:pytest.fail("Cannot create particle history"))
    errors=[]
    monkeypatch.setattr(window,"_show_error",errors.append)
    window._load_mixed_stem()
    assert not errors
    assert capture_instrument_snapshot(window.state).digest==before
    assert window.workspace._last_result is previous and window.workspace._high_accuracy_result is high
    loaded=window.workspace._saved_mixed_acquisition
    assert loaded.particle_result is None and loaded.read_only
    assert isinstance(loaded.scan.source_checkpoint,SavedSourceSummary)
    assert not hasattr(loaded.scan.source_checkpoint.beam,"modes")
    frame=window.workspace.scan_control._stem_frame
    np.testing.assert_array_equal(frame.fractions['bf'],actual.scan.fractions['bf'])
    np.testing.assert_array_equal(frame.fractions['haadf'],actual.scan.fractions['adf'])
    assert not window.workspace.scan_control.export_mixed_scan.isEnabled()
    assert {path.name:path.read_bytes() for path in (tmp_path/"saved").iterdir()}==original_files
    if historical:
        assert loaded.input_snapshot.implementation=='historical-ui-fixture'
        assert loaded.input_compatibility['physical_inputs_unchanged']
        assert loaded.input_compatibility['metadata_migrations']
        assert frame.metrics['saved_input_digest']==json.loads(original_files['report.json'])['input_digest']


@pytest.mark.parametrize("cancelled",[False,True])
def test_window_cancelled_or_invalid_saved_load_keeps_the_display(window,monkeypatch,cancelled):
    from temsim import mixed_stem_artifact
    from temsim.gui import main_window
    before=capture_instrument_snapshot(window.state).digest
    previous=window.workspace.scan_control._stem_frame
    errors=[]
    monkeypatch.setattr(window,"_show_error",errors.append)
    monkeypatch.setattr(main_window.QFileDialog,"getExistingDirectory",lambda *_:"" if cancelled else "invalid-export")
    def invalid(_path):
        if cancelled:
            pytest.fail("Cancelling the directory picker cannot read files")
        raise ValueError("Checksum mismatch")
    monkeypatch.setattr(mixed_stem_artifact,"load_saved_acquisition",invalid)
    window._load_mixed_stem()
    assert capture_instrument_snapshot(window.state).digest==before
    assert window.workspace.scan_control._stem_frame is previous
    assert bool(errors) is not cancelled
    if not cancelled:
        assert "Checksum mismatch" in errors[0]


from test_working_point_restore_gui import window
