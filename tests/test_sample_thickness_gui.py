"""Fixed-entrance thickness editing through actual GUI and geometry code.

These tests prepare geometry only; no particles or waves are propagated.
"""
import pytest
from PySide6.QtCore import Qt

from temsim.gui.parameter_panel import ParameterPanel
from temsim.gui.sample_panel import SamplePage
from temsim.gui.sample_panel import SampleSceneView
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.immutable_json import thaw_json
from temsim.optics.column import default_state
from temsim.runtime_parameters import runtime_targets
from temsim.specimen.geometry import quaternion_from_euler_xyz_deg,quaternion_to_matrix
from types import SimpleNamespace
import numpy as np


def make_page(qtbot, state=None):
    state=default_state() if state is None else state
    page=SamplePage()
    qtbot.addWidget(page)
    page.set_state(state)
    return page,state


def thickness_item(panel):
    return next(panel.runtime_table.item(row,1) for row in range(panel.runtime_table.rowCount())
        if panel.runtime_table.item(row,0).text()=="thickness_nm")


def assembly_geometry(state):
    return {part.key:(thaw_json(part.data),part.start_z_mm,part.center_z_mm,part.end_z_mm)
        for part in state._resolved_assembly.parts if part.key!="sample"}


def test_surface_readouts_are_copyable_and_viewing_does_not_edit_inputs(qtbot):
    state=default_state()
    before=capture_instrument_snapshot(state).digest
    page,_=make_page(qtbot,state)
    assert capture_instrument_snapshot(state).digest==before
    assert page.upper_surface_position.text()==f"{state.sample.upper_surface_z_mm:.17g} mm"
    assert page.downstream_surface_position.text()==f"{state.sample.lower_surface_z_mm:.17g} mm"
    for label in (page.upper_surface_position,page.downstream_surface_position):
        assert label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        assert label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByKeyboard
    assert "upper surface fixed" in page.scalar_controls["thickness_nm"].toolTip()


def test_thickness_editor_preserves_entrance_and_all_other_parts_across_repeated_edits(qtbot):
    page,state=make_page(qtbot)
    upper=state.sample.upper_surface_z_mm
    sample=state.sample
    geometry=assembly_geometry(state)
    changed=[]
    errors=[]
    page.parameters_changed.connect(changed.append)
    page.error.connect(errors.append)
    for thickness in (10.,2.,7.5,10.):
        page.scalar_controls["thickness_nm"].setValue(thickness)
        assert not errors
        assert state.sample is sample
        assert state.sample.thickness_nm==thickness
        assert state.sample.upper_surface_z_mm==upper
        assert state.sample.lower_surface_z_mm==upper+thickness*1e-6
        assert state._resolved_assembly.part("sample").center_z_mm==pytest.approx(
            state.sample.z_mm,abs=1e-11)
        assert assembly_geometry(state)==geometry
        assert page.upper_surface_position.text()==f"{upper:.17g} mm"
        assert page.downstream_surface_position.text()==f"{state.sample.lower_surface_z_mm:.17g} mm"
    assert changed==["sample.thickness_nm"]*4


@pytest.mark.parametrize("value",[0.,-1.,1e9,float("nan")])
def test_rejected_thickness_keeps_input_objects_and_completed_frame(qtbot,value):
    page,state=make_page(qtbot)
    before=capture_instrument_snapshot(state).digest
    old_thickness=state.sample.thickness_nm
    sample=state.sample
    snapshot=object()
    result=object()
    page._snapshot=snapshot
    page._result=result
    page.calculation_bar.status.setText("Completed result displayed.")
    changed=[]
    errors=[]
    page.parameters_changed.connect(changed.append)
    page.error.connect(errors.append)
    if value==value:
        page.scalar_controls["thickness_nm"].setValue(value)
    else:
        page._set_scalar("thickness_nm",value)
    assert errors and not changed
    assert capture_instrument_snapshot(state).digest==before
    assert state.sample is sample
    assert page.scalar_controls["thickness_nm"].value()==old_thickness
    assert page._snapshot is snapshot and page._result is result
    assert page.calculation_bar.status.text()=="Completed result displayed."


def test_selected_legacy_sample_captures_its_existing_upper_face_on_first_edit(qtbot):
    state=default_state()
    object.__setattr__(state.sample,"upper_surface_reference_z_mm",None)
    object.__setattr__(state.sample,"lattice_reference_thickness_nm",None)
    upper=state.sample.upper_surface_z_mm
    page,_=make_page(qtbot,state)
    assert state.sample.upper_surface_reference_z_mm is None
    page.scalar_controls["thickness_nm"].setValue(10.)
    assert state.sample.thickness_nm==10.
    assert state.sample.upper_surface_reference_z_mm==upper
    assert state.sample.upper_surface_z_mm==upper


def test_generic_sample_edit_uses_the_same_atomic_operation(qtbot,monkeypatch):
    from temsim.specimen import axial_geometry
    state=default_state()
    panel=ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context("Sample",runtime_targets(state)["sample"],None,(),None,state=state)
    item=thickness_item(panel)
    assert item.flags() & Qt.ItemFlag.ItemIsEditable
    upper=state.sample.upper_surface_z_mm
    geometry=assembly_geometry(state)
    calls=[]
    original=axial_geometry.set_sample_thickness
    def execute(current,value):
        calls.append((current,value))
        return original(current,value)
    monkeypatch.setattr(axial_geometry,"set_sample_thickness",execute)
    errors=[]
    panel.error.connect(errors.append)
    item.setText("10")
    assert not errors
    assert calls==[(state,10.)]
    assert state.sample.thickness_nm==10. and state.sample.upper_surface_z_mm==upper
    assert assembly_geometry(state)==geometry


@pytest.mark.parametrize("context",["missing","stale"])
def test_generic_sample_without_matching_state_is_read_only(qtbot,context):
    state=default_state()
    panel=ParameterPanel()
    qtbot.addWidget(panel)
    supplied=None if context=="missing" else default_state()
    panel.set_context("Sample",runtime_targets(state)["sample"],None,(),None,state=supplied)
    item=thickness_item(panel)
    assert not item.flags() & Qt.ItemFlag.ItemIsEditable
    before=capture_instrument_snapshot(state).digest
    errors=[]
    panel.error.connect(errors.append)
    item.setText("10")
    assert errors
    assert capture_instrument_snapshot(state).digest==before
    assert item.text()==str(state.sample.thickness_nm)


def test_generic_invalid_thickness_keeps_original_value_and_reference(qtbot):
    state=default_state()
    panel=ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context("Sample",runtime_targets(state)["sample"],None,(),None,state=state)
    before=capture_instrument_snapshot(state).digest
    errors=[]
    panel.error.connect(errors.append)
    thickness_item(panel).setText("1000000000")
    assert errors and capture_instrument_snapshot(state).digest==before
    assert thickness_item(panel).text()==str(state.sample.thickness_nm)


from test_working_point_restore_gui import window


def test_window_passes_live_state_to_generic_sample_context(window):
    window.assembly_panel.select_key("sample")
    panel=window.parameter_panel
    assert panel._runtime_target.key=="sample"
    assert panel._runtime_state is window.state
    assert thickness_item(panel).flags() & Qt.ItemFlag.ItemIsEditable


def test_window_keeps_both_thickness_editors_and_geometry_on_the_applied_state(window,monkeypatch):
    window.assembly_panel.select_key("sample")
    monkeypatch.setattr(window,"schedule_preview",lambda *_a,**_k:None)
    page=window.workspace.sample_page
    upper=window.state.sample.upper_surface_z_mm
    thickness_item(window.parameter_panel).setText("10")
    assert window.state.sample.thickness_nm==10.
    assert page.scalar_controls["thickness_nm"].value()==10.
    assert page.upper_surface_position.text()==f"{upper:.17g} mm"
    assert window.assembly is window.state._resolved_assembly
    assert window.workspace.physical_layout.assembly_3d._assembly is window.assembly
    page.scalar_controls["thickness_nm"].setValue(2.)
    assert window.state.sample.thickness_nm==2.
    assert thickness_item(window.parameter_panel).text()=="2.0"
    assert window.state.sample.upper_surface_z_mm==upper


def test_orientation_preview_preserves_the_lab_axial_lattice_reference():
    current=quaternion_to_matrix(quaternion_from_euler_xyz_deg((10.,20.,30.)))
    target=quaternion_to_matrix(quaternion_from_euler_xyz_deg((40.,-25.,0.)))
    lattice=np.asarray(((0.,0.,0.),(.1,.2,.3),(-.4,.2,-.1)))
    phase=np.array((0.,0.,-2.5))
    positions=lattice@current.T+phase
    snapshot=SimpleNamespace(atom_positions_nm=positions,orientation_matrix=current,
        lattice_translation_z_nm=-2.5)
    rotated=SampleSceneView._oriented_atoms(snapshot,target)
    np.testing.assert_allclose(rotated,lattice@target.T+phase,atol=1e-15)
    np.testing.assert_array_equal(rotated[0],phase)
    np.testing.assert_array_equal(snapshot.atom_positions_nm,positions)


def test_retained_gl_rotation_pivots_about_the_fixed_phase_origin(qtbot):
    from PySide6.QtGui import QVector3D
    scene=SampleSceneView()
    qtbot.addWidget(scene)
    matrices=[]
    item=SimpleNamespace(setTransform=matrices.append)
    scene.opengl_available=True
    scene._atom_item=item
    scene._bond_item=None
    scene._base_rotation=np.eye(3)
    scene._shown_rotation=np.eye(3)
    scene._shown_probe=None
    target=quaternion_to_matrix(quaternion_from_euler_xyz_deg((35.,25.,0.)))
    snapshot=SimpleNamespace(centre_nm=(2.,3.,0.),lattice_translation_z_nm=-2.5,
        current_probe_nm=None)
    scene._update_retained_model(snapshot,target)
    phase_origin=QVector3D(2.,3.,-2.5)
    mapped=matrices[0].map(phase_origin)
    np.testing.assert_allclose((mapped.x(),mapped.y(),mapped.z()),(2.,3.,-2.5),atol=1e-6)


@pytest.mark.parametrize("explicit",[False,True])
def test_explicit_saved_sample_placement_anchors_only_the_detached_candidate(window,monkeypatch,explicit):
    from temsim.manifest_editor import ManifestTarget
    from temsim.specimen.axial_geometry import set_sample_thickness
    window.state=default_state()
    window.assembly=window.state._resolved_assembly
    previous=window.state
    upper=previous.sample.upper_surface_z_mm
    sample=window.assembly.part("sample")
    target=ManifestTarget(sample.source_file,"sample")
    delta=4.8e-6
    updates={('parts','sample','local_center_z_mm'):sample.data['local_center_z_mm']+delta} if explicit else {
        ('parts','sample','colour'):'#123456'}
    candidate=capture_instrument_snapshot(previous).restore()
    monkeypatch.setattr(window,"_copy_state_for_assembly",lambda:candidate)
    monkeypatch.setattr(window.manifest_editor,"save",lambda *_:{})
    observed=[]
    def apply(current,*_args,**_kwargs):
        assert window.state is previous and previous.sample.upper_surface_z_mm==upper
        observed.append((current,current.sample.upper_surface_z_mm))
        return current._resolved_assembly
    monkeypatch.setattr(window.catalog,"apply",apply)
    for name in ('_invalidate_direct_alignment','_refresh_assembly_views','_mark_operating_preset_stale','schedule_preview'):
        monkeypatch.setattr(window,name,lambda *_a,**_k:None)
    monkeypatch.setattr(window.assembly_panel,'select_key',lambda *_:None)
    assert window._save_manifest_updates(target,updates,report_error=False)
    assert observed and observed[0][0] is candidate
    centre=(sample.center_z_mm-float(sample.data['local_center_z_mm'])
            +float(updates[('parts','sample','local_center_z_mm')])) if explicit else sample.center_z_mm
    expected=centre-previous.sample.thickness_nm*.5e-6 if explicit else upper
    assert candidate.sample.upper_surface_z_mm==expected
    assert previous.sample.upper_surface_z_mm==upper
    set_sample_thickness(candidate,10.)
    assert candidate.sample.upper_surface_z_mm==expected
