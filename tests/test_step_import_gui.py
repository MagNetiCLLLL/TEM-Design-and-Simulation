"""STEP editing uses the existing controls and only adopts current worker results."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PySide6.QtCore import QObject, QProcess, QTimer, Qt, Signal
from PySide6.QtWidgets import QApplication, QDialogButtonBox

from temsim.gui.cad_geometry_job import CadGeometryJob, CadGeometryProgress, request_digest
from temsim.gui.cad_step_import import StepImportDialog
from temsim.gui.part_model_editor import PartModelEditorPage
from temsim.part_model_document import PartModelDocument


ROOT = Path(__file__).resolve().parents[1]
CAD_PYTHON = ROOT / "environment/cad/Scripts/python.exe"


@pytest.fixture
def fake_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text("""import hashlib,json,pathlib,sys,time
r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf8'))
print('Worker ready',flush=True)
if r.get('wait'):time.sleep(30)
d=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
out=pathlib.Path(sys.argv[3]);out.mkdir()
result=dict(schema='temsim-cad-worker-result-v1',status='complete',request_id=r['request_id'],draft_digest=r.get('draft_digest'),request_sha256='wrong' if r.get('wrong') else d,result={'echo':r})
if r.get('fail'):result.update(status='failed',error='Known malformed solid')
(out/'result.json').write_text(json.dumps(result))
sys.exit(1 if r.get('fail') else 0)
""", encoding="utf-8")
    return worker


def test_worker_is_nonblocking_and_result_is_bound_to_immutable_request(qtbot, fake_worker):
    request = dict(op="import_step", path="input-µ.step", draft_digest="draft", marker=[1])
    job = CadGeometryJob(request, interpreter=sys.executable, worker=fake_worker)
    with qtbot.waitSignal(job.completed, timeout=10000) as result:
        job.start()
        assert job.busy
        request["marker"][0] = 9
    assert result.args[0]
    assert result.args[1]["result"]["echo"]["marker"] == [1]
    assert result.args[1]["request_sha256"] == request_digest(job.request)
    assert not job.process.processEnvironment().contains("PYTHONPATH")
    assert job.process.processEnvironment().value("PYTHONNOUSERSITE") == "1"
    assert job._temporary is None


@pytest.mark.parametrize("bad", ["wrong", "fail"])
def test_unrelated_or_failed_worker_never_returns_geometry(qtbot, fake_worker, bad):
    job = CadGeometryJob({"op": "import_step", bad: True}, interpreter=sys.executable, worker=fake_worker)
    with qtbot.waitSignal(job.completed, timeout=10000) as result:
        job.start()
    assert result.args[0] is False and result.args[1] is None
    assert "not changed" in result.args[2]


@pytest.mark.parametrize("method", ["reject", "close"])
def test_closing_progress_cancels_owned_process_and_retains_old_geometry(qtbot, fake_worker, method):
    job = CadGeometryJob({"op": "import_step", "wait": True}, interpreter=sys.executable, worker=fake_worker)
    progress = CadGeometryProgress(job, "Import STEP")
    qtbot.addWidget(progress)
    progress.show()
    job.start()
    qtbot.waitUntil(lambda: "Worker ready" in job.log_text, timeout=10000)
    with qtbot.waitSignal(job.completed, timeout=5000) as result:
        getattr(progress, method)()
    assert result.args[0] is False and result.args[1] is None
    assert job.cancelled and not job.busy
    assert job.process.state() == QProcess.ProcessState.NotRunning
    assert "unchanged" in result.args[2]


def test_import_dialog_defaults_to_preserved_coordinates_and_explicit_registration(qtbot, tmp_path):
    dialog = StepImportDialog(tmp_path / "Coil model.step", selected_part={"key": "coil", "name": "Coil", "local_center_z_mm": 42.}, existing_keys=["coil", "step_Coil_model"])
    qtbot.addWidget(dialog)
    assert dialog.mode.currentData() == "add"
    assert dialog.registration.currentData() == "preserve" and not dialog.registration.isEnabled()
    assert dialog.key.text() == "step_Coil_model_2"
    dialog.mode.setCurrentIndex(dialog.mode.findData("replace"))
    dialog.registration.setCurrentIndex(dialog.registration.findData("center"))
    assert dialog.key.isReadOnly() and dialog.key.text() == "coil"
    assert "does not infer or move an optical plane" in dialog.registration_note.text()
    qtbot.mouseClick(dialog.import_button, Qt.MouseButton.LeftButton)
    assert dialog.result_value["registration"] == "center"
    assert dialog.result_value["key"] == "coil"


@pytest.fixture(scope="module")
def actual_step(tmp_path_factory):
    if not CAD_PYTHON.is_file():
        pytest.skip("Optional isolated CAD environment is not installed")
    directory = tmp_path_factory.mktemp("actual_step_gui")
    path = directory / "translated.step"
    script = directory / "make.py"
    script.write_text("import cadquery as cq,sys\nsolid=cq.Workplane('XY').box(8,6,4).translate((20,-7,35))\ncq.exporters.export(solid,sys.argv[1])\n", encoding="utf-8")
    subprocess.run([str(CAD_PYTHON), str(script), str(path)], check=True, capture_output=True, timeout=60)
    request = directory / "request.json"
    request.write_text(json.dumps({"op": "import_step", "path": str(path)}), encoding="utf-8")
    subprocess.run([str(CAD_PYTHON), str(ROOT / "scripts/cad_geometry_worker.py"), str(request), "--output", str(directory / "import")], check=True, capture_output=True, timeout=60)
    receipt = json.loads((directory / "import/result.json").read_text())
    return path, receipt["result"]


def _vertices(page):
    return np.concatenate([row["vertices"] for row in page._mesh_records])


def _dimension(page, *tail):
    path = ("parts", page._selected_key, *tail)
    for row in range(page.dimensions.rowCount()):
        item = page.dimensions.item(row, 1)
        if tuple(item.data(Qt.ItemDataRole.UserRole)) == path:
            assert item.flags() & Qt.ItemFlag.ItemIsEditable
            return item
    raise AssertionError(f"Missing editor dimension {path}")


def _asset_page(qtbot, tmp_path, asset):
    path = tmp_path / "imported.toml"
    document = PartModelDocument.from_step_asset(asset, path, key="step_test", name="Imported test")
    document.save()
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(path, selected_key="step_test"), page.status.text()
    return page


def test_actual_qt_import_edit_hole_undo_reset_save_and_exact_export(qtbot, tmp_path, actual_step, monkeypatch):
    from temsim.gui import part_model_editor
    source, asset = actual_step
    path = tmp_path / "imported.toml"
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    monkeypatch.setattr(part_model_editor.QFileDialog, "getOpenFileName", lambda *_: (str(source), "STEP"))
    monkeypatch.setattr(part_model_editor.QFileDialog, "getSaveFileName", lambda *_: (str(path), "TOML"))
    qtbot.mouseClick(page.import_step_button, Qt.MouseButton.LeftButton)
    dialog = page._step_import_dialog
    qtbot.mouseClick(dialog.import_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._cad_geometry_job.finished, timeout=60000)
    assert page.session is not None, page.status.text()
    assert page.session.dirty and not path.exists()
    np.testing.assert_allclose(_vertices(page).min(axis=0), [16., -10., 33.], atol=1e-6)
    np.testing.assert_allclose(_vertices(page).max(axis=0), [24., -4., 37.], atol=1e-6)
    assert page.base_shape.currentData() == "step"
    key = page._selected_key
    source_json = page.session.part(key)["model_3d"]["base"]["asset_json"]
    nodes = [page.parameters.topLevelItem(i) for i in range(page.parameters.topLevelItemCount())]
    while nodes:
        node = nodes.pop()
        nodes.extend(node.child(i) for i in range(node.childCount()))
        if node.text(0) == "asset_json":
            assert "Embedded STEP source" in node.text(1)
            assert len(node.toolTip(1)) < 1000
            assert "brep_base64" not in node.text(1)
    _dimension(page, "model_3d", "base", "width_mm").setText("12")
    np.testing.assert_allclose(np.ptp(_vertices(page), axis=0), [12., 6., 4.], atol=1e-6)

    def fill_hole():
        modal = QApplication.activeModalWidget()
        assert modal is not None
        for i in range(3):
            modal.values[("center_mm", i)].setValue(0.)
        modal.values["diameter_mm"].setValue(2.)
        modal.values["depth_mm"].setValue(6.)
        modal.axis.setCurrentIndex(modal.axis.findData("z"))
        qtbot.mouseClick(modal.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    QTimer.singleShot(0, fill_hole)
    qtbot.mouseClick(page.add_hole_button, Qt.MouseButton.LeftButton)
    assert len(page.session.part(key)["model_3d"]["features"]) == 1, page.status.text()
    assert len(_vertices(page)) > len(asset["leaves"][0]["mesh"]["vertices_mm"])
    def fill_slot():
        modal = QApplication.activeModalWidget()
        for i, value in enumerate([4., 0., 0.]):
            modal.values[("center_mm", i)].setValue(value)
        modal.values["width_mm"].setValue(1.)
        modal.values["length_mm"].setValue(2.)
        modal.values["depth_mm"].setValue(6.)
        modal.axis.setCurrentIndex(modal.axis.findData("z"))
        qtbot.mouseClick(modal.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok), Qt.MouseButton.LeftButton)
    QTimer.singleShot(0, fill_slot)
    qtbot.mouseClick(page.add_slot_button, Qt.MouseButton.LeftButton)
    assert len(page.session.part(key)["model_3d"]["features"]) == 2, page.status.text()
    qtbot.mouseClick(page.save_button, Qt.MouseButton.LeftButton)
    assert path.is_file() and not page.session.dirty, page.status.text()
    saved = PartModelDocument(path)
    assert saved.part(key)["model_3d"]["base"]["asset_json"] == source_json
    assert len(saved.part(key)["model_3d"]["features"]) == 2

    # Use the existing export action: its actual OCC receipt must describe the
    # edited volume, not silently return the uncut original STEP body.
    monkeypatch.setattr(part_model_editor.QFileDialog, "getExistingDirectory", lambda *_: str(tmp_path))
    qtbot.mouseClick(page.export_step_button, Qt.MouseButton.LeftButton)
    export = page._cad_export_dialog
    assert export is not None, page.status.text()
    qtbot.waitUntil(lambda: not export.busy, timeout=60000)
    assert export.success, export.log.toPlainText()
    manifest = json.loads((export.output_directory / "step-manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert list(export.output_directory.rglob("*.step"))
    for exported in manifest["exports"]:
        assert exported["volume_mm3"] == pytest.approx(284. - 5. * np.pi, rel=1e-7)

    qtbot.mouseClick(page.reset_model_button, Qt.MouseButton.LeftButton)
    restored = page.session.part(key)["model_3d"]
    assert restored["base"]["width_mm"] == 8 and restored["features"] == []
    np.testing.assert_allclose(_vertices(page).min(axis=0), [16., -10., 33.], atol=1e-6)
    qtbot.mouseClick(page.undo_button, Qt.MouseButton.LeftButton)
    assert page.session.part(key)["model_3d"]["base"]["width_mm"] == 12
    assert len(page.session.part(key)["model_3d"]["features"]) == 2


class DeferredJob(QObject):
    completed = Signal(bool, object, str)
    output = Signal(str)

    def __init__(self, request, parent=None):
        super().__init__(parent)
        self.request = deepcopy(request)
        self.request.setdefault("request_id", "controlled-test")
        self.busy = False
        self.finished = False

    def start(self):
        self.busy = True

    def finish(self, receipt):
        self.busy = False
        self.finished = True
        self.completed.emit(True, receipt, "Complete")


def test_finished_import_cannot_replace_a_newer_draft(qtbot, tmp_path, actual_step, monkeypatch):
    from temsim.gui import cad_geometry_job
    source, asset = actual_step
    page = _asset_page(qtbot, tmp_path, asset)
    original = page.session
    snapshot = deepcopy(original.document)
    monkeypatch.setattr(cad_geometry_job, "CadGeometryJob", DeferredJob)
    page._start_step_import(dict(path=str(source), mode="replace", key="step_test", name="Imported test", registration="center"), original, snapshot)
    _dimension(page, "model_3d", "base", "width_mm").setText("11")
    new_draft = deepcopy(page.session.document)
    vertices = _vertices(page).copy()
    page._cad_geometry_job.finish({"result": asset})
    assert page.session is original and page.session.document == new_draft
    np.testing.assert_array_equal(_vertices(page), vertices)
    assert "after the draft changed" in page.status.text()


def test_actual_check_geometry_receipt_and_staleness_are_visible(qtbot, tmp_path, actual_step):
    _, asset = actual_step
    page = _asset_page(qtbot, tmp_path, asset)
    before = deepcopy(page.session.document)
    qtbot.mouseClick(page.geometry_check_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._cad_geometry_job.finished, timeout=60000)
    assert page._cad_validation_report is not None, page.status.text()
    assert page._cad_validation_report["can_apply"]
    assert not page._cad_validation_report["electromagnetic_fields_recomputed"]
    assert page.session.document == before
    report = page._cad_validation_dialog
    assert "Current draft" in report.report_status.text()
    _dimension(page, "model_3d", "base", "width_mm").setText("10")
    assert "stale" in report.report_status.text()


def test_geometry_check_captures_installed_context_and_rejects_a_later_context(qtbot, tmp_path, actual_step, monkeypatch):
    from temsim import cad_assembly_edit
    from temsim.gui import cad_geometry_job
    from temsim.gui.cad_export import draft_catalog
    _, asset = actual_step
    page = _asset_page(qtbot, tmp_path, asset)
    context = object()
    page._cad_context_assembly = context
    page._runtime_values = {"another_installed_aperture": {"radius_mm": .02}}
    captured = {}
    checks = []
    def capture(session, **kwargs):
        captured.update(kwargs)
        return draft_catalog(session.document, source=session.path)
    def current(_catalog, **kwargs):
        checks.append(kwargs)
        return kwargs["assembly"] is context
    monkeypatch.setattr(cad_assembly_edit, "build_current_assembly_edit_catalog", capture)
    monkeypatch.setattr(cad_assembly_edit, "assembly_edit_context_is_current", current)
    monkeypatch.setattr(cad_geometry_job, "CadGeometryJob", DeferredJob)
    qtbot.mouseClick(page.geometry_check_button, Qt.MouseButton.LeftButton)
    assert captured["assembly"] is context
    assert captured["runtime_values"] == page._runtime_values
    assert captured["coordinate_source_bytes"] == page._coordinate_source_bytes
    before = deepcopy(page.session.document)
    page._cad_context_assembly = object()
    page._cad_geometry_job.finish({})
    assert checks[-1]["full_source_check"] is True
    assert page._cad_validation_report is None
    assert page.session.document == before
    assert "installed assembly" in page.status.text()


def test_save_copy_invalidates_report_and_pending_jobs_even_with_identical_parameters(qtbot, tmp_path, actual_step, monkeypatch):
    from temsim.gui import cad_geometry_job, part_model_editor
    source, asset = actual_step
    page = _asset_page(qtbot, tmp_path, asset)
    qtbot.mouseClick(page.geometry_check_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: page._cad_geometry_job.finished, timeout=60000)
    assert page._cad_validation_report is not None, page.status.text()
    assert page._geometry_report_is_current()
    report = page._cad_validation_report
    session, before = page.session, deepcopy(page.session.document)
    destinations = iter([tmp_path / f"copy-{i}.toml" for i in range(3)])
    monkeypatch.setattr(part_model_editor.QFileDialog, "getSaveFileName", lambda *_: (str(next(destinations)), "TOML"))
    qtbot.mouseClick(page.save_copy_button, Qt.MouseButton.LeftButton)
    assert page.session is session and page.session.document == before
    assert not page._geometry_report_is_current()
    assert "stale" in page._cad_validation_dialog.report_status.text()

    monkeypatch.setattr(cad_geometry_job, "CadGeometryJob", DeferredJob)
    page._start_step_import(dict(path=str(source), mode="replace", key="step_test", name="Imported test", registration="center"), session, deepcopy(before))
    captured_path = str(session.path.resolve())
    assert page._cad_geometry_job.request["target_document_path"] == captured_path
    qtbot.mouseClick(page.save_copy_button, Qt.MouseButton.LeftButton)
    assert page.session.document == before and str(session.path.resolve()) != captured_path
    page._cad_geometry_job.finish({"result": asset})
    assert page.session is session and page.session.document == before
    assert "after the draft changed" in page.status.text()

    qtbot.mouseClick(page.geometry_check_button, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(page.save_copy_button, Qt.MouseButton.LeftButton)
    page._cad_geometry_job.finish({})
    assert page._cad_validation_report is report
    assert "target draft changed" in page.status.text()


def test_missing_geometry_engine_does_not_change_an_open_draft(qtbot, tmp_path, actual_step, monkeypatch):
    from temsim.gui import cad_geometry_job
    source, asset = actual_step
    page = _asset_page(qtbot, tmp_path, asset)
    before = deepcopy(page.session.document)
    monkeypatch.setattr(cad_geometry_job, "CadGeometryJob", lambda request, parent: CadGeometryJob(request, parent, interpreter=tmp_path / "missing.exe"))
    page._start_step_import(dict(path=str(source), mode="add", key="new", name="New", registration="preserve"), page.session, deepcopy(page.session.document))
    assert page.session.document == before
    assert "requirements-cad.txt" in page.status.text()
    assert _vertices(page).size > 0
