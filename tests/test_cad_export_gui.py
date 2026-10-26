"""STEP export snapshots drafts, stays nonblocking and isolates CAD materials."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from PySide6.QtCore import QProcess

from temsim.gui.cad_export import CadExportDialog, draft_catalog
from temsim.gui.part_model_editor import PartModelEditorPage


ROOT = Path(__file__).resolve().parents[1]
CATALOG = {"schema": "temsim-analytic-material-cad-v1", "units": "mm", "modules": []}


@pytest.fixture
def exporter(tmp_path):
    script = tmp_path / "exporter.py"
    script.write_text("""import hashlib,json,pathlib,sys,time
catalog=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf8'))
out=pathlib.Path(sys.argv[3]);out.mkdir(exist_ok=True)
print('Exporting test.step',flush=True)
if catalog.get('wait'):time.sleep(30)
if catalog.get('fail'):raise RuntimeError('Known geometry failure')
digest=hashlib.sha256(json.dumps(catalog,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
(out/'step-manifest.json').write_text(json.dumps({'status':'passed','catalog_sha256':digest}))
""", encoding="utf-8")
    return script


def test_export_process_is_async_retains_exact_draft_and_verifies_manifest(qtbot, tmp_path, exporter):
    catalog = deepcopy(CATALOG)
    catalog["draft_marker"] = {"radius": 7.25}
    dialog = CadExportDialog(catalog, tmp_path / "output", project_root=ROOT,
                             interpreter=sys.executable, exporter=exporter)
    qtbot.addWidget(dialog)
    with qtbot.waitSignal(dialog.completed, timeout=10000) as signal:
        dialog.start()
        assert dialog.busy
        assert dialog.process.processEnvironment().value("PYTHONNOUSERSITE") == "1"
        assert not dialog.process.processEnvironment().contains("PYTHONPATH")
        catalog["draft_marker"]["radius"] = 99
    assert signal.args[0] is True
    assert dialog.success and not dialog.busy
    saved = json.loads((dialog.output_directory / "catalog-input.json").read_text())
    assert saved["draft_marker"]["radius"] == 7.25
    assert "Exporting test.step" in dialog.log.toPlainText()
    assert dialog._temporary is None


def test_export_error_does_not_claim_success(qtbot, tmp_path, exporter):
    dialog = CadExportDialog({**CATALOG, "fail": True}, tmp_path / "failed", project_root=ROOT,
                             interpreter=sys.executable, exporter=exporter)
    qtbot.addWidget(dialog)
    with qtbot.waitSignal(dialog.completed, timeout=10000) as signal:
        dialog.start()
    assert signal.args[0] is False
    assert "Known geometry failure" in dialog.log.toPlainText()
    assert not dialog.success


@pytest.mark.parametrize("close_method", ["close", "reject"])
def test_close_cancels_only_owned_export_and_keeps_input_receipt(qtbot, tmp_path, exporter, close_method):
    dialog = CadExportDialog({**CATALOG, "wait": True}, tmp_path / "cancelled", project_root=ROOT,
                             interpreter=sys.executable, exporter=exporter)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.start()
    qtbot.waitUntil(lambda: "Exporting" in dialog.log.toPlainText(), timeout=10000)
    with qtbot.waitSignal(dialog.completed, timeout=6000) as signal:
        getattr(dialog, close_method)()
    assert signal.args[0] is False
    assert dialog.cancelled and not dialog.busy
    assert dialog.process.state() == QProcess.ProcessState.NotRunning
    assert (dialog.output_directory / "catalog-input.json").exists()
    assert "incomplete" in dialog.status.text()


def test_missing_optional_environment_gives_setup_without_changing_main_python(qtbot, tmp_path, monkeypatch):
    monkeypatch.delenv("TEMSIM_CAD_PYTHON", raising=False)
    dialog = CadExportDialog(CATALOG, tmp_path / "absent", project_root=tmp_path)
    qtbot.addWidget(dialog)
    with pytest.raises(FileNotFoundError, match="requirements-cad.txt"):
        dialog.start()
    assert dialog.process.state() == QProcess.ProcessState.NotRunning
    assert not dialog.output_directory.exists()


@pytest.fixture
def installed_cad_package(tmp_path):
    """A wheel layout with no scripts directory or importable simulator."""
    package = tmp_path / "simulator environment" / "Lib" / "site-packages" / "temsim"
    (package / "gui").mkdir(parents=True)
    for name in ("cad_geometry_worker.py", "cad_export_worker.py", "cad_kernel.py",
                 "cad_step.py", "cad_assembly_validation.py"):
        shutil.copyfile(ROOT / "src" / "temsim" / name, package / name)
    (package / "__init__.py").write_text("raise RuntimeError('Simulator must stay unloaded')\n", encoding="utf-8")
    return package


def test_installed_gui_uses_packaged_cad_entries_and_configured_python(qtbot, tmp_path, monkeypatch, installed_cad_package):
    from temsim.gui import cad_export, cad_geometry_job
    package = installed_cad_package
    install_root = package.parents[2]
    monkeypatch.setenv("TEMSIM_CAD_PYTHON", sys.executable)
    for module in (cad_export, cad_geometry_job):
        monkeypatch.setattr(module, "__file__", str(package / "gui" / Path(module.__file__).name))
        monkeypatch.setattr(module, "cad_process_root", lambda: install_root)
    job = cad_geometry_job.CadGeometryJob({"op": "installation_probe"})
    assert job.interpreter == Path(sys.executable).resolve()
    assert job.worker == package / "cad_geometry_worker.py"
    with qtbot.waitSignal(job.completed, timeout=10000) as signal:
        job.start()
    assert signal.args[0] is False
    assert job.receipt["error"]["message"] == "Unknown CAD operation: installation_probe"
    assert job.receipt["request_sha256"] == job.request_sha256
    dialog = cad_export.CadExportDialog({**CATALOG, "units": "m"}, tmp_path / "export")
    qtbot.addWidget(dialog)
    assert dialog.interpreter == job.interpreter
    assert dialog.exporter == package / "cad_export_worker.py"
    with qtbot.waitSignal(dialog.completed, timeout=10000) as signal:
        dialog.start()
    assert signal.args[0] is False
    assert "CAD catalogs must explicitly use millimetres" in dialog.log.toPlainText()
    assert not (install_root / "scripts").exists()


def test_cad_runtime_root_tracks_installation_not_configuration_override(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from temsim import cad_runtime
    monkeypatch.setenv("TEMSIM_PROJECT_ROOT", str(tmp_path / "custom-configs"))
    assert cad_runtime.cad_process_root() == ROOT
    installed_root = tmp_path / "simulator environment"
    package = installed_root / "Lib" / "site-packages" / "temsim"
    monkeypatch.setattr(cad_runtime, "__file__", str(package / "cad_runtime.py"))
    monkeypatch.setattr(cad_runtime, "sys", SimpleNamespace(prefix=str(installed_root)))
    assert cad_runtime.cad_process_root() == installed_root


def test_cad_interpreter_argument_overrides_environment_and_preserves_source_default(qtbot, tmp_path, monkeypatch):
    import os
    from temsim.gui.cad_geometry_job import CadGeometryJob
    monkeypatch.setenv("TEMSIM_CAD_PYTHON", str(tmp_path / "configured-python"))
    job = CadGeometryJob({}, interpreter=sys.executable)
    assert job.interpreter == Path(sys.executable).resolve()
    dialog = CadExportDialog(CATALOG, tmp_path / "output", interpreter=sys.executable)
    qtbot.addWidget(dialog)
    assert dialog.interpreter == job.interpreter
    monkeypatch.delenv("TEMSIM_CAD_PYTHON")
    default_job = CadGeometryJob({}, project_root=ROOT)
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    assert default_job.interpreter == ROOT / "environment" / "cad" / executable


@pytest.mark.parametrize("source_entry", [False, True])
def test_cad_entries_need_only_stdlib_until_an_operation_requests_cad(tmp_path, installed_cad_package, source_entry):
    """Run actual entry points without site packages or editable import hooks."""
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"op": "installation_probe", "request_id": "probe"}), encoding="utf-8")
    worker = (ROOT / "scripts" / "cad_geometry_worker.py" if source_entry else
              installed_cad_package / "cad_geometry_worker.py")
    process = subprocess.run([sys.executable, "-I", "-S", str(worker), str(request), "--output", str(tmp_path)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert process.returncode == 1, process.stdout + process.stderr
    receipt = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert receipt["error"]["message"] == "Unknown CAD operation: installation_probe"
    assert receipt["request_id"] == "probe"
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({**CATALOG, "units": "m"}), encoding="utf-8")
    exporter = (ROOT / "scripts" / "export_cad_models.py" if source_entry else
                installed_cad_package / "cad_export_worker.py")
    process = subprocess.run([sys.executable, "-I", "-S", str(exporter), str(catalog), "--output", str(tmp_path)],
                             cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert process.returncode == 1
    assert "CAD catalogs must explicitly use millimetres" in process.stderr


def test_draft_catalog_passes_displayed_runtime_and_aperture_without_mutation(monkeypatch):
    from temsim import cad_catalog
    captured = {}
    def build(document, **kwargs):
        captured.update(document=document, **kwargs)
        return {"solids": [{"key": "part"}], "key": "module"}
    monkeypatch.setattr(cad_catalog, "module_catalog", build)
    document = {"parts": [{"key": "part", "length_mm": 9.5}]}
    runtime = {"part": {"radius_mm": .1}}
    result = draft_catalog(document, source="draft.toml", runtime_values=runtime, aperture_indices={"part": 2})
    assert captured["runtime_values"] == runtime
    assert captured["aperture_indices"] == {"part": 2}
    assert captured["document"] == document
    document["parts"][0]["length_mm"] = 10
    runtime["part"]["radius_mm"] = 100
    assert captured["document"]["parts"][0]["length_mm"] == 9.5
    assert captured["runtime_values"]["part"]["radius_mm"] == .1
    assert result["units"] == "mm"


def test_new_cad_region_selection_is_readonly_without_disabling_dimensions(qtbot):
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    path = ROOT / "configs/instruments/column/C3_ProbeCorrector.toml"
    assert page.open_path(path, selected_key="probe_hp2_hexapole")
    before = deepcopy(page.session.document)
    names = [page.region.itemData(i) for i in range(page.region.count())]
    picked = next(name for name in names if str(name).startswith("pole_"))
    page.select_part("probe_hp2_hexapole", region=picked)
    assert page._selected_region == picked
    assert "Read-only reconstruction" in page.material_current.text()
    assert "Unsupported material region" not in page.material_current.text()
    assert not page.assign_button.isEnabled()
    assert not page.material.isEnabled()
    assert page.region.isEnabled()
    assert page.dimensions.rowCount() > 0
    page.assign_material()
    assert page.session.document == before
    assert page._mesh_records
    page.select_part("condenser_lens_3_excitation_coil")
    assert page.assign_button.isEnabled()
    assert page.material.isEnabled()
    assert page.export_step_button.isEnabled()
