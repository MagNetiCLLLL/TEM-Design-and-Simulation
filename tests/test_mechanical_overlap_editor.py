"""Dimension edits refresh geometry and show exact material pairs, offscreen."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import tomli_w
from PySide6.QtCore import Qt

from temsim.component_operations import make_component
from temsim.gui.mechanical_overlap_panel import MechanicalOverlapPanel
from temsim.gui.part_model_editor import PartModelEditorPage


def editor(qtbot, tmp_path):
    parts = [make_component(key=key, shape="box", width_mm=2., height_mm=3.,
                            length_mm=4., center_z_mm=z)
             for key, z in (("block_a", 0.), ("block_b", 5.), ("block_c", 10.))]
    document = dict(format_version=1, coordinate_system="module_local_z_mm",
                    module=dict(key="mechanical:overlap_test", type="mechanical", name="Overlap fixture"),
                    geometry=dict(length_mm=20.), parts=parts)
    path = tmp_path / "parts.toml"
    path.write_text(tomli_w.dumps(document), encoding="utf-8")
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    page.resize(1100, 800)
    assert page.open_path(path, selected_key="block_a")
    page.show()
    return page, path


def dimension(page, tail):
    path = ("parts", "block_a", *tail)
    for row in range(page.dimensions.rowCount()):
        item = page.dimensions.item(row, 1)
        if tuple(item.data(Qt.ItemDataRole.UserRole)) == path:
            return item
    raise AssertionError(path)


def vertices(page):
    return np.concatenate([mesh["vertices"] for mesh in page._mesh_records if mesh["key"] == "block_a"])


def pair(kind="intersection", volume=6.):
    side = lambda key: dict(key=key, region="body", module_key="mechanical:test")
    return dict(first=side("block_a"), second=side("block_b"), kind=kind,
                intersection_volume_mm3=volume)


def test_panel_lists_material_pairs_and_does_not_call_contacts_collisions(qtbot):
    panel = MechanicalOverlapPanel()
    qtbot.addWidget(panel)
    report = dict(scope="Fixture material only", geometry_report=dict(assemblies=[dict(
        collisions=[pair()], contacts=[pair("contact", 0)],
        uncertain_occupancy_intersections=[pair()], sub_tolerance_intersections=[],
        unresolved_solids=[dict(key="unknown")])]))
    panel.set_report(report, names={"block_a": "First block"})
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "First block"
    assert panel.table.item(0, 4).text() == "6"
    assert "1 surface contacts" in panel.status.text()
    assert "1 unresolved solids" in panel.status.text()
    with qtbot.waitSignal(panel.component_requested) as signal:
        panel._activate(0, 2)
    assert signal.args == ["block_b", "body"]
    panel.set_report(report, current=False)
    assert panel.table.rowCount() == 0 and "stale" in panel.status.text()


def test_rapid_size_edits_refresh_mesh_and_coalesce_the_background_request(qtbot, tmp_path, monkeypatch):
    page, source = editor(qtbot, tmp_path)
    raw = source.read_bytes()
    before = deepcopy(page.session.document)
    calls = []
    monkeypatch.setattr(page, "check_geometry", lambda **kw: calls.append((deepcopy(page.session.document), kw)))
    dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    assert np.ptp(vertices(page)[:, 2]) == pytest.approx(8.)
    dimension(page, ("model_3d", "base", "length_mm")).setText("10")
    assert np.ptp(vertices(page)[:, 2]) == pytest.approx(10.)
    assert not calls and "Waiting" in page.overlaps.status.text()
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=3000)
    assert calls[0][1] == {"automatic": True}
    assert calls[0][0]["parts"][0]["length_mm"] == 10.
    assert page.session.document["parts"][1:] == before["parts"][1:]
    assert source.read_bytes() == raw
    page.undo()
    assert np.ptp(vertices(page)[:, 2]) == pytest.approx(8.)
    page.redo()
    assert np.ptp(vertices(page)[:, 2]) == pytest.approx(10.)
    page.overlaps.auto_check.setChecked(False)
    assert not page._overlap_timer.isActive()


def test_rejected_edit_clears_current_overlap_evidence(qtbot, tmp_path):
    page, _ = editor(qtbot, tmp_path)
    page.overlaps.auto_check.setChecked(False)
    before = deepcopy(page.session.document)
    dimension(page, ("model_3d", "base", "length_mm")).setText("-1")
    assert page.session.document == before
    assert "Edit rejected" in page.overlaps.status.text()
    assert page.overlaps.table.rowCount() == 0


def test_check_now_lists_buildable_conflicts_with_auto_disabled_and_keeps_manual_ui(qtbot, monkeypatch):
    from temsim.cad_optical_binding import canonical_digest
    from temsim.gui.cad_geometry_job import CadGeometryJob
    from temsim.paths import INSTRUMENT_CONFIG_ROOT

    source = INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml"
    raw = source.read_bytes()
    page = PartModelEditorPage()
    qtbot.addWidget(page)
    assert page.open_path(source, selected_key="condenser_lens_3_housing")
    page.show()
    page.overlaps.auto_check.setChecked(False)
    page.session.set_dimension(("parts", "condenser_lens_3_housing", "mechanical_inner_diameter_mm"), 169.)
    draft = deepcopy(page.session.document)
    with pytest.raises(ValueError, match="Mechanical radial layers overlap"):
        page.session.validate()

    jobs = []
    def capture_start(job):
        job.busy = True
        jobs.append(job)
    monkeypatch.setattr(CadGeometryJob, "start", capture_start)
    page.check_geometry()  # The separate, strict Check geometry action remains strict.
    assert not jobs and "Check unavailable" in page.overlaps.status.text()

    page.overlaps.check_now.click()
    assert len(jobs) == 1
    job = jobs[0]
    assert not job.automatic_overlap_check
    assert page._cad_geometry_dialog.isVisible()
    assert not page._auto_geometry_pending and not page._overlap_timer.isActive()
    request = job.request
    catalog = request["catalog"]
    keys = {solid["key"] for module in catalog["modules"] for solid in module["solids"]}
    assert {"condenser_lens_3_housing", "condenser_lens_3_yoke"} <= keys

    # Exercise the immutable receipt/UI path with a matching synthetic worker
    # response. The exact OCC calculation is covered by the real-worker test.
    module = next(m for m in catalog["modules"] if any(s["key"] == "condenser_lens_3_housing" for s in m["solids"]))
    collision = dict(first=dict(key="condenser_lens_3_housing", region="body", module_key=module["key"]),
                     second=dict(key="condenser_lens_3_yoke", region="body", module_key=module["key"]),
                     kind="intersection", intersection_volume_mm3=1.)
    result = dict(schema="temsim-cad-geometry-validation-v1", units="mm",
        catalog_sha256=canonical_digest(catalog), binding_report_sha256=request["binding_report_sha256"],
        geometry_report=dict(status="completed", assemblies=[dict(key=m["key"],
            collisions=[collision] if m["key"] == module["key"] else [], unresolved_solids=[])
            for m in catalog["modules"]]),
        probe_results=[dict(id=p["id"], module_key=p["module_key"], kind=p["kind"],
            status="passed", probe_volume_mm3=.001, uncovered_volume_mm3=0.,
            material_blockers=[], unresolved_intersections=[]) for p in request["optical_probes"]])
    result["report_sha256"] = canonical_digest(result)
    receipt = dict(status="complete", request_sha256=canonical_digest(request),
                   draft_digest=request["draft_digest"], request_id=request["request_id"], result=result)
    job._finish(True, receipt, "Synthetic receipt for GUI state regression")
    assert page._geometry_report_is_current(), page.overlaps.status.text()
    assert page.overlaps.table.rowCount() == 1
    assert page.overlaps.table.item(0, 4).text() == "1"
    assert page._cad_validation_dialog.isVisible()
    assert page._cad_validation_report["status"] == "conflict"
    assert page.session.document == draft and source.read_bytes() == raw


def test_real_automatic_exact_overlap_list_tracks_resize_and_undo(qtbot, tmp_path):
    if not (Path(__file__).resolve().parents[1]/"environment/cad/Scripts/python.exe").is_file():
        pytest.skip("Optional isolated exact CAD engine is not installed")
    page, source = editor(qtbot, tmp_path)
    raw = source.read_bytes()
    dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    qtbot.waitUntil(lambda: page._cad_validation_report is not None, timeout=60000)
    assert page._geometry_report_is_current(), page.overlaps.status.text()
    assert page.overlaps.table.rowCount() == 1
    assert page.overlaps.table.item(0, 4).text() == "6"
    report = page._cad_validation_report
    assert not report["electromagnetic_fields_recomputed"]
    assert page._cad_validation_dialog is None  # background checks never open dialogs
    assert not page._cad_geometry_dialog.isVisible()
    page.undo()
    assert page.overlaps.table.rowCount() == 0
    qtbot.waitUntil(lambda: page._cad_validation_report is not report and page._geometry_report_is_current(), timeout=60000)
    assert page.overlaps.table.rowCount() == 0
    assert "0 material overlap pairs" in page.overlaps.status.text()
    assert source.read_bytes() == raw
