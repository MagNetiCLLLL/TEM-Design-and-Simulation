"""Offscreen overlap-result navigation and delayed-worker ownership checks.

The worker transport is controlled here, while requests, draft edits and
receipt/staleness validation use the production implementations. Exact OCC
geometry itself is covered by test_mechanical_overlap_editor.
"""
from copy import deepcopy

import pytest
import tomli_w

from temsim.cad_optical_binding import canonical_digest
from temsim.gui.mechanical_overlap_panel import MechanicalOverlapPanel
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from test_cad_optical_binding import receipt
from test_mechanical_overlap_editor import dimension, editor, pair


@pytest.fixture
def delayed_jobs(monkeypatch):
    from temsim.gui import cad_geometry_job

    jobs = []

    class DelayedJob(cad_geometry_job.CadGeometryJob):
        def start(self):
            self.busy = True
            self.cancel_calls = 0
            jobs.append(self)

        def cancel(self):
            # A completed kernel response can race the cancellation request.
            # Keep the old job running until the test delivers that response.
            self.cancel_calls += 1

        def complete(self, *, volume=6.):
            evidence = receipt(self.request) if self.request["op"] == "validate_catalog" else {}
            if evidence:
                result = evidence["result"]
                assemblies = self.request["catalog"].get("assemblies")
                if assemblies:
                    result["geometry_report"]["assemblies"] = [
                        dict(key=row["key"], collisions=[], unresolved_solids=[]) for row in assemblies]
                    for row, probe in zip(result["probe_results"], self.request["optical_probes"]):
                        if "assembly_key" in probe:
                            row.update(assembly_key=probe["assembly_key"], coordinate_frame="global_column_mm",
                                       module_offset_mm=probe["module_offset_mm"])
                result["geometry_report"]["assemblies"][0]["collisions"] = [pair(volume=volume)]
                result["report_sha256"] = canonical_digest(
                    {key: value for key, value in result.items() if key != "report_sha256"})
            self._finish(True, evidence, "Controlled delayed completion")

        def _stop_for_exit(self):
            self.busy = False

    monkeypatch.setattr(cad_geometry_job, "CadGeometryJob", DelayedJob)
    return jobs


def start_automatic(page, jobs, qtbot):
    page._overlap_timer.setInterval(20)
    dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    qtbot.waitUntil(lambda: len(jobs) == 1, timeout=3000)
    assert jobs[0].automatic_overlap_check
    assert not page._cad_geometry_dialog.isVisible()
    assert page._cad_validation_dialog is None
    return jobs[0]


def test_continuous_edit_cancels_old_check_and_rejects_its_racing_result(qtbot, tmp_path, delayed_jobs):
    page, source = editor(qtbot, tmp_path)
    original = source.read_bytes()
    old = start_automatic(page, delayed_jobs, qtbot)
    dimension(page, ("model_3d", "base", "length_mm")).setText("10")
    assert old.cancel_calls == 1
    assert page.overlaps.table.rowCount() == 0
    old.complete(volume=123.)
    assert page.overlaps.table.rowCount() == 0
    assert page._cad_validation_report is None
    qtbot.waitUntil(lambda: len(delayed_jobs) == 2, timeout=3000)
    current = delayed_jobs[-1]
    assert current.request["draft_digest"] == canonical_digest(page.session.document)
    current.complete(volume=9.)
    assert page._geometry_report_is_current()
    assert page.overlaps.table.item(0, 4).text() == "9"
    assert not page._cad_geometry_dialog.isVisible()
    assert page._cad_validation_dialog is None
    assert source.read_bytes() == original


def test_collision_navigation_and_other_nonaperture_selection_keep_report_current(qtbot, tmp_path, delayed_jobs):
    page, _ = editor(qtbot, tmp_path)
    job = start_automatic(page, delayed_jobs, qtbot)
    job.complete()
    assert page._geometry_report_is_current()
    report = page._cad_validation_report
    page.overlaps._activate(0, 2)
    assert page._selected_key == "block_b"
    assert page._geometry_report_is_current()
    assert page.overlaps.table.rowCount() == 1
    page.select_part("block_c")
    assert page._geometry_report_is_current()
    assert page._cad_validation_report is report
    assert page.overlaps.table.rowCount() == 1


@pytest.mark.parametrize("already_pending", [False, True])
def test_edits_do_not_cancel_manual_step_import_and_queue_check_after_it(qtbot, tmp_path, delayed_jobs, already_pending):
    page, source = editor(qtbot, tmp_path)
    original = source.read_bytes()
    session = page.session
    page._overlap_timer.setInterval(20)
    if already_pending:
        dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    page._start_step_import(
        dict(path=str(tmp_path / "controlled.step"), key="block_a", name="Imported", mode="replace"),
        session, deepcopy(session.document))
    manual = delayed_jobs[0]
    assert manual.request["op"] == "import_step"
    if already_pending:
        assert page._auto_geometry_pending  # An import is not a geometry check.
    dimension(page, ("model_3d", "base", "length_mm")).setText("10" if already_pending else "8")
    qtbot.wait(60)
    assert manual.cancel_calls == 0 and len(delayed_jobs) == 1
    assert page._auto_geometry_pending
    manual.complete()
    assert page.session is session  # The stale import cannot replace this draft.
    qtbot.waitUntil(lambda: len(delayed_jobs) == 2, timeout=3000)
    delayed_jobs[-1].complete()
    assert page._geometry_report_is_current()
    assert source.read_bytes() == original


@pytest.mark.parametrize("edit_during_check", [False, True])
def test_manual_check_consumes_only_its_existing_pending_request(qtbot, tmp_path, delayed_jobs, edit_during_check):
    page, source = editor(qtbot, tmp_path)
    document = deepcopy(page.session.document)
    for index, row in enumerate(document["parts"]):
        row["order"] = index
    source.write_text(tomli_w.dumps(document), encoding="utf-8")
    assert page.open_path(source, selected_key="block_a")
    page._overlap_timer.setInterval(20)
    dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    assert page._auto_geometry_pending
    page.check_geometry()
    assert len(delayed_jobs) == 1, page.overlaps.status.text()
    assert not delayed_jobs[0].automatic_overlap_check
    if edit_during_check:
        dimension(page, ("model_3d", "base", "length_mm")).setText("10")
        assert delayed_jobs[0].cancel_calls == 0
    delayed_jobs[0].complete()
    if edit_during_check:
        assert page._auto_geometry_pending
        assert page._cad_validation_report is None
        qtbot.waitUntil(lambda: len(delayed_jobs) == 2, timeout=3000)
        delayed_jobs[-1].complete()
        assert page._geometry_report_is_current()
        return
    assert page._geometry_report_is_current()
    assert not page._auto_geometry_pending
    qtbot.wait(80)
    assert len(delayed_jobs) == 1


def test_hidden_page_defers_pending_check_until_it_is_visible(qtbot, tmp_path, delayed_jobs):
    page, _ = editor(qtbot, tmp_path)
    page._overlap_timer.setInterval(20)
    dimension(page, ("model_3d", "base", "length_mm")).setText("8")
    page.hide()
    qtbot.wait(60)
    assert not delayed_jobs and not page._overlap_timer.isActive()
    assert page._auto_geometry_pending
    page.show()
    qtbot.waitUntil(lambda: len(delayed_jobs) == 1, timeout=3000)
    delayed_jobs[0].complete()
    assert page._geometry_report_is_current()


def test_old_file_completion_does_not_populate_new_file_list(qtbot, tmp_path, delayed_jobs):
    page, source = editor(qtbot, tmp_path)
    other = tmp_path / "other.toml"
    other.write_bytes(source.read_bytes())
    old = start_automatic(page, delayed_jobs, qtbot)
    page.undo()
    assert not page.session.dirty
    assert page.open_path(other, selected_key="block_b")
    old.complete(volume=123.)
    assert page.session.path == other.resolve()
    assert page.overlaps.table.rowCount() == 0
    assert page._cad_validation_report is None
    assert not page._auto_geometry_pending
    qtbot.wait(60)
    assert len(delayed_jobs) == 1


def live_aperture_editor(qtbot, tmp_path):
    from temsim.column.module_assembly import AssemblyPart, ResolvedAssembly, _load_module

    page, source = editor(qtbot, tmp_path)
    page.hide()
    document = deepcopy(page.session.document)
    document["module"]["type"] = "isolated_aperture_fixture"
    actual = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector_ImageCorrector.toml")
    aperture = deepcopy(next(row for row in actual["parts"] if row["key"] == "condenser_aperture_2"))
    document["parts"].append(aperture)
    document["geometry"].update(length_mm=400., vacuum_drift_inner_diameter_mm=20.,
                                vacuum_liner_wall_thickness_mm=.25)
    document["ports"] = dict(entrance=dict(interface="beam", local_z_mm=0.),
                             exit=dict(interface="beam", local_z_mm=400.))
    for index, row in enumerate(document["parts"]):
        row["order"] = index
    live = tmp_path / "live.toml"
    live.write_text(tomli_w.dumps(document), encoding="utf-8")
    (tmp_path / "catalog.toml").write_text("# Captured fixture catalog\n", encoding="utf-8")
    assert page.open_path(live, selected_key=aperture["key"])
    module = _load_module(live)
    parts = tuple(AssemblyPart(module.key, live.name, p.key, p.name, p.branch,
                              p.start_z_mm, p.center_z_mm, p.end_z_mm, p.length_mm,
                              p.parent_key, p.data) for p in module.parts)
    assembly = ResolvedAssembly((module,), parts, 400., tmp_path,
                                (("mechanical", live.name),), (), ())
    page.set_project_context(tmp_path, assembly, None,
                             runtime_values={aperture["key"]: {"radius_mm": .05}})
    page.show()
    page._overlap_timer.setInterval(20)
    # Establish a stable starting state; each test owns its first request.
    page._overlap_timer.stop()
    page._auto_geometry_pending = False
    return page, live, aperture["key"]


def test_runtime_opening_invalidates_existing_report_and_racing_worker(qtbot, tmp_path, delayed_jobs):
    page, source, key = live_aperture_editor(qtbot, tmp_path)
    original = source.read_bytes()
    draft = deepcopy(page.session.document)
    page.check_geometry(automatic=True)
    assert delayed_jobs, page.overlaps.status.text()
    delayed_jobs[0].complete()
    assert page.overlaps.table.rowCount() == 1
    assert page._geometry_report_is_current()
    page.set_runtime_values({key: {"radius_mm": .1}})
    assert page.overlaps.table.rowCount() == 0
    qtbot.waitUntil(lambda: len(delayed_jobs) == 2, timeout=3000)
    old = delayed_jobs[-1]
    page.set_runtime_values({key: {"radius_mm": .15}})
    assert old.cancel_calls == 1
    old.complete(volume=123.)
    assert page.overlaps.table.rowCount() == 0
    qtbot.waitUntil(lambda: len(delayed_jobs) == 3, timeout=3000)
    delayed_jobs[-1].complete(volume=4.)
    assert page._geometry_report_is_current()
    assert page.overlaps.table.item(0, 4).text() == "4"
    assert page.session.document == draft
    assert source.read_bytes() == original


def test_runtime_opening_elsewhere_in_checked_assembly_also_queues_new_check(qtbot, tmp_path, delayed_jobs):
    page, _, key = live_aperture_editor(qtbot, tmp_path)
    page.select_part("block_a")
    page.scope.setCurrentIndex(page.scope.findData("part"))
    page.check_geometry(automatic=True)
    delayed_jobs[0].complete()
    assert page._geometry_report_is_current()
    page.set_runtime_values({key: {"radius_mm": .1}})
    assert page.overlaps.table.rowCount() == 0
    assert page._auto_geometry_pending, "The check covers the assembly, including unselected live apertures."
    qtbot.waitUntil(lambda: len(delayed_jobs) == 2, timeout=3000)
    delayed_jobs[-1].complete()
    assert page._geometry_report_is_current()


def test_contact_and_uncertain_filters_do_not_change_confirmed_collision_count(qtbot):
    panel = MechanicalOverlapPanel()
    qtbot.addWidget(panel)
    report = dict(scope="Fixture", geometry_report=dict(assemblies=[dict(
        collisions=[pair(volume=6.)], contacts=[pair("contact", 0.)],
        uncertain_occupancy_intersections=[pair(volume=12.)],
        sub_tolerance_intersections=[pair("sub_tolerance_intersection", 1e-12)],
        unresolved_solids=[dict(key="unresolved", region="envelope", occupancy="unknown")])]))
    before = deepcopy(report)
    panel.set_report(report, names={"block_a": "First block"})
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 4).text() == "6"
    panel.filter.setCurrentIndex(panel.filter.findData("contacts"))
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 4).text() == "0"
    assert panel.table.item(0, 5).text() == "Surface contact"
    with qtbot.waitSignal(panel.component_requested) as signal:
        panel._activate(0, 2)
    assert signal.args == ["block_b", "body"]
    panel.filter.setCurrentIndex(panel.filter.findData("uncertain"))
    assert panel.table.rowCount() == 3
    assert [panel.table.item(i, 5).text() for i in range(3)] == [
        "Uncertain occupancy", "Below volume tolerance", "Unresolved solid"]
    assert panel.table.item(2, 4).text() == "—"
    with qtbot.waitSignal(panel.component_requested) as signal:
        panel._activate(2, 0)
    assert signal.args == ["unresolved", "envelope"]
    panel._activate(2, 2)  # No imaginary second component for a single unresolved solid.
    assert "1 material overlap pairs" in panel.status.text()
    panel.filter.setCurrentIndex(0)
    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "First block"
    assert report == before


@pytest.mark.parametrize("transition", ["pending", "stale"])
def test_filter_switch_cannot_revive_stale_geometry_rows(qtbot, transition):
    panel = MechanicalOverlapPanel()
    qtbot.addWidget(panel)
    report = dict(geometry_report=dict(assemblies=[dict(collisions=[pair()], contacts=[pair("contact", 0.)])]))
    panel.set_report(report)
    if transition == "pending":
        panel.pending()
    else:
        panel.set_report(report, current=False)
    for index in range(panel.filter.count()):
        panel.filter.setCurrentIndex(index)
        assert panel.table.rowCount() == 0
