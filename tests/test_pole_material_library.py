"""Sourced pole grades, independent installed hardware assignments and static FEM."""
from dataclasses import replace
import json

import numpy as np
import pytest
from PySide6.QtWidgets import QTabWidget

from temsim.assembly_catalog import AssemblyCatalog
from temsim.cad_catalog import module_catalog
from temsim.cad_vacuum_tube import TERMINAL_KEY
from temsim.column.state_layout import layout_configuration_from_state
from temsim.gui.model_inspector import ModelInspectorPage
from temsim.gui.pole_materials import installed_targets
from temsim.magnetic_materials import BHCurve, MU0, lens_material_defaults
from temsim.manifest_editor import ManifestEditor
from temsim.module_manifest import read_document
from temsim.optics.column import default_state
from temsim.part_materials import NONMAGNETIC_STEEL, material_catalog, material_for_region, part_material_updates
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.axisymmetric_magnetostatics import solve_geometry_field_details
from temsim.physics.lens_field_provider import lens_geometry_binding
from temsim.shared_tip import copy_catalog_tree


@pytest.mark.parametrize("key,h,b,upper", [
    ("armco_pure_iron", 100, 1.289, 1.996),
    ("vacoflux_50", 1000, 2.105, 2.293),
    ("permenorm_5000_h2", 10, .938, 1.514),
    ("mumetall_solid", 1, .479, .756),
])
def test_published_curve_points_bounds_and_linear_reference(key, h, b, upper):
    grade = next(row for row in material_catalog() if row["material_key"] == key)
    data = grade["bh_material"]
    assert data["source_url"].startswith("https://") and data["source_url"].endswith(".pdf")
    assert len(data["source_sha256"]) == 64 and data["source_block"]
    curve = BHCurve(data)
    assert curve.evaluate(b)[0] == pytest.approx(h)
    assert curve.b[-1] == upper
    with pytest.raises(ValueError, match="range exceeded"):
        curve.evaluate(upper + .001)
    source = grade["linear_source"]
    assert source["reference_b_t"] / (MU0 * source["reference_h_a_per_m"]) == grade["relative_permeability"]


def choose(box, value):
    index = box.findData(value)
    assert index >= 0, value
    box.setCurrentIndex(index)


def test_material_source_click_uses_selected_grade_and_native_url_handler(qtbot):
    from PySide6.QtCore import QObject, QPoint, Qt, QUrl, Slot
    from PySide6.QtGui import QDesktopServices
    from temsim.gui.material_reference import MaterialReferenceLabel
    from temsim.gui.pole_materials import PoleMaterialsPage

    class Receiver(QObject):
        @Slot(QUrl)
        def capture(self, url):
            received.append(bytes(url.toEncoded()).decode("ascii"))

    received = []
    receiver = Receiver()
    panel = PoleMaterialsPage()
    qtbot.addWidget(panel)
    panel.set_assembly(default_state()._resolved_assembly)
    source = MaterialReferenceLabel()
    qtbot.addWidget(source)
    source.setFixedWidth(380)
    source.show()
    QDesktopServices.setUrlHandler("https", receiver, "capture")
    try:
        for key in ("armco_pure_iron", "vacoflux_50", "permenorm_5000_h2", "mumetall_solid"):
            choose(panel.material, key)
            href = panel.catalog[key]["bh_material"]["source_url"]
            assert f'href="{href}"' in panel.source.text()
            source.setText(panel.source.text())
            source.setFixedHeight(source.heightForWidth(source.width()))
            qtbot.mouseClick(source, Qt.LeftButton, pos=QPoint(10, source.height() - source.fontMetrics().height() // 2))
            assert received[-1] == href
        assert len(received) == 4
        choose(panel.material, NONMAGNETIC_STEEL)
        assert "href=" not in panel.source.text()
        source.set_reference("Imported <curve>", "file:///local.csv")
        assert "href=" not in source.text() and "&lt;curve&gt;" in source.text()
    finally:
        QDesktopServices.unsetUrlHandler("https")


def test_inspector_saved_group_and_component_assignments(qtbot, tmp_path, monkeypatch):
    root = tmp_path / "instruments"
    copy_catalog_tree(INSTRUMENT_CONFIG_ROOT, root)
    catalog = AssemblyCatalog(root)
    state = default_state()
    selection = replace(catalog.selection_for_resolved(state._resolved_assembly), column="C3 + Probe Corrector + B-COR")
    assembly = catalog.apply(state, selection)
    page = ModelInspectorPage()
    qtbot.addWidget(page)
    page.set_state(state)
    panel = page.pole_materials
    tabs = page.findChild(QTabWidget, "modelInspectorTabs")
    assert tabs.tabText(tabs.indexOf(panel)) == "Pole materials"
    targets = installed_targets(assembly)
    for group in ("Objective", "Probe corrector", "Image corrector"):
        roles = {t.role for t in targets if t.group == group}
        assert {"Pole pieces", "Sleeves", "Return plates"} <= roles
    for part in assembly.parts:
        if part.data.get("mechanical_profile") == "magnetic_pole_piece":
            assert material_for_region(part)["material_key"] == "armco_pure_iron"
    for t in targets:
        if t.role == "Pole pieces":
            assert material_for_region(assembly.part(t.key), t.region)["material_key"] == "armco_pure_iron"

    choose(panel.group, "Image corrector")
    choose(panel.role, "Sleeves")
    choose(panel.target, "*")
    choose(panel.material, "permenorm_5000_h2")
    with qtbot.waitSignal(panel.save_requested) as requested:
        panel.save.click()
    target, updates = requested.args
    edited = {path[1] for path in updates}
    assert edited == {t.key for t in targets if t.group == "Image corrector" and t.role == "Sleeves"}
    editor = ManifestEditor(root)
    configuration = layout_configuration_from_state(state)
    editor.save(target, updates, configuration)
    # An old dialog must not overwrite a new edit to the same module.
    with pytest.raises(ValueError, match="changed outside this editor"):
        editor.save(target, updates, configuration)
    assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
    page.set_state(state)
    assert panel.material.currentData() == "permenorm_5000_h2"
    assert material_for_region(assembly.part("image_hp3_hexapole"), "pole_root_sleeve_6")["material_key"] == "permenorm_5000_h2"
    assert material_for_region(assembly.part("image_hp3_hexapole"), "pole_pin_6")["material_key"] == "armco_pure_iron"
    assert material_for_region(assembly.part("probe_hp2_hexapole"), "pole_root_sleeve_6") is None

    choose(panel.group, "Objective")
    choose(panel.role, "Return plates")
    choose(panel.target, "objective_return_plates/upper")
    choose(panel.material, "vacoflux_50")
    with qtbot.waitSignal(panel.save_requested) as requested:
        panel.save.click()
    target, updates = requested.args
    editor.save(target, updates, configuration)
    assembly = catalog.apply(state, selection, preserve_operating_parameters=True)
    plate = assembly.part("objective_return_plates")
    assert material_for_region(plate, "upper")["material_key"] == "vacoflux_50"
    assert material_for_region(plate, "lower") is None
    assert material_for_region(assembly.part("objective_pole_sleeves"), "upper") is None
    page.set_state(state)
    choose(panel.target, "*")
    assert panel.current.text() == "Mixed materials"
    choose(panel.role, "Terminal vacuum tube")
    assert panel.material.currentData() == NONMAGNETIC_STEEL
    panel.set_assembly(assembly, read_only=True)
    assert not panel.save.isEnabled()
    assert not panel.material.isEnabled()
    with monkeypatch.context() as archived:
        def reject_live_read(*_):
            pytest.fail("Browsing archived material assignments must use the captured data")
        archived.setattr("temsim.gui.pole_materials.material_for_region", reject_live_read)
        choose(panel.role, "Return plates")
        choose(panel.target, "objective_return_plates/upper")
        assert panel.material.currentData() == "vacoflux_50"
        assert "https://" in panel.source.text()

    document = read_document(root / target.module_path)
    hp = next(p for p in document["parts"] if p["key"] == "image_hp3_hexapole")
    solids = module_catalog(dict(module=document["module"], parts=[hp]))["solids"]
    pin_materials = [s["material"] for s in solids if s["region"].startswith("pole_pin_")]
    sleeves = [s["material"] for s in solids if s["region"].startswith("pole_root_sleeve_")]
    assert pin_materials == ["armco_pure_iron"] * 6
    assert sleeves == ["permenorm_5000_h2"] * 6


def test_terminal_tube_changes_only_wall_law_and_changes_solved_field():
    state = default_state()
    assembly = state._resolved_assembly
    upper = assembly.part("objective_upper_pole")
    settings = dict(solver="axisymmetric_nonlinear_fem", ampere_turns=1.,
                    channel_ampere_turns={"objective_lens": 1.}, bh_material=lens_material_defaults()["bh_material"],
                    radial_nodes=20, axial_nodes=36, padding_factor=1.5)
    state.lens_field_map_descriptors["objective_lens"] = settings
    payloads, fingerprints, solutions = [], [], []
    for material in (NONMAGNETIC_STEEL, "armco_pure_iron"):
        regions, = part_material_updates(upper, material, "terminal_tube").values()
        edited = replace(upper, data=dict(upper.data, material_regions=regions))
        state._resolved_assembly = replace(assembly, parts=tuple(edited if p.key == upper.key else p for p in assembly.parts))
        binding = lens_geometry_binding(state, "objective_lens")
        payload = json.loads(binding.canonical_geometry_json)["lens_assembly"]
        assert payload["magnetostatic_neighbours"]
        assert all(row["key"] != TERMINAL_KEY for row in payload["magnetostatic_neighbours"])
        tube, = (row for row in payload["parts"] if row["key"] == TERMINAL_KEY)
        assert tube["end_z_mm"] == upper.start_z_mm
        assert tube["length_mm"] == 40.
        assert all(p[1] > 0 for p in tube["data"]["magnetic_radial_profile_mm"])
        assert tube["data"].pop("material_regions")["body"]["material_key"] == material
        payloads.append(payload)
        fingerprints.append(binding.geometry_fingerprint)
        _, sol = solve_geometry_field_details(binding, settings)
        assert sol.relative_residual < 1.e-8
        solutions.append(sol)
    assert payloads[0] == payloads[1] and fingerprints[0] != fingerprints[1]
    a, b = solutions
    np.testing.assert_array_equal(a.r_m, b.r_m)
    np.testing.assert_array_equal(a.z_m, b.z_m)
    z = state.sample.upper_surface_z_mm * 1e-3
    first, second = (np.interp(z, sol.z_m, sol.bz_t[0]) for sol in solutions)
    assert abs(second - first) > .01 * abs(first) > 0.


def test_bore_and_gap_refinement_keeps_boundary_and_changes_the_nonlinear_mesh():
    from temsim.physics.nonlinear_circuits import operator_settings

    state = default_state()
    binding = lens_geometry_binding(state, "objective_lens")
    settings = dict(solver="axisymmetric_nonlinear_fem", channel_ampere_turns={"objective_lens": 2.},
                    bh_material=lens_material_defaults()["bh_material"], radial_nodes=20, axial_nodes=36,
                    padding_factor=1.5)
    _, base = solve_geometry_field_details(binding, settings)
    refined = dict(settings, refine_pole_mesh=True)
    assert operator_settings(refined)["refine_pole_mesh"] is True
    with pytest.raises(ValueError, match="boolean"):
        operator_settings(dict(settings, refine_pole_mesh="false"))
    _, local = solve_geometry_field_details(binding, refined)
    np.testing.assert_array_equal(local.r_m[[0,-1]], base.r_m[[0,-1]])
    np.testing.assert_array_equal(local.z_m[[0,-1]], base.z_m[[0,-1]])
    pole = state._resolved_assembly.part("objective_upper_pole")
    lower = state._resolved_assembly.part("objective_lower_pole")
    bore = pole.data["mechanical_bore_diameter_mm"] * .0005
    gap = np.array([pole.end_z_mm, lower.start_z_mm]) * .001
    assert (local.r_m <= bore).sum() > (base.r_m <= bore).sum()
    assert ((local.z_m > gap[0]) & (local.z_m < gap[1])).sum() >= 8
    assert local.r_m.size * local.z_m.size <= 250000
