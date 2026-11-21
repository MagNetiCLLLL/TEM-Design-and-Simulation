"""Control coordinates stay selectable without becoming mechanical bodies."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pyqtgraph as pg
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsRectItem

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.cad_round_lenses import round_lens_solids
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.component_representation import (
    IMAGE_CONTROL_CHANNEL_KEYS, PROBE_CONTROL_CHANNEL_KEYS,
    SHARED_DEFLECTOR_HOSTS, shared_deflector_field_owner,
)
from temsim.diagnostics import physical_layout_records
from temsim.gui.diagnostic_tabs import PhysicalLayoutView
from temsim.multipole_winding import FIELDS as WINDING_FIELDS
from temsim.optics.column import default_state


CHANNEL_KEYS = (PROBE_CONTROL_CHANNEL_KEYS | IMAGE_CONTROL_CHANNEL_KEYS
                | SHARED_DEFLECTOR_HOSTS.keys() | {"probe_dp12_scan_deflector"})


def _present_channels(resolved):
    return CHANNEL_KEYS & {part.key for part in resolved.assembly.parts}


@pytest.fixture(scope="module", params=(
    "C3 + Probe Corrector", "C3 + Image Corrector",
    "C3 + Probe Corrector + Image Corrector",
))
def resolved(request):
    state = default_state()
    catalog = AssemblyCatalog()
    defaults = catalog.default_selection()
    assembly = catalog.apply(state, AssemblySelection(defaults.gun, request.param, defaults.recording))
    layout = apply_physical_layout_to_state(state)
    return SimpleNamespace(assembly=assembly, state_snapshot=state, layout=layout)


def test_display_records_distinguish_channels_without_mutating_transport(resolved):
    before = deepcopy(resolved.state_snapshot.to_dict())
    parts_before = [dict(part.data) for part in resolved.assembly.parts]
    records = {record.key: record for record in physical_layout_records(resolved)}
    for key in _present_channels(resolved):
        record = records[key]
        data = resolved.assembly.part(key).data
        assert record.layout_role
        assert record.start_z_mm == record.center_z_mm == record.end_z_mm
        assert record.outer_diameter_mm == record.mechanical_bore_diameter_mm == 0
        assert record.optical_references_mm
        assert record.center_z_mm == pytest.approx(resolved.assembly.part(key).center_z_mm)
        if WINDING_FIELDS.intersection(data):
            assert "winding-only CAD" in record.representation_note
        elif data.get("physical_host_status") == "modelled":
            assert data["physical_host_key"] in record.representation_note
            assert "no independent material body" in record.representation_note
        elif key in IMAGE_CONTROL_CHANNEL_KEYS:
            assert "in image corrector" in record.representation_note
            assert "unverified" in record.representation_note
            assert "no independent material body" in record.representation_note
        else:
            assert "no independent material body" in record.representation_note
    for key, host in SHARED_DEFLECTOR_HOSTS.items():
        assert records[key].center_z_mm == pytest.approx(records[host].center_z_mm)
        assert "same physical coils" in records[key].representation_note
        assert "unverified" not in records[key].representation_note
    for key in ("probe_hp1_hexapole", "probe_hp2_hexapole", "probe_tl21_lens", "probe_tl22_lens",
                "image_hp1_hexapole", "image_hp2_hexapole", "image_ol_post_lens",
                "image_tl11_lens", "image_tl12_lens", "image_tl21_lens", "image_tl22_lens",
                "image_adapter_lens", "ac_deflector", "beam_deflector",
                *SHARED_DEFLECTOR_HOSTS.values()):
        if key not in records:
            continue
        assert not records[key].layout_role
        assert records[key].end_z_mm > records[key].start_z_mm
        assert records[key].outer_diameter_mm > 0
    assert records["ac_deflector"].center_z_mm > records["beam_deflector"].center_z_mm
    assert resolved.state_snapshot.to_dict() == before
    assert [dict(part.data) for part in resolved.assembly.parts] == parts_before


def test_2d_channels_have_only_markers_and_selection_never_adds_a_solid(qtbot, resolved):
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.resize(1350, 750)
    page.display_result(resolved)
    channels = _present_channels(resolved)
    assert set(page._channel_marker_items) == channels
    # All independently drawn rectangles remain owned by real components.
    physical_rectangles = set()
    for item in page.plot.scene().items():
        key = page._selectable_item_keys.get(id(item))
        if key and isinstance(item, QGraphicsRectItem):
            assert key not in CHANNEL_KEYS
            physical_rectangles.add(key)
    expected_hexapoles = {key for key in ("probe_hp1_hexapole", "probe_hp2_hexapole",
                                        "image_hp1_hexapole", "image_hp2_hexapole")
                         if key in page._component_label_items}
    assert expected_hexapoles <= physical_rectangles
    for key in channels:
        line = page._channel_marker_items[key]
        assert line.opts["pen"].style() == Qt.PenStyle.DashLine
        assert "channel" in page._component_label_items[key].toPlainText() or "virtual" in page._component_label_items[key].toPlainText()
        page.focus_component(resolved.assembly.part(key))
        assert isinstance(page._highlight, pg.InfiniteLine)
        assert "no separate body" in page.summary.text()
        assert "OD " not in page.summary.text()
    if "probe_hp1_hexapole" in page._component_label_items:
        assert page._component_label_items["probe_hp1_hexapole"].toPlainText() == "HP1 Hexapole"
    page.display_result(resolved)
    assert set(page._channel_marker_items) == channels


def test_2d_return_material_matches_both_sides_of_each_3d_annulus(qtbot, resolved):
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    page.display_result(resolved)
    parts = {p.key: p for p in resolved.assembly.parts}
    items_by_key = {}
    for item in page.plot.scene().items():
        key = page._selectable_item_keys.get(id(item))
        if key and isinstance(item, QGraphicsRectItem):
            r = item.rect()
            items_by_key.setdefault(key, []).append((r.left(), r.right(), r.top(), r.bottom()))
    found = set()
    for part in parts.values():
        role = part.data.get("mechanical_part_role")
        if role not in {"pole_sleeves", "radial_return_plates"}:
            continue
        found.add(role)
        offset = part.center_z_mm - part.data["local_center_z_mm"]
        expected = []
        for solid in round_lens_solids(part, parts):
            z, radius = zip(*solid.shape["section_zr_mm"])
            for lo, hi in ((min(radius), max(radius)), (-max(radius), -min(radius))):
                expected.append((offset + min(z), offset + max(z), lo, hi))
        actual = sorted(items_by_key[part.key])
        assert len(actual) == len(expected) == 4
        for visible, native in zip(actual, sorted(expected), strict=True):
            assert visible == pytest.approx(native)
            assert not visible[0] < part.center_z_mm < visible[1]
    assert found == {"pole_sleeves", "radial_return_plates"}


def test_2d_return_plate_preserves_thin_material_and_a_local_3d_hole(qtbot):
    state, catalog = default_state(), AssemblyCatalog()
    assembly = catalog.apply(state, replace(catalog.default_selection(), column="C3 + Probe Corrector"))
    key = "probe_tl22_lens_return_plates"
    plate = assembly.part(key)
    (z0, _), last = plate.data["material_intervals_mm"]
    origin = plate.center_z_mm - plate.data["local_center_z_mm"]
    data = {**plate.data, "material_intervals_mm": [[z0, z0 + .1], last]}
    hole = dict(id="through_plate", kind="hole", axis="z", center_mm=[42., 0., 0.],
                depth_mm=40., diameter_mm=2.)
    page = PhysicalLayoutView()
    qtbot.addWidget(page)
    for cut in (False, True):
        if cut:
            data = {**data, "model_3d": dict(schema_version=1, features=[hole])}
        installed = replace(assembly, parts=tuple(replace(p, data=data) if p.key == key else p
                                                  for p in assembly.parts))
        page.display_result(SimpleNamespace(assembly=installed, state_snapshot=state,
                                            layout=apply_physical_layout_to_state(state)))
        shapes = [item for item in page.plot.scene().items() if page._selectable_item_keys.get(id(item)) == key
                  and isinstance(item, (QGraphicsRectItem, QGraphicsPathItem))]
        def contains(z, x):
            return any((item.path() if isinstance(item, QGraphicsPathItem) else item.rect()).contains(
                QPointF(origin + z, x)) for item in shapes)
        assert contains(z0 + .05, 40.) and contains(z0 + .05, -42.)
        assert contains(z0 + .05, 42.) == (not cut)
        assert not contains(z0 + .16, 40.)
        assert not contains(plate.data["local_center_z_mm"], 40.)


def test_projector_plate_section_uses_its_thickness_and_installed_position(qtbot):
    state, catalog = default_state(), AssemblyCatalog()
    assembly = catalog.apply(state, catalog.default_selection())
    plate = assembly.part("dl_il_return_yoke")
    z = plate.center_z_mm
    local = plate.data["local_center_z_mm"]
    plate = replace(plate, start_z_mm=z - .05, end_z_mm=z + .05, length_mm=.1,
                    data={**plate.data, "local_start_z_mm": local - .05,
                          "local_end_z_mm": local + .05, "length_mm": .1})
    view = PhysicalLayoutView()
    qtbot.addWidget(view)
    for offset in (0., 30.):
        parts = {p.key: p for p in assembly.parts}
        parts[plate.key] = plate
        parent = parts[plate.parent_key]
        parts[parent.key] = replace(parent, data={**parent.data, "offset_y_mm": offset})
        view.display_result(SimpleNamespace(assembly=replace(assembly, parts=tuple(parts.values())),
                                            state_snapshot=state, layout=apply_physical_layout_to_state(state)))
        sections = [(item.path() if isinstance(item, QGraphicsPathItem) else item.rect())
                    for item in view.plot.scene().items()
                    if view._selectable_item_keys.get(id(item)) == plate.key
                    and isinstance(item, (QGraphicsPathItem, QGraphicsRectItem))]
        assert any(section.contains(QPointF(z, 50.)) for section in sections)
        assert not any(section.contains(QPointF(z + .1, 50.)) for section in sections)
        assert any(section.contains(QPointF(z, 0.)) for section in sections) == bool(offset)
        assert any(section.contains(QPointF(z, 84.)) for section in sections) == (offset == 0.)


def test_legacy_nonzero_channel_envelopes_do_not_reappear_without_new_metadata(resolved):
    parts = []
    for part in resolved.assembly.parts:
        data = dict(part.data)
        for field in ("layout_role", "layout_owner", "physical_host_key", "physical_host_status", *WINDING_FIELDS):
            data.pop(field, None)
        parts.append(SimpleNamespace(**{
            name: getattr(part, name) for name in ("key", "name", "branch", "start_z_mm", "center_z_mm", "end_z_mm")
        }, data=data))
    legacy = SimpleNamespace(assembly=SimpleNamespace(parts=parts), layout=resolved.layout)
    records = {record.key: record for record in physical_layout_records(legacy)}
    channels = _present_channels(resolved)
    assert all(records[key].layout_role for key in channels)
    assert all(records[key].outer_diameter_mm == 0 for key in channels)
    assert all("in image corrector" in records[key].representation_note
               for key in channels & IMAGE_CONTROL_CHANNEL_KEYS)


@pytest.mark.parametrize("key,host", SHARED_DEFLECTOR_HOSTS.items())
def test_shared_channel_structure_is_read_only_in_both_editors(qtbot, key, host):
    from temsim.gui.parameter_panel import ParameterPanel
    from temsim.manifest_editor import ManifestEditor, ManifestTarget
    from temsim.module_manifest import read_document
    from temsim.part_model_3d import part_dimension_specs
    from temsim.part_model_document import PartModelDocument

    editor = ManifestEditor()
    target = ManifestTarget("column/C3_ProbeCorrector.toml", key)
    fields = editor.fields(target)
    owned = [field for field in fields if shared_deflector_field_owner(key, field.path[2])]
    assert owned and all(not field.editable for field in owned)
    assert next(field for field in fields if field.label == "name").editable
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context(key, None, target, fields, None)
    assert host in panel.manifest_draft_notice.text()
    for row, field in enumerate(fields):
        if shared_deflector_field_owner(key, field.path[2]):
            value = panel.manifest_table.item(row, 1)
            assert not value.flags() & Qt.ItemFlag.ItemIsEditable
            assert host in value.toolTip()
    path = editor.root / target.module_path
    specs = part_dimension_specs(read_document(path), key)
    owned_specs = [item for item in specs if shared_deflector_field_owner(key, item.path[2])]
    assert owned_specs and all(not item.editable for item in owned_specs)
    session = PartModelDocument(path)
    before = deepcopy(session.document)
    with pytest.raises(ValueError, match=host):
        session.set_dimension(("parts", key, "length_mm"), 100)
    with pytest.raises(ValueError, match=host):
        session.place_component(key, 100)
    with pytest.raises(ValueError, match=host):
        session.copy_component_from(session.document, key, key + "_copy", 100)
    assert session.document == before
    with pytest.raises(ValueError, match=host):
        editor.save(target, {("parts", key, "local_center_z_mm"): 100}, None)
