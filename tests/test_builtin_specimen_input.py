"""Portable built-in silicon references preserve the captured optical state."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from temsim.immutable_json import thaw_json
from temsim.input_design import restore_input_design
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.paths import BUNDLED_SI_CIF, CONFIG_ROOT, project_root
from temsim.specimen.cif_io import read_cif_atoms
from temsim.specimen.geometry import build_sample_geometry_snapshot
from temsim.specimen.scene import SpecimenScene
from temsim.working_point import WorkingPointCheckpoint


HISTORICAL_POINT = project_root() / "profiles" / "feg-nanoprobe-300kv-12p87mrad-initial-20261110.temwp"
REFERENCE_CIF = Path(__file__).parent / "fixtures" / "cif" / "Si.cif"


def test_explicit_historical_input_rebinds_structure_and_initializes_only_entrance_metadata():
    point = WorkingPointCheckpoint.read_package(HISTORICAL_POINT)
    captured = decode_instrument(point.snapshot.graph)
    restored = point.compatible_state()

    assert captured.sample.cif_path == "tests/fixtures/cif/Si.cif"
    assert restored.sample.cif_path == str(BUNDLED_SI_CIF)
    captured.sample.cif_path = restored.sample.cif_path
    captured.sample.ensure_upper_surface_reference()
    assert restored.sample.z_mm == 1599.2
    assert restored.sample.thickness_nm == .4
    assert restored.sample.upper_surface_z_mm == 1599.1999998000001
    assert restored.sample.lattice_reference_thickness_nm == .4
    assert encode_instrument(restored) == encode_instrument(captured)
    assert restored.to_dict() == captured.to_dict()
    assert WorkingPointCheckpoint.read_package(HISTORICAL_POINT).digest == point.digest


def test_bundled_silicon_preserves_the_reference_structure():
    assert BUNDLED_SI_CIF.read_bytes() == REFERENCE_CIF.read_bytes()
    bundled, reference = map(read_cif_atoms, (BUNDLED_SI_CIF, REFERENCE_CIF))
    np.testing.assert_array_equal(bundled.numbers, reference.numbers)
    np.testing.assert_array_equal(bundled.positions, reference.positions)
    np.testing.assert_array_equal(bundled.cell.array, reference.cell.array)


def test_relocated_input_rebinds_a_dormant_config_structure(tmp_path):
    point = WorkingPointCheckpoint.read_package(HISTORICAL_POINT)
    graph = thaw_json(point.snapshot.graph)
    old_root = tmp_path / "old-install" / "configs"

    def relocate(value):
        if isinstance(value, str):
            return value.replace(str(CONFIG_ROOT), str(old_root))
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        return value

    graph = relocate(graph)
    sample = next(node for node in graph["nodes"]
                  if node["type"] == "temsim.optics.model:Sample")
    sample["attributes"]["cif_path"] = str(old_root / "specimens" / "Si.cif")
    inputs = relocate(thaw_json(point.snapshot.external_inputs))
    snapshot = replace(point.snapshot, graph=graph, external_inputs=inputs)

    restored = restore_input_design(snapshot)
    assert restored.sample.cif_path == str(BUNDLED_SI_CIF)


def test_explicit_historical_material_and_imported_cif_gui_work_from_foreign_directory(
        tmp_path, monkeypatch, qtbot):
    from temsim.gui.sample_panel import SamplePage

    point = WorkingPointCheckpoint.read_package(HISTORICAL_POINT)
    monkeypatch.chdir(tmp_path)
    state = point.compatible_state()
    original = deepcopy(state.to_dict())
    errors = []
    page = SamplePage()
    qtbot.addWidget(page)
    page.error.connect(errors.append)
    page.set_state(state)
    page.show()
    page.mode.setCurrentIndex(page.mode.findData("atomic"))

    qtbot.waitUntil(lambda: page._snapshot is not None
                    and bool(page._snapshot.atomic_numbers.size), timeout=10000)
    assert state.sample.specimen_mode == "atomic" and state.sample.inserted
    assert "unavailable" not in page.atom_display_label.text().lower()
    geometry = build_sample_geometry_snapshot(state.sample, load_atoms=True)
    scene = SpecimenScene.from_state(state, include_eds_materials=True)
    assert geometry.atomic_numbers.size and set(geometry.atomic_numbers) == {14}
    assert scene.source_kind == "cif" and scene.structure_available
    assert not errors
    restored_settings = state.to_dict()
    original["sample"]["specimen_mode"] = "atomic"
    original["sample"]["inserted"] = True
    assert restored_settings == original
