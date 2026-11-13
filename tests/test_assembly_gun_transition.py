"""Static gun-hardware transitions; no emission, fields, rays, or waves."""
from dataclasses import replace
from hashlib import sha256
import re

import numpy as np
import pytest

from temsim import input_io
from temsim.assembly_catalog import AssemblyCatalog
from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot
from temsim.optics.column import default_state


def selected(state, name):
    catalog = AssemblyCatalog()
    selection = replace(catalog.default_selection(), gun=name, column="C2", recording="No Energy Filter")
    catalog.apply(state, selection)
    return catalog, selection


@pytest.mark.parametrize("legacy_sampling", [False, True])
@pytest.mark.parametrize("gun_name,required,absent", [
    ("FEG + Mono", "feg_monochromator_wien", "thermionic_cathode"),
    ("Thermionic", "thermionic_cathode", "feg_tip"),
])
def test_catalog_replacement_binds_runtime_gun_to_matching_selected_parts(legacy_sampling, gun_name, required, absent):
    state = default_state()
    if legacy_sampling:
        state.electron_gun.emitter.sampling = None
    catalog, selection = selected(state, gun_name)
    parts = {part.key for part in state._resolved_assembly.parts}
    assert required in parts and absent not in parts
    assert {part.key for part in state.electron_gun.components} <= parts
    assert catalog.selection_for_resolved(state._resolved_assembly) == selection
    assert state.sample.upper_surface_z_mm >= state._resolved_assembly.part("sample_stage").start_z_mm
    assert state.sample.lower_surface_z_mm <= state._resolved_assembly.part("sample_stage").end_z_mm


def test_direct_monochromator_round_trip_translates_foil_with_port_without_changing_drives():
    state = default_state()
    selected(state, "FEG")
    state.objective_lens.percent = 29.123456
    state.sample.thickness_nm = 10.
    original_upper = state.sample.upper_surface_z_mm
    original_phase = state.sample.lattice_reference_thickness_nm
    original_exit = state.electron_gun.exit_plane_z_mm
    drives = {lens.key: (lens.percent, lens.polarity) for lens in state.lenses}
    state.monochromator_installed = True
    delta = state.electron_gun.exit_plane_z_mm-original_exit
    assert delta > 0
    assert abs(state.sample.upper_surface_z_mm-(original_upper+delta)) <= 4*abs(np.spacing(state.sample.z_mm))
    assert state.sample.lattice_reference_thickness_nm == original_phase
    assert {lens.key: (lens.percent, lens.polarity) for lens in state.lenses} == drives
    assert "feg_monochromator_wien" in {part.key for part in state._resolved_assembly.parts}
    state.monochromator_installed = False
    assert abs(state.sample.upper_surface_z_mm-original_upper) <= 4*abs(np.spacing(original_upper))
    assert state.sample.lattice_reference_thickness_nm == original_phase
    assert {lens.key: (lens.percent, lens.polarity) for lens in state.lenses} == drives


def test_matching_gun_keeps_archived_calibration_and_captured_assembly_when_live_bytes_differ():
    state = default_state()
    catalog, selection = selected(state, "FEG")
    archive = thaw_json(input_io.capture_input_archive(state))
    path = state._resolved_assembly.root / dict(state._resolved_assembly.selected_module_paths)["column"]
    row = next(value for value in archive["files"] if value["path"] == str(path))
    original = bytes.fromhex(row["content_hex"])
    # A historical captured calibration differs from the current live file.
    # Change only a private archive, never the repository's source definition.
    content, changed = re.subn(rb"field_half_width_mm\s*=\s*[0-9.]+",
        b"field_half_width_mm = 7.0", original, count=1)
    assert changed == 1 and content != original
    row.update(content_hex=content.hex(), sha256=sha256(content).hexdigest())
    archive["digest"] = json_digest({key: value for key, value in archive.items() if key != "digest"})
    input_io.bind_archive(state, archive)
    with input_io.input_scope(state):
        catalog.apply(state, selection)
    assert state.condenser_lens_1.lens.a_mm == 7.0
    restored = capture_instrument_snapshot(state).restore()
    captured = restored._resolved_assembly
    upper = restored.sample.upper_surface_z_mm
    restored._set_monochromator_column_offset(0.)
    assert restored._resolved_assembly is captured
    assert restored.condenser_lens_1.lens.a_mm == 7.0
    assert restored.sample.upper_surface_z_mm == upper
    assert path.read_bytes() == original
