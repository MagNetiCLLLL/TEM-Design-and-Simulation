"""Fixed entrance and downstream thickness edits using static geometry only."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog
from temsim.column.state_layout import apply_physical_layout_to_state, layout_configuration_from_state
from temsim.immutable_json import json_digest, thaw_json
from temsim.instrument_snapshot import capture_instrument_snapshot, decode_instrument, encode_instrument
from temsim.optics.model import Sample, State
from temsim.specimen.axial_geometry import set_sample_thickness
from temsim.working_point import WorkingPointCheckpoint


PROFILE = Path(__file__).resolve().parents[1]/"profiles/default-nanoprobe.temwp"


@pytest.fixture
def state():
    before = sha256(PROFILE.read_bytes()).hexdigest()
    result = WorkingPointCheckpoint.read_package(PROFILE).compatible_state()
    yield result
    assert sha256(PROFILE.read_bytes()).hexdigest() == before


def _non_sample_geometry(state):
    return {part.key: (part.start_z_mm, part.center_z_mm, part.end_z_mm, dict(part.data))
            for part in state._resolved_assembly.parts if part.key != "sample"}


def test_increase_and_decrease_thickness_keep_upper_and_only_move_lower_face(state):
    sample = state.sample
    upper = sample.upper_surface_z_mm
    original = _non_sample_geometry(state)
    excitations = {lens.key: (lens.percent, lens.enabled, lens.z_mm) for lens in state.lenses}
    for thickness in (10., 20., .2, .4):
        receipt = set_sample_thickness(state, thickness)
        assert state.sample is sample
        assert sample.upper_surface_z_mm == upper
        assert sample.lattice_reference_thickness_nm == .4
        assert sample.z_mm == upper+thickness*.5e-6
        assert sample.lower_surface_z_mm == upper+thickness*1e-6
        assert receipt["new_thickness_nm"] == thickness
        assert state._resolved_assembly.part("sample").center_z_mm == sample.z_mm
        assert _non_sample_geometry(state) == original
        assert {lens.key: (lens.percent, lens.enabled, lens.z_mm) for lens in state.lenses} == excitations


def test_active_default_uses_its_actual_point_four_nm_reference_not_the_class_default(state):
    assert sha256(PROFILE.read_bytes()).hexdigest() == "4d8e068166a9b03176e9a83167ba72f8743645241866d778b1d7b7fb8de277fc"
    assert state.sample.z_mm == 1599.2
    assert state.sample.thickness_nm == .4
    assert state.sample.upper_surface_z_mm == state.sample.upper_surface_reference_z_mm == 1599.1999998000001
    assert state.sample.lattice_reference_thickness_nm == .4


@pytest.mark.parametrize("initial_thickness", [.4, 5.])
def test_direct_sample_thickness_assignment_and_normal_layout_use_the_same_reference(state, initial_thickness):
    # A historical 5 nm sample at its stored centre exposes cancellation that
    # does not occur for every 0.4 nm default value.
    object.__setattr__(state.sample, "thickness_nm", initial_thickness)
    object.__setattr__(state.sample, "upper_surface_reference_z_mm", None)
    object.__setattr__(state.sample, "lattice_reference_thickness_nm", None)
    upper = state.sample.upper_surface_z_mm
    state.sample.thickness_nm = 10.
    apply_physical_layout_to_state(state, preserve_operating_parameters=True)
    assert state.sample.upper_surface_z_mm == upper
    assert abs(state.sample.z_mm-(upper+5e-6)) <= 4*abs(np.spacing(upper))
    assert state._resolved_assembly.part("sample").center_z_mm == state.sample.z_mm


def test_repeated_layout_snapshot_restore_and_state_roundtrip_do_not_drift(state):
    set_sample_thickness(state, 10.)
    upper = state.sample.upper_surface_z_mm
    for _ in range(3):
        apply_physical_layout_to_state(state, preserve_operating_parameters=True)
        assert layout_configuration_from_state(state).resolved_assembly.part("sample").center_z_mm == state.sample.z_mm
        state = capture_instrument_snapshot(state).restore()
        assert state.sample.upper_surface_z_mm == upper
    reloaded = State.from_dict(state.to_dict())
    assert reloaded.sample.upper_surface_z_mm == upper
    assert reloaded.sample.thickness_nm == 10.
    assert reloaded.sample.lattice_reference_thickness_nm == .4


def test_catalog_reload_cannot_replace_the_fixed_entrance_with_live_toml_centre(state):
    set_sample_thickness(state, 10.)
    upper = state.sample.upper_surface_z_mm
    catalog = AssemblyCatalog()
    selection = catalog.selection_for_resolved(state._resolved_assembly)
    catalog.apply(state, selection, preserve_operating_parameters=True)
    assert state.sample.upper_surface_z_mm == upper
    assert state._resolved_assembly.part("sample").center_z_mm == state.sample.z_mm


def test_shared_explicit_sample_place_chooses_new_reference_and_roundtrips(state):
    from temsim import input_io
    from temsim.geometry_experiments import prepare_geometry_changes
    old_upper = state.sample.upper_surface_z_mm
    old = state._resolved_assembly.part("sample")
    target_local = old.data["local_center_z_mm"]+4.8e-6
    catalog = AssemblyCatalog()
    selection = catalog.selection_for_resolved(state._resolved_assembly)
    work, _ = prepare_geometry_changes(state, selection,
        (("column", "sample", "local_center_z_mm", target_local),), input_io.capture_input_archive(state))
    assert abs(work.sample.z_mm-(old.center_z_mm+4.8e-6)) <= 4*abs(np.spacing(old.center_z_mm))
    reference = work.sample.upper_surface_z_mm
    assert reference != old_upper
    restored = capture_instrument_snapshot(work).restore()
    apply_physical_layout_to_state(restored, preserve_operating_parameters=True)
    assert restored.sample.upper_surface_z_mm == reference
    set_sample_thickness(restored, 10.)
    assert restored.sample.upper_surface_z_mm == reference


@pytest.mark.parametrize("thickness", [-1., 0., float("nan"), float("inf"), True, 1e9])
def test_invalid_or_outside_holder_edits_roll_back_all_objects_and_archive(state, thickness):
    graph = json_digest(encode_instrument(state))
    sample, objective, assembly = state.sample, state.objective_lens, state._resolved_assembly
    sample_values = deepcopy(vars(sample))
    with pytest.raises(ValueError):
        set_sample_thickness(state, thickness)
    assert state.sample is sample and state.objective_lens is objective and state._resolved_assembly is assembly
    assert vars(sample) == sample_values
    assert json_digest(encode_instrument(state)) == graph


def test_failure_after_candidate_archive_creation_restores_old_input_graph(state, monkeypatch):
    from temsim.column import state_layout
    original = state_layout.apply_physical_layout_to_state
    before = json_digest(encode_instrument(state))
    def fail_after_layout(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("Synthetic final validation failure")
    monkeypatch.setattr(state_layout, "apply_physical_layout_to_state", fail_after_layout)
    with pytest.raises(ValueError, match="Synthetic final validation"):
        set_sample_thickness(state, 10.)
    assert json_digest(encode_instrument(state)) == before


def test_legacy_exact_graph_missing_metadata_decodes_without_relocating_or_changing_graph(state):
    old_graph = thaw_json(encode_instrument(state))
    for node in old_graph["nodes"]:
        if node["type"] == "temsim.optics.model:Sample":
            for name in ("upper_surface_reference_z_mm", "lattice_reference_thickness_nm"):
                node["attributes"].pop(name, None)
                if name in node["fields"]:node["fields"].remove(name)
    legacy = decode_instrument(old_graph)
    assert legacy.sample.upper_surface_reference_z_mm is None
    assert legacy.sample.lattice_reference_thickness_nm is None
    assert legacy.sample.upper_surface_z_mm == state.sample.upper_surface_z_mm
    assert json_digest(encode_instrument(legacy)) == json_digest(old_graph)
    captured = capture_instrument_snapshot(legacy)
    assert legacy.sample.upper_surface_reference_z_mm == state.sample.upper_surface_z_mm
    assert legacy.sample.lattice_reference_thickness_nm == .4
    assert captured.restore().sample.z_mm == state.sample.z_mm


def test_already_correct_ten_nm_legacy_sample_initializes_its_own_phase_and_upper(state):
    set_sample_thickness(state, 10.)
    upper, centre = state.sample.upper_surface_z_mm, state.sample.z_mm
    object.__setattr__(state.sample, "upper_surface_reference_z_mm", None)
    object.__setattr__(state.sample, "lattice_reference_thickness_nm", None)
    apply_physical_layout_to_state(state, preserve_operating_parameters=True)
    assert state.sample.upper_surface_z_mm == upper and state.sample.z_mm == centre
    assert state.sample.lattice_reference_thickness_nm == 10.
    set_sample_thickness(state, 20.)
    assert state.sample.upper_surface_z_mm == upper
    assert state.sample.lattice_reference_thickness_nm == 10.


def test_operating_profile_persists_both_references_and_restores_before_thickness_edit(state, tmp_path):
    from temsim.profile_io import save_profile, read_profile, apply_profile_values
    set_sample_thickness(state, 10.)
    upper = state.sample.upper_surface_z_mm
    catalog = AssemblyCatalog()
    path = tmp_path/"anchored-operating.toml"
    save_profile(path, state, catalog.selection_for_resolved(state._resolved_assembly))
    _, values = read_profile(path)
    fresh = WorkingPointCheckpoint.read_package(PROFILE).compatible_state()
    apply_profile_values(fresh, values)
    apply_physical_layout_to_state(fresh, preserve_operating_parameters=True)
    assert fresh.sample.upper_surface_z_mm == upper
    assert fresh.sample.thickness_nm == 10.
    assert fresh.sample.lattice_reference_thickness_nm == .4
    set_sample_thickness(fresh, 20.)
    assert fresh.sample.upper_surface_z_mm == upper


def test_legacy_operating_profile_initializes_reference_from_loaded_thickness_not_five_nm_class_default():
    from temsim.optics.column import default_state
    from temsim.profile_io import apply_profile_values
    state = default_state()
    centre = state.sample.z_mm
    apply_profile_values(state, {"sample": {"thickness_nm": .4}})
    assert state.sample.z_mm == centre
    assert state.sample.thickness_nm == state.sample.lattice_reference_thickness_nm == .4
    assert state.sample.upper_surface_z_mm == centre-.4*.5e-6
    set_sample_thickness(state, 10.)
    assert state.sample.upper_surface_z_mm == centre-.4*.5e-6


def test_fixed_coordinate_magnetic_field_is_unchanged_by_thickness(state):
    from temsim.physics.instrument_magnetic import capture_instrument_magnetic_field
    points = np.array([[0.,0.,1.5992], [1e-9,0.,1.5992], [0.,-2e-9,1.599201]])
    before = capture_instrument_magnetic_field(state).field_at_global_positions_t(points)
    set_sample_thickness(state, 10.)
    after = capture_instrument_magnetic_field(state).field_at_global_positions_t(points)
    np.testing.assert_array_equal(after, before)


def test_intentional_position_change_selects_a_new_reference_without_thickness_refocus():
    sample = Sample(z_mm=100., thickness_nm=10.)
    sample.z_mm = 100.001
    upper = sample.upper_surface_z_mm
    sample.thickness_nm = 20.
    assert sample.upper_surface_z_mm == upper
    assert sample.z_mm == upper+10e-6
