"""CETCOR topology/coordinates; no assertion that defaults are Cs-corrected."""
from copy import deepcopy
from dataclasses import replace
import tomllib

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim import module_manifest
from temsim.column.module_assembly import apply_column_manifest_geometry
from temsim.optics.column import default_state
from temsim.optics.field_hardware_binding import resolve_field_hardware_parts
from temsim.optics.image_corrector import resolve_image_corrector_mechanical_anchors
from temsim.optics.model import State
from temsim.part_model_3d import part_model_from_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


VARIANTS = (
    ("C3_ImageCorrector.toml", "C3 + Image Corrector"),
    ("C3_ProbeCorrector_ImageCorrector.toml", "C3 + Probe Corrector + Image Corrector"),
)
FIVE_LENSES = (
    "image_tl11_lens", "image_tl12_lens", "image_tl21_lens",
    "image_tl22_lens", "image_adapter_lens",
)


def _assembled(column):
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "Energy Filter"))
    return state


@pytest.mark.parametrize("filename,column", VARIANTS)
def test_five_physical_image_lenses_survive_assembly_and_state_roundtrip(filename, column):
    state = _assembled(column)
    payload = state.to_dict()
    restored = State.from_dict(deepcopy(payload))
    for candidate in (state, restored):
        system = candidate.image_corrector_system
        assert tuple(c.key for c in system.round_lens_components) == FIVE_LENSES
        assert len({id(c) for c in system.round_lens_components}) == 5
        assert not any(c.key == "image_ol_post_lens" for c in candidate.lenses)
        assert sum(c.key == "objective_lens" for c in candidate.lenses) == 1
        assert system.tl11_lens.z_mm < system.dp12_deflector.z_mm < system.tl12_lens.z_mm
        for lens in system.round_lens_components:
            assert lens.mechanical_length_mm > 1.0  # no thin marker masquerading as hardware
    np.testing.assert_array_equal(
        [c.z_mm for c in state.image_corrector_system.components],
        [c.z_mm for c in restored.image_corrector_system.components],
    )
    document = tomllib.loads((INSTRUMENT_CONFIG_ROOT / "column" / filename).read_text(encoding="utf-8"))
    rows = {p["key"]: p for p in document["parts"]}
    assert not any(k == "image_ol_post_lens" or k.startswith("image_ol_post_lens_") for k in rows)
    assert all(rows[k]["mechanical_profile"] == "magnetic_lens_assembly" for k in FIVE_LENSES)


@pytest.mark.parametrize("filename,column", VARIANTS)
@pytest.mark.parametrize("number", (1, 2))
def test_image_dph_reference_follows_hp_through_one_material_host(filename, column, number):
    document = tomllib.loads((INSTRUMENT_CONFIG_ROOT / "column" / filename).read_text(encoding="utf-8"))
    rows = {p["key"]: p for p in document["parts"]}
    hp_key, dph_key = f"image_hp{number}_hexapole", f"image_dph{number}_deflector"
    hp, dph = rows[hp_key], rows[dph_key]
    host_key = dph["physical_host_key"]
    assert rows[host_key]["optical_reference_host_key"] == hp_key
    assert not dph.get("optical_reference_host_key")
    keys_before, length_before = set(rows), dph["length_mm"]
    # Move an existing host. The stale DPH coordinate must not act as a second offset.
    for field in ("local_start_z_mm", "local_center_z_mm", "local_end_z_mm", "optical_reference_local_z_mm"):
        hp[field] += 1.25
    before = deepcopy(document)
    resolved = resolve_field_hardware_parts(document)
    assert document == before
    assert resolve_field_hardware_parts(resolved) == resolved
    after = {p["key"]: p for p in resolved["parts"]}
    assert set(after) == keys_before
    assert after[dph_key]["local_center_z_mm"] == hp["optical_reference_local_z_mm"]
    assert after[host_key]["local_center_z_mm"] == hp["optical_reference_local_z_mm"]
    assert after[dph_key]["length_mm"] == length_before
    assert not part_model_from_document(resolved, dph_key, angular_segments=8).meshes


def test_runtime_image_dph_follows_hp_once_and_keeps_its_drive():
    state = _assembled(VARIANTS[0][1])
    system = state.image_corrector_system
    resolve_image_corrector_mechanical_anchors(state)
    hp, channel = system.hp1_hexapole, system.dph1_deflector
    channel.kick_x_mrad, channel.kick_y_mrad = .031, -.047
    previous = hp.z_mm
    hp.mechanical_center_from_tip_mm += .75
    hp.optical_reference_from_tip_mm += .75
    hp.apply_optical_position()
    resolve_image_corrector_mechanical_anchors(state)
    assert hp.z_mm == pytest.approx(previous + .75)
    assert channel.z_mm == hp.z_mm
    assert (channel.kick_x_mrad, channel.kick_y_mrad) == (.031, -.047)
    positions = tuple(c.z_mm for c in system.components)
    resolve_image_corrector_mechanical_anchors(state)
    assert tuple(c.z_mm for c in system.components) == positions


def test_selected_manifest_owns_image_field_and_default_but_preserves_live_drive():
    state = _assembled(VARIANTS[0][1])
    original = state._resolved_assembly
    key = "image_tl12_lens"
    part = original.part(key)
    custom_data = dict(part.data, default_excitation_percent=53.25,
                       maximum_peak_field_t=0.812, field_half_width_mm=7.25,
                       field_profile_terms=((1.0, 0.0, 0.8),),
                       normalise_field_profile_peak=True)
    selected = replace(original, parts=tuple(
        replace(p, data=custom_data) if p.key == key else p for p in original.parts
    ))
    apply_column_manifest_geometry(state, None, assembly=selected,
                                   preserve_operating_parameters=False)
    lens = state.image_corrector_system.tl12_lens
    assert lens.percent == 53.25
    assert (lens.b0_t, lens.a_mm, lens.normalise_profile_peak) == (.812, 7.25, True)
    assert [(g.amplitude, g.offset, g.sigma) for g in lens.gaussian] == [(1., 0., .8)]
    lens.percent = 41.5
    apply_column_manifest_geometry(state, None, assembly=selected,
                                   preserve_operating_parameters=True)
    assert lens.percent == 41.5
    assert original.part(key).data == part.data


def test_image_calibration_cannot_silently_fall_back_to_another_module():
    document = module_manifest.read_document(INSTRUMENT_CONFIG_ROOT / "column" / VARIANTS[0][0])
    lens = next(p for p in document["parts"] if p["key"] == "image_tl12_lens")
    lens.pop("maximum_peak_field_t")
    with pytest.raises(ValueError, match="Missing image_tl12_lens TOML field calibration"):
        module_manifest.validate_document(document)


def test_relay_public_methods_return_full_lab_drift_from_their_declared_planes():
    # Actual propagation with a field-free diagnostic copy, not a mocked matrix.
    # Its independent analytic answer detects old 2x2 slicing and the former
    # objective-image-plane source without declaring physical defaults focused.
    state = _assembled(VARIANTS[0][1])
    state.step_mm = state.history_step_mm = .5
    state.acceleration_enabled = False
    for component in (*state.lenses, *state.stigmators, *state.corrector_elements):
        if hasattr(component, "enabled"):
            component.enabled = False
    system = state.image_corrector_system
    assert system.relay_input_plane_z_mm == state.sample.z_mm
    for matrix, start, end in (
        (system.relay_transfer_matrix(), state.sample.z_mm, system.sad_plane.z_mm),
        (system.main_hexapole_transfer_matrix(), system.hp1_hexapole.z_mm, system.hp2_hexapole.z_mm),
    ):
        expected = np.eye(4)
        expected[:2, 2:] = np.eye(2) * ((end - start) * 1e-3)
        assert matrix.shape == (4, 4)
        np.testing.assert_allclose(matrix, expected, rtol=0., atol=2e-8)
