"""B-COR topology and shared execution inputs, not manufacturer qualification."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.component_keys import BCOR_EXTRA_KEYS, BCOR_KEYS, IMAGE_CORRECTOR_KEYS
from temsim.optics.column import default_state
from temsim.optics.corrector_structure import ensure_corrector_structure
from temsim.optics.image_corrector import (
    ImageCorrectorHp3HexapoleComponent,
    ImageCorrectorTl31LensComponent,
    ImageCorrectorTl32LensComponent,
    resolve_image_corrector_mechanical_anchors,
)
from temsim.optics.model import State
from temsim.physics.core import hexapole_field_components


def assembled(probe=False):
    state = default_state()
    name = "C3 + Probe Corrector + B-COR" if probe else "C3 + B-COR"
    AssemblyCatalog().apply(state, AssemblySelection("FEG", name, "No Energy Filter"))
    return state


@pytest.mark.parametrize("probe", [False, True])
def test_bcor_restores_typed_hardware_positions_and_independent_drives(probe):
    state = assembled(probe)
    system = state.image_corrector_system
    system.hp3_hexapole.strength_m3 = 123456.5
    system.hp3_hexapole.orientation_rad = .234
    system.tl31_lens.percent = 42.5
    system.tl32_lens.enabled = False
    payload = state.to_dict()
    assert payload["image_corrector_design"] == "bcor"
    restored = State.from_dict(deepcopy(payload))
    for candidate in (state, restored):
        system = candidate.image_corrector_system
        assert tuple(c.key for c in system.components) == BCOR_KEYS
        assert len(system.round_lens_components) == 7
        assert isinstance(system.hp3_hexapole, ImageCorrectorHp3HexapoleComponent)
        assert isinstance(system.tl31_lens, ImageCorrectorTl31LensComponent)
        assert isinstance(system.tl32_lens, ImageCorrectorTl32LensComponent)
        assert system.hp3_hexapole.strength_m3 == 123456.5
        assert system.hp3_hexapole.orientation_rad == .234
        assert system.tl31_lens.percent == 42.5
        assert not system.tl32_lens.enabled
        assert system.hp2_hexapole.z_mm < system.tl31_lens.z_mm < system.tl32_lens.z_mm
        assert system.tl32_lens.z_mm < system.hp3_hexapole.z_mm < system.adapter_lens.z_mm
        for component in system.components:
            part = candidate._resolved_assembly.part(component.key)
            expected_z = (part.center_z_mm + part.data["optical_reference_local_z_mm"]
                          - part.data["local_center_z_mm"])
            assert component.z_mm == pytest.approx(expected_z, abs=1e-10)
    np.testing.assert_array_equal(
        [c.z_mm for c in state.image_corrector_system.components],
        [c.z_mm for c in restored.image_corrector_system.components],
    )


def test_switching_to_cetcor_or_no_corrector_disables_extra_hardware_after_restore():
    state = assembled()
    catalog = AssemblyCatalog()
    catalog.apply(state, AssemblySelection("FEG", "C3 + Image Corrector", "No Energy Filter"))
    for candidate in (state, State.from_dict(state.to_dict())):
        ensure_corrector_structure(candidate)
        assert candidate.image_corrector_design == "cetcor"
        assert tuple(c.key for c in candidate.image_corrector_system.components) == IMAGE_CORRECTOR_KEYS
        extra = [c for c in (*candidate.lenses, *candidate.corrector_elements)
                 if c.key in BCOR_EXTRA_KEYS]
        assert len(extra) == 3
        assert all(not c.enabled and not c._layout_installed for c in extra)
        assert not set(BCOR_EXTRA_KEYS).intersection(p.key for p in candidate._resolved_assembly.parts)
    catalog.apply(state, AssemblySelection("FEG", "C3", "No Energy Filter"))
    assert not any(c.enabled for c in (*state.lenses, *state.corrector_elements)
                   if c.key in BCOR_KEYS)


def test_switching_design_preserves_disabled_extra_hardware_preference():
    state = assembled()
    catalog = AssemblyCatalog()
    state.image_corrector_system.tl31_lens.enabled = False
    state.image_corrector_system.hp3_hexapole.enabled = False
    for name in ("C3 + Image Corrector", "C3 + B-COR"):
        catalog.apply(state, AssemblySelection("FEG", name, "No Energy Filter"),
                      preserve_operating_parameters=True)
    assert not state.image_corrector_system.tl31_lens.enabled
    assert not state.image_corrector_system.hp3_hexapole.enabled
    assert state.image_corrector_system.tl32_lens.enabled


def test_third_hexapole_contributes_nonlinear_field_and_follows_one_geometry():
    state = assembled()
    hp = state.image_corrector_system.hp3_hexapole
    hp.strength_m3, hp.orientation_rad = 2e7, .27
    z = np.array([hp.z_mm-.25, hp.z_mm, hp.z_mm+.25])
    enabled = np.array(hexapole_field_components(z, state))
    expected = np.array(hp.hexapole_strength_components_m3(z))
    hp.enabled = False
    disabled = np.array(hexapole_field_components(z, state))
    np.testing.assert_allclose(enabled-disabled, expected, rtol=1e-13, atol=1e-8)
    assert np.linalg.norm(expected) > 1e7
    hp.enabled = True
    resolve_image_corrector_mechanical_anchors(state)
    previous = hp.z_mm
    hp.mechanical_center_from_tip_mm += .125
    hp.apply_optical_position()
    resolve_image_corrector_mechanical_anchors(state)
    assert hp.z_mm == pytest.approx(previous+.125)
    assert hp.optical_reference_from_tip_mm == hp.z_mm


def test_invalid_design_is_rejected_and_old_inputs_keep_cetcor():
    state = default_state()
    payload = state.to_dict()
    payload.pop("image_corrector_design")
    assert State.from_dict(payload).image_corrector_design == "cetcor"
    payload["image_corrector_design"] = "unrecognized"
    with pytest.raises(ValueError, match="Image corrector design"):
        State.from_dict(payload)


def test_coherent_plan_uses_same_third_hexapole_fields_as_particle_plan():
    from temsim.physics.column_wave import _prepare_column
    state = assembled()
    state.acceleration_enabled = False
    state.acceleration_backend = "CPU"
    state.simulation_mode = "ideal"
    state.step_mm = state.history_step_mm = .1
    hp = state.image_corrector_system.hp3_hexapole
    hp.orientation_rad = .23
    start, end = hp.z_mm-.2, hp.z_mm+.2
    plan = _prepare_column(state, start, end, .1)[0]
    expected_normal, expected_skew = hexapole_field_components(plan.z_mm, state)
    np.testing.assert_array_equal(plan.hex_normal_m3, expected_normal)
    np.testing.assert_array_equal(plan.hex_skew_m3, expected_skew)
    assert np.linalg.norm(plan.hex_normal_m3) > 0
    assert np.linalg.norm(plan.hex_skew_m3) > 0
    hp.enabled = False
    disabled = _prepare_column(state, start, end, .1)[0]
    assert plan.signature != disabled.signature
    assert not np.array_equal(plan.hex_normal_m3, disabled.hex_normal_m3)
