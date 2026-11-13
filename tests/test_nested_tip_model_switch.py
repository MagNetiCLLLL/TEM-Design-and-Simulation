"""Explicit source-family edits with nested defaults; no particle transport."""
from copy import deepcopy

import pytest

from temsim import module_manifest
from temsim.instrument_snapshot import encode_instrument
from temsim.optics.electron_gun.emitter import EmissionCDFQuadrature, EmissionSampling, HALTON_PREFIX_SAMPLING_SCHEMA
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_assembly import apply_tip_part, model_from_part
from temsim.optics.electron_gun.tip_coherence import TipCoherence
from temsim.optics.electron_gun.tip_edit import candidate_tip_edit


def tip_part():
    return deepcopy(module_manifest.part_data("gun/FEG.toml", "feg_tip"))


@pytest.mark.parametrize("family", ["surface", "coherent"])
def test_explicit_model_switch_replaces_only_default_plan_and_returns_to_nested_flat(family):
    gun = FieldEmissionGun()
    assert gun.emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    original = encode_instrument(gun)
    original_controls = gun.to_dict()
    old_cache = object()
    gun._trace_cache = old_cache
    values = ({"surface_model": model_from_part(tip_part())} if family == "surface" else
              {"coherence": TipCoherence(boundary_model="driven_gaussian_schell")})

    switched = candidate_tip_edit(gun, values)
    assert switched.emitter.sampling is None
    assert encode_instrument(gun) == original
    assert gun._trace_cache is old_cache
    assert switched._trace_cache is None
    restored = candidate_tip_edit(switched, {"surface_model": None, "coherence": None})
    assert restored.emitter.sampling == gun.emitter.sampling
    assert restored.to_dict() == original_controls


def test_regular_legacy_edit_keeps_legacy_sampling_and_explicit_sobol_stays_exclusive():
    gun = FieldEmissionGun()
    gun.emitter.sampling = None
    changed = candidate_tip_edit(gun, {"energy_spread_fwhm_ev": .3})
    assert changed.emitter.sampling is None
    gun.emitter.ray_count = 1024
    gun.emitter.sampling = EmissionSampling(seed=7)
    original = encode_instrument(gun)
    with pytest.raises(ValueError, match="Sobol tip sampling cannot be combined"):
        candidate_tip_edit(gun, {"surface_model": model_from_part(tip_part())})
    assert encode_instrument(gun) == original


def test_explicit_catalog_reset_selects_default_plan_without_migrating_geometry_refresh():
    gun = FieldEmissionGun()
    part = tip_part()
    part["default_tip_emission"] = "curved_surface_particles"
    apply_tip_part(gun.emitter, part, reset_source=True)
    assert gun.emitter.sampling is None
    assert gun.emitter.surface_model == model_from_part(part)
    gun.validate()
    part["default_tip_emission"] = "flat_tip"
    apply_tip_part(gun.emitter, part, reset_source=True)
    assert gun.emitter.surface_model is None
    assert gun.emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    gun.validate()
    gun.emitter.sampling = None
    before = gun.to_dict()
    apply_tip_part(gun.emitter, part)
    assert gun.to_dict() == before


def test_source_dialog_can_switch_fresh_nested_flat_to_surface_and_back(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog

    gun = FieldEmissionGun()
    original = encode_instrument(gun)
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(True)
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    curved = candidate_tip_edit(gun, dialog.value())
    assert curved.emitter.surface_model is not None
    assert curved.emitter.sampling is None
    assert encode_instrument(gun) == original
    back = GunSourceDialog(curved)
    qtbot.addWidget(back)
    back.surface_enabled.setChecked(False)
    back.accept()
    assert back.result() == back.DialogCode.Accepted, back.error.text()
    restored = candidate_tip_edit(curved, back.value())
    assert restored.emitter.surface_model is None
    assert restored.emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    assert restored.emitter.energy_spread_fwhm_ev == gun.emitter.energy_spread_fwhm_ev == .3


def test_historical_flat_dialog_requires_explicit_new_sampling_selection(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog

    gun = FieldEmissionGun()
    gun.emitter.sampling = None
    original = gun.to_dict()
    kept = GunSourceDialog(gun)
    qtbot.addWidget(kept)
    kept.accept()
    assert kept.result() == kept.DialogCode.Accepted, kept.error.text()
    assert candidate_tip_edit(gun, kept.value()).emitter.sampling is None
    selected = GunSourceDialog(gun)
    qtbot.addWidget(selected)
    assert selected.nested_sampling_button.isEnabled()
    selected.nested_sampling_button.click()
    assert "Apply starts a new calculation" in selected.source_sampling_status.text()
    assert gun.to_dict() == original
    selected.accept()
    assert selected.result() == selected.DialogCode.Accepted, selected.error.text()
    current = candidate_tip_edit(gun, selected.value())
    expected = deepcopy(original)
    expected["components"]["feg_tip"]["sampling"] = {"schema": HALTON_PREFIX_SAMPLING_SCHEMA, "seed": 0}
    assert current.to_dict() == expected
    assert gun.to_dict() == original


def test_explicit_new_sampler_can_replace_saved_sobol_without_silent_ordinary_conversion(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog

    gun = FieldEmissionGun()
    gun.emitter.ray_count = 1024
    gun.emitter.sampling = EmissionSampling(seed=123)
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.nested_sampling_button.click()
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    current = candidate_tip_edit(gun, dialog.value())
    assert current.emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    assert gun.emitter.sampling == EmissionSampling(seed=123)


def test_saved_table_stays_exclusive_and_requires_explicit_replacement(qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog

    gun = FieldEmissionGun()
    gun.emitter.sampling = None
    gun.emitter.ray_count = 9
    table = EmissionCDFQuadrature(
        coordinates=((.5, .5, .5, .5, .5, .5),)*9, weights=(1/9,)*9,
        signs=(1,)*9, ray_ids=tuple(range(9)))
    gun.emitter.cdf_quadrature = table
    with pytest.raises(ValueError, match="cannot be combined"):
        candidate_tip_edit(gun, {"surface_model": model_from_part(tip_part())})
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.nested_sampling_button.click()
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted, dialog.error.text()
    replacement = candidate_tip_edit(gun, dialog.value())
    assert replacement.emitter.sampling.schema == HALTON_PREFIX_SAMPLING_SCHEMA
    assert replacement.emitter.cdf_quadrature is None
    assert gun.emitter.cdf_quadrature is table
    assert gun.emitter.sampling is None


def test_ordinary_edit_does_not_repair_an_invalid_mixed_family_record():
    gun = FieldEmissionGun()
    gun.emitter.surface_model = model_from_part(tip_part())
    before = encode_instrument(gun)
    with pytest.raises(ValueError, match="Halton prefix tip sampling cannot be combined"):
        candidate_tip_edit(gun, {"energy_spread_fwhm_ev": .3})
    assert encode_instrument(gun) == before
