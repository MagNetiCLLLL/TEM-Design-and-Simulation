"""Model switching retains electrode gauges and validates the whole candidate."""
from copy import deepcopy

import numpy as np
import pytest

from temsim import module_manifest
from temsim.instrument_snapshot import encode_instrument
from temsim.optics.electron_gun.field_emission import FieldEmissionGun
from temsim.optics.electron_gun.tip_assembly import model_from_part
from temsim.optics.electron_gun.tip_coherence import TipCoherence
from temsim.optics.electron_gun.tip_edit import candidate_tip_edit
from temsim.physics.closed_gun_field import closed_field_request
from temsim.physics.continuous_gun_field import continuous_field_request
from temsim.physics.grounded_tip_field import field_request


def surface_gun(reference="extractor", monochromator=False):
    gun = FieldEmissionGun()
    if monochromator:
        gun.install_monochromator()
    gun.emitter.surface_model = model_from_part(module_manifest.part_data("gun/FEG.toml", "feg_tip"))
    gun.electrostatic_lens.voltage_reference = reference
    gun.extractor.voltage_kv = 4.5
    gun.electrostatic_lens.voltage_kv = 1.1
    return gun


@pytest.mark.parametrize("reference,expected_rise", [("tip", 1100.), ("extractor", 5600.), ("ground", 301100.)])
@pytest.mark.parametrize("curvature", [0., 1e-8])
@pytest.mark.parametrize("monochromator", [False, True])
def test_legal_surface_switch_retains_physical_electrode_potentials_and_round_trips(reference, expected_rise, curvature, monochromator):
    gun = surface_gun(reference, monochromator)
    model = gun.emitter.surface_model
    before = gun.to_dict()
    before_graph = encode_instrument(gun)
    before_request = field_request(gun)
    original_cache = object()
    gun._trace_cache = original_cache
    gun._trace_cache_key = "prior calculated state"

    off = candidate_tip_edit(gun, {"surface_model": None, "coherence": None, "curvature_nm_inv": curvature})
    assert off.uses_geometry_electric_field
    assert off.to_dict()["integrator"]["method"] == "static_discrete_gradient"
    request = continuous_field_request(off) if curvature else closed_field_request(off)
    # The closed field also contains the grounded outlet liner and optional
    # Wien housing. Compare the same gun electrodes, not those extra domains.
    by_key = {row["key"]: row for row in request["rings"]}
    electrode_keys = ["extractor", "electrostatic_lens", *(
        f"accelerator:{index}" for index in range(len(gun.accelerator.stages)))]
    actual = [[by_key[key][name] for name in
               ("start_m", "stop_m", "inner_m", "outer_m", "potential_rise_v")]
              for key in electrode_keys]
    np.testing.assert_array_equal(actual, [row[1:] for row in before_request["rings"]])
    assert by_key["electrostatic_lens"]["potential_rise_v"] == expected_rise
    assert off.electrostatic_lens.voltage_reference == reference
    for key, component in before["components"].items():
        if key != gun.emitter.key:
            assert off.to_dict()["components"][key] == component
    assert off._trace_cache is None and off._trace_cache_key is None
    assert off.c1_aperture is not gun.c1_aperture
    assert off.emitter is not gun.emitter
    assert encode_instrument(gun) == before_graph
    assert gun._trace_cache is original_cache and gun._trace_cache_key == "prior calculated state"

    off_before = encode_instrument(off)
    on = candidate_tip_edit(off, {"surface_model": model, "curvature_nm_inv": 0.})
    assert encode_instrument(off) == off_before
    assert on.to_dict() == before
    assert field_request(on) == before_request


@pytest.mark.parametrize("reference", ["tip", "extractor", "ground"])
def test_dialog_accepts_legal_gauges_without_changing_live_source(reference, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = surface_gun(reference)
    before = encode_instrument(gun)
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(False)
    dialog.accept()
    assert dialog.result() == dialog.DialogCode.Accepted
    assert not dialog.error.text()
    assert dialog.value()["surface_model"] is None
    candidate = candidate_tip_edit(gun, dialog.value())
    assert candidate.electrostatic_lens.voltage_reference == reference
    assert candidate.electrostatic_lens.voltage_kv == gun.electrostatic_lens.voltage_kv
    assert encode_instrument(gun) == before


@pytest.mark.parametrize("invalid", ["voltage_reference", "electrode_geometry", "inactive_emission"])
def test_invalid_off_transition_is_rejected_by_both_entry_points_without_mutation(invalid, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = surface_gun()
    if invalid == "voltage_reference":
        gun.electrostatic_lens.voltage_reference = "undefined"
        expected = "voltage reference"
    elif invalid == "electrode_geometry":
        gun.extractor.mechanical_clear_bore_diameter_mm = gun.extractor.mechanical_outer_diameter_mm
        expected = "electrode geometry"
    else:
        # The surface source does not use this stored continuous-source law.
        # It becomes active when surface emission is turned off.
        gun.emitter.minimum_kinetic_energy_ev = gun.emitter.emission_energy_ev
        expected = "minimum kinetic energy"
    before_graph = encode_instrument(gun)
    before_components = deepcopy(gun.to_dict())
    old_cache = object()
    gun._trace_cache = old_cache
    gun._trace_cache_key = "retained state"
    with pytest.raises(ValueError, match=expected):
        candidate_tip_edit(gun, {"surface_model": None, "coherence": None})
    assert encode_instrument(gun) == before_graph
    assert gun.to_dict() == before_components
    assert gun._trace_cache is old_cache and gun._trace_cache_key == "retained state"

    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(False)
    dialog.accept()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert expected in dialog.error.text()
    assert dialog._value is None
    with pytest.raises(ValueError, match="No tip emission values"):
        dialog.value()
    assert encode_instrument(gun) == before_graph
    assert gun._trace_cache is old_cache and gun._trace_cache_key == "retained state"


@pytest.mark.parametrize("boundary,width,tilt,accepted", [
    ("forward_gaussian_schell", 5., 0., False),
    ("forward_gaussian_schell", 100., 0., True),
    ("driven_gaussian_schell", 5., 0., True),
    ("driven_gaussian_schell", 5., 1000., False),
])
def test_gaussian_particle_admission_is_identical_in_dialog_and_backend(boundary, width, tilt, accepted, qtbot):
    from temsim.gui.gun_source_dialog import GunSourceDialog
    gun = surface_gun()
    before = encode_instrument(gun)
    old_cache = object()
    gun._trace_cache = old_cache
    gun._trace_cache_key = "prior source"
    values = {"surface_model": None, "coherence": TipCoherence(boundary_model=boundary, tilt_x_mrad=tilt),
              "virtual_source_fwhm_nm": width}
    if accepted:
        candidate = candidate_tip_edit(gun, values)
        assert candidate.emitter.coherence == values["coherence"]
        assert candidate.emitter.virtual_source_fwhm_nm == width
        assert candidate._trace_cache is None
    else:
        with pytest.raises(ValueError, match="forward support"):
            candidate_tip_edit(gun, values)
    dialog = GunSourceDialog(gun)
    qtbot.addWidget(dialog)
    dialog.surface_enabled.setChecked(False)
    dialog.inputs["virtual_source_fwhm_nm"].setText(str(width))
    dialog.coherence_enabled.setChecked(True)
    dialog.tip_boundary.setCurrentIndex(dialog.tip_boundary.findData(boundary))
    dialog.coherence_inputs["tilt_x_mrad"].setText(str(tilt))
    dialog.accept()
    assert (dialog.result() == dialog.DialogCode.Accepted) == accepted
    if accepted:
        assert dialog.value()["coherence"] == values["coherence"]
        assert dialog.value()["virtual_source_fwhm_nm"] == width
    else:
        assert "forward support" in dialog.error.text()
        assert dialog._value is None
    assert encode_instrument(gun) == before
    assert gun._trace_cache is old_cache and gun._trace_cache_key == "prior source"
