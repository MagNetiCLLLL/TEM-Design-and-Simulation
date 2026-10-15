"""Reject invalid native corrector hardware before edits or field execution."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.gui.parameter_panel import ParameterPanel
from temsim.immutable_json import thaw_json
from temsim.instrument_snapshot import decode_instrument, encode_instrument
from temsim.optics.column import default_state
from temsim.optics.probe_corrector import create_hpc_hexapole, create_qpc_quadrupole
from temsim.physics.core import build_propagation_plan
from temsim.runtime_parameters import RuntimeTarget, validate_runtime_assignment


@pytest.fixture(params=[create_hpc_hexapole, create_qpc_quadrupole], ids=["hexapole", "quadrupole"])
def component(request):
    return request.param()


def _strength_field(component):
    return "strength_m3" if hasattr(component, "strength_m3") else "strength_m2"


@pytest.mark.parametrize("field,value", [
    ("strength", np.nan), ("strength", np.inf),
    ("maximum", np.inf), ("maximum", 0.),
    ("effective_length_mm", np.nan), ("effective_length_mm", 0.),
    ("effective_length_mm", -1.), ("effective_length_mm", 100.),
    ("mechanical_length_mm", np.inf), ("mechanical_center_from_tip_mm", np.nan),
    ("mechanical_outer_diameter_mm", np.inf),
    ("mechanical_clear_bore_diameter_mm", np.nan),
])
def test_native_hardware_rejects_nonfinite_or_invalid_geometry(component, field, value):
    if field == "strength":
        field = _strength_field(component)
    elif field == "maximum":
        field = "maximum_" + _strength_field(component)
    setattr(component, field, value)
    with pytest.raises(ValueError):
        component.validate()


def test_hexapole_orientation_must_be_finite():
    component = create_hpc_hexapole()
    component.orientation_rad = np.nan
    with pytest.raises(ValueError, match="orientation_rad.*finite"):
        component.validate()


@pytest.mark.parametrize("sign", [-1., 1.])
def test_signed_limit_and_full_body_field_length_remain_valid(component, sign):
    strength_field = _strength_field(component)
    strength = sign * getattr(component, "maximum_" + strength_field)
    setattr(component, strength_field, strength)
    component.effective_length_mm = component.mechanical_length_mm
    component.enabled = True
    component.validate()
    field = (component.hexapole_strength_m3 if strength_field == "strength_m3"
             else component.quadrupole_strength_m2)
    # The valid field retains its peak and FWHM; validation cannot rescale it.
    z = component.z_mm + np.array([0., component.effective_length_mm / 2.])
    np.testing.assert_allclose(field(z), strength * np.array([1., np.exp(-.5 * (2.355 / 2.)**2)]))


@pytest.mark.parametrize("width", [-1., 0., np.nan])
def test_invalid_active_width_is_not_silently_turned_into_a_thin_field(component, width):
    component.enabled = True
    component.effective_length_mm = width
    field = (component.hexapole_strength_m3 if hasattr(component, "strength_m3")
             else component.quadrupole_strength_m2)
    with pytest.raises(ValueError, match="length must be finite and positive"):
        field(np.array([component.z_mm]))


@pytest.mark.parametrize("factory,field,value", [
    (create_hpc_hexapole, "strength_m3", 2e9),
    (create_qpc_quadrupole, "strength_m2", 600.),
    (create_hpc_hexapole, "effective_length_mm", -1.),
    (create_qpc_quadrupole, "effective_length_mm", 100.),
])
def test_invalid_gui_edit_keeps_hardware_and_text_unchanged(qtbot, factory, field, value):
    component = factory()
    before = deepcopy(vars(component))
    panel = ParameterPanel()
    qtbot.addWidget(panel)
    panel.set_context(component.name, RuntimeTarget(component.key, component.name, component), None, (), None)
    errors, changes = [], []
    panel.error.connect(errors.append)
    panel.runtime_changed.connect(changes.append)
    row = next(i for i in range(panel.runtime_table.rowCount())
               if panel.runtime_table.item(i, 0).text() == field)
    item = panel.runtime_table.item(row, 1)
    old_text = item.text()
    item.setText(str(value))
    assert errors
    assert not changes
    assert item.text() == old_text
    assert vars(component) == before


def test_invalid_runtime_position_candidate_does_not_move_any_coordinate(component):
    # Position coupling changes three fields: a failed candidate must change none.
    before = deepcopy(vars(component))
    target = RuntimeTarget(component.key, component.name, component)
    with pytest.raises(ValueError, match="mechanical centre must follow the tip"):
        validate_runtime_assignment(target, "mechanical_center_from_tip_mm", -1.)
    assert vars(component) == before


def test_valid_runtime_position_candidate_is_detached(component):
    before = deepcopy(vars(component))
    value = component.mechanical_center_from_tip_mm + 1.
    target = RuntimeTarget(component.key, component.name, component)
    converted = validate_runtime_assignment(target, "mechanical_center_from_tip_mm", value)
    assert converted == value
    assert vars(component) == before
    setattr(component, "mechanical_center_from_tip_mm", converted)
    assert component.z_mm == pytest.approx(before["z_mm"] + 1.)
    assert component.optical_reference_from_tip_mm == pytest.approx(before["optical_reference_from_tip_mm"] + 1.)


@pytest.fixture(scope="module")
def instrument_graph():
    return encode_instrument(default_state())


@pytest.mark.parametrize("key,field,value", [
    ("probe_hpc_hexapole", "effective_length_mm", -1.),
    ("probe_hpc_hexapole", "strength_m3", 2e9),
    ("probe_hpc_hexapole", "maximum_strength_m3", {"float": "inf"}),
    ("probe_hpc_hexapole", "orientation_rad", {"float": "nan"}),
    ("probe_qpc_quadrupole", "strength_m2", 600.),
    ("probe_qpc_quadrupole", "strength_m2", {"float": "nan"}),
    ("probe_qpc_quadrupole", "effective_length_mm", 100.),
])
def test_snapshot_rejects_invalid_native_hardware_even_when_disabled(instrument_graph, key, field, value):
    graph = thaw_json(instrument_graph)
    attributes = next(node["attributes"] for node in graph["nodes"]
                      if node.get("attributes", {}).get("key") == key
                      and field in node["attributes"])
    attributes["enabled"] = False
    attributes[field] = value
    with pytest.raises(ValueError):
        decode_instrument(graph)


def test_valid_snapshot_preserves_signed_drives_and_independent_lengths(instrument_graph):
    state = decode_instrument(instrument_graph)
    state.hpc_hexapole.strength_m3 = -1234.
    state.qpc_quadrupole.strength_m2 = -12.
    state.hpc_hexapole.effective_length_mm = 4.4
    body = state.hpc_hexapole.mechanical_length_mm
    restored = decode_instrument(encode_instrument(state))
    assert restored.hpc_hexapole.strength_m3 == -1234.
    assert restored.qpc_quadrupole.strength_m2 == -12.
    assert restored.hpc_hexapole.effective_length_mm == 4.4
    assert restored.hpc_hexapole.mechanical_length_mm == body


@pytest.mark.parametrize("enabled", [False, True])
def test_plan_rejects_invalid_native_hardware_before_any_field_sampling(component, enabled):
    component.enabled = enabled
    component.effective_length_mm = -1.
    # No gun, field providers or other state are needed: rejection precedes them.
    state = SimpleNamespace(corrector_elements=[component])
    with pytest.raises(ValueError, match="effective length must fit its body"):
        build_propagation_plan(state, component.z_mm - .1, component.z_mm + .1)
