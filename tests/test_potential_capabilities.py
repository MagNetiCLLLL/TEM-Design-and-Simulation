"""No field solve or implicit source admission in a coverage report."""
from copy import deepcopy

import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics.potential_capabilities import potential_capability_inventory


@pytest.mark.parametrize("variant", ("C2", "C3", "C3_ImageCorrector", "C3_ProbeCorrector",
    "C3_ProbeCorrector_ImageCorrector", "C3_BCOR", "C3_ProbeCorrector_BCOR"))
def test_inventory_covers_actual_channels_without_solving_or_changing_inputs(variant, monkeypatch):
    state = default_state()
    catalog = AssemblyCatalog()
    column = next(c for c in catalog.columns if c.file == f"column/{variant}.toml")
    catalog.apply(state, AssemblySelection("FEG", column.name, "No Energy Filter"))
    before = deepcopy(state.to_dict())
    def forbidden(*args, **kwargs):
        raise AssertionError("Inventory must not solve a field")
    monkeypatch.setattr("temsim.physics.potential_capture.capture_round_lens_potential", forbidden)
    report = potential_capability_inventory(state)
    rows = {r["key"]: r for r in report["components"]}
    assert len(rows) == len(report["components"])
    assert {c.key for c in (*state.lenses, *state.stigmators, *state.deflectors, *state.corrector_elements)} <= set(rows)
    assert state.to_dict() == before
    assert not any(row["production_admission"] for row in rows.values())
    from temsim.component_representation import SHARED_DEFLECTOR_HOSTS
    for alias, host in SHARED_DEFLECTOR_HOSTS.items():
        if alias in rows:
            assert rows[alias]["physical_owner"] == host
            assert rows[alias]["classical"] == "no independent field"
    assert rows["energy_filter"]["coherent"] == "later capability"
    assert rows["nanopulser"]["coherent"] == "later capability"
    assert rows["sample"]["family"] == "material"
    assert rows["receivers"]["family"] == "measurement"


def test_imported_b_only_recipe_never_gains_wave_admission():
    state = default_state()
    state.simulation_mode = "custom"
    key = state.lenses[0].key
    state.lens_field_map_descriptors[key] = {"source": "unresolved-b-only-map"}
    rows = {r["key"]: r for r in potential_capability_inventory(state)["components"]}
    assert rows[key]["coherent"] == "requires retained potential adapter"


def test_electric_tail_and_native_timed_coils_are_inventory_capabilities_not_blanket_admission():
    from temsim.optics.ac_deflector import AcDeflectorComponent
    from temsim.optics.descan_deflector import DescanDeflectorComponent
    state = default_state()
    for component in state.deflectors:
        if type(component) in (AcDeflectorComponent, DescanDeflectorComponent):
            component.scan_enabled = True
    before = deepcopy(state.to_dict())
    rows = {row["key"]: row for row in potential_capability_inventory(state)["components"]}
    assert "positive-branch electric carrier" in rows["gun_electric"]["coherent"]
    timed = [row for row in rows.values() if "retarded scalar wave" in row["coherent"]]
    assert timed
    assert all("flyback" in row["detail"] and "dynamic material" in row["detail"] for row in timed)
    assert not any(row["production_admission"] for row in rows.values())
    assert state.to_dict() == before

