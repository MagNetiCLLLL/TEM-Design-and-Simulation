"""Physical-owner capture and honest support gates for all seven columns."""
import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.optics.column import default_state
from temsim.physics.unified_field_runtime import (
    UnifiedFieldRuntime, capture_unified_field_runtime, get_unified_field_runtime,
    unified_runtime_preflight, unified_source_selection,
)


def parked_state(column=None):
    state = default_state()
    if column is not None:
        AssemblyCatalog().apply(state,AssemblySelection("FEG",column,"No Energy Filter"))
    state.simulation_mode="unified"
    for item in (*state.deflectors,*state.stigmators,*state.corrector_elements):
        if hasattr(item,"scan_enabled"): item.scan_enabled=False
        if hasattr(item,"wobble_enabled"): item.wobble_enabled=False
    state.acceleration_backend="CPU"
    return state


@pytest.mark.parametrize("column",[option.name for option in AssemblyCatalog().columns])
def test_seven_actual_columns_capture_every_selected_source_and_shared_host_once(column):
    state=parked_state(column)
    report=unified_runtime_preflight(state)
    assert report["supported"],report
    runtime=capture_unified_field_runtime(state)
    lenses,auxiliary=unified_source_selection(state)
    owners={owner.split(":")[0] for row in runtime.snapshot.contributions for owner in row.owner_ids}
    assert set(lenses)|set(auxiliary) <= owners
    assert "descan_deflector" not in owners
    assert "image_diffraction_deflector" in owners
    assert "instrument_electric" in owners
    assert runtime.geometry.source_scene_geometry_digest==runtime.snapshot.scene.geometry_digest
    assert runtime.metadata["empirical_aberrations_applied"] is False
    assert runtime.metadata["equivalent_image_maps_applied"] is False


def test_runtime_roundtrip_and_lens_edits_invalidate_common_capture():
    from temsim.instrument_snapshot import capture_instrument_snapshot
    state=parked_state()
    first=get_unified_field_runtime(state)
    assert get_unified_field_runtime(state) is first
    clone=capture_instrument_snapshot(state).restore()
    assert get_unified_field_runtime(clone) is first
    restored=UnifiedFieldRuntime.from_dict(first.to_dict())
    assert restored.digest==first.digest
    points=np.array([[1e-6,-2e-6,.8]])
    np.testing.assert_array_equal(restored.snapshot.sample(points)["B_t"],first.snapshot.sample(points)["B_t"])
    lens=next(item for item in state.lenses if item.key=="condenser_lens_2")
    lens.percent*=1.01
    with pytest.raises(ValueError,match="changed"):
        first.require_state(state)
    second=get_unified_field_runtime(state)
    assert first.snapshot.numerical_digest!=second.snapshot.numerical_digest
    assert first.digest!=second.digest


def test_native_dynamic_requests_are_retained_and_b_only_requests_are_rejected():
    state=parked_state()
    state.ac_deflector.scan_enabled=True
    assert unified_runtime_preflight(state)["supported"]
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    assert type(get_unified_field_runtime(state).snapshot) is DynamicPotentialSnapshot
    state.ac_deflector.scan_enabled=False
    state.lens_field_map_descriptors={"condenser_lens_2":{"solver":"external_B_only"}}
    with pytest.raises(ValueError,match="B-only"):
        get_unified_field_runtime(state)


def test_default_dynamic_scan_is_not_silently_parked_but_exact_zero_waveform_is_static():
    state=default_state()
    state.simulation_mode="unified"
    assert unified_runtime_preflight(state)["supported"]
    for item in (state.ac_deflector,state.descan_deflector):
        item.set_scan_command_matrix_mrad(((0.,0.),(0.,0.)))
    assert unified_runtime_preflight(state)["supported"]
    runtime=get_unified_field_runtime(state)
    for row in runtime.snapshot.contributions:
        assert not getattr(getattr(getattr(row.potential,"base",None),"field",None),"dynamic",False)
    assert state.ac_deflector.scan_enabled and state.descan_deflector.scan_enabled
    state.ac_deflector.set_scan_command_matrix_mrad(((1e-30,0.),(0.,0.)))
    assert unified_runtime_preflight(state)["supported"]
    from temsim.physics.dynamic_potential import DynamicPotentialSnapshot
    assert type(get_unified_field_runtime(state).snapshot) is DynamicPotentialSnapshot
