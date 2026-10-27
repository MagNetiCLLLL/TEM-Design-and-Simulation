"""Explicit preview numerics preserve source physics, modes and cache identity."""
from dataclasses import asdict, replace

import numpy as np
import pytest

from temsim.immutable_json import json_digest
from temsim.optics.column import default_state
from temsim.physics.tip_wave_pipeline import TipWaveRequest
from temsim.physics.unified_wave_runtime import UnifiedPotentialWaveAdapter
from temsim.physics.wave_checkpoint_store import ExecutedWaveStore
from test_unified_wave_runtime import _request, _runtime
from test_unified_weighted_resampling import _modes


def _preview(tmp_path, **kwargs):
    request = _request(tmp_path, column_support_projection=True, column_weighted_resampling=True,
                       column_exploratory_preview=True, **kwargs)
    return replace(request, wave_grid=replace(request.wave_grid,
        maximum_support_loss_fraction=1e-3, maximum_accumulated_support_l2=.1))


@pytest.mark.parametrize("value", [True, 31, 8193, 2048., None])
def test_preview_pixel_limit_rejects_implicit_or_invalid_values(value):
    with pytest.raises(ValueError, match="column_preview_pixels"):
        replace(TipWaveRequest(), column_preview_pixels=value).validate()


@pytest.mark.parametrize("value", [1, 0, "true", None, np.bool_(True)])
def test_preview_flag_requires_an_explicit_boolean(value):
    with pytest.raises(ValueError, match="column_exploratory_preview"):
        replace(TipWaveRequest(), column_exploratory_preview=value).validate()


@pytest.mark.parametrize("support,weighted", [(False, False), (True, False), (False, True)])
def test_preview_cannot_bypass_projection_authorization(support, weighted):
    with pytest.raises(ValueError, match="requires controlled support"):
        replace(TipWaveRequest(), column_exploratory_preview=True,
            column_support_projection=support, column_weighted_resampling=weighted).validate()


def test_preview_identity_and_effective_limit_do_not_change_gun_or_optical_fields(tmp_path):
    state = default_state()
    runtime = _runtime(state, 1., 1.000001)
    request = _preview(tmp_path)
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    strict_request = replace(request, column_exploratory_preview=False)
    strict = UnifiedPotentialWaveAdapter(state, strict_request, runtime=runtime)
    assert adapter.column_grid.maximum_pixels == 2048
    assert strict.column_grid.maximum_pixels == request.wave_grid.maximum_pixels == 4096
    assert adapter.column_grid.maximum_incoherent_mode_loss_probability == 0.
    assert adapter.request is request
    assert adapter.snapshot.numerical_digest == strict.snapshot.numerical_digest
    assert adapter.runtime.geometry.digest == strict.runtime.geometry.digest
    assert adapter.operator_options == strict.operator_options
    for name in ("source", "surface", "gun", "radial_gun"):
        assert getattr(adapter.request, name) == getattr(strict.request, name)
    assert adapter.identity != strict.identity
    assert json_digest(asdict(request)) != json_digest(asdict(strict_request))
    reduced_request = replace(request, column_preview_pixels=1024)
    reduced = UnifiedPotentialWaveAdapter(state, reduced_request, runtime=runtime)
    assert reduced.identity != adapter.identity
    capped = UnifiedPotentialWaveAdapter(state, replace(request,
        wave_grid=replace(request.wave_grid, maximum_pixels=512)), runtime=runtime)
    assert capped.column_grid.maximum_pixels == 512
    assert capped.numerical_profile["effective_column_maximum_pixels"] == 512
    assert capped.numerical_profile["accuracy_qualified"] is False
    with pytest.raises(TypeError):
        capped.numerical_profile["accuracy_qualified"] = True


def test_preview_keeps_all_nine_complex_modes_and_labels_completed_cache(tmp_path):
    state = default_state()
    runtime = _runtime(state, 1., 1.0000004)
    request = _preview(tmp_path, column_preview_pixels=256)
    source = _modes(9)
    original_digest = source.digest
    adapter = UnifiedPotentialWaveAdapter(state, request, runtime=runtime)
    output = adapter.column(source, 1000.0002)
    assert source.digest == original_digest
    assert len(output.beam.modes) == len(source.beam.modes) == 9
    assert [mode.mode_id for mode in output.beam.modes] == [mode.mode_id for mode in source.beam.modes]
    for before, after in zip(source.beam.modes, output.beam.modes, strict=True):
        assert after.weight_per_reference_electron == pytest.approx(before.weight_per_reference_electron, rel=1e-13)
        assert after.energy_kev == before.energy_kev
        assert np.isfinite(after.plane.amplitude).all()
    profile = output.record["unified_numerical_profile"]
    assert profile["exploratory_preview"] is True
    assert profile["label"] == "Exploratory preview"
    assert profile["accuracy_qualified"] is profile["normalization_applied"] is False
    assert profile["effective_column_maximum_pixels"] == 256
    assert profile["maximum_support_probability_fraction"] == 1e-3
    assert profile["maximum_accumulated_support_l2"] == .1
    summary = output.record["unified_support_summary"]
    assert profile["numerical_projection_loss_per_reference"] == summary["used_probability_per_reference"]
    assert profile["complex_l2_projection_bound_per_reference"] == summary["used_l2_per_reference"]
    store = ExecutedWaveStore(tmp_path/"preview", "preview-numerical-profile", 2**28)
    restored = store.put(store.key("completed", output.digest), output)
    assert restored.record["unified_numerical_profile"] == profile
    assert adapter.column(restored, restored.plane_z_mm) is restored
    for changed in (replace(request, column_exploratory_preview=False),
                    replace(request, column_preview_pixels=512),
                    replace(request, wave_grid=replace(request.wave_grid, maximum_support_loss_fraction=5e-4))):
        other = UnifiedPotentialWaveAdapter(state, changed, runtime=runtime)
        with pytest.raises(ValueError, match="inputs changed"):
            other.column(restored, restored.plane_z_mm)
