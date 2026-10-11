"""Legacy pupils must not silently omit or reinterpret local lens coefficients."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.column import default_state
from temsim.physics.source_admission import (
    UnsupportedIntrinsicLensLegacy, require_gun_wave_source,
    require_legacy_aberration_compatibility,
)


def test_legacy_cs_comparison_excludes_new_terms_without_mutating_live_state(monkeypatch):
    from temsim.optics.aberrations import effective_aberration_comparison
    import temsim.physics.core as core
    import temsim.physics.instrument_magnetic as magnetic

    state = default_state()
    state.lens_intrinsic_aberrations = {state.lenses[0].key: {"a1_mm": .125, "c5_mm": .5}}
    original = deepcopy(state.lens_intrinsic_aberrations)
    operating = [(lens.key, lens.percent, lens.polarity, lens.cs_mm, lens.cc_mm)
                 for lens in state.lenses]
    inputs_seen = []

    def trace(snapshot, z0, z1, x, tx, y, ty, **kwargs):
        inputs_seen.append(deepcopy(snapshot.lens_intrinsic_aberrations))
        assert snapshot is not state
        factor = (0. if not kwargs["include_spherical_aberration"] else
                  .25 if kwargs["include_hexapole"] else 1.)
        zero = np.zeros((2, len(x)))
        return np.array((z0, z1)), zero+factor, zero, zero, zero

    monkeypatch.setattr(core, "propagate", trace)
    monkeypatch.setattr(magnetic, "active_column_events", lambda _state: ())
    before, after, diagnostics = effective_aberration_comparison(state, "probe")
    assert inputs_seen == [{}, {}, {}]
    assert after.c3_mm == pytest.approx(before.c3_mm*.25)
    assert "excluded from this legacy" in diagnostics["diagnostic_scope"]
    assert state.lens_intrinsic_aberrations == original
    assert [(lens.key, lens.percent, lens.polarity, lens.cs_mm, lens.cc_mm)
            for lens in state.lenses] == operating


@pytest.mark.parametrize("product", ["TEM image", "STEM image", "TEM reprojection"])
def test_legacy_imaging_admission_names_supported_full_column_route(product):
    state = default_state()
    lens = state.lenses[0]
    state.lens_intrinsic_aberrations = {lens.key: {"a1_mm": .01}}
    original = deepcopy(state.lens_intrinsic_aberrations)
    with pytest.raises(UnsupportedIntrinsicLensLegacy, match="Electron beam / Calculate beam") as caught:
        require_gun_wave_source(state, product=product)
    assert lens.key in str(caught.value)
    assert "not ignored" in str(caught.value)
    assert state.lens_intrinsic_aberrations == original


@pytest.mark.parametrize("kind", ["empty", "zero", "disabled"])
def test_inactive_terms_do_not_build_field_providers(monkeypatch, kind):
    import temsim.physics.lens_field_provider as fields

    state = default_state()
    lens = state.lenses[0]
    state.lens_intrinsic_aberrations = (
        {} if kind == "empty" else {lens.key: {"a1_mm": 0. if kind == "zero" else .1}}
    )
    if kind == "disabled":
        lens.enabled = False

    def unexpected(_state):
        raise AssertionError("Inactive legacy guard must not resolve fields")

    monkeypatch.setattr(fields, "active_mapped_providers", unexpected)
    require_legacy_aberration_compatibility(state, product="Legacy")


def test_matching_field_map_does_not_activate_its_native_coefficients(monkeypatch):
    import temsim.physics.lens_field_provider as fields

    state = default_state()
    lens = state.lenses[0]
    state.lens_intrinsic_aberrations = {lens.key: {"c5_mm": .1}}
    monkeypatch.setattr(fields, "active_mapped_providers", lambda _state: (SimpleNamespace(lens_key=lens.key),))
    require_legacy_aberration_compatibility(state, product="Legacy")
    assert state.lens_intrinsic_aberrations[lens.key]["c5_mm"] == .1


def test_legacy_camera_guard_covers_only_its_post_sample_interval():
    state = default_state()
    condenser = state.lenses[0]
    downstream = next(lens for lens in state.lenses if lens.key == "diffraction_lens")
    state.lens_intrinsic_aberrations = {condenser.key: {"b2_mm": .1}}
    require_legacy_aberration_compatibility(state, product="Legacy camera",
        start_z_mm=state.sample.z_mm, stop_z_mm=state.camera.z_mm)
    state.lens_intrinsic_aberrations[downstream.key] = {"b2_mm": .1}
    with pytest.raises(UnsupportedIntrinsicLensLegacy, match=downstream.key):
        require_legacy_aberration_compatibility(state, product="Legacy camera",
            start_z_mm=state.sample.z_mm, stop_z_mm=state.camera.z_mm)


@pytest.mark.parametrize("route", ["camera", "multiplane"])
def test_legacy_projection_rejects_before_wave_preparation(route):
    state = default_state()
    lens = next(lens for lens in state.lenses if lens.key == "diffraction_lens")
    state.lens_intrinsic_aberrations = {lens.key: {"a2_mm": .1}}
    state.camera.inserted = True
    with pytest.raises(UnsupportedIntrinsicLensLegacy, match="full-column coherent"):
        if route == "camera":
            from temsim.physics.camera_wave import _project_wave_to_plane
            _project_wave_to_plane(state, None, None, None, .025, recording_plane=state.camera)
        else:
            from temsim.physics.multiplane_wave import project_through_apertures
            project_through_apertures(state, None, None, None, 2.5e-12, state.camera.z_mm)
