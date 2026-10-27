"""Native gun alignment potential capture, not a new coil-field solution."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.optics.electron_gun.alignment import GunDeflector, GunStigmator
from temsim.physics.gun_magnetic_potential import GunMagneticPotential
from temsim.physics.potential_capture import capture_selected_column_potentials
from temsim.physics.potential_snapshot import ElectromagneticPotentialSnapshot


def _deflector(**kwargs):
    return GunDeflector(20., 30., 10., 4., 15., 25., 3.,
        upper_field_x_mt=.3, upper_field_y_mt=-.5, lower_field_x_mt=-.1, lower_field_y_mt=.7, **kwargs)


def _stigmator(**kwargs):
    return GunStigmator(35., 8., 10., 4., 4., gradient_t_per_m=.7, rotation_deg=23., **kwargs)


@pytest.mark.parametrize("factory", [_deflector, _stigmator])
@pytest.mark.parametrize("enabled", [True, False])
def test_potential_curl_matches_native_field_with_original_soft_edges(factory, enabled):
    source = factory(enabled=enabled)
    captured = GunMagneticPotential.from_provider(source)
    z = np.linspace(0., .05, 311)
    points = np.column_stack((np.full_like(z, .0003), np.full_like(z, -.0004), z))
    sampled = captured.sample(points)
    np.testing.assert_allclose(sampled["B_t"], source.field_at_global_positions_t(points), rtol=3e-14, atol=2e-18)
    step = 1e-8
    gradient = np.stack([(captured.sample(points+np.eye(3)[axis]*step)["A_t_m"]
                         -captured.sample(points-np.eye(3)[axis]*step)["A_t_m"])/(2*step)
                        for axis in range(3)], axis=-1)
    curl = np.column_stack((gradient[:, 2, 1]-gradient[:, 1, 2], gradient[:, 0, 2]-gradient[:, 2, 0],
                            gradient[:, 1, 0]-gradient[:, 0, 1]))
    np.testing.assert_allclose(curl, sampled["B_t"], rtol=2e-9, atol=2e-15)
    np.testing.assert_allclose(gradient, captured.sample_jet(points)["jac_A_t"], rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize("enabled", [True, False])
def test_static_gun_blank_replaces_saved_alignment_without_overwriting_it(enabled):
    source = _deflector(enabled=enabled, beam_blanked=True)
    before = source.upper_field_y_mt, source.lower_field_y_mt
    captured = GunMagneticPotential.from_provider(source)
    points = np.array(((0., 0., .015), (0., 0., .025)))
    np.testing.assert_allclose(captured.sample(points)["B_t"], source.field_at_global_positions_t(points), atol=0)
    np.testing.assert_allclose(captured.sample(points)["B_t"], ((0., .05, 0.), (0., 0., 0.)), atol=0)
    assert (source.upper_field_y_mt, source.lower_field_y_mt) == before
    assert captured.to_metadata()["provenance"]["blanking_replaces_alignment"] is True


@pytest.mark.parametrize("factory", [_deflector, _stigmator])
def test_detached_record_domain_and_native_zero_exterior(factory):
    source = factory()
    captured = GunMagneticPotential.from_provider(source)
    record = captured.to_dict()
    restored = GunMagneticPotential.from_dict(json.loads(json.dumps(record)))
    center = source.optical_reference_from_tip_mm*.001
    # The double deflector's body centre lies between its coils.
    if type(source) is GunDeflector:
        center = source.upper_center_from_tip_mm*.001
    with pytest.raises(ValueError, match="bore"):
        restored.sample((.003, 0., center))
    np.testing.assert_array_equal(restored.sample((100., -100., 1.))["B_t"], np.zeros(3))
    source.field_center_offset_mm += 1
    assert captured.to_dict() == record and restored.digest == captured.digest
    assert GunMagneticPotential.from_provider(source).digest != captured.digest
    corrupt = deepcopy(record)
    corrupt["metadata"]["terms"][0][3] += 1.
    with pytest.raises(ValueError, match="identity"):
        GunMagneticPotential.from_dict(corrupt)


@pytest.mark.parametrize("factory", [_deflector, _stigmator])
def test_unknown_native_override_and_nonboolean_enabled_are_rejected(factory):
    source = factory()
    source.field_at_global_positions_t = lambda points: np.ones_like(points)
    with pytest.raises(ValueError, match="unchanged native"):
        GunMagneticPotential.from_provider(source)
    source = factory()
    source.enabled = "false"
    with pytest.raises(ValueError, match="boolean"):
        GunMagneticPotential.from_provider(source)


def test_selected_gun_alignment_capture_owns_each_original_source_once():
    gun = SimpleNamespace(deflector=_deflector(), stigmator=_stigmator())
    state = SimpleNamespace(electron_gun=gun)
    snapshot = capture_selected_column_potentials(state, include_gun_alignment=True)
    assert len(snapshot.contributions) == 2
    restored = ElectromagneticPotentialSnapshot.from_dict(json.loads(json.dumps(snapshot.to_dict())))
    points = np.array(((.0001, .0002, .015), (.0001, .0002, .035)))
    expected = gun.deflector.field_at_global_positions_t(points)+gun.stigmator.field_at_global_positions_t(points)
    np.testing.assert_allclose(restored.sample(points)["B_t"], expected, rtol=1e-14, atol=1e-17)
    gun.stigmator.gradient_t_per_m *= 2
    assert capture_selected_column_potentials(state, include_gun_alignment=True).digest != snapshot.digest


@pytest.mark.parametrize("point", [(np.nan, 0, 0), (0, 0), (True, False, True), (1j, 0, 0)])
def test_invalid_coordinates_fail(point):
    with pytest.raises(ValueError, match="finite real XYZ"):
        GunMagneticPotential.from_provider(_deflector()).sample(point)
