"""Bounded material planning and bit-preserving storage; no physical wave trial."""
from dataclasses import replace
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from temsim.physics import mixed_stem_scan as scan
from temsim.physics.mixed_stem_operators import ScanPacket, periodic_coefficients
from test_mixed_stem_scan_operators import NumpyDevice


def test_explicit_material_reference_continues_at_requested_levels_without_repeating_low_grids():
    levels = [[1920, 800], [2160, 1600]]
    request = scan.MixedStemScanRequest(material_levels=levels).validate()
    levels[0][0] = 960
    assert request.material_levels == ((1920, 800), (2160, 1600))
    assert scan._material_level_plan(request, (61., 54.), .0197, (8, 10), 10.) == ((1920, 800), (2160, 1600))
    assert request.maximum_working_bytes == 24*1024**3
    assert request.source.maximum_working_bytes == 24*1024**3
    assert request.maximum_host_working_bytes == 48*1024**3


@pytest.mark.parametrize("levels", [(), ((1920, 800),), ((1920, 800), (1920, 800)),
    ((1920, 800), (1800, 1600)), ((1920, 800), (2160, 799)),
    ((True, 800), (2160, 1600)), ((1920, 0), (2160, 1600)), "invalid"])
def test_explicit_material_levels_refuse_coarsening_duplicates_or_invalid_controls(levels):
    with pytest.raises(ValueError):
        scan.MixedStemScanRequest(material_levels=levels).validate()


def test_commensurate_rounding_cannot_erase_a_requested_refinement():
    request = scan.MixedStemScanRequest(material_levels=((1921, 800), (1922, 800))).validate()
    with pytest.raises(ValueError, match="collapses"):
        scan._material_level_plan(request, (61., 54.), .0197, (32, 20), 10.)


def test_streamed_1920_by_800_plan_fits_separate_24_gib_device_and_host_stages():
    plan = scan._scan_memory_plan(1920, 800, 8, 40000, 96, 16*96*512**2, tiles=(8, 10))
    assert 0 < plan["device_working_bytes"] < 24*1024**3
    assert 0 < plan["host_stage_peak_bytes"] < 24*1024**3
    assert plan["host_potential_storage_bytes"] == 4*800*1920**2//80


def test_nonzero_material_centre_uses_primitive_phase_without_full_volume_fft_work():
    arguments = (1920, 800, 8, 40000, 96, 16*96*512**2)
    plain = scan._scan_memory_plan(*arguments, tiles=(8, 10))
    shifted = scan._scan_memory_plan(*arguments, tiles=(8, 10), shifted=True)
    assert shifted["device_working_bytes"] == plain["device_working_bytes"]
    assert shifted==plain


def test_host_reservation_does_not_raise_the_independent_device_or_source_limit():
    request = replace(scan.MixedStemScanRequest(), maximum_host_working_bytes=96*1024**3).validate()
    assert request.maximum_working_bytes == request.source.maximum_working_bytes == 24*1024**3
    assert request.maximum_host_working_bytes == 96*1024**3


def test_default_material_ladder_can_refine_beyond_the_old_1920_limit():
    levels = scan._material_level_plan(scan.MixedStemScanRequest(), (61., 54.), .0197, (8, 10), 10.)
    assert 2160 in [pixels for pixels, _ in levels]
    assert max(pixels for pixels, _ in levels) >= 2560
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(levels, levels[1:]))


@pytest.fixture
def fake_periodic_builder(monkeypatch):
    stored = {}
    class Potential:
        def __init__(self, atoms, **kwargs):
            self.shape = kwargs["gpts"]
            self.dz = kwargs["slice_thickness"]
            self.sampling = (1., 1.)
            self.integrator = SimpleNamespace(get_integral_table=lambda *_:
                SimpleNamespace(radial_gpts=np.array([0., 1.])))
        def build(self, *, first_slice, last_slice, lazy):
            values = np.random.default_rng(174).normal(size=(last_slice-first_slice, *self.shape)).astype(np.float32)
            values.flat[0] = -0.
            stored["primitive"] = values.transpose(0, 2, 1).copy()
            return SimpleNamespace(array=values, slice_thickness=np.full(last_slice-first_slice, self.dz))
    monkeypatch.setitem(sys.modules, "abtem", SimpleNamespace(Potential=Potential,
        config=SimpleNamespace(set=lambda _: None)))
    return stored


@pytest.mark.parametrize("tiles", [(2, 2), (8, 10), (10, 12)])
def test_even_tiled_potential_avoids_redundant_shift_without_changing_any_array_bits(fake_periodic_builder, monkeypatch, tiles):
    pixels = math.lcm(*tiles)*4
    crystal = (np.array([2., 3., 4.]), np.zeros((1, 3)), np.array([14]), {})
    scene = SimpleNamespace(thickness_nm=1., centre_xy_nm=(0., 0.))
    original_shift = np.fft.fftshift
    monkeypatch.setattr(np.fft, "fftshift", lambda *_a, **_k: pytest.fail("Even-period identity allocated a full shift copy"))
    result, _, _ = scan._legacy_abtem_periodic_potential(crystal, scene, tuple(crystal[0][:2]*tiles), tiles, pixels, 2)
    oracle = original_shift(np.tile(fake_periodic_builder["primitive"], (1, tiles[1], tiles[0])), axes=(-2, -1))
    for index in range(len(result)):
        assert result[index].tobytes()==oracle[index].tobytes()


def test_nonzero_material_centre_still_uses_the_original_full_fourier_shift(fake_periodic_builder):
    tiles, pixels = (2, 2), 16
    crystal = (np.array([2., 3., 4.]), np.zeros((1, 3)), np.array([14]), {})
    scene = SimpleNamespace(thickness_nm=1., centre_xy_nm=(.01, -.02))
    fov = tuple(crystal[0][:2]*tiles)
    result, _, _ = scan._legacy_abtem_periodic_potential(crystal, scene, fov, tiles, pixels, 2)
    primitive = fake_periodic_builder["primitive"]
    tiled = np.fft.fftshift(np.tile(primitive, (1, 2, 2)), axes=(-2, -1))
    fx, fy = np.meshgrid(np.fft.fftfreq(pixels, fov[0]/pixels), np.fft.fftfreq(pixels, fov[1]/pixels))
    translated = np.fft.ifft2(np.fft.fft2(tiled)*np.exp(-2j*np.pi*(fx*.1+fy*(-.2)))).real
    expected = np.tile(translated[:, :pixels//2, :pixels//2], (1, 2, 2))
    for index in range(len(result)):
        assert result[index].tobytes()==expected[index].tobytes()


def test_bloch_periodicity_proof_checks_tiles_without_allocating_another_supercell(monkeypatch):
    packet = ScanPacket(np.ones((1, 1), complex), np.array([.25]), np.array([0]), np.array([0]), .02)
    potential = np.zeros((1, 8, 8))
    monkeypatch.setattr(np, "tile", lambda *_a, **_k: pytest.fail("Periodicity proof allocated another full tiled potential"))
    coefficients, record = periodic_coefficients(NumpyDevice(), packet, potential, np.array([1.]),
        (8., 8.), (2, 2), {"all": (0., 1000.)}, interaction=.1, bandwidth_fraction=2/3)
    assert coefficients.shape == (1, 1, 1)
    assert coefficients.item() == pytest.approx(.25, abs=2e-16)
    potential[0, 7, 7] = 1e-6
    with pytest.raises(ValueError, match="exact slice-by-slice periodicity"):
        periodic_coefficients(NumpyDevice(), packet, potential, np.array([1.]), (8., 8.), (2, 2),
            {"all": (0., 1000.)}, interaction=.1, bandwidth_fraction=2/3)
