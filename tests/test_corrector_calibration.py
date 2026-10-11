import tomllib
from pathlib import Path

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim import module_manifest
from temsim.optics.column import default_state
from temsim.optics.magnetic_lens_aberration import spherical_aberration_mm
from temsim.physics.core import complex_transfer, propagate
from temsim.physics.first_order import trace_transverse_transfers


ROOT = Path(__file__).resolve().parents[1]
LATEST_IMAGE_SEQUENCE = (
    "image_hpol_hexapole",
    "image_qpol_quadrupole",
    "image_dp11_deflector",
    "image_tl11_lens",
    "image_dp12_deflector",
    "image_tl12_lens",
    "image_dph1_deflector",
    "image_hp1_hexapole",
    "image_tl21_lens",
    "image_dp21_deflector",
    "image_dp22_deflector",
    "image_tl22_lens",
    "image_dph2_deflector",
    "image_hp2_hexapole",
    "image_adapter_lens",
    "image_ish_deflector",
    "image_dsh_deflector",
    "image_dstg_quadrupole",
    "image_sad_plane",
)


def _combined_state():
    state = default_state()
    AssemblyCatalog().apply(
        state,
        AssemblySelection(
            "FEG",
            "C3 + Probe Corrector + Image Corrector",
            "Energy Filter",
        ),
    )
    state.step_mm = 0.1
    state.history_step_mm = 0.1
    return state


@pytest.mark.parametrize(
    "filename",
    ("C3_ImageCorrector.toml", "C3_ProbeCorrector_ImageCorrector.toml"),
)
def test_image_corrector_uses_separate_transfer_lenses_and_bound_dph(filename):
    path = ROOT / "configs" / "instruments" / "column" / filename
    parts = tomllib.loads(path.read_text(encoding="utf-8"))["parts"]
    image_parts = [
        part for part in sorted(parts, key=lambda item: item["order"])
        if str(part["key"]).startswith("image_")
        and not bool(part.get("mechanical_only", False))
    ]
    keys = tuple(part["key"] for part in image_parts)
    start = keys.index(LATEST_IMAGE_SEQUENCE[0])
    assert keys[start:start + len(LATEST_IMAGE_SEQUENCE)] == (
        LATEST_IMAGE_SEQUENCE
    )
    by_key = {part["key"]: part for part in parts}
    assert "image_ol_post_lens" not in by_key
    tl11 = by_key["image_tl11_lens"]
    tl12 = by_key["image_tl12_lens"]
    dp12 = by_key["image_dp12_deflector"]
    assert tl11["local_end_z_mm"] < dp12["local_start_z_mm"]
    assert dp12["local_end_z_mm"] < tl12["local_start_z_mm"]
    assert tl12["mechanical_outer_diameter_mm"] == 160.0
    assert tl12["length_mm"] == 46.0
    assert tl11["field_polarity"] == -1
    assert tl12["field_polarity"] == 1
    assert {
        part["key"] for part in parts
        if part.get("parent_key") == "image_tl12_lens"
    } == {
        f"image_tl12_lens_{suffix}"
        for suffix in ("upper_pole", "lower_pole", "housing", "yoke", "excitation_coil")
    }
    assert by_key["image_tl21_lens"]["local_end_z_mm"] < (
        by_key["image_dp21_deflector"]["local_center_z_mm"]
    ) < by_key["image_dp22_deflector"]["local_center_z_mm"]
    for index in (1, 2):
        channel = by_key[f"image_dph{index}_deflector"]
        host = by_key[f"image_hp{index}_hexapole"]
        assert channel["local_center_z_mm"] == host["local_center_z_mm"]
        assert channel["optical_reference_host_key"] == host["key"]
        assert channel["optical_reference_offset_mm"] == 0.0
        assert channel["physical_host_status"] == "unverified"


def test_per_lens_image_envelope_requires_provenance_and_contained_material():
    path = ROOT / "configs/instruments/column/C3_ImageCorrector.toml"
    document = module_manifest.read_document(path)
    module_manifest.validate_document(document)
    by_key = {part["key"]: part for part in document["parts"]}
    lens = by_key["image_tl12_lens"]
    provenance = lens.pop("mechanical_geometry_source")
    with pytest.raises(ValueError, match="reconstruction provenance"):
        module_manifest.validate_document(document)
    lens["mechanical_geometry_source"] = provenance
    by_key["image_tl12_lens_housing"]["mechanical_outer_diameter_mm"] = 181.0
    with pytest.raises(ValueError, match="envelope|outer diameter"):
        module_manifest.validate_document(document)


def test_both_main_hexapole_relays_are_unit_conjugates():
    state = _combined_state()
    by_key = {
        component.key: component
        for component in (*state.lenses, *state.corrector_elements)
    }
    for upstream, downstream in (
        ("probe_hp2_hexapole", "probe_hp1_hexapole"),
        ("image_hp1_hexapole", "image_hp2_hexapole"),
    ):
        matrix = complex_transfer(
            state, by_key[upstream].z_mm, by_key[downstream].z_mm
        )
        assert abs(matrix[0, 1]) < 2.0e-4
        assert abs(abs(matrix[0, 0]) - 1.0) < 5.0e-3

    # CETCOR images the specimen, not the former arbitrary objective-image
    # label. Keep the whole X/Y coupling, including magnetic image rotation.
    system = state.image_corrector_system
    targets = (
        .5 * (system.tl11_lens.z_mm + system.tl12_lens.z_mm),
        .5 * (system.tl21_lens.z_mm + system.tl22_lens.z_mm),
        system.sad_plane.z_mm,
    )
    for transfer in trace_transverse_transfers(state, state.sample.z_mm, targets).values():
        assert np.linalg.norm(transfer.matrix[:2, 2:], 2) < 1.e-7
    hp = system.main_hexapole_transfer_matrix()
    assert hp.shape == (4, 4)
    assert np.linalg.norm(hp[:2, 2:], 2) < 1.e-7
    np.testing.assert_allclose(np.linalg.svd(hp[:2, :2], compute_uv=False), 1., atol=1.e-5)
    # Position conjugacy alone is not a certificate of the full -I transform.


def _trace_aberration(state, z0, z1, x, tx, y, ty, *, cs, hexapole):
    return propagate(
        state, z0, z1, x, tx, y, ty,
        include_spherical_aberration=cs,
        include_hexapole=hexapole,
    )


def _errors(result, baseline):
    return (
        result[1][-1] - baseline[1][-1]
        + 1j * (result[3][-1] - baseline[3][-1])
    )


def _assert_negative_corrector_contribution(
    state, z0, z1, x, tx, y, ty, main_hexapoles
):
    baseline = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=False, hexapole=False
    )
    positive = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=True, hexapole=False
    )
    saved_cs = [getattr(lens, "cs_mm", None) for lens in state.lenses]
    try:
        for lens in state.lenses:
            lens.cs_mm = 0.0
        correction = _trace_aberration(
            state, z0, z1, x, tx, y, ty, cs=True, hexapole=True
        )
    finally:
        for lens, value in zip(state.lenses, saved_cs):
            lens.cs_mm = value
    net = _trace_aberration(
        state, z0, z1, x, tx, y, ty, cs=True, hexapole=True
    )
    positive_error = _errors(positive, baseline)
    correction_error = _errors(correction, baseline)
    net_error = _errors(net, baseline)
    assert all(hexapole.strength_m3 > 0.0 for hexapole in main_hexapoles)
    assert np.vdot(positive_error, correction_error).real < 0.0
    assert np.sqrt(np.mean(abs(net_error) ** 2)) < (
        0.20 * np.sqrt(np.mean(abs(positive_error) ** 2))
    )


def test_positive_round_lenses_and_calibrated_probe_corrector_cs_contribution():
    state = _combined_state()
    by_key = {
        component.key: component
        for component in (*state.lenses, *state.corrector_elements)
    }
    assert all(
        spherical_aberration_mm(lens, state.beam_voltage_kv) > 0.0
        for lens in state.lenses if lens.enabled
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 13)[:-1]
    alpha = 1.0e-3
    zero = np.zeros(phi.size)
    # Image-side Cs cancellation was formerly asserted without an angular
    # objective-Cs model. Its finite-HP residual is measured separately by
    # scripts/validate_image_corrector.py and is explicitly unqualified.

    radius = 1.0e-4
    probe_hp2 = by_key["probe_hp2_hexapole"]
    probe_start = (
        probe_hp2.z_mm
        - 5.0 * probe_hp2.effective_length_mm / 2.355
    )
    _assert_negative_corrector_contribution(
        state,
        probe_start,
        state.sample.z_mm,
        radius * np.cos(phi),
        zero,
        radius * np.sin(phi),
        zero,
        (probe_hp2, by_key["probe_hp1_hexapole"]),
    )
