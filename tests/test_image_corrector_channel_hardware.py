"""Existing optical channels own one explicit, non-OEM material insert each."""
from copy import deepcopy

import numpy as np
import pytest

from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.cad_multipoles import deflector_insert_solids
from temsim.corrector_model_3d import winding_axial_extent, winding_dimensions
from temsim.lens_pose import effective_part_transform_mm, mechanical_part_transform_mm
from temsim.module_manifest import read_document, validate_document
from temsim.optics.column import default_state
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.physics.instrument_magnetic import column_dipole_fields


VARIANTS = (("C3_ImageCorrector.toml", "C3 + Image Corrector"),
            ("C3_ProbeCorrector_ImageCorrector.toml", "C3 + Probe Corrector + Image Corrector"))
LABELS = ("dp11", "dp12", "dph1", "dp21", "dp22", "dph2", "ish", "dsh")
FIELD_LENGTHS = (6., 15., 30., 14., 14., 30., 14., 8.)
COPPER_LENGTHS = (4., 15., 33., 3., 3., 33., 14., 8.)


@pytest.mark.parametrize("filename,column", VARIANTS)
def test_each_channel_has_one_host_and_keeps_exact_nonzero_finite_field(filename, column):
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column" / filename)
    validate_document(doc)
    rows = {r["key"]: r for r in doc["parts"]}
    state = default_state()
    AssemblyCatalog().apply(state, AssemblySelection("FEG", column, "Energy Filter"))
    components = {c.key: c for c in state.corrector_elements}
    for label, field_length, copper_length in zip(LABELS, FIELD_LENGTHS, COPPER_LENGTHS, strict=True):
        key, host_key = f"image_{label}_deflector", f"image_{label}_coils"
        channel, host = rows[key], rows[host_key]
        assert channel["physical_host_key"] == host_key
        assert host["mechanical_only"] and host_key not in components
        assert channel["effective_thickness_mm"] == field_length
        assert winding_axial_extent(host)[1] - winding_axial_extent(host)[0] == copper_length
        solids = deflector_insert_solids(host, rows)
        assert len(solids) == 4
        assert all(s.parameters["linked_field_channels"] == [key] for s in solids)
        assert all(s.parameters["effective_field_length_mm"] == field_length for s in solids)
        components[key].kick_x_mrad, components[key].kick_y_mrad = .017, -.023
    channel_keys = {f"image_{label}_deflector" for label in LABELS}
    fields = [f for f in column_dipole_fields(state) if f.key.rsplit(":", 1)[0] in channel_keys]
    assert len(fields) == len(LABELS)
    for label, field_length in zip(LABELS, FIELD_LENGTHS, strict=True):
        key = f"image_{label}_deflector"
        field = next(f for f in fields if f.key == key + ":0")
        z = components[key].z_mm
        assert field.lower_m == (z - field_length / 2) * 1e-3
        assert field.upper_m == (z + field_length / 2) * 1e-3
        assert field.drive_keys == (key,)
        np.testing.assert_array_equal((field.event_dx_rad, field.event_dy_rad), (.017 * 1e-3, -.023 * 1e-3))
    for key in ("image_hpol_hexapole", "image_qpol_quadrupole", "image_dstg_quadrupole"):
        assert rows[key]["physical_host_status"] == "unverified"
        assert not rows[key].get("physical_host_key")


def test_copper_length_cannot_silently_change_field_or_close_saddle_window():
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column" / VARIANTS[0][0])
    row = next(r for r in doc["parts"] if r["key"] == "image_dp21_coils")
    before = deepcopy(row)
    assert winding_axial_extent(row) == (row["local_center_z_mm"] - 1.5, row["local_center_z_mm"] + 1.5)
    winding_dimensions(row)
    assert row == before and row["effective_thickness_mm"] == 14.
    for length in (0., -1., float("nan"), True, 14.01):
        with pytest.raises(ValueError, match="copper length"):
            winding_axial_extent({**row, "winding_axial_length_mm": length})
    with pytest.raises(ValueError, match="band width"):
        winding_dimensions({**row, "winding_axial_length_mm": .3})


def test_old_insert_without_copper_length_retains_entire_allocated_extent():
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column/C3_ProbeCorrector.toml")
    row = next(r for r in doc["parts"] if r["key"] == "probe_dph1_coils")
    assert "winding_axial_length_mm" not in row
    assert winding_axial_extent(row) == (row["local_start_z_mm"], row["local_end_z_mm"])


@pytest.mark.parametrize("filename,column", VARIANTS)
def test_dph_channel_inherits_optical_host_once_while_material_only_translates(filename, column):
    doc = read_document(INSTRUMENT_CONFIG_ROOT / "column" / filename)
    rows = {r["key"]: r for r in doc["parts"]}
    rows["image_hp1_hexapole"].update(offset_x_mm=.2, rotation_y_mrad=.6)
    rows["image_dph1_coils"].update(offset_y_mm=.1, rotation_x_mrad=-.4)
    channel = effective_part_transform_mm(rows["image_dph1_deflector"], rows)
    optical_host = effective_part_transform_mm(rows["image_dph1_coils"], rows)
    assert not np.array_equal(optical_host[0], np.eye(3))
    assert np.linalg.norm(optical_host[1]) > 0
    for actual, expected in zip(channel, optical_host, strict=True):
        np.testing.assert_array_equal(actual, expected)
    mechanical_rotation, mechanical_translation = mechanical_part_transform_mm(rows["image_dph1_coils"], rows)
    np.testing.assert_array_equal(mechanical_rotation, np.eye(3))
    np.testing.assert_allclose(mechanical_translation, (.2, .1, 0.), atol=1e-15)
