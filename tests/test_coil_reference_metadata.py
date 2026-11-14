"""Air-core references are explicit current inputs, never inferred windings."""
from copy import deepcopy
from pathlib import Path
import tomllib

import pytest

from temsim.coil_reference_metadata import current_intervals_mm, validate_coil_references
from temsim.module_manifest import read_document, validate_document


ROOT = Path(__file__).resolve().parents[1]
PATHS = (*sorted((ROOT/"configs/instruments/column").glob("*.toml")),
         ROOT/"configs/instruments/project_and_recording_system/NoEnergyFilter.toml")


def reference_parts():
    return [dict(key="round", mechanical_profile="magnetic_lens_assembly", local_start_z_mm=0.,
        air_core_reference_ampere_turns=3000.,
        air_core_reference_status="engineering_reference_not_oem_not_particle_qualified",
        air_core_reference_source="Explicit synthetic engineering input; not an OEM calibration.",
        air_core_reference_source_urls=["https://web.mit.edu/6.013_book/www/chapter8/8.2.html"],
        gaussian_reference_coil_length_mm=10.),
        dict(key="coil", mechanical_profile="magnetic_excitation_coil", field_source_key="round",
             mechanical_inner_diameter_mm=8., mechanical_outer_diameter_mm=12.,
             local_start_z_mm=0., local_end_z_mm=10., coil_ampere_turn_fraction=1.)]


@pytest.mark.parametrize("path", PATHS, ids=lambda path: path.stem)
def test_bundled_round_owners_have_explicit_current_and_current_bearing_geometry(path):
    document = validate_document(read_document(path))
    owners = [part for part in document["parts"] if part.get("mechanical_profile") == "magnetic_lens_assembly"]
    for owner in owners:
        assert owner["air_core_reference_ampere_turns"] == 3000.
        coils = [part for part in document["parts"] if part.get("mechanical_profile") == "magnetic_excitation_coil"
                 and part.get("field_source_key") == owner["key"]]
        assert coils
        actual_length = sum(end-start for coil in coils for start, end in current_intervals_mm(coil, owner))
        assert owner["gaussian_reference_coil_length_mm"] == pytest.approx(actual_length)
        assert sum(coil["coil_ampere_turn_fraction"] for coil in coils) == 1.


@pytest.mark.parametrize("value", [0., -1., float("nan"), float("inf"), True, "3000"])
def test_reference_rejects_invalid_ampere_turns(value):
    parts = reference_parts()
    parts[0]["air_core_reference_ampere_turns"] = value
    with pytest.raises(ValueError, match="finite positive"):
        validate_coil_references(parts)


def test_partial_reference_and_missing_coil_are_not_synthesized():
    parts = reference_parts()
    del parts[0]["air_core_reference_source"]
    with pytest.raises(ValueError, match="incomplete"):
        validate_coil_references(parts)
    with pytest.raises(ValueError, match="explicit current-bearing"):
        validate_coil_references(reference_parts()[:1])


def test_multiple_coils_partition_one_total_reference_current():
    parts = reference_parts()
    second = dict(parts[1], key="second")
    parts[1]["coil_ampere_turn_fraction"] = second["coil_ampere_turn_fraction"] = .5
    parts.append(second)
    validate_coil_references(parts)
    parts[-1]["coil_ampere_turn_fraction"] = .6
    with pytest.raises(ValueError, match="sum to one"):
        validate_coil_references(parts)


def test_multipole_is_not_silently_reinterpreted_as_a_round_coil():
    parts = reference_parts()
    parts[0]["mechanical_profile"] = "hexapole_corrector"
    with pytest.raises(ValueError, match="multipole or deflector"):
        validate_coil_references(parts)


@pytest.mark.parametrize("intervals", [[], [(0., 1.), (.5, 2.)], [(0., 0.)], [("0", 1.)]])
def test_empty_or_invalid_current_support_is_rejected(intervals):
    parts = reference_parts()
    parts[1]["material_intervals_mm"] = intervals
    with pytest.raises(ValueError, match="current-bearing interval"):
        validate_coil_references(parts)


def test_metadata_free_legacy_documents_remain_valid_without_invented_current():
    parts = [dict(key="legacy", mechanical_profile="magnetic_lens_assembly")]
    before = deepcopy(parts)
    validate_coil_references(parts)
    assert parts == before


def test_objective_reference_uses_split_physical_layers_instead_of_display_envelope():
    document = tomllib.loads((ROOT/"configs/instruments/column/C2.toml").read_text(encoding="utf-8"))
    by_key = {part["key"]: part for part in document["parts"]}
    owner, coil = by_key["objective_lens"], by_key["objective_lens_excitation_coil"]
    intervals = current_intervals_mm(coil, owner)
    assert len(intervals) == 2
    assert sum(end-start for start, end in intervals) == 290.
    assert owner["gaussian_reference_coil_length_mm"] == 290.
    assert coil["length_mm"] == 162.
