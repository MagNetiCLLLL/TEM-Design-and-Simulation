"""Installed fixed-turn electrical definitions across every catalog variant."""
from fractions import Fraction
from pathlib import Path

import pytest

from temsim import module_manifest
from temsim.cad_round_winding import round_winding_envelope, resolve_round_winding_geometry
from temsim.coil_reference_metadata import validate_coil_references
from temsim.coil_winding import OWNER_CURRENT_FIELD, INTERVAL_TURNS_FIELD

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT/'configs/instruments/column').glob('*.toml')) + [
    ROOT/'configs/instruments/project_and_recording_system/NoEnergyFilter.toml',
    ROOT/'configs/subassemblies/projector_stack.toml',
]


@pytest.mark.parametrize('path', FILES, ids=lambda p:p.stem)
def test_catalog_round_windings_have_estimated_fixed_turns_and_explicit_current(path):
    document = module_manifest.read_document(path)
    parts = {p['key']:p for p in document['parts']}
    validate_coil_references(document['parts'])
    found = 0
    for coil in document['parts']:
        owner = parts.get(coil.get('field_source_key'), {})
        if (coil.get('mechanical_profile') != 'magnetic_excitation_coil' or
                owner.get('mechanical_profile') not in {'magnetic_lens_assembly', 'integrated_magnetic_lens_channel'}):
            continue
        found += 1
        envelope = round_winding_envelope(coil, parts)
        width = (Fraction(str(envelope['outer_diameter_mm']))-Fraction(str(envelope['inner_diameter_mm'])))/2
        length = sum(Fraction(str(b))-Fraction(str(a)) for a,b in envelope['intervals_mm'])
        estimate = width*length/9
        expected = (estimate.numerator*2+estimate.denominator)//(estimate.denominator*2)
        assert coil['winding_turn_count'] == expected
        assert sum(coil[INTERVAL_TURNS_FIELD]) == expected
        assert len(coil[INTERVAL_TURNS_FIELD]) == len(envelope['intervals_mm'])
        assert coil['parameter_metadata']['winding_turn_count']['source_kind'] == 'estimated'
        assert owner[OWNER_CURRENT_FIELD] > 0
        assert owner['air_core_reference_ampere_turns'] == 3000.
        assert 'winding_wire_diameter_mm' not in coil
        assert resolve_round_winding_geometry(coil,parts) is None
        if owner['key'] == 'objective_lens':
            assert expected == 1160 and tuple(coil[INTERVAL_TURNS_FIELD]) == (580,580)
            note=coil['parameter_metadata']['winding_turn_count']['source_note']
            assert '586' in note and '20%' in note and 'not a new measurement' in note
    assert found > 0


def test_catalog_scope_includes_corrector_transfer_and_composed_recording_lenses():
    owners = []
    for path in FILES:
        parts = module_manifest.read_document(path)['parts']
        owners.extend(p['key'] for p in parts if p.get('mechanical_profile') in
                      {'magnetic_lens_assembly', 'integrated_magnetic_lens_channel'})
    assert len(owners) == 78 and len(set(owners)) == 20
    assert {'image_tl31_lens','image_tl32_lens','mini_condenser','projector_lens_2'}.issubset(owners)
