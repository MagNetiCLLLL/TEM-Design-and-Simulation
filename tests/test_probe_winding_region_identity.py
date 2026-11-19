"""Detailed-wire material classification must not rebuild Boolean solids."""
from copy import deepcopy

import pytest

from temsim.cad_catalog import is_reconstruction_region
from temsim.cad_probe_winding import TURN_FIELD, AREA_FIELD, STATUS_FIELD, NOTE_FIELD
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT


@pytest.fixture
def part():
    rows=read_document(INSTRUMENT_CONFIG_ROOT/'column/C3_ProbeCorrector.toml')['parts']
    value=deepcopy(next(p for p in rows if p['key']=='probe_hp2_hexapole'))
    value.update({TURN_FIELD:2,AREA_FIELD:.05,STATUS_FIELD:'engineering_reconstruction_not_oem',
                  NOTE_FIELD:'Explicit illustrative CAD test, not a default winding specification'})
    return value


@pytest.mark.parametrize(('region','expected'),[
    ('winding_turn_coil_1_000001',True),('winding_turn_coil_6_000002',True),
    ('winding_insulation_coil_3',True),('winding_turn_coil_1_000003',False),
    ('winding_turn_coil_1_000000',False),('winding_turn_coil_7_000001',False),
    ('winding_insulation_coil_0',False),
])
def test_wire_identity_uses_pure_layout_and_valid_regional_indices(part,monkeypatch,region,expected):
    def forbidden(*args,**kwargs):raise AssertionError('Region classification must not build multipole CAD')
    monkeypatch.setattr('temsim.cad_multipoles.multipole_solids',forbidden)
    assert is_reconstruction_region(part,region) is expected


def test_replaced_or_incomplete_recipe_does_not_claim_wire_material(part):
    part.pop(AREA_FIELD)
    assert not is_reconstruction_region(part,'winding_turn_coil_1_000001')
    part[AREA_FIELD]=.05
    part['model_3d']={'base':{'kind':'box'}}
    assert not is_reconstruction_region(part,'winding_turn_coil_1_000001')
