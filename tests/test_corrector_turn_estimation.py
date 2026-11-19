"""Engineering fixed-N initialization without guessing current or metal size."""
from copy import deepcopy
import math
from pathlib import Path
import tomllib

import pytest

from temsim.cad_multipoles import multipole_solids
from temsim.cad_probe_winding import (
    AREA_FIELD, NOTE_FIELD, STATUS_FIELD, TURN_FIELD,
    estimate_probe_winding_turns, probe_winding_dimensions,
)
from temsim.module_manifest import read_document
from temsim.paths import INSTRUMENT_CONFIG_ROOT
from temsim.probe_hexapole_geometry import probe_hexapole_dimensions


CONFIGS=('C3_ProbeCorrector.toml','C3_ProbeCorrector_ImageCorrector.toml',
         'C3_ProbeCorrector_BCOR.toml')
HOSTS=('probe_hp1_hexapole','probe_hp2_hexapole')


def _part(name=CONFIGS[0],key=HOSTS[0]):
    document=read_document(INSTRUMENT_CONFIG_ROOT/'column'/name)
    return next(part for part in document['parts'] if part.get('key')==key)


@pytest.mark.parametrize('name',CONFIGS)
@pytest.mark.parametrize('key',HOSTS)
def test_installed_fixed_initial_count_uses_limiting_radial_pack_section(name,key):
    part=_part(name,key);before=deepcopy(part)
    d=probe_hexapole_dimensions(part)
    estimate=estimate_probe_winding_turns(part,9.0)
    band=min((d['coil_width_mm']-d['coil_window_width_mm'])/2,
             (d['coil_height_mm']-d['coil_window_height_mm'])/2)
    area=d['coil_radial_length_mm']*band
    assert estimate['cross_section_mm2']==area
    assert estimate['minimum_band_mm']==band
    assert estimate['turn_count_per_coil']==part[TURN_FIELD]==2
    assert estimate['total_turn_count']==12
    assert estimate['actual_allocated_cross_section_mm2_per_turn']==area/2
    assert estimate['nominal_allocated_cross_section_mm2_per_turn']==9.0
    assert estimate['wire_cross_section_mm2'] is None
    assert estimate['wire_diameter_mm'] is None and estimate['current_a'] is None
    assert not estimate['field_or_current_calibration']
    assert not estimate['automatic_geometry_binding']
    assert part[STATUS_FIELD]=='engineering_reconstruction_not_oem'
    assert part['parameter_metadata'][TURN_FIELD]['source_kind']=='estimated'
    assert 'not measured/OEM' in part[NOTE_FIELD]
    assert AREA_FIELD not in part and probe_winding_dimensions(part) is None
    assert part==before


@pytest.mark.parametrize('target,expected',[(.2,1),(1.,1),(1.5,2),(2.49,2),(2.5,3),(3.,3)])
def test_rounding_and_minimum_are_explicit_authoring_choices(target,expected):
    part=_part()
    area=estimate_probe_winding_turns(part,9.0)['cross_section_mm2']
    estimate=estimate_probe_winding_turns(part,area/target)
    assert estimate['turn_count_per_coil']==expected


@pytest.mark.parametrize('nominal',[True,False,0.,-1.,math.inf,math.nan,'9',None])
def test_invalid_nominal_allocations_fail(nominal):
    with pytest.raises(ValueError,match='finite and positive'):
        estimate_probe_winding_turns(_part(),nominal)


def test_column_annulus_and_nonmaterial_channels_cannot_define_radial_turns():
    with pytest.raises(ValueError,match='material HP1/HP2'):
        estimate_probe_winding_turns(_part(key='probe_tl22_lens'),9.0)
    part=_part();part['layout_role']='control_channel'
    with pytest.raises(ValueError,match='material HP1/HP2'):
        estimate_probe_winding_turns(part,9.0)


def test_saved_turn_count_does_not_follow_later_geometry_edits():
    part=_part();part['mechanical_hexapole_coil_radial_length_mm']=5.0
    assert estimate_probe_winding_turns(part,9.0)['turn_count_per_coil']==1
    assert part[TURN_FIELD]==2
    assert probe_winding_dimensions(part) is None


@pytest.mark.parametrize('name',CONFIGS)
def test_initialization_changes_only_explicit_new_hp_metadata(name):
    current=tomllib.loads((INSTRUMENT_CONFIG_ROOT/'column'/name).read_text(encoding='utf-8'))
    original_path=Path(__file__).resolve().parents[1]/'docs/development/corrector_windings_20261115/validation/before'/name
    original=tomllib.loads(original_path.read_text(encoding='utf-8'))
    stripped=deepcopy(current)
    for part in stripped['parts']:
        if part.get('key') in HOSTS:
            for field in (TURN_FIELD,STATUS_FIELD,NOTE_FIELD):part.pop(field)
            part['parameter_metadata'].pop(TURN_FIELD)
    assert stripped==original


@pytest.mark.parametrize('name',CONFIGS)
@pytest.mark.parametrize('key',HOSTS)
def test_unknown_metal_size_keeps_all_previous_cad_regions_exact(name,key):
    document=read_document(INSTRUMENT_CONFIG_ROOT/'column'/name)
    current={part['key']:part for part in document['parts']}
    original_path=Path(__file__).resolve().parents[1]/'docs/development/corrector_windings_20261115/validation/before'/name
    original_document=tomllib.loads(original_path.read_text(encoding='utf-8'))
    original={part['key']:part for part in original_document['parts']}
    assert multipole_solids(current[key],current)==multipole_solids(original[key],original)


def test_undefined_wire_size_does_not_apply_a_cad_allocation_budget():
    part=_part();part[TURN_FIELD]=100000
    assert probe_winding_dimensions(part) is None
    part[AREA_FIELD]=.0001
    with pytest.raises(ValueError,match='No turns were omitted'):
        probe_winding_dimensions(part)
