"""Author named multipole windings, then calibrate current at fixed N/geometry.

The two explicit phases preserve raw baseline bytes and optical controls.
No field-strength condition enters the initial turn estimator.
"""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import tomllib
from unittest.mock import patch

from temsim import module_manifest
from temsim.assembly_catalog import AssemblyCatalog, AssemblySelection
from temsim.column.state_layout import apply_physical_layout_to_state
from temsim.multipole_winding import (
    AREA_FIELD, BASIS_FIELD, INSULATION_FIELD, NOTE_FIELD, SOURCE_NOTE,
    STATUS_FIELD, STATUS_VALUE, TURN_FIELD, area_field, turn_field,
    estimate_multipole_winding, is_primary_multipole,
    multipole_winding_definition, supported_multipole_winding, uses_legacy_probe_winding,
)
from temsim.operating_modes import apply_operating_mode_pair
from temsim.optics.column import default_state
from temsim.physics.multipole_coil_field import (
    calibrate_multipole_current_matrix, recommended_basis_angle_deg,
)
from temsim.simulation_modes import switch_mode


ROOT = Path(__file__).resolve().parents[3]
VALIDATION = Path(__file__).resolve().parent / 'validation'


def forbidden(*args, **kwargs):
    raise AssertionError('Winding/current initialization must not emit or transport particles')


def originals():
    return json.loads((VALIDATION/'before/inputs.json').read_text(encoding='utf-8'))


def state_for(record, catalog, columns):
    default = catalog.default_selection()
    selection = AssemblySelection(default.gun, columns[Path(record['path']).name], default.recording)
    state = default_state()
    catalog.apply(state, selection)
    switch_mode(state, 'unified')
    if selection.column == default.column:
        apply_operating_mode_pair(state, 'nano_probe', 'diffraction',
            column_name=selection.column, recording_name=selection.recording)
    apply_physical_layout_to_state(state)
    return selection, state


def write_source(name, text):
    raw = (VALIDATION/'before'/name).read_bytes()
    if b'\r\n' in raw:
        text = text.replace('\r\n','\n').replace('\n','\r\n')
    (ROOT/name).write_bytes(text.encode('utf-8'))


def seed_geometry():
    destination = VALIDATION/'turn-initialization.json'
    assert not destination.exists(), 'Preserve the original geometry initialization receipt'
    catalog = AssemblyCatalog()
    columns = {Path(option.file).name:option.name for option in catalog.columns}
    staged, records = {}, []
    for original in originals():
        path = ROOT/original['path']
        assert sha256(path.read_bytes()).hexdigest() == original['sha256']
        selection, state = state_for(original,catalog,columns)
        optical_before = json.dumps(state.to_dict(),sort_keys=True)
        components = {row.key:row for row in state.corrector_elements}
        document = tomllib.loads(path.read_text(encoding='utf-8'))
        updates = {}
        for part in document.get('parts',[]):
            if not supported_multipole_winding(part):
                continue
            component = components[part['key']]
            estimated = estimate_multipole_winding(part,.02,.60,.005)
            count = estimated['turn_count_per_coil']
            count_method = estimated['method']
            if is_primary_multipole(part) and not uses_legacy_probe_winding(part):
                # Use the same declared pack allocation reference as Probe HP,
                # rather than fitting N to a requested field or current.
                count = max(1,math.floor(estimated['limiting_cross_section_mm2']/9.+.5))
                count_method = 'image_primary_HP_limiting_pack_area_over_existing_9_mm2_reference'
            basis = recommended_basis_angle_deg(component) if is_primary_multipole(part) else 0.
            old_count = part.get(turn_field(part))
            if uses_legacy_probe_winding(part):
                assert count == old_count, 'Existing physical Probe HP counts are fixed'
            note = (f'Explicit 2026-11-15 ideal-vacuum multipole engineering initialization. '
                f'Per-coil N={count}; count method {count_method}. Metal area 0.02 mm2, '
                'radial enamel assumption 0.005 mm. Auxiliary nominal metal fill 0.60; '
                'fixed original Probe HP N and Image HP nominal 9 mm2 pack allocation '
                'are separate count references. Large residual insulation does not '
                'establish a manufacturing design or current rating. '
                f'Saved mechanical winding angular datum {basis:.16g} degrees; '
                f'unchanged native optical orientation {getattr(component,"orientation_rad",0.):.16g} rad. '
                'The primary six-coil datum defines its previously uncalibrated CAD/electrical '
                'phase; it is shared by winding and passive cartridge CAD and does not edit '
                'an optical drive, installation-error input or clipping boundary. '+SOURCE_NOTE)
            fields = {turn_field(part):count,area_field(part):.02,INSULATION_FIELD:.005,
                BASIS_FIELD:basis,STATUS_FIELD:STATUS_VALUE,NOTE_FIELD:note}
            candidate = {**part,**fields}
            definition = multipole_winding_definition(candidate)
            assert definition['detailed_cad'] and definition['turn_count_per_coil']==count
            metadata = deepcopy(part.get('parameter_metadata',{}))
            for name,value in fields.items():
                if name in part and part[name] == value:
                    continue
                updates['parts',part['key'],name] = value
                metadata[name] = dict(source_kind='estimated',source_note=note)
            updates['parts',part['key'],'parameter_metadata'] = metadata
            records.append(dict(path=original['path'],selection=asdict(selection),key=part['key'],
                multipole_order=definition['order'],electrical_channels=definition['electrical_channel_count'],
                turn_count_per_coil=count,previous_turn_count=old_count,
                total_turn_count=definition['total_turn_count'],coil_count=definition['coil_count'],
                wire_cross_section_mm2=.02,wire_diameter_mm=definition['wire_diameter_mm'],
                insulation_thickness_mm=.005,basis_angle_deg=basis,count_method=count_method,
                limiting_cross_section_mm2=estimated['limiting_cross_section_mm2'],
                actual_metal_fill_fraction=count*.02/estimated['limiting_cross_section_mm2'],
                previous_optical_orientation_rad=getattr(component,'orientation_rad',0.),
                optical_orientation_unchanged=True,fields=fields))
        assert json.dumps(state.to_dict(),sort_keys=True)==optical_before
        staged[original['path']] = module_manifest.stage_manifest_text(
            path.read_text(encoding='utf-8'),updates,source_path=path)
        print('Prepared winding geometry:',path.name,flush=True)
    assert len(records)==40
    for row in originals():
        assert sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256']
    for name,text in staged.items():
        write_source(name,text)
    destination.write_text(json.dumps(dict(schema='multipole-engineering-winding-initialization-v1',
        source_files=len(staged),hosts=len(records),records=records),indent=2)+'\n',encoding='utf-8')
    print('Initialized 40 multipole winding definitions with no optical retuning')


def calibrate_currents(*, final=False):
    from temsim.multipole_current import MATRIX_FIELD, NOTE_FIELD as CURRENT_NOTE
    destination = VALIDATION/('current-calibration-final.json' if final else 'current-calibration.json')
    assert not destination.exists(), 'Preserve the original current calibration receipt'
    catalog = AssemblyCatalog()
    columns = {Path(option.file).name:option.name for option in catalog.columns}
    before_hashes = {row['path']:sha256((ROOT/row['path']).read_bytes()).hexdigest() for row in originals()}
    records, staged = [], {}
    for original in originals():
        path = ROOT/original['path']
        selection,state = state_for(original,catalog,columns)
        optical_before = json.dumps(state.to_dict(),sort_keys=True)
        document = tomllib.loads(path.read_text(encoding='utf-8'))
        updates = {}
        for part in document.get('parts',[]):
            if not supported_multipole_winding(part):
                continue
            if final:
                assert MATRIX_FIELD in part, 'Final reference requires the preserved initial fit'
            else:
                assert MATRIX_FIELD not in part, 'Do not overwrite an existing current calibration'
            result = calibrate_multipole_current_matrix(state,part['key'])
            metadata = deepcopy(part.get('parameter_metadata',{}))
            for name,value in result['fields'].items():
                updates['parts',part['key'],name] = value
                metadata[name] = dict(source_kind='estimated',source_note=result['fields'][CURRENT_NOTE])
            updates['parts',part['key'],'parameter_metadata'] = metadata
            records.append(dict(path=original['path'],selection=asdict(selection),**result))
            print('Calibrated fixed-N current:',path.name,part['key'],flush=True)
        assert json.dumps(state.to_dict(),sort_keys=True)==optical_before
        staged[original['path']] = module_manifest.stage_manifest_text(
            path.read_text(encoding='utf-8'),updates,source_path=path)
    assert len(records)==40
    assert all(sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in before_hashes.items())
    for name,text in staged.items():
        write_source(name,text)
    destination.write_text(json.dumps(dict(schema='multipole-native-peak-current-calibration-v1',
        source_files=len(staged),hosts=len(records),records=records),indent=2)+'\n',encoding='utf-8')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--seed-geometry',action='store_true')
    parser.add_argument('--calibrate-currents',action='store_true')
    parser.add_argument('--finalize-calibration',action='store_true')
    args=parser.parse_args()
    if args.seed_geometry:
        seed_geometry()
    if args.calibrate_currents:
        calibrate_currents()
    if args.finalize_calibration:
        calibrate_currents(final=True)


if __name__=='__main__':
    with patch('temsim.optics.electron_gun.emitter.ColdFieldEmitter.emit',forbidden), \
         patch('temsim.optics.electron_gun.source.trace_source_to_exit',forbidden):
        main()
